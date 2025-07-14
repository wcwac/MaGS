import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general_utils import get_expon_lr_func


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        # Store p_fn and freq directly in lambda scope
        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        # Ensure inputs are float for trig functions
        inputs_float = inputs.float()
        return torch.cat([fn(inputs_float) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    if multires <= 0:  # Handle multires=0 or less case gracefully
        return nn.Identity(), input_dims

    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)

    return embed, embedder_obj.out_dim


# --- Strength Predictor ---
class StrengthPredictor(nn.Module):
    # Takes the *raw* aggregated feature dimension as input
    def __init__(self, input_feature_dim, hidden_dim=64, num_outputs=1):
        super().__init__()
        print(
            f"[StrengthPredictor] Initializing with input_dim={input_feature_dim}, hidden_dim={hidden_dim}, num_outputs={num_outputs}"
        )
        self.net = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_outputs),
        )
        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, aggregated_input_features):
        needs_squeeze = False
        # Handle potential lack of batch dimension if input is [feat_dim]
        if aggregated_input_features.ndim == 1:
            aggregated_input_features = aggregated_input_features.unsqueeze(0)
            needs_squeeze = True

        raw_scale = self.net(aggregated_input_features)

        if needs_squeeze:
            raw_scale = raw_scale.squeeze(0)

        return raw_scale


# --- Optimized UvwDeformNetwork ---
# (No changes needed in this class)
class UvwDeformNetwork(nn.Module):
    def __init__(self, config, vertices_pos_emb_dim_per_vertex, combined_pose_dim):
        super(UvwDeformNetwork, self).__init__()
        self.config = config
        self.D = config.uvw_deform_D
        self.W = config.uvw_deform_W
        self.skips = [self.D // 2]

        self.input_ch_uvw_vert_feat = 3 + vertices_pos_emb_dim_per_vertex * 3
        self.combined_pose_dim = combined_pose_dim

        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(self.input_ch_uvw_vert_feat, self.W))

        for i in range(self.D - 1):
            input_dim = self.W
            if i == 0:
                input_dim += self.combined_pose_dim
            if i + 1 in self.skips:
                input_dim += self.input_ch_uvw_vert_feat

            self.linear.append(nn.Linear(input_dim, self.W))

        self.uvw_warp = nn.Linear(self.W, 3)

        for layer in self.linear:
            nn.init.kaiming_uniform_(
                layer.weight, a=np.sqrt(5), mode="fan_in", nonlinearity="relu"
            )
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.normal_(self.uvw_warp.weight, mean=0, std=1e-5)
        nn.init.zeros_(self.uvw_warp.bias)

    def forward(self, uvw, vertices_pos_emb_flat, pose_embeddings_expanded):
        h = torch.cat([uvw, vertices_pos_emb_flat], dim=-1)
        initial_input_for_skip = h

        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)

            if i == 0:
                h = torch.cat([h, pose_embeddings_expanded], dim=-1)

            if i + 1 in self.skips:
                h = torch.cat([initial_input_for_skip, h], -1)

        return {"d_uvw": self.uvw_warp(h)}


# --- Optimized MeshDeformNetwork ---
# (No changes needed in this class)
class MeshDeformNetwork(nn.Module):
    def __init__(self, config, vertices_pos_emb_dim, combined_pose_dim):
        super(MeshDeformNetwork, self).__init__()
        self.config = config
        self.D = config.mesh_deform_D
        self.W = config.mesh_deform_W
        self.skips = [self.D // 2]

        self.input_ch_vert = vertices_pos_emb_dim  # Embedded vertex *positions*
        self.combined_pose_dim = combined_pose_dim

        self.pred_opacity = config.get("pred_opacity", False)
        self.pred_color = config.get("pred_color", False)
        self.max_d_scale = config.get("max_d_scale", 0.0)

        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(self.input_ch_vert, self.W))

        for i in range(self.D - 1):
            input_dim = self.W
            if i == 0:
                input_dim += self.combined_pose_dim
            if i + 1 in self.skips:
                input_dim += self.input_ch_vert
            self.linear.append(nn.Linear(input_dim, self.W))

        self.gaussian_warp = nn.Linear(self.W, 3)
        self.gaussian_scaling = nn.Linear(self.W, 3)
        self.gaussian_rotation = nn.Linear(self.W, 4)

        if self.pred_opacity:
            self.gaussian_opacity = nn.Linear(self.W, 1)
        if self.pred_color:
            color_W = config.get("color_W", self.W // 2)
            color_D = config.get("color_D", 2)
            color_layers = []
            color_in_dim = self.W
            for _ in range(color_D):
                color_layers.extend([nn.Linear(color_in_dim, color_W), nn.ReLU()])
                color_in_dim = color_W
            color_layers.append(nn.Linear(color_in_dim, 3))
            self.gaussian_color = nn.Sequential(*color_layers)

        for layer in self.linear:
            nn.init.kaiming_uniform_(
                layer.weight, a=np.sqrt(5), mode="fan_in", nonlinearity="relu"
            )
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        nn.init.normal_(self.gaussian_warp.weight, mean=0, std=1e-5)
        nn.init.zeros_(self.gaussian_warp.bias)
        nn.init.normal_(self.gaussian_scaling.weight, mean=0, std=1e-8)
        nn.init.zeros_(self.gaussian_scaling.bias)
        nn.init.normal_(self.gaussian_rotation.weight, mean=0, std=1e-5)
        nn.init.zeros_(self.gaussian_rotation.bias)

        if self.pred_opacity:
            nn.init.normal_(self.gaussian_opacity.weight, mean=0, std=1e-5)
            nn.init.zeros_(self.gaussian_opacity.bias)
        if self.pred_color:
            for layer in self.gaussian_color:
                if isinstance(layer, nn.Linear):
                    if layer is self.gaussian_color[-1]:
                        nn.init.normal_(layer.weight, mean=0, std=1e-5)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                    else:
                        nn.init.kaiming_uniform_(
                            layer.weight,
                            a=np.sqrt(5),
                            mode="fan_in",
                            nonlinearity="relu",
                        )
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)

    def forward(self, vertices_pos_emb, pose_embeddings_expanded):
        h = vertices_pos_emb
        initial_vert_emb_for_skip = h

        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)

            if i == 0:
                h = torch.cat([h, pose_embeddings_expanded], dim=-1)

            if i + 1 in self.skips:
                h = torch.cat([initial_vert_emb_for_skip, h], -1)

        d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        if self.max_d_scale > 0:
            log_max_scale = torch.log(
                torch.tensor(self.max_d_scale, device=h.device) + 1e-9
            )
            scaling = torch.tanh(scaling) * log_max_scale

        return_dict = {
            "d_xyz": d_xyz,
            "d_rotation": rotation,
            "d_scaling": scaling,
        }
        if self.pred_opacity:
            return_dict["d_opacity"] = self.gaussian_opacity(h)
        else:
            return_dict["d_opacity"] = None

        if self.pred_color:
            color_input = h
            return_dict["d_color"] = self.gaussian_color(color_input)
        else:
            return_dict["d_color"] = None

        return return_dict


# --- Optimized DeformModel with Adaptive Strength Control using RAW Vertex Features ---
class DeformModel(nn.Module):
    def __init__(self, config):
        super(DeformModel, self).__init__()
        self.config = config

        # 1. Embedder for vertex *positions* (input dim 3)
        self.vertices_embed, vertices_pos_emb_dim = get_embedder(
            config.multires, input_dims=3
        )

        # 2. Vertex *feature* dimension (raw)
        # --- Feature embedding removed ---

        # 3. Pose processing setup
        self.control_points = config.control_points
        pose_input_raw_dim = self.control_points * 3
        self.pose_embed_dim = config.pose_embed_dim
        pose_mlp_width = config.get("pose_mlp_width", 128)
        self.pose_embed = nn.Sequential(
            nn.Linear(pose_input_raw_dim, pose_mlp_width),
            nn.ReLU(inplace=True),
            nn.Linear(pose_mlp_width, self.pose_embed_dim),
        )
        for layer in self.pose_embed:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(
                    layer.weight, a=np.sqrt(5), mode="fan_in", nonlinearity="relu"
                )
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        combined_pose_dim = self.pose_embed_dim * 2

        # 4. Adaptive Strength setup (using RAW aggregated features)
        self.adaptive_strength_config = self.config.get("adaptive_strength", {})
        self.use_adaptive_strength = self.adaptive_strength_config.get("enabled", False)
        self.strength_predictor = None
        if self.use_adaptive_strength:
            self.vertex_feature_dim = config.vertex_feature_dim  # Expect this in config
            predictor_hidden_dim = self.adaptive_strength_config.hidden_dim
            predictor_outputs = self.adaptive_strength_config.num_outputs
            # Strength predictor input dim is the RAW feature dimension
            self.strength_predictor = StrengthPredictor(
                self.vertex_feature_dim,  # <<< Use raw feature dim
                predictor_hidden_dim,
                predictor_outputs,
            )
            self.strength_min_scale = self.adaptive_strength_config.min_scale
            self.strength_outputs_to_scale = (
                self.adaptive_strength_config.outputs_to_scale
            )
            self.strength_aggregation_method = self.adaptive_strength_config.aggregation

        # 5. Deformation Networks
        # UvwDeform uses position embeddings
        self.uvw_deform = UvwDeformNetwork(
            config, vertices_pos_emb_dim, combined_pose_dim
        )
        # MeshDeform uses position embeddings
        self.mesh_deform = MeshDeformNetwork(
            config, vertices_pos_emb_dim, combined_pose_dim
        )

        # 6. Optimizer and Scheduler
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=config.lr_init, eps=1e-15
        )
        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=config.lr_init,
            lr_final=config.lr_final,
            lr_delay_mult=config.lr_delay_mult,
            max_steps=config.lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        lr = self.deform_scheduler_args(iteration)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def forward(
        self,
        refined_vertices,  # [N_verts_refined, 3] positions
        refined_vertex_features,  # [N_verts_refined, feature_dim] RAW features
        refined_faces,  # [N_faces_refined, 3] indices
        refined_faceid,  # [N_gaussians,] indices into faces
        refined_uvw,  # [N_gaussians, 3] barycentric coords
        coarse_original_pose,  # [N_control_points, 3]
        coarse_deformed_pose,  # [N_control_points, 3]
    ):
        N_gauss = refined_faceid.shape[0]
        N_verts = refined_vertices.shape[0]
        assert refined_uvw.shape == (
            N_gauss,
            3,
        ), f"Shape mismatch: refined_uvw {refined_uvw.shape}"
        # assert refined_vertex_features.shape == (N_verts, self.vertex_feature_dim), \
        #     f"Shape mismatch: refined_vertex_features {refined_vertex_features.shape} vs expected {(N_verts, self.vertex_feature_dim)}"
        assert coarse_original_pose.shape[0] == self.control_points, (
            "Original pose control points mismatch"
        )
        assert coarse_deformed_pose.shape[0] == self.control_points, (
            "Deformed pose control points mismatch"
        )

        # --- 1. Embeddings (Only Positions and Poses) ---
        # Embed vertex positions
        refined_vertices_pos_emb = self.vertices_embed(
            refined_vertices
        )  # [N_verts, pos_emb_dim]

        # Embed poses
        original_pose_flat = coarse_original_pose.flatten().unsqueeze(0)
        deformed_pose_flat = coarse_deformed_pose.flatten().unsqueeze(0)
        coarse_original_pose_emb = self.pose_embed(original_pose_flat).squeeze(0)
        coarse_deformed_pose_emb = self.pose_embed(deformed_pose_flat).squeeze(0)
        combined_pose_emb = torch.cat(
            [coarse_original_pose_emb, coarse_deformed_pose_emb]
        )

        # --- 2. Calculate Adaptive Strength (using RAW features) ---
        strength_scale = 1.0
        if self.use_adaptive_strength and self.strength_predictor is not None:
            # Aggregate RAW vertex features across all vertices
            if self.strength_aggregation_method == "mean":
                aggregated_raw_features = refined_vertex_features.mean(
                    dim=0
                )  # [feature_dim]
            elif self.strength_aggregation_method == "max":
                aggregated_raw_features = refined_vertex_features.max(
                    dim=0
                ).values  # [feature_dim]
            else:  # Default to mean
                aggregated_raw_features = refined_vertex_features.mean(dim=0)

            # Predict raw scale value(s) from aggregated RAW features
            # Ensure features are float for the Linear layer
            raw_scale_output = self.strength_predictor(
                aggregated_raw_features.float()
            )  # Shape [num_outputs]

            # Apply activation and scale mapping
            if raw_scale_output.numel() == 1:
                scale_0_1 = torch.sigmoid(raw_scale_output)
                strength_scale = (
                    self.strength_min_scale
                    + (1.0 - self.strength_min_scale) * scale_0_1
                )
            else:
                print(
                    f"Warning: Strength predictor has {raw_scale_output.numel()} outputs, averaging for single scale."
                )
                scale_0_1 = torch.sigmoid(raw_scale_output).mean()
                strength_scale = (
                    self.strength_min_scale
                    + (1.0 - self.strength_min_scale) * scale_0_1
                )

        # --- 3. Prepare Inputs for Deformation Networks ---
        # Expand pose embedding for batch operations
        pose_emb_expanded_gaussians = combined_pose_emb.unsqueeze(0).expand(N_gauss, -1)
        pose_emb_expanded_vertices = combined_pose_emb.unsqueeze(0).expand(N_verts, -1)

        # Get vertex *position* embeddings corresponding to each face, flatten per face
        face_vertex_pos_embeddings = refined_vertices_pos_emb[refined_faces]
        face_pos_embeddings_flat = face_vertex_pos_embeddings.reshape(
            refined_faces.shape[0], -1
        )

        # Select the flattened face embeddings for each Gaussian
        gaussians_vertex_pos_emb_flat = face_pos_embeddings_flat[refined_faceid]

        # --- 4. Forward Pass through Deformation Networks ---
        return_dict = {}

        # UVW Deformation (operates per Gaussian, uses pos embeddings)
        uvw_output = self.uvw_deform(
            refined_uvw, gaussians_vertex_pos_emb_flat, pose_emb_expanded_gaussians
        )
        return_dict.update(uvw_output)

        # Mesh Deformation (operates per vertex, uses pos embeddings)
        mesh_output = self.mesh_deform(
            refined_vertices_pos_emb, pose_emb_expanded_vertices
        )
        return_dict.update(mesh_output)

        # --- 5. Apply Adaptive Strength Scaling ---
        if self.use_adaptive_strength:
            if isinstance(strength_scale, float):
                example_tensor = next(
                    (v for v in return_dict.values() if isinstance(v, torch.Tensor)),
                    None,
                )
                if example_tensor is not None:
                    strength_scale = torch.tensor(
                        strength_scale,
                        device=example_tensor.device,
                        dtype=example_tensor.dtype,
                    )
                else:
                    strength_scale = torch.tensor(
                        strength_scale,
                        device=refined_vertices.device,
                        dtype=refined_vertices.dtype,
                    )

            for key in self.strength_outputs_to_scale:
                if key in return_dict and return_dict[key] is not None:
                    current_scale = strength_scale.to(
                        device=return_dict[key].device, dtype=return_dict[key].dtype
                    )
                    return_dict[key] = return_dict[key] * current_scale

        return return_dict
