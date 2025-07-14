import json
import os
from pathlib import Path
from typing import List, NamedTuple, Optional

import numpy as np
import torch
import torchvision

from model.smplx_utils import smplx_utils
from utils.camera_utils import camera_nerfies_from_JSON
from utils.graphics_utils import (
    focal2fov,
    fov2focal,
    getProjectionMatrix,
    getProjectionMatrix2,
    getWorld2View2,
    read_mesh,
)
import open3d as o3d


class Camera(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FoVx: float
    FoVy: float
    image_name: str
    original_image: torch.Tensor
    gt_mask: torch.Tensor
    fid: torch.Tensor
    image_width: int
    image_height: int
    zfar: float
    znear: float
    trans: np.ndarray
    scale: float
    world_view_transform: torch.Tensor
    projection_matrix: torch.Tensor
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor
    mesh_vertices: Optional[torch.Tensor] = None
    mesh_faces: Optional[torch.Tensor] = None
    mesh_normals: Optional[torch.Tensor] = None
    lang_feature: Optional[torch.Tensor] = None


def load_nerf_synthetic_camera_data(config, train=True) -> List[Camera]:
    resolution = config.dataset.resolution
    source_path = config.dataset.source_path
    white_background = config.dataset.white_background
    extension = config.dataset.extension
    device = config.dataset.data_device

    with open(
        os.path.join(
            source_path,
            "transforms_train.json" if train else "transforms_test.json",
        )
    ) as f:
        scene_json = json.load(f)

    fovx = scene_json["camera_angle_x"]
    frames = scene_json["frames"]
    cameras = [None] * len(frames)

    mesh_folder = os.path.join(source_path, "train_meshes" if train else "test_meshes")

    for idx in range(len(frames)):
        frame = frames[idx]
        image_path = os.path.join(source_path, frame["file_path"] + extension)
        image = torchvision.io.read_image(image_path).to(device) / 255.0

        C, orig_H, orig_W = image.shape
        if C == 4:
            bg = (
                torch.tensor([1, 1, 1], device=device)
                if white_background
                else torch.tensor([0, 0, 0], device=device)
            )
            mask = image[3:4, ...]
            image = image[:3, ...] * mask + bg[:, None, None] * (1 - mask)

        if resolution != 1:
            H, W = orig_H // resolution, orig_W // resolution
            image = torchvision.transforms.functional.resize(
                image.unsqueeze(0), (H, W)
            ).squeeze(0)
            mask = torchvision.transforms.functional.resize(
                mask.unsqueeze(0), (H, W)
            ).squeeze(0)

        else:
            H, W = orig_H, orig_W

        matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        fovy = focal2fov(fov2focal(fovx, W), H)
        FovY = fovx
        FovX = fovy
        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0
        znear = 0.1
        zfar = 100.0
        FoVx = FovX
        FoVy = FovY
        world_view_transform = torch.tensor(
            getWorld2View2(R, T, trans, scale)
        ).transpose(0, 1)
        projection_matrix = getProjectionMatrix(
            znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy
        ).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        mesh_path = os.path.join(
            mesh_folder, frame["file_path"].split("/")[-1] + ".ply"
        )
        mesh_vertices, mesh_faces, mesh_normals = read_mesh(mesh_path)
        mesh_vertices = mesh_vertices.to(device)
        mesh_faces = mesh_faces.to(device)
        mesh_normals = mesh_normals.to(device)

        cameras[idx] = Camera(
            uid=idx,
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            image_name=image_path,
            original_image=image,
            gt_mask=mask,
            fid=frame["time"],
            image_width=W,
            image_height=H,
            zfar=zfar,
            znear=znear,
            trans=trans,
            scale=scale,
            world_view_transform=world_view_transform,
            projection_matrix=projection_matrix,
            full_proj_transform=full_proj_transform,
            camera_center=camera_center,
            mesh_vertices=mesh_vertices,
            mesh_faces=mesh_faces,
            mesh_normals=mesh_normals,
            lang_feature=None,
        )

        for k, v in cameras[idx]._asdict().items():
            if isinstance(v, torch.Tensor):
                cameras[idx] = cameras[idx]._replace(**{k: v.to(device)})

    return cameras


def load_instant_avatar_data(config, train=True) -> List[Camera]:
    split = "train" if train else "test"
    source_path = config.dataset.source_path
    resolution = config.dataset.resolution
    white_background = config.dataset.white_background
    device = config.dataset.data_device

    assert os.path.exists(os.path.join(source_path, "config.json"))
    assert os.path.exists(os.path.join(source_path, "cameras.npz"))
    assert os.path.exists(os.path.join(source_path, "poses.npz"))

    with open(os.path.join(source_path, "config.json"), "r") as fp:
        scene_json = json.load(fp)

    start_idx = scene_json[split]["start"]
    end_idx = scene_json[split]["end"]
    step = scene_json[split]["skip"]
    frm_list = [i for i in range(start_idx, end_idx + 1, step)]

    smpl_config = {
        "model_type": "smpl",
        "gender": scene_json["gender"],
    }

    contents = np.load(os.path.join(source_path, "cameras.npz"))
    K = contents["intrinsic"]
    c2w = np.linalg.inv(contents["extrinsic"])
    contents["height"]
    contents["width"]
    w2c = np.linalg.inv(c2w)

    R = w2c[:3, :3]
    T = w2c[:3, 3]

    smpl_params = dict(np.load(os.path.join(source_path, "poses.npz")))

    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]

    smpl_params = {
        "betas": torch.tensor(smpl_params["betas"].astype(np.float32).reshape(1, 10)),
        "body_pose": torch.tensor(smpl_params["body_pose"].astype(np.float32))[
            frm_list
        ],
        "global_orient": torch.tensor(smpl_params["global_orient"].astype(np.float32))[
            frm_list
        ],
        "transl": torch.tensor(smpl_params["transl"].astype(np.float32))[frm_list],
    }

    smpl_model = smplx_utils.create_smplx_model(**smpl_config)
    with torch.no_grad():
        out = smpl_model(**smpl_params)

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(
        out["vertices"][0].detach().cpu().numpy()
    )
    mesh_o3d.triangles = o3d.utility.Vector3iVector(smpl_model.faces.astype(int))
    mesh_o3d.compute_vertex_normals()

    refine_fn = os.path.join(source_path, f"poses/anim_nerf_{split}.npz")
    if os.path.exists(refine_fn):
        print(f"[InstantAvatar] use refined smpl: {refine_fn}")
        split_smpl_params = np.load(refine_fn)
        refined_keys = [k for k in split_smpl_params if k != "betas"]
        smpl_params["betas"] = torch.tensor(split_smpl_params["betas"]).float()
        for key in refined_keys:
            smpl_params[key] = torch.tensor(split_smpl_params[key]).float()

    smpl_model = smplx_utils.create_smplx_model(**smpl_config)
    with torch.no_grad():
        out = smpl_model(**smpl_params)
    smpl_verts = out["vertices"]

    cameras = [None] * len(frm_list)
    for idx in range(len(frm_list)):
        frm_idx = frm_list[idx]

        image_path = os.path.join(source_path, f"images/image_{frm_idx:04d}.png")
        mask_path = os.path.join(source_path, f"masks/mask_{frm_idx:04d}.png")

        image = torchvision.io.read_image(image_path).to(device) / 255.0
        mask = torchvision.io.read_image(mask_path).to(device) / 255.0
        bg = (
            torch.tensor([1, 1, 1], device=device)
            if white_background
            else torch.tensor([0, 0, 0], device=device)
        )
        image = image[:3, ...] * mask + bg[:, None, None] * (1 - mask)
        C, H, W = image.shape

        if resolution != 1:
            H, W = H // resolution, W // resolution
            image = torchvision.transforms.functional.resize(
                image.unsqueeze(0), (H, W)
            ).squeeze(0)
            mask = torchvision.transforms.functional.resize(
                mask.unsqueeze(0), (H, W)
            ).squeeze(0)

        fx = K[0, 0].item()
        fy = K[1, 1].item()
        cx = K[0, 2].item()
        cy = K[1, 2].item()

        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0
        znear = 0.1
        zfar = 100.0
        world_view_transform = torch.tensor(
            getWorld2View2(R, T, trans, scale)
        ).transpose(0, 1)
        FoVy = 2 * np.arctan(H / (2.0 * fy))
        FoVx = 2 * np.arctan(W / (2.0 * fx))
        projection_matrix = getProjectionMatrix2(
            W, H, fx, fy, cx, cy, znear, zfar
        ).transpose(0, 1)

        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        mesh_o3d.vertices = o3d.utility.Vector3dVector(
            smpl_verts[idx].detach().cpu().numpy()
        )
        mesh_o3d.compute_vertex_normals()

        cameras[idx] = Camera(
            uid=idx,
            R=R,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            image_name=image_path,
            original_image=image,
            gt_mask=mask,
            fid=frm_idx,
            image_width=W,
            image_height=H,
            zfar=zfar,
            znear=znear,
            trans=trans,
            scale=scale,
            world_view_transform=world_view_transform,
            projection_matrix=projection_matrix,
            full_proj_transform=full_proj_transform,
            camera_center=camera_center,
            mesh_vertices=torch.tensor(
                np.asarray(mesh_o3d.vertices), device=device
            ).float(),
            mesh_normals=torch.tensor(
                np.asarray(mesh_o3d.triangle_normals), device=device
            ).float(),
            mesh_faces=torch.tensor(
                np.asarray(mesh_o3d.triangles), device=device
            ).long(),
        )

        for k, v in cameras[idx]._asdict().items():
            if isinstance(v, torch.Tensor):
                cameras[idx] = cameras[idx]._replace(**{k: v.to(device)})

    return cameras


def load_nerfies_camera_data(config, train: bool = True) -> List[Camera]:
    cfg_dataset_attrs = config.dataset
    dataset_path = cfg_dataset_attrs.source_path
    device = config.dataset.data_device

    with open(os.path.join(dataset_path, "scene.json"), "r") as f:
        scene_json = json.load(f)
    with open(os.path.join(dataset_path, "metadata.json"), "r") as f:
        meta_json = json.load(f)
    with open(os.path.join(dataset_path, "dataset.json"), "r") as f:
        dataset_json = json.load(f)

    coord_scale = float(scene_json.get("scale", 1.0))
    scene_center = np.array(scene_json.get("center", [0.0, 0.0, 0.0]), dtype=np.float32)

    path_parts = Path(dataset_path).parts
    dataset_name_identifier = path_parts[-1] if path_parts[-1] else path_parts[-2]

    nerfies_type = config.dataset.get("nerfies_type", "default")
    image_resize_ratio, split_type = {
        "vrig": (0.25, "train_val"),
        "NeRF": (1.0, "train_val"),
        "interp": (0.25, "interp_split"),
        "default": (0.5, "hypernerf_split"),
    }[nerfies_type]

    if split_type == "train_val":
        train_ids_orig = dataset_json["train_ids"]
        val_ids_orig = dataset_json["val_ids"]
    elif split_type == "interp_split":
        all_ids_orig_list = dataset_json["ids"]
        train_ids_orig = all_ids_orig_list[::4]
        val_ids_orig = all_ids_orig_list[2::4]
    elif split_type == "hypernerf_split":
        all_ids_orig_list = dataset_json["ids"]
        train_ids_orig = all_ids_orig_list
        val_ids_orig = all_ids_orig_list[: min(4, len(all_ids_orig_list))]
    else:
        train_ids_orig = dataset_json.get("train_ids", dataset_json.get("ids"))
        val_ids_orig = dataset_json.get("val_ids", [])
        if not train_ids_orig:
            raise NotImplementedError()

    all_img = train_ids_orig if train else val_ids_orig

    all_cam = [meta_json[i]["camera_id"] for i in all_img]
    if "time_id" in meta_json[all_img[0]]:
        all_time = [meta_json[i]["time_id"] for i in all_img]
        max_time = max(all_time)
        all_time = [meta_json[i]["time_id"] / max_time for i in all_img]
    else:
        all_time = all_cam
        max_time = max(all_time)
        all_time = [meta_json[i]["camera_id"] / max_time for i in all_img]
    set(all_time)

    all_cam_params = []
    for im in all_img:
        camera = camera_nerfies_from_JSON(
            f"{config.dataset.source_path}/camera/{im}.json", image_resize_ratio
        )
        camera["position"] = camera["position"] - scene_center
        camera["position"] = camera["position"] * coord_scale
        all_cam_params.append(camera)

    all_img = [
        f"{config.dataset.source_path}/rgb/{int(1 / image_resize_ratio)}x/{i}.png"
        for i in all_img
    ]

    cam_infos: List[Camera] = [None] * len(all_img)
    for idx in range(len(all_img)):
        image_path = all_img[idx]
        image = torchvision.io.read_image(image_path).to(device) / 255.0
        image_name = Path(image_path).stem

        orientation = all_cam_params[idx]["orientation"].T
        position = -all_cam_params[idx]["position"] @ orientation
        focal = all_cam_params[idx]["focal_length"]
        fid = all_time[idx]
        T = position
        R = orientation
        H, W = image.shape[1], image.shape[2]
        FovY = focal2fov(focal, H)
        FovX = focal2fov(focal, W)
        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0
        znear = 0.1
        zfar = 100.0
        world_view_transform = torch.tensor(
            getWorld2View2(R, T, trans, scale)
        ).transpose(0, 1)
        projection_matrix = getProjectionMatrix(
            znear=znear, zfar=zfar, fovX=FovX, fovY=FovY
        ).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        mesh_path = os.path.join(
            dataset_path,
            "train_meshes" if train else "test_meshes",
            image_name + ".ply",
        )
        mesh_vertices, mesh_faces, mesh_normals = read_mesh(mesh_path)
        cam_info = Camera(
            uid=idx,
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            image_name=image_name,
            original_image=image,
            gt_mask=torch.ones_like(image[:1, ...]),
            fid=fid,
            image_width=W,
            image_height=H,
            zfar=zfar,
            znear=znear,
            trans=trans,
            scale=scale,
            world_view_transform=world_view_transform,
            projection_matrix=projection_matrix,
            full_proj_transform=full_proj_transform,
            camera_center=camera_center,
            mesh_vertices=mesh_vertices,
            mesh_faces=mesh_faces,
            mesh_normals=mesh_normals,
        )
        cam_infos[idx] = cam_info
        for k, v in cam_infos[idx]._asdict().items():
            if isinstance(v, torch.Tensor):
                cam_infos[idx] = cam_infos[idx]._replace(**{k: v.to(device)})

    return cam_infos


datasetCallbacks = {
    "Blender": load_nerf_synthetic_camera_data,
    "InstantAvatar": load_instant_avatar_data,
    "Nerfies": load_nerfies_camera_data,
}
