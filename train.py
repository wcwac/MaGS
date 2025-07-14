#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import datetime
import os
import random
import sys

import cv2

# import lpips
import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from dataset import get_dataset, getNerfppNorm
from gaussian_renderer import inria_render
from model.deform_model import DeformModel
from model.mags_model import MaGSModel
from utils.general_utils import safe_state
from utils.loss_utils import arap_loss, l1_loss, ssim, psnr

lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").cuda()

best_psnr = 0.0
best_l1 = 1e10
best_lpips = 1e10
best_iter = 0

cnt = 0


def collect_loss(config, gt_image, image):
    Ll1 = l1_loss(image, gt_image)

    if config.loss.lambda_ssim > 0:
        Lssim = 1.0 - ssim(image, gt_image)
        loss = (1.0 - config.loss.lambda_ssim) * Ll1 + config.loss.lambda_ssim * Lssim
    else:
        loss = Ll1
    return {"loss": loss}


def training(config):
    train_dataset = get_dataset(config, train=True)
    test_dataset = get_dataset(config, train=False)
    refined_faces = train_dataset[0].mesh_faces.clone().cuda()
    refined_meshes = {
        "train_verts": torch.stack(
            [i.mesh_vertices.clone().cuda() for i in train_dataset]
        ),
        "train_normals": torch.stack(
            [i.mesh_normals.clone().cuda() for i in train_dataset]
        ),
        "test_verts": torch.stack(
            [i.mesh_vertices.clone().cuda() for i in test_dataset]
        ),
        "test_normals": torch.stack(
            [i.mesh_normals.clone().cuda() for i in test_dataset]
        ),
    }

    gaussians = MaGSModel(config)
    gaussians.create_from_mesh(
        refined_meshes["train_verts"][0],
        refined_faces,
        refined_meshes["train_normals"][0],
    )
    gaussians.training_setup(config.optim)
    nerf_normalization = getNerfppNorm(train_dataset)
    (nerf_normalization["radius"] if nerf_normalization["radius"] > 0 else 2.0)

    config.deform.control_points = train_dataset[0].mesh_vertices.shape[0]
    deformer = DeformModel(config.deform).cuda()

    bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(config.model.model_path, "tensorboard"))
    progress_bar = tqdm(
        range(1, config.optim.iterations + 1),
        desc="Training progress",
        dynamic_ncols=True,
    )
    BATCH_SIZE = config.optim.batch_size
    batch_loss = 0.0

    for iteration in progress_bar:
        if iteration % 100 == 0:
            gaussians.oneupSHdegree()
        if iteration % len(train_dataset) == 0:
            random.shuffle(train_dataset)

        cam = train_dataset[iteration % len(train_dataset)]
        gaussians.update_mesh(
            refined_faces,
            refined_meshes["train_verts"][cam.uid],
            refined_meshes["train_normals"][cam.uid],
            refined_meshes["train_verts"][0],
            refined_meshes["train_verts"][0],
            refined_meshes["train_verts"][cam.uid],
            deformer if iteration > config.optim.warm_up else None,
        )

        if config.optim.random_bg:
            background = torch.rand(3, dtype=torch.float32, device="cuda")
        render_pkg = inria_render(
            cam,
            gaussians,
            config.pipe,
            background,
            iteration=iteration,
        )
        # image = render_pkg["render"]
        image, _viewspace_point_tensor, _visibility_filter, _radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        gt_image = cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)

        if config.loss.lambda_ssim > 0:
            Lssim = 1.0 - ssim(image, gt_image)
            loss = (
                1.0 - config.loss.lambda_ssim
            ) * Ll1 + config.loss.lambda_ssim * Lssim
        else:
            loss = Ll1
        batch_loss += loss

        if iteration % BATCH_SIZE == 0:
            batch_loss /= BATCH_SIZE
            batch_loss.backward()
            batch_loss = 0.0

        current_psnr = psnr(
            image.unsqueeze(0).contiguous(), gt_image.unsqueeze(0)
        ).item()
        progress_bar.set_postfix(
            {
                "#gauss": gaussians.xyz.shape[0],
                "#faces": refined_faces.shape[0],
                "#verts": refined_meshes["train_verts"][cam.uid].shape[0],
                "psnr": current_psnr,
            }
        )

        writer.add_scalar("Loss/train", loss.item(), iteration)
        writer.add_scalar("PSNR/train", current_psnr, iteration)
        writer.add_scalar("Gaussians/count", gaussians.xyz.shape[0], iteration)
        writer.add_scalar("Faces/count", refined_faces.shape[0], iteration)
        writer.add_scalar(
            "LearningRate/optim_uvw",
            gaussians.optimizer.param_groups[0]["lr"],
            iteration,
        )

        with torch.no_grad():
            if iteration % config.dataset.test_per_iter == 0:
                render_test(
                    config,
                    gaussians,
                    deformer,
                    refined_meshes,
                    refined_faces,
                    iteration,
                    train_dataset,
                    test_dataset,
                    background,
                    writer,
                )

            # if iteration < config.optim.densify_until_iter:
            #     gaussians.max_radii2D[visibility_filter] = torch.max(
            #         gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            #     )
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1])

            # if (
            #     iteration > config.optim.densify_from_iter
            #     and iteration % config.optim.densification_interval == 0
            # ):
            #     size_threshold = (
            #         config.optim.size_threshold
            #         if iteration > config.optim.opacity_reset_interval
            #         else None
            #     )
            #     gaussians.densify_and_prune(
            #         config.optim.densify_grad_threshold,
            #         config.optim.min_opacity,
            #         cameras_extent,
            #         size_threshold,
            #     )

            if iteration < config.optim.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)
                deformer.optimizer.step()
                deformer.update_learning_rate(iteration)
                deformer.optimizer.zero_grad(set_to_none=True)

    writer.close()


def colorizeWeightsMap(
    weights, colormap=cv2.COLORMAP_JET, min_val=None, max_val=None, to_rgb=False
):
    if min_val is None:
        min_val = weights.min()
    if max_val is None:
        max_val = weights.max()

    vals = (weights - min_val) / (max_val - min_val)
    vals = (vals.clip(0, 1) * 255).astype(np.uint8)
    canvas = cv2.applyColorMap(vals, colormap=colormap)
    if to_rgb:
        return canvas[..., [2, 1, 0]]
    else:
        return canvas


def visualize_compare(gt_image, image, psnr, ssim, lpips):
    compare = torch.concat([gt_image, image], dim=2)
    compare = (compare.permute([1, 2, 0]) * 255)[:, :, [2, 1, 0]].detach().cpu().numpy()
    compare = cv2.putText(
        compare, "psnr/ssim/lpips", (20, compare.shape[0] - 50), 0, 1, (0, 0, 255)
    )
    compare = cv2.putText(
        compare,
        f"{psnr:.4f}/{ssim:.4f}/{lpips:.4f}",
        (20, compare.shape[0] - 10),
        0,
        1,
        (0, 0, 255),
    )

    err = (image - gt_image).abs().max(dim=0)[0].clip(0, 1)
    err_map = colorizeWeightsMap(err.detach().cpu().numpy(), min_val=0, max_val=1)
    compare = np.concatenate([compare, err_map], axis=1)
    return compare


def render_test(
    config,
    gaussians,
    deformer,
    refined_meshes,
    refined_faces,
    iteration,
    train_dataset,
    test_dataset,
    background,
    writer,
):
    global best_psnr, best_l1, best_lpips, best_ssim, best_iter
    torch.cuda.empty_cache()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    if config.dataset.white_background or config.optim.random_bg:
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    for name, case in {"test": test_dataset, "train": train_dataset}.items():
        psnrs = []
        l1s = []
        lpipss = []
        ssims = []
        output_path = f"{config.model.model_path}/{iteration}"
        os.makedirs(output_path, exist_ok=True)
        sum_time = 0
        for idx, cam in enumerate(case):
            start_time = datetime.datetime.now()
            gaussians.update_mesh(
                refined_faces,
                refined_meshes[f"{name}_verts"][cam.uid],
                refined_meshes[f"{name}_normals"][cam.uid],
                refined_meshes["train_verts"][0],
                refined_meshes["train_verts"][0],
                refined_meshes[f"{name}_verts"][cam.uid],
                deformer if iteration > config.optim.warm_up else None,
            )
            render_pkg = inria_render(
                cam,
                gaussians,
                config.pipe,
                background,
            )
            end_time = datetime.datetime.now()
            sum_time += (end_time - start_time).total_seconds()
            image = torch.clamp(
                render_pkg["render"],
                0.0,
                1.0,
            )
            gt_image = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
            if config.optim.random_bg or config.dataset.white_background:
                H, W = gt_image.shape[1:]
                mask = cam.gt_mask.repeat(3, 1, 1)
                bg = background.reshape(3, 1, 1).repeat(1, H, W)
                gt_image = gt_image * mask + bg * (1 - mask)

            image = image.contiguous()
            psnr_val = psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).item()
            l1_val = l1_loss(image.unsqueeze(0), gt_image.unsqueeze(0)).item()
            lpips_val = lpips(image.unsqueeze(0), gt_image.unsqueeze(0)).item()
            ssim_val = ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).item()

            psnrs.append(psnr_val)
            l1s.append(l1_val)
            lpipss.append(lpips_val)
            ssims.append(ssim_val)

            if name == "test":
                torchvision.utils.save_image(
                    image, f"{output_path}/test_{idx}_render.png"
                )
                torchvision.utils.save_image(
                    gt_image, f"{output_path}/test_{idx}_gt.png"
                )
                compare = visualize_compare(
                    gt_image, image, psnr_val, ssim_val, lpips_val
                )
                cv2.imwrite(f"{output_path}/test_{idx}_compare.png", compare)

        l1_test = sum(l1s) / len(l1s)
        psnr_test = sum(psnrs) / len(psnrs)
        lpips_test = sum(lpipss) / len(lpipss)
        ssim_test = sum(ssims) / len(ssims)
        fps = len(case) / sum_time

        print(
            "\n[ITER {}] Evaluating {}: L1 {:.7f} PSNR {:.7f} LPIPS {:.7f} SSIM {:.7f}".format(
                iteration, name, l1_test, psnr_test, lpips_test, ssim_test
            )
        )
        print(f"FPS: {fps:.2f}")
        if psnr_test > best_psnr and name == "test":
            best_psnr = psnr_test
            best_l1 = l1_test
            best_lpips = lpips_test
            best_ssim = ssim_test
            best_iter = iteration

        # Log test metrics to TensorBoard
        writer.add_scalar(f"Loss/{name}", l1_test, iteration)
        writer.add_scalar(f"PSNR/{name}", psnr_test, iteration)
        writer.add_scalar(f"LPIPS/{name}", lpips_test, iteration)
        writer.add_scalar(f"SSIM/{name}", ssim_test, iteration)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    configs = [OmegaConf.load(_) for _ in sys.argv[-1].split(",")]
    config = OmegaConf.merge(*configs)
    print("Optimizing " + config.model.model_path)
    experiment_id = datetime.datetime.now().strftime("%d%H%M%S")
    config.model.model_path = os.path.join(config.model.model_path, experiment_id)
    os.makedirs(config.model.model_path, exist_ok=True)
    with open(os.path.join(config.model.model_path, "cfg_args.yaml"), "w") as cfg_log_f:
        OmegaConf.save(config, cfg_log_f)

    # Initialize system state (RNG)
    safe_state(config)
    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(True)
    training(config)
    print(
        f"Best PSNR: {best_psnr:.4f} L1: {best_l1:.4f} LPIPS: {best_lpips:.4f} SSIM: {best_ssim:.4f} at iteration {best_iter}"
    )
    # All done
    print("\nTraining complete.")
