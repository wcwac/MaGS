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

import math
from typing import NamedTuple, Tuple

import numpy as np
import torch
import open3d as o3d
import os


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getProjectionMatrix2(w, h, fx, fy, cx, cy, znear=0.1, zfar=100.0):
    z_sign = 1.0
    P = torch.tensor(
        [
            [2 * fx / w, 0, (2 * cx - w) / w, 0],
            [0, 2 * fy / h, (2 * cy - h) / h, 0],
            [0, 0, z_sign * zfar / (zfar - znear), -(zfar * znear) / (zfar - znear)],
            [0, 0, z_sign, 0],
        ]
    ).float()
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, -1, keepdim=True)


def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2 * dot(x, n) * n - x


def length(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / length(x, eps)


def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0, 1), mode="constant", value=w)


def compute_face_normals(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    return face_normals


def compute_face_orientation(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]

    a0 = safe_normalize(v1 - v0)
    a1 = safe_normalize(torch.cross(a0, v2 - v0, dim=-1))
    a2 = -safe_normalize(
        torch.cross(a1, a0, dim=-1)
    )  # will have artifacts without negation

    orientation = torch.cat([a0[..., None], a1[..., None], a2[..., None]], dim=-1)

    s0 = length(v1 - v0)
    s1 = dot(a2, (v2 - v0)).abs()
    scale = (s0 + s1) / 2
    return orientation, scale


def compute_vertex_normals(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    v_normals = torch.zeros_like(verts)
    N = verts.shape[0]
    v_normals.scatter_add_(1, i0[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i1[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i2[..., None].repeat(N, 1, 3), face_normals)

    v_normals = torch.where(
        dot(v_normals, v_normals) > 1e-20,
        v_normals,
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device="cuda"),
    )
    v_normals = safe_normalize(v_normals)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_normals))
    return v_normals


def get_mesh_normals(
    mesh_vertices: torch.Tensor, mesh_faces: torch.Tensor
) -> torch.Tensor:
    # use open3d to compute normals
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices.numpy())
    mesh.triangles = o3d.utility.Vector3iVector(mesh_faces.numpy())
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    normals = torch.tensor(mesh.triangle_normals, dtype=torch.float32)
    return normals


def read_mesh(mesh_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert os.path.exists(mesh_path), f"Mesh file {mesh_path} does not exist."
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh_vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    mesh_faces = torch.tensor(mesh.triangles, dtype=torch.int64)
    mesh_normals = get_mesh_normals(mesh_vertices, mesh_faces)
    return mesh_vertices, mesh_faces, mesh_normals


def uvw_to_xyz(faces, vertices, normals, face_ids, uvws):
    v0 = vertices[faces[face_ids, 0]]
    v1 = vertices[faces[face_ids, 1]]
    v2 = vertices[faces[face_ids, 2]]
    u, v, w = uvws[:, 0], uvws[:, 1], uvws[:, 2]
    points_on_surface = (
        u.unsqueeze(-1) * v0 + v.unsqueeze(-1) * v1 + (1 - u - v).unsqueeze(-1) * v2
    )
    return points_on_surface + w.unsqueeze(-1) * normals[face_ids]


def uvw_to_value(faces, vertices_value, face_ids, uvws):
    v0 = vertices_value[faces[face_ids, 0]]
    v1 = vertices_value[faces[face_ids, 1]]
    v2 = vertices_value[faces[face_ids, 2]]
    u, v = uvws[:, 0], uvws[:, 1]
    return u.unsqueeze(-1) * v0 + v.unsqueeze(-1) * v1 + (1 - u - v).unsqueeze(-1) * v2
