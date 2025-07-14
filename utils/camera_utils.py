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

import json

import numpy as np
from utils.graphics_utils import fov2focal


def camera_to_JSON(id, camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry


def camera_nerfies_from_JSON(path, scale):
    """Loads a JSON camera into memory."""
    with open(path, "r") as fp:
        camera_json = json.load(fp)

    # Fix old camera JSON.
    if "tangential" in camera_json:
        camera_json["tangential_distortion"] = camera_json["tangential"]

    return dict(
        orientation=np.array(camera_json["orientation"]),
        position=np.array(camera_json["position"]),
        focal_length=camera_json["focal_length"] * scale,
        principal_point=np.array(camera_json["principal_point"]) * scale,
        skew=camera_json["skew"],
        pixel_aspect_ratio=camera_json["pixel_aspect_ratio"],
        radial_distortion=np.array(camera_json["radial_distortion"]),
        tangential_distortion=np.array(camera_json["tangential_distortion"]),
        image_size=np.array(
            (
                int(round(camera_json["image_size"][0] * scale)),
                int(round(camera_json["image_size"][1] * scale)),
            )
        ),
    )
