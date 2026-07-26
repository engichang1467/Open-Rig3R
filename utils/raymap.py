"""Geometric ground truth from camera calibration and rig poses.

The model's raymap heads predict one ray per patch token, so the targets are
built on the same patch grid: `pose_raymap` as unit directions relative to the
first view, `rig_raymap` as (origin, direction) in the rig frame.

Waymo camera convention: x forward, y left, z up, with
    u = f_u * (-y / x) + c_u
    v = f_v * (-z / x) + c_v
so a pixel back-projects to [1, -(u - c_u) / f_u, -(v - c_v) / f_v].
"""

import torch


def patch_centers(image_size, patch_size, device):
    """Pixel centres of each patch, row-major to match ViT token order. (P, 2)"""
    height, width = image_size
    rows = (torch.arange(height // patch_size, device=device) + 0.5) * patch_size
    cols = (torch.arange(width // patch_size, device=device) + 0.5) * patch_size
    v, u = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([u.reshape(-1), v.reshape(-1)], dim=-1)


def camera_ray_directions(intrinsics, image_size, patch_size):
    """Unit ray directions in each camera's own frame.

    Args:
        intrinsics: (B, V, 4) as [f_u, f_v, c_u, c_v], scaled to image_size
        image_size: (H, W) of the images actually fed to the model
        patch_size: model patch size
    Returns:
        (B, V, P, 3)
    """
    centers = patch_centers(image_size, patch_size, intrinsics.device)  # (P, 2)
    u, v = centers[:, 0], centers[:, 1]

    f_u, f_v, c_u, c_v = (intrinsics[..., i].unsqueeze(-1) for i in range(4))
    y = -(u - c_u) / f_u  # broadcasts (B, V, 1) against (P,) -> (B, V, P)
    z = -(v - c_v) / f_v
    directions = torch.stack([torch.ones_like(y), y, z], dim=-1)
    return torch.nn.functional.normalize(directions, dim=-1)


def reference_from_camera(cam2rig, world_from_rig):
    """Each view's camera pose relative to view 0. (B, V, 4, 4)"""
    world_from_camera = world_from_rig @ cam2rig
    return torch.linalg.inv(world_from_camera[:, :1]) @ world_from_camera


def build_pointmap_target(pointmap, cam2rig, world_from_rig, patch_size, image_size):
    """Sparse per-camera lidar pointmaps -> pointmap target in view 0's frame.

    Args:
        pointmap:       (B, V, G, G, 3) points in each view's own camera frame,
                        NaN where no lidar return landed
        cam2rig:        (B, V, 4, 4)
        world_from_rig: (B, V, 4, 4)
        patch_size:     model patch size
        image_size:     (H, W)
    Returns:
        (points, confidence) of shape (B, V, P, 3) and (B, V, P), confidence being
        the fraction of cells in each patch that carried a return.
    """
    B, V, G, _, _ = pointmap.shape
    grid = image_size[0] // patch_size

    # pool the export grid down to the patch grid, ignoring empty cells
    blocks = pointmap.reshape(B, V, grid, G // grid, grid, G // grid, 3)
    blocks = blocks.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, V, grid * grid, -1, 3)
    valid = ~torch.isnan(blocks[..., 0])
    confidence = valid.float().mean(-1)
    points = torch.nan_to_num(blocks).sum(-2) / valid.sum(-1).clamp(min=1).unsqueeze(-1)

    # each view's points sit in its own camera frame; the prediction is in view 0's
    transform = reference_from_camera(cam2rig, world_from_rig)
    points = torch.einsum("bvij,bvpj->bvpi", transform[..., :3, :3], points)
    points = points + transform[..., :3, 3].unsqueeze(2)

    return points * (confidence > 0).unsqueeze(-1), confidence


def build_raymap_targets(cam2rig, intrinsics, world_from_rig, image_size, patch_size):
    """Targets for the rig and pose raymap heads.

    Args:
        cam2rig:        (B, V, 4, 4) vehicle_from_camera
        intrinsics:     (B, V, 4) [f_u, f_v, c_u, c_v] scaled to image_size
        world_from_rig: (B, V, 4, 4) world_from_vehicle at each view's timestamp
        image_size:     (H, W)
        patch_size:     model patch size
    Returns:
        {"rig_raymap": (B, V, P, 6), "pose_raymap": (B, V, P, 3)}
    """
    directions = camera_ray_directions(intrinsics, image_size, patch_size)

    # --- rig frame: origin is the camera's mounting point, static per camera ---
    rotation = cam2rig[..., :3, :3]
    translation = cam2rig[..., :3, 3]
    rig_directions = torch.einsum("bvij,bvpj->bvpi", rotation, directions)
    rig_origins = translation.unsqueeze(2).expand_as(rig_directions)
    rig_raymap = torch.cat([rig_origins, rig_directions], dim=-1)

    # --- pose frame: relative to view 0, so rig motion shows up in the target ---
    transform = reference_from_camera(cam2rig, world_from_rig)
    pose_directions = torch.einsum("bvij,bvpj->bvpi", transform[..., :3, :3], directions)

    return {
        "rig_raymap": rig_raymap,
        "pose_raymap": torch.nn.functional.normalize(pose_directions, dim=-1),
    }
