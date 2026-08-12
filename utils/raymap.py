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


def scene_scale(points, eps=1e-6):
    """Average scene depth z-bar, per sample. (B,)

    Eq. 3 and Eq. 4 divide the ground truth by z-bar so the model learns geometry at a
    scale-invariant magnitude - indoor rooms and driving scenes land in the same
    numerical range. The paper says only "the average scene depth"; following DUSt3R
    this is the mean distance from the camera to its valid points, so it is a depth in
    the camera's own frame rather than a distance in some reference view.

    Args:
        points: (B, ..., 3) ground-truth points in camera frame, NaN where invalid
    """
    flat = points.reshape(points.shape[0], -1, 3)
    distances = flat.norm(dim=-1)

    valid = torch.isfinite(distances) & (distances > 0)
    total = torch.where(valid, distances, torch.zeros_like(distances)).sum(dim=-1)
    return (total / valid.sum(dim=-1).clamp(min=1)).clamp(min=eps)


def build_raymap_targets(cam2rig, intrinsics, world_from_rig, image_size, patch_size,
                         z_bar=None):
    """Targets for the rig and pose raymap heads.

    Args:
        cam2rig:        (B, V, 4, 4) vehicle_from_camera
        intrinsics:     (B, V, 4) [f_u, f_v, c_u, c_v] scaled to image_size
        world_from_rig: (B, V, 4, 4) world_from_vehicle at each view's timestamp
        image_size:     (H, W)
        patch_size:     model patch size
        z_bar:          (B,) average scene depth. Eq. 4 normalizes the camera centre
                        by it; pass the same value used for the pointmap target or the
                        two supervisions disagree about scale.
    Returns:
        {"rig_raymap":  (B, V, P, 6), "camera_center_rig":  (B, V, 3),
         "pose_raymap": (B, V, P, 6), "camera_center_pose": (B, V, 3)}

        Each raymap is the frame's camera centre broadcast over its patches,
        concatenated with per-patch unit ray directions, matching the head's output.
        The centres are returned separately because Eq. 4 scores them on their own
        term - all rays of a frame share one centre, so it is not P values.
    """
    directions = camera_ray_directions(intrinsics, image_size, patch_size)

    # --- rig frame: centre is the camera's mounting point, static per camera ---
    rotation = cam2rig[..., :3, :3]
    rig_center = cam2rig[..., :3, 3]
    rig_directions = torch.einsum("bvij,bvpj->bvpi", rotation, directions)

    # --- pose frame: relative to view 0, so rig motion shows up in the target ---
    transform = reference_from_camera(cam2rig, world_from_rig)
    pose_center = transform[..., :3, 3]
    pose_directions = torch.einsum("bvij,bvpj->bvpi", transform[..., :3, :3], directions)
    pose_directions = torch.nn.functional.normalize(pose_directions, dim=-1)

    # Eq. 4 normalizes only the ground-truth centre; the directions are unit vectors
    # already and carry no scale to remove.
    if z_bar is not None:
        scale = z_bar.view(-1, 1, 1)
        rig_center = rig_center / scale
        pose_center = pose_center / scale

    return {
        "rig_raymap": _raymap(rig_center, rig_directions),
        "camera_center_rig": rig_center,
        "pose_raymap": _raymap(pose_center, pose_directions),
        "camera_center_pose": pose_center,
    }


def _raymap(center, directions):
    """(B, V, 3) centre + (B, V, P, 3) directions -> (B, V, P, 6)"""
    return torch.cat([center.unsqueeze(2).expand_as(directions), directions], dim=-1)
