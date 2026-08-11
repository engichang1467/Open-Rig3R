# tests/test_scene_scale.py
"""Eq. 3 / Eq. 4 scale normalization: one z-bar per sample, shared by every target."""
import sys
from pathlib import Path

import torch

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from models.losses import MultiTaskLoss
from utils.raymap import build_raymap_targets, scene_scale

IMAGE_SIZE = (64, 64)
PATCH_SIZE = 16
P = (IMAGE_SIZE[0] // PATCH_SIZE) ** 2
V = 2


def test_scene_scale_is_mean_distance_of_valid_points():
    points = torch.tensor([[[3.0, 4.0, 0.0], [6.0, 8.0, 0.0]]])  # norms 5 and 10
    torch.testing.assert_close(scene_scale(points), torch.tensor([7.5]))
    print("scene_scale averages point distances test passed!")


def test_scene_scale_ignores_missing_returns():
    """Waymo pointmaps are NaN wherever no lidar return landed."""
    points = torch.tensor([[[3.0, 4.0, 0.0], [float("nan")] * 3, [0.0, 0.0, 0.0]]])
    torch.testing.assert_close(scene_scale(points), torch.tensor([5.0]))
    print("scene_scale skips NaN and empty points test passed!")


def test_scene_scale_is_per_sample():
    points = torch.stack([
        torch.tensor([[3.0, 4.0, 0.0]]),   # 5
        torch.tensor([[30.0, 40.0, 0.0]]),  # 50
    ])
    torch.testing.assert_close(scene_scale(points), torch.tensor([5.0, 50.0]))
    print("scene_scale is per sample test passed!")


def scene(scale=1.0):
    """A 2-view rig with a mounted second camera, optionally blown up by `scale`."""
    cam2rig = torch.eye(4).repeat(1, V, 1, 1)
    cam2rig[0, 1, :3, 3] = torch.tensor([1.5, -0.5, 0.25]) * scale
    world_from_rig = torch.eye(4).repeat(1, V, 1, 1)
    world_from_rig[0, 1, :3, 3] = torch.tensor([4.0, 0.0, 0.0]) * scale
    intrinsics = torch.tensor([[[32.0, 32.0, 32.0, 32.0]] * V])
    points = torch.tensor([[[[10.0, 1.0, 2.0], [20.0, -3.0, 1.0]]]]) * scale
    return cam2rig, intrinsics, world_from_rig, points


def normalized_targets(scale):
    cam2rig, intrinsics, world_from_rig, points = scene(scale)
    z_bar = scene_scale(points)
    return build_raymap_targets(
        cam2rig, intrinsics, world_from_rig, IMAGE_SIZE, PATCH_SIZE, z_bar=z_bar
    ), z_bar


def test_targets_are_invariant_to_scene_scale():
    """The whole point of z-bar: the same geometry ten times bigger trains the same.

    A driving scene and a tabletop differ by orders of magnitude in metres; after
    normalization they must land on the same numbers.
    """
    small, z_small = normalized_targets(1.0)
    large, z_large = normalized_targets(10.0)

    assert torch.allclose(z_large, z_small * 10), f"{z_small} vs {z_large}"

    for key in ("camera_center_rig", "camera_center_pose", "rig_raymap", "pose_raymap"):
        torch.testing.assert_close(small[key], large[key], rtol=1e-5, atol=1e-6)

    print(f"z-bar {z_small.item():.3f} -> {z_large.item():.3f}, targets identical test passed!")


def test_unnormalized_targets_are_not_invariant():
    """Negative control: without z-bar the same test would pass vacuously."""
    cam2rig, intrinsics, world_from_rig, _ = scene(1.0)
    small = build_raymap_targets(cam2rig, intrinsics, world_from_rig, IMAGE_SIZE, PATCH_SIZE)
    cam2rig, intrinsics, world_from_rig, _ = scene(10.0)
    large = build_raymap_targets(cam2rig, intrinsics, world_from_rig, IMAGE_SIZE, PATCH_SIZE)

    delta = (large["camera_center_rig"] - small["camera_center_rig"]).abs().max()
    assert delta > 1.0, "scaling the scene should move unnormalized centres"
    print(f"unnormalized centres move by {delta:.3f} test passed!")


def test_loss_is_invariant_to_scene_scale():
    """End to end: a model predicting normalized geometry scores the same either way."""
    torch.manual_seed(0)
    small, _ = normalized_targets(1.0)
    large, _ = normalized_targets(10.0)

    preds = {
        "rig_raymap": small["rig_raymap"] + 0.05 * torch.randn(1, V, P, 6),
        "camera_center_rig": small["camera_center_rig"] + 0.05 * torch.randn(1, V, 3),
    }
    criterion = MultiTaskLoss()
    loss_small, _ = criterion(preds, small)
    loss_large, _ = criterion(preds, large)

    torch.testing.assert_close(loss_small, loss_large, rtol=1e-5, atol=1e-6)
    print(f"loss {loss_small.item():.6f} unchanged across a 10x scene test passed!")


if __name__ == "__main__":
    test_scene_scale_is_mean_distance_of_valid_points()
    test_scene_scale_ignores_missing_returns()
    test_scene_scale_is_per_sample()
    test_targets_are_invariant_to_scene_scale()
    test_unnormalized_targets_are_not_invariant()
    test_loss_is_invariant_to_scene_scale()
