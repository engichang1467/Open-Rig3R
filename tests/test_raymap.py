import sys
import traceback
from pathlib import Path

import torch

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from utils.raymap import (
    build_pointmap_target,
    build_raymap_targets,
    camera_ray_directions,
    patch_centers,
    reference_from_camera,
)


IMAGE_SIZE = (128, 128)
PATCH_SIZE = 8
GRID = IMAGE_SIZE[0] // PATCH_SIZE
P = GRID * GRID

# a 128x128 image with the principal point at the centre
INTRINSICS = torch.tensor([[[100.0, 100.0, 64.0, 64.0]]])  # (1, 1, 4)


def mean_axis(rays):
    """Optical axis of a (P, 3) ray bundle: the rays fan out, so renormalize."""
    return torch.nn.functional.normalize(rays.mean(0), dim=0)


def identity_batch(n_views=1):
    cam2rig = torch.eye(4).repeat(1, n_views, 1, 1)
    world_from_rig = torch.eye(4).repeat(1, n_views, 1, 1)
    intrinsics = INTRINSICS.repeat(1, n_views, 1)
    return cam2rig, intrinsics, world_from_rig


def test_patch_centers_are_row_major():
    """Token order is row-major, so the grid must be built the same way"""
    centers = patch_centers(IMAGE_SIZE, PATCH_SIZE, torch.device("cpu"))

    assert centers.shape == (P, 2), f"Expected {(P, 2)}, got {tuple(centers.shape)}"
    assert torch.allclose(centers[0], torch.tensor([4.0, 4.0])), "First patch centre wrong"
    # second token steps along u (same row), not down a column
    assert centers[1][1] == centers[0][1], "Token 1 should be in the same row as token 0"
    assert centers[GRID][0] == centers[0][0], "Token GRID should start the next row"
    print("✓ test_patch_centers_are_row_major passed")


def test_principal_ray_points_forward():
    """The ray through the principal point is the camera's forward axis"""
    directions = camera_ray_directions(INTRINSICS, IMAGE_SIZE, PATCH_SIZE)

    assert directions.shape == (1, 1, P, 3), f"Got {tuple(directions.shape)}"
    assert torch.allclose(directions.norm(dim=-1), torch.ones(1, 1, P), atol=1e-5), (
        "Directions should be unit length"
    )
    # patch centres straddle the principal point, so no ray sits exactly on it;
    # the four central patches must be symmetric about forward = +x
    grid = directions.reshape(GRID, GRID, 3)
    central = grid[GRID // 2 - 1 : GRID // 2 + 1, GRID // 2 - 1 : GRID // 2 + 1]
    mean = central.reshape(-1, 3).mean(0)
    assert torch.allclose(mean[1:], torch.zeros(2), atol=1e-6), (
        f"Central rays should straddle +x, got mean {mean}"
    )
    assert mean[0] > 0.99, f"Forward component should dominate, got {mean[0]:.3f}"
    print("✓ test_principal_ray_points_forward passed")


def test_image_axes_map_to_camera_axes():
    """u grows to the right (-y in Waymo), v grows downward (-z)"""
    directions = camera_ray_directions(INTRINSICS, IMAGE_SIZE, PATCH_SIZE)
    grid = directions.reshape(GRID, GRID, 3)

    left, right = grid[GRID // 2, 0], grid[GRID // 2, -1]
    top, bottom = grid[0, GRID // 2], grid[-1, GRID // 2]

    assert left[1] > 0 > right[1], f"Left ray should have +y, got {left[1]:.3f} / {right[1]:.3f}"
    assert top[2] > 0 > bottom[2], f"Top ray should have +z, got {top[2]:.3f} / {bottom[2]:.3f}"
    print("✓ test_image_axes_map_to_camera_axes passed")


def test_rig_origin_is_camera_mount():
    """rig_raymap origins are the camera positions in the rig frame"""
    cam2rig, intrinsics, world_from_rig = identity_batch(n_views=2)
    mount = torch.tensor([1.5, -0.1, 2.1])
    cam2rig[0, 1, :3, 3] = mount

    targets = build_raymap_targets(cam2rig, intrinsics, world_from_rig, IMAGE_SIZE, PATCH_SIZE)
    origins = targets["rig_raymap"][..., :3]

    assert targets["rig_raymap"].shape == (1, 2, P, 6), f"Got {tuple(targets['rig_raymap'].shape)}"
    assert torch.allclose(origins[0, 0], torch.zeros(3)), "View 0 sits at the rig origin"
    assert torch.allclose(origins[0, 1], mount.expand(P, 3)), "View 1 origin should be its mount"
    print("✓ test_rig_origin_is_camera_mount passed")


def test_rig_rotation_is_applied():
    """A camera rotated on the rig produces rays rotated in the rig frame"""
    cam2rig, intrinsics, world_from_rig = identity_batch(n_views=2)
    # yaw view 1 by 90 degrees: camera +x (forward) becomes rig +y
    cam2rig[0, 1, :3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    targets = build_raymap_targets(cam2rig, intrinsics, world_from_rig, IMAGE_SIZE, PATCH_SIZE)
    directions = targets["rig_raymap"][..., 3:]

    unrotated = mean_axis(directions[0, 0])
    rotated = mean_axis(directions[0, 1])
    assert unrotated[0] > 0.99, f"View 0 should look along +x, got {unrotated}"
    assert rotated[1] > 0.99, f"View 1 should look along +y, got {rotated}"
    print("✓ test_rig_rotation_is_applied passed")


def test_pose_raymap_is_relative_to_first_view():
    """Pose targets are expressed in view 0's frame, so view 0 is always canonical"""
    cam2rig, intrinsics, world_from_rig = identity_batch(n_views=2)
    world_from_rig[0, 0, :3, 3] = torch.tensor([10.0, 5.0, 0.0])  # anywhere in the world
    world_from_rig[0, 1, :3, 3] = torch.tensor([14.0, 5.0, 0.0])  # rig drove 4 m forward

    targets = build_raymap_targets(cam2rig, intrinsics, world_from_rig, IMAGE_SIZE, PATCH_SIZE)
    pose = targets["pose_raymap"]
    camera = camera_ray_directions(intrinsics, IMAGE_SIZE, PATCH_SIZE)

    assert pose.shape == (1, 2, P, 3), f"Got {tuple(pose.shape)}"
    assert torch.allclose(pose[0, 0], camera[0, 0], atol=1e-5), (
        "View 0 in its own frame should equal its camera-frame rays"
    )
    # pure translation leaves directions unchanged; only rotation turns them
    assert torch.allclose(pose[0, 1], camera[0, 1], atol=1e-5), (
        "Translation alone should not rotate ray directions"
    )
    print("✓ test_pose_raymap_is_relative_to_first_view passed")


def test_pose_raymap_sees_rig_rotation():
    """A turning rig must change the pose targets, or there is nothing to learn"""
    cam2rig, intrinsics, world_from_rig = identity_batch(n_views=2)
    world_from_rig[0, 1, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    targets = build_raymap_targets(cam2rig, intrinsics, world_from_rig, IMAGE_SIZE, PATCH_SIZE)
    pose = targets["pose_raymap"]

    assert not torch.allclose(pose[0, 0], pose[0, 1], atol=1e-3), (
        "Pose targets are identical across a 90 degree turn - supervision is degenerate"
    )
    turned = mean_axis(pose[0, 1])
    assert turned[1] > 0.99, f"View 1 should look along view 0's +y, got {turned}"
    print("✓ test_pose_raymap_sees_rig_rotation passed")


def test_pointmap_pools_to_patch_grid():
    """Export grid is coarser than the model's patch grid, so it must pool down"""
    cam2rig, _, world_from_rig = identity_batch(n_views=1)
    export_grid = GRID * 2
    pointmap = torch.full((1, 1, export_grid, export_grid, 3), 3.0)

    points, confidence = build_pointmap_target(
        pointmap, cam2rig, world_from_rig, PATCH_SIZE, IMAGE_SIZE
    )

    assert points.shape == (1, 1, P, 3), f"Got {tuple(points.shape)}"
    assert confidence.shape == (1, 1, P), f"Got {tuple(confidence.shape)}"
    assert torch.allclose(points, torch.full((1, 1, P, 3), 3.0)), "Pooling changed the values"
    assert torch.allclose(confidence, torch.ones(1, 1, P)), "Fully covered patches should be 1"
    print("✓ test_pointmap_pools_to_patch_grid passed")


def test_pointmap_confidence_tracks_coverage():
    """Empty cells must not be averaged in as zeros, or every target drifts toward 0"""
    cam2rig, _, world_from_rig = identity_batch(n_views=1)
    export_grid = GRID * 2
    pointmap = torch.full((1, 1, export_grid, export_grid, 3), torch.nan)
    # fill one cell of each 2x2 block with a real return
    pointmap[:, :, ::2, ::2] = 4.0
    # and leave one whole patch completely empty
    pointmap[:, :, 0:2, 0:2] = torch.nan

    points, confidence = build_pointmap_target(
        pointmap, cam2rig, world_from_rig, PATCH_SIZE, IMAGE_SIZE
    )

    assert confidence[0, 0, 0] == 0.0, "The empty patch should have zero confidence"
    assert torch.allclose(points[0, 0, 0], torch.zeros(3)), "Empty patches should be zeroed"
    assert torch.allclose(confidence[0, 0, 1], torch.tensor(0.25)), (
        f"One of four cells covered should be 0.25, got {confidence[0, 0, 1]}"
    )
    assert torch.allclose(points[0, 0, 1], torch.full((3,), 4.0)), (
        f"Covered cell should survive averaging, got {points[0, 0, 1]}"
    )
    print("✓ test_pointmap_confidence_tracks_coverage passed")


def test_pointmap_moves_into_reference_frame():
    """Points arrive in each camera's own frame and must land in view 0's"""
    cam2rig, _, world_from_rig = identity_batch(n_views=2)
    world_from_rig[0, 1, :3, 3] = torch.tensor([4.0, 0.0, 0.0])  # view 1 is 4 m ahead

    pointmap = torch.full((1, 2, GRID, GRID, 3), 0.0)
    pointmap[..., 0] = 10.0  # both views see something 10 m down their own +x

    points, _ = build_pointmap_target(
        pointmap, cam2rig, world_from_rig, PATCH_SIZE, IMAGE_SIZE
    )

    assert torch.allclose(points[0, 0, :, 0], torch.full((P,), 10.0)), "View 0 is the reference"
    assert torch.allclose(points[0, 1, :, 0], torch.full((P,), 14.0)), (
        f"View 1's point should be 14 m out in view 0's frame, got {points[0, 1, 0]}"
    )
    print("✓ test_pointmap_moves_into_reference_frame passed")


def test_pointmap_lies_along_its_own_rays():
    """The end-to-end invariant: a target point sits on the ray of its own patch"""
    cam2rig, intrinsics, world_from_rig = identity_batch(n_views=2)
    cam2rig[0, 1, :3, 3] = torch.tensor([0.0, 1.0, 0.0])  # view 1 mounted a metre left
    world_from_rig[0, 1, :3, 3] = torch.tensor([2.0, 0.0, 0.0])

    # place a point 12 m along each patch's own camera ray
    directions = camera_ray_directions(intrinsics, IMAGE_SIZE, PATCH_SIZE)
    pointmap = (directions * 12.0).reshape(1, 2, GRID, GRID, 3)

    points, _ = build_pointmap_target(
        pointmap, cam2rig, world_from_rig, PATCH_SIZE, IMAGE_SIZE
    )
    targets = build_raymap_targets(cam2rig, intrinsics, world_from_rig, IMAGE_SIZE, PATCH_SIZE)

    centers = reference_from_camera(cam2rig, world_from_rig)[..., :3, 3]
    for view in range(2):
        offset = points[0, view] - centers[0, view]
        cosine = torch.nn.functional.cosine_similarity(
            offset, targets["pose_raymap"][0, view], dim=-1
        )
        assert cosine.min() > 0.9999, f"View {view} points drift off their rays: {cosine.min()}"
    print("✓ test_pointmap_lies_along_its_own_rays passed")


def run_all_tests():
    tests = [
        test_patch_centers_are_row_major,
        test_principal_ray_points_forward,
        test_image_axes_map_to_camera_axes,
        test_rig_origin_is_camera_mount,
        test_rig_rotation_is_applied,
        test_pose_raymap_is_relative_to_first_view,
        test_pose_raymap_sees_rig_rotation,
        test_pointmap_pools_to_patch_grid,
        test_pointmap_confidence_tracks_coverage,
        test_pointmap_moves_into_reference_frame,
        test_pointmap_lies_along_its_own_rays,
    ]

    passed = 0
    failed = 0

    print("\nRunning Raymap Target Tests")
    print("=" * 50)

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            traceback.print_exc()
            failed += 1

    print("=" * 50)
    print(f"\nResults: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
