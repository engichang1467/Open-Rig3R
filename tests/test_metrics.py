# tests/test_metrics.py
"""Ray angular error: the one number that survives a change to the loss."""
import sys
from pathlib import Path

import torch

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from utils.metrics import (
    align_scale,
    chamfer_distance,
    ray_angular_error,
    raymap_metrics,
    rig_discovery_accuracy,
    rig_maa,
)


def raymap(directions, center=None):
    """Wrap directions as a (..., 6) raymap, as the heads emit them."""
    center = torch.zeros_like(directions) if center is None else center
    return torch.cat([center, directions], dim=-1)


def test_identical_rays_score_zero():
    directions = torch.nn.functional.normalize(torch.randn(1, 2, 8, 3), dim=-1)
    error = ray_angular_error(raymap(directions), raymap(directions))
    torch.testing.assert_close(error, torch.tensor(0.0), atol=1e-4, rtol=0)
    print("Identical rays score zero degrees test passed!")


def test_known_angles_are_recovered():
    """A 90 degree turn must read as 90 degrees, not as a loss-shaped number."""
    x = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    y = torch.tensor([[[[0.0, 1.0, 0.0]]]])
    torch.testing.assert_close(
        ray_angular_error(raymap(x), raymap(y)), torch.tensor(90.0), atol=1e-3, rtol=0
    )

    diagonal = torch.tensor([[[[1.0, 1.0, 0.0]]]])
    torch.testing.assert_close(
        ray_angular_error(raymap(x), raymap(diagonal)), torch.tensor(45.0), atol=1e-3, rtol=0
    )
    print("Known angles recovered exactly test passed!")


def test_camera_center_is_ignored():
    """Centres are metric in some arms and z-bar normalized in others.

    The metric must read only the direction half, or it stops being comparable across
    exactly the change it exists to measure.
    """
    directions = torch.nn.functional.normalize(torch.randn(1, 2, 8, 3), dim=-1)
    metric = raymap(directions, center=torch.zeros_like(directions))
    normalized = raymap(directions, center=torch.full_like(directions, 37.0))

    torch.testing.assert_close(
        ray_angular_error(metric, normalized), torch.tensor(0.0), atol=1e-4, rtol=0
    )
    print("Camera centre does not affect the angle test passed!")


def test_unnormalized_predictions_still_give_an_angle():
    """An angle is scale-free; a head that stopped normalizing must not skew it."""
    directions = torch.nn.functional.normalize(torch.randn(1, 2, 8, 3), dim=-1)
    scaled = directions * 7.5

    torch.testing.assert_close(
        ray_angular_error(raymap(scaled), raymap(directions)),
        torch.tensor(0.0), atol=1e-4, rtol=0,
    )
    print("Direction magnitude does not affect the angle test passed!")


def test_opposite_rays_are_180_degrees():
    """Both domain edges must stay finite, not just the near-zero one."""
    x = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    error = ray_angular_error(raymap(x), raymap(-x))
    assert torch.isfinite(error), "angle went NaN at the domain edge"
    torch.testing.assert_close(error, torch.tensor(180.0), atol=1e-2, rtol=0)
    print("Opposed rays give 180 degrees, no NaN test passed!")


def test_near_zero_angles_stay_precise():
    """A converged model lives near zero, which is where acos loses its conditioning.

    acos(a.b) reports ~3e-3 degrees of noise on rays identical up to float32 rounding;
    2*atan2(|a-b|, |a+b|) stays exact there. The metric has to resolve differences
    smaller than the run-to-run noise floor, so this precision is load-bearing.
    """
    directions = torch.nn.functional.normalize(torch.randn(1, 2, 64, 3), dim=-1)
    rescaled = directions * 7.5  # same rays, different magnitude

    stable = ray_angular_error(raymap(rescaled), raymap(directions))

    cosine = (torch.nn.functional.normalize(rescaled, dim=-1) * directions).sum(-1)
    naive = torch.rad2deg(torch.acos(cosine.clamp(-1, 1))).mean()

    assert stable < 1e-4, f"atan2 form should be exact here, got {stable}"
    print(f"near-zero angle: atan2 {stable:.2e} deg vs acos {naive:.2e} deg test passed!")


def test_metrics_skip_absent_raymaps():
    """CO3D supervises no raymaps, so nothing should be reported for it."""
    directions = torch.nn.functional.normalize(torch.randn(1, 2, 8, 3), dim=-1)
    preds = {"rig_raymap": raymap(directions), "pose_raymap": raymap(directions)}

    both = raymap_metrics(preds, dict(preds))
    assert set(both) == {"pose_deg", "rig_deg"}, both

    partial = raymap_metrics(preds, {"rig_raymap": raymap(directions)})
    assert set(partial) == {"rig_deg"}, partial

    assert raymap_metrics(preds, {}) == {}
    print("Metrics reported only for supervised raymaps test passed!")


def test_matches_a_hand_computed_mean():
    """The reduction is a mean over every ray, not per view or per batch."""
    a = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])  # 90 and 45 degrees

    error = ray_angular_error(raymap(a), raymap(b))
    torch.testing.assert_close(error, torch.tensor(67.5), atol=1e-3, rtol=0)
    print(f"Mean of 90 and 45 is {error:.1f} degrees test passed!")


def test_align_scale_recovers_a_known_factor():
    """The model predicts at z-bar scale; metric evaluation has to undo that."""
    gt = torch.randn(200, 3) * 12.0
    for factor in (26.0, 1.0, 0.1):
        recovered = align_scale(gt / factor, gt)
        torch.testing.assert_close(recovered, torch.tensor(factor), rtol=1e-4, atol=1e-4)
    print("align_scale recovers the normalization factor test passed!")


def test_align_scale_survives_outliers():
    """Reconstructed clouds carry outliers, which is why this is a median not a mean."""
    gt = torch.randn(200, 3) * 12.0
    pred = gt / 26.0
    pred[0] = torch.tensor([1e6, 1e6, 1e6])  # one wild point

    recovered = align_scale(pred, gt)
    assert abs(recovered - 26.0) < 1.0, f"one outlier moved the scale to {recovered}"
    print(f"align_scale with an outlier: {recovered:.3f} vs 26.0 test passed!")


def test_align_scale_handles_empty_clouds():
    """evaluate.py can hand over an empty prediction; that must not divide by zero."""
    gt = torch.randn(10, 3)
    assert align_scale(torch.empty(0, 3), gt) == 1.0
    assert align_scale(gt, torch.empty(0, 3)) == 1.0
    print("align_scale is safe on empty clouds test passed!")


def test_existing_metrics_are_still_here():
    """utils/metrics.py predates this branch: evaluate.py and infer_rig_discovery.py
    import chamfer_distance, rig_discovery_accuracy and rig_maa from it."""
    a = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert chamfer_distance(a, a) == 0.0
    assert rig_discovery_accuracy(a, a) == 1.0

    identity = [{"R": torch.eye(3)}, {"R": torch.eye(3)}]
    torch.testing.assert_close(rig_maa(identity, identity), torch.tensor(0.0), atol=1e-5, rtol=0)
    print("Pre-existing metrics still present and working test passed!")


if __name__ == "__main__":
    test_align_scale_recovers_a_known_factor()
    test_align_scale_survives_outliers()
    test_align_scale_handles_empty_clouds()
    test_existing_metrics_are_still_here()
    test_identical_rays_score_zero()
    test_known_angles_are_recovered()
    test_camera_center_is_ignored()
    test_unnormalized_predictions_still_give_an_angle()
    test_opposite_rays_are_180_degrees()
    test_near_zero_angles_stay_precise()
    test_metrics_skip_absent_raymaps()
    test_matches_a_hand_computed_mean()
