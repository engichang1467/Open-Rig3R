# tests/test_visualization.py
import sys
import numpy as np
import torch
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from utils.visualization import (
    filter_by_confidence,
    compute_colors,
    resolve_poses,
    build_point_cloud,
    build_camera_frustum,
    build_all_frustums,
    build_ray_visualization,
    visualize_outputs,
)

# Shared dimensions: img_size=128, patch_size=8 → P=16*16=256, HW=128*128=16384
B, V, HW, P = 1, 3, 128 * 128, 16 * 16
H = W = 128


def _make_outputs():
    return {
        "pointmap":      torch.randn(B, V, HW, 3),
        "pointmap_conf": torch.randn(B, V, HW, 1),
        "rig_raymap":    torch.randn(B, V, P, 6),
        "pose_raymap":   torch.randn(B, V, P, 3),
    }


def _make_images():
    return torch.rand(B, V, 3, H, W)


# ── filter_by_confidence ──────────────────────────────────────────────────────

def test_filter_by_confidence_shape():
    conf = torch.randn(B, V, HW, 1)
    mask = filter_by_confidence(conf, percentile=80.0)
    assert mask.shape == (B, V, HW)
    assert mask.dtype == torch.bool


def test_filter_by_confidence_percentile():
    conf = torch.randn(B, V, HW, 1)
    mask = filter_by_confidence(conf, percentile=80.0)
    # ~20 % of pixels kept at the 80th percentile
    kept_frac = mask.float().mean().item()
    assert 0.15 < kept_frac < 0.25, f"unexpected keep fraction: {kept_frac:.3f}"


def test_filter_by_confidence_zero_percentile():
    conf = torch.randn(B, V, HW, 1)
    mask = filter_by_confidence(conf, percentile=0.0)
    assert mask.all(), "percentile=0 should keep every pixel"


def test_filter_by_confidence_squeezed_input():
    # Should also accept (B, V, HW) without trailing dim
    conf = torch.randn(B, V, HW)
    mask = filter_by_confidence(conf, percentile=50.0)
    assert mask.shape == (B, V, HW)


# ── compute_colors ────────────────────────────────────────────────────────────

def test_compute_colors_rgb():
    images = _make_images()
    conf = torch.randn(B, V, HW)
    colors = compute_colors(images, conf, None, mode="rgb")
    assert colors.shape == (B, V, HW, 3)
    assert colors.min() >= 0.0 and colors.max() <= 1.0


def test_compute_colors_per_camera():
    images = _make_images()
    conf = torch.randn(B, V, HW)
    colors = compute_colors(images, conf, None, mode="per-camera")
    assert colors.shape == (B, V, HW, 3)
    # All pixels in a given view share the same color
    for v in range(V):
        assert torch.allclose(colors[0, v, 0], colors[0, v, 1])


def test_compute_colors_confidence():
    images = _make_images()
    conf = torch.randn(B, V, HW)
    colors = compute_colors(images, conf, None, mode="confidence", cmap_name="plasma")
    assert colors.shape == (B, V, HW, 3)


def test_compute_colors_rig_cluster():
    images = _make_images()
    conf = torch.randn(B, V, HW)
    labels = np.array([0, 0, 1])
    colors = compute_colors(images, conf, labels, mode="rig-cluster")
    assert colors.shape == (B, V, HW, 3)
    # Views in the same cluster must share the same solid color
    assert torch.allclose(colors[0, 0, 0], colors[0, 1, 0])
    assert not torch.allclose(colors[0, 0, 0], colors[0, 2, 0])


# ── resolve_poses ─────────────────────────────────────────────────────────────

def test_resolve_poses_no_cam2rig():
    rig_raymap = torch.randn(V, P, 6)
    poses = resolve_poses(rig_raymap, cam2rig=None)
    assert len(poses) == V
    for p in poses:
        assert "R" in p and "t" in p
        assert p["R"].shape == (3, 3)
        assert p["t"].shape == (3,)


def test_resolve_poses_all_identity_falls_back():
    rig_raymap = torch.randn(V, P, 6)
    cam2rig = torch.eye(4).unsqueeze(0).repeat(V, 1, 1)  # all identity
    poses = resolve_poses(rig_raymap, cam2rig=cam2rig)
    assert len(poses) == V


def test_resolve_poses_uses_provided_cam2rig():
    rig_raymap = torch.randn(V, P, 6)
    cam2rig = torch.eye(4).unsqueeze(0).repeat(V, 1, 1)
    expected_t = torch.tensor([1.0, 2.0, 3.0])
    cam2rig[0, :3, 3] = expected_t  # make view 0 non-identity
    poses = resolve_poses(rig_raymap, cam2rig=cam2rig)
    assert torch.allclose(poses[0]["t"], expected_t)


# ── build_point_cloud ─────────────────────────────────────────────────────────

def test_build_point_cloud_output_type():
    import open3d as o3d
    rig_raymap = torch.randn(V, P, 6)
    poses = resolve_poses(rig_raymap, cam2rig=None)
    conf = torch.randn(B, V, HW, 1)
    conf_mask = filter_by_confidence(conf, 80.0)[0]   # (V, HW)
    images = _make_images()
    colors_v = compute_colors(images, conf.squeeze(-1), None, "rgb")[0]  # (V,HW,3)
    pointmaps = torch.randn(V, HW, 3)
    pcd = build_point_cloud(pointmaps, conf_mask, colors_v, poses)
    assert isinstance(pcd, o3d.geometry.PointCloud)
    assert len(pcd.points) > 0
    assert len(pcd.colors) == len(pcd.points)


def test_build_point_cloud_no_nan():
    import open3d as o3d
    import numpy as np
    rig_raymap = torch.randn(V, P, 6)
    poses = resolve_poses(rig_raymap, cam2rig=None)
    conf = torch.randn(B, V, HW, 1)
    conf_mask = filter_by_confidence(conf, 0.0)[0]    # keep all
    images = _make_images()
    colors_v = compute_colors(images, conf.squeeze(-1), None, "rgb")[0]
    pointmaps = torch.randn(V, HW, 3)
    # Inject some NaN/Inf values
    pointmaps[0, 0] = float("nan")
    pointmaps[1, 0] = float("inf")
    pcd = build_point_cloud(pointmaps, conf_mask, colors_v, poses)
    pts = np.asarray(pcd.points)
    assert np.isfinite(pts).all(), "point cloud must not contain NaN or Inf"


# ── build_camera_frustum ──────────────────────────────────────────────────────

def test_build_camera_frustum_edges():
    import open3d as o3d
    frustum = build_camera_frustum(
        np.eye(3), np.zeros(3), focal_length=128.0,
        img_hw=(H, W), scale=0.1,
    )
    assert isinstance(frustum, o3d.geometry.LineSet)
    assert len(frustum.lines) == 8
    assert len(frustum.points) == 5


def test_build_all_frustums_count():
    rig_raymap = torch.randn(V, P, 6)
    poses = resolve_poses(rig_raymap, cam2rig=None)
    frustums = build_all_frustums(poses, cluster_labels=None, focal_lengths=None, img_hw=(H, W))
    assert len(frustums) == V


def test_build_all_frustums_cluster_coloring():
    import open3d as o3d
    rig_raymap = torch.randn(V, P, 6)
    poses = resolve_poses(rig_raymap, cam2rig=None)
    labels = np.array([0, 0, 1])
    frustums = build_all_frustums(poses, cluster_labels=labels, focal_lengths=None, img_hw=(H, W))
    assert len(frustums) == V
    # Views 0 and 1 are in the same cluster → same edge colors
    c0 = np.asarray(frustums[0].colors)[0]
    c1 = np.asarray(frustums[1].colors)[0]
    c2 = np.asarray(frustums[2].colors)[0]
    assert np.allclose(c0, c1)
    assert not np.allclose(c0, c2)


# ── build_ray_visualization ───────────────────────────────────────────────────

def test_build_ray_visualization_segment_count():
    rig_raymap = torch.randn(V, P, 6)
    n_samples = 8
    rays = build_ray_visualization(rig_raymap, n_samples=n_samples, ray_length=0.5)
    assert len(rays.lines) == V * n_samples
    assert len(rays.points) == V * n_samples * 2


def test_build_ray_visualization_finite_points():
    import numpy as np
    rig_raymap = torch.randn(V, P, 6)
    rays = build_ray_visualization(rig_raymap, n_samples=16)
    pts = np.asarray(rays.points)
    assert np.isfinite(pts).all()


# ── visualize_outputs (end-to-end, headless) ──────────────────────────────────

def test_visualize_outputs_returns_pcd():
    import open3d as o3d
    result = visualize_outputs(
        _make_outputs(), _make_images(),
        cam2rig=None,
        conf_percentile=80.0,
        color_mode="per-camera",
        show_frustums=True,
        show_rays=True,
        n_ray_samples=8,
        show=False,
    )
    assert isinstance(result["pcd"], o3d.geometry.PointCloud)
    assert len(result["pcd"].points) > 0
    assert len(result["frustums"]) == V
    assert result["rays"] is not None


def test_visualize_outputs_all_color_modes():
    for mode in ["rgb", "per-camera", "confidence"]:
        result = visualize_outputs(
            _make_outputs(), _make_images(),
            color_mode=mode, show=False,
        )
        assert len(result["pcd"].points) > 0, f"no points for color_mode={mode}"


def test_visualize_outputs_rig_cluster_mode():
    result = visualize_outputs(
        _make_outputs(), _make_images(),
        color_mode="rig-cluster",
        n_clusters=2,
        show=False,
    )
    assert result["labels"] is not None
    assert len(result["labels"]) == V


def test_visualize_outputs_export(tmp_path):
    out_ply = str(tmp_path / "test_output.ply")
    visualize_outputs(
        _make_outputs(), _make_images(),
        export_path=out_ply, show=False,
    )
    assert Path(out_ply).exists()
    assert Path(out_ply).stat().st_size > 0


def test_visualize_outputs_no_frustums_no_rays():
    result = visualize_outputs(
        _make_outputs(), _make_images(),
        show_frustums=False, show_rays=False, show=False,
    )
    assert result["frustums"] == []
    assert result["rays"] is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
