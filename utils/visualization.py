"""
Open3D visualization pipeline for Rig3R outputs.

Provides primitives for converting raw model outputs (pointmaps, confidence,
raymaps, poses) into Open3D geometries, plus a top-level convenience function
that runs the full pipeline.
"""

import numpy as np
import torch
import open3d as o3d
import matplotlib.cm as mpl_cm
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.rig_discovery import (
    recover_pose_closed_form,
    cluster_rig_poses,
    reconstruct_pointcloud,
)

# ─────────────────────────────────────────────
# 1. Confidence filtering
# ─────────────────────────────────────────────

def filter_by_confidence(
    conf: torch.Tensor,
    percentile: float = 80.0,
) -> torch.Tensor:
    """
    Build a boolean keep-mask from raw confidence values.

    The confidence channel from PointMapHead has no activation function, so
    values are arbitrary floats — higher means more confident.  A global
    percentile threshold is used so that views with uniformly low confidence
    do not contribute disproportionate points.

    Args:
        conf:       (B, V, H*W, 1) or (B, V, H*W) float tensor.
        percentile: Keep pixels at or above this percentile, computed globally
                    across all pixels in the batch.  0.0 keeps everything.

    Returns:
        mask: (B, V, H*W) bool tensor. True = include this pixel.
    """
    # Squeeze trailing dim if present
    if conf.dim() == 4 and conf.shape[-1] == 1:
        conf = conf.squeeze(-1)  # (B, V, H*W)

    if percentile <= 0.0:
        return torch.ones_like(conf, dtype=torch.bool)

    threshold = torch.quantile(conf.float().reshape(-1), percentile / 100.0)
    return conf >= threshold


# ─────────────────────────────────────────────
# 2. Color computation
# ─────────────────────────────────────────────

def _tab20_color(idx: int) -> Tuple[float, float, float]:
    """Return an RGB tuple from the tab20 colormap."""
    return mpl_cm.get_cmap("tab20")(idx % 20)[:3]


def compute_colors(
    images: torch.Tensor,
    conf: torch.Tensor,
    cluster_labels: Optional[np.ndarray],
    mode: str = "rgb",
    cmap_name: str = "plasma",
) -> torch.Tensor:
    """
    Compute per-pixel RGB colors for all views.

    Args:
        images:         (B, V, 3, H, W) float32 [0, 1] source images.
        conf:           (B, V, H*W) float tensor (squeezed, before masking).
        cluster_labels: (V,) int array from cluster_rig_poses, or None.
                        Only used for 'rig-cluster' mode.
        mode:           'rgb'        — per-pixel color from source image
                        'per-camera' — each view gets a solid tab20 color
                        'rig-cluster'— views in same cluster share a tab20 color
                        'confidence' — matplotlib cmap applied to per-view
                                       min-max normalized confidence
        cmap_name:      matplotlib colormap name used for 'confidence' mode.

    Returns:
        colors: (B, V, H*W, 3) float32 [0, 1] RGB tensor.
    """
    B, V, _, H, W = images.shape
    HW = H * W
    device = images.device

    if mode == "rgb":
        # (B, V, 3, H, W) → (B, V, H, W, 3) → (B, V, H*W, 3)
        colors = images.permute(0, 1, 3, 4, 2).reshape(B, V, HW, 3)
        colors = colors.clamp(0.0, 1.0)

    elif mode == "per-camera":
        colors = torch.zeros(B, V, HW, 3, device=device)
        for v in range(V):
            c = torch.tensor(_tab20_color(v), dtype=torch.float32, device=device)
            colors[:, v, :, :] = c.unsqueeze(0).expand(HW, 3)

    elif mode == "rig-cluster":
        if cluster_labels is None:
            # Fall back to per-camera coloring if no cluster info
            cluster_labels = np.arange(V)
        colors = torch.zeros(B, V, HW, 3, device=device)
        for v in range(V):
            c = torch.tensor(
                _tab20_color(int(cluster_labels[v])),
                dtype=torch.float32,
                device=device,
            )
            colors[:, v, :, :] = c.unsqueeze(0).expand(HW, 3)

    elif mode == "confidence":
        cmap = mpl_cm.get_cmap(cmap_name)
        colors = torch.zeros(B, V, HW, 3, device=device)
        for b in range(B):
            for v in range(V):
                c_flat = conf[b, v].float().cpu().numpy()  # (H*W,)
                c_min, c_max = c_flat.min(), c_flat.max()
                if c_max > c_min:
                    c_norm = (c_flat - c_min) / (c_max - c_min)
                else:
                    c_norm = np.zeros_like(c_flat)
                c_rgb = torch.tensor(
                    cmap(c_norm)[:, :3], dtype=torch.float32, device=device
                )
                colors[b, v] = c_rgb
    else:
        raise ValueError(
            f"Unknown color mode '{mode}'. "
            "Choose from: rgb, per-camera, rig-cluster, confidence"
        )

    return colors


# ─────────────────────────────────────────────
# 3. Pose resolution
# ─────────────────────────────────────────────

def resolve_poses(
    rig_raymap: torch.Tensor,
    cam2rig: Optional[torch.Tensor],
) -> List[Dict[str, torch.Tensor]]:
    """
    Produce per-view poses for a single batch element.

    For each view:
    - If cam2rig[v] is not an identity matrix, extract R and t from it.
    - Otherwise fall back to recover_pose_closed_form(rig_raymap[v]).

    This handles partial metadata dropout gracefully: views that have real
    extrinsics use them; views with identity matrices (dropped during training)
    are recovered from the predicted rig raymap.

    Args:
        rig_raymap: (V, P, 6) tensor for one batch element.
        cam2rig:    (V, 4, 4) SE(3) tensor for one batch element, or None.

    Returns:
        List of V dicts, each {'R': (3,3) tensor, 't': (3,) tensor}.
    """
    V, P, _ = rig_raymap.shape
    H_p = int(P ** 0.5)
    poses = []

    for v in range(V):
        use_provided = False
        if cam2rig is not None:
            eye4 = torch.eye(4, device=cam2rig.device, dtype=cam2rig.dtype)
            if not torch.allclose(cam2rig[v], eye4, atol=1e-4):
                use_provided = True

        if use_provided:
            R_mat = cam2rig[v, :3, :3]
            t_vec = cam2rig[v, :3, 3]
            poses.append({"R": R_mat, "t": t_vec})
        else:
            rmap = rig_raymap[v].reshape(H_p, H_p, 6)
            pose = recover_pose_closed_form(rmap)
            poses.append(pose)

    return poses


# ─────────────────────────────────────────────
# 4. Point cloud assembly
# ─────────────────────────────────────────────

def build_point_cloud(
    pointmaps: torch.Tensor,
    conf_mask: torch.Tensor,
    colors: torch.Tensor,
    poses: List[Dict[str, torch.Tensor]],
) -> o3d.geometry.PointCloud:
    """
    Assemble an Open3D PointCloud for a single batch element.

    Args:
        pointmaps:  (V, H*W, 3) float tensor, camera-frame 3D points.
        conf_mask:  (V, H*W) bool tensor; True pixels are included.
        colors:     (V, H*W, 3) float32 [0, 1] tensor.
        poses:      list of V dicts {'R': (3,3), 't': (3,)}.

    Returns:
        o3d.geometry.PointCloud with .points and .colors set.
    """
    V = pointmaps.shape[0]

    pmaps_list = [pointmaps[v] for v in range(V)]          # list of (H*W, 3)
    masks_list = [conf_mask[v] for v in range(V)]          # list of (H*W,)

    # reconstruct_pointcloud reshapes each pmap to (-1, 3) internally
    global_pts = reconstruct_pointcloud(pmaps_list, poses, masks_list)  # (N, 3)

    # Gather colors in the same view-major order
    color_list = [colors[v][conf_mask[v]] for v in range(V)]  # list of (n_v, 3)
    all_colors = torch.cat(color_list, dim=0)                  # (N, 3)

    pts_np = global_pts.cpu().detach().float().numpy()
    col_np = all_colors.cpu().detach().float().numpy()

    # Remove non-finite points (can occur with random-init models)
    valid = np.isfinite(pts_np).all(axis=1)
    pts_np = pts_np[valid]
    col_np = col_np[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(col_np, 0.0, 1.0).astype(np.float64))
    return pcd


# ─────────────────────────────────────────────
# 5. Camera frustum visualization
# ─────────────────────────────────────────────

def build_camera_frustum(
    R: np.ndarray,
    t: np.ndarray,
    focal_length: float,
    img_hw: Tuple[int, int],
    scale: float = 0.1,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> o3d.geometry.LineSet:
    """
    Build a LineSet camera-frustum pyramid (apex + 4 projected image corners).

    Args:
        R:            (3, 3) numpy array, cam→global rotation.
        t:            (3,) numpy array, camera center in global frame.
        focal_length: Estimated focal length in pixels.
        img_hw:       (H, W) image resolution.
        scale:        World-space depth of the frustum (controls visual size).
        color:        (r, g, b) float [0, 1] for all edges.

    Returns:
        o3d.geometry.LineSet with 8 edges.
    """
    H, W = img_hw
    cx, cy = W / 2.0, H / 2.0
    f = float(focal_length) if focal_length > 0 else float(W)

    # 4 image corners in camera frame, projected at depth=scale
    corners_cam = np.array([
        [(0 - cx) / f,  (0 - cy) / f,  1.0],
        [(W - cx) / f,  (0 - cy) / f,  1.0],
        [(W - cx) / f,  (H - cy) / f,  1.0],
        [(0 - cx) / f,  (H - cy) / f,  1.0],
    ])
    # Normalize and scale to desired world depth
    norms = np.linalg.norm(corners_cam, axis=1, keepdims=True)
    corners_cam = corners_cam / norms * scale

    # Transform to global frame: p_global = R @ p_cam + t
    corners_global = (R @ corners_cam.T).T + t  # (4, 3)

    # 5 points: apex (camera center) + 4 corners
    apex = t.reshape(1, 3)
    points = np.vstack([apex, corners_global])  # (5, 3)

    # 8 edges: 4 from apex to each corner + 4 forming the base rectangle
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # apex → corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # base rectangle
    ]
    colors_list = [list(color)] * 8

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(colors_list)
    return lineset


def build_all_frustums(
    poses: List[Dict[str, torch.Tensor]],
    cluster_labels: Optional[np.ndarray],
    focal_lengths: Optional[List[float]],
    img_hw: Tuple[int, int],
    frustum_scale: float = 0.1,
) -> List[o3d.geometry.LineSet]:
    """
    Build one frustum LineSet per camera view.

    Color is determined by cluster_labels (if provided) or by view index,
    using the tab20 colormap.

    Args:
        poses:          list of V dicts {'R', 't'} (tensors, any device).
        cluster_labels: (V,) int array or None.
        focal_lengths:  list of V focal lengths, or None (uses img_hw[1]).
        img_hw:         (H, W) image resolution.
        frustum_scale:  see build_camera_frustum.

    Returns:
        list of V o3d.geometry.LineSet.
    """
    frustums = []
    V = len(poses)
    for v, pose in enumerate(poses):
        label = int(cluster_labels[v]) if cluster_labels is not None else v
        color = _tab20_color(label)
        fl = focal_lengths[v] if focal_lengths is not None else float(img_hw[1])
        R_np = pose["R"].cpu().detach().float().numpy()
        t_np = pose["t"].cpu().detach().float().numpy()
        frustums.append(
            build_camera_frustum(R_np, t_np, fl, img_hw, frustum_scale, color)
        )
    return frustums


# ─────────────────────────────────────────────
# 6. Ray visualization
# ─────────────────────────────────────────────

def build_ray_visualization(
    rig_raymap: torch.Tensor,
    n_samples: int = 64,
    ray_length: float = 0.5,
    color: Tuple[float, float, float] = (0.0, 1.0, 0.5),
) -> o3d.geometry.LineSet:
    """
    Draw uniformly sampled rig_raymap rays as line segments.

    Args:
        rig_raymap: (V, P, 6) tensor for one batch element.
                    Each entry is (ox, oy, oz, dx, dy, dz) in rig frame.
        n_samples:  Rays to draw per view (uniformly sampled from P patches).
        ray_length: Length of each segment in world units.
        color:      (r, g, b) for all segments.

    Returns:
        o3d.geometry.LineSet where each segment is one ray.
    """
    V, P, _ = rig_raymap.shape
    indices = torch.linspace(0, P - 1, min(n_samples, P)).long()

    all_starts = []
    all_ends = []

    for v in range(V):
        sampled = rig_raymap[v][indices]  # (n_samples, 6)
        origins = sampled[:, :3].cpu().detach().float().numpy()
        dirs = sampled[:, 3:].cpu().detach().float().numpy()
        # Normalize directions for consistent segment length
        norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
        dirs = dirs / norms
        all_starts.append(origins)
        all_ends.append(origins + dirs * ray_length)

    starts = np.vstack(all_starts)  # (V*n_samples, 3)
    ends = np.vstack(all_ends)       # (V*n_samples, 3)

    n_rays = len(starts)
    points = np.vstack([starts, ends])  # (2*n_rays, 3)
    lines = [[i, i + n_rays] for i in range(n_rays)]
    colors_list = [list(color)] * n_rays

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(colors_list)
    return lineset


# ─────────────────────────────────────────────
# 7. Export
# ─────────────────────────────────────────────

def export_point_cloud(pcd: o3d.geometry.PointCloud, path: str) -> None:
    """
    Write a point cloud to disk in PLY or PCD format.

    The format is determined by the file extension (.ply or .pcd).
    Parent directories are created automatically.

    Args:
        pcd:  Open3D PointCloud.
        path: Output file path.
    """
    out = Path(path)
    ext = out.suffix.lower()
    if ext not in {".ply", ".pcd"}:
        raise ValueError(f"Unsupported extension '{ext}'. Use .ply or .pcd.")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ascii = ext == ".ply"
    ok = o3d.io.write_point_cloud(str(out), pcd, write_ascii=write_ascii)
    if not ok:
        raise RuntimeError(f"open3d failed to write point cloud to {out}")
    print(f"Exported point cloud ({len(pcd.points)} points) → {out}")


# ─────────────────────────────────────────────
# 8. Top-level pipeline
# ─────────────────────────────────────────────

def visualize_outputs(
    outputs: Dict[str, torch.Tensor],
    images: torch.Tensor,
    cam2rig: Optional[torch.Tensor] = None,
    batch_idx: int = 0,
    conf_percentile: float = 80.0,
    color_mode: str = "rgb",
    cmap_name: str = "plasma",
    show_frustums: bool = True,
    show_rays: bool = False,
    n_ray_samples: int = 64,
    ray_length: float = 0.5,
    frustum_scale: float = 0.1,
    n_clusters: Optional[int] = None,
    export_path: Optional[str] = None,
    show: bool = True,
) -> Dict[str, object]:
    """
    Full pipeline: raw Rig3R outputs → Open3D geometries.

    Optionally exports the point cloud and/or opens an interactive viewer.

    Args:
        outputs:         dict with keys 'pointmap' (B,V,H*W,3),
                         'pointmap_conf' (B,V,H*W,1), 'rig_raymap' (B,V,P,6).
        images:          (B, V, 3, H, W) float32 [0, 1] source images.
        cam2rig:         (B, V, 4, 4) SE(3) matrices or None.
        batch_idx:       Which element in the batch to visualize.
        conf_percentile: Confidence threshold percentile (0–100).
        color_mode:      'rgb' | 'per-camera' | 'rig-cluster' | 'confidence'.
        cmap_name:       matplotlib cmap name for 'confidence' mode.
        show_frustums:   Whether to include camera frustum LineSet objects.
        show_rays:       Whether to include rig_raymap ray segments.
        n_ray_samples:   Rays per view when show_rays is True.
        ray_length:      Ray segment length in world units.
        frustum_scale:   Frustum pyramid depth in world units.
        n_clusters:      Number of rig clusters for 'rig-cluster' coloring.
                         None defaults to 1 (all same cluster).
        export_path:     If set, write point cloud to this path (.ply/.pcd).
        show:            If True, open interactive Open3D viewer (blocking).

    Returns:
        dict with keys:
            'pcd':      o3d.geometry.PointCloud
            'frustums': list of o3d.geometry.LineSet
            'rays':     o3d.geometry.LineSet or None
            'poses':    list of pose dicts
            'labels':   cluster label array or None
    """
    b = batch_idx
    pointmaps = outputs["pointmap"][b]          # (V, H*W, 3)
    conf = outputs["pointmap_conf"][b]          # (V, H*W, 1)
    rig_raymap = outputs["rig_raymap"][b]       # (V, P, 6)
    imgs_b = images[b]                          # (V, 3, H, W)
    cam2rig_b = cam2rig[b] if cam2rig is not None else None

    V = pointmaps.shape[0]
    _, H, W = imgs_b.shape[0], imgs_b.shape[2], imgs_b.shape[3]

    # Step 1: Resolve poses
    poses = resolve_poses(rig_raymap, cam2rig_b)

    # Step 2: Cluster (needed for 'rig-cluster' coloring and frustum colors)
    cluster_labels = None
    if n_clusters is not None and n_clusters > 1:
        cluster_labels = cluster_rig_poses(poses, n_clusters=n_clusters)
    elif color_mode == "rig-cluster" and n_clusters is None:
        print(
            "[visualize] Warning: --color-mode rig-cluster with n_clusters=None "
            "defaults to 1 cluster (all same color). Pass --n-clusters N for "
            "meaningful rig-structure coloring."
        )

    # Step 3: Confidence filtering
    # conf shape is (V, H*W, 1); unsqueeze batch dim for filter_by_confidence
    conf_b = conf.squeeze(-1)  # (V, H*W)
    conf_batch = conf_b.unsqueeze(0)  # (1, V, H*W) — treat as B=1
    mask_batch = filter_by_confidence(conf_batch, percentile=conf_percentile)
    conf_mask = mask_batch[0]  # (V, H*W)

    n_total = V * pointmaps.shape[1]
    n_kept = conf_mask.sum().item()
    print(
        f"[visualize] Confidence filter ({conf_percentile:.0f}th percentile): "
        f"{int(n_kept)}/{n_total} pixels kept "
        f"({100.0 * n_kept / max(n_total, 1):.1f}%)"
    )

    # Step 4: Colors
    focal_lengths = [
        pose.get("focal_length", float(W)) for pose in poses
    ]
    # focal_length may not be present if pose came from cam2rig
    focal_lengths = [
        fl if isinstance(fl, (int, float)) and fl > 0 else float(W)
        for fl in focal_lengths
    ]

    imgs_batch = imgs_b.unsqueeze(0)           # (1, V, 3, H, W)
    conf_batch_full = conf_b.unsqueeze(0)      # (1, V, H*W)
    colors_batch = compute_colors(
        imgs_batch, conf_batch_full, cluster_labels, mode=color_mode, cmap_name=cmap_name
    )
    colors_v = colors_batch[0]  # (V, H*W, 3)

    # Step 5: Build point cloud
    pcd = build_point_cloud(pointmaps, conf_mask, colors_v, poses)
    print(
        f"[visualize] Point cloud: {len(pcd.points)} points, "
        f"{V} views, color_mode='{color_mode}'"
    )

    # Step 6a: Camera frustums
    frustums = []
    if show_frustums:
        frustums = build_all_frustums(
            poses, cluster_labels, focal_lengths, (H, W), frustum_scale
        )

    # Step 6b: Ray visualization
    rays = None
    if show_rays:
        rays = build_ray_visualization(
            rig_raymap, n_samples=n_ray_samples, ray_length=ray_length
        )

    # Step 7: Export
    if export_path is not None:
        export_point_cloud(pcd, export_path)

    # Step 8: Interactive viewer
    if show:
        geometries = [pcd] + frustums
        if rays is not None:
            geometries.append(rays)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        geometries.append(coord_frame)
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Rig3R Visualization",
            width=1280,
            height=720,
        )

    return {
        "pcd": pcd,
        "frustums": frustums,
        "rays": rays,
        "poses": poses,
        "labels": cluster_labels,
    }
