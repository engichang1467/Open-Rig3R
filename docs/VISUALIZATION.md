# Open3D Visualization Pipeline

This document explains how to use the Open3D visualization pipeline to inspect Rig3R predictions interactively. It covers installation, basic usage, all available options, and a description of how each stage of the pipeline works.

---

## Installing Open3D on WSL2

The interactive viewer requires a display. Follow these steps to set up X11 forwarding on WSL2 before running `scripts/visualize.py` without `--no-show`.

1. Install [VcXsrv Windows X Server (XLaunch)](https://sourceforge.net/projects/vcxsrv/)

2. Configure X11 via XLaunch:

    **Screen 1: Multiple windows** → Next
    ```
    Display settings:  -multiwindow  (default)
    Display number:  0
    ```

    **Screen 2: Uncheck Native OpenGL** ← **CRITICAL**
    ```
    [ ]  Native opengl   ← UNCHECK THIS (causes silent hang)
    [✔] Start no client  ← Keep checked
    ```

    **Screen 3: Check Disable Access Control** ← **CRITICAL**
    ```
    [✔] Disable access control  ← CHECK THIS
    [✔] Clipboard               ← Optional
    [✔] Primary Selection       ← Optional
    [✔] Extra Settings: add -wgl
    ```

    **Screen 4: Save Configuration** → Next → Finish

3. Windows Firewall — when VcXsrv prompts, click **"Allow access"** and select **Private AND Public networks**.

4. Add these lines to `~/.bashrc`:

    ```bash
    export XDG_SESSION_TYPE=x11
    export DISPLAY=:0
    export LIBGL_ALWAYS_INDIRECT=0
    ```

    Then reload your shell:

    ```bash
    source ~/.bashrc
    ```

5. Install X11 and OpenGL utilities:

    ```bash
    sudo apt-get install x11-apps mesa-utils
    ```

6. Install the Open3D viewer:

    ```bash
    wget https://github.com/isl-org/Open3D/releases/download/v0.19.0/open3d-viewer-0.19.0-Linux.deb
    sudo mv ~/open3d-viewer-0.19.0-Linux.deb /tmp/
    cd /tmp
    sudo apt install -f ./open3d-viewer-0.19.0-Linux.deb
    ```

7. Verify the setup:

    ```bash
    echo $DISPLAY  # Should show: :0
    xeyes          # Eyes should appear immediately
    glxgears       # Gears should spin
    wget https://raw.githubusercontent.com/McNopper/OpenGL/master/Binaries/teapot.obj
    Open3D teapot.obj  # Teapot viewer should pop up
    ```

---

## Requirements

Open3D and the other visualization dependencies are already listed in `requirements.txt`:

```
open3d >= 0.19.0
plyfile >= 1.1.3
matplotlib >= 3.10.7
```

Install them with:

```bash
pip install -r requirements.txt
```

> **WSL / Headless note:** The interactive Open3D viewer requires a display. On WSL without an X server, skip the viewer with `--no-show` and use `--export` to save the result to a file instead.

---

## Quick Start

```bash
# Interactive viewer — RGB colors, default settings
python scripts/visualize.py --config configs/evaluate.yaml --device cpu

# Headless export — useful on servers or WSL
python scripts/visualize.py \
    --config configs/evaluate.yaml \
    --export outputs/scene000.ply \
    --no-show \
    --device cpu
```

The config file must contain at minimum:

```yaml
checkpoint: "checkpoints/rig3r_epoch50.pt"
data: "data/wayve_scenes_101/scene_001"
```

---

## Usage

```
python scripts/visualize.py --config <path> [options]
```

### Data and model options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--config` | required | Path to YAML config with `checkpoint` and `data` keys |
| `--checkpoint` | from config | Override the checkpoint path |
| `--data` | from config | Override the data root path |
| `--seq-idx` | `0` | Index of the dataset sequence to run inference on |
| `--n-frames` | `2` | Number of views to load per sequence |
| `--image-size H W` | `128 128` | Inference resolution — must match the training resolution |

### Confidence filtering

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--conf-percentile` | `80.0` | Keep pixels at or above this global percentile (0 = keep all) |

### Point cloud coloring

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--color-mode` | `rgb` | `rgb`, `per-camera`, `rig-cluster`, or `confidence` |
| `--cmap` | `plasma` | matplotlib colormap name, used with `--color-mode confidence` |
| `--n-clusters` | `None` | Number of rig clusters for `rig-cluster` coloring |

### Camera frustums

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--no-frustums` | — | Disable camera frustum overlay |
| `--frustum-scale` | `0.1` | Frustum pyramid depth in world units |

### Ray overlay

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--rays` | — | Enable rig_raymap ray segment overlay |
| `--n-ray-samples` | `64` | Rays drawn per view |
| `--ray-length` | `0.5` | Length of each ray segment in world units |

### Export and display

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--export` | — | Save point cloud to this path (`.ply` or `.pcd`) |
| `--no-show` | — | Skip the interactive viewer |
| `--batch-idx` | `0` | Which element within the batch to visualize |

---

## Examples

### Inspect rig structure with cluster coloring

```bash
python scripts/visualize.py \
    --config configs/evaluate.yaml \
    --color-mode rig-cluster \
    --n-clusters 3 \
    --seq-idx 1 \
    --device cpu
```

Cameras belonging to the same discovered rig cluster are colored identically. Camera frustums follow the same color scheme, making it easy to see which physical cameras were grouped together.

### Inspect confidence — keep only high-quality points

```bash
python scripts/visualize.py \
    --config configs/evaluate.yaml \
    --color-mode confidence \
    --cmap plasma \
    --conf-percentile 90.0 \
    --device cpu
```

Points are colored from low (dark) to high (bright) confidence. Raising `--conf-percentile` keeps fewer, higher-quality points.

### Export for external inspection

```bash
python scripts/visualize.py \
    --config configs/evaluate.yaml \
    --n-frames 5 \
    --export outputs/scene001.ply \
    --no-show \
    --device cpu
```

The exported PLY file can be opened in MeshLab, CloudCompare, or any other tool that reads point clouds.

### Overlay predicted rays for debugging

```bash
python scripts/visualize.py \
    --config configs/evaluate.yaml \
    --rays \
    --n-ray-samples 128 \
    --ray-length 0.3 \
    --no-frustums \
    --device cpu
```

Each green segment starts at the predicted ray origin in rig space and extends along the predicted ray direction. This is useful for verifying that the rig raymap head is outputting consistent ray geometry.

---

## How It Works

The pipeline converts raw model outputs into Open3D geometry objects through a sequence of steps. Each step corresponds to a function in `utils/visualization.py`.

### Step 1 — Inference

`scripts/visualize.py` loads the model and dataset, then runs a forward pass under `torch.no_grad()`. The `cam2rig` extrinsics are withheld from the model so that it exercises the rig-discovery path (the same way it is evaluated).

```
outputs = model(images, metadata_without_cam2rig)
```

The model returns:

| Key | Shape | Description |
| :--- | :--- | :--- |
| `pointmap` | `(B, V, H×W, 3)` | Dense 3D point predictions in camera frame |
| `pointmap_conf` | `(B, V, H×W, 1)` | Per-pixel confidence (unbounded raw floats) |
| `pose_raymap` | `(B, V, P, 3)` | Unit ray directions in camera frame (patch-level) |
| `rig_raymap` | `(B, V, P, 6)` | Ray origins and directions in rig frame (patch-level) |

`P = (image_size / patch_size)²`. For the default 128×128 image with patch size 8, `P = 256`.

---

### Step 2 — Pose resolution

For each view, the pipeline decides how to obtain a camera pose `{R, t}`:

1. **If `cam2rig[v]` is not identity** — the provided SE(3) matrix is used directly:
   - `R = cam2rig[v, :3, :3]`
   - `t = cam2rig[v, :3, 3]`

2. **If `cam2rig[v]` is identity** (dropped by metadata dropout, or not provided) — the pose is recovered from the predicted rig raymap using `recover_pose_closed_form` from `utils/rig_discovery.py`. This function reshapes the `(P, 6)` raymap to `(H_patch, W_patch, 6)` and solves for `R` and `t` via SVD.

This handles mixed cases: sequences where some views have real extrinsics and others do not.

---

### Step 3 — Confidence filtering

The `PointMapHead` outputs a confidence channel with no activation function, so values are raw floats with no guaranteed range. A **global percentile threshold** is computed across all pixels in the loaded batch:

```
threshold = percentile(all_conf_values, conf_percentile)
mask = conf >= threshold
```

`--conf-percentile 80` keeps the top 20% most confident pixels. Setting it to `0` keeps every pixel.

---

### Step 4 — Color computation

Colors are assigned per-pixel before masking, then sampled by the mask in step 5.

| Mode | How color is determined |
| :--- | :--- |
| `rgb` | Each pixel's color is read directly from the source image. No resampling is needed because the `pointmap` and `images` tensors share the same `H×W` resolution. |
| `per-camera` | Each view receives a distinct solid color from the `tab20` colormap, indexed by view number. |
| `rig-cluster` | Cameras are clustered by pose using K-means (`cluster_rig_poses`). All views in the same cluster share a `tab20` color, making rig structure immediately visible. |
| `confidence` | Confidence values are normalized per-view (min-max within each view) and mapped through a matplotlib colormap. Per-view normalization is used so local structure is visible even when absolute confidence varies across views. |

---

### Step 5 — Point cloud assembly

The confidence mask is applied and the surviving camera-frame points are transformed to the global rig frame:

```
p_global = R @ p_camera + t
```

This is handled by `reconstruct_pointcloud` from `utils/rig_discovery.py`. The results from all views are concatenated into a single `(N, 3)` array, matched with the corresponding `(N, 3)` colors, and passed to an `o3d.geometry.PointCloud`.

Non-finite values (`NaN`, `Inf`) are removed before the Open3D objects are created. These can appear when using a randomly initialized or early-checkpoint model.

---

### Step 6 — Camera frustums

Each camera is represented as a pyramid `LineSet` with five vertices:

- **Apex** — the camera center `t` in global space.
- **Four corners** — the image corners, projected at depth `frustum_scale` into the scene and transformed to global space via `R`.

The corner directions are computed from the estimated focal length and image dimensions:

```
corner_cam = [(±W/2) / f,  (±H/2) / f,  1.0]  (normalized, then scaled)
corner_global = R @ corner_cam + t
```

Frustum colors follow the same palette as the point cloud: `tab20` indexed by cluster label (if clustering is enabled) or by view index otherwise.

---

### Step 7 — Ray visualization (optional)

When `--rays` is passed, `n_ray_samples` rays are uniformly sampled from each view's `rig_raymap`. Each ray is drawn as a line segment:

```
start = (ox, oy, oz)
end   = (ox + dx * ray_length,  oy + dy * ray_length,  oz + dz * ray_length)
```

Ray origins and directions come directly from the predicted `rig_raymap` — no transformation is applied since they are already in the global rig frame.

---

### Step 8 — Display and export

All geometry objects are passed to `o3d.visualization.draw_geometries`, which opens an interactive viewer. A coordinate frame axes indicator (`size=0.3`) is added to the scene to help with orientation but is not exported.

For export, `o3d.io.write_point_cloud` is used. The format is determined by the file extension:

- `.ply` — ASCII PLY (human-readable, larger file)
- `.pcd` — binary PCD (compact, fast to load)

---

## Coloring Mode Reference

| Mode | Best for |
| :--- | :--- |
| `rgb` | General scene inspection — shows the actual appearance of each point |
| `per-camera` | Checking coverage — seeing which camera contributes which region |
| `rig-cluster` | Rig discovery — verifying that cameras are grouped into the correct physical rigs |
| `confidence` | Quality inspection — identifying which regions the model is uncertain about |

---

## Programmatic Usage

The pipeline can be called directly from Python without going through the CLI:

```python
import torch
from utils.visualization import visualize_outputs

# outputs: dict from model forward pass
# images:  (B, V, 3, H, W) float32 tensor
result = visualize_outputs(
    outputs=outputs,
    images=images,
    cam2rig=metadata.get("cam2rig"),   # optional
    conf_percentile=80.0,
    color_mode="rig-cluster",
    n_clusters=3,
    export_path="outputs/scene.ply",
    show=False,                         # headless
)

pcd      = result["pcd"]       # o3d.geometry.PointCloud
frustums = result["frustums"]  # list of o3d.geometry.LineSet
poses    = result["poses"]     # list of {'R': tensor, 't': tensor}
labels   = result["labels"]    # cluster label array or None
```

Individual primitives can also be used independently:

```python
from utils.visualization import (
    filter_by_confidence,
    build_camera_frustum,
    export_point_cloud,
)
```

---

## Related Files

| File | Description |
| :--- | :--- |
| [utils/visualization.py](../utils/visualization.py) | All visualization primitives |
| [scripts/visualize.py](../scripts/visualize.py) | CLI entrypoint |
| [tests/test_visualization.py](../tests/test_visualization.py) | 23 unit and integration tests |
| [utils/rig_discovery.py](../utils/rig_discovery.py) | Pose recovery and point cloud reconstruction (called by the pipeline) |
