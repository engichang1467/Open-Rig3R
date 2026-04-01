"""
CLI entrypoint for the Rig3R Open3D visualization pipeline.

Loads a trained Rig3R model, runs inference on a single dataset sequence,
and visualizes the resulting point cloud with optional camera frustums and
rig-raymap ray overlays.

Usage:
    python scripts/visualize.py --config configs/evaluate.yaml [options]

Example (headless export):
    python scripts/visualize.py \\
        --config configs/evaluate.yaml \\
        --color-mode rig-cluster --n-clusters 3 \\
        --export runs/viz/scene001.ply \\
        --no-show
"""

import open3d  # noqa: F401 — must be imported before torch to avoid libstdc++ conflict
import argparse
import sys
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# Allow imports from project root
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from models.rig3r import Rig3R
from datasets.wayve101 import Wayve101Dataset
from utils.visualization import visualize_outputs


# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Rig3R predictions with Open3D",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data / model
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (must contain 'checkpoint' and 'data' keys).",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Override the checkpoint path from the config file.",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Override the data root path from the config file.",
    )
    parser.add_argument(
        "--seq-idx", type=int, default=0,
        help="Index of the dataset sequence to run inference on.",
    )
    parser.add_argument(
        "--n-frames", type=int, default=2,
        help="Number of frames / views to load per sequence.",
    )
    parser.add_argument(
        "--image-size", type=int, nargs=2, default=[128, 128],
        metavar=("H", "W"),
        help="Image resolution for inference (must match training resolution).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for inference: 'cpu' or 'cuda'. "
             "Defaults to cuda if available. "
             "Use --device cpu on WSL2 to avoid libdxcore/open3d conflicts.",
    )

    # Confidence
    parser.add_argument(
        "--conf-percentile", type=float, default=80.0,
        help="Keep pixels at or above this confidence percentile (0=keep all).",
    )

    # Coloring
    parser.add_argument(
        "--color-mode",
        choices=["rgb", "per-camera", "rig-cluster", "confidence"],
        default="rgb",
        help="Point cloud coloring mode.",
    )
    parser.add_argument(
        "--cmap", type=str, default="plasma",
        help="matplotlib colormap name (used with --color-mode confidence).",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=None,
        help="Number of rig clusters for 'rig-cluster' coloring. "
             "None defaults to 1 (no clustering).",
    )

    # Frustums
    parser.add_argument(
        "--no-frustums", dest="show_frustums", action="store_false",
        help="Disable camera frustum visualization.",
    )
    parser.add_argument(
        "--frustum-scale", type=float, default=0.1,
        help="World-space depth of camera frustum pyramids.",
    )

    # Rays
    parser.add_argument(
        "--rays", dest="show_rays", action="store_true",
        help="Overlay sampled rig_raymap ray segments.",
    )
    parser.add_argument(
        "--n-ray-samples", type=int, default=64,
        help="Number of rays to draw per view (requires --rays).",
    )
    parser.add_argument(
        "--ray-length", type=float, default=0.5,
        help="World-space length of each ray segment.",
    )

    # Export / display
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export point cloud to this path (.ply or .pcd).",
    )
    parser.add_argument(
        "--no-show", dest="show", action="store_false",
        help="Skip the interactive Open3D viewer (useful on headless / WSL).",
    )

    # batch element (rarely needed, but available)
    parser.add_argument(
        "--batch-idx", type=int, default=0,
        help="Which element within the loaded batch to visualize.",
    )

    parser.set_defaults(show_frustums=True, show_rays=False, show=True)
    return parser.parse_args()


# ─────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[visualize] Device: {device}")

    # Resolve paths (CLI flags override config)
    data_root = Path(args.data or cfg["data"])
    ckpt_path = Path(args.checkpoint or cfg["checkpoint"])
    image_size = tuple(args.image_size)  # (H, W)

    # ── Dataset ──────────────────────────────
    dataset = Wayve101Dataset(
        root_dir=data_root,
        n_frames=args.n_frames,
        image_size=image_size,
        transforms=None,
        use_masks=False,
    )
    print(f"[visualize] Dataset: {len(dataset)} sequences  |  loading seq_idx={args.seq_idx}")

    if args.seq_idx >= len(dataset):
        raise IndexError(
            f"--seq-idx {args.seq_idx} is out of range for dataset of size {len(dataset)}"
        )

    # Wrap single sample in a batch of 1
    sample = dataset[args.seq_idx]
    images = sample["images"].unsqueeze(0).to(device)          # (1, V, 3, H, W)
    metadata = {
        k: v.unsqueeze(0).to(device)
        for k, v in sample["metadata"].items()
        if isinstance(v, torch.Tensor)
    }

    # ── Model ────────────────────────────────
    model = Rig3R(
        img_size=image_size[0],
        patch_size=8,
        embed_dim=128,
        metadata_dim=128,
        num_decoder_layers=2,
        num_heads=2,
        mlp_dim=128 * 4,
    )

    try:
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"[visualize] Loaded checkpoint: {ckpt_path}")
    except Exception as e:
        print(f"[visualize] Warning: could not load checkpoint ({e})")
        print("[visualize] Proceeding with randomly initialized model.")

    model.to(device)
    model.eval()

    # ── Inference ────────────────────────────
    # Strip cam2rig from model input so we exercise the rig-discovery path.
    # The gt cam2rig is kept separately for pose resolution fallback.
    model_metadata = {k: v for k, v in metadata.items() if k != "cam2rig"}
    cam2rig = metadata.get("cam2rig")  # (1, V, 4, 4) or None

    with torch.no_grad():
        outputs = model(images, model_metadata)

    print(
        f"[visualize] Inference done  |  "
        f"pointmap {tuple(outputs['pointmap'].shape)}  |  "
        f"rig_raymap {tuple(outputs['rig_raymap'].shape)}"
    )

    # ── Visualize ────────────────────────────
    result = visualize_outputs(
        outputs=outputs,
        images=images,
        cam2rig=cam2rig,
        batch_idx=args.batch_idx,
        conf_percentile=args.conf_percentile,
        color_mode=args.color_mode,
        cmap_name=args.cmap,
        show_frustums=args.show_frustums,
        show_rays=args.show_rays,
        n_ray_samples=args.n_ray_samples,
        ray_length=args.ray_length,
        frustum_scale=args.frustum_scale,
        n_clusters=args.n_clusters,
        export_path=args.export,
        show=args.show,
    )

    # ── Summary ──────────────────────────────
    labels = result["labels"]
    if labels is not None:
        unique, counts = zip(*sorted(
            {int(l): int((labels == l).sum()) for l in set(labels)}.items()
        ))
        cluster_summary = "  ".join(
            f"cluster {c}: {n} views" for c, n in zip(unique, counts)
        )
        print(f"[visualize] Rig clusters: {cluster_summary}")
    else:
        print(f"[visualize] No clustering (use --n-clusters N for rig-structure coloring)")

    print(f"[visualize] Done.")


if __name__ == "__main__":
    main()
