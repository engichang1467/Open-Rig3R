import os
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from pathlib import Path
import sys

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from datasets.wayve101 import Wayve101Dataset
from models.encoder_vit import DUST3R_ENCODER
from models.rig3r import Rig3R
from utils.metrics import align_scale, chamfer_distance, rig_discovery_accuracy, rig_maa
from utils.rig_discovery import recover_pose_closed_form, reconstruct_pointcloud

# -----------------------------
# 1. Configuration
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Rig3R model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (YAML)")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


args = parse_args()
eval_cfg = load_config(args.config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = Path.cwd().joinpath(eval_cfg["data"])
n_frames = eval_cfg.get("n_frames", 2)
image_size = tuple(eval_cfg.get("image_size", [128, 128]))
batch_size = 1  # evaluation usually works with 1

# patch_size is not read from config: the DUSt3R ViT-L/16 encoder pins it, so a
# config key would have exactly one legal value. A wrong image_size or patch_size
# cannot go unnoticed either way - both change parameter shapes, so the strict
# load below raises rather than scoring the wrong model.
patch_size = DUST3R_ENCODER["patch_size"]

# -----------------------------
# 2. Load dataset
# -----------------------------
dataset = Wayve101Dataset(root_dir=data_root,
                          n_frames=n_frames,
                          image_size=image_size,
                          transforms=None,
                          use_masks=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

print(f"Loaded {len(dataset)} sequences from Wayve101")

# -----------------------------
# 3. Load trained Rig3R model
# -----------------------------
model_ckpt = Path.cwd().joinpath(eval_cfg["checkpoint"])
model = Rig3R(
    img_size=image_size[0],
    patch_size=patch_size,
    embed_dim=1024,
    num_decoder_layers=2,
    num_heads=8,
    mlp_dim=4096
)
model.load_state_dict(torch.load(model_ckpt, map_location=device))
model.to(device)
model.eval()
print(f"Loaded model from {model_ckpt}")

# -----------------------------
# 4. Evaluation loop
# -----------------------------
all_chamfer = []
all_rig_acc = []
all_rig_maa = []

for batch in tqdm(dataloader, desc="Evaluating Wayve101"):
    images = batch['images'].to(device)            # (B,N,3,H,W)
    metadata = {k: v.to(device) for k, v in batch['metadata'].items() if v is not None}
    gt_pc = batch['pointcloud'].to(device)         # (B,M,3)

    # cam2rig is the ground truth we score against, so the model must not see it.
    model_metadata = {k: v for k, v in metadata.items() if k != 'cam2rig'}

    with torch.no_grad():
        outputs = model(images, model_metadata)

    # The decoder emits pointmaps and raymaps, never a finished point cloud - the
    # cloud is assembled here from the poses recovered out of the rig raymap.
    rig_raymaps = outputs['rig_raymap']            # (B,N,P,6)
    pointmaps = outputs['pointmap']                # (B,N,P,3)
    B, N, P, _ = rig_raymaps.shape
    H_patch = W_patch = int(P ** 0.5)

    for b in range(B):
        poses = [recover_pose_closed_form(rig_raymaps[b, n].reshape(H_patch, W_patch, 6))
                 for n in range(N)]
        pred_pc = reconstruct_pointcloud([pointmaps[b, n] for n in range(N)], poses)

        # Eq. 3 normalizes only the ground truth, so predictions come out at
        # z-bar scale. Both metrics below are in metres, so rescale first.
        scale = align_scale(pred_pc, gt_pc[b])

        cd = chamfer_distance(pred_pc * scale, gt_pc[b])
        all_chamfer.append(cd.item())

        # rig_discovery_accuracy matches rig keypoints - one 3-vector per view -
        # under a 0.1 m threshold. Handing it the dense cloud instead measures
        # nothing the metric is named for, and is cubic in the point count.
        if 'cam2rig' in metadata:
            gt_cam2rig = metadata['cam2rig'][b]

            pred_keypoints = torch.stack([p['t'] for p in poses]) * scale
            gt_keypoints = gt_cam2rig[:, :3, 3]
            all_rig_acc.append(rig_discovery_accuracy(pred_keypoints, gt_keypoints).item())

            # Rotation-only, so no align_scale and no metres: unlike Chamfer, this
            # stays comparable across changes to the scale convention.
            gt_poses = [{'R': gt_cam2rig[n, :3, :3]} for n in range(N)]
            all_rig_maa.append(rig_maa(poses, gt_poses).item())

# -----------------------------
# 5. Report results
# -----------------------------
if not all_chamfer:
    raise RuntimeError("No sequences were scored - nothing to average.")

avg_chamfer = sum(all_chamfer) / len(all_chamfer)

print(f"\nEvaluation finished!")
print(f"Average Chamfer Distance over {len(all_chamfer)} sequences: {avg_chamfer:.6f}")
if all_rig_acc:
    print(f"Average Rig Discovery Accuracy: {sum(all_rig_acc)/len(all_rig_acc):.4f}")
    print(f"Average Rig mAA (deg): {sum(all_rig_maa)/len(all_rig_maa):.4f}")
else:
    print("Rig metrics: not scored (no cam2rig ground truth in batch)")