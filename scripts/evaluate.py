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
from models.rig3r import Rig3R
from utils.metrics import chamfer_distance, rig_discovery_accuracy

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
n_frames = 2
image_size = (128, 128)
batch_size = 1  # evaluation usually works with 1

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
    patch_size=8,
    embed_dim=128,
    metadata_dim=128,
    num_decoder_layers=2,
    num_heads=2,
    mlp_dim=128*4
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

for batch in tqdm(dataloader, desc="Evaluating Wayve101"):
    images = batch['images'].to(device)            # (N,3,H,W)
    metadata = {k: v.to(device) for k, v in batch['metadata'].items() if v is not None}
    gt_pc = batch['pointcloud'].to(device)        # (M,3)

    with torch.no_grad():
        outputs = model(images, metadata)
        pred_pc = outputs.get('pointcloud_pred', torch.empty(0,3, device=device))

    cd = chamfer_distance(pred_pc, gt_pc)
    rig_acc = rig_discovery_accuracy(pred_pc, gt_pc)

    all_chamfer.append(cd.item())
    all_rig_acc.append(rig_acc.item())

# -----------------------------
# 5. Report results
# -----------------------------
avg_chamfer = sum(all_chamfer)/len(all_chamfer)
avg_rig_acc = sum(all_rig_acc)/len(all_rig_acc)

print(f"\nEvaluation finished!")
print(f"Average Chamfer Distance over {len(dataset)} sequences: {avg_chamfer:.6f}")
print(f"Average Rig Discovery Accuracy: {avg_rig_acc:.4f}")