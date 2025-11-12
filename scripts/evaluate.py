import os
import torch
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data_root = "data/wayve_scenes_101"
data_root = Path.cwd().joinpath("data/wayve_scenes_101/scene_001")
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
# Replace with your trained checkpoint and model config
# model_ckpt = "checkpoints/rig3r_epoch50.pt"
model_ckpt = Path.cwd().joinpath("checkpoints/rig3r_epoch50.pt")
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

    # with torch.no_grad():
    #     outputs = model(images, metadata)

    #     # Convert model outputs to 3D points
    #     # rig_raymap: (B, V, P, 3) = rig-relative ray directions
    #     rig_rays = outputs['rig_raymap']                   # (B, V, P, 3)
    #     B, V, P, _ = rig_rays.shape

    #     # Option A: use unit rays as points
    #     pred_pc = rig_rays.reshape(B, V * P, 3)           # (B, N_pred, 3)

    #     # Option B: scale rays by predicted depth (if available)
    #     # if 'pointmap' in outputs:
    #     #     depths = outputs['pointmap'].reshape(B, V * P, 1)
    #     #     pred_pc = rig_rays.reshape(B, V * P, 3) * depths

    # for b in range(B):
    #     pc_pred_b = pred_pc[b]                             # (N_pred, 3)
    #     pc_gt_b = gt_pc[b].reshape(-1, 3)                  # (N_gt, 3)

    #     # debug prints
    #     # print(f"[DEBUG] pred_pc shape: {pc_pred_b.shape}, gt_pc shape: {pc_gt_b.shape}")
    #     # print(f"[DEBUG] pred_pc mean: {pc_pred_b.mean().item():.4f}, std: {pc_pred_b.std().item():.4f}")
    #     # print(f"[DEBUG] gt_pc mean: {pc_gt_b.mean().item():.4f}, std: {pc_gt_b.std().item():.4f}")

    #     # compute Chamfer Distance for this sample
    #     cd = chamfer_distance(pc_pred_b, pc_gt_b)
    #     all_chamfer.append(cd.item())



    with torch.no_grad():
        outputs = model(images, metadata)
        pred_pc = outputs.get('pointcloud_pred', torch.empty(0,3, device=device))

    # print(f"[DEBUG] pred_pc shape: {pred_pc.shape}, "
    #   f"gt_pc shape: {gt_pc.shape}")
    # print(f"[DEBUG] pred_pc mean: {pred_pc.mean().item():.4f}, "
    #     f"std: {pred_pc.std().item():.4f}")
    # print(f"[DEBUG] gt_pc mean: {gt_pc.mean().item():.4f}, "
    #     f"std: {gt_pc.std().item():.4f}")

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