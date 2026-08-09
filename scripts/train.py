import os
import yaml
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.amp import autocast

import sys

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

# --- Import datasets and transforms ---
from datasets.co3d import Co3DDataset
from datasets.waymo import WaymoDataset
from datasets.transform import get_train_transforms, get_val_transforms

# --- Import model ---
from models.rig3r import Rig3R
from models.losses import MultiTaskLoss
from utils.raymap import build_pointmap_target, build_raymap_targets

# --- Optional: logging ---
import wandb

# -----------------------------
# 1. Load configs
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Rig3R model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (YAML)")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

args = parse_args()
train_cfg = load_config(args.config)

device = torch.device(train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
dataset_type = train_cfg.get("dataset_type", "co3d")

# -----------------------------
# 2. Prepare datasets
# -----------------------------
img_size = tuple(train_cfg.get("image_size", [128, 128]))
patch_size = train_cfg.get("patch_size", 8)

if dataset_type == "co3d":
    co3d_path = Path.cwd().joinpath("data/co3d")
    
    train_dataset = Co3DDataset(
        root_dir=co3d_path,
        subset="train",
        n_frames=train_cfg["n_frames"],
        image_size=img_size,
        transforms=get_train_transforms(image_size=img_size),
        metadata_dropout=train_cfg.get("metadata_dropout", 0.5)
    )

    val_dataset = Co3DDataset(
        root_dir=co3d_path,
        subset="val",
        n_frames=train_cfg["n_frames"],
        image_size=img_size,
        transforms=get_val_transforms(image_size=img_size),
        metadata_dropout=0.0
    )

elif dataset_type == "waymo":
    waymo_path = Path.cwd().joinpath(train_cfg.get("waymo_path", "/data/waymo_mini"))
    cameras = train_cfg.get("waymo_cameras", None)
    sequence_ids = train_cfg.get("sequence_ids", None)

    train_dataset = WaymoDataset(
        root_dir=waymo_path,
        split="train",
        cameras=cameras,
        sequence_ids=sequence_ids,
        n_frames=train_cfg["n_frames"],
        image_size=img_size,
        transforms=get_train_transforms(image_size=img_size)
    )

    val_dataset = WaymoDataset(
        root_dir=waymo_path,
        split="validation",
        cameras=cameras,
        sequence_ids=sequence_ids,
        n_frames=train_cfg["n_frames"],
        image_size=img_size,
        transforms=get_val_transforms(image_size=img_size)
    )
else:
    raise ValueError(f"Unknown dataset type: {dataset_type}")

train_loader = DataLoader(
    train_dataset,
    batch_size=train_cfg["batch_size"],
    shuffle=True,
    num_workers=train_cfg.get("num_workers", 0)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=train_cfg["batch_size"],
    shuffle=False,
    num_workers=train_cfg.get("num_workers", 0)
)

# -----------------------------
# 3. Initialize model
# -----------------------------

pretrain_ckpt_path = Path.cwd().joinpath("checkpoints/pretrained/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")

model = Rig3R(
    encoder_ckpt=pretrain_ckpt_path,
    img_size=img_size[0],
    patch_size=patch_size,
    embed_dim=1024,
    num_decoder_layers=2,
    num_heads=8,
    mlp_dim=4096
)
        
model.to(device)

# -----------------------------
# 4. Optimizer & scheduler
# -----------------------------

optimizer = optim.AdamW(model.parameters(),
                        lr=float(train_cfg["optimizer"]["lr"]),
                        weight_decay=train_cfg["optimizer"].get("weight_decay", 0.01))

scheduler = CosineAnnealingLR(optimizer,
                              T_max=train_cfg["scheduler"]["T_max"],
                              eta_min=float(train_cfg["scheduler"]["eta_min"]))

# -----------------------------
# 5. Loss function
# -----------------------------
# the pointmap term is a metric MSE in metres and runs an order of magnitude above
# the two raymap terms, which are cosine-based and bounded by 2 - hence the weights
loss_cfg = train_cfg.get("loss", {})
criterion = MultiTaskLoss(
    w_point=loss_cfg.get("w_point", 1.0),
    w_pose=loss_cfg.get("w_pose", 1.0),
    w_rig=loss_cfg.get("w_rig", 1.0),
)


def downsample_pointmap(pointmap, img_size, patch_size):
    """Dense (B, V, H*W, 3) prediction -> (B, V, P, 3) at patch resolution.

    The pointmap head predicts at image resolution, but pointcloud ground truth
    is sparse (P points, P = (img_size / patch_size)^2), so average-pool the
    prediction down to meet it.
    """
    B, V, _, C = pointmap.shape
    H = W = img_size
    patch_grid = img_size // patch_size  # e.g., 128 // 8 = 16

    # (B, V, H*W, 3) -> (B*V, 3, H, W)
    spatial = pointmap.view(B * V, H, W, C).permute(0, 3, 1, 2)
    pooled = nn.functional.avg_pool2d(spatial, kernel_size=patch_size, stride=patch_size)

    # (B*V, 3, patch_grid, patch_grid) -> (B, V, P, 3)
    return pooled.permute(0, 2, 3, 1).reshape(B, V, patch_grid * patch_grid, C)


def compute_loss(outputs, batch, device, img_size, patch_size):
    """Score whatever supervision this batch actually carries.

    Waymo currently supplies raymap targets (from calibration + rig poses) but no
    pointcloud; CO3D supplies a pointcloud but no calibration. MultiTaskLoss skips
    any term missing from either side, so each dataset trains on what it has.
    """
    preds = dict(outputs)
    targets = {}

    pointcloud = batch["pointcloud"].to(device)
    pointmap = batch.get("pointmap", torch.empty(0)).to(device)

    if pointcloud.numel() > 0:
        preds["pointmap"] = downsample_pointmap(preds["pointmap"], img_size[0], patch_size)
        targets["pointmap"] = pointcloud
    elif pointmap.numel() > 0:
        preds["pointmap"] = downsample_pointmap(preds["pointmap"], img_size[0], patch_size)
        targets["pointmap"], targets["pointmap_conf"] = build_pointmap_target(
            pointmap=pointmap,
            cam2rig=batch["metadata"]["cam2rig"].to(device),
            world_from_rig=batch["world_from_rig"].to(device),
            patch_size=patch_size,
            image_size=img_size,
        )

    if "intrinsics" in batch:
        targets.update(build_raymap_targets(
            cam2rig=batch["metadata"]["cam2rig"].to(device),
            intrinsics=batch["intrinsics"].to(device),
            world_from_rig=batch["world_from_rig"].to(device),
            image_size=img_size,
            patch_size=patch_size,
        ))

    total, loss_dict = criterion(preds, targets)

    # with no matching target MultiTaskLoss returns a plain 0.0 float and the
    # optimizer becomes a no-op - the exact failure this pipeline shipped with.
    # grad_fn is only expected under grad; validation runs inside no_grad.
    connected = torch.is_tensor(total) and (
        total.grad_fn is not None or not torch.is_grad_enabled()
    )
    assert connected, (
        f"No supervision in this batch (targets: {sorted(targets)}). "
        f"Export calibration/poses for raymaps, or a pointcloud for pointmaps."
    )
    return total, loss_dict


# -----------------------------
# 6. Batch unpacking
# -----------------------------
def unpack_batch(batch, device):
    """Move a collated sample dict onto the device. Same shape for co3d and waymo."""
    images = batch["images"].to(device)

    metadata = batch["metadata"]
    for key, value in metadata.items():
        if value is not None:
            metadata[key] = value.to(device)

    return images, metadata


# -----------------------------
# 7. Logging setup
# -----------------------------
run = wandb.init(
    project=train_cfg.get("wandb_project", "open-rig3r"),
    entity=train_cfg.get("wandb_entity"),
    name=train_cfg.get("wandb_run_name"),
    mode=train_cfg.get("wandb_mode", "disabled"),  # "offline" / "disabled" for no network
    config={**train_cfg, "config_file": args.config, "device": str(device)},
    dir="runs",
)

# -----------------------------
# 8. Training loop
# -----------------------------
num_epochs = train_cfg.get("epochs", 50)
global_step = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    
    for batch_idx, batch in enumerate(train_bar):
        images, metadata = unpack_batch(batch, device)

        optimizer.zero_grad()
        with autocast(device_type=str(device)):
            outputs = model(images, metadata)
        loss, loss_dict = compute_loss(outputs, batch, device, img_size, patch_size)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        wandb.log(
            {f"train/batch_{name}": value.item() for name, value in loss_dict.items()},
            step=global_step,
        )
        global_step += 1

    scheduler.step()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_loss:.4f}")

    # -----------------------------
    # 9. Validation loop
    # -----------------------------
    model.eval()
    val_loss = 0.0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
    with torch.no_grad():
        for batch in val_bar:
            images, metadata = unpack_batch(batch, device)

            with autocast(device_type=str(device)):
                outputs = model(images, metadata)
            loss, _ = compute_loss(outputs, batch, device, img_size, patch_size)

            val_loss += loss.item()
            val_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Val Loss: {avg_val_loss:.4f}")

    wandb.log(
        {
            "epoch": epoch + 1,
            "train/loss": avg_loss,
            "val/loss": avg_val_loss,
            "lr": scheduler.get_last_lr()[0],
        },
        step=global_step,
    )

    # -----------------------------
    # 10. Save checkpoints
    # -----------------------------
    if (epoch + 1) % 5 == 0:
        ckpt_path = os.path.join("checkpoints", f"rig3r_epoch{epoch+1}.pt")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

        artifact = wandb.Artifact(
            f"rig3r-{run.id}", type="model",
            metadata={"epoch": epoch + 1, "val_loss": avg_val_loss},
        )
        artifact.add_file(ckpt_path)
        run.log_artifact(artifact, aliases=["latest", f"epoch-{epoch+1}"])

run.finish()
print("Training finished!")