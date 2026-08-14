import os
import random
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
from models.encoder_vit import DUST3R_ENCODER
from models.losses import MultiTaskLoss
from utils.amp import needs_grad_scaler, select_amp_dtype
from utils.metrics import raymap_metrics
from utils.raymap import build_pointmap_target, build_raymap_targets, scene_scale

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

# Without this every run draws a different init and shuffle, so two runs of the same config are not comparable (this is what made the 37a A/B unreadable), and DataLoader workers inherit deterministic per-worker seeds from the torch seed.
seed = train_cfg.get("seed", 0)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device(train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
dataset_type = train_cfg.get("dataset_type", "co3d")

# -----------------------------
# 2. Prepare datasets
# -----------------------------
img_size = tuple(train_cfg.get("image_size", [128, 128]))

# patch_size is pinned to 16 by the DUSt3R ViT-L/16 encoder (which loads strict) and also sets the raymap target resolution, so it is hardcoded rather than read from config, where a missing key silently fell back to 8 and broke loading.
patch_size = DUST3R_ENCODER["patch_size"]
if train_cfg.get("patch_size", patch_size) != patch_size:
    raise ValueError(
        f"config sets patch_size={train_cfg['patch_size']}, but the DUSt3R encoder "
        f"is fixed at {patch_size}. Drop the key rather than have it silently ignored."
    )

if dataset_type == "co3d":
    co3d_path = Path.cwd().joinpath("data/co3d")
    
    train_dataset = Co3DDataset(
        root_dir=co3d_path,
        subset="train",
        n_frames=train_cfg["n_frames"],
        image_size=img_size,
        transforms=get_train_transforms(image_size=img_size)
    )

    val_dataset = Co3DDataset(
        root_dir=co3d_path,
        subset="val",
        n_frames=train_cfg["n_frames"],
        image_size=img_size,
        transforms=get_val_transforms(image_size=img_size)
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
    mlp_dim=4096,
    metadata_dropout=train_cfg.get("metadata_dropout", 0.5)
)
        
model.to(device)

# -----------------------------
# 4. Optimizer & scheduler
# -----------------------------

optimizer = optim.AdamW(model.parameters(),
                        lr=float(train_cfg["optimizer"]["lr"]),
                        weight_decay=train_cfg["optimizer"].get("weight_decay", 0.01))

# autocast defaults to fp16 on CUDA, whose small gradients underflow without a scaler, so prefer bf16 for fp32's exponent range and leave the scaler below a no-op except on pre-Ampere cards that fall back to fp16.
amp_dtype = select_amp_dtype(device, train_cfg.get("amp_dtype"))
scaler = torch.amp.GradScaler(device.type, enabled=needs_grad_scaler(amp_dtype))
print(f"AMP: {amp_dtype} on {device.type}, grad scaler {'on' if scaler.is_enabled() else 'off'}")

# T_max is the cosine period in epochs and only makes sense as the run length, since setting it apart silently decays a fraction of the schedule (a 10-epoch run against T_max 50 moved lr by 9%, i.e. constant), so default it to epochs rather than validating agreement between two sources.
scheduler = CosineAnnealingLR(optimizer,
                              T_max=train_cfg["scheduler"].get("T_max")
                                    or train_cfg.get("epochs", 50),
                              eta_min=float(train_cfg["scheduler"]["eta_min"]))

# -----------------------------
# 5. Loss function
# -----------------------------
# Every term is scale-free now that the ground truth is divided by the average scene
# depth (Eq. 3, Eq. 4), so these weights finally compare like with like. The paper gives
# no value for alpha or beta.
loss_cfg = train_cfg.get("loss", {})
criterion = MultiTaskLoss(
    w_point=loss_cfg.get("w_point", 1.0),
    w_pose=loss_cfg.get("w_pose", 1.0),
    w_rig=loss_cfg.get("w_rig", 1.0),
    alpha=loss_cfg.get("alpha", 0.2),
    beta=loss_cfg.get("beta", 1.0),
    conf_max=loss_cfg.get("conf_max", 10.0),
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


def compute_loss(outputs, batch, device, img_size, patch_size, z_bar):
    """Score whatever supervision this batch actually carries.

    Waymo currently supplies raymap targets (from calibration + rig poses) but no
    pointcloud; CO3D supplies a pointcloud but no calibration. MultiTaskLoss skips
    any term missing from either side, so each dataset trains on what it has.

    z_bar is (B,) average scene depth. Every target is divided by the same value, so
    the pointmap and the two camera centres stay on one consistent scale - computing
    it separately per target would let them disagree.
    """
    preds = dict(outputs)
    targets = {}

    pointcloud = batch["pointcloud"].to(device)
    pointmap = batch.get("pointmap", torch.empty(0)).to(device)

    if pointcloud.numel() > 0 or pointmap.numel() > 0:
        # the heads predict at image resolution, the targets live on the patch grid
        preds["pointmap"] = downsample_pointmap(preds["pointmap"], img_size[0], patch_size)
        if "pointmap_conf" in preds:
            preds["pointmap_conf"] = downsample_pointmap(
                preds["pointmap_conf"], img_size[0], patch_size
            )

    if pointcloud.numel() > 0:
        targets["pointmap"] = pointcloud / z_bar.view(-1, 1, 1)
    elif pointmap.numel() > 0:
        points, valid = build_pointmap_target(
            pointmap=pointmap,
            cam2rig=batch["metadata"]["cam2rig"].to(device),
            world_from_rig=batch["world_from_rig"].to(device),
            patch_size=patch_size,
            image_size=img_size,
        )
        # Eq. 3 normalizes only the ground truth, so the model learns to predict at
        # normalized scale. `valid` is the lidar coverage fraction, used as the mask
        # D_v rather than as a weight.
        targets["pointmap"] = points / z_bar.view(-1, 1, 1, 1)
        targets["pointmap_conf"] = valid

    if "intrinsics" in batch:
        targets.update(build_raymap_targets(
            cam2rig=batch["metadata"]["cam2rig"].to(device),
            intrinsics=batch["intrinsics"].to(device),
            world_from_rig=batch["world_from_rig"].to(device),
            image_size=img_size,
            patch_size=patch_size,
            z_bar=z_bar,
        ))

    total, loss_dict = criterion(preds, targets)

    # Geometric metrics, not loss terms: degrees are degrees whatever the objective is,
    # so these stay comparable across a change to the loss itself.
    loss_dict.update(raymap_metrics(preds, targets))

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
def unpack_batch(batch, device, img_size, patch_size, rig_metadata=False):
    """Move a collated sample dict onto the device. Same shape for co3d and waymo.

    rig_metadata adds the sec 3.3 rig raymap patch r_i to the metadata. It is the
    same tensor the rig raymap head is scored against, so it is only ever supplied
    during training, where the decoder's field dropout withholds it half the time.
    Validation runs without it on purpose: with dropout off, a model handed r_i would
    score a near-zero rig loss by copying, which measures nothing. Withholding it
    makes val/loss report what the paper actually claims - rig structure inferred
    from images.
    """
    images = batch["images"].to(device)

    metadata = batch["metadata"]
    for key, value in metadata.items():
        if value is not None:
            metadata[key] = value.to(device)

    # z_bar comes from the raw ground truth, before any target construction, so the
    # same value can normalize the pointmap target, both camera centres, and the rig
    # raymap fed back in as metadata.
    pointmap = batch.get("pointmap", torch.empty(0)).to(device)
    source = pointmap if pointmap.numel() > 0 else batch["pointcloud"].to(device)
    z_bar = scene_scale(source)

    if rig_metadata and "intrinsics" in batch:
        metadata["rig_raymap"] = build_raymap_targets(
            cam2rig=metadata["cam2rig"],
            intrinsics=batch["intrinsics"].to(device),
            world_from_rig=batch["world_from_rig"].to(device),
            image_size=img_size,
            patch_size=patch_size,
            z_bar=z_bar,
        )["rig_raymap"]

    return images, metadata, z_bar


# -----------------------------
# 7. Logging setup
# -----------------------------
run = wandb.init(
    project=train_cfg.get("wandb_project", "open-rig3r"),
    entity=train_cfg.get("wandb_entity"),
    name=train_cfg.get("wandb_run_name"),
    mode=train_cfg.get("wandb_mode", "online"),  # "offline" / "disabled" for no network
    config={**train_cfg, "config_file": args.config, "device": str(device),
            "amp_dtype": str(amp_dtype), "grad_scaler": scaler.is_enabled()},
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
        images, metadata, z_bar = unpack_batch(
            batch, device, img_size, patch_size, rig_metadata=True
        )

        optimizer.zero_grad()
        with autocast(device_type=device.type, dtype=amp_dtype):
            outputs = model(images, metadata)
        loss, loss_dict = compute_loss(outputs, batch, device, img_size, patch_size, z_bar)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
    val_totals = {}
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
    with torch.no_grad():
        for batch in val_bar:
            images, metadata, z_bar = unpack_batch(batch, device, img_size, patch_size)

            with autocast(device_type=device.type, dtype=amp_dtype):
                outputs = model(images, metadata)
            loss, val_dict = compute_loss(outputs, batch, device, img_size, patch_size, z_bar)

            val_loss += loss.item()
            for name, value in val_dict.items():
                val_totals[name] = val_totals.get(name, 0.0) + float(value)
            val_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    batches = len(val_loader)
    avg_val_loss = val_loss / batches
    val_averages = {name: total / batches for name, total in val_totals.items()}
    degrees = " ".join(
        f"{name} {value:.2f}deg" for name, value in val_averages.items() if name.endswith("_deg")
    )
    print(f"Epoch [{epoch+1}/{num_epochs}] - Val Loss: {avg_val_loss:.4f}  {degrees}")

    wandb.log(
        {
            "epoch": epoch + 1,
            "train/loss": avg_loss,
            "val/loss": avg_val_loss,
            "lr": scheduler.get_last_lr()[0],
            **{f"val/{name}": value for name, value in val_averages.items()},
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
            metadata={
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,  # for the record, not for selection
                "val_pose_deg": val_averages.get("pose_deg"),
                "val_rig_deg": val_averages.get("rig_deg"),
            },
        )
        artifact.add_file(ckpt_path)
        run.log_artifact(artifact, aliases=["latest", f"epoch-{epoch+1}"])

run.finish()
print("Training finished!")