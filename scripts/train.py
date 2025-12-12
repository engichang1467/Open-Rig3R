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

# --- Optional: logging ---
from torch.utils.tensorboard import SummaryWriter

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
    waymo_path = Path.cwd().joinpath(train_cfg.get("waymo_path", "data/waymo_mini"))
    component = train_cfg.get("waymo_component", "camera_image")
    sequence_ids = train_cfg.get("sequence_ids", None)
    
    train_dataset = WaymoDataset(
        root_dir=waymo_path,
        split="train",
        component=component,
        sequence_ids=sequence_ids,
        n_frames=train_cfg["n_frames"]
    )
    
    val_dataset = WaymoDataset(
        root_dir=waymo_path,
        split="validation",
        component=component,
        sequence_ids=sequence_ids,
        n_frames=train_cfg["n_frames"]
    )
else:
    raise ValueError(f"Unknown dataset type: {dataset_type}")

train_loader = DataLoader(
    train_dataset, 
    batch_size=train_cfg["batch_size"], 
    shuffle=True, 
    num_workers=train_cfg.get("num_workers", 0),
    collate_fn=None if dataset_type == "co3d" else lambda x: x  # waymo returns dicts
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=train_cfg["batch_size"], 
    shuffle=False, 
    num_workers=train_cfg.get("num_workers", 0),
    collate_fn=None if dataset_type == "co3d" else lambda x: x
)

# -----------------------------
# 3. Initialize model
# -----------------------------

pretrain_ckpt_path = Path.cwd().joinpath("checkpoints/pretrained/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")

model = Rig3R(
    encoder_ckpt=pretrain_ckpt_path,
    img_size=img_size[0],
    patch_size=8,
    embed_dim=128,
    metadata_dim=128,
    num_decoder_layers=2,
    num_heads=2,
    mlp_dim=128*4
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
# 5. Loss function (example)
# -----------------------------
def compute_loss(outputs, pointcloud_gt):
    if pointcloud_gt.numel() == 0:
        return torch.tensor(0.0, device=pointcloud_gt.device, requires_grad=True)
    return nn.MSELoss()(outputs["pointmap"], pointcloud_gt)


# -----------------------------
# 6. Waymo data processing helper
# -----------------------------
def process_waymo_batch(batch_list, device, img_size):
    """
    Process raw waymo parquet data into model inputs.
    
    Args:
        batch_list: List of dictionaries from WaymoDataset, each containing 'frames'
        device: torch device
        img_size: tuple of (H, W)
        
    Returns:
        images: (B, N, 3, H, W) tensor
        metadata: dict with camera parameters (only cam2rig)
        pointcloud: (B, num_points, 3) dummy pointcloud for now
    """
    batch_size = len(batch_list)
    n_frames = len(batch_list[0]['frames'])

    # determine dtype based on device - match autocast default
    dtype = torch.bfloat16 if device.type == 'cpu' else torch.float16
    
    # create dummy data matching model's expected format
    # shape: (B, N, 3, H, W)
    images = torch.randn(batch_size, n_frames, 3, *img_size, dtype=dtype).to(device)
    
    # create cam2rig: (B, N, 3, 3) - 3x3 rotation part of transformation
    # then reshape to (B, N*3, 3) for the decoder
    cam2rig_3x3 = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_frames, 1, 1)
    
    # reshape: (B, N, 3, 3) -> (B, N*3, 3)
    cam2rig = cam2rig_3x3.reshape(batch_size, n_frames * 3, 3) #.to(device)
    
    metadata = {
        "cam2rig": cam2rig
    }
    
    # dummy pointcloud - now match model output shape (B, N, P, 3)
    # P = number of patches = (H // patch_size) * (W // patch_size)
    # assuming patch_size=8 and img_size from config
    patch_size = 8
    num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
    pointcloud = torch.randn(batch_size, n_frames, num_patches, 3, dtype=dtype).to(device)
    
    return images, metadata, pointcloud

# -----------------------------
# 7. Logging setup
# -----------------------------
writer = SummaryWriter(log_dir="runs/rig3r_train")

# -----------------------------
# 8. Training loop
# -----------------------------
num_epochs = train_cfg.get("epochs", 50)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    
    for batch_idx, batch in enumerate(train_bar):
        if dataset_type == "co3d":
            images = batch["images"].to(device)
            metadata = batch["metadata"]
            pointcloud = batch["pointcloud"].to(device)

            # Move metadata tensors to device
            for key, value in metadata.items():
                if value is not None:
                    metadata[key] = value.to(device)

            optimizer.zero_grad()
            with autocast(device_type=str(device)):
                outputs = model(images, metadata)
            loss = compute_loss(outputs, pointcloud)
            
        elif dataset_type == "waymo":
            # Process waymo batch - now with correct shape
            images, metadata, pointcloud = process_waymo_batch(batch, device, img_size)
            
            optimizer.zero_grad()
            with autocast(device_type=str(device)):
                outputs = model(images, metadata)
            loss = compute_loss(outputs, pointcloud)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    scheduler.step()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_loss:.4f}")
    writer.add_scalar("Loss/train", avg_loss, epoch)

    # -----------------------------
    # 9. Validation loop
    # -----------------------------
    model.eval()
    val_loss = 0.0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
    with torch.no_grad():
        for batch in val_bar:
            if dataset_type == "co3d":
                images = batch["images"].to(device)
                metadata = batch["metadata"]
                pointcloud = batch["pointcloud"].to(device)
                for key, value in metadata.items():
                    if value is not None:
                        metadata[key] = value.to(device)
                outputs = model(images, metadata)
                loss = compute_loss(outputs, pointcloud)
                
            elif dataset_type == "waymo":
                images, metadata, pointcloud = process_waymo_batch(batch, device, img_size)
                outputs = model(images, metadata)
                loss = compute_loss(outputs, pointcloud)
            
            val_loss += loss.item()
            val_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Val Loss: {avg_val_loss:.4f}")
    writer.add_scalar("Loss/val", avg_val_loss, epoch)

    # -----------------------------
    # 10. Save checkpoints
    # -----------------------------
    if (epoch + 1) % 5 == 0:
        ckpt_path = os.path.join("checkpoints", f"rig3r_epoch{epoch+1}.pt")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

writer.close()
print("Training finished!")