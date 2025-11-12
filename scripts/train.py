import os
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.amp import autocast

import sys

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

# --- Import dataset and transforms ---
from datasets.co3d import Co3DDataset
from datasets.transform import get_train_transforms, get_val_transforms

# --- Import model ---
from models.rig3r import Rig3R  

# --- Optional: logging ---
from torch.utils.tensorboard import SummaryWriter

# -----------------------------
# 1. Load configs
# -----------------------------
with open("configs/train.yaml", "r") as f:
    train_cfg = yaml.safe_load(f)

with open("configs/model.yaml", "r") as f:
    model_cfg = yaml.safe_load(f)

device = torch.device(train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2. Prepare datasets
# -----------------------------
co3d_path = Path.cwd().joinpath("data/co3d")
img_size=(128, 128)

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

train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0)

# -----------------------------
# 3. Initialize model
# -----------------------------

pretrain_ckpt_path = Path.cwd().joinpath("checkpoints/pretrained/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")

# model = Rig3R(model_cfg)
model = Rig3R(
    encoder_ckpt=pretrain_ckpt_path,    # use sinusoidal encoder for test
    img_size=128,
    patch_size=8,
    embed_dim=128,
    metadata_dim=128,
    num_decoder_layers=2,  # small for test
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
    return nn.MSELoss()(outputs["pointcloud_pred"], pointcloud_gt)

# -----------------------------
# 6. Logging setup
# -----------------------------
writer = SummaryWriter(log_dir="runs/rig3r_train")

# -----------------------------
# 7. Training loop
# -----------------------------
# num_epochs = train_cfg.get("epochs", 50)
num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    
    for batch_idx, batch in enumerate(train_bar):
        images = batch["images"].to(device)
        metadata = batch["metadata"]
        pointcloud = batch["pointcloud"].to(device)

        # Move metadata tensors to device
        for key, value in metadata.items():
            if value is not None:
                metadata[key] = value.to(device)

        optimizer.zero_grad()
        # print(f"device: {device}")
        # print(f"device: {type(device)}")
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
    # 8. Validation loop (optional)
    # -----------------------------
    model.eval()
    val_loss = 0.0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
    with torch.no_grad():
        for batch in val_bar:
            images = batch["images"].to(device)
            metadata = batch["metadata"]
            pointcloud = batch["pointcloud"].to(device)
            for key, value in metadata.items():
                if value is not None:
                    metadata[key] = value.to(device)
            outputs = model(images, metadata)
            loss = compute_loss(outputs, pointcloud)
            val_loss += loss.item()
            val_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Val Loss: {avg_val_loss:.4f}")
    writer.add_scalar("Loss/val", avg_val_loss, epoch)

    # -----------------------------
    # 9. Save checkpoints
    # -----------------------------
    if (epoch + 1) % 5 == 0:
        ckpt_path = os.path.join("checkpoints", f"rig3r_epoch{epoch+1}.pt")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

writer.close()
print("Training finished!")
