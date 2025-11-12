import torch
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader

import sys

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

# --- Assume Co3DDataset is imported ---
from datasets.co3d import Co3DDataset

def smoke_test_co3d():
    print("Starting Co3D Dataset smoke test...")

    # --- Dummy transforms (resize + normalize) ---
    dummy_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    root_dir = Path.cwd().joinpath("data/co3d")

    # --- Initialize dataset ---
    dataset = Co3DDataset(
        root_dir=root_dir,  # replace with actual path
        subset='train',
        n_frames=4,  # small number for smoke test
        image_size=(128, 128),  # smaller size to save memory
        transforms=dummy_transforms,
        metadata_dropout=0.5
    )

    # --- Wrap in DataLoader ---
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # --- Take one batch ---
    batch = next(iter(loader))
    images = batch['images']       # (B, N, 3, H, W)
    metadata = batch['metadata']   # dict with 'cam2rig'
    pointcloud = batch['pointcloud']

    print("Images shape:", images.shape)
    if metadata.get('cam2rig') is not None:
        print("Cam2Rig metadata shape:", metadata['cam2rig'].shape)
    else:
        print("Cam2Rig metadata dropped (None)")

    if pointcloud is not None:
        print("Pointcloud shape:", pointcloud.shape)
    else:
        print("No pointcloud available")

    print("Co3D Dataset smoke test passed ✅")

if __name__ == "__main__":
    smoke_test_co3d()
