from torchvision import transforms

from pathlib import Path
import sys

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from datasets.wayve101 import Wayve101Dataset


# --- Define transforms (optional) ---
img_size = (128, 128)
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])

data_dir = Path.cwd().joinpath("data/wayve_scenes_101/scene_001")

# --- Instantiate dataset ---
dataset = Wayve101Dataset(
    root_dir=data_dir,
    n_frames=2,
    image_size=img_size,
    transforms=transform,
    use_masks=True,
    metadata_dropout=0.0
)

# --- Grab one sample ---
sample = dataset[0]

print("Images shape:", sample['images'].shape)       # Expect (num_cameras * n_frames, 3, H, W)
if sample['masks'] is not None:
    print("Masks shape:", sample['masks'].shape)     # Expect (num_cameras * n_frames, 1, H, W)
print("Metadata keys:", sample['metadata'].keys())  # Expect dict with 'cam2rig'
print("Cam2rig shape:", sample['metadata']['cam2rig'].shape)  # (num_cameras * n_frames, 3, 3)
print("Pointcloud shape:", sample['pointcloud'].shape)       # (N_points, 3)
print("Sequence path:", sample['seq_path'])
