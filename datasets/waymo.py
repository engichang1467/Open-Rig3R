from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# waymo CameraName enum order, as written by ops/parquet2jpeg.py
CAMERAS = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]


class WaymoDataset(Dataset):
    """
    Waymo Open Dataset camera images, exported to JPEG by ops/parquet2jpeg.py.

    Expected layout:
        root_dir/<split>/<segment>/<CAMERA>/<frame_timestamp_micros>.jpeg

    One sample is a rig capture window: `n_frames` consecutive timestamps of one
    segment, each contributing one image per camera. Views per sample is
    therefore `n_frames * len(cameras)`, constant across the dataset so the
    default collate can stack them.

    Args:
        root_dir: Root of the exported JPEG tree (e.g. /data/waymo_mini)
        split: 'train' or 'validation'
        cameras: Camera names to load. Defaults to the full 5-camera rig.
        sequence_ids: Optional list of segment names to keep. None = all.
        n_frames: Number of consecutive timestamps per sample
        image_size: (H, W) used by the fallback transform
        transforms: torchvision transform applied per image
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        cameras: Optional[Sequence[str]] = None,
        sequence_ids: Optional[List[str]] = None,
        n_frames: int = 2,
        image_size: Tuple[int, int] = (128, 128),
        transforms=None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        self.n_frames = n_frames
        self.image_size = tuple(image_size)
        self.cameras = list(cameras) if cameras else CAMERAS
        self.transforms = transforms or T.Compose(
            [T.Resize(self.image_size), T.ToTensor()]
        )

        if not self.split_dir.exists():
            raise ValueError(
                f"Split directory not found: {self.split_dir}\n"
                f"Expected structure: {self.root_dir}/{split}/<segment>/<CAMERA>/*.jpeg\n"
                f"Run `python ops/parquet2jpeg.py` to export the parquet dataset first."
            )

        segments = sorted(p for p in self.split_dir.iterdir() if p.is_dir())
        if sequence_ids is not None:
            segments = [p for p in segments if p.name in sequence_ids]

        # index of (segment_dir, [timestamps]) sliding windows
        self.samples: List[Tuple[Path, List[str]]] = []
        for segment in segments:
            timestamps = self._shared_timestamps(segment)
            for i in range(len(timestamps) - n_frames + 1):
                self.samples.append((segment, timestamps[i : i + n_frames]))

        if len(self.samples) == 0:
            raise ValueError(
                f"No usable frames found in {self.split_dir} "
                f"for cameras {self.cameras} with n_frames={n_frames}"
            )

    def _shared_timestamps(self, segment: Path) -> List[str]:
        """Timestamps present in every requested camera of this segment."""
        shared = None
        for camera in self.cameras:
            camera_dir = segment / camera
            if not camera_dir.is_dir():
                return []  # incomplete rig, skip the whole segment
            stems = {p.stem for p in camera_dir.glob("*.jpeg")}
            shared = stems if shared is None else shared & stems
        return sorted(shared or [], key=int)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        segment, timestamps = self.samples[idx]

        images = []
        for timestamp in timestamps:
            for camera in self.cameras:
                image = Image.open(segment / camera / f"{timestamp}.jpeg").convert("RGB")
                images.append(self.transforms(image))

        images = torch.stack(images)  # (n_frames * n_cameras, 3, H, W)

        return {
            "images": images,
            "metadata": {"cam2rig": self._cam2rig(images.shape[0])},
            "pointcloud": torch.empty(0, 3),
            "segment_id": segment.name,
            "timestamps": [int(t) for t in timestamps],
        }

    def _cam2rig(self, n_views: int) -> torch.Tensor:
        # ponytail: identity extrinsics - the JPEG export carries no calibration.
        # Real values live in the camera_calibration parquet component; export
        # them alongside the images and read them here when rig geometry matters.
        return torch.eye(3).repeat(n_views, 1)  # (n_views * 3, 3)

    def get_sequence_ids(self) -> List[str]:
        """Returns the segment IDs actually indexed, in order, without duplicates."""
        return list(dict.fromkeys(segment.name for segment, _ in self.samples))
