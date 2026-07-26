import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy
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
        root_dir/<split>/<segment>/calibration.json
        root_dir/<split>/<segment>/poses.json

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
        # segment name -> per-camera geometry, static for the whole segment
        self.cam2rig: Dict[str, torch.Tensor] = {}  # (n_cameras, 4, 4)
        self.intrinsics: Dict[str, torch.Tensor] = {}  # (n_cameras, 4)
        # segment name -> {timestamp: {camera: world_from_rig (4, 4)}}
        self.poses: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
        # segment name -> ({timestamp: row}, {camera: memmapped (n_frames, G, G, 3)})
        self.pointmaps: Dict[str, Tuple[Dict[str, int], Dict[str, numpy.ndarray]]] = {}
        for segment in segments:
            timestamps = self._shared_timestamps(segment)
            if not timestamps:
                continue  # incomplete rig
            poses = self._load_poses(segment)
            timestamps = [t for t in timestamps if t in poses]

            pointmaps = self._load_pointmaps(segment)
            if pointmaps is not None:
                self.pointmaps[segment.name] = pointmaps
                timestamps = [t for t in timestamps if t in pointmaps[0]]

            if not timestamps:
                continue
            self.cam2rig[segment.name], self.intrinsics[segment.name] = (
                self._load_calibration(segment)
            )
            self.poses[segment.name] = poses
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

        # the rig is rigid, so one block of per-camera geometry tiles over frames,
        # matching the (timestamp, camera) order the images were stacked in
        n_frames = len(timestamps)
        cam2rig = self.cam2rig[segment.name].repeat(n_frames, 1, 1)
        intrinsics = self.intrinsics[segment.name].repeat(n_frames, 1)

        # the rig moves and each camera fires at its own instant, so this is per view
        poses = self.poses[segment.name]
        world_from_rig = torch.stack([
            poses[timestamp][camera]
            for timestamp in timestamps
            for camera in self.cameras
        ])

        return {
            "images": images,
            "metadata": {"cam2rig": cam2rig},  # (n_frames * n_cameras, 4, 4)
            "intrinsics": intrinsics,  # (n_frames * n_cameras, 4)
            "world_from_rig": world_from_rig,  # (n_frames * n_cameras, 4, 4)
            "pointmap": self._pointmap(segment, timestamps),
            "pointcloud": torch.empty(0, 3),
            "segment_id": segment.name,
            "timestamps": [int(t) for t in timestamps],
        }

    def _load_calibration(self, segment: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-camera (cam2rig, intrinsics) in self.cameras order.

        Waymo's rig frame is the vehicle frame, so the calibration component's
        extrinsic.transform is cam2rig directly - no composition needed.

        Intrinsics are stored at native resolution and scaled here to self.image_size,
        which assumes the transform pipeline is a plain resize to that size. A
        transform that crops or pads would need its own adjustment.
        """
        calibration_file = segment / "calibration.json"
        if not calibration_file.exists():
            raise ValueError(
                f"Calibration not found: {calibration_file}\n"
                f"Re-run `python ops/parquet2jpeg.py` to export it alongside the images."
            )

        calibration = json.loads(calibration_file.read_text())
        missing = [camera for camera in self.cameras if camera not in calibration]
        if missing:
            raise ValueError(f"{calibration_file} has no calibration for {missing}")

        height, width = self.image_size
        cam2rig, intrinsics = [], []
        for camera in self.cameras:
            entry = calibration[camera]
            scale_u = width / entry["width"]
            scale_v = height / entry["height"]
            cam2rig.append(entry["cam2rig"])
            intrinsics.append([
                entry["f_u"] * scale_u,
                entry["f_v"] * scale_v,
                entry["c_u"] * scale_u,
                entry["c_v"] * scale_v,
            ])

        return (
            torch.tensor(cam2rig, dtype=torch.float32).reshape(len(self.cameras), 4, 4),
            torch.tensor(intrinsics, dtype=torch.float32),
        )

    def _load_poses(self, segment: Path) -> Dict[str, Dict[str, torch.Tensor]]:
        """world_from_rig SE(3) per frame timestamp, per camera capture instant."""
        poses_file = segment / "poses.json"
        if not poses_file.exists():
            raise ValueError(
                f"Poses not found: {poses_file}\n"
                f"Re-run `python ops/parquet2jpeg.py` to export it alongside the images."
            )

        poses = json.loads(poses_file.read_text())
        return {
            timestamp: {
                camera: torch.tensor(transform, dtype=torch.float32).reshape(4, 4)
                for camera, transform in cameras.items()
            }
            for timestamp, cameras in poses.items()
            if all(camera in cameras for camera in self.cameras)
        }

    def _load_pointmaps(self, segment: Path):
        """({timestamp: row}, {camera: memmapped array}), or None if not exported.

        Optional on purpose: without it training still runs on raymap supervision
        alone, which is what every tree exported before ops/parquet2pointmap.py has.
        """
        directory = segment / "pointmap"
        if not directory.is_dir():
            return None

        timestamps = json.loads((directory / "timestamps.json").read_text())
        arrays = {
            camera: numpy.load(directory / f"{camera}.npy", mmap_mode="r")
            for camera in self.cameras
        }
        return {timestamp: row for row, timestamp in enumerate(timestamps)}, arrays

    def _pointmap(self, segment: Path, timestamps: List[str]) -> torch.Tensor:
        """Sparse lidar pointmap per view, in that view's own camera frame.

        (n_frames * n_cameras, grid, grid, 3), NaN where no lidar return landed.
        Empty (0, 0, 0, 3) when the segment has no pointmap export, so training
        falls back to raymap-only supervision instead of failing.
        """
        grids = self.pointmaps.get(segment.name)
        if grids is None:
            return torch.empty(0, 0, 0, 3)

        rows, arrays = grids
        return torch.from_numpy(
            numpy.stack([
                arrays[camera][rows[timestamp]]
                for timestamp in timestamps
                for camera in self.cameras
            ])
        ).float()

    def get_sequence_ids(self) -> List[str]:
        """Returns the segment IDs actually indexed, in order, without duplicates."""
        return list(dict.fromkeys(segment.name for segment, _ in self.samples))
