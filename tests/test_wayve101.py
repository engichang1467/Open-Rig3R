"""Tests for the Wayve101 dataset loader.

This file used to be a print script with no assertions: under pytest it collected
zero tests, ran at import time, and happily printed `torch.Size([1, 3])` for a
144005-point cloud while reporting no failure. The COLMAP parser bugs it should
have caught survived because of that.

The parser tests below build their own points3D.bin, so they run in CI without
the ~100 GB dataset. The dataset tests skip when the data is not present.
"""

import struct
from pathlib import Path

import pytest
import torch

import sys

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from datasets.wayve101 import Wayve101Dataset


DATA_DIR = root_path / "data" / "wayve_scenes_101"
needs_data = pytest.mark.skipif(
    not (DATA_DIR / "scene_001").is_dir(),
    reason="WayveScenes101 not downloaded (make download-wayve101)",
)


def write_points3D_bin(path, points, track_len=3):
    """Write a COLMAP points3D.bin containing `points`, each with `track_len` tracks.

    Format: uint64 num_points, then per point
        uint64 point_id | 3x double xyz | 3x uint8 rgb | double error
        | uint64 track_length | track_length x (uint32 image_id, uint32 point2D_idx)
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for i, (x, y, z) in enumerate(points):
            f.write(struct.pack("<Q", 1000 + i))
            f.write(struct.pack("<ddd", x, y, z))
            f.write(struct.pack("<BBB", 10, 20, 30))
            f.write(struct.pack("<d", 0.5))
            f.write(struct.pack("<Q", track_len))
            for t in range(track_len):
                f.write(struct.pack("<II", t, t + 1))


def load_pointcloud_from(tmp_path, points, track_len=3):
    """Build a one-scene root, then read the cloud back through the dataset."""
    scene = tmp_path / "scene" / "colmap_sparse" / "rig"
    scene.mkdir(parents=True, exist_ok=True)
    write_points3D_bin(scene / "points3D.bin", points, track_len=track_len)

    dataset = Wayve101Dataset(root_dir=str(tmp_path))
    return dataset._load_pointcloud(dataset.samples[0])


def test_pointcloud_reads_every_record(tmp_path):
    """The count is the first 8 bytes. Consuming a phantom header first made the
    first point's id decode as the count and every field after it 8 bytes off."""
    points = [(float(i), float(i) + 0.5, float(i) - 0.5) for i in range(50)]
    pc = load_pointcloud_from(tmp_path, points)

    assert pc.shape == (50, 3), f"expected all 50 points, parser returned {tuple(pc.shape)}"
    torch.testing.assert_close(pc, torch.tensor(points, dtype=torch.float32))
    print("points3D.bin returns every record test passed!")


def test_pointcloud_skips_tracks_by_eight_bytes(tmp_path):
    """A track element is (uint32, uint32) = 8 bytes. Skipping 16 walks past the
    record boundary, so the next track_len decodes as garbage and the loop bails
    out after a single point."""
    points = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
    for track_len in (1, 7, 64):
        pc = load_pointcloud_from(tmp_path / f"t{track_len}", points, track_len=track_len)
        assert pc.shape == (3, 3), f"track_len={track_len} gave {tuple(pc.shape)}"
        torch.testing.assert_close(pc, torch.tensor(points, dtype=torch.float32))
    print("track records are skipped 8 bytes at a time test passed!")


def test_pointcloud_is_finite(tmp_path):
    """A misaligned read decodes xyz from the wrong offsets and yields inf, which
    then makes align_scale infinite and the Hungarian cost matrix infeasible."""
    points = [(float(i), 0.0, -float(i)) for i in range(20)]
    pc = load_pointcloud_from(tmp_path, points)
    assert torch.isfinite(pc).all(), "parsed pointcloud contains inf/nan"
    print("parsed pointcloud is finite test passed!")


def test_missing_pointcloud_file_is_empty(tmp_path):
    (tmp_path / "scene").mkdir()
    dataset = Wayve101Dataset(root_dir=str(tmp_path))
    assert dataset._load_pointcloud(dataset.samples[0]).shape == (0, 3)
    print("missing points3D.bin yields an empty cloud test passed!")


def test_truncated_file_is_empty(tmp_path):
    scene = tmp_path / "scene" / "colmap_sparse" / "rig"
    scene.mkdir(parents=True)
    (scene / "points3D.bin").write_bytes(b"\x01\x02")

    dataset = Wayve101Dataset(root_dir=str(tmp_path))
    assert dataset._load_pointcloud(dataset.samples[0]).shape == (0, 3)
    print("truncated points3D.bin yields an empty cloud test passed!")


@needs_data
def test_root_dir_lists_scenes_not_scene_internals():
    """root_dir is the directory *containing* scenes. It used to be pointed at a
    single scene, so the three "sequences" were its images/, colmap_sparse/ and
    masks/ subdirectories, all walked back to the same scene by a dirname hack."""
    dataset = Wayve101Dataset(root_dir=str(DATA_DIR), n_frames=2, image_size=(128, 128))

    assert len(dataset) > 0
    names = [Path(s).name for s in dataset.samples]
    assert all(n.startswith("scene_") for n in names), names
    assert len(set(names)) == len(names), f"duplicate sequences: {names}"
    print(f"root_dir lists {len(names)} distinct scenes test passed!")


@needs_data
def test_real_sample_has_a_dense_finite_pointcloud():
    dataset = Wayve101Dataset(root_dir=str(DATA_DIR), n_frames=2, image_size=(128, 128))
    sample = dataset[0]

    pc = sample["pointcloud"]
    assert pc.shape[0] > 1000, f"only {pc.shape[0]} points - parser is truncating"
    assert torch.isfinite(pc).all()

    assert sample["images"].shape == (10, 3, 128, 128)
    assert sample["metadata"]["cam2rig"].shape == (10, 4, 4)
    print(f"real sample carries {pc.shape[0]} finite points test passed!")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
