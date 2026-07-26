import json
import shutil
import sys
from pathlib import Path
import traceback

import numpy
import torch
from PIL import Image

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from datasets.waymo import CAMERAS, WaymoDataset


SPLITS = ["train", "validation"]
SEQUENCE_IDS = [
    "10017090168044687777_6380_000_6400_000",
    "10023947602400723454_1120_000_1140_000",
]
TIMESTAMPS = [1550083470045260 + i for i in range(10)]

TMP_PATH = Path("tmp_test_data")


def mock_cam2rig(index):
    """Distinct SE(3) per camera so tests can tell the rig apart from identity."""
    transform = torch.eye(4)
    transform[0, 3] = float(index)
    return transform


def mock_world_from_rig(index):
    """A rig driving straight along world +x, one metre per frame."""
    transform = torch.eye(4)
    transform[0, 3] = float(index)
    return transform


POINTMAP_GRID = 8


def mock_pointmap():
    """(n_timestamps, G, G, 3) with the bottom half empty, as lidar tends to be."""
    grid = numpy.full(
        (len(TIMESTAMPS), POINTMAP_GRID, POINTMAP_GRID, 3), numpy.nan, dtype=numpy.float16
    )
    grid[:, : POINTMAP_GRID // 2] = 5.0
    return grid


def create_mock_waymo_dataset(tmp_path):
    """
    Creates a minimal mock waymo JPEG tree for testing.
    Matches ops/parquet2jpeg.py output: root/split/segment/CAMERA/<timestamp>.jpeg
    plus root/split/segment/{calibration,poses}.json
    """
    for split in SPLITS:
        for sequence_id in SEQUENCE_IDS:
            for camera in CAMERAS:
                camera_dir = tmp_path / split / sequence_id / camera
                camera_dir.mkdir(parents=True, exist_ok=True)

                for timestamp in TIMESTAMPS:
                    Image.new("RGB", (32, 24), color=(10, 20, 30)).save(
                        camera_dir / f"{timestamp}.jpeg"
                    )

            calibration = {
                camera: {
                    "cam2rig": mock_cam2rig(i).flatten().tolist(),
                    "f_u": 2000.0, "f_v": 2000.0, "c_u": 960.0, "c_v": 640.0,
                    "width": 1920, "height": 1280,
                }
                for i, camera in enumerate(CAMERAS)
            }
            segment_dir = tmp_path / split / sequence_id
            (segment_dir / "calibration.json").write_text(json.dumps(calibration))

            poses = {
                str(timestamp): {
                    camera: mock_world_from_rig(i).flatten().tolist() for camera in CAMERAS
                }
                for i, timestamp in enumerate(TIMESTAMPS)
            }
            (segment_dir / "poses.json").write_text(json.dumps(poses))

            pointmap_dir = segment_dir / "pointmap"
            pointmap_dir.mkdir(exist_ok=True)
            for camera in CAMERAS:
                numpy.save(pointmap_dir / f"{camera}.npy", mock_pointmap())
            (pointmap_dir / "timestamps.json").write_text(
                json.dumps([str(timestamp) for timestamp in TIMESTAMPS])
            )

    return tmp_path


def with_mock_dataset(fn):
    """Build the mock tree, run fn(mock_path), always clean up."""
    def wrapper():
        TMP_PATH.mkdir(exist_ok=True)
        try:
            fn(create_mock_waymo_dataset(TMP_PATH))
        finally:
            if TMP_PATH.exists():
                shutil.rmtree(TMP_PATH)

    wrapper.__name__ = fn.__name__
    return wrapper


@with_mock_dataset
def test_dataset_initialization(mock_data):
    """Test that dataset initializes correctly"""
    dataset = WaymoDataset(root_dir=mock_data, split="train")

    assert len(dataset) > 0, "Dataset length should be > 0"
    assert dataset.cameras == CAMERAS, f"Expected full rig, got {dataset.cameras}"
    print("✓ test_dataset_initialization passed")


@with_mock_dataset
def test_dataset_splits(mock_data):
    """Test that both train and validation splits work"""
    # 2 sequences x (10 timestamps - 2 + 1) windows = 18
    train_dataset = WaymoDataset(root_dir=mock_data, split="train")
    val_dataset = WaymoDataset(root_dir=mock_data, split="validation")

    assert len(train_dataset) == 18, f"Expected 18 train samples, got {len(train_dataset)}"
    assert len(val_dataset) == 18, f"Expected 18 val samples, got {len(val_dataset)}"
    print("✓ test_dataset_splits passed")


@with_mock_dataset
def test_dataset_getitem(mock_data):
    """Test that a sample carries real image tensors and rig metadata"""
    dataset = WaymoDataset(root_dir=mock_data, split="train", n_frames=2, image_size=(64, 64))
    sample = dataset[0]

    n_views = 2 * len(CAMERAS)

    assert isinstance(sample, dict), f"Sample should be dict, got {type(sample)}"
    assert sample["images"].shape == (n_views, 3, 64, 64), (
        f"Expected images {(n_views, 3, 64, 64)}, got {tuple(sample['images'].shape)}"
    )
    assert sample["metadata"]["cam2rig"].shape == (n_views, 4, 4), (
        f"Expected cam2rig {(n_views, 4, 4)}, got {tuple(sample['metadata']['cam2rig'].shape)}"
    )
    assert sample["segment_id"] in SEQUENCE_IDS, "Sample should report its segment"
    assert len(sample["timestamps"]) == 2, "Sample should carry one timestamp per frame"
    print("✓ test_dataset_getitem passed")


@with_mock_dataset
def test_cam2rig_is_real_calibration(mock_data):
    """Guard against identity extrinsics: cam2rig must come from calibration.json"""
    dataset = WaymoDataset(root_dir=mock_data, split="train", n_frames=2)
    cam2rig = dataset[0]["metadata"]["cam2rig"]

    expected = torch.stack([mock_cam2rig(i) for i in range(len(CAMERAS))])
    assert torch.allclose(cam2rig[: len(CAMERAS)], expected), "cam2rig does not match calibration.json"
    # rig is rigid: the same block repeats for every frame in the window
    assert torch.allclose(cam2rig[len(CAMERAS) :], expected), "cam2rig should tile over frames"
    assert not torch.allclose(cam2rig, torch.eye(4)), "cam2rig collapsed back to identity"
    print("✓ test_cam2rig_is_real_calibration passed")


@with_mock_dataset
def test_cam2rig_follows_camera_order(mock_data):
    """cam2rig rows must line up with the cameras the images were stacked in"""
    cameras = ["SIDE_RIGHT", "FRONT"]
    dataset = WaymoDataset(root_dir=mock_data, split="train", cameras=cameras, n_frames=1)
    cam2rig = dataset[0]["metadata"]["cam2rig"]

    expected = torch.stack([mock_cam2rig(CAMERAS.index(c)) for c in cameras])
    assert torch.allclose(cam2rig, expected), f"cam2rig out of order: {cam2rig[:, 0, 3]}"
    print("✓ test_cam2rig_follows_camera_order passed")


@with_mock_dataset
def test_intrinsics_scaled_to_image_size(mock_data):
    """Intrinsics are exported at native resolution, so the loader must rescale"""
    dataset = WaymoDataset(root_dir=mock_data, split="train", n_frames=1, image_size=(64, 64))
    intrinsics = dataset[0]["intrinsics"]

    assert intrinsics.shape == (len(CAMERAS), 4), f"Got {tuple(intrinsics.shape)}"
    # mock calibration is f=2000, c=(960, 640) at 1920x1280 -> 64x64
    expected = torch.tensor([2000 / 30, 2000 / 20, 960 / 30, 640 / 20])
    assert torch.allclose(intrinsics[0], expected), f"Expected {expected}, got {intrinsics[0]}"
    print("✓ test_intrinsics_scaled_to_image_size passed")


@with_mock_dataset
def test_world_from_rig_is_per_frame(mock_data):
    """Rig pose varies per timestamp and is shared by every camera in that frame"""
    dataset = WaymoDataset(root_dir=mock_data, split="train", n_frames=2)
    world_from_rig = dataset[0]["world_from_rig"]

    n_cameras = len(CAMERAS)
    assert world_from_rig.shape == (2 * n_cameras, 4, 4), f"Got {tuple(world_from_rig.shape)}"
    assert torch.allclose(world_from_rig[:n_cameras], world_from_rig[0]), (
        "All cameras in a frame share one rig pose"
    )
    displacement = world_from_rig[n_cameras, 0, 3] - world_from_rig[0, 0, 3]
    assert displacement != 0, "Rig pose should differ between frames, or pose targets are static"
    print("✓ test_world_from_rig_is_per_frame passed")


@with_mock_dataset
def test_pointmap_is_loaded_per_view(mock_data):
    """Lidar pointmaps line up one-per-view and keep their empty cells as NaN"""
    dataset = WaymoDataset(root_dir=mock_data, split="train", n_frames=2)
    pointmap = dataset[0]["pointmap"]

    n_views = 2 * len(CAMERAS)
    assert pointmap.shape == (n_views, POINTMAP_GRID, POINTMAP_GRID, 3), (
        f"Got {tuple(pointmap.shape)}"
    )
    assert torch.isnan(pointmap[:, POINTMAP_GRID // 2 :]).all(), "Empty cells should stay NaN"
    assert not torch.isnan(pointmap[:, : POINTMAP_GRID // 2]).any(), "Filled cells lost"
    print("✓ test_pointmap_is_loaded_per_view passed")


@with_mock_dataset
def test_pointmap_is_optional(mock_data):
    """A tree exported before lidar support still loads, just without pointmaps"""
    for path in mock_data.glob("train/*/pointmap"):
        shutil.rmtree(path)

    dataset = WaymoDataset(root_dir=mock_data, split="train", n_frames=2)
    sample = dataset[0]

    assert sample["pointmap"].numel() == 0, "Missing pointmaps should give an empty tensor"
    assert sample["images"].shape[0] == 2 * len(CAMERAS), "Images should still load"
    print("✓ test_pointmap_is_optional passed")


@with_mock_dataset
def test_dataset_missing_poses(mock_data):
    """A segment exported before pose support fails loudly, not silently"""
    for path in mock_data.glob("train/*/poses.json"):
        path.unlink()

    error_raised = False
    try:
        WaymoDataset(root_dir=mock_data, split="train")
    except ValueError as e:
        error_raised = "Poses not found" in str(e)

    assert error_raised, "Should raise ValueError when poses.json is missing"
    print("✓ test_dataset_missing_poses passed")


@with_mock_dataset
def test_dataset_missing_calibration(mock_data):
    """A segment exported before calibration support fails loudly, not silently"""
    for path in mock_data.glob("train/*/calibration.json"):
        path.unlink()

    error_raised = False
    try:
        WaymoDataset(root_dir=mock_data, split="train")
    except ValueError as e:
        error_raised = "Calibration not found" in str(e)

    assert error_raised, "Should raise ValueError when calibration.json is missing"
    print("✓ test_dataset_missing_calibration passed")


@with_mock_dataset
def test_images_are_decoded_pixels(mock_data):
    """Guard against silently feeding random/dummy tensors to the model"""
    dataset = WaymoDataset(root_dir=mock_data, split="train", image_size=(16, 16), transforms=None)
    images = dataset[0]["images"]

    assert images.dtype == torch.float32, f"Expected float32, got {images.dtype}"
    assert 0.0 <= images.min() and images.max() <= 1.0, "ToTensor output should be in [0, 1]"
    # every mock pixel is the same colour, so a real decode has near-zero variance
    assert images.std() < 0.05, f"Images look like noise, not decoded JPEGs (std={images.std():.3f})"
    print("✓ test_images_are_decoded_pixels passed")


@with_mock_dataset
def test_dataset_n_frames(mock_data):
    """Test that n_frames drives both window count and view count"""
    dataset = WaymoDataset(root_dir=mock_data, split="train", n_frames=4)

    # 2 sequences x (10 - 4 + 1) = 14
    assert len(dataset) == 14, f"Expected 14 samples, got {len(dataset)}"
    assert dataset[0]["images"].shape[0] == 4 * len(CAMERAS), "Views should be n_frames * cameras"
    print("✓ test_dataset_n_frames passed")


@with_mock_dataset
def test_dataset_camera_subset(mock_data):
    """Test loading a subset of the rig"""
    dataset = WaymoDataset(root_dir=mock_data, split="train", cameras=["FRONT"], n_frames=2)

    assert dataset[0]["images"].shape[0] == 2, "Single camera x 2 frames = 2 views"
    print("✓ test_dataset_camera_subset passed")


@with_mock_dataset
def test_dataset_sequence_filtering(mock_data):
    """Test filtering by sequence IDs"""
    dataset = WaymoDataset(root_dir=mock_data, split="train", sequence_ids=[SEQUENCE_IDS[0]])

    # 1 sequence x (10 - 2 + 1) = 9
    assert len(dataset) == 9, f"Expected 9 samples, got {len(dataset)}"
    print("✓ test_dataset_sequence_filtering passed")


@with_mock_dataset
def test_dataset_invalid_split(mock_data):
    """Test that invalid split raises error"""
    error_raised = False
    try:
        WaymoDataset(root_dir=mock_data, split="invalid_split")
    except ValueError as e:
        error_raised = "Split directory not found" in str(e)

    assert error_raised, "Should raise ValueError for invalid split"
    print("✓ test_dataset_invalid_split passed")


@with_mock_dataset
def test_dataset_missing_camera(mock_data):
    """A segment missing a requested camera is skipped, not half-loaded"""
    error_raised = False
    try:
        WaymoDataset(root_dir=mock_data, split="train", cameras=["REAR"])
    except ValueError as e:
        error_raised = "No usable frames" in str(e)

    assert error_raised, "Should raise ValueError when no segment has the camera"
    print("✓ test_dataset_missing_camera passed")


@with_mock_dataset
def test_dataset_index_out_of_range(mock_data):
    """Test that out of range index raises error"""
    dataset = WaymoDataset(root_dir=mock_data, split="train")

    error_raised = False
    try:
        _ = dataset[1000]
    except IndexError:
        error_raised = True

    assert error_raised, "Should raise IndexError for out of range index"
    print("✓ test_dataset_index_out_of_range passed")


@with_mock_dataset
def test_get_sequence_ids(mock_data):
    """Test retrieving sequence IDs"""
    dataset = WaymoDataset(root_dir=mock_data, split="train")
    seq_ids = dataset.get_sequence_ids()

    assert len(seq_ids) == 2, f"Expected 2 sequence IDs, got {len(seq_ids)}"
    assert SEQUENCE_IDS[0] in seq_ids, "Missing expected sequence ID"
    print("✓ test_get_sequence_ids passed")


def run_all_tests():
    """Run all tests and report results"""
    tests = [
        test_dataset_initialization,
        test_dataset_splits,
        test_dataset_getitem,
        test_cam2rig_is_real_calibration,
        test_cam2rig_follows_camera_order,
        test_intrinsics_scaled_to_image_size,
        test_world_from_rig_is_per_frame,
        test_pointmap_is_loaded_per_view,
        test_pointmap_is_optional,
        test_dataset_missing_poses,
        test_dataset_missing_calibration,
        test_images_are_decoded_pixels,
        test_dataset_n_frames,
        test_dataset_camera_subset,
        test_dataset_sequence_filtering,
        test_dataset_invalid_split,
        test_dataset_missing_camera,
        test_dataset_index_out_of_range,
        test_get_sequence_ids,
    ]

    passed = 0
    failed = 0

    print("\nRunning Waymo Dataset Tests")
    print("=" * 50)

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            traceback.print_exc()
            failed += 1

    print("=" * 50)
    print(f"\nResults: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
