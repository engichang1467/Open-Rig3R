import os
import sys
from pathlib import Path
import traceback

import pandas as pd

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from datasets.waymo import WaymoDataset


def create_mock_waymo_dataset(tmp_path):
    """
    Creates a minimal mock waymo dataset structure for testing.
    Matches the actual download structure: root/split/component/
    """
    # define splits and components
    splits = ["train", "validation"]
    components = ["camera_image", "lidar", "camera_box"]
    sequence_ids = [
        "10017090168044687777_6380_000_6400_000",
        "10023947602400723454_1120_000_1140_000",
    ]
    
    for split in splits:
        for component in components:
            # create directory: tmp_path / split / component
            component_dir = tmp_path / split / component
            component_dir.mkdir(parents=True, exist_ok=True)
            
            # create mock parquet files for each sequence
            for seq_id in sequence_ids:
                # create dummy data
                if component == "camera_image":
                    df = pd.DataFrame({
                        "frame_id": range(10),
                        "camera_name": ["FRONT"] * 10,
                        "timestamp": range(1000000, 1000010),
                    })
                elif component == "lidar":
                    df = pd.DataFrame({
                        "frame_id": range(10),
                        "laser_name": ["TOP"] * 10,
                        "timestamp": range(1000000, 1000010),
                    })
                else:  # camera_box
                    df = pd.DataFrame({
                        "frame_id": range(5),
                        "box_id": range(100, 105),
                        "center_x": [1.0, 2.0, 3.0, 4.0, 5.0],
                        "center_y": [1.0, 2.0, 3.0, 4.0, 5.0],
                    })
                
                parquet_path = component_dir / f"{seq_id}.parquet"
                df.to_parquet(parquet_path)
    
    return tmp_path


def test_dataset_initialization():
    """Test that dataset initializes correctly"""
    tmp_path = Path("tmp_test_data")
    tmp_path.mkdir(exist_ok=True)
    
    try:
        mock_data = create_mock_waymo_dataset(tmp_path)
        dataset = WaymoDataset(
            root_dir=mock_data,
            split="train",
            component="camera_image"
        )
        
        assert len(dataset) > 0, "Dataset length should be > 0"
        assert len(dataset.parquet_files) == 2, f"Expected 2 parquet files, got {len(dataset.parquet_files)}"
        print("✓ test_dataset_initialization passed")
        
    finally:
        import shutil
        if tmp_path.exists():
            shutil.rmtree(tmp_path)


def test_dataset_splits():
    """Test that both train and validation splits work"""
    tmp_path = Path("tmp_test_data")
    tmp_path.mkdir(exist_ok=True)
    
    try:
        mock_data = create_mock_waymo_dataset(tmp_path)
        
        train_dataset = WaymoDataset(
            root_dir=mock_data,
            split="train",
            component="camera_image"
        )
        
        val_dataset = WaymoDataset(
            root_dir=mock_data,
            split="validation",
            component="camera_image"
        )
        
        assert len(train_dataset) == 20, f"Expected 20 train samples, got {len(train_dataset)}"
        assert len(val_dataset) == 20, f"Expected 20 val samples, got {len(val_dataset)}"
        print("✓ test_dataset_splits passed")
        
    finally:
        import shutil
        if tmp_path.exists():
            shutil.rmtree(tmp_path)


def test_dataset_getitem():
    """Test that we can retrieve items from the dataset"""
    tmp_path = Path("tmp_test_data")
    tmp_path.mkdir(exist_ok=True)
    
    try:
        mock_data = create_mock_waymo_dataset(tmp_path)
        dataset = WaymoDataset(
            root_dir=mock_data,
            split="train",
            component="camera_image"
        )
        
        sample = dataset[0]
        
        assert isinstance(sample, dict), f"Sample should be dict, got {type(sample)}"
        assert "frame_id" in sample, "Sample should contain 'frame_id'"
        assert "camera_name" in sample, "Sample should contain 'camera_name'"
        assert "timestamp" in sample, "Sample should contain 'timestamp'"
        print("✓ test_dataset_getitem passed")
        
    finally:
        import shutil
        if tmp_path.exists():
            shutil.rmtree(tmp_path)


def test_dataset_length():
    """Test that dataset length is correct"""
    tmp_path = Path("tmp_test_data")
    tmp_path.mkdir(exist_ok=True)
    
    try:
        mock_data = create_mock_waymo_dataset(tmp_path)
        dataset = WaymoDataset(
            root_dir=mock_data,
            split="train",
            component="camera_image"
        )
        
        # 2 sequences × 10 frames each = 20 total
        assert len(dataset) == 20, f"Expected 20 samples, got {len(dataset)}"
        print("✓ test_dataset_length passed")
        
    finally:
        import shutil
        if tmp_path.exists():
            shutil.rmtree(tmp_path)


def test_dataset_sequence_filtering():
    """Test filtering by sequence IDs"""
    tmp_path = Path("tmp_test_data")
    tmp_path.mkdir(exist_ok=True)
    
    try:
        mock_data = create_mock_waymo_dataset(tmp_path)
        dataset = WaymoDataset(
            root_dir=mock_data,
            split="train",
            component="camera_image",
            sequence_ids=["10017090168044687777_6380_000_6400_000"]
        )
        
        # only 1 sequence × 10 frames = 10 total
        assert len(dataset) == 10, f"Expected 10 samples, got {len(dataset)}"
        print("✓ test_dataset_sequence_filtering passed")
        
    finally:
        import shutil
        if tmp_path.exists():
            shutil.rmtree(tmp_path)


def test_dataset_different_components():
    """Test loading different components"""
    tmp_path = Path("tmp_test_data")
    tmp_path.mkdir(exist_ok=True)
    
    try:
        mock_data = create_mock_waymo_dataset(tmp_path)
        camera_dataset = WaymoDataset(
            root_dir=mock_data,
            split="train",
            component="camera_image"
        )
        
        lidar_dataset = WaymoDataset(
            root_dir=mock_data,
            split="train",
            component="lidar"
        )
        
        assert len(camera_dataset) == 20, f"Expected 20 camera samples, got {len(camera_dataset)}"
        assert len(lidar_dataset) == 20, f"Expected 20 lidar samples, got {len(lidar_dataset)}"
        print("✓ test_dataset_different_components passed")
        
    finally:
        import shutil
        if tmp_path.exists():
            shutil.rmtree(tmp_path)


def test_dataset_invalid_split():
    """Test that invalid split raises error"""
    tmp_path = Path("tmp_test_data")
    tmp_path.mkdir(exist_ok=True)
    
    try:
        mock_data = create_mock_waymo_dataset(tmp_path)
        
        error_raised = False
        try:
            WaymoDataset(
                root_dir=mock_data,
                split="invalid_split",
                component="camera_image"
            )
        except ValueError as e:
            if "Component directory not found" in str(e):
                error_raised = True
        
        assert error_raised, "Should raise ValueError for invalid split"
        print("✓ test_dataset_invalid_split passed")
        
    finally:
        import shutil
        if tmp_path.exists():
            shutil.rmtree(tmp_path)


def test_dataset_invalid_component():
    """Test that invalid component raises error"""
    tmp_path = Path("tmp_test_data")
    tmp_path.mkdir(exist_ok=True)
    
    try:
        mock_data = create_mock_waymo_dataset(tmp_path)
        
        error_raised = False
        try:
            WaymoDataset(
                root_dir=mock_data,
                split="train",
                component="invalid_component"
            )
        except ValueError as e:
            if "Component directory not found" in str(e):
                error_raised = True
        
        assert error_raised, "Should raise ValueError for invalid component"
        print("✓ test_dataset_invalid_component passed")
        
    finally:
        import shutil
        if tmp_path.exists():
            shutil.rmtree(tmp_path)


def test_dataset_index_out_of_range():
    """Test that out of range index raises error"""
    tmp_path = Path("tmp_test_data")
    tmp_path.mkdir(exist_ok=True)
    
    try:
        mock_data = create_mock_waymo_dataset(tmp_path)
        dataset = WaymoDataset(
            root_dir=mock_data,
            split="train",
            component="camera_image"
        )
        
        error_raised = False
        try:
            _ = dataset[1000]
        except IndexError:
            error_raised = True
        
        assert error_raised, "Should raise IndexError for out of range index"
        print("✓ test_dataset_index_out_of_range passed")
        
    finally:
        import shutil
        if tmp_path.exists():
            shutil.rmtree(tmp_path)


def test_get_sequence_ids():
    """Test retrieving sequence IDs"""
    tmp_path = Path("tmp_test_data")
    tmp_path.mkdir(exist_ok=True)
    
    try:
        mock_data = create_mock_waymo_dataset(tmp_path)
        dataset = WaymoDataset(
            root_dir=mock_data,
            split="train",
            component="camera_image"
        )
        
        seq_ids = dataset.get_sequence_ids()
        
        assert len(seq_ids) == 2, f"Expected 2 sequence IDs, got {len(seq_ids)}"
        assert "10017090168044687777_6380_000_6400_000" in seq_ids, "Missing expected sequence ID"
        print("✓ test_get_sequence_ids passed")
        
    finally:
        import shutil
        if tmp_path.exists():
            shutil.rmtree(tmp_path)


def test_get_component_schema():
    """Test retrieving component schema"""
    tmp_path = Path("tmp_test_data")
    tmp_path.mkdir(exist_ok=True)
    
    try:
        mock_data = create_mock_waymo_dataset(tmp_path)
        dataset = WaymoDataset(
            root_dir=mock_data,
            split="train",
            component="camera_image"
        )
        
        schema = dataset.get_component_schema()
        
        assert "frame_id" in schema, "Schema should contain 'frame_id'"
        assert "camera_name" in schema, "Schema should contain 'camera_name'"
        assert "timestamp" in schema, "Schema should contain 'timestamp'"
        print("✓ test_get_component_schema passed")
        
    finally:
        import shutil
        if tmp_path.exists():
            shutil.rmtree(tmp_path)


def run_all_tests():
    """Run all tests and report results"""
    tests = [
        test_dataset_initialization,
        test_dataset_splits,
        test_dataset_getitem,
        test_dataset_length,
        test_dataset_sequence_filtering,
        test_dataset_different_components,
        test_dataset_invalid_split,
        test_dataset_invalid_component,
        test_dataset_index_out_of_range,
        test_get_sequence_ids,
        test_get_component_schema,
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