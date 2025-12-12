import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class WaymoDataset(Dataset):
    """
    Dataset class for Waymo Open Dataset stored in Parquet format.
    
    Args:
        root_dir: Root directory containing the waymo_mini folder structure
        split: Which split to load ('train' or 'validation')
        component: Which component to load (e.g., 'camera_image', 'lidar', 'camera_box')
        sequence_ids: Optional list of sequence IDs to filter. If None, loads all sequences.
        n_frames: Number of frames to sample per sequence
    """
    
    def __init__(
        self, 
        root_dir: str,
        split: str = "train",
        component: str = "camera_image",
        sequence_ids: Optional[List[str]] = None,
        n_frames: int = 2
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.component = component
        self.n_frames = n_frames
        
        # build path: root_dir / split / component
        self.component_dir = self.root_dir / split / component
        
        if not self.component_dir.exists():
            raise ValueError(
                f"Component directory not found: {self.component_dir}\n"
                f"Expected structure: {self.root_dir}/{split}/{component}/"
            )
        
        # find all parquet files in the component directory
        self.parquet_files = sorted(list(self.component_dir.glob("*.parquet")))
        
        if len(self.parquet_files) == 0:
            raise ValueError(f"No parquet files found in {self.component_dir}")
        
        # filter by sequence_ids if provided
        if sequence_ids is not None:
            self.parquet_files = [
                f for f in self.parquet_files 
                if f.stem in sequence_ids
            ]
        
        # load all parquet files and group by sequence and timestamp
        self.sequences = []
        
        for parquet_file in self.parquet_files:
            df = pd.read_parquet(parquet_file)
            
            # group by timestamp to get frames
            if 'key.frame_timestamp_micros' in df.columns:
                grouped = df.groupby('key.frame_timestamp_micros')
                timestamps = list(grouped.groups.keys())
                
                # store sequence info
                self.sequences.append({
                    'dataframe': df,
                    'timestamps': sorted(timestamps),
                    'sequence_id': parquet_file.stem
                })
        
        self.total_length = sum(
            max(0, len(seq['timestamps']) - self.n_frames + 1) 
            for seq in self.sequences
        )
    
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx: int) -> Dict:
        if idx < 0 or idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range [0, {self.total_length})")
        
        # find which sequence contains this index
        cumulative = 0
        for seq in self.sequences:
            seq_length = max(0, len(seq['timestamps']) - self.n_frames + 1)
            if idx < cumulative + seq_length:
                # found the sequence
                frame_start_idx = idx - cumulative
                timestamps = seq['timestamps'][frame_start_idx:frame_start_idx + self.n_frames]
                
                # extract frames for these timestamps
                frames = []
                for ts in timestamps:
                    frame_data = seq['dataframe'][
                        seq['dataframe']['key.frame_timestamp_micros'] == ts
                    ].iloc[0].to_dict()
                    frames.append(frame_data)
                
                return {
                    'frames': frames,
                    'sequence_id': seq['sequence_id'],
                    'timestamps': timestamps
                }
            cumulative += seq_length
        
        raise IndexError(f"Index {idx} not found in sequences")
    
    def get_sequence_ids(self) -> List[str]:
        """Returns list of all sequence IDs in the dataset"""
        return [seq['sequence_id'] for seq in self.sequences]
    
    def get_component_schema(self) -> Dict:
        """Returns the schema (column names and types) of the component"""
        if len(self.sequences) > 0:
            return dict(self.sequences[0]['dataframe'].dtypes)
        return {}