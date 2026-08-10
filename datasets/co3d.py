import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import gzip

import random
import open3d as o3d 


class Co3DDataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        subset='train', 
        n_frames=24, 
        image_size=(384, 384),
        transforms=None
    ):
        self.root_dir = root_dir
        self.n_frames = n_frames
        self.transforms = transforms

        self.image_size = image_size

        # 1. Load split list
        self.samples = self._load_split(subset)

    def _sample_frames(self, image_files):
        """
        Sample self.n_frames images from the list of image files.
        If there are fewer images than n_frames, repeat or pad.
        """
        if len(image_files) <= self.n_frames:
            # repeat last frame if not enough images
            sampled = image_files + [image_files[-1]] * (self.n_frames - len(image_files))
            return sampled
        else:
            # randomly sample self.n_frames without replacement
            return sorted(random.sample(image_files, self.n_frames))
    
    def _load_split(self, subset):
        samples = []
        # iterate categories
        for category in os.listdir(self.root_dir):
            category_path = os.path.join(self.root_dir, category)
            set_list_file = os.path.join(category_path, f'set_lists/set_lists_manyview_test_0.json')
            if os.path.exists(set_list_file):
                with open(set_list_file, 'r') as f:
                    seq_list = json.load(f)
                    seq_list = seq_list[subset]

                for seq_name in seq_list:
                    seq_name = seq_name[0] # Get the sequence number from the 1st element


                    seq_path = os.path.join(category_path, seq_name)
                    if os.path.exists(seq_path):  # sanity check
                        samples.append(seq_path)
        
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_path = self.samples[idx]

        # print(f"samples: {self.samples}")
        # print(f"idx: {idx}")
        # print(f"seq_path: {seq_path}")

        # --- Load all images ---
        image_dir = os.path.join(seq_path, 'images')
        image_files = sorted(os.listdir(image_dir))
        selected_files = self._sample_frames(image_files)
        images = [Image.open(os.path.join(image_dir, f)).convert('RGB') for f in selected_files]

        if self.transforms:
            images = [self.transforms(img) for img in images]
        else:
            # Basic resize + to tensor
            images = [torch.tensor(img.resize(self.image_size)).permute(2,0,1).float()/255.0 for img in images]

        images = torch.stack(images)  # (N, 3, H, W)

        # --- Load frame metadata ---
        # Field dropout is applied to the embedding inside the decoder (sec 3.4), not
        # here - a masked value has to read as absent, and a substituted placeholder
        # reads as a specific wrong value instead.
        metadata = self._load_metadata(seq_path, selected_files)

        # --- Load pointcloud GT if exists ---
        pointcloud_file = os.path.join(seq_path, 'pointcloud.ply')
        pointcloud = None
        if os.path.exists(pointcloud_file):
            pcd = o3d.io.read_point_cloud(pointcloud_file)
            pointcloud = torch.tensor(pcd.points, dtype=torch.float32)
        else:
            pointcloud = torch.empty(0, 3)

        return {
            "images": images,
            "metadata": metadata,
            "pointcloud": pointcloud,
            "seq_path": seq_path
        }

    def _load_metadata(self, seq_path, selected_files):
        # load gzipped frame annotations
        frame_file = os.path.join(seq_path, 'frame_annotations.jgz')
        metadata = {}
        if os.path.exists(frame_file):
            with gzip.open(frame_file, 'rt') as f:
                frame_annots = json.load(f)
            cam2rigs = []
            for f in selected_files:
                if f in frame_annots:
                    cam2rigs.append(torch.tensor(frame_annots[f]['cam2rig'], dtype=torch.float32))
                else:
                    cam2rigs.append(torch.eye(4))
            metadata['cam2rig'] = torch.stack(cam2rigs)  # (V, 4, 4)

        # CO3D is one camera orbiting one object: no camera identity to convey and no
        # rig timestamps to normalize, so both of those slices stay empty.
        metadata['frame_index'] = torch.arange(len(selected_files))
        return metadata
