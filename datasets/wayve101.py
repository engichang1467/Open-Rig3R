import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
import struct

from torchvision import transforms

class Wayve101Dataset(Dataset):
    def __init__(self, root_dir, subset='train', n_frames=2, image_size=(384,384),
                 transforms=None, use_masks=False, metadata_dropout=0.0):
        """
        Wayve101 Dataset

        Args:
            root_dir (str): path to WayveScenes101 dataset root.
            subset (str): train/val/test split (optional, placeholder).
            n_frames (int): number of frames to sample per camera.
            image_size (tuple): output image size (H, W).
            transforms: torchvision transforms.
            use_masks (bool): whether to load masks.
            metadata_dropout (float): dropout for cam2rig metadata.
        """
        self.root_dir = root_dir
        self.n_frames = n_frames
        self.transforms = transforms
        self.image_size = image_size
        self.use_masks = use_masks
        self.metadata_dropout = metadata_dropout

        self.camera_dirs = ['front-forward', 'left-backward', 'left-forward', 'right-backward', 'right-forward']

        # 1. Load sequences
        self.samples = [os.path.join(root_dir, seq) for seq in os.listdir(root_dir)
                        if os.path.isdir(os.path.join(root_dir, seq))]

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, image_files):
        if len(image_files) <= self.n_frames:
            return image_files + [image_files[-1]] * (self.n_frames - len(image_files))
        else:
            return sorted(random.sample(image_files, self.n_frames))

    def _load_images(self, seq_path):
        images = []
        for cam in self.camera_dirs:
            cam_dir = os.path.join(seq_path, 'images', cam)
            img_files = sorted(os.listdir(cam_dir))
            selected_files = self._sample_frames(img_files)
            for f in selected_files:
                img = Image.open(os.path.join(cam_dir, f)).convert('RGB')
                if self.transforms:
                    img = self.transforms(img)
                else:
                    img = img.resize(self.image_size)
                    img = transforms.ToTensor()(img)  # converts to [0,1], shape (3,H,W)
                images.append(img)
        return torch.stack(images)  # (N, 3, H, W)

    def _load_masks(self, seq_path):
        if not self.use_masks:
            # Return zeros with the expected shape: (num_cameras*n_frames,1,H,W)
            num_images = len(self.camera_dirs) * self.n_frames
            return torch.zeros(num_images, 1, *self.image_size)

        masks = []
        for cam in self.camera_dirs:
            mask_dir = os.path.join(seq_path, 'masks', cam)
            mask_files = sorted(os.listdir(mask_dir))
            selected_files = self._sample_frames(mask_files)
            for f in selected_files:
                mask = Image.open(os.path.join(mask_dir, f)).convert('L')

                mask = mask.resize(self.image_size)
                mask = transforms.ToTensor()(mask)  # converts to [0,1], shape (1,H,W)

                masks.append(mask)
        return torch.stack(masks)  # (N, 1, H, W)

    def _read_colmap_extrinsics(self, seq_path):
        """
        Read images.bin from COLMAP to get extrinsics (R, t) for each image.
        Returns:
            dict: filename -> {'R': (3,3), 't': (3,)} (World-to-Camera)
        """
        images_bin = os.path.join(seq_path, 'colmap_sparse', 'rig', 'images.bin')
        if not os.path.exists(images_bin):
            return {}

        extrinsics = {}
        with open(images_bin, 'rb') as f:
            num_reg_images = struct.unpack('<Q', f.read(8))[0]
            for _ in range(num_reg_images):
                # Image ID, Qw, Qx, Qy, Qz, tx, ty, tz, Camera ID, Name
                binary_image_header = f.read(64)
                image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = struct.unpack('<IdddddddI', binary_image_header)
                
                name = ""
                char = f.read(1)
                while char != b'\x00':
                    name += char.decode('utf-8')
                    char = f.read(1)
                
                # Skip 2D points
                num_points2D = struct.unpack('<Q', f.read(8))[0]
                f.read(num_points2D * 24) # x, y, point3D_id (2 doubles + 1 uint64 = 16 + 8 = 24 bytes)

                # Convert Quaternion to Rotation Matrix
                # scipy Rotation uses (x, y, z, w)
                from scipy.spatial.transform import Rotation as R
                rot = R.from_quat([qx, qy, qz, qw])
                R_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32)
                t_vec = torch.tensor([tx, ty, tz], dtype=torch.float32)
                
                extrinsics[name] = {'R': R_mat, 't': t_vec}
                
        return extrinsics

    def _load_metadata(self, seq_path):
        # Load real extrinsics from COLMAP
        colmap_extrinsics = self._read_colmap_extrinsics(seq_path)
        
        # We need to match the sampled images to the COLMAP data
        # _load_images samples frames but doesn't store filenames. 
        # We need to replicate the sampling logic or modify _load_images to return filenames.
        # For now, let's duplicate the sampling logic here to find the filenames.
        
        cam2rig_list = []
        
        for cam in self.camera_dirs:
            cam_dir = os.path.join(seq_path, 'images', cam)
            img_files = sorted(os.listdir(cam_dir))
            selected_files = self._sample_frames(img_files)
            
            for f in selected_files:
                # Filename in COLMAP usually includes the folder structure relative to the image path used in COLMAP
                # In Wayve101, it might be "images/front-forward/00001.jpg" or just "front-forward/00001.jpg"
                # We check a few variants
                
                # Variant 1: "images/cam/file"
                name_v1 = os.path.join('images', cam, f)
                # Variant 2: "cam/file"
                name_v2 = os.path.join(cam, f)
                
                pose = None
                if name_v1 in colmap_extrinsics:
                    pose = colmap_extrinsics[name_v1]
                elif name_v2 in colmap_extrinsics:
                    pose = colmap_extrinsics[name_v2]
                
                if pose is None:
                    # Fallback to identity if not found
                    cam2rig_list.append(torch.eye(4))
                else:
                    # COLMAP gives World-to-Camera: P_c = R P_w + t
                    # We want Camera-to-Rig (assuming World == Rig): P_r = R_inv (P_c - t)
                    # T_c2r = [R^T | -R^T t]
                    R_wc = pose['R']
                    t_wc = pose['t']
                    
                    R_cr = R_wc.T
                    t_cr = -torch.matmul(R_cr, t_wc)
                    
                    T_cr = torch.eye(4)
                    T_cr[:3, :3] = R_cr
                    T_cr[:3, 3] = t_cr
                    
                    cam2rig_list.append(T_cr)

        cam2rig = torch.stack(cam2rig_list) # (N, 4, 4)
        
        # Apply metadata dropout
        if self.metadata_dropout > 0:
            for i in range(cam2rig.shape[0]):
                if random.random() < self.metadata_dropout:
                    cam2rig[i] = torch.eye(4)
        
        return {'cam2rig': cam2rig}

    def _load_pointcloud(self, seq_path):
        """
        Load pointcloud from colmap_sparse/rig/points3D.bin
        COLMAP binary format: https://colmap.github.io/format.html
        Returns:
            torch.Tensor (N_points, 3)
        """
        bin_file = os.path.join(seq_path, 'colmap_sparse', 'rig', 'points3D.bin')
        if not os.path.exists(bin_file):
            return torch.empty(0,3)

        def read_bin_point3D(file_path):
            points = []
            with open(file_path, 'rb') as f:
                header = f.read(8)
                if len(header) < 8:
                    return torch.empty(0, 3)
                # header
                num_points = struct.unpack('<Q', f.read(8))[0]
                for _ in range(num_points):
                    data = f.read(8 + 24 + 3 + 8 + 8)  # point_id + xyz + rgb + error + track_len
                    if len(data) < 51:  # 8+24+3+8+8=51
                        break  # EOF reached unexpectedly
                    
                    point_id = struct.unpack('<Q', data[:8])[0]
                    xyz = struct.unpack('<ddd', data[8:32])
                    track_len = struct.unpack('<Q', data[43:51])[0]

                    # Read & discard track data in small chunks to avoid invalid seek
                    bytes_to_skip = track_len * 16
                    chunk_size = 1024 * 1024  # 1 MB chunks
                    while bytes_to_skip > 0:
                        skip = min(bytes_to_skip, chunk_size)
                        read_bytes = f.read(skip)
                        if len(read_bytes) < skip:
                            break
                        bytes_to_skip -= len(read_bytes)


                    points.append(xyz)
            return torch.tensor(points, dtype=torch.float32)
        
        return read_bin_point3D(bin_file)

    def __getitem__(self, idx):
        seq_path = self.samples[idx]
        seq_path = os.path.dirname(seq_path) + "/" # cd
        images = self._load_images(seq_path)
        masks = self._load_masks(seq_path)
        metadata = self._load_metadata(seq_path)
        pointcloud = self._load_pointcloud(seq_path)
 
        return {
            'images': images,             # (N,3,H,W)
            'masks': masks,               # (N,1,H,W) or None
            'metadata': metadata,         # dict with cam2rig
            'pointcloud': pointcloud,     # (M,3)
            'seq_path': seq_path
        }
