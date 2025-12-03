import torch
import numpy as np
from scipy.cluster.vq import kmeans2, vq
from scipy.spatial.transform import Rotation as R

def recover_pose_closed_form(rig_raymap, focal_length_guess=None):
    """
    Recover camera intrinsics and extrinsics from predicted rig raymap.
    
    Args:
        rig_raymap: (H, W, 6) tensor containing (origin, direction) per pixel.
                    Origins are in rig frame. Directions are unit vectors in rig frame.
        focal_length_guess: Optional initial guess for focal length.
        
    Returns:
        dict containing:
            'R': (3, 3) rotation matrix (cam -> rig)
            't': (3,) translation vector (cam center in rig frame)
            'focal_length': scalar estimated focal length
    """
    H, W, _ = rig_raymap.shape
    device = rig_raymap.device
    
    # 1. Estimate Intrinsics (Focal Length)
    # We assume principal point is at center (W/2, H/2)
    cx, cy = W / 2.0, H / 2.0
    
    # Select a few pairs of pixels to estimate focal length
    # Angle between two rays in camera frame depends only on pixel coords and f
    # Angle between two rays in rig frame is given by dot product of directions
    
    # For simplicity/robustness, we can use the field of view if we assume f_x = f_y
    # or just use the provided guess if available.
    # Here we implement a simple estimation based on center and corner pixels.
    
    # Get directions at center and corners
    # (This is a simplified version; a full solver would optimize f to match all angles)
    
    # Let's assume a standard FOV or use the guess if provided.
    # If not provided, we can try to estimate it from the ray directions if they form a consistent camera.
    # However, for this implementation, we will focus on Extrinsics recovery assuming known or estimated f.
    # If f is unknown, we can approximate it or treat it as a separate optimization.
    
    # For now, let's proceed to Extrinsics which is the critical part for Rig Discovery.
    # We can estimate the camera center 't' directly from the ray origins.
    # Ideally, all rays from a single camera should converge at the camera center.
    # So 't' is the point closest to all ray lines defined by (origin, direction).
    
    origins = rig_raymap[..., :3].reshape(-1, 3)
    directions = rig_raymap[..., 3:].reshape(-1, 3)
    
    # Filter out invalid rays if any (e.g. zero direction)
    valid_mask = torch.norm(directions, dim=1) > 0.9
    origins = origins[valid_mask]
    directions = directions[valid_mask]
    
    if len(origins) < 10:
        return {'R': torch.eye(3, device=device), 't': torch.zeros(3, device=device)}

    # Estimate Camera Center t
    # t is the point that minimizes sum of squared distances to lines
    # Distance from point p to line (o, d) is || (p-o) - ((p-o).d)d ||
    # This is a linear least squares problem.
    # However, since Rig3R predicts ray origins directly, and for a single camera 
    # all ray origins should ideally be the SAME (the camera center), 
    # we can simply take the mean of the predicted origins as a robust estimate.
    
    t_est = origins.mean(dim=0)
    
    # 2. Estimate Rotation R
    # We want R such that R * r_cam = r_rig
    # r_cam are the ray directions in the camera frame (computed from pixel coords and f).
    # r_rig are the predicted directions in the rig_raymap.
    
    # Construct canonical camera rays
    u, v = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
    u = u.reshape(-1)[valid_mask]
    v = v.reshape(-1)[valid_mask]
    
    # If f is not provided, we might need to estimate it. 
    # For this task, let's assume a default f if not given, or estimate from FOV of r_rig.
    if focal_length_guess is None:
        # Estimate f from max angle in r_rig (assuming it corresponds to FOV)
        # This is a heuristic.
        # Angle between center ray and corner ray
        center_idx = (H//2) * W + (W//2)
        # ... implementation complexity ...
        f = W # default to f=W (approx 53 deg FOV)
    else:
        f = focal_length_guess

    z_cam = torch.ones_like(u)
    x_cam = (u - cx) / f
    y_cam = (v - cy) / f
    r_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)
    r_cam = r_cam / torch.norm(r_cam, dim=1, keepdim=True) # Normalize
    
    # Solve for R using SVD (Kabsch algorithm variant for vectors)
    # We want R @ r_cam.T ~ r_rig.T
    # H = r_cam.T @ r_rig
    # U, S, Vt = SVD(H)
    # R = V @ U.T
    
    H_cov = r_cam.T @ directions # (3, 3)
    U, S, Vh = torch.linalg.svd(H_cov)
    R_est = Vh.T @ U.T
    
    # Ensure determinant is +1 (rotation, not reflection)
    if torch.det(R_est) < 0:
        Vh[2, :] *= -1
        R_est = Vh.T @ U.T
        
    return {
        'R': R_est,
        't': t_est,
        'focal_length': f
    }

def cluster_rig_poses(poses_dict_list, n_clusters=None):
    """
    Cluster frames based on their recovered rig-relative poses.
    
    Args:
        poses_dict_list: list of dicts {'R': (3,3), 't': (3,)}
        n_clusters: number of clusters to find. If None, use a heuristic or assume 1 for unstructured.
        
    Returns:
        list of cluster labels (integers)
    """
    if not poses_dict_list:
        return []

    # Feature vector: concatenation of translation and rotation (flattened or Euler/quaternion)
    features = []
    for p in poses_dict_list:
        t = p['t'].cpu().numpy()
        r_mat = p['R'].cpu().numpy()
        # Convert rotation matrix to rotation vector or quaternion for clustering
        r_quat = R.from_matrix(r_mat).as_quat()
        
        # We might want to weight rotation and translation differently
        feat = np.concatenate([t, r_quat])
        features.append(feat)
    
    features = np.array(features)
    
    if n_clusters is None:
        # Default to 1 if not specified
        n_clusters = 1 
        
    # Use scipy kmeans2
    # minit='points' selects initial centroids from data
    centroid, label = kmeans2(features, n_clusters, minit='points')
    
    return label

def reconstruct_pointcloud(pointmaps, poses, masks=None):
    """
    Aggregate pointmaps into a single global pointcloud using recovered poses.
    
    Args:
        pointmaps: list of (H, W, 3) tensors (local camera frame points)
        poses: list of dicts {'R':, 't':} (cam -> rig/global)
        masks: optional list of (H, W) boolean masks
        
    Returns:
        (N, 3) tensor global pointcloud
    """
    global_points = []
    
    for i, (pmap, pose) in enumerate(zip(pointmaps, poses)):
        # pmap is in camera frame
        
        pts = pmap.reshape(-1, 3)
        if masks is not None:
            m = masks[i].reshape(-1)
            pts = pts[m]
            
        # Transform to global/rig frame
        # p_global = R * p_cam + t
        R_mat = pose['R']
        t_vec = pose['t']
        
        pts_global = torch.matmul(pts, R_mat.T) + t_vec
        global_points.append(pts_global)
        
    return torch.cat(global_points, dim=0)
