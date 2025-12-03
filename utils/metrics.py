# utils/metrics.py
import torch
from scipy.optimize import linear_sum_assignment

def chamfer_distance(pc1, pc2):
    """
    Compute Chamfer Distance between two point clouds.

    Args:
        pc1: (N1, 3) tensor
        pc2: (N2, 3) tensor
    Returns:
        scalar tensor
    """
    if pc1.numel() == 0 or pc2.numel() == 0:
        return torch.tensor(0.0, device=pc1.device)

    pc1 = pc1.unsqueeze(0)  # (1, N1, 3)
    pc2 = pc2.unsqueeze(0)  # (1, N2, 3)

    diff = torch.cdist(pc1, pc2)  # (1, N1, N2)
    dist1 = diff.min(dim=2)[0].mean()
    dist2 = diff.min(dim=1)[0].mean()
    return dist1 + dist2


def rig_discovery_accuracy(pred_pc, gt_pc):
    """
    Evaluate Rig Discovery Accuracy using Hungarian matching.

    Args:
        pred_pc: (N, 3) predicted rig keypoints
        gt_pc: (N, 3) ground truth rig keypoints

    Returns:
        fraction of correctly matched points
    """
    if pred_pc.numel() == 0 or gt_pc.numel() == 0:
        return torch.tensor(0.0, device=pred_pc.device)

    # Compute distance matrix
    dist_matrix = torch.cdist(pred_pc.unsqueeze(0), gt_pc.unsqueeze(0))[0].cpu().numpy()  # (N_pred, N_gt)

    # Hungarian matching (minimize total distance)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    # Define a threshold for correct match (e.g., 0.1 meters)
    threshold = 0.1
    correct = (dist_matrix[row_ind, col_ind] < threshold).sum()

    acc = correct / len(gt_pc)
    return torch.tensor(acc, device=pred_pc.device)


def rig_maa(pred_poses, gt_poses):
    """
    Compute Rig Mean Angular Accuracy (mAA).
    
    Args:
        pred_poses: list of dicts {'R': (3,3)}
        gt_poses: list of dicts {'R': (3,3)}
        
    Returns:
        scalar tensor (mean angular error in degrees)
    """
    if len(pred_poses) != len(gt_poses):
        return torch.tensor(0.0)
    
    angular_errors = []
    for pred, gt in zip(pred_poses, gt_poses):
        R_pred = pred['R']
        R_gt = gt['R']
        
        # Relative rotation: R_rel = R_pred @ R_gt.T
        R_rel = torch.matmul(R_pred, R_gt.T)
        
        # Trace of R is 1 + 2cos(theta)
        trace = torch.trace(R_rel)
        cos_theta = (trace - 1) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        theta = torch.acos(cos_theta) # radians
        
        angular_errors.append(torch.rad2deg(theta))
        
    return torch.stack(angular_errors).mean()
