# utils/metrics.py
import torch
from scipy.optimize import linear_sum_assignment

def chamfer_distance(pc1, pc2, chunk_size=1024):
    """
    Compute Chamfer Distance between two point clouds.

    Args:
        pc1: (N1, 3) tensor
        pc2: (N2, 3) tensor
        chunk_size: rows of pc1 held in the distance matrix at once
    Returns:
        scalar tensor, or nan if either cloud is empty
    """
    # nan, not 0.0: a zero here is indistinguishable from a perfect score, so an
    # empty prediction used to read as a flawless reconstruction.
    if pc1.numel() == 0 or pc2.numel() == 0:
        return torch.tensor(float('nan'), device=pc1.device)

    pc1 = pc1.reshape(-1, 3)
    pc2 = pc2.reshape(-1, 3)

    # A full cdist is N1 x N2 floats - 163840 predicted points against 144005
    # ground-truth ones is 94 GB. Chunking over pc1 keeps peak memory at
    # chunk_size x N2 while giving the exact same answer as the dense version:
    # min over pc2 is per-chunk, min over pc1 is a running min across chunks.
    min_over_pc2 = []
    min_over_pc1 = torch.full((pc2.shape[0],), float('inf'),
                              device=pc2.device, dtype=pc2.dtype)

    for start in range(0, pc1.shape[0], chunk_size):
        block = torch.cdist(pc1[start:start + chunk_size], pc2)  # (chunk, N2)
        min_over_pc2.append(block.min(dim=1)[0])
        min_over_pc1 = torch.minimum(min_over_pc1, block.min(dim=0)[0])

    return torch.cat(min_over_pc2).mean() + min_over_pc1.mean()


def rig_discovery_accuracy(pred_pc, gt_pc):
    """
    Evaluate Rig Discovery Accuracy using Hungarian matching.

    Args:
        pred_pc: (N, 3) predicted rig keypoints
        gt_pc: (N, 3) ground truth rig keypoints

    Returns:
        fraction of correctly matched points, or nan if either input is empty
    """
    # nan rather than 0.0 - see chamfer_distance. Zero is a legitimate score here,
    # so it must not double as "there was nothing to score".
    if pred_pc.numel() == 0 or gt_pc.numel() == 0:
        return torch.tensor(float('nan'), device=pred_pc.device)

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


# ---------------------------------------------------------------------------
# Loss-independent geometry metrics
#
# A loss value is only comparable to itself. The moment the objective changes -
# Eq. 3 unsquaring the pointmap error, Eq. 4 replacing cosine with a ray norm,
# z-bar rescaling the targets - the number changes units and two runs stop being
# comparable even at a fixed seed. These answer the same geometric question
# regardless of what the model was trained to minimize.
# ---------------------------------------------------------------------------


def align_scale(pred_points, gt_points, eps=1e-8):
    """Scale factor putting scale-normalized predictions back into metres. scalar tensor

    Eq. 3 divides only the ground truth by z-bar, so a trained model predicts geometry
    at normalized scale. Any metric in physical units - Chamfer distance, the 0.1 m
    match threshold in rig_discovery_accuracy - therefore needs one scale factor per
    sample, recovered from the ground truth. This is the usual scale-invariant
    evaluation protocol, not a fudge: the model is never asked to learn absolute scale.

    Median of the norms rather than a least-squares fit, because reconstructed point
    clouds carry outliers that would drag a mean.
    """
    if pred_points.numel() == 0 or gt_points.numel() == 0:
        return torch.tensor(1.0, device=gt_points.device)

    pred_scale = pred_points.reshape(-1, 3).norm(dim=-1).median()
    gt_scale = gt_points.reshape(-1, 3).norm(dim=-1).median()
    return gt_scale / pred_scale.clamp(min=eps)


def ray_angular_error(pred_raymap, gt_raymap):
    """Mean angle between predicted and ground-truth ray directions, in degrees.

    Both raymaps carry unit directions, so this needs no scale normalization and means
    the same thing on either side of a loss change - unlike the camera centre, which is
    metric in some arms and z-bar normalized in others.

    Args:
        pred_raymap, gt_raymap: (..., 6) centre + direction, or (..., 3) directions
    Returns:
        scalar tensor, degrees
    """
    pred = torch.nn.functional.normalize(pred_raymap[..., -3:].float(), dim=-1)
    gt = torch.nn.functional.normalize(gt_raymap[..., -3:].float(), dim=-1)

    # 2*atan2(|a-b|, |a+b|) rather than acos(a.b): acos is ill-conditioned near cos=1,
    # where its derivative diverges, and a converged model lives exactly there. The
    # acos form reports ~0.003 deg of noise on rays that are identical up to float32.
    angle = 2.0 * torch.atan2(
        (pred - gt).norm(dim=-1), (pred + gt).norm(dim=-1)
    )
    return torch.rad2deg(angle).mean()


def raymap_metrics(preds, targets):
    """Angular error for whichever raymaps this batch actually supervises. dict"""
    metrics = {}
    for name in ("pose", "rig"):
        key = f"{name}_raymap"
        if key in preds and key in targets:
            metrics[f"{name}_deg"] = ray_angular_error(preds[key], targets[key])
    return metrics
