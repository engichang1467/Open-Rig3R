"""Loss-independent evaluation metrics.

A loss value is only comparable to itself. The moment the objective changes - Eq. 3
unsquaring the pointmap error, Eq. 4 replacing cosine with a ray norm, z-bar rescaling
the targets - the number changes units and two runs stop being comparable even at a
fixed seed. These metrics answer the same geometric question regardless of what the
model was trained to minimize, so they survive an A/B across a loss change.
"""

import torch


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
