import torch
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from models.losses import MultiTaskLoss

def test_loss_smoke():
    B, V, P = 2, 3, 16 # batch, views, patches


    # Fake predictions
    preds = {
        'pointmap': torch.randn(B, V, P, 3),
        'pose_raymap': torch.randn(B, V, P, 3),
        'rig_raymap': torch.randn(B, V, P, 3),


        # Camera centers
        'camera_center_pose': torch.randn(B, V, 3),
        'camera_center_rig': torch.randn(B, V, 3),
    }


    # Fake ground-truth
    gts = {
        'pointmap': torch.randn(B, V, P, 3),
        'pointmap_conf': torch.rand(B, V, P), # confidence weights


        'pose_raymap': torch.randn(B, V, P, 3),
        'rig_raymap': torch.randn(B, V, P, 3),


        # Camera centers
        'camera_center_pose': torch.randn(B, V, 3),
        'camera_center_rig': torch.randn(B, V, 3),
    }


    # Normalize GT ray directions
    gts['pose_raymap'] = torch.nn.functional.normalize(gts['pose_raymap'], dim=-1)
    gts['rig_raymap'] = torch.nn.functional.normalize(gts['rig_raymap'], dim=-1)


    # Run loss
    criterion = MultiTaskLoss()
    total, loss_dict = criterion(preds, gts)


    print("Total loss:", total.item())
    for k, v in loss_dict.items():
        print(f"{k}: {v}")


    # assert total.requires_grad, "Loss should require grad"
    assert not torch.isnan(total), "Loss contains NaNs"

if __name__ == "__main__":
    test_loss_smoke()