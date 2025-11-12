import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
        Combines pointmap, pose-raymap, and rig-raymap losses with fixed scalar weights.
    """
    def __init__(self, w_point=1.0, w_pose=1.0, w_rig=1.0, reduction='mean'):
        super().__init__()
        self.w_point = w_point
        self.w_pose = w_pose
        self.w_rig = w_rig
        self.reduction = reduction

    def forward(self, preds, gts):
        """
            Args:
                preds: dict with keys ['pointmap', 'pose_raymap', 'rig_raymap']
                gts:   dict with same keys
            Returns:
                total_loss, loss_dict
        """

        loss_dict = {}

        # ==========================================================
        # 1. Confidence‑weighted Pointmap loss (L2)
        # gts must include: 'pointmap' and 'pointmap_conf'
        # ==========================================================
        if 'pointmap' in preds and 'pointmap' in gts:
            point_pred = preds['pointmap']
            point_gt = gts['pointmap']
            if 'pointmap_conf' in gts:
                conf = gts['pointmap_conf'] # shape (B, V, P) or (B, V, P, 1)
                if conf.dim() == 3:
                    conf = conf.unsqueeze(-1)
                loss_point = conf * (point_pred - point_gt) ** 2
                loss_point = self._reduce(loss_point)
            else:
                loss_point = F.mse_loss(point_pred, point_gt, reduction=self.reduction)
            loss_dict['pointmap'] = loss_point
        else:
            loss_point = 0.0


        # ==========================================================
        # 2. Pose raymap loss = direction loss + camera center loss
        # ==========================================================
        if 'pose_raymap' in preds and 'pose_raymap' in gts:
            # ---- Direction loss (cosine) ----
            dir_pred = preds['pose_raymap']
            dir_gt = gts['pose_raymap']
            loss_dir_pose = 1.0 - F.cosine_similarity(dir_pred, dir_gt, dim=-1)
            loss_dir_pose = self._reduce(loss_dir_pose)

            # ---- Camera center loss (L2) ----
            if 'camera_center_pose' in preds and 'camera_center_pose' in gts:
                cc_pred = preds['camera_center_pose']
                cc_gt = gts['camera_center_pose']
                loss_cc_pose = F.mse_loss(cc_pred, cc_gt, reduction=self.reduction)
            else:
                loss_cc_pose = 0.0

            loss_pose = loss_dir_pose + loss_cc_pose
            loss_dict['pose_raymap'] = loss_pose
        else:
            loss_pose = 0.0


        # ==========================================================
        # 3. Rig raymap loss = direction loss + camera center loss
        # ==========================================================
        if 'rig_raymap' in preds and 'rig_raymap' in gts:
            # ---- Direction loss ----
            dir_pred = preds['rig_raymap']
            dir_gt = gts['rig_raymap']
            loss_dir_rig = 1.0 - F.cosine_similarity(dir_pred, dir_gt, dim=-1)
            loss_dir_rig = self._reduce(loss_dir_rig)


            # ---- Camera center loss ----
            if 'camera_center_rig' in preds and 'camera_center_rig' in gts:
                cc_pred = preds['camera_center_rig']
                cc_gt = gts['camera_center_rig']
                loss_cc_rig = F.mse_loss(cc_pred, cc_gt, reduction=self.reduction)
            else:
                loss_cc_rig = 0.0

            loss_rig = loss_dir_rig + loss_cc_rig
            loss_dict['rig_raymap'] = loss_rig
        else:
            loss_rig = 0.0

        # # --- Pointmap loss (L2) ---
        # if 'pointmap' in preds and 'pointmap' in gts:
        #     loss_point = F.mse_loss(preds['pointmap'], gts['pointmap'], reduction=self.reduction)
        #     loss_dict['pointmap'] = loss_point
        # else:
        #     loss_point = 0.0

        # # --- Pose-raymap loss (cosine / angular) ---
        # if 'pose_raymap' in preds and 'pose_raymap' in gts:
        #     loss_pose = 1.0 - F.cosine_similarity(
        #         preds['pose_raymap'], gts['pose_raymap'], dim=-1
        #     )
        #     loss_pose = self._reduce(loss_pose)
        #     loss_dict['pose_raymap'] = loss_pose
        # else:
        #     loss_pose = 0.0

        # # --- Rig-raymap loss (cosine / angular) ---
        # if 'rig_raymap' in preds and 'rig_raymap' in gts:
        #     loss_rig = 1.0 - F.cosine_similarity(
        #         preds['rig_raymap'], gts['rig_raymap'], dim=-1
        #     )
        #     loss_rig = self._reduce(loss_rig)
        #     loss_dict['rig_raymap'] = loss_rig
        # else:
        #     loss_rig = 0.0

        # --- Combine ---
        total = (
            self.w_point * loss_point +
            self.w_pose  * loss_pose +
            self.w_rig   * loss_rig
        )
        loss_dict['total'] = total
        return total, loss_dict
    
    def _reduce(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
