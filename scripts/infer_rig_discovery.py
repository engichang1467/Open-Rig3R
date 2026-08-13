import os
import yaml
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Add root to path
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from models.rig3r import Rig3R
from utils.metrics import align_scale, chamfer_distance, rig_discovery_accuracy, rig_maa
from utils.rig_discovery import recover_pose_closed_form, cluster_rig_poses, reconstruct_pointcloud
from datasets.wayve101 import Wayve101Dataset
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Infer Rig Discovery")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (YAML)")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Dataset
    # We use Wayve101 as in evaluate.py, or whatever is specified in config
    data_root = Path.cwd().joinpath(cfg["data"])
    n_frames = 2 # Fixed for this task/test
    image_size = (128, 128)
    
    # Note: Wayve101Dataset might need to be imported or mocked if not available in context, 
    # but based on evaluate.py it exists.
    dataset = Wayve101Dataset(root_dir=data_root,
                              n_frames=n_frames,
                              image_size=image_size,
                              transforms=None,
                              use_masks=False)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Loaded {len(dataset)} sequences.")

    # 2. Load Model
    model_ckpt = Path.cwd().joinpath(cfg["checkpoint"])
    model = Rig3R(
        img_size=image_size[0],
        patch_size=16,
        embed_dim=1024,
        num_decoder_layers=2,
        num_heads=8,
        mlp_dim=4096
    )
    
    # Strict, and no try/except: a checkpoint that fails to load used to fall through
    # to a randomly initialized model that still printed metrics, which is the same
    # trap as a broken pipeline reporting 0.000000. Let it raise instead.
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    print(f"Loaded model from {model_ckpt}")

    model.to(device)
    model.eval()
    
    # 3. Inference Loop
    all_chamfer = []
    all_rig_acc = []
    all_rig_maa = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Rig Discovery")):
        images = batch['images'].to(device) # (B, N, 3, H, W)
        # We might not have metadata in unstructured setting, but code expects it.
        # Pass empty or dummy metadata if needed, but here we use what's in batch
        metadata = {k: v.to(device) for k, v in batch['metadata'].items() if v is not None}
        gt_pc = batch['pointcloud'].to(device)
        
        # Ground truth poses for metrics (if available in metadata)
        # Assuming metadata contains 'cam2rig' or similar if we want to compute MAA
        # If not, we skip MAA.
        # Filter out cam2rig from metadata passed to model for Rig Discovery
        # We want the model to predict rig rays without using provided extrinsics
        model_metadata = {k: v for k, v in metadata.items() if k != 'cam2rig'}
        
        with torch.no_grad():
            outputs = model(images, model_metadata)
            
            # outputs['rig_raymap']: (B, N, P, 6)
            # outputs['pointmap']: (B, N, P, 3)
            
            rig_raymaps = outputs['rig_raymap']
            pointmaps = outputs['pointmap']
            
            B, N, P, _ = rig_raymaps.shape
            H_patch = int(P**0.5) # Assuming square patches
            W_patch = H_patch
            
            # Process each sample in batch (usually B=1)
            for b in range(B):
                # 4. Rig Discovery Steps
                
                # A. Raymap Extraction & Pose Recovery
                # We do this per frame
                recovered_poses = []
                for n in range(N):
                    # Reshape to (H, W, 6)
                    rmap = rig_raymaps[b, n].reshape(H_patch, W_patch, 6)
                    pose = recover_pose_closed_form(rmap)
                    recovered_poses.append(pose)
                
                # B. Clustering
                # Cluster frames based on poses
                cluster_labels = cluster_rig_poses(recovered_poses, n_clusters=None) # Auto-detect or 1
                
                # C. Pointcloud Reconstruction
                # Reconstruct using recovered poses
                # pointmaps[b, n] is (P, 3)
                # We need to reshape/pass correct format
                pmaps_list = [pointmaps[b, n] for n in range(N)]
                pred_pc = reconstruct_pointcloud(pmaps_list, recovered_poses)

                # The model predicts at z-bar normalized scale (Eq. 3 normalizes only
                # the ground truth), so put it back into metres before any metric that
                # is expressed in them - Chamfer, and the 0.1 m threshold below.
                scale = align_scale(pred_pc, gt_pc[b])
                pred_pc = pred_pc * scale

                # 5. Metrics

                # Chamfer Distance
                cd = chamfer_distance(pred_pc, gt_pc[b])
                all_chamfer.append(cd.item())

                # Check if we have valid GT for Rig Metrics
                if 'cam2rig' in metadata:
                    # Let's construct "pred_rig_keypoints" from our recovered poses (translations)
                    # Same normalized scale as the pointmap, so the same factor applies.
                    pred_rig_keypoints = torch.stack([p['t'] for p in recovered_poses]) * scale

                    gt_cam2rig = metadata['cam2rig'][b] # (N, ...)
                    
                    # Check if gt_cam2rig is a valid rotation/transform matrix (N, 3, 3) or (N, 4, 4)
                    if gt_cam2rig.dim() >= 3 and gt_cam2rig.shape[-1] >= 3:
                        gt_rig_keypoints = gt_cam2rig[:, :3, 3]
                        
                        # Rig ID Accuracy
                        r_acc = rig_discovery_accuracy(pred_rig_keypoints, gt_rig_keypoints)
                        all_rig_acc.append(r_acc.item())
                        
                        # Rig mAA
                        gt_poses_list = []
                        for n in range(N):
                            gt_poses_list.append({
                                'R': gt_cam2rig[n, :3, :3],
                                't': gt_cam2rig[n, :3, 3]
                            })
                        
                        maa = rig_maa(recovered_poses, gt_poses_list)
                        all_rig_maa.append(maa.item())
                    else:
                        # print(f"Skipping Rig Metrics: Invalid GT cam2rig shape {gt_cam2rig.shape}")
                        pass
                
    # 6. Report Results
    avg_chamfer = sum(all_chamfer) / len(all_chamfer) if all_chamfer else 0.0
    avg_rig_acc = sum(all_rig_acc) / len(all_rig_acc) if all_rig_acc else 0.0
    avg_rig_maa = sum(all_rig_maa) / len(all_rig_maa) if all_rig_maa else 0.0
    
    print("\n=== Rig Discovery Results ===")
    print(f"Avg Chamfer Distance: {avg_chamfer:.6f}")
    print(f"Avg Rig ID Accuracy:  {avg_rig_acc:.4f}")
    print(f"Avg Rig mAA (deg):    {avg_rig_maa:.4f}")
    
    # Save results (optional, for visualization)
    # torch.save(...)

if __name__ == "__main__":
    main()
