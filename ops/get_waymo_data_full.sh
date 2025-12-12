#!/bin/sh

# CONFIGURATION
# --------------------
# Number of subfolders to download in parallel.
MAX_PARALLEL=4 

# Define the shared list of subfolders (Same structure for both)
subfolders="camera_box camera_calibration camera_hkp camera_image camera_segmentation camera_to_lidar_box_association lidar lidar_box lidar_calibration lidar_camera_projection lidar_camera_synced_box lidar_hkp lidar_pose lidar_segmentation projected_lidar_box stats vehicle_pose"

# FUNCTION: Download Logic
# Arguments: $1 = GCS Split Name (training/validation), $2 = Local Dir Name
download_dataset() {
    gcs_split=$1
    local_dir=$2
    
    echo "--------------------------------------------------------"
    echo "Starting FULL download for: $gcs_split -> data/waymo_mini/$local_dir/"
    echo "--------------------------------------------------------"

    counter=0
    
    for subfolder in $subfolders; do
      (
        # Create directory
        mkdir -p "data/waymo/$local_dir/$subfolder/"
        
        # Run gsutil to download ALL parquet files in this subfolder
        # Note: This downloads the entire dataset for this split/subfolder combination
        echo "[$gcs_split] Downloading all files for: $subfolder..."
        gsutil -m cp "gs://waymo_open_dataset_v_2_0_1/$gcs_split/$subfolder/*.parquet" "data/waymo_mini/$local_dir/$subfolder/" > /dev/null 2>&1
        
        echo "[$gcs_split] Completed: $subfolder"
      ) &

      # Parallelization Control
      counter=$((counter + 1))
      if [ $((counter % MAX_PARALLEL)) -eq 0 ]; then
        wait
      fi
    done
    
    # Wait for remaining jobs in this dataset to finish before moving to the next
    wait 
}

# EXECUTION
# --------------------

# 1. Download Training Data (Full Set)
# GCS folder: 'training' -> Local folder: 'train'
download_dataset "training" "train"

# 2. Download Validation Data (Full Set)
# GCS folder: 'validation' -> Local folder: 'validation'
download_dataset "validation" "validation"

echo "--------------------------------------------------------"
echo "All downloads (Train & Validation) complete!"