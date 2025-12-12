#!/bin/sh

# CONFIGURATION
# --------------------
# Number of subfolders to download in parallel.
MAX_PARALLEL=4 

# Define the shared list of subfolders (Same structure for both)
subfolders="camera_box camera_calibration camera_hkp camera_image camera_segmentation camera_to_lidar_box_association lidar lidar_box lidar_calibration lidar_camera_projection lidar_camera_synced_box lidar_hkp lidar_pose lidar_segmentation projected_lidar_box stats vehicle_pose"

# 1. Define Training File IDs
train_file_ids="10017090168044687777_6380_000_6400_000 10023947602400723454_1120_000_1140_000 1005081002024129653_5313_150_5333_150 10061305430875486848_1080_000_1100_000 10072140764565668044_4060_000_4080_000"

# 2. Define Validation File IDs (Extracted from your command)
val_file_ids="10203656353524179475_7625_000_7645_000 1024360143612057520_3580_000_3600_000 10247954040621004675_2180_000_2200_000 10289507859301986274_4200_000_4220_000 10335539493577748957_1372_870_1392_870"

# FUNCTION: Download Logic
# Arguments: $1 = GCS Split Name (training/validation), $2 = Local Dir Name, $3 = File IDs
download_dataset() {
    gcs_split=$1
    local_dir=$2
    file_ids=$3
    
    echo "--------------------------------------------------------"
    echo "Starting download for: $gcs_split -> data/waymo_mini/$local_dir/"
    echo "--------------------------------------------------------"

    counter=0
    
    for subfolder in $subfolders; do
      (
        # Create directory
        mkdir -p "data/waymo_mini/$local_dir/$subfolder/"
        
        # Build path list
        gcs_paths=""
        for file_id in $file_ids; do
          gcs_paths="$gcs_paths gs://waymo_open_dataset_v_2_0_1/$gcs_split/$subfolder/${file_id}.parquet"
        done
        
        # Run gsutil (output silenced)
        gsutil -m cp $gcs_paths "data/waymo_mini/$local_dir/$subfolder/" > /dev/null 2>&1
        
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

# 1. Download Training Data
# GCS folder: 'training' -> Local folder: 'train'
download_dataset "training" "train" "$train_file_ids"

# 2. Download Validation Data
# GCS folder: 'validation' -> Local folder: 'validation'
download_dataset "validation" "validation" "$val_file_ids"

echo "--------------------------------------------------------"
echo "All downloads (Train & Validation) complete!"
