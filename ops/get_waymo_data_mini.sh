#!/bin/sh

# CONFIGURATION
# --------------------
# Number of subfolders to download in parallel.
MAX_PARALLEL=4 

# Define the shared list of subfolders (Same structure for both)
subfolders="camera_box camera_calibration camera_hkp camera_image camera_segmentation camera_to_lidar_box_association lidar lidar_box lidar_calibration lidar_camera_projection lidar_camera_synced_box lidar_hkp lidar_pose lidar_segmentation projected_lidar_box stats vehicle_pose"

# 1. Define Training File IDs
train_file_ids="10017090168044687777_6380_000_6400_000 10023947602400723454_1120_000_1140_000 1005081002024129653_5313_150_5333_150 10061305430875486848_1080_000_1100_000 10072140764565668044_4060_000_4080_000 10072231702153043603_5725_000_5745_000 10075870402459732738_1060_000_1080_000 10082223140073588526_6140_000_6160_000 10094743350625019937_3420_000_3440_000 10096619443888687526_2820_000_2840_000 10107710434105775874_760_000_780_000 10153695247769592104_787_000_807_000 10206293520369375008_2796_800_2816_800 10212406498497081993_5300_000_5320_000 1022527355599519580_4866_960_4886_960 10226164909075980558_180_000_200_000 10231929575853664160_1160_000_1180_000 10235335145367115211_5420_000_5440_000 10241508783381919015_2889_360_2909_360 10275144660749673822_5755_561_5775_561 10327752107000040525_1120_000_1140_000 10391312872392849784_4099_400_4119_400 10444454289801298640_4360_000_4380_000 10455472356147194054_1560_000_1580_000 10485926982439064520_4980_000_5000_000"

# 2. Define Validation File IDs (Extracted from your command)
val_file_ids="10203656353524179475_7625_000_7645_000 1024360143612057520_3580_000_3600_000 10247954040621004675_2180_000_2200_000 10289507859301986274_4200_000_4220_000 10335539493577748957_1372_870_1392_870 10359308928573410754_720_000_740_000 10448102132863604198_472_000_492_000 10689101165701914459_2072_300_2092_300 1071392229495085036_1844_790_1864_790 10837554759555844344_6525_000_6545_000 10868756386479184868_3000_000_3020_000 11037651371539287009_77_670_97_670 11048712972908676520_545_000_565_000 1105338229944737854_1280_000_1300_000 11356601648124485814_409_000_429_000 11387395026864348975_3820_000_3840_000 11406166561185637285_1753_750_1773_750 11434627589960744626_4829_660_4849_660 11450298750351730790_1431_750_1451_750 11616035176233595745_3548_820_3568_820 11660186733224028707_420_000_440_000 11901761444769610243_556_000_576_000 12102100359426069856_3931_470_3951_470 12134738431513647889_3118_000_3138_000 12306251798468767010_560_000_580_000"

# FUNCTION: Download Logic
# Arguments: $1 = GCS Split Name (training/validation), $2 = Local Dir Name, $3 = File IDs
download_dataset() {
    gcs_split=$1
    local_dir=$2
    file_ids=$3
    
    echo "--------------------------------------------------------"
    echo "Starting download for: $gcs_split -> /data/waymo_mini/$local_dir/"
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
        
        # Run gsutil (progress silenced, errors kept)
        if gsutil -m cp $gcs_paths "data/waymo_mini/$local_dir/$subfolder/" > /dev/null; then
          echo "[$gcs_split] Completed: $subfolder"
        else
          echo "[$gcs_split] FAILED: $subfolder" >&2
        fi
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
