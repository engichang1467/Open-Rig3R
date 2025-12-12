# Dataset Preparation

## Waymo

To work with Waymo Open Dataset, make sure to register on their [official website](https://waymo.com/open/) and install [Google Cloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk)

### 1. Authenticate `gsutil` to your account

- It will generate a verification code for you to copy and paste in

```bash
gcloud auth login
gcloud auth list
```

### 2. Configure `gsutil` to your account credentials

```bash
gsutil config
```

### 3. Install the dataset

#### Original Training Dataset (~1 TB)

```bash
make download-waymo-full
```

#### Small Subset of Training Dataset (~5.8 GB)

```bash
make download-waymo-mini
```


## CO3D

To work with a lightweight subset of the CO3D dataset (Common Objects in 3D), follow the steps below. These instructions are adapted from the [official CO3D GitHub repository](https://github.com/facebookresearch/co3d).

### 1. Create the dataset directory

Create the directory in the current project folder

```bash
mkdir -p data/co3d
```

### 2. Clone the CO3D repository

Clone the CO3D codebase **outside** of your current project folder:

```bash
git clone git@github.com:facebookresearch/co3d.git
cd co3d/
```

### 3. Install dependencies

Install the required Python packages:

```bash
pip install visdom tqdm requests h5py
```

Then install the CO3D package itself:

```bash
pip install -e .
```

- **Note**: Make sure to install these packages in a separate environment

### 4. Download the small subset of the dataset

Use the CO3D download script with the `--single_sequence_subset` flag to fetch a compact subset suitable for the many-view, single-sequence task:

```bash
python ./co3d/download_dataset.py \
  --download_folder DOWNLOAD_FOLDER \
  --single_sequence_subset
```

Example (downloading into this repo’s `data/co3d` folder):

```bash
python ./co3d/download_dataset.py \
  --download_folder ../Open-Rig3R/data/co3d/ \
  --single_sequence_subset
```

This subset requires ~8.9 GB, significantly smaller than the full dataset (~5.5 TB).