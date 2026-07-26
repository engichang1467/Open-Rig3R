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

Each target downloads the parquet files, then exports the camera images to JPEG.

#### Original Training Dataset (~1 TB)

```bash
make download-waymo-full
```

#### Small Subset of Training Dataset (~5.8 GB)

```bash
make download-waymo-mini
```

### 4. The JPEG export

Training reads images, not parquet. `ops/parquet2jpeg.py` pulls the
`[CameraImageComponent].image` column out of the `camera_image` parquet files and
writes the JPEG bytes straight to disk — no decode/re-encode, the payload is
already JPEG. It also writes one `calibration.json` per segment from the
`camera_calibration` component, which is where the rig extrinsics come from. The
`make` targets above run it for you; run it directly to re-export, or to use a
different location:

```bash
python ops/parquet2jpeg.py --src data/waymo_mini --dst /data/waymo_mini
python ops/parquet2jpeg.py --src data/waymo_mini --dst /data/waymo_mini --splits validation
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--src` | `data/waymo_mini` | Root of the downloaded parquet dataset |
| `--dst` | `/data/waymo_mini` | Where to write the JPEG tree |
| `--splits` | `train validation` | Splits to convert |

Output layout:

```
<dst>/<split>/<segment>/<CAMERA>/<frame_timestamp_micros>.jpeg
<dst>/<split>/<segment>/calibration.json
<dst>/<split>/<segment>/poses.json
<dst>/<split>/<segment>/pointmap/<CAMERA>.npy      # from parquet2pointmap.py
<dst>/<split>/<segment>/pointmap/timestamps.json
```

with `<CAMERA>` one of `FRONT`, `FRONT_LEFT`, `FRONT_RIGHT`, `SIDE_LEFT`, `SIDE_RIGHT`.
Re-running overwrites existing files in place.

`calibration.json` holds, per camera, the 4x4 `cam2rig` (Waymo's
`extrinsic.transform`, a `vehicle_from_camera` — the rig frame *is* the vehicle
frame) plus `f_u`/`f_v`/`c_u`/`c_v` and native `width`/`height`. Intrinsics are
stored unscaled; `WaymoDataset` rescales them to `image_size` on load.

`poses.json` maps each frame timestamp to a 4x4 `world_from_vehicle` **per camera**,
taken at that camera's own trigger instant rather than once per frame. The five
cameras do not fire together, so a single per-frame pose leaves points tens of
pixels off their own camera's rays whenever the rig is moving.

Both files are required: `WaymoDataset` raises if either is missing, and a tree
exported before they existed needs a re-run.

Together they are what supervises training — `utils/raymap.py` turns calibration
plus poses into the `rig_raymap` and `pose_raymap` targets that `MultiTaskLoss`
scores.

### 5. The lidar pointmaps

`ops/parquet2pointmap.py` adds the third loss term, projecting the TOP lidar into
each camera to get sparse depth. The `make` targets run it; run it directly to
re-export:

```bash
python ops/parquet2pointmap.py --src data/waymo_mini --dst /data/waymo_mini
python ops/parquet2pointmap.py --src data/waymo_mini --verify
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--src` | `data/waymo_mini` | Root of the downloaded parquet dataset |
| `--dst` | `/data/waymo_mini` | Where to write the pointmaps |
| `--splits` | `train validation` | Splits to convert |
| `--grid` | `64` | Cells per side, pooled to the patch grid at load |
| `--verify` | off | Reproject against Waymo's own projections instead of writing |

Each `<CAMERA>.npy` is `(n_frames, grid, grid, 3)` float16 holding points in that
camera's own frame, `NaN` where no return landed, about 24 MB per segment. The
loader pools the grid down to the model's patch grid and turns coverage into
`pointmap_conf`, so empty patches contribute nothing to the loss.

Unlike the other two files this one is **optional** — without it training still
runs on raymap supervision alone.

`--verify` is the check on the range-image math: it reprojects the computed points
through the camera intrinsics and compares against the `lidar_camera_projection`
component. Expect a median around 2 px at native 1920x1280 resolution. The residual
is per-row rolling shutter, which needs the `velocity` and `rolling_shutter_params`
fields to model; at 128x128 training resolution even the worst observed segment
stays under half a patch, so it is left uncorrected.

### 6. Point the training config at it

`waymo_path` in [`configs/train_waymo.yaml`](../configs/train_waymo.yaml) must match
your `--dst`:

```yaml
waymo_path: "/data/waymo_mini"
waymo_cameras: null   # null = full 5-camera rig
n_frames: 2           # consecutive timestamps per sample; views = n_frames * num_cameras
```

Then:

```bash
make train
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