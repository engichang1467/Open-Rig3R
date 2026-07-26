"""Extract Waymo camera_image parquet -> JPEG files, plus per-segment geometry.

Layout: DST/<split>/<segment>/<camera>/<frame_timestamp_micros>.jpeg
        DST/<split>/<segment>/calibration.json
        DST/<split>/<segment>/poses.json
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

SPLITS = ["train", "validation"]

# waymo CameraName enum
CAMERAS = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}

COLS = [
    "key.camera_name",
    "key.frame_timestamp_micros",
    "[CameraImageComponent].image",
]

CALIB = "[CameraCalibrationComponent]"
CALIB_COLS = [
    "key.camera_name",
    f"{CALIB}.extrinsic.transform",
    f"{CALIB}.intrinsic.f_u",
    f"{CALIB}.intrinsic.f_v",
    f"{CALIB}.intrinsic.c_u",
    f"{CALIB}.intrinsic.c_v",
    f"{CALIB}.width",
    f"{CALIB}.height",
]

POSE_COLS = [
    "key.frame_timestamp_micros",
    "[VehiclePoseComponent].world_from_vehicle.transform",
]


def convert_calibration(src, dst, split):
    """Write one calibration.json per segment: 5 rows, static for the whole segment.

    extrinsic.transform is a 4x4 vehicle_from_camera, i.e. cam2rig as-is (the rig
    frame is the vehicle frame). Intrinsics are stored at native resolution; a
    consumer that resizes must scale f/c by its own (out / width, out / height).
    """
    files = sorted((src / split / "camera_calibration").glob("*.parquet"))

    for path in tqdm(files, desc=f"{split} calibration"):
        rows = pq.read_table(path, columns=CALIB_COLS).to_pylist()
        calibration = {
            CAMERAS.get(row["key.camera_name"], f"CAMERA_{row['key.camera_name']}"): {
                "cam2rig": row[f"{CALIB}.extrinsic.transform"],  # row-major 4x4
                "f_u": row[f"{CALIB}.intrinsic.f_u"],
                "f_v": row[f"{CALIB}.intrinsic.f_v"],
                "c_u": row[f"{CALIB}.intrinsic.c_u"],
                "c_v": row[f"{CALIB}.intrinsic.c_v"],
                "width": row[f"{CALIB}.width"],
                "height": row[f"{CALIB}.height"],
            }
            for row in rows
        }

        out_dir = dst / split / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2))


def convert_poses(src, dst, split):
    """Write one poses.json per segment: world_from_vehicle 4x4 per frame timestamp.

    This is what makes rays from different frames comparable, so pose supervision
    is not just the static rig repeated.
    """
    files = sorted((src / split / "vehicle_pose").glob("*.parquet"))

    for path in tqdm(files, desc=f"{split} poses"):
        rows = pq.read_table(path, columns=POSE_COLS).to_pylist()
        poses = {
            str(row["key.frame_timestamp_micros"]): row[
                "[VehiclePoseComponent].world_from_vehicle.transform"
            ]
            for row in rows
        }

        out_dir = dst / split / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "poses.json").write_text(json.dumps(poses))


def convert_split(src, dst, split):
    files = sorted((src / split / "camera_image").glob("*.parquet"))

    for path in tqdm(files, desc=split):
        segment = path.stem
        pf = pq.ParquetFile(path)

        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg, columns=COLS)

            for cam, ts, image in zip(*(c.to_pylist() for c in table.columns)):
                out_dir = dst / split / segment / CAMERAS.get(cam, f"CAMERA_{cam}")
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{ts}.jpeg").write_bytes(image)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/waymo_mini"),
                        help="Root of the downloaded parquet dataset (default: %(default)s)")
    parser.add_argument("--dst", type=Path, default=Path("/data/waymo_mini"),
                        help="Where to write the JPEG tree (default: %(default)s)")
    parser.add_argument("--splits", nargs="+", default=SPLITS,
                        help="Splits to convert (default: %(default)s)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for split in args.splits:
        convert_split(args.src, args.dst, split)
        convert_calibration(args.src, args.dst, split)
        convert_poses(args.src, args.dst, split)
