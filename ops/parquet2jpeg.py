"""Extract Waymo camera_image parquet -> JPEG files.

Layout: DST/<split>/<segment>/<camera>/<frame_timestamp_micros>.jpeg
"""

import argparse
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
