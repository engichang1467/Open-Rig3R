"""Project Waymo TOP lidar into each camera -> per-frame sparse pointmaps.

Layout: DST/<split>/<segment>/pointmap/<CAMERA>.npy   (n_frames, grid, grid, 3) float16
        DST/<split>/<segment>/pointmap/timestamps.json

Points are stored in each camera's *own* frame, with NaN where no lidar return
landed in that cell. Per-camera is the only frame that is consistent: the five
cameras are triggered at different times within a frame and the lidar spins
across it, so a single shared vehicle frame leaves points off their own camera's
rays by tens of pixels whenever the rig is moving. The loader re-expresses them
relative to whichever view a training window starts at.

`--verify` skips writing and instead reprojects the computed points back through
the camera intrinsics, comparing against Waymo's own `lidar_camera_projection`.
That is the check on the range-image math: if the geometry is wrong, the
reprojection error blows up.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

SPLITS = ["train", "validation"]

# waymo CameraName enum, matching ops/parquet2jpeg.py
CAMERAS = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}

# ponytail: TOP lidar only. It is the 64-beam 360-degree sensor covering every
# camera out to ~75m; the four side lidars are 20m near-field and use a different
# (min/max) inclination convention. Add them if close-range density matters.
TOP = 1


def rotation_matrix(roll, pitch, yaw):
    """ZYX euler -> (..., 3, 3), matching waymo's transform_utils.get_rotation_matrix."""
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    zeros, ones = np.zeros_like(cos_roll), np.ones_like(cos_roll)

    shape = cos_roll.shape + (3, 3)
    r_roll = np.stack([ones, zeros, zeros, zeros, cos_roll, -sin_roll,
                       zeros, sin_roll, cos_roll], -1).reshape(shape)
    r_pitch = np.stack([cos_pitch, zeros, sin_pitch, zeros, ones, zeros,
                        -sin_pitch, zeros, cos_pitch], -1).reshape(shape)
    r_yaw = np.stack([cos_yaw, -sin_yaw, zeros, sin_yaw, cos_yaw, zeros,
                      zeros, zeros, ones], -1).reshape(shape)
    return r_yaw @ r_pitch @ r_roll


def transform(points, matrix):
    """Apply a 4x4 SE(3) to (..., 3) points."""
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def range_image_to_world(range_image, extrinsic, inclinations, pixel_pose):
    """Range image -> cartesian points in the world frame.

    The TOP lidar spins during the frame, so each pixel carries its own vehicle
    pose. Points go sensor -> vehicle(pixel time) -> world, which is what removes
    the rolling-shutter smear. World is the only frame shared by every sensor, so
    the per-camera step downstream can pick up each camera's own capture time.
    """
    height, width, _ = range_image.shape

    # azimuth runs backwards across the image and is offset by the sensor yaw
    azimuth_correction = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
    ratios = (np.arange(width, 0, -1) - 0.5) / width
    azimuth = ((ratios * 2.0 - 1.0) * np.pi - azimuth_correction)[None, :]
    inclination = inclinations[:, None]

    ranges = range_image[..., 0]
    points = np.stack([
        np.cos(azimuth) * np.cos(inclination) * ranges,
        np.sin(azimuth) * np.cos(inclination) * ranges,
        np.sin(inclination) * ranges,
    ], axis=-1)

    points = transform(points, extrinsic)

    pixel_rotation = rotation_matrix(pixel_pose[..., 0], pixel_pose[..., 1], pixel_pose[..., 2])
    world = np.einsum("hwij,hwj->hwi", pixel_rotation, points) + pixel_pose[..., 3:]

    return world, ranges


def bin_to_grid(points, ranges, pixels, size, grid):
    """Scatter projected points into a (grid, grid, 3) cell map, nearest return wins."""
    width, height = size
    cells = np.stack([
        np.clip(pixels[:, 1] / height * grid, 0, grid - 1),
        np.clip(pixels[:, 0] / width * grid, 0, grid - 1),
    ], -1).astype(np.int32)

    # writing far points first leaves the nearest surface in each cell
    order = np.argsort(-ranges)
    flat = np.full((grid * grid, 3), np.nan, dtype=np.float32)
    flat[cells[order, 0] * grid + cells[order, 1]] = points[order]
    return flat.reshape(grid, grid, 3)


def camera_calibration(src, split, segment):
    """{camera_name: (cam2rig, intrinsics, (width, height))}"""
    table = pq.read_table(f"{src}/{split}/camera_calibration/{segment}.parquet").to_pylist()
    prefix = "[CameraCalibrationComponent]"
    return {
        row["key.camera_name"]: (
            np.array(row[f"{prefix}.extrinsic.transform"]).reshape(4, 4),
            np.array([row[f"{prefix}.intrinsic.{k}"] for k in ("f_u", "f_v", "c_u", "c_v")]),
            (row[f"{prefix}.width"], row[f"{prefix}.height"]),
        )
        for row in table
    }


def project(camera_points, intrinsics):
    """Camera-frame points -> pixels, Waymo camera model (x forward, y left, z up)."""
    f_u, f_v, c_u, c_v = intrinsics
    u = f_u * (-camera_points[:, 1] / camera_points[:, 0]) + c_u
    v = f_v * (-camera_points[:, 2] / camera_points[:, 0]) + c_v
    return np.stack([u, v], -1)


def read_camera_poses(src, split, segment):
    """{(timestamp, camera_name): world_from_vehicle at that camera's capture time}.

    Each camera is triggered at its own instant, so this is not the frame's
    vehicle pose. Reading it skips the image column, so it stays cheap.
    """
    table = pq.read_table(
        f"{src}/{split}/camera_image/{segment}.parquet",
        columns=["key.frame_timestamp_micros", "key.camera_name",
                 "[CameraImageComponent].pose.transform"],
    ).to_pylist()
    return {
        (row["key.frame_timestamp_micros"], row["key.camera_name"]): np.array(
            row["[CameraImageComponent].pose.transform"]
        ).reshape(4, 4)
        for row in table
    }


def read_top_frames(src, split, component, prefix, segment):
    """{timestamp: range image array} for the TOP laser only."""
    path = f"{src}/{split}/{component}/{segment}.parquet"
    columns = ["key.frame_timestamp_micros", f"{prefix}.range_image_return1.values",
               f"{prefix}.range_image_return1.shape"]
    if component != "lidar_pose":
        columns.insert(0, "key.laser_name")

    parquet = pq.ParquetFile(path)
    frames = {}
    for group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(group, columns=columns)
        lasers = table.column("key.laser_name").to_pylist() if "key.laser_name" in columns else None
        timestamps = table.column("key.frame_timestamp_micros").to_pylist()
        values = table.column(f"{prefix}.range_image_return1.values")
        shapes = table.column(f"{prefix}.range_image_return1.shape").to_pylist()

        for i, timestamp in enumerate(timestamps):
            if lasers is not None and lasers[i] != TOP:
                continue
            frames[timestamp] = np.asarray(values[i].values).reshape(shapes[i])
    return frames


def convert_segment(src, dst, split, segment, grid, verify):
    calibration = pq.read_table(f"{src}/{split}/lidar_calibration/{segment}.parquet").to_pylist()
    prefix = "[LiDARCalibrationComponent]"
    top = next(row for row in calibration if row["key.laser_name"] == TOP)
    extrinsic = np.array(top[f"{prefix}.extrinsic.transform"]).reshape(4, 4)
    # range image row 0 is the topmost beam, so the stored inclinations are reversed
    inclinations = np.array(top[f"{prefix}.beam_inclination.values"])[::-1]

    poses = read_camera_poses(src, split, segment)
    cameras = camera_calibration(src, split, segment)

    lidar = read_top_frames(src, split, "lidar", "[LiDARComponent]", segment)
    pixel_poses = read_top_frames(src, split, "lidar_pose", "[LiDARPoseComponent]", segment)
    projections = read_top_frames(
        src, split, "lidar_camera_projection", "[LiDARCameraProjectionComponent]", segment
    )

    timestamps = sorted(
        t for t in lidar
        if t in pixel_poses and t in projections
        and all((t, camera_name) in poses for camera_name in CAMERAS)
    )
    pointmaps = {name: [] for name in CAMERAS.values()}
    errors = {name: [] for name in CAMERAS.values()}

    for timestamp in timestamps:
        world, ranges = range_image_to_world(
            lidar[timestamp], extrinsic, inclinations, pixel_poses[timestamp]
        )
        projection = projections[timestamp]
        valid = ranges > 0

        for camera_name, name in CAMERAS.items():
            cam2rig, intrinsics, size = cameras[camera_name]
            # each return may project into up to two cameras
            hits = [
                (projection[..., 0] == camera_name) & valid,
                (projection[..., 3] == camera_name) & valid,
            ]
            selected = np.concatenate([world[hit] for hit in hits])
            pixels = np.concatenate(
                [projection[..., 1:3][hits[0]], projection[..., 4:6][hits[1]]]
            )
            depths = np.concatenate([ranges[hit] for hit in hits])

            # world -> vehicle at this camera's capture time -> camera
            world_from_camera = poses[(timestamp, camera_name)] @ cam2rig
            selected = transform(selected, np.linalg.inv(world_from_camera))

            if verify:
                if len(selected):
                    reprojected = project(selected, intrinsics)
                    errors[name].append(np.linalg.norm(reprojected - pixels, axis=-1))
                continue

            pointmaps[name].append(bin_to_grid(selected, depths, pixels, size, grid))

    if verify:
        for name, batch in errors.items():
            stacked = np.concatenate(batch)
            print(f"  {name:12s} n={len(stacked):8d}  median={np.median(stacked):6.2f}px  "
                  f"p99={np.percentile(stacked, 99):7.2f}px")
        return

    out_dir = dst / split / segment / "pointmap"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frames in pointmaps.items():
        np.save(out_dir / f"{name}.npy", np.stack(frames).astype(np.float16))
    (out_dir / "timestamps.json").write_text(json.dumps([str(t) for t in timestamps]))


def convert_split(src, dst, split, grid, verify):
    segments = sorted(p.stem for p in (src / split / "lidar").glob("*.parquet"))
    for segment in tqdm(segments, desc=f"{split} pointmaps"):
        if verify:
            print(f"\n{segment}")
        convert_segment(str(src), dst, split, segment, grid, verify)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/waymo_mini"),
                        help="Root of the downloaded parquet dataset (default: %(default)s)")
    parser.add_argument("--dst", type=Path, default=Path("/data/waymo_mini"),
                        help="Where to write the pointmaps (default: %(default)s)")
    parser.add_argument("--splits", nargs="+", default=SPLITS,
                        help="Splits to convert (default: %(default)s)")
    parser.add_argument("--grid", type=int, default=64,
                        help="Cells per side, pooled down to the patch grid at load "
                             "(default: %(default)s)")
    parser.add_argument("--verify", action="store_true",
                        help="Reproject against waymo's own projections instead of writing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for split in args.splits:
        convert_split(args.src, args.dst, split, args.grid, args.verify)
