#!/usr/bin/env python3
"""Analyze dense ScanCover Quest clone capture sessions without copying inputs.

The tool consumes SCQ3BIN2 eye frames plus capture JSON metadata and emits a
compact, reproducible Quest depth-degradation profile for offline Replica /
Open3D experiments.  It deliberately distinguishes measured proxies from
ground-truth error: local planarity and repeated-voxel spread are repeatability
signals, not absolute metric accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MAGIC = b"SCQ3BIN2"
HEADER_BYTES = 272
RECORD_DTYPE = np.dtype(
    [
        ("valid", "u1"),
        ("point", "<f4", (3,)),
        ("meta", "<f4", (4,)),
    ],
    align=False,
)
DISTANCE_EDGES = np.asarray([0.35, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00], dtype=np.float64)
ANGLE_EDGES = np.asarray([0.0, 20.0, 40.0, 60.0, 75.0, 90.001], dtype=np.float64)
RESIDUAL_EDGES = np.linspace(0.0, 0.20, 401, dtype=np.float64)
CONFIDENCE_EDGES = np.linspace(0.0, 1.000001, 201, dtype=np.float64)


@dataclass(frozen=True)
class EyeHeader:
    version: int
    eye: int
    frame: int
    width: int
    height: int
    count: int
    dispatch_seconds: float
    completion_seconds: float
    has_pose: bool
    camera_position: np.ndarray
    camera_rotation: np.ndarray
    has_projection_matrix: bool
    projection_matrix: np.ndarray
    has_world_to_camera_matrix: bool
    world_to_camera_matrix: np.ndarray
    has_depth_reprojection_matrix: bool
    depth_reprojection_matrix: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="ScanCoverDiagnostics root or one session directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-depth", type=float, default=0.35)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--low-confidence", type=float, default=0.12)
    parser.add_argument("--local-stride", type=int, default=2, help="Pixel stride for local normal/residual analysis")
    parser.add_argument("--repeat-frame-step", type=int, default=5)
    parser.add_argument("--repeat-pixel-stride", type=int, default=8)
    parser.add_argument("--repeat-voxel", type=float, default=0.08)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 processes the complete session")
    return parser.parse_args()


def quantile_from_hist(hist: np.ndarray, edges: np.ndarray, probability: float) -> float:
    total = int(hist.sum())
    if total <= 0:
        return 0.0
    target = max(1, int(math.ceil(total * probability)))
    index = int(np.searchsorted(np.cumsum(hist), target, side="left"))
    index = min(index, len(edges) - 2)
    return float((edges[index] + edges[index + 1]) * 0.5)


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability)) if values.size else 0.0


def find_session(root: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    search = [root] if (root / "quest_clone_capture").is_dir() else list(root.glob("ScanCover_RepeatCoverage_*"))
    for session in search:
        capture = session / "quest_clone_capture"
        if not capture.is_dir():
            continue
        candidates.append((len(list(capture.glob("frame_*_capture.json"))), session))
    if not candidates:
        raise FileNotFoundError(f"No Quest clone capture session below {root}")
    return max(candidates, key=lambda item: item[0])[1]


def load_manifest(capture: Path) -> list[dict[str, str]]:
    manifest = capture / "quest_clone_capture_manifest.csv"
    lines = manifest.read_text(encoding="utf-8-sig").splitlines()
    header_index = next((index for index, line in enumerate(lines) if line.startswith("frame,")), None)
    if header_index is None:
        raise ValueError(f"Manifest header is missing: {manifest}")
    rows = list(csv.DictReader(lines[header_index:]))
    return [row for row in rows if row.get("status", "exported") == "exported" and row.get("frame")]


def read_eye(path: Path) -> tuple[EyeHeader, np.ndarray]:
    with path.open("rb") as stream:
        if stream.read(8) != MAGIC:
            raise ValueError(f"Invalid magic: {path}")
        version, eye, frame, width, height, count = struct.unpack("<6i", stream.read(24))
        dispatch_seconds, completion_seconds = struct.unpack("<2d", stream.read(16))
        has_pose = bool(struct.unpack("<B", stream.read(1))[0])
        pose = np.asarray(struct.unpack("<7f", stream.read(28)), dtype=np.float64)
        has_projection_matrix = bool(struct.unpack("<B", stream.read(1))[0])
        projection_matrix = np.asarray(
            struct.unpack("<16f", stream.read(64)), dtype=np.float64
        ).reshape(4, 4)
        has_world_to_camera_matrix = bool(struct.unpack("<B", stream.read(1))[0])
        world_to_camera_matrix = np.asarray(
            struct.unpack("<16f", stream.read(64)), dtype=np.float64
        ).reshape(4, 4)
        has_depth_reprojection_matrix = bool(struct.unpack("<B", stream.read(1))[0])
        depth_reprojection_matrix = np.asarray(
            struct.unpack("<16f", stream.read(64)), dtype=np.float64
        ).reshape(4, 4)
        if stream.tell() != HEADER_BYTES:
            raise ValueError(
                f"Unexpected SCQ3BIN2 header size: {path} "
                f"expected={HEADER_BYTES} actual={stream.tell()}"
            )
        records = np.fromfile(stream, dtype=RECORD_DTYPE, count=count)
    if records.size != count:
        raise ValueError(f"Truncated records: {path} expected={count} actual={records.size}")
    return EyeHeader(
        version,
        eye,
        frame,
        width,
        height,
        count,
        dispatch_seconds,
        completion_seconds,
        has_pose,
        pose[:3],
        pose[3:],
        has_projection_matrix,
        projection_matrix,
        has_world_to_camera_matrix,
        world_to_camera_matrix,
        has_depth_reprojection_matrix,
        depth_reprojection_matrix,
    ), records


def quaternion_forward(rotation: list[float]) -> np.ndarray:
    x, y, z, w = rotation
    return np.asarray((2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)))


def quaternion_angle_degrees(a: list[float], b: list[float]) -> float:
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


class Accumulator:
    def __init__(self, low_confidence: float) -> None:
        nd = len(DISTANCE_EDGES) - 1
        na = len(ANGLE_EDGES) - 1
        nr = len(RESIDUAL_EDGES) - 1
        nc = len(CONFIDENCE_EDGES) - 1
        self.low_confidence = low_confidence
        self.total_pixels = 0
        self.sensor_valid_pixels = 0
        self.working_valid_pixels = 0
        self.distance_points = np.zeros(nd, dtype=np.int64)
        self.distance_low_confidence = np.zeros(nd, dtype=np.int64)
        self.distance_frame_presence = np.zeros(nd, dtype=np.int64)
        self.distance_strong_frame_presence = np.zeros(nd, dtype=np.int64)
        self.confidence_hist = np.zeros((nd, nc), dtype=np.int64)
        self.angle_points = np.zeros(na, dtype=np.int64)
        self.angle_residual_hist = np.zeros((na, nr), dtype=np.int64)
        self.distance_residual_hist = np.zeros((nd, nr), dtype=np.int64)
        self.joint_points = np.zeros((nd, na), dtype=np.int64)
        self.joint_residual_hist = np.zeros((nd, na, nr), dtype=np.int64)
        self.local_candidate_points = 0
        self.local_supported_points = 0
        self.repeat_keys: list[np.ndarray] = []
        self.repeat_positions: list[np.ndarray] = []
        self.repeat_depths: list[np.ndarray] = []

    def add_eye(
        self,
        header: EyeHeader,
        records: np.ndarray,
        local_stride: int,
        collect_repeat: bool,
        repeat_pixel_stride: int,
        repeat_voxel: float,
        min_depth: float,
        max_depth: float,
    ) -> None:
        points = records["point"].astype(np.float64, copy=False)
        meta = records["meta"].astype(np.float64, copy=False)
        depth = meta[:, 2]
        finite = np.isfinite(points).all(axis=1) & np.isfinite(depth)
        sensor_valid = (records["valid"] != 0) & (meta[:, 0] >= 0.5) & (depth > 0.0) & finite
        working = sensor_valid & (depth >= min_depth) & (depth <= max_depth)
        self.total_pixels += int(records.size)
        self.sensor_valid_pixels += int(sensor_valid.sum())
        self.working_valid_pixels += int(working.sum())

        frame_bins = np.bincount(
            np.clip(np.digitize(depth[working], DISTANCE_EDGES) - 1, 0, len(DISTANCE_EDGES) - 2),
            minlength=len(DISTANCE_EDGES) - 1,
        ) if working.any() else np.zeros(len(DISTANCE_EDGES) - 1, dtype=np.int64)
        self.distance_points += frame_bins
        self.distance_frame_presence += frame_bins > 0
        self.distance_strong_frame_presence += frame_bins >= 100
        if working.any():
            db = np.digitize(depth[working], DISTANCE_EDGES) - 1
            confidence = np.clip(meta[working, 1], 0.0, 1.0)
            for d_index in range(len(DISTANCE_EDGES) - 1):
                selected = db == d_index
                if not selected.any():
                    continue
                values = confidence[selected]
                self.distance_low_confidence[d_index] += int((values < self.low_confidence).sum())
                self.confidence_hist[d_index] += np.histogram(values, CONFIDENCE_EDGES)[0]

        if header.width * header.height == records.size and header.has_pose:
            self._add_local_geometry(header, points, working, depth, max(1, local_stride))

        if collect_repeat:
            selected_indices = np.arange(0, records.size, max(1, repeat_pixel_stride))
            selected_indices = selected_indices[working[selected_indices]]
            if selected_indices.size:
                selected_points = points[selected_indices].astype(np.float32, copy=True)
                voxel_coordinates = np.floor(selected_points / max(0.001, repeat_voxel)).astype(np.int64)
                offset = 1 << 20
                packed = ((voxel_coordinates[:, 0] + offset) << 42) | ((voxel_coordinates[:, 1] + offset) << 21) | (voxel_coordinates[:, 2] + offset)
                self.repeat_keys.append(packed)
                self.repeat_positions.append(selected_points)
                self.repeat_depths.append(depth[selected_indices].astype(np.float32, copy=True))

    def _add_local_geometry(
        self,
        header: EyeHeader,
        flat_points: np.ndarray,
        flat_valid: np.ndarray,
        flat_depth: np.ndarray,
        stride: int,
    ) -> None:
        h, w = header.height, header.width
        points = flat_points.reshape(h, w, 3)
        valid = flat_valid.reshape(h, w)
        depth = flat_depth.reshape(h, w)
        ys = slice(1, h - 1, stride)
        xs = slice(1, w - 1, stride)
        center = points[ys, xs]
        left = points[ys, slice(0, w - 2, stride)]
        right = points[ys, slice(2, w, stride)]
        down = points[slice(0, h - 2, stride), xs]
        up = points[slice(2, h, stride), xs]
        supported = valid[ys, xs] & valid[ys, slice(0, w - 2, stride)] & valid[ys, slice(2, w, stride)] & valid[slice(0, h - 2, stride), xs] & valid[slice(2, h, stride), xs]
        self.local_candidate_points += supported.size
        if not supported.any():
            return
        horizontal = right - left
        vertical = up - down
        normals = np.cross(vertical, horizontal)
        normal_length = np.linalg.norm(normals, axis=2)
        supported &= np.isfinite(normal_length) & (normal_length > 1e-8)
        if not supported.any():
            return
        normals[supported] /= normal_length[supported, None]
        view = header.camera_position.reshape(1, 1, 3) - center
        view_length = np.linalg.norm(view, axis=2)
        supported &= np.isfinite(view_length) & (view_length > 1e-6)
        if not supported.any():
            return
        view[supported] /= view_length[supported, None]
        cosine = np.clip(np.abs(np.sum(normals * view, axis=2)), 0.0, 1.0)
        angle = np.degrees(np.arccos(cosine))
        neighbor_center = (left + right + down + up) * 0.25
        residual = np.abs(np.sum((center - neighbor_center) * normals, axis=2))
        local_depth = depth[ys, xs]
        supported &= np.isfinite(angle) & np.isfinite(residual) & np.isfinite(local_depth)
        self.local_supported_points += int(supported.sum())
        if not supported.any():
            return
        d_values = local_depth[supported]
        a_values = angle[supported]
        r_values = np.clip(residual[supported], 0.0, RESIDUAL_EDGES[-1] - 1e-9)
        d_bins = np.digitize(d_values, DISTANCE_EDGES) - 1
        a_bins = np.digitize(a_values, ANGLE_EDGES) - 1
        inside = (d_bins >= 0) & (d_bins < self.joint_points.shape[0]) & (a_bins >= 0) & (a_bins < self.joint_points.shape[1])
        d_bins, a_bins, r_values = d_bins[inside], a_bins[inside], r_values[inside]
        for d_index in range(self.joint_points.shape[0]):
            d_selected = d_bins == d_index
            if not d_selected.any():
                continue
            self.distance_residual_hist[d_index] += np.histogram(r_values[d_selected], RESIDUAL_EDGES)[0]
            for a_index in range(self.joint_points.shape[1]):
                selected = d_selected & (a_bins == a_index)
                if not selected.any():
                    continue
                count = int(selected.sum())
                hist = np.histogram(r_values[selected], RESIDUAL_EDGES)[0]
                self.joint_points[d_index, a_index] += count
                self.joint_residual_hist[d_index, a_index] += hist
                self.angle_points[a_index] += count
                self.angle_residual_hist[a_index] += hist


def build_repeatability(accumulator: Accumulator) -> list[dict[str, Any]]:
    if not accumulator.repeat_keys:
        return []
    keys = np.concatenate(accumulator.repeat_keys)
    positions = np.concatenate(accumulator.repeat_positions).astype(np.float64, copy=False)
    depths = np.concatenate(accumulator.repeat_depths).astype(np.float64, copy=False)
    order = np.argsort(keys, kind="stable")
    keys, positions, depths = keys[order], positions[order], depths[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    counts = np.diff(np.r_[starts, keys.size]).astype(np.int64)
    sum_position = np.add.reduceat(positions, starts, axis=0)
    sum_magnitude_sq = np.add.reduceat(np.sum(positions * positions, axis=1), starts)
    sum_depth = np.add.reduceat(depths, starts)
    keep = counts >= 3
    counts = counts[keep]
    means = sum_position[keep] / counts[:, None]
    variance = np.maximum(0.0, sum_magnitude_sq[keep] / counts - np.sum(means * means, axis=1))
    spread = np.sqrt(variance)
    mean_depth = sum_depth[keep] / counts
    result = []
    for index in range(len(DISTANCE_EDGES) - 1):
        selected = (mean_depth >= DISTANCE_EDGES[index]) & (mean_depth < DISTANCE_EDGES[index + 1])
        values = spread[selected]
        result.append(
            {
                "minDepthMeters": float(DISTANCE_EDGES[index]),
                "maxDepthMeters": float(DISTANCE_EDGES[index + 1]),
                "voxelCount": int(values.size),
                "positionSpreadP50Meters": quantile(values, 0.50),
                "positionSpreadP90Meters": quantile(values, 0.90),
                "positionSpreadP95Meters": quantile(values, 0.95),
                "positionSpreadP99Meters": quantile(values, 0.99),
            }
        )
    return result


def analyze_metadata(capture: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    poses: list[tuple[np.ndarray, list[float], float, float]] = []
    eye_deltas: list[float] = []
    invalid_json = 0
    for row in rows:
        path = capture / f"{row['frame']}_capture.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            position = np.asarray(data["headPoseAtExport"]["position"], dtype=np.float64)
            rotation = [float(value) for value in data["headPoseAtExport"]["rotation"]]
            timestamp = float(data["eyes"][0]["dispatchRealtimeSeconds"])
            poses.append((position, rotation, timestamp, float(data.get("interEyeDeltaMs", 0.0))))
            eye_deltas.append(float(data.get("interEyeDeltaMs", 0.0)))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            invalid_json += 1
    if not poses:
        return {"validPoseFrames": 0, "invalidJson": invalid_json}
    path_length = 0.0
    linear_speeds: list[float] = []
    angular_speeds: list[float] = []
    for previous, current in zip(poses, poses[1:]):
        dt = max(1e-6, current[2] - previous[2])
        movement = float(np.linalg.norm(current[0] - previous[0]))
        path_length += movement
        linear_speeds.append(movement / dt)
        angular_speeds.append(quaternion_angle_degrees(previous[1], current[1]) / dt)
    positions = np.stack([item[0] for item in poses])
    forwards = np.stack([quaternion_forward(item[1]) for item in poses])
    yaws = np.degrees(np.arctan2(forwards[:, 0], forwards[:, 2]))
    pitches = np.degrees(np.arcsin(np.clip(forwards[:, 1], -1.0, 1.0)))
    position_cells = np.floor(positions / 0.25).astype(np.int64)
    orientation_cells = np.stack((np.round(yaws / 15.0), np.round(pitches / 15.0)), axis=1).astype(np.int64)
    eye_array = np.asarray(eye_deltas, dtype=np.float64)
    return {
        "validPoseFrames": len(poses),
        "invalidJson": invalid_json,
        "durationSeconds": poses[-1][2] - poses[0][2],
        "pathLengthMeters": path_length,
        "positionExtentMeters": (positions.max(axis=0) - positions.min(axis=0)).tolist(),
        "positionCells25cm": int(np.unique(position_cells, axis=0).shape[0]),
        "orientationCells15deg": int(np.unique(orientation_cells, axis=0).shape[0]),
        "yawRangeDegrees": [float(yaws.min()), float(yaws.max())],
        "pitchRangeDegrees": [float(pitches.min()), float(pitches.max())],
        "linearSpeedMetersPerSecond": {"p50": quantile(np.asarray(linear_speeds), 0.5), "p90": quantile(np.asarray(linear_speeds), 0.9), "p95": quantile(np.asarray(linear_speeds), 0.95)},
        "angularSpeedDegreesPerSecond": {"p50": quantile(np.asarray(angular_speeds), 0.5), "p90": quantile(np.asarray(angular_speeds), 0.9), "p95": quantile(np.asarray(angular_speeds), 0.95)},
        "interEyeDeltaMs": {"p50": quantile(eye_array, 0.5), "p90": quantile(eye_array, 0.9), "p95": quantile(eye_array, 0.95), "p99": quantile(eye_array, 0.99), "max": float(eye_array.max()), "over50msRatio": float((eye_array > 50.0).mean())},
    }


def build_profiles(accumulator: Accumulator, repeatability: list[dict[str, Any]], eye_frame_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    global_dropout = 1.0 - accumulator.sensor_valid_pixels / max(1, accumulator.total_pixels)
    distance_rows = []
    for index in range(len(DISTANCE_EDGES) - 1):
        count = int(accumulator.distance_points[index])
        repeat = repeatability[index] if index < len(repeatability) else {}
        p90 = float(repeat.get("positionSpreadP90Meters", 0.0))
        p95 = float(repeat.get("positionSpreadP95Meters", 0.0))
        stable = min(0.060, max(0.032, p90 * 1.12 if p90 > 0 else 0.040))
        hard = min(0.120, max(0.065, p95 * 1.80 if p95 > 0 else 0.070))
        residual_p90 = quantile_from_hist(accumulator.distance_residual_hist[index], RESIDUAL_EDGES, 0.90)
        distance_rows.append(
            {
                "minDepthMeters": float(DISTANCE_EDGES[index]),
                "maxDepthMeters": float(DISTANCE_EDGES[index + 1]),
                "points": count,
                "framePresenceRatio": float(accumulator.distance_frame_presence[index] / max(1, eye_frame_count)),
                "strongFramePresenceRatio": float(accumulator.distance_strong_frame_presence[index] / max(1, eye_frame_count)),
                "lowConfidenceRatio": float(accumulator.distance_low_confidence[index] / max(1, count)),
                "confidenceP10": quantile_from_hist(accumulator.confidence_hist[index], CONFIDENCE_EDGES, 0.10),
                "confidenceP50": quantile_from_hist(accumulator.confidence_hist[index], CONFIDENCE_EDGES, 0.50),
                "confidenceP90": quantile_from_hist(accumulator.confidence_hist[index], CONFIDENCE_EDGES, 0.90),
                "localResidualP50Meters": quantile_from_hist(accumulator.distance_residual_hist[index], RESIDUAL_EDGES, 0.50),
                "localResidualP90Meters": residual_p90,
                "localResidualP95Meters": quantile_from_hist(accumulator.distance_residual_hist[index], RESIDUAL_EDGES, 0.95),
                **repeat,
                "recommendedStableSpreadMeters": stable,
                "recommendedHardSpreadMeters": hard,
                "recommendedGaussianNoiseSigmaMeters": max(0.0005, residual_p90 / 1.644854),
                "recommendedDropoutProbability": min(0.25, global_dropout + accumulator.distance_low_confidence[index] / max(1, count)),
            }
        )
    angle_rows = []
    for index in range(len(ANGLE_EDGES) - 1):
        residual_p90 = quantile_from_hist(accumulator.angle_residual_hist[index], RESIDUAL_EDGES, 0.90)
        angle_rows.append(
            {
                "minAngleDegrees": float(ANGLE_EDGES[index]),
                "maxAngleDegrees": float(ANGLE_EDGES[index + 1]),
                "points": int(accumulator.angle_points[index]),
                "localResidualP50Meters": quantile_from_hist(accumulator.angle_residual_hist[index], RESIDUAL_EDGES, 0.50),
                "localResidualP90Meters": residual_p90,
                "localResidualP95Meters": quantile_from_hist(accumulator.angle_residual_hist[index], RESIDUAL_EDGES, 0.95),
                "recommendedGaussianNoiseSigmaMeters": max(0.0005, residual_p90 / 1.644854),
            }
        )
    joint_rows = []
    for d_index in range(len(DISTANCE_EDGES) - 1):
        for a_index in range(len(ANGLE_EDGES) - 1):
            residual_p90 = quantile_from_hist(accumulator.joint_residual_hist[d_index, a_index], RESIDUAL_EDGES, 0.90)
            joint_rows.append(
                {
                    "minDepthMeters": float(DISTANCE_EDGES[d_index]),
                    "maxDepthMeters": float(DISTANCE_EDGES[d_index + 1]),
                    "minAngleDegrees": float(ANGLE_EDGES[a_index]),
                    "maxAngleDegrees": float(ANGLE_EDGES[a_index + 1]),
                    "points": int(accumulator.joint_points[d_index, a_index]),
                    "localResidualP90Meters": residual_p90,
                    "localResidualP95Meters": quantile_from_hist(accumulator.joint_residual_hist[d_index, a_index], RESIDUAL_EDGES, 0.95),
                    "recommendedGaussianNoiseSigmaMeters": max(0.0005, residual_p90 / 1.644854) if accumulator.joint_points[d_index, a_index] else 0.0,
                }
            )
    return distance_rows, angle_rows, joint_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    session = find_session(args.root)
    capture = session / "quest_clone_capture"
    rows = load_manifest(capture)
    if args.max_pairs > 0:
        rows = rows[: args.max_pairs]
    args.out.mkdir(parents=True, exist_ok=True)

    audit: dict[str, Any] = {
        "schema": "ScanCoverQuestCloneCaptureAudit/v1",
        "sourceSession": str(session),
        "manifestPairs": len(rows),
        "completePairs": 0,
        "missingFiles": [],
        "invalidFiles": [],
    }
    accumulator = Accumulator(args.low_confidence)
    valid_pairs = 0
    for pair_index, row in enumerate(rows):
        frame = row["frame"]
        right_path = capture / f"{frame}_right.scq3bin"
        left_path = capture / f"{frame}_left.scq3bin"
        metadata_path = capture / f"{frame}_capture.json"
        missing = [str(path) for path in (right_path, left_path, metadata_path) if not path.exists()]
        if missing:
            audit["missingFiles"].extend(missing)
            continue
        try:
            right_header, right_records = read_eye(right_path)
            left_header, left_records = read_eye(left_path)
            if right_header.eye != 1 or left_header.eye != 0:
                raise ValueError(f"Eye provenance mismatch: {frame} right={right_header.eye} left={left_header.eye}")
            collect_repeat = pair_index % max(1, args.repeat_frame_step) == 0
            accumulator.add_eye(right_header, right_records, args.local_stride, collect_repeat, args.repeat_pixel_stride, args.repeat_voxel, args.min_depth, args.max_depth)
            accumulator.add_eye(left_header, left_records, args.local_stride, collect_repeat, args.repeat_pixel_stride, args.repeat_voxel, args.min_depth, args.max_depth)
            valid_pairs += 1
        except (OSError, ValueError, struct.error) as error:
            audit["invalidFiles"].append({"frame": frame, "error": str(error)})
        if (pair_index + 1) % 100 == 0 or pair_index + 1 == len(rows):
            print(f"Processed {pair_index + 1}/{len(rows)} pairs", flush=True)

    audit["completePairs"] = valid_pairs
    audit["eyeFrames"] = valid_pairs * 2
    audit["totalPixels"] = accumulator.total_pixels
    audit["sensorValidPixels"] = accumulator.sensor_valid_pixels
    audit["workingRangeValidPixels"] = accumulator.working_valid_pixels
    audit["sensorValidRatio"] = accumulator.sensor_valid_pixels / max(1, accumulator.total_pixels)
    audit["workingRangeValidRatio"] = accumulator.working_valid_pixels / max(1, accumulator.total_pixels)
    audit["metadata"] = analyze_metadata(capture, rows)

    repeatability = build_repeatability(accumulator)
    distance_rows, angle_rows, joint_rows = build_profiles(accumulator, repeatability, valid_pairs * 2)
    profile = {
        "schema": "ScanCoverQuestDepthDegradationModel/v1",
        "sourceSession": str(session),
        "purpose": "Distance/angle-conditioned Quest observation model for Replica/Open3D offline experiments",
        "workingRangeMeters": {"min": args.min_depth, "max": args.max_depth},
        "confidenceRiskThreshold": args.low_confidence,
        "globalValidity": {
            "sensorValidRatio": audit["sensorValidRatio"],
            "sensorDropoutRatio": 1.0 - audit["sensorValidRatio"],
            "workingRangeValidRatio": audit["workingRangeValidRatio"],
        },
        "measurementSemantics": {
            "localResidual": "absolute center-to-four-neighbor-plane residual; single-frame roughness proxy",
            "positionSpread": "RMS world-position spread inside repeated 8cm voxels; repeatability proxy, not ground-truth error",
            "warning": "Do not interpret either proxy as absolute metric accuracy without a registered truth mesh",
        },
        "distanceProfile": distance_rows,
        "angleProfile": angle_rows,
        "distanceAngleProfile": joint_rows,
        "temporalModel": audit["metadata"].get("interEyeDeltaMs", {}),
        "readiness": {
            "complete": len(rows) > 0 and valid_pairs == len(rows) and not audit["invalidFiles"] and not audit["missingFiles"],
            "enoughForInitialReplicaCalibration": valid_pairs >= 1200,
            "requiresTruthMeshForAbsoluteBias": True,
        },
    }

    (args.out / "dataset_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "quest3_depth_degradation_model_v1.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.out / "distance_profile.csv", distance_rows)
    write_csv(args.out / "angle_profile.csv", angle_rows)
    write_csv(args.out / "distance_angle_profile.csv", joint_rows)

    report = [
        "# Quest Clone Capture Processing Report",
        "",
        f"- Source: `{session}`",
        f"- Complete binocular pairs: **{valid_pairs}/{len(rows)}**",
        f"- Dense pixels processed: **{accumulator.total_pixels:,}**",
        f"- Sensor-valid ratio: **{audit['sensorValidRatio'] * 100:.2f}%**",
        f"- Working-range valid ratio ({args.min_depth:.2f}-{args.max_depth:.2f} m): **{audit['workingRangeValidRatio'] * 100:.2f}%**",
        f"- Local geometry samples: **{accumulator.local_supported_points:,}/{accumulator.local_candidate_points:,}**",
        "",
        "The emitted model is suitable for initial distance/angle-conditioned Replica noise injection. "
        "It measures real Quest repeatability and local roughness, but absolute depth bias still requires a registered truth surface.",
    ]
    (args.out / "processing_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(profile["readiness"], indent=2, ensure_ascii=False))
    print(f"Wrote outputs: {args.out}")
    return 0 if profile["readiness"]["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
