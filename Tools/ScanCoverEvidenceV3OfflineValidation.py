#!/usr/bin/env python3
"""Offline validation for ScanCoverDepthEvidence/v3 Quest captures.

The script deliberately keeps three claims separate:

1. frozen-input fidelity (raw depth -> CPU reconstruction matches Quest GPU),
2. a paper-style projective TSDF baseline (Open3D voxel projection), and
3. empirical Quest degradation statistics (repeatability, not absolute truth).

Replica truth validation consumes the emitted degradation model in a separate
step.  Real captures from different Quest sessions are never fused together,
because their tracking origins are not guaranteed to share one world frame.
"""

from __future__ import annotations

import argparse
import binascii
import csv
import hashlib
import json
import math
import struct
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


MAGIC = b"SCDEVID1"
DISTANCE_BINS = (
    (0.35, 0.50),
    (0.50, 0.75),
    (0.75, 1.00),
    (1.00, 1.50),
    (1.50, 2.00),
    (2.00, 3.00),
    (3.00, 4.00),
    (4.00, 5.00),
)
ANGLE_BINS = ((0.0, 15.0), (15.0, 30.0), (30.0, 45.0), (45.0, 60.0), (60.0, 75.0), (75.0, 90.001))
ANALYSIS_BUFFERS = {
    "raw_depth_r32f",
    "depth_metrics_rgba32f",
    "world_position_raw_rgba32f",
    "world_position_neighbour_rgba32f",
    "world_normal_raw_rgba32f",
    "world_normal_neighbour_rgba32f",
    "diagnostics_rgba32f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Quest ScanCoverDepthEvidence/v3 offline")
    parser.add_argument("--root", type=Path, required=True, help="Directory containing Evidence_* sessions")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, nargs="*", default=None)
    parser.add_argument("--frames-per-session", type=int, default=24, help="Evenly selected frames for statistics")
    parser.add_argument("--tsdf-frames-per-session", type=int, default=24)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--stereo-stride", type=int, default=6)
    parser.add_argument("--voxel", type=float, default=0.025)
    parser.add_argument("--sdf-trunc", type=float, default=0.08)
    parser.add_argument("--min-depth", type=float, default=0.35)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--order-replay-sessions", type=int, default=1)
    parser.add_argument("--correction-replay-sessions", type=int, default=1)
    parser.add_argument("--correction-corrupt-frames", type=int, default=6)
    parser.add_argument("--correction-depth-offset", type=float, default=0.08)
    parser.add_argument("--correction-recovery-passes", type=int, default=2)
    parser.add_argument("--skip-tsdf", action="store_true")
    parser.add_argument("--verify-crc", action="store_true")
    return parser.parse_args()


def read_exact(stream, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"unexpected EOF: wanted {size}, got {len(data)}")
    return data


def read_i32(stream) -> int:
    return struct.unpack("<i", read_exact(stream, 4))[0]


def read_i64(stream) -> int:
    return struct.unpack("<q", read_exact(stream, 8))[0]


def read_u32(stream) -> int:
    return struct.unpack("<I", read_exact(stream, 4))[0]


def read_text(stream) -> str:
    size = read_i32(stream)
    if size < 0 or size > 16 * 1024 * 1024:
        raise ValueError(f"invalid text length {size}")
    return read_exact(stream, size).decode("utf-8")


def matrix4(values: Iterable[float]) -> np.ndarray:
    # Unity Matrix4x4[int] is column-major: index = row + column * 4.
    return np.asarray(list(values), dtype=np.float64).reshape((4, 4), order="F")


@dataclass
class EvidenceFrame:
    path: Path
    metadata: dict[str, Any]
    buffers: dict[str, np.ndarray]
    descriptors: dict[str, dict[str, Any]]


def load_frame(path: Path, requested: set[str], verify_crc: bool = False) -> EvidenceFrame:
    buffers: dict[str, np.ndarray] = {}
    descriptors: dict[str, dict[str, Any]] = {}
    with path.open("rb") as stream:
        if read_exact(stream, 8) != MAGIC:
            raise ValueError(f"bad magic: {path}")
        version = read_i32(stream)
        if version != 1:
            raise ValueError(f"unsupported container version {version}: {path}")
        metadata_size = read_i32(stream)
        buffer_count = read_i32(stream)
        metadata = json.loads(read_exact(stream, metadata_size).decode("utf-8"))
        for _ in range(buffer_count):
            name = read_text(stream)
            semantic = read_text(stream)
            graphics_format = read_text(stream)
            width = read_i32(stream)
            height = read_i32(stream)
            depth = read_i32(stream)
            byte_count = read_i64(stream)
            expected_crc = read_u32(stream)
            descriptors[name] = {
                "semantic": semantic,
                "graphicsFormat": graphics_format,
                "width": width,
                "height": height,
                "depth": depth,
                "bytes": byte_count,
            }
            if name not in requested:
                stream.seek(byte_count, 1)
                continue
            payload = read_exact(stream, byte_count)
            if verify_crc and (binascii.crc32(payload) & 0xFFFFFFFF) != expected_crc:
                raise ValueError(f"CRC mismatch for {name}: {path}")
            components = 1 if name == "raw_depth_r32f" else 4
            expected = width * height * depth * components
            data = np.frombuffer(payload, dtype="<f4")
            if data.size != expected:
                raise ValueError(f"shape mismatch for {name}: {data.size} != {expected}")
            buffers[name] = data.reshape(depth, height, width, components)
    if metadata.get("schema") != "ScanCoverDepthEvidence/v3":
        raise ValueError(f"wrong frame schema: {path}")
    return EvidenceFrame(path, metadata, buffers, descriptors)


def read_manifest(session: Path) -> list[dict[str, str]]:
    with (session / "manifest.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if row.get("status") == "ok"]


def select_evenly(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0 or count >= len(rows):
        return rows
    indices = np.linspace(0, len(rows) - 1, count).round().astype(np.int64)
    return [rows[int(index)] for index in np.unique(indices)]


def quantiles(values: Iterable[float]) -> dict[str, float]:
    data = np.asarray(list(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": int(len(data)),
        "mean": float(np.mean(data)),
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
    }


def rotation_angle_degrees(a: Iterable[float], b: Iterable[float]) -> float:
    qa = np.asarray(list(a), dtype=np.float64)
    qb = np.asarray(list(b), dtype=np.float64)
    qa /= max(1e-12, float(np.linalg.norm(qa)))
    qb /= max(1e-12, float(np.linalg.norm(qb)))
    dot = min(1.0, max(-1.0, abs(float(np.dot(qa, qb)))))
    return math.degrees(2.0 * math.acos(dot))


def bin_index(value: np.ndarray, bins: tuple[tuple[float, float], ...]) -> np.ndarray:
    result = np.full(value.shape, -1, dtype=np.int16)
    for index, (lo, hi) in enumerate(bins):
        result[(value >= lo) & (value < hi)] = index
    return result


@dataclass
class ProfileAccumulator:
    points: int = 0
    edge_risk: list[float] = field(default_factory=list)
    radial_jump: list[float] = field(default_factory=list)
    neighbour_support: list[float] = field(default_factory=list)
    filter_move: list[float] = field(default_factory=list)
    normal_delta_deg: list[float] = field(default_factory=list)
    stereo_residual: list[float] = field(default_factory=list)

    def append(
        self,
        edge: np.ndarray,
        jump: np.ndarray,
        support: np.ndarray,
        move: np.ndarray,
        normal_delta: np.ndarray,
    ) -> None:
        self.points += int(len(edge))
        self.edge_risk.extend(edge.astype(np.float32).tolist())
        self.radial_jump.extend(jump.astype(np.float32).tolist())
        self.neighbour_support.extend(support.astype(np.float32).tolist())
        self.filter_move.extend(move.astype(np.float32).tolist())
        self.normal_delta_deg.extend(normal_delta.astype(np.float32).tolist())


def add_profile_samples(
    accumulators: dict[Any, ProfileAccumulator],
    keys: np.ndarray,
    edge: np.ndarray,
    jump: np.ndarray,
    support: np.ndarray,
    move: np.ndarray,
    normal_delta: np.ndarray,
) -> None:
    for key in np.unique(keys):
        if int(key) < 0:
            continue
        mask = keys == key
        accumulators[int(key)].append(edge[mask], jump[mask], support[mask], move[mask], normal_delta[mask])


def cpu_reconstruction_audit(frame: EvidenceFrame, stride: int) -> dict[str, list[float]]:
    raw = frame.buffers["raw_depth_r32f"][..., 0]
    metrics = frame.buffers["depth_metrics_rgba32f"]
    world = frame.buffers["world_position_raw_rgba32f"]
    zparams = np.asarray(frame.metadata["zBufferParams"], dtype=np.float64)
    linear_delta: list[float] = []
    world_delta: list[float] = []
    radial_delta: list[float] = []
    for eye in range(2):
        raw_eye = raw[eye, ::stride, ::stride].astype(np.float64)
        metric_eye = metrics[eye, ::stride, ::stride].astype(np.float64)
        world_eye = world[eye, ::stride, ::stride].astype(np.float64)
        valid = (raw_eye > 0.0) & (metric_eye[..., 3] > 0.0) & (world_eye[..., 3] > 0.0)
        ndc_depth = raw_eye * 2.0 - 1.0
        denominator = ndc_depth + zparams[1]
        linear = np.divide(zparams[0], denominator, out=np.zeros_like(denominator), where=np.abs(denominator) > 1e-8)
        linear_delta.extend(np.abs(linear[valid] - metric_eye[..., 1][valid]).tolist())

        height, width = raw.shape[1:]
        ys = np.arange(0, height, stride, dtype=np.float64)
        xs = np.arange(0, width, stride, dtype=np.float64)
        x_grid, y_grid = np.meshgrid(xs, ys)
        ndc_x = ((x_grid + 0.5) / width) * 2.0 - 1.0
        ndc_y = ((y_grid + 0.5) / height) * 2.0 - 1.0
        clip = np.stack((ndc_x, ndc_y, ndc_depth, np.ones_like(ndc_depth)), axis=-1)
        inverse = matrix4(frame.metadata["inverseReprojectionMatrices"][eye])
        homogeneous = clip.reshape(-1, 4) @ inverse.T
        reconstructed_flat = np.zeros((len(homogeneous), 3), dtype=np.float64)
        safe_w = np.abs(homogeneous[:, 3]) > 1e-12
        reconstructed_flat[safe_w] = homogeneous[safe_w, :3] / homogeneous[safe_w, 3:4]
        reconstructed = reconstructed_flat.reshape(clip.shape[:-1] + (3,))
        world_error = np.linalg.norm(reconstructed - world_eye[..., :3], axis=-1)
        world_delta.extend(world_error[valid].tolist())
        eye_position = np.asarray(frame.metadata["eyeWorldPositions"][eye], dtype=np.float64)
        radial = np.linalg.norm(reconstructed - eye_position, axis=-1)
        radial_delta.extend(np.abs(radial[valid] - metric_eye[..., 2][valid]).tolist())
    return {"linear": linear_delta, "world": world_delta, "radial": radial_delta}


def exact_open3d_calibration(metadata: dict[str, Any], eye: int, width: int, height: int) -> tuple[o3d.camera.PinholeCameraIntrinsic, np.ndarray]:
    projection = matrix4(metadata["stereoProjectionMatrices"][eye])
    view = matrix4(metadata["stereoViewMatrices"][eye])
    fx = float(projection[0, 0]) * width * 0.5
    fy = float(projection[1, 1]) * height * 0.5
    cx = (1.0 - float(projection[0, 2])) * width * 0.5 - 0.5
    # Captured texture rows run bottom-to-top.  We flip the depth image and use
    # Open3D's conventional camera y-down, z-forward coordinates.
    cy = (1.0 + float(projection[1, 2])) * height * 0.5 - 0.5
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    axis_conversion = np.diag([1.0, -1.0, -1.0, 1.0])
    extrinsic = axis_conversion @ view
    return intrinsic, extrinsic


def new_tsdf(voxel: float, sdf_trunc: float) -> o3d.pipelines.integration.ScalableTSDFVolume:
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )


def integrate_frame(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
    frame: EvidenceFrame,
    min_depth: float,
    max_depth: float,
    depth_offset: float = 0.0,
) -> list[float]:
    metrics = frame.buffers["depth_metrics_rgba32f"]
    width = metrics.shape[2]
    height = metrics.shape[1]
    blank = o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8))
    timings: list[float] = []
    for eye in range(2):
        depth = metrics[eye, :, :, 1].copy()
        valid = (metrics[eye, :, :, 3] > 0.0) & np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
        depth[valid] += float(depth_offset)
        valid &= (depth >= min_depth) & (depth <= max_depth)
        depth[~valid] = 0.0
        depth = np.ascontiguousarray(np.flipud(depth).astype(np.float32))
        intrinsic, extrinsic = exact_open3d_calibration(frame.metadata, eye, width, height)
        depth_image = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            blank,
            depth_image,
            depth_scale=1.0,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
        )
        started = time.perf_counter()
        volume.integrate(rgbd, intrinsic, extrinsic)
        timings.append((time.perf_counter() - started) * 1000.0)
    return timings


def mesh_topology(mesh: o3d.geometry.TriangleMesh) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(triangles) == 0:
        return {
            "vertices": int(len(vertices)), "triangles": 0, "boundaryEdges": 0,
            "boundaryEdgesPerKTriangles": 0.0, "nonManifoldEdges": 0,
            "duplicateTriangles": 0, "degenerateTriangles": 0, "components": 0,
            "significantComponents50Triangles": 0,
        }
    degenerate = int(np.count_nonzero(
        (triangles[:, 0] == triangles[:, 1]) |
        (triangles[:, 1] == triangles[:, 2]) |
        (triangles[:, 0] == triangles[:, 2])
    ))
    canonical_triangles = np.sort(triangles, axis=1)
    _, triangle_counts = np.unique(canonical_triangles, axis=0, return_counts=True)
    duplicate = int(np.sum(np.maximum(0, triangle_counts - 1)))
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0)
    edges.sort(axis=1)
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    boundary = int(np.count_nonzero(edge_counts == 1))
    non_manifold = int(np.count_nonzero(edge_counts > 2))
    labels, counts, _ = mesh.cluster_connected_triangles()
    counts_array = np.asarray(counts, dtype=np.int64)
    return {
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "boundaryEdges": boundary,
        "boundaryEdgesPerKTriangles": boundary * 1000.0 / max(1, len(triangles)),
        "nonManifoldEdges": non_manifold,
        "duplicateTriangles": duplicate,
        "degenerateTriangles": degenerate,
        "components": int(len(counts_array)),
        "significantComponents50Triangles": int(np.count_nonzero(counts_array >= 50)),
    }


def sample_mesh(mesh: o3d.geometry.TriangleMesh, count: int = 30000) -> np.ndarray:
    if len(mesh.triangles) == 0:
        return np.empty((0, 3), dtype=np.float64)
    cloud = mesh.sample_points_uniformly(number_of_points=min(count, max(1000, len(mesh.triangles) * 2)))
    return np.asarray(cloud.points, dtype=np.float64)


def cloud_distance(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if len(source) == 0 or len(target) == 0:
        return quantiles([])
    tree = cKDTree(target)
    distance, _ = tree.query(source, k=1, workers=-1)
    return quantiles(distance)


def profile_row(accumulator: ProfileAccumulator, lo: float, hi: float, global_dropout: float) -> dict[str, Any]:
    stereo = quantiles(accumulator.stereo_residual)
    # Two independent eye observations contribute to one residual.  Dividing by
    # sqrt(2) produces a conservative per-observation repeatability sigma.  It
    # remains explicitly labelled as repeatability, not absolute depth bias.
    sigma = max(0.0005, min(0.03, stereo["p50"] / math.sqrt(2.0))) if stereo["count"] else 0.001
    edge = quantiles(accumulator.edge_risk)
    return {
        "min": lo,
        "max": hi,
        "points": accumulator.points,
        "edgeRiskRatio": float(np.mean(accumulator.edge_risk)) if accumulator.edge_risk else 0.0,
        "radialJumpMeters": quantiles(accumulator.radial_jump),
        "neighbourSupport": quantiles(accumulator.neighbour_support),
        "filterMoveMeters": quantiles(accumulator.filter_move),
        "normalDeltaDegrees": quantiles(accumulator.normal_delta_deg),
        "stereoRepeatabilityMeters": stereo,
        "recommendedGaussianNoiseSigmaMeters": sigma,
        "recommendedDropoutProbability": global_dropout,
    }


def resolve_sessions(args: argparse.Namespace) -> list[Path]:
    if args.sessions:
        sessions = [path.resolve() for path in args.sessions]
    else:
        sessions = sorted(path.resolve() for path in args.root.glob("Evidence_*") if path.is_dir())
    if not sessions:
        raise RuntimeError(f"no Evidence_* sessions found under {args.root}")
    return sessions


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    sessions = resolve_sessions(args)
    args.out.mkdir(parents=True, exist_ok=True)
    distance_profiles: dict[int, ProfileAccumulator] = defaultdict(ProfileAccumulator)
    angle_profiles: dict[int, ProfileAccumulator] = defaultdict(ProfileAccumulator)
    joint_profiles: dict[tuple[int, int], ProfileAccumulator] = defaultdict(ProfileAccumulator)
    linear_deltas: list[float] = []
    world_deltas: list[float] = []
    radial_deltas: list[float] = []
    all_readback_latency: list[float] = []
    selected_frame_rows: list[dict[str, Any]] = []
    total_pixels = 0
    sensor_valid_pixels = 0
    range_valid_pixels = 0
    raw_hashes: set[str] = set()
    duplicate_raw_hashes = 0
    session_reports: list[dict[str, Any]] = []
    raw_normal_valid_pixels = 0
    filtered_normal_valid_pixels = 0
    agreeing_normal_pixels = 0
    reliable_normal_pixels = 0
    edge_crossing_normal_pixels = 0

    for session_index, session in enumerate(sessions):
        rows = read_manifest(session)
        selected = select_evenly(rows, args.frames_per_session)
        tsdf_rows = select_evenly(rows, args.tsdf_frames_per_session)
        selected_paths = {str((session / row["file"]).resolve()) for row in selected}
        tsdf_paths = {str((session / row["file"]).resolve()) for row in tsdf_rows}
        processing_paths = sorted(selected_paths | tsdf_paths)
        frame_cache: dict[str, EvidenceFrame] = {}
        previous_position: np.ndarray | None = None
        previous_rotation: list[float] | None = None
        pose_step: list[float] = []
        rotation_step: list[float] = []

        for path_string in processing_paths:
            path = Path(path_string)
            frame = load_frame(path, ANALYSIS_BUFFERS, args.verify_crc)
            frame_cache[path_string] = frame
            if path_string not in selected_paths:
                continue
            audit = cpu_reconstruction_audit(frame, max(1, args.pixel_stride))
            linear_deltas.extend(audit["linear"])
            world_deltas.extend(audit["world"])
            radial_deltas.extend(audit["radial"])
            all_readback_latency.append(float(frame.metadata.get("gpuReadbackLatencySeconds", 0.0)))
            position = np.asarray(frame.metadata["cameraPositionAtRequest"], dtype=np.float64)
            rotation = frame.metadata["cameraRotationAtRequest"]
            if previous_position is not None:
                pose_step.append(float(np.linalg.norm(position - previous_position)))
                rotation_step.append(rotation_angle_degrees(previous_rotation or rotation, rotation))
            previous_position = position
            previous_rotation = rotation

            raw = frame.buffers["raw_depth_r32f"][..., 0]
            metrics = frame.buffers["depth_metrics_rgba32f"]
            world = frame.buffers["world_position_raw_rgba32f"]
            filtered_world = frame.buffers["world_position_neighbour_rgba32f"]
            normals = frame.buffers["world_normal_raw_rgba32f"]
            filtered_normals = frame.buffers["world_normal_neighbour_rgba32f"]
            diagnostics = frame.buffers["diagnostics_rgba32f"]
            total_pixels += int(raw.size)
            sensor_valid = np.isfinite(raw) & (raw > 0.0)
            range_valid = (
                (metrics[..., 3] > 0.0)
                & (metrics[..., 1] >= args.min_depth)
                & (metrics[..., 1] <= args.max_depth)
            )
            raw_normal_valid = normals[..., 3] > 0.0
            filtered_normal_valid = filtered_normals[..., 3] > 0.0
            both_normal_valid = range_valid & raw_normal_valid & filtered_normal_valid
            normal_agreement = np.abs(np.sum(normals[..., :3] * filtered_normals[..., :3], axis=-1))
            agreeing_normal = both_normal_valid & (normal_agreement >= math.cos(math.radians(20.0)))
            edge_risk = diagnostics[..., 3] > 0.0
            reliable_normal = agreeing_normal & ~edge_risk
            sensor_valid_pixels += int(np.count_nonzero(sensor_valid))
            range_valid_pixels += int(np.count_nonzero(range_valid))
            raw_normal_valid_pixels += int(np.count_nonzero(range_valid & raw_normal_valid))
            filtered_normal_valid_pixels += int(np.count_nonzero(range_valid & filtered_normal_valid))
            agreeing_normal_pixels += int(np.count_nonzero(agreeing_normal))
            reliable_normal_pixels += int(np.count_nonzero(reliable_normal))
            edge_crossing_normal_pixels += int(np.count_nonzero(range_valid & raw_normal_valid & edge_risk))

            stride = max(1, args.pixel_stride)
            for eye in range(2):
                metric = metrics[eye, ::stride, ::stride]
                point = world[eye, ::stride, ::stride]
                filtered_point = filtered_world[eye, ::stride, ::stride]
                normal = normals[eye, ::stride, ::stride]
                filtered_normal = filtered_normals[eye, ::stride, ::stride]
                diagnostic = diagnostics[eye, ::stride, ::stride]
                valid = (
                    (metric[..., 3] > 0.0) & (point[..., 3] > 0.0) & (normal[..., 3] > 0.0) &
                    (metric[..., 2] >= args.min_depth) & (metric[..., 2] <= args.max_depth)
                )
                if not np.any(valid):
                    continue
                eye_position = np.asarray(frame.metadata["eyeWorldPositions"][eye], dtype=np.float64)
                toward_eye = eye_position - point[..., :3]
                toward_eye /= np.maximum(1e-9, np.linalg.norm(toward_eye, axis=-1, keepdims=True))
                normal_xyz = normal[..., :3]
                incidence_cos = np.clip(np.abs(np.sum(normal_xyz * toward_eye, axis=-1)), 0.0, 1.0)
                angle = np.degrees(np.arccos(incidence_cos))
                normal_dot = np.clip(np.abs(np.sum(normal_xyz * filtered_normal[..., :3], axis=-1)), 0.0, 1.0)
                normal_delta = np.degrees(np.arccos(normal_dot))
                move = np.linalg.norm(filtered_point[..., :3] - point[..., :3], axis=-1)
                distance_index = bin_index(metric[..., 2], DISTANCE_BINS)
                angle_index = bin_index(angle, ANGLE_BINS)
                values = (
                    diagnostic[..., 3][valid], diagnostic[..., 0][valid], diagnostic[..., 2][valid],
                    move[valid], normal_delta[valid]
                )
                add_profile_samples(distance_profiles, distance_index[valid], *values)
                add_profile_samples(angle_profiles, angle_index[valid], *values)
                joint_key = distance_index.astype(np.int32) * 100 + angle_index.astype(np.int32)
                add_profile_samples(joint_profiles, joint_key[valid], *values)

            stereo_stride = max(1, args.stereo_stride)
            left = world[0, ::stereo_stride, ::stereo_stride]
            right = world[1, ::stereo_stride, ::stereo_stride]
            left_metrics = metrics[0, ::stereo_stride, ::stereo_stride]
            left_normals = normals[0, ::stereo_stride, ::stereo_stride]
            left_valid = (left[..., 3] > 0.0) & (left_normals[..., 3] > 0.0)
            right_valid = right[..., 3] > 0.0
            left_points = left[..., :3][left_valid]
            right_points = right[..., :3][right_valid]
            if len(left_points) and len(right_points):
                distance, _ = cKDTree(right_points).query(left_points, k=1, workers=-1)
                keep = np.isfinite(distance) & (distance <= 0.10)
                left_distance = left_metrics[..., 2][left_valid]
                eye_position = np.asarray(frame.metadata["eyeWorldPositions"][0], dtype=np.float64)
                ray = eye_position - left_points
                ray /= np.maximum(1e-9, np.linalg.norm(ray, axis=1, keepdims=True))
                incidence = np.clip(np.abs(np.sum(left_normals[..., :3][left_valid] * ray, axis=1)), 0.0, 1.0)
                left_angle = np.degrees(np.arccos(incidence))
                di = bin_index(left_distance, DISTANCE_BINS)
                ai = bin_index(left_angle, ANGLE_BINS)
                for index, residual in zip(di[keep], distance[keep]):
                    if index >= 0:
                        distance_profiles[int(index)].stereo_residual.append(float(residual))
                for index, residual in zip(ai[keep], distance[keep]):
                    if index >= 0:
                        angle_profiles[int(index)].stereo_residual.append(float(residual))
                for d_index, a_index, residual in zip(di[keep], ai[keep], distance[keep]):
                    if d_index >= 0 and a_index >= 0:
                        joint_profiles[int(d_index) * 100 + int(a_index)].stereo_residual.append(float(residual))

            raw_sha = next((row["raw_sha256"] for row in selected if (session / row["file"]).resolve() == path.resolve()), "")
            if raw_sha:
                if raw_sha in raw_hashes:
                    duplicate_raw_hashes += 1
                raw_hashes.add(raw_sha)
            selected_frame_rows.append({
                "session": session.name,
                "frame": int(frame.metadata["frameIndex"]),
                "unityFrame": int(frame.metadata["unityFrame"]),
                "validRatio": float(np.mean(metrics[..., 3] > 0.0)),
                "edgeRiskRatio": float(np.mean(diagnostics[..., 3] > 0.0)),
                "rawNormalCoverageOfWorkingRange": float(
                    np.count_nonzero(range_valid & raw_normal_valid) / max(1, np.count_nonzero(range_valid))
                ),
                "filteredNormalCoverageOfWorkingRange": float(
                    np.count_nonzero(range_valid & filtered_normal_valid) / max(1, np.count_nonzero(range_valid))
                ),
                "normalAgreementCoverageOfWorkingRange": float(
                    np.count_nonzero(agreeing_normal) / max(1, np.count_nonzero(range_valid))
                ),
                "normalSelfConsistentCoverageOfWorkingRange": float(
                    np.count_nonzero(reliable_normal) / max(1, np.count_nonzero(range_valid))
                ),
                "edgeCrossingNormalCoverageOfWorkingRange": float(
                    np.count_nonzero(range_valid & raw_normal_valid & edge_risk) / max(1, np.count_nonzero(range_valid))
                ),
                "meanRadialMeters": float(np.mean(metrics[..., 2][metrics[..., 3] > 0.0])),
                "gpuReadbackLatencyMs": float(frame.metadata["gpuReadbackLatencySeconds"]) * 1000.0,
            })

        tsdf_report: dict[str, Any] = {"skipped": bool(args.skip_tsdf)}
        if not args.skip_tsdf:
            ordered_frames = [frame_cache[str((session / row["file"]).resolve())] for row in tsdf_rows]
            volume = new_tsdf(args.voxel, args.sdf_trunc)
            integration_ms: list[float] = []
            checkpoints: list[dict[str, Any]] = []
            checkpoint_indices = set(np.linspace(1, len(ordered_frames), min(4, len(ordered_frames))).round().astype(int).tolist())
            for index, frame in enumerate(ordered_frames, start=1):
                integration_ms.extend(integrate_frame(volume, frame, args.min_depth, args.max_depth))
                if index in checkpoint_indices:
                    checkpoint_mesh = volume.extract_triangle_mesh()
                    checkpoint_mesh.remove_duplicated_vertices()
                    checkpoint_mesh.remove_degenerate_triangles()
                    # Keep the extracted vertices themselves.  Independent
                    # uniform surface samples introduce a centimetre-scale
                    # Monte-Carlo floor even when two meshes are identical,
                    # which is unacceptable for a temporal-correction audit.
                    checkpoints.append({
                        "frames": index,
                        "topology": mesh_topology(checkpoint_mesh),
                        "points": np.asarray(checkpoint_mesh.vertices, dtype=np.float64).copy(),
                    })
            mesh = volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            mesh.remove_duplicated_vertices()
            mesh.remove_degenerate_triangles()
            mesh_path = args.out / f"{session.name}_projective_tsdf.ply"
            o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False)
            final_points = np.asarray(mesh.vertices, dtype=np.float64)
            checkpoint_summary = []
            for checkpoint in checkpoints:
                checkpoint_summary.append({
                    "frames": checkpoint["frames"],
                    "topology": checkpoint["topology"],
                    "checkpointToFinalMeters": cloud_distance(checkpoint["points"], final_points),
                    "finalToCheckpointMeters": cloud_distance(final_points, checkpoint["points"]),
                })
            tsdf_report = {
                "skipped": False,
                "selectedFrames": len(ordered_frames),
                "selectedEyeFrames": len(ordered_frames) * 2,
                "voxelMeters": args.voxel,
                "truncationMeters": args.sdf_trunc,
                "integrationMsPerEye": quantiles(integration_ms),
                "mesh": str(mesh_path),
                "topology": mesh_topology(mesh),
                "checkpoints": checkpoint_summary,
            }
            if session_index < args.order_replay_sessions:
                reverse_volume = new_tsdf(args.voxel, args.sdf_trunc)
                for frame in reversed(ordered_frames):
                    integrate_frame(reverse_volume, frame, args.min_depth, args.max_depth)
                reverse_mesh = reverse_volume.extract_triangle_mesh()
                reverse_mesh.remove_duplicated_vertices()
                reverse_mesh.remove_degenerate_triangles()
                # The extraction is deterministic.  Comparing actual vertices
                # avoids mistaking independent random surface samples for an
                # order-dependent TSDF difference.
                reverse_points = np.asarray(reverse_mesh.vertices, dtype=np.float64)
                reverse_path = args.out / f"{session.name}_projective_tsdf_reverse.ply"
                o3d.io.write_triangle_mesh(str(reverse_path), reverse_mesh, write_ascii=False)
                tsdf_report["orderInvariance"] = {
                    "reverseMesh": str(reverse_path),
                    "forwardToReverseMeters": cloud_distance(np.asarray(mesh.vertices, dtype=np.float64), reverse_points),
                    "reverseToForwardMeters": cloud_distance(reverse_points, np.asarray(mesh.vertices, dtype=np.float64)),
                    "reverseTopology": mesh_topology(reverse_mesh),
                }
            if session_index < args.correction_replay_sessions:
                correction_volume = new_tsdf(args.voxel, args.sdf_trunc)
                corrupt_count = min(max(1, args.correction_corrupt_frames), len(ordered_frames))
                for frame_index, frame in enumerate(ordered_frames):
                    integrate_frame(
                        correction_volume,
                        frame,
                        args.min_depth,
                        args.max_depth,
                        args.correction_depth_offset if frame_index < corrupt_count else 0.0,
                    )
                mixed_mesh = correction_volume.extract_triangle_mesh()
                mixed_mesh.remove_duplicated_vertices()
                mixed_mesh.remove_degenerate_triangles()
                mixed_points = np.asarray(mixed_mesh.vertices, dtype=np.float64)
                mixed_to_clean = cloud_distance(mixed_points, final_points)
                clean_to_mixed = cloud_distance(final_points, mixed_points)

                for _ in range(max(0, args.correction_recovery_passes)):
                    for frame in ordered_frames:
                        integrate_frame(correction_volume, frame, args.min_depth, args.max_depth)
                recovered_mesh = correction_volume.extract_triangle_mesh()
                recovered_mesh.remove_duplicated_vertices()
                recovered_mesh.remove_degenerate_triangles()
                recovered_points = np.asarray(recovered_mesh.vertices, dtype=np.float64)
                recovered_to_clean = cloud_distance(recovered_points, final_points)
                clean_to_recovered = cloud_distance(final_points, recovered_points)
                before_p95 = max(mixed_to_clean["p95"], clean_to_mixed["p95"])
                after_p95 = max(recovered_to_clean["p95"], clean_to_recovered["p95"])
                recovered_path = args.out / f"{session.name}_projective_tsdf_recovered.ply"
                o3d.io.write_triangle_mesh(str(recovered_path), recovered_mesh, write_ascii=False)
                tsdf_report["correctionRecovery"] = {
                    "reference": "Open3D projective TSDF; this is a recovery envelope, not Unity production proof",
                    "corruptPrefixFrames": corrupt_count,
                    "depthOffsetMeters": args.correction_depth_offset,
                    "cleanRecoveryPasses": max(0, args.correction_recovery_passes),
                    "beforeRecovery": {
                        "mixedToCleanMeters": mixed_to_clean,
                        "cleanToMixedMeters": clean_to_mixed,
                        "topology": mesh_topology(mixed_mesh),
                    },
                    "afterRecovery": {
                        "recoveredToCleanMeters": recovered_to_clean,
                        "cleanToRecoveredMeters": clean_to_recovered,
                        "topology": mesh_topology(recovered_mesh),
                        "mesh": str(recovered_path),
                    },
                    "symmetricP95BeforeMeters": before_p95,
                    "symmetricP95AfterMeters": after_p95,
                    "p95ReductionRatio": 1.0 - after_p95 / max(before_p95, 1e-12),
                    "passed": bool(
                        after_p95 <= before_p95 * 0.8
                        and mesh_topology(recovered_mesh)["nonManifoldEdges"] == 0
                    ),
                }

        session_report = {
            "session": str(session),
            "manifestFrames": len(rows),
            "statisticsFrames": len(selected),
            "poseStepMeters": quantiles(pose_step),
            "rotationStepDegrees": quantiles(rotation_step),
            "projectiveTsdf": tsdf_report,
        }
        session_reports.append(session_report)
        (args.out / f"{session.name}_report.json").write_text(json.dumps(session_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[evidence {session_index + 1}/{len(sessions)}] {session.name}: stats={len(selected)} tsdf={len(tsdf_rows)}", flush=True)

    sensor_valid_ratio = sensor_valid_pixels / max(1, total_pixels)
    range_valid_ratio = range_valid_pixels / max(1, total_pixels)
    dropout = 1.0 - sensor_valid_ratio
    normal_supply = {
        "workingRangePixels": range_valid_pixels,
        "rawNormalCoverageOfWorkingRange": raw_normal_valid_pixels / max(1, range_valid_pixels),
        "filteredNormalCoverageOfWorkingRange": filtered_normal_valid_pixels / max(1, range_valid_pixels),
        "normalAgreementCoverageOfWorkingRange": agreeing_normal_pixels / max(1, range_valid_pixels),
        "normalSelfConsistentCoverageOfWorkingRange": reliable_normal_pixels / max(1, range_valid_pixels),
        "edgeCrossingNormalCoverageOfWorkingRange": edge_crossing_normal_pixels / max(1, range_valid_pixels),
        "reliabilityDefinition": "raw and neighbour-filtered normals agree within 20 degrees and the frozen four-neighbour depth-edge diagnostic is clear",
        "claimBoundary": "conservative self-consistency proxy, not truth-normal accuracy",
    }
    distance_rows: list[dict[str, Any]] = []
    for index, (lo, hi) in enumerate(DISTANCE_BINS):
        row = profile_row(distance_profiles[index], lo, hi, dropout)
        row["minDepthMeters"] = row.pop("min")
        row["maxDepthMeters"] = row.pop("max")
        distance_rows.append(row)
    angle_rows: list[dict[str, Any]] = []
    for index, (lo, hi) in enumerate(ANGLE_BINS):
        row = profile_row(angle_profiles[index], lo, hi, dropout)
        row["minAngleDegrees"] = row.pop("min")
        row["maxAngleDegrees"] = row.pop("max")
        angle_rows.append(row)
    joint_rows: list[dict[str, Any]] = []
    for distance_index, (distance_lo, distance_hi) in enumerate(DISTANCE_BINS):
        for angle_index, (angle_lo, angle_hi) in enumerate(ANGLE_BINS):
            row = profile_row(joint_profiles[distance_index * 100 + angle_index], distance_lo, distance_hi, dropout)
            row["minDepthMeters"] = row.pop("min")
            row["maxDepthMeters"] = row.pop("max")
            row["minAngleDegrees"] = angle_lo
            row["maxAngleDegrees"] = angle_hi
            joint_rows.append(row)

    degradation_model = {
        "schema": "ScanCoverQuestDepthDegradationModel/v2",
        "sourceSessions": [str(session) for session in sessions],
        "purpose": "Distance/angle-conditioned Quest 3 observation model for Replica/Open3D validation",
        "workingRangeMeters": {"min": args.min_depth, "max": args.max_depth},
        "globalValidity": {
            "sensorValidRatio": sensor_valid_ratio,
            "sensorDropoutRatio": dropout,
            "workingRangeValidRatio": range_valid_ratio,
        },
        "measurementSemantics": {
            "stereoRepeatability": "nearest surface disagreement between simultaneous left/right observations; not absolute truth error",
            "filterMove": "Quest GPU neighbour-filter displacement from the preserved raw world point",
            "radialJump": "maximum four-neighbour radial jump from the frozen Quest GPU diagnostic",
            "warning": "Absolute depth bias remains a Replica/registered-truth question; do not treat repeatability as metric accuracy",
        },
        "distanceProfile": distance_rows,
        "angleProfile": angle_rows,
        "distanceAngleProfile": joint_rows,
    }
    model_path = args.out / "quest3_depth_degradation_model_v2.json"
    model_path.write_text(json.dumps(degradation_model, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out / "selected_frame_metrics.csv", selected_frame_rows)

    cross_implementation = {
        "rawDepthLinearizationDeltaMeters": quantiles(linear_deltas),
        "inverseReprojectionWorldDeltaMeters": quantiles(world_deltas),
        "reconstructedRadialDeltaMeters": quantiles(radial_deltas),
    }
    gates = {
        "allSessionsComplete": all(report["manifestFrames"] > 0 for report in session_reports),
        "noSelectedRawDuplicates": duplicate_raw_hashes == 0,
        "cpuLinearizationP95Below0_1mm": cross_implementation["rawDepthLinearizationDeltaMeters"]["p95"] <= 0.0001,
        "cpuWorldReprojectionP95Below0_1mm": cross_implementation["inverseReprojectionWorldDeltaMeters"]["p95"] <= 0.0001,
        "cpuRadialP95Below0_1mm": cross_implementation["reconstructedRadialDeltaMeters"]["p95"] <= 0.0001,
        "projectiveMeshesGenerated": args.skip_tsdf or all(
            report["projectiveTsdf"].get("topology", {}).get("triangles", 0) > 0 for report in session_reports
        ),
        "projectiveMeshesNonManifoldFree": args.skip_tsdf or all(
            report["projectiveTsdf"].get("topology", {}).get("nonManifoldEdges", 1) == 0 for report in session_reports
        ),
        "projectiveCorrectionRecovery": args.skip_tsdf or args.correction_replay_sessions <= 0 or all(
            report["projectiveTsdf"].get("correctionRecovery", {}).get("passed", False)
            for report in session_reports[:min(args.correction_replay_sessions, len(session_reports))]
        ),
    }
    report = {
        "schema": "ScanCoverEvidenceV3OfflineValidation/v1",
        "sourceRoot": str(args.root.resolve()),
        "output": str(args.out.resolve()),
        "parameters": {
            "framesPerSession": args.frames_per_session,
            "tsdfFramesPerSession": args.tsdf_frames_per_session,
            "pixelStride": args.pixel_stride,
            "stereoStride": args.stereo_stride,
            "voxelMeters": args.voxel,
            "truncationMeters": args.sdf_trunc,
            "correctionReplaySessions": args.correction_replay_sessions,
            "correctionCorruptFrames": args.correction_corrupt_frames,
            "correctionDepthOffsetMeters": args.correction_depth_offset,
            "correctionRecoveryPasses": args.correction_recovery_passes,
        },
        "sessions": session_reports,
        "selectedFrames": len(selected_frame_rows),
        "selectedRawHashes": len(raw_hashes),
        "duplicateSelectedRawHashes": duplicate_raw_hashes,
        "totalPixels": total_pixels,
        "sensorValidPixels": sensor_valid_pixels,
        "workingRangeValidPixels": range_valid_pixels,
        "globalValidity": degradation_model["globalValidity"],
        "normalSupply": normal_supply,
        "gpuReadbackLatencySeconds": quantiles(all_readback_latency),
        "crossImplementation": cross_implementation,
        "degradationModel": str(model_path),
        "gates": gates,
        "passed": all(gates.values()),
        "claimBoundary": {
            "realQuestRepeatabilityMeasured": True,
            "absoluteDepthBiasMeasured": False,
            "replicaTruthValidationCompleted": False,
            "questRuntimeBudgetMeasuredByOfflineRun": False,
        },
    }
    report_path = args.out / "offline_validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Quest Evidence v3 Offline Validation",
        "",
        f"- sessions: **{len(sessions)}**",
        f"- selected frames: **{len(selected_frame_rows)}**",
        f"- projective TSDF: **{'skipped' if args.skip_tsdf else 'Open3D projective voxel integration'}**",
        f"- passed: **{report['passed']}**",
        "",
        "## Cross implementation",
        "",
        f"- depth linearization p95: {cross_implementation['rawDepthLinearizationDeltaMeters']['p95']:.9f} m",
        f"- world reprojection p95: {cross_implementation['inverseReprojectionWorldDeltaMeters']['p95']:.9f} m",
        f"- radial reconstruction p95: {cross_implementation['reconstructedRadialDeltaMeters']['p95']:.9f} m",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {name}: {value}" for name, value in gates.items())
    lines.extend([
        "",
        "The real captures measure repeatability and implementation parity. Absolute bias and topology accuracy require the following Replica truth pass.",
    ])
    (args.out / "offline_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "gates": gates, "out": str(args.out)}, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
