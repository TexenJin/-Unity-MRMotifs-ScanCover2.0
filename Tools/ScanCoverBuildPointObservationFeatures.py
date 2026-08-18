#!/usr/bin/env python3
"""Build per-voxel observation features from a ScanCover multi-frame session.

This is the bridge between Quest3 capture data and offline teacher/student
experiments.  Frame-local vertex ids are not stable, so observations are joined
by world-space voxel keys.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import open3d as o3d
except Exception:  # pragma: no cover - csv output remains useful without Open3D.
    o3d = None


DISTANCE_BINS = [
    (0.0, 0.5, "0.0-0.5m"),
    (0.5, 1.0, "0.5-1.0m"),
    (1.0, 1.5, "1.0-1.5m"),
    (1.5, 2.0, "1.5-2.0m"),
    (2.0, 3.0, "2.0-3.0m"),
    (3.0, 5.0, "3.0-5.0m"),
    (5.0, 8.0, "5.0-8.0m"),
    (8.0, float("inf"), "8.0m+"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path, help="ScanCover_MultiFrame_... session folder.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory. Defaults to session/observation_features.")
    parser.add_argument("--voxel", type=float, default=0.02, help="World-space feature voxel size in meters.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit for quick tests.")
    parser.add_argument("--crease-angle", type=float, default=28.0, help="Triangle normal spread threshold in degrees.")
    parser.add_argument("--stable-min-frames", type=int, default=3)
    parser.add_argument("--risk-ratio-threshold", type=float, default=0.35)
    parser.add_argument("--write-ply", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_manifest(session: Path) -> list[dict[str, str]]:
    manifest = session / "session_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    lines = manifest.read_text(encoding="utf-8-sig").splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.startswith("frame,")), None)
    if header_index is None:
        raise RuntimeError(f"Could not find manifest header in {manifest}")
    return list(csv.DictReader(lines[header_index:]))


def read_csv_after_header(path: Path, header_prefix: str) -> Iterable[dict[str, str]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.startswith(header_prefix)), None)
    if header_index is None:
        raise RuntimeError(f"Could not find {header_prefix!r} header in {path}")
    return csv.DictReader(lines[header_index:])


def parse_vec(row: dict[str, str], x: str, y: str, z: str) -> np.ndarray:
    return np.asarray((float(row[x]), float(row[y]), float(row[z])), dtype=np.float64)


def normalize(v: np.ndarray, fallback: tuple[float, float, float] = (0.0, 1.0, 0.0)) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length > 1e-8 and np.isfinite(length):
        return v / length
    return np.asarray(fallback, dtype=np.float64)


def distance_bin(distance: float) -> str:
    for lo, hi, label in DISTANCE_BINS:
        if lo <= distance < hi:
            return label
    return "8.0m+"


def normal_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    dot = abs(float(np.dot(normalize(a), normalize(b))))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def read_camera_pose(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    camera_path = Path(row.get("cameraJson") or "")
    if camera_path.exists():
        try:
            data = json.loads(camera_path.read_text(encoding="utf-8-sig"))
            pose = data.get("pose") or {}
            position = np.asarray(pose.get("position", []), dtype=np.float64)
            forward = np.asarray(pose.get("forward", []), dtype=np.float64)
            if position.shape == (3,) and forward.shape == (3,):
                return position, normalize(forward, (0.0, 0.0, 1.0))
        except Exception:
            pass

    position = np.asarray(
        (
            float(row.get("poseX", 0.0)),
            float(row.get("poseY", 0.0)),
            float(row.get("poseZ", 0.0)),
        ),
        dtype=np.float64,
    )
    return position, np.asarray((0.0, 0.0, 1.0), dtype=np.float64)


def read_vertices(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices: list[int] = []
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for row in read_csv_after_header(path, "index,"):
        indices.append(int(row["index"]))
        points.append(parse_vec(row, "worldX", "worldY", "worldZ"))
        normals.append(normalize(parse_vec(row, "normalWorldX", "normalWorldY", "normalWorldZ")))
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(points, dtype=np.float64),
        np.asarray(normals, dtype=np.float64),
    )


def read_triangle_risk(path: Path, vertex_count: int, crease_angle: float) -> tuple[np.ndarray, np.ndarray]:
    triangle_counts = np.zeros(vertex_count, dtype=np.int32)
    max_normal_angle = np.zeros(vertex_count, dtype=np.float64)
    normal_sums: list[np.ndarray] = [np.zeros(3, dtype=np.float64) for _ in range(vertex_count)]
    face_normals_by_vertex: list[list[np.ndarray]] = [[] for _ in range(vertex_count)]

    if not path.exists():
        return triangle_counts, max_normal_angle

    for row in read_csv_after_header(path, "triangle,"):
        try:
            verts = [int(row["index0"]), int(row["index1"]), int(row["index2"])]
            normal = normalize(parse_vec(row, "normalX", "normalY", "normalZ"))
        except Exception:
            continue
        for idx in verts:
            if 0 <= idx < vertex_count:
                triangle_counts[idx] += 1
                normal_sums[idx] += normal
                face_normals_by_vertex[idx].append(normal)

    for i, face_normals in enumerate(face_normals_by_vertex):
        if len(face_normals) < 2:
            continue
        avg = normalize(normal_sums[i])
        max_normal_angle[i] = max(normal_angle_deg(avg, n) for n in face_normals)
    return triangle_counts, max_normal_angle


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

    @property
    def variance(self) -> float:
        return self.m2 / max(1, self.count - 1) if self.count > 1 else 0.0


@dataclass
class VectorRunningStats:
    count: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    m2: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def add(self, value: np.ndarray) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance_sum(self) -> float:
        if self.count <= 1:
            return 0.0
        return float(np.sum(self.m2 / (self.count - 1)))


@dataclass
class VoxelFeature:
    key: tuple[int, int, int]
    hit_count: int = 0
    frames: set[str] = field(default_factory=set)
    position: VectorRunningStats = field(default_factory=VectorRunningStats)
    normal_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    distance: RunningStats = field(default_factory=RunningStats)
    view_depth: RunningStats = field(default_factory=RunningStats)
    view_angle: RunningStats = field(default_factory=RunningStats)
    face_normal_angle: RunningStats = field(default_factory=RunningStats)
    triangle_count: RunningStats = field(default_factory=RunningStats)
    boundary_risk_count: int = 0
    crease_risk_count: int = 0
    risk_count: int = 0
    distance_bin_counts: dict[str, int] = field(default_factory=dict)

    def add(
        self,
        frame: str,
        point: np.ndarray,
        normal: np.ndarray,
        camera_pos: np.ndarray,
        camera_forward: np.ndarray,
        triangle_count: int,
        max_face_angle: float,
        crease_angle: float,
    ) -> None:
        self.hit_count += 1
        self.frames.add(frame)
        self.position.add(point)
        self.normal_sum += normal

        to_point = point - camera_pos
        distance = float(np.linalg.norm(to_point))
        ray_dir = normalize(to_point, (0.0, 0.0, 1.0))
        view_depth = float(np.dot(to_point, camera_forward))
        view_angle = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(np.dot(-ray_dir, normal)))))))

        boundary_risk = triangle_count < 3
        crease_risk = max_face_angle >= crease_angle
        any_risk = boundary_risk or crease_risk

        self.distance.add(distance)
        self.view_depth.add(view_depth)
        self.view_angle.add(view_angle)
        self.face_normal_angle.add(max_face_angle)
        self.triangle_count.add(float(triangle_count))
        if boundary_risk:
            self.boundary_risk_count += 1
        if crease_risk:
            self.crease_risk_count += 1
        if any_risk:
            self.risk_count += 1
        label = distance_bin(distance)
        self.distance_bin_counts[label] = self.distance_bin_counts.get(label, 0) + 1

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def mean_normal(self) -> np.ndarray:
        return normalize(self.normal_sum)

    def normal_variance(self) -> float:
        normal = self.mean_normal
        length = float(np.linalg.norm(self.normal_sum))
        if self.hit_count <= 0:
            return 1.0
        return max(0.0, 1.0 - length / self.hit_count)

    def stability_score(self) -> float:
        frame_term = min(1.0, self.frame_count / 5.0)
        hit_term = min(1.0, self.hit_count / 20.0)
        position_penalty = min(1.0, math.sqrt(max(0.0, self.position.variance_sum)) / 0.04)
        normal_penalty = min(1.0, self.normal_variance() / 0.25)
        risk_penalty = self.risk_count / max(1, self.hit_count)
        return max(0.0, frame_term * 0.45 + hit_term * 0.25 + (1.0 - position_penalty) * 0.15 + (1.0 - normal_penalty) * 0.10 + (1.0 - risk_penalty) * 0.05)

    def dominant_distance_bin(self) -> str:
        if not self.distance_bin_counts:
            return ""
        return max(self.distance_bin_counts.items(), key=lambda item: item[1])[0]


def build_features(rows: list[dict[str, str]], voxel_size: float, crease_angle: float, max_frames: int) -> dict[tuple[int, int, int], VoxelFeature]:
    voxels: dict[tuple[int, int, int], VoxelFeature] = {}
    selected_rows = rows[:max_frames] if max_frames > 0 else rows

    for frame_index, row in enumerate(selected_rows):
        frame_name = row.get("frame") or f"frame_{frame_index:04d}"
        vertices_path = Path(row.get("verticesCsv") or "")
        triangles_path = Path(row.get("trianglesCsv") or "")
        if not vertices_path.exists():
            continue

        indices, points, normals = read_vertices(vertices_path)
        triangle_counts, max_face_angles = read_triangle_risk(triangles_path, len(points), crease_angle)
        camera_pos, camera_forward = read_camera_pose(row)

        keys = np.floor(points / voxel_size).astype(np.int64)
        for local_i, point in enumerate(points):
            key = (int(keys[local_i, 0]), int(keys[local_i, 1]), int(keys[local_i, 2]))
            feature = voxels.get(key)
            if feature is None:
                feature = VoxelFeature(key=key)
                voxels[key] = feature
            feature.add(
                frame=frame_name,
                point=point,
                normal=normals[local_i],
                camera_pos=camera_pos,
                camera_forward=camera_forward,
                triangle_count=int(triangle_counts[local_i]),
                max_face_angle=float(max_face_angles[local_i]),
                crease_angle=crease_angle,
            )
    return voxels


def write_features_csv(path: Path, voxels: dict[tuple[int, int, int], VoxelFeature], voxel_size: float) -> None:
    fieldnames = [
        "voxelX",
        "voxelY",
        "voxelZ",
        "centerX",
        "centerY",
        "centerZ",
        "meanX",
        "meanY",
        "meanZ",
        "normalX",
        "normalY",
        "normalZ",
        "hit_count",
        "frame_count",
        "mean_distance",
        "min_distance",
        "max_distance",
        "dominant_distance_bin",
        "mean_view_depth",
        "mean_view_angle",
        "max_view_angle",
        "position_variance",
        "normal_variance",
        "depth_variance",
        "distance_variance",
        "mean_triangle_count",
        "mean_face_normal_angle",
        "boundary_risk_ratio",
        "crease_risk_ratio",
        "any_risk_ratio",
        "stability_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, feature in sorted(voxels.items()):
            mean = feature.position.mean
            normal = feature.mean_normal
            center = (np.asarray(key, dtype=np.float64) + 0.5) * voxel_size
            hit_count = max(1, feature.hit_count)
            writer.writerow(
                {
                    "voxelX": key[0],
                    "voxelY": key[1],
                    "voxelZ": key[2],
                    "centerX": f"{center[0]:.6f}",
                    "centerY": f"{center[1]:.6f}",
                    "centerZ": f"{center[2]:.6f}",
                    "meanX": f"{mean[0]:.6f}",
                    "meanY": f"{mean[1]:.6f}",
                    "meanZ": f"{mean[2]:.6f}",
                    "normalX": f"{normal[0]:.6f}",
                    "normalY": f"{normal[1]:.6f}",
                    "normalZ": f"{normal[2]:.6f}",
                    "hit_count": feature.hit_count,
                    "frame_count": feature.frame_count,
                    "mean_distance": f"{feature.distance.mean:.6f}",
                    "min_distance": f"{feature.distance.min_value:.6f}",
                    "max_distance": f"{feature.distance.max_value:.6f}",
                    "dominant_distance_bin": feature.dominant_distance_bin(),
                    "mean_view_depth": f"{feature.view_depth.mean:.6f}",
                    "mean_view_angle": f"{feature.view_angle.mean:.6f}",
                    "max_view_angle": f"{feature.view_angle.max_value:.6f}",
                    "position_variance": f"{feature.position.variance_sum:.8f}",
                    "normal_variance": f"{feature.normal_variance():.8f}",
                    "depth_variance": f"{feature.view_depth.variance:.8f}",
                    "distance_variance": f"{feature.distance.variance:.8f}",
                    "mean_triangle_count": f"{feature.triangle_count.mean:.6f}",
                    "mean_face_normal_angle": f"{feature.face_normal_angle.mean:.6f}",
                    "boundary_risk_ratio": f"{feature.boundary_risk_count / hit_count:.6f}",
                    "crease_risk_ratio": f"{feature.crease_risk_count / hit_count:.6f}",
                    "any_risk_ratio": f"{feature.risk_count / hit_count:.6f}",
                    "stability_score": f"{feature.stability_score():.6f}",
                }
            )


def write_summary(path: Path, session: Path, voxels: dict[tuple[int, int, int], VoxelFeature], frame_count: int, voxel_size: float) -> None:
    values = list(voxels.values())
    hit_counts = np.asarray([v.hit_count for v in values], dtype=np.float64)
    frame_counts = np.asarray([v.frame_count for v in values], dtype=np.float64)
    risk_ratios = np.asarray([v.risk_count / max(1, v.hit_count) for v in values], dtype=np.float64)
    stability = np.asarray([v.stability_score() for v in values], dtype=np.float64)
    data = {
        "session": str(session),
        "framesUsed": frame_count,
        "voxelSizeMeters": voxel_size,
        "voxelCount": len(values),
        "hitCount": {
            "mean": float(hit_counts.mean()) if len(hit_counts) else 0.0,
            "max": int(hit_counts.max()) if len(hit_counts) else 0,
        },
        "frameCount": {
            "mean": float(frame_counts.mean()) if len(frame_counts) else 0.0,
            "max": int(frame_counts.max()) if len(frame_counts) else 0,
        },
        "riskRatioMean": float(risk_ratios.mean()) if len(risk_ratios) else 0.0,
        "stabilityScoreMean": float(stability.mean()) if len(stability) else 0.0,
        "stableVoxelsFrame3": int(np.sum(frame_counts >= 3)) if len(frame_counts) else 0,
        "stableVoxelsFrame5": int(np.sum(frame_counts >= 5)) if len(frame_counts) else 0,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_cloud(path: Path, points: np.ndarray, colors: np.ndarray, normals: np.ndarray | None = None) -> None:
    if o3d is None or len(points) == 0:
        return
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None and len(normals) == len(points):
        cloud.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False)


def write_ply_outputs(out_dir: Path, voxels: dict[tuple[int, int, int], VoxelFeature], stable_min_frames: int, risk_ratio_threshold: float) -> None:
    if o3d is None:
        return
    values = list(voxels.values())
    points = np.asarray([v.position.mean for v in values], dtype=np.float64)
    normals = np.asarray([v.mean_normal for v in values], dtype=np.float64)
    colors = []
    stable_mask = []
    risk_mask = []
    for v in values:
        risk_ratio = v.risk_count / max(1, v.hit_count)
        stable = v.frame_count >= stable_min_frames and risk_ratio < risk_ratio_threshold
        risky = risk_ratio >= risk_ratio_threshold
        stable_mask.append(stable)
        risk_mask.append(risky)
        if risky:
            colors.append((1.0, 0.08, 0.08))
        elif stable:
            colors.append((0.0, 0.9, 1.0))
        else:
            colors.append((1.0, 0.78, 0.0))
    colors_arr = np.asarray(colors, dtype=np.float64)
    stable_mask_arr = np.asarray(stable_mask, dtype=bool)
    risk_mask_arr = np.asarray(risk_mask, dtype=bool)
    write_cloud(out_dir / "observation_features_all.ply", points, colors_arr, normals)
    write_cloud(out_dir / "observation_features_stable_surface.ply", points[stable_mask_arr], np.tile((0.0, 0.9, 1.0), (int(np.sum(stable_mask_arr)), 1)), normals[stable_mask_arr])
    write_cloud(out_dir / "observation_features_risk_layer.ply", points[risk_mask_arr], np.tile((1.0, 0.08, 0.08), (int(np.sum(risk_mask_arr)), 1)), normals[risk_mask_arr])


def main() -> None:
    args = parse_args()
    session = args.session.resolve()
    rows = read_manifest(session)
    if args.max_frames > 0:
        rows = rows[: args.max_frames]
    out_dir = args.out or (session / "observation_features")
    out_dir.mkdir(parents=True, exist_ok=True)

    voxels = build_features(rows, args.voxel, args.crease_angle, args.max_frames)
    write_features_csv(out_dir / "point_observation_features.csv", voxels, args.voxel)
    write_summary(out_dir / "point_observation_features_summary.json", session, voxels, len(rows), args.voxel)
    if args.write_ply:
        write_ply_outputs(out_dir, voxels, args.stable_min_frames, args.risk_ratio_threshold)

    print(f"frames={len(rows)} voxels={len(voxels)} out={out_dir}")


if __name__ == "__main__":
    main()
