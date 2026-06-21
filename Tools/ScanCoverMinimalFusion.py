#!/usr/bin/env python3
"""Minimal ScanCover multi-frame fusion.

This script produces the four canonical diagnostic artifacts requested for the
current architecture check:

- stable_surface.ply
- risk_boundary.ply
- rejected_noise.ply
- fusion_report.json

The classifier is intentionally small and inspectable. It does not try to build
the final room mesh. It answers whether multi-frame fusion is moving in the
right direction by measuring repeatability, fused-surface thickness, local
support, and optional deviation from Meta Scene Mesh.
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
import open3d as o3d


DEFAULT_META_SESSION = Path(
    r"D:\PCA\Unity-MRMotifs-ScanCover-main\ScanCoverExports"
    r"\MetaSceneMeshAuditSessions\ScanCover_MetaSceneMeshAudit_20260611_180459_512"
)


@dataclass
class VoxelBucket:
    key: tuple[int, int, int]
    count: int = 0
    point_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    normal_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    frames: set[int] = field(default_factory=set)
    thickness_min: float = math.inf
    thickness_max: float = -math.inf
    center: np.ndarray | None = None
    normal: np.ndarray | None = None
    frame_count: int = 0
    normal_consistency: float = 0.0
    neighbor_count: int = 0
    meta_distance: float | None = None
    label: str = "unclassified"
    reasons: list[str] = field(default_factory=list)

    @property
    def thickness(self) -> float:
        if not math.isfinite(self.thickness_min) or not math.isfinite(self.thickness_max):
            return 0.0
        return max(0.0, self.thickness_max - self.thickness_min)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal ScanCover multi-frame fusion.")
    parser.add_argument("session", type=Path, help="ScanCover_MultiFrame_... session folder.")
    parser.add_argument("--out", type=Path, default=None, help="Output folder. Default: <session>/minimal_fusion")
    parser.add_argument("--voxel", type=float, default=0.02, help="World-space fusion voxel size in meters.")
    parser.add_argument("--stable-min-frames", type=int, default=3, help="Min distinct frames for stable candidates.")
    parser.add_argument("--risk-min-frames", type=int, default=2, help="Min distinct frames for risk layer candidates.")
    parser.add_argument("--stable-max-thickness", type=float, default=0.055, help="Max fused thickness for stable surface.")
    parser.add_argument("--risk-max-thickness", type=float, default=0.14, help="Max fused thickness before severe rejection.")
    parser.add_argument("--stable-min-normal-consistency", type=float, default=0.72)
    parser.add_argument("--risk-min-normal-consistency", type=float, default=0.35)
    parser.add_argument("--stable-min-neighbors", type=int, default=6, help="Min occupied 26-neighborhood voxels for stable.")
    parser.add_argument("--stable-meta-soft-max-distance", type=float, default=0.18)
    parser.add_argument("--growth-iterations", type=int, default=2, help="Promote thin near-stable risk voxels into stable surface.")
    parser.add_argument("--growth-min-stable-neighbors", type=int, default=3)
    parser.add_argument("--growth-max-thickness", type=float, default=0.045)
    parser.add_argument("--growth-min-normal-consistency", type=float, default=0.90)
    parser.add_argument("--meta-session", type=Path, default=DEFAULT_META_SESSION)
    parser.add_argument("--meta", type=Path, default=None, help="Explicit Meta Scene Mesh OBJ/PLY reference.")
    parser.add_argument("--meta-sample-points", type=int, default=250000)
    parser.add_argument("--stable-meta-max-distance", type=float, default=0.12)
    parser.add_argument("--risk-meta-max-distance", type=float, default=0.30)
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


def read_vertices_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.startswith("index,")), None)
    if header_index is None:
        raise RuntimeError(f"Could not find vertex header in {path}")

    points: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    for row in csv.DictReader(lines[header_index:]):
        points.append((float(row["worldX"]), float(row["worldY"]), float(row["worldZ"])))
        normals.append((float(row["normalWorldX"]), float(row["normalWorldY"]), float(row["normalWorldZ"])))
    return np.asarray(points, dtype=np.float64), normalize(np.asarray(normals, dtype=np.float64))


def normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.reshape((-1, 3))
    lengths = np.linalg.norm(vectors, axis=1)
    safe = np.where(lengths > 1e-8, lengths, 1.0)
    return vectors / safe[:, None]


def write_cloud(path: Path, points: np.ndarray, colors: np.ndarray | None = None, normals: np.ndarray | None = None) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.reshape((-1, 3)))
    if colors is not None and len(colors) == len(points):
        cloud.colors = o3d.utility.Vector3dVector(colors.reshape((-1, 3)))
    if normals is not None and len(normals) == len(points):
        cloud.normals = o3d.utility.Vector3dVector(normals.reshape((-1, 3)))
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False)


def auto_meta_path(args: argparse.Namespace) -> Path | None:
    if args.meta and args.meta.exists():
        return args.meta
    candidates: list[Path] = []
    if args.meta_session and args.meta_session.exists():
        preferred = args.meta_session / "stage0_weld" / "meta_scene_mesh_aligned_all_welded_eps1e-05.ply"
        if preferred.exists():
            return preferred
        candidates.extend(args.meta_session.rglob("meta_scene_mesh_aligned_all*.ply"))
        candidates.extend(args.meta_session.rglob("meta_scene_mesh_aligned_all*.obj"))
        candidates.extend(args.meta_session.rglob("*MeshVolume_world.obj"))
    return candidates[0] if candidates else None


def load_meta_points(path: Path | None, sample_count: int) -> tuple[np.ndarray | None, str | None]:
    if path is None or not path.exists():
        return None, None

    suffix = path.suffix.lower()
    if suffix == ".ply":
        cloud = o3d.io.read_point_cloud(str(path))
        if len(cloud.points) > 0:
            points = np.asarray(cloud.points, dtype=np.float64)
            return downsample_points(points, sample_count), str(path)

    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) == 0:
        return None, str(path)
    if len(mesh.triangles) > 0:
        try:
            cloud = mesh.sample_points_uniformly(number_of_points=sample_count)
            return np.asarray(cloud.points, dtype=np.float64), str(path)
        except Exception:
            pass
    return downsample_points(np.asarray(mesh.vertices, dtype=np.float64), sample_count), str(path)


def downsample_points(points: np.ndarray, max_count: int) -> np.ndarray:
    if len(points) <= max_count:
        return points
    step = max(1, int(math.ceil(len(points) / max_count)))
    return points[::step][:max_count]


def build_voxels(rows: list[dict[str, str]], voxel_size: float) -> tuple[list[VoxelBucket], list[dict[str, object]], int]:
    buckets: list[VoxelBucket] = []
    index_by_key: dict[tuple[int, int, int], int] = {}
    point_chunks: list[np.ndarray] = []
    normal_chunks: list[np.ndarray] = []
    bucket_id_chunks: list[np.ndarray] = []
    frame_stats: list[dict[str, object]] = []

    raw_count = 0
    for frame_id, row in enumerate(rows):
        points, normals = read_vertices_csv(Path(row["verticesCsv"]))
        ids = np.empty((len(points),), dtype=np.int32)
        keys = np.floor(points / voxel_size).astype(np.int64)

        for i, (point, normal, key_arr) in enumerate(zip(points, normals, keys)):
            key = (int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))
            bucket_id = index_by_key.get(key)
            if bucket_id is None:
                bucket_id = len(buckets)
                index_by_key[key] = bucket_id
                buckets.append(VoxelBucket(key=key))
            bucket = buckets[bucket_id]
            bucket.count += 1
            bucket.point_sum += point
            bucket.normal_sum += normal
            bucket.frames.add(frame_id)
            ids[i] = bucket_id

        point_chunks.append(points)
        normal_chunks.append(normals)
        bucket_id_chunks.append(ids)
        raw_count += len(points)
        frame_stats.append(
            {
                "frame": row.get("frame", f"frame_{frame_id:04d}"),
                "vertexCount": int(len(points)),
                "pose": [float(row["poseX"]), float(row["poseY"]), float(row["poseZ"])],
            }
        )

    for bucket in buckets:
        bucket.center = bucket.point_sum / max(1, bucket.count)
        n = bucket.normal_sum
        n_len = float(np.linalg.norm(n))
        bucket.normal = n / n_len if n_len > 1e-8 else np.array((0.0, 1.0, 0.0), dtype=np.float64)
        bucket.normal_consistency = n_len / max(1, bucket.count)
        bucket.frame_count = len(bucket.frames)

    all_points = np.vstack(point_chunks) if point_chunks else np.empty((0, 3), dtype=np.float64)
    all_bucket_ids = np.concatenate(bucket_id_chunks) if bucket_id_chunks else np.empty((0,), dtype=np.int32)
    for point, bucket_id in zip(all_points, all_bucket_ids):
        bucket = buckets[int(bucket_id)]
        assert bucket.center is not None and bucket.normal is not None
        proj = float(np.dot(point - bucket.center, bucket.normal))
        bucket.thickness_min = min(bucket.thickness_min, proj)
        bucket.thickness_max = max(bucket.thickness_max, proj)

    occupied = {bucket.key for bucket in buckets}
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
    for bucket in buckets:
        x, y, z = bucket.key
        bucket.neighbor_count = sum((x + dx, y + dy, z + dz) in occupied for dx, dy, dz in offsets)

    return buckets, frame_stats, raw_count


def attach_meta_distances(buckets: list[VoxelBucket], meta_points: np.ndarray | None) -> dict[str, object]:
    if meta_points is None or len(meta_points) == 0 or not buckets:
        return {"available": False}

    meta_cloud = o3d.geometry.PointCloud()
    meta_cloud.points = o3d.utility.Vector3dVector(meta_points)
    tree = o3d.geometry.KDTreeFlann(meta_cloud)
    distances: list[float] = []
    for bucket in buckets:
        assert bucket.center is not None
        _, _, d2 = tree.search_knn_vector_3d(bucket.center, 1)
        dist = math.sqrt(float(d2[0])) if d2 else math.inf
        bucket.meta_distance = dist
        if math.isfinite(dist):
            distances.append(dist)
    return distribution(distances)


def classify_buckets(buckets: list[VoxelBucket], args: argparse.Namespace) -> None:
    bucket_by_key = {bucket.key: bucket for bucket in buckets}
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]

    for bucket in buckets:
        reasons: list[str] = []
        if bucket.frame_count < args.risk_min_frames:
            reasons.append("single_or_low_frame_hit")
        if bucket.thickness > args.risk_max_thickness:
            reasons.append("severe_layer_thickness")
        if bucket.normal_consistency < args.risk_min_normal_consistency:
            reasons.append("severe_normal_disagreement")
        if bucket.meta_distance is not None and bucket.meta_distance > args.risk_meta_max_distance:
            reasons.append("far_from_meta_reference")

        if reasons:
            bucket.label = "rejected_noise"
            bucket.reasons = reasons
            continue

        stable_reasons: list[str] = []
        if bucket.frame_count < args.stable_min_frames:
            stable_reasons.append("not_enough_frames")
        if bucket.thickness > args.stable_max_thickness:
            stable_reasons.append("thick_or_cross_layer")
        if bucket.normal_consistency < args.stable_min_normal_consistency:
            stable_reasons.append("normal_unstable")
        if bucket.neighbor_count < args.stable_min_neighbors:
            stable_reasons.append("weak_local_support")
        if bucket.meta_distance is not None and bucket.meta_distance > args.stable_meta_max_distance:
            if bucket.meta_distance <= args.stable_meta_soft_max_distance:
                stable_reasons.append("meta_soft_bias")
            else:
                stable_reasons.append("meta_bias_or_offset")

        if stable_reasons:
            bucket.label = "risk_boundary"
            bucket.reasons = stable_reasons
        else:
            bucket.label = "stable_surface"
            bucket.reasons = ["stable_multi_frame_surface"]

    for _ in range(max(0, args.growth_iterations)):
        promoted: list[VoxelBucket] = []
        for bucket in buckets:
            if bucket.label != "risk_boundary":
                continue
            if bucket.frame_count < args.risk_min_frames:
                continue
            if bucket.thickness > args.growth_max_thickness:
                continue
            if bucket.normal_consistency < args.growth_min_normal_consistency:
                continue
            if bucket.meta_distance is not None and bucket.meta_distance > args.stable_meta_soft_max_distance:
                continue

            x, y, z = bucket.key
            stable_neighbors = 0
            aligned_neighbors = 0
            for dx, dy, dz in offsets:
                neighbor = bucket_by_key.get((x + dx, y + dy, z + dz))
                if neighbor is None or neighbor.normal is None or bucket.normal is None:
                    continue
                if float(np.dot(bucket.normal, neighbor.normal)) < math.cos(math.radians(25.0)):
                    continue
                aligned_neighbors += 1
                if neighbor.label == "stable_surface":
                    stable_neighbors += 1
            if stable_neighbors >= args.growth_min_stable_neighbors or (
                bucket.frame_count >= args.stable_min_frames and aligned_neighbors >= args.stable_min_neighbors
            ):
                promoted.append(bucket)

        if not promoted:
            break
        for bucket in promoted:
            bucket.label = "stable_surface"
            bucket.reasons = ["stable_by_neighbor_growth"]


def bucket_arrays(buckets: Iterable[VoxelBucket], label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [bucket for bucket in buckets if bucket.label == label]
    if not selected:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    points = np.vstack([bucket.center for bucket in selected if bucket.center is not None])
    normals = np.vstack([bucket.normal for bucket in selected if bucket.normal is not None])
    palette = {
        "stable_surface": np.array((0.10, 1.00, 0.85), dtype=np.float64),
        "risk_boundary": np.array((1.00, 0.82, 0.05), dtype=np.float64),
        "rejected_noise": np.array((1.00, 0.05, 0.05), dtype=np.float64),
    }
    colors = np.tile(palette[label], (len(points), 1))
    return points, colors, normals


def distribution(values: Iterable[float]) -> dict[str, object]:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def label_stats(buckets: list[VoxelBucket], label: str) -> dict[str, object]:
    selected = [bucket for bucket in buckets if bucket.label == label]
    reason_counts: dict[str, int] = {}
    for bucket in selected:
        for reason in bucket.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "voxelCount": int(len(selected)),
        "rawSampleCount": int(sum(bucket.count for bucket in selected)),
        "distinctFrameCount": distribution(bucket.frame_count for bucket in selected),
        "thicknessMeters": distribution(bucket.thickness for bucket in selected),
        "normalConsistency": distribution(bucket.normal_consistency for bucket in selected),
        "neighborCount": distribution(bucket.neighbor_count for bucket in selected),
        "metaDistanceMetersUnsigned": distribution(
            bucket.meta_distance for bucket in selected if bucket.meta_distance is not None
        ),
        "reasonCounts": reason_counts,
    }


def coverage_stats(buckets: list[VoxelBucket]) -> dict[str, object]:
    if not buckets:
        return {}
    stable = sum(1 for b in buckets if b.label == "stable_surface")
    risk = sum(1 for b in buckets if b.label == "risk_boundary")
    rejected = sum(1 for b in buckets if b.label == "rejected_noise")
    keys = np.asarray([b.key for b in buckets], dtype=np.int64)
    mins = keys.min(axis=0)
    maxs = keys.max(axis=0)
    volume_voxels = int(np.prod(maxs - mins + 1))
    occupied = len(buckets)
    return {
        "occupiedVoxelCount": int(occupied),
        "stableVoxelRatio": stable / max(1, occupied),
        "riskVoxelRatio": risk / max(1, occupied),
        "rejectedVoxelRatio": rejected / max(1, occupied),
        "axisAlignedVoxelBox": {
            "minKey": mins.astype(int).tolist(),
            "maxKey": maxs.astype(int).tolist(),
            "boxVoxelCount": volume_voxels,
            "occupancyRatioInBox": occupied / max(1, volume_voxels),
        },
    }


def route_judgement(report: dict[str, object]) -> dict[str, object]:
    coverage = report["coverage"]
    stable_ratio = float(coverage["stableVoxelRatio"])
    risk_ratio = float(coverage["riskVoxelRatio"])
    rejected_ratio = float(coverage["rejectedVoxelRatio"])
    stable_thickness = report["labels"]["stable_surface"]["thicknessMeters"]
    stable_p90 = float(stable_thickness.get("p90", 999.0)) if stable_thickness.get("count", 0) else 999.0
    meta = report["labels"]["stable_surface"]["metaDistanceMetersUnsigned"]
    stable_meta_p90 = float(meta.get("p90", 999.0)) if meta.get("count", 0) else None

    pass_basic = stable_ratio >= 0.25 and rejected_ratio <= 0.45 and stable_p90 <= 0.08
    pass_meta = stable_meta_p90 is None or stable_meta_p90 <= 0.18
    status = "pass" if pass_basic and pass_meta else "needs_more_work"
    notes: list[str] = []
    if stable_ratio < 0.25:
        notes.append("stable surface ratio is low; fusion is still dominated by uncertain or rejected samples")
    if rejected_ratio > 0.45:
        notes.append("rejected noise ratio is high; multi-frame alignment or raw BL quality is still weak")
    if stable_p90 > 0.08:
        notes.append("stable fused thickness is too large; cross-layer/ghosting remains visible")
    if stable_meta_p90 is not None and stable_meta_p90 > 0.18:
        notes.append("stable surface is far from Meta reference; reference bias or coordinate mismatch needs review")
    if risk_ratio > stable_ratio:
        notes.append("risk layer is larger than stable layer; the route is promising but fusion rules are not final")
    if not notes:
        notes.append("stable layer is large enough and thickness is controlled; route is worth continuing")

    return {
        "status": status,
        "stableVoxelRatio": stable_ratio,
        "riskVoxelRatio": risk_ratio,
        "rejectedVoxelRatio": rejected_ratio,
        "stableThicknessP90Meters": stable_p90,
        "stableMetaDistanceP90MetersUnsigned": stable_meta_p90,
        "notes": notes,
    }


def main() -> int:
    args = parse_args()
    session = args.session.resolve()
    if not session.exists():
        raise FileNotFoundError(session)
    out_dir = args.out.resolve() if args.out else session / "minimal_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(session)
    buckets, frame_stats, raw_count = build_voxels(rows, args.voxel)

    meta_path = auto_meta_path(args)
    meta_points, meta_used = load_meta_points(meta_path, args.meta_sample_points)
    meta_distribution = attach_meta_distances(buckets, meta_points)
    classify_buckets(buckets, args)

    for label, filename in (
        ("stable_surface", "stable_surface.ply"),
        ("risk_boundary", "risk_boundary.ply"),
        ("rejected_noise", "rejected_noise.ply"),
    ):
        points, colors, normals = bucket_arrays(buckets, label)
        write_cloud(out_dir / filename, points, colors, normals)

    report: dict[str, object] = {
        "session": str(session),
        "outputDirectory": str(out_dir),
        "frameCount": int(len(rows)),
        "rawPointCount": int(raw_count),
        "voxelSizeMeters": args.voxel,
        "thresholds": {
            "stableMinFrames": args.stable_min_frames,
            "riskMinFrames": args.risk_min_frames,
            "stableMaxThicknessMeters": args.stable_max_thickness,
            "riskMaxThicknessMeters": args.risk_max_thickness,
            "stableMinNormalConsistency": args.stable_min_normal_consistency,
            "riskMinNormalConsistency": args.risk_min_normal_consistency,
            "stableMinNeighbors": args.stable_min_neighbors,
            "stableMetaMaxDistanceMeters": args.stable_meta_max_distance,
            "riskMetaMaxDistanceMeters": args.risk_meta_max_distance,
        },
        "metaReference": {
            "path": meta_used,
            "samplePointCount": int(len(meta_points)) if meta_points is not None else 0,
            "distanceModel": "unsigned nearest-neighbor distance to sampled Meta Scene Mesh",
            "allVoxelDistanceMetersUnsigned": meta_distribution,
        },
        "coverage": coverage_stats(buckets),
        "labels": {
            "stable_surface": label_stats(buckets, "stable_surface"),
            "risk_boundary": label_stats(buckets, "risk_boundary"),
            "rejected_noise": label_stats(buckets, "rejected_noise"),
        },
        "frameStats": frame_stats,
    }
    report["judgement"] = route_judgement(report)

    (out_dir / "fusion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["judgement"], indent=2))
    print(f"wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
