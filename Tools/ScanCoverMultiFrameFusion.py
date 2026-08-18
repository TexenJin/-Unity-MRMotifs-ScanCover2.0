#!/usr/bin/env python3
"""Fuse and inspect a ScanCover multi-frame BL surface capture session.

The script intentionally stops at diagnostic point-cloud outputs. It answers:
- did all frame meshes land in the same world coordinate system?
- which surface samples are repeatedly observed across frames?
- where are single-hit/noisy/risk samples?
- what does an offline Open3D plane-teacher see after multi-frame fusion?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path, help="ScanCover_MultiFrame_... session folder.")
    parser.add_argument("--voxel", type=float, default=0.02, help="World-space merge voxel size in meters.")
    parser.add_argument("--stable-min-frames", type=int, default=3, help="Min distinct frames for stable voxels.")
    parser.add_argument("--plane-distance", type=float, default=0.04, help="Plane RANSAC threshold in meters.")
    parser.add_argument("--plane-min-inliers", type=int, default=900)
    parser.add_argument("--plane-iterations", type=int, default=1200)
    parser.add_argument("--max-planes", type=int, default=12)
    parser.add_argument("--merge-normal-deg", type=float, default=7.5, help="Max normal angle for plane-family merge.")
    parser.add_argument("--merge-distance", type=float, default=0.18, help="Max plane d difference for plane-family merge.")
    parser.add_argument("--reclassify-distance", type=float, default=0.07, help="Max point-to-family-plane distance.")
    parser.add_argument("--reclassify-normal-deg", type=float, default=45.0, help="Max point normal angle to family normal.")
    parser.add_argument("--skip-planes", action="store_true", help="Only write fused point clouds.")
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
    return np.asarray(points, dtype=np.float64), np.asarray(normals, dtype=np.float64)


def frame_color(index: int, count: int) -> tuple[float, float, float]:
    if count <= 1:
        return (1.0, 1.0, 1.0)
    t = index / max(1, count - 1)
    return (0.15 + 0.85 * t, 0.85 * (1.0 - abs(t - 0.5) * 2.0), 1.0 - 0.85 * t)


def write_cloud(path: Path, points: np.ndarray, colors: np.ndarray | None = None, normals: np.ndarray | None = None) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) == len(points):
        cloud.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None and len(normals) == len(points):
        cloud.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1)
    safe = np.where(lengths > 1e-8, lengths, 1.0)
    return vectors / safe[:, None]


def build_voxels(
    points: np.ndarray,
    normals: np.ndarray,
    frame_ids: np.ndarray,
    voxel_size: float,
) -> dict[tuple[int, int, int], dict[str, object]]:
    buckets: dict[tuple[int, int, int], dict[str, object]] = {}
    keys = np.floor(points / voxel_size).astype(np.int64)
    for point, normal, frame_id, key_arr in zip(points, normals, frame_ids, keys):
        key = (int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))
        bucket = buckets.get(key)
        if bucket is None:
            bucket = {
                "count": 0,
                "sum": np.zeros(3, dtype=np.float64),
                "normal_sum": np.zeros(3, dtype=np.float64),
                "frames": set(),
            }
            buckets[key] = bucket
        bucket["count"] = int(bucket["count"]) + 1
        bucket["sum"] = bucket["sum"] + point
        bucket["normal_sum"] = bucket["normal_sum"] + normal
        bucket["frames"].add(int(frame_id))
    return buckets


def voxel_arrays(
    buckets: dict[tuple[int, int, int], dict[str, object]],
    stable_min_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    colors: list[tuple[float, float, float]] = []
    distinct_frames: list[int] = []

    max_frames = max((len(bucket["frames"]) for bucket in buckets.values()), default=1)
    for bucket in buckets.values():
        count = int(bucket["count"])
        frame_count = len(bucket["frames"])
        point = bucket["sum"] / max(1, count)
        normal = bucket["normal_sum"]
        norm = np.linalg.norm(normal)
        if norm > 1e-8:
            normal = normal / norm
        else:
            normal = np.array((0.0, 1.0, 0.0))

        points.append(point)
        normals.append(normal)
        distinct_frames.append(frame_count)

        if frame_count >= stable_min_frames:
            strength = min(1.0, frame_count / max(1, max_frames))
            colors.append((0.1, 0.45 + 0.5 * strength, 1.0 - 0.35 * strength))
        elif frame_count == 1:
            colors.append((1.0, 0.08, 0.08))
        else:
            colors.append((1.0, 0.8, 0.1))

    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(normals, dtype=np.float64),
        np.asarray(colors, dtype=np.float64),
        np.asarray(distinct_frames, dtype=np.int32),
    )


def write_voxel_stats(path: Path, frame_counts: np.ndarray) -> None:
    total = int(len(frame_counts))
    rows = []
    for threshold in [1, 2, 3, 5, 8, 12, 20]:
        count = int(np.sum(frame_counts >= threshold))
        rows.append(
            {
                "minDistinctFrames": threshold,
                "voxelCount": count,
                "ratio": f"{(count / total if total else 0):.6f}",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["minDistinctFrames", "voxelCount", "ratio"])
        writer.writeheader()
        writer.writerows(rows)


def normal_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    dot = abs(float(np.dot(a, b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def canonical_plane(normal: np.ndarray, d: float) -> tuple[np.ndarray, float]:
    normal = np.asarray(normal, dtype=np.float64)
    length = np.linalg.norm(normal)
    if length > 1e-8:
        normal = normal / length
        d = d / length
    # Keep signs stable so d comparisons do not split opposite-facing copies.
    axis = int(np.argmax(np.abs(normal)))
    if normal[axis] < 0.0:
        normal = -normal
        d = -d
    return normal, float(d)


def weighted_family_plane(planes: list[dict[str, object]]) -> tuple[np.ndarray, float]:
    base = np.asarray(planes[0]["normal"], dtype=np.float64)
    normal_sum = np.zeros(3, dtype=np.float64)
    d_sum = 0.0
    weight_sum = 0.0
    for plane in planes:
        normal = np.asarray(plane["normal"], dtype=np.float64)
        d = float(plane["d"])
        if float(np.dot(base, normal)) < 0.0:
            normal = -normal
            d = -d
        weight = float(plane["inliers"])
        normal_sum += normal * weight
        d_sum += d * weight
        weight_sum += weight
    normal = normal_sum
    normal_length = np.linalg.norm(normal)
    if normal_length > 1e-8:
        normal = normal / normal_length
    d = d_sum / max(1.0, weight_sum)
    axis = int(np.argmax(np.abs(normal)))
    if normal[axis] < 0.0:
        normal = -normal
        d = -d
    return normal, float(d)


def merge_plane_families(
    planes: list[dict[str, object]],
    normal_deg: float,
    distance: float,
) -> list[list[dict[str, object]]]:
    families: list[list[dict[str, object]]] = []
    for plane in sorted(planes, key=lambda item: int(item["inliers"]), reverse=True):
        placed = False
        for family in families:
            family_normal, family_d = weighted_family_plane(family)
            angle = normal_angle_degrees(np.asarray(plane["normal"], dtype=np.float64), family_normal)
            d_delta = abs(float(plane["d"]) - family_d)
            if angle <= normal_deg and d_delta <= distance:
                family.append(plane)
                placed = True
                break
        if not placed:
            families.append([plane])
    families.sort(key=lambda group: -sum(int(plane["inliers"]) for plane in group))
    return families


def write_reclassified_outputs(
    out_dir: Path,
    source_points: np.ndarray,
    source_normals: np.ndarray,
    families: list[list[dict[str, object]]],
    distance: float,
    normal_deg: float,
) -> None:
    for path in out_dir.glob("reclassified_family_*.ply"):
        path.unlink()
    for path in out_dir.glob("reclassified_outliers*.ply"):
        path.unlink()

    palette = np.asarray(
        [
            (1.0, 0.05, 0.05),
            (0.0, 0.75, 1.0),
            (0.05, 1.0, 0.18),
            (1.0, 0.85, 0.0),
            (0.9, 0.15, 1.0),
            (1.0, 0.5, 0.0),
            (0.1, 0.25, 1.0),
            (0.7, 1.0, 0.0),
        ],
        dtype=np.float64,
    )
    if not families:
        write_cloud(out_dir / "reclassified_outliers_all.ply", source_points, np.tile((0.45, 0.45, 0.45), (len(source_points), 1)), source_normals)
        return

    family_planes = [weighted_family_plane(family) for family in families]
    family_normals = np.vstack([normal for normal, _ in family_planes])
    family_ds = np.asarray([d for _, d in family_planes], dtype=np.float64)
    normal_cos = math.cos(math.radians(normal_deg))

    assignments: list[list[int]] = [[] for _ in families]
    outliers: list[int] = []
    for point_index, (point, normal) in enumerate(zip(source_points, source_normals)):
        distances = np.abs(family_normals @ point + family_ds)
        order = np.argsort(distances)
        assigned = False
        for family_index in order:
            if distances[family_index] > distance:
                break
            if abs(float(np.dot(normal, family_normals[family_index]))) < normal_cos:
                continue
            assignments[int(family_index)].append(point_index)
            assigned = True
            break
        if not assigned:
            outliers.append(point_index)

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    summary_rows: list[dict[str, object]] = []
    for family_index, indices in enumerate(assignments):
        if not indices:
            continue
        idx = np.asarray(indices, dtype=np.int64)
        color = palette[family_index % len(palette)]
        plane_names = "-".join(str(plane["index"]) for plane in families[family_index])
        family_points = source_points[idx]
        family_colors = np.tile(color, (len(idx), 1))
        write_cloud(
            out_dir / f"reclassified_family_{family_index:02d}_{len(idx)}pts_planes_{plane_names}.ply",
            family_points,
            family_colors,
            source_normals[idx],
        )
        all_points.append(family_points)
        all_colors.append(family_colors)
        normal, d = family_planes[family_index]
        summary_rows.append(
            {
                "family": family_index,
                "planes": plane_names,
                "points": len(idx),
                "normalX": f"{normal[0]:.8f}",
                "normalY": f"{normal[1]:.8f}",
                "normalZ": f"{normal[2]:.8f}",
                "d": f"{d:.8f}",
            }
        )

    if outliers:
        idx = np.asarray(outliers, dtype=np.int64)
        outlier_points = source_points[idx]
        outlier_colors = np.tile(np.asarray((0.45, 0.45, 0.45), dtype=np.float64), (len(idx), 1))
        write_cloud(out_dir / f"reclassified_outliers_{len(idx)}pts.ply", outlier_points, outlier_colors, source_normals[idx])
        all_points.append(outlier_points)
        all_colors.append(outlier_colors)

    if all_points:
        write_cloud(out_dir / "reclassified_all.ply", np.vstack(all_points), np.vstack(all_colors))

    with (out_dir / "reclassified_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["family", "planes", "points", "normalX", "normalY", "normalZ", "d"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
        writer.writerow(
            {
                "family": "outliers",
                "planes": "",
                "points": len(outliers),
                "normalX": "",
                "normalY": "",
                "normalZ": "",
                "d": "",
            }
        )


def run_plane_teacher(
    out_dir: Path,
    points: np.ndarray,
    normals: np.ndarray,
    distance: float,
    min_inliers: int,
    iterations: int,
    max_planes: int,
    merge_normal_deg: float,
    merge_distance: float,
    reclassify_distance: float,
    reclassify_normal_deg: float,
) -> None:
    if len(points) < min_inliers:
        return

    remaining = o3d.geometry.PointCloud()
    remaining.points = o3d.utility.Vector3dVector(points)
    remaining.normals = o3d.utility.Vector3dVector(normals)

    colors = [
        (1.0, 0.1, 0.1),
        (0.0, 0.85, 1.0),
        (0.1, 1.0, 0.25),
        (1.0, 0.85, 0.0),
        (0.9, 0.25, 1.0),
        (1.0, 0.55, 0.0),
        (0.1, 0.25, 1.0),
        (0.8, 1.0, 0.0),
    ]
    summary: list[dict[str, object]] = []
    plane_infos: list[dict[str, object]] = []
    combined_points: list[np.ndarray] = []
    combined_colors: list[np.ndarray] = []

    for plane_index in range(max_planes):
        if len(remaining.points) < min_inliers:
            break
        model, inliers = remaining.segment_plane(
            distance_threshold=distance,
            ransac_n=3,
            num_iterations=iterations,
        )
        if len(inliers) < min_inliers:
            break

        plane_cloud = remaining.select_by_index(inliers)
        rest = remaining.select_by_index(inliers, invert=True)
        color = colors[plane_index % len(colors)]
        plane_cloud.paint_uniform_color(color)
        o3d.io.write_point_cloud(str(out_dir / f"teacher_plane_{plane_index:02d}.ply"), plane_cloud)

        arr = np.asarray(plane_cloud.points)
        combined_points.append(arr)
        combined_colors.append(np.tile(np.asarray(color, dtype=np.float64), (len(arr), 1)))

        a, b, c, d = model
        normal, d = canonical_plane(np.asarray((a, b, c), dtype=np.float64), float(d))
        a, b, c = normal
        plane_infos.append(
            {
                "index": plane_index,
                "inliers": len(inliers),
                "normal": normal,
                "d": d,
            }
        )
        summary.append(
            {
                "plane": plane_index,
                "inliers": len(inliers),
                "a": f"{a:.8f}",
                "b": f"{b:.8f}",
                "c": f"{c:.8f}",
                "d": f"{d:.8f}",
            }
        )
        remaining = rest

    if len(remaining.points):
        remaining.paint_uniform_color((0.45, 0.45, 0.45))
        o3d.io.write_point_cloud(str(out_dir / "teacher_outliers.ply"), remaining)
        arr = np.asarray(remaining.points)
        combined_points.append(arr)
        combined_colors.append(np.tile(np.asarray((0.45, 0.45, 0.45), dtype=np.float64), (len(arr), 1)))

    if combined_points:
        write_cloud(
            out_dir / "teacher_all_planes_and_outliers.ply",
            np.vstack(combined_points),
            np.vstack(combined_colors),
        )

    with (out_dir / "teacher_plane_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["plane", "inliers", "a", "b", "c", "d"])
        writer.writeheader()
        writer.writerows(summary)

    families = merge_plane_families(plane_infos, merge_normal_deg, merge_distance)
    with (out_dir / "teacher_family_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["family", "planes", "planeCount", "inliers", "normalX", "normalY", "normalZ", "d"])
        writer.writeheader()
        for family_index, family in enumerate(families):
            normal, d = weighted_family_plane(family)
            writer.writerow(
                {
                    "family": family_index,
                    "planes": "-".join(str(plane["index"]) for plane in family),
                    "planeCount": len(family),
                    "inliers": sum(int(plane["inliers"]) for plane in family),
                    "normalX": f"{normal[0]:.8f}",
                    "normalY": f"{normal[1]:.8f}",
                    "normalZ": f"{normal[2]:.8f}",
                    "d": f"{d:.8f}",
                }
            )
    write_reclassified_outputs(
        out_dir,
        points,
        normals,
        families,
        reclassify_distance,
        reclassify_normal_deg,
    )


def main() -> int:
    args = parse_args()
    session = args.session.resolve()
    if not session.exists():
        raise FileNotFoundError(session)

    out_dir = session / "multi_frame_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(session)
    all_points: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_frame_ids: list[np.ndarray] = []
    frame_stats: list[dict[str, object]] = []

    for frame_id, row in enumerate(rows):
        frame = row["frame"]
        vertices_csv = Path(row["verticesCsv"])
        points, normals = read_vertices_csv(vertices_csv)
        normals = normalize_vectors(normals)
        all_points.append(points)
        all_normals.append(normals)
        all_frame_ids.append(np.full((len(points),), frame_id, dtype=np.int32))
        color = frame_color(frame_id, len(rows))
        all_colors.append(np.tile(np.asarray(color, dtype=np.float64), (len(points), 1)))
        frame_stats.append(
            {
                "frame": frame,
                "vertexCount": len(points),
                "poseX": row["poseX"],
                "poseY": row["poseY"],
                "poseZ": row["poseZ"],
                "depthGridFrame": row["depthGridFrame"],
            }
        )

    points = np.vstack(all_points)
    normals = np.vstack(all_normals)
    colors = np.vstack(all_colors)
    frame_ids = np.concatenate(all_frame_ids)

    write_cloud(out_dir / "multi_frame_raw_all.ply", points, colors, normals)

    buckets = build_voxels(points, normals, frame_ids, args.voxel)
    voxel_points, voxel_normals, voxel_colors, distinct_frame_counts = voxel_arrays(buckets, args.stable_min_frames)
    write_cloud(out_dir / f"multi_frame_voxel_{args.voxel:.3f}_all.ply", voxel_points, voxel_colors, voxel_normals)

    stable_mask = distinct_frame_counts >= args.stable_min_frames
    risk_mask = distinct_frame_counts == 1
    mid_mask = ~(stable_mask | risk_mask)
    write_cloud(
        out_dir / f"multi_frame_voxel_{args.voxel:.3f}_stable_min{args.stable_min_frames}.ply",
        voxel_points[stable_mask],
        voxel_colors[stable_mask],
        voxel_normals[stable_mask],
    )
    write_cloud(
        out_dir / f"multi_frame_voxel_{args.voxel:.3f}_risk_single_frame.ply",
        voxel_points[risk_mask],
        voxel_colors[risk_mask],
        voxel_normals[risk_mask],
    )
    write_cloud(
        out_dir / f"multi_frame_voxel_{args.voxel:.3f}_mid_2_to_{args.stable_min_frames - 1}.ply",
        voxel_points[mid_mask],
        voxel_colors[mid_mask],
        voxel_normals[mid_mask],
    )

    with (out_dir / "frame_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "vertexCount", "poseX", "poseY", "poseZ", "depthGridFrame"])
        writer.writeheader()
        writer.writerows(frame_stats)
    write_voxel_stats(out_dir / "voxel_observation_stats.csv", distinct_frame_counts)

    if not args.skip_planes:
        run_plane_teacher(
            out_dir,
            voxel_points[stable_mask],
            voxel_normals[stable_mask],
            args.plane_distance,
            args.plane_min_inliers,
            args.plane_iterations,
            args.max_planes,
            args.merge_normal_deg,
            args.merge_distance,
            args.reclassify_distance,
            args.reclassify_normal_deg,
        )

    summary = {
        "session": str(session),
        "frameCount": len(rows),
        "rawPointCount": int(len(points)),
        "voxelSizeMeters": args.voxel,
        "voxelCount": int(len(voxel_points)),
        "stableMinDistinctFrames": args.stable_min_frames,
        "stableVoxelCount": int(np.sum(stable_mask)),
        "singleFrameRiskVoxelCount": int(np.sum(risk_mask)),
        "outputDirectory": str(out_dir),
    }
    (out_dir / "fusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
