#!/usr/bin/env python3
"""Analyze Open3D RANSAC plane fragments and suggest merge candidates.

Usage:
  py -3.11 analyze_plane_merge.py C:/path/to/xxx_open3d_planes
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay


@dataclass
class PlaneInfo:
    index: int
    inliers: int
    normal: np.ndarray
    d: float
    path: Path
    count: int
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    bbox_size: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("plane_dir", type=Path, help="Directory containing plane_summary.csv and plane_*.ply")
    parser.add_argument("--normal-deg", type=float, default=5.0, help="Normal angle threshold for merge candidates")
    parser.add_argument("--plane-distance", type=float, default=0.10, help="Plane equation d threshold in meters")
    parser.add_argument("--bbox-gap", type=float, default=0.15, help="Allowed AABB gap in meters")
    parser.add_argument("--sample-points", type=int, default=2500, help="Max points sampled per plane for nearest distance")
    parser.add_argument("--nearest-distance", type=float, default=0.18, help="Nearest sampled point threshold in meters")
    parser.add_argument("--reclassify-distance", type=float, default=0.05, help="Max point-to-family-plane distance in meters")
    parser.add_argument("--reclassify-normal-deg", type=float, default=35.0, help="Max point normal angle to family normal")
    parser.add_argument("--reclassify-samples", type=int, default=60000, help="Source mesh samples for reclassification")
    parser.add_argument("--source", type=Path, default=None, help="Optional source OBJ/PLY used for reclassification")
    parser.add_argument("--no-reclassify-normal", action="store_true", help="Disable normal filtering during reclassification")
    parser.add_argument("--surface-cell", type=float, default=0.08, help="Planar family surface grid cell size in meters")
    parser.add_argument("--surface-min-points", type=int, default=2, help="Minimum reclassified points per cell")
    parser.add_argument("--surface-offset", type=float, default=0.004, help="Offset generated surfaces along plane normal")
    parser.add_argument("--alpha-boundary", type=float, default=0.18, help="Alpha shape max circumradius in projected meters")
    parser.add_argument("--alpha-max-points", type=int, default=9000, help="Max projected points per family for alpha boundary")
    parser.add_argument(
        "--no-absorb-small-families",
        action="store_true",
        help="Disable absorbing nearby small plane families into larger main-plane families",
    )
    parser.add_argument("--absorb-max-ratio", type=float, default=0.28, help="Families below this ratio of largest family points can be absorbed")
    parser.add_argument("--absorb-max-points", type=int, default=4500, help="Families below this point count can be absorbed")
    parser.add_argument("--absorb-normal-deg", type=float, default=16.0, help="Normal angle threshold for small-family absorption")
    parser.add_argument("--absorb-plane-distance", type=float, default=0.18, help="Centroid-to-target-plane distance threshold for absorption")
    parser.add_argument("--absorb-nearest-distance", type=float, default=0.24, help="Nearest sampled point threshold for absorption")
    parser.add_argument("--absorb-bbox-gap", type=float, default=0.20, help="AABB gap threshold for absorption")
    parser.add_argument("--absorb-sample-points", type=int, default=1800, help="Max points sampled per family for absorption proximity checks")
    return parser.parse_args()


def load_summary(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v)
    if length <= 1e-8:
        return v
    return v / length


def read_plane(row: dict[str, str], plane_dir: Path) -> PlaneInfo:
    rel_path = Path(row["path"])
    ply_path = rel_path if rel_path.is_absolute() else plane_dir / rel_path
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise RuntimeError(f"Empty point cloud: {ply_path}")

    bbox = pcd.get_axis_aligned_bounding_box()
    normal = normalize(np.array([float(row["a"]), float(row["b"]), float(row["c"])], dtype=np.float64))
    return PlaneInfo(
        index=int(row["plane"]),
        inliers=int(row["inliers"]),
        normal=normal,
        d=float(row["d"]),
        path=ply_path,
        count=len(points),
        centroid=np.asarray(pcd.get_center()),
        bbox_min=np.asarray(bbox.min_bound),
        bbox_max=np.asarray(bbox.max_bound),
        bbox_size=np.asarray(bbox.get_extent()),
    )


def normal_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    dot = abs(float(np.dot(a, b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def aabb_gap(a: PlaneInfo, b: PlaneInfo) -> tuple[float, int]:
    gaps = np.maximum(0.0, np.maximum(a.bbox_min - b.bbox_max, b.bbox_min - a.bbox_max))
    overlap_axes = int(np.count_nonzero(gaps <= 1e-6))
    return float(np.linalg.norm(gaps)), overlap_axes


def sampled_nearest_distance(a_path: Path, b_path: Path, max_points: int) -> float:
    a = np.asarray(o3d.io.read_point_cloud(str(a_path)).points)
    b = np.asarray(o3d.io.read_point_cloud(str(b_path)).points)
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    if len(a) > max_points:
        a = a[np.linspace(0, len(a) - 1, max_points, dtype=np.int64)]
    if len(b) > max_points:
        b = b[np.linspace(0, len(b) - 1, max_points, dtype=np.int64)]

    # Chunked brute force is fine for this diagnostic size and avoids scipy dependency.
    best = float("inf")
    for start in range(0, len(a), 256):
        chunk = a[start : start + 256]
        distances = np.linalg.norm(chunk[:, None, :] - b[None, :, :], axis=2)
        best = min(best, float(np.min(distances)))
    return best


def write_plane_stats(plane_dir: Path, planes: list[PlaneInfo]) -> Path:
    out_path = plane_dir / "plane_stats.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "plane",
                "points",
                "inliers",
                "normal_x",
                "normal_y",
                "normal_z",
                "d",
                "centroid_x",
                "centroid_y",
                "centroid_z",
                "bbox_min_x",
                "bbox_min_y",
                "bbox_min_z",
                "bbox_max_x",
                "bbox_max_y",
                "bbox_max_z",
                "bbox_size_x",
                "bbox_size_y",
                "bbox_size_z",
                "path",
            ]
        )
        for p in planes:
            writer.writerow(
                [
                    p.index,
                    p.count,
                    p.inliers,
                    *p.normal,
                    p.d,
                    *p.centroid,
                    *p.bbox_min,
                    *p.bbox_max,
                    *p.bbox_size,
                    p.path.name,
                ]
            )
    return out_path


def write_merge_candidates(plane_dir: Path, planes: list[PlaneInfo], args: argparse.Namespace) -> Path:
    out_path = plane_dir / "plane_merge_candidates.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "plane_a",
                "plane_b",
                "candidate",
                "normal_angle_deg",
                "abs_d_delta_m",
                "aabb_gap_m",
                "aabb_overlap_axes",
                "nearest_sampled_point_m",
                "reason",
            ]
        )
        for i, a in enumerate(planes):
            for b in planes[i + 1 :]:
                angle = normal_angle_deg(a.normal, b.normal)
                d_delta = abs(a.d - b.d)
                gap, overlap_axes = aabb_gap(a, b)
                close_by_bbox = overlap_axes >= 2 or gap <= args.bbox_gap
                nearest = sampled_nearest_distance(a.path, b.path, args.sample_points) if close_by_bbox else float("inf")
                close_by_points = nearest <= args.nearest_distance
                candidate = angle <= args.normal_deg and d_delta <= args.plane_distance and (close_by_bbox or close_by_points)
                reasons = []
                if angle <= args.normal_deg:
                    reasons.append("normal")
                if d_delta <= args.plane_distance:
                    reasons.append("d")
                if close_by_bbox:
                    reasons.append("bbox")
                if close_by_points:
                    reasons.append("nearest")
                writer.writerow(
                    [
                        a.index,
                        b.index,
                        int(candidate),
                        f"{angle:.4f}",
                        f"{d_delta:.4f}",
                        f"{gap:.4f}",
                        overlap_axes,
                        "inf" if math.isinf(nearest) else f"{nearest:.4f}",
                        "+".join(reasons),
                    ]
                )
    return out_path


def build_merge_families(candidate_path: Path, planes: list[PlaneInfo]) -> list[list[PlaneInfo]]:
    parent = {p.index: p.index for p in planes}
    by_index = {p.index: p for p in planes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    with candidate_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["candidate"] == "1":
                union(int(row["plane_a"]), int(row["plane_b"]))

    groups: dict[int, list[PlaneInfo]] = {}
    for plane in planes:
        groups.setdefault(find(plane.index), []).append(by_index[plane.index])

    families = list(groups.values())
    families.sort(key=lambda group: (-sum(p.count for p in group), min(p.index for p in group)))
    return families


def weighted_average_plane(family: list[PlaneInfo]) -> tuple[np.ndarray, float]:
    base = family[0].normal
    normal_acc = np.zeros(3, dtype=np.float64)
    d_acc = 0.0
    weight_acc = 0.0

    for plane in family:
        weight = float(max(plane.count, plane.inliers, 1))
        sign = 1.0 if float(np.dot(base, plane.normal)) >= 0.0 else -1.0
        normal_acc += plane.normal * sign * weight
        d_acc += plane.d * sign * weight
        weight_acc += weight

    normal = normalize(normal_acc)
    d = d_acc / max(weight_acc, 1.0)
    return normal, d


def family_total_points(family: list[PlaneInfo]) -> int:
    return sum(p.count for p in family)


def family_centroid(family: list[PlaneInfo]) -> np.ndarray:
    total_points = family_total_points(family)
    return sum((p.centroid * p.count for p in family), np.zeros(3, dtype=np.float64)) / max(total_points, 1)


def family_bbox(family: list[PlaneInfo]) -> tuple[np.ndarray, np.ndarray]:
    bbox_min = np.min(np.vstack([p.bbox_min for p in family]), axis=0)
    bbox_max = np.max(np.vstack([p.bbox_max for p in family]), axis=0)
    return bbox_min, bbox_max


def bbox_gap_arrays(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    gaps = np.maximum(0.0, np.maximum(a_min - b_max, b_min - a_max))
    return float(np.linalg.norm(gaps))


def family_sample_points(family: list[PlaneInfo], max_points: int) -> np.ndarray:
    clouds = []
    for plane in family:
        points = np.asarray(o3d.io.read_point_cloud(str(plane.path)).points)
        if len(points) > 0:
            clouds.append(points)
    if not clouds:
        return np.empty((0, 3), dtype=np.float64)
    points = np.vstack(clouds)
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points, dtype=np.int64)]
    return points


def nearest_distance_points(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    best = float("inf")
    for start in range(0, len(a), 256):
        chunk = a[start : start + 256]
        distances = np.linalg.norm(chunk[:, None, :] - b[None, :, :], axis=2)
        best = min(best, float(np.min(distances)))
    return best


def absorb_small_families(families: list[list[PlaneInfo]], args: argparse.Namespace) -> list[list[PlaneInfo]]:
    if args.no_absorb_small_families or len(families) <= 1:
        return families

    families = [list(family) for family in families]
    max_points = max(family_total_points(family) for family in families)
    absorb_limit = max(args.absorb_max_points, int(max_points * args.absorb_max_ratio))

    changed = True
    while changed:
        changed = False
        families.sort(key=lambda group: (-family_total_points(group), min(p.index for p in group)))
        metadata = []
        for family in families:
            normal, d = weighted_average_plane(family)
            bbox_min, bbox_max = family_bbox(family)
            metadata.append(
                {
                    "points": family_total_points(family),
                    "normal": normal,
                    "d": d,
                    "centroid": family_centroid(family),
                    "bbox_min": bbox_min,
                    "bbox_max": bbox_max,
                    "samples": family_sample_points(family, args.absorb_sample_points),
                }
            )

        for source_index in range(len(families) - 1, -1, -1):
            source_meta = metadata[source_index]
            if source_meta["points"] > absorb_limit:
                continue

            best_target = -1
            best_score = float("inf")
            for target_index, target_meta in enumerate(metadata):
                if source_index == target_index:
                    continue
                if target_meta["points"] <= source_meta["points"]:
                    continue

                angle = normal_angle_deg(source_meta["normal"], target_meta["normal"])
                if angle > args.absorb_normal_deg:
                    continue

                centroid_plane_distance = abs(float(np.dot(target_meta["normal"], source_meta["centroid"]) + target_meta["d"]))
                if centroid_plane_distance > args.absorb_plane_distance:
                    continue

                gap = bbox_gap_arrays(
                    source_meta["bbox_min"],
                    source_meta["bbox_max"],
                    target_meta["bbox_min"],
                    target_meta["bbox_max"],
                )
                if gap > args.absorb_bbox_gap:
                    nearest = nearest_distance_points(source_meta["samples"], target_meta["samples"])
                    if nearest > args.absorb_nearest_distance:
                        continue
                else:
                    nearest = 0.0

                score = angle + centroid_plane_distance * 50.0 + nearest * 10.0
                if score < best_score:
                    best_score = score
                    best_target = target_index

            if best_target >= 0:
                families[best_target].extend(families[source_index])
                del families[source_index]
                changed = True
                break

    families.sort(key=lambda group: (-family_total_points(group), min(p.index for p in group)))
    return families


def write_plane_family_summary(plane_dir: Path, families: list[list[PlaneInfo]]) -> Path:
    out_path = plane_dir / "plane_family_summary.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "family",
                "planes",
                "plane_count",
                "total_points",
                "total_inliers",
                "normal_x",
                "normal_y",
                "normal_z",
                "d",
                "centroid_x",
                "centroid_y",
                "centroid_z",
                "bbox_min_x",
                "bbox_min_y",
                "bbox_min_z",
                "bbox_max_x",
                "bbox_max_y",
                "bbox_max_z",
                "bbox_size_x",
                "bbox_size_y",
                "bbox_size_z",
            ]
        )
        for family_index, family in enumerate(families):
            total_points = sum(p.count for p in family)
            total_inliers = sum(p.inliers for p in family)
            normal, d = weighted_average_plane(family)

            centroid = sum((p.centroid * p.count for p in family), np.zeros(3, dtype=np.float64)) / max(total_points, 1)
            bbox_min = np.min(np.vstack([p.bbox_min for p in family]), axis=0)
            bbox_max = np.max(np.vstack([p.bbox_max for p in family]), axis=0)
            bbox_size = bbox_max - bbox_min

            writer.writerow(
                [
                    family_index,
                    " ".join(str(p.index) for p in sorted(family, key=lambda p: p.index)),
                    len(family),
                    total_points,
                    total_inliers,
                    *normal,
                    d,
                    *centroid,
                    *bbox_min,
                    *bbox_max,
                    *bbox_size,
                ]
            )
    return out_path


def write_family_point_clouds(plane_dir: Path, families: list[list[PlaneInfo]]) -> list[Path]:
    out_paths: list[Path] = []
    palette = [
        (1.0, 0.15, 0.15),
        (0.15, 0.85, 1.0),
        (0.2, 1.0, 0.25),
        (1.0, 0.85, 0.15),
        (1.0, 0.2, 0.95),
        (0.45, 0.25, 1.0),
        (1.0, 0.55, 0.15),
        (0.75, 1.0, 0.15),
        (0.15, 0.55, 1.0),
        (1.0, 1.0, 1.0),
    ]

    for family_index, family in enumerate(families):
        merged = o3d.geometry.PointCloud()
        for plane in family:
            pcd = o3d.io.read_point_cloud(str(plane.path))
            merged += pcd

        if len(merged.points) == 0:
            continue

        color = palette[family_index % len(palette)]
        merged.paint_uniform_color(color)
        plane_names = "-".join(str(p.index) for p in sorted(family, key=lambda p: p.index))
        out_path = plane_dir / f"family_{family_index:02d}_{len(merged.points)}pts_planes_{plane_names}.ply"
        o3d.io.write_point_cloud(str(out_path), merged, write_ascii=False, compressed=False)
        out_paths.append(out_path)

    return out_paths


def find_default_source_path(plane_dir: Path) -> Path | None:
    suffix = "_open3d_planes"
    if plane_dir.name.endswith(suffix):
        stem = plane_dir.name[: -len(suffix)]
        for extension in (".obj", ".ply"):
            candidate = plane_dir.parent / f"{stem}{extension}"
            if candidate.exists():
                return candidate
    return None


def load_source_point_cloud(source_path: Path, sample_count: int) -> o3d.geometry.PointCloud:
    extension = source_path.suffix.lower()
    if extension == ".ply":
        pcd = o3d.io.read_point_cloud(str(source_path))
        if len(pcd.points) == 0:
            mesh = o3d.io.read_triangle_mesh(str(source_path))
            mesh.compute_vertex_normals()
            return mesh.sample_points_uniformly(number_of_points=sample_count)
        return pcd

    mesh = o3d.io.read_triangle_mesh(str(source_path))
    if len(mesh.vertices) == 0:
        raise RuntimeError(f"Unable to read source mesh: {source_path}")
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh.sample_points_uniformly(number_of_points=sample_count)


def ensure_normals(pcd: o3d.geometry.PointCloud) -> None:
    if pcd.has_normals():
        return
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=24))
    pcd.normalize_normals()


def remove_old_reclassified_outputs(plane_dir: Path) -> None:
    for path in plane_dir.glob("reclassified_family_*.ply"):
        path.unlink()
    outliers = plane_dir / "reclassified_outliers.ply"
    if outliers.exists():
        outliers.unlink()
    csv_path = plane_dir / "reclassified_summary.csv"
    if csv_path.exists():
        csv_path.unlink()


def write_reclassified_family_clouds(
    plane_dir: Path,
    families: list[list[PlaneInfo]],
    source_path: Path | None,
    args: argparse.Namespace,
) -> tuple[list[Path], list[list[int]], o3d.geometry.PointCloud | None]:
    if source_path is None:
        print("Skipped reclassification: source OBJ/PLY not found")
        return [], [], None

    source_pcd = load_source_point_cloud(source_path, args.reclassify_samples)
    if len(source_pcd.points) == 0:
        print(f"Skipped reclassification: empty source point cloud {source_path}")
        return [], [], source_pcd

    use_normals = not args.no_reclassify_normal
    if use_normals:
        ensure_normals(source_pcd)

    points = np.asarray(source_pcd.points)
    point_normals = np.asarray(source_pcd.normals) if use_normals and source_pcd.has_normals() else None

    family_planes = [weighted_average_plane(family) for family in families]
    family_normals = np.vstack([normal for normal, _ in family_planes])
    family_ds = np.array([d for _, d in family_planes], dtype=np.float64)
    normal_cos = math.cos(math.radians(args.reclassify_normal_deg))

    assignments: list[list[int]] = [[] for _ in families]
    outliers: list[int] = []

    for point_index, point in enumerate(points):
        distances = np.abs(family_normals @ point + family_ds)
        order = np.argsort(distances)
        assigned = False
        for family_index in order:
            if distances[family_index] > args.reclassify_distance:
                break
            if point_normals is not None:
                dot = abs(float(np.dot(point_normals[point_index], family_normals[family_index])))
                if dot < normal_cos:
                    continue
            assignments[int(family_index)].append(point_index)
            assigned = True
            break
        if not assigned:
            outliers.append(point_index)

    remove_old_reclassified_outputs(plane_dir)

    palette = [
        (1.0, 0.15, 0.15),
        (0.15, 0.85, 1.0),
        (0.2, 1.0, 0.25),
        (1.0, 0.85, 0.15),
        (1.0, 0.2, 0.95),
        (0.45, 0.25, 1.0),
        (1.0, 0.55, 0.15),
        (0.75, 1.0, 0.15),
        (0.15, 0.55, 1.0),
        (1.0, 1.0, 1.0),
    ]

    out_paths: list[Path] = []
    summary_path = plane_dir / "reclassified_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["family", "planes", "points", "output"])
        for family_index, indices in enumerate(assignments):
            if not indices:
                continue
            cloud = source_pcd.select_by_index(indices)
            cloud.paint_uniform_color(palette[family_index % len(palette)])
            plane_names = "-".join(str(p.index) for p in sorted(families[family_index], key=lambda p: p.index))
            out_path = plane_dir / f"reclassified_family_{family_index:02d}_{len(indices)}pts_planes_{plane_names}.ply"
            o3d.io.write_point_cloud(str(out_path), cloud, write_ascii=False, compressed=False)
            out_paths.append(out_path)
            writer.writerow([family_index, plane_names, len(indices), out_path.name])

        if outliers:
            cloud = source_pcd.select_by_index(outliers)
            cloud.paint_uniform_color((0.25, 0.25, 0.25))
            out_path = plane_dir / f"reclassified_outliers_{len(outliers)}pts.ply"
            o3d.io.write_point_cloud(str(out_path), cloud, write_ascii=False, compressed=False)
            out_paths.append(out_path)
            writer.writerow(["outliers", "", len(outliers), out_path.name])

    return out_paths, assignments, source_pcd


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normalize(normal)
    reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(reference, normal))) > 0.92:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = normalize(np.cross(reference, normal))
    v = normalize(np.cross(normal, u))
    return u, v


def build_planar_grid_surface(
    points: np.ndarray,
    normal: np.ndarray,
    d: float,
    cell_size: float,
    min_points_per_cell: int,
    surface_offset: float,
) -> tuple[o3d.geometry.TriangleMesh | None, o3d.geometry.LineSet | None]:
    if len(points) == 0:
        return None, None

    normal = normalize(normal)
    projected = points - (points @ normal + d)[:, None] * normal[None, :]
    origin = np.mean(projected, axis=0)
    origin = origin - (float(np.dot(normal, origin)) + d) * normal
    u, v = plane_basis(normal)

    xy = np.column_stack(((projected - origin) @ u, (projected - origin) @ v))
    ij = np.floor(xy / cell_size).astype(np.int64)
    counts: dict[tuple[int, int], int] = {}
    for i, j in ij:
        key = (int(i), int(j))
        counts[key] = counts.get(key, 0) + 1

    occupied = {key for key, count in counts.items() if count >= min_points_per_cell}
    if not occupied:
        return None, None

    vertex_map: dict[tuple[int, int], int] = {}
    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []

    def vertex_index(corner: tuple[int, int]) -> int:
        if corner not in vertex_map:
            x, y = corner
            position = origin + u * (x * cell_size) + v * (y * cell_size) + normal * surface_offset
            vertex_map[corner] = len(vertices)
            vertices.append(position)
        return vertex_map[corner]

    edge_counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}

    def add_edge(a: tuple[int, int], b: tuple[int, int]) -> None:
        edge = (a, b) if a <= b else (b, a)
        edge_counts[edge] = edge_counts.get(edge, 0) + 1

    for i, j in occupied:
        c00 = (i, j)
        c10 = (i + 1, j)
        c11 = (i + 1, j + 1)
        c01 = (i, j + 1)
        v00 = vertex_index(c00)
        v10 = vertex_index(c10)
        v11 = vertex_index(c11)
        v01 = vertex_index(c01)
        triangles.append((v00, v10, v11))
        triangles.append((v00, v11, v01))
        add_edge(c00, c10)
        add_edge(c10, c11)
        add_edge(c11, c01)
        add_edge(c01, c00)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    mesh.compute_vertex_normals()

    boundary_lines = []
    for edge, count in edge_counts.items():
        if count == 1:
            boundary_lines.append((vertex_index(edge[0]), vertex_index(edge[1])))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(vertices))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(boundary_lines, dtype=np.int32))
    line_set.paint_uniform_color((1.0, 1.0, 1.0))
    return mesh, line_set


def remove_old_surface_outputs(plane_dir: Path) -> None:
    for pattern in ("surface_family_*.ply", "surface_family_*.obj", "boundary_family_*.ply", "boundary_family_*.obj"):
        for path in plane_dir.glob(pattern):
            path.unlink()
    csv_path = plane_dir / "surface_summary.csv"
    if csv_path.exists():
        csv_path.unlink()


def line_set_to_sampled_point_cloud(line_set: o3d.geometry.LineSet, spacing: float) -> o3d.geometry.PointCloud:
    points = np.asarray(line_set.points)
    lines = np.asarray(line_set.lines)
    samples: list[np.ndarray] = []
    for a_index, b_index in lines:
        a = points[int(a_index)]
        b = points[int(b_index)]
        length = float(np.linalg.norm(b - a))
        steps = max(1, int(math.ceil(length / max(spacing, 1e-4))))
        for step in range(steps + 1):
            t = step / steps
            samples.append(a * (1.0 - t) + b * t)

    cloud = o3d.geometry.PointCloud()
    if samples:
        cloud.points = o3d.utility.Vector3dVector(np.asarray(samples))
    cloud.paint_uniform_color((1.0, 1.0, 1.0))
    return cloud


def write_obj_lines(path: Path, line_set: o3d.geometry.LineSet) -> None:
    points = np.asarray(line_set.points)
    lines = np.asarray(line_set.lines)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# ScanCover boundary lines\n")
        for point in points:
            f.write(f"v {point[0]:.7f} {point[1]:.7f} {point[2]:.7f}\n")
        for a_index, b_index in lines:
            f.write(f"l {int(a_index) + 1} {int(b_index) + 1}\n")


def triangle_circumradius_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = float(np.linalg.norm(a - b))
    bc = float(np.linalg.norm(b - c))
    ca = float(np.linalg.norm(c - a))
    area2 = abs(float(np.cross(b - a, c - a)))
    if area2 <= 1e-8:
        return float("inf")
    return (ab * bc * ca) / (2.0 * area2)


def build_alpha_boundary(
    points: np.ndarray,
    normal: np.ndarray,
    d: float,
    alpha_radius: float,
    max_points: int,
    surface_offset: float,
) -> o3d.geometry.LineSet | None:
    if len(points) < 4:
        return None

    normal = normalize(normal)
    projected = points - (points @ normal + d)[:, None] * normal[None, :]
    if len(projected) > max_points:
        chosen = np.linspace(0, len(projected) - 1, max_points, dtype=np.int64)
        projected = projected[chosen]

    origin = np.mean(projected, axis=0)
    origin = origin - (float(np.dot(normal, origin)) + d) * normal
    u, v = plane_basis(normal)
    xy = np.column_stack(((projected - origin) @ u, (projected - origin) @ v))

    # Remove duplicate projected samples. Dense BL grids often contain many nearly
    # identical points after projection, and Delaunay is sensitive to duplicates.
    quantized = np.round(xy / 1e-4).astype(np.int64)
    _, unique_indices = np.unique(quantized, axis=0, return_index=True)
    unique_indices.sort()
    xy = xy[unique_indices]
    projected = projected[unique_indices]
    if len(xy) < 4:
        return None

    try:
        triangulation = Delaunay(xy)
    except Exception:
        return None

    edge_counts: dict[tuple[int, int], int] = {}
    for simplex in triangulation.simplices:
        a_index, b_index, c_index = [int(x) for x in simplex]
        radius = triangle_circumradius_2d(xy[a_index], xy[b_index], xy[c_index])
        if radius > alpha_radius:
            continue
        for e0, e1 in ((a_index, b_index), (b_index, c_index), (c_index, a_index)):
            edge = (e0, e1) if e0 <= e1 else (e1, e0)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    if not boundary_edges:
        return None

    used_indices = sorted({i for edge in boundary_edges for i in edge})
    remap = {old: new for new, old in enumerate(used_indices)}
    boundary_points = projected[used_indices] + normal * surface_offset
    boundary_lines = [(remap[a], remap[b]) for a, b in boundary_edges]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(boundary_points)
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(boundary_lines, dtype=np.int32))
    line_set.paint_uniform_color((1.0, 1.0, 1.0))
    return line_set


def remove_old_alpha_outputs(plane_dir: Path) -> None:
    for pattern in ("alpha_boundary_family_*.ply", "alpha_boundary_family_*.obj"):
        for path in plane_dir.glob(pattern):
            path.unlink()
    csv_path = plane_dir / "alpha_boundary_summary.csv"
    if csv_path.exists():
        csv_path.unlink()


def write_alpha_family_boundaries(
    plane_dir: Path,
    families: list[list[PlaneInfo]],
    assignments: list[list[int]],
    source_pcd: o3d.geometry.PointCloud | None,
    args: argparse.Namespace,
) -> list[Path]:
    if source_pcd is None or not assignments:
        print("Skipped alpha boundary output: reclassified source points unavailable")
        return []

    points = np.asarray(source_pcd.points)
    remove_old_alpha_outputs(plane_dir)
    out_paths: list[Path] = []
    summary_path = plane_dir / "alpha_boundary_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["family", "planes", "points", "boundary_vertices", "boundary_lines", "alpha_radius", "boundary_ply", "boundary_obj"])
        for family_index, indices in enumerate(assignments):
            if not indices:
                continue
            normal, d = weighted_average_plane(families[family_index])
            family_points = points[np.asarray(indices, dtype=np.int64)]
            boundary = build_alpha_boundary(
                family_points,
                normal,
                d,
                args.alpha_boundary,
                args.alpha_max_points,
                args.surface_offset,
            )
            if boundary is None:
                continue

            plane_names = "-".join(str(p.index) for p in sorted(families[family_index], key=lambda p: p.index))
            boundary_ply = plane_dir / f"alpha_boundary_family_{family_index:02d}_planes_{plane_names}.ply"
            boundary_obj = plane_dir / f"alpha_boundary_family_{family_index:02d}_planes_{plane_names}.obj"
            boundary_cloud = line_set_to_sampled_point_cloud(boundary, max(args.surface_cell * 0.25, 0.01))
            o3d.io.write_point_cloud(str(boundary_ply), boundary_cloud, write_ascii=False, compressed=False)
            write_obj_lines(boundary_obj, boundary)
            out_paths.extend([boundary_ply, boundary_obj])
            writer.writerow(
                [
                    family_index,
                    plane_names,
                    len(indices),
                    len(boundary.points),
                    len(boundary.lines),
                    args.alpha_boundary,
                    boundary_ply.name,
                    boundary_obj.name,
                ]
            )
    return out_paths


def write_planar_family_surfaces(
    plane_dir: Path,
    families: list[list[PlaneInfo]],
    assignments: list[list[int]],
    source_pcd: o3d.geometry.PointCloud | None,
    args: argparse.Namespace,
) -> list[Path]:
    if source_pcd is None or not assignments:
        print("Skipped planar surface output: reclassified source points unavailable")
        return []

    points = np.asarray(source_pcd.points)
    remove_old_surface_outputs(plane_dir)
    out_paths: list[Path] = []
    summary_path = plane_dir / "surface_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["family", "planes", "points", "cells", "mesh_triangles", "boundary_lines", "mesh_ply", "mesh_obj", "boundary_ply", "boundary_obj"])
        for family_index, indices in enumerate(assignments):
            if not indices:
                continue
            normal, d = weighted_average_plane(families[family_index])
            family_points = points[np.asarray(indices, dtype=np.int64)]
            mesh, boundary = build_planar_grid_surface(
                family_points,
                normal,
                d,
                args.surface_cell,
                args.surface_min_points,
                args.surface_offset,
            )
            if mesh is None or boundary is None:
                continue

            plane_names = "-".join(str(p.index) for p in sorted(families[family_index], key=lambda p: p.index))
            mesh_ply = plane_dir / f"surface_family_{family_index:02d}_planes_{plane_names}.ply"
            mesh_obj = plane_dir / f"surface_family_{family_index:02d}_planes_{plane_names}.obj"
            boundary_ply = plane_dir / f"boundary_family_{family_index:02d}_planes_{plane_names}.ply"
            boundary_obj = plane_dir / f"boundary_family_{family_index:02d}_planes_{plane_names}.obj"
            o3d.io.write_triangle_mesh(str(mesh_ply), mesh, write_ascii=False, compressed=False)
            o3d.io.write_triangle_mesh(str(mesh_obj), mesh, write_ascii=True)
            boundary_cloud = line_set_to_sampled_point_cloud(boundary, args.surface_cell * 0.35)
            o3d.io.write_point_cloud(str(boundary_ply), boundary_cloud, write_ascii=False, compressed=False)
            write_obj_lines(boundary_obj, boundary)
            out_paths.extend([mesh_ply, mesh_obj, boundary_ply, boundary_obj])

            writer.writerow(
                [
                    family_index,
                    plane_names,
                    len(indices),
                    len(mesh.triangles) // 2,
                    len(mesh.triangles),
                    len(boundary.lines),
                    mesh_ply.name,
                    mesh_obj.name,
                    boundary_ply.name,
                    boundary_obj.name,
                ]
            )
    return out_paths


def main() -> int:
    args = parse_args()
    plane_dir = args.plane_dir
    summary_path = plane_dir / "plane_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    rows = load_summary(summary_path)
    planes = [read_plane(row, plane_dir) for row in rows]
    planes.sort(key=lambda p: p.index)

    stats_path = write_plane_stats(plane_dir, planes)
    candidates_path = write_merge_candidates(plane_dir, planes, args)
    families = build_merge_families(candidates_path, planes)
    pre_absorb_family_count = len(families)
    families = absorb_small_families(families, args)
    family_path = write_plane_family_summary(plane_dir, families)
    family_cloud_paths = write_family_point_clouds(plane_dir, families)
    source_path = args.source or find_default_source_path(plane_dir)
    reclassified_paths, assignments, source_pcd = write_reclassified_family_clouds(plane_dir, families, source_path, args)
    surface_paths = write_planar_family_surfaces(plane_dir, families, assignments, source_pcd, args)
    alpha_boundary_paths = write_alpha_family_boundaries(plane_dir, families, assignments, source_pcd, args)

    candidate_count = 0
    with candidates_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            candidate_count += int(row["candidate"])

    print(f"Loaded {len(planes)} planes from {plane_dir}")
    print(f"Wrote {stats_path}")
    print(f"Wrote {candidates_path}")
    print(f"Wrote {family_path}")
    print(f"Wrote {len(family_cloud_paths)} family point clouds")
    print(f"Wrote {len(reclassified_paths)} reclassified point clouds")
    print(f"Wrote {len(surface_paths)} planar surface files")
    print(f"Wrote {len(alpha_boundary_paths)} alpha boundary files")
    print(f"Merge candidates: {candidate_count}")
    print(f"Plane families: {len(families)} (pre-absorb {pre_absorb_family_count})")
    if source_path is not None:
        print(f"Reclassification source: {source_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
