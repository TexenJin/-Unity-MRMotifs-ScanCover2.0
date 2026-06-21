#!/usr/bin/env python3
"""Build a surface-oriented mapping candidate from Raw evidence and Meta-guided landing outputs.

This script intentionally separates three layers:
- Raw evidence points: dense but noisy observations.
- Meta/structure landed surface: continuous surface scaffold corrected by Raw where trusted.
- Mapping surface candidate: the scaffold with Raw support/confidence diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import open3d as o3d


def read_cloud(path: Path, required: bool = True) -> o3d.geometry.PointCloud:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return o3d.geometry.PointCloud()
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty() and required:
        raise RuntimeError(f"Empty point cloud: {path}")
    return cloud


def write_cloud(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    cloud = o3d.geometry.PointCloud()
    if points.size:
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
        if colors is not None and colors.size:
            cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64, copy=False))
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False)


def cloud_points(cloud: o3d.geometry.PointCloud) -> np.ndarray:
    if cloud.is_empty():
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(cloud.points, dtype=np.float64)


def cloud_colors(cloud: o3d.geometry.PointCloud, fallback: Tuple[float, float, float]) -> np.ndarray:
    pts = cloud_points(cloud)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if cloud.has_colors():
        cols = np.asarray(cloud.colors, dtype=np.float64)
        if len(cols) == len(pts):
            return np.clip(cols, 0.0, 1.0)
    return np.tile(np.asarray(fallback, dtype=np.float64), (len(pts), 1))


def radius_counts(query: np.ndarray, support: np.ndarray, radius: float) -> np.ndarray:
    if len(query) == 0 or len(support) == 0:
        return np.zeros(len(query), dtype=np.int32)
    support_cloud = o3d.geometry.PointCloud()
    support_cloud.points = o3d.utility.Vector3dVector(support)
    kdtree = o3d.geometry.KDTreeFlann(support_cloud)
    counts = np.zeros(len(query), dtype=np.int32)
    for i, p in enumerate(query):
        count, _, _ = kdtree.search_radius_vector_3d(p, radius)
        counts[i] = count
    return counts


def nearest_distances(query: np.ndarray, support: np.ndarray) -> np.ndarray:
    if len(query) == 0 or len(support) == 0:
        return np.full(len(query), np.inf, dtype=np.float64)
    support_cloud = o3d.geometry.PointCloud()
    support_cloud.points = o3d.utility.Vector3dVector(support)
    kdtree = o3d.geometry.KDTreeFlann(support_cloud)
    distances = np.full(len(query), np.inf, dtype=np.float64)
    for i, p in enumerate(query):
        count, _, d2 = kdtree.search_knn_vector_3d(p, 1)
        if count:
            distances[i] = float(np.sqrt(d2[0]))
    return distances


def confidence_colors(raw_counts: np.ndarray, strict_counts: np.ndarray, min_support: int, high_support: int) -> Tuple[np.ndarray, Dict[str, int]]:
    colors = np.zeros((len(raw_counts), 3), dtype=np.float64)
    high = (raw_counts >= high_support) | ((strict_counts >= 2) & (raw_counts >= min_support))
    medium = ~high & ((raw_counts >= min_support) | (strict_counts >= 1))
    weak = ~(high | medium) & (raw_counts > 0)
    none = ~(high | medium | weak)

    colors[high] = (0.05, 1.0, 0.25)      # green: strong Raw support
    colors[medium] = (0.05, 0.85, 1.0)    # cyan: usable support
    colors[weak] = (1.0, 0.82, 0.05)      # yellow: weak support, keep as review
    colors[none] = (1.0, 0.2, 0.05)       # red: structure exists but Raw support is missing

    stats = {
        "high": int(np.count_nonzero(high)),
        "medium": int(np.count_nonzero(medium)),
        "weak": int(np.count_nonzero(weak)),
        "missing": int(np.count_nonzero(none)),
    }
    return colors, stats


def voxel_downsample_points(points: np.ndarray, colors: np.ndarray | None, voxel: float) -> Tuple[np.ndarray, np.ndarray | None]:
    if voxel <= 0 or len(points) == 0:
        return points, colors
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) == len(points):
        cloud.colors = o3d.utility.Vector3dVector(colors)
    down = cloud.voxel_down_sample(voxel)
    down_points = cloud_points(down)
    down_colors = cloud_colors(down, (1.0, 1.0, 1.0)) if down.has_colors() else None
    return down_points, down_colors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mapping surface candidate from Raw evidence + Meta/Raw landed surface.")
    parser.add_argument("raw_mapping_dir", type=Path, help="Folder containing mapping_input_candidate.ply")
    parser.add_argument("meta_surface_dir", type=Path, help="Folder containing trusted_raw_surface.ply")
    parser.add_argument("--out", type=Path, required=True, help="Output folder")
    parser.add_argument("--support-radius", type=float, default=0.08, help="Raw support radius in meters")
    parser.add_argument("--raw-covered-radius", type=float, default=0.08, help="Distance from mapping surface used to mark unresolved Raw regions")
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--high-support", type=int, default=8)
    parser.add_argument("--surface-voxel", type=float, default=0.0, help="Optional visual downsample for candidate surface")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    raw_candidate_cloud = read_cloud(args.raw_mapping_dir / "mapping_input_candidate.ply")
    raw_strict_cloud = read_cloud(args.raw_mapping_dir / "mapping_input_strict_candidate.ply", required=False)
    if raw_strict_cloud.is_empty():
        raw_strict_cloud = raw_candidate_cloud

    trusted_cloud = read_cloud(args.meta_surface_dir / "trusted_raw_surface.ply")
    unconfirmed_cloud = read_cloud(args.meta_surface_dir / "unconfirmed_candidate_surface.ply", required=False)

    raw_points = cloud_points(raw_candidate_cloud)
    raw_colors = cloud_colors(raw_candidate_cloud, (1.0, 0.85, 0.05))
    strict_points = cloud_points(raw_strict_cloud)
    trusted_points = cloud_points(trusted_cloud)
    unconfirmed_points = cloud_points(unconfirmed_cloud)

    raw_support_counts = radius_counts(trusted_points, raw_points, args.support_radius)
    strict_support_counts = radius_counts(trusted_points, strict_points, args.support_radius)
    conf_colors, conf_stats = confidence_colors(raw_support_counts, strict_support_counts, args.min_support, args.high_support)

    # Candidate geometry is the Raw-landed continuous surface, not the raw evidence cloud.
    candidate_points = trusted_points
    candidate_colors = np.tile(np.asarray((1.0, 1.0, 1.0), dtype=np.float64), (len(candidate_points), 1))
    candidate_points_out, candidate_colors_out = voxel_downsample_points(candidate_points, candidate_colors, args.surface_voxel)

    raw_to_surface = nearest_distances(raw_points, trusted_points)
    unresolved_raw_mask = raw_to_surface > args.raw_covered_radius
    unresolved_raw_points = raw_points[unresolved_raw_mask]
    unresolved_raw_colors = np.tile(np.asarray((1.0, 0.25, 0.05), dtype=np.float64), (len(unresolved_raw_points), 1))

    # Context: bright candidate surface + dim unresolved structure, useful for orientation.
    context_points = candidate_points
    context_colors = np.tile(np.asarray((0.88, 1.0, 1.0), dtype=np.float64), (len(context_points), 1))
    if len(unconfirmed_points):
        context_points = np.vstack([context_points, unconfirmed_points])
        context_colors = np.vstack([
            context_colors,
            np.tile(np.asarray((0.35, 0.35, 0.35), dtype=np.float64), (len(unconfirmed_points), 1)),
        ])

    write_cloud(args.out / "raw_evidence_points.ply", raw_points, raw_colors)
    write_cloud(args.out / "mapping_surface_candidate.ply", candidate_points_out, candidate_colors_out)
    write_cloud(args.out / "mapping_surface_confidence.ply", trusted_points, conf_colors)
    write_cloud(args.out / "mapping_surface_with_context.ply", context_points, context_colors)
    write_cloud(args.out / "unresolved_raw_regions.ply", unresolved_raw_points, unresolved_raw_colors)
    write_cloud(args.out / "unconfirmed_structure_regions.ply", unconfirmed_points, np.tile(np.asarray((0.55, 0.55, 0.55)), (len(unconfirmed_points), 1)) if len(unconfirmed_points) else None)

    report = {
        "rawMappingDir": str(args.raw_mapping_dir),
        "metaSurfaceDir": str(args.meta_surface_dir),
        "outputDir": str(args.out),
        "parameters": {
            "supportRadiusMeters": args.support_radius,
            "rawCoveredRadiusMeters": args.raw_covered_radius,
            "minSupport": args.min_support,
            "highSupport": args.high_support,
            "surfaceVoxelMeters": args.surface_voxel,
        },
        "counts": {
            "rawEvidencePoints": int(len(raw_points)),
            "strictRawEvidencePoints": int(len(strict_points)),
            "trustedSurfacePoints": int(len(trusted_points)),
            "candidateSurfacePoints": int(len(candidate_points_out)),
            "unconfirmedStructurePoints": int(len(unconfirmed_points)),
            "unresolvedRawPoints": int(len(unresolved_raw_points)),
        },
        "ratios": {
            "unresolvedRawRatio": float(len(unresolved_raw_points) / max(1, len(raw_points))),
            "trustedSurfaceHighOrMediumSupportRatio": float((conf_stats["high"] + conf_stats["medium"]) / max(1, len(trusted_points))),
            "trustedSurfaceWeakOrMissingSupportRatio": float((conf_stats["weak"] + conf_stats["missing"]) / max(1, len(trusted_points))),
        },
        "surfaceConfidenceCounts": conf_stats,
        "semantics": {
            "raw_evidence_points.ply": "Raw depth evidence layer; dense/noisy material, not final mapping surface.",
            "mapping_surface_candidate.ply": "Surface-oriented candidate: Meta structure with Raw-confirmed landing. This is the first file to inspect for mapping surface continuity.",
            "mapping_surface_confidence.ply": "Same candidate geometry, colored by nearby Raw support strength.",
            "mapping_surface_with_context.ply": "Candidate surface plus dim unconfirmed structure for orientation.",
            "unresolved_raw_regions.ply": "Raw evidence not absorbed by the candidate surface; use for later supplement/diagnosis, not as immediate wanted list.",
            "unconfirmed_structure_regions.ply": "Meta structure without enough Raw landing; often under-scanned or unreliable regions.",
        },
    }
    (args.out / "mapping_surface_candidate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print(json.dumps(report["ratios"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
