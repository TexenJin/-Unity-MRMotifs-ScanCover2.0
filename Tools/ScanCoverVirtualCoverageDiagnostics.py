#!/usr/bin/env python3
"""Coverage diagnostics for virtual Quest3 room scanning.

This compares a truth room mesh against observed virtual Quest3 points or
observation-feature cells. The output is meant for CloudCompare: covered points
and missing points are saved separately, and coverage is summarized by height
band so ceiling/upper-wall gaps are visible as data, not guesswork.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-mesh", type=Path, required=True)
    parser.add_argument("--observed-ply", type=Path)
    parser.add_argument("--feature-csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=220000)
    parser.add_argument("--coverage-radius", type=float, default=0.06)
    parser.add_argument(
        "--vertical-axis",
        default="auto",
        choices=["auto", "x", "y", "z"],
        help="Room vertical axis. Auto picks the smallest mesh extent, matching the Replica room convention used here.",
    )
    parser.add_argument(
        "--top-band",
        type=float,
        default=0.35,
        help="Meters below the room top considered the ceiling/upper-band coverage area.",
    )
    parser.add_argument(
        "--bottom-band",
        type=float,
        default=0.25,
        help="Meters above the room bottom considered the floor/lower-band coverage area.",
    )
    return parser.parse_args()


def axis_index(axis: str, bounds: o3d.geometry.AxisAlignedBoundingBox) -> int:
    if axis != "auto":
        return {"x": 0, "y": 1, "z": 2}[axis]
    extent = np.asarray(bounds.get_extent(), dtype=np.float64)
    return int(np.argmin(extent))


def load_truth_samples(mesh_path: Path, sample_count: int) -> tuple[np.ndarray, o3d.geometry.AxisAlignedBoundingBox]:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Could not load truth mesh: {mesh_path}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    sampled = mesh.sample_points_uniformly(number_of_points=sample_count, use_triangle_normal=True)
    points = np.asarray(sampled.points, dtype=np.float64)
    if len(points) == 0:
        raise RuntimeError(f"Could not sample truth mesh: {mesh_path}")
    return points, mesh.get_axis_aligned_bounding_box()


def load_observed_points(args: argparse.Namespace) -> np.ndarray:
    sources = [args.observed_ply is not None, args.feature_csv is not None]
    if sum(sources) != 1:
        raise ValueError("Pass exactly one of --observed-ply or --feature-csv.")

    if args.observed_ply is not None:
        pcd = o3d.io.read_point_cloud(str(args.observed_ply))
        if pcd.is_empty():
            raise RuntimeError(f"Could not load observed point cloud: {args.observed_ply}")
        return np.asarray(pcd.points, dtype=np.float64)

    points: list[list[float]] = []
    with args.feature_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points.append([float(row["meanX"]), float(row["meanY"]), float(row["meanZ"])])
            except (KeyError, ValueError) as exc:
                raise RuntimeError(f"Invalid feature CSV row in {args.feature_csv}") from exc
    if not points:
        raise RuntimeError(f"No observed feature cells in {args.feature_csv}")
    return np.asarray(points, dtype=np.float64)


def make_cloud(points: np.ndarray, colors: np.ndarray | tuple[float, float, float]) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if isinstance(colors, tuple):
        color_array = np.tile(np.asarray(colors, dtype=np.float64), (len(points), 1))
    else:
        color_array = colors.astype(np.float64)
    pcd.colors = o3d.utility.Vector3dVector(color_array)
    return pcd


def ratio(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if len(mask) else 0.0


def summarize_band(name: str, band_mask: np.ndarray, covered_mask: np.ndarray) -> dict[str, float | int | str]:
    count = int(np.sum(band_mask))
    if count == 0:
        return {"name": name, "sampleCount": 0, "coveredCount": 0, "missingCount": 0, "coverageRatio": 0.0}
    covered_count = int(np.sum(covered_mask & band_mask))
    return {
        "name": name,
        "sampleCount": count,
        "coveredCount": covered_count,
        "missingCount": count - covered_count,
        "coverageRatio": covered_count / count,
    }


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    truth_points, bounds = load_truth_samples(args.truth_mesh, args.sample_count)
    observed_points = load_observed_points(args)
    if len(observed_points) == 0:
        raise RuntimeError("Observed point set is empty.")

    kdtree = o3d.geometry.KDTreeFlann(make_cloud(observed_points, (1.0, 1.0, 1.0)))
    nearest = np.full(len(truth_points), np.inf, dtype=np.float64)
    for i, point in enumerate(truth_points):
        _, _, dist2 = kdtree.search_knn_vector_3d(point, 1)
        if dist2:
            nearest[i] = float(np.sqrt(dist2[0]))

    covered = nearest <= args.coverage_radius
    missing = ~covered

    vertical = axis_index(args.vertical_axis, bounds)
    min_bound = np.asarray(bounds.get_min_bound(), dtype=np.float64)
    max_bound = np.asarray(bounds.get_max_bound(), dtype=np.float64)
    coord = truth_points[:, vertical]
    top_mask = coord >= (max_bound[vertical] - args.top_band)
    bottom_mask = coord <= (min_bound[vertical] + args.bottom_band)
    middle_mask = ~(top_mask | bottom_mask)

    colors = np.zeros((len(truth_points), 3), dtype=np.float64)
    colors[covered] = np.asarray((0.0, 0.95, 1.0), dtype=np.float64)
    colors[missing] = np.asarray((1.0, 0.05, 0.05), dtype=np.float64)
    o3d.io.write_point_cloud(str(args.out / "coverage_all.ply"), make_cloud(truth_points, colors), write_ascii=False)
    o3d.io.write_point_cloud(str(args.out / "covered_coverage.ply"), make_cloud(truth_points[covered], (0.0, 0.95, 1.0)), write_ascii=False)
    o3d.io.write_point_cloud(str(args.out / "missing_coverage.ply"), make_cloud(truth_points[missing], (1.0, 0.05, 0.05)), write_ascii=False)

    summary = {
        "schema": "ScanCoverVirtualCoverageDiagnostics/v1",
        "truthMesh": str(args.truth_mesh),
        "observedSource": str(args.observed_ply or args.feature_csv),
        "sampleCount": int(len(truth_points)),
        "observedPointCount": int(len(observed_points)),
        "coverageRadiusMeters": args.coverage_radius,
        "verticalAxis": "xyz"[vertical],
        "coverageRatio": ratio(covered),
        "missingRatio": ratio(missing),
        "nearestDistanceMeters": {
            "mean": float(np.mean(nearest[np.isfinite(nearest)])),
            "p50": float(np.percentile(nearest[np.isfinite(nearest)], 50)),
            "p90": float(np.percentile(nearest[np.isfinite(nearest)], 90)),
            "p95": float(np.percentile(nearest[np.isfinite(nearest)], 95)),
        },
        "bands": [
            summarize_band("top", top_mask, covered),
            summarize_band("middle", middle_mask, covered),
            summarize_band("bottom", bottom_mask, covered),
        ],
        "files": {
            "coverageAll": str(args.out / "coverage_all.ply"),
            "covered": str(args.out / "covered_coverage.ply"),
            "missing": str(args.out / "missing_coverage.ply"),
        },
    }
    (args.out / "coverage_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
