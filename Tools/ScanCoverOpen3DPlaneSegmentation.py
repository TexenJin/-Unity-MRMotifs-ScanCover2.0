#!/usr/bin/env python3
"""Analyze an exported ScanCover BL OBJ with Open3D plane segmentation.

Usage:
  python Tools/ScanCoverOpen3DPlaneSegmentation.py ScanCoverExports/ScanCover_BLSurfaceMesh_....obj

Outputs are written next to the input OBJ in a folder named
<obj-name>_open3d_planes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path, help="OBJ/PLY mesh exported from ScanCover.")
    parser.add_argument("--sample-count", type=int, default=60000)
    parser.add_argument("--distance", type=float, default=0.035, help="RANSAC inlier distance in meters.")
    parser.add_argument("--ransac-n", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--min-inliers", type=int, default=900)
    parser.add_argument("--max-planes", type=int, default=12)
    parser.add_argument("--normal-radius", type=float, default=0.12)
    parser.add_argument("--normal-max-nn", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.mesh.exists():
        raise FileNotFoundError(args.mesh)

    out_dir = args.mesh.with_suffix("")
    out_dir = out_dir.parent / f"{out_dir.name}_open3d_planes"
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    if mesh.is_empty():
        raise RuntimeError(f"Open3D could not read mesh: {args.mesh}")

    mesh.compute_vertex_normals()
    sampled = mesh.sample_points_uniformly(number_of_points=max(1000, args.sample_count))
    sampled.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.normal_radius,
            max_nn=args.normal_max_nn,
        )
    )

    remaining = sampled
    summary_rows: list[dict[str, object]] = []
    colors = [
        (1.0, 0.15, 0.15),
        (0.1, 0.8, 1.0),
        (0.2, 1.0, 0.35),
        (1.0, 0.85, 0.1),
        (0.9, 0.35, 1.0),
        (1.0, 0.55, 0.15),
    ]

    for plane_index in range(args.max_planes):
        if len(remaining.points) < args.min_inliers:
            break

        plane_model, inliers = remaining.segment_plane(
            distance_threshold=args.distance,
            ransac_n=args.ransac_n,
            num_iterations=args.iterations,
        )
        if len(inliers) < args.min_inliers:
            break

        plane_cloud = remaining.select_by_index(inliers)
        rest_cloud = remaining.select_by_index(inliers, invert=True)
        color = colors[plane_index % len(colors)]
        plane_cloud.paint_uniform_color(color)

        plane_path = out_dir / f"plane_{plane_index:02d}_{len(inliers)}pts.ply"
        o3d.io.write_point_cloud(str(plane_path), plane_cloud, write_ascii=False)

        a, b, c, d = plane_model
        summary_rows.append(
            {
                "plane": plane_index,
                "inliers": len(inliers),
                "a": a,
                "b": b,
                "c": c,
                "d": d,
                "path": plane_path.name,
            }
        )
        remaining = rest_cloud

    remaining_path = out_dir / "remaining_outliers.ply"
    o3d.io.write_point_cloud(str(remaining_path), remaining, write_ascii=False)

    summary_path = out_dir / "plane_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["plane", "inliers", "a", "b", "c", "d", "path"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"mesh: {args.mesh}")
    print(f"sampled points: {len(sampled.points)}")
    print(f"planes: {len(summary_rows)}")
    print(f"out: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
