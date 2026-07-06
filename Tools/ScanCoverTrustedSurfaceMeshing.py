#!/usr/bin/env python3
"""Build mapping-mesh candidates from trusted Raw surface points.

This is an offline diagnostic step. It intentionally consumes only
trusted_raw_surface.ply by default, so unconfirmed Meta-only areas do not get
silently filled into the primary mapping layer.

Outputs:

- trusted_cleaned_points.ply
- trusted_rejected_noise.ply
- mapping_mesh_candidate_conservative_bpa.ply
- mapping_mesh_candidate_main_only.ply
- mapping_mesh_candidate_continuous_poisson.ply
- mapping_mesh_quality_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d


DEFAULT_SESSION = Path(
    r"E:\PCAII\NEW-SCANCOVER"
    r"\ScanCoverExports\RepeatCoverageSessions"
    r"\ScanCover_RepeatCoverage_20260612_173442_610"
    r"\trusted_region_extraction"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mesh trusted Raw surface points.")
    parser.add_argument(
        "trusted_surface",
        nargs="?",
        type=Path,
        default=DEFAULT_SESSION / "trusted_raw_surface.ply",
        help="trusted_raw_surface.ply path.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output folder.")
    parser.add_argument("--voxel", type=float, default=0.025, help="Voxel downsample size in meters.")
    parser.add_argument("--normal-radius", type=float, default=0.10)
    parser.add_argument("--normal-max-nn", type=int, default=48)
    parser.add_argument("--stat-nb", type=int, default=24)
    parser.add_argument("--stat-std", type=float, default=2.3)
    parser.add_argument("--radius-nb", type=int, default=5)
    parser.add_argument("--radius", type=float, default=0.10)
    parser.add_argument("--bpa-radius-scale", type=float, default=1.8)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--poisson-density-quantile", type=float, default=0.08)
    parser.add_argument("--sample-points", type=int, default=120000)
    return parser.parse_args()


def distribution(values: np.ndarray) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
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


def bbox_report(points: np.ndarray) -> dict[str, object]:
    if len(points) == 0:
        return {"count": 0}
    mn = np.min(points, axis=0)
    mx = np.max(points, axis=0)
    return {
        "count": int(len(points)),
        "min": mn.tolist(),
        "max": mx.tolist(),
        "size": (mx - mn).tolist(),
        "center": ((mn + mx) * 0.5).tolist(),
    }


def make_cloud(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.reshape((-1, 3)))
    if colors is not None and len(colors) == len(points):
        cloud.colors = o3d.utility.Vector3dVector(colors.reshape((-1, 3)))
    return cloud


def write_cloud(path: Path, cloud: o3d.geometry.PointCloud) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False)


def clean_cloud(
    cloud: o3d.geometry.PointCloud,
    voxel: float,
    stat_nb: int,
    stat_std: float,
    radius_nb: int,
    radius: float,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, dict[str, object]]:
    original_count = len(cloud.points)
    working = cloud
    if voxel > 0:
        working = working.voxel_down_sample(voxel)

    down_count = len(working.points)
    if down_count == 0:
        return working, o3d.geometry.PointCloud(), {"originalPoints": original_count, "downsampledPoints": 0}

    stat_cloud, stat_indices = working.remove_statistical_outlier(nb_neighbors=stat_nb, std_ratio=stat_std)
    stat_set = set(int(i) for i in stat_indices)
    stat_rejected = working.select_by_index(stat_indices, invert=True)

    radius_cloud, radius_indices = stat_cloud.remove_radius_outlier(nb_points=radius_nb, radius=radius)
    radius_rejected = stat_cloud.select_by_index(radius_indices, invert=True)
    rejected = stat_rejected + radius_rejected

    return radius_cloud, rejected, {
        "originalPoints": int(original_count),
        "downsampledPoints": int(down_count),
        "statisticalKeptPoints": int(len(stat_indices)),
        "statisticalRejectedPoints": int(down_count - len(stat_set)),
        "radiusKeptPoints": int(len(radius_cloud.points)),
        "radiusRejectedPoints": int(len(radius_rejected.points)),
        "totalRejectedAfterDownsample": int(len(rejected.points)),
        "retainedRatioFromOriginal": float(len(radius_cloud.points) / max(1, original_count)),
        "retainedRatioFromDownsample": float(len(radius_cloud.points) / max(1, down_count)),
    }


def estimate_normals(cloud: o3d.geometry.PointCloud, radius: float, max_nn: int) -> None:
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    try:
        cloud.orient_normals_consistent_tangent_plane(32)
    except RuntimeError:
        # Sparse/disconnected clouds can fail orientation. Estimated normals are
        # still useful for BPA/Poisson diagnostics.
        pass


def mesh_quality(
    mesh: o3d.geometry.TriangleMesh,
    reference: o3d.geometry.PointCloud,
    sample_points: int,
) -> dict[str, object]:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    report: dict[str, object] = {
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "bbox": bbox_report(vertices),
    }
    if len(triangles) == 0 or len(vertices) == 0:
        return report

    report["surfaceArea"] = float(mesh.get_surface_area())
    try:
        clusters, counts, areas = mesh.cluster_connected_triangles()
        counts_np = np.asarray(counts, dtype=np.int64)
        areas_np = np.asarray(areas, dtype=np.float64)
        report["triangleComponents"] = {
            "count": int(len(counts_np)),
            "largestTriangleCount": int(np.max(counts_np)) if len(counts_np) else 0,
            "largestArea": float(np.max(areas_np)) if len(areas_np) else 0.0,
        }
    except RuntimeError:
        report["triangleComponents"] = {"count": 0}

    try:
        report["nonManifoldEdges"] = int(len(mesh.get_non_manifold_edges(allow_boundary_edges=True)))
        report["boundaryOrNonManifoldEdges"] = int(len(mesh.get_non_manifold_edges(allow_boundary_edges=False)))
    except RuntimeError:
        pass

    if len(reference.points) > 0:
        n = min(sample_points, max(1000, len(vertices)))
        sampled = mesh.sample_points_uniformly(number_of_points=n)
        distances = np.asarray(sampled.compute_point_cloud_distance(reference), dtype=np.float64)
        report["meshToTrustedPointDistanceMeters"] = distribution(distances)
    return report


def build_bpa_mesh(cloud: o3d.geometry.PointCloud, voxel: float, radius_scale: float) -> o3d.geometry.TriangleMesh:
    base = max(0.01, voxel * radius_scale)
    radii = o3d.utility.DoubleVector([base, base * 1.5, base * 2.25])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, radii)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def build_poisson_mesh(
    cloud: o3d.geometry.PointCloud,
    depth: int,
    density_quantile: float,
) -> tuple[o3d.geometry.TriangleMesh, dict[str, object]]:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=depth)
    density_arr = np.asarray(densities, dtype=np.float64)
    threshold = float(np.quantile(density_arr, density_quantile)) if len(density_arr) else 0.0
    if len(density_arr):
        mesh.remove_vertices_by_mask(density_arr < threshold)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh, {
        "depth": int(depth),
        "densityQuantile": float(density_quantile),
        "densityThreshold": threshold,
        "density": distribution(density_arr),
    }


def color_mesh(mesh: o3d.geometry.TriangleMesh, color: tuple[float, float, float]) -> None:
    if len(mesh.vertices) == 0:
        return
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=np.float64), (len(mesh.vertices), 1))
    )


def extract_main_component(mesh: o3d.geometry.TriangleMesh) -> tuple[o3d.geometry.TriangleMesh, dict[str, object]]:
    if len(mesh.triangles) == 0:
        return mesh, {"selectedComponent": -1, "selectedTriangles": 0, "componentCount": 0}

    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    clusters = np.asarray(triangle_clusters, dtype=np.int64)
    counts = np.asarray(cluster_n_triangles, dtype=np.int64)
    areas = np.asarray(cluster_area, dtype=np.float64)
    if len(counts) == 0:
        return mesh, {"selectedComponent": -1, "selectedTriangles": 0, "componentCount": 0}

    selected = int(np.argmax(areas))
    remove_mask = clusters != selected
    main = o3d.geometry.TriangleMesh(mesh)
    main.remove_triangles_by_mask(remove_mask)
    main.remove_unreferenced_vertices()
    main.remove_duplicated_vertices()
    main.remove_degenerate_triangles()
    main.remove_duplicated_triangles()
    main.compute_vertex_normals()
    return main, {
        "selectedComponent": selected,
        "componentCount": int(len(counts)),
        "selectedTriangles": int(counts[selected]),
        "selectedArea": float(areas[selected]),
        "selectedTriangleRatio": float(counts[selected] / max(1, np.sum(counts))),
        "selectedAreaRatio": float(areas[selected] / max(1e-8, np.sum(areas))),
    }


def main() -> None:
    args = parse_args()
    trusted_path = args.trusted_surface
    out_dir = args.out or (trusted_path.parent / "trusted_mesh_candidates")
    out_dir.mkdir(parents=True, exist_ok=True)

    cloud = o3d.io.read_point_cloud(str(trusted_path))
    if len(cloud.points) == 0:
        raise RuntimeError(f"Could not read trusted surface points: {trusted_path}")

    clean, rejected, clean_report = clean_cloud(
        cloud,
        args.voxel,
        args.stat_nb,
        args.stat_std,
        args.radius_nb,
        args.radius,
    )
    estimate_normals(clean, args.normal_radius, args.normal_max_nn)

    clean.paint_uniform_color((0.0, 0.95, 1.0))
    rejected.paint_uniform_color((1.0, 0.05, 0.05))
    write_cloud(out_dir / "trusted_cleaned_points.ply", clean)
    write_cloud(out_dir / "trusted_rejected_noise.ply", rejected)

    bpa_mesh = build_bpa_mesh(clean, args.voxel, args.bpa_radius_scale)
    color_mesh(bpa_mesh, (1.0, 1.0, 1.0))
    o3d.io.write_triangle_mesh(str(out_dir / "mapping_mesh_candidate_conservative_bpa.ply"), bpa_mesh, write_ascii=False)
    o3d.io.write_triangle_mesh(str(out_dir / "mapping_mesh_candidate_conservative_bpa.obj"), bpa_mesh, write_ascii=False)

    main_mesh, main_component_report = extract_main_component(bpa_mesh)
    color_mesh(main_mesh, (1.0, 1.0, 1.0))
    o3d.io.write_triangle_mesh(str(out_dir / "mapping_mesh_candidate_main_only.ply"), main_mesh, write_ascii=False)
    o3d.io.write_triangle_mesh(str(out_dir / "mapping_mesh_candidate_main_only.obj"), main_mesh, write_ascii=False)

    poisson_mesh, poisson_report = build_poisson_mesh(clean, args.poisson_depth, args.poisson_density_quantile)
    color_mesh(poisson_mesh, (0.0, 0.85, 1.0))
    o3d.io.write_triangle_mesh(str(out_dir / "mapping_mesh_candidate_continuous_poisson.ply"), poisson_mesh, write_ascii=False)
    o3d.io.write_triangle_mesh(str(out_dir / "mapping_mesh_candidate_continuous_poisson.obj"), poisson_mesh, write_ascii=False)

    report = {
        "input": str(trusted_path),
        "outputFolder": str(out_dir),
        "parameters": {
            "voxel": args.voxel,
            "normalRadius": args.normal_radius,
            "normalMaxNN": args.normal_max_nn,
            "statNb": args.stat_nb,
            "statStd": args.stat_std,
            "radiusNb": args.radius_nb,
            "radius": args.radius,
            "bpaRadiusScale": args.bpa_radius_scale,
            "poissonDepth": args.poisson_depth,
            "poissonDensityQuantile": args.poisson_density_quantile,
        },
        "cleaning": clean_report,
        "cleanedPointBounds": bbox_report(np.asarray(clean.points, dtype=np.float64)),
        "rejectedPointBounds": bbox_report(np.asarray(rejected.points, dtype=np.float64)),
        "bpaMesh": mesh_quality(bpa_mesh, clean, args.sample_points),
        "mainOnly": {
            "componentSelection": main_component_report,
            "mesh": mesh_quality(main_mesh, clean, args.sample_points),
        },
        "poisson": poisson_report,
        "poissonMesh": mesh_quality(poisson_mesh, clean, args.sample_points),
    }
    (out_dir / "mapping_mesh_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["cleaning"], ensure_ascii=False, indent=2))
    print(json.dumps({"bpaMesh": report["bpaMesh"], "poissonMesh": report["poissonMesh"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
