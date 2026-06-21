#!/usr/bin/env python3
"""Stabilize Raw-depth mapping points before meshing.

This step is intentionally between "candidate point cloud" and mesh creation:

- collapse multi-frame thickness back to a local median surface;
- peel edge/oblique/unstable points into a risk layer;
- keep small coherent structures as a separate detail layer;
- mesh only after the local surface has been stabilized.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d


DEFAULT_INPUT = Path(
    r"D:\PCA\Unity-MRMotifs-ScanCover-main"
    r"\ScanCoverExports\RepeatCoverageSessions\RepeatCoverageSessions"
    r"\raw_depth_mapping_input_full_reworked\mapping_input_candidate.ply"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stabilize local Raw-depth surface patches.")
    parser.add_argument("candidate", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--risk-points",
        type=Path,
        default=None,
        help="Optional risk_boundary_or_oblique.ply to peel from the main surface.",
    )
    parser.add_argument("--work-voxel", type=float, default=0.04)
    parser.add_argument("--patch-radius", type=float, default=0.13)
    parser.add_argument("--min-main-neighbors", type=int, default=14)
    parser.add_argument("--min-detail-neighbors", type=int, default=8)
    parser.add_argument("--max-main-thickness", type=float, default=0.040)
    parser.add_argument("--max-detail-thickness", type=float, default=0.075)
    parser.add_argument("--max-collapse-move", type=float, default=0.090)
    parser.add_argument("--max-main-surface-ratio", type=float, default=0.060)
    parser.add_argument("--max-detail-surface-ratio", type=float, default=0.120)
    parser.add_argument("--max-linearity", type=float, default=0.92)
    parser.add_argument("--normal-radius", type=float, default=0.12)
    parser.add_argument("--normal-max-nn", type=int, default=48)
    parser.add_argument("--bpa-radius-scale", type=float, default=2.0)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--poisson-density-quantile", type=float, default=0.08)
    parser.add_argument("--progress-step", type=int, default=25000)
    return parser.parse_args()


def voxel_keys(points: np.ndarray, voxel: float) -> set[tuple[int, int, int]]:
    if len(points) == 0:
        return set()
    q = np.floor(points / max(voxel, 1e-6)).astype(np.int64)
    return {tuple(row) for row in q}


def as_cloud(points: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    if len(points) > 0:
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        cloud.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, dtype=np.float64), (len(points), 1)))
    return cloud


def write_cloud(path: Path, points: np.ndarray, color: tuple[float, float, float]) -> None:
    o3d.io.write_point_cloud(str(path), as_cloud(points, color), write_ascii=False, compressed=False)


def write_labeled_cloud(path: Path, groups: list[tuple[np.ndarray, tuple[float, float, float]]]) -> None:
    pts_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    for points, color in groups:
        if len(points) == 0:
            continue
        pts_parts.append(points.astype(np.float64))
        color_parts.append(np.tile(np.array(color, dtype=np.float64), (len(points), 1)))
    cloud = o3d.geometry.PointCloud()
    if pts_parts:
        cloud.points = o3d.utility.Vector3dVector(np.vstack(pts_parts))
        cloud.colors = o3d.utility.Vector3dVector(np.vstack(color_parts))
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False)


def distribution(values: list[float] | np.ndarray) -> dict[str, object]:
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


def pca_patch(neighbors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(neighbors, axis=0)
    centered = neighbors - center
    cov = (centered.T @ centered) / max(1, len(neighbors))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)
    return eigvals[order], eigvecs[:, order], center


def stabilize_points(
    cloud: o3d.geometry.PointCloud,
    args: argparse.Namespace,
    risk_keys: set[tuple[int, int, int]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    points = np.asarray(cloud.points)
    tree = o3d.geometry.KDTreeFlann(cloud)
    risk_keys = risk_keys or set()

    main: list[np.ndarray] = []
    detail: list[np.ndarray] = []
    risk: list[np.ndarray] = []
    rejected: list[np.ndarray] = []

    thicknesses: list[float] = []
    surface_ratios: list[float] = []
    collapse_moves: list[float] = []
    neighbor_counts: list[int] = []
    reasons: dict[str, int] = {
        "low_support": 0,
        "line_like_patch": 0,
        "thick_main": 0,
        "thick_detail": 0,
        "large_collapse_move": 0,
        "main": 0,
        "detail": 0,
    }

    eps = 1e-9
    total = len(points)
    for i, point in enumerate(points):
        if args.progress_step > 0 and i > 0 and i % args.progress_step == 0:
            print(f"[stabilize] {i}/{total}")

        if tuple(np.floor(point / max(args.work_voxel, 1e-6)).astype(np.int64)) in risk_keys:
            risk.append(point)
            reasons["source_risk_voxel"] = reasons.get("source_risk_voxel", 0) + 1
            continue

        _, idxs, _ = tree.search_radius_vector_3d(point, args.patch_radius)
        count = len(idxs)
        neighbor_counts.append(count)
        if count < args.min_detail_neighbors:
            rejected.append(point)
            reasons["low_support"] += 1
            continue

        neighbors = points[np.asarray(idxs, dtype=np.int64)]
        eigvals, eigvecs, center = pca_patch(neighbors)
        normal = eigvecs[:, 0]
        thickness = math.sqrt(max(float(eigvals[0]), 0.0))
        total_var = float(np.sum(eigvals)) + eps
        surface_ratio = float(eigvals[0]) / total_var
        linearity = float(eigvals[2] - eigvals[1]) / (float(eigvals[2]) + eps)

        signed = (neighbors - center) @ normal
        median_offset = float(np.median(signed))
        plane_point = center + normal * median_offset
        move = float((point - plane_point) @ normal)
        corrected = point - normal * move

        thicknesses.append(thickness)
        surface_ratios.append(surface_ratio)
        collapse_moves.append(abs(move))

        if linearity > args.max_linearity and count < args.min_main_neighbors * 2:
            detail.append(corrected)
            reasons["line_like_patch"] += 1
            continue

        if abs(move) > args.max_collapse_move:
            risk.append(point)
            reasons["large_collapse_move"] += 1
            continue

        if (
            count >= args.min_main_neighbors
            and thickness <= args.max_main_thickness
            and surface_ratio <= args.max_main_surface_ratio
            and linearity <= args.max_linearity
        ):
            main.append(corrected)
            reasons["main"] += 1
        elif thickness <= args.max_detail_thickness and surface_ratio <= args.max_detail_surface_ratio:
            detail.append(corrected)
            reasons["detail"] += 1
        else:
            risk.append(point)
            if thickness > args.max_detail_thickness:
                reasons["thick_detail"] += 1
            else:
                reasons["thick_main"] += 1

    groups = {
        "main": np.asarray(main, dtype=np.float64).reshape((-1, 3)),
        "detail": np.asarray(detail, dtype=np.float64).reshape((-1, 3)),
        "risk": np.asarray(risk, dtype=np.float64).reshape((-1, 3)),
        "rejected": np.asarray(rejected, dtype=np.float64).reshape((-1, 3)),
    }
    report = {
        "inputPoints": int(total),
        "counts": {name: int(len(value)) for name, value in groups.items()},
        "ratios": {name: float(len(value) / max(1, total)) for name, value in groups.items()},
        "reasons": reasons,
        "neighborCount": distribution(neighbor_counts),
        "patchThicknessMeters": distribution(thicknesses),
        "surfaceRatio": distribution(surface_ratios),
        "collapseMoveMeters": distribution(collapse_moves),
    }
    return groups, report


def clean_for_mesh(points: np.ndarray, voxel: float) -> o3d.geometry.PointCloud:
    cloud = as_cloud(points, (1.0, 1.0, 1.0))
    if len(points) == 0:
        return cloud
    cloud = cloud.voxel_down_sample(voxel)
    if len(cloud.points) >= 32:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.4)
        cloud, _ = cloud.remove_radius_outlier(nb_points=4, radius=voxel * 3.0)
    if len(cloud.points) >= 32:
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel * 4.0, 0.10), max_nn=48)
        )
        cloud.orient_normals_consistent_tangent_plane(24)
    return cloud


def mesh_report(mesh: o3d.geometry.TriangleMesh) -> dict[str, object]:
    if len(mesh.vertices) == 0:
        return {"vertices": 0, "triangles": 0}
    clusters, counts, areas = mesh.cluster_connected_triangles()
    counts_arr = np.asarray(counts, dtype=np.int64)
    areas_arr = np.asarray(areas, dtype=np.float64)
    return {
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
        "components": int(len(counts_arr)),
        "largestComponentTriangles": int(np.max(counts_arr)) if len(counts_arr) else 0,
        "largestComponentArea": float(np.max(areas_arr)) if len(areas_arr) else 0.0,
        "boundaryOrNonManifoldEdges": int(len(mesh.get_non_manifold_edges(allow_boundary_edges=False))),
        "surfaceArea": float(mesh.get_surface_area()) if len(mesh.triangles) else 0.0,
    }


def build_meshes(out_dir: Path, main_points: np.ndarray, detail_points: np.ndarray, args: argparse.Namespace) -> dict[str, object]:
    report: dict[str, object] = {}
    mesh_inputs = {
        "main": main_points,
        "main_plus_detail": np.vstack([main_points, detail_points]) if len(detail_points) else main_points,
    }
    for label, points in mesh_inputs.items():
        cloud = clean_for_mesh(points, args.work_voxel)
        o3d.io.write_point_cloud(str(out_dir / f"stabilized_{label}_mesh_input.ply"), cloud, write_ascii=False)
        if len(cloud.points) < 64:
            report[label] = {"meshInputPoints": int(len(cloud.points)), "skipped": True}
            continue

        radii = o3d.utility.DoubleVector(
            [args.work_voxel * args.bpa_radius_scale, args.work_voxel * args.bpa_radius_scale * 1.8]
        )
        bpa = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, radii)
        bpa.remove_duplicated_vertices()
        bpa.remove_duplicated_triangles()
        bpa.remove_degenerate_triangles()
        bpa.remove_unreferenced_vertices()
        bpa.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out_dir / f"mapping_mesh_stabilized_{label}_bpa.ply"), bpa, write_ascii=False)

        poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud, depth=args.poisson_depth
        )
        density_arr = np.asarray(densities)
        if len(density_arr):
            cutoff = float(np.quantile(density_arr, args.poisson_density_quantile))
            poisson.remove_vertices_by_mask(density_arr < cutoff)
        poisson.remove_duplicated_vertices()
        poisson.remove_duplicated_triangles()
        poisson.remove_degenerate_triangles()
        poisson.remove_unreferenced_vertices()
        poisson.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out_dir / f"mapping_mesh_stabilized_{label}_poisson.ply"), poisson, write_ascii=False)

        report[label] = {
            "meshInputPoints": int(len(cloud.points)),
            "bpa": mesh_report(bpa),
            "poisson": mesh_report(poisson),
        }
    return report


def main() -> None:
    args = parse_args()
    candidate = args.candidate
    out_dir = args.out or candidate.parent / "surface_stabilized_from_candidate"
    out_dir.mkdir(parents=True, exist_ok=True)

    cloud = o3d.io.read_point_cloud(str(candidate))
    if len(cloud.points) == 0:
        raise RuntimeError(f"No points in {candidate}")

    original_count = len(cloud.points)
    print(f"[input] {candidate}")
    print(f"[input] points={original_count}")
    if args.work_voxel > 0:
        cloud = cloud.voxel_down_sample(args.work_voxel)
        print(f"[downsample] voxel={args.work_voxel:.3f}m points={len(cloud.points)}")

    if len(cloud.points) >= 32:
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=args.normal_radius, max_nn=args.normal_max_nn)
        )

    source_risk_points = np.empty((0, 3), dtype=np.float64)
    risk_keys_set: set[tuple[int, int, int]] = set()
    if args.risk_points and args.risk_points.exists():
        risk_cloud = o3d.io.read_point_cloud(str(args.risk_points))
        if len(risk_cloud.points) > 0 and args.work_voxel > 0:
            risk_cloud = risk_cloud.voxel_down_sample(args.work_voxel)
        source_risk_points = np.asarray(risk_cloud.points, dtype=np.float64)
        risk_keys_set = voxel_keys(source_risk_points, args.work_voxel)
        print(f"[risk] {args.risk_points}")
        print(f"[risk] points={len(source_risk_points)} voxels={len(risk_keys_set)}")

    groups, report = stabilize_points(cloud, args, risk_keys_set)
    if len(source_risk_points) > 0:
        if len(groups["risk"]) > 0:
            groups["risk"] = np.vstack([groups["risk"], source_risk_points])
        else:
            groups["risk"] = source_risk_points
        report["counts"]["risk"] = int(len(groups["risk"]))
        report["ratios"]["risk"] = float(len(groups["risk"]) / max(1, len(cloud.points)))
        report["sourceRiskPoints"] = int(len(source_risk_points))
        report["sourceRiskVoxels"] = int(len(risk_keys_set))
    report["source"] = str(candidate)
    report["originalInputPoints"] = int(original_count)
    report["workVoxel"] = float(args.work_voxel)
    report["patchRadius"] = float(args.patch_radius)

    write_cloud(out_dir / "stabilized_surface_points.ply", groups["main"], (0.15, 0.95, 1.0))
    write_cloud(out_dir / "stabilized_detail_points.ply", groups["detail"], (1.0, 0.85, 0.05))
    write_cloud(out_dir / "edge_risk_points.ply", groups["risk"], (1.0, 0.05, 0.05))
    write_cloud(out_dir / "rejected_unstable_points.ply", groups["rejected"], (0.85, 0.0, 1.0))
    write_labeled_cloud(
        out_dir / "stabilized_all_labeled.ply",
        [
            (groups["main"], (0.15, 0.95, 1.0)),
            (groups["detail"], (1.0, 0.85, 0.05)),
            (groups["risk"], (1.0, 0.05, 0.05)),
            (groups["rejected"], (0.85, 0.0, 1.0)),
        ],
    )

    report["meshes"] = build_meshes(out_dir, groups["main"], groups["detail"], args)
    with (out_dir / "stabilization_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[done] {out_dir}")
    print(json.dumps({"counts": report["counts"], "ratios": report["ratios"]}, indent=2))


if __name__ == "__main__":
    main()
