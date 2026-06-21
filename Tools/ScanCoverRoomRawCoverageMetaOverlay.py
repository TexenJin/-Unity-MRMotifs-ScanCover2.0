#!/usr/bin/env python3
"""Compare room-scale raw-depth coverage voxels against Meta Scene Mesh.

This is a direct diagnostic for the current architecture question:

- Raw Depth coverage is local/noisy but close to observed depth.
- Meta Scene Mesh is room-scale/smooth but can be offset.
- Their difference is a correction/residual field, not something to blindly
  erase in the live pipeline.

Outputs are intended for CloudCompare:

- raw_coverage_all_by_meta_distance.ply
- raw_coverage_stable_by_meta_distance.ply
- raw_coverage_risk_by_meta_distance.ply
- raw_coverage_snapped_to_meta.ply
- fusion_stable_surface.ply
- fusion_corrected_surface_raw.ply
- fusion_risk_boundary.ply
- fusion_meta_only_gaps.ply
- fusion_candidate_meta_guided_raw.ply
- meta_reference_sample.ply
- meta_reference_by_raw_distance.ply
- raw_meta_delta_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d


DEFAULT_PROJECT = Path(r"D:\PCA\Unity-MRMotifs-ScanCover-main")
DEFAULT_META = (
    DEFAULT_PROJECT
    / "ScanCoverExports"
    / "MetaSceneMeshAuditSessions"
    / "ScanCover_MetaSceneMeshAudit_20260611_180459_512"
    / "stage0_weld"
    / "meta_scene_mesh_aligned_all_welded_eps1e-05.ply"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay room raw coverage voxels with Meta Scene Mesh.")
    parser.add_argument("repeat_session", type=Path, help="RepeatCoverage session folder.")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META, help="Welded Meta Scene Mesh PLY/OBJ.")
    parser.add_argument("--out", type=Path, default=None, help="Output folder.")
    parser.add_argument("--meta-sample-points", type=int, default=400000)
    parser.add_argument("--auto-align", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--align-voxel", type=float, default=0.05)
    parser.add_argument("--align-max-distance", type=float, default=0.35)
    parser.add_argument("--close-threshold", type=float, default=0.06, help="Close raw/meta distance in meters.")
    parser.add_argument("--usable-threshold", type=float, default=0.12, help="Usable raw/meta distance in meters.")
    parser.add_argument("--far-threshold", type=float, default=0.20, help="Far raw/meta distance in meters.")
    parser.add_argument("--meta-gap-threshold", type=float, default=0.16, help="Meta sample distance to nearest Raw above this is treated as Meta-only coverage.")
    parser.add_argument("--correction-alpha", type=float, default=0.25, help="Diagnostic blend strength from usable Raw points toward the nearest Meta plane.")
    return parser.parse_args()


def normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.reshape((-1, 3))
    lengths = np.linalg.norm(vectors, axis=1)
    safe = np.where(lengths > 1e-8, lengths, 1.0)
    return vectors / safe[:, None]


def distribution(values: Iterable[float]) -> dict[str, object]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
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


def read_room_voxels(path: Path) -> dict[str, np.ndarray]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.startswith("voxelX,")), None)
    if header_index is None:
        raise RuntimeError(f"Could not find voxel header in {path}")

    points: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    frame_hits: list[int] = []
    point_hits: list[int] = []
    stable: list[bool] = []
    risk: list[bool] = []

    for row in csv.DictReader(lines[header_index:]):
        points.append((float(row["avgX"]), float(row["avgY"]), float(row["avgZ"])))
        normals.append((float(row["avgNormalX"]), float(row["avgNormalY"]), float(row["avgNormalZ"])))
        frame_hits.append(int(row["frameHits"]))
        point_hits.append(int(row["pointHits"]))
        stable.append(row["stable"].strip() == "1")
        risk.append(row["risk"].strip() == "1")

    return {
        "points": np.asarray(points, dtype=np.float64),
        "normals": normalize(np.asarray(normals, dtype=np.float64)),
        "frame_hits": np.asarray(frame_hits, dtype=np.int32),
        "point_hits": np.asarray(point_hits, dtype=np.int32),
        "stable": np.asarray(stable, dtype=bool),
        "risk": np.asarray(risk, dtype=bool),
    }


def load_meta_sample(path: Path, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)

    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        cloud = mesh.sample_points_uniformly(number_of_points=sample_count)
        cloud.estimate_normals()
        return np.asarray(cloud.points, dtype=np.float64), normalize(np.asarray(cloud.normals, dtype=np.float64))

    cloud = o3d.io.read_point_cloud(str(path))
    if len(cloud.points) == 0:
        raise RuntimeError(f"Could not read Meta mesh/cloud: {path}")
    if not cloud.has_normals():
        cloud.estimate_normals()
    points = np.asarray(cloud.points, dtype=np.float64)
    normals = normalize(np.asarray(cloud.normals, dtype=np.float64))
    if len(points) > sample_count:
        step = max(1, int(math.ceil(len(points) / sample_count)))
        points = points[::step][:sample_count]
        normals = normals[::step][:sample_count]
    return points, normals


def nearest_meta(raw_points: np.ndarray, meta_points: np.ndarray, meta_normals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(meta_points)
    tree = o3d.geometry.KDTreeFlann(cloud)

    nearest_points = np.empty_like(raw_points)
    nearest_normals = np.empty_like(raw_points)
    distances = np.empty((len(raw_points),), dtype=np.float64)
    signed = np.empty((len(raw_points),), dtype=np.float64)

    for i, point in enumerate(raw_points):
        _, idx, d2 = tree.search_knn_vector_3d(point, 1)
        j = int(idx[0])
        nearest = meta_points[j]
        normal = meta_normals[j] if j < len(meta_normals) else np.array((0.0, 1.0, 0.0), dtype=np.float64)
        delta = point - nearest
        nearest_points[i] = nearest
        nearest_normals[i] = normal
        distances[i] = math.sqrt(float(d2[0]))
        signed[i] = float(np.dot(delta, normal))
    return nearest_points, nearest_normals, distances, signed


def nearest_reference(query_points: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(reference_points)
    tree = o3d.geometry.KDTreeFlann(cloud)

    distances = np.empty((len(query_points),), dtype=np.float64)
    for i, point in enumerate(query_points):
        _, _, d2 = tree.search_knn_vector_3d(point, 1)
        distances[i] = math.sqrt(float(d2[0]))
    return distances


def distance_colors(distances: np.ndarray, close: float, usable: float, far: float) -> np.ndarray:
    colors = np.zeros((len(distances), 3), dtype=np.float64)
    for i, d in enumerate(distances):
        if d <= close:
            colors[i] = (0.0, 1.0, 0.15)  # green: raw and Meta already agree
        elif d <= usable:
            colors[i] = (0.0, 0.75, 1.0)  # cyan: usable correction zone
        elif d <= far:
            colors[i] = (1.0, 0.85, 0.0)  # yellow: biased, inspect
        else:
            colors[i] = (1.0, 0.05, 0.05)  # red: mismatch/outlier
    return colors


def write_cloud(path: Path, points: np.ndarray, colors: np.ndarray | None = None, normals: np.ndarray | None = None) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.reshape((-1, 3)))
    if colors is not None and len(colors) == len(points):
        cloud.colors = o3d.utility.Vector3dVector(colors.reshape((-1, 3)))
    if normals is not None and len(normals) == len(points):
        cloud.normals = o3d.utility.Vector3dVector(normals.reshape((-1, 3)))
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False)


def make_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.reshape((-1, 3)))
    return cloud


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    homo = np.ones((len(points), 4), dtype=np.float64)
    homo[:, :3] = points
    return (homo @ transform.T)[:, :3]


def auto_align_raw_to_meta(
    raw_points: np.ndarray,
    stable_mask: np.ndarray,
    meta_points: np.ndarray,
    voxel_size: float,
    max_distance: float,
) -> tuple[np.ndarray, dict[str, object]]:
    source_points = raw_points[stable_mask] if np.count_nonzero(stable_mask) >= 100 else raw_points
    source_cloud = make_cloud(source_points)
    target_cloud = make_cloud(meta_points)
    if voxel_size > 0:
        source_cloud = source_cloud.voxel_down_sample(voxel_size)
        target_cloud = target_cloud.voxel_down_sample(voxel_size)

    source_down = np.asarray(source_cloud.points, dtype=np.float64)
    target_down = np.asarray(target_cloud.points, dtype=np.float64)
    if len(source_down) == 0 or len(target_down) == 0:
        return np.eye(4, dtype=np.float64), {"enabled": False, "reason": "empty_cloud"}

    initial = np.eye(4, dtype=np.float64)
    initial[:3, 3] = target_down.mean(axis=0) - source_down.mean(axis=0)
    result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        max_distance,
        initial,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
    )
    transform = np.asarray(result.transformation, dtype=np.float64)
    return transform, {
        "enabled": True,
        "source": "stable raw voxels" if np.count_nonzero(stable_mask) >= 100 else "all raw voxels",
        "sourceDownsampledPoints": int(len(source_down)),
        "targetDownsampledPoints": int(len(target_down)),
        "voxelSizeMeters": voxel_size,
        "maxCorrespondenceDistanceMeters": max_distance,
        "initialTranslation": initial[:3, 3].tolist(),
        "fitness": float(result.fitness),
        "inlierRmse": float(result.inlier_rmse),
        "transformRawToMeta": transform.tolist(),
    }


def mask_stats(name: str, mask: np.ndarray, distances: np.ndarray, signed: np.ndarray) -> dict[str, object]:
    return {
        "name": name,
        "count": int(np.count_nonzero(mask)),
        "unsignedDistanceMeters": distribution(distances[mask]),
        "signedDistanceMetersByNearestMetaNormal": distribution(signed[mask]),
        "positiveSignedCount": int(np.count_nonzero(signed[mask] > 0.0)),
        "negativeSignedCount": int(np.count_nonzero(signed[mask] < 0.0)),
    }


def main() -> int:
    args = parse_args()
    session = args.repeat_session.resolve()
    room_dir = session / "room_raw_coverage"
    voxels_csv = room_dir / "room_raw_coverage_voxels.csv"
    if not voxels_csv.exists():
        raise FileNotFoundError(voxels_csv)

    out_dir = args.out.resolve() if args.out else session / "room_raw_meta_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_room_voxels(voxels_csv)
    points = raw["points"]
    normals = raw["normals"]
    stable = raw["stable"]
    risk = raw["risk"]

    meta_points, meta_normals = load_meta_sample(args.meta.resolve(), args.meta_sample_points)
    alignment: dict[str, object] = {"enabled": False}
    if args.auto_align:
        transform, alignment = auto_align_raw_to_meta(
            points,
            stable,
            meta_points,
            args.align_voxel,
            args.align_max_distance,
        )
        points = apply_transform(points, transform)
        normals = normalize((normals @ transform[:3, :3].T))

    nearest_points, nearest_normals, distances, signed = nearest_meta(points, meta_points, meta_normals)
    snapped_points = points - signed[:, None] * nearest_normals
    meta_to_raw_distances = nearest_reference(meta_points, points)

    colors = distance_colors(distances, args.close_threshold, args.usable_threshold, args.far_threshold)
    meta_colors = distance_colors(meta_to_raw_distances, args.close_threshold, args.usable_threshold, args.far_threshold)
    write_cloud(out_dir / "raw_coverage_all_by_meta_distance.ply", points, colors, normals)
    write_cloud(out_dir / "raw_coverage_stable_by_meta_distance.ply", points[stable], colors[stable], normals[stable])
    write_cloud(out_dir / "raw_coverage_risk_by_meta_distance.ply", points[risk], colors[risk], normals[risk])
    write_cloud(out_dir / "raw_coverage_snapped_to_meta.ply", snapped_points, colors, nearest_normals)
    write_cloud(out_dir / "meta_reference_sample.ply", meta_points, np.tile((0.8, 0.8, 0.8), (len(meta_points), 1)), meta_normals)
    write_cloud(out_dir / "meta_reference_by_raw_distance.ply", meta_points, meta_colors, meta_normals)

    close = distances <= args.close_threshold
    usable = distances <= args.usable_threshold
    far = distances > args.far_threshold
    stable_surface = stable & ~risk & close
    correction_surface = stable & ~risk & ~close & usable
    risk_boundary = risk | far
    meta_only = meta_to_raw_distances > args.meta_gap_threshold

    alpha = max(0.0, min(1.0, args.correction_alpha))
    corrected_points = points.copy()
    corrected_points[correction_surface] = (
        points[correction_surface]
        - (signed[correction_surface] * alpha)[:, None] * nearest_normals[correction_surface]
    )
    candidate_mask = stable_surface | correction_surface
    candidate_colors = np.zeros((np.count_nonzero(candidate_mask), 3), dtype=np.float64)
    candidate_colors[: np.count_nonzero(stable_surface[candidate_mask])] = (0.05, 1.0, 0.25)
    candidate_colors[np.count_nonzero(stable_surface[candidate_mask]) :] = (0.0, 0.8, 1.0)

    write_cloud(
        out_dir / "fusion_stable_surface.ply",
        points[stable_surface],
        np.tile((0.05, 1.0, 0.25), (np.count_nonzero(stable_surface), 1)),
        normals[stable_surface],
    )
    write_cloud(
        out_dir / "fusion_corrected_surface_raw.ply",
        corrected_points[correction_surface],
        np.tile((0.0, 0.8, 1.0), (np.count_nonzero(correction_surface), 1)),
        normals[correction_surface],
    )
    write_cloud(
        out_dir / "fusion_risk_boundary.ply",
        points[risk_boundary],
        np.tile((1.0, 0.08, 0.08), (np.count_nonzero(risk_boundary), 1)),
        normals[risk_boundary],
    )
    write_cloud(
        out_dir / "fusion_meta_only_gaps.ply",
        meta_points[meta_only],
        np.tile((1.0, 0.75, 0.0), (np.count_nonzero(meta_only), 1)),
        meta_normals[meta_only],
    )
    write_cloud(
        out_dir / "fusion_candidate_meta_guided_raw.ply",
        corrected_points[candidate_mask],
        colors[candidate_mask],
        normals[candidate_mask],
    )

    report = {
        "repeatSession": str(session),
        "roomVoxelCsv": str(voxels_csv),
        "metaReference": str(args.meta.resolve()),
        "outputDirectory": str(out_dir),
        "rawVoxelCount": int(len(points)),
        "stableVoxelCount": int(np.count_nonzero(stable)),
        "riskVoxelCount": int(np.count_nonzero(risk)),
        "thresholdsMeters": {
            "close": args.close_threshold,
            "usable": args.usable_threshold,
            "far": args.far_threshold,
            "metaGap": args.meta_gap_threshold,
            "correctionAlpha": alpha,
        },
        "alignment": alignment,
        "all": mask_stats("all", np.ones((len(points),), dtype=bool), distances, signed),
        "stable": mask_stats("stable", stable, distances, signed),
        "risk": mask_stats("risk", risk, distances, signed),
        "closeAgreement": mask_stats("closeAgreement", close, distances, signed),
        "usableCorrectionZone": mask_stats("usableCorrectionZone", usable, distances, signed),
        "farMismatch": mask_stats("farMismatch", far, distances, signed),
        "fusionStableSurface": mask_stats("fusionStableSurface", stable_surface, distances, signed),
        "fusionCorrectedSurfaceRaw": mask_stats("fusionCorrectedSurfaceRaw", correction_surface, distances, signed),
        "fusionRiskBoundary": mask_stats("fusionRiskBoundary", risk_boundary, distances, signed),
        "metaToRawDistanceMeters": distribution(meta_to_raw_distances),
        "fusionMetaOnlyGaps": {
            "count": int(np.count_nonzero(meta_only)),
            "ratio": float(np.count_nonzero(meta_only) / max(1, len(meta_points))),
            "distanceMeters": distribution(meta_to_raw_distances[meta_only]),
        },
        "interpretation": [
            "Green points are already close to Meta Scene Mesh.",
            "Cyan/yellow points are the useful correction field: Raw Depth observes them, Meta provides a smooth structural target.",
            "Red points are too far from Meta and should be treated as mismatch/outlier before any correction is applied.",
            "raw_coverage_snapped_to_meta.ply is diagnostic only. It shows what blindly removing the signed Meta distance would look like.",
            "fusion_candidate_meta_guided_raw.ply is also diagnostic. It keeps close Raw points unchanged and only nudges usable Raw points partway toward the nearest Meta plane.",
            "fusion_meta_only_gaps.ply shows Meta structure that lacks nearby Raw coverage; this is a coverage/completion hint, not observed true depth.",
        ],
    }
    (out_dir / "raw_meta_delta_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({
        "rawVoxelCount": report["rawVoxelCount"],
        "stableVoxelCount": report["stableVoxelCount"],
        "riskVoxelCount": report["riskVoxelCount"],
        "allDistanceMedian": report["all"]["unsignedDistanceMeters"].get("median"),
        "allDistanceP90": report["all"]["unsignedDistanceMeters"].get("p90"),
        "stableDistanceMedian": report["stable"]["unsignedDistanceMeters"].get("median"),
        "stableDistanceP90": report["stable"]["unsignedDistanceMeters"].get("p90"),
        "outputDirectory": str(out_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
