#!/usr/bin/env python3
"""Offline comparison for a self-contained ScanCover replay session.

The Meta room mesh is an external reference, not ground truth.  One rigid
registration is estimated from post-QRS depth and then reused unchanged for
platform depth, GunGel candidates, and the published paper front.
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
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare replay geometry with an external system room mesh.")
    parser.add_argument("session", type=Path)
    parser.add_argument("--system-mesh", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--pixel-stride", type=int, default=6)
    parser.add_argument("--system-samples", type=int, default=400_000)
    parser.add_argument("--paper-samples", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=260825)
    return parser.parse_args()


def recorded_eye(record: dict) -> int:
    eye = int(record.get("recordedEyeIndex", 1))
    return eye if eye in (0, 1) else 1


def linearize(depth: np.ndarray, projection: list[float]) -> np.ndarray:
    z = depth.astype(np.float64) * 2.0 - 1.0
    return np.abs(float(projection[11]) / (z + float(projection[10])))


def backproject_samples(record: dict, depth: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    width = int(record["width"])
    height = int(record["height"])
    image = depth.reshape(height, width)
    ys = np.arange(0, height, stride, dtype=np.int32)
    xs = np.arange(0, width, stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    sampled = image[grid_y, grid_x]
    valid = np.isfinite(sampled) & (sampled > 0.0) & (sampled < 1.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    u = grid_x[valid].astype(np.float64) / width
    v = grid_y[valid].astype(np.float64) / height
    ndc = sampled[valid].astype(np.float64)
    hcs = np.stack((u * 2.0 - 1.0, v * 2.0 - 1.0, ndc * 2.0 - 1.0, np.ones_like(u)))
    eye_index = recorded_eye(record)
    projection_inverse = np.asarray(record["projectionInverse"][eye_index], dtype=np.float64).reshape(4, 4)
    view_inverse = np.asarray(record["viewInverse"][eye_index], dtype=np.float64).reshape(4, 4)
    world_h = view_inverse @ (projection_inverse @ hcs)
    world = (world_h[:3] / world_h[3]).T
    eye = np.asarray(record["trackingPoses"][eye_index]["position"], dtype=np.float64)
    distance = np.linalg.norm(world - eye, axis=1)
    keep = np.isfinite(world).all(axis=1) & (distance >= 0.15) & (distance <= 8.0)

    # Elliptical normalized field position: 0 is optical center, about 1 is a corner.
    nx = (grid_x[valid].astype(np.float64) + 0.5 - width * 0.5) / (width * 0.5)
    ny = (grid_y[valid].astype(np.float64) + 0.5 - height * 0.5) / (height * 0.5)
    radius = np.sqrt((nx * nx + ny * ny) * 0.5)
    return world[keep].astype(np.float32), radius[keep].astype(np.float32)


def load_depth_samples(session: Path, frame_stride: int, pixel_stride: int) -> dict[str, object]:
    root = session / "depth_pairs"
    frames = root / "frames"
    rows = list(csv.DictReader((root / "manifest.csv").open("r", encoding="utf-8-sig", newline="")))
    rows = [row for row in rows if row.get("status") == "ok"]
    selected = rows[:: max(frame_stride, 1)]

    raw_points: list[np.ndarray] = []
    post_points: list[np.ndarray] = []
    raw_radius: list[np.ndarray] = []
    post_radius: list[np.ndarray] = []
    raw_motion: list[np.ndarray] = []
    post_motion: list[np.ndarray] = []
    diff_metres: list[np.ndarray] = []
    raw_valid = post_valid = removed = introduced = 0

    for row in selected:
        meta = json.loads((frames / row["metadataFile"]).read_text(encoding="utf-8-sig"))
        count = int(meta["width"]) * int(meta["height"]) * int(meta["layers"])
        raw = np.fromfile(frames / row["rawFile"], dtype="<f4", count=count)
        post = np.fromfile(frames / row["processedFile"], dtype="<f4", count=count)
        if len(raw) != count or len(post) != count:
            raise RuntimeError(f"Short depth payload at pair {row['pairIndex']}")

        raw_ok = np.isfinite(raw) & (raw > 0.0) & (raw < 1.0)
        post_ok = np.isfinite(post) & (post > 0.0) & (post < 1.0)
        common = raw_ok & post_ok
        raw_valid += int(np.count_nonzero(raw_ok))
        post_valid += int(np.count_nonzero(post_ok))
        removed += int(np.count_nonzero(raw_ok & ~post_ok))
        introduced += int(np.count_nonzero(~raw_ok & post_ok))
        if np.any(common):
            projection = meta["projection"][recorded_eye(meta)]
            raw_m = linearize(raw[common], projection)
            post_m = linearize(post[common], projection)
            usable = np.isfinite(raw_m) & np.isfinite(post_m) & (raw_m <= 10.0) & (post_m <= 10.0)
            diff_metres.append(np.abs(raw_m[usable] - post_m[usable]).astype(np.float32))

        angular = float(meta.get("angularDegPerSec", 0.0))
        linear = float(meta.get("linearMps", 0.0))
        motion_class = 2 if angular > 45.0 or linear > 0.20 else (1 if angular > 10.0 or linear > 0.05 else 0)
        raw_xyz, raw_r = backproject_samples(meta, raw, pixel_stride)
        post_xyz, post_r = backproject_samples(meta, post, pixel_stride)
        raw_points.append(raw_xyz)
        post_points.append(post_xyz)
        raw_radius.append(raw_r)
        post_radius.append(post_r)
        raw_motion.append(np.full(len(raw_xyz), motion_class, dtype=np.uint8))
        post_motion.append(np.full(len(post_xyz), motion_class, dtype=np.uint8))

    diffs = np.concatenate(diff_metres) if diff_metres else np.empty((0,), dtype=np.float32)
    filter_summary = {
        "totalFrames": len(rows),
        "sampledFrames": len(selected),
        "frameStride": frame_stride,
        "pixelStride": pixel_stride,
        "rawValidPixels": raw_valid,
        "postValidPixels": post_valid,
        "rawValidRemovedPct": removed * 100.0 / max(raw_valid, 1),
        "postValidIntroducedPct": introduced * 100.0 / max(post_valid, 1),
        "absRawPostDiffMm": distribution(diffs * 1000.0),
    }
    return {
        "raw": np.concatenate(raw_points) if raw_points else np.empty((0, 3), dtype=np.float32),
        "post": np.concatenate(post_points) if post_points else np.empty((0, 3), dtype=np.float32),
        "rawRadius": np.concatenate(raw_radius) if raw_radius else np.empty((0,), dtype=np.float32),
        "postRadius": np.concatenate(post_radius) if post_radius else np.empty((0,), dtype=np.float32),
        "rawMotion": np.concatenate(raw_motion) if raw_motion else np.empty((0,), dtype=np.uint8),
        "postMotion": np.concatenate(post_motion) if post_motion else np.empty((0,), dtype=np.uint8),
        "filterSummary": filter_summary,
    }


def normalize(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(lengths, 1e-12)


def load_mesh_sample(path: Path, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(f"No triangle mesh in {path}")
    mesh.compute_triangle_normals()
    cloud = mesh.sample_points_uniformly(number_of_points=sample_count, use_triangle_normal=True)
    points = np.asarray(cloud.points, dtype=np.float64)
    normals = normalize(np.asarray(cloud.normals, dtype=np.float64))
    return points, normals


def load_paper_sample(path: Path, sample_count: int) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        cloud = mesh.sample_points_uniformly(number_of_points=sample_count)
        return np.asarray(cloud.points, dtype=np.float64)
    cloud = o3d.io.read_point_cloud(str(path))
    if len(cloud.points) == 0:
        raise RuntimeError(f"No paper geometry in {path}")
    return np.asarray(cloud.points, dtype=np.float64)


def load_gungel_candidates(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    points: list[tuple[float, float, float]] = []
    stable: list[bool] = []
    metric_names = (
        "observation_count",
        "sigma_mm",
        "effective_support",
        "dual_agree_support",
        "opposition_votes",
        "view_spread_deg",
    )
    metrics: dict[str, list[float]] = {name: [] for name in metric_names}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            points.append((float(row["center_x_m"]), float(row["center_y_m"]), float(row["center_z_m"])))
            stable.append(row.get("state", "").strip().lower() == "stable")
            for name in metric_names:
                metrics[name].append(float(row[name]))
    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(stable, dtype=bool),
        {name: np.asarray(values, dtype=np.float64) for name, values in metrics.items()},
    )


def candidate_quality_by_paper(
    candidate_to_paper: np.ndarray,
    stable_mask: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> list[dict]:
    stable_metrics = {name: values[stable_mask] for name, values in metrics.items()}
    groups = {
        "paper_within_10cm": candidate_to_paper <= 0.10,
        "paper_10_to_20cm": (candidate_to_paper > 0.10) & (candidate_to_paper <= 0.20),
        "paper_over_20cm": candidate_to_paper > 0.20,
    }
    result: list[dict] = []
    for name, mask in groups.items():
        result.append(
            {
                "name": name,
                "count": int(np.count_nonzero(mask)),
                "sharePct": float(np.mean(mask) * 100.0),
                "metrics": {
                    metric_name: distribution(values[mask])
                    for metric_name, values in stable_metrics.items()
                },
            }
        )
    return result


def make_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.reshape((-1, 3)))
    return cloud


def yaw_transform(angle_deg: float, source_center: np.ndarray, target_center: np.ndarray) -> np.ndarray:
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)), dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = target_center - rotation @ source_center
    return result


def estimate_registration(source: np.ndarray, target: np.ndarray, voxel: float = 0.08) -> tuple[np.ndarray, dict]:
    source_cloud = make_cloud(source).voxel_down_sample(voxel)
    target_cloud = make_cloud(target).voxel_down_sample(voxel)
    target_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3.0, max_nn=60))
    source_down = np.asarray(source_cloud.points)
    target_down = np.asarray(target_cloud.points)
    if len(source_down) < 100 or len(target_down) < 100:
        raise RuntimeError("Not enough points for registration")
    source_center = np.median(source_down, axis=0)
    target_center = np.median(target_down, axis=0)

    hypotheses: list[tuple[float, float, float, np.ndarray]] = []
    for yaw in np.arange(0.0, 360.0, 30.0):
        initial = yaw_transform(float(yaw), source_center, target_center)
        result = o3d.pipelines.registration.registration_icp(
            source_cloud,
            target_cloud,
            0.45,
            initial,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=70),
        )
        hypotheses.append((float(result.fitness), float(result.inlier_rmse), float(yaw), np.asarray(result.transformation)))

    hypotheses.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    coarse = hypotheses[0]
    fine = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        0.18,
        coarse[3],
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    transform = np.asarray(fine.transformation, dtype=np.float64)
    return transform, {
        "method": "12 yaw hypotheses plus point-to-plane ICP",
        "source": "post-QRS replay depth",
        "rigidOnly": True,
        "voxelSizeMeters": voxel,
        "coarseMaxDistanceMeters": 0.45,
        "fineMaxDistanceMeters": 0.18,
        "coarseBestYawDegrees": coarse[2],
        "coarseFitness": coarse[0],
        "coarseInlierRmseMeters": coarse[1],
        "fineFitness": float(fine.fitness),
        "fineInlierRmseMeters": float(fine.inlier_rmse),
        "transformReplayToSystem": transform.tolist(),
        "hypotheses": [
            {"yawDegrees": h[2], "fitness": h[0], "inlierRmseMeters": h[1]}
            for h in hypotheses
        ],
    }


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.astype(np.float64)
    homogeneous = np.ones((len(points), 4), dtype=np.float64)
    homogeneous[:, :3] = points
    return (homogeneous @ transform.T)[:, :3]


def distribution(values: Iterable[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None, "mean": None}
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def compare_cloud(name: str, points: np.ndarray, system_points: np.ndarray, system_normals: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    system_tree = cKDTree(system_points)
    distances, indices = system_tree.query(points, k=1, workers=-1)
    nearest = system_points[indices]
    signed = np.einsum("ij,ij->i", points - nearest, system_normals[indices])
    source_tree = cKDTree(points)
    reverse, _ = source_tree.query(system_points, k=1, workers=-1)
    return {
        "name": name,
        "pointCount": int(len(points)),
        "sourceToSystemMeters": distribution(distances),
        "sourceSignedBySystemNormalMeters": distribution(signed),
        "sourceWithin2cmPct": float(np.mean(distances <= 0.02) * 100.0),
        "sourceWithin5cmPct": float(np.mean(distances <= 0.05) * 100.0),
        "sourceWithin10cmPct": float(np.mean(distances <= 0.10) * 100.0),
        "sourceOver20cmPct": float(np.mean(distances > 0.20) * 100.0),
        "systemToSourceMeters": distribution(reverse),
        "systemCoveredWithin5cmPct": float(np.mean(reverse <= 0.05) * 100.0),
        "systemCoveredWithin10cmPct": float(np.mean(reverse <= 0.10) * 100.0),
        "systemGapOver20cmPct": float(np.mean(reverse > 0.20) * 100.0),
    }, distances, reverse


def distance_colors(distances: np.ndarray) -> np.ndarray:
    colors = np.empty((len(distances), 3), dtype=np.float64)
    colors[distances <= 0.02] = (0.0, 1.0, 0.15)
    colors[(distances > 0.02) & (distances <= 0.05)] = (0.0, 0.75, 1.0)
    colors[(distances > 0.05) & (distances <= 0.10)] = (1.0, 0.85, 0.0)
    colors[distances > 0.10] = (1.0, 0.05, 0.05)
    return colors


def responsibility_chain(
    system_points: np.ndarray,
    post_reverse: np.ndarray,
    candidate_reverse: np.ndarray,
    paper_reverse: np.ndarray,
    threshold: float = 0.10,
) -> tuple[dict, np.ndarray]:
    """Partition every sampled system point into exactly one downstream stage."""
    paper = paper_reverse <= threshold
    candidate = candidate_reverse <= threshold
    depth = post_reverse <= threshold
    masks = {
        "paper_published": paper,
        "paper_missing_despite_stable_candidate": ~paper & candidate,
        "stable_candidate_missing_despite_post_qrs_depth": ~paper & ~candidate & depth,
        "post_qrs_depth_absent_or_external_reference_disagrees": ~paper & ~candidate & ~depth,
    }
    palette = {
        "paper_published": (0.0, 0.75, 1.0),
        "paper_missing_despite_stable_candidate": (1.0, 0.05, 0.05),
        "stable_candidate_missing_despite_post_qrs_depth": (1.0, 0.65, 0.0),
        "post_qrs_depth_absent_or_external_reference_disagrees": (0.45, 0.45, 0.45),
    }
    colors = np.zeros((len(system_points), 3), dtype=np.float64)
    buckets: list[dict] = []
    for name, mask in masks.items():
        colors[mask] = palette[name]
        buckets.append(
            {
                "name": name,
                "count": int(np.count_nonzero(mask)),
                "sharePct": float(np.mean(mask) * 100.0),
            }
        )
    if sum(bucket["count"] for bucket in buckets) != len(system_points):
        raise RuntimeError("Responsibility chain did not partition system points exactly")
    return {
        "grain": "uniformly sampled external system mesh surface points",
        "pointCount": int(len(system_points)),
        "thresholdMeters": threshold,
        "mutuallyExclusive": True,
        "buckets": buckets,
    }, colors


def paper_hold_reason_summary(path: Path) -> dict:
    names = {
        0: "none",
        1: "supported_open",
        2: "unobserved",
        3: "not_contradicted",
        4: "incoherent_refine",
        5: "ambiguous_empty",
        6: "cross_layer_candidate",
        7: "sliver_candidate",
        8: "unsupported_face",
    }
    counts: dict[int, int] = {}
    owned_counts: dict[int, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            reason = int(row["hold_reason"])
            counts[reason] = counts.get(reason, 0) + 1
            if int(row["owned"]) != 0:
                owned_counts[reason] = owned_counts.get(reason, 0) + 1
    total = sum(counts.values())
    owned_total = sum(owned_counts.values())
    return {
        "allTopologyCells": [
            {
                "code": code,
                "name": names.get(code, f"unknown_{code}"),
                "count": count,
                "sharePct": count * 100.0 / max(total, 1),
            }
            for code, count in sorted(counts.items())
        ],
        "ownedPaperCells": [
            {
                "code": code,
                "name": names.get(code, f"unknown_{code}"),
                "count": count,
                "sharePct": count * 100.0 / max(owned_total, 1),
            }
            for code, count in sorted(owned_counts.items())
        ],
        "topologyCellCount": total,
        "ownedCellCount": owned_total,
    }


def write_cloud(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    cloud = make_cloud(points)
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False):
        raise RuntimeError(f"Failed to write {path}")


def stratified_stats(distances: np.ndarray, radius: np.ndarray, motion: np.ndarray) -> dict:
    fov_masks = {
        "center": radius <= 0.45,
        "middle": (radius > 0.45) & (radius <= 0.75),
        "edge": radius > 0.75,
    }
    motion_masks = {"still": motion == 0, "medium": motion == 1, "fast": motion == 2}

    def one(mask: np.ndarray) -> dict:
        values = distances[mask]
        return {
            "count": int(len(values)),
            "distanceMeters": distribution(values),
            "within5cmPct": float(np.mean(values <= 0.05) * 100.0) if len(values) else None,
            "over20cmPct": float(np.mean(values > 0.20) * 100.0) if len(values) else None,
        }

    return {
        "byFieldPosition": {name: one(mask) for name, mask in fov_masks.items()},
        "byMotion": {name: one(mask) for name, mask in motion_masks.items()},
    }


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# ScanCover offline system-reference comparison",
        "",
        "> The Quest system room mesh is an external comparison reference, not absolute ground truth.",
        "",
        "## Registration",
        "",
        f"- Fine fitness: {report['registration']['fineFitness']:.4f}",
        f"- Fine inlier RMSE: {report['registration']['fineInlierRmseMeters'] * 1000.0:.1f} mm",
        f"- Best coarse yaw: {report['registration']['coarseBestYawDegrees']:.0f} deg",
        "",
        "## Geometry ledger",
        "",
        "| Layer | p50 | p95 | <=5 cm | >20 cm | system coverage <=10 cm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["comparisons"]:
        d = item["sourceToSystemMeters"]
        lines.append(
            f"| {item['name']} | {d['p50'] * 1000.0:.1f} mm | {d['p95'] * 1000.0:.1f} mm | "
            f"{format_pct(item['sourceWithin5cmPct'])} | {format_pct(item['sourceOver20cmPct'])} | "
            f"{format_pct(item['systemCoveredWithin10cmPct'])} |"
        )
    lines.extend((
        "",
        "## Reading boundary",
        "",
        "- A layer-to-system mismatch proves disagreement, not which side is physically correct.",
        "- Because one rigid transform is reused for every ScanCover layer, relative differences between layers are directly comparable.",
        "- Colored PLY files use green <=2 cm, cyan <=5 cm, yellow <=10 cm, red >10 cm.",
        "",
    ))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    session = args.session.resolve()
    system_mesh = args.system_mesh.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    o3d.utility.random.seed(args.seed)

    depth = load_depth_samples(session, args.frame_stride, args.pixel_stride)
    system_points, system_normals = load_mesh_sample(system_mesh, args.system_samples)
    transform, registration = estimate_registration(np.asarray(depth["post"], dtype=np.float64), system_points)

    layers: list[tuple[str, np.ndarray]] = [
        ("platform_pre_qrs", apply_transform(np.asarray(depth["raw"]), transform)),
        ("post_qrs", apply_transform(np.asarray(depth["post"]), transform)),
    ]
    gun_points, gun_stable, gun_metrics = load_gungel_candidates(
        session / "artifacts" / "gungel_candidate_audit" / "candidates.csv"
    )
    layers.append(("gungel_stable_candidates", apply_transform(gun_points[gun_stable], transform)))
    paper_points = load_paper_sample(session / "artifacts" / "paper_audit" / "paper_front.ply", args.paper_samples)
    layers.append(("paper_front", apply_transform(paper_points, transform)))

    comparisons: list[dict] = []
    per_layer_distances: dict[str, np.ndarray] = {}
    per_layer_reverse: dict[str, np.ndarray] = {}
    for name, points in layers:
        stats, distances, reverse = compare_cloud(name, points, system_points, system_normals)
        comparisons.append(stats)
        per_layer_distances[name] = distances
        per_layer_reverse[name] = reverse
        write_cloud(out / f"{name}_aligned_distance.ply", points, distance_colors(distances))
        if name == "paper_front":
            write_cloud(out / "system_reference_paper_gaps.ply", system_points, distance_colors(reverse))

    chain, chain_colors = responsibility_chain(
        system_points,
        per_layer_reverse["post_qrs"],
        per_layer_reverse["gungel_stable_candidates"],
        per_layer_reverse["paper_front"],
    )
    chain_sensitivity = [
        responsibility_chain(
            system_points,
            per_layer_reverse["post_qrs"],
            per_layer_reverse["gungel_stable_candidates"],
            per_layer_reverse["paper_front"],
            threshold=threshold,
        )[0]
        for threshold in (0.05, 0.10, 0.15, 0.20)
    ]
    write_cloud(out / "system_reference_responsibility_chain.ply", system_points, chain_colors)

    candidate_to_paper = cKDTree(layers[-1][1]).query(layers[2][1], k=1, workers=-1)[0]
    stable_candidate_publish = {
        "stableCandidateCount": int(len(candidate_to_paper)),
        "candidateToPaperMeters": distribution(candidate_to_paper),
        "candidateWithPaperWithin5cmPct": float(np.mean(candidate_to_paper <= 0.05) * 100.0),
        "candidateWithPaperWithin10cmPct": float(np.mean(candidate_to_paper <= 0.10) * 100.0),
        "candidateWithoutPaperOver20cmPct": float(np.mean(candidate_to_paper > 0.20) * 100.0),
        "qualityByPaperProximity": candidate_quality_by_paper(
            candidate_to_paper, gun_stable, gun_metrics
        ),
    }

    raw_strata = stratified_stats(
        per_layer_distances["platform_pre_qrs"],
        np.asarray(depth["rawRadius"]),
        np.asarray(depth["rawMotion"]),
    )
    post_strata = stratified_stats(
        per_layer_distances["post_qrs"],
        np.asarray(depth["postRadius"]),
        np.asarray(depth["postMotion"]),
    )

    report = {
        "schema": "scancover.offline_system_reference_compare.v1",
        "session": str(session),
        "systemMesh": str(system_mesh),
        "systemReferenceIsGroundTruth": False,
        "seed": args.seed,
        "depthFilterSummary": depth["filterSummary"],
        "registration": registration,
        "comparisons": comparisons,
        "responsibilityChain": chain,
        "responsibilityChainSensitivity": chain_sensitivity,
        "stableCandidateToPaper": stable_candidate_publish,
        "paperHoldReasons": paper_hold_reason_summary(session / "artifacts" / "paper_audit" / "paper_cells.csv"),
        "platformPreQrsStrata": raw_strata,
        "postQrsStrata": post_strata,
        "artifacts": {
            "alignment": "alignment.json",
            "report": "comparison_report.json",
            "summary": "comparison_summary.md",
            "coloredClouds": [f"{name}_aligned_distance.ply" for name, _ in layers],
            "paperGapCloud": "system_reference_paper_gaps.ply",
            "responsibilityChainCloud": "system_reference_responsibility_chain.ply",
        },
    }
    (out / "alignment.json").write_text(json.dumps(registration, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "comparison_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out / "comparison_summary.md", report)

    print(f"REPORT={out / 'comparison_report.json'}")
    print(f"SUMMARY={out / 'comparison_summary.md'}")
    print(f"REGISTRATION_FITNESS={registration['fineFitness']:.6f}")
    print(f"REGISTRATION_RMSE_MM={registration['fineInlierRmseMeters'] * 1000.0:.3f}")
    for item in comparisons:
        d = item["sourceToSystemMeters"]
        print(
            f"{item['name']}: p50={d['p50'] * 1000.0:.1f}mm "
            f"p95={d['p95'] * 1000.0:.1f}mm <=5cm={item['sourceWithin5cmPct']:.1f}% "
            f"system<=10cm={item['systemCoveredWithin10cmPct']:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
