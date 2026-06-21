#!/usr/bin/env python3
"""Plane-family validation from ScanCover multi-frame observation features.

This is the bridge between Quest3 multi-frame capture and the offline
teacher/student checks. It consumes point_observation_features.csv instead of a
single raw OBJ, so the classifier can use repeat hits, view angle, distance, and
risk statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


PALETTE = np.array(
    [
        [1.0, 0.05, 0.05],
        [0.0, 0.9, 1.0],
        [0.05, 1.0, 0.15],
        [1.0, 0.85, 0.0],
        [1.0, 0.15, 1.0],
        [1.0, 0.45, 0.0],
        [0.15, 0.25, 1.0],
        [0.0, 1.0, 0.65],
    ],
    dtype=np.float64,
)
WEAK_COLOR = np.array([0.85, 0.85, 0.85], dtype=np.float64)
RISK_COLOR = np.array([0.02, 0.02, 0.02], dtype=np.float64)
STABLE_CORE_COLOR = np.array([1.0, 1.0, 1.0], dtype=np.float64)


@dataclass
class FeatureCloud:
    points: np.ndarray
    normals: np.ndarray
    frame_count: np.ndarray
    hit_count: np.ndarray
    mean_distance: np.ndarray
    mean_view_angle: np.ndarray
    position_variance: np.ndarray
    normal_variance: np.ndarray
    any_risk_ratio: np.ndarray
    stability_score: np.ndarray


@dataclass
class PlanePatch:
    patch_id: int
    points: np.ndarray
    normal: np.ndarray
    d: float
    centroid: np.ndarray


@dataclass
class PlaneLayer:
    normal: np.ndarray
    d: float
    centroid: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    uv_min: np.ndarray
    uv_max: np.ndarray
    points: np.ndarray


@dataclass
class PlaneFamily:
    family_id: int
    patch_ids: list[int]
    points: np.ndarray
    normal: np.ndarray
    d: float
    centroid: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    uv_min: np.ndarray
    uv_max: np.ndarray
    color: np.ndarray
    layers: list[PlaneLayer]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "features",
        type=Path,
        help="point_observation_features.csv or its observation_features directory.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--stable-min-frames", type=int, default=3)
    parser.add_argument("--max-distance", type=float, default=5.0)
    parser.add_argument("--max-view-angle", type=float, default=82.0)
    parser.add_argument("--max-risk-ratio", type=float, default=0.55)
    parser.add_argument("--max-position-variance", type=float, default=0.0064)
    parser.add_argument(
        "--teacher-max-view-angle",
        type=float,
        default=72.0,
        help="Stricter view-angle gate for points used to build primary plane families.",
    )
    parser.add_argument(
        "--teacher-max-risk-ratio",
        type=float,
        default=0.22,
        help="Stricter risk gate for points used to build primary plane families.",
    )
    parser.add_argument(
        "--teacher-max-position-variance",
        type=float,
        default=0.0040,
        help="Stricter position-variance gate for points used to build primary plane families.",
    )
    parser.add_argument("--voxel", type=float, default=0.025)
    parser.add_argument("--normal-radius", type=float, default=0.12)
    parser.add_argument("--normal-max-nn", type=int, default=48)
    parser.add_argument("--ransac-distance", type=float, default=0.055)
    parser.add_argument("--ransac-iterations", type=int, default=1400)
    parser.add_argument("--min-inliers", type=int, default=1200)
    parser.add_argument("--max-planes", type=int, default=18)
    parser.add_argument("--family-normal-deg", type=float, default=16.0)
    parser.add_argument("--family-distance", type=float, default=0.14)
    parser.add_argument("--classify-distance", type=float, default=0.09)
    parser.add_argument("--classify-normal-deg", type=float, default=48.0)
    parser.add_argument(
        "--use-structural-consensus-classify",
        action="store_true",
        help="Classify stable points by plane distance, projected coverage, and local neighbor agreement; normals become a soft score only.",
    )
    parser.add_argument("--neighbor-radius", type=float, default=0.10)
    parser.add_argument("--neighbor-min-same", type=int, default=4)
    parser.add_argument("--neighbor-min-ratio", type=float, default=0.45)
    parser.add_argument("--distance-strong-ratio", type=float, default=0.45)
    parser.add_argument("--normal-score-weight", type=float, default=0.03)
    parser.add_argument("--extent-padding", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=15319, help="Deterministic seed for Open3D RANSAC where supported.")
    parser.add_argument(
        "--risk-assign-distance",
        type=float,
        default=0.06,
        help="Distance used only for optional risk-layer family hints after stable families are fixed.",
    )
    parser.add_argument(
        "--risk-assign-normal-deg",
        type=float,
        default=36.0,
        help="Normal angle used only for optional risk-layer family hints after stable families are fixed.",
    )
    parser.add_argument(
        "--fold-parallel-layers",
        action="store_true",
        help="Merge near-parallel, projection-overlapping family layers into one structural family while keeping layer planes for classification.",
    )
    parser.add_argument("--layer-fold-normal-deg", type=float, default=12.0)
    parser.add_argument("--layer-fold-min-distance", type=float, default=0.05)
    parser.add_argument("--layer-fold-max-distance", type=float, default=0.20)
    parser.add_argument("--layer-fold-min-overlap", type=float, default=0.12)
    return parser.parse_args()


def normalize(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length <= 1e-8:
        return v
    return v / length


def resolve_feature_csv(path: Path) -> Path:
    if path.is_dir():
        path = path / "point_observation_features.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_features(path: Path) -> FeatureCloud:
    fields = {
        "meanX": [],
        "meanY": [],
        "meanZ": [],
        "normalX": [],
        "normalY": [],
        "normalZ": [],
        "hit_count": [],
        "frame_count": [],
        "mean_distance": [],
        "mean_view_angle": [],
        "position_variance": [],
        "normal_variance": [],
        "any_risk_ratio": [],
        "stability_score": [],
    }
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in fields:
                fields[key].append(float(row[key]))

    points = np.column_stack([fields["meanX"], fields["meanY"], fields["meanZ"]]).astype(np.float64)
    normals = np.column_stack([fields["normalX"], fields["normalY"], fields["normalZ"]]).astype(np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-8
    normals[valid] = normals[valid] / lengths[valid, None]
    return FeatureCloud(
        points=points,
        normals=normals,
        hit_count=np.asarray(fields["hit_count"], dtype=np.float64),
        frame_count=np.asarray(fields["frame_count"], dtype=np.float64),
        mean_distance=np.asarray(fields["mean_distance"], dtype=np.float64),
        mean_view_angle=np.asarray(fields["mean_view_angle"], dtype=np.float64),
        position_variance=np.asarray(fields["position_variance"], dtype=np.float64),
        normal_variance=np.asarray(fields["normal_variance"], dtype=np.float64),
        any_risk_ratio=np.asarray(fields["any_risk_ratio"], dtype=np.float64),
        stability_score=np.asarray(fields["stability_score"], dtype=np.float64),
    )


def stable_mask(features: FeatureCloud, args: argparse.Namespace) -> np.ndarray:
    return (
        (features.frame_count >= args.stable_min_frames)
        & (features.mean_distance <= args.max_distance)
        & (features.mean_view_angle <= args.max_view_angle)
        & (features.any_risk_ratio <= args.max_risk_ratio)
        & (features.position_variance <= args.max_position_variance)
    )


def teacher_stable_mask(features: FeatureCloud, args: argparse.Namespace) -> np.ndarray:
    """High-confidence stable surface points used to define the main plane skeleton.

    Risk, boundary, oblique, and noisy cells are deliberately kept out of this
    mask. They may be assigned later for diagnostics, but they must not steer
    the primary plane-family fit.
    """
    return (
        (features.frame_count >= args.stable_min_frames)
        & (features.mean_distance <= args.max_distance)
        & (features.mean_view_angle <= min(args.max_view_angle, args.teacher_max_view_angle))
        & (features.any_risk_ratio <= min(args.max_risk_ratio, args.teacher_max_risk_ratio))
        & (features.position_variance <= min(args.max_position_variance, args.teacher_max_position_variance))
    )


def estimate_cloud(points: np.ndarray, args: argparse.Namespace) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if args.voxel > 0:
        cloud = cloud.voxel_down_sample(args.voxel)
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.normal_radius,
            max_nn=args.normal_max_nn,
        )
    )
    try:
        cloud.orient_normals_consistent_tangent_plane(18)
    except RuntimeError:
        pass
    return cloud


def split_planes(cloud: o3d.geometry.PointCloud, args: argparse.Namespace) -> tuple[list[PlanePatch], np.ndarray]:
    remaining = cloud
    patches: list[PlanePatch] = []
    for patch_id in range(args.max_planes):
        if len(remaining.points) < args.min_inliers:
            break
        model, inliers = remaining.segment_plane(
            distance_threshold=args.ransac_distance,
            ransac_n=3,
            num_iterations=args.ransac_iterations,
        )
        if len(inliers) < args.min_inliers:
            break
        patch_cloud = remaining.select_by_index(inliers)
        points = np.asarray(patch_cloud.points, dtype=np.float64)
        normal = normalize(np.asarray(model[:3], dtype=np.float64))
        d = float(model[3])
        centroid = np.mean(points, axis=0)
        if float(np.dot(normal, centroid) + d) < 0:
            normal = -normal
            d = -d
        patches.append(PlanePatch(patch_id, points, normal, d, centroid))
        remaining = remaining.select_by_index(inliers, invert=True)
    return patches, np.asarray(remaining.points, dtype=np.float64)


def fit_family(family_id: int, patch_ids: list[int], points: np.ndarray) -> PlaneFamily:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis_u = normalize(vh[0])
    axis_v = normalize(vh[1])
    normal = normalize(vh[2])
    d = -float(np.dot(normal, centroid))
    signed = points @ normal + d
    if np.median(signed) < 0:
        normal = -normal
        d = -d
    rel = points - centroid
    uv = np.column_stack([rel @ axis_u, rel @ axis_v])
    uv_min = np.min(uv, axis=0)
    uv_max = np.max(uv, axis=0)
    return PlaneFamily(
        family_id=family_id,
        patch_ids=patch_ids,
        points=points,
        normal=normal,
        d=d,
        centroid=centroid,
        axis_u=axis_u,
        axis_v=axis_v,
        uv_min=uv_min,
        uv_max=uv_max,
        color=PALETTE[family_id % len(PALETTE)],
        layers=[
            PlaneLayer(
                normal=normal,
                d=d,
                centroid=centroid,
                axis_u=axis_u,
                axis_v=axis_v,
                uv_min=uv_min,
                uv_max=uv_max,
                points=points,
            )
        ],
    )


def merge_families(patches: list[PlanePatch], args: argparse.Namespace) -> list[PlaneFamily]:
    cos_threshold = math.cos(math.radians(args.family_normal_deg))
    groups: list[list[PlanePatch]] = []
    for patch in sorted(patches, key=lambda p: len(p.points), reverse=True):
        best_group = -1
        best_score = -1.0
        for group_index, group in enumerate(groups):
            family = fit_family(-1, [g.patch_id for g in group], np.vstack([g.points for g in group]))
            normal_score = abs(float(np.dot(patch.normal, family.normal)))
            plane_gap = abs(float(np.dot(family.normal, patch.centroid) + family.d))
            if normal_score >= cos_threshold and plane_gap <= args.family_distance and normal_score > best_score:
                best_group = group_index
                best_score = normal_score
        if best_group >= 0:
            groups[best_group].append(patch)
        else:
            groups.append([patch])
    return [
        fit_family(i, [p.patch_id for p in group], np.vstack([p.points for p in group]))
        for i, group in enumerate(groups)
    ]


def projected_overlap_ratio(a: PlaneFamily, b: PlaneFamily) -> float:
    rel_a = a.points - a.centroid
    rel_b = b.points - a.centroid
    uv_a = np.column_stack([rel_a @ a.axis_u, rel_a @ a.axis_v])
    uv_b = np.column_stack([rel_b @ a.axis_u, rel_b @ a.axis_v])
    a_min = np.min(uv_a, axis=0)
    a_max = np.max(uv_a, axis=0)
    b_min = np.min(uv_b, axis=0)
    b_max = np.max(uv_b, axis=0)
    overlap_min = np.maximum(a_min, b_min)
    overlap_max = np.minimum(a_max, b_max)
    overlap = np.maximum(overlap_max - overlap_min, 0.0)
    overlap_area = float(overlap[0] * overlap[1])
    a_area = float(np.prod(np.maximum(a_max - a_min, 1e-6)))
    b_area = float(np.prod(np.maximum(b_max - b_min, 1e-6)))
    return overlap_area / max(min(a_area, b_area), 1e-6)


def layer_gap(a: PlaneFamily, b: PlaneFamily) -> float:
    return min(
        abs(float(np.dot(a.normal, b.centroid) + a.d)),
        abs(float(np.dot(b.normal, a.centroid) + b.d)),
    )


def fold_parallel_layers(families: list[PlaneFamily], args: argparse.Namespace) -> list[PlaneFamily]:
    if not getattr(args, "fold_parallel_layers", False) or len(families) < 2:
        return families

    cos_threshold = math.cos(math.radians(args.layer_fold_normal_deg))
    groups: list[list[PlaneFamily]] = []
    for family in sorted(families, key=lambda f: len(f.points), reverse=True):
        best_group = -1
        best_overlap = -1.0
        for group_index, group in enumerate(groups):
            representative = max(group, key=lambda f: len(f.points))
            normal_score = abs(float(np.dot(family.normal, representative.normal)))
            gap = layer_gap(family, representative)
            overlap = projected_overlap_ratio(representative, family)
            if (
                normal_score >= cos_threshold
                and args.layer_fold_min_distance <= gap <= args.layer_fold_max_distance
                and overlap >= args.layer_fold_min_overlap
                and overlap > best_overlap
            ):
                best_group = group_index
                best_overlap = overlap
        if best_group >= 0:
            groups[best_group].append(family)
        else:
            groups.append([family])

    folded: list[PlaneFamily] = []
    for family_id, group in enumerate(groups):
        points = np.vstack([f.points for f in group])
        patch_ids = [patch_id for f in group for patch_id in f.patch_ids]
        family = fit_family(family_id, patch_ids, points)
        family.layers = [layer for f in group for layer in f.layers]
        family.color = PALETTE[family_id % len(PALETTE)]
        folded.append(family)
    return folded


def classify(
    points: np.ndarray,
    normals: np.ndarray,
    families: list[PlaneFamily],
    distance: float,
    normal_degrees: float,
    extent_padding: float,
) -> np.ndarray:
    labels = np.full(len(points), -1, dtype=np.int32)
    scores = np.full(len(points), np.inf, dtype=np.float64)
    cos_threshold = math.cos(math.radians(normal_degrees))
    for family in families:
        for layer in family.layers:
            dist = np.abs(points @ layer.normal + layer.d)
            rel = points - layer.centroid
            uv = np.column_stack([rel @ layer.axis_u, rel @ layer.axis_v])
            inside = (
                (uv[:, 0] >= layer.uv_min[0] - extent_padding)
                & (uv[:, 0] <= layer.uv_max[0] + extent_padding)
                & (uv[:, 1] >= layer.uv_min[1] - extent_padding)
                & (uv[:, 1] <= layer.uv_max[1] + extent_padding)
            )
            normal_ok = np.abs(normals @ layer.normal) >= cos_threshold
            candidate = (dist <= distance) & inside & normal_ok
            better = candidate & (dist < scores)
            labels[better] = family.family_id
            scores[better] = dist[better]
    return labels


def structural_consensus_classify(
    points: np.ndarray,
    normals: np.ndarray,
    families: list[PlaneFamily],
    distance: float,
    extent_padding: float,
    args: argparse.Namespace,
) -> np.ndarray:
    candidate_labels = np.full(len(points), -1, dtype=np.int32)
    scores = np.full(len(points), np.inf, dtype=np.float64)
    best_dist = np.full(len(points), np.inf, dtype=np.float64)

    for family in families:
        for layer in family.layers:
            dist = np.abs(points @ layer.normal + layer.d)
            rel = points - layer.centroid
            uv = np.column_stack([rel @ layer.axis_u, rel @ layer.axis_v])
            inside = (
                (uv[:, 0] >= layer.uv_min[0] - extent_padding)
                & (uv[:, 0] <= layer.uv_max[0] + extent_padding)
                & (uv[:, 1] >= layer.uv_min[1] - extent_padding)
                & (uv[:, 1] <= layer.uv_max[1] + extent_padding)
            )
            abs_dot = np.abs(normals @ layer.normal)
            score = dist + args.normal_score_weight * (1.0 - abs_dot) * max(distance, 1e-6)
            candidate = (dist <= distance) & inside
            better = candidate & (score < scores)
            candidate_labels[better] = family.family_id
            scores[better] = score[better]
            best_dist = np.minimum(best_dist, dist)

    labels = np.full(len(points), -1, dtype=np.int32)
    has_candidate = candidate_labels >= 0
    if not np.any(has_candidate):
        return labels

    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    strong = has_candidate & (best_dist <= distance * args.distance_strong_ratio)
    labels[strong] = candidate_labels[strong]

    candidate_indices = np.flatnonzero(has_candidate & ~strong)
    for index in candidate_indices:
        neighbor_indices = tree.query_ball_point(points[index], args.neighbor_radius)
        if not neighbor_indices:
            continue
        neighbor_labels = candidate_labels[np.asarray(neighbor_indices, dtype=np.int64)]
        valid = neighbor_labels >= 0
        valid_count = int(np.count_nonzero(valid))
        if valid_count <= 0:
            continue
        same_count = int(np.count_nonzero(neighbor_labels[valid] == candidate_labels[index]))
        same_ratio = same_count / valid_count
        if same_count >= args.neighbor_min_same and same_ratio >= args.neighbor_min_ratio:
            labels[index] = candidate_labels[index]
    return labels


def write_colored_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)


def write_outputs(
    out_dir: Path,
    features: FeatureCloud,
    families: list[PlaneFamily],
    labels: np.ndarray,
    risk_labels: np.ndarray,
    stable: np.ndarray,
    teacher_stable: np.ndarray,
    args: argparse.Namespace,
    feature_csv: Path,
    leftover_points: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = np.repeat(WEAK_COLOR[None, :], len(features.points), axis=0)
    risk = ~stable
    colors[risk] = RISK_COLOR
    for family in families:
        colors[(labels == family.family_id) & stable] = family.color

    write_colored_cloud(out_dir / "observation_student_all.ply", features.points, colors)
    write_colored_cloud(out_dir / "observation_student_stable_classified.ply", features.points[stable], colors[stable])
    write_colored_cloud(
        out_dir / "observation_student_teacher_core_stable.ply",
        features.points[teacher_stable],
        np.repeat(STABLE_CORE_COLOR[None, :], int(np.count_nonzero(teacher_stable)), axis=0),
    )
    write_colored_cloud(out_dir / "observation_student_risk_layer.ply", features.points[risk], colors[risk])

    risk_hint_colors = np.repeat(RISK_COLOR[None, :], len(features.points), axis=0)
    for family in families:
        risk_hint_colors[(risk_labels == family.family_id) & risk] = family.color
    write_colored_cloud(
        out_dir / "observation_student_risk_layer_family_hints.ply",
        features.points[risk],
        risk_hint_colors[risk],
    )

    for family in families:
        mask = (labels == family.family_id) & stable
        write_colored_cloud(
            out_dir / f"observation_student_family_{family.family_id:02d}_{int(np.count_nonzero(mask))}pts.ply",
            features.points[mask],
            np.repeat(family.color[None, :], int(np.count_nonzero(mask)), axis=0),
        )

    if len(leftover_points) > 0:
        write_colored_cloud(
            out_dir / f"observation_teacher_leftover_{len(leftover_points)}pts.ply",
            leftover_points,
            np.repeat(RISK_COLOR[None, :], len(leftover_points), axis=0),
        )

    with (out_dir / "observation_plane_validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "featureCsv": str(feature_csv),
                "out": str(out_dir),
                "totalPoints": int(len(features.points)),
                "stableInputPoints": int(np.count_nonzero(stable)),
                "teacherStableInputPoints": int(np.count_nonzero(teacher_stable)),
                "riskInputPoints": int(np.count_nonzero(~stable)),
                "planePatches": int(sum(len(f.patch_ids) for f in families)),
                "planeFamilies": int(len(families)),
                "planeFamilyLayers": int(sum(len(f.layers) for f in families)),
                "stableClassifiedPoints": int(np.count_nonzero((labels >= 0) & stable)),
                "stableUnclassifiedPoints": int(np.count_nonzero((labels < 0) & stable)),
                "riskAssignedHintPoints": int(np.count_nonzero((risk_labels >= 0) & (~stable))),
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "families": [
                    {
                        "familyId": f.family_id,
                        "patchIds": f.patch_ids,
                        "layerCount": int(len(f.layers)),
                        "teacherPoints": int(len(f.points)),
                        "normal": [float(x) for x in f.normal],
                        "d": float(f.d),
                        "centroid": [float(x) for x in f.centroid],
                    }
                    for f in families
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    try:
        o3d.utility.random.seed(args.seed)
    except AttributeError:
        pass
    feature_csv = resolve_feature_csv(args.features)
    out_dir = args.out or (feature_csv.parent / "plane_validation")
    features = load_features(feature_csv)
    stable = stable_mask(features, args)
    teacher_stable = teacher_stable_mask(features, args)
    if np.count_nonzero(teacher_stable) < args.min_inliers:
        teacher_stable = stable
    if np.count_nonzero(stable) < args.min_inliers:
        raise RuntimeError(
            f"Too few stable points: {int(np.count_nonzero(stable))}. "
            f"Try lowering --stable-min-frames or widening thresholds."
        )

    teacher_cloud = estimate_cloud(features.points[teacher_stable], args)
    patches, leftover_points = split_planes(teacher_cloud, args)
    if not patches:
        raise RuntimeError("No plane patches found from stable observation points.")
    families = fold_parallel_layers(merge_families(patches, args), args)
    if args.use_structural_consensus_classify:
        labels = structural_consensus_classify(
            features.points,
            features.normals,
            families,
            args.classify_distance,
            args.extent_padding,
            args,
        )
    else:
        labels = classify(
            features.points,
            features.normals,
            families,
            args.classify_distance,
            args.classify_normal_deg,
            args.extent_padding,
        )
    risk_labels = classify(
        features.points,
        features.normals,
        families,
        args.risk_assign_distance,
        args.risk_assign_normal_deg,
        args.extent_padding,
    )
    write_outputs(out_dir, features, families, labels, risk_labels, stable, teacher_stable, args, feature_csv, leftover_points)

    print(f"features={len(features.points)} stable={int(np.count_nonzero(stable))}")
    print(f"teacherStable={int(np.count_nonzero(teacher_stable))}")
    print(f"patches={len(patches)} families={len(families)}")
    print(f"classifiedStable={int(np.count_nonzero((labels >= 0) & stable))}")
    print(f"out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
