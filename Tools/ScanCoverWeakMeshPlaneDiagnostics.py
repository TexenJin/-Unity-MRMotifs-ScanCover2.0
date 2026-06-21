#!/usr/bin/env python3
"""Targeted diagnostics for weak ScanCover plane-family validation cases."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d


def load_plane_validation_module():
    module_path = Path(__file__).with_name("ScanCoverObservationFeaturePlaneValidation.py")
    spec = importlib.util.spec_from_file_location("scan_cover_plane_validation", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path, help="point_observation_features.csv or its directory.")
    parser.add_argument("validation", type=Path, help="Validation output directory with observation_plane_validation_summary.json.")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def namespace_from_summary(summary: dict, feature_csv: Path, out_dir: Path) -> argparse.Namespace:
    raw = dict(summary.get("args", {}))
    raw["features"] = feature_csv
    raw["out"] = out_dir
    numeric_ints = {
        "stable_min_frames",
        "normal_max_nn",
        "ransac_iterations",
        "min_inliers",
        "max_planes",
        "seed",
    }
    numeric_floats = {
        "max_distance",
        "max_view_angle",
        "max_risk_ratio",
        "max_position_variance",
        "teacher_max_view_angle",
        "teacher_max_risk_ratio",
        "teacher_max_position_variance",
        "voxel",
        "normal_radius",
        "ransac_distance",
        "family_normal_deg",
        "family_distance",
        "classify_distance",
        "classify_normal_deg",
        "extent_padding",
        "risk_assign_distance",
        "risk_assign_normal_deg",
    }
    for key in list(raw.keys()):
        if key in numeric_ints:
            raw[key] = int(raw[key])
        elif key in numeric_floats:
            raw[key] = float(raw[key])
    return SimpleNamespace(**raw)


def write_cloud(path: Path, points: np.ndarray, color: np.ndarray) -> None:
    if len(points) == 0:
        return
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(np.repeat(color[None, :], len(points), axis=0))
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)


def classify_with_reasons(points: np.ndarray, normals: np.ndarray, families: list, args: argparse.Namespace):
    labels = np.full(len(points), -1, dtype=np.int32)
    scores = np.full(len(points), np.inf, dtype=np.float64)
    reason = np.full(len(points), "distance", dtype=object)
    best_dist = np.full(len(points), np.inf, dtype=np.float64)
    best_abs_dot = np.zeros(len(points), dtype=np.float64)
    best_inside = np.zeros(len(points), dtype=bool)
    best_inside_normal = np.zeros(len(points), dtype=bool)
    cos_threshold = math.cos(math.radians(args.classify_normal_deg))

    for family in families:
        dist = np.abs(points @ family.normal + family.d)
        best_dist = np.minimum(best_dist, dist)
        rel = points - family.centroid
        uv = np.column_stack([rel @ family.axis_u, rel @ family.axis_v])
        inside = (
            (uv[:, 0] >= family.uv_min[0] - args.extent_padding)
            & (uv[:, 0] <= family.uv_max[0] + args.extent_padding)
            & (uv[:, 1] >= family.uv_min[1] - args.extent_padding)
            & (uv[:, 1] <= family.uv_max[1] + args.extent_padding)
        )
        abs_dot = np.abs(normals @ family.normal)
        normal_ok = abs_dot >= cos_threshold
        close = dist <= args.classify_distance
        best_abs_dot = np.maximum(best_abs_dot, abs_dot)
        best_inside |= close & inside
        best_inside_normal |= close & inside & normal_ok
        candidate = close & inside & normal_ok
        better = candidate & (dist < scores)
        labels[better] = family.family_id
        scores[better] = dist[better]

    unclassified = labels < 0
    reason[unclassified & (best_dist <= args.classify_distance) & ~best_inside] = "extent"
    reason[unclassified & best_inside & ~best_inside_normal] = "normal"
    reason[~unclassified] = "classified"
    return labels, reason, best_dist, best_abs_dot


def analyze_unclassified_candidates(pv, features, mask: np.ndarray, args: argparse.Namespace, out_dir: Path):
    count = int(np.count_nonzero(mask))
    if count < args.min_inliers:
        return []
    candidate_args = SimpleNamespace(**vars(args))
    candidate_args.max_planes = min(8, args.max_planes)
    candidate_args.min_inliers = max(350, min(args.min_inliers, count // 8))
    cloud = pv.estimate_cloud(features.points[mask], candidate_args)
    patches, _ = pv.split_planes(cloud, candidate_args)
    rows = []
    for patch in patches:
        color = pv.PALETTE[patch.patch_id % len(pv.PALETTE)]
        write_cloud(out_dir / f"unclassified_candidate_patch_{patch.patch_id:02d}_{len(patch.points)}pts.ply", patch.points, color)
        rows.append(
            {
                "patchId": patch.patch_id,
                "points": int(len(patch.points)),
                "normal": [float(x) for x in patch.normal],
                "d": float(patch.d),
                "centroid": [float(x) for x in patch.centroid],
            }
        )
    return rows


def main() -> int:
    cli = parse_args()
    pv = load_plane_validation_module()
    feature_csv = pv.resolve_feature_csv(cli.features)
    summary_path = cli.validation / "observation_plane_validation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    out_dir = cli.out or (cli.validation / "weak_plane_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    args = namespace_from_summary(summary, feature_csv, out_dir)
    np.random.seed(args.seed)
    try:
        o3d.utility.random.seed(args.seed)
    except AttributeError:
        pass

    features = pv.load_features(feature_csv)
    stable = pv.stable_mask(features, args)
    teacher_stable = pv.teacher_stable_mask(features, args)
    if np.count_nonzero(teacher_stable) < args.min_inliers:
        teacher_stable = stable

    teacher_cloud = pv.estimate_cloud(features.points[teacher_stable], args)
    patches, leftover_points = pv.split_planes(teacher_cloud, args)
    families = pv.merge_families(patches, args)
    labels, reason, best_dist, best_abs_dot = classify_with_reasons(features.points, features.normals, families, args)

    stable_classified = stable & (labels >= 0)
    stable_unclassified = stable & (labels < 0)
    risk = ~stable
    write_cloud(out_dir / "stable_classified.ply", features.points[stable_classified], np.array([0.0, 1.0, 0.0]))
    write_cloud(out_dir / "stable_unclassified_all.ply", features.points[stable_unclassified], np.array([1.0, 0.35, 0.0]))
    write_cloud(out_dir / "teacher_core_stable.ply", features.points[teacher_stable], np.array([1.0, 1.0, 1.0]))
    write_cloud(out_dir / "risk_points.ply", features.points[risk], np.array([0.05, 0.05, 0.05]))
    for name, color in [
        ("distance", np.array([1.0, 0.0, 0.0])),
        ("extent", np.array([1.0, 1.0, 0.0])),
        ("normal", np.array([0.2, 0.3, 1.0])),
    ]:
        mask = stable_unclassified & (reason == name)
        write_cloud(out_dir / f"stable_unclassified_reason_{name}_{int(np.count_nonzero(mask))}pts.ply", features.points[mask], color)

    candidates = analyze_unclassified_candidates(pv, features, stable_unclassified, args, out_dir)
    family_rows = []
    for family in families:
        family_mask = stable & (labels == family.family_id)
        family_rows.append(
            {
                "familyId": family.family_id,
                "patchIds": family.patch_ids,
                "teacherPoints": int(len(family.points)),
                "classifiedStablePoints": int(np.count_nonzero(family_mask)),
                "normal": [float(x) for x in family.normal],
                "d": float(family.d),
                "centroid": [float(x) for x in family.centroid],
            }
        )

    reason_counts = {
        key: int(np.count_nonzero(stable_unclassified & (reason == key)))
        for key in ["distance", "extent", "normal"]
    }
    quantile_mask = stable_unclassified
    dist_quantiles = (
        np.quantile(best_dist[quantile_mask], [0.1, 0.5, 0.9, 0.99]).tolist()
        if np.count_nonzero(quantile_mask) > 0
        else []
    )
    dot_quantiles = (
        np.quantile(best_abs_dot[quantile_mask], [0.1, 0.5, 0.9]).tolist()
        if np.count_nonzero(quantile_mask) > 0
        else []
    )
    report = {
        "featureCsv": str(feature_csv),
        "validation": str(cli.validation),
        "totalPoints": int(len(features.points)),
        "stablePoints": int(np.count_nonzero(stable)),
        "teacherStablePoints": int(np.count_nonzero(teacher_stable)),
        "riskPoints": int(np.count_nonzero(risk)),
        "families": family_rows,
        "stableClassifiedPoints": int(np.count_nonzero(stable_classified)),
        "stableUnclassifiedPoints": int(np.count_nonzero(stable_unclassified)),
        "stableUnclassifiedReasonCounts": reason_counts,
        "stableUnclassifiedBestPlaneDistanceQuantiles": dist_quantiles,
        "stableUnclassifiedBestNormalDotQuantiles": dot_quantiles,
        "unclassifiedCandidatePatches": candidates,
        "teacherLeftoverPoints": int(len(leftover_points)),
    }
    with (out_dir / "weak_plane_diagnostic_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# Weak Plane Diagnostic Report",
        "",
        f"- total points: {report['totalPoints']}",
        f"- stable points: {report['stablePoints']}",
        f"- teacher core stable points: {report['teacherStablePoints']}",
        f"- risk points: {report['riskPoints']}",
        f"- families: {len(families)}",
        f"- stable classified: {report['stableClassifiedPoints']}",
        f"- stable unclassified: {report['stableUnclassifiedPoints']}",
        f"- unclassified reasons: {reason_counts}",
        f"- unclassified best plane distance quantiles p10/p50/p90/p99: {dist_quantiles}",
        f"- unclassified best normal dot quantiles p10/p50/p90: {dot_quantiles}",
        "",
        "## Families",
    ]
    for row in family_rows:
        lines.append(
            f"- family {row['familyId']}: patches={row['patchIds']} teacher={row['teacherPoints']} "
            f"classifiedStable={row['classifiedStablePoints']} d={row['d']:.4f} normal={row['normal']}"
        )
    lines.extend(["", "## Unclassified Candidate Patches"])
    if candidates:
        for row in candidates:
            lines.append(f"- patch {row['patchId']}: points={row['points']} d={row['d']:.4f} normal={row['normal']}")
    else:
        lines.append("- none")
    (out_dir / "weak_plane_diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"out={out_dir}")
    print(f"stable={report['stablePoints']} classified={report['stableClassifiedPoints']} unclassified={report['stableUnclassifiedPoints']}")
    print(f"reasons={reason_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
