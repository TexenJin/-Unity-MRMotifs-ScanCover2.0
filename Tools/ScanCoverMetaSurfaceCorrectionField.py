#!/usr/bin/env python3
"""Build a Meta-surface correction field from room-scale Raw Depth coverage.

This intentionally does not segment Meta Scene Mesh into plane families. Meta is
treated as one continuous reference surface. Raw coverage contributes local
support, signed depth offset, variance, and risk statistics around each sampled
Meta point.

Diagnostic outputs for CloudCompare:

- meta_support_good.ply
- meta_corrected_surface.ply
- meta_corrected_surface_full.ply
- meta_patch_corrected_surface_full.ply
- meta_patch_correction_field_all.ply
- meta_patch_offset_heatmap.ply
- meta_patch_applied_only.ply
- meta_raw_landed_surface.ply
- meta_raw_landed_full_with_candidates.ply
- meta_structure_candidate_unlanded.ply
- meta_landing_confidence_field.ply
- trusted_raw_surface.ply
- probable_raw_surface.ply
- candidate_supported_surface.ply
- evidence_supported_surface.ply
- evidence_observed_surface.ply
- weak_raw_support.ply
- rejected_raw_surface.ply
- layered_evidence_surface.ply
- meta_correctable_surface.ply
- unconfirmed_candidate_surface.ply
- trusted_region_report.json
- meta_unsupported.ply
- meta_risk_boundary.ply
- meta_correction_field_all.ply
- meta_surface_correction_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d

from ScanCoverRoomRawCoverageMetaOverlay import (
    DEFAULT_META,
    apply_transform,
    auto_align_raw_to_meta,
    distribution,
    load_meta_sample,
    normalize,
    read_room_voxels,
    write_cloud,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Meta surface correction field from Raw Depth coverage.")
    parser.add_argument("repeat_session", type=Path, help="RepeatCoverage session folder.")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META, help="Welded Meta Scene Mesh PLY/OBJ.")
    parser.add_argument("--out", type=Path, default=None, help="Output folder.")
    parser.add_argument("--meta-sample-points", type=int, default=400000)
    parser.add_argument("--auto-align", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--align-voxel", type=float, default=0.05)
    parser.add_argument("--align-max-distance", type=float, default=0.35)
    parser.add_argument("--support-radius", type=float, default=0.10)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--good-min-stable", type=int, default=2)
    parser.add_argument("--good-max-abs-offset", type=float, default=0.08)
    parser.add_argument("--good-max-offset-std", type=float, default=0.04)
    parser.add_argument("--risk-max-abs-offset", type=float, default=0.18)
    parser.add_argument("--risk-ratio-threshold", type=float, default=0.35)
    parser.add_argument("--risk-stable-override-ratio", type=float, default=0.55)
    parser.add_argument("--landing-min-stable-ratio", type=float, default=0.35)
    parser.add_argument("--landing-min-support", type=int, default=4)
    parser.add_argument("--landing-max-risk-ratio", type=float, default=0.65)
    parser.add_argument("--landing-max-offset-std", type=float, default=0.10)
    parser.add_argument("--probable-score", type=float, default=0.60)
    parser.add_argument("--candidate-score", type=float, default=0.35)
    parser.add_argument("--weak-score", type=float, default=0.10)
    parser.add_argument("--probable-to-trusted-radius", type=float, default=0.15)
    parser.add_argument("--candidate-to-strong-radius", type=float, default=0.12)
    parser.add_argument("--weak-to-strong-radius", type=float, default=0.08)
    parser.add_argument("--evidence-distance-radius-mult", type=float, default=2.0)
    parser.add_argument("--max-correction", type=float, default=0.12)
    parser.add_argument("--smooth-radius", type=float, default=0.12)
    parser.add_argument("--smooth-min-neighbors", type=int, default=4)
    parser.add_argument("--patch-size", type=float, default=0.18)
    parser.add_argument("--patch-min-points", type=int, default=24)
    parser.add_argument("--patch-min-correction-ratio", type=float, default=0.35)
    parser.add_argument("--patch-max-offset-std", type=float, default=0.07)
    return parser.parse_args()


def signed_offsets(raw_points: np.ndarray, meta_points: np.ndarray, meta_normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_cloud = o3d.geometry.PointCloud()
    raw_cloud.points = o3d.utility.Vector3dVector(raw_points)
    tree = o3d.geometry.KDTreeFlann(raw_cloud)

    nearest_raw_ids = np.empty((len(meta_points),), dtype=np.int64)
    nearest_distances = np.empty((len(meta_points),), dtype=np.float64)
    for i, point in enumerate(meta_points):
        _, idx, d2 = tree.search_knn_vector_3d(point, 1)
        nearest_raw_ids[i] = int(idx[0])
        nearest_distances[i] = math.sqrt(float(d2[0]))
    nearest_raw = raw_points[nearest_raw_ids]
    signed = np.sum((nearest_raw - meta_points) * meta_normals, axis=1)
    return nearest_distances, signed


def build_support_stats(
    meta_points: np.ndarray,
    meta_normals: np.ndarray,
    raw_points: np.ndarray,
    raw_stable: np.ndarray,
    raw_risk: np.ndarray,
    radius: float,
) -> dict[str, np.ndarray]:
    raw_cloud = o3d.geometry.PointCloud()
    raw_cloud.points = o3d.utility.Vector3dVector(raw_points)
    tree = o3d.geometry.KDTreeFlann(raw_cloud)

    count = np.zeros((len(meta_points),), dtype=np.int32)
    stable_count = np.zeros((len(meta_points),), dtype=np.int32)
    risk_count = np.zeros((len(meta_points),), dtype=np.int32)
    offset_mean = np.zeros((len(meta_points),), dtype=np.float64)
    offset_std = np.zeros((len(meta_points),), dtype=np.float64)
    nearest_distance = np.zeros((len(meta_points),), dtype=np.float64)
    nearest_signed = np.zeros((len(meta_points),), dtype=np.float64)

    for i, point in enumerate(meta_points):
        _, idx, d2 = tree.search_radius_vector_3d(point, radius)
        if len(idx) == 0:
            _, idx, d2 = tree.search_knn_vector_3d(point, 1)
        ids = np.asarray(idx, dtype=np.int64)
        deltas = raw_points[ids] - point
        signed = np.sum(deltas * meta_normals[i], axis=1)
        distances = np.sqrt(np.asarray(d2, dtype=np.float64))

        count[i] = int(len(ids))
        stable_count[i] = int(np.count_nonzero(raw_stable[ids]))
        risk_count[i] = int(np.count_nonzero(raw_risk[ids]))
        offset_mean[i] = float(np.mean(signed))
        offset_std[i] = float(np.std(signed))
        nearest_pos = int(np.argmin(distances))
        nearest_distance[i] = float(distances[nearest_pos])
        nearest_signed[i] = float(signed[nearest_pos])

    return {
        "count": count,
        "stable_count": stable_count,
        "risk_count": risk_count,
        "offset_mean": offset_mean,
        "offset_std": offset_std,
        "nearest_distance": nearest_distance,
        "nearest_signed": nearest_signed,
    }


def smooth_offsets(
    meta_points: np.ndarray,
    offsets: np.ndarray,
    good_mask: np.ndarray,
    radius: float,
    min_neighbors: int,
) -> np.ndarray:
    smoothed = offsets.copy()
    if np.count_nonzero(good_mask) == 0:
        return smoothed

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(meta_points)
    tree = o3d.geometry.KDTreeFlann(cloud)

    for i, point in enumerate(meta_points):
        if not good_mask[i]:
            continue
        _, idx, _ = tree.search_radius_vector_3d(point, radius)
        ids = np.asarray(idx, dtype=np.int64)
        ids = ids[good_mask[ids]]
        if len(ids) < min_neighbors:
            continue
        smoothed[i] = float(np.median(offsets[ids]))
    return smoothed


def support_colors(
    good: np.ndarray,
    corrected: np.ndarray,
    unsupported: np.ndarray,
    risk: np.ndarray,
) -> np.ndarray:
    colors = np.zeros((len(good), 3), dtype=np.float64)
    colors[:] = (0.18, 0.18, 0.18)
    colors[unsupported] = (0.9, 0.9, 0.9)
    colors[corrected] = (0.0, 0.85, 1.0)
    colors[good] = (0.05, 1.0, 0.2)
    colors[risk] = (1.0, 0.05, 0.05)
    return colors


def signed_offset_heatmap(offsets: np.ndarray, active: np.ndarray, max_abs: float) -> np.ndarray:
    colors = np.full((len(offsets), 3), 0.18, dtype=np.float64)
    if len(offsets) == 0 or max_abs <= 0:
        return colors

    values = np.clip(offsets / max_abs, -1.0, 1.0)
    negative = active & (values < 0)
    positive = active & (values > 0)
    neutral = active & (np.abs(values) <= 0.08)

    colors[neutral] = (0.95, 0.95, 0.95)
    neg_t = np.abs(values[negative])[:, None]
    pos_t = values[positive][:, None]
    # Negative offset is blue/cyan, positive offset is red/orange. This makes
    # the correction direction visible instead of only showing support classes.
    colors[negative] = (1.0 - 0.85 * neg_t) * np.array((0.95, 0.95, 0.95)) + neg_t * np.array((0.0, 0.25, 1.0))
    colors[positive] = (1.0 - 0.85 * pos_t) * np.array((0.95, 0.95, 0.95)) + pos_t * np.array((1.0, 0.05, 0.0))
    return np.clip(colors, 0.0, 1.0)


def landing_confidence_colors(
    landed: np.ndarray,
    good: np.ndarray,
    corrected: np.ndarray,
    unsupported: np.ndarray,
    risk: np.ndarray,
) -> np.ndarray:
    colors = np.full((len(landed), 3), (0.16, 0.16, 0.16), dtype=np.float64)
    colors[unsupported] = (0.65, 0.65, 0.65)
    colors[corrected] = (0.0, 0.85, 1.0)
    colors[good] = (0.05, 1.0, 0.2)
    colors[risk] = (1.0, 0.05, 0.05)
    # White means this sample is accepted into the Raw-position landing layer.
    colors[landed] = (1.0, 1.0, 1.0)
    return colors


def clamp01(values: np.ndarray | float) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def nearest_distance_to_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    distances = np.full((len(points),), np.inf, dtype=np.float64)
    if len(points) == 0 or np.count_nonzero(mask) == 0:
        return distances

    reference = o3d.geometry.PointCloud()
    reference.points = o3d.utility.Vector3dVector(points[mask])
    tree = o3d.geometry.KDTreeFlann(reference)
    for i, point in enumerate(points):
        _, _, d2 = tree.search_knn_vector_3d(point, 1)
        distances[i] = math.sqrt(float(d2[0]))
    return distances


def evidence_colors(
    trusted: np.ndarray,
    probable: np.ndarray,
    candidate: np.ndarray,
    weak: np.ndarray,
    rejected: np.ndarray,
) -> np.ndarray:
    colors = np.full((len(trusted), 3), (0.16, 0.16, 0.16), dtype=np.float64)
    colors[weak] = (0.10, 0.35, 1.0)
    colors[candidate] = (1.0, 0.78, 0.05)
    colors[probable] = (0.0, 0.88, 1.0)
    colors[trusted] = (0.05, 1.0, 0.20)
    colors[rejected] = (1.0, 0.05, 0.05)
    return colors


def build_layered_evidence(
    meta_points: np.ndarray,
    raw_landed: np.ndarray,
    supported: np.ndarray,
    unsupported: np.ndarray,
    hard_risk: np.ndarray,
    vote_risk: np.ndarray,
    support_count: np.ndarray,
    stable_count: np.ndarray,
    stable_ratio: np.ndarray,
    risk_ratio: np.ndarray,
    offset_mean: np.ndarray,
    offset_std: np.ndarray,
    nearest_distance: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """Score Meta samples with layered Raw evidence instead of a binary gate.

    Green/trusted keeps the existing strict landing rule. The other layers are
    evidence grades: they can guide alignment, meshing, and rescan decisions,
    but they do not overwrite the strict trusted surface.
    """
    min_support = max(1, int(args.min_support))
    landing_support = max(min_support, int(args.landing_min_support))
    support_score = clamp01(support_count / max(1.0, landing_support * 2.0))
    stable_count_score = clamp01(stable_count / max(1.0, args.good_min_stable * 3.0))
    stable_score = np.maximum(stable_ratio, stable_count_score)
    risk_score = clamp01(1.0 - risk_ratio)
    distance_radius = max(1e-4, args.support_radius * max(1.0, args.evidence_distance_radius_mult))
    distance_score = clamp01(1.0 - nearest_distance / distance_radius)
    offset_score = clamp01(1.0 - np.abs(offset_mean) / max(1e-4, args.risk_max_abs_offset))
    variance_score = clamp01(1.0 - offset_std / max(1e-4, args.landing_max_offset_std))

    evidence_score = (
        0.24 * stable_score
        + 0.20 * support_score
        + 0.20 * distance_score
        + 0.16 * offset_score
        + 0.10 * variance_score
        + 0.10 * risk_score
    )
    evidence_score = clamp01(evidence_score)
    evidence_score[unsupported] *= 0.55
    evidence_score[vote_risk] *= 0.70
    evidence_score[hard_risk] = 0.0
    evidence_score[raw_landed] = np.maximum(evidence_score[raw_landed], 0.90)

    trusted = raw_landed.copy()
    rejected = hard_risk | (vote_risk & (evidence_score < args.candidate_score))

    dist_to_trusted = nearest_distance_to_mask(meta_points, trusted)
    probable_seed = (
        ~trusted
        & ~rejected
        & supported
        & (evidence_score >= args.probable_score)
    )
    probable = probable_seed & (dist_to_trusted <= args.probable_to_trusted_radius)

    strong = trusted | probable
    dist_to_strong = nearest_distance_to_mask(meta_points, strong)
    candidate_seed = (
        ~trusted
        & ~probable
        & ~rejected
        & (evidence_score >= args.candidate_score)
    )
    candidate = candidate_seed & (dist_to_strong <= args.candidate_to_strong_radius)

    stronger = trusted | probable | candidate
    dist_to_evidence = nearest_distance_to_mask(meta_points, stronger)
    weak_seed = (
        ~trusted
        & ~probable
        & ~candidate
        & ~rejected
        & (evidence_score > args.weak_score)
    )
    weak = weak_seed & (dist_to_evidence <= args.weak_to_strong_radius)

    blocked = ~(trusted | probable | candidate | weak | rejected)
    return {
        "score": evidence_score,
        "supportScore": support_score,
        "stableScore": stable_score,
        "riskScore": risk_score,
        "distanceScore": distance_score,
        "offsetScore": offset_score,
        "varianceScore": variance_score,
        "trusted": trusted,
        "probable": probable,
        "candidate": candidate,
        "weak": weak,
        "rejected": rejected,
        "blocked": blocked,
        "distToTrusted": dist_to_trusted,
        "distToStrong": dist_to_strong,
        "distToEvidence": dist_to_evidence,
    }


def write_layered_evidence_outputs(
    out_dir: Path,
    meta_points: np.ndarray,
    meta_normals: np.ndarray,
    raw_landed_points: np.ndarray,
    layers: dict[str, np.ndarray],
) -> dict[str, object]:
    trusted = layers["trusted"]
    probable = layers["probable"]
    candidate = layers["candidate"]
    weak = layers["weak"]
    rejected = layers["rejected"]
    blocked = layers["blocked"]
    score = layers["score"]

    display_points = meta_points.copy()
    display_points[trusted | probable | candidate] = raw_landed_points[trusted | probable | candidate]
    colors = evidence_colors(trusted, probable, candidate, weak, rejected)
    supported_evidence = trusted | probable | candidate
    observed_evidence = supported_evidence | weak

    write_cloud(out_dir / "layered_evidence_surface.ply", display_points, colors, meta_normals)
    write_cloud(out_dir / "evidence_supported_surface.ply", display_points[supported_evidence], colors[supported_evidence], meta_normals[supported_evidence])
    write_cloud(out_dir / "evidence_observed_surface.ply", display_points[observed_evidence], colors[observed_evidence], meta_normals[observed_evidence])
    write_cloud(out_dir / "probable_raw_surface.ply", display_points[probable], np.tile((0.0, 0.88, 1.0), (np.count_nonzero(probable), 1)), meta_normals[probable])
    write_cloud(out_dir / "candidate_supported_surface.ply", display_points[candidate], np.tile((1.0, 0.78, 0.05), (np.count_nonzero(candidate), 1)), meta_normals[candidate])
    write_cloud(out_dir / "weak_raw_support.ply", meta_points[weak], np.tile((0.10, 0.35, 1.0), (np.count_nonzero(weak), 1)), meta_normals[weak])
    write_cloud(out_dir / "rejected_raw_surface.ply", meta_points[rejected], np.tile((1.0, 0.05, 0.05), (np.count_nonzero(rejected), 1)), meta_normals[rejected])

    total = max(1, len(meta_points))
    report = {
        "meaning": {
            "trusted": "Green. Strict Raw-landed surface; this keeps the previous trusted_raw_surface contract.",
            "probable": "Cyan. Non-red evidence with enough score and proximity to trusted green support.",
            "candidate": "Yellow. Weaker evidence that is close to green/cyan support; useful for voting and scan guidance.",
            "weak": "Blue. Observed Raw support with low confidence; cannot prove structure by itself.",
            "rejected": "Red. Hard risk, or vote-risk evidence too weak to pass.",
            "blocked": "Not colored as evidence. Too far from stronger evidence or too low scoring.",
        },
        "counts": {
            "trusted": int(np.count_nonzero(trusted)),
            "probable": int(np.count_nonzero(probable)),
            "candidate": int(np.count_nonzero(candidate)),
            "weak": int(np.count_nonzero(weak)),
            "rejected": int(np.count_nonzero(rejected)),
            "blocked": int(np.count_nonzero(blocked)),
            "supportedEvidence": int(np.count_nonzero(supported_evidence)),
            "observedEvidence": int(np.count_nonzero(observed_evidence)),
        },
        "ratios": {
            "trusted": float(np.count_nonzero(trusted) / total),
            "probable": float(np.count_nonzero(probable) / total),
            "candidate": float(np.count_nonzero(candidate) / total),
            "weak": float(np.count_nonzero(weak) / total),
            "rejected": float(np.count_nonzero(rejected) / total),
            "blocked": float(np.count_nonzero(blocked) / total),
            "supportedEvidence": float(np.count_nonzero(supported_evidence) / total),
            "observedEvidence": float(np.count_nonzero(observed_evidence) / total),
        },
        "scoreDistributions": {
            "all": distribution(score),
            "trusted": distribution(score[trusted]),
            "probable": distribution(score[probable]),
            "candidate": distribution(score[candidate]),
            "weak": distribution(score[weak]),
            "rejected": distribution(score[rejected]),
            "blocked": distribution(score[blocked]),
        },
        "voteRule": [
            "Red rejects first.",
            "Green trusted keeps the strict raw_landed gate and can anchor other evidence.",
            "Cyan probable must score high and stay near green.",
            "Yellow candidate must stay near green/cyan; yellow does not prove yellow.",
            "Blue weak must stay near stronger evidence and cannot prove structure alone.",
        ],
    }
    (out_dir / "layered_evidence_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_trusted_region_outputs(
    out_dir: Path,
    meta_points: np.ndarray,
    meta_normals: np.ndarray,
    raw_landed_points: np.ndarray,
    raw_landed: np.ndarray,
    correction_mask: np.ndarray,
    unsupported: np.ndarray,
    risk: np.ndarray,
    hard_risk: np.ndarray,
    vote_risk: np.ndarray,
    support_count: np.ndarray,
    stable_count: np.ndarray,
    stable_ratio: np.ndarray,
    risk_ratio: np.ndarray,
    offset_mean: np.ndarray,
    offset_std: np.ndarray,
    nearest_distance: np.ndarray,
    support_radius: float,
) -> dict[str, object]:
    """Emit architecture-level trusted-region artifacts.

    Meta is the structure prior. Raw is allowed to land only where local evidence
    is stable, close, and low variance. Everything else remains diagnostic or
    candidate data instead of being silently promoted to true geometry.
    """
    meta_correctable = correction_mask & ~hard_risk
    unconfirmed_candidate = ~raw_landed
    weak_support = unconfirmed_candidate & unsupported
    uncertain_support = unconfirmed_candidate & ~unsupported & ~risk

    trusted_colors = np.tile((1.0, 1.0, 1.0), (np.count_nonzero(raw_landed), 1))
    correctable_colors = np.tile((0.0, 0.9, 1.0), (np.count_nonzero(meta_correctable), 1))
    candidate_colors = np.full((len(meta_points), 3), (0.45, 0.45, 0.45), dtype=np.float64)
    candidate_colors[weak_support] = (0.80, 0.80, 0.80)
    candidate_colors[uncertain_support] = (1.0, 0.75, 0.10)
    candidate_colors[vote_risk] = (1.0, 0.30, 0.05)
    candidate_colors[hard_risk] = (1.0, 0.0, 0.0)

    write_cloud(
        out_dir / "trusted_raw_surface.ply",
        raw_landed_points[raw_landed],
        trusted_colors,
        meta_normals[raw_landed],
    )
    write_cloud(
        out_dir / "meta_correctable_surface.ply",
        raw_landed_points[meta_correctable],
        correctable_colors,
        meta_normals[meta_correctable],
    )
    write_cloud(
        out_dir / "unconfirmed_candidate_surface.ply",
        meta_points[unconfirmed_candidate],
        candidate_colors[unconfirmed_candidate],
        meta_normals[unconfirmed_candidate],
    )

    nearest_good = nearest_distance <= support_radius
    report = {
        "meaning": {
            "trusted_raw_surface.ply": "Meta-structured samples that have landed on stable Raw evidence. This is the current trusted true-depth surface seed.",
            "meta_correctable_surface.ply": "Meta samples that are eligible for Raw position correction. It includes trusted landed points and other non-hard-risk correction candidates.",
            "unconfirmed_candidate_surface.ply": "Meta structure that remains useful for continuity but is not yet confirmed by stable Raw evidence.",
        },
        "counts": {
            "metaSamples": int(len(meta_points)),
            "trustedRawSurface": int(np.count_nonzero(raw_landed)),
            "metaCorrectableSurface": int(np.count_nonzero(meta_correctable)),
            "unconfirmedCandidateSurface": int(np.count_nonzero(unconfirmed_candidate)),
            "weakSupportCandidate": int(np.count_nonzero(weak_support)),
            "uncertainSupportCandidate": int(np.count_nonzero(uncertain_support)),
            "voteRiskCandidate": int(np.count_nonzero(vote_risk & unconfirmed_candidate)),
            "hardRiskCandidate": int(np.count_nonzero(hard_risk & unconfirmed_candidate)),
        },
        "ratios": {
            "trustedRawSurface": float(np.count_nonzero(raw_landed) / max(1, len(meta_points))),
            "metaCorrectableSurface": float(np.count_nonzero(meta_correctable) / max(1, len(meta_points))),
            "unconfirmedCandidateSurface": float(np.count_nonzero(unconfirmed_candidate) / max(1, len(meta_points))),
            "weakSupportCandidate": float(np.count_nonzero(weak_support) / max(1, len(meta_points))),
            "uncertainSupportCandidate": float(np.count_nonzero(uncertain_support) / max(1, len(meta_points))),
            "nearestWithinSupportRadius": float(np.count_nonzero(nearest_good) / max(1, len(meta_points))),
        },
        "trustedRegionDistributions": {
            "supportCount": distribution(support_count[raw_landed]),
            "stableCount": distribution(stable_count[raw_landed]),
            "stableRatio": distribution(stable_ratio[raw_landed]),
            "riskRatio": distribution(risk_ratio[raw_landed]),
            "offsetMeanMeters": distribution(offset_mean[raw_landed]),
            "offsetStdMeters": distribution(offset_std[raw_landed]),
            "nearestRawDistanceMeters": distribution(nearest_distance[raw_landed]),
        },
        "candidateDistributions": {
            "supportCount": distribution(support_count[unconfirmed_candidate]),
            "stableCount": distribution(stable_count[unconfirmed_candidate]),
            "stableRatio": distribution(stable_ratio[unconfirmed_candidate]),
            "riskRatio": distribution(risk_ratio[unconfirmed_candidate]),
            "offsetMeanMeters": distribution(offset_mean[unconfirmed_candidate]),
            "offsetStdMeters": distribution(offset_std[unconfirmed_candidate]),
            "nearestRawDistanceMeters": distribution(nearest_distance[unconfirmed_candidate]),
        },
        "decisionRule": [
            "Meta supplies the continuous structure.",
            "Raw evidence lands the Meta sample only when support is close, stable, and low variance.",
            "Unsupported high/ceiling gaps remain unconfirmed candidates, not trusted geometry.",
            "Hard risk is never promoted into the trusted Raw surface.",
        ],
    }
    (out_dir / "trusted_region_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_patch_correction(
    meta_points: np.ndarray,
    meta_normals: np.ndarray,
    base_offsets: np.ndarray,
    correction_mask: np.ndarray,
    risk_mask: np.ndarray,
    patch_size: float,
    patch_min_points: int,
    patch_min_correction_ratio: float,
    patch_max_offset_std: float,
    max_correction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    patch_offsets = np.zeros((len(meta_points),), dtype=np.float64)
    patch_applied = np.zeros((len(meta_points),), dtype=bool)
    patch_keys = np.floor(meta_points / patch_size).astype(np.int64)

    key_to_indices: dict[tuple[int, int, int], list[int]] = {}
    for i, key in enumerate(patch_keys):
        key_to_indices.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(i)

    patch_reports: list[dict[str, object]] = []
    for key, indices_list in key_to_indices.items():
        ids = np.asarray(indices_list, dtype=np.int64)
        if len(ids) < patch_min_points:
            continue
        candidate_ids = ids[correction_mask[ids]]
        if len(candidate_ids) < patch_min_points:
            continue
        correction_ratio = float(len(candidate_ids) / max(1, len(ids)))
        if correction_ratio < patch_min_correction_ratio:
            continue
        offsets = base_offsets[candidate_ids]
        offset_std = float(np.std(offsets))
        if offset_std > patch_max_offset_std:
            continue

        patch_offset = float(np.median(offsets))
        patch_offset = max(-max_correction, min(max_correction, patch_offset))
        # Apply to all non-risk points in this local patch. This preserves the
        # continuous Meta surface but moves coherent local areas as a slab.
        apply_ids = ids[~risk_mask[ids]]
        if len(apply_ids) == 0:
            continue
        patch_offsets[apply_ids] = patch_offset
        patch_applied[apply_ids] = True
        patch_reports.append(
            {
                "key": key,
                "points": int(len(ids)),
                "candidatePoints": int(len(candidate_ids)),
                "appliedPoints": int(len(apply_ids)),
                "correctionRatio": correction_ratio,
                "offsetMedian": patch_offset,
                "offsetStd": offset_std,
            }
        )

    patch_corrected_points = meta_points.copy()
    patch_corrected_points[patch_applied] = (
        meta_points[patch_applied]
        + patch_offsets[patch_applied, None] * meta_normals[patch_applied]
    )
    return patch_corrected_points, patch_applied, patch_offsets, {
        "enabled": True,
        "patchSizeMeters": patch_size,
        "patchMinPoints": patch_min_points,
        "patchMinCorrectionRatio": patch_min_correction_ratio,
        "patchMaxOffsetStd": patch_max_offset_std,
        "patchCount": int(len(key_to_indices)),
        "appliedPatchCount": int(len(patch_reports)),
        "appliedPointCount": int(np.count_nonzero(patch_applied)),
        "appliedOffsetMeters": distribution(patch_offsets[patch_applied]),
        "patches": patch_reports[:500],
    }


def main() -> int:
    args = parse_args()
    session = args.repeat_session.resolve()
    voxels_csv = session / "room_raw_coverage" / "room_raw_coverage_voxels.csv"
    if not voxels_csv.exists():
        raise FileNotFoundError(voxels_csv)

    out_dir = args.out.resolve() if args.out else session / "meta_surface_correction_field"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_room_voxels(voxels_csv)
    raw_points = raw["points"]
    raw_normals = raw["normals"]
    raw_stable = raw["stable"]
    raw_risk = raw["risk"]

    meta_points, meta_normals = load_meta_sample(args.meta.resolve(), args.meta_sample_points)
    alignment: dict[str, object] = {"enabled": False}
    if args.auto_align:
        transform, alignment = auto_align_raw_to_meta(
            raw_points,
            raw_stable,
            meta_points,
            args.align_voxel,
            args.align_max_distance,
        )
        raw_points = apply_transform(raw_points, transform)
        raw_normals = normalize(raw_normals @ transform[:3, :3].T)

    stats = build_support_stats(
        meta_points,
        meta_normals,
        raw_points,
        raw_stable,
        raw_risk,
        args.support_radius,
    )

    support_count = stats["count"]
    stable_count = stats["stable_count"]
    risk_count = stats["risk_count"]
    risk_ratio = risk_count / np.maximum(1, support_count)
    stable_ratio = stable_count / np.maximum(1, support_count)
    offset_mean = stats["offset_mean"]
    offset_std = stats["offset_std"]
    nearest_distance = stats["nearest_distance"]

    supported = support_count >= args.min_support
    unsupported = ~supported
    stable_overrides_vote_risk = (
        (stable_ratio >= args.risk_stable_override_ratio)
        | ((stable_count >= args.good_min_stable) & (offset_std <= args.landing_max_offset_std))
    )
    good = (
        supported
        & (stable_count >= args.good_min_stable)
        & (np.abs(offset_mean) <= args.good_max_abs_offset)
        & (offset_std <= args.good_max_offset_std)
        & ((risk_ratio <= args.risk_ratio_threshold) | stable_overrides_vote_risk)
    )
    corrected = (
        supported
        & ~good
        & (np.abs(offset_mean) <= args.max_correction)
        & ((risk_ratio <= args.risk_ratio_threshold) | stable_overrides_vote_risk)
    )
    hard_risk = (
        supported
        & (
            (np.abs(offset_mean) > args.risk_max_abs_offset)
            | (offset_std > args.good_max_offset_std * 2.5)
        )
    )
    vote_risk = supported & (risk_ratio > args.risk_ratio_threshold) & ~stable_overrides_vote_risk
    risk = hard_risk | vote_risk

    clamped_offsets = np.clip(offset_mean, -args.max_correction, args.max_correction)
    correction_mask = good | corrected
    smoothed_offsets = smooth_offsets(
        meta_points,
        clamped_offsets,
        correction_mask,
        args.smooth_radius,
        args.smooth_min_neighbors,
    )
    corrected_points = meta_points.copy()
    corrected_points[correction_mask] = (
        meta_points[correction_mask]
        + smoothed_offsets[correction_mask, None] * meta_normals[correction_mask]
    )
    patch_corrected_points, patch_applied, patch_offsets, patch_report = build_patch_correction(
        meta_points,
        meta_normals,
        smoothed_offsets,
        correction_mask,
        risk,
        args.patch_size,
        args.patch_min_points,
        args.patch_min_correction_ratio,
        args.patch_max_offset_std,
        args.max_correction,
    )

    landing_evidence = (
        (
            (stable_count >= args.good_min_stable)
            & (stable_ratio >= args.landing_min_stable_ratio)
        )
        | (
            (support_count >= args.landing_min_support)
            & (risk_ratio <= args.landing_max_risk_ratio)
            & (offset_std <= args.landing_max_offset_std)
        )
    )

    # Meta gives the complete structure, but Raw decides whether a sample is
    # allowed to become real geometry. This avoids silently using Meta-only
    # completion as if it were true depth.
    raw_landed = (
        correction_mask
        & ~hard_risk
        & landing_evidence
        & (nearest_distance <= args.support_radius)
    )
    raw_landed_points = patch_corrected_points.copy()
    raw_landed_points[~patch_applied] = corrected_points[~patch_applied]
    unlanded = ~raw_landed
    layered_evidence = build_layered_evidence(
        meta_points,
        raw_landed,
        supported,
        unsupported,
        hard_risk,
        vote_risk,
        support_count,
        stable_count,
        stable_ratio,
        risk_ratio,
        offset_mean,
        offset_std,
        nearest_distance,
        args,
    )

    colors = support_colors(good, corrected, unsupported, risk)
    patch_colors = colors.copy()
    patch_colors[patch_applied] = (0.0, 1.0, 1.0)
    patch_offset_colors = signed_offset_heatmap(patch_offsets, patch_applied, args.max_correction)
    landing_colors = landing_confidence_colors(raw_landed, good, corrected, unsupported, risk)
    full_landing_points = meta_points.copy()
    full_landing_points[raw_landed] = raw_landed_points[raw_landed]
    write_cloud(out_dir / "meta_correction_field_all.ply", meta_points, colors, meta_normals)
    write_cloud(out_dir / "meta_support_good.ply", meta_points[good], np.tile((0.05, 1.0, 0.2), (np.count_nonzero(good), 1)), meta_normals[good])
    write_cloud(out_dir / "meta_corrected_surface.ply", corrected_points[correction_mask], colors[correction_mask], meta_normals[correction_mask])
    write_cloud(out_dir / "meta_corrected_surface_full.ply", corrected_points, colors, meta_normals)
    write_cloud(out_dir / "meta_patch_corrected_surface_full.ply", patch_corrected_points, patch_colors, meta_normals)
    write_cloud(out_dir / "meta_patch_correction_field_all.ply", meta_points, patch_colors, meta_normals)
    write_cloud(out_dir / "meta_patch_offset_heatmap.ply", meta_points, patch_offset_colors, meta_normals)
    write_cloud(out_dir / "meta_patch_applied_only.ply", patch_corrected_points[patch_applied], patch_offset_colors[patch_applied], meta_normals[patch_applied])
    write_cloud(out_dir / "meta_landing_confidence_field.ply", meta_points, landing_colors, meta_normals)
    write_cloud(out_dir / "meta_raw_landed_surface.ply", raw_landed_points[raw_landed], np.tile((1.0, 1.0, 1.0), (np.count_nonzero(raw_landed), 1)), meta_normals[raw_landed])
    write_cloud(out_dir / "meta_raw_landed_full_with_candidates.ply", full_landing_points, landing_colors, meta_normals)
    write_cloud(out_dir / "meta_structure_candidate_unlanded.ply", meta_points[unlanded], landing_colors[unlanded], meta_normals[unlanded])
    write_cloud(out_dir / "meta_unsupported.ply", meta_points[unsupported], np.tile((0.9, 0.9, 0.9), (np.count_nonzero(unsupported), 1)), meta_normals[unsupported])
    write_cloud(out_dir / "meta_risk_boundary.ply", meta_points[risk], np.tile((1.0, 0.05, 0.05), (np.count_nonzero(risk), 1)), meta_normals[risk])
    layered_evidence_report = write_layered_evidence_outputs(
        out_dir,
        meta_points,
        meta_normals,
        raw_landed_points,
        layered_evidence,
    )
    trusted_region_report = write_trusted_region_outputs(
        out_dir,
        meta_points,
        meta_normals,
        raw_landed_points,
        raw_landed,
        correction_mask,
        unsupported,
        risk,
        hard_risk,
        vote_risk,
        support_count,
        stable_count,
        stable_ratio,
        risk_ratio,
        offset_mean,
        offset_std,
        nearest_distance,
        args.support_radius,
    )

    report = {
        "repeatSession": str(session),
        "roomVoxelCsv": str(voxels_csv),
        "metaReference": str(args.meta.resolve()),
        "outputDirectory": str(out_dir),
        "alignment": alignment,
        "rawVoxelCount": int(len(raw_points)),
        "metaSampleCount": int(len(meta_points)),
        "parameters": {
            "supportRadius": args.support_radius,
            "minSupport": args.min_support,
            "goodMinStable": args.good_min_stable,
            "goodMaxAbsOffset": args.good_max_abs_offset,
            "goodMaxOffsetStd": args.good_max_offset_std,
            "riskMaxAbsOffset": args.risk_max_abs_offset,
            "riskRatioThreshold": args.risk_ratio_threshold,
            "riskStableOverrideRatio": args.risk_stable_override_ratio,
            "landingMinStableRatio": args.landing_min_stable_ratio,
            "landingMinSupport": args.landing_min_support,
            "landingMaxRiskRatio": args.landing_max_risk_ratio,
            "landingMaxOffsetStd": args.landing_max_offset_std,
            "landingMinStableRatio": args.landing_min_stable_ratio,
            "probableScore": args.probable_score,
            "candidateScore": args.candidate_score,
            "weakScore": args.weak_score,
            "probableToTrustedRadius": args.probable_to_trusted_radius,
            "candidateToStrongRadius": args.candidate_to_strong_radius,
            "weakToStrongRadius": args.weak_to_strong_radius,
            "evidenceDistanceRadiusMult": args.evidence_distance_radius_mult,
            "maxCorrection": args.max_correction,
            "smoothRadius": args.smooth_radius,
            "patchSize": args.patch_size,
            "patchMinPoints": args.patch_min_points,
            "patchMinCorrectionRatio": args.patch_min_correction_ratio,
            "patchMaxOffsetStd": args.patch_max_offset_std,
        },
        "counts": {
            "supported": int(np.count_nonzero(supported)),
            "unsupported": int(np.count_nonzero(unsupported)),
            "good": int(np.count_nonzero(good)),
            "corrected": int(np.count_nonzero(corrected)),
            "hardRisk": int(np.count_nonzero(hard_risk)),
            "voteRisk": int(np.count_nonzero(vote_risk)),
            "risk": int(np.count_nonzero(risk)),
            "correctionMask": int(np.count_nonzero(correction_mask)),
            "patchApplied": int(np.count_nonzero(patch_applied)),
            "rawLanded": int(np.count_nonzero(raw_landed)),
            "structureCandidateUnlanded": int(np.count_nonzero(unlanded)),
            "evidenceTrusted": int(np.count_nonzero(layered_evidence["trusted"])),
            "evidenceProbable": int(np.count_nonzero(layered_evidence["probable"])),
            "evidenceCandidate": int(np.count_nonzero(layered_evidence["candidate"])),
            "evidenceWeak": int(np.count_nonzero(layered_evidence["weak"])),
            "evidenceRejected": int(np.count_nonzero(layered_evidence["rejected"])),
        },
        "ratios": {
            "supported": float(np.count_nonzero(supported) / max(1, len(meta_points))),
            "unsupported": float(np.count_nonzero(unsupported) / max(1, len(meta_points))),
            "good": float(np.count_nonzero(good) / max(1, len(meta_points))),
            "corrected": float(np.count_nonzero(corrected) / max(1, len(meta_points))),
            "hardRisk": float(np.count_nonzero(hard_risk) / max(1, len(meta_points))),
            "voteRisk": float(np.count_nonzero(vote_risk) / max(1, len(meta_points))),
            "risk": float(np.count_nonzero(risk) / max(1, len(meta_points))),
            "patchApplied": float(np.count_nonzero(patch_applied) / max(1, len(meta_points))),
            "rawLanded": float(np.count_nonzero(raw_landed) / max(1, len(meta_points))),
            "structureCandidateUnlanded": float(np.count_nonzero(unlanded) / max(1, len(meta_points))),
            "evidenceTrusted": float(np.count_nonzero(layered_evidence["trusted"]) / max(1, len(meta_points))),
            "evidenceProbable": float(np.count_nonzero(layered_evidence["probable"]) / max(1, len(meta_points))),
            "evidenceCandidate": float(np.count_nonzero(layered_evidence["candidate"]) / max(1, len(meta_points))),
            "evidenceWeak": float(np.count_nonzero(layered_evidence["weak"]) / max(1, len(meta_points))),
            "evidenceRejected": float(np.count_nonzero(layered_evidence["rejected"]) / max(1, len(meta_points))),
        },
        "landingDecision": {
            "meaning": "Meta provides the full structure. A Meta sample lands only when nearby Raw coverage is stable, low-risk, close enough, and has bounded local offset variance.",
            "landedFiles": [
                "meta_raw_landed_surface.ply",
                "meta_raw_landed_full_with_candidates.ply",
                "trusted_raw_surface.ply",
            ],
            "candidateFiles": [
                "meta_structure_candidate_unlanded.ply",
                "meta_landing_confidence_field.ply",
                "unconfirmed_candidate_surface.ply",
            ],
            "rawLandedCount": int(np.count_nonzero(raw_landed)),
            "rawLandedRatio": float(np.count_nonzero(raw_landed) / max(1, len(meta_points))),
            "structureCandidateUnlandedCount": int(np.count_nonzero(unlanded)),
            "structureCandidateUnlandedRatio": float(np.count_nonzero(unlanded) / max(1, len(meta_points))),
        },
        "trustedRegion": trusted_region_report,
        "layeredEvidence": layered_evidence_report,
        "patchCorrection": patch_report,
        "supportCount": distribution(support_count),
        "stableCount": distribution(stable_count),
        "stableRatio": distribution(stable_ratio),
        "riskRatio": distribution(risk_ratio),
        "rawEvidenceScore": distribution(layered_evidence["score"]),
        "offsetMeanMeters": distribution(offset_mean[supported]),
        "offsetStdMeters": distribution(offset_std[supported]),
        "nearestRawDistanceMeters": distribution(nearest_distance),
        "appliedCorrectionMeters": distribution(smoothed_offsets[correction_mask]),
        "interpretation": [
            "meta_correction_field_all.ply colors the complete Meta surface by Raw support status.",
            "meta_corrected_surface_full.ply keeps the full Meta surface; supported areas are moved by Raw correction, unsupported/risk areas stay at their Meta positions.",
            "meta_corrected_surface.ply keeps Meta continuity but moves supported areas along Meta normals using local Raw offset statistics.",
            "meta_patch_corrected_surface_full.ply applies one correction offset per local Meta patch, testing slab-like surface correction instead of point-by-point correction.",
            "meta_patch_offset_heatmap.ply colors full Meta by signed patch correction: blue/cyan pulls one way along the Meta normal, red/orange pulls the other way, white is near zero, dark is not corrected.",
            "meta_patch_applied_only.ply shows only the Meta samples that received a local patch correction.",
            "meta_raw_landed_surface.ply is the trusted Raw-position landing layer. Meta-only candidates are excluded from this file.",
            "meta_raw_landed_full_with_candidates.ply keeps a complete view: landed samples move to Raw-corrected positions, unlanded samples remain at Meta candidate positions and are colored by decision state.",
            "meta_structure_candidate_unlanded.ply is the not-yet-real Meta structure candidate layer.",
            "trusted_raw_surface.ply is the architecture-level trusted region: Meta structure with Raw-confirmed positions.",
            "meta_correctable_surface.ply shows Meta samples that are eligible for Raw position correction.",
            "unconfirmed_candidate_surface.ply shows Meta structure that remains useful but not yet Raw-confirmed.",
            "meta_unsupported.ply is Meta-only completion: visually complete but not directly confirmed by Raw coverage.",
            "meta_risk_boundary.ply marks areas where Raw support is noisy, risky, or too far from Meta.",
            "layered_evidence_surface.ply keeps green trusted, cyan probable, yellow candidate, blue weak, and red rejected evidence separate instead of collapsing everything into trusted/unconfirmed.",
        ],
    }
    (out_dir / "meta_surface_correction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({
        "rawVoxelCount": report["rawVoxelCount"],
        "metaSampleCount": report["metaSampleCount"],
        "counts": report["counts"],
        "ratios": report["ratios"],
        "outputDirectory": str(out_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
