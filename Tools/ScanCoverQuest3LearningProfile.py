#!/usr/bin/env python3
"""Build a compact Quest3 observation learning profile from ScanCover stats.

The profile is intentionally not a room model. It is a device-observation
profile: distance bands, view-angle bands, and risk/dropout priors that can be
applied to synthetic rooms or offline teacher/student validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ScanCoverQuest3CloneDataManifest import DEFAULT_MANIFEST, derived_output, load_manifest


DEFAULT_SUMMARY = Path(
    r"D:\PCA\Unity-MRMotifs-ScanCover-main\ScanCoverExports"
    r"\Quest3ObservationStatsSessions\CombinedObservationStats\combined_observation_summary.json"
)

CORE_TARGET_METRICS = [
    ("stableMainFrames", "stable main"),
    ("topCoverageFrames", "top coverage"),
    ("middleCoverageFrames", "middle coverage"),
    ("bottomCoverageFrames", "bottom coverage"),
    ("nearDistanceFrames", "near distance"),
    ("midDistanceFrames", "mid distance"),
    ("farDistanceFrames", "far distance"),
    ("frontAngleFrames", "front angle"),
    ("obliqueAngleFrames", "oblique angle"),
    ("extremeAngleFrames", "extreme angle"),
    ("riskLayerFrames", "risk layer"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Quest3 clone data manifest. When set, the combined summary and output path are read from it.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-working-distance", type=float, default=5.0)
    parser.add_argument("--min-bin-points", type=int, default=5000)
    return parser.parse_args()


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def parse_distance_label(label: str) -> tuple[float, float]:
    if label.endswith("m+"):
        return float(label[:-2]), float("inf")
    lo, hi = label[:-1].split("-")
    return float(lo), float(hi)


def normalize_rows(rows: list[dict[str, Any]], total: int) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        count = int(row.get("count", 0))
        risk = float(row.get("riskRatio", 0.0))
        boundary = float(row.get("boundaryRiskRatio", 0.0))
        crease = float(row.get("creaseRiskRatio", 0.0))
        normalized.append(
            {
                "bin": row["bin"],
                "count": count,
                "share": count / max(1, total),
                "riskRatio": risk,
                "boundaryRiskRatio": boundary,
                "creaseRiskRatio": crease,
                "avgViewDepth": float(row.get("avgViewDepth", 0.0)),
                "avgEuclideanDistance": float(row.get("avgEuclideanDistance", 0.0)),
                "avgViewAngleDeg": float(row.get("avgViewAngleDeg", 0.0)),
                "confidence": clamp(count / 100000.0, 0.0, 1.0),
            }
        )
    return normalized


def build_profile(summary: dict[str, Any], max_working_distance: float, min_bin_points: int) -> dict[str, Any]:
    distance_rows = []
    excluded_rows = []
    for row in summary.get("distanceBins", []):
        lo, hi = parse_distance_label(row["bin"])
        if lo >= max_working_distance:
            excluded_rows.append(row)
            continue
        if hi > max_working_distance:
            excluded_rows.append(row)
            continue
        distance_rows.append(row)

    working_point_count = sum(int(row.get("count", 0)) for row in distance_rows)
    angle_rows = list(summary.get("angleBins", []))
    angle_point_count = sum(int(row.get("count", 0)) for row in angle_rows)

    weak_distance_bins = [
        row["bin"]
        for row in distance_rows
        if int(row.get("count", 0)) < min_bin_points
    ]
    excluded_distance_bins = [row["bin"] for row in excluded_rows]

    target_sessions = summary.get("targetGateSessions", [])
    core_metric_target = max(
        (int(item.get("coreMetricTarget", 0)) for item in target_sessions),
        default=0,
    )
    core_metrics = {
        key: sum(int(item.get("coreMetrics", {}).get(key, 0)) for item in target_sessions)
        for key, _ in CORE_TARGET_METRICS
    }
    core_metric_complete = core_metric_target > 0 and all(
        value >= core_metric_target for value in core_metrics.values()
    )
    target_gate = {
        "farStableFrames": sum(int(item.get("targetFarFrames", 0)) for item in target_sessions),
        "highAngleFrames": sum(int(item.get("targetHighAngleFrames", 0)) for item in target_sessions),
        "nearEdgeFrames": sum(int(item.get("targetNearEdgeFrames", 0)) for item in target_sessions),
        "coreMetricTarget": core_metric_target,
        "completedCoreMetrics": sum(1 for value in core_metrics.values() if core_metric_target > 0 and value >= core_metric_target),
        "coreMetricCount": len(CORE_TARGET_METRICS),
        "minCoreMetricFrames": min(core_metrics.values()) if core_metrics else 0,
        "coreMetrics": core_metrics,
    }
    enough_for_clone_refresh = (
        core_metric_complete
        and working_point_count >= 250000
        and angle_point_count >= 250000
    )

    risk_mean = float(summary.get("anyRiskRatio", 0.0))
    profile = {
        "schema": "ScanCoverQuest3LearningProfile/v2",
        "sourceSummary": summary.get("sessions", []),
        "purpose": "Quest3 observation clone profile for synthetic scan and student validation",
        "workingRangeMeters": {
            "min": 0.0,
            "max": max_working_distance,
            "excludedDistanceBins": excluded_distance_bins,
        },
        "sampleTotals": {
            "frames": int(summary.get("frames", 0)),
            "points": int(summary.get("points", 0)),
            "workingRangePoints": working_point_count,
            "anglePoints": angle_point_count,
        },
        "globalRisk": {
            "anyRiskRatio": risk_mean,
            "boundaryRiskRatio": float(summary.get("boundaryRiskRatio", 0.0)),
            "creaseRiskRatio": float(summary.get("creaseRiskRatio", 0.0)),
        },
        "distanceBins": normalize_rows(distance_rows, working_point_count),
        "angleBins": normalize_rows(angle_rows, angle_point_count),
        "targetGate": target_gate,
        "trainingWeights": {
            "distanceWeight": 0.45,
            "angleWeight": 0.35,
            "edgeRiskWeight": 0.20,
            "temporalStabilityWeight": 0.35,
            "outlierPenalty": 0.50,
            "coreTargetGateWeight": 0.45,
            "coverageBandWeight": 0.30,
            "riskLayerWeight": 0.25,
        },
        "readiness": {
            "enoughForQuest3CloneMetricRefresh": enough_for_clone_refresh,
            "enoughForInitialOfflineLearning": (
                enough_for_clone_refresh
                or (
                    int(summary.get("frames", 0)) >= 240
                    and working_point_count >= 1000000
                    and not weak_distance_bins
                    and target_gate["farStableFrames"] >= 60
                    and target_gate["highAngleFrames"] >= 60
                    and target_gate["nearEdgeFrames"] >= 60
                )
            ),
            "enoughForGeneralTraining": False,
            "weakDistanceBins": weak_distance_bins,
            "note": (
                "Use as a Quest3 observation/noise profile. Do not treat this as a room-structure model or final general-training dataset."
            ),
        },
    }
    return profile


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest) if args.manifest is not None else None
    summary_path = args.summary
    if summary_path is None and manifest is not None:
        summary_path = derived_output(manifest, "combinedObservationSummary")
    if summary_path is None:
        summary_path = DEFAULT_SUMMARY
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    profile = build_profile(summary, args.max_working_distance, args.min_bin_points)

    out = args.out
    if out is None:
        manifest_out = derived_output(manifest, "learningProfile") if manifest else None
        out = manifest_out or (summary_path.parent / "quest3_learning_profile.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(profile["readiness"], indent=2, ensure_ascii=False))
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
