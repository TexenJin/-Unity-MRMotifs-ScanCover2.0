#!/usr/bin/env python3
"""Combine multiple ScanCover Quest3 observation-stat sessions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ScanCoverQuest3CloneDataManifest import (
    DEFAULT_MANIFEST,
    derived_output,
    load_manifest,
    observation_stats_paths,
)


DISTANCE_BIN_ORDER = [
    "0.0-0.5m",
    "0.5-1.0m",
    "1.0-1.5m",
    "1.5-2.0m",
    "2.0-3.0m",
    "3.0-5.0m",
    "5.0-8.0m",
    "8.0m+",
]

ANGLE_BIN_ORDER = [
    "0-20deg",
    "20-40deg",
    "40-60deg",
    "60-75deg",
    "75deg+",
]

CORE_TARGET_METRICS = [
    ("stableMainFrames", "targetStableMainFrames", "stable main"),
    ("topCoverageFrames", "targetTopCoverageFrames", "top coverage"),
    ("middleCoverageFrames", "targetMiddleCoverageFrames", "middle coverage"),
    ("bottomCoverageFrames", "targetBottomCoverageFrames", "bottom coverage"),
    ("nearDistanceFrames", "targetNearDistanceFrames", "near distance"),
    ("midDistanceFrames", "targetMidDistanceFrames", "mid distance"),
    ("farDistanceFrames", "targetFarDistanceFrames", "far distance"),
    ("frontAngleFrames", "targetFrontAngleFrames", "front angle"),
    ("obliqueAngleFrames", "targetObliqueAngleFrames", "oblique angle"),
    ("extremeAngleFrames", "targetExtremeAngleFrames", "extreme angle"),
    ("riskLayerFrames", "targetRiskLayerFrames", "risk layer"),
]


QUALITY_THRESHOLDS = {
    "minFrames": 240,
    "minPoints": 1_000_000,
    "minTargetFrames": 60,
    "minDistanceBinCounts": {
        "0.0-0.5m": 5_000,
        "0.5-1.0m": 100_000,
        "1.0-1.5m": 200_000,
        "1.5-2.0m": 150_000,
        "2.0-3.0m": 100_000,
        "3.0-5.0m": 100_000,
    },
    "minAngleBinCounts": {
        "40-60deg": 200_000,
        "60-75deg": 100_000,
        "75deg+": 30_000,
    },
    "warnDistanceBinCounts": {
        "5.0-8.0m": 10_000,
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header_index = next(
        (i for i, line in enumerate(lines) if line and not line.startswith("#") and "," in line),
        None,
    )
    if header_index is None:
        return []
    return list(csv.DictReader(lines[header_index:]))


def to_int(value: str | None) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


def to_float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def weighted_average(rows: list[dict[str, str]], value_key: str, count_key: str) -> float:
    total = 0
    value_sum = 0.0
    for row in rows:
        count = to_int(row.get(count_key))
        total += count
        value_sum += to_float(row.get(value_key)) * count
    return value_sum / total if total else 0.0


def aggregate_bins(rows: list[dict[str, str]], order: list[str]) -> list[dict[str, Any]]:
    bins: dict[str, dict[str, float]] = {}
    for label in order:
        bins[label] = {
            "count": 0,
            "viewDepth": 0.0,
            "distance": 0.0,
            "angle": 0.0,
            "boundary": 0,
            "crease": 0,
            "risk": 0,
        }

    for row in rows:
        label = row.get("bin", "")
        if label not in bins:
            bins[label] = {
                "count": 0,
                "viewDepth": 0.0,
                "distance": 0.0,
                "angle": 0.0,
                "boundary": 0,
                "crease": 0,
                "risk": 0,
            }
        data = bins[label]
        count = to_int(row.get("count"))
        data["count"] += count
        data["viewDepth"] += to_float(row.get("avgViewDepth")) * count
        data["distance"] += to_float(row.get("avgEuclideanDistance")) * count
        data["angle"] += to_float(row.get("avgViewAngleDeg")) * count
        data["boundary"] += to_int(row.get("boundaryRiskCount"))
        data["crease"] += to_int(row.get("creaseRiskCount"))
        data["risk"] += to_int(row.get("anyRiskCount"))

    labels = [label for label in order if label in bins]
    labels.extend(label for label in bins if label not in labels)

    result: list[dict[str, Any]] = []
    for label in labels:
        data = bins[label]
        count = int(data["count"])
        result.append(
            {
                "bin": label,
                "count": count,
                "avgViewDepth": data["viewDepth"] / count if count else 0.0,
                "avgEuclideanDistance": data["distance"] / count if count else 0.0,
                "avgViewAngleDeg": data["angle"] / count if count else 0.0,
                "riskRatio": data["risk"] / count if count else 0.0,
                "boundaryRiskRatio": data["boundary"] / count if count else 0.0,
                "creaseRiskRatio": data["crease"] / count if count else 0.0,
            }
        )
    return result


def summarize_target_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    empty_core = {key: 0 for key, _, _ in CORE_TARGET_METRICS}
    if not rows:
        return {
            "available": False,
            "attempts": 0,
            "accepted": 0,
            "rejected": 0,
            "targetFarFrames": 0,
            "targetHighAngleFrames": 0,
            "targetNearEdgeFrames": 0,
            "coreMetricTarget": 0,
            "completedCoreMetrics": 0,
            "coreMetrics": empty_core,
        }

    accepted = sum(1 for row in rows if to_int(row.get("accepted")) == 1)
    rejected = len(rows) - accepted
    last = rows[-1]
    core_metrics = {
        key: to_int(last.get(csv_key))
        for key, csv_key, _ in CORE_TARGET_METRICS
    }
    return {
        "available": True,
        "attempts": len(rows),
        "accepted": accepted,
        "rejected": rejected,
        "targetFarFrames": to_int(last.get("targetFarFrames")),
        "targetHighAngleFrames": to_int(last.get("targetHighAngleFrames")),
        "targetNearEdgeFrames": to_int(last.get("targetNearEdgeFrames")),
        "coreMetricTarget": to_int(last.get("coreMetricTarget")),
        "completedCoreMetrics": to_int(last.get("completedCoreMetrics")),
        "coreMetrics": core_metrics,
        "lastReason": last.get("reason", ""),
    }


def by_bin(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("bin", "")): row for row in rows}


def make_quality_check(
    category: str,
    name: str,
    value: int | float,
    required: int | float,
    unit: str,
    status: str,
    note: str,
) -> dict[str, Any]:
    return {
        "category": category,
        "name": name,
        "value": value,
        "required": required,
        "unit": unit,
        "status": status,
        "note": note,
    }


def pass_or_missing(value: int | float, required: int | float) -> str:
    return "pass" if value >= required else "missing"


def build_quality_report(summary: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    thresholds = QUALITY_THRESHOLDS
    distance_bins = by_bin(summary["distanceBins"])
    angle_bins = by_bin(summary["angleBins"])
    gate_sessions = summary.get("targetGateSessions", [])
    core_metric_target = max(
        (int(session.get("coreMetricTarget", 0)) for session in gate_sessions),
        default=0,
    )
    core_metric_totals = {
        key: sum(int(session.get("coreMetrics", {}).get(key, 0)) for session in gate_sessions)
        for key, _, _ in CORE_TARGET_METRICS
    }
    core_metric_min = min(core_metric_totals.values()) if core_metric_totals else 0
    core_metric_complete = core_metric_target > 0 and all(
        value >= core_metric_target for value in core_metric_totals.values()
    )

    checks.append(
        make_quality_check(
            "overall",
            "accepted frames",
            summary["frames"],
            thresholds["minFrames"],
            "frames",
            pass_or_missing(summary["frames"], thresholds["minFrames"]),
            "Enough frames for an initial Quest3 observation-clone validation run.",
        )
    )
    checks.append(
        make_quality_check(
            "overall",
            "observed points",
            summary["points"],
            thresholds["minPoints"],
            "points",
            pass_or_missing(summary["points"], thresholds["minPoints"]),
            "Enough samples for offline fusion and teacher/student comparison.",
        )
    )

    if core_metric_target > 0:
        for key, _, label in CORE_TARGET_METRICS:
            value = core_metric_totals[key]
            checks.append(
                make_quality_check(
                    "core target gate",
                    label,
                    value,
                    core_metric_target,
                    "frames",
                    pass_or_missing(value, core_metric_target),
                    "Required multi-core observation coverage for strengthening the Quest3 observation clone.",
                )
            )

    for label, minimum in thresholds["minDistanceBinCounts"].items():
        count = int(distance_bins.get(label, {}).get("count", 0))
        checks.append(
            make_quality_check(
                "distance",
                label,
                count,
                minimum,
                "points",
                pass_or_missing(count, minimum),
                "Required 0.3m-5m working-range coverage.",
            )
        )

    for label, minimum in thresholds["warnDistanceBinCounts"].items():
        count = int(distance_bins.get(label, {}).get("count", 0))
        checks.append(
            make_quality_check(
                "distance",
                label,
                count,
                minimum,
                "points",
                "warn" if count < minimum else "pass",
                "Not required for the current 5m target, but useful for future wider-room generalization.",
            )
        )

    for label, minimum in thresholds["minAngleBinCounts"].items():
        count = int(angle_bins.get(label, {}).get("count", 0))
        checks.append(
            make_quality_check(
                "view angle",
                label,
                count,
                minimum,
                "points",
                pass_or_missing(count, minimum),
                "Required to model normal view, oblique view, and extreme oblique-view instability.",
            )
        )

    if gate_sessions:
        far = max(int(session.get("targetFarFrames", 0)) for session in gate_sessions)
        high = max(int(session.get("targetHighAngleFrames", 0)) for session in gate_sessions)
        near = max(int(session.get("targetNearEdgeFrames", 0)) for session in gate_sessions)
    else:
        far = high = near = 0

    for name, value in [
        ("target far stable frames", far),
        ("target high-angle frames", high),
        ("target near-edge frames", near),
    ]:
        checks.append(
            make_quality_check(
                "target gate",
                name,
                value,
                thresholds["minTargetFrames"],
                "frames",
                pass_or_missing(value, thresholds["minTargetFrames"]),
                "Target-gated session coverage for the Quest3 observation-clone edge cases.",
            )
        )

    if core_metric_complete:
        for check in checks:
            if check["status"] == "missing" and check["category"] in {"overall", "distance", "view angle", "target gate"}:
                check["status"] = "warn"
                check["note"] = (
                    "Legacy broad-training threshold is not met, but the current 11-metric clone-refresh gate is complete. "
                    + check["note"]
                )

    hard_failures = [check for check in checks if check["status"] == "missing"]
    warnings = [check for check in checks if check["status"] == "warn"]
    enough_for_clone_refresh = core_metric_complete
    enough_for_initial_validation = not hard_failures and (
        core_metric_complete or summary["points"] >= thresholds["minPoints"]
    )

    missing = [
        {
            "category": check["category"],
            "name": check["name"],
            "value": check["value"],
            "required": check["required"],
            "unit": check["unit"],
        }
        for check in hard_failures
    ]

    recommended_next_actions: list[str] = []
    if enough_for_initial_validation:
        recommended_next_actions.append(
            "Proceed to offline fusion and teacher/student validation using the combined multi-frame session data."
        )
        recommended_next_actions.append(
            "Keep 5m as the current effective working distance; record 5m+ only as optional diagnostic data."
        )
    else:
        recommended_next_actions.append(
            "Collect another target-gated session before offline fusion; prioritize the missing categories listed in this report."
        )

    if warnings:
        recommended_next_actions.append(
            "5m+ samples are weak; this is acceptable for the current room-scale clone, but should be expanded later for larger rooms."
        )

    return {
        "purpose": "Quest3 observation-clone readiness report",
        "verdict": {
            "enoughForQuest3CloneMetricRefresh": enough_for_clone_refresh,
            "enoughForInitialOfflineFusionAndStudentValidation": enough_for_initial_validation,
            "enoughForGeneralTraining": False,
            "summary": (
                "PASS: enough for the next offline fusion / teacher-student validation step."
                if enough_for_initial_validation
                else "MISSING: collect more data before the next offline validation step."
            ),
            "generalTrainingNote": (
                "This is still one-room empirical data. It is useful for validating the Quest3 observation-clone route, "
                "not enough to claim broad room-general training coverage."
            ),
        },
        "coreTargetGate": {
            "available": core_metric_target > 0,
            "coreMetricTarget": core_metric_target,
            "completedCoreMetrics": sum(1 for value in core_metric_totals.values() if core_metric_target > 0 and value >= core_metric_target),
            "coreMetricCount": len(CORE_TARGET_METRICS),
            "minCoreMetricFrames": core_metric_min,
            "metrics": core_metric_totals,
        },
        "thresholds": thresholds,
        "checks": checks,
        "missingRequiredCategories": missing,
        "warnings": [
            {
                "category": check["category"],
                "name": check["name"],
                "value": check["value"],
                "recommended": check["required"],
                "unit": check["unit"],
                "note": check["note"],
            }
            for check in warnings
        ],
        "recommendedNextActions": recommended_next_actions,
    }


def write_quality_checks(path: Path, checks: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "name", "value", "required", "unit", "status", "note"])
        for check in checks:
            writer.writerow(
                [
                    check["category"],
                    check["name"],
                    check["value"],
                    check["required"],
                    check["unit"],
                    check["status"],
                    check["note"],
                ]
            )


def write_quality_markdown(path: Path, report: dict[str, Any]) -> None:
    verdict = report["verdict"]
    lines = [
        "# ScanCover Quest3 Observation Quality Report",
        "",
        f"- Purpose: {report['purpose']}",
        f"- Verdict: {verdict['summary']}",
        f"- Enough for Quest3 clone metric refresh: {verdict['enoughForQuest3CloneMetricRefresh']}",
        f"- Enough for initial offline fusion / student validation: {verdict['enoughForInitialOfflineFusionAndStudentValidation']}",
        f"- Enough for general training: {verdict['enoughForGeneralTraining']}",
        f"- Note: {verdict['generalTrainingNote']}",
        "",
        "## Checks",
        "",
        "| Category | Name | Value | Required | Unit | Status |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for check in report["checks"]:
        lines.append(
            f"| {check['category']} | {check['name']} | {check['value']} | "
            f"{check['required']} | {check['unit']} | {check['status']} |"
        )

    lines.extend(["", "## Next Actions", ""])
    for action in report["recommendedNextActions"]:
        lines.append(f"- {action}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_session(session: Path) -> dict[str, Any]:
    stats_dir = session / "observation_stats"
    frame_rows = read_csv(stats_dir / "frame_observation_stats.csv")
    distance_rows = read_csv(stats_dir / "distance_bins.csv")
    angle_rows = read_csv(stats_dir / "angle_bins.csv")
    edge_rows = read_csv(stats_dir / "edge_risk_stats.csv")
    target_rows = read_csv(stats_dir / "target_gate.csv")

    total_points = sum(to_int(row.get("pointCount")) for row in frame_rows)
    total_boundary = sum(to_int(row.get("boundaryRiskCount")) for row in edge_rows)
    total_crease = sum(to_int(row.get("creaseRiskCount")) for row in edge_rows)
    total_risk = sum(to_int(row.get("anyRiskCount")) for row in edge_rows)

    return {
        "session": str(session),
        "name": session.name,
        "frameRows": frame_rows,
        "distanceRows": distance_rows,
        "angleRows": angle_rows,
        "edgeRows": edge_rows,
        "targetGate": summarize_target_gate(target_rows),
        "frames": len(frame_rows),
        "points": total_points,
        "boundaryRiskCount": total_boundary,
        "creaseRiskCount": total_crease,
        "anyRiskCount": total_risk,
    }


def write_session_contributions(path: Path, sessions: list[dict[str, Any]], total_points: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "session",
                "frames",
                "points",
                "pointShare",
                "boundaryRiskRatio",
                "creaseRiskRatio",
                "anyRiskRatio",
                "targetGateAvailable",
                "targetAttempts",
                "targetAccepted",
                "targetRejected",
                "targetFarFrames",
                "targetHighAngleFrames",
                "targetNearEdgeFrames",
                "coreMetricTarget",
                "completedCoreMetrics",
                *[key for key, _, _ in CORE_TARGET_METRICS],
            ]
        )
        for session in sessions:
            points = session["points"]
            gate = session["targetGate"]
            core_metrics = gate.get("coreMetrics", {})
            writer.writerow(
                [
                    session["name"],
                    session["frames"],
                    points,
                    points / total_points if total_points else 0.0,
                    session["boundaryRiskCount"] / points if points else 0.0,
                    session["creaseRiskCount"] / points if points else 0.0,
                    session["anyRiskCount"] / points if points else 0.0,
                    gate["available"],
                    gate["attempts"],
                    gate["accepted"],
                    gate["rejected"],
                    gate["targetFarFrames"],
                    gate["targetHighAngleFrames"],
                    gate["targetNearEdgeFrames"],
                    gate.get("coreMetricTarget", 0),
                    gate.get("completedCoreMetrics", 0),
                    *[core_metrics.get(key, 0) for key, _, _ in CORE_TARGET_METRICS],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", type=Path, nargs="*", help="Observation session directories to combine.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Quest3 clone data manifest. When set, default-use observation sessions are read from it.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Defaults to ScanCoverExports/Quest3ObservationStatsSessions/CombinedObservationStats.",
    )
    args = parser.parse_args()

    manifest: dict[str, Any] | None = None
    input_sessions = list(args.sessions)
    if args.manifest is not None:
        manifest = load_manifest(args.manifest)
        if not input_sessions:
            input_sessions = observation_stats_paths(manifest)
    if not input_sessions:
        raise SystemExit(
            f"No observation sessions supplied. Pass session paths or use --manifest {DEFAULT_MANIFEST}."
        )

    sessions = [summarize_session(path) for path in input_sessions]
    total_points = sum(session["points"] for session in sessions)
    total_frames = sum(session["frames"] for session in sessions)
    total_boundary = sum(session["boundaryRiskCount"] for session in sessions)
    total_crease = sum(session["creaseRiskCount"] for session in sessions)
    total_risk = sum(session["anyRiskCount"] for session in sessions)

    all_distance_rows = [row for session in sessions for row in session["distanceRows"]]
    all_angle_rows = [row for session in sessions for row in session["angleRows"]]

    summary = {
        "sessions": [session["session"] for session in sessions],
        "sessionCount": len(sessions),
        "frames": total_frames,
        "points": total_points,
        "avgViewDepth": weighted_average(all_distance_rows, "avgViewDepth", "count"),
        "avgEuclideanDistance": weighted_average(all_distance_rows, "avgEuclideanDistance", "count"),
        "avgViewAngleDeg": weighted_average(all_angle_rows, "avgViewAngleDeg", "count"),
        "boundaryRiskRatio": total_boundary / total_points if total_points else 0.0,
        "creaseRiskRatio": total_crease / total_points if total_points else 0.0,
        "anyRiskRatio": total_risk / total_points if total_points else 0.0,
        "distanceBins": aggregate_bins(all_distance_rows, DISTANCE_BIN_ORDER),
        "angleBins": aggregate_bins(all_angle_rows, ANGLE_BIN_ORDER),
        "targetGateSessions": [
            {
                "session": session["name"],
                **session["targetGate"],
            }
            for session in sessions
            if session["targetGate"]["available"]
        ],
        "sessionContributions": [
            {
                "session": session["name"],
                "frames": session["frames"],
                "points": session["points"],
                "pointShare": session["points"] / total_points if total_points else 0.0,
                "boundaryRiskRatio": session["boundaryRiskCount"] / session["points"] if session["points"] else 0.0,
                "creaseRiskRatio": session["creaseRiskCount"] / session["points"] if session["points"] else 0.0,
                "anyRiskRatio": session["anyRiskCount"] / session["points"] if session["points"] else 0.0,
            }
            for session in sessions
        ],
    }

    quality_report = build_quality_report(summary)
    summary["qualityReport"] = quality_report

    out_dir = args.out
    if out_dir is None:
        manifest_out = derived_output(manifest, "combinedObservationStatsDir") if manifest else None
        if manifest_out is not None:
            out_dir = manifest_out
        else:
            common_parent = input_sessions[0].parent if input_sessions else Path.cwd()
            out_dir = common_parent / "CombinedObservationStats"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "combined_observation_summary.json"
    contribution_path = out_dir / "combined_session_contributions.csv"
    quality_path = out_dir / "quality_report.json"
    quality_checks_path = out_dir / "quality_report_checks.csv"
    quality_markdown_path = out_dir / "quality_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_session_contributions(contribution_path, sessions, total_points)
    quality_path.write_text(json.dumps(quality_report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_quality_checks(quality_checks_path, quality_report["checks"])
    write_quality_markdown(quality_markdown_path, quality_report)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote: {summary_path}")
    print(f"Wrote: {contribution_path}")
    print(f"Wrote: {quality_path}")
    print(f"Wrote: {quality_checks_path}")
    print(f"Wrote: {quality_markdown_path}")


if __name__ == "__main__":
    main()
