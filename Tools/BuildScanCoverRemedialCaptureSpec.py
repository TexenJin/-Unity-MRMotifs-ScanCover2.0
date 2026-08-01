#!/usr/bin/env python3
"""Turn frozen offline ledgers and four Quest sessions into a capture spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-evidence-report", type=Path, required=True)
    parser.add_argument("--degradation-model", type=Path, required=True)
    parser.add_argument("--normal-supply-report", type=Path, required=True)
    parser.add_argument("--normal-tolerance-report", type=Path, required=True)
    parser.add_argument("--interaction-report", type=Path, required=True)
    parser.add_argument("--ideal-dmc-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    evidence = load(args.full_evidence_report)
    model = load(args.degradation_model)
    normal_supply_report = load(args.normal_supply_report)
    tolerance = load(args.normal_tolerance_report)
    interaction = load(args.interaction_report)
    ideal_dmc = load(args.ideal_dmc_report)

    condition_means = tolerance["means"]
    condition_gates = tolerance["perConditionAllRoomsPassed"]
    # Explicit axis names avoid treating edge-combination arms as global axes.
    safe_global_conditions = [
        name for name in ("G05", "G07_5", "G10", "G20")
        if condition_gates.get(name, False)
    ]
    safe_dropout_conditions = [
        name for name in ("D05", "D07_5", "D10", "D20")
        if condition_gates.get(name, False)
    ]
    global_rms_limit = max(
        condition_means[name]["angularRmsDegrees"] for name in safe_global_conditions
    )
    dropout_limit = max(
        condition_means[name]["dropoutRatio"] for name in safe_dropout_conditions
    )
    edge_safe = condition_means["E10_ED10"]
    edge_fail = condition_means["E15_ED17_5"]

    total_profile_points = sum(int(row["points"]) for row in model["distanceProfile"])
    distance_point_floor = max(50000, int(round(total_profile_points * 0.02)))
    joint_point_floor = max(2500, int(round(total_profile_points * 0.001)))
    distance_rows: list[dict[str, Any]] = []
    for row in model["distanceProfile"]:
        points = int(row["points"])
        distance_rows.append(
            {
                "minDepthMeters": row["minDepthMeters"],
                "maxDepthMeters": row["maxDepthMeters"],
                "points": points,
                "pointFloor": distance_point_floor,
                "coverageDeficitPoints": max(0, distance_point_floor - points),
                "normalDeltaP95Degrees": row["normalDeltaDegrees"]["p95"],
                "edgeRiskRatio": row["edgeRiskRatio"],
                "stereoRepeatabilityP95Meters": row["stereoRepeatabilityMeters"]["p95"],
                "needsMoreCoverage": points < distance_point_floor,
            }
        )

    angle_rows: list[dict[str, Any]] = []
    for row in model["angleProfile"]:
        normal_p95 = float(row["normalDeltaDegrees"]["p95"])
        angle_rows.append(
            {
                "minAngleDegrees": row["minAngleDegrees"],
                "maxAngleDegrees": row["maxAngleDegrees"],
                "points": int(row["points"]),
                "normalDeltaP95Degrees": normal_p95,
                "edgeRiskRatio": row["edgeRiskRatio"],
                "stereoRepeatabilityP95Meters": row["stereoRepeatabilityMeters"]["p95"],
                "instabilityTrigger": normal_p95 > global_rms_limit,
                "triggerSemantics": (
                    "raw-vs-filtered p95 is a self-consistency trigger, not truth angular error"
                ),
            }
        )

    joint_rows: list[dict[str, Any]] = []
    for row in model["distanceAngleProfile"]:
        points = int(row["points"])
        normal_p95 = float(row["normalDeltaDegrees"]["p95"])
        joint_rows.append(
            {
                "minDepthMeters": row["minDepthMeters"],
                "maxDepthMeters": row["maxDepthMeters"],
                "minAngleDegrees": row["minAngleDegrees"],
                "maxAngleDegrees": row["maxAngleDegrees"],
                "points": points,
                "pointFloor": joint_point_floor,
                "coverageDeficitPoints": max(0, joint_point_floor - points),
                "normalDeltaP95Degrees": normal_p95,
                "edgeRiskRatio": row["edgeRiskRatio"],
                "stereoRepeatabilityP95Meters": row["stereoRepeatabilityMeters"]["p95"],
                "needsMoreCoverage": points < joint_point_floor,
                "instabilityTrigger": normal_p95 > global_rms_limit,
            }
        )

    normal_supply = normal_supply_report["normalSupply"]
    sessions = []
    for row in evidence["sessions"]:
        sessions.append(
            {
                "session": Path(row["session"]).name,
                "frames": row["manifestFrames"],
                "statisticsFrames": row["statisticsFrames"],
                "poseStepP50Meters": row["poseStepMeters"]["p50"],
                "poseStepP95Meters": row["poseStepMeters"]["p95"],
                "rotationStepP50Degrees": row["rotationStepDegrees"]["p50"],
                "rotationStepP95Degrees": row["rotationStepDegrees"]["p95"],
            }
        )

    far_deficit = next(
        row for row in distance_rows
        if row["minDepthMeters"] == 4.0 and row["maxDepthMeters"] == 5.0
    )
    grazing = next(
        row for row in angle_rows
        if row["minAngleDegrees"] == 75.0
    )
    required_joint = [
        row for row in joint_rows
        if row["needsMoreCoverage"] or row["instabilityTrigger"]
    ]
    output = {
        "schema": "scancover.remedial_capture_spec.v1",
        "scope": "formal targeted supplement derived from four frozen Quest sessions and offline tolerance/DMC ledgers",
        "existingEvidence": {
            "sessions": sessions,
            "totalFrames": sum(row["frames"] for row in sessions),
            "selectedRawDuplicates": evidence["duplicateSelectedRawHashes"],
            "sensorDropoutRatio": evidence["globalValidity"]["sensorDropoutRatio"],
            "workingRangeValidRatio": evidence["globalValidity"]["workingRangeValidRatio"],
            "rawNormalCoverageOfWorkingRange": normal_supply[
                "rawNormalCoverageOfWorkingRange"
            ],
            "filteredNormalCoverageOfWorkingRange": normal_supply[
                "filteredNormalCoverageOfWorkingRange"
            ],
            "normalSelfConsistencyProxy": normal_supply.get(
                "normalSelfConsistentCoverageOfWorkingRange",
                normal_supply.get("reliableNormalCoverageOfWorkingRange"),
            ),
            "claimBoundary": normal_supply["claimBoundary"],
            "crossImplementationGates": evidence["gates"],
            "keepExistingSessions": True,
            "fullRecaptureRequired": False,
        },
        "measuredTolerance": {
            "globalNormalAngularRmsDegreesMaximumPassing": global_rms_limit,
            "independentNormalDropoutMaximumPassing": dropout_limit,
            "edgeCombinationPassing": {
                "actualAngularRmsDegrees": edge_safe["angularRmsDegrees"],
                "actualTotalDropoutRatio": edge_safe["dropoutRatio"],
                "actualEdgeDropoutRatio": edge_safe["edgeDropoutRatio"],
            },
            "edgeCombinationFailing": {
                "actualAngularRmsDegrees": edge_fail["angularRmsDegrees"],
                "actualTotalDropoutRatio": edge_fail["dropoutRatio"],
                "actualEdgeDropoutRatio": edge_fail["edgeDropoutRatio"],
            },
        },
        "coverageAudit": {
            "distancePointFloor": distance_point_floor,
            "jointPointFloor": joint_point_floor,
            "distanceRows": distance_rows,
            "angleRows": angle_rows,
            "vulnerableJointRows": required_joint,
        },
        "requiredSupplement": [
            {
                "priority": 1,
                "category": "paired_grazing_anchor",
                "reason": (
                    "75-90 degree observations have high edge concentration and "
                    "raw/filter normal disagreement; more grazing-only frames would repeat bad evidence"
                ),
                "action": (
                    "For each grazing/occlusion-edge patch, acquire a paired 30-60 degree "
                    "anchor view and change side while keeping the same physical patch in view."
                ),
                "completion": (
                    "Every targeted patch has both an anchor sector and a grazing sector; "
                    "offline correspondence shows the anchor observation exists before the grazing sample is admitted."
                ),
            },
            {
                "priority": 2,
                "category": "far_4_to_5m",
                "reason": (
                    f"Only {far_deficit['points']} profiled samples versus a frozen "
                    f"{distance_point_floor} point floor; deficit {far_deficit['coverageDeficitPoints']}."
                ),
                "action": (
                    "Add 4-5 m observations across front/oblique sectors; avoid satisfying "
                    "the bucket with one repeated wall patch."
                ),
                "completion": (
                    f"The 4-5 m distance bucket reaches {distance_point_floor} profiled samples "
                    f"and each required distance-angle cell reaches {joint_point_floor}."
                ),
            },
            {
                "priority": 3,
                "category": "near_concave_convex_thin",
                "reason": (
                    "Replica ideal-input structure audit leaves the largest supply loss at "
                    "mesh boundaries and ambiguous/concave/convex creases; current Quest data "
                    "has no truth label proving these structures are independently covered."
                ),
                "action": (
                    "Scan near concave corners, convex outside corners, door/table thin edges, "
                    "and small depth steps from both sides, with an anchor view for each side."
                ),
                "completion": (
                    "Offline automatic edge/multi-normal classification observes both incident "
                    "surface families with valid stereo depth; no human labels are used."
                ),
            },
        ],
        "doNotSupplementBlindly": [
            "1.0-3.0 m generic front-facing walls are already the dominant bins.",
            "Do not add grazing-only footage without paired anchor views.",
            "Do not repeat a fixed seat/fixed rotation path merely to increase frame count.",
        ],
        "offlineCodeBeforeUnityAdmission": {
            "required": True,
            "inputRecoverableMeanPercentagePoints": interaction["responsibility"][
                "captureAndNormalGeneration"
            ]["meanRecoverablePercentagePoints"],
            "idealResidualMeanPercentagePoints": interaction["responsibility"][
                "offlineFusionCode"
            ]["meanResidualPercentagePoints"],
            "primaryCodeStage": interaction["responsibility"]["offlineFusionCode"][
                "primaryStage"
            ],
            "idealStructureMeans": ideal_dmc["structureMeans"],
            "order": [
                "repair ideal-input incomplete-corner and supported-no-zero supply",
                "rerun ideal and Quest interaction ledgers",
                "then activate the targeted supplement progress gates in Unity",
            ],
        },
        "admissionRules": {
            "automaticOnly": True,
            "humanAnnotations": False,
            "completeStereoFrameRequired": True,
            "bufferDimensionsMustMatch": True,
            "normalSelfConsistencyIsNotTruth": True,
            "timeLimitIsSafetyOnly": True,
            "stopCondition": "all required category progress gates pass, not elapsed time",
        },
    }
    (args.out / "remedial_capture_spec.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# ScanCover targeted remedial capture specification",
        "",
        f"- Existing sessions: {len(sessions)}; frames: {output['existingEvidence']['totalFrames']}; keep all four.",
        f"- Full recapture required: {output['existingEvidence']['fullRecaptureRequired']}.",
        f"- Passing normal envelope: RMS <= {global_rms_limit:.2f} degrees; independent dropout <= {dropout_limit * 100:.2f}%.",
        f"- Existing sensor dropout: {output['existingEvidence']['sensorDropoutRatio'] * 100:.3f}% (passes dropout envelope).",
        "- Raw/filter self-consistency is retained only as a trigger; it is not truth-normal accuracy.",
        "",
        "## Required supplement",
        "",
        "| Priority | Category | Why | Completion |",
        "| ---: | --- | --- | --- |",
    ]
    for row in output["requiredSupplement"]:
        lines.append(
            f"| {row['priority']} | {row['category']} | {row['reason']} | {row['completion']} |"
        )
    lines.extend(
        [
            "",
            "## Do not collect blindly",
            "",
            *[f"- {item}" for item in output["doNotSupplementBlindly"]],
            "",
            "## Required order",
            "",
            *[
                f"{index}. {item}"
                for index, item in enumerate(
                    output["offlineCodeBeforeUnityAdmission"]["order"], start=1
                )
            ],
            "",
            "## Admission rules",
            "",
            "- Automatic classification only; no human labels.",
            "- Incomplete stereo frames or mismatched buffers fail the round.",
            "- Elapsed time is only a safety timeout; completion is category progress.",
        ]
    )
    (args.out / "remedial_capture_spec.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "fullRecaptureRequired": False,
                "globalNormalRmsLimitDegrees": global_rms_limit,
                "normalDropoutLimit": dropout_limit,
                "farDistanceDeficitPoints": far_deficit["coverageDeficitPoints"],
                "grazingNormalDeltaP95Degrees": grazing["normalDeltaP95Degrees"],
                "out": str(args.out),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
