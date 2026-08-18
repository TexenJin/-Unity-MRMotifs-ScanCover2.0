#!/usr/bin/env python3
"""Split input recovery from residual TSDF/DMC implementation loss."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any


ROOMS = ("office0", "office4", "room0")
RUN_ARMS = (
    ("B0_quest_estimated", False, "estimated"),
    ("B1_ideal_estimated", True, "estimated"),
    ("B2_quest_truth", False, "raycast-truth"),
)
ALL_ARMS = tuple(item[0] for item in RUN_ARMS) + ("B3_ideal_truth",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--degradation-model", type=Path, required=True)
    parser.add_argument("--ideal-ledger-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=112)
    parser.add_argument("--height", type=int, default=84)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_arm(
    args: argparse.Namespace,
    room: str,
    arm: str,
    ideal_depth: bool,
    normal_source: str,
) -> tuple[str, str, str]:
    output = args.out / room / arm
    report = output / "directional_composition_report.json"
    if report.exists() and not args.force:
        return room, arm, "reused"
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.python),
        str(args.validator),
        "--mesh",
        str(args.mesh_root / f"{room}.ply"),
        "--degradation-model",
        str(args.degradation_model),
        "--out",
        str(output),
        "--frames",
        "90",
        "--camera-path-checkpoints",
        "12",
        "24",
        "48",
        "90",
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--truth-samples",
        "30000",
        "--voxel",
        "0.045",
        "--sdf-trunc",
        "0.135",
        "--sample-stride",
        "1",
        "--integration-mode",
        "paper-normal-raycast",
        "--paper-normal-source",
        normal_source,
        "--paper-growth-ledger-only",
        "--paper-growth-stage-attribution",
        "--paper-growth-upstream-attribution",
    ]
    if ideal_depth:
        command.append("--ideal-depth")
    completed = subprocess.run(
        command,
        cwd=args.validator.parent.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0 or not report.exists():
        raise RuntimeError(f"{room}/{arm} failed; see {output / 'run.log'}")
    return room, arm, "ran"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def report_for(args: argparse.Namespace, room: str, arm: str) -> dict[str, Any]:
    if arm == "B3_ideal_truth":
        return load(args.ideal_ledger_root / room / "directional_composition_report.json")
    return load(args.out / room / arm / "directional_composition_report.json")


def metrics(report: dict[str, Any]) -> dict[str, Any]:
    checkpoint = report["checkpoints"][-1]
    stage = checkpoint["stageAttribution"]
    losses = stage["netStageLossAt0.05m"]
    supply = checkpoint["tsdfSupplyAttribution"]
    failures = supply["firstFailure"]
    return {
        "visible": checkpoint["exactVisibleCoverageAt0.05m"],
        "visibleSampledMesh": checkpoint["visibleCoverageAt0.05m"],
        "wholeRoom": checkpoint["exactWholeRoomCoverageAt0.05m"],
        "wholeRoomSampledMesh": checkpoint["wholeRoomCoverageAt0.05m"],
        "extra": checkpoint["extraSurfaceRatioAt0.05m"],
        "p95Meters": checkpoint["accuracyP95m"],
        "boundaryPerK": checkpoint["boundaryEdgesPerKTriangles"],
        "nonManifold": checkpoint["nonManifoldEdges"],
        "rawSupplyMissing": losses["tsdfOrCornerAvailabilityMissing"],
        "directionFilteringLoss": (
            losses["intraDirectionFilterLoss"]
            + losses["interDirectionVoteLoss"]
        ),
        "dmcExtractionLoss": losses["dmcExtractionLoss"],
        "depthNormalMissing": failures[
            "depthPixelOrUsableNormalSampleMissing"
        ]["ratioOfVisibleTruth"],
        "untouchedLoss": failures[
            "projectiveVoxelOrCornerUntouched"
        ]["ratioOfVisibleTruth"],
        "incompleteCornerLoss": failures[
            "insufficientCompleteCornerWeight"
        ]["ratioOfVisibleTruth"],
        "supportedNoZeroLoss": failures[
            "supportedButNoUsableZeroCrossing"
        ]["ratioOfVisibleTruth"],
        "stageAccountingClosed": stage["accountingClosed"],
        "supplyAccountingClosed": supply["accountingClosed"],
        "structures": checkpoint["structureStageAttribution"],
    }


def factorial_effects(values: dict[str, float]) -> dict[str, float]:
    b0 = values["B0_quest_estimated"]
    b1 = values["B1_ideal_estimated"]
    b2 = values["B2_quest_truth"]
    b3 = values["B3_ideal_truth"]
    return {
        "normalGainAtQuestDepth": b2 - b0,
        "normalGainAtIdealDepth": b3 - b1,
        "depthGainWithEstimatedNormals": b1 - b0,
        "depthGainWithTruthNormals": b3 - b2,
        "normalShapleyGain": 0.5 * ((b2 - b0) + (b3 - b1)),
        "depthShapleyGain": 0.5 * ((b1 - b0) + (b3 - b2)),
        "interaction": b3 - b2 - b1 + b0,
        "totalInputRecoverableGap": b3 - b0,
        "idealResidualGapToFullVisible": 1.0 - b3,
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    reports = {
        room: {arm: metrics(report_for(args, room, arm)) for arm in ALL_ARMS}
        for room in ROOMS
    }
    rows = [
        {"room": room, "arm": arm, **{k: v for k, v in values.items() if k != "structures"}}
        for room, arms in reports.items()
        for arm, values in arms.items()
    ]
    room_effects = {
        room: factorial_effects(
            {arm: float(arms[arm]["visible"]) for arm in ALL_ARMS}
        )
        for room, arms in reports.items()
    }
    metric_names = (
        "visible",
        "extra",
        "p95Meters",
        "boundaryPerK",
        "rawSupplyMissing",
    )
    room_metric_effects = {
        room: {
            metric: factorial_effects(
                {arm: float(arms[arm][metric]) for arm in ALL_ARMS}
            )
            for metric in metric_names
        }
        for room, arms in reports.items()
    }
    labels = sorted(
        {
            label
            for arms in reports.values()
            for values in arms.values()
            for label in values["structures"]
        }
    )
    structure_effects: list[dict[str, Any]] = []
    for room, arms in reports.items():
        for label in labels:
            values = {
                arm: float(
                    arms[arm]["structures"][label]["stageCoverageAt0.05m"]["finalDmc"]
                )
                for arm in ALL_ARMS
            }
            structure_effects.append(
                {"room": room, "structure": label, **factorial_effects(values)}
            )

    mean_effects = {
        key: mean(values[key] for values in room_effects.values())
        for key in next(iter(room_effects.values()))
    }
    mean_metric_effects = {
        metric: {
            key: mean(
                room_metric_effects[room][metric][key] for room in ROOMS
            )
            for key in next(iter(room_metric_effects.values()))[metric]
        }
        for metric in metric_names
    }
    structure_means = {
        label: {
            key: mean(
                row[key] for row in structure_effects if row["structure"] == label
            )
            for key in (
                "normalShapleyGain",
                "depthShapleyGain",
                "interaction",
                "totalInputRecoverableGap",
                "idealResidualGapToFullVisible",
            )
        }
        for label in labels
    }
    b3_rows = [row for row in rows if row["arm"] == "B3_ideal_truth"]
    output = {
        "schema": "scancover.input_dmc_interaction_audit.v2",
        "scope": "frozen 2x2 depth/normal input factorial; Unity disabled",
        "raster": {"width": args.width, "height": args.height},
        "rows": rows,
        "roomEffects": room_effects,
        "meanEffects": mean_effects,
        "roomMetricEffects": room_metric_effects,
        "meanMetricEffects": mean_metric_effects,
        "structureEffects": structure_effects,
        "structureMeans": structure_means,
        "responsibility": {
            "captureAndNormalGeneration": {
                "metric": "B0 to B3 visible recovery, split by normal/depth Shapley gains",
                "meanRecoverablePercentagePoints": mean_effects[
                    "totalInputRecoverableGap"
                ]
                * 100.0,
            },
            "offlineFusionCode": {
                "metric": "visible gap remaining under ideal depth + truth normals",
                "meanResidualPercentagePoints": mean_effects[
                    "idealResidualGapToFullVisible"
                ]
                * 100.0,
                "primaryStage": "TSDF/corner supply before DMC extraction",
            },
        },
        "gates": {
            "allLedgersClosed": all(
                row["stageAccountingClosed"] and row["supplyAccountingClosed"]
                for row in rows
            ),
            "idealDmcExtractionLossBounded": all(
                row["dmcExtractionLoss"] <= 0.005 for row in b3_rows
            ),
            "idealResidualRequiresCodeWork": any(
                row["rawSupplyMissing"] > 0.05 for row in b3_rows
            ),
            "inputRecoveryMaterial": mean_effects["totalInputRecoverableGap"] > 0.05,
        },
    }
    (args.out / "input_dmc_interaction_audit.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Input × DMC interaction audit",
        "",
        "| Room | Normal gain (Shapley) | Depth gain (Shapley) | Interaction | Input-recoverable gap | Ideal residual |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for room in ROOMS:
        values = room_effects[room]
        lines.append(
            f"| {room} | {values['normalShapleyGain']:+.4f} | "
            f"{values['depthShapleyGain']:+.4f} | {values['interaction']:+.4f} | "
            f"{values['totalInputRecoverableGap']:+.4f} | "
            f"{values['idealResidualGapToFullVisible']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Structure mean effects",
            "",
            "| Structure | Normal gain | Depth gain | Interaction | Input-recoverable | Ideal residual |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        [
            "",
            "## Mean B3 ideal-input delta from B0 Quest-estimated input",
            "",
            "| Metric | B3-B0 | Normal Shapley | Depth Shapley | Interaction |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric in metric_names:
        values = mean_metric_effects[metric]
        lines.append(
            f"| {metric} | {values['totalInputRecoverableGap']:+.6f} | "
            f"{values['normalShapleyGain']:+.6f} | "
            f"{values['depthShapleyGain']:+.6f} | {values['interaction']:+.6f} |"
        )
    for label in labels:
        values = structure_means[label]
        lines.append(
            f"| {label} | {values['normalShapleyGain']:+.4f} | "
            f"{values['depthShapleyGain']:+.4f} | {values['interaction']:+.4f} | "
            f"{values['totalInputRecoverableGap']:+.4f} | "
            f"{values['idealResidualGapToFullVisible']:.4f} |"
        )
    lines.extend(["", "## Responsibility", ""])
    lines.append(
        f"- Capture/normal input can recover {output['responsibility']['captureAndNormalGeneration']['meanRecoverablePercentagePoints']:.2f} pp on mean visible coverage."
    )
    lines.append(
        f"- Ideal input still leaves {output['responsibility']['offlineFusionCode']['meanResidualPercentagePoints']:.2f} pp; this belongs to offline TSDF/corner supply code, not remedial capture."
    )
    lines.extend(["", "## Gates", ""])
    lines.extend(f"- {key}: {value}" for key, value in output["gates"].items())
    (args.out / "input_dmc_interaction_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return output


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [
        (room, arm, ideal, normal)
        for arm, ideal, normal in RUN_ARMS
        for room in ROOMS
    ]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_arm, args, room, arm, ideal, normal): (room, arm)
            for room, arm, ideal, normal in tasks
        }
        for index, future in enumerate(as_completed(futures), start=1):
            room, arm, status = future.result()
            print(f"[input-interaction {index}/{len(tasks)}] {room}/{arm}: {status}", flush=True)
    output = summarize(args)
    print(json.dumps({"gates": output["gates"], "responsibility": output["responsibility"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
