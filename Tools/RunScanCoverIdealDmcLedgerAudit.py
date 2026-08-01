#!/usr/bin/env python3
"""Audit the DMC ledger under ideal depth plus immutable truth normals."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--degradation-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=112)
    parser.add_argument("--height", type=int, default=84)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_room(args: argparse.Namespace, room: str) -> tuple[str, str]:
    output = args.out / room
    report = output / "directional_composition_report.json"
    if report.exists() and not args.force:
        return room, "reused"
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
        "raycast-truth",
        "--ideal-depth",
        "--paper-growth-ledger-only",
        "--paper-growth-stage-attribution",
        "--paper-growth-upstream-attribution",
    ]
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
        raise RuntimeError(f"{room} failed; see {output / 'run.log'}")
    return room, "ran"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    room_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    for room in ROOMS:
        report = load(args.out / room / "directional_composition_report.json")
        checkpoint = report["checkpoints"][-1]
        stage = checkpoint["stageAttribution"]
        stage_loss = stage["netStageLossAt0.05m"]
        supply = checkpoint["tsdfSupplyAttribution"]
        supply_failure = supply["firstFailure"]
        row = {
            "room": room,
            "visible": checkpoint["exactVisibleCoverageAt0.05m"],
            "visibleSampledMesh": checkpoint["visibleCoverageAt0.05m"],
            "wholeRoom": checkpoint["exactWholeRoomCoverageAt0.05m"],
            "wholeRoomSampledMesh": checkpoint["wholeRoomCoverageAt0.05m"],
            "extra": checkpoint["extraSurfaceRatioAt0.05m"],
            "p95Meters": checkpoint["accuracyP95m"],
            "boundaryPerK": checkpoint["boundaryEdgesPerKTriangles"],
            "nonManifold": checkpoint["nonManifoldEdges"],
            "rawSupplyMissing": stage_loss["tsdfOrCornerAvailabilityMissing"],
            "directionFilteringLoss": (
                stage_loss["intraDirectionFilterLoss"]
                + stage_loss["interDirectionVoteLoss"]
            ),
            "dmcExtractionLoss": stage_loss["dmcExtractionLoss"],
            "untouchedLoss": supply_failure[
                "projectiveVoxelOrCornerUntouched"
            ]["ratioOfVisibleTruth"],
            "incompleteCornerLoss": supply_failure[
                "insufficientCompleteCornerWeight"
            ]["ratioOfVisibleTruth"],
            "supportedNoZeroLoss": supply_failure[
                "supportedButNoUsableZeroCrossing"
            ]["ratioOfVisibleTruth"],
            "stageAccountingClosed": stage["accountingClosed"],
            "supplyAccountingClosed": supply["accountingClosed"],
            "diagnosticPassed": report["passed"],
        }
        row["extractionCoreGate"] = (
            row["directionFilteringLoss"] <= 0.02
            and row["dmcExtractionLoss"] <= 0.005
            and row["nonManifold"] == 0
            and row["stageAccountingClosed"]
            and row["supplyAccountingClosed"]
        )
        row["idealSupplyGate"] = row["rawSupplyMissing"] <= 0.05
        row["idealVisibleGate"] = row["visible"] >= 0.95
        room_rows.append(row)

        structure_stage = checkpoint["structureStageAttribution"]
        structure_supply = checkpoint["structureTsdfSupplyAttribution"]
        visible_bands = checkpoint.get("visibleStructureBands", {})
        for label, values in structure_stage.items():
            losses = values["netStageLossAt0.05m"]
            failures = structure_supply[label]["firstFailure"]
            structure_rows.append(
                {
                    "room": room,
                    "structure": label,
                    "visibleTruthSamples": values["visibleTruthSamples"],
                    "visibleCoverage": values["stageCoverageAt0.05m"]["finalDmc"],
                    "visibleCoverageSampledMesh": visible_bands.get(label, {}).get(
                        "visibleCoverageAt0.05m"
                    ),
                    "rawSupplyMissing": losses["tsdfOrCornerAvailabilityMissing"],
                    "directionFilteringLoss": (
                        losses["intraDirectionFilterLoss"]
                        + losses["interDirectionVoteLoss"]
                    ),
                    "dmcExtractionLoss": losses["dmcExtractionLoss"],
                    "untouchedLoss": failures[
                        "projectiveVoxelOrCornerUntouched"
                    ]["ratioOfVisibleTruth"],
                    "incompleteCornerLoss": failures[
                        "insufficientCompleteCornerWeight"
                    ]["ratioOfVisibleTruth"],
                    "supportedNoZeroLoss": failures[
                        "supportedButNoUsableZeroCrossing"
                    ]["ratioOfVisibleTruth"],
                    "stageAccountingClosed": values["accountingClosed"],
                    "supplyAccountingClosed": structure_supply[label][
                        "accountingClosed"
                    ],
                }
            )

    labels = sorted({row["structure"] for row in structure_rows})
    structure_means = {
        label: {
            key: mean(
                float(row[key])
                for row in structure_rows
                if row["structure"] == label and row[key] is not None
            )
            for key in (
                "visibleCoverage",
                "rawSupplyMissing",
                "directionFilteringLoss",
                "dmcExtractionLoss",
                "untouchedLoss",
                "incompleteCornerLoss",
                "supportedNoZeroLoss",
            )
        }
        for label in labels
    }
    output = {
        "schema": "scancover.ideal_dmc_ledger_audit.v2",
        "scope": "ideal depth + immutable truth normals; Unity disabled",
        "raster": {"width": args.width, "height": args.height},
        "roomRows": room_rows,
        "structureRows": structure_rows,
        "structureMeans": structure_means,
        "gates": {
            "allExtractionCoreGatesPass": all(
                row["extractionCoreGate"] for row in room_rows
            ),
            "allIdealSupplyGatesPass": all(row["idealSupplyGate"] for row in room_rows),
            "allIdealVisibleGatesPass": all(row["idealVisibleGate"] for row in room_rows),
            "allStructureLedgersClosed": all(
                row["stageAccountingClosed"] and row["supplyAccountingClosed"]
                for row in structure_rows
            ),
        },
        "verdict": {
            "dmcExtractionIsPrimaryLoss": any(
                row["dmcExtractionLoss"] > row["rawSupplyMissing"]
                for row in room_rows
            ),
            "idealTsdfSupplyRemainsBlocking": any(
                not row["idealSupplyGate"] for row in room_rows
            ),
        },
    }
    (args.out / "ideal_dmc_ledger_audit.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Ideal-input DMC ledger audit",
        "",
        f"- Input: ideal Replica depth + immutable truth normals at {args.width}x{args.height}.",
        "- Unity: disabled.",
        "",
        "| Room | Exact visible | Sampled-mesh visible | Raw supply missing | Direction filter loss | DMC extraction loss | Untouched | Incomplete corner | Supported/no zero | Boundary/1k | Non-manifold | Extract core | Ideal supply | Ideal visible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in room_rows:
        lines.append(
            f"| {row['room']} | {row['visible']:.4f} | {row['visibleSampledMesh']:.4f} | {row['rawSupplyMissing']:.4f} | "
            f"{row['directionFilteringLoss']:.4f} | {row['dmcExtractionLoss']:.4f} | "
            f"{row['untouchedLoss']:.4f} | {row['incompleteCornerLoss']:.4f} | "
            f"{row['supportedNoZeroLoss']:.4f} | {row['boundaryPerK']:.1f} | "
            f"{row['nonManifold']} | {row['extractionCoreGate']} | {row['idealSupplyGate']} | {row['idealVisibleGate']} |"
        )
    lines.extend(
        [
            "",
            "## Structure means",
            "",
            "| Structure | Visible | Raw supply missing | Direction filter | DMC extraction | Untouched | Incomplete corner | Supported/no zero |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label in labels:
        row = structure_means[label]
        lines.append(
            f"| {label} | {row['visibleCoverage']:.4f} | {row['rawSupplyMissing']:.4f} | "
            f"{row['directionFilteringLoss']:.4f} | {row['dmcExtractionLoss']:.4f} | "
            f"{row['untouchedLoss']:.4f} | {row['incompleteCornerLoss']:.4f} | "
            f"{row['supportedNoZeroLoss']:.4f} |"
        )
    lines.extend(["", "## Gates", ""])
    lines.extend(f"- {key}: {value}" for key, value in output["gates"].items())
    lines.extend(["", "## Verdict", ""])
    lines.extend(f"- {key}: {value}" for key, value in output["verdict"].items())
    (args.out / "ideal_dmc_ledger_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return output


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_room, args, room): room for room in ROOMS}
        for index, future in enumerate(as_completed(futures), start=1):
            room, status = future.result()
            print(f"[ideal-dmc {index}/{len(ROOMS)}] {room}: {status}", flush=True)
    output = summarize(args)
    print(json.dumps({"gates": output["gates"], "verdict": output["verdict"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
