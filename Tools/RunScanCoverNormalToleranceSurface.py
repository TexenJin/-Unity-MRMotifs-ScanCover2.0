#!/usr/bin/env python3
"""Run and summarize the frozen paper-normal tolerance experiment.

Camera path, selected Quest-degraded depth, truth samples, voxel size, TSDF
integration and DMC extraction remain fixed.  Only the oracle normal field is
corrupted.  This is an input-tolerance measurement, not a parameter search and
never writes Unity assets or settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any


ROOMS = ("office0", "office4", "room0")
CONDITIONS = (
    {
        "name": "G05",
        "angular": 5.0,
        "dropout": 0.0,
        "edge_angular": 0.0,
        "edge_dropout": 0.0,
    },
    {
        "name": "G10",
        "angular": 10.0,
        "dropout": 0.0,
        "edge_angular": 0.0,
        "edge_dropout": 0.0,
    },
    {
        "name": "G07_5",
        "angular": 7.5,
        "dropout": 0.0,
        "edge_angular": 0.0,
        "edge_dropout": 0.0,
    },
    {
        "name": "G20",
        "angular": 20.0,
        "dropout": 0.0,
        "edge_angular": 0.0,
        "edge_dropout": 0.0,
    },
    {
        "name": "D05",
        "angular": 0.0,
        "dropout": 0.05,
        "edge_angular": 0.0,
        "edge_dropout": 0.0,
    },
    {
        "name": "D10",
        "angular": 0.0,
        "dropout": 0.10,
        "edge_angular": 0.0,
        "edge_dropout": 0.0,
    },
    {
        "name": "D07_5",
        "angular": 0.0,
        "dropout": 0.075,
        "edge_angular": 0.0,
        "edge_dropout": 0.0,
    },
    {
        "name": "D20",
        "angular": 0.0,
        "dropout": 0.20,
        "edge_angular": 0.0,
        "edge_dropout": 0.0,
    },
    {
        "name": "E10_ED10",
        "angular": 0.0,
        "dropout": 0.0,
        "edge_angular": 10.0,
        "edge_dropout": 0.10,
    },
    {
        "name": "E20_ED25",
        "angular": 0.0,
        "dropout": 0.0,
        "edge_angular": 20.0,
        "edge_dropout": 0.25,
    },
    {
        "name": "E15_ED17_5",
        "angular": 0.0,
        "dropout": 0.0,
        "edge_angular": 15.0,
        "edge_dropout": 0.175,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--degradation-model", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--truth-samples", type=int, default=30000)
    parser.add_argument("--width", type=int, default=112)
    parser.add_argument("--height", type=int, default=84)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def report_path(root: Path, room: str, condition: str) -> Path:
    return root / room / condition / "directional_composition_report.json"


def run_one(
    args: argparse.Namespace,
    room: str,
    condition: dict[str, Any],
) -> tuple[str, str, str]:
    output = args.out / room / condition["name"]
    report = output / "directional_composition_report.json"
    log = output / "run.log"
    if report.exists() and not args.force:
        return room, condition["name"], "reused"
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
        str(args.frames),
        "--camera-path-checkpoints",
        "12",
        "24",
        "48",
        str(args.frames),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--truth-samples",
        str(args.truth_samples),
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
        "--paper-normal-angular-noise-sigma-degrees",
        str(condition["angular"]),
        "--paper-normal-dropout-probability",
        str(condition["dropout"]),
        "--paper-normal-edge-angular-noise-sigma-degrees",
        str(condition["edge_angular"]),
        "--paper-normal-edge-dropout-probability",
        str(condition["edge_dropout"]),
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
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0 or not report.exists():
        raise RuntimeError(
            f"{room}/{condition['name']} failed with exit {completed.returncode}; "
            f"see {log}"
        )
    return room, condition["name"], "ran"


def baseline_path(root: Path, room: str, width: int, height: int) -> Path:
    candidates = (
        root / room / "B2_quest_truth" / "directional_composition_report.json",
        root / room / f"B2_degraded_truth_{width}x{height}" / "directional_composition_report.json",
        root / room / "B2_degraded_truth_56x42" / "directional_composition_report.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No degraded-depth/truth-normal baseline found for {room} under {root}"
    )


def extract_metrics(
    report: dict[str, Any], room: str, condition: str
) -> dict[str, Any]:
    checkpoint = report["checkpoints"][-1]
    perturbation = report.get("integrationAudit", {}).get(
        "paperNormalPerturbation", {}
    )
    supply = checkpoint.get("tsdfSupplyAttribution", {})
    supply_failures = supply.get("firstFailure", {})

    def failure_ratio(name: str) -> float | None:
        value = supply_failures.get(name)
        return float(value["ratioOfVisibleTruth"]) if value else None

    return {
        "room": room,
        "condition": condition,
        "visible": float(checkpoint["exactVisibleCoverageAt0.05m"]),
        "visibleSampledMesh": float(checkpoint["visibleCoverageAt0.05m"]),
        "wholeRoom": float(checkpoint["exactWholeRoomCoverageAt0.05m"]),
        "wholeRoomSampledMesh": float(checkpoint["wholeRoomCoverageAt0.05m"]),
        "extra": float(checkpoint["extraSurfaceRatioAt0.05m"]),
        "p95Meters": float(checkpoint["accuracyP95m"]),
        "boundaryPerK": float(checkpoint["boundaryEdgesPerKTriangles"]),
        "nonManifold": int(checkpoint["nonManifoldEdges"]),
        "triangles": int(checkpoint["triangles"]),
        "dropoutRatio": float(perturbation.get("dropoutRatio", 0.0)),
        "angularRmsDegrees": float(
            perturbation.get("appliedAngularErrorDegreesRms", 0.0)
        ),
        "edgeDropoutRatio": float(perturbation.get("edgeDropoutRatio", 0.0)),
        "incompleteCornerRatio": failure_ratio(
            "insufficientCompleteCornerWeight"
        ),
        "supportedNoZeroRatio": failure_ratio(
            "supportedButNoUsableZeroCrossing"
        ),
        "cameraHash": report["strictInputAudit"]["cameraPathSha256"],
        "depthHash": report["strictInputAudit"]["observationDepthSequenceSha256"],
        "truthHash": report["strictInputAudit"]["truthSamplesSha256"],
        "normalHash": report["strictInputAudit"]["normalSequenceSha256"],
        "accountingClosed": bool(supply.get("accountingClosed", False)),
    }


def average(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row[key]) for row in rows if row[key] is not None)


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    condition_parameters = {item["name"]: item for item in CONDITIONS}
    rows: list[dict[str, Any]] = []
    baseline_reports: dict[str, dict[str, Any]] = {}
    for room in ROOMS:
        baseline = load(
            baseline_path(args.baseline_root, room, args.width, args.height)
        )
        baseline_reports[room] = baseline
        rows.append(extract_metrics(baseline, room, "B2_TRUTH_BASELINE"))
        for condition in CONDITIONS:
            rows.append(
                extract_metrics(
                    load(report_path(args.out, room, condition["name"])),
                    room,
                    condition["name"],
                )
            )

    baseline_rows = [
        row for row in rows if row["condition"] == "B2_TRUTH_BASELINE"
    ]
    per_room_baseline = {row["room"]: row for row in baseline_rows}
    for row in rows:
        base = per_room_baseline[row["room"]]
        row["visibleRatioToBaseline"] = row["visible"] / base["visible"]
        row["wholeRoomRatioToBaseline"] = row["wholeRoom"] / base["wholeRoom"]
        row["extraDelta"] = row["extra"] - base["extra"]
        row["p95DeltaMeters"] = row["p95Meters"] - base["p95Meters"]
        row["boundaryRatioToBaseline"] = (
            row["boundaryPerK"] / base["boundaryPerK"]
            if base["boundaryPerK"] else 1.0
        )
        row["inputHashesFrozen"] = (
            row["cameraHash"] == base["cameraHash"]
            and row["depthHash"] == base["depthHash"]
            and row["truthHash"] == base["truthHash"]
        )
        row["normalHashChanged"] = (
            row["normalHash"] != base["normalHash"]
            if row["condition"] != "B2_TRUTH_BASELINE"
            else True
        )
        row["toleranceGate"] = (
            row["visibleRatioToBaseline"] >= 0.95
            and row["wholeRoomRatioToBaseline"] >= 0.95
            and row["extraDelta"] <= 0.002
            and row["p95DeltaMeters"] <= 0.002
            and row["boundaryRatioToBaseline"] <= 1.25
            and row["nonManifold"] == 0
            and row["inputHashesFrozen"]
            and row["normalHashChanged"]
            and row["accountingClosed"]
        )

    conditions = ("B2_TRUTH_BASELINE", *(item["name"] for item in CONDITIONS))
    means = {
        condition: {
            key: average(
                [row for row in rows if row["condition"] == condition], key
            )
            for key in (
                "visible",
                "wholeRoom",
                "extra",
                "p95Meters",
                "boundaryPerK",
                "visibleRatioToBaseline",
                "wholeRoomRatioToBaseline",
                "extraDelta",
                "p95DeltaMeters",
                "boundaryRatioToBaseline",
                "dropoutRatio",
                "angularRmsDegrees",
                "edgeDropoutRatio",
                "incompleteCornerRatio",
                "supportedNoZeroRatio",
            )
        }
        for condition in conditions
    }
    per_condition_gates = {
        condition: all(
            row["toleranceGate"]
            for row in rows
            if row["condition"] == condition
        )
        for condition in conditions
        if condition != "B2_TRUTH_BASELINE"
    }
    output = {
        "schema": "scancover.paper_normal_tolerance_surface.v2",
        "scope": (
            "offline-only controlled normal degradation on frozen Quest-degraded "
            "depth; Unity, TSDF, and DMC unchanged"
        ),
        "rooms": list(ROOMS),
        "conditions": condition_parameters,
        "gateDefinition": {
            "visibleAndWholeRoomAtLeastFractionOfTruthNormalBaseline": 0.95,
            "extraIncreaseMaximum": 0.002,
            "p95IncreaseMetersMaximum": 0.002,
            "boundaryDensityMaximumRatio": 1.25,
            "nonManifoldMaximum": 0,
        },
        "rows": rows,
        "means": means,
        "perConditionAllRoomsPassed": per_condition_gates,
        "allInputHashesFrozen": all(row["inputHashesFrozen"] for row in rows),
        "allSupplyAttributionClosed": all(row["accountingClosed"] for row in rows),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "normal_tolerance_surface_report.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    csv_fields = [
        "room",
        "condition",
        "angularRmsDegrees",
        "dropoutRatio",
        "edgeDropoutRatio",
        "visible",
        "wholeRoom",
        "visibleRatioToBaseline",
        "wholeRoomRatioToBaseline",
        "extra",
        "extraDelta",
        "p95Meters",
        "p95DeltaMeters",
        "boundaryPerK",
        "boundaryRatioToBaseline",
        "incompleteCornerRatio",
        "supportedNoZeroRatio",
        "toleranceGate",
    ]
    with (args.out / "normal_tolerance_surface_rows.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Paper-normal tolerance surface",
        "",
        "- Scope: frozen Quest-degraded depth; only oracle normals are corrupted.",
        "- Gate: every room must retain at least 95% of truth-normal baseline coverage, bound extra/p95/boundary regressions, remain non-manifold-free, and close the supply ledger.",
        "",
        "| Condition | Angular RMS | Dropout | Edge dropout | Visible/base | Whole/base | Extra delta | p95 delta m | Boundary/base | All rooms pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for condition in conditions:
        values = means[condition]
        gate = (
            "baseline"
            if condition == "B2_TRUTH_BASELINE"
            else str(per_condition_gates[condition])
        )
        lines.append(
            f"| {condition} | {values['angularRmsDegrees']:.2f} | "
            f"{values['dropoutRatio']:.4f} | {values['edgeDropoutRatio']:.4f} | "
            f"{values['visibleRatioToBaseline']:.4f} | "
            f"{values['wholeRoomRatioToBaseline']:.4f} | "
            f"{values['extraDelta']:+.4f} | {values['p95DeltaMeters']:+.4f} | "
            f"{values['boundaryRatioToBaseline']:.3f} | {gate} |"
        )
    (args.out / "normal_tolerance_surface_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return output


def main() -> int:
    args = parse_args()
    tasks = [(room, condition) for condition in CONDITIONS for room in ROOMS]
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {
            executor.submit(run_one, args, room, condition): (room, condition["name"])
            for room, condition in tasks
        }
        completed_count = 0
        for future in as_completed(futures):
            room, condition = futures[future]
            result = future.result()
            completed_count += 1
            print(
                f"[normal-tolerance {completed_count}/{len(tasks)}] "
                f"{room}/{condition}: {result[2]}",
                flush=True,
            )
    output = summarize(args)
    print(
        json.dumps(
            {
                "allInputHashesFrozen": output["allInputHashesFrozen"],
                "allSupplyAttributionClosed": output["allSupplyAttributionClosed"],
                "perConditionAllRoomsPassed": output["perConditionAllRoomsPassed"],
                "out": str(args.out),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
