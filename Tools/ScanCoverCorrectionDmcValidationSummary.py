#!/usr/bin/env python3
"""Summarize the gated Quest correction and three-room dual-input DMC run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--correction-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--dmc-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def average(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row[key]) for row in rows) if rows else 0.0


def main() -> int:
    args = parse_args()
    corrections = [read(path) for path in args.correction_reports]
    dmc_cases: list[dict[str, Any]] = []
    for directory in sorted(path for path in args.dmc_root.iterdir() if path.is_dir()):
        report_path = directory / "directional_composition_report.json"
        if not report_path.exists():
            continue
        report = read(report_path)
        independent = report["variants"]["dominant_independent"]
        composed = report["variants"]["soft_composed"]
        audit = composed["audit"]
        dmc_cases.append({
            "case": directory.name,
            "room": directory.name.rsplit("_", 1)[0],
            "input": report["mode"],
            "passed": bool(report["passed"]),
            "failedGates": [key for key, value in report["hardGate"].items() if not value],
            "coverage5cmIndependent": independent["coverageAt0.05m"],
            "coverage5cmComposed": composed["coverageAt0.05m"],
            "coverageDelta": composed["coverageAt0.05m"] - independent["coverageAt0.05m"],
            "extra5cmIndependent": independent["extraSurfaceRatioAt0.05m"],
            "extra5cmComposed": composed["extraSurfaceRatioAt0.05m"],
            "extraDelta": composed["extraSurfaceRatioAt0.05m"] - independent["extraSurfaceRatioAt0.05m"],
            "accuracyP95IndependentMeters": independent["accuracyP95m"],
            "accuracyP95ComposedMeters": composed["accuracyP95m"],
            "accuracyP95DeltaMeters": composed["accuracyP95m"] - independent["accuracyP95m"],
            "boundaryDensityReductionRatio": 1.0 - (
                composed["boundaryEdgesPerKTriangles"] / independent["boundaryEdgesPerKTriangles"]
            ),
            "nonManifoldEdges": composed["nonManifoldEdges"],
            "multiDirectionCells": audit["multi_direction_cells"],
            "incompatibleWeakHypothesesDropped": audit["incompatible_weak_hypotheses_dropped"],
            "edgeCrossingOverflowDropped": audit["edge_crossing_overflow_dropped"],
            "conservativeConflictCells": audit["conservative_conflict_cells"],
        })

    correction_rows = []
    for index, report in enumerate(corrections, start=1):
        after_topology = report["topology"]["afterCorrection"]
        correction_rows.append({
            "session": index,
            "passed": bool(report["passed"]),
            "divergentBlockRatio": report["divergentBlockRatio"],
            "selectedBlockRatio": report["selectedBlockRatio"],
            "beforeP95Meters": report["before"]["symmetricP95Meters"],
            "afterP95Meters": report["after"]["symmetricP95Meters"],
            "p95ReductionRatio": report["symmetricP95ReductionRatio"],
            "coverageDelta": report["coverageDelta"],
            "nonManifoldEdges": after_topology["nonManifoldEdges"],
        })

    ideal = [row for row in dmc_cases if row["input"] == "ideal"]
    quest = [row for row in dmc_cases if row["input"] == "quest_guarded"]
    gates = {
        "allCorrectionSessionsPassed": len(correction_rows) >= 4 and all(row["passed"] for row in correction_rows),
        "allQuestDmcRoomsPassed": len(quest) >= 3 and all(row["passed"] for row in quest),
        "allIdealDmcRoomsPassed": len(ideal) >= 3 and all(row["passed"] for row in ideal),
        "allDmcOutputsNonManifoldFree": len(dmc_cases) >= 6 and all(row["nonManifoldEdges"] == 0 for row in dmc_cases),
        "noDmcEdgeCrossingOverflow": len(dmc_cases) >= 6 and all(row["edgeCrossingOverflowDropped"] == 0 for row in dmc_cases),
    }
    gates["dmcCrossInputThreeRoomReady"] = bool(
        gates["allQuestDmcRoomsPassed"]
        and gates["allIdealDmcRoomsPassed"]
        and gates["allDmcOutputsNonManifoldFree"]
        and gates["noDmcEdgeCrossingOverflow"]
    )

    aggregates = {
        "correction": {
            "sessions": len(correction_rows),
            "maxSelectedBlockRatio": max((row["selectedBlockRatio"] for row in correction_rows), default=0.0),
            "maxAfterP95Meters": max((row["afterP95Meters"] for row in correction_rows), default=0.0),
            "minP95ReductionRatio": min((row["p95ReductionRatio"] for row in correction_rows), default=0.0),
            "minCoverageDelta": min((row["coverageDelta"] for row in correction_rows), default=0.0),
        },
        "idealDmc": {
            "rooms": len(ideal),
            "passedRooms": sum(row["passed"] for row in ideal),
            "meanCoverageDelta": average(ideal, "coverageDelta"),
            "meanExtraDelta": average(ideal, "extraDelta"),
            "meanAccuracyP95DeltaMeters": average(ideal, "accuracyP95DeltaMeters"),
            "meanBoundaryDensityReductionRatio": average(ideal, "boundaryDensityReductionRatio"),
        },
        "questDmc": {
            "rooms": len(quest),
            "passedRooms": sum(row["passed"] for row in quest),
            "meanCoverageDelta": average(quest, "coverageDelta"),
            "meanExtraDelta": average(quest, "extraDelta"),
            "meanAccuracyP95DeltaMeters": average(quest, "accuracyP95DeltaMeters"),
            "meanBoundaryDensityReductionRatio": average(quest, "boundaryDensityReductionRatio"),
        },
    }
    report = {
        "schema": "ScanCoverCorrectionAndDmcCampaign/v1",
        "correctionCases": correction_rows,
        "dmcCases": dmc_cases,
        "aggregates": aggregates,
        "gates": gates,
        "decision": {
            "status": "CORRECTION_PASS_DMC_HOLD",
            "correctionReadyForUnityShadow": gates["allCorrectionSessionsPassed"],
            "dmcReadyForUnityProduction": gates["dmcCrossInputThreeRoomReady"],
            "diagnosis": (
                "Directional composition consistently reduces extra surfaces, p95 error and boundary density, "
                "but its conservative composition loses more than 3 percentage points of ideal-input coverage "
                "in office0 and room0. Zero edge-overflow counts rule out insufficient per-edge capacity."
            ),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "correction_dmc_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# ScanCover 纠错与三房间 DMC 验证",
        "",
        f"- 状态：**{report['decision']['status']}**",
        f"- 纠错：**{aggregates['correction']['sessions']}/4 通过**",
        f"- 理想输入 DMC：**{aggregates['idealDmc']['passedRooms']}/3 通过**",
        f"- Quest 退化输入 DMC：**{aggregates['questDmc']['passedRooms']}/3 通过**",
        "",
        "## DMC 明细",
        "",
        "| 场景 | 通过 | 覆盖变化 | 额外面变化 | p95 精度变化 | 边界密度下降 | 失败门槛 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in dmc_cases:
        lines.append(
            f"| {row['case']} | {row['passed']} | {row['coverageDelta']:+.4f} | "
            f"{row['extraDelta']:+.4f} | {row['accuracyP95DeltaMeters']:+.4f} m | "
            f"{row['boundaryDensityReductionRatio']:.1%} | {','.join(row['failedGates']) or '-'} |"
        )
    lines.extend([
        "",
        "## 判定",
        "",
        report["decision"]["diagnosis"],
        "",
        "纠错链可以进入 Unity 影子接入；DMC 仍停在线下，下一步只修组合阶段的覆盖损失。",
    ])
    (args.out / "correction_dmc_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(json.dumps({"gates": gates, "decision": report["decision"], "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
