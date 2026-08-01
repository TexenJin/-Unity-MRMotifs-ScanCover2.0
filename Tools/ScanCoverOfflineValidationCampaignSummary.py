#!/usr/bin/env python3
"""Aggregate the Quest Evidence, Replica, DMC and correction-replay gates.

This report deliberately keeps a failed gate visible.  It is a campaign
decision record, not a tool for turning partial smoke tests into a pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quest-report", type=Path, required=True)
    parser.add_argument("--replica-report", type=Path, required=True)
    parser.add_argument("--dmc-ideal-report", type=Path, required=True)
    parser.add_argument("--dmc-quest-report", type=Path, required=True)
    parser.add_argument("--correction-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def variant_row(report: dict[str, Any], name: str) -> dict[str, Any]:
    row = report["variants"][name]
    return {
        "coverage5cm": row["coverageAt0.05m"],
        "extra5cm": row["extraSurfaceRatioAt0.05m"],
        "accuracyP95m": row["accuracyP95m"],
        "completenessP95m": row["completenessP95m"],
        "boundaryPerKTriangles": row["boundaryEdgesPerKTriangles"],
        "nonManifoldEdges": row["nonManifoldEdges"],
        "triangles": row["triangles"],
    }


def main() -> int:
    args = parse_args()
    quest = read_json(args.quest_report)
    replica = read_json(args.replica_report)
    dmc_ideal = read_json(args.dmc_ideal_report)
    dmc_quest = read_json(args.dmc_quest_report)
    correction_session = read_json(args.correction_report)
    correction = correction_session["projectiveTsdf"]["correctionRecovery"]

    q_cross = quest["crossImplementation"]
    rep = replica["aggregates"]
    ideal_independent = variant_row(dmc_ideal, "dominant_independent")
    ideal_composed = variant_row(dmc_ideal, "dominant_composed")
    quest_independent = variant_row(dmc_quest, "dominant_independent")
    quest_composed = variant_row(dmc_quest, "dominant_composed")

    gates = {
        "questEvidenceAndInputParity": bool(quest["passed"]),
        "replicaProjectiveReferenceEnvelope": bool(
            rep["minQuestCoverage5cm"] >= 0.84
            and rep["maxExtraDelta5cm"] <= 0.02
            and rep["maxQuestAccuracyP95m"] <= 0.04
            and rep["maxCheckpointRegression"] <= 0.01
        ),
        # A production guard may trade a little coverage for fewer false
        # surfaces, but the current loss is too large to call conservative.
        "ownershipGuardReady": bool(
            rep["meanGuardExtraDelta5cm"] < 0.0
            and rep["meanGuardCoverageDelta5cm"] >= -0.01
        ),
        "directionalCompositionIdealSmoke": bool(dmc_ideal["passed"]),
        "directionalCompositionQuestSmoke": bool(dmc_quest["passed"]),
        "directionalCompositionCrossInputReady": bool(dmc_ideal["passed"] and dmc_quest["passed"]),
        "temporalCorrectionRecovery": bool(correction["passed"]),
        "nonManifoldFreeInEvaluatedOutputs": bool(
            quest["gates"]["projectiveMeshesNonManifoldFree"]
            and ideal_composed["nonManifoldEdges"] == 0
            and quest_composed["nonManifoldEdges"] == 0
            and correction["afterRecovery"]["topology"]["nonManifoldEdges"] == 0
        ),
        # Desktop timings are useful for relative profiling only.  They cannot
        # certify the Quest main-thread/GPU budget.
        "questRuntimeBudgetMeasured": False,
    }
    gates["readyForUnityProductionReplacement"] = bool(
        gates["questEvidenceAndInputParity"]
        and gates["replicaProjectiveReferenceEnvelope"]
        and gates["ownershipGuardReady"]
        and gates["directionalCompositionCrossInputReady"]
        and gates["temporalCorrectionRecovery"]
        and gates["nonManifoldFreeInEvaluatedOutputs"]
        and gates["questRuntimeBudgetMeasured"]
    )

    report = {
        "schema": "ScanCoverOfflineValidationCampaign/v1",
        "inputs": {
            "questReport": str(args.quest_report.resolve()),
            "replicaReport": str(args.replica_report.resolve()),
            "dmcIdealReport": str(args.dmc_ideal_report.resolve()),
            "dmcQuestReport": str(args.dmc_quest_report.resolve()),
            "correctionReport": str(args.correction_report.resolve()),
        },
        "questEvidence": {
            "frames": quest["selectedFrames"],
            "sensorValidRatio": quest["globalValidity"]["sensorValidRatio"],
            "workingRangeValidRatio": quest["globalValidity"]["workingRangeValidRatio"],
            "gpuReadbackLatencyMsP95": quest["gpuReadbackLatencySeconds"]["p95"] * 1000.0,
            "linearizationDeltaMetersP95": q_cross["rawDepthLinearizationDeltaMeters"]["p95"],
            "worldReprojectionDeltaMetersP95": q_cross["inverseReprojectionWorldDeltaMeters"]["p95"],
            "radialDeltaMetersP95": q_cross["reconstructedRadialDeltaMeters"]["p95"],
        },
        "replicaProjective": rep,
        "directionalComposition": {
            "scope": "one-room, eight-frame, 5 cm voxel smoke test; not a production accuracy claim",
            "ideal": {
                "passed": dmc_ideal["passed"],
                "independent": ideal_independent,
                "composed": ideal_composed,
                "hardGate": dmc_ideal["hardGate"],
            },
            "quest": {
                "passed": dmc_quest["passed"],
                "independent": quest_independent,
                "composed": quest_composed,
                "hardGate": dmc_quest["hardGate"],
            },
            "customExtensions": {
                "questFeatureQefPassed": dmc_quest["featureShadowPassed"],
                "questTsdfHermiteEvidencePassed": dmc_quest["tsdfHermiteFeature"]["passed"],
                "questTsdfHermiteDualPassed": dmc_quest["tsdfHermiteDualMesh"]["passed"],
                "questSharedEdgeLedgerDualPassed": dmc_quest["tsdfHermiteLedgerDualMesh"]["passed"],
            },
        },
        "temporalCorrection": correction,
        "gates": gates,
        "decision": {
            "status": "HOLD_OFFLINE",
            "reason": (
                "Input parity and the Projective TSDF reference pass, but the ownership guard loses too much "
                "coverage, DMC is not cross-input stable, explicit error recovery fails, and Quest runtime is unmeasured."
            ),
            "next": [
                "Add a mature contradiction/free-space clearing or local weight-decay experiment to the offline Projective TSDF replay.",
                "Keep the directional-composition paper baseline; do not promote the failed QEF/Hermite/ledger extensions.",
                "After correction passes, rerun ideal and Quest-degraded DMC on at least three Replica rooms at the same resolution.",
                "Only then port the frozen voxel ledger to Unity and measure Quest runtime and cross-implementation voxel parity.",
            ],
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "offline_campaign_summary.json"
    md_path = args.out / "offline_campaign_summary.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# ScanCover 离线验证总报告",
        "",
        f"- 状态：**{report['decision']['status']}**",
        f"- 实机统计帧：**{quest['selectedFrames']}**",
        f"- Replica 房间：**{rep['rooms']}**",
        "",
        "## 门槛",
        "",
    ]
    md.extend(f"- {name}: **{value}**" for name, value in gates.items())
    md.extend([
        "",
        "## 关键数值",
        "",
        f"- Quest 有效深度：{quest['globalValidity']['workingRangeValidRatio']:.4f}",
        f"- CPU/Quest 世界坐标反投影 p95：{q_cross['inverseReprojectionWorldDeltaMeters']['p95'] * 1e6:.3f} µm",
        f"- Replica 平均 5 cm 覆盖：{rep['meanQuestCoverage5cm']:.4f}",
        f"- 门控平均覆盖变化：{rep['meanGuardCoverageDelta5cm']:+.4f}",
        f"- 门控平均额外面变化：{rep['meanGuardExtraDelta5cm']:+.4f}",
        f"- 纠错对称 p95：{correction['symmetricP95BeforeMeters']:.4f} m → {correction['symmetricP95AfterMeters']:.4f} m",
        "",
        "## 决策",
        "",
        report["decision"]["reason"],
        "",
        "## 下一步",
        "",
    ])
    md.extend(f"{index}. {item}" for index, item in enumerate(report["decision"]["next"], start=1))
    # BOM keeps the Chinese campaign report readable in Windows PowerShell and
    # older editors without changing the JSON interchange encoding.
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8-sig")
    print(json.dumps({"status": report["decision"]["status"], "gates": gates, "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
