#!/usr/bin/env python3
"""Summarize the frozen 2x2 paper-normal input-ceiling experiment.

The experiment changes only depth and normal inputs.  Camera path, truth
samples, voxel size, integration, and paper DMC remain fixed.  It is therefore
an attribution report, not a production parameter search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOMS = ("office0", "office4", "room0")
ARMS = (
    "B0_degraded_estimated_56x42",
    "B1_ideal_estimated_56x42",
    "B2_degraded_truth_56x42",
    "B3_ideal_truth_56x42",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(report: dict[str, Any]) -> dict[str, Any]:
    row = report["checkpoints"][-1]
    supply = row["tsdfSupplyAttribution"]
    failures = supply["firstFailure"]
    normals = report["integrationAudit"]["paperNormalEstimation"]
    audit = report["strictInputAudit"]
    valid_depth = audit.get("selectedDepthValidPixels", audit["degradedDepthValidPixels"])
    paper_normal_mode = report["parameters"]["integrationMode"] == "paper-normal-raycast"
    return {
        "visible": row["visibleCoverageAt0.05m"],
        "wholeRoom": row["wholeRoomCoverageAt0.05m"],
        "extra": row["extraSurfaceRatioAt0.05m"],
        "p95Meters": row["accuracyP95m"],
        "boundaryPerK": row["boundaryEdgesPerKTriangles"],
        "nonManifold": row["nonManifoldEdges"],
        "validDepthPixels": valid_depth,
        "usableNormalPixels": normals["filteredNormalPixels"],
        "usableNormalRatio": (
            normals["filteredNormalPixels"] / valid_depth if valid_depth else 0.0
        ) if paper_normal_mode else None,
        "incompleteCornerRatio": failures["insufficientCompleteCornerWeight"][
            "ratioOfVisibleTruth"
        ],
        "supportedNoZeroRatio": failures["supportedButNoUsableZeroCrossing"][
            "ratioOfVisibleTruth"
        ],
        "supplyAccountingClosed": supply["accountingClosed"],
    }


def average(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row[key] is not None]
    return mean(values) if values else None


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, dict[str, Any]]] = {}
    rows: dict[str, list[dict[str, Any]]] = {"A_projective": []}
    rows.update({arm: [] for arm in ARMS})
    for room in ROOMS:
        room_reports = {
            "A_projective": load(
                args.a_root / room / "A_projective" / "directional_composition_report.json"
            )
        }
        room_reports.update(
            {
                arm: load(
                    args.experiment_root
                    / room
                    / arm
                    / "directional_composition_report.json"
                )
                for arm in ARMS
            }
        )
        reports[room] = room_reports
        for arm, report in room_reports.items():
            row = {"room": room, "arm": arm, **metrics(report)}
            rows[arm].append(row)

    input_audit: dict[str, Any] = {}
    for room, room_reports in reports.items():
        audits = {name: report["strictInputAudit"] for name, report in room_reports.items()}
        input_audit[room] = {
            "cameraPathEqual": len({item["cameraPathSha256"] for item in audits.values()}) == 1,
            "truthSamplesEqual": len({item["truthSamplesSha256"] for item in audits.values()}) == 1,
            "degradedDepthEqualBetweenB0B2": (
                audits[ARMS[0]]["observationDepthSequenceSha256"]
                == audits[ARMS[2]]["observationDepthSequenceSha256"]
            ),
            "idealDepthEqualBetweenB1B3": (
                audits[ARMS[1]]["observationDepthSequenceSha256"]
                == audits[ARMS[3]]["observationDepthSequenceSha256"]
            ),
        }

    means = {
        arm: {
            key: average(arm_rows, key)
            for key in (
                "visible",
                "wholeRoom",
                "extra",
                "p95Meters",
                "boundaryPerK",
                "usableNormalRatio",
                "incompleteCornerRatio",
                "supportedNoZeroRatio",
            )
        }
        for arm, arm_rows in rows.items()
    }
    a = means["A_projective"]
    b0, b1, b2, b3 = (means[arm] for arm in ARMS)
    per_room_ceiling = {
        room: {
            "visibleRatioToA": (
                metrics(reports[room][ARMS[3]])["visible"]
                / metrics(reports[room]["A_projective"])["visible"]
            ),
            "wholeRoomRatioToA": (
                metrics(reports[room][ARMS[3]])["wholeRoom"]
                / metrics(reports[room]["A_projective"])["wholeRoom"]
            ),
        }
        for room in ROOMS
    }
    gates = {
        "allInputAuditsMatch": all(all(value.values()) for value in input_audit.values()),
        "idealTruthVisibleAtLeast95PercentAEveryRoom": all(
            value["visibleRatioToA"] >= 0.95 for value in per_room_ceiling.values()
        ),
        "idealTruthWholeRoomAtLeast95PercentAEveryRoom": all(
            value["wholeRoomRatioToA"] >= 0.95 for value in per_room_ceiling.values()
        ),
        "idealTruthNonManifoldFree": all(row["nonManifold"] == 0 for row in rows[ARMS[3]]),
        "supplyAccountingClosed": all(
            row["supplyAccountingClosed"] for arm_rows in rows.values() for row in arm_rows
        ),
    }
    conclusion = {
        "primaryBranch": (
            "capture_and_normal_supply"
            if gates["idealTruthVisibleAtLeast95PercentAEveryRoom"]
            and gates["idealTruthWholeRoomAtLeast95PercentAEveryRoom"]
            else "normal_ray_to_discrete_grid_fusion"
        ),
        "normalReplacementVisibleGainPercentagePoints": (b2["visible"] - b0["visible"]) * 100.0,
        "idealDepthAdditionalVisibleGainPercentagePoints": (b3["visible"] - b2["visible"]) * 100.0,
        "idealEstimatedVisibleDeltaPercentagePoints": (b1["visible"] - b0["visible"]) * 100.0,
        "idealTruthVisibleDeltaVsAPercentagePoints": (b3["visible"] - a["visible"]) * 100.0,
        "residualFusionFinding": (
            "secondary: ideal truth reaches A coverage, but boundary density and "
            "room-specific incomplete-corner support remain above A"
        ),
    }
    output = {
        "schema": "scancover.paper_normal_input_ceiling.v1",
        "scope": "offline input attribution only; Unity and DMC unchanged",
        "rooms": list(ROOMS),
        "rows": [row for arm in ("A_projective", *ARMS) for row in rows[arm]],
        "means": means,
        "inputAudit": input_audit,
        "perRoomCeiling": per_room_ceiling,
        "gates": gates,
        "conclusion": conclusion,
    }
    (args.out / "paper_normal_input_ceiling_report.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Paper-normal input ceiling attribution",
        "",
        "- Scope: offline only; camera path, truth samples, voxel fusion, and paper DMC are frozen.",
        f"- Primary branch: `{conclusion['primaryBranch']}`",
        "",
        "| Room | Arm | Visible | Whole room | Extra | p95 m | Boundary/1k | Usable normal | Incomplete corners | Supported/no zero |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for room in ROOMS:
        for arm in ("A_projective", *ARMS):
            row = next(item for item in rows[arm] if item["room"] == room)
            normal_ratio = (
                f"{row['usableNormalRatio']:.4f}"
                if row["usableNormalRatio"] is not None
                else "n/a"
            )
            lines.append(
                f"| {room} | {arm} | {row['visible']:.4f} | {row['wholeRoom']:.4f} | "
                f"{row['extra']:.4f} | {row['p95Meters']:.4f} | {row['boundaryPerK']:.1f} | "
                f"{normal_ratio} | {row['incompleteCornerRatio']:.4f} | "
                f"{row['supportedNoZeroRatio']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Frozen verdict",
            "",
            f"- Truth-normal replacement gain on degraded depth: {conclusion['normalReplacementVisibleGainPercentagePoints']:.2f} pp.",
            f"- Ideal depth gain after truth normals: {conclusion['idealDepthAdditionalVisibleGainPercentagePoints']:.2f} pp.",
            f"- Ideal depth with the current estimator changes coverage by {conclusion['idealEstimatedVisibleDeltaPercentagePoints']:.2f} pp; the estimator rejects many otherwise valid ideal pixels.",
            f"- Ideal depth + truth normal is {conclusion['idealTruthVisibleDeltaVsAPercentagePoints']:.2f} pp versus A on mean visible coverage.",
            "- Primary action: repair capture/normal supply first. Keep the residual shared-corner/boundary issue as a separate fusion task; do not use it to justify changing DMC now.",
            "",
            "## Gates",
            "",
        ]
    )
    lines.extend(f"- {key}: {value}" for key, value in gates.items())
    (args.out / "paper_normal_input_ceiling_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gates": gates, "conclusion": conclusion}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
