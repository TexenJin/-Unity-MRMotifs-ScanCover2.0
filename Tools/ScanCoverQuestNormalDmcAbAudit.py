#!/usr/bin/env python3
"""Compare frozen Quest raw-normal and neighbour-normal paper DMC replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


REPORT_RAW = "quest_directional_replay_report_raw.json"
REPORT_NEIGHBOUR = "quest_directional_replay_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--neighbour-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(report: dict[str, Any]) -> dict[str, Any]:
    topology = report["topology"]
    audit = topology["audit"]
    return {
        "normalSource": report["parameters"]["normalSource"],
        "acceptedNormalRays": report["inputAudit"]["acceptedNormalRays"],
        "completeCornerCellRatio": report["integrationAudit"][
            "completeCornerCellRatioOfCandidates"
        ],
        "triangles": topology["triangles"],
        "boundaryPerK": topology["boundaryEdgesPerKTriangles"],
        "nonManifold": topology["nonManifoldEdges"],
        "neighbourDisagreementsAfter": audit["paper_neighbor_disagreements_after"],
        "regularizationRevertedCells": audit["paper_regularization_reverted_cells"],
        "components": audit["paper_components"],
        "unresolvedEdgeSlots": audit["paper_unresolved_edge_slots"],
        "unmeasuredEdgeDeferredTriangles": audit[
            "paper_unmeasured_edge_deferred_triangles"
        ],
        "allGatesPass": all(report["gates"].values()),
    }


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sessions = sorted(
        path.name
        for path in args.neighbour_root.glob("Evidence_*")
        if path.is_dir() and (path / REPORT_NEIGHBOUR).exists()
    )
    if not sessions:
        raise RuntimeError("no matched neighbour replay reports")

    rows = []
    for session in sessions:
        raw_path = args.raw_root / session / REPORT_RAW
        neighbour_path = args.neighbour_root / session / REPORT_NEIGHBOUR
        if not raw_path.exists():
            raise RuntimeError(f"missing raw replay report: {raw_path}")
        raw = metrics(load(raw_path))
        neighbour = metrics(load(neighbour_path))
        rows.append(
            {
                "session": session,
                "raw": raw,
                "neighbour": neighbour,
                "neighbourMinusRaw": {
                    key: neighbour[key] - raw[key]
                    for key in (
                        "completeCornerCellRatio",
                        "triangles",
                        "boundaryPerK",
                        "nonManifold",
                        "neighbourDisagreementsAfter",
                        "regularizationRevertedCells",
                        "components",
                    )
                },
            }
        )

    delta_keys = tuple(rows[0]["neighbourMinusRaw"])
    mean_deltas = {
        key: mean(row["neighbourMinusRaw"][key] for row in rows)
        for key in delta_keys
    }
    all_sessions_boundary_improved = all(
        row["neighbourMinusRaw"]["boundaryPerK"] < 0.0 for row in rows
    )
    all_sessions_corner_supply_improved = all(
        row["neighbourMinusRaw"]["completeCornerCellRatio"] > 0.0
        for row in rows
    )
    all_topology_safe = all(
        row["raw"]["allGatesPass"] and row["neighbour"]["allGatesPass"]
        for row in rows
    )
    output = {
        "schema": "scancover.quest_evidence_v3.normal_source_dmc_ab.v1",
        "scope": "same frozen depth, frames, voxel, truncation and paper DMC; normal source is the only changed input",
        "sessions": rows,
        "meanNeighbourMinusRaw": mean_deltas,
        "gates": {
            "allTopologySafe": all_topology_safe,
            "allSessionsBoundaryImproved": all_sessions_boundary_improved,
            "allSessionsCornerSupplyImproved": all_sessions_corner_supply_improved,
            "neighbourInputContractAccepted": (
                all_topology_safe
                and all_sessions_boundary_improved
                and all_sessions_corner_supply_improved
            ),
        },
        "acceptedInputContract": {
            "pointSource": "world_position_raw_rgba32f",
            "normalSource": "world_normal_neighbour_rgba32f",
            "normalOrientation": "flip toward the observing eye before integration",
            "distanceRangeMeters": [0.35, 5.0],
            "weighting": "paper depth noise weight × view-angle weight × directional-sector weight",
            "additionalHardNearOrGrazingReject": False,
            "reason": "near/grazing buckets remain noisy, but the mature paper weighting already attenuates grazing input; no hard rejection is justified by this A/B",
        },
        "limitations": [
            "No registered real-room truth is available, so this audit does not claim absolute surface accuracy.",
            "The contract is admitted only to a Unity shadow chain, not directly to the production mesh.",
        ],
    }
    json_path = args.out / "quest_normal_source_dmc_ab.json"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Quest 法线源 DMC A/B",
        "",
        "唯一变量是法线源；深度、帧、体素、截断距离与论文 DMC 完全相同。",
        "",
        "| Session | Raw boundary/1k | Neighbour boundary/1k | Δ | Raw complete | Neighbour complete |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['session']} | {row['raw']['boundaryPerK']:.2f} | "
            f"{row['neighbour']['boundaryPerK']:.2f} | "
            f"{row['neighbourMinusRaw']['boundaryPerK']:+.2f} | "
            f"{row['raw']['completeCornerCellRatio']:.4f} | "
            f"{row['neighbour']['completeCornerCellRatio']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"平均边界变化：{mean_deltas['boundaryPerK']:+.2f}/千三角。",
            f"平均完整角点单元变化：{mean_deltas['completeCornerCellRatio']:+.6f}。",
            "",
            f"Neighbour 输入契约准入：{output['gates']['neighbourInputContractAccepted']}。",
        ]
    )
    (args.out / "quest_normal_source_dmc_ab.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["gates"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
