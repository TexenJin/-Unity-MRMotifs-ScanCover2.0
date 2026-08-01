#!/usr/bin/env python3
"""Compare strict projective-vs-paper-normal DTSDF growth reports.

Directory layout is intentionally fixed so accidental report mixing fails fast:

  ROOT/<room>/A_projective/directional_composition_report.json
  ROOT/<room>/B_paper_normal/directional_composition_report.json

The comparison is read-only and never writes Unity assets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPORT_NAME = "directional_composition_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rooms", nargs="+", required=True)
    parser.add_argument("--b-folder", default="B_paper_normal")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def read_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not report.get("checkpoints"):
        raise ValueError(f"growth report has no checkpoints: {path}")
    return report


def metric_delta(a: dict[str, Any], b: dict[str, Any], name: str) -> float:
    return float(b[name]) - float(a[name])


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    global_gate: dict[str, bool] = {}

    frozen_parameter_names = (
        "frames",
        "cameraPathCheckpoints",
        "width",
        "height",
        "truthSamples",
        "voxelMeters",
        "truncationMeters",
        "paperMinimumWeight",
    )
    for room in args.rooms:
        a_path = args.root / room / "A_projective" / REPORT_NAME
        b_path = args.root / room / args.b_folder / REPORT_NAME
        a_report = read_report(a_path)
        b_report = read_report(b_path)
        a_final = a_report["checkpoints"][-1]
        b_final = b_report["checkpoints"][-1]

        parameter_match = all(
            a_report["parameters"].get(name) == b_report["parameters"].get(name)
            for name in frozen_parameter_names
        )
        input_a = a_report.get("strictInputAudit", {})
        input_b = b_report.get("strictInputAudit", {})
        input_match = all(
            input_a.get(name) == input_b.get(name)
            for name in (
                "cameraPathSha256",
                "observationDepthSequenceSha256",
                "truthSamplesSha256",
                "degradedDepthValidPixels",
            )
        )
        checkpoint_match = [row["frame"] for row in a_report["checkpoints"]] == [
            row["frame"] for row in b_report["checkpoints"]
        ]
        gate = {
            "frozenParametersMatch": parameter_match,
            "inputHashesMatch": input_match,
            "checkpointFramesMatch": checkpoint_match,
            "aUsesProjectiveFullDepthLookup": (
                a_report["parameters"].get("integrationMode") == "projective"
            ),
            "bUsesEveryValidDepthPixel": (
                b_report["parameters"].get("integrationMode")
                == "paper-normal-raycast"
                and int(b_report["parameters"].get("sampleStride", 0)) == 1
            ),
            "bVisibleCoverageAtLeast95PercentOfA": (
                float(b_final["visibleCoverageAt0.05m"])
                >= 0.95 * float(a_final["visibleCoverageAt0.05m"])
            ),
            "bWholeRoomCoverageAtLeast95PercentOfA": (
                float(b_final["wholeRoomCoverageAt0.05m"])
                >= 0.95 * float(a_final["wholeRoomCoverageAt0.05m"])
            ),
            "bExtraSurfaceNotWorseByMoreThan0.5pp": (
                float(b_final["extraSurfaceRatioAt0.05m"])
                <= float(a_final["extraSurfaceRatioAt0.05m"]) + 0.005
            ),
            "bAccuracyP95NotWorseByMoreThan5mm": (
                float(b_final["accuracyP95m"])
                <= float(a_final["accuracyP95m"]) + 0.005
            ),
            "bNonManifoldFree": int(b_final["nonManifoldEdges"]) == 0,
            "bBoundaryRateBounded": (
                float(b_final["boundaryEdgesPerKTriangles"])
                <= max(
                    float(a_final["boundaryEdgesPerKTriangles"]) * 1.10,
                    float(a_final["boundaryEdgesPerKTriangles"]) + 5.0,
                )
            ),
            "bGrowthDiagnosticPassed": bool(b_report.get("passed", False)),
        }
        for name, value in gate.items():
            global_gate[f"{room}.{name}"] = bool(value)

        rows.append(
            {
                "room": room,
                "reports": {"A": str(a_path), "B": str(b_path)},
                "strictInputAudit": {"A": input_a, "B": input_b},
                "final": {
                    "A": a_final,
                    "B": b_final,
                    "BminusA": {
                        name: metric_delta(a_final, b_final, name)
                        for name in (
                            "visibleCoverageAt0.05m",
                            "wholeRoomCoverageAt0.05m",
                            "extraSurfaceRatioAt0.05m",
                            "accuracyP95m",
                            "boundaryEdgesPerKTriangles",
                        )
                    },
                },
                "integrationAudit": {
                    "A": a_report.get("integrationAudit", {}),
                    "B": b_report.get("integrationAudit", {}),
                },
                "integrationSeconds": {
                    "A": float(a_report["integrationMs"]) / 1000.0,
                    "B": float(b_report["integrationMs"]) / 1000.0,
                    "AdivB": (
                        float(a_report["integrationMs"])
                        / max(1e-9, float(b_report["integrationMs"]))
                    ),
                },
                "visibleLossFirstFailureRatio": {
                    arm: {
                        name: float(values["ratioOfVisibleTruth"])
                        for name, values in report["checkpoints"][-1]
                        .get("tsdfSupplyAttribution", {})
                        .get("firstFailure", {})
                        .items()
                    }
                    for arm, report in (("A", a_report), ("B", b_report))
                },
                "gate": gate,
                "passed": all(gate.values()),
            }
        )

    means: dict[str, dict[str, float]] = {}
    for metric in (
        "visibleCoverageAt0.05m",
        "wholeRoomCoverageAt0.05m",
        "extraSurfaceRatioAt0.05m",
        "accuracyP95m",
        "boundaryEdgesPerKTriangles",
    ):
        a_values = [float(row["final"]["A"][metric]) for row in rows]
        b_values = [float(row["final"]["B"][metric]) for row in rows]
        means[metric] = {
            "A": sum(a_values) / len(a_values),
            "B": sum(b_values) / len(b_values),
            "BminusA": sum(b_values) / len(b_values) - sum(a_values) / len(a_values),
        }

    result = {
        "schema": "scancover.directional_tsdf.strict_integration_ab.v1",
        "scope": "offline only; identical observations and DMC; integration arm differs",
        "arms": {
            "A": (
                "voxel-centre projective axial-depth TSDF; allocation stride may "
                "be sparse but every allocated voxel queries the full depth image"
            ),
            "B": (
                "validity-aware bilateral normals + two-sided Amanatides-Woo traversal "
                "+ point-to-plane DTSDF + Nguyen depth/cosine angle/direction weights; "
                "sample stride fixed to one so every usable depth pixel is fused"
            ),
        },
        "rooms": rows,
        "meanFinalMetrics": means,
        "meanIntegrationSeconds": {
            "A": sum(row["integrationSeconds"]["A"] for row in rows) / len(rows),
            "B": sum(row["integrationSeconds"]["B"] for row in rows) / len(rows),
            "AdivB": sum(row["integrationSeconds"]["AdivB"] for row in rows) / len(rows),
        },
        "replacementGate": global_gate,
        "passed": all(global_gate.values()),
        "unityTouched": False,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "strict_integration_ab_report.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    lines = [
        "# ScanCover Strict Offline Integration A/B",
        "",
        f"- replacement gate passed: {result['passed']}",
        "- Unity touched: false",
        "- only the depth-to-DTSDF integration arm differs",
        "",
        "| Room | Arm | Visible 5cm | Whole room 5cm | Extra 5cm | Accuracy p95 | Boundary / 1k | Non-manifold | Integrate s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        for arm in ("A", "B"):
            values = row["final"][arm]
            lines.append(
                f"| {row['room']} | {arm} | {values['visibleCoverageAt0.05m']:.4f} | "
                f"{values['wholeRoomCoverageAt0.05m']:.4f} | "
                f"{values['extraSurfaceRatioAt0.05m']:.4f} | "
                f"{values['accuracyP95m']:.4f} | "
                f"{values['boundaryEdgesPerKTriangles']:.2f} | "
                f"{values['nonManifoldEdges']} | "
                f"{row['integrationSeconds'][arm]:.2f} |"
            )
    lines.extend(["", "## Mean", ""])
    lines.append(
        f"- integration speedup A/B: {result['meanIntegrationSeconds']['AdivB']:.2f}x"
    )
    for metric, values in means.items():
        lines.append(
            f"- {metric}: A={values['A']:.6f}; B={values['B']:.6f}; "
            f"B-A={values['BminusA']:.6f}"
        )
    lines.extend(["", "## Upstream visible-loss attribution", ""])
    for row in rows:
        lines.append(f"- {row['room']}")
        for arm in ("A", "B"):
            values = row["visibleLossFirstFailureRatio"][arm]
            lines.append(
                f"  - {arm}: "
                + ", ".join(f"{name}={value:.4f}" for name, value in values.items())
            )
    lines.extend(["", "## Pre-registered replacement gate", ""])
    lines.extend(f"- {name}: {value}" for name, value in global_gate.items())
    (args.out / "strict_integration_ab_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": result["passed"], "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
