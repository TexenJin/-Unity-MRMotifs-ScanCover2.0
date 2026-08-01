#!/usr/bin/env python3
"""Summarize multiple Quest3 virtual-clone experiment reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "reports",
        type=Path,
        nargs="*",
        help="virtual_clone_similarity_report.json files. If omitted, scans the default output folder.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            r"E:\PCAII\NEW-SCANCOVER\ScanCoverExports\Quest3VirtualCloneExperiments"
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def status(distance_delta: float, angle_delta: float) -> str:
    if distance_delta <= 0.05 and angle_delta <= 0.03:
        return "pass"
    if distance_delta <= 0.10 and angle_delta <= 0.04:
        return "usable-with-calibration"
    return "needs-path-calibration"


def nested_float(data: dict[str, Any], path: list[str], default: float = 0.0) -> float:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return float(value)


def read_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    args = parse_args()
    reports = args.reports
    if not reports:
        reports = sorted(args.root.glob("replica_auto-scan_*/virtual_clone_similarity_report.json"))
    if not reports:
        raise FileNotFoundError(f"No reports found under {args.root}")

    rows = []
    for path in reports:
        report = read_report(path)
        similarity = report["similarity"]
        distance_delta = float(similarity["distanceMeanAbsShareDelta"])
        angle_delta = float(similarity["angleMeanAbsShareDelta"])
        rows.append(
            {
                "experiment": path.parent.name,
                "truthMesh": report["truthMesh"],
                "poseSource": report.get("poseSource", ""),
                "frames": report["framesReplayed"],
                "hitRatio": report["hitRatio"],
                "acceptedRatio": report["acceptedRatio"],
                "riskRatio": report["riskRatio"],
                "coverageAt0.05m": nested_float(report, ["surfaceCoverage", "coverageAtMeters", "0.05"]),
                "coverageAt0.10m": nested_float(report, ["surfaceCoverage", "coverageAtMeters", "0.10"]),
                "lowerCoverageAt0.05m": nested_float(report, ["surfaceCoverage", "surfaceBands", "lower0.30m", "coverageAt0.05m"]),
                "upperCoverageAt0.05m": nested_float(report, ["surfaceCoverage", "surfaceBands", "upper0.30m", "coverageAt0.05m"]),
                "verticalCoverageAt0.05m": nested_float(report, ["surfaceCoverage", "surfaceBands", "vertical", "coverageAt0.05m"]),
                "observationErrorP95m": nested_float(report, ["observationErrorMeters", "p95"]),
                "mature3FrameRatio": nested_float(report, ["multiViewMaturity", "ratioAtLeast3Frames"]),
                "distanceMeanAbsShareDelta": distance_delta,
                "angleMeanAbsShareDelta": angle_delta,
                "status": status(distance_delta, angle_delta),
            }
        )

    out_dir = args.out or args.root / "BenchmarkSummary"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "virtual_clone_benchmark_summary.csv"
    json_path = out_dir / "virtual_clone_benchmark_summary.json"
    md_path = out_dir / "virtual_clone_benchmark_summary.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    pass_count = sum(1 for row in rows if row["status"] == "pass")
    usable_count = sum(1 for row in rows if row["status"] in ("pass", "usable-with-calibration"))
    summary = {
        "reportCount": len(rows),
        "passCount": pass_count,
        "usableWithCalibrationCount": usable_count,
        "rows": rows,
        "aggregates": {
            "meanCoverageAt0.05m": sum(row["coverageAt0.05m"] for row in rows) / len(rows),
            "minCoverageAt0.05m": min(row["coverageAt0.05m"] for row in rows),
            "meanCoverageAt0.10m": sum(row["coverageAt0.10m"] for row in rows) / len(rows),
            "minLowerCoverageAt0.05m": min(row["lowerCoverageAt0.05m"] for row in rows),
            "minVerticalCoverageAt0.05m": min(row["verticalCoverageAt0.05m"] for row in rows),
            "meanMature3FrameRatio": sum(row["mature3FrameRatio"] for row in rows) / len(rows),
            "maxObservationErrorP95m": max(row["observationErrorP95m"] for row in rows),
        },
        "verdict": {
            "pipelineReusable": usable_count == len(rows),
            "fullyCalibrated": pass_count == len(rows),
            "note": (
                "The virtual-clone route is reusable across tested rooms, but rooms marked usable-with-calibration "
                "still need distance/path calibration before being used as strong training evidence."
            ),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Quest3 Virtual Clone Benchmark Summary",
        "",
        f"- Reports: {len(rows)}",
        f"- Pass: {pass_count}",
        f"- Usable with calibration: {usable_count}",
        f"- Pipeline reusable: {summary['verdict']['pipelineReusable']}",
        f"- Fully calibrated: {summary['verdict']['fullyCalibrated']}",
        f"- Note: {summary['verdict']['note']}",
        "",
        "| Experiment | Cov 5cm | Cov 10cm | Lower 5cm | Vertical 5cm | Mature 3+ | Error p95 | Distance delta | Angle delta | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['coverageAt0.05m']:.4f} | {row['coverageAt0.10m']:.4f} | "
            f"{row['lowerCoverageAt0.05m']:.4f} | {row['verticalCoverageAt0.05m']:.4f} | "
            f"{row['mature3FrameRatio']:.4f} | {row['observationErrorP95m']:.4f} | "
            f"{row['distanceMeanAbsShareDelta']:.4f} | "
            f"{row['angleMeanAbsShareDelta']:.4f} | {row['status']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
