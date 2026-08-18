#!/usr/bin/env python3
"""Summarize ScanCover Quest3 observation-stat capture sessions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header_index = next((i for i, line in enumerate(lines) if line and not line.startswith("#") and "," in line), None)
    if header_index is None:
        return []
    return list(csv.DictReader(lines[header_index:]))


def weighted(rows: list[dict[str, str]], key: str) -> float:
    total = 0
    value_sum = 0.0
    for row in rows:
        count = int(float(row.get("count", "0") or 0))
        total += count
        value_sum += float(row.get(key, "0") or 0) * count
    return value_sum / total if total else 0.0


def aggregate_bins(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    bins: dict[str, dict[str, float]] = {}
    for row in rows:
        label = row.get("bin", "")
        data = bins.setdefault(label, {
            "count": 0,
            "viewDepth": 0.0,
            "distance": 0.0,
            "angle": 0.0,
            "boundary": 0,
            "crease": 0,
            "risk": 0,
        })
        count = int(float(row.get("count", "0") or 0))
        data["count"] += count
        data["viewDepth"] += float(row.get("avgViewDepth", "0") or 0) * count
        data["distance"] += float(row.get("avgEuclideanDistance", "0") or 0) * count
        data["angle"] += float(row.get("avgViewAngleDeg", "0") or 0) * count
        data["boundary"] += int(float(row.get("boundaryRiskCount", "0") or 0))
        data["crease"] += int(float(row.get("creaseRiskCount", "0") or 0))
        data["risk"] += int(float(row.get("anyRiskCount", "0") or 0))

    result = []
    for label, data in bins.items():
        count = int(data["count"])
        result.append({
            "bin": label,
            "count": count,
            "avgViewDepth": data["viewDepth"] / count if count else 0.0,
            "avgEuclideanDistance": data["distance"] / count if count else 0.0,
            "avgViewAngleDeg": data["angle"] / count if count else 0.0,
            "riskRatio": data["risk"] / count if count else 0.0,
            "boundaryRiskRatio": data["boundary"] / count if count else 0.0,
            "creaseRiskRatio": data["crease"] / count if count else 0.0,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    args = parser.parse_args()

    stats_dir = args.session / "observation_stats"
    frame_rows = read_csv(stats_dir / "frame_observation_stats.csv")
    distance_rows = read_csv(stats_dir / "distance_bins.csv")
    angle_rows = read_csv(stats_dir / "angle_bins.csv")
    edge_rows = read_csv(stats_dir / "edge_risk_stats.csv")

    total_points = sum(int(float(row.get("pointCount", "0") or 0)) for row in frame_rows)
    total_boundary = sum(int(float(row.get("boundaryRiskCount", "0") or 0)) for row in edge_rows)
    total_crease = sum(int(float(row.get("creaseRiskCount", "0") or 0)) for row in edge_rows)
    total_risk = sum(int(float(row.get("anyRiskCount", "0") or 0)) for row in edge_rows)

    summary = {
        "session": str(args.session),
        "frames": len(frame_rows),
        "points": total_points,
        "avgViewDepth": weighted(distance_rows, "avgViewDepth"),
        "avgEuclideanDistance": weighted(distance_rows, "avgEuclideanDistance"),
        "avgViewAngleDeg": weighted(angle_rows, "avgViewAngleDeg"),
        "boundaryRiskRatio": total_boundary / total_points if total_points else 0.0,
        "creaseRiskRatio": total_crease / total_points if total_points else 0.0,
        "anyRiskRatio": total_risk / total_points if total_points else 0.0,
        "distanceBins": aggregate_bins(distance_rows),
        "angleBins": aggregate_bins(angle_rows),
    }

    out_path = stats_dir / "observation_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
