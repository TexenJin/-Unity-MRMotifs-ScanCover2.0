#!/usr/bin/env python3
"""Summarize Quest3 raw-depth observation badness exports.

Input is either a session directory containing:
  quest3_observation_badness/raw_depth_badness_frames.csv
or the quest3_observation_badness directory itself.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def resolve_badness_dir(path: Path) -> Path:
    if (path / "raw_depth_badness_frames.csv").exists():
        return path
    candidate = path / "quest3_observation_badness"
    if (candidate / "raw_depth_badness_frames.csv").exists():
        return candidate
    raise FileNotFoundError(f"raw_depth_badness_frames.csv not found under {path}")


def read_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        filtered = (line for line in handle if line.strip() and not line.startswith("#"))
        reader = csv.DictReader(filtered)
        for row in reader:
            rows.append(row)
    return rows


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def as_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    return int(float(value)) if value else 0


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    if not rows:
        return {"status": "empty"}

    eye_samples = len(rows)
    frames = sorted({row["frame"] for row in rows})
    total_pixels = sum(as_int(row, "totalPixels") for row in rows)
    valid_pixels = sum(as_int(row, "validPixels") for row in rows)
    invalid_pixels = sum(as_int(row, "invalidPixels") for row in rows)
    large_holes = sum(as_int(row, "largeHoleComponentCount") for row in rows)
    persistent_invalid = sum(as_int(row, "persistentInvalidPixels") for row in rows)
    newly_invalid = sum(as_int(row, "newlyInvalidPixels") for row in rows)
    recovered = sum(as_int(row, "recoveredPixels") for row in rows)
    edge_risk = sum(as_int(row, "edgeRiskValidPixels") for row in rows)
    largest_hole = max(as_int(row, "largestHolePixels") for row in rows)
    invalid_ratios = [as_float(row, "invalidRatio") for row in rows]

    total_pixels = max(1, total_pixels)
    valid_pixels_safe = max(1, valid_pixels)
    return {
        "status": "ok",
        "frames": len(frames),
        "eyeSamples": eye_samples,
        "validPixels": valid_pixels,
        "invalidPixels": invalid_pixels,
        "invalidRatio": invalid_pixels / total_pixels,
        "meanEyeInvalidRatio": sum(invalid_ratios) / max(1, len(invalid_ratios)),
        "maxEyeInvalidRatio": max(invalid_ratios),
        "largeHoleComponents": large_holes,
        "largestHolePixels": largest_hole,
        "persistentInvalidPixels": persistent_invalid,
        "persistentInvalidPerEyeSample": persistent_invalid / max(1, eye_samples),
        "newlyInvalidPixels": newly_invalid,
        "recoveredPixels": recovered,
        "temporalFlipPixels": newly_invalid + recovered,
        "edgeRiskValidPixels": edge_risk,
        "edgeRiskRatioOfValid": edge_risk / valid_pixels_safe,
        "notes": [
            "high invalidRatio/largeHoleComponents indicates regional depth holes",
            "high persistentInvalidPixels indicates holes lasting across frames",
            "high temporalFlipPixels indicates unstable hit/loss behavior",
            "high edgeRiskRatioOfValid indicates depth discontinuity or valid-mask edges",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_or_badness_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    badness_dir = resolve_badness_dir(args.session_or_badness_dir)
    rows = read_rows(badness_dir / "raw_depth_badness_frames.csv")
    report = summarize(rows)
    report["source"] = str(badness_dir)

    output = args.output or (badness_dir / "quest3_badness_report.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
