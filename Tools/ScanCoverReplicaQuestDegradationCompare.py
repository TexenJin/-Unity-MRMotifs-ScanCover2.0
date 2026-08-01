#!/usr/bin/env python3
"""Compare fixed-path ideal and Quest-degraded Replica batch summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quest", type=Path, required=True)
    parser.add_argument("--ideal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    args = parse_args()
    quest = load(args.quest)
    ideal = load(args.ideal)
    ideal_by_room = {Path(row["truthMesh"]).stem: row for row in ideal["rows"]}
    rows = []
    for q in quest["rows"]:
        room = Path(q["truthMesh"]).stem
        i = ideal_by_room[room]
        row = {
            "room": room,
            "idealCoverageAt0.05m": i["coverageAt0.05m"],
            "questCoverageAt0.05m": q["coverageAt0.05m"],
            "coverageLossAt0.05m": i["coverageAt0.05m"] - q["coverageAt0.05m"],
            "idealLowerCoverageAt0.05m": i["lowerCoverageAt0.05m"],
            "questLowerCoverageAt0.05m": q["lowerCoverageAt0.05m"],
            "lowerCoverageLossAt0.05m": i["lowerCoverageAt0.05m"] - q["lowerCoverageAt0.05m"],
            "idealUpperCoverageAt0.05m": i["upperCoverageAt0.05m"],
            "questUpperCoverageAt0.05m": q["upperCoverageAt0.05m"],
            "upperCoverageLossAt0.05m": i["upperCoverageAt0.05m"] - q["upperCoverageAt0.05m"],
            "idealVerticalCoverageAt0.05m": i["verticalCoverageAt0.05m"],
            "questVerticalCoverageAt0.05m": q["verticalCoverageAt0.05m"],
            "verticalCoverageLossAt0.05m": i["verticalCoverageAt0.05m"] - q["verticalCoverageAt0.05m"],
            "idealMature3FrameRatio": i["mature3FrameRatio"],
            "questMature3FrameRatio": q["mature3FrameRatio"],
            "mature3FrameLoss": i["mature3FrameRatio"] - q["mature3FrameRatio"],
        }
        row["pathLimited"] = row["idealCoverageAt0.05m"] < 0.95
        row["degradationLimited"] = row["coverageLossAt0.05m"] > 0.03
        rows.append(row)

    aggregate = {
        "meanCoverageLossAt0.05m": sum(row["coverageLossAt0.05m"] for row in rows) / len(rows),
        "maxCoverageLossAt0.05m": max(row["coverageLossAt0.05m"] for row in rows),
        "pathLimitedRooms": [row["room"] for row in rows if row["pathLimited"]],
        "degradationLimitedRooms": [row["room"] for row in rows if row["degradationLimited"]],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "ideal_vs_quest_degradation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "ideal_vs_quest_degradation.json").write_text(
        json.dumps({"rows": rows, "aggregates": aggregate}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Replica Ideal vs Quest Degradation",
        "",
        f"- Mean 5 cm coverage loss: {aggregate['meanCoverageLossAt0.05m']:.4f}",
        f"- Max 5 cm coverage loss: {aggregate['maxCoverageLossAt0.05m']:.4f}",
        f"- Path-limited rooms: {', '.join(aggregate['pathLimitedRooms']) or 'none'}",
        f"- Degradation-limited rooms: {', '.join(aggregate['degradationLimitedRooms']) or 'none'}",
        "",
        "| Room | Ideal 5cm | Quest 5cm | Loss | Ideal mature 3+ | Quest mature 3+ | Path limited | Degradation limited |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['room']} | {row['idealCoverageAt0.05m']:.4f} | {row['questCoverageAt0.05m']:.4f} | "
            f"{row['coverageLossAt0.05m']:.4f} | {row['idealMature3FrameRatio']:.4f} | "
            f"{row['questMature3FrameRatio']:.4f} | {row['pathLimited']} | {row['degradationLimited']} |"
        )
    (args.out / "ideal_vs_quest_degradation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows, "aggregates": aggregate}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
