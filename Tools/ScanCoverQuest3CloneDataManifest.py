#!/usr/bin/env python3
"""Validate and consume the ScanCover Quest3 clone data manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path(
    r"E:\PCAII\NEW-SCANCOVER\ScanCoverExports"
    r"\Quest3CloneDataManifest\quest3_clone_data_manifest.json"
)


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def selected_paths(manifest: dict[str, Any], section: str) -> list[Path]:
    return [
        Path(item["path"])
        for item in manifest.get(section, [])
        if bool(item.get("defaultUse", False))
    ]


def observation_stats_paths(manifest: dict[str, Any]) -> list[Path]:
    return selected_paths(manifest, "observationStatsSessions")


def badness_paths(manifest: dict[str, Any]) -> list[Path]:
    return selected_paths(manifest, "badnessSessions")


def derived_output(manifest: dict[str, Any], key: str) -> Path | None:
    value = manifest.get("derivedOutputs", {}).get(key)
    return Path(value) if value else None


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add_check(section: str, item: dict[str, Any], required_children: list[str]) -> None:
        path = Path(item["path"])
        missing_children = [child for child in required_children if not (path / child).exists()]
        checks.append(
            {
                "section": section,
                "id": item.get("id", ""),
                "role": item.get("role", ""),
                "defaultUse": bool(item.get("defaultUse", False)),
                "path": str(path),
                "exists": path.exists(),
                "missingRequiredChildren": missing_children,
                "usable": path.exists() and not missing_children,
            }
        )

    for item in manifest.get("observationStatsSessions", []):
        add_check(
            "observationStatsSessions",
            item,
            [
                "observation_stats/frame_observation_stats.csv",
                "observation_stats/distance_bins.csv",
                "observation_stats/angle_bins.csv",
                "observation_stats/edge_risk_stats.csv",
            ],
        )

    for item in manifest.get("badnessSessions", []):
        add_check(
            "badnessSessions",
            item,
            [
                "quest3_observation_badness/raw_depth_badness_frames.csv",
                "quest3_observation_badness/raw_depth_badness_summary.json",
            ],
        )

    default_failures = [
        check for check in checks if check["defaultUse"] and not check["usable"]
    ]
    return {
        "schema": "ScanCoverQuest3CloneDataManifestValidation/v1",
        "manifest": str(DEFAULT_MANIFEST),
        "defaultObservationStatsSessions": len(
            [c for c in checks if c["section"] == "observationStatsSessions" and c["defaultUse"]]
        ),
        "defaultBadnessSessions": len(
            [c for c in checks if c["section"] == "badnessSessions" and c["defaultUse"]]
        ),
        "defaultFailures": default_failures,
        "valid": not default_failures,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--print-observation-stats",
        action="store_true",
        help="Print default observation-stat session paths, one per line.",
    )
    parser.add_argument(
        "--print-badness",
        action="store_true",
        help="Print default badness session paths, one per line.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if args.print_observation_stats:
        for path in observation_stats_paths(manifest):
            print(path)
    if args.print_badness:
        for path in badness_paths(manifest):
            print(path)
    if args.validate or not (args.print_observation_stats or args.print_badness):
        report = validate_manifest(manifest)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["valid"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
