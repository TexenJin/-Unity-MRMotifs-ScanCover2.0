#!/usr/bin/env python3
"""Consolidate ScanCover virtual-room validation into a rule-hardening baseline.

This script does not train a neural model. It freezes the currently validated
virtual Quest3 scanning and plane-family rules into a reproducible baseline:
- one preferred result set per Replica scene;
- coverage and plane-family metrics;
- pass/fail gates for moving rules back into Unity;
- a compact JSON rule profile that Unity-side work can consume later.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path(r"C:\Users\15319\Desktop\Destill test\outputs")

DEFAULT_SCENE_SOURCES = {
    "room0": DEFAULT_OUTPUT_ROOT / "replica_stratified_batch_v02",
    "room1": DEFAULT_OUTPUT_ROOT / "replica_stratified_room1_v01",
    "room2": DEFAULT_OUTPUT_ROOT / "replica_stratified_expansion_v01",
    "office0": DEFAULT_OUTPUT_ROOT / "replica_stratified_batch_v02",
    "office1": DEFAULT_OUTPUT_ROOT / "replica_stratified_batch_v02",
    "office2": DEFAULT_OUTPUT_ROOT / "replica_stratified_expansion_v01",
    "office3": DEFAULT_OUTPUT_ROOT / "replica_stratified_top_pass_v01",
    "office4": DEFAULT_OUTPUT_ROOT / "replica_stratified_expansion_v01",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT / "scan_cover_rule_hardening_v01")
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--min-top-coverage", type=float, default=0.95)
    parser.add_argument("--min-middle-coverage", type=float, default=0.93)
    parser.add_argument("--min-bottom-coverage", type=float, default=0.93)
    parser.add_argument("--min-stable-classified-ratio", type=float, default=0.80)
    parser.add_argument("--min-plane-families", type=int, default=6)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def band_coverage(coverage: dict[str, Any], name: str) -> float:
    for band in coverage.get("bands", []):
        if str(band.get("name", "")).lower() == name:
            return float(band.get("coverageRatio", 0.0))
    return 0.0


def scene_paths(scene: str, source_root: Path) -> tuple[Path, Path]:
    result_dir = source_root / f"{scene}_600"
    coverage_dir = source_root / f"{scene}_600_coverage"
    return (
        result_dir / "observation_plane_validation_summary.json",
        coverage_dir / "coverage_summary.json",
    )


def collect_scene(scene: str, source_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    validation_path, coverage_path = scene_paths(scene, source_root)
    validation = load_json(validation_path)
    coverage = load_json(coverage_path)

    stable = int(validation.get("stableInputPoints", 0))
    stable_classified = int(validation.get("stableClassifiedPoints", 0))
    stable_ratio = stable_classified / stable if stable > 0 else 0.0

    total_coverage = float(coverage.get("coverageRatio", 0.0))
    top = band_coverage(coverage, "top")
    middle = band_coverage(coverage, "middle")
    bottom = band_coverage(coverage, "bottom")
    families = int(validation.get("planeFamilies", 0))

    failures: list[str] = []
    if total_coverage < args.min_coverage:
        failures.append("coverage")
    if top < args.min_top_coverage:
        failures.append("top")
    if middle < args.min_middle_coverage:
        failures.append("middle")
    if bottom < args.min_bottom_coverage:
        failures.append("bottom")
    if stable_ratio < args.min_stable_classified_ratio:
        failures.append("stableClassifiedRatio")
    if families < args.min_plane_families:
        failures.append("planeFamilies")

    return {
        "scene": scene,
        "sourceRoot": str(source_root),
        "validationSummary": str(validation_path),
        "coverageSummary": str(coverage_path),
        "coverage": total_coverage,
        "topCoverage": top,
        "middleCoverage": middle,
        "bottomCoverage": bottom,
        "totalPoints": int(validation.get("totalPoints", 0)),
        "stableInputPoints": stable,
        "stableClassifiedPoints": stable_classified,
        "stableClassifiedRatio": stable_ratio,
        "riskInputPoints": int(validation.get("riskInputPoints", 0)),
        "planePatches": int(validation.get("planePatches", 0)),
        "planeFamilies": families,
        "planeFamilyLayers": int(validation.get("planeFamilyLayers", 0)),
        "passed": len(failures) == 0,
        "failures": ",".join(failures),
    }


def average(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scene",
        "passed",
        "failures",
        "coverage",
        "topCoverage",
        "middleCoverage",
        "bottomCoverage",
        "stableClassifiedRatio",
        "planeFamilies",
        "planePatches",
        "planeFamilyLayers",
        "totalPoints",
        "stableInputPoints",
        "stableClassifiedPoints",
        "riskInputPoints",
        "sourceRoot",
        "validationSummary",
        "coverageSummary",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# ScanCover Training Rule Hardening Baseline",
        "",
        "This is a rule-hardening baseline, not a final mesh-generation model.",
        "",
        "## Verdict",
        "",
        f"- Scenes: {summary['sceneCount']}",
        f"- Passed scenes: {summary['passedSceneCount']}/{summary['sceneCount']}",
        f"- Average coverage: {summary['averageCoverage']:.3f}",
        f"- Average top/middle/bottom: {summary['averageTopCoverage']:.3f} / {summary['averageMiddleCoverage']:.3f} / {summary['averageBottomCoverage']:.3f}",
        f"- Average stable-classified ratio: {summary['averageStableClassifiedRatio']:.3f}",
        f"- Ready for Unity rule freeze: {summary['readyForUnityRuleFreeze']}",
        "",
        "## Scene Results",
        "",
        "| scene | pass | coverage | top | middle | bottom | stable ratio | families | failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {scene} | {passed} | {coverage:.3f} | {topCoverage:.3f} | {middleCoverage:.3f} | "
            "{bottomCoverage:.3f} | {stableClassifiedRatio:.3f} | {planeFamilies} | {failures} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Frozen Rule Profile",
            "",
            "- Scan pattern: stratified room slices.",
            "- Simulation-only upper-room pass: enabled for virtual training coverage; do not copy as a real Quest3 scan rule.",
            "- Stable plane classifier: structural consensus, parallel-layer folding, risk layer excluded from primary family fitting.",
            "- Main failure signal: scene fails only when coverage or stable-family classification drops below gates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = [collect_scene(scene, source, args) for scene, source in DEFAULT_SCENE_SOURCES.items()]
    passed = [row for row in rows if row["passed"]]
    summary = {
        "schema": "ScanCoverTrainingRuleHardening/v1",
        "sceneCount": len(rows),
        "passedSceneCount": len(passed),
        "averageCoverage": average(rows, "coverage"),
        "averageTopCoverage": average(rows, "topCoverage"),
        "averageMiddleCoverage": average(rows, "middleCoverage"),
        "averageBottomCoverage": average(rows, "bottomCoverage"),
        "averageStableClassifiedRatio": average(rows, "stableClassifiedRatio"),
        "averagePlaneFamilies": average(rows, "planeFamilies"),
        "readyForUnityRuleFreeze": len(passed) == len(rows),
        "gates": {
            "minCoverage": args.min_coverage,
            "minTopCoverage": args.min_top_coverage,
            "minMiddleCoverage": args.min_middle_coverage,
            "minBottomCoverage": args.min_bottom_coverage,
            "minStableClassifiedRatio": args.min_stable_classified_ratio,
            "minPlaneFamilies": args.min_plane_families,
        },
        "rows": rows,
    }

    rule_profile = {
        "schema": "ScanCoverUnityRuleProfile/v1",
        "source": "Replica 8-scene rule-hardening baseline",
        "scan": {
            "virtualPattern": "stratified-slices",
            "sliceGrid": 3,
            "virtualFrameCount": 600,
            "simulationOnlyTopCoveragePass": True,
            "simulationOnlyTopFrameShare": 0.30,
            "note": "Top coverage pass uses truth bounds and is not a real-device scanning instruction.",
        },
        "stableSurfaceClassifier": {
            "stableMinFrames": 3,
            "maxDistanceMeters": 5.0,
            "maxViewAngleDegrees": 82.0,
            "maxRiskRatio": 0.55,
            "maxPositionVariance": 0.0064,
            "teacherMaxViewAngleDegrees": 72.0,
            "teacherMaxRiskRatio": 0.22,
            "teacherMaxPositionVariance": 0.0040,
        },
        "planeFamilyRules": {
            "useStructuralConsensusClassify": True,
            "foldParallelLayers": True,
            "familyNormalDegrees": 16.0,
            "familyDistanceMeters": 0.14,
            "classifyDistanceMeters": 0.09,
            "neighborRadiusMeters": 0.10,
            "neighborMinSame": 4,
            "neighborMinRatio": 0.45,
            "distanceStrongRatio": 0.45,
            "normalScoreWeight": 0.03,
            "riskLayerPolicy": "exclude from primary family fitting; keep as boundary/risk hints",
        },
        "validationGates": summary["gates"],
    }

    write_csv(args.out / "rule_hardening_scene_summary.csv", rows)
    (args.out / "rule_hardening_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.out / "unity_rule_profile_v01.json").write_text(
        json.dumps(rule_profile, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(args.out / "rule_hardening_report.md", rows, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote: {args.out / 'rule_hardening_scene_summary.csv'}")
    print(f"Wrote: {args.out / 'rule_hardening_summary.json'}")
    print(f"Wrote: {args.out / 'unity_rule_profile_v01.json'}")
    print(f"Wrote: {args.out / 'rule_hardening_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
