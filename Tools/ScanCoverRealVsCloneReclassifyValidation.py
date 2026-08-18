#!/usr/bin/env python3
"""Compare real Quest3 reclassify output with virtual-clone outputs.

This is a direction-validation tool. It answers whether virtual Quest3
observations, after the same plane-family reclassify step, resemble the real
Quest3 multi-frame observation result enough to move toward training
strengthening.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


STRICT_ARGS = [
    "--family-normal-deg",
    "10",
    "--family-distance",
    "0.08",
    "--min-inliers",
    "800",
    "--ransac-distance",
    "0.05",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-features", type=Path, required=True)
    parser.add_argument("--clone-features", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--validation-args", nargs="*", default=STRICT_ARGS)
    parser.add_argument("--pass-normal-coverage", type=float, default=0.60)
    parser.add_argument("--pass-ratio-floor", type=float, default=0.70)
    parser.add_argument("--pass-family-min", type=int, default=3)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("[real-vs-clone]", " ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def clone_display_name(features_path: Path) -> str:
    for parent in features_path.parents:
        name = parent.name
        if name.startswith("proxy_auto-scan_"):
            return name.removeprefix("proxy_auto-scan_")
    return features_path.stem


def stable_ratio(summary: dict[str, Any]) -> float:
    stable = int(summary.get("stableInputPoints", 0))
    classified = int(summary.get("stableClassifiedPoints", 0))
    return classified / stable if stable > 0 else 0.0


def family_weight(family: dict[str, Any]) -> float:
    return float(family.get("teacherPoints", 0))


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def normal_angle_deg(a: list[float], b: list[float]) -> float:
    av = normalize(np.asarray(a, dtype=np.float64))
    bv = normalize(np.asarray(b, dtype=np.float64))
    dot = abs(float(np.dot(av, bv)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def compare_families(real: dict[str, Any], clone: dict[str, Any]) -> dict[str, Any]:
    real_families = real.get("families", [])
    clone_families = clone.get("families", [])
    if not real_families or not clone_families:
        return {
            "weightedNormalCoverage15deg": 0.0,
            "weightedNormalCoverage20deg": 0.0,
            "meanBestNormalAngleDeg": 180.0,
            "weightedMeanBestNormalAngleDeg": 180.0,
            "realMatchedFamilies15deg": 0,
            "cloneFamilyCount": len(clone_families),
            "realFamilyCount": len(real_families),
        }

    total_weight = sum(family_weight(f) for f in real_families) or float(len(real_families))
    best_angles: list[float] = []
    weighted_angles: list[float] = []
    covered15 = 0.0
    covered20 = 0.0
    matched15 = 0

    for family in real_families:
        angles = [
            normal_angle_deg(family["normal"], clone_family["normal"])
            for clone_family in clone_families
        ]
        best = min(angles)
        weight = family_weight(family) or 1.0
        best_angles.append(best)
        weighted_angles.append(best * weight)
        if best <= 15.0:
            covered15 += weight
            matched15 += 1
        if best <= 20.0:
            covered20 += weight

    return {
        "weightedNormalCoverage15deg": covered15 / total_weight,
        "weightedNormalCoverage20deg": covered20 / total_weight,
        "meanBestNormalAngleDeg": sum(best_angles) / len(best_angles),
        "weightedMeanBestNormalAngleDeg": sum(weighted_angles) / total_weight,
        "realMatchedFamilies15deg": matched15,
        "cloneFamilyCount": len(clone_families),
        "realFamilyCount": len(real_families),
    }


def validation_summary(
    python: str,
    validation_script: Path,
    features: Path,
    out_dir: Path,
    validation_args: list[str],
    skip_existing: bool,
) -> dict[str, Any]:
    summary = out_dir / "observation_plane_validation_summary.json"
    if not skip_existing or not summary.exists():
        run([python, str(validation_script), str(features), "--out", str(out_dir), *validation_args])
    return load_json(summary)


def write_reports(out: Path, real_summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "real_vs_clone_summary.csv"
    json_path = out / "real_vs_clone_summary.json"
    md_path = out / "real_vs_clone_report.md"

    fieldnames = [
        "clone",
        "passed",
        "clonePlaneFamilies",
        "realPlaneFamilies",
        "cloneFamilyCount",
        "realFamilyCount",
        "familyCountDelta",
        "cloneStableRatio",
        "realStableRatio",
        "stableRatioRelative",
        "weightedNormalCoverage15deg",
        "weightedNormalCoverage20deg",
        "weightedMeanBestNormalAngleDeg",
        "meanBestNormalAngleDeg",
        "realMatchedFamilies15deg",
        "cloneStableInputPoints",
        "realStableInputPoints",
        "cloneRiskInputPoints",
        "realRiskInputPoints",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"real": real_summary, "rows": rows}, f, ensure_ascii=False, indent=2)

    real_ratio = stable_ratio(real_summary)
    lines = [
        "# Real Quest3 vs Virtual Clone Reclassify Validation",
        "",
        "## Real Quest3 Baseline",
        "",
        f"- Plane families: {int(real_summary.get('planeFamilies', 0))}",
        f"- Plane patches: {int(real_summary.get('planePatches', 0))}",
        f"- Stable classified ratio: {real_ratio:.3f}",
        f"- Stable input points: {int(real_summary.get('stableInputPoints', 0))}",
        f"- Risk input points: {int(real_summary.get('riskInputPoints', 0))}",
        "",
        "## Clone Comparisons",
        "",
        "| clone | pass | clone families | stable ratio | normal coverage 15deg | normal coverage 20deg | weighted angle | matched real families |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {clone} | {passed} | {clonePlaneFamilies} | {cloneStableRatio:.3f} | "
            "{weightedNormalCoverage15deg:.3f} | {weightedNormalCoverage20deg:.3f} | "
            "{weightedMeanBestNormalAngleDeg:.1f} | {realMatchedFamilies15deg}/{realPlaneFamilies} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `normal coverage` measures whether large real Quest3 plane families have a matching clone family normal.",
            "- This compares structure directions and stability, not exact point-to-point overlap.",
            "- If a clone has fewer families but high normal coverage, it is usable for direction validation but still too coarse for final mesh generation.",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[real-vs-clone] wrote {csv_path}")
    print(f"[real-vs-clone] wrote {json_path}")
    print(f"[real-vs-clone] wrote {md_path}")


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    validation_script = script_dir / "ScanCoverObservationFeaturePlaneValidation.py"
    out = args.out

    real_out = out / "real"
    real_summary = validation_summary(
        args.python,
        validation_script,
        args.real_features,
        real_out,
        args.validation_args,
        args.skip_existing,
    )
    real_ratio = stable_ratio(real_summary)

    rows: list[dict[str, Any]] = []
    for clone_features in args.clone_features:
        clone_features = clone_features.resolve()
        clone_name = clone_display_name(clone_features)
        clone_out = out / clone_name
        clone_summary = validation_summary(
            args.python,
            validation_script,
            clone_features,
            clone_out,
            args.validation_args,
            args.skip_existing,
        )
        family_compare = compare_families(real_summary, clone_summary)
        clone_ratio = stable_ratio(clone_summary)
        stable_relative = clone_ratio / real_ratio if real_ratio > 0 else 0.0
        passed = (
            int(clone_summary.get("planeFamilies", 0)) >= args.pass_family_min
            and stable_relative >= args.pass_ratio_floor
            and float(family_compare["weightedNormalCoverage15deg"]) >= args.pass_normal_coverage
        )
        rows.append(
            {
                "clone": clone_name,
                "passed": passed,
                "clonePlaneFamilies": int(clone_summary.get("planeFamilies", 0)),
                "realPlaneFamilies": int(real_summary.get("planeFamilies", 0)),
                "familyCountDelta": int(clone_summary.get("planeFamilies", 0))
                - int(real_summary.get("planeFamilies", 0)),
                "cloneStableRatio": clone_ratio,
                "realStableRatio": real_ratio,
                "stableRatioRelative": stable_relative,
                "cloneStableInputPoints": int(clone_summary.get("stableInputPoints", 0)),
                "realStableInputPoints": int(real_summary.get("stableInputPoints", 0)),
                "cloneRiskInputPoints": int(clone_summary.get("riskInputPoints", 0)),
                "realRiskInputPoints": int(real_summary.get("riskInputPoints", 0)),
                **family_compare,
            }
        )

    write_reports(out, real_summary, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
