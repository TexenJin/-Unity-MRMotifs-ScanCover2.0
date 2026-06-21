#!/usr/bin/env python3
"""Run Quest3 virtual-clone direction validation across multiple input meshes.

The goal is not to prove generalization to every room. It is to make the
current "virtual Quest3 -> observation features -> plane family reclassify"
result reproducible across several raw ScanCover BL meshes without manually
running each case.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_LEARNING_PROFILE = Path(
    r"C:\Users\15319\Desktop\Destill test\outputs"
    r"\quest3_real_20260604_222905_671_summary_v02\quest3_learning_profile_v02.json"
)
DEFAULT_REAL_SUMMARY = Path(
    r"C:\Users\15319\Desktop\Destill test\outputs"
    r"\quest3_real_20260604_222905_671_summary_v02\combined_observation_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meshes", type=Path, nargs="+", required=True)
    parser.add_argument("--learning-profile", type=Path, default=DEFAULT_LEARNING_PROFILE)
    parser.add_argument("--real-summary", type=Path, default=DEFAULT_REAL_SUMMARY)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--reuse-virtual-observations",
        action="store_true",
        help="Reuse existing virtual clone feature CSV/report and rerun only student validation.",
    )
    parser.add_argument("--pass-default-families", type=int, default=3)
    parser.add_argument("--pass-default-ratio", type=float, default=0.55)
    parser.add_argument("--pass-direction-families", type=int, default=3)
    parser.add_argument("--pass-direction-ratio", type=float, default=0.45)
    parser.add_argument("--pass-strict-families", type=int, default=3)
    parser.add_argument("--pass-strict-ratio", type=float, default=0.55)
    parser.add_argument("--pass-structural-families", type=int, default=3)
    parser.add_argument("--pass-structural-ratio", type=float, default=0.80)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("[cross-mesh]", " ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def ratio(summary: dict[str, Any]) -> float:
    stable = int(summary.get("stableInputPoints", 0))
    classified = int(summary.get("stableClassifiedPoints", 0))
    return classified / stable if stable > 0 else 0.0


def collect_row(
    mesh: Path,
    mode: str,
    validation_summary: dict[str, Any],
    clone_report: dict[str, Any],
    pass_families: int,
    pass_ratio: float,
) -> dict[str, Any]:
    classified_ratio = ratio(validation_summary)
    families = int(validation_summary.get("planeFamilies", 0))
    patches = int(validation_summary.get("planePatches", 0))
    passed = families >= pass_families and classified_ratio >= pass_ratio

    return {
        "mesh": mesh.stem,
        "validationMode": mode,
        "passed": passed,
        "passRule": f"families>={pass_families} and classifiedStableRatio>={pass_ratio:.2f}",
        "totalPoints": int(validation_summary.get("totalPoints", 0)),
        "stableInputPoints": int(validation_summary.get("stableInputPoints", 0)),
        "riskInputPoints": int(validation_summary.get("riskInputPoints", 0)),
        "planePatches": patches,
        "planeFamilies": families,
        "stableClassifiedPoints": int(validation_summary.get("stableClassifiedPoints", 0)),
        "stableUnclassifiedPoints": int(validation_summary.get("stableUnclassifiedPoints", 0)),
        "classifiedStableRatio": classified_ratio,
        "attemptedRays": int(clone_report.get("attemptedRays", 0)),
        "meshHits": int(clone_report.get("meshHits", 0)),
        "acceptedObservations": int(clone_report.get("acceptedObservations", 0)),
        "observationFeatureVoxels": int(clone_report.get("observationFeatureVoxels", 0)),
        "hitRatio": float(clone_report.get("hitRatio", 0.0)),
        "acceptedRatio": float(clone_report.get("acceptedRatio", 0.0)),
        "riskRatio": float(clone_report.get("riskRatio", 0.0)),
    }


def write_reports(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cross_mesh_repro_summary.csv"
    json_path = out_dir / "cross_mesh_repro_summary.json"
    md_path = out_dir / "cross_mesh_repro_summary.md"

    fieldnames = [
        "mesh",
        "validationMode",
        "passed",
        "passRule",
        "totalPoints",
        "stableInputPoints",
        "riskInputPoints",
        "planePatches",
        "planeFamilies",
        "stableClassifiedPoints",
        "stableUnclassifiedPoints",
        "classifiedStableRatio",
        "attemptedRays",
        "meshHits",
        "acceptedObservations",
        "observationFeatureVoxels",
        "hitRatio",
        "acceptedRatio",
        "riskRatio",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, ensure_ascii=False, indent=2)

    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(str(row["validationMode"]), []).append(row)

    lines: list[str] = []
    lines.append("# Quest3 Virtual Clone Cross-Mesh Reproducibility")
    lines.append("")
    for mode, mode_rows in by_mode.items():
        passed = sum(1 for row in mode_rows if row["passed"])
        lines.append(f"## {mode}")
        lines.append("")
        lines.append(f"- Passed: {passed}/{len(mode_rows)}")
        if mode_rows:
            avg_ratio = sum(float(row["classifiedStableRatio"]) for row in mode_rows) / len(mode_rows)
            avg_families = sum(int(row["planeFamilies"]) for row in mode_rows) / len(mode_rows)
            lines.append(f"- Average classified stable ratio: {avg_ratio:.3f}")
            lines.append(f"- Average plane families: {avg_families:.2f}")
        lines.append("")
        lines.append(
            "| mesh | pass | families | patches | stable ratio | stable pts | risk pts | hit ratio | accepted obs |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in mode_rows:
            lines.append(
                "| {mesh} | {passed} | {planeFamilies} | {planePatches} | {classifiedStableRatio:.3f} | "
                "{stableInputPoints} | {riskInputPoints} | {hitRatio:.3f} | {acceptedObservations} |".format(
                    **row
                )
            )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[cross-mesh] wrote {csv_path}")
    print(f"[cross-mesh] wrote {json_path}")
    print(f"[cross-mesh] wrote {md_path}")


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    virtual_script = script_dir / "ScanCoverQuest3VirtualCloneExperiment.py"
    validation_script = script_dir / "ScanCoverObservationFeaturePlaneValidation.py"

    rows: list[dict[str, Any]] = []
    for mesh in args.meshes:
        mesh = mesh.resolve()
        if not mesh.exists():
            raise FileNotFoundError(mesh)

        mesh_out = args.out / mesh.stem
        run_dir = mesh_out / f"proxy_auto-scan_{mesh.stem}"
        clone_report_path = run_dir / "virtual_clone_similarity_report.json"
        feature_csv = run_dir / "virtual_observation_features" / "point_observation_features.csv"

        virtual_outputs_exist = clone_report_path.exists() and feature_csv.exists()
        if (not args.reuse_virtual_observations or not virtual_outputs_exist) and (
            not args.skip_existing or not virtual_outputs_exist
        ):
            run(
                [
                    args.python,
                    str(virtual_script),
                    "--truth-mesh",
                    str(mesh),
                    "--learning-profile",
                    str(args.learning_profile),
                    "--real-summary",
                    str(args.real_summary),
                    "--pose-source",
                    "auto-scan",
                    "--coverage-pass",
                    "full",
                    "--mode",
                    "proxy",
                    "--out",
                    str(mesh_out),
                ]
            )

        clone_report = load_json(clone_report_path)

        validation_runs = [
            (
                "default",
                run_dir / "student_validation_default",
                [],
                args.pass_default_families,
                args.pass_default_ratio,
            ),
            (
                "strict_family",
                run_dir / "student_validation_strict",
                [
                    "--family-normal-deg",
                    "10",
                    "--family-distance",
                    "0.08",
                    "--min-inliers",
                    "800",
                    "--ransac-distance",
                    "0.05",
                ],
                args.pass_strict_families,
                args.pass_strict_ratio,
            ),
            (
                "direction_threshold",
                run_dir / "student_validation_direction",
                [
                    "--stable-min-frames",
                    "2",
                    "--max-risk-ratio",
                    "0.75",
                    "--max-position-variance",
                    "0.02",
                    "--min-inliers",
                    "450",
                    "--ransac-distance",
                    "0.08",
                    "--family-normal-deg",
                    "22",
                    "--family-distance",
                    "0.22",
                ],
                args.pass_direction_families,
                args.pass_direction_ratio,
            ),
            (
                "structural_consensus",
                run_dir / "student_validation_structural",
                [
                    "--family-normal-deg",
                    "10",
                    "--family-distance",
                    "0.08",
                    "--min-inliers",
                    "800",
                    "--ransac-distance",
                    "0.05",
                    "--use-structural-consensus-classify",
                ],
                args.pass_structural_families,
                args.pass_structural_ratio,
            ),
        ]

        for mode, out_dir, extra_args, pass_families, pass_ratio in validation_runs:
            summary_path = out_dir / "observation_plane_validation_summary.json"
            if not args.skip_existing or not summary_path.exists():
                run([args.python, str(validation_script), str(feature_csv), "--out", str(out_dir), *extra_args])
            validation_summary = load_json(summary_path)
            rows.append(
                collect_row(mesh, mode, validation_summary, clone_report, pass_families, pass_ratio)
            )

        write_reports(args.out, rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
