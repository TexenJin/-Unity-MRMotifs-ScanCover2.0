#!/usr/bin/env python3
"""Run the older ScanCover room-coverage reference pipeline.

This wrapper intentionally restores the pre-dense-Raw-Depth workflow:

1. Convert room_raw_coverage_voxels.csv to coverage preview PLYs.
2. Compare room raw coverage against a Meta Scene Mesh reference.
3. Run Meta-guided structural fusion diagnostics.

It does not consume room_raw_depth_frames. Those dense frame dumps are useful
for separate experiments, but they are not the old reference method.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "Tools"

DEFAULT_META_CANDIDATES = [
    Path(r"C:\Users\15319\Desktop\参考\meta_reference_sample.ply"),
    Path(r"C:\Users\15319\Desktop\参考\meta_surface_correction_reference_layer\meta_reference_sample.ply"),
    PROJECT_ROOT
    / "ScanCoverExports"
    / "MetaSceneMeshAuditSessions"
    / "ScanCover_MetaSceneMeshAudit_20260611_180459_512"
    / "stage0_weld"
    / "meta_scene_mesh_aligned_all_welded_eps1e-05.ply",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run old ScanCover room raw coverage pipeline.")
    parser.add_argument("input", type=Path, help="RepeatCoverage session folder or a parent folder containing sessions.")
    parser.add_argument("--out", type=Path, default=None, help="Output root. Default: <input>/old_reference_pipeline/run_*")
    parser.add_argument("--meta", type=Path, default=None, help="Meta Scene Mesh reference PLY/OBJ.")
    parser.add_argument(
        "--session-mode",
        choices=("latest", "all"),
        default="latest",
        help="Run Meta overlay/fusion for latest session or all discovered sessions.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    parser.add_argument("--skip-coverage", action="store_true", help="Skip coverage preview conversion.")
    parser.add_argument("--skip-overlay", action="store_true", help="Skip Raw/Meta overlay.")
    parser.add_argument("--skip-fusion", action="store_true", help="Skip Meta-guided fusion.")
    parser.add_argument("--meta-sample-points", type=int, default=350000)
    parser.add_argument("--close-threshold", type=float, default=0.06)
    parser.add_argument("--usable-threshold", type=float, default=0.12)
    parser.add_argument("--far-threshold", type=float, default=0.20)
    return parser.parse_args()


def newest_path(paths: list[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def find_meta(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None

    candidate = newest_path(DEFAULT_META_CANDIDATES)
    if candidate is not None:
        return candidate

    exports = PROJECT_ROOT / "ScanCoverExports"
    if exports.exists():
        matches = list(exports.rglob("meta_reference_sample.ply"))
        matches += list(exports.rglob("*welded*.ply"))
        return newest_path(matches)

    return None


def is_repeat_session(path: Path) -> bool:
    return (
        (path / "room_raw_coverage" / "room_raw_coverage_voxels.csv").exists()
        or (path / "room_raw_coverage_voxels.csv").exists()
    )


def find_sessions(root: Path) -> list[Path]:
    if is_repeat_session(root):
        return [root]

    sessions = []
    for csv_path in root.rglob("room_raw_coverage_voxels.csv"):
        parent = csv_path.parent
        session = parent.parent if parent.name == "room_raw_coverage" else parent
        if is_repeat_session(session):
            sessions.append(session)

    unique = sorted(set(sessions), key=lambda p: p.stat().st_mtime)
    return unique


def run(command: list[str]) -> None:
    print("+ " + " ".join(f'"{c}"' if " " in c else c for c in command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    input_root = args.input.resolve()
    if not input_root.exists():
        raise FileNotFoundError(input_root)

    run_root = args.out
    if run_root is None:
        stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_root = input_root / "old_reference_pipeline" / stamp
    run_root.mkdir(parents=True, exist_ok=True)

    sessions = find_sessions(input_root)
    if not sessions:
        raise FileNotFoundError(f"No room_raw_coverage_voxels.csv found under {input_root}")

    selected_sessions = sessions if args.session_mode == "all" else [sessions[-1]]
    meta = find_meta(args.meta)

    report: dict[str, object] = {
        "input": str(input_root),
        "output": str(run_root),
        "sessionMode": args.session_mode,
        "discoveredSessions": [str(p) for p in sessions],
        "selectedSessions": [str(p) for p in selected_sessions],
        "meta": str(meta) if meta is not None else None,
        "denseRawDepthFramesUsed": False,
    }

    if not args.skip_coverage:
        coverage_out = run_root / "01_coverage_preview"
        run(
            [
                args.python,
                str(TOOLS_DIR / "ScanCoverCoverageToPly.py"),
                str(input_root),
                "--out",
                str(coverage_out),
            ]
        )
        report["coveragePreview"] = str(coverage_out)

    if meta is None and (not args.skip_overlay or not args.skip_fusion):
        print("WARNING: No Meta reference found. Skipping overlay/fusion.", flush=True)
        report["metaMissing"] = True
    else:
        for session in selected_sessions:
            safe_name = session.name
            if not args.skip_overlay:
                overlay_out = run_root / "02_meta_overlay" / safe_name
                run(
                    [
                        args.python,
                        str(TOOLS_DIR / "ScanCoverRoomRawCoverageMetaOverlay.py"),
                        str(session),
                        "--meta",
                        str(meta),
                        "--out",
                        str(overlay_out),
                        "--meta-sample-points",
                        str(args.meta_sample_points),
                        "--close-threshold",
                        str(args.close_threshold),
                        "--usable-threshold",
                        str(args.usable_threshold),
                        "--far-threshold",
                        str(args.far_threshold),
                    ]
                )

            if not args.skip_fusion:
                fusion_out = run_root / "03_meta_guided_fusion" / safe_name
                run(
                    [
                        args.python,
                        str(TOOLS_DIR / "ScanCoverMetaGuidedStructureFusion.py"),
                        str(session),
                        "--meta",
                        str(meta),
                        "--out",
                        str(fusion_out),
                        "--meta-sample-points",
                        str(args.meta_sample_points),
                        "--close-threshold",
                        str(args.close_threshold),
                        "--usable-threshold",
                        str(args.usable_threshold),
                        "--far-threshold",
                        str(args.far_threshold),
                    ]
                )

    report_path = run_root / "old_reference_pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
