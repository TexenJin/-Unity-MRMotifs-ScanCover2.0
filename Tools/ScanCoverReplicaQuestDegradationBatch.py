#!/usr/bin/env python3
"""Run one fixed Quest-degradation validation protocol across Replica rooms."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica-root", type=Path, required=True)
    parser.add_argument("--degradation-model", type=Path, required=True)
    parser.add_argument(
        "--real-summary",
        type=Path,
        default=Path(
            r"E:\PCAII\Unity-MRMotifs-ScanCover-main\ScanCoverHandoff\ExternalData"
            r"\Desktop_Destill_test_outputs\quest3_real_20260604_222905_671_summary_v02"
            r"\combined_observation_summary.json"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--coverage-samples", type=int, default=50000)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--ideal-observer", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    experiment = script_dir / "ScanCoverQuest3VirtualCloneExperiment.py"
    summary = script_dir / "ScanCoverQuest3VirtualCloneBenchmarkSummary.py"
    meshes = sorted(args.replica_root.glob("office*.ply")) + sorted(args.replica_root.glob("room*.ply"))
    if not meshes:
        raise FileNotFoundError(f"No Replica office/room PLY files under {args.replica_root}")
    if not args.degradation_model.exists():
        raise FileNotFoundError(args.degradation_model)
    if not args.real_summary.exists():
        raise FileNotFoundError(args.real_summary)

    args.out.mkdir(parents=True, exist_ok=True)
    reports: list[Path] = []
    run_manifest = {
        "schema": "ScanCoverReplicaQuestDegradationBatch/v1",
        "degradationModel": str(args.degradation_model.resolve()),
        "realObservationSummary": str(args.real_summary.resolve()),
        "fixedProtocol": {
            "poseSource": "auto-scan",
            "scanPattern": "stratified-slices",
            "frames": args.frames,
            "rayGrid": [args.width, args.height],
            "surfaceCoverageSamples": args.coverage_samples,
            "seed": 15319,
            "idealObserver": bool(args.ideal_observer),
        },
        "rooms": [mesh.stem for mesh in meshes],
    }
    (args.out / "batch_protocol.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for index, mesh in enumerate(meshes, start=1):
        report = args.out / f"replica_auto-scan_{mesh.stem}" / "virtual_clone_similarity_report.json"
        reports.append(report)
        if args.skip_existing and report.exists():
            print(f"[batch {index}/{len(meshes)}] reuse {mesh.stem}", flush=True)
            continue
        print(f"[batch {index}/{len(meshes)}] run {mesh.stem}", flush=True)
        log_path = args.out / f"{mesh.stem}.log"
        command = [
            sys.executable,
            str(experiment),
            "--truth-mesh", str(mesh),
            "--degradation-model", str(args.degradation_model),
            "--real-summary", str(args.real_summary),
            "--pose-source", "auto-scan",
            "--scan-pattern", "stratified-slices",
            "--mode", "replica",
            "--max-frames", str(args.frames),
            "--width", str(args.width),
            "--height", str(args.height),
            "--surface-coverage-samples", str(args.coverage_samples),
            "--seed", "15319",
            "--out", str(args.out),
        ]
        if args.ideal_observer:
            command.append("--ideal-observer")
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)

    subprocess.run(
        [sys.executable, str(summary), *[str(report) for report in reports], "--out", str(args.out / "BenchmarkSummary")],
        check=True,
    )
    print(f"[batch] complete: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
