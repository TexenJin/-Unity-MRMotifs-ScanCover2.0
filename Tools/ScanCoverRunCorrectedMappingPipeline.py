#!/usr/bin/env python3
"""
Run the corrected ScanCover mapping pipeline with explicit output semantics.

Important naming contract:
- 00_snapshot_shell/snapshot_shell_mesh.ply is the current snapshot-first room shell.
  It does not use stable/candidate coverage labels; each 160x160 Raw Depth snapshot
  is treated as a topology-preserving local surface patch and stitched by world pose.
- 02_raw_mapping_input_unconstrained/mapping_input_candidate.ply is RAW-ONLY diagnostic.
  It is allowed to look fuzzy/thick/broken and must not be judged as final mapping.
- 03_meta_constrained_trusted_raw/layered_evidence_surface.ply is the current
  evidence view. trusted_raw_surface.ply is strict green only; evidence_supported_surface.ply
  is the relaxed green+cyan+yellow mapping seed.
- 04_scan_wanted_from_meta_gap/missing_scan_* are true rescan targets when a Meta
  reference exists. bad_observation_* are diagnostics, not direct rescan commands.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "Tools"

DEFAULT_META_CANDIDATES = [
    Path(r"C:\Users\15319\Desktop\参考\room_raw_meta_fusion_layers\meta_reference_sample.ply"),
    Path(r"C:\Users\15319\Desktop\参考\room_raw_meta_overlay\meta_reference_sample.ply"),
    Path(r"C:\Users\15319\Desktop\参考\meta_surface_correction_reference_layer\meta_reference_sample.ply"),
    ROOT / "ScanCoverExports" / "RepeatCoverageSessions" / "ScanCover_RepeatCoverage_20260612_173442_610" / "room_raw_meta_fusion_layers" / "meta_reference_sample.ply",
    ROOT / "ScanCoverExports" / "RepeatCoverageSessions" / "ScanCover_RepeatCoverage_20260612_173442_610" / "room_raw_meta_overlay" / "meta_reference_sample.ply",
]

RAW_MAPPING_ARGS = [
    "--voxel", "0.025",
    "--preview-voxel", "0.08",
    "--min-depth", "0.20",
    "--max-depth", "5.00",
    "--frame-stride", "1",
    "--min-hits", "3",
    "--stable-hits", "10",
    "--stable-sessions", "1",
    "--usable-depth-std", "0.18",
    "--usable-projection-risk", "0.35",
    "--usable-confidence", "0.30",
    "--hard-depth-std", "0.45",
    "--hard-projection-risk", "0.60",
    "--hard-confidence", "0.20",
    "--neighbor-vote-radius", "1",
    "--neighbor-vote-threshold", "0.55",
    "--neighbor-min-usable", "5",
    "--neighbor-normal-angle", "28",
    "--neighbor-depth-delta", "0.20",
    "--neighbor-rescue-depth-std", "0.18",
    "--write-csv",
]

SNAPSHOT_SHELL_ARGS = [
    "--pixel-step", "1",
    "--min-depth", "0.20",
    "--max-depth", "5.00",
    "--min-confidence", "0.0",
    "--max-edge", "0.18",
    "--max-depth-jump", "0.18",
    "--max-normal-angle", "70",
    "--preview-voxel", "0.08",
]

WANTED_ARGS = [
    "--target-voxel", "0.35",
    "--min-cluster-points", "450",
    "--max-targets", "24",
]

MESH_ARGS = [
    "--voxel", "0.025",
    "--normal-radius", "0.10",
    "--radius", "0.10",
    "--bpa-radius-scale", "1.8",
    "--poisson-depth", "8",
    "--sample-points", "120000",
]


def run(cmd: List[str], log: List[str]) -> None:
    text = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    log.append(text)
    print(f"[run] {text}")
    subprocess.run(cmd, check=True)


def find_meta(explicit: Optional[Path]) -> Optional[Path]:
    if explicit:
        p = explicit.expanduser().resolve()
        return p if p.exists() else None
    for p in DEFAULT_META_CANDIDATES:
        if p.exists():
            return p
    return None


def find_repeat_sessions(root: Path) -> List[Path]:
    root = root.resolve()
    sessions = []
    if (root / "room_raw_coverage" / "room_raw_coverage_voxels.csv").exists():
        sessions.append(root)
    for csv in root.rglob("room_raw_coverage_voxels.csv"):
        if csv.parent.name == "room_raw_coverage":
            sessions.append(csv.parent.parent)
    unique = []
    seen = set()
    for s in sessions:
        key = str(s.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(s.resolve())
    unique.sort(key=lambda p: p.stat().st_mtime)
    return unique


def write_readme(out: Path, input_root: Path, meta: Optional[Path], trusted_session: Optional[Path]) -> None:
    lines = [
        "ScanCover corrected mapping pipeline outputs",
        "==========================================",
        "",
        f"Input root: {input_root}",
        f"Meta reference: {meta if meta else 'NOT FOUND'}",
        f"Trusted Raw session used for Meta correction: {trusted_session if trusted_session else 'NONE'}",
        "",
        "00_snapshot_shell/",
        "  Main snapshot-first shell path. It reads room_raw_depth_snapshots, not room_raw_coverage stable labels.",
        "  snapshot_shell_mesh.ply preserves per-frame 160x160 pixel topology and stitches frames by exported world-space pose.",
        "  snapshot_shell_points.ply is the full valid Raw snapshot point set.",
        "  snapshot_shell_merged_preview.ply is an 8cm merged point preview for quick room-shell inspection.",
        "  This is now the first output to inspect when evaluating room shell capture.",
        "",
        "01_coverage_preview_raw_only/",
        "  Legacy raw coverage preview. Use it only as a diagnostic; grey/red does not mean snapshot shell evidence is unusable.",
        "",
        "01b_raw_shell_extractor/",
        "  Optional Raw-only shell extraction. This does not use Meta; it separates shell candidates, objects, boundaries, sparse tails, and rejected noise.",
        "",
        "02_raw_mapping_input_unconstrained/",
        "  Legacy Raw-only mapping diagnostics. mapping_input_candidate.ply is NOT the final snapshot shell seed.",
        "  If it looks thick, fuzzy, or broken, that is expected for unconstrained multi-frame Raw depth.",
        "",
        "03_meta_constrained_trusted_raw/",
        "  Main output to inspect first. trusted_raw_surface.ply means Meta structure landed onto stable Raw evidence.",
        "  This is the closest current artifact to: Meta gives structure, Raw gives true depth.",
        "  layered_evidence_surface.ply separates evidence grades: green trusted, cyan probable, yellow candidate, blue weak, red rejected.",
        "  evidence_supported_surface.ply contains green+cyan+yellow only; it is the practical relaxed evidence layer.",
        "  probable/candidate layers can guide meshing/training, but they do not replace the strict trusted_raw_surface contract.",
        "",
        "04_scan_wanted_from_meta_gap/",
        "  missing_scan_*: Meta says geometry exists but Raw coverage is absent nearby. These are real rescan hints.",
        "  processing_gap_*: Raw exists but current usable/mapping rules rejected it. This is a rule/fusion problem first.",
        "  bad_observation_*: Raw saw it, but it was unstable/risky. This is diagnostic, not automatically a rescan target.",
        "",
        "05_mesh_from_trusted_raw/",
        "  Optional mesh attempt. Default source is trusted_raw_surface.ply; --mesh-source evidence uses green+cyan+yellow evidence_supported_surface.ply.",
        "",
    ]
    (out / "README_OUTPUTS.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run corrected ScanCover mapping pipeline with separated output semantics.")
    ap.add_argument("input", type=Path, help="RepeatCoverageSessions root or one ScanCover_RepeatCoverage_* session")
    ap.add_argument("--out", type=Path, default=None, help="Output root. Default: <input>/corrected_mapping_pipeline_<timestamp>")
    ap.add_argument("--meta", type=Path, default=None, help="Meta reference sample ply. Auto-detects known reference paths when omitted.")
    ap.add_argument("--trusted-session", type=Path, default=None, help="Specific RepeatCoverage session for Meta correction. Default: latest session with room_raw_coverage_voxels.csv")
    ap.add_argument("--run-mesh", action="store_true", help="Also generate a mesh from trusted_raw_surface.ply")
    ap.add_argument("--mesh-source", choices=("trusted", "evidence"), default="trusted", help="Mesh source: strict trusted green or relaxed green+cyan+yellow evidence.")
    ap.add_argument("--run-raw-shell", action="store_true", help="Also run Raw-only shell extraction from room_raw_coverage_voxels.csv")
    ap.add_argument("--skip-snapshot-shell", action="store_true", help="Skip the snapshot-first room shell output.")
    ap.add_argument("--skip-raw-preview", action="store_true")
    ap.add_argument("--skip-raw-mapping", action="store_true")
    ap.add_argument("--skip-meta-correction", action="store_true")
    ap.add_argument("--skip-wanted", action="store_true")
    args = ap.parse_args()

    input_root = args.input.expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(input_root)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (args.out.expanduser().resolve() if args.out else input_root / f"corrected_mapping_pipeline_{ts}")
    out.mkdir(parents=True, exist_ok=True)

    meta = find_meta(args.meta)
    sessions = find_repeat_sessions(input_root)
    trusted_session = args.trusted_session.expanduser().resolve() if args.trusted_session else (sessions[-1] if sessions else None)
    if trusted_session and not (trusted_session / "room_raw_coverage" / "room_raw_coverage_voxels.csv").exists():
        raise FileNotFoundError(trusted_session / "room_raw_coverage" / "room_raw_coverage_voxels.csv")

    log: List[str] = []

    snapshot_shell = out / "00_snapshot_shell"
    if not args.skip_snapshot_shell:
        snapshot_shell.mkdir(parents=True, exist_ok=True)
        snapshot_dirs = list(input_root.rglob("room_raw_depth_snapshots")) if input_root.is_dir() else []
        if input_root.name == "room_raw_depth_snapshots" or (input_root / "room_raw_depth_snapshots").exists() or snapshot_dirs:
            run([sys.executable, str(TOOLS / "ScanCoverRawDepthSnapshotsToShell.py"), str(input_root), "--out", str(snapshot_shell), *SNAPSHOT_SHELL_ARGS], log)
        else:
            (snapshot_shell / "README_SKIPPED.txt").write_text(
                "Snapshot shell skipped because no room_raw_depth_snapshots folder was found.\n",
                encoding="utf-8",
            )

    if not args.skip_raw_preview:
        raw_preview = out / "01_coverage_preview_raw_only"
        raw_preview.mkdir(parents=True, exist_ok=True)
        run([sys.executable, str(TOOLS / "ScanCoverCoverageToPly.py"), str(input_root), "--out", str(raw_preview)], log)

    if args.run_raw_shell:
        shell_out = out / "01b_raw_shell_extractor"
        shell_out.mkdir(parents=True, exist_ok=True)
        shell_source = trusted_session if trusted_session else input_root
        run([sys.executable, str(TOOLS / "ScanCoverRawShellExtractor.py"), str(shell_source), "--out", str(shell_out)], log)

    raw_mapping = out / "02_raw_mapping_input_unconstrained"
    if not args.skip_raw_mapping:
        raw_mapping.mkdir(parents=True, exist_ok=True)
        run([sys.executable, str(TOOLS / "ScanCoverRawDepthFramesToMappingInput.py"), str(input_root), "--out", str(raw_mapping), *RAW_MAPPING_ARGS], log)

    trusted_out = out / "03_meta_constrained_trusted_raw"
    if not args.skip_meta_correction:
        trusted_out.mkdir(parents=True, exist_ok=True)
        if not meta:
            (trusted_out / "README_META_REFERENCE_MISSING.txt").write_text(
                "Meta reference sample was not found. Cannot create trusted_raw_surface.ply.\n"
                "Pass --meta <meta_reference_sample.ply> or restore C:\\Users\\15319\\Desktop\\参考.\n",
                encoding="utf-8",
            )
            print("[warn] Meta reference not found; skipped Meta-constrained trusted Raw output.")
        elif not trusted_session:
            (trusted_out / "README_REPEAT_SESSION_MISSING.txt").write_text(
                "No RepeatCoverage session with room_raw_coverage/room_raw_coverage_voxels.csv was found.\n",
                encoding="utf-8",
            )
            print("[warn] No RepeatCoverage room_raw_coverage_voxels.csv found; skipped Meta correction.")
        else:
            run([
                sys.executable,
                str(TOOLS / "ScanCoverMetaSurfaceCorrectionField.py"),
                str(trusted_session),
                "--meta", str(meta),
                "--out", str(trusted_out),
            ], log)

    if not args.skip_wanted:
        wanted_out = out / "04_scan_wanted_from_meta_gap"
        wanted_out.mkdir(parents=True, exist_ok=True)
        if meta and trusted_out.exists():
            wanted_source = trusted_out if (trusted_out / "trusted_raw_surface.ply").exists() else raw_mapping
            run([sys.executable, str(TOOLS / "ScanCoverBuildScanWantedList.py"), str(wanted_source), "--reference-ply", str(meta), "--out", str(wanted_out), *WANTED_ARGS], log)
        else:
            (wanted_out / "README_SKIPPED.txt").write_text(
                "Wanted list skipped because Meta reference or trusted output was unavailable.\n",
                encoding="utf-8",
            )

    if args.run_mesh:
        mesh_out = out / "05_mesh_from_trusted_raw"
        mesh_out.mkdir(parents=True, exist_ok=True)
        mesh_ply = trusted_out / ("evidence_supported_surface.ply" if args.mesh_source == "evidence" else "trusted_raw_surface.ply")
        if mesh_ply.exists():
            run([sys.executable, str(TOOLS / "ScanCoverTrustedSurfaceMeshing.py"), str(mesh_ply), "--out", str(mesh_out), *MESH_ARGS], log)
        else:
            (mesh_out / "README_SKIPPED.txt").write_text(
                f"Mesh skipped because 03_meta_constrained_trusted_raw/{mesh_ply.name} does not exist.\n",
                encoding="utf-8",
            )

    (out / "commands_used.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    write_readme(out, input_root, meta, trusted_session)

    print("\nDone.")
    print(f"Output: {out}")
    print("Inspect first: 00_snapshot_shell/snapshot_shell_mesh.ply")
    print("Snapshot merged preview: 00_snapshot_shell/snapshot_shell_merged_preview.ply")
    print("Legacy evidence diagnostic: 03_meta_constrained_trusted_raw/layered_evidence_surface.ply")
    print("Strict green only: 03_meta_constrained_trusted_raw/trusted_raw_surface.ply")
    print("Relaxed green+cyan+yellow: 03_meta_constrained_trusted_raw/evidence_supported_surface.ply")
    print("Do not judge final mapping by: 02_raw_mapping_input_unconstrained/mapping_input_candidate.ply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
