#!/usr/bin/env python3
"""Replay frozen Quest Evidence/v3 frames through the paper DTSDF/DMC chain.

Each capture session is fused independently because tracking origins are not
portable across sessions.  This is a cross-implementation and topology audit,
not a truth-accuracy benchmark: the recorded raw world point is paired with the
recorded neighbourhood-filtered normal, matching the paper's unfiltered-depth
plus bilaterally filtered-normal input contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from ScanCoverDirectionalTSDFCompositionValidation import (
    DirectionalGrid,
    extract_paper_dmc,
    paper_complete_corner_cells,
    topology_metrics,
    write_mesh,
)
from ScanCoverEvidenceV3OfflineValidation import (
    load_frame,
    read_manifest,
    select_evenly,
)


REQUESTED_BUFFERS = {
    "depth_metrics_rgba32f",
    "world_position_raw_rgba32f",
    "world_normal_raw_rgba32f",
    "world_normal_neighbour_rgba32f",
}

NORMAL_BUFFER_BY_SOURCE = {
    "raw": "world_normal_raw_rgba32f",
    "neighbour": "world_normal_neighbour_rgba32f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, nargs="*", default=None)
    parser.add_argument("--frames-per-session", type=int, default=12)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--voxel", type=float, default=0.045)
    parser.add_argument("--sdf-trunc", type=float, default=0.135)
    parser.add_argument("--paper-minimum-weight", type=float, default=1e-6)
    parser.add_argument(
        "--normal-source",
        choices=tuple(NORMAL_BUFFER_BY_SOURCE),
        default="neighbour",
        help="Frozen Quest normal buffer used by the paper normal-ray integration arm.",
    )
    parser.add_argument("--min-depth", type=float, default=0.35)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--verify-crc", action="store_true")
    return parser.parse_args()


def resolve_sessions(args: argparse.Namespace) -> list[Path]:
    if args.sessions:
        sessions = [path.resolve() for path in args.sessions]
    else:
        sessions = sorted(
            path.resolve() for path in args.root.glob("Evidence_*") if path.is_dir()
        )
    if not sessions:
        raise RuntimeError(f"no Evidence_* sessions found under {args.root}")
    return sessions


def integrate_session(
    args: argparse.Namespace,
    session: Path,
    output: Path,
) -> dict[str, Any]:
    rows = select_evenly(read_manifest(session), args.frames_per_session)
    if not rows:
        raise RuntimeError(f"no valid frames in {session}")
    grid = DirectionalGrid(
        args.voxel,
        args.sdf_trunc,
        0.35,
        True,
    )
    stride = max(1, int(args.pixel_stride))
    selected_hash = hashlib.sha256()
    accepted = 0
    valid_pixels = 0
    rejected_range = 0
    rejected_point_or_normal = 0
    frame_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for selected_index, row in enumerate(rows, start=1):
        frame_path = (session / row["file"]).resolve()
        frame = load_frame(frame_path, REQUESTED_BUFFERS, args.verify_crc)
        selected_hash.update(frame_path.name.encode("utf-8"))
        selected_hash.update(row.get("sha256", "").encode("ascii", errors="ignore"))
        metrics = frame.buffers["depth_metrics_rgba32f"]
        points = frame.buffers["world_position_raw_rgba32f"]
        normal_buffer = NORMAL_BUFFER_BY_SOURCE[args.normal_source]
        normals = frame.buffers[normal_buffer]
        shape = metrics.shape
        if shape[:3] != points.shape[:3] or shape[:3] != normals.shape[:3]:
            raise RuntimeError(f"buffer shape mismatch in {frame_path}")
        if shape[0] != 2:
            raise RuntimeError(f"complete binocular frame required: {frame_path}")

        frame_accepted = 0
        frame_valid = 0
        for eye in range(2):
            camera = np.asarray(
                frame.metadata["eyeWorldPositions"][eye], dtype=np.float64
            )
            metric = metrics[eye, ::stride, ::stride]
            point = points[eye, ::stride, ::stride]
            normal = normals[eye, ::stride, ::stride]
            radial = metric[..., 2]
            range_valid = (
                (metric[..., 3] > 0.0)
                & np.isfinite(radial)
                & (radial >= args.min_depth)
                & (radial <= args.max_depth)
            )
            geometry_valid = (
                (point[..., 3] > 0.0)
                & (normal[..., 3] > 0.0)
                & np.all(np.isfinite(point[..., :3]), axis=-1)
                & np.all(np.isfinite(normal[..., :3]), axis=-1)
                & (np.linalg.norm(normal[..., :3], axis=-1) > 1e-6)
            )
            valid = range_valid & geometry_valid
            valid_pixels += int(np.count_nonzero(valid))
            frame_valid += int(np.count_nonzero(valid))
            rejected_range += int(np.count_nonzero(~range_valid))
            rejected_point_or_normal += int(np.count_nonzero(range_valid & ~geometry_valid))
            flat_points = point[..., :3].reshape(-1, 3)
            flat_normals = normal[..., :3].reshape(-1, 3)
            flat_valid = valid.reshape(-1)
            for index in np.flatnonzero(flat_valid):
                if grid.integrate_paper_normal_raycast(
                    camera,
                    flat_points[index].astype(np.float64),
                    flat_normals[index].astype(np.float64),
                    args.min_depth,
                ):
                    accepted += 1
                    frame_accepted += 1
        frame_rows.append(
            {
                "selectedIndex": selected_index,
                "frame": int(frame.metadata["frameIndex"]),
                "validDepthNormalPixels": frame_valid,
                "acceptedNormalRays": frame_accepted,
            }
        )
        print(
            f"[quest-v3-paper {session.name} {selected_index}/{len(rows)}] "
            f"rays={accepted} candidates={len(grid.candidates)}",
            flush=True,
        )

    integration_ms = (time.perf_counter() - started) * 1000.0
    extraction_started = time.perf_counter()
    build = extract_paper_dmc(
        grid,
        args.paper_minimum_weight,
        regularize=True,
    )
    extraction_ms = (time.perf_counter() - extraction_started) * 1000.0
    output.mkdir(parents=True, exist_ok=True)
    mesh_name = (
        "quest_paper_dmc.ply"
        if args.normal_source == "neighbour"
        else f"quest_paper_dmc_{args.normal_source}.ply"
    )
    write_mesh(output / mesh_name, build)
    topology = topology_metrics(build)
    complete_cells = paper_complete_corner_cells(grid, args.paper_minimum_weight)
    audit = vars(build.audit)
    candidate_count = len(grid.candidates)
    complete_count = len(complete_cells)
    report = {
        "schema": "scancover.quest_evidence_v3.paper_dtsdf_replay.v1",
        "scope": "frozen Quest session replay; no truth and no cross-session fusion",
        "session": str(session),
        "parameters": {
            "framesPerSession": args.frames_per_session,
            "selectedFrames": len(rows),
            "pixelStride": stride,
            "sourceResolution": [int(shape[2]), int(shape[1])],
            "effectiveRasterPerEye": [
                int(math.ceil(shape[2] / stride)),
                int(math.ceil(shape[1] / stride)),
            ],
            "voxelMeters": args.voxel,
            "truncationMeters": args.sdf_trunc,
            "paperMinimumWeight": args.paper_minimum_weight,
            "pointSource": "world_position_raw_rgba32f",
            "normalSource": NORMAL_BUFFER_BY_SOURCE[args.normal_source],
        },
        "inputAudit": {
            "selectedFrameSequenceSha256": selected_hash.hexdigest(),
            "validDepthNormalPixels": valid_pixels,
            "acceptedNormalRays": accepted,
            "rejectedRangePixels": rejected_range,
            "rejectedPointOrNormalPixels": rejected_point_or_normal,
            "frames": frame_rows,
        },
        "integrationAudit": {
            "integrationMs": integration_ms,
            "normalRays": grid.paper_normal_rays,
            "traversedVoxels": grid.paper_traversed_voxels,
            "integratedVoxels": grid.paper_integrated_voxels,
            "voxelDirectionWrites": grid.voxel_updates,
            "writtenVoxelKeys": len(
                {
                    key
                    for layer in grid.values
                    for key, record in layer.items()
                    if float(record[1]) > 1e-8
                }
            ),
            "candidateCells": candidate_count,
            "completeCornerCells": complete_count,
            "completeCornerCellRatioOfCandidates": (
                complete_count / candidate_count if candidate_count else 0.0
            ),
        },
        "extractionAudit": {
            "extractionMs": extraction_ms,
            "paperCellsWithCompleteDirection": audit[
                "paper_cells_with_complete_direction"
            ],
            "paperRawCrossingCells": audit["paper_raw_crossing_cells"],
            "paperFilteredCrossingCells": audit["paper_filtered_crossing_cells"],
            "paperVotedCrossingCells": audit["paper_voted_crossing_cells"],
            "missingUnwrittenCornerDirections": audit[
                "paper_directions_missing_unwritten_corner"
            ],
            "blockedOnlyByWeightDirections": audit[
                "paper_directions_blocked_only_by_weight"
            ],
            "unmeasuredEdgeDeferredTriangles": audit[
                "paper_unmeasured_edge_deferred_triangles"
            ],
            "unresolvedEdgeSlots": audit["paper_unresolved_edge_slots"],
        },
        "topology": topology,
        "gates": {
            "meshGenerated": topology["triangles"] > 0,
            "nonManifoldFree": topology["nonManifoldEdges"] == 0,
            "measuredEdgeDecisionsComplete": (
                audit["paper_unmeasured_edge_deferred_triangles"] == 0
                and audit["paper_unresolved_edge_slots"] == 0
            ),
        },
    }
    report_name = (
        "quest_directional_replay_report.json"
        if args.normal_source == "neighbour"
        else f"quest_directional_replay_report_{args.normal_source}.json"
    )
    (output / report_name).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    for session in resolve_sessions(args):
        reports.append(integrate_session(args, session, args.out / session.name))
    summary = {
        "schema": "scancover.quest_evidence_v3.paper_dtsdf_replay_summary.v1",
        "sessions": [
            {
                "name": Path(report["session"]).name,
                "acceptedNormalRays": report["inputAudit"]["acceptedNormalRays"],
                "completeCornerCellRatioOfCandidates": report["integrationAudit"][
                    "completeCornerCellRatioOfCandidates"
                ],
                "triangles": report["topology"]["triangles"],
                "boundaryEdgesPerKTriangles": report["topology"][
                    "boundaryEdgesPerKTriangles"
                ],
                "nonManifoldEdges": report["topology"]["nonManifoldEdges"],
                "gates": report["gates"],
            }
            for report in reports
        ],
        "allGatesPass": all(
            all(bool(value) for value in report["gates"].values())
            for report in reports
        ),
        "interpretation": (
            "Topology/cross-implementation replay only. Without room truth, this cannot "
            "claim absolute surface accuracy or visible recovery."
        ),
    }
    summary_name = (
        "quest_directional_replay_summary.json"
        if args.normal_source == "neighbour"
        else f"quest_directional_replay_summary_{args.normal_source}.json"
    )
    (args.out / summary_name).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
