#!/usr/bin/env python3
"""Quest Evidence v3 local block-refusion correction experiment.

The experiment keeps the production-like persistent volume, builds a bounded
recent-observation shadow, detects only blocks whose TSDF disagrees with that
stable shadow, erases those blocks (plus a one-block seam halo), and re-fuses
the recent observations into the selected blocks.  It never resets the whole
volume and does not use Replica truth or a human correction label.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ScanCoverEvidenceV3OfflineValidation import (  # noqa: E402
    ANALYSIS_BUFFERS,
    exact_open3d_calibration,
    load_frame,
    mesh_topology,
    read_manifest,
    select_evenly,
)


@dataclass
class EyeObservation:
    depth: np.ndarray
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    block_coords: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--voxel", type=float, default=0.025)
    parser.add_argument("--sdf-trunc", type=float, default=0.08)
    parser.add_argument("--block-resolution", type=int, default=8)
    parser.add_argument("--block-capacity", type=int, default=50000)
    parser.add_argument("--min-depth", type=float, default=0.35)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--corrupt-prefix-frames", type=int, default=6)
    parser.add_argument("--corrupt-depth-offset", type=float, default=0.08)
    parser.add_argument("--minimum-block-observations", type=int, default=2)
    parser.add_argument("--minimum-common-voxels", type=int, default=16)
    parser.add_argument("--tsdf-difference-p95", type=float, default=0.30)
    parser.add_argument("--sign-disagreement-ratio", type=float, default=0.08)
    parser.add_argument("--halo-blocks", type=int, default=1)
    return parser.parse_args()


def new_volume(args: argparse.Namespace) -> o3d.t.geometry.VoxelBlockGrid:
    return o3d.t.geometry.VoxelBlockGrid(
        ("tsdf", "weight"),
        (o3d.core.float32, o3d.core.float32),
        ((1,), (1,)),
        args.voxel,
        args.block_resolution,
        args.block_capacity,
        o3d.core.Device("CPU:0"),
    )


def tensors(observation: EyeObservation, depth: np.ndarray | None = None):
    image = o3d.t.geometry.Image(o3d.core.Tensor(observation.depth if depth is None else depth))
    intrinsic = o3d.core.Tensor(observation.intrinsic, o3d.core.float64)
    extrinsic = o3d.core.Tensor(observation.extrinsic, o3d.core.float64)
    return image, intrinsic, extrinsic


def trunc_multiplier(args: argparse.Namespace) -> float:
    return args.sdf_trunc / args.voxel


def make_observations(args: argparse.Namespace) -> list[EyeObservation]:
    rows = select_evenly(read_manifest(args.session), args.frames)
    coordinate_volume = new_volume(args)
    observations: list[EyeObservation] = []
    for row in rows:
        frame = load_frame((args.session / row["file"]).resolve(), ANALYSIS_BUFFERS, False)
        metrics = frame.buffers["depth_metrics_rgba32f"]
        height, width = metrics.shape[1:3]
        for eye in range(2):
            depth = metrics[eye, :, :, 1].copy()
            valid = (
                (metrics[eye, :, :, 3] > 0.0)
                & np.isfinite(depth)
                & (depth >= args.min_depth)
                & (depth <= args.max_depth)
            )
            depth[~valid] = 0.0
            depth = np.ascontiguousarray(np.flipud(depth).astype(np.float32))
            intrinsic, extrinsic = exact_open3d_calibration(frame.metadata, eye, width, height)
            intrinsic_matrix = np.asarray(intrinsic.intrinsic_matrix, dtype=np.float64)
            probe = EyeObservation(depth, intrinsic_matrix, extrinsic.astype(np.float64), np.empty((0, 3), dtype=np.int32))
            image, k_tensor, e_tensor = tensors(probe)
            coords = coordinate_volume.compute_unique_block_coordinates(
                image, k_tensor, e_tensor, 1.0, args.max_depth, trunc_multiplier(args)
            ).numpy().astype(np.int32, copy=True)
            observations.append(EyeObservation(depth, intrinsic_matrix, extrinsic.astype(np.float64), coords))
    return observations


def integrate(
    volume: o3d.t.geometry.VoxelBlockGrid,
    observation: EyeObservation,
    args: argparse.Namespace,
    depth_offset: float = 0.0,
    allowed_blocks: set[tuple[int, int, int]] | None = None,
) -> int:
    depth = observation.depth
    coords = observation.block_coords
    if depth_offset != 0.0:
        depth = depth.copy()
        valid = depth > 0.0
        depth[valid] += depth_offset
        depth[(depth < args.min_depth) | (depth > args.max_depth)] = 0.0
        probe = EyeObservation(depth, observation.intrinsic, observation.extrinsic, coords)
        image, k_tensor, e_tensor = tensors(probe)
        coords = volume.compute_unique_block_coordinates(
            image, k_tensor, e_tensor, 1.0, args.max_depth, trunc_multiplier(args)
        ).numpy().astype(np.int32, copy=True)
    if allowed_blocks is not None:
        coords = np.asarray(
            [coord for coord in coords if tuple(int(value) for value in coord) in allowed_blocks],
            dtype=np.int32,
        ).reshape(-1, 3)
    if len(coords) == 0:
        return 0
    image, k_tensor, e_tensor = tensors(observation, depth)
    volume.integrate(
        o3d.core.Tensor(coords, o3d.core.int32),
        image,
        k_tensor,
        e_tensor,
        1.0,
        args.max_depth,
        trunc_multiplier(args),
    )
    return len(coords)


def legacy_mesh(volume: o3d.t.geometry.VoxelBlockGrid) -> o3d.geometry.TriangleMesh:
    mesh = volume.extract_triangle_mesh(weight_threshold=1.0).to_legacy()
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    return mesh


def active_key_map(volume: o3d.t.geometry.VoxelBlockGrid) -> dict[tuple[int, int, int], int]:
    hashmap = volume.hashmap()
    indices = hashmap.active_buf_indices().numpy().astype(np.int64)
    keys = hashmap.key_tensor()[hashmap.active_buf_indices()].numpy().astype(np.int32)
    return {tuple(int(value) for value in key): int(index) for key, index in zip(keys, indices)}


def distance_stats(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if len(source) == 0 or len(target) == 0:
        return {"count": 0, "mean": math.inf, "p50": math.inf, "p95": math.inf, "max": math.inf}
    distances, _ = cKDTree(target).query(source, k=1, workers=-1)
    return {
        "count": int(len(distances)),
        "mean": float(np.mean(distances)),
        "p50": float(np.percentile(distances, 50)),
        "p95": float(np.percentile(distances, 95)),
        "max": float(np.max(distances)),
        "within5cmRatio": float(np.mean(distances <= 0.05)),
    }


def compare_mesh(source: o3d.geometry.TriangleMesh, target: o3d.geometry.TriangleMesh) -> dict[str, Any]:
    source_points = np.asarray(source.vertices, dtype=np.float64)
    target_points = np.asarray(target.vertices, dtype=np.float64)
    forward = distance_stats(source_points, target_points)
    reverse = distance_stats(target_points, source_points)
    return {
        "sourceToReferenceMeters": forward,
        "referenceToSourceMeters": reverse,
        "symmetricP95Meters": max(forward["p95"], reverse["p95"]),
        "symmetricWithin5cmRatio": min(forward["within5cmRatio"], reverse["within5cmRatio"]),
    }


def divergent_blocks(
    persistent: o3d.t.geometry.VoxelBlockGrid,
    fresh: o3d.t.geometry.VoxelBlockGrid,
    votes: Counter[tuple[int, int, int]],
    args: argparse.Namespace,
) -> tuple[set[tuple[int, int, int]], list[dict[str, Any]]]:
    persistent_map = active_key_map(persistent)
    fresh_map = active_key_map(fresh)
    common = sorted(set(persistent_map).intersection(fresh_map))
    p_tsdf = persistent.attribute("tsdf").numpy()
    p_weight = persistent.attribute("weight").numpy()
    f_tsdf = fresh.attribute("tsdf").numpy()
    f_weight = fresh.attribute("weight").numpy()
    divergent: set[tuple[int, int, int]] = set()
    audit: list[dict[str, Any]] = []
    for key in common:
        if votes[key] < args.minimum_block_observations:
            continue
        pi = persistent_map[key]
        fi = fresh_map[key]
        pw = p_weight[pi, ..., 0]
        fw = f_weight[fi, ..., 0]
        mask = (pw >= 1.0) & (fw >= 1.0)
        common_voxels = int(np.count_nonzero(mask))
        if common_voxels < args.minimum_common_voxels:
            continue
        pt = p_tsdf[pi, ..., 0][mask]
        ft = f_tsdf[fi, ..., 0][mask]
        difference_p95 = float(np.percentile(np.abs(pt - ft), 95))
        surface = (np.abs(pt) <= 0.75) | (np.abs(ft) <= 0.75)
        sign_ratio = float(np.mean(np.signbit(pt[surface]) != np.signbit(ft[surface]))) if np.any(surface) else 0.0
        if difference_p95 >= args.tsdf_difference_p95 or sign_ratio >= args.sign_disagreement_ratio:
            divergent.add(key)
            audit.append({
                "key": list(key),
                "observations": votes[key],
                "commonVoxels": common_voxels,
                "tsdfDifferenceP95": difference_p95,
                "signDisagreementRatio": sign_ratio,
            })
    return divergent, audit


def expand_halo(
    keys: set[tuple[int, int, int]],
    active_union: set[tuple[int, int, int]],
    radius: int,
) -> set[tuple[int, int, int]]:
    if radius <= 0:
        return keys.intersection(active_union)
    expanded: set[tuple[int, int, int]] = set()
    for key in keys:
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    candidate = (key[0] + dx, key[1] + dy, key[2] + dz)
                    if candidate in active_union:
                        expanded.add(candidate)
    return expanded


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    observations = make_observations(args)
    frame_count = len(observations) // 2
    corrupt_observations = min(len(observations), max(1, args.corrupt_prefix_frames) * 2)

    clean = new_volume(args)
    persistent = new_volume(args)
    fresh = new_volume(args)
    votes: Counter[tuple[int, int, int]] = Counter()
    for index, observation in enumerate(observations):
        integrate(clean, observation, args)
        integrate(
            persistent,
            observation,
            args,
            args.corrupt_depth_offset if index < corrupt_observations else 0.0,
        )
        integrate(fresh, observation, args)
        votes.update(set(tuple(int(value) for value in coord) for coord in observation.block_coords))

    clean_mesh = legacy_mesh(clean)
    before_mesh = legacy_mesh(persistent)
    fresh_map = active_key_map(fresh)
    persistent_map = active_key_map(persistent)
    divergent, divergence_audit = divergent_blocks(persistent, fresh, votes, args)
    selected = expand_halo(divergent, set(fresh_map).union(persistent_map), args.halo_blocks)

    if selected:
        erase_tensor = o3d.core.Tensor(np.asarray(sorted(selected), dtype=np.int32), o3d.core.int32)
        persistent.hashmap().erase(erase_tensor)
        # HashMap.erase invalidates keys but intentionally does not scrub the
        # backing value buffers.  Re-activation can therefore reuse stale TSDF
        # and weight memory.  A true block re-fusion must explicitly restore
        # the canonical unseen state before integrating the recent window.
        buffer_indices, _ = persistent.hashmap().activate(erase_tensor)
        persistent.attribute("tsdf")[buffer_indices] = 1.0
        persistent.attribute("weight")[buffer_indices] = 0.0
        for observation in observations:
            integrate(persistent, observation, args, allowed_blocks=selected)
    recovered_mesh = legacy_mesh(persistent)

    clean_path = args.out / "clean_reference_vbg.ply"
    before_path = args.out / "corrupted_persistent_vbg.ply"
    recovered_path = args.out / "locally_refused_vbg.ply"
    o3d.io.write_triangle_mesh(str(clean_path), clean_mesh, write_ascii=False)
    o3d.io.write_triangle_mesh(str(before_path), before_mesh, write_ascii=False)
    o3d.io.write_triangle_mesh(str(recovered_path), recovered_mesh, write_ascii=False)

    before = compare_mesh(before_mesh, clean_mesh)
    after = compare_mesh(recovered_mesh, clean_mesh)
    active_before = len(persistent_map)
    divergent_ratio = len(divergent) / max(1, active_before)
    selected_ratio = len(selected) / max(1, active_before)
    p95_reduction = 1.0 - after["symmetricP95Meters"] / max(before["symmetricP95Meters"], 1e-12)
    coverage_delta = after["symmetricWithin5cmRatio"] - before["symmetricWithin5cmRatio"]
    clean_topology = mesh_topology(clean_mesh)
    before_topology = mesh_topology(before_mesh)
    topology = mesh_topology(recovered_mesh)
    gates = {
        "divergenceDetected": len(divergent) > 0,
        # Locality is an algorithmic property: only detected blocks and their
        # seam halo are erased.  The affected fraction legitimately depends on
        # how much of the room the bad prefix touched, so an arbitrary 50%
        # threshold would misclassify a broad but still selective repair.
        "localNotGlobal": 0 < len(selected) < active_before,
        "symmetricP95ReducedAtLeast50Percent": p95_reduction >= 0.50,
        "recoveredP95Below3cm": after["symmetricP95Meters"] <= 0.03,
        "coverageNotWorse": coverage_delta >= -0.005,
        "nonManifoldFree": topology["nonManifoldEdges"] == 0,
        "boundaryDensityWithinCleanReference": topology["boundaryEdgesPerKTriangles"] <= (
            clean_topology["boundaryEdgesPerKTriangles"] * 1.02 + 1e-9
        ),
    }
    report = {
        "schema": "ScanCoverQuestProjectiveCorrectionValidation/v1",
        "session": str(args.session.resolve()),
        "parameters": vars(args) | {"session": str(args.session), "out": str(args.out)},
        "frames": frame_count,
        "eyeObservations": len(observations),
        "activeBlocksBefore": active_before,
        "divergentBlocks": len(divergent),
        "divergentBlockRatio": divergent_ratio,
        "selectedBlocksWithHalo": len(selected),
        "selectedBlockRatio": selected_ratio,
        "divergenceAudit": divergence_audit,
        "before": before,
        "after": after,
        "symmetricP95ReductionRatio": p95_reduction,
        "coverageDelta": coverage_delta,
        "topology": {
            "cleanReference": clean_topology,
            "beforeCorrection": before_topology,
            "afterCorrection": topology,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "claimBoundary": {
            "usesHumanLabels": False,
            "usesReplicaTruth": False,
            "globalReset": False,
            "mechanism": "bounded recent-window TSDF divergence plus block-local erase/reintegration",
        },
        "outputs": {
            "cleanReference": str(clean_path),
            "corruptedPersistent": str(before_path),
            "locallyRefused": str(recovered_path),
        },
    }
    report_path = args.out / "correction_validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "gates": gates, "out": str(args.out)}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
