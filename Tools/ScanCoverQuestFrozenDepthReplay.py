#!/usr/bin/env python3
"""Deterministically replay recorded Quest depth into the offline DTSDF/DMC chain.

This is deliberately a replay of the sensor-facing boundary, not a truth test.
SCQ3BIN2 stores per-pixel radial linear depth, validity/confidence, a filtered
world-position buffer, and camera/reprojection metadata.  Historical captures
do not contain the GPU world-normal buffer, so normals are recomputed from the
recorded linear depth and this limitation is made explicit in the report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from ScanCoverDirectionalTSDFCompositionValidation import (
    DIRECTION_VECTORS,
    DirectionalGrid,
    extract_composed,
    extract_tsdf_hermite_feature_points,
    extract_tsdf_hermite_ledger_dual_mesh,
    topology_metrics,
    write_mesh,
)
from ScanCoverQuestCloneCaptureAnalyze import EyeHeader, find_session, load_manifest, read_eye


DIRECTION_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Capture root or one RepeatCoverage session")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=12, help="Evenly selected binocular pairs; 0 means all")
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--voxel", type=float, default=0.04)
    parser.add_argument("--sdf-trunc", type=float, default=0.12)
    parser.add_argument("--minimum-weight", type=float, default=0.75)
    parser.add_argument("--soft-direction-threshold", type=float, default=0.35)
    parser.add_argument("--valid-gradient-dot", type=float, default=0.15)
    parser.add_argument("--parallel-dot", type=float, default=0.82)
    parser.add_argument("--edge-merge-voxel-ratio", type=float, default=0.35)
    parser.add_argument("--feature-angle-degrees", type=float, default=32.0)
    parser.add_argument("--feature-min-family-support-ratio", type=float, default=0.12)
    parser.add_argument("--feature-rank-ratio", type=float, default=0.08)
    parser.add_argument("--min-depth", type=float, default=0.35)
    parser.add_argument("--max-depth", type=float, default=6.0)
    parser.add_argument("--min-confidence", type=float, default=0.01)
    parser.add_argument("--smoothing-depth-delta", type=float, default=0.08)
    parser.add_argument("--eye-baseline", type=float, default=0.063)
    parser.add_argument(
        "--input-source",
        choices=("linear-depth", "recorded-world"),
        default="linear-depth",
        help="Reconstruct from per-pixel radial depth, or replay the already filtered world-position buffer.",
    )
    parser.add_argument("--max-preview-points", type=int, default=200000)
    return parser.parse_args()


def select_rows(rows: list[dict[str, str]], requested: int) -> list[dict[str, str]]:
    if requested <= 0 or requested >= len(rows):
        return rows
    indices = np.rint(np.linspace(0, len(rows) - 1, requested)).astype(np.int64)
    return [rows[int(index)] for index in np.unique(indices)]


def quaternion_matrix(rotation: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(rotation, dtype=np.float64)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def estimated_head_center(header: EyeHeader, baseline: float) -> np.ndarray:
    """Recover Camera.main position used to write meta.b from a stereo-eye pose.

    Unity's recorded stereo matrices use a handedness in which the stored local
    +X basis points from the right eye toward the centre.  The sign below was
    verified against headPoseAtExport in the capture JSON (sub-millimetre to
    low-millimetre agreement for static frames).
    """

    right = quaternion_matrix(header.camera_rotation)[:, 0]
    sign = 1.0 if header.eye == 1 else -1.0
    return header.camera_position + sign * right * (baseline * 0.5)


def reconstruct_linear_depth_points(
    header: EyeHeader,
    records: np.ndarray,
    baseline: float,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not header.has_pose:
        raise ValueError(f"frame {header.frame} is missing pose provenance")
    if header.count != header.width * header.height:
        raise ValueError(
            f"frame {header.frame} is not a dense image: count={header.count} "
            f"resolution={header.width}x{header.height}"
        )

    width, height = header.width, header.height
    linear_depth = records["meta"][:, 2].astype(np.float64)
    confidence = records["meta"][:, 1].astype(np.float64)
    valid = (
        (records["valid"] != 0)
        & (records["meta"][:, 0] >= 0.5)
        & np.isfinite(linear_depth)
        & (linear_depth >= min_depth)
        & (linear_depth <= max_depth)
    )

    recorded_world = records["point"].astype(np.float64)
    eye = np.asarray(header.camera_position, dtype=np.float64)
    head = estimated_head_center(header, baseline)
    # SCQ3BIN2/v2 stores radial linear depth but not the source depth01 needed
    # to unproject the exact point through the recorded reprojection matrix.
    # Its filtered world point is nevertheless on the corresponding eye ray to
    # within the local 8 cm edge-aware smoothing band.  Use that stored ray as
    # the direction seed, then restore the unfiltered radial distance from
    # meta.b.  Re-running the production smoothing after this reconstruction is
    # validated against the stored filtered point for every replayed frame.
    directions = recorded_world - eye[None, :]
    direction_length = np.linalg.norm(directions, axis=1)
    ray_valid = (
        np.all(np.isfinite(recorded_world), axis=1)
        & np.isfinite(direction_length)
        & (direction_length > 1e-8)
    )
    directions[ray_valid] /= direction_length[ray_valid, None]

    # The shader stores cameraDistanceMeters measured from Camera.main, while
    # the reprojection matrix and serialized pose are eye-specific.  Intersect
    # each eye ray with the sphere centred on the estimated head position.
    offset = eye - head
    b = directions @ offset
    c = float(np.dot(offset, offset)) - linear_depth * linear_depth
    discriminant = b * b - c
    valid &= ray_valid & np.isfinite(discriminant) & (discriminant >= 0.0)
    distance_on_ray = np.zeros(header.count, dtype=np.float64)
    distance_on_ray[valid] = -b[valid] + np.sqrt(discriminant[valid])
    valid &= distance_on_ray > 0.0

    points = np.full((header.count, 3), np.nan, dtype=np.float64)
    points[valid] = eye[None, :] + directions[valid] * distance_on_ray[valid, None]
    return points.reshape(height, width, 3), valid.reshape(height, width), confidence.reshape(height, width)


def reconstruct_recorded_world_points(
    header: EyeHeader,
    records: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width, height = header.width, header.height
    points = records["point"].astype(np.float64).reshape(height, width, 3)
    linear_depth = records["meta"][:, 2].astype(np.float64).reshape(height, width)
    confidence = records["meta"][:, 1].astype(np.float64).reshape(height, width)
    valid = (
        (records["valid"].reshape(height, width) != 0)
        & (records["meta"][:, 0].reshape(height, width) >= 0.5)
        & np.all(np.isfinite(points), axis=2)
        & np.isfinite(linear_depth)
        & (linear_depth >= min_depth)
        & (linear_depth <= max_depth)
    )
    points[~valid] = np.nan
    return points, valid, confidence


def preprocess_points_and_normals(
    raw_points: np.ndarray,
    valid: np.ndarray,
    linear_depth: np.ndarray,
    smoothing_depth_delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror ScanCoverDepthPreprocessor.compute using recorded linear depth."""

    height, width = valid.shape
    filtered = np.full_like(raw_points, np.nan)
    normals = np.full_like(raw_points, np.nan)
    normal_valid = np.zeros_like(valid)
    for y in range(height):
        for x in range(width):
            if not valid[y, x]:
                continue
            center = raw_points[y, x]
            depth = linear_depth[y, x]
            total = center.copy()
            weight = 1.0
            neighbors: dict[str, tuple[np.ndarray, bool]] = {}
            for name, nx, ny in (
                ("left", x - 1, y),
                ("right", x + 1, y),
                ("down", x, y - 1),
                ("up", x, y + 1),
            ):
                usable = 0 <= nx < width and 0 <= ny < height and bool(valid[ny, nx])
                point = raw_points[ny, nx] if usable else np.zeros(3, dtype=np.float64)
                neighbors[name] = (point, usable)
                if usable and abs(float(linear_depth[ny, nx] - depth)) <= smoothing_depth_delta:
                    total += point
                    weight += 1.0
            filtered_center = total / weight
            filtered[y, x] = filtered_center

            left, left_valid = neighbors["left"]
            right, right_valid = neighbors["right"]
            down, down_valid = neighbors["down"]
            up, up_valid = neighbors["up"]
            if not (left_valid or right_valid) or not (down_valid or up_valid):
                continue
            horizontal = (
                right - left
                if left_valid and right_valid
                else (right - filtered_center if right_valid else filtered_center - left)
            )
            vertical = (
                up - down
                if down_valid and up_valid
                else (up - filtered_center if up_valid else filtered_center - down)
            )
            normal = np.cross(vertical, horizontal)
            length = float(np.linalg.norm(normal))
            if not math.isfinite(length) or length <= 1e-8:
                continue
            normals[y, x] = normal / length
            normal_valid[y, x] = True
    return filtered, normals, normal_valid


@dataclass
class LedgerDirectionalGrid(DirectionalGrid):
    update_counts: list[dict[tuple[int, int, int], int]] = field(
        default_factory=lambda: [dict() for _ in range(6)]
    )
    first_sequences: list[dict[tuple[int, int, int], int]] = field(
        default_factory=lambda: [dict() for _ in range(6)]
    )
    last_sequences: list[dict[tuple[int, int, int], int]] = field(
        default_factory=lambda: [dict() for _ in range(6)]
    )
    current_sequence: int = -1

    def _integrate_voxel(
        self, direction: int, key: tuple[int, int, int], tsdf: float, weight: float
    ) -> None:
        before = self.values[direction].get(key)
        before_weight = float(before[1]) if before is not None else 0.0
        super()._integrate_voxel(direction, key, tsdf, weight)
        after = self.values[direction].get(key)
        after_weight = float(after[1]) if after is not None else 0.0
        if after_weight <= before_weight + 1e-12:
            return
        self.update_counts[direction][key] = self.update_counts[direction].get(key, 0) + 1
        self.first_sequences[direction].setdefault(key, self.current_sequence)
        self.last_sequences[direction][key] = self.current_sequence


def quantiles(values: list[float]) -> dict[str, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_voxel_ledger(path: Path, grid: LedgerDirectionalGrid) -> int:
    rows: list[dict[str, Any]] = []
    for direction, layer in enumerate(grid.values):
        for key in sorted(layer):
            tsdf_sum, weight = layer[key]
            mean = tsdf_sum / weight if weight > 0.0 else 1.0
            rows.append(
                {
                    "directionIndex": direction,
                    "direction": DIRECTION_NAMES[direction],
                    "x": key[0],
                    "y": key[1],
                    "z": key[2],
                    "tsdfSum": f"{tsdf_sum:.9g}",
                    "weight": f"{weight:.9g}",
                    "meanTsdf": f"{mean:.9g}",
                    "sign": -1 if mean < 0.0 else (1 if mean > 0.0 else 0),
                    "updates": grid.update_counts[direction].get(key, 0),
                    "firstSequence": grid.first_sequences[direction].get(key, -1),
                    "lastSequence": grid.last_sequences[direction].get(key, -1),
                }
            )
    write_rows(path, rows)
    return len(rows)


def load_capture_metadata(capture: Path, frame: str) -> dict[str, Any]:
    path = capture / f"{frame}_capture.json"
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def main() -> int:
    args = parse_args()
    session = find_session(args.root)
    capture = session / "quest_clone_capture"
    all_rows = load_manifest(capture)
    rows = select_rows(all_rows, args.pairs)
    if not rows:
        raise RuntimeError("No exported Quest binocular frames were found")
    args.out.mkdir(parents=True, exist_ok=True)

    grid = LedgerDirectionalGrid(
        args.voxel,
        args.sdf_trunc,
        args.soft_direction_threshold,
        True,
    )
    frame_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    reconstruction_deltas: list[float] = []
    head_pose_deltas: list[float] = []
    preview_points: list[np.ndarray] = []
    accepted_total = 0
    input_valid_total = 0
    normal_valid_total = 0
    integration_started = time.perf_counter()
    sequence = 0

    for pair_number, row in enumerate(rows):
        frame = row["frame"]
        metadata = load_capture_metadata(capture, frame)
        exported_head = np.asarray(metadata.get("headPoseAtExport", {}).get("position", [np.nan] * 3), dtype=np.float64)
        eyes: list[tuple[EyeHeader, np.ndarray]] = []
        for suffix in ("right", "left"):
            source_path = capture / f"{frame}_{suffix}.scq3bin"
            header, records = read_eye(source_path)
            input_rows.append(
                {
                    "manifestFrame": frame,
                    "eye": "Right" if header.eye == 1 else "Left",
                    "sourceFrame": header.frame,
                    "dispatchSeconds": f"{header.dispatch_seconds:.9f}",
                    "width": header.width,
                    "height": header.height,
                    "bytes": source_path.stat().st_size,
                    "sha256": sha256_file(source_path),
                    "relativePath": source_path.relative_to(session).as_posix(),
                }
            )
            eyes.append((header, records))
        eyes.sort(key=lambda item: item[0].dispatch_seconds)

        for header, records in eyes:
            eye_started = time.perf_counter()
            linear_depth = records["meta"][:, 2].astype(np.float64).reshape(header.height, header.width)
            if args.input_source == "linear-depth":
                raw_points, valid, confidence = reconstruct_linear_depth_points(
                    header, records, args.eye_baseline, args.min_depth, args.max_depth
                )
                points, normals, normal_valid = preprocess_points_and_normals(
                    raw_points,
                    valid,
                    linear_depth,
                    args.smoothing_depth_delta,
                )
                recorded = records["point"].astype(np.float64).reshape(header.height, header.width, 3)
                comparable = valid & np.all(np.isfinite(recorded), axis=2)
                if np.any(comparable):
                    delta = np.linalg.norm(points[comparable] - recorded[comparable], axis=1)
                    reconstruction_deltas.extend(delta[np.isfinite(delta)].tolist())
            else:
                points, valid, confidence = reconstruct_recorded_world_points(
                    header, records, args.min_depth, args.max_depth
                )
                _, normals, normal_valid = preprocess_points_and_normals(
                    points, valid, linear_depth, 0.0
                )

            estimated_head = estimated_head_center(header, args.eye_baseline)
            if np.all(np.isfinite(exported_head)):
                head_pose_deltas.append(float(np.linalg.norm(estimated_head - exported_head)))

            sample_mask = valid & normal_valid & (confidence >= args.min_confidence)
            stride = max(1, args.pixel_stride)
            stride_mask = np.zeros_like(sample_mask)
            stride_mask[::stride, ::stride] = True
            sample_mask &= stride_mask
            indices = np.flatnonzero(sample_mask.reshape(-1))
            flat_points = points.reshape(-1, 3)
            flat_normals = normals.reshape(-1, 3)
            grid.current_sequence = sequence
            writes_before = grid.voxel_updates
            for index in indices:
                grid.integrate_normal_raycast(
                    header.camera_position,
                    flat_points[index],
                    flat_normals[index],
                )
            writes_this_eye = grid.voxel_updates - writes_before
            accepted = int(len(indices))
            accepted_total += accepted
            input_valid_total += int(np.count_nonzero(valid))
            normal_valid_total += int(np.count_nonzero(valid & normal_valid))
            if accepted > 0 and sum(len(chunk) for chunk in preview_points) < args.max_preview_points:
                remaining = max(0, args.max_preview_points - sum(len(chunk) for chunk in preview_points))
                preview_points.append(flat_points[indices[:remaining]].copy())

            frame_rows.append(
                {
                    "sequence": sequence,
                    "pairNumber": pair_number,
                    "manifestFrame": frame,
                    "eye": "Right" if header.eye == 1 else "Left",
                    "sourceFrame": header.frame,
                    "dispatchSeconds": f"{header.dispatch_seconds:.9f}",
                    "width": header.width,
                    "height": header.height,
                    "inputValidPixels": int(np.count_nonzero(valid)),
                    "normalValidPixels": int(np.count_nonzero(valid & normal_valid)),
                    "integratedSamples": accepted,
                    "directionWrites": writes_this_eye,
                    "cumulativeDirectionalVoxels": sum(len(layer) for layer in grid.values),
                    "cumulativeCandidateCells": len(grid.candidates),
                    "reconstructionDeltaP50Meters": (
                        f"{float(np.percentile(delta, 50)):.9g}"
                        if args.input_source == "linear-depth" and np.any(comparable)
                        else ""
                    ),
                    "reconstructionDeltaP95Meters": (
                        f"{float(np.percentile(delta, 95)):.9g}"
                        if args.input_source == "linear-depth" and np.any(comparable)
                        else ""
                    ),
                    "elapsedMs": f"{(time.perf_counter() - eye_started) * 1000.0:.3f}",
                }
            )
            sequence += 1
        print(
            f"[quest-frozen {pair_number + 1}/{len(rows)}] frame={frame} "
            f"samples={accepted_total} voxels={sum(len(layer) for layer in grid.values)} "
            f"cells={len(grid.candidates)}",
            flush=True,
        )

    integration_ms = (time.perf_counter() - integration_started) * 1000.0
    write_rows(args.out / "frame_replay_stats.csv", frame_rows)
    frozen_input_manifest = args.out / "frozen_input_manifest.csv"
    write_rows(frozen_input_manifest, input_rows)
    replay_set_digest = hashlib.sha256()
    for row in input_rows:
        replay_set_digest.update(
            f"{row['manifestFrame']}|{row['eye']}|{row['sha256']}\n".encode("utf-8")
        )
    ledger_rows = write_voxel_ledger(args.out / "directional_voxel_ledger.csv", grid)

    if preview_points:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.concatenate(preview_points, axis=0))
        o3d.io.write_point_cloud(str(args.out / "replayed_input_points.ply"), cloud, write_ascii=False)

    extraction_started = time.perf_counter()
    composed = extract_composed(
        grid,
        args.minimum_weight,
        args.valid_gradient_dot,
        args.parallel_dot,
        args.edge_merge_voxel_ratio,
    )
    features = extract_tsdf_hermite_feature_points(
        grid,
        args.minimum_weight,
        args.valid_gradient_dot,
        args.feature_angle_degrees,
        args.feature_min_family_support_ratio,
        args.feature_rank_ratio,
    )
    ledger_dmc = extract_tsdf_hermite_ledger_dual_mesh(
        grid,
        features,
        args.minimum_weight,
        args.valid_gradient_dot,
        args.parallel_dot,
        args.edge_merge_voxel_ratio,
        args.feature_rank_ratio,
    )
    extraction_ms = (time.perf_counter() - extraction_started) * 1000.0
    write_mesh(args.out / "soft_composed.ply", composed)
    write_mesh(args.out / "directional_dmc_ledger_shadow.ply", ledger_dmc)

    saturated = sum(
        1
        for layer in grid.values
        for _, weight in layer.values()
        if weight >= grid.maximum_weight - 1e-6
    )
    report = {
        "schema": "scancover.quest_frozen_depth_replay.v1",
        "sourceSession": str(session),
        "sourceManifestPairs": len(all_rows),
        "selectedPairs": len(rows),
        "selectedEyeFrames": len(frame_rows),
        "selectedFrames": [row["frame"] for row in rows],
        "frozenInputManifest": str(frozen_input_manifest),
        "frozenInputManifestSha256": sha256_file(frozen_input_manifest),
        "replaySetSha256": replay_set_digest.hexdigest(),
        "inputSource": args.input_source,
        "inputSemantics": (
            "SCQ3BIN2 meta.b radial linear depth restored along the eye ray seeded by the recorded filtered world point"
            if args.input_source == "linear-depth"
            else "SCQ3BIN2 filtered worldPositions captured after ScanCoverDepthPreprocessor"
        ),
        "historicalCaptureLimitations": {
            "originalGpuWorldNormalsStored": False,
            "rawHardwareDepth01Stored": False,
            "linearDepthMetersStored": True,
            "exactHistoricalPreprocessorReplayPossible": False,
            "currentAlgorithmDepthBoundaryReplayPossible": True,
            "reason": (
                "SCQ3BIN2/v2 omitted worldNormals and Camera.main dispatch pose. "
                "It also omitted raw depth01, so the ray direction is seeded by the recorded filtered world point; "
                "normals are recomputed and Camera.main is estimated from the stereo-eye pose and baseline."
            ),
        },
        "parameters": {
            "pixelStride": args.pixel_stride,
            "voxelMeters": args.voxel,
            "truncationMeters": args.sdf_trunc,
            "minimumWeight": args.minimum_weight,
            "softDirectionThreshold": args.soft_direction_threshold,
            "minDepthMeters": args.min_depth,
            "maxDepthMeters": args.max_depth,
            "minConfidence": args.min_confidence,
            "smoothingDepthDeltaMeters": args.smoothing_depth_delta,
            "eyeBaselineMeters": args.eye_baseline,
        },
        "inputValidPixels": input_valid_total,
        "normalValidPixels": normal_valid_total,
        "normalValidRatio": normal_valid_total / max(1, input_valid_total),
        "integratedSamples": accepted_total,
        "directionalVoxels": ledger_rows,
        "candidateCells": len(grid.candidates),
        "directionWrites": grid.direction_writes.tolist(),
        "voxelUpdateWrites": grid.voxel_updates,
        "saturatedDirectionalVoxels": saturated,
        "saturatedDirectionalVoxelRatio": saturated / max(1, ledger_rows),
        "integrationMs": integration_ms,
        "extractionMs": extraction_ms,
        "reconstructedToRecordedFilteredPositionDeltaMeters": quantiles(reconstruction_deltas),
        "estimatedHeadToExportHeadDeltaMeters": quantiles(head_pose_deltas),
        "topology": {
            "softComposed": topology_metrics(composed),
            "directionalDmcLedgerShadow": topology_metrics(ledger_dmc),
        },
        "crossImplementationGate": {
            "offlineFrozenReplayBuilt": True,
            "inputFramesComplete": len(frame_rows) == len(rows) * 2,
            "meshGenerated": len(ledger_dmc.triangles) > 0,
            "nonManifoldFree": topology_metrics(ledger_dmc)["nonManifoldEdges"] == 0,
            "readyForUnityReplayContract": True,
            "bitExactHistoricalReplayClaimed": False,
        },
        "interpretation": (
            "This replay can expose algorithmic corner/bevel/topology behaviour under real Quest observations. "
            "It is not a registered truth test and cannot reproduce omitted historical GPU normals bit-for-bit."
        ),
    }
    with (args.out / "quest_frozen_replay_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    lines = [
        "# Quest Frozen Depth Offline Replay",
        "",
        f"- source: `{session}`",
        f"- selected pairs: **{len(rows)}/{len(all_rows)}**",
        f"- input: **{args.input_source}**",
        f"- integrated samples: **{accepted_total:,}**",
        f"- directional voxels: **{ledger_rows:,}**",
        f"- candidate cells: **{len(grid.candidates):,}**",
        f"- original GPU normals present: **no**",
        f"- bit-exact historical replay claimed: **no**",
        "",
        "## Topology",
        "",
        "| Variant | Vertices | Triangles | Boundary / 1k tri | Non-manifold |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, build in (("soft_composed", composed), ("directional_dmc_ledger_shadow", ledger_dmc)):
        metrics = topology_metrics(build)
        lines.append(
            f"| {name} | {metrics['vertices']} | {metrics['triangles']} | "
            f"{metrics['boundaryEdgesPerKTriangles']:.2f} | {metrics['nonManifoldEdges']} |"
        )
    lines.extend(
        [
            "",
            "This is a real-input deterministic algorithm replay, not a geometry-truth evaluation. "
            "Unity comparison must consume the same frames and emit the same voxel ledger schema.",
        ]
    )
    (args.out / "quest_frozen_replay_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["crossImplementationGate"], ensure_ascii=False))
    print(f"Wrote outputs: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
