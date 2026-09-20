#!/usr/bin/env python3
"""Separate pose-like error from local depth deformation in a sealed replay.

Inputs are the held-out candidate observations produced by
ScanCoverRawDepthTrustAudit.py.  Even independent view clusters define each
candidate's platform-depth surface mode; only odd clusters are tested here.

Two read-only counterfactuals are evaluated:

1. Fit one robust six-degree-of-freedom rigid camera correction per frame.
2. Re-associate each frame with interpolated eye poses at nearby time offsets.

The candidate geometry remains an address and normal direction, not ground
truth.  A correction can explain pose-like coherence; it cannot prove that the
Quest pose provider was the original cause.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ScanCoverDepthPairDeviceAudit import recorded_eye
from ScanCoverRawDepthTrustAudit import candidate_number, distribution, read_csv, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose/timing counterfactual for raw depth.")
    parser.add_argument("session", type=Path)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-frame-observations", type=int, default=50)
    parser.add_argument("--maximum-near-surface-mm", type=float, default=100.0)
    parser.add_argument("--maximum-raw-post-mm", type=float, default=40.0)
    parser.add_argument("--time-offset-ms", default="-80,-60,-40,-20,0,20,40,60,80")
    return parser.parse_args()


def load_metadata(session: Path) -> list[dict]:
    root = session / "depth_pairs"
    rows = read_csv(root / "manifest.csv")
    records: list[dict] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        path = root / "frames" / row["metadataFile"]
        record = json.loads(path.read_text(encoding="utf-8-sig"))
        records.append(record)
    records.sort(key=lambda item: int(item["pairIndex"]))
    return records


def eye_pose(record: dict) -> tuple[np.ndarray, np.ndarray]:
    eye = recorded_eye(record)
    matrix = np.asarray(record["viewInverse"][eye], dtype=np.float64).reshape(4, 4)
    position = matrix[:3, 3].copy()
    proper = matrix[:3, :3].copy()
    proper[:, 2] *= -1.0
    if np.linalg.det(proper) < 0.0:
        raise RuntimeError(f"Cannot recover proper eye rotation at pair {record['pairIndex']}")
    return position, proper


def motion_name(angular: float, linear: float) -> str:
    if angular <= 2.0 and linear <= 0.01:
        return "still"
    if angular <= 10.0 and linear <= 0.05:
        return "slow"
    if angular <= 30.0 and linear <= 0.20:
        return "medium"
    return "fast"


def robust_rigid_fit(a: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    weights = np.ones(len(residual), dtype=np.float64)
    solution = np.zeros(6, dtype=np.float64)
    rank = 0
    for _ in range(12):
        root = np.sqrt(weights)
        weighted_a = a * root[:, None]
        weighted_b = -residual * root
        solution, _, rank, _ = np.linalg.lstsq(weighted_a, weighted_b, rcond=None)
        remaining = residual + a @ solution
        center = float(np.median(remaining))
        mad = float(np.median(np.abs(remaining - center)))
        scale = max(1.4826 * mad, 0.002)
        normalized = np.abs(remaining - center) / (1.5 * scale)
        updated = np.ones_like(normalized)
        outside = normalized > 1.0
        updated[outside] = 1.0 / normalized[outside]
        if np.max(np.abs(updated - weights)) < 1e-3:
            weights = updated
            break
        weights = updated
    return solution, residual + a @ solution, int(rank)


def metrics(prefix: str, values_metres: np.ndarray) -> dict[str, object]:
    millimetres = values_metres * 1000.0
    absolute = np.abs(millimetres)
    result: dict[str, object] = {}
    for key, value in distribution(absolute).items():
        result[f"{prefix}AbsMm_{key}"] = value
    return result


def interpolate_setup(records: list[dict]) -> tuple[np.ndarray, np.ndarray, Rotation, Slerp]:
    times = np.asarray([float(record["callbackUnscaledTime"]) for record in records])
    positions: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    for record in records:
        position, rotation = eye_pose(record)
        positions.append(position)
        matrices.append(rotation)
    rotations = Rotation.from_matrix(np.asarray(matrices))
    return times, np.asarray(positions), rotations, Slerp(times, rotations)


def interpolate_pose(
    query: np.ndarray,
    times: np.ndarray,
    positions: np.ndarray,
    slerp: Slerp,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.clip(query, times[0], times[-1])
    x = np.interp(query, times, positions[:, 0])
    y = np.interp(query, times, positions[:, 1])
    z = np.interp(query, times, positions[:, 2])
    return np.column_stack((x, y, z)), slerp(query).as_matrix()


def summary_by_motion(frame_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for name in ("still", "slow", "medium", "fast", "all"):
        rows = frame_rows if name == "all" else [row for row in frame_rows if row["motion"] == name]
        if not rows:
            continue
        before = np.asarray([float(row["beforeAbsP50Mm"]) for row in rows])
        after = np.asarray([float(row["afterRigidAbsP50Mm"]) for row in rows])
        scalar = np.asarray([float(row["afterScalarAbsP50Mm"]) for row in rows])
        before_p95 = np.asarray([float(row["beforeAbsP95Mm"]) for row in rows])
        after_p95 = np.asarray([float(row["afterRigidAbsP95Mm"]) for row in rows])
        scalar_p95 = np.asarray([float(row["afterScalarAbsP95Mm"]) for row in rows])
        output.append(
            {
                "motion": name,
                "frames": len(rows),
                "observations": sum(int(row["observations"]) for row in rows),
                "medianFrameBeforeAbsP50Mm": float(np.median(before)),
                "medianFrameAfterRigidAbsP50Mm": float(np.median(after)),
                "medianFrameAfterScalarAbsP50Mm": float(np.median(scalar)),
                "medianFrameBeforeAbsP95Mm": float(np.median(before_p95)),
                "medianFrameAfterRigidAbsP95Mm": float(np.median(after_p95)),
                "medianFrameAfterScalarAbsP95Mm": float(np.median(scalar_p95)),
                "medianFrameScalarP50ImprovementPct": float(np.median((before - scalar) / np.maximum(before, 1e-6)) * 100.0),
                "medianFrameScalarP95ImprovementPct": float(np.median((before_p95 - scalar_p95) / np.maximum(before_p95, 1e-6)) * 100.0),
                "medianFrameP50ImprovementPct": float(np.median((before - after) / np.maximum(before, 1e-6)) * 100.0),
                "medianFrameP95ImprovementPct": float(np.median((before_p95 - after_p95) / np.maximum(before_p95, 1e-6)) * 100.0),
                "medianTranslationMm": float(np.median([float(row["translationMm"]) for row in rows])),
                "medianRotationDeg": float(np.median([float(row["rotationDeg"]) for row in rows])),
            }
        )
    return output


def main() -> int:
    args = parse_args()
    session = args.session.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    candidates = read_csv(args.candidates.resolve())
    centers = np.asarray(
        [[candidate_number(row, "centerX", "center_x_m"), candidate_number(row, "centerY", "center_y_m"), candidate_number(row, "centerZ", "center_z_m")] for row in candidates],
        dtype=np.float64,
    )
    normals = np.asarray(
        [[candidate_number(row, "normalX", "normal_x"), candidate_number(row, "normalY", "normal_y"), candidate_number(row, "normalZ", "normal_z")] for row in candidates],
        dtype=np.float64,
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    records = load_metadata(session)
    by_pair = {int(record["pairIndex"]): record for record in records}
    times, positions, rotations, slerp = interpolate_setup(records)

    payload = np.load(args.observations.resolve())
    candidate_ids = payload["candidate"].astype(np.int32)
    pair_ids = payload["pair"].astype(np.int32)
    view_ids = payload["view"].astype(np.int32)
    raw_normal_mm = payload["rawNormalMm"].astype(np.float64)
    raw_post_mm = payload["rawPostNormalMm"].astype(np.float64)
    incidence = payload["incidence"].astype(np.float64)
    ranges = payload["range"].astype(np.float64)
    angular = payload["angular"].astype(np.float64)
    linear = payload["linear"].astype(np.float64)
    anchors = payload["anchorRawNormalMm"].astype(np.float64)
    baseline = anchors[candidate_ids]
    base_error_mm = raw_normal_mm - baseline
    eligible = (
        (view_ids % 2 == 1)
        & np.isfinite(baseline)
        & (incidence >= 0.30)
        & (np.abs(base_error_mm) <= args.maximum_near_surface_mm)
        & (np.abs(raw_post_mm) <= args.maximum_raw_post_mm)
    )

    eligible_indices = np.flatnonzero(eligible)
    order = eligible_indices[np.argsort(pair_ids[eligible_indices], kind="stable")]
    ordered_pairs = pair_ids[order]
    unique_pairs, starts = np.unique(ordered_pairs, return_index=True)
    ends = np.r_[starts[1:], len(order)]
    frame_rows: list[dict[str, object]] = []
    timing_accumulator: dict[tuple[str, float], list[np.ndarray]] = {}
    offsets = [float(value) for value in args.time_offset_ms.split(",") if value.strip()]

    print(
        f"Pose audit: eligible observations={len(eligible_indices)}, frames={len(unique_pairs)}, offsets={offsets}",
        flush=True,
    )
    for ordinal, (pair, start, end) in enumerate(zip(unique_pairs, starts, ends), 1):
        indices = order[start:end]
        if len(indices) < args.minimum_frame_observations:
            continue
        record = by_pair.get(int(pair))
        if record is None:
            continue
        ids = candidate_ids[indices]
        eye_position, eye_rotation = eye_pose(record)
        candidate_centers = centers[ids]
        candidate_normals = normals[ids]
        to_candidate = candidate_centers - eye_position[None, :]
        candidate_ranges = np.linalg.norm(to_candidate, axis=1)
        rays = to_candidate / np.maximum(candidate_ranges[:, None], 1e-12)
        signed_incidence = np.einsum("ij,ij->i", rays, candidate_normals)
        sign = np.where(signed_incidence >= 0.0, 1.0, -1.0)
        raw_ranges = candidate_ranges + raw_normal_mm[indices] / np.maximum(incidence[indices] * 1000.0, 1e-9)
        lever = rays * raw_ranges[:, None]
        points = eye_position[None, :] + lever
        reference = candidate_centers + candidate_normals * (baseline[indices] * sign / 1000.0)[:, None]
        residual = np.einsum("ij,ij->i", points - reference, candidate_normals)
        design = np.column_stack((candidate_normals, np.cross(lever, candidate_normals)))
        spatial_train = ids % 2 == 0
        spatial_test = ~spatial_train
        if np.count_nonzero(spatial_train) < 25 or np.count_nonzero(spatial_test) < 25:
            continue
        solution, _, rank = robust_rigid_fit(
            design[spatial_train], residual[spatial_train]
        )
        if rank < 6:
            continue
        test_residual = residual[spatial_test]
        after = test_residual + design[spatial_test] @ solution
        scalar_shift = -float(np.median(residual[spatial_train]))
        after_scalar = test_residual + scalar_shift
        before_abs = np.abs(test_residual) * 1000.0
        after_abs = np.abs(after) * 1000.0
        after_scalar_abs = np.abs(after_scalar) * 1000.0
        motion = motion_name(
            float(record.get("angularDegPerSec", 0.0)),
            float(record.get("linearMps", 0.0)),
        )
        frame_rows.append(
            {
                "pairIndex": int(pair),
                "platformFrame": int(record["platformFrame"]),
                "motion": motion,
                "angularDegPerSec": float(record.get("angularDegPerSec", 0.0)),
                "linearMps": float(record.get("linearMps", 0.0)),
                "trainingObservations": int(np.count_nonzero(spatial_train)),
                "observations": int(np.count_nonzero(spatial_test)),
                "rank": rank,
                "beforeAbsP50Mm": float(np.percentile(before_abs, 50)),
                "beforeAbsP90Mm": float(np.percentile(before_abs, 90)),
                "beforeAbsP95Mm": float(np.percentile(before_abs, 95)),
                "afterRigidAbsP50Mm": float(np.percentile(after_abs, 50)),
                "afterRigidAbsP90Mm": float(np.percentile(after_abs, 90)),
                "afterRigidAbsP95Mm": float(np.percentile(after_abs, 95)),
                "afterScalarAbsP50Mm": float(np.percentile(after_scalar_abs, 50)),
                "afterScalarAbsP90Mm": float(np.percentile(after_scalar_abs, 90)),
                "afterScalarAbsP95Mm": float(np.percentile(after_scalar_abs, 95)),
                "scalarNormalShiftMm": scalar_shift * 1000.0,
                "translationMm": float(np.linalg.norm(solution[:3]) * 1000.0),
                "rotationDeg": float(np.linalg.norm(solution[3:]) * 180.0 / math.pi),
                "translationXmm": float(solution[0] * 1000.0),
                "translationYmm": float(solution[1] * 1000.0),
                "translationZmm": float(solution[2] * 1000.0),
                "rotationXdeg": float(solution[3] * 180.0 / math.pi),
                "rotationYdeg": float(solution[4] * 180.0 / math.pi),
                "rotationZdeg": float(solution[5] * 180.0 / math.pi),
            }
        )

        camera_local = (eye_rotation.T @ lever.T).T
        query_base = float(record["callbackUnscaledTime"])
        for offset in offsets:
            shifted_position, shifted_rotation = interpolate_pose(
                np.asarray([query_base + offset / 1000.0]), times, positions, slerp
            )
            shifted_points = shifted_position[0][None, :] + (shifted_rotation[0] @ camera_local.T).T
            shifted_residual = np.einsum(
                "ij,ij->i", shifted_points - reference, candidate_normals
            )
            timing_accumulator.setdefault((motion, offset), []).append(shifted_residual)
            timing_accumulator.setdefault(("all", offset), []).append(shifted_residual)
        if ordinal % 100 == 0 or ordinal == len(unique_pairs):
            print(f"  candidate frames {ordinal}/{len(unique_pairs)}, fitted={len(frame_rows)}", flush=True)

    frame_fields = list(frame_rows[0].keys()) if frame_rows else ["pairIndex"]
    write_csv(out / "per_frame_rigid_correction.csv", frame_rows, frame_fields)
    motion_summary = summary_by_motion(frame_rows)
    write_csv(out / "rigid_summary_by_motion.csv", motion_summary, list(motion_summary[0].keys()) if motion_summary else ["motion"])

    timing_rows: list[dict[str, object]] = []
    for motion in ("still", "slow", "medium", "fast", "all"):
        for offset in offsets:
            parts = timing_accumulator.get((motion, offset), [])
            values = np.concatenate(parts) if parts else np.empty(0)
            absolute = np.abs(values) * 1000.0
            stats = distribution(absolute)
            timing_rows.append(
                {
                    "motion": motion,
                    "offsetMs": offset,
                    "observations": int(len(values)),
                    "absP50Mm": stats["p50"],
                    "absP75Mm": stats["p75"],
                    "absP90Mm": stats["p90"],
                    "absP95Mm": stats["p95"],
                    "within20mmPct": float(np.mean(absolute <= 20.0) * 100.0) if len(absolute) else None,
                    "within50mmPct": float(np.mean(absolute <= 50.0) * 100.0) if len(absolute) else None,
                }
            )
    write_csv(out / "timing_offset_sweep.csv", timing_rows, list(timing_rows[0].keys()))

    best_timing: dict[str, dict[str, object]] = {}
    for motion in ("still", "slow", "medium", "fast", "all"):
        eligible_rows = [row for row in timing_rows if row["motion"] == motion and row["absP50Mm"] is not None]
        if not eligible_rows:
            continue
        best = min(eligible_rows, key=lambda row: (float(row["absP50Mm"]), float(row["absP90Mm"])))
        zero = min(eligible_rows, key=lambda row: abs(float(row["offsetMs"])))
        best_timing[motion] = {
            "bestOffsetMs": best["offsetMs"],
            "bestAbsP50Mm": best["absP50Mm"],
            "bestAbsP90Mm": best["absP90Mm"],
            "zeroAbsP50Mm": zero["absP50Mm"],
            "zeroAbsP90Mm": zero["absP90Mm"],
            "p50ImprovementPct": (float(zero["absP50Mm"]) - float(best["absP50Mm"])) / max(float(zero["absP50Mm"]), 1e-9) * 100.0,
            "p90ImprovementPct": (float(zero["absP90Mm"]) - float(best["absP90Mm"])) / max(float(zero["absP90Mm"]), 1e-9) * 100.0,
        }

    summary = {
        "schema": "scancover.raw_depth_pose_audit.v1",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "authority": "read_only_offline_diagnostic",
        "productionConsumers": 0,
        "session": str(session),
        "inputObservationCount": int(len(candidate_ids)),
        "eligibleHeldoutNearSurfaceObservations": int(len(eligible_indices)),
        "fittedFrames": len(frame_rows),
        "rigidSummaryByMotion": motion_summary,
        "timingBestByMotion": best_timing,
        "interpretationRules": {
            "rigid": "large reduction means one coherent frame transform can explain part of the disagreement; remaining residual is not rigid-pose explainable",
            "timing": "a stable non-zero optimum that improves moving frames more than still frames implicates time association",
            "boundary": "neither test proves sensor truth because the reference mode is built from the same platform depth source",
        },
        "artifacts": {
            "perFrameRigid": "per_frame_rigid_correction.csv",
            "rigidByMotion": "rigid_summary_by_motion.csv",
            "timingSweep": "timing_offset_sweep.csv",
        },
    }
    (out / "raw_depth_pose_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
