#!/usr/bin/env python3
"""Build a read-only five-field shadow ledger from one ScanCover replay session.

The purpose of this tool is attribution, not another production gate.  It joins
the complete depth-pair surface observations with the probe candidate ledger so
that a visible paper hole can be separated into two very different cases:

* the surface was repeatedly observed but never became a probe candidate; or
* a candidate existed and was later held/rejected downstream.

Five fields are kept independently per 5 cm world cell:

1. observation coverage;
2. raw/post-QRS agreement;
3. timing and headset-motion context;
4. existence/free-space verdict evidence; and
5. geometry/fingerprint maturity.

All thresholds in this file are audit labels only.  No Unity asset, TSDF,
candidate, paper cell, or production decision is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


CELL_METRES = 0.05
RAW_POST_AGREEMENT_METRES = 0.05
CANDIDATE_SURFACE_BAND_CELLS = 3
INDEPENDENT_BASELINE_METRES = 0.08
INDEPENDENT_ANGLE_DEG = 3.0
INDEPENDENT_FRAME_GAP = 2


def number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def recorded_eye(meta: dict) -> int:
    eye = int(meta.get("recordedEyeIndex", 1))
    poses = meta.get("trackingPoses", [])
    return max(0, min(eye, len(poses) - 1)) if poses else eye


def matrix(values: list[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.size != 16:
        raise ValueError(f"matrix requires 16 values, got {result.size}")
    return result.reshape(4, 4)


def angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def backproject(
    meta: dict,
    depth_flat: np.ndarray,
    pixel_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return world points, sampled indices, view radius, camera and valid mask."""
    width = int(meta["width"])
    height = int(meta["height"])
    eye = recorded_eye(meta)
    image = depth_flat.reshape(height, width)
    ys = np.arange(0, height, max(1, pixel_stride), dtype=np.int32)
    xs = np.arange(0, width, max(1, pixel_stride), dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    sampled = image[grid_y, grid_x]
    valid = np.isfinite(sampled) & (sampled > 0.0) & (sampled < 1.0)
    flat_indices = (grid_y * width + grid_x).reshape(-1)
    if not np.any(valid):
        camera = matrix(meta["viewInverse"][eye])[:3, 3]
        return (
            np.empty((0, 3), dtype=np.float64),
            flat_indices,
            np.empty(0, dtype=np.float64),
            camera,
            valid.reshape(-1),
        )

    u = grid_x[valid].astype(np.float64) / float(width)
    v = grid_y[valid].astype(np.float64) / float(height)
    ndc = sampled[valid].astype(np.float64)
    hcs = np.stack(
        (u * 2.0 - 1.0, v * 2.0 - 1.0, ndc * 2.0 - 1.0, np.ones_like(u))
    )
    projection_inverse = matrix(meta["projectionInverse"][eye])
    view_inverse = matrix(meta["viewInverse"][eye])
    world_h = view_inverse @ (projection_inverse @ hcs)
    world = (world_h[:3] / world_h[3]).T
    camera = view_inverse[:3, 3]
    radius = np.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2) / math.sqrt(0.5)
    distance = np.linalg.norm(world - camera, axis=1)
    keep = np.isfinite(world).all(axis=1) & np.isfinite(distance) & (distance >= 0.15) & (distance <= 8.0)
    valid_positions = np.flatnonzero(valid.reshape(-1))
    final_valid = np.zeros(valid.size, dtype=bool)
    final_valid[valid_positions[keep]] = True
    return world[keep], flat_indices, radius[keep], camera, final_valid


@dataclass
class CoverageCell:
    point_samples: int = 0
    coverage_frames: int = 0
    first_frame: int = 2**31 - 1
    last_frame: int = -1
    first_time: float = math.inf
    last_time: float = -math.inf
    raw_valid_samples: int = 0
    raw_agree_samples: int = 0
    raw_post_delta_sum_m: float = 0.0
    raw_post_delta_max_m: float = 0.0
    view_radius_sum: float = 0.0
    view_radius_max: float = 0.0
    callback_ms_sum: float = 0.0
    callback_ms_max: float = 0.0
    angular_sum: float = 0.0
    angular_max: float = 0.0
    linear_sum: float = 0.0
    linear_max: float = 0.0
    independent_views: int = 0
    last_independent_frame: int = -10**9
    reference_camera: Optional[np.ndarray] = field(default=None, repr=False)
    reference_direction: Optional[np.ndarray] = field(default=None, repr=False)
    max_baseline_m: float = 0.0
    max_spread_deg: float = 0.0

    def add_view(self, frame: int, camera: np.ndarray, direction: np.ndarray) -> None:
        if self.reference_camera is None or self.reference_direction is None:
            self.reference_camera = camera.copy()
            self.reference_direction = direction.copy()
            self.independent_views = 1
            self.last_independent_frame = frame
            return
        baseline = float(np.linalg.norm(camera - self.reference_camera))
        spread = angle_deg(direction, self.reference_direction)
        self.max_baseline_m = max(self.max_baseline_m, baseline)
        self.max_spread_deg = max(self.max_spread_deg, spread)
        if (
            frame - self.last_independent_frame >= INDEPENDENT_FRAME_GAP
            and (baseline >= INDEPENDENT_BASELINE_METRES or spread >= INDEPENDENT_ANGLE_DEG)
        ):
            self.independent_views += 1
            self.last_independent_frame = frame
            self.reference_camera = camera.copy()
            self.reference_direction = direction.copy()


@dataclass
class CandidateCell:
    axes: set[int] = field(default_factory=set)
    verdicts: Counter = field(default_factory=Counter)
    safe_support_frames: int = 0
    independent_support_views: int = 0
    challenge_votes: int = 0
    independent_challenge_views: int = 0
    recovery_support_views: int = 0
    max_baseline_mm: float = 0.0
    max_view_spread_deg: float = 0.0
    max_free_gap_mm: float = 0.0
    decision_strength: float = 0.0
    fingerprint_count: int = 0
    fingerprint_patch_count: int = 0


def candidate_cells(path: Path) -> dict[tuple[int, int, int], CandidateCell]:
    result: dict[tuple[int, int, int], CandidateCell] = {}
    for row in read_csv(path):
        key = (integer(row.get("cellX")), integer(row.get("cellY")), integer(row.get("cellZ")))
        item = result.setdefault(key, CandidateCell())
        item.axes.add(integer(row.get("axis")))
        item.verdicts[row.get("verdict", "unknown")] += 1
        item.safe_support_frames = max(item.safe_support_frames, integer(row.get("safeSupportFrames")))
        item.independent_support_views = max(
            item.independent_support_views, integer(row.get("independentSupportViews"))
        )
        item.challenge_votes = max(item.challenge_votes, integer(row.get("challengeVotes")))
        item.independent_challenge_views = max(
            item.independent_challenge_views, integer(row.get("independentChallengeViews"))
        )
        item.recovery_support_views = max(
            item.recovery_support_views, integer(row.get("recoverySupportViews"))
        )
        item.max_baseline_mm = max(item.max_baseline_mm, number(row.get("maxBaselineMm")))
        item.max_view_spread_deg = max(
            item.max_view_spread_deg, number(row.get("maxViewSpreadDeg"))
        )
        item.max_free_gap_mm = max(item.max_free_gap_mm, number(row.get("maxFreeGapMm")))
        item.decision_strength = max(item.decision_strength, number(row.get("decisionStrength")))
        item.fingerprint_count += int(integer(row.get("fingerprintId")) > 0)
        item.fingerprint_patch_count += integer(row.get("fingerprintPatchCount"))
    return result


def nearest_candidate(
    key: tuple[int, int, int],
    candidates: dict[tuple[int, int, int], CandidateCell],
) -> tuple[Optional[CandidateCell], int]:
    exact = candidates.get(key)
    if exact is not None:
        return exact, 0
    for radius in range(1, CANDIDATE_SURFACE_BAND_CELLS + 1):
        best: Optional[CandidateCell] = None
        best_distance = math.inf
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    item = candidates.get((key[0] + dx, key[1] + dy, key[2] + dz))
                    distance = dx * dx + dy * dy + dz * dz
                    if item is not None and distance < best_distance:
                        best = item
                        best_distance = distance
        if best is not None:
            return best, radius
    return None, -1


def aggregate_depth(
    session: Path,
    frame_stride: int,
    pixel_stride: int,
) -> tuple[dict[tuple[int, int, int], CoverageCell], dict[str, object]]:
    depth_root = session / "depth_pairs"
    frames = depth_root / "frames"
    rows = [
        row
        for row in read_csv(depth_root / "manifest.csv")
        if row.get("status", "").lower() in {"ok", "complete"}
    ]
    selected = rows[:: max(1, frame_stride)]
    cells: dict[tuple[int, int, int], CoverageCell] = {}
    errors: list[str] = []
    processed = 0
    for selection_index, row in enumerate(selected, start=1):
        try:
            meta = json.loads((frames / row["metadataFile"]).read_text(encoding="utf-8-sig"))
            count = int(meta["width"]) * int(meta["height"]) * int(meta["layers"])
            raw = np.fromfile(frames / row["rawFile"], dtype="<f4", count=count)
            post = np.fromfile(frames / row["processedFile"], dtype="<f4", count=count)
            if raw.size != count or post.size != count:
                raise IOError(f"short frame raw={raw.size} post={post.size} expected={count}")
            world, sampled_indices, view_radius, camera, final_valid = backproject(
                meta, post, pixel_stride
            )
            if world.size == 0:
                processed += 1
                continue
            valid_sample_indices = sampled_indices[final_valid]
            raw_sample = raw[valid_sample_indices]
            post_sample = post[valid_sample_indices]
            raw_valid = np.isfinite(raw_sample) & (raw_sample > 0.0) & (raw_sample < 1.0)
            raw_delta = np.full(post_sample.shape, np.nan, dtype=np.float64)
            if np.any(raw_valid):
                eye = recorded_eye(meta)
                projection = matrix(meta["projection"][eye])
                raw_z = raw_sample[raw_valid].astype(np.float64) * 2.0 - 1.0
                post_z = post_sample[raw_valid].astype(np.float64) * 2.0 - 1.0
                raw_m = np.abs(float(projection[2, 3]) / (raw_z + float(projection[2, 2])))
                post_m = np.abs(float(projection[2, 3]) / (post_z + float(projection[2, 2])))
                raw_delta[raw_valid] = np.abs(raw_m - post_m)

            quantized = np.floor(world / CELL_METRES).astype(np.int32)
            unique, inverse, counts = np.unique(quantized, axis=0, return_inverse=True, return_counts=True)
            order = np.argsort(inverse, kind="stable")
            offsets = np.r_[0, np.cumsum(counts)]
            frame = integer(row.get("pairIndex"))
            time_seconds = number(meta.get("preprocessUnscaledTime"), number(row.get("preprocessUnscaledTime")))
            callback_ms = number(meta.get("callbackToPreprocessMs"))
            angular = number(meta.get("angularDegPerSec"))
            linear = number(meta.get("linearMps"))
            for group_index, coordinates in enumerate(unique):
                local_indices = order[offsets[group_index] : offsets[group_index + 1]]
                key = (int(coordinates[0]), int(coordinates[1]), int(coordinates[2]))
                item = cells.setdefault(key, CoverageCell())
                n = int(counts[group_index])
                item.point_samples += n
                item.coverage_frames += 1
                item.first_frame = min(item.first_frame, frame)
                item.last_frame = max(item.last_frame, frame)
                item.first_time = min(item.first_time, time_seconds)
                item.last_time = max(item.last_time, time_seconds)
                local_raw_valid = raw_valid[local_indices]
                local_delta = raw_delta[local_indices]
                finite_delta = local_delta[np.isfinite(local_delta)]
                item.raw_valid_samples += int(np.count_nonzero(local_raw_valid))
                item.raw_agree_samples += int(
                    np.count_nonzero(finite_delta <= RAW_POST_AGREEMENT_METRES)
                )
                if finite_delta.size:
                    item.raw_post_delta_sum_m += float(np.sum(finite_delta))
                    item.raw_post_delta_max_m = max(
                        item.raw_post_delta_max_m, float(np.max(finite_delta))
                    )
                local_radius = view_radius[local_indices]
                item.view_radius_sum += float(np.sum(local_radius))
                item.view_radius_max = max(item.view_radius_max, float(np.max(local_radius)))
                item.callback_ms_sum += callback_ms
                item.callback_ms_max = max(item.callback_ms_max, callback_ms)
                item.angular_sum += angular
                item.angular_max = max(item.angular_max, angular)
                item.linear_sum += linear
                item.linear_max = max(item.linear_max, linear)
                centre = (coordinates.astype(np.float64) + 0.5) * CELL_METRES
                direction = camera - centre
                length = float(np.linalg.norm(direction))
                if length > 1e-8:
                    item.add_view(frame, camera, direction / length)
            processed += 1
            if selection_index % 250 == 0:
                print(
                    f"depth progress {selection_index}/{len(selected)} frames; cells={len(cells)}",
                    flush=True,
                )
        except Exception as exc:
            errors.append(f"pair={row.get('pairIndex')} error={exc}")
    return cells, {
        "manifestFrames": len(rows),
        "selectedFrames": len(selected),
        "processedFrames": processed,
        "frameStride": max(1, frame_stride),
        "pixelStride": max(1, pixel_stride),
        "errors": errors,
    }


CELL_FIELDS = [
    "cellX", "cellY", "cellZ", "centerX", "centerY", "centerZ",
    "pointSamples", "coverageFrames", "independentViews", "firstFrame", "lastFrame",
    "observationSpanSeconds", "rawValidRatio", "rawPostAgreeRatio", "rawPostDeltaMeanMm",
    "rawPostDeltaMaxMm", "viewRadiusMean", "viewRadiusMax", "callbackToPreprocessMeanMs",
    "callbackToPreprocessMaxMs", "angularMeanDegPerSec", "angularMaxDegPerSec",
    "linearMeanMps", "linearMaxMps", "maxBaselineMm", "maxViewSpreadDeg",
    "candidateDistanceCells", "candidateAxes", "acceptAxes", "holdAxes", "rejectAxes",
    "safeSupportFrames", "candidateIndependentViews", "challengeVotes",
    "challengeIndependentViews", "recoverySupportViews", "maxFreeGapMm",
    "decisionStrength", "fingerprintCount", "fingerprintPatchCount", "shadowClass",
]


def joined_row(
    key: tuple[int, int, int],
    coverage: CoverageCell,
    candidate: Optional[CandidateCell],
    candidate_distance: int,
) -> dict[str, object]:
    frames = max(coverage.coverage_frames, 1)
    raw = max(coverage.raw_valid_samples, 1)
    points = max(coverage.point_samples, 1)
    if candidate is None:
        if (
            coverage.coverage_frames >= 8
            and coverage.independent_views >= 2
            and coverage.raw_valid_samples / points >= 0.75
            and coverage.raw_agree_samples / raw >= 0.75
        ):
            shadow_class = "repeated_clean_observation_without_candidate"
        elif coverage.coverage_frames >= 3:
            shadow_class = "observation_without_candidate_unresolved"
        else:
            shadow_class = "sparse_observation_without_candidate"
    elif candidate_distance > 0:
        # A nearby candidate proves that this sample belongs to an already
        # represented surface band, but its axis/verdict must not be borrowed
        # as if it were the exact observed cell.
        shadow_class = "near_candidate_surface_band"
    elif candidate.verdicts.get("accept", 0) > 0:
        shadow_class = "candidate_accepted"
    elif candidate.verdicts.get("reject", 0) > 0:
        shadow_class = "candidate_rejected"
    else:
        shadow_class = "candidate_held"
    return {
        "cellX": key[0], "cellY": key[1], "cellZ": key[2],
        "centerX": (key[0] + 0.5) * CELL_METRES,
        "centerY": (key[1] + 0.5) * CELL_METRES,
        "centerZ": (key[2] + 0.5) * CELL_METRES,
        "pointSamples": coverage.point_samples,
        "coverageFrames": coverage.coverage_frames,
        "independentViews": coverage.independent_views,
        "firstFrame": coverage.first_frame,
        "lastFrame": coverage.last_frame,
        "observationSpanSeconds": max(0.0, coverage.last_time - coverage.first_time),
        "rawValidRatio": coverage.raw_valid_samples / points,
        "rawPostAgreeRatio": coverage.raw_agree_samples / raw,
        "rawPostDeltaMeanMm": coverage.raw_post_delta_sum_m / raw * 1000.0,
        "rawPostDeltaMaxMm": coverage.raw_post_delta_max_m * 1000.0,
        "viewRadiusMean": coverage.view_radius_sum / points,
        "viewRadiusMax": coverage.view_radius_max,
        "callbackToPreprocessMeanMs": coverage.callback_ms_sum / frames,
        "callbackToPreprocessMaxMs": coverage.callback_ms_max,
        "angularMeanDegPerSec": coverage.angular_sum / frames,
        "angularMaxDegPerSec": coverage.angular_max,
        "linearMeanMps": coverage.linear_sum / frames,
        "linearMaxMps": coverage.linear_max,
        "maxBaselineMm": coverage.max_baseline_m * 1000.0,
        "maxViewSpreadDeg": coverage.max_spread_deg,
        "candidateDistanceCells": candidate_distance,
        "candidateAxes": len(candidate.axes) if candidate else 0,
        "acceptAxes": candidate.verdicts.get("accept", 0) if candidate else 0,
        "holdAxes": candidate.verdicts.get("hold", 0) if candidate else 0,
        "rejectAxes": candidate.verdicts.get("reject", 0) if candidate else 0,
        "safeSupportFrames": candidate.safe_support_frames if candidate else 0,
        "candidateIndependentViews": candidate.independent_support_views if candidate else 0,
        "challengeVotes": candidate.challenge_votes if candidate else 0,
        "challengeIndependentViews": candidate.independent_challenge_views if candidate else 0,
        "recoverySupportViews": candidate.recovery_support_views if candidate else 0,
        "maxFreeGapMm": candidate.max_free_gap_mm if candidate else 0.0,
        "decisionStrength": candidate.decision_strength if candidate else 0.0,
        "fingerprintCount": candidate.fingerprint_count if candidate else 0,
        "fingerprintPatchCount": candidate.fingerprint_patch_count if candidate else 0,
        "shadowClass": shadow_class,
    }


def quantiles(values: Iterable[float]) -> dict[str, Optional[float]]:
    clean = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if clean.size == 0:
        return {"p50": None, "p90": None, "p95": None, "max": None}
    return {
        "p50": float(np.percentile(clean, 50)),
        "p90": float(np.percentile(clean, 90)),
        "p95": float(np.percentile(clean, 95)),
        "max": float(np.max(clean)),
    }


def gap_components(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (integer(row["cellX"]), integer(row["cellY"]), integer(row["cellZ"])): row
        for row in rows
    }
    remaining = set(by_key)
    result: list[dict[str, object]] = []
    component_id = 0
    neighbours = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    while remaining:
        component_id += 1
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            current = stack.pop()
            for delta in neighbours:
                candidate = (
                    current[0] + delta[0], current[1] + delta[1], current[2] + delta[2]
                )
                if candidate in remaining:
                    remaining.remove(candidate)
                    stack.append(candidate)
                    component.append(candidate)
        array = np.asarray(component, dtype=np.int32)
        component_rows = [by_key[key] for key in component]
        low = array.min(axis=0).astype(np.float64) * CELL_METRES
        high = (array.max(axis=0).astype(np.float64) + 1.0) * CELL_METRES
        result.append(
            {
                "componentId": component_id,
                "cells": len(component),
                "minX": low[0], "minY": low[1], "minZ": low[2],
                "maxX": high[0], "maxY": high[1], "maxZ": high[2],
                "coverageFramesP50": float(np.median([number(row["coverageFrames"]) for row in component_rows])),
                "independentViewsP50": float(np.median([number(row["independentViews"]) for row in component_rows])),
                "rawPostDeltaMeanMmP50": float(np.median([number(row["rawPostDeltaMeanMm"]) for row in component_rows])),
            }
        )
    result.sort(key=lambda row: integer(row["cells"]), reverse=True)
    for index, row in enumerate(result, start=1):
        row["componentId"] = index
    return result


def write_point_cloud(path: Path, rows: list[dict[str, object]]) -> None:
    colors = {
        "candidate_accepted": (50, 220, 80),
        "candidate_held": (255, 190, 40),
        "candidate_rejected": (230, 40, 40),
        "near_candidate_surface_band": (70, 160, 230),
        "repeated_clean_observation_without_candidate": (255, 0, 210),
        "observation_without_candidate_unresolved": (170, 80, 220),
        "sparse_observation_without_candidate": (100, 100, 100),
    }
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(rows)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for row in rows:
            red, green, blue = colors.get(str(row["shadowClass"]), (255, 255, 255))
            handle.write(
                f"{number(row['centerX']):.6f} {number(row['centerY']):.6f} "
                f"{number(row['centerZ']):.6f} {red} {green} {blue}\n"
            )


def hold_audit(
    candidate_axis_rows: list[dict[str, str]],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    held = [row for row in candidate_axis_rows if row.get("verdict") == "hold"]
    reasons: Counter = Counter()
    for item in held:
        safe_support_frames = integer(item.get("safeSupportFrames"))
        independent_support_views = integer(item.get("independentSupportViews"))
        if safe_support_frames <= 0:
            reasons["no_safe_support"] += 1
        elif safe_support_frames < 2:
            reasons["only_one_safe_support_frame"] += 1
        elif independent_support_views < 2:
            reasons["repeated_safe_support_but_no_second_independent_support"] += 1
        else:
            reasons["other_hold_state"] += 1

    exact_held = [row for row in rows if row["shadowClass"] == "candidate_held"]
    sensitivity: list[dict[str, object]] = []
    for frames, views, agreement, delta_mm in (
        (8, 2, 0.75, 50.0),
        (30, 3, 0.90, 20.0),
        (60, 4, 0.90, 10.0),
        (120, 4, 0.95, 10.0),
    ):
        count = sum(
            number(row["coverageFrames"]) >= frames
            and number(row["independentViews"]) >= views
            and number(row["rawValidRatio"]) >= agreement
            and number(row["rawPostAgreeRatio"]) >= agreement
            and number(row["rawPostDeltaMeanMm"]) <= delta_mm
            for row in exact_held
        )
        sensitivity.append(
            {
                "minimumCoverageFrames": frames,
                "minimumIndependentCoverageViews": views,
                "minimumRawAndAgreementRatio": agreement,
                "maximumMeanRawPostDeltaMm": delta_mm,
                "heldCellsPassingCoverageEvidence": count,
                "ratioOfObservedExactHeld": count / max(len(exact_held), 1),
            }
        )
    return {
        "candidateHeldCells": len(held),
        "observedExactHeldCells": len(exact_held),
        "candidateHoldReasons": dict(sorted(reasons.items())),
        "coverageEvidenceSensitivity": sensitivity,
        "interpretation": (
            "Independent coverage views are computed directly from depth-pair camera poses, while "
            "candidate independent views require the production safe-support correspondence path. "
            "A held cell with strong raw coverage but one candidate support view identifies a "
            "coverage-to-safe-witness conversion bottleneck; it does not authorize acceptance."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="Immutable local replay session")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--pixel-stride", type=int, default=4)
    args = parser.parse_args()

    session = args.session.resolve()
    output = args.output.resolve()
    required = [
        session / "depth_pairs" / "manifest.csv",
        session / "depth_pairs" / "frames",
        session / "probe_shadow" / "verdict_cells.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing five-field inputs: " + "; ".join(missing))

    coverage, depth_summary = aggregate_depth(
        session, max(1, args.frame_stride), max(1, args.pixel_stride)
    )
    candidate_axis_rows = read_csv(session / "probe_shadow" / "verdict_cells.csv")
    candidates = candidate_cells(session / "probe_shadow" / "verdict_cells.csv")
    class_counts: Counter = Counter()
    rows: list[dict[str, object]] = []
    for key in sorted(coverage):
        candidate, distance = nearest_candidate(key, candidates)
        row = joined_row(key, coverage[key], candidate, distance)
        rows.append(row)
        class_counts[str(row["shadowClass"])] += 1

    gap_rows = [
        row for row in rows
        if row["shadowClass"] in {
            "repeated_clean_observation_without_candidate",
            "observation_without_candidate_unresolved",
        }
    ]
    candidate_rows = [row for row in rows if integer(row["candidateDistanceCells"], -1) >= 0]
    components = gap_components(
        [row for row in gap_rows if row["shadowClass"] == "repeated_clean_observation_without_candidate"]
    )
    hold_summary = hold_audit(candidate_axis_rows, rows)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "five_field_cells.csv", rows, CELL_FIELDS)
    write_csv(output / "five_field_gap_cells.csv", gap_rows, CELL_FIELDS)
    write_csv(output / "five_field_candidate_cells.csv", candidate_rows, CELL_FIELDS)
    component_fields = [
        "componentId", "cells", "minX", "minY", "minZ", "maxX", "maxY", "maxZ",
        "coverageFramesP50", "independentViewsP50", "rawPostDeltaMeanMmP50",
    ]
    write_csv(output / "five_field_gap_components.csv", components, component_fields)
    write_point_cloud(output / "five_field_cells.ply", rows)

    summary = {
        "schema": "scancover.five_field_shadow.v1",
        "session": session.name,
        "authority": "offline_read_only_diagnostic",
        "cellMetres": CELL_METRES,
        "depth": depth_summary,
        "observedCells": len(rows),
        "candidateLedgerCells": len(candidates),
        "joinedCandidateCells": len(candidate_rows),
        "classes": dict(sorted(class_counts.items())),
        "strongGapComponents": len(components),
        "largestStrongGapComponents": [integer(row["cells"]) for row in components[:10]],
        "holdAudit": hold_summary,
        "coverageFrames": quantiles(float(row["coverageFrames"]) for row in rows),
        "independentViews": quantiles(float(row["independentViews"]) for row in rows),
        "rawPostAgreeRatio": quantiles(float(row["rawPostAgreeRatio"]) for row in rows),
        "rawPostDeltaMeanMm": quantiles(float(row["rawPostDeltaMeanMm"]) for row in rows),
        "auditLabels": {
            "strongObservationFrames": 8,
            "strongIndependentViews": 2,
            "rawValidRatio": 0.75,
            "rawPostAgreementRatio": 0.75,
            "rawPostAgreementMetres": RAW_POST_AGREEMENT_METRES,
            "candidateSurfaceBandCells": CANDIDATE_SURFACE_BAND_CELLS,
        },
        "interpretationBoundary": (
            "A repeated_clean_observation_without_candidate cell lies outside the configured "
            "candidate surface band and is evidence of an upstream "
            "candidate-formation/association gap, not proof that production should fill it. "
            "A candidate_held/rejected cell is downstream adjudication evidence. The labels do "
            "not alter fusion, verdicts, paper coverage, or graduation."
        ),
        "files": {
            "allCells": "five_field_cells.csv",
            "gapCells": "five_field_gap_cells.csv",
            "candidateCells": "five_field_candidate_cells.csv",
            "gapComponents": "five_field_gap_components.csv",
            "spatialPointCloud": "five_field_cells.ply",
        },
    }
    (output / "five_field_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# 五场影子回放",
        "",
        f"- 会话：`{session.name}`",
        f"- 深度帧：清单 {depth_summary['manifestFrames']}，抽取 {depth_summary['selectedFrames']}，成功 {depth_summary['processedFrames']}",
        f"- 观察到的 5 cm 空间单元：{len(rows)}",
        f"- 候选账本单元：{len(candidates)}",
        f"- 与候选邻域相交的观察单元：{len(candidate_rows)}",
        "",
        "## 责任分层",
        "",
    ]
    for label, count in sorted(class_counts.items()):
        report.append(f"- `{label}`：{count}")
    report.extend(
        [
            "",
            "## Hold 瓶颈",
            "",
            f"- 候选账本 Hold：{hold_summary['candidateHeldCells']}",
            f"- 在抽样深度中精确命中的 Hold：{hold_summary['observedExactHeldCells']}",
            "- 原因：" + "；".join(
                f"`{name}` {count}"
                for name, count in hold_summary["candidateHoldReasons"].items()
            ),
            "",
            "`repeated_clean_observation_without_candidate` 表示原料层反复、跨视角且前后处理一致地看见了表面，",
            "但 5 cm 邻域内没有进入候选账本；这是候选形成/空间关联之前的覆盖损失。",
            "`candidate_held` / `candidate_rejected` 则说明表面已经走到候选层，空缺责任在后续裁决。",
            "本工具只记账，不给生产链放行或补洞。",
        ]
    )
    (output / "five_field_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not depth_summary["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
