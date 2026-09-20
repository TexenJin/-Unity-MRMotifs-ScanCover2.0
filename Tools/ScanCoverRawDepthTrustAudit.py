#!/usr/bin/env python3
"""Audit Quest platform depth before TSDF without treating any fused surface as truth.

The final-buffer candidates are used only as stable world-space addresses and local
normal directions.  A surface position is estimated from low-motion, central,
high-incidence platform-depth observations in even-numbered independent view
clusters.  Odd-numbered clusters are held out for evaluation.

The audit also measures single-frame local planarity directly in camera space.
Neither the candidate surface nor TSDF values are used as geometric ground truth.
All outputs are read-only offline evidence and have no production consumers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from ScanCoverDepthPairDeviceAudit import annotate_motion, recorded_eye
from ScanCoverOfflineReferenceCourt import (
    candidate_number,
    cluster_independent_views,
    project_and_sample_ranges,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only self-consistency audit for platform pre-QRS depth."
    )
    parser.add_argument("session", type=Path, help="local replay session root")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--frame-limit", type=int, default=0)
    parser.add_argument("--patch-pixel-stride", type=int, default=8)
    parser.add_argument("--patch-grid-size", type=int, default=5)
    parser.add_argument("--outer-radius", type=float, default=0.90)
    parser.add_argument("--minimum-incidence", type=float, default=0.15)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    quantiles = np.percentile(values, (5, 25, 50, 75, 90, 95, 99))
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def format_number(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def signed_degrees(value: object) -> float:
    number = float(value)
    return (number + 180.0) % 360.0 - 180.0


def load_records(session: Path) -> tuple[list[dict], list[dict[str, str]]]:
    root = session / "depth_pairs"
    manifest = read_csv(root / "manifest.csv")
    records: list[dict] = []
    for row in manifest:
        if row.get("status") != "ok":
            continue
        metadata = root / "frames" / row["metadataFile"]
        if not metadata.is_file():
            raise RuntimeError(f"Missing metadata: {metadata}")
        record = json.loads(metadata.read_text(encoding="utf-8-sig"))
        record["_manifest"] = row
        records.append(record)
    records.sort(key=lambda item: int(item["pairIndex"]))
    annotate_motion(records)
    return records, manifest


def load_depth_pair(session: Path, record: dict) -> tuple[np.ndarray, np.ndarray]:
    row = record["_manifest"]
    frames = session / "depth_pairs" / "frames"
    count = int(record["width"]) * int(record["height"]) * int(record["layers"])
    raw_path = frames / row["rawFile"]
    post_path = frames / row["processedFile"]
    raw = np.fromfile(raw_path, dtype="<f4", count=count)
    post = np.fromfile(post_path, dtype="<f4", count=count)
    if len(raw) != count or len(post) != count:
        raise RuntimeError(
            f"Short depth payload at pair {record['pairIndex']}: "
            f"raw={len(raw)} post={len(post)} expected={count}"
        )
    return raw, post


def camera_grid(record: dict, depth: np.ndarray, stride: int) -> np.ndarray:
    width = int(record["width"])
    height = int(record["height"])
    image = depth.reshape(height, width)
    ys = np.arange(stride // 2, height, stride, dtype=np.int32)
    xs = np.arange(stride // 2, width, stride, dtype=np.int32)
    xx, yy = np.meshgrid(xs, ys)
    sampled = image[yy, xx].astype(np.float64)
    valid = np.isfinite(sampled) & (sampled > 0.0) & (sampled < 1.0)
    eye = recorded_eye(record)
    inverse = np.asarray(record["projectionInverse"][eye], dtype=np.float64).reshape(4, 4)
    hcs = np.stack(
        (
            xx.astype(np.float64) / width * 2.0 - 1.0,
            yy.astype(np.float64) / height * 2.0 - 1.0,
            sampled * 2.0 - 1.0,
            np.ones_like(sampled),
        )
    ).reshape(4, -1)
    eye_h = inverse @ hcs
    with np.errstate(divide="ignore", invalid="ignore"):
        points = (eye_h[:3] / eye_h[3]).T.reshape((*sampled.shape, 3))
    ranges = np.linalg.norm(points, axis=2)
    valid &= np.isfinite(points).all(axis=2) & (ranges >= 0.15) & (ranges <= 8.0)
    points[~valid] = np.nan
    return points


def fit_patch(points: np.ndarray) -> tuple[np.ndarray, float, float, float] | None:
    flat = points.reshape(-1, 3)
    flat = flat[np.isfinite(flat).all(axis=1)]
    if len(flat) < max(12, int(points.shape[0] * points.shape[1] * 0.55)):
        return None
    center = np.median(flat, axis=0)
    ranges = np.linalg.norm(flat, axis=1)
    if np.percentile(ranges, 95) - np.percentile(ranges, 5) > 0.30:
        return None
    _, _, vh = np.linalg.svd(flat - center, full_matrices=False)
    normal = vh[-1]
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    residual_mm = np.abs((flat - center) @ normal) * 1000.0
    ray = center / max(float(np.linalg.norm(center)), 1e-12)
    incidence = abs(float(np.dot(normal, ray)))
    return normal, float(np.median(residual_mm)), float(np.percentile(residual_mm, 95)), incidence


def local_planarity_rows(
    record: dict,
    raw: np.ndarray,
    post: np.ndarray,
    stride: int,
    grid_size: int,
) -> list[dict[str, object]]:
    raw_grid = camera_grid(record, raw, stride)
    post_grid = camera_grid(record, post, stride)
    rows: list[dict[str, object]] = []
    world_basis = np.asarray(
        record["viewInverse"][recorded_eye(record)], dtype=np.float64
    ).reshape(4, 4)[:3, :3]
    pitch = signed_degrees(
        record.get("callbackHeadWorldPose", {}).get("eulerDegrees", [0, 0, 0])[0]
    )
    height, width = raw_grid.shape[:2]
    for y in range(0, height - grid_size + 1, grid_size):
        for x in range(0, width - grid_size + 1, grid_size):
            raw_fit = fit_patch(raw_grid[y : y + grid_size, x : x + grid_size])
            post_fit = fit_patch(post_grid[y : y + grid_size, x : x + grid_size])
            if raw_fit is None or post_fit is None:
                continue
            raw_normal, raw_p50, raw_p95, incidence = raw_fit
            _, post_p50, post_p95, _ = post_fit
            world_normal = world_basis @ raw_normal
            world_normal /= max(float(np.linalg.norm(world_normal)), 1e-12)
            abs_y = abs(float(world_normal[1]))
            orientation = "horizontal" if abs_y >= 0.75 else ("vertical" if abs_y <= 0.30 else "oblique")
            center = np.nanmedian(
                raw_grid[y : y + grid_size, x : x + grid_size].reshape(-1, 3), axis=0
            )
            image_x = (x + grid_size * 0.5) / width * 2.0 - 1.0
            image_y = (y + grid_size * 0.5) / height * 2.0 - 1.0
            rows.append(
                {
                    "pairIndex": int(record["pairIndex"]),
                    "platformFrame": int(record["platformFrame"]),
                    "motion": record["_motion"],
                    "angularDegPerSec": float(record.get("angularDegPerSec", 0.0)),
                    "linearMps": float(record.get("linearMps", 0.0)),
                    "pitchDeg": pitch,
                    "orientation": orientation,
                    "rangeMetres": float(np.linalg.norm(center)),
                    "incidence": incidence,
                    "fieldRadius": float(max(abs(image_x), abs(image_y))),
                    "rawPlaneAbsP50Mm": raw_p50,
                    "rawPlaneAbsP95Mm": raw_p95,
                    "postPlaneAbsP50Mm": post_p50,
                    "postPlaneAbsP95Mm": post_p95,
                    "postMinusRawP95Mm": post_p95 - raw_p95,
                }
            )
    return rows


def dominant_mode(values: np.ndarray) -> tuple[float, int] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (np.abs(values) <= 150.0)]
    if len(values) < 5:
        return None
    edges = np.arange(-155.0, 165.0, 10.0)
    counts, _ = np.histogram(values, bins=edges)
    peak = int(np.argmax(counts))
    center = (edges[peak] + edges[peak + 1]) * 0.5
    members = values[np.abs(values - center) <= 20.0]
    if len(members) < 5:
        return None
    return float(np.median(members)), int(len(members))


def two_mode_test(values: np.ndarray) -> dict[str, float | int | bool | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (np.abs(values) <= 200.0)]
    if len(values) < 12:
        return {"detected": False, "count": int(len(values)), "separationMm": None, "minorSharePct": None, "sseImprovementPct": None, "histogramValleyRatio": None}
    centers = np.percentile(values, (25, 75)).astype(np.float64)
    for _ in range(24):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        updated = centers.copy()
        for cluster in (0, 1):
            if np.any(labels == cluster):
                updated[cluster] = np.median(values[labels == cluster])
        if np.allclose(updated, centers, atol=1e-4):
            break
        centers = updated
    labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
    counts = np.bincount(labels, minlength=2)
    median = float(np.median(values))
    sse_one = float(np.sum((values - median) ** 2))
    sse_two = float(np.sum((values - centers[labels]) ** 2))
    improvement = 1.0 - sse_two / max(sse_one, 1e-9)
    separation = abs(float(centers[1] - centers[0]))
    minor_share = float(np.min(counts) / len(values))
    histogram, edges = np.histogram(values, bins=np.arange(-102.5, 107.5, 5.0))
    smooth = np.convolve(histogram.astype(np.float64), np.asarray([0.25, 0.50, 0.25]), mode="same")
    local_peaks = [
        index
        for index in range(1, len(smooth) - 1)
        if smooth[index] >= smooth[index - 1] and smooth[index] >= smooth[index + 1]
    ]
    best_valley_ratio = 1.0
    has_separated_valley = False
    for left_ordinal, left in enumerate(local_peaks):
        for right in local_peaks[left_ordinal + 1 :]:
            peak_separation = (right - left) * 5.0
            if peak_separation < 20.0:
                continue
            smaller_peak = min(smooth[left], smooth[right])
            if smaller_peak < max(3.0, len(values) * 0.02):
                continue
            valley = float(np.min(smooth[left + 1 : right])) if right > left + 1 else smaller_peak
            ratio = valley / max(float(smaller_peak), 1e-9)
            best_valley_ratio = min(best_valley_ratio, ratio)
            if ratio <= 0.60:
                has_separated_valley = True
    detected = (
        separation >= 20.0
        and minor_share >= 0.15
        and improvement >= 0.50
        and has_separated_valley
    )
    return {
        "detected": bool(detected),
        "count": int(len(values)),
        "separationMm": separation,
        "minorSharePct": minor_share * 100.0,
        "sseImprovementPct": improvement * 100.0,
        "histogramValleyRatio": best_valley_ratio if local_peaks else None,
    }


def quantile_fields(prefix: str, values: np.ndarray) -> dict[str, object]:
    stats = distribution(values)
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def candidate_consistency(
    candidate_count: int,
    candidate_ids: np.ndarray,
    view_ids: np.ndarray,
    raw_normal_mm: np.ndarray,
    post_normal_mm: np.ndarray,
    incidence: np.ndarray,
    radius: np.ndarray,
    angular: np.ndarray,
    linear: np.ndarray,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    anchors = np.full(candidate_count, np.nan, dtype=np.float64)
    anchor_member_count = np.zeros(candidate_count, dtype=np.int32)
    rows: list[dict[str, object]] = []
    order = np.argsort(candidate_ids, kind="stable")
    ordered_ids = candidate_ids[order]
    starts = np.searchsorted(ordered_ids, np.arange(candidate_count), side="left")
    ends = np.searchsorted(ordered_ids, np.arange(candidate_count), side="right")
    for candidate in range(candidate_count):
        indices = order[starts[candidate] : ends[candidate]]
        if not len(indices):
            continue
        unique_views = np.unique(view_ids[indices])
        training = indices[
            (view_ids[indices] % 2 == 0)
            & (angular[indices] <= 10.0)
            & (linear[indices] <= 0.05)
            & (incidence[indices] >= 0.60)
            & (radius[indices] <= 0.75)
        ]
        if len(np.unique(view_ids[training])) < 3:
            continue
        mode = dominant_mode(raw_normal_mm[training])
        if mode is None:
            continue
        anchor, members = mode
        anchors[candidate] = anchor
        anchor_member_count[candidate] = members
        heldout = indices[view_ids[indices] % 2 == 1]
        heldout_raw = raw_normal_mm[heldout] - anchor
        heldout_post = post_normal_mm[heldout] - anchor
        near = indices[
            (np.abs(raw_normal_mm[indices] - anchor) <= 100.0)
            & (np.abs(post_normal_mm[indices] - raw_normal_mm[indices]) <= 40.0)
        ]
        heldout_near = heldout[
            (np.abs(raw_normal_mm[heldout] - anchor) <= 100.0)
            & (np.abs(post_normal_mm[heldout] - raw_normal_mm[heldout]) <= 40.0)
        ]
        modes = two_mode_test(raw_normal_mm[near] - anchor)
        row: dict[str, object] = {
            "candidateIndex": candidate,
            "observationCount": int(len(indices)),
            "independentViews": int(len(unique_views)),
            "trainingObservationCount": int(len(training)),
            "trainingIndependentViews": int(len(np.unique(view_ids[training]))),
            "anchorModeMembers": members,
            "anchorRawNormalOffsetMm": anchor,
            "heldoutObservationCount": int(len(heldout)),
            "heldoutIndependentViews": int(len(np.unique(view_ids[heldout]))),
            "heldoutNearSurfaceCount": int(len(heldout_near)),
            "heldoutOtherSurfaceOrFreePct": float(
                (1.0 - len(heldout_near) / max(len(heldout), 1)) * 100.0
            ),
            "twoModesDetected": int(bool(modes["detected"])),
            "twoModeSeparationMm": modes["separationMm"],
            "twoModeMinorSharePct": modes["minorSharePct"],
            "twoModeSseImprovementPct": modes["sseImprovementPct"],
            "twoModeHistogramValleyRatio": modes["histogramValleyRatio"],
        }
        row.update(quantile_fields("heldoutRawSignedMm", heldout_raw))
        row.update(quantile_fields("heldoutRawAbsMm", np.abs(heldout_raw)))
        row.update(quantile_fields("heldoutPostAbsMm", np.abs(heldout_post)))
        rows.append(row)
    return rows, anchors, anchor_member_count


def binned_rows(
    stage: str,
    errors: np.ndarray,
    candidate_ids: np.ndarray,
    dimensions: dict[str, list[tuple[str, np.ndarray]]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for dimension, groups in dimensions.items():
        for label, mask in groups:
            values = errors[mask]
            finite = np.isfinite(values)
            values = values[finite]
            ids = candidate_ids[mask][finite]
            stats = distribution(values)
            result.append(
                {
                    "stage": stage,
                    "dimension": dimension,
                    "group": label,
                    "observations": int(len(values)),
                    "candidates": int(len(np.unique(ids))),
                    "signedP50Mm": stats["p50"],
                    "signedP05Mm": stats["p05"],
                    "signedP95Mm": stats["p95"],
                    "absP50Mm": float(np.percentile(np.abs(values), 50)) if len(values) else None,
                    "absP90Mm": float(np.percentile(np.abs(values), 90)) if len(values) else None,
                    "absP95Mm": float(np.percentile(np.abs(values), 95)) if len(values) else None,
                    "over20mmPct": float(np.mean(np.abs(values) >= 20.0) * 100.0) if len(values) else None,
                    "over50mmPct": float(np.mean(np.abs(values) >= 50.0) * 100.0) if len(values) else None,
                }
            )
    return result


def fixed_effect_regression(
    errors: np.ndarray,
    candidate_ids: np.ndarray,
    features: dict[str, np.ndarray],
) -> dict[str, object]:
    valid = np.isfinite(errors)
    for values in features.values():
        valid &= np.isfinite(values)
    y = errors[valid].astype(np.float64)
    ids = candidate_ids[valid]
    x = np.column_stack([values[valid] for values in features.values()]).astype(np.float64)
    if len(y) < 100:
        return {"observationCount": int(len(y)), "status": "insufficient"}
    counts = np.bincount(ids)
    y_means = np.bincount(ids, weights=y) / np.maximum(counts, 1)
    y = y - y_means[ids]
    for column in range(x.shape[1]):
        means = np.bincount(ids, weights=x[:, column]) / np.maximum(counts, 1)
        x[:, column] -= means[ids]
    scales = np.std(x, axis=0)
    usable = scales > 1e-9
    x_scaled = x[:, usable] / scales[usable]
    y_scale = float(np.std(y))
    design = np.column_stack((np.ones(len(y)), x_scaled))
    coefficients, _, _, singular = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coefficients
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum((y - predicted) ** 2)) / max(denominator, 1e-12)
    names = np.asarray(list(features.keys()), dtype=object)[usable]
    return {
        "status": "ok",
        "observationCount": int(len(y)),
        "candidateCount": int(len(np.unique(ids))),
        "withinCandidateR2": r2,
        "responseStdMm": y_scale,
        "conditionNumber": float(singular[0] / max(singular[-1], 1e-12)),
        "standardizedEffectsMmPerOneFeatureStd": {
            str(name): float(value) for name, value in zip(names, coefficients[1:])
        },
        "featureStd": {
            name: float(scale) for name, scale in zip(features.keys(), scales)
        },
        "readingBoundary": "association after candidate fixed effects; not causal proof",
    }


def main() -> int:
    args = parse_args()
    session = args.session.resolve()
    candidates_path = args.candidates.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    all_records, manifest = load_records(session)
    records = all_records[:: max(args.frame_stride, 1)]
    if args.frame_limit > 0:
        records = records[: args.frame_limit]
    view_ids_list, view_summary = cluster_independent_views(records, 0.08, 3.0)

    candidate_rows = read_csv(candidates_path)
    centers = np.asarray(
        [
            [candidate_number(row, "centerX", "center_x_m"), candidate_number(row, "centerY", "center_y_m"), candidate_number(row, "centerZ", "center_z_m")]
            for row in candidate_rows
        ],
        dtype=np.float32,
    )
    normals = np.asarray(
        [
            [candidate_number(row, "normalX", "normal_x"), candidate_number(row, "normalY", "normal_y"), candidate_number(row, "normalZ", "normal_z")]
            for row in candidate_rows
        ],
        dtype=np.float32,
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    observation_parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "candidate",
            "pair",
            "view",
            "rawNormalMm",
            "postNormalMm",
            "rawPostNormalMm",
            "incidence",
            "range",
            "radius",
            "angular",
            "linear",
            "pitch",
            "yaw",
            "roll",
        )
    }
    planarity: list[dict[str, object]] = []
    raw_file_count = post_file_count = meta_file_count = short_payloads = 0

    print(
        f"Raw-depth trust audit: frames={len(records)}/{len(all_records)}, "
        f"candidates={len(candidate_rows)}, views={len(view_summary)}",
        flush=True,
    )
    for ordinal, (record, view_id) in enumerate(zip(records, view_ids_list), 1):
        row = record["_manifest"]
        frames = session / "depth_pairs" / "frames"
        raw_file_count += int((frames / row["rawFile"]).is_file())
        post_file_count += int((frames / row["processedFile"]).is_file())
        meta_file_count += int((frames / row["metadataFile"]).is_file())
        try:
            raw, post = load_depth_pair(session, record)
        except RuntimeError:
            short_payloads += 1
            raise

        sampled = project_and_sample_ranges(
            record,
            raw,
            post,
            centers,
            normals,
            args.outer_radius,
            args.minimum_incidence,
        )
        valid = (
            np.isfinite(sampled["rawRange"])
            & np.isfinite(sampled["postRange"])
            & (sampled["commonSamples"] >= 3)
        )
        ids = sampled["indices"][valid]
        incidence = sampled["incidence"][valid]
        raw_normal = (sampled["rawRange"][valid] - sampled["candidateRange"][valid]) * incidence * 1000.0
        post_normal = (sampled["postRange"][valid] - sampled["candidateRange"][valid]) * incidence * 1000.0
        pose = record.get("callbackHeadWorldPose", {})
        euler = [signed_degrees(value) for value in pose.get("eulerDegrees", [0.0, 0.0, 0.0])]
        count = len(ids)
        observation_parts["candidate"].append(ids.astype(np.int32))
        observation_parts["pair"].append(np.full(count, int(record["pairIndex"]), dtype=np.int32))
        observation_parts["view"].append(np.full(count, view_id, dtype=np.int16))
        observation_parts["rawNormalMm"].append(raw_normal.astype(np.float32))
        observation_parts["postNormalMm"].append(post_normal.astype(np.float32))
        observation_parts["rawPostNormalMm"].append((post_normal - raw_normal).astype(np.float32))
        observation_parts["incidence"].append(incidence.astype(np.float32))
        observation_parts["range"].append(sampled["candidateRange"][valid].astype(np.float32))
        observation_parts["radius"].append(sampled["viewRadius"][valid].astype(np.float32))
        observation_parts["angular"].append(np.full(count, float(record.get("angularDegPerSec", 0.0)), dtype=np.float32))
        observation_parts["linear"].append(np.full(count, float(record.get("linearMps", 0.0)), dtype=np.float32))
        observation_parts["pitch"].append(np.full(count, float(euler[0]), dtype=np.float32))
        observation_parts["yaw"].append(np.full(count, float(euler[1]), dtype=np.float32))
        observation_parts["roll"].append(np.full(count, float(euler[2]), dtype=np.float32))
        planarity.extend(
            local_planarity_rows(
                record,
                raw,
                post,
                args.patch_pixel_stride,
                args.patch_grid_size,
            )
        )
        if ordinal % 100 == 0 or ordinal == len(records):
            observations = sum(len(part) for part in observation_parts["candidate"])
            print(f"  frames {ordinal}/{len(records)}, observations={observations}, patches={len(planarity)}", flush=True)

    arrays = {
        key: np.concatenate(parts) if parts else np.empty(0)
        for key, parts in observation_parts.items()
    }
    consistency, anchors, anchor_counts = candidate_consistency(
        len(candidate_rows),
        arrays["candidate"],
        arrays["view"],
        arrays["rawNormalMm"],
        arrays["postNormalMm"],
        arrays["incidence"],
        arrays["radius"],
        arrays["angular"],
        arrays["linear"],
    )
    baseline = anchors[arrays["candidate"]]
    heldout = (arrays["view"] % 2 == 1) & np.isfinite(baseline)
    raw_error = arrays["rawNormalMm"] - baseline
    post_error = arrays["postNormalMm"] - baseline
    near_surface_heldout = (
        heldout
        & (np.abs(raw_error) <= 100.0)
        & (np.abs(arrays["rawPostNormalMm"]) <= 40.0)
        & (arrays["incidence"] >= 0.15)
    )

    motion_groups = [
        ("still", (arrays["angular"] <= 2.0) & (arrays["linear"] <= 0.01)),
        ("slow", (arrays["angular"] > 2.0) & (arrays["angular"] <= 10.0) & (arrays["linear"] <= 0.05)),
        ("medium", (arrays["angular"] > 10.0) & (arrays["angular"] <= 30.0) & (arrays["linear"] <= 0.20)),
        ("fast", (arrays["angular"] > 30.0) | (arrays["linear"] > 0.20)),
    ]
    dimensions = {
        "incidence": [
            ("0.15-0.30_grazing", (arrays["incidence"] >= 0.15) & (arrays["incidence"] < 0.30)),
            ("0.30-0.50", (arrays["incidence"] >= 0.30) & (arrays["incidence"] < 0.50)),
            ("0.50-0.70", (arrays["incidence"] >= 0.50) & (arrays["incidence"] < 0.70)),
            ("0.70-0.85", (arrays["incidence"] >= 0.70) & (arrays["incidence"] < 0.85)),
            ("0.85-1.00_front", arrays["incidence"] >= 0.85),
        ],
        "range": [
            ("0.15-0.75m", (arrays["range"] >= 0.15) & (arrays["range"] < 0.75)),
            ("0.75-1.25m", (arrays["range"] >= 0.75) & (arrays["range"] < 1.25)),
            ("1.25-2.00m", (arrays["range"] >= 1.25) & (arrays["range"] < 2.00)),
            ("2.00-3.00m", (arrays["range"] >= 2.00) & (arrays["range"] < 3.00)),
            ("3.00-8.00m", arrays["range"] >= 3.00),
        ],
        "field": [
            ("center", arrays["radius"] <= 0.45),
            ("middle", (arrays["radius"] > 0.45) & (arrays["radius"] <= 0.75)),
            ("edge", arrays["radius"] > 0.75),
        ],
        "motion": motion_groups,
    }
    heldout_dimensions = {
        name: [(label, mask & near_surface_heldout) for label, mask in groups]
        for name, groups in dimensions.items()
    }
    strata = binned_rows("platform_pre_qrs", raw_error, arrays["candidate"], heldout_dimensions)
    strata += binned_rows("post_qrs", post_error, arrays["candidate"], heldout_dimensions)

    regression_mask = near_surface_heldout
    signed_regression = fixed_effect_regression(
        np.where(regression_mask, raw_error, np.nan),
        arrays["candidate"],
        {
            "grazing_1_minus_incidence": 1.0 - arrays["incidence"],
            "range_metres": arrays["range"],
            "field_radius": arrays["radius"],
            "angular_deg_per_sec": arrays["angular"],
            "linear_metres_per_sec": arrays["linear"],
            "head_pitch_degrees": arrays["pitch"],
            "head_yaw_degrees": arrays["yaw"],
        },
    )
    absolute_regression = fixed_effect_regression(
        np.where(regression_mask, np.abs(raw_error), np.nan),
        arrays["candidate"],
        {
            "grazing_1_minus_incidence": 1.0 - arrays["incidence"],
            "range_metres": arrays["range"],
            "field_radius": arrays["radius"],
            "angular_deg_per_sec": arrays["angular"],
            "linear_metres_per_sec": arrays["linear"],
            "head_pitch_degrees": arrays["pitch"],
            "head_yaw_degrees": arrays["yaw"],
        },
    )

    planarity_fields = [
        "pairIndex", "platformFrame", "motion", "angularDegPerSec", "linearMps",
        "pitchDeg", "orientation", "rangeMetres", "incidence", "fieldRadius",
        "rawPlaneAbsP50Mm", "rawPlaneAbsP95Mm", "postPlaneAbsP50Mm",
        "postPlaneAbsP95Mm", "postMinusRawP95Mm",
    ]
    consistency_fields = list(consistency[0].keys()) if consistency else ["candidateIndex"]
    strata_fields = list(strata[0].keys()) if strata else ["stage", "dimension", "group"]
    write_csv(out / "local_planarity_patches.csv", planarity, planarity_fields)
    write_csv(out / "candidate_consistency.csv", consistency, consistency_fields)
    write_csv(out / "heldout_condition_strata.csv", strata, strata_fields)
    write_csv(out / "independent_view_clusters.csv", view_summary, list(view_summary[0].keys()) if view_summary else ["viewId"])
    np.savez_compressed(out / "candidate_observations.npz", **arrays, anchorRawNormalMm=anchors, anchorMemberCount=anchor_counts)

    patch_raw_all = np.asarray([float(row["rawPlaneAbsP95Mm"]) for row in planarity])
    patch_post_all = np.asarray([float(row["postPlaneAbsP95Mm"]) for row in planarity])
    structural_patch = np.asarray(
        [
            float(row["incidence"]) >= 0.70
            and float(row["fieldRadius"]) <= 0.75
            and 0.50 <= float(row["rangeMetres"]) <= 4.0
            and row["orientation"] in ("horizontal", "vertical")
            for row in planarity
        ],
        dtype=bool,
    )
    patch_raw = patch_raw_all[structural_patch]
    patch_post = patch_post_all[structural_patch]
    candidate_two_mode = np.asarray([int(row["twoModesDetected"]) for row in consistency], dtype=np.int32)
    heldout_raw = raw_error[heldout]
    heldout_post = post_error[heldout]
    near_heldout_raw = raw_error[near_surface_heldout]
    near_heldout_post = post_error[near_surface_heldout]
    qrs_delta = arrays["rawPostNormalMm"]
    data_quality = {
        "manifestRows": len(manifest),
        "statusOkRows": sum(row.get("status") == "ok" for row in manifest),
        "selectedFrames": len(records),
        "metadataFiles": meta_file_count,
        "rawFiles": raw_file_count,
        "postFiles": post_file_count,
        "shortPayloads": short_payloads,
        "timestampValidRows": sum(bool(record.get("timestampValid")) for record in records),
        "poseAvailableRows": sum(bool(record.get("callbackHeadWorldPose", {}).get("available")) for record in records),
        "candidateAddresses": len(candidate_rows),
        "independentViewClusters": len(view_summary),
        "candidateObservations": int(len(arrays["candidate"])),
        "anchoredCandidates": int(np.count_nonzero(np.isfinite(anchors))),
        "heldoutObservations": int(np.count_nonzero(heldout)),
        "nearSurfaceHeldoutObservations": int(np.count_nonzero(near_surface_heldout)),
        "otherSurfaceFreeOrLargeConflictObservations": int(
            np.count_nonzero(heldout & ~near_surface_heldout)
        ),
    }
    summary = {
        "schema": "scancover.raw_depth_trust_audit.v1",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "authority": "read_only_offline_diagnostic",
        "productionConsumers": 0,
        "session": str(session),
        "candidateAddressSource": str(candidates_path),
        "candidateAddressIsGroundTruth": False,
        "surfacePositionRule": "dominant platform-depth mode from low-motion central high-incidence even view clusters",
        "evaluationRule": "odd independent view clusters held out from surface-position estimation",
        "dataQuality": data_quality,
        "platformPreQrsHeldoutNormalErrorMm": distribution(heldout_raw),
        "platformPreQrsHeldoutAbsNormalErrorMm": distribution(np.abs(heldout_raw)),
        "postQrsHeldoutAbsNormalErrorMm": distribution(np.abs(heldout_post)),
        "nearSurfacePlatformPreQrsHeldoutNormalErrorMm": distribution(near_heldout_raw),
        "nearSurfacePlatformPreQrsHeldoutAbsNormalErrorMm": distribution(np.abs(near_heldout_raw)),
        "nearSurfacePostQrsHeldoutAbsNormalErrorMm": distribution(np.abs(near_heldout_post)),
        "postMinusRawNormalMm": distribution(qrs_delta),
        "localPlanarity": {
            "allPatchCount": len(planarity),
            "structuralPriorPatchCount": int(np.count_nonzero(structural_patch)),
            "platformPreQrsPlaneAbsP95Mm": distribution(patch_raw),
            "postQrsPlaneAbsP95Mm": distribution(patch_post),
            "rawPatchP95Over10mmPct": float(np.mean(patch_raw >= 10.0) * 100.0) if len(patch_raw) else None,
            "rawPatchP95Over20mmPct": float(np.mean(patch_raw >= 20.0) * 100.0) if len(patch_raw) else None,
            "postImprovesPatchP95Pct": float(np.mean(patch_post < patch_raw) * 100.0) if len(patch_raw) else None,
        },
        "candidateConsistency": {
            "auditableCandidates": len(consistency),
            "twoModeCandidates": int(np.sum(candidate_two_mode)),
            "twoModeCandidatePct": float(np.mean(candidate_two_mode) * 100.0) if len(candidate_two_mode) else None,
        },
        "withinCandidateSignedRegression": signed_regression,
        "withinCandidateAbsoluteErrorRegression": absolute_regression,
        "limits": [
            "candidate centers and normals are addresses only, not truth",
            "room planarity is a structural prior; real non-planar objects can inflate patch residuals",
            "one natural scan can expose associations but cannot fully randomize angle, range, motion, and surface identity",
            "no external ruler or mesh means stable absolute range bias cannot be convicted",
            "timing-offset and best-rigid-pose sweeps are a separate next audit stage",
        ],
        "artifacts": {
            "candidateObservations": "candidate_observations.npz",
            "candidateConsistency": "candidate_consistency.csv",
            "conditionStrata": "heldout_condition_strata.csv",
            "localPlanarity": "local_planarity_patches.csv",
            "viewClusters": "independent_view_clusters.csv",
        },
    }
    (out / "raw_depth_trust_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# ScanCover raw-depth trust audit",
        "",
        "> Read-only offline evidence. Candidate geometry is used as an address, not as ground truth.",
        "",
        "## Data quality",
        "",
        f"- Complete selected frames: {data_quality['selectedFrames']}",
        f"- Candidate observations: {data_quality['candidateObservations']}",
        f"- Anchored candidates: {data_quality['anchoredCandidates']}",
        f"- Held-out observations: {data_quality['heldoutObservations']}",
        "",
        "## First-pass geometry",
        "",
        f"- All projective disagreements (includes occlusion/free space): {data_quality['heldoutObservations']}",
        f"- Near-surface held-out observations used for depth consistency: {data_quality['nearSurfaceHeldoutObservations']}",
        f"- Near-surface raw absolute normal error p50/p95: {format_number(summary['nearSurfacePlatformPreQrsHeldoutAbsNormalErrorMm']['p50'])} / {format_number(summary['nearSurfacePlatformPreQrsHeldoutAbsNormalErrorMm']['p95'])} mm",
        f"- Near-surface post-QRS absolute normal error p50/p95: {format_number(summary['nearSurfacePostQrsHeldoutAbsNormalErrorMm']['p50'])} / {format_number(summary['nearSurfacePostQrsHeldoutAbsNormalErrorMm']['p95'])} mm",
        f"- Auditable candidates with two modes: {summary['candidateConsistency']['twoModeCandidates']} / {summary['candidateConsistency']['auditableCandidates']} ({format_number(summary['candidateConsistency']['twoModeCandidatePct'])}%)",
        f"- Raw structural-prior patch plane-residual p95 distribution median/p95: {format_number(summary['localPlanarity']['platformPreQrsPlaneAbsP95Mm']['p50'])} / {format_number(summary['localPlanarity']['platformPreQrsPlaneAbsP95Mm']['p95'])} mm",
        "",
        "## Reading boundary",
        "",
        "- This run can convict relative inconsistency and structured associations.",
        "- It cannot convict a stable absolute range bias without an external physical reference.",
        "- Timing-offset and rigid-pose sweeps remain a separate causal stage.",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
