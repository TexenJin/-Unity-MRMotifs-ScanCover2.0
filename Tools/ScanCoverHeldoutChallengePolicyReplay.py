#!/usr/bin/env python3
"""Held-out Quest replay for GunGel free-space challenge policies.

The reference court used a deterministic 160-still/220-slow frame sample. This
tool recreates that sample, removes it, and evaluates policies on disjoint depth
pairs.  It never writes to the Quest or the Unity project.

Positive controls are production stable candidates already confirmed current
false after their final safe geometry. Negative controls are production stable
candidates conservatively confirmed as surfaces. Unknown candidates are never
counted as either success or failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from ScanCoverDepthPairDeviceAudit import (
    DEFAULT_ADB,
    annotate_motion,
    even_sample,
    fetch_pair_batch,
    load_metadata,
    recorded_eye,
)
from ScanCoverOfflineReferenceCourt import apply_matrix, project_and_sample_ranges
from ScanCoverProductionFreeRayAudit import (
    CHALLENGE_MOTION_MIN,
    DUAL_AGREE_BASE,
    DUAL_AGREE_RANGE_SCALE,
    FREE_SPACE_MARGIN_BASE,
    FREE_SPACE_MARGIN_RANGE_SCALE,
    device_manifest,
    independent_event_views,
    integer,
    number,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def camera_state(record: dict) -> tuple[np.ndarray, np.ndarray]:
    eye = recorded_eye(record)
    view_inverse = np.asarray(record["viewInverse"][eye], dtype=np.float64).reshape(4, 4)
    position = view_inverse[:3, 3]
    forward = -view_inverse[:3, 2]
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    return position, forward


def project_candidates(record: dict, centers: np.ndarray) -> dict[str, np.ndarray]:
    eye = recorded_eye(record)
    projection = np.asarray(record["projection"][eye], dtype=np.float64).reshape(4, 4)
    view = np.asarray(record["view"][eye], dtype=np.float64).reshape(4, 4)
    world_h = np.column_stack((centers.astype(np.float64), np.ones(len(centers))))
    clip = (projection @ (view @ world_h.T)).T
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc = clip[:, :3] / clip[:, 3:4]
    uv = (ndc[:, :2] + 1.0) * 0.5
    width = int(record["width"])
    height = int(record["height"])
    pixel_x = np.rint(uv[:, 0] * width).astype(np.int64)
    pixel_y = np.rint(uv[:, 1] * height).astype(np.int64)
    camera, _ = camera_state(record)
    to_candidate = centers.astype(np.float64) - camera[None, :]
    candidate_range = np.linalg.norm(to_candidate, axis=1)
    visible = (
        np.isfinite(ndc).all(axis=1)
        & (clip[:, 3] > 0.0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < 1.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < 1.0)
        & (candidate_range >= 0.15)
        & (candidate_range <= 8.0)
    )
    return {
        "ndc": ndc,
        "uv": uv,
        "pixelX": pixel_x,
        "pixelY": pixel_y,
        "candidateRange": candidate_range,
        "visible": visible,
        "camera": camera,
    }


def reconstruct_grid(record: dict, raw: np.ndarray, post: np.ndarray, stride: int) -> dict[str, np.ndarray]:
    width = int(record["width"])
    height = int(record["height"])
    raw_image = raw.reshape(height, width)
    post_image = post.reshape(height, width)
    xs = np.arange(0, width, stride, dtype=np.int64)
    ys = np.arange(0, height, stride, dtype=np.int64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    raw_depth = raw_image[grid_y, grid_x].reshape(-1).astype(np.float64)
    post_depth = post_image[grid_y, grid_x].reshape(-1).astype(np.float64)
    valid = (
        np.isfinite(raw_depth)
        & np.isfinite(post_depth)
        & (raw_depth > 0.0)
        & (raw_depth < 1.0)
        & (post_depth > 0.0)
        & (post_depth < 1.0)
    )
    eye = recorded_eye(record)
    projection_inverse = np.asarray(record["projectionInverse"][eye], dtype=np.float64).reshape(4, 4)
    view_inverse = np.asarray(record["viewInverse"][eye], dtype=np.float64).reshape(4, 4)
    u = grid_x.reshape(-1).astype(np.float64) / width
    v = grid_y.reshape(-1).astype(np.float64) / height

    def reconstruct(depth: np.ndarray) -> np.ndarray:
        hcs = np.stack((u * 2.0 - 1.0, v * 2.0 - 1.0, depth * 2.0 - 1.0, np.ones_like(u)))
        world_h = view_inverse @ (projection_inverse @ hcs)
        return (world_h[:3] / world_h[3]).T

    raw_world = reconstruct(raw_depth)
    post_world = reconstruct(post_depth)
    camera = view_inverse[:3, 3]
    raw_range = np.linalg.norm(raw_world - camera[None, :], axis=1)
    post_range = np.linalg.norm(post_world - camera[None, :], axis=1)
    processed_range = post_range
    dual_limit = DUAL_AGREE_BASE + processed_range * DUAL_AGREE_RANGE_SCALE
    dual = valid & (np.abs(raw_range - post_range) <= dual_limit)
    observed_world = (raw_world + post_world) * 0.5
    ray = observed_world - camera[None, :]
    observed_range = np.linalg.norm(ray, axis=1)
    direction = ray / np.maximum(observed_range[:, None], 1e-12)
    gx_count = len(xs)
    gy_count = len(ys)
    indices = np.arange(gx_count * gy_count, dtype=np.int64)
    gx = indices % gx_count
    gy = indices // gx_count
    central = (gx >= 5) & (gx <= gx_count - 6) & (gy >= 5) & (gy <= gy_count - 6)
    return {
        "direction": direction,
        "observedRange": observed_range,
        "dual": dual,
        "central": central,
        "gridX": gx_count,
        "gridY": gy_count,
        "stride": stride,
    }


def tile_half_diagonal(record: dict, candidate_range: np.ndarray, grid_x: int, grid_y: int) -> np.ndarray:
    eye = recorded_eye(record)
    fov = record["fovRadians"][eye]
    horizontal = abs(float(fov[1]) - float(fov[0]))
    vertical = abs(float(fov[2]) - float(fov[3]))
    angular = math.hypot(horizontal / grid_x * 0.5, vertical / grid_y * 0.5)
    return candidate_range * math.tan(angular)


def cone_hits(
    record: dict,
    centers: np.ndarray,
    projection: dict[str, np.ndarray],
    grid: dict[str, np.ndarray],
    eligible: np.ndarray,
    qrs_metres: float,
    multiplier: float = 2.0,
    neighbour_radius: int = 4,
) -> np.ndarray:
    result = np.zeros(len(centers), dtype=bool)
    target_indices = np.flatnonzero(eligible & projection["visible"])
    if not len(target_indices):
        return result
    grid_x = int(grid["gridX"])
    grid_y = int(grid["gridY"])
    stride = int(grid["stride"])
    offsets = np.asarray(
        [(dx, dy) for dy in range(-neighbour_radius, neighbour_radius + 1)
         for dx in range(-neighbour_radius, neighbour_radius + 1)],
        dtype=np.int64,
    )
    base_x = np.rint(projection["pixelX"][target_indices] / stride).astype(np.int64)
    base_y = np.rint(projection["pixelY"][target_indices] / stride).astype(np.int64)
    nx = np.clip(base_x[:, None] + offsets[None, :, 0], 0, grid_x - 1)
    ny = np.clip(base_y[:, None] + offsets[None, :, 1], 0, grid_y - 1)
    ray_indices = ny * grid_x + nx
    ray_direction = grid["direction"][ray_indices]
    observed_range = grid["observedRange"][ray_indices]
    ray_valid = grid["dual"][ray_indices] & grid["central"][ray_indices]
    camera = projection["camera"]
    to_candidate = centers[target_indices].astype(np.float64) - camera[None, :]
    candidate_along = np.sum(to_candidate[:, None, :] * ray_direction, axis=2)
    tangent = np.linalg.norm(
        to_candidate[:, None, :] - ray_direction * candidate_along[:, :, None], axis=2
    )
    free_margin = FREE_SPACE_MARGIN_BASE + observed_range * FREE_SPACE_MARGIN_RANGE_SCALE
    free = (
        ray_valid
        & (observed_range > 0.20)
        & (candidate_along > 0.15)
        & (candidate_along < observed_range - free_margin)
    )
    radius = multiplier * tile_half_diagonal(
        record, projection["candidateRange"][target_indices], grid_x, grid_y
    ) + qrs_metres
    hit = np.any(free & (tangent <= radius[:, None]), axis=1)
    result[target_indices] = hit
    return result


def append_events(
    destination: defaultdict[tuple[int, str], list[dict[str, object]]],
    stable_ids: np.ndarray,
    mask: np.ndarray,
    policy: str,
    record: dict,
    gun_gel_frame: int,
    centers: np.ndarray,
) -> None:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return
    camera, _ = camera_state(record)
    for index in indices:
        view = camera - centers[index]
        view /= max(float(np.linalg.norm(view)), 1e-12)
        destination[(int(stable_ids[index]), policy)].append(
            {
                "stableId": int(stable_ids[index]),
                "policy": policy,
                "pairIndex": integer(record.get("pairIndex")),
                "platformFrame": integer(record.get("platformFrame")),
                "gunGelFrame": gun_gel_frame,
                "cameraX": float(camera[0]),
                "cameraY": float(camera[1]),
                "cameraZ": float(camera[2]),
                "candidateViewX": float(view[0]),
                "candidateViewY": float(view[1]),
                "candidateViewZ": float(view[2]),
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb", type=Path, default=DEFAULT_ADB)
    parser.add_argument("--serial", default="2G97C5ZH4501R5")
    parser.add_argument("--session", required=True)
    parser.add_argument("--reference-candidates", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--current-false-finding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-still", type=int, default=160)
    parser.add_argument("--train-slow", type=int, default=220)
    parser.add_argument("--heldout-still", type=int, default=240)
    parser.add_argument("--heldout-slow", type=int, default=360)
    parser.add_argument("--tail-start-frame", type=int, default=4437)
    parser.add_argument("--batch-size", type=int, default=6)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_rows = read_csv(args.reference_candidates)
    controls = [
        row for row in reference_rows
        if row.get("state") == "stable" and row.get("referenceStatus") == "confirmed_surface"
    ]
    finding = json.loads(args.current_false_finding.read_text(encoding="utf-8"))
    positive_ids = {
        integer(item) for item in finding.get("currentFalseConfirmedStableIds", [])
    }
    positives = [row for row in reference_rows if integer(row.get("stable_id")) in positive_ids]
    targets = positives + controls
    stable_ids = np.asarray([integer(row.get("stable_id")) for row in targets], dtype=np.int64)
    centers = np.asarray(
        [[number(row.get("center_x_m")), number(row.get("center_y_m")), number(row.get("center_z_m"))]
         for row in targets], dtype=np.float64,
    )
    normals = np.asarray(
        [[number(row.get("normal_x")), number(row.get("normal_y")), number(row.get("normal_z"))]
         for row in targets], dtype=np.float64,
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    last_safe = np.asarray([integer(row.get("last_seen_frame")) for row in targets], dtype=np.int64)
    is_positive = np.asarray([int(stable_id in positive_ids) for stable_id in stable_ids], dtype=bool)

    summary = json.loads(args.reference_summary.read_text(encoding="utf-8"))
    qrs_metres = number(summary["qrsCalibration"]["absRangeDifferenceMmP95"]) / 1000.0
    nominal = next(profile for profile in summary["errorProfiles"] if profile["name"] == "nominal")
    support_metres = number(nominal["supportMm"]) / 1000.0

    session = args.session.rstrip("/")
    accepted = device_manifest(args.adb, args.serial, session)
    records = load_metadata(args.adb, args.serial, session)
    groups = annotate_motion(records)
    training = even_sample(groups["still"], args.train_still) + even_sample(groups["slow"], args.train_slow)
    training_ids = {integer(row.get("pairIndex")) for row in training}
    heldout_groups = {
        key: [row for row in values if integer(row.get("pairIndex")) not in training_ids]
        for key, values in groups.items()
    }
    selected = even_sample(heldout_groups["still"], args.heldout_still)
    selected += even_sample(heldout_groups["slow"], args.heldout_slow)
    selected_ids = {integer(row.get("pairIndex")) for row in selected}
    # Preserve every available accepted tail frame. This is required to test the
    # known current-false candidate after its final safe geometry, not merely to
    # make the held-out set look statistically large.
    for record in records:
        pair_index = integer(record.get("pairIndex"))
        if pair_index in training_ids or pair_index in selected_ids:
            continue
        fusion = accepted.get(integer(record.get("platformFrame")))
        if fusion is not None and integer(fusion.get("gunGelFrame")) >= args.tail_start_frame:
            selected.append(record)
            selected_ids.add(pair_index)
    selected.sort(key=lambda row: integer(row.get("pairIndex")))

    policy_events: defaultdict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    skipped = Counter()
    processed = 0
    accepted_frames = 0
    policies = (
        "exact_observed",
        "support_nominal",
        "direct_free_30mm",
        "direct_free_35mm",
        "direct_free_45mm",
        "direct_free_production_margin",
        "cone_2x_production_margin",
        "cone_2x_authorized_35mm",
    )

    print(
        f"Held-out replay: {len(selected)} depth pairs, {len(positives)} positive, "
        f"{len(controls)} surface controls",
        flush=True,
    )
    for start in range(0, len(selected), args.batch_size):
        batch = fetch_pair_batch(args.adb, args.serial, session, selected[start:start + args.batch_size])
        for record, raw, post in batch:
            processed += 1
            fusion = accepted.get(integer(record.get("platformFrame")))
            if fusion is None:
                skipped["no_accepted_fusion"] += 1
                continue
            accepted_frames += 1
            gun_gel_frame = integer(fusion.get("gunGelFrame"))
            motion_quality = number(fusion.get("motionQuality"))
            if motion_quality < CHALLENGE_MOTION_MIN:
                skipped["motion_quality"] += 1
                continue
            time_eligible = gun_gel_frame >= last_safe
            if not np.any(time_eligible):
                skipped["before_all_final_safe"] += 1
                continue

            sampled = project_and_sample_ranges(record, raw, post, centers, normals, 0.75, 0.15)
            exact_raw = np.full(len(targets), np.nan, dtype=np.float64)
            exact_post = np.full(len(targets), np.nan, dtype=np.float64)
            candidate_range = np.full(len(targets), np.nan, dtype=np.float64)
            exact_raw[sampled["indices"]] = sampled["rawRange"]
            exact_post[sampled["indices"]] = sampled["postRange"]
            candidate_range[sampled["indices"]] = sampled["candidateRange"]
            exact_valid = time_eligible & np.isfinite(exact_raw) & np.isfinite(exact_post)
            observed = np.minimum(exact_raw, exact_post)
            dual_limit = DUAL_AGREE_BASE + np.maximum(exact_post, 0.0) * DUAL_AGREE_RANGE_SCALE
            exact_valid &= np.abs(exact_raw - exact_post) <= dual_limit
            gap = observed - candidate_range
            support = exact_valid & (np.abs((exact_raw + exact_post) * 0.5 - candidate_range) <= support_metres)
            free_30 = exact_valid & (gap >= max(0.030, 3.0 * qrs_metres))
            free_35 = exact_valid & (gap >= 0.035)
            free_45 = exact_valid & (gap >= 0.045)
            production_margin = FREE_SPACE_MARGIN_BASE + observed * FREE_SPACE_MARGIN_RANGE_SCALE
            free_production = exact_valid & (gap >= production_margin)

            projection = project_candidates(record, centers)
            grid = reconstruct_grid(record, raw, post, 8)
            cone_2x = cone_hits(
                record, centers, projection, grid, time_eligible, qrs_metres, 2.0
            )
            cone_authorized = cone_2x & free_35

            append_events(policy_events, stable_ids, exact_valid, "exact_observed", record, gun_gel_frame, centers)
            append_events(policy_events, stable_ids, support, "support_nominal", record, gun_gel_frame, centers)
            append_events(policy_events, stable_ids, free_30, "direct_free_30mm", record, gun_gel_frame, centers)
            append_events(policy_events, stable_ids, free_35, "direct_free_35mm", record, gun_gel_frame, centers)
            append_events(policy_events, stable_ids, free_45, "direct_free_45mm", record, gun_gel_frame, centers)
            append_events(policy_events, stable_ids, free_production, "direct_free_production_margin", record, gun_gel_frame, centers)
            append_events(policy_events, stable_ids, cone_2x, "cone_2x_production_margin", record, gun_gel_frame, centers)
            append_events(policy_events, stable_ids, cone_authorized, "cone_2x_authorized_35mm", record, gun_gel_frame, centers)

        if processed % 60 < len(batch) or processed == len(selected):
            print(
                f"  depth {processed}/{len(selected)}, accepted {accepted_frames}, "
                f"event buckets {len(policy_events)}",
                flush=True,
            )

    candidate_results: list[dict[str, object]] = []
    for index, stable_id in enumerate(stable_ids):
        row: dict[str, object] = {
            "stableId": int(stable_id),
            "role": "positive_current_false" if is_positive[index] else "negative_confirmed_surface",
            "lastSafeFrame": int(last_safe[index]),
        }
        for policy in policies:
            events = policy_events.get((int(stable_id), policy), [])
            row[f"{policy}_events"] = len(events)
            row[f"{policy}_independentViews"] = independent_event_views(events)
        candidate_results.append(row)

    policy_results: list[dict[str, object]] = []
    challenge_policies = [
        policy for policy in policies if policy not in {"exact_observed", "support_nominal"}
    ]
    for policy in challenge_policies:
        positive_rows = [row for row in candidate_results if row["role"] == "positive_current_false"]
        negative_rows = [row for row in candidate_results if row["role"] == "negative_confirmed_surface"]
        caught = [row for row in positive_rows if integer(row[f"{policy}_independentViews"]) >= 2]
        challenged_negatives = [row for row in negative_rows if integer(row[f"{policy}_independentViews"]) >= 2]
        any_challenged_negatives = [
            row for row in negative_rows if integer(row[f"{policy}_events"]) > 0
        ]
        one_view_negatives = [
            row for row in negative_rows
            if integer(row[f"{policy}_independentViews"]) == 1
        ]
        persistent_negative = [
            row for row in challenged_negatives
            if integer(row["support_nominal_independentViews"]) >= 2
        ]
        exact_negative_covered = [
            row for row in negative_rows
            if integer(row["exact_observed_independentViews"]) >= 2
        ]
        negative_any_covered = [
            row for row in negative_rows
            if integer(row["exact_observed_independentViews"]) >= 2
            or integer(row[f"{policy}_events"]) > 0
        ]
        negative_two_view_covered = [
            row for row in negative_rows
            if integer(row["exact_observed_independentViews"]) >= 2
            or integer(row[f"{policy}_independentViews"]) >= 2
        ]
        policy_results.append(
            {
                "policy": policy,
                "positiveControls": len(positive_rows),
                "positiveCaughtAtTwoViews": len(caught),
                "negativeControls": len(negative_rows),
                "negativeControlsExactObservedAtTwoViews": len(exact_negative_covered),
                "negativeControlsWithAnyEventCoverage": len(negative_any_covered),
                "negativeControlsWithTwoViewCoverage": len(negative_two_view_covered),
                "negativeControlsAnyChallenge": len(any_challenged_negatives),
                "negativeControlsOneIndependentView": len(one_view_negatives),
                "negativeChallengedAtTwoViews": len(challenged_negatives),
                "negativeChallengedAndSupportedAtTwoViews": len(persistent_negative),
                "negativeChallengeRateAmongCovered": (
                    len(challenged_negatives) / len(negative_two_view_covered)
                    if negative_two_view_covered else None
                ),
                "negativeAnyChallengeRateAmongCovered": (
                    len(any_challenged_negatives) / len(negative_any_covered)
                    if negative_any_covered else None
                ),
                "productionReady": 0,
            }
        )

    flat_events = [event for events in policy_events.values() for event in events]
    write_csv(
        output / "heldout_policy_events.csv",
        flat_events,
        list(flat_events[0]) if flat_events else ["stableId"],
    )
    write_csv(
        output / "heldout_candidate_results.csv",
        candidate_results,
        sorted({key for row in candidate_results for key in row}),
    )
    write_csv(
        output / "heldout_policy_results.csv",
        policy_results,
        list(policy_results[0]) if policy_results else ["policy"],
    )

    report = {
        "schema": "scancover.heldout-challenge-policy-replay.v1",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "authority": "read_only_offline_diagnostic",
        "productionConsumers": 0,
        "split": {
            "trainingPairCount": len(training_ids),
            "heldoutSelectedPairCount": len(selected),
            "overlapCount": len(training_ids & selected_ids),
            "heldoutAcceptedFusionFrames": accepted_frames,
            "tailStartGunGelFrame": args.tail_start_frame,
        },
        "controls": {
            "positiveCurrentFalse": len(positives),
            "negativeConfirmedSurface": len(controls),
            "unknownExcluded": sum(
                row.get("state") == "stable" and row.get("referenceStatus") == "unknown"
                for row in reference_rows
            ),
        },
        "calibration": {
            "qrsP95Mm": qrs_metres * 1000.0,
            "nominalSupportMm": support_metres * 1000.0,
        },
        "skippedFrames": dict(skipped),
        "policyResults": policy_results,
        "decisionBoundary": [
            "held-out depth pairs are disjoint from the reference-court sample",
            "final candidate geometry is evaluated only at or after its last safe frame",
            "unknown candidates do not contribute to positive or negative rates",
            "a challenged surface control with later support is reported, not silently relabelled",
            "no policy is production-ready until the positive control is caught and surface-control challenge risk is acceptably low",
        ],
    }
    (output / "heldout_policy_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = "\n".join(
        "| {policy} | {positiveCaughtAtTwoViews}/{positiveControls} | "
        "{negativeControlsAnyChallenge}/{negativeControlsWithAnyEventCoverage} | "
        "{negativeChallengedAtTwoViews}/{negativeControlsWithTwoViewCoverage} | "
        "{negativeChallengedAndSupportedAtTwoViews} |".format(**row)
        for row in policy_results
    )
    markdown = f"""# Held-out challenge policy replay

This is a read-only Quest replay. Training and held-out depth-pair overlap: 0.

| policy | positive caught | surface any challenge / covered | surface >=2 views / covered | challenged and also supported |
|---|---:|---:|---:|---:|
{lines}

Unknown stable candidates excluded: {report['controls']['unknownExcluded']}.

Read `heldout_policy_summary.json` for the exact split, calibration and decision boundaries.
"""
    (output / "heldout_policy_report.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
