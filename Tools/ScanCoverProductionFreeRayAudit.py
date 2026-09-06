#!/usr/bin/env python3
"""Replay reliable free-space opportunities for production GunGel candidates.

Targets are active stable production candidates that the conservative offline
reference court labelled ``confirmed_free``.  For the same deterministic depth
sample used by the court, this audit reproduces the production challenge gates:
candidate lifetime, fusion admission, motion, central view, raw/post agreement,
free margin, 40x40 observation rays, and the 0.35-cell ray distance.

The result locates where a reliable retrospective free-space witness stopped;
it does not alter runtime policy or write anything back to the device.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
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
    run_adb,
)
from ScanCoverOfflineReferenceCourt import (
    OBSERVATION_DTYPE,
    apply_matrix,
    project_and_sample_ranges,
)
from ScanCoverWitnessConversionAudit import CORRESPONDENCE_DTYPE


DUAL_AGREE_BASE = 0.025
DUAL_AGREE_RANGE_SCALE = 0.010
CHALLENGE_MOTION_MIN = 0.70
FREE_SPACE_MARGIN_BASE = 0.060
FREE_SPACE_MARGIN_RANGE_SCALE = 0.015
CELL_SIZE_METRES = 0.10
RAY_DISTANCE_METRES = CELL_SIZE_METRES * 0.35


def integer(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def device_manifest(adb: Path, serial: str, session: str) -> dict[int, dict[str, str]]:
    text = run_adb(
        adb,
        serial,
        "exec-out",
        f'cat "{session}/fusion_inputs/manifest.csv"',
    ).decode("utf-8-sig", "replace")
    rows = list(csv.DictReader(io.StringIO(text)))
    accepted: dict[int, dict[str, str]] = {}
    for row in rows:
        if integer(row.get("accepted")) != 1 or row.get("status") != "ok":
            continue
        accepted[integer(row.get("sourceFrame"))] = row
    return accepted


def fetch_observations(
    adb: Path, serial: str, session: str, sequence: int
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    prefix = session.rstrip("/") + f"/fusion_inputs/frames/fusion_{sequence:06d}"
    meta = json.loads(
        run_adb(adb, serial, "exec-out", f'cat "{prefix}_meta.json"').decode(
            "utf-8", "replace"
        )
    )
    count = integer(meta.get("gunGelAdmission", {}).get("observationCount"), 1600)
    payload = run_adb(
        adb,
        serial,
        "exec-out",
        "cat "
        + " ".join(
            f'"{path}"'
            for path in (
                prefix + "_gungel_observations.bin",
                prefix + "_gungel_correspondences.bin",
                prefix + "_gungel_correspondence_identity.bin",
            )
        ),
    )
    observation_bytes = count * OBSERVATION_DTYPE.itemsize
    correspondence_bytes = count * CORRESPONDENCE_DTYPE.itemsize
    identity_bytes = count * 16
    expected = observation_bytes + correspondence_bytes + identity_bytes
    if len(payload) != expected:
        raise RuntimeError(
            f"fusion_{sequence:06d} observation bytes expected {expected}, got {len(payload)}"
        )
    observations = np.frombuffer(
        payload[:observation_bytes], dtype=OBSERVATION_DTYPE, count=count
    ).copy()
    correspondences = np.frombuffer(
        payload[observation_bytes : observation_bytes + correspondence_bytes],
        dtype=CORRESPONDENCE_DTYPE,
        count=count,
    ).copy()
    identities = np.frombuffer(
        payload[observation_bytes + correspondence_bytes :], dtype="<u4", count=count * 4
    ).reshape(count, 4).copy()
    return meta, observations, correspondences, identities


def camera_from_meta(meta: dict) -> np.ndarray:
    eye = integer(meta.get("recordedEyeIndex"), 1)
    view_inverse = np.asarray(meta["viewInverse"][eye], dtype=np.float64).reshape(4, 4)
    camera = view_inverse[:3, 3][None, :].astype(np.float32)
    return apply_matrix(camera, meta["fusionCorrection"])[0].astype(np.float64)


def challenge_geometry(
    meta: dict, observations: np.ndarray, candidate: np.ndarray
) -> dict[str, object]:
    correction = meta["fusionCorrection"]
    processed = apply_matrix(observations["positionSigma"][:, :3], correction).astype(
        np.float64
    )
    raw = apply_matrix(observations["rawPositionDelta"][:, :3], correction).astype(
        np.float64
    )
    camera = camera_from_meta(meta)
    valid = observations["positionSigma"][:, 3] > 0.0
    raw_available = observations["rawPositionDelta"][:, 3] >= 0.0
    processed_range = np.linalg.norm(processed - camera[None, :], axis=1)
    dual_limit = DUAL_AGREE_BASE + processed_range * DUAL_AGREE_RANGE_SCALE
    dual = raw_available & (
        observations["rawPositionDelta"][:, 3].astype(np.float64) <= dual_limit
    )
    observed = (processed + raw) * 0.5
    ray = observed - camera[None, :]
    observed_range = np.linalg.norm(ray, axis=1)
    ray_direction = ray / np.maximum(observed_range[:, None], 1e-12)
    free_margin = FREE_SPACE_MARGIN_BASE + observed_range * FREE_SPACE_MARGIN_RANGE_SCALE
    trace_end = observed_range - free_margin
    to_candidate = candidate[None, :] - camera[None, :]
    candidate_along = np.sum(to_candidate * ray_direction, axis=1)
    ray_distance = np.linalg.norm(
        to_candidate - ray_direction * candidate_along[:, None], axis=1
    )
    indices = np.arange(len(observations), dtype=np.int32)
    grid_x = indices % 40
    grid_y = indices // 40
    central = (grid_x >= 5) & (grid_x <= 34) & (grid_y >= 5) & (grid_y <= 34)
    ray_reaches_beyond = (
        valid
        & (observed_range > 0.20)
        & (trace_end > 0.15)
        & (candidate_along > 0.15)
        & (candidate_along < trace_end)
    )
    geometric_hit = ray_reaches_beyond & (ray_distance <= RAY_DISTANCE_METRES)
    all_gates = geometric_hit & raw_available & dual & central
    motion_safe = number(meta.get("motionQuality")) >= CHALLENGE_MOTION_MIN
    if not motion_safe:
        all_gates &= False
    qualifying = np.flatnonzero(all_gates)
    central_hits = np.flatnonzero(geometric_hit & central)
    dual_hits = np.flatnonzero(geometric_hit & central & raw_available & dual)
    nearest_ray_mm = (
        float(np.min(ray_distance[ray_reaches_beyond]) * 1000.0)
        if np.any(ray_reaches_beyond)
        else float("nan")
    )
    maximum_gap_mm = (
        float(np.max(observed_range[geometric_hit] - candidate_along[geometric_hit]) * 1000.0)
        if np.any(geometric_hit)
        else 0.0
    )
    if not motion_safe:
        stage = "blocked_motion_quality"
    elif not np.any(ray_reaches_beyond):
        stage = "blocked_production_free_margin_or_grid_depth"
    elif not np.any(geometric_hit):
        stage = "blocked_40x40_ray_footprint"
    elif not np.any(geometric_hit & central):
        stage = "blocked_outer_deletion_zone"
    elif not np.any(geometric_hit & central & raw_available):
        stage = "blocked_grid_raw_missing"
    elif not np.any(geometric_hit & central & raw_available & dual):
        stage = "blocked_grid_dual_disagree"
    else:
        stage = "challenge_should_reach_candidate"
    view_direction = camera - candidate
    view_direction /= max(float(np.linalg.norm(view_direction)), 1e-12)
    return {
        "stage": stage,
        "motionQuality": number(meta.get("motionQuality")),
        "rayReachesBeyondCount": int(np.count_nonzero(ray_reaches_beyond)),
        "geometricHitCount": int(np.count_nonzero(geometric_hit)),
        "centralHitCount": int(len(central_hits)),
        "dualCentralHitCount": int(len(dual_hits)),
        "qualifyingChallengeRayCount": int(len(qualifying)),
        "nearestRayDistanceMm": nearest_ray_mm,
        "maximumGridRayGapMm": maximum_gap_mm,
        "cameraX": float(camera[0]),
        "cameraY": float(camera[1]),
        "cameraZ": float(camera[2]),
        "candidateX": float(candidate[0]),
        "candidateY": float(candidate[1]),
        "candidateZ": float(candidate[2]),
        "candidateViewX": float(view_direction[0]),
        "candidateViewY": float(view_direction[1]),
        "candidateViewZ": float(view_direction[2]),
    }


def independent_event_views(events: list[dict[str, object]]) -> int:
    witnesses: list[tuple[np.ndarray, np.ndarray, int]] = []
    for event in sorted(events, key=lambda row: integer(row.get("gunGelFrame"))):
        if event.get("cameraX", "") == "":
            continue
        camera = np.asarray(
            [number(event.get("cameraX")), number(event.get("cameraY")), number(event.get("cameraZ"))]
        )
        direction = np.asarray(
            [
                number(event.get("candidateViewX")),
                number(event.get("candidateViewY")),
                number(event.get("candidateViewZ")),
            ]
        )
        frame = integer(event.get("gunGelFrame"))
        correlated = False
        for old_camera, old_direction, old_frame in witnesses:
            cosine = float(np.clip(np.dot(direction, old_direction), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cosine)))
            if (
                abs(frame - old_frame) < 2
                or (
                    float(np.linalg.norm(camera - old_camera)) < 0.08
                    and angle < 3.0
                )
            ):
                correlated = True
                break
        if not correlated:
            witnesses.append((camera, direction, frame))
    return len(witnesses)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb", type=Path, default=DEFAULT_ADB)
    parser.add_argument("--serial", default="2G97C5ZH4501R5")
    parser.add_argument("--session", required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--identity-timeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-still", type=int, default=160)
    parser.add_argument("--sample-slow", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    references = read_csv(args.reference_csv)
    targets = [
        row
        for row in references
        if row.get("state") == "stable"
        and row.get("referenceStatus") == "confirmed_free"
    ]
    stable_ids = [integer(row.get("stable_id")) for row in targets]
    centers = np.asarray(
        [
            [number(row.get("center_x_m")), number(row.get("center_y_m")), number(row.get("center_z_m"))]
            for row in targets
        ],
        dtype=np.float32,
    )
    normals = np.asarray(
        [
            [number(row.get("normal_x")), number(row.get("normal_y")), number(row.get("normal_z"))]
            for row in targets
        ],
        dtype=np.float32,
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    promotions = {
        integer(row.get("stable_id")): row
        for row in read_csv(args.candidate_audit / "promotions.csv")
    }
    identity_rows = read_csv(args.identity_timeline)
    identity_opposition = {
        (integer(row.get("sequence")), integer(row.get("stableId"))): integer(
            row.get("maxOppositionVotes")
        )
        for row in identity_rows
    }
    accepted_fusion = device_manifest(args.adb, args.serial, args.session.rstrip("/"))
    depth_records = load_metadata(args.adb, args.serial, args.session.rstrip("/"))
    groups = annotate_motion(depth_records)
    selected = even_sample(groups["still"], args.sample_still)
    selected += even_sample(groups["slow"], args.sample_slow)
    selected.sort(key=lambda item: integer(item.get("pairIndex")))
    summary_json = json.loads(
        (args.reference_csv.parent / "reference_surface_summary.json").read_text(
            encoding="utf-8"
        )
    )
    nominal = next(
        profile for profile in summary_json["errorProfiles"] if profile["name"] == "nominal"
    )
    agreement_metres = number(nominal["rawPostAgreementMm"]) / 1000.0
    free_gap_metres = number(nominal["freeGapMm"]) / 1000.0
    print(
        f"Replaying {len(selected)} depth pairs for {len(targets)} production stable targets",
        flush=True,
    )

    events: list[dict[str, object]] = []
    observation_cache: dict[
        int, tuple[dict, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    frame_count = 0
    for start in range(0, len(selected), args.batch_size):
        batch = fetch_pair_batch(
            args.adb,
            args.serial,
            args.session.rstrip("/"),
            selected[start : start + args.batch_size],
        )
        for record, raw, post in batch:
            sampled = project_and_sample_ranges(
                record, raw, post, centers, normals, 0.90, 0.15
            )
            valid = np.isfinite(sampled["rawRange"]) & np.isfinite(sampled["postRange"])
            if np.any(valid):
                raw_range = sampled["rawRange"][valid]
                post_range = sampled["postRange"][valid]
                candidate_range = sampled["candidateRange"][valid]
                indices = sampled["indices"][valid]
                free = (
                    (np.abs(raw_range - post_range) <= agreement_metres)
                    & (np.minimum(raw_range, post_range) > candidate_range + free_gap_metres)
                )
                platform_frame = integer(record.get("platformFrame"))
                fusion_row = accepted_fusion.get(platform_frame)
                for local_index in np.flatnonzero(free):
                    target_index = int(indices[local_index])
                    stable_id = stable_ids[target_index]
                    promotion = promotions.get(stable_id, {})
                    offline_gap_mm = float(
                        (min(raw_range[local_index], post_range[local_index]) - candidate_range[local_index])
                        * 1000.0
                    )
                    base = {
                        "stableId": stable_id,
                        "pairIndex": integer(record.get("pairIndex")),
                        "platformFrame": platform_frame,
                        "motionClass": record.get("_motion", ""),
                        "offlineFreeGapMm": offline_gap_mm,
                        "candidateBirthFrame": promotion.get("birth_frame", ""),
                        "candidatePromotionFrame": promotion.get("promotion_frame", ""),
                        "candidateFinalLastSafeFrame": targets[target_index].get(
                            "last_seen_frame", ""
                        ),
                    }
                    if fusion_row is None:
                        events.append({**base, "sequence": "", "gunGelFrame": "", "stage": "blocked_no_accepted_fusion_frame"})
                        continue
                    sequence = integer(fusion_row.get("sequence"))
                    gun_gel_frame = integer(fusion_row.get("gunGelFrame"))
                    if gun_gel_frame < integer(promotion.get("birth_frame")):
                        events.append({**base, "sequence": sequence, "gunGelFrame": gun_gel_frame, "stage": "candidate_not_born_yet"})
                        continue
                    if sequence not in observation_cache:
                        observation_cache[sequence] = fetch_observations(
                            args.adb, args.serial, args.session.rstrip("/"), sequence
                        )
                    meta, observations, correspondences, identities = observation_cache[
                        sequence
                    ]
                    identity_mask = identities[:, 1] == np.uint32(stable_id)
                    target_positions = correspondences["targetSigma"][:, :3]
                    target_valid = identity_mask & np.isfinite(target_positions).all(axis=1) & (
                        np.linalg.norm(target_positions, axis=1) > 0.10
                    )
                    if np.any(target_valid):
                        candidate_center = np.median(
                            target_positions[target_valid].astype(np.float64), axis=0
                        )
                        center_source = "frame_final_correspondence"
                    elif gun_gel_frame >= integer(
                        targets[target_index].get("last_seen_frame")
                    ):
                        candidate_center = centers[target_index].astype(np.float64)
                        center_source = "final_snapshot_after_last_safe"
                    else:
                        events.append(
                            {
                                **base,
                                "sequence": sequence,
                                "gunGelFrame": gun_gel_frame,
                                "stage": "candidate_geometry_unresolved_for_frame",
                                "candidateCenterSource": "none",
                            }
                        )
                        continue
                    geometry = challenge_geometry(
                        meta, observations, candidate_center
                    )
                    events.append(
                        {
                            **base,
                            "sequence": sequence,
                            "gunGelFrame": gun_gel_frame,
                            "candidateCenterSource": center_source,
                            **geometry,
                            "identityOppositionVotes": identity_opposition.get(
                                (sequence, stable_id), ""
                            ),
                        }
                    )
            frame_count += 1
        if frame_count % 40 < len(batch) or frame_count == len(selected):
            print(
                f"  depth {frame_count}/{len(selected)}, free events {len(events)}, "
                f"fusion observations cached {len(observation_cache)}",
                flush=True,
            )

    by_target: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    for event in events:
        by_target[integer(event.get("stableId"))].append(event)
    target_summaries: list[dict[str, object]] = []
    overall_stages: Counter[str] = Counter()
    for stable_id, source in zip(stable_ids, targets):
        candidate_events = by_target.get(stable_id, [])
        stages = Counter(str(event.get("stage", "")) for event in candidate_events)
        overall_stages.update(stages)
        expected = stages.get("challenge_should_reach_candidate", 0)
        before_birth = stages.get("candidate_not_born_yet", 0)
        unresolved = stages.get("candidate_geometry_unresolved_for_frame", 0)
        no_fusion = stages.get("blocked_no_accepted_fusion_frame", 0)
        visible_votes = sum(
            integer(event.get("identityOppositionVotes")) > 0 for event in candidate_events
        )
        last_safe_frame = integer(source.get("last_seen_frame"))
        post_last_safe = [
            event
            for event in candidate_events
            if integer(event.get("gunGelFrame"), -1) >= last_safe_frame
            and event.get("cameraX", "") != ""
        ]
        post_last_safe_views = independent_event_views(post_last_safe)
        if candidate_events and before_birth * 2 >= len(candidate_events):
            conclusion = "candidate_born_after_confirmed_free_history"
        elif candidate_events and unresolved == len(candidate_events):
            conclusion = "candidate_geometry_timeline_not_recorded_at_free_views"
        elif expected > 0 and visible_votes == 0:
            conclusion = "challenge_geometry_passed_but_vote_not_visible"
        elif stages.get("blocked_40x40_ray_footprint", 0) > 0 and expected == 0:
            conclusion = "observation_grid_footprint_is_primary_blocker"
        elif stages.get("blocked_production_free_margin_or_grid_depth", 0) > 0 and expected == 0:
            conclusion = "production_free_margin_or_grid_depth_is_primary_blocker"
        elif candidate_events and no_fusion * 2 >= len(candidate_events):
            conclusion = "fusion_admission_is_primary_blocker"
        else:
            conclusion = "mixed_or_insufficient_replayed_opportunities"
        target_summaries.append(
            {
                "stableId": stable_id,
                "conclusion": conclusion,
                "offlineNominalFreeViews": source.get("nominalFreeViews", ""),
                "offlineFreeEvents": len(candidate_events),
                "expectedChallengeEvents": expected,
                "eventsWithVisibleIdentityOpposition": visible_votes,
                "postLastSafeFreeEvents": len(post_last_safe),
                "postLastSafeIndependentFreeViews": post_last_safe_views,
                "currentFalseSurfaceConfirmed": int(post_last_safe_views >= 2),
                **{f"stage_{key}": value for key, value in sorted(stages.items())},
            }
        )

    event_fields = sorted({key for row in events for key in row}) if events else ["stableId"]
    summary_fields = sorted({key for row in target_summaries for key in row})
    write_csv(output / "production_free_ray_events.csv", events, event_fields)
    write_csv(output / "production_free_ray_candidates.csv", target_summaries, summary_fields)
    report = {
        "schema": "scancover.production-free-ray-audit.v1",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "authority": "read_only_offline_diagnostic",
        "productionConsumers": 0,
        "targetCandidates": len(targets),
        "selectedDepthPairs": len(selected),
        "offlineFreeEvents": len(events),
        "acceptedFusionObservationFramesLoaded": len(observation_cache),
        "stageCounts": dict(overall_stages),
        "candidateConclusionCounts": dict(
            Counter(row["conclusion"] for row in target_summaries)
        ),
        "productionChallengeContract": {
            "motionQualityMinimum": CHALLENGE_MOTION_MIN,
            "centralGrid": "40x40 cells with gx and gy in [5,34]",
            "dualAgreement": "25mm + 1 percent of range",
            "freeMargin": "60mm + 1.5 percent of observed range",
            "rayDistance": "35mm from candidate center",
        },
        "boundary": [
            "only the same 380 still/slow pairs used by the reference court are replayed",
            "a missing identity opposition value can mean the candidate had no matched observation in that frame",
            "an expected challenge isolates the remaining fault to candidate presence, spatial hash search, dispatch, or vote visibility",
        ],
    }
    (output / "production_free_ray_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
