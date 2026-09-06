#!/usr/bin/env python3
"""Build a conservative, session-local surface reference court from Quest evidence.

This is an offline diagnostic only.  It streams paired depth from the device,
cross-examines final virtual-probe candidates against independent viewpoints,
and labels each candidate as one of:

    confirmed_surface  repeated depth agrees that a surface is here
    confirmed_free     repeated depth proves the ray was free here
    unknown            evidence is absent, occluded, conflicting, or sensitive

The tool deliberately runs three error profiles.  A positive label is emitted
only when all three profiles agree.  Coordinates, IDs, and labels remain local
to the captured session and are never written back into Unity or production.

It also samples the recorded pre/post candidate-transaction identity buffers so
that identity churn can be joined to the reference court by world position.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import subprocess
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

from ScanCoverDepthPairDeviceAudit import (
    DEFAULT_ADB,
    annotate_motion,
    even_sample,
    fetch_pair_batch,
    linearize,
    load_metadata,
    recorded_eye,
    run_adb,
)


OBSERVATION_DTYPE = np.dtype(
    [
        ("positionSigma", "<f4", (4,)),
        ("normalQuality", "<f4", (4,)),
        ("rawPositionDelta", "<f4", (4,)),
        ("sourceReason", "<u4", (4,)),
    ],
    align=False,
)

IDENTITY_DTYPE = np.dtype("<u4")


@dataclass(frozen=True)
class ErrorProfile:
    name: str
    support_metres: float
    agreement_metres: float
    free_gap_metres: float


def read_device_text(adb: Path, serial: str, path: str) -> str:
    return run_adb(adb, serial, "exec-out", f'cat "{path}"').decode(
        "utf-8-sig", "replace"
    )


def read_device_csv(adb: Path, serial: str, path: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(read_device_text(adb, serial, path))))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_number(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def integer(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def candidate_number(row: dict[str, str], camel: str, snake: str) -> float:
    return finite_number(row.get(camel, row.get(snake)))


def candidate_identifier(row: dict[str, object]) -> object:
    return row.get("fingerprintId", row.get("stable_id", row.get("stableId", "")))


def camera_state(record: dict) -> tuple[np.ndarray, np.ndarray]:
    eye = recorded_eye(record)
    transform = np.asarray(record["viewInverse"][eye], dtype=np.float64).reshape(4, 4)
    position = transform[:3, 3].copy()
    forward = transform[:3, 2].copy()
    norm = float(np.linalg.norm(forward))
    if norm > 1e-12:
        forward /= norm
    return position, forward


def direction_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    dot = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def cluster_independent_views(
    records: list[dict], baseline_metres: float, angle_deg: float
) -> tuple[list[int], list[dict[str, object]]]:
    representatives: list[tuple[np.ndarray, np.ndarray]] = []
    assignments: list[int] = []
    counts: Counter[int] = Counter()
    for record in records:
        position, forward = camera_state(record)
        assigned = -1
        for index, (rep_position, rep_forward) in enumerate(representatives):
            if (
                float(np.linalg.norm(position - rep_position)) < baseline_metres
                and direction_angle_deg(forward, rep_forward) < angle_deg
            ):
                assigned = index
                break
        if assigned < 0:
            assigned = len(representatives)
            representatives.append((position, forward))
        assignments.append(assigned)
        counts[assigned] += 1
    summary = [
        {
            "viewId": index,
            "sampledFrames": counts[index],
            "cameraX": float(position[0]),
            "cameraY": float(position[1]),
            "cameraZ": float(position[2]),
            "forwardX": float(forward[0]),
            "forwardY": float(forward[1]),
            "forwardZ": float(forward[2]),
        }
        for index, (position, forward) in enumerate(representatives)
    ]
    return assignments, summary


def calibrate_qrs_p95(record: dict, raw: np.ndarray, post: np.ndarray) -> np.ndarray:
    raw_sample = raw[::16]
    post_sample = post[::16]
    common = (
        np.isfinite(raw_sample)
        & np.isfinite(post_sample)
        & (raw_sample > 0.0)
        & (raw_sample < 1.0)
        & (post_sample > 0.0)
        & (post_sample < 1.0)
    )
    if not np.any(common):
        return np.empty(0, dtype=np.float32)
    eye = recorded_eye(record)
    raw_m = linearize(raw_sample[common], record["projection"][eye])
    post_m = linearize(post_sample[common], record["projection"][eye])
    valid = np.isfinite(raw_m) & np.isfinite(post_m) & (raw_m < 8.0) & (post_m < 8.0)
    return np.abs(raw_m[valid] - post_m[valid]).astype(np.float32)


def profiles_from_calibration(qrs_p95_metres: float) -> list[ErrorProfile]:
    base = float(np.clip(qrs_p95_metres, 0.006, 0.020))
    return [
        ErrorProfile(
            "strict",
            max(0.010, 1.25 * base),
            max(0.012, 1.50 * base),
            max(0.030, 3.00 * base),
        ),
        ErrorProfile(
            "nominal",
            max(0.015, 1.75 * base),
            max(0.018, 2.00 * base),
            max(0.035, 3.50 * base),
        ),
        ErrorProfile(
            "permissive",
            max(0.020, 2.50 * base),
            max(0.025, 3.00 * base),
            max(0.045, 4.00 * base),
        ),
    ]


def project_and_sample_ranges(
    record: dict,
    raw: np.ndarray,
    post: np.ndarray,
    centers: np.ndarray,
    normals: np.ndarray,
    outer_radius: float,
    minimum_incidence: float,
) -> dict[str, np.ndarray]:
    eye = recorded_eye(record)
    width = int(record["width"])
    height = int(record["height"])
    projection = np.asarray(record["projection"][eye], dtype=np.float64).reshape(4, 4)
    view = np.asarray(record["view"][eye], dtype=np.float64).reshape(4, 4)
    projection_inverse = np.asarray(
        record["projectionInverse"][eye], dtype=np.float64
    ).reshape(4, 4)
    view_inverse = np.asarray(record["viewInverse"][eye], dtype=np.float64).reshape(4, 4)
    eye_position = view_inverse[:3, 3]

    world_h = np.column_stack((centers.astype(np.float64), np.ones(len(centers))))
    clip = (projection @ (view @ world_h.T)).T
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc = clip[:, :3] / clip[:, 3:4]
    uv = (ndc[:, :2] + 1.0) * 0.5
    view_radius = np.max(np.abs(ndc[:, :2]), axis=1)
    to_candidate = centers.astype(np.float64) - eye_position[None, :]
    candidate_range = np.linalg.norm(to_candidate, axis=1)
    ray_direction = to_candidate / np.maximum(candidate_range[:, None], 1e-12)
    incidence = np.abs(np.sum(normals.astype(np.float64) * ray_direction, axis=1))
    visible = (
        np.isfinite(ndc).all(axis=1)
        & (clip[:, 3] > 0.0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < 1.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < 1.0)
        & (view_radius <= outer_radius)
        & (incidence >= minimum_incidence)
        & (candidate_range >= 0.15)
        & (candidate_range <= 8.0)
    )
    indices = np.flatnonzero(visible)
    if not len(indices):
        return {
            "indices": indices,
            "rawRange": np.empty(0),
            "postRange": np.empty(0),
            "candidateRange": np.empty(0),
            "viewRadius": np.empty(0),
            "incidence": np.empty(0),
            "commonSamples": np.empty(0, dtype=np.int32),
        }

    x_center = np.rint(uv[indices, 0] * width).astype(np.int64)
    y_center = np.rint(uv[indices, 1] * height).astype(np.int64)
    offsets = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    raw_ranges = np.full((len(indices), len(offsets)), np.nan, dtype=np.float64)
    post_ranges = np.full_like(raw_ranges, np.nan)
    raw_image = raw.reshape(height, width)
    post_image = post.reshape(height, width)

    for column, (dx, dy) in enumerate(offsets):
        xs = np.clip(x_center + dx, 0, width - 1)
        ys = np.clip(y_center + dy, 0, height - 1)
        raw_depth = raw_image[ys, xs].astype(np.float64)
        post_depth = post_image[ys, xs].astype(np.float64)
        common = (
            np.isfinite(raw_depth)
            & np.isfinite(post_depth)
            & (raw_depth > 0.0)
            & (raw_depth < 1.0)
            & (post_depth > 0.0)
            & (post_depth < 1.0)
        )
        if not np.any(common):
            continue
        u = xs[common].astype(np.float64) / width
        v = ys[common].astype(np.float64) / height
        for depth, destination in ((raw_depth, raw_ranges), (post_depth, post_ranges)):
            hcs = np.stack(
                (
                    u * 2.0 - 1.0,
                    v * 2.0 - 1.0,
                    depth[common] * 2.0 - 1.0,
                    np.ones(np.count_nonzero(common), dtype=np.float64),
                )
            )
            reconstructed_h = view_inverse @ (projection_inverse @ hcs)
            reconstructed = (reconstructed_h[:3] / reconstructed_h[3]).T
            destination[common, column] = np.linalg.norm(
                reconstructed - eye_position[None, :], axis=1
            )

    common_samples = np.sum(np.isfinite(raw_ranges) & np.isfinite(post_ranges), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        raw_median = np.nanmedian(raw_ranges, axis=1)
        post_median = np.nanmedian(post_ranges, axis=1)
    enough = common_samples >= 3
    raw_median[~enough] = np.nan
    post_median[~enough] = np.nan
    return {
        "indices": indices,
        "rawRange": raw_median,
        "postRange": post_median,
        "candidateRange": candidate_range[indices],
        "viewRadius": view_radius[indices],
        "incidence": incidence[indices],
        "commonSamples": common_samples,
    }


def set_view_bits(mask: np.ndarray, candidate_indices: np.ndarray, view_id: int) -> None:
    if not len(candidate_indices):
        return
    chunk = view_id // 64
    bit = np.uint64(1) << np.uint64(view_id % 64)
    mask[candidate_indices, chunk] |= bit


def bit_counts(mask: np.ndarray) -> np.ndarray:
    byte_view = np.ascontiguousarray(mask).view(np.uint8).reshape(mask.shape[0], -1)
    return np.unpackbits(byte_view, axis=1).sum(axis=1).astype(np.int32)


def profile_status(support_views: int, free_views: int) -> str:
    if support_views >= 3 and free_views == 0:
        return "confirmed_surface"
    if free_views >= 2 and support_views == 0:
        return "confirmed_free"
    return "unknown"


def unknown_reason(
    profile_statuses: list[str], observed_views: int, nominal_support: int,
    nominal_free: int, nominal_occluded: int,
) -> str:
    if len(set(profile_statuses)) > 1:
        return "threshold_sensitive"
    if nominal_support > 0 and nominal_free > 0:
        return "support_free_conflict"
    if observed_views == 0:
        return "not_observed"
    if nominal_occluded > 0 and nominal_support == 0 and nominal_free == 0:
        return "occluded_only"
    return "insufficient_independent_evidence"


def apply_matrix(points: np.ndarray, matrix_values: list[float]) -> np.ndarray:
    matrix = np.asarray(matrix_values, dtype=np.float64).reshape(4, 4)
    homogeneous = np.column_stack((points.astype(np.float64), np.ones(len(points))))
    transformed_h = (matrix @ homogeneous.T).T
    return (transformed_h[:, :3] / transformed_h[:, 3:4]).astype(np.float32)


def transition_name(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    names = np.full(len(pre), "unchanged", dtype=object)
    pre_present = pre[:, 1] != 0
    post_present = post[:, 1] != 0
    names[~pre_present & post_present] = "born"
    names[pre_present & ~post_present] = "removed"
    reassigned = pre_present & post_present & (pre[:, 1] != post[:, 1])
    names[reassigned] = "reassigned_stable_id"
    reindexed = (
        pre_present
        & post_present
        & (pre[:, 1] == post[:, 1])
        & (pre[:, 0] != post[:, 0])
    )
    names[reindexed] = "reindexed_same_stable_id"
    evidence = (
        pre_present
        & post_present
        & (pre[:, 1] == post[:, 1])
        & (pre[:, 0] == post[:, 0])
        & np.any(pre[:, 2:4] != post[:, 2:4], axis=1)
    )
    names[evidence] = "evidence_changed"
    return names


def audit_identity_transitions(
    adb: Path,
    serial: str,
    session: str,
    sample_count: int,
    join_radius_metres: float,
    centers: np.ndarray,
    reference_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    frames_root = session + "/fusion_inputs/frames"
    listing = run_adb(
        adb,
        serial,
        "shell",
        f"ls -1 {frames_root}/*_gungel_pre_transaction_identity.bin 2>/dev/null",
    ).decode("utf-8", "replace")
    sequences = []
    for line in listing.splitlines():
        stem = Path(line.strip()).name
        try:
            sequences.append(int(stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    sequences = sorted(set(sequences))
    if not sequences:
        return [], {"sampledFrames": 0, "reason": "no_pre_transaction_identity_files"}
    selected = [sequences[int(index)] for index in np.linspace(
        0, len(sequences) - 1, min(sample_count, len(sequences)), dtype=np.int64
    )]
    tree = cKDTree(centers)
    events: list[dict[str, object]] = []
    aggregate: Counter[str] = Counter()
    by_reference: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for ordinal, sequence in enumerate(selected, 1):
        prefix = f"{frames_root}/fusion_{sequence:06d}"
        meta = json.loads(read_device_text(adb, serial, prefix + "_meta.json"))
        payload = run_adb(
            adb,
            serial,
            "exec-out",
            "cat "
            + " ".join(
                f'"{path}"'
                for path in (
                    prefix + "_gungel_observations.bin",
                    prefix + "_gungel_pre_transaction_identity.bin",
                    prefix + "_gungel_correspondence_identity.bin",
                )
            ),
        )
        count = integer(meta.get("gunGelAdmission", {}).get("observationCount"), 1600)
        observation_bytes = count * OBSERVATION_DTYPE.itemsize
        identity_bytes = count * 16
        expected = observation_bytes + identity_bytes * 2
        if len(payload) != expected:
            raise RuntimeError(
                f"fusion_{sequence:06d}: expected {expected} bytes, got {len(payload)}"
            )
        observations = np.frombuffer(
            payload[:observation_bytes], dtype=OBSERVATION_DTYPE, count=count
        )
        pre = np.frombuffer(
            payload[observation_bytes : observation_bytes + identity_bytes],
            dtype=IDENTITY_DTYPE,
        ).reshape(count, 4)
        post = np.frombuffer(
            payload[observation_bytes + identity_bytes :], dtype=IDENTITY_DTYPE
        ).reshape(count, 4)
        names = transition_name(pre, post)
        changed = np.flatnonzero(names != "unchanged")
        aggregate.update(names.tolist())
        if not len(changed):
            continue
        positions = observations["positionSigma"][:, :3]
        positions = apply_matrix(positions, meta["fusionCorrection"])
        distances, nearest = tree.query(positions[changed], k=1)
        for local_index, observation_index in enumerate(changed):
            candidate_index = int(nearest[local_index])
            distance = float(distances[local_index])
            if distance <= join_radius_metres:
                reference = str(reference_rows[candidate_index]["referenceStatus"])
                fingerprint = candidate_identifier(reference_rows[candidate_index])
            else:
                reference = "unjoined"
                fingerprint = ""
            transition = str(names[observation_index])
            by_reference[reference][transition] += 1
            position = positions[observation_index]
            events.append(
                {
                    "sequence": sequence,
                    "sourceFrame": integer(meta.get("sourceFrame")),
                    "observationIndex": int(observation_index),
                    "transition": transition,
                    "preCandidateIndexPlusOne": int(pre[observation_index, 0]),
                    "preStableId": int(pre[observation_index, 1]),
                    "preDualSupportFrames": int(pre[observation_index, 2]),
                    "preOppositionVotes": int(pre[observation_index, 3]),
                    "postCandidateIndexPlusOne": int(post[observation_index, 0]),
                    "postStableId": int(post[observation_index, 1]),
                    "postDualSupportFrames": int(post[observation_index, 2]),
                    "postOppositionVotes": int(post[observation_index, 3]),
                    "sourceX": float(position[0]),
                    "sourceY": float(position[1]),
                    "sourceZ": float(position[2]),
                    "nearestFingerprintId": fingerprint,
                    "nearestDistanceMm": distance * 1000.0,
                    "referenceStatus": reference,
                }
            )
        if ordinal % 20 == 0 or ordinal == len(selected):
            print(
                f"  identity audit {ordinal}/{len(selected)} frames, "
                f"changed events {len(events)}",
                flush=True,
            )

    return events, {
        "availableFrames": len(sequences),
        "sampledFrames": len(selected),
        "transitionCounts": dict(aggregate),
        "changedJoinCounts": {
            reference: dict(counter) for reference, counter in sorted(by_reference.items())
        },
        "joinRadiusMm": join_radius_metres * 1000.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb", type=Path, default=DEFAULT_ADB)
    parser.add_argument("--serial", default="2G97C5ZH4501R5")
    parser.add_argument("--session", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=None,
        help="optional local production candidate table; default is device shadow verdict_cells",
    )
    parser.add_argument("--candidate-source", default="probe_shadow_final_buffer")
    parser.add_argument("--sample-still", type=int, default=160)
    parser.add_argument("--sample-slow", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--identity-samples", type=int, default=120)
    parser.add_argument("--identity-join-mm", type=float, default=25.0)
    parser.add_argument("--outer-radius", type=float, default=0.90)
    parser.add_argument("--minimum-incidence", type=float, default=0.15)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    session = args.session.rstrip("/")
    print(f"Reference court session: {session}", flush=True)
    print(f"Output: {output}", flush=True)

    manifest_path = session + "/fusion_inputs/manifest.csv"
    if args.candidate_csv is None:
        candidate_path = session + "/probe_shadow/final_buffer/verdict_cells.csv"
        candidate_text = read_device_text(args.adb, args.serial, candidate_path)
    else:
        candidate_path = str(args.candidate_csv.resolve())
        candidate_text = args.candidate_csv.read_text(encoding="utf-8-sig")
    manifest_text = read_device_text(args.adb, args.serial, manifest_path)
    (output / "source_verdict_cells.csv").write_text(candidate_text, encoding="utf-8")
    (output / "source_fusion_manifest.csv").write_text(manifest_text, encoding="utf-8")
    candidate_rows = list(csv.DictReader(io.StringIO(candidate_text)))
    if not candidate_rows:
        raise RuntimeError("verdict_cells.csv contains no candidates")
    centers = np.asarray(
        [
            [
                candidate_number(row, "centerX", "center_x_m"),
                candidate_number(row, "centerY", "center_y_m"),
                candidate_number(row, "centerZ", "center_z_m"),
            ]
            for row in candidate_rows
        ],
        dtype=np.float32,
    )
    normals = np.asarray(
        [
            [
                candidate_number(row, "normalX", "normal_x"),
                candidate_number(row, "normalY", "normal_y"),
                candidate_number(row, "normalZ", "normal_z"),
            ]
            for row in candidate_rows
        ],
        dtype=np.float32,
    )
    normal_length = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(normal_length[:, None], 1e-12)

    records = load_metadata(args.adb, args.serial, session)
    groups = annotate_motion(records)
    selected = even_sample(groups["still"], args.sample_still)
    selected += even_sample(groups["slow"], args.sample_slow)
    selected.sort(key=lambda item: int(item["pairIndex"]))
    view_ids, view_summary = cluster_independent_views(selected, 0.08, 3.0)
    chunks = max(1, (len(view_summary) + 63) // 64)
    print(
        f"Candidates {len(candidate_rows)}, selected depth pairs {len(selected)}, "
        f"independent view clusters {len(view_summary)}",
        flush=True,
    )

    calibration: list[np.ndarray] = []
    cached_batches: list[list[tuple[dict, np.ndarray, np.ndarray]]] = []
    for start in range(0, len(selected), args.batch_size):
        batch = fetch_pair_batch(
            args.adb, args.serial, session, selected[start : start + args.batch_size]
        )
        cached_batches.append(batch)
        for record, raw, post in batch:
            diff = calibrate_qrs_p95(record, raw, post)
            if len(diff):
                calibration.append(diff)
        if (start // args.batch_size + 1) % 10 == 0 or start + args.batch_size >= len(selected):
            print(
                f"  streamed depth {min(start + args.batch_size, len(selected))}/{len(selected)}",
                flush=True,
            )
    calibration_values = (
        np.concatenate(calibration) if calibration else np.asarray([0.010], dtype=np.float32)
    )
    qrs_p95 = float(np.percentile(calibration_values, 95))
    profiles = profiles_from_calibration(qrs_p95)
    print(
        "QRS p95 %.2f mm; profiles %s"
        % (
            qrs_p95 * 1000.0,
            ", ".join(
                f"{p.name}(surface={p.support_metres*1000:.1f},"
                f"free={p.free_gap_metres*1000:.1f})"
                for p in profiles
            ),
        ),
        flush=True,
    )

    profile_masks = {
        profile.name: {
            "support": np.zeros((len(candidate_rows), chunks), dtype=np.uint64),
            "free": np.zeros((len(candidate_rows), chunks), dtype=np.uint64),
        }
        for profile in profiles
    }
    observed_mask = np.zeros((len(candidate_rows), chunks), dtype=np.uint64)
    occluded_mask = np.zeros((len(candidate_rows), chunks), dtype=np.uint64)

    frame_ordinal = 0
    for batch in cached_batches:
        for record, raw, post in batch:
            sampled = project_and_sample_ranges(
                record,
                raw,
                post,
                centers,
                normals,
                args.outer_radius,
                args.minimum_incidence,
            )
            indices = sampled["indices"]
            valid = np.isfinite(sampled["rawRange"]) & np.isfinite(sampled["postRange"])
            valid_indices = indices[valid]
            view_id = view_ids[frame_ordinal]
            set_view_bits(observed_mask, valid_indices, view_id)
            if len(valid_indices):
                raw_range = sampled["rawRange"][valid]
                post_range = sampled["postRange"][valid]
                candidate_range = sampled["candidateRange"][valid]
                agreement = np.abs(raw_range - post_range)
                residual = np.maximum(
                    np.abs(candidate_range - raw_range),
                    np.abs(candidate_range - post_range),
                )
                nominal = profiles[1]
                nominal_agree = agreement <= nominal.agreement_metres
                occluded = nominal_agree & (
                    np.maximum(raw_range, post_range)
                    < candidate_range - nominal.free_gap_metres
                )
                set_view_bits(occluded_mask, valid_indices[occluded], view_id)
                for profile in profiles:
                    agree = agreement <= profile.agreement_metres
                    support = agree & (residual <= profile.support_metres)
                    free = agree & (
                        np.minimum(raw_range, post_range)
                        > candidate_range + profile.free_gap_metres
                    )
                    set_view_bits(
                        profile_masks[profile.name]["support"], valid_indices[support], view_id
                    )
                    set_view_bits(
                        profile_masks[profile.name]["free"], valid_indices[free], view_id
                    )
            frame_ordinal += 1
        if frame_ordinal % 40 < len(batch) or frame_ordinal == len(selected):
            print(f"  judged depth {frame_ordinal}/{len(selected)}", flush=True)

    observed_counts = bit_counts(observed_mask)
    occluded_counts = bit_counts(occluded_mask)
    support_counts = {
        profile.name: bit_counts(profile_masks[profile.name]["support"])
        for profile in profiles
    }
    free_counts = {
        profile.name: bit_counts(profile_masks[profile.name]["free"])
        for profile in profiles
    }

    reference_rows: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for index, source in enumerate(candidate_rows):
        statuses = [
            profile_status(
                int(support_counts[profile.name][index]),
                int(free_counts[profile.name][index]),
            )
            for profile in profiles
        ]
        if statuses[0] == statuses[1] == statuses[2] and statuses[0] != "unknown":
            reference_status = statuses[0]
            reason = "three_profiles_agree"
        else:
            reference_status = "unknown"
            reason = unknown_reason(
                statuses,
                int(observed_counts[index]),
                int(support_counts["nominal"][index]),
                int(free_counts["nominal"][index]),
                int(occluded_counts[index]),
            )
        status_counts[reference_status] += 1
        reason_counts[reason] += 1
        reference_rows.append(
            {
                **source,
                "referenceStatus": reference_status,
                "referenceReason": reason,
                "observedIndependentViews": int(observed_counts[index]),
                "occludedIndependentViews": int(occluded_counts[index]),
                **{
                    f"{profile.name}SupportViews": int(
                        support_counts[profile.name][index]
                    )
                    for profile in profiles
                },
                **{
                    f"{profile.name}FreeViews": int(free_counts[profile.name][index])
                    for profile in profiles
                },
            }
        )

    reference_fields = list(candidate_rows[0].keys()) + [
        "referenceStatus",
        "referenceReason",
        "observedIndependentViews",
        "occludedIndependentViews",
        "strictSupportViews",
        "nominalSupportViews",
        "permissiveSupportViews",
        "strictFreeViews",
        "nominalFreeViews",
        "permissiveFreeViews",
    ]
    write_csv(output / "reference_surface_candidates.csv", reference_rows, reference_fields)
    write_csv(
        output / "reference_view_clusters.csv",
        view_summary,
        list(view_summary[0].keys()) if view_summary else ["viewId"],
    )

    print("Auditing A -> B identity transitions...", flush=True)
    transition_rows, transition_summary = audit_identity_transitions(
        args.adb,
        args.serial,
        session,
        args.identity_samples,
        args.identity_join_mm / 1000.0,
        centers,
        reference_rows,
    )
    transition_fields = [
        "sequence",
        "sourceFrame",
        "observationIndex",
        "transition",
        "preCandidateIndexPlusOne",
        "preStableId",
        "preDualSupportFrames",
        "preOppositionVotes",
        "postCandidateIndexPlusOne",
        "postStableId",
        "postDualSupportFrames",
        "postOppositionVotes",
        "sourceX",
        "sourceY",
        "sourceZ",
        "nearestFingerprintId",
        "nearestDistanceMm",
        "referenceStatus",
    ]
    write_csv(output / "identity_transition_events.csv", transition_rows, transition_fields)

    source_verdict_counts = Counter(
        row.get("verdict", row.get("state", "")) for row in candidate_rows
    )
    verdict_reference_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in reference_rows:
        verdict_reference_counts[str(row.get("verdict", row.get("state", "")))][
            str(row["referenceStatus"])
        ] += 1
    summary = {
        "schema": "scancover.offline-reference-court.v1",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "authority": "read_only_offline_diagnostic",
        "productionConsumers": 0,
        "session": session,
        "deviceSerial": args.serial,
        "recordedEyeIndices": sorted({recorded_eye(record) for record in records}),
        "completeDepthPairs": len(records),
        "selectedDepthPairs": len(selected),
        "selectedMotion": {
            "still": sum(record["_motion"] == "still" for record in selected),
            "slow": sum(record["_motion"] == "slow" for record in selected),
            "fast": sum(record["_motion"] == "fast" for record in selected),
        },
        "candidateCount": len(candidate_rows),
        "candidateSource": args.candidate_source,
        "candidateSourcePath": candidate_path,
        "sourceVerdictCounts": dict(source_verdict_counts),
        "sourceVerdictByReferenceStatus": {
            verdict: dict(counter)
            for verdict, counter in sorted(verdict_reference_counts.items())
        },
        "referenceStatusCounts": dict(status_counts),
        "referenceReasonCounts": dict(reason_counts),
        "independentViewClusters": len(view_summary),
        "independenceDefinition": {
            "baselineMetres": 0.08,
            "viewAngleDegrees": 3.0,
            "note": "global clusters are a conservative offline approximation",
        },
        "eligibility": {
            "outerViewRadius": args.outer_radius,
            "minimumAbsoluteNormalIncidence": args.minimum_incidence,
            "minimumCommonNeighbourSamples": 3,
            "fastMotionFramesUsed": False,
        },
        "qrsCalibration": {
            "sparseCommonPixels": int(len(calibration_values)),
            "absRangeDifferenceMmP95": qrs_p95 * 1000.0,
        },
        "errorProfiles": [
            {
                "name": profile.name,
                "supportMm": profile.support_metres * 1000.0,
                "rawPostAgreementMm": profile.agreement_metres * 1000.0,
                "freeGapMm": profile.free_gap_metres * 1000.0,
            }
            for profile in profiles
        ],
        "referenceRule": {
            "confirmedSurface": "all profiles: >=3 support views and 0 free views",
            "confirmedFree": "all profiles: >=2 free views and 0 support views",
            "unknown": "all other cases, including conflict and threshold sensitivity",
        },
        "identityTransitions": transition_summary,
        "generalizationBoundary": [
            "world coordinates and IDs are session-local audit joins only",
            "no room coordinate, object identity, or session label is a production feature",
            "unknown is intentionally not converted into empty space or a surface",
            "this report measures evidence; it does not define a runtime replacement policy",
        ],
    }
    (output / "reference_surface_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "README.txt").write_text(
        "ScanCover offline reference court\n\n"
        "This directory is a read-only forensic result. It does not alter Unity, TSDF, "
        "paper, mesh, or runtime thresholds.\n\n"
        "reference_surface_candidates.csv: one final candidate per row with conservative "
        "surface/free/unknown labels.\n"
        "identity_transition_events.csv: changed A->B identity events joined spatially "
        "to the reference court only within the configured conservative join radius.\n"
        "reference_surface_summary.json: rules, thresholds, counts, and limits.\n"
        "source_*.csv: exact small source tables copied from the sealed session.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
