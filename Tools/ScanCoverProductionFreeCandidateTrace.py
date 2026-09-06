#!/usr/bin/env python3
"""Trace production GunGel candidates that a reference court marks as free.

The input reference CSV must have been built from the authoritative exported
``artifacts/gungel_candidate_audit/candidates.csv`` snapshot, not from the
virtual-probe shadow ledger.  The tool streams every recorded final identity
buffer and reconstructs the visible opposition/support history of each active
stable candidate labelled ``confirmed_free``.

Read-only diagnostic: no Unity asset, device session, TSDF, mesh, or runtime
state is modified.
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

from ScanCoverDepthPairDeviceAudit import DEFAULT_ADB, run_adb


def integer(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
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


def device_sequences(adb: Path, serial: str, frames_root: str) -> list[int]:
    listing = run_adb(
        adb,
        serial,
        "shell",
        f"ls -1 {frames_root}/*_gungel_correspondence_identity.bin 2>/dev/null",
    ).decode("utf-8", "replace")
    sequences = []
    for line in listing.splitlines():
        try:
            sequences.append(int(Path(line.strip()).name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(sequences))


def read_device_manifest(adb: Path, serial: str, session: str) -> dict[int, dict[str, str]]:
    text = run_adb(
        adb,
        serial,
        "exec-out",
        f'cat "{session}/fusion_inputs/manifest.csv"',
    ).decode("utf-8-sig", "replace")
    return {
        integer(row.get("sequence")): row
        for row in csv.DictReader(io.StringIO(text))
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb", type=Path, default=DEFAULT_ADB)
    parser.add_argument("--serial", default="2G97C5ZH4501R5")
    parser.add_argument("--session", required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_rows = read_csv(args.reference_csv)
    targets = [
        row
        for row in reference_rows
        if row.get("state") == "stable"
        and row.get("referenceStatus") == "confirmed_free"
        and integer(row.get("stable_id")) > 0
    ]
    by_stable = {integer(row["stable_id"]): row for row in targets}
    target_ids = np.asarray(sorted(by_stable), dtype=np.uint32)
    target_set = set(int(value) for value in target_ids)
    if not len(target_ids):
        raise RuntimeError("reference CSV contains no stable confirmed_free production candidates")

    promotions = {
        integer(row.get("stable_id")): row
        for row in read_csv(args.candidate_audit / "promotions.csv")
    }
    waves = {
        integer(row.get("stable_id")): row
        for row in read_csv(args.candidate_audit / "court_waves.csv")
    }
    manifest = read_device_manifest(args.adb, args.serial, args.session.rstrip("/"))
    frames_root = args.session.rstrip("/") + "/fusion_inputs/frames"
    sequences = device_sequences(args.adb, args.serial, frames_root)
    print(
        f"Tracing {len(target_ids)} stable confirmed-free candidates across "
        f"{len(sequences)} final identity frames",
        flush=True,
    )

    timeline: list[dict[str, object]] = []
    history: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    bytes_per_identity = 1600 * 16
    for start in range(0, len(sequences), args.batch_size):
        batch = sequences[start : start + args.batch_size]
        command = "cat " + " ".join(
            f'"{frames_root}/fusion_{sequence:06d}_gungel_correspondence_identity.bin"'
            for sequence in batch
        )
        payload = run_adb(args.adb, args.serial, "exec-out", command)
        expected = len(batch) * bytes_per_identity
        if len(payload) != expected:
            raise RuntimeError(
                f"identity batch {batch[0]}..{batch[-1]} expected {expected} bytes, "
                f"got {len(payload)}"
            )
        for offset, sequence in enumerate(batch):
            identity = np.frombuffer(
                payload,
                dtype="<u4",
                count=1600 * 4,
                offset=offset * bytes_per_identity,
            ).reshape(1600, 4)
            present = np.isin(identity[:, 1], target_ids)
            if not np.any(present):
                continue
            frame_identity = identity[present]
            row_manifest = manifest.get(sequence, {})
            for stable_id_raw in np.unique(frame_identity[:, 1]):
                stable_id = int(stable_id_raw)
                if stable_id not in target_set:
                    continue
                group = frame_identity[frame_identity[:, 1] == stable_id_raw]
                candidate_indices = group[:, 0]
                nonzero_candidates = candidate_indices[candidate_indices > 0]
                candidate_index = (
                    int(np.bincount(nonzero_candidates.astype(np.int64)).argmax()) - 1
                    if len(nonzero_candidates)
                    else -1
                )
                record = {
                    "sequence": sequence,
                    "sourceFrame": integer(row_manifest.get("sourceFrame")),
                    "gunGelFrame": integer(row_manifest.get("gunGelFrame")),
                    "stableId": stable_id,
                    "candidateIndex": candidate_index,
                    "matchedObservations": int(len(group)),
                    "maxDualSupportFrames": int(np.max(group[:, 2])),
                    "maxOppositionVotes": int(np.max(group[:, 3])),
                    "minOppositionVotes": int(np.min(group[:, 3])),
                    "motionQuality": row_manifest.get("motionQuality", ""),
                    "decision": row_manifest.get("decision", ""),
                }
                timeline.append(record)
                history[stable_id].append(record)
        done = min(start + args.batch_size, len(sequences))
        if done % 320 < args.batch_size or done == len(sequences):
            print(f"  identity frames {done}/{len(sequences)}", flush=True)

    summaries: list[dict[str, object]] = []
    responsibility_counts: Counter[str] = Counter()
    for stable_id in sorted(target_set):
        source = by_stable[stable_id]
        records = history.get(stable_id, [])
        max_opposition = max(
            (integer(row["maxOppositionVotes"]) for row in records), default=0
        )
        opposition_frames = sum(
            integer(row["maxOppositionVotes"]) > 0 for row in records
        )
        vote_two_frames = sum(
            integer(row["maxOppositionVotes"]) >= 2 for row in records
        )
        vote_three_frames = sum(
            integer(row["maxOppositionVotes"]) >= 3 for row in records
        )
        resets = 0
        previous = 0
        candidate_indices = []
        for row in records:
            opposition = integer(row["maxOppositionVotes"])
            if previous > 0 and opposition == 0:
                resets += 1
            previous = opposition
            if integer(row["candidateIndex"], -1) >= 0:
                candidate_indices.append(integer(row["candidateIndex"]))
        last_challenge = integer(source.get("last_challenge_frame"))
        if last_challenge <= 0:
            responsibility = "no_challenge_ever_reached_candidate"
        elif max_opposition <= 0:
            responsibility = "challenge_recorded_but_not_visible_in_identity_snapshot"
        elif max_opposition < 3:
            responsibility = "challenge_started_but_reset_before_three_votes"
        else:
            responsibility = "three_votes_seen_but_candidate_remained_active_anomaly"
        responsibility_counts[responsibility] += 1
        promotion = promotions.get(stable_id, {})
        wave = waves.get(stable_id, {})
        summaries.append(
            {
                "stableId": stable_id,
                "responsibilityStage": responsibility,
                "identityFrames": len(records),
                "identityFirstSequence": integer(records[0]["sequence"]) if records else -1,
                "identityLastSequence": integer(records[-1]["sequence"]) if records else -1,
                "maxOppositionVotesSeen": max_opposition,
                "oppositionVisibleFrames": opposition_frames,
                "voteTwoOrMoreFrames": vote_two_frames,
                "voteThreeOrMoreFrames": vote_three_frames,
                "visibleResetEvents": resets,
                "distinctCandidateIndices": len(set(candidate_indices)),
                "finalCandidateIndex": source.get("candidate_index", ""),
                "finalLastChallengeFrame": source.get("last_challenge_frame", ""),
                "finalLastSafeGeometryFrame": source.get("last_seen_frame", ""),
                "finalLastObservedFrame": source.get("last_observed_frame", ""),
                "finalDualAgreeSupport": source.get("dual_agree_support", ""),
                "finalOppositionVotes": source.get("opposition_votes", ""),
                "referenceStrictFreeViews": source.get("strictFreeViews", ""),
                "referenceNominalFreeViews": source.get("nominalFreeViews", ""),
                "referencePermissiveFreeViews": source.get("permissiveFreeViews", ""),
                "promotionFrame": promotion.get("promotion_frame", ""),
                "promotionMotionQuality": promotion.get("promotion_motion_quality", ""),
                "promotionViewRadius": promotion.get("promotion_view_radius", ""),
                "promotionRawProcessedDeltaMm": promotion.get(
                    "promotion_raw_processed_delta_mm", ""
                ),
                "courtWaveAdjudication": wave.get("adjudication", ""),
                "courtWaveSeverityMm": wave.get("severity_mm", ""),
                "courtWaveSourceFrame": wave.get("source_frame", ""),
            }
        )

    timeline_fields = [
        "sequence",
        "sourceFrame",
        "gunGelFrame",
        "stableId",
        "candidateIndex",
        "matchedObservations",
        "maxDualSupportFrames",
        "maxOppositionVotes",
        "minOppositionVotes",
        "motionQuality",
        "decision",
    ]
    summary_fields = list(summaries[0].keys()) if summaries else ["stableId"]
    write_csv(output / "production_free_identity_timeline.csv", timeline, timeline_fields)
    write_csv(output / "production_free_candidate_trace.csv", summaries, summary_fields)
    report = {
        "schema": "scancover.production-free-candidate-trace.v1",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "authority": "read_only_offline_diagnostic",
        "productionConsumers": 0,
        "session": args.session,
        "source": "authoritative final production identity snapshots plus candidate audit v9",
        "targetDefinition": "active stable production candidates classified confirmed_free by all three offline reference profiles",
        "targetCandidates": len(target_ids),
        "identityFramesAvailable": len(sequences),
        "timelineRows": len(timeline),
        "responsibilityStageCounts": dict(responsibility_counts),
        "boundary": [
            "identity snapshots expose votes only while an observation is associated with the stable ID",
            "no-challenge is a stage finding, not yet proof whether ray sampling, spatial search, or gating caused the miss",
            "the next audit must replay candidate-specific free rays for no-challenge candidates",
        ],
    }
    (output / "production_free_trace_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
