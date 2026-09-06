#!/usr/bin/env python3
"""Summarize the zero-authority virtual probe shadow adjudicator outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def number(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    return float(value) if value not in ("", None) else 0.0


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sum_field(rows: Iterable[dict[str, str]], name: str) -> int:
    return int(round(sum(number(row, name) for row in rows)))


def ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def cell_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return tuple(row.get(name, "") for name in ("cellX", "cellY", "cellZ", "axis"))


def build_rejection_lifecycles(
    event_rows: list[dict[str, str]], cell_rows: list[dict[str, str]]
) -> list[dict[str, object]]:
    snapshots = {cell_key(row): row for row in cell_rows}
    active: dict[tuple[str, str, str, str], dict[str, object]] = {}
    completed: list[dict[str, object]] = []
    indexed_events = list(enumerate(event_rows))
    indexed_events.sort(key=lambda pair: (int(number(pair[1], "gunGelFrame")), pair[0]))

    for _, event in indexed_events:
        key = cell_key(event)
        reason = event.get("reason", "")
        frame = int(number(event, "gunGelFrame"))
        if reason == "rejected_by_three_free_space_votes":
            episode = {
                "cell": ":".join(key),
                "rejectGunGelFrame": frame,
                "targetMetres": [
                    number(event, "targetX"),
                    number(event, "targetY"),
                    number(event, "targetZ"),
                ],
                "rejectGapMm": number(event, "gapMm"),
                "challengeVotes": int(number(event, "challengeVotes")),
                "independentChallengeViews": int(
                    number(event, "independentChallengeViews")
                ),
                "recoveryWitnessFrames": [],
                "reopenGunGelFrame": None,
            }
            active[key] = episode
            continue

        episode = active.get(key)
        if episode is None or event.get("event") != "recovery_independent":
            continue
        recovery_frames = episode["recoveryWitnessFrames"]
        assert isinstance(recovery_frames, list)
        recovery_frames.append(frame)
        if reason == "reopened_by_independent_safe_support":
            episode["reopenGunGelFrame"] = frame
            completed.append(episode)
            del active[key]

    completed.extend(active.values())
    completed.sort(key=lambda row: int(row["rejectGunGelFrame"]))
    for episode in completed:
        key = tuple(str(episode["cell"]).split(":"))
        snapshot = snapshots.get(key, {})
        reject_frame = int(episode["rejectGunGelFrame"])
        last_seen = int(number(snapshot, "lastSeenFrame"))
        recovery_frames = episode["recoveryWitnessFrames"]
        assert isinstance(recovery_frames, list)
        reopened = episode["reopenGunGelFrame"] is not None
        episode["lastStableSeenFrame"] = last_seen
        episode["stableCandidateSeenAfterReject"] = last_seen > reject_frame
        if reopened:
            episode["outcome"] = "reopened"
        elif recovery_frames:
            episode["outcome"] = "recovery_started_not_reopened"
        elif last_seen > reject_frame:
            episode["outcome"] = "revisited_without_safe_recovery"
        else:
            episode["outcome"] = "not_revisited_after_reject"
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit probe_shadow hold/accept/reject outputs without changing production data."
    )
    parser.add_argument("session", type=Path, help="replay session directory")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()

    session = args.session.resolve()
    complete_path = session / "capture_complete.json"
    if not complete_path.is_file():
        raise SystemExit("capture_complete.json is missing; refuse an unsealed session")
    complete = json.loads(complete_path.read_text(encoding="utf-8-sig"))
    if complete.get("state") != "complete":
        raise SystemExit(f"session state is {complete.get('state')!r}, not complete")

    probe = session / "probe_shadow"
    frame_rows = read_csv(probe / "verdict_frames.csv")
    event_rows = read_csv(probe / "verdict_events.csv")
    cell_rows = read_csv(probe / "verdict_cells.csv")
    schema = json.loads((probe / "verdict_schema.json").read_text(encoding="utf-8-sig"))

    valid = sum_field(frame_rows, "valid")
    observation_accept = sum_field(frame_rows, "observationAccept")
    observation_hold = sum_field(frame_rows, "observationHold")
    observation_reject = sum_field(frame_rows, "observationReject")
    free_challenges = sum_field(frame_rows, "freeSpaceChallenges")
    independent_free = sum_field(frame_rows, "independentFreeSpaceChallenges")
    hold_fields = [
        "holdRawMissing",
        "holdDualDisagree",
        "holdNoStable",
        "holdResidual",
        "holdNormal",
        "holdOpposed",
        "holdMotion",
        "holdOuter",
    ]
    hold_reasons = {name: sum_field(frame_rows, name) for name in hold_fields}
    final_verdicts = Counter(row.get("verdict", "") for row in cell_rows)
    event_types = Counter(row.get("event", "") for row in event_rows)
    event_reasons = Counter(row.get("reason", "") for row in event_rows)
    reliable_free_votes = (
        event_types.get("challenge_independent", 0)
        + event_types.get("challenge_correlated", 0)
    )
    lifecycles = build_rejection_lifecycles(event_rows, cell_rows)
    lifecycle_outcomes = Counter(str(row["outcome"]) for row in lifecycles)

    expected_frames = int(complete.get("probeAdjudicatorFrameRows", -1))
    expected_events = int(complete.get("probeAdjudicatorEventRows", -1))
    write_errors = int(complete.get("probeAdjudicatorWriteErrors", -1))
    integrity = {
        "frameRowsMatchSeal": expected_frames == len(frame_rows),
        "eventRowsMatchSeal": expected_events == len(event_rows),
        "writeErrorsZero": write_errors == 0,
        "allFrameObservationsAccounted":
            observation_accept + observation_hold + observation_reject == valid,
    }

    report = {
        "schema": "scancover.virtual_probe_shadow_verdict_summary.v3",
        "session": session.name,
        "authority": schema.get("authority", "unknown"),
        "profile": schema.get("profile", "unknown"),
        "integrity": integrity,
        "rows": {
            "frames": len(frame_rows),
            "events": len(event_rows),
            "cells": len(cell_rows),
        },
        "observations": {
            "valid": valid,
            "accept": observation_accept,
            "hold": observation_hold,
            "reject": observation_reject,
            "acceptRate": ratio(observation_accept, valid),
            "holdRate": ratio(observation_hold, valid),
            "rejectRate": ratio(observation_reject, valid),
        },
        "holdReasons": hold_reasons,
        "transitions": {
            "accept": sum_field(frame_rows, "acceptTransitions"),
            "reject": sum_field(frame_rows, "rejectTransitions"),
            "reopen": sum_field(frame_rows, "reopenTransitions"),
        },
        "freeSpace": {
            "allChallenges": free_challenges,
            "reliableVotes": reliable_free_votes,
            "independentReliableChallenges": independent_free,
            "independentRateAmongReliableVotes":
                ratio(independent_free, reliable_free_votes),
            "gapLt20Mm": sum_field(frame_rows, "freeGapLt20Mm"),
            "gap20To50Mm": sum_field(frame_rows, "freeGap20To50Mm"),
            "gap50To100Mm": sum_field(frame_rows, "freeGap50To100Mm"),
            "gapGe100Mm": sum_field(frame_rows, "freeGapGe100Mm"),
        },
        "finalCells": dict(sorted(final_verdicts.items())),
        "eventTypes": dict(sorted(event_types.items())),
        "eventReasons": dict(sorted(event_reasons.items())),
        "rejectionLifecycle": {
            "episodes": len(lifecycles),
            "outcomes": dict(sorted(lifecycle_outcomes.items())),
            "reversiblePathObserved": any(
                row["outcome"] == "reopened" for row in lifecycles
            ),
            "details": lifecycles,
        },
        "readyForProductionAuthority": False,
        "nextValidation":
            "Compare shadow rejects/reopens against deterministic depth-pair replay and visible defect lineage before any production connection.",
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if all(integrity.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
