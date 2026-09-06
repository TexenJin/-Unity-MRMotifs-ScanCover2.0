#!/usr/bin/env python3
"""Deterministic retirement-contract replay for the candidate-footprint ledger.

This is a read-only audit.  It compares three candidate-lifecycle routes on one
sealed Quest session without writing Unity production state:

1. legacy: stable candidates actually retired by the current three-vote path;
2. footprint: two independent universal-free footprint witnesses;
3. guarded: footprint confirmation plus a bounded recent-support veto.

The formulas never use room coordinates or hand-picked stable IDs.  Stable IDs
shown in the report are evidence examples selected by their observed state.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_WINDOWS = (4, 8, 16, 32, 64, 128, 256)
AUDIT_STRIDE_FRAMES = 4
DEFAULT_HZ = 20.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def integer(value: object, default: int = 0) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def number(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def require_columns(path: Path, rows: list[dict[str, str]], required: set[str]) -> None:
    columns = set(rows[0]) if rows else set()
    missing = sorted(required - columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {', '.join(missing)}")


@dataclass
class GenerationState:
    stable_id: int
    generation: int
    first_frame: int = 2**31 - 1
    last_frame: int = -1
    support_frames: list[int] = field(default_factory=list)
    free_frames: list[int] = field(default_factory=list)
    support_confirmed_frame: int | None = None
    free_confirmed_frame: int | None = None
    conflict_frame: int | None = None
    max_support_views: int = 0
    max_free_views: int = 0
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0

    def add(self, row: dict[str, str]) -> None:
        frame = integer(row.get("gunGelFrame"))
        self.first_frame = min(self.first_frame, frame)
        self.last_frame = max(self.last_frame, frame)
        self.center_x = number(row.get("centerX"), self.center_x)
        self.center_y = number(row.get("centerY"), self.center_y)
        self.center_z = number(row.get("centerZ"), self.center_z)
        event_class = row.get("class", "")
        if event_class == "support":
            self.support_frames.append(frame)
            self.max_support_views = max(
                self.max_support_views, integer(row.get("independentCount"))
            )
        elif event_class == "free":
            self.free_frames.append(frame)
            self.max_free_views = max(
                self.max_free_views, integer(row.get("independentCount"))
            )
        if row.get("supportConfirmed") == "1" and self.support_confirmed_frame is None:
            self.support_confirmed_frame = frame
        if row.get("freeConfirmed") == "1" and self.free_confirmed_frame is None:
            self.free_confirmed_frame = frame
        if row.get("conflict") == "1" and self.conflict_frame is None:
            self.conflict_frame = frame

    @property
    def support_confirmed(self) -> bool:
        return self.support_confirmed_frame is not None

    @property
    def free_confirmed(self) -> bool:
        return self.free_confirmed_frame is not None

    @property
    def conflicted(self) -> bool:
        return self.conflict_frame is not None

    @property
    def clean_free(self) -> bool:
        return self.free_confirmed and not self.support_frames and not self.conflicted

    def recent_support(self, decision_frame: int, window: int, settle: int) -> int | None:
        """Latest support visible to a delayed decision, or None.

        A decision is committed `settle` frames after the second free witness.
        Support in the look-back window or during that settle interval vetoes it.
        """
        eligible = [
            frame for frame in self.support_frames
            if decision_frame - window <= frame <= decision_frame + settle
        ]
        return max(eligible) if eligible else None


def generation_states(events: list[dict[str, str]]) -> dict[tuple[int, int], GenerationState]:
    states: dict[tuple[int, int], GenerationState] = {}
    for row in sorted(
        events,
        key=lambda item: (
            integer(item.get("gunGelFrame")),
            integer(item.get("stableId")),
            integer(item.get("generation")),
        ),
    ):
        key = (integer(row.get("stableId")), integer(row.get("generation")))
        state = states.setdefault(key, GenerationState(*key))
        state.add(row)
    return states


def relevant_state(
    states_by_id: dict[int, list[GenerationState]], stable_id: int, frame: int
) -> GenerationState | None:
    """Select the generation observed nearest to a production retirement.

    The audit runs before the candidate mutation and is sampled every four
    GunGel frames, so a row up to one audit stride after the retirement counter
    is still considered the same transaction boundary.
    """
    options = [
        state for state in states_by_id.get(stable_id, [])
        if state.first_frame <= frame + AUDIT_STRIDE_FRAMES
    ]
    if not options:
        return None
    return max(options, key=lambda item: min(item.last_frame, frame + AUDIT_STRIDE_FRAMES))


def guarded_outcome(state: GenerationState, window: int, settle: int) -> tuple[str, int | None]:
    if not state.free_confirmed:
        return "no_footprint_free_confirmation", None
    decision = int(state.free_confirmed_frame)
    if (
        state.conflict_frame is not None
        and state.conflict_frame <= decision + settle
    ):
        return "veto_conflict", decision + settle
    support = state.recent_support(decision, window, settle)
    if support is not None:
        return "veto_recent_support", decision + settle
    return "allow_guarded_footprint", decision + settle


def run_self_tests() -> None:
    clean = GenerationState(1, 0, free_confirmed_frame=20, free_frames=[8, 20])
    assert guarded_outcome(clean, 16, 8)[0] == "allow_guarded_footprint"

    recent = GenerationState(
        2, 0, free_confirmed_frame=20, free_frames=[8, 20], support_frames=[12]
    )
    assert guarded_outcome(recent, 16, 8)[0] == "veto_recent_support"
    assert guarded_outcome(recent, 4, 8)[0] == "allow_guarded_footprint"

    conflict = GenerationState(
        3, 0, free_confirmed_frame=20, conflict_frame=20,
        free_frames=[8, 20], support_frames=[16], support_confirmed_frame=16,
    )
    assert guarded_outcome(conflict, 4, 8)[0] == "veto_conflict"

    no_free = GenerationState(4, 0, support_frames=[8, 20], support_confirmed_frame=20)
    assert guarded_outcome(no_free, 16, 8)[0] == "no_footprint_free_confirmation"


def main() -> None:
    run_self_tests()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="sealed replay session directory")
    parser.add_argument(
        "--retirements", type=Path,
        help="optional explicit gungel_candidate_audit/retirements.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ)
    parser.add_argument("--settle-frames", type=int, default=8)
    parser.add_argument(
        "--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS),
        help="recent-support look-back windows in GunGel frames",
    )
    args = parser.parse_args()

    session = args.session.resolve()
    output = args.output.resolve()
    events_path = session / "probe_shadow" / "direct45" / "events.csv"
    candidates_path = session / "probe_shadow" / "direct45" / "candidates.csv"
    summary_path = session / "probe_shadow" / "direct45" / "summary.json"
    complete_path = session / "capture_complete.json"
    retirements_path = (
        args.retirements.resolve() if args.retirements
        else session / "artifacts" / "gungel_candidate_audit" / "retirements.csv"
    )
    for path in (events_path, candidates_path, summary_path, complete_path, retirements_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    complete = json.loads(complete_path.read_text(encoding="utf-8-sig"))
    footprint_summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if complete.get("state") != "complete":
        raise RuntimeError(f"session is not sealed complete: {complete_path}")
    if footprint_summary.get("authority") != "shadow_only_zero_production_authority":
        raise RuntimeError("input footprint ledger is not marked shadow-only")

    events = read_csv(events_path)
    candidates = read_csv(candidates_path)
    retirements = read_csv(retirements_path)
    require_columns(
        events_path, events,
        {
            "gunGelFrame", "stableId", "generation", "class", "independentCount",
            "supportConfirmed", "freeConfirmed", "conflict", "centerX", "centerY", "centerZ",
        },
    )
    require_columns(
        retirements_path, retirements,
        {"stable_id", "reason", "retired_frame", "was_stable"},
    )

    states = generation_states(events)
    candidate_ids = {integer(row.get("stableId")) for row in candidates}
    states_by_id: dict[int, list[GenerationState]] = defaultdict(list)
    for state in states.values():
        states_by_id[state.stable_id].append(state)
    for value in states_by_id.values():
        value.sort(key=lambda state: (state.first_frame, state.generation))

    stable_legacy = [
        row for row in retirements
        if row.get("reason") == "contradicted"
        and integer(row.get("stable_id")) > 0
        and integer(row.get("was_stable")) == 1
    ]
    retired_by_id = {integer(row["stable_id"]): row for row in stable_legacy}
    free_states = [state for state in states.values() if state.free_confirmed]
    expected_free = integer(footprint_summary.get("freeConfirmedTransitions"), -1)
    expected_conflicts = integer(footprint_summary.get("conflicts"), -1)
    if expected_free >= 0 and len(free_states) != expected_free:
        raise RuntimeError(
            f"free confirmation reconciliation failed: events={len(free_states)} "
            f"summary={expected_free}"
        )
    event_conflicts = sum(state.conflicted for state in states.values())
    if expected_conflicts >= 0 and event_conflicts != expected_conflicts:
        raise RuntimeError(
            f"conflict reconciliation failed: events={event_conflicts} "
            f"summary={expected_conflicts}"
        )

    decision_ids = sorted(set(retired_by_id) | {state.stable_id for state in free_states})
    decision_rows: list[dict[str, object]] = []
    for stable_id in decision_ids:
        legacy = retired_by_id.get(stable_id)
        retire_frame = integer(legacy.get("retired_frame")) if legacy else -1
        state = (
            relevant_state(states_by_id, stable_id, retire_frame)
            if legacy else min(
                (item for item in states_by_id.get(stable_id, []) if item.free_confirmed),
                key=lambda item: int(item.free_confirmed_frame),
                default=None,
            )
        )
        row: dict[str, object] = {
            "stableId": stable_id,
            "generation": state.generation if state else "",
            "legacyRetired": 1 if legacy else 0,
            "legacyRetiredFrame": retire_frame if legacy else "",
            "footprintObserved": 1 if stable_id in candidate_ids else 0,
            "eventEvidenceObserved": 1 if state else 0,
            "supportEvents": len(state.support_frames) if state else 0,
            "freeEvents": len(state.free_frames) if state else 0,
            "independentSupportViews": state.max_support_views if state else 0,
            "independentFreeViews": state.max_free_views if state else 0,
            "supportConfirmedFrame": state.support_confirmed_frame if state else "",
            "freeConfirmedFrame": state.free_confirmed_frame if state else "",
            "conflictFrame": state.conflict_frame if state else "",
            "cleanFree": 1 if state and state.clean_free else 0,
            "centerX": state.center_x if state else "",
            "centerY": state.center_y if state else "",
            "centerZ": state.center_z if state else "",
        }
        for window in args.windows:
            if state:
                outcome, frame = guarded_outcome(state, window, args.settle_frames)
            else:
                outcome, frame = "not_observed_by_footprint_audit", None
            row[f"guard{window}Outcome"] = outcome
            row[f"guard{window}Frame"] = frame if frame is not None else ""
        decision_rows.append(row)

    sweep_rows: list[dict[str, object]] = []
    for window in args.windows:
        guarded = [guarded_outcome(state, window, args.settle_frames) for state in free_states]
        guarded_allowed = sum(outcome == "allow_guarded_footprint" for outcome, _ in guarded)
        guarded_recent = sum(outcome == "veto_recent_support" for outcome, _ in guarded)
        guarded_conflict = sum(outcome == "veto_conflict" for outcome, _ in guarded)

        legacy_vetoed: list[int] = []
        legacy_covered = 0
        legacy_event_detailed = 0
        for legacy in stable_legacy:
            stable_id = integer(legacy["stable_id"])
            retire_frame = integer(legacy["retired_frame"])
            if stable_id in candidate_ids:
                legacy_covered += 1
            state = relevant_state(states_by_id, stable_id, retire_frame)
            if state is None:
                continue
            legacy_event_detailed += 1
            eligible_support = [
                frame for frame in state.support_frames
                if retire_frame - window <= frame <= retire_frame + AUDIT_STRIDE_FRAMES
            ]
            if eligible_support:
                legacy_vetoed.append(stable_id)

        sweep_rows.append(
            {
                "supportWindowFrames": window,
                "supportWindowSeconds": window / max(args.hz, 1e-6),
                "settleFrames": args.settle_frames,
                "settleSeconds": args.settle_frames / max(args.hz, 1e-6),
                "legacyStableRetirements": len(stable_legacy),
                "legacyCoveredByFootprint": legacy_covered,
                "legacyWithDetailedEvents": legacy_event_detailed,
                "hybridLegacyVetoed": len(legacy_vetoed),
                "hybridLegacyAllowed": len(stable_legacy) - len(legacy_vetoed),
                "footprintFreeConfirmedGenerations": len(free_states),
                "guardedFootprintAllowed": guarded_allowed,
                "guardedRecentSupportVeto": guarded_recent,
                "guardedConflictVeto": guarded_conflict,
                "cleanFreeAllowed": sum(
                    state.clean_free
                    and guarded_outcome(state, window, args.settle_frames)[0]
                    == "allow_guarded_footprint"
                    for state in free_states
                ),
                "vetoStableIds": ";".join(str(value) for value in sorted(legacy_vetoed)),
            }
        )

    fields = [
        "stableId", "generation", "legacyRetired", "legacyRetiredFrame",
        "footprintObserved", "eventEvidenceObserved", "supportEvents", "freeEvents",
        "independentSupportViews", "independentFreeViews",
        "supportConfirmedFrame", "freeConfirmedFrame", "conflictFrame", "cleanFree",
        "centerX", "centerY", "centerZ",
    ]
    for window in args.windows:
        fields.extend([f"guard{window}Outcome", f"guard{window}Frame"])
    write_csv(output / "candidate_decisions.csv", decision_rows, fields)
    sweep_fields = list(sweep_rows[0]) if sweep_rows else []
    write_csv(output / "window_sweep.csv", sweep_rows, sweep_fields)

    legacy_covered = sum(integer(row["stable_id"]) in candidate_ids for row in stable_legacy)
    legacy_event_detailed = sum(
        relevant_state(states_by_id, integer(row["stable_id"]), integer(row["retired_frame"]))
        is not None for row in stable_legacy
    )
    clean_free = [state for state in free_states if state.clean_free]
    conflicted_free = [state for state in free_states if state.conflicted]
    summary = {
        "schema": "scancover.footprint-retirement-contract-replay.v1",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "authority": "offline_read_only_zero_production_authority",
        "session": str(session),
        "inputs": {
            "events": str(events_path),
            "candidates": str(candidates_path),
            "retirements": str(retirements_path),
            "captureComplete": str(complete_path),
        },
        "contract": {
            "legacy": "actual stable contradiction retirements from production",
            "footprint": "same stable ID and geometry generation reaches two independent universal-free witnesses",
            "guarded": "footprint confirmation delayed by settleFrames and vetoed by support inside the selected look-back/settle interval",
            "auditStrideFrames": AUDIT_STRIDE_FRAMES,
            "settleFrames": args.settle_frames,
            "supportWindowsFrames": args.windows,
            "assumedGunGelHzForDisplayOnly": args.hz,
        },
        "inputQuality": {
            "sealedComplete": True,
            "footprintFrames": integer(footprint_summary.get("frames")),
            "footprintRecords": integer(footprint_summary.get("records")),
            "footprintOverflow": integer(footprint_summary.get("overflowRecords")),
            "footprintReadbackErrors": integer(footprint_summary.get("readbackErrors")),
            "footprintWriteErrors": integer(footprint_summary.get("writeErrors")),
        },
        "observed": {
            "legacyStableRetirements": len(stable_legacy),
            "legacyStableRetirementsCoveredByFootprint": legacy_covered,
            "legacyStableRetirementsWithDetailedEvents": legacy_event_detailed,
            "footprintFreeConfirmedGenerations": len(free_states),
            "cleanFreeConfirmedGenerations": len(clean_free),
            "conflictedFreeConfirmedGenerations": len(conflicted_free),
            "cleanFreeStableIds": sorted(state.stable_id for state in clean_free),
            "conflictedFreeStableIds": sorted(state.stable_id for state in conflicted_free),
        },
        "limits": [
            "absence of footprint confirmation is not proof that a legacy retirement was wrong because the audit samples every four frames and applies visibility/motion gates",
            "support events are intentionally sparse in events.csv; the replay is strongest for positive support vetoes, not for proving support absence",
            "one session contains only one clean-free and one conflicted-free confirmed generation, so it cannot select a universal production time window",
            "no room coordinate or hand-picked stable ID participates in any decision formula",
        ],
        "outputs": {
            "candidateDecisions": "candidate_decisions.csv",
            "windowSweep": "window_sweep.csv",
            "report": "report.md",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report_lines = [
        "# Candidate footprint retirement contract replay",
        "",
        "## TL;DR",
        "",
        f"The sealed session contains **{len(stable_legacy)}** legacy stable-candidate retirements; "
        f"**{legacy_covered}** were visible to the footprint audit and **{legacy_event_detailed}** "
        f"have replayable high-value event rows.  The footprint route produced "
        f"**{len(free_states)}** double-view free confirmations: **{len(clean_free)}** clean and "
        f"**{len(conflicted_free)}** conflicted.",
        "",
        "The result supports a staged migration: first use recent footprint support as a veto on "
        "legacy deletion, then collect more clean/conflicted free confirmations before replacing "
        "legacy deletion authority wholesale.",
        "",
        "## Window sweep",
        "",
        "| support window | hybrid legacy vetoed | legacy allowed | strict footprint allowed | support veto | conflict veto |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sweep_rows:
        report_lines.append(
            f"| {row['supportWindowFrames']} frames / {number(row['supportWindowSeconds']):.2f}s "
            f"| {row['hybridLegacyVetoed']} | {row['hybridLegacyAllowed']} "
            f"| {row['guardedFootprintAllowed']} | {row['guardedRecentSupportVeto']} "
            f"| {row['guardedConflictVeto']} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- A positive recent-support veto is direct counter-evidence against immediate deletion.",
            "- Missing footprint confirmation is only 'not validated', not automatically a false legacy retirement.",
            "- The sweep reports sensitivity; it does not select a universal window from this room.",
            "- All routes are offline-only and make no Unity or Quest production changes.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(json.dumps(summary["observed"], indent=2, ensure_ascii=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
