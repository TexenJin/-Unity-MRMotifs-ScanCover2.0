#!/usr/bin/env python3
"""Deterministically replay ScanCover graduation witness events.

This tool consumes probe_shadow/graduation_race_witness_events.csv. It validates
that the exported raw camera/candidate samples reproduce both the production
and race independence decisions, then rebuilds every graduation lane without
running Unity or touching production geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path


BASELINE_METRES = 0.08
ANGLE_DEG = 3.0
FRAME_GAP = 2

CONTRACTS = (
    ("baseline_2view", 2, 0.0),
    ("support_3view", 3, 0.0),
    ("support_4view", 4, 0.0),
    ("revisit_1s", 2, 1.0),
    ("revisit_2s", 2, 2.0),
    ("revisit_4s", 2, 4.0),
    ("support_3view_revisit_2s", 3, 2.0),
)

REQUIRED_COLUMNS = {
    "frame", "timeSeconds", "cellX", "cellY", "cellZ", "axis",
    "generation", "productionStream", "cameraX", "cameraY", "cameraZ",
    "centerX", "centerY", "centerZ", "productionIndependent",
    "productionDecision", "productionCountBefore", "productionCountAfter",
    "productionCapacity", "raceIndependent", "raceDecision",
    "raceCountBefore", "raceCountAfter",
}


def as_int(row: dict[str, str], name: str) -> int:
    return int(row[name])


def as_float(row: dict[str, str], name: str) -> float:
    return float(row[name])


def vector(row: dict[str, str], prefix: str) -> tuple[float, float, float]:
    return tuple(as_float(row, prefix + axis) for axis in ("X", "Y", "Z"))


def subtract(a: tuple[float, float, float],
             b: tuple[float, float, float]) -> tuple[float, float, float]:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def distance(a: tuple[float, float, float],
             b: tuple[float, float, float]) -> float:
    delta = subtract(a, b)
    return math.sqrt(sum(value * value for value in delta))


def normalize(value: tuple[float, float, float]) -> tuple[float, float, float]:
    magnitude = math.sqrt(sum(component * component for component in value))
    if magnitude <= 1e-6:
        return 0.0, 0.0, 1.0
    return tuple(component / magnitude for component in value)


def angle_degrees(a: tuple[float, float, float],
                  b: tuple[float, float, float]) -> float:
    dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
    return math.degrees(math.acos(dot))


@dataclass(frozen=True)
class Witness:
    camera: tuple[float, float, float]
    direction: tuple[float, float, float]
    frame: int
    time_seconds: float


@dataclass
class Ledger:
    witnesses: list[Witness] = field(default_factory=list)
    max_consecutive_gap_seconds: float = 0.0
    graduated: dict[str, tuple[int, float, int, float]] = field(
        default_factory=dict
    )

    def evaluate(self, camera: tuple[float, float, float],
                 center: tuple[float, float, float], frame: int,
                 capacity: int | None) -> tuple[str, float, float]:
        direction = normalize(subtract(camera, center))
        blocked_time = False
        blocked_pose = False
        nearest_score = math.inf
        nearest_baseline = 0.0
        nearest_spread = 0.0
        for witness in self.witnesses:
            baseline = distance(witness.camera, camera)
            spread = angle_degrees(witness.direction, direction)
            score = max(baseline / BASELINE_METRES, spread / ANGLE_DEG)
            if score < nearest_score:
                nearest_score = score
                nearest_baseline = baseline
                nearest_spread = spread
            if frame - witness.frame < FRAME_GAP:
                blocked_time = True
            if baseline < BASELINE_METRES and spread < ANGLE_DEG:
                blocked_pose = True
        if blocked_time and blocked_pose:
            return "correlated_time_and_pose", nearest_baseline, nearest_spread
        if blocked_time:
            return "correlated_time", nearest_baseline, nearest_spread
        if blocked_pose:
            return "correlated_pose", nearest_baseline, nearest_spread
        if capacity is not None and len(self.witnesses) >= capacity:
            return "capacity_full", nearest_baseline, nearest_spread
        return "accepted", nearest_baseline, nearest_spread

    def append(self, camera: tuple[float, float, float],
               center: tuple[float, float, float], frame: int,
               time_seconds: float) -> None:
        if self.witnesses:
            gap = max(0.0, time_seconds - self.witnesses[-1].time_seconds)
            self.max_consecutive_gap_seconds = max(
                self.max_consecutive_gap_seconds, gap
            )
        self.witnesses.append(Witness(
            camera=camera,
            direction=normalize(subtract(camera, center)),
            frame=frame,
            time_seconds=time_seconds,
        ))
        for name, required_views, required_seconds in CONTRACTS:
            if name in self.graduated:
                continue
            if len(self.witnesses) < required_views:
                continue
            if self.max_consecutive_gap_seconds + 1e-6 < required_seconds:
                continue
            self.graduated[name] = (
                frame, time_seconds, len(self.witnesses),
                self.max_consecutive_gap_seconds,
            )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key_from(row: dict[str, str]) -> tuple[int, int, int, int, int]:
    return tuple(as_int(row, name) for name in (
        "cellX", "cellY", "cellZ", "axis", "generation"
    ))


def run(session: Path, output: Path) -> dict[str, object]:
    witness_path = session / "probe_shadow" / "graduation_race_witness_events.csv"
    if not witness_path.is_file():
        raise FileNotFoundError(f"missing witness event log: {witness_path}")
    with witness_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        source_rows = list(reader)

    race_ledgers: dict[tuple[int, int, int, int, int], Ledger] = {}
    # The production support ledger belongs to the live cell state and survives
    # fingerprint graduation.  Its generation can therefore advance while the
    # bounded SupportWitnesses array (and its count) remains unchanged.  Recovery
    # witnesses are different: they describe the next generation and are reset
    # when that recovery cycle ends, so generation remains part of their key.
    production_ledgers: dict[tuple[object, ...], Ledger] = {}
    validation_rows: list[dict[str, object]] = []

    for sequence, row in enumerate(source_rows):
        key = key_from(row)
        stream = row["productionStream"]
        frame = as_int(row, "frame")
        time_seconds = as_float(row, "timeSeconds")
        camera = vector(row, "camera")
        center = vector(row, "center")

        # ApplySafeSupport always evaluates the long-lived support ledger first.
        # While a cell is rejected the exported production stream describes the
        # separate recovery decision, but the same callback has already updated
        # SupportWitnesses.  Reproduce that hidden-in-the-row state transition so
        # later support generations start with the same bounded ledger as device.
        if stream == "recovery":
            support_key = ((key[0], key[1], key[2], key[3]), "support")
            support = production_ledgers.setdefault(support_key, Ledger())
            hidden_decision, _, _ = support.evaluate(
                camera, center, frame, 4
            )
            if hidden_decision == "accepted":
                support.append(camera, center, frame, time_seconds)

        production_key = (
            (key[0], key[1], key[2], key[3]), stream
        ) if stream == "support" else (key, stream)
        production = production_ledgers.setdefault(production_key, Ledger())
        production_capacity = as_int(row, "productionCapacity")
        production_decision, production_baseline, production_spread = (
            production.evaluate(camera, center, frame, production_capacity)
        )
        if production_decision == "accepted":
            production.append(camera, center, frame, time_seconds)

        race = race_ledgers.setdefault(key, Ledger())
        race_decision, race_baseline, race_spread = race.evaluate(
            camera, center, frame, None
        )
        if race_decision == "accepted":
            race.append(camera, center, frame, time_seconds)

        logged_production = row["productionDecision"]
        logged_race = row["raceDecision"]
        validation_rows.append({
            "sequence": sequence,
            "frame": frame,
            "cellX": key[0], "cellY": key[1], "cellZ": key[2],
            "axis": key[3], "generation": key[4],
            "productionStream": stream,
            "loggedProductionDecision": logged_production,
            "replayedProductionDecision": production_decision,
            "productionDecisionMatch": int(logged_production == production_decision),
            "loggedProductionCountBefore": as_int(row, "productionCountBefore"),
            "replayedProductionCountBefore": len(production.witnesses) - int(production_decision == "accepted"),
            "loggedProductionCountAfter": as_int(row, "productionCountAfter"),
            "replayedProductionCountAfter": len(production.witnesses),
            "replayedProductionNearestBaselineMm": production_baseline * 1000.0,
            "replayedProductionNearestSpreadDeg": production_spread,
            "loggedRaceDecision": logged_race,
            "replayedRaceDecision": race_decision,
            "raceDecisionMatch": int(logged_race == race_decision),
            "loggedRaceCountBefore": as_int(row, "raceCountBefore"),
            "replayedRaceCountBefore": len(race.witnesses) - int(race_decision == "accepted"),
            "loggedRaceCountAfter": as_int(row, "raceCountAfter"),
            "replayedRaceCountAfter": len(race.witnesses),
            "replayedRaceNearestBaselineMm": race_baseline * 1000.0,
            "replayedRaceNearestSpreadDeg": race_spread,
        })

    contract_rows: list[dict[str, object]] = []
    for key in sorted(race_ledgers):
        ledger = race_ledgers[key]
        for name, required_views, required_seconds in CONTRACTS:
            graduation = ledger.graduated.get(name)
            contract_rows.append({
                "cellX": key[0], "cellY": key[1], "cellZ": key[2],
                "axis": key[3], "generation": key[4],
                "contract": name,
                "requiredIndependentViews": required_views,
                "requiredRevisitSeconds": required_seconds,
                "graduated": int(graduation is not None),
                "graduatedFrame": graduation[0] if graduation else -1,
                "graduatedTimeSeconds": graduation[1] if graduation else "",
                "viewsAtGraduation": graduation[2] if graduation else 0,
                "revisitGapAtGraduationSeconds": graduation[3] if graduation else 0.0,
                "finalIndependentViews": len(ledger.witnesses),
                "maxIndependentGapSeconds": ledger.max_consecutive_gap_seconds,
            })

    summary_rows: list[dict[str, object]] = []
    for name, required_views, required_seconds in CONTRACTS:
        rows = [row for row in contract_rows if row["contract"] == name]
        graduated = sum(int(row["graduated"]) for row in rows)
        summary_rows.append({
            "contract": name,
            "requiredIndependentViews": required_views,
            "requiredRevisitSeconds": required_seconds,
            "trials": len(rows),
            "graduated": graduated,
            "notGraduated": len(rows) - graduated,
            "graduatedRate": graduated / len(rows) if rows else math.nan,
        })

    exported_contract_mismatches = 0
    exported_candidate_path = (
        session / "probe_shadow" / "graduation_race_candidates.csv"
    )
    if exported_candidate_path.is_file():
        with exported_candidate_path.open(
                "r", encoding="utf-8-sig", newline="") as handle:
            exported_rows = list(csv.DictReader(handle))
        exported_map = {
            (
                int(row["cellX"]), int(row["cellY"]), int(row["cellZ"]),
                int(row["axis"]), int(row["candidateGeneration"]),
                row["contract"],
            ): row
            for row in exported_rows
        }
        replay_map = {
            (
                int(row["cellX"]), int(row["cellY"]), int(row["cellZ"]),
                int(row["axis"]), int(row["generation"]), row["contract"],
            ): row
            for row in contract_rows
        }
        all_keys = set(exported_map) | set(replay_map)
        for contract_key in all_keys:
            exported = exported_map.get(contract_key)
            replayed = replay_map.get(contract_key)
            if exported is None or replayed is None:
                exported_contract_mismatches += 1
                continue
            if int(exported["graduated"]) != int(replayed["graduated"]):
                exported_contract_mismatches += 1
                continue
            if int(exported["graduatedFrame"]) != int(replayed["graduatedFrame"]):
                exported_contract_mismatches += 1
                continue
            if int(exported["viewsAtGraduation"]) != int(replayed["viewsAtGraduation"]):
                exported_contract_mismatches += 1

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "witness_replay_validation.csv", validation_rows)
    write_csv(output / "witness_replay_contracts.csv", contract_rows)
    write_csv(output / "witness_replay_summary.csv", summary_rows)

    production_mismatches = sum(
        1 - int(row["productionDecisionMatch"]) for row in validation_rows
    )
    race_mismatches = sum(
        1 - int(row["raceDecisionMatch"]) for row in validation_rows
    )
    count_mismatches = sum(
        int(row["loggedProductionCountBefore"] != row["replayedProductionCountBefore"] or
            row["loggedProductionCountAfter"] != row["replayedProductionCountAfter"] or
            row["loggedRaceCountBefore"] != row["replayedRaceCountBefore"] or
            row["loggedRaceCountAfter"] != row["replayedRaceCountAfter"])
        for row in validation_rows
    )
    result = {
        "schema": "scancover.graduation_witness_offline_replay.v1",
        "session": str(session),
        "witnessEvents": len(source_rows),
        "candidateGenerations": len(race_ledgers),
        "productionDecisionMismatches": production_mismatches,
        "raceDecisionMismatches": race_mismatches,
        "countMismatches": count_mismatches,
        "exportedContractMismatches": exported_contract_mismatches,
        "replayValid": production_mismatches == 0 and
                       race_mismatches == 0 and count_mismatches == 0 and
                       exported_contract_mismatches == 0,
        "contracts": summary_rows,
        "authority": "offline_read_only_no_production_authority",
    }
    (output / "witness_replay_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path,
                        help="session_* directory containing probe_shadow")
    parser.add_argument("--output", type=Path,
                        help="output directory; defaults to session/offline_witness_replay")
    args = parser.parse_args()
    session = args.session.resolve()
    output = (args.output or (session / "offline_witness_replay")).resolve()
    result = run(session, output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["replayValid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
