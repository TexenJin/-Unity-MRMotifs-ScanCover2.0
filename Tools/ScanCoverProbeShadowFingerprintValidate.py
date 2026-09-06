#!/usr/bin/env python3
"""Validate a sealed probe-shadow candidate-fingerprint contract.

The validator is read-only.  It checks stable generation identity, patch-backed
challenge attribution, intersection/radius consistency, frame counter closure,
and Reject/Reopen thresholds.  Use --self-test before device data exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


REQUIRED_EVENT_COLUMNS = {
    "gunGelFrame", "event", "cellX", "cellY", "cellZ", "axis",
    "reason", "gapMm", "challengeVotes", "independentChallengeViews",
    "fingerprintId", "fingerprintGeneration", "fingerprintAcceptedFrame",
    "fingerprintPatchCount", "fingerprintPatchIndex", "intersectionX",
    "intersectionY", "intersectionZ", "intersectionRangeMm",
    "footprintRadiusMm", "footprintOffsetMm", "association",
}
REQUIRED_FRAME_COLUMNS = {
    "gunGelFrame", "freeSpaceChallenges", "fingerprintAssociationTests",
    "fingerprintAssociationHits", "fingerprintAssociationMisses",
}
REQUIRED_PATCH_COLUMNS = {
    "cellX", "cellY", "cellZ", "axis", "fingerprintId",
    "fingerprintGeneration", "fingerprintAcceptedFrame", "patchIndex",
    "patchFrame", "centerX", "centerY", "centerZ", "normalX", "normalY",
    "normalZ", "radiusMm",
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def integer(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def number(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def cell(row: dict[str, str]) -> tuple[int, int, int, int]:
    return tuple(integer(row.get(name)) for name in ("cellX", "cellY", "cellZ", "axis"))


def intersect_patch(
    camera: np.ndarray,
    hit: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
) -> Optional[tuple[np.ndarray, float, float]]:
    ray = hit - camera
    hit_range = float(np.linalg.norm(ray))
    if hit_range <= 1e-12:
        return None
    direction = ray / hit_range
    denominator = float(np.dot(direction, normal))
    if abs(denominator) <= 1e-5:
        return None
    distance = float(np.dot(center - camera, normal) / denominator)
    if distance <= 0.0 or distance >= hit_range:
        return None
    point = camera + direction * distance
    offset = float(np.linalg.norm(point - center))
    return (point, distance, offset) if offset <= radius else None


def self_test() -> None:
    camera = np.asarray([0.0, 0.0, 0.0])
    center = np.asarray([0.0, 0.0, 1.0])
    normal = np.asarray([0.0, 0.0, 1.0])
    central = intersect_patch(camera, np.asarray([0.0, 0.0, 2.0]), center, normal, 0.05)
    assert central is not None and abs(central[1] - 1.0) < 1e-9
    # The ray remains near the old 5 cm cell, but misses the real 5 cm-radius patch.
    outside = intersect_patch(camera, np.asarray([0.12, 0.0, 2.0]), center, normal, 0.05)
    assert outside is None
    behind = intersect_patch(
        camera, np.asarray([0.0, 0.0, 2.0]), np.asarray([0.0, 0.0, 3.0]), normal, 0.05
    )
    assert behind is None
    parallel = intersect_patch(
        camera, np.asarray([0.0, 0.0, 2.0]), center, np.asarray([1.0, 0.0, 0.0]), 0.05
    )
    assert parallel is None


def validate(session: Path) -> dict[str, object]:
    probe = session / "probe_shadow"
    event_fields, events = read_csv(probe / "verdict_events.csv")
    frame_fields, frames = read_csv(probe / "verdict_frames.csv")
    patch_fields, patches = read_csv(probe / "verdict_fingerprints.csv")
    errors: list[str] = []
    warnings: list[str] = []
    for label, actual, required in (
        ("events", set(event_fields), REQUIRED_EVENT_COLUMNS),
        ("frames", set(frame_fields), REQUIRED_FRAME_COLUMNS),
        ("fingerprints", set(patch_fields), REQUIRED_PATCH_COLUMNS),
    ):
        missing = sorted(required - actual)
        if missing:
            errors.append(f"{label} missing columns: {', '.join(missing)}")

    patch_map: dict[tuple[int, int, int], dict[str, str]] = {}
    identity_cells: dict[int, set[tuple[int, int, int, int]]] = defaultdict(set)
    generations: dict[tuple[int, int, int, int], set[int]] = defaultdict(set)
    for row in patches:
        fingerprint_id = integer(row.get("fingerprintId"))
        generation = integer(row.get("fingerprintGeneration"))
        patch_index = integer(row.get("patchIndex"), -1)
        radius = number(row.get("radiusMm"))
        if fingerprint_id <= 0 or generation <= 0 or patch_index < 0:
            errors.append(f"invalid fingerprint row id={fingerprint_id} gen={generation} patch={patch_index}")
            continue
        if not math.isfinite(radius) or radius <= 0.0:
            errors.append(f"fingerprint {fingerprint_id} patch {patch_index} has invalid radius {radius}")
        key = (fingerprint_id, generation, patch_index)
        if key in patch_map:
            errors.append(f"duplicate fingerprint patch {key}")
        patch_map[key] = row
        identity_cells[fingerprint_id].add(cell(row))
        generations[cell(row)].add(generation)
    for fingerprint_id, cells in identity_cells.items():
        if len(cells) != 1:
            errors.append(f"fingerprint {fingerprint_id} belongs to multiple cells: {sorted(cells)}")
    for key, values in generations.items():
        ordered = sorted(values)
        if ordered and ordered != list(range(ordered[0], ordered[-1] + 1)):
            errors.append(f"cell {key} has non-contiguous generations {ordered}")

    challenge_rows = 0
    reject_rows = 0
    reopen_rows = 0
    for row in events:
        event = row.get("event", "")
        reason = row.get("reason", "")
        if event.startswith("challenge_"):
            challenge_rows += 1
            fingerprint_id = integer(row.get("fingerprintId"))
            generation = integer(row.get("fingerprintGeneration"))
            patch_index = integer(row.get("fingerprintPatchIndex"), -1)
            patch = patch_map.get((fingerprint_id, generation, patch_index))
            if row.get("association") != "fingerprint_patch_hit":
                errors.append(f"challenge frame {row.get('gunGelFrame')} lacks patch-hit association")
            if patch is None:
                errors.append(
                    f"challenge frame {row.get('gunGelFrame')} references missing patch "
                    f"({fingerprint_id},{generation},{patch_index})"
                )
            elif cell(patch) != cell(row):
                errors.append(f"challenge frame {row.get('gunGelFrame')} patch cell mismatch")
            offset = number(row.get("footprintOffsetMm"))
            radius = number(row.get("footprintRadiusMm"))
            if not math.isfinite(offset) or not math.isfinite(radius) or offset > radius + 0.25:
                errors.append(
                    f"challenge frame {row.get('gunGelFrame')} offset {offset} exceeds radius {radius}"
                )
            if number(row.get("intersectionRangeMm"), 0.0) <= 0.0:
                errors.append(f"challenge frame {row.get('gunGelFrame')} has no forward intersection")
            if number(row.get("gapMm"), 0.0) < 50.0 - 0.01:
                errors.append(f"challenge frame {row.get('gunGelFrame')} has unreliable gap")
        if reason == "rejected_by_three_free_space_votes":
            reject_rows += 1
            if integer(row.get("challengeVotes")) < 3 or integer(row.get("independentChallengeViews")) < 2:
                errors.append(f"Reject frame {row.get('gunGelFrame')} violates vote contract")
        if reason == "reopened_by_independent_safe_support":
            reopen_rows += 1
            if integer(row.get("fingerprintId")) <= 0 or integer(row.get("fingerprintGeneration")) < 2:
                errors.append(f"Reopen frame {row.get('gunGelFrame')} did not mint a new generation")

    frame_tests = frame_hits = frame_misses = 0
    for row in frames:
        tests = integer(row.get("fingerprintAssociationTests"))
        hits = integer(row.get("fingerprintAssociationHits"))
        misses = integer(row.get("fingerprintAssociationMisses"))
        free = integer(row.get("freeSpaceChallenges"))
        frame_tests += tests
        frame_hits += hits
        frame_misses += misses
        if tests != hits + misses:
            errors.append(f"frame {row.get('gunGelFrame')} fingerprint counters do not close")
        if hits != free:
            errors.append(f"frame {row.get('gunGelFrame')} hits {hits} != challenges {free}")

    if not patches:
        warnings.append("no accepted fingerprint patches were exported")
    if not challenge_rows:
        warnings.append("no fingerprint-backed challenges were observed")
    return {
        "schema": "scancover.probe_shadow_fingerprint_validation.v1",
        "session": session.name,
        "valid": not errors,
        "eventRows": len(events),
        "frameRows": len(frames),
        "fingerprintPatches": len(patches),
        "fingerprintIds": len(identity_cells),
        "challengeRows": challenge_rows,
        "rejectRows": reject_rows,
        "reopenRows": reopen_rows,
        "associationTests": frame_tests,
        "associationHits": frame_hits,
        "associationMisses": frame_misses,
        "errors": errors,
        "warnings": warnings,
        "authority": "diagnostic_validation_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, nargs="?", help="Sealed replay session root")
    parser.add_argument("--self-test", action="store_true", help="Run deterministic geometry cases")
    parser.add_argument("--output", type=Path, help="Optional summary JSON path")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("candidate fingerprint geometry self-test: PASS")
    if args.session is None:
        return 0
    summary = validate(args.session.resolve())
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if summary["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
