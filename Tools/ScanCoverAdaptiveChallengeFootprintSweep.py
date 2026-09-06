#!/usr/bin/env python3
"""Read-only sweep for Quest GunGel free-space challenge footprints.

This audit intentionally does not change Unity or the production compute path.
It answers a narrower question first: can a range-adaptive footprint bridge the
known sparse-ray misses without silently treating an arbitrarily large world
radius as a device model?

The input events are produced by ScanCoverProductionFreeRayAudit.py.  Those
events already passed the conservative full-depth free-space reference test.
For each event with replayable challenge geometry, this tool compares:

* the production fixed 35 mm radius;
* one, 1.5, two and three Quest depth-tile half diagonals, plus QRS uncertainty;
* candidate-centric projection, recorded only as a separate reference route.

The surface control in this pass is aggregate-only: confirmed_surface rows had
zero conservative free views in the reference court.  They do NOT prove that a
wide cone is safe around discontinuities because the saved event table does not
contain their per-frame neighbouring rays.  The report keeps that boundary
explicit instead of reporting a circular false-positive rate.
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

from ScanCoverProductionFreeRayAudit import independent_event_views, integer, number


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fov_from_meta(path: Path, eye: int) -> tuple[float, float, int, int]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    fov = meta.get("fovRadians", [])[eye]
    if len(fov) != 4:
        raise RuntimeError(f"Missing four-angle fovRadians[{eye}] in {path}")
    # Meta records left/right/top/bottom signed angles.  Span is independent of
    # the asymmetric principal point and is enough for this conservative sweep.
    horizontal = abs(float(fov[1]) - float(fov[0]))
    vertical = abs(float(fov[2]) - float(fov[3]))
    return horizontal, vertical, int(meta["width"]), int(meta["height"])


def event_range(row: dict[str, str]) -> float:
    camera = np.asarray(
        [number(row.get("cameraX")), number(row.get("cameraY")), number(row.get("cameraZ"))],
        dtype=np.float64,
    )
    candidate = np.asarray(
        [
            number(row.get("candidateX")),
            number(row.get("candidateY")),
            number(row.get("candidateZ")),
        ],
        dtype=np.float64,
    )
    return float(np.linalg.norm(candidate - camera))


def tile_half_diagonal(
    range_metres: float, horizontal_fov: float, vertical_fov: float,
    grid_x: int, grid_y: int,
) -> float:
    half_x = horizontal_fov / max(grid_x, 1) * 0.5
    half_y = vertical_fov / max(grid_y, 1) * 0.5
    angular = math.hypot(half_x, half_y)
    return range_metres * math.tan(angular)


def eligible_geometry(row: dict[str, str]) -> bool:
    return row.get("stage") in {
        "blocked_40x40_ray_footprint",
        "challenge_should_reach_candidate",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--reference-candidates", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--calibration-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eye", type=int, default=1)
    parser.add_argument("--grid-x", type=int, default=40)
    parser.add_argument("--grid-y", type=int, default=40)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    events = read_csv(args.events)
    candidate_rows = read_csv(args.candidate_summary)
    references = read_csv(args.reference_candidates)
    reference_summary = json.loads(args.reference_summary.read_text(encoding="utf-8"))
    horizontal_fov, vertical_fov, depth_width, depth_height = fov_from_meta(
        args.calibration_meta, args.eye
    )
    qrs_metres = number(reference_summary["qrsCalibration"]["absRangeDifferenceMmP95"]) / 1000.0

    candidate_by_id = {integer(row.get("stableId")): row for row in candidate_rows}
    reference_by_id = {integer(row.get("stable_id")): row for row in references}

    policies = [
        ("fixed_35mm", None, 0.035),
        ("tile_1x_plus_qrs", 1.0, qrs_metres),
        ("tile_1_5x_plus_qrs", 1.5, qrs_metres),
        ("tile_2x_plus_qrs", 2.0, qrs_metres),
        ("tile_3x_plus_qrs", 3.0, qrs_metres),
    ]

    evaluated: list[dict[str, object]] = []
    by_candidate: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    for source in events:
        stable_id = integer(source.get("stableId"))
        # The ray-event table owns the time-aligned final-safe boundary.  The
        # per-candidate summary intentionally contains only aggregate counts.
        # Reading the boundary from that summary would silently turn blank into
        # frame zero and contaminate the current-false control with old history.
        last_safe = integer(source.get("candidateFinalLastSafeFrame"))
        frame = integer(source.get("gunGelFrame"), -1)
        if source.get("cameraX", "") == "" or frame < 0:
            continue
        range_metres = event_range(source)
        tile_metres = tile_half_diagonal(
            range_metres, horizontal_fov, vertical_fov, args.grid_x, args.grid_y
        )
        nearest_metres = number(source.get("nearestRayDistanceMm"), float("nan")) / 1000.0
        row: dict[str, object] = {
            "stableId": stable_id,
            "gunGelFrame": frame,
            "stage": source.get("stage", ""),
            "postLastSafe": int(frame >= last_safe),
            "candidateRangeM": range_metres,
            "nearestQualifyingRayMm": nearest_metres * 1000.0,
            "questTileHalfDiagonalMm": tile_metres * 1000.0,
            "requiredTileMultiplesWithoutQrs": (
                nearest_metres / tile_metres if tile_metres > 0.0 else float("nan")
            ),
            "cameraX": source.get("cameraX", ""),
            "cameraY": source.get("cameraY", ""),
            "cameraZ": source.get("cameraZ", ""),
            "candidateViewX": source.get("candidateViewX", ""),
            "candidateViewY": source.get("candidateViewY", ""),
            "candidateViewZ": source.get("candidateViewZ", ""),
        }
        geometry_ok = eligible_geometry(source) and math.isfinite(nearest_metres)
        for name, multiplier, additive in policies:
            radius = additive if multiplier is None else multiplier * tile_metres + additive
            row[f"{name}_radiusMm"] = radius * 1000.0
            row[f"{name}_reaches"] = int(geometry_ok and nearest_metres <= radius)
        # This is not a cone result.  The event exists because exact candidate
        # projection found reliable empty depth, so it is a route-availability
        # marker only and must never be compared as a measured false-positive rate.
        row["candidate_projection_reference_reaches"] = int(geometry_ok)
        evaluated.append(row)
        by_candidate[stable_id].append(row)

    candidate_results: list[dict[str, object]] = []
    for stable_id, rows in sorted(by_candidate.items()):
        source = candidate_by_id.get(stable_id, {})
        post = [row for row in rows if integer(row.get("postLastSafe")) == 1]
        result: dict[str, object] = {
            "stableId": stable_id,
            "currentFalseSurfaceConfirmed": integer(source.get("currentFalseSurfaceConfirmed")),
            "postLastSafeReplayableEvents": len(post),
            "referenceStatus": reference_by_id.get(stable_id, {}).get("referenceStatus", ""),
        }
        for name, _, _ in policies:
            hits = [row for row in post if integer(row.get(f"{name}_reaches")) == 1]
            result[f"{name}_events"] = len(hits)
            result[f"{name}_independentViews"] = independent_event_views(hits)
        projection_hits = [
            row for row in post if integer(row.get("candidate_projection_reference_reaches")) == 1
        ]
        result["candidate_projection_reference_events"] = len(projection_hits)
        result["candidate_projection_reference_independentViews"] = independent_event_views(
            projection_hits
        )
        candidate_results.append(result)

    stable_surface_controls = [
        row
        for row in references
        if row.get("state") == "stable" and row.get("referenceStatus") == "confirmed_surface"
    ]
    stable_unknown = [
        row
        for row in references
        if row.get("state") == "stable" and row.get("referenceStatus") == "unknown"
    ]
    control_free_counts = Counter(integer(row.get("nominalFreeViews")) for row in stable_surface_controls)
    convicted = [row for row in candidate_results if integer(row["currentFalseSurfaceConfirmed"]) == 1]

    policy_summary: list[dict[str, object]] = []
    for name, multiplier, additive in policies:
        positive_caught = sum(integer(row.get(f"{name}_independentViews")) >= 2 for row in convicted)
        policy_summary.append(
            {
                "policy": name,
                "tileMultiplier": "" if multiplier is None else multiplier,
                "additiveMm": additive * 1000.0,
                "confirmedCurrentFalseControls": len(convicted),
                "confirmedCurrentFalseCaughtAtTwoViews": positive_caught,
                "surfaceFalseChallengeRate": "not_measured_without_surface_event_rays",
                "productionReady": 0,
            }
        )

    write_csv(
        output / "adaptive_footprint_events.csv",
        evaluated,
        sorted({key for row in evaluated for key in row}) if evaluated else ["stableId"],
    )
    write_csv(
        output / "adaptive_footprint_candidates.csv",
        candidate_results,
        sorted({key for row in candidate_results for key in row}) if candidate_results else ["stableId"],
    )
    write_csv(
        output / "adaptive_footprint_policies.csv",
        policy_summary,
        list(policy_summary[0]) if policy_summary else ["policy"],
    )

    report = {
        "schema": "scancover.adaptive-challenge-footprint-sweep.v1",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "authority": "read_only_offline_diagnostic",
        "productionConsumers": 0,
        "inputs": {
            "events": str(args.events.resolve()),
            "candidateSummary": str(args.candidate_summary.resolve()),
            "referenceCandidates": str(args.reference_candidates.resolve()),
            "calibrationMeta": str(args.calibration_meta.resolve()),
        },
        "questCalibration": {
            "eye": args.eye,
            "depthWidth": depth_width,
            "depthHeight": depth_height,
            "horizontalFovDegrees": math.degrees(horizontal_fov),
            "verticalFovDegrees": math.degrees(vertical_fov),
            "observationGrid": [args.grid_x, args.grid_y],
            "qrsP95Mm": qrs_metres * 1000.0,
        },
        "dataQuality": {
            "inputEvents": len(events),
            "replayableGeometryEvents": len(evaluated),
            "candidateRows": len(candidate_rows),
            "referenceRows": len(references),
            "confirmedStableSurfaceControls": len(stable_surface_controls),
            "confirmedStableSurfaceControlsWithNominalFreeViews": sum(
                count for free_views, count in control_free_counts.items() if free_views > 0
            ),
            "stableUnknownExcludedFromControls": len(stable_unknown),
        },
        "policySummary": policy_summary,
        "decisionBoundary": [
            "this pass can reject an undersized footprint using confirmed-free opportunities",
            "this pass cannot approve a wide cone because per-frame surface-neighbour rays are not in the saved event table",
            "candidate_projection_reference is a route marker derived from the reference event and is not an independent score",
            "unknown candidates are excluded from both positive and negative controls",
            "no room coordinate or stable ID participates in any footprint formula",
        ],
    }
    (output / "adaptive_footprint_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme = (
        "ScanCover adaptive challenge footprint sweep\n\n"
        "This directory is read-only offline evidence. It does not alter Unity or the Quest.\n"
        "Read adaptive_footprint_summary.json first. A policy remains non-production-ready\n"
        "until a held-out surface-event replay measures edge/discontinuity false challenges.\n"
    )
    (output / "README.txt").write_text(readme, encoding="utf-8")
    convicted_ids = ", ".join(str(row["stableId"]) for row in convicted) or "none"
    policy_lines = "\n".join(
        "| {policy} | {caught}/{total} | not measured | no |".format(
            policy=row["policy"],
            caught=row["confirmedCurrentFalseCaughtAtTwoViews"],
            total=row["confirmedCurrentFalseControls"],
        )
        for row in policy_summary
    )
    report_md = f"""# Adaptive challenge footprint sweep

## Outcome first

The production 35 mm radius, one Quest tile plus QRS uncertainty, and 1.5 tiles
all fail the two-independent-view positive-control requirement. Two tiles plus
QRS is the first tested cone that reaches the confirmed current false surface.
It is **not approved for production** because this saved event table does not
contain the per-frame neighbouring rays required to measure edge false deletes.

Confirmed current-false control IDs: {convicted_ids}

| policy | current-false caught at >=2 views | surface false challenge | production ready |
|---|---:|---:|---:|
{policy_lines}

## Data quality

- Input free opportunities: {len(events)}
- Replayable production geometry events: {len(evaluated)}
- Stable confirmed-surface aggregate controls: {len(stable_surface_controls)}
- Confirmed-surface controls with a nominal free view: {sum(count for free_views, count in control_free_counts.items() if free_views > 0)}
- Stable unknown candidates excluded: {len(stable_unknown)}
- Device calibration: {depth_width}x{depth_height}, right-eye FOV approximately
  {math.degrees(horizontal_fov):.1f} by {math.degrees(vertical_fov):.1f} degrees

## Decision

This pass rejects the fixed-radius and ordinary one-tile-footprint hypotheses.
It only nominates two-tile cone and candidate-centric exact projection for the
next held-out replay. The next pass must load complete depth pairs and evaluate
confirmed surfaces at discontinuities; aggregate reference labels cannot be
reused as a circular false-positive score.
"""
    (output / "adaptive_footprint_report.md").write_text(report_md, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
