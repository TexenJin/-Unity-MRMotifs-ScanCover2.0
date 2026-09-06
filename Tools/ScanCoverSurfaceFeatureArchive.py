#!/usr/bin/env python3
"""Build and aggregate ScanCover's diagnostic-only surface feature archive.

The archive deliberately keeps world cells and fingerprint IDs only as session-local
provenance.  Cross-session phenotype summaries are grouped exclusively from geometry,
view, motion, and evidence features.  Nothing written by this tool is a production
admission input.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


SCHEMA = "scancover.surface_feature_archive.v2"
CATALOG_SCHEMA = "scancover.surface_feature_catalog.v2"
AUTHORITY = "diagnostic_only"


def number(row: dict[str, str], name: str, default: float = 0.0) -> float:
    try:
        return float(row.get(name, ""))
    except (TypeError, ValueError):
        return default


def integer(row: dict[str, str], name: str, default: int = 0) -> int:
    try:
        return int(row.get(name, ""))
    except (TypeError, ValueError):
        return default


def boolean(row: dict[str, str], name: str) -> bool:
    return str(row.get(name, "")).strip().lower() in {"1", "true", "yes"}


def vector(row: dict[str, str], x: str, y: str, z: str) -> tuple[float, float, float]:
    return number(row, x), number(row, y), number(row, z)


def normalized(value: Sequence[float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in value))
    if length <= 1e-10:
        return 0.0, 1.0, 0.0
    return tuple(component / length for component in value)  # type: ignore[return-value]


def quantile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return None
    index = round((len(ordered) - 1) * fraction)
    return ordered[max(0, min(len(ordered) - 1, index))]


def maximum(values: Iterable[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else None


def text_number(value: float | int | None) -> str | int:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return value


Cell = tuple[int, int, int, int]
RecordKey = tuple[int, int, int, int, int, int]


def cell_of(row: dict[str, str]) -> Cell:
    return (
        integer(row, "cellX"), integer(row, "cellY"),
        integer(row, "cellZ"), integer(row, "axis"),
    )


@dataclass
class Patch:
    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    radius_mm: float
    frame: int


@dataclass
class Record:
    key: RecordKey
    final_verdict: str = "superseded"
    accepted_frame: int = -1
    first_frame: int = 2**31 - 1
    last_frame: int = -1
    snapshot_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    snapshot_normal: tuple[float, float, float] = (0.0, 1.0, 0.0)
    has_snapshot: bool = False
    patches: list[Patch] = field(default_factory=list)
    support_residual_mm: list[float] = field(default_factory=list)
    support_angle_deg: list[float] = field(default_factory=list)
    view_radius: list[float] = field(default_factory=list)
    motion_quality: list[float] = field(default_factory=list)
    free_gap_mm: list[float] = field(default_factory=list)
    samples: int = 0
    support_samples: int = 0
    independent_support_samples: int = 0
    recovery_samples: int = 0
    challenge_samples: int = 0
    independent_challenge_samples: int = 0
    correlated_challenge_samples: int = 0
    accept_transitions: int = 0
    reject_transitions: int = 0
    reopen_transitions: int = 0
    fingerprint_hit_samples: int = 0
    fingerprint_edge_hit_samples: int = 0
    max_support_views: int = 0
    max_challenge_votes: int = 0
    max_challenge_views: int = 0
    max_recovery_views: int = 0
    max_baseline_mm: float = 0.0
    max_spread_deg: float = 0.0
    max_free_gap_mm: float = 0.0
    post_verdict_outcome: str = ""
    post_verdict_window_complete: bool = False
    post_observed_frames: int = 0
    post_contact_frames: int = 0
    post_ray_frames: int = 0
    post_reliable_free_frames: int = 0
    post_free_views: int = 0
    post_support_frames: int = 0
    post_support_views: int = 0
    post_reconfirmed: bool = False
    post_reopened: bool = False
    post_followup_span_frames: int = 0

    @property
    def cell(self) -> Cell:
        return self.key[:4]  # type: ignore[return-value]

    @property
    def fingerprint_id(self) -> int:
        return self.key[4]

    @property
    def generation(self) -> int:
        return self.key[5]


def archive_key(key: RecordKey) -> str:
    return ":".join(str(value) for value in key)


def get_record(records: dict[RecordKey, Record], by_cell: dict[Cell, list[Record]],
               key: RecordKey) -> Record:
    if key not in records:
        records[key] = Record(key)
        by_cell[key[:4]].append(records[key])  # type: ignore[index]
    return records[key]


def resolve_event_record(records: dict[RecordKey, Record],
                         by_cell: dict[Cell, list[Record]], row: dict[str, str]) -> Record:
    cell = cell_of(row)
    fingerprint_id = integer(row, "fingerprintId")
    generation = integer(row, "fingerprintGeneration")
    frame = integer(row, "gunGelFrame")
    if fingerprint_id > 0 and generation > 0:
        return get_record(records, by_cell, (*cell, fingerprint_id, generation))
    future = [candidate for candidate in by_cell.get(cell, [])
              if candidate.fingerprint_id > 0 and candidate.accepted_frame >= frame]
    if future:
        return min(future, key=lambda candidate: candidate.accepted_frame)
    return get_record(records, by_cell, (*cell, 0, 0))


def orientation(normal: Sequence[float]) -> str:
    nx, ny, nz = normalized(normal)
    absolute = (abs(nx), abs(ny), abs(nz))
    if absolute[1] >= 0.85:
        return "horizontal_up" if ny >= 0 else "horizontal_down"
    if absolute[0] >= 0.85:
        return "vertical_x_positive" if nx >= 0 else "vertical_x_negative"
    if absolute[2] >= 0.85:
        return "vertical_z_positive" if nz >= 0 else "vertical_z_negative"
    return "oblique_or_structured"


def outcome(record: Record) -> str:
    if record.fingerprint_id <= 0:
        return "unconfirmed_hold"
    if record.final_verdict == "reject" or record.reject_transitions > 0:
        return "free_space_refuted"
    if record.reopen_transitions > 0 or record.generation > 1:
        return "recovered_generation"
    if record.final_verdict == "superseded":
        return "superseded_generation"
    if record.challenge_samples > 0:
        return "challenged_supported"
    return "supported"


def geometry(record: Record) -> dict[str, object]:
    if record.patches:
        count = len(record.patches)
        center = tuple(sum(patch.center[i] for patch in record.patches) / count
                       for i in range(3))
        normal = normalized(tuple(sum(patch.normal[i] for patch in record.patches)
                                  for i in range(3)))
    else:
        center = record.snapshot_center
        normal = normalized(record.snapshot_normal)
    radii = [patch.radius_mm for patch in record.patches]
    spreads = []
    normal_offsets = []
    tangent_extents = []
    for patch in record.patches:
        dot_normal = max(-1.0, min(1.0, sum(normal[i] * patch.normal[i] for i in range(3))))
        spreads.append(math.degrees(math.acos(dot_normal)))
        delta = tuple(patch.center[i] - center[i] for i in range(3))
        along = sum(delta[i] * normal[i] for i in range(3))
        tangent = tuple(delta[i] - normal[i] * along for i in range(3))
        normal_offsets.append(along)
        tangent_extents.append(math.sqrt(sum(value * value for value in tangent))
                               + patch.radius_mm * 0.001)
    thickness_mm = ((max(normal_offsets) - min(normal_offsets)) * 1000.0
                    if normal_offsets else 0.0)
    tangential_span_mm = max(tangent_extents, default=0.0) * 2000.0
    return {
        "center": center, "normal": normal, "radii": radii, "spreads": spreads,
        "thickness_mm": thickness_mm, "tangential_span_mm": tangential_span_mm,
    }


ARCHIVE_FIELDS = [
    "sessionId", "archiveKey", "cellX", "cellY", "cellZ", "axis", "fingerprintId",
    "fingerprintGeneration", "acceptedFrame", "firstFrame", "lastFrame", "finalVerdict",
    "evidenceOutcome", "orientationFamily", "centerX", "centerY", "centerZ", "normalX",
    "normalY", "normalZ", "patchCount", "patchRadiusP50Mm", "patchRadiusP95Mm",
    "patchRadiusMaxMm", "patchNormalSpreadP50Deg", "patchNormalSpreadP95Deg",
    "patchNormalSpreadMaxDeg", "patchThicknessMm", "patchTangentialSpanMm", "samples",
    "supportSamples", "independentSupportSamples", "recoverySamples", "challengeSamples",
    "independentChallengeSamples", "correlatedChallengeSamples", "acceptTransitions",
    "rejectTransitions", "reopenTransitions", "fingerprintHitSamples",
    "fingerprintEdgeHitSamples", "maxSupportViews", "maxChallengeVotes", "maxChallengeViews",
    "maxRecoveryViews", "supportResidualP50Mm", "supportResidualP95Mm",
    "supportResidualMaxMm", "supportAngleP50Deg", "supportAngleP95Deg",
    "supportAngleMaxDeg", "viewRadiusP50", "viewRadiusP95", "motionQualityP05",
    "motionQualityP50", "freeGapP50Mm", "freeGapP95Mm", "maxBaselineMm", "maxSpreadDeg",
    "maxFreeGapMm", "postVerdictOutcome", "postVerdictWindowComplete",
    "postVerdictObservedFrames", "postVerdictContactFrames", "postVerdictRayFrames",
    "postVerdictReliableFreeFrames", "postVerdictFreeViews", "postVerdictSupportFrames",
    "postVerdictSupportViews", "postVerdictReconfirmed", "postVerdictReopened",
    "postVerdictFollowupSpanFrames", "descriptorTags", "authority",
]

SAMPLE_FIELDS = [
    "sessionId", "eventRow", "archiveKey", "gunGelFrame", "platformFrame", "event", "reason",
    "sourceX", "sourceY", "sourceZ", "targetX", "targetY", "targetZ", "residualMm",
    "angleDeg", "gapMm", "viewRadius", "motionQuality", "baselineMm", "spreadDeg",
    "fingerprintPatchIndex", "footprintRadiusMm", "footprintOffsetMm", "footprintOffsetRatio",
    "association", "probeEventReference", "depthPairManifestReference", "authority",
]


def build_archive(session: Path, output: Path) -> dict[str, object]:
    session = session.resolve()
    probe = session / "probe_shadow"
    paths = {name: probe / f"verdict_{name}.csv" for name in ("cells", "events", "fingerprints")}
    followups_path = probe / "verdict_followups.csv"
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing probe ledger: " + ", ".join(missing))
    records: dict[RecordKey, Record] = {}
    by_cell: dict[Cell, list[Record]] = defaultdict(list)

    with paths["fingerprints"].open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            fingerprint_id = integer(row, "fingerprintId")
            generation = integer(row, "fingerprintGeneration")
            if fingerprint_id <= 0 or generation <= 0:
                continue
            record = get_record(records, by_cell, (*cell_of(row), fingerprint_id, generation))
            record.accepted_frame = integer(row, "fingerprintAcceptedFrame", -1)
            patch_frame = integer(row, "patchFrame", -1)
            record.first_frame = min(record.first_frame,
                                     record.accepted_frame if record.accepted_frame >= 0 else patch_frame)
            record.last_frame = max(record.last_frame, patch_frame)
            record.patches.append(Patch(
                vector(row, "centerX", "centerY", "centerZ"),
                normalized(vector(row, "normalX", "normalY", "normalZ")),
                number(row, "radiusMm"), patch_frame,
            ))

    with paths["cells"].open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            fingerprint_id = max(0, integer(row, "fingerprintId"))
            generation = max(0, integer(row, "fingerprintGeneration"))
            record = get_record(records, by_cell, (*cell_of(row), fingerprint_id, generation))
            record.final_verdict = row.get("verdict", "")
            record.accepted_frame = integer(row, "fingerprintAcceptedFrame", record.accepted_frame)
            record.first_frame = integer(row, "firstFrame", record.first_frame)
            record.last_frame = integer(row, "lastSeenFrame", record.last_frame)
            record.snapshot_center = vector(row, "centerX", "centerY", "centerZ")
            record.snapshot_normal = normalized(vector(row, "normalX", "normalY", "normalZ"))
            record.has_snapshot = True
            record.max_support_views = integer(row, "independentSupportViews")
            record.max_challenge_votes = integer(row, "challengeVotes")
            record.max_challenge_views = integer(row, "independentChallengeViews")
            record.max_recovery_views = integer(row, "recoverySupportViews")
            record.max_baseline_mm = number(row, "maxBaselineMm")
            record.max_spread_deg = number(row, "maxViewSpreadDeg")
            record.max_free_gap_mm = number(row, "maxFreeGapMm")

    samples: list[dict[str, object]] = []
    with paths["events"].open(encoding="utf-8-sig", newline="") as handle:
        for event_row, row in enumerate(csv.DictReader(handle), 1):
            record = resolve_event_record(records, by_cell, row)
            event = row.get("event", "")
            reason = row.get("reason", "")
            frame = integer(row, "gunGelFrame")
            platform_frame = integer(row, "platformFrame")
            residual = number(row, "residualMm")
            angle = number(row, "angleDeg")
            gap = number(row, "gapMm")
            view_radius = number(row, "viewRadius")
            motion_quality = number(row, "motionQuality")
            baseline = number(row, "baselineMm")
            spread = number(row, "spreadDeg")
            patch_radius = number(row, "footprintRadiusMm")
            patch_offset = number(row, "footprintOffsetMm")
            association = row.get("association", "")
            record.samples += 1
            record.first_frame = min(record.first_frame, frame)
            record.last_frame = max(record.last_frame, frame)
            record.view_radius.append(view_radius)
            record.motion_quality.append(motion_quality)
            record.max_baseline_mm = max(record.max_baseline_mm, baseline)
            record.max_spread_deg = max(record.max_spread_deg, spread)
            record.max_support_views = max(record.max_support_views, integer(row, "independentSupportViews"))
            record.max_challenge_votes = max(record.max_challenge_votes, integer(row, "challengeVotes"))
            record.max_challenge_views = max(record.max_challenge_views, integer(row, "independentChallengeViews"))
            record.max_recovery_views = max(record.max_recovery_views, integer(row, "recoverySupportViews"))
            support = event == "support_independent" or event.startswith("recovery_")
            challenge = event.startswith("challenge_")
            if support:
                record.support_samples += 1
                if event == "support_independent":
                    record.independent_support_samples += 1
                else:
                    record.recovery_samples += 1
                record.support_residual_mm.append(residual)
                record.support_angle_deg.append(angle)
            if challenge:
                record.challenge_samples += 1
                if event == "challenge_independent":
                    record.independent_challenge_samples += 1
                else:
                    record.correlated_challenge_samples += 1
                record.free_gap_mm.append(gap)
                record.max_free_gap_mm = max(record.max_free_gap_mm, gap)
            if row.get("previousVerdict") != "accept" and row.get("newVerdict") == "accept":
                record.accept_transitions += 1
            if row.get("previousVerdict") != "reject" and row.get("newVerdict") == "reject":
                record.reject_transitions += 1
            if reason == "reopened_by_independent_safe_support":
                record.reopen_transitions += 1
            if association == "fingerprint_patch_hit":
                record.fingerprint_hit_samples += 1
                if patch_radius > 0 and patch_offset / patch_radius >= 0.95:
                    record.fingerprint_edge_hit_samples += 1
            source = vector(row, "sourceX", "sourceY", "sourceZ")
            target = vector(row, "targetX", "targetY", "targetZ")
            samples.append({
                "sessionId": session.name, "eventRow": event_row,
                "archiveKey": archive_key(record.key), "gunGelFrame": frame,
                "platformFrame": platform_frame, "event": event, "reason": reason,
                "sourceX": source[0], "sourceY": source[1], "sourceZ": source[2],
                "targetX": target[0], "targetY": target[1], "targetZ": target[2],
                "residualMm": residual, "angleDeg": angle, "gapMm": gap,
                "viewRadius": view_radius, "motionQuality": motion_quality,
                "baselineMm": baseline, "spreadDeg": spread,
                "fingerprintPatchIndex": integer(row, "fingerprintPatchIndex", -1),
                "footprintRadiusMm": patch_radius, "footprintOffsetMm": patch_offset,
                "footprintOffsetRatio": patch_offset / patch_radius if patch_radius > 0 else 0.0,
                "association": association,
                "probeEventReference": f"../../probe_shadow/verdict_events.csv#row={event_row}",
                "depthPairManifestReference": f"../../depth_pairs/manifest.csv#platformFrame={platform_frame}",
                "authority": AUTHORITY,
            })

    if followups_path.is_file():
        with followups_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                fingerprint_id = integer(row, "rejectedFingerprintId")
                generation = integer(row, "rejectedFingerprintGeneration")
                if fingerprint_id <= 0 or generation <= 0:
                    continue
                record = get_record(records, by_cell,
                                    (*cell_of(row), fingerprint_id, generation))
                record.post_verdict_outcome = row.get("finalOutcome", "")
                record.post_verdict_window_complete = boolean(row, "windowComplete")
                record.post_observed_frames = integer(row, "observedFrames")
                record.post_contact_frames = integer(row, "contactFrames")
                record.post_ray_frames = integer(row, "rayIntersectionFrames")
                record.post_reliable_free_frames = integer(row, "reliableFreeSpaceFrames")
                record.post_free_views = integer(row, "independentFreeSpaceViews")
                record.post_support_frames = integer(row, "safeSupportFrames")
                record.post_support_views = integer(row, "independentSupportViews")
                record.post_reconfirmed = boolean(row, "freeSpaceReconfirmed")
                record.post_reopened = boolean(row, "reopened")
                record.post_followup_span_frames = integer(row, "followupSpanFrames")

    archive_rows: list[dict[str, object]] = []
    for record in sorted(records.values(), key=lambda item: item.key):
        geo = geometry(record)
        center = geo["center"]
        normal = geo["normal"]
        assert isinstance(center, tuple) and isinstance(normal, tuple)
        radii = geo["radii"]
        spreads = geo["spreads"]
        assert isinstance(radii, list) and isinstance(spreads, list)
        result = outcome(record)
        family = orientation(normal)
        tags = [family, result]
        if not record.patches:
            tags.append("no_fingerprint_footprint")
        elif len(record.patches) < 3:
            tags.append("sparse_fingerprint_footprint")
        elif len(record.patches) >= 12:
            tags.append("full_fingerprint_footprint")
        if record.fingerprint_edge_hit_samples:
            tags.append("footprint_edge_challenge")
        if record.max_spread_deg >= 3.0 or record.max_baseline_mm >= 80.0:
            tags.append("multi_view_observed")
        if record.post_verdict_outcome:
            tags.append("post_" + record.post_verdict_outcome)
        row = {
            "sessionId": session.name, "archiveKey": archive_key(record.key),
            "cellX": record.key[0], "cellY": record.key[1], "cellZ": record.key[2],
            "axis": record.key[3], "fingerprintId": record.fingerprint_id,
            "fingerprintGeneration": record.generation, "acceptedFrame": record.accepted_frame,
            "firstFrame": -1 if record.first_frame == 2**31 - 1 else record.first_frame,
            "lastFrame": record.last_frame, "finalVerdict": record.final_verdict,
            "evidenceOutcome": result, "orientationFamily": family,
            "centerX": center[0], "centerY": center[1], "centerZ": center[2],
            "normalX": normal[0], "normalY": normal[1], "normalZ": normal[2],
            "patchCount": len(record.patches), "patchRadiusP50Mm": text_number(quantile(radii, .5)),
            "patchRadiusP95Mm": text_number(quantile(radii, .95)),
            "patchRadiusMaxMm": text_number(maximum(radii)),
            "patchNormalSpreadP50Deg": text_number(quantile(spreads, .5)),
            "patchNormalSpreadP95Deg": text_number(quantile(spreads, .95)),
            "patchNormalSpreadMaxDeg": text_number(maximum(spreads)),
            "patchThicknessMm": geo["thickness_mm"],
            "patchTangentialSpanMm": geo["tangential_span_mm"], "samples": record.samples,
            "supportSamples": record.support_samples,
            "independentSupportSamples": record.independent_support_samples,
            "recoverySamples": record.recovery_samples, "challengeSamples": record.challenge_samples,
            "independentChallengeSamples": record.independent_challenge_samples,
            "correlatedChallengeSamples": record.correlated_challenge_samples,
            "acceptTransitions": record.accept_transitions, "rejectTransitions": record.reject_transitions,
            "reopenTransitions": record.reopen_transitions,
            "fingerprintHitSamples": record.fingerprint_hit_samples,
            "fingerprintEdgeHitSamples": record.fingerprint_edge_hit_samples,
            "maxSupportViews": record.max_support_views,
            "maxChallengeVotes": record.max_challenge_votes,
            "maxChallengeViews": record.max_challenge_views,
            "maxRecoveryViews": record.max_recovery_views,
            "supportResidualP50Mm": text_number(quantile(record.support_residual_mm, .5)),
            "supportResidualP95Mm": text_number(quantile(record.support_residual_mm, .95)),
            "supportResidualMaxMm": text_number(maximum(record.support_residual_mm)),
            "supportAngleP50Deg": text_number(quantile(record.support_angle_deg, .5)),
            "supportAngleP95Deg": text_number(quantile(record.support_angle_deg, .95)),
            "supportAngleMaxDeg": text_number(maximum(record.support_angle_deg)),
            "viewRadiusP50": text_number(quantile(record.view_radius, .5)),
            "viewRadiusP95": text_number(quantile(record.view_radius, .95)),
            "motionQualityP05": text_number(quantile(record.motion_quality, .05)),
            "motionQualityP50": text_number(quantile(record.motion_quality, .5)),
            "freeGapP50Mm": text_number(quantile(record.free_gap_mm, .5)),
            "freeGapP95Mm": text_number(quantile(record.free_gap_mm, .95)),
            "maxBaselineMm": record.max_baseline_mm, "maxSpreadDeg": record.max_spread_deg,
            "maxFreeGapMm": record.max_free_gap_mm,
            "postVerdictOutcome": record.post_verdict_outcome,
            "postVerdictWindowComplete": int(record.post_verdict_window_complete),
            "postVerdictObservedFrames": record.post_observed_frames,
            "postVerdictContactFrames": record.post_contact_frames,
            "postVerdictRayFrames": record.post_ray_frames,
            "postVerdictReliableFreeFrames": record.post_reliable_free_frames,
            "postVerdictFreeViews": record.post_free_views,
            "postVerdictSupportFrames": record.post_support_frames,
            "postVerdictSupportViews": record.post_support_views,
            "postVerdictReconfirmed": int(record.post_reconfirmed),
            "postVerdictReopened": int(record.post_reopened),
            "postVerdictFollowupSpanFrames": record.post_followup_span_frames,
            "descriptorTags": "|".join(tags),
            "authority": AUTHORITY,
        }
        archive_rows.append(row)

    archive_keys = [str(row["archiveKey"]) for row in archive_rows]
    if len(archive_keys) != len(set(archive_keys)):
        raise ValueError("duplicate session-local archiveKey")
    known_keys = set(archive_keys)
    orphan_samples = [row for row in samples if str(row["archiveKey"]) not in known_keys]
    if orphan_samples:
        raise ValueError(f"{len(orphan_samples)} samples have no archive record")
    if any(row["authority"] != AUTHORITY for row in archive_rows + samples):
        raise ValueError("surface archive acquired production authority")

    output.mkdir(parents=True, exist_ok=True)
    with (output / "surface_feature_archive.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, ARCHIVE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(archive_rows)
    with (output / "surface_feature_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, SAMPLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)
    outcomes = Counter(str(row["evidenceOutcome"]) for row in archive_rows)
    orientations = Counter(str(row["orientationFamily"]) for row in archive_rows)
    post_outcomes = Counter(str(row["postVerdictOutcome"]) for row in archive_rows
                            if row["postVerdictOutcome"])
    summary = {
        "schema": SCHEMA, "sessionId": session.name, "authority": AUTHORITY,
        "generalizationUnit": "continuous surface and evidence features; never world coordinates or fingerprint ids",
        "records": len(archive_rows),
        "fingerprintRecords": sum(int(row["fingerprintId"]) > 0 for row in archive_rows),
        "unconfirmedCellRecords": sum(int(row["fingerprintId"]) <= 0 for row in archive_rows),
        "sampleRows": len(samples),
        "challengedRecords": sum(int(row["challengeSamples"]) > 0 for row in archive_rows),
        "rejectTransitions": sum(int(row["rejectTransitions"]) for row in archive_rows),
        "reopenTransitions": sum(int(row["reopenTransitions"]) for row in archive_rows),
        "postVerdictRecords": sum(post_outcomes.values()),
        "postVerdictResolved": sum(count for name, count in post_outcomes.items()
                                   if not name.startswith("unknown_")),
        "postVerdictUnknown": sum(count for name, count in post_outcomes.items()
                                  if name.startswith("unknown_")),
        "postVerdictOutcomes": dict(sorted(post_outcomes.items())),
        "outcomes": dict(sorted(outcomes.items())), "orientations": dict(sorted(orientations.items())),
        "contractValidation": {
            "uniqueArchiveKeys": True, "orphanSamples": 0,
            "authority": AUTHORITY, "productionConsumers": 0,
        },
    }
    (output / "surface_feature_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "schema.json").write_text(json.dumps({
        "schema": SCHEMA,
        "authority": "none; this archive is never consumed by production",
        "identity": "sessionId plus cell plus fingerprint generation is an audit key only",
        "generalization": "cross-session analysis groups continuous geometry, view, motion and evidence features; coordinates and ids are forbidden as reusable blacklist keys",
        "postVerdict": "optional verdict_followups rows distinguish confirmed false surfaces, support reestablishment, conflicts and explicit unknowns after Reject",
        "missingValues": "empty numeric values mean unavailable evidence, not zero",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.txt").write_text(
        "ScanCover 表面特征隔离仓\n"
        "- 会话内坐标和指纹只用于追溯，不得跨会话充当黑名单。\n"
        "- 跨会话学习单位是连续特征、证据过程和裁决结果。\n"
        "- Reject 后续结局单列：自由空间复核、重新获证、冲突或明确未知。\n"
        "- 本目录零生产权限，枪胶、TSDF、纸皮和网格均不得读取。\n",
        encoding="utf-8")
    return summary


CATALOG_NUMERIC = [
    "patchRadiusP50Mm", "patchNormalSpreadP95Deg", "patchThicknessMm",
    "patchTangentialSpanMm", "supportResidualP95Mm", "supportAngleP95Deg",
    "viewRadiusP50", "motionQualityP50", "freeGapP95Mm", "maxBaselineMm", "maxSpreadDeg",
]


def phenotype_key(row: dict[str, str]) -> tuple[str, ...]:
    """Return a feature-only bucket; coordinates and IDs are intentionally absent."""
    def bucket(name: str, cuts: Sequence[float]) -> str:
        value = number(row, name, math.nan)
        if not math.isfinite(value):
            return "missing"
        for index, cut in enumerate(cuts):
            if value < cut:
                return str(index)
        return str(len(cuts))

    return (
        row.get("orientationFamily", "unknown"),
        bucket("patchThicknessMm", (2.0, 5.0, 12.0, 30.0)),
        bucket("patchNormalSpreadP95Deg", (0.5, 1.5, 4.0, 10.0)),
        bucket("supportResidualP95Mm", (1.5, 3.0, 6.0, 12.0)),
        bucket("freeGapP95Mm", (8.0, 15.0, 30.0, 60.0)),
        bucket("motionQualityP50", (0.55, 0.75, 0.90, 0.97)),
        "challenged" if integer(row, "challengeSamples") > 0 else "unchallenged",
    )


def find_archive(path: Path) -> Path:
    candidates = [
        path, path / "surface_feature_archive.csv",
        path / "artifacts" / "surface_feature_archive" / "surface_feature_archive.csv",
        path / "artifacts" / "surface_feature_archive_offline" / "surface_feature_archive.csv",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "surface_feature_archive.csv":
            return candidate
    raise FileNotFoundError(f"surface_feature_archive.csv not found under {path}")


def build_catalog(inputs: Sequence[Path], output: Path) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    sources: list[str] = []
    for item in inputs:
        archive = find_archive(item.resolve())
        sources.append(str(archive))
        with archive.open(encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("authority") != AUTHORITY:
            raise ValueError("catalog input is not diagnostic-only")
        groups[phenotype_key(row)].append(row)
    output.mkdir(parents=True, exist_ok=True)
    catalog_fields = ["phenotypeKey", "recordCount", "sessionCount", "supportedCount",
                      "challengedSupportedCount", "refutedCount", "recoveredCount",
                      "postConfirmedFalseCount", "postSupportReestablishedCount",
                      "postConflictCount", "postUnknownCount"] + [
        f"{name}P50" for name in CATALOG_NUMERIC
    ]
    summary_rows: list[dict[str, object]] = []
    for key, members in sorted(groups.items()):
        counts = Counter(row.get("evidenceOutcome", "unknown") for row in members)
        post_counts = Counter(row.get("postVerdictOutcome", "") for row in members)
        summary_row: dict[str, object] = {
            "phenotypeKey": "|".join(key), "recordCount": len(members),
            "sessionCount": len({row.get("sessionId", "") for row in members}),
            "supportedCount": counts["supported"],
            "challengedSupportedCount": counts["challenged_supported"],
            "refutedCount": counts["free_space_refuted"],
            "recoveredCount": counts["recovered_generation"],
            "postConfirmedFalseCount": post_counts["confirmed_false_surface"],
            "postSupportReestablishedCount": post_counts["support_reestablished"],
            "postConflictCount": post_counts["support_reestablished_after_conflict"],
            "postUnknownCount": sum(count for name, count in post_counts.items()
                                    if name.startswith("unknown_")),
        }
        for name in CATALOG_NUMERIC:
            values = [number(row, name, math.nan) for row in members]
            summary_row[f"{name}P50"] = text_number(quantile(values, .5))
        summary_rows.append(summary_row)
    with (output / "surface_phenotype_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, catalog_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output / "surface_feature_catalog.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, ARCHIVE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    post_outcomes = Counter(row.get("postVerdictOutcome", "") for row in rows
                            if row.get("postVerdictOutcome", ""))
    summary = {
        "schema": CATALOG_SCHEMA, "authority": AUTHORITY,
        "sourceArchives": sources, "sourceSessions": len({row.get("sessionId", "") for row in rows}),
        "records": len(rows), "phenotypes": len(groups),
        "postVerdictOutcomes": dict(sorted(post_outcomes.items())),
        "groupingContract": "feature-only; cell coordinates, centers, fingerprint IDs and archive keys are forbidden",
        "contractValidation": {
            "featureOnlyGrouping": True, "productionConsumers": 0,
            "forbiddenGroupingFields": ["cellX", "cellY", "cellZ", "centerX", "centerY",
                                         "centerZ", "fingerprintId", "archiveKey"],
        },
    }
    (output / "catalog_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def self_test() -> None:
    row_a = {
        "orientationFamily": "vertical_z_positive", "patchThicknessMm": "3",
        "patchNormalSpreadP95Deg": "1", "supportResidualP95Mm": "2",
        "freeGapP95Mm": "20", "motionQualityP50": ".9", "challengeSamples": "2",
        "cellX": "1", "cellY": "2", "cellZ": "3", "fingerprintId": "7",
    }
    row_b = dict(row_a, cellX="999", cellY="-400", fingerprintId="87654")
    assert phenotype_key(row_a) == phenotype_key(row_b), "coordinates leaked into phenotype key"
    row_c = dict(row_a, patchThicknessMm="40")
    assert phenotype_key(row_a) != phenotype_key(row_c), "geometry feature failed to affect phenotype"
    with tempfile.TemporaryDirectory(prefix="scancover_surface_feature_test_") as temp:
        root = Path(temp)
        session = root / "session"
        probe = session / "probe_shadow"
        probe.mkdir(parents=True)
        fingerprint_fields = [
            "cellX", "cellY", "cellZ", "axis", "fingerprintId",
            "fingerprintGeneration", "fingerprintAcceptedFrame", "patchIndex",
            "patchFrame", "centerX", "centerY", "centerZ", "normalX", "normalY",
            "normalZ", "radiusMm",
        ]
        cell_fields = [
            "cellX", "cellY", "cellZ", "axis", "verdict", "centerX", "centerY",
            "centerZ", "normalX", "normalY", "normalZ", "firstFrame", "lastSeenFrame",
            "independentSupportViews", "challengeVotes", "independentChallengeViews",
            "recoverySupportViews", "maxBaselineMm", "maxViewSpreadDeg", "maxFreeGapMm",
            "fingerprintId", "fingerprintGeneration", "fingerprintAcceptedFrame",
        ]
        event_fields = [
            "gunGelFrame", "platformFrame", "event", "cellX", "cellY", "cellZ", "axis",
            "previousVerdict", "newVerdict", "reason", "sourceX", "sourceY", "sourceZ",
            "targetX", "targetY", "targetZ", "residualMm", "angleDeg", "gapMm",
            "viewRadius", "motionQuality", "baselineMm", "spreadDeg",
            "independentSupportViews", "challengeVotes", "independentChallengeViews",
            "recoverySupportViews", "fingerprintId", "fingerprintGeneration",
            "fingerprintPatchIndex", "footprintRadiusMm", "footprintOffsetMm", "association",
        ]
        followup_fields = [
            "cellX", "cellY", "cellZ", "axis", "rejectedFingerprintId",
            "rejectedFingerprintGeneration", "windowComplete", "finalOutcome",
            "followupSpanFrames", "observedFrames", "contactFrames",
            "rayIntersectionFrames", "reliableFreeSpaceFrames", "independentFreeSpaceViews",
            "safeSupportFrames", "independentSupportViews", "freeSpaceReconfirmed", "reopened",
        ]
        synthetic = {
            "cellX": 1, "cellY": 2, "cellZ": 3, "axis": 4,
            "fingerprintId": 7, "fingerprintGeneration": 1,
            "fingerprintAcceptedFrame": 10,
        }
        with (probe / "verdict_fingerprints.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fingerprint_fields)
            writer.writeheader()
            writer.writerow(dict(synthetic, patchIndex=0, patchFrame=10,
                                 centerX=0, centerY=0, centerZ=1,
                                 normalX=0, normalY=0, normalZ=1, radiusMm=8))
        with (probe / "verdict_cells.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, cell_fields)
            writer.writeheader()
            writer.writerow(dict(synthetic, verdict="reject", centerX=0, centerY=0,
                                 centerZ=1, normalX=0, normalY=0, normalZ=1,
                                 firstFrame=1, lastSeenFrame=30, independentSupportViews=2,
                                 challengeVotes=3, independentChallengeViews=2,
                                 recoverySupportViews=0, maxBaselineMm=100,
                                 maxViewSpreadDeg=5, maxFreeGapMm=80))
        with (probe / "verdict_events.csv").open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, event_fields).writeheader()
        with (probe / "verdict_followups.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, followup_fields)
            writer.writeheader()
            writer.writerow({
                "cellX": 1, "cellY": 2, "cellZ": 3, "axis": 4,
                "rejectedFingerprintId": 7, "rejectedFingerprintGeneration": 1,
                "windowComplete": 1, "finalOutcome": "confirmed_false_surface",
                "followupSpanFrames": 15, "observedFrames": 4, "contactFrames": 0,
                "rayIntersectionFrames": 4, "reliableFreeSpaceFrames": 3,
                "independentFreeSpaceViews": 2, "safeSupportFrames": 0,
                "independentSupportViews": 0, "freeSpaceReconfirmed": 1, "reopened": 0,
            })
        built = build_archive(session, root / "built")
        assert built["postVerdictOutcomes"] == {"confirmed_false_surface": 1}
        with (root / "built" / "surface_feature_archive.csv").open(
                encoding="utf-8", newline="") as handle:
            built_rows = list(csv.DictReader(handle))
        assert len(built_rows) == 1
        assert built_rows[0]["postVerdictOutcome"] == "confirmed_false_surface"
        assert built_rows[0]["postVerdictFreeViews"] == "2"

        archive_a = root / "a" / "surface_feature_archive.csv"
        archive_b = root / "b" / "surface_feature_archive.csv"
        archive_a.parent.mkdir(parents=True)
        archive_b.parent.mkdir(parents=True)
        full_a = {name: "" for name in ARCHIVE_FIELDS}
        full_b = {name: "" for name in ARCHIVE_FIELDS}
        full_a.update(row_a, sessionId="a", evidenceOutcome="challenged_supported",
                      postVerdictOutcome="support_reestablished", authority=AUTHORITY)
        full_b.update(row_b, sessionId="b", evidenceOutcome="free_space_refuted",
                      postVerdictOutcome="confirmed_false_surface", authority=AUTHORITY)
        for path, row in ((archive_a, full_a), (archive_b, full_b)):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, ARCHIVE_FIELDS)
                writer.writeheader()
                writer.writerow(row)
        result = build_catalog([archive_a, archive_b], root / "catalog")
        assert result["phenotypes"] == 1 and result["records"] == 2
        assert result["postVerdictOutcomes"] == {
            "confirmed_false_surface": 1, "support_reestablished": 1,
        }
    print("self-test: PASS (feature-only grouping; coordinates and IDs ignored)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build one session archive")
    build.add_argument("session", type=Path)
    build.add_argument("--output", type=Path)
    catalog = sub.add_parser("catalog", help="aggregate one or more archives")
    catalog.add_argument("inputs", type=Path, nargs="+")
    catalog.add_argument("--output", type=Path, required=True)
    sub.add_parser("self-test", help="verify feature-only grouping contract")
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "build":
        destination = args.output or args.session / "artifacts" / "surface_feature_archive_offline"
        print(json.dumps(build_archive(args.session, destination), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(build_catalog(args.inputs, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
