#!/usr/bin/env python3
"""Build one joined, read-only blame ledger for a ScanCover replay session.

The report starts from the final paper defect site and walks backwards through
paper publication, zero crossing, TSDF transaction, final-court testimony,
GunGel admission, processed depth and platform depth.  It does not declare the
Quest sensor physically wrong without an external reference.  It names the
first internally observable divergence and separately records later retention,
blocking or amplification.

No output from this tool is consumed by production.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable, Optional


STAGE_MARKER = "stage_responsibility_spatial_csv:"
VISUAL_MARKER = "visual_quality_spatial_csv:"


def rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def section(path: Path, marker: str) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    try:
        start = lines.index(marker) + 1
    except ValueError:
        return []
    payload: list[str] = []
    for line in lines[start:]:
        if not line.strip() or (line.endswith(":") and payload):
            break
        payload.append(line)
    return list(csv.DictReader(payload)) if len(payload) >= 2 else []


def number(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value: object, default: int = -1) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def finite(value: float) -> bool:
    return math.isfinite(value)


def percentile(values: Iterable[float], fraction: float) -> Optional[float]:
    clean = sorted(value for value in values if finite(value))
    if not clean:
        return None
    index = max(0, min(len(clean) - 1, math.ceil(len(clean) * fraction) - 1))
    return clean[index]


def mean(values: Iterable[float]) -> Optional[float]:
    clean = [value for value in values if finite(value)]
    return sum(clean) / len(clean) if clean else None


def fmt(value: object) -> object:
    if isinstance(value, float) and not finite(value):
        return ""
    return value


def write_csv(path: Path, output: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in output:
            writer.writerow({key: fmt(item.get(key, "")) for key in fields})


def key(row: dict[str, str]) -> tuple[int, int, int, int, int, int]:
    return tuple(integer(row.get(name)) for name in (
        "chunk_x", "chunk_y", "chunk_z", "bin_x", "bin_y", "bin_z"
    ))  # type: ignore[return-value]


def site_id(item: tuple[int, int, int, int, int, int]) -> str:
    return "c%d_%d_%d_b%d_%d_%d" % item


def signed_euler(value: float) -> float:
    if not finite(value):
        return value
    value %= 360.0
    return value - 360.0 if value > 180.0 else value


def bucket(value: float, limits: tuple[float, ...], labels: tuple[str, ...]) -> str:
    if not finite(value):
        return "unknown"
    for limit, label in zip(limits, labels):
        if value < limit:
            return label
    return labels[-1]


def point_in_padded_bounds(testimony: dict[str, str], stage: dict[str, str],
                           padding: float = 0.10,
                           prefix: str = "source") -> tuple[bool, float]:
    point = [number(testimony.get(prefix + axis.upper())) for axis in "xyz"]
    lo = [number(stage.get("local_min_" + axis + "_m")) for axis in "xyz"]
    hi = [number(stage.get("local_max_" + axis + "_m")) for axis in "xyz"]
    if not all(finite(value) for value in point + lo + hi):
        return False, math.nan
    squared = 0.0
    for i in range(3):
        delta = max(lo[i] - point[i], 0.0, point[i] - hi[i])
        squared += delta * delta
    distance = math.sqrt(squared)
    return distance <= padding, distance


def first_court_match(sample: dict[str, str], stage: dict[str, str],
                      by_source: dict[int, list[dict[str, str]]],
                      stable_id: int = -1) -> tuple[Optional[dict[str, str]], float]:
    source_frame = integer(sample.get("sourceFrame"))
    best: Optional[dict[str, str]] = None
    best_distance = math.inf
    candidates = by_source.get(source_frame, [])
    if stable_id > 0:
        exact = [item for item in candidates
                 if integer(item.get("stableId")) == stable_id]
        if exact:
            candidates = exact
    for testimony in candidates:
        inside, distance = point_in_padded_bounds(testimony, stage)
        if inside and distance < best_distance:
            best, best_distance = testimony, distance
    return best, best_distance


class FusionEvidenceReader:
    """Read the exact GunGel identity bound to one recorded production ray."""

    def __init__(self, session: Path):
        self.session = session
        self.frame_dir = session / "fusion_inputs" / "frames"
        manifest = rows(session / "fusion_inputs" / "manifest.csv")
        self.meta_by_key = {
            (integer(item.get("attemptIndex")), integer(item.get("sourceFrame"))):
                item.get("metaFile", "")
            for item in manifest if item.get("metaFile")
        }
        self.cache: dict[str, tuple[dict, bytes, bytes]] = {}

    def sample(self, item: dict[str, str]) -> dict[str, object]:
        key_value = (integer(item.get("attemptIndex")),
                     integer(item.get("sourceFrame")))
        meta_name = self.meta_by_key.get(key_value, "")
        u, v = number(item.get("u")), number(item.get("v"))
        if not meta_name or not finite(u) or not finite(v):
            return {}
        try:
            if meta_name not in self.cache:
                meta = json.loads((self.frame_dir / meta_name).read_text(
                    encoding="utf-8-sig"))
                admission = meta.get("gunGelAdmission", {})
                identity_name = admission.get("finalIdentityFile",
                    admission.get("identityFile", ""))
                correspondence_name = admission.get("correspondenceFile", "")
                identity = (self.frame_dir / identity_name).read_bytes()
                correspondence = (self.frame_dir / correspondence_name).read_bytes()
                self.cache[meta_name] = (meta, identity, correspondence)
            meta, identity, correspondence = self.cache[meta_name]
            admission = meta["gunGelAdmission"]
            depth = meta["textures"]["depth"]
            width, height = int(depth["width"]), int(depth["height"])
            px = min(max(int(u * width), 0), width - 1)
            py = min(max(int(v * height), 0), height - 1)
            stride = max(int(admission["pixelStride"]), 1)
            gx = min(max(px // stride, 0), int(admission["gridX"]) - 1)
            gy = min(max(py // stride, 0), int(admission["gridY"]) - 1)
            index = gy * int(admission["gridX"]) + gx
            identity_stride = int(admission["identityStrideBytes"])
            candidate, stable, dual, opposition = struct.unpack_from(
                "<4I", identity, index * identity_stride)
            correspondence_stride = int(admission["correspondenceStrideBytes"])
            flags = int(round(max(struct.unpack_from(
                "<4f", correspondence, index * correspondence_stride)[3], 0.0)))
            return {
                "gunGelSampleIndex": index,
                "gunGelCandidateIndex": candidate - 1 if candidate else -1,
                "gunGelStableId": stable,
                "gunGelDualSupportFrames": dual,
                "gunGelOppositionVotes": opposition,
                "gunGelEvidenceFlags": flags,
                "gunGelObservationValid": 1 if flags & (1 << 0) else 0,
                "gunGelRawAvailable": 1 if flags & (1 << 1) else 0,
                "gunGelDualAgree": 1 if flags & (1 << 2) else 0,
                "gunGelStableFound": 1 if flags & (1 << 6) else 0,
                "gunGelResidualPass": 1 if flags & (1 << 7) else 0,
                "gunGelNormalPass": 1 if flags & (1 << 8) else 0,
            }
        except (OSError, KeyError, ValueError, IndexError, struct.error,
                json.JSONDecodeError):
            return {}


def court_trace(frame_gates: list[dict[str, str]],
                testimonies: list[dict[str, str]],
                decisions: list[dict[str, str]],
                events: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    testimony_by_address: dict[int, list[dict[str, str]]] = defaultdict(list)
    for item in testimonies:
        testimony_by_address[integer(item.get("stableId"))].append(item)
        output.append({"rowType": "independent_testimony", **item})
    for item in frame_gates:
        output.append({"rowType": "frame_gate_summary", **item})
    event_index = {(integer(e.get("gunGelFrame")), integer(e.get("stableId"))): e
                   for e in events}
    for item in decisions:
        address = integer(item.get("stableId"))
        frame = integer(item.get("gunGelFrame"))
        latest = None
        for testimony in testimony_by_address.get(address, []):
            if integer(testimony.get("gunGelFrame")) <= frame:
                if latest is None or integer(testimony.get("gunGelFrame")) > integer(latest.get("gunGelFrame")):
                    latest = testimony
        event = event_index.get((frame, address), {})
        merged: dict[str, object] = {"rowType": "decision_check", **item}
        if latest:
            for name in ("sourceFrame", "attemptIndex", "role", "rangeM",
                         "incidenceAbs", "targetNormalResidualMm",
                         "angularDegPerSec", "linearMps", "motionQuality",
                         "headPitchDeg", "headYawDeg", "headRollDeg"):
                merged["latest_" + name] = latest.get(name, "")
        merged["publishedEventReason"] = event.get("reason", "")
        merged["publishedEventGeneration"] = event.get("generation", "")
        output.append(merged)
    # Legacy court sessions have events but no decision checks. Preserve the
    # evidence and make the missing detail explicit instead of silently dropping it.
    decision_keys = {(integer(d.get("gunGelFrame")), integer(d.get("stableId")))
                     for d in decisions}
    for item in events:
        if (integer(item.get("gunGelFrame")), integer(item.get("stableId"))) not in decision_keys:
            output.append({"rowType": "publication_event_legacy_or_unjoined", **item})
    return output


def build(session: Path, output: Path) -> dict[str, object]:
    paper_dir = session / "artifacts" / "paper_audit"
    surface_path = paper_dir / "production_paper_surface_ledger.txt"
    stage_rows = section(surface_path, STAGE_MARKER)
    visual_rows = section(surface_path, VISUAL_MARKER)
    visual_by_key = {key(item): item for item in visual_rows}
    quality_rows = rows(paper_dir / "production_paper_quality_timeline.csv")
    quality_by_key: dict[tuple[int, int, int, int, int, int], list[dict[str, str]]] = defaultdict(list)
    for item in quality_rows:
        quality_by_key[key(item)].append(item)

    responsibility_dir = session / "artifacts" / "tsdf_responsibility"
    endpoint_rows = rows(responsibility_dir / "endpoint_responsibility.csv")
    sample_rows = rows(responsibility_dir / "endpoint_source_samples.csv")
    endpoint_by_bin: dict[int, list[dict[str, str]]] = defaultdict(list)
    samples_by_bin: dict[int, list[dict[str, str]]] = defaultdict(list)
    for item in endpoint_rows:
        endpoint_by_bin[integer(item.get("binIndex"))].append(item)
    for item in sample_rows:
        samples_by_bin[integer(item.get("binIndex"))].append(item)
    sample_by_endpoint_role = {
        (integer(item.get("binIndex")), integer(item.get("voxelX")),
         integer(item.get("voxelY")), integer(item.get("voxelZ")),
         item.get("role", "")): item
        for item in sample_rows
    }

    motion_confirm_quality_min = 0.70
    production_config_path = session / "production_config.json"
    if production_config_path.is_file():
        try:
            production_config = json.loads(production_config_path.read_text(
                encoding="utf-8-sig"))
            motion_confirm_quality_min = float(
                production_config.get("components", {})
                .get("Genesis.RoomScan.VolumeIntegrator", {})
                .get("motionConfirmQualityMin", motion_confirm_quality_min))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    court_dir = session / "probe_shadow" / "final_buffer" / "final_surface_court"
    testimonies = rows(court_dir / "independent_testimonies.csv")
    frame_gates = rows(court_dir / "frame_gate_summary.csv")
    decisions = rows(court_dir / "decision_checks.csv")
    events = rows(court_dir / "runtime_events.csv")
    testimony_by_source: dict[int, list[dict[str, str]]] = defaultdict(list)
    events_by_stable: dict[int, list[dict[str, str]]] = defaultdict(list)
    for item in testimonies:
        testimony_by_source[integer(item.get("sourceFrame"))].append(item)
    for item in events:
        events_by_stable[integer(item.get("stableId"))].append(item)
    fusion_evidence = FusionEvidenceReader(session)

    voxel_size = 0.05
    schema_path = responsibility_dir / "schema.json"
    if schema_path.is_file():
        try:
            voxel_size = float(json.loads(schema_path.read_text(
                encoding="utf-8-sig"))["voxelSizeMetres"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    # AnalyzeTsdfResponsibility numbers only bins convicted at the TSDF crossing.
    convicted = [item for item in stage_rows
                 if item.get("first_deformation_stage") == "tsdf_zero_crossing"]
    if not convicted:
        convicted = [item for item in stage_rows
                     if number(item.get("crossing_plane_rms_vox"), 0.0) >= 0.20]
    tsdf_bin_index = {key(item): index for index, item in enumerate(convicted)}

    timeline: list[dict[str, object]] = []
    lineage: list[dict[str, object]] = []
    factor_records: list[dict[str, object]] = []
    blocker_totals: Counter[str] = Counter()
    for stage in stage_rows:
        address = key(stage)
        identifier = site_id(address)
        visual = visual_by_key.get(address, {})
        bin_index = tsdf_bin_index.get(address, -1)
        endpoints = endpoint_by_bin.get(bin_index, []) if bin_index >= 0 else []
        samples = samples_by_bin.get(bin_index, []) if bin_index >= 0 else []
        quality = sorted(quality_by_key.get(address, []),
                         key=lambda item: integer(item.get("replacement_sequence")))
        first_quality = quality[0] if quality else {}
        last_quality = quality[-1] if quality else {}
        quality_candidates = [item for item in quality
                              if integer(item.get("final_plane_candidate")) != 0 and
                              finite(number(item.get("final_plane_rms_vox")))]
        final_rms_vox = number(stage.get("final_plane_rms_vox"))
        final_rms_mm = final_rms_vox * voxel_size * 1000.0
        crossing_rms_mm = number(stage.get("crossing_plane_rms_vox")) * voxel_size * 1000.0
        first_publish_candidate = integer(first_quality.get("final_plane_candidate")) != 0
        last_publish_candidate = integer(last_quality.get("final_plane_candidate")) != 0
        first_rms_mm = (number(first_quality.get("final_plane_rms_vox")) * voxel_size * 1000.0
                        if first_publish_candidate else math.nan)
        last_rms_mm = (number(last_quality.get("final_plane_rms_vox")) * voxel_size * 1000.0
                       if last_publish_candidate else math.nan)
        correction = "timeline_unavailable"
        if quality and not first_publish_candidate and last_publish_candidate:
            correction = "became_plane_comparable"
        elif quality and first_publish_candidate and not last_publish_candidate:
            correction = "lost_plane_comparability"
        elif quality and not quality_candidates:
            correction = "never_plane_comparable"
        elif quality and finite(first_rms_mm) and finite(last_rms_mm):
            delta = first_rms_mm - last_rms_mm
            correction = "flattened" if delta >= voxel_size * 100.0 else (
                "partially_flattened" if delta >= voxel_size * 25.0 else (
                    "worsened" if delta <= -voxel_size * 25.0 else "persisted"))

        site_matches: list[dict[str, str]] = []
        source_preprocess: list[float] = []
        source_sdf: list[float] = []
        incidences: list[float] = []
        ranges: list[float] = []
        source_items = samples or [{}]
        for sample in source_items:
            gun_gel = fusion_evidence.sample(sample) if sample else {}
            testimony, match_distance = first_court_match(
                sample, stage, testimony_by_source,
                integer(gun_gel.get("gunGelStableId"))) if sample else (None, math.inf)
            if testimony:
                site_matches.append(testimony)
                incidences.append(number(testimony.get("incidenceAbs")))
                ranges.append(number(testimony.get("rangeM")))
            preprocess = number(sample.get("processedMinusRawMm"))
            projective_sdf = number(sample.get("rawProjectiveSdfMm"))
            source_preprocess.append(abs(preprocess))
            source_sdf.append(abs(projective_sdf))
            row: dict[str, object] = {
                "siteId": identifier,
                "chunkX": address[0], "chunkY": address[1], "chunkZ": address[2],
                "binX": address[3], "binY": address[4], "binZ": address[5],
                "builtEpoch": stage.get("built_epoch", ""),
                "firstDeformationStage": (
                    "not_comparable" if
                    stage.get("first_deformation_stage") == "no_material_final_deformation" and
                    integer(stage.get("final_plane_candidate")) == 0
                    else stage.get("first_deformation_stage", "")),
                "crossingRmsMm": crossing_rms_mm,
                "finalRmsMm": final_rms_mm,
                "qualityTimelineSamples": len(quality),
                "qualityPlaneCandidateSamples": len(quality_candidates),
                "firstPublishedPlaneCandidate": 1 if first_publish_candidate else 0,
                "firstPublishedRmsMm": first_rms_mm,
                "lastPublishedPlaneCandidate": 1 if last_publish_candidate else 0,
                "lastPublishedRmsMm": last_rms_mm,
                "sourceFrame": sample.get("sourceFrame", ""),
                "integrationCount": sample.get("integrationCount", ""),
                "tsdfRole": sample.get("role", ""),
                "attemptIndex": sample.get("attemptIndex", ""),
                "dirtyEpoch": sample.get("dirtyEpoch", ""),
                "fusionDepthM": sample.get("fusionDepthM", ""),
                "platformRawDepthM": sample.get("platformRawDepthM", ""),
                "processedDepthM": sample.get("processedDepthM", ""),
                "processedMinusRawMm": sample.get("processedMinusRawMm", ""),
                "rawProjectiveSdfMm": sample.get("rawProjectiveSdfMm", ""),
                "angularDegPerSec": sample.get("angularDegPerSec", ""),
                "linearMps": sample.get("linearMps", ""),
                "headPitchDeg": sample.get("headPitchDeg", ""),
                "headYawDeg": sample.get("headYawDeg", ""),
                "headRollDeg": sample.get("headRollDeg", ""),
                "motionQuality": sample.get("motionQuality", ""),
                **gun_gel,
                "courtSpatialJoin": (
                    "same_source_stable_id_and_bounded" if testimony and
                    integer(gun_gel.get("gunGelStableId")) > 0 and
                    integer(testimony.get("stableId")) == integer(gun_gel.get("gunGelStableId"))
                    else "bounded_same_source_frame" if testimony else "unavailable"),
                "courtJoinDistanceM": match_distance if testimony else math.nan,
                "courtStableId": testimony.get("stableId", "") if testimony else "",
                "courtRole": testimony.get("role", "") if testimony else "",
                "courtRangeM": testimony.get("rangeM", "") if testimony else "",
                "courtIncidenceAbs": testimony.get("incidenceAbs", "") if testimony else "",
                "courtTargetResidualMm": testimony.get("targetNormalResidualMm", "") if testimony else "",
                "courtPlaneCoordinateM": testimony.get("planeCoordinateM", "") if testimony else "",
            }
            timeline.append(row)
            factor_records.append(row)

        blocked_endpoints = sum(1 for item in endpoints
                                if integer(item.get("strongestBlockIntegration"), 0) > 0)
        site_blockers: Counter[str] = Counter()
        for endpoint in endpoints:
            reason = endpoint.get("strongestBlockReason", "unknown") or "unknown"
            if reason == "fov_motion_authority":
                block_sample = sample_by_endpoint_role.get((
                    bin_index, integer(endpoint.get("voxelX")),
                    integer(endpoint.get("voxelY")), integer(endpoint.get("voxelZ")),
                    "strongest_block"), {})
                motion_quality = number(block_sample.get("motionQuality"))
                reason = ("motion_authority" if finite(motion_quality) and
                          motion_quality < motion_confirm_quality_min
                          else "fov_authority")
            site_blockers[reason] += 1
            blocker_totals[reason] += 1
        court_spreads: list[float] = []
        by_stable: dict[int, list[float]] = defaultdict(list)
        for item in site_matches:
            by_stable[integer(item.get("stableId"))].append(number(item.get("planeCoordinateM")))
        for values in by_stable.values():
            clean = [value for value in values if finite(value)]
            if len(clean) >= 2:
                court_spreads.append((max(clean) - min(clean)) * 1000.0)

        published_events: list[dict[str, str]] = []
        published_by_stable: dict[int, list[float]] = defaultdict(list)
        for stable_id in by_stable:
            for event in events_by_stable.get(stable_id, []):
                inside, _ = point_in_padded_bounds(event, stage,
                                                   prefix="published")
                # PublishedPlaneCoordinateM uses the current fitted normal and
                # therefore moves when that normal rotates around the world
                # origin. PublishedLayerCoordinateM stays on the stable-id's
                # fixed reference-normal axis and is the comparable layer value.
                coordinate = number(event.get("publishedLayerCoordinateM"))
                if inside and finite(coordinate):
                    published_events.append(event)
                    published_by_stable[stable_id].append(coordinate)
        published_spreads = [
            (max(values) - min(values)) * 1000.0
            for values in published_by_stable.values() if len(values) >= 2]
        published_spread_p95 = percentile(published_spreads, 0.95)

        first_stage = stage.get("first_deformation_stage", "not_comparable")
        if (first_stage == "no_material_final_deformation" and
                integer(stage.get("final_plane_candidate")) == 0):
            first_stage = "not_comparable"
        first_responsible = {
            "tsdf_zero_crossing": "tsdf_or_pre_tsdf_constraint_at_zero_crossing",
            "surface_nets_cell_reduction": "surface_nets_cell_reduction",
            "post_extract_transform": "post_extract_transform",
            "no_material_final_deformation": "no_material_deformation_detected",
            "not_comparable": "unresolved_not_comparable",
        }.get(first_stage, first_stage)
        amplifiers: list[str] = []
        if blocked_endpoints:
            amplifiers.append("blocked_later_tsdf_correction")
        if quality and correction in ("persisted", "worsened"):
            amplifiers.append("paper_roughness_persisted_across_commits")
        testimony_spread_p95 = percentile(court_spreads, 0.95)
        if (published_spread_p95 is not None and
                finite(published_spread_p95) and published_spread_p95 >= 10.0):
            amplifiers.append("court_published_layer_changed_ge_10mm")
        elif (testimony_spread_p95 is not None and
              finite(testimony_spread_p95) and testimony_spread_p95 >= 10.0):
            amplifiers.append("court_admitted_testimony_layer_spread_ge_10mm")
        lineage.append({
            "siteId": identifier,
            "chunkX": address[0], "chunkY": address[1], "chunkZ": address[2],
            "binX": address[3], "binY": address[4], "binZ": address[5],
            "triangles": stage.get("triangles", ""),
            "firstDeformationStage": first_stage,
            "firstResponsibleBoundary": first_responsible,
            "laterAmplifiersOrBlockers": "|".join(amplifiers) if amplifiers else "none_observed",
            "crossingRmsMm": crossing_rms_mm,
            "finalRmsMm": final_rms_mm,
            "firstPublishedRmsMm": first_rms_mm,
            "lastPublishedRmsMm": last_rms_mm,
            "publishedQualitySamples": len(quality),
            "publishedPlaneCandidateSamples": len(quality_candidates),
            "firstPublishedPlaneCandidate": 1 if first_publish_candidate else 0,
            "lastPublishedPlaneCandidate": 1 if last_publish_candidate else 0,
            "correctionOutcome": correction,
            "tsdfEndpointCount": len(endpoints),
            "blockedCorrectionEndpoints": blocked_endpoints,
            "blockedByMotionAuthority": site_blockers["motion_authority"],
            "blockedByFovAuthority": site_blockers["fov_authority"],
            "blockedByGunGel": site_blockers["gungel"],
            "blockedByOther": sum(
                count for reason, count in site_blockers.items()
                if reason not in ("motion_authority", "fov_authority", "gungel")),
            "sourceSampleCount": len(samples),
            "rawToProcessedAbsP95Mm": percentile(source_preprocess, 0.95),
            "rawProjectiveSdfAbsP95Mm": percentile(source_sdf, 0.95),
            "courtMatchedTestimonies": len(site_matches),
            "courtStableIds": len(by_stable),
            "courtWithinStableLayerSpreadP95Mm": testimony_spread_p95,
            "courtPublishedEvents": len(published_events),
            "courtPublishedStableIds": len(published_by_stable),
            "courtReplacementPublications": sum(
                1 for item in published_events
                if item.get("reason") == "replacement_published"),
            "courtPublishedLayerSpreadP95Mm": published_spread_p95,
            "incidenceAbsMin": min((v for v in incidences if finite(v)), default=math.nan),
            "rangeMedianM": median([v for v in ranges if finite(v)]) if any(finite(v) for v in ranges) else math.nan,
            "absoluteTruthBoundary": "internal_divergence_only_without_external_surface_reference",
        })

    factor_definitions = {
        "incidence_abs": lambda r: bucket(number(r.get("courtIncidenceAbs")),
            (0.30, 0.50, 0.75), ("<0.30", "0.30-0.50", "0.50-0.75", ">=0.75")),
        "range_m": lambda r: bucket(number(r.get("courtRangeM"))
            if finite(number(r.get("courtRangeM"))) else number(r.get("fusionDepthM")),
            (0.75, 1.25, 2.0), ("<0.75", "0.75-1.25", "1.25-2.0", ">=2.0")),
        "absolute_pitch_deg": lambda r: bucket(abs(signed_euler(number(r.get("headPitchDeg")))),
            (15, 35, 60), ("<15", "15-35", "35-60", ">=60")),
        "angular_speed_deg_s": lambda r: bucket(number(r.get("angularDegPerSec")),
            (5, 20, 60), ("<5", "5-20", "20-60", ">=60")),
        "linear_speed_m_s": lambda r: bucket(number(r.get("linearMps")),
            (0.02, 0.08, 0.20), ("<0.02", "0.02-0.08", "0.08-0.20", ">=0.20")),
        "raw_to_processed_abs_mm": lambda r: bucket(abs(number(r.get("processedMinusRawMm"))),
            (2, 5, 10, 20), ("<2", "2-5", "5-10", "10-20", ">=20")),
        "first_deformation_stage": lambda r: str(r.get("firstDeformationStage", "unknown")),
        "tsdf_transaction_role": lambda r: str(r.get("tsdfRole", "unknown")) or "unknown",
    }
    # Cross-buckets expose masking: an angle effect can otherwise be a distance
    # or motion effect wearing the same label.
    factor_definitions["incidence_x_range"] = lambda r: (
        factor_definitions["incidence_abs"](r) + "|" + factor_definitions["range_m"](r))
    factor_definitions["incidence_x_motion"] = lambda r: (
        factor_definitions["incidence_abs"](r) + "|" + factor_definitions["angular_speed_deg_s"](r))

    factor_output: list[dict[str, object]] = []
    for factor_name, selector in factor_definitions.items():
        groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for item in factor_records:
            groups[selector(item)].append(item)
        for label, group in sorted(groups.items()):
            factor_output.append({
                "factor": factor_name,
                "bucket": label,
                "records": len(group),
                "sites": len({str(item.get("siteId")) for item in group}),
                "finalRmsMeanMm": mean(number(item.get("finalRmsMm")) for item in group),
                "finalRmsP90Mm": percentile((number(item.get("finalRmsMm")) for item in group), 0.90),
                "rawToProcessedAbsMeanMm": mean(abs(number(item.get("processedMinusRawMm"))) for item in group),
                "rawProjectiveSdfAbsMeanMm": mean(abs(number(item.get("rawProjectiveSdfMm"))) for item in group),
                "interpretation": "association_not_single_factor_conviction",
            })

    trace = court_trace(frame_gates, testimonies, decisions, events)
    timeline_fields = [
        "siteId", "chunkX", "chunkY", "chunkZ", "binX", "binY", "binZ",
        "builtEpoch", "firstDeformationStage", "crossingRmsMm", "finalRmsMm",
        "qualityTimelineSamples", "qualityPlaneCandidateSamples",
        "firstPublishedPlaneCandidate", "firstPublishedRmsMm",
        "lastPublishedPlaneCandidate", "lastPublishedRmsMm",
        "sourceFrame", "integrationCount", "tsdfRole", "attemptIndex", "dirtyEpoch",
        "fusionDepthM", "platformRawDepthM", "processedDepthM", "processedMinusRawMm",
        "rawProjectiveSdfMm", "angularDegPerSec", "linearMps", "headPitchDeg",
        "headYawDeg", "headRollDeg", "motionQuality", "courtSpatialJoin",
        "gunGelSampleIndex", "gunGelCandidateIndex", "gunGelStableId",
        "gunGelDualSupportFrames", "gunGelOppositionVotes", "gunGelEvidenceFlags",
        "gunGelObservationValid", "gunGelRawAvailable", "gunGelDualAgree",
        "gunGelStableFound", "gunGelResidualPass", "gunGelNormalPass",
        "courtJoinDistanceM", "courtStableId", "courtRole", "courtRangeM",
        "courtIncidenceAbs", "courtTargetResidualMm", "courtPlaneCoordinateM",
    ]
    lineage_fields = list(lineage[0].keys()) if lineage else ["siteId"]
    factor_fields = list(factor_output[0].keys()) if factor_output else ["factor", "bucket"]
    trace_fields = sorted({field for item in trace for field in item},
                          key=lambda name: (name != "rowType", name)) or ["rowType"]
    write_csv(output / "surface_site_timeline.csv", timeline, timeline_fields)
    write_csv(output / "court_decision_trace.csv", trace, trace_fields)
    write_csv(output / "roughness_lineage.csv", lineage, lineage_fields)
    write_csv(output / "factor_responsibility.csv", factor_output, factor_fields)

    missing = []
    for name, present in (
        ("paper_stage_responsibility", bool(stage_rows)),
        ("paper_quality_timeline", bool(quality_rows)),
        ("tsdf_endpoint_responsibility", bool(endpoint_rows)),
        ("tsdf_source_samples", bool(sample_rows)),
        ("court_independent_testimonies", bool(testimonies)),
        ("court_frame_gate_summary", bool(frame_gates)),
        ("court_decision_checks", bool(decisions)),
    ):
        if not present:
            missing.append(name)
    final_comparable_defects = sum(
        1 for item in stage_rows
        if integer(item.get("final_plane_candidate")) != 0 and
        number(item.get("final_plane_rms_vox"), 0.0) >= 0.20)
    final_comparable_flat = sum(
        1 for item in stage_rows
        if integer(item.get("final_plane_candidate")) != 0 and
        number(item.get("final_plane_rms_vox"), 0.0) < 0.20)
    final_not_comparable = sum(
        1 for item in stage_rows
        if integer(item.get("final_plane_candidate")) == 0)
    summary: dict[str, object] = {
        "schema": "scancover.full_chain_responsibility.v1",
        "session": str(session),
        "authority": "read_only_offline_diagnostic",
        "siteGrain": "committed paper chunk 4x4x4 diagnostic bin",
        "siteCount": len(stage_rows),
        "finalComparableDefectSites": final_comparable_defects,
        "finalComparableFlatSites": final_comparable_flat,
        "finalNotPlaneComparableSites": final_not_comparable,
        "timelineRows": len(timeline),
        "courtTraceRows": len(trace),
        "qualityTimelineRows": len(quality_rows),
        "motionConfirmQualityMin": motion_confirm_quality_min,
        "blockedCorrectionTotals": dict(sorted(blocker_totals.items())),
        "missingLanes": missing,
        "completeForNewCapture": len(missing) == 0,
        "outputs": {
            "surfaceTimeline": "surface_site_timeline.csv",
            "courtTrace": "court_decision_trace.csv",
            "roughnessLineage": "roughness_lineage.csv",
            "factorResponsibility": "factor_responsibility.csv",
        },
        "convictionRule": "name earliest internally observed material divergence; keep later retention, blocking and amplification separate",
        "angleRule": "incidence, range, pitch and motion are joint factors; no single-factor conviction without cross-bucket separation",
        "truthBoundary": "platform pre-QRS depth is the earliest accessible depth, not sensor raw or an external physical surface reference",
        "productionEffect": "none",
    }
    (output / "responsibility_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    session = args.session.resolve()
    output = args.output.resolve() if args.output else (
        session / "artifacts" / "responsibility_ledger")
    try:
        summary = build(session, output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
