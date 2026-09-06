#!/usr/bin/env python3
"""Attribute paper coverage failures to the recorded production stage.

This tool is deliberately offline.  It consumes the immutable replay-session
payload that ScanReplaySessionPackage already writes; it never adds atomics,
readbacks, or buffers to the Quest fusion hot path.

For every direct paper failure cell it reprojects the cell into every accepted
right-eye fusion frame, samples the recorded production depth, then reads the
exact GunGel correspondence selected by the same pixel/grid mapping used by
VolumeIntegration.compute.  The result separates:

* no recorded surface observation at that cell;
* the exact GunGel evidence bit that rejected the observation;
* GunGel-authorised observations that still failed to become a TSDF crossing;
* paper maturity and paper topology failures downstream of TSDF.

The projection/AABB association is exact with respect to the recorded frame
matrices and pixel-grid contract, but a final cell-level cause remains a replay
attribution: one paper cell spans several TSDF voxels and is sampled at its
centre plus eight corners.  The output records that boundary explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence


FUSION_AUTHORITY = 0x3F
STATUS_LABELS = {
    0: "accepted",
    1: "observation_invalid",
    2: "platform_raw_missing",
    3: "dual_testimony_disagree",
    4: "stable_candidate_match_missing_legacy",
    5: "stable_candidate_dual_immature",
    6: "stable_candidate_opposed",
    7: "pixel_candidate_locality_or_normal_mismatch",
    8: "stable_candidate_absent",
    9: "stable_candidate_residual_mismatch",
    10: "stable_candidate_normal_mismatch",
    11: "stable_candidate_residual_and_normal_mismatch",
}
STATUS_CODES = tuple(sorted(STATUS_LABELS))
DIRECT_FAILURE_CAUSES = {
    "raw_crossing_insufficient",
    "raw_not_mature",
    "face_parity_open",
}
ABSENCE_CONTEXT_LABELS = (
    "promotion_authority_safe",
    "edge_risk",
    "temporal_risk",
    "outer_view_risk",
    "motion_risk",
)
LOCALITY_CONTEXT_LABELS = (
    "locality_distance",
    "sample_normal_invalid",
    "depth_normal_invalid",
    "normal_disagree",
)


def has_promotion_hard_edge_risk(edge_reason: int) -> bool:
    """Mirror GunGelEvidenceShadow.compute promotion edge semantics."""
    skirt = (edge_reason & (1 << 2)) != 0
    cross_eye = (edge_reason & (1 << 6)) != 0
    rejected_edge = (edge_reason & (1 << 7)) != 0
    grazing = (edge_reason & (1 << 4)) != 0
    grazing_rescued = (edge_reason & (1 << 10)) != 0
    return skirt or cross_eye or rejected_edge or (grazing and not grazing_rescued)
EDGE_REASON_LABELS = {
    0: "span_jump",
    1: "gap",
    2: "skirt",
    3: "plane_trusted",
    4: "grazing",
    5: "dual_cluster",
    6: "cross_eye",
    7: "rejected_edge",
    8: "grazing_span",
    9: "grazing_plane_supported",
    10: "grazing_rescued",
}


@dataclass
class Cell:
    row: dict[str, str]
    center: tuple[float, float, float]
    counts: Counter = field(default_factory=Counter)
    visible_frames: int = 0
    depth_frames: int = 0
    local_surface_frames: int = 0
    accepted_frames: int = 0
    first_surface_sequence: Optional[int] = None
    last_surface_sequence: Optional[int] = None
    first_accepted_sequence: Optional[int] = None
    last_accepted_sequence: Optional[int] = None
    surface_probe_mask: int = 0
    accepted_probe_mask: int = 0
    surface_probe_votes: list[int] = field(default_factory=lambda: [0] * 9)
    accepted_probe_votes: list[int] = field(default_factory=lambda: [0] * 9)
    absent_context_counts: Counter = field(default_factory=Counter)
    locality_context_counts: Counter = field(default_factory=Counter)
    neighbour_rescue_counts: Counter = field(default_factory=Counter)


def matrix(values: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    if len(values) != 16:
        raise ValueError(f"matrix requires 16 values, got {len(values)}")
    return tuple(tuple(float(values[r * 4 + c]) for c in range(4)) for r in range(4))


def mat_vec(m: Sequence[Sequence[float]], v: Sequence[float]) -> tuple[float, float, float, float]:
    return tuple(sum(m[r][c] * v[c] for c in range(4)) for r in range(4))  # type: ignore[return-value]


def mat3_vec(m: Sequence[Sequence[float]], v: Sequence[float]) -> tuple[float, float, float]:
    return tuple(sum(m[r][c] * v[c] for c in range(3)) for r in range(3))  # type: ignore[return-value]


def norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return norm(tuple(a[i] - b[i] for i in range(3)))


def normalize(v: Sequence[float]) -> tuple[float, float, float]:
    length = norm(v)
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def transform_point(m: Sequence[Sequence[float]], p: Sequence[float]) -> tuple[float, float, float]:
    h = mat_vec(m, (p[0], p[1], p[2], 1.0))
    if abs(h[3]) <= 1e-12:
        return (h[0], h[1], h[2])
    return (h[0] / h[3], h[1] / h[3], h[2] / h[3])


def project_world(p: Sequence[float], view, projection) -> Optional[tuple[float, float, float]]:
    view_p = mat_vec(view, (p[0], p[1], p[2], 1.0))
    h = mat_vec(projection, view_p)
    if abs(h[3]) <= 1e-12:
        return None
    return (
        (h[0] / h[3]) * 0.5 + 0.5,
        (h[1] / h[3]) * 0.5 + 0.5,
        (h[2] / h[3]) * 0.5 + 0.5,
    )


def reconstruct_world(uvx: float, uvy: float, depth_ndc: float, projection_inv, view_inv):
    hcs = (uvx * 2.0 - 1.0, uvy * 2.0 - 1.0, depth_ndc * 2.0 - 1.0, 1.0)
    local = mat_vec(projection_inv, hcs)
    world = mat_vec(view_inv, local)
    if abs(world[3]) <= 1e-12:
        return (world[0], world[1], world[2])
    return (world[0] / world[3], world[1] / world[3], world[2] / world[3])


def point_aabb_distance(point: Sequence[float], center: Sequence[float], half: float) -> float:
    squared = 0.0
    for axis in range(3):
        delta = abs(point[axis] - center[axis]) - half
        if delta > 0.0:
            squared += delta * delta
    return math.sqrt(squared)


def probes(center: Sequence[float], half: float) -> Iterable[tuple[float, float, float]]:
    yield (center[0], center[1], center[2])
    for z in (-half, half):
        for y in (-half, half):
            for x in (-half, half):
                yield (center[0] + x, center[1] + y, center[2] + z)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_paper_csv(session: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    candidates = list(session.glob("artifacts/paper_audit/paper_hole_cells.csv"))
    if not candidates:
        raise FileNotFoundError(
            "paper_hole_cells.csv not found; copy the matching artifacts/paper_audit "
            "directory or pass --paper-csv"
        )
    return candidates[0]


def load_cells(path: Path) -> list[Cell]:
    result: list[Cell] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            cause = row.get("cause", "")
            if cause not in DIRECT_FAILURE_CAUSES:
                continue
            if row.get("empty_surface_halo", "0") not in ("", "0"):
                continue
            result.append(
                Cell(
                    row=row,
                    center=(
                        float(row["world_x_m"]),
                        float(row["world_y_m"]),
                        float(row["world_z_m"]),
                    ),
                )
            )
    if not result:
        raise ValueError("paper audit contains no direct failure cells")
    return result


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def depth_unorm16(payload: bytes, index: int) -> float:
    offset = index * 2
    if offset + 2 > len(payload):
        return 0.0
    return struct.unpack_from("<H", payload, offset)[0] / 65535.0


def snorm8(value: int) -> float:
    signed = value if value < 128 else value - 256
    return max(-1.0, signed / 127.0)


def normal_snorm8(payload: bytes, index: int) -> tuple[float, float, float]:
    offset = index * 4
    if offset + 4 > len(payload):
        return (0.0, 0.0, 0.0)
    return tuple(snorm8(payload[offset + axis]) for axis in range(3))  # type: ignore[return-value]


def correspondence(payload: bytes, index: int, stride: int) -> tuple[int, tuple[float, float, float]]:
    offset = index * stride
    if offset + 48 > len(payload):
        return (0, (0.0, 0.0, 0.0))
    values = struct.unpack_from("<12f", payload, offset)
    flags = int(round(max(values[3], 0.0)))
    return flags, (values[4], values[5], values[6])


def observation(payload: bytes, index: int, stride: int):
    offset = index * stride
    if offset + 64 > len(payload):
        return None
    values = struct.unpack_from("<12f", payload, offset)
    source_reason = struct.unpack_from("<4I", payload, offset + 48)
    return {
        "position": (values[0], values[1], values[2]),
        "sigma": values[3],
        "normal": (values[4], values[5], values[6]),
        "quality": values[7],
        "raw_delta": values[11],
        "packed_pixel": source_reason[0],
        "edge_reason": source_reason[1],
        "temporal_reason": source_reason[2],
        "platform_frame": source_reason[3],
    }


def stable_absence_context(obs: Optional[dict], width: int, height: int,
                           motion_quality: float) -> dict[str, bool]:
    if obs is None:
        return {label: False for label in ABSENCE_CONTEXT_LABELS}
    packed_pixel = int(obs["packed_pixel"])
    px = packed_pixel & 0xFFFF
    py = packed_pixel >> 16
    uvx = (px + 0.5) / max(width, 1)
    uvy = (py + 0.5) / max(height, 1)
    view_radius = max(abs(uvx * 2.0 - 1.0), abs(uvy * 2.0 - 1.0))
    edge_risk = has_promotion_hard_edge_risk(int(obs["edge_reason"]))
    temporal_risk = int(obs["temporal_reason"]) != 6
    outer_view_risk = view_radius >= 0.84
    motion_risk = motion_quality < 0.70
    result = {
        "promotion_authority_safe": not (
            edge_risk or temporal_risk or outer_view_risk or motion_risk
        ),
        "edge_risk": edge_risk,
        "temporal_risk": temporal_risk,
        "outer_view_risk": outer_view_risk,
        "motion_risk": motion_risk,
    }
    edge_reason = int(obs["edge_reason"])
    for bit, label in EDGE_REASON_LABELS.items():
        result[f"edge_reason:{label}"] = (edge_reason & (1 << bit)) != 0
    return result


def first_missing_flag(flags: int) -> int:
    for bit, status in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)):
        if (flags & (1 << bit)) == 0:
            return status
    return 0


def gun_gel_status(
    pixel_index: int,
    depth_position: Sequence[float],
    depth_normal: Sequence[float],
    depth_range: float,
    grid_x: int,
    grid_y: int,
    pixel_stride: int,
    width: int,
    corr_payload: bytes,
    corr_stride: int,
    obs_payload: bytes,
    obs_stride: int,
    correction,
    evidence_schema: str,
    height: int,
    motion_quality: float,
) -> tuple[int, int, dict[str, bool]]:
    px = pixel_index % width
    py = pixel_index // width
    gx = min(max(px // max(pixel_stride, 1), 0), max(grid_x - 1, 0))
    gy = min(max(py // max(pixel_stride, 1), 0), max(grid_y - 1, 0))
    sample_index = gy * grid_x + gx
    flags, _target = correspondence(corr_payload, sample_index, corr_stride)
    missing = first_missing_flag(flags)
    obs = observation(obs_payload, sample_index, obs_stride)
    if missing == 4 and evidence_schema.startswith("v2_low6_authority_high3"):
        stable_found = (flags & (1 << 6)) != 0
        residual_pass = (flags & (1 << 7)) != 0
        normal_pass = (flags & (1 << 8)) != 0
        if not stable_found:
            missing = 8
        elif not residual_pass and not normal_pass:
            missing = 11
        elif not residual_pass:
            missing = 9
        elif not normal_pass:
            missing = 10
    if missing:
        context = (
            stable_absence_context(obs, width, height, motion_quality)
            if missing == 8 else {}
        )
        return missing, sample_index, context

    if obs is None:
        return 1, sample_index, {}
    sample_position = transform_point(correction, obs["position"])
    sample_normal = mat3_vec(correction, obs["normal"])
    corrected_depth_normal = mat3_vec(correction, depth_normal)
    sample_normal_length = norm(sample_normal)
    depth_normal_length = norm(corrected_depth_normal)
    locality_limit = max(0.06, depth_range * 0.025 + max(obs["sigma"], 0.004) * 3.0)
    normal_dot = 0.0
    if sample_normal_length > 1e-12 and depth_normal_length > 1e-12:
        a = normalize(sample_normal)
        b = normalize(corrected_depth_normal)
        normal_dot = abs(sum(a[i] * b[i] for i in range(3)))
    locality_context = {
        "locality_distance": distance(depth_position, sample_position) > locality_limit,
        "sample_normal_invalid": sample_normal_length <= 0.25,
        "depth_normal_invalid": depth_normal_length <= 0.25,
        "normal_disagree": (
            sample_normal_length > 0.25
            and depth_normal_length > 0.25
            and normal_dot < 0.5
        ),
    }
    if any(locality_context.values()):
        return 7, sample_index, locality_context
    return 0, sample_index, {}


def neighbouring_authority_rescue(
    pixel_index: int,
    current_index: int,
    depth_position: Sequence[float],
    depth_normal: Sequence[float],
    depth_range: float,
    grid_x: int,
    grid_y: int,
    pixel_stride: int,
    width: int,
    corr_payload: bytes,
    corr_stride: int,
    obs_payload: bytes,
    obs_stride: int,
    correction,
) -> bool:
    """Test a 3x3 low-resolution neighbourhood without relaxing authority.

    This is a zero-authority counterfactual.  A neighbour may rescue the
    receiver only when it already owns the complete production authority bit
    set and its recorded source point independently passes the same locality
    and normal checks used by the current receiver contract.
    """
    px = pixel_index % width
    py = pixel_index // width
    gx = min(max(px // max(pixel_stride, 1), 0), max(grid_x - 1, 0))
    gy = min(max(py // max(pixel_stride, 1), 0), max(grid_y - 1, 0))
    corrected_depth_normal = mat3_vec(correction, depth_normal)
    depth_normal_length = norm(corrected_depth_normal)
    if depth_normal_length <= 0.25:
        return False
    depth_normal_unit = normalize(corrected_depth_normal)
    for ny in range(max(0, gy - 1), min(grid_y, gy + 2)):
        for nx in range(max(0, gx - 1), min(grid_x, gx + 2)):
            index = ny * grid_x + nx
            if index == current_index:
                continue
            flags, _target = correspondence(corr_payload, index, corr_stride)
            if (flags & FUSION_AUTHORITY) != FUSION_AUTHORITY:
                continue
            obs = observation(obs_payload, index, obs_stride)
            if obs is None:
                continue
            sample_position = transform_point(correction, obs["position"])
            sample_normal = mat3_vec(correction, obs["normal"])
            sample_normal_length = norm(sample_normal)
            if sample_normal_length <= 0.25:
                continue
            locality_limit = max(
                0.06,
                depth_range * 0.025 + max(obs["sigma"], 0.004) * 3.0,
            )
            if distance(depth_position, sample_position) > locality_limit:
                continue
            sample_normal_unit = normalize(sample_normal)
            if abs(sum(depth_normal_unit[i] * sample_normal_unit[i] for i in range(3))) < 0.5:
                continue
            return True
    return False


def classify(cell: Cell) -> tuple[str, str, int]:
    cause = cell.row.get("cause", "")
    if cause == "face_parity_open":
        return "paper_topology", "face_parity_open", cell.counts.get("accepted", 0)
    if cause == "raw_not_mature":
        return "tsdf_maturity", "raw_crossing_not_mature", cell.counts.get("accepted", 0)
    if cell.local_surface_frames == 0:
        return "depth_or_preprocess", "no_local_surface_depth", 0
    accepted = cell.counts.get("accepted", 0)
    status_counts = {
        status: cell.counts.get(STATUS_LABELS[status], 0) for status in STATUS_CODES if status != 0
    }
    dominant_status = max(status_counts, key=status_counts.get)
    rejected = sum(status_counts.values())
    # One successful observation does not acquit a gate that rejects most
    # spatial/temporal opportunities in the same paper cell.  A crossing needs
    # several neighbouring TSDF endpoints, not one isolated authorised ray.
    if accepted > 0 and accepted >= rejected:
        return "tsdf_after_gungel", "accepted_majority_but_no_raw_crossing", accepted
    if accepted > 0 and rejected > accepted:
        if dominant_status in (1, 2, 3):
            stage = "depth_preprocess_or_dual_testimony_mixed_starvation"
        else:
            stage = "gungel_candidate_contract_mixed_starvation"
        return stage, STATUS_LABELS[dominant_status], status_counts[dominant_status]
    if status_counts[dominant_status] <= 0:
        return "unresolved", "surface_seen_without_attributable_vote", 0
    if dominant_status in (1, 2, 3):
        stage = "depth_preprocess_or_dual_testimony"
    else:
        stage = "gungel_candidate_contract"
    return stage, STATUS_LABELS[dominant_status], status_counts[dominant_status]


def analyse(
    session: Path,
    paper_csv: Path,
    output_dir: Path,
    cell_side: float,
    surface_margin: float,
    max_frames: Optional[int],
) -> dict:
    cells = load_cells(paper_csv)
    manifest_path = session / "fusion_inputs" / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} is missing; the compact artifact-only copy cannot establish "
            "upstream responsibility. Copy the matching fusion_inputs directory."
        )
    rows = load_manifest(manifest_path)
    accepted_rows = [
        row for row in rows
        if row.get("accepted") == "1" and row.get("status") == "ok" and row.get("metaFile")
    ]
    if max_frames is not None:
        accepted_rows = accepted_rows[: max(0, max_frames)]
    frame_dir = session / "fusion_inputs" / "frames"
    half = cell_side * 0.5

    processed_frames = 0
    for manifest_row in accepted_rows:
        meta_path = frame_dir / manifest_row["metaFile"]
        meta = read_json(meta_path)
        texture = meta["textures"]
        depth_meta = texture["depth"]
        normal_meta = texture["normal"]
        width = int(depth_meta["width"])
        height = int(depth_meta["height"])
        depth_payload = (frame_dir / depth_meta["file"]).read_bytes()
        normal_payload = (frame_dir / normal_meta["file"]).read_bytes()
        admission = meta["gunGelAdmission"]
        if not admission.get("active"):
            continue
        obs_payload = (frame_dir / admission["observationFile"]).read_bytes()
        corr_payload = (frame_dir / admission["correspondenceFile"]).read_bytes()
        obs_stride = int(admission["observationStrideBytes"])
        corr_stride = int(admission["correspondenceStrideBytes"])
        grid_x = int(admission["gridX"])
        grid_y = int(admission["gridY"])
        pixel_stride = int(admission["pixelStride"])
        evidence_schema = str(admission.get("correspondenceEvidenceSchema", "legacy_low6"))
        motion_quality = float(meta.get("motionQuality", 1.0))
        eye = int(meta.get("recordedEyeIndex", 1))
        view = matrix(meta["view"][eye])
        projection = matrix(meta["projection"][eye])
        view_inv = matrix(meta["viewInverse"][eye])
        projection_inv = matrix(meta["projectionInverse"][eye])
        correction = matrix(meta["fusionCorrection"])
        eye_position = (view_inv[0][3], view_inv[1][3], view_inv[2][3])
        sequence = int(meta.get("sequence", manifest_row.get("sequence", 0)))

        for cell in cells:
            visible = False
            depth_seen = False
            local_surface = False
            accepted_this_frame = False
            sampled_groups: set[int] = set()
            for probe_index, probe in enumerate(probes(cell.center, half)):
                ndc = project_world(probe, view, projection)
                if ndc is None or ndc[0] < 0.01 or ndc[0] > 0.99 or ndc[1] < 0.01 or ndc[1] > 0.99:
                    continue
                visible = True
                px = min(max(int(ndc[0] * width), 0), width - 1)
                py = min(max(int(ndc[1] * height), 0), height - 1)
                pixel_index = py * width + px
                depth_ndc = depth_unorm16(depth_payload, pixel_index)
                if depth_ndc <= 0.0:
                    continue
                depth_seen = True
                uvx = (px + 0.5) / width
                uvy = (py + 0.5) / height
                depth_world = reconstruct_world(uvx, uvy, depth_ndc, projection_inv, view_inv)
                if point_aabb_distance(depth_world, cell.center, half) > surface_margin:
                    continue
                local_surface = True
                cell.surface_probe_mask |= 1 << probe_index
                cell.surface_probe_votes[probe_index] += 1
                depth_normal = normal_snorm8(normal_payload, pixel_index)
                status, group_index, absence_context = gun_gel_status(
                    pixel_index,
                    depth_world,
                    depth_normal,
                    distance(depth_world, eye_position),
                    grid_x,
                    grid_y,
                    pixel_stride,
                    width,
                    corr_payload,
                    corr_stride,
                    obs_payload,
                    obs_stride,
                    correction,
                    evidence_schema,
                    height,
                    motion_quality,
                )
                if status == 0:
                    cell.accepted_probe_mask |= 1 << probe_index
                    cell.accepted_probe_votes[probe_index] += 1
                if group_index in sampled_groups:
                    continue
                sampled_groups.add(group_index)
                label = STATUS_LABELS[status]
                cell.counts[label] += 1
                if status != 0 and neighbouring_authority_rescue(
                    pixel_index,
                    group_index,
                    depth_world,
                    depth_normal,
                    distance(depth_world, eye_position),
                    grid_x,
                    grid_y,
                    pixel_stride,
                    width,
                    corr_payload,
                    corr_stride,
                    obs_payload,
                    obs_stride,
                    correction,
                ):
                    cell.neighbour_rescue_counts[label] += 1
                if status == 8:
                    for context_label, present in absence_context.items():
                        if present:
                            cell.absent_context_counts[context_label] += 1
                    active_risks = sorted(
                        label for label, present in absence_context.items()
                        if present and label in ABSENCE_CONTEXT_LABELS
                        and label != "promotion_authority_safe"
                    )
                    combination = (
                        "promotion_authority_safe" if not active_risks
                        else "+".join(active_risks)
                    )
                    cell.absent_context_counts[f"combination:{combination}"] += 1
                if status == 7:
                    for context_label, present in absence_context.items():
                        if present:
                            cell.locality_context_counts[context_label] += 1
                    active_reasons = sorted(
                        label for label, present in absence_context.items()
                        if present and label in LOCALITY_CONTEXT_LABELS
                    )
                    combination = "+".join(active_reasons) if active_reasons else "unresolved"
                    cell.locality_context_counts[f"combination:{combination}"] += 1
                if status == 0:
                    accepted_this_frame = True

            if visible:
                cell.visible_frames += 1
            if depth_seen:
                cell.depth_frames += 1
            if local_surface:
                cell.local_surface_frames += 1
                if cell.first_surface_sequence is None:
                    cell.first_surface_sequence = sequence
                cell.last_surface_sequence = sequence
            if accepted_this_frame:
                cell.accepted_frames += 1
                if cell.first_accepted_sequence is None:
                    cell.first_accepted_sequence = sequence
                cell.last_accepted_sequence = sequence
        processed_frames += 1
        if processed_frames % 100 == 0:
            print(f"processed accepted frames: {processed_frames}/{len(accepted_rows)}", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "paper_admission_attribution.csv"
    output_summary = output_dir / "paper_admission_attribution_summary.json"
    stage_counts: Counter = Counter()
    gate_counts: Counter = Counter()
    cause_stage: dict[str, Counter] = {}
    crossing_vote_totals: Counter = Counter()
    crossing_absence_context_totals: Counter = Counter()
    crossing_locality_context_totals: Counter = Counter()
    crossing_neighbour_rescue_totals: Counter = Counter()
    crossing_neighbour_rescue_cells: Counter = Counter()
    crossing_acceptance_buckets: Counter = Counter()
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "cell_x", "cell_y", "cell_z", "world_x_m", "world_y_m", "world_z_m",
            "paper_cause", "root_stage", "dominant_gate", "dominant_gate_votes",
            "visible_frames", "depth_frames", "local_surface_frames", "accepted_frames",
            "surface_vote_count", "accepted_vote_count", "rejected_vote_count", "acceptance_ratio",
            "surface_probe_mask_hex", "accepted_probe_mask_hex",
            "surface_probe_count", "accepted_probe_count",
            "surface_probe_votes", "accepted_probe_votes",
            "first_surface_sequence", "last_surface_sequence",
            "first_accepted_sequence", "last_accepted_sequence",
        ] + [f"votes_{STATUS_LABELS[s]}" for s in STATUS_CODES]
        fields += [f"absent_votes_{label}" for label in ABSENCE_CONTEXT_LABELS]
        fields += [f"locality_votes_{label}" for label in LOCALITY_CONTEXT_LABELS]
        fields += [f"neighbour_rescue_from_{STATUS_LABELS[s]}" for s in STATUS_CODES if s != 0]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            stage, gate, votes = classify(cell)
            stage_counts[stage] += 1
            gate_counts[gate] += 1
            cause_stage.setdefault(cell.row.get("cause", ""), Counter())[stage] += 1
            total_votes = sum(cell.counts.get(STATUS_LABELS[s], 0) for s in STATUS_CODES)
            accepted_votes = cell.counts.get("accepted", 0)
            rejected_votes = total_votes - accepted_votes
            acceptance_ratio = accepted_votes / total_votes if total_votes else 0.0
            if cell.row.get("cause", "") == "raw_crossing_insufficient":
                for status in STATUS_CODES:
                    crossing_vote_totals[STATUS_LABELS[status]] += cell.counts.get(
                        STATUS_LABELS[status], 0
                    )
                crossing_absence_context_totals.update(cell.absent_context_counts)
                crossing_locality_context_totals.update(cell.locality_context_counts)
                crossing_neighbour_rescue_totals.update(cell.neighbour_rescue_counts)
                if sum(cell.neighbour_rescue_counts.values()) > 0:
                    crossing_neighbour_rescue_cells[gate] += 1
                bucket = (
                    "zero" if accepted_votes == 0 else
                    "gt0_to_10pct" if acceptance_ratio < 0.10 else
                    "10_to_25pct" if acceptance_ratio < 0.25 else
                    "25_to_50pct" if acceptance_ratio < 0.50 else
                    "50_to_75pct" if acceptance_ratio < 0.75 else
                    "75_to_100pct"
                )
                crossing_acceptance_buckets[bucket] += 1
            out = {
                "cell_x": cell.row.get("cell_x", ""),
                "cell_y": cell.row.get("cell_y", ""),
                "cell_z": cell.row.get("cell_z", ""),
                "world_x_m": cell.center[0],
                "world_y_m": cell.center[1],
                "world_z_m": cell.center[2],
                "paper_cause": cell.row.get("cause", ""),
                "root_stage": stage,
                "dominant_gate": gate,
                "dominant_gate_votes": votes,
                "visible_frames": cell.visible_frames,
                "depth_frames": cell.depth_frames,
                "local_surface_frames": cell.local_surface_frames,
                "accepted_frames": cell.accepted_frames,
                "surface_vote_count": total_votes,
                "accepted_vote_count": accepted_votes,
                "rejected_vote_count": rejected_votes,
                "acceptance_ratio": f"{acceptance_ratio:.9g}",
                "surface_probe_mask_hex": f"0x{cell.surface_probe_mask:03X}",
                "accepted_probe_mask_hex": f"0x{cell.accepted_probe_mask:03X}",
                "surface_probe_count": cell.surface_probe_mask.bit_count(),
                "accepted_probe_count": cell.accepted_probe_mask.bit_count(),
                "surface_probe_votes": "|".join(str(value) for value in cell.surface_probe_votes),
                "accepted_probe_votes": "|".join(str(value) for value in cell.accepted_probe_votes),
                "first_surface_sequence": cell.first_surface_sequence or "",
                "last_surface_sequence": cell.last_surface_sequence or "",
                "first_accepted_sequence": cell.first_accepted_sequence or "",
                "last_accepted_sequence": cell.last_accepted_sequence or "",
            }
            for status in STATUS_CODES:
                out[f"votes_{STATUS_LABELS[status]}"] = cell.counts.get(STATUS_LABELS[status], 0)
            for label in ABSENCE_CONTEXT_LABELS:
                out[f"absent_votes_{label}"] = cell.absent_context_counts.get(label, 0)
            for label in LOCALITY_CONTEXT_LABELS:
                out[f"locality_votes_{label}"] = cell.locality_context_counts.get(label, 0)
            for status in STATUS_CODES:
                if status == 0:
                    continue
                label = STATUS_LABELS[status]
                out[f"neighbour_rescue_from_{label}"] = cell.neighbour_rescue_counts.get(label, 0)
            writer.writerow(out)

    summary = {
        "schema": "scancover.paper_admission_attribution.v1",
        "session": str(session),
        "paperAudit": str(paper_csv),
        "classificationIsInference": True,
        "association": {
            "eye": "recorded production eye from each frame (normally right/1)",
            "paperCellSideM": cell_side,
            "surfaceMarginOutsideCellM": surface_margin,
            "cellProbes": "centre plus eight corners",
            "gungelMapping": "same pixel/pixelStride correspondence index and status order as VolumeIntegration.compute",
            "boundary": "cell attribution is deterministic from recorded payload; exact final TSDF state still requires engine replay",
        },
        "input": {
            "manifestRows": len(rows),
            "acceptedRowsAvailable": len([
                row for row in rows if row.get("accepted") == "1" and row.get("status") == "ok"
            ]),
            "acceptedRowsProcessed": processed_frames,
            "partial": max_frames is not None,
            "directFailureCells": len(cells),
        },
        "rootStageCounts": dict(sorted(stage_counts.items())),
        "dominantGateCounts": dict(sorted(gate_counts.items())),
        "rawCrossingInsufficientAdmissionVoteTotals": dict(sorted(crossing_vote_totals.items())),
        "stableCandidateAbsentPromotionContextVoteTotals": dict(
            sorted(crossing_absence_context_totals.items())
        ),
        "pixelCandidateMismatchReasonVoteTotals": dict(
            sorted(crossing_locality_context_totals.items())
        ),
        "neighbourAuthorityRescueVoteTotals": dict(
            sorted(crossing_neighbour_rescue_totals.items())
        ),
        "neighbourAuthorityRescueCellCountsByCurrentDominantGate": dict(
            sorted(crossing_neighbour_rescue_cells.items())
        ),
        "rawCrossingInsufficientAcceptanceRatioBuckets": dict(
            sorted(crossing_acceptance_buckets.items())
        ),
        "paperCauseByRootStage": {
            cause: dict(sorted(counts.items())) for cause, counts in sorted(cause_stage.items())
        },
        "outputs": {"cellsCsv": output_csv.name},
        "productionEffect": "none; offline read-only replay attribution",
    }
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="matching replay_sessions/session_* directory")
    parser.add_argument("--paper-csv", type=Path, help="matching paper_hole_cells.csv")
    parser.add_argument("--output", type=Path, help="output directory (default: session/artifacts/admission_attribution)")
    parser.add_argument("--cell-side", type=float, default=0.10, help="paper cell side in metres")
    parser.add_argument("--surface-margin", type=float, default=0.05, help="allowed distance outside the paper-cell AABB")
    parser.add_argument("--max-frames", type=int, help="smoke-test only; marks output partial")
    args = parser.parse_args()

    session = args.session.resolve()
    paper_csv = discover_paper_csv(session, args.paper_csv.resolve() if args.paper_csv else None)
    output = args.output.resolve() if args.output else session / "artifacts" / "admission_attribution"
    try:
        summary = analyse(
            session=session,
            paper_csv=paper_csv,
            output_dir=output,
            cell_side=max(args.cell_side, 0.001),
            surface_margin=max(args.surface_margin, 0.0),
            max_frames=args.max_frames,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
