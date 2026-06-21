#!/usr/bin/env python3
"""Run a minimal Quest3 virtual-observer experiment on a room mesh.

This script is intentionally narrow:
- use a truth/source mesh as the virtual room;
- replay real Quest3/Unity camera poses from a ScanCover multi-frame session;
- raycast the virtual room;
- apply lightweight Quest3-like observation risk/dropout rules calibrated from
  the existing observation summary;
- compare the simulated observation distribution against real exported stats.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from ScanCoverQuest3CloneDataManifest import DEFAULT_MANIFEST, derived_output, load_manifest


DEFAULT_LEARNING_PROFILE = Path(
    r"D:\PCA\Unity-MRMotifs-ScanCover-main\ScanCoverExports"
    r"\Quest3CloneDataManifest\CombinedObservationStats_UsableCloneData\quest3_learning_profile.json"
)

DEFAULT_REAL_SUMMARY = Path(
    r"D:\PCA\Unity-MRMotifs-ScanCover-main\ScanCoverExports"
    r"\Quest3CloneDataManifest\CombinedObservationStats_UsableCloneData\combined_observation_summary.json"
)

DISTANCE_BINS = [
    ("0.0-0.5m", 0.0, 0.5),
    ("0.5-1.0m", 0.5, 1.0),
    ("1.0-1.5m", 1.0, 1.5),
    ("1.5-2.0m", 1.5, 2.0),
    ("2.0-3.0m", 2.0, 3.0),
    ("3.0-5.0m", 3.0, 5.0),
    ("5.0-8.0m", 5.0, 8.0),
    ("8.0m+", 8.0, float("inf")),
]

ANGLE_BINS = [
    ("0-20deg", 0.0, 20.0),
    ("20-40deg", 20.0, 40.0),
    ("40-60deg", 40.0, 60.0),
    ("60-75deg", 60.0, 75.0),
    ("75deg+", 75.0, 180.0),
]

RISK_COLOR = np.asarray((1.0, 0.12, 0.08), dtype=np.float64)
STABLE_COLOR = np.asarray((0.0, 0.85, 1.0), dtype=np.float64)
FAR_COLOR = np.asarray((1.0, 0.8, 0.05), dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-mesh", type=Path, required=True, help="Replica or proxy room mesh.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Quest3 clone data manifest. When set, learning profile and real summary defaults are read from it.",
    )
    parser.add_argument(
        "--pose-session",
        type=Path,
        default=Path(
            r"D:\PCA\Unity-MRMotifs-ScanCover-main\ScanCoverExports\ScanSessions"
            r"\ScanCover_MultiFrame_20260603_115601_360"
        ),
        help="ScanCover_MultiFrame_... session folder used as real Quest3 pose path.",
    )
    parser.add_argument(
        "--real-summary",
        type=Path,
        default=DEFAULT_REAL_SUMMARY,
        help="Combined real observation summary for similarity comparison.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            r"D:\PCA\Unity-MRMotifs-ScanCover-main\ScanCoverExports"
            r"\Quest3VirtualCloneExperiments"
        ),
    )
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--max-distance", type=float, default=5.0)
    parser.add_argument("--min-distance", type=float, default=0.3)
    parser.add_argument("--noise-std", type=float, default=0.008)
    parser.add_argument("--seed", type=int, default=15319)
    parser.add_argument(
        "--learning-profile",
        type=Path,
        default=None,
        help="Quest3 observation learning profile. If present, it replaces hard-coded risk/dropout priors.",
    )
    parser.add_argument(
        "--pose-source",
        default="real",
        choices=["real", "auto-scan"],
        help="Use real exported Quest3 poses or generate virtual room-scan poses from the mesh bounds.",
    )
    parser.add_argument(
        "--center-poses-in-mesh",
        action="store_true",
        help="Translate replayed Quest3 poses so their center lands at the mesh bounding-box center.",
    )
    parser.add_argument(
        "--vertical-axis",
        default="auto",
        choices=["auto", "x", "y", "z"],
        help="Vertical axis for auto-scan. Auto picks the smallest room-mesh extent.",
    )
    parser.add_argument("--head-height", type=float, default=1.45, help="Auto-scan head height above room minimum.")
    parser.add_argument("--scan-radius-scale", type=float, default=0.42, help="Auto-scan path radius relative to room size.")
    parser.add_argument("--auto-pitch-degrees", type=float, nargs="*", default=[-35.0, -18.0, 0.0, 18.0, 35.0])
    parser.add_argument(
        "--scan-pattern",
        default="orbit",
        choices=["orbit", "stratified-slices"],
        help="Auto-scan pattern. stratified-slices samples room positions, heights, yaw sectors, and pitch sectors.",
    )
    parser.add_argument("--slice-grid", type=int, default=3, help="Grid count per horizontal axis for stratified-slices.")
    parser.add_argument(
        "--slice-heights",
        type=float,
        nargs="*",
        default=[0.35, 0.75, 1.25, 1.65, 2.10],
        help="Stratified camera heights above room minimum in meters.",
    )
    parser.add_argument(
        "--slice-yaw-degrees",
        type=float,
        nargs="*",
        default=[0.0, 60.0, 120.0, 180.0, 240.0, 300.0],
        help="Horizontal yaw sectors for stratified-slices.",
    )
    parser.add_argument(
        "--sim-top-coverage-pass",
        action="store_true",
        help=(
            "Simulation-only: reserve extra stratified-slices views near the upper room bound. "
            "This uses truth-mesh bounds and must not be treated as a real Quest3 scan rule."
        ),
    )
    parser.add_argument(
        "--sim-top-frame-share",
        type=float,
        default=0.30,
        help="Share of stratified-slices frames reserved for the simulation-only upper-room coverage pass.",
    )
    parser.add_argument(
        "--coverage-pass",
        default="full",
        choices=["none", "full"],
        help="Add dedicated upper-surface and upper-corner virtual scan views for room coverage.",
    )
    parser.add_argument(
        "--coverage-frame-share",
        type=float,
        default=0.50,
        help="Share of auto-scan frames reserved for upper-surface coverage views.",
    )
    parser.add_argument(
        "--scan-feedback",
        type=Path,
        default=None,
        help="Optional ScanCoverVirtualScanFeedback/v1 JSON. Missing-coverage targets become extra virtual look-at views.",
    )
    parser.add_argument(
        "--bottom-coverage-priority",
        action="store_true",
        help="Reserve coverage views for lower surfaces even when no scan-feedback room priority is present.",
    )
    parser.add_argument(
        "--mode",
        default="proxy",
        choices=["proxy", "replica"],
        help="Labels the run. Use replica only when the input is an actual Replica room mesh.",
    )
    parser.add_argument(
        "--feature-voxel",
        type=float,
        default=0.02,
        help="World-space cell size for exported virtual Quest3 observation features.",
    )
    return parser.parse_args()


def filter_scan_feedback_for_mesh(scan_feedback: dict[str, Any] | None, truth_mesh: Path) -> dict[str, Any] | None:
    if not scan_feedback:
        return None
    mesh_room = truth_mesh.stem.lower()
    rooms = scan_feedback.get("rooms", [])
    matched_rooms = [
        item for item in rooms
        if str(item.get("room", "")).lower() == mesh_room
        or Path(str(item.get("truthMesh", ""))).stem.lower() == mesh_room
    ]
    if not matched_rooms:
        return scan_feedback

    filtered = dict(scan_feedback)
    filtered["rooms"] = matched_rooms
    global_targets = []
    for item in scan_feedback.get("globalScanTargets", []):
        if str(item.get("room", "")).lower() == mesh_room:
            global_targets.append(item)
    if not global_targets:
        for room_item in matched_rooms:
            for target in room_item.get("scanTargets", []):
                target_copy = dict(target)
                target_copy.setdefault("room", room_item.get("room", mesh_room))
                global_targets.append(target_copy)
    filtered["globalScanTargets"] = global_targets
    return filtered


def read_manifest(session: Path) -> list[dict[str, str]]:
    manifest = session / "session_manifest.csv"
    lines = manifest.read_text(encoding="utf-8-sig").splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.startswith("frame,")), None)
    if header_index is None:
        raise RuntimeError(f"Could not find manifest header in {manifest}")
    return list(csv.DictReader(lines[header_index:]))


def pick_evenly(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0 or count >= len(rows):
        return rows
    indices = np.linspace(0, len(rows) - 1, count).round().astype(np.int64)
    return [rows[int(i)] for i in indices]


def load_camera(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    return v / length if length > 1e-8 else v


def bin_label(value: float, bins: list[tuple[str, float, float]]) -> str:
    for label, lo, hi in bins:
        if lo <= value < hi:
            return label
    return bins[-1][0]


def profile_bin(value: float, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        label = str(row.get("bin", ""))
        if label.endswith("m+"):
            lo = float(label[:-2])
            if value >= lo:
                return row
            continue
        if label.endswith("m") and "-" in label:
            lo_s, hi_s = label[:-1].split("-", 1)
            if float(lo_s) <= value < float(hi_s):
                return row
            continue
        if label.endswith("deg+"):
            lo = float(label[:-4])
            if value >= lo:
                return row
            continue
        if label.endswith("deg") and "-" in label:
            lo_s, hi_s = label[:-3].split("-", 1)
            if float(lo_s) <= value < float(hi_s):
                return row
    return None


def risk_probability(
    distance: float,
    angle_deg: float,
    near_edge_like: bool,
    learning_profile: dict[str, Any] | None = None,
) -> float:
    if learning_profile is not None:
        distance_row = profile_bin(distance, learning_profile.get("distanceBins", []))
        angle_row = profile_bin(angle_deg, learning_profile.get("angleBins", []))
        global_risk = float(learning_profile.get("globalRisk", {}).get("anyRiskRatio", 0.12))
        distance_risk = float(distance_row.get("riskRatio", global_risk)) if distance_row else global_risk
        angle_risk = float(angle_row.get("riskRatio", global_risk)) if angle_row else global_risk
        weights = learning_profile.get("trainingWeights", {})
        distance_weight = float(weights.get("distanceWeight", 0.45))
        angle_weight = float(weights.get("angleWeight", 0.35))
        edge_weight = float(weights.get("edgeRiskWeight", 0.20))
        base_weight = max(0.0, 1.0 - distance_weight - angle_weight)
        p = global_risk * base_weight + distance_risk * distance_weight + angle_risk * angle_weight
        if near_edge_like:
            p += edge_weight
        return min(0.85, max(0.0, p))

    # Calibrated from the current combined real-data shape, not trained.
    p = 0.06
    if distance < 0.5:
        p += 0.12
    elif distance < 1.0:
        p += 0.06
    elif distance >= 3.0:
        p += 0.08

    if angle_deg >= 75.0:
        p += 0.24
    elif angle_deg >= 60.0:
        p += 0.08
    elif angle_deg >= 40.0:
        p += 0.03

    if near_edge_like:
        p += 0.20
    return min(0.85, max(0.0, p))


def dropout_probability(
    distance: float,
    angle_deg: float,
    risk: bool,
    learning_profile: dict[str, Any] | None = None,
) -> float:
    if learning_profile is not None:
        distance_row = profile_bin(distance, learning_profile.get("distanceBins", []))
        angle_row = profile_bin(angle_deg, learning_profile.get("angleBins", []))
        global_risk = float(learning_profile.get("globalRisk", {}).get("anyRiskRatio", 0.12))
        distance_risk = float(distance_row.get("riskRatio", global_risk)) if distance_row else global_risk
        angle_risk = float(angle_row.get("riskRatio", global_risk)) if angle_row else global_risk
        p = 0.015 + 0.12 * distance_risk + 0.10 * angle_risk
        if risk:
            p += 0.08
        if angle_deg >= 75.0:
            p += 0.06
        if distance >= 3.0:
            p += 0.035
        return min(0.7, max(0.0, p))

    p = 0.02
    if distance < 0.35:
        p += 0.08
    if distance >= 3.0:
        p += 0.05
    if angle_deg >= 75.0:
        p += 0.16
    elif angle_deg >= 60.0:
        p += 0.05
    if risk:
        p += 0.08
    return min(0.7, max(0.0, p))


def load_legacy_mesh(mesh_path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Could not load mesh: {mesh_path}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh


def build_scene(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    return scene


def axis_index(axis: str, bounds: o3d.geometry.AxisAlignedBoundingBox) -> int:
    if axis != "auto":
        return {"x": 0, "y": 1, "z": 2}[axis]
    extent = np.asarray(bounds.get_extent(), dtype=np.float64)
    return int(np.argmin(extent))


def make_camera_dict(
    origin: np.ndarray,
    forward: np.ndarray,
    up_hint: np.ndarray,
    fov_y: float,
    aspect: float,
    name: str,
) -> dict[str, Any]:
    forward = normalize(forward.astype(np.float64))
    right = normalize(np.cross(forward, up_hint))
    if np.linalg.norm(right) < 1e-8:
        right = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    up = normalize(np.cross(right, forward))
    return {
        "pose": {
            "position": origin.astype(float).tolist(),
            "forward": forward.tolist(),
            "right": right.tolist(),
            "up": up.tolist(),
        },
        "camera": {
            "fieldOfView": fov_y,
            "aspect": aspect,
        },
        "syntheticName": name,
    }


def rotate_around_axis(vector: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    axis = normalize(axis.astype(np.float64))
    vector = vector.astype(np.float64)
    return (
        vector * math.cos(radians)
        + np.cross(axis, vector) * math.sin(radians)
        + axis * np.dot(axis, vector) * (1.0 - math.cos(radians))
    )


def build_stratified_slice_cameras(
    mesh: o3d.geometry.TriangleMesh,
    frame_count: int,
    vertical_axis_name: str,
    slice_grid: int,
    slice_heights: list[float],
    yaw_degrees: list[float],
    fov_y: float,
    aspect: float,
    sim_top_coverage_pass: bool = False,
    sim_top_frame_share: float = 0.30,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bounds = mesh.get_axis_aligned_bounding_box()
    min_bound = np.asarray(bounds.get_min_bound(), dtype=np.float64)
    max_bound = np.asarray(bounds.get_max_bound(), dtype=np.float64)
    center = np.asarray(bounds.get_center(), dtype=np.float64)
    extent = np.asarray(bounds.get_extent(), dtype=np.float64)
    vertical = axis_index(vertical_axis_name, bounds)
    horizontal = [i for i in range(3) if i != vertical]

    up = np.zeros(3, dtype=np.float64)
    up[vertical] = 1.0
    h0 = np.zeros(3, dtype=np.float64)
    h1 = np.zeros(3, dtype=np.float64)
    h0[horizontal[0]] = 1.0
    h1[horizontal[1]] = 1.0

    grid = max(1, slice_grid)
    fractions = [0.5] if grid == 1 else np.linspace(0.18, 0.82, grid).tolist()
    room_height = max(0.01, extent[vertical])
    heights = []
    for height in (slice_heights if slice_heights else [0.75, 1.45]):
        world_height = min_bound[vertical] + max(0.08, min(float(height), room_height - 0.08))
        if min_bound[vertical] + 0.05 <= world_height <= max_bound[vertical] - 0.02:
            heights.append(world_height)
    if not heights:
        heights = [center[vertical]]

    yaws = yaw_degrees if yaw_degrees else [0.0, 90.0, 180.0, 270.0]
    candidates: list[dict[str, Any]] = []
    for world_height in heights:
        relative_height = (world_height - min_bound[vertical]) / room_height
        if relative_height < 0.32:
            pitches = [-55.0, -30.0, -10.0, 15.0]
        elif relative_height > 0.72:
            pitches = [-10.0, 15.0, 40.0, 58.0]
        else:
            pitches = [-35.0, -12.0, 12.0, 35.0]

        for u in fractions:
            for v in fractions:
                origin = center.copy()
                origin[horizontal[0]] = min_bound[horizontal[0]] + extent[horizontal[0]] * float(u)
                origin[horizontal[1]] = min_bound[horizontal[1]] + extent[horizontal[1]] * float(v)
                origin[vertical] = world_height
                to_center = center - origin
                to_center[vertical] = 0.0
                if np.linalg.norm(to_center) <= 1e-6:
                    to_center = h0.copy()
                base_forward = normalize(to_center)
                for yaw in yaws:
                    yaw_forward = rotate_around_axis(base_forward, up, yaw)
                    right = normalize(np.cross(yaw_forward, up))
                    if np.linalg.norm(right) <= 1e-6:
                        right = h0.copy()
                    for pitch in pitches:
                        forward = rotate_around_axis(yaw_forward, right, pitch)
                        candidates.append(
                            make_camera_dict(
                                origin=origin,
                                forward=forward,
                                up_hint=up,
                                fov_y=fov_y,
                                aspect=aspect,
                                name=(
                                    f"slice_{len(candidates):04d}_h{relative_height:.2f}"
                                    f"_u{u:.2f}_v{v:.2f}_yaw{yaw:.0f}_pitch{pitch:.0f}"
                                ),
                            )
                        )

    top_candidates: list[dict[str, Any]] = []
    if sim_top_coverage_pass:
        # Simulation-only pass: the virtual scanner is allowed to use truth-mesh
        # bounds to stress-test whether upper-room samples can be covered. Do not
        # copy this as a real Quest3 online rule.
        top_grid = max(grid + 1, 4)
        top_fractions = np.linspace(0.12, 0.88, top_grid).tolist()
        top_heights = []
        for relative_height in (0.72, 0.84, 0.92):
            world_height = min_bound[vertical] + room_height * relative_height
            world_height = max(min_bound[vertical] + 0.12, min(world_height, max_bound[vertical] - 0.12))
            top_heights.append(world_height)
        top_heights = sorted(set(round(float(h), 4) for h in top_heights))
        top_pitches = [28.0, 45.0, 62.0, 75.0]
        for world_height in top_heights:
            relative_height = (world_height - min_bound[vertical]) / room_height
            for u in top_fractions:
                for v in top_fractions:
                    origin = center.copy()
                    origin[horizontal[0]] = min_bound[horizontal[0]] + extent[horizontal[0]] * float(u)
                    origin[horizontal[1]] = min_bound[horizontal[1]] + extent[horizontal[1]] * float(v)
                    origin[vertical] = world_height
                    to_center = center - origin
                    to_center[vertical] = 0.0
                    if np.linalg.norm(to_center) <= 1e-6:
                        to_center = h0.copy()
                    base_forward = normalize(to_center)
                    for yaw in yaws:
                        yaw_forward = rotate_around_axis(base_forward, up, yaw)
                        right = normalize(np.cross(yaw_forward, up))
                        if np.linalg.norm(right) <= 1e-6:
                            right = h0.copy()
                        for pitch in top_pitches:
                            forward = rotate_around_axis(yaw_forward, right, pitch)
                            top_candidates.append(
                                make_camera_dict(
                                    origin=origin,
                                    forward=forward,
                                    up_hint=up,
                                    fov_y=fov_y,
                                    aspect=aspect,
                                    name=(
                                        f"sim_top_{len(top_candidates):04d}_h{relative_height:.2f}"
                                        f"_u{u:.2f}_v{v:.2f}_yaw{yaw:.0f}_pitch{pitch:.0f}"
                                    ),
                                )
                            )

    if sim_top_coverage_pass and top_candidates and frame_count > 1:
        top_share = max(0.05, min(0.75, float(sim_top_frame_share)))
        top_count = min(frame_count - 1, max(1, int(round(frame_count * top_share))))
        base_count = max(1, frame_count - top_count)
        cameras = pick_evenly(candidates, base_count) + pick_evenly(top_candidates, top_count)
    else:
        cameras = pick_evenly(candidates, frame_count)

    metadata = {
        "verticalAxis": "xyz"[vertical],
        "scanPattern": "stratified-slices",
        "sliceGrid": grid,
        "sliceHeights": [float(h - min_bound[vertical]) for h in heights],
        "sliceYawDegrees": [float(y) for y in yaws],
        "candidateFrames": len(candidates),
        "simTopCoveragePass": bool(sim_top_coverage_pass),
        "simTopCandidateFrames": len(top_candidates),
        "simTopFrameShare": float(sim_top_frame_share),
        "generatedFrames": len(cameras),
        "roomCenter": center.tolist(),
        "roomExtent": extent.tolist(),
    }
    return cameras, metadata


def build_auto_scan_cameras(
    mesh: o3d.geometry.TriangleMesh,
    frame_count: int,
    vertical_axis_name: str,
    head_height: float,
    radius_scale: float,
    pitch_degrees: list[float],
    fov_y: float,
    aspect: float,
    coverage_pass: str = "full",
    coverage_frame_share: float = 0.30,
    scan_feedback: dict[str, Any] | None = None,
    force_bottom_coverage_priority: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bounds = mesh.get_axis_aligned_bounding_box()
    min_bound = np.asarray(bounds.get_min_bound(), dtype=np.float64)
    max_bound = np.asarray(bounds.get_max_bound(), dtype=np.float64)
    center = np.asarray(bounds.get_center(), dtype=np.float64)
    extent = np.asarray(bounds.get_extent(), dtype=np.float64)
    vertical = axis_index(vertical_axis_name, bounds)
    horizontal = [i for i in range(3) if i != vertical]

    up = np.zeros(3, dtype=np.float64)
    up[vertical] = 1.0
    h0 = np.zeros(3, dtype=np.float64)
    h1 = np.zeros(3, dtype=np.float64)
    h0[horizontal[0]] = 1.0
    h1[horizontal[1]] = 1.0

    room_center = center.copy()
    room_center[vertical] = min_bound[vertical] + head_height
    radius0 = max(0.25, extent[horizontal[0]] * radius_scale)
    radius1 = max(0.25, extent[horizontal[1]] * radius_scale)

    pitches = pitch_degrees if pitch_degrees else [0.0]
    cameras: list[dict[str, Any]] = []
    feedback_targets: list[dict[str, Any]] = []
    if scan_feedback:
        feedback_targets = list(scan_feedback.get("globalScanTargets", []))
        if not feedback_targets:
            for room_item in scan_feedback.get("rooms", []):
                feedback_targets.extend(room_item.get("scanTargets", []))
        feedback_targets = [item for item in feedback_targets if "center" in item]
        feedback_targets.sort(key=lambda item: float(item.get("priority", item.get("pointCount", 0))), reverse=True)
    bottom_coverage_priority = force_bottom_coverage_priority
    if scan_feedback:
        for room_item in scan_feedback.get("rooms", []):
            for band_item in room_item.get("coverageBandPriorities", []):
                band_name = str(band_item.get("band", "")).lower()
                if "bottom" in band_name or "lower" in band_name:
                    bottom_coverage_priority = True

    feedback_frame_count = 0
    if feedback_targets and frame_count >= 24:
        feedback_frame_count = min(len(feedback_targets), max(8, int(round(frame_count * 0.20))))
        feedback_frame_count = min(feedback_frame_count, max(0, frame_count - 8))
    planned_frame_count = max(1, frame_count - feedback_frame_count)
    coverage_frame_count = 0
    if coverage_pass != "none":
        coverage_frame_count = int(round(planned_frame_count * max(0.0, min(0.75, coverage_frame_share))))
        coverage_frame_count = max(4, coverage_frame_count) if planned_frame_count >= 12 else coverage_frame_count
        coverage_frame_count = min(planned_frame_count - 1, coverage_frame_count) if planned_frame_count > 1 else 0
    base_frame_count = max(1, planned_frame_count - coverage_frame_count)
    yaw_variant_count = 3
    base_count = max(1, math.ceil(base_frame_count / (len(pitches) * yaw_variant_count)))
    for i in range(base_count):
        theta = (2.0 * math.pi * i) / base_count
        orbit_offset = h0 * (math.cos(theta) * radius0) + h1 * (math.sin(theta) * radius1)
        origin = room_center + orbit_offset

        # Scan outward more often than inward so wall distances spread across 1m-5m.
        outward = normalize(orbit_offset)
        inward = normalize(room_center - origin)
        tangent = normalize(-math.sin(theta) * h0 + math.cos(theta) * h1)
        yaw_dirs = [outward, normalize(outward * 0.7 + tangent * 0.3), normalize(inward * 0.5 + tangent * 0.5)]
        for yaw_index, base_dir in enumerate(yaw_dirs):
            for pitch in pitches:
                if len(cameras) >= base_frame_count:
                    break
                forward = rotate_around_axis(base_dir, tangent, pitch)
                cameras.append(
                    make_camera_dict(
                        origin=origin,
                        forward=forward,
                        up_hint=up,
                        fov_y=fov_y,
                        aspect=aspect,
                        name=f"auto_{len(cameras):04d}_yaw{yaw_index}_pitch{pitch:.1f}",
                    )
                )
            if len(cameras) >= base_frame_count:
                break

    if coverage_frame_count > 0 and len(cameras) < planned_frame_count:
        top_center = center.copy()
        top_center[vertical] = max_bound[vertical] - max(0.04, extent[vertical] * 0.06)
        upper_center = center.copy()
        upper_center[vertical] = max_bound[vertical] - max(0.08, extent[vertical] * 0.12)
        lower_center = center.copy()
        lower_center[vertical] = min_bound[vertical] + max(0.08, extent[vertical] * 0.10)
        target_radius0 = max(0.12, extent[horizontal[0]] * 0.36)
        target_radius1 = max(0.12, extent[horizontal[1]] * 0.36)
        low_head_height = min_bound[vertical] + max(0.55, min(head_height, 0.85))
        coverage_samples = max(4, coverage_frame_count)
        coverage_targets: list[tuple[str, np.ndarray]] = []
        for i in range(coverage_samples):
            theta = (2.0 * math.pi * i) / coverage_samples
            orbit_offset = h0 * (math.cos(theta) * radius0) + h1 * (math.sin(theta) * radius1)
            target_offset = h0 * (math.cos(theta) * target_radius0) + h1 * (math.sin(theta) * target_radius1)
            origin = room_center + orbit_offset
            targets_this_sample: list[tuple[str, np.ndarray, np.ndarray]] = [
                ("top_center", top_center, origin),
                ("upper_edge", upper_center + target_offset, origin),
                ("upper_opposite", upper_center - target_offset * 0.55, origin),
            ]
            if bottom_coverage_priority:
                low_origin = origin.copy()
                low_origin[vertical] = low_head_height
                targets_this_sample.extend(
                    [
                        ("lower_center", lower_center, low_origin),
                        ("lower_edge", lower_center + target_offset, low_origin),
                    ]
                )
            for target_name, target, target_origin in targets_this_sample:
                if len(cameras) >= planned_frame_count:
                    break
                coverage_targets.append((target_name, target))
                cameras.append(
                    make_camera_dict(
                        origin=target_origin,
                        forward=target - target_origin,
                        up_hint=up,
                        fov_y=fov_y,
                        aspect=aspect,
                        name=f"auto_{len(cameras):04d}_coverage_{target_name}",
                    )
                )
            if len(cameras) >= planned_frame_count:
                break

    feedback_targets_added = 0
    if feedback_targets and len(cameras) < frame_count:
        max_feedback_frames = max(0, frame_count - len(cameras))
        for target_index, item in enumerate(feedback_targets[:max_feedback_frames]):
            target = np.asarray(item["center"], dtype=np.float64)
            horizontal_delta = target - room_center
            horizontal_delta[vertical] = 0.0
            if np.linalg.norm(horizontal_delta) <= 1e-6:
                horizontal_delta = h0.copy()
            outward = normalize(horizontal_delta)
            origin = room_center + outward * max(radius0, radius1)
            target_is_low = target[vertical] < room_center[vertical] - extent[vertical] * 0.18
            if target_is_low:
                origin[vertical] = min_bound[vertical] + max(0.55, min(head_height, 0.85))
            else:
                origin[vertical] = min_bound[vertical] + head_height
            target_name = str(item.get("kind", "feedback_target"))
            cameras.append(
                make_camera_dict(
                    origin=origin,
                    forward=target - origin,
                    up_hint=up,
                    fov_y=fov_y,
                    aspect=aspect,
                    name=f"auto_{len(cameras):04d}_feedback_{target_name}_{target_index:02d}",
                )
            )
            feedback_targets_added += 1
            if len(cameras) >= frame_count:
                break

    metadata = {
        "verticalAxis": "xyz"[vertical],
        "headHeight": head_height,
        "radiusScale": radius_scale,
        "roomCenter": room_center.tolist(),
        "roomExtent": extent.tolist(),
        "pitchDegrees": pitches,
        "coveragePass": coverage_pass,
        "coverageFrameShare": coverage_frame_share,
        "bottomCoveragePriority": bottom_coverage_priority,
        "baseFramesRequested": base_frame_count,
        "coverageFramesRequested": coverage_frame_count,
        "feedbackFramesRequested": feedback_frame_count,
        "feedbackTargetsAvailable": len(feedback_targets),
        "feedbackTargetsAdded": feedback_targets_added,
        "generatedFrames": len(cameras),
    }
    return cameras, metadata


def make_rays(
    camera_data: dict[str, Any],
    width: int,
    height: int,
    pose_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pose = camera_data["pose"]
    camera = camera_data["camera"]
    origin = np.asarray(pose["position"], dtype=np.float32) + pose_offset.astype(np.float32)
    right = normalize(np.asarray(pose["right"], dtype=np.float64))
    up = normalize(np.asarray(pose["up"], dtype=np.float64))
    forward = normalize(np.asarray(pose["forward"], dtype=np.float64))

    fov_y = math.radians(float(camera["fieldOfView"]))
    aspect = float(camera["aspect"])
    tan_y = math.tan(fov_y * 0.5)
    tan_x = tan_y * aspect

    rays = np.zeros((width * height, 6), dtype=np.float32)
    dirs = np.zeros((width * height, 3), dtype=np.float64)
    index = 0
    for y in range(height):
        ndc_y = 1.0 - (2.0 * (y + 0.5) / height)
        for x in range(width):
            ndc_x = (2.0 * (x + 0.5) / width) - 1.0
            direction = normalize(forward + right * (ndc_x * tan_x) + up * (ndc_y * tan_y))
            rays[index, :3] = origin
            rays[index, 3:] = direction.astype(np.float32)
            dirs[index] = direction
            index += 1
    return rays, dirs


def estimate_edge_like(points: np.ndarray, width: int, height: int, valid_mask: np.ndarray) -> np.ndarray:
    depth = np.full((height, width), np.nan, dtype=np.float64)
    flat_depth = np.linalg.norm(points, axis=1)
    depth.reshape(-1)[valid_mask] = flat_depth[valid_mask]

    edge = np.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            if not np.isfinite(depth[y, x]):
                continue
            local = []
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < height and 0 <= xx < width and np.isfinite(depth[yy, xx]):
                    local.append(abs(depth[y, x] - depth[yy, xx]))
            if local and max(local) > 0.08:
                edge[y, x] = True
    return edge.reshape(-1)


def color_for_point(distance: float, risk: bool) -> np.ndarray:
    if risk:
        return RISK_COLOR
    if distance >= 3.0:
        return FAR_COLOR
    return STABLE_COLOR


def write_cloud(path: Path, points: np.ndarray, colors: np.ndarray, normals: np.ndarray | None = None) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None and len(normals) == len(points):
        cloud.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False)


def ratio_table(counts: dict[str, int], risk_counts: dict[str, int], order: list[tuple[str, float, float]]) -> list[dict[str, Any]]:
    rows = []
    total = max(1, sum(counts.values()))
    for label, _, _ in order:
        count = counts.get(label, 0)
        risk = risk_counts.get(label, 0)
        rows.append(
            {
                "bin": label,
                "count": count,
                "share": count / total,
                "riskRatio": risk / count if count else 0.0,
            }
        )
    return rows


def compare_bins(
    real_rows: list[dict[str, Any]],
    simulated_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    real_total = max(1, sum(int(row.get("count", 0)) for row in real_rows))
    sim_total = max(1, sum(int(row.get("count", 0)) for row in simulated_rows))
    sim_by_label = {row["bin"]: row for row in simulated_rows}
    result = []
    for row in real_rows:
        label = row["bin"]
        sim = sim_by_label.get(label, {"count": 0, "riskRatio": 0.0})
        real_share = int(row.get("count", 0)) / real_total
        sim_share = int(sim.get("count", 0)) / sim_total
        result.append(
            {
                "bin": label,
                "realShare": real_share,
                "simulatedShare": sim_share,
                "shareDelta": sim_share - real_share,
                "realRiskRatio": float(row.get("riskRatio", 0.0)),
                "simulatedRiskRatio": float(sim.get("riskRatio", 0.0)),
                "riskDelta": float(sim.get("riskRatio", 0.0)) - float(row.get("riskRatio", 0.0)),
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def feature_key(point: np.ndarray, voxel_size: float) -> tuple[int, int, int]:
    return tuple(np.floor(point / voxel_size).astype(np.int64).tolist())


def make_feature_cell() -> dict[str, Any]:
    return {
        "positions": [],
        "normals": [],
        "distances": [],
        "view_depths": [],
        "view_angles": [],
        "risk_count": 0,
        "boundary_risk_count": 0,
        "crease_risk_count": 0,
        "frames": set(),
        "hits": 0,
    }


def add_feature_observation(
    cells: dict[tuple[int, int, int], dict[str, Any]],
    point: np.ndarray,
    normal: np.ndarray,
    distance: float,
    view_depth: float,
    view_angle: float,
    risk: bool,
    edge_like: bool,
    frame_index: int,
    voxel_size: float,
) -> None:
    key = feature_key(point, voxel_size)
    cell = cells[key]
    cell["positions"].append(point.astype(np.float64))
    cell["normals"].append(normal.astype(np.float64))
    cell["distances"].append(float(distance))
    cell["view_depths"].append(float(view_depth))
    cell["view_angles"].append(float(view_angle))
    cell["risk_count"] += 1 if risk else 0
    cell["boundary_risk_count"] += 1 if edge_like else 0
    cell["crease_risk_count"] += 1 if (risk and not edge_like) else 0
    cell["frames"].add(int(frame_index))
    cell["hits"] += 1


def dominant_distance_bin(distances: list[float]) -> str:
    counts: dict[str, int] = {}
    for distance in distances:
        label = bin_label(float(distance), DISTANCE_BINS)
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: item[1])[0]


def write_observation_feature_csv(
    out_dir: Path,
    cells: dict[tuple[int, int, int], dict[str, Any]],
    voxel_size: float,
) -> Path:
    feature_dir = out_dir / "virtual_observation_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    path = feature_dir / "point_observation_features.csv"
    header = [
        "voxelX",
        "voxelY",
        "voxelZ",
        "centerX",
        "centerY",
        "centerZ",
        "meanX",
        "meanY",
        "meanZ",
        "normalX",
        "normalY",
        "normalZ",
        "hit_count",
        "frame_count",
        "mean_distance",
        "min_distance",
        "max_distance",
        "dominant_distance_bin",
        "mean_view_depth",
        "mean_view_angle",
        "max_view_angle",
        "position_variance",
        "normal_variance",
        "depth_variance",
        "distance_variance",
        "mean_triangle_count",
        "mean_face_normal_angle",
        "boundary_risk_ratio",
        "crease_risk_ratio",
        "any_risk_ratio",
        "stability_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for key, cell in sorted(cells.items()):
            positions = np.asarray(cell["positions"], dtype=np.float64)
            normals = np.asarray(cell["normals"], dtype=np.float64)
            distances = np.asarray(cell["distances"], dtype=np.float64)
            depths = np.asarray(cell["view_depths"], dtype=np.float64)
            angles = np.asarray(cell["view_angles"], dtype=np.float64)
            mean_pos = np.mean(positions, axis=0)
            mean_normal = normalize(np.mean(normals, axis=0))
            hit_count = int(cell["hits"])
            frame_count = len(cell["frames"])
            risk_ratio = float(cell["risk_count"]) / max(1, hit_count)
            boundary_ratio = float(cell["boundary_risk_count"]) / max(1, hit_count)
            crease_ratio = float(cell["crease_risk_count"]) / max(1, hit_count)
            position_var = float(np.mean(np.sum((positions - mean_pos) ** 2, axis=1))) if hit_count > 1 else 0.0
            normal_var = float(np.mean(1.0 - np.clip(np.abs(normals @ mean_normal), 0.0, 1.0))) if hit_count > 1 else 0.0
            depth_var = float(np.var(depths)) if hit_count > 1 else 0.0
            distance_var = float(np.var(distances)) if hit_count > 1 else 0.0
            repeat_score = min(1.0, frame_count / 5.0)
            risk_score = max(0.0, 1.0 - risk_ratio)
            variance_score = max(0.0, 1.0 - min(1.0, position_var / 0.0064))
            stability_score = 0.45 * repeat_score + 0.35 * risk_score + 0.20 * variance_score
            center = (np.asarray(key, dtype=np.float64) + 0.5) * voxel_size
            writer.writerow(
                [
                    *key,
                    *[f"{value:.6f}" for value in center],
                    *[f"{value:.6f}" for value in mean_pos],
                    *[f"{value:.6f}" for value in mean_normal],
                    hit_count,
                    frame_count,
                    f"{float(np.mean(distances)):.6f}",
                    f"{float(np.min(distances)):.6f}",
                    f"{float(np.max(distances)):.6f}",
                    dominant_distance_bin(cell["distances"]),
                    f"{float(np.mean(depths)):.6f}",
                    f"{float(np.mean(angles)):.6f}",
                    f"{float(np.max(angles)):.6f}",
                    f"{position_var:.8f}",
                    f"{normal_var:.8f}",
                    f"{depth_var:.8f}",
                    f"{distance_var:.8f}",
                    "4.000000",
                    f"{float(np.mean(angles)):.6f}",
                    f"{boundary_ratio:.6f}",
                    f"{crease_ratio:.6f}",
                    f"{risk_ratio:.6f}",
                    f"{stability_score:.6f}",
                ]
            )
    summary = {
        "schema": "ScanCoverVirtualObservationFeatures/v1",
        "voxelSizeMeters": voxel_size,
        "voxelCount": len(cells),
        "hitCountMean": float(np.mean([cell["hits"] for cell in cells.values()])) if cells else 0.0,
        "frameCountMean": float(np.mean([len(cell["frames"]) for cell in cells.values()])) if cells else 0.0,
    }
    (feature_dir / "point_observation_features_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    sim = report["similarity"]
    verdict = report["verdict"]
    lines = [
        "# ScanCover Quest3 Virtual Clone Experiment",
        "",
        f"- Mode: {report['mode']}",
        f"- Truth mesh: `{report['truthMesh']}`",
        f"- Pose session: `{report['poseSession']}`",
        f"- Frames replayed: {report['framesReplayed']}",
        f"- Ray grid: {report['rayGrid']['width']} x {report['rayGrid']['height']}",
        f"- Mesh hit ratio: {report['hitRatio']:.4f}",
        f"- Accepted ratio: {report['acceptedRatio']:.4f}",
        f"- Simulated risk ratio: {report['riskRatio']:.4f}",
        f"- Distance mean abs share delta: {sim['distanceMeanAbsShareDelta']:.4f}",
        f"- Angle mean abs share delta: {sim['angleMeanAbsShareDelta']:.4f}",
        f"- Replica conclusion: {verdict['isReplicaDatasetConclusion']}",
        f"- Note: {verdict['note']}",
        "",
        "## Interpretation",
        "",
    ]
    if sim["angleMeanAbsShareDelta"] <= 0.03:
        lines.append("- View-angle distribution is close enough for the first virtual-observer validation.")
    else:
        lines.append("- View-angle distribution is not close enough; pose sampling needs adjustment.")
    if sim["distanceMeanAbsShareDelta"] <= 0.05:
        lines.append("- Distance distribution is close enough for the first virtual-observer validation.")
    else:
        lines.append(
            "- Distance distribution is not close enough yet; replayed poses must be fitted to the virtual room layout, "
            "not only translated to its center."
        )
    lines.extend(["", "## Files", "", "- `virtual_quest3_observed_points.ply`", "- `virtual_clone_similarity_report.json`", "- `similarity_distance_bins.csv`", "- `similarity_angle_bins.csv`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest) if args.manifest is not None else None
    if manifest is not None:
        manifest_real_summary = derived_output(manifest, "combinedObservationSummary")
        manifest_learning_profile = derived_output(manifest, "learningProfile")
        if manifest_real_summary is not None:
            args.real_summary = manifest_real_summary
        if args.learning_profile is None and manifest_learning_profile is not None:
            args.learning_profile = manifest_learning_profile
    if args.learning_profile is None:
        args.learning_profile = DEFAULT_LEARNING_PROFILE

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.truth_mesh.exists():
        raise FileNotFoundError(args.truth_mesh)
    if not args.pose_session.exists():
        raise FileNotFoundError(args.pose_session)
    if not args.real_summary.exists():
        raise FileNotFoundError(args.real_summary)

    out_dir = args.out / f"{args.mode}_{args.pose_source}_{args.truth_mesh.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    real_summary = json.loads(args.real_summary.read_text(encoding="utf-8-sig"))
    learning_profile = None
    if args.learning_profile and args.learning_profile.exists():
        learning_profile = json.loads(args.learning_profile.read_text(encoding="utf-8-sig"))
        profile_range = learning_profile.get("workingRangeMeters", {})
        args.max_distance = min(args.max_distance, float(profile_range.get("max", args.max_distance)))
    scan_feedback = None
    if args.scan_feedback and args.scan_feedback.exists():
        scan_feedback = json.loads(args.scan_feedback.read_text(encoding="utf-8-sig"))
        scan_feedback = filter_scan_feedback_for_mesh(scan_feedback, args.truth_mesh)

    mesh = load_legacy_mesh(args.truth_mesh)
    scene = build_scene(mesh)
    manifest_rows = pick_evenly(read_manifest(args.pose_session), args.max_frames)
    first_real_camera = load_camera(Path(manifest_rows[0]["cameraJson"])) if manifest_rows else {}
    default_fov = float(first_real_camera.get("camera", {}).get("fieldOfView", 100.2439))
    default_aspect = float(first_real_camera.get("camera", {}).get("aspect", args.width / max(1, args.height)))

    scan_metadata: dict[str, Any] = {"poseSource": args.pose_source}
    pose_offset = np.zeros(3, dtype=np.float64)
    if args.pose_source == "auto-scan":
        if args.scan_pattern == "stratified-slices":
            camera_frames, scan_metadata = build_stratified_slice_cameras(
                mesh=mesh,
                frame_count=args.max_frames,
                vertical_axis_name=args.vertical_axis,
                slice_grid=args.slice_grid,
                slice_heights=args.slice_heights,
                yaw_degrees=args.slice_yaw_degrees,
                fov_y=default_fov,
                aspect=default_aspect,
                sim_top_coverage_pass=args.sim_top_coverage_pass,
                sim_top_frame_share=args.sim_top_frame_share,
            )
        else:
            camera_frames, scan_metadata = build_auto_scan_cameras(
                mesh=mesh,
                frame_count=args.max_frames,
                vertical_axis_name=args.vertical_axis,
                head_height=args.head_height,
                radius_scale=args.scan_radius_scale,
                pitch_degrees=args.auto_pitch_degrees,
                fov_y=default_fov,
                aspect=default_aspect,
                coverage_pass=args.coverage_pass,
                coverage_frame_share=args.coverage_frame_share,
                scan_feedback=scan_feedback,
                force_bottom_coverage_priority=args.bottom_coverage_priority,
            )
        scan_metadata["poseSource"] = args.pose_source
        scan_metadata["scanFeedback"] = str(args.scan_feedback) if scan_feedback is not None else None
    else:
        camera_frames = [load_camera(Path(row["cameraJson"])) for row in manifest_rows]

    if args.center_poses_in_mesh and args.pose_source == "real":
        pose_positions = []
        for camera_data in camera_frames:
            pose_positions.append(np.asarray(camera_data["pose"]["position"], dtype=np.float64))
        pose_center = np.mean(np.asarray(pose_positions, dtype=np.float64), axis=0)
        mesh_center = np.asarray(mesh.get_axis_aligned_bounding_box().get_center(), dtype=np.float64)
        pose_offset = mesh_center - pose_center

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []
    distance_counts = {label: 0 for label, _, _ in DISTANCE_BINS}
    distance_risks = {label: 0 for label, _, _ in DISTANCE_BINS}
    angle_counts = {label: 0 for label, _, _ in ANGLE_BINS}
    angle_risks = {label: 0 for label, _, _ in ANGLE_BINS}
    attempted = 0
    hit = 0
    accepted = 0
    risk_total = 0
    feature_cells: dict[tuple[int, int, int], dict[str, Any]] = defaultdict(make_feature_cell)

    for frame_index, camera_data in enumerate(camera_frames):
        rays, ray_dirs = make_rays(camera_data, args.width, args.height, pose_offset)
        answer = scene.cast_rays(o3d.core.Tensor(rays))
        t_hit = answer["t_hit"].numpy()
        primitive_normals = answer["primitive_normals"].numpy()
        finite = np.isfinite(t_hit)

        attempted += len(t_hit)
        hit += int(np.sum(finite))
        if not np.any(finite):
            continue

        origins = rays[:, :3].astype(np.float64)
        directions = rays[:, 3:].astype(np.float64)
        points = origins + directions * t_hit[:, None]
        edge_like = estimate_edge_like(points - origins, args.width, args.height, finite)

        for i in np.where(finite)[0]:
            distance = float(t_hit[i])
            if distance < args.min_distance or distance > args.max_distance:
                continue

            normal = normalize(np.asarray(primitive_normals[i], dtype=np.float64))
            view_dir = normalize(-directions[i])
            angle = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(np.dot(normal, view_dir)))))))
            risk_p = risk_probability(distance, angle, bool(edge_like[i]), learning_profile)
            risk = random.random() < risk_p
            if random.random() < dropout_probability(distance, angle, risk, learning_profile):
                continue

            noise = np.random.normal(0.0, args.noise_std * (1.0 + 0.25 * distance), 3)
            observed_point = points[i] + noise
            view_depth = float(np.dot(observed_point - origins[i], directions[i]))
            add_feature_observation(
                feature_cells,
                observed_point,
                normal,
                distance,
                view_depth,
                angle,
                risk,
                bool(edge_like[i]),
                frame_index,
                args.feature_voxel,
            )
            all_points.append(observed_point)
            all_normals.append(normal)
            all_colors.append(color_for_point(distance, risk))

            d_label = bin_label(distance, DISTANCE_BINS)
            a_label = bin_label(angle, ANGLE_BINS)
            distance_counts[d_label] += 1
            angle_counts[a_label] += 1
            if risk:
                distance_risks[d_label] += 1
                angle_risks[a_label] += 1
                risk_total += 1
            accepted += 1

    points_array = np.asarray(all_points, dtype=np.float64)
    colors_array = np.asarray(all_colors, dtype=np.float64)
    normals_array = np.asarray(all_normals, dtype=np.float64)

    cloud_path = out_dir / "virtual_quest3_observed_points.ply"
    if len(points_array):
        write_cloud(cloud_path, points_array, colors_array, normals_array)
    feature_csv = write_observation_feature_csv(out_dir, feature_cells, args.feature_voxel)

    distance_rows = ratio_table(distance_counts, distance_risks, DISTANCE_BINS)
    angle_rows = ratio_table(angle_counts, angle_risks, ANGLE_BINS)
    write_csv(out_dir / "virtual_distance_bins.csv", distance_rows)
    write_csv(out_dir / "virtual_angle_bins.csv", angle_rows)

    distance_compare = compare_bins(real_summary["distanceBins"], distance_rows)
    angle_compare = compare_bins(real_summary["angleBins"], angle_rows)
    write_csv(out_dir / "similarity_distance_bins.csv", distance_compare)
    write_csv(out_dir / "similarity_angle_bins.csv", angle_compare)

    report = {
        "mode": args.mode,
        "poseSource": args.pose_source,
        "truthMesh": str(args.truth_mesh),
        "poseSession": str(args.pose_session),
        "poseOffset": pose_offset.tolist(),
        "scanMetadata": scan_metadata,
        "realSummary": str(args.real_summary),
        "learningProfile": str(args.learning_profile) if learning_profile is not None else None,
        "scanFeedback": str(args.scan_feedback) if scan_feedback is not None else None,
        "effectiveWorkingMaxDistance": args.max_distance,
        "framesReplayed": len(camera_frames),
        "rayGrid": {"width": args.width, "height": args.height},
        "attemptedRays": attempted,
        "meshHits": hit,
        "acceptedObservations": accepted,
        "hitRatio": hit / attempted if attempted else 0.0,
        "acceptedRatio": accepted / attempted if attempted else 0.0,
        "riskRatio": risk_total / accepted if accepted else 0.0,
        "observationFeatureCsv": str(feature_csv),
        "observationFeatureVoxels": len(feature_cells),
        "distanceBins": distance_rows,
        "angleBins": angle_rows,
        "similarity": {
            "distanceMeanAbsShareDelta": float(np.mean([abs(row["shareDelta"]) for row in distance_compare])),
            "angleMeanAbsShareDelta": float(np.mean([abs(row["shareDelta"]) for row in angle_compare])),
            "distanceBins": distance_compare,
            "angleBins": angle_compare,
        },
        "verdict": {
            "pipelineRan": True,
            "isReplicaDatasetConclusion": args.mode == "replica",
            "note": (
                "This is a proxy run because no local Replica room mesh was found."
                if args.mode == "proxy"
                else "This run used a Replica-labeled input mesh."
            ),
        },
    }
    report_path = out_dir / "virtual_clone_similarity_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path = out_dir / "virtual_clone_similarity_report.md"
    write_markdown_report(markdown_path, report)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote: {cloud_path}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
