#!/usr/bin/env python3
"""Trace unsupported V2 mesh triangles back to stereo/timing/TSDF evidence.

This is an offline, read-only audit.  It reconstructs only the aligned 10 cm
cell corners used by the final mesh (which are exact samples of the 5 cm source
field), then records where contradictory signs first entered those source
voxels.  It does not change replay, fusion, extraction, or display behavior.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from build_spatial_scar_receipt import frames, read_ply


SOURCE_VOXEL_METERS = 0.05
DERIVED_VOXEL_METERS = 0.10
MINIMUM_DEPTH_METERS = 0.30
MAXIMUM_DEPTH_METERS = 6.00
POSITIVE_TRUNCATION_SCALE = 0.20
NEGATIVE_TRUNCATION_SCALE = 0.20 / 3.0
GRADIENT_THRESHOLD_SQUARED = 0.01
MAXIMUM_WEIGHT = 1000.0
SIGN_EVIDENCE_METERS = SOURCE_VOXEL_METERS * 0.5
DIRECT_SUPPORT_METERS = SOURCE_VOXEL_METERS
LAYER_JUMP_METERS = DERIVED_VOXEL_METERS
REPEATED_SIGN_PAIRS = 3
REPEATED_SWITCHES = 3
SAME_VIEW_EYE_DISTANCE_METERS = 0.15
SAME_VIEW_RAY_ANGLE_DEGREES = 5.0
DOMINANT_ANCHOR_SEPARATION_METERS = 0.15
CORNER_OFFSETS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=np.int64)
EDGE_PAIRS = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
], dtype=np.int64)


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def xyz(value):
    return np.array([value[axis] for axis in "xyz"], dtype=np.float64)


def decode_depth_plane(ndc, width, height, world_to_clip,
                       projection_inverse, view_inverse, pixel_clip):
    valid_ndc = np.isfinite(ndc) & (ndc > 0.0) & (ndc < 1.0)
    clip = pixel_clip.copy()
    clip[:, 2] = ndc.astype(np.float64) * 2.0 - 1.0
    view_h = clip @ projection_inverse.T
    stage_h = view_h @ view_inverse.T
    valid = valid_ndc.copy()
    valid &= np.isfinite(stage_h[:, 3]) & (np.abs(stage_h[:, 3]) > 1e-5)
    stage = np.zeros((len(ndc), 3), dtype=np.float64)
    stage[valid] = stage_h[valid, :3] / stage_h[valid, 3, None]
    stage_h_normalized = np.column_stack((stage, np.ones(len(stage))))
    projected = stage_h_normalized @ world_to_clip.T
    depth = projected[:, 3]
    valid &= np.isfinite(stage).all(axis=1)
    valid &= np.isfinite(depth)
    valid &= (depth >= MINIMUM_DEPTH_METERS) & (depth <= MAXIMUM_DEPTH_METERS)
    depth[~valid] = 0.0

    image = depth.reshape(height, width)
    accepted = np.zeros((height, width), dtype=bool)
    center = image[1:-1, 1:-1]
    left = image[1:-1, :-2]
    right = image[1:-1, 2:]
    up = image[:-2, 1:-1]
    down = image[2:, 1:-1]
    neighbours = ((center > 0.0) & (left > 0.0) & (right > 0.0) &
                  (up > 0.0) & (down > 0.0))
    gradient = (right - left) ** 2 + (down - up) ** 2
    accepted[1:-1, 1:-1] = neighbours & (
        gradient <= GRADIENT_THRESHOLD_SQUARED)
    return depth, stage, accepted.reshape(-1)


def sample_voxels(points_h, width, height, world_to_clip, depth, stage,
                  accepted):
    clip = points_h @ world_to_clip.T
    voxel_depth = clip[:, 3]
    valid = np.isfinite(voxel_depth) & (voxel_depth > 1e-6)
    inverse = np.zeros(len(points_h), dtype=np.float64)
    inverse[valid] = 1.0 / voxel_depth[valid]
    u = clip[:, 0] * inverse * 0.5 + 0.5
    v = clip[:, 1] * inverse * 0.5 + 0.5
    valid &= np.isfinite(u) & np.isfinite(v)
    valid &= (u >= 0.0) & (u < 1.0) & (v >= 0.0) & (v < 1.0)
    x = np.clip((u * width).astype(np.int64), 0, width - 1)
    y = np.clip((v * height).astype(np.int64), 0, height - 1)
    index = x + y * width
    valid &= accepted[index] & (depth[index] > 0.0)
    residual = depth[index] - voxel_depth
    factor = voxel_depth + 1.0
    valid &= residual > -NEGATIVE_TRUNCATION_SCALE * factor
    sampled_distance = np.minimum(
        residual, POSITIVE_TRUNCATION_SCALE * factor)
    sampled_weight = 1.0 / np.maximum(1e-6, factor * factor)
    anchors = stage[index]
    valid &= np.isfinite(sampled_distance) & np.isfinite(sampled_weight)
    valid &= np.isfinite(anchors).all(axis=1)
    return valid, residual, sampled_distance, sampled_weight, anchors


def quantize_half(values):
    return values.astype(np.float16).astype(np.float64)


def minimum_nonnegative(values):
    valid = values[values >= 0]
    return int(np.min(valid)) if len(valid) else -1


def integrate(distance, weight, valid, sample_distance, sample_weight):
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return
    retained = np.minimum(
        weight[indices], MAXIMUM_WEIGHT - sample_weight[indices])
    retained = np.maximum(0.0, retained)
    new_weight = retained + sample_weight[indices]
    previous = distance[indices]
    combined = np.where(
        retained > 0.0,
        (previous * retained + sample_distance[indices] *
         sample_weight[indices]) / new_weight,
        sample_distance[indices])
    distance[indices] = quantize_half(combined)
    weight[indices] = quantize_half(new_weight)


def build_targets(vertices, triangles, spatial):
    triangle_receipts = spatial["triangleReceipts"]
    triangle_ids = np.array(
        [item["triangle"] for item in triangle_receipts], dtype=np.int64)
    centroids = vertices[triangles[triangle_ids]].mean(axis=1)
    # A Marching-Cubes triangle lies inside one world-anchored 10 cm cell.
    # A tiny positive epsilon keeps numerically negative boundary points in the
    # intended cell without moving ordinary samples.
    cells = np.floor(
        (centroids + 1e-7) / DERIVED_VOXEL_METERS).astype(np.int64)
    cell_corners = cells[:, None, :] + CORNER_OFFSETS[None, :, :]
    unique_corners, inverse = np.unique(
        cell_corners.reshape(-1, 3), axis=0, return_inverse=True)
    return triangle_receipts, triangle_ids, cells, unique_corners, inverse.reshape(-1, 8)


def audit(mesh_path: Path, tape_path: Path, spatial_path: Path,
          output_path: Path):
    vertices, triangles = read_ply(mesh_path)
    spatial = json.loads(spatial_path.read_text(encoding="utf-8-sig"))
    if sha256(mesh_path) != spatial["sourceMeshSha256"]:
        raise ValueError("Mesh SHA-256 does not match the spatial receipt")
    if "sourceTapeSha256" in spatial and sha256(tape_path) != spatial["sourceTapeSha256"]:
        raise ValueError("Tape SHA-256 does not match the spatial receipt")

    (triangle_receipts, triangle_ids, cells, corner_lattice,
     triangle_corner_indices) = build_targets(vertices, triangles, spatial)
    points = corner_lattice.astype(np.float64) * DERIVED_VOXEL_METERS
    points_h = np.column_stack((points, np.ones(len(points))))
    count = len(points)
    distance = np.zeros(count, dtype=np.float64)
    weight = np.zeros(count, dtype=np.float64)
    observation_pairs = np.zeros(count, dtype=np.int32)
    positive_pairs = np.zeros(count, dtype=np.int32)
    negative_pairs = np.zeros(count, dtype=np.int32)
    direct_pairs = np.zeros(count, dtype=np.int32)
    stereo_conflicts = np.zeros(count, dtype=np.int32)
    temporal_switches = np.zeros(count, dtype=np.int32)
    layer_jumps = np.zeros(count, dtype=np.int32)
    first_stereo_conflict = np.full(count, -1, dtype=np.int32)
    first_temporal_switch = np.full(count, -1, dtype=np.int32)
    first_layer_jump = np.full(count, -1, dtype=np.int32)
    previous_pair_sign = np.zeros(count, dtype=np.int8)
    previous_anchor = np.zeros((count, 3), dtype=np.float64)
    previous_anchor_valid = np.zeros(count, dtype=bool)

    matrix_inverse_errors = []
    stereo_baselines = []
    capture_times = []
    platform_frames = []
    pair_sign_history = []
    mid_eye_positions = []
    pair_anchor_history = []
    pair_anchor_valid_history = []
    frame_count = 0
    pixel_clip = None

    for (width, height, platform_frame, capture_time, matrices,
         left_ndc, right_ndc) in frames(tape_path):
        if pixel_clip is None:
            x = (np.arange(width, dtype=np.float64) + 0.5) / width
            y = (np.arange(height, dtype=np.float64) + 0.5) / height
            xx, yy = np.meshgrid(x * 2.0 - 1.0, y * 2.0 - 1.0)
            pixel_clip = np.column_stack((
                xx.reshape(-1), yy.reshape(-1),
                np.zeros(width * height), np.ones(width * height)))
        frame_count += 1
        platform_frames.append(platform_frame)
        capture_times.append(capture_time)
        left_eye = matrices[4][:3, 3]
        right_eye = matrices[5][:3, 3]
        mid_eye_positions.append((left_eye + right_eye) * 0.5)
        stereo_baselines.append(float(np.linalg.norm(left_eye - right_eye)))
        identity = np.eye(4)
        for eye in range(2):
            composed = matrices[eye] @ matrices[4 + eye] @ matrices[2 + eye]
            matrix_inverse_errors.append(float(np.max(np.abs(composed - identity))))

        sampled = []
        for eye, ndc in ((0, left_ndc), (1, right_ndc)):
            depth, stage, accepted = decode_depth_plane(
                ndc, width, height, matrices[eye], matrices[2 + eye],
                matrices[4 + eye], pixel_clip)
            result = sample_voxels(
                points_h, width, height, matrices[eye], depth, stage,
                accepted)
            sampled.append(result)
            integrate(distance, weight, result[0], result[2], result[3])

        valid_left, residual_left, _, _, anchor_left = sampled[0]
        valid_right, residual_right, _, _, anchor_right = sampled[1]
        sign_left = np.zeros(count, dtype=np.int8)
        sign_right = np.zeros(count, dtype=np.int8)
        evidence_left = valid_left & (np.abs(residual_left) >= SIGN_EVIDENCE_METERS)
        evidence_right = valid_right & (np.abs(residual_right) >= SIGN_EVIDENCE_METERS)
        sign_left[evidence_left] = np.where(
            residual_left[evidence_left] > 0.0, 1, -1)
        sign_right[evidence_right] = np.where(
            residual_right[evidence_right] > 0.0, 1, -1)

        both = (sign_left != 0) & (sign_right != 0)
        both_valid = valid_left & valid_right
        conflict = both & (sign_left != sign_right)
        stereo_conflicts[conflict] += 1
        first = conflict & (first_stereo_conflict < 0)
        first_stereo_conflict[first] = frame_count

        pair_sign = np.zeros(count, dtype=np.int8)
        agree = both & ~conflict
        pair_sign[agree] = sign_left[agree]
        only_left = (sign_left != 0) & (sign_right == 0)
        only_right = (sign_right != 0) & (sign_left == 0)
        pair_sign[only_left] = sign_left[only_left]
        pair_sign[only_right] = sign_right[only_right]
        pair_sign_history.append(pair_sign.copy())
        has_pair = pair_sign != 0
        observation_pairs[has_pair] += 1
        positive_pairs[pair_sign > 0] += 1
        negative_pairs[pair_sign < 0] += 1
        direct = ((valid_left & (np.abs(residual_left) <= DIRECT_SUPPORT_METERS)) |
                  (valid_right & (np.abs(residual_right) <= DIRECT_SUPPORT_METERS)))
        direct_pairs[direct] += 1

        switched = has_pair & (previous_pair_sign != 0) & (
            pair_sign != previous_pair_sign)
        temporal_switches[switched] += 1
        first = switched & (first_temporal_switch < 0)
        first_temporal_switch[first] = frame_count
        previous_pair_sign[has_pair] = pair_sign[has_pair]

        anchor = np.zeros((count, 3), dtype=np.float64)
        anchor_valid = valid_left | valid_right
        anchor[valid_left] = anchor_left[valid_left]
        anchor[valid_right] += anchor_right[valid_right]
        anchor[both_valid] *= 0.5
        jumped = anchor_valid & previous_anchor_valid & (
            np.linalg.norm(anchor - previous_anchor, axis=1) >= LAYER_JUMP_METERS)
        layer_jumps[jumped] += 1
        first = jumped & (first_layer_jump < 0)
        first_layer_jump[first] = frame_count
        previous_anchor[anchor_valid] = anchor[anchor_valid]
        previous_anchor_valid[anchor_valid] = True
        pair_anchor_history.append(anchor.copy())
        pair_anchor_valid_history.append(anchor_valid.copy())

    corner_observed = weight > 0.0
    corner_final_sign = np.zeros(count, dtype=np.int8)
    corner_final_sign[corner_observed] = np.where(
        distance[corner_observed] < 0.0, -1, 1)
    cell_signs = corner_final_sign[triangle_corner_indices]
    cell_observed = np.all(cell_signs != 0, axis=1)
    cell_crossing = cell_observed & (np.min(cell_signs, axis=1) < 0) & (
        np.max(cell_signs, axis=1) > 0)

    sign_history = np.stack(pair_sign_history, axis=0)
    mid_eyes = np.asarray(mid_eye_positions, dtype=np.float64)
    anchor_history = np.stack(pair_anchor_history, axis=0)
    anchor_valid_history = np.stack(pair_anchor_valid_history, axis=0)
    anchor_history[~anchor_valid_history] = np.nan
    with np.errstate(invalid="ignore"):
        dominant_anchor = np.nanmedian(anchor_history, axis=0)
    dominant_anchor_valid = np.isfinite(dominant_anchor).all(axis=1)
    same_view_mixed = np.zeros(count, dtype=bool)
    view_separated_mixed = np.zeros(count, dtype=bool)
    minimum_cross_sign_eye_distance = np.full(count, np.inf)
    minimum_cross_sign_ray_angle = np.full(count, np.inf)
    repeated_mixed = ((positive_pairs >= REPEATED_SIGN_PAIRS) &
                      (negative_pairs >= REPEATED_SIGN_PAIRS))
    for corner in np.flatnonzero(repeated_mixed):
        positive = np.flatnonzero(sign_history[:, corner] > 0)
        negative = np.flatnonzero(sign_history[:, corner] < 0)
        eye_delta = mid_eyes[positive, None, :] - mid_eyes[None, negative, :]
        eye_distance = np.linalg.norm(eye_delta, axis=2)
        positive_rays = points[corner] - mid_eyes[positive]
        negative_rays = points[corner] - mid_eyes[negative]
        positive_rays /= np.linalg.norm(positive_rays, axis=1)[:, None]
        negative_rays /= np.linalg.norm(negative_rays, axis=1)[:, None]
        cosine = np.clip(positive_rays @ negative_rays.T, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))
        minimum_cross_sign_eye_distance[corner] = float(np.min(eye_distance))
        minimum_cross_sign_ray_angle[corner] = float(np.min(angle))
        close_view = ((eye_distance <= SAME_VIEW_EYE_DISTANCE_METERS) &
                      (angle <= SAME_VIEW_RAY_ANGLE_DEGREES))
        same_view_mixed[corner] = bool(np.any(close_view))
        view_separated_mixed[corner] = not same_view_mixed[corner]

    cell_distances = distance[triangle_corner_indices]
    gx = (cell_distances[:, 1] + cell_distances[:, 2] +
          cell_distances[:, 5] + cell_distances[:, 6] -
          cell_distances[:, 0] - cell_distances[:, 3] -
          cell_distances[:, 4] - cell_distances[:, 7]) * 0.25
    gy = (cell_distances[:, 2] + cell_distances[:, 3] +
          cell_distances[:, 6] + cell_distances[:, 7] -
          cell_distances[:, 0] - cell_distances[:, 1] -
          cell_distances[:, 4] - cell_distances[:, 5]) * 0.25
    gz = (cell_distances[:, 4] + cell_distances[:, 5] +
          cell_distances[:, 6] + cell_distances[:, 7] -
          cell_distances[:, 0] - cell_distances[:, 1] -
          cell_distances[:, 2] - cell_distances[:, 3]) * 0.25
    cell_normals = np.column_stack((gx, gy, gz))
    normal_length = np.linalg.norm(cell_normals, axis=1)
    normal_valid = normal_length > 1e-6
    cell_normals[normal_valid] /= normal_length[normal_valid, None]
    dominant_anchor_incompatible_edges = np.zeros(
        len(triangle_ids), dtype=np.int32)
    dominant_anchor_comparable_edges = np.zeros(
        len(triangle_ids), dtype=np.int32)
    dominant_anchor_maximum_separation = np.zeros(
        len(triangle_ids), dtype=np.float64)
    for edge_a, edge_b in EDGE_PAIRS:
        corner_a = triangle_corner_indices[:, edge_a]
        corner_b = triangle_corner_indices[:, edge_b]
        crossing = ((cell_distances[:, edge_a] < 0.0) !=
                    (cell_distances[:, edge_b] < 0.0))
        comparable = (crossing & normal_valid &
                      dominant_anchor_valid[corner_a] &
                      dominant_anchor_valid[corner_b])
        separation = np.zeros(len(triangle_ids), dtype=np.float64)
        delta = dominant_anchor[corner_a] - dominant_anchor[corner_b]
        separation[comparable] = np.abs(np.sum(
            delta[comparable] * cell_normals[comparable], axis=1))
        dominant_anchor_comparable_edges[comparable] += 1
        incompatible = comparable & (
            separation > DOMINANT_ANCHOR_SEPARATION_METERS)
        dominant_anchor_incompatible_edges[incompatible] += 1
        dominant_anchor_maximum_separation = np.maximum(
            dominant_anchor_maximum_separation, separation)

    triangle_records = []
    overlap_counts = {
        "samePairStereoOpposition": 0,
        "repeatedSamePairStereoOpposition": 0,
        "anyTemporalSignSwitching": 0,
        "repeatedTemporalSignSwitching": 0,
        "repeatedMixedSignSourceVoxel": 0,
        "sameViewRepeatedMixedSignSourceVoxel": 0,
        "viewSeparatedRepeatedMixedSignSourceVoxel": 0,
        "dominantAnchorLayerBridge": 0,
        "noRepeatedMixedCornerZeroCrossing": 0,
        "reconstructedCellCrossing": 0,
    }
    cluster_totals = {}
    for index, item in enumerate(triangle_receipts):
        corners = triangle_corner_indices[index]
        stereo = bool(np.any(stereo_conflicts[corners] > 0))
        stereo_repeated = bool(np.any(stereo_conflicts[corners] >= 3))
        temporal_any = bool(np.any(
            (temporal_switches[corners] > 0) &
            (positive_pairs[corners] > 0) &
            (negative_pairs[corners] > 0)))
        temporal_repeated = bool(np.any(
            (temporal_switches[corners] >= REPEATED_SWITCHES) &
            repeated_mixed[corners]))
        mixed = bool(np.any(repeated_mixed[corners]))
        same_view = bool(np.any(same_view_mixed[corners]))
        view_separated = bool(np.any(view_separated_mixed[corners]))
        dominant_bridge = bool(
            dominant_anchor_incompatible_edges[index] > 0)
        no_repeated_mixed = bool(cell_crossing[index] and not mixed)
        mechanisms = []
        if stereo:
            mechanisms.append("SamePairStereoOpposition")
            overlap_counts["samePairStereoOpposition"] += 1
        if stereo_repeated:
            mechanisms.append("RepeatedSamePairStereoOpposition")
            overlap_counts["repeatedSamePairStereoOpposition"] += 1
        if temporal_any:
            overlap_counts["anyTemporalSignSwitching"] += 1
        if temporal_repeated:
            mechanisms.append("RepeatedTemporalSignSwitching")
            overlap_counts["repeatedTemporalSignSwitching"] += 1
        if mixed:
            mechanisms.append("RepeatedMixedSignSourceVoxel")
            overlap_counts["repeatedMixedSignSourceVoxel"] += 1
        if same_view:
            mechanisms.append("SameViewRepeatedMixedSignSourceVoxel")
            overlap_counts["sameViewRepeatedMixedSignSourceVoxel"] += 1
        if view_separated:
            mechanisms.append("ViewSeparatedRepeatedMixedSignSourceVoxel")
            overlap_counts["viewSeparatedRepeatedMixedSignSourceVoxel"] += 1
        if dominant_bridge:
            mechanisms.append("DominantAnchorLayerBridge")
            overlap_counts["dominantAnchorLayerBridge"] += 1
        if no_repeated_mixed:
            mechanisms.append("NoRepeatedMixedCornerZeroCrossing")
            overlap_counts["noRepeatedMixedCornerZeroCrossing"] += 1
        if cell_crossing[index]:
            overlap_counts["reconstructedCellCrossing"] += 1
        finite_eye = minimum_cross_sign_eye_distance[corners]
        finite_eye = finite_eye[np.isfinite(finite_eye)]
        finite_angle = minimum_cross_sign_ray_angle[corners]
        finite_angle = finite_angle[np.isfinite(finite_angle)]
        record = {
            "triangle": int(triangle_ids[index]),
            "cluster": int(item["cluster"]),
            "spatialClassification": item["classification"],
            "cell": [int(value) for value in cells[index]],
            "reconstructedCellCrossing": bool(cell_crossing[index]),
            "dominantAnchorComparableCrossingEdges": int(
                dominant_anchor_comparable_edges[index]),
            "dominantAnchorIncompatibleCrossingEdges": int(
                dominant_anchor_incompatible_edges[index]),
            "dominantAnchorMaximumNormalSeparationMeters": float(
                dominant_anchor_maximum_separation[index]),
            "mechanisms": mechanisms,
            "cornerEvidence": {
                "observed": int(np.count_nonzero(corner_observed[corners])),
                "directPairMaximum": int(np.max(direct_pairs[corners])),
                "positivePairMaximum": int(np.max(positive_pairs[corners])),
                "negativePairMaximum": int(np.max(negative_pairs[corners])),
                "stereoConflictMaximum": int(np.max(stereo_conflicts[corners])),
                "temporalSwitchMaximum": int(np.max(temporal_switches[corners])),
                "layerJumpMaximum": int(np.max(layer_jumps[corners])),
                "sameViewRepeatedMixedCorners": int(np.count_nonzero(
                    same_view_mixed[corners])),
                "viewSeparatedRepeatedMixedCorners": int(np.count_nonzero(
                    view_separated_mixed[corners])),
                "minimumCrossSignEyeDistanceMeters": (
                    float(np.min(finite_eye)) if len(finite_eye) else -1.0),
                "minimumCrossSignRayAngleDegrees": (
                    float(np.min(finite_angle)) if len(finite_angle) else -1.0),
                "firstStereoConflictPair": minimum_nonnegative(
                    first_stereo_conflict[corners]),
                "firstTemporalSwitchPair": minimum_nonnegative(
                    first_temporal_switch[corners]),
                "firstLayerJumpPair": minimum_nonnegative(
                    first_layer_jump[corners]),
            },
        }
        triangle_records.append(record)
        cluster = cluster_totals.setdefault(int(item["cluster"]), {
            "triangles": 0,
            "samePairStereoOpposition": 0,
            "repeatedSamePairStereoOpposition": 0,
            "repeatedTemporalSignSwitching": 0,
            "repeatedMixedSignSourceVoxel": 0,
            "sameViewRepeatedMixedSignSourceVoxel": 0,
            "viewSeparatedRepeatedMixedSignSourceVoxel": 0,
            "dominantAnchorLayerBridge": 0,
            "noRepeatedMixedCornerZeroCrossing": 0,
            "reconstructedCellCrossing": 0,
        })
        cluster["triangles"] += 1
        for mechanism in mechanisms:
            key = mechanism[0].lower() + mechanism[1:]
            cluster[key] += 1
        if cell_crossing[index]:
            cluster["reconstructedCellCrossing"] += 1

    cluster_records = [
        {"cluster": cluster, **values}
        for cluster, values in sorted(cluster_totals.items())
    ]
    frame_steps = np.diff(np.asarray(platform_frames, dtype=np.int64))
    time_steps = np.diff(np.asarray(capture_times, dtype=np.float64))
    receipt = {
        "audit": "RoomSetupReplicaV2ScarSourceAuditV1",
        "coordinateSpace": "QuestStage",
        "sourceMesh": str(mesh_path),
        "sourceMeshSha256": sha256(mesh_path),
        "sourceTape": str(tape_path),
        "sourceTapeSha256": sha256(tape_path),
        "sourceSpatialReceipt": str(spatial_path),
        "sourceSpatialReceiptSha256": sha256(spatial_path),
        "thresholds": {
            "sourceVoxelMeters": SOURCE_VOXEL_METERS,
            "derivedVoxelMeters": DERIVED_VOXEL_METERS,
            "signEvidenceMeters": SIGN_EVIDENCE_METERS,
            "directSupportMeters": DIRECT_SUPPORT_METERS,
            "layerJumpMeters": LAYER_JUMP_METERS,
            "repeatedSignPairs": REPEATED_SIGN_PAIRS,
            "repeatedSwitches": REPEATED_SWITCHES,
            "sameViewEyeDistanceMeters": SAME_VIEW_EYE_DISTANCE_METERS,
            "sameViewRayAngleDegrees": SAME_VIEW_RAY_ANGLE_DEGREES,
            "dominantAnchorSeparationMeters":
                DOMINANT_ANCHOR_SEPARATION_METERS,
        },
        "matrixAndTimingAudit": {
            "pairs": frame_count,
            "worldClipInverseMaximumAbsoluteError": float(
                np.max(matrix_inverse_errors)),
            "worldClipInverseP95AbsoluteError": float(
                np.percentile(matrix_inverse_errors, 95)),
            "stereoBaselineMeanMeters": float(np.mean(stereo_baselines)),
            "stereoBaselineMinimumMeters": float(np.min(stereo_baselines)),
            "stereoBaselineMaximumMeters": float(np.max(stereo_baselines)),
            "nonIncreasingCaptureTimes": int(np.count_nonzero(time_steps <= 0.0)),
            "nonIncreasingPlatformFrames": int(np.count_nonzero(frame_steps <= 0)),
            "platformFrameGapMaximum": int(np.max(frame_steps, initial=0)),
            "captureIntervalMeanSeconds": float(np.mean(time_steps)),
            "captureIntervalMaximumSeconds": float(np.max(time_steps, initial=0.0)),
        },
        "reconstructionCheck": {
            "unsupportedTriangles": len(triangle_records),
            "uniqueDerivedCells": int(len(np.unique(cells, axis=0))),
            "uniqueAlignedSourceCorners": count,
            "reconstructedCellCrossings": int(np.count_nonzero(cell_crossing)),
            "reconstructedCellCrossingRatio": float(np.mean(cell_crossing)),
            "note": "Target-only fusion approximates candidate-chunk admission; use the crossing ratio as its validity boundary.",
        },
        "mechanismOverlapCounts": overlap_counts,
        "limitations": [
            "A same-pair stereo conflict proves recorded left/right disagreement at the target voxel, but does not alone distinguish depth-edge noise from pre-callback calibration error.",
            "Temporal switching proves recorded sign alternation at an aligned 5 cm source sample; it does not by itself identify the unavailable native uncertainty value.",
            "Pose matrices can be checked for internal inverse/timing consistency only; the tape has no independent external pose ground truth.",
            "Stable inter-corner crossing is an extraction amplification candidate, not proof that the underlying physical surface is absent.",
        ],
        "clusterReceipts": cluster_records,
        "triangleReceipts": triangle_records,
    }
    output_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("tape", type=Path)
    parser.add_argument("spatial_receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt = audit(
        args.mesh, args.tape, args.spatial_receipt, args.output)
    print(json.dumps({
        "matrixAndTimingAudit": receipt["matrixAndTimingAudit"],
        "reconstructionCheck": receipt["reconstructionCheck"],
        "mechanismOverlapCounts": receipt["mechanismOverlapCounts"],
    }, indent=2))


if __name__ == "__main__":
    main()
