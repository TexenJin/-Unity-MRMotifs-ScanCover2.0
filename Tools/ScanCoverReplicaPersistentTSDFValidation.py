#!/usr/bin/env python3
"""Persistent Open3D TSDF validation across Replica rooms.

For every room, the same camera path feeds two persistent volumes:
an ideal depth control and the fitted Quest-style degraded depth stream.
The output separates observation/path limits from fusion-induced loss, extras,
regression and steady per-frame integration cost.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from ScanCoverQuest3VirtualCloneExperiment import (
    STRUCTURE_LABELS,
    build_scene,
    build_mesh_structure_reference,
    build_stratified_slice_cameras,
    classify_structural_pixels,
    degradation_parameters,
    distance_to_reference,
    dropout_probability,
    estimate_edge_like,
    load_legacy_mesh,
    make_rays,
    normalize,
    pick_evenly,
    risk_probability,
    sample_depth_noise,
    structured_edge_observation_range,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meshes", type=Path, nargs="+", required=True)
    parser.add_argument("--degradation-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--voxel", type=float, default=0.02)
    parser.add_argument("--sdf-trunc", type=float, default=0.08)
    parser.add_argument("--min-distance", type=float, default=0.35)
    parser.add_argument("--max-distance", type=float, default=5.0)
    parser.add_argument("--truth-samples", type=int, default=50000)
    parser.add_argument("--checkpoint-every", type=int, default=30)
    parser.add_argument(
        "--camera-clearance",
        type=float,
        default=0.0,
        help="Reject synthetic camera origins closer than this distance to Replica geometry.",
    )
    parser.add_argument(
        "--camera-selection",
        choices=["even", "visibility-balanced"],
        default="even",
        help="Select eligible cameras evenly or by balanced new/repeat surface visibility.",
    )
    parser.add_argument("--selection-width", type=int, default=32)
    parser.add_argument("--selection-height", type=int, default=24)
    parser.add_argument("--selection-voxel", type=float, default=0.10)
    parser.add_argument("--publish-chunk", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=15319)
    parser.add_argument("--structured-edge-degradation", action="store_true")
    parser.add_argument("--structure-depth-jump", type=float, default=0.06)
    parser.add_argument("--structure-crease-degrees", type=float, default=32.0)
    parser.add_argument("--structure-band-meters", type=float, default=0.08)
    parser.add_argument(
        "--ownership-strong-jump-scale",
        type=float,
        default=1.5,
        help="Hold only far-side edge samples whose competing nearer layer exceeds this multiple of the jump threshold.",
    )
    parser.add_argument(
        "--ownership-min-mix-alpha",
        type=float,
        default=0.30,
        help="Minimum fractional distance from the nearer legal layer endpoint within the observed depth span.",
    )
    parser.add_argument(
        "--ownership-min-endpoint-residual",
        type=float,
        default=0.03,
        help="Minimum distance from both legal layer centers before a mixed observation is held.",
    )
    return parser.parse_args()


def percentile(values: list[float] | np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if len(values) else 0.0


def camera_calibration(camera_data: dict[str, Any], width: int, height: int) -> tuple[o3d.camera.PinholeCameraIntrinsic, np.ndarray, np.ndarray]:
    pose = camera_data["pose"]
    camera = camera_data["camera"]
    right = normalize(np.asarray(pose["right"], dtype=np.float64))
    up = normalize(np.asarray(pose["up"], dtype=np.float64))
    forward = normalize(np.asarray(pose["forward"], dtype=np.float64))
    origin = np.asarray(pose["position"], dtype=np.float64)
    fov_y = math.radians(float(camera["fieldOfView"]))
    aspect = float(camera["aspect"])
    tan_y = math.tan(0.5 * fov_y)
    tan_x = tan_y * aspect
    fx = width / (2.0 * tan_x)
    fy = height / (2.0 * tan_y)
    cx = (width - 1.0) * 0.5
    cy = (height - 1.0) * 0.5
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, 0] = right
    camera_to_world[:3, 1] = -up
    camera_to_world[:3, 2] = forward
    camera_to_world[:3, 3] = origin
    return intrinsic, np.linalg.inv(camera_to_world), forward


def new_volume(voxel: float, sdf_trunc: float) -> o3d.pipelines.integration.ScalableTSDFVolume:
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )


def integrate_depth(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
    depth: np.ndarray,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    extrinsic: np.ndarray,
    max_distance: float,
    color: o3d.geometry.Image,
) -> float:
    started = time.perf_counter()
    depth_image = o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32)))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth_image,
        depth_scale=1.0,
        depth_trunc=max_distance,
        convert_rgb_to_intensity=False,
    )
    volume.integrate(rgbd, intrinsic, extrinsic)
    return (time.perf_counter() - started) * 1000.0


def make_depth_pair(
    scene: o3d.t.geometry.RaycastingScene,
    camera_data: dict[str, Any],
    width: int,
    height: int,
    min_distance: float,
    max_distance: float,
    model: dict[str, Any],
    structured_edge_degradation: bool = False,
    structure_depth_jump: float = 0.06,
    structure_crease_degrees: float = 32.0,
    ownership_strong_jump_scale: float = 1.5,
    ownership_min_mix_alpha: float = 0.45,
    ownership_min_endpoint_residual: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, dict[str, Any]]:
    rays, _ = make_rays(camera_data, width, height, np.zeros(3, dtype=np.float64))
    answer = scene.cast_rays(o3d.core.Tensor(rays))
    t_hit = answer["t_hit"].numpy()
    primitive_normals = answer["primitive_normals"].numpy()
    finite = np.isfinite(t_hit)
    origins = rays[:, :3].astype(np.float64)
    directions = rays[:, 3:].astype(np.float64)
    safe_hit = np.where(finite, t_hit, 0.0)
    points = origins + directions * safe_hit[:, None]
    edge_like = estimate_edge_like(points - origins, width, height, finite)
    structure_labels, structure_partner_depths = classify_structural_pixels(
        t_hit,
        primitive_normals,
        rays,
        width,
        height,
        finite,
        structure_depth_jump,
        structure_crease_degrees,
    )
    _, _, forward = camera_calibration(camera_data, width, height)

    ideal = np.zeros(width * height, dtype=np.float32)
    quest_ungated = np.zeros(width * height, dtype=np.float32)
    quest_guarded = np.zeros(width * height, dtype=np.float32)
    observed_ranges = np.full(width * height, np.nan, dtype=np.float64)
    observed_points = np.full((width * height, 3), np.nan, dtype=np.float64)
    valid = np.where(finite & (t_hit >= min_distance) & (t_hit <= max_distance))[0]
    accepted_ungated = 0
    accepted_guarded = 0
    structure_stats = {
        label: {"candidates": 0, "mixed": 0, "ambiguous": 0, "held": 0}
        for label in STRUCTURE_LABELS.values()
    }
    for index in valid:
        direction = directions[index]
        axial_factor = float(np.dot(direction, forward))
        if axial_factor <= 1e-6:
            continue
        ideal[index] = float(t_hit[index] * axial_factor)
        normal = normalize(np.asarray(primitive_normals[index], dtype=np.float64))
        view_dir = normalize(-direction)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(np.dot(normal, view_dir)))))))
        risk = random.random() < risk_probability(float(t_hit[index]), angle, bool(edge_like[index]), None)
        sigma, measured_dropout = degradation_parameters(float(t_hit[index]), angle, model, 0.008)
        dropout = dropout_probability(float(t_hit[index]), angle, risk, None)
        if measured_dropout >= 0.0:
            dropout = min(0.7, measured_dropout + (0.02 if risk else 0.0) + (0.02 if angle >= 75.0 else 0.0))
        if random.random() < dropout:
            continue
        structure_label = int(structure_labels[index])
        structure_name = STRUCTURE_LABELS.get(structure_label, "smooth")
        structure_stats[structure_name]["candidates"] += 1
        observed_range = float(t_hit[index])
        ownership_ambiguous = False
        mix_alpha = 0.0
        if structured_edge_degradation:
            observed_range, ownership_ambiguous, mix_alpha = structured_edge_observation_range(
                observed_range,
                float(structure_partner_depths[index]),
                structure_label,
                angle,
                model,
            )
        if mix_alpha > 0.0:
            structure_stats[structure_name]["mixed"] += 1
        if ownership_ambiguous:
            structure_stats[structure_name]["ambiguous"] += 1
        observed = origins[index] + direction * observed_range + sample_depth_noise(direction, sigma)
        z_depth = float(np.dot(observed - origins[index], forward))
        if 0.0 < z_depth <= max_distance:
            quest_ungated[index] = z_depth
            observed_ranges[index] = float(np.linalg.norm(observed - origins[index]))
            observed_points[index] = observed
            accepted_ungated += 1

    quest_guarded[:] = quest_ungated
    accepted_guarded = accepted_ungated
    for index in np.where(quest_ungated > 0.0)[0]:
        x = int(index % width)
        y = int(index // width)
        center_depth = float(observed_ranges[index])
        if not math.isfinite(center_depth):
            continue
        threshold = max(0.07, center_depth * 0.03)
        neighbor_indices: list[int] = []
        same_layer_depths = [center_depth]
        closer = 0
        farther = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                neighbor = ny * width + nx
                neighbor_depth = float(observed_ranges[neighbor])
                if not math.isfinite(neighbor_depth):
                    continue
                neighbor_indices.append(neighbor)
                if neighbor_depth < center_depth - threshold:
                    closer += 1
                elif neighbor_depth > center_depth + threshold:
                    farther += 1
                else:
                    same_layer_depths.append(neighbor_depth)

        center_normal = normalize(np.asarray(primitive_normals[index], dtype=np.float64))
        view_facing = abs(float(np.dot(center_normal, normalize(-directions[index]))))
        plane_compatible = 0
        if 0.15 <= view_facing < 0.50:
            for neighbor in neighbor_indices:
                neighbor_normal = normalize(np.asarray(primitive_normals[neighbor], dtype=np.float64))
                normal_dot = abs(float(np.dot(center_normal, neighbor_normal)))
                plane_residual = abs(float(np.dot(observed_points[neighbor] - observed_points[index], center_normal)))
                if normal_dot >= 0.85 and plane_residual <= 0.035:
                    plane_compatible += 1
        grazing_supported = 0.15 <= view_facing < 0.50 and plane_compatible >= 5
        same_neighbor_count = max(0, len(same_layer_depths) - 1)
        one_sided_competing_layer = (closer > 0) != (farther > 0)
        separated_layer_rescue = one_sided_competing_layer and same_neighbor_count >= 3
        depth_gate_passed = same_neighbor_count >= 5 or separated_layer_rescue or grazing_supported
        if not depth_gate_passed:
            quest_guarded[index] = 0.0
            accepted_guarded -= 1
            structure_name = STRUCTURE_LABELS.get(int(structure_labels[index]), "smooth")
            structure_stats[structure_name]["held"] += 1
            continue

        if not grazing_supported and len(same_layer_depths) >= 3:
            median_depth = float(np.median(np.asarray(same_layer_depths, dtype=np.float64)))
            if abs(center_depth - median_depth) > 0.055:
                axial_factor = float(np.dot(directions[index], forward))
                corrected_z = median_depth * axial_factor
                if 0.0 < corrected_z <= max_distance:
                    quest_guarded[index] = corrected_z
    return (
        ideal.reshape(height, width),
        quest_ungated.reshape(height, width),
        quest_guarded.reshape(height, width),
        int(len(valid)),
        accepted_ungated,
        accepted_guarded,
        structure_stats,
    )


def observed_surface_chunks(
    camera_data: dict[str, Any],
    depth: np.ndarray,
    width: int,
    height: int,
    chunk_size: float,
) -> set[tuple[int, int, int]]:
    rays, _ = make_rays(camera_data, width, height, np.zeros(3, dtype=np.float64))
    _, _, forward = camera_calibration(camera_data, width, height)
    flat_depth = depth.reshape(-1).astype(np.float64)
    valid = flat_depth > 0.0
    if not np.any(valid):
        return set()
    directions = rays[valid, 3:].astype(np.float64)
    axial = directions @ forward
    safe = axial > 1e-6
    if not np.any(safe):
        return set()
    origins = rays[valid, :3].astype(np.float64)[safe]
    directions = directions[safe]
    ranges = flat_depth[valid][safe] / axial[safe]
    points = origins + directions * ranges[:, None]
    keys = np.unique(np.floor(points / chunk_size).astype(np.int64), axis=0)
    return {tuple(int(value) for value in key) for key in keys}


def select_visibility_balanced_cameras(
    scene: o3d.t.geometry.RaycastingScene,
    candidates: list[dict[str, Any]],
    count: int,
    width: int,
    height: int,
    min_distance: float,
    max_distance: float,
    surface_voxel: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedily balance new surface coverage and repeated multi-view support."""
    voxel_ids: dict[tuple[int, int, int], int] = {}
    candidate_ids: list[np.ndarray] = []
    for camera in candidates:
        rays, _ = make_rays(camera, width, height, np.zeros(3, dtype=np.float64))
        answer = scene.cast_rays(o3d.core.Tensor(rays))
        hit = answer["t_hit"].numpy()
        valid = np.isfinite(hit) & (hit >= min_distance) & (hit <= max_distance)
        if not np.any(valid):
            candidate_ids.append(np.empty(0, dtype=np.int32))
            continue
        points = rays[valid, :3].astype(np.float64) + rays[valid, 3:].astype(np.float64) * hit[valid, None]
        keys = np.unique(np.floor(points / surface_voxel).astype(np.int64), axis=0)
        ids = []
        for key_array in keys:
            key = tuple(int(value) for value in key_array)
            if key not in voxel_ids:
                voxel_ids[key] = len(voxel_ids)
            ids.append(voxel_ids[key])
        candidate_ids.append(np.asarray(ids, dtype=np.int32))

    support = np.zeros(len(voxel_ids), dtype=np.int16)
    remaining = np.ones(len(candidates), dtype=bool)
    selected_indices: list[int] = []
    for _ in range(min(count, len(candidates))):
        best_index = -1
        best_score = -1.0
        for index, ids in enumerate(candidate_ids):
            if not remaining[index] or len(ids) == 0:
                continue
            current = support[ids]
            # New cells dominate early; inverse support keeps later frames useful
            # for maturity instead of repeatedly observing already dense cells.
            score = float(np.sum(1.0 / (1.0 + current)) + 0.35 * np.sum(current == 0))
            if score > best_score:
                best_score = score
                best_index = index
        if best_index < 0:
            break
        selected_indices.append(best_index)
        remaining[best_index] = False
        support[candidate_ids[best_index]] += 1

    if len(selected_indices) < count:
        unused = [index for index in range(len(candidates)) if remaining[index]]
        selected_indices.extend(pick_evenly(unused, count - len(selected_indices)))
    selected = [candidates[index] for index in selected_indices[:count]]
    return selected, {
        "selectionMode": "visibility-balanced",
        "selectionRayGrid": [width, height],
        "selectionSurfaceVoxelMeters": surface_voxel,
        "visibleSurfaceVoxels": int(len(support)),
        "selectedCoveredSurfaceVoxels": int(np.sum(support > 0)),
        "selectedRepeated2SurfaceVoxels": int(np.sum(support >= 2)),
        "selectedRepeated3SurfaceVoxels": int(np.sum(support >= 3)),
        "selectedRepeat2Ratio": float(np.mean(support >= 2)) if len(support) else 0.0,
        "selectedRepeat3Ratio": float(np.mean(support >= 3)) if len(support) else 0.0,
    }


def cloud_metrics(
    truth: o3d.geometry.PointCloud,
    truth_scene: o3d.t.geometry.RaycastingScene,
    reconstructed: o3d.geometry.PointCloud,
    structure_reference: dict[str, np.ndarray] | None = None,
    structure_band_meters: float = 0.08,
) -> tuple[dict[str, Any], np.ndarray]:
    if len(reconstructed.points) == 0:
        count = len(truth.points)
        return {
            "truthToReconstructionMeters": {"p50": math.inf, "p90": math.inf, "p95": math.inf},
            "reconstructionToTruthMeters": {"p50": math.inf, "p90": math.inf, "p95": math.inf},
            "coverageAtMeters": {"0.03": 0.0, "0.05": 0.0, "0.10": 0.0, "0.20": 0.0},
            "extraSurfaceRatioAtMeters": {"0.03": 1.0, "0.05": 1.0, "0.10": 1.0},
            "reconstructedPointCount": 0,
        }, np.zeros(count, dtype=bool)
    reconstructed = reconstructed.voxel_down_sample(0.01)
    truth_to_recon = np.asarray(truth.compute_point_cloud_distance(reconstructed), dtype=np.float64)
    reconstructed_points = np.asarray(reconstructed.points, dtype=np.float32)
    recon_to_truth = truth_scene.compute_distance(o3d.core.Tensor(reconstructed_points)).numpy().astype(np.float64)
    coverage = {f"{v:.2f}": float(np.mean(truth_to_recon <= v)) for v in (0.03, 0.05, 0.10, 0.20)}
    extras = {f"{v:.2f}": float(np.mean(recon_to_truth > v)) for v in (0.03, 0.05, 0.10)}
    structure_bands: dict[str, Any] = {}
    if structure_reference:
        truth_points = np.asarray(truth.points, dtype=np.float64)
        reconstructed_points64 = np.asarray(reconstructed.points, dtype=np.float64)
        for label, reference_points in structure_reference.items():
            truth_mask = distance_to_reference(truth_points, reference_points) <= max(0.01, structure_band_meters)
            reconstructed_mask = distance_to_reference(reconstructed_points64, reference_points) <= max(0.01, structure_band_meters)
            truth_count = int(np.sum(truth_mask))
            reconstructed_count = int(np.sum(reconstructed_mask))
            structure_bands[label] = {
                "truthSamples": truth_count,
                "reconstructedSamples": reconstructed_count,
                "coverageAt0.05m": float(np.mean(truth_to_recon[truth_mask] <= 0.05)) if truth_count else 0.0,
                "completenessP95m": percentile(truth_to_recon[truth_mask], 95) if truth_count else 0.0,
                "extraSurfaceRatioAt0.05m": (
                    float(np.mean(recon_to_truth[reconstructed_mask] > 0.05)) if reconstructed_count else 0.0
                ),
                "accuracyP95m": percentile(recon_to_truth[reconstructed_mask], 95) if reconstructed_count else 0.0,
            }
    return {
        "truthToReconstructionMeters": {
            "p50": percentile(truth_to_recon, 50),
            "p90": percentile(truth_to_recon, 90),
            "p95": percentile(truth_to_recon, 95),
        },
        "reconstructionToTruthMeters": {
            "p50": percentile(recon_to_truth, 50),
            "p90": percentile(recon_to_truth, 90),
            "p95": percentile(recon_to_truth, 95),
        },
        "coverageAtMeters": coverage,
        "extraSurfaceRatioAtMeters": extras,
        "structureBands": structure_bands,
        "structureBandMeters": structure_band_meters,
        "reconstructedPointCount": int(len(reconstructed.points)),
    }, truth_to_recon <= 0.05


def checkpoint(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
    truth: o3d.geometry.PointCloud,
    truth_scene: o3d.t.geometry.RaycastingScene,
    frame: int,
    previous_mask: np.ndarray | None,
    previous_chunks: dict[tuple[int, int, int], set[tuple[int, int, int]]] | None,
    chunk_size: float,
    structure_reference: dict[str, np.ndarray] | None = None,
    structure_band_meters: float = 0.08,
) -> tuple[dict[str, Any], np.ndarray, dict[tuple[int, int, int], set[tuple[int, int, int]]]]:
    started = time.perf_counter()
    cloud = volume.extract_point_cloud()
    extraction_ms = (time.perf_counter() - started) * 1000.0
    metrics, mask = cloud_metrics(
        truth, truth_scene, cloud, structure_reference, structure_band_meters
    )
    chunk_cloud = cloud.voxel_down_sample(0.03)
    chunk_points = np.asarray(chunk_cloud.points, dtype=np.float64)
    chunks: dict[tuple[int, int, int], set[tuple[int, int, int]]] = {}
    for point in chunk_points:
        chunk_key = tuple(np.floor(point / chunk_size).astype(np.int64).tolist())
        surface_key = tuple(np.floor(point / 0.03).astype(np.int64).tolist())
        chunks.setdefault(chunk_key, set()).add(surface_key)
    dirty_chunks = 0
    dirty_surface_voxels = 0
    if previous_chunks is None:
        dirty_chunks = len(chunks)
        dirty_surface_voxels = sum(len(values) for values in chunks.values())
    else:
        for key in set(previous_chunks) | set(chunks):
            before = previous_chunks.get(key, set())
            after = chunks.get(key, set())
            changed = len(before ^ after)
            threshold = max(3, int(math.ceil(0.05 * max(1, len(before | after)))))
            if changed >= threshold:
                dirty_chunks += 1
                dirty_surface_voxels += changed
    previous_count = int(np.sum(previous_mask)) if previous_mask is not None else 0
    lost = int(np.sum(previous_mask & ~mask)) if previous_mask is not None else 0
    gained = int(np.sum(~previous_mask & mask)) if previous_mask is not None else int(np.sum(mask))
    return {
        "frame": frame,
        "coverageAt0.05m": metrics["coverageAtMeters"]["0.05"],
        "extraSurfaceRatioAt0.05m": metrics["extraSurfaceRatioAtMeters"]["0.05"],
        "lostCoveredTruthSamples": lost,
        "lostFromPreviousRatio": lost / previous_count if previous_count else 0.0,
        "gainedTruthSamples": gained,
        "activePublishChunks": len(chunks),
        "dirtyPublishChunks": dirty_chunks,
        "dirtyPublishChunkRatio": dirty_chunks / len(chunks) if chunks else 0.0,
        "dirtySurfaceVoxels": dirty_surface_voxels,
        "extractionMs": extraction_ms,
        "structureBands": metrics.get("structureBands", {}),
    }, mask, chunks


def final_mesh_metrics(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
    truth: o3d.geometry.PointCloud,
    truth_scene: o3d.t.geometry.RaycastingScene,
    path: Path,
    structure_reference: dict[str, np.ndarray] | None = None,
    structure_band_meters: float = 0.08,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    mesh = volume.extract_triangle_mesh()
    extraction_ms = (time.perf_counter() - started) * 1000.0
    if len(mesh.vertices):
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False, compressed=False)
        sample_count = max(10000, min(100000, len(mesh.triangles) * 2))
        reconstructed = mesh.sample_points_uniformly(number_of_points=sample_count)
    else:
        reconstructed = volume.extract_point_cloud()
    metrics, _ = cloud_metrics(
        truth, truth_scene, reconstructed, structure_reference, structure_band_meters
    )
    metrics["vertices"] = int(len(mesh.vertices))
    metrics["triangles"] = int(len(mesh.triangles))
    if len(mesh.triangles):
        _, cluster_counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(cluster_counts, dtype=np.int64)
        metrics["connectedComponents"] = int(len(counts))
        metrics["significantComponents50Triangles"] = int(np.sum(counts >= 50))
        metrics["largestComponentTriangleRatio"] = float(np.max(counts) / np.sum(counts)) if len(counts) else 0.0
    return metrics, extraction_ms


def run_room(mesh_path: Path, model: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    o3d.utility.random.seed(args.seed)
    mesh = load_legacy_mesh(mesh_path)
    scene = build_scene(mesh)
    candidate_frames = args.frames if args.camera_clearance <= 0.0 else max(args.frames * 9, args.frames)
    camera_candidates, scan_metadata = build_stratified_slice_cameras(
        mesh, candidate_frames, "auto", 3, [0.35, 0.75, 1.25, 1.65, 2.10],
        [0.0, 60.0, 120.0, 180.0, 240.0, 300.0], 100.2439, args.width / args.height,
    )
    candidate_origins = np.asarray([camera["pose"]["position"] for camera in camera_candidates], dtype=np.float32)
    candidate_clearances = scene.compute_distance(o3d.core.Tensor(candidate_origins)).numpy().astype(np.float64)
    if args.camera_clearance > 0.0:
        eligible = [
            camera for camera, clearance in zip(camera_candidates, candidate_clearances)
            if float(clearance) >= args.camera_clearance
        ]
        if len(eligible) < args.frames:
            raise RuntimeError(
                f"{mesh_path.stem}: only {len(eligible)} camera candidates satisfy "
                f"{args.camera_clearance:.3f}m clearance; need {args.frames}"
            )
        if args.camera_selection == "visibility-balanced":
            cameras, selection_metadata = select_visibility_balanced_cameras(
                scene, eligible, args.frames, args.selection_width, args.selection_height,
                args.min_distance, args.max_distance, args.selection_voxel,
            )
        else:
            cameras = pick_evenly(eligible, args.frames)
            selection_metadata = {"selectionMode": "even"}
    else:
        eligible = camera_candidates
        cameras = camera_candidates
        selection_metadata = {"selectionMode": "even"}
    scan_metadata["cameraClearanceMeters"] = args.camera_clearance
    scan_metadata["cameraCandidatesEvaluated"] = len(camera_candidates)
    scan_metadata["cameraCandidatesEligible"] = len(eligible)
    scan_metadata["cameraCandidatesRejected"] = len(camera_candidates) - len(eligible)
    scan_metadata["candidateClearanceP10Meters"] = percentile(candidate_clearances, 10)
    scan_metadata["candidateClearanceP50Meters"] = percentile(candidate_clearances, 50)
    scan_metadata["generatedFrames"] = len(cameras)
    scan_metadata.update(selection_metadata)
    truth = mesh.sample_points_uniformly(number_of_points=args.truth_samples, use_triangle_normal=True)
    structure_reference = build_mesh_structure_reference(mesh, args.structure_crease_degrees)
    ideal_volume = new_volume(args.voxel, args.sdf_trunc)
    quest_ungated_volume = new_volume(args.voxel, args.sdf_trunc)
    quest_volume = new_volume(args.voxel, args.sdf_trunc)
    color = o3d.geometry.Image(np.zeros((args.height, args.width, 3), dtype=np.uint8))
    ideal_times: list[float] = []
    quest_ungated_times: list[float] = []
    quest_times: list[float] = []
    checkpoints: dict[str, list[dict[str, Any]]] = {"ideal": [], "questUngated": [], "quest": []}
    previous_masks: dict[str, np.ndarray | None] = {"ideal": None, "questUngated": None, "quest": None}
    previous_chunks: dict[str, dict[tuple[int, int, int], set[tuple[int, int, int]]] | None] = {
        "ideal": None, "questUngated": None, "quest": None
    }
    valid_pixels = 0
    quest_ungated_pixels = 0
    quest_pixels = 0
    structure_totals = {
        label: {"candidates": 0, "mixed": 0, "ambiguous": 0, "held": 0}
        for label in STRUCTURE_LABELS.values()
    }
    active_input_chunks: set[tuple[int, int, int]] = set()
    per_frame_input_chunks: list[dict[str, Any]] = []

    for frame_index, camera in enumerate(cameras, start=1):
        (
            ideal_depth,
            quest_ungated_depth,
            quest_depth,
            valid,
            accepted_ungated,
            accepted,
            frame_structure_stats,
        ) = make_depth_pair(
            scene,
            camera,
            args.width,
            args.height,
            args.min_distance,
            args.max_distance,
            model,
            args.structured_edge_degradation,
            args.structure_depth_jump,
            args.structure_crease_degrees,
            args.ownership_strong_jump_scale,
            args.ownership_min_mix_alpha,
            args.ownership_min_endpoint_residual,
        )
        valid_pixels += valid
        quest_ungated_pixels += accepted_ungated
        quest_pixels += accepted
        for label, values in frame_structure_stats.items():
            for key, value in values.items():
                structure_totals[label][key] += int(value)
        frame_chunks = observed_surface_chunks(camera, quest_depth, args.width, args.height, args.publish_chunk)
        new_chunks = frame_chunks - active_input_chunks
        active_input_chunks.update(frame_chunks)
        per_frame_input_chunks.append({
            "frame": frame_index,
            "dirtySurfaceChunks": len(frame_chunks),
            "newSurfaceChunks": len(new_chunks),
            "activeSurfaceChunks": len(active_input_chunks),
            "dirtyToActiveRatio": len(frame_chunks) / len(active_input_chunks) if active_input_chunks else 0.0,
        })
        intrinsic, extrinsic, _ = camera_calibration(camera, args.width, args.height)
        ideal_times.append(integrate_depth(ideal_volume, ideal_depth, intrinsic, extrinsic, args.max_distance, color))
        quest_ungated_times.append(
            integrate_depth(quest_ungated_volume, quest_ungated_depth, intrinsic, extrinsic, args.max_distance, color)
        )
        quest_times.append(integrate_depth(quest_volume, quest_depth, intrinsic, extrinsic, args.max_distance, color))
        if frame_index % args.checkpoint_every == 0 or frame_index == len(cameras):
            for name, volume in (
                ("ideal", ideal_volume),
                ("questUngated", quest_ungated_volume),
                ("quest", quest_volume),
            ):
                row, mask, chunks = checkpoint(
                    volume,
                    truth,
                    scene,
                    frame_index,
                    previous_masks[name],
                    previous_chunks[name],
                    args.publish_chunk,
                    structure_reference,
                    args.structure_band_meters,
                )
                checkpoints[name].append(row)
                previous_masks[name] = mask
                previous_chunks[name] = chunks

    room_out = args.out / mesh_path.stem
    room_out.mkdir(parents=True, exist_ok=True)
    ideal_final, ideal_extract = final_mesh_metrics(
        ideal_volume, truth, scene, room_out / "ideal_persistent_tsdf.ply",
        structure_reference, args.structure_band_meters,
    )
    quest_ungated_final, quest_ungated_extract = final_mesh_metrics(
        quest_ungated_volume, truth, scene, room_out / "quest_ungated_persistent_tsdf.ply",
        structure_reference, args.structure_band_meters,
    )
    quest_final, quest_extract = final_mesh_metrics(
        quest_volume, truth, scene, room_out / "quest_persistent_tsdf.ply",
        structure_reference, args.structure_band_meters,
    )

    def time_stats(values: list[float]) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)), "p50": percentile(values, 50), "p90": percentile(values, 90),
            "p95": percentile(values, 95), "max": float(np.max(values)),
        }

    report = {
        "schema": "ScanCoverReplicaPersistentTSDFValidation/v1",
        "room": mesh_path.stem,
        "truthMesh": str(mesh_path),
        "degradationModel": str(args.degradation_model),
        "protocol": {
            "frames": len(cameras), "rayGrid": [args.width, args.height], "voxelMeters": args.voxel,
            "sdfTruncMeters": args.sdf_trunc, "workingRangeMeters": [args.min_distance, args.max_distance],
            "truthSamples": args.truth_samples, "checkpointEveryFrames": args.checkpoint_every,
            "cameraClearanceMeters": args.camera_clearance,
            "cameraSelection": args.camera_selection,
            "publishChunkMeters": args.publish_chunk,
            "structuredEdgeDegradation": bool(args.structured_edge_degradation),
            "structureDepthJumpMeters": args.structure_depth_jump,
            "structureCreaseDegrees": args.structure_crease_degrees,
            "structureBandMeters": args.structure_band_meters,
            "ownershipStrongJumpScale": args.ownership_strong_jump_scale,
            "ownershipMinMixAlpha": args.ownership_min_mix_alpha,
            "ownershipMinEndpointResidualMeters": args.ownership_min_endpoint_residual,
            "scanMetadata": scan_metadata,
        },
        "input": {
            "idealValidPixels": valid_pixels,
            "questUngatedAcceptedPixels": quest_ungated_pixels,
            "questAcceptedPixels": quest_pixels,
            "questUngatedAcceptedRatioOfIdeal": quest_ungated_pixels / valid_pixels if valid_pixels else 0.0,
            "questAcceptedRatioOfIdeal": quest_pixels / valid_pixels if valid_pixels else 0.0,
            "ownershipHeldPixels": quest_ungated_pixels - quest_pixels,
            "ownershipHeldRatioOfUngated": (
                (quest_ungated_pixels - quest_pixels) / quest_ungated_pixels if quest_ungated_pixels else 0.0
            ),
            "structure": structure_totals,
            "truthStructureReferencePoints": {
                label: int(len(points)) for label, points in structure_reference.items()
            },
        },
        "incrementalPublish": {
            "chunkMeters": args.publish_chunk,
            "activeSurfaceChunks": len(active_input_chunks),
            "perFrame": per_frame_input_chunks,
            "dirtyChunksPerFrame": {
                "mean": float(np.mean([row["dirtySurfaceChunks"] for row in per_frame_input_chunks])),
                "p50": percentile([row["dirtySurfaceChunks"] for row in per_frame_input_chunks], 50),
                "p90": percentile([row["dirtySurfaceChunks"] for row in per_frame_input_chunks], 90),
                "max": max((row["dirtySurfaceChunks"] for row in per_frame_input_chunks), default=0),
            },
            "dirtyToActiveRatio": {
                "mean": float(np.mean([row["dirtyToActiveRatio"] for row in per_frame_input_chunks])),
                "p50": percentile([row["dirtyToActiveRatio"] for row in per_frame_input_chunks], 50),
                "p90": percentile([row["dirtyToActiveRatio"] for row in per_frame_input_chunks], 90),
                "late30Mean": float(np.mean([row["dirtyToActiveRatio"] for row in per_frame_input_chunks[-30:]])),
            },
        },
        "ideal": {"final": ideal_final, "checkpoints": checkpoints["ideal"], "integrationMs": time_stats(ideal_times), "finalExtractionMs": ideal_extract},
        "questUngated": {
            "final": quest_ungated_final,
            "checkpoints": checkpoints["questUngated"],
            "integrationMs": time_stats(quest_ungated_times),
            "finalExtractionMs": quest_ungated_extract,
        },
        "quest": {"final": quest_final, "checkpoints": checkpoints["quest"], "integrationMs": time_stats(quest_times), "finalExtractionMs": quest_extract},
    }
    report["delta"] = {
        "coverageAt0.05m": quest_final["coverageAtMeters"]["0.05"] - ideal_final["coverageAtMeters"]["0.05"],
        "coverageAt0.10m": quest_final["coverageAtMeters"]["0.10"] - ideal_final["coverageAtMeters"]["0.10"],
        "extraSurfaceRatioAt0.05m": quest_final["extraSurfaceRatioAtMeters"]["0.05"] - ideal_final["extraSurfaceRatioAtMeters"]["0.05"],
        "accuracyP95m": quest_final["reconstructionToTruthMeters"]["p95"] - ideal_final["reconstructionToTruthMeters"]["p95"],
        "integrationP95ms": report["quest"]["integrationMs"]["p95"] - report["ideal"]["integrationMs"]["p95"],
    }
    report["ownershipGuardDelta"] = {
        "coverageAt0.05m": (
            quest_final["coverageAtMeters"]["0.05"] - quest_ungated_final["coverageAtMeters"]["0.05"]
        ),
        "coverageAt0.10m": (
            quest_final["coverageAtMeters"]["0.10"] - quest_ungated_final["coverageAtMeters"]["0.10"]
        ),
        "extraSurfaceRatioAt0.05m": (
            quest_final["extraSurfaceRatioAtMeters"]["0.05"] -
            quest_ungated_final["extraSurfaceRatioAtMeters"]["0.05"]
        ),
        "accuracyP95m": (
            quest_final["reconstructionToTruthMeters"]["p95"] -
            quest_ungated_final["reconstructionToTruthMeters"]["p95"]
        ),
        "structureBands": {
            label: {
                "coverageAt0.05m": (
                    quest_final.get("structureBands", {}).get(label, {}).get("coverageAt0.05m", 0.0) -
                    quest_ungated_final.get("structureBands", {}).get(label, {}).get("coverageAt0.05m", 0.0)
                ),
                "extraSurfaceRatioAt0.05m": (
                    quest_final.get("structureBands", {}).get(label, {}).get("extraSurfaceRatioAt0.05m", 0.0) -
                    quest_ungated_final.get("structureBands", {}).get(label, {}).get("extraSurfaceRatioAt0.05m", 0.0)
                ),
                "accuracyP95m": (
                    quest_final.get("structureBands", {}).get(label, {}).get("accuracyP95m", 0.0) -
                    quest_ungated_final.get("structureBands", {}).get(label, {}).get("accuracyP95m", 0.0)
                ),
            }
            for label in structure_reference
        },
    }
    (room_out / "persistent_tsdf_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def write_summary(reports: list[dict[str, Any]], out: Path) -> None:
    rows = []
    for report in reports:
        ideal = report["ideal"]["final"]
        quest_ungated = report["questUngated"]["final"]
        quest = report["quest"]["final"]
        quest_checkpoints = report["quest"]["checkpoints"]
        rows.append({
            "room": report["room"],
            "questAcceptedRatioOfIdeal": report["input"]["questAcceptedRatioOfIdeal"],
            "ownershipHeldRatioOfUngated": report["input"]["ownershipHeldRatioOfUngated"],
            "idealCoverage5cm": ideal["coverageAtMeters"]["0.05"],
            "questUngatedCoverage5cm": quest_ungated["coverageAtMeters"]["0.05"],
            "questCoverage5cm": quest["coverageAtMeters"]["0.05"],
            "guardCoverageDelta5cm": report["ownershipGuardDelta"]["coverageAt0.05m"],
            "coverageDelta5cm": report["delta"]["coverageAt0.05m"],
            "idealExtra5cm": ideal["extraSurfaceRatioAtMeters"]["0.05"],
            "questExtra5cm": quest["extraSurfaceRatioAtMeters"]["0.05"],
            "questUngatedExtra5cm": quest_ungated["extraSurfaceRatioAtMeters"]["0.05"],
            "guardExtraDelta5cm": report["ownershipGuardDelta"]["extraSurfaceRatioAt0.05m"],
            "extraDelta5cm": report["delta"]["extraSurfaceRatioAt0.05m"],
            "questAccuracyP95m": quest["reconstructionToTruthMeters"]["p95"],
            "questCompletenessP95m": quest["truthToReconstructionMeters"]["p95"],
            "questIntegrationP95ms": report["quest"]["integrationMs"]["p95"],
            "questIntegrationMaxMs": report["quest"]["integrationMs"]["max"],
            "maxCheckpointRegression": max((c["lostFromPreviousRatio"] for c in quest_checkpoints[1:]), default=0.0),
            "finalDirtyPublishChunkRatio": quest_checkpoints[-1]["dirtyPublishChunkRatio"] if quest_checkpoints else 0.0,
            "perFrameDirtyChunkRatioP50": report["incrementalPublish"]["dirtyToActiveRatio"]["p50"],
            "perFrameDirtyChunkRatioP90": report["incrementalPublish"]["dirtyToActiveRatio"]["p90"],
            "late30DirtyChunkRatioMean": report["incrementalPublish"]["dirtyToActiveRatio"]["late30Mean"],
            "questTriangles": quest["triangles"],
            "questSignificantComponents": quest.get("significantComponents50Triangles", 0),
        })
    with (out / "persistent_tsdf_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    aggregate = {
        "rooms": len(rows),
        "meanQuestCoverage5cm": float(np.mean([r["questCoverage5cm"] for r in rows])),
        "meanOwnershipHeldRatio": float(np.mean([r["ownershipHeldRatioOfUngated"] for r in rows])),
        "meanGuardCoverageDelta5cm": float(np.mean([r["guardCoverageDelta5cm"] for r in rows])),
        "minGuardCoverageDelta5cm": min(r["guardCoverageDelta5cm"] for r in rows),
        "meanGuardExtraDelta5cm": float(np.mean([r["guardExtraDelta5cm"] for r in rows])),
        "maxGuardExtraDelta5cm": max(r["guardExtraDelta5cm"] for r in rows),
        "minQuestCoverage5cm": min(r["questCoverage5cm"] for r in rows),
        "meanCoverageDelta5cm": float(np.mean([r["coverageDelta5cm"] for r in rows])),
        "maxExtraDelta5cm": max(r["extraDelta5cm"] for r in rows),
        "maxQuestAccuracyP95m": max(r["questAccuracyP95m"] for r in rows),
        "maxQuestCompletenessP95m": max(r["questCompletenessP95m"] for r in rows),
        "maxCheckpointRegression": max(r["maxCheckpointRegression"] for r in rows),
        "meanFinalDirtyPublishChunkRatio": float(np.mean([r["finalDirtyPublishChunkRatio"] for r in rows])),
        "maxFinalDirtyPublishChunkRatio": max(r["finalDirtyPublishChunkRatio"] for r in rows),
        "meanLate30DirtyChunkRatio": float(np.mean([r["late30DirtyChunkRatioMean"] for r in rows])),
        "maxLate30DirtyChunkRatio": max(r["late30DirtyChunkRatioMean"] for r in rows),
        "maxQuestIntegrationP95ms": max(r["questIntegrationP95ms"] for r in rows),
    }
    (out / "persistent_tsdf_summary.json").write_text(json.dumps({"rows": rows, "aggregates": aggregate}, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Replica Persistent TSDF Validation", "", *[f"- {k}: {v}" for k, v in aggregate.items()], "", "| Room | Ungated cov | Guarded cov | Guard cov delta | Ungated extra | Guarded extra | Guard extra delta | Held | Accuracy p95 | Completeness p95 | Regress |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['room']} | {r['questUngatedCoverage5cm']:.4f} | {r['questCoverage5cm']:.4f} | {r['guardCoverageDelta5cm']:.4f} | {r['questUngatedExtra5cm']:.4f} | {r['questExtra5cm']:.4f} | {r['guardExtraDelta5cm']:.4f} | {r['ownershipHeldRatioOfUngated']:.4f} | {r['questAccuracyP95m']:.4f} | {r['questCompletenessP95m']:.4f} | {r['maxCheckpointRegression']:.4f} |")
    (out / "persistent_tsdf_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows, "aggregates": aggregate}, indent=2, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    if not args.degradation_model.exists():
        raise FileNotFoundError(args.degradation_model)
    for mesh in args.meshes:
        if not mesh.exists(): raise FileNotFoundError(mesh)
    args.out.mkdir(parents=True, exist_ok=True)
    model = json.loads(args.degradation_model.read_text(encoding="utf-8-sig"))
    reports = []
    for index, mesh in enumerate(args.meshes, start=1):
        print(f"[persistent-tsdf {index}/{len(args.meshes)}] {mesh.stem}", flush=True)
        reports.append(run_room(mesh, model, args))
    write_summary(reports, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
