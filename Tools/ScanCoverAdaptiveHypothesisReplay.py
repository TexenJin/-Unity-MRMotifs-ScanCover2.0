#!/usr/bin/env python3
"""Blind offline comparison of single- and multi-hypothesis surface evidence.

This is intentionally an evidence-layer test, not a claim of byte-equivalent
replay of VolumeIntegration.compute.  It streams paired-depth frames directly
from a connected Quest, builds models on the first 70 percent of time, freezes
them, and scores later observations without updating or environment labels.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ScanCoverDepthPairDeviceAudit import (
    DEFAULT_ADB,
    annotate_motion,
    fetch_pair_batch,
    latest_session,
    load_metadata,
    recorded_eye,
)


@dataclass
class Candidate:
    owner_key: tuple[int, int, int]
    center: np.ndarray
    normal: np.ndarray
    sigma2: float
    support: float
    observations: int
    first_view: np.ndarray
    mean_view: np.ndarray
    max_view_angle_deg: float
    last_frame: int
    stable: bool = False
    test_hits: int = 0
    corrected_test_hits: int = 0


@dataclass
class ScoreLedger:
    observations: int = 0
    covered: int = 0
    standardized_inliers: int = 0
    residuals: list[float] = field(default_factory=list)
    normal_angles: list[float] = field(default_factory=list)
    visited_cells: set[tuple[int, int, int]] = field(default_factory=set)


class SurfaceEvidenceModel:
    def __init__(
        self,
        name: str,
        cell_size: float,
        capacity: int,
        robust: bool,
        spawn_threshold: float,
        sigma_floor: float,
        sigma_ceiling: float,
    ) -> None:
        self.name = name
        self.cell_size = cell_size
        self.capacity = capacity
        self.robust = robust
        self.spawn_threshold = spawn_threshold
        self.sigma_floor = sigma_floor
        self.sigma_ceiling = sigma_ceiling
        self.cells: dict[tuple[int, int, int], list[Candidate]] = {}
        self.births = 0
        self.updates = 0
        self.abstentions = 0
        self.score_ledger = ScoreLedger()
        self.corrected_score_ledger = ScoreLedger()

    def key(self, point: np.ndarray) -> tuple[int, int, int]:
        value = np.floor(point / self.cell_size).astype(np.int32)
        return int(value[0]), int(value[1]), int(value[2])

    def _new_candidate(
        self,
        owner_key: tuple[int, int, int],
        point: np.ndarray,
        normal: np.ndarray,
        view: np.ndarray,
        observation_sigma: float,
        quality: float,
        frame: int,
    ) -> Candidate:
        self.births += 1
        return Candidate(
            owner_key=owner_key,
            center=point.astype(np.float64).copy(),
            normal=normal.astype(np.float64).copy(),
            sigma2=max(observation_sigma * observation_sigma, self.sigma_floor**2),
            support=float(quality),
            observations=1,
            first_view=view.astype(np.float64).copy(),
            mean_view=view.astype(np.float64).copy(),
            max_view_angle_deg=0.0,
            last_frame=frame,
        )

    @staticmethod
    def _neighbor_keys(key: tuple[int, int, int]):
        x, y, z = key
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    yield x + dx, y + dy, z + dz

    def _nearby_candidates(
        self,
        key: tuple[int, int, int],
        stable_only: bool = False,
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        for nearby_key in self._neighbor_keys(key):
            for candidate in self.cells.get(nearby_key, []):
                if not stable_only or candidate.stable:
                    candidates.append(candidate)
        return candidates

    @staticmethod
    def _aligned_normal(candidate: Candidate, normal: np.ndarray) -> tuple[np.ndarray, float]:
        dot = float(np.clip(np.dot(candidate.normal, normal), -1.0, 1.0))
        aligned = normal if dot >= 0.0 else -normal
        angle = math.degrees(math.acos(float(np.clip(abs(dot), 0.0, 1.0))))
        return aligned, angle

    def _match(
        self,
        candidate: Candidate,
        point: np.ndarray,
        normal: np.ndarray,
        observation_sigma: float,
    ) -> tuple[float, float, float]:
        _, angle_deg = self._aligned_normal(candidate, normal)
        residual = abs(float(np.dot(point - candidate.center, candidate.normal)))
        combined_sigma = math.sqrt(
            max(candidate.sigma2, self.sigma_floor**2) + observation_sigma**2
        )
        depth_z = residual / max(combined_sigma, 1e-6)
        normal_z = angle_deg / 25.0
        return math.sqrt(depth_z * depth_z + normal_z * normal_z), residual, angle_deg

    def observe(
        self,
        point: np.ndarray,
        normal: np.ndarray,
        view: np.ndarray,
        observation_sigma: float,
        quality: float,
        frame: int,
    ) -> None:
        key = self.key(point)
        local_candidates = self.cells.setdefault(key, [])
        candidates = self._nearby_candidates(key)
        if not candidates:
            local_candidates.append(
                self._new_candidate(
                    key, point, normal, view, observation_sigma, quality, frame
                )
            )
            return

        matches = [self._match(item, point, normal, observation_sigma) for item in candidates]
        best_index = int(np.argmin([item[0] for item in matches]))
        score, residual, _ = matches[best_index]

        if self.capacity > 1 and score > self.spawn_threshold:
            if len(local_candidates) < self.capacity:
                local_candidates.append(
                    self._new_candidate(
                        key, point, normal, view, observation_sigma, quality, frame
                    )
                )
            else:
                self.abstentions += 1
            return

        candidate = candidates[best_index]
        robust_weight = 1.0
        if self.robust:
            scale = max(self.spawn_threshold, 1.0)
            robust_weight = 1.0 / (1.0 + (score / scale) ** 2)
        effective = max(float(quality) * robust_weight, 1e-4)
        alpha = min(0.20, effective / max(candidate.support + effective, 1e-5))
        aligned_normal, _ = self._aligned_normal(candidate, normal)
        candidate.center = candidate.center * (1.0 - alpha) + point * alpha
        mixed_normal = candidate.normal * (1.0 - alpha) + aligned_normal * alpha
        length = float(np.linalg.norm(mixed_normal))
        if length > 1e-8:
            candidate.normal = mixed_normal / length
        innovation2 = residual * residual
        candidate.sigma2 = float(
            np.clip(
                candidate.sigma2 * (1.0 - alpha) + innovation2 * alpha,
                self.sigma_floor**2,
                self.sigma_ceiling**2,
            )
        )
        view_dot = float(np.clip(np.dot(candidate.first_view, view), -1.0, 1.0))
        candidate.max_view_angle_deg = max(
            candidate.max_view_angle_deg,
            math.degrees(math.acos(view_dot)),
        )
        mean_view = candidate.mean_view * (1.0 - alpha) + view * alpha
        mean_length = float(np.linalg.norm(mean_view))
        if mean_length > 1e-8:
            candidate.mean_view = mean_view / mean_length
        candidate.support += effective
        candidate.observations += 1
        candidate.last_frame = frame
        self.updates += 1

    def freeze_training(self, min_support: float, max_sigma: float, min_view_angle: float) -> None:
        for candidates in self.cells.values():
            for candidate in candidates:
                mature = candidate.support >= min_support
                diverse = candidate.max_view_angle_deg >= min_view_angle
                concentrated = math.sqrt(candidate.sigma2) <= max_sigma
                candidate.stable = mature and diverse and (concentrated or not self.robust)

    def score(
        self,
        point: np.ndarray,
        normal: np.ndarray,
        observation_sigma: float,
        corrected: bool = False,
    ) -> None:
        ledger = self.corrected_score_ledger if corrected else self.score_ledger
        ledger.observations += 1
        key = self.key(point)
        ledger.visited_cells.add(key)
        candidates = self._nearby_candidates(key, stable_only=True)
        if not candidates:
            return
        matches = [self._match(item, point, normal, observation_sigma) for item in candidates]
        best_index = int(np.argmin([item[0] for item in matches]))
        standardized, residual, normal_angle = matches[best_index]
        candidate = candidates[best_index]
        if corrected:
            candidate.corrected_test_hits += 1
        else:
            candidate.test_hits += 1
        ledger.covered += 1
        ledger.standardized_inliers += int(standardized <= 3.0)
        ledger.residuals.append(residual)
        ledger.normal_angles.append(normal_angle)

    @staticmethod
    def _percentile(values: list[float], q: float, scale: float = 1.0) -> float | None:
        if not values:
            return None
        return float(np.percentile(np.asarray(values, dtype=np.float64), q) * scale)

    def report(self) -> dict:
        all_candidates = [item for values in self.cells.values() for item in values]
        stable = [item for item in all_candidates if item.stable]
        visited_stable = [
            item
            for item in stable
            if any(
                key in self.score_ledger.visited_cells
                for key in self._neighbor_keys(item.owner_key)
            )
        ]
        unconfirmed = [item for item in visited_stable if item.test_hits == 0]
        multi_cells = sum(
            1 for values in self.cells.values() if sum(item.stable for item in values) >= 2
        )
        sigmas = [math.sqrt(item.sigma2) for item in stable]
        ledger = self.score_ledger
        corrected = self.corrected_score_ledger
        report = {
            "name": self.name,
            "capacityPerCell": self.capacity,
            "matchingNeighborhoodCells": 27,
            "robust": self.robust,
            "occupiedCells": sum(1 for values in self.cells.values() if values),
            "candidateBirths": self.births,
            "candidateUpdates": self.updates,
            "capacityAbstentions": self.abstentions,
            "allCandidates": len(all_candidates),
            "stableCandidates": len(stable),
            "stableMultiHypothesisCells": multi_cells,
            "stableSigmaMmP50": self._percentile(sigmas, 50, 1000.0),
            "stableSigmaMmP95": self._percentile(sigmas, 95, 1000.0),
            "heldoutObservations": ledger.observations,
            "heldoutCellCoveragePct": ledger.covered * 100.0 / max(ledger.observations, 1),
            "heldoutStandardizedInlierPct": ledger.standardized_inliers * 100.0 / max(ledger.covered, 1),
            "heldoutResidualMmP50": self._percentile(ledger.residuals, 50, 1000.0),
            "heldoutResidualMmP95": self._percentile(ledger.residuals, 95, 1000.0),
            "heldoutNormalDegP50": self._percentile(ledger.normal_angles, 50),
            "heldoutNormalDegP95": self._percentile(ledger.normal_angles, 95),
            "revisitedStableCandidates": len(visited_stable),
            "revisitedStableWithoutHeldoutHitPct": len(unconfirmed) * 100.0 / max(len(visited_stable), 1),
        }
        report.update(
            {
                "poseCorrectedHeldoutObservations": corrected.observations,
                "poseCorrectedCoveragePct": corrected.covered
                * 100.0
                / max(corrected.observations, 1),
                "poseCorrectedStandardizedInlierPct": corrected.standardized_inliers
                * 100.0
                / max(corrected.covered, 1),
                "poseCorrectedResidualMmP50": self._percentile(
                    corrected.residuals, 50, 1000.0
                ),
                "poseCorrectedResidualMmP95": self._percentile(
                    corrected.residuals, 95, 1000.0
                ),
                "poseCorrectedNormalDegP50": self._percentile(
                    corrected.normal_angles, 50
                ),
                "poseCorrectedNormalDegP95": self._percentile(
                    corrected.normal_angles, 95
                ),
            }
        )
        return report


def rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-10:
        return np.eye(3, dtype=np.float64)
    axis = vector / angle
    x, y, z = axis
    skew = np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def transform_geometry(
    points: np.ndarray,
    normals: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return points @ rotation.T + translation[None, :], normals @ rotation.T


def alignment_correspondences(
    reference: SurfaceEvidenceModel,
    points: np.ndarray,
    normals: np.ndarray,
    observation_sigmas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_points: list[np.ndarray] = []
    target_points: list[np.ndarray] = []
    target_normals: list[np.ndarray] = []
    combined_sigmas: list[float] = []
    normal_angles: list[float] = []
    for point, normal, observation_sigma in zip(points, normals, observation_sigmas):
        key = reference.key(point)
        candidates = reference._nearby_candidates(key, stable_only=True)
        if not candidates:
            continue
        matches = [
            reference._match(candidate, point, normal, float(observation_sigma))
            for candidate in candidates
        ]
        best_index = int(np.argmin([match[0] for match in matches]))
        _score, residual, normal_angle = matches[best_index]
        candidate = candidates[best_index]
        combined_sigma = math.sqrt(
            max(candidate.sigma2, reference.sigma_floor**2)
            + float(observation_sigma) ** 2
        )
        # Broad normalized association gate: it permits a coherent pose offset
        # to be solved, but rejects unrelated surfaces before least squares.
        if normal_angle > 45.0 or residual > 5.0 * combined_sigma:
            continue
        source_points.append(point)
        target_points.append(candidate.center)
        target_normals.append(candidate.normal)
        combined_sigmas.append(combined_sigma)
        normal_angles.append(normal_angle)
    if not source_points:
        empty3 = np.empty((0, 3), dtype=np.float64)
        empty1 = np.empty((0,), dtype=np.float64)
        return empty3, empty3.copy(), empty3.copy(), empty1, empty1.copy()
    return (
        np.asarray(source_points, dtype=np.float64),
        np.asarray(target_points, dtype=np.float64),
        np.asarray(target_normals, dtype=np.float64),
        np.asarray(combined_sigmas, dtype=np.float64),
        np.asarray(normal_angles, dtype=np.float64),
    )


def estimate_frame_correction(
    reference: SurfaceEvidenceModel,
    points: np.ndarray,
    normals: np.ndarray,
    observation_sigmas: np.ndarray,
    iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    effective_rank = 0
    used = 0
    raw_median = None

    for iteration in range(iterations):
        transformed_points, transformed_normals = transform_geometry(
            points, normals, rotation, translation
        )
        source, target, target_normal, sigma, normal_angle = alignment_correspondences(
            reference, transformed_points, transformed_normals, observation_sigmas
        )
        used = len(source)
        if used < 64:
            break
        residual = np.einsum("ij,ij->i", source - target, target_normal)
        if iteration == 0:
            raw_median = float(np.median(np.abs(residual)))
        standardized = np.abs(residual) / np.maximum(sigma, 1e-6)
        robust = 1.0 / (1.0 + (standardized / 3.0) ** 2)
        facing = np.clip(np.cos(np.radians(normal_angle)), 0.1, 1.0)
        weights = robust * facing / np.maximum(sigma * sigma, 1e-8)
        jacobian = np.concatenate(
            (np.cross(source, target_normal), target_normal), axis=1
        )
        sqrt_weight = np.sqrt(weights)[:, None]
        weighted_jacobian = jacobian * sqrt_weight
        weighted_rhs = -residual * sqrt_weight[:, 0]
        u, singular, vt = np.linalg.svd(weighted_jacobian, full_matrices=False)
        if singular.size == 0 or singular[0] <= 1e-9:
            break
        observable = singular > singular[0] * 1e-4
        effective_rank = int(np.count_nonzero(observable))
        if effective_rank < 3:
            break
        inverse = np.zeros_like(singular)
        inverse[observable] = 1.0 / singular[observable]
        delta = vt.T @ (inverse * (u.T @ weighted_rhs))

        rotation_step = delta[:3]
        translation_step = delta[3:]
        rotation_limit = math.radians(1.5)
        rotation_length = float(np.linalg.norm(rotation_step))
        if rotation_length > rotation_limit:
            rotation_step *= rotation_limit / rotation_length
        translation_length = float(np.linalg.norm(translation_step))
        if translation_length > 0.03:
            translation_step *= 0.03 / translation_length

        delta_rotation = rotation_from_vector(rotation_step)
        rotation = delta_rotation @ rotation
        translation = delta_rotation @ translation + translation_step
        if float(np.linalg.norm(delta)) < 1e-6:
            break

    transformed_points, transformed_normals = transform_geometry(
        points, normals, rotation, translation
    )
    source, target, target_normal, _sigma, _angle = alignment_correspondences(
        reference, transformed_points, transformed_normals, observation_sigmas
    )
    corrected_median = None
    if len(source) > 0:
        corrected_median = float(
            np.median(np.abs(np.einsum("ij,ij->i", source - target, target_normal)))
        )

    translation_metres = float(np.linalg.norm(translation))
    rotation_degrees = rotation_angle_degrees(rotation)
    improved = (
        raw_median is not None
        and corrected_median is not None
        and corrected_median < raw_median * 0.995
    )
    safe = translation_metres <= 0.10 and rotation_degrees <= 5.0
    applied = used >= 64 and effective_rank >= 3 and improved and safe
    if not applied:
        rotation = np.eye(3, dtype=np.float64)
        translation = np.zeros(3, dtype=np.float64)

    return rotation, translation, {
        "applied": applied,
        "correspondences": used,
        "effectiveRank": effective_rank,
        "translationMm": translation_metres * 1000.0,
        "rotationDeg": rotation_degrees,
        "alignmentResidualMmBefore": None
        if raw_median is None
        else raw_median * 1000.0,
        "alignmentResidualMmAfter": None
        if corrected_median is None
        else corrected_median * 1000.0,
    }


def normalize_rows(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(lengths) & (lengths > 1e-8)
    output = np.zeros_like(vectors, dtype=np.float64)
    output[valid] = vectors[valid] / lengths[valid, None]
    return output, valid


def frame_observations(
    record: dict,
    depth: np.ndarray,
    stride: int,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width = int(record["width"])
    height = int(record["height"])
    image = depth.reshape(height, width)
    ys = np.arange(0, height - 2, stride, dtype=np.int32)
    xs = np.arange(0, width - 2, stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    sample_x = np.concatenate((grid_x.ravel(), (grid_x + 2).ravel(), grid_x.ravel()))
    sample_y = np.concatenate((grid_y.ravel(), grid_y.ravel(), (grid_y + 2).ravel()))
    sample_d = image[sample_y, sample_x].astype(np.float64)
    valid_depth = np.isfinite(sample_d) & (sample_d > 0.0) & (sample_d < 1.0)

    u = sample_x.astype(np.float64) / width
    v = sample_y.astype(np.float64) / height
    hcs = np.stack((u * 2.0 - 1.0, v * 2.0 - 1.0, sample_d * 2.0 - 1.0, np.ones_like(u)))
    eye_index = recorded_eye(record)
    projection_inverse = np.asarray(record["projectionInverse"][eye_index], dtype=np.float64).reshape(4, 4)
    view_inverse = np.asarray(record["viewInverse"][eye_index], dtype=np.float64).reshape(4, 4)
    world_h = view_inverse @ (projection_inverse @ hcs)
    world = (world_h[:3] / world_h[3]).T
    count = grid_x.size
    center = world[:count]
    horizontal = world[count : count * 2]
    vertical = world[count * 2 :]
    valid = valid_depth[:count] & valid_depth[count : count * 2] & valid_depth[count * 2 :]

    normals, normal_valid = normalize_rows(-np.cross(horizontal - center, vertical - center))
    eye = np.asarray(record["trackingPoses"][eye_index]["position"], dtype=np.float64)
    views, view_valid = normalize_rows(eye[None, :] - center)
    ranges = np.linalg.norm(center - eye[None, :], axis=1)
    valid &= normal_valid & view_valid & np.isfinite(ranges) & (ranges >= 0.15) & (ranges <= max_depth)

    incidence = np.abs(np.einsum("ij,ij->i", normals, views))
    distance_quality = np.clip(1.0 - ranges / max_depth, 0.0, 1.0)
    angular_speed = float(record.get("_angularSpeedDeg", 0.0))
    linear_speed = float(record.get("_linearSpeedMps", 0.0))
    motion_level = max(angular_speed / 30.0, linear_speed / 0.30)
    motion_quality = 1.0 - 0.85 * min(max(motion_level, 0.0), 1.0)
    quality = np.clip(distance_quality * incidence * motion_quality, 0.05, 1.0)
    sigma = np.clip(0.004 / np.sqrt(quality), 0.004, 0.025)
    return center[valid], normals[valid], views[valid], sigma[valid], quality[valid]


def add_motion_speeds(records: list[dict]) -> None:
    previous = None
    for record in records:
        pose = record["trackingPoses"][recorded_eye(record)]
        position = np.asarray(pose["position"], dtype=np.float64)
        if previous is None:
            dt = 0.0
        else:
            dt = max(float(record["unscaledTime"]) - previous[0], 1e-4)
        record["_linearSpeedMps"] = 0.0 if previous is None else float(
            np.linalg.norm(position - previous[1]) / dt
        )
        record["_angularSpeedDeg"] = 0.0 if previous is None else float(record["_angleDeg"] / dt)
        previous = (float(record["unscaledTime"]), position)


def select_even(records: list[dict], limit: int) -> list[dict]:
    if limit <= 0 or len(records) <= limit:
        return records
    indices = np.linspace(0, len(records) - 1, limit, dtype=np.int64)
    return [records[int(index)] for index in np.unique(indices)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb", type=Path, default=DEFAULT_ADB)
    parser.add_argument("--serial", default="2G97C5ZH4501R5")
    parser.add_argument("--session", default="")
    parser.add_argument("--frames", type=int, default=450)
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument("--cell-size", type=float, default=0.10)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--capacity", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()

    session = args.session or latest_session(args.adb, args.serial)
    records = load_metadata(args.adb, args.serial, session)
    annotate_motion(records)
    add_motion_speeds(records)
    selected = select_even(records, args.frames)
    split_time = float(records[int((len(records) - 1) * args.train_fraction)]["unscaledTime"])

    sigma_floor = 0.004
    sigma_ceiling = min(0.030, args.cell_size * 0.40)
    models = [
        SurfaceEvidenceModel("Mean-K1", args.cell_size, 1, False, 3.0, sigma_floor, sigma_ceiling),
        SurfaceEvidenceModel("Adaptive-K1", args.cell_size, 1, True, 3.0, sigma_floor, sigma_ceiling),
        SurfaceEvidenceModel(
            f"Adaptive-K{args.capacity}",
            args.cell_size,
            args.capacity,
            True,
            3.0,
            sigma_floor,
            sigma_ceiling,
        ),
    ]

    training_frames = heldout_frames = 0
    training_observations = heldout_observations = 0
    alignment_observations = scoring_observations = 0
    frame_corrections: list[dict] = []
    frozen = False
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start : start + args.batch_size]
        for record, _raw, post in fetch_pair_batch(args.adb, args.serial, session, batch):
            is_training = float(record["unscaledTime"]) <= split_time
            if not is_training and not frozen:
                for model in models:
                    model.freeze_training(min_support=4.0, max_sigma=0.020, min_view_angle=3.0)
                frozen = True
            points, normals, views, sigmas, qualities = frame_observations(
                record, post, args.pixel_stride, 5.0
            )
            if is_training:
                training_frames += 1
                training_observations += len(points)
                for point, normal, view, sigma, quality in zip(
                    points, normals, views, sigmas, qualities
                ):
                    for model in models:
                        model.observe(
                            point,
                            normal,
                            view,
                            float(sigma),
                            float(quality),
                            int(record["pairIndex"]),
                        )
            else:
                heldout_frames += 1
                heldout_observations += len(points)
                observation_indices = np.arange(len(points), dtype=np.int64)
                alignment_mask = observation_indices % 5 == 0
                scoring_mask = ~alignment_mask
                alignment_observations += int(np.count_nonzero(alignment_mask))
                scoring_observations += int(np.count_nonzero(scoring_mask))
                correction_rotation, correction_translation, correction_report = (
                    estimate_frame_correction(
                        models[0],
                        points[alignment_mask],
                        normals[alignment_mask],
                        sigmas[alignment_mask],
                    )
                )
                correction_report["pairIndex"] = int(record["pairIndex"])
                frame_corrections.append(correction_report)

                score_points = points[scoring_mask]
                score_normals = normals[scoring_mask]
                score_sigmas = sigmas[scoring_mask]
                corrected_points, corrected_normals = transform_geometry(
                    score_points,
                    score_normals,
                    correction_rotation,
                    correction_translation,
                )
                for point, normal, sigma in zip(
                    score_points, score_normals, score_sigmas
                ):
                    for model in models:
                        model.score(point, normal, float(sigma))
                for point, normal, sigma in zip(
                    corrected_points, corrected_normals, score_sigmas
                ):
                    for model in models:
                        model.score(point, normal, float(sigma), corrected=True)

    if not frozen:
        for model in models:
            model.freeze_training(min_support=4.0, max_sigma=0.020, min_view_angle=3.0)

    reports = [model.report() for model in models]
    baseline = reports[0]
    for report in reports:
        raw_p95 = report["heldoutResidualMmP95"]
        corrected_p95 = report["poseCorrectedResidualMmP95"]
        if raw_p95 is not None and corrected_p95 is not None:
            report["poseCorrectionDeltaResidualP95Pct"] = (
                corrected_p95 / raw_p95 - 1.0
            ) * 100.0
    for report in reports[1:]:
        if baseline["heldoutResidualMmP95"] is not None and report["heldoutResidualMmP95"] is not None:
            report["deltaVsMeanResidualP95Pct"] = (
                report["heldoutResidualMmP95"] / baseline["heldoutResidualMmP95"] - 1.0
            ) * 100.0
        report["deltaVsMeanCoveragePoints"] = (
            report["heldoutCellCoveragePct"] - baseline["heldoutCellCoveragePct"]
        )
        corrected_baseline_p95 = baseline["poseCorrectedResidualMmP95"]
        corrected_report_p95 = report["poseCorrectedResidualMmP95"]
        if corrected_baseline_p95 is not None and corrected_report_p95 is not None:
            report["poseCorrectedDeltaVsMeanResidualP95Pct"] = (
                corrected_report_p95 / corrected_baseline_p95 - 1.0
            ) * 100.0
        report["poseCorrectedDeltaVsMeanCoveragePoints"] = (
            report["poseCorrectedCoveragePct"]
            - baseline["poseCorrectedCoveragePct"]
        )

    applied_corrections = [item for item in frame_corrections if item["applied"]]

    def correction_percentile(field: str, q: float) -> float | None:
        values = [
            float(item[field])
            for item in applied_corrections
            if item.get(field) is not None
        ]
        return None if not values else float(np.percentile(values, q))

    output = {
        "scope": "evidence-layer validation; not byte-equivalent production TSDF replay",
        "session": session,
        "completePairs": len(records),
        "selectedFrames": len(selected),
        "timeSplit": {"trainFraction": args.train_fraction, "splitUnscaledTime": split_time},
        "trainingFrames": training_frames,
        "heldoutFrames": heldout_frames,
        "trainingObservations": training_observations,
        "heldoutObservations": heldout_observations,
        "heldoutAlignmentObservations": alignment_observations,
        "heldoutScoringObservations": scoring_observations,
        "framePoseCorrection": {
            "referenceModel": "Mean-K1 frozen training evidence",
            "alignmentFractionPerHeldoutFrame": 0.20,
            "scoringFractionPerHeldoutFrame": 0.80,
            "attemptedFrames": len(frame_corrections),
            "appliedFrames": len(applied_corrections),
            "appliedPct": len(applied_corrections)
            * 100.0
            / max(len(frame_corrections), 1),
            "effectiveRankP50": correction_percentile("effectiveRank", 50),
            "translationMmP50": correction_percentile("translationMm", 50),
            "translationMmP95": correction_percentile("translationMm", 95),
            "rotationDegP50": correction_percentile("rotationDeg", 50),
            "rotationDegP95": correction_percentile("rotationDeg", 95),
            "alignmentResidualMmBeforeP50": correction_percentile(
                "alignmentResidualMmBefore", 50
            ),
            "alignmentResidualMmAfterP50": correction_percentile(
                "alignmentResidualMmAfter", 50
            ),
        },
        "sharedParameters": {
            "pixelStride": args.pixel_stride,
            "cellSizeMetres": args.cell_size,
            "matchingNeighborhoodCells": 27,
            "sigmaFloorMetres": sigma_floor,
            "sigmaCeilingMetres": sigma_ceiling,
            "standardizedSpawnThreshold": 3.0,
            "stableMinSupport": 4.0,
            "stableMaxSigmaMetres": 0.020,
            "stableMinViewAngleDegrees": 3.0,
        },
        "models": reports,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
