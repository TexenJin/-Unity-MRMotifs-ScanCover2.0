#!/usr/bin/env python3
"""Meta-structure guided Raw Depth fusion diagnostic.

This script moves one step beyond point-level Raw/Meta distance coloring:

1. Extract coarse structural plane families from Meta Scene Mesh.
2. Vote Raw room-coverage voxels onto those Meta structure families.
3. Keep Raw depth as the measured position, but use Meta families as the
   structural ID and smoothing/correction reference.

Outputs are diagnostic PLYs for CloudCompare, not final runtime meshes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d

from ScanCoverRoomRawCoverageMetaOverlay import (
    DEFAULT_META,
    apply_transform,
    auto_align_raw_to_meta,
    distance_colors,
    distribution,
    load_meta_sample,
    nearest_meta,
    normalize,
    read_room_voxels,
    write_cloud,
)


PALETTE = np.asarray(
    [
        (1.00, 0.05, 0.05),  # red
        (0.00, 0.85, 1.00),  # cyan
        (0.10, 1.00, 0.10),  # green
        (1.00, 0.80, 0.00),  # yellow
        (1.00, 0.00, 1.00),  # magenta
        (1.00, 0.45, 0.00),  # orange
        (0.20, 0.25, 1.00),  # blue
        (0.85, 1.00, 0.10),  # lime
        (0.75, 0.20, 1.00),  # purple
        (1.00, 1.00, 1.00),  # white
    ],
    dtype=np.float64,
)


@dataclass
class PlaneSegment:
    index: int
    normal: np.ndarray
    d: float
    inliers: np.ndarray


@dataclass
class PlaneFamily:
    index: int
    normal_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    d_sum: float = 0.0
    weight: int = 0
    segment_indices: list[int] = field(default_factory=list)

    @property
    def normal(self) -> np.ndarray:
        return normalize(self.normal_sum.reshape((1, 3)))[0]

    @property
    def d(self) -> float:
        return self.d_sum / max(1, self.weight)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Meta-guided structural fusion diagnostic.")
    parser.add_argument("repeat_session", type=Path, help="RepeatCoverage session folder.")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META, help="Welded Meta Scene Mesh PLY/OBJ.")
    parser.add_argument("--out", type=Path, default=None, help="Output folder.")
    parser.add_argument("--meta-sample-points", type=int, default=350000)
    parser.add_argument("--auto-align", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--align-voxel", type=float, default=0.05)
    parser.add_argument("--align-max-distance", type=float, default=0.35)
    parser.add_argument("--plane-distance", type=float, default=0.035)
    parser.add_argument("--plane-iterations", type=int, default=1200)
    parser.add_argument("--max-planes", type=int, default=18)
    parser.add_argument("--min-plane-points", type=int, default=4500)
    parser.add_argument("--merge-normal-deg", type=float, default=8.0)
    parser.add_argument("--merge-distance", type=float, default=0.12)
    parser.add_argument("--close-threshold", type=float, default=0.06)
    parser.add_argument("--usable-threshold", type=float, default=0.12)
    parser.add_argument("--far-threshold", type=float, default=0.20)
    parser.add_argument("--correction-alpha", type=float, default=0.20)
    parser.add_argument("--vote-radius", type=float, default=0.09, help="Raw-point neighborhood radius for in-plane family voting.")
    parser.add_argument("--vote-min-neighbors", type=int, default=8)
    parser.add_argument("--vote-majority", type=float, default=0.58)
    parser.add_argument("--vote-iterations", type=int, default=2)
    parser.add_argument("--island-voxel", type=float, default=0.08)
    parser.add_argument("--min-island-points", type=int, default=60)
    return parser.parse_args()


def canonical_plane(normal: np.ndarray, d: float) -> tuple[np.ndarray, float]:
    normal = normalize(normal.reshape((1, 3)))[0]
    axis = int(np.argmax(np.abs(normal)))
    if normal[axis] < 0.0:
        normal = -normal
        d = -d
    return normal, float(d)


def segment_meta_planes(
    meta_points: np.ndarray,
    distance: float,
    iterations: int,
    max_planes: int,
    min_points: int,
) -> tuple[list[PlaneSegment], np.ndarray]:
    remaining_cloud = o3d.geometry.PointCloud()
    remaining_cloud.points = o3d.utility.Vector3dVector(meta_points)
    remaining_indices = np.arange(len(meta_points), dtype=np.int64)

    segments: list[PlaneSegment] = []
    meta_segment_ids = np.full((len(meta_points),), -1, dtype=np.int32)

    for plane_index in range(max_planes):
        if len(remaining_indices) < min_points:
            break
        model, inliers_local = remaining_cloud.segment_plane(
            distance_threshold=distance,
            ransac_n=3,
            num_iterations=iterations,
        )
        if len(inliers_local) < min_points:
            break

        inliers_local_arr = np.asarray(inliers_local, dtype=np.int64)
        inliers_global = remaining_indices[inliers_local_arr]
        n, d = canonical_plane(np.asarray(model[:3], dtype=np.float64), float(model[3]))
        segments.append(PlaneSegment(plane_index, n, d, inliers_global))
        meta_segment_ids[inliers_global] = plane_index

        keep = np.ones((len(remaining_indices),), dtype=bool)
        keep[inliers_local_arr] = False
        remaining_indices = remaining_indices[keep]
        remaining_cloud = remaining_cloud.select_by_index(inliers_local, invert=True)

    return segments, meta_segment_ids


def merge_plane_families(
    segments: list[PlaneSegment],
    normal_deg: float,
    distance: float,
) -> tuple[list[PlaneFamily], dict[int, int]]:
    families: list[PlaneFamily] = []
    segment_to_family: dict[int, int] = {}
    cos_threshold = math.cos(math.radians(normal_deg))

    for segment in segments:
        matched: PlaneFamily | None = None
        sn = segment.normal
        sd = segment.d
        for family in families:
            fn = family.normal
            dot = float(np.dot(segment.normal, fn))
            sd = segment.d
            sn = segment.normal
            if dot < 0.0:
                dot = -dot
                sd = -sd
                sn = -sn
            if dot >= cos_threshold and abs(sd - family.d) <= distance:
                matched = family
                break

        if matched is None:
            matched = PlaneFamily(index=len(families))
            families.append(matched)

        weight = int(len(segment.inliers))
        matched.normal_sum += sn * weight
        matched.d_sum += sd * weight
        matched.weight += weight
        matched.segment_indices.append(segment.index)
        segment_to_family[segment.index] = matched.index

    return families, segment_to_family


def family_colors(ids: np.ndarray) -> np.ndarray:
    colors = np.zeros((len(ids), 3), dtype=np.float64)
    colors[:] = (0.12, 0.12, 0.12)
    for i, family_id in enumerate(ids):
        if family_id >= 0:
            colors[i] = PALETTE[int(family_id) % len(PALETTE)]
    return colors


def nearest_indices(query_points: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(reference_points)
    tree = o3d.geometry.KDTreeFlann(cloud)
    ids = np.empty((len(query_points),), dtype=np.int64)
    for i, point in enumerate(query_points):
        _, idx, _ = tree.search_knn_vector_3d(point, 1)
        ids[i] = int(idx[0])
    return ids


def smooth_family_votes(
    points: np.ndarray,
    family_ids: np.ndarray,
    distances: np.ndarray,
    families: list[PlaneFamily],
    radius: float,
    min_neighbors: int,
    majority: float,
    iterations: int,
    usable_distance: float,
) -> tuple[np.ndarray, dict[str, object]]:
    if len(points) == 0 or len(families) == 0 or iterations <= 0:
        return family_ids.copy(), {"enabled": False}

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    tree = o3d.geometry.KDTreeFlann(cloud)
    smoothed = family_ids.copy()
    changes_by_iteration: list[int] = []

    for _ in range(iterations):
        next_ids = smoothed.copy()
        changed = 0
        for i, point in enumerate(points):
            if distances[i] > usable_distance:
                continue
            _, idx, _ = tree.search_radius_vector_3d(point, radius)
            if len(idx) < min_neighbors:
                continue
            neighbor_ids = smoothed[np.asarray(idx, dtype=np.int64)]
            neighbor_ids = neighbor_ids[neighbor_ids >= 0]
            if len(neighbor_ids) < min_neighbors:
                continue
            values, counts = np.unique(neighbor_ids, return_counts=True)
            winner_pos = int(np.argmax(counts))
            winner = int(values[winner_pos])
            ratio = float(counts[winner_pos] / max(1, len(neighbor_ids)))
            if ratio < majority:
                continue
            current = int(smoothed[i])
            if current == winner:
                continue
            if current >= 0:
                dot = abs(float(np.dot(families[current].normal, families[winner].normal)))
                if dot < math.cos(math.radians(15.0)):
                    continue
            next_ids[i] = winner
            changed += 1
        smoothed = next_ids
        changes_by_iteration.append(changed)
        if changed == 0:
            break

    return smoothed, {
        "enabled": True,
        "radiusMeters": radius,
        "minNeighbors": min_neighbors,
        "majority": majority,
        "iterationsRequested": iterations,
        "changesByIteration": changes_by_iteration,
        "totalChanged": int(sum(changes_by_iteration)),
    }


def connected_component_filter(
    points: np.ndarray,
    family_ids: np.ndarray,
    voxel_size: float,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    stable_ids = family_ids.copy()
    small_island = np.zeros((len(points),), dtype=bool)
    if len(points) == 0:
        return stable_ids, small_island, {"enabled": False}

    total_removed = 0
    family_reports: list[dict[str, int]] = []
    for family_id in sorted(int(v) for v in np.unique(family_ids) if v >= 0):
        indices = np.flatnonzero(family_ids == family_id)
        if len(indices) == 0:
            continue
        keys = np.floor(points[indices] / voxel_size).astype(np.int64)
        key_to_local: dict[tuple[int, int, int], list[int]] = {}
        for local_i, key in enumerate(keys):
            key_to_local.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(local_i)

        visited: set[tuple[int, int, int]] = set()
        component_count = 0
        kept_components = 0
        removed_points = 0
        for key in key_to_local:
            if key in visited:
                continue
            stack = [key]
            visited.add(key)
            component_keys: list[tuple[int, int, int]] = []
            while stack:
                cur = stack.pop()
                component_keys.append(cur)
                cx, cy, cz = cur
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            nb = (cx + dx, cy + dy, cz + dz)
                            if nb in key_to_local and nb not in visited:
                                visited.add(nb)
                                stack.append(nb)

            component_count += 1
            component_locals: list[int] = []
            for ck in component_keys:
                component_locals.extend(key_to_local[ck])
            component_indices = indices[np.asarray(component_locals, dtype=np.int64)]
            if len(component_indices) < min_points:
                small_island[component_indices] = True
                stable_ids[component_indices] = -1
                removed_points += int(len(component_indices))
            else:
                kept_components += 1

        total_removed += removed_points
        family_reports.append(
            {
                "family": family_id,
                "components": component_count,
                "keptComponents": kept_components,
                "removedSmallIslandPoints": removed_points,
            }
        )

    return stable_ids, small_island, {
        "enabled": True,
        "voxelSizeMeters": voxel_size,
        "minIslandPoints": min_points,
        "removedSmallIslandPoints": int(total_removed),
        "families": family_reports,
    }


def write_family_summary(
    path: Path,
    families: list[PlaneFamily],
    meta_family_ids: np.ndarray,
    raw_family_ids: np.ndarray,
    raw_masks: dict[str, np.ndarray],
    raw_signed_to_family: np.ndarray,
) -> None:
    fields = [
        "family",
        "segments",
        "metaPoints",
        "rawPoints",
        "stableRaw",
        "correctedRaw",
        "riskRaw",
        "normalX",
        "normalY",
        "normalZ",
        "d",
        "rawSignedMedian",
        "rawSignedP90Abs",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for family in families:
            fid = family.index
            raw_mask = raw_family_ids == fid
            signed = raw_signed_to_family[raw_mask]
            abs_signed = np.abs(signed) if len(signed) > 0 else np.asarray([], dtype=np.float64)
            n = family.normal
            writer.writerow(
                {
                    "family": fid,
                    "segments": " ".join(str(v) for v in family.segment_indices),
                    "metaPoints": int(np.count_nonzero(meta_family_ids == fid)),
                    "rawPoints": int(np.count_nonzero(raw_mask)),
                    "stableRaw": int(np.count_nonzero(raw_masks["stable"] & raw_mask)),
                    "correctedRaw": int(np.count_nonzero(raw_masks["corrected"] & raw_mask)),
                    "riskRaw": int(np.count_nonzero(raw_masks["risk"] & raw_mask)),
                    "normalX": float(n[0]),
                    "normalY": float(n[1]),
                    "normalZ": float(n[2]),
                    "d": float(family.d),
                    "rawSignedMedian": float(np.median(signed)) if len(signed) else "",
                    "rawSignedP90Abs": float(np.percentile(abs_signed, 90)) if len(abs_signed) else "",
                }
            )


def main() -> int:
    args = parse_args()
    session = args.repeat_session.resolve()
    voxels_csv = session / "room_raw_coverage" / "room_raw_coverage_voxels.csv"
    if not voxels_csv.exists():
        raise FileNotFoundError(voxels_csv)

    out_dir = args.out.resolve() if args.out else session / "meta_guided_structure_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_room_voxels(voxels_csv)
    raw_points = raw["points"]
    raw_normals = raw["normals"]
    raw_stable = raw["stable"]
    raw_risk = raw["risk"]

    meta_points, meta_normals = load_meta_sample(args.meta.resolve(), args.meta_sample_points)
    alignment: dict[str, object] = {"enabled": False}
    if args.auto_align:
        transform, alignment = auto_align_raw_to_meta(
            raw_points,
            raw_stable,
            meta_points,
            args.align_voxel,
            args.align_max_distance,
        )
        raw_points = apply_transform(raw_points, transform)
        raw_normals = normalize(raw_normals @ transform[:3, :3].T)

    nearest_meta_points, nearest_meta_normals, raw_meta_distances, raw_meta_signed = nearest_meta(
        raw_points,
        meta_points,
        meta_normals,
    )

    segments, meta_segment_ids = segment_meta_planes(
        meta_points,
        args.plane_distance,
        args.plane_iterations,
        args.max_planes,
        args.min_plane_points,
    )
    families, segment_to_family = merge_plane_families(
        segments,
        args.merge_normal_deg,
        args.merge_distance,
    )

    meta_family_ids = np.full((len(meta_points),), -1, dtype=np.int32)
    for segment_id, family_id in segment_to_family.items():
        meta_family_ids[meta_segment_ids == segment_id] = family_id

    nearest_ids = nearest_indices(raw_points, meta_points)
    raw_family_ids = meta_family_ids[nearest_ids]
    raw_family_ids_before_stabilize = raw_family_ids.copy()
    raw_family_ids, vote_report = smooth_family_votes(
        raw_points,
        raw_family_ids,
        raw_meta_distances,
        families,
        args.vote_radius,
        args.vote_min_neighbors,
        args.vote_majority,
        args.vote_iterations,
        args.usable_threshold,
    )
    raw_family_ids, small_island_mask, island_report = connected_component_filter(
        raw_points,
        raw_family_ids,
        args.island_voxel,
        args.min_island_points,
    )

    family_normals = np.zeros_like(raw_points)
    family_ds = np.zeros((len(raw_points),), dtype=np.float64)
    for family in families:
        mask = raw_family_ids == family.index
        family_normals[mask] = family.normal
        family_ds[mask] = family.d
    raw_signed_to_family = np.sum(raw_points * family_normals, axis=1) + family_ds
    corrected_points = raw_points.copy()
    family_valid = raw_family_ids >= 0

    close = raw_meta_distances <= args.close_threshold
    usable = raw_meta_distances <= args.usable_threshold
    far = raw_meta_distances > args.far_threshold
    stable_mask = family_valid & raw_stable & ~raw_risk & close
    corrected_mask = family_valid & raw_stable & ~raw_risk & ~close & usable
    risk_mask = raw_risk | far | ~family_valid | small_island_mask

    alpha = max(0.0, min(1.0, args.correction_alpha))
    corrected_points[corrected_mask] = (
        raw_points[corrected_mask]
        - (raw_signed_to_family[corrected_mask] * alpha)[:, None] * family_normals[corrected_mask]
    )

    candidate_mask = stable_mask | corrected_mask
    raw_family_colors = family_colors(raw_family_ids)
    meta_family_colors = family_colors(meta_family_ids)
    distance_layer_colors = distance_colors(
        raw_meta_distances,
        args.close_threshold,
        args.usable_threshold,
        args.far_threshold,
    )

    write_cloud(out_dir / "meta_structure_families.ply", meta_points, meta_family_colors, meta_normals)
    write_cloud(out_dir / "raw_voted_structure_families_before_stabilize.ply", raw_points, family_colors(raw_family_ids_before_stabilize), raw_normals)
    write_cloud(out_dir / "raw_voted_structure_families.ply", raw_points, raw_family_colors, raw_normals)
    write_cloud(out_dir / "raw_structure_stable.ply", raw_points[stable_mask], raw_family_colors[stable_mask], raw_normals[stable_mask])
    write_cloud(out_dir / "raw_structure_corrected.ply", corrected_points[corrected_mask], raw_family_colors[corrected_mask], raw_normals[corrected_mask])
    write_cloud(out_dir / "raw_structure_risk.ply", raw_points[risk_mask], np.tile((1.0, 0.0, 0.0), (np.count_nonzero(risk_mask), 1)), raw_normals[risk_mask])
    write_cloud(out_dir / "raw_structure_candidate.ply", corrected_points[candidate_mask], raw_family_colors[candidate_mask], raw_normals[candidate_mask])
    write_cloud(out_dir / "raw_structure_distance_layers.ply", raw_points, distance_layer_colors, raw_normals)

    masks = {"stable": stable_mask, "corrected": corrected_mask, "risk": risk_mask}
    write_family_summary(out_dir / "structure_family_summary.csv", families, meta_family_ids, raw_family_ids, masks, raw_signed_to_family)

    report = {
        "repeatSession": str(session),
        "roomVoxelCsv": str(voxels_csv),
        "metaReference": str(args.meta.resolve()),
        "outputDirectory": str(out_dir),
        "alignment": alignment,
        "rawVoxelCount": int(len(raw_points)),
        "metaSampleCount": int(len(meta_points)),
        "planeSegmentCount": int(len(segments)),
        "planeFamilyCount": int(len(families)),
        "thresholdsMeters": {
            "planeDistance": args.plane_distance,
            "mergeDistance": args.merge_distance,
            "close": args.close_threshold,
            "usable": args.usable_threshold,
            "far": args.far_threshold,
            "correctionAlpha": alpha,
            "voteRadius": args.vote_radius,
            "islandVoxel": args.island_voxel,
        },
        "stabilization": {
            "neighborhoodVote": vote_report,
            "connectedComponents": island_report,
        },
        "counts": {
            "rawAssignedToMetaFamily": int(np.count_nonzero(family_valid)),
            "stable": int(np.count_nonzero(stable_mask)),
            "corrected": int(np.count_nonzero(corrected_mask)),
            "risk": int(np.count_nonzero(risk_mask)),
            "candidate": int(np.count_nonzero(candidate_mask)),
        },
        "rawMetaDistanceMeters": distribution(raw_meta_distances),
        "rawSignedToFamilyMeters": distribution(raw_signed_to_family[family_valid]),
        "stableRawSignedToFamilyMeters": distribution(raw_signed_to_family[stable_mask]),
        "correctedRawSignedToFamilyMeters": distribution(raw_signed_to_family[corrected_mask]),
        "familySummaryCsv": str(out_dir / "structure_family_summary.csv"),
        "interpretation": [
            "meta_structure_families.ply shows the coarse structural faces extracted from Meta Scene Mesh.",
            "raw_voted_structure_families.ply shows Raw Depth voxels voting onto those Meta structure families.",
            "raw_structure_candidate.ply is the current structural seed candidate: stable Raw points unchanged, usable Raw points lightly corrected toward their family plane.",
            "raw_structure_risk.ply remains excluded from the main structural seed.",
        ],
    }
    (out_dir / "structure_fusion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({
        "rawVoxelCount": report["rawVoxelCount"],
        "metaSampleCount": report["metaSampleCount"],
        "planeSegmentCount": report["planeSegmentCount"],
        "planeFamilyCount": report["planeFamilyCount"],
        "counts": report["counts"],
        "outputDirectory": str(out_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
