#!/usr/bin/env python3
"""Offline space-time attribution for virtual-probe Reject episodes.

The tool consumes an immutable replay session.  It never changes Unity state or
production data.  Every Reject is joined to its consecutive challenge witnesses,
reprojected into the matching platform/post-QRS depth pair, and grouped into
space-time components.  Classifications are evidence labels, not physical ground
truth; the component containing a successful Reopen is the controlled positive
anchor when one exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


CELL_METRES = 0.05
RELIABLE_GAP_MM = 50.0
CLUSTER_RADIUS_METRES = 0.16
CLUSTER_FRAME_GAP = 200
EDGE_SPAN_MM = 80.0
SOURCE_REPROJECTION_MM = 35.0


def number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def matrix(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.size != 16:
        raise ValueError(f"matrix requires 16 values, got {result.size}")
    return result.reshape(4, 4)


def recorded_eye(meta: dict) -> int:
    eye = int(meta.get("recordedEyeIndex", 1))
    poses = meta.get("trackingPoses", [])
    return max(0, min(eye, len(poses) - 1)) if poses else eye


def project_world(point: np.ndarray, view: np.ndarray, projection: np.ndarray) -> Optional[np.ndarray]:
    clip = projection @ (view @ np.r_[point, 1.0])
    if abs(float(clip[3])) <= 1e-12:
        return None
    ndc = clip[:3] / clip[3]
    return ndc * 0.5 + 0.5


def reconstruct_world(
    px: int,
    py: int,
    depth: float,
    width: int,
    height: int,
    projection_inverse: np.ndarray,
    view_inverse: np.ndarray,
) -> np.ndarray:
    hcs = np.asarray(
        [px / width * 2.0 - 1.0, py / height * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0],
        dtype=np.float64,
    )
    world_h = view_inverse @ (projection_inverse @ hcs)
    return world_h[:3] / world_h[3]


def valid_depth(value: float) -> bool:
    return math.isfinite(value) and 0.0 < value < 1.0


def quaternion_angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    dot = float(np.clip(abs(np.dot(aa, bb)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def cell_key(row: dict[str, str]) -> tuple[int, int, int, int]:
    return (
        integer(row.get("cellX")),
        integer(row.get("cellY")),
        integer(row.get("cellZ")),
        integer(row.get("axis")),
    )


def xyz(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray(
        [number(row.get(prefix + "X")), number(row.get(prefix + "Y")), number(row.get(prefix + "Z"))],
        dtype=np.float64,
    )


@dataclass
class Episode:
    index: int
    key: tuple[int, int, int, int]
    reject: dict[str, str]
    challenges: list[dict[str, str]]
    recovery: list[dict[str, str]]
    reopened: bool
    cluster_id: int = -1

    @property
    def frame(self) -> int:
        return integer(self.reject.get("gunGelFrame"))

    @property
    def target(self) -> np.ndarray:
        return xyz(self.reject, "target")


class UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: int, b: int) -> None:
        aa = self.find(a)
        bb = self.find(b)
        if aa != bb:
            self.parent[bb] = aa


class FrameStore:
    def __init__(self, session: Path) -> None:
        self.frames = session / "depth_pairs" / "frames"
        self.manifest = read_csv(session / "depth_pairs" / "manifest.csv")
        self.by_pair = {integer(row.get("pairIndex")): row for row in self.manifest}
        self.by_platform = {integer(row.get("platformFrame")): row for row in self.manifest}
        self.cache: dict[int, tuple[dict, np.ndarray, np.ndarray]] = {}

    def resolve_pair(self, event: dict[str, str]) -> int:
        direct = integer(event.get("gunGelFrame"))
        platform = integer(event.get("platformFrame"))
        if direct in self.by_pair and integer(self.by_pair[direct].get("platformFrame")) == platform:
            return direct
        row = self.by_platform.get(platform)
        if row is None:
            raise KeyError(f"no depth pair for gunGel={direct} platform={platform}")
        return integer(row.get("pairIndex"))

    def load(self, event: dict[str, str]) -> tuple[int, dict, np.ndarray, np.ndarray]:
        pair = self.resolve_pair(event)
        if pair not in self.cache:
            meta_path = self.frames / f"frame_{pair:06d}_meta.json"
            raw_path = self.frames / f"frame_{pair:06d}_platform_pre_qrs.f32"
            post_path = self.frames / f"frame_{pair:06d}_post_qrs.f32"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            count = int(meta["width"]) * int(meta["height"]) * int(meta["layers"])
            raw = np.fromfile(raw_path, dtype="<f4", count=count)
            post = np.fromfile(post_path, dtype="<f4", count=count)
            if raw.size != count or post.size != count:
                raise IOError(f"short depth pair {pair}: raw={raw.size} post={post.size} expected={count}")
            self.cache[pair] = (meta, raw, post)
        meta, raw, post = self.cache[pair]
        return pair, meta, raw, post


def build_episodes(events: list[dict[str, str]]) -> list[Episode]:
    histories: dict[tuple[int, int, int, int], list[dict[str, str]]] = defaultdict(list)
    episodes: list[Episode] = []
    active: dict[tuple[int, int, int, int], Episode] = {}
    ordered = sorted(events, key=lambda row: (integer(row.get("gunGelFrame")), row.get("event", "")))
    for row in ordered:
        key = cell_key(row)
        event = row.get("event", "")
        if event.startswith("challenge_"):
            votes = integer(row.get("challengeVotes"))
            history = histories[key]
            previous_votes = integer(history[-1].get("challengeVotes")) if history else 0
            if votes <= 1 or votes != previous_votes + 1:
                history = []
                histories[key] = history
            history.append(row)
            if row.get("reason") == "rejected_by_three_free_space_votes":
                episode = Episode(len(episodes), key, row, list(history), [], False)
                episodes.append(episode)
                active[key] = episode
        elif event == "recovery_independent" and key in active:
            episode = active[key]
            episode.recovery.append(row)
            if row.get("reason") == "reopened_by_independent_safe_support":
                episode.reopened = True
                active.pop(key, None)
        elif event == "support_independent":
            histories.pop(key, None)
    return episodes


def assign_clusters(episodes: list[Episode]) -> dict[int, list[Episode]]:
    union = UnionFind(len(episodes))
    for i, left in enumerate(episodes):
        for j in range(i + 1, len(episodes)):
            right = episodes[j]
            if abs(left.frame - right.frame) > CLUSTER_FRAME_GAP:
                continue
            if float(np.linalg.norm(left.target - right.target)) <= CLUSTER_RADIUS_METRES:
                union.union(i, j)
    roots: dict[int, list[Episode]] = defaultdict(list)
    for index, episode in enumerate(episodes):
        roots[union.find(index)].append(episode)
    ordered = sorted(roots.values(), key=lambda group: min(item.frame for item in group))
    result: dict[int, list[Episode]] = {}
    for cluster_id, group in enumerate(ordered, start=1):
        result[cluster_id] = group
        for episode in group:
            episode.cluster_id = cluster_id
    return result


def local_depth_span_mm(image: np.ndarray, px: int, py: int, radius: int = 2) -> float:
    height, width = image.shape
    x0, x1 = max(0, px - radius), min(width, px + radius + 1)
    y0, y1 = max(0, py - radius), min(height, py + radius + 1)
    patch = image[y0:y1, x0:x1]
    patch = patch[np.isfinite(patch) & (patch > 0.0) & (patch < 1.0)]
    if patch.size < 3:
        return math.nan
    # This is converted to metres by the caller because the projection is frame-specific.
    return float(np.percentile(patch, 90) - np.percentile(patch, 10))


def linearize(depth: np.ndarray, projection: np.ndarray) -> np.ndarray:
    z = depth.astype(np.float64) * 2.0 - 1.0
    return np.abs(float(projection[2, 3]) / (z + float(projection[2, 2])))


def witness_metrics(
    store: FrameStore,
    row: dict[str, str],
    key: tuple[int, int, int, int],
    candidate_normal: np.ndarray,
) -> dict[str, object]:
    pair, meta, raw_flat, post_flat = store.load(row)
    eye = recorded_eye(meta)
    width = int(meta["width"])
    height = int(meta["height"])
    raw = raw_flat.reshape(height, width)
    post = post_flat.reshape(height, width)
    view = matrix(meta["view"][eye])
    projection = matrix(meta["projection"][eye])
    view_inverse = matrix(meta["viewInverse"][eye])
    projection_inverse = matrix(meta["projectionInverse"][eye])
    camera = view_inverse[:3, 3]
    source = xyz(row, "source")
    target = xyz(row, "target")
    source_uv = project_world(source, view, projection)
    target_uv = project_world(target, view, projection)
    if source_uv is None or target_uv is None:
        raise ValueError(f"unprojectable witness pair={pair}")

    projected_x = int(np.clip(source_uv[0] * width, 0, width - 1))
    projected_y = int(np.clip(source_uv[1] * height, 0, height - 1))
    best: Optional[tuple[float, int, int, np.ndarray]] = None
    for py in range(max(0, projected_y - 3), min(height, projected_y + 4)):
        for px in range(max(0, projected_x - 3), min(width, projected_x + 4)):
            value = float(post[py, px])
            if not valid_depth(value):
                continue
            point = reconstruct_world(px, py, value, width, height, projection_inverse, view_inverse)
            residual = float(np.linalg.norm(point - source))
            if best is None or residual < best[0]:
                best = (residual, px, py, point)
    if best is None:
        return {
            "pairIndex": pair,
            "depthLocated": False,
            "callbackToPreprocessMs": number(meta.get("callbackToPreprocessMs")),
        }

    reprojection, px, py, post_world = best
    raw_value = float(raw[py, px])
    post_value = float(post[py, px])
    ray = source - camera
    hit_range = float(np.linalg.norm(ray))
    ray_direction = ray / max(hit_range, 1e-12)
    candidate_along = float(np.dot(target - camera, ray_direction))
    closest = camera + ray_direction * candidate_along
    ray_distance_mm = float(np.linalg.norm(closest - target) * 1000.0)
    plane_denominator = float(np.dot(ray_direction, candidate_normal))
    plane_patch_hit = False
    plane_lateral_mm = math.nan
    plane_intersection_range = math.nan
    if abs(plane_denominator) > 1e-5:
        plane_intersection_range = float(
            np.dot(target - camera, candidate_normal) / plane_denominator
        )
        plane_point = camera + ray_direction * plane_intersection_range
        cell_min = np.asarray(key[:3], dtype=np.float64) * CELL_METRES
        cell_max = cell_min + CELL_METRES
        plane_patch_hit = bool(
            0.0 < plane_intersection_range < hit_range
            and np.all(plane_point >= cell_min - 0.001)
            and np.all(plane_point <= cell_max + 0.001)
        )
        plane_lateral_mm = float(np.linalg.norm(plane_point - target) * 1000.0)
    fingerprint_attributed = row.get("association") == "fingerprint_patch_hit"
    if fingerprint_attributed:
        # v3 sessions seal the Reject-time patch intersection directly.  Prefer
        # that immutable attribution over reconstructing with the final cell normal.
        plane_patch_hit = True
        plane_lateral_mm = number(row.get("footprintOffsetMm"), plane_lateral_mm)
        plane_intersection_range = number(row.get("intersectionRangeMm"), 0.0) / 1000.0
    post_range = float(np.linalg.norm(post_world - camera))
    post_gap_mm = (post_range - candidate_along) * 1000.0
    raw_gap_mm = math.nan
    raw_post_mm = math.nan
    if valid_depth(raw_value):
        raw_world = reconstruct_world(px, py, raw_value, width, height, projection_inverse, view_inverse)
        raw_range = float(np.linalg.norm(raw_world - camera))
        raw_gap_mm = (raw_range - candidate_along) * 1000.0
        raw_post_mm = abs(raw_range - post_range) * 1000.0

    post_patch = post[max(0, py - 2) : min(height, py + 3), max(0, px - 2) : min(width, px + 3)]
    valid_patch = post_patch[np.isfinite(post_patch) & (post_patch > 0.0) & (post_patch < 1.0)]
    if valid_patch.size >= 3:
        metres = linearize(valid_patch, projection)
        edge_span_mm = float((np.percentile(metres, 90) - np.percentile(metres, 10)) * 1000.0)
    else:
        edge_span_mm = math.nan

    callback_pose = meta.get("callbackHeadWorldPose", {}).get("pose", {})
    preprocess_pose = meta.get("preprocessHeadWorldPose", {}).get("pose", {})
    pose_translation_mm = math.nan
    pose_rotation_deg = math.nan
    if callback_pose and preprocess_pose:
        pose_translation_mm = float(
            np.linalg.norm(
                np.asarray(callback_pose["position"], dtype=np.float64)
                - np.asarray(preprocess_pose["position"], dtype=np.float64)
            )
            * 1000.0
        )
        pose_rotation_deg = quaternion_angle_deg(callback_pose["rotation"], preprocess_pose["rotation"])

    target_px = np.asarray([target_uv[0] * width, target_uv[1] * height])
    source_px = np.asarray([source_uv[0] * width, source_uv[1] * height])
    return {
        "pairIndex": pair,
        "depthLocated": True,
        "sourcePixelX": px,
        "sourcePixelY": py,
        "sourceTargetPixelDistance": float(np.linalg.norm(source_px - target_px)),
        "sourceReprojectionMm": reprojection * 1000.0,
        "targetRayDistanceMm": ray_distance_mm,
        "planePatchHit": plane_patch_hit,
        "planeIntersectionLateralMm": plane_lateral_mm,
        "planeIntersectionRange": plane_intersection_range,
        "fingerprintAttributed": fingerprint_attributed,
        "fingerprintId": integer(row.get("fingerprintId")),
        "fingerprintGeneration": integer(row.get("fingerprintGeneration")),
        "fingerprintPatchIndex": integer(row.get("fingerprintPatchIndex"), -1),
        "rawGapMm": raw_gap_mm,
        "postGapMm": post_gap_mm,
        "rawPostMm": raw_post_mm,
        "localPostDepthSpanMm": edge_span_mm,
        "callbackToPreprocessMs": number(meta.get("callbackToPreprocessMs")),
        "callbackPoseTranslationMm": pose_translation_mm,
        "callbackPoseRotationDeg": pose_rotation_deg,
        "angularDegPerSec": number(meta.get("angularDegPerSec")),
        "linearMps": number(meta.get("linearMps")),
    }


def finite(values: Iterable[object]) -> list[float]:
    result = []
    for value in values:
        candidate = number(value, math.nan)
        if math.isfinite(candidate):
            result.append(candidate)
    return result


def percentile(values: Iterable[object], value: float) -> float:
    clean = finite(values)
    return float(np.percentile(clean, value)) if clean else math.nan


def episode_row(
    episode: Episode,
    witnesses: list[dict[str, object]],
    cluster_size: int,
    controlled_cluster: bool,
) -> dict[str, object]:
    raw_support = sum(number(item.get("rawGapMm"), -math.inf) >= RELIABLE_GAP_MM for item in witnesses)
    post_support = sum(number(item.get("postGapMm"), -math.inf) >= RELIABLE_GAP_MM for item in witnesses)
    preprocess_only = sum(
        number(item.get("postGapMm"), -math.inf) >= RELIABLE_GAP_MM
        and number(item.get("rawGapMm"), -math.inf) < RELIABLE_GAP_MM
        for item in witnesses
    )
    plane_hits = sum(bool(item.get("planePatchHit")) for item in witnesses)
    fingerprint_hits = sum(bool(item.get("fingerprintAttributed")) for item in witnesses)
    plane_misses = len(witnesses) - plane_hits
    reprojection_suspect = any(
        number(item.get("sourceReprojectionMm"), math.inf) >= SOURCE_REPROJECTION_MM
        for item in witnesses
    )
    edge_witnesses = sum(number(item.get("localPostDepthSpanMm"), 0.0) >= EDGE_SPAN_MM for item in witnesses)
    timing_witnesses = sum(
        number(item.get("callbackToPreprocessMs"), 0.0) >= 50.0
        or number(item.get("callbackPoseTranslationMm"), 0.0) >= 10.0
        or number(item.get("callbackPoseRotationDeg"), 0.0) >= 1.0
        for item in witnesses
    )

    if episode.reopened:
        classification = "controlled_change_reopened"
        confidence = "confirmed_contract"
    elif controlled_cluster:
        classification = "controlled_change_not_revisited"
        confidence = "spatiotemporal_inference"
    elif raw_support < 2 or preprocess_only > 0:
        classification = "preprocess_or_raw_testimony_suspect"
        confidence = "diagnostic"
    elif plane_misses * 2 >= max(len(witnesses), 1) or reprojection_suspect:
        classification = "ray_candidate_plane_miss_suspect"
        confidence = "diagnostic"
    elif edge_witnesses * 2 >= max(len(witnesses), 1):
        classification = "edge_or_parallax_suspect"
        confidence = "diagnostic"
    elif timing_witnesses > 0:
        classification = "timing_or_pose_suspect"
        confidence = "diagnostic"
    elif cluster_size > 1:
        classification = "coherent_unverified_change"
        confidence = "unverified_physical_truth"
    else:
        classification = "isolated_unverified_change"
        confidence = "unverified_physical_truth"

    target = episode.target
    return {
        "episode": episode.index + 1,
        "cluster": episode.cluster_id,
        "controlled_cluster": int(controlled_cluster),
        "cell": ":".join(str(value) for value in episode.key),
        "reject_frame": episode.frame,
        "target_x_m": target[0],
        "target_y_m": target[1],
        "target_z_m": target[2],
        "challenge_votes": integer(episode.reject.get("challengeVotes")),
        "independent_challenge_views": integer(episode.reject.get("independentChallengeViews")),
        "witness_rows": len(witnesses),
        "raw_free_space_witnesses": raw_support,
        "post_free_space_witnesses": post_support,
        "preprocess_only_witnesses": preprocess_only,
        "candidate_plane_hit_witnesses": plane_hits,
        "candidate_plane_miss_witnesses": plane_misses,
        "fingerprint_attributed_witnesses": fingerprint_hits,
        "fingerprint_id": integer(episode.reject.get("fingerprintId")),
        "fingerprint_generation": integer(episode.reject.get("fingerprintGeneration")),
        "edge_witnesses": edge_witnesses,
        "timing_witnesses": timing_witnesses,
        "target_ray_distance_mm_p95": percentile((item.get("targetRayDistanceMm") for item in witnesses), 95),
        "plane_intersection_lateral_mm_p95": percentile(
            (item.get("planeIntersectionLateralMm") for item in witnesses), 95
        ),
        "source_reprojection_mm_p95": percentile((item.get("sourceReprojectionMm") for item in witnesses), 95),
        "raw_post_mm_p95": percentile((item.get("rawPostMm") for item in witnesses), 95),
        "local_depth_span_mm_p95": percentile((item.get("localPostDepthSpanMm") for item in witnesses), 95),
        "callback_to_preprocess_ms_p95": percentile((item.get("callbackToPreprocessMs") for item in witnesses), 95),
        "recovery_witnesses": len(episode.recovery),
        "reopened": int(episode.reopened),
        "classification": classification,
        "confidence": confidence,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="Local replay session root")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    session = args.session.resolve()
    events = read_csv(session / "probe_shadow" / "verdict_events.csv")
    episodes = build_episodes(events)
    if not episodes:
        raise RuntimeError("session contains no Reject episodes")
    clusters = assign_clusters(episodes)
    controlled_ids = {
        cluster_id
        for cluster_id, group in clusters.items()
        if any(episode.reopened for episode in group)
    }
    store = FrameStore(session)
    cell_rows = read_csv(session / "probe_shadow" / "verdict_cells.csv")
    cell_normals = {
        cell_key(row): np.asarray(
            [number(row.get("normalX")), number(row.get("normalY")), number(row.get("normalZ"))],
            dtype=np.float64,
        )
        for row in cell_rows
    }
    audit_rows: list[dict[str, object]] = []
    witness_errors: list[str] = []
    for episode in episodes:
        metrics = []
        candidate_normal = cell_normals.get(episode.key, np.zeros(3, dtype=np.float64))
        normal_length = float(np.linalg.norm(candidate_normal))
        if normal_length > 1e-8:
            candidate_normal = candidate_normal / normal_length
        for witness in episode.challenges:
            try:
                metrics.append(witness_metrics(store, witness, episode.key, candidate_normal))
            except Exception as exc:  # preserve the rest of the immutable audit
                witness_errors.append(
                    f"episode={episode.index + 1} frame={witness.get('gunGelFrame')} error={exc}"
                )
        audit_rows.append(
            episode_row(
                episode,
                metrics,
                len(clusters[episode.cluster_id]),
                episode.cluster_id in controlled_ids,
            )
        )

    cluster_rows: list[dict[str, object]] = []
    by_episode = {integer(row["episode"]): row for row in audit_rows}
    for cluster_id, group in clusters.items():
        targets = np.vstack([episode.target for episode in group])
        labels = Counter(by_episode[episode.index + 1]["classification"] for episode in group)
        cluster_rows.append(
            {
                "cluster": cluster_id,
                "episodes": len(group),
                "first_reject_frame": min(episode.frame for episode in group),
                "last_reject_frame": max(episode.frame for episode in group),
                "centroid_x_m": float(np.mean(targets[:, 0])),
                "centroid_y_m": float(np.mean(targets[:, 1])),
                "centroid_z_m": float(np.mean(targets[:, 2])),
                "extent_x_m": float(np.ptp(targets[:, 0])),
                "extent_y_m": float(np.ptp(targets[:, 1])),
                "extent_z_m": float(np.ptp(targets[:, 2])),
                "contains_reopen": int(any(episode.reopened for episode in group)),
                "controlled_anchor": int(cluster_id in controlled_ids),
                "classification_counts": json.dumps(labels, ensure_ascii=False, sort_keys=True),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "reject_audit.csv", audit_rows)
    write_csv(args.output / "reject_clusters.csv", cluster_rows)
    summary = {
        "schema": "scancover.probe_shadow_reject_audit.v1",
        "session": session.name,
        "authority": "offline_read_only",
        "episodeCount": len(episodes),
        "clusterCount": len(clusters),
        "controlledAnchorClusters": sorted(controlled_ids),
        "classificationCounts": dict(Counter(row["classification"] for row in audit_rows)),
        "witnessFramesRead": sum(integer(row["witness_rows"]) for row in audit_rows),
        "witnessErrors": witness_errors,
        "thresholds": {
            "clusterRadiusMetres": CLUSTER_RADIUS_METRES,
            "clusterFrameGap": CLUSTER_FRAME_GAP,
            "reliableFreeGapMm": RELIABLE_GAP_MM,
            "edgeSpanMm": EDGE_SPAN_MM,
            "sourceReprojectionMm": SOURCE_REPROJECTION_MM,
        },
        "interpretationBoundary": (
            "The reopened component is a controlled positive anchor. Other labels diagnose "
            "recorded evidence consistency; they do not independently prove physical truth."
        ),
        "files": {
            "episodes": "reject_audit.csv",
            "clusters": "reject_clusters.csv",
        },
    }
    (args.output / "reject_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not witness_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
