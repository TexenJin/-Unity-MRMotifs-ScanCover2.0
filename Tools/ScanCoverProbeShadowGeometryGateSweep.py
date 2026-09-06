#!/usr/bin/env python3
"""Counterfactual geometry-gate replay for probe-shadow Reject episodes.

This tool is deliberately offline and read-only.  It replays the already
recorded free-space challenge sequence with the production witness contract
(three votes, at least two pairwise-independent views), while replacing the
current 40 mm ray-to-centre association with a ray/candidate-plane intersection
inside the 50 mm candidate cell, optionally dilated by 0/5/10 mm.

The candidate normal comes from the session's final verdict_cells.csv.  That is
the best normal currently sealed in the session, but it is not guaranteed to be
the exact normal at Reject time; the output keeps this limitation explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ScanCoverProbeShadowRejectAudit import (
    CELL_METRES,
    assign_clusters,
    build_episodes,
    cell_key,
    integer,
    matrix,
    number,
    read_csv,
    recorded_eye,
    xyz,
)


INDEPENDENT_BASELINE_METRES = 0.08
INDEPENDENT_ANGLE_DEG = 3.0
INDEPENDENT_FRAME_GAP = 2
CHALLENGE_WITNESS_CAPACITY = 4
REQUIRED_CHALLENGE_VOTES = 3
REQUIRED_CHALLENGE_VIEWS = 2
DEFAULT_MARGINS_MM = (0.0, 5.0, 10.0)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


class MetadataStore:
    def __init__(self, session: Path) -> None:
        self.frames = session / "depth_pairs" / "frames"
        manifest = read_csv(session / "depth_pairs" / "manifest.csv")
        self.by_pair = {integer(row.get("pairIndex")): row for row in manifest}
        self.by_platform = {integer(row.get("platformFrame")): row for row in manifest}
        self.cache: dict[int, dict] = {}

    def resolve_pair(self, event: dict[str, str]) -> int:
        direct = integer(event.get("gunGelFrame"))
        platform = integer(event.get("platformFrame"))
        if direct in self.by_pair and integer(self.by_pair[direct].get("platformFrame")) == platform:
            return direct
        row = self.by_platform.get(platform)
        if row is None:
            raise KeyError(f"no depth-pair metadata for gunGel={direct} platform={platform}")
        return integer(row.get("pairIndex"))

    def load(self, event: dict[str, str]) -> tuple[int, dict]:
        pair = self.resolve_pair(event)
        if pair not in self.cache:
            path = self.frames / f"frame_{pair:06d}_meta.json"
            self.cache[pair] = json.loads(path.read_text(encoding="utf-8"))
        return pair, self.cache[pair]


@dataclass
class GeometryWitness:
    frame: int
    camera: np.ndarray
    direction: np.ndarray
    normal_valid: bool
    plane_before_depth_hit: bool
    required_margin_mm: float


def geometry_witness(
    store: MetadataStore,
    row: dict[str, str],
    key: tuple[int, int, int, int],
    candidate_normal: np.ndarray,
) -> GeometryWitness:
    _, meta = store.load(row)
    eye = recorded_eye(meta)
    view_inverse = matrix(meta["viewInverse"][eye])
    camera = view_inverse[:3, 3]
    source = xyz(row, "source")
    target = xyz(row, "target")
    ray = source - camera
    hit_range = float(np.linalg.norm(ray))
    ray_direction = ray / max(hit_range, 1e-12)
    view_direction = camera - target
    view_direction /= max(float(np.linalg.norm(view_direction)), 1e-12)

    normal_length = float(np.linalg.norm(candidate_normal))
    if normal_length <= 1e-8 or hit_range <= 1e-8:
        return GeometryWitness(
            integer(row.get("gunGelFrame")), camera, view_direction, False, False, math.inf
        )

    normal = candidate_normal / normal_length
    denominator = float(np.dot(ray_direction, normal))
    if abs(denominator) <= 1e-5:
        return GeometryWitness(
            integer(row.get("gunGelFrame")), camera, view_direction, True, False, math.inf
        )

    intersection_range = float(np.dot(target - camera, normal) / denominator)
    before_hit = 0.0 < intersection_range < hit_range
    if not before_hit:
        required_margin_mm = math.inf
    else:
        point = camera + ray_direction * intersection_range
        cell_min = np.asarray(key[:3], dtype=np.float64) * CELL_METRES
        cell_max = cell_min + CELL_METRES
        below = np.maximum(cell_min - point, 0.0)
        above = np.maximum(point - cell_max, 0.0)
        required_margin_mm = float(np.max(np.maximum(below, above)) * 1000.0)
    return GeometryWitness(
        integer(row.get("gunGelFrame")),
        camera,
        view_direction,
        True,
        before_hit,
        required_margin_mm,
    )


def add_independent(witnesses: list[GeometryWitness], candidate: GeometryWitness) -> bool:
    if not witnesses:
        witnesses.append(candidate)
        return True
    independent = True
    for existing in witnesses:
        baseline = float(np.linalg.norm(existing.camera - candidate.camera))
        spread = angle_deg(existing.direction, candidate.direction)
        time_separated = candidate.frame - existing.frame >= INDEPENDENT_FRAME_GAP
        pose_separated = (
            baseline >= INDEPENDENT_BASELINE_METRES or spread >= INDEPENDENT_ANGLE_DEG
        )
        if not time_separated or not pose_separated:
            independent = False
    if not independent or len(witnesses) >= CHALLENGE_WITNESS_CAPACITY:
        return False
    witnesses.append(candidate)
    return True


def replay(
    witnesses: list[GeometryWitness],
    qualifies: Callable[[GeometryWitness], bool],
) -> dict[str, object]:
    votes = 0
    independent: list[GeometryWitness] = []
    reject_frame: Optional[int] = None
    qualified = 0
    for witness in witnesses:
        if not qualifies(witness):
            continue
        qualified += 1
        votes += 1
        add_independent(independent, witness)
        if votes >= REQUIRED_CHALLENGE_VOTES and len(independent) >= REQUIRED_CHALLENGE_VIEWS:
            reject_frame = witness.frame
            break
    return {
        "would_reject": int(reject_frame is not None),
        "reject_frame": "" if reject_frame is None else reject_frame,
        "qualified_witnesses_before_decision": qualified,
        "independent_views_before_decision": len(independent),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="Local immutable replay session root")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--margins-mm",
        type=float,
        nargs="*",
        default=list(DEFAULT_MARGINS_MM),
        help="Candidate-cell dilation margins in millimetres (default: 0 5 10)",
    )
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
    reopened_episodes = {episode.index for episode in episodes if episode.reopened}

    cell_rows = read_csv(session / "probe_shadow" / "verdict_cells.csv")
    cell_normals = {
        cell_key(row): np.asarray(
            [number(row.get("normalX")), number(row.get("normalY")), number(row.get("normalZ"))],
            dtype=np.float64,
        )
        for row in cell_rows
    }
    store = MetadataStore(session)
    geometry: dict[int, list[GeometryWitness]] = {}
    errors: list[str] = []
    for episode in episodes:
        normal = cell_normals.get(episode.key, np.zeros(3, dtype=np.float64))
        rows: list[GeometryWitness] = []
        for witness in episode.challenges:
            try:
                rows.append(geometry_witness(store, witness, episode.key, normal))
            except Exception as exc:
                errors.append(
                    f"episode={episode.index + 1} frame={witness.get('gunGelFrame')} error={exc}"
                )
        geometry[episode.index] = rows

    modes: list[tuple[str, Optional[float]]] = [("current_40mm_centerline", None)]
    modes.extend((f"plane_cell_plus_{margin:g}mm", margin) for margin in args.margins_mm)
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    controlled_episode_count = sum(
        episode.cluster_id in controlled_ids for episode in episodes
    )
    outside_episode_count = len(episodes) - controlled_episode_count

    for mode, margin in modes:
        retained_controlled = 0
        retained_outside = 0
        retained_reopened = 0
        reproduced = 0
        for episode in episodes:
            witness_rows = geometry[episode.index]
            if margin is None:
                result = replay(witness_rows, lambda _: True)
            else:
                result = replay(
                    witness_rows,
                    lambda item, limit=margin: (
                        item.normal_valid
                        and item.plane_before_depth_hit
                        and item.required_margin_mm <= limit + 1e-9
                    ),
                )
            would_reject = bool(result["would_reject"])
            reproduced += int(would_reject)
            controlled = episode.cluster_id in controlled_ids
            if controlled:
                retained_controlled += int(would_reject)
            else:
                retained_outside += int(would_reject)
            if episode.index in reopened_episodes:
                retained_reopened += int(would_reject)
            required_margins = [
                item.required_margin_mm
                for item in witness_rows
                if item.plane_before_depth_hit and math.isfinite(item.required_margin_mm)
            ]
            detail_rows.append(
                {
                    "mode": mode,
                    "margin_mm": "" if margin is None else margin,
                    "episode": episode.index + 1,
                    "cluster": episode.cluster_id,
                    "controlled_cluster": int(controlled),
                    "reopened_anchor": int(episode.index in reopened_episodes),
                    "cell": ":".join(str(value) for value in episode.key),
                    "original_reject_frame": episode.frame,
                    "original_witnesses": len(witness_rows),
                    "normal_available": int(any(item.normal_valid for item in witness_rows)),
                    "min_required_margin_mm": (
                        min(required_margins) if required_margins else ""
                    ),
                    "median_required_margin_mm": (
                        float(np.median(required_margins)) if required_margins else ""
                    ),
                    **result,
                }
            )
        summary_rows.append(
            {
                "mode": mode,
                "margin_mm": "" if margin is None else margin,
                "reproduced_rejects": reproduced,
                "suppressed_rejects": len(episodes) - reproduced,
                "controlled_retained": retained_controlled,
                "controlled_total": controlled_episode_count,
                "outside_retained": retained_outside,
                "outside_total": outside_episode_count,
                "reopened_anchor_retained": retained_reopened,
                "reopened_anchor_total": len(reopened_episodes),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "reject_geometry_gate_sweep.csv", detail_rows)
    write_csv(args.output / "reject_geometry_gate_summary.csv", summary_rows)
    summary = {
        "schema": "scancover.probe_shadow_geometry_gate_sweep.v1",
        "session": session.name,
        "authority": "offline_read_only",
        "episodeCount": len(episodes),
        "controlledAnchorClusters": sorted(controlled_ids),
        "modes": summary_rows,
        "errors": errors,
        "contract": {
            "challengeVotes": REQUIRED_CHALLENGE_VOTES,
            "challengeIndependentViews": REQUIRED_CHALLENGE_VIEWS,
            "independentBaselineMetres": INDEPENDENT_BASELINE_METRES,
            "independentAngleDeg": INDEPENDENT_ANGLE_DEG,
            "independentFrameGap": INDEPENDENT_FRAME_GAP,
            "challengeWitnessCapacity": CHALLENGE_WITNESS_CAPACITY,
        },
        "interpretationBoundary": (
            "The controlled component is identified by its observed Reopen. Candidate normals are "
            "the final sealed cell normals, not guaranteed Reject-time normals. A gate that cannot "
            "retain the reopened anchor is too strict for production; a gate that retains many "
            "outside Rejects has not demonstrated sufficient separation without physical truth."
        ),
        "files": {
            "episodes": "reject_geometry_gate_sweep.csv",
            "summaryCsv": "reject_geometry_gate_summary.csv",
        },
    }
    (args.output / "reject_geometry_gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
