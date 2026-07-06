#!/usr/bin/env python3
"""Assign a growth order to ScanCover room raw coverage.

The default mode is timestamp-constrained: captured sessions define when shell
patches appear. Spatial adjacency does not reorder them; it only records which
patches connect, start a new island, or bridge islands later. The older
adjacency-grown mode is kept as a diagnostic.

Outputs:

- growth_order_cloud.ply: all accepted points colored by growth step, with
  scalar fields for growth_step, island_id, component_id, and source session.
- growth_steps/: cumulative point clouds for inspecting the shell growth.
- growth_order_report.json: component/island stats and parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from ScanCoverRoomRawCoverageMetaOverlay import normalize


DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1]
    / "ScanCoverExports"
    / "RepeatCoverageSessions"
    / "new data"
)


@dataclass(frozen=True)
class GrowthInput:
    session: Path
    csv_path: Path
    session_index: int


@dataclass
class Component:
    component_id: int
    indices: np.ndarray
    size: int
    first_session: int
    last_session: int
    centroid: np.ndarray
    normal: np.ndarray
    island_id: int = -1
    initial_island_id: int = -1
    growth_step: int = -1
    attach_distance: float = math.inf
    attached_to_component: int = -1
    bridge_components: list[int] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build spatial growth order from room_raw_coverage_voxels.csv files.")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Folder containing ScanCover_RepeatCoverage_* sessions, one session, room_raw_coverage, or a voxels CSV.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output folder. Default: <input>/coverage_growth_order")
    parser.add_argument("--neighbor-radius", type=float, default=0.13, help="Radius for building local components.")
    parser.add_argument("--attach-radius", type=float, default=0.20, help="Max distance for a patch to grow from current shell.")
    parser.add_argument("--normal-angle-deg", type=float, default=70.0, help="Max normal angle for component connectivity.")
    parser.add_argument("--growth-mode", choices=["timestamp", "adjacency"], default="timestamp")
    parser.add_argument("--component-scope", choices=["session", "global"], default="session", help="Build patches per session or over all points.")
    parser.add_argument("--min-point-hits", type=int, default=8)
    parser.add_argument("--min-frame-hits", type=int, default=1)
    parser.add_argument("--min-component-points", type=int, default=18)
    parser.add_argument("--seed-min-component-points", type=int, default=30)
    parser.add_argument("--include-risk", action="store_true", help="Include risk voxels. Default rejects them.")
    parser.add_argument("--dedupe-voxels", action=argparse.BooleanOptionalAction, default=True, help="Merge repeated history voxels across sessions.")
    parser.add_argument("--start-new-islands", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-step-files", type=int, default=80, help="Limit cumulative growth_step_*.ply files.")
    parser.add_argument("--write-animation", action="store_true", help="Write growth_preview.mp4 using a dependency-free rasterizer.")
    parser.add_argument("--animation-fps", type=int, default=2)
    parser.add_argument("--animation-size", type=int, default=1080, help="Square animation frame size in pixels.")
    parser.add_argument("--animation-point-size", type=int, default=2)
    parser.add_argument("--animation-view", choices=["iso", "top"], default="iso")
    parser.add_argument("--keep-animation-frames", action="store_true")
    parser.add_argument("--ascii", action="store_true", help="Write ASCII PLY files for easier inspection.")
    return parser.parse_args()


def resolve_inputs(path: Path) -> list[GrowthInput]:
    path = path.expanduser().resolve()
    if path.is_file():
        return [GrowthInput(path.parent.parent, path, 0)]
    if path.name == "room_raw_coverage" and (path / "room_raw_coverage_voxels.csv").exists():
        return [GrowthInput(path.parent, path / "room_raw_coverage_voxels.csv", 0)]
    if (path / "room_raw_coverage" / "room_raw_coverage_voxels.csv").exists():
        return [GrowthInput(path, path / "room_raw_coverage" / "room_raw_coverage_voxels.csv", 0)]

    sessions = sorted(
        [p for p in path.iterdir() if p.is_dir() and (p / "room_raw_coverage" / "room_raw_coverage_voxels.csv").exists()],
        key=lambda p: p.name.lower(),
    )
    if not sessions:
        sessions = sorted({p.parent.parent for p in path.rglob("room_raw_coverage_voxels.csv")}, key=lambda p: str(p).lower())
    return [
        GrowthInput(session, session / "room_raw_coverage" / "room_raw_coverage_voxels.csv", i)
        for i, session in enumerate(sessions)
    ]


def default_output_dir(input_path: Path) -> Path:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        return input_path.parent / "coverage_growth_order"
    if input_path.name == "room_raw_coverage":
        return input_path.parent / "coverage_growth_order"
    return input_path / "coverage_growth_order"


def read_coverage_csv(csv_path: Path, session_index: int) -> dict[str, np.ndarray]:
    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.startswith("voxelX,")), None)
    if header_index is None:
        raise RuntimeError(f"Could not find voxel header in {csv_path}")

    points: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    frame_hits: list[int] = []
    point_hits: list[int] = []
    stable: list[bool] = []
    risk: list[bool] = []
    source_sessions: list[int] = []
    voxel_keys: list[tuple[int, int, int]] = []

    for row in csv.DictReader(lines[header_index:]):
        voxel_keys.append((int(row["voxelX"]), int(row["voxelY"]), int(row["voxelZ"])))
        points.append((float(row["avgX"]), float(row["avgY"]), float(row["avgZ"])))
        normals.append((float(row["avgNormalX"]), float(row["avgNormalY"]), float(row["avgNormalZ"])))
        frame_hits.append(int(row["frameHits"]))
        point_hits.append(int(row["pointHits"]))
        stable.append(row["stable"].strip() == "1")
        risk.append(row["risk"].strip() == "1")
        source_sessions.append(session_index)

    return {
        "points": np.asarray(points, dtype=np.float64),
        "normals": normalize(np.asarray(normals, dtype=np.float64)),
        "frame_hits": np.asarray(frame_hits, dtype=np.int32),
        "point_hits": np.asarray(point_hits, dtype=np.int32),
        "stable": np.asarray(stable, dtype=bool),
        "risk": np.asarray(risk, dtype=bool),
        "source_sessions": np.asarray(source_sessions, dtype=np.int32),
        "voxel_keys": np.asarray(voxel_keys, dtype=np.int32),
    }


def load_all(inputs: list[GrowthInput]) -> dict[str, np.ndarray]:
    chunks = [read_coverage_csv(item.csv_path, item.session_index) for item in inputs]
    if not chunks:
        raise FileNotFoundError("No room_raw_coverage_voxels.csv files found.")
    return {
        key: np.concatenate([chunk[key] for chunk in chunks], axis=0)
        for key in chunks[0].keys()
    }


def dedupe_voxels(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    voxel_keys = raw["voxel_keys"]
    groups: dict[tuple[int, int, int], list[int]] = {}
    for i, key in enumerate(voxel_keys):
        groups.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(i)

    if len(groups) == len(voxel_keys):
        return raw

    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    frame_hits: list[int] = []
    point_hits: list[int] = []
    stable: list[bool] = []
    risk: list[bool] = []
    source_sessions: list[int] = []
    deduped_keys: list[tuple[int, int, int]] = []

    for key, indices in groups.items():
        ids = np.asarray(indices, dtype=np.int64)
        weights = np.maximum(raw["point_hits"][ids].astype(np.float64), 1.0)
        point = np.average(raw["points"][ids], axis=0, weights=weights)
        normal = np.average(raw["normals"][ids], axis=0, weights=weights)
        normal_len = np.linalg.norm(normal)
        if normal_len > 1e-8:
            normal = normal / normal_len

        points.append(point)
        normals.append(normal)
        frame_hits.append(int(np.sum(raw["frame_hits"][ids])))
        point_hits.append(int(np.sum(raw["point_hits"][ids])))
        stable.append(bool(np.any(raw["stable"][ids])))
        risk.append(bool(np.all(raw["risk"][ids])))
        source_sessions.append(int(np.min(raw["source_sessions"][ids])))
        deduped_keys.append(key)

    return {
        "points": np.asarray(points, dtype=np.float64),
        "normals": normalize(np.asarray(normals, dtype=np.float64)),
        "frame_hits": np.asarray(frame_hits, dtype=np.int32),
        "point_hits": np.asarray(point_hits, dtype=np.int32),
        "stable": np.asarray(stable, dtype=bool),
        "risk": np.asarray(risk, dtype=bool),
        "source_sessions": np.asarray(source_sessions, dtype=np.int32),
        "voxel_keys": np.asarray(deduped_keys, dtype=np.int32),
    }


def make_tree(points: np.ndarray) -> o3d.geometry.KDTreeFlann:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.reshape((-1, 3)))
    return o3d.geometry.KDTreeFlann(cloud)


def connected_components(
    points: np.ndarray,
    normals: np.ndarray,
    usable: np.ndarray,
    radius: float,
    min_normal_dot: float,
    min_component_points: int,
    source_sessions: np.ndarray,
    component_scope: str,
) -> tuple[list[Component], np.ndarray]:
    labels = np.full((len(points),), -1, dtype=np.int32)
    tree = make_tree(points)
    components: list[Component] = []

    for seed in np.flatnonzero(usable):
        if labels[seed] >= 0:
            continue
        component_id = len(components)
        labels[seed] = component_id
        q: deque[int] = deque([int(seed)])
        ids: list[int] = []

        while q:
            i = q.popleft()
            ids.append(i)
            _, neigh, _ = tree.search_radius_vector_3d(points[i], radius)
            for j in neigh:
                j = int(j)
                if labels[j] >= 0 or not usable[j]:
                    continue
                if component_scope == "session" and source_sessions[j] != source_sessions[i]:
                    continue
                if abs(float(normals[i] @ normals[j])) < min_normal_dot:
                    continue
                labels[j] = component_id
                q.append(j)

        indices = np.asarray(ids, dtype=np.int64)
        if len(indices) < min_component_points:
            labels[indices] = -1
            continue

        # Re-pack labels so component ids stay dense after small components are dropped.
        component_id = len(components)
        labels[indices] = component_id
        component_points = points[indices]
        component_normals = normals[indices]
        normal = np.mean(component_normals, axis=0)
        normal_len = np.linalg.norm(normal)
        if normal_len > 1e-8:
            normal = normal / normal_len
        sessions = source_sessions[indices]
        components.append(
            Component(
                component_id=component_id,
                indices=indices,
                size=int(len(indices)),
                first_session=int(np.min(sessions)),
                last_session=int(np.max(sessions)),
                centroid=np.mean(component_points, axis=0),
                normal=normal,
            )
        )

    return components, labels


def choose_seed(components: list[Component], min_seed_size: int) -> int:
    if not components:
        return -1
    earliest = min(component.first_session for component in components)
    earliest_components = [c for c in components if c.first_session == earliest and c.size >= min_seed_size]
    if not earliest_components:
        earliest_components = [c for c in components if c.first_session == earliest]
    seed = max(earliest_components, key=lambda c: c.size)
    return seed.component_id


def nearest_component_distance(
    candidate: Component,
    points: np.ndarray,
    grown_points: np.ndarray,
    grown_component_by_point: np.ndarray,
    sample_limit: int = 160,
) -> tuple[float, int]:
    candidate_indices = candidate.indices
    if len(candidate_indices) > sample_limit:
        step = max(1, int(math.ceil(len(candidate_indices) / sample_limit)))
        candidate_indices = candidate_indices[::step][:sample_limit]

    tree = make_tree(grown_points)
    best_distance = math.inf
    best_component = -1
    for idx in candidate_indices:
        _, neigh, d2 = tree.search_knn_vector_3d(points[idx], 1)
        if not neigh:
            continue
        distance = math.sqrt(float(d2[0]))
        if distance < best_distance:
            best_distance = distance
            best_component = int(grown_component_by_point[int(neigh[0])])
    return best_distance, best_component


def assign_growth_order(
    components: list[Component],
    points: np.ndarray,
    seed_component_id: int,
    attach_radius: float,
    start_new_islands: bool,
) -> None:
    if seed_component_id < 0:
        return

    remaining = {c.component_id for c in components}
    step = 0
    island = 0

    def grow(component_id: int, distance: float, attached_to: int) -> None:
        nonlocal step
        component = components[component_id]
        component.growth_step = step
        component.island_id = island
        component.attach_distance = distance
        component.attached_to_component = attached_to
        remaining.remove(component_id)
        step += 1

    grow(seed_component_id, 0.0, -1)

    while remaining:
        grown_components = [c for c in components if c.growth_step >= 0 and c.island_id == island]
        grown_indices = np.concatenate([c.indices for c in grown_components])
        grown_points = points[grown_indices]
        grown_component_by_point = np.concatenate([
            np.full((len(c.indices),), c.component_id, dtype=np.int32)
            for c in grown_components
        ])

        best: tuple[float, int, int] | None = None
        for component_id in list(remaining):
            component = components[component_id]
            distance, attached_to = nearest_component_distance(component, points, grown_points, grown_component_by_point)
            if distance > attach_radius:
                continue
            score = (distance, -component.size, component.first_session)
            if best is None or score < (best[0], -components[best[1]].size, components[best[1]].first_session):
                best = (distance, component_id, attached_to)

        if best is not None:
            grow(best[1], best[0], best[2])
            continue

        if not start_new_islands:
            break

        island += 1
        next_component = max((components[i] for i in remaining), key=lambda c: (c.size, -c.first_session))
        grow(next_component.component_id, 0.0, -1)


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            parent = self.find(parent)
            self.parent[item] = parent
        return parent

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def assign_timestamp_order(
    components: list[Component],
    points: np.ndarray,
    attach_radius: float,
) -> None:
    if not components:
        return

    ordered = sorted(components, key=lambda c: (c.first_session, -c.size, c.component_id))
    dsu = DisjointSet(len(components))
    next_island = 0
    previous_components: list[Component] = []

    for component in ordered:
        component.growth_step = component.first_session
        component.bridge_components = []

        if not previous_components:
            component.initial_island_id = next_island
            component.island_id = next_island
            next_island += 1
            previous_components.append(component)
            continue

        previous_indices = np.concatenate([c.indices for c in previous_components])
        previous_points = points[previous_indices]
        previous_component_by_point = np.concatenate([
            np.full((len(c.indices),), c.component_id, dtype=np.int32)
            for c in previous_components
        ])
        distance, attached_to = nearest_component_distance(
            component,
            points,
            previous_points,
            previous_component_by_point,
        )
        component.attach_distance = distance
        component.attached_to_component = attached_to if distance <= attach_radius else -1

        nearby_components: set[int] = set()
        if distance <= attach_radius:
            tree = make_tree(previous_points)
            sample_indices = component.indices
            if len(sample_indices) > 220:
                step = max(1, int(math.ceil(len(sample_indices) / 220)))
                sample_indices = sample_indices[::step][:220]
            for idx in sample_indices:
                _, neigh, _ = tree.search_radius_vector_3d(points[idx], attach_radius)
                for near_index in neigh:
                    nearby_components.add(int(previous_component_by_point[int(near_index)]))

        if nearby_components:
            primary = min(nearby_components, key=lambda cid: components[cid].growth_step)
            component.initial_island_id = components[primary].initial_island_id
            for other in nearby_components:
                dsu.union(component.component_id, other)
            component.bridge_components = sorted(nearby_components)
        else:
            component.initial_island_id = next_island
            component.island_id = next_island
            next_island += 1

        previous_components.append(component)

    root_to_final: dict[int, int] = {}
    for component in components:
        root = dsu.find(component.component_id)
        if root not in root_to_final:
            root_to_final[root] = len(root_to_final)
        component.island_id = root_to_final[root]


def color_for_step(step: int, max_step: int, island: int) -> tuple[int, int, int]:
    if step < 0:
        return (80, 80, 80)
    if max_step <= 0:
        t = 0.0
    else:
        t = step / max_step
    # Blue/cyan -> green/yellow -> orange/red.
    stops = [
        (0.00, (0, 110, 255)),
        (0.25, (0, 230, 255)),
        (0.50, (70, 255, 80)),
        (0.75, (255, 220, 0)),
        (1.00, (255, 70, 30)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            color = tuple(int(round(c0[j] * (1.0 - a) + c1[j] * a)) for j in range(3))
            break
    else:
        color = stops[-1][1]
    if island > 0:
        color = tuple(int(round(channel * 0.72 + 60)) for channel in color)
    return color


def write_growth_ply(
    path: Path,
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    growth_steps: np.ndarray,
    island_ids: np.ndarray,
    component_ids: np.ndarray,
    source_sessions: np.ndarray,
    ascii_format: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if ascii_format else "wb"
    with path.open(mode) as fh:
        header = "\n".join(
            [
                "ply",
                "format ascii 1.0" if ascii_format else "format binary_little_endian 1.0",
                f"element vertex {len(points)}",
                "property float x",
                "property float y",
                "property float z",
                "property float nx",
                "property float ny",
                "property float nz",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "property int growth_step",
                "property int island_id",
                "property int component_id",
                "property int source_session",
                "end_header\n",
            ]
        )
        if ascii_format:
            fh.write(header)
            for i, point in enumerate(points):
                fh.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{normals[i,0]:.6f} {normals[i,1]:.6f} {normals[i,2]:.6f} "
                    f"{int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])} "
                    f"{int(growth_steps[i])} {int(island_ids[i])} {int(component_ids[i])} {int(source_sessions[i])}\n"
                )
        else:
            import struct

            fh.write(header.encode("ascii"))
            pack = struct.Struct("<ffffffBBBiiii").pack
            for i, point in enumerate(points):
                fh.write(
                    pack(
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                        float(normals[i, 0]),
                        float(normals[i, 1]),
                        float(normals[i, 2]),
                        int(colors[i, 0]),
                        int(colors[i, 1]),
                        int(colors[i, 2]),
                        int(growth_steps[i]),
                        int(island_ids[i]),
                        int(component_ids[i]),
                        int(source_sessions[i]),
                    )
                )


def build_point_fields(
    components: list[Component],
    labels: np.ndarray,
    point_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    growth_steps = np.full((point_count,), -1, dtype=np.int32)
    island_ids = np.full((point_count,), -1, dtype=np.int32)
    component_ids = labels.astype(np.int32, copy=True)
    for component in components:
        growth_steps[component.indices] = component.growth_step
        island_ids[component.indices] = component.island_id
    return growth_steps, island_ids, component_ids


def write_step_clouds(
    out_dir: Path,
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    growth_steps: np.ndarray,
    island_ids: np.ndarray,
    component_ids: np.ndarray,
    source_sessions: np.ndarray,
    max_step: int,
    max_files: int,
    ascii_format: bool,
) -> list[str]:
    steps_dir = out_dir / "growth_steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    if max_step < 0:
        return []
    if max_files <= 0:
        step_values = list(range(max_step + 1))
    else:
        stride = max(1, int(math.ceil((max_step + 1) / max_files)))
        step_values = list(range(0, max_step + 1, stride))
        if step_values[-1] != max_step:
            step_values.append(max_step)

    written: list[str] = []
    for step in step_values:
        mask = (growth_steps >= 0) & (growth_steps <= step)
        path = steps_dir / f"growth_step_{step:04d}.ply"
        write_growth_ply(
            path,
            points[mask],
            normals[mask],
            colors[mask],
            growth_steps[mask],
            island_ids[mask],
            component_ids[mask],
            source_sessions[mask],
            ascii_format,
        )
        written.append(str(path))
    return written


def project_points(points: np.ndarray, view: str, size: int, margin: int) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
    centered = points - np.mean(points, axis=0)
    if view == "top":
        u = centered[:, 0]
        v = -centered[:, 2]
    else:
        u = centered[:, 0] - centered[:, 2] * 0.55
        v = -centered[:, 1] + centered[:, 2] * 0.28
    min_u, max_u = float(np.min(u)), float(np.max(u))
    min_v, max_v = float(np.min(v)), float(np.max(v))
    span_u = max(max_u - min_u, 1e-6)
    span_v = max(max_v - min_v, 1e-6)
    scale = (size - margin * 2) / max(span_u, span_v)
    px = np.round((u - (min_u + max_u) * 0.5) * scale + size * 0.5).astype(np.int32)
    py = np.round((v - (min_v + max_v) * 0.5) * scale + size * 0.5).astype(np.int32)
    return px, py


def draw_square(image: np.ndarray, x: int, y: int, radius: int, color: tuple[int, int, int]) -> None:
    h, w, _ = image.shape
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    if x0 < x1 and y0 < y1:
        image[y0:y1, x0:x1, :] = color


def write_ppm(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = image.shape
    with path.open("wb") as fh:
        fh.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        fh.write(image.astype(np.uint8, copy=False).tobytes())


def write_animation(
    out_dir: Path,
    points: np.ndarray,
    colors: np.ndarray,
    growth_steps: np.ndarray,
    max_step: int,
    fps: int,
    size: int,
    point_radius: int,
    view: str,
    keep_frames: bool,
) -> str | None:
    if max_step < 0 or len(points) == 0:
        return None

    frames_dir = out_dir / "growth_animation_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    px, py = project_points(points, view, size, margin=max(42, size // 18))
    background = np.array((5, 39, 57), dtype=np.uint8)
    dim = np.array((28, 70, 86), dtype=np.uint8)
    highlight = (255, 255, 245)
    progress = (60, 220, 255)
    bar_bg = (18, 74, 94)

    ordered = np.argsort(growth_steps)
    for step in range(max_step + 1):
        image = np.zeros((size, size, 3), dtype=np.uint8)
        image[:, :, :] = background

        future_mask = growth_steps > step
        for i in np.flatnonzero(future_mask):
            draw_square(image, int(px[i]), int(py[i]), max(0, point_radius - 1), tuple(int(v) for v in dim))

        grown_ids = ordered[(growth_steps[ordered] >= 0) & (growth_steps[ordered] <= step)]
        for i in grown_ids:
            color = tuple(int(v) for v in colors[i])
            draw_square(image, int(px[i]), int(py[i]), point_radius, color)

        current_ids = np.flatnonzero(growth_steps == step)
        for i in current_ids:
            draw_square(image, int(px[i]), int(py[i]), point_radius + 1, highlight)

        bar_x0 = size // 16
        bar_x1 = size - bar_x0
        bar_y0 = size - size // 20
        bar_y1 = bar_y0 + max(6, size // 160)
        image[bar_y0:bar_y1, bar_x0:bar_x1, :] = bar_bg
        filled = int(bar_x0 + (bar_x1 - bar_x0) * ((step + 1) / (max_step + 1)))
        image[bar_y0:bar_y1, bar_x0:filled, :] = progress

        write_ppm(frames_dir / f"growth_{step:04d}.ppm", image)

    mp4_path = out_dir / "growth_preview.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(max(1, fps)),
        "-i",
        str(frames_dir / "growth_%04d.ppm"),
        "-vf",
        "format=yuv420p",
        "-movflags",
        "+faststart",
        str(mp4_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not keep_frames:
        for frame in frames_dir.glob("growth_*.ppm"):
            frame.unlink()
        try:
            frames_dir.rmdir()
        except OSError:
            pass
    return str(mp4_path)


def main() -> int:
    args = parse_args()
    inputs = resolve_inputs(args.input)
    if not inputs:
        raise FileNotFoundError(f"No room_raw_coverage_voxels.csv found under {args.input}")

    out_dir = args.out.expanduser().resolve() if args.out else default_output_dir(args.input)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_before_dedupe = load_all(inputs)
    raw = dedupe_voxels(raw_before_dedupe) if args.dedupe_voxels else raw_before_dedupe
    points = raw["points"]
    normals = raw["normals"]
    frame_hits = raw["frame_hits"]
    point_hits = raw["point_hits"]
    risk = raw["risk"]
    source_sessions = raw["source_sessions"]

    usable = (frame_hits >= args.min_frame_hits) & (point_hits >= args.min_point_hits)
    if not args.include_risk:
        usable &= ~risk

    min_dot = math.cos(math.radians(args.normal_angle_deg))
    components, labels = connected_components(
        points,
        normals,
        usable,
        args.neighbor_radius,
        min_dot,
        args.min_component_points,
        source_sessions,
        args.component_scope,
    )
    seed_id = choose_seed(components, args.seed_min_component_points)
    if args.growth_mode == "adjacency":
        assign_growth_order(components, points, seed_id, args.attach_radius, args.start_new_islands)
    else:
        assign_timestamp_order(components, points, args.attach_radius)

    accepted = labels >= 0
    growth_steps, island_ids, component_ids = build_point_fields(components, labels, len(points))
    max_step = int(np.max(growth_steps[accepted])) if np.any(accepted & (growth_steps >= 0)) else -1
    colors = np.asarray(
        [color_for_step(int(growth_steps[i]), max_step, int(island_ids[i])) for i in range(len(points))],
        dtype=np.uint8,
    )

    cloud_path = out_dir / "growth_order_cloud.ply"
    write_growth_ply(
        cloud_path,
        points[accepted],
        normals[accepted],
        colors[accepted],
        growth_steps[accepted],
        island_ids[accepted],
        component_ids[accepted],
        source_sessions[accepted],
        args.ascii,
    )
    step_files = write_step_clouds(
        out_dir,
        points[accepted],
        normals[accepted],
        colors[accepted],
        growth_steps[accepted],
        island_ids[accepted],
        component_ids[accepted],
        source_sessions[accepted],
        max_step,
        args.max_step_files,
        args.ascii,
    )
    animation_path = None
    if args.write_animation:
        animation_path = write_animation(
            out_dir,
            points[accepted],
            colors[accepted],
            growth_steps[accepted],
            max_step,
            args.animation_fps,
            args.animation_size,
            args.animation_point_size,
            args.animation_view,
            args.keep_animation_frames,
        )

    report = {
        "input": str(args.input),
        "output": str(out_dir),
        "sessions": [str(item.session) for item in inputs],
        "parameters": {
            "neighborRadius": args.neighbor_radius,
            "attachRadius": args.attach_radius,
            "normalAngleDeg": args.normal_angle_deg,
            "growthMode": args.growth_mode,
            "componentScope": args.component_scope,
            "minPointHits": args.min_point_hits,
            "minFrameHits": args.min_frame_hits,
            "minComponentPoints": args.min_component_points,
            "seedMinComponentPoints": args.seed_min_component_points,
            "includeRisk": args.include_risk,
            "dedupeVoxels": args.dedupe_voxels,
            "startNewIslands": args.start_new_islands,
            "writeAnimation": args.write_animation,
            "animationView": args.animation_view,
        },
        "totalRowsBeforeDedupe": int(len(raw_before_dedupe["points"])),
        "totalInputPoints": int(len(points)),
        "usablePoints": int(np.count_nonzero(usable)),
        "acceptedPoints": int(np.count_nonzero(accepted)),
        "rejectedSmallOrUnusablePoints": int(len(points) - np.count_nonzero(accepted)),
        "components": int(len(components)),
        "seedComponent": int(seed_id),
        "islands": int(max((c.island_id for c in components), default=-1) + 1),
        "initialIslands": int(max((c.initial_island_id for c in components), default=-1) + 1),
        "disconnectedStarts": int(sum(1 for c in components if c.attached_to_component < 0 and c.growth_step > min((x.growth_step for x in components), default=0))),
        "bridgeComponents": int(sum(1 for c in components if c.bridge_components and len(c.bridge_components) > 1)),
        "maxGrowthStep": int(max_step),
        "growthOrderCloud": str(cloud_path),
        "growthStepFiles": step_files,
        "growthPreviewMp4": animation_path,
        "componentStats": [
            {
                "componentId": int(c.component_id),
                "growthStep": int(c.growth_step),
                "islandId": int(c.island_id),
                "initialIslandId": int(c.initial_island_id),
                "size": int(c.size),
                "firstSession": int(c.first_session),
                "lastSession": int(c.last_session),
                "attachDistance": None if not math.isfinite(c.attach_distance) else float(c.attach_distance),
                "attachedToComponent": int(c.attached_to_component),
                "bridgeComponents": [int(v) for v in (c.bridge_components or [])],
                "centroid": [float(v) for v in c.centroid],
            }
            for c in sorted(components, key=lambda item: (item.growth_step if item.growth_step >= 0 else 10**9, item.component_id))
        ],
    }
    (out_dir / "growth_order_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({k: report[k] for k in [
        "output",
        "totalInputPoints",
        "usablePoints",
        "acceptedPoints",
        "components",
        "seedComponent",
        "islands",
        "maxGrowthStep",
        "growthOrderCloud",
    ]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
