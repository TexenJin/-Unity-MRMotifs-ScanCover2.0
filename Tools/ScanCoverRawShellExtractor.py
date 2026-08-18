#!/usr/bin/env python3
"""Extract Raw-only room-shell candidates from ScanCover coverage voxels.

This tool intentionally does not use Meta Scene Mesh. It reads
room_raw_coverage_voxels.csv and separates stable Raw coverage into structural
layers:

- raw_shell_candidates.ply: large connected, locally planar Raw surfaces.
- raw_object_surfaces.ply: stable local surfaces that look real but are not a
  room-scale shell component.
- raw_boundary_edges.ply: stable/candidate points near high curvature or weak
  planar support.
- raw_sparse_tail.ply: observed points with too little neighborhood support.
- raw_reject_noise.ply: risky or very weak observations.
- raw_shell_all_by_class.ply: all input voxels colored by the class above.

The output is a Raw self-supervised structural diagnostic: "does the capture
itself contain enough room-shell evidence?" It is not a Meta-constrained
alignment result.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import open3d as o3d

from ScanCoverRoomRawCoverageMetaOverlay import distribution, read_room_voxels, write_cloud


CLASS_COLORS = {
    "shell": (0.05, 1.00, 0.20),
    "object": (0.00, 0.88, 1.00),
    "boundary": (1.00, 0.78, 0.05),
    "sparse": (0.10, 0.35, 1.00),
    "reject": (1.00, 0.05, 0.05),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw-only shell extractor for ScanCover coverage voxels.")
    parser.add_argument("input", type=Path, help="RepeatCoverage session folder, room_raw_coverage folder, or room_raw_coverage_voxels.csv.")
    parser.add_argument("--out", type=Path, default=None, help="Output folder. Default: <session>/raw_shell_extractor")
    parser.add_argument("--neighbor-radius", type=float, default=0.18)
    parser.add_argument("--component-radius", type=float, default=0.13)
    parser.add_argument("--min-neighbors", type=int, default=7)
    parser.add_argument("--min-shell-component", type=int, default=220)
    parser.add_argument("--min-object-component", type=int, default=35)
    parser.add_argument("--normal-angle-deg", type=float, default=35.0)
    parser.add_argument("--max-plane-std", type=float, default=0.035)
    parser.add_argument("--min-planarity", type=float, default=0.35)
    parser.add_argument("--candidate-min-frame-hits", type=int, default=2)
    parser.add_argument("--candidate-min-point-hits", type=int, default=12)
    parser.add_argument("--reject-min-point-hits", type=int, default=4)
    return parser.parse_args()


def resolve_voxels_csv(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.name != "room_raw_coverage_voxels.csv":
            raise FileNotFoundError(f"Expected room_raw_coverage_voxels.csv, got {path}")
        return path

    candidates = [
        path / "room_raw_coverage_voxels.csv",
        path / "room_raw_coverage" / "room_raw_coverage_voxels.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    matches = sorted(path.rglob("room_raw_coverage_voxels.csv"), key=lambda p: p.stat().st_mtime)
    if matches:
        return matches[-1].resolve()
    raise FileNotFoundError(f"No room_raw_coverage_voxels.csv found under {path}")


def default_output_dir(input_path: Path, csv_path: Path) -> Path:
    if input_path.is_file():
        return csv_path.parent / "raw_shell_extractor"
    if input_path.name == "room_raw_coverage":
        return input_path.parent / "raw_shell_extractor"
    return input_path / "raw_shell_extractor"


def build_kdtree(points: np.ndarray) -> o3d.geometry.KDTreeFlann:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return o3d.geometry.KDTreeFlann(cloud)


def local_geometry_scores(
    points: np.ndarray,
    normals: np.ndarray,
    usable: np.ndarray,
    radius: float,
    min_neighbors: int,
) -> dict[str, np.ndarray]:
    tree = build_kdtree(points)
    n = len(points)
    neighbor_count = np.zeros((n,), dtype=np.int32)
    normal_agreement = np.zeros((n,), dtype=np.float64)
    plane_std = np.full((n,), np.inf, dtype=np.float64)
    planarity = np.zeros((n,), dtype=np.float64)

    for i, point in enumerate(points):
        _, idx, _ = tree.search_radius_vector_3d(point, radius)
        ids = np.asarray(idx, dtype=np.int64)
        ids = ids[usable[ids]]
        neighbor_count[i] = int(len(ids))
        if len(ids) < max(3, min_neighbors):
            continue

        center = np.mean(points[ids], axis=0)
        centered = points[ids] - center
        cov = centered.T @ centered / max(1, len(ids))
        eig = np.linalg.eigvalsh(cov)
        eig = np.sort(np.maximum(eig, 0.0))
        total = float(np.sum(eig))
        if total > 1e-10:
            planarity[i] = float((eig[1] - eig[0]) / total)
        plane_std[i] = math.sqrt(float(eig[0]))

        dots = np.abs(normals[ids] @ normals[i])
        normal_agreement[i] = float(np.mean(dots))

    return {
        "neighborCount": neighbor_count,
        "normalAgreement": normal_agreement,
        "planeStd": plane_std,
        "planarity": planarity,
    }


def connected_components(
    points: np.ndarray,
    normals: np.ndarray,
    mask: np.ndarray,
    radius: float,
    normal_dot: float,
) -> tuple[np.ndarray, list[int]]:
    labels = np.full((len(points),), -1, dtype=np.int32)
    component_sizes: list[int] = []
    tree = build_kdtree(points)
    component_id = 0

    for seed in np.flatnonzero(mask):
        if labels[seed] >= 0:
            continue
        labels[seed] = component_id
        q: deque[int] = deque([int(seed)])
        size = 0
        while q:
            i = q.popleft()
            size += 1
            _, idx, _ = tree.search_radius_vector_3d(points[i], radius)
            for j in idx:
                j = int(j)
                if labels[j] >= 0 or not mask[j]:
                    continue
                if abs(float(normals[i] @ normals[j])) < normal_dot:
                    continue
                labels[j] = component_id
                q.append(j)
        component_sizes.append(size)
        component_id += 1

    return labels, component_sizes


def color_by_class(classes: np.ndarray) -> np.ndarray:
    colors = np.zeros((len(classes), 3), dtype=np.float64)
    for name, color in CLASS_COLORS.items():
        colors[classes == name] = color
    return colors


def main() -> int:
    args = parse_args()
    csv_path = resolve_voxels_csv(args.input)
    out_dir = args.out.expanduser().resolve() if args.out else default_output_dir(args.input.expanduser().resolve(), csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_room_voxels(csv_path)
    points = raw["points"]
    normals = raw["normals"]
    stable = raw["stable"]
    risk = raw["risk"]
    frame_hits = raw["frame_hits"]
    point_hits = raw["point_hits"]

    usable = stable | ((~risk) & (frame_hits >= args.candidate_min_frame_hits) & (point_hits >= args.candidate_min_point_hits))
    reject = risk | (point_hits < args.reject_min_point_hits)
    usable &= ~reject

    scores = local_geometry_scores(points, normals, usable, args.neighbor_radius, args.min_neighbors)
    normal_dot = math.cos(math.radians(args.normal_angle_deg))
    plane_like = (
        usable
        & (scores["neighborCount"] >= args.min_neighbors)
        & (scores["normalAgreement"] >= normal_dot)
        & (scores["planeStd"] <= args.max_plane_std)
        & (scores["planarity"] >= args.min_planarity)
    )

    labels, component_sizes = connected_components(points, normals, plane_like, args.component_radius, normal_dot)
    component_sizes_arr = np.asarray(component_sizes, dtype=np.int32)
    component_size_by_point = np.zeros((len(points),), dtype=np.int32)
    valid_labels = labels >= 0
    if len(component_sizes_arr) > 0:
        component_size_by_point[valid_labels] = component_sizes_arr[labels[valid_labels]]

    classes = np.full((len(points),), "sparse", dtype=object)
    classes[reject] = "reject"
    shell = plane_like & (component_size_by_point >= args.min_shell_component)
    object_surface = plane_like & ~shell & (component_size_by_point >= args.min_object_component)
    boundary = usable & ~shell & ~object_surface & (
        (scores["neighborCount"] >= args.min_neighbors)
        | ((scores["normalAgreement"] < normal_dot) & (scores["neighborCount"] >= max(3, args.min_neighbors // 2)))
    )
    sparse = ~(reject | shell | object_surface | boundary)

    classes[shell] = "shell"
    classes[object_surface] = "object"
    classes[boundary] = "boundary"
    classes[sparse] = "sparse"

    colors = color_by_class(classes)
    write_cloud(out_dir / "raw_shell_all_by_class.ply", points, colors, normals)
    write_cloud(out_dir / "raw_shell_candidates.ply", points[shell], np.tile(CLASS_COLORS["shell"], (np.count_nonzero(shell), 1)), normals[shell])
    write_cloud(out_dir / "raw_object_surfaces.ply", points[object_surface], np.tile(CLASS_COLORS["object"], (np.count_nonzero(object_surface), 1)), normals[object_surface])
    write_cloud(out_dir / "raw_boundary_edges.ply", points[boundary], np.tile(CLASS_COLORS["boundary"], (np.count_nonzero(boundary), 1)), normals[boundary])
    write_cloud(out_dir / "raw_sparse_tail.ply", points[sparse], np.tile(CLASS_COLORS["sparse"], (np.count_nonzero(sparse), 1)), normals[sparse])
    write_cloud(out_dir / "raw_reject_noise.ply", points[reject], np.tile(CLASS_COLORS["reject"], (np.count_nonzero(reject), 1)), normals[reject])

    top_components = sorted((int(v) for v in component_sizes), reverse=True)[:20]
    report = {
        "input": str(csv_path),
        "outputDirectory": str(out_dir),
        "meaning": {
            "raw_shell_candidates.ply": "Green: large connected locally planar Raw surfaces. These are Raw-only room-shell candidates.",
            "raw_object_surfaces.ply": "Cyan: stable local surfaces that look real but are too small to be room shell.",
            "raw_boundary_edges.ply": "Yellow: likely boundaries, occlusion edges, or high-curvature transitions.",
            "raw_sparse_tail.ply": "Blue: observed but too sparse/weak for structure.",
            "raw_reject_noise.ply": "Red: risk or very weak observations.",
            "raw_shell_all_by_class.ply": "All coverage voxels colored by the classes above.",
        },
        "parameters": {
            "neighborRadius": args.neighbor_radius,
            "componentRadius": args.component_radius,
            "minNeighbors": args.min_neighbors,
            "minShellComponent": args.min_shell_component,
            "minObjectComponent": args.min_object_component,
            "normalAngleDeg": args.normal_angle_deg,
            "maxPlaneStd": args.max_plane_std,
            "minPlanarity": args.min_planarity,
            "candidateMinFrameHits": args.candidate_min_frame_hits,
            "candidateMinPointHits": args.candidate_min_point_hits,
            "rejectMinPointHits": args.reject_min_point_hits,
        },
        "counts": {
            "total": int(len(points)),
            "stable": int(np.count_nonzero(stable)),
            "risk": int(np.count_nonzero(risk)),
            "usable": int(np.count_nonzero(usable)),
            "planeLike": int(np.count_nonzero(plane_like)),
            "shell": int(np.count_nonzero(shell)),
            "object": int(np.count_nonzero(object_surface)),
            "boundary": int(np.count_nonzero(boundary)),
            "sparse": int(np.count_nonzero(sparse)),
            "reject": int(np.count_nonzero(reject)),
        },
        "ratios": {
            "shell": float(np.count_nonzero(shell) / max(1, len(points))),
            "object": float(np.count_nonzero(object_surface) / max(1, len(points))),
            "boundary": float(np.count_nonzero(boundary) / max(1, len(points))),
            "sparse": float(np.count_nonzero(sparse) / max(1, len(points))),
            "reject": float(np.count_nonzero(reject) / max(1, len(points))),
        },
        "components": {
            "planeLikeComponentCount": int(len(component_sizes)),
            "topPlaneLikeComponentSizes": top_components,
            "shellComponentMinSize": int(args.min_shell_component),
        },
        "distributions": {
            "neighborCount": distribution(scores["neighborCount"]),
            "normalAgreement": distribution(scores["normalAgreement"][usable]),
            "planeStdMeters": distribution(scores["planeStd"][usable & np.isfinite(scores["planeStd"])]),
            "planarity": distribution(scores["planarity"][usable]),
            "shellNeighborCount": distribution(scores["neighborCount"][shell]),
            "shellPlaneStdMeters": distribution(scores["planeStd"][shell]),
            "objectPlaneStdMeters": distribution(scores["planeStd"][object_surface]),
        },
        "interpretation": [
            "This is Raw-only. Meta Scene Mesh and Replica Room are not used.",
            "Green shell candidates are large connected local planes, not guaranteed semantic room walls.",
            "Cyan object surfaces are stable and real-looking but should not be treated as room shell without another stage.",
            "Yellow boundary edges are useful for topology, but should not fill surfaces by themselves.",
        ],
    }
    (out_dir / "raw_shell_extractor_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"counts": report["counts"], "ratios": report["ratios"], "outputDirectory": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
