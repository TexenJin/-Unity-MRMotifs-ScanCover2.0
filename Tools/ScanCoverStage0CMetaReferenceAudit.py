#!/usr/bin/env python3
"""Stage 0-C audit: compare ScanCover observations against welded Meta Scene Mesh.

The purpose is not to rebuild a mesh. It treats the welded Meta Scene Mesh as a
room-scale structural reference and audits BL surface exports / grid-node point
exports as local observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ScanCoverStage0MetaMeshWeldAndAudit import (  # type: ignore
    Vec3,
    aabb_overlap_ratio,
    bounds,
    center_delta_length,
    mesh_stats,
    nearest_neighbor_stats,
    parse_obj,
    sample_vertices,
    summarize_values,
    vec_cross,
    vec_len,
    vec_sub,
    weld_mesh,
    write_ply,
)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_grid_nodes_csv(path: Path) -> List[Vec3]:
    header: Optional[List[str]] = None
    points: List[Vec3] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" in line and not line.startswith("index,"):
                continue
            if header is None:
                candidate = [part.strip() for part in line.split(",")]
                if {"worldX", "worldY", "worldZ"}.issubset(set(candidate)):
                    header = candidate
                continue

            row = next(csv.reader([line]))
            if len(row) < len(header):
                continue
            rec = dict(zip(header, row))
            if rec.get("hasPosition", "1") not in ("1", "True", "true"):
                continue
            try:
                points.append((float(rec["worldX"]), float(rec["worldY"]), float(rec["worldZ"])))
            except (KeyError, ValueError):
                continue
    return points


def read_ply_vertices(path: Path, max_points: int = 120000) -> List[Vec3]:
    try:
        import numpy as np  # type: ignore
        import open3d as o3d  # type: ignore

        cloud = o3d.io.read_point_cloud(str(path))
        arr = np.asarray(cloud.points)
        if arr.size:
            if len(arr) > max_points:
                step = max(1, math.ceil(len(arr) / max_points))
                arr = arr[::step]
            return [(float(p[0]), float(p[1]), float(p[2])) for p in arr]
    except Exception:
        pass

    points: List[Vec3] = []
    vertex_count: Optional[int] = None
    header_done = False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not header_done:
                if line.startswith("format ") and "ascii" not in line:
                    return []
                if line.startswith("element vertex "):
                    try:
                        vertex_count = int(line.split()[-1])
                    except ValueError:
                        vertex_count = None
                if line == "end_header":
                    header_done = True
                continue
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                points.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
            if len(points) >= max_points:
                break
            if vertex_count is not None and len(points) >= vertex_count:
                break
    return points


def face_centroids(vertices: Sequence[Vec3], faces: Sequence[Tuple[int, int, int]], max_count: int = 40000) -> List[Vec3]:
    if not faces:
        return []
    step = max(1, math.ceil(len(faces) / max_count))
    centroids: List[Vec3] = []
    for i in range(0, len(faces), step):
        a, b, c = faces[i]
        va, vb, vc = vertices[a], vertices[b], vertices[c]
        centroids.append(((va[0] + vb[0] + vc[0]) / 3.0, (va[1] + vb[1] + vc[1]) / 3.0, (va[2] + vb[2] + vc[2]) / 3.0))
    return centroids


def point_cloud_nn_stats(src: Sequence[Vec3], dst: Sequence[Vec3], max_points: int = 20000) -> Optional[Dict[str, object]]:
    return nearest_neighbor_stats(src, dst, max_points=max_points)


def inlier_ratios(nn: Optional[Dict[str, object]], thresholds: Sequence[float]) -> Dict[str, Optional[float]]:
    if not nn:
        return {f"inlier_le_{t:.2f}m": None for t in thresholds}
    # Recompute exact list is not retained by nearest_neighbor_stats. Run a small local helper with numpy.
    return {f"inlier_le_{t:.2f}m": None for t in thresholds}


def nn_distances(src: Sequence[Vec3], dst: Sequence[Vec3], max_points: int = 20000) -> Optional[List[float]]:
    if not src or not dst:
        return None
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None
    src_sample = np.asarray(sample_vertices(src, max_points), dtype=np.float32)
    dst_sample = np.asarray(sample_vertices(dst, max_points), dtype=np.float32)
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(dst_sample)
        distances, _ = tree.query(src_sample, k=1, workers=-1)
        return distances.astype(float).tolist()
    except Exception:
        pass

    distances: List[float] = []
    chunk_size = 1024
    for start in range(0, len(src_sample), chunk_size):
        chunk = src_sample[start:start + chunk_size]
        diff = chunk[:, None, :] - dst_sample[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        distances.extend(np.sqrt(np.min(d2, axis=1)).astype(float).tolist())
    return distances


def distance_summary(src: Sequence[Vec3], dst: Sequence[Vec3], thresholds: Sequence[float]) -> Dict[str, object]:
    distances = nn_distances(src, dst)
    summary: Dict[str, object] = {"distance": summarize_values(distances or [])}
    for threshold in thresholds:
        key = f"inlier_le_{threshold:.2f}m"
        summary[key] = (sum(1 for d in distances if d <= threshold) / len(distances)) if distances else None
    summary["sample_count"] = len(distances) if distances else 0
    return summary


def obj_points_and_stats(path: Path, eps: float) -> Tuple[List[Vec3], Optional[Dict[str, object]]]:
    vertices, faces = parse_obj(path)
    welded_vertices, welded_faces, weld = weld_mesh(vertices, faces, eps)
    stats = {
        "weld": weld,
        "mesh": mesh_stats(welded_vertices, welded_faces),
    }
    return welded_vertices, stats


def find_inputs(exports_root: Path, max_bl: int, max_csv: int, max_ply: int) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    bl_files = sorted(exports_root.glob("ScanCover_BLSurfaceMesh_*.obj"), key=lambda p: p.stat().st_mtime, reverse=True)[:max_bl]
    csv_files = sorted(exports_root.glob("ScanCover_DepthGridNodes_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)[:max_csv]
    ply_candidates: List[Path] = []
    for session_dir in (exports_root / "ScanSessions").glob("ScanCover_MultiFrame_*"):
        analysis_dir = session_dir / "multi_frame_analysis"
        if not analysis_dir.exists():
            continue
        preferred = [
            "multi_frame_voxel_0.020_stable_min3.ply",
            "multi_frame_voxel_0.020_all.ply",
            "multi_frame_voxel_0.020_risk_single_frame.ply",
            "reclassified_all.ply",
        ]
        for name in preferred:
            path = analysis_dir / name
            if path.exists():
                ply_candidates.append(path)
        ply_candidates.extend(sorted(analysis_dir.glob("reclassified_family_*.ply"), key=lambda p: p.name)[:4])
    ply_files = sorted(set(ply_candidates), key=lambda p: p.stat().st_mtime, reverse=True)[:max_ply]
    for path in bl_files:
        items.append({"kind": "bl_mesh_obj", "path": path})
    for path in csv_files:
        items.append({"kind": "depth_grid_nodes_csv", "path": path})
    for path in ply_files:
        items.append({"kind": "multi_frame_ply", "path": path})
    return items


def write_point_cloud_ply(path: Path, points: Sequence[Vec3], color: Tuple[int, int, int]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        r, g, b = color
        for x, y, z in points:
            handle.write(f"{x:.9g} {y:.9g} {z:.9g} {r} {g} {b}\n")


def classify_points_by_meta_distance(points: Sequence[Vec3], meta_ref: Sequence[Vec3], thresholds: Tuple[float, float]) -> Dict[str, List[Vec3]]:
    distances = nn_distances(points, meta_ref, max_points=50000)
    if distances is None:
        return {"near": list(points), "mid": [], "far": []}
    sampled_points = sample_vertices(points, len(distances))
    near: List[Vec3] = []
    mid: List[Vec3] = []
    far: List[Vec3] = []
    for point, d in zip(sampled_points, distances):
        if d <= thresholds[0]:
            near.append(point)
        elif d <= thresholds[1]:
            mid.append(point)
        else:
            far.append(point)
    return {"near": near, "mid": mid, "far": far}


def write_report(report_path: Path, payload: Dict[str, object]) -> None:
    aggregate = payload["aggregate"]
    assert isinstance(aggregate, dict)
    lines: List[str] = []
    lines.append("# Stage 0-C Meta Reference Audit")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Use the welded Meta Scene Mesh as the room-scale structural reference, then measure how current ScanCover BL mesh and grid-node observations behave against that reference.")
    lines.append("")
    lines.append("## Reference")
    lines.append("")
    lines.append(f"- Welded Meta OBJ: `{payload['meta_welded_obj']}`")
    lines.append(f"- Meta vertices used for reference cloud: `{payload['meta_reference_point_count']}`")
    lines.append("")
    lines.append("## Batch Result")
    lines.append("")
    lines.append(f"- Files audited: `{aggregate['file_count']}`")
    lines.append(f"- BL mesh OBJ files: `{aggregate['bl_mesh_count']}`")
    lines.append(f"- Depth grid node CSV files: `{aggregate['grid_csv_count']}`")
    lines.append(f"- Multi-frame PLY files: `{aggregate['multi_frame_ply_count']}`")
    lines.append(f"- Median observation-to-Meta p50 distance: `{aggregate['median_obs_to_meta_p50_m']}` m")
    lines.append(f"- Median observation-to-Meta p95 distance: `{aggregate['median_obs_to_meta_p95_m']}` m")
    lines.append(f"- Median local inlier ratio <= 10cm: `{aggregate['median_inlier_le_0_10m']}`")
    lines.append(f"- Median global Meta coverage <= 20cm: `{aggregate['median_meta_coverage_le_0_20m']}`")
    lines.append("")
    lines.append("## Architecture Reading")
    lines.append("")
    for item in payload["architecture_reading"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Next Engineering Step")
    lines.append("")
    for item in payload["next_steps"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- Per-file CSV: `{payload['per_file_csv']}`")
    lines.append(f"- Summary JSON: `{payload['summary_json']}`")
    lines.append(f"- Diagnostic PLY folder: `{payload['diagnostic_ply_dir']}`")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return summarize_values(values)["p50"]


def as_float(value: object) -> Optional[float]:
    return value if isinstance(value, float) else None


def run(args: argparse.Namespace) -> None:
    exports_root = Path(args.exports_root)
    meta_session = Path(args.meta_session)
    meta_welded_obj = Path(args.meta_welded_obj) if args.meta_welded_obj else meta_session / "stage0_weld" / "meta_scene_mesh_aligned_all_welded_eps1e-05.obj"
    if not meta_welded_obj.exists():
        raise FileNotFoundError(f"Missing welded Meta OBJ: {meta_welded_obj}")

    out_dir = Path(args.output_dir) if args.output_dir else meta_session / "stage0c_meta_reference_audit" / f"run_{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = out_dir / "diagnostic_ply"
    ply_dir.mkdir(parents=True, exist_ok=True)

    meta_vertices, meta_faces = parse_obj(meta_welded_obj)
    meta_reference = list(meta_vertices) + face_centroids(meta_vertices, meta_faces)
    meta_bounds = bounds(meta_vertices)

    inputs = find_inputs(exports_root, args.max_bl, args.max_csv, args.max_ply)
    rows: List[Dict[str, object]] = []
    p50s: List[float] = []
    p95s: List[float] = []
    inlier10s: List[float] = []
    meta_cov20s: List[float] = []

    combined_near: List[Vec3] = []
    combined_mid: List[Vec3] = []
    combined_far: List[Vec3] = []

    for item in inputs:
        kind = str(item["kind"])
        path = Path(item["path"])
        mesh_detail: Optional[Dict[str, object]] = None
        if kind == "bl_mesh_obj":
            points, mesh_detail = obj_points_and_stats(path, args.eps)
        elif kind == "multi_frame_ply":
            points = read_ply_vertices(path, max_points=args.max_ply_points)
        else:
            points = read_grid_nodes_csv(path)

        obs_bounds = bounds(points)
        obs_to_meta = distance_summary(points, meta_reference, args.thresholds)
        meta_to_obs = distance_summary(meta_reference, points, (0.10, 0.20, 0.35)) if points else {"distance": summarize_values([]), "inlier_le_0.10m": None, "inlier_le_0.20m": None, "inlier_le_0.35m": None, "sample_count": 0}
        dist_info = obs_to_meta["distance"]
        meta_dist_info = meta_to_obs["distance"]
        assert isinstance(dist_info, dict)
        assert isinstance(meta_dist_info, dict)

        p50 = as_float(dist_info.get("p50"))
        p95 = as_float(dist_info.get("p95"))
        inlier10 = as_float(obs_to_meta.get("inlier_le_0.10m"))
        meta_cov20 = as_float(meta_to_obs.get("inlier_le_0.20m"))
        if p50 is not None:
            p50s.append(p50)
        if p95 is not None:
            p95s.append(p95)
        if inlier10 is not None:
            inlier10s.append(inlier10)
        if meta_cov20 is not None:
            meta_cov20s.append(meta_cov20)

        classes = classify_points_by_meta_distance(points, meta_reference, (0.10, 0.25))
        combined_near.extend(classes["near"][:5000])
        combined_mid.extend(classes["mid"][:5000])
        combined_far.extend(classes["far"][:5000])

        row = {
            "kind": kind,
            "file": str(path),
            "point_count": len(points),
            "obs_to_meta_p50_m": p50,
            "obs_to_meta_p95_m": p95,
            "obs_to_meta_max_m": dist_info.get("max"),
            "obs_inlier_le_0_05m": obs_to_meta.get("inlier_le_0.05m"),
            "obs_inlier_le_0_10m": obs_to_meta.get("inlier_le_0.10m"),
            "obs_inlier_le_0_20m": obs_to_meta.get("inlier_le_0.20m"),
            "meta_coverage_le_0_10m": meta_to_obs.get("inlier_le_0.10m"),
            "meta_coverage_le_0_20m": meta_to_obs.get("inlier_le_0.20m"),
            "meta_to_obs_p50_m": meta_dist_info.get("p50"),
            "aabb_overlap_min_volume": aabb_overlap_ratio(meta_bounds, obs_bounds),
            "bounds_center_delta_m": center_delta_length(meta_bounds, obs_bounds),
            "bounds_size": obs_bounds.get("size"),
        }
        if mesh_detail:
            mesh = mesh_detail["mesh"]
            assert isinstance(mesh, dict)
            row.update({
                "mesh_vertices": mesh["vertex_count"],
                "mesh_faces": mesh["face_count"],
                "mesh_components": mesh["connected_component_count"],
                "mesh_boundary_edges": mesh["boundary_edge_count"],
            })
        rows.append(row)

    write_point_cloud_ply(ply_dir / "stage0c_observations_near_meta_le_10cm.ply", combined_near, (0, 255, 255))
    write_point_cloud_ply(ply_dir / "stage0c_observations_mid_meta_10_25cm.ply", combined_mid, (255, 220, 0))
    write_point_cloud_ply(ply_dir / "stage0c_observations_far_meta_gt_25cm.ply", combined_far, (255, 0, 80))

    per_file_csv = out_dir / "stage0c_per_file_observation_audit.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with per_file_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    aggregate = {
        "file_count": len(rows),
        "bl_mesh_count": sum(1 for row in rows if row["kind"] == "bl_mesh_obj"),
        "grid_csv_count": sum(1 for row in rows if row["kind"] == "depth_grid_nodes_csv"),
        "multi_frame_ply_count": sum(1 for row in rows if row["kind"] == "multi_frame_ply"),
        "median_obs_to_meta_p50_m": median(p50s),
        "median_obs_to_meta_p95_m": median(p95s),
        "median_inlier_le_0_10m": median(inlier10s),
        "median_meta_coverage_le_0_20m": median(meta_cov20s),
    }

    architecture_reading = [
        "Welded Meta Scene Mesh is valid as a room-scale structural baseline: it is welded, connected, closed, and much broader than local BL observations.",
        "Current BL surface exports should be interpreted as local observations over the Meta reference, not as competing full-room reconstruction results.",
        "If observation-to-Meta distances are low but Meta coverage is low, the ScanCover issue is mostly coverage / sampling / update policy, not necessarily coordinate-system failure.",
        "If observation-to-Meta distances are high in repeated local exports, Stage 0 should focus on depth-to-world reconstruction and mesh export correctness before any learning/model layer.",
    ]
    next_steps = [
        "Use this report as the first architecture gate: Meta Scene Mesh becomes the reference BL candidate unless a later audit proves it is unavailable or semantically insufficient.",
        "Run the same audit on explicit RawDepth projected point snapshots and multi-frame sessions so raw depth, BL mesh, and point cloud can be compared against the same reference.",
        "Add cropped Meta-reference evaluation around each BL export if we need fair local coverage instead of full-room coverage.",
    ]

    payload: Dict[str, object] = {
        "meta_welded_obj": str(meta_welded_obj),
        "meta_reference_point_count": len(meta_reference),
        "meta_bounds": meta_bounds,
        "aggregate": aggregate,
        "rows": rows,
        "architecture_reading": architecture_reading,
        "next_steps": next_steps,
        "per_file_csv": str(per_file_csv),
        "summary_json": str(out_dir / "stage0c_summary.json"),
        "diagnostic_ply_dir": str(ply_dir),
    }
    summary_json = out_dir / "stage0c_summary.json"
    report_md = out_dir / "stage0c_architecture_report.md"
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_md, payload)

    print(json.dumps({
        "output_dir": str(out_dir),
        "report": str(report_md),
        "per_file_csv": str(per_file_csv),
        "summary_json": str(summary_json),
        "diagnostic_ply_dir": str(ply_dir),
        "aggregate": aggregate,
    }, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0-C Meta reference audit for ScanCover observations.")
    parser.add_argument("--exports-root", default=r"E:\PCAII\NEW-SCANCOVER\ScanCoverExports")
    parser.add_argument("--meta-session", default=r"E:\PCAII\NEW-SCANCOVER\ScanCoverExports\MetaSceneMeshAuditSessions\ScanCover_MetaSceneMeshAudit_20260611_180459_512")
    parser.add_argument("--meta-welded-obj", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-bl", type=int, default=24)
    parser.add_argument("--max-csv", type=int, default=24)
    parser.add_argument("--max-ply", type=int, default=12)
    parser.add_argument("--max-ply-points", type=int, default=120000)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
