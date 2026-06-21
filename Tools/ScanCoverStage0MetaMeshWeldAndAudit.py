#!/usr/bin/env python3
"""Stage 0 utilities for Meta Scene Mesh welding and first-pass BL comparison.

This script intentionally avoids Open3D so it can run in a plain Python setup.
It welds Meta Scene Mesh triangle-soup vertices, writes mesh artifacts, and
produces a Stage 0-B reference audit against a BL surface OBJ when provided.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Vec3 = Tuple[float, float, float]
Face = Tuple[int, int, int]


def parse_obj(path: Path) -> Tuple[List[Vec3], List[Face]]:
    vertices: List[Vec3] = []
    faces: List[Face] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                parts = line.split()[1:]
                indices: List[int] = []
                for part in parts:
                    token = part.split("/")[0]
                    if not token:
                        continue
                    idx = int(token)
                    if idx < 0:
                        idx = len(vertices) + idx
                    else:
                        idx -= 1
                    indices.append(idx)
                if len(indices) >= 3:
                    base = indices[0]
                    for i in range(1, len(indices) - 1):
                        faces.append((base, indices[i], indices[i + 1]))
    return vertices, faces


def fmt_float(value: float) -> str:
    return f"{value:.9g}"


def vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vec_len(a: Vec3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def dist(a: Vec3, b: Vec3) -> float:
    return vec_len(vec_sub(a, b))


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def summarize_values(values: Sequence[float]) -> Dict[str, Optional[float]]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p05": percentile(values, 0.05),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def bounds(vertices: Sequence[Vec3]) -> Dict[str, object]:
    if not vertices:
        return {"min": None, "max": None, "center": None, "size": None, "diagonal": 0.0}
    mins = [min(v[i] for v in vertices) for i in range(3)]
    maxs = [max(v[i] for v in vertices) for i in range(3)]
    center = [(mins[i] + maxs[i]) * 0.5 for i in range(3)]
    size = [maxs[i] - mins[i] for i in range(3)]
    diagonal = math.sqrt(sum(s * s for s in size))
    return {"min": mins, "max": maxs, "center": center, "size": size, "diagonal": diagonal}


def weld_mesh(vertices: Sequence[Vec3], faces: Sequence[Face], eps: float) -> Tuple[List[Vec3], List[Face], Dict[str, int]]:
    key_to_new: Dict[Tuple[int, int, int], int] = {}
    old_to_new: List[int] = []
    welded_vertices: List[Vec3] = []

    inv = 1.0 / eps
    for vertex in vertices:
        key = (round(vertex[0] * inv), round(vertex[1] * inv), round(vertex[2] * inv))
        new_idx = key_to_new.get(key)
        if new_idx is None:
            new_idx = len(welded_vertices)
            key_to_new[key] = new_idx
            welded_vertices.append(vertex)
        old_to_new.append(new_idx)

    welded_faces: List[Face] = []
    seen_faces = set()
    degenerate = 0
    duplicate = 0
    for face in faces:
        a, b, c = old_to_new[face[0]], old_to_new[face[1]], old_to_new[face[2]]
        if a == b or b == c or c == a:
            degenerate += 1
            continue
        key = tuple(sorted((a, b, c)))
        if key in seen_faces:
            duplicate += 1
            continue
        seen_faces.add(key)
        welded_faces.append((a, b, c))

    return welded_vertices, welded_faces, {
        "raw_vertices": len(vertices),
        "raw_faces": len(faces),
        "welded_vertices": len(welded_vertices),
        "welded_faces": len(welded_faces),
        "degenerate_faces_removed": degenerate,
        "duplicate_faces_removed": duplicate,
    }


def edge_counts(faces: Sequence[Face]) -> Dict[Tuple[int, int], int]:
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for a, b, c in faces:
        for x, y in ((a, b), (b, c), (c, a)):
            if x > y:
                x, y = y, x
            counts[(x, y)] += 1
    return counts


def connected_components(vertex_count: int, faces: Sequence[Face]) -> List[int]:
    adjacency: List[List[int]] = [[] for _ in range(vertex_count)]
    for a, b, c in faces:
        adjacency[a].extend((b, c))
        adjacency[b].extend((a, c))
        adjacency[c].extend((a, b))

    visited = [False] * vertex_count
    sizes: List[int] = []
    for start in range(vertex_count):
        if visited[start] or not adjacency[start]:
            continue
        q: deque[int] = deque([start])
        visited[start] = True
        size = 0
        while q:
            node = q.popleft()
            size += 1
            for nxt in adjacency[node]:
                if not visited[nxt]:
                    visited[nxt] = True
                    q.append(nxt)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def mesh_stats(vertices: Sequence[Vec3], faces: Sequence[Face]) -> Dict[str, object]:
    edges = edge_counts(faces)
    edge_lengths = [dist(vertices[a], vertices[b]) for a, b in edges.keys()]
    areas = []
    for a, b, c in faces:
        ab = vec_sub(vertices[b], vertices[a])
        ac = vec_sub(vertices[c], vertices[a])
        areas.append(vec_len(vec_cross(ab, ac)) * 0.5)

    comps = connected_components(len(vertices), faces)
    boundary_edges = sum(1 for count in edges.values() if count == 1)
    nonmanifold_edges = sum(1 for count in edges.values() if count > 2)
    isolated_vertices = len(vertices) - sum(comps)
    return {
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "edge_count": len(edges),
        "boundary_edge_count": boundary_edges,
        "nonmanifold_edge_count": nonmanifold_edges,
        "connected_component_count": len(comps),
        "connected_component_sizes_top10": comps[:10],
        "isolated_vertex_count": isolated_vertices,
        "bounds": bounds(vertices),
        "edge_length": summarize_values(edge_lengths),
        "triangle_area": summarize_values(areas),
    }


def write_obj(path: Path, vertices: Sequence[Vec3], faces: Sequence[Face]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# ScanCover Stage 0 welded mesh\n")
        for x, y, z in vertices:
            handle.write(f"v {fmt_float(x)} {fmt_float(y)} {fmt_float(z)}\n")
        for a, b, c in faces:
            handle.write(f"f {a + 1} {b + 1} {c + 1}\n")


def write_ply(path: Path, vertices: Sequence[Vec3], faces: Sequence[Face]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for x, y, z in vertices:
            handle.write(f"{fmt_float(x)} {fmt_float(y)} {fmt_float(z)}\n")
        for a, b, c in faces:
            handle.write(f"3 {a} {b} {c}\n")


def aabb_overlap_ratio(a: Dict[str, object], b: Dict[str, object]) -> Optional[float]:
    if a["min"] is None or b["min"] is None:
        return None
    amin, amax = a["min"], a["max"]
    bmin, bmax = b["min"], b["max"]
    assert isinstance(amin, list) and isinstance(amax, list)
    assert isinstance(bmin, list) and isinstance(bmax, list)
    overlap = [max(0.0, min(amax[i], bmax[i]) - max(amin[i], bmin[i])) for i in range(3)]
    avol = max(0.0, (amax[0] - amin[0]) * (amax[1] - amin[1]) * (amax[2] - amin[2]))
    bvol = max(0.0, (bmax[0] - bmin[0]) * (bmax[1] - bmin[1]) * (bmax[2] - bmin[2]))
    denom = min(avol, bvol)
    if denom <= 0:
        return None
    return (overlap[0] * overlap[1] * overlap[2]) / denom


def sample_vertices(vertices: Sequence[Vec3], max_count: int) -> List[Vec3]:
    if len(vertices) <= max_count:
        return list(vertices)
    step = len(vertices) / max_count
    return [vertices[int(i * step)] for i in range(max_count)]


def nearest_neighbor_stats(src: Sequence[Vec3], dst: Sequence[Vec3], max_points: int = 20000) -> Optional[Dict[str, object]]:
    if not src or not dst:
        return None
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None

    src_sample = np.asarray(sample_vertices(src, max_points), dtype=np.float32)
    dst_sample = np.asarray(sample_vertices(dst, max_points), dtype=np.float32)
    distances: List[float] = []
    chunk_size = 1024
    for start in range(0, len(src_sample), chunk_size):
        chunk = src_sample[start:start + chunk_size]
        diff = chunk[:, None, :] - dst_sample[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        distances.extend(np.sqrt(np.min(d2, axis=1)).astype(float).tolist())
    return {
        "sampled_src": int(len(src_sample)),
        "sampled_dst": int(len(dst_sample)),
        "distance": summarize_values(distances),
    }


def write_summary_csv(path: Path, summary: Dict[str, object]) -> None:
    rows = [
        ("raw_vertices", summary["weld"]["raw_vertices"]),
        ("raw_faces", summary["weld"]["raw_faces"]),
        ("welded_vertices", summary["weld"]["welded_vertices"]),
        ("welded_faces", summary["weld"]["welded_faces"]),
        ("degenerate_faces_removed", summary["weld"]["degenerate_faces_removed"]),
        ("duplicate_faces_removed", summary["weld"]["duplicate_faces_removed"]),
        ("connected_components", summary["mesh"]["connected_component_count"]),
        ("boundary_edges", summary["mesh"]["boundary_edge_count"]),
        ("nonmanifold_edges", summary["mesh"]["nonmanifold_edge_count"]),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def md_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_stage0b_report(path: Path, meta_obj: Path, bl_obj: Optional[Path], meta_summary: Dict[str, object], bl_summary: Optional[Dict[str, object]], compare: Optional[Dict[str, object]]) -> None:
    lines: List[str] = []
    lines.append("# Stage 0-B Reference Audit")
    lines.append("")
    lines.append("This is a first-pass reference audit. It checks whether the welded Meta Scene Mesh is usable as a structural reference and how the latest BL surface export compares at the mesh/statistics level.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Meta welded source: `{meta_obj}`")
    lines.append(f"- BL comparison source: `{bl_obj if bl_obj else 'not provided'}`")
    lines.append("")
    lines.append("## Meta Scene Mesh After Welding")
    lines.append("")
    mesh = meta_summary["mesh"]
    weld = meta_summary["weld"]
    assert isinstance(mesh, dict) and isinstance(weld, dict)
    lines.append(f"- Raw vertices -> welded vertices: `{weld['raw_vertices']}` -> `{weld['welded_vertices']}`")
    lines.append(f"- Faces kept: `{weld['welded_faces']}`; degenerate removed: `{weld['degenerate_faces_removed']}`; duplicate removed: `{weld['duplicate_faces_removed']}`")
    lines.append(f"- Connected components: `{mesh['connected_component_count']}`; top sizes: `{mesh['connected_component_sizes_top10']}`")
    lines.append(f"- Boundary edges: `{mesh['boundary_edge_count']}`; non-manifold edges: `{mesh['nonmanifold_edge_count']}`")
    bounds_info = mesh["bounds"]
    assert isinstance(bounds_info, dict)
    lines.append(f"- Bounds center: `{bounds_info['center']}`")
    lines.append(f"- Bounds size: `{bounds_info['size']}`")
    lines.append("")

    if bl_summary:
        bl_mesh = bl_summary["mesh"]
        assert isinstance(bl_mesh, dict)
        lines.append("## BL Surface First-Pass Stats")
        lines.append("")
        lines.append(f"- Vertices: `{bl_mesh['vertex_count']}`; faces: `{bl_mesh['face_count']}`")
        lines.append(f"- Connected components: `{bl_mesh['connected_component_count']}`; top sizes: `{bl_mesh['connected_component_sizes_top10']}`")
        lines.append(f"- Boundary edges: `{bl_mesh['boundary_edge_count']}`; non-manifold edges: `{bl_mesh['nonmanifold_edge_count']}`")
        bl_bounds = bl_mesh["bounds"]
        assert isinstance(bl_bounds, dict)
        lines.append(f"- Bounds center: `{bl_bounds['center']}`")
        lines.append(f"- Bounds size: `{bl_bounds['size']}`")
        lines.append("")

    if compare:
        lines.append("## Meta-vs-BL Comparison")
        lines.append("")
        lines.append(f"- AABB overlap ratio against smaller bounds volume: `{md_value(compare.get('aabb_overlap_ratio_min_volume'))}`")
        lines.append(f"- Bounds center delta length: `{md_value(compare.get('bounds_center_delta_length'))}` m")
        lines.append(f"- Bounds diagonal ratio BL/Meta: `{md_value(compare.get('bounds_diagonal_ratio_bl_over_meta'))}`")
        for key, label in (
            ("bl_to_meta_nn", "BL vertices to nearest Meta vertex"),
            ("meta_to_bl_nn", "Meta vertices to nearest BL vertex"),
        ):
            nn = compare.get(key)
            if isinstance(nn, dict):
                dist_info = nn.get("distance")
                lines.append(f"- {label}: `{dist_info}`")
            else:
                lines.append(f"- {label}: `n/a`")
        lines.append("")

    lines.append("## Initial Reading")
    lines.append("")
    lines.append("- The welded Meta Scene Mesh should be treated as the current Stage 0 structural reference, because it is room-scale and connected after welding.")
    lines.append("- The BL comparison is only meaningful as a locality/coverage diagnostic unless the BL export was captured over the same room-scale scope.")
    lines.append("- If BL bounds are much smaller or weakly overlapping, Stage 0-B should use Meta Scene Mesh as the reference target and BL as a sampled/cropped observation, not as an equivalent full-room mesh.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def center_delta_length(a: Dict[str, object], b: Dict[str, object]) -> Optional[float]:
    ac, bc = a.get("center"), b.get("center")
    if not isinstance(ac, list) or not isinstance(bc, list):
        return None
    return math.sqrt(sum((ac[i] - bc[i]) ** 2 for i in range(3)))


def run(args: argparse.Namespace) -> None:
    meta_session = Path(args.meta_session)
    meta_obj = Path(args.meta_obj) if args.meta_obj else meta_session / "meta_scene_mesh_aligned_all.obj"
    if not meta_obj.exists():
        raise FileNotFoundError(f"Meta OBJ not found: {meta_obj}")

    out_dir = Path(args.output_dir) if args.output_dir else meta_session / "stage0_weld"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_vertices, raw_faces = parse_obj(meta_obj)
    welded_vertices, welded_faces, weld = weld_mesh(raw_vertices, raw_faces, args.eps)
    meta_mesh = mesh_stats(welded_vertices, welded_faces)
    eps_name = f"{args.eps:.0e}".replace("+", "")

    welded_obj = out_dir / f"meta_scene_mesh_aligned_all_welded_eps{eps_name}.obj"
    welded_ply = out_dir / f"meta_scene_mesh_aligned_all_welded_eps{eps_name}.ply"
    write_obj(welded_obj, welded_vertices, welded_faces)
    write_ply(welded_ply, welded_vertices, welded_faces)

    meta_summary: Dict[str, object] = {
        "source_obj": str(meta_obj),
        "epsilon": args.eps,
        "welded_obj": str(welded_obj),
        "welded_ply": str(welded_ply),
        "weld": weld,
        "mesh": meta_mesh,
    }
    summary_json = out_dir / "stage0a_weld_summary.json"
    summary_csv = out_dir / "stage0a_weld_summary.csv"
    summary_json.write_text(json.dumps(meta_summary, indent=2), encoding="utf-8")
    write_summary_csv(summary_csv, meta_summary)

    bl_summary: Optional[Dict[str, object]] = None
    compare: Optional[Dict[str, object]] = None
    bl_obj = Path(args.bl_obj) if args.bl_obj else None
    if bl_obj:
        if not bl_obj.exists():
            raise FileNotFoundError(f"BL OBJ not found: {bl_obj}")
        bl_vertices_raw, bl_faces_raw = parse_obj(bl_obj)
        bl_vertices, bl_faces, bl_weld = weld_mesh(bl_vertices_raw, bl_faces_raw, args.eps)
        bl_mesh = mesh_stats(bl_vertices, bl_faces)
        bl_summary = {
            "source_obj": str(bl_obj),
            "weld": bl_weld,
            "mesh": bl_mesh,
        }
        meta_bounds = meta_mesh["bounds"]
        bl_bounds = bl_mesh["bounds"]
        assert isinstance(meta_bounds, dict) and isinstance(bl_bounds, dict)
        compare = {
            "aabb_overlap_ratio_min_volume": aabb_overlap_ratio(meta_bounds, bl_bounds),
            "bounds_center_delta_length": center_delta_length(meta_bounds, bl_bounds),
            "bounds_diagonal_ratio_bl_over_meta": (
                bl_bounds["diagonal"] / meta_bounds["diagonal"]
                if isinstance(bl_bounds["diagonal"], float) and isinstance(meta_bounds["diagonal"], float) and meta_bounds["diagonal"] > 0
                else None
            ),
            "bl_to_meta_nn": nearest_neighbor_stats(bl_vertices, welded_vertices),
            "meta_to_bl_nn": nearest_neighbor_stats(welded_vertices, bl_vertices),
        }

        stage0b_dir = meta_session / "stage0b_reference_audit"
        stage0b_dir.mkdir(parents=True, exist_ok=True)
        report_json = stage0b_dir / "stage0b_reference_audit.json"
        report_md = stage0b_dir / "stage0b_reference_audit.md"
        payload = {
            "meta": meta_summary,
            "bl": bl_summary,
            "compare": compare,
        }
        report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_stage0b_report(report_md, meta_obj, bl_obj, meta_summary, bl_summary, compare)

    print(json.dumps({
        "welded_obj": str(welded_obj),
        "welded_ply": str(welded_ply),
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "stage0b_report": str(meta_session / "stage0b_reference_audit" / "stage0b_reference_audit.md") if bl_obj else None,
        "raw_vertices": weld["raw_vertices"],
        "welded_vertices": weld["welded_vertices"],
        "welded_faces": weld["welded_faces"],
        "connected_components": meta_mesh["connected_component_count"],
        "boundary_edges": meta_mesh["boundary_edge_count"],
        "nonmanifold_edges": meta_mesh["nonmanifold_edge_count"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Weld Meta Scene Mesh and run Stage 0-B first-pass audit.")
    parser.add_argument("--meta-session", required=True, help="MetaSceneMeshAudit session directory.")
    parser.add_argument("--meta-obj", default=None, help="Optional explicit Meta Scene Mesh OBJ path.")
    parser.add_argument("--bl-obj", default=None, help="Optional BL Surface Mesh OBJ path for Stage 0-B comparison.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory for welded artifacts.")
    parser.add_argument("--eps", type=float, default=1e-5, help="Vertex welding epsilon in meters.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
