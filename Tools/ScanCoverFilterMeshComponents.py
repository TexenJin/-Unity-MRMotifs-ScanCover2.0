#!/usr/bin/env python3
"""Filter mapping mesh candidates by connected component size.

This is a conservative post-process for diagnostic meshes generated from Raw
depth points. It does not repair geometry. It removes small floating islands so
we can tell whether the remaining large structure is worth improving.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep large connected mesh components.")
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--min-area", type=float, default=1.0)
    parser.add_argument("--min-triangles", type=int, default=1000)
    parser.add_argument("--keep-top", type=int, default=8)
    return parser.parse_args()


def mesh_stats(mesh: o3d.geometry.TriangleMesh) -> dict[str, object]:
    if len(mesh.triangles) == 0:
        return {"vertices": int(len(mesh.vertices)), "triangles": 0, "components": 0}
    _, counts, areas = mesh.cluster_connected_triangles()
    counts_arr = np.asarray(counts, dtype=np.int64)
    areas_arr = np.asarray(areas, dtype=np.float64)
    return {
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
        "components": int(len(counts_arr)),
        "largestTriangleCount": int(np.max(counts_arr)) if len(counts_arr) else 0,
        "largestArea": float(np.max(areas_arr)) if len(areas_arr) else 0.0,
        "surfaceArea": float(mesh.get_surface_area()) if len(mesh.triangles) else 0.0,
        "boundaryOrNonManifoldEdges": int(len(mesh.get_non_manifold_edges(allow_boundary_edges=False))),
    }


def main() -> None:
    args = parse_args()
    out = args.out or args.mesh.with_name(args.mesh.stem + "_large_components.ply")
    report_path = out.with_suffix(".json")

    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    if len(mesh.triangles) == 0:
        raise RuntimeError(f"No triangles in {args.mesh}")

    clusters, counts, areas = mesh.cluster_connected_triangles()
    cluster_ids = np.asarray(clusters, dtype=np.int64)
    counts_arr = np.asarray(counts, dtype=np.int64)
    areas_arr = np.asarray(areas, dtype=np.float64)

    order = np.argsort(-areas_arr)
    keep_components: set[int] = set()
    for rank, component_id in enumerate(order):
        if rank >= args.keep_top:
            break
        if areas_arr[component_id] >= args.min_area and counts_arr[component_id] >= args.min_triangles:
            keep_components.add(int(component_id))

    remove_mask = np.array([int(component_id) not in keep_components for component_id in cluster_ids], dtype=bool)
    filtered = o3d.geometry.TriangleMesh(mesh)
    filtered.remove_triangles_by_mask(remove_mask)
    filtered.remove_unreferenced_vertices()
    filtered.remove_degenerate_triangles()
    filtered.remove_duplicated_triangles()
    filtered.remove_duplicated_vertices()
    filtered.compute_vertex_normals()

    o3d.io.write_triangle_mesh(str(out), filtered, write_ascii=False)
    report = {
        "input": str(args.mesh),
        "output": str(out),
        "parameters": {
            "minArea": float(args.min_area),
            "minTriangles": int(args.min_triangles),
            "keepTop": int(args.keep_top),
        },
        "keptComponents": sorted(keep_components),
        "componentCountBefore": int(len(counts_arr)),
        "componentAreasTop": [float(areas_arr[i]) for i in order[: min(12, len(order))]],
        "componentTrianglesTop": [int(counts_arr[i]) for i in order[: min(12, len(order))]],
        "before": mesh_stats(mesh),
        "after": mesh_stats(filtered),
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["after"], indent=2))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
