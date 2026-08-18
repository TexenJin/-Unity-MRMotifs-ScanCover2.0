#!/usr/bin/env python
"""Lightweight PLY viewer for ScanCover workbench outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d


def load_geometry(path: Path) -> o3d.geometry.Geometry:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if mesh is not None and len(mesh.triangles) > 0:
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        return mesh

    points = o3d.io.read_point_cloud(str(path))
    if points is None or len(points.points) == 0:
        raise RuntimeError(f"No mesh triangles or point samples found in {path}")
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description="Open one or more PLY/OBJ files with Open3D.")
    parser.add_argument("paths", nargs="+", help="PLY/OBJ files to view.")
    parser.add_argument("--point-size", type=float, default=2.0, help="Point size for point clouds.")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    geometries: list[o3d.geometry.Geometry] = []
    for raw_path in args.paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        geometries.append(load_geometry(path))
        print(f"[ScanCoverViewPly] loaded {path}")

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        window_name="ScanCover PLY Viewer",
        width=args.width,
        height=args.height,
    )
    for geometry in geometries:
        visualizer.add_geometry(geometry)

    render = visualizer.get_render_option()
    render.point_size = args.point_size
    render.mesh_show_back_face = True
    render.background_color = [0.03, 0.03, 0.035]

    print("[ScanCoverViewPly] mouse: rotate/pan/zoom, Q or Esc: close")
    visualizer.run()
    visualizer.destroy_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
