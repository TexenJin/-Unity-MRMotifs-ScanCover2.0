#!/usr/bin/env python3
"""Build a topology-preserving mesh preview from dense Raw Depth frames.

This is a diagnostic tool, not the final mapping mesh builder.

The previous point-cloud meshing path intentionally forgot where each raw point
came from in the depth image. That allows arbitrary connections between nearby
points from different frames, thickness layers, holes, or edges. This script
keeps the original 2D depth-image topology: it only connects neighboring pixels
inside the same frame, and only when the local depth/edge tests pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path(
    r"D:\PCA\Unity-MRMotifs-ScanCover-main"
    r"\ScanCoverExports\RepeatCoverageSessions\RepeatCoverageSessions"
)


@dataclass(frozen=True)
class RawPoint:
    x: float
    y: float
    z: float
    depth: float
    nx: float
    ny: float
    nz: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a per-frame topology mesh preview from room_raw_depth_frames."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help="Folder containing ScanCover_RepeatCoverage_* sessions, a single session, or room_raw_depth_frames.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output folder.")
    parser.add_argument("--max-frames", type=int, default=80, help="Global max frames to consume. 0 means all.")
    parser.add_argument("--max-sessions", type=int, default=0, help="Max sessions to consume. 0 means all.")
    parser.add_argument("--frame-stride", type=int, default=10, help="Use one frame every N frames per session.")
    parser.add_argument("--pixel-step", type=int, default=2, help="Use every Nth raw depth pixel.")
    parser.add_argument("--min-depth", type=float, default=0.20)
    parser.add_argument("--max-depth", type=float, default=5.00)
    parser.add_argument("--max-edge", type=float, default=0.18, help="Max 3D distance between adjacent sampled pixels.")
    parser.add_argument("--max-depth-jump", type=float, default=0.18, help="Max depth difference for a connected edge.")
    parser.add_argument("--max-normal-angle", type=float, default=70.0, help="Max normal angle in degrees for a connected edge.")
    parser.add_argument("--min-triangle-area", type=float, default=0.00001)
    parser.add_argument("--points-only", action="store_true", help="Only write point preview, no mesh faces.")
    return parser.parse_args()


def find_sessions(path: Path) -> list[Path]:
    if path.name == "room_raw_depth_frames":
        return [path.parent]
    if (path / "room_raw_depth_frames").exists():
        return [path]
    if not path.is_dir():
        return []
    sessions = sorted(
        [p for p in path.iterdir() if p.is_dir() and (p / "room_raw_depth_frames").exists()],
        key=lambda p: p.name.lower(),
    )
    if sessions:
        return sessions
    return sorted({p.parent for p in path.rglob("room_raw_depth_frames")}, key=lambda p: str(p).lower())


def iter_frame_csvs(session: Path, frame_stride: int) -> Iterable[Path]:
    files = sorted((session / "room_raw_depth_frames").glob("frame_*_raw_depth.csv"))
    if frame_stride > 1:
        files = files[::frame_stride]
    return files


def to_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def to_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def read_frame_points(
    csv_path: Path,
    pixel_step: int,
    min_depth: float,
    max_depth: float,
) -> tuple[dict[tuple[int, int], RawPoint], dict[str, int]]:
    points: dict[tuple[int, int], RawPoint] = {}
    stats = {"rows": 0, "used": 0, "rejectedDepth": 0, "rejectedInvalid": 0}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        first = fh.readline()
        second = fh.readline()
        if not first.startswith("#") or not second.startswith("resolution="):
            fh.seek(0)
        reader = csv.DictReader(fh)
        for row in reader:
            stats["rows"] += 1
            px = to_int(row.get("pixelX"), -1)
            py = to_int(row.get("pixelY"), -1)
            if px < 0 or py < 0 or px % pixel_step != 0 or py % pixel_step != 0:
                continue
            depth = to_float(row.get("depthM"), -1.0)
            if not math.isfinite(depth) or depth < min_depth or depth > max_depth:
                stats["rejectedDepth"] += 1
                continue
            x = to_float(row.get("worldX"), math.nan)
            y = to_float(row.get("worldY"), math.nan)
            z = to_float(row.get("worldZ"), math.nan)
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                stats["rejectedInvalid"] += 1
                continue
            points[(px, py)] = RawPoint(
                x=x,
                y=y,
                z=z,
                depth=depth,
                nx=to_float(row.get("normalX"), 0.0),
                ny=to_float(row.get("normalY"), 0.0),
                nz=to_float(row.get("normalZ"), 0.0),
            )
            stats["used"] += 1
    return points, stats


def dist(a: RawPoint, b: RawPoint) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def normal_angle_ok(a: RawPoint, b: RawPoint, min_dot: float) -> bool:
    al = math.sqrt(a.nx * a.nx + a.ny * a.ny + a.nz * a.nz)
    bl = math.sqrt(b.nx * b.nx + b.ny * b.ny + b.nz * b.nz)
    if al < 1e-6 or bl < 1e-6:
        return True
    dot = (a.nx * b.nx + a.ny * b.ny + a.nz * b.nz) / (al * bl)
    return abs(dot) >= min_dot


def edge_ok(a: RawPoint, b: RawPoint, max_edge: float, max_depth_jump: float, min_dot: float) -> bool:
    if abs(a.depth - b.depth) > max_depth_jump:
        return False
    if dist(a, b) > max_edge:
        return False
    return normal_angle_ok(a, b, min_dot)


def triangle_area(a: RawPoint, b: RawPoint, c: RawPoint) -> float:
    ab = (b.x - a.x, b.y - a.y, b.z - a.z)
    ac = (c.x - a.x, c.y - a.y, c.z - a.z)
    cx = ab[1] * ac[2] - ab[2] * ac[1]
    cy = ab[2] * ac[0] - ab[0] * ac[2]
    cz = ab[0] * ac[1] - ab[1] * ac[0]
    return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)


def add_frame_mesh(
    frame_points: dict[tuple[int, int], RawPoint],
    pixel_step: int,
    vertices: list[tuple[float, float, float, int, int, int]],
    faces: list[tuple[int, int, int]],
    frame_index: int,
    max_edge: float,
    max_depth_jump: float,
    min_dot: float,
    min_triangle_area: float,
    write_faces: bool,
) -> dict[str, int]:
    local_index: dict[tuple[int, int], int] = {}
    palette = [
        (255, 255, 255),
        (0, 220, 255),
        (255, 210, 0),
        (255, 70, 70),
        (90, 255, 120),
        (255, 80, 220),
    ]
    color = palette[frame_index % len(palette)]
    for key, point in frame_points.items():
        local_index[key] = len(vertices)
        vertices.append((point.x, point.y, point.z, color[0], color[1], color[2]))

    stats = {"vertices": len(frame_points), "faces": 0, "skippedQuads": 0, "candidateQuads": 0}
    if not write_faces:
        return stats

    xs = sorted({px for px, _ in frame_points.keys()})
    ys = sorted({py for _, py in frame_points.keys()})
    if not xs or not ys:
        return stats
    x_set = set(xs)
    y_set = set(ys)
    for y in ys:
        if y + pixel_step not in y_set:
            continue
        for x in xs:
            if x + pixel_step not in x_set:
                continue
            k00 = (x, y)
            k10 = (x + pixel_step, y)
            k01 = (x, y + pixel_step)
            k11 = (x + pixel_step, y + pixel_step)
            p00 = frame_points.get(k00)
            p10 = frame_points.get(k10)
            p01 = frame_points.get(k01)
            p11 = frame_points.get(k11)
            if p00 is None or p10 is None or p01 is None or p11 is None:
                stats["skippedQuads"] += 1
                continue
            stats["candidateQuads"] += 1

            if (
                edge_ok(p00, p10, max_edge, max_depth_jump, min_dot)
                and edge_ok(p10, p01, max_edge, max_depth_jump, min_dot)
                and edge_ok(p01, p00, max_edge, max_depth_jump, min_dot)
                and triangle_area(p00, p10, p01) >= min_triangle_area
            ):
                faces.append((local_index[k00], local_index[k10], local_index[k01]))
                stats["faces"] += 1

            if (
                edge_ok(p10, p11, max_edge, max_depth_jump, min_dot)
                and edge_ok(p11, p01, max_edge, max_depth_jump, min_dot)
                and edge_ok(p01, p10, max_edge, max_depth_jump, min_dot)
                and triangle_area(p10, p11, p01) >= min_triangle_area
            ):
                faces.append((local_index[k10], local_index[k11], local_index[k01]))
                stats["faces"] += 1
    return stats


def write_ply(path: Path, vertices: list[tuple[float, float, float, int, int, int]], faces: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(vertices)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            f"element face {len(faces)}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        fh.write(header.encode("ascii"))
        for x, y, z, r, g, b in vertices:
            fh.write(struct.pack("<fffBBB", x, y, z, r, g, b))
        for a, b, c in faces:
            fh.write(struct.pack("<Biii", 3, a, b, c))


def main() -> int:
    args = parse_args()
    sessions = find_sessions(args.input)
    if args.max_sessions > 0:
        sessions = sessions[: args.max_sessions]
    if not sessions:
        raise SystemExit(f"No sessions with room_raw_depth_frames found under {args.input}")

    out_dir = args.out or (args.input / "raw_depth_topology_mesh_preview")
    out_dir.mkdir(parents=True, exist_ok=True)

    vertices: list[tuple[float, float, float, int, int, int]] = []
    faces: list[tuple[int, int, int]] = []
    frame_reports: list[dict[str, object]] = []
    consumed = 0
    min_dot = math.cos(math.radians(max(0.0, min(180.0, args.max_normal_angle))))

    for session in sessions:
        for csv_path in iter_frame_csvs(session, max(1, args.frame_stride)):
            if args.max_frames > 0 and consumed >= args.max_frames:
                break
            frame_points, read_stats = read_frame_points(
                csv_path,
                max(1, args.pixel_step),
                args.min_depth,
                args.max_depth,
            )
            mesh_stats = add_frame_mesh(
                frame_points=frame_points,
                pixel_step=max(1, args.pixel_step),
                vertices=vertices,
                faces=faces,
                frame_index=consumed,
                max_edge=args.max_edge,
                max_depth_jump=args.max_depth_jump,
                min_dot=min_dot,
                min_triangle_area=args.min_triangle_area,
                write_faces=not args.points_only,
            )
            frame_reports.append(
                {
                    "session": session.name,
                    "frame": csv_path.name,
                    "read": read_stats,
                    "mesh": mesh_stats,
                }
            )
            consumed += 1
        if args.max_frames > 0 and consumed >= args.max_frames:
            break

    mesh_path = out_dir / "raw_topology_mesh_preview.ply"
    points_path = out_dir / "raw_topology_points_preview.ply"
    if not args.points_only:
        write_ply(mesh_path, vertices, faces)
    write_ply(points_path, vertices, [])

    report = {
        "input": str(args.input),
        "out": str(out_dir),
        "sessionsFound": len(sessions),
        "framesConsumed": consumed,
        "pixelStep": args.pixel_step,
        "frameStride": args.frame_stride,
        "maxEdge": args.max_edge,
        "maxDepthJump": args.max_depth_jump,
        "maxNormalAngle": args.max_normal_angle,
        "vertices": len(vertices),
        "faces": len(faces),
        "meshPath": str(mesh_path) if not args.points_only else None,
        "pointsPath": str(points_path),
        "frames": frame_reports,
    }
    (out_dir / "raw_topology_mesh_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in ("framesConsumed", "vertices", "faces", "meshPath", "pointsPath")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
