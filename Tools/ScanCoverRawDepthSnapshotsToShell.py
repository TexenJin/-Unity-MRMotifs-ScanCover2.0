#!/usr/bin/env python3
"""Build a room shell directly from full 160x160 Raw Depth snapshots.

This is the snapshot-first path. It intentionally does not use stable/candidate
coverage labels. Each snapshot is treated as a small depth-image surface patch:
neighboring pixels inside the same frame may form triangles when local depth,
3D edge length, and normal continuity are reasonable. Multiple snapshots are
then placed together by their already-exported world-space positions.
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
    r"\ScanCoverExports\RepeatCoverageSessions"
)


@dataclass(frozen=True)
class SnapshotPoint:
    x: float
    y: float
    z: float
    depth: float
    nx: float
    ny: float
    nz: float
    confidence: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a topology-preserving room shell from room_raw_depth_snapshots."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help="Folder containing ScanCover_RepeatCoverage_* sessions, one session, or room_raw_depth_snapshots.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output folder.")
    parser.add_argument("--max-sessions", type=int, default=0, help="0 means all.")
    parser.add_argument("--max-snapshots", type=int, default=0, help="0 means all.")
    parser.add_argument("--pixel-step", type=int, default=1, help="Use every Nth raw-depth pixel.")
    parser.add_argument("--min-depth", type=float, default=0.20)
    parser.add_argument("--max-depth", type=float, default=5.00)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-edge", type=float, default=0.18, help="Max 3D distance between adjacent sampled pixels.")
    parser.add_argument("--max-depth-jump", type=float, default=0.18, help="Max depth difference for a connected edge.")
    parser.add_argument("--max-normal-angle", type=float, default=70.0, help="Max normal angle in degrees for a connected edge.")
    parser.add_argument("--min-triangle-area", type=float, default=0.00001)
    parser.add_argument("--preview-voxel", type=float, default=0.08, help="Voxel size for the merged point preview.")
    parser.add_argument("--points-only", action="store_true", help="Only write points, no mesh faces.")
    parser.add_argument("--uniform-color", action="store_true", help="Use one color instead of coloring by snapshot.")
    parser.add_argument("--write-frame-points", action="store_true", help="Also write per-snapshot point PLY files.")
    return parser.parse_args()


def find_sessions(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.name == "room_raw_depth_snapshots":
        return [path.parent]
    if (path / "room_raw_depth_snapshots").exists():
        return [path]
    if not path.is_dir():
        return []
    sessions = sorted(
        [p for p in path.iterdir() if p.is_dir() and (p / "room_raw_depth_snapshots").exists()],
        key=lambda p: p.name.lower(),
    )
    if sessions:
        return sessions
    return sorted({p.parent for p in path.rglob("room_raw_depth_snapshots")}, key=lambda p: str(p).lower())


def iter_snapshot_csvs(session: Path) -> Iterable[Path]:
    return sorted((session / "room_raw_depth_snapshots").glob("*_raw_snapshot.csv"))


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


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y"}


def read_snapshot_points(
    csv_path: Path,
    pixel_step: int,
    min_depth: float,
    max_depth: float,
    min_confidence: float,
) -> tuple[dict[tuple[int, int], SnapshotPoint], dict[str, int]]:
    points: dict[tuple[int, int], SnapshotPoint] = {}
    stats = {
        "rows": 0,
        "used": 0,
        "invalidFlag": 0,
        "rejectedPixelStep": 0,
        "rejectedDepth": 0,
        "rejectedConfidence": 0,
        "rejectedPosition": 0,
    }
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        first = fh.readline()
        second = fh.readline()
        if not first.startswith("#") or not second.startswith("resolution="):
            fh.seek(0)
        reader = csv.DictReader(fh)
        for row in reader:
            stats["rows"] += 1
            if "valid" in row and not truthy(row.get("valid")):
                stats["invalidFlag"] += 1
                continue
            px = to_int(row.get("pixelX"), -1)
            py = to_int(row.get("pixelY"), -1)
            if px < 0 or py < 0 or px % pixel_step != 0 or py % pixel_step != 0:
                stats["rejectedPixelStep"] += 1
                continue
            depth = to_float(row.get("depthM"), -1.0)
            if not math.isfinite(depth) or depth < min_depth or depth > max_depth:
                stats["rejectedDepth"] += 1
                continue
            confidence = to_float(row.get("confidence"), 1.0)
            if confidence < min_confidence:
                stats["rejectedConfidence"] += 1
                continue
            x = to_float(row.get("worldX"), math.nan)
            y = to_float(row.get("worldY"), math.nan)
            z = to_float(row.get("worldZ"), math.nan)
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                stats["rejectedPosition"] += 1
                continue
            points[(px, py)] = SnapshotPoint(
                x=x,
                y=y,
                z=z,
                depth=depth,
                nx=to_float(row.get("normalX"), 0.0),
                ny=to_float(row.get("normalY"), 0.0),
                nz=to_float(row.get("normalZ"), 0.0),
                confidence=confidence,
            )
            stats["used"] += 1
    return points, stats


def dist(a: SnapshotPoint, b: SnapshotPoint) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def normal_angle_ok(a: SnapshotPoint, b: SnapshotPoint, min_dot: float) -> bool:
    al = math.sqrt(a.nx * a.nx + a.ny * a.ny + a.nz * a.nz)
    bl = math.sqrt(b.nx * b.nx + b.ny * b.ny + b.nz * b.nz)
    if al < 1e-6 or bl < 1e-6:
        return True
    dot = (a.nx * b.nx + a.ny * b.ny + a.nz * b.nz) / (al * bl)
    return abs(dot) >= min_dot


def edge_ok(a: SnapshotPoint, b: SnapshotPoint, max_edge: float, max_depth_jump: float, min_dot: float) -> bool:
    if abs(a.depth - b.depth) > max_depth_jump:
        return False
    if dist(a, b) > max_edge:
        return False
    return normal_angle_ok(a, b, min_dot)


def triangle_area(a: SnapshotPoint, b: SnapshotPoint, c: SnapshotPoint) -> float:
    ab = (b.x - a.x, b.y - a.y, b.z - a.z)
    ac = (c.x - a.x, c.y - a.y, c.z - a.z)
    cx = ab[1] * ac[2] - ab[2] * ac[1]
    cy = ab[2] * ac[0] - ab[0] * ac[2]
    cz = ab[0] * ac[1] - ab[1] * ac[0]
    return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)


def voxel_key(point: SnapshotPoint, voxel: float) -> tuple[int, int, int]:
    return (
        math.floor(point.x / voxel),
        math.floor(point.y / voxel),
        math.floor(point.z / voxel),
    )


def add_snapshot_mesh(
    frame_points: dict[tuple[int, int], SnapshotPoint],
    pixel_step: int,
    vertices: list[tuple[float, float, float, int, int, int]],
    faces: list[tuple[int, int, int]],
    snapshot_index: int,
    max_edge: float,
    max_depth_jump: float,
    min_dot: float,
    min_triangle_area: float,
    write_faces: bool,
    uniform_color: bool,
) -> dict[str, int]:
    local_index: dict[tuple[int, int], int] = {}
    palette = [
        (205, 205, 205),
        (0, 220, 255),
        (255, 210, 0),
        (255, 85, 85),
        (90, 255, 120),
        (255, 80, 220),
        (110, 170, 255),
        (255, 160, 60),
    ]
    color = (210, 210, 210) if uniform_color else palette[snapshot_index % len(palette)]
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


def bbox(vertices: list[tuple[float, float, float, int, int, int]]) -> dict[str, object]:
    if not vertices:
        return {"count": 0, "min": None, "max": None, "size": None}
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    mn = [min(xs), min(ys), min(zs)]
    mx = [max(xs), max(ys), max(zs)]
    return {
        "count": len(vertices),
        "min": mn,
        "max": mx,
        "size": [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]],
    }


def main() -> int:
    args = parse_args()
    sessions = find_sessions(args.input)
    if args.max_sessions > 0:
        sessions = sessions[: args.max_sessions]
    if not sessions:
        raise SystemExit(f"No sessions with room_raw_depth_snapshots found under {args.input}")

    out_dir = args.out or (args.input / "snapshot_shell")
    out_dir.mkdir(parents=True, exist_ok=True)

    vertices: list[tuple[float, float, float, int, int, int]] = []
    faces: list[tuple[int, int, int]] = []
    merged_preview: dict[tuple[int, int, int], tuple[float, float, float, int, int, int, int]] = {}
    frame_reports: list[dict[str, object]] = []
    consumed = 0
    min_dot = math.cos(math.radians(max(0.0, min(180.0, args.max_normal_angle))))
    pixel_step = max(1, args.pixel_step)
    preview_voxel = max(0.001, args.preview_voxel)

    for session in sessions:
        for csv_path in iter_snapshot_csvs(session):
            if args.max_snapshots > 0 and consumed >= args.max_snapshots:
                break
            frame_points, read_stats = read_snapshot_points(
                csv_path,
                pixel_step=pixel_step,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                min_confidence=args.min_confidence,
            )
            start_vertex = len(vertices)
            mesh_stats = add_snapshot_mesh(
                frame_points=frame_points,
                pixel_step=pixel_step,
                vertices=vertices,
                faces=faces,
                snapshot_index=consumed,
                max_edge=args.max_edge,
                max_depth_jump=args.max_depth_jump,
                min_dot=min_dot,
                min_triangle_area=args.min_triangle_area,
                write_faces=not args.points_only,
                uniform_color=args.uniform_color,
            )
            for point in frame_points.values():
                key = voxel_key(point, preview_voxel)
                prev = merged_preview.get(key)
                if prev is None:
                    merged_preview[key] = (point.x, point.y, point.z, 1, 210, 210, 210)
                else:
                    sx, sy, sz, hits, r, g, b = prev
                    merged_preview[key] = (sx + point.x, sy + point.y, sz + point.z, hits + 1, r, g, b)

            frame_vertices = vertices[start_vertex:]
            if args.write_frame_points and frame_vertices:
                write_ply(out_dir / "per_snapshot_points" / f"{consumed:04d}_{session.name}_{csv_path.stem}.ply", frame_vertices, [])
            frame_reports.append(
                {
                    "session": session.name,
                    "snapshot": csv_path.name,
                    "read": read_stats,
                    "mesh": mesh_stats,
                    "bbox": bbox(frame_vertices),
                }
            )
            consumed += 1
        if args.max_snapshots > 0 and consumed >= args.max_snapshots:
            break

    preview_vertices = [
        (sx / hits, sy / hits, sz / hits, r, g, b)
        for sx, sy, sz, hits, r, g, b in merged_preview.values()
        if hits > 0
    ]
    shell_points = out_dir / "snapshot_shell_points.ply"
    shell_mesh = out_dir / "snapshot_shell_mesh.ply"
    merged_points = out_dir / "snapshot_shell_merged_preview.ply"
    write_ply(shell_points, vertices, [])
    if not args.points_only:
        write_ply(shell_mesh, vertices, faces)
    write_ply(merged_points, preview_vertices, [])

    report = {
        "input": str(args.input),
        "out": str(out_dir),
        "source": "room_raw_depth_snapshots",
        "semantics": "snapshot topology shell; does not use stable/candidate coverage labels",
        "sessionsFound": len(sessions),
        "snapshotsConsumed": consumed,
        "pixelStep": pixel_step,
        "minDepth": args.min_depth,
        "maxDepth": args.max_depth,
        "minConfidence": args.min_confidence,
        "maxEdge": args.max_edge,
        "maxDepthJump": args.max_depth_jump,
        "maxNormalAngle": args.max_normal_angle,
        "vertices": len(vertices),
        "faces": len(faces),
        "mergedPreviewVoxel": preview_voxel,
        "mergedPreviewVertices": len(preview_vertices),
        "pointsPath": str(shell_points),
        "meshPath": str(shell_mesh) if not args.points_only else None,
        "mergedPreviewPath": str(merged_points),
        "bbox": bbox(vertices),
        "frames": frame_reports,
    }
    (out_dir / "snapshot_shell_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in ("snapshotsConsumed", "vertices", "faces", "mergedPreviewVertices")}, indent=2))
    print(f"[snapshot-shell] done: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
