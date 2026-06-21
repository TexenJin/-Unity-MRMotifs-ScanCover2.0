#!/usr/bin/env python3
"""Convert ScanCover scan coverage data to PLY point clouds.

The primary input is room_raw_coverage_voxels.csv exported by
ScanCoverMultiFrameSessionExporter. That CSV already contains world-space
coverage voxel centers, so it is the safest source for checking scan coverage.

Outputs intentionally separate two uses:
- coverage_preview.ply: loose coverage view, for checking what area was scanned.
- trusted_mapping.ply: stable non-risk points, for downstream mapping/fusion tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Point:
    x: float
    y: float
    z: float
    nx: float
    ny: float
    nz: float
    r: int
    g: int
    b: int
    stable: bool = False
    risk: bool = False
    high: bool = False
    low: bool = False
    frame_hits: int = 0
    point_hits: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ScanCover coverage CSV/session folders to PLY point clouds."
    )
    parser.add_argument(
        "input",
        help="Session folder, room_raw_coverage folder, room_raw_coverage_voxels.csv, or a folder containing *_vertices.csv.",
    )
    parser.add_argument(
        "--out",
        help="Output folder. Default: <input>/coverage_ply or sibling coverage_ply folder.",
    )
    parser.add_argument(
        "--vertices-fallback",
        action="store_true",
        help="If no room_raw_coverage_voxels.csv is found, convert *_vertices.csv files instead.",
    )
    return parser.parse_args()


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y"}


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
        return int(float(value))
    except ValueError:
        return default


def find_coverage_csv(input_path: Path) -> Path | None:
    if input_path.is_file() and input_path.name.lower() == "room_raw_coverage_voxels.csv":
        return input_path

    if input_path.is_dir():
        direct = input_path / "room_raw_coverage_voxels.csv"
        if direct.exists():
            return direct

        nested = input_path / "room_raw_coverage" / "room_raw_coverage_voxels.csv"
        if nested.exists():
            return nested

        matches = list(input_path.rglob("room_raw_coverage_voxels.csv"))
        if matches:
            return matches[0]

    return None


def find_coverage_csvs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.name.lower() == "room_raw_coverage_voxels.csv" else []

    if not input_path.is_dir():
        return []

    matches: list[Path] = []
    direct = input_path / "room_raw_coverage_voxels.csv"
    if direct.exists():
        matches.append(direct)

    nested = input_path / "room_raw_coverage" / "room_raw_coverage_voxels.csv"
    if nested.exists():
        matches.append(nested)

    matches.extend(input_path.rglob("room_raw_coverage_voxels.csv"))

    unique: dict[str, Path] = {}
    for match in matches:
        # Ignore any accidental copies under previous output folders.
        if "coverage_ply" in {part.lower() for part in match.parts}:
            continue
        unique[str(match.resolve()).lower()] = match.resolve()

    return sorted(unique.values(), key=lambda p: str(p).lower())


def find_csv_header(lines: list[str], starts_with: str) -> int:
    for i, line in enumerate(lines):
        if line.startswith(starts_with):
            return i
    raise ValueError(f"CSV header starting with {starts_with!r} not found")


def read_coverage_points(csv_path: Path) -> tuple[list[Point], dict[str, str]]:
    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    header_index = find_csv_header(lines, "voxelX,")

    meta: dict[str, str] = {}
    for line in lines[:header_index]:
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        meta[key.strip()] = value.strip()

    points: list[Point] = []
    reader = csv.DictReader(lines[header_index:])
    for row in reader:
        stable = truthy(row.get("stable"))
        risk = truthy(row.get("risk"))
        high = truthy(row.get("high"))
        low = truthy(row.get("low"))
        frame_hits = to_int(row.get("frameHits"))
        point_hits = to_int(row.get("pointHits"))

        if risk:
            color = (255, 55, 35)
        elif high and stable:
            color = (255, 225, 35)
        elif low and stable:
            color = (70, 130, 255)
        elif stable:
            color = (0, 235, 235)
        else:
            color = (115, 115, 115)

        points.append(
            Point(
                x=to_float(row.get("avgX")),
                y=to_float(row.get("avgY")),
                z=to_float(row.get("avgZ")),
                nx=to_float(row.get("avgNormalX")),
                ny=to_float(row.get("avgNormalY")),
                nz=to_float(row.get("avgNormalZ")),
                r=color[0],
                g=color[1],
                b=color[2],
                stable=stable,
                risk=risk,
                high=high,
                low=low,
                frame_hits=frame_hits,
                point_hits=point_hits,
            )
        )

    return points, meta


def find_vertices_csvs(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.name.endswith("_vertices.csv"):
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("*_vertices.csv"))
    return []


def read_vertex_points(paths: Iterable[Path]) -> list[Point]:
    points: list[Point] = []
    seen: set[tuple[int, int, int]] = set()

    for path in paths:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        try:
            header_index = find_csv_header(lines, "index,")
        except ValueError:
            continue

        reader = csv.DictReader(lines[header_index:])
        for row in reader:
            x = to_float(row.get("worldX"))
            y = to_float(row.get("worldY"))
            z = to_float(row.get("worldZ"))

            # Coarse dedupe keeps repeated frame exports readable in CloudCompare.
            key = (round(x * 1000), round(y * 1000), round(z * 1000))
            if key in seen:
                continue
            seen.add(key)

            points.append(
                Point(
                    x=x,
                    y=y,
                    z=z,
                    nx=to_float(row.get("normalWorldX")),
                    ny=to_float(row.get("normalWorldY")),
                    nz=to_float(row.get("normalWorldZ")),
                    r=230,
                    g=230,
                    b=230,
                    stable=True,
                )
            )

    return points


def hit_heat_color(value: int, min_value: int, max_value: int) -> tuple[int, int, int]:
    if max_value <= min_value:
        t = 1.0
    else:
        # Log scale makes both sparse and dense areas visible.
        lo = math.log1p(max(0, min_value))
        hi = math.log1p(max(0, max_value))
        t = (math.log1p(max(0, value)) - lo) / max(1e-6, hi - lo)
        t = max(0.0, min(1.0, t))

    if t < 0.5:
        u = t / 0.5
        return (round(40 + 215 * u), round(90 + 140 * u), 255)

    u = (t - 0.5) / 0.5
    return (255, round(230 + 25 * u), round(255 * (1.0 - u)))


def with_hit_heat_colors(points: list[Point]) -> list[Point]:
    if not points:
        return []

    values = [p.frame_hits if p.frame_hits > 0 else p.point_hits for p in points]
    min_value = min(values)
    max_value = max(values)

    out: list[Point] = []
    for p, value in zip(points, values):
        r, g, b = hit_heat_color(value, min_value, max_value)
        out.append(
            Point(
                x=p.x,
                y=p.y,
                z=p.z,
                nx=p.nx,
                ny=p.ny,
                nz=p.nz,
                r=r,
                g=g,
                b=b,
                stable=p.stable,
                risk=p.risk,
                high=p.high,
                low=p.low,
                frame_hits=p.frame_hits,
                point_hits=p.point_hits,
            )
        )
    return out


def with_mapping_input_colors(points: list[Point]) -> list[Point]:
    out: list[Point] = []
    for p in points:
        if p.stable and not p.risk:
            color = (0, 245, 245)
        elif p.stable and p.risk:
            color = (255, 185, 35)
        elif p.risk:
            color = (255, 55, 35)
        else:
            color = (120, 120, 120)

        out.append(
            Point(
                x=p.x,
                y=p.y,
                z=p.z,
                nx=p.nx,
                ny=p.ny,
                nz=p.nz,
                r=color[0],
                g=color[1],
                b=color[2],
                stable=p.stable,
                risk=p.risk,
                high=p.high,
                low=p.low,
                frame_hits=p.frame_hits,
                point_hits=p.point_hits,
            )
        )

    return out


def write_ply(path: Path, points: list[Point]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p in points:
            f.write(
                f"{p.x:.6f} {p.y:.6f} {p.z:.6f} "
                f"{p.nx:.6f} {p.ny:.6f} {p.nz:.6f} "
                f"{p.r} {p.g} {p.b}\n"
            )


def bounds(points: list[Point]) -> dict[str, list[float]]:
    if not points:
        return {"min": [0, 0, 0], "max": [0, 0, 0], "size": [0, 0, 0]}

    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    mn = [min(xs), min(ys), min(zs)]
    mx = [max(xs), max(ys), max(zs)]
    return {
        "min": [round(v, 6) for v in mn],
        "max": [round(v, 6) for v in mx],
        "size": [round(mx[i] - mn[i], 6) for i in range(3)],
    }


def average(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def default_output_dir(input_path: Path, coverage_csv: Path | None) -> Path:
    if coverage_csv is not None:
        return coverage_csv.parent / "coverage_ply"
    if input_path.is_dir():
        return input_path / "coverage_ply"
    return input_path.parent / "coverage_ply"


def default_batch_output_dir(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path / "coverage_ply_batch"
    return input_path.parent / "coverage_ply_batch"


def session_name_for_csv(input_root: Path, csv_path: Path, index: int) -> str:
    session_dir = csv_path.parent.parent if csv_path.parent.name == "room_raw_coverage" else csv_path.parent
    try:
        rel = session_dir.relative_to(input_root if input_root.is_dir() else input_root.parent)
        text = "__".join(rel.parts)
    except ValueError:
        text = session_dir.name

    text = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in text)
    return f"{index:03d}_{text or session_dir.name}"


def write_outputs(points: list[Point], out_dir: Path, source: Path, source_type: str, meta: dict[str, str]) -> None:
    stable = [p for p in points if p.stable and not p.risk]
    stable_all = [p for p in points if p.stable]
    stable_risk = [p for p in points if p.stable and p.risk]
    risk = [p for p in points if p.risk]
    unstable = [p for p in points if not p.stable and not p.risk]
    high = [p for p in points if p.high]
    low = [p for p in points if p.low]

    outputs = {
        "coverage_preview.ply": points,
        "trusted_mapping.ply": stable,
        "mapping_input_candidate.ply": with_mapping_input_colors(stable_all),
        "mapping_input_stable_all.ply": with_mapping_input_colors(stable_all),
        "mapping_input_review.ply": with_mapping_input_colors(points),
        "coverage_all_by_status.ply": points,
        "coverage_stable.ply": stable,
        "coverage_stable_all.ply": with_mapping_input_colors(stable_all),
        "coverage_stable_risk.ply": with_mapping_input_colors(stable_risk),
        "coverage_risk.ply": risk,
        "coverage_unstable.ply": unstable,
        "coverage_hit_heatmap.ply": with_hit_heat_colors(points),
    }

    if high:
        outputs["coverage_high.ply"] = high
    if low:
        outputs["coverage_low.ply"] = low

    for name, cloud in outputs.items():
        write_ply(out_dir / name, cloud)

    frame_hits = [p.frame_hits for p in points]
    point_hits = [p.point_hits for p in points]
    report = {
        "sourceType": source_type,
        "sourcePath": str(source),
        "sourceMeta": meta,
        "pointCount": len(points),
        "stableCount": len(stable),
        "stableAllCount": len(stable_all),
        "stableRiskCount": len(stable_risk),
        "riskCount": len(risk),
        "unstableCount": len(unstable),
        "coveragePreviewCount": len(points),
        "trustedMappingCount": len(stable),
        "mappingInputStableAllCount": len(stable_all),
        "highCount": len(high),
        "lowCount": len(low),
        "bounds": bounds(points),
        "frameHits": {
            "min": min(frame_hits) if frame_hits else 0,
            "max": max(frame_hits) if frame_hits else 0,
            "avg": round(average(frame_hits), 3),
        },
        "pointHits": {
            "min": min(point_hits) if point_hits else 0,
            "max": max(point_hits) if point_hits else 0,
            "avg": round(average(point_hits), 3),
        },
        "outputs": sorted(outputs.keys()),
    }

    (out_dir / "coverage_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_batch_outputs(input_path: Path, coverage_csvs: list[Path], out_dir: Path) -> None:
    all_points: list[Point] = []
    sessions: list[dict[str, object]] = []

    for index, csv_path in enumerate(coverage_csvs, start=1):
        points, meta = read_coverage_points(csv_path)
        session_name = session_name_for_csv(input_path, csv_path, index)
        session_out = out_dir / "sessions" / session_name
        write_outputs(points, session_out, csv_path, "room_raw_coverage_voxels", meta)
        all_points.extend(points)
        sessions.append(
            {
                "index": index,
                "name": session_name,
                "sourcePath": str(csv_path),
                "outputPath": str(session_out),
                "pointCount": len(points),
                "stableCount": sum(1 for p in points if p.stable and not p.risk),
                "riskCount": sum(1 for p in points if p.risk),
                "unstableCount": sum(1 for p in points if not p.stable and not p.risk),
                "bounds": bounds(points),
            }
        )

    combined_out = out_dir / "combined"
    write_outputs(
        all_points,
        combined_out,
        input_path,
        "room_raw_coverage_voxels_batch",
        {"sessionCount": str(len(coverage_csvs))},
    )

    batch_report = {
        "sourceRoot": str(input_path),
        "sessionCount": len(coverage_csvs),
        "totalPointCount": len(all_points),
        "combinedOutputPath": str(combined_out),
        "sessions": sessions,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "batch_coverage_summary.json").write_text(
        json.dumps(batch_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    coverage_csvs = find_coverage_csvs(input_path)
    meta: dict[str, str] = {}

    if len(coverage_csvs) > 1:
        out_dir = Path(args.out).expanduser().resolve() if args.out else default_batch_output_dir(input_path).resolve()
        write_batch_outputs(input_path, coverage_csvs, out_dir)
        print(f"[ScanCoverCoverageToPly] batchSource={input_path}")
        print(f"[ScanCoverCoverageToPly] sessions={len(coverage_csvs)}")
        print(f"[ScanCoverCoverageToPly] out={out_dir}")
        return 0

    coverage_csv = coverage_csvs[0] if coverage_csvs else find_coverage_csv(input_path)

    if coverage_csv is not None:
        points, meta = read_coverage_points(coverage_csv)
        source = coverage_csv
        source_type = "room_raw_coverage_voxels"
    elif args.vertices_fallback:
        vertex_csvs = find_vertices_csvs(input_path)
        if not vertex_csvs:
            raise FileNotFoundError(f"No room_raw_coverage_voxels.csv or *_vertices.csv found under {input_path}")
        points = read_vertex_points(vertex_csvs)
        meta = {"vertexCsvCount": str(len(vertex_csvs))}
        source = input_path
        source_type = "bl_surface_vertices"
    else:
        raise FileNotFoundError(
            f"No room_raw_coverage_voxels.csv found under {input_path}. "
            "Use --vertices-fallback only if you intentionally want BL mesh vertices instead."
        )

    out_dir = Path(args.out).expanduser().resolve() if args.out else default_output_dir(input_path, coverage_csv).resolve()
    write_outputs(points, out_dir, source, source_type, meta)

    print(f"[ScanCoverCoverageToPly] source={source}")
    print(f"[ScanCoverCoverageToPly] type={source_type} points={len(points)}")
    print(f"[ScanCoverCoverageToPly] out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
