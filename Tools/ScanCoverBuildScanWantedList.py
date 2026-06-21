#!/usr/bin/env python3
"""Build raw-depth scan diagnostics with explicit meanings.

This tool intentionally separates two concepts that were previously mixed:

- bad_observation_*: Raw Depth already saw the region, but current rules reject
  it because of unstable depth, projection error, low confidence, oblique view,
  source risk, or insufficient repeated hits. This diagnoses bad observations
  and rule/fusion problems. It is not direct proof that the region was not
  scanned.
- missing_scan_*: an optional reference surface has geometry here, but full Raw
  coverage has no nearby point. This is the direct rescan-target layer.
- processing_gap_*: full Raw coverage has points here, but the current usable
  mapping/candidate layers have no nearby point. This is a rule/fusion repair
  target, not a rescan target.

Legacy wanted_* files are now aliases of missing_scan_* when a reference surface
is provided. Without a reference surface there is no direct wanted/rescan layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_INPUT = Path(
    r"D:\PCA\Unity-MRMotifs-ScanCover-main"
    r"\ScanCoverExports\RepeatCoverageSessions\RepeatCoverageSessions"
    r"\raw_depth_mapping_input_stride5_neighbor_relaxed"
)

REASON_FILES = {
    "depth_unstable": ("risk_reason_depth_unstable.ply", 1.00, (255, 40, 40)),
    "projection_error": ("risk_reason_projection_error.ply", 1.35, (255, 130, 20)),
    "oblique_view": ("risk_reason_oblique_view.ply", 1.10, (255, 230, 40)),
    "low_confidence": ("risk_reason_low_confidence.ply", 1.50, (210, 40, 255)),
    "source_risk_flag": ("risk_reason_source_flag.ply", 0.85, (80, 120, 255)),
    "insufficient_observed": ("insufficient_coverage.ply", 1.25, (80, 255, 255)),
}

RAW_COVERAGE_CANDIDATE_NAMES = (
    "raw_depth_voxel_all_by_status.ply",
    "coverage_all_by_status.ply",
    "coverage_preview.ply",
)

CONTEXT_FILES = (
    "raw_depth_voxel_all_by_status.ply",
    "coverage_all_by_status.ply",
    "coverage_preview.ply",
    "mapping_input_candidate.ply",
    "mapping_input_stable_all.ply",
    "trusted_mapping.ply",
    "mapping_input_soft_candidate.ply",
)

USABLE_CONTEXT_FILES = (
    "mapping_input_candidate.ply",
    "mapping_input_stable_all.ply",
    "trusted_mapping.ply",
    "coverage_stable_all.ply",
    "mapping_input_soft_candidate.ply",
    "mapping_input_neighbor_candidate.ply",
)

PRIORITY_COLOR = {
    "S": (255, 0, 0),
    "A": (255, 140, 0),
    "B": (255, 240, 0),
    "C": (90, 220, 255),
}

MISSING_COLOR = (255, 0, 255)

REFERENCE_CANDIDATE_NAMES = (
    "meta_reference_sample.ply",
    "meta_reference_by_raw_distance.ply",
    "meta_scene_reference_sample.ply",
)

REFERENCE_CANDIDATE_DIRS = (
    "room_raw_meta_fusion_layers",
    "room_raw_meta_overlay_aligned",
    "room_raw_meta_overlay",
    "meta_surface_correction_reference_layer",
)


@dataclass
class DiagnosticCell:
    key: tuple[int, int, int]
    voxel: float
    sx: float = 0.0
    sy: float = 0.0
    sz: float = 0.0
    count: int = 0
    score: float = 0.0
    reasons: Counter = field(default_factory=Counter)

    def add(self, x: float, y: float, z: float, reason: str, weight: float) -> None:
        self.sx += x
        self.sy += y
        self.sz += z
        self.count += 1
        self.score += weight
        self.reasons[reason] += 1

    @property
    def center(self) -> tuple[float, float, float]:
        inv = 1.0 / max(1, self.count)
        return self.sx * inv, self.sy * inv, self.sz * inv

    @property
    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        x, y, z = self.key
        return (
            (x * self.voxel, y * self.voxel, z * self.voxel),
            ((x + 1) * self.voxel, (y + 1) * self.voxel, (z + 1) * self.voxel),
        )


@dataclass
class DiagnosticCluster:
    cells: list[DiagnosticCell]
    index: int = 0

    @property
    def count(self) -> int:
        return sum(c.count for c in self.cells)

    @property
    def score(self) -> float:
        return sum(c.score for c in self.cells)

    @property
    def reasons(self) -> Counter:
        out: Counter = Counter()
        for cell in self.cells:
            out.update(cell.reasons)
        return out

    @property
    def dominant_reason(self) -> str:
        reasons = self.reasons
        return reasons.most_common(1)[0][0] if reasons else "unknown"

    @property
    def center(self) -> tuple[float, float, float]:
        total = max(1, self.count)
        return (
            sum(c.sx for c in self.cells) / total,
            sum(c.sy for c in self.cells) / total,
            sum(c.sz for c in self.cells) / total,
        )

    @property
    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        mins = []
        maxs = []
        for axis in range(3):
            mins.append(min(c.bounds[0][axis] for c in self.cells))
            maxs.append(max(c.bounds[1][axis] for c in self.cells))
        return (mins[0], mins[1], mins[2]), (maxs[0], maxs[1], maxs[2])

    @property
    def size(self) -> tuple[float, float, float]:
        mins, maxs = self.bounds
        return tuple(maxs[i] - mins[i] for i in range(3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--target-voxel", type=float, default=0.35, help="Meters per diagnostic target block.")
    parser.add_argument(
        "--grouping",
        choices=("grid", "connected"),
        default="grid",
        help="grid creates separate target blocks; connected merges touching blocks.",
    )
    parser.add_argument("--min-cluster-points", type=int, default=450, help="Ignore smaller target blocks.")
    parser.add_argument("--max-targets", type=int, default=24)
    parser.add_argument("--box-padding", type=float, default=0.05)
    parser.add_argument("--box-step", type=float, default=0.04)
    parser.add_argument(
        "--reference-ply",
        type=Path,
        default=None,
        help=(
            "Optional Meta/reference surface sample PLY for true missing-scan diagnostics. "
            "If omitted, the tool tries to find meta_reference_sample.ply near the input session."
        ),
    )
    parser.add_argument(
        "--reference-coverage-radius",
        type=float,
        default=0.10,
        help="Reference points farther than this from any full Raw coverage point are treated as missing.",
    )
    parser.add_argument(
        "--include-usable-risk",
        action="store_true",
        help="Include risk points even when they overlap current usable mapping context. Off by default.",
    )
    parser.add_argument(
        "--usable-match-radius",
        type=float,
        default=0.035,
        help="Risk points near current usable context are treated as usable, not bad observations.",
    )
    parser.add_argument(
        "--processing-gap-radius",
        type=float,
        default=0.08,
        help=(
            "Full Raw points farther than this from any usable candidate point are "
            "reported as processing_gap rule/fusion repair targets."
        ),
    )
    return parser.parse_args()


def resolve_reference_ply(input_dir: Path, requested: Path | None) -> tuple[Path | None, list[str]]:
    notes: list[str] = []
    if requested is not None:
        requested = requested.resolve()
        if requested.exists():
            notes.append(f"using explicit reference: {requested}")
            return requested, notes
        notes.append(f"explicit reference missing: {requested}")

    search_roots: list[Path] = []
    for root in (input_dir, *input_dir.parents):
        search_roots.append(root)
        if root.name == "ScanCoverExports":
            break

    candidates: list[Path] = []
    for root in search_roots:
        for dirname in REFERENCE_CANDIDATE_DIRS:
            for filename in REFERENCE_CANDIDATE_NAMES:
                candidates.append(root / dirname / filename)
        for filename in REFERENCE_CANDIDATE_NAMES:
            candidates.append(root / filename)

    for candidate in candidates:
        if candidate.exists():
            notes.append(f"auto reference found: {candidate}")
            return candidate.resolve(), notes

    # Last chance: search only one directory level below nearby session roots.
    # Avoid a full recursive scan of room_raw_depth_frames, which can be tens of GB.
    for root in search_roots[:4]:
        try:
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                for filename in REFERENCE_CANDIDATE_NAMES:
                    candidate = child / filename
                    if candidate.exists():
                        notes.append(f"auto reference found in sibling dir: {candidate}")
                        return candidate.resolve(), notes
        except OSError as exc:
            notes.append(f"reference search skipped {root}: {exc}")

    notes.append(
        "no reference PLY found near input; missing_scan/wanted requires "
        "--reference-ply pointing to meta_reference_sample.ply or equivalent."
    )
    return None, notes


def resolve_named_ply(input_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = input_dir / name
        if path.exists():
            return path
    return None


def resolve_raw_coverage_ply(input_dir: Path) -> Path | None:
    return resolve_named_ply(input_dir, RAW_COVERAGE_CANDIDATE_NAMES)


def read_ply_points(path: Path) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        vertex_count = 0
        for line in f:
            line = line.strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                break
        for index, line in enumerate(f):
            if vertex_count and index >= vertex_count:
                break
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                points.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    return points


def quantize(point: tuple[float, float, float], voxel: float) -> tuple[int, int, int]:
    return tuple(math.floor(v / voxel) for v in point)


def build_cells_from_points(
    points: list[tuple[float, float, float]],
    reason: str,
    weight: float,
    voxel: float,
) -> dict[tuple[int, int, int], DiagnosticCell]:
    cells: dict[tuple[int, int, int], DiagnosticCell] = {}
    for x, y, z in points:
        key = quantize((x, y, z), voxel)
        cell = cells.get(key)
        if cell is None:
            cell = DiagnosticCell(key, voxel)
            cells[key] = cell
        cell.add(x, y, z, reason, weight)
    return cells


def build_bad_observation_cells(
    input_dir: Path,
    voxel: float,
    include_usable_risk: bool,
    usable_match_radius: float,
) -> dict[tuple[int, int, int], DiagnosticCell]:
    usable_keys: set[tuple[int, int, int]] = set()
    usable_cell = max(0.01, voxel)
    if not include_usable_risk:
        for filename in USABLE_CONTEXT_FILES:
            path = input_dir / filename
            if not path.exists():
                continue
            usable_keys.update(quantize(point, usable_cell) for point in read_ply_points(path))

    def overlaps_usable(point: tuple[float, float, float]) -> bool:
        if not usable_keys:
            return False
        qx, qy, qz = quantize(point, usable_cell)
        radius_cells = max(1, math.ceil(usable_match_radius / usable_cell))
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                for dz in range(-radius_cells, radius_cells + 1):
                    if (qx + dx, qy + dy, qz + dz) in usable_keys:
                        return True
        return False

    cells: dict[tuple[int, int, int], DiagnosticCell] = {}
    for reason, (filename, weight, _color) in REASON_FILES.items():
        path = input_dir / filename
        if not path.exists():
            continue
        for x, y, z in read_ply_points(path):
            if overlaps_usable((x, y, z)):
                continue
            key = quantize((x, y, z), voxel)
            cell = cells.get(key)
            if cell is None:
                cell = DiagnosticCell(key, voxel)
                cells[key] = cell
            cell.add(x, y, z, reason, weight)
    return cells


def reference_missing_points(input_dir: Path, reference_ply: Path, radius: float) -> list[tuple[float, float, float]]:
    raw_path = resolve_raw_coverage_ply(input_dir)
    if raw_path is None:
        candidates = ", ".join(RAW_COVERAGE_CANDIDATE_NAMES)
        raise FileNotFoundError(f"Missing Raw coverage layer in {input_dir}. Tried: {candidates}")
    if not reference_ply.exists():
        raise FileNotFoundError(f"Missing reference PLY: {reference_ply}")

    # True coverage must be compared against full Raw evidence, not candidate.
    cell = max(0.01, radius)
    raw_keys = {quantize(point, cell) for point in read_ply_points(raw_path)}

    def has_raw_neighbor(point: tuple[float, float, float]) -> bool:
        qx, qy, qz = quantize(point, cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if (qx + dx, qy + dy, qz + dz) in raw_keys:
                        return True
        return False

    return [point for point in read_ply_points(reference_ply) if not has_raw_neighbor(point)]


def processing_gap_points(input_dir: Path, radius: float) -> list[tuple[float, float, float]]:
    raw_path = resolve_raw_coverage_ply(input_dir)
    if raw_path is None:
        return []

    cell = max(0.01, radius)
    usable_keys: set[tuple[int, int, int]] = set()
    for filename in USABLE_CONTEXT_FILES:
        path = input_dir / filename
        if not path.exists():
            continue
        usable_keys.update(quantize(point, cell) for point in read_ply_points(path))

    if not usable_keys:
        return read_ply_points(raw_path)

    def has_usable_neighbor(point: tuple[float, float, float]) -> bool:
        qx, qy, qz = quantize(point, cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if (qx + dx, qy + dy, qz + dz) in usable_keys:
                        return True
        return False

    return [point for point in read_ply_points(raw_path) if not has_usable_neighbor(point)]


def neighbors(key: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    x, y, z = key
    return [
        (x - 1, y, z), (x + 1, y, z),
        (x, y - 1, z), (x, y + 1, z),
        (x, y, z - 1), (x, y, z + 1),
    ]


def connected_clusters(cells: dict[tuple[int, int, int], DiagnosticCell], min_points: int) -> list[DiagnosticCluster]:
    remaining = set(cells)
    clusters: list[DiagnosticCluster] = []
    while remaining:
        start = remaining.pop()
        queue: deque[tuple[int, int, int]] = deque([start])
        group = [cells[start]]
        while queue:
            key = queue.popleft()
            for nk in neighbors(key):
                if nk not in remaining:
                    continue
                remaining.remove(nk)
                queue.append(nk)
                group.append(cells[nk])
        cluster = DiagnosticCluster(group)
        if cluster.count >= min_points:
            clusters.append(cluster)
    return sorted(clusters, key=lambda c: (c.score, c.count), reverse=True)


def grid_clusters(cells: dict[tuple[int, int, int], DiagnosticCell], min_points: int) -> list[DiagnosticCluster]:
    clusters = [DiagnosticCluster([cell]) for cell in cells.values() if cell.count >= min_points]
    return sorted(clusters, key=lambda c: (c.score, c.count), reverse=True)


def priority_for(cluster: DiagnosticCluster) -> str:
    if len(cluster.cells) == 1:
        if cluster.count >= 2200 or cluster.score >= 2600:
            return "S"
        if cluster.count >= 1500 or cluster.score >= 1800:
            return "A"
        if cluster.count >= 800 or cluster.score >= 950:
            return "B"
        return "C"

    sx, sy, sz = cluster.size
    if cluster.count >= 25000 or cluster.score >= 30000 or max(sx, sy, sz) >= 1.20:
        return "S"
    if cluster.count >= 8000 or cluster.score >= 10000 or max(sx, sy, sz) >= 0.70:
        return "A"
    if cluster.count >= 2500 or cluster.score >= 3000:
        return "B"
    return "C"


def instruction_for(cluster: DiagnosticCluster, layer: str) -> str:
    reason = cluster.dominant_reason
    cx, cy, cz = cluster.center
    height = "high/ceiling" if cy > 2.0 else "low/floor/under-object" if cy < 0.45 else "mid-height"
    if layer == "missing_scan":
        action = "true rescan target: reference exists here but full Raw coverage is missing."
    elif layer == "processing_gap":
        action = "rule/fusion repair target: full Raw exists here but current usable mapping candidate rejects it."
    elif reason == "insufficient_observed":
        action = "observed but repeated hits are insufficient; rescan slowly only if this region matters."
    elif reason == "projection_error":
        action = "observed with high projection error; retry from a more frontal view or tune fusion rules."
    elif reason == "oblique_view":
        action = "observed from an oblique view; retry from a more frontal angle."
    elif reason == "depth_unstable":
        action = "observed but depth is unstable; diagnose fusion/risk rules before calling it missing."
    elif reason == "low_confidence":
        action = "observed with low confidence; retry closer/brighter if this surface matters."
    else:
        action = "diagnostic region; inspect with full Raw coverage and reference layers."
    return f"{height}; center=({cx:.2f}, {cy:.2f}, {cz:.2f}); {action}"


def write_ply(path: Path, rows: list[tuple[float, float, float, int, int, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in rows:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def write_context_overlay(
    input_dir: Path,
    out: Path,
    name: str,
    centers: list[tuple[float, float, float, int, int, int]],
    boxes: list[tuple[float, float, float, int, int, int]],
) -> None:
    rows: list[tuple[float, float, float, int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for file_index, filename in enumerate(CONTEXT_FILES):
        path = input_dir / filename
        if not path.exists():
            continue
        # Full Raw coverage is dark gray; current usable layers are lighter gray.
        color = (70, 70, 70) if file_index == 0 else (135, 135, 135)
        quant = 0.025 if file_index == 0 else 0.015
        for x, y, z in read_ply_points(path):
            key = (round(x / quant), round(y / quant), round(z / quant))
            if key in seen:
                continue
            seen.add(key)
            rows.append((x, y, z, *color))
    rows.extend(boxes)
    for x, y, z, r, g, b in centers:
        rows.append((x, y, z, r, g, b))
        for dx, dy, dz in ((0.03, 0, 0), (-0.03, 0, 0), (0, 0.03, 0), (0, -0.03, 0), (0, 0, 0.03), (0, 0, -0.03)):
            rows.append((x + dx, y + dy, z + dz, r, g, b))
    write_ply(out / name, rows)


def box_edge_points(cluster: DiagnosticCluster, padding: float, step: float) -> list[tuple[float, float, float]]:
    mins, maxs = cluster.bounds
    minx, miny, minz = (mins[0] - padding, mins[1] - padding, mins[2] - padding)
    maxx, maxy, maxz = (maxs[0] + padding, maxs[1] + padding, maxs[2] + padding)
    corners = [
        (minx, miny, minz), (maxx, miny, minz), (maxx, maxy, minz), (minx, maxy, minz),
        (minx, miny, maxz), (maxx, miny, maxz), (maxx, maxy, maxz), (minx, maxy, maxz),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    pts: list[tuple[float, float, float]] = []
    for a, b in edges:
        ax, ay, az = corners[a]
        bx, by, bz = corners[b]
        length = math.dist(corners[a], corners[b])
        count = max(2, int(length / max(0.01, step)) + 1)
        for i in range(count):
            t = i / max(1, count - 1)
            pts.append((ax + (bx - ax) * t, ay + (by - ay) * t, az + (bz - az) * t))
    return pts


def write_cluster_outputs(
    clusters: list[DiagnosticCluster],
    out: Path,
    args: argparse.Namespace,
    prefix: str,
    legacy_wanted_alias: bool = False,
) -> list[dict[str, object]]:
    selected = clusters[: args.max_targets]
    summary_rows: list[dict[str, object]] = []
    centers: list[tuple[float, float, float, int, int, int]] = []
    boxes: list[tuple[float, float, float, int, int, int]] = []

    stale_names = [
        f"{prefix}_target_centers.ply",
        f"{prefix}_target_boxes.ply",
        f"{prefix}_overlay_with_mapping_context.ply",
        f"{prefix}_targets.csv",
        f"{prefix}_targets.json",
    ]
    if legacy_wanted_alias:
        stale_names.extend([
            "wanted_target_centers.ply",
            "wanted_target_boxes.ply",
            "wanted_overlay_with_mapping_context.ply",
            "scan_wanted_targets.csv",
            "scan_wanted_targets.json",
        ])
    for stale_name in stale_names:
        stale_path = out / stale_name
        if stale_path.exists():
            stale_path.unlink()
    for stale in out.glob(f"{prefix}_target_*.ply"):
        stale.unlink()
    if legacy_wanted_alias:
        for stale in out.glob("wanted_target_*.ply"):
            stale.unlink()

    for cluster in selected:
        priority = priority_for(cluster)
        color = MISSING_COLOR if prefix == "missing_scan" else PRIORITY_COLOR[priority]
        reason = cluster.dominant_reason
        cx, cy, cz = cluster.center
        sx, sy, sz = cluster.size
        centers.append((cx, cy, cz, *color))
        for px, py, pz in box_edge_points(cluster, args.box_padding, args.box_step):
            boxes.append((px, py, pz, *color))

        reason_counts = dict(cluster.reasons)
        summary_rows.append({
            "target": f"T{cluster.index:02d}",
            "priority": priority,
            "dominantReason": reason,
            "pointCount": cluster.count,
            "score": round(cluster.score, 2),
            "centerX": round(cx, 4),
            "centerY": round(cy, 4),
            "centerZ": round(cz, 4),
            "sizeX": round(sx, 4),
            "sizeY": round(sy, 4),
            "sizeZ": round(sz, 4),
            "reasonCounts": json.dumps(reason_counts, ensure_ascii=False),
            "instruction": instruction_for(cluster, prefix),
        })

        target_color = MISSING_COLOR if prefix == "missing_scan" else REASON_FILES.get(reason, ("", 1.0, color))[2]
        target_points = []
        for cell in cluster.cells:
            x, y, z = cell.center
            target_points.append((x, y, z, *target_color))
        write_ply(out / f"{prefix}_target_{cluster.index:02d}_{priority}_{reason}.ply", target_points)

    write_ply(out / f"{prefix}_target_centers.ply", centers)
    write_ply(out / f"{prefix}_target_boxes.ply", boxes)
    write_context_overlay(args.input.resolve(), out, f"{prefix}_overlay_with_mapping_context.ply", centers, boxes)

    if summary_rows:
        with (out / f"{prefix}_targets.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    (out / f"{prefix}_targets.json").write_text(
        json.dumps({"targetCount": len(summary_rows), "targets": summary_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if legacy_wanted_alias:
        write_ply(out / "wanted_target_centers.ply", centers)
        write_ply(out / "wanted_target_boxes.ply", boxes)
        write_context_overlay(args.input.resolve(), out, "wanted_overlay_with_mapping_context.ply", centers, boxes)
        if summary_rows:
            with (out / "scan_wanted_targets.csv").open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
        (out / "scan_wanted_targets.json").write_text(
            json.dumps(
                {
                    "semanticWarning": "Legacy alias. These are true missing-scan clusters derived from reference-vs-Raw coverage.",
                    "targetCount": len(summary_rows),
                    "targets": summary_rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return summary_rows


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    reference_ply, reference_notes = resolve_reference_ply(input_dir, args.reference_ply)
    out = args.out.resolve() if args.out else input_dir / "scan_wanted_list"
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("wanted_target_*.ply"):
        stale.unlink()
    for stale_name in (
        "wanted_overlay_with_mapping_context.ply",
        "scan_wanted_targets.csv",
        "scan_wanted_targets.json",
    ):
        stale_path = out / stale_name
        if stale_path.exists():
            stale_path.unlink()

    bad_cells = build_bad_observation_cells(
        input_dir,
        args.target_voxel,
        args.include_usable_risk,
        args.usable_match_radius,
    )
    bad_clusters = grid_clusters(bad_cells, args.min_cluster_points) if args.grouping == "grid" else connected_clusters(bad_cells, args.min_cluster_points)
    for index, cluster in enumerate(bad_clusters, start=1):
        cluster.index = index
    bad_rows = write_cluster_outputs(bad_clusters, out, args, "bad_observation", legacy_wanted_alias=False)

    gap_points = processing_gap_points(input_dir, args.processing_gap_radius)
    gap_cells = build_cells_from_points(gap_points, "processing_gap", 1.0, args.target_voxel)
    gap_clusters = grid_clusters(gap_cells, args.min_cluster_points) if args.grouping == "grid" else connected_clusters(gap_cells, args.min_cluster_points)
    for index, cluster in enumerate(gap_clusters, start=1):
        cluster.index = index
    gap_rows = write_cluster_outputs(gap_clusters, out, args, "processing_gap", legacy_wanted_alias=False)

    missing_rows: list[dict[str, object]] = []
    missing_point_count = 0
    if reference_ply is not None:
        missing_points = reference_missing_points(input_dir, reference_ply, args.reference_coverage_radius)
        missing_point_count = len(missing_points)
        missing_cells = build_cells_from_points(missing_points, "missing_reference", 1.25, args.target_voxel)
        missing_clusters = grid_clusters(missing_cells, args.min_cluster_points) if args.grouping == "grid" else connected_clusters(missing_cells, args.min_cluster_points)
        for index, cluster in enumerate(missing_clusters, start=1):
            cluster.index = index
        missing_rows = write_cluster_outputs(missing_clusters, out, args, "missing_scan", legacy_wanted_alias=True)

    report = {
        "input": str(input_dir),
        "referencePly": str(reference_ply) if reference_ply else None,
        "referenceResolutionNotes": reference_notes,
        "semantics": {
            "bad_observation": "Raw-covered voxels rejected by current rules. Use for rule/fusion diagnostics.",
            "processing_gap": "Full Raw-covered voxels not represented in current usable mapping candidates. Use for rule/fusion repair targets.",
            "missing_scan": "Reference-surface points not covered by full Raw coverage. Use as direct rescan targets.",
            "wanted_legacy": "Alias of missing_scan when --reference-ply is provided; otherwise no direct wanted layer exists.",
        },
        "badObservationClusterCount": len(bad_rows),
        "processingGapPointCount": len(gap_points),
        "processingGapClusterCount": len(gap_rows),
        "missingReferencePointCount": missing_point_count,
        "missingScanClusterCount": len(missing_rows),
    }
    (out / "diagnostic_semantics_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[diagnostics] input={input_dir}")
    print(f"[diagnostics] bad_observation cells={len(bad_cells)} clusters={len(bad_clusters)} out={out}")
    print(f"[diagnostics] processing_gap points={len(gap_points)} clusters={len(gap_rows)}")
    for note in reference_notes:
        print(f"[diagnostics] reference: {note}")
    if reference_ply is None:
        print("[diagnostics] missing_scan skipped: no Meta/reference PLY was found.")
    else:
        print(f"[diagnostics] missing_scan reference_points_without_raw={missing_point_count} clusters={len(missing_rows)}")
    for cluster in bad_clusters[: min(args.max_targets, 12)]:
        print(
            f"[bad-observation] T{cluster.index:02d} priority={priority_for(cluster)} "
            f"reason={cluster.dominant_reason} points={cluster.count} "
            f"center=({cluster.center[0]:.2f},{cluster.center[1]:.2f},{cluster.center[2]:.2f})"
        )


if __name__ == "__main__":
    main()
