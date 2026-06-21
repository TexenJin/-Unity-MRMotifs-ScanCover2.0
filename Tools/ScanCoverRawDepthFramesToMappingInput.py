#!/usr/bin/env python3
"""Build mapping-input diagnostics directly from dense room_raw_depth_frames.

This tool intentionally bypasses room_raw_coverage_voxels.csv. It consumes the
per-frame dense Raw Depth CSV files exported by ScanCoverMultiFrameSessionExporter
and builds compact voxel statistics suitable for:

- checking whether the dense Raw Depth capture is healthy;
- creating a practical mapping-input point cloud;
- separating stable, review, risk, and rejected samples.

It does not create the final mapping mesh. It produces the point-cloud inputs and
reports needed before meshing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path(
    r"D:\PCA\Unity-MRMotifs-ScanCover-main"
    r"\ScanCoverExports\RepeatCoverageSessions\RepeatCoverageSessions"
)


@dataclass
class VoxelStats:
    sx: float = 0.0
    sy: float = 0.0
    sz: float = 0.0
    snx: float = 0.0
    sny: float = 0.0
    snz: float = 0.0
    sdepth: float = 0.0
    sdepth2: float = 0.0
    sangle: float = 0.0
    sprojection: float = 0.0
    sconfidence: float = 0.0
    hits: int = 0
    risk_hits: int = 0
    focus_hits: int = 0
    high_hits: int = 0
    low_hits: int = 0
    session_mask: int = 0

    def add(
        self,
        x: float,
        y: float,
        z: float,
        nx: float,
        ny: float,
        nz: float,
        depth: float,
        angle: float,
        projection: float,
        confidence: float,
        risk: bool,
        focus: bool,
        high: bool,
        low: bool,
        session_index: int,
    ) -> None:
        self.sx += x
        self.sy += y
        self.sz += z
        self.snx += nx
        self.sny += ny
        self.snz += nz
        self.sdepth += depth
        self.sdepth2 += depth * depth
        self.sangle += angle
        self.sprojection += projection
        self.sconfidence += confidence
        self.hits += 1
        self.risk_hits += 1 if risk else 0
        self.focus_hits += 1 if focus else 0
        self.high_hits += 1 if high else 0
        self.low_hits += 1 if low else 0
        if 0 <= session_index < 63:
            self.session_mask |= 1 << session_index

    def row(self) -> dict[str, float | int]:
        inv = 1.0 / max(1, self.hits)
        x = self.sx * inv
        y = self.sy * inv
        z = self.sz * inv
        nx = self.snx * inv
        ny = self.sny * inv
        nz = self.snz * inv
        n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
        if n_len > 1e-6:
            nx /= n_len
            ny /= n_len
            nz /= n_len
        depth_mean = self.sdepth * inv
        depth_var = max(0.0, self.sdepth2 * inv - depth_mean * depth_mean)
        return {
            "x": x,
            "y": y,
            "z": z,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "hits": self.hits,
            "sessions": self.session_mask.bit_count(),
            "riskRatio": self.risk_hits * inv,
            "focusRatio": self.focus_hits * inv,
            "highRatio": self.high_hits * inv,
            "lowRatio": self.low_hits * inv,
            "depthMean": depth_mean,
            "depthStd": math.sqrt(depth_var),
            "angleMean": self.sangle * inv,
            "projectionMean": self.sprojection * inv,
            "confidenceMean": self.sconfidence * inv,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert dense ScanCover room_raw_depth_frames into mapping input PLY diagnostics."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help="Folder containing ScanCover_RepeatCoverage_* sessions, a single session, or room_raw_depth_frames.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output folder.")
    parser.add_argument("--voxel", type=float, default=0.025, help="Voxel size in meters for dense mapping input.")
    parser.add_argument("--preview-voxel", type=float, default=0.08, help="Additional coarse preview voxel size.")
    parser.add_argument("--min-depth", type=float, default=0.20)
    parser.add_argument("--max-depth", type=float, default=5.00)
    parser.add_argument("--max-sessions", type=int, default=0, help="0 means all sessions.")
    parser.add_argument("--max-frames-per-session", type=int, default=0, help="0 means all frames.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--stable-hits", type=int, default=10)
    parser.add_argument("--stable-sessions", type=int, default=1)
    parser.add_argument("--risk-ratio", type=float, default=0.35)
    parser.add_argument("--angle-risk", type=float, default=72.0)
    parser.add_argument("--projection-risk", type=float, default=0.20)
    parser.add_argument("--depth-std-risk", type=float, default=0.075)
    parser.add_argument("--candidate-depth-std", type=float, default=0.11)
    parser.add_argument("--usable-depth-std", type=float, default=0.18, help="Preferred depth std for clean Raw points allowed into the mapping reference layer.")
    parser.add_argument("--usable-projection-risk", type=float, default=0.35, help="Preferred projection-error gate for clean mapping evidence.")
    parser.add_argument("--usable-confidence", type=float, default=0.30, help="Preferred confidence gate for clean mapping evidence.")
    parser.add_argument("--hard-depth-std", type=float, default=0.45, help="True rejection depth std. Above this, a point is too unstable for mapping input.")
    parser.add_argument("--hard-projection-risk", type=float, default=0.60, help="True rejection projection error. Above this, a point is too inconsistent for mapping input.")
    parser.add_argument("--hard-confidence", type=float, default=0.20, help="True rejection confidence. Below this, a point is too weak for mapping input.")
    parser.add_argument("--dense-hit-depth-std", type=float, default=0.35, help="Depth std allowed when repeated evidence is dense enough.")
    parser.add_argument("--dense-hit-multiplier", type=float, default=2.0, help="stable-hits multiplier for dense-evidence rescue.")
    parser.add_argument("--candidate-risk-ratio", type=float, default=0.70, help="Diagnostic source-risk ratio; no longer hard-rejects mapping evidence by itself.")
    parser.add_argument("--candidate-angle-risk", type=float, default=84.0, help="Soft view angle allowed for candidate mapping evidence.")
    parser.add_argument("--neighbor-vote-radius", type=int, default=1, help="Voxel-neighborhood radius for same-surface rescue.")
    parser.add_argument("--neighbor-vote-threshold", type=float, default=0.55, help="Required same-surface usable-neighbor ratio.")
    parser.add_argument("--neighbor-min-usable", type=int, default=5, help="Minimum usable neighbors required for rescue.")
    parser.add_argument("--neighbor-normal-angle", type=float, default=28.0, help="Max normal angle in degrees for same-surface voting.")
    parser.add_argument("--neighbor-depth-delta", type=float, default=0.20, help="Max depth-mean delta in meters for same-surface voting.")
    parser.add_argument("--neighbor-rescue-depth-std", type=float, default=0.18, help="Max depth std for a risk voxel rescued by neighbors.")
    parser.add_argument("--surface-stabilize-radius", type=int, default=2, help="Voxel-neighborhood radius used to collapse repeated same-surface hits before mapping.")
    parser.add_argument("--surface-stabilize-min-support", type=int, default=7, help="Minimum same-surface support voxels required for a stabilized mapping point.")
    parser.add_argument("--surface-stabilize-normal-angle", type=float, default=26.0, help="Max normal angle in degrees for local surface stabilization.")
    parser.add_argument("--surface-stabilize-depth-delta", type=float, default=0.18, help="Max depth-mean delta in meters for local surface stabilization.")
    parser.add_argument("--surface-stabilize-max-depth-std", type=float, default=0.30, help="Reject stabilized candidates above this local voxel depth std.")
    parser.add_argument("--surface-stabilize-max-risk-ratio", type=float, default=0.80, help="Reject stabilized candidates above this source-risk ratio.")
    parser.add_argument("--surface-stabilize-plane-pull", type=float, default=1.0, help="0 keeps original points; 1 projects them fully onto the local support plane.")
    parser.add_argument("--write-csv", action="store_true", help="Also write voxel_stats.csv.")
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


def find_sessions(path: Path) -> list[Path]:
    if (path / "frame_0000_raw_depth.csv").exists() or path.name == "room_raw_depth_frames":
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


def iter_frame_csvs(session: Path, stride: int, max_frames: int) -> Iterable[Path]:
    raw_dir = session / "room_raw_depth_frames"
    files = sorted(raw_dir.glob("frame_*_raw_depth.csv"))
    if stride > 1:
        files = files[::stride]
    if max_frames > 0:
        files = files[:max_frames]
    return files


def voxel_key(x: float, y: float, z: float, voxel: float) -> tuple[int, int, int]:
    return (math.floor(x / voxel), math.floor(y / voxel), math.floor(z / voxel))


def add_csv_to_voxels(
    csv_path: Path,
    voxels: dict[tuple[int, int, int], VoxelStats],
    voxel: float,
    session_index: int,
    min_depth: float,
    max_depth: float,
    report: dict[str, int],
) -> None:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        first = fh.readline()
        second = fh.readline()
        if not first.startswith("#") or not second.startswith("resolution="):
            fh.seek(0)
        reader = csv.DictReader(fh)
        for row in reader:
            report["rawRows"] += 1
            depth = to_float(row.get("depthM"), -1.0)
            if not math.isfinite(depth) or depth < min_depth or depth > max_depth:
                report["rejectedDepthRows"] += 1
                continue
            x = to_float(row.get("worldX"), math.nan)
            y = to_float(row.get("worldY"), math.nan)
            z = to_float(row.get("worldZ"), math.nan)
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                report["rejectedInvalidRows"] += 1
                continue
            nx = to_float(row.get("normalX"), 0.0)
            ny = to_float(row.get("normalY"), 0.0)
            nz = to_float(row.get("normalZ"), 0.0)
            key = voxel_key(x, y, z, voxel)
            stats = voxels.get(key)
            if stats is None:
                stats = VoxelStats()
                voxels[key] = stats
            stats.add(
                x=x,
                y=y,
                z=z,
                nx=nx,
                ny=ny,
                nz=nz,
                depth=depth,
                angle=to_float(row.get("viewAngleDeg"), 0.0),
                projection=to_float(row.get("projectionErrorM"), 0.0),
                confidence=to_float(row.get("confidence"), 1.0),
                risk=truthy(row.get("risk")),
                focus=truthy(row.get("focus")),
                high=truthy(row.get("high")),
                low=truthy(row.get("low")),
                session_index=session_index,
            )
            report["acceptedRows"] += 1


def hard_limits(args: argparse.Namespace) -> tuple[float, float, float]:
    hard_depth = max(float(args.usable_depth_std), float(args.hard_depth_std))
    hard_projection = max(float(args.usable_projection_risk), float(args.hard_projection_risk))
    hard_confidence = min(float(args.usable_confidence), float(args.hard_confidence))
    return hard_depth, hard_projection, hard_confidence


def status_for(row: dict[str, float | int], args: argparse.Namespace) -> str:
    hits = int(row["hits"])
    sessions = int(row["sessions"])
    projection = float(row["projectionMean"])
    depth_std = float(row["depthStd"])
    confidence = float(row["confidenceMean"])

    if hits < args.min_hits:
        return "insufficient"

    hard_depth, hard_projection, hard_confidence = hard_limits(args)

    # Hard rejection is reserved for true geometry instability. Oblique view
    # and source risk are diagnostic tags; rejecting them here made dense and
    # repeated Raw evidence look like missing scan data.
    if confidence < hard_confidence or projection > hard_projection or depth_std > hard_depth:
        return "risk"

    if (
        hits >= args.stable_hits
        and sessions >= args.stable_sessions
        and depth_std <= args.depth_std_risk
        and confidence >= 0.55
        and projection <= args.projection_risk
    ):
        return "stable"

    if (
        depth_std <= max(args.candidate_depth_std, 0.18)
        and projection <= min(hard_projection, max(args.usable_projection_risk, 0.45))
        and confidence >= max(hard_confidence, 0.30)
    ):
        return "candidate"

    dense_hit_gate = max(args.stable_hits, int(math.ceil(args.stable_hits * args.dense_hit_multiplier)))
    if (
        hits >= dense_hit_gate
        and depth_std <= min(hard_depth, max(args.dense_hit_depth_std, args.usable_depth_std))
        and projection <= hard_projection
        and confidence >= hard_confidence
    ):
        return "candidate"

    # Passed hard rejection but not clean enough to be called stable/candidate.
    # Keep it visible as review evidence instead of hiding it as????.
    return "review"

def risk_reason_for(row: dict[str, float | int], args: argparse.Namespace) -> str:
    """Return the dominant diagnostic reason for a voxel.

    Reasons are not equivalent to rejection. Oblique/source-risk observations can
    still be mapping evidence when repeated hits and local depth are consistent.
    """
    hits = int(row["hits"])
    risk_ratio = float(row["riskRatio"])
    angle = float(row["angleMean"])
    projection = float(row["projectionMean"])
    depth_std = float(row["depthStd"])
    confidence = float(row["confidenceMean"])
    hard_depth, hard_projection, hard_confidence = hard_limits(args)

    if hits < args.min_hits:
        return "insufficient"
    if depth_std > hard_depth:
        return "depth_unstable"
    if projection > hard_projection:
        return "projection_error"
    if confidence < hard_confidence:
        return "low_confidence"
    if angle > args.angle_risk:
        return "oblique_view"
    if risk_ratio > args.risk_ratio:
        return "source_risk_flag"
    return "none"


def soft_usable(row: dict[str, float | int], args: argparse.Namespace) -> bool:
    """Mapping-input gate for existing Raw evidence.

    This keeps repeated Raw observations unless they are truly unstable. It is
    intentionally wider than stable/candidate classification so the downstream
    mapping layer can use dense evidence before deciding final surfaces.
    """
    hits = int(row["hits"])
    depth_std = float(row["depthStd"])
    projection = float(row["projectionMean"])
    confidence = float(row["confidenceMean"])
    if hits < args.min_hits:
        return False
    hard_depth, hard_projection, hard_confidence = hard_limits(args)
    if confidence < hard_confidence:
        return False
    if projection > hard_projection:
        return False
    if depth_std > hard_depth:
        return False
    return True

def color_for(status: str, row: dict[str, float | int]) -> tuple[int, int, int]:
    if status == "stable":
        return (0, 235, 235)
    if status == "candidate":
        return (255, 220, 20)
    if status == "review":
        return (255, 135, 20)
    if status == "risk":
        return (255, 35, 35)
    return (80, 90, 105)


def reason_color(reason: str) -> tuple[int, int, int]:
    if reason == "depth_unstable":
        return (255, 35, 35)
    if reason == "projection_error":
        return (255, 130, 20)
    if reason == "low_confidence":
        return (170, 60, 255)
    if reason == "oblique_view":
        return (255, 225, 20)
    if reason == "source_risk_flag":
        return (30, 120, 255)
    if reason == "insufficient":
        return (90, 90, 90)
    return (0, 235, 235)


def same_surface_neighbor(
    row: dict[str, float | int],
    neighbor: dict[str, float | int],
    args: argparse.Namespace,
) -> bool:
    nx = float(row["nx"])
    ny = float(row["ny"])
    nz = float(row["nz"])
    nnx = float(neighbor["nx"])
    nny = float(neighbor["ny"])
    nnz = float(neighbor["nz"])
    dot = nx * nnx + ny * nny + nz * nnz
    normal_threshold = math.cos(math.radians(args.neighbor_normal_angle))
    if dot < normal_threshold:
        return False
    return abs(float(row["depthMean"]) - float(neighbor["depthMean"])) <= args.neighbor_depth_delta


def same_surface_for_stabilization(
    row: dict[str, float | int],
    neighbor: dict[str, float | int],
    args: argparse.Namespace,
) -> bool:
    nx = float(row["nx"])
    ny = float(row["ny"])
    nz = float(row["nz"])
    nnx = float(neighbor["nx"])
    nny = float(neighbor["ny"])
    nnz = float(neighbor["nz"])
    dot = nx * nnx + ny * nny + nz * nnz
    normal_threshold = math.cos(math.radians(args.surface_stabilize_normal_angle))
    if abs(dot) < normal_threshold:
        return False
    if abs(float(row["depthMean"]) - float(neighbor["depthMean"])) > args.surface_stabilize_depth_delta:
        return False
    if float(neighbor["depthStd"]) > args.surface_stabilize_max_depth_std:
        return False
    if float(neighbor["riskRatio"]) > args.surface_stabilize_max_risk_ratio:
        return False
    return True


def rescue_by_neighbor_vote(
    key: tuple[int, int, int],
    row: dict[str, float | int],
    rows_by_key: dict[tuple[int, int, int], dict[str, float | int]],
    args: argparse.Namespace,
) -> bool:
    if int(row["hits"]) < args.min_hits:
        return False
    hard_depth, hard_projection, hard_confidence = hard_limits(args)
    if float(row["confidenceMean"]) < hard_confidence:
        return False
    if float(row["depthStd"]) > max(args.neighbor_rescue_depth_std, min(hard_depth, args.dense_hit_depth_std)):
        return False
    if float(row["projectionMean"]) > hard_projection:
        return False

    radius = max(1, int(args.neighbor_vote_radius))
    considered = 0
    usable = 0
    kx, ky, kz = key
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbor = rows_by_key.get((kx + dx, ky + dy, kz + dz))
                if neighbor is None:
                    continue
                if not same_surface_neighbor(row, neighbor, args):
                    continue
                considered += 1
                if int(neighbor.get("statusCode", 4)) in {0, 1, 2} or soft_usable(neighbor, args):
                    usable += 1

    if usable < args.neighbor_min_usable:
        return False
    return usable / max(1, considered) >= args.neighbor_vote_threshold


def stabilized_copy(
    row: dict[str, float | int],
    rows_by_key: dict[tuple[int, int, int], dict[str, float | int]],
    args: argparse.Namespace,
    allow_unstabilized: bool = False,
) -> dict[str, float | int] | None:
    if int(row["hits"]) < args.min_hits:
        return None
    if float(row["depthStd"]) > args.surface_stabilize_max_depth_std:
        return None
    if float(row["riskRatio"]) > args.surface_stabilize_max_risk_ratio:
        return None

    kx = int(row["_kx"])
    ky = int(row["_ky"])
    kz = int(row["_kz"])
    radius = max(1, int(args.surface_stabilize_radius))
    support: list[dict[str, float | int]] = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                neighbor = rows_by_key.get((kx + dx, ky + dy, kz + dz))
                if neighbor is None:
                    continue
                if int(neighbor.get("statusCode", 4)) not in {0, 1, 2} and not soft_usable(neighbor, args):
                    continue
                if not same_surface_for_stabilization(row, neighbor, args):
                    continue
                support.append(neighbor)

    status_code = int(row.get("statusCode", 4))
    if len(support) < args.surface_stabilize_min_support and status_code != 0:
        if allow_unstabilized:
            out_row = dict(row)
            out_row["surfaceStabilized"] = 0
            out_row["surfaceSupport"] = len(support)
            out_row["surfaceSupportHits"] = sum(max(1, int(item["hits"])) for item in support)
            return out_row
        return None
    if not support:
        out_row = dict(row)
        out_row["surfaceStabilized"] = 0
        out_row["surfaceSupport"] = 0
        out_row["surfaceSupportHits"] = 0
        return out_row

    base_nx = float(row["nx"])
    base_ny = float(row["ny"])
    base_nz = float(row["nz"])
    sx = sy = sz = 0.0
    snx = sny = snz = 0.0
    weight_sum = 0.0
    support_hits = 0
    for item in support:
        hits = max(1, int(item["hits"]))
        confidence = max(0.05, float(item["confidenceMean"]))
        depth_std = max(0.01, float(item["depthStd"]))
        risk = min(1.0, max(0.0, float(item["riskRatio"])))
        weight = hits * confidence * (1.0 - 0.5 * risk) / (1.0 + depth_std * 8.0)
        sx += float(item["x"]) * weight
        sy += float(item["y"]) * weight
        sz += float(item["z"]) * weight
        nx = float(item["nx"])
        ny = float(item["ny"])
        nz = float(item["nz"])
        if base_nx * nx + base_ny * ny + base_nz * nz < 0.0:
            nx = -nx
            ny = -ny
            nz = -nz
        snx += nx * weight
        sny += ny * weight
        snz += nz * weight
        weight_sum += weight
        support_hits += hits

    if weight_sum <= 1e-6:
        return dict(row)

    cx = sx / weight_sum
    cy = sy / weight_sum
    cz = sz / weight_sum
    nx = snx / weight_sum
    ny = sny / weight_sum
    nz = snz / weight_sum
    n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
    if n_len <= 1e-6:
        nx = base_nx
        ny = base_ny
        nz = base_nz
    else:
        nx /= n_len
        ny /= n_len
        nz /= n_len

    pull = min(1.0, max(0.0, float(args.surface_stabilize_plane_pull)))
    px = float(row["x"])
    py = float(row["y"])
    pz = float(row["z"])
    signed = (px - cx) * nx + (py - cy) * ny + (pz - cz) * nz
    out_row = dict(row)
    out_row["x"] = px - signed * nx * pull
    out_row["y"] = py - signed * ny * pull
    out_row["z"] = pz - signed * nz * pull
    out_row["nx"] = nx
    out_row["ny"] = ny
    out_row["nz"] = nz
    out_row["surfaceStabilized"] = 1
    out_row["surfaceSupport"] = len(support)
    out_row["surfaceSupportHits"] = support_hits
    return out_row


def stabilize_surface_rows(
    rows: list[dict[str, float | int]],
    rows_by_key: dict[tuple[int, int, int], dict[str, float | int]],
    args: argparse.Namespace,
    allow_unstabilized: bool = False,
) -> list[dict[str, float | int]]:
    stabilized: list[dict[str, float | int]] = []
    for row in rows:
        out_row = stabilized_copy(row, rows_by_key, args, allow_unstabilized)
        if out_row is not None:
            stabilized.append(out_row)
    return stabilized


def write_ply(path: Path, rows: list[dict[str, float | int]], colors: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for row, color in zip(rows, colors):
            f.write(
                f"{float(row['x']):.6f} {float(row['y']):.6f} {float(row['z']):.6f} "
                f"{float(row['nx']):.6f} {float(row['ny']):.6f} {float(row['nz']):.6f} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )


def bbox(rows: list[dict[str, float | int]]) -> dict[str, object]:
    if not rows:
        return {"count": 0}
    xs = [float(r["x"]) for r in rows]
    ys = [float(r["y"]) for r in rows]
    zs = [float(r["z"]) for r in rows]
    mn = [min(xs), min(ys), min(zs)]
    mx = [max(xs), max(ys), max(zs)]
    return {
        "count": len(rows),
        "min": mn,
        "max": mx,
        "size": [mx[i] - mn[i] for i in range(3)],
    }


def dist(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    vals = sorted(values)
    def pct(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        idx = (len(vals) - 1) * p
        lo = math.floor(idx)
        hi = math.ceil(idx)
        t = idx - lo
        return vals[lo] * (1.0 - t) + vals[hi] * t
    return {
        "count": len(vals),
        "min": vals[0],
        "mean": sum(vals) / len(vals),
        "p50": pct(0.5),
        "p75": pct(0.75),
        "p90": pct(0.9),
        "p95": pct(0.95),
        "max": vals[-1],
    }


def aggregate_voxels(input_path: Path, args: argparse.Namespace, voxel: float) -> tuple[dict[tuple[int, int, int], VoxelStats], dict[str, object]]:
    sessions = find_sessions(input_path)
    if args.max_sessions > 0:
        sessions = sessions[: args.max_sessions]
    report: dict[str, object] = {
        "input": str(input_path),
        "voxel": voxel,
        "sessionCount": len(sessions),
        "sessions": [],
        "rawRows": 0,
        "acceptedRows": 0,
        "rejectedDepthRows": 0,
        "rejectedInvalidRows": 0,
    }
    counters = {
        "rawRows": 0,
        "acceptedRows": 0,
        "rejectedDepthRows": 0,
        "rejectedInvalidRows": 0,
    }
    voxels: dict[tuple[int, int, int], VoxelStats] = {}
    for session_index, session in enumerate(sessions):
        files = list(iter_frame_csvs(session, max(1, args.frame_stride), args.max_frames_per_session))
        before_rows = counters["acceptedRows"]
        for frame in files:
            add_csv_to_voxels(
                frame,
                voxels,
                voxel,
                session_index,
                args.min_depth,
                args.max_depth,
                counters,
            )
        report["sessions"].append(
            {
                "name": session.name,
                "frameCsvCount": len(files),
                "acceptedRows": counters["acceptedRows"] - before_rows,
            }
        )
        print(f"[raw-depth] session {session_index + 1}/{len(sessions)} {session.name}: frames={len(files)} voxels={len(voxels)}")

    report.update(counters)
    report["voxelCount"] = len(voxels)
    return voxels, report


def build_outputs(
    voxels: dict[tuple[int, int, int], VoxelStats],
    args: argparse.Namespace,
) -> tuple[dict[str, list[dict[str, float | int]]], dict[str, object]]:
    by_status: dict[str, list[dict[str, float | int]]] = {
        "stable": [],
        "candidate": [],
        "review": [],
        "risk": [],
        "insufficient": [],
    }
    by_reason: dict[str, list[dict[str, float | int]]] = {
        "depth_unstable": [],
        "projection_error": [],
        "low_confidence": [],
        "oblique_view": [],
        "source_risk_flag": [],
        "insufficient": [],
    }
    rows_by_key: dict[tuple[int, int, int], dict[str, float | int]] = {}
    for key, stats in voxels.items():
        row = stats.row()
        row["_kx"] = key[0]
        row["_ky"] = key[1]
        row["_kz"] = key[2]
        status = status_for(row, args)
        row["statusCode"] = {"stable": 0, "candidate": 1, "review": 2, "risk": 3, "insufficient": 4}[status]
        row["neighborRescued"] = 0
        reason = risk_reason_for(row, args)
        row["riskReasonCode"] = {
            "none": 0,
            "depth_unstable": 1,
            "projection_error": 2,
            "low_confidence": 3,
            "oblique_view": 4,
            "source_risk_flag": 5,
            "insufficient": 6,
        }[reason]
        rows_by_key[key] = row

    neighbor_rescued: list[dict[str, float | int]] = []
    for key, row in rows_by_key.items():
        if int(row["statusCode"]) not in {2, 3}:
            continue
        if rescue_by_neighbor_vote(key, row, rows_by_key, args):
            row["statusCode"] = 1
            row["neighborRescued"] = 1
            row["riskReasonCode"] = 0
            neighbor_rescued.append(row)

    all_rows: list[dict[str, float | int]] = []
    soft_rows: list[dict[str, float | int]] = []
    status_names = ["stable", "candidate", "review", "risk", "insufficient"]
    reason_names = [
        "none",
        "depth_unstable",
        "projection_error",
        "low_confidence",
        "oblique_view",
        "source_risk_flag",
        "insufficient",
    ]
    for row in rows_by_key.values():
        status = status_names[max(0, min(4, int(row["statusCode"])))]
        by_status[status].append(row)
        reason = reason_names[max(0, min(6, int(row["riskReasonCode"])))]
        if reason in by_reason:
            by_reason[reason].append(row)
        if soft_usable(row, args) or int(row.get("neighborRescued", 0)) == 1:
            soft_rows.append(row)
        all_rows.append(row)

    strict_usable = by_status["stable"] + by_status["candidate"]
    raw_usable = soft_rows
    raw_mapping_review = by_status["stable"] + by_status["candidate"] + by_status["review"]
    strictly_stabilized_usable = stabilize_surface_rows(raw_usable, rows_by_key, args, allow_unstabilized=False)
    usable = stabilize_surface_rows(raw_usable, rows_by_key, args, allow_unstabilized=True)
    mapping_review = stabilize_surface_rows(raw_mapping_review, rows_by_key, args, allow_unstabilized=True)
    report = {
        "statusCounts": {k: len(v) for k, v in by_status.items()},
        "riskReasonCounts": {k: len(v) for k, v in by_reason.items()},
        "strictUsableMappingPointCount": len(strict_usable),
        "usableMappingPointCount": len(usable),
        "strictlyStabilizedMappingPointCount": len(strictly_stabilized_usable),
        "rawSoftUsableMappingPointCount": len(raw_usable),
        "softUsableMappingPointCount": len(usable),
        "neighborRescuedCandidateCount": len(neighbor_rescued),
        "reviewMappingPointCount": len(mapping_review),
        "rawReviewMappingPointCount": len(raw_mapping_review),
        "allVoxelCount": len(all_rows),
        "usableRatio": len(usable) / max(1, len(all_rows)),
        "strictUsableRatio": len(strict_usable) / max(1, len(all_rows)),
        "rawSoftUsableRatio": len(raw_usable) / max(1, len(all_rows)),
        "softUsableRatio": len(usable) / max(1, len(all_rows)),
        "bboxAll": bbox(all_rows),
        "bboxUsable": bbox(usable),
        "bboxStrictUsable": bbox(strict_usable),
        "bboxRawSoftUsable": bbox(raw_usable),
        "bboxSoftUsable": bbox(usable),
        "hitDistributionAll": dist([float(r["hits"]) for r in all_rows]),
        "depthStdDistributionAll": dist([float(r["depthStd"]) for r in all_rows]),
        "riskRatioDistributionAll": dist([float(r["riskRatio"]) for r in all_rows]),
    }
    return {
        "raw_depth_voxel_all_by_status": all_rows,
        "stable_raw_depth_surface": by_status["stable"],
        "mapping_input_strict_candidate": strict_usable,
        "mapping_input_candidate_raw_voxel": raw_usable,
        "mapping_input_candidate": usable,
        "mapping_input_candidate_stabilized_only": strictly_stabilized_usable,
        "mapping_input_soft_candidate": usable,
        "mapping_input_neighbor_candidate": neighbor_rescued,
        "mapping_input_review_raw_voxel": raw_mapping_review,
        "mapping_input_review": mapping_review,
        "risk_boundary_or_oblique": by_status["risk"],
        "risk_reason_depth_unstable": by_reason["depth_unstable"],
        "risk_reason_projection_error": by_reason["projection_error"],
        "risk_reason_low_confidence": by_reason["low_confidence"],
        "risk_reason_oblique_view": by_reason["oblique_view"],
        "risk_reason_source_flag": by_reason["source_risk_flag"],
        "insufficient_coverage": by_status["insufficient"],
    }, report


def write_outputs(out: Path, outputs: dict[str, list[dict[str, float | int]]], report: dict[str, object], args: argparse.Namespace) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        colors = []
        for row in rows:
            if name.startswith("risk_reason_"):
                code = int(row.get("riskReasonCode", 0))
                reason = [
                    "none",
                    "depth_unstable",
                    "projection_error",
                    "low_confidence",
                    "oblique_view",
                    "source_risk_flag",
                    "insufficient",
                ][max(0, min(6, code))]
                colors.append(reason_color(reason))
            else:
                code = int(row.get("statusCode", 2))
                status = ["stable", "candidate", "review", "risk", "insufficient"][max(0, min(4, code))]
                colors.append(color_for(status, row))
        write_ply(out / f"{name}.ply", rows, colors)

    if args.write_csv:
        with (out / "raw_depth_voxel_stats.csv").open("w", encoding="utf-8-sig", newline="") as f:
            fieldnames = [
                "x", "y", "z", "nx", "ny", "nz", "hits", "sessions", "riskRatio", "focusRatio",
                "highRatio", "lowRatio", "depthMean", "depthStd", "angleMean", "projectionMean",
                "confidenceMean", "statusCode", "riskReasonCode", "neighborRescued",
                "surfaceStabilized", "surfaceSupport", "surfaceSupportHits",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in outputs["raw_depth_voxel_all_by_status"]:
                writer.writerow(row)

    (out / "raw_depth_mapping_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    out = args.out
    if out is None:
        out = input_path / "raw_depth_mapping_input"
    out = out.resolve()

    voxels, read_report = aggregate_voxels(input_path, args, args.voxel)
    outputs, output_report = build_outputs(voxels, args)
    report: dict[str, object] = {
        "source": "room_raw_depth_frames",
        "parameters": {
            "voxel": args.voxel,
            "previewVoxel": args.preview_voxel,
            "minDepth": args.min_depth,
            "maxDepth": args.max_depth,
            "frameStride": args.frame_stride,
            "maxSessions": args.max_sessions,
            "maxFramesPerSession": args.max_frames_per_session,
            "minHits": args.min_hits,
            "stableHits": args.stable_hits,
            "stableSessions": args.stable_sessions,
            "riskRatio": args.risk_ratio,
            "angleRisk": args.angle_risk,
            "projectionRisk": args.projection_risk,
            "depthStdRisk": args.depth_std_risk,
            "candidateDepthStd": args.candidate_depth_std,
            "usableDepthStd": args.usable_depth_std,
            "usableProjectionRisk": args.usable_projection_risk,
            "usableConfidence": args.usable_confidence,
            "hardDepthStd": args.hard_depth_std,
            "hardProjectionRisk": args.hard_projection_risk,
            "hardConfidence": args.hard_confidence,
            "denseHitDepthStd": args.dense_hit_depth_std,
            "denseHitMultiplier": args.dense_hit_multiplier,
            "candidateRiskRatio": args.candidate_risk_ratio,
            "candidateAngleRisk": args.candidate_angle_risk,
            "neighborVoteRadius": args.neighbor_vote_radius,
            "neighborVoteThreshold": args.neighbor_vote_threshold,
            "neighborMinUsable": args.neighbor_min_usable,
            "neighborNormalAngle": args.neighbor_normal_angle,
            "neighborDepthDelta": args.neighbor_depth_delta,
            "neighborRescueDepthStd": args.neighbor_rescue_depth_std,
            "surfaceStabilizeRadius": args.surface_stabilize_radius,
            "surfaceStabilizeMinSupport": args.surface_stabilize_min_support,
            "surfaceStabilizeNormalAngle": args.surface_stabilize_normal_angle,
            "surfaceStabilizeDepthDelta": args.surface_stabilize_depth_delta,
            "surfaceStabilizeMaxDepthStd": args.surface_stabilize_max_depth_std,
            "surfaceStabilizeMaxRiskRatio": args.surface_stabilize_max_risk_ratio,
            "surfaceStabilizePlanePull": args.surface_stabilize_plane_pull,
        },
        "read": read_report,
        "outputs": output_report,
    }
    write_outputs(out, outputs, report, args)

    # Coarse preview is useful for quickly locating missing scan areas.
    if args.preview_voxel > args.voxel:
        preview_args = argparse.Namespace(**vars(args))
        preview_args.voxel = args.preview_voxel
        preview_voxels, preview_read = aggregate_voxels(input_path, preview_args, args.preview_voxel)
        preview_outputs, preview_report = build_outputs(preview_voxels, preview_args)
        preview_out = out / f"preview_voxel_{int(args.preview_voxel * 1000):03d}mm"
        write_outputs(preview_out, preview_outputs, {"read": preview_read, "outputs": preview_report}, preview_args)
        report["preview"] = {"folder": str(preview_out), "outputs": preview_report}
        (out / "raw_depth_mapping_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"[raw-depth] done: {out}")
    print(json.dumps(output_report["statusCounts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
