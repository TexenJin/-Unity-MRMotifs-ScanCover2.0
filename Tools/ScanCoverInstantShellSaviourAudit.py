#!/usr/bin/env python3
"""Qualify the instantaneous depth shell as a persistent-surface authority.

This is a read-only, held-out multi-view audit.  A processed depth frame is
treated as a provisional shell, never as truth.  Points from that shell are
reprojected into later frames captured from materially different eye poses.
Only observations that pass overlap, target connectivity, round-trip pixel,
surface-normal and ray-angle gates become comparable evidence.

The audit deliberately does not read TSDF or paper geometry.  Its question is
strictly upstream: can a short-lived instantaneous shell recover the same 3-D
surface from independent views often and accurately enough to be promoted from
display aid to geometry witness?
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


BASELINE_BANDS = (
    ("same_view_control", 0.000, 0.030),
    ("weak_baseline", 0.030, 0.080),
    ("independent_08_15cm", 0.080, 0.150),
    ("independent_15_30cm", 0.150, 0.300),
    ("independent_30_45cm", 0.300, 0.450),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only independent-view qualification of the QRS instant shell."
    )
    parser.add_argument("session", type=Path, help="sealed replay session root")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: <session>/artifacts/instant_shell_saviour_audit",
    )
    parser.add_argument("--max-source-frames", type=int, default=160)
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument("--maximum-future-seconds", type=float, default=8.0)
    parser.add_argument("--minimum-independent-baseline", type=float, default=0.08)
    parser.add_argument("--minimum-independent-ray-angle", type=float, default=3.0)
    parser.add_argument("--maximum-baseline", type=float, default=0.45)
    parser.add_argument("--roundtrip-pixels", type=float, default=2.5)
    parser.add_argument("--normal-dot", type=float, default=0.75)
    parser.add_argument("--strong-support-mm", type=float, default=5.0)
    parser.add_argument("--usable-support-mm", type=float, default=10.0)
    parser.add_argument("--patch-metres", type=float, default=0.10)
    parser.add_argument("--minimum-comparable-samples", type=int, default=5000)
    parser.add_argument("--minimum-auditable-patches", type=int, default=50)
    parser.add_argument("--minimum-comparable-coverage-pct", type=float, default=35.0)
    parser.add_argument("--minimum-qualified-patch-pct", type=float, default=80.0)
    return parser.parse_args()


def matrix(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64).reshape(4, 4)


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    data = np.asarray(list(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    q = np.percentile(data, (5, 25, 50, 75, 90, 95, 99))
    return {
        "count": int(len(data)),
        "min": float(np.min(data)),
        "p05": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p90": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
    }


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def pct(numerator: int, denominator: int) -> float:
    return float(numerator) * 100.0 / max(int(denominator), 1)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def recorded_eye(record: dict) -> int:
    eye = int(record.get("recordedEyeIndex", 1))
    poses = record.get("trackingPoses", [])
    return max(0, min(eye, len(poses) - 1)) if poses else eye


@dataclass
class Frame:
    row: dict[str, str]
    meta: dict
    image_path: Path
    width: int
    height: int
    eye: int
    projection: np.ndarray
    projection_inverse: np.ndarray
    view: np.ndarray
    view_inverse: np.ndarray
    camera: np.ndarray
    time: float


@dataclass
class ShellSamples:
    x: np.ndarray
    y: np.ndarray
    world: np.ndarray
    normal: np.ndarray
    range_m: np.ndarray
    incidence_cos: np.ndarray


@dataclass
class PatchAccumulator:
    residual_mm: list[float] = field(default_factory=list)
    signed_mm: list[float] = field(default_factory=list)
    pair_ids: set[str] = field(default_factory=set)
    source_frames: set[int] = field(default_factory=set)
    target_frames: set[int] = field(default_factory=set)
    distances_m: list[float] = field(default_factory=list)
    incidence_cos: list[float] = field(default_factory=list)


class ImageCache:
    def __init__(self, maximum: int = 12):
        self.maximum = maximum
        self.images: dict[Path, np.ndarray] = {}
        self.order: list[Path] = []

    def get(self, frame: Frame) -> np.ndarray:
        if frame.image_path in self.images:
            self.order.remove(frame.image_path)
            self.order.append(frame.image_path)
            return self.images[frame.image_path]
        image = np.fromfile(frame.image_path, dtype="<f4")
        expected = frame.width * frame.height
        if image.size != expected:
            raise ValueError(
                f"depth size mismatch for {frame.image_path}: {image.size}/{expected}"
            )
        image = image.reshape(frame.height, frame.width)
        self.images[frame.image_path] = image
        self.order.append(frame.image_path)
        while len(self.order) > self.maximum:
            old = self.order.pop(0)
            self.images.pop(old, None)
        return image


def load_frames(session: Path) -> list[Frame]:
    root = session / "depth_pairs"
    rows = read_csv(root / "manifest.csv")
    frames: list[Frame] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        meta_path = root / "frames" / row["metadataFile"]
        image_path = root / "frames" / row["processedFile"]
        if not meta_path.is_file() or not image_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        eye = recorded_eye(meta)
        view_inverse = matrix(meta["viewInverse"][eye])
        frames.append(
            Frame(
                row=row,
                meta=meta,
                image_path=image_path,
                width=int(meta["width"]),
                height=int(meta["height"]),
                eye=eye,
                projection=matrix(meta["projection"][eye]),
                projection_inverse=matrix(meta["projectionInverse"][eye]),
                view=matrix(meta["view"][eye]),
                view_inverse=view_inverse,
                camera=view_inverse[:3, 3].copy(),
                time=float(meta.get("preprocessUnscaledTime", meta.get("unscaledTime", 0.0))),
            )
        )
    frames.sort(key=lambda item: int(item.meta["pairIndex"]))
    return frames


def valid_depth(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > 0.0) & (values < 1.0)


def linearize(values: np.ndarray, projection: np.ndarray) -> np.ndarray:
    z = values.astype(np.float64) * 2.0 - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(float(projection[2, 3]) / (z + float(projection[2, 2])))


def reconstruct(frame: Frame, x: np.ndarray, y: np.ndarray, depth: np.ndarray) -> np.ndarray:
    u = (x.astype(np.float64) + 0.5) / frame.width
    v = (y.astype(np.float64) + 0.5) / frame.height
    hcs = np.stack(
        (u * 2.0 - 1.0, v * 2.0 - 1.0, depth * 2.0 - 1.0, np.ones_like(u)),
        axis=1,
    )
    world_h = (frame.view_inverse @ (frame.projection_inverse @ hcs.T)).T
    return world_h[:, :3] / world_h[:, 3:4]


def project(frame: Frame, world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world_h = np.column_stack((world, np.ones(len(world), dtype=np.float64)))
    clip = (frame.projection @ (frame.view @ world_h.T)).T
    safe = np.abs(clip[:, 3]) > 1e-8
    uv = np.full((len(world), 2), np.nan, dtype=np.float64)
    uv[safe] = clip[safe, :2] / clip[safe, 3:4] * 0.5 + 0.5
    inside = safe & (uv[:, 0] >= 0.0) & (uv[:, 0] < 1.0) & (uv[:, 1] >= 0.0) & (uv[:, 1] < 1.0)
    return uv, clip[:, 3], inside


def normals_at(frame: Frame, image: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xp = np.minimum(x + 1, frame.width - 1)
    yp = np.minimum(y + 1, frame.height - 1)
    d0 = image[y, x]
    dx = image[y, xp]
    dy = image[yp, x]
    ok = valid_depth(d0) & valid_depth(dx) & valid_depth(dy)
    normals = np.full((len(x), 3), np.nan, dtype=np.float64)
    if np.any(ok):
        p0 = reconstruct(frame, x[ok], y[ok], d0[ok])
        px = reconstruct(frame, xp[ok], y[ok], dx[ok])
        py = reconstruct(frame, x[ok], yp[ok], dy[ok])
        n = np.cross(px - p0, py - p0)
        length = np.linalg.norm(n, axis=1)
        usable = length > 1e-8
        good_indices = np.flatnonzero(ok)[usable]
        normals[good_indices] = n[usable] / length[usable, None]
        ok[:] = False
        ok[good_indices] = True
    return normals, ok


def connectivity_at(
    frame: Frame, image: np.ndarray, x: np.ndarray, y: np.ndarray, step: int = 3
) -> np.ndarray:
    xl = np.maximum(x - step, 0)
    xr = np.minimum(x + step, frame.width - 1)
    yu = np.maximum(y - step, 0)
    yd = np.minimum(y + step, frame.height - 1)
    samples = np.stack(
        (image[y, x], image[y, xl], image[y, xr], image[yu, x], image[yd, x]),
        axis=1,
    )
    ok = np.all(valid_depth(samples), axis=1)
    depths = linearize(samples, frame.projection)
    centre = depths[:, 0]
    hard_gap = np.maximum(0.030, centre * 0.025)
    return ok & ((np.max(depths, axis=1) - np.min(depths, axis=1)) <= hard_gap)


def build_shell_samples(frame: Frame, image: np.ndarray, pixel_stride: int) -> ShellSamples:
    border = 4
    xs = np.arange(border, frame.width - border, max(2, pixel_stride), dtype=np.int32)
    ys = np.arange(border, frame.height - border, max(2, pixel_stride), dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    x = grid_x.ravel()
    y = grid_y.ravel()
    depth = image[y, x]
    normal, normal_ok = normals_at(frame, image, x, y)
    connected = connectivity_at(frame, image, x, y)
    ok = valid_depth(depth) & normal_ok & connected
    if not np.any(ok):
        empty_i = np.empty(0, dtype=np.int32)
        empty_v = np.empty((0, 3), dtype=np.float64)
        empty_f = np.empty(0, dtype=np.float64)
        return ShellSamples(empty_i, empty_i, empty_v, empty_v, empty_f, empty_f)
    x = x[ok]
    y = y[ok]
    world = reconstruct(frame, x, y, depth[ok])
    normal = normal[ok]
    rays = world - frame.camera
    ranges = np.linalg.norm(rays, axis=1)
    ray_dir = rays / np.maximum(ranges[:, None], 1e-9)
    incidence = np.abs(np.sum(normal * -ray_dir, axis=1))
    finite = np.isfinite(world).all(axis=1) & np.isfinite(normal).all(axis=1)
    finite &= (ranges >= 0.20) & (ranges <= 8.0)
    return ShellSamples(
        x=x[finite],
        y=y[finite],
        world=world[finite],
        normal=normal[finite],
        range_m=ranges[finite],
        incidence_cos=incidence[finite],
    )


def evenly_selected_indices(count: int, maximum: int) -> list[int]:
    if count <= 0:
        return []
    if maximum <= 0 or count <= maximum:
        return list(range(count))
    return sorted(set(int(value) for value in np.linspace(0, count - 1, maximum)))


def select_target_indices(frames: list[Frame], source_index: int, maximum_future_seconds: float) -> list[tuple[str, int]]:
    source = frames[source_index]
    selected: list[tuple[str, int]] = []
    used: set[int] = set()
    for label, low, high in BASELINE_BANDS:
        best: tuple[float, int] | None = None
        midpoint = (low + high) * 0.5
        for target_index in range(source_index + 1, len(frames)):
            target = frames[target_index]
            delta_time = target.time - source.time
            if delta_time < 0.20:
                continue
            if delta_time > maximum_future_seconds:
                break
            baseline = float(np.linalg.norm(target.camera - source.camera))
            if baseline < low or baseline >= high:
                continue
            score = abs(baseline - midpoint) + delta_time * 0.002
            if best is None or score < best[0]:
                best = (score, target_index)
        if best is not None and best[1] not in used:
            selected.append((label, best[1]))
            used.add(best[1])
    return selected


def distance_bucket(distance_m: float) -> str:
    if distance_m < 1.0:
        return "0.2_1.0m"
    if distance_m < 2.0:
        return "1.0_2.0m"
    if distance_m < 3.0:
        return "2.0_3.0m"
    return "3.0_8.0m"


def incidence_bucket(cosine: float) -> str:
    if cosine < 0.35:
        return "grazing_cos_lt_0.35"
    if cosine < 0.60:
        return "oblique_cos_0.35_0.60"
    if cosine < 0.82:
        return "moderate_cos_0.60_0.82"
    return "frontal_cos_ge_0.82"


def classify_residual(abs_mm: float, strong_mm: float, usable_mm: float) -> str:
    if abs_mm <= strong_mm:
        return "strong_support"
    if abs_mm <= usable_mm:
        return "usable_support"
    return "conflict"


def fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def main() -> int:
    args = parse_args()
    session = args.session.resolve()
    output = (args.out or (session / "artifacts" / "instant_shell_saviour_audit")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = load_frames(session)
    if len(frames) < 2:
        raise RuntimeError("Need at least two complete paired-depth frames")

    source_indices = evenly_selected_indices(len(frames), args.max_source_frames)
    cache = ImageCache()
    all_abs: list[float] = []
    all_signed: list[float] = []
    pair_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    patches: dict[tuple[int, int, int], PatchAccumulator] = defaultdict(PatchAccumulator)
    condition_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    condition_counts: Counter[tuple[str, str]] = Counter()
    baseline_band_values: dict[str, list[float]] = defaultdict(list)
    baseline_band_signed: dict[str, list[float]] = defaultdict(list)
    reason_counts: Counter[str] = Counter()
    independent_shell_samples = 0
    independent_comparable_samples = 0
    independent_source_samples_seen = 0

    sample_fields = [
        "sourcePair", "targetPair", "baselineBand", "sourceX", "sourceY",
        "worldX", "worldY", "worldZ", "distanceM", "incidenceCos",
        "baselineM", "rayAngleDeg", "targetX", "targetY", "status",
        "signedResidualMm", "absResidualMm", "supportClass", "normalDot",
        "roundtripPixels", "patchX", "patchY", "patchZ",
    ]
    samples_path = output / "samples.csv.gz"
    with gzip.open(samples_path, "wt", encoding="utf-8", newline="") as sample_handle:
        sample_writer = csv.DictWriter(sample_handle, fieldnames=sample_fields)
        sample_writer.writeheader()

        for source_index in source_indices:
            source = frames[source_index]
            source_image = cache.get(source)
            shell = build_shell_samples(source, source_image, args.pixel_stride)
            targets = select_target_indices(frames, source_index, args.maximum_future_seconds)
            source_any_comparable = np.zeros(len(shell.x), dtype=bool)
            source_has_independent_target = False
            source_pair_count = 0

            for band, target_index in targets:
                target = frames[target_index]
                baseline_m = float(np.linalg.norm(target.camera - source.camera))
                is_independent_band = baseline_m >= args.minimum_independent_baseline
                if is_independent_band:
                    source_has_independent_target = True
                target_image = cache.get(target)
                count = len(shell.x)
                if count == 0:
                    continue
                uv, _, inside = project(target, shell.world)
                tx = np.clip((uv[:, 0] * target.width).astype(np.int32), 0, target.width - 1)
                ty = np.clip((uv[:, 1] * target.height).astype(np.int32), 0, target.height - 1)
                target_depth = target_image[ty, tx]
                target_depth_ok = valid_depth(target_depth)
                target_connected = np.zeros(count, dtype=bool)
                target_normal = np.full((count, 3), np.nan, dtype=np.float64)
                target_normal_ok = np.zeros(count, dtype=bool)
                candidate = inside & target_depth_ok
                if np.any(candidate):
                    target_connected[candidate] = connectivity_at(
                        target, target_image, tx[candidate], ty[candidate]
                    )
                    normals, normal_ok = normals_at(
                        target, target_image, tx[candidate], ty[candidate]
                    )
                    target_normal[candidate] = normals
                    target_normal_ok[candidate] = normal_ok

                target_world = np.full_like(shell.world, np.nan)
                if np.any(candidate):
                    target_world[candidate] = reconstruct(
                        target, tx[candidate], ty[candidate], target_depth[candidate]
                    )
                source_ray = shell.world - source.camera
                target_expected_ray = shell.world - target.camera
                source_ray /= np.maximum(np.linalg.norm(source_ray, axis=1)[:, None], 1e-9)
                target_expected_range = np.linalg.norm(target_expected_ray, axis=1)
                target_expected_ray /= np.maximum(target_expected_range[:, None], 1e-9)
                ray_cos = np.clip(np.sum(source_ray * target_expected_ray, axis=1), -1.0, 1.0)
                ray_angle = np.degrees(np.arccos(ray_cos))
                ray_independent = ray_angle >= args.minimum_independent_ray_angle

                observed_range = np.linalg.norm(target_world - target.camera, axis=1)
                signed_mm = (observed_range - target_expected_range) * 1000.0
                abs_mm = np.abs(signed_mm)
                normal_dot = np.abs(np.sum(shell.normal * target_normal, axis=1))

                cycle_uv = np.full((count, 2), np.nan, dtype=np.float64)
                cycle_inside = np.zeros(count, dtype=bool)
                if np.any(candidate):
                    cycle_uv[candidate], _, cycle_inside[candidate] = project(
                        source, target_world[candidate]
                    )
                cycle_px = np.sqrt(
                    (cycle_uv[:, 0] * source.width - (shell.x + 0.5)) ** 2
                    + (cycle_uv[:, 1] * source.height - (shell.y + 0.5)) ** 2
                )

                statuses = np.full(count, "comparable", dtype=object)
                statuses[~inside] = "outside_target"
                statuses[inside & ~target_depth_ok] = "target_depth_missing"
                statuses[candidate & ~target_connected] = "target_depth_edge"
                statuses[candidate & target_connected & ~target_normal_ok] = "target_normal_missing"
                statuses[candidate & target_connected & target_normal_ok & ~cycle_inside] = "roundtrip_outside"
                statuses[
                    candidate & target_connected & target_normal_ok & cycle_inside
                    & (cycle_px > args.roundtrip_pixels)
                ] = "roundtrip_mismatch"
                statuses[
                    candidate & target_connected & target_normal_ok & cycle_inside
                    & (cycle_px <= args.roundtrip_pixels) & (normal_dot < args.normal_dot)
                ] = "normal_conflict"
                if is_independent_band:
                    statuses[
                        (statuses == "comparable") & ~ray_independent
                    ] = "ray_angle_too_small"

                comparable = statuses == "comparable"
                pair_id = f"{int(source.meta['pairIndex'])}->{int(target.meta['pairIndex'])}"
                comparable_abs = abs_mm[comparable]
                comparable_signed = signed_mm[comparable]
                baseline_band_values[band].extend(float(value) for value in comparable_abs)
                baseline_band_signed[band].extend(float(value) for value in comparable_signed)
                pair_offset = float(np.median(comparable_signed)) if len(comparable_signed) else None
                shape = np.abs(comparable_signed - pair_offset) if pair_offset is not None else np.empty(0)
                pair_rows.append(
                    {
                        "pairId": pair_id,
                        "sourcePair": int(source.meta["pairIndex"]),
                        "targetPair": int(target.meta["pairIndex"]),
                        "baselineBand": band,
                        "baselineM": baseline_m,
                        "deltaSeconds": target.time - source.time,
                        "sourceShellSamples": count,
                        "targetInside": int(np.count_nonzero(inside)),
                        "comparable": int(np.count_nonzero(comparable)),
                        "comparablePct": pct(int(np.count_nonzero(comparable)), count),
                        "signedOffsetMedianMm": pair_offset,
                        "shapeAbsP50Mm": float(np.percentile(shape, 50)) if len(shape) else None,
                        "shapeAbsP95Mm": float(np.percentile(shape, 95)) if len(shape) else None,
                        "absResidualP50Mm": float(np.percentile(comparable_abs, 50)) if len(comparable_abs) else None,
                        "absResidualP95Mm": float(np.percentile(comparable_abs, 95)) if len(comparable_abs) else None,
                        "over10mmPct": float(np.mean(comparable_abs > args.usable_support_mm) * 100.0) if len(comparable_abs) else None,
                    }
                )
                source_pair_count += 1

                if is_independent_band:
                    independent_shell_samples += count
                    independent_comparable_samples += int(np.count_nonzero(comparable))
                    source_any_comparable |= comparable
                    all_abs.extend(float(value) for value in comparable_abs)
                    all_signed.extend(float(value) for value in comparable_signed)

                for index in range(count):
                    status = str(statuses[index])
                    reason_counts[status] += 1
                    patch_key = tuple(
                        int(math.floor(float(value) / args.patch_metres))
                        for value in shell.world[index]
                    )
                    support_class = ""
                    signed_value: float | str = ""
                    abs_value: float | str = ""
                    if comparable[index]:
                        signed_value = float(signed_mm[index])
                        abs_value = float(abs_mm[index])
                        support_class = classify_residual(
                            float(abs_mm[index]), args.strong_support_mm, args.usable_support_mm
                        )
                        if is_independent_band:
                            accumulator = patches[patch_key]
                            accumulator.residual_mm.append(float(abs_mm[index]))
                            accumulator.signed_mm.append(float(signed_mm[index]))
                            accumulator.pair_ids.add(pair_id)
                            accumulator.source_frames.add(int(source.meta["pairIndex"]))
                            accumulator.target_frames.add(int(target.meta["pairIndex"]))
                            accumulator.distances_m.append(float(shell.range_m[index]))
                            accumulator.incidence_cos.append(float(shell.incidence_cos[index]))
                            for factor, bucket in (
                                ("distance", distance_bucket(float(shell.range_m[index]))),
                                ("incidence", incidence_bucket(float(shell.incidence_cos[index]))),
                                ("baseline", band),
                            ):
                                condition_values[(factor, bucket)].append(float(abs_mm[index]))
                                condition_counts[(factor, bucket)] += 1
                    sample_writer.writerow(
                        {
                            "sourcePair": int(source.meta["pairIndex"]),
                            "targetPair": int(target.meta["pairIndex"]),
                            "baselineBand": band,
                            "sourceX": int(shell.x[index]),
                            "sourceY": int(shell.y[index]),
                            "worldX": float(shell.world[index, 0]),
                            "worldY": float(shell.world[index, 1]),
                            "worldZ": float(shell.world[index, 2]),
                            "distanceM": float(shell.range_m[index]),
                            "incidenceCos": float(shell.incidence_cos[index]),
                            "baselineM": baseline_m,
                            "rayAngleDeg": float(ray_angle[index]),
                            "targetX": int(tx[index]) if inside[index] else "",
                            "targetY": int(ty[index]) if inside[index] else "",
                            "status": status,
                            "signedResidualMm": signed_value,
                            "absResidualMm": abs_value,
                            "supportClass": support_class,
                            "normalDot": float(normal_dot[index]) if np.isfinite(normal_dot[index]) else "",
                            "roundtripPixels": float(cycle_px[index]) if np.isfinite(cycle_px[index]) else "",
                            "patchX": patch_key[0],
                            "patchY": patch_key[1],
                            "patchZ": patch_key[2],
                        }
                    )

            if source_has_independent_target:
                independent_source_samples_seen += len(shell.x)
            source_rows.append(
                {
                    "sourcePair": int(source.meta["pairIndex"]),
                    "sourceTime": source.time,
                    "shellSamples": len(shell.x),
                    "selectedTargetPairs": source_pair_count,
                    "hasIndependentTarget": int(source_has_independent_target),
                    "independentComparableSamples": int(np.count_nonzero(source_any_comparable)),
                    "independentComparablePct": pct(int(np.count_nonzero(source_any_comparable)), len(shell.x)),
                }
            )

    patch_rows: list[dict[str, object]] = []
    qualified_patches = 0
    auditable_patches = 0
    for key, item in sorted(patches.items()):
        count = len(item.residual_mm)
        p50 = percentile(item.residual_mm, 50)
        p95 = percentile(item.residual_mm, 95)
        conflict_pct = pct(sum(value > args.usable_support_mm for value in item.residual_mm), count)
        offset = float(np.median(item.signed_mm)) if item.signed_mm else None
        shape_p95 = (
            float(np.percentile(np.abs(np.asarray(item.signed_mm) - offset), 95))
            if offset is not None else None
        )
        auditable = count >= 20 and len(item.pair_ids) >= 2 and len(item.target_frames) >= 2
        qualified = bool(
            auditable and p50 is not None and p95 is not None
            and p50 <= args.strong_support_mm
            and p95 <= args.usable_support_mm
            and conflict_pct <= 5.0
        )
        auditable_patches += int(auditable)
        qualified_patches += int(qualified)
        patch_rows.append(
            {
                "patchX": key[0], "patchY": key[1], "patchZ": key[2],
                "samples": count,
                "pairCount": len(item.pair_ids),
                "sourceFrames": len(item.source_frames),
                "targetFrames": len(item.target_frames),
                "distanceMeanM": float(np.mean(item.distances_m)),
                "incidenceCosMean": float(np.mean(item.incidence_cos)),
                "signedOffsetMedianMm": offset,
                "shapeAbsP95Mm": shape_p95,
                "absResidualP50Mm": p50,
                "absResidualP95Mm": p95,
                "over10mmPct": conflict_pct,
                "auditable": int(auditable),
                "qualified": int(qualified),
            }
        )

    condition_rows: list[dict[str, object]] = []
    condition_failures: list[str] = []
    for (factor, bucket), values in sorted(condition_values.items()):
        stats = distribution(values)
        populated = len(values) >= 200
        passed = bool(populated and stats["p95"] is not None and float(stats["p95"]) <= 15.0)
        if populated and not passed:
            condition_failures.append(f"{factor}:{bucket}")
        condition_rows.append(
            {
                "factor": factor,
                "bucket": bucket,
                "samples": len(values),
                "absResidualP50Mm": stats["p50"],
                "absResidualP95Mm": stats["p95"],
                "over10mmPct": pct(sum(value > args.usable_support_mm for value in values), len(values)),
                "populated": int(populated),
                "conditionPassP95Le15Mm": int(passed),
            }
        )

    overall = distribution(all_abs)
    baseline_band_summary = {}
    for band, values in sorted(baseline_band_values.items()):
        signed_values = baseline_band_signed[band]
        baseline_band_summary[band] = {
            "absResidualMm": distribution(values),
            "signedResidualMm": distribution(signed_values),
            "over10mmConflictPct": pct(
                sum(value > args.usable_support_mm for value in values), len(values)
            ),
        }
    comparable_coverage = pct(independent_comparable_samples, independent_shell_samples)
    unique_source_coverage = pct(
        sum(int(row["independentComparableSamples"]) for row in source_rows),
        independent_source_samples_seen,
    )
    qualified_patch_pct = pct(qualified_patches, auditable_patches)
    conflict_pct = pct(sum(value > args.usable_support_mm for value in all_abs), len(all_abs))

    evidence_gate = (
        len(all_abs) >= args.minimum_comparable_samples
        and auditable_patches >= args.minimum_auditable_patches
        and sum(int(row["hasIndependentTarget"]) for row in source_rows) >= 20
    )
    accuracy_gate = bool(
        overall["p50"] is not None and overall["p95"] is not None
        and float(overall["p50"]) <= args.strong_support_mm
        and float(overall["p95"]) <= args.usable_support_mm
        and conflict_pct <= 5.0
    )
    coverage_gate = (
        comparable_coverage >= args.minimum_comparable_coverage_pct
        and qualified_patch_pct >= args.minimum_qualified_patch_pct
    )
    condition_gate = not condition_failures
    if not evidence_gate:
        verdict = "INSUFFICIENT_EVIDENCE"
        chinese = "证据不足：尚不能决定即时壳能否取得正式几何权。"
    elif accuracy_gate and coverage_gate and condition_gate:
        verdict = "QUALIFIED_FOR_SHADOW_GEOMETRY_AUTHORITY"
        chinese = "通过影子资格考试：可以进入零权限几何裁决 A/B，不代表已经授权生产。"
    elif accuracy_gate:
        verdict = "QUALIFIED_ONLY_AS_PROVISIONAL_WITNESS"
        chinese = "精度通过但覆盖或条件稳定性不足：只能当候选证人，不能单独决定正式表面。"
    else:
        verdict = "NOT_QUALIFIED_AS_SURFACE_AUTHORITY"
        chinese = "未通过：跨独立视角后深度不能稳定复现，不足以担当正式表面裁判。"

    pair_fields = list(pair_rows[0].keys()) if pair_rows else ["pairId"]
    source_fields = list(source_rows[0].keys()) if source_rows else ["sourcePair"]
    patch_fields = list(patch_rows[0].keys()) if patch_rows else ["patchX"]
    condition_fields = list(condition_rows[0].keys()) if condition_rows else ["factor"]
    write_csv(output / "frame_pairs.csv", pair_rows, pair_fields)
    write_csv(output / "source_frames.csv", source_rows, source_fields)
    write_csv(output / "patches_10cm.csv", patch_rows, patch_fields)
    write_csv(output / "condition_bins.csv", condition_rows, condition_fields)

    summary = {
        "schema": "scancover-instant-shell-saviour-audit-v1",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "session": str(session),
        "question": "Can post-QRS instantaneous shells reproduce the same 3-D surface from independent later views?",
        "verdict": verdict,
        "verdictChinese": chinese,
        "gates": {
            "evidence": evidence_gate,
            "accuracy": accuracy_gate,
            "coverage": coverage_gate,
            "conditionRobustness": condition_gate,
        },
        "thresholds": {
            "minimumIndependentBaselineM": args.minimum_independent_baseline,
            "minimumIndependentRayAngleDeg": args.minimum_independent_ray_angle,
            "strongSupportMm": args.strong_support_mm,
            "usableSupportMm": args.usable_support_mm,
            "maximumConflictPct": 5.0,
            "minimumComparableCoveragePct": args.minimum_comparable_coverage_pct,
            "minimumQualifiedPatchPct": args.minimum_qualified_patch_pct,
            "populatedConditionMaximumP95Mm": 15.0,
        },
        "evidence": {
            "availableFrames": len(frames),
            "sampledSourceFrames": len(source_rows),
            "sourceFramesWithIndependentTarget": sum(int(row["hasIndependentTarget"]) for row in source_rows),
            "framePairs": len(pair_rows),
            "independentShellSampleOpportunities": independent_shell_samples,
            "independentComparableSamples": len(all_abs),
            "independentComparableCoveragePct": comparable_coverage,
            "uniqueSourceSampleComparablePct": unique_source_coverage,
            "auditable10cmPatches": auditable_patches,
            "qualified10cmPatches": qualified_patches,
            "qualified10cmPatchPct": qualified_patch_pct,
        },
        "independentViewAbsResidualMm": overall,
        "independentViewSignedResidualMm": distribution(all_signed),
        "baselineBandControls": baseline_band_summary,
        "over10mmConflictPct": conflict_pct,
        "conditionFailures": condition_failures,
        "comparisonStatusCounts": dict(reason_counts),
        "interpretationLimits": [
            "This proves or refutes cross-view self-consistency, not absolute metric truth.",
            "Occlusions and disocclusions are censored by round-trip and connectivity gates instead of counted as depth conflicts.",
            "Passing authorizes only a zero-production-authority A/B; TSDF input must remain unchanged until that A/B also passes.",
        ],
        "outputs": {
            "exactSamplesGzip": "samples.csv.gz",
            "framePairs": "frame_pairs.csv",
            "sourceFrames": "source_frames.csv",
            "patches": "patches_10cm.csv",
            "conditionBins": "condition_bins.csv",
            "humanVerdict": "VERDICT.md",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# 即时壳‘救世主’资格考试",
        "",
        f"结论：**{chinese}**",
        "",
        "这不是同视角贴合测试。每个即时壳点都必须在后续独立视角中重新命中，并通过往返投影、局部连通和法线一致性检查。",
        "",
        "## 四道门",
        "",
        f"- 证据量：{'通过' if evidence_gate else '未通过'}",
        f"- 跨视角精度：{'通过' if accuracy_gate else '未通过'}",
        f"- 可裁决覆盖：{'通过' if coverage_gate else '未通过'}",
        f"- 距离/入射角/基线分档：{'通过' if condition_gate else '未通过'}",
        "",
        "## 核心数字",
        "",
        f"- 独立视角可比较样本：{len(all_abs)}",
        f"- 可比较覆盖率：{comparable_coverage:.2f}%（按全部独立视角机会）",
        f"- 去重源点覆盖率：{unique_source_coverage:.2f}%",
        f"- 绝对残差 p50/p95：{fmt(overall['p50'])} / {fmt(overall['p95'])} mm",
        f"- 超过 {args.usable_support_mm:.1f} mm 的冲突：{conflict_pct:.2f}%",
        f"- 可审计/合格 10 cm 空间片：{auditable_patches} / {qualified_patches}（{qualified_patch_pct:.2f}%）",
        f"- 条件失败分档：{', '.join(condition_failures) if condition_failures else '无'}",
        "",
        "## 同视错觉与独立视角",
        "",
        "| 基线档 | 样本 | 绝对残差 p50 | 绝对残差 p95 | >10 mm |",
        "|---|---:|---:|---:|---:|",
        *[
            "| " + band + " | "
            + str(stats["absResidualMm"]["count"]) + " | "
            + fmt(stats["absResidualMm"]["p50"]) + " | "
            + fmt(stats["absResidualMm"]["p95"]) + " | "
            + fmt(stats["over10mmConflictPct"]) + "% |"
            for band, stats in baseline_band_summary.items()
        ],
        "",
        "## 判读边界",
        "",
        "- 通过只能证明即时壳具有跨视角自洽性，不能凭空证明绝对距离真值。",
        "- 通过后的下一步仍是零权限 A/B：裁决壳与现有 TSDF 同时运行，但不得改变生产纸皮。",
        "- `samples.csv.gz` 保留每个成功、冲突及被遮挡/不连通而不可比较的样本，避免只看汇总值。",
    ]
    (output / "VERDICT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "out": str(output), "gates": summary["gates"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
