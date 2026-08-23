#!/usr/bin/env python3
"""Read a QRS paired-depth session directly from a connected Quest over ADB.

The audit deliberately keeps no frame binaries on the host.  It reads all small
JSON metadata, then streams a deterministic stratified frame sample to measure
pre/post-QRS disagreement and to test whether repeated observations support a
narrow robust plane band.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


DEFAULT_ADB = Path(
    r"C:\Program Files\Unity\Hub\Editor\6000.3.9f1\Editor\Data"
    r"\PlaybackEngines\AndroidPlayer\SDK\platform-tools\adb.exe"
)
PACKAGE_ROOT = (
    "/sdcard/Android/data/com.pcaii.scancover.quest3/files/"
    "ScanCoverDiagnostics/depth_pair_capture"
)


@dataclass
class Bucket:
    frames: int = 0
    raw_valid: int = 0
    post_valid: int = 0
    common_valid: int = 0
    removed: int = 0
    introduced: int = 0
    diffs: list[np.ndarray] = field(default_factory=list)


def run_adb(adb: Path, serial: str, *args: str) -> bytes:
    cmd = [str(adb)]
    if serial:
        cmd += ["-s", serial]
    cmd += list(args)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(
            f"ADB failed ({result.returncode}): {' '.join(cmd)}\n"
            + result.stderr.decode("utf-8", "replace")
        )
    return result.stdout


def latest_session(adb: Path, serial: str) -> str:
    text = run_adb(
        adb,
        serial,
        "shell",
        f"ls -1d {PACKAGE_ROOT}/session_* 2>/dev/null | tail -n 1",
    ).decode("utf-8", "replace").strip()
    if not text:
        raise RuntimeError("No paired-depth session found on device")
    return text


def load_metadata(adb: Path, serial: str, session: str) -> list[dict]:
    frames = session + "/frames"
    separator = b"\n__QRS_META_SPLIT__\n"
    script = (
        f'for f in "{frames}"/*_meta.json; do cat "$f"; '
        'printf "\\n__QRS_META_SPLIT__\\n"; done'
    )
    payload = run_adb(adb, serial, "exec-out", script)
    records = []
    for part in payload.split(separator):
        if not part.strip():
            continue
        try:
            records.append(json.loads(part))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid metadata stream near {part[:600]!r}") from exc
    records.sort(key=lambda item: int(item["pairIndex"]))
    if not records:
        raise RuntimeError("Session contains no complete frame metadata")
    return records


def quaternion_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def annotate_motion(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {"still": [], "slow": [], "fast": []}
    previous = None
    for record in records:
        pose = record["trackingPoses"][0]
        position = np.asarray(pose["position"], dtype=np.float64)
        rotation = np.asarray(pose["rotation"], dtype=np.float64)
        if previous is None:
            trans_mm = 0.0
            angle_deg = 0.0
        else:
            trans_mm = float(np.linalg.norm(position - previous[0]) * 1000.0)
            angle_deg = quaternion_angle_deg(rotation, previous[1])
        record["_translationMm"] = trans_mm
        record["_angleDeg"] = angle_deg
        if trans_mm <= 1.0 and angle_deg <= 0.10:
            label = "still"
        elif trans_mm >= 5.0 or angle_deg >= 0.50:
            label = "fast"
        else:
            label = "slow"
        record["_motion"] = label
        groups[label].append(record)
        previous = (position, rotation)
    return groups


def even_sample(items: list[dict], limit: int) -> list[dict]:
    if len(items) <= limit:
        return list(items)
    indices = np.linspace(0, len(items) - 1, limit, dtype=np.int64)
    return [items[int(index)] for index in indices]


def fetch_pair_batch(
    adb: Path,
    serial: str,
    session: str,
    records: list[dict],
) -> list[tuple[dict, np.ndarray, np.ndarray]]:
    frames = session + "/frames"
    names: list[str] = []
    for record in records:
        stem = f"frame_{int(record['pairIndex']):06d}"
        names += [
            f'"{frames}/{stem}_platform_pre_qrs.f32"',
            f'"{frames}/{stem}_post_qrs.f32"',
        ]
    payload = run_adb(
        adb,
        serial,
        "exec-out",
        "cat " + " ".join(names),
    )
    output: list[tuple[dict, np.ndarray, np.ndarray]] = []
    offset = 0
    for record in records:
        count = int(record["width"]) * int(record["height"]) * int(record["layers"])
        byte_count = count * 4
        if offset + byte_count * 2 > len(payload):
            raise RuntimeError("Short binary stream returned by ADB")
        raw = np.frombuffer(payload, dtype="<f4", count=count, offset=offset).copy()
        offset += byte_count
        post = np.frombuffer(payload, dtype="<f4", count=count, offset=offset).copy()
        offset += byte_count
        output.append((record, raw, post))
    if offset != len(payload):
        raise RuntimeError(f"Unexpected trailing binary bytes: {len(payload) - offset}")
    return output


def linearize(depth: np.ndarray, projection: list[float]) -> np.ndarray:
    z = depth.astype(np.float64) * 2.0 - 1.0
    return np.abs(float(projection[11]) / (z + float(projection[10])))


def add_depth_metrics(bucket: Bucket, record: dict, raw: np.ndarray, post: np.ndarray) -> None:
    raw_valid = np.isfinite(raw) & (raw > 0.0) & (raw < 1.0)
    post_valid = np.isfinite(post) & (post > 0.0) & (post < 1.0)
    common = raw_valid & post_valid
    bucket.frames += 1
    bucket.raw_valid += int(np.count_nonzero(raw_valid))
    bucket.post_valid += int(np.count_nonzero(post_valid))
    bucket.common_valid += int(np.count_nonzero(common))
    bucket.removed += int(np.count_nonzero(raw_valid & ~post_valid))
    bucket.introduced += int(np.count_nonzero(~raw_valid & post_valid))
    if np.any(common):
        raw_m = linearize(raw[common], record["projection"][0])
        post_m = linearize(post[common], record["projection"][0])
        usable = np.isfinite(raw_m) & np.isfinite(post_m) & (raw_m <= 10.0) & (post_m <= 10.0)
        bucket.diffs.append(np.abs(raw_m[usable] - post_m[usable]).astype(np.float32))


def backproject_points(record: dict, depth: np.ndarray, stride: int = 8) -> np.ndarray:
    width = int(record["width"])
    height = int(record["height"])
    image = depth.reshape(height, width)
    ys = np.arange(0, height, stride, dtype=np.int32)
    xs = np.arange(0, width, stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    sampled = image[grid_y, grid_x]
    valid = np.isfinite(sampled) & (sampled > 0.0) & (sampled < 1.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)
    u = grid_x[valid].astype(np.float64) / width
    v = grid_y[valid].astype(np.float64) / height
    ndc = sampled[valid].astype(np.float64)
    hcs = np.stack((u * 2.0 - 1.0, v * 2.0 - 1.0, ndc * 2.0 - 1.0, np.ones_like(u)))
    projection_inverse = np.asarray(record["projectionInverse"][0], dtype=np.float64).reshape(4, 4)
    view_inverse = np.asarray(record["viewInverse"][0], dtype=np.float64).reshape(4, 4)
    world_h = view_inverse @ (projection_inverse @ hcs)
    world = (world_h[:3] / world_h[3]).T
    eye = np.asarray(record["trackingPoses"][0]["position"], dtype=np.float64)
    distance = np.linalg.norm(world - eye, axis=1)
    keep = np.isfinite(world).all(axis=1) & (distance >= 0.15) & (distance <= 8.0)
    return world[keep].astype(np.float32)


def summarize_bucket(bucket: Bucket) -> dict:
    diffs = np.concatenate(bucket.diffs) if bucket.diffs else np.empty(0, dtype=np.float32)
    raw_denominator = max(bucket.raw_valid, 1)
    post_denominator = max(bucket.post_valid, 1)
    result = {
        "sampledFrames": bucket.frames,
        "commonValidPixels": bucket.common_valid,
        "rawValidRemovedPct": bucket.removed * 100.0 / raw_denominator,
        "postValidIntroducedPct": bucket.introduced * 100.0 / post_denominator,
    }
    if diffs.size:
        result.update(
            {
                "absDiffMmP50": float(np.percentile(diffs, 50) * 1000.0),
                "absDiffMmP95": float(np.percentile(diffs, 95) * 1000.0),
                "absDiffMmP99": float(np.percentile(diffs, 99) * 1000.0),
                "over10mmPct": float(np.mean(diffs > 0.010) * 100.0),
                "over20mmPct": float(np.mean(diffs > 0.020) * 100.0),
                "over30mmPct": float(np.mean(diffs > 0.030) * 100.0),
                "over50mmPct": float(np.mean(diffs > 0.050) * 100.0),
            }
        )
    return result


def local_modal_band(points: np.ndarray, normal: np.ndarray, d: float) -> dict:
    signed = points @ normal + d
    near = np.abs(signed) <= 0.060
    points = points[near]
    signed = signed[near]
    if len(points) < 500:
        return {"qualified100mmCells": 0}

    reference = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(reference, normal))) > 0.90:
        reference = np.array([1.0, 0.0, 0.0])
    tangent_u = np.cross(normal, reference)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(normal, tangent_u)
    uv = np.stack((points @ tangent_u, points @ tangent_v), axis=1)
    cells = np.floor(uv / 0.100).astype(np.int32)
    order = np.lexsort((cells[:, 1], cells[:, 0]))
    cells = cells[order]
    signed = signed[order]

    boundaries = np.flatnonzero(np.any(cells[1:] != cells[:-1], axis=1)) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(cells)]
    cell_p95: list[float] = []
    within5 = within10 = within15 = total = 0
    support_fractions: list[float] = []
    edges = np.arange(-0.061, 0.0611, 0.002)
    for start, end in zip(starts, ends):
        values = signed[start:end]
        if len(values) < 30:
            continue
        histogram, _ = np.histogram(values, bins=edges)
        peak = int(np.argmax(histogram))
        mode = float((edges[peak] + edges[peak + 1]) * 0.5)
        residual = np.abs(values - mode)
        core = residual <= 0.015
        if np.count_nonzero(core) < 20:
            continue
        core_residual = residual[core]
        cell_p95.append(float(np.percentile(core_residual, 95)))
        support_fractions.append(float(np.mean(core)))
        within5 += int(np.count_nonzero(residual <= 0.005))
        within10 += int(np.count_nonzero(residual <= 0.010))
        within15 += int(np.count_nonzero(core))
        total += len(values)

    if not cell_p95:
        return {"qualified100mmCells": 0}
    widths = np.asarray(cell_p95)
    supports = np.asarray(support_fractions)
    return {
        "qualified100mmCells": int(len(widths)),
        "cellCoreP95HalfWidthMmP50": float(np.percentile(widths, 50) * 1000.0),
        "cellCoreP95HalfWidthMmP90": float(np.percentile(widths, 90) * 1000.0),
        "dominantCoreFractionCellP50Pct": float(np.percentile(supports, 50) * 100.0),
        "dominantCoreFractionCellP10Pct": float(np.percentile(supports, 10) * 100.0),
        "allQualifiedPointsWithin5mmPct": within5 * 100.0 / max(total, 1),
        "allQualifiedPointsWithin10mmPct": within10 * 100.0 / max(total, 1),
        "allQualifiedPointsWithin15mmPct": within15 * 100.0 / max(total, 1),
    }


def fit_planes(points: np.ndarray, maximum_planes: int = 5) -> dict:
    if len(points) < 1000:
        return {"inputPoints": int(len(points)), "planes": []}
    import open3d as o3d

    o3d.utility.random.seed(42)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    cloud = cloud.voxel_down_sample(voxel_size=0.015)
    remaining = cloud
    planes: list[dict] = []
    for index in range(maximum_planes):
        if len(remaining.points) < 1000:
            break
        model, inliers = remaining.segment_plane(
            distance_threshold=0.018,
            ransac_n=3,
            num_iterations=700,
            probability=0.999,
        )
        if len(inliers) < 700:
            break
        normal = np.asarray(model[:3], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        d = float(model[3]) / np.linalg.norm(np.asarray(model[:3], dtype=np.float64))
        all_points = np.asarray(remaining.points)
        residual = np.abs(all_points @ normal + d)
        inlier_residual = residual[np.asarray(inliers, dtype=np.int64)]
        near = residual <= 0.060
        near_points = all_points[near]
        near_residual = residual[near]
        orientation = "horizontal" if abs(normal[1]) >= 0.80 else "vertical_or_sloped"
        planes.append(
            {
                "rank": index + 1,
                "orientation": orientation,
                "ransacInliers": int(len(inliers)),
                "ransacBandAbsMmP50": float(np.percentile(inlier_residual, 50) * 1000.0),
                "ransacBandAbsMmP90": float(np.percentile(inlier_residual, 90) * 1000.0),
                "ransacBandAbsMmP95": float(np.percentile(inlier_residual, 95) * 1000.0),
                "near60mmPoints": int(len(near_points)),
                "bandAbsMmP50": float(np.percentile(near_residual, 50) * 1000.0),
                "bandAbsMmP90": float(np.percentile(near_residual, 90) * 1000.0),
                "bandAbsMmP95": float(np.percentile(near_residual, 95) * 1000.0),
                "normal": [float(value) for value in normal],
                "planeOffset": d,
                "localModalBand": local_modal_band(all_points, normal, d),
            }
        )
        remaining = remaining.select_by_index(inliers, invert=True)
    return {
        "inputPoints": int(len(points)),
        "voxelDownsampledPoints": int(len(cloud.points)),
        "planes": planes,
    }


def motion_summary(groups: dict[str, list[dict]]) -> dict:
    result = {}
    for label, records in groups.items():
        translations = np.asarray([item["_translationMm"] for item in records])
        angles = np.asarray([item["_angleDeg"] for item in records])
        result[label] = {
            "frames": len(records),
            "translationMmP50": float(np.percentile(translations, 50)) if len(records) else None,
            "translationMmP95": float(np.percentile(translations, 95)) if len(records) else None,
            "angleDegP50": float(np.percentile(angles, 50)) if len(records) else None,
            "angleDegP95": float(np.percentile(angles, 95)) if len(records) else None,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb", type=Path, default=DEFAULT_ADB)
    parser.add_argument("--serial", default="2G97C5ZH4501R5")
    parser.add_argument("--session", default="")
    parser.add_argument("--sample-per-motion", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()

    session = args.session or latest_session(args.adb, args.serial)
    print(f"Reading device session: {session}", flush=True)
    records = load_metadata(args.adb, args.serial, session)
    groups = annotate_motion(records)
    selected: list[dict] = []
    for label in ("still", "slow", "fast"):
        selected += even_sample(groups[label], args.sample_per_motion)
    selected.sort(key=lambda item: int(item["pairIndex"]))

    buckets = {label: Bucket() for label in ("all", "still", "slow", "fast")}
    world_points: list[np.ndarray] = []
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start : start + args.batch_size]
        for record, raw, post in fetch_pair_batch(args.adb, args.serial, session, batch):
            add_depth_metrics(buckets["all"], record, raw, post)
            add_depth_metrics(buckets[record["_motion"]], record, raw, post)
            points = backproject_points(record, raw, stride=8)
            if len(points):
                world_points.append(points)

    point_array = np.concatenate(world_points) if world_points else np.empty((0, 3), dtype=np.float32)
    report = {
        "session": session,
        "completePairs": len(records),
        "motionDefinition": {
            "still": "translation<=1mm and rotation<=0.10deg",
            "fast": "translation>=5mm or rotation>=0.50deg",
            "slow": "between still and fast",
        },
        "motion": motion_summary(groups),
        "selectedPairs": len(selected),
        "depthDifference": {label: summarize_bucket(bucket) for label, bucket in buckets.items()},
        "rawObservationPlaneBands": fit_planes(point_array),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
