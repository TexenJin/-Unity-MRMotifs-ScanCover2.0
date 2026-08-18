#!/usr/bin/env python3
"""Audit the frozen Quest normal buffers without changing TSDF or DMC.

The audit is deliberately truth-free.  It checks only properties that the
captured evidence can support directly: availability, agreement with a local
surface differential, and cross-eye/cross-frame coherence inside the same
4.5 cm production voxel.  Absolute room accuracy remains out of scope.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ScanCoverEvidenceV3OfflineValidation import load_frame, read_manifest, select_evenly


BUFFERS = {
    "depth_metrics_rgba32f",
    "world_position_raw_rgba32f",
    "world_normal_raw_rgba32f",
    "world_normal_neighbour_rgba32f",
}

DEPTH_EDGES = (0.35, 0.75, 1.5, 3.0, 5.0)
INCIDENCE_EDGES = (0.0, 30.0, 50.0, 70.0, 90.0001)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, nargs="*", default=None)
    parser.add_argument("--frames-per-session", type=int, default=24)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--bucket-stride", type=int, default=2)
    parser.add_argument("--voxel", type=float, default=0.045)
    parser.add_argument("--min-depth", type=float, default=0.35)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--verify-crc", action="store_true")
    return parser.parse_args()


def resolve_sessions(args: argparse.Namespace) -> list[Path]:
    if args.sessions:
        sessions = [path.resolve() for path in args.sessions]
    else:
        sessions = sorted(path.resolve() for path in args.root.glob("Evidence_*") if path.is_dir())
    if not sessions:
        raise RuntimeError(f"no Evidence_* sessions found under {args.root}")
    return sessions


def normalize_image(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(values[..., :3], dtype=np.float64)
    lengths = np.linalg.norm(xyz, axis=-1)
    valid = np.all(np.isfinite(xyz), axis=-1) & (lengths > 1e-6)
    result = np.zeros_like(xyz)
    result[valid] = xyz[valid] / lengths[valid, None]
    return result, valid


def acute_angle_degrees(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    dots = np.sum(first * second, axis=-1)
    return np.degrees(np.arccos(np.clip(np.abs(dots), 0.0, 1.0)))


def distribution(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "median": None, "p90": None, "p95": None}
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90.0)),
        "p95": float(np.percentile(finite, 95.0)),
    }


def bin_rows(
    selector: np.ndarray,
    edges: tuple[float, ...],
    raw_neighbour: np.ndarray,
    raw_local: np.ndarray,
    neighbour_local: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (selector >= lower) & (selector < upper)
        rows.append(
            {
                "lower": lower,
                "upper": upper,
                "samples": int(np.count_nonzero(mask)),
                "rawVsNeighbourDegrees": distribution(raw_neighbour[mask]),
                "rawVsLocalDegrees": distribution(raw_local[mask]),
                "neighbourVsLocalDegrees": distribution(neighbour_local[mask]),
            }
        )
    return rows


def bucket_update(
    buckets: dict[tuple[int, int, int], dict[str, Any]],
    point: np.ndarray,
    raw: np.ndarray,
    neighbour: np.ndarray,
    voxel: float,
) -> None:
    key = tuple(np.floor(point / voxel).astype(np.int64).tolist())
    record = buckets.get(key)
    if record is None:
        buckets[key] = {
            "count": 1,
            "rawRef": raw.copy(),
            "neighbourRef": neighbour.copy(),
            "rawSum": raw.copy(),
            "neighbourSum": neighbour.copy(),
        }
        return
    raw_aligned = raw if float(np.dot(raw, record["rawRef"])) >= 0.0 else -raw
    neighbour_aligned = (
        neighbour
        if float(np.dot(neighbour, record["neighbourRef"])) >= 0.0
        else -neighbour
    )
    record["count"] += 1
    record["rawSum"] += raw_aligned
    record["neighbourSum"] += neighbour_aligned


def analyse_session(args: argparse.Namespace, session: Path) -> dict[str, Any]:
    rows = select_evenly(read_manifest(session), args.frames_per_session)
    if not rows:
        raise RuntimeError(f"no valid frames in {session}")

    raw_neighbour_parts: list[np.ndarray] = []
    raw_local_parts: list[np.ndarray] = []
    neighbour_local_parts: list[np.ndarray] = []
    depth_parts: list[np.ndarray] = []
    incidence_parts: list[np.ndarray] = []
    buckets: dict[tuple[int, int, int], dict[str, Any]] = {}
    counters: defaultdict[str, int] = defaultdict(int)
    stride = max(1, int(args.pixel_stride))
    bucket_stride = max(1, int(args.bucket_stride))

    for selected_index, row in enumerate(rows, start=1):
        frame_path = (session / row["file"]).resolve()
        frame = load_frame(frame_path, BUFFERS, args.verify_crc)
        metrics = frame.buffers["depth_metrics_rgba32f"]
        points = frame.buffers["world_position_raw_rgba32f"]
        raw_buffer = frame.buffers["world_normal_raw_rgba32f"]
        neighbour_buffer = frame.buffers["world_normal_neighbour_rgba32f"]
        if metrics.shape[0] != 2:
            raise RuntimeError(f"complete binocular frame required: {frame_path}")

        for eye in range(2):
            camera = np.asarray(frame.metadata["eyeWorldPositions"][eye], dtype=np.float64)
            metric = metrics[eye, ::stride, ::stride]
            point4 = points[eye, ::stride, ::stride]
            raw4 = raw_buffer[eye, ::stride, ::stride]
            neighbour4 = neighbour_buffer[eye, ::stride, ::stride]
            point = np.asarray(point4[..., :3], dtype=np.float64)
            raw, raw_finite = normalize_image(raw4)
            neighbour, neighbour_finite = normalize_image(neighbour4)
            radial = np.asarray(metric[..., 2], dtype=np.float64)
            depth_valid = (
                (metric[..., 3] > 0.0)
                & np.isfinite(radial)
                & (radial >= args.min_depth)
                & (radial <= args.max_depth)
                & (point4[..., 3] > 0.0)
                & np.all(np.isfinite(point), axis=-1)
            )
            raw_valid = depth_valid & (raw4[..., 3] > 0.0) & raw_finite
            neighbour_valid = depth_valid & (neighbour4[..., 3] > 0.0) & neighbour_finite
            common = raw_valid & neighbour_valid
            counters["depthValid"] += int(np.count_nonzero(depth_valid))
            counters["rawValid"] += int(np.count_nonzero(raw_valid))
            counters["neighbourValid"] += int(np.count_nonzero(neighbour_valid))
            counters["commonValid"] += int(np.count_nonzero(common))

            dx = np.roll(point, -1, axis=1) - np.roll(point, 1, axis=1)
            dy = np.roll(point, -1, axis=0) - np.roll(point, 1, axis=0)
            local = np.cross(dx, dy)
            local_lengths = np.linalg.norm(local, axis=-1)
            local_valid = depth_valid & np.isfinite(local_lengths) & (local_lengths > 1e-8)
            neighbour_depth_valid = (
                np.roll(depth_valid, 1, axis=0)
                & np.roll(depth_valid, -1, axis=0)
                & np.roll(depth_valid, 1, axis=1)
                & np.roll(depth_valid, -1, axis=1)
            )
            local_valid &= neighbour_depth_valid
            local_valid[[0, -1], :] = False
            local_valid[:, [0, -1]] = False
            local_normal = np.zeros_like(local)
            local_normal[local_valid] = local[local_valid] / local_lengths[local_valid, None]
            compare = common & local_valid

            if np.any(compare):
                to_camera = camera[None, None, :] - point
                to_camera_length = np.linalg.norm(to_camera, axis=-1)
                view = np.zeros_like(to_camera)
                view_valid = np.isfinite(to_camera_length) & (to_camera_length > 1e-8)
                view[view_valid] = to_camera[view_valid] / to_camera_length[view_valid, None]
                incidence = np.degrees(
                    np.arccos(np.clip(np.abs(np.sum(neighbour * view, axis=-1)), 0.0, 1.0))
                )
                raw_neighbour_parts.append(acute_angle_degrees(raw[compare], neighbour[compare]))
                raw_local_parts.append(acute_angle_degrees(raw[compare], local_normal[compare]))
                neighbour_local_parts.append(
                    acute_angle_degrees(neighbour[compare], local_normal[compare])
                )
                depth_parts.append(radial[compare])
                incidence_parts.append(incidence[compare])
                counters["localComparable"] += int(np.count_nonzero(compare))

            bucket_mask = common.copy()
            bucket_mask[::bucket_stride, ::bucket_stride] &= True
            sparse = np.zeros_like(bucket_mask)
            sparse[::bucket_stride, ::bucket_stride] = bucket_mask[::bucket_stride, ::bucket_stride]
            for y, x in np.argwhere(sparse):
                bucket_update(buckets, point[y, x], raw[y, x], neighbour[y, x], args.voxel)

        print(
            f"[normal-contract {session.name} {selected_index}/{len(rows)}] "
            f"common={counters['commonValid']} buckets={len(buckets)}",
            flush=True,
        )

    raw_neighbour = np.concatenate(raw_neighbour_parts) if raw_neighbour_parts else np.empty(0)
    raw_local = np.concatenate(raw_local_parts) if raw_local_parts else np.empty(0)
    neighbour_local = (
        np.concatenate(neighbour_local_parts) if neighbour_local_parts else np.empty(0)
    )
    depth = np.concatenate(depth_parts) if depth_parts else np.empty(0)
    incidence = np.concatenate(incidence_parts) if incidence_parts else np.empty(0)
    raw_coherence = []
    neighbour_coherence = []
    bucket_counts = []
    for record in buckets.values():
        if record["count"] < 4:
            continue
        count = float(record["count"])
        raw_coherence.append(float(np.linalg.norm(record["rawSum"]) / count))
        neighbour_coherence.append(float(np.linalg.norm(record["neighbourSum"]) / count))
        bucket_counts.append(int(record["count"]))

    depth_valid = max(1, counters["depthValid"])
    report = {
        "session": session.name,
        "selectedFrames": len(rows),
        "effectiveRasterPerEye": [
            int(math.ceil(metrics.shape[2] / stride)),
            int(math.ceil(metrics.shape[1] / stride)),
        ],
        "availability": {
            **dict(counters),
            "rawValidRatioOfDepth": counters["rawValid"] / depth_valid,
            "neighbourValidRatioOfDepth": counters["neighbourValid"] / depth_valid,
            "commonValidRatioOfDepth": counters["commonValid"] / depth_valid,
        },
        "angleDistributionsDegrees": {
            "rawVsNeighbour": distribution(raw_neighbour),
            "rawVsLocalDifferential": distribution(raw_local),
            "neighbourVsLocalDifferential": distribution(neighbour_local),
        },
        "depthBinsMeters": bin_rows(
            depth, DEPTH_EDGES, raw_neighbour, raw_local, neighbour_local
        ),
        "incidenceBinsDegrees": bin_rows(
            incidence, INCIDENCE_EDGES, raw_neighbour, raw_local, neighbour_local
        ),
        "productionVoxelCoherence": {
            "voxelMeters": args.voxel,
            "minimumSamplesPerVoxel": 4,
            "qualifiedVoxels": len(raw_coherence),
            "samplesPerVoxel": distribution(np.asarray(bucket_counts, dtype=np.float64)),
            "rawResultantLength": distribution(np.asarray(raw_coherence, dtype=np.float64)),
            "neighbourResultantLength": distribution(
                np.asarray(neighbour_coherence, dtype=np.float64)
            ),
            "interpretation": "1 is fully coherent; lower values indicate angular disagreement inside one production voxel.",
        },
    }
    return report


def mean_optional(reports: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    values = []
    for report in reports:
        value: Any = report
        for key in path:
            value = value[key]
        if value is not None:
            values.append(float(value))
    return float(np.mean(values)) if values else None


def format_optional(value: float | None, digits: int = 4, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reports = [analyse_session(args, session) for session in resolve_sessions(args)]
    summary = {
        "schema": "scancover.quest_evidence_v3.normal_input_contract_audit.v1",
        "scope": "truth-free normal availability/local differential/spatial coherence audit",
        "parameters": {
            "framesPerSession": args.frames_per_session,
            "pixelStride": args.pixel_stride,
            "bucketStride": args.bucket_stride,
            "voxelMeters": args.voxel,
        },
        "sessions": reports,
        "means": {
            "rawValidRatioOfDepth": mean_optional(reports, ("availability", "rawValidRatioOfDepth")),
            "neighbourValidRatioOfDepth": mean_optional(
                reports, ("availability", "neighbourValidRatioOfDepth")
            ),
            "rawVsNeighbourMedianDegrees": mean_optional(
                reports, ("angleDistributionsDegrees", "rawVsNeighbour", "median")
            ),
            "rawLocalP95Degrees": mean_optional(
                reports, ("angleDistributionsDegrees", "rawVsLocalDifferential", "p95")
            ),
            "neighbourLocalP95Degrees": mean_optional(
                reports,
                ("angleDistributionsDegrees", "neighbourVsLocalDifferential", "p95"),
            ),
            "rawVoxelCoherenceMedian": mean_optional(
                reports, ("productionVoxelCoherence", "rawResultantLength", "median")
            ),
            "neighbourVoxelCoherenceMedian": mean_optional(
                reports,
                ("productionVoxelCoherence", "neighbourResultantLength", "median"),
            ),
        },
        "decisionRule": {
            "candidateOnly": True,
            "preferredSourceRequires": [
                "no material validity loss",
                "lower local-differential p95 angle",
                "higher production-voxel coherence",
                "independent DMC topology A/B confirmation",
            ],
            "absoluteAccuracyClaimed": False,
        },
    }
    output = args.out / "quest_normal_input_contract_audit.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    means = summary["means"]
    markdown = [
        "# Quest 法线输入契约审计",
        "",
        "本报告不使用房间真值，只检查实机法线的可用性、局部几何一致性和生产体素内跨眼/跨帧一致性。",
        "",
        "| 指标 | Raw 法线 | Neighbour 法线 |",
        "| --- | ---: | ---: |",
        f"| 有效深度上的可用率 | {format_optional(means['rawValidRatioOfDepth'])} | {format_optional(means['neighbourValidRatioOfDepth'])} |",
        f"| 相对局部微分法线 P95 | {format_optional(means['rawLocalP95Degrees'], 2, '°')} | {format_optional(means['neighbourLocalP95Degrees'], 2, '°')} |",
        f"| 4.5 cm 体素内法线相干中位数 | {format_optional(means['rawVoxelCoherenceMedian'])} | {format_optional(means['neighbourVoxelCoherenceMedian'])} |",
        "",
        f"Raw 与 Neighbour 法线的中位夹角为 {format_optional(means['rawVsNeighbourMedianDegrees'], 2, '°')}。",
        "",
        "最终输入源不能由本表单独决定，必须再通过同深度、同帧、同 DMC 的拓扑 A/B。",
    ]
    (args.out / "quest_normal_input_contract_audit.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["means"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
