#!/usr/bin/env python3
"""Audit how GunGel evidence is converted into virtual-probe witnesses.

This is a read-only diagnostic.  It never edits Unity assets, production
configuration, TSDF state, paper state, or mesh state.

The audit compares two device-recorded moments of the same frame:

* ``probe_shadow``: the virtual probe verdict ledger built from the
  pre-candidate-transaction correspondence snapshot;
* ``fusion_inputs``: the final per-point correspondence buffer saved after the
  candidate transaction and production correspondence rebuild.

For every final ``hold`` cell the tool asks whether the final stream contained
two genuinely independent opportunities at each cumulative gate:

    stable match -> raw depth -> dual agreement -> unopposed
    -> motion quality -> view radius -> eligible support

Independence mirrors ``IndependentViewWitnessPolicy`` for the *second* witness:
frame gap >= 2 and (camera baseline >= 8 cm or view spread >= 3 degrees).
Reaching the second witness only needs comparison with the first witness, so
this test is exact for the 1 -> 2 transition even though production stores up
to four witnesses.

Important boundary: the final correspondence buffer is a downstream snapshot.
It can prove that evidence existed after the candidate transaction; it cannot
retroactively prove that the pre-transaction shadow ledger saw the same target.
That difference is precisely the conversion/order gap this audit is intended
to expose.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


CELL_METRES = 0.05
SAFE_MOTION_QUALITY = 0.70
AUTHORITY_OUTER_RADIUS = 0.84
INDEPENDENT_BASELINE_METRES = 0.08
INDEPENDENT_ANGLE_DEG = 3.0
INDEPENDENT_FRAME_GAP = 2

OBSERVATION_DTYPE = np.dtype(
    [
        ("positionSigma", "<f4", (4,)),
        ("normalQuality", "<f4", (4,)),
        ("rawPositionDelta", "<f4", (4,)),
        ("sourceReason", "<u4", (4,)),
    ],
    align=False,
)

CORRESPONDENCE_DTYPE = np.dtype(
    [
        ("sourceValid", "<f4", (4,)),
        ("targetSigma", "<f4", (4,)),
        ("normalAngle", "<f4", (4,)),
    ],
    align=False,
)

OBSERVATION_VALID = 1 << 0
RAW_AVAILABLE = 1 << 1
DUAL_AGREE = 1 << 2
STABLE_MATCH = 1 << 3
UNOPPOSED = 1 << 5

STAGES = (
    "stable_match",
    "raw_available",
    "dual_agree",
    "unopposed",
    "motion_safe",
    "view_safe",
    "eligible_support",
)


def number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def safe_direction(value: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if length <= 1e-10:
        return np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    return value / length


def axis_from_normal(normal: np.ndarray) -> np.ndarray:
    absolute = np.abs(normal)
    major = np.argmax(absolute, axis=1)
    result = np.empty(len(normal), dtype=np.int16)
    result[major == 0] = np.where(normal[major == 0, 0] >= 0.0, 0, 1)
    result[major == 1] = np.where(normal[major == 1, 1] >= 0.0, 2, 3)
    result[major == 2] = np.where(normal[major == 2, 2] >= 0.0, 4, 5)
    return result


def view_radius_from_index(indices: np.ndarray, grid_x: int, pixel_stride: int,
                           width: int, height: int) -> np.ndarray:
    gx = indices % max(grid_x, 1)
    gy = indices // max(grid_x, 1)
    px = np.minimum(gx * max(pixel_stride, 1), max(width - 1, 0))
    py = np.minimum(gy * max(pixel_stride, 1), max(height - 1, 0))
    nx = px.astype(np.float64) / max(width, 1) * 2.0 - 1.0
    ny = py.astype(np.float64) / max(height, 1) * 2.0 - 1.0
    return np.maximum(np.abs(nx), np.abs(ny))


@dataclass
class SecondWitness:
    frames: int = 0
    first_frame: int = -1
    first_camera: Optional[np.ndarray] = field(default=None, repr=False)
    first_direction: Optional[np.ndarray] = field(default=None, repr=False)
    second_frame: int = -1
    second_baseline_m: float = 0.0
    second_spread_deg: float = 0.0
    max_baseline_m: float = 0.0
    max_spread_deg: float = 0.0

    @property
    def has_second(self) -> bool:
        return self.second_frame >= 0

    def observe(self, frame: int, camera: np.ndarray, center: np.ndarray) -> None:
        self.frames += 1
        direction = safe_direction(camera - center)
        if self.first_camera is None or self.first_direction is None:
            self.first_frame = frame
            self.first_camera = camera.copy()
            self.first_direction = direction.copy()
            return
        baseline = float(np.linalg.norm(camera - self.first_camera))
        spread = angle_deg(direction, self.first_direction)
        self.max_baseline_m = max(self.max_baseline_m, baseline)
        self.max_spread_deg = max(self.max_spread_deg, spread)
        if self.has_second:
            return
        time_separated = frame - self.first_frame >= INDEPENDENT_FRAME_GAP
        pose_separated = baseline >= INDEPENDENT_BASELINE_METRES or spread >= INDEPENDENT_ANGLE_DEG
        if time_separated and pose_separated:
            self.second_frame = frame
            self.second_baseline_m = baseline
            self.second_spread_deg = spread


@dataclass
class HeldCell:
    key: tuple[int, int, int, int]
    center: np.ndarray
    normal: np.ndarray
    first_frame: int
    last_seen_frame: int
    safe_support_frames: int
    independent_support_views: int
    max_baseline_mm: float
    max_view_spread_deg: float
    fingerprint_id: int
    fingerprint_generation: int
    stages: dict[str, SecondWitness] = field(
        default_factory=lambda: {name: SecondWitness() for name in STAGES}
    )
    race_decisions: Counter = field(default_factory=Counter)
    race_event_rows: int = 0
    post_observations: int = 0


def load_held_cells(path: Path) -> dict[tuple[int, int, int, int], HeldCell]:
    result: dict[tuple[int, int, int, int], HeldCell] = {}
    for row in read_csv(path):
        if row.get("verdict", "").strip().lower() != "hold":
            continue
        key = (
            integer(row.get("cellX")),
            integer(row.get("cellY")),
            integer(row.get("cellZ")),
            integer(row.get("axis")),
        )
        result[key] = HeldCell(
            key=key,
            center=np.asarray(
                (number(row.get("centerX")), number(row.get("centerY")), number(row.get("centerZ"))),
                dtype=np.float64,
            ),
            normal=safe_direction(
                np.asarray(
                    (number(row.get("normalX")), number(row.get("normalY")), number(row.get("normalZ"))),
                    dtype=np.float64,
                )
            ),
            first_frame=integer(row.get("firstFrame"), -1),
            last_seen_frame=integer(row.get("lastSeenFrame"), 2**31 - 1),
            safe_support_frames=integer(row.get("safeSupportFrames")),
            independent_support_views=integer(row.get("independentSupportViews")),
            max_baseline_mm=number(row.get("maxBaselineMm")),
            max_view_spread_deg=number(row.get("maxViewSpreadDeg")),
            fingerprint_id=integer(row.get("fingerprintId")),
            fingerprint_generation=integer(row.get("fingerprintGeneration")),
        )
    return result


def load_probe_frames(path: Path) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for row in read_csv(path):
        frame = integer(row.get("gunGelFrame"), -1)
        if frame < 0:
            continue
        result[frame] = {
            "cameraX": number(row.get("sourceCameraX")),
            "cameraY": number(row.get("sourceCameraY")),
            "cameraZ": number(row.get("sourceCameraZ")),
            "motionQuality": number(row.get("motionQuality")),
            "gridX": integer(row.get("gridX"), 40),
            "gridY": integer(row.get("gridY"), 40),
            "pixelStride": integer(row.get("pixelStride"), 8),
            "depthWidth": integer(row.get("depthWidth"), 320),
            "depthHeight": integer(row.get("depthHeight"), 320),
        }
    return result


def audit_race_events(path: Path, held: dict[tuple[int, int, int, int], HeldCell]) -> dict[str, object]:
    total = 0
    matched = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            key = (
                integer(row.get("cellX")),
                integer(row.get("cellY")),
                integer(row.get("cellZ")),
                integer(row.get("axis")),
            )
            cell = held.get(key)
            if cell is None:
                continue
            matched += 1
            cell.race_event_rows += 1
            cell.race_decisions[row.get("productionDecision", "unknown") or "unknown"] += 1
    return {"totalRows": total, "heldRows": matched}


def load_structured(path: Path, dtype: np.dtype) -> np.ndarray:
    size = path.stat().st_size
    if size % dtype.itemsize != 0:
        raise IOError(f"{path.name}: {size} bytes is not divisible by stride {dtype.itemsize}")
    return np.fromfile(path, dtype=dtype)


def per_frame_levels(
    observations: np.ndarray,
    correspondences: np.ndarray,
    motion_quality: float,
    grid_x: int,
    pixel_stride: int,
    width: int,
    height: int,
) -> dict[tuple[int, int, int, int], int]:
    count = min(len(observations), len(correspondences))
    if count <= 0:
        return {}
    flags = np.rint(correspondences["sourceValid"][:count, 3]).astype(np.int64)
    target = correspondences["targetSigma"][:count, :3].astype(np.float64)
    normal = correspondences["normalAngle"][:count, :3].astype(np.float64)
    indices = np.arange(count, dtype=np.int64)
    valid_target = (
        ((flags & OBSERVATION_VALID) != 0)
        & ((flags & STABLE_MATCH) != 0)
        & np.isfinite(target).all(axis=1)
        & np.isfinite(normal).all(axis=1)
        & (np.linalg.norm(normal, axis=1) > 1e-5)
    )
    selected = np.flatnonzero(valid_target)
    if selected.size == 0:
        return {}
    selected_target = target[selected]
    selected_normal = normal[selected]
    cell = np.floor(selected_target / CELL_METRES).astype(np.int32)
    axis = axis_from_normal(selected_normal)
    radius = view_radius_from_index(selected, grid_x, pixel_stride, width, height)
    selected_flags = flags[selected]

    result: dict[tuple[int, int, int, int], int] = {}
    for local_index in range(selected.size):
        key = (
            int(cell[local_index, 0]),
            int(cell[local_index, 1]),
            int(cell[local_index, 2]),
            int(axis[local_index]),
        )
        point_flags = int(selected_flags[local_index])
        level = 1
        if (point_flags & RAW_AVAILABLE) != 0:
            level = 2
            if (point_flags & DUAL_AGREE) != 0:
                level = 3
                if (point_flags & UNOPPOSED) != 0:
                    level = 4
                    if motion_quality >= SAFE_MOTION_QUALITY:
                        level = 5
                        if float(radius[local_index]) < AUTHORITY_OUTER_RADIUS:
                            level = 7
                        else:
                            level = 5
        previous = result.get(key, 0)
        if level > previous:
            result[key] = level
    return result


def stage_names_for_level(level: int) -> tuple[str, ...]:
    if level <= 0:
        return ()
    names = ["stable_match"]
    if level >= 2:
        names.append("raw_available")
    if level >= 3:
        names.append("dual_agree")
    if level >= 4:
        names.append("unopposed")
    if level >= 5:
        names.append("motion_safe")
    if level >= 7:
        names.extend(("view_safe", "eligible_support"))
    return tuple(names)


def audit_fusion_stream(
    session: Path,
    held: dict[tuple[int, int, int, int], HeldCell],
    probe_frames: dict[int, dict[str, float]],
    frame_stride: int,
) -> dict[str, object]:
    root = session / "fusion_inputs"
    frames_root = root / "frames"
    rows = [
        row for row in read_csv(root / "manifest.csv")
        if row.get("gunGelObservationsFile") and row.get("gunGelCorrespondencesFile")
    ]
    selected_rows = rows[:: max(1, frame_stride)]
    processed = 0
    missing_probe = 0
    errors: list[str] = []
    mapped_frame_cells = 0
    unmapped_frame_cells = 0
    for row_index, row in enumerate(selected_rows, start=1):
        frame = integer(row.get("gunGelFrame"), -1)
        probe = probe_frames.get(frame)
        if probe is None:
            missing_probe += 1
            continue
        observation_path = frames_root / row["gunGelObservationsFile"]
        correspondence_path = frames_root / row["gunGelCorrespondencesFile"]
        try:
            observations = load_structured(observation_path, OBSERVATION_DTYPE)
            correspondences = load_structured(correspondence_path, CORRESPONDENCE_DTYPE)
            levels = per_frame_levels(
                observations,
                correspondences,
                probe["motionQuality"],
                int(probe["gridX"]),
                int(probe["pixelStride"]),
                int(probe["depthWidth"]),
                int(probe["depthHeight"]),
            )
            camera = np.asarray(
                (probe["cameraX"], probe["cameraY"], probe["cameraZ"]),
                dtype=np.float64,
            )
            for key, level in levels.items():
                cell = held.get(key)
                if cell is None:
                    unmapped_frame_cells += 1
                    continue
                if frame < cell.first_frame or frame > cell.last_seen_frame:
                    continue
                mapped_frame_cells += 1
                cell.post_observations += 1
                for stage in stage_names_for_level(level):
                    cell.stages[stage].observe(frame, camera, cell.center)
            processed += 1
            if row_index % 500 == 0:
                print(
                    f"fusion {row_index}/{len(selected_rows)} processed={processed} "
                    f"mapped={mapped_frame_cells}",
                    flush=True,
                )
        except Exception as exc:  # keep the audit closed and report every loss
            errors.append(f"sequence={row.get('sequence')} frame={frame}: {exc}")
    return {
        "manifestRows": len(rows),
        "selectedRows": len(selected_rows),
        "processedRows": processed,
        "missingProbeRows": missing_probe,
        "mappedFrameCells": mapped_frame_cells,
        "unmappedFrameCells": unmapped_frame_cells,
        "errors": errors,
    }


def classify(cell: HeldCell) -> tuple[str, str, bool]:
    second = {name: cell.stages[name].has_second for name in STAGES}
    if cell.independent_support_views >= 2:
        return (
            "ledger_contract_inconsistency",
            "裁决账已记录至少两独立证词但最终仍为 Hold，应核对状态/导出契约。",
            True,
        )
    if second["eligible_support"]:
        return (
            "pre_post_transaction_conversion_gap",
            "候选事务后的最终缓冲已有第二独立合格证词，但事务前探针账仍不足；责任落在候选更新/重建与探针消费顺序或身份交接。",
            True,
        )
    if not second["stable_match"]:
        matched_frames = cell.stages["stable_match"].frames
        if matched_frames <= 0:
            return (
                "no_post_transaction_stable_match",
                "候选事务后的已接收融合流里也没有可归属到该 Hold 单元的稳定匹配；责任仍在候选身份/几何法线匹配或未被接收的帧，尚未走到证词独立性。",
                False,
            )
        if matched_frames == 1:
            return (
                "single_post_transaction_stable_match",
                "最终流只给该单元留下一帧稳定匹配，客观上不可能组成第二独立证词。",
                False,
            )
        return (
            "correlated_post_transaction_stable_matches",
            "最终流反复匹配到该单元，但相对第一票始终未同时满足时间间隔与基线/夹角条件；这是相关视角，不是计数器漏票。",
            True,
        )
    if not second["raw_available"]:
        return (
            "raw_witness_bottleneck",
            "稳定匹配有第二独立机会，但平台原始深度没有随同通过。",
            True,
        )
    if not second["dual_agree"]:
        return (
            "dual_agreement_bottleneck",
            "两路都在，但原始深度与预处理深度未在第二独立视角达成一致。",
            True,
        )
    if not second["unopposed"]:
        return (
            "opposition_bottleneck",
            "第二独立几何机会存在，但候选仍带反对票，未形成无反对支持。",
            True,
        )
    if not second["motion_safe"]:
        return (
            "motion_bottleneck",
            "证据链已到无反对匹配，但第二独立机会发生在运动质量保护线以下。",
            True,
        )
    if not second["view_safe"]:
        return (
            "view_radius_bottleneck",
            "证据与运动均合格，但第二独立机会只落在视野外圈。",
            True,
        )
    return (
        "unclassified_conversion_gap",
        "阶段计数未闭合，需要检查缓冲丢失或键/生命周期映射。",
        False,
    )


def build_rows(held: dict[tuple[int, int, int, int], HeldCell]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in sorted(held):
        cell = held[key]
        category, explanation, strong = classify(cell)
        row: dict[str, object] = {
            "cellX": key[0],
            "cellY": key[1],
            "cellZ": key[2],
            "axis": key[3],
            "centerX": cell.center[0],
            "centerY": cell.center[1],
            "centerZ": cell.center[2],
            "firstFrame": cell.first_frame,
            "lastSeenFrame": cell.last_seen_frame,
            "shadowSafeSupportFrames": cell.safe_support_frames,
            "shadowIndependentSupportViews": cell.independent_support_views,
            "shadowMaxBaselineMm": cell.max_baseline_mm,
            "shadowMaxViewSpreadDeg": cell.max_view_spread_deg,
            "raceEventRows": cell.race_event_rows,
            "raceAccepted": cell.race_decisions.get("accepted", 0),
            "raceCorrelatedTime": cell.race_decisions.get("correlated_time", 0),
            "raceCorrelatedPose": cell.race_decisions.get("correlated_pose", 0),
            "raceCorrelatedTimeAndPose": cell.race_decisions.get("correlated_time_and_pose", 0),
            "raceCapacityFull": cell.race_decisions.get("capacity_full", 0),
            "postMappedFrameObservations": cell.post_observations,
            "classification": category,
            "strongEvidence": int(strong),
            "explanation": explanation,
            "fingerprintId": cell.fingerprint_id,
            "fingerprintGeneration": cell.fingerprint_generation,
        }
        for stage in STAGES:
            witness = cell.stages[stage]
            prefix = "post_" + stage
            row[prefix + "Frames"] = witness.frames
            row[prefix + "HasSecond"] = int(witness.has_second)
            row[prefix + "FirstFrame"] = witness.first_frame
            row[prefix + "SecondFrame"] = witness.second_frame
            row[prefix + "SecondBaselineMm"] = witness.second_baseline_m * 1000.0
            row[prefix + "SecondSpreadDeg"] = witness.second_spread_deg
            row[prefix + "MaxBaselineMm"] = witness.max_baseline_m * 1000.0
            row[prefix + "MaxSpreadDeg"] = witness.max_spread_deg
        rows.append(row)
    return rows


def write_report(
    output: Path,
    rows: list[dict[str, object]],
    race_summary: dict[str, object],
    fusion_summary: dict[str, object],
) -> dict[str, object]:
    classifications = Counter(str(row["classification"]) for row in rows)
    strong = sum(integer(row["strongEvidence"]) for row in rows)
    stage_rows: list[dict[str, object]] = []
    for stage in STAGES:
        stage_rows.append(
            {
                "stage": stage,
                "cellsWithAnyFrame": sum(integer(row[f"post_{stage}Frames"]) > 0 for row in rows),
                "cellsWithSecondIndependent": sum(integer(row[f"post_{stage}HasSecond"]) > 0 for row in rows),
                "totalCellFrames": sum(integer(row[f"post_{stage}Frames"]) for row in rows),
            }
        )

    summary = {
        "schema": "scancover.witness_conversion_audit.v1",
        "authority": "read_only_offline_diagnostic",
        "heldCells": len(rows),
        "stronglyAttributedCells": strong,
        "classifications": dict(classifications),
        "stages": stage_rows,
        "raceEvents": race_summary,
        "fusionStream": fusion_summary,
        "thresholds": {
            "cellMetres": CELL_METRES,
            "motionQualityMin": SAFE_MOTION_QUALITY,
            "viewRadiusMaxExclusive": AUTHORITY_OUTER_RADIUS,
            "independentBaselineMetres": INDEPENDENT_BASELINE_METRES,
            "independentAngleDeg": INDEPENDENT_ANGLE_DEG,
            "independentFrameGap": INDEPENDENT_FRAME_GAP,
        },
        "semanticBoundary": {
            "shadow": "pre-candidate-transaction correspondence consumed by VirtualProbeShadowAdjudicator",
            "post": "final saved correspondence after candidate transaction/rebuild on accepted fusion frames",
            "proof": "post eligible second + shadow views < 2 proves a conversion/order discrepancy, not that the pre snapshot itself contained the same target",
            "coverage": "fusion buffers exist only for accepted fusion frames; missing/rejected decision-only frames cannot create a post-stream opportunity here",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "witness_conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stage_fields = ["stage", "cellsWithAnyFrame", "cellsWithSecondIndependent", "totalCellFrames"]
    write_csv(output / "conversion_stage_summary.csv", stage_rows, stage_fields)
    if rows:
        write_csv(output / "held_cell_witness_conversion.csv", rows, list(rows[0].keys()))

    top = classifications.most_common()
    lines = [
        "# 证词转换审计",
        "",
        "本审计只读，不修改 Unity、枪胶、TSDF、纸皮或网格。",
        "",
        "## 结论总览",
        "",
        f"- 最终 Hold 单元：{len(rows)}",
        f"- 可由现有记录明确归因：{strong}",
        f"- 实机证词事件行：{race_summary.get('totalRows', 0)}（落入 Hold 单元 {race_summary.get('heldRows', 0)}）",
        f"- 最终融合缓冲：处理 {fusion_summary.get('processedRows', 0)}/{fusion_summary.get('selectedRows', 0)} 帧，错误 {len(fusion_summary.get('errors', []))}",
        "",
        "### 责任分类",
        "",
    ]
    for name, count in top:
        lines.append(f"- `{name}`：{count}")
    lines.extend(("", "### 各门形成第二独立证词的单元数", ""))
    for item in stage_rows:
        lines.append(
            f"- `{item['stage']}`：有记录 {item['cellsWithAnyFrame']}；形成第二独立证词 {item['cellsWithSecondIndependent']}"
        )
    lines.extend(
        (
            "",
            "## 如何读",
            "",
            "- `pre_post_transaction_conversion_gap` 是最直接的交接证据：同一 Hold 单元在候选事务后的最终缓冲里已经有第二独立合格证词，但事务前探针账仍只有 0/1 个。",
            "- 其他 `*_bottleneck` 表示第二独立机会在哪一道门首次消失；这是定位责任，不是建议放宽阈值。",
            "- `no_second_independent_stable_match` 不能仅凭本表定罪某个门，因为最终流没有给该单元留下第二个可归属的稳定匹配。",
            "",
            "## 证据边界",
            "",
            "最终缓冲与探针账位于同帧的两个不同时刻。最终缓冲能证明候选事务后证据存在，但不能倒推事务前快照也拥有同一候选目标；因此它适合审计转换/顺序，不应被误写成原始深度真值。",
            "",
        )
    )
    (output / "witness_conversion_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="local ScanCover replay session")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--skip-race-events", action="store_true")
    args = parser.parse_args()

    session = args.session.resolve()
    output = (args.output or (session / "offline_witness_conversion")).resolve()
    probe = session / "probe_shadow"
    held = load_held_cells(probe / "verdict_cells.csv")
    probe_frames = load_probe_frames(probe / "frames.csv")
    print(f"held={len(held)} probeFrames={len(probe_frames)}", flush=True)
    if args.skip_race_events:
        race_summary = {"totalRows": 0, "heldRows": 0, "skipped": True}
    else:
        print("streaming graduation witness events...", flush=True)
        race_summary = audit_race_events(probe / "graduation_race_witness_events.csv", held)
        print(f"race rows={race_summary['totalRows']} heldRows={race_summary['heldRows']}", flush=True)
    print("auditing final fusion correspondence stream...", flush=True)
    fusion_summary = audit_fusion_stream(session, held, probe_frames, args.frame_stride)
    rows = build_rows(held)
    summary = write_report(output, rows, race_summary, fusion_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if not fusion_summary["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
