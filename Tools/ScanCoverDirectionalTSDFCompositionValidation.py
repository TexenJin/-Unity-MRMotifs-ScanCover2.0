#!/usr/bin/env python3
"""Offline validation for conservative directional-TSDF mesh composition.

This tool deliberately compares three stages on the same Replica/Quest-style
observations:

1. dominant-axis directional fusion + independent per-direction extraction;
2. the same dominant-axis volume + conservative cross-direction composition;
3. soft (up to three directions) fusion + conservative composition.

The composed variants prefer missing a narrow boundary over publishing two
coincident or incompatible sheets.  They are an offline gate for the Unity
foreground and do not write Unity assets or settings.

The observation integration kernel is selectable.  ``surface-splat`` keeps the
historical point/normal neighbourhood writer as a control.  ``projective``
allocates sparse blocks around observed surfaces, projects every candidate voxel
centre back into the depth image, and integrates depth(u, v) - voxel_z.  Block
allocation therefore cannot manufacture a zero crossing by itself.
"""

from __future__ import annotations

import argparse
import base64
import copy
import gzip
import hashlib
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import open3d as o3d

from ScanCoverQuest3VirtualCloneExperiment import (
    build_scene,
    build_mesh_structure_reference,
    build_stratified_slice_cameras,
    distance_to_reference,
    load_legacy_mesh,
    make_rays,
    normalize,
)
from ScanCoverReplicaPersistentTSDFValidation import (
    camera_calibration,
    cloud_metrics,
    make_depth_pair,
)


DIRECTION_VECTORS = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)

CORNER_OFFSETS = np.asarray(
    [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ],
    dtype=np.int64,
)

TETRAHEDRA = (
    (0, 5, 1, 6), (0, 1, 2, 6), (0, 2, 3, 6),
    (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6),
)

CUBE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

CURRENT_TO_MC_EDGE = (8, 4, 9, 0, 11, 6, 10, 2, 3, 7, 5, 1)
MC33_SOURCE = Path(
    r"E:\PCAII\NEW-SCANCOVER\Assets\MRMotifs\ScanCover\Scripts"
    r"\MetaXR\08_Preprocess\ScanCoverMc33Topology.cs"
)
_MC33_TABLE: list[int] | None = None

PAPER_DMC_TABLE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "Assets"
    / "MRMotifs"
    / "ScanCover"
    / "Scripts"
    / "MetaXR"
    / "08_Preprocess"
    / "ScanCoverPaperDmcTables.cs"
)
PAPER_DIRECTION_TO_SCANCOVER = (2, 3, 0, 1, 5, 4)
PAPER_DIRECTION_THRESHOLD = 0.38268343
PAPER_CORNER_OFFSETS = np.asarray(
    [
        [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0],
    ],
    dtype=np.int64,
)
PAPER_EDGE_ENDPOINT_CORNERS = (
    (0, 1), (2, 1), (3, 2), (3, 0),
    (4, 5), (6, 5), (7, 6), (7, 4),
    (4, 0), (5, 1), (6, 2), (7, 3),
)
PAPER_DIRECTION_EDGES_TO_CHECK = (
    (0, 4, 1, 5, 2, 6, 3, 7),
    (4, 0, 5, 1, 6, 2, 7, 3),
    (1, 3, 5, 7, 9, 8, 10, 11),
    (3, 1, 7, 5, 8, 9, 11, 10),
    (2, 0, 6, 4, 10, 9, 11, 8),
    (0, 2, 4, 6, 8, 11, 9, 10),
)
_PAPER_DMC_TABLES: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def paper_dmc_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the authors' immutable DMC tables embedded in the Unity source."""

    global _PAPER_DMC_TABLES
    if _PAPER_DMC_TABLES is not None:
        return _PAPER_DMC_TABLES
    source = PAPER_DMC_TABLE_SOURCE.read_text(encoding="utf-8")

    def payload(name: str) -> bytes:
        match = re.search(
            rf'private const string {name}\s*=\s*"(?P<body>[^"]+)";',
            source,
        )
        if match is None:
            raise RuntimeError(
                f"Could not parse {name} from {PAPER_DMC_TABLE_SOURCE}"
            )
        return gzip.decompress(base64.b64decode(match.group("body")))

    triangle_edges = np.frombuffer(
        payload("TriangleEdgesGzipBase64"), dtype=np.int8
    ).reshape(256, 16)
    index_decomposition = np.frombuffer(
        payload("IndexDecompositionGzipBase64"), dtype="<i2"
    ).reshape(256, 4)
    direction_compatibility = np.frombuffer(
        payload("DirectionCompatibilityGzipBase64"), dtype=np.uint8
    ).reshape(256, 6)
    _PAPER_DMC_TABLES = (
        triangle_edges,
        index_decomposition,
        direction_compatibility,
    )
    return _PAPER_DMC_TABLES


def mc33_table() -> list[int]:
    global _MC33_TABLE
    if _MC33_TABLE is None:
        source = MC33_SOURCE.read_text(encoding="utf-8")
        match = re.search(
            r"private static readonly ushort\[\] Table\s*=\s*\{(?P<body>.*?)\n\s*\};",
            source,
            flags=re.DOTALL,
        )
        if match is None:
            raise RuntimeError(f"Could not parse MC33 table from {MC33_SOURCE}")
        _MC33_TABLE = [
            int(token, 16)
            for token in re.findall(r"0X[0-9A-F]+", match.group("body"))
        ]
        if len(_MC33_TABLE) < 2300:
            raise RuntimeError(f"MC33 table is unexpectedly short: {len(_MC33_TABLE)}")
    return _MC33_TABLE


def mc33_face_tests(values: np.ndarray, mask: int) -> list[int]:
    face = [0] * 6
    if mask & 0x80:
        face[0] = (-1 if values[0] * values[5] < values[1] * values[4] else 1) if (mask & 0xCC) == 0x84 else 0
        face[3] = (-1 if values[0] * values[7] < values[3] * values[4] else 1) if (mask & 0x99) == 0x81 else 0
        face[4] = (-1 if values[0] * values[2] < values[1] * values[3] else 1) if (mask & 0xF0) == 0xA0 else 0
    else:
        face[0] = (1 if values[0] * values[5] < values[1] * values[4] else -1) if (mask & 0xCC) == 0x48 else 0
        face[3] = (1 if values[0] * values[7] < values[3] * values[4] else -1) if (mask & 0x99) == 0x18 else 0
        face[4] = (1 if values[0] * values[2] < values[1] * values[3] else -1) if (mask & 0xF0) == 0x50 else 0
    if mask & 0x02:
        face[1] = (-1 if values[1] * values[6] < values[2] * values[5] else 1) if (mask & 0x66) == 0x42 else 0
        face[2] = (-1 if values[3] * values[6] < values[2] * values[7] else 1) if (mask & 0x33) == 0x12 else 0
        face[5] = (-1 if values[4] * values[6] < values[5] * values[7] else 1) if (mask & 0x0F) == 0x0A else 0
    else:
        face[1] = (1 if values[1] * values[6] < values[2] * values[5] else -1) if (mask & 0x66) == 0x24 else 0
        face[2] = (1 if values[3] * values[6] < values[2] * values[7] else -1) if (mask & 0x33) == 0x21 else 0
        face[5] = (1 if values[4] * values[6] < values[5] * values[7] else -1) if (mask & 0x0F) == 0x05 else 0
    return face


def mc33_face_test_one(values: np.ndarray, face: int) -> int:
    if face == 0:
        return 0x48 if values[0] * values[5] < values[1] * values[4] else 0x84
    if face == 1:
        return 0x24 if values[1] * values[6] < values[2] * values[5] else 0x42
    if face == 2:
        return 0x21 if values[3] * values[6] < values[2] * values[7] else 0x12
    if face == 3:
        return 0x18 if values[0] * values[7] < values[3] * values[4] else 0x81
    if face == 4:
        return 0x50 if values[0] * values[2] < values[1] * values[3] else 0xA0
    return 0x05 if values[4] * values[6] < values[5] * values[7] else 0x0A


def mc33_interior_test(values: np.ndarray, diagonal: int, flag13: int) -> int:
    at = values[4] - values[0]
    bt = values[5] - values[1]
    ct = values[6] - values[2]
    dt = values[7] - values[3]
    t = at * ct - bt * dt
    if t < 0.0:
        if diagonal & 1:
            return 0
    elif not (diagonal & 1) or t == 0.0:
        return 0
    t = 0.5 * (
        values[3] * bt + values[1] * dt - values[2] * at - values[0] * ct
    ) / t
    if not 0.0 < t < 1.0:
        return 0
    at = values[0] + at * t
    bt = values[1] + bt * t
    ct = (values[2] + ct * t) * at
    dt = (values[3] + dt * t) * bt
    if diagonal & 1:
        if ct < dt and dt >= 0.0:
            return int((bt < 0.0) == (values[diagonal] < 0.0)) + flag13
    elif ct > dt and ct >= 0.0:
        return int((at < 0.0) == (values[diagonal] < 0.0)) + flag13
    return 0


def mc33_select_pattern(values: np.ndarray, sign_mask: int) -> int:
    table = mc33_table()
    if sign_mask & 0x80:
        case = table[sign_mask ^ 0xFF]
        invert = (case & 0x800) == 0
    else:
        case = table[sign_mask]
        invert = (case & 0x800) != 0
    key = case & 0x7FF
    oriented = sign_mask if invert else sign_mask ^ 0xFF
    category = case >> 12
    if category == 0:
        return key
    if category == 1:
        return 183 + (key << 1) if oriented & mc33_face_test_one(values, key >> 2) else 159 + key
    if category == 2:
        return 239 + 6 * key if mc33_interior_test(values, key, 0) else 231 + (key << 1)
    if category == 3:
        if oriented & mc33_face_test_one(values, key % 6):
            return 575 + 5 * key
        return 407 + 7 * key if mc33_interior_test(values, key // 6, 0) else 335 + 3 * key
    face = mc33_face_tests(values, oriented if category != 7 else 165)
    total = sum(face)
    if category == 4:
        if total == -3:
            return 695 + 3 * key
        if total == -1:
            return (759 if face[0] + face[2] < 0 else 799) + 5 * key if face[4] + face[5] < 0 else 719 + 5 * key
        if total == 1:
            return 983 + 9 * key if face[4] + face[5] < 0 else (839 if face[0] + face[2] < 0 else 911) + 9 * key
        return 1095 + 9 * key if mc33_interior_test(values, key >> 1, 0) else 1055 + 5 * key
    if category == 5:
        if total == -2:
            connected = (
                bool(mc33_interior_test(values, 0, 0))
                if key & 2
                else bool(mc33_interior_test(values, 0, 0))
                or bool(mc33_interior_test(values, 1 if key else 3, 0))
            )
            return (1213 + (key << 3)) if connected else (1189 + (key << 2))
        if total == 0:
            return (1261 if face[2 + key] < 0 else 1285) + (key << 3)
        connected = (
            bool(mc33_interior_test(values, 1, 0))
            if key & 2
            else bool(mc33_interior_test(values, 2, 0))
            or bool(mc33_interior_test(values, 3 if key else 1, 0))
        )
        return (1237 + (key << 3)) if connected else (1201 + (key << 2))
    if category == 6:
        if total == -2:
            diagonal = (0xDA010C >> (key << 1)) & 3
            return 1453 + (key << 3) if mc33_interior_test(values, diagonal, 0) else 1357 + (key << 2)
        if total == 0:
            return (1645 if face[key >> 1] < 0 else 1741) + (key << 3)
        diagonal = (0xA7B7E5 >> (key << 1)) & 3
        return 1549 + (key << 3) if mc33_interior_test(values, diagonal, 0) else 1405 + (key << 2)
    total = abs(total)
    if total == 0:
        key = ((1 if face[1] < 0 else 0) << 1) | (1 if face[5] < 0 else 0)
        if face[0] * face[1] == face[5]:
            return 2157 + 12 * key
        interior = mc33_interior_test(values, key, 1)
        return 2285 + (10 * key - 40 * interior if interior else 6 * key)
    if total == 2:
        first = (
            (1 if face[2] > 0 else 0)
            if face[0] < 0
            else 12 + (1 if face[2] < 0 else 0)
        )
        second = (
            (1 if face[3] < 0 else 0)
            if face[1] < 0
            else 6 + (1 if face[3] > 0 else 0)
        )
        offset = 1917 + 10 * (first + second)
        return offset + (30 if face[4] > 0 else 0)
    if total == 4:
        key = 21 + 11 * face[0] + 4 * face[1] + 3 * face[2] + 2 * face[3] + face[4]
        if key >> 4:
            key -= 20 if key & 32 else 10
        return 1845 + 3 * key
    return 1839 + 2 * face[0]


def mc33_components(current_values: np.ndarray, active_edges: list[bool]) -> list[int] | None:
    values = -np.asarray(
        [
            current_values[0], current_values[3], current_values[7], current_values[4],
            current_values[1], current_values[2], current_values[6], current_values[5],
        ],
        dtype=np.float64,
    )
    sign_mask = 0
    for corner, value in enumerate(values):
        if value < 0.0:
            sign_mask |= 1 << (7 - corner)
    if sign_mask in (0, 0xFF):
        return None
    table = mc33_table()
    offset = mc33_select_pattern(values, sign_mask)
    if offset < 0 or offset + 1 >= len(table):
        return None
    parent = [-1] * 13

    def activate(node: int) -> None:
        if parent[node] < 0:
            parent[node] = node

    def find(node: int) -> int:
        if node < 0 or node >= len(parent) or parent[node] < 0:
            return -1
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            next_node = parent[node]
            parent[node] = root
            node = next_node
        return root

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first >= 0 and root_second >= 0 and root_first != root_second:
            parent[root_second] = root_first

    cursor = offset
    guard = 0
    while cursor + 1 < len(table) and guard < 32:
        cursor += 1
        guard += 1
        triangle = table[cursor]
        nodes = (triangle & 0xF, (triangle >> 4) & 0xF, (triangle >> 8) & 0xF)
        if any(node > 12 for node in nodes):
            return None
        for node in nodes:
            activate(node)
        union(nodes[0], nodes[1])
        union(nodes[1], nodes[2])
        if (triangle & 0xF000) == 0:
            break
    if guard <= 0 or guard >= 32:
        return None
    result = [-1] * 12
    roots: dict[int, int] = {}
    for current_edge in range(12):
        if not active_edges[current_edge]:
            continue
        root = find(CURRENT_TO_MC_EDGE[current_edge])
        if root < 0:
            return None
        roots.setdefault(root, len(roots))
        result[current_edge] = roots[root]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--degradation-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=18)
    parser.add_argument(
        "--camera-path-checkpoints",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Build one deterministic progressive camera path whose prefixes end "
            "at these frame counts. This makes frame-count sweeps nested instead "
            "of independently resampling the synthetic camera candidates."
        ),
    )
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--voxel", type=float, default=0.04)
    parser.add_argument("--sdf-trunc", type=float, default=0.12)
    parser.add_argument("--min-distance", type=float, default=0.35)
    parser.add_argument("--max-distance", type=float, default=5.0)
    parser.add_argument("--minimum-weight", type=float, default=0.75)
    parser.add_argument(
        "--paper-minimum-weight",
        type=float,
        default=1e-6,
        help=(
            "Paper DMC sample-presence threshold. Confidence is still carried "
            "by the fused SDF weight during voting; this must not inherit the "
            "legacy extractor's mature-surface threshold."
        ),
    )
    parser.add_argument("--truth-samples", type=int, default=30000)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument(
        "--integration-mode",
        choices=[
            "surface-splat",
            "projective",
            "normal-raycast",
            "paper-normal-raycast",
        ],
        default="surface-splat",
        help=(
            "Historical point splat control, voxel-centre projective TSDF, or "
            "historical approximate / paper-faithful normal-directed point-to-plane "
            "ray casting."
        ),
    )
    parser.add_argument(
        "--projective-block-voxels",
        type=int,
        default=8,
        help="Sparse allocation block edge used only to enumerate projective candidates.",
    )
    parser.add_argument("--soft-direction-threshold", type=float, default=0.35)
    parser.add_argument(
        "--paper-normal-radius",
        type=int,
        default=2,
        help=(
            "Maximum image-neighbour radius for the validity-aware paper normal "
            "estimate. Frozen before the strict A/B run."
        ),
    )
    parser.add_argument(
        "--paper-normal-depth-change-factor",
        type=float,
        default=0.02,
        help="Relative depth-discontinuity guard for paper normal estimation.",
    )
    parser.add_argument(
        "--paper-normal-depth-change-floor",
        type=float,
        default=0.015,
        help="Minimum metric depth-discontinuity guard for paper normal estimation.",
    )
    parser.add_argument(
        "--paper-normal-bilateral-radius",
        type=int,
        default=1,
        help="Image radius of the bilateral normal-only filter; raw depth is unchanged.",
    )
    parser.add_argument(
        "--paper-normal-source",
        choices=["estimated", "raycast-truth"],
        default="estimated",
        help=(
            "Normal source for the paper-normal-raycast arm. 'estimated' uses the "
            "paper-style depth-neighbourhood estimator; 'raycast-truth' reads the "
            "first-hit triangle normal from the immutable evaluation mesh. The "
            "latter is an offline oracle diagnostic and never changes Unity."
        ),
    )
    parser.add_argument(
        "--paper-normal-angular-noise-sigma-degrees",
        type=float,
        default=0.0,
        help=(
            "Deterministic zero-mean angular noise applied after selecting the "
            "paper normal source. Used only for offline normal-tolerance sweeps."
        ),
    )
    parser.add_argument(
        "--paper-normal-dropout-probability",
        type=float,
        default=0.0,
        help="Independent probability of removing an otherwise usable paper normal.",
    )
    parser.add_argument(
        "--paper-normal-edge-angular-noise-sigma-degrees",
        type=float,
        default=0.0,
        help=(
            "Additional angular noise at depth discontinuities and valid/invalid "
            "depth boundaries. Independent of the global angular noise."
        ),
    )
    parser.add_argument(
        "--paper-normal-edge-dropout-probability",
        type=float,
        default=0.0,
        help="Additional normal-dropout probability at depth boundaries.",
    )
    parser.add_argument(
        "--paper-normal-perturbation-edge-depth-factor",
        type=float,
        default=0.04,
        help="Relative depth jump used to classify a pixel as a perturbation edge.",
    )
    parser.add_argument(
        "--paper-normal-perturbation-edge-depth-floor",
        type=float,
        default=0.03,
        help="Metric depth-jump floor used to classify a perturbation edge.",
    )
    parser.add_argument(
        "--paper-normal-perturbation-seed",
        type=int,
        default=9127,
        help="Independent deterministic seed for paper-normal tolerance sweeps.",
    )
    parser.add_argument("--valid-gradient-dot", type=float, default=0.15)
    parser.add_argument("--parallel-dot", type=float, default=0.82)
    parser.add_argument("--edge-merge-voxel-ratio", type=float, default=0.35)
    parser.add_argument("--feature-angle-degrees", type=float, default=32.0)
    parser.add_argument("--feature-neighborhood-voxel-ratio", type=float, default=1.75)
    parser.add_argument("--feature-max-move-voxel-ratio", type=float, default=0.75)
    parser.add_argument("--feature-min-family-support-ratio", type=float, default=0.12)
    parser.add_argument("--feature-rank-ratio", type=float, default=0.08)
    parser.add_argument(
        "--feature-certificate-min-frames-per-family",
        type=int,
        default=3,
        help=(
            "Strict Hermite feature certificate: every selected normal family "
            "must be supported by this many distinct fused frames."
        ),
    )
    parser.add_argument(
        "--feature-certificate-min-views-per-family",
        type=int,
        default=2,
        help=(
            "Strict Hermite feature certificate: every selected normal family "
            "must be observed from this many coarse spherical view bins."
        ),
    )
    parser.add_argument(
        "--feature-certificate-min-samples-per-family",
        type=int,
        default=1,
        help=(
            "Minimum independently persistent Hermite edge samples in each "
            "selected family.  Persistence is checked on each physical edge; "
            "disjoint frame/view evidence is never pooled to satisfy the gate."
        ),
    )
    parser.add_argument(
        "--feature-certificate-min-rank-ratio",
        type=float,
        default=0.12,
        help=(
            "Minimum retained singular-value ratio for a certified crease/corner "
            "QEF. This rejects nearly parallel, noise-created families."
        ),
    )
    parser.add_argument(
        "--feature-certificate-min-cell-margin-ratio",
        type=float,
        default=0.02,
        help=(
            "Minimum QEF feature-point distance from every cell face, expressed "
            "as a voxel ratio. Boundary-hugging points retain the source DMC."
        ),
    )
    parser.add_argument(
        "--feature-certificate-min-family-weight-ratio",
        type=float,
        default=0.30,
        help=(
            "Minimum weakest-to-strongest selected normal-family weight. "
            "A weak secondary family is not allowed to manufacture a crease."
        ),
    )
    parser.add_argument(
        "--feature-certificate-min-qef-displacement-ratio",
        type=float,
        default=0.08,
        help=(
            "Minimum QEF displacement from the family centroid in voxel units. "
            "Below this, the source DMC is already the safer representation."
        ),
    )
    parser.add_argument(
        "--structure-band-meters",
        type=float,
        default=0.08,
        help="Truth-space band used to report smooth/crease/depth-edge geometry separately.",
    )
    parser.add_argument("--seed", type=int, default=15319)
    parser.add_argument("--ideal-depth", action="store_true")
    parser.add_argument(
        "--paper-baseline-only",
        action="store_true",
        help="Evaluate only independent-direction and composed directional TSDF meshes; skip custom QEF/Hermite extensions.",
    )
    parser.add_argument(
        "--paper-hermite-qef-ab-only",
        action="store_true",
        help=(
            "Run the strict paper-DMC versus TSDF-Hermite/QEF feature-placement "
            "A/B. The shared physical-edge boundary ledger is frozen; only a "
            "proven single-patch cell may receive an interior feature vertex."
        ),
    )
    parser.add_argument(
        "--paper-growth-ledger-only",
        action="store_true",
        help=(
            "Integrate the nested camera path once and extract only the regularized "
            "paper DMC mesh at each camera-path checkpoint. This is a read-only "
            "coverage diagnostic and skips unrelated legacy/QEF variants."
        ),
    )
    parser.add_argument(
        "--paper-growth-stage-attribution",
        action="store_true",
        help=(
            "At the final growth checkpoint, partition every still-missing visible "
            "truth sample by its first failed paper-DMC stage: raw TSDF/corner "
            "availability, direction filtering/voting, or final DMC extraction."
        ),
    )
    parser.add_argument(
        "--paper-growth-upstream-attribution",
        action="store_true",
        help=(
            "At the final growth checkpoint, split raw TSDF/corner-availability "
            "loss into usable depth/normal sample availability, projective voxel "
            "touch, complete corner-weight support, and usable zero crossing. "
            "This is an offline diagnostic and never changes fusion or topology."
        ),
    )
    return parser.parse_args()


def voxel_key(point: np.ndarray, voxel: float) -> tuple[int, int, int]:
    return tuple(np.rint(point / voxel).astype(np.int64).tolist())


def voxel_center(key: tuple[int, int, int], voxel: float) -> np.ndarray:
    return np.asarray(key, dtype=np.float64) * voxel


def input_sampling_audit(
    width: int,
    height: int,
    voxel: float,
    vertical_fov_degrees: float,
    aspect: float,
) -> dict[str, Any]:
    """Report the spatial pitch represented by the synthetic depth raster.

    Gradient-directed ray casting consumes every valid depth pixel.  Reducing a
    320x320-class Quest depth image to a tiny validation raster therefore also
    removes fusion rays; it is not a free evaluation optimization.  This audit
    keeps that input-supply approximation visible without changing integration
    or DMC topology.
    """

    width = max(1, int(width))
    height = max(1, int(height))
    tan_y = math.tan(math.radians(float(vertical_fov_degrees)) * 0.5)
    tan_x = tan_y * float(aspect)
    vertical_slope = 2.0 * tan_y / height
    horizontal_slope = 2.0 * tan_x / width
    worst_slope = max(vertical_slope, horizontal_slope)
    reference_distances = (1.5, 3.0, 5.0)
    rows = []
    for distance in reference_distances:
        horizontal_pitch = horizontal_slope * distance
        vertical_pitch = vertical_slope * distance
        worst_pitch = max(horizontal_pitch, vertical_pitch)
        rows.append(
            {
                "distanceMeters": distance,
                "horizontalPixelPitchMeters": horizontal_pitch,
                "verticalPixelPitchMeters": vertical_pitch,
                "worstPixelPitchMeters": worst_pitch,
                "worstPixelPitchVoxelRatio": worst_pitch / max(voxel, 1e-12),
                "pixelPitchNoLargerThanVoxel": worst_pitch <= voxel,
            }
        )
    return {
        "semantics": (
            "centre-to-centre pinhole-ray pitch; diagnostic only; multiple views may "
            "cover gaps but cannot turn a severely downsampled raster into ideal input"
        ),
        "verticalFieldOfViewDegrees": float(vertical_fov_degrees),
        "aspect": float(aspect),
        "voxelMeters": float(voxel),
        "referenceDistances": rows,
        "maximumDistanceForSingleViewPixelPitchAtMostOneVoxelMeters": (
            float(voxel / worst_slope) if worst_slope > 0.0 else float("inf")
        ),
    }


def traverse_voxel_point_cells(
    start: np.ndarray,
    end: np.ndarray,
    voxel: float,
) -> list[tuple[int, int, int]]:
    """Amanatides-Woo traversal of Voronoi cells around TSDF grid points.

    TSDF samples live at integer grid points.  Shifting grid coordinates by
    half a voxel turns their nearest-grid-point regions into ordinary cells,
    allowing the standard traversal to enumerate every grid sample whose cell
    is crossed by the normal segment.
    """
    if voxel <= 0.0:
        raise ValueError("voxel size must be positive")
    start_grid = np.asarray(start, dtype=np.float64) / voxel + 0.5
    end_grid = np.asarray(end, dtype=np.float64) / voxel + 0.5
    direction = end_grid - start_grid
    current = np.floor(start_grid).astype(np.int64)
    target = np.floor(end_grid).astype(np.int64)
    keys: list[tuple[int, int, int]] = [tuple(current.tolist())]
    if np.array_equal(current, target):
        return keys

    step = np.sign(direction).astype(np.int64)
    t_max = np.full(3, np.inf, dtype=np.float64)
    t_delta = np.full(3, np.inf, dtype=np.float64)
    for axis in range(3):
        component = float(direction[axis])
        if abs(component) <= 1e-15:
            continue
        next_boundary = float(current[axis] + (1 if step[axis] > 0 else 0))
        t_max[axis] = (next_boundary - float(start_grid[axis])) / component
        t_delta[axis] = abs(1.0 / component)

    maximum_steps = int(np.sum(np.abs(target - current))) + 4
    for _ in range(maximum_steps):
        minimum_t = float(np.min(t_max))
        tied_axes = np.flatnonzero(np.abs(t_max - minimum_t) <= 1e-12)
        # A simultaneous boundary hit advances every tied axis.  Such exact
        # edge/corner hits have zero area and must not manufacture side cells.
        for axis in tied_axes:
            current[axis] += step[axis]
            t_max[axis] += t_delta[axis]
        key = tuple(current.tolist())
        if key != keys[-1]:
            keys.append(key)
        if np.array_equal(current, target):
            break
    else:
        raise RuntimeError("voxel traversal did not reach its target cell")
    return keys


def adjacent_cells(key: tuple[int, int, int]) -> Iterable[tuple[int, int, int]]:
    x, y, z = key
    for dz in (-1, 0):
        for dy in (-1, 0):
            for dx in (-1, 0):
                yield x + dx, y + dy, z + dz


@dataclass
class DirectionalGrid:
    voxel: float
    truncation: float
    soft_threshold: float
    soft_assignment: bool
    maximum_weight: float = 32.0
    values: list[dict[tuple[int, int, int], list[float]]] = field(
        default_factory=lambda: [dict() for _ in range(6)]
    )
    candidates: set[tuple[int, int, int]] = field(default_factory=set)
    samples: int = 0
    voxel_updates: int = 0
    direction_writes: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.int64))
    projective_candidate_blocks: int = 0
    projective_candidate_voxels: int = 0
    projective_visible_voxels: int = 0
    projective_valid_depth_voxels: int = 0
    projective_truncation_rejects: int = 0
    paper_normal_rays: int = 0
    paper_traversed_voxels: int = 0
    paper_integrated_voxels: int = 0
    paper_depth_weight_sum: float = 0.0
    paper_angle_weight_sum: float = 0.0
    paper_combined_weight_sum: float = 0.0
    frame_masks: list[dict[tuple[int, int, int], int]] = field(
        default_factory=lambda: [dict() for _ in range(6)]
    )
    view_masks: list[dict[tuple[int, int, int], int]] = field(
        default_factory=lambda: [dict() for _ in range(6)]
    )
    current_frame_index: int = 0
    current_view_bin: int = -1

    def begin_frame(self, frame_index: int) -> None:
        self.current_frame_index = max(0, int(frame_index))
        self.current_view_bin = -1

    @staticmethod
    def spherical_view_bin(view_direction: np.ndarray) -> int:
        """Quantize an observation ray into 12 yaw x 3 elevation bins."""
        direction = normalize(np.asarray(view_direction, dtype=np.float64))
        if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) <= 1e-8:
            return -1
        yaw = math.atan2(float(direction[1]), float(direction[0]))
        yaw_bin = int(math.floor(((yaw + math.pi) / (2.0 * math.pi)) * 12.0)) % 12
        elevation = math.asin(float(np.clip(direction[2], -1.0, 1.0)))
        if elevation < math.radians(-25.0):
            elevation_bin = 0
        elif elevation > math.radians(25.0):
            elevation_bin = 2
        else:
            elevation_bin = 1
        return elevation_bin * 12 + yaw_bin

    def read_evidence(
        self, direction: int, key: tuple[int, int, int]
    ) -> tuple[int, int]:
        return (
            int(self.frame_masks[direction].get(key, 0)),
            int(self.view_masks[direction].get(key, 0)),
        )

    def directions_for(self, normal: np.ndarray) -> list[tuple[int, float]]:
        dots = DIRECTION_VECTORS @ normal
        if not self.soft_assignment:
            direction = int(np.argmax(dots))
            return [(direction, max(0.25, float(dots[direction])))]
        chosen: list[tuple[int, float]] = []
        denominator = max(1e-6, 1.0 - self.soft_threshold)
        for direction, dot in enumerate(dots):
            if dot <= self.soft_threshold:
                continue
            membership = min(1.0, max(0.0, (float(dot) - self.soft_threshold) / denominator))
            if membership > 1e-5:
                chosen.append((direction, membership))
        if not chosen:
            direction = int(np.argmax(dots))
            chosen.append((direction, max(0.25, float(dots[direction]))))
        return chosen

    def canonical_directions_for(self, normal: np.ndarray) -> list[tuple[int, float]]:
        """Original DTSDF direction assignment used by the Unity canonical path."""
        dots = DIRECTION_VECTORS @ normal
        chosen = [
            (direction, float(dot))
            for direction, dot in enumerate(dots)
            if float(dot) > self.soft_threshold
        ]
        if not chosen:
            direction = int(np.argmax(dots))
            chosen.append((direction, max(0.25, float(dots[direction]))))
        return chosen

    @staticmethod
    def paper_directions_for(normal: np.ndarray) -> list[tuple[int, float]]:
        """Exact sector membership from Splietker and Behnke, Eq. (3)."""
        dots = DIRECTION_VECTORS @ normal
        return [
            (direction, float(dot))
            for direction, dot in enumerate(dots)
            if float(dot) > PAPER_DIRECTION_THRESHOLD
        ]

    def integrate(self, camera: np.ndarray, point: np.ndarray, normal: np.ndarray) -> None:
        normal = normalize(np.asarray(normal, dtype=np.float64))
        if not np.all(np.isfinite(normal)) or np.linalg.norm(normal) < 1e-6:
            return
        toward_camera = camera - point
        self.current_view_bin = self.spherical_view_bin(toward_camera)
        if float(np.dot(normal, toward_camera)) < 0.0:
            normal = -normal
        writes = self.directions_for(normal)
        self.samples += 1

        extent = np.abs(normal) * self.truncation + self.voxel * 0.75
        lower = np.floor((point - extent) / self.voxel).astype(np.int64)
        upper = np.ceil((point + extent) / self.voxel).astype(np.int64)
        lateral_limit_sq = (self.voxel * 0.90) ** 2
        for z in range(int(lower[2]), int(upper[2]) + 1):
            for y in range(int(lower[1]), int(upper[1]) + 1):
                for x in range(int(lower[0]), int(upper[0]) + 1):
                    key = (x, y, z)
                    delta = voxel_center(key, self.voxel) - point
                    signed_distance = float(np.dot(delta, normal))
                    if abs(signed_distance) > self.truncation + self.voxel * 0.25:
                        continue
                    lateral = delta - normal * signed_distance
                    if float(np.dot(lateral, lateral)) > lateral_limit_sq:
                        continue
                    tsdf = min(1.0, max(-1.0, signed_distance / self.truncation))
                    for direction, direction_weight in writes:
                        self._integrate_voxel(direction, key, tsdf, direction_weight)

    def integrate_projective_frame(
        self,
        camera: dict[str, Any],
        depth: np.ndarray,
        points: np.ndarray,
        normals: np.ndarray,
        valid: np.ndarray,
        width: int,
        height: int,
        sample_stride: int,
        block_voxels: int,
    ) -> int:
        """Integrate one frame with a voxel-centre projective TSDF update.

        Surface samples only allocate sparse candidate blocks.  A voxel is
        actually updated only after its centre projects to a valid depth pixel;
        the signed distance is the observed axial depth minus the voxel axial
        depth.  This intentionally keeps allocation and fusion semantics
        separate.
        """
        block_voxels = max(1, int(block_voxels))
        stride = max(1, int(sample_stride))
        half_extent = self.truncation + self.voxel
        allocated: set[tuple[int, int, int]] = set()
        accepted_samples = 0

        for index in range(0, len(points), stride):
            if not valid[index] or not np.all(np.isfinite(normals[index])):
                continue
            point = points[index]
            lower = np.floor((point - half_extent) / self.voxel).astype(np.int64)
            upper = np.ceil((point + half_extent) / self.voxel).astype(np.int64)
            block_lower = np.floor_divide(lower, block_voxels)
            block_upper = np.floor_divide(upper, block_voxels)
            for bz in range(int(block_lower[2]), int(block_upper[2]) + 1):
                for by in range(int(block_lower[1]), int(block_upper[1]) + 1):
                    for bx in range(int(block_lower[0]), int(block_upper[0]) + 1):
                        allocated.add((bx, by, bz))
            accepted_samples += 1

        intrinsic, _, forward = camera_calibration(camera, width, height)
        pose = camera["pose"]
        camera_position = np.asarray(pose["position"], dtype=np.float64)
        right = normalize(np.asarray(pose["right"], dtype=np.float64))
        down = -normalize(np.asarray(pose["up"], dtype=np.float64))
        fx = float(intrinsic.intrinsic_matrix[0, 0])
        fy = float(intrinsic.intrinsic_matrix[1, 1])
        cx = float(intrinsic.intrinsic_matrix[0, 2])
        cy = float(intrinsic.intrinsic_matrix[1, 2])
        depth_image = depth.reshape(height, width).astype(np.float64)
        normal_image = normals.reshape(height, width, 3)

        self.samples += accepted_samples
        self.projective_candidate_blocks += len(allocated)
        for bx, by, bz in allocated:
            start_x = bx * block_voxels
            start_y = by * block_voxels
            start_z = bz * block_voxels
            for local_z in range(block_voxels):
                for local_y in range(block_voxels):
                    for local_x in range(block_voxels):
                        key = (
                            start_x + local_x,
                            start_y + local_y,
                            start_z + local_z,
                        )
                        self.projective_candidate_voxels += 1
                        center = voxel_center(key, self.voxel)
                        camera_delta = center - camera_position
                        self.current_view_bin = self.spherical_view_bin(-camera_delta)
                        voxel_z = float(np.dot(camera_delta, forward))
                        if voxel_z <= 1e-6:
                            continue
                        u = fx * float(np.dot(camera_delta, right)) / voxel_z + cx
                        v = fy * float(np.dot(camera_delta, down)) / voxel_z + cy
                        pixel_x = int(math.floor(u + 0.5))
                        pixel_y = int(math.floor(v + 0.5))
                        if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
                            continue
                        self.projective_visible_voxels += 1
                        observed_depth = float(depth_image[pixel_y, pixel_x])
                        normal = normal_image[pixel_y, pixel_x]
                        if observed_depth <= 0.0 or not np.all(np.isfinite(normal)):
                            continue
                        self.projective_valid_depth_voxels += 1
                        signed_distance = observed_depth - voxel_z
                        if signed_distance < -self.truncation:
                            self.projective_truncation_rejects += 1
                            continue
                        tsdf = min(1.0, signed_distance / self.truncation)
                        normal = normalize(np.asarray(normal, dtype=np.float64))
                        if float(np.dot(normal, camera_position - center)) < 0.0:
                            normal = -normal
                        for direction, direction_weight in self.canonical_directions_for(normal):
                            self._integrate_voxel(direction, key, tsdf, direction_weight)
        return accepted_samples

    def integrate_normal_raycast(
        self,
        camera: np.ndarray,
        point: np.ndarray,
        normal: np.ndarray,
    ) -> None:
        """Match the canonical Unity/DTSDF normal-directed P2PL traversal."""
        normal = normalize(np.asarray(normal, dtype=np.float64))
        if not np.all(np.isfinite(normal)) or np.linalg.norm(normal) < 1e-6:
            return
        if float(np.dot(normal, camera - point)) < 0.0:
            normal = -normal
        self.current_view_bin = self.spherical_view_bin(camera - point)
        writes = self.canonical_directions_for(normal)
        self.samples += 1

        step_length = max(self.voxel * 0.45, 1e-4)
        step_count = max(1, int(math.ceil((self.truncation * 2.0) / step_length)))
        previous_key: tuple[int, int, int] | None = None
        for step in range(step_count + 1):
            t = -self.truncation + (self.truncation * 2.0) * (step / step_count)
            key = voxel_key(point + normal * t, self.voxel)
            if key == previous_key:
                continue
            previous_key = key
            center = voxel_center(key, self.voxel)
            signed_distance = float(np.dot(center - point, normal))
            if abs(signed_distance) > self.truncation + self.voxel * 0.25:
                continue
            tsdf = min(1.0, max(-1.0, signed_distance / self.truncation))
            for direction, direction_weight in writes:
                self._integrate_voxel(direction, key, tsdf, direction_weight)

    def integrate_paper_normal_raycast(
        self,
        camera: np.ndarray,
        point: np.ndarray,
        normal: np.ndarray,
        minimum_depth: float,
    ) -> bool:
        """Gradient-directed two-sided voxel traversal with point-to-plane SDF.

        This is the paper-side integration arm of the strict A/B.  The stored
        TSDF sign is mapped to ScanCover's existing convention (positive toward
        the observing camera); traversal, distance metric, sector threshold and
        combined observation weights otherwise follow the cited DTSDF route.
        """
        normal = normalize(np.asarray(normal, dtype=np.float64))
        if not np.all(np.isfinite(normal)) or np.linalg.norm(normal) < 1e-6:
            return False
        to_camera = np.asarray(camera, dtype=np.float64) - point
        depth = float(np.linalg.norm(to_camera))
        if not math.isfinite(depth) or depth <= 1e-6:
            return False
        view_direction = to_camera / depth
        self.current_view_bin = self.spherical_view_bin(view_direction)
        if float(np.dot(normal, view_direction)) < 0.0:
            normal = -normal
        angle_weight = max(0.0, float(np.dot(normal, view_direction)))
        if angle_weight <= 1e-8:
            return False

        minimum_depth = max(1e-3, float(minimum_depth))

        def noise_sigma(distance: float) -> float:
            # Nguyen et al. Kinect noise model used by the weighting reference.
            return 0.0012 + 0.0019 * (distance - 0.4) ** 2

        depth_weight = (
            noise_sigma(minimum_depth)
            / max(1e-12, noise_sigma(depth))
            * (minimum_depth * minimum_depth)
            / max(1e-12, depth * depth)
        )
        depth_weight = min(1.0, max(0.0, depth_weight))
        writes = self.paper_directions_for(normal)
        if not writes or depth_weight <= 1e-12:
            return False

        self.samples += 1
        self.paper_normal_rays += 1
        self.paper_depth_weight_sum += depth_weight
        self.paper_angle_weight_sum += angle_weight
        start = point - normal * self.truncation
        end = point + normal * self.truncation
        traversed = traverse_voxel_point_cells(start, end, self.voxel)
        self.paper_traversed_voxels += len(traversed)
        for key in traversed:
            center = voxel_center(key, self.voxel)
            signed_distance = float(np.dot(center - point, normal))
            if abs(signed_distance) > self.truncation + 1e-9:
                continue
            tsdf = min(1.0, max(-1.0, signed_distance / self.truncation))
            integrated = False
            for direction, direction_weight in writes:
                combined_weight = depth_weight * angle_weight * direction_weight
                if combined_weight <= 1e-12:
                    continue
                self._integrate_voxel(direction, key, tsdf, combined_weight)
                self.paper_combined_weight_sum += combined_weight
                integrated = True
            if integrated:
                self.paper_integrated_voxels += 1
        return True

    def _integrate_voxel(
        self, direction: int, key: tuple[int, int, int], tsdf: float, weight: float
    ) -> None:
        if weight > 1e-12 and self.current_frame_index > 0:
            frame_bit = 1 << (self.current_frame_index - 1)
            self.frame_masks[direction][key] = (
                int(self.frame_masks[direction].get(key, 0)) | frame_bit
            )
            if self.current_view_bin >= 0:
                view_bit = 1 << self.current_view_bin
                self.view_masks[direction][key] = (
                    int(self.view_masks[direction].get(key, 0)) | view_bit
                )
        record = self.values[direction].get(key)
        if record is None:
            record = [0.0, 0.0]
            self.values[direction][key] = record
        accepted = min(weight, max(0.0, self.maximum_weight - record[1]))
        if accepted <= 1e-8:
            return
        record[0] += tsdf * accepted
        record[1] += accepted
        self.voxel_updates += 1
        self.direction_writes[direction] += 1
        for cell in adjacent_cells(key):
            self.candidates.add(cell)

    def read(self, direction: int, key: tuple[int, int, int]) -> tuple[float, float]:
        record = self.values[direction].get(key)
        if record is None or record[1] <= 0.0:
            return 1.0, 0.0
        return record[0] / record[1], record[1]


@dataclass
class CellHypothesis:
    direction: int
    cell: tuple[int, int, int]
    values: np.ndarray
    weights: np.ndarray
    positions: np.ndarray
    normal: np.ndarray
    centroid: np.ndarray
    support: float


@dataclass
class ExtractionAudit:
    scanned_cells: int = 0
    zero_crossing_hypotheses: int = 0
    multi_direction_cells: int = 0
    invalid_direction_hypotheses: int = 0
    parallel_hypotheses_collapsed: int = 0
    incompatible_weak_hypotheses_dropped: int = 0
    edge_crossings_merged: int = 0
    edge_crossing_overflow_dropped: int = 0
    duplicate_triangles_dropped: int = 0
    degenerate_triangles_dropped: int = 0
    conservative_conflict_cells: int = 0
    nonmanifold_triangles_dropped: int = 0
    paper_incomplete_direction_cells: int = 0
    paper_incomplete_zero_weight_corners: int = 0
    paper_incomplete_low_weight_corners: int = 0
    paper_directions_missing_unwritten_corner: int = 0
    paper_directions_blocked_only_by_weight: int = 0
    paper_cells_with_complete_direction: int = 0
    paper_raw_crossing_cells: int = 0
    paper_filtered_crossing_cells: int = 0
    paper_voted_crossing_cells: int = 0
    paper_raw_hypotheses: int = 0
    paper_filtered_hypotheses: int = 0
    paper_voted_hypotheses: int = 0
    paper_intra_direction_rejected: int = 0
    paper_inter_direction_rejected: int = 0
    paper_empty_after_voting_cells: int = 0
    paper_components: int = 0
    paper_single_surface_cells: int = 0
    paper_double_surface_cells: int = 0
    paper_overflow_deferred_components: int = 0
    paper_raw_transition_edges: int = 0
    paper_filtered_transition_edges: int = 0
    paper_voted_transition_edges: int = 0
    paper_combined_transition_edges: int = 0
    paper_regularized_transition_edges: int = 0
    paper_edge_offset_slots: int = 0
    paper_dual_offset_edges: int = 0
    paper_required_edge_slots: int = 0
    paper_local_edge_slots_used: int = 0
    paper_shared_edge_slots_recovered: int = 0
    paper_unresolved_edge_slots: int = 0
    paper_regularized_corners: int = 0
    paper_regularized_cells: int = 0
    paper_regularization_reverted_cells: int = 0
    paper_regularization_reverted_nonmanifold_cells: int = 0
    paper_neighbor_face_comparisons: int = 0
    paper_neighbor_disagreements_before: int = 0
    paper_neighbor_disagreements_after: int = 0
    paper_unmeasured_edge_deferred_triangles: int = 0


@dataclass
class MeshBuild:
    vertices: list[np.ndarray] = field(default_factory=list)
    triangles: list[tuple[int, int, int]] = field(default_factory=list)
    triangle_directions: list[int] = field(default_factory=list)
    triangle_support: list[float] = field(default_factory=list)
    audit: ExtractionAudit = field(default_factory=ExtractionAudit)
    elapsed_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def cell_hypothesis(
    grid: DirectionalGrid,
    direction: int,
    cell: tuple[int, int, int],
    minimum_weight: float,
) -> CellHypothesis | None:
    base = np.asarray(cell, dtype=np.int64)
    keys = [tuple((base + offset).tolist()) for offset in CORNER_OFFSETS]
    samples = [grid.read(direction, key) for key in keys]
    values = np.asarray([sample[0] for sample in samples], dtype=np.float64)
    weights = np.asarray([sample[1] for sample in samples], dtype=np.float64)
    valid = weights >= minimum_weight
    if np.sum(valid) < 4:
        return None
    signed = values[valid]
    if not (np.any(signed < 0.0) and np.any(signed >= 0.0)):
        return None
    positions = np.asarray([voxel_center(key, grid.voxel) for key in keys], dtype=np.float64)
    # Fit only supported corners.  Treating missing corners as +1 (which is
    # useful for sign tests) corrupts the gradient near scan boundaries and
    # makes a conservative direction filter delete otherwise valid surfaces.
    fit_positions = positions[valid]
    fit_values = values[valid]
    design = np.column_stack((fit_positions, np.ones(len(fit_positions), dtype=np.float64)))
    coefficients, *_ = np.linalg.lstsq(design, fit_values, rcond=None)
    gradient = coefficients[:3]
    if np.linalg.norm(gradient) <= 1e-8:
        gradient = DIRECTION_VECTORS[direction].copy()
    normal = normalize(gradient)
    support = float(np.sum(np.minimum(weights[valid], minimum_weight * 4.0)))
    near = np.abs(values) <= 0.5
    centroid = np.mean(positions[near & valid], axis=0) if np.any(near & valid) else np.mean(positions[valid], axis=0)
    return CellHypothesis(direction, cell, values, weights, positions, normal, centroid, support)


def tetra_triangles(values: np.ndarray, tetra: tuple[int, int, int, int]) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    inside = [corner for corner in tetra if values[corner] < 0.0]
    outside = [corner for corner in tetra if values[corner] >= 0.0]
    if len(inside) in (0, 4):
        return []
    if len(inside) in (1, 3):
        invert = len(inside) == 3
        pivot = outside[0] if invert else inside[0]
        others = inside if invert else outside
        return [((pivot, others[0]), (pivot, others[1]), (pivot, others[2]))]
    return [
        ((inside[0], outside[0]), (inside[0], outside[1]), (inside[1], outside[1])),
        ((inside[0], outside[0]), (inside[1], outside[1]), (inside[1], outside[0])),
    ]


class EdgeVertexCache:
    def __init__(self, vertices: list[np.ndarray], merge_ratio: float):
        self.vertices = vertices
        self.merge_ratio = merge_ratio
        self.entries: dict[tuple[tuple[int, int, int], tuple[int, int, int]], list[tuple[float, int]]] = {}

    def vertex(
        self,
        key_a: tuple[int, int, int],
        key_b: tuple[int, int, int],
        value_a: float,
        value_b: float,
        position_a: np.ndarray,
        position_b: np.ndarray,
        audit: ExtractionAudit,
        isolated_namespace: int | None = None,
    ) -> int | None:
        denominator = value_a - value_b
        t = min(1.0, max(0.0, value_a / denominator if abs(denominator) > 1e-9 else 0.5))
        if key_b < key_a:
            key_a, key_b = key_b, key_a
            t = 1.0 - t
            position_a, position_b = position_b, position_a
        edge = (key_a, key_b) if isolated_namespace is None else (
            (key_a[0], key_a[1], key_a[2], isolated_namespace),
            (key_b[0], key_b[1], key_b[2], isolated_namespace),
        )
        entries = self.entries.setdefault(edge, [])
        for previous_t, index in entries:
            if abs(previous_t - t) <= self.merge_ratio:
                audit.edge_crossings_merged += 1
                return index
        if isolated_namespace is None and len(entries) >= 2:
            audit.edge_crossing_overflow_dropped += 1
            return None
        position = position_a + (position_b - position_a) * t
        index = len(self.vertices)
        self.vertices.append(position)
        entries.append((t, index))
        entries.sort(key=lambda item: item[0])
        return index


def append_hypothesis_triangles(
    build: MeshBuild,
    cache: EdgeVertexCache,
    hypothesis: CellHypothesis,
    minimum_weight: float,
    namespace: int | None,
) -> None:
    base = np.asarray(hypothesis.cell, dtype=np.int64)
    keys = [tuple((base + offset).tolist()) for offset in CORNER_OFFSETS]
    outward = hypothesis.normal
    for tetra in TETRAHEDRA:
        if any(hypothesis.weights[corner] < minimum_weight for corner in tetra):
            continue
        for edge0, edge1, edge2 in tetra_triangles(hypothesis.values, tetra):
            indices: list[int] = []
            for a, b in (edge0, edge1, edge2):
                vertex = cache.vertex(
                    keys[a], keys[b],
                    float(hypothesis.values[a]), float(hypothesis.values[b]),
                    hypothesis.positions[a], hypothesis.positions[b],
                    build.audit, namespace,
                )
                if vertex is None:
                    indices = []
                    break
                indices.append(vertex)
            if len(indices) != 3:
                continue
            a, b, c = indices
            if len({a, b, c}) < 3:
                build.audit.degenerate_triangles_dropped += 1
                continue
            cross = np.cross(build.vertices[b] - build.vertices[a], build.vertices[c] - build.vertices[a])
            if float(np.dot(cross, cross)) <= 1e-14:
                build.audit.degenerate_triangles_dropped += 1
                continue
            if float(np.dot(cross, outward)) < 0.0:
                b, c = c, b
            build.triangles.append((a, b, c))
            build.triangle_directions.append(hypothesis.direction)
            build.triangle_support.append(hypothesis.support)


def extract_independent(grid: DirectionalGrid, minimum_weight: float) -> MeshBuild:
    started = time.perf_counter()
    build = MeshBuild()
    cache = EdgeVertexCache(build.vertices, 0.0)
    for cell in sorted(grid.candidates):
        build.audit.scanned_cells += 1
        count = 0
        for direction in range(6):
            hypothesis = cell_hypothesis(grid, direction, cell, minimum_weight)
            if hypothesis is None:
                continue
            count += 1
            build.audit.zero_crossing_hypotheses += 1
            append_hypothesis_triangles(build, cache, hypothesis, minimum_weight, direction)
        if count > 1:
            build.audit.multi_direction_cells += 1
    deduplicate_triangles(build)
    enforce_edge_manifold(build)
    build.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return build


@dataclass
class PaperDmcCell:
    cell: tuple[int, int, int]
    index0: int
    index1: int
    regularized_index0: int
    regularized_index1: int
    edge_slot_mask: np.ndarray
    edge_offsets: np.ndarray
    edge_weights: np.ndarray

    @property
    def surface_count(self) -> int:
        return 2 if self.index1 > 0 else 1


def paper_transition_edge_mask(mc_index: int) -> int:
    mask = 0
    for edge, (corner_a, corner_b) in enumerate(PAPER_EDGE_ENDPOINT_CORNERS):
        if bool(mc_index & (1 << corner_a)) != bool(mc_index & (1 << corner_b)):
            mask |= 1 << edge
    return mask


def paper_mc_edge_compatible(first_index: int, second_index: int, edge: int) -> bool:
    corner_a, corner_b = PAPER_EDGE_ENDPOINT_CORNERS[edge]
    first_a = bool(first_index & (1 << corner_a))
    first_b = bool(first_index & (1 << corner_b))
    second_a = bool(second_index & (1 << corner_a))
    second_b = bool(second_index & (1 << corner_b))
    return not (
        (first_a and not first_b and not second_a and second_b)
        or (not first_a and first_b and second_a and not second_b)
    )


def paper_mc_index_compatible(first_index: int, second_index: int) -> bool:
    intersection = first_index & second_index
    if intersection == 0:
        return all(
            paper_mc_edge_compatible(first_index, second_index, edge)
            for edge in range(12)
        )
    compatible = False
    for edge, (corner_a, corner_b) in enumerate(PAPER_EDGE_ENDPOINT_CORNERS):
        if not (
            (intersection & (1 << corner_a))
            and (intersection & (1 << corner_b))
        ) and paper_mc_edge_compatible(first_index, second_index, edge):
            compatible = True
    return compatible


def paper_surface_offset(first: float, second: float) -> float:
    denominator = second - first
    return -first / denominator if abs(denominator) > 1e-8 else 0.5


def paper_direction_compatible(
    mc_index: int,
    paper_direction: int,
    sdf_values: np.ndarray,
    direction_compatibility: np.ndarray,
) -> bool:
    compatibility = int(direction_compatibility[mc_index, paper_direction])
    if compatibility == 0:
        return False
    if compatibility != 2:
        return True
    edges = PAPER_DIRECTION_EDGES_TO_CHECK[paper_direction]
    for pair in range(4):
        edge_index = edges[pair * 2]
        opposite_edge_index = edges[pair * 2 + 1]
        edge_a, edge_b = PAPER_EDGE_ENDPOINT_CORNERS[edge_index]
        opposite_a, opposite_b = PAPER_EDGE_ENDPOINT_CORNERS[opposite_edge_index]
        edge_a_inside = bool(mc_index & (1 << edge_a))
        edge_b_inside = bool(mc_index & (1 << edge_b))
        if edge_a_inside == edge_b_inside:
            continue
        if edge_b_inside:
            edge_a, edge_b = edge_b, edge_a
            opposite_a, opposite_b = opposite_b, opposite_a
        offset = paper_surface_offset(
            float(sdf_values[edge_a]), float(sdf_values[edge_b])
        )
        opposite_offset = paper_surface_offset(
            float(sdf_values[opposite_a]), float(sdf_values[opposite_b])
        )
        if offset > opposite_offset:
            return False
    return True


def filter_paper_mc_index_direction(
    mc_index: int,
    paper_direction: int,
    sdf_values: np.ndarray,
    index_decomposition: np.ndarray,
    direction_compatibility: np.ndarray,
) -> int:
    if mc_index <= 0 or mc_index == 255:
        return mc_index
    filtered_index = 0
    for raw_component in index_decomposition[mc_index]:
        component = int(raw_component)
        if component < 0:
            break
        if paper_direction_compatible(
            component, paper_direction, sdf_values, direction_compatibility
        ):
            filtered_index |= component
    return filtered_index if filtered_index != 0 else -1


def read_paper_direction_cell(
    grid: DirectionalGrid,
    scan_direction: int,
    cell: tuple[int, int, int],
    minimum_weight: float,
    audit: ExtractionAudit,
) -> tuple[int, float, np.ndarray, np.ndarray] | None:
    base = np.asarray(cell, dtype=np.int64)
    values = np.empty(8, dtype=np.float64)
    weights = np.empty(8, dtype=np.float64)
    mc_index = 0
    zero_weight_corners = 0
    low_weight_corners = 0
    for corner, offset in enumerate(PAPER_CORNER_OFFSETS):
        value, weight = grid.read(
            scan_direction, tuple((base + offset).tolist())
        )
        values[corner] = value
        weights[corner] = weight
        if weight <= 1e-8:
            zero_weight_corners += 1
        elif weight < minimum_weight:
            low_weight_corners += 1
        if value < 0.0:
            mc_index |= 1 << corner
    if zero_weight_corners or low_weight_corners:
        audit.paper_incomplete_zero_weight_corners += zero_weight_corners
        audit.paper_incomplete_low_weight_corners += low_weight_corners
        if zero_weight_corners:
            audit.paper_directions_missing_unwritten_corner += 1
        else:
            audit.paper_directions_blocked_only_by_weight += 1
        return None
    signed_offsets = PAPER_CORNER_OFFSETS.astype(np.float64) * 2.0 - 1.0
    gradient = np.sum(values[:, None] * signed_offsets, axis=0) * 0.25
    return mc_index, float(np.mean(weights)), gradient, values


@dataclass
class PaperDirectionCellState:
    raw_indices: list[int]
    filtered_indices: list[int]
    voted_indices: list[int]
    sdf_weights: np.ndarray
    sdf_values: np.ndarray


def evaluate_paper_direction_cell(
    grid: DirectionalGrid,
    cell_key: tuple[int, int, int],
    minimum_weight: float,
    index_decomposition: np.ndarray,
    direction_compatibility: np.ndarray,
    audit: ExtractionAudit,
) -> PaperDirectionCellState | None:
    """Run the paper's per-direction filter and Algorithm 1 exactly once.

    Keeping this stage shared by the independent control and the composed
    extractor makes their delta a measurement of Algorithm 2 composition only.
    """

    raw_indices = [-1] * 6
    filtered_indices = [-1] * 6
    sdf_weights = np.zeros(6, dtype=np.float64)
    sdf_values = np.ones((6, 8), dtype=np.float64)
    valid_directions = 0
    for paper_direction, scan_direction in enumerate(
        PAPER_DIRECTION_TO_SCANCOVER
    ):
        read = read_paper_direction_cell(
            grid, scan_direction, cell_key, minimum_weight, audit
        )
        if read is None:
            audit.paper_incomplete_direction_cells += 1
            continue
        valid_directions += 1
        mc_index, sdf_weight, gradient, values = read
        raw_indices[paper_direction] = mc_index
        sdf_values[paper_direction] = values
        sdf_weights[paper_direction] = sdf_weight
        if 0 < mc_index < 255:
            audit.paper_raw_hypotheses += 1
            audit.paper_raw_transition_edges += int(
                paper_transition_edge_mask(mc_index).bit_count()
            )
        filtered_index = filter_paper_mc_index_direction(
            mc_index,
            paper_direction,
            values,
            index_decomposition,
            direction_compatibility,
        )
        if 0 <= filtered_index < 255:
            gradient_length = float(np.linalg.norm(gradient))
            normalized_gradient = (
                gradient / gradient_length
                if gradient_length > 1e-8
                else np.zeros(3, dtype=np.float64)
            )
            compliance = float(
                np.dot(
                    normalized_gradient,
                    DIRECTION_VECTORS[scan_direction],
                )
            )
            sdf_weights[paper_direction] *= compliance
            if compliance < PAPER_DIRECTION_THRESHOLD:
                filtered_index = -1
                sdf_weights[paper_direction] = 0.0
        if 0 < mc_index < 255 and filtered_index < 0:
            audit.paper_intra_direction_rejected += 1
        filtered_indices[paper_direction] = filtered_index

    if valid_directions == 0:
        return None
    audit.paper_cells_with_complete_direction += 1
    raw_crossings = [index for index in raw_indices if 0 < index < 255]
    filtered_crossings = [
        index for index in filtered_indices if 0 < index < 255
    ]
    if raw_crossings:
        audit.paper_raw_crossing_cells += 1
    if filtered_crossings:
        audit.paper_filtered_crossing_cells += 1
        audit.paper_filtered_hypotheses += len(filtered_crossings)
        audit.paper_filtered_transition_edges += sum(
            paper_transition_edge_mask(index).bit_count()
            for index in filtered_crossings
        )

    voted_indices = filtered_indices.copy()
    # Algorithm 1: retain the authors' direction order and in-place update.
    for direction in range(6):
        mc_index = voted_indices[direction]
        if mc_index <= 0 or mc_index == 255:
            continue
        direction_weight = float(sdf_weights[direction])
        if direction_weight <= 1e-8:
            voted_indices[direction] = -1
            audit.paper_inter_direction_rejected += 1
            continue
        support_weight = 1.0
        for other in range(6):
            if voted_indices[other] < 0:
                continue
            if voted_indices[other] == 0:
                support_weight -= (
                    float(sdf_weights[other]) / direction_weight
                )
                voted_indices[direction] = 0
                break
            if other != direction and paper_mc_index_compatible(
                mc_index, voted_indices[other]
            ):
                support_weight += (
                    float(sdf_weights[other]) / direction_weight
                )
        if support_weight < 0.0:
            voted_indices[direction] = 0
        if voted_indices[direction] <= 0:
            audit.paper_inter_direction_rejected += 1

    voted_crossings = [
        index for index in voted_indices if 0 < index < 255
    ]
    if voted_crossings:
        audit.paper_voted_crossing_cells += 1
        audit.paper_voted_hypotheses += len(voted_crossings)
        audit.paper_voted_transition_edges += sum(
            paper_transition_edge_mask(index).bit_count()
            for index in voted_crossings
        )
    elif raw_crossings:
        audit.paper_empty_after_voting_cells += 1

    return PaperDirectionCellState(
        raw_indices,
        filtered_indices,
        voted_indices,
        sdf_weights,
        sdf_values,
    )


def evaluate_paper_edge_offsets(
    mc_indices: list[int],
    sdf_values: np.ndarray,
    sdf_weights: np.ndarray,
    audit: ExtractionAudit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slot_mask = np.zeros(12, dtype=np.uint8)
    offsets = np.zeros((12, 2), dtype=np.float64)
    weights = np.zeros((12, 2), dtype=np.float64)
    for edge, (corner_a, corner_b) in enumerate(PAPER_EDGE_ENDPOINT_CORNERS):
        numerators = [0.0, 0.0]
        totals = [0.0, 0.0]
        for direction in range(6):
            mc_index = mc_indices[direction]
            if mc_index < 0 or not (
                paper_transition_edge_mask(mc_index) & (1 << edge)
            ):
                continue
            weight = float(sdf_weights[direction])
            if weight <= 0.0:
                continue
            offset = paper_surface_offset(
                float(sdf_values[direction, corner_a]),
                float(sdf_values[direction, corner_b]),
            )
            slot = 0 if mc_index & (1 << corner_a) else 1
            numerators[slot] += weight * offset
            totals[slot] += weight
        for slot in range(2):
            if totals[slot] <= 0.0:
                continue
            slot_mask[edge] |= 1 << slot
            offsets[edge, slot] = numerators[slot] / totals[slot]
            weights[edge, slot] = totals[slot]
            audit.paper_edge_offset_slots += 1
        if slot_mask[edge] == 3:
            audit.paper_dual_offset_edges += 1
            if offsets[edge, 1] < offsets[edge, 0]:
                mean = float(np.mean(offsets[edge]))
                offsets[edge, :] = mean
    return slot_mask, offsets, weights


def paper_face_disagreements(left_index: int, right_index: int, axis: int) -> int:
    right_lookup = {
        tuple(offset.tolist()): corner
        for corner, offset in enumerate(PAPER_CORNER_OFFSETS)
    }
    disagreements = 0
    for left_corner, offset in enumerate(PAPER_CORNER_OFFSETS):
        if int(offset[axis]) != 1:
            continue
        right_offset = offset.copy()
        right_offset[axis] = 0
        right_corner = right_lookup[tuple(right_offset.tolist())]
        if bool(left_index & (1 << left_corner)) != bool(
            right_index & (1 << right_corner)
        ):
            disagreements += 1
    return disagreements


def audit_paper_neighbors(
    cells: dict[tuple[int, int, int], PaperDmcCell],
) -> tuple[int, int]:
    comparisons = 0
    disagreements = 0
    for cell_key, left in cells.items():
        for axis in range(3):
            neighbor = list(cell_key)
            neighbor[axis] += 1
            right = cells.get(tuple(neighbor))
            if right is None:
                continue
            comparisons += 1
            disagreements += paper_face_disagreements(
                left.regularized_index0, right.regularized_index0, axis
            )
    return comparisons, disagreements


def regularize_paper_indices(
    cells: dict[tuple[int, int, int], PaperDmcCell],
    audit: ExtractionAudit,
) -> None:
    source_indices = {key: cell.index0 for key, cell in cells.items()}
    for cell_key, cell in cells.items():
        source_index = cell.index0
        if source_index <= 0 or source_index == 255:
            continue
        regularized = source_index
        for corner, offset in enumerate(PAPER_CORNER_OFFSETS):
            physical = np.asarray(cell_key, dtype=np.int64) + offset
            inside_votes = 1 if source_index & (1 << corner) else 0
            votes = 1
            for neighbor_corner, neighbor_offset in enumerate(PAPER_CORNER_OFFSETS):
                neighbor_key = tuple((physical - neighbor_offset).tolist())
                if neighbor_key == cell_key:
                    continue
                neighbor_index = source_indices.get(neighbor_key)
                if neighbor_index is None or neighbor_index <= 0:
                    continue
                if neighbor_index & (1 << neighbor_corner):
                    inside_votes += 1
                votes += 1
            inside = inside_votes > votes // 2
            was_inside = bool(source_index & (1 << corner))
            if inside == was_inside:
                continue
            if inside:
                regularized |= 1 << corner
            else:
                regularized &= ~(1 << corner)
            audit.paper_regularized_corners += 1
        cell.regularized_index0 = regularized
        if regularized != source_index:
            audit.paper_regularized_cells += 1


def paper_physical_edge_key(
    cell_key: tuple[int, int, int],
    edge: int,
    slot: int,
) -> tuple[tuple[int, int, int], int, int]:
    base = np.asarray(cell_key, dtype=np.int64)
    corner_a, corner_b = PAPER_EDGE_ENDPOINT_CORNERS[edge]
    endpoint_a = tuple(
        (base + PAPER_CORNER_OFFSETS[corner_a]).tolist()
    )
    endpoint_b = tuple(
        (base + PAPER_CORNER_OFFSETS[corner_b]).tolist()
    )
    axis = next(
        index
        for index in range(3)
        if endpoint_a[index] != endpoint_b[index]
    )
    minimum = endpoint_a if endpoint_a < endpoint_b else endpoint_b
    return minimum, axis, slot


def paper_shared_edge_offsets(
    cells: dict[tuple[int, int, int], PaperDmcCell],
) -> dict[tuple[tuple[int, int, int], int, int], float]:
    samples: dict[
        tuple[tuple[int, int, int], int, int], list[float]
    ] = {}
    for cell_key, cell in cells.items():
        for edge in range(12):
            for slot in range(2):
                if not (int(cell.edge_slot_mask[edge]) & (1 << slot)):
                    continue
                weight = float(cell.edge_weights[edge, slot])
                if weight <= 0.0:
                    continue
                key = paper_physical_edge_key(cell_key, edge, slot)
                sample = samples.setdefault(key, [0.0, 0.0])
                sample[0] += float(cell.edge_offsets[edge, slot]) * weight
                sample[1] += weight
    return {
        key: numerator / weight
        for key, (numerator, weight) in samples.items()
        if weight > 0.0
    }


def quarantine_unmeasured_regularization(
    cells: dict[tuple[int, int, int], PaperDmcCell],
    audit: ExtractionAudit,
) -> None:
    """Reject a topology edit that has no measured physical-edge position.

    The paper regularizes MC indices, while its mesh-unit structure owns edge
    vertices globally.  A regularization result is publishable only when every
    requested edge/slot can be resolved from that shared physical-edge ledger.
    Reverting the cell preserves the measured pre-regularization topology and
    is strictly more conservative than inventing a midpoint vertex.
    """

    shared_offsets = paper_shared_edge_offsets(cells)
    for cell_key, cell in cells.items():
        if cell.regularized_index0 == cell.index0:
            continue
        supported = True
        edge_mask = paper_transition_edge_mask(cell.regularized_index0)
        for edge, (corner_a, _) in enumerate(PAPER_EDGE_ENDPOINT_CORNERS):
            if not (edge_mask & (1 << edge)):
                continue
            slot = 0 if cell.regularized_index0 & (1 << corner_a) else 1
            if paper_physical_edge_key(cell_key, edge, slot) not in shared_offsets:
                supported = False
                break
        if supported:
            continue
        cell.regularized_index0 = cell.index0
        audit.paper_regularization_reverted_cells += 1


def quarantine_nonmanifold_regularization(
    cells: dict[tuple[int, int, int], PaperDmcCell],
    triangle_edges: np.ndarray,
    audit: ExtractionAudit,
) -> None:
    """Rollback only regularization edits that create non-manifold edges."""

    reverted: set[tuple[int, int, int]] = set()
    for _ in range(4):
        edge_users: dict[
            tuple[
                tuple[tuple[int, int, int], int, int],
                tuple[tuple[int, int, int], int, int],
            ],
            list[tuple[int, int, int]],
        ] = {}
        for cell_key, cell in cells.items():
            indices = [cell.regularized_index0]
            if cell.surface_count == 2:
                indices.append(cell.regularized_index1)
            for mc_index in indices:
                if mc_index <= 0 or mc_index == 255:
                    continue
                for entry in range(0, 16, 3):
                    first_edge = int(triangle_edges[mc_index, entry])
                    if first_edge < 0:
                        break
                    vertex_keys: list[
                        tuple[tuple[int, int, int], int, int]
                    ] = []
                    for edge in (
                        first_edge,
                        int(triangle_edges[mc_index, entry + 1]),
                        int(triangle_edges[mc_index, entry + 2]),
                    ):
                        corner_a, _ = PAPER_EDGE_ENDPOINT_CORNERS[edge]
                        slot = 0 if mc_index & (1 << corner_a) else 1
                        vertex_keys.append(
                            paper_physical_edge_key(cell_key, edge, slot)
                        )
                    if len(set(vertex_keys)) != 3:
                        continue
                    for index in range(3):
                        left = vertex_keys[index]
                        right = vertex_keys[(index + 1) % 3]
                        mesh_edge = (
                            (left, right) if left < right else (right, left)
                        )
                        edge_users.setdefault(mesh_edge, []).append(cell_key)
        offenders = {
            cell_key
            for users in edge_users.values()
            if len(users) > 2
            for cell_key in users
            if cells[cell_key].regularized_index0 != cells[cell_key].index0
        }
        if not offenders:
            break
        changed = False
        for cell_key in offenders:
            cell = cells[cell_key]
            if cell.regularized_index0 == cell.index0:
                continue
            cell.regularized_index0 = cell.index0
            reverted.add(cell_key)
            changed = True
        if not changed:
            break
    audit.paper_regularization_reverted_nonmanifold_cells = len(reverted)


def build_paper_dmc_mesh(
    grid: DirectionalGrid,
    cells: dict[tuple[int, int, int], PaperDmcCell],
    triangle_edges: np.ndarray,
    audit: ExtractionAudit,
    ownership_output: dict[str, Any] | None = None,
) -> MeshBuild:
    build = MeshBuild(audit=audit)
    # The paper's mesh-unit layout stores a physical edge vertex once and lets
    # all incident cells reference that owner.  A composed/regularized cell can
    # therefore legally need an edge slot measured by a neighboring cell.  The
    # old offline port checked only the current cell and incorrectly deferred
    # such triangles.  Build the shared edge ledger first; it only supplies a
    # position for topology already authorized by the combined MC index.
    shared_edge_offsets = paper_shared_edge_offsets(cells)
    edge_vertices: dict[
        tuple[tuple[int, int, int], int, int], int
    ] = {}
    required_slots: set[tuple[tuple[int, int, int], int, int]] = set()
    local_slots: set[tuple[tuple[int, int, int], int, int]] = set()
    recovered_slots: set[tuple[tuple[int, int, int], int, int]] = set()
    unresolved_slots: set[tuple[tuple[int, int, int], int, int]] = set()
    cell_triangles: dict[tuple[int, int, int], list[int]] = {}
    for cell_key in sorted(cells):
        cell = cells[cell_key]
        for mc_index in (
            cell.regularized_index0,
            cell.regularized_index1 if cell.surface_count == 2 else 0,
        ):
            if mc_index <= 0 or mc_index == 255:
                continue
            local_vertices: list[int | None] = [None] * 12
            for entry in range(0, 16, 3):
                edge_a = int(triangle_edges[mc_index, entry])
                if edge_a < 0:
                    break
                triangle_edges_current = (
                    edge_a,
                    int(triangle_edges[mc_index, entry + 1]),
                    int(triangle_edges[mc_index, entry + 2]),
                )
                triangle: list[int] = []
                for edge in triangle_edges_current:
                    cached = local_vertices[edge]
                    if cached is not None:
                        triangle.append(cached)
                        continue
                    corner_a, corner_b = PAPER_EDGE_ENDPOINT_CORNERS[edge]
                    slot = 0 if mc_index & (1 << corner_a) else 1
                    endpoint_a = tuple(
                        (np.asarray(cell_key, dtype=np.int64)
                         + PAPER_CORNER_OFFSETS[corner_a]).tolist()
                    )
                    endpoint_b = tuple(
                        (np.asarray(cell_key, dtype=np.int64)
                         + PAPER_CORNER_OFFSETS[corner_b]).tolist()
                    )
                    axis = next(
                        index
                        for index in range(3)
                        if endpoint_a[index] != endpoint_b[index]
                    )
                    minimum = endpoint_a if endpoint_a < endpoint_b else endpoint_b
                    key = (minimum, axis, slot)
                    required_slots.add(key)
                    local_has_slot = bool(
                        int(cell.edge_slot_mask[edge]) & (1 << slot)
                    )
                    if local_has_slot:
                        local_slots.add(key)
                    elif key in shared_edge_offsets:
                        recovered_slots.add(key)
                    else:
                        unresolved_slots.add(key)
                        triangle = []
                        break
                    vertex = edge_vertices.get(key)
                    if vertex is None:
                        minimum_position = (
                            np.asarray(minimum, dtype=np.float64) * grid.voxel
                        )
                        axis_vector = np.zeros(3, dtype=np.float64)
                        axis_vector[axis] = grid.voxel
                        t = float(shared_edge_offsets[key])
                        vertex = len(build.vertices)
                        build.vertices.append(
                            minimum_position + axis_vector * t
                        )
                        edge_vertices[key] = vertex
                    local_vertices[edge] = vertex
                    triangle.append(vertex)
                if len(triangle) != 3 or len(set(triangle)) != 3:
                    audit.paper_unmeasured_edge_deferred_triangles += 1
                    continue
                build.triangles.append(tuple(triangle))
                build.triangle_directions.append(0)
                build.triangle_support.append(1.0)
                cell_triangles.setdefault(cell_key, []).append(
                    len(build.triangles) - 1
                )
    audit.paper_required_edge_slots = len(required_slots)
    audit.paper_local_edge_slots_used = len(local_slots)
    audit.paper_shared_edge_slots_recovered = len(recovered_slots)
    audit.paper_unresolved_edge_slots = len(unresolved_slots)
    if ownership_output is not None:
        ownership_output.clear()
        ownership_output.update(
            {
                "cellTriangles": cell_triangles,
                "edgeVertices": edge_vertices,
            }
        )
    return build


def extract_paper_stage_independent(
    grid: DirectionalGrid,
    minimum_weight: float,
    stage: str,
) -> MeshBuild:
    """Mesh one pre-composition paper stage independently.

    This is an attribution control, not a production candidate.  It deliberately
    preserves one mesh namespace per direction so that the only semantic
    difference between adjacent controls is one upstream paper stage.
    """

    if stage not in {"raw", "filtered", "voted"}:
        raise ValueError(f"unsupported paper independent stage: {stage}")
    started = time.perf_counter()
    triangle_edges, index_decomposition, direction_compatibility = (
        paper_dmc_tables()
    )
    build = MeshBuild()
    edge_vertices: dict[
        tuple[int, tuple[int, int, int], int], int
    ] = {}
    for cell_key in sorted(grid.candidates):
        build.audit.scanned_cells += 1
        state = evaluate_paper_direction_cell(
            grid,
            cell_key,
            minimum_weight,
            index_decomposition,
            direction_compatibility,
            build.audit,
        )
        if state is None:
            continue
        stage_indices = {
            "raw": state.raw_indices,
            "filtered": state.filtered_indices,
            "voted": state.voted_indices,
        }[stage]
        kept = [
            index for index in stage_indices if 0 < index < 255
        ]
        if not kept:
            continue
        build.audit.zero_crossing_hypotheses += len(kept)
        if len(kept) > 1:
            build.audit.multi_direction_cells += 1
        for paper_direction, mc_index in enumerate(stage_indices):
            if mc_index <= 0 or mc_index == 255:
                continue
            local_vertices: list[int | None] = [None] * 12
            for entry in range(0, 16, 3):
                edge_a = int(triangle_edges[mc_index, entry])
                if edge_a < 0:
                    break
                triangle: list[int] = []
                for edge in (
                    edge_a,
                    int(triangle_edges[mc_index, entry + 1]),
                    int(triangle_edges[mc_index, entry + 2]),
                ):
                    cached = local_vertices[edge]
                    if cached is not None:
                        triangle.append(cached)
                        continue
                    corner_a, corner_b = PAPER_EDGE_ENDPOINT_CORNERS[edge]
                    endpoint_a = tuple(
                        (
                            np.asarray(cell_key, dtype=np.int64)
                            + PAPER_CORNER_OFFSETS[corner_a]
                        ).tolist()
                    )
                    endpoint_b = tuple(
                        (
                            np.asarray(cell_key, dtype=np.int64)
                            + PAPER_CORNER_OFFSETS[corner_b]
                        ).tolist()
                    )
                    axis = next(
                        index
                        for index in range(3)
                        if endpoint_a[index] != endpoint_b[index]
                    )
                    minimum = endpoint_a if endpoint_a < endpoint_b else endpoint_b
                    key = (paper_direction, minimum, axis)
                    vertex = edge_vertices.get(key)
                    if vertex is None:
                        value_a = float(state.sdf_values[paper_direction, corner_a])
                        value_b = float(state.sdf_values[paper_direction, corner_b])
                        t = paper_surface_offset(value_a, value_b)
                        position_a = np.asarray(endpoint_a, dtype=np.float64) * grid.voxel
                        position_b = np.asarray(endpoint_b, dtype=np.float64) * grid.voxel
                        vertex = len(build.vertices)
                        build.vertices.append(
                            position_a + (position_b - position_a) * t
                        )
                        edge_vertices[key] = vertex
                    local_vertices[edge] = vertex
                    triangle.append(vertex)
                if len(triangle) != 3 or len(set(triangle)) != 3:
                    build.audit.degenerate_triangles_dropped += 1
                    continue
                build.triangles.append(tuple(triangle))
                build.triangle_directions.append(
                    PAPER_DIRECTION_TO_SCANCOVER[paper_direction]
                )
                build.triangle_support.append(
                    float(state.sdf_weights[paper_direction])
                )
    build.metadata["paperDmcControl"] = {
        "stage": stage,
        "purpose": "attribute pre-composition coverage loss without changing fusion",
    }
    build.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return build


def extract_paper_dmc(
    grid: DirectionalGrid,
    minimum_weight: float,
    regularize: bool = True,
    component_compatibility: bool = True,
    ownership_output: dict[str, Any] | None = None,
    cells_output: dict[tuple[int, int, int], PaperDmcCell] | None = None,
) -> MeshBuild:
    """Faithful Splietker et al. composition and classic MC extraction."""

    started = time.perf_counter()
    triangle_edges, index_decomposition, direction_compatibility = (
        paper_dmc_tables()
    )
    audit = ExtractionAudit()
    cells: dict[tuple[int, int, int], PaperDmcCell] = {}
    for cell_key in sorted(grid.candidates):
        audit.scanned_cells += 1
        state = evaluate_paper_direction_cell(
            grid,
            cell_key,
            minimum_weight,
            index_decomposition,
            direction_compatibility,
            audit,
        )
        if state is None:
            continue
        mc_indices = state.voted_indices
        sdf_weights = state.sdf_weights
        sdf_values = state.sdf_values
        kept = [index for index in mc_indices if 0 < index < 255]
        if not kept:
            continue
        audit.zero_crossing_hypotheses += len(kept)
        if len(kept) > 1:
            audit.multi_direction_cells += 1

        slot_mask, offsets, weights = evaluate_paper_edge_offsets(
            mc_indices, sdf_values, sdf_weights, audit
        )
        combined0 = 0
        combined1 = 0
        for mc_index in mc_indices:
            if mc_index <= 0 or mc_index == 255:
                continue
            for raw_component in index_decomposition[mc_index]:
                component = int(raw_component)
                if component < 0:
                    break
                audit.paper_components += 1
                # Algorithm 2 compares the current connected component with
                # the accumulated index.  Comparing the complete directional
                # MC index here lets an unrelated component authorize this
                # one, after which the binary intersection can erase legal
                # topology.  Keep the old whole-index behavior available only
                # as an offline A/B control.
                compatibility_index = component if component_compatibility else mc_index
                if combined0 == 0:
                    combined0 = component
                elif paper_mc_index_compatible(compatibility_index, combined0):
                    combined0 &= component
                elif combined1 == 0:
                    combined1 = component
                elif paper_mc_index_compatible(compatibility_index, combined1):
                    combined1 &= component
                else:
                    audit.paper_overflow_deferred_components += 1
        if combined0 <= 0 or combined0 == 255:
            if 0 < combined1 < 255:
                combined0, combined1 = combined1, 0
            else:
                continue
        if combined1 == 255:
            combined1 = 0
        cell = PaperDmcCell(
            cell_key,
            combined0,
            combined1,
            combined0,
            combined1,
            slot_mask,
            offsets,
            weights,
        )
        cells[cell_key] = cell
        if cell.surface_count == 2:
            audit.paper_double_surface_cells += 1
        else:
            audit.paper_single_surface_cells += 1
        audit.paper_combined_transition_edges += int(
            paper_transition_edge_mask(combined0).bit_count()
        )
        if combined1 > 0:
            audit.paper_combined_transition_edges += int(
                paper_transition_edge_mask(combined1).bit_count()
            )

    comparisons, disagreements = audit_paper_neighbors(cells)
    audit.paper_neighbor_face_comparisons = comparisons
    audit.paper_neighbor_disagreements_before = disagreements
    if regularize:
        regularize_paper_indices(cells, audit)
        quarantine_unmeasured_regularization(cells, audit)
        quarantine_nonmanifold_regularization(
            cells, triangle_edges, audit
        )
    comparisons_after, disagreements_after = audit_paper_neighbors(cells)
    audit.paper_neighbor_face_comparisons = max(comparisons, comparisons_after)
    audit.paper_neighbor_disagreements_after = disagreements_after
    for cell in cells.values():
        audit.paper_regularized_transition_edges += int(
            paper_transition_edge_mask(cell.regularized_index0).bit_count()
        )
        if cell.surface_count == 2:
            audit.paper_regularized_transition_edges += int(
                paper_transition_edge_mask(cell.regularized_index1).bit_count()
            )
    build = build_paper_dmc_mesh(
        grid,
        cells,
        triangle_edges,
        audit,
        ownership_output=ownership_output,
    )
    if cells_output is not None:
        cells_output.clear()
        cells_output.update(cells)
    build.metadata["paperDmc"] = {
        "source": "Splietker et al. IROS 2019 / MeshHashingDTSDF",
        "composition": "Algorithm 1 weighted voting + MCIndexCompatible",
        "topology": "authors' kIndexDecomposition and classic MC triangle table",
        "regularized": regularize,
        "compatibilityOperand": (
            "connected_component" if component_compatibility else "whole_direction_index"
        ),
        "edgeOwnership": "shared physical edge and orientation slot",
        "regularizationPolicy": (
            "neighbor MC sign edit is quarantined when the shared edge ledger "
            "cannot supply every requested intersection"
        ),
        "cellCount": len(cells),
    }
    build.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return build


def cluster_hypotheses(
    hypotheses: list[CellHypothesis],
    parallel_dot: float,
    voxel: float,
    audit: ExtractionAudit,
) -> list[list[CellHypothesis]]:
    clusters: list[list[CellHypothesis]] = []
    for hypothesis in sorted(hypotheses, key=lambda item: item.support, reverse=True):
        joined = False
        for cluster in clusters:
            reference = cluster[0]
            normal_dot = float(np.dot(hypothesis.normal, reference.normal))
            separation = float(np.linalg.norm(hypothesis.centroid - reference.centroid))
            if normal_dot >= parallel_dot and separation <= voxel * 0.85:
                cluster.append(hypothesis)
                audit.parallel_hypotheses_collapsed += 1
                joined = True
                break
        if not joined:
            clusters.append([hypothesis])
    clusters.sort(key=lambda cluster: sum(item.support for item in cluster), reverse=True)
    if len(clusters) > 2:
        audit.incompatible_weak_hypotheses_dropped += sum(len(cluster) for cluster in clusters[2:])
        clusters = clusters[:2]
    return clusters


def combine_cluster(cluster: list[CellHypothesis], minimum_weight: float) -> CellHypothesis | None:
    reference = cluster[0]
    combined_values = np.ones(8, dtype=np.float64)
    combined_weights = np.zeros(8, dtype=np.float64)
    for corner in range(8):
        numerator = 0.0
        denominator = 0.0
        for hypothesis in cluster:
            weight = float(hypothesis.weights[corner])
            if weight < minimum_weight:
                continue
            direction_alignment = max(0.0, float(np.dot(hypothesis.normal, DIRECTION_VECTORS[hypothesis.direction])))
            effective = weight * max(0.15, direction_alignment)
            numerator += float(hypothesis.values[corner]) * effective
            denominator += effective
        if denominator > 0.0:
            combined_values[corner] = numerator / denominator
            combined_weights[corner] = denominator
    valid = combined_weights >= minimum_weight
    if np.sum(valid) < 4 or not (np.any(combined_values[valid] < 0.0) and np.any(combined_values[valid] >= 0.0)):
        return None
    weighted_normal = sum((item.normal * item.support for item in cluster), np.zeros(3, dtype=np.float64))
    normal = normalize(weighted_normal) if np.linalg.norm(weighted_normal) > 1e-8 else reference.normal
    centroid = sum((item.centroid * item.support for item in cluster), np.zeros(3, dtype=np.float64))
    support = sum(item.support for item in cluster)
    centroid /= max(1e-8, support)
    return CellHypothesis(
        reference.direction,
        reference.cell,
        combined_values,
        combined_weights,
        reference.positions,
        normal,
        centroid,
        support,
    )


def extract_composed(
    grid: DirectionalGrid,
    minimum_weight: float,
    valid_gradient_dot: float,
    parallel_dot: float,
    edge_merge_ratio: float,
) -> MeshBuild:
    started = time.perf_counter()
    build = MeshBuild()
    cache = EdgeVertexCache(build.vertices, edge_merge_ratio)
    for cell in sorted(grid.candidates):
        build.audit.scanned_cells += 1
        valid_hypotheses: list[CellHypothesis] = []
        raw_count = 0
        for direction in range(6):
            hypothesis = cell_hypothesis(grid, direction, cell, minimum_weight)
            if hypothesis is None:
                continue
            raw_count += 1
            build.audit.zero_crossing_hypotheses += 1
            compliance = float(np.dot(hypothesis.normal, DIRECTION_VECTORS[direction]))
            if compliance < valid_gradient_dot:
                build.audit.invalid_direction_hypotheses += 1
                continue
            valid_hypotheses.append(hypothesis)
        if raw_count > 1:
            build.audit.multi_direction_cells += 1
        if not valid_hypotheses:
            continue
        clusters = cluster_hypotheses(valid_hypotheses, parallel_dot, grid.voxel, build.audit)
        if len(clusters) > 1:
            first = clusters[0][0]
            second = clusters[1][0]
            angle_dot = abs(float(np.dot(first.normal, second.normal)))
            separation = float(np.linalg.norm(first.centroid - second.centroid))
            if angle_dot >= parallel_dot and separation < grid.voxel:
                build.audit.conservative_conflict_cells += 1
                clusters = clusters[:1]
        for cluster in clusters:
            combined = combine_cluster(cluster, minimum_weight)
            if combined is None:
                build.audit.incompatible_weak_hypotheses_dropped += len(cluster)
                continue
            append_hypothesis_triangles(build, cache, combined, minimum_weight, None)
    deduplicate_triangles(build)
    enforce_edge_manifold(build)
    build.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return build


def deduplicate_triangles(build: MeshBuild) -> None:
    unique: set[tuple[int, int, int]] = set()
    triangles: list[tuple[int, int, int]] = []
    directions: list[int] = []
    supports: list[float] = []
    for triangle, direction, support in zip(
        build.triangles, build.triangle_directions, build.triangle_support
    ):
        key = tuple(sorted(triangle))
        if key in unique:
            build.audit.duplicate_triangles_dropped += 1
            continue
        unique.add(key)
        triangles.append(triangle)
        directions.append(direction)
        supports.append(support)
    build.triangles = triangles
    build.triangle_directions = directions
    build.triangle_support = supports


def enforce_edge_manifold(build: MeshBuild) -> None:
    """Keep the strongest triangles while preventing more than two per edge.

    This is intentionally conservative: a narrow incomplete junction is safer
    than a branch or interpenetrating sheet in the production safety mesh.
    """
    edge_use: dict[tuple[int, int], int] = {}
    kept: list[int] = []
    order = sorted(
        range(len(build.triangles)),
        key=lambda index: (-build.triangle_support[index], index),
    )
    for index in order:
        triangle = build.triangles[index]
        edges = []
        for a, b in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edges.append((a, b) if a < b else (b, a))
        if any(edge_use.get(edge, 0) >= 2 for edge in edges):
            build.audit.nonmanifold_triangles_dropped += 1
            continue
        kept.append(index)
        for edge in edges:
            edge_use[edge] = edge_use.get(edge, 0) + 1
    kept.sort()
    build.triangles = [build.triangles[index] for index in kept]
    build.triangle_directions = [build.triangle_directions[index] for index in kept]
    build.triangle_support = [build.triangle_support[index] for index in kept]


def extract_feature_qef_shadow(
    source: MeshBuild,
    voxel: float,
    feature_angle_degrees: float,
    neighborhood_voxel_ratio: float,
    maximum_move_voxel_ratio: float,
    minimum_family_support_ratio: float,
    rank_ratio: float,
) -> MeshBuild:
    """Relocate only proven crease/corner vertices with a Hermite-style QEF.

    Connectivity is copied verbatim from the conservative composed mesh.  This
    isolates the question that motivated this experiment: can established
    feature-point placement recover a sharp junction when the DMC topology is
    already coherent?  Boundary vertices are deliberately left untouched so
    an apparent sharpness gain cannot be purchased by closing scan frontiers.
    """
    started = time.perf_counter()
    build = MeshBuild(
        vertices=[vertex.copy() for vertex in source.vertices],
        triangles=list(source.triangles),
        triangle_directions=list(source.triangle_directions),
        triangle_support=list(source.triangle_support),
        audit=copy.deepcopy(source.audit),
    )
    if not build.vertices or not build.triangles:
        build.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return build

    vertices = np.asarray(source.vertices, dtype=np.float64)
    triangles = np.asarray(source.triangles, dtype=np.int64)
    triangle_points = vertices[triangles]
    cross = np.cross(
        triangle_points[:, 1] - triangle_points[:, 0],
        triangle_points[:, 2] - triangle_points[:, 0],
    )
    double_area = np.linalg.norm(cross, axis=1)
    valid_triangle = double_area > 1e-10
    normals = np.zeros_like(cross)
    normals[valid_triangle] = cross[valid_triangle] / double_area[valid_triangle, None]
    centroids = np.mean(triangle_points, axis=1)
    areas = double_area * 0.5

    vertex_faces: list[list[int]] = [[] for _ in range(len(vertices))]
    vertex_neighbors: list[set[int]] = [set() for _ in range(len(vertices))]
    edge_use: dict[tuple[int, int], int] = {}
    for face_index, triangle in enumerate(triangles):
        a, b, c = (int(triangle[0]), int(triangle[1]), int(triangle[2]))
        vertex_faces[a].append(face_index)
        vertex_faces[b].append(face_index)
        vertex_faces[c].append(face_index)
        vertex_neighbors[a].update((b, c))
        vertex_neighbors[b].update((a, c))
        vertex_neighbors[c].update((a, b))
        for edge_a, edge_b in ((a, b), (b, c), (c, a)):
            edge = (edge_a, edge_b) if edge_a < edge_b else (edge_b, edge_a)
            edge_use[edge] = edge_use.get(edge, 0) + 1
    boundary_vertices: set[int] = set()
    for (a, b), use in edge_use.items():
        if use == 1:
            boundary_vertices.add(a)
            boundary_vertices.add(b)

    family_merge_dot = math.cos(math.radians(max(8.0, feature_angle_degrees * 0.55)))
    feature_dot = math.cos(math.radians(feature_angle_degrees))
    neighborhood_radius = voxel * neighborhood_voxel_ratio
    maximum_move = voxel * maximum_move_voxel_ratio
    output = vertices.copy()
    displacement: list[float] = []
    residual_ratios: list[float] = []
    audit = {
        "source": "Kobbelt_2001_OpenVDB_findFeaturePoint_Ju_2002_QEF",
        "candidates": 0,
        "movedVertices": 0,
        "creaseRank2Vertices": 0,
        "cornerRank3Vertices": 0,
        "boundaryVerticesHeld": len(boundary_vertices),
        "insufficientFamilies": 0,
        "parallelFamiliesRejected": 0,
        "illConditionedRejected": 0,
        "residualRejected": 0,
        "moveClamped": 0,
    }

    for vertex_index, origin in enumerate(vertices):
        if vertex_index in boundary_vertices:
            continue
        face_candidates: set[int] = set(vertex_faces[vertex_index])
        for neighbor in vertex_neighbors[vertex_index]:
            face_candidates.update(vertex_faces[neighbor])
        local_faces = [
            face
            for face in face_candidates
            if valid_triangle[face]
            and float(np.linalg.norm(centroids[face] - origin)) <= neighborhood_radius
        ]
        if len(local_faces) < 4:
            audit["insufficientFamilies"] += 1
            continue

        families: list[dict[str, Any]] = []
        for face in sorted(local_faces, key=lambda item: float(areas[item]), reverse=True):
            normal = normals[face]
            distance = float(np.linalg.norm(centroids[face] - origin))
            weight = float(areas[face]) / max(0.15 * voxel, distance + 0.15 * voxel)
            best_family = -1
            best_dot = -1.0
            for family_index, family in enumerate(families):
                family_normal = normalize(family["normalSum"])
                normal_dot = float(np.dot(normal, family_normal))
                if normal_dot > best_dot:
                    best_dot = normal_dot
                    best_family = family_index
            if best_family >= 0 and best_dot >= family_merge_dot:
                family = families[best_family]
                family["normalSum"] += normal * weight
                family["pointSum"] += centroids[face] * weight
                family["weight"] += weight
                family["faces"] += 1
            else:
                families.append(
                    {
                        "normalSum": normal * weight,
                        "pointSum": centroids[face] * weight,
                        "weight": weight,
                        "faces": 1,
                    }
                )

        total_weight = sum(float(family["weight"]) for family in families)
        families = [
            family
            for family in families
            if family["faces"] >= 2
            and float(family["weight"]) >= total_weight * minimum_family_support_ratio
        ]
        families.sort(key=lambda family: float(family["weight"]), reverse=True)
        if len(families) < 2:
            audit["insufficientFamilies"] += 1
            continue

        selected: list[dict[str, Any]] = []
        for family in families:
            family_normal = normalize(family["normalSum"])
            if not selected:
                selected.append(family)
                continue
            if all(
                abs(float(np.dot(family_normal, normalize(previous["normalSum"])))) < feature_dot
                for previous in selected
            ):
                selected.append(family)
            if len(selected) == 3:
                break
        if len(selected) < 2:
            audit["parallelFamiliesRejected"] += 1
            continue
        audit["candidates"] += 1

        plane_normals = np.asarray(
            [normalize(family["normalSum"]) for family in selected],
            dtype=np.float64,
        )
        plane_points = np.asarray(
            [family["pointSum"] / max(1e-12, float(family["weight"])) for family in selected],
            dtype=np.float64,
        )
        plane_weights = np.asarray(
            [float(family["weight"]) for family in selected],
            dtype=np.float64,
        )
        plane_weights /= max(1e-12, float(np.max(plane_weights)))
        weighted_a = plane_normals * np.sqrt(plane_weights[:, None])
        weighted_b = (
            np.sum(plane_normals * (plane_points - origin), axis=1)
            * np.sqrt(plane_weights)
        )
        u, singular_values, vt = np.linalg.svd(weighted_a, full_matrices=False)
        if len(singular_values) == 0 or singular_values[0] <= 1e-8:
            audit["illConditionedRejected"] += 1
            continue
        retained = singular_values >= singular_values[0] * rank_ratio
        rank = int(np.sum(retained))
        if rank < 2:
            audit["illConditionedRejected"] += 1
            continue
        coefficients = np.zeros_like(singular_values)
        coefficients[retained] = (u.T @ weighted_b)[retained] / singular_values[retained]
        proposal = origin + vt.T @ coefficients
        move = proposal - origin
        move_length = float(np.linalg.norm(move))
        if not np.all(np.isfinite(proposal)) or move_length <= 1e-6:
            continue
        if move_length > maximum_move:
            proposal = origin + move * (maximum_move / move_length)
            move_length = maximum_move
            audit["moveClamped"] += 1

        before = float(np.sum(plane_weights * np.square(
            np.sum(plane_normals * (origin - plane_points), axis=1)
        )))
        after = float(np.sum(plane_weights * np.square(
            np.sum(plane_normals * (proposal - plane_points), axis=1)
        )))
        if after >= before * 0.98:
            audit["residualRejected"] += 1
            continue
        output[vertex_index] = proposal
        displacement.append(move_length)
        residual_ratios.append(after / max(1e-12, before))
        audit["movedVertices"] += 1
        if rank >= 3:
            audit["cornerRank3Vertices"] += 1
        else:
            audit["creaseRank2Vertices"] += 1

    build.vertices = [vertex.copy() for vertex in output]
    feature_elapsed_ms = (time.perf_counter() - started) * 1000.0
    build.elapsed_ms = source.elapsed_ms + feature_elapsed_ms
    if displacement:
        displacement_array = np.asarray(displacement, dtype=np.float64)
        audit["displacementMetersP50"] = float(np.percentile(displacement_array, 50))
        audit["displacementMetersP95"] = float(np.percentile(displacement_array, 95))
        audit["displacementMetersMax"] = float(np.max(displacement_array))
        audit["qefResidualRatioP50"] = float(np.percentile(residual_ratios, 50))
        audit["qefResidualRatioP95"] = float(np.percentile(residual_ratios, 95))
    else:
        audit["displacementMetersP50"] = 0.0
        audit["displacementMetersP95"] = 0.0
        audit["displacementMetersMax"] = 0.0
        audit["qefResidualRatioP50"] = 1.0
        audit["qefResidualRatioP95"] = 1.0
    build.metadata["featureQef"] = audit
    build.metadata["featureQef"]["placementMs"] = feature_elapsed_ms
    return build


def trilinear_gradient(values: np.ndarray, local: np.ndarray, voxel: float) -> np.ndarray:
    """Evaluate the gradient of the cell's trilinear scalar interpolant."""
    u, v, w = np.clip(local, 0.0, 1.0)
    dx = (
        (values[1] - values[0]) * (1.0 - v) * (1.0 - w)
        + (values[2] - values[3]) * v * (1.0 - w)
        + (values[5] - values[4]) * (1.0 - v) * w
        + (values[6] - values[7]) * v * w
    )
    dy = (
        (values[3] - values[0]) * (1.0 - u) * (1.0 - w)
        + (values[2] - values[1]) * u * (1.0 - w)
        + (values[7] - values[4]) * (1.0 - u) * w
        + (values[6] - values[5]) * u * w
    )
    dz = (
        (values[4] - values[0]) * (1.0 - u) * (1.0 - v)
        + (values[5] - values[1]) * u * (1.0 - v)
        + (values[7] - values[3]) * (1.0 - u) * v
        + (values[6] - values[2]) * u * v
    )
    return np.asarray([dx, dy, dz], dtype=np.float64) / max(1e-12, voxel)


def extract_tsdf_hermite_feature_points(
    grid: DirectionalGrid,
    minimum_weight: float,
    valid_gradient_dot: float,
    feature_angle_degrees: float,
    minimum_family_support_ratio: float,
    rank_ratio: float,
    certificate_min_frames_per_family: int = 0,
    certificate_min_views_per_family: int = 0,
    certificate_min_samples_per_family: int = 1,
    certificate_min_rank_ratio: float = 0.0,
    certificate_min_cell_margin_ratio: float = 0.0,
    certificate_min_family_weight_ratio: float = 0.0,
    certificate_min_qef_displacement_ratio: float = 0.0,
) -> dict[str, Any]:
    """Build a read-only sharp-feature oracle from TSDF Hermite evidence.

    Unlike ``extract_feature_qef_shadow``, this never reads triangle normals.
    Every plane comes directly from a supported TSDF zero crossing and the
    gradient of its trilinear cell interpolant.  It emits points only; mesh
    connectivity remains untouched until this evidence passes the truth audit.
    """
    started = time.perf_counter()
    family_merge_dot = math.cos(math.radians(max(8.0, feature_angle_degrees * 0.55)))
    feature_dot = math.cos(math.radians(feature_angle_degrees))
    feature_points: list[np.ndarray] = []
    baseline_points: list[np.ndarray] = []
    ranks: list[int] = []
    feature_cells: list[tuple[int, int, int]] = []
    certificates: list[dict[str, Any]] = []
    residual_ratios: list[float] = []
    certificate_frame_counts: list[int] = []
    certificate_view_counts: list[int] = []
    certificate_sample_counts: list[int] = []
    certificate_rank_ratios: list[float] = []
    certificate_cell_margins: list[float] = []
    audit: dict[str, Any] = {
        "source": "TSDF_zero_crossing_plus_trilinear_gradient",
        "scannedCells": 0,
        "multiHypothesisCells": 0,
        "hermiteSamples": 0,
        "featureCandidates": 0,
        "creaseRank2": 0,
        "cornerRank3": 0,
        "insufficientFamilies": 0,
        "parallelFamiliesRejected": 0,
        "outsideCellRejected": 0,
        "residualRejected": 0,
        "certificateCandidates": 0,
        "certificateFrameRejected": 0,
        "certificateViewRejected": 0,
        "certificateSampleRejected": 0,
        "certificateRankRejected": 0,
        "certificateCellMarginRejected": 0,
        "certificateFamilyBalanceRejected": 0,
        "certificateDisplacementRejected": 0,
        "certificatePassed": 0,
        "certificateThresholds": {
            "minFramesPerFamily": int(certificate_min_frames_per_family),
            "minViewsPerFamily": int(certificate_min_views_per_family),
            "minSamplesPerFamily": int(certificate_min_samples_per_family),
            "minRankRatio": float(certificate_min_rank_ratio),
            "minCellMarginRatio": float(certificate_min_cell_margin_ratio),
            "minFamilyWeightRatio": float(certificate_min_family_weight_ratio),
            "minQefDisplacementRatio": float(
                certificate_min_qef_displacement_ratio
            ),
        },
    }

    for cell in sorted(grid.candidates):
        audit["scannedCells"] += 1
        hypotheses: list[CellHypothesis] = []
        for direction in range(6):
            hypothesis = cell_hypothesis(grid, direction, cell, minimum_weight)
            if hypothesis is None:
                continue
            if float(np.dot(hypothesis.normal, DIRECTION_VECTORS[direction])) < valid_gradient_dot:
                continue
            hypotheses.append(hypothesis)
        if len(hypotheses) < 2:
            continue
        audit["multiHypothesisCells"] += 1

        samples: list[dict[str, Any]] = []
        base = np.asarray(cell, dtype=np.float64)
        cell_origin = base * grid.voxel
        for hypothesis in hypotheses:
            for corner_a, corner_b in CUBE_EDGES:
                if (
                    hypothesis.weights[corner_a] < minimum_weight
                    or hypothesis.weights[corner_b] < minimum_weight
                ):
                    continue
                value_a = float(hypothesis.values[corner_a])
                value_b = float(hypothesis.values[corner_b])
                if (value_a < 0.0) == (value_b < 0.0):
                    continue
                denominator = value_a - value_b
                t = min(1.0, max(0.0, value_a / denominator if abs(denominator) > 1e-9 else 0.5))
                local = (
                    CORNER_OFFSETS[corner_a].astype(np.float64)
                    + (
                        CORNER_OFFSETS[corner_b].astype(np.float64)
                        - CORNER_OFFSETS[corner_a].astype(np.float64)
                    )
                    * t
                )
                point = cell_origin + local * grid.voxel
                normal = trilinear_gradient(hypothesis.values, local, grid.voxel)
                if np.linalg.norm(normal) <= 1e-8:
                    continue
                normal = normalize(normal)
                if float(np.dot(normal, hypothesis.normal)) < 0.0:
                    normal = -normal
                key_a = tuple(
                    (
                        np.asarray(cell, dtype=np.int64)
                        + CORNER_OFFSETS[corner_a]
                    ).tolist()
                )
                key_b = tuple(
                    (
                        np.asarray(cell, dtype=np.int64)
                        + CORNER_OFFSETS[corner_b]
                    ).tolist()
                )
                frame_mask_a, view_mask_a = grid.read_evidence(
                    hypothesis.direction, key_a
                )
                frame_mask_b, view_mask_b = grid.read_evidence(
                    hypothesis.direction, key_b
                )
                samples.append(
                    {
                        "point": point,
                        "normal": normal,
                        "weight": float(
                            min(hypothesis.weights[corner_a], hypothesis.weights[corner_b])
                        ),
                        # A physical zero crossing is temporally/view supported
                        # only when both endpoint voxels carried that evidence.
                        "frameMask": frame_mask_a & frame_mask_b,
                        "viewMask": view_mask_a & view_mask_b,
                    }
                )
        audit["hermiteSamples"] += len(samples)
        if len(samples) < 4:
            audit["insufficientFamilies"] += 1
            continue

        families: list[dict[str, Any]] = []
        for sample in sorted(samples, key=lambda item: float(item["weight"]), reverse=True):
            best_index = -1
            best_dot = -1.0
            best_sign = 1.0
            for family_index, family in enumerate(families):
                family_normal = normalize(family["normalSum"])
                normal_dot = float(np.dot(sample["normal"], family_normal))
                if abs(normal_dot) > best_dot:
                    best_dot = abs(normal_dot)
                    best_sign = 1.0 if normal_dot >= 0.0 else -1.0
                    best_index = family_index
            if best_index >= 0 and best_dot >= family_merge_dot:
                family = families[best_index]
                family["normalSum"] += sample["normal"] * best_sign * sample["weight"]
                family["pointSum"] += sample["point"] * sample["weight"]
                family["weight"] += sample["weight"]
                family["samples"] += 1
                family["frameMask"] |= int(sample["frameMask"])
                family["viewMask"] |= int(sample["viewMask"])
                family["evidenceSamples"].append(
                    {
                        "frameMask": int(sample["frameMask"]),
                        "viewMask": int(sample["viewMask"]),
                    }
                )
            else:
                families.append(
                    {
                        "normalSum": sample["normal"] * sample["weight"],
                        "pointSum": sample["point"] * sample["weight"],
                        "weight": sample["weight"],
                        "samples": 1,
                        "frameMask": int(sample["frameMask"]),
                        "viewMask": int(sample["viewMask"]),
                        "evidenceSamples": [
                            {
                                "frameMask": int(sample["frameMask"]),
                                "viewMask": int(sample["viewMask"]),
                            }
                        ],
                    }
                )

        total_weight = sum(float(family["weight"]) for family in families)
        families = [
            family
            for family in families
            if family["samples"] >= 2
            and float(family["weight"]) >= total_weight * minimum_family_support_ratio
        ]
        families.sort(key=lambda family: float(family["weight"]), reverse=True)
        selected: list[dict[str, Any]] = []
        for family in families:
            family_normal = normalize(family["normalSum"])
            if not selected or all(
                abs(float(np.dot(family_normal, normalize(previous["normalSum"])))) < feature_dot
                for previous in selected
            ):
                selected.append(family)
            if len(selected) == 3:
                break
        if len(selected) < 2:
            if len(families) >= 2:
                audit["parallelFamiliesRejected"] += 1
            else:
                audit["insufficientFamilies"] += 1
            continue
        audit["featureCandidates"] += 1
        audit["certificateCandidates"] += 1

        required_frames = max(0, certificate_min_frames_per_family)
        required_views = max(0, certificate_min_views_per_family)
        required_samples = max(0, certificate_min_samples_per_family)
        persistent_by_family: list[list[dict[str, int]]] = []
        frame_stable_by_family: list[list[dict[str, int]]] = []
        for family in selected:
            evidence_samples = list(family["evidenceSamples"])
            frame_stable = [
                evidence
                for evidence in evidence_samples
                if int(evidence["frameMask"]).bit_count() >= required_frames
            ]
            persistent = [
                evidence
                for evidence in frame_stable
                if int(evidence["viewMask"]).bit_count() >= required_views
            ]
            frame_stable_by_family.append(frame_stable)
            persistent_by_family.append(persistent)

        # A feature family must be carried by the same physical Hermite edge
        # over time and viewpoints.  OR-ing masks from unrelated edges lets
        # transient Quest noise manufacture a certificate, while requiring
        # three distinct cube edges suppresses valid grid-aligned creases.
        if any(not family_samples for family_samples in frame_stable_by_family):
            audit["certificateFrameRejected"] += 1
            continue
        if any(not family_samples for family_samples in persistent_by_family):
            audit["certificateViewRejected"] += 1
            continue
        minimum_selected_samples = min(
            len(family_samples) for family_samples in persistent_by_family
        )
        if minimum_selected_samples < required_samples:
            audit["certificateSampleRejected"] += 1
            continue
        minimum_selected_frames = min(
            max(
                int(evidence["frameMask"]).bit_count()
                for evidence in family_samples
            )
            for family_samples in persistent_by_family
        )
        minimum_selected_views = min(
            max(
                int(evidence["viewMask"]).bit_count()
                for evidence in family_samples
            )
            for family_samples in persistent_by_family
        )

        plane_normals = np.asarray(
            [normalize(family["normalSum"]) for family in selected],
            dtype=np.float64,
        )
        plane_points = np.asarray(
            [family["pointSum"] / max(1e-12, float(family["weight"])) for family in selected],
            dtype=np.float64,
        )
        plane_weights = np.asarray([float(family["weight"]) for family in selected], dtype=np.float64)
        plane_weights /= max(1e-12, float(np.max(plane_weights)))
        minimum_family_weight_ratio = float(np.min(plane_weights))
        if minimum_family_weight_ratio < max(
            0.0, certificate_min_family_weight_ratio
        ):
            audit["certificateFamilyBalanceRejected"] += 1
            continue
        baseline = np.average(plane_points, axis=0, weights=plane_weights)
        weighted_a = plane_normals * np.sqrt(plane_weights[:, None])
        weighted_b = (
            np.sum(plane_normals * (plane_points - baseline), axis=1)
            * np.sqrt(plane_weights)
        )
        u_matrix, singular_values, vt = np.linalg.svd(weighted_a, full_matrices=False)
        if len(singular_values) == 0 or singular_values[0] <= 1e-8:
            audit["parallelFamiliesRejected"] += 1
            continue
        retained = singular_values >= singular_values[0] * rank_ratio
        rank = int(np.sum(retained))
        if rank < 2:
            audit["parallelFamiliesRejected"] += 1
            continue
        retained_rank_ratio = float(
            singular_values[min(rank - 1, len(singular_values) - 1)]
            / singular_values[0]
        )
        if retained_rank_ratio < max(0.0, certificate_min_rank_ratio):
            audit["certificateRankRejected"] += 1
            continue
        coefficients = np.zeros_like(singular_values)
        coefficients[retained] = (
            (u_matrix.T @ weighted_b)[retained] / singular_values[retained]
        )
        proposal = baseline + vt.T @ coefficients
        qef_displacement_ratio = float(
            np.linalg.norm(proposal - baseline) / max(1e-12, grid.voxel)
        )
        if qef_displacement_ratio < max(
            0.0, certificate_min_qef_displacement_ratio
        ):
            audit["certificateDisplacementRejected"] += 1
            continue
        cell_min = cell_origin - grid.voxel * 0.25
        cell_max = cell_origin + grid.voxel * 1.25
        if not np.all(np.isfinite(proposal)) or np.any(proposal < cell_min) or np.any(proposal > cell_max):
            audit["outsideCellRejected"] += 1
            continue
        cell_margin_ratio = float(
            np.min(
                np.concatenate(
                    (
                        proposal - cell_origin,
                        cell_origin + grid.voxel - proposal,
                    )
                )
            )
            / max(1e-12, grid.voxel)
        )
        if cell_margin_ratio < max(0.0, certificate_min_cell_margin_ratio):
            audit["certificateCellMarginRejected"] += 1
            continue
        before = float(np.sum(plane_weights * np.square(
            np.sum(plane_normals * (baseline - plane_points), axis=1)
        )))
        after = float(np.sum(plane_weights * np.square(
            np.sum(plane_normals * (proposal - plane_points), axis=1)
        )))
        if before > 1e-12 and after >= before * 0.98:
            audit["residualRejected"] += 1
            continue
        pair_dots = [
            abs(float(np.dot(plane_normals[left], plane_normals[right])))
            for left in range(len(plane_normals))
            for right in range(left + 1, len(plane_normals))
        ]
        minimum_normal_separation_degrees = (
            math.degrees(
                math.acos(min(1.0, max(-1.0, max(pair_dots))))
            )
            if pair_dots
            else 0.0
        )
        residual_ratio = after / max(1e-12, before)
        feature_points.append(proposal)
        baseline_points.append(baseline)
        ranks.append(rank)
        feature_cells.append(cell)
        certificates.append(
            {
                "minimumFramesPerFamily": minimum_selected_frames,
                "minimumViewsPerFamily": minimum_selected_views,
                "minimumSamplesPerFamily": minimum_selected_samples,
                "retainedRankRatio": retained_rank_ratio,
                "cellMarginRatio": cell_margin_ratio,
                "qefDisplacementRatio": qef_displacement_ratio,
                "qefResidualRatio": residual_ratio,
                "minimumNormalSeparationDegrees": minimum_normal_separation_degrees,
                "minimumFamilyWeightRatio": minimum_family_weight_ratio,
                "persistentSamplesPerFamily": [
                    int(len(family_samples))
                    for family_samples in persistent_by_family
                ],
                "rawSamplesPerFamily": [
                    int(family["samples"]) for family in selected
                ],
                "rank": rank,
            }
        )
        residual_ratios.append(residual_ratio)
        certificate_frame_counts.append(minimum_selected_frames)
        certificate_view_counts.append(minimum_selected_views)
        certificate_sample_counts.append(minimum_selected_samples)
        certificate_rank_ratios.append(retained_rank_ratio)
        certificate_cell_margins.append(cell_margin_ratio)
        audit["certificatePassed"] += 1
        if rank >= 3:
            audit["cornerRank3"] += 1
        else:
            audit["creaseRank2"] += 1

    audit["acceptedFeaturePoints"] = len(feature_points)
    audit["elapsedMs"] = (time.perf_counter() - started) * 1000.0
    audit["qefResidualRatioP95"] = (
        float(np.percentile(np.asarray(residual_ratios), 95)) if residual_ratios else 1.0
    )
    for name, values in (
        ("certificateMinFrames", certificate_frame_counts),
        ("certificateMinViews", certificate_view_counts),
        ("certificateMinSamples", certificate_sample_counts),
        ("certificateRankRatio", certificate_rank_ratios),
        ("certificateCellMarginRatio", certificate_cell_margins),
    ):
        audit[f"{name}P50"] = (
            float(np.percentile(np.asarray(values, dtype=np.float64), 50))
            if values
            else 0.0
        )
        audit[f"{name}P95"] = (
            float(np.percentile(np.asarray(values, dtype=np.float64), 95))
            if values
            else 0.0
        )
    return {
        "points": np.asarray(feature_points, dtype=np.float64).reshape((-1, 3)),
        "baselines": np.asarray(baseline_points, dtype=np.float64).reshape((-1, 3)),
        "ranks": np.asarray(ranks, dtype=np.int64),
        "cells": feature_cells,
        "certificates": certificates,
        "audit": audit,
    }


def extract_paper_dmc_tsdf_hermite_qef_feature_mesh(
    source: MeshBuild,
    ownership: dict[str, Any],
    paper_cells: dict[tuple[int, int, int], PaperDmcCell],
    features: dict[str, Any],
    voxel: float,
) -> tuple[MeshBuild, dict[str, Any]]:
    """Insert proven Hermite/QEF feature points without changing cell boundaries.

    This is the strict Extended-MC-style bridge experiment.  The paper DMC
    physical-edge ledger and every inter-cell boundary segment remain
    authoritative.  A feature point may replace only the *interior*
    triangulation of one closed, connected DMC patch.  Multi-patch, open,
    degenerate, or orientation-incompatible cells keep the source triangles.

    Ground truth is intentionally absent from this function.  It is used only
    by the caller's post-hoc A/B evaluation.
    """

    started = time.perf_counter()
    build = MeshBuild(
        vertices=[vertex.copy() for vertex in source.vertices],
        triangles=[],
        triangle_directions=[],
        triangle_support=[],
        audit=copy.deepcopy(source.audit),
        elapsed_ms=source.elapsed_ms,
        metadata=copy.deepcopy(source.metadata),
    )
    cell_triangles: dict[tuple[int, int, int], list[int]] = ownership.get(
        "cellTriangles", {}
    )
    audit: dict[str, Any] = {
        "source": (
            "paper DMC shared-edge topology + TSDF zero-crossing/trilinear-"
            "gradient Hermite QEF"
        ),
        "policy": (
            "freeze all inter-cell DMC boundary segments; replace only one "
            "closed connected cell-interior patch with a feature fan"
        ),
        "featureCandidates": int(len(features.get("cells", []))),
        "missingSourceCell": 0,
        "missingSourceTriangles": 0,
        "multiPatchRejected": 0,
        "openOrBranchBoundaryRejected": 0,
        "outsideCellRejected": 0,
        "degenerateFanRejected": 0,
        "orientationRejected": 0,
        "appliedCells": 0,
        "appliedSingleSurfaceCells": 0,
        "appliedDoubleSurfaceCells": 0,
        "sourceTrianglesReplaced": 0,
        "featureTrianglesAdded": 0,
        "boundarySegmentsBefore": 0,
        "boundarySegmentsAfter": 0,
        "boundarySignatureMismatchCells": 0,
        "minimumReferenceNormalDot": 1.0,
    }
    if not source.vertices or not source.triangles:
        audit["elapsedMs"] = (time.perf_counter() - started) * 1000.0
        build.metadata["paperHermiteQefFeature"] = audit
        return build, {
            "points": np.empty((0, 3), dtype=np.float64),
            "baselines": np.empty((0, 3), dtype=np.float64),
            "ranks": np.empty((0,), dtype=np.int64),
            "cells": [],
            "audit": audit,
        }

    source_vertices = np.asarray(source.vertices, dtype=np.float64)
    source_triangles = np.asarray(source.triangles, dtype=np.int64)
    replacements: dict[
        tuple[int, int, int],
        dict[str, Any],
    ] = {}
    replaced_triangle_indices: set[int] = set()
    applied_points: list[np.ndarray] = []
    applied_baselines: list[np.ndarray] = []
    applied_ranks: list[int] = []
    applied_cells: list[tuple[int, int, int]] = []
    applied_certificates: list[dict[str, Any]] = []

    for feature_index, raw_cell in enumerate(features.get("cells", [])):
        cell = tuple(int(value) for value in raw_cell)
        if cell not in paper_cells:
            audit["missingSourceCell"] += 1
            continue
        local_indices = sorted(
            {
                int(index)
                for index in cell_triangles.get(cell, [])
                if 0 <= int(index) < len(source.triangles)
            }
        )
        if not local_indices:
            audit["missingSourceTriangles"] += 1
            continue

        # The local source patch must be a single edge-connected disk.  A QEF
        # point must never bridge two independent DMC surfaces in one cell.
        edge_faces: dict[tuple[int, int], list[int]] = {}
        triangle_edges: dict[int, list[tuple[int, int]]] = {}
        for triangle_index in local_indices:
            triangle = source.triangles[triangle_index]
            edges: list[tuple[int, int]] = []
            for left, right in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ):
                edge = (
                    (int(left), int(right))
                    if int(left) < int(right)
                    else (int(right), int(left))
                )
                edges.append(edge)
                edge_faces.setdefault(edge, []).append(triangle_index)
            triangle_edges[triangle_index] = edges

        triangle_neighbors: dict[int, set[int]] = {
            index: set() for index in local_indices
        }
        for users in edge_faces.values():
            if len(users) != 2:
                continue
            left, right = users
            triangle_neighbors[left].add(right)
            triangle_neighbors[right].add(left)
        remaining = set(local_indices)
        components: list[set[int]] = []
        while remaining:
            seed = min(remaining)
            stack = [seed]
            component: set[int] = set()
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                remaining.discard(current)
                stack.extend(triangle_neighbors[current] - component)
            components.append(component)
        if len(components) != 1:
            audit["multiPatchRejected"] += 1
            continue

        boundary_edges = sorted(
            edge for edge, users in edge_faces.items() if len(users) == 1
        )
        boundary_neighbors: dict[int, set[int]] = {}
        for left, right in boundary_edges:
            boundary_neighbors.setdefault(left, set()).add(right)
            boundary_neighbors.setdefault(right, set()).add(left)
        if (
            len(boundary_edges) < 3
            or any(len(neighbors) != 2 for neighbors in boundary_neighbors.values())
        ):
            audit["openOrBranchBoundaryRejected"] += 1
            continue
        boundary_seen: set[int] = set()
        boundary_stack = [min(boundary_neighbors)]
        while boundary_stack:
            current = boundary_stack.pop()
            if current in boundary_seen:
                continue
            boundary_seen.add(current)
            boundary_stack.extend(boundary_neighbors[current] - boundary_seen)
        if len(boundary_seen) != len(boundary_neighbors):
            audit["openOrBranchBoundaryRejected"] += 1
            continue

        proposal = np.asarray(features["points"][feature_index], dtype=np.float64)
        cell_origin = np.asarray(cell, dtype=np.float64) * voxel
        tolerance = voxel * 0.02
        if (
            not np.all(np.isfinite(proposal))
            or np.any(proposal < cell_origin - tolerance)
            or np.any(proposal > cell_origin + voxel + tolerance)
        ):
            audit["outsideCellRejected"] += 1
            continue
        proposal = np.clip(proposal, cell_origin, cell_origin + voxel)

        staged_triangles: list[tuple[int, int, int]] = []
        staged_directions: list[int] = []
        staged_support: list[float] = []
        staged_reference_dots: list[float] = []
        candidate_vertex = len(build.vertices)
        candidate_valid = True
        for edge in boundary_edges:
            reference_index = edge_faces[edge][0]
            reference_triangle = source_triangles[reference_index]
            reference_points = source_vertices[reference_triangle]
            reference_cross = np.cross(
                reference_points[1] - reference_points[0],
                reference_points[2] - reference_points[0],
            )
            reference_length = float(np.linalg.norm(reference_cross))
            if reference_length <= 1e-12:
                candidate_valid = False
                audit["degenerateFanRejected"] += 1
                break
            reference_normal = reference_cross / reference_length

            left, right = edge
            fan_cross = np.cross(
                source_vertices[right] - source_vertices[left],
                proposal - source_vertices[left],
            )
            fan_length = float(np.linalg.norm(fan_cross))
            if fan_length <= voxel * voxel * 1e-6:
                candidate_valid = False
                audit["degenerateFanRejected"] += 1
                break
            fan_normal = fan_cross / fan_length
            reference_dot = float(np.dot(fan_normal, reference_normal))
            if reference_dot < 0.0:
                left, right = right, left
                fan_normal = -fan_normal
                reference_dot = -reference_dot
            if reference_dot < 0.05:
                candidate_valid = False
                audit["orientationRejected"] += 1
                break
            staged_triangles.append((left, right, candidate_vertex))
            staged_directions.append(source.triangle_directions[reference_index])
            staged_support.append(source.triangle_support[reference_index])
            staged_reference_dots.append(reference_dot)
        if not candidate_valid:
            continue

        staged_boundary = {
            (min(triangle[0], triangle[1]), max(triangle[0], triangle[1]))
            for triangle in staged_triangles
        }
        source_boundary = set(boundary_edges)
        if staged_boundary != source_boundary:
            audit["boundarySignatureMismatchCells"] += 1
            continue

        build.vertices.append(proposal.copy())
        replacements[cell] = {
            "sourceIndices": local_indices,
            "triangles": staged_triangles,
            "directions": staged_directions,
            "support": staged_support,
        }
        replaced_triangle_indices.update(local_indices)
        applied_points.append(proposal.copy())
        applied_baselines.append(
            np.asarray(features["baselines"][feature_index], dtype=np.float64).copy()
        )
        applied_ranks.append(int(features["ranks"][feature_index]))
        applied_cells.append(cell)
        source_certificates = features.get("certificates", [])
        applied_certificates.append(
            copy.deepcopy(source_certificates[feature_index])
            if feature_index < len(source_certificates)
            else {}
        )
        audit["appliedCells"] += 1
        if paper_cells[cell].surface_count == 2:
            audit["appliedDoubleSurfaceCells"] += 1
        else:
            audit["appliedSingleSurfaceCells"] += 1
        audit["sourceTrianglesReplaced"] += len(local_indices)
        audit["featureTrianglesAdded"] += len(staged_triangles)
        audit["boundarySegmentsBefore"] += len(boundary_edges)
        audit["boundarySegmentsAfter"] += len(staged_boundary)
        audit["minimumReferenceNormalDot"] = min(
            float(audit["minimumReferenceNormalDot"]),
            min(staged_reference_dots),
        )

    for triangle_index, triangle in enumerate(source.triangles):
        if triangle_index in replaced_triangle_indices:
            continue
        build.triangles.append(triangle)
        build.triangle_directions.append(source.triangle_directions[triangle_index])
        build.triangle_support.append(source.triangle_support[triangle_index])
    for cell in sorted(replacements):
        replacement = replacements[cell]
        build.triangles.extend(replacement["triangles"])
        build.triangle_directions.extend(replacement["directions"])
        build.triangle_support.extend(replacement["support"])

    placement_ms = (time.perf_counter() - started) * 1000.0
    build.elapsed_ms += placement_ms
    audit["elapsedMs"] = placement_ms
    audit["interCellBoundaryLedgerPreserved"] = (
        audit["boundarySignatureMismatchCells"] == 0
        and audit["boundarySegmentsBefore"] == audit["boundarySegmentsAfter"]
    )
    build.metadata["paperHermiteQefFeature"] = audit
    return build, {
        "points": np.asarray(applied_points, dtype=np.float64).reshape((-1, 3)),
        "baselines": np.asarray(applied_baselines, dtype=np.float64).reshape((-1, 3)),
        "ranks": np.asarray(applied_ranks, dtype=np.int64),
        "cells": applied_cells,
        "certificates": applied_certificates,
        "audit": audit,
    }


def extract_tsdf_hermite_dual_mesh(
    grid: DirectionalGrid,
    features: dict[str, Any],
    minimum_weight: float,
    valid_gradient_dot: float,
    parallel_dot: float,
    edge_merge_ratio: float,
    rank_ratio: float,
) -> MeshBuild:
    """Extract an independent multi-layer dual-cell shadow mesh.

    Each active cell owns one conservative vertex per compatible surface
    cluster.  Cells whose TSDF Hermite evidence passed the sharp-feature gate
    may share one rank-2/3 QEF vertex.  Connectivity is created only around a
    measured sign-changing primal edge with three or four adjacent decisions.
    """
    started = time.perf_counter()
    build = MeshBuild()
    feature_by_cell = {
        tuple(cell): (features["points"][index], int(features["ranks"][index]))
        for index, cell in enumerate(features["cells"])
    }
    decisions: list[dict[str, Any]] = []
    edge_entries: dict[
        tuple[tuple[int, int, int], tuple[int, int, int]],
        list[dict[str, Any]],
    ] = {}
    audit: dict[str, Any] = {
        "source": "Hermite_dual_cell_shared_primal_edge",
        "activeCells": 0,
        "cellDecisions": 0,
        "featureCellsMerged": 0,
        "featureMergeOverflowFallback": 0,
        "qefOutsideCellClamped": 0,
        "measuredPrimalEdges": 0,
        "connectedEdgeGroups": 0,
        "insufficientAdjacentCells": 0,
        "edgeGroupOverflow": 0,
        "trianglesBeforeTopologyAudit": 0,
    }

    def hypothesis_samples(
        hypotheses: list[CellHypothesis],
        cell: tuple[int, int, int],
    ) -> list[dict[str, Any]]:
        base_int = np.asarray(cell, dtype=np.int64)
        cell_origin = base_int.astype(np.float64) * grid.voxel
        samples: list[dict[str, Any]] = []
        for hypothesis in hypotheses:
            for corner_a, corner_b in CUBE_EDGES:
                if (
                    hypothesis.weights[corner_a] < minimum_weight
                    or hypothesis.weights[corner_b] < minimum_weight
                ):
                    continue
                value_a = float(hypothesis.values[corner_a])
                value_b = float(hypothesis.values[corner_b])
                if (value_a < 0.0) == (value_b < 0.0):
                    continue
                denominator = value_a - value_b
                t = min(1.0, max(0.0, value_a / denominator if abs(denominator) > 1e-9 else 0.5))
                offset_a = CORNER_OFFSETS[corner_a].astype(np.float64)
                offset_b = CORNER_OFFSETS[corner_b].astype(np.float64)
                local = offset_a + (offset_b - offset_a) * t
                point = cell_origin + local * grid.voxel
                normal = trilinear_gradient(hypothesis.values, local, grid.voxel)
                if np.linalg.norm(normal) <= 1e-8:
                    continue
                normal = normalize(normal)
                if float(np.dot(normal, hypothesis.normal)) < 0.0:
                    normal = -normal
                key_a = tuple((base_int + CORNER_OFFSETS[corner_a]).tolist())
                key_b = tuple((base_int + CORNER_OFFSETS[corner_b]).tolist())
                if key_b < key_a:
                    key_a, key_b = key_b, key_a
                    t = 1.0 - t
                samples.append(
                    {
                        "edge": (key_a, key_b),
                        "t": t,
                        "point": point,
                        "normal": normal,
                        "weight": float(
                            min(hypothesis.weights[corner_a], hypothesis.weights[corner_b])
                        ),
                    }
                )
        return samples

    def qef_vertex(
        samples: list[dict[str, Any]],
        cell: tuple[int, int, int],
    ) -> np.ndarray:
        points = np.asarray([sample["point"] for sample in samples], dtype=np.float64)
        normals = np.asarray([sample["normal"] for sample in samples], dtype=np.float64)
        weights = np.asarray([sample["weight"] for sample in samples], dtype=np.float64)
        weights /= max(1e-12, float(np.max(weights)))
        center = np.average(points, axis=0, weights=weights)
        weighted_a = normals * np.sqrt(weights[:, None])
        weighted_b = (
            np.sum(normals * (points - center), axis=1)
            * np.sqrt(weights)
        )
        u_matrix, singular_values, vt = np.linalg.svd(weighted_a, full_matrices=False)
        coefficients = np.zeros_like(singular_values)
        if len(singular_values) and singular_values[0] > 1e-8:
            retained = singular_values >= singular_values[0] * rank_ratio
            coefficients[retained] = (
                (u_matrix.T @ weighted_b)[retained] / singular_values[retained]
            )
        proposal = center + vt.T @ coefficients
        cell_origin = np.asarray(cell, dtype=np.float64) * grid.voxel
        cell_min = cell_origin - grid.voxel * 0.10
        cell_max = cell_origin + grid.voxel * 1.10
        clamped = np.minimum(cell_max, np.maximum(cell_min, proposal))
        if float(np.linalg.norm(clamped - proposal)) > 1e-8:
            audit["qefOutsideCellClamped"] += 1
        return clamped

    for cell in sorted(grid.candidates):
        valid_hypotheses: list[CellHypothesis] = []
        for direction in range(6):
            hypothesis = cell_hypothesis(grid, direction, cell, minimum_weight)
            if hypothesis is None:
                continue
            if float(np.dot(hypothesis.normal, DIRECTION_VECTORS[direction])) < valid_gradient_dot:
                continue
            valid_hypotheses.append(hypothesis)
        if not valid_hypotheses:
            continue
        audit["activeCells"] += 1
        clusters = cluster_hypotheses(
            valid_hypotheses,
            parallel_dot,
            grid.voxel,
            build.audit,
        )

        decision_hypotheses: list[list[CellHypothesis]] = []
        feature = feature_by_cell.get(cell)
        if feature is not None:
            merged_samples = hypothesis_samples(valid_hypotheses, cell)
            per_edge: dict[
                tuple[tuple[int, int, int], tuple[int, int, int]],
                list[float],
            ] = {}
            for sample in merged_samples:
                per_edge.setdefault(sample["edge"], []).append(float(sample["t"]))
            has_double_crossing = any(
                max(values) - min(values) > edge_merge_ratio
                for values in per_edge.values()
                if len(values) > 1
            )
            if has_double_crossing:
                audit["featureMergeOverflowFallback"] += 1
            else:
                decision_hypotheses = [valid_hypotheses]
                audit["featureCellsMerged"] += 1
        if not decision_hypotheses:
            for cluster in clusters:
                combined = combine_cluster(cluster, minimum_weight)
                if combined is not None:
                    decision_hypotheses.append([combined])

        for hypotheses in decision_hypotheses:
            samples = hypothesis_samples(hypotheses, cell)
            if len(samples) < 3:
                continue
            use_feature = feature is not None and len(decision_hypotheses) == 1
            vertex = np.asarray(feature[0], dtype=np.float64) if use_feature else qef_vertex(samples, cell)
            vertex_index = len(build.vertices)
            build.vertices.append(vertex)
            support = float(sum(sample["weight"] for sample in samples))
            normal_sum = sum(
                (sample["normal"] * sample["weight"] for sample in samples),
                np.zeros(3, dtype=np.float64),
            )
            decision_normal = (
                normalize(normal_sum)
                if np.linalg.norm(normal_sum) > 1e-8
                else hypotheses[0].normal
            )
            decision_index = len(decisions)
            decisions.append(
                {
                    "vertex": vertex_index,
                    "cell": cell,
                    "normal": decision_normal,
                    "support": support,
                }
            )

            consolidated: dict[
                tuple[tuple[int, int, int], tuple[int, int, int]],
                list[dict[str, Any]],
            ] = {}
            for sample in sorted(samples, key=lambda item: float(item["t"])):
                groups = consolidated.setdefault(sample["edge"], [])
                match = next(
                    (
                        group
                        for group in groups
                        if abs(float(group["t"]) - float(sample["t"])) <= edge_merge_ratio
                    ),
                    None,
                )
                if match is None:
                    groups.append(
                        {
                            "t": float(sample["t"]),
                            "normalSum": sample["normal"] * sample["weight"],
                            "weight": float(sample["weight"]),
                        }
                    )
                else:
                    total = float(match["weight"]) + float(sample["weight"])
                    match["t"] = (
                        float(match["t"]) * float(match["weight"])
                        + float(sample["t"]) * float(sample["weight"])
                    ) / max(1e-12, total)
                    match["normalSum"] += sample["normal"] * sample["weight"]
                    match["weight"] = total
            for edge, groups in consolidated.items():
                for group in groups:
                    edge_entries.setdefault(edge, []).append(
                        {
                            "decision": decision_index,
                            "t": float(group["t"]),
                            "normal": normalize(group["normalSum"]),
                            "support": float(group["weight"]),
                        }
                    )

    audit["cellDecisions"] = len(decisions)
    audit["measuredPrimalEdges"] = len(edge_entries)

    def add_triangle(indices: tuple[int, int, int], outward: np.ndarray, support: float) -> None:
        a, b, c = indices
        if len({a, b, c}) < 3:
            build.audit.degenerate_triangles_dropped += 1
            return
        cross = np.cross(build.vertices[b] - build.vertices[a], build.vertices[c] - build.vertices[a])
        if float(np.dot(cross, cross)) <= 1e-14:
            build.audit.degenerate_triangles_dropped += 1
            return
        if float(np.dot(cross, outward)) < 0.0:
            b, c = c, b
        build.triangles.append((a, b, c))
        build.triangle_directions.append(0)
        build.triangle_support.append(support)

    for edge, entries in edge_entries.items():
        groups: list[list[dict[str, Any]]] = []
        for entry in sorted(entries, key=lambda item: float(item["t"])):
            match = next(
                (
                    group
                    for group in groups
                    if abs(
                        float(np.average(
                            [member["t"] for member in group],
                            weights=[member["support"] for member in group],
                        ))
                        - float(entry["t"])
                    )
                    <= edge_merge_ratio
                ),
                None,
            )
            if match is None:
                groups.append([entry])
            else:
                match.append(entry)
        for group in groups:
            by_cell: dict[tuple[int, int, int], dict[str, Any]] = {}
            for entry in group:
                decision = decisions[int(entry["decision"])]
                previous = by_cell.get(decision["cell"])
                if previous is None or float(entry["support"]) > float(previous["support"]):
                    by_cell[decision["cell"]] = entry
            group = list(by_cell.values())
            if len(group) < 3:
                audit["insufficientAdjacentCells"] += 1
                continue
            if len(group) > 4:
                audit["edgeGroupOverflow"] += len(group) - 4
                group.sort(key=lambda item: float(item["support"]), reverse=True)
                group = group[:4]

            edge_a = np.asarray(edge[0], dtype=np.float64) * grid.voxel
            edge_b = np.asarray(edge[1], dtype=np.float64) * grid.voxel
            edge_axis = normalize(edge_b - edge_a)
            reference = (
                np.asarray([0.0, 0.0, 1.0])
                if abs(float(edge_axis[2])) < 0.9
                else np.asarray([0.0, 1.0, 0.0])
            )
            basis_u = normalize(np.cross(edge_axis, reference))
            basis_v = normalize(np.cross(edge_axis, basis_u))
            edge_midpoint = (edge_a + edge_b) * 0.5
            ordered = sorted(
                group,
                key=lambda item: math.atan2(
                    float(np.dot(
                        (np.asarray(decisions[int(item["decision"])]["cell"], dtype=np.float64) + 0.5)
                        * grid.voxel
                        - edge_midpoint,
                        basis_v,
                    )),
                    float(np.dot(
                        (np.asarray(decisions[int(item["decision"])]["cell"], dtype=np.float64) + 0.5)
                        * grid.voxel
                        - edge_midpoint,
                        basis_u,
                    )),
        ),
    )
            indices = [int(decisions[int(item["decision"])]["vertex"]) for item in ordered]
            normal_sum = sum(
                (item["normal"] * item["support"] for item in ordered),
                np.zeros(3, dtype=np.float64),
            )
            outward = normalize(normal_sum) if np.linalg.norm(normal_sum) > 1e-8 else edge_axis
            support = float(sum(item["support"] for item in ordered))
            if len(indices) == 3:
                add_triangle((indices[0], indices[1], indices[2]), outward, support)
            else:
                first_diagonal = float(np.linalg.norm(
                    build.vertices[indices[0]] - build.vertices[indices[2]]
                ))
                second_diagonal = float(np.linalg.norm(
                    build.vertices[indices[1]] - build.vertices[indices[3]]
                ))
                if first_diagonal <= second_diagonal:
                    add_triangle((indices[0], indices[1], indices[2]), outward, support)
                    add_triangle((indices[0], indices[2], indices[3]), outward, support)
                else:
                    add_triangle((indices[0], indices[1], indices[3]), outward, support)
                    add_triangle((indices[1], indices[2], indices[3]), outward, support)
            audit["connectedEdgeGroups"] += 1

    audit["trianglesBeforeTopologyAudit"] = len(build.triangles)
    deduplicate_triangles(build)
    enforce_edge_manifold(build)
    audit["trianglesAfterTopologyAudit"] = len(build.triangles)
    build.metadata["hermiteDualMesh"] = audit
    build.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return build


def extract_tsdf_hermite_ledger_dual_mesh(
    grid: DirectionalGrid,
    features: dict[str, Any],
    minimum_weight: float,
    valid_gradient_dot: float,
    parallel_dot: float,
    edge_merge_ratio: float,
    rank_ratio: float,
) -> MeshBuild:
    """Dual contouring with a persistent shared-edge decision as authority."""
    started = time.perf_counter()
    build = MeshBuild()
    feature_by_cell = {
        tuple(cell): np.asarray(features["points"][index], dtype=np.float64)
        for index, cell in enumerate(features["cells"])
    }
    raw_edges: dict[
        tuple[tuple[int, int, int], tuple[int, int, int]],
        list[dict[str, Any]],
    ] = {}
    audit: dict[str, Any] = {
        "source": "persistent_DmcEdgeDecision_then_cell_consumers",
        "rawEdgeObservations": 0,
        "sharedEdgeDecisions": 0,
        "secondCrossingDecisions": 0,
        "edgeCrossingOverflowRejected": 0,
        "mc33Hypotheses": 0,
        "mc33TopologyDeferredHypotheses": 0,
        "crossDirectionCornersCompleted": 0,
        "crossDirectionCornerConflicts": 0,
        "sharedEdgeOrientationConflicts": 0,
        "sharedEdgeCornersCompleted": 0,
        "sharedEdgeCornerConflicts": 0,
        "mc33ComponentFailures": 0,
        "mc33Components": 0,
        "observedCellEdgeOwners": 0,
        "fabricatedAdjacentCellOwners": 0,
        "localComponentMerges": 0,
        "featureTopologyUnions": 0,
        "incidentCandidateCells": 0,
        "cellDecisions": 0,
        "featureCellDecisions": 0,
        "singleEdgeCellDecisions": 0,
        "qefOutsideCellClamped": 0,
        "connectedEdgeGroups": 0,
        "insufficientAdjacentCells": 0,
        "trianglesBeforeTopologyAudit": 0,
    }
    cell_hypotheses: dict[tuple[int, int, int], list[CellHypothesis]] = {}

    for cell in sorted(grid.candidates):
        base = np.asarray(cell, dtype=np.int64)
        cell_origin = base.astype(np.float64) * grid.voxel
        for direction in range(6):
            hypothesis = cell_hypothesis(grid, direction, cell, minimum_weight)
            if hypothesis is None:
                continue
            if float(np.dot(hypothesis.normal, DIRECTION_VECTORS[direction])) < valid_gradient_dot:
                continue
            cell_hypotheses.setdefault(cell, []).append(hypothesis)
            for corner_a, corner_b in CUBE_EDGES:
                if (
                    hypothesis.weights[corner_a] < minimum_weight
                    or hypothesis.weights[corner_b] < minimum_weight
                ):
                    continue
                value_a = float(hypothesis.values[corner_a])
                value_b = float(hypothesis.values[corner_b])
                if (value_a < 0.0) == (value_b < 0.0):
                    continue
                denominator = value_a - value_b
                t = min(1.0, max(0.0, value_a / denominator if abs(denominator) > 1e-9 else 0.5))
                offset_a = CORNER_OFFSETS[corner_a].astype(np.float64)
                offset_b = CORNER_OFFSETS[corner_b].astype(np.float64)
                local = offset_a + (offset_b - offset_a) * t
                point = cell_origin + local * grid.voxel
                normal = trilinear_gradient(hypothesis.values, local, grid.voxel)
                if np.linalg.norm(normal) <= 1e-8:
                    continue
                normal = normalize(normal)
                if float(np.dot(normal, hypothesis.normal)) < 0.0:
                    normal = -normal
                key_a = tuple((base + CORNER_OFFSETS[corner_a]).tolist())
                key_b = tuple((base + CORNER_OFFSETS[corner_b]).tolist())
                sign_a = value_a < 0.0
                if key_b < key_a:
                    key_a, key_b = key_b, key_a
                    t = 1.0 - t
                    sign_a = not sign_a
                raw_edges.setdefault((key_a, key_b), []).append(
                    {
                        "t": t,
                        "point": point,
                        "normal": normal,
                        "signA": sign_a,
                        "weight": float(
                            min(hypothesis.weights[corner_a], hypothesis.weights[corner_b])
                        ),
                    }
                )
                audit["rawEdgeObservations"] += 1

    ledger: list[dict[str, Any]] = []
    ledger_by_edge: dict[
        tuple[tuple[int, int, int], tuple[int, int, int]],
        list[int],
    ] = {}
    for edge, observations in raw_edges.items():
        groups: list[list[dict[str, Any]]] = []
        for observation in sorted(observations, key=lambda item: float(item["t"])):
            match = next(
                (
                    group
                    for group in groups
                    if abs(
                        float(np.average(
                            [item["t"] for item in group],
                            weights=[item["weight"] for item in group],
                        ))
                        - float(observation["t"])
                    )
                    <= edge_merge_ratio
                ),
                None,
            )
            if match is None:
                groups.append([observation])
            else:
                match.append(observation)
        if len(groups) > 2:
            groups.sort(
                key=lambda group: sum(float(item["weight"]) for item in group),
                reverse=True,
            )
            audit["edgeCrossingOverflowRejected"] += len(groups) - 2
            groups = groups[:2]
            groups.sort(key=lambda group: float(np.mean([item["t"] for item in group])))
        for surface_index, group in enumerate(groups):
            weights = np.asarray([item["weight"] for item in group], dtype=np.float64)
            t = float(np.average([item["t"] for item in group], weights=weights))
            point = np.average(
                np.asarray([item["point"] for item in group], dtype=np.float64),
                axis=0,
                weights=weights,
            )
            reference_normal = group[0]["normal"]
            normal_sum = np.zeros(3, dtype=np.float64)
            aligned_observations: list[dict[str, Any]] = []
            aligned_signs: list[bool] = []
            for item in group:
                normal = item["normal"].copy()
                sign_a = bool(item["signA"])
                if float(np.dot(normal, reference_normal)) < 0.0:
                    normal = -normal
                    sign_a = not sign_a
                normal_sum += normal * item["weight"]
                aligned_signs.append(sign_a)
                aligned_observations.append(
                    {
                        "point": item["point"],
                        "normal": normal,
                        "signA": sign_a,
                        "weight": item["weight"],
                    }
                )
            canonical_sign_a: bool | None
            if len(set(aligned_signs)) == 1:
                canonical_sign_a = aligned_signs[0]
            else:
                canonical_sign_a = None
                audit["sharedEdgeOrientationConflicts"] += 1
            ledger_index = len(ledger)
            ledger.append(
                {
                    "edge": edge,
                    "surfaceIndex": surface_index,
                    "t": t,
                    "point": point,
                    "normal": normalize(normal_sum),
                    "signA": canonical_sign_a,
                    "weight": float(np.sum(weights)),
                    "observations": aligned_observations,
                }
            )
            ledger_by_edge.setdefault(edge, []).append(ledger_index)
            audit["sharedEdgeDecisions"] += 1
            if surface_index == 1:
                audit["secondCrossingDecisions"] += 1

    def transition_mask(index: int) -> int:
        mask = 0
        for edge_index, (corner_a, corner_b) in enumerate(CUBE_EDGES):
            if bool(index & (1 << corner_a)) != bool(index & (1 << corner_b)):
                mask |= 1 << edge_index
        return mask

    # A shared edge ledger owns only the crossing position.  Cell ownership
    # must be proven by that cell's own sign field; geometrically assigning a
    # crossing to all four adjacent cells fabricates quads and was the source
    # of the previous apparent "coverage" gain, high p95 error, and cracks.
    incident: dict[tuple[int, int, int], list[int]] = {}
    local_groups_by_cell: dict[tuple[int, int, int], list[set[int]]] = {}

    for cell, hypotheses in cell_hypotheses.items():
        base = np.asarray(cell, dtype=np.int64)
        local_groups: list[set[int]] = []
        for hypothesis in hypotheses:
            topology_values = hypothesis.values.copy()
            topology_weights = hypothesis.weights.copy()
            # First consult the persistent shared-edge ledger.  A crossing
            # whose oriented endpoint signs and normal agree with this surface
            # is direct topological evidence shared by every incident cell.
            for corner in range(8):
                if topology_weights[corner] >= minimum_weight:
                    continue
                inferred: list[tuple[bool, float]] = []
                for edge_corner_a, edge_corner_b in CUBE_EDGES:
                    if corner not in (edge_corner_a, edge_corner_b):
                        continue
                    other_corner = (
                        edge_corner_b if corner == edge_corner_a else edge_corner_a
                    )
                    key_corner = tuple((base + CORNER_OFFSETS[corner]).tolist())
                    key_other = tuple((base + CORNER_OFFSETS[other_corner]).tolist())
                    key_a, key_b = (
                        (key_corner, key_other)
                        if key_corner < key_other
                        else (key_other, key_corner)
                    )
                    candidates = ledger_by_edge.get((key_a, key_b), [])
                    for ledger_index in candidates:
                        decision = ledger[ledger_index]
                        if decision["signA"] is None:
                            continue
                        normal_dot = float(np.dot(
                            decision["normal"],
                            hypothesis.normal,
                        ))
                        if abs(normal_dot) < parallel_dot:
                            continue
                        sign_a = bool(decision["signA"])
                        if normal_dot < 0.0:
                            sign_a = not sign_a
                        corner_sign = sign_a if key_corner == key_a else not sign_a
                        inferred.append((corner_sign, float(decision["weight"])))
                if not inferred:
                    continue
                signs = {sign for sign, _ in inferred}
                if len(signs) != 1:
                    audit["sharedEdgeCornerConflicts"] += 1
                    continue
                inside = inferred[0][0]
                topology_values[corner] = -1.0 if inside else 1.0
                topology_weights[corner] = sum(weight for _, weight in inferred)
                audit["sharedEdgeCornersCompleted"] += 1

            # Directional TSDF channels are separate surface observations, but
            # an otherwise missing corner may be completed from another
            # channel when that channel faces the same fitted surface and every
            # qualifying measurement agrees on the sign.  This is measured
            # cross-direction evidence, not a synthetic free-space default.
            for corner in range(8):
                if topology_weights[corner] >= minimum_weight:
                    continue
                key = tuple((base + CORNER_OFFSETS[corner]).tolist())
                candidates: list[tuple[float, float, float]] = []
                for other_direction in range(6):
                    if other_direction == hypothesis.direction:
                        continue
                    compliance = float(np.dot(
                        hypothesis.normal,
                        DIRECTION_VECTORS[other_direction],
                    ))
                    if compliance < valid_gradient_dot:
                        continue
                    value, weight = grid.read(other_direction, key)
                    if weight < minimum_weight:
                        continue
                    candidates.append((
                        float(value),
                        float(weight * compliance),
                        float(weight),
                    ))
                if not candidates:
                    continue
                signs = {value < 0.0 for value, _, _ in candidates}
                if len(signs) != 1:
                    audit["crossDirectionCornerConflicts"] += 1
                    continue
                effective_weight = sum(weight for _, weight, _ in candidates)
                measured_weight = sum(weight for _, _, weight in candidates)
                if effective_weight <= 1e-8:
                    continue
                topology_values[corner] = sum(
                    value * weight for value, weight, _ in candidates
                ) / effective_weight
                # Direction compliance controls the value average, but must
                # not erase whether the source corner was actually measured.
                # The paper reference likewise applies compliance to support
                # voting after the per-corner validity check.
                topology_weights[corner] = measured_weight
                audit["crossDirectionCornersCompleted"] += 1

            known = topology_weights >= minimum_weight
            known_mask = 0
            known_index = 0
            for corner in range(8):
                if not bool(known[corner]):
                    continue
                known_mask |= 1 << corner
                if float(topology_values[corner]) < 0.0:
                    known_index |= 1 << corner
            unknown_mask = 0xFF & ~known_mask
            # Missing is a third state.  Accept it only when both possible
            # assignments produce the same crossing-edge ownership.  This is
            # the same conservative invariant used by the Unity shadow path;
            # it prevents an unobserved corner from masquerading as free space.
            if (
                unknown_mask
                and transition_mask(known_index)
                != transition_mask(known_index | unknown_mask)
            ):
                audit["mc33TopologyDeferredHypotheses"] += 1
                continue

            active_edges: list[bool] = []
            edge_ledger: list[int | None] = []
            for corner_a, corner_b in CUBE_EDGES:
                active = (
                    topology_weights[corner_a] >= minimum_weight
                    and topology_weights[corner_b] >= minimum_weight
                    and (
                        (float(topology_values[corner_a]) < 0.0)
                        != (float(topology_values[corner_b]) < 0.0)
                    )
                )
                active_edges.append(active)
                if not active:
                    edge_ledger.append(None)
                    continue
                value_a = float(topology_values[corner_a])
                value_b = float(topology_values[corner_b])
                denominator = value_a - value_b
                t = min(1.0, max(0.0, value_a / denominator if abs(denominator) > 1e-9 else 0.5))
                key_a = tuple((base + CORNER_OFFSETS[corner_a]).tolist())
                key_b = tuple((base + CORNER_OFFSETS[corner_b]).tolist())
                if key_b < key_a:
                    key_a, key_b = key_b, key_a
                    t = 1.0 - t
                candidates = ledger_by_edge.get((key_a, key_b), [])
                edge_ledger.append(
                    min(
                        candidates,
                        key=lambda index: abs(float(ledger[index]["t"]) - t),
                    )
                    if candidates
                    else None
                )
            audit["mc33Hypotheses"] += 1
            components = mc33_components(topology_values, active_edges)
            if components is None:
                audit["mc33ComponentFailures"] += 1
                continue
            grouped: dict[int, list[int]] = {}
            for edge_index, component in enumerate(components):
                ledger_index = edge_ledger[edge_index]
                if component < 0 or ledger_index is None:
                    continue
                grouped.setdefault(component, []).append(ledger_index)
            audit["mc33Components"] += len(grouped)
            for group in grouped.values():
                group_set = set(group)
                if group_set:
                    local_groups.append(group_set)
                    audit["observedCellEdgeOwners"] += len(group_set)

        # Merge only overlapping MC33 components inside this cell.  Never
        # propagate a local topology decision through a scene-wide DSU.
        merged = True
        while merged:
            merged = False
            for first in range(len(local_groups)):
                for second in range(first + 1, len(local_groups)):
                    if local_groups[first].isdisjoint(local_groups[second]):
                        continue
                    local_groups[first].update(local_groups[second])
                    del local_groups[second]
                    audit["localComponentMerges"] += 1
                    merged = True
                    break
                if merged:
                    break
        if local_groups:
            local_groups_by_cell[cell] = local_groups
            owned = sorted(set().union(*local_groups))
            incident[cell] = owned
            audit["incidentCandidateCells"] += len(owned)

    cell_decisions: list[dict[str, Any]] = []
    ledger_consumers: dict[int, list[int]] = {}

    def solve_cell_qef(
        cell: tuple[int, int, int],
        ledger_indices: list[int],
        feature_point: np.ndarray | None,
    ) -> np.ndarray:
        if feature_point is not None:
            return feature_point.copy()
        # The shared-edge record is the authority.  Feeding every raw
        # observation back into the cell QEF lets noise on one physical
        # crossing masquerade as several independent planes and falsely raises
        # the QEF rank.  Consume exactly one consolidated Hermite plane per
        # DmcEdgeDecision.
        observations = [
            {
                "point": ledger[ledger_index]["point"],
                "normal": ledger[ledger_index]["normal"],
                "weight": ledger[ledger_index]["weight"],
            }
            for ledger_index in ledger_indices
        ]
        points = np.asarray([item["point"] for item in observations], dtype=np.float64)
        normals = np.asarray([item["normal"] for item in observations], dtype=np.float64)
        weights = np.asarray([item["weight"] for item in observations], dtype=np.float64)
        weights /= max(1e-12, float(np.max(weights)))
        # A rank-1 QEF leaves two tangent dimensions unconstrained.  Anchoring
        # those null-space dimensions at the shared edge crossing collapses
        # all four incident cells onto one point and degenerates the dual quad.
        # Use the owning cell center as the deterministic null-space anchor;
        # measured Hermite planes still determine every constrained dimension.
        center = (np.asarray(cell, dtype=np.float64) + 0.5) * grid.voxel
        weighted_a = normals * np.sqrt(weights[:, None])
        weighted_b = np.sum(normals * (points - center), axis=1) * np.sqrt(weights)
        u_matrix, singular_values, vt = np.linalg.svd(weighted_a, full_matrices=False)
        coefficients = np.zeros_like(singular_values)
        if len(singular_values) and singular_values[0] > 1e-8:
            retained = singular_values >= singular_values[0] * rank_ratio
            coefficients[retained] = (
                (u_matrix.T @ weighted_b)[retained] / singular_values[retained]
            )
        proposal = center + vt.T @ coefficients
        origin = np.asarray(cell, dtype=np.float64) * grid.voxel
        lower = origin - grid.voxel * 0.10
        upper = origin + grid.voxel * 1.10
        clamped = np.minimum(upper, np.maximum(lower, proposal))
        if float(np.linalg.norm(clamped - proposal)) > 1e-8:
            audit["qefOutsideCellClamped"] += 1
        return clamped

    for cell, groups_as_sets in sorted(local_groups_by_cell.items()):
        feature_point = feature_by_cell.get(cell)
        groups = [sorted(group) for group in groups_as_sets]

        # A proven Hermite crease/corner is the only admissible local
        # cross-component union.  Its rank-2/3 QEF point must agree with every
        # participating consolidated plane.
        if feature_point is not None and len(groups) > 1:
            matching_groups = []
            for group_index, group in enumerate(groups):
                if any(
                    float(np.linalg.norm(ledger[index]["point"] - feature_point))
                    <= grid.voxel * 1.8
                    and abs(float(np.dot(
                        ledger[index]["normal"],
                        feature_point - ledger[index]["point"],
                    )))
                    <= grid.voxel * 0.45
                    for index in group
                ):
                    matching_groups.append(group_index)
            if len(matching_groups) >= 2:
                combined_group = sorted(set().union(
                    *(set(groups[index]) for index in matching_groups)
                ))
                first = matching_groups[0]
                groups[first] = combined_group
                for index in reversed(matching_groups[1:]):
                    del groups[index]
                audit["featureTopologyUnions"] += len(matching_groups) - 1

        feature_group_index = -1
        if feature_point is not None:
            scores = [
                float(np.mean([
                    np.linalg.norm(ledger[index]["point"] - feature_point)
                    for index in group
                ]))
                for group in groups
            ]
            if scores:
                candidate = int(np.argmin(np.asarray(scores)))
                if len(groups[candidate]) >= 2 and scores[candidate] <= grid.voxel * 1.8:
                    feature_group_index = candidate
                    audit["featureCellDecisions"] += 1

        for group_index, group in enumerate(groups):
            group_feature = (
                feature_point
                if feature_point is not None and group_index == feature_group_index
                else None
            )
            vertex = solve_cell_qef(cell, group, group_feature)
            vertex_index = len(build.vertices)
            build.vertices.append(vertex)
            support = float(sum(ledger[index]["weight"] for index in group))
            normal_sum = sum(
                (ledger[index]["normal"] * ledger[index]["weight"] for index in group),
                np.zeros(3, dtype=np.float64),
            )
            decision_index = len(cell_decisions)
            cell_decisions.append(
                {
                    "cell": cell,
                    "vertex": vertex_index,
                    "normal": normalize(normal_sum),
                    "support": support,
                    "ledger": group,
                }
            )
            for ledger_index in group:
                ledger_consumers.setdefault(ledger_index, []).append(decision_index)
            audit["cellDecisions"] += 1
            if len(group) == 1:
                audit["singleEdgeCellDecisions"] += 1

    def add_triangle(indices: tuple[int, int, int], outward: np.ndarray, support: float) -> None:
        a, b, c = indices
        if len({a, b, c}) < 3:
            build.audit.degenerate_triangles_dropped += 1
            return
        cross = np.cross(build.vertices[b] - build.vertices[a], build.vertices[c] - build.vertices[a])
        if float(np.dot(cross, cross)) <= 1e-14:
            build.audit.degenerate_triangles_dropped += 1
            return
        if float(np.dot(cross, outward)) < 0.0:
            b, c = c, b
        build.triangles.append((a, b, c))
        build.triangle_directions.append(0)
        build.triangle_support.append(support)

    for ledger_index, edge_decision in enumerate(ledger):
        consumer_indices = ledger_consumers.get(ledger_index, [])
        by_cell: dict[tuple[int, int, int], int] = {}
        for decision_index in consumer_indices:
            decision = cell_decisions[decision_index]
            previous = by_cell.get(decision["cell"])
            if (
                previous is None
                or decision["support"] > cell_decisions[previous]["support"]
            ):
                by_cell[decision["cell"]] = decision_index
        consumer_indices = list(by_cell.values())
        if len(consumer_indices) < 3:
            audit["insufficientAdjacentCells"] += 1
            continue
        if len(consumer_indices) > 4:
            consumer_indices.sort(
                key=lambda index: cell_decisions[index]["support"],
                reverse=True,
            )
            consumer_indices = consumer_indices[:4]

        edge_a = np.asarray(edge_decision["edge"][0], dtype=np.float64) * grid.voxel
        edge_b = np.asarray(edge_decision["edge"][1], dtype=np.float64) * grid.voxel
        axis = normalize(edge_b - edge_a)
        reference = (
            np.asarray([0.0, 0.0, 1.0])
            if abs(float(axis[2])) < 0.9
            else np.asarray([0.0, 1.0, 0.0])
        )
        basis_u = normalize(np.cross(axis, reference))
        basis_v = normalize(np.cross(axis, basis_u))
        midpoint = (edge_a + edge_b) * 0.5
        consumer_indices.sort(
            key=lambda index: math.atan2(
                float(np.dot(
                    (np.asarray(cell_decisions[index]["cell"], dtype=np.float64) + 0.5)
                    * grid.voxel
                    - midpoint,
                    basis_v,
                )),
                float(np.dot(
                    (np.asarray(cell_decisions[index]["cell"], dtype=np.float64) + 0.5)
                    * grid.voxel
                    - midpoint,
                    basis_u,
                )),
            )
        )
        vertices = [int(cell_decisions[index]["vertex"]) for index in consumer_indices]
        outward = edge_decision["normal"]
        support = float(sum(cell_decisions[index]["support"] for index in consumer_indices))
        if len(vertices) == 3:
            add_triangle((vertices[0], vertices[1], vertices[2]), outward, support)
        else:
            diagonal_a = float(np.linalg.norm(
                build.vertices[vertices[0]] - build.vertices[vertices[2]]
            ))
            diagonal_b = float(np.linalg.norm(
                build.vertices[vertices[1]] - build.vertices[vertices[3]]
            ))
            if diagonal_a <= diagonal_b:
                add_triangle((vertices[0], vertices[1], vertices[2]), outward, support)
                add_triangle((vertices[0], vertices[2], vertices[3]), outward, support)
            else:
                add_triangle((vertices[0], vertices[1], vertices[3]), outward, support)
                add_triangle((vertices[1], vertices[2], vertices[3]), outward, support)
        audit["connectedEdgeGroups"] += 1

    audit["trianglesBeforeTopologyAudit"] = len(build.triangles)
    deduplicate_triangles(build)
    enforce_edge_manifold(build)
    audit["trianglesAfterTopologyAudit"] = len(build.triangles)
    build.metadata["hermiteLedgerDualMesh"] = audit
    build.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return build


def to_open3d(build: MeshBuild) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    if build.vertices:
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(build.vertices, dtype=np.float64))
    if build.triangles:
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(build.triangles, dtype=np.int32))
        mesh.compute_vertex_normals()
    return mesh


def topology_metrics(build: MeshBuild) -> dict[str, Any]:
    edge_use: dict[tuple[int, int], int] = {}
    for triangle in build.triangles:
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge = (a, b) if a < b else (b, a)
            edge_use[edge] = edge_use.get(edge, 0) + 1
    boundary = sum(1 for use in edge_use.values() if use == 1)
    nonmanifold = sum(1 for use in edge_use.values() if use > 2)
    result = {
        "vertices": len(build.vertices),
        "triangles": len(build.triangles),
        "boundaryEdges": boundary,
        "nonManifoldEdges": nonmanifold,
        "boundaryEdgesPerKTriangles": boundary * 1000.0 / max(1, len(build.triangles)),
        "extractionMs": build.elapsed_ms,
        "audit": vars(build.audit),
    }
    if build.metadata:
        result["metadata"] = build.metadata
    return result


def progressive_checkpoint_order(
    cameras: list[dict[str, Any]],
    checkpoints: Iterable[int],
) -> list[dict[str, Any]]:
    """Return a nested, coverage-spread order for frame-count growth sweeps.

    The first checkpoint is distributed evenly over the complete master path.
    Later checkpoints append the still-unselected cameras that are farthest in
    master-path index from an existing sample.  Every smaller run is therefore
    an exact prefix of every larger run while retaining broad room coverage.
    """

    count = len(cameras)
    if count <= 1:
        return cameras
    targets = sorted({max(1, min(count, int(value))) for value in checkpoints})
    if not targets:
        return cameras

    first_count = targets[0]
    first_indices = np.linspace(0, count - 1, first_count).round().astype(np.int64)
    ordered_indices = [int(value) for value in first_indices]
    selected = set(ordered_indices)
    for target in targets[1:]:
        additions: list[int] = []
        while len(selected) + len(additions) < target:
            anchors = selected.union(additions)
            candidate = max(
                (index for index in range(count) if index not in anchors),
                key=lambda index: (min(abs(index - anchor) for anchor in anchors), -index),
            )
            additions.append(candidate)
        ordered_indices.extend(sorted(additions))
        selected.update(additions)
    ordered_indices.extend(index for index in range(count) if index not in selected)
    return [cameras[index] for index in ordered_indices]


def update_observed_truth_mask(
    scene: o3d.t.geometry.RaycastingScene,
    truth_points: np.ndarray,
    camera: dict[str, Any],
    valid_depth: np.ndarray,
    width: int,
    height: int,
    min_distance: float,
    max_distance: float,
    observed: np.ndarray,
    pixel_validity_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Mark truth samples genuinely observable through a valid depth pixel.

    A sample must be inside the camera frustum, map to a non-zero depth pixel,
    and be the first mesh surface along its exact camera ray.  The measured
    (possibly degraded) depth is used only as the sensor-validity gate; truth is
    never used to change fusion or topology.
    """

    remaining = np.flatnonzero(~observed)
    if len(remaining) == 0:
        return observed
    points = truth_points[remaining]
    pose = camera["pose"]
    camera_data = camera["camera"]
    origin = np.asarray(pose["position"], dtype=np.float64)
    right = normalize(np.asarray(pose["right"], dtype=np.float64))
    up = normalize(np.asarray(pose["up"], dtype=np.float64))
    forward = normalize(np.asarray(pose["forward"], dtype=np.float64))
    relative = points - origin
    ranges = np.linalg.norm(relative, axis=1)
    axial = relative @ forward
    tan_y = math.tan(math.radians(float(camera_data["fieldOfView"])) * 0.5)
    tan_x = tan_y * float(camera_data["aspect"])
    safe_axial = np.where(axial > 1e-9, axial, 1.0)
    ndc_x = (relative @ right) / (safe_axial * tan_x)
    ndc_y = (relative @ up) / (safe_axial * tan_y)
    pixel_x = np.floor((ndc_x + 1.0) * 0.5 * width).astype(np.int64)
    pixel_y = np.floor((1.0 - ndc_y) * 0.5 * height).astype(np.int64)
    in_view = (
        (axial > 0.0)
        & (ranges >= min_distance)
        & (ranges <= max_distance)
        & (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
    )
    candidate_local = np.flatnonzero(in_view)
    if len(candidate_local) == 0:
        return observed
    flat_depth = np.asarray(valid_depth).reshape(-1)
    pixel_validity = flat_depth > 0.0
    if pixel_validity_mask is not None:
        supplied_validity = np.asarray(pixel_validity_mask, dtype=bool).reshape(-1)
        if len(supplied_validity) != len(flat_depth):
            raise RuntimeError("truth-observation pixel gate does not match depth image")
        pixel_validity &= supplied_validity
    pixel_indices = pixel_y[candidate_local] * width + pixel_x[candidate_local]
    candidate_local = candidate_local[pixel_validity[pixel_indices]]
    if len(candidate_local) == 0:
        return observed

    candidate_ranges = ranges[candidate_local]
    directions = relative[candidate_local] / candidate_ranges[:, None]
    rays = np.concatenate(
        (
            np.repeat(origin[None, :], len(candidate_local), axis=0),
            directions,
        ),
        axis=1,
    ).astype(np.float32)
    first_hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(np.float64)
    tolerance = np.maximum(0.002, candidate_ranges * 0.001)
    unobstructed = np.isfinite(first_hit) & (np.abs(first_hit - candidate_ranges) <= tolerance)
    observed[remaining[candidate_local[unobstructed]]] = True
    return observed


def evaluate_mesh(
    build: MeshBuild,
    truth: o3d.geometry.PointCloud,
    truth_scene: o3d.t.geometry.RaycastingScene,
    observed_truth_mask: np.ndarray | None = None,
    structure_reference: dict[str, np.ndarray] | None = None,
    structure_band_meters: float = 0.08,
    coverage_mask_sink: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    mesh = to_open3d(build)
    result = topology_metrics(build)
    if len(mesh.triangles) == 0:
        if coverage_mask_sink is not None:
            coverage_mask_sink["coveredAt0.05m"] = np.zeros(len(truth.points), dtype=bool)
        result["coverageAt0.05m"] = 0.0
        result["wholeRoomCoverageAt0.05m"] = 0.0
        result["visibleCoverageAt0.05m"] = 0.0
        result["observedTruthSamples"] = int(np.sum(observed_truth_mask)) if observed_truth_mask is not None else 0
        result["observedTruthRatioOfRoom"] = (
            float(np.mean(observed_truth_mask)) if observed_truth_mask is not None and len(observed_truth_mask) else 0.0
        )
        result["extraSurfaceRatioAt0.05m"] = 1.0
        result["accuracyP95m"] = float("inf")
        return result
    sample_count = max(5000, min(50000, len(mesh.triangles) * 2))
    reconstructed = mesh.sample_points_uniformly(sample_count)
    geometry, covered_at_5cm = cloud_metrics(
        truth,
        truth_scene,
        reconstructed,
        structure_reference,
        structure_band_meters,
    )
    if coverage_mask_sink is not None:
        reconstruction_scene = o3d.t.geometry.RaycastingScene()
        reconstruction_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        exact_truth_distance = reconstruction_scene.compute_distance(
            o3d.core.Tensor(
                np.asarray(truth.points, dtype=np.float32),
                dtype=o3d.core.Dtype.Float32,
            )
        ).numpy().astype(np.float64)
        exact_covered_at_5cm = exact_truth_distance <= 0.05
        coverage_mask_sink["coveredAt0.05m"] = exact_covered_at_5cm
        result["exactWholeRoomCoverageAt0.05m"] = float(np.mean(exact_covered_at_5cm))
        if observed_truth_mask is not None and len(observed_truth_mask) == len(exact_covered_at_5cm):
            observed_count = int(np.sum(observed_truth_mask))
            result["exactVisibleCoverageAt0.05m"] = (
                float(np.mean(exact_covered_at_5cm[observed_truth_mask])) if observed_count else 0.0
            )
    result["coverageAt0.05m"] = geometry["coverageAtMeters"]["0.05"]
    result["wholeRoomCoverageAt0.05m"] = geometry["coverageAtMeters"]["0.05"]
    if observed_truth_mask is not None and len(observed_truth_mask) == len(covered_at_5cm):
        observed_count = int(np.sum(observed_truth_mask))
        result["visibleCoverageAt0.05m"] = (
            float(np.mean(covered_at_5cm[observed_truth_mask])) if observed_count else 0.0
        )
        result["observedTruthSamples"] = observed_count
        result["observedTruthRatioOfRoom"] = float(np.mean(observed_truth_mask)) if len(observed_truth_mask) else 0.0
    else:
        result["visibleCoverageAt0.05m"] = result["coverageAt0.05m"]
        result["observedTruthSamples"] = len(covered_at_5cm)
        result["observedTruthRatioOfRoom"] = 1.0
    result["extraSurfaceRatioAt0.05m"] = geometry["extraSurfaceRatioAtMeters"]["0.05"]
    result["accuracyP95m"] = geometry["reconstructionToTruthMeters"]["p95"]
    result["completenessP95m"] = geometry["truthToReconstructionMeters"]["p95"]
    result["structureBands"] = geometry.get("structureBands", {})
    if (
        structure_reference
        and observed_truth_mask is not None
        and len(observed_truth_mask) == len(covered_at_5cm)
    ):
        truth_points = np.asarray(truth.points, dtype=np.float64)
        structure_masks = build_truth_structure_masks(
            truth_points, structure_reference, structure_band_meters
        )
        result["visibleStructureBands"] = {}
        for label, structure_mask in structure_masks.items():
            visible_structure = np.asarray(observed_truth_mask, dtype=bool) & structure_mask
            visible_count = int(np.sum(visible_structure))
            result["visibleStructureBands"][label] = {
                "visibleTruthSamples": visible_count,
                "visibleRatioOfStructureTruth": (
                    float(visible_count / np.sum(structure_mask))
                    if np.sum(structure_mask) else 0.0
                ),
                "visibleCoverageAt0.05m": (
                    float(np.mean(covered_at_5cm[visible_structure]))
                    if visible_count else 0.0
                ),
            }
    if len(mesh.triangles):
        _, counts, _ = mesh.cluster_connected_triangles()
        counts_array = np.asarray(counts, dtype=np.int64)
        result["connectedComponents"] = int(len(counts_array))
        result["significantComponents50Triangles"] = int(np.sum(counts_array >= 50))
    return result


def build_truth_structure_masks(
    truth_points: np.ndarray,
    structure_reference: dict[str, np.ndarray],
    structure_band_meters: float,
) -> dict[str, np.ndarray]:
    """Build fixed truth-domain bands plus their smooth-interior complement."""
    masks: dict[str, np.ndarray] = {}
    union = np.zeros(len(truth_points), dtype=bool)
    threshold = max(0.01, structure_band_meters)
    for label, reference_points in structure_reference.items():
        mask = distance_to_reference(truth_points, reference_points) <= threshold
        masks[label] = mask
        union |= mask
    masks["smooth_interior"] = ~union
    return masks


def attribute_visible_surface_loss(
    observed_truth_mask: np.ndarray,
    stage_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Partition final visible-surface loss by the first failed DMC stage."""

    visible = np.asarray(observed_truth_mask, dtype=bool)
    raw = np.asarray(stage_masks["raw"], dtype=bool)
    filtered = np.asarray(stage_masks["filtered"], dtype=bool)
    voted = np.asarray(stage_masks["voted"], dtype=bool)
    final = np.asarray(stage_masks["dmc"], dtype=bool)
    if not all(len(mask) == len(visible) for mask in (raw, filtered, voted, final)):
        raise RuntimeError("DMC stage masks do not share the truth-sample domain")

    missing = visible & ~final
    categories = {
        "tsdfOrCornerAvailability": missing & ~raw,
        "intraDirectionFilter": missing & raw & ~filtered,
        "interDirectionVote": missing & raw & filtered & ~voted,
        "dmcExtraction": missing & raw & filtered & voted,
    }
    visible_count = int(np.sum(visible))
    missing_count = int(np.sum(missing))
    category_counts = {name: int(np.sum(mask)) for name, mask in categories.items()}
    attributed_count = int(sum(category_counts.values()))
    if attributed_count != missing_count:
        raise RuntimeError(
            f"visible loss attribution mismatch: attributed={attributed_count} missing={missing_count}"
        )

    def ratio(count: int, denominator: int) -> float:
        return float(count / denominator) if denominator else 0.0

    stage_coverage = {
        "rawTsdfCornerAvailability": ratio(int(np.sum(visible & raw)), visible_count),
        "postIntraDirectionFilter": ratio(int(np.sum(visible & filtered)), visible_count),
        "postInterDirectionVote": ratio(int(np.sum(visible & voted)), visible_count),
        "finalDmc": ratio(int(np.sum(visible & final)), visible_count),
    }
    net_stage_delta = {
        "tsdfOrCornerAvailabilityMissing": 1.0 - stage_coverage["rawTsdfCornerAvailability"],
        "intraDirectionFilterLoss": (
            stage_coverage["rawTsdfCornerAvailability"]
            - stage_coverage["postIntraDirectionFilter"]
        ),
        "interDirectionVoteLoss": (
            stage_coverage["postIntraDirectionFilter"]
            - stage_coverage["postInterDirectionVote"]
        ),
        "dmcExtractionLoss": (
            stage_coverage["postInterDirectionVote"]
            - stage_coverage["finalDmc"]
        ),
    }
    first_failure = {
        name: {
            "truthSamples": count,
            "ratioOfVisibleTruth": ratio(count, visible_count),
            "shareOfMissingVisible": ratio(count, missing_count),
        }
        for name, count in category_counts.items()
    }
    first_failure["directionFilteringTotal"] = {
        "truthSamples": category_counts["intraDirectionFilter"] + category_counts["interDirectionVote"],
        "ratioOfVisibleTruth": ratio(
            category_counts["intraDirectionFilter"] + category_counts["interDirectionVote"],
            visible_count,
        ),
        "shareOfMissingVisible": ratio(
            category_counts["intraDirectionFilter"] + category_counts["interDirectionVote"],
            missing_count,
        ),
    }
    return {
        "visibleTruthSamples": visible_count,
        "finalRecoveredVisibleSamples": int(np.sum(visible & final)),
        "finalMissingVisibleSamples": missing_count,
        "finalMissingVisibleRatio": ratio(missing_count, visible_count),
        "stageCoverageAt0.05m": stage_coverage,
        "netStageLossAt0.05m": net_stage_delta,
        "firstFailure": first_failure,
        "nonMonotonicCoverage": {
            "filteredWithoutRaw": int(np.sum(visible & filtered & ~raw)),
            "votedWithoutFiltered": int(np.sum(visible & voted & ~filtered)),
            "finalWithoutVoted": int(np.sum(visible & final & ~voted)),
        },
        "attributionSemantics": (
            "netStageLoss is the authoritative aggregate attribution; firstFailure is a "
            "per-truth geometric proxy and nonMonotonicCoverage records local stage rescues"
        ),
        "accountingClosed": attributed_count == missing_count,
    }


def points_near_written_voxels(
    points: np.ndarray,
    written_voxels: set[tuple[int, int, int]],
    voxel: float,
    threshold: float,
) -> np.ndarray:
    """Return exact point-to-written-voxel-centre proximity on the truth domain."""

    result = np.zeros(len(points), dtype=bool)
    if not written_voxels:
        return result
    radius = int(math.ceil(threshold / voxel))
    threshold_sq = threshold * threshold
    for index, point in enumerate(np.asarray(points, dtype=np.float64)):
        if not np.all(np.isfinite(point)):
            continue
        nearest = np.rint(point / voxel).astype(np.int64)
        found = False
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    key = (
                        int(nearest[0] + dx),
                        int(nearest[1] + dy),
                        int(nearest[2] + dz),
                    )
                    if key not in written_voxels:
                        continue
                    delta = voxel_center(key, voxel) - point
                    if float(np.dot(delta, delta)) <= threshold_sq:
                        result[index] = True
                        found = True
                        break
                if found:
                    break
            if found:
                break
    return result


def paper_complete_corner_cells(
    grid: DirectionalGrid,
    minimum_weight: float,
) -> set[tuple[int, int, int]]:
    """Cells with all eight measured corners in at least one paper direction."""

    complete: set[tuple[int, int, int]] = set()
    for cell in grid.candidates:
        base = np.asarray(cell, dtype=np.int64)
        for scan_direction in PAPER_DIRECTION_TO_SCANCOVER:
            if all(
                grid.read(
                    scan_direction,
                    tuple((base + offset).tolist()),
                )[1]
                >= minimum_weight
                for offset in PAPER_CORNER_OFFSETS
            ):
                complete.add(cell)
                break
    return complete


def points_near_cells(
    points: np.ndarray,
    cells: set[tuple[int, int, int]],
    voxel: float,
    threshold: float,
) -> np.ndarray:
    """Return exact point-to-axis-aligned-cell proximity on the truth domain."""

    result = np.zeros(len(points), dtype=bool)
    if not cells:
        return result
    radius = int(math.ceil(threshold / voxel)) + 1
    threshold_sq = threshold * threshold
    for index, point in enumerate(np.asarray(points, dtype=np.float64)):
        if not np.all(np.isfinite(point)):
            continue
        base = np.floor(point / voxel).astype(np.int64)
        found = False
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    cell = (
                        int(base[0] + dx),
                        int(base[1] + dy),
                        int(base[2] + dz),
                    )
                    if cell not in cells:
                        continue
                    lower = np.asarray(cell, dtype=np.float64) * voxel
                    upper = lower + voxel
                    outside = np.maximum(np.maximum(lower - point, point - upper), 0.0)
                    if float(np.dot(outside, outside)) <= threshold_sq:
                        result[index] = True
                        found = True
                        break
                if found:
                    break
            if found:
                break
    return result


def build_tsdf_supply_masks(
    grid: DirectionalGrid,
    truth_points: np.ndarray,
    minimum_weight: float,
    threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build read-only spatial evidence for the two middle TSDF supply gates."""

    written_voxels = {
        key
        for layer in grid.values
        for key, record in layer.items()
        if float(record[1]) > 1e-8
    }
    complete_cells = paper_complete_corner_cells(grid, minimum_weight)
    masks = {
        "projectiveVoxelTouched": points_near_written_voxels(
            truth_points,
            written_voxels,
            grid.voxel,
            threshold,
        ),
        "completeCornerWeightSupport": points_near_cells(
            truth_points,
            complete_cells,
            grid.voxel,
            threshold,
        ),
    }
    return masks, {
        "thresholdMeters": threshold,
        "writtenVoxelKeys": len(written_voxels),
        "candidateCells": len(grid.candidates),
        "completeCornerCells": len(complete_cells),
        "minimumCornerWeight": minimum_weight,
    }


def attribute_tsdf_supply_loss(
    ideal_visible_truth_mask: np.ndarray,
    observed_truth_mask: np.ndarray,
    usable_depth_normal_mask: np.ndarray,
    projective_touched_mask: np.ndarray,
    complete_corner_mask: np.ndarray,
    raw_zero_crossing_mask: np.ndarray,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Split raw TSDF loss into four mutually exclusive upstream gates.

    Direct spatial masks can exhibit a few local inversions because a downstream
    triangle or complete cell may sit inside the 5 cm truth band while its source
    centre lies just outside it.  Downstream evidence is therefore propagated
    upstream before attribution, and every direct inversion is reported.
    """

    ideal_visible = np.asarray(ideal_visible_truth_mask, dtype=bool)
    visible = np.asarray(observed_truth_mask, dtype=bool)
    direct_usable = np.asarray(usable_depth_normal_mask, dtype=bool)
    direct_touched = np.asarray(projective_touched_mask, dtype=bool)
    direct_complete = np.asarray(complete_corner_mask, dtype=bool)
    raw = np.asarray(raw_zero_crossing_mask, dtype=bool)
    masks = (ideal_visible, direct_usable, direct_touched, direct_complete, raw)
    if not all(len(mask) == len(visible) for mask in masks):
        raise RuntimeError("TSDF supply masks do not share the truth-sample domain")

    # A downstream observation proves its upstream prerequisites even when the
    # independently measured proximity bands differ at their exact boundary.
    complete = direct_complete | raw
    touched = direct_touched | complete
    usable = direct_usable | touched

    raw_missing = visible & ~raw
    categories = {
        "depthPixelOrUsableNormalSampleMissing": raw_missing & ~usable,
        "projectiveVoxelOrCornerUntouched": raw_missing & usable & ~touched,
        "insufficientCompleteCornerWeight": raw_missing & touched & ~complete,
        "supportedButNoUsableZeroCrossing": raw_missing & complete,
    }
    visible_count = int(np.sum(visible))
    missing_count = int(np.sum(raw_missing))
    category_counts = {name: int(np.sum(mask)) for name, mask in categories.items()}
    attributed_count = int(sum(category_counts.values()))
    if attributed_count != missing_count:
        raise RuntimeError(
            f"TSDF supply attribution mismatch: attributed={attributed_count} "
            f"missing={missing_count}"
        )

    def ratio(count: int, denominator: int) -> float:
        return float(count / denominator) if denominator else 0.0

    stage_coverage = {
        "usableDepthNormalSample": ratio(int(np.sum(visible & usable)), visible_count),
        "projectiveVoxelTouched": ratio(int(np.sum(visible & touched)), visible_count),
        "completeCornerWeightSupport": ratio(int(np.sum(visible & complete)), visible_count),
        "rawUsableZeroCrossing": ratio(int(np.sum(visible & raw)), visible_count),
    }
    first_failure = {
        name: {
            "truthSamples": count,
            "ratioOfVisibleTruth": ratio(count, visible_count),
            "shareOfRawTsdfMissing": ratio(count, missing_count),
        }
        for name, count in category_counts.items()
    }
    ideal_visible_count = int(np.sum(ideal_visible))
    observed_from_ideal = int(np.sum(ideal_visible & visible))
    sensor_missing = int(np.sum(ideal_visible & ~visible))
    return {
        "visibleTruthSamples": visible_count,
        "rawTsdfMissingVisibleSamples": missing_count,
        "rawTsdfMissingVisibleRatio": ratio(missing_count, visible_count),
        "stageCoverageAt0.05m": stage_coverage,
        "firstFailure": first_failure,
        "sensorValidityLedgerOutsideConditionalVisibleLoss": {
            "idealVisibleTruthSamples": ideal_visible_count,
            "questValidDepthObservedSamples": observed_from_ideal,
            "missingValidDepthSamples": sensor_missing,
            "missingValidDepthRatioOfIdealVisible": ratio(sensor_missing, ideal_visible_count),
            "note": (
                "The primary visible ledger already requires at least one valid depth pixel; "
                "pure sensor dropout is therefore reported separately and is not double-counted."
            ),
        },
        "directMaskNonMonotonicity": {
            "touchedWithoutUsableSample": int(np.sum(visible & direct_touched & ~direct_usable)),
            "completeWithoutNearbyTouchedCentre": int(np.sum(visible & direct_complete & ~direct_touched)),
            "rawCrossingWithoutNearbyCompleteCell": int(np.sum(visible & raw & ~direct_complete)),
        },
        "diagnostics": diagnostics,
        "attributionSemantics": (
            "first failure in a nested, downstream-evidence-propagated supply ledger; "
            "truth is evaluation-only and never changes integration or extraction"
        ),
        "accountingClosed": attributed_count == missing_count,
    }


def evaluate_feature_geometry_delta(
    source: MeshBuild,
    feature: MeshBuild,
    truth_scene: o3d.t.geometry.RaycastingScene,
) -> dict[str, Any]:
    """Audit whether a placement-only shadow improves the geometry it touched.

    Global sampled-mesh metrics can hide a bad local feature relocation because
    the moved vertices are a small fraction of a room mesh.  This audit measures
    exactly those vertices and the incident internal edges.  Ground truth is
    evaluation-only and never participates in the placement decision.
    """
    source_vertices = np.asarray(source.vertices, dtype=np.float64)
    feature_vertices = np.asarray(feature.vertices, dtype=np.float64)
    if (
        source_vertices.shape != feature_vertices.shape
        or len(source_vertices) == 0
        or source.triangles != feature.triangles
    ):
        return {
            "comparable": False,
            "reason": "placement shadow must preserve vertex count and triangle indices",
        }

    displacement = np.linalg.norm(feature_vertices - source_vertices, axis=1)
    moved = displacement > 1e-7
    moved_indices = np.flatnonzero(moved)
    if len(moved_indices) == 0:
        return {
            "comparable": True,
            "movedVertices": 0,
            "truthImprovedFraction": 0.0,
            "severeFoldoverRateBefore": 0.0,
            "severeFoldoverRateAfter": 0.0,
        }

    def truth_distance(points: np.ndarray) -> np.ndarray:
        tensor = o3d.core.Tensor(points.astype(np.float32), dtype=o3d.core.Dtype.Float32)
        return truth_scene.compute_distance(tensor).numpy().astype(np.float64)

    before_distance = truth_distance(source_vertices[moved])
    after_distance = truth_distance(feature_vertices[moved])
    distance_delta = after_distance - before_distance

    triangles = np.asarray(source.triangles, dtype=np.int64)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, triangle in enumerate(triangles):
        a, b, c = (int(triangle[0]), int(triangle[1]), int(triangle[2]))
        for edge_a, edge_b in ((a, b), (b, c), (c, a)):
            edge = (edge_a, edge_b) if edge_a < edge_b else (edge_b, edge_a)
            edge_faces.setdefault(edge, []).append(face_index)

    def face_normals(vertices: np.ndarray) -> np.ndarray:
        points = vertices[triangles]
        cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        length = np.linalg.norm(cross, axis=1)
        normals = np.zeros_like(cross)
        valid = length > 1e-12
        normals[valid] = cross[valid] / length[valid, None]
        return normals

    before_normals = face_normals(source_vertices)
    after_normals = face_normals(feature_vertices)
    before_angles: list[float] = []
    after_angles: list[float] = []
    for (a, b), faces in edge_faces.items():
        if len(faces) != 2 or not (moved[a] or moved[b]):
            continue
        first, second = faces
        before_dot = float(np.clip(np.dot(before_normals[first], before_normals[second]), -1.0, 1.0))
        after_dot = float(np.clip(np.dot(after_normals[first], after_normals[second]), -1.0, 1.0))
        before_angles.append(math.degrees(math.acos(before_dot)))
        after_angles.append(math.degrees(math.acos(after_dot)))

    before_angle_array = np.asarray(before_angles, dtype=np.float64)
    after_angle_array = np.asarray(after_angles, dtype=np.float64)
    severe_before = float(np.mean(before_angle_array > 120.0)) if len(before_angle_array) else 0.0
    severe_after = float(np.mean(after_angle_array > 120.0)) if len(after_angle_array) else 0.0
    result = {
        "comparable": True,
        "movedVertices": int(len(moved_indices)),
        "truthDistanceBeforeMeters": {
            "mean": float(np.mean(before_distance)),
            "p50": float(np.percentile(before_distance, 50)),
            "p95": float(np.percentile(before_distance, 95)),
            "max": float(np.max(before_distance)),
        },
        "truthDistanceAfterMeters": {
            "mean": float(np.mean(after_distance)),
            "p50": float(np.percentile(after_distance, 50)),
            "p95": float(np.percentile(after_distance, 95)),
            "max": float(np.max(after_distance)),
        },
        "truthDistanceDeltaMeters": {
            "mean": float(np.mean(distance_delta)),
            "p50": float(np.percentile(distance_delta, 50)),
            "p95": float(np.percentile(distance_delta, 95)),
        },
        "truthImprovedFraction": float(np.mean(distance_delta < 0.0)),
        "touchedInternalEdges": int(len(before_angle_array)),
        "dihedralDegreesBefore": {
            "p50": float(np.percentile(before_angle_array, 50)) if len(before_angle_array) else 0.0,
            "p95": float(np.percentile(before_angle_array, 95)) if len(before_angle_array) else 0.0,
        },
        "dihedralDegreesAfter": {
            "p50": float(np.percentile(after_angle_array, 50)) if len(after_angle_array) else 0.0,
            "p95": float(np.percentile(after_angle_array, 95)) if len(after_angle_array) else 0.0,
        },
        "severeFoldoverRateBefore": severe_before,
        "severeFoldoverRateAfter": severe_after,
        "severeFoldoverRateDelta": severe_after - severe_before,
    }
    return result


def reconstruct_depth_points(
    camera: dict[str, Any], depth: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rays, _ = make_rays(camera, width, height, np.zeros(3, dtype=np.float64))
    directions = rays[:, 3:].astype(np.float64)
    origins = rays[:, :3].astype(np.float64)
    forward = normalize(np.asarray(camera["pose"]["forward"], dtype=np.float64))
    axial = directions @ forward
    flat_depth = depth.reshape(-1).astype(np.float64)
    valid = (flat_depth > 0.0) & (axial > 1e-6)
    ranges = np.zeros_like(flat_depth)
    ranges[valid] = flat_depth[valid] / axial[valid]
    points = origins + directions * ranges[:, None]
    points[~valid] = np.nan
    return points, origins, flat_depth, valid


def reconstruct_points_and_normals(
    camera: dict[str, Any], depth: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Historical fixed-cross normal estimator retained exactly for arm A."""
    points, origins, _, valid = reconstruct_depth_points(camera, depth, width, height)
    image = points.reshape(height, width, 3)
    normals = np.full_like(image, np.nan)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            center = image[y, x]
            left, right = image[y, x - 1], image[y, x + 1]
            up, down = image[y - 1, x], image[y + 1, x]
            if not all(np.all(np.isfinite(value)) for value in (center, left, right, up, down)):
                continue
            dx = right - left
            dy = down - up
            normal = np.cross(dx, dy)
            length = float(np.linalg.norm(normal))
            if length <= 1e-8:
                continue
            normal /= length
            camera_position = origins[y * width + x]
            if float(np.dot(normal, camera_position - center)) < 0.0:
                normal = -normal
            normals[y, x] = normal
    return points, normals.reshape(-1, 3), valid


def reconstruct_points_and_paper_normals(
    camera: dict[str, Any],
    depth: np.ndarray,
    width: int,
    height: int,
    neighbor_radius: int,
    depth_change_factor: float,
    depth_change_floor: float,
    bilateral_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Validity-aware neighborhood normals followed by normal-only bilateral filtering.

    Raw depth is never smoothed.  Nearest valid samples are sought independently
    on both image axes and are rejected across depth discontinuities.  This is a
    deterministic, edge-preserving realization of the paper's neighborhood
    normal estimate plus bilateral normal filtering.
    """
    points, origins, flat_depth, valid = reconstruct_depth_points(
        camera, depth, width, height
    )
    point_image = points.reshape(height, width, 3)
    depth_image = flat_depth.reshape(height, width)
    raw = np.full_like(point_image, np.nan)
    neighbor_radius = max(1, int(neighbor_radius))
    bilateral_radius = max(0, int(bilateral_radius))
    depth_change_factor = max(0.0, float(depth_change_factor))
    depth_change_floor = max(0.0, float(depth_change_floor))
    one_sided_axes = 0
    discontinuity_rejects = 0

    def nearest_valid(
        x: int,
        y: int,
        axis_x: int,
        axis_y: int,
        sign: int,
        center_depth: float,
        threshold: float,
    ) -> np.ndarray | None:
        nonlocal discontinuity_rejects
        for offset in range(1, neighbor_radius + 1):
            nx = x + axis_x * sign * offset
            ny = y + axis_y * sign * offset
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                break
            candidate = point_image[ny, nx]
            candidate_depth = float(depth_image[ny, nx])
            if not np.all(np.isfinite(candidate)) or candidate_depth <= 0.0:
                continue
            if abs(candidate_depth - center_depth) > threshold:
                discontinuity_rejects += 1
                continue
            return candidate
        return None

    for y in range(height):
        for x in range(width):
            center = point_image[y, x]
            center_depth = float(depth_image[y, x])
            if not np.all(np.isfinite(center)) or center_depth <= 0.0:
                continue
            threshold = max(depth_change_floor, center_depth * depth_change_factor)
            left = nearest_valid(x, y, 1, 0, -1, center_depth, threshold)
            right = nearest_valid(x, y, 1, 0, 1, center_depth, threshold)
            up = nearest_valid(x, y, 0, 1, -1, center_depth, threshold)
            down = nearest_valid(x, y, 0, 1, 1, center_depth, threshold)

            if left is not None and right is not None:
                tangent_x = right - left
            elif right is not None:
                tangent_x = right - center
                one_sided_axes += 1
            elif left is not None:
                tangent_x = center - left
                one_sided_axes += 1
            else:
                continue
            if up is not None and down is not None:
                tangent_y = down - up
            elif down is not None:
                tangent_y = down - center
                one_sided_axes += 1
            elif up is not None:
                tangent_y = center - up
                one_sided_axes += 1
            else:
                continue

            normal = np.cross(tangent_x, tangent_y)
            length = float(np.linalg.norm(normal))
            if length <= 1e-8:
                continue
            normal /= length
            camera_position = origins[y * width + x]
            if float(np.dot(normal, camera_position - center)) < 0.0:
                normal = -normal
            raw[y, x] = normal

    filtered = raw.copy()
    if bilateral_radius > 0:
        spatial_sigma = max(1.0, bilateral_radius * 0.75)
        normal_sigma = 0.25
        for y in range(height):
            for x in range(width):
                base = raw[y, x]
                center_depth = float(depth_image[y, x])
                if not np.all(np.isfinite(base)) or center_depth <= 0.0:
                    continue
                threshold = max(depth_change_floor, center_depth * depth_change_factor)
                accumulated = np.zeros(3, dtype=np.float64)
                total_weight = 0.0
                for oy in range(-bilateral_radius, bilateral_radius + 1):
                    ny = y + oy
                    if ny < 0 or ny >= height:
                        continue
                    for ox in range(-bilateral_radius, bilateral_radius + 1):
                        nx = x + ox
                        if nx < 0 or nx >= width:
                            continue
                        candidate = raw[ny, nx]
                        candidate_depth = float(depth_image[ny, nx])
                        if not np.all(np.isfinite(candidate)) or candidate_depth <= 0.0:
                            continue
                        depth_delta = abs(candidate_depth - center_depth)
                        if depth_delta > threshold:
                            continue
                        orientation = max(0.0, float(np.dot(base, candidate)))
                        if orientation <= 0.0:
                            continue
                        spatial_weight = math.exp(
                            -(ox * ox + oy * oy) / (2.0 * spatial_sigma * spatial_sigma)
                        )
                        depth_weight = math.exp(
                            -(depth_delta * depth_delta) / (2.0 * threshold * threshold)
                        )
                        normal_delta = 1.0 - orientation
                        normal_weight = math.exp(
                            -(normal_delta * normal_delta) / (2.0 * normal_sigma * normal_sigma)
                        )
                        weight = spatial_weight * depth_weight * normal_weight
                        accumulated += candidate * weight
                        total_weight += weight
                if total_weight <= 1e-12:
                    continue
                normal = accumulated / total_weight
                length = float(np.linalg.norm(normal))
                if length > 1e-8:
                    filtered[y, x] = normal / length

    diagnostics = {
        "validDepthPixels": int(np.sum(valid)),
        "rawNormalPixels": int(np.sum(np.all(np.isfinite(raw), axis=2))),
        "filteredNormalPixels": int(np.sum(np.all(np.isfinite(filtered), axis=2))),
        "oneSidedAxisUses": int(one_sided_axes),
        "depthDiscontinuityRejects": int(discontinuity_rejects),
    }
    return points, filtered.reshape(-1, 3), valid, diagnostics


def reconstruct_points_and_raycast_truth_normals(
    scene: o3d.t.geometry.RaycastingScene,
    camera: dict[str, Any],
    depth: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Pair the selected depth image with immutable first-hit mesh normals.

    This is deliberately an oracle-only input arm.  Points still come from the
    selected ideal or degraded depth image, while normals are read from the
    corresponding first-hit triangle in the Replica mesh.  Consequently a
    degraded-depth run retains its holes and depth displacement but removes
    normal-estimator failure; an ideal-depth run removes both input defects.
    """
    points, origins, _, valid = reconstruct_depth_points(
        camera, depth, width, height
    )
    rays, _ = make_rays(camera, width, height, np.zeros(3, dtype=np.float64))
    answer = scene.cast_rays(o3d.core.Tensor(rays))
    hit_distance = answer["t_hit"].numpy().reshape(-1)
    hit_normals = answer["primitive_normals"].numpy().reshape(-1, 3).astype(np.float64)
    usable = valid & np.isfinite(hit_distance) & np.all(np.isfinite(hit_normals), axis=1)
    normals = np.full_like(points, np.nan)
    for index in np.flatnonzero(usable):
        normal = normalize(hit_normals[index])
        if np.linalg.norm(normal) <= 1e-8:
            continue
        if float(np.dot(normal, origins[index] - points[index])) < 0.0:
            normal = -normal
        normals[index] = normal
    oracle_pixels = int(np.sum(np.all(np.isfinite(normals), axis=1)))
    diagnostics = {
        "validDepthPixels": int(np.sum(valid)),
        "rawNormalPixels": oracle_pixels,
        "filteredNormalPixels": oracle_pixels,
        "oneSidedAxisUses": 0,
        "depthDiscontinuityRejects": 0,
        "raycastTruthNormalPixels": oracle_pixels,
    }
    return points, normals, valid, diagnostics


def perturb_paper_normals(
    normals: np.ndarray,
    valid: np.ndarray,
    depth: np.ndarray,
    width: int,
    height: int,
    angular_noise_sigma_degrees: float,
    dropout_probability: float,
    edge_angular_noise_sigma_degrees: float,
    edge_dropout_probability: float,
    edge_depth_factor: float,
    edge_depth_floor: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply reproducible offline-only corruption to a selected normal field.

    The perturbation is deliberately downstream of the normal source.  This
    keeps camera paths, depth observations and truth samples bit-identical while
    measuring how much normal error the fusion/extraction chain can tolerate.
    """
    if angular_noise_sigma_degrees < 0.0:
        raise ValueError("paper normal angular noise sigma must be non-negative")
    if edge_angular_noise_sigma_degrees < 0.0:
        raise ValueError("paper normal edge angular noise sigma must be non-negative")
    if not 0.0 <= dropout_probability <= 1.0:
        raise ValueError("paper normal dropout probability must be in [0, 1]")
    if not 0.0 <= edge_dropout_probability <= 1.0:
        raise ValueError("paper normal edge dropout probability must be in [0, 1]")
    if edge_depth_factor < 0.0 or edge_depth_floor < 0.0:
        raise ValueError("paper normal perturbation edge thresholds must be non-negative")

    output = normals.copy()
    depth_image = np.asarray(depth, dtype=np.float64).reshape(height, width)
    depth_valid = depth_image > 0.0
    edge_mask = np.zeros((height, width), dtype=bool)

    # A valid sample is an edge sample when it borders a depth hole or when a
    # four-neighbour depth jump exceeds a metric+relative conservative guard.
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        source_y = slice(max(0, -dy), min(height, height - dy))
        source_x = slice(max(0, -dx), min(width, width - dx))
        neighbour_y = slice(max(0, dy), min(height, height + dy))
        neighbour_x = slice(max(0, dx), min(width, width + dx))
        center_valid = depth_valid[source_y, source_x]
        neighbour_valid = depth_valid[neighbour_y, neighbour_x]
        center_depth = depth_image[source_y, source_x]
        neighbour_depth = depth_image[neighbour_y, neighbour_x]
        jump_threshold = np.maximum(
            edge_depth_floor,
            np.minimum(center_depth, neighbour_depth) * edge_depth_factor,
        )
        discontinuity = (
            center_valid
            & neighbour_valid
            & (np.abs(center_depth - neighbour_depth) > jump_threshold)
        )
        hole_boundary = center_valid & ~neighbour_valid
        edge_mask[source_y, source_x] |= discontinuity | hole_boundary

    flat_edge_mask = edge_mask.reshape(-1)
    source_usable = valid & np.all(np.isfinite(output), axis=1)
    rng = np.random.default_rng(seed)
    global_dropout = rng.random(len(output)) < dropout_probability
    edge_dropout = (
        flat_edge_mask
        & (rng.random(len(output)) < edge_dropout_probability)
    )
    dropped = source_usable & (global_dropout | edge_dropout)
    output[dropped] = np.nan

    retained = source_usable & ~dropped
    global_angles = rng.normal(
        0.0, math.radians(angular_noise_sigma_degrees), len(output)
    )
    edge_angles = rng.normal(
        0.0, math.radians(edge_angular_noise_sigma_degrees), len(output)
    )
    angles = global_angles + np.where(flat_edge_mask, edge_angles, 0.0)
    rotate_mask = retained & (np.abs(angles) > 1e-15)
    rotate_indices = np.flatnonzero(rotate_mask)
    if len(rotate_indices):
        base = output[rotate_indices]
        random_vectors = rng.normal(size=(len(rotate_indices), 3))
        tangents = random_vectors - base * np.sum(random_vectors * base, axis=1)[:, None]
        tangent_lengths = np.linalg.norm(tangents, axis=1)
        degenerate = tangent_lengths <= 1e-10
        if np.any(degenerate):
            fallback = np.cross(base[degenerate], np.asarray([1.0, 0.0, 0.0]))
            fallback_lengths = np.linalg.norm(fallback, axis=1)
            second_fallback = fallback_lengths <= 1e-10
            if np.any(second_fallback):
                fallback[second_fallback] = np.cross(
                    base[degenerate][second_fallback],
                    np.asarray([0.0, 1.0, 0.0]),
                )
                fallback_lengths = np.linalg.norm(fallback, axis=1)
            tangents[degenerate] = fallback
            tangent_lengths[degenerate] = fallback_lengths
        tangents /= tangent_lengths[:, None]
        selected_angles = angles[rotate_indices]
        output[rotate_indices] = (
            base * np.cos(selected_angles)[:, None]
            + tangents * np.sin(selected_angles)[:, None]
        )

    applied_degrees = np.degrees(np.abs(angles[retained]))
    retained_edges = retained & flat_edge_mask
    retained_non_edges = retained & ~flat_edge_mask
    diagnostics: dict[str, Any] = {
        "sourceUsableNormals": int(np.sum(source_usable)),
        "edgeUsableNormals": int(np.sum(source_usable & flat_edge_mask)),
        "droppedNormals": int(np.sum(dropped)),
        "droppedEdgeNormals": int(np.sum(dropped & flat_edge_mask)),
        "retainedNormals": int(np.sum(retained)),
        "retainedEdgeNormals": int(np.sum(retained_edges)),
        "retainedNonEdgeNormals": int(np.sum(retained_non_edges)),
        "perturbedNormals": int(np.sum(rotate_mask)),
        "appliedAbsoluteAngularErrorDegreesSum": float(np.sum(applied_degrees)),
        "appliedAbsoluteAngularErrorDegreesSquaredSum": float(
            np.sum(applied_degrees * applied_degrees)
        ),
        "appliedAbsoluteAngularErrorSamples": int(len(applied_degrees)),
    }
    return output, diagnostics


def summarize_paper_normal_perturbation(
    audit: dict[str, float],
) -> dict[str, Any]:
    summary: dict[str, Any] = dict(audit)
    source = float(audit.get("sourceUsableNormals", 0.0))
    edge_source = float(audit.get("edgeUsableNormals", 0.0))
    angular_samples = float(audit.get("appliedAbsoluteAngularErrorSamples", 0.0))
    summary["dropoutRatio"] = (
        float(audit.get("droppedNormals", 0.0)) / source if source else 0.0
    )
    summary["edgeDropoutRatio"] = (
        float(audit.get("droppedEdgeNormals", 0.0)) / edge_source
        if edge_source else 0.0
    )
    summary["appliedAbsoluteAngularErrorDegreesMean"] = (
        float(audit.get("appliedAbsoluteAngularErrorDegreesSum", 0.0))
        / angular_samples
        if angular_samples else 0.0
    )
    summary["appliedAngularErrorDegreesRms"] = (
        math.sqrt(
            float(audit.get("appliedAbsoluteAngularErrorDegreesSquaredSum", 0.0))
            / angular_samples
        )
        if angular_samples else 0.0
    )
    return summary


def write_mesh(path: Path, build: MeshBuild) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(path), to_open3d(build), write_ascii=False, compressed=False)


def write_hermite_feature_points(
    output_directory: Path,
    feature: dict[str, Any],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    points = feature["points"]
    baselines = feature["baselines"]
    ranks = feature["ranks"]
    if len(points):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        colors = np.zeros((len(points), 3), dtype=np.float64)
        colors[ranks == 2] = np.asarray([0.0, 0.85, 1.0])
        colors[ranks >= 3] = np.asarray([1.0, 0.0, 0.7])
        cloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(
            str(output_directory / "tsdf_hermite_feature_points.ply"),
            cloud,
            write_ascii=False,
            compressed=False,
        )
        baseline_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(baselines))
        baseline_cloud.paint_uniform_color([0.95, 0.65, 0.0])
        o3d.io.write_point_cloud(
            str(output_directory / "tsdf_hermite_feature_baselines.ply"),
            baseline_cloud,
            write_ascii=False,
            compressed=False,
        )


def evaluate_hermite_feature_points(
    feature: dict[str, Any],
    truth_scene: o3d.t.geometry.RaycastingScene,
) -> dict[str, Any]:
    points = feature["points"]
    baselines = feature["baselines"]
    if len(points) == 0:
        return {
            "acceptedFeaturePoints": 0,
            "truthImprovedFraction": 0.0,
        }

    def truth_distance(values: np.ndarray) -> np.ndarray:
        tensor = o3d.core.Tensor(values.astype(np.float32), dtype=o3d.core.Dtype.Float32)
        return truth_scene.compute_distance(tensor).numpy().astype(np.float64)

    before = truth_distance(baselines)
    after = truth_distance(points)
    delta = after - before
    cells = list(feature.get("cells", []))
    certificates = list(feature.get("certificates", []))
    per_feature: list[dict[str, Any]] = []
    for index in range(len(points)):
        row: dict[str, Any] = {
            "index": int(index),
            "baselineTruthDistanceMeters": float(before[index]),
            "qefTruthDistanceMeters": float(after[index]),
            "truthDistanceDeltaMeters": float(delta[index]),
            "improved": bool(delta[index] < 0.0),
        }
        if index < len(cells):
            row["cell"] = [int(value) for value in cells[index]]
        if index < len(certificates):
            row["certificate"] = copy.deepcopy(certificates[index])
        per_feature.append(row)
    return {
        "acceptedFeaturePoints": int(len(points)),
        "truthDistanceBaselineMeters": {
            "mean": float(np.mean(before)),
            "p50": float(np.percentile(before, 50)),
            "p95": float(np.percentile(before, 95)),
        },
        "truthDistanceQefMeters": {
            "mean": float(np.mean(after)),
            "p50": float(np.percentile(after, 50)),
            "p95": float(np.percentile(after, 95)),
        },
        "truthDistanceDeltaMeters": {
            "mean": float(np.mean(delta)),
            "p50": float(np.percentile(delta, 50)),
            "p95": float(np.percentile(delta, 95)),
        },
        "truthImprovedFraction": float(np.mean(delta < 0.0)),
        # Offline audit only.  Truth is attached after extraction and is never
        # available to the certificate or topology decision.
        "perFeatureTruthAudit": per_feature,
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    o3d.utility.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    with args.degradation_model.open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    mesh = load_legacy_mesh(args.mesh)
    scene = build_scene(mesh)
    path_checkpoints = sorted({int(value) for value in args.camera_path_checkpoints if int(value) > 0})
    master_frame_count = max([args.frames, *path_checkpoints])
    synthetic_fov_y_degrees = 100.2439
    synthetic_aspect = args.width / args.height
    cameras, scan_metadata = build_stratified_slice_cameras(
        mesh,
        master_frame_count,
        "auto",
        3,
        [0.35, 0.75, 1.25, 1.65, 2.10],
        [0.0, 60.0, 120.0, 180.0, 240.0, 300.0],
        synthetic_fov_y_degrees,
        synthetic_aspect,
    )
    if path_checkpoints:
        cameras = progressive_checkpoint_order(cameras, path_checkpoints)
    cameras = cameras[: args.frames]
    scan_metadata["masterPathFrames"] = master_frame_count
    scan_metadata["cameraPathCheckpoints"] = path_checkpoints
    scan_metadata["generatedFrames"] = len(cameras)
    scan_metadata["nestedProgressivePrefixes"] = bool(path_checkpoints)
    truth = mesh.sample_points_uniformly(number_of_points=args.truth_samples, use_triangle_normal=True)
    truth_points = np.asarray(truth.points, dtype=np.float64)
    truth_sample_sha256 = hashlib.sha256(
        np.ascontiguousarray(truth_points.astype("<f8")).tobytes()
    ).hexdigest()
    observed_truth_mask = np.zeros(len(truth_points), dtype=bool)
    ideal_visible_truth_mask = np.zeros(len(truth_points), dtype=bool)
    usable_depth_normal_truth_mask = np.zeros(len(truth_points), dtype=bool)
    structure_reference = build_mesh_structure_reference(
        mesh, args.feature_angle_degrees
    )
    truth_structure_masks = build_truth_structure_masks(
        truth_points, structure_reference, args.structure_band_meters
    )
    dominant = DirectionalGrid(args.voxel, args.sdf_trunc, args.soft_direction_threshold, False)
    soft = DirectionalGrid(args.voxel, args.sdf_trunc, args.soft_direction_threshold, True)
    accepted_points = 0
    paper_normal_estimation_audit = {
        "validDepthPixels": 0,
        "rawNormalPixels": 0,
        "filteredNormalPixels": 0,
        "oneSidedAxisUses": 0,
        "depthDiscontinuityRejects": 0,
        "raycastTruthNormalPixels": 0,
    }
    paper_normal_perturbation_audit: dict[str, float] = {
        "sourceUsableNormals": 0.0,
        "edgeUsableNormals": 0.0,
        "droppedNormals": 0.0,
        "droppedEdgeNormals": 0.0,
        "retainedNormals": 0.0,
        "retainedEdgeNormals": 0.0,
        "retainedNonEdgeNormals": 0.0,
        "perturbedNormals": 0.0,
        "appliedAbsoluteAngularErrorDegreesSum": 0.0,
        "appliedAbsoluteAngularErrorDegreesSquaredSum": 0.0,
        "appliedAbsoluteAngularErrorSamples": 0.0,
    }
    growth_checkpoints = set(path_checkpoints if path_checkpoints else [args.frames])
    growth_results: list[dict[str, Any]] = []
    growth_pause_ms = 0.0
    integration_started = time.perf_counter()
    observation_hasher = hashlib.sha256()
    normal_hasher = hashlib.sha256()
    camera_path_hasher = hashlib.sha256()
    degraded_depth_valid_pixels = 0
    for frame_index, camera in enumerate(cameras, start=1):
        dominant.begin_frame(frame_index)
        soft.begin_frame(frame_index)
        ideal, _, guarded, *_ = make_depth_pair(
            scene,
            camera,
            args.width,
            args.height,
            args.min_distance,
            args.max_distance,
            model,
            True,
        )
        depth = ideal if args.ideal_depth else guarded
        camera_payload = json.dumps(
            camera,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        camera_path_hasher.update(len(camera_payload).to_bytes(8, "little"))
        camera_path_hasher.update(camera_payload)
        depth_payload = np.ascontiguousarray(depth.astype("<f4")).tobytes()
        observation_hasher.update(len(camera_payload).to_bytes(8, "little"))
        observation_hasher.update(camera_payload)
        observation_hasher.update(len(depth_payload).to_bytes(8, "little"))
        observation_hasher.update(depth_payload)
        degraded_depth_valid_pixels += int(np.sum(depth > 0.0))
        update_observed_truth_mask(
            scene,
            truth_points,
            camera,
            ideal,
            args.width,
            args.height,
            args.min_distance,
            args.max_distance,
            ideal_visible_truth_mask,
        )
        update_observed_truth_mask(
            scene,
            truth_points,
            camera,
            depth,
            args.width,
            args.height,
            args.min_distance,
            args.max_distance,
            observed_truth_mask,
        )
        if args.integration_mode == "paper-normal-raycast":
            if args.paper_normal_source == "raycast-truth":
                points, normals, valid, frame_normal_audit = (
                    reconstruct_points_and_raycast_truth_normals(
                        scene,
                        camera,
                        depth,
                        args.width,
                        args.height,
                    )
                )
            else:
                points, normals, valid, frame_normal_audit = (
                    reconstruct_points_and_paper_normals(
                        camera,
                        depth,
                        args.width,
                        args.height,
                        args.paper_normal_radius,
                        args.paper_normal_depth_change_factor,
                        args.paper_normal_depth_change_floor,
                        args.paper_normal_bilateral_radius,
                    )
                )
            for key, value in frame_normal_audit.items():
                paper_normal_estimation_audit[key] += int(value)
            normals, frame_perturbation_audit = perturb_paper_normals(
                normals,
                valid,
                depth,
                args.width,
                args.height,
                args.paper_normal_angular_noise_sigma_degrees,
                args.paper_normal_dropout_probability,
                args.paper_normal_edge_angular_noise_sigma_degrees,
                args.paper_normal_edge_dropout_probability,
                args.paper_normal_perturbation_edge_depth_factor,
                args.paper_normal_perturbation_edge_depth_floor,
                args.paper_normal_perturbation_seed + frame_index * 1_000_003,
            )
            for key, value in frame_perturbation_audit.items():
                paper_normal_perturbation_audit[key] += float(value)
        else:
            points, normals, valid = reconstruct_points_and_normals(
                camera, depth, args.width, args.height
            )
        normal_payload = np.ascontiguousarray(normals.astype("<f4")).tobytes()
        normal_hasher.update(len(normal_payload).to_bytes(8, "little"))
        normal_hasher.update(normal_payload)
        usable_pixel_mask = valid & np.all(np.isfinite(normals), axis=1)
        update_observed_truth_mask(
            scene,
            truth_points,
            camera,
            depth,
            args.width,
            args.height,
            args.min_distance,
            args.max_distance,
            usable_depth_normal_truth_mask,
            usable_pixel_mask,
        )
        if args.integration_mode == "projective":
            accepted_this_frame = dominant.integrate_projective_frame(
                camera,
                depth,
                points,
                normals,
                valid,
                args.width,
                args.height,
                args.sample_stride,
                args.projective_block_voxels,
            )
            soft_accepted = soft.integrate_projective_frame(
                camera,
                depth,
                points,
                normals,
                valid,
                args.width,
                args.height,
                args.sample_stride,
                args.projective_block_voxels,
            )
            if soft_accepted != accepted_this_frame:
                raise RuntimeError("dominant/soft projective candidate inputs diverged")
            accepted_points += accepted_this_frame
        else:
            camera_position = np.asarray(camera["pose"]["position"], dtype=np.float64)
            for index in range(0, len(points), max(1, args.sample_stride)):
                if not valid[index] or not np.all(np.isfinite(normals[index])):
                    continue
                if args.integration_mode == "paper-normal-raycast":
                    dominant_accepted = dominant.integrate_paper_normal_raycast(
                        camera_position,
                        points[index],
                        normals[index],
                        args.min_distance,
                    )
                    soft_accepted = soft.integrate_paper_normal_raycast(
                        camera_position,
                        points[index],
                        normals[index],
                        args.min_distance,
                    )
                    if dominant_accepted != soft_accepted:
                        raise RuntimeError("dominant/soft paper ray inputs diverged")
                    accepted_points += int(soft_accepted)
                elif args.integration_mode == "normal-raycast":
                    dominant.integrate_normal_raycast(
                        camera_position, points[index], normals[index]
                    )
                    soft.integrate_normal_raycast(
                        camera_position, points[index], normals[index]
                    )
                    accepted_points += 1
                else:
                    dominant.integrate(camera_position, points[index], normals[index])
                    soft.integrate(camera_position, points[index], normals[index])
                    accepted_points += 1
        print(
            f"[directional-offline {frame_index}/{len(cameras)}] "
            f"mode={args.integration_mode} points={accepted_points} "
            f"dominantVoxels={sum(len(layer) for layer in dominant.values)} "
            f"softVoxels={sum(len(layer) for layer in soft.values)}",
            flush=True,
        )
        if args.paper_growth_ledger_only and frame_index in growth_checkpoints:
            checkpoint_started = time.perf_counter()
            checkpoint_build = extract_paper_dmc(
                soft, args.paper_minimum_weight, regularize=True
            )
            checkpoint_mask_sink: dict[str, np.ndarray] = {}
            checkpoint_metrics = evaluate_mesh(
                checkpoint_build,
                truth,
                scene,
                observed_truth_mask,
                structure_reference,
                args.structure_band_meters,
                checkpoint_mask_sink,
            )
            checkpoint_metrics["frame"] = frame_index
            checkpoint_metrics["acceptedSurfacePoints"] = accepted_points
            checkpoint_metrics["observedTruthSamples"] = int(np.sum(observed_truth_mask))
            checkpoint_metrics["observedTruthRatioOfRoom"] = (
                float(np.mean(observed_truth_mask)) if len(observed_truth_mask) else 0.0
            )
            if (
                (
                    args.paper_growth_stage_attribution
                    or args.paper_growth_upstream_attribution
                )
                and frame_index == max(growth_checkpoints)
            ):
                stage_builds = {
                    "raw": extract_paper_stage_independent(
                        soft, args.paper_minimum_weight, "raw"
                    ),
                    "filtered": extract_paper_stage_independent(
                        soft, args.paper_minimum_weight, "filtered"
                    ),
                    "voted": extract_paper_stage_independent(
                        soft, args.paper_minimum_weight, "voted"
                    ),
                }
                stage_masks = {
                    "dmc": checkpoint_mask_sink["coveredAt0.05m"],
                }
                stage_metrics: dict[str, Any] = {}
                for stage_name, stage_build in stage_builds.items():
                    stage_sink: dict[str, np.ndarray] = {}
                    stage_result = evaluate_mesh(
                        stage_build,
                        truth,
                        scene,
                        observed_truth_mask,
                        structure_reference,
                        args.structure_band_meters,
                        stage_sink,
                    )
                    stage_masks[stage_name] = stage_sink["coveredAt0.05m"]
                    stage_metrics[stage_name] = {
                        "visibleCoverageAt0.05m": stage_result["visibleCoverageAt0.05m"],
                        "wholeRoomCoverageAt0.05m": stage_result["wholeRoomCoverageAt0.05m"],
                        "exactVisibleCoverageAt0.05m": stage_result["exactVisibleCoverageAt0.05m"],
                        "exactWholeRoomCoverageAt0.05m": stage_result["exactWholeRoomCoverageAt0.05m"],
                        "extraSurfaceRatioAt0.05m": stage_result["extraSurfaceRatioAt0.05m"],
                        "accuracyP95m": stage_result["accuracyP95m"],
                        "triangles": stage_result["triangles"],
                        "nonManifoldEdges": stage_result["nonManifoldEdges"],
                    }
                stage_metrics["dmc"] = {
                    "visibleCoverageAt0.05m": checkpoint_metrics["visibleCoverageAt0.05m"],
                    "wholeRoomCoverageAt0.05m": checkpoint_metrics["wholeRoomCoverageAt0.05m"],
                    "exactVisibleCoverageAt0.05m": checkpoint_metrics["exactVisibleCoverageAt0.05m"],
                    "exactWholeRoomCoverageAt0.05m": checkpoint_metrics["exactWholeRoomCoverageAt0.05m"],
                    "extraSurfaceRatioAt0.05m": checkpoint_metrics["extraSurfaceRatioAt0.05m"],
                    "accuracyP95m": checkpoint_metrics["accuracyP95m"],
                    "triangles": checkpoint_metrics["triangles"],
                    "nonManifoldEdges": checkpoint_metrics["nonManifoldEdges"],
                }
                checkpoint_metrics["stageMetrics"] = stage_metrics
                checkpoint_metrics["stageAttribution"] = attribute_visible_surface_loss(
                    observed_truth_mask,
                    stage_masks,
                )
                checkpoint_metrics["structureStageAttribution"] = {
                    label: attribute_visible_surface_loss(
                        observed_truth_mask & structure_mask,
                        stage_masks,
                    )
                    for label, structure_mask in truth_structure_masks.items()
                }
                if args.paper_growth_upstream_attribution:
                    supply_masks, supply_diagnostics = build_tsdf_supply_masks(
                        soft,
                        truth_points,
                        args.paper_minimum_weight,
                        0.05,
                    )
                    checkpoint_metrics["tsdfSupplyAttribution"] = attribute_tsdf_supply_loss(
                        ideal_visible_truth_mask,
                        observed_truth_mask,
                        usable_depth_normal_truth_mask,
                        supply_masks["projectiveVoxelTouched"],
                        supply_masks["completeCornerWeightSupport"],
                        stage_masks["raw"],
                        supply_diagnostics,
                    )
                    checkpoint_metrics["structureTsdfSupplyAttribution"] = {
                        label: attribute_tsdf_supply_loss(
                            ideal_visible_truth_mask & structure_mask,
                            observed_truth_mask & structure_mask,
                            usable_depth_normal_truth_mask & structure_mask,
                            supply_masks["projectiveVoxelTouched"],
                            supply_masks["completeCornerWeightSupport"],
                            stage_masks["raw"],
                            supply_diagnostics,
                        )
                        for label, structure_mask in truth_structure_masks.items()
                    }
            growth_results.append(checkpoint_metrics)
            growth_pause_ms += (time.perf_counter() - checkpoint_started) * 1000.0
            print(
                f"[coverage-ledger {frame_index}] "
                f"visible={checkpoint_metrics['visibleCoverageAt0.05m']:.4f} "
                f"wholeRoom={checkpoint_metrics['wholeRoomCoverageAt0.05m']:.4f} "
                f"observed={checkpoint_metrics['observedTruthRatioOfRoom']:.4f}",
                flush=True,
            )
    integration_ms = (time.perf_counter() - integration_started) * 1000.0 - growth_pause_ms
    paper_normal_perturbation_summary = summarize_paper_normal_perturbation(
        paper_normal_perturbation_audit
    )

    if args.paper_growth_ledger_only:
        if not growth_results:
            raise RuntimeError("No growth-ledger checkpoint was reached")
        visible_deltas = [
            current["visibleCoverageAt0.05m"] - previous["visibleCoverageAt0.05m"]
            for previous, current in zip(growth_results, growth_results[1:])
        ]
        whole_room_deltas = [
            current["wholeRoomCoverageAt0.05m"] - previous["wholeRoomCoverageAt0.05m"]
            for previous, current in zip(growth_results, growth_results[1:])
        ]
        observation_deltas = [
            current["observedTruthRatioOfRoom"] - previous["observedTruthRatioOfRoom"]
            for previous, current in zip(growth_results, growth_results[1:])
        ]
        diagnostic_gate = {
            "observationLedgerMonotonic": all(delta >= -1e-12 for delta in observation_deltas),
            "visibleCoverageNoLargeRegression": all(delta >= -0.01 for delta in visible_deltas),
            "wholeRoomCoverageNoLargeRegression": all(delta >= -0.01 for delta in whole_room_deltas),
            "nonManifoldFree": all(row["nonManifoldEdges"] == 0 for row in growth_results),
            "measuredEdgeDecisionsComplete": all(
                row["audit"]["paper_unmeasured_edge_deferred_triangles"] == 0
                and row["audit"]["paper_unresolved_edge_slots"] == 0
                for row in growth_results
            ),
        }
        attributed_rows = [row for row in growth_results if "stageAttribution" in row]
        if args.paper_growth_stage_attribution:
            diagnostic_gate["stageAttributionProduced"] = len(attributed_rows) == 1
            diagnostic_gate["stageAttributionAccountingClosed"] = (
                len(attributed_rows) == 1
                and bool(attributed_rows[0]["stageAttribution"]["accountingClosed"])
            )
            diagnostic_gate["structureStageAttributionAccountingClosed"] = (
                len(attributed_rows) == 1
                and all(
                    bool(values["accountingClosed"])
                    for values in attributed_rows[0]
                    .get("structureStageAttribution", {})
                    .values()
                )
            )
        upstream_rows = [row for row in growth_results if "tsdfSupplyAttribution" in row]
        if args.paper_growth_upstream_attribution:
            diagnostic_gate["tsdfSupplyAttributionProduced"] = len(upstream_rows) == 1
            diagnostic_gate["tsdfSupplyAttributionAccountingClosed"] = (
                len(upstream_rows) == 1
                and bool(upstream_rows[0]["tsdfSupplyAttribution"]["accountingClosed"])
            )
            diagnostic_gate["structureTsdfSupplyAttributionAccountingClosed"] = (
                len(upstream_rows) == 1
                and all(
                    bool(values["accountingClosed"])
                    for values in upstream_rows[0]
                    .get("structureTsdfSupplyAttribution", {})
                    .values()
                )
            )
        report = {
            "schema": "scancover.directional_tsdf_composition.coverage_growth.v1",
            "mesh": str(args.mesh),
            "degradationModel": str(args.degradation_model),
            "mode": "ideal" if args.ideal_depth else "quest_guarded",
            "scope": "paper DMC dual-ledger frame-growth diagnostic; Unity disabled",
            "parameters": {
                "frames": args.frames,
                "cameraPathCheckpoints": path_checkpoints,
                "width": args.width,
                "height": args.height,
                "truthSamples": args.truth_samples,
                "voxelMeters": args.voxel,
                "truncationMeters": args.sdf_trunc,
                "paperMinimumWeight": args.paper_minimum_weight,
                "integrationMode": args.integration_mode,
                "sampleStride": args.sample_stride,
                "paperDirectionThreshold": PAPER_DIRECTION_THRESHOLD,
                "paperNormalRadius": args.paper_normal_radius,
                "paperNormalDepthChangeFactor": args.paper_normal_depth_change_factor,
                "paperNormalDepthChangeFloorMeters": args.paper_normal_depth_change_floor,
                "paperNormalBilateralRadius": args.paper_normal_bilateral_radius,
                "paperNormalSource": args.paper_normal_source,
                "paperNormalPerturbation": {
                    "angularNoiseSigmaDegrees": args.paper_normal_angular_noise_sigma_degrees,
                    "dropoutProbability": args.paper_normal_dropout_probability,
                    "edgeAngularNoiseSigmaDegrees": args.paper_normal_edge_angular_noise_sigma_degrees,
                    "edgeDropoutProbability": args.paper_normal_edge_dropout_probability,
                    "edgeDepthFactor": args.paper_normal_perturbation_edge_depth_factor,
                    "edgeDepthFloorMeters": args.paper_normal_perturbation_edge_depth_floor,
                    "seed": args.paper_normal_perturbation_seed,
                },
                "paperDepthWeight": "Nguyen_sigma_ratio_times_inverse_depth_squared",
                "paperAngleWeight": "cosine_surface_normal_to_view",
                "stageAttribution": bool(args.paper_growth_stage_attribution),
                "upstreamTsdfSupplyAttribution": bool(
                    args.paper_growth_upstream_attribution
                ),
            },
            "scan": scan_metadata,
            "inputSamplingAudit": input_sampling_audit(
                args.width,
                args.height,
                args.voxel,
                synthetic_fov_y_degrees,
                synthetic_aspect,
            ),
            "coverageLedgers": {
                "definition": {
                    "visible": "truth samples in a valid depth pixel and unobstructed from at least one camera",
                    "wholeRoom": "all uniformly sampled truth-mesh points",
                    "thresholdMeters": 0.05,
                },
                "visibilityOcclusionTolerance": "max(0.002m, range*0.001)",
            },
            "strictInputAudit": {
                "cameraPathSha256": camera_path_hasher.hexdigest(),
                "observationDepthSequenceSha256": observation_hasher.hexdigest(),
                "normalSequenceSha256": normal_hasher.hexdigest(),
                "truthSamplesSha256": truth_sample_sha256,
                "degradedDepthValidPixels": degraded_depth_valid_pixels,
                "selectedDepthValidPixels": degraded_depth_valid_pixels,
            },
            "integrationMs": integration_ms,
            "checkpointExtractionAndEvaluationMs": growth_pause_ms,
            "integrationAudit": {
                "projective": {
                    "candidateBlocks": soft.projective_candidate_blocks,
                    "candidateVoxels": soft.projective_candidate_voxels,
                    "visibleVoxels": soft.projective_visible_voxels,
                    "validDepthVoxels": soft.projective_valid_depth_voxels,
                    "behindSurfaceTruncationRejects": soft.projective_truncation_rejects,
                    "integratedVoxelDirectionWrites": soft.voxel_updates,
                },
                "paperNormalEstimation": paper_normal_estimation_audit,
                "paperNormalPerturbation": paper_normal_perturbation_summary,
                "paperNormalRaycast": {
                    "rays": soft.paper_normal_rays,
                    "traversedVoxels": soft.paper_traversed_voxels,
                    "integratedVoxels": soft.paper_integrated_voxels,
                    "depthWeightMean": (
                        soft.paper_depth_weight_sum / soft.paper_normal_rays
                        if soft.paper_normal_rays else 0.0
                    ),
                    "angleWeightMean": (
                        soft.paper_angle_weight_sum / soft.paper_normal_rays
                        if soft.paper_normal_rays else 0.0
                    ),
                    "combinedDirectionalWeightSum": soft.paper_combined_weight_sum,
                    "integratedVoxelDirectionWrites": soft.voxel_updates,
                },
            },
            "checkpoints": growth_results,
            "deltas": {
                "visibleCoverageAt0.05m": visible_deltas,
                "wholeRoomCoverageAt0.05m": whole_room_deltas,
                "observedTruthRatioOfRoom": observation_deltas,
            },
            "diagnosticGate": diagnostic_gate,
            "passed": all(diagnostic_gate.values()),
        }
        with (args.out / "directional_composition_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        lines = [
            "# ScanCover DMC Dual-Ledger Growth Diagnostic",
            "",
            f"- mesh: {args.mesh.stem}",
            f"- input: {report['mode']}",
            f"- diagnostic integrity passed: {report['passed']}",
            "- Unity: disabled",
            "",
            "| Frames | Observed truth / room | Visible recovery 5cm | Whole-room coverage 5cm | Extra 5cm | Accuracy p95 | Boundary / 1k tri | Non-manifold |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in growth_results:
            lines.append(
                f"| {row['frame']} | {row['observedTruthRatioOfRoom']:.4f} | "
                f"{row['visibleCoverageAt0.05m']:.4f} | {row['wholeRoomCoverageAt0.05m']:.4f} | "
                f"{row['extraSurfaceRatioAt0.05m']:.4f} | {row['accuracyP95m']:.4f} | "
                f"{row['boundaryEdgesPerKTriangles']:.2f} | {row['nonManifoldEdges']} |"
            )
        lines.extend(["", "## Diagnostic gate", ""])
        lines.extend(f"- {key}: {value}" for key, value in diagnostic_gate.items())
        if attributed_rows:
            attribution = attributed_rows[0]["stageAttribution"]
            lines.extend(["", "## Final visible-loss first-failure attribution", ""])
            lines.append(
                f"- final missing visible ratio: {attribution['finalMissingVisibleRatio']:.4f}"
            )
            for name, values in attribution["firstFailure"].items():
                lines.append(
                    f"- {name}: {values['truthSamples']} samples; "
                    f"visible={values['ratioOfVisibleTruth']:.4f}; "
                    f"missing-share={values['shareOfMissingVisible']:.4f}"
                )
        if upstream_rows:
            upstream = upstream_rows[0]["tsdfSupplyAttribution"]
            lines.extend(["", "## Raw TSDF/corner supply first-failure attribution", ""])
            lines.append(
                f"- raw TSDF missing visible ratio: {upstream['rawTsdfMissingVisibleRatio']:.4f}"
            )
            for name, values in upstream["firstFailure"].items():
                lines.append(
                    f"- {name}: {values['truthSamples']} samples; "
                    f"visible={values['ratioOfVisibleTruth']:.4f}; "
                    f"raw-missing-share={values['shareOfRawTsdfMissing']:.4f}"
                )
            sensor = upstream["sensorValidityLedgerOutsideConditionalVisibleLoss"]
            lines.append(
                "- pure Quest valid-depth loss outside the conditional visible ledger: "
                f"{sensor['missingValidDepthRatioOfIdealVisible']:.4f}"
            )
        (args.out / "directional_composition_report.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            "passed": report["passed"],
            "diagnosticGate": diagnostic_gate,
            "out": str(args.out),
        }, ensure_ascii=False))
        return 0

    dominant_independent = extract_independent(dominant, args.minimum_weight)
    dominant_composed = extract_composed(
        dominant,
        args.minimum_weight,
        args.valid_gradient_dot,
        args.parallel_dot,
        args.edge_merge_voxel_ratio,
    )
    soft_composed = extract_composed(
        soft,
        args.minimum_weight,
        args.valid_gradient_dot,
        args.parallel_dot,
        args.edge_merge_voxel_ratio,
    )
    if args.paper_hermite_qef_ab_only:
        paper_ownership: dict[str, Any] = {}
        paper_cells: dict[tuple[int, int, int], PaperDmcCell] = {}
        paper_dmc = extract_paper_dmc(
            soft,
            args.paper_minimum_weight,
            regularize=True,
            ownership_output=paper_ownership,
            cells_output=paper_cells,
        )
        tsdf_hermite_features = extract_tsdf_hermite_feature_points(
            soft,
            args.minimum_weight,
            args.valid_gradient_dot,
            args.feature_angle_degrees,
            args.feature_min_family_support_ratio,
            args.feature_rank_ratio,
            args.feature_certificate_min_frames_per_family,
            args.feature_certificate_min_views_per_family,
            args.feature_certificate_min_samples_per_family,
            args.feature_certificate_min_rank_ratio,
            args.feature_certificate_min_cell_margin_ratio,
            args.feature_certificate_min_family_weight_ratio,
            args.feature_certificate_min_qef_displacement_ratio,
        )
        feature_mesh, applied_features = (
            extract_paper_dmc_tsdf_hermite_qef_feature_mesh(
                paper_dmc,
                paper_ownership,
                paper_cells,
                tsdf_hermite_features,
                args.voxel,
            )
        )
        write_hermite_feature_points(
            args.out / "all_tsdf_hermite_candidates",
            tsdf_hermite_features,
        )
        write_hermite_feature_points(
            args.out / "applied_tsdf_hermite_features",
            applied_features,
        )
        variants = {
            "paper_dmc": paper_dmc,
            "paper_dmc_tsdf_hermite_qef": feature_mesh,
        }
        results: dict[str, Any] = {}
        for name, build in variants.items():
            write_mesh(args.out / f"{name}.ply", build)
            results[name] = evaluate_mesh(
                build,
                truth,
                scene,
                observed_truth_mask,
                structure_reference,
                args.structure_band_meters,
            )
        baseline = results["paper_dmc"]
        feature_result = results["paper_dmc_tsdf_hermite_qef"]
        all_feature_geometry = evaluate_hermite_feature_points(
            tsdf_hermite_features, scene
        )
        applied_feature_geometry = evaluate_hermite_feature_points(
            applied_features, scene
        )
        placement_audit = applied_features["audit"]

        def structure_floor(label: str, metric: str, tolerance: float) -> bool:
            baseline_band = baseline.get("structureBands", {}).get(label)
            feature_band = feature_result.get("structureBands", {}).get(label)
            if not baseline_band or not feature_band:
                return True
            if metric == "coverageAt0.05m":
                return (
                    float(feature_band[metric])
                    >= float(baseline_band[metric]) - tolerance
                )
            return (
                float(feature_band[metric])
                <= float(baseline_band[metric]) + tolerance
            )

        feature_mean_before = (
            applied_feature_geometry.get("truthDistanceBaselineMeters", {}).get(
                "mean", float("inf")
            )
        )
        feature_mean_after = (
            applied_feature_geometry.get("truthDistanceQefMeters", {}).get(
                "mean", float("inf")
            )
        )
        feature_p95_before = (
            applied_feature_geometry.get("truthDistanceBaselineMeters", {}).get(
                "p95", float("inf")
            )
        )
        feature_p95_after = (
            applied_feature_geometry.get("truthDistanceQefMeters", {}).get(
                "p95", float("inf")
            )
        )
        applied_feature_count = int(
            applied_feature_geometry.get("acceptedFeaturePoints", 0)
        )
        hard_gate = {
            "hermiteCandidatesFound": (
                int(tsdf_hermite_features["audit"].get("featureCandidates", 0)) > 0
            ),
            "featureCellsApplied": int(placement_audit.get("appliedCells", 0)) > 0,
            "interCellBoundaryLedgerPreserved": bool(
                placement_audit.get("interCellBoundaryLedgerPreserved", False)
            ),
            "globalBoundaryEdgesPreserved": (
                feature_result["boundaryEdges"] == baseline["boundaryEdges"]
            ),
            "nonManifoldNotWorse": (
                feature_result["nonManifoldEdges"] <= baseline["nonManifoldEdges"]
            ),
            "connectedComponentsPreserved": (
                feature_result["connectedComponents"]
                == baseline["connectedComponents"]
            ),
            "generatedFanOrientationSafe": (
                int(placement_audit.get("appliedCells", 0)) > 0
                and float(
                    placement_audit.get("minimumReferenceNormalDot", -1.0)
                )
                >= 0.05
            ),
            "coverageFloor": (
                feature_result["coverageAt0.05m"]
                >= baseline["coverageAt0.05m"] - 0.005
            ),
            "visibleCoverageFloor": (
                feature_result["visibleCoverageAt0.05m"]
                >= baseline["visibleCoverageAt0.05m"] - 0.005
            ),
            "extraSurfaceNotWorse": (
                feature_result["extraSurfaceRatioAt0.05m"]
                <= baseline["extraSurfaceRatioAt0.05m"] + 0.002
            ),
            "accuracyP95NotWorse": (
                feature_result["accuracyP95m"]
                <= baseline["accuracyP95m"] + 0.001
            ),
            "convexCreaseCoverageFloor": structure_floor(
                "convex_crease", "coverageAt0.05m", 0.005
            ),
            "concaveCreaseCoverageFloor": structure_floor(
                "concave_crease", "coverageAt0.05m", 0.005
            ),
            "convexCreaseAccuracyNotWorse": structure_floor(
                "convex_crease", "accuracyP95m", 0.001
            ),
            "concaveCreaseAccuracyNotWorse": structure_floor(
                "concave_crease", "accuracyP95m", 0.001
            ),
            "appliedFeatureTruthImprovementMajority": (
                float(applied_feature_geometry.get("truthImprovedFraction", 0.0))
                >= 0.5
            ),
            "appliedFeatureTruthMeanNotWorse": (
                applied_feature_count > 0
                and feature_mean_after <= feature_mean_before + 0.00025
            ),
            "appliedFeatureTruthP95NotWorse": (
                applied_feature_count > 0
                and feature_p95_after <= feature_p95_before + 0.001
            ),
            "appliedFeatureHasMeasuredGain": (
                applied_feature_count > 0
                and (
                    feature_mean_after <= feature_mean_before - 0.0001
                    or feature_p95_after <= feature_p95_before - 0.0001
                )
            ),
        }
        report = {
            "schema": "scancover.paper_dmc_tsdf_hermite_qef_ab.v1",
            "mesh": str(args.mesh),
            "degradationModel": str(args.degradation_model),
            "mode": "ideal" if args.ideal_depth else "quest_guarded",
            "scope": (
                "strict offline A/B; paper DMC shared-edge/inter-cell topology "
                "frozen; TSDF-Hermite QEF may change only a proven cell's "
                "interior triangulation; Unity disabled"
            ),
            "parameters": {
                "frames": args.frames,
                "width": args.width,
                "height": args.height,
                "voxelMeters": args.voxel,
                "truncationMeters": args.sdf_trunc,
                "minimumWeight": args.minimum_weight,
                "paperMinimumWeight": args.paper_minimum_weight,
                "featureAngleDegrees": args.feature_angle_degrees,
                "featureMinFamilySupportRatio": (
                    args.feature_min_family_support_ratio
                ),
                "featureRankRatio": args.feature_rank_ratio,
                "featureCertificate": {
                    "minFramesPerFamily": (
                        args.feature_certificate_min_frames_per_family
                    ),
                    "minViewsPerFamily": (
                        args.feature_certificate_min_views_per_family
                    ),
                    "minSamplesPerFamily": (
                        args.feature_certificate_min_samples_per_family
                    ),
                    "minRankRatio": args.feature_certificate_min_rank_ratio,
                    "minCellMarginRatio": (
                        args.feature_certificate_min_cell_margin_ratio
                    ),
                    "minFamilyWeightRatio": (
                        args.feature_certificate_min_family_weight_ratio
                    ),
                    "minQefDisplacementRatio": (
                        args.feature_certificate_min_qef_displacement_ratio
                    ),
                },
                "integrationMode": args.integration_mode,
                "paperNormalSource": args.paper_normal_source,
            },
            "scan": scan_metadata,
            "acceptedSurfacePoints": accepted_points,
            "integrationMs": integration_ms,
            "variants": results,
            "allHermiteEvidence": {
                "audit": tsdf_hermite_features["audit"],
                "truthGeometry": all_feature_geometry,
            },
            "appliedHermiteFeature": {
                "audit": placement_audit,
                "truthGeometry": applied_feature_geometry,
            },
            "hardGate": hard_gate,
            "passed": all(hard_gate.values()),
        }
        with (args.out / "directional_composition_report.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        lines = [
            "# Paper DMC + TSDF-Hermite QEF Strict Offline A/B",
            "",
            f"- mesh: {args.mesh.stem}",
            f"- input: {report['mode']}",
            f"- integration: {args.integration_mode}",
            f"- passed: {report['passed']}",
            "- Unity/Quest production code changed: no",
            "- frozen authority: paper DMC physical-edge and inter-cell boundary ledger",
            "- experimental change: proven single-patch cell interior feature fan only",
            "",
            "| Variant | Visible coverage 5cm | Whole-room coverage 5cm | Extra 5cm | Accuracy p95 | Boundary / 1k tri | Non-manifold | Components | Triangles |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, values in results.items():
            lines.append(
                f"| {name} | {values['visibleCoverageAt0.05m']:.4f} | "
                f"{values['wholeRoomCoverageAt0.05m']:.4f} | "
                f"{values['extraSurfaceRatioAt0.05m']:.4f} | "
                f"{values['accuracyP95m']:.4f} | "
                f"{values['boundaryEdgesPerKTriangles']:.2f} | "
                f"{values['nonManifoldEdges']} | "
                f"{values['connectedComponents']} | {values['triangles']} |"
            )
        lines.extend(["", "## Hard gate", ""])
        lines.extend(f"- {key}: {value}" for key, value in hard_gate.items())
        lines.extend(["", "## Hermite evidence audit", ""])
        lines.extend(
            f"- {key}: {value}"
            for key, value in tsdf_hermite_features["audit"].items()
        )
        lines.extend(["", "## Applied feature audit", ""])
        lines.extend(
            f"- {key}: {value}" for key, value in placement_audit.items()
        )
        lines.extend(["", "## Applied feature truth geometry", ""])
        lines.extend(
            f"- {key}: {value}"
            for key, value in applied_feature_geometry.items()
        )
        (args.out / "directional_composition_report.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "passed": report["passed"],
                    "hardGate": hard_gate,
                    "appliedFeatureAudit": placement_audit,
                    "appliedFeatureTruthGeometry": applied_feature_geometry,
                    "out": str(args.out),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.paper_baseline_only:
        soft_paper_raw_independent = extract_paper_stage_independent(
            soft, args.paper_minimum_weight, "raw"
        )
        soft_paper_filtered_independent = extract_paper_stage_independent(
            soft, args.paper_minimum_weight, "filtered"
        )
        soft_paper_voted_independent = extract_paper_stage_independent(
            soft, args.paper_minimum_weight, "voted"
        )
        soft_paper_dmc_unregularized = extract_paper_dmc(
            soft, args.paper_minimum_weight, regularize=False
        )
        soft_paper_dmc = extract_paper_dmc(
            soft, args.paper_minimum_weight, regularize=True
        )
        variants = {
            "dominant_independent": dominant_independent,
            "dominant_composed": dominant_composed,
            "soft_composed": soft_composed,
            "soft_paper_raw_independent": soft_paper_raw_independent,
            "soft_paper_filtered_independent": soft_paper_filtered_independent,
            "soft_paper_voted_independent": soft_paper_voted_independent,
            "soft_paper_dmc_unregularized": soft_paper_dmc_unregularized,
            "soft_paper_dmc": soft_paper_dmc,
        }
        results: dict[str, Any] = {}
        for name, build in variants.items():
            write_mesh(args.out / f"{name}.ply", build)
            results[name] = evaluate_mesh(
                build,
                truth,
                scene,
                observed_truth_mask,
                structure_reference,
                args.structure_band_meters,
            )
        baseline = results["dominant_independent"]
        legacy_composed = results["soft_composed"]
        raw_independent = results["soft_paper_raw_independent"]
        filtered_independent = results["soft_paper_filtered_independent"]
        voted_independent = results["soft_paper_voted_independent"]
        composed = results["soft_paper_dmc"]
        composition_coverage_loss = (
            voted_independent["coverageAt0.05m"]
            - composed["coverageAt0.05m"]
        )
        hard_gate = {
            "extraSurfaceNotWorse": composed["extraSurfaceRatioAt0.05m"] <= baseline["extraSurfaceRatioAt0.05m"] + 0.002,
            "accuracyP95NotWorse": composed["accuracyP95m"] <= baseline["accuracyP95m"] + 0.002,
            "nonManifoldNotWorse": composed["nonManifoldEdges"] <= baseline["nonManifoldEdges"],
            "boundaryDensityReduced": composed["boundaryEdgesPerKTriangles"] < baseline["boundaryEdgesPerKTriangles"],
            "coverageFloor": composed["coverageAt0.05m"] >= baseline["coverageAt0.05m"] - 0.03,
            "visibleCoverageFloor": (
                composed["visibleCoverageAt0.05m"]
                >= baseline["visibleCoverageAt0.05m"] - 0.03
            ),
            "coverageLossReducedVsLegacyComposer": (
                composed["coverageAt0.05m"]
                >= legacy_composed["coverageAt0.05m"]
            ),
            "preCompositionCoverageFloor": (
                voted_independent["coverageAt0.05m"]
                >= baseline["coverageAt0.05m"] - 0.03
            ),
            "algorithm2CompositionCoverageLossBounded": (
                composition_coverage_loss <= 0.005
            ),
            "composedEdgesHaveMeasuredPositions": (
                composed["audit"]["paper_unmeasured_edge_deferred_triangles"] == 0
                and composed["audit"]["paper_unresolved_edge_slots"] == 0
            ),
        }
        report = {
            "schema": "scancover.directional_tsdf_composition.paper_baseline.v4",
            "mesh": str(args.mesh),
            "degradationModel": str(args.degradation_model),
            "mode": "ideal" if args.ideal_depth else "quest_guarded",
            "scope": "exact paper DMC composition and classic MC extraction; legacy custom composer retained only as A/B control; Unity disabled",
            "parameters": {
                "frames": args.frames,
                "width": args.width,
                "height": args.height,
                "voxelMeters": args.voxel,
                "truncationMeters": args.sdf_trunc,
                "minimumWeight": args.minimum_weight,
                "paperMinimumWeight": args.paper_minimum_weight,
                "softDirectionThreshold": args.soft_direction_threshold,
                "validGradientDot": args.valid_gradient_dot,
                "parallelDot": args.parallel_dot,
                "edgeMergeVoxelRatio": args.edge_merge_voxel_ratio,
                "structureBandMeters": args.structure_band_meters,
                "integrationMode": args.integration_mode,
                "projectiveBlockVoxels": args.projective_block_voxels,
                "cameraPathCheckpoints": path_checkpoints,
                "paperNormalSource": args.paper_normal_source,
                "paperNormalPerturbation": {
                    "angularNoiseSigmaDegrees": args.paper_normal_angular_noise_sigma_degrees,
                    "dropoutProbability": args.paper_normal_dropout_probability,
                    "edgeAngularNoiseSigmaDegrees": args.paper_normal_edge_angular_noise_sigma_degrees,
                    "edgeDropoutProbability": args.paper_normal_edge_dropout_probability,
                    "edgeDepthFactor": args.paper_normal_perturbation_edge_depth_factor,
                    "edgeDepthFloorMeters": args.paper_normal_perturbation_edge_depth_floor,
                    "seed": args.paper_normal_perturbation_seed,
                },
            },
            "scan": scan_metadata,
            "coverageLedgers": {
                "definition": {
                    "visible": "truth samples in a valid depth pixel and unobstructed from at least one camera",
                    "wholeRoom": "all uniformly sampled truth-mesh points",
                    "thresholdMeters": 0.05,
                },
                "observedTruthSamples": int(np.sum(observed_truth_mask)),
                "truthSamples": int(len(observed_truth_mask)),
                "observedTruthRatioOfRoom": float(np.mean(observed_truth_mask)) if len(observed_truth_mask) else 0.0,
                "visibilityOcclusionTolerance": "max(0.002m, range*0.001)",
            },
            "acceptedSurfacePoints": accepted_points,
            "integrationMs": integration_ms,
            "dominantDirectionWrites": dominant.direction_writes.tolist(),
            "softDirectionWrites": soft.direction_writes.tolist(),
            "projectiveAudit": {
                "candidateBlocks": soft.projective_candidate_blocks,
                "candidateVoxels": soft.projective_candidate_voxels,
                "visibleVoxels": soft.projective_visible_voxels,
                "validDepthVoxels": soft.projective_valid_depth_voxels,
                "behindSurfaceTruncationRejects": soft.projective_truncation_rejects,
                "integratedVoxelDirectionWrites": soft.voxel_updates,
            },
            "paperNormalAudit": {
                "estimation": paper_normal_estimation_audit,
                "perturbation": paper_normal_perturbation_summary,
            },
            "structureReferencePoints": {
                label: int(len(points)) for label, points in structure_reference.items()
            },
            "variants": results,
            "stageAttribution": {
                "fullCornerReadinessCoverageLossAt0.05m": (
                    baseline["coverageAt0.05m"]
                    - raw_independent["coverageAt0.05m"]
                ),
                "intraDirectionFilterCoverageLossAt0.05m": (
                    raw_independent["coverageAt0.05m"]
                    - filtered_independent["coverageAt0.05m"]
                ),
                "interDirectionVoteCoverageLossAt0.05m": (
                    filtered_independent["coverageAt0.05m"]
                    - voted_independent["coverageAt0.05m"]
                ),
                "algorithm2CoverageLossAt0.05m": composition_coverage_loss,
                "algorithm2VisibleCoverageLossAt0.05m": (
                    voted_independent["visibleCoverageAt0.05m"]
                    - composed["visibleCoverageAt0.05m"]
                ),
                "partialCornerBaselineCoverageAt0.05m": baseline["coverageAt0.05m"],
                "fullCornerRawCoverageAt0.05m": raw_independent["coverageAt0.05m"],
                "postIntraFilterCoverageAt0.05m": filtered_independent["coverageAt0.05m"],
                "preAlgorithm2CoverageAt0.05m": voted_independent["coverageAt0.05m"],
                "postAlgorithm2CoverageAt0.05m": composed["coverageAt0.05m"],
                "baselineToPreAlgorithm2CoverageDelta": (
                    voted_independent["coverageAt0.05m"]
                    - baseline["coverageAt0.05m"]
                ),
            },
            "hardGate": hard_gate,
            "passed": all(hard_gate.values()),
        }
        with (args.out / "directional_composition_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        lines = [
            "# ScanCover Directional TSDF Paper Baseline",
            "",
            f"- mesh: {args.mesh.stem}",
            f"- input: {report['mode']}",
            f"- integration: {args.integration_mode}",
            f"- passed: {report['passed']}",
            "- custom QEF/Hermite extensions: disabled",
            "",
            "| Variant | Visible coverage 5cm | Whole-room coverage 5cm | Extra 5cm | Accuracy p95 | Boundary / 1k tri | Non-manifold | Triangles |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, values in results.items():
            lines.append(
                f"| {name} | {values['visibleCoverageAt0.05m']:.4f} | "
                f"{values['wholeRoomCoverageAt0.05m']:.4f} | "
                f"{values['extraSurfaceRatioAt0.05m']:.4f} | {values['accuracyP95m']:.4f} | "
                f"{values['boundaryEdgesPerKTriangles']:.2f} | {values['nonManifoldEdges']} | "
                f"{values['triangles']} |"
            )
        lines.extend(["", "## Hard gate", ""])
        lines.extend(f"- {key}: {value}" for key, value in hard_gate.items())
        (args.out / "directional_composition_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps({
            "passed": report["passed"],
            "hardGate": hard_gate,
            "out": str(args.out),
        }, ensure_ascii=False))
        return 0
    soft_composed_feature_qef = extract_feature_qef_shadow(
        soft_composed,
        args.voxel,
        args.feature_angle_degrees,
        args.feature_neighborhood_voxel_ratio,
        args.feature_max_move_voxel_ratio,
        args.feature_min_family_support_ratio,
        args.feature_rank_ratio,
    )
    tsdf_hermite_features = extract_tsdf_hermite_feature_points(
        soft,
        args.minimum_weight,
        args.valid_gradient_dot,
        args.feature_angle_degrees,
        args.feature_min_family_support_ratio,
        args.feature_rank_ratio,
    )
    write_hermite_feature_points(args.out, tsdf_hermite_features)
    hermite_geometry = evaluate_hermite_feature_points(tsdf_hermite_features, scene)
    hermite_gate = {
        "featureCandidatesFound": (
            int(tsdf_hermite_features["audit"].get("featureCandidates", 0)) > 0
        ),
        "featurePointsAccepted": int(hermite_geometry.get("acceptedFeaturePoints", 0)) > 0,
        "truthImprovementMajority": (
            float(hermite_geometry.get("truthImprovedFraction", 0.0)) >= 0.5
        ),
        "truthMeanNotWorse": (
            hermite_geometry.get("truthDistanceQefMeters", {}).get("mean", float("inf"))
            <= hermite_geometry.get("truthDistanceBaselineMeters", {}).get("mean", 0.0)
            + 0.00025
        ),
        "truthP95NotWorse": (
            hermite_geometry.get("truthDistanceQefMeters", {}).get("p95", float("inf"))
            <= hermite_geometry.get("truthDistanceBaselineMeters", {}).get("p95", 0.0)
            + 0.001
        ),
    }
    hermite_dual_shadow = extract_tsdf_hermite_dual_mesh(
        soft,
        tsdf_hermite_features,
        args.minimum_weight,
        args.valid_gradient_dot,
        args.parallel_dot,
        args.edge_merge_voxel_ratio,
        args.feature_rank_ratio,
    )
    hermite_ledger_dual_shadow = extract_tsdf_hermite_ledger_dual_mesh(
        soft,
        tsdf_hermite_features,
        args.minimum_weight,
        args.valid_gradient_dot,
        args.parallel_dot,
        args.edge_merge_voxel_ratio,
        args.feature_rank_ratio,
    )
    variants = {
        "dominant_independent": dominant_independent,
        "dominant_composed": dominant_composed,
        "soft_composed": soft_composed,
        "soft_composed_feature_qef": soft_composed_feature_qef,
        "tsdf_hermite_dual_shadow": hermite_dual_shadow,
        "tsdf_hermite_ledger_dual_shadow": hermite_ledger_dual_shadow,
    }
    results: dict[str, Any] = {}
    for name, build in variants.items():
        write_mesh(args.out / f"{name}.ply", build)
        results[name] = evaluate_mesh(
            build,
            truth,
            scene,
            observed_truth_mask,
            structure_reference,
            args.structure_band_meters,
        )

    baseline = results["dominant_independent"]
    composed = results["soft_composed"]
    feature_qef = results["soft_composed_feature_qef"]
    hermite_dual = results["tsdf_hermite_dual_shadow"]
    hermite_ledger_dual = results["tsdf_hermite_ledger_dual_shadow"]
    feature_audit = feature_qef.get("metadata", {}).get("featureQef", {})
    feature_geometry_delta = evaluate_feature_geometry_delta(
        soft_composed,
        soft_composed_feature_qef,
        scene,
    )
    feature_qef["featureGeometryDelta"] = feature_geometry_delta
    hard_gate = {
        "extraSurfaceNotWorse": composed["extraSurfaceRatioAt0.05m"] <= baseline["extraSurfaceRatioAt0.05m"] + 0.002,
        "accuracyP95NotWorse": composed["accuracyP95m"] <= baseline["accuracyP95m"] + 0.002,
        "nonManifoldNotWorse": composed["nonManifoldEdges"] <= baseline["nonManifoldEdges"],
        "boundaryDensityReduced": composed["boundaryEdgesPerKTriangles"] < baseline["boundaryEdgesPerKTriangles"],
        "coverageFloor": composed["coverageAt0.05m"] >= baseline["coverageAt0.05m"] - 0.03,
        "visibleCoverageFloor": (
            composed["visibleCoverageAt0.05m"]
            >= baseline["visibleCoverageAt0.05m"] - 0.03
        ),
    }
    feature_gate = {
        "topologyPreserved": (
            feature_qef["triangles"] == composed["triangles"]
            and feature_qef["boundaryEdges"] == composed["boundaryEdges"]
            and feature_qef["nonManifoldEdges"] == composed["nonManifoldEdges"]
        ),
        "featureCandidatesFound": int(feature_audit.get("candidates", 0)) > 0,
        "featureVerticesMoved": int(feature_audit.get("movedVertices", 0)) > 0,
        "qefResidualImproved": float(feature_audit.get("qefResidualRatioP95", 1.0)) < 0.98,
        "extraSurfaceNotWorse": (
            feature_qef["extraSurfaceRatioAt0.05m"]
            <= composed["extraSurfaceRatioAt0.05m"] + 0.002
        ),
        "accuracyP95NotWorse": feature_qef["accuracyP95m"] <= composed["accuracyP95m"] + 0.002,
        "coverageFloor": feature_qef["coverageAt0.05m"] >= composed["coverageAt0.05m"] - 0.01,
        "movedTruthMeanNotWorse": (
            feature_geometry_delta.get("truthDistanceAfterMeters", {}).get("mean", float("inf"))
            <= feature_geometry_delta.get("truthDistanceBeforeMeters", {}).get("mean", 0.0)
            + 0.00025
        ),
        "movedTruthImprovementMajority": (
            float(feature_geometry_delta.get("truthImprovedFraction", 0.0)) >= 0.5
        ),
        "severeFoldoverIncreaseBounded": (
            float(feature_geometry_delta.get("severeFoldoverRateDelta", float("inf"))) <= 0.005
        ),
    }
    dual_gate = {
        "hermiteEvidencePassed": all(hermite_gate.values()),
        "meshGenerated": hermite_dual["triangles"] > 0,
        "nonManifoldFree": hermite_dual["nonManifoldEdges"] == 0,
        "extraSurfaceNotWorse": (
            hermite_dual["extraSurfaceRatioAt0.05m"]
            <= composed["extraSurfaceRatioAt0.05m"] + 0.002
        ),
        "accuracyP95NotWorse": (
            hermite_dual["accuracyP95m"] <= composed["accuracyP95m"] + 0.002
        ),
        "coverageFloor": (
            hermite_dual["coverageAt0.05m"] >= composed["coverageAt0.05m"] - 0.03
        ),
        "boundaryDensityNotWorse": (
            hermite_dual["boundaryEdgesPerKTriangles"]
            <= composed["boundaryEdgesPerKTriangles"] * 1.05
        ),
        "significantComponentsBounded": (
            hermite_dual.get("significantComponents50Triangles", 0)
            <= composed.get("significantComponents50Triangles", 0) + 2
        ),
    }
    ledger_dual_gate = {
        "hermiteEvidencePassed": all(hermite_gate.values()),
        "meshGenerated": hermite_ledger_dual["triangles"] > 0,
        "nonManifoldFree": hermite_ledger_dual["nonManifoldEdges"] == 0,
        "extraSurfaceNotWorse": (
            hermite_ledger_dual["extraSurfaceRatioAt0.05m"]
            <= composed["extraSurfaceRatioAt0.05m"] + 0.002
        ),
        "accuracyP95NotWorse": (
            hermite_ledger_dual["accuracyP95m"] <= composed["accuracyP95m"] + 0.002
        ),
        "coverageFloor": (
            hermite_ledger_dual["coverageAt0.05m"]
            >= composed["coverageAt0.05m"] - 0.03
        ),
        "boundaryDensityNotWorse": (
            hermite_ledger_dual["boundaryEdgesPerKTriangles"]
            <= composed["boundaryEdgesPerKTriangles"] * 1.05
        ),
        "significantComponentsBounded": (
            hermite_ledger_dual.get("significantComponents50Triangles", 0)
            <= composed.get("significantComponents50Triangles", 0) + 2
        ),
    }
    report = {
        "schema": "scancover.directional_tsdf_composition.v6",
        "mesh": str(args.mesh),
        "degradationModel": str(args.degradation_model),
        "mode": "ideal" if args.ideal_depth else "quest_guarded",
        "parameters": {
            "frames": args.frames,
            "width": args.width,
            "height": args.height,
            "voxelMeters": args.voxel,
            "truncationMeters": args.sdf_trunc,
            "minimumWeight": args.minimum_weight,
            "softDirectionThreshold": args.soft_direction_threshold,
            "validGradientDot": args.valid_gradient_dot,
            "parallelDot": args.parallel_dot,
            "edgeMergeVoxelRatio": args.edge_merge_voxel_ratio,
            "featureAngleDegrees": args.feature_angle_degrees,
            "featureNeighborhoodVoxelRatio": args.feature_neighborhood_voxel_ratio,
            "featureMaxMoveVoxelRatio": args.feature_max_move_voxel_ratio,
            "featureMinFamilySupportRatio": args.feature_min_family_support_ratio,
            "featureRankRatio": args.feature_rank_ratio,
            "structureBandMeters": args.structure_band_meters,
            "integrationMode": args.integration_mode,
            "directionWeightSemantics": (
                "raw_normal_axis_dot_above_threshold"
                if args.integration_mode in (
                    "projective",
                    "normal-raycast",
                    "paper-normal-raycast",
                )
                else "legacy_normalized_overlap_membership"
            ),
            "paperDirectionThreshold": PAPER_DIRECTION_THRESHOLD,
            "paperNormalRadius": args.paper_normal_radius,
            "paperNormalDepthChangeFactor": args.paper_normal_depth_change_factor,
            "paperNormalDepthChangeFloorMeters": args.paper_normal_depth_change_floor,
            "paperNormalBilateralRadius": args.paper_normal_bilateral_radius,
            "paperNormalSource": args.paper_normal_source,
            "paperNormalPerturbation": {
                "angularNoiseSigmaDegrees": args.paper_normal_angular_noise_sigma_degrees,
                "dropoutProbability": args.paper_normal_dropout_probability,
                "edgeAngularNoiseSigmaDegrees": args.paper_normal_edge_angular_noise_sigma_degrees,
                "edgeDropoutProbability": args.paper_normal_edge_dropout_probability,
                "edgeDepthFactor": args.paper_normal_perturbation_edge_depth_factor,
                "edgeDepthFloorMeters": args.paper_normal_perturbation_edge_depth_floor,
                "seed": args.paper_normal_perturbation_seed,
            },
            "paperDepthWeight": "Nguyen_sigma_ratio_times_inverse_depth_squared",
            "paperAngleWeight": "cosine_surface_normal_to_view",
            "projectiveBlockVoxels": args.projective_block_voxels,
            "cameraPathCheckpoints": path_checkpoints,
        },
        "scan": scan_metadata,
        "coverageLedgers": {
            "definition": {
                "visible": "truth samples in a valid depth pixel and unobstructed from at least one camera",
                "wholeRoom": "all uniformly sampled truth-mesh points",
                "thresholdMeters": 0.05,
            },
            "observedTruthSamples": int(np.sum(observed_truth_mask)),
            "truthSamples": int(len(observed_truth_mask)),
            "observedTruthRatioOfRoom": float(np.mean(observed_truth_mask)) if len(observed_truth_mask) else 0.0,
            "visibilityOcclusionTolerance": "max(0.002m, range*0.001)",
        },
        "strictInputAudit": {
            "cameraPathSha256": camera_path_hasher.hexdigest(),
            "observationDepthSequenceSha256": observation_hasher.hexdigest(),
            "normalSequenceSha256": normal_hasher.hexdigest(),
            "truthSamplesSha256": truth_sample_sha256,
            "degradedDepthValidPixels": degraded_depth_valid_pixels,
            "selectedDepthValidPixels": degraded_depth_valid_pixels,
        },
        "acceptedSurfacePoints": accepted_points,
        "integrationMs": integration_ms,
        "dominantDirectionWrites": dominant.direction_writes.tolist(),
        "softDirectionWrites": soft.direction_writes.tolist(),
        "projectiveAudit": {
            "candidateBlocks": soft.projective_candidate_blocks,
            "candidateVoxels": soft.projective_candidate_voxels,
            "visibleVoxels": soft.projective_visible_voxels,
            "validDepthVoxels": soft.projective_valid_depth_voxels,
            "behindSurfaceTruncationRejects": soft.projective_truncation_rejects,
            "integratedVoxelDirectionWrites": soft.voxel_updates,
        },
        "paperNormalAudit": {
            "estimation": paper_normal_estimation_audit,
            "perturbation": paper_normal_perturbation_summary,
            "raycast": {
                "rays": soft.paper_normal_rays,
                "traversedVoxels": soft.paper_traversed_voxels,
                "integratedVoxels": soft.paper_integrated_voxels,
                "depthWeightMean": (
                    soft.paper_depth_weight_sum / soft.paper_normal_rays
                    if soft.paper_normal_rays else 0.0
                ),
                "angleWeightMean": (
                    soft.paper_angle_weight_sum / soft.paper_normal_rays
                    if soft.paper_normal_rays else 0.0
                ),
                "combinedDirectionalWeightSum": soft.paper_combined_weight_sum,
                "integratedVoxelDirectionWrites": soft.voxel_updates,
            },
        },
        "structureReferencePoints": {
            label: int(len(points)) for label, points in structure_reference.items()
        },
        "variants": results,
        "hardGate": hard_gate,
        "featureGate": feature_gate,
        "featureShadowPassed": all(feature_gate.values()),
        "tsdfHermiteFeature": {
            "audit": tsdf_hermite_features["audit"],
            "geometry": hermite_geometry,
            "gate": hermite_gate,
            "passed": all(hermite_gate.values()),
        },
        "tsdfHermiteDualMesh": {
            "gate": dual_gate,
            "passed": all(dual_gate.values()),
        },
        "tsdfHermiteLedgerDualMesh": {
            "gate": ledger_dual_gate,
            "passed": all(ledger_dual_gate.values()),
        },
        "passed": all(hard_gate.values()),
    }
    with (args.out / "directional_composition_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    lines = [
        "# ScanCover Directional TSDF Composition Validation",
        "",
        f"- mesh: {args.mesh.stem}",
        f"- input: {report['mode']}",
        f"- integration: {args.integration_mode}",
        f"- accepted surface points: {accepted_points}",
        f"- directional composition passed: {report['passed']}",
        f"- feature QEF shadow passed: {report['featureShadowPassed']}",
        f"- TSDF Hermite evidence passed: {report['tsdfHermiteFeature']['passed']}",
        f"- TSDF Hermite dual mesh passed: {report['tsdfHermiteDualMesh']['passed']}",
        f"- shared-edge ledger dual mesh passed: {report['tsdfHermiteLedgerDualMesh']['passed']}",
        "",
        "| Variant | Visible coverage 5cm | Whole-room coverage 5cm | Extra 5cm | Accuracy p95 | Boundary / 1k tri | Non-manifold | Triangles | Extract ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in results.items():
        lines.append(
            f"| {name} | {values['visibleCoverageAt0.05m']:.4f} | "
            f"{values['wholeRoomCoverageAt0.05m']:.4f} | "
            f"{values['extraSurfaceRatioAt0.05m']:.4f} | {values['accuracyP95m']:.4f} | "
            f"{values['boundaryEdgesPerKTriangles']:.2f} | {values['nonManifoldEdges']} | "
            f"{values['triangles']} | {values['extractionMs']:.1f} |"
        )
    lines.extend(["", "## Hard gate", ""])
    lines.extend(f"- {key}: {value}" for key, value in hard_gate.items())
    lines.extend(["", "## Feature QEF shadow gate", ""])
    lines.extend(f"- {key}: {value}" for key, value in feature_gate.items())
    if feature_audit:
        lines.extend(["", "## Feature QEF audit", ""])
        lines.extend(f"- {key}: {value}" for key, value in feature_audit.items())
    lines.extend(["", "## TSDF Hermite feature gate", ""])
    lines.extend(f"- {key}: {value}" for key, value in hermite_gate.items())
    lines.extend(["", "## TSDF Hermite feature audit", ""])
    lines.extend(
        f"- {key}: {value}" for key, value in tsdf_hermite_features["audit"].items()
    )
    lines.extend(["", "## TSDF Hermite truth geometry", ""])
    lines.extend(f"- {key}: {value}" for key, value in hermite_geometry.items())
    lines.extend(["", "## TSDF Hermite dual mesh gate", ""])
    lines.extend(f"- {key}: {value}" for key, value in dual_gate.items())
    lines.extend(["", "## Shared-edge ledger dual mesh gate", ""])
    lines.extend(f"- {key}: {value}" for key, value in ledger_dual_gate.items())
    if args.integration_mode == "projective":
        lines.extend(["", "## Projective integration audit", ""])
        lines.extend(
            f"- {key}: {value}" for key, value in report["projectiveAudit"].items()
        )
    if args.integration_mode == "paper-normal-raycast":
        lines.extend(["", "## Paper normal-directed integration audit", ""])
        lines.extend(
            f"- normal.{key}: {value}"
            for key, value in report["paperNormalAudit"]["estimation"].items()
        )
        lines.extend(
            f"- raycast.{key}: {value}"
            for key, value in report["paperNormalAudit"]["raycast"].items()
        )
    (args.out / "directional_composition_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "featureShadowPassed": report["featureShadowPassed"],
        "tsdfHermiteFeaturePassed": report["tsdfHermiteFeature"]["passed"],
        "tsdfHermiteDualMeshPassed": report["tsdfHermiteDualMesh"]["passed"],
        "tsdfHermiteLedgerDualMeshPassed": report["tsdfHermiteLedgerDualMesh"]["passed"],
        "hardGate": hard_gate,
        "featureGate": feature_gate,
        "tsdfHermiteGate": hermite_gate,
        "tsdfHermiteDualMeshGate": dual_gate,
        "tsdfHermiteLedgerDualMeshGate": ledger_dual_gate,
        "out": str(args.out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
