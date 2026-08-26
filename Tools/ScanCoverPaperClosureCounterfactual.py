#!/usr/bin/env python3
"""Counterfactual paper-cell closure audit from stable GunGel evidence.

This is deliberately narrower than a full CPU replay of VolumeIntegration.compute.
The replay session currently seals the final stable GunGel candidates and the
published paper front, but it does not seal the final TSDF volume or the 12
persistent paper-edge states.  Consequently this tool must not claim bit-exact
GPU replay.  Instead it asks a falsifiable question:

    If the stable GunGel candidates are treated as the authoritative local
    surface evidence, which blank 10 cm paper cells already contain enough
    coherent evidence to form a closed, supported local polygon?

The local candidate is subjected to the same structural semantics used by
SupportTruthExtract.compute: >=3 crossings, no 1/3-crossing cube face,
same-layer separation, dimensionless sliver rejection, and three-point normal
support.  The resulting ledger separates evidence shortage from a paper
publication-contract bottleneck without changing production code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


EDGE_CORNERS = np.asarray(
    (
        (0, 1), (3, 2), (4, 5), (7, 6),
        (0, 3), (1, 2), (4, 7), (5, 6),
        (0, 4), (1, 5), (3, 7), (2, 6),
    ),
    dtype=np.int32,
)
FACE_EDGES = np.asarray(
    (
        (0, 5, 1, 4),
        (2, 7, 3, 6),
        (0, 9, 2, 8),
        (1, 11, 3, 10),
        (4, 10, 6, 8),
        (5, 11, 7, 9),
    ),
    dtype=np.int32,
)
CORNER_OFFSETS = np.asarray(
    (
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ),
    dtype=np.float64,
)

SAME_LAYER_NORMAL_COS = 0.82
SAME_LAYER_GAP_FRACTION = 0.30
CANDIDATE_THICKNESS_RATIO = 0.05
TRIANGLE_NORMAL_SUPPORT_COS = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit paper closure counterfactually from stable GunGel candidates."
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--neighbour-radius-cells",
        type=float,
        default=1.75,
        help="Local stable-candidate search radius in paper-cell widths.",
    )
    parser.add_argument(
        "--minimum-local-candidates",
        type=int,
        default=3,
        help="Minimum local stable candidates needed to estimate one surface hypothesis.",
    )
    return parser.parse_args()


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(lengths, 1e-12)


def parse_voxel_count(value: object) -> np.ndarray:
    if isinstance(value, str):
        values = value.removeprefix("int3(").removesuffix(")").split(",")
        return np.asarray([int(item.strip()) for item in values], dtype=np.int32)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return np.asarray(value, dtype=np.int32)
    raise ValueError(f"Unsupported voxel count: {value!r}")


def load_configuration(session: Path) -> dict[str, object]:
    config = json.loads((session / "production_config.json").read_text(encoding="utf-8-sig"))
    volume = config["components"]["Genesis.RoomScan.VolumeIntegrator"]
    mesh = config["components"]["Genesis.RoomScan.MeshExtractor"]
    vox_count = parse_voxel_count(volume["voxelCount"])
    vox_size = float(volume["voxelSize"])
    stride = int(mesh["supportTruthStride"])
    sample_count = np.maximum(1, (vox_count - 1) // stride)
    topology_count = np.maximum(1, sample_count - 1)
    return {
        "voxelCount": vox_count,
        "voxelSize": vox_size,
        "stride": stride,
        "cellSize": vox_size * stride,
        "sampleCount": sample_count,
        "topologyCount": topology_count,
    }


def load_stable_candidates(path: Path) -> dict[str, np.ndarray]:
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    support: list[float] = []
    sigma: list[float] = []
    observations: list[float] = []
    stable_ids: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("state", "").strip().lower() != "stable":
                continue
            positions.append(
                (float(row["center_x_m"]), float(row["center_y_m"]), float(row["center_z_m"]))
            )
            normals.append(
                (float(row["normal_x"]), float(row["normal_y"]), float(row["normal_z"]))
            )
            support.append(float(row["effective_support"]))
            sigma.append(float(row["sigma_mm"]))
            observations.append(float(row["observation_count"]))
            stable_ids.append(int(row["stable_id"]))
    if not positions:
        raise RuntimeError(f"No stable candidates in {path}")
    return {
        "positions": np.asarray(positions, dtype=np.float64),
        "normals": normalize_rows(np.asarray(normals, dtype=np.float64)),
        "support": np.asarray(support, dtype=np.float64),
        "sigmaMm": np.asarray(sigma, dtype=np.float64),
        "observations": np.asarray(observations, dtype=np.float64),
        "stableIds": np.asarray(stable_ids, dtype=np.int64),
    }


def load_owned_paper(path: Path) -> dict[tuple[int, int, int], dict[str, str]]:
    result: dict[tuple[int, int, int], dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["owned"]) == 0:
                continue
            result[(int(row["x"]), int(row["y"]), int(row["z"]))] = row
    return result


def sample_coordinate(position: np.ndarray, cfg: dict[str, object]) -> np.ndarray:
    vox_count = np.asarray(cfg["voxelCount"], dtype=np.float64)
    return (position / float(cfg["voxelSize"]) + vox_count * 0.5 - 0.5) / int(cfg["stride"])


def cell_minimum(cell: np.ndarray, cfg: dict[str, object]) -> np.ndarray:
    vox_count = np.asarray(cfg["voxelCount"], dtype=np.float64)
    return (
        cell.astype(np.float64) * int(cfg["stride"]) + 0.5 - vox_count * 0.5
    ) * float(cfg["voxelSize"])


def fit_primary_layer(
    center: np.ndarray,
    indices: np.ndarray,
    candidates: dict[str, np.ndarray],
    cell_size: float,
) -> dict[str, object]:
    positions = candidates["positions"][indices]
    normals = candidates["normals"][indices]
    support = candidates["support"][indices]
    anchor_local = int(np.argmin(np.linalg.norm(positions - center, axis=1)))
    anchor_position = positions[anchor_local]
    anchor_normal = normals[anchor_local]
    alignment = np.abs(normals @ anchor_normal)
    signed_anchor = (positions - anchor_position) @ anchor_normal
    layer_limit = max(cell_size * 0.60, 0.03)
    primary_mask = (alignment >= SAME_LAYER_NORMAL_COS) & (np.abs(signed_anchor) <= layer_limit)
    primary_indices = indices[primary_mask]
    primary_positions = candidates["positions"][primary_indices]
    primary_normals = candidates["normals"][primary_indices]
    primary_support = candidates["support"][primary_indices]

    result: dict[str, object] = {
        "primaryIndices": primary_indices,
        "multiOrientationCount": int(np.count_nonzero(alignment < SAME_LAYER_NORMAL_COS)),
        "parallelSeparatedCount": int(
            np.count_nonzero((alignment >= SAME_LAYER_NORMAL_COS) & (np.abs(signed_anchor) > layer_limit))
        ),
    }
    if len(primary_positions) < 3:
        return result

    weights = np.clip(primary_support, 1.0, np.percentile(primary_support, 85))
    weights = weights / np.sum(weights)
    centroid = np.sum(primary_positions * weights[:, None], axis=0)
    centered = primary_positions - centroid
    covariance = (centered * weights[:, None]).T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    plane_normal = eigenvectors[:, 0]

    aligned_normals = primary_normals.copy()
    signs = np.sign(aligned_normals @ plane_normal)
    signs[signs == 0.0] = 1.0
    aligned_normals *= signs[:, None]
    evidence_normal = np.sum(aligned_normals * weights[:, None], axis=0)
    if np.linalg.norm(evidence_normal) > 1e-9 and np.dot(plane_normal, evidence_normal) < 0.0:
        plane_normal = -plane_normal
    residuals = np.abs((primary_positions - centroid) @ plane_normal)
    normal_support = np.abs(primary_normals @ plane_normal)
    result.update(
        {
            "centroid": centroid,
            "normal": plane_normal,
            "residualP50": float(np.percentile(residuals, 50)),
            "residualP95": float(np.percentile(residuals, 95)),
            "normalSupportMin": float(np.min(normal_support)),
            "normalSupportP50": float(np.percentile(normal_support, 50)),
        }
    )
    return result


def intersect_plane_with_cell(
    cell: np.ndarray,
    centroid: np.ndarray,
    normal: np.ndarray,
    cfg: dict[str, object],
) -> tuple[int, np.ndarray, list[int]]:
    minimum = cell_minimum(cell, cfg)
    corners = minimum + CORNER_OFFSETS * float(cfg["cellSize"])
    signed = (corners - centroid) @ normal
    epsilon = max(float(cfg["cellSize"]) * 1e-6, 1e-9)
    points: list[np.ndarray] = []
    edges: list[int] = []
    mask = 0
    for edge_index, (a_index, b_index) in enumerate(EDGE_CORNERS):
        a = float(signed[a_index])
        b = float(signed[b_index])
        if abs(a) <= epsilon and abs(b) <= epsilon:
            continue
        if (a > epsilon and b > epsilon) or (a < -epsilon and b < -epsilon):
            continue
        denominator = a - b
        if abs(denominator) <= epsilon:
            continue
        t = float(np.clip(a / denominator, 0.0, 1.0))
        points.append(corners[a_index] + (corners[b_index] - corners[a_index]) * t)
        edges.append(edge_index)
        mask |= 1 << edge_index
    return mask, np.asarray(points, dtype=np.float64).reshape((-1, 3)), edges


def closure_classification(
    edge_mask: int,
    intersection_points: np.ndarray,
    evidence_normals: np.ndarray,
    parallel_separated_count: int,
) -> tuple[str, list[int], dict[str, float]]:
    edge_count = int(edge_mask.bit_count())
    face_counts = [sum((edge_mask >> int(edge)) & 1 for edge in face) for face in FACE_EDGES]
    metrics = {"edgeCount": float(edge_count)}
    if edge_count < 3:
        return "crossing_shortage", face_counts, metrics
    if any(count in (1, 3) for count in face_counts):
        return "partial_face_1_or_3", face_counts, metrics
    if parallel_separated_count > 0:
        return "cross_layer_candidate", face_counts, metrics

    maximum_span_sq = 0.0
    maximum_area_sq = 0.0
    maximum_area_vector = np.zeros(3, dtype=np.float64)
    for a in range(len(intersection_points)):
        for b in range(a + 1, len(intersection_points)):
            delta = intersection_points[b] - intersection_points[a]
            maximum_span_sq = max(maximum_span_sq, float(np.dot(delta, delta)))
    if len(intersection_points) >= 3:
        point_a = intersection_points[0]
        distances = np.sum((intersection_points - point_a) ** 2, axis=1)
        point_b = intersection_points[int(np.argmax(distances))]
        for point_c in intersection_points:
            area_vector = np.cross(point_b - point_a, point_c - point_a)
            area_sq = float(np.dot(area_vector, area_vector))
            if area_sq > maximum_area_sq:
                maximum_area_sq = area_sq
                maximum_area_vector = area_vector
    scaled_floor = maximum_span_sq * maximum_span_sq * CANDIDATE_THICKNESS_RATIO**2
    metrics.update(
        {
            "maximumSpanSq": maximum_span_sq,
            "maximumAreaSq": maximum_area_sq,
            "scaledAreaFloor": scaled_floor,
        }
    )
    if maximum_area_sq < max(1e-10, scaled_floor):
        return "sliver_candidate", face_counts, metrics
    if edge_count == 3:
        face_normal = maximum_area_vector / math.sqrt(max(maximum_area_sq, 1e-12))
        minimum_support = float(np.min(np.abs(evidence_normals @ face_normal)))
        metrics["triangleNormalSupportMin"] = minimum_support
        if minimum_support < TRIANGLE_NORMAL_SUPPORT_COS:
            return "unsupported_face", face_counts, metrics
    return "counterfactual_publishable", face_counts, metrics


def polygon_order(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    center = np.mean(points, axis=0)
    tangent = points[0] - center
    if np.linalg.norm(tangent) < 1e-9:
        tangent = np.cross(normal, np.asarray((1.0, 0.0, 0.0)))
    if np.linalg.norm(tangent) < 1e-9:
        tangent = np.cross(normal, np.asarray((0.0, 1.0, 0.0)))
    tangent /= max(np.linalg.norm(tangent), 1e-12)
    bitangent = np.cross(normal, tangent)
    local = points - center
    angles = np.arctan2(local @ bitangent, local @ tangent)
    return np.argsort(angles)


def write_mesh(path: Path, polygons: list[tuple[np.ndarray, np.ndarray]]) -> None:
    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    for points, normal in polygons:
        if len(points) < 3:
            continue
        ordered = points[polygon_order(points, normal)]
        base = len(vertices)
        vertices.extend(ordered)
        for index in range(1, len(ordered) - 1):
            triangles.append((base, base + index, base + index + 1))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64).reshape((-1, 3)))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32).reshape((-1, 3)))
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False, compressed=False):
        raise RuntimeError(f"Failed to write {path}")


def write_centers(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False):
        raise RuntimeError(f"Failed to write {path}")


def main() -> int:
    args = parse_args()
    session = args.session.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_configuration(session)
    candidates = load_stable_candidates(
        session / "artifacts" / "gungel_candidate_audit" / "candidates.csv"
    )
    owned = load_owned_paper(session / "artifacts" / "paper_audit" / "paper_cells.csv")
    positions = candidates["positions"]
    topology_count = np.asarray(cfg["topologyCount"], dtype=np.int32)
    cell_size = float(cfg["cellSize"])
    radius = max(args.neighbour_radius_cells, 0.5) * cell_size
    tree = cKDTree(positions)

    candidate_cells: set[tuple[int, int, int]] = set()
    for position in positions:
        base = np.floor(sample_coordinate(position, cfg)).astype(np.int32)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cell = base + np.asarray((dx, dy, dz), dtype=np.int32)
                    if np.all(cell >= 0) and np.all(cell < topology_count):
                        candidate_cells.add((int(cell[0]), int(cell[1]), int(cell[2])))

    rows: list[dict[str, object]] = []
    polygons: list[tuple[np.ndarray, np.ndarray]] = []
    center_points: list[np.ndarray] = []
    center_colors: list[tuple[float, float, float]] = []
    palette = {
        "paper_owned": (0.0, 0.75, 1.0),
        "counterfactual_publishable": (0.0, 1.0, 0.15),
        "insufficient_local_candidates": (1.0, 0.82, 0.0),
        "crossing_shortage": (1.0, 0.35, 0.0),
        "partial_face_1_or_3": (1.0, 0.0, 0.0),
        "cross_layer_candidate": (1.0, 0.0, 1.0),
        "sliver_candidate": (0.65, 0.0, 1.0),
        "unsupported_face": (0.55, 0.15, 0.0),
    }

    for cell_tuple in sorted(candidate_cells, key=lambda item: (item[2], item[1], item[0])):
        cell = np.asarray(cell_tuple, dtype=np.int32)
        center = cell_minimum(cell, cfg) + cell_size * 0.5
        local_indices = np.asarray(tree.query_ball_point(center, radius), dtype=np.int64)
        paper_row = owned.get(cell_tuple)
        paper_owned = paper_row is not None
        classification = "paper_owned" if paper_owned else "insufficient_local_candidates"
        edge_mask = 0
        face_counts = [0] * 6
        intersections = np.empty((0, 3), dtype=np.float64)
        fit: dict[str, object] = {}
        closure_metrics: dict[str, float] = {}

        if len(local_indices) >= args.minimum_local_candidates:
            fit = fit_primary_layer(center, local_indices, candidates, cell_size)
            primary_indices = np.asarray(fit.get("primaryIndices", []), dtype=np.int64)
            if len(primary_indices) >= args.minimum_local_candidates and "normal" in fit:
                normal = np.asarray(fit["normal"], dtype=np.float64)
                edge_mask, intersections, _ = intersect_plane_with_cell(
                    cell, np.asarray(fit["centroid"]), normal, cfg
                )
                counterfactual, face_counts, closure_metrics = closure_classification(
                    edge_mask,
                    intersections,
                    candidates["normals"][primary_indices],
                    int(fit.get("parallelSeparatedCount", 0)),
                )
                if not paper_owned:
                    classification = counterfactual
                if counterfactual == "counterfactual_publishable" and not paper_owned:
                    polygons.append((intersections, normal))

        center_points.append(center)
        center_colors.append(palette[classification])
        rows.append(
            {
                "x": cell_tuple[0],
                "y": cell_tuple[1],
                "z": cell_tuple[2],
                "center_x_m": center[0],
                "center_y_m": center[1],
                "center_z_m": center[2],
                "paper_owned": int(paper_owned),
                "paper_hold_reason": int(paper_row["hold_reason"]) if paper_row else -1,
                "classification": classification,
                "local_candidates": len(local_indices),
                "primary_layer_candidates": len(fit.get("primaryIndices", [])),
                "multi_orientation_candidates": int(fit.get("multiOrientationCount", 0)),
                "parallel_separated_candidates": int(fit.get("parallelSeparatedCount", 0)),
                "edge_mask": edge_mask,
                "edge_count": int(edge_mask.bit_count()),
                "face_counts": "/".join(str(value) for value in face_counts),
                "plane_residual_p50_mm": float(fit.get("residualP50", math.nan)) * 1000.0,
                "plane_residual_p95_mm": float(fit.get("residualP95", math.nan)) * 1000.0,
                "normal_support_min": float(fit.get("normalSupportMin", math.nan)),
                "normal_support_p50": float(fit.get("normalSupportP50", math.nan)),
                "candidate_sigma_p50_mm": float(np.percentile(candidates["sigmaMm"][local_indices], 50))
                if len(local_indices) else math.nan,
                "candidate_support_p50": float(np.percentile(candidates["support"][local_indices], 50))
                if len(local_indices) else math.nan,
                "maximum_span_sq": closure_metrics.get("maximumSpanSq", math.nan),
                "maximum_area_sq": closure_metrics.get("maximumAreaSq", math.nan),
                "scaled_area_floor": closure_metrics.get("scaledAreaFloor", math.nan),
                "triangle_normal_support_min": closure_metrics.get("triangleNormalSupportMin", math.nan),
            }
        )

    fields = list(rows[0].keys())
    with (out / "paper_closure_counterfactual_cells.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    write_centers(
        out / "paper_closure_counterfactual_cells.ply",
        np.asarray(center_points, dtype=np.float64),
        np.asarray(center_colors, dtype=np.float64),
    )
    write_mesh(out / "counterfactual_publishable_blank_cells.ply", polygons)

    counts = Counter(str(row["classification"]) for row in rows)
    blank_rows = [row for row in rows if int(row["paper_owned"]) == 0]
    blank_counts = Counter(str(row["classification"]) for row in blank_rows)
    publishable_blank = blank_counts.get("counterfactual_publishable", 0)
    summary = {
        "schema": "scancover.paper_closure_counterfactual.v1",
        "inputSession": str(session),
        "method": {
            "kind": "stable-GunGel local-plane counterfactual",
            "bitExactGpuReplay": False,
            "reasonNotBitExact": (
                "session lacks final TSDF volume and persistent 12-edge paper evidence states"
            ),
            "cellSizeMeters": cell_size,
            "neighbourRadiusMeters": radius,
            "minimumLocalCandidates": args.minimum_local_candidates,
            "rulesMirrored": [
                "at least 3 crossings",
                "no cube face with 1 or 3 crossings",
                "parallel separated layers rejected",
                "dimensionless 5 percent sliver test",
                "three-crossing face normal support >= 0.25",
            ],
        },
        "inputs": {
            "stableGunGelCandidates": int(len(positions)),
            "ownedPaperCells": int(len(owned)),
            "candidateNeighbourhoodCells": int(len(rows)),
        },
        "allCandidateNeighbourhoodCells": dict(sorted(counts.items())),
        "blankCandidateNeighbourhoodCells": {
            "count": len(blank_rows),
            "byClassification": dict(sorted(blank_counts.items())),
            "counterfactualPublishableCount": publishable_blank,
            "counterfactualPublishableSharePct": publishable_blank * 100.0 / max(len(blank_rows), 1),
        },
        "reading": {
            "publishableBlankMeaning": (
                "stable local evidence can form a structurally valid paper polygon, yet the GPU paper front is blank"
            ),
            "nonPublishableMeaning": (
                "this counterfactual could not form a safe polygon; it does not by itself prove upstream depth absence"
            ),
            "nextExactStep": (
                "seal final TSDF corners and persistent paper-edge states, or finish full production fusion replay"
            ),
        },
        "files": {
            "cellsCsv": "paper_closure_counterfactual_cells.csv",
            "cellCentersPly": "paper_closure_counterfactual_cells.ply",
            "publishableBlankMeshPly": "counterfactual_publishable_blank_cells.ply",
        },
    }
    (out / "paper_closure_counterfactual_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
