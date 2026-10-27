#!/usr/bin/env python3
"""Locate long vertical unwelded seams on planar room-mesh walls."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_supported_creases import face_geometry
from build_spatial_scar_receipt import read_ply


WELD_QUANTIZATION_METERS = 0.0001
CHUNK_SIZE_METERS = 0.8
MINIMUM_VERTICALITY = 0.70
MINIMUM_PAIRED_NORMAL_COHERENCE = 0.97
MINIMUM_CHAIN_NORMAL_COHERENCE = 0.85
MINIMUM_VERTICAL_SPAN_METERS = 0.60
MAXIMUM_CHUNK_PLANE_DISTANCE_METERS = 0.03
MAXIMUM_TANGENTIAL_SPREAD_METERS = 0.18


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build(mesh_path):
    vertices, faces = read_ply(mesh_path)
    _, normals, _ = face_geometry(vertices, faces)
    quantized = np.rint(
        vertices / WELD_QUANTIZATION_METERS).astype(np.int64)
    welded_vertices, vertex_to_welded = np.unique(
        quantized, axis=0, return_inverse=True)
    welded_positions = (
        welded_vertices.astype(np.float64) * WELD_QUANTIZATION_METERS)

    original_edge_faces = defaultdict(list)
    welded_edge_faces = defaultdict(list)
    for triangle, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]),
                     (face[2], face[0])):
            original_edge_faces[tuple(sorted((int(a), int(b))))].append(
                triangle)
            welded_a = int(vertex_to_welded[a])
            welded_b = int(vertex_to_welded[b])
            welded_edge_faces[tuple(sorted((welded_a, welded_b)))].append(
                triangle)
    original_face_pairs = {
        tuple(sorted(triangles))
        for triangles in original_edge_faces.values()
        if len(triangles) == 2
    }

    candidate_edges = []
    for (a, b), triangle_owners in welded_edge_faces.items():
        triangle_owners = sorted(set(triangle_owners))
        if len(triangle_owners) != 2:
            continue
        if tuple(triangle_owners) in original_face_pairs:
            continue
        delta = welded_positions[b] - welded_positions[a]
        length = float(np.linalg.norm(delta))
        if length <= 1e-6:
            continue
        verticality = abs(float(delta[1])) / length
        paired_normal_coherence = abs(float(np.dot(
            normals[triangle_owners[0]], normals[triangle_owners[1]])))
        if verticality < MINIMUM_VERTICALITY:
            continue
        if paired_normal_coherence < MINIMUM_PAIRED_NORMAL_COHERENCE:
            continue
        candidate_edges.append({
            "a": a,
            "b": b,
            "triangles": triangle_owners,
            "lengthMeters": length,
            "verticality": verticality,
            "pairedNormalCoherence": paired_normal_coherence,
        })

    vertex_edges = defaultdict(list)
    for edge, record in enumerate(candidate_edges):
        vertex_edges[record["a"]].append(edge)
        vertex_edges[record["b"]].append(edge)

    seen = set()
    chains = []
    for seed in range(len(candidate_edges)):
        if seed in seen:
            continue
        pending = [seed]
        seen.add(seed)
        edge_ids = []
        welded_ids = set()
        triangle_ids = set()
        while pending:
            edge = pending.pop()
            record = candidate_edges[edge]
            edge_ids.append(edge)
            welded_ids.update((record["a"], record["b"]))
            triangle_ids.update(record["triangles"])
            for vertex in (record["a"], record["b"]):
                for neighbour in vertex_edges[vertex]:
                    if neighbour not in seen:
                        seen.add(neighbour)
                        pending.append(neighbour)

        positions = welded_positions[sorted(welded_ids)]
        chain_normals = normals[sorted(triangle_ids)]
        wall_axis = max(
            (0, 2),
            key=lambda axis: float(np.mean(np.abs(
                chain_normals[:, axis]))))
        tangent_axis = 2 if wall_axis == 0 else 0
        tangent_coordinate = float(np.median(
            positions[:, tangent_axis]))
        chunk_plane_distance = abs(
            tangent_coordinate -
            round(tangent_coordinate / CHUNK_SIZE_METERS) *
            CHUNK_SIZE_METERS)
        tangential_spread = float(np.ptp(
            positions[:, tangent_axis]))
        vertical_span = float(np.ptp(positions[:, 1]))
        normal_coherence = float(np.mean(np.abs(
            chain_normals[:, wall_axis])))
        if vertical_span < MINIMUM_VERTICAL_SPAN_METERS:
            continue
        if chunk_plane_distance > MAXIMUM_CHUNK_PLANE_DISTANCE_METERS:
            continue
        if tangential_spread > MAXIMUM_TANGENTIAL_SPREAD_METERS:
            continue
        if normal_coherence < MINIMUM_CHAIN_NORMAL_COHERENCE:
            continue
        chains.append({
            "edgeIds": sorted(edge_ids),
            "weldedVertices": sorted(welded_ids),
            "triangles": sorted(triangle_ids),
            "wallNormalAxis": "xyz"[wall_axis],
            "chunkBoundaryAxis": "xyz"[tangent_axis],
            "chunkBoundaryCoordinateMeters": tangent_coordinate,
            "chunkPlaneDistanceMeters": chunk_plane_distance,
            "tangentialSpreadMeters": tangential_spread,
            "verticalSpanMeters": vertical_span,
            "normalCoherence": normal_coherence,
            "boundsMin": np.min(positions, axis=0).tolist(),
            "boundsMax": np.max(positions, axis=0).tolist(),
        })

    chains.sort(key=lambda record: (
        -record["verticalSpanMeters"], -len(record["edgeIds"])))
    affected_triangles = set()
    for identifier, chain in enumerate(chains):
        chain["chain"] = identifier
        affected_triangles.update(chain["triangles"])

    receipt = {
        "kind": "RoomSetupReplicaV2VerticalChunkSeamReceipt",
        "sourceMesh": str(mesh_path),
        "sourceMeshSha256": sha256(mesh_path),
        "thresholds": {
            "weldQuantizationMeters": WELD_QUANTIZATION_METERS,
            "chunkSizeMeters": CHUNK_SIZE_METERS,
            "minimumVerticality": MINIMUM_VERTICALITY,
            "minimumPairedNormalCoherence": (
                MINIMUM_PAIRED_NORMAL_COHERENCE),
            "minimumChainNormalCoherence": (
                MINIMUM_CHAIN_NORMAL_COHERENCE),
            "minimumVerticalSpanMeters": MINIMUM_VERTICAL_SPAN_METERS,
            "maximumChunkPlaneDistanceMeters": (
                MAXIMUM_CHUNK_PLANE_DISTANCE_METERS),
            "maximumTangentialSpreadMeters": (
                MAXIMUM_TANGENTIAL_SPREAD_METERS),
        },
        "summary": {
            "meshTriangles": int(len(faces)),
            "duplicateCoordinateVertices": int(
                len(vertices) - len(welded_positions)),
            "verticalChunkSeamChains": len(chains),
            "seamEdges": sum(len(chain["edgeIds"]) for chain in chains),
            "adjacentTriangles": len(affected_triangles),
            "fullHeightChains": sum(
                chain["verticalSpanMeters"] >= 2.4 for chain in chains),
        },
        "interpretation": (
            "Each chain consists of coordinate-coincident edges that become "
            "paired only after a 0.1 mm weld, have coherent adjacent wall "
            "normals, run vertically for at least 0.6 m, and lie on a "
            "tangential 0.8 m chunk plane. This is a chunk publication/mesh "
            "welding seam class, distinct from the 104-face derived-grid "
            "zero-plane phase class."),
        "chains": chains,
    }

    payload_chains = []
    for chain in chains:
        edge_coordinates = []
        for edge in chain["edgeIds"]:
            record = candidate_edges[edge]
            for vertex in (record["a"], record["b"]):
                edge_coordinates.extend(np.rint(
                    welded_positions[vertex] * 1000.0).astype(
                        np.int64).tolist())
        payload_chains.append({
            "id": chain["chain"],
            "e": edge_coordinates,
            "f": chain["triangles"],
            "y": int(round(chain["verticalSpanMeters"] * 1000.0)),
            "a": chain["wallNormalAxis"],
            "t": chain["chunkBoundaryAxis"],
            "c": int(round(
                chain["chunkBoundaryCoordinateMeters"] * 1000.0)),
            "n": int(round(chain["normalCoherence"] * 1000.0)),
            "b0": np.rint(np.asarray(
                chain["boundsMin"]) * 1000.0).astype(np.int64).tolist(),
            "b1": np.rint(np.asarray(
                chain["boundsMax"]) * 1000.0).astype(np.int64).tolist(),
        })
    payload = {
        "v": np.rint(vertices.reshape(-1) * 1000.0).astype(
            np.int64).tolist(),
        "f": faces.reshape(-1).astype(np.int64).tolist(),
        "q": payload_chains,
        "s": receipt["summary"],
    }
    return receipt, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("output_receipt", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--visualization", type=Path)
    args = parser.parse_args()
    receipt, payload = build(args.mesh)
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipt.write_text(
        json.dumps(receipt, indent=2), encoding="utf-8")
    result = dict(receipt["summary"])
    if args.template or args.visualization:
        if not args.template or not args.visualization:
            raise ValueError("--template and --visualization must be paired")
        template = args.template.read_text(encoding="utf-8")
        marker = "__VERTICAL_SEAM_PAYLOAD__"
        if marker not in template:
            raise ValueError("Template lacks vertical-seam payload marker")
        output = template.replace(
            marker, json.dumps(payload, separators=(",", ":")))
        args.visualization.parent.mkdir(parents=True, exist_ok=True)
        args.visualization.write_text(output, encoding="utf-8")
        size = args.visualization.stat().st_size
        if size >= 1_000_000:
            raise ValueError(f"Visualization exceeds 1 MB: {size} bytes")
        result["visualizationBytes"] = size
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
