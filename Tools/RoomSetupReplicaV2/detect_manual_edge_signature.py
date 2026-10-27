"""Expand representative, manually selected defect edges over a frozen mesh.

The detector is deliberately conservative.  It learns two structural forms
from the positive edge atlas: derived-cell phase boundaries, and same-cell or
open-boundary continuations connected to those phase anchors.  Repeated mixed
source evidence alone is far too common to be treated as a defect.
"""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_supported_creases import face_geometry
from build_spatial_scar_receipt import read_ply


DERIVED_VOXEL_METERS = 0.10
PLANE_EPSILON_METERS = 0.0005
MIN_EDGE_METERS = 0.015
MAX_EDGE_METERS = 0.160
HIGH_DIHEDRAL_DEGREES = 15.0
PROBABLE_DIHEDRAL_DEGREES = 35.0

MIXED_REQUIREMENTS = {
    "RepeatedMixedSignSourceVoxel",
    "SameViewRepeatedMixedSignSourceVoxel",
}
TEMPORAL_REQUIREMENTS = MIXED_REQUIREMENTS | {
    "RepeatedTemporalSignSwitching",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def edge_owners(faces):
    owners = defaultdict(list)
    for triangle, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]),
                     (face[2], face[0])):
            owners[tuple(sorted((int(a), int(b))))].append(triangle)
    return owners


def components(edges):
    vertex_edges = defaultdict(list)
    for edge in edges:
        for vertex in edge:
            vertex_edges[vertex].append(edge)
    result = []
    seen = set()
    for seed in edges:
        if seed in seen:
            continue
        pending = [seed]
        seen.add(seed)
        component = []
        while pending:
            edge = pending.pop()
            component.append(edge)
            for vertex in edge:
                for neighbour in vertex_edges[vertex]:
                    if neighbour not in seen:
                        seen.add(neighbour)
                        pending.append(neighbour)
        result.append(component)
    return result


def touched_components(edges, anchors, manual):
    anchor_vertices = {vertex for edge in anchors for vertex in edge}
    selected = set()
    for component in components(edges):
        if (set(component) & manual or
                any(vertex in anchor_vertices
                    for edge in component for vertex in edge)):
            selected.update(component)
    return selected


def build(mesh_path, source_audit_path, positive_atlas_path):
    vertices, faces = read_ply(mesh_path)
    audit = json.loads(source_audit_path.read_text(encoding="utf-8-sig"))
    atlas = json.loads(positive_atlas_path.read_text(encoding="utf-8-sig"))
    mesh_hash = sha256(mesh_path)
    if mesh_hash != str(audit.get("sourceMeshSha256", "")).upper():
        raise ValueError("Mesh SHA-256 does not match full-mesh source audit")
    records = audit["triangleReceipts"]
    if len(records) != len(faces):
        raise ValueError("Full-mesh source audit does not contain every face")

    _, normals, _ = face_geometry(vertices, faces)
    owners = edge_owners(faces)
    positive_records = atlas.get("selectedMeshRecords", [])
    positives = set()
    for row in positive_records:
        triangle, a, b = int(row[0]), int(row[1]), int(row[2])
        edge = tuple(sorted((a, b)))
        if edge not in owners:
            raise ValueError(f"Selected edge {edge} is absent from source mesh")
        face = set(map(int, faces[triangle]))
        if a not in face or b not in face:
            raise ValueError(
                f"Selected face {triangle} does not own edge {edge}")
        positives.add(edge)

    def source_matches(triangle, temporal):
        mechanisms = set(records[triangle].get("mechanisms", []))
        required = TEMPORAL_REQUIREMENTS if temporal else MIXED_REQUIREMENTS
        return required <= mechanisms

    def edge_type(edge):
        adjacent = owners[edge]
        if len(adjacent) != 2:
            return "boundary"
        cells = np.asarray(
            [records[triangle]["cell"] for triangle in adjacent],
            dtype=np.int64)
        delta = np.abs(cells[0] - cells[1])
        if int(np.sum(delta)) != 1:
            return "same-cell"
        axis = int(np.argmax(delta))
        boundary = float(np.max(cells[:, axis])) * DERIVED_VOXEL_METERS
        if float(np.max(np.abs(
                vertices[list(edge), axis] - boundary))) <= PLANE_EPSILON_METERS:
            return "phase-boundary"
        return "same-cell"

    def configuration_key(edge):
        return tuple(sorted(
            int(records[triangle]["finalCell"][
                "marchingCubesConfiguration"])
            for triangle in owners[edge]))

    def dihedral_degrees(edge):
        adjacent = owners[edge]
        if len(adjacent) != 2:
            return 0.0
        cosine = abs(float(np.dot(normals[adjacent[0]], normals[adjacent[1]])))
        return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))

    def eligible(edge, learned_keys, temporal, maximum_dihedral):
        adjacent = owners[edge]
        if configuration_key(edge) not in learned_keys:
            return False
        if not any(source_matches(triangle, temporal)
                   for triangle in adjacent):
            return False
        length = float(np.linalg.norm(
            vertices[edge[0]] - vertices[edge[1]]))
        return (MIN_EDGE_METERS <= length <= MAX_EDGE_METERS and
                dihedral_degrees(edge) <= maximum_dihedral)

    phase_positives = {
        edge for edge in positives if edge_type(edge) == "phase-boundary"
    }
    continuation_positives = positives - phase_positives
    phase_keys = {configuration_key(edge) for edge in phase_positives}
    continuation_keys = {
        configuration_key(edge) for edge in continuation_positives
    }

    all_edges = set(owners)
    high_phase = {
        edge for edge in all_edges
        if edge_type(edge) == "phase-boundary" and
        eligible(edge, phase_keys, True, HIGH_DIHEDRAL_DEGREES)
    } | phase_positives
    high_continuation_pool = {
        edge for edge in all_edges
        if edge_type(edge) != "phase-boundary" and
        eligible(edge, continuation_keys, True, HIGH_DIHEDRAL_DEGREES)
    }
    high_continuation = touched_components(
        high_continuation_pool, high_phase, continuation_positives)
    high_continuation |= continuation_positives

    probable_phase = {
        edge for edge in all_edges
        if edge_type(edge) == "phase-boundary" and
        eligible(edge, phase_keys, False, PROBABLE_DIHEDRAL_DEGREES)
    } | phase_positives
    probable_continuation_pool = {
        edge for edge in all_edges
        if edge_type(edge) != "phase-boundary" and
        eligible(edge, continuation_keys, False,
                 PROBABLE_DIHEDRAL_DEGREES)
    }
    probable_continuation = touched_components(
        probable_continuation_pool, probable_phase,
        continuation_positives)
    probable_continuation |= continuation_positives

    high_phase_new = high_phase - positives
    high_continuation_new = high_continuation - positives
    probable_new = (
        probable_phase | probable_continuation
    ) - positives - high_phase_new - high_continuation_new
    categories = [
        sorted(positives),
        sorted(high_phase_new),
        sorted(high_continuation_new),
        sorted(probable_new),
    ]

    source_signature_edges = {
        edge for edge in all_edges
        if any(source_matches(triangle, False) for triangle in owners[edge])
    }
    displayed = set().union(*map(set, categories))

    type_codes = {
        "phase-boundary": 0,
        "same-cell": 1,
        "boundary": 2,
    }
    metadata = []
    category_by_edge = {
        edge: category
        for category, edges in enumerate(categories)
        for edge in edges
    }
    for edge in sorted(displayed):
        adjacent = owners[edge]
        configurations = list(configuration_key(edge))
        temporal_maximum = max(
            int(records[triangle]["cornerEvidence"][
                "temporalSwitchMaximum"])
            for triangle in adjacent)
        layer_jump_maximum = max(
            int(records[triangle]["cornerEvidence"]["layerJumpMaximum"])
            for triangle in adjacent)
        metadata.append([
            edge[0], edge[1], category_by_edge[edge],
            type_codes[edge_type(edge)], len(adjacent),
            configurations[0],
            configurations[1] if len(configurations) > 1 else -1,
            temporal_maximum, layer_jump_maximum,
            int(round(dihedral_degrees(edge) * 1000.0)),
            int(round(np.linalg.norm(
                vertices[edge[0]] - vertices[edge[1]]) * 1_000_000.0)),
        ])

    summary = {
        "meshTriangles": int(len(faces)),
        "meshEdges": len(all_edges),
        "manualPositiveEdges": len(positives),
        "manualPhaseBoundaryEdges": len(phase_positives),
        "manualContinuationEdges": len(continuation_positives),
        "highConfidencePhaseEdges": len(high_phase_new),
        "highConfidenceContinuationEdges": len(high_continuation_new),
        "highConfidenceNewEdges": (
            len(high_phase_new) + len(high_continuation_new)),
        "probableNewEdges": len(probable_new),
        "sourceSignatureEdges": len(source_signature_edges),
        "sourceSignatureExcludedEdges": len(source_signature_edges - displayed),
    }
    receipt = {
        "kind": "RoomSetupReplicaV2ManualEdgeSignatureReceipt",
        "sourceMesh": str(mesh_path),
        "sourceMeshSha256": mesh_hash,
        "sourceAudit": str(source_audit_path),
        "sourceAuditSha256": sha256(source_audit_path),
        "sourcePositiveAtlas": str(positive_atlas_path),
        "sourcePositiveAtlasSha256": sha256(positive_atlas_path),
        "rule": {
            "sourceRequirements": sorted(MIXED_REQUIREMENTS),
            "highAdditionalRequirement":
                "RepeatedTemporalSignSwitching",
            "phasePlaneEpsilonMeters": PLANE_EPSILON_METERS,
            "edgeLengthMeters": [MIN_EDGE_METERS, MAX_EDGE_METERS],
            "highDihedralDegrees": HIGH_DIHEDRAL_DEGREES,
            "probableDihedralDegrees": PROBABLE_DIHEDRAL_DEGREES,
            "configurationPairs": {
                "phaseBoundary": [list(value)
                                  for value in sorted(phase_keys)],
                "continuation": [list(value)
                                 for value in sorted(continuation_keys)],
            },
        },
        "summary": summary,
        "categories": {
            "manualPositive": [list(edge) for edge in categories[0]],
            "highConfidencePhase": [list(edge) for edge in categories[1]],
            "highConfidenceContinuation": [list(edge)
                                             for edge in categories[2]],
            "probable": [list(edge) for edge in categories[3]],
        },
        "metadataRows": metadata,
        "interpretationBoundary": (
            "The manual 48 edges are pinned positives. High-confidence "
            "matches require the learned MC configuration family, repeated "
            "mixed-source evidence, temporal switching, compatible geometry, "
            "and either a derived-cell phase boundary or a connected "
            "continuation. Probable matches relax temporal and dihedral "
            "requirements. Repeated mixed-source evidence alone is explicitly "
            "excluded from automatic defect classification."),
    }
    payload = {
        "v": np.rint(vertices.reshape(-1) * 1000.0).astype(
            np.int64).tolist(),
        "f": faces.reshape(-1).astype(np.int64).tolist(),
        "c": [[[edge[0], edge[1]] for edge in values]
              for values in categories],
        "m": metadata,
        "s": summary,
    }
    return receipt, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("source_audit", type=Path)
    parser.add_argument("positive_atlas", type=Path)
    parser.add_argument("output_receipt", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--visualization", type=Path)
    args = parser.parse_args()
    receipt, payload = build(
        args.mesh, args.source_audit, args.positive_atlas)
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipt.write_text(
        json.dumps(receipt, indent=2), encoding="utf-8")
    result = dict(receipt["summary"])
    if args.template or args.visualization:
        if not args.template or not args.visualization:
            raise ValueError("--template and --visualization must be paired")
        template = args.template.read_text(encoding="utf-8")
        marker = "__MANUAL_EDGE_SIGNATURE_PAYLOAD__"
        if marker not in template:
            raise ValueError("Template lacks manual-edge payload marker")
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
