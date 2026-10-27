#!/usr/bin/env python3
"""Locate the frozen replay's 10 cm zero-plane phase signature.

The detector deliberately separates evidence levels:

* controller-confirmed faces in the 104-face class;
* unpainted faces in the same edge-connected signature components;
* high/probable nearest-neighbour matches to the 104-face feature atlas;
* broad static-signature hits that remain review-only.

The third set is a review set, not an automatic deletion mask.  A real planar
surface can lie close to the world-anchored 10 cm lattice and produce the same
static signature without being visually defective.
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
ZERO_PLANE_BAND_METERS = 0.005
AXIS_PLANAR_CONFIGURATIONS = (
    {102, 153},
    {51, 204},
    {15, 240},
)
CAUSE_GRID_PHASE = "DerivedGridZeroPlanePhaseAmbiguity"
CAUSE_CODES = {
    "RepeatedMixedSourceVoxel": 1,
    "WeakTemporalLayerSwitch": 2,
    "PaintBoundaryOrUnresolved": 3,
}
CORNER_OFFSETS = np.asarray([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=np.int64)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def edge_adjacency(faces):
    owners = defaultdict(list)
    for triangle, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]),
                     (face[2], face[0])):
            owners[tuple(sorted((int(a), int(b))))].append(triangle)
    adjacency = [set() for _ in range(len(faces))]
    for triangles in owners.values():
        for triangle in triangles:
            adjacency[triangle].update(
                other for other in triangles if other != triangle)
    return adjacency


def connected_components(mask, adjacency):
    component_id = np.full(len(mask), -1, dtype=np.int64)
    components = []
    seen = set()
    for seed in np.flatnonzero(mask):
        seed = int(seed)
        if seed in seen:
            continue
        pending = [seed]
        seen.add(seed)
        component = []
        while pending:
            triangle = pending.pop()
            component.append(triangle)
            for neighbour in adjacency[triangle]:
                if mask[neighbour] and neighbour not in seen:
                    seen.add(neighbour)
                    pending.append(neighbour)
        identifier = len(components)
        for triangle in component:
            component_id[triangle] = identifier
        components.append(sorted(component))
    return component_id, components


def atlas_features(records, centroids, normals, areas, axes,
                   plane_distances, signature, adjacency, component_ids,
                   components):
    """Build rotation-independent local features learned from the 104 faces."""
    maximum_component = max((len(value) for value in components), default=1)
    component_scale = math.log1p(maximum_component)
    features = np.zeros((len(records), 12), dtype=np.float64)
    for triangle, record in enumerate(records):
        axis = int(axes[triangle])
        cell = np.asarray(record["cell"], dtype=np.int64)
        nearest_plane = int(round(
            float(centroids[triangle, axis]) / DERIVED_VOXEL_METERS))
        plane_offset = nearest_plane - int(cell[axis])
        if plane_offset in (0, 1):
            layer = np.flatnonzero(
                CORNER_OFFSETS[:, axis] == plane_offset)
        else:
            layer = np.arange(8)
        distances = np.asarray(
            record["finalCell"]["distancesMeters"], dtype=np.float64)
        layer_absolute = np.abs(distances[layer])
        evidence = record["cornerEvidence"]
        candidate_neighbours = sum(
            1 for neighbour in adjacency[triangle]
            if signature[neighbour])
        component = int(component_ids[triangle])
        component_size = (
            len(components[component]) if component >= 0 else 0)
        features[triangle] = [
            plane_distances[triangle] / ZERO_PLANE_BAND_METERS,
            1.0 - float(np.max(np.abs(normals[triangle]))),
            math.log10(max(float(areas[triangle]), 1e-9)),
            float(evidence["temporalSwitchMaximum"]) / 4.0,
            float(evidence["layerJumpMaximum"]) / 11.0,
            float(record["finalCell"]["crossingEdges"]) / 12.0,
            float(record["finalCell"]["negativeCorners"]) / 8.0,
            float(np.median(layer_absolute)) / 0.02,
            float(np.min(layer_absolute)) / 0.02,
            float(np.std(distances[layer])) / 0.02,
            float(candidate_neighbours) / 3.0,
            math.log1p(component_size) / component_scale,
        ]
    return features


def atlas_scores(features, positives, candidates):
    positive_features = features[positives]
    median = np.median(positive_features, axis=0)
    lower, upper = np.quantile(positive_features, [0.25, 0.75], axis=0)
    scale = np.maximum(
        upper - lower, np.std(positive_features, axis=0) * 0.5)
    # Each input was normalized to a comparable physical range above.  This
    # floor prevents a nearly constant positive feature from dominating.
    scale = np.maximum(scale, 0.05)
    normalized = (features - median) / scale
    positive_normalized = normalized[positives]
    pairwise = np.sqrt(np.mean(
        (positive_normalized[:, None, :] -
         positive_normalized[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(pairwise, np.inf)
    leave_one_out = np.min(pairwise, axis=1)
    candidate_scores = np.sqrt(np.min(np.mean(
        (normalized[candidates, None, :] -
         positive_normalized[None, :, :]) ** 2, axis=2), axis=1))
    return (
        candidate_scores,
        float(np.quantile(leave_one_out, 0.90)),
        float(np.quantile(leave_one_out, 0.95)),
        leave_one_out,
    )


def build(mesh_path, audit_path, human_path):
    vertices, faces = read_ply(mesh_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    human = json.loads(human_path.read_text(encoding="utf-8-sig"))
    mesh_sha = sha256(mesh_path)
    if mesh_sha != str(audit.get("sourceMeshSha256", "")).upper():
        raise ValueError("Mesh SHA-256 does not match full-mesh source audit")
    if mesh_sha != str(human.get("sourceMeshSha256", "")).upper():
        raise ValueError("Mesh SHA-256 does not match human classification")
    records = audit["triangleReceipts"]
    if len(records) != len(faces):
        raise ValueError("Full-mesh audit does not contain every triangle")

    centroids, normals, areas = face_geometry(vertices, faces)
    axes = np.argmax(np.abs(normals), axis=1)
    configurations = np.asarray([
        int(record["finalCell"]["marchingCubesConfiguration"])
        for record in records
    ], dtype=np.int64)
    repeated_mixed = np.asarray([
        "RepeatedMixedSignSourceVoxel" in record.get("mechanisms", [])
        for record in records
    ], dtype=bool)
    axis_planar = np.asarray([
        int(configurations[index]) in
        AXIS_PLANAR_CONFIGURATIONS[int(axes[index])]
        for index in range(len(faces))
    ], dtype=bool)
    axis_coordinates = centroids[np.arange(len(faces)), axes]
    plane_distances = np.abs(
        axis_coordinates -
        np.round(axis_coordinates / DERIVED_VOXEL_METERS) *
        DERIVED_VOXEL_METERS)
    signature = (
        ~repeated_mixed & ~axis_planar &
        (plane_distances <= ZERO_PLANE_BAND_METERS))

    human_causes = {
        int(record["triangle"]): record["cause"]
        for record in human["triangles"]
    }
    confirmed = {
        triangle for triangle, cause in human_causes.items()
        if cause == CAUSE_GRID_PHASE
    }
    if not all(signature[triangle] for triangle in confirmed):
        raise ValueError("The frozen 104-face class no longer matches detector")

    adjacency = edge_adjacency(faces)
    component_ids, components = connected_components(signature, adjacency)
    touched_components = {
        int(component_ids[triangle]) for triangle in confirmed
    }
    connected_continuations = {
        triangle
        for component in touched_components
        for triangle in components[component]
    } - confirmed
    all_candidates = np.flatnonzero(signature).astype(np.int64)
    independent_candidates = set(all_candidates.tolist()) - (
        confirmed | connected_continuations)
    other_human = set(human_causes) - confirmed

    confirmed_array = np.asarray(sorted(confirmed), dtype=np.int64)
    features = atlas_features(
        records, centroids, normals, areas, axes, plane_distances,
        signature, adjacency, component_ids, components)
    candidate_scores, high_threshold, probable_threshold, leave_one_out = (
        atlas_scores(features, confirmed_array, all_candidates))
    score_by_triangle = {
        int(triangle): float(score)
        for triangle, score in zip(all_candidates, candidate_scores)
    }
    high_confidence = connected_continuations | {
        triangle for triangle in independent_candidates
        if score_by_triangle[triangle] <= high_threshold
    }
    probable_matches = {
        triangle for triangle in independent_candidates
        if (high_threshold < score_by_triangle[triangle] <=
            probable_threshold)
    }
    broad_signature_only = independent_candidates - (
        high_confidence | probable_matches)

    categories = [
        sorted(confirmed),
        sorted(high_confidence),
        sorted(probable_matches),
        sorted(broad_signature_only),
        sorted(other_human),
    ]
    category_by_triangle = {}
    for category, triangles in enumerate(categories):
        for triangle in triangles:
            category_by_triangle[triangle] = category

    metadata = []
    for triangle, category in sorted(category_by_triangle.items()):
        evidence = records[triangle]["cornerEvidence"]
        metadata.append([
            triangle,
            category,
            int(component_ids[triangle]),
            int(axes[triangle]),
            int(round(plane_distances[triangle] * 1_000_000.0)),
            int(configurations[triangle]),
            int(evidence["temporalSwitchMaximum"]),
            int(evidence["layerJumpMaximum"]),
            int(round(areas[triangle] * 1_000_000.0)),
            int(CAUSE_CODES.get(human_causes.get(triangle, ""), 0)),
            int(round(score_by_triangle.get(triangle, -1.0) * 1000.0)),
            int(triangle in connected_continuations),
        ])

    component_receipts = []
    for identifier, triangles in enumerate(components):
        points = vertices[faces[triangles].reshape(-1)]
        bounds_minimum = np.min(points, axis=0)
        bounds_maximum = np.max(points, axis=0)
        confirmed_count = len(set(triangles) & confirmed)
        component_receipts.append({
            "component": identifier,
            "triangles": len(triangles),
            "confirmedTriangles": confirmed_count,
            "boundsMin": bounds_minimum.tolist(),
            "boundsMax": bounds_maximum.tolist(),
        })

    summary = {
        "totalMeshTriangles": int(len(faces)),
        "humanConfirmedGridPhaseTriangles": len(confirmed),
        "connectedUnpaintedContinuationTriangles": len(
            connected_continuations),
        "highConfidenceSameProblemTriangles": len(high_confidence),
        "highConfidenceIndependentTriangles": len(
            high_confidence - connected_continuations),
        "probableSameProblemTriangles": len(probable_matches),
        "broadSignatureOnlyTriangles": len(broad_signature_only),
        "independentSameSignatureCandidateTriangles": len(independent_candidates),
        "sameSignatureCandidateTriangles": int(np.count_nonzero(signature)),
        "sameSignatureComponents": len(components),
        "componentsTouchedByHumanConfirmation": len(touched_components),
        "otherHumanConfirmedTriangles": len(other_human),
        "humanGridPhaseRecall": (
            float(sum(signature[triangle] for triangle in confirmed) /
                  len(confirmed)) if confirmed else 0.0),
        "atlasLeaveOneOutHighRecall": float(np.mean(
            leave_one_out <= high_threshold)),
        "atlasLeaveOneOutProbableRecall": float(np.mean(
            leave_one_out <= probable_threshold)),
    }
    receipt = {
        "kind": "RoomSetupReplicaV2GridPhaseCandidateReceipt",
        "sourceMesh": str(mesh_path),
        "sourceMeshSha256": mesh_sha,
        "sourceAudit": str(audit_path),
        "sourceAuditSha256": sha256(audit_path),
        "sourceHumanClassification": str(human_path),
        "sourceHumanClassificationSha256": sha256(human_path),
        "rule": {
            "required": [
                "NoRepeatedMixedSignSourceVoxel",
                "NonAxisPlanarMarchingCubesConfiguration",
                "FaceCentroidWithin5mmOfDominantAxis10cmLatticePlane",
            ],
            "derivedVoxelMeters": DERIVED_VOXEL_METERS,
            "zeroPlaneBandMeters": ZERO_PLANE_BAND_METERS,
        },
        "atlas": {
            "positiveExamples": len(confirmed),
            "features": [
                "normalizedDistanceTo10cmPlane",
                "dominantNormalAxisDeviation",
                "logTriangleArea",
                "temporalSwitchMaximum",
                "layerJumpMaximum",
                "crossingEdgeCount",
                "negativeCornerCount",
                "nearGridLayerAbsoluteSdfMedian",
                "nearGridLayerAbsoluteSdfMinimum",
                "nearGridLayerSdfStandardDeviation",
                "sameSignatureEdgeNeighbourCount",
                "sameSignatureComponentSize",
            ],
            "distance": "RobustScaledNearestPositiveRms",
            "highThreshold": high_threshold,
            "probableThreshold": probable_threshold,
            "highThresholdSource": "PositiveLeaveOneOut90thPercentile",
            "probableThresholdSource": "PositiveLeaveOneOut95thPercentile",
        },
        "interpretationBoundary": (
            "The 104 painted faces are exact confirmed positives. The high "
            "class combines their edge-connected continuations with "
            "independent faces inside the positive atlas 90th-percentile "
            "leave-one-out distance. Probable matches extend to the 95th "
            "percentile. Broad static-signature hits are review-only, not an "
            "automatic deletion mask; unlabelled mesh is not assumed clean "
            "or defective."),
        "summary": summary,
        "categories": {
            "humanConfirmedGridPhase": categories[0],
            "highConfidenceSameProblem": categories[1],
            "probableSameProblem": categories[2],
            "broadSignatureOnly": categories[3],
            "otherHumanConfirmed": categories[4],
        },
        "metadataRows": metadata,
        "components": component_receipts,
    }
    payload = {
        "v": np.rint(vertices.reshape(-1) * 1000.0).astype(
            np.int64).tolist(),
        "f": faces.reshape(-1).astype(np.int64).tolist(),
        "c": categories,
        "m": metadata,
        "s": summary,
        "p": {
            "connectedContinuation": sorted(connected_continuations),
        },
    }
    return receipt, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("full_source_audit", type=Path)
    parser.add_argument("human_classification", type=Path)
    parser.add_argument("output_receipt", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--visualization", type=Path)
    args = parser.parse_args()
    receipt, payload = build(
        args.mesh, args.full_source_audit, args.human_classification)
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipt.write_text(
        json.dumps(receipt, indent=2), encoding="utf-8")
    result = dict(receipt["summary"])
    if args.template or args.visualization:
        if not args.template or not args.visualization:
            raise ValueError("--template and --visualization must be paired")
        template = args.template.read_text(encoding="utf-8")
        marker = "__GRID_PHASE_PAYLOAD__"
        if marker not in template:
            raise ValueError("Template lacks grid-phase payload marker")
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
