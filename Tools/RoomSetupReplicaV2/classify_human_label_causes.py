#!/usr/bin/env python3
"""Classify controller-painted mesh defects from a frozen source audit.

This is an evidence receipt, not a production deletion filter.  In particular,
unlabelled mesh faces are not assumed to be clean and the 5 mm zero-plane
threshold is accepted only for this frozen replay comparison.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from audit_supported_creases import face_geometry
from build_spatial_scar_receipt import read_ply


DERIVED_VOXEL_METERS = 0.10
ZERO_PLANE_BAND_METERS = 0.005
AXIS_PLANAR_CONFIGURATIONS = (
    {102, 153},  # X-normal plane
    {51, 204},   # Y-normal plane
    {15, 240},   # Z-normal plane
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def classify(record, centroid, normal):
    mechanisms = set(record.get("mechanisms", []))
    axis = int(np.argmax(np.abs(normal)))
    final_cell = record["finalCell"]
    configuration = int(final_cell["marchingCubesConfiguration"])
    axis_planar = configuration in AXIS_PLANAR_CONFIGURATIONS[axis]
    coordinate = float(centroid[axis])
    nearest_plane = round(coordinate / DERIVED_VOXEL_METERS)
    plane_distance = abs(
        coordinate - nearest_plane * DERIVED_VOXEL_METERS)
    evidence = record["cornerEvidence"]
    switched = int(evidence["temporalSwitchMaximum"]) > 0
    jumped = int(evidence["layerJumpMaximum"]) > 0

    if "RepeatedMixedSignSourceVoxel" in mechanisms:
        cause = "RepeatedMixedSourceVoxel"
    elif not axis_planar and plane_distance <= ZERO_PLANE_BAND_METERS:
        cause = "DerivedGridZeroPlanePhaseAmbiguity"
    elif not axis_planar and (switched or jumped):
        cause = "WeakTemporalLayerSwitch"
    else:
        cause = "PaintBoundaryOrUnresolved"

    return {
        "triangle": int(record["triangle"]),
        "stroke": int(record["cluster"]),
        "cause": cause,
        "dominantNormalAxis": "xyz"[axis],
        "centroid": [float(value) for value in centroid],
        "distanceToNearestDerivedPlaneMeters": plane_distance,
        "marchingCubesConfiguration": configuration,
        "axisPlanarConfiguration": axis_planar,
        "temporalSwitchMaximum": int(evidence["temporalSwitchMaximum"]),
        "layerJumpMaximum": int(evidence["layerJumpMaximum"]),
        "sameViewRepeatedMixedCorners": int(
            evidence["sameViewRepeatedMixedCorners"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("source_audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    audit = json.loads(args.source_audit.read_text(encoding="utf-8-sig"))
    mesh_sha = sha256(args.mesh)
    if mesh_sha != str(audit.get("sourceMeshSha256", "")).upper():
        raise ValueError("Mesh SHA-256 does not match the source audit")
    vertices, faces = read_ply(args.mesh)
    centroids, normals, _ = face_geometry(vertices, faces)
    records = [classify(
        record,
        centroids[int(record["triangle"])],
        normals[int(record["triangle"])])
        for record in audit["triangleReceipts"]]
    counts = Counter(record["cause"] for record in records)
    receipt = {
        "kind": "RoomSetupReplicaV2HumanLabelCauseClassification",
        "sourceMesh": str(args.mesh),
        "sourceMeshSha256": mesh_sha,
        "sourceAudit": str(args.source_audit),
        "sourceAuditSha256": sha256(args.source_audit),
        "thresholds": {
            "derivedVoxelMeters": DERIVED_VOXEL_METERS,
            "zeroPlaneBandMeters": ZERO_PLANE_BAND_METERS,
            "interpretation": (
                "Frozen-input evidence partition only; not a production "
                "mesh deletion rule."),
        },
        "counts": dict(sorted(counts.items())),
        "triangles": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt["counts"], indent=2))


if __name__ == "__main__":
    main()
