#!/usr/bin/env python3
"""Build a compact interactive 3D view from the accepted source audit."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from build_spatial_scar_receipt import read_ply


CATEGORY_CONFIRMED = 0
CATEGORY_BRIDGE_CANDIDATE = 1
CATEGORY_MIXING_CLUE = 2
CATEGORY_UNPROVEN = 3


def category(record):
    mechanisms = set(record["mechanisms"])
    mixed = "RepeatedMixedSignSourceVoxel" in mechanisms
    bridge = "DominantAnchorLayerBridge" in mechanisms
    if record.get("coherentHighSeparationBridge", False):
        return CATEGORY_CONFIRMED
    if bridge:
        return CATEGORY_BRIDGE_CANDIDATE
    if mixed:
        return CATEGORY_MIXING_CLUE
    return CATEGORY_UNPROVEN


def build(mesh_path: Path, audit_path: Path, control_audit_path: Path,
          template_path: Path, output_path: Path, crease_audit_path=None,
          crease_seed=None, crease_source_audit_path=None):
    vertices, faces = read_ply(mesh_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    control_audit = json.loads(
        control_audit_path.read_text(encoding="utf-8-sig"))
    categories = [[], [], [], []]
    metadata = []
    cluster_data = defaultdict(lambda: {
        "faces": [], "counts": [0, 0, 0, 0],
        "minimum": np.full(3, np.inf),
        "maximum": np.full(3, -np.inf),
    })
    for record in audit["triangleReceipts"]:
        triangle = int(record["triangle"])
        cluster = int(record["cluster"])
        code = category(record)
        categories[code].append(triangle)
        evidence = record["cornerEvidence"]
        metadata.append([
            triangle,
            cluster,
            code,
            int(round(record[
                "dominantAnchorMaximumNormalSeparationMeters"] * 1000.0)),
            int(evidence["stereoConflictMaximum"]),
            int(evidence["temporalSwitchMaximum"]),
            int(evidence["positivePairMaximum"]),
            int(evidence["negativePairMaximum"]),
            int(evidence["sameViewRepeatedMixedCorners"]),
            int(evidence["viewSeparatedRepeatedMixedCorners"]),
            int(record.get("highSeparationComponentTriangles", 0)),
        ])
        points = vertices[faces[triangle]]
        data = cluster_data[cluster]
        data["faces"].append(triangle)
        data["counts"][code] += 1
        data["minimum"] = np.minimum(data["minimum"], np.min(points, axis=0))
        data["maximum"] = np.maximum(data["maximum"], np.max(points, axis=0))

    clusters = []
    cluster_faces = {}
    for cluster, data in cluster_data.items():
        proven = data["counts"][CATEGORY_CONFIRMED]
        clusters.append([
            cluster,
            len(data["faces"]),
            proven,
            *data["counts"],
            *np.rint(data["minimum"] * 1000.0).astype(np.int64).tolist(),
            *np.rint(data["maximum"] * 1000.0).astype(np.int64).tolist(),
        ])
        cluster_faces[str(cluster)] = data["faces"]
    clusters.sort(key=lambda value: (-value[2], -value[1], value[0]))

    payload = {
        "v": np.rint(vertices.reshape(-1) * 1000.0).astype(
            np.int64).tolist(),
        "f": faces.reshape(-1).astype(np.int64).tolist(),
        "c": categories,
        "m": metadata,
        "k": clusters,
        "kf": cluster_faces,
        "s": {
            "triangles": int(len(faces)),
            "targets": int(len(metadata)),
            "confirmed": int(len(categories[CATEGORY_CONFIRMED])),
            "bridgeCandidates": int(
                len(categories[CATEGORY_CONFIRMED]) +
                len(categories[CATEGORY_BRIDGE_CANDIDATE])),
            "mixingClues": int(len(categories[CATEGORY_MIXING_CLUE])),
            "controlConfirmed": int(control_audit[
                "mechanismOverlapCounts"]["coherentHighSeparationBridge"]),
            "unproven": int(len(categories[CATEGORY_UNPROVEN])),
        },
    }
    if crease_audit_path is not None:
        crease_audit = json.loads(
            crease_audit_path.read_text(encoding="utf-8-sig"))
        seed_records = crease_audit.get("seeds", [])
        if crease_seed is None and seed_records:
            crease_seed = int(seed_records[0]["seedTriangle"])
        seed_record = next((value for value in seed_records
                            if int(value["seedTriangle"]) == crease_seed), None)
        if seed_record is None:
            raise ValueError(f"Crease seed {crease_seed} is absent from audit")
        component = next(value for value in seed_record["highAngleComponents"]
                         if float(value["thresholdDegrees"]) == 24.0)
        trace = next(value for value in seed_record["seedCreaseTraces"]
                     if float(value["thresholdDegrees"]) == 24.0)
        bounds_min = np.asarray(component["boundsMin"], dtype=np.float64)
        bounds_max = np.asarray(component["boundsMax"], dtype=np.float64)
        component_faces = sorted(set(
            int(value) for value in component.get("faceIds", [])) | {
                int(crease_seed)})
        fold_candidates = [value for value in crease_audit.get(
            "foldComponents", []) if value.get(
                "supportedCreaseCandidate", False)]
        all_candidate_faces = sorted({
            int(face) for value in fold_candidates
            for face in value.get("faceIds", [])})
        component_face_set = set(component.get("faceIds", []))
        matching_component = max(
            fold_candidates,
            key=lambda value: len(component_face_set.intersection(
                value.get("faceIds", []))),
            default=None)
        payload["hp"] = {
            "seed": int(crease_seed),
            "componentFaces": component_faces,
            "traceFaces": trace.get("faceIds", []),
            "edges": int(component["edges"]),
            "faces": int(component["faces"]),
            "edgeLengthMm": int(round(
                float(component["totalLengthMeters"]) * 1000.0)),
            "spanMm": np.rint((bounds_max - bounds_min) * 1000.0).astype(
                np.int64).tolist(),
            "boundsMinMm": np.rint(bounds_min * 1000.0).astype(
                np.int64).tolist(),
            "boundsMaxMm": np.rint(bounds_max * 1000.0).astype(
                np.int64).tolist(),
            "allCandidateFaces": all_candidate_faces,
            "candidateComponents": len(fold_candidates),
            "candidateFaces": len(all_candidate_faces),
            "componentRank": (int(matching_component["component"]) + 1
                              if matching_component is not None else None),
            "elongation": (round(float(matching_component["elongation"]), 2)
                            if matching_component is not None else None),
            "normalCoherence": (round(float(
                matching_component["normalCoherence"]), 3)
                if matching_component is not None else None),
            "angleMedianDegrees": (round(float(
                matching_component["angleMedianDegrees"]), 2)
                if matching_component is not None else None),
        }
        if crease_source_audit_path is not None:
            crease_source = json.loads(crease_source_audit_path.read_text(
                encoding="utf-8-sig"))
            overlap = crease_source["mechanismOverlapCounts"]
            payload["hp"]["source"] = {
                "triangles": len(crease_source["triangleReceipts"]),
                "repeatedMixed": int(overlap[
                    "repeatedMixedSignSourceVoxel"]),
                "sameViewMixed": int(overlap[
                    "sameViewRepeatedMixedSignSourceVoxel"]),
                "temporalSwitch": int(overlap[
                    "repeatedTemporalSignSwitching"]),
                "anchorBridge": int(overlap[
                    "dominantAnchorLayerBridge"]),
                "cellCrossing": int(overlap[
                    "reconstructedCellCrossing"]),
            }
    template = template_path.read_text(encoding="utf-8")
    encoded = json.dumps(payload, separators=(",", ":"))
    if "__SCAR_SOURCE_PAYLOAD__" not in template:
        raise ValueError("Template lacks payload marker")
    output = template.replace("__SCAR_SOURCE_PAYLOAD__", encoded)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    size = output_path.stat().st_size
    if size >= 1_000_000:
        raise ValueError(f"Visualization exceeds 1 MB: {size} bytes")
    return size, payload["s"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("control_audit", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--crease-audit", type=Path)
    parser.add_argument("--crease-seed", type=int)
    parser.add_argument("--crease-source-audit", type=Path)
    args = parser.parse_args()
    size, summary = build(
        args.mesh, args.audit, args.control_audit,
        args.template, args.output, args.crease_audit, args.crease_seed,
        args.crease_source_audit)
    print(json.dumps({"bytes": size, **summary}, indent=2))


if __name__ == "__main__":
    main()
