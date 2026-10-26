#!/usr/bin/env python3
"""Build spatially matched directly-supported controls for scar-source audit."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from build_spatial_scar_receipt import read_ply


MATCH_CHUNK_METERS = 0.8


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build(mesh_path: Path, scar_receipt_path: Path, output_path: Path):
    vertices, triangles = read_ply(mesh_path)
    scar = json.loads(scar_receipt_path.read_text(encoding="utf-8-sig"))
    if sha256(mesh_path) != scar["sourceMeshSha256"]:
        raise ValueError("Mesh SHA-256 does not match scar receipt")
    centroids = vertices[triangles].mean(axis=1)
    unsupported = {int(item["triangle"]) for item in scar["triangleReceipts"]}
    supported = np.array([
        triangle for triangle in range(len(triangles))
        if triangle not in unsupported
    ], dtype=np.int64)
    bins = defaultdict(list)
    supported_chunks = np.floor(
        centroids[supported] / MATCH_CHUNK_METERS).astype(np.int64)
    for triangle, chunk in zip(supported, supported_chunks):
        bins[tuple(chunk)].append(int(triangle))

    selected = []
    selected_set = set()
    matches = []
    for target in sorted(unsupported):
        center = centroids[target]
        base = np.floor(center / MATCH_CHUNK_METERS).astype(np.int64)
        candidates = []
        for radius in range(3):
            for dz in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if radius > 0 and max(abs(dx), abs(dy), abs(dz)) != radius:
                            continue
                        key = tuple(base + np.array([dx, dy, dz]))
                        candidates.extend(
                            face for face in bins.get(key, [])
                            if face not in selected_set)
            if candidates:
                break
        if not candidates:
            candidates = [
                int(face) for face in supported if int(face) not in selected_set
            ]
        if not candidates:
            break
        candidate_array = np.asarray(candidates, dtype=np.int64)
        distances = np.linalg.norm(
            centroids[candidate_array] - center, axis=1)
        chosen_index = int(np.argmin(distances))
        chosen = int(candidate_array[chosen_index])
        selected.append(chosen)
        selected_set.add(chosen)
        matches.append({
            "triangle": chosen,
            "cluster": -1,
            "spatialClassification": "DirectSupportedControl",
            "classification": "DirectSupportedControl",
            "matchedScarTriangle": target,
            "centroidDistanceMeters": float(distances[chosen_index]),
        })

    receipt = {
        "coordinateSpace": "QuestStage",
        "sourceMesh": str(mesh_path),
        "sourceMeshSha256": scar["sourceMeshSha256"],
        "sourceTape": scar.get("sourceTape", ""),
        "sourceTapeSha256": scar.get("sourceTapeSha256", ""),
        "controlRule": (
            "Unique directly-supported triangle nearest each scar triangle, "
            "preferring the same 0.8 m source chunk"),
        "totalTriangles": len(triangles),
        "unsupportedTriangles": len(unsupported),
        "controlTriangles": len(matches),
        "matchDistanceMeanMeters": float(np.mean([
            item["centroidDistanceMeters"] for item in matches])),
        "matchDistanceP95Meters": float(np.percentile([
            item["centroidDistanceMeters"] for item in matches], 95)),
        "triangleReceipts": matches,
    }
    output_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("scar_receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt = build(args.mesh, args.scar_receipt, args.output)
    print(json.dumps({
        "controlTriangles": receipt["controlTriangles"],
        "matchDistanceMeanMeters": receipt["matchDistanceMeanMeters"],
        "matchDistanceP95Meters": receipt["matchDistanceP95Meters"],
    }, indent=2))


if __name__ == "__main__":
    main()
