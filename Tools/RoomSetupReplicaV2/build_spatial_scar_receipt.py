#!/usr/bin/env python3
"""Build the offline V2-C scar receipt from an RSV2 tape and binary PLY."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


MAGIC = 0x32565352
FRAME = 0x4D415246
END = 0x21444E45
DIRECT_SUPPORT_METERS = 0.05
CONTRADICTION_METERS = 0.10
REPEATED_PAIRS = 3


def read_ply(path: Path):
    with path.open("rb") as stream:
        vertex_count = face_count = None
        while True:
            line = stream.readline().decode("ascii").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            elif line.startswith("element face "):
                face_count = int(line.split()[-1])
            elif line == "end_header":
                break
        if vertex_count is None or face_count is None:
            raise ValueError("PLY header lacks vertex/face counts")
        vertices = np.fromfile(
            stream, dtype="<f4", count=vertex_count * 3
        ).reshape(vertex_count, 3).astype(np.float64)
        face_dtype = np.dtype([
            ("count", "u1"), ("indices", "<i4", (3,))
        ])
        faces = np.fromfile(stream, dtype=face_dtype, count=face_count)
    if len(faces) != face_count or np.any(faces["count"] != 3):
        raise ValueError("PLY is truncated or contains non-triangle faces")
    return vertices, faces["indices"].astype(np.int64)


def matrix(values):
    # Unity Matrix4x4[index] is column-major: row + column * 4.
    return values.reshape((4, 4), order="F").astype(np.float64)


def project(points, world_to_clip, projection_inverse, view_inverse, ndc,
            width, height):
    count = len(points)
    homogeneous = np.empty((count, 4), dtype=np.float64)
    homogeneous[:, :3] = points
    homogeneous[:, 3] = 1.0
    clip = homogeneous @ world_to_clip.T
    valid = np.isfinite(clip[:, 3]) & (clip[:, 3] > 1e-5)
    inverse_w = np.zeros(count, dtype=np.float64)
    inverse_w[valid] = 1.0 / clip[valid, 3]
    u = clip[:, 0] * inverse_w * 0.5 + 0.5
    v = clip[:, 1] * inverse_w * 0.5 + 0.5
    valid &= np.isfinite(u) & np.isfinite(v)
    valid &= (u >= 0.0) & (u < 1.0) & (v >= 0.0) & (v < 1.0)
    x = np.clip((u * width).astype(np.int64), 0, width - 1)
    y = np.clip((v * height).astype(np.int64), 0, height - 1)
    sampled_ndc = ndc[x + y * width].astype(np.float64)
    valid &= (sampled_ndc > 0.0) & (sampled_ndc < 1.0)

    sample_u = (x + 0.5) / width
    sample_v = (y + 0.5) / height
    ndc_h = np.column_stack((
        sample_u * 2.0 - 1.0,
        sample_v * 2.0 - 1.0,
        sampled_ndc * 2.0 - 1.0,
        np.ones(count),
    ))
    view_h = ndc_h @ projection_inverse.T
    stage_h = view_h @ view_inverse.T
    valid &= np.isfinite(stage_h[:, 3]) & (np.abs(stage_h[:, 3]) > 1e-5)
    observed = np.zeros((count, 3), dtype=np.float64)
    observed[valid] = (
        stage_h[valid, :3] / stage_h[valid, 3, np.newaxis]
    )
    valid &= np.isfinite(observed).all(axis=1)
    point_distance = np.full(count, np.inf, dtype=np.float64)
    point_distance[valid] = np.linalg.norm(
        observed[valid] - points[valid], axis=1
    )
    eye = view_inverse[:3, 3]
    residual = np.zeros(count, dtype=np.float64)
    residual[valid] = (
        np.linalg.norm(observed[valid] - eye, axis=1)
        - np.linalg.norm(points[valid] - eye, axis=1)
    )
    valid &= np.isfinite(residual) & np.isfinite(point_distance)
    return valid, point_distance, residual


def frames(path: Path):
    with path.open("rb") as stream:
        magic, version, width, height = struct.unpack("<Iiii", stream.read(16))
        if magic != MAGIC or version != 2:
            raise ValueError("Not an RSV2 version-2 tape")
        samples = width * height
        while True:
            marker_bytes = stream.read(4)
            if len(marker_bytes) != 4:
                break
            marker = struct.unpack("<I", marker_bytes)[0]
            if marker == END:
                break
            if marker != FRAME:
                raise ValueError("Corrupt RSV2 frame marker")
            platform_frame, capture_time = struct.unpack("<id", stream.read(12))
            matrices = np.fromfile(stream, dtype="<f4", count=96)
            if len(matrices) != 96:
                raise ValueError("Truncated RSV2 matrices")
            left = np.fromfile(stream, dtype="<f4", count=samples)
            right = np.fromfile(stream, dtype="<f4", count=samples)
            if len(left) != samples or len(right) != samples:
                raise ValueError("Truncated RSV2 depth plane")
            yield width, height, platform_frame, capture_time, [
                matrix(matrices[i * 16:(i + 1) * 16]) for i in range(6)
            ], left, right


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first, second):
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[b] = a


def build(mesh_path: Path, tape_path: Path, output_path: Path):
    vertices, triangles = read_ply(mesh_path)
    triangle_vertices = vertices[triangles]
    centroids = triangle_vertices.mean(axis=1)
    count = len(triangles)
    supported = np.zeros(count, dtype=bool)
    valid_observations = np.zeros(count, dtype=np.int32)
    foreground_pairs = np.zeros(count, dtype=np.int32)
    background_pairs = np.zeros(count, dtype=np.int32)
    minimum = np.full(count, np.inf)
    maximum = np.full(count, -np.inf)
    nearest = np.full(count, np.inf)
    frame_count = 0

    for width, height, _, _, matrices, left, right in frames(tape_path):
        frame_count += 1
        unresolved = np.flatnonzero(~supported)
        if len(unresolved) == 0:
            continue
        points = np.concatenate((
            centroids[unresolved],
            triangle_vertices[unresolved, 0],
            triangle_vertices[unresolved, 1],
            triangle_vertices[unresolved, 2],
        ))
        eye_results = []
        for eye, depths in ((0, left), (1, right)):
            eye_results.append(project(
                points,
                matrices[eye], matrices[2 + eye], matrices[4 + eye],
                depths, width, height,
            ))
        span = len(unresolved)
        centroid_valid = []
        centroid_residual = []
        point_support = []
        for valid, distance, residual in eye_results:
            centroid_valid.append(valid[:span])
            centroid_residual.append(residual[:span])
            point_support.append((valid & (distance <= DIRECT_SUPPORT_METERS)).reshape(4, span))
        for valid, residual in zip(centroid_valid, centroid_residual):
            indices = unresolved[valid]
            values = residual[valid]
            valid_observations[indices] += 1
            minimum[indices] = np.minimum(minimum[indices], values)
            maximum[indices] = np.maximum(maximum[indices], values)
            nearest[indices] = np.minimum(nearest[indices], np.abs(values))
        foreground = np.zeros(span, dtype=bool)
        background = np.zeros(span, dtype=bool)
        for valid, residual in zip(centroid_valid, centroid_residual):
            foreground |= valid & (residual <= -CONTRADICTION_METERS)
            background |= valid & (residual >= CONTRADICTION_METERS)
        foreground_pairs[unresolved[foreground]] += 1
        background_pairs[unresolved[background]] += 1

        combined = point_support[0] | point_support[1]
        newly_supported = combined[0] | (combined[1:].sum(axis=0) >= 2)
        supported[unresolved[newly_supported]] = True

    unsupported = np.flatnonzero(~supported)
    union = UnionFind(count)
    owners = {}
    for triangle in unsupported:
        for vertex in triangle_vertices[triangle]:
            key = tuple(np.rint(vertex * 10000.0).astype(np.int64))
            if key in owners:
                union.union(int(triangle), owners[key])
            else:
                owners[key] = int(triangle)
    roots = {}
    records = []
    cluster_data = []
    for triangle in unsupported:
        root = union.find(int(triangle))
        if root not in roots:
            roots[root] = len(roots)
            cluster_data.append({
                "triangles": 0,
                "boundsMin": np.full(3, np.inf),
                "boundsMax": np.full(3, -np.inf),
                "crossLayerCandidates": 0,
                "historicalShellCandidates": 0,
                "insufficientEvidence": 0,
            })
        cluster = roots[root]
        if foreground_pairs[triangle] > 0 and background_pairs[triangle] > 0:
            classification = "CrossLayerConnectionCandidate"
            cluster_data[cluster]["crossLayerCandidates"] += 1
        elif (foreground_pairs[triangle] >= REPEATED_PAIRS or
              background_pairs[triangle] >= REPEATED_PAIRS):
            classification = "HistoricalShellCandidate"
            cluster_data[cluster]["historicalShellCandidates"] += 1
        else:
            classification = "InsufficientEvidence"
            cluster_data[cluster]["insufficientEvidence"] += 1
        a, b, c = triangle_vertices[triangle]
        cross = np.cross(b - a, c - a)
        magnitude = np.linalg.norm(cross)
        normal = cross / magnitude if magnitude > 1e-8 else np.zeros(3)
        data = cluster_data[cluster]
        data["triangles"] += 1
        data["boundsMin"] = np.minimum(data["boundsMin"], np.min(triangle_vertices[triangle], axis=0))
        data["boundsMax"] = np.maximum(data["boundsMax"], np.max(triangle_vertices[triangle], axis=0))
        records.append({
            "triangle": int(triangle),
            "cluster": cluster,
            "centroid": dict(zip("xyz", map(float, centroids[triangle]))),
            "normal": dict(zip("xyz", map(float, normal))),
            "areaSquareMeters": float(magnitude * 0.5),
            "validDepthObservations": int(valid_observations[triangle]),
            "foregroundContradictionPairs": int(foreground_pairs[triangle]),
            "backgroundContradictionPairs": int(background_pairs[triangle]),
            "nearestAbsoluteResidualMeters": float(nearest[triangle]) if np.isfinite(nearest[triangle]) else -1.0,
            "minimumResidualMeters": float(minimum[triangle]) if np.isfinite(minimum[triangle]) else 0.0,
            "maximumResidualMeters": float(maximum[triangle]) if np.isfinite(maximum[triangle]) else 0.0,
            "classification": classification,
        })
    clusters = []
    for index, data in enumerate(cluster_data):
        clusters.append({
            "cluster": index,
            "triangles": data["triangles"],
            "boundsMin": dict(zip("xyz", map(float, data["boundsMin"]))),
            "boundsMax": dict(zip("xyz", map(float, data["boundsMax"]))),
            "crossLayerCandidates": data["crossLayerCandidates"],
            "historicalShellCandidates": data["historicalShellCandidates"],
            "insufficientEvidence": data["insufficientEvidence"],
        })
    receipt = {
        "coordinateSpace": "QuestStage",
        "sourceMesh": str(mesh_path),
        "sourceMeshSha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest().upper(),
        "sourceTape": str(tape_path),
        "sourceTapeSha256": hashlib.sha256(tape_path.read_bytes()).hexdigest().upper(),
        "replayedPairs": frame_count,
        "directSupportMeters": DIRECT_SUPPORT_METERS,
        "contradictionMeters": CONTRADICTION_METERS,
        "classificationRule": "BothResidualSigns=CrossLayer;ThreeDistinctOneSignPairs=HistoricalShell",
        "totalTriangles": count,
        "unsupportedTriangles": len(records),
        "directDepthSupportRatio": 1.0 - len(records) / count,
        "clusters": len(clusters),
        "classificationCounts": {
            "crossLayerCandidates": sum(c["crossLayerCandidates"] for c in clusters),
            "historicalShellCandidates": sum(c["historicalShellCandidates"] for c in clusters),
            "insufficientEvidence": sum(c["insufficientEvidence"] for c in clusters),
        },
        "clusterReceipts": clusters,
        "triangleReceipts": records,
    }
    output_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("tape", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt = build(args.mesh, args.tape, args.output)
    print(json.dumps({
        key: receipt[key] for key in (
            "replayedPairs", "totalTriangles", "unsupportedTriangles",
            "directDepthSupportRatio", "clusters", "classificationCounts"
        )
    }, indent=2))


if __name__ == "__main__":
    main()
