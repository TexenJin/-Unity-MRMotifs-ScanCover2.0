#!/usr/bin/env python3
"""Audit human-labelled, directly-supported crease defects in a frozen mesh.

This tool deliberately starts from mesh geometry rather than the old
"unsupported triangle" receipt.  It welds coincident vertices, follows
edge-adjacent faces around each labelled seed, and records the local dihedral
and plane-residual field for later correlation with the frozen depth tape.
"""

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from build_spatial_scar_receipt import read_ply


WELD_METERS = 1.0e-4


def face_geometry(vertices, faces):
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0])
    magnitude = np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross)
    valid = magnitude > 1.0e-12
    normals[valid] = cross[valid] / magnitude[valid, np.newaxis]
    return triangles.mean(axis=1), normals, magnitude * 0.5


def welded_topology(vertices, faces):
    keys = np.rint(vertices / WELD_METERS).astype(np.int64)
    _, welded = np.unique(keys, axis=0, return_inverse=True)
    edge_faces = defaultdict(list)
    face_edges = []
    for face_index, indices in enumerate(faces):
        w = welded[indices]
        edges = []
        for a, b in ((w[0], w[1]), (w[1], w[2]), (w[2], w[0])):
            edge = (int(min(a, b)), int(max(a, b)))
            edge_faces[edge].append(face_index)
            edges.append(edge)
        face_edges.append(edges)
    return welded, edge_faces, face_edges


def angle_degrees(a, b):
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(dot)))
    # Mesh winding can flip at chunk seams. Geometry only needs the smaller
    # unsigned angle between the two tangent planes.
    return min(angle, 180.0 - angle)


def adjacency(normals, edge_faces):
    graph = defaultdict(list)
    boundary_edges = defaultdict(int)
    nonmanifold_edges = defaultdict(int)
    for edge, owners in edge_faces.items():
        if len(owners) == 1:
            boundary_edges[owners[0]] += 1
            continue
        if len(owners) != 2:
            for owner in owners:
                nonmanifold_edges[owner] += 1
            continue
        a, b = owners
        angle = angle_degrees(normals[a], normals[b])
        graph[a].append((b, angle, edge))
        graph[b].append((a, angle, edge))
    return graph, boundary_edges, nonmanifold_edges


def high_angle_edge_component(seed, threshold, vertices, faces, welded,
                              edge_faces, normals):
    representative = {}
    for vertex_index, welded_index in enumerate(welded):
        representative.setdefault(int(welded_index), vertices[vertex_index])
    high_edges = {}
    vertex_edges = defaultdict(list)
    for edge, owners in edge_faces.items():
        if len(owners) != 2:
            continue
        angle = angle_degrees(normals[owners[0]], normals[owners[1]])
        if angle < threshold:
            continue
        high_edges[edge] = (owners, angle)
        vertex_edges[edge[0]].append(edge)
        vertex_edges[edge[1]].append(edge)

    seed_vertices = set(int(value) for value in welded[faces[seed]])
    starts = [edge for vertex in seed_vertices
              for edge in vertex_edges.get(vertex, [])]
    if not starts:
        return {
            "thresholdDegrees": threshold,
            "edges": 0,
            "faces": 0,
            "totalLengthMeters": 0.0,
            "boundsMin": None,
            "boundsMax": None,
            "edgeRecords": [],
        }
    visited = set(starts)
    queue = deque(starts)
    while queue:
        edge = queue.popleft()
        for vertex in edge:
            for neighbour in vertex_edges[vertex]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
    points = []
    face_ids = set()
    records = []
    total_length = 0.0
    for edge in sorted(visited):
        a = representative[edge[0]]
        b = representative[edge[1]]
        owners, angle = high_edges[edge]
        length = float(np.linalg.norm(b - a))
        total_length += length
        points.extend((a, b))
        face_ids.update(int(owner) for owner in owners)
        records.append({
            "weldedEdge": [edge[0], edge[1]],
            "faces": [int(owners[0]), int(owners[1])],
            "angleDegrees": angle,
            "lengthMeters": length,
            "midpoint": ((a + b) * 0.5).tolist(),
        })
    point_array = np.asarray(points)
    return {
        "thresholdDegrees": threshold,
        "edges": len(visited),
        "faces": len(face_ids),
        "faceIds": sorted(face_ids),
        "totalLengthMeters": total_length,
        "boundsMin": np.min(point_array, axis=0).tolist(),
        "boundsMax": np.max(point_array, axis=0).tolist(),
        "edgeRecords": records,
    }


def trace_seed_crease(seed, threshold, vertices, faces, welded, edge_faces,
                      normals, maximum_turn_degrees=70.0):
    representative = {}
    for vertex_index, welded_index in enumerate(welded):
        representative.setdefault(int(welded_index), vertices[vertex_index])
    candidate_edges = {}
    vertex_edges = defaultdict(list)
    seed_edges = []
    seed_welded = welded[faces[seed]]
    seed_edge_keys = {
        (int(min(a, b)), int(max(a, b)))
        for a, b in ((seed_welded[0], seed_welded[1]),
                     (seed_welded[1], seed_welded[2]),
                     (seed_welded[2], seed_welded[0]))
    }
    for edge, owners in edge_faces.items():
        if len(owners) != 2:
            continue
        angle = angle_degrees(normals[owners[0]], normals[owners[1]])
        if angle < threshold:
            continue
        candidate_edges[edge] = (owners, angle)
        vertex_edges[edge[0]].append(edge)
        vertex_edges[edge[1]].append(edge)
        if edge in seed_edge_keys:
            seed_edges.append(edge)
    if not seed_edges:
        return {
            "thresholdDegrees": threshold,
            "seedEdge": None,
            "edges": 0,
            "faces": 0,
            "lengthMeters": 0.0,
            "edgeRecords": [],
        }
    seed_edge = max(seed_edges, key=lambda edge: candidate_edges[edge][1])
    minimum_alignment = float(np.cos(np.radians(maximum_turn_degrees)))

    def walk(previous, current, used):
        chain = []
        while True:
            incoming = representative[current] - representative[previous]
            incoming /= max(float(np.linalg.norm(incoming)), 1.0e-12)
            choices = []
            for edge in vertex_edges[current]:
                if edge in used:
                    continue
                other = edge[1] if edge[0] == current else edge[0]
                outgoing = representative[other] - representative[current]
                outgoing /= max(float(np.linalg.norm(outgoing)), 1.0e-12)
                alignment = float(np.dot(incoming, outgoing))
                if alignment < minimum_alignment:
                    continue
                angle = candidate_edges[edge][1]
                choices.append((alignment * angle, alignment, angle, edge,
                                other))
            if not choices:
                break
            _, _, _, edge, other = max(choices)
            used.add(edge)
            chain.append(edge)
            previous, current = current, other
        return chain

    used = {seed_edge}
    forward = walk(seed_edge[0], seed_edge[1], used)
    backward = walk(seed_edge[1], seed_edge[0], used)
    chain_edges = list(reversed(backward)) + [seed_edge] + forward
    records = []
    face_ids = set()
    length = 0.0
    for edge in chain_edges:
        owners, angle = candidate_edges[edge]
        a = representative[edge[0]]
        b = representative[edge[1]]
        edge_length = float(np.linalg.norm(b - a))
        length += edge_length
        face_ids.update(int(owner) for owner in owners)
        records.append({
            "weldedEdge": [edge[0], edge[1]],
            "faces": [int(owners[0]), int(owners[1])],
            "angleDegrees": angle,
            "lengthMeters": edge_length,
            "a": a.tolist(),
            "b": b.tolist(),
        })
    return {
        "thresholdDegrees": threshold,
        "maximumTurnDegrees": maximum_turn_degrees,
        "seedEdge": list(seed_edge),
        "seedEdgeAngleDegrees": candidate_edges[seed_edge][1],
        "edges": len(chain_edges),
        "faces": len(face_ids),
        "faceIds": sorted(face_ids),
        "lengthMeters": length,
        "edgeRecords": records,
    }


def enumerate_elongated_fold_components(threshold, vertices, welded,
                                        edge_faces, normals):
    representative = {}
    for vertex_index, welded_index in enumerate(welded):
        representative.setdefault(int(welded_index), vertices[vertex_index])
    candidates = {}
    vertex_edges = defaultdict(list)
    for edge, owners in edge_faces.items():
        if len(owners) != 2:
            continue
        angle = angle_degrees(normals[owners[0]], normals[owners[1]])
        if angle < threshold:
            continue
        candidates[edge] = (owners, angle)
        vertex_edges[edge[0]].append(edge)
        vertex_edges[edge[1]].append(edge)

    unvisited = set(candidates)
    components = []
    while unvisited:
        first = unvisited.pop()
        queue = deque([first])
        component_edges = [first]
        while queue:
            edge = queue.popleft()
            for vertex in edge:
                for neighbour in vertex_edges[vertex]:
                    if neighbour in unvisited:
                        unvisited.remove(neighbour)
                        component_edges.append(neighbour)
                        queue.append(neighbour)
        welded_vertices = sorted({vertex for edge in component_edges
                                  for vertex in edge})
        points = np.asarray([representative[value]
                             for value in welded_vertices])
        origin = points.mean(axis=0)
        if len(points) >= 3:
            _, _, vh = np.linalg.svd(points - origin,
                                      full_matrices=False)
            projections = (points - origin) @ vh.T
            axis_spans = np.ptp(projections, axis=0)
        else:
            axis_spans = np.sort(np.ptp(points, axis=0))[::-1]
            axis_spans = np.pad(axis_spans, (0, 3-len(axis_spans)))
        axis_spans = np.sort(axis_spans)[::-1]
        angles = np.asarray([candidates[edge][1]
                             for edge in component_edges])
        faces = sorted({int(face) for edge in component_edges
                        for face in candidates[edge][0]})
        component_normals = normals[faces]
        normal_covariance = component_normals.T @ component_normals
        normal_eigenvalues = np.linalg.eigvalsh(normal_covariance)
        normal_coherence = float(normal_eigenvalues[-1] /
                                 max(np.sum(normal_eigenvalues), 1.0e-12))
        total_length = sum(float(np.linalg.norm(
            representative[edge[1]] - representative[edge[0]]))
            for edge in component_edges)
        record = {
            "edges": len(component_edges),
            "faces": len(faces),
            "faceIds": faces,
            "weldedVertices": welded_vertices,
            "axisSpansMeters": axis_spans.tolist(),
            "elongation": float(axis_spans[0] /
                                 max(axis_spans[1], 1.0e-6)),
            "totalLengthMeters": total_length,
            "angleMedianDegrees": float(np.median(angles)),
            "angleP95Degrees": float(np.percentile(angles, 95.0)),
            "normalCoherence": normal_coherence,
            "boundsMin": np.min(points, axis=0).tolist(),
            "boundsMax": np.max(points, axis=0).tolist(),
        }
        record["supportedCreaseCandidate"] = bool(
            record["edges"] >= 6 and axis_spans[0] >= 0.50 and
            record["elongation"] >= 4.0 and
            record["angleMedianDegrees"] <= 65.0 and
            normal_coherence >= 0.65)
        components.append(record)

    components.sort(key=lambda value: (
        not value["supportedCreaseCandidate"],
        -value["axisSpansMeters"][0], -value["elongation"],
        -value["edges"]))
    for index, component in enumerate(components):
        component["component"] = index
    return components


def collect_ring(seed, graph, rings):
    distance = {seed: 0}
    queue = deque([seed])
    while queue:
        face = queue.popleft()
        if distance[face] >= rings:
            continue
        for neighbour, _, _ in graph.get(face, []):
            if neighbour not in distance:
                distance[neighbour] = distance[face] + 1
                queue.append(neighbour)
    return distance


def fit_plane(points):
    origin = points.mean(axis=0)
    _, singular, vh = np.linalg.svd(points - origin, full_matrices=False)
    normal = vh[-1]
    residual = np.abs((points - origin) @ normal)
    return origin, normal, residual, singular


def describe_seed(seed, vertices, faces, centroids, normals, areas, welded,
                  edge_faces, graph, boundary_edges, nonmanifold_edges,
                  rings):
    if seed < 0 or seed >= len(faces):
        raise ValueError(f"Triangle {seed} is outside 0..{len(faces)-1}")
    ring = collect_ring(seed, graph, rings)
    local_faces = np.array(sorted(ring), dtype=np.int64)
    local_vertices = vertices[faces[local_faces]].reshape(-1, 3)
    plane_origin, plane_normal, plane_residual, singular = fit_plane(
        local_vertices)

    internal_edges = []
    seen = set()
    for face in local_faces:
        for neighbour, angle, edge in graph.get(int(face), []):
            if neighbour not in ring or edge in seen:
                continue
            seen.add(edge)
            internal_edges.append({
                "faceA": int(face),
                "faceB": int(neighbour),
                "angleDegrees": angle,
                "edge": [edge[0], edge[1]],
            })
    internal_edges.sort(key=lambda value: value["angleDegrees"], reverse=True)
    angles = np.array([edge["angleDegrees"] for edge in internal_edges],
                      dtype=np.float64)
    seed_neighbours = sorted(({
        "triangle": int(neighbour),
        "angleDegrees": angle,
        "edge": [edge[0], edge[1]],
    } for neighbour, angle, edge in graph.get(seed, [])),
        key=lambda value: value["angleDegrees"], reverse=True)

    return {
        "seedTriangle": seed,
        "seedCentroid": centroids[seed].tolist(),
        "seedNormal": normals[seed].tolist(),
        "seedAreaSquareMeters": float(areas[seed]),
        "seedBoundaryEdges": int(boundary_edges.get(seed, 0)),
        "seedNonmanifoldEdges": int(nonmanifold_edges.get(seed, 0)),
        "edgeNeighbours": seed_neighbours,
        "ringCount": len(local_faces),
        "ringDepth": rings,
        "ringFaceIds": local_faces.tolist(),
        "ringBoundsMin": np.min(local_vertices, axis=0).tolist(),
        "ringBoundsMax": np.max(local_vertices, axis=0).tolist(),
        "planeOrigin": plane_origin.tolist(),
        "planeNormal": plane_normal.tolist(),
        "planeResidualMedianMeters": float(np.median(plane_residual)),
        "planeResidualP95Meters": float(np.percentile(plane_residual, 95.0)),
        "planeResidualMaximumMeters": float(np.max(plane_residual)),
        "planeSingularValues": singular.tolist(),
        "internalEdgeCount": len(internal_edges),
        "dihedralMedianDegrees": float(np.median(angles)) if len(angles) else 0.0,
        "dihedralP90Degrees": float(np.percentile(angles, 90.0)) if len(angles) else 0.0,
        "dihedralP95Degrees": float(np.percentile(angles, 95.0)) if len(angles) else 0.0,
        "dihedralMaximumDegrees": float(np.max(angles)) if len(angles) else 0.0,
        "largestDihedralEdges": internal_edges[:32],
        "highAngleComponents": [high_angle_edge_component(
            seed, threshold, vertices, faces, welded, edge_faces, normals)
            for threshold in (8.0, 12.0, 16.0, 20.0, 24.0)],
        "seedCreaseTraces": [trace_seed_crease(
            seed, threshold, vertices, faces, welded, edge_faces, normals)
            for threshold in (8.0, 12.0, 16.0, 20.0, 24.0)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("seeds", nargs="+", type=int)
    parser.add_argument("--rings", type=int, default=10)
    args = parser.parse_args()

    vertices, faces = read_ply(args.mesh)
    centroids, normals, areas = face_geometry(vertices, faces)
    welded, edge_faces, _ = welded_topology(vertices, faces)
    graph, boundary, nonmanifold = adjacency(normals, edge_faces)
    fold_components = enumerate_elongated_fold_components(
        24.0, vertices, welded, edge_faces, normals)
    result = {
        "sourceMesh": str(args.mesh),
        "triangleCount": int(len(faces)),
        "weldMeters": WELD_METERS,
        "ringDepth": args.rings,
        "foldComponentThresholdDegrees": 24.0,
        "foldComponents": fold_components,
        "foldCandidateCount": sum(value["supportedCreaseCandidate"]
                                  for value in fold_components),
        "seeds": [describe_seed(
            seed, vertices, faces, centroids, normals, areas, welded,
            edge_faces, graph,
            boundary, nonmanifold, args.rings) for seed in args.seeds],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = []
    for seed in result["seeds"]:
        containing = [value for value in fold_components
                      if seed["seedTriangle"] in value["faceIds"] or any(
                          vertex in value["weldedVertices"]
                          for vertex in welded[faces[seed["seedTriangle"]]])]
        summary.append({
            "seedTriangle": seed["seedTriangle"],
            "centroid": seed["seedCentroid"],
            "boundaryEdges": seed["seedBoundaryEdges"],
            "neighbours": len(seed["edgeNeighbours"]),
            "ringFaces": seed["ringCount"],
            "planeP95mm": round(seed["planeResidualP95Meters"] * 1000.0, 2),
            "dihedralP95deg": round(seed["dihedralP95Degrees"], 2),
            "dihedralMaxDeg": round(seed["dihedralMaximumDegrees"], 2),
            "seedNeighbourAngles": [round(value["angleDegrees"], 2)
                                    for value in seed["edgeNeighbours"]],
            "highAngleComponents": [{
                "threshold": value["thresholdDegrees"],
                "edges": value["edges"],
                "faces": value["faces"],
                "lengthM": round(value["totalLengthMeters"], 3),
                "boundsMin": value["boundsMin"],
                "boundsMax": value["boundsMax"],
            } for value in seed["highAngleComponents"]],
            "seedCreaseTraces": [{
                "threshold": value["thresholdDegrees"],
                "edges": value["edges"],
                "faces": value["faces"],
                "lengthM": round(value["lengthMeters"], 3),
                "seedAngle": round(value.get(
                    "seedEdgeAngleDegrees", 0.0), 2),
            } for value in seed["seedCreaseTraces"]],
            "globalFoldComponents": [{
                "component": value["component"],
                "candidate": value["supportedCreaseCandidate"],
                "edges": value["edges"],
                "faces": value["faces"],
                "axisSpansM": [round(item, 3)
                               for item in value["axisSpansMeters"]],
                "elongation": round(value["elongation"], 2),
                "angleMedian": round(value["angleMedianDegrees"], 2),
                "normalCoherence": round(value["normalCoherence"], 3),
            } for value in containing],
        })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
