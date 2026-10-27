"""Conservatively expand human scar edges into bounded causal chains.

Unlike ``detect_manual_edge_signature.py`` this detector does not treat a
matching Marching-Cubes pair or vertex connectivity as sufficient.  A new
edge must have strict source-history evidence on both owner triangles, and a
chain must contain multiple phase anchors, remain directionally coherent, and
have finite geometric extent.  The result is still a review candidate, not an
automatic deletion mask.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_supported_creases import face_geometry
from build_spatial_scar_receipt import read_ply
from detect_manual_edge_signature import edge_owners, sha256


DERIVED_VOXEL_METERS = 0.10
PLANE_EPSILON_METERS = 0.0005
MIN_EDGE_METERS = 0.015
MAX_EDGE_METERS = 0.160
MAX_TURN_DEGREES = 42.0
MIN_COMPONENT_EDGES = 3
MIN_COMPONENT_PHASE_EDGES = 2
MIN_COMPONENT_LENGTH_METERS = 0.12
MIN_COMPONENT_EXTENT_METERS = 0.10
MAX_COMPONENT_LENGTH_METERS = 2.40

STRICT_MECHANISMS = {
    "RepeatedMixedSignSourceVoxel",
    "SameViewRepeatedMixedSignSourceVoxel",
    "RepeatedTemporalSignSwitching",
}


def load_positive_edges(path, owners, faces):
    atlas = json.loads(path.read_text(encoding="utf-8-sig"))
    result = set()
    for row in atlas.get("selectedMeshRecords", []):
        triangle, a, b = int(row[0]), int(row[1]), int(row[2])
        edge = tuple(sorted((a, b)))
        if edge not in owners:
            raise ValueError(f"Selected edge {edge} is absent from source mesh")
        if a not in faces[triangle] or b not in faces[triangle]:
            raise ValueError(f"Selected face {triangle} does not own edge {edge}")
        result.add(edge)
    return result


def edge_type(edge, owners, records, vertices):
    adjacent = owners[edge]
    if len(adjacent) != 2:
        return "boundary"
    cells = np.asarray([records[t]["cell"] for t in adjacent], dtype=np.int64)
    delta = np.abs(cells[0] - cells[1])
    if int(np.sum(delta)) != 1:
        return "same-cell"
    axis = int(np.argmax(delta))
    boundary = float(np.max(cells[:, axis])) * DERIVED_VOXEL_METERS
    if float(np.max(np.abs(vertices[list(edge), axis] - boundary))) <= PLANE_EPSILON_METERS:
        return "phase-boundary"
    return "same-cell"


def configuration_key(edge, owners, records):
    return tuple(sorted(
        int(records[t]["finalCell"]["marchingCubesConfiguration"])
        for t in owners[edge]))


def dihedral_degrees(edge, owners, normals):
    adjacent = owners[edge]
    if len(adjacent) != 2:
        return 0.0
    cosine = abs(float(np.dot(normals[adjacent[0]], normals[adjacent[1]])))
    return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))


def owner_is_strict(record):
    mechanisms = set(record.get("mechanisms", []))
    evidence = record.get("cornerEvidence", {})
    return (
        STRICT_MECHANISMS <= mechanisms and
        int(evidence.get("sameViewRepeatedMixedCorners", 0)) >= 1 and
        int(evidence.get("temporalSwitchMaximum", 0)) >= 3
    )


def strict_source_edge(edge, owners, records):
    adjacent = owners[edge]
    return len(adjacent) == 2 and all(owner_is_strict(records[t]) for t in adjacent)


def unit_direction(edge, shared_vertex, vertices):
    other = edge[1] if edge[0] == shared_vertex else edge[0]
    value = vertices[other] - vertices[shared_vertex]
    length = float(np.linalg.norm(value))
    return value / max(length, 1e-9)


def coherent_components(edges, vertices):
    edges = set(edges)
    vertex_edges = defaultdict(list)
    for edge in edges:
        vertex_edges[edge[0]].append(edge)
        vertex_edges[edge[1]].append(edge)
    cosine_limit = math.cos(math.radians(MAX_TURN_DEGREES))
    neighbours = defaultdict(set)
    for vertex, incident in vertex_edges.items():
        for index, first in enumerate(incident):
            first_direction = unit_direction(first, vertex, vertices)
            for second in incident[index + 1:]:
                second_direction = unit_direction(second, vertex, vertices)
                if abs(float(np.dot(first_direction, second_direction))) >= cosine_limit:
                    neighbours[first].add(second)
                    neighbours[second].add(first)
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
            for neighbour in neighbours[edge]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    pending.append(neighbour)
        result.append(component)
    return result


def component_geometry(component, vertices):
    points = vertices[sorted({v for edge in component for v in edge})]
    extent = float(np.max(np.ptp(points, axis=0))) if len(points) else 0.0
    length = sum(float(np.linalg.norm(vertices[a] - vertices[b])) for a, b in component)
    return length, extent


def learned_rule(positives, owners, records, vertices, normals):
    phase = {e for e in positives if edge_type(e, owners, records, vertices) == "phase-boundary"}
    continuation = positives - phase
    phase_keys = {configuration_key(e, owners, records) for e in phase}
    continuation_keys = {configuration_key(e, owners, records) for e in continuation}
    positive_dihedrals = [dihedral_degrees(e, owners, normals) for e in positives]
    maximum_dihedral = min(55.0, max(8.0, float(np.percentile(positive_dihedrals, 90)) + 4.0))
    return phase_keys, continuation_keys, maximum_dihedral


def old_candidate_edges(path):
    receipt = json.loads(path.read_text(encoding="utf-8-sig"))
    categories = receipt.get("categories", {})
    names = ("highConfidencePhase", "highConfidenceContinuation", "probable")
    return {
        tuple(sorted((int(row[0]), int(row[1]))))
        for name in names for row in categories.get(name, [])
    }


def build(mesh_path, source_audit_path, atlas_paths, old_receipt_paths):
    vertices, faces = read_ply(mesh_path)
    audit = json.loads(source_audit_path.read_text(encoding="utf-8-sig"))
    records = audit["triangleReceipts"]
    if len(records) != len(faces):
        raise ValueError("Full-mesh source audit does not contain every face")
    mesh_hash = sha256(mesh_path)
    if mesh_hash != str(audit.get("sourceMeshSha256", "")).upper():
        raise ValueError("Mesh SHA-256 does not match full-mesh source audit")

    _, normals, _ = face_geometry(vertices, faces)
    owners = edge_owners(faces)
    positives_by_class = [load_positive_edges(p, owners, faces) for p in atlas_paths]
    positives = set().union(*positives_by_class)
    rules = [learned_rule(p, owners, records, vertices, normals) for p in positives_by_class]

    strict_phase = set()
    strict_pool = set()
    class_by_edge = {}
    for class_index, (phase_keys, continuation_keys, maximum_dihedral) in enumerate(rules):
        for edge in owners:
            edge_length = float(np.linalg.norm(vertices[edge[0]] - vertices[edge[1]]))
            if not (MIN_EDGE_METERS <= edge_length <= MAX_EDGE_METERS):
                continue
            if not strict_source_edge(edge, owners, records):
                continue
            if dihedral_degrees(edge, owners, normals) > maximum_dihedral:
                continue
            kind = edge_type(edge, owners, records, vertices)
            key = configuration_key(edge, owners, records)
            if kind == "phase-boundary" and key in phase_keys:
                strict_phase.add(edge)
                strict_pool.add(edge)
                class_by_edge.setdefault(edge, class_index)
            elif kind != "phase-boundary" and key in continuation_keys:
                strict_pool.add(edge)
                class_by_edge.setdefault(edge, class_index)

    kept_components = []
    for class_index in range(len(rules)):
        class_edges = {e for e in strict_pool if class_by_edge.get(e) == class_index}
        for component in coherent_components(class_edges, vertices):
            phase_count = len(set(component) & strict_phase)
            length, extent = component_geometry(component, vertices)
            if (
                len(component) >= MIN_COMPONENT_EDGES and
                phase_count >= MIN_COMPONENT_PHASE_EDGES and
                MIN_COMPONENT_LENGTH_METERS <= length <= MAX_COMPONENT_LENGTH_METERS and
                extent >= MIN_COMPONENT_EXTENT_METERS
            ):
                kept_components.append(set(component))

    kept = set().union(*kept_components) if kept_components else set()
    strict_phase_kept = (kept & strict_phase) - positives
    strict_chain_kept = (kept - strict_phase) - positives
    old_candidates = set().union(*(old_candidate_edges(p) for p in old_receipt_paths))
    withdrawn = old_candidates - kept - positives

    source_signature_edges = {
        edge for edge in owners
        if any("RepeatedMixedSignSourceVoxel" in records[t].get("mechanisms", [])
               for t in owners[edge])
    }
    categories = [
        sorted(positives),
        sorted(strict_phase_kept),
        sorted(strict_chain_kept),
        sorted(withdrawn),
    ]
    displayed = set().union(*map(set, categories))
    type_codes = {"phase-boundary": 0, "same-cell": 1, "boundary": 2}
    category_by_edge = {
        edge: category for category, edges in enumerate(categories) for edge in edges
    }
    metadata = []
    for edge in sorted(displayed):
        adjacent = owners[edge]
        configurations = list(configuration_key(edge, owners, records))
        temporal_maximum = max(
            int(records[t]["cornerEvidence"]["temporalSwitchMaximum"])
            for t in adjacent)
        layer_jump_maximum = max(
            int(records[t]["cornerEvidence"]["layerJumpMaximum"])
            for t in adjacent)
        metadata.append([
            edge[0], edge[1], category_by_edge[edge],
            type_codes[edge_type(edge, owners, records, vertices)], len(adjacent),
            configurations[0], configurations[1] if len(configurations) > 1 else -1,
            temporal_maximum, layer_jump_maximum,
            int(round(dihedral_degrees(edge, owners, normals) * 1000.0)),
            int(round(np.linalg.norm(vertices[edge[0]] - vertices[edge[1]]) * 1_000_000.0)),
        ])

    summary = {
        "meshTriangles": int(len(faces)),
        "meshEdges": len(owners),
        "manualPositiveEdges": len(positives),
        "manualPhaseBoundaryEdges": sum(
            edge_type(e, owners, records, vertices) == "phase-boundary" for e in positives),
        "manualContinuationEdges": sum(
            edge_type(e, owners, records, vertices) != "phase-boundary" for e in positives),
        "highConfidencePhaseEdges": len(strict_phase_kept),
        "highConfidenceContinuationEdges": len(strict_chain_kept),
        "highConfidenceNewEdges": len(strict_phase_kept) + len(strict_chain_kept),
        "probableNewEdges": len(withdrawn),
        "sourceSignatureEdges": len(source_signature_edges),
        "sourceSignatureExcludedEdges": len(source_signature_edges - displayed),
        "keptComponents": len(kept_components),
        "oldCandidateEdges": len(old_candidates - positives),
        "retainedOldCandidateEdges": len((old_candidates & kept) - positives),
        "withdrawnOldCandidateEdges": len(withdrawn),
    }
    receipt = {
        "kind": "RoomSetupReplicaV2CausalScarPatchReceiptV1",
        "sourceMesh": str(mesh_path),
        "sourceMeshSha256": mesh_hash,
        "sourceAudit": str(source_audit_path),
        "sourceAuditSha256": sha256(source_audit_path),
        "sourcePositiveAtlases": [str(p) for p in atlas_paths],
        "sourcePositiveAtlasSha256": [sha256(p) for p in atlas_paths],
        "sourceOldReceipts": [str(p) for p in old_receipt_paths],
        "rule": {
            "strictOwnerMechanisms": sorted(STRICT_MECHANISMS),
            "requiresEveryOwnerTriangle": True,
            "minimumSameViewRepeatedMixedCornersPerOwner": 1,
            "minimumTemporalSwitchMaximumPerOwner": 3,
            "maximumTurnDegrees": MAX_TURN_DEGREES,
            "minimumComponentEdges": MIN_COMPONENT_EDGES,
            "minimumComponentPhaseEdges": MIN_COMPONENT_PHASE_EDGES,
            "componentLengthMeters": [MIN_COMPONENT_LENGTH_METERS, MAX_COMPONENT_LENGTH_METERS],
            "minimumComponentExtentMeters": MIN_COMPONENT_EXTENT_METERS,
            "classMaximumDihedralDegrees": [r[2] for r in rules],
        },
        "summary": summary,
        "categories": {
            "manualPositive": [list(e) for e in categories[0]],
            "strictPhaseCore": [list(e) for e in categories[1]],
            "boundedChainExtension": [list(e) for e in categories[2]],
            "withdrawnOldRule": [list(e) for e in categories[3]],
        },
        "metadataRows": metadata,
        "interpretationBoundary": (
            "Strict results are causal review candidates, not confirmed defects. "
            "The rule requires direct source-history evidence on both owner faces "
            "and bounded directionally coherent components with multiple phase anchors. "
            "Only user review or same-tape intervention can confirm visible defects."
        ),
    }
    payload = {
        "v": np.rint(vertices.reshape(-1) * 1000.0).astype(np.int64).tolist(),
        "f": faces.reshape(-1).astype(np.int64).tolist(),
        "c": [[[e[0], e[1]] for e in values] for values in categories],
        "m": metadata,
        "s": summary,
    }
    return receipt, payload


def causal_template(source):
    # Replace longer explanatory phrases before their shorter labels.
    replacements = {
        "manual-edge-signature-3d": "causal-scar-patches-3d",
        "人工边图鉴 · 全网格同型定位": "因果链收紧 · 局部伤疤候选",
        "48 条人工边已固定为正样本": "A/B 人工边已合并为正样本",
        "高置信结果同时要求同源历史、相同 MC 构型家族、相容折角，以及相位边界或连接到相位边界的链结构。": "严格候选要求相邻两侧都具备反复混源与时间翻转，并形成含多个相位锚点的有限方向链。",
        "反复混层只作为来源证据；没有匹配几何与拓扑的宽泛混层边不会被判成伤疤。": "这些仍是复核候选；未经过人工确认或同回放干预消失前，不作为删除掩码。",
        "<strong>高置信相位边：</strong>跨相邻 10 cm 派生单元、落在共同采样平面，并复现正样本 MC 与混层时序特征。": "<strong>双侧证据相位核心：</strong>两个 owner 均有混源/时间翻转证据，且位于相邻 10 cm 单元共同采样平面。",
        "<strong>高置信链延伸：</strong>同一来源与构型家族，并在拓扑上连接高置信相位边。": "<strong>有限方向链延伸：</strong>方向连续、长度有限，并属于包含至少两个相位核心的局部链。",
        "<strong>待复核：</strong>同一图鉴家族，但时间翻转或折角条件被放宽，不进入自动修复。": "<strong>旧规则已撤回：</strong>先前仅凭单侧来源、MC 构型或宽松连通传播得到的结果，默认隐藏。",
        "高置信同型新边": "严格因果候选",
        "待复核候选": "旧规则已撤回",
        "放宽时间与折角条件": "默认隐藏，仅供前后对照",
        "已排除的宽泛混层边": "未进入局部链的混源边",
        "高置信相位边": "双侧证据相位核心",
        "高置信链延伸": "有限方向链延伸",
        "适配高置信": "适配严格候选",
        "人工边正样本与全房间同型候选的可旋转三维模型": "人工正样本、严格因果候选与撤回旧结果的可旋转三维模型",
        "const categoryNames = ['人工正样本', '高置信相位边', '高置信链延伸', '待复核候选'];": "const categoryNames = ['人工正样本', '双侧证据相位核心', '有限方向链延伸', '旧规则已撤回'];",
        "这是放宽规则后的待复核边，不进入自动退休或删除。": "这是旧规则已撤回的边，只用于和收紧结果对照。",
        "该边同时满足来源、构型和链结构证据；仍需在修复前后用同一回放输入复核。": "该边满足双侧来源、构型和有限链结构证据；仍需人工或同回放干预复核。",
        "待复核候选不会自动进入修复。": "旧规则已撤回，不会自动进入修复。",
        "颜色表达证据层级，不代表所有混层网格都被判错。": "颜色表达证据层级；严格候选仍不是自动删除掩码。",
        "放宽规则候选，只供人工复核，不进入自动修复。": "旧规则结果已撤回，仅供前后对照。",
        "当前图层是放宽规则候选，只供人工复核。": "当前图层是旧规则撤回结果，仅供前后对照。",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("source_audit", type=Path)
    parser.add_argument("atlas_a", type=Path)
    parser.add_argument("atlas_b", type=Path)
    parser.add_argument("old_receipt_a", type=Path)
    parser.add_argument("old_receipt_b", type=Path)
    parser.add_argument("output_receipt", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--visualization", type=Path)
    args = parser.parse_args()

    receipt, payload = build(
        args.mesh, args.source_audit,
        [args.atlas_a, args.atlas_b],
        [args.old_receipt_a, args.old_receipt_b])
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    result = dict(receipt["summary"])
    if args.template or args.visualization:
        if not args.template or not args.visualization:
            raise ValueError("--template and --visualization must be paired")
        template = causal_template(args.template.read_text(encoding="utf-8"))
        marker = "__MANUAL_EDGE_SIGNATURE_PAYLOAD__"
        if marker not in template:
            raise ValueError("Template lacks manual-edge payload marker")
        output = template.replace(marker, json.dumps(payload, separators=(",", ":")))
        args.visualization.parent.mkdir(parents=True, exist_ok=True)
        args.visualization.write_text(output, encoding="utf-8")
        size = args.visualization.stat().st_size
        if size >= 1_000_000:
            raise ValueError(f"Visualization exceeds 1 MB: {size} bytes")
        result["visualizationBytes"] = size
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
