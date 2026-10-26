#!/usr/bin/env python3
"""Convert Quest controller-painted triangle labels into a source audit receipt."""

import argparse
import hashlib
import json
from pathlib import Path

from build_spatial_scar_receipt import read_ply


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("tape", type=Path)
    parser.add_argument("human_labels", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    labels = json.loads(args.human_labels.read_text(encoding="utf-8-sig"))
    mesh_sha = sha256(args.mesh)
    expected_sha = str(labels.get("meshSha256", "")).upper()
    if expected_sha != mesh_sha:
        raise ValueError(
            "Human labels belong to a different final mesh: "
            f"labels={expected_sha} mesh={mesh_sha}")
    _, faces = read_ply(args.mesh)
    if int(labels.get("meshTriangles", -1)) != len(faces):
        raise ValueError(
            "Human-label triangle count does not match the PLY")

    by_triangle = {}
    for label in labels.get("labels", []):
        triangle = int(label["triangle"])
        if triangle < 0 or triangle >= len(faces):
            raise ValueError(f"Label triangle {triangle} is out of range")
        by_triangle[triangle] = label
    triangle_receipts = []
    for triangle in sorted(by_triangle):
        label = by_triangle[triangle]
        triangle_receipts.append({
            "triangle": triangle,
            "cluster": int(label.get("stroke", 0)),
            "classification": "HumanVisibleDefect",
            "labelOrdinal": int(label.get("ordinal", 0)),
            "hitStage": label.get("hitStage"),
        })

    receipt = {
        "coordinateSpace": "QuestStage",
        "sourceMesh": str(args.mesh),
        "sourceMeshSha256": mesh_sha,
        "sourceTape": str(args.tape),
        "sourceTapeSha256": sha256(args.tape),
        "sourceHumanLabels": str(args.human_labels),
        "sourceHumanLabelSchemaVersion": int(
            labels.get("schemaVersion", 0)),
        "selection": {
            "kind": "ControllerPaintedVisibleDefects",
            "triangles": len(triangle_receipts),
            "strokes": len({item["cluster"]
                            for item in triangle_receipts}),
        },
        "triangleReceipts": triangle_receipts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt["selection"], indent=2))


if __name__ == "__main__":
    main()
