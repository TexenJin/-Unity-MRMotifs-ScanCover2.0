#!/usr/bin/env python3
"""Build a source-audit receipt for a human-labelled supported crease."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument("tape", type=Path)
    parser.add_argument("crease_audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    audit = json.loads(args.crease_audit.read_text(encoding="utf-8-sig"))
    seed = next((value for value in audit["seeds"]
                 if int(value["seedTriangle"]) == args.seed), None)
    if seed is None:
        raise ValueError(f"Seed {args.seed} is absent from crease audit")
    local_component = next(
        value for value in seed["highAngleComponents"]
        if float(value["thresholdDegrees"]) == 24.0)
    local_faces = set(int(value) for value in local_component["faceIds"])
    local_faces.add(args.seed)

    receipt = {
        "coordinateSpace": "QuestStage",
        "sourceMesh": str(args.mesh),
        "sourceMeshSha256": sha256(args.mesh),
        "sourceTape": str(args.tape),
        "sourceTapeSha256": sha256(args.tape),
        "selection": {
            "kind": "HumanPositiveSupportedCrease",
            "seedTriangle": args.seed,
            "dihedralThresholdDegrees": 24.0,
            "triangles": len(local_faces),
            "connectedHighAngleEdges": int(local_component["edges"]),
            "totalEdgeLengthMeters": float(
                local_component["totalLengthMeters"]),
            "boundsMin": local_component["boundsMin"],
            "boundsMax": local_component["boundsMax"],
        },
        "triangleReceipts": [{
            "triangle": face,
            "cluster": 0,
            "classification": "HumanPositiveSupportedCrease",
        } for face in sorted(local_faces)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt["selection"], indent=2))


if __name__ == "__main__":
    main()
