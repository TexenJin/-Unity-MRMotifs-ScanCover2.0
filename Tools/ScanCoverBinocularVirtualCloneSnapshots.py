#!/usr/bin/env python3
"""Generate binocular virtual-clone Raw Depth snapshots from ScanCover metadata.

The Unity exporter writes a lightweight virtual_clone_input folder beside each
real binocular Raw Depth snapshot. This script replays those center poses
against a Replica/shell mesh, raycasts right-eye then left-eye samples, and
writes CSVs that keep the same core columns as real room_raw_depth_snapshots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_session", type=Path, help="Snapshot folder, session folder, or virtual_clone_input folder.")
    parser.add_argument("--truth-mesh", type=Path, required=True, help="Replica room mesh or reconstructed shell mesh.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all metadata frames.")
    parser.add_argument("--width", type=int, default=0, help="Override per-eye ray width. 0 uses metadata.")
    parser.add_argument("--height", type=int, default=0, help="Override per-eye ray height. 0 uses metadata.")
    parser.add_argument("--eye-baseline", type=float, default=0.0, help="Override eye baseline meters. 0 uses metadata.")
    parser.add_argument("--min-depth", type=float, default=0.20)
    parser.add_argument("--max-depth", type=float, default=5.00)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=15319)
    return parser.parse_args()


def find_virtual_clone_input(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name == "virtual_clone_input":
        return path
    candidate = path / "virtual_clone_input"
    if candidate.exists():
        return candidate
    matches = sorted(path.rglob("virtual_clone_input"), key=lambda item: str(item).lower()) if path.is_dir() else []
    if matches:
        return matches[0]
    raise FileNotFoundError(f"virtual_clone_input folder not found under {path}")


def load_metadata(input_dir: Path) -> list[dict[str, Any]]:
    manifest = input_dir / "virtual_clone_input_manifest.csv"
    if manifest.exists():
        rows: list[dict[str, Any]] = []
        with manifest.open("r", encoding="utf-8-sig", newline="") as fh:
            data_lines = [line for line in fh if not line.startswith("#")]
        for row in csv.DictReader(data_lines):
            metadata_path = Path(row["metadataJson"])
            if not metadata_path.is_absolute():
                metadata_path = (input_dir / metadata_path).resolve()
            rows.append(json.loads(metadata_path.read_text(encoding="utf-8-sig")))
        return rows
    return [
        json.loads(path.read_text(encoding="utf-8-sig"))
        for path in sorted(input_dir.glob("*_virtual_clone_input.json"))
    ]


def normalize(value: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if length <= 1e-8 or not np.isfinite(length):
        return np.asarray(fallback, dtype=np.float64)
    return value / length


def load_scene(mesh_path: Path) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Could not load mesh: {mesh_path}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def make_rays(
    pose: dict[str, Any],
    eye: str,
    width: int,
    height: int,
    fov_degrees: float,
    aspect: float,
    baseline: float,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(pose.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
    forward = normalize(np.asarray(pose.get("forward", [0.0, 0.0, 1.0]), dtype=np.float64), (0.0, 0.0, 1.0))
    right = normalize(np.asarray(pose.get("right", [1.0, 0.0, 0.0]), dtype=np.float64), (1.0, 0.0, 0.0))
    up = normalize(np.asarray(pose.get("up", [0.0, 1.0, 0.0]), dtype=np.float64), (0.0, 1.0, 0.0))
    eye_sign = 1.0 if eye.lower() == "right" else -1.0
    origin = position + right * (baseline * 0.5 * eye_sign)

    fov_y = math.radians(float(fov_degrees))
    tan_y = math.tan(fov_y * 0.5)
    tan_x = tan_y * float(aspect)
    rays = np.zeros((width * height, 6), dtype=np.float32)
    directions = np.zeros((width * height, 3), dtype=np.float64)
    index = 0
    for y in range(height):
        ndc_y = 1.0 - (2.0 * (y + 0.5) / height)
        for x in range(width):
            ndc_x = (2.0 * (x + 0.5) / width) - 1.0
            direction = normalize(forward + right * (ndc_x * tan_x) + up * (ndc_y * tan_y), (0.0, 0.0, 1.0))
            rays[index, :3] = origin.astype(np.float32)
            rays[index, 3:] = direction.astype(np.float32)
            directions[index] = direction
            index += 1
    return rays, directions


def cast_eye(
    scene: o3d.t.geometry.RaycastingScene,
    metadata: dict[str, Any],
    eye_info: dict[str, Any],
    width: int,
    height: int,
    baseline: float,
    rng: np.random.Generator,
    noise_std: float,
    min_depth: float,
    max_depth: float,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    eye = str(eye_info["eye"])
    rays, directions = make_rays(
        metadata["pose"],
        eye,
        width,
        height,
        float(metadata.get("fieldOfViewDegrees", 100.2439)),
        float(metadata.get("aspect", width / max(1, height))),
        baseline,
    )
    answer = scene.cast_rays(o3d.core.Tensor(rays))
    t_hit = answer["t_hit"].numpy()
    primitive_normals = answer["primitive_normals"].numpy()
    origins = rays[:, :3].astype(np.float64)

    rows: list[dict[str, Any]] = []
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for i in range(width * height):
        x = i % width
        y = i // width
        depth = float(t_hit[i])
        valid = math.isfinite(depth) and min_depth <= depth <= max_depth
        row: dict[str, Any] = {
            "index": int(eye_info.get("startIndex", 0)) + i,
            "pixelX": x,
            "pixelY": y,
            "u": x / max(1, width - 1),
            "v": y / max(1, height - 1),
            "valid": 1 if valid else 0,
            "depthM": "",
            "forwardDepthM": "",
            "worldX": "",
            "worldY": "",
            "worldZ": "",
            "normalX": "",
            "normalY": "",
            "normalZ": "",
            "confidence": "",
            "eye": eye,
        }
        if valid:
            normal = normalize(np.asarray(primitive_normals[i], dtype=np.float64), (0.0, 1.0, 0.0))
            point = origins[i] + directions[i] * depth
            if noise_std > 0.0:
                point = point + rng.normal(0.0, noise_std * (1.0 + 0.25 * depth), 3)
            forward_depth = float(np.dot(point - origins[i], directions[i]))
            row.update(
                {
                    "depthM": depth,
                    "forwardDepthM": forward_depth,
                    "worldX": point[0],
                    "worldY": point[1],
                    "worldZ": point[2],
                    "normalX": normal[0],
                    "normalY": normal[1],
                    "normalZ": normal[2],
                    "confidence": 1.0,
                }
            )
            points.append(point)
            normals.append(normal)
        rows.append(row)
    return rows, points, normals


def fmt(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_snapshot_csv(path: Path, width: int, height: int, rows: list[dict[str, Any]]) -> None:
    header = [
        "index", "pixelX", "pixelY", "u", "v", "valid", "depthM", "forwardDepthM",
        "worldX", "worldY", "worldZ", "normalX", "normalY", "normalZ", "confidence", "eye",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("# ScanCover virtual binocular raw-depth snapshot\n")
        fh.write(f"resolution={width}x{height}\n")
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key, "")) for key in header})


def write_cloud(path: Path, points: list[np.ndarray], normals: list[np.ndarray]) -> None:
    if not points:
        return
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    cloud.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.tile(np.asarray((0.0, 0.85, 1.0)), (len(points), 1)))
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False)


def main() -> int:
    args = parse_args()
    input_dir = find_virtual_clone_input(args.snapshot_session)
    metadata_rows = load_metadata(input_dir)
    if args.max_frames > 0:
        metadata_rows = metadata_rows[: args.max_frames]
    if not metadata_rows:
        raise RuntimeError(f"No virtual clone metadata found in {input_dir}")

    out_dir = args.out or input_dir.parent / "virtual_clone_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = out_dir / "room_raw_depth_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    scene = load_scene(args.truth_mesh)
    rng = np.random.default_rng(args.seed)
    manifest_rows: list[dict[str, Any]] = []
    all_points: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []

    for frame_i, metadata in enumerate(metadata_rows):
        eyes = metadata.get("eyes", [])
        if len(eyes) < 2:
            continue
        right_eye = next((item for item in eyes if str(item.get("eye", "")).lower() == "right"), eyes[0])
        left_eye = next((item for item in eyes if str(item.get("eye", "")).lower() == "left"), eyes[1])
        width = args.width or int(right_eye.get("width", metadata.get("width", 160)))
        right_height = args.height or int(right_eye.get("height", max(1, int(right_eye.get("count", width) / max(1, width)))))
        left_height = args.height or int(left_eye.get("height", right_height))
        baseline = args.eye_baseline if args.eye_baseline > 0.0 else float(metadata.get("eyeBaselineMeters", 0.063))

        right_rows, right_points, right_normals = cast_eye(
            scene, metadata, right_eye, width, right_height, baseline, rng, args.noise_std, args.min_depth, args.max_depth
        )
        left_rows, left_points, left_normals = cast_eye(
            scene, metadata, left_eye, width, left_height, baseline, rng, args.noise_std, args.min_depth, args.max_depth
        )
        rows = right_rows + left_rows
        height = right_height + left_height
        frame = str(metadata.get("frame") or f"frame_{frame_i:04d}")
        path = snapshot_dir / f"{frame}_virtual_raw_snapshot.csv"
        write_snapshot_csv(path, width, height, rows)
        valid = sum(1 for row in rows if row["valid"] == 1)
        manifest_rows.append(
            {
                "frame": frame,
                "rawDepthFrame": metadata.get("rawDepthFrame", ""),
                "totalPixels": len(rows),
                "validPixels": valid,
                "width": width,
                "height": height,
                "rightCount": len(right_rows),
                "leftCount": len(left_rows),
                "path": str(path),
                "status": "exported",
            }
        )
        all_points.extend(right_points)
        all_points.extend(left_points)
        all_normals.extend(right_normals)
        all_normals.extend(left_normals)

    with (snapshot_dir / "room_raw_depth_snapshot_manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        fh.write("# ScanCover virtual binocular raw-depth snapshots\n")
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    cloud_path = out_dir / "virtual_binocular_clone_points.ply"
    write_cloud(cloud_path, all_points, all_normals)
    report = {
        "schema": "ScanCoverBinocularVirtualCloneSnapshots/v1",
        "sourceVirtualCloneInput": str(input_dir),
        "truthMesh": str(args.truth_mesh),
        "frames": len(manifest_rows),
        "points": len(all_points),
        "snapshotDirectory": str(snapshot_dir),
        "pointCloud": str(cloud_path),
    }
    (out_dir / "virtual_binocular_clone_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
