#!/usr/bin/env python3
"""Join final malformed TSDF zero crossings to their exact fusion transaction chain."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from collections import Counter, OrderedDict
from pathlib import Path


OPS = {0: "none", 1: "seed", 2: "positive_blend", 3: "lifetime_reset"}
BLOCKS = {
    0: "none", 1: "gungel", 2: "exclusion", 3: "normal",
    4: "dilation_occlusion", 5: "truncation_band", 6: "abstain_neighbour",
    9: "raw_projective_gate",
    10: "dilation_supply", 11: "fov_motion_authority", 12: "frozen",
    13: "blend_quantized", 14: "mature_discount", 15: "near_distance",
    16: "invalid_depth", 17: "seed_quality", 18: "motion_seed",
    19: "behind_camera", 20: "outside_fov",
}
SUPPORT_OPS = {0: "none", 1: "seed", 2: "positive_growth", 3: "carve",
               4: "freeze", 5: "unfreeze", 6: "reset"}


def rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def extract_stage_bins(path: Path):
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    try:
        start = lines.index("stage_responsibility_spatial_csv:") + 1
    except ValueError:
        return []
    section = []
    for line in lines[start:]:
        if not line.strip() or line.endswith(":"):
            break
        section.append(line)
    if len(section) < 2:
        return []
    parsed = list(csv.DictReader(section))
    convicted = [r for r in parsed if r.get("first_deformation_stage") == "tsdf_zero_crossing"]
    if convicted:
        return convicted
    return [r for r in parsed if float(r.get("crossing_plane_rms_vox") or 0) >= 0.20]


def signed_byte(value):
    return value - 256 if value >= 128 else value


def snorm8(value):
    return max(-1.0, signed_byte(value) / 127.0)


def packed_unorm_to_snorm(value):
    return value / 255.0 * 2.0 - 1.0


CUBE_CORNERS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))
CUBE_EDGES = ((0, 1), (0, 2), (1, 3), (2, 3),
              (4, 5), (4, 6), (5, 7), (6, 7),
              (0, 4), (1, 5), (2, 6), (3, 7))


def derive_paper_holes(session, tsdf, nx, ny, nz, voxel_size, min_weight):
    """Compare final TSDF crossings with exact committed-paper spatial occupancy.

    This deliberately does not invent a surface where final TSDF has no sign
    change.  It separates weak TSDF support from a mature crossing that was
    lost later, and labels one-cell paper-edge neighbours as context rather
    than direct holes.
    """
    paper = session / "artifacts" / "paper_audit"
    schema_path = paper / "production_paper_occupancy_schema.json"
    binary_path = paper / "production_paper_occupancy.r32_uint.bin"
    schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
    cx, cy, cz = schema["cellCount"]
    if schema.get("paperStrideVoxels") != 1 or [cx, cy, cz] != [nx - 1, ny - 1, nz - 1]:
        raise SystemExit("production paper occupancy lattice does not match final TSDF")
    if schema.get("productionTriangles", 0) <= 0:
        raise SystemExit("production paper occupancy reports no production triangles")
    if schema.get("rasterizedTriangles") != schema.get("productionTriangles"):
        raise SystemExit("production paper occupancy raster is incomplete")
    occupancy = binary_path.read_bytes()
    if len(occupancy) != cx * cy * cz * 4:
        raise SystemExit("production paper occupancy byte length does not match schema")

    def voxel_flat(x, y, z):
        return x + nx * (y + ny * z)

    def occupancy_flat(x, y, z):
        return x + cx * (y + cy * z)

    def occupied(x, y, z):
        if not (0 <= x < cx and 0 <= y < cy and 0 <= z < cz):
            return False
        return struct.unpack_from("<I", occupancy, occupancy_flat(x, y, z) * 4)[0] != 0xFFFFFFFF

    def voxel_center(axis, size):
        return (axis + 0.5 - size / 2.0) * voxel_size

    output = []
    for z in range(nz - 1):
        for y in range(ny - 1):
            for x in range(nx - 1):
                values = []
                for dx, dy, dz in CUBE_CORNERS:
                    index = voxel_flat(x + dx, y + dy, z + dz)
                    values.append((snorm8(tsdf[index * 2]),
                                   abs(snorm8(tsdf[index * 2 + 1])),
                                   x + dx, y + dy, z + dz))
                raw_points, mature_points = [], []
                for a, b in CUBE_EDGES:
                    d0, w0, x0, y0, z0 = values[a]
                    d1, w1, x1, y1, z1 = values[b]
                    crosses = (d0 == 0.0 or d1 == 0.0 or (d0 < 0.0) != (d1 < 0.0)) and not (d0 == 0.0 and d1 == 0.0)
                    if not crosses or w0 <= 0.0 or w1 <= 0.0:
                        continue
                    t = abs(d0) / max(abs(d0) + abs(d1), 1e-9)
                    point = [voxel_center(x0, nx) + (x1 - x0) * voxel_size * t,
                             voxel_center(y0, ny) + (y1 - y0) * voxel_size * t,
                             voxel_center(z0, nz) + (z1 - z0) * voxel_size * t]
                    raw_points.append(point)
                    if w0 >= min_weight and w1 >= min_weight:
                        mature_points.append(point)
                if not raw_points:
                    continue

                target_bins = set()
                for point in mature_points or raw_points:
                    target_bins.add(tuple(max(0, min([cx, cy, cz][axis] - 1,
                        int(math.floor(point[axis] / voxel_size + [nx, ny, nz][axis] / 2.0))))
                        for axis in range(3)))
                exact_paper = any(occupied(*cell) for cell in target_bins)
                if exact_paper:
                    continue
                neighbour_paper = any(occupied(px + dx, py + dy, pz + dz)
                    for px, py, pz in target_bins
                    for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
                points = mature_points or raw_points
                centre = [sum(p[axis] for p in points) / len(points) for axis in range(3)]
                if mature_points:
                    cause = ("mature_crossing_near_paper_edge" if neighbour_paper
                             else "mature_crossing_without_paper")
                else:
                    cause = ("weak_crossing_near_paper_edge" if neighbour_paper
                             else "raw_crossing_below_weight_without_paper")
                output.append({
                    "cell_x": x, "cell_y": y, "cell_z": z,
                    "world_x_m": centre[0], "world_y_m": centre[1], "world_z_m": centre[2],
                    "cause": cause, "final_published": "0",
                    "empty_surface_halo": "1" if neighbour_paper else "0",
                    "raw_edges": len(raw_points), "mature_edges": len(mature_points),
                    "paper_target_bins": len(target_bins),
                })
    return output


def mat_vec(matrix, vector):
    return [sum(matrix[r * 4 + c] * vector[c] for c in range(4)) for r in range(4)]


def local_to_world(matrix, point):
    value = mat_vec(matrix, [point[0], point[1], point[2], 1.0])
    w = value[3] if abs(value[3]) > 1e-9 else 1.0
    return [value[0] / w, value[1] / w, value[2] / w]


def resolve_frame_file(session: Path, lane: str, name: str):
    if not name:
        return None
    candidate = Path(name)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for base in (session, session / lane / "frames"):
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


class ByteCache:
    def __init__(self, limit=8):
        self.limit = limit
        self.items = OrderedDict()

    def get(self, path):
        if path is None:
            return None
        key = str(path)
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        data = path.read_bytes()
        self.items[key] = data
        if len(self.items) > self.limit:
            self.items.popitem(last=False)
        return data


def sample_f32(data, width, height, u, v):
    if data is None or width <= 0 or height <= 0 or not (0 <= u <= 1 and 0 <= v <= 1):
        return None
    x = min(width - 1, max(0, int(u * width)))
    y = min(height - 1, max(0, int(v * height)))
    offset = 4 * (x + width * y)
    if offset + 4 > len(data):
        return None
    return struct.unpack_from("<f", data, offset)[0]


def sample_unorm16(data, width, height, u, v):
    if data is None or width <= 0 or height <= 0 or not (0 <= u <= 1 and 0 <= v <= 1):
        return None
    x = min(width - 1, max(0, int(u * width)))
    y = min(height - 1, max(0, int(v * height)))
    offset = 2 * (x + width * y)
    if offset + 2 > len(data):
        return None
    return struct.unpack_from("<H", data, offset)[0] / 65535.0


def sample_depth(data, width, height, u, v, graphics_format):
    if graphics_format == "R16_UNorm":
        return sample_unorm16(data, width, height, u, v)
    return sample_f32(data, width, height, u, v)


def ndc_to_linear(ndc, projection):
    if ndc is None or ndc <= 0 or not math.isfinite(ndc):
        return None
    z = ndc * 2.0 - 1.0
    denominator = z + projection[10]
    if abs(denominator) < 1e-9:
        return None
    return abs(projection[11] / denominator)


def project_sample(session, fusion_row, pair_row, world, cache):
    result = {"u": "", "v": "", "fusionDepthM": "", "platformRawDepthM": "",
              "processedDepthM": "", "processedMinusRawMm": "", "rawProjectiveSdfMm": ""}
    if not fusion_row:
        return result
    meta_path = resolve_frame_file(session, "fusion_inputs", fusion_row.get("metaFile", ""))
    if meta_path is None:
        return result
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        eye = int(meta.get("recordedEyeIndex", 1))
        projection = meta["projection"][eye]
        view = meta["view"][eye]
        clip = mat_vec(projection, mat_vec(view, [world[0], world[1], world[2], 1.0]))
        if abs(clip[3]) < 1e-9:
            return result
        u = clip[0] / clip[3] * 0.5 + 0.5
        v = clip[1] / clip[3] * 0.5 + 0.5
        result["u"], result["v"] = f"{u:.6f}", f"{v:.6f}"
        depth_meta = meta["textures"]["depth"]
        depth_path = resolve_frame_file(session, "fusion_inputs", fusion_row.get("depthFile", ""))
        fusion_ndc = sample_depth(cache.get(depth_path), int(depth_meta["width"]),
                                  int(depth_meta["height"]), u, v,
                                  depth_meta.get("graphicsFormat", ""))
        fusion_m = ndc_to_linear(fusion_ndc, projection)
        if fusion_m is not None:
            result["fusionDepthM"] = f"{fusion_m:.6f}"
            camera_point = mat_vec(view, [world[0], world[1], world[2], 1.0])
            camera_w = camera_point[3] if abs(camera_point[3]) > 1e-9 else 1.0
            camera_xyz = [camera_point[i] / camera_w for i in range(3)]
            camera_depth = abs(camera_xyz[2])
            ray_scale = (math.sqrt(sum(component * component for component in camera_xyz)) /
                         max(camera_depth, 1e-9))
            result["rawProjectiveSdfMm"] = f"{(fusion_m - camera_depth) * ray_scale * 1000.0:.3f}"
        if pair_row:
            pw, ph = int(pair_row["width"]), int(pair_row["height"])
            raw_path = resolve_frame_file(session, "depth_pairs", pair_row.get("rawFile", ""))
            post_path = resolve_frame_file(session, "depth_pairs", pair_row.get("processedFile", ""))
            raw_m = ndc_to_linear(sample_f32(cache.get(raw_path), pw, ph, u, v), projection)
            post_m = ndc_to_linear(sample_f32(cache.get(post_path), pw, ph, u, v), projection)
            if raw_m is not None:
                result["platformRawDepthM"] = f"{raw_m:.6f}"
            if post_m is not None:
                result["processedDepthM"] = f"{post_m:.6f}"
            if raw_m is not None and post_m is not None:
                result["processedMinusRawMm"] = f"{(post_m - raw_m) * 1000.0:.3f}"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    args = parser.parse_args()
    session = Path(args.session).resolve()
    artifact = session / "artifacts" / "tsdf_responsibility"
    schema = json.loads((artifact / "schema.json").read_text(encoding="utf-8-sig"))
    nx, ny, nz = schema["voxelCount"]
    voxel_size = float(schema["voxelSizeMetres"])
    truncation = float(schema["truncationDistanceMetres"])
    count = nx * ny * nz
    tsdf = (artifact / "final_tsdf.rg8_snorm.bin").read_bytes()
    responsibility = (artifact / "responsibility.rg32_uint.bin").read_bytes()
    support_responsibility = (artifact / "support_responsibility.r32_uint.bin").read_bytes()
    if (len(tsdf) != count * 2 or len(responsibility) != count * 8 or
            len(support_responsibility) != count * 4):
        raise SystemExit("responsibility artifact byte length does not match schema")

    paper = session / "artifacts" / "paper_audit" / "production_paper_surface_ledger.txt"
    bins = extract_stage_bins(paper)

    dispatches = rows(session / "runtime_timeline" / "integration_dispatches.csv")
    dispatch_by_integration = {int(r["integrationCount"]): r for r in dispatches}
    def dispatch_for(packed_id):
        # 4095 is deliberately the packed saturation marker.  It may mean the
        # exact 4095th integration or any later integration, so never pretend
        # that it is an exact frame join.
        return {} if packed_id == 4095 else dispatch_by_integration.get(packed_id, {})
    fusion_rows = rows(session / "fusion_inputs" / "manifest.csv")
    fusion_by_key = {(r.get("attemptIndex"), r.get("sourceFrame")): r for r in fusion_rows
                     if r.get("accepted") == "1"}
    pair_rows = rows(session / "depth_pairs" / "manifest.csv")
    pair_by_frame = {r.get("platformFrame"): r for r in pair_rows}

    transform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    coordinate = session / "coordinate_contract_stop.json"
    if coordinate.exists():
        try:
            transform = json.loads(coordinate.read_text(encoding="utf-8-sig"))["volumeIntegratorLocalToWorld"]
        except (KeyError, ValueError, json.JSONDecodeError):
            pass

    min_weight = 0.08
    config = session / "production_config.json"
    if config.exists():
        try:
            components = json.loads(config.read_text(encoding="utf-8-sig")).get("components", {})
            min_weight = float(components.get("Genesis.RoomScan.VolumeIntegrator", {}).get("minMeshWeight", min_weight))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    def flat(x, y, z):
        return x + nx * (y + ny * z)

    def center(axis, size):
        return (axis + 0.5 - size / 2.0) * voxel_size

    endpoint_bins = []
    for number, item in enumerate(bins):
        bounds_min = [float(item[f"local_min_{a}_m"]) for a in "xyz"]
        bounds_max = [float(item[f"local_max_{a}_m"]) for a in "xyz"]
        indices = set()
        lows = [max(0, int(math.floor(bounds_min[i] / voxel_size + [nx, ny, nz][i] / 2.0)) - 1)
                for i in range(3)]
        highs = [min([nx, ny, nz][i] - 1,
                     int(math.ceil(bounds_max[i] / voxel_size + [nx, ny, nz][i] / 2.0)) + 1)
                 for i in range(3)]
        for z in range(lows[2], highs[2] + 1):
            for y in range(lows[1], highs[1] + 1):
                for x in range(lows[0], highs[0] + 1):
                    i0 = flat(x, y, z)
                    d0, w0 = snorm8(tsdf[i0 * 2]), abs(snorm8(tsdf[i0 * 2 + 1]))
                    if w0 < min_weight:
                        continue
                    for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                        x1, y1, z1 = x + dx, y + dy, z + dz
                        if x1 >= nx or y1 >= ny or z1 >= nz:
                            continue
                        i1 = flat(x1, y1, z1)
                        d1, w1 = snorm8(tsdf[i1 * 2]), abs(snorm8(tsdf[i1 * 2 + 1]))
                        if w1 < min_weight or (d0 < 0) == (d1 < 0) or (d0 == 0 and d1 == 0):
                            continue
                        t = abs(d0) / max(abs(d0) + abs(d1), 1e-9)
                        point = [center(x, nx) + dx * voxel_size * t,
                                 center(y, ny) + dy * voxel_size * t,
                                 center(z, nz) + dz * voxel_size * t]
                        if all(bounds_min[k] - 1e-5 <= point[k] <= bounds_max[k] + 1e-5 for k in range(3)):
                            indices.add(i0)
                            indices.add(i1)
        endpoint_bins.append((number, item, sorted(indices)))

    event_fields = ["binIndex", "chunkX", "chunkY", "chunkZ", "builtEpoch",
                    "binX", "binY", "binZ", "crossingRmsVox", "voxelX", "voxelY", "voxelZ",
                    "localX", "localY", "localZ", "finalTsdf", "finalWeight",
                    "seedIntegration", "seedSourceFrame", "lastGeometryIntegration",
                    "lastGeometrySourceFrame", "lastOperation", "lastPreTsdf",
                    "lastToFinalMm", "birthSource", "birthGrazingRescue", "birthHighMotion",
                    "lastSupportIntegration", "lastSupportSourceFrame", "lastSupportOperation",
                    "lastSupportPreWeight", "lastSupportPostWeight",
                    "strongestBlockIntegration", "strongestBlockSourceFrame", "strongestBlockReason",
                    "strongestBlockDeltaMm"]
    sample_fields = ["binIndex", "voxelX", "voxelY", "voxelZ", "role", "integrationCount",
                     "sourceFrame", "attemptIndex", "dirtyEpoch", "angularDegPerSec", "linearMps",
                     "headPitchDeg", "headYawDeg", "headRollDeg", "motionQuality", "u", "v",
                     "fusionDepthM", "platformRawDepthM", "processedDepthM",
                     "processedMinusRawMm", "rawProjectiveSdfMm"]
    events, samples, bin_summaries = [], [], []
    cache = ByteCache()

    for bin_index, item, indices in endpoint_bins:
        op_counts, block_counts, seed_frames, last_frames, block_frames = Counter(), Counter(), Counter(), Counter(), Counter()
        missing = moved = blocked = 0
        for index in indices:
            z, rem = divmod(index, nx * ny)
            y, x = divmod(rem, nx)
            lane_x, lane_y = struct.unpack_from("<II", responsibility, index * 8)
            seed_id = lane_x & 0xFFF
            last_id = (lane_x >> 12) & 0xFFF
            op = (lane_x >> 24) & 0xF
            flags = (lane_x >> 28) & 0xF
            block_id = lane_y & 0xFFF
            block_reason = (lane_y >> 12) & 0x1F
            block_signed = (lane_y >> 17) & 0x7F
            block_delta = (block_signed & 0x3F) / 63.0 * (-1 if block_signed & 0x40 else 1)
            last_pre = packed_unorm_to_snorm((lane_y >> 24) & 0xFF)
            support_lane = struct.unpack_from("<I", support_responsibility, index * 4)[0]
            support_id = support_lane & 0xFFF
            support_op = (support_lane >> 12) & 0xF
            support_pre = packed_unorm_to_snorm((support_lane >> 16) & 0xFF)
            support_post = packed_unorm_to_snorm((support_lane >> 24) & 0xFF)
            final_sd = snorm8(tsdf[index * 2])
            final_weight = snorm8(tsdf[index * 2 + 1])
            seed_dispatch = dispatch_for(seed_id)
            last_dispatch = dispatch_for(last_id)
            block_dispatch = dispatch_for(block_id)
            support_dispatch = dispatch_for(support_id)
            seed_source = seed_dispatch.get("sourceFrame", "")
            last_source = last_dispatch.get("sourceFrame", "")
            block_source = block_dispatch.get("sourceFrame", "")
            support_source = support_dispatch.get("sourceFrame", "")
            if not seed_id or not last_id:
                missing += 1
            movement_mm = (final_sd - last_pre) * truncation * 1000.0 if last_id else 0.0
            moved += abs(movement_mm) >= 2.0
            blocked += block_id != 0
            op_counts[OPS.get(op, f"unknown_{op}")] += 1
            block_counts[BLOCKS.get(block_reason, f"unknown_{block_reason}")] += block_id != 0
            if seed_source != "": seed_frames[seed_source] += 1
            if last_source != "": last_frames[last_source] += 1
            if block_source != "": block_frames[block_source] += 1
            local = [center(x, nx), center(y, ny), center(z, nz)]
            events.append({
                "binIndex": bin_index, "chunkX": item["chunk_x"], "chunkY": item["chunk_y"],
                "chunkZ": item["chunk_z"], "builtEpoch": item["built_epoch"],
                "binX": item["bin_x"], "binY": item["bin_y"], "binZ": item["bin_z"],
                "crossingRmsVox": item["crossing_plane_rms_vox"], "voxelX": x, "voxelY": y,
                "voxelZ": z, "localX": f"{local[0]:.6f}", "localY": f"{local[1]:.6f}",
                "localZ": f"{local[2]:.6f}", "finalTsdf": f"{final_sd:.6f}",
                "finalWeight": f"{final_weight:.6f}", "seedIntegration": seed_id,
                "seedSourceFrame": seed_source, "lastGeometryIntegration": last_id,
                "lastGeometrySourceFrame": last_source, "lastOperation": OPS.get(op, str(op)),
                "lastPreTsdf": f"{last_pre:.6f}", "lastToFinalMm": f"{movement_mm:.3f}",
                "birthSource": flags & 3, "birthGrazingRescue": 1 if flags & 4 else 0,
                "birthHighMotion": 1 if flags & 8 else 0, "strongestBlockIntegration": block_id,
                "lastSupportIntegration": support_id, "lastSupportSourceFrame": support_source,
                "lastSupportOperation": SUPPORT_OPS.get(support_op, str(support_op)),
                "lastSupportPreWeight": f"{support_pre:.6f}",
                "lastSupportPostWeight": f"{support_post:.6f}",
                "strongestBlockSourceFrame": block_source,
                "strongestBlockReason": BLOCKS.get(block_reason, str(block_reason)),
                "strongestBlockDeltaMm": f"{block_delta * truncation * 1000.0:.3f}",
            })

            world = local_to_world(transform, local)
            seen_roles = set()
            for role, integration_id, dispatch in (("seed", seed_id, seed_dispatch),
                                                    ("last_geometry", last_id, last_dispatch),
                                                    ("last_support", support_id, support_dispatch),
                                                    ("strongest_block", block_id, block_dispatch)):
                if not integration_id or (role, integration_id) in seen_roles:
                    continue
                seen_roles.add((role, integration_id))
                source = dispatch.get("sourceFrame", "")
                fusion = fusion_by_key.get((dispatch.get("attemptIndex"), source))
                pair = pair_by_frame.get(source)
                projected = project_sample(session, fusion, pair, world, cache)
                sample = {"binIndex": bin_index, "voxelX": x, "voxelY": y, "voxelZ": z,
                          "role": role, "integrationCount": integration_id, "sourceFrame": source,
                          "attemptIndex": dispatch.get("attemptIndex", ""),
                          "dirtyEpoch": dispatch.get("dirtyEpoch", ""),
                          "angularDegPerSec": (fusion or {}).get("angularDegPerSec", ""),
                          "linearMps": (fusion or {}).get("linearMps", ""),
                          "headPitchDeg": (fusion or {}).get("fusionHeadPitchDeg", ""),
                          "headYawDeg": (fusion or {}).get("fusionHeadYawDeg", ""),
                          "headRollDeg": (fusion or {}).get("fusionHeadRollDeg", ""),
                          "motionQuality": (fusion or {}).get("motionQuality", "")}
                sample.update(projected)
                samples.append(sample)

        bin_summaries.append({
            "binIndex": bin_index, "chunk": [int(item["chunk_x"]), int(item["chunk_y"]), int(item["chunk_z"])],
            "bin": [int(item["bin_x"]), int(item["bin_y"]), int(item["bin_z"])],
            "builtEpoch": int(item["built_epoch"]), "crossingRmsVox": float(item["crossing_plane_rms_vox"]),
            "endpointCount": len(indices), "missingResponsibility": missing,
            "materialLastMoves": moved, "blockedEndpoints": blocked,
            "lastOperations": dict(op_counts), "blockReasons": dict(block_counts),
            "topSeedSourceFrames": seed_frames.most_common(8),
            "topLastGeometrySourceFrames": last_frames.most_common(8),
            "topBlockedSourceFrames": block_frames.most_common(8),
        })

    artifact.mkdir(parents=True, exist_ok=True)
    with (artifact / "endpoint_responsibility.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=event_fields)
        writer.writeheader(); writer.writerows(events)
    with (artifact / "endpoint_source_samples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields)
        writer.writeheader(); writer.writerows(samples)

    # Compare the final TSDF directly with the exact committed native-5cm paper.
    # Unlike the old SupportTruth ledger this is valid regardless of display route.
    hole_rows = derive_paper_holes(
        session, tsdf, nx, ny, nz, voxel_size, min_weight)
    hole_fields = ["cellX", "cellY", "cellZ", "localX", "localY", "localZ", "scope",
                   "crossingClass", "responsibleBoundary", "finalPublished",
                   "rawCrossingEdges", "matureCrossingEdges", "paperTargetBins",
                   "nearbyVoxels", "nearbyReadable", "nearbyCrossingEdges", "nearbySeeded",
                   "nearbyReset", "nearbyBlocked", "topLastOperation", "topSupportOperation",
                   "topBlockReason", "topSeedIntegration", "topSeedSourceFrame",
                   "topLastIntegration", "topLastSourceFrame", "topSupportIntegration",
                   "topSupportSourceFrame", "topBlockedIntegration", "topBlockedSourceFrame"]
    hole_output = []
    hole_samples = []
    hole_sample_fields = ["cellX", "cellY", "cellZ", "role", "sourceFrame",
                          "integrationCount", "attemptIndex", "dirtyEpoch", "angularDegPerSec",
                          "linearMps", "headPitchDeg", "headYawDeg", "headRollDeg", "motionQuality",
                          "u", "v", "fusionDepthM", "platformRawDepthM", "processedDepthM",
                          "processedMinusRawMm", "rawProjectiveSdfMm"]
    extraction_boundary = {
        "mature_crossing_without_paper": "after_tsdf_before_committed_paper",
        "mature_crossing_near_paper_edge": "paper_boundary_context",
        "raw_crossing_below_weight_without_paper": "tsdf_support_below_formal_threshold",
        "weak_crossing_near_paper_edge": "tsdf_support_boundary_context",
        "no_raw_crossing": "tsdf_has_no_zero_crossing",
        "raw_crossing_insufficient": "tsdf_crossing_insufficient_for_topology",
        "raw_not_mature": "tsdf_support_below_formal_threshold",
        "face_parity_open": "paper_topology_face_pairing",
        "closed_graph_failed": "paper_topology_closed_graph",
        "fallback_rejected": "paper_topology_fallback",
        "candidate_emitted": "paper_candidate_transaction",
        "published_replay": "paper_published",
        "topology_vertex_overflow": "paper_topology_capacity",
        "other_no_emit": "paper_topology_other",
        "published_snapshot_missing_materialization": "paper_snapshot_materialization",
    }
    for hole in hole_rows:
        if hole.get("final_published") == "1" and hole.get("empty_surface_halo") != "1":
            continue
        local = [float(hole[f"world_{a}_m"]) for a in "xyz"]
        centre = [int(math.floor(local[i] / voxel_size + [nx, ny, nz][i] / 2.0)) for i in range(3)]
        neighbourhood = []
        for z in range(max(0, centre[2] - 2), min(nz, centre[2] + 3)):
            for y in range(max(0, centre[1] - 2), min(ny, centre[1] + 3)):
                for x in range(max(0, centre[0] - 2), min(nx, centre[0] + 3)):
                    neighbourhood.append((x, y, z, flat(x, y, z)))
        readable = crossing_edges = seeded = reset = blocked_count = 0
        last_ops, support_ops, block_reasons = Counter(), Counter(), Counter()
        seed_ids, last_ids, support_ids, blocked_ids = Counter(), Counter(), Counter(), Counter()
        readable_set = {}
        for x, y, z, index in neighbourhood:
            sd = snorm8(tsdf[index * 2]); weight = abs(snorm8(tsdf[index * 2 + 1]))
            readable_set[(x, y, z)] = (sd, weight)
            readable += weight >= min_weight
            lane_x, lane_y = struct.unpack_from("<II", responsibility, index * 8)
            seed_id, last_id, op = lane_x & 0xFFF, (lane_x >> 12) & 0xFFF, (lane_x >> 24) & 0xF
            block_id, block_reason = lane_y & 0xFFF, (lane_y >> 12) & 0x1F
            support_lane = struct.unpack_from("<I", support_responsibility, index * 4)[0]
            support_id, support_op = support_lane & 0xFFF, (support_lane >> 12) & 0xF
            seeded += seed_id != 0
            reset += op == 3 or support_op == 6
            blocked_count += block_id != 0
            last_ops[OPS.get(op, str(op))] += op != 0
            support_ops[SUPPORT_OPS.get(support_op, str(support_op))] += support_op != 0
            block_reasons[BLOCKS.get(block_reason, str(block_reason))] += block_id != 0
            for counter, ident in ((seed_ids, seed_id), (last_ids, last_id),
                                   (support_ids, support_id), (blocked_ids, block_id)):
                if ident != 0:
                    counter[ident] += 1
        for (x, y, z), (sd, weight) in readable_set.items():
            if weight < min_weight:
                continue
            for other in ((x + 1, y, z), (x, y + 1, z), (x, y, z + 1)):
                if other not in readable_set:
                    continue
                other_sd, other_weight = readable_set[other]
                crossing_edges += other_weight >= min_weight and (sd < 0) != (other_sd < 0)
        cause = hole.get("cause", "unknown")
        boundary = extraction_boundary.get(cause, "unknown")
        if cause == "no_raw_crossing":
            if seeded == 0 and blocked_count > 0: boundary = "pre_tsdf_or_seed_admission_blocked"
            elif reset > 0: boundary = "tsdf_lifetime_reset_or_prune"
            elif readable == 0: boundary = "tsdf_unseeded_or_support_removed"
            elif crossing_edges == 0: boundary = "tsdf_no_sign_change"
        top_seed_id = seed_ids.most_common(1)[0][0] if seed_ids else 0
        top_last_id = last_ids.most_common(1)[0][0] if last_ids else 0
        top_support_id = support_ids.most_common(1)[0][0] if support_ids else 0
        top_blocked_id = blocked_ids.most_common(1)[0][0] if blocked_ids else 0
        top_seed_source = dispatch_for(top_seed_id).get("sourceFrame", "")
        top_last_source = dispatch_for(top_last_id).get("sourceFrame", "")
        top_support_source = dispatch_for(top_support_id).get("sourceFrame", "")
        top_blocked_source = dispatch_for(top_blocked_id).get("sourceFrame", "")
        hole_record = {
            "cellX": hole.get("cell_x", ""), "cellY": hole.get("cell_y", ""), "cellZ": hole.get("cell_z", ""),
            "localX": local[0], "localY": local[1], "localZ": local[2],
            "scope": "empty_surface_halo_context" if hole.get("empty_surface_halo") == "1"
                     else "direct_missing_paper",
            "crossingClass": cause, "responsibleBoundary": boundary,
            "finalPublished": hole.get("final_published", ""), "nearbyVoxels": len(neighbourhood),
            "rawCrossingEdges": hole.get("raw_edges", ""),
            "matureCrossingEdges": hole.get("mature_edges", ""),
            "paperTargetBins": hole.get("paper_target_bins", ""),
            "nearbyReadable": readable, "nearbyCrossingEdges": crossing_edges, "nearbySeeded": seeded,
            "nearbyReset": reset, "nearbyBlocked": blocked_count,
            "topLastOperation": last_ops.most_common(1)[0][0] if last_ops else "none",
            "topSupportOperation": support_ops.most_common(1)[0][0] if support_ops else "none",
            "topBlockReason": block_reasons.most_common(1)[0][0] if block_reasons else "none",
            "topSeedIntegration": top_seed_id, "topSeedSourceFrame": top_seed_source,
            "topLastIntegration": top_last_id, "topLastSourceFrame": top_last_source,
            "topSupportIntegration": top_support_id, "topSupportSourceFrame": top_support_source,
            "topBlockedIntegration": top_blocked_id, "topBlockedSourceFrame": top_blocked_source,
        }
        hole_output.append(hole_record)
        world = local_to_world(transform, local)
        for role, integration_id in (("seed", top_seed_id),
                                     ("last_geometry", top_last_id),
                                     ("last_support", top_support_id),
                                     ("strongest_block", top_blocked_id)):
            if not integration_id:
                continue
            dispatch = dispatch_for(integration_id)
            source = dispatch.get("sourceFrame", "")
            fusion = fusion_by_key.get((dispatch.get("attemptIndex"), str(source)))
            pair = pair_by_frame.get(str(source))
            projected = project_sample(session, fusion, pair, world, cache)
            sample = {"cellX": hole_record["cellX"], "cellY": hole_record["cellY"],
                      "cellZ": hole_record["cellZ"], "role": role, "sourceFrame": source,
                      "integrationCount": dispatch.get("integrationCount", ""),
                      "attemptIndex": dispatch.get("attemptIndex", ""),
                      "dirtyEpoch": dispatch.get("dirtyEpoch", ""),
                      "angularDegPerSec": (fusion or {}).get("angularDegPerSec", ""),
                      "linearMps": (fusion or {}).get("linearMps", ""),
                      "headPitchDeg": (fusion or {}).get("fusionHeadPitchDeg", ""),
                      "headYawDeg": (fusion or {}).get("fusionHeadYawDeg", ""),
                      "headRollDeg": (fusion or {}).get("fusionHeadRollDeg", ""),
                      "motionQuality": (fusion or {}).get("motionQuality", "")}
            sample.update(projected)
            hole_samples.append(sample)
    with (artifact / "paper_hole_responsibility.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=hole_fields)
        writer.writeheader(); writer.writerows(hole_output)
    with (artifact / "paper_hole_source_samples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=hole_sample_fields)
        writer.writeheader(); writer.writerows(hole_samples)

    sample_preprocess = [abs(float(r["processedMinusRawMm"])) for r in samples
                         if r.get("processedMinusRawMm") not in (None, "")]
    summary = {
        "schema": "scancover.tsdf_responsibility_analysis.v2",
        "targetRule": "paper bins whose earliest material deformation stage is tsdf_zero_crossing",
        "targetBins": len(bins), "zeroCrossingEndpoints": len(events),
        "saturatedResponsibilityEndpoints": sum(
            4095 in (r["seedIntegration"], r["lastGeometryIntegration"],
                     r["lastSupportIntegration"], r["strongestBlockIntegration"])
            for r in events),
        "endpointsWithoutCurrentLifetimeProvenance": sum(b["missingResponsibility"] for b in bin_summaries),
        "endpointsWithBlockedCorrection": sum(b["blockedEndpoints"] for b in bin_summaries),
        "endpointLastOperationCounts": dict(Counter(r["lastOperation"] for r in events)),
        "endpointStrongestBlockReasonCounts": dict(Counter(r["strongestBlockReason"] for r in events
                                                          if r["strongestBlockIntegration"])),
        "endpointLastSupportOperationCounts": dict(Counter(r["lastSupportOperation"] for r in events)),
        "exactProjectedSamples": len(samples),
        "projectedRawPostComparable": len(sample_preprocess),
        "projectedRawPostDeltaAtLeast10Mm": sum(v >= 10.0 for v in sample_preprocess),
        "holeCells": sum(r["scope"] == "direct_missing_paper" for r in hole_output),
        "holeHaloContextCells": sum(r["scope"] == "empty_surface_halo_context" for r in hole_output),
        "holeExactProjectedSamples": len(hole_samples),
        "holeResponsibleBoundaryCounts": dict(Counter(
            r["responsibleBoundary"] for r in hole_output if r["scope"] == "direct_missing_paper")),
        "blankDetectionBoundary": "Direct holes require a final raw or mature TSDF sign change with no committed production-paper occupancy in the same or adjacent native 5cm cell. Regions with no final sign change require an independent surface reference and are not silently called holes.",
        "bins": bin_summaries,
        "interpretationBoundary": "A recorded observation is evidence of what each stage supplied or blocked, not external ground truth. Convict the earliest stage with a material divergence and preserve later amplification/blockage as separate responsibility.",
    }
    (artifact / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "targetBins", "zeroCrossingEndpoints", "saturatedResponsibilityEndpoints",
        "endpointsWithoutCurrentLifetimeProvenance",
        "endpointsWithBlockedCorrection", "exactProjectedSamples", "projectedRawPostComparable",
        "projectedRawPostDeltaAtLeast10Mm", "holeCells", "holeHaloContextCells",
        "holeExactProjectedSamples")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
