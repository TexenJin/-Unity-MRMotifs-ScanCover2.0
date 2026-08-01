#!/usr/bin/env python3
"""ScanCoverDepthEvidence/v3 完整性验证器。只验证证据，不修补数据。"""

from __future__ import annotations

import argparse
import binascii
import csv
import hashlib
import json
import struct
from pathlib import Path


MAGIC = b"SCDEVID1"
EXPECTED_BUFFERS = {
    "raw_depth_r32f": 4,
    "depth_metrics_rgba32f": 16,
    "world_position_raw_rgba32f": 16,
    "world_position_neighbour_rgba32f": 16,
    "world_normal_raw_rgba32f": 16,
    "world_normal_neighbour_rgba32f": 16,
    "diagnostics_rgba32f": 16,
}


def read_exact(stream, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"文件提前结束：需要 {size} 字节，只得到 {len(data)} 字节")
    return data


def read_i32(stream) -> int:
    return struct.unpack("<i", read_exact(stream, 4))[0]


def read_i64(stream) -> int:
    return struct.unpack("<q", read_exact(stream, 8))[0]


def read_u32(stream) -> int:
    return struct.unpack("<I", read_exact(stream, 4))[0]


def read_text(stream) -> str:
    size = read_i32(stream)
    if size < 0 or size > 16 * 1024 * 1024:
        raise ValueError(f"非法字符串长度：{size}")
    return read_exact(stream, size).decode("utf-8")


def validate_frame(path: Path) -> dict:
    result = {"file": path.as_posix(), "ok": False, "errors": [], "buffers": []}
    try:
        with path.open("rb") as stream:
            if read_exact(stream, 8) != MAGIC:
                raise ValueError("magic 不匹配")
            version = read_i32(stream)
            if version != 1:
                raise ValueError(f"容器版本不支持：{version}")
            metadata_size = read_i32(stream)
            buffer_count = read_i32(stream)
            if metadata_size < 2 or metadata_size > 64 * 1024 * 1024:
                raise ValueError(f"metadata 长度异常：{metadata_size}")
            if buffer_count < 0 or buffer_count > 64:
                raise ValueError(f"buffer 数量异常：{buffer_count}")
            metadata = json.loads(read_exact(stream, metadata_size).decode("utf-8"))
            if metadata.get("schema") != "ScanCoverDepthEvidence/v3":
                result["errors"].append("帧 schema 不匹配")
            if metadata.get("expectedEyeCount") != 2:
                result["errors"].append("帧未声明完整双眼")
            if metadata.get("readbackLayout") != "gpu_structured_buffer_left_then_right":
                result["errors"].append("帧双眼展开布局不匹配")

            names = set()
            for _ in range(buffer_count):
                name = read_text(stream)
                semantic = read_text(stream)
                graphics_format = read_text(stream)
                width = read_i32(stream)
                height = read_i32(stream)
                depth = read_i32(stream)
                byte_count = read_i64(stream)
                expected_crc = read_u32(stream)
                if byte_count < 0 or byte_count > 4 * 1024 * 1024 * 1024:
                    raise ValueError(f"{name} 字节数异常：{byte_count}")
                payload = read_exact(stream, byte_count)
                actual_crc = binascii.crc32(payload) & 0xFFFFFFFF
                if actual_crc != expected_crc:
                    result["errors"].append(f"{name} CRC32 不匹配")
                bpp = EXPECTED_BUFFERS.get(name)
                if bpp is not None and depth != 2:
                    result["errors"].append(f"{name} 不是完整双眼：depth={depth}")
                if bpp is not None and byte_count != width * height * depth * bpp:
                    result["errors"].append(
                        f"{name} 尺寸不匹配：{byte_count} != {width}x{height}x{depth}x{bpp}"
                    )
                if name in names:
                    result["errors"].append(f"缓冲重名：{name}")
                names.add(name)
                result["buffers"].append(
                    {
                        "name": name,
                        "semantic": semantic,
                        "graphicsFormat": graphics_format,
                        "width": width,
                        "height": height,
                        "depth": depth,
                        "bytes": byte_count,
                    }
                )

            trailing = stream.read(1)
            if trailing:
                result["errors"].append("帧尾存在未声明字节")
            missing = sorted(set(EXPECTED_BUFFERS) - names)
            if missing:
                result["errors"].append("缺少缓冲：" + ", ".join(missing))
            result["metadata"] = {
                "frameIndex": metadata.get("frameIndex"),
                "unityFrame": metadata.get("unityFrame"),
                "requestUtc": metadata.get("requestUtc"),
                "readbackFailures": metadata.get("readbackFailures"),
            }
            result["ok"] = not result["errors"]
    except Exception as exc:  # 保留每一帧的失败原因，继续验证余下文件。
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def validate_session(session: Path) -> dict:
    report = {
        "schema": "ScanCoverDepthEvidenceValidation/v1",
        "session": str(session.resolve()),
        "ok": False,
        "errors": [],
        "frameCount": 0,
        "validFrameCount": 0,
        "failedFrameCount": 0,
        "runtimeStatus": None,
        "systemRoomMeshStatus": None,
        "warnings": [],
        "frames": [],
    }
    session_json = session / "session.json"
    manifest_path = session / "manifest.csv"
    runtime_summary_path = session / "runtime_summary.json"
    room_mesh_expected = False
    if not session_json.exists():
        report["errors"].append("缺少 session.json")
    else:
        try:
            descriptor = json.loads(session_json.read_text(encoding="utf-8"))
            if descriptor.get("schema") != "ScanCoverDepthEvidence/v3":
                report["errors"].append("session schema 不匹配")
            if descriptor.get("eyeCount") != 2:
                report["errors"].append("session 未声明完整双眼")
            if descriptor.get("readbackLayout") != "gpu_structured_buffer_left_then_right":
                report["errors"].append("session 双眼展开布局不匹配")
            room_mesh_expected = descriptor.get("systemRoomMeshCompanionEnabled") is True
            if room_mesh_expected and descriptor.get("systemRoomMeshRenderedInCaptureScene") is not False:
                report["errors"].append("系统房间网格伴随源未声明为不可见")
        except Exception as exc:
            report["errors"].append(f"session.json 无法读取：{exc}")

    room_mesh_status_path = session / "system_room_mesh_status.json"
    if room_mesh_status_path.exists():
        try:
            room_mesh_status = json.loads(room_mesh_status_path.read_text(encoding="utf-8"))
            status = room_mesh_status.get("status")
            report["systemRoomMeshStatus"] = status
            if room_mesh_status.get("renderedInCaptureScene") is not False:
                report["errors"].append("系统房间网格状态未确认不可见")
            if status == "exported":
                relative_directory = room_mesh_status.get("relativeDirectory") or "system_room_mesh"
                room_mesh_directory = session / relative_directory
                combined_obj = room_mesh_directory / "meta_scene_mesh_aligned_all.obj"
                info_json = room_mesh_directory / "session_info.json"
                if not combined_obj.exists() or combined_obj.stat().st_size <= 0:
                    report["errors"].append("系统房间网格已标记导出，但缺少有效的世界坐标 OBJ")
                if not info_json.exists():
                    report["errors"].append("系统房间网格已标记导出，但缺少 session_info.json")
                checksum_path = room_mesh_directory / "checksums.sha256"
                if checksum_path.exists():
                    for line in checksum_path.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        expected_sha, relative_path = line.split(None, 1)
                        payload_path = room_mesh_directory / relative_path.strip()
                        if not payload_path.exists():
                            report["errors"].append(f"系统房间网格校验文件缺失：{relative_path.strip()}")
                            continue
                        actual_sha = hashlib.sha256(payload_path.read_bytes()).hexdigest()
                        if actual_sha.lower() != expected_sha.lower():
                            report["errors"].append(f"系统房间网格 SHA-256 不匹配：{relative_path.strip()}")
            elif status in {"unavailable", "failed"}:
                issue = room_mesh_status.get("issue") or "未提供原因"
                report["warnings"].append(f"系统房间网格未随本轮导出：{status}，{issue}")
            else:
                report["warnings"].append(f"系统房间网格状态未完成：{status}")
        except Exception as exc:
            report["warnings"].append(f"system_room_mesh_status.json 无法读取：{exc}")
    elif room_mesh_expected:
        report["warnings"].append("本轮声明启用系统房间网格伴随导出，但缺少状态文件")

    if not runtime_summary_path.exists():
        report["errors"].append("缺少 runtime_summary.json；本轮可能尚未安全收尾")
    else:
        try:
            runtime_summary = json.loads(runtime_summary_path.read_text(encoding="utf-8"))
            report["runtimeStatus"] = runtime_summary.get("sessionStatus")
            if runtime_summary.get("sessionStatus") != "complete" or runtime_summary.get("sessionFailed") is not False:
                code = runtime_summary.get("fatalFailureCode") or "UNKNOWN_FATAL_ERROR"
                detail = runtime_summary.get("fatalFailureDetail") or ""
                report["errors"].append(f"实机已判定本轮失败：{code} {detail}".rstrip())
            if int(runtime_summary.get("gpuReadbackFailedFrames", 0) or 0) != 0:
                report["errors"].append("实机报告存在 GPU 回读/双眼完整性失败帧")
            if int(runtime_summary.get("writerFailedFrames", 0) or 0) != 0:
                report["errors"].append("实机报告存在写盘失败帧")
            requested = int(runtime_summary.get("requestedFrames", -1))
            written = int(runtime_summary.get("writtenFrames", -1))
            if requested < 0 or written < 0 or requested != written:
                report["errors"].append(f"实机请求帧与完整写入帧不一致：requested={requested},written={written}")
        except Exception as exc:
            report["errors"].append(f"runtime_summary.json 无法读取：{exc}")

    manifest_rows = []
    if not manifest_path.exists():
        report["errors"].append("缺少 manifest.csv")
    else:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            manifest_rows = list(csv.DictReader(stream))

    expected_index = 0
    for row in manifest_rows:
        try:
            frame_index = int(row["frame"])
        except Exception:
            report["errors"].append("manifest 存在非法 frame 编号")
            continue
        if frame_index != expected_index:
            report["errors"].append(f"帧序列不连续：期望 {expected_index}，实际 {frame_index}")
            expected_index = frame_index
        expected_index += 1
        relative = row.get("file", "")
        frame_path = session / relative
        frame_result = validate_frame(frame_path)
        if frame_path.exists() and row.get("sha256"):
            actual_sha = hashlib.sha256(frame_path.read_bytes()).hexdigest()
            if actual_sha.lower() != row["sha256"].lower():
                frame_result["errors"].append("整帧 SHA-256 不匹配")
                frame_result["ok"] = False
        report["frames"].append(frame_result)

    report["frameCount"] = len(report["frames"])
    report["validFrameCount"] = sum(1 for frame in report["frames"] if frame["ok"])
    report["failedFrameCount"] = report["frameCount"] - report["validFrameCount"]
    report["ok"] = not report["errors"] and report["frameCount"] > 0 and report["failedFrameCount"] == 0
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="验证 Quest 深度证据采集会话")
    parser.add_argument("session", type=Path, help="Evidence_... 会话目录")
    parser.add_argument("--write-report", action="store_true", help="写出 validation_report.json")
    args = parser.parse_args()

    report = validate_session(args.session)
    if args.write_report:
        output = args.session / "validation_report.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"验证{'通过' if report['ok'] else '失败'}："
        f"总帧 {report['frameCount']}，完整 {report['validFrameCount']}，失败 {report['failedFrameCount']}"
    )
    for error in report["errors"]:
        print("- " + error)
    for warning in report["warnings"]:
        print("- 提示：" + warning)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
