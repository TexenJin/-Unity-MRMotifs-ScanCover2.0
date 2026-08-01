#!/usr/bin/env python3
"""证据容器最小回归测试；只在系统临时目录生成数据。"""

from __future__ import annotations

import binascii
import csv
import hashlib
import json
import struct
import tempfile
from pathlib import Path

from validate_evidence_session import EXPECTED_BUFFERS, validate_frame, validate_session


def write_text(stream, value: str) -> None:
    encoded = value.encode("utf-8")
    stream.write(struct.pack("<i", len(encoded)))
    stream.write(encoded)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="sc_evidence_test_") as temporary:
        root = Path(temporary)
        frames = root / "frames"
        frames.mkdir()
        (root / "session.json").write_text(
            json.dumps(
                {
                    "schema": "ScanCoverDepthEvidence/v3",
                    "eyeCount": 2,
                    "readbackLayout": "gpu_structured_buffer_left_then_right",
                }
            ),
            encoding="utf-8",
        )
        runtime_summary_path = root / "runtime_summary.json"
        runtime_summary_path.write_text(
            json.dumps(
                {
                    "schema": "ScanCoverDepthEvidenceRuntimeSummary/v1",
                    "sessionStatus": "complete",
                    "sessionFailed": False,
                    "fatalFailureCode": "",
                    "fatalFailureDetail": "",
                    "fatalFailureFrame": -1,
                    "requestedFrames": 1,
                    "writtenFrames": 1,
                    "writerFailedFrames": 0,
                    "gpuReadbackFailedFrames": 0,
                }
            ),
            encoding="utf-8",
        )

        frame_path = frames / "frame_0000000.scede"
        metadata = json.dumps(
            {
                "schema": "ScanCoverDepthEvidence/v3",
                "expectedEyeCount": 2,
                "readbackLayout": "gpu_structured_buffer_left_then_right",
                "frameIndex": 0,
                "unityFrame": 42,
                "requestUtc": "2026-01-01T00:00:00Z",
                "readbackFailures": 0,
            }
        ).encode("utf-8")
        with frame_path.open("wb") as stream:
            stream.write(b"SCDEVID1")
            stream.write(struct.pack("<iii", 1, len(metadata), len(EXPECTED_BUFFERS)))
            stream.write(metadata)
            for name, bytes_per_pixel in EXPECTED_BUFFERS.items():
                payload = bytes(bytes_per_pixel * 2 * 1 * 2)
                write_text(stream, name)
                write_text(stream, "测试")
                write_text(stream, "R32_SFloat" if bytes_per_pixel == 4 else "R32G32B32A32_SFloat")
                stream.write(struct.pack("<iiiqI", 2, 1, 2, len(payload), binascii.crc32(payload) & 0xFFFFFFFF))
                stream.write(payload)

        sha = hashlib.sha256(frame_path.read_bytes()).hexdigest()
        with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["frame", "file", "sha256"])
            writer.writeheader()
            writer.writerow({"frame": 0, "file": "frames/frame_0000000.scede", "sha256": sha})

        report = validate_session(root)
        if not report["ok"]:
            raise AssertionError(json.dumps(report, ensure_ascii=False, indent=2))
        print("证据容器回归测试通过")

        failed_runtime = json.loads(runtime_summary_path.read_text(encoding="utf-8"))
        failed_runtime.update(
            {
                "sessionStatus": "failed",
                "sessionFailed": True,
                "fatalFailureCode": "DEPTH_SOURCE_EYE_COUNT",
                "fatalFailureDetail": "actual=1,expected=2",
                "fatalFailureFrame": 0,
                "requestedFrames": 1,
                "writtenFrames": 0,
            }
        )
        runtime_summary_path.write_text(json.dumps(failed_runtime), encoding="utf-8")
        failed_session_report = validate_session(root)
        if failed_session_report["ok"] or not any(
            "DEPTH_SOURCE_EYE_COUNT" in error for error in failed_session_report["errors"]
        ):
            raise AssertionError(json.dumps(failed_session_report, ensure_ascii=False, indent=2))

        runtime_summary_path.write_text(
            json.dumps(
                {
                    "schema": "ScanCoverDepthEvidenceRuntimeSummary/v1",
                    "sessionStatus": "complete",
                    "sessionFailed": False,
                    "fatalFailureCode": "",
                    "fatalFailureDetail": "",
                    "fatalFailureFrame": -1,
                    "requestedFrames": 1,
                    "writtenFrames": 1,
                    "writerFailedFrames": 0,
                    "gpuReadbackFailedFrames": 0,
                }
            ),
            encoding="utf-8",
        )
        # A frame that merely labels a single slice as valid must never pass as dual-eye data.
        with frame_path.open("r+b") as stream:
            stream.seek(8 + 12 + len(metadata))
            for _ in range(3):
                text_size = struct.unpack("<i", stream.read(4))[0]
                stream.seek(text_size, 1)
            stream.seek(8, 1)  # width + height
            stream.write(struct.pack("<i", 1))
        half_eye_report = validate_frame(frame_path)
        if half_eye_report["ok"] or not any("depth=1" in error for error in half_eye_report["errors"]):
            raise AssertionError(json.dumps(half_eye_report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
