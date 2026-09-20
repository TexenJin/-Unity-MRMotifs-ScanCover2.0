#!/usr/bin/env python3
"""Small deterministic contract test for BuildScanCoverResponsibilityLedger."""

from __future__ import annotations

import csv
import json
import struct
import tempfile
from pathlib import Path

from BuildScanCoverResponsibilityLedger import build


def write_csv(path: Path, fields: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        session = Path(temporary) / "session_test"
        paper = session / "artifacts" / "paper_audit"
        paper.mkdir(parents=True)
        (paper / "production_paper_surface_ledger.txt").write_text(
            "stage_responsibility_spatial_csv:\n"
            "chunk_x,chunk_y,chunk_z,built_epoch,bin_x,bin_y,bin_z,local_min_x_m,local_min_y_m,local_min_z_m,local_max_x_m,local_max_y_m,local_max_z_m,triangles,crossing_samples,crossing_plane_candidate,crossing_plane_rms_vox,crossing_plane_thin_ratio,raw_vertices,raw_plane_candidate,raw_plane_rms_vox,raw_plane_thin_ratio,final_vertices,final_plane_candidate,final_plane_rms_vox,final_plane_thin_ratio,first_deformation_stage\n"
            "0,0,0,7,0,0,0,-0.4,-0.4,-0.4,0.4,0.4,0.4,10,8,1,0.5,0.1,8,1,0.4,0.1,8,1,0.3,0.1,tsdf_zero_crossing\n\n",
            encoding="utf-8")
        quality_fields = [
            "session_id", "replacement_sequence", "chunk_x", "chunk_y", "chunk_z",
            "candidate_epoch", "initial", "bin_x", "bin_y", "bin_z", "triangles",
            "crossing_samples", "crossing_plane_candidate", "crossing_plane_rms_vox",
            "raw_vertices", "raw_plane_candidate", "raw_plane_rms_vox", "final_vertices",
            "final_plane_candidate", "final_plane_rms_vox", "first_deformation_stage",
        ]
        write_csv(paper / "production_paper_quality_timeline.csv", quality_fields, [
            ["s", 1, 0, 0, 0, 3, 1, 0, 0, 0, 10, 8, 1, .7, 8, 1, .65, 8, 1, .6, "tsdf_zero_crossing"],
            ["s", 2, 0, 0, 0, 7, 0, 0, 0, 0, 10, 8, 1, .5, 8, 1, .4, 8, 1, .3, "tsdf_zero_crossing"],
        ])

        tsdf = session / "artifacts" / "tsdf_responsibility"
        tsdf.mkdir(parents=True)
        (tsdf / "schema.json").write_text(json.dumps({"voxelSizeMetres": .05}), encoding="utf-8")
        write_csv(tsdf / "endpoint_responsibility.csv",
                  ["binIndex", "strongestBlockIntegration"], [[0, 4]])
        sample_fields = ["binIndex", "sourceFrame", "attemptIndex", "integrationCount",
                         "role", "dirtyEpoch", "u", "v", "fusionDepthM",
                         "platformRawDepthM", "processedDepthM", "processedMinusRawMm",
                         "rawProjectiveSdfMm", "angularDegPerSec", "linearMps",
                         "headPitchDeg", "headYawDeg", "headRollDeg", "motionQuality"]
        write_csv(tsdf / "endpoint_source_samples.csv", sample_fields,
                  [[0, 11, 2, 4, "last_geometry", 7, .5, .5, 1.0, 1.01, 1.0,
                    -10, 12, 3, .01, 5, 10, 1, .95]])

        frames = session / "fusion_inputs" / "frames"
        frames.mkdir(parents=True)
        write_csv(session / "fusion_inputs" / "manifest.csv",
                  ["attemptIndex", "sourceFrame", "metaFile"],
                  [[2, 11, "fusion_000001_meta.json"]])
        identity_name = "fusion_000001_identity.bin"
        correspondence_name = "fusion_000001_correspondence.bin"
        (frames / identity_name).write_bytes(struct.pack("<4I", 3, 17, 5, 0))
        (frames / correspondence_name).write_bytes(
            struct.pack("<12f", 0, 0, 0, float(0x1C7), 0, 0, 0, 0, 0, 0, 0, 0))
        (frames / "fusion_000001_meta.json").write_text(json.dumps({
            "textures": {"depth": {"width": 1, "height": 1}},
            "gunGelAdmission": {
                "pixelStride": 1, "gridX": 1, "gridY": 1,
                "identityStrideBytes": 16, "finalIdentityFile": identity_name,
                "correspondenceStrideBytes": 48,
                "correspondenceFile": correspondence_name,
            },
        }), encoding="utf-8")

        court = session / "probe_shadow" / "final_buffer" / "final_surface_court"
        testimony_fields = ["gunGelFrame", "sourceFrame", "attemptIndex", "stableId",
                            "role", "sourceX", "sourceY", "sourceZ", "rangeM",
                            "incidenceAbs", "targetNormalResidualMm", "planeCoordinateM",
                            "angularDegPerSec", "linearMps", "motionQuality",
                            "headPitchDeg", "headYawDeg", "headRollDeg"]
        write_csv(court / "independent_testimonies.csv", testimony_fields,
                  [[9, 11, 2, 17, "audit", 0, 0, 0, 1.0, .8, 3, 1.0, 3, .01, .95, 5, 10, 1]])
        write_csv(court / "frame_gate_summary.csv",
                  ["gunGelFrame", "sourceFrame", "attemptIndex", "totalRows"],
                  [[9, 11, 2, 1]])
        write_csv(court / "decision_checks.csv",
                  ["gunGelFrame", "stableId", "outcome"], [[9, 17, "winner_published"]])
        write_csv(court / "runtime_events.csv",
                  ["gunGelFrame", "stableId", "generation", "reason"],
                  [[9, 17, 1, "winner_published"]])

        summary = build(session, session / "artifacts" / "responsibility_ledger")
        assert summary["missingLanes"] == [], summary
        lineage = list(csv.DictReader((session / "artifacts" / "responsibility_ledger" /
                                       "roughness_lineage.csv").open(encoding="utf-8-sig")))
        timeline = list(csv.DictReader((session / "artifacts" / "responsibility_ledger" /
                                        "surface_site_timeline.csv").open(encoding="utf-8-sig")))
        assert lineage[0]["correctionOutcome"] == "flattened", lineage[0]
        assert lineage[0]["blockedCorrectionEndpoints"] == "1", lineage[0]
        assert timeline[0]["gunGelStableId"] == "17", timeline[0]
        assert timeline[0]["courtSpatialJoin"] == "same_source_stable_id_and_bounded", timeline[0]
    print("PASS: full-chain responsibility ledger joins first publish, court, TSDF and later correction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
