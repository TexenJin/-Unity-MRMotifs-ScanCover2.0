#!/usr/bin/env python3
"""Validate and inventory a ScanCover deterministic replay-session package.

This tool intentionally fails closed: a directory without capture_complete.json,
or with one mismatched SHA-256 entry, is not advertised as independently replayable.
It does not approximate VolumeIntegration.compute; it proves whether all recorded
inputs and provenance needed by an exact engine replay are present and intact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksums(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid checksums.sha256 line {line_number}") from exc
        entries.append((digest.strip().lower(), relative.strip()))
    return entries


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate(root: Path, allow_incomplete: bool) -> tuple[list[Check], dict]:
    root = root.resolve()
    checks: list[Check] = []
    complete_path = root / "capture_complete.json"
    complete = json.loads(complete_path.read_text("utf-8")) if complete_path.exists() else None
    checks.append(Check(
        "complete_marker",
        complete is not None or allow_incomplete,
        "present" if complete is not None else "missing (capture still open or did not drain)",
    ))
    loss_fields = (
        "fusionCaptureDrops",
        "fusionReadbackErrors",
        "fusionWriteErrors",
        "depthPairDrops",
        "depthPairReadbackErrors",
        "depthPairWriteErrors",
    )
    marker_ok = bool(
        complete
        and complete.get("schema") == "scancover.replay_session.v1"
        and complete.get("state") == "complete"
        and all(int(complete.get(key, -1)) == 0 for key in loss_fields)
    )
    checks.append(Check(
        "complete_contract",
        marker_ok or allow_incomplete,
        "clean" if marker_ok else "invalid state/schema or non-zero capture loss",
    ))

    checksum_path = root / "checksums.sha256"
    checksum_entries = parse_checksums(checksum_path) if checksum_path.exists() else []
    checks.append(Check(
        "checksum_manifest",
        bool(checksum_entries) or allow_incomplete,
        f"{len(checksum_entries)} entries" if checksum_entries else "missing/empty",
    ))
    expected_manifest_hash = (complete or {}).get("checksumsManifestSha256", "")
    actual_manifest_hash = sha256(checksum_path) if checksum_path.is_file() else ""
    checks.append(Check(
        "checksum_manifest_hash",
        bool(expected_manifest_hash)
        and actual_manifest_hash == expected_manifest_hash.lower()
        or allow_incomplete,
        "match" if expected_manifest_hash and actual_manifest_hash == expected_manifest_hash.lower()
        else "missing/mismatch",
    ))
    bad_hashes: list[str] = []
    missing_files: list[str] = []
    for expected, relative in checksum_entries:
        path = root / Path(relative)
        if not path.is_file():
            missing_files.append(relative)
        elif sha256(path) != expected:
            bad_hashes.append(relative)
    checks.append(Check("payload_files", not missing_files, f"missing={len(missing_files)}"))
    checks.append(Check("payload_hashes", not bad_hashes, f"mismatch={len(bad_hashes)}"))

    pair_rows = csv_rows(root / "depth_pairs" / "manifest.csv")
    pair_ok = [row for row in pair_rows if row.get("status") == "ok"]
    pair_errors = [row for row in pair_rows if row.get("status") != "ok"]
    checks.append(Check(
        "depth_pairs",
        bool(pair_ok) and not pair_errors,
        f"ok={len(pair_ok)} error={len(pair_errors)} total={len(pair_rows)}",
    ))
    pair_missing = []
    for row in pair_ok:
        for key in ("rawFile", "processedFile", "metadataFile"):
            path = root / "depth_pairs" / "frames" / row[key]
            if not path.is_file():
                pair_missing.append(path.name)
    checks.append(Check("depth_pair_payloads", not pair_missing, f"missing={len(pair_missing)}"))

    fusion_rows = csv_rows(root / "fusion_inputs" / "manifest.csv")
    accepted = [row for row in fusion_rows if row.get("accepted") == "1"]
    rejected = [row for row in fusion_rows if row.get("accepted") == "0"]
    accepted_ok = [row for row in accepted if row.get("status") == "ok"]
    accepted_bad = [row for row in accepted if row.get("status") != "ok"]
    checks.append(Check(
        "fusion_inputs",
        bool(accepted_ok) and not accepted_bad,
        f"accepted_ok={len(accepted_ok)} accepted_bad={len(accepted_bad)} "
        f"accepted={len(accepted)} rejected={len(rejected)}",
    ))
    fusion_missing = []
    gun_gel_contract_errors: list[str] = []
    for row in accepted_ok:
        required = ["depthFile", "normalFile", "dilatedFile", "edgeReasonFile", "metaFile"]
        if row.get("gunGelAdmissionActive") == "1":
            required += ["gunGelObservationsFile", "gunGelCorrespondencesFile"]
        if row.get("cameraAvailable") == "1":
            required += ["cameraFile"]
        for key in required:
            value = row.get(key, "")
            if not value or not (root / "fusion_inputs" / "frames" / value).is_file():
                fusion_missing.append(f"{row.get('sequence')}:{key}")
        if row.get("gunGelAdmissionActive") == "1" and row.get("metaFile"):
            meta_path = root / "fusion_inputs" / "frames" / row["metaFile"]
            if meta_path.is_file():
                try:
                    admission = json.loads(meta_path.read_text("utf-8")).get(
                        "gunGelAdmission", {}
                    )
                    layout = str(admission.get("observationLayout", ""))
                    if int(admission.get("observationStrideBytes", -1)) != 64:
                        gun_gel_contract_errors.append(
                            f"{row.get('sequence')}:observationStrideBytes"
                        )
                    if int(admission.get("correspondenceStrideBytes", -1)) != 48:
                        gun_gel_contract_errors.append(
                            f"{row.get('sequence')}:correspondenceStrideBytes"
                        )
                    if "uint4 sourceReason" not in layout:
                        gun_gel_contract_errors.append(
                            f"{row.get('sequence')}:observationLayout"
                        )
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    gun_gel_contract_errors.append(f"{row.get('sequence')}:metaParse")
    checks.append(Check("fusion_payloads", not fusion_missing, f"missing={len(fusion_missing)}"))
    checks.append(Check(
        "gungel_binary_contract",
        not gun_gel_contract_errors,
        f"errors={len(gun_gel_contract_errors)}",
    ))

    config_ok = (root / "production_config.json").is_file()
    replay_contract_ok = (root / "replay_contract.json").is_file()
    coordinate_ok = (
        (root / "coordinate_contract.json").is_file()
        and (root / "coordinate_contract_stop.json").is_file()
    )
    checks.append(Check("production_config", config_ok, "present" if config_ok else "missing"))
    checks.append(Check(
        "replay_contract",
        replay_contract_ok,
        "present" if replay_contract_ok else "missing",
    ))
    checks.append(Check(
        "coordinate_contract",
        coordinate_ok,
        "start+stop present" if coordinate_ok else "start or stop contract missing",
    ))

    system_status_path = root / "system_reference" / "status.json"
    system_status = json.loads(system_status_path.read_text("utf-8")) if system_status_path.exists() else {}
    system_detail = system_status.get("status", "missing")
    # MRUK is a comparison companion, not a prerequisite for exact QRS replay.
    checks.append(Check("system_reference_status", bool(system_status), system_detail))

    artifact_statuses = {}
    artifacts = root / "artifacts"
    if artifacts.exists():
        for path in sorted(artifacts.glob("*_status.json")):
            artifact_statuses[path.stem] = json.loads(path.read_text("utf-8"))

    summary = {
        "schema": "scancover.replay_session_validation.v1",
        "session": str(root),
        "replayable": all(check.ok for check in checks if check.name != "system_reference_status"),
        "complete": complete,
        "counts": {
            "depthPairsOk": len(pair_ok),
            "fusionAcceptedOk": len(accepted_ok),
            "fusionRejected": len(rejected),
            "checksumEntries": len(checksum_entries),
        },
        "systemReference": system_status,
        "artifactStatuses": artifact_statuses,
        "failures": {
            "missingFiles": missing_files,
            "hashMismatches": bad_hashes,
            "depthPairMissing": pair_missing,
            "fusionMissing": fusion_missing,
            "gunGelBinaryContract": gun_gel_contract_errors,
        },
        "checks": [check.__dict__ for check in checks],
    }
    return checks, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path, help="local replay_sessions/session_* directory")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()

    checks, summary = validate(args.session, args.allow_incomplete)
    for check in checks:
        print(("OK  " if check.ok else "FAIL") + f" {check.name}: {check.detail}")
    print("REPLAYABLE=" + ("yes" if summary["replayable"] else "no"))

    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8"
        )
    return 0 if summary["replayable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
