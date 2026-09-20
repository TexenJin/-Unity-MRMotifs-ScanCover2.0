#!/usr/bin/env python3
"""Compare lightweight pre-TSDF surface estimators on independent view clusters.

Even view clusters are training testimony.  Odd view clusters are never used by
the estimators; their dominant platform-depth mode is the held-out target.  Each
view contributes one median testimony per candidate so dwell time cannot create
extra votes.  This is an offline predictive test, not absolute ground truth and
not an exact TSDF replay.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ScanCoverRawDepthTrustAudit import distribution, dominant_mode, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Held-out surface-estimator replay.")
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-train-views", type=int, default=5)
    parser.add_argument("--minimum-heldout-views", type=int, default=5)
    parser.add_argument("--candidate-band-mm", type=float, default=150.0)
    parser.add_argument("--maximum-raw-post-mm", type=float, default=40.0)
    parser.add_argument("--minimum-incidence", type=float, default=0.30)
    return parser.parse_args()


def huber_location(values: np.ndarray) -> float:
    estimate = float(np.median(values))
    for _ in range(30):
        residual = values - estimate
        scale = max(1.4826 * float(np.median(np.abs(residual))), 2.0)
        threshold = 1.5 * scale
        weights = np.ones(len(values), dtype=np.float64)
        outside = np.abs(residual) > threshold
        weights[outside] = threshold / np.abs(residual[outside])
        updated = float(np.sum(weights * values) / np.sum(weights))
        if abs(updated - estimate) < 1e-4:
            return updated
        estimate = updated
    return estimate


def trimmed_mean(values: np.ndarray, fraction: float = 0.20) -> float:
    ordered = np.sort(values)
    trim = int(len(ordered) * fraction)
    if trim * 2 >= len(ordered):
        return float(np.mean(ordered))
    return float(np.mean(ordered[trim : len(ordered) - trim]))


def per_view_medians(values: np.ndarray, views: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(views)
    medians = np.asarray([np.median(values[views == view]) for view in unique], dtype=np.float64)
    return unique, medians


def main() -> int:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    payload = np.load(args.observations.resolve())
    candidate = payload["candidate"].astype(np.int32)
    view = payload["view"].astype(np.int32)
    raw = payload["rawNormalMm"].astype(np.float64)
    raw_post = payload["rawPostNormalMm"].astype(np.float64)
    incidence = payload["incidence"].astype(np.float64)
    candidate_count = len(payload["anchorRawNormalMm"])
    usable = (
        np.isfinite(raw)
        & (np.abs(raw) <= args.candidate_band_mm)
        & (np.abs(raw_post) <= args.maximum_raw_post_mm)
        & (incidence >= args.minimum_incidence)
    )

    usable_indices = np.flatnonzero(usable)
    order = usable_indices[np.argsort(candidate[usable_indices], kind="stable")]
    ordered_candidates = candidate[order]
    starts = np.searchsorted(ordered_candidates, np.arange(candidate_count), side="left")
    ends = np.searchsorted(ordered_candidates, np.arange(candidate_count), side="right")
    algorithms = ("first_view", "mean", "median", "trimmed_mean", "huber", "dominant_mode")
    rows: list[dict[str, object]] = []
    errors: dict[str, list[float]] = {name: [] for name in algorithms}

    for candidate_id in range(candidate_count):
        indices = order[starts[candidate_id] : ends[candidate_id]]
        if not len(indices):
            continue
        train_mask = view[indices] % 2 == 0
        heldout_mask = ~train_mask
        train_views, train_values = per_view_medians(raw[indices][train_mask], view[indices][train_mask])
        heldout_views, heldout_values = per_view_medians(raw[indices][heldout_mask], view[indices][heldout_mask])
        if len(train_views) < args.minimum_train_views or len(heldout_views) < args.minimum_heldout_views:
            continue
        heldout_mode = dominant_mode(heldout_values)
        training_mode = dominant_mode(train_values)
        if heldout_mode is None or training_mode is None:
            continue
        target = heldout_mode[0]
        predictions = {
            "first_view": float(train_values[0]),
            "mean": float(np.mean(train_values)),
            "median": float(np.median(train_values)),
            "trimmed_mean": trimmed_mean(train_values),
            "huber": huber_location(train_values),
            "dominant_mode": float(training_mode[0]),
        }
        row: dict[str, object] = {
            "candidateIndex": candidate_id,
            "trainViews": len(train_views),
            "heldoutViews": len(heldout_views),
            "heldoutTargetMm": target,
            "heldoutModeMembers": heldout_mode[1],
            "trainingModeMembers": training_mode[1],
        }
        for name, prediction in predictions.items():
            error = abs(prediction - target)
            errors[name].append(error)
            row[f"{name}PredictionMm"] = prediction
            row[f"{name}AbsErrorMm"] = error
        rows.append(row)

    fields = list(rows[0].keys()) if rows else ["candidateIndex"]
    write_csv(out / "surface_estimator_candidates.csv", rows, fields)
    summary_rows: list[dict[str, object]] = []
    for name in algorithms:
        values = np.asarray(errors[name], dtype=np.float64)
        stats = distribution(values)
        summary_rows.append(
            {
                "algorithm": name,
                "candidateCount": int(len(values)),
                "absErrorP50Mm": stats["p50"],
                "absErrorP75Mm": stats["p75"],
                "absErrorP90Mm": stats["p90"],
                "absErrorP95Mm": stats["p95"],
                "within10mmPct": float(np.mean(values <= 10.0) * 100.0) if len(values) else None,
                "within20mmPct": float(np.mean(values <= 20.0) * 100.0) if len(values) else None,
                "over50mmPct": float(np.mean(values > 50.0) * 100.0) if len(values) else None,
            }
        )
    write_csv(out / "surface_estimator_summary.csv", summary_rows, list(summary_rows[0].keys()) if summary_rows else ["algorithm"])
    ranked = sorted(
        summary_rows,
        key=lambda row: (float(row["absErrorP50Mm"]), float(row["absErrorP90Mm"])),
    ) if summary_rows and summary_rows[0]["absErrorP50Mm"] is not None else []
    report = {
        "schema": "scancover.offline_surface_estimator_replay.v1",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "authority": "read_only_offline_diagnostic",
        "productionConsumers": 0,
        "grain": "one candidate, one median testimony per independent view cluster",
        "training": "even independent view clusters",
        "heldoutTarget": "dominant mode of odd independent view clusters",
        "candidateAddressIsGroundTruth": False,
        "scoredCandidates": len(rows),
        "ranking": [row["algorithm"] for row in ranked],
        "algorithms": summary_rows,
        "boundaries": [
            "held-out target is independent in viewpoint, not independent in sensor source",
            "mean is a surface-position proxy for historical averaging, not an exact voxel TSDF replay",
            "candidate addresses only define where testimony is gathered",
            "this comparison chooses an estimator but cannot establish absolute physical range",
        ],
        "artifacts": {
            "perCandidate": "surface_estimator_candidates.csv",
            "summary": "surface_estimator_summary.csv",
        },
    }
    (out / "surface_estimator_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
