#!/usr/bin/env python3
"""Replay a conservative surface-position court before TSDF integration.

The replay treats each GunGel candidate index as a surface address, never as
ground truth.  Every independent view cluster contributes exactly one median
range testimony.  Views are ordered by source-frame index and divided into
three disjoint roles:

* proposal views expose one or more persistent depth-layer hypotheses;
* selection views choose a hypothesis or force the court to abstain;
* audit views are untouched until the final predictive score is measured.

This is a read-only offline trial of the final adjudication core.  It neither
changes Unity production state nor claims absolute physical range truth.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "p50": None, "p75": None, "p90": None, "p95": None}
    return {
        "count": int(len(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def dominant_mode(values: np.ndarray) -> tuple[float, int] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (np.abs(values) <= 150.0)]
    if len(values) < 5:
        return None
    edges = np.arange(-155.0, 165.0, 10.0)
    counts, _ = np.histogram(values, bins=edges)
    peak = int(np.argmax(counts))
    center = (edges[peak] + edges[peak + 1]) * 0.5
    members = values[np.abs(values - center) <= 20.0]
    if len(members) < 5:
        return None
    return float(np.median(members)), int(len(members))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class Hypothesis:
    center_mm: float
    support_views: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-way held-out final surface court replay.")
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-role-views", type=int, default=5)
    parser.add_argument("--candidate-band-mm", type=float, default=150.0)
    parser.add_argument("--maximum-raw-post-mm", type=float, default=40.0)
    parser.add_argument("--minimum-incidence", type=float, default=0.30)
    parser.add_argument("--mode-radius-mm", type=float, default=20.0)
    parser.add_argument("--minimum-mode-share", type=float, default=0.15)
    parser.add_argument("--maximum-selection-residual-mm", type=float, default=20.0)
    parser.add_argument("--minimum-selection-mode-share", type=float, default=0.50)
    parser.add_argument("--minimum-winner-margin-mm", type=float, default=5.0)
    parser.add_argument("--maximum-hypotheses", type=int, default=3)
    parser.add_argument(
        "--role-split",
        choices=("chronological", "interleaved"),
        default="chronological",
        help="Chronological thirds test delayed prediction; interleaved thirds test cross-view consistency.",
    )
    return parser.parse_args()


def per_view_testimony(
    values: np.ndarray,
    views: np.ndarray,
    pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(views)
    rows = [
        (
            int(view_id),
            float(np.median(values[views == view_id])),
            float(np.median(pairs[views == view_id])),
        )
        for view_id in unique
    ]
    rows.sort(key=lambda row: (row[2], row[0]))
    return (
        np.asarray([row[0] for row in rows], dtype=np.int32),
        np.asarray([row[1] for row in rows], dtype=np.float64),
        np.asarray([row[2] for row in rows], dtype=np.float64),
    )


def split_three(
    values: np.ndarray,
    minimum: int,
    role_split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    count = len(values)
    if count < minimum * 3:
        return None
    if role_split == "interleaved":
        proposal, selection, audit = values[0::3], values[1::3], values[2::3]
    else:
        proposal_end = count // 3
        selection_end = (count * 2) // 3
        proposal = values[:proposal_end]
        selection = values[proposal_end:selection_end]
        audit = values[selection_end:]
    if min(len(proposal), len(selection), len(audit)) < minimum:
        return None
    return proposal, selection, audit


def mode_hypotheses(
    values: np.ndarray,
    radius_mm: float,
    minimum_share: float,
    maximum_hypotheses: int,
) -> list[Hypothesis]:
    """Extract non-averaged 1D depth layers from one-vote-per-view testimony."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return []

    remaining = np.ones(len(values), dtype=bool)
    minimum_support = max(2, int(np.ceil(len(values) * minimum_share)))
    hypotheses: list[Hypothesis] = []
    while np.any(remaining) and len(hypotheses) < maximum_hypotheses:
        active = values[remaining]
        # A testimony-centred fixed-radius density search preserves separated
        # layers without inventing a mean between them.
        support = np.asarray(
            [np.count_nonzero(np.abs(active - seed) <= radius_mm) for seed in active],
            dtype=np.int32,
        )
        peak_seed = float(active[int(np.argmax(support))])
        members = remaining & (np.abs(values - peak_seed) <= radius_mm)
        if np.count_nonzero(members) < minimum_support:
            break
        center = float(np.median(values[members]))
        # Re-centre once so an edge seed cannot bias the membership interval.
        members = remaining & (np.abs(values - center) <= radius_mm)
        member_count = int(np.count_nonzero(members))
        if member_count < minimum_support:
            break
        hypotheses.append(Hypothesis(center, member_count))
        remaining[members] = False

    if not hypotheses:
        hypotheses.append(Hypothesis(float(np.median(values)), len(values)))
    hypotheses.sort(key=lambda item: (-item.support_views, item.center_mm))
    return hypotheses


def mode_target(values: np.ndarray, radius_mm: float) -> tuple[float, int, float] | None:
    mode = dominant_mode(values)
    if mode is None:
        return None
    center, members = mode
    share = float(members / max(len(values), 1))
    # dominant_mode uses a fixed 20 mm membership band; keep the explicit
    # radius argument in the contract so trial variants remain auditable.
    if radius_mm != 20.0:
        members = int(np.count_nonzero(np.abs(values - center) <= radius_mm))
        share = float(members / max(len(values), 1))
    return float(center), int(members), share


def algorithm_row(name: str, errors: np.ndarray, population: str) -> dict[str, object]:
    stats = distribution(errors)
    return {
        "algorithm": name,
        "population": population,
        "candidateCount": int(len(errors)),
        "absErrorP50Mm": stats["p50"],
        "absErrorP75Mm": stats["p75"],
        "absErrorP90Mm": stats["p90"],
        "absErrorP95Mm": stats["p95"],
        "within10mmPct": float(np.mean(errors <= 10.0) * 100.0) if len(errors) else None,
        "within20mmPct": float(np.mean(errors <= 20.0) * 100.0) if len(errors) else None,
        "over50mmPct": float(np.mean(errors > 50.0) * 100.0) if len(errors) else None,
    }


def main() -> int:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    payload = np.load(args.observations.resolve())
    candidate = payload["candidate"].astype(np.int32)
    view = payload["view"].astype(np.int32)
    pair = payload["pair"].astype(np.int32)
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

    rows: list[dict[str, object]] = []
    for candidate_id in range(candidate_count):
        indices = order[starts[candidate_id] : ends[candidate_id]]
        if not len(indices):
            continue
        _, testimonies, _ = per_view_testimony(raw[indices], view[indices], pair[indices])
        split = split_three(testimonies, args.minimum_role_views, args.role_split)
        if split is None:
            continue
        proposal, selection, audit = split
        selection_target = mode_target(selection, args.mode_radius_mm)
        audit_target = mode_target(audit, args.mode_radius_mm)
        if selection_target is None or audit_target is None:
            continue

        hypotheses = mode_hypotheses(
            proposal,
            args.mode_radius_mm,
            args.minimum_mode_share,
            args.maximum_hypotheses,
        )
        if not hypotheses:
            continue
        selection_center, selection_members, selection_share = selection_target
        audit_center, audit_members, audit_share = audit_target
        ranked = sorted(
            ((abs(item.center_mm - selection_center), item) for item in hypotheses),
            key=lambda item: (item[0], -item[1].support_views),
        )
        winner_residual, winner = ranked[0]
        runner_residual = ranked[1][0] if len(ranked) > 1 else None
        winner_margin = (runner_residual - winner_residual) if runner_residual is not None else None
        reasons: list[str] = []
        if winner_residual > args.maximum_selection_residual_mm:
            reasons.append("selection_residual")
        if selection_share < args.minimum_selection_mode_share:
            reasons.append("selection_mode_share")
        if winner_margin is not None and winner_margin < args.minimum_winner_margin_mm:
            reasons.append("winner_margin")
        accepted = not reasons

        proposal_median = float(np.median(proposal))
        proposal_mode = dominant_mode(proposal)
        proposal_mode_center = float(proposal_mode[0]) if proposal_mode is not None else proposal_median
        selection_direct = float(selection_center)
        court_prediction = float(winner.center_mm)
        oracle = min(hypotheses, key=lambda item: abs(item.center_mm - audit_center))
        rows.append(
            {
                "candidateIndex": candidate_id,
                "totalViews": len(testimonies),
                "proposalViews": len(proposal),
                "selectionViews": len(selection),
                "auditViews": len(audit),
                "hypothesisCount": len(hypotheses),
                "hypothesesMm": ";".join(f"{item.center_mm:.4f}" for item in hypotheses),
                "hypothesisSupportViews": ";".join(str(item.support_views) for item in hypotheses),
                "selectionTargetMm": selection_center,
                "selectionModeMembers": selection_members,
                "selectionModeSharePct": selection_share * 100.0,
                "auditTargetMm": audit_center,
                "auditModeMembers": audit_members,
                "auditModeSharePct": audit_share * 100.0,
                "courtAccepted": int(accepted),
                "courtAbstainReason": "+".join(reasons) if reasons else "none",
                "winnerSelectionResidualMm": winner_residual,
                "winnerMarginMm": winner_margin,
                "proposalMedianPredictionMm": proposal_median,
                "proposalMedianAbsErrorMm": abs(proposal_median - audit_center),
                "proposalDominantPredictionMm": proposal_mode_center,
                "proposalDominantAbsErrorMm": abs(proposal_mode_center - audit_center),
                "selectionDirectPredictionMm": selection_direct,
                "selectionDirectAbsErrorMm": abs(selection_direct - audit_center),
                "courtPredictionMm": court_prediction,
                "courtAbsErrorMm": abs(court_prediction - audit_center),
                "oracleHypothesisPredictionMm": oracle.center_mm,
                "oracleHypothesisAbsErrorMm": abs(oracle.center_mm - audit_center),
            }
        )

    fields = list(rows[0].keys()) if rows else ["candidateIndex"]
    write_csv(out / "final_court_candidates.csv", rows, fields)
    accepted_rows = [row for row in rows if row["courtAccepted"] == 1]

    mappings = (
        ("proposal_median", "proposalMedianAbsErrorMm"),
        ("proposal_dominant", "proposalDominantAbsErrorMm"),
        ("selection_direct", "selectionDirectAbsErrorMm"),
        ("final_court", "courtAbsErrorMm"),
        ("oracle_hypothesis_upper_bound", "oracleHypothesisAbsErrorMm"),
    )
    summary_rows: list[dict[str, object]] = []
    for population, source in (("all_scorable", rows), ("court_accepted_common_set", accepted_rows)):
        for name, field in mappings:
            errors = np.asarray([float(row[field]) for row in source], dtype=np.float64)
            summary_rows.append(algorithm_row(name, errors, population))
    write_csv(out / "final_court_summary.csv", summary_rows, list(summary_rows[0].keys()))

    reason_counts: dict[str, int] = {}
    for row in rows:
        if row["courtAccepted"] == 1:
            continue
        for reason in str(row["courtAbstainReason"]).split("+"):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    accepted_count = len(accepted_rows)
    report = {
        "schema": "scancover.offline_final_surface_court.v1",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "authority": "read_only_offline_diagnostic",
        "productionConsumers": 0,
        "surfaceAddressSource": "GunGel candidate index; address only, never truth",
        "grain": "one candidate address and one median testimony per independent view cluster",
        "configuration": {
            "minimumRoleViews": args.minimum_role_views,
            "candidateBandMm": args.candidate_band_mm,
            "maximumRawPostMm": args.maximum_raw_post_mm,
            "minimumIncidence": args.minimum_incidence,
            "modeRadiusMm": args.mode_radius_mm,
            "minimumModeShare": args.minimum_mode_share,
            "maximumSelectionResidualMm": args.maximum_selection_residual_mm,
            "minimumSelectionModeShare": args.minimum_selection_mode_share,
            "minimumWinnerMarginMm": args.minimum_winner_margin_mm,
            "maximumHypotheses": args.maximum_hypotheses,
        },
        "roles": {
            "split": args.role_split,
            "proposal": "first role; exposes separated depth hypotheses",
            "selection": "second role; selects one proposal hypothesis or abstains",
            "audit": "third role; untouched blind predictive target",
        },
        "scorableCandidates": len(rows),
        "acceptedCandidates": accepted_count,
        "abstainedCandidates": len(rows) - accepted_count,
        "acceptancePct": float(accepted_count / max(len(rows), 1) * 100.0),
        "multiHypothesisCandidates": sum(int(row["hypothesisCount"]) > 1 for row in rows),
        "abstainReasonCounts": reason_counts,
        "algorithms": summary_rows,
        "boundaries": [
            "audit target is independent in view but still comes from the same Quest sensor",
            "chronological split measures delayed prediction; interleaved split measures cross-view consistency",
            "candidate addresses define association only and are not geometric truth",
            "this trial tests surface-layer selection; it does not replay TSDF writing, withdrawal, or atomic replacement",
            "accuracy must be read together with acceptance rate and same-population baselines",
            "a sensor bias shared by every independent view remains unobservable without external truth",
        ],
        "artifacts": {
            "perCandidate": "final_court_candidates.csv",
            "summary": "final_court_summary.csv",
        },
    }
    (out / "final_court_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
