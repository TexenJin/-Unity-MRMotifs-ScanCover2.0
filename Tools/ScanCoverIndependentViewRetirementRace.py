#!/usr/bin/env python3
"""Race independent-view retirement contracts on one sealed ScanCover session.

This tool is deliberately offline and read-only.  It uses the post-retirement
ghost ledger to compare four policies without changing Unity production state:

1. production: every audited ghost was actually deleted;
2. legacy8: preserve the old eight-visible-frame shadow result;
3. support_veto: two later support observations with mutually separated
   candidate rays veto deletion, while every other ghost remains deletable;
4. symmetric: cross-view support retains, two mutually separated free views
   authorize deletion, support/free conflict defers, and missing evidence defers.

The policy formulas never use room coordinates, stable IDs, or hand-picked
examples.  World positions are used only after adjudication to associate an
outcome with a nearby paper failure cell.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_ANGLE_THRESHOLDS_DEG = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0)
DEFAULT_PAPER_LINK_RADIUS_M = 0.225
DEFAULT_SAME_VIEW_TEMPORAL_CONE_DEG = 0.25


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def integer(value: object, default: int = 0) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def number(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def require_columns(path: Path, rows: list[dict[str, str]], required: set[str]) -> None:
    columns = set(rows[0]) if rows else set()
    missing = sorted(required - columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {', '.join(missing)}")


def unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in vector))
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return tuple(component / length for component in vector)  # type: ignore[return-value]


def angle_deg(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    left_unit = unit(left)
    right_unit = unit(right)
    if left_unit == (0.0, 0.0, 0.0) or right_unit == (0.0, 0.0, 0.0):
        return 0.0
    cosine = max(-1.0, min(1.0, sum(a * b for a, b in zip(left_unit, right_unit))))
    return math.degrees(math.acos(cosine))


def distance(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


@dataclass(frozen=True)
class Observation:
    frame: int
    classification: str
    camera: tuple[float, float, float]
    origin_separation_deg: float
    lateral_baseline_mm: float
    incidence_cosine: float
    motion_quality: float


@dataclass
class Ghost:
    index: int
    stable_id: int
    retired_frame: int
    legacy_status: str
    center: tuple[float, float, float]
    observations: list[Observation]
    nearest_paper_cause: str = ""
    nearest_paper_distance_m: float = math.inf

    def observations_of(self, classification: str) -> list[Observation]:
        return [
            observation for observation in self.observations
            if observation.classification == classification
        ]


def maximum_pairwise_ray_separation(
    center: tuple[float, float, float], observations: list[Observation]
) -> float:
    maximum = 0.0
    for index, left in enumerate(observations):
        left_ray = tuple(c - p for c, p in zip(center, left.camera))
        for right in observations[index + 1:]:
            right_ray = tuple(c - p for c, p in zip(center, right.camera))
            maximum = max(maximum, angle_deg(left_ray, right_ray))
    return maximum


def maximum_origin_separation(observations: list[Observation]) -> float:
    return max((observation.origin_separation_deg for observation in observations), default=0.0)


def temporal_consistency_filter(
    center: tuple[float, float, float], observations: list[Observation], cone_deg: float
) -> tuple[list[Observation], int, int, int]:
    """Remove both sides of same-view support/free contradictions.

    Returns eligible observations, conflicting pair count, invalidated support
    count and invalidated free count. The operation is order-independent and
    preserves every raw receipt in the source CSV.
    """
    relevant = [
        item for item in observations
        if item.classification in {"coherent_support", "universal_free"}
    ]
    invalid: set[int] = set()
    conflict_pairs = 0
    for left_index, left in enumerate(relevant):
        left_ray = tuple(c - p for c, p in zip(center, left.camera))
        for right_index in range(left_index + 1, len(relevant)):
            right = relevant[right_index]
            if left.classification == right.classification:
                continue
            right_ray = tuple(c - p for c, p in zip(center, right.camera))
            if angle_deg(left_ray, right_ray) > cone_deg:
                continue
            invalid.add(left_index)
            invalid.add(right_index)
            conflict_pairs += 1
    eligible = [item for index, item in enumerate(relevant) if index not in invalid]
    invalid_support = sum(
        relevant[index].classification == "coherent_support" for index in invalid
    )
    invalid_free = sum(
        relevant[index].classification == "universal_free" for index in invalid
    )
    return eligible, conflict_pairs, invalid_support, invalid_free


def symmetric_outcome(support_cross_view: bool, free_cross_view: bool) -> str:
    if support_cross_view and free_cross_view:
        return "defer_conflict"
    if support_cross_view:
        return "retain_cross_view_support"
    if free_cross_view:
        return "delete_cross_view_free"
    return "defer_insufficient_view_diversity"


def legacy8_outcome(status: str) -> str:
    if status == "support_reappeared":
        return "retain_support_reappeared"
    if status == "sustained_free":
        return "delete_sustained_free"
    return "defer_legacy_inconclusive"


def run_self_tests() -> None:
    assert symmetric_outcome(True, False) == "retain_cross_view_support"
    assert symmetric_outcome(False, True) == "delete_cross_view_free"
    assert symmetric_outcome(True, True) == "defer_conflict"
    assert symmetric_outcome(False, False) == "defer_insufficient_view_diversity"
    assert abs(angle_deg((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)) - 90.0) < 1e-6


def load_ghosts(ghost_path: Path, observation_path: Path) -> list[Ghost]:
    ghost_rows = read_csv(ghost_path)
    observation_rows = read_csv(observation_path)
    require_columns(
        ghost_path,
        ghost_rows,
        {
            "ghost_index", "stable_id", "retired_frame", "status",
            "center_x_m", "center_y_m", "center_z_m",
        },
    )
    require_columns(
        observation_path,
        observation_rows,
        {
            "ghost_index", "frame", "classification", "camera_x_m",
            "camera_y_m", "camera_z_m", "lateral_baseline_mm",
            "ray_separation_deg", "incidence_cosine", "motion_quality",
        },
    )
    observations_by_ghost: dict[int, list[Observation]] = defaultdict(list)
    for row in observation_rows:
        observations_by_ghost[integer(row.get("ghost_index"))].append(
            Observation(
                frame=integer(row.get("frame")),
                classification=row.get("classification", ""),
                camera=(
                    number(row.get("camera_x_m")),
                    number(row.get("camera_y_m")),
                    number(row.get("camera_z_m")),
                ),
                origin_separation_deg=number(row.get("ray_separation_deg")),
                lateral_baseline_mm=number(row.get("lateral_baseline_mm")),
                incidence_cosine=number(row.get("incidence_cosine")),
                motion_quality=number(row.get("motion_quality")),
            )
        )
    ghosts: list[Ghost] = []
    for row in ghost_rows:
        index = integer(row.get("ghost_index"))
        observations = sorted(
            observations_by_ghost.get(index, []), key=lambda item: item.frame
        )
        ghosts.append(
            Ghost(
                index=index,
                stable_id=integer(row.get("stable_id")),
                retired_frame=integer(row.get("retired_frame")),
                legacy_status=row.get("status", ""),
                center=(
                    number(row.get("center_x_m")),
                    number(row.get("center_y_m")),
                    number(row.get("center_z_m")),
                ),
                observations=observations,
            )
        )
    return ghosts


def associate_paper_failures(
    ghosts: list[Ghost], paper_path: Path, link_radius_m: float
) -> Counter[str]:
    if not paper_path.is_file():
        return Counter()
    rows = read_csv(paper_path)
    require_columns(
        paper_path,
        rows,
        {"world_x_m", "world_y_m", "world_z_m", "cause", "empty_surface_halo"},
    )
    failures = [
        (
            (
                number(row.get("world_x_m")),
                number(row.get("world_y_m")),
                number(row.get("world_z_m")),
            ),
            row.get("cause", ""),
        )
        for row in rows
        if row.get("empty_surface_halo") == "0"
        and row.get("cause") not in {"candidate_emitted", "published_replay"}
    ]
    linked_causes: Counter[str] = Counter()
    for ghost in ghosts:
        for position, cause in failures:
            candidate_distance = distance(ghost.center, position)
            if candidate_distance < ghost.nearest_paper_distance_m:
                ghost.nearest_paper_distance_m = candidate_distance
                ghost.nearest_paper_cause = cause
        if ghost.nearest_paper_distance_m <= link_radius_m:
            linked_causes[ghost.nearest_paper_cause] += 1
    return linked_causes


def main() -> None:
    run_self_tests()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="sealed replay session directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--angles-deg", type=float, nargs="+",
        default=list(DEFAULT_ANGLE_THRESHOLDS_DEG),
        help="view-ray separation thresholds to race",
    )
    parser.add_argument(
        "--paper-link-radius-m", type=float, default=DEFAULT_PAPER_LINK_RADIUS_M,
        help="diagnostic-only paper failure association radius",
    )
    parser.add_argument(
        "--same-view-temporal-cone-deg", type=float,
        default=DEFAULT_SAME_VIEW_TEMPORAL_CONE_DEG,
        help="support/free receipts inside this candidate-ray cone invalidate each other",
    )
    args = parser.parse_args()

    session = args.session.resolve()
    output = args.output.resolve()
    complete_path = session / "capture_complete.json"
    audit = session / "artifacts" / "gungel_candidate_audit"
    summary_path = audit / "candidate_summary.json"
    ghost_path = audit / "retirement_ghosts.csv"
    observation_path = audit / "retirement_ghost_observations.csv"
    paper_path = session / "artifacts" / "paper_audit" / "paper_hole_cells.csv"
    lineage_summary_path = (
        session / "artifacts" / "evidence_lineage" / "evidence_lineage_summary.json"
    )
    for path in (complete_path, summary_path, ghost_path, observation_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    complete = json.loads(complete_path.read_text(encoding="utf-8-sig"))
    candidate_summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if complete.get("state") != "complete":
        raise RuntimeError(f"session is not sealed complete: {complete_path}")
    ghost_summary = candidate_summary.get("retirement_ghost_shadow", {})
    if ghost_summary.get("authority") != "shadow_only_zero_production_authority":
        raise RuntimeError("retirement ghost input is not marked shadow-only")
    if ghost_summary.get("independent_view_threshold") != "none_feature_ledger_only":
        raise RuntimeError("input does not contain the continuous independent-view ledger")

    link_radius_m = max(0.0, args.paper_link_radius_m)
    if lineage_summary_path.is_file():
        lineage_summary = json.loads(
            lineage_summary_path.read_text(encoding="utf-8-sig")
        )
        link_radius_m = number(
            lineage_summary.get("spatialLinkRadiusM"), link_radius_m
        )

    ghosts = load_ghosts(ghost_path, observation_path)
    linked_causes = associate_paper_failures(ghosts, paper_path, link_radius_m)
    expected_ghosts = integer(ghost_summary.get("retained_rows"), -1)
    expected_observations = integer(
        ghost_summary.get("independent_observation_rows"), -1
    )
    if expected_ghosts >= 0 and len(ghosts) != expected_ghosts:
        raise RuntimeError(
            f"ghost reconciliation failed: csv={len(ghosts)} summary={expected_ghosts}"
        )
    actual_observations = sum(len(ghost.observations) for ghost in ghosts)
    if expected_observations >= 0 and actual_observations != expected_observations:
        raise RuntimeError(
            "observation reconciliation failed: "
            f"csv={actual_observations} summary={expected_observations}"
        )

    per_ghost_features: dict[int, dict[str, float | int]] = {}
    same_view_cone_deg = max(0.0, args.same_view_temporal_cone_deg)
    for ghost in ghosts:
        support = ghost.observations_of("coherent_support")
        free = ghost.observations_of("universal_free")
        mixed = ghost.observations_of("mixed_boundary")
        eligible, conflict_pairs, invalid_support, invalid_free = (
            temporal_consistency_filter(
                ghost.center, ghost.observations, same_view_cone_deg
            )
        )
        consistent_support = [
            item for item in eligible if item.classification == "coherent_support"
        ]
        consistent_free = [
            item for item in eligible if item.classification == "universal_free"
        ]
        per_ghost_features[ghost.index] = {
            "support_rows": len(support),
            "free_rows": len(free),
            "mixed_rows": len(mixed),
            "support_origin_span_deg": maximum_origin_separation(support),
            "free_pair_span_deg": maximum_pairwise_ray_separation(ghost.center, free),
            "support_pair_span_deg": maximum_pairwise_ray_separation(ghost.center, support),
            "consistent_support_pair_span_deg": maximum_pairwise_ray_separation(
                ghost.center, consistent_support
            ),
            "consistent_free_pair_span_deg": maximum_pairwise_ray_separation(
                ghost.center, consistent_free
            ),
            "same_view_temporal_conflict_pairs": conflict_pairs,
            "same_view_support_invalidated": invalid_support,
            "same_view_free_invalidated": invalid_free,
            "max_support_lateral_mm": max(
                (item.lateral_baseline_mm for item in support), default=0.0
            ),
            "max_free_lateral_mm": max(
                (item.lateral_baseline_mm for item in free), default=0.0
            ),
        }

    outcome_rows: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []
    for threshold in sorted(set(max(0.0, value) for value in args.angles_deg)):
        symmetric_counts: Counter[str] = Counter()
        support_veto_retained = 0
        support_veto_deleted = 0
        legacy_counts: Counter[str] = Counter()
        linked_by_symmetric: Counter[str] = Counter()
        for ghost in ghosts:
            features = per_ghost_features[ghost.index]
            support_cross = number(
                features["consistent_support_pair_span_deg"]
            ) >= threshold
            free_cross = number(
                features["consistent_free_pair_span_deg"]
            ) >= threshold
            symmetric = symmetric_outcome(support_cross, free_cross)
            support_veto = (
                "retain_cross_view_support" if support_cross
                else "delete_no_cross_view_support"
            )
            legacy = legacy8_outcome(ghost.legacy_status)
            symmetric_counts[symmetric] += 1
            legacy_counts[legacy] += 1
            if support_cross:
                support_veto_retained += 1
            else:
                support_veto_deleted += 1
            linked = ghost.nearest_paper_distance_m <= link_radius_m
            if linked:
                linked_by_symmetric[symmetric] += 1
            outcome_rows.append(
                {
                    "angleThresholdDeg": threshold,
                    "ghostIndex": ghost.index,
                    "stableId": ghost.stable_id,
                    "retiredFrame": ghost.retired_frame,
                    "legacy8Status": ghost.legacy_status,
                    "legacy8Outcome": legacy,
                    "supportRows": features["support_rows"],
                    "freeRows": features["free_rows"],
                    "mixedRows": features["mixed_rows"],
                    "supportOriginSpanDeg": features["support_origin_span_deg"],
                    "supportPairSpanDeg": features["support_pair_span_deg"],
                    "freePairSpanDeg": features["free_pair_span_deg"],
                    "consistentSupportPairSpanDeg": features[
                        "consistent_support_pair_span_deg"
                    ],
                    "consistentFreePairSpanDeg": features[
                        "consistent_free_pair_span_deg"
                    ],
                    "sameViewTemporalConeDeg": same_view_cone_deg,
                    "sameViewTemporalConflictPairs": features[
                        "same_view_temporal_conflict_pairs"
                    ],
                    "sameViewSupportInvalidated": features[
                        "same_view_support_invalidated"
                    ],
                    "sameViewFreeInvalidated": features[
                        "same_view_free_invalidated"
                    ],
                    "maxSupportLateralMm": features["max_support_lateral_mm"],
                    "maxFreeLateralMm": features["max_free_lateral_mm"],
                    "supportCrossView": int(support_cross),
                    "freeCrossView": int(free_cross),
                    "supportVetoOutcome": support_veto,
                    "symmetricOutcome": symmetric,
                    "nearestPaperCause": ghost.nearest_paper_cause,
                    "nearestPaperDistanceM": (
                        ghost.nearest_paper_distance_m
                        if math.isfinite(ghost.nearest_paper_distance_m) else ""
                    ),
                    "paperFailureLinked": int(linked),
                    "authority": "offline_read_only_zero_production_authority",
                }
            )
        sweep_rows.append(
            {
                "angleThresholdDeg": threshold,
                "auditedProductionDeletions": len(ghosts),
                "legacy8Retain": legacy_counts["retain_support_reappeared"],
                "legacy8Delete": legacy_counts["delete_sustained_free"],
                "legacy8Defer": legacy_counts["defer_legacy_inconclusive"],
                "supportVetoRetain": support_veto_retained,
                "supportVetoDelete": support_veto_deleted,
                "symmetricRetain": symmetric_counts["retain_cross_view_support"],
                "symmetricDelete": symmetric_counts["delete_cross_view_free"],
                "symmetricConflict": symmetric_counts["defer_conflict"],
                "symmetricInsufficient": symmetric_counts[
                    "defer_insufficient_view_diversity"
                ],
                "linkedRetain": linked_by_symmetric["retain_cross_view_support"],
                "linkedDelete": linked_by_symmetric["delete_cross_view_free"],
                "linkedConflict": linked_by_symmetric["defer_conflict"],
                "linkedInsufficient": linked_by_symmetric[
                    "defer_insufficient_view_diversity"
                ],
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    outcome_fields = list(outcome_rows[0]) if outcome_rows else []
    sweep_fields = list(sweep_rows[0]) if sweep_rows else []
    write_csv(output / "ghost_contract_outcomes.csv", outcome_rows, outcome_fields)
    write_csv(output / "threshold_sweep.csv", sweep_rows, sweep_fields)

    observed = {
        "productionStableContradictionRetirements": integer(
            candidate_summary.get("legacy_retirement_guard16_shadow", {}).get(
                "stable_contradiction_rows"
            )
        ),
        "highRiskGhostsWithRecentSupport": len(ghosts),
        "independentObservationRows": actual_observations,
        "legacy8StatusCounts": dict(Counter(ghost.legacy_status for ghost in ghosts)),
        "paperLinkedGhosts": sum(
            ghost.nearest_paper_distance_m <= link_radius_m for ghost in ghosts
        ),
        "paperLinkedCauseCounts": dict(linked_causes),
        "sameViewTemporalConflictGhosts": sum(
            number(features["same_view_temporal_conflict_pairs"]) > 0
            for features in per_ghost_features.values()
        ),
        "sameViewTemporalConflictPairs": sum(
            integer(features["same_view_temporal_conflict_pairs"])
            for features in per_ghost_features.values()
        ),
        "sameViewSupportObservationsInvalidated": sum(
            integer(features["same_view_support_invalidated"])
            for features in per_ghost_features.values()
        ),
        "sameViewFreeObservationsInvalidated": sum(
            integer(features["same_view_free_invalidated"])
            for features in per_ghost_features.values()
        ),
    }
    summary = {
        "schema": "scancover.independent-view-retirement-race.v2",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "authority": "offline_read_only_zero_production_authority",
        "session": str(session),
        "inputs": {
            "captureComplete": str(complete_path),
            "candidateSummary": str(summary_path),
            "retirementGhosts": str(ghost_path),
            "retirementGhostObservations": str(observation_path),
            "paperHoleCells": str(paper_path) if paper_path.is_file() else "",
        },
        "contract": {
            "sameViewTemporalConsistency": "support/free observations inside the same candidate-ray cone invalidate each other before cross-view voting",
            "sameViewTemporalConeDeg": same_view_cone_deg,
            "supportWitness": "two temporally consistent later coherent-support observations whose candidate rays are mutually separated by the swept angular threshold",
            "freeWitness": "two temporally consistent later universal-free observations whose candidate rays are mutually separated by the swept angular threshold",
            "conflict": "qualifying support and free witnesses coexist; defer instead of retaining or deleting",
            "missingEvidence": "defer; absence of a qualifying receipt is not evidence of absence",
            "policyUsesRoomCoordinates": False,
            "paperAssociationAffectsPolicy": False,
            "paperLinkRadiusM": link_radius_m,
            "angleThresholdsDeg": [row["angleThresholdDeg"] for row in sweep_rows],
            "winnerSelection": "none_single_session_sensitivity_sweep_only",
        },
        "observed": observed,
        "limits": [
            "the ghost ledger covers only production retirements that also had valid recent direct support; the remaining stable retirements are unaudited, not validated deletes",
            "the retirement-origin camera is not the exact camera of the earlier support receipt, so origin separation is exported only as context and never used to prove support-view independence",
            "one room session can expose contradictions and sensitivity but cannot select a universal production angle threshold",
            "paper proximity is a diagnostic association and never participates in a retirement outcome",
            "dynamic objects and persistent platform depth bias remain possible explanations for support/free conflict",
        ],
        "outputs": {
            "ghostOutcomes": "ghost_contract_outcomes.csv",
            "thresholdSweep": "threshold_sweep.csv",
            "report": "report.md",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report_lines = [
        "# Independent-view retirement contract race",
        "",
        "This is a read-only sensitivity race. It does not choose a production threshold.",
        "",
        "| ray threshold | support-veto retain | support-veto delete | symmetric retain | symmetric delete | conflict defer | insufficient defer |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sweep_rows:
        report_lines.append(
            f"| {number(row['angleThresholdDeg']):.2f} deg "
            f"| {row['supportVetoRetain']} | {row['supportVetoDelete']} "
            f"| {row['symmetricRetain']} | {row['symmetricDelete']} "
            f"| {row['symmetricConflict']} | {row['symmetricInsufficient']} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            f"- Support/free observations within {same_view_cone_deg:.2f} degrees of the same candidate ray invalidate each other before voting.",
            "- Cross-view support and free each require two temporally consistent, mutually separated same-class observations; repeated same-view frames do not qualify.",
            "- Coexisting support/free witnesses are an explicit conflict, not a majority vote.",
            "- No room coordinate, stable ID, or paper state participates in the policy formulas.",
            "- Paper linkage is reported only to test whether retirement outcomes spatially accompany visible coverage failures.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(json.dumps(observed, indent=2, ensure_ascii=False))
    print("threshold sweep")
    for row in sweep_rows:
        print(
            f"  {number(row['angleThresholdDeg']):.2f}deg: "
            f"retain={row['symmetricRetain']} delete={row['symmetricDelete']} "
            f"conflict={row['symmetricConflict']} "
            f"insufficient={row['symmetricInsufficient']}"
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
