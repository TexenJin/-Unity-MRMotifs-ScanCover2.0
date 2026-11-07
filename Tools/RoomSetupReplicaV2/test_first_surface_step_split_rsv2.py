#!/usr/bin/env python3
"""Contract checks for the bounded multiscale first-surface StepBand route."""

from __future__ import annotations

from pathlib import Path


MIN_OUTER = 5
MAX_OUTER = 19
OUTER_STEP = 2
PLATEAU_THICKNESS = 2
MIN_SEPARATION = 0.15
MAX_PLATEAU_SPREAD = 0.025
MIN_SNAP = 0.025
SAME_LAYER_SUPPORT = 0.04


def detect_1d(depths: list[float], center_index: int) -> float | None:
    center = depths[center_index]
    best: tuple[float, float] | None = None
    for outer in range(MIN_OUTER, MAX_OUTER + 1, OUTER_STEP):
        if center_index - outer < 0 or center_index + outer >= len(depths):
            continue
        inner = outer - PLATEAU_THICKNESS
        first_samples = [
            depths[center_index - inner],
            depths[center_index - outer],
        ]
        second_samples = [
            depths[center_index + inner],
            depths[center_index + outer],
        ]
        if (max(first_samples) - min(first_samples) > MAX_PLATEAU_SPREAD or
                max(second_samples) - min(second_samples) > MAX_PLATEAU_SPREAD):
            continue
        first = sum(first_samples) / len(first_samples)
        second = sum(second_samples) / len(second_samples)
        low, high = sorted((first, second))
        if (high - low < MIN_SEPARATION or
                center <= low + MIN_SNAP or
                center >= high - MIN_SNAP):
            continue
        low_delta = center - low
        high_delta = high - center
        target = low if low_delta <= high_delta + 0.0005 else high
        score = high - low - min(low_delta, high_delta) * 0.1 - outer * 0.001
        if best is None or score > best[0]:
            best = (score, target)
    return None if best is None else best[1]


def same_layer_gradient_accepts(center: float, neighbors: list[float]) -> bool:
    return sum(abs(value - center) <= SAME_LAYER_SUPPORT
               for value in neighbors) >= 2


def test_smoothed_step_returns_to_one_measured_plateau() -> None:
    depths = [2.0] * 24 + [2.09, 2.18, 2.27] + [2.36] * 24
    assert detect_1d(depths, 25) == 2.0
    assert detect_1d(depths, 26) == 2.36


def test_wide_public_depth_blur_is_recovered_across_the_band() -> None:
    width = 15
    depths = ([2.0] * 24 +
              [2.0 + 0.36 * (index + 1) / (width + 1)
               for index in range(width)] +
              [2.36] * 24)
    hits = [detect_1d(depths, 24 + index) is not None
            for index in range(width)]
    assert sum(hits) >= 13


def test_real_linear_slope_is_not_quantized() -> None:
    depths = [2.0 + index * 0.02 for index in range(61)]
    assert detect_1d(depths, 30) is None


def test_small_depth_change_is_not_treated_as_step() -> None:
    depths = [2.0] * 24 + [2.03, 2.05, 2.07] + [2.09] * 24
    assert detect_1d(depths, 25) is None


def test_qualified_hard_edge_uses_same_layer_support() -> None:
    assert same_layer_gradient_accepts(2.0, [2.0, 2.36, 2.01, 2.0])
    assert not same_layer_gradient_accepts(2.0, [2.36, 2.36, 2.36, 2.0])


def test_production_route_is_before_gradient_and_first_tsdf_write() -> None:
    repo = Path(__file__).resolve().parents[2]
    source = (repo / "Assets" / "RoomSetupReplicaV2" / "Runtime" /
              "RoomSetupReplicaV2Reconstruction.cs").read_text(encoding="utf-8")
    switch_start = source.index("case WorkStage.DecodeRight:")
    detect = source.index("case WorkStage.DetectStepLeft:", switch_start)
    apply_right = source.index("case WorkStage.ApplyStepRight:", detect)
    gradient = source.index("case WorkStage.GradientLeft:", apply_right)
    integrate = source.index("case WorkStage.IntegrateLeft:", gradient)
    assert switch_start < detect < apply_right < gradient < integrate
    assert '"A-gpu-v0.1.1-fusion-mc-hard3m-v9"' in source
    assert "StepSplitMinimumLayerSeparationMeters = 0.15f" in source
    assert "StepSplitMaximumStereoAnchorSeparationMeters = 0.075f" in source
    assert "StepSplitMaximumOuterOffsetPixels = 19" in source
    assert "HasSameStepLayerSupport(depths, index)" in source
    assert "firstSurfaceStepAcceptedEyePixels" in source


def main() -> int:
    test_smoothed_step_returns_to_one_measured_plateau()
    test_wide_public_depth_blur_is_recovered_across_the_band()
    test_real_linear_slope_is_not_quantized()
    test_small_depth_change_is_not_treated_as_step()
    test_qualified_hard_edge_uses_same_layer_support()
    test_production_route_is_before_gradient_and_first_tsdf_write()
    print("6 multiscale first-surface StepBand contract tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
