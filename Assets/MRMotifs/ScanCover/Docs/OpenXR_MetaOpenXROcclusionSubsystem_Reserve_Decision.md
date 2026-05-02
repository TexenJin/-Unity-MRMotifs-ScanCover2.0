# ScanCover OpenXR Reserve Decision

Date: 2026-03-14

## Decision

`MetaOpenXROcclusionSubsystem` is retained as a validated reserve and future-facing upstream option.

Current `ScanCover` production-facing depth consumption returns to:

- `Meta.XR.EnvironmentDepth.EnvironmentDepthManager`
- `_EnvironmentDepthTexture`
- `_EnvironmentDepthReprojectionMatrices`
- `_EnvironmentDepthZBufferParams`

This means the active architecture is:

`MetaOpenXROcclusionSubsystem`
-> `DepthProviderOpenXR`
-> `EnvironmentDepthManager`
-> existing MetaXR/ScanCover depth consumers

## Why This Decision Was Made

1. The OpenXR bridge has been validated.

- `ScanCoverOpenXROcclusionContext` successfully resolved `MetaOpenXROcclusionSubsystem`.
- `ScanCoverOpenXRDepthFrameSource` confirmed that even when direct `TryGetFrame(...)` diagnostics were unstable, `EnvironmentDepthManager` was still publishing `_EnvironmentDepthTexture`.

2. The direct OpenXR path is not yet the best mainline for ScanCover.

- It increases lifecycle and timing complexity.
- It is not required for the current ScanCover objective.
- It does not by itself solve the current "large rift" issue in point-cloud generation.

3. The current problem is downstream, not upstream.

The visible "rift" is mainly caused by the current consumer logic in:

- `Scripts/MetaXR/07/DepthGridPointCloud.cs`
- `Scripts/MetaXR/07/CustomEnvironmentDepthRaycaster.cs`

Specifically:

- left/right eye reconstruction is performed separately
- the results are compared by agreement thresholds
- disagreement can cause point rejection or mono fallback

Changing the upstream source alone does not remove that behavior.

## Engineering Position

For the current phase:

- `EnvironmentDepthManager` is the official consumption layer
- OpenXR scripts under `Scripts/OpenXR/02` are diagnostics and reserve assets
- they must not become the main ScanCover depth owner

## What Remains Valuable About OpenXR

`MetaOpenXROcclusionSubsystem` remains strategically important because it:

- confirms the true upstream source can already be OpenXR
- reduces dependence on the old opaque provider path
- preserves a future route toward custom preprocessing and custom surface reconstruction

But this value is currently architectural, not product-critical.

## Immediate Practical Rule

Use `EnvironmentDepthManager` for:

- depth texture publication
- reprojection matrices
- z-buffer parameters
- current ScanCover depth consumption

Use `OpenXR/02` scripts only for:

- subsystem verification
- bridge verification
- future migration reference

## Next Recommended Work

If ScanCover continues in the MetaXR consumption architecture, the next high-value work is:

1. Reduce or replace the dual-eye disagreement logic that creates the large visible rift.
2. Move away from the current `CustomEnvironmentDepthRaycaster` plus hard agreement rejection path.
3. Build a direct depth-texture preprocessing layer before point-cloud or surface observation generation.

## Non-Goal For Now

Do not force direct `TryGetFrame(...)` to become the current mainline requirement.

That issue only becomes critical when ScanCover fully detaches from `EnvironmentDepthManager`.
