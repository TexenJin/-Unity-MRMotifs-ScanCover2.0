# InfiniTAM-style baseline — V1.3 behaviour boundary restored

## Active production contract

The active route has deliberately returned to the last boundary at which mesh
production and model prediction were separate:

1. each distinct Quest platform depth frame keeps its Quest world pose;
2. the frame writes the sole raw-projective TSDF, subject only to the existing
   input-health and hard-motion gates;
3. dirty blocks are extracted and published after the first real integration;
4. model raycast, zero-crossing selection and residual statistics are read-only;
5. raycast cannot retain a frame, solve or apply a pose correction, veto TSDF
   fusion, clear the model, require bootstrap confirmation, or hide the mesh.

The V1.6-V1.10 tracker code is retained only as dormant historical reference
behind `enableInfiniTamTrackingAuthority=false`. It is not a production fallback
and must not be re-enabled piecemeal. Tracking authority returns only after the
complete TAM unit (pyramid, per-level regime, robust correspondence, damped
rollback, convergence, GOOD/POOR/FAILED state and relocalisation) is ported.

This is an independent reconstruction baseline, not another GunGel/court mode.
It is enabled by `VolumeIntegrator.enableInfiniTamBaseline` and owns the full
production route while enabled.

## Active route

1. Quest depth preprocessing supplies the current depth, intrinsics and the
   matching pose.
2. A compact raw-projective TSDF weighted average writes the sole production
   TSDF volume. One accepted observation is one complete vote and the canonical
   accumulation weight is capped at 100.
3. A baseline-private `R16_SFloat` vote volume (with `R32_SFloat` fallback)
   stores that accumulation weight.
   The TSDF `.g` lane now only says "observed / immediately extractable", so a
   first surface still appears quickly without giving every later frame roughly
   one sixth of the authority over a mature zero crossing.
4. A baseline-private dirty owner/boundary ledger schedules only TSDF blocks
   whose zero crossing can have changed. The ledger is metadata and has no
   authority over TSDF values.
5. Each active block uses a stateless GPU Surface Nets worker. Its last complete
   immutable front remains visible while a replacement is extracted and read
   back; a completed replacement is published atomically.
6. A whole-volume immutable front remains the hand-over fallback until the
   first dirty-ledger pass and every affected block have completed. It is then
   released, so the steady-state route does not run two mesh factories.
7. The model raycaster predicts from the same sole TSDF for receipts and
   diagnostics only. It cannot retain an observation, alter its pose, stop a
   write, clear the TSDF, or delay publication.

## Explicitly bypassed

- GunGel evidence, admission and pose correction
- final-court planes and scene-ruler/seed-plane admission
- projective A/B shadow volume
- carve, prune, freeze and frozen-block supervision
- candidate history and temporal mesh smoothing
- productizer, quality-gate retention, 10 cm topology and HERA
- instantaneous-shell authority (the shell remains observation-only)

The private vote volume is allocated at full size only while the baseline is
enabled. The old route receives a 1x1 descriptor placeholder for Vulkan binding
validity and never reads or writes the baseline votes.

## Deliberate restored-baseline compromises

- The existing Surface Nets kernel is reused only as a stateless block
  extractor. It is not yet the canonical InfiniTAM Marching Cubes extractor.
- Dirty block scheduling and stable-front publication are present. Model
  raycast prediction is read-only in the active V1.3 contract; there is no
  production ICP correction, tracking gate, lost/recovery probation, or
  keyframe/submap relocalisation yet.
- Baseline vote persistence across saved TSDF packages is not implemented yet;
  V1.1 acceptance is for a fresh live scan.
- Device-visible behaviour is not validated until the user builds and runs it
  on Quest. Static source checks do not prove runtime shader or GPU behaviour.

## Dormant historical V1.8-V1.10 experiments

Everything in this section is retained for forensic comparison only. None of
it has TSDF-write, pose-correction, mesh-publication or model-clear authority
while `enableInfiniTamTrackingAuthority=false`. These stages must not be
reactivated separately; they are evidence for the later complete TAM port.

### V1.8 bootstrap and lost-tracking boundary

- A Unity render loop may see one Quest depth texture many times. Bootstrap
  now keys evidence by `CurrentPlatformFrame`, so a persistent texture can
  contribute at most one seed observation.
- Bootstrap requires a configurable run of low-angular-speed and
  low-linear-speed depth frames. The normal scan motion gate is intentionally
  not reused because its 90 degree/s emergency threshold is too permissive for
  establishing the ICP reference model.
- The raw-pose seed count only makes the TSDF raycastable. It does not grant
  mesh publication authority. Consecutive exact-frame tracking acceptances are
  required before either the whole-volume fallback or block fronts may render.
- A rejected but unconfirmed seed is automatically discarded after a bounded
  streak. This is the local equivalent of abandoning a bad initial map; it is
  not allowed to erase a model that has already passed publication.
- After publication a tracking failure freezes fusion and keeps the last good
  mesh visible. Keyframe relocalisation/submap recovery remains the next
  InfiniTAM parity stage; V1.8 does not pretend that automatic reseeding is a
  replacement for mature-map relocalisation.

### V1.9 Quest-pose-assisted cadence and recovery boundary

- Quest supplies a world-tracked pose for every retained depth frame. The
  production tracker therefore uses that pose as its strong prior and performs
  only a medium and a fine point-to-plane refinement (`4,2` at the default
  pixel stride), instead of paying five serial GPU-to-CPU readback round trips
  intended for a camera with no external pose.
- Both passes still re-raycast the sole TSDF and the retained frame cannot fuse
  before the final quality verdict. This is a throughput correction, not a
  raw-pose fallback.

### V1.10 pose-prior observability boundary

- A single planar wall or ceiling does not make all six rigid-pose axes
  observable to point-to-plane ICP. Requiring rank six therefore caused the
  provisional model to reject every verification frame, reseed, and appear to
  hang even when the Quest world pose and depth agreed.
- The retained Quest world pose is the strong prior. ICP may now validate the
  three plane-observable directions; damping leaves the unobservable correction
  axes at the Quest prior. This does not grant a raw-depth fallback: minimum
  correspondence count, residual/inlier support, total translation and total
  rotation limits still have to pass before the exact frame can fuse.

This V1.10 rank relaxation was not accepted as a production fix. The backward
audit found that the systemic fault began earlier: an incomplete V1.6 tracker
had been promoted to the sole fusion and publication gate. The active V1.3
contract therefore withdraws that authority instead of tuning rank by scene.

- Once a published model loses tracking, one accidental good correspondence
  set cannot reopen the TSDF writer. A configurable run of consecutive accepted
  frames is required; the last committed mesh remains visible during this
  re-lock probation and none of those probation frames is fused.

Run the source-only architecture check with:

```powershell
& .\Tools\TestInfiniTamBaselineArchitecture.ps1
```

## V1.1 sampled receipt (retained in V1.2)

The baseline emits a diagnostic-only receipt at most once per second. It counts
near-zero-surface writes as first, continuing, or mature observations, reports
their average canonical vote age and the implied authority of one new frame,
and records attempted/fused/stopped frame totals. The GPU counters are read only
after the production dispatch and are never sampled by fusion or extraction.
They appear as one compact HUD line and under `infinitam_baseline_ticket` in the
existing ScanCoverDiagnostics report.

## V1.2 extraction contract

- Dirty tracking records owner and six boundary epochs during the same baseline
  integration dispatch that writes the sole TSDF.
- A block owns a half-open core cell range and reads a halo only for topology;
  neighbouring blocks cannot both own the same emitted cell.
- Live shell depth is deliberately passed as unavailable to block extraction.
  It can be observed independently but cannot become a second mesh authority.
- No smoothing, temporal candidate archive, GunGel, court, productizer,
  destructive quality gate, 10 cm rewrite or HERA stage is instantiated here.
- An asynchronous result whose epoch is already obsolete is discarded before
  publication. The previous front remains displayed until the latest complete
  replacement exists.

## V1.3 model-prediction contract

- Prediction reads the sole production TSDF immediately before the current
  accepted frame is integrated. Therefore a bad frame cannot erase its own
  disagreement before the receipt is measured.
- The ray uses the accepted depth frame's projection, inverse projection, view
  and inverse-view matrices. It does not sample a later headset pose.
- A model hit requires a positive-to-negative TSDF zero crossing. Trilinear
  sampling is valid only when all eight supporting voxels are observed, so
  unknown space cannot manufacture a false surface.
- Ray steps use the TSDF magnitude with a half-voxel lower bound. This follows
  the mature InfiniTAM/KinectFusion model-rendering principle without yet
  granting the predicted surface tracking authority.
- The HUD line reports model hit percentage, the percentage of comparable rays
  within 3 cm, mean/peak absolute residual, front/behind disagreement counts
  and model misses. A miss is split into outside-volume, absent eight-voxel
  support, and supported TSDF without a valid zero crossing, so the next stage
  does not confuse missing model data with geometric disagreement. The full export appears under
  `infinitam_model_raycast_ticket`.
- Model depth and residual textures remain available to the dormant V1.6
  experiment; the prediction shader itself remains unable to write integration,
  mesh extraction or rendering state.

## Dormant historical V1.6-V1.7 experiments

The following tracker and fixed pyramid schedule are retained for backward
audit only. They remain behind `enableInfiniTamTrackingAuthority=false` and are
not part of the active V1.3 production route.

### V1.6 track-before-fuse contract

- The tracker receives an immutable copy of the exact candidate depth, normal,
  dilation/reason textures and all stereo matrices. While its asynchronous GPU
  readback is pending, fusion waits; no newer live frame can replace the
  retained observation.
- Prediction reads the sole TSDF and cannot modify it. Only the integration
  dispatch following an accepted decision can write the model.
- A decision is accepted only when it has enough correspondences and rank, its
  translation/rotation is within the configured trust region, it measurably
  reduces mean residual, and the corrected result retains sufficient 3 cm
  support.
- Rejection means no TSDF write for that observation. There is deliberately no
  raw-pose escape route after bootstrap.
- The HUD tracking gate counts one attempt per retained source frame. Async
  waiting polls are reported separately as busy/abstained work and no longer
  inflate the stopped-frame total.
- This closes the first predict-track-gate-fuse loop. V1.7 extends its solver;
  full parity still requires persistent pose prediction, tracking-loss
  recovery, relocalisation and device validation.

### V1.7 coarse-to-fine tracking contract (historical V1.7 schedule)

- One retained source frame runs the fixed production schedule
  `8,8,4,4,2`: two coarse passes, two medium passes and one fine pass.
- Every pass raycasts the sole TSDF again using the pose corrected by the
  preceding pass. Correspondences are therefore regenerated; the solver never
  applies five updates to one stale pair set.
- The final quality decision reports the accumulated rigid correction, the
  first pass's pre-correction residual and the final pass's post-correction
  residual. This prevents a small final convergence step from hiding the
  improvement produced by earlier levels.
- Effective rank is estimated from the normalized full 6x6 normal equation,
  not by counting non-zero diagonal entries. A single plane therefore cannot
  pretend to constrain all six degrees of freedom.
- Hitting a per-pass translation or rotation trust boundary stops the pyramid
  immediately and returns rejection evidence. It cannot use later levels to
  walk through an unsafe correction.
- The final raycast statistics and tracking pairs complete one shared
  transaction before the retained frame is released to the fusion gate.
