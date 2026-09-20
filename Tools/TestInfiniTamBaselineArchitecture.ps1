$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$volumePath = Join-Path $repo 'Assets\QuestRoomScanStandalone\Core\VolumeIntegrator.cs'
$meshPath = Join-Path $repo 'Assets\QuestRoomScanStandalone\Core\MeshExtractor.cs'
$blockPath = Join-Path $repo 'Assets\QuestRoomScanStandalone\Core\InfiniTamBlockMeshPipeline.cs'
$raycastPath = Join-Path $repo 'Assets\QuestRoomScanStandalone\Core\InfiniTamModelRaycastAudit.cs'
$scannerPath = Join-Path $repo 'Assets\QuestRoomScanStandalone\Core\StandaloneRoomScanner.cs'
$shaderPath = Join-Path $repo 'Assets\QuestRoomScanStandalone\Shaders\VolumeIntegration.compute'
$raycastShaderPath = Join-Path $repo 'Assets\QuestRoomScanStandalone\Resources\InfiniTamModelRaycast.compute'

$volume = Get-Content -LiteralPath $volumePath -Raw
$mesh = Get-Content -LiteralPath $meshPath -Raw
$block = Get-Content -LiteralPath $blockPath -Raw
$raycast = Get-Content -LiteralPath $raycastPath -Raw
$scanner = Get-Content -LiteralPath $scannerPath -Raw
$shader = Get-Content -LiteralPath $shaderPath -Raw
$raycastShader = Get-Content -LiteralPath $raycastShaderPath -Raw

function Assert-Contains([string]$Text, [string]$Pattern, [string]$Message) {
    if ($Text -notmatch $Pattern) { throw "FAIL: $Message" }
    Write-Host "PASS: $Message"
}

Assert-Contains $volume 'private bool enableInfiniTamBaseline = true;' `
    'baseline is enabled by default'
Assert-Contains $volume 'private bool enableInfiniTamTrackingAuthority = false;' `
    'V1.6-V1.10 tracking authority is dormant in the restored V1.3 baseline'
Assert-Contains $volume '!enableInfiniTamTrackingAuthority[\s\S]*?IntegrationCount > 0' `
    'V1.3 mesh publication waits only for the first real TSDF integration'
Assert-Contains $volume 'infinitam_v13_external_pose' `
    'V1.3 production consumes the unique Quest-pose depth frame directly'
Assert-Contains $volume 'infiniTamStartupActive = enableInfiniTamBaseline &&[\s\S]*?enableInfiniTamTrackingAuthority' `
    'bootstrap verification cannot run while V1.3 owns production'
Assert-Contains $volume 'infiniTamTrackingRequired = enableInfiniTamBaseline &&[\s\S]*?enableInfiniTamTrackingAuthority' `
    'the dormant ICP path cannot gate V1.3 TSDF writes'
Assert-Contains $volume 'enableGunGelGuardedFusionExperiment = !enableInfiniTamBaseline;' `
    'GunGel guarded fusion is mutually exclusive with the baseline'
Assert-Contains $volume 'if \(enableProjectiveShadow && !enableInfiniTamBaseline\)' `
    'the A/B shadow volume cannot become a second baseline TSDF'
Assert-Contains $volume 'SetFloat\(UseRawProjectiveSdfID, enableInfiniTamBaseline \? 1f : 0f\)' `
    'the sole production TSDF selects raw projective distance'
Assert-Contains $volume 'SetFloat\(InfiniTamBaselineID, enableInfiniTamBaseline \? 1f : 0f\)' `
    'the compact baseline shader branch is selected explicitly'
Assert-Contains $volume 'if \(enableInfiniTamBaseline\) return;\s+if \(!enableGunGelEvidenceShadow' `
    'GunGel evidence resources are not created in baseline mode'
Assert-Contains $volume 'FrozenBlockReady => !enableInfiniTamBaseline' `
    'freeze supervision cannot acquire baseline voxels'
Assert-Contains $volume 'voteWidth = enableInfiniTamBaseline \? voxelCount\.x : 1' `
    'the full vote sidecar is allocated only for the isolated baseline'
Assert-Contains $volume '"ClearInfiniTamVotes"' `
    'the baseline vote sidecar has an independent clear path'
Assert-Contains $volume '_integrateKernel\.Set\(InfiniTamVoteWeightRWID, _infiniTamVoteWeightVolume\)' `
    'the baseline-private vote sidecar is bound directly to integration'
Assert-Contains $volume '_integrateKernel\.Set\(InfiniTamTicketStatsID, _infiniTamTicketStats\)' `
    'the baseline receipt buffer is always descriptor-bound'
Assert-Contains $volume 'PrepareInfiniTamTicketSample\(\)[\s\S]*?_integrateKernel\.DispatchFit[\s\S]*?RequestInfiniTamTicketReadback\(\)' `
    'the receipt samples the production dispatch and reads back afterwards'
Assert-Contains $volume 'infinitam_baseline_ticket:' `
    'the receipt is included in the existing diagnostic export'
Assert-Contains $volume '_infiniTamModelRaycast\.Dispatch\([\s\S]*?fusionDepth, fusionProjection, fusionView[\s\S]*?minMeshWeight,[\s\S]*?maxUpdateDist, acceptedSourceFrame\);[\s\S]*?BeginDirtyEpoch\(\);' `
    'the old TSDF model is predicted at the exact accepted pose before the current write'
Assert-Contains $volume '_infiniTamModelRaycast\?\.AppendReport\(sb\)' `
    'the model residual receipt is included in the existing diagnostic export'
Assert-Contains $volume 'Baseline[\s\S]*?BeginDirtyEpoch\(\);[\s\S]*?SetFloat\(UseRawProjectiveSdfID' `
    'baseline fusion advances extraction-only dirty epochs without changing its TSDF branch'

Assert-Contains $shader 'if \(gsInfiniTamBaseline > 0\.5\)' `
    'the shader has an explicit baseline branch'
Assert-Contains $shader 'reconstructedTsdf = previousVoteWeight > 0\.0' `
    'baseline integration uses a projective TSDF weighted average'
Assert-Contains $shader 'const float maxVoteWeight = 100\.0' `
    'baseline maturity is a canonical bounded vote count'
Assert-Contains $shader 'gsInfiniTamVoteWeightRW\[coord\] = reconstructedVoteWeight' `
    'baseline maturity no longer reuses the legacy TSDF confidence lane'
Assert-Contains $shader 'gsInfiniTamTicketEnabled > 0\.5[\s\S]*?abs\(observedTsdf\) <= 0\.5' `
    'the receipt is throttled and restricted to the visible zero-surface band'
Assert-Contains $shader '_InfiniTamTicketStats\[0\]' `
    'the receipt counts first writes separately'
Assert-Contains $shader '_InfiniTamTicketStats\[2\]' `
    'the receipt counts mature writes separately'
Assert-Contains $shader 'extractionEligibility = max\(gsFormalSurfaceWeight \+ 1\.0 / 127\.0, 0\.1\)' `
    'one accepted baseline observation crosses the mesh confidence threshold'
Assert-Contains $shader 'WriteTrackedTsdf\(coord, float2\(reconstructedTsdf, extractionEligibility\)\);[\s\S]*?gsColorVolumeRW\[coord\][\s\S]*?return;\s+\}\s+bool fovLedgerNearSurface' `
    'baseline exits before experimental admission and correction rules'

Assert-Contains $mesh 'if \(_volume != null && _volume\.InfiniTamBaselineEnabled\)\s+\{\s+EnsureInfiniTamBaselineResources\(\);\s+return;' `
    'baseline initialization bypasses support/coarse/product/chunk pipelines'
Assert-Contains $mesh 'CandidateHistoryUpdateEnabled = false' `
    'baseline extraction has no candidate history'
Assert-Contains $mesh 'SmoothIterations = 0' `
    'baseline extraction has no smoothing authority'
Assert-Contains $mesh 'FoundationTopologyMode = false' `
    'baseline extraction has no 10 cm productizer'
Assert-Contains $mesh 'VisualQualityDiagnosticsEnabled = false' `
    'baseline extraction has no quality-gate geometry authority'
Assert-Contains $mesh 'renderProductionMesh = true;' `
    'the sole baseline renderer cannot inherit a hidden A/B state'
Assert-Contains $mesh 'if \(IsInfiniTamBaselineActive\)[\s\S]*?ExtractLegacyGlobal\(\);\s+return;' `
    'whole-volume extraction remains only as the baseline hand-over fallback'
Assert-Contains $mesh 'if \(_counterReadbackPending\) return;' `
    'baseline does not overwrite working buffers while publication is in flight'
Assert-Contains $mesh 'PublishInfiniTamBaselineFront\(snapshot\);' `
    'baseline publishes an immutable whole-volume front after exact count readback'
Assert-Contains $mesh '_gpuRenderer\.SetMeshSource\(_infiniTamFront\);' `
    'baseline uses the CPU-known direct-draw snapshot path on Quest'
Assert-Contains $mesh 'UseJointDiagnosticDisplay = false;' `
    'baseline cannot inherit the class-zero fragment discard policy'
Assert-Contains $mesh 'if \(IsInfiniTamBaselineActive\)[\s\S]*?EnsureInfiniTamBaselineResources\(\);[\s\S]*?return;[\s\S]*?if \(_gpuSurfaceNets == null\)' `
    'legacy extraction support cannot replace the baseline renderer source or policy'
Assert-Contains $mesh '_infiniTamBlocks\?\.Tick\(\);' `
    'baseline extraction services its private dirty-block scheduler'
Assert-Contains $mesh '_infiniTamBlocks\.InitialBuildComplete[\s\S]*?ReleaseInfiniTamGlobalFallback\(\);' `
    'whole-volume fallback is released only after the initial block front is complete'
Assert-Contains $mesh 'private void ExtractInfiniTamBlock[\s\S]*?null,\s+null,\s+false\);' `
    'block extraction consumes only the fused TSDF and not the current shell'

Assert-Contains $block 'DirtyChunkEpochs' `
    'the private block route reads owner dirty epochs'
Assert-Contains $block 'DirtyBoundaryEpochs' `
    'the private block route propagates dirty work across shared boundaries'
Assert-Contains $block 'CandidateHistoryUpdateEnabled = false' `
    'block workers keep candidate history disabled'
Assert-Contains $block 'SmoothIterations = 0' `
    'block workers keep post-mesh smoothing disabled'
Assert-Contains $block 'FoundationTopologyMode = false' `
    'block workers do not instantiate the old product topology'
Assert-Contains $block 'CopyCurrentMeshTo\(block\.Back, vertices, indices\)[\s\S]*?SetMeshSource\(block\.Front\)' `
    'a complete immutable block is copied before the renderer swaps fronts'
Assert-Contains $block 'candidateEpoch < block\.TargetEpoch' `
    'obsolete asynchronous block candidates cannot overwrite a newer target'
Assert-Contains $block 'readback timed out; requeued' `
    'a lost Quest GPU callback cannot permanently strand one block'
if ($block -match 'PersistentChunkMeshPipeline|SurfaceProductizer|ChunkQualityGate|GunGelEvidenceShadow|HeraHierarchicalReplay') {
    throw 'FAIL: isolated baseline block route references an old policy pipeline'
}
Write-Host 'PASS: isolated baseline block route references no old policy pipeline'

Assert-Contains $raycast 'tsdf_binding=read_only;prediction_textures=diagnostic_only;tracking_decision=quality_gated_by_volume_integrator' `
    'model prediction keeps TSDF read-only while exposing a gated tracking decision'
Assert-Contains $raycast 'Set\(ModelTsdfID, soleTsdf\)' `
    'model prediction reads the sole TSDF directly'
Assert-Contains $raycast 'Set\(ModelVoteWeightID, soleVoteWeight\)' `
    'model residuals are joined to the sole baseline vote history'
Assert-Contains $raycast 'SetMatrixArray\(DepthCapture\.ViewID, view\)' `
    'model prediction consumes the accepted-frame view matrix'
Assert-Contains $raycast 'SetMatrixArray\(DepthCapture\.ProjID, projection\)' `
    'model prediction consumes the accepted-frame projection matrix'
Assert-Contains $raycast 'AsyncGPUReadback\.Request\(_stats' `
    'model residual statistics use throttled asynchronous readback'
Assert-Contains $raycast 'AsyncGPUReadback\.Request\(_trackingPairs' `
    'frame-to-model correspondences use throttled asynchronous readback'
Assert-Contains $raycast 'SolveTrackingCorrection[\s\S]*?normalEquation[\s\S]*?trackingMeanAfterMm' `
    'tracking solves and measures one rigid point-to-plane correction'
Assert-Contains $raycast 'ProductionStrideDivisors = \{ 2, 4 \}' `
    'Quest-pose-assisted production tracking avoids five serial readback round trips'
Assert-Contains $raycast 'ApplyCorrection\(state\.View, state\.ViewInverse,[\s\S]*?DispatchProductionPass\(state\)' `
    'every pyramid pass re-raycasts after applying the preceding pose correction'
if ($raycast -match 'if \(state\.AnyTranslationClamped \|\| state\.AnyRotationClamped\)') {
    throw 'FAIL: an intermediate trust-region clamp still aborts the pyramid schedule'
}
Write-Host 'PASS: an intermediate trust-region clamp limits one step without aborting later pyramid passes'
Assert-Contains $raycast 'state\.AccumulatedCorrection = incremental\.Correction \*[\s\S]*?BuildAggregateDecision' `
    'the fusion gate receives the accumulated correction rather than only the final small step'
Assert-Contains $raycast 'TrackingDecision\.Underdetermined\(sourceFrame, count' `
    'insufficient correspondences remain a readable tracking result for explicit gate attribution'
Assert-Contains $raycast 'InitialMeanBeforeMm[\s\S]*?FinalMeanAfterMm[\s\S]*?new TrackingDecision' `
    'tracking quality compares the first residual with the final pyramid residual'
Assert-Contains $raycast 'EstimateNormalizedRank\(normalEquation\)' `
    'tracking rank uses the normalized full normal equation rather than diagonal occupancy'
Assert-Contains $raycast 'FinalPairsDone[\s\S]*?FinalStatsDone[\s\S]*?CompleteProductionTracking' `
    'the final pyramid pass publishes one complete pairs-and-stats transaction'
Assert-Contains $raycast 'TryDispatchProductionTracking[\s\S]*?_productionTrackingPending[\s\S]*?AsyncGPUReadback\.Request\(_trackingPairs' `
    'production tracking owns one exact asynchronous frame transaction at a time'
Assert-Contains $raycast 'if \(_readbackPending \|\| _trackingReadbackPending \|\|[\s\S]*?_productionTrackingPending\) return;' `
    'diagnostic dispatch cannot tear a tracking or statistics readback buffer'
Assert-Contains $volume 'QueueInfiniTamTrackedFrame[\s\S]*?TryDispatchProductionTracking' `
    'the baseline retains the exact frame before requesting its model correction'
Assert-Contains $volume 'if \(dispatched\)[\s\S]*?_infiniTamAttemptedFrames\+\+' `
    'one retained source frame counts as one tracking attempt rather than one count per async poll'
Assert-Contains $volume 'AcceptInfiniTamTrackedFrame[\s\S]*?CorrespondenceCount[\s\S]*?EffectiveRank[\s\S]*?MeanBeforeMm - decision\.MeanAfterMm' `
    'production fusion has correspondence, rank and residual-improvement gates'
$acceptMethod = [regex]::Match($volume,
    'private bool AcceptInfiniTamTrackedFrame[\s\S]*?private static void ApplyInfiniTamTrackingCorrection').Value
if ($acceptMethod -match 'decision\.TranslationClamped|decision\.RotationClamped') {
    throw 'FAIL: the final InfiniTAM frame gate still treats a bounded intermediate step as a rejection'
}
Write-Host 'PASS: the final InfiniTAM frame gate judges accumulated correction rather than intermediate step limits'
Assert-Contains $volume 'ApplyInfiniTamTrackingCorrection[\s\S]*?correction \* frame\.ViewInverse' `
    'an accepted correction updates the retained frame pose before fusion'
Assert-Contains $volume 'if \(infiniTamFrame\.Pending\) return false;[\s\S]*?if \(!infiniTamFrame\.Ready\)[\s\S]*?return false;[\s\S]*?ApplyInfiniTamTrackingCorrection' `
    'the exact frame cannot write TSDF before its tracking decision is ready'
Assert-Contains $volume 'infinitam_tracking_reject:' `
    'a rejected tracking decision records a stop instead of falling back to raw fusion'
Assert-Contains $volume 'dc\.CurrentPlatformFrame ==[\s\S]*?_infiniTamLastStartupPlatformFrame[\s\S]*?return false' `
    'bootstrap cannot count one persistent Quest depth texture as several independent seed frames'
Assert-Contains $volume 'IsInfiniTamBootstrapMotionSafe[\s\S]*?infinitam_bootstrap_motion_hold' `
    'bootstrap has a stricter angular and linear stillness gate before any raw write'
Assert-Contains $volume '_infiniTamBootstrapConfirmedFrames >=[\s\S]*?_infiniTamTrackingInitialised = true' `
    'the seed model needs consecutive tracked confirmations before publication'
Assert-Contains $volume 'recoveringPublishedModel[\s\S]*?_infiniTamRecoveryConfirmedFrames <[\s\S]*?infinitam_recovery_probation[\s\S]*?ReleaseInfiniTamDeferredFrame\(\)[\s\S]*?return false' `
    'a published model must regain consecutive tracking confirmations before fusion resumes'
Assert-Contains $volume 'TryReseedUnconfirmedInfiniTam[\s\S]*?ClearInternal\(preserveGunGelEvidence: false\)' `
    'the dormant historical tracker still retains its isolated reseed transaction'
Assert-Contains $volume 'public bool InfiniTamMeshPublicationReady[\s\S]*?_infiniTamTrackingInitialised' `
    'the reconstruction owner keeps the historical publication boundary isolated behind authority'
Assert-Contains $mesh 'if \(!InfiniTamPublicationReady\)[\s\S]*?DisposeInfiniTamBaselineFronts\(\)[\s\S]*?return;' `
    'mesh visibility follows the reconstruction owner publication contract'
Assert-Contains $mesh '_gpuRenderer\.RenderVisible = renderProductionMesh &&[\s\S]*?InfiniTamPublicationReady' `
    'V1.3 publishes after its first integration while dormant history keeps its own boundary'
if ($raycast -match 'GunGel|FinalCourt|SurfaceProductizer|PersistentChunkMeshPipeline|ChunkQualityGate') {
    throw 'FAIL: model raycast audit references an old policy pipeline'
}
Write-Host 'PASS: model raycast audit references no old policy pipeline'

Assert-Contains $raycastShader 'Texture3D<float2> gsModelTsdf;' `
    'raycast shader receives only a read-only TSDF binding'
if ($raycastShader -match 'RWTexture3D') {
    throw 'FAIL: read-only model raycast shader declares a writable 3D volume'
}
Write-Host 'PASS: model raycast shader cannot write a 3D volume'
Assert-Contains $raycastShader 'previousTsdf > 0\.0 && value <= 0\.0' `
    'raycast accepts only a positive-to-negative TSDF zero crossing'
Assert-Contains $raycastShader 'abs\(value\) \* gsVoxDist' `
    'raycast uses TSDF magnitude for InfiniTAM-style adaptive steps'
Assert-Contains $raycastShader 'o000 && o100 && o010 && o110' `
    'trilinear prediction refuses to interpolate through unknown voxel support'
Assert-Contains $raycastShader 'modelRange - observedRange' `
    'the signed model-to-observation residual is explicit'
Assert-Contains $raycastShader 'sawObservedSupport \? 19u : 18u' `
    'model misses distinguish absent TSDF support from absent zero crossings'
Assert-Contains $raycastShader 'rawDepth > 0\.0 && rawDepth < 1\.0' `
    'the ruler rejects depth sentinels outside the projection domain'
Assert-Contains $raycastShader 'observedRange < gsModelMinObservedRange[\s\S]*?observedRange > gsModelMaxObservedRange' `
    'the ruler shares the production near/far measurement gate'
Assert-Contains $raycastShader 'nearestModelRange[\s\S]*?crossingCount > 1u' `
    'first-visible and nearest alternative zero crossings are accounted separately'
Assert-Contains $raycastShader 'modelVoteWeight < 10\.0[\s\S]*?modelVoteWeight < 99\.5' `
    'large residuals are classified by their exact model-surface maturity'
Assert-Contains $raycastShader 'observedPointSupported \? 38u : 39u' `
    'large residuals distinguish supported replacement depth from an unseeded hole'
Assert-Contains $raycastShader 'RWStructuredBuffer<ModelTrackingPair> gsModelTrackingPairs' `
    'raycast exports read-only frame-to-model tracking pairs'

Assert-Contains $scanner 'enableFrozenChunkAbExperiment = false;' `
    'scanner startup cannot replace the baseline with freeze/HERA acquisition'
Assert-Contains $scanner 'meshExtractionHz = Mathf\.Min\(meshExtractionHz, 4f\);' `
    'V1 full-volume extraction is capped below the old 12 Hz cadence'
Assert-Contains $scanner '"InfiniTAM V1\.3直融"' `
    'HUD identifies the active V1.3 direct-fusion contract'
Assert-Contains $scanner '顶[\s\S]*?LastVertexCount[\s\S]*?面[\s\S]*?LastIndexCount / 3' `
    'HUD exposes the minimum integration-to-draw breakpoint counters'
Assert-Contains $scanner '_meshExtractor\.LastSubmittedDrawVertexCount' `
    'HUD proves whether the final renderer submitted the direct draw'
Assert-Contains $scanner 'GetInfiniTamTicketCompact\(\)' `
    'HUD exposes the compact baseline receipt'
Assert-Contains $scanner 'GetInfiniTamModelRaycastCompact\(\)' `
    'HUD exposes model hit rate and residuals separately from fusion votes'

Write-Host 'PASS: InfiniTAM V1.3 behaviour boundary is restored; later tracking authority is dormant and raycast remains read-only.'
