$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot

function Read-Source([string]$relativePath) {
    $path = Join-Path $repo $relativePath
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing source: $relativePath"
    }
    return [IO.File]::ReadAllText($path)
}

function Assert-Contains([string]$text, [string]$needle, [string]$label) {
    if (-not $text.Contains($needle)) {
        throw "FAIL $label - missing: $needle"
    }
    Write-Host "PASS $label"
}

function Assert-Before([string]$text, [string]$first, [string]$second, [string]$label) {
    $a = $text.IndexOf($first, [StringComparison]::Ordinal)
    $b = $text.IndexOf($second, [StringComparison]::Ordinal)
    if ($a -lt 0 -or $b -lt 0 -or $a -ge $b) {
        throw "FAIL $label - expected '$first' before '$second'"
    }
    Write-Host "PASS $label"
}

$paired = Read-Source 'Assets/QuestRoomScanStandalone/Core/PairedDepthFrameRecorder.cs'
$session = Read-Source 'Assets/QuestRoomScanStandalone/Core/ScanReplaySessionPackage.cs'
$volume = Read-Source 'Assets/QuestRoomScanStandalone/Core/VolumeIntegrator.cs'
$mesh = Read-Source 'Assets/QuestRoomScanStandalone/Core/MeshExtractor.cs'
$pipeline = Read-Source 'Assets/QuestRoomScanStandalone/Core/PersistentChunkMeshPipeline.cs'
$surface = Read-Source 'Assets/QuestRoomScanStandalone/Core/GPUSurfaceNets.cs'
$shader = Read-Source 'Assets/QuestRoomScanStandalone/Shaders/SurfaceNetsExtract.compute'
$integrationShader = Read-Source 'Assets/QuestRoomScanStandalone/Shaders/VolumeIntegration.compute'
$scanner = Read-Source 'Assets/QuestRoomScanStandalone/Core/StandaloneRoomScanner.cs'
$instantShell = Read-Source 'Assets/QuestRoomScanStandalone/Core/InstantDepthShellOverlay.cs'
$instantShellCompute = Read-Source 'Assets/QuestRoomScanStandalone/Resources/InstantShellReview.compute'
$analyzer = Read-Source 'Tools/AnalyzeFullOutputLedger.ps1'
$responsibilityAnalyzer = Read-Source 'Tools/AnalyzeTsdfResponsibility.py'
$court = Read-Source 'Assets/QuestRoomScanStandalone/Core/RuntimeFinalSurfaceCourt.cs'
$fullChain = Read-Source 'Tools/BuildScanCoverResponsibilityLedger.py'

if (31089 -ne (3691 + 6 * 65 * 65 + 2048)) {
    throw 'FAIL expanded visual counter arithmetic'
}
Write-Host 'PASS expanded visual counter arithmetic'

Assert-Contains $paired 'callbackToPreprocessPositionMm' 'depth manifest contains pose drift'
Assert-Contains $paired 'processedMinusRawRmsMetres' 'depth manifest contains raw-to-processed metric'
Assert-Contains $paired 'TryLinearizeMetres' 'depth statistics use projection linearization'
Assert-Contains $session 'runtime_timeline' 'continuous runtime timeline exists'
Assert-Contains $session 'integration_dispatches.csv' 'fusion-to-dirty-epoch bridge exists'
Assert-Contains $session 'RecordRuntimeFrame();' 'timeline is sampled continuously while active'
Assert-Contains $volume 'RecordIntegrationDispatch(' 'accepted integration records its epoch'
Assert-Before $volume 'IntegrationCount++;' 'RecordIntegrationDispatch(' 'dispatch row uses committed integration number'
Assert-Contains $mesh 'production_paper_replacements.csv' 'production paper replacement events are exported'
Assert-Contains $mesh 'production_paper_quality_timeline.csv' 'every accepted paper commit exports spatial roughness history'
Assert-Contains $pipeline 'RecordLocalQualityBins' 'accepted paper commits retain first and later per-bin quality'
Assert-Contains $court 'frame_gate_summary.csv' 'final court exports frame gate responsibility'
Assert-Contains $court 'independent_testimonies.csv' 'final court exports accepted independent testimony'
Assert-Contains $court 'decision_checks.csv' 'final court exports every rejudge outcome'
Assert-Contains $court 'performanceBoundary' 'court ledger explicitly avoids per-pixel disk logging'
Assert-Contains $mesh 'fusion_fov_periods.csv' 'fusion FOV-period evidence is exported'
Assert-Contains $mesh 'fusion_forensic_ledger.txt' 'fusion forensic counters are exported'
Assert-Contains $mesh 'FlushAllForensicLedgers' 'final production and projective periods are awaited'
Assert-Contains $mesh 'fusion_all_counters.csv' 'all production and projective counters are exported'
Assert-Contains $mesh 'replacement_event_drops=' 'replacement ledger overflow is explicit'
Assert-Contains $session '!contract.Contains("replacement_event_drops=0")' 'replacement overflow makes seal incomplete'
Assert-Contains $volume 'CumulativeProjectiveShadowCarveStats' 'projective shadow counters survive periodic clearing'

$foundationStart = $pipeline.IndexOf('if (_config.FoundationAtomicReplacement)', [StringComparison]::Ordinal)
$foundationEnd = $pipeline.IndexOf('bool spatialDestructive', $foundationStart, [StringComparison]::Ordinal)
if ($foundationStart -lt 0 -or $foundationEnd -lt 0) {
    throw 'FAIL production-paper branch not found'
}
$foundationBlock = $pipeline.Substring($foundationStart, $foundationEnd - $foundationStart)
Assert-Before $foundationBlock 'CaptureVisualQualityPage(chunk, counters);' 'PublishFullCandidate(' 'drawn paper quality is captured before publish'
Assert-Contains $foundationBlock 'chunk.Surface.SmoothIterations = 0;' 'foundation disables post-extraction smoothing'
Assert-Contains $foundationBlock 'chunk.Surface.TemporalAlphaMax = 1f;' 'foundation disables post-extraction temporal blending'
Assert-Contains $mesh 'SetFullOutputAuditActive(true)' 'production shape diagnostics follow the explicit paper-ledger session'
Assert-Contains $pipeline 'observed_platform_frame_at_decision' 'replacement rows distinguish commit context from provenance'
Assert-Contains $mesh 'candidate_epoch -> integration_dispatches.dirtyEpoch' 'paper provenance joins through dirty epoch'
Assert-Contains $pipeline 'head_pitch_deg' 'replacement rows retain full head attitude'

Assert-Contains $shader '#define VISUAL_BOUNDARY_FACE_RESOLUTION 65u' 'shader seam backing supports 64-cube pages'
Assert-Contains $shader '#define STAGE_RESPONSIBILITY_COUNTER_COUNT 2048u' 'shader reserves same-epoch stage ledger'
Assert-Contains $shader '#define COUNTER_COUNT 31089u' 'shader counter count matches seam plus stage ledger'
Assert-Contains $surface 'private const int CounterCount64 = LegacyCounterCount64 + StageResponsibilityCounterCount;' 'CPU 64-page counter count includes stage ledger'
Assert-Contains $surface 'private const int CounterCount32 = LegacyCounterCount32;' '32-page workers avoid production-paper stage overhead'
Assert-Contains $shader '_StageResponsibilityEnabled != 0u' 'stage audit is limited to production-size pages'
Assert-Contains $shader '_CounterCount' 'shader bounds counter clear to actual allocation'
Assert-Contains $shader '[numthreads(256, 1, 1)]' 'expanded counter clear is parallel'
Assert-Contains $surface 'CeilDiv(clearCounterCount, 256)' 'CPU dispatch covers parallel counter clear'
Assert-Contains $pipeline 'private const int VisualBoundaryFaceResolution = 65;' 'CPU seam stride matches shader'
Assert-Contains $pipeline 'int logicalResolution = _config.ChunkSize + 1;' 'seam report reads logical page area only'
Assert-Contains $shader 'crossingSum += q;' 'TSDF zero-crossing moments are retained before cell reduction'
Assert-Contains $shader 'rawSurfaceNetsPosition = posCoord;' 'raw Surface Nets representatives are retained before post transforms'
Assert-Contains $shader 'FitVisualPlaneAt(' 'all responsibility stages use the same plane-fit implementation'
Assert-Contains $pipeline 'stage_responsibility_spatial_csv:' 'paper report exports same-bin stage responsibility'
Assert-Contains $pipeline 'raw_equals_final_by_production_contract=' 'paper report records whether post extraction can be responsible'
Assert-Contains $analyzer 'worst_tsdf_zero_crossing_bins:' 'offline audit surfaces the first malformed TSDF bins'
Assert-Contains $integrationShader 'gsTsdfResponsibilityRW' 'fusion owns a diagnostic-only per-voxel responsibility sidecar'
Assert-Contains $integrationShader 'RecordTsdfGeometryMutation' 'zero-surface-moving writes retain their before/after identity'
Assert-Contains $integrationShader 'RecordTsdfBlockedCorrection' 'strongest blocked correction is retained separately'
Assert-Contains $integrationShader 'RecordTsdfSupportMutation' 'weight growth carve freeze and reset retain independent responsibility'
Assert-Contains $volume 'BeginTsdfResponsibilityCapture' 'responsibility capture starts before first integration'
Assert-Contains $volume 'final_tsdf.rg8_snorm.bin' 'stop-time export includes the final TSDF'
Assert-Contains $volume 'responsibility.rg32_uint.bin' 'stop-time export includes packed transaction provenance'
Assert-Contains $volume 'support_responsibility.r32_uint.bin' 'stop-time export includes support-lifecycle provenance'
Assert-Contains $volume 'const int slabDepth = 1;' 'Quest 3D texture responsibility readback requests exactly one Z slice'
Assert-Contains $integrationShader 'CopyTsdfResponsibilityExportSlice' 'integer 3D responsibility is copied through a structured staging buffer'
Assert-Contains $integrationShader 'CopyTsdfSupportResponsibilityExportSlice' 'support responsibility uses the same Quest-safe staging path'
Assert-Contains $volume 'AsyncGPUReadback.Request(staging)' 'Quest reads integer responsibility from a structured buffer rather than a 3D texture'
Assert-Contains $volume 'each completed resource is written immediately' 'responsibility schema records partial checkpoint persistence'
Assert-Contains $volume 'partial_status.json' 'responsibility export retains completed lanes when a later lane fails'
Assert-Before $volume '"final_tsdf.rg8_snorm.bin"' 'ReadIntegerResponsibilitySlices(' 'final TSDF is persisted before integer responsibility readback begins'
Assert-Before $volume '"responsibility.rg32_uint.bin"' '"support_responsibility"' 'geometry responsibility is persisted before support responsibility begins'
Assert-Contains $session 'tsdf_responsibility' 'full session seals the TSDF responsibility artifact'
Assert-Contains $responsibilityAnalyzer 'endpoint_responsibility.csv' 'offline audit emits endpoint transaction responsibility'
Assert-Contains $responsibilityAnalyzer 'endpoint_source_samples.csv' 'offline audit reprojects exact source depth samples'
Assert-Contains $responsibilityAnalyzer 'paper_hole_responsibility.csv' 'offline audit joins blank paper cells to TSDF and publication causes'
Assert-Contains $responsibilityAnalyzer 'paper_hole_source_samples.csv' 'blank cells reproject their responsible source frames through raw and processed depth'
Assert-Contains $responsibilityAnalyzer 'production_paper_occupancy.r32_uint.bin' 'blank audit consumes exact committed native-5cm paper occupancy'
Assert-Contains $responsibilityAnalyzer 'mature_crossing_without_paper' 'blank audit separates mature TSDF crossings lost after fusion'
Assert-Contains $responsibilityAnalyzer 'raw_crossing_below_weight_without_paper' 'blank audit separates weak TSDF support from later paper loss'
if ($responsibilityAnalyzer.Contains('raise SystemExit("no TSDF-zero-crossing deformation bins found')) {
    throw 'FAIL blank-region audit must still run when no malformed surface bin exists'
}
Write-Host 'PASS blank-region audit does not depend on a malformed surface bin'
Assert-Contains $session 'exact production paper occupancy ledger is missing' 'session seal requires exact production-paper occupancy'
Assert-Contains $analyzer "'production paper occupancy'" 'main audit requires exact production-paper occupancy'
Assert-Contains $analyzer "'production paper occupancy schema'" 'main audit requires occupancy self-description'
Assert-Contains $mesh 'RequestProductionPaperOccupancyAuditExport' 'paper audit does not depend on SupportTruth visibility'
Assert-Contains $instantShell 'paperStrideVoxels\": 1' 'paper occupancy uses the native 5cm cell lattice'
Assert-Contains $instantShell 'rasterizedTriangles != expectedTriangles' 'paper occupancy rejects a truncated committed front'
Assert-Contains $instantShellCompute '_PaperAuditClearOffset' 'native 5cm occupancy clear is split below GPU dispatch limits'
Assert-Contains $analyzer 'tsdf_transaction_responsibility:' 'main audit includes internal TSDF responsibility verdict inputs'

$foundationShaderStart = $shader.IndexOf('// The foundation is a uniform 10 cm topology consumer', [StringComparison]::Ordinal)
$foundationShaderEnd = $shader.IndexOf('// One quad produces two triangles.', $foundationShaderStart, [StringComparison]::Ordinal)
if ($foundationShaderStart -lt 0 -or $foundationShaderEnd -lt 0) {
    throw 'FAIL production-paper shader branch not found'
}
$foundationShaderBlock = $shader.Substring($foundationShaderStart, $foundationShaderEnd - $foundationShaderStart)
Assert-Contains $foundationShaderBlock 'uint visualSpatialBin = MatureSpatialBin(coord);' 'foundation paper maps quality to the same spatial bins'
Assert-Contains $foundationShaderBlock 'AuditVisualTriangle(triangle0.x, triangle0.y, triangle0.z,' 'foundation paper audits its first emitted triangle'
Assert-Contains $foundationShaderBlock 'AuditVisualTriangle(triangle1.x, triangle1.y, triangle1.z,' 'foundation paper audits its second emitted triangle'
Assert-Before $foundationShaderBlock 'AuditVisualTriangle(triangle0.x' 'uint writeIndex = baseIdx;' 'foundation quality is recorded before the branch returns'

Assert-Contains $pipeline 'public void BeginFinalDrain()' 'paper pipeline exposes an explicit final settlement'
Assert-Contains $pipeline '_finalDrainRequestInFlight = _finalDrainActive;' 'final settlement requires a fresh post-A dirty-ledger readback'
Assert-Contains $mesh '_sparseFoundationSkin.Tick();' 'A-key seal advances the stopped paper scheduler'
Assert-Contains $mesh 'visual_quality_pages=' 'paper contract makes pages=0 an explicit failure signal'
Assert-Contains $session 'production paper final drain timed out' 'artifact status reports the actual paper drain failure'
Assert-Contains $session 'production paper visual quality pages are missing' 'artifact status rejects pages=0'
Assert-Contains $session 'same-epoch surface stage responsibility ledger is missing' 'artifact status requires the stage ledger'
Assert-Contains $session 'same-epoch surface stages have no comparable spatial bins' 'artifact status rejects an empty responsibility comparison'

$newScanStart = $scanner.IndexOf('if (!resuming)', [StringComparison]::Ordinal)
$ledgerStart = $scanner.IndexOf('_meshExtractor.BeginLedgerSession();', $newScanStart, [StringComparison]::Ordinal)
$depthStart = $scanner.IndexOf('_depthCapture.StartDepthCapture();', $newScanStart, [StringComparison]::Ordinal)
if ($newScanStart -lt 0 -or $ledgerStart -lt 0 -or $depthStart -lt 0 -or $ledgerStart -ge $depthStart) {
    throw 'FAIL paper ledger must start before depth capture on every new scan'
}
Write-Host 'PASS paper ledger starts before depth capture on every new scan'

$captureStop = $scanner.IndexOf('if (replayPackageFinalizing)', [StringComparison]::Ordinal)
$freezeWrite = $scanner.IndexOf('_chunkAbFrozen = true;', $captureStop, [StringComparison]::Ordinal)
$earlyReturn = $scanner.IndexOf('return;', $captureStop, [StringComparison]::Ordinal)
if ($captureStop -lt 0 -or $earlyReturn -lt 0 -or $freezeWrite -lt 0 -or $earlyReturn -ge $freezeWrite) {
    throw 'FAIL A-key ledger stop must return before frozen replay state'
}
Write-Host 'PASS A-key ledger stop does not enter frozen replay'

Assert-Contains $session '_runtimeLedgerWriteErrors == 0' 'runtime-ledger loss makes seal incomplete'
Assert-Contains $session '(_finalSurfaceCourt?.WriteErrors ?? 0) == 0' 'final-court ledger loss makes seal incomplete'
Assert-Contains $session '\"finalSurfaceCourtWriteErrors\"' 'session status exposes final-court ledger loss'
Assert-Contains $session 'quality_timeline_bin_drops=0' 'paper quality timeline overflow makes artifact incomplete'
Assert-Contains $session 'full_output_index.json' 'session includes one join/index contract'
Assert-Contains $session 'sameEpochSurfaceStageResponsibility' 'session index routes the same-epoch stage ledger'
Assert-Contains $mesh 'join_surface_stages=' 'paper contract defines the in-page first-divergence join'
Assert-Contains $analyzer 'distance_to_chunk_m' 'offline audit groups distance'
Assert-Contains $analyzer 'absolute_pitch_deg' 'offline audit groups pitch'
Assert-Contains $analyzer 'angular_speed_deg_per_s' 'offline audit groups turning speed'
Assert-Contains $analyzer 'paper_candidate_epoch_missing_dispatch' 'offline audit checks causal joins'
Assert-Contains $analyzer 'BuildScanCoverResponsibilityLedger.py' 'main audit builds one joined responsibility ledger'
Assert-Contains $analyzer 'blocked_correction_totals=' 'main audit exposes split correction blockers'
Assert-Contains $fullChain 'surface_site_timeline.csv' 'joined ledger exports surface chronology'
Assert-Contains $fullChain 'firstPublishedPlaneCandidate' 'joined ledger never treats a missing plane fit as zero roughness'
Assert-Contains $fullChain 'became_plane_comparable' 'joined ledger distinguishes newly measurable paper from worsening'
Assert-Contains $fullChain 'court_decision_trace.csv' 'joined ledger exports court trace'
Assert-Contains $fullChain 'courtPublishedLayerSpreadP95Mm' 'joined ledger separates published court movement from testimony scatter'
Assert-Contains $fullChain 'blockedByMotionAuthority' 'joined ledger splits motion authority from FOV authority'
Assert-Contains $fullChain 'blockedByFovAuthority' 'joined ledger retains the independent FOV blocker count'
Assert-Contains $fullChain 'roughness_lineage.csv' 'joined ledger exports first culprit and later amplifiers'
Assert-Contains $fullChain 'factor_responsibility.csv' 'joined ledger compares all suspect factors together'
Assert-Contains $fullChain 'incidence_x_range' 'joined ledger separates angle from distance confounding'
Assert-Contains $fullChain 'incidence_x_motion' 'joined ledger separates angle from motion confounding'

Write-Host 'Full output ledger static contract: PASS'
