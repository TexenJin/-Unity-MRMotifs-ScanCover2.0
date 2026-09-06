$ErrorActionPreference = 'Stop'
$root = Join-Path $PSScriptRoot '../Assets/QuestRoomScanStandalone/Core'
$overlay = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $root 'InstantDepthShellOverlay.cs')
$scanner = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $root 'StandaloneRoomScanner.cs')
$meshExtractor = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $root 'MeshExtractor.cs')
$meshShader = Get-Content -Raw -Encoding UTF8 -LiteralPath (
    Join-Path $PSScriptRoot '../Assets/QuestRoomScanStandalone/Shaders/ScanMeshVertexColor.shader')
$supportShader = Get-Content -Raw -Encoding UTF8 -LiteralPath (
    Join-Path $PSScriptRoot '../Assets/QuestRoomScanStandalone/Shaders/ScanSupportTruth.shader')
$supportRenderer = Get-Content -Raw -Encoding UTF8 -LiteralPath (
    Join-Path $root 'SupportTruthRenderer.cs')
$supportExtract = Get-Content -Raw -Encoding UTF8 -LiteralPath (
    Join-Path $PSScriptRoot '../Assets/QuestRoomScanStandalone/Shaders/SupportTruthExtract.compute')
$chunkPipeline = Get-Content -Raw -Encoding UTF8 -LiteralPath (
    Join-Path $root 'PersistentChunkMeshPipeline.cs')
function Method([string]$source, [string]$signature) {
    $start = $source.IndexOf($signature)
    if ($start -lt 0) { throw "Missing $signature" }
    $open = $source.IndexOf('{', $start)
    $depth = 1; $end = $open + 1
    while ($depth -gt 0 -and $end -lt $source.Length) {
        if ($source[$end] -eq '{') { $depth++ }
        if ($source[$end] -eq '}') { $depth-- }
        $end++
    }
    return $source.Substring($start, $end - $start)
}
$visible = Method $overlay 'public void SetVisible(bool visible)'
$acquiring = Method $overlay 'public void SetAcquiring(bool acquiring)'
$gate = [regex]::Match($overlay, '(?s)bool relayAuditThisBatch = (.*?);').Groups[1].Value
if (!$gate -or $gate.Contains('_visible')) { throw 'Audit depends on display mode' }
if ($visible -match '_nextCaptureAt|ResetRelay|ClearAll') { throw 'View resets acquisition/history' }
$harness = @'
public class HudViewHarness {
 public bool _visible, enabled=true, CompositeWithProduction, _acquiring, _reviewReadbackPending;
 public float _nextCaptureAt=7, _nextReviewReadbackAt=10;
 public void SetRejectedTriangleVisibility(bool value) {}
 public void SetDepthAwareOwnership(bool value) {}
 public void SetAllRenderers(bool value) {}
'@ + $visible + $acquiring + "public bool Audit(bool frozen, float now) { return $gate; } }"
Add-Type -TypeDefinition $harness
$checks = 0
foreach ($show in @($false,$true)) {
    foreach ($capture in @($false,$true)) {
        foreach ($frozen in @($false,$true)) {
            foreach ($pending in @($false,$true)) {
                foreach ($now in @(9,10,11)) {
                    $case = New-Object HudViewHarness
                    $case.SetAcquiring($capture)
                    $case._nextCaptureAt = 7
                    $case.SetVisible($show)
                    $case.SetAcquiring($capture)
                    $case._reviewReadbackPending = $pending
                    $expected = $capture -and !$frozen -and !$pending -and $now -ge 10
                    if ($case.Audit($frozen,$now) -ne $expected) { throw 'Audit gate mismatch' }
                    if ($case._nextCaptureAt -ne 7) { throw 'View switch/repeated state resets capture timer' }
                    $checks++
                }
            }
        }
    }
}
$case = New-Object HudViewHarness
$case.SetAcquiring($true)
if ($case._nextCaptureAt -ne 0) { throw 'Resume no longer schedules capture' }
$setup = Method $scanner 'private System.Collections.IEnumerator CreateStatusBadgeWhenCameraReady()'
$update = Method $scanner 'private void Update()'
if ($setup.Contains('CreateProbeTargetReticle(') -or $update.Contains('UpdateProbeTargetReticle();')) {
    throw 'Retired yellow overlay still active'
}
$hud = Method $scanner 'private void RefreshStatusBadge()'
$body = $hud.Substring($hud.IndexOf('// Fixed semantic columns'))
if ($body -match '\.Visible|_shellPaperOnlyView|relayHud') { throw 'HUD sections gated by view' }
foreach ($token in @('RelayDiagnosticsFixed','AttributionDiagnosticsFixed','probeHud','sealHud','SparseFoundationHudFixed')) {
    if (!$body.Contains($token)) { throw "HUD missing $token" }
}

# The retired colour split must stay absent. Shell->paper reasserts the last
# device-visible native 5 cm source. The unverified correction quarantine must
# not participate in any production shader until visibility is re-established.
foreach ($token in @('_surfaceHealthColorView','viewState = "结构分色"',
                     'SurfaceHealthColorViewID','_RSSurfaceHealthColorView')) {
    if ($scanner.Contains($token) -or $meshShader.Contains($token)) {
        throw "Retired surface-health state returned: $token"
    }
}
$toggle = Method $scanner 'public void ToggleCoverageMarkers()'
$paperBranch = $toggle.Substring($toggle.IndexOf('else if (shellWasComposite)'))
if ($paperBranch.Substring(0, $paperBranch.IndexOf('else if (paperOnlyWasVisible)')).Contains(
        '_meshExtractor.ShowProductionPaperView()')) {
    throw 'Composite-to-paper-only handoff reselects or restarts the already-visible paper source'
}
if (!$meshExtractor.Contains(
        'private RouteValidationView _routeValidationView = RouteValidationView.SparseFoundation;')) {
    throw 'Production paper no longer defaults to the last device-visible native 5 cm route'
}
$showPaper = Method $meshExtractor 'public string ShowProductionPaperView()'
if (!$showPaper.Contains('_routeValidationView = RouteValidationView.SparseFoundation;') -or
    $showPaper.Contains('RouteValidationView.PaperFineHybrid')) {
    throw 'Production-paper restore no longer selects the last device-visible native 5 cm route'
}
$supportUpdate = Method $supportRenderer 'private void Update()'
if (!$supportUpdate.Contains('!_visible')) {
    throw 'Support-truth diagnostic producer no longer matches the known visible baseline'
}
foreach ($shader in @($meshShader,$supportShader)) {
    if ($shader.Contains('PaperCorrectionQuarantine.hlsl') -or
        $shader.Contains('RSPaperCorrectionPointRejected')) {
        throw 'Unverified correction quarantine still participates in production drawing'
    }
}
if ($meshShader.Contains('correctionRejected') -or
    $supportShader.Contains('paperCorrectionRejected') -or
    $supportShader.Contains('triangleBase + cornerIndex')) {
    throw 'Correction visibility state or compiler-crashing hash walk remains in a production shader'
}
if ($update.Contains('RefreshPaperCorrectionMask();') -or
    !$update.Contains('Shader.SetGlobalFloat(PaperCorrectionHideActiveID, 0f);')) {
    throw 'Dormant correction quarantine is still refreshed or can be re-enabled by Update'
}
$displayMode = Method $scanner 'private void ApplyDisplayMode()'
if (!$displayMode.Contains('Shader.SetGlobalInt(PaperCorrectionHashMaskID, -1);') -or
    !$displayMode.Contains('Shader.SetGlobalFloat(PaperCorrectionHideActiveID, 0f);')) {
    throw 'Display-mode switch can re-authorize the unverified correction quarantine'
}
$correctionRefresh = Method $scanner 'private void RefreshPaperCorrectionMask()'
if ($correctionRefresh.Contains('RebuildPaperCorrectionHash();') -or
    !$correctionRefresh.Contains('Shader.SetGlobalInt(PaperCorrectionHashMaskID, -1);') -or
    !$correctionRefresh.Contains('Shader.SetGlobalFloat(PaperCorrectionHideActiveID, 0f);')) {
    throw 'Correction compatibility seam can rebuild or re-authorize the quarantine'
}
if (!$supportRenderer.Contains('Graphics.RenderPrimitivesIndirect(rp, MeshTopology.Triangles,') -or
    !$supportRenderer.Contains('drawTopologyMesh ? _front.TopologyDrawArgs : _front.DrawArgs')) {
    throw 'Restored public paper is not using its established indirect draw path'
}
foreach ($token in @('_SupportTopologyCellGeometry','candidateClosed',
                     'committingCandidate','_SupportTopologyCellGeometry[topologyFlat] = candidateGeometry')) {
    if (!$supportExtract.Contains($token)) {
        throw "Public paper lost cell-local atomic replacement: $token"
    }
}
foreach ($token in @('FoundationAtomicReplacement','ShowCompletedChunksImmediately',
                     'foundation_atomic_replace','CopyCurrentMeshTo')) {
    if (!$chunkPipeline.Contains($token)) {
        throw "Native 5 cm paper lost chunk-atomic publication: $token"
    }
}

foreach ($token in @('自动 壳→TSDF[','自动 因[','自动 补','PushRelayAutoBatch','RecomputeRelayAutoVerdict')) {
    if (-not $overlay.Contains($token)) {
        throw "Missing automatic relay verdict token: $token"
    }
}
$layerUpdate = Method $overlay 'private void Update()'
if ($layerUpdate -match 'Renderer.enabled = true') { throw 'Hidden layer may render' }
if (!$overlay.Contains('_lastRelayResultAt = Time.unscaledTime;') -or
    !$overlay.Contains('_lastRelayResultAt = -1f;')) { throw 'Missing result freshness/reset' }
"PASS: $checks actual C# gate/state combinations, resume, retired UI, fixed HUD sections, hidden renderers and freshness wiring. Device layout/performance not tested."
