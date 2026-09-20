param(
    [Parameter(Mandatory = $true)]
    [string]$SessionDirectory,
    [string]$OutputPath = ''
)

$ErrorActionPreference = 'Stop'
$session = [IO.Path]::GetFullPath($SessionDirectory)
if (-not (Test-Path -LiteralPath $session -PathType Container)) {
    throw "Session directory not found: $session"
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $session 'full_output_audit.txt'
}

$required = [ordered]@{
    'depth manifest' = 'depth_pairs/manifest.csv'
    'fusion manifest' = 'fusion_inputs/manifest.csv'
    'runtime timeline' = 'runtime_timeline/frames.csv'
    'integration dispatches' = 'runtime_timeline/integration_dispatches.csv'
    'paper replacements' = 'artifacts/paper_audit/production_paper_replacements.csv'
    'paper quality timeline' = 'artifacts/paper_audit/production_paper_quality_timeline.csv'
    'paper surface' = 'artifacts/paper_audit/production_paper_surface_ledger.txt'
    'production paper occupancy' = 'artifacts/paper_audit/production_paper_occupancy.r32_uint.bin'
    'production paper occupancy schema' = 'artifacts/paper_audit/production_paper_occupancy_schema.json'
    'fusion counters' = 'artifacts/paper_audit/fusion_all_counters.csv'
    'TSDF responsibility schema' = 'artifacts/tsdf_responsibility/schema.json'
    'TSDF responsibility volume' = 'artifacts/tsdf_responsibility/responsibility.rg32_uint.bin'
    'TSDF support responsibility volume' = 'artifacts/tsdf_responsibility/support_responsibility.r32_uint.bin'
    'final TSDF volume' = 'artifacts/tsdf_responsibility/final_tsdf.rg8_snorm.bin'
    'court frame gates' = 'probe_shadow/final_buffer/final_surface_court/frame_gate_summary.csv'
    'court independent testimonies' = 'probe_shadow/final_buffer/final_surface_court/independent_testimonies.csv'
    'court decision checks' = 'probe_shadow/final_buffer/final_surface_court/decision_checks.csv'
    'court publication events' = 'probe_shadow/final_buffer/final_surface_court/runtime_events.csv'
    'join contract' = 'full_output_index.json'
    'checksums' = 'checksums.sha256'
}

$missing = [Collections.Generic.List[string]]::new()
foreach ($entry in $required.GetEnumerator()) {
    $path = Join-Path $session $entry.Value
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $missing.Add("$($entry.Key):$($entry.Value)")
    }
}

$completeMarker = Join-Path $session 'capture_complete.json'
$incompleteMarker = Join-Path $session 'capture_incomplete.json'
$seal = if (Test-Path -LiteralPath $completeMarker) {
    'complete'
} elseif (Test-Path -LiteralPath $incompleteMarker) {
    'incomplete_with_capture_loss'
} else {
    'unsealed'
}

function Import-SafeCsv([string]$relativePath) {
    $path = Join-Path $session $relativePath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return @() }
    return @(Import-Csv -LiteralPath $path)
}

function Number($value) {
    $parsed = 0.0
    if ([double]::TryParse([string]$value,
        [Globalization.NumberStyles]::Float,
        [Globalization.CultureInfo]::InvariantCulture,
        [ref]$parsed)) { return $parsed }
    return [double]::NaN
}

function Percent([double]$numerator, [double]$denominator) {
    if ($denominator -le 0) { return 'n/a' }
    return (($numerator * 100.0 / $denominator).ToString('F2',
        [Globalization.CultureInfo]::InvariantCulture) + '%')
}

function Percentile([double[]]$values, [double]$fraction) {
    $valid = @($values | Where-Object { -not [double]::IsNaN($_) } | Sort-Object)
    if ($valid.Count -eq 0) { return [double]::NaN }
    $index = [Math]::Max(0, [Math]::Min($valid.Count - 1,
        [Math]::Ceiling($valid.Count * $fraction) - 1))
    return $valid[$index]
}

function Mean([double[]]$values) {
    $valid = @($values | Where-Object { -not [double]::IsNaN($_) })
    if ($valid.Count -eq 0) { return [double]::NaN }
    return ($valid | Measure-Object -Average).Average
}

function Signed-Euler([double]$degrees) {
    if ([double]::IsNaN($degrees)) { return $degrees }
    $wrapped = $degrees % 360.0
    if ($wrapped -gt 180.0) { $wrapped -= 360.0 }
    if ($wrapped -lt -180.0) { $wrapped += 360.0 }
    return $wrapped
}

function Bucket([double]$value, [double[]]$limits, [string[]]$labels) {
    if ([double]::IsNaN($value)) { return 'unknown' }
    for ($i = 0; $i -lt $limits.Count; $i++) {
        if ($value -lt $limits[$i]) { return $labels[$i] }
    }
    return $labels[$labels.Count - 1]
}

function Append-FactorTable(
    [Text.StringBuilder]$builder,
    [object[]]$rows,
    [string]$factorName,
    [scriptblock]$bucketSelector
) {
    $builder.AppendLine("factor=$factorName") | Out-Null
    $builder.AppendLine('bucket,events,accepted,initial,mean_lost_cells,mean_added_cells,mean_suspected_moved_cells,p95_queue_to_decision_ms') | Out-Null
    $groups = $rows | Group-Object { & $bucketSelector $_ } | Sort-Object Name
    foreach ($group in $groups) {
        $accepted = @($group.Group | Where-Object { $_.accepted -eq '1' }).Count
        $initial = @($group.Group | Where-Object { $_.initial -eq '1' }).Count
        $lost = @($group.Group | ForEach-Object { Number $_.lost_cells })
        $added = @($group.Group | ForEach-Object { Number $_.added_cells })
        $moved = @($group.Group | ForEach-Object { Number $_.suspected_moved_cells })
        $queue = @($group.Group | ForEach-Object { Number $_.queue_to_decision_ms })
        $line = [string]::Join(',', @(
            $group.Name,
            $group.Count,
            $accepted,
            $initial,
            (Mean $lost).ToString('F3', [Globalization.CultureInfo]::InvariantCulture),
            (Mean $added).ToString('F3', [Globalization.CultureInfo]::InvariantCulture),
            (Mean $moved).ToString('F3', [Globalization.CultureInfo]::InvariantCulture),
            (Percentile $queue 0.95).ToString('F3', [Globalization.CultureInfo]::InvariantCulture)
        ))
        $builder.AppendLine($line) | Out-Null
    }
    $builder.AppendLine() | Out-Null
}

$depth = Import-SafeCsv 'depth_pairs/manifest.csv'
$fusion = Import-SafeCsv 'fusion_inputs/manifest.csv'
$timeline = Import-SafeCsv 'runtime_timeline/frames.csv'
$dispatch = Import-SafeCsv 'runtime_timeline/integration_dispatches.csv'
$paper = Import-SafeCsv 'artifacts/paper_audit/production_paper_replacements.csv'

$depthFrames = [Collections.Generic.HashSet[int]]::new()
foreach ($row in $depth) { [void]$depthFrames.Add([int]$row.platformFrame) }
$dispatchAttempts = [Collections.Generic.HashSet[int]]::new()
$dispatchEpochs = [Collections.Generic.HashSet[uint32]]::new()
foreach ($row in $dispatch) {
    [void]$dispatchAttempts.Add([int]$row.attemptIndex)
    [void]$dispatchEpochs.Add([uint32]$row.dirtyEpoch)
}

$acceptedFusion = @($fusion | Where-Object { $_.accepted -eq '1' })
$missingDepthJoin = @($acceptedFusion | Where-Object {
    -not $depthFrames.Contains([int]$_.sourceFrame)
})
$missingDispatchJoin = @($acceptedFusion | Where-Object {
    -not $dispatchAttempts.Contains([int]$_.attemptIndex)
})
$traceablePaper = @($paper | Where-Object {
    $_.initial -ne '1'
})
$missingPaperEpoch = @($traceablePaper | Where-Object {
    -not $dispatchEpochs.Contains([uint32]$_.candidate_epoch)
})

$sb = [Text.StringBuilder]::new(128KB)
$sb.AppendLine('ScanCover full output audit') | Out-Null
$sb.AppendLine("session=$session") | Out-Null
$sb.AppendLine("seal=$seal") | Out-Null
$sb.AppendLine("required_missing=$($missing.Count)") | Out-Null
foreach ($item in $missing) { $sb.AppendLine("missing=$item") | Out-Null }
$sb.AppendLine() | Out-Null

$sb.AppendLine('chain_reconciliation:') | Out-Null
$sb.AppendLine("depth_rows=$($depth.Count)") | Out-Null
$sb.AppendLine("fusion_rows=$($fusion.Count)") | Out-Null
$sb.AppendLine("fusion_accepted=$($acceptedFusion.Count)") | Out-Null
$sb.AppendLine("integration_dispatch_rows=$($dispatch.Count)") | Out-Null
$sb.AppendLine("paper_replacement_rows=$($paper.Count)") | Out-Null
$sb.AppendLine("paper_initial_publish_rows=$($paper.Count - $traceablePaper.Count)") | Out-Null
$sb.AppendLine("paper_traceable_replacement_rows=$($traceablePaper.Count)") | Out-Null
$sb.AppendLine("accepted_fusion_missing_depth_pair=$($missingDepthJoin.Count)") | Out-Null
$sb.AppendLine("accepted_fusion_missing_dispatch=$($missingDispatchJoin.Count)") | Out-Null
$sb.AppendLine("paper_candidate_epoch_missing_dispatch=$($missingPaperEpoch.Count)") | Out-Null
$sb.AppendLine('join_rule=paper candidate_epoch -> integration dirtyEpoch -> sourceFrame -> raw/processed depth') | Out-Null
$sb.AppendLine() | Out-Null

if ($timeline.Count -gt 0) {
    $delta = @($timeline | ForEach-Object { Number $_.deltaMs })
    $maxQueue = ($timeline | ForEach-Object { Number $_.paperQueuedChunks } |
        Measure-Object -Maximum).Maximum
    $maxCommit = ($timeline | ForEach-Object { Number $_.paperCommitPending } |
        Measure-Object -Maximum).Maximum
    $maxOutstanding = ($timeline | ForEach-Object { Number $_.fusionOutstanding } |
        Measure-Object -Maximum).Maximum
    $maxDropped = ($timeline | ForEach-Object { Number $_.fusionDropped } |
        Measure-Object -Maximum).Maximum
    $sb.AppendLine('runtime_pressure:') | Out-Null
    $sb.AppendLine("timeline_rows=$($timeline.Count)") | Out-Null
    $sb.AppendLine("frame_delta_p50_ms=$((Percentile $delta 0.50).ToString('F3', [Globalization.CultureInfo]::InvariantCulture))") | Out-Null
    $sb.AppendLine("frame_delta_p95_ms=$((Percentile $delta 0.95).ToString('F3', [Globalization.CultureInfo]::InvariantCulture))") | Out-Null
    $sb.AppendLine("paper_queue_max=$maxQueue") | Out-Null
    $sb.AppendLine("paper_commit_pending_max=$maxCommit") | Out-Null
    $sb.AppendLine("fusion_outstanding_max=$maxOutstanding") | Out-Null
    $sb.AppendLine("fusion_dropped_max=$maxDropped") | Out-Null
    $sb.AppendLine() | Out-Null
}

if ($depth.Count -gt 0) {
    $rawInvalid = ($depth | ForEach-Object { Number $_.rawInvalid } | Measure-Object -Sum).Sum
    $rawTotal = ($depth | ForEach-Object { Number $_.rawTotal } | Measure-Object -Sum).Sum
    $processedInvalid = ($depth | ForEach-Object { Number $_.processedInvalid } | Measure-Object -Sum).Sum
    $processedTotal = ($depth | ForEach-Object { Number $_.processedTotal } | Measure-Object -Sum).Sum
    $pairRms = @($depth | ForEach-Object { Number $_.processedMinusRawRmsMetres })
    $poseMm = @($depth | ForEach-Object { Number $_.callbackToPreprocessPositionMm })
    $poseDeg = @($depth | ForEach-Object { Number $_.callbackToPreprocessRotationDeg })
    $sb.AppendLine('depth_stage:') | Out-Null
    $sb.AppendLine("raw_invalid_rate=$(Percent $rawInvalid $rawTotal)") | Out-Null
    $sb.AppendLine("processed_invalid_rate=$(Percent $processedInvalid $processedTotal)") | Out-Null
    $sb.AppendLine("raw_to_processed_rms_p50_m=$((Percentile $pairRms 0.50).ToString('F6', [Globalization.CultureInfo]::InvariantCulture))") | Out-Null
    $sb.AppendLine("raw_to_processed_rms_p95_m=$((Percentile $pairRms 0.95).ToString('F6', [Globalization.CultureInfo]::InvariantCulture))") | Out-Null
    $sb.AppendLine("callback_to_preprocess_position_p95_mm=$((Percentile $poseMm 0.95).ToString('F3', [Globalization.CultureInfo]::InvariantCulture))") | Out-Null
    $sb.AppendLine("callback_to_preprocess_rotation_p95_deg=$((Percentile $poseDeg 0.95).ToString('F3', [Globalization.CultureInfo]::InvariantCulture))") | Out-Null
    $sb.AppendLine() | Out-Null
}

if ($paper.Count -gt 0) {
    Append-FactorTable $sb $paper 'distance_to_chunk_m' {
        param($row)
        Bucket (Number $row.head_to_chunk_center_m) @(0.75, 1.25, 2.0) @('<0.75','0.75-1.25','1.25-2.0','>=2.0')
    }
    Append-FactorTable $sb $paper 'absolute_pitch_deg' {
        param($row)
        $pitch = [Math]::Abs((Signed-Euler (Number $row.head_pitch_deg)))
        Bucket $pitch @(15, 35, 60) @('<15','15-35','35-60','>=60')
    }
    Append-FactorTable $sb $paper 'angular_speed_deg_per_s' {
        param($row)
        Bucket (Number $row.angular_deg_per_sec_at_decision) @(5, 20, 60) @('<5','5-20','20-60','>=60')
    }
    Append-FactorTable $sb $paper 'linear_speed_m_per_s' {
        param($row)
        Bucket (Number $row.linear_mps_at_decision) @(0.02, 0.08, 0.20) @('<0.02','0.02-0.08','0.08-0.20','>=0.20')
    }
}

$surfacePath = Join-Path $session 'artifacts/paper_audit/production_paper_surface_ledger.txt'
if (Test-Path -LiteralPath $surfacePath) {
    $surfaceLines = [IO.File]::ReadAllLines($surfacePath)
    $start = [Array]::IndexOf($surfaceLines, 'visual_quality_spatial_csv:')
    if ($start -ge 0 -and $start + 2 -lt $surfaceLines.Count) {
        $rows = [Collections.Generic.List[string]]::new()
        for ($i = $start + 1; $i -lt $surfaceLines.Count; $i++) {
            if ([string]::IsNullOrWhiteSpace($surfaceLines[$i]) -or
                $surfaceLines[$i] -eq 'stage_responsibility_contract:' -or
                $surfaceLines[$i] -eq 'boundary_seam_csv:') { break }
            $rows.Add($surfaceLines[$i])
        }
        if ($rows.Count -gt 1) {
            $spatial = @($rows | ConvertFrom-Csv)
            $worst = @($spatial | Where-Object { $_.plane_candidate -eq '1' } |
                Sort-Object { Number $_.plane_rms_vox } -Descending | Select-Object -First 20)
            $sb.AppendLine('worst_planar_surface_bins:') | Out-Null
            $sb.AppendLine('chunk,bin,triangles,plane_rms_vox,plane_thin_ratio,local_min_m,local_max_m') | Out-Null
            foreach ($row in $worst) {
                $line = [string]::Join(',', @(
                    "$($row.chunk_x)/$($row.chunk_y)/$($row.chunk_z)",
                    "$($row.bin_x)/$($row.bin_y)/$($row.bin_z)",
                    $row.triangles,
                    $row.plane_rms_vox,
                    $row.plane_thin_ratio,
                    "$($row.local_min_x_m)/$($row.local_min_y_m)/$($row.local_min_z_m)",
                    "$($row.local_max_x_m)/$($row.local_max_y_m)/$($row.local_max_z_m)"
                ))
                $sb.AppendLine($line) | Out-Null
            }
            $sb.AppendLine() | Out-Null
        }
    }

    $stageStart = [Array]::IndexOf($surfaceLines, 'stage_responsibility_spatial_csv:')
    if ($stageStart -ge 0 -and $stageStart + 2 -lt $surfaceLines.Count) {
        $stageRows = [Collections.Generic.List[string]]::new()
        for ($i = $stageStart + 1; $i -lt $surfaceLines.Count; $i++) {
            if ([string]::IsNullOrWhiteSpace($surfaceLines[$i]) -or
                $surfaceLines[$i] -eq 'stage_first_divergence_summary_csv:' -or
                $surfaceLines[$i] -eq 'boundary_seam_csv:') { break }
            $stageRows.Add($surfaceLines[$i])
        }
        if ($stageRows.Count -gt 1) {
            $stageSpatial = @($stageRows | ConvertFrom-Csv)
            $sb.AppendLine('stage_first_divergence:') | Out-Null
            $sb.AppendLine('stage,bins,triangles') | Out-Null
            foreach ($group in ($stageSpatial | Group-Object first_deformation_stage | Sort-Object Name)) {
                $triangles = ($group.Group | ForEach-Object { Number $_.triangles } |
                    Measure-Object -Sum).Sum
                $sb.AppendLine("$($group.Name),$($group.Count),$triangles") | Out-Null
            }
            $sb.AppendLine() | Out-Null

            $worstCrossing = @($stageSpatial |
                Where-Object { $_.crossing_plane_candidate -eq '1' } |
                Sort-Object { Number $_.crossing_plane_rms_vox } -Descending |
                Select-Object -First 20)
            $sb.AppendLine('worst_tsdf_zero_crossing_bins:') | Out-Null
            $sb.AppendLine('chunk,bin,epoch,triangles,crossing_rms_vox,raw_rms_vox,final_rms_vox,first_deformation_stage,local_min_m,local_max_m') | Out-Null
            foreach ($row in $worstCrossing) {
                $line = [string]::Join(',', @(
                    "$($row.chunk_x)/$($row.chunk_y)/$($row.chunk_z)",
                    "$($row.bin_x)/$($row.bin_y)/$($row.bin_z)",
                    $row.built_epoch,
                    $row.triangles,
                    $row.crossing_plane_rms_vox,
                    $row.raw_plane_rms_vox,
                    $row.final_plane_rms_vox,
                    $row.first_deformation_stage,
                    "$($row.local_min_x_m)/$($row.local_min_y_m)/$($row.local_min_z_m)",
                    "$($row.local_max_x_m)/$($row.local_max_y_m)/$($row.local_max_z_m)"
                ))
                $sb.AppendLine($line) | Out-Null
            }
            $sb.AppendLine() | Out-Null
        }
    }

    $seamStart = [Array]::IndexOf($surfaceLines, 'boundary_seam_csv:')
    if ($seamStart -ge 0 -and $seamStart + 2 -lt $surfaceLines.Count) {
        $seamRows = [Collections.Generic.List[string]]::new()
        for ($i = $seamStart + 1; $i -lt $surfaceLines.Count; $i++) {
            if ([string]::IsNullOrWhiteSpace($surfaceLines[$i]) -or
                $surfaceLines[$i].StartsWith('boundary_semantics=')) { break }
            $seamRows.Add($surfaceLines[$i])
        }
        if ($seamRows.Count -gt 1) {
            $seams = @($seamRows | ConvertFrom-Csv)
            $badSeams = @($seams | Where-Object { $_.status -ne 'aligned' })
            $sb.AppendLine('boundary_seams:') | Out-Null
            $sb.AppendLine("tested=$($seams.Count)") | Out-Null
            $sb.AppendLine("mismatch_candidates=$($badSeams.Count)") | Out-Null
            $sb.AppendLine("position_gt_5mm=$(($seams | ForEach-Object { Number $_.delta_gt_5mm } | Measure-Object -Sum).Sum)") | Out-Null
            $sb.AppendLine() | Out-Null
        }
    }
}

$responsibilitySummaryPath = Join-Path $session 'artifacts/tsdf_responsibility/analysis_summary.json'
$responsibilityAnalyzer = Join-Path $PSScriptRoot 'AnalyzeTsdfResponsibility.py'
if ((Test-Path -LiteralPath (Join-Path $session 'artifacts/tsdf_responsibility/schema.json') -PathType Leaf) -and
    (Test-Path -LiteralPath $responsibilityAnalyzer -PathType Leaf)) {
    $python = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        & $python.Source -3 $responsibilityAnalyzer $session | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "TSDF responsibility analyzer failed with exit code $LASTEXITCODE" }
    } else {
        $python = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $python) { throw 'Python is required for TSDF responsibility analysis' }
        & $python.Source $responsibilityAnalyzer $session | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "TSDF responsibility analyzer failed with exit code $LASTEXITCODE" }
    }
}

if (Test-Path -LiteralPath $responsibilitySummaryPath -PathType Leaf) {
    $responsibility = Get-Content -LiteralPath $responsibilitySummaryPath -Raw | ConvertFrom-Json
    $sb.AppendLine('tsdf_transaction_responsibility:') | Out-Null
    $sb.AppendLine("target_bad_bins=$($responsibility.targetBins)") | Out-Null
    $sb.AppendLine("zero_crossing_endpoints=$($responsibility.zeroCrossingEndpoints)") | Out-Null
    $sb.AppendLine("saturated_responsibility_endpoints=$($responsibility.saturatedResponsibilityEndpoints)") | Out-Null
    $sb.AppendLine("missing_current_lifetime_provenance=$($responsibility.endpointsWithoutCurrentLifetimeProvenance)") | Out-Null
    $sb.AppendLine("endpoints_with_blocked_correction=$($responsibility.endpointsWithBlockedCorrection)") | Out-Null
    $sb.AppendLine("exact_projected_source_samples=$($responsibility.exactProjectedSamples)") | Out-Null
    $sb.AppendLine("raw_post_comparable=$($responsibility.projectedRawPostComparable)") | Out-Null
    $sb.AppendLine("raw_post_delta_ge_10mm=$($responsibility.projectedRawPostDeltaAtLeast10Mm)") | Out-Null
    $sb.AppendLine("hole_cells=$($responsibility.holeCells)") | Out-Null
    $sb.AppendLine("hole_halo_context_cells=$($responsibility.holeHaloContextCells)") | Out-Null
    $sb.AppendLine("hole_exact_projected_samples=$($responsibility.holeExactProjectedSamples)") | Out-Null
    $sb.AppendLine("hole_boundaries=$($responsibility.holeResponsibleBoundaryCounts | ConvertTo-Json -Compress)") | Out-Null
    $sb.AppendLine('detail=artifacts/tsdf_responsibility/endpoint_responsibility.csv, endpoint_source_samples.csv, paper_hole_responsibility.csv and paper_hole_source_samples.csv') | Out-Null
    $sb.AppendLine() | Out-Null
}

$joinedResponsibilityTool = Join-Path $PSScriptRoot 'BuildScanCoverResponsibilityLedger.py'
$joinedResponsibilitySummary = Join-Path $session 'artifacts/responsibility_ledger/responsibility_summary.json'
if (Test-Path -LiteralPath $joinedResponsibilityTool -PathType Leaf) {
    $python = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        & $python.Source -3 $joinedResponsibilityTool $session | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Full-chain responsibility ledger failed with exit code $LASTEXITCODE" }
    } else {
        $python = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $python) { throw 'Python is required for full-chain responsibility analysis' }
        & $python.Source $joinedResponsibilityTool $session | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Full-chain responsibility ledger failed with exit code $LASTEXITCODE" }
    }
}
if (Test-Path -LiteralPath $joinedResponsibilitySummary -PathType Leaf) {
    $joined = Get-Content -LiteralPath $joinedResponsibilitySummary -Raw | ConvertFrom-Json
    $sb.AppendLine('full_chain_responsibility:') | Out-Null
    $sb.AppendLine("site_count=$($joined.siteCount)") | Out-Null
    $sb.AppendLine("final_comparable_defect_sites=$($joined.finalComparableDefectSites)") | Out-Null
    $sb.AppendLine("final_comparable_flat_sites=$($joined.finalComparableFlatSites)") | Out-Null
    $sb.AppendLine("final_not_plane_comparable_sites=$($joined.finalNotPlaneComparableSites)") | Out-Null
    $sb.AppendLine("timeline_rows=$($joined.timelineRows)") | Out-Null
    $sb.AppendLine("court_trace_rows=$($joined.courtTraceRows)") | Out-Null
    $sb.AppendLine("quality_timeline_rows=$($joined.qualityTimelineRows)") | Out-Null
    $sb.AppendLine("motion_confirm_quality_min=$($joined.motionConfirmQualityMin)") | Out-Null
    $sb.AppendLine("blocked_correction_totals=$($joined.blockedCorrectionTotals | ConvertTo-Json -Compress)") | Out-Null
    $sb.AppendLine("missing_lanes=$([string]::Join('|', @($joined.missingLanes)))") | Out-Null
    $sb.AppendLine('detail=artifacts/responsibility_ledger/surface_site_timeline.csv, court_decision_trace.csv, roughness_lineage.csv and factor_responsibility.csv') | Out-Null
    $sb.AppendLine() | Out-Null
}

# The instant shell can look correct from its birth view even when its metric
# depth is wrong.  Run a read-only held-out reprojection exam before anyone
# promotes that display path into production geometry authority.
$shellAuditTool = Join-Path $PSScriptRoot 'ScanCoverInstantShellSaviourAudit.py'
$shellAuditSummary = Join-Path $session 'artifacts/instant_shell_saviour_audit/summary.json'
if ((Test-Path -LiteralPath $shellAuditTool -PathType Leaf) -and
    (Test-Path -LiteralPath (Join-Path $session 'depth_pairs/manifest.csv') -PathType Leaf)) {
    $projectRoot = Split-Path -Parent $PSScriptRoot
    $scanCoverPython = Join-Path $projectRoot '.venv-scancover/Scripts/python.exe'
    if (Test-Path -LiteralPath $scanCoverPython -PathType Leaf) {
        & $scanCoverPython $shellAuditTool $session | Out-Null
    } else {
        $python = Get-Command py -ErrorAction SilentlyContinue
        if ($null -ne $python) {
            & $python.Source -3 $shellAuditTool $session | Out-Null
        } else {
            $python = Get-Command python -ErrorAction SilentlyContinue
            if ($null -eq $python) { throw 'Python with numpy is required for instant-shell qualification' }
            & $python.Source $shellAuditTool $session | Out-Null
        }
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Instant-shell saviour audit failed with exit code $LASTEXITCODE"
    }
}
if (Test-Path -LiteralPath $shellAuditSummary -PathType Leaf) {
    $shellAudit = Get-Content -LiteralPath $shellAuditSummary -Raw | ConvertFrom-Json
    $sb.AppendLine('instant_shell_saviour_qualification:') | Out-Null
    $sb.AppendLine("verdict=$($shellAudit.verdict)") | Out-Null
    $sb.AppendLine("evidence_gate=$($shellAudit.gates.evidence)") | Out-Null
    $sb.AppendLine("accuracy_gate=$($shellAudit.gates.accuracy)") | Out-Null
    $sb.AppendLine("coverage_gate=$($shellAudit.gates.coverage)") | Out-Null
    $sb.AppendLine("condition_gate=$($shellAudit.gates.conditionRobustness)") | Out-Null
    $sb.AppendLine("independent_comparable_samples=$($shellAudit.evidence.independentComparableSamples)") | Out-Null
    $sb.AppendLine("independent_comparable_coverage_pct=$($shellAudit.evidence.independentComparableCoveragePct)") | Out-Null
    $sb.AppendLine("qualified_patch_pct=$($shellAudit.evidence.qualified10cmPatchPct)") | Out-Null
    $sb.AppendLine("abs_residual_p50_mm=$($shellAudit.independentViewAbsResidualMm.p50)") | Out-Null
    $sb.AppendLine("abs_residual_p95_mm=$($shellAudit.independentViewAbsResidualMm.p95)") | Out-Null
    $sb.AppendLine("over_10mm_conflict_pct=$($shellAudit.over10mmConflictPct)") | Out-Null
    $sb.AppendLine('detail=artifacts/instant_shell_saviour_audit/VERDICT.md, samples.csv.gz, frame_pairs.csv, patches_10cm.csv and condition_bins.csv') | Out-Null
    $sb.AppendLine() | Out-Null
}

$verdict = if ($seal -ne 'complete' -or $missing.Count -gt 0 -or
    $missingDepthJoin.Count -gt 0 -or $missingDispatchJoin.Count -gt 0 -or
    $missingPaperEpoch.Count -gt 0) {
    'INCOMPLETE_OR_BROKEN_CHAIN'
} else {
    'CHAIN_COMPLETE_READY_FOR_CAUSAL_COMPARISON'
}
$sb.AppendLine("verdict=$verdict") | Out-Null
$sb.AppendLine('interpretation=associations expose suspects; they do not alone prove a sensor or fusion cause') | Out-Null

[IO.File]::WriteAllText([IO.Path]::GetFullPath($OutputPath), $sb.ToString(),
    [Text.UTF8Encoding]::new($false))
Write-Host "Wrote $OutputPath"
Write-Host "Verdict $verdict"
