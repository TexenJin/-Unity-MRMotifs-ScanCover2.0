$ErrorActionPreference = 'Stop'

$repo = Split-Path $PSScriptRoot -Parent
$core = Join-Path $repo 'Assets/QuestRoomScanStandalone/Core'
$shaders = Join-Path $repo 'Assets/QuestRoomScanStandalone/Shaders'

$integrator = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $core 'VolumeIntegrator.cs')
$fusion = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $shaders 'VolumeIntegration.compute')
$pipeline = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $core 'PersistentChunkMeshPipeline.cs')
$surface = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $shaders 'SurfaceNetsExtract.compute')
$scanner = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $core 'StandaloneRoomScanner.cs')

function Require([bool]$condition, [string]$message) {
    if (!$condition) { throw $message }
}

Require ($integrator.Contains('private bool enableFrustumPhaseCoverage = true;')) `
    'Frustum phase coverage is not enabled by default.'
Require ($integrator.Contains('FrustumCoveragePhases[IntegrationCount & 7] * voxelSize')) `
    'Accepted-frame phase cycle is missing.'
Require ($integrator.Contains('compute.SetVector(FrustumPhaseOffsetID, frustumPhase);')) `
    'Frustum phase is not published to the compute shader.'
Require ($fusion.Contains('gsFrustumVolume[id] + gsFrustumPhaseOffset')) `
    'Fusion does not apply the phase before exact world-voxel reprojection.'

$phaseBlock = [regex]::Match($integrator,
    '(?s)FrustumCoveragePhases\s*=\s*\{(.*?)\};').Groups[1].Value
Require (![string]::IsNullOrWhiteSpace($phaseBlock)) 'Coverage phase block was not found.'
$phaseMatches = [regex]::Matches($phaseBlock,
    'new Vector3\(\s*([-0-9.]+)f,\s*([-0-9.]+)f,\s*([-0-9.]+)f\)')
Require ($phaseMatches.Count -eq 8) 'Coverage schedule must contain exactly eight phases.'
foreach ($axis in 1..3) {
    $sum = 0.0
    foreach ($match in $phaseMatches) { $sum += [double]$match.Groups[$axis].Value }
    Require ([math]::Abs($sum) -lt 0.000001) "Coverage phase axis $axis is not centred."
}

Require ($pipeline.Contains('chunk.Surface.FoundationCellStride = 1;')) `
    'Diagnostic foundation is not fixed to the native 5 cm lattice.'
Require ($pipeline.Contains('chunk.Surface.FoundationConstrainedSimplification = false;')) `
    'Conditional 5-to-10 cm collapse would reintroduce mixed-scale output.'
Require ($surface.Contains('int3 cMid = (cA + cB) / 2;')) `
    '10 cm vertex extraction does not read its native 5 cm midpoint.'
Require ($surface.Contains('foundationMultipleCrossings = true;')) `
    'Multi-layer coarse edges are not rejected during vertex extraction.'
Require ($surface.Contains('bool strictTopologyEdge = _StrictObservedEdges > 0.5 ||')) `
    '10 cm index generation is not forced to strict observed support.'
Require ($surface.Contains('bool observedMid = TrySampleObservedSDF(coord + axis, vm);')) `
    '10 cm index generation does not validate the native 5 cm midpoint.'
Require ($surface.Contains('if (crossesFirst == crossesSecond)')) `
    '10 cm index generation does not reject zero/double-crossing edges.'

Require ($scanner.Contains('renderQueue = 4990')) 'HUD plate render queue is missing.'
Require ($scanner.Contains('renderQueue = 4991')) 'HUD text render queue is missing.'
Require (([regex]::Matches($scanner,
    'new Color\(0\.008f, 0\.012f, 0\.018f, 1f\)')).Count -eq 2) `
    'Both HUD plates must be fully opaque high-contrast backgrounds.'
Require ($scanner.Contains('蒙皮[原生5cm诊断]')) `
    'HUD does not identify the native 5 cm diagnostic baseline.'

'PASS: accepted-frame TSDF phase coverage, native 5 cm diagnostic topology, dormant 10 cm guards, and opaque late HUD plate.'
