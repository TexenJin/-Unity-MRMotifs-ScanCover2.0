param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'

function Require-Text([string]$Path, [string]$Pattern, [string]$Message) {
    $content = Get-Content -LiteralPath $Path -Raw
    if ($content -notmatch $Pattern) {
        throw $Message
    }
}

function Reject-Text([string]$Path, [string]$Pattern, [string]$Message) {
    $content = Get-Content -LiteralPath $Path -Raw
    if ($content -match $Pattern) {
        throw $Message
    }
}

$core = Join-Path $ProjectRoot 'Assets/QuestRoomScanStandalone/Core'
$resources = Join-Path $ProjectRoot 'Assets/QuestRoomScanStandalone/Resources'
$shaders = Join-Path $ProjectRoot 'Assets/QuestRoomScanStandalone/Shaders'

$volume = Join-Path $core 'VolumeIntegrator.cs'
$pipeline = Join-Path $core 'PersistentChunkMeshPipeline.cs'
$productizer = Join-Path $core 'SurfaceProductizer.cs'
$qualityGate = Join-Path $core 'ChunkQualityGate.cs'
$productCompute = Join-Path $resources 'SurfaceProductizer.compute'
$extractCompute = Join-Path $shaders 'SurfaceNetsExtract.compute'
$scanner = Join-Path $core 'StandaloneRoomScanner.cs'
$extractor = Join-Path $core 'MeshExtractor.cs'
$depth = Join-Path $core 'DepthCapture.cs'
$court = Join-Path $core 'RuntimeFinalSurfaceCourt.cs'

Require-Text $volume 'enableGunGelTsdfAdmission\s*=\s*false' `
    'GunGel TSDF authority must default off.'
Require-Text $volume 'enableGunGelPoseCorrection\s*=\s*false' `
    'GunGel pose correction must default off.'
Require-Text $volume 'enableFinalCourtAdmissionExperiment\s*=\s*false' `
    'The legacy pre-TSDF final court must default off.'
Require-Text $pipeline '_productizer\.Build[\s\S]*_qualityGate\.Evaluate[\s\S]*PublishFullCandidate' `
    'Candidate order must be productizer -> quality gate -> atomic publish.'
Require-Text $pipeline 'FoundationCellStride\s*=\s*1' `
    'Reconstruction candidate must remain native 5 cm.'
Require-Text $pipeline 'FoundationConstrainedSimplification\s*=\s*false' `
    'Unsafe 2x2x2 index aliasing must stay disabled.'
Require-Text $pipeline 'SetProductGridDisplay' `
    'Product surface is not connected to topology-safe 10 cm presentation.'
Require-Text (Join-Path $shaders 'ScanMeshVertexColor.shader') '_RSProductGridMode' `
    'Topology-safe product-grid shader mode is missing.'
Require-Text $pipeline 'FoundationVisibleSkirtVoxels\s*=\s*1' `
    'Visible mesh-block skirt must be enabled.'
Require-Text $pipeline 'MaximumConcurrentProductCommits\s*=\s*4' `
    'Product mesh generation needs a hard global concurrency bound.'
Require-Text $pipeline 'candidateEpoch\s*<\s*chunk\.TargetEpoch' `
    'Superseded asynchronous candidates must not replace a newer product front.'
Require-Text $pipeline 'candidateProductRevision\s*<\s*chunk\.TargetProductRevision' `
    'A superseded post-TSDF constraint result must not replace the newer product.'
Require-Text $pipeline 'commitSerial\s*!=\s*chunk\.CommitSerial' `
    'Late callbacks from watchdog-retired product requests must be ignored.'
Require-Text $pipeline 'chunk\.CommitSerial\+\+' `
    'The watchdog must invalidate an abandoned request before retrying it.'
Require-Text $qualityGate 'candidateEpoch\s*<\s*state\.CommittedEpoch' `
    'Same-TSDF-epoch product-constraint repairs must remain publishable.'
Require-Text $court 'EnsureProductScratch\(destination\.Length\)' `
    'Per-block product-plane clustering must not allocate fresh scratch arrays.'
Require-Text $pipeline 'FoundationAtomicReplacement\s*&&\s*!chunk\.Built' `
    'Initial product blocks must be prioritized ahead of ordinary updates.'
Require-Text $extractCompute '_FoundationVisibleSkirtVoxels' `
    'Surface extraction shader lacks mesh-block skirt ownership.'
Require-Text $volume '_productSurfaceCourt\.BeginInMemory\(\)' `
    'Production court must exist without a diagnostic recording session.'
Require-Text $volume 'QueueProductSurfaceCourtReadback' `
    'Accepted GunGel identity evidence is not reaching the production court.'
Require-Text $productizer '_volume\.CopyProductSurfacePlanes' `
    'Productizer is not consuming the independent production court.'
Require-Text $productizer 'CopySeedProductConstraints' `
    'Scene ruler is not connected as a post-TSDF product sidecar.'
Require-Text $productCompute '_ProductPlaneAxesU[\s\S]*supportU\.w\s*>=\s*0\.0' `
    'Finite seed-ruler support is not enforced by the productizer.'
Reject-Text $depth '_depthTex\s*=\s*_seedPlaneDepthTex' `
    'Seed ruler must not replace the screened Quest depth before GunGel/TSDF.'
Require-Text $pipeline '_volume\.DrainProductSurfaceChanges' `
    'Constraint changes must re-productize intersecting chunks.'
Require-Text $extractor 'public string ShowProductionPaperView\(\)[\s\S]*?_routeValidationView\s*=\s*RouteValidationView\.SparseFoundation' `
    'Production view is not connected to the productized surface chain.'
Require-Text $productCompute 'ambiguous \|\| boundary' `
    'Productizer must preserve ambiguous features and page boundaries.'
Require-Text $qualityGate 'RemovalConfirmationsRequired' `
    'Quality gate must confirm removals before deleting a visible front.'
Require-Text $scanner 'enablePlaneFlatten\s*=\s*false' `
    'Legacy B2 TSDF flattening must stay disabled.'
Require-Text $scanner 'showManagementBlockWireOverlay\s*=\s*false' `
    'Production must not cold-start with the cyan management-block overlay visible.'

Write-Output 'PASS: single TSDF -> 5 cm candidate -> productizer -> quality gate -> atomic front -> conservative 10 cm product'
