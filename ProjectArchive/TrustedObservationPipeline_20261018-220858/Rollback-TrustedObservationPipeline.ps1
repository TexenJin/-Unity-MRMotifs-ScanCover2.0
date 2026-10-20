param(
    [switch]$Apply
)

$ErrorActionPreference = "Stop"
$checkpointRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $checkpointRoot "..\..")).Path
$expectedArchive = (Resolve-Path $checkpointRoot).Path

if (-not $expectedArchive.StartsWith($projectRoot,
        [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Checkpoint is not inside the expected project root."
}

$relativeFiles = @(
    "Assets\QuestRoomScanStandalone\Core\DepthCapture.cs",
    "Assets\QuestRoomScanStandalone\Core\InfiniTamModelRaycastAudit.cs",
    "Assets\QuestRoomScanStandalone\Core\VolumeIntegrator.cs",
    "Assets\QuestRoomScanStandalone\Resources\InfiniTamModelRaycast.compute",
    "Assets\QuestRoomScanStandalone\Shaders\VolumeIntegration.compute"
)
$expectedHashes = @{}
Get-Content -LiteralPath (Join-Path $checkpointRoot "SHA256SUMS.txt") |
    ForEach-Object {
        if ($_ -match "^([0-9a-f]{64})  (.+)$") {
            $expectedHashes[$Matches[2]] = $Matches[1]
        }
    }

foreach ($relativeFile in $relativeFiles) {
    $source = Join-Path $checkpointRoot $relativeFile
    $destination = Join-Path $projectRoot $relativeFile
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Missing checkpoint file: $source"
    }
    if (-not (Test-Path -LiteralPath $destination -PathType Leaf)) {
        throw "Missing project file: $destination"
    }
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.
        ToLowerInvariant()
    if ($expectedHashes[$relativeFile] -ne $actualHash) {
        throw "Checkpoint hash mismatch: $source"
    }
}

if (-not $Apply) {
    Write-Host "Dry run only. These files would be restored:"
    $relativeFiles | ForEach-Object { Write-Host "  $_" }
    Write-Host "Run again with -Apply to perform the recoverable rollback."
    return
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$safetyRoot = Join-Path $checkpointRoot "BeforeRollback_$stamp"
New-Item -ItemType Directory -Path $safetyRoot | Out-Null

foreach ($relativeFile in $relativeFiles) {
    $source = Join-Path $checkpointRoot $relativeFile
    $destination = Join-Path $projectRoot $relativeFile
    $safety = Join-Path $safetyRoot $relativeFile
    New-Item -ItemType Directory -Path (Split-Path -Parent $safety) -Force |
        Out-Null
    Copy-Item -LiteralPath $destination -Destination $safety
    Copy-Item -LiteralPath $source -Destination $destination
}

Write-Host "Rollback complete. Replaced files were preserved at: $safetyRoot"
