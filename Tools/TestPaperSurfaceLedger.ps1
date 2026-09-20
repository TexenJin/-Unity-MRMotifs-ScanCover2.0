$ErrorActionPreference = 'Stop'

$repo = Split-Path $PSScriptRoot -Parent
$rendererPath = Join-Path $repo 'Assets/QuestRoomScanStandalone/Core/SupportTruthRenderer.cs'
$meshPath = Join-Path $repo 'Assets/QuestRoomScanStandalone/Core/MeshExtractor.cs'
$replayPath = Join-Path $repo 'Assets/QuestRoomScanStandalone/Core/ScanReplaySessionPackage.cs'
$renderer = Get-Content -Raw -Encoding UTF8 -LiteralPath $rendererPath
$mesh = Get-Content -Raw -Encoding UTF8 -LiteralPath $meshPath
$replay = Get-Content -Raw -Encoding UTF8 -LiteralPath $replayPath

function Require([bool]$condition, [string]$message) {
    if (!$condition) { throw $message }
}

# The operator-visible production paper is the native 5 cm sparse foundation,
# not the hidden 10 cm support topology. Its already-committed per-chunk visual
# ledger must therefore be the authoritative A-key roughness artifact.
Require ($mesh.Contains('source=sparse_foundation_committed_fronts')) `
    'Paper ledger is not labelled as the native 5 cm committed front.'
Require ($mesh.Contains('_sparseFoundationSkin.AcceptedTriangleCount')) `
    'Paper ledger does not verify that the drawn production source is non-empty.'
Require ($mesh.Contains('_sparseFoundationSkin.AppendVisualQualityReport(')) `
    'Paper ledger does not export production spatial plane residuals.'
Require ($mesh.Contains('production_paper_surface_ledger.txt')) `
    'Production paper surface artifact is missing.'
Require ($mesh.Contains('completedPath = string.Empty;')) `
    'A zero or failed production-paper ledger can still be reported as successful.'

# The old support renderer remains only the first-break/hole ledger. It must
# not emit another file named as though it described the visible 5 cm paper.
Require (!$renderer.Contains('paper_surface_patches.csv')) `
    'Hidden support topology still masquerades as the production surface ledger.'
Require (!$renderer.Contains('WritePaperSurfaceLedger')) `
    'Redundant hidden-topology roughness calculation was not removed.'

# Replay packaging already copies the complete paper-audit directory; the new
# files therefore leave the headset with the same established capture action.
Require ($replay.Contains('CopyArtifact(path, "paper_audit")')) `
    'Paper audit directory is no longer copied into the replay package.'

'PASS: the paper ledger reads the native 5 cm committed fronts and exports spatial roughness with corner qualifiers.'
