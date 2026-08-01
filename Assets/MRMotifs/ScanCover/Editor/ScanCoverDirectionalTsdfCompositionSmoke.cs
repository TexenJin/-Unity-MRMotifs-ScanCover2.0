#if UNITY_EDITOR
using System;
using UnityEditor;
using UnityEngine;

public static class ScanCoverDirectionalTsdfCompositionSmoke
{
    [MenuItem("ScanCover/Validation/Run Directional TSDF Composition Smoke")]
    public static void RunInteractive()
    {
        Debug.Log(RunValidation());
    }

    public static void Run()
    {
        try
        {
            Debug.Log(RunValidation());
            EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            EditorApplication.Exit(1);
        }
    }

    private static string RunValidation()
    {
        ScanCoverDirectionalTsdfShadow volume = new ScanCoverDirectionalTsdfShadow();
        const float voxel = 0.025f;
        volume.Configure(
            32, 32, 32,
            new Vector3(-0.4f, -0.4f, -0.4f),
            voxel, 0.075f, 32f, 8);

        // The Z sheet is intentionally written into both opposite channels.
        // Independent extraction publishes two copies; conservative composition
        // must retain only the direction whose gradient agrees with its channel.
        for (int y = -9; y <= 9; y++)
        for (int x = -9; x <= 9; x++)
        {
            Vector3 point = new Vector3(x * voxel, y * voxel, 0f);
            volume.Integrate(
                new Vector3(0f, 0f, -1f), point, Vector3.back,
                1f, 5, -1, false, false, 0.12f, 3);
            volume.Integrate(
                new Vector3(0f, 0f, -1f), point, Vector3.back,
                1f, 6, -1, true, false, 0.12f, 3);
        }

        // Add a genuine orthogonal face.  The composed result must not solve
        // overlap by deleting all multi-direction geometry around a corner.
        for (int y = -9; y <= 9; y++)
        for (int z = -9; z <= 2; z++)
        {
            Vector3 point = new Vector3(0f, y * voxel, z * voxel);
            volume.Integrate(
                new Vector3(-1f, 0f, 0f), point, Vector3.left,
                1f, 2, -1, true, false, 0.12f, 3);
        }

        ScanCoverDirectionalTsdfShadow.MeshResult independent =
            volume.BuildMesh(0.75f, 200000, 300000);
        ScanCoverDirectionalTsdfShadow.MeshResult composed =
            volume.BuildComposedMesh(
                0.75f, 200000, 300000,
                0.15f, 0.82f, 0.35f,
                true, false, 0.94f, 0.40f);

        if (composed.TriangleCount <= 0)
            throw new InvalidOperationException("Directional composition removed all valid geometry.");
        if (composed.NonManifoldEdges != 0)
            throw new InvalidOperationException(
                $"Directional composition produced {composed.NonManifoldEdges} non-manifold edges.");
        if (composed.TriangleCount >= independent.TriangleCount)
            throw new InvalidOperationException(
                $"Directional composition did not remove the duplicate sheet: " +
                $"independent={independent.TriangleCount}, composed={composed.TriangleCount}.");
        if (composed.InvalidDirectionHypotheses <= 0 &&
            composed.ParallelHypothesesCollapsed <= 0)
            throw new InvalidOperationException("No duplicate directional hypothesis was audited.");
        if (composed.SurfaceCandidateNodes <= composed.SurfaceComponents ||
            composed.SurfaceComponentLinks <= 0)
            throw new InvalidOperationException(
                $"Adjacent coplanar cells were not linked: nodes={composed.SurfaceCandidateNodes} " +
                $"components={composed.SurfaceComponents} links={composed.SurfaceComponentLinks}.");
        if (composed.SurfaceConsistentEdgeMerges <= 0)
            throw new InvalidOperationException("No same-surface grid edge was shared across cells.");
        if (!composed.DirectionalMcShadowEnabled ||
            composed.DirectionalMcShadowCellsEvaluated <= 0 ||
            composed.DirectionalMcShadowRawHypotheses <= 0 ||
            composed.DirectionalMcShadowValidHypotheses <= 0 ||
            composed.DirectionalMcShadowCombinedTransitionEdges <= 0 ||
            composed.DirectionalMcShadowOffsetClusters <= 0)
            throw new InvalidOperationException(
                "Directional Marching Cubes shadow did not complete the " +
                "index/filter/vote/offset chain: enabled/cells/raw/valid/edges/offsets=" +
                $"{composed.DirectionalMcShadowEnabled}/" +
                $"{composed.DirectionalMcShadowCellsEvaluated}/" +
                $"{composed.DirectionalMcShadowRawHypotheses}/" +
                $"{composed.DirectionalMcShadowValidHypotheses}/" +
                $"{composed.DirectionalMcShadowCombinedTransitionEdges}/" +
                $"{composed.DirectionalMcShadowOffsetClusters}.");
        if (composed.DirectionalMcShadowNeighborDisagreementsAfter >
            composed.DirectionalMcShadowNeighborDisagreementsBefore)
            throw new InvalidOperationException(
                "Directional MC neighbor regularization increased border " +
                "disagreements: before/after=" +
                $"{composed.DirectionalMcShadowNeighborDisagreementsBefore}/" +
                $"{composed.DirectionalMcShadowNeighborDisagreementsAfter}.");

        RunSplitSurfaceContinuityFixture(
            out ScanCoverDirectionalTsdfShadow.MeshResult splitLocal,
            out ScanCoverDirectionalTsdfShadow.MeshResult splitContinuous);
        if (splitContinuous.SurfaceComponentLinks <= 0 ||
            splitContinuous.SurfaceComponents >= splitLocal.SurfaceComponents)
            throw new InvalidOperationException(
                $"Primary/secondary wall patches did not become one surface: " +
                $"local={splitLocal.SurfaceComponents} continuous={splitContinuous.SurfaceComponents} " +
                $"links={splitContinuous.SurfaceComponentLinks}.");
        int splitLocalInternalBoundary = CountBoundaryEdgesNearX(splitLocal, -0.0125f, 0.04f);
        int splitContinuousInternalBoundary = CountBoundaryEdgesNearX(splitContinuous, -0.0125f, 0.04f);
        if (splitContinuous.BoundaryEdges > splitLocal.BoundaryEdges ||
            splitContinuousInternalBoundary > splitLocalInternalBoundary)
            throw new InvalidOperationException(
                $"Surface identity increased split-wall boundaries: " +
                $"internal={splitLocalInternalBoundary}>{splitContinuousInternalBoundary} " +
                $"all={splitLocal.BoundaryEdges}>{splitContinuous.BoundaryEdges} " +
                $"bridges={splitContinuous.SurfaceGapBridgeCandidates}/" +
                $"{splitContinuous.SurfaceGapBridgeTriangles} " +
                $"nodes/components/links={splitContinuous.SurfaceCandidateNodes}/" +
                $"{splitContinuous.SurfaceComponents}/{splitContinuous.SurfaceComponentLinks} " +
                $"triangles={splitLocal.TriangleCount}>{splitContinuous.TriangleCount}.");
        if (splitContinuous.NonManifoldEdges != 0)
            throw new InvalidOperationException(
                $"Split-wall continuity produced {splitContinuous.NonManifoldEdges} non-manifold edges.");

        RunDirectionalOwnershipFixture(
            out ScanCoverDirectionalTsdfShadow.MeshResult directionLocal,
            out ScanCoverDirectionalTsdfShadow.MeshResult directionContinuous);
        int localDirections = CountActiveDirections(directionLocal);
        int continuousDirections = CountActiveDirections(directionContinuous);
        if (localDirections < 2 || continuousDirections != 1)
            throw new InvalidOperationException(
                $"One oblique physical wall retained visible direction partitions: " +
                $"directions={localDirections}>{continuousDirections} " +
                $"components={directionContinuous.SurfaceComponents}.");
        if (directionContinuous.NonManifoldEdges != 0)
            throw new InvalidOperationException(
                $"Directional ownership fixture produced " +
                $"{directionContinuous.NonManifoldEdges} non-manifold edges.");

        ScanCoverDirectionalTsdfShadow.MeshResult parallelStep =
            RunParallelStepPreservationFixture();
        if (parallelStep.SurfaceComponents < 2)
            throw new InvalidOperationException(
                $"A real parallel depth step was merged into one surface: " +
                $"components={parallelStep.SurfaceComponents}.");
        if (parallelStep.NonManifoldEdges != 0)
            throw new InvalidOperationException(
                $"Parallel-step fixture produced {parallelStep.NonManifoldEdges} non-manifold edges.");

        ScanCoverDirectionalTsdfShadow.MeshResult obliquePhysicalMerge =
            RunObliquePhysicalSurfaceMergeFixture();
        if (obliquePhysicalMerge.PhysicalSurfaceLocalMerges <= 0)
            throw new InvalidOperationException(
                "Oblique canonical evidence was not merged into one physical " +
                "surface before triangulation.");
        if (obliquePhysicalMerge.NonManifoldEdges != 0)
            throw new InvalidOperationException(
                $"Oblique physical-surface merge produced " +
                $"{obliquePhysicalMerge.NonManifoldEdges} non-manifold edges.");

        RunMatureDepthCorrectionFixture(
            out float oldSurfaceDepth,
            out float correctedSurfaceDepth,
            out ScanCoverDirectionalTsdfShadow.Metrics heldCorrection,
            out ScanCoverDirectionalTsdfShadow.Metrics appliedCorrection);
        if (heldCorrection.BatchMatureDepthConflictHeld <= 0 ||
            heldCorrection.BatchMatureDepthCorrections != 0)
            throw new InvalidOperationException(
                $"A one-batch mature-depth challenge was not held conservatively: " +
                $"held={heldCorrection.BatchMatureDepthConflictHeld} " +
                $"corrected={heldCorrection.BatchMatureDepthCorrections}.");
        if (appliedCorrection.BatchMatureDepthCorrections <= 0 ||
            appliedCorrection.BatchSaturatedRollingUpdates <= 0)
            throw new InvalidOperationException(
                $"Confirmed mature-depth evidence did not correct capped voxels: " +
                $"corrected={appliedCorrection.BatchMatureDepthCorrections} " +
                $"rolling={appliedCorrection.BatchSaturatedRollingUpdates}.");
        if (correctedSurfaceDepth <= oldSurfaceDepth + 0.0075f)
            throw new InvalidOperationException(
                $"Confirmed depth correction did not move the zero crossing: " +
                $"old={oldSurfaceDepth:F4} corrected={correctedSurfaceDepth:F4}.");

        RunSameBatchFastDepthCorrectionFixture(
            out float fastOldSurfaceDepth,
            out float fastCorrectedSurfaceDepth,
            out ScanCoverDirectionalTsdfShadow.Metrics fastCorrection);
        if (fastCorrection.BatchMatureDepthFastConfirmed <= 0 ||
            fastCorrection.BatchMatureDepthFastCandidates <= 0)
            throw new InvalidOperationException(
                "Three independent coplanar frames did not activate the " +
                "same-batch mature-depth correction path.");
        if (fastCorrection.BatchMatureDepthSignFlips <= 0)
            throw new InvalidOperationException(
                "Four independent coplanar frames with shared-neighbor " +
                "support did not advance the zero crossing.");
        if (fastCorrectedSurfaceDepth <= fastOldSurfaceDepth + 0.001f)
            throw new InvalidOperationException(
                $"Same-batch bounded correction did not move the surface: " +
                $"old={fastOldSurfaceDepth:F6} " +
                $"corrected={fastCorrectedSurfaceDepth:F6} " +
                $"candidate/confirmed/held/shared/risk=" +
                $"{fastCorrection.BatchMatureDepthFastCandidates}/" +
                $"{fastCorrection.BatchMatureDepthFastConfirmed}/" +
                $"{fastCorrection.BatchMatureDepthFastHeld}/" +
                $"{fastCorrection.BatchMatureDepthFastSharedNeighbors}/" +
                $"{fastCorrection.BatchMatureDepthFastRiskDeferred}.");

        ScanCoverDirectionalTsdfShadow.MeshResult fastCornerIsolation =
            RunFastCorrectionCornerIsolationFixture();
        if (CountActiveDirections(fastCornerIsolation) < 2 ||
            fastCornerIsolation.SurfaceComponents < 2)
            throw new InvalidOperationException(
                "Same-batch correction joined two adjacent non-coplanar " +
                "corner surfaces.");
        if (fastCornerIsolation.NonManifoldEdges != 0)
            throw new InvalidOperationException(
                $"Same-batch corner correction produced " +
                $"{fastCornerIsolation.NonManifoldEdges} non-manifold edges.");

        RunCrossVoxelPatchCorrectionFixture(
            out ScanCoverDirectionalTsdfShadow.Metrics crossVoxelPatch,
            out ScanCoverDirectionalTsdfShadow.MeshResult crossVoxelComposition);
        if (crossVoxelPatch.BatchMatureDepthPatchConfirmed <= 0 ||
            crossVoxelPatch.BatchMatureDepthPatchSharedNeighbors <= 0)
            throw new InvalidOperationException(
                "A consecutive physical-surface challenge that moved into " +
                "neighboring voxels was not confirmed.");
        if (crossVoxelComposition.PhysicalSurfaceCertificateCandidatePairs <= 0 ||
            crossVoxelComposition.PhysicalSurfaceCertificateLinks <= 0 ||
            crossVoxelComposition.PhysicalSurfaceCertificateIndexedNodePairs <= 0)
            throw new InvalidOperationException(
                "Confirmed cross-voxel physical-surface evidence did not enter " +
                "component composition: candidates/links/indexed/relaxed/rejected=" +
                $"{crossVoxelComposition.PhysicalSurfaceCertificateCandidatePairs}/" +
                $"{crossVoxelComposition.PhysicalSurfaceCertificateLinks}/" +
                $"{crossVoxelComposition.PhysicalSurfaceCertificateIndexedNodePairs}/" +
                $"{crossVoxelComposition.PhysicalSurfaceCertificateRelaxedLinks}/" +
                $"{crossVoxelComposition.PhysicalSurfaceCertificateRejectedPairs}.");
        if (crossVoxelComposition.PhysicalSurfaceCertificateDirectLinks != 0 ||
            crossVoxelComposition.PhysicalSurfaceIdentityEdgeMerges != 0)
            throw new InvalidOperationException(
                "Real-device rollback failed: unstable direct identity paths " +
                "must remain out of production: direct/edge=" +
                $"{crossVoxelComposition.PhysicalSurfaceCertificateDirectLinks}/" +
                $"{crossVoxelComposition.PhysicalSurfaceIdentityEdgeMerges}.");
        if (crossVoxelComposition.NonManifoldEdges != 0)
            throw new InvalidOperationException(
                "Certificate-guided composition produced " +
                $"{crossVoxelComposition.NonManifoldEdges} non-manifold edges.");

        RunCanonicalThreeFaceCornerFixture(
            out ScanCoverDirectionalTsdfShadow.MeshResult canonicalCorner,
            out ScanCoverDirectionalTsdfShadow.Metrics canonicalMetrics);
        if (!canonicalMetrics.CanonicalSixDirectionSemantics ||
            canonicalMetrics.BatchCanonicalDirectionWrites <= 0)
            throw new InvalidOperationException(
                "Canonical six-direction fusion did not report directional writes.");
        if (canonicalCorner.ThreePlusDirectionHypothesisCells <= 0 ||
            canonicalCorner.ThreePlusSurfaceClusterCells <= 0 ||
            canonicalCorner.MaximumSurfaceClustersPerCell < 3)
            throw new InvalidOperationException(
                $"A three-face corner was still reduced to two local surfaces: " +
                $"hyp3={canonicalCorner.ThreePlusDirectionHypothesisCells} " +
                $"cluster3={canonicalCorner.ThreePlusSurfaceClusterCells} " +
                $"max={canonicalCorner.MaximumSurfaceClustersPerCell}.");
        if (CountActiveDirections(canonicalCorner) < 3)
            throw new InvalidOperationException(
                $"Canonical three-face corner lost a direction before publication: " +
                $"directions={CountActiveDirections(canonicalCorner)}.");
        if (canonicalCorner.NonManifoldEdges != 0)
            throw new InvalidOperationException(
                $"Canonical three-face corner produced " +
                $"{canonicalCorner.NonManifoldEdges} non-manifold edges.");
        if (canonicalCorner.PhysicalSurfaceOneToOneLinks <= 0)
            throw new InvalidOperationException(
                "Canonical three-face corner did not exercise six-slot " +
                "one-to-one physical-surface linking.");
        RunRefinementProbeFixture(
            out ScanCoverDirectionalTsdfShadow.Metrics refinementMetrics);
        if (refinementMetrics.LastRefinementSameDirectionSpreadCells <= 0 ||
            refinementMetrics.LastRefinementHalfVoxelResolvableCells <= 0 ||
            refinementMetrics.LastRefinementCandidateBlocks <= 0 ||
            refinementMetrics.LastRefinementPersistentBlocks <= 0)
            throw new InvalidOperationException(
                $"Half-voxel refinement probe did not retain a persistent " +
                $"same-direction depth split: spread=" +
                $"{refinementMetrics.LastRefinementSameDirectionSpreadCells} " +
                $"half={refinementMetrics.LastRefinementHalfVoxelResolvableCells} " +
                $"blocks={refinementMetrics.LastRefinementCandidateBlocks}/" +
                $"{refinementMetrics.LastRefinementPersistentBlocks}.");
        if (!refinementMetrics.HalfVoxelShadowEnabled ||
            refinementMetrics.LastHalfVoxelShadowActiveBlocks <= 0 ||
            refinementMetrics.LastHalfVoxelShadowAllocatedBlocks <= 0 ||
            refinementMetrics.LastHalfVoxelShadowReplayedSamples <= 0 ||
            refinementMetrics.LastHalfVoxelShadowVoxelUpdates <= 0 ||
            refinementMetrics.LastHalfVoxelShadowZeroCrossingCells <= 0 ||
            refinementMetrics.LastHalfVoxelShadowPredictedCellsEvaluated <= 0 ||
            refinementMetrics.LastHalfVoxelScaledTruncationReplayedSamples <= 0 ||
            refinementMetrics.LastHalfVoxelScaledTruncationVoxelUpdates <= 0 ||
            refinementMetrics.LastHalfVoxelScaledTruncationZeroCrossingCells <= 0 ||
            refinementMetrics.LastHalfVoxelScaledTruncationPredictedCellsEvaluated <= 0)
            throw new InvalidOperationException(
                $"Real half-voxel shadow did not activate and replay evidence: " +
                $"enabled={refinementMetrics.HalfVoxelShadowEnabled} " +
                $"active/allocated=" +
                $"{refinementMetrics.LastHalfVoxelShadowActiveBlocks}/" +
                $"{refinementMetrics.LastHalfVoxelShadowAllocatedBlocks} " +
                $"replayed/updates=" +
                $"{refinementMetrics.LastHalfVoxelShadowReplayedSamples}/" +
                $"{refinementMetrics.LastHalfVoxelShadowVoxelUpdates} " +
                $"crossing/evaluated=" +
                $"{refinementMetrics.LastHalfVoxelShadowZeroCrossingCells}/" +
                $"{refinementMetrics.LastHalfVoxelShadowPredictedCellsEvaluated} " +
                $"scaled replay/updates/crossing/evaluated=" +
                $"{refinementMetrics.LastHalfVoxelScaledTruncationReplayedSamples}/" +
                $"{refinementMetrics.LastHalfVoxelScaledTruncationVoxelUpdates}/" +
                $"{refinementMetrics.LastHalfVoxelScaledTruncationZeroCrossingCells}/" +
                $"{refinementMetrics.LastHalfVoxelScaledTruncationPredictedCellsEvaluated}.");
        RunPaperProjectiveRefinementFixture(
            out ScanCoverDirectionalTsdfShadow.Metrics paperRefinementMetrics);
        if (paperRefinementMetrics.LastRefinementProbeEntries <= 0 ||
            (paperRefinementMetrics.LastRefinementCreaseCells <= 0 &&
             paperRefinementMetrics.LastRefinementDmcDoubleSurfaceCells <= 0) ||
            paperRefinementMetrics.LastRefinementCandidateBlocks <= 0 ||
            paperRefinementMetrics.LastRefinementPersistentBlocks <= 0 ||
            paperRefinementMetrics.LastHalfVoxelShadowBufferedSamples <= 0 ||
            paperRefinementMetrics.LastHalfVoxelShadowActiveBlocks <= 0)
        {
            throw new InvalidOperationException(
                "Paper projective TSDF bypassed the local half-voxel " +
                "refinement trigger: probe/crease/dmcDouble/blocks/persistent/" +
                "buffered/active=" +
                $"{paperRefinementMetrics.LastRefinementProbeEntries}/" +
                $"{paperRefinementMetrics.LastRefinementCreaseCells}/" +
                $"{paperRefinementMetrics.LastRefinementDmcDoubleSurfaceCells}/" +
                $"{paperRefinementMetrics.LastRefinementCandidateBlocks}/" +
                $"{paperRefinementMetrics.LastRefinementPersistentBlocks}/" +
                $"{paperRefinementMetrics.LastHalfVoxelShadowBufferedSamples}/" +
                $"{paperRefinementMetrics.LastHalfVoxelShadowActiveBlocks}.");
        }

        return
            "[ScanCoverDirectionalTsdfCompositionSmoke] PASS " +
            $"independentTriangles={independent.TriangleCount} " +
            $"composedTriangles={composed.TriangleCount} " +
            $"invalid={composed.InvalidDirectionHypotheses} " +
            $"collapsed={composed.ParallelHypothesesCollapsed} " +
            $"surfaceNodes/components/links={composed.SurfaceCandidateNodes}/" +
            $"{composed.SurfaceComponents}/{composed.SurfaceComponentLinks} " +
            $"surfaceEdgeMerges={composed.SurfaceConsistentEdgeMerges} " +
            $"dmcCells/raw/valid={composed.DirectionalMcShadowCellsEvaluated}/" +
            $"{composed.DirectionalMcShadowRawHypotheses}/" +
            $"{composed.DirectionalMcShadowValidHypotheses} " +
            $"dmcSingle/double/overflow=" +
            $"{composed.DirectionalMcShadowSingleSurfaceCells}/" +
            $"{composed.DirectionalMcShadowDoubleSurfaceCells}/" +
            $"{composed.DirectionalMcShadowOverflowDeferredComponents} " +
            $"dmcEdges={composed.DirectionalMcShadowRawTransitionEdges}/" +
            $"{composed.DirectionalMcShadowCombinedTransitionEdges}/" +
            $"{composed.DirectionalMcShadowRegularizedTransitionEdges} " +
            $"dmcNeighbor={composed.DirectionalMcShadowNeighborDisagreementsBefore}>" +
            $"{composed.DirectionalMcShadowNeighborDisagreementsAfter} " +
            $"physicalLocal/opposite/oneToOne=" +
            $"{composed.PhysicalSurfaceLocalMerges}/" +
            $"{composed.PhysicalSurfaceOppositeDirectionMerges}/" +
            $"{canonicalCorner.PhysicalSurfaceOneToOneLinks} " +
            $"creaseMerges={composed.CreaseJunctionEdgeMerges} " +
            $"splitBoundary={splitLocal.BoundaryEdges}>{splitContinuous.BoundaryEdges} " +
            $"splitInternal={splitLocalInternalBoundary}>{splitContinuousInternalBoundary} " +
            $"splitComponents={splitLocal.SurfaceComponents}>{splitContinuous.SurfaceComponents} " +
            $"obliqueDirections={localDirections}>{continuousDirections} " +
            $"parallelStepComponents={parallelStep.SurfaceComponents} " +
            $"obliquePhysicalMerges={obliquePhysicalMerge.PhysicalSurfaceLocalMerges} " +
            $"depthCorrection={oldSurfaceDepth:F4}>{correctedSurfaceDepth:F4} " +
            $"held/applied={heldCorrection.BatchMatureDepthConflictHeld}/" +
            $"{appliedCorrection.BatchMatureDepthCorrections} " +
            $"fastDepth={fastOldSurfaceDepth:F4}>{fastCorrectedSurfaceDepth:F4} " +
            $"fastCandidate/confirmed/shared=" +
            $"{fastCorrection.BatchMatureDepthFastCandidates}/" +
            $"{fastCorrection.BatchMatureDepthFastConfirmed}/" +
            $"{fastCorrection.BatchMatureDepthFastSharedNeighbors} " +
            $"fastCornerComponents={fastCornerIsolation.SurfaceComponents} " +
            $"patchConfirmed/shared=" +
            $"{crossVoxelPatch.BatchMatureDepthPatchConfirmed}/" +
            $"{crossVoxelPatch.BatchMatureDepthPatchSharedNeighbors} " +
            $"certCandidate/link/indexed/relaxed/reject=" +
            $"{crossVoxelComposition.PhysicalSurfaceCertificateCandidatePairs}/" +
            $"{crossVoxelComposition.PhysicalSurfaceCertificateLinks}/" +
            $"{crossVoxelComposition.PhysicalSurfaceCertificateIndexedNodePairs}/" +
            $"{crossVoxelComposition.PhysicalSurfaceCertificateRelaxedLinks}/" +
            $"{crossVoxelComposition.PhysicalSurfaceCertificateRejectedPairs} " +
            $"certDirectCandidate/link/reject=" +
            $"{crossVoxelComposition.PhysicalSurfaceCertificateDirectCandidates}/" +
            $"{crossVoxelComposition.PhysicalSurfaceCertificateDirectLinks}/" +
            $"{crossVoxelComposition.PhysicalSurfaceCertificateDirectRejected} " +
            $"identityEdgeCandidate/merge/reject=" +
            $"{crossVoxelComposition.PhysicalSurfaceIdentityEdgeCandidates}/" +
            $"{crossVoxelComposition.PhysicalSurfaceIdentityEdgeMerges}/" +
            $"{crossVoxelComposition.PhysicalSurfaceIdentityEdgeRejected} " +
            $"canonicalFanout={canonicalMetrics.BatchCanonicalFanoutOne}/" +
            $"{canonicalMetrics.BatchCanonicalFanoutTwo}/" +
            $"{canonicalMetrics.BatchCanonicalFanoutThree} " +
            $"canonicalThreePlus={canonicalCorner.ThreePlusDirectionHypothesisCells}/" +
            $"{canonicalCorner.ThreePlusSurfaceClusterCells} " +
            $"refinementSpread/half/blocks=" +
            $"{refinementMetrics.LastRefinementSameDirectionSpreadCells}/" +
            $"{refinementMetrics.LastRefinementHalfVoxelResolvableCells}/" +
            $"{refinementMetrics.LastRefinementCandidateBlocks} " +
            $"halfShadow={refinementMetrics.LastHalfVoxelShadowActiveBlocks}/" +
            $"{refinementMetrics.LastHalfVoxelShadowReplayedSamples}/" +
            $"{refinementMetrics.LastHalfVoxelShadowZeroCrossingCells} " +
            $"fineRecover/miss/extra=" +
            $"{refinementMetrics.LastHalfVoxelShadowRecoveredCells}/" +
            $"{refinementMetrics.LastHalfVoxelShadowMissingCells}/" +
            $"{refinementMetrics.LastHalfVoxelShadowExtraEnvelopeCells} " +
            $"scaledRecover/miss/extra=" +
            $"{refinementMetrics.LastHalfVoxelScaledTruncationRecoveredCells}/" +
            $"{refinementMetrics.LastHalfVoxelScaledTruncationMissingCells}/" +
            $"{refinementMetrics.LastHalfVoxelScaledTruncationExtraEnvelopeCells} " +
            $"conflict={composed.ConservativeConflictCells} " +
            $"nonManifoldDropped={composed.NonManifoldTrianglesDropped}";
    }

    private static void RunRefinementProbeFixture(
        out ScanCoverDirectionalTsdfShadow.Metrics metrics)
    {
        ScanCoverDirectionalTsdfShadow volume =
            new ScanCoverDirectionalTsdfShadow();
        const float voxel = 0.10f;
        volume.Configure(
            20, 20, 20,
            new Vector3(-1f, -1f, -1f),
            voxel, 0.20f, 32f, 8,
            true, true, true);

        for (int batch = 0; batch < 2; batch++)
        {
            if (batch > 0)
                volume.BeginBatch();
            for (int repeat = 0; repeat < 4; repeat++)
            {
                float lateral = 0.015f + repeat * 0.004f;
                volume.Integrate(
                    new Vector3(-1f, 0f, 0f),
                    new Vector3(0.02f, lateral, lateral),
                    Vector3.left,
                    1f, 0, -1, true, false, 0.12f, 3);
                volume.Integrate(
                    new Vector3(-1f, 0f, 0f),
                    new Vector3(0.07f, lateral, lateral),
                    Vector3.left,
                    1f, 0, -1, true, false, 0.12f, 3);
            }
            volume.BuildComposedMesh(
                0.25f, 10000, 20000,
                0.15f, 0.82f, 0.35f,
                false, false, 0.94f, 0.40f);
        }
        metrics = volume.GetMetrics();
    }

    private static void RunPaperProjectiveRefinementFixture(
        out ScanCoverDirectionalTsdfShadow.Metrics metrics)
    {
        ScanCoverDirectionalTsdfShadow volume =
            new ScanCoverDirectionalTsdfShadow();
        const float voxel = 0.10f;
        volume.Configure(
            20, 20, 20,
            new Vector3(-1f, -1f, -1f),
            voxel, 0.20f, 32f, 8,
            true, true, true);

        Vector3 eye = new Vector3(-1f, -1f, -0.5f);
        for (int batch = 0; batch < 2; batch++)
        {
            if (batch > 0)
                volume.BeginBatch();
            for (int v = -4; v <= 4; v++)
            for (int u = -4; u <= 4; u++)
            {
                float firstTangent = u * 0.05f + 0.012f;
                float secondTangent = v * 0.05f + 0.012f;
                volume.IntegratePaperNormalRaycast(
                    eye,
                    new Vector3(0.02f, firstTangent, secondTangent),
                    Vector3.left,
                    0.20f);
                volume.IntegratePaperNormalRaycast(
                    eye,
                    new Vector3(firstTangent, 0.02f, secondTangent),
                    Vector3.down,
                    0.20f);
            }
            volume.BuildComposedMesh(
                0.001f, 20000, 40000,
                0.15f, 0.82f, 0.35f,
                false, false, 0.94f, 0.40f);
        }
        metrics = volume.GetMetrics();
    }

    private static void RunCanonicalThreeFaceCornerFixture(
        out ScanCoverDirectionalTsdfShadow.MeshResult mesh,
        out ScanCoverDirectionalTsdfShadow.Metrics metrics)
    {
        ScanCoverDirectionalTsdfShadow volume = new ScanCoverDirectionalTsdfShadow();
        const float voxel = 0.025f;
        volume.Configure(
            40, 40, 40,
            new Vector3(-0.5f, -0.5f, -0.5f),
            voxel, 0.075f, 24f, 8,
            true);

        for (int observation = 0; observation < 3; observation++)
        {
            for (int v = -10; v <= 2; v++)
            for (int u = -10; u <= 2; u++)
            {
                volume.Integrate(
                    new Vector3(-1f, 0f, 0f),
                    new Vector3(0f, u * voxel, v * voxel),
                    Vector3.left,
                    1f, 0, -1, true, false, 0.12f, 3);
                volume.Integrate(
                    new Vector3(0f, -1f, 0f),
                    new Vector3(u * voxel, 0f, v * voxel),
                    Vector3.down,
                    1f, 0, -1, true, false, 0.12f, 3);
                volume.Integrate(
                    new Vector3(0f, 0f, -1f),
                    new Vector3(u * voxel, v * voxel, 0f),
                    Vector3.back,
                    1f, 0, -1, true, false, 0.12f, 3);
            }
        }

        // An oblique observation must fan out into two compatible channels,
        // proving that canonical routing is not dominant-axis ownership under
        // a different name.
        Vector3 diagonalNormal = new Vector3(-1f, 0f, -1f).normalized;
        volume.Integrate(
            diagonalNormal * 1.5f,
            new Vector3(voxel * 6f, voxel * 6f, voxel * 6f),
            diagonalNormal,
            1f, 0, -1, false, false, 0.12f, 3);

        mesh = volume.BuildComposedMesh(
            0.75f, 200000, 300000,
            0.15f, 0.82f, 0.35f,
            true, false, 0.94f, 0.40f);
        metrics = volume.GetMetrics();
        if (metrics.BatchCanonicalFanoutTwo <= 0)
            throw new InvalidOperationException(
                "Oblique canonical evidence did not fan out into two directions.");
    }

    private static ScanCoverDirectionalTsdfShadow.MeshResult
        RunObliquePhysicalSurfaceMergeFixture()
    {
        ScanCoverDirectionalTsdfShadow volume =
            new ScanCoverDirectionalTsdfShadow();
        const float voxel = 0.025f;
        volume.Configure(
            40, 40, 40,
            new Vector3(-0.5f, -0.5f, -0.5f),
            voxel, 0.075f, 24f, 8,
            true);

        Vector3 normal = new Vector3(-1f, 0f, -1f).normalized;
        Vector3 tangent = new Vector3(1f, 0f, -1f).normalized;
        for (int observation = 0; observation < 3; observation++)
        for (int v = -9; v <= 9; v++)
        for (int u = -9; u <= 9; u++)
        {
            Vector3 point = Vector3.up * (u * voxel) +
                            tangent * (v * voxel);
            volume.Integrate(
                point + normal * 1.5f,
                point,
                normal,
                1f, 0, -1, false, false, 0.12f, 3);
        }

        return volume.BuildComposedMesh(
            0.75f, 200000, 300000,
            0.15f, 0.82f, 0.35f,
            true, false, 0.94f, 0.40f);
    }

    private static void RunSplitSurfaceContinuityFixture(
        out ScanCoverDirectionalTsdfShadow.MeshResult local,
        out ScanCoverDirectionalTsdfShadow.MeshResult continuous)
    {
        ScanCoverDirectionalTsdfShadow volume = new ScanCoverDirectionalTsdfShadow();
        const float voxel = 0.025f;
        volume.Configure(
            40, 32, 24,
            new Vector3(-0.5f, -0.4f, -0.3f),
            voxel, 0.075f, 32f, 8);

        // One physical Z- wall is deliberately split between the primary and
        // secondary certificate slots.  A cell-local extractor sees a seam;
        // surface continuity must connect it without changing the wall depth.
        for (int y = -8; y <= 8; y++)
        for (int x = -12; x <= 12; x++)
        {
            bool secondary = x >= 0;
            // 9.5 mm of slot-dependent drift stays below the conservative
            // physical-surface gate but exceeds the old edge-only merge gate.
            Vector3 point = new Vector3(
                x * voxel, y * voxel, secondary ? voxel * 0.38f : 0f);
            for (int observation = 0; observation < 3; observation++)
            {
                volume.Integrate(
                    new Vector3(0f, 0f, -1f), point, Vector3.back,
                    1f, 6, -1, false, secondary, 0.12f, 3);
            }
        }

        local = volume.BuildComposedMesh(
            0.75f, 200000, 300000,
            0.15f, 0.82f, 0.35f,
            false, false, 0.94f, 0.40f);
        continuous = volume.BuildComposedMesh(
            0.75f, 200000, 300000,
            0.15f, 0.82f, 0.35f,
            true, false, 0.94f, 0.40f);
    }

    private static int CountBoundaryEdgesNearX(
        ScanCoverDirectionalTsdfShadow.MeshResult mesh,
        float targetX,
        float radius)
    {
        System.Collections.Generic.Dictionary<ulong, int> edgeUse =
            new System.Collections.Generic.Dictionary<ulong, int>();
        for (int direction = 0;
             direction < ScanCoverDirectionalTsdfShadow.DirectionCount;
             direction++)
        {
            System.Collections.Generic.List<int> triangles = mesh.TrianglesByDirection[direction];
            for (int triangle = 0; triangle + 2 < triangles.Count; triangle += 3)
            {
                AddEdgeUse(triangles[triangle], triangles[triangle + 1], edgeUse);
                AddEdgeUse(triangles[triangle + 1], triangles[triangle + 2], edgeUse);
                AddEdgeUse(triangles[triangle + 2], triangles[triangle], edgeUse);
            }
        }
        int count = 0;
        foreach (System.Collections.Generic.KeyValuePair<ulong, int> pair in edgeUse)
        {
            if (pair.Value != 1)
                continue;
            int a = unchecked((int)(pair.Key >> 32));
            int b = unchecked((int)(pair.Key & 0xffffffffu));
            float midpointX = (mesh.Vertices[a].x + mesh.Vertices[b].x) * 0.5f;
            if (Mathf.Abs(midpointX - targetX) <= radius)
                count++;
        }
        return count;
    }

    private static void RunDirectionalOwnershipFixture(
        out ScanCoverDirectionalTsdfShadow.MeshResult local,
        out ScanCoverDirectionalTsdfShadow.MeshResult continuous)
    {
        ScanCoverDirectionalTsdfShadow volume = new ScanCoverDirectionalTsdfShadow();
        const float voxel = 0.025f;
        volume.Configure(
            48, 32, 48,
            new Vector3(-0.6f, -0.4f, -0.6f),
            voxel, 0.075f, 32f, 8);
        Vector3 normal = new Vector3(1f, 0f, 1f).normalized;
        Vector3 tangent = new Vector3(1f, 0f, -1f).normalized;
        for (int y = -8; y <= 8; y++)
        for (int u = -12; u <= 12; u++)
        {
            Vector3 point = tangent * (u * voxel) + Vector3.up * (y * voxel);
            int directionChannel = u < 0 ? 1 : 5;
            for (int observation = 0; observation < 3; observation++)
            {
                volume.Integrate(
                    point + normal, point, normal,
                    1f, directionChannel, -1, false, false, 0.12f, 3);
            }
        }
        local = volume.BuildComposedMesh(
            0.75f, 200000, 300000,
            0.15f, 0.82f, 0.35f,
            false, false, 0.94f, 0.40f);
        continuous = volume.BuildComposedMesh(
            0.75f, 200000, 300000,
            0.15f, 0.82f, 0.35f,
            true, false, 0.94f, 0.40f);
    }

    private static int CountActiveDirections(
        ScanCoverDirectionalTsdfShadow.MeshResult mesh)
    {
        int count = 0;
        for (int direction = 0;
             direction < ScanCoverDirectionalTsdfShadow.DirectionCount;
             direction++)
        {
            if (mesh.TrianglesByDirection[direction].Count > 0)
                count++;
        }
        return count;
    }

    private static ScanCoverDirectionalTsdfShadow.MeshResult RunParallelStepPreservationFixture()
    {
        ScanCoverDirectionalTsdfShadow volume = new ScanCoverDirectionalTsdfShadow();
        const float voxel = 0.025f;
        volume.Configure(
            40, 32, 28,
            new Vector3(-0.5f, -0.4f, -0.35f),
            voxel, 0.075f, 32f, 8);
        for (int y = -8; y <= 8; y++)
        for (int x = -12; x <= 12; x++)
        {
            bool raised = x >= 0;
            Vector3 point = new Vector3(
                x * voxel, y * voxel, raised ? voxel * 1.2f : 0f);
            for (int observation = 0; observation < 3; observation++)
            {
                volume.Integrate(
                    new Vector3(0f, 0f, -1f), point, Vector3.back,
                    1f, 6, -1, true, raised, 0.12f, 3);
            }
        }
        return volume.BuildComposedMesh(
            0.75f, 200000, 300000,
            0.15f, 0.82f, 0.35f,
            true, false, 0.94f, 0.40f);
    }

    private static void RunMatureDepthCorrectionFixture(
        out float oldSurfaceDepth,
        out float correctedSurfaceDepth,
        out ScanCoverDirectionalTsdfShadow.Metrics heldCorrection,
        out ScanCoverDirectionalTsdfShadow.Metrics appliedCorrection)
    {
        ScanCoverDirectionalTsdfShadow volume = new ScanCoverDirectionalTsdfShadow();
        const float voxel = 0.05f;
        volume.Configure(
            28, 28, 24,
            new Vector3(-0.7f, -0.7f, -0.45f),
            voxel, 0.15f, 4f, 8);

        volume.BeginBatch();
        IntegrateCorrectionPlane(volume, 0f, voxel, 8);
        ScanCoverDirectionalTsdfShadow.MeshResult oldMesh = volume.BuildComposedMesh(
            0.75f, 200000, 300000,
            0.15f, 0.82f, 0.35f,
            true, false, 0.94f, 0.40f);
        oldSurfaceDepth = AverageVertexZ(oldMesh);

        // First contradictory batch is evidence, not permission to erase a
        // mature wall.  The second consecutive batch authorizes rolling
        // replacement at capped weight.
        volume.BeginBatch();
        IntegrateCorrectionPlane(volume, voxel * 1.2f, voxel, 4);
        heldCorrection = volume.GetMetrics();

        volume.BeginBatch();
        IntegrateCorrectionPlane(volume, voxel * 1.2f, voxel, 10);
        appliedCorrection = volume.GetMetrics();
        ScanCoverDirectionalTsdfShadow.MeshResult correctedMesh = volume.BuildComposedMesh(
            0.75f, 200000, 300000,
            0.15f, 0.82f, 0.35f,
            true, false, 0.94f, 0.40f);
        if (correctedMesh.NonManifoldEdges != 0)
            throw new InvalidOperationException(
                $"Depth correction produced {correctedMesh.NonManifoldEdges} non-manifold edges.");
        correctedSurfaceDepth = AverageVertexZ(correctedMesh);
    }

    private static void IntegrateCorrectionPlane(
        ScanCoverDirectionalTsdfShadow volume,
        float depth,
        float voxel,
        int observations)
    {
        for (int y = -6; y <= 6; y++)
        for (int x = -6; x <= 6; x++)
        {
            Vector3 point = new Vector3(x * voxel, y * voxel, depth);
            for (int observation = 0; observation < observations; observation++)
            {
                volume.Integrate(
                    new Vector3(0f, 0f, -1f), point, Vector3.back,
                    1f, 6, -1, false, false, 0.12f, 3);
            }
        }
    }

    private static void RunSameBatchFastDepthCorrectionFixture(
        out float oldSurfaceDepth,
        out float correctedSurfaceDepth,
        out ScanCoverDirectionalTsdfShadow.Metrics metrics)
    {
        ScanCoverDirectionalTsdfShadow volume =
            new ScanCoverDirectionalTsdfShadow();
        const float voxel = 0.05f;
        volume.Configure(
            28, 28, 24,
            new Vector3(-0.7f, -0.7f, -0.45f),
            voxel, 0.15f, 4f, 8);

        volume.BeginBatch();
        IntegrateCorrectionPlane(volume, 0f, voxel, 8);
        ScanCoverDirectionalTsdfShadow.MeshResult oldMesh =
            volume.BuildComposedMesh(
                0.75f, 200000, 300000,
                0.15f, 0.82f, 0.35f,
                true, false, 0.94f, 0.40f);
        oldSurfaceDepth = AverageVertexZ(oldMesh);

        volume.BeginBatch();
        for (int frame = 0; frame < 4; frame++)
            IntegrateCorrectionPlaneFrame(
                volume, voxel * 1.5f, voxel, 100 + frame);
        metrics = volume.GetMetrics();
        ScanCoverDirectionalTsdfShadow.MeshResult correctedMesh =
            volume.BuildComposedMesh(
                0.75f, 200000, 300000,
                0.15f, 0.82f, 0.35f,
                true, false, 0.94f, 0.40f);
        correctedSurfaceDepth = AverageVertexZ(correctedMesh);
    }

    private static void IntegrateCorrectionPlaneFrame(
        ScanCoverDirectionalTsdfShadow volume,
        float depth,
        float voxel,
        int frameIndex)
    {
        for (int y = -6; y <= 6; y++)
        for (int x = -6; x <= 6; x++)
        {
            Vector3 point = new Vector3(x * voxel, y * voxel, depth);
            volume.Integrate(
                new Vector3(0f, 0f, -1f), point, Vector3.back,
                1f, 6, -1, false, false, 0.12f, 3,
                frameIndex);
        }
    }

    private static ScanCoverDirectionalTsdfShadow.MeshResult
        RunFastCorrectionCornerIsolationFixture()
    {
        ScanCoverDirectionalTsdfShadow volume =
            new ScanCoverDirectionalTsdfShadow();
        const float voxel = 0.05f;
        volume.Configure(
            28, 28, 28,
            new Vector3(-0.7f, -0.7f, -0.7f),
            voxel, 0.15f, 4f, 8);

        volume.BeginBatch();
        for (int observation = 0; observation < 8; observation++)
            IntegrateCorrectionCornerFrame(volume, 0f, voxel, -1);

        volume.BeginBatch();
        for (int frame = 0; frame < 4; frame++)
            IntegrateCorrectionCornerFrame(
                volume, voxel * 1.5f, voxel, 200 + frame);

        ScanCoverDirectionalTsdfShadow.Metrics metrics = volume.GetMetrics();
        if (metrics.BatchMatureDepthFastConfirmed <= 0)
            throw new InvalidOperationException(
                "Corner isolation fixture did not exercise fast correction.");
        return volume.BuildComposedMesh(
            0.75f, 200000, 300000,
            0.15f, 0.82f, 0.35f,
            true, false, 0.94f, 0.40f);
    }

    private static void IntegrateCorrectionCornerFrame(
        ScanCoverDirectionalTsdfShadow volume,
        float shift,
        float voxel,
        int frameIndex)
    {
        for (int v = -5; v <= 5; v++)
        for (int u = -5; u <= 5; u++)
        {
            volume.Integrate(
                new Vector3(-1f, 0f, 0f),
                new Vector3(shift, u * voxel, v * voxel),
                Vector3.left,
                1f, 2, -1, false, false, 0.12f, 3,
                frameIndex);
            volume.Integrate(
                new Vector3(0f, 0f, -1f),
                new Vector3(u * voxel, v * voxel, shift),
                Vector3.back,
                1f, 6, -1, false, false, 0.12f, 3,
                frameIndex);
        }
    }

    private static void RunCrossVoxelPatchCorrectionFixture(
        out ScanCoverDirectionalTsdfShadow.Metrics metrics,
        out ScanCoverDirectionalTsdfShadow.MeshResult composition)
    {
        ScanCoverDirectionalTsdfShadow volume =
            new ScanCoverDirectionalTsdfShadow();
        const float voxel = 0.05f;
        volume.Configure(
            28, 28, 24,
            new Vector3(-0.7f, -0.7f, -0.45f),
            voxel, 0.15f, 4f, 8);

        volume.BeginBatch();
        IntegrateCorrectionPlane(volume, 0f, voxel, 8);

        volume.BeginBatch();
        IntegrateSparseCorrectionPlaneFrame(
            volume, voxel * 1.5f, voxel, 300, 0f);
        volume.BeginBatch();
        IntegrateSparseCorrectionPlaneFrame(
            volume, voxel * 1.5f, voxel, 400, voxel);
        metrics = volume.GetMetrics();
        composition = volume.BuildComposedMesh(
            0.75f, 200000, 300000,
            0.15f, 0.82f, 0.35f,
            true, false, 0.94f, 0.40f);
    }

    private static void IntegrateSparseCorrectionPlaneFrame(
        ScanCoverDirectionalTsdfShadow volume,
        float depth,
        float voxel,
        int frameIndex,
        float tangentOffset)
    {
        for (int y = -4; y <= 4; y += 2)
        for (int x = -4; x <= 4; x += 2)
        {
            Vector3 point = new Vector3(
                x * voxel * 2f + tangentOffset,
                y * voxel * 2f,
                depth);
            volume.Integrate(
                new Vector3(0f, 0f, -1f), point, Vector3.back,
                1f, 6, -1, false, false, 0.12f, 3,
                frameIndex);
        }
    }

    private static float AverageVertexZ(ScanCoverDirectionalTsdfShadow.MeshResult mesh)
    {
        if (mesh.Vertices.Count == 0)
            throw new InvalidOperationException("Correction fixture produced no surface vertices.");
        float sum = 0f;
        for (int vertex = 0; vertex < mesh.Vertices.Count; vertex++)
            sum += mesh.Vertices[vertex].z;
        return sum / mesh.Vertices.Count;
    }

    private static void AddEdgeUse(
        int a,
        int b,
        System.Collections.Generic.Dictionary<ulong, int> edgeUse)
    {
        uint minimum = (uint)Mathf.Min(a, b);
        uint maximum = (uint)Mathf.Max(a, b);
        ulong key = ((ulong)minimum << 32) | maximum;
        edgeUse.TryGetValue(key, out int use);
        edgeUse[key] = use + 1;
    }
}
#endif
