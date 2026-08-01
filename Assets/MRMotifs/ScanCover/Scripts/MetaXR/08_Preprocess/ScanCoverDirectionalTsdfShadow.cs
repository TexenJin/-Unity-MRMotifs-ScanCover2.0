using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using UnityEngine;

/// <summary>
/// Sparse six-direction TSDF with a legacy two-slot rollback mode and a canonical
/// Directional TSDF mode. In canonical mode each observation is distributed to
/// every compatible X+/X-/Y+/Y-/Z+/Z- channel (up to three), so differently
/// oriented surfaces do not compete for one scalar TSDF value.
/// </summary>
public sealed class ScanCoverDirectionalTsdfShadow
{
    public const int DirectionCount = 6;
    private const int SurfaceSlotCount = 2;
    private const int LayerCount = DirectionCount * SurfaceSlotCount;
    private const float CanonicalDirectionThreshold = 0.38268343f;
    // Hard reference mode: the visible DMC shadow follows the IROS 2019
    // algorithm and the authors' published lookup tables.  The retired MC33,
    // certificate, lease, precompletion, quarantine and half-voxel rules remain
    // compiled for A/B diagnostics, but cannot affect this path.
    private const bool PaperReferenceDirectionalMcEnabled = true;
    // Keep adaptive half-voxel extraction diagnostic-only until a conforming
    // coarse/fine transition topology exists.  Publishing a partial fine patch
    // next to retained coarse cells creates the visible mixed-size seams.
    private static readonly bool PaperHalfVoxelProductionPromotionEnabled = false;
    private const int PaperPersistentTopologyConfirmationBatches = 2;
    private const int PaperPersistentMissingCellLeaseBatches = 2;
    private const int PaperCanonicalEdgeConfirmationBatches = 2;
    private const int PaperCanonicalEdgeLeaseBatches = 4;
    private const float PaperCanonicalEdgeJitterVoxels = 0.08f;
    private const float PaperCanonicalEdgePendingMatchVoxels = 0.05f;
    private const int PaperQefMinimumStableTopologyBatches = 2;
    private const int HalfVoxelShadowHaloCoarseVoxels = 1;
    private const int HalfVoxelShadowLeaseBatches = 2;
    private const int HalfVoxelShadowMaximumActiveBlocks = 8;
    private const int RefinementMinimumNormalFamilySamples = 4;
    private const float RefinementNormalFamilyMergeAbsDot = 0.94f;
    private const float RefinementCreaseMaximumAbsNormalDot = 0.75f;
    private const int FastDepthCorrectionMinimumIndependentFrames = 3;
    private const float FastDepthCorrectionMinimumNormalDot = 0.97f;
    private const float FastDepthCorrectionMaximumPlaneOffsetVoxels = 0.25f;
    private const float FastDepthCorrectionMaximumTangentDistanceVoxels = 2.0f;
    // Mature TSDF corrections must follow a physical surface when small pose
    // changes move its evidence into an adjacent voxel/directional channel.
    // This challenger never joins geometry: it only confirms that repeated
    // contradictory samples belong to one locally coplanar patch.
    private const int PersistentDepthChallengerLeaseBatches = 3;
    private const int PersistentDepthChallengerMinimumStableBatches = 2;
    private const int PersistentDepthChallengerRiskMinimumStableBatches = 3;
    private const int PersistentDepthChallengerMinimumIndependentFrames = 4;
    private const int PersistentDepthChallengerRiskMinimumIndependentFrames = 6;
    private const int PersistentDepthChallengerMinimumSpatialVoxels = 2;
    private const int PersistentDepthChallengerRiskMinimumSpatialVoxels = 3;
    private const float PersistentDepthChallengerMinimumNormalDot = 0.94f;
    private const float PersistentDepthChallengerMaximumPlaneOffsetVoxels = 0.35f;
    private const float PersistentDepthChallengerMaximumTangentDistanceVoxels = 2.5f;
    private const int PhysicalSurfaceCertificateLeaseBatches = 4;
    private const float CertifiedSurfaceLinkMinimumNormalDot = 0.90f;
    private const float CertifiedSurfaceLinkMaximumPlaneOffsetVoxels = 0.55f;
    // Real-device replay showed that temporal certificates are still moving
    // while local depth re-anchoring is active.  Keep both broader identity
    // paths dormant until a stable, cross-implementation certificate exists.
    private const bool CertifiedSurfaceDirectComponentLinkProductionEnabled = false;
    private const bool PhysicalIdentityEdgeOwnerProductionEnabled = false;
    // Two component roots may reach the same grid edge without ever sharing a
    // face (a sparse boundary or a one-cell ownership hand-off is enough).
    // Treat only near-identical, parallel crossings as one physical identity.
    // This is deliberately tighter than the ordinary surface-link gate: small
    // steps and thin parallel faces must remain separate.
    private const float PhysicalIdentityEdgeMinimumNormalDot = 0.94f;
    private const float PhysicalIdentityEdgeMaximumCrossingDeltaVoxels = 0.18f;
    private const float PhysicalIdentityEdgeMaximumPlaneOffsetVoxels = 0.22f;
    private const float DirectionalMcShadowOffsetSeparation = 0.35f;
    private const float DirectionalMcShadowRegularizationNormalDot = 0.85f;
    private const float DirectionalMcShadowRegularizationSupportRatio = 1.25f;
    private const int DmcFaceDecisionMinimumStableBatches = 2;
    private const int DmcFaceDecisionLeaseBatches = 2;
    private const float DmcFaceDecisionMinimumNormalDot = 0.90f;
    private const float DmcFaceDecisionMaximumPlaneOffsetVoxels = 0.40f;
    // A face certificate is useful evidence, but a sign without a measured
    // crossing position is not geometry.  Keep synthetic precompletion out of
    // the visible DMC mesh until a persistent shared-edge decision can supply
    // the corresponding intersection.
    private const bool DmcSyntheticFacePrecompletionEnabled = false;
    // Frozen thresholds from the final offline A/B.  This bridge is deliberately
    // read-only: paper DMC keeps every physical/inter-cell edge and QEF may only
    // replace the interior triangulation of one proven, closed cell patch.
    private const float HermiteQefFeatureAngleDegrees = 32f;
    private const float HermiteQefMinimumFamilySupportRatio = 0.12f;
    private const float HermiteQefSolveRankRatio = 0.08f;
    private const int HermiteQefMinimumFramesPerFamily = 3;
    private const int HermiteQefMinimumViewsPerFamily = 2;
    private const int HermiteQefMinimumPersistentSamplesPerFamily = 1;
    private const float HermiteQefMinimumCertificateRankRatio = 0.12f;
    private const float HermiteQefMinimumCellMarginRatio = 0.02f;
    private const float HermiteQefMinimumFamilyWeightRatio = 0.30f;
    private const float HermiteQefMinimumDisplacementRatio = 0.08f;

    public struct PaperSurfaceAlignmentSample
    {
        public float SignedResidualMeters;
        public float SupportWeight;
        public int SupportVoxelCount;
        public int SaturatedVoxelCount;
        public int DominantDirection;
    }

    public sealed class MeshResult
    {
        public readonly List<Vector3> Vertices = new List<Vector3>(32768);
        public readonly List<int>[] TrianglesByDirection = CreateDirectionLists(98304);
        public readonly List<int>[] LinesByDirection = CreateDirectionLists(131072);
        // The paper-style DMC extractor is deliberately isolated from the
        // composed/production mesh.  Unity may display these buffers, but they
        // are never consumed by the production publication chain.
        public readonly List<Vector3> DmcShadowVertices = new List<Vector3>(32768);
        public readonly List<int> DmcShadowTriangles = new List<int>(98304);
        public readonly List<int> DmcShadowLines = new List<int>(131072);
        // Cell ownership is internal audit data. A negative entry denotes a
        // promoted fine-cell triangle which the coarse QEF bridge must not edit.
        public readonly List<int> DmcShadowTriangleCells = new List<int>(32768);
        public readonly List<Vector3> HermiteQefShadowVertices =
            new List<Vector3>(32768);
        public readonly List<int> HermiteQefShadowTriangles =
            new List<int>(98304);
        public readonly List<int> HermiteQefShadowLines =
            new List<int>(131072);
        public int ScannedCells;
        public int ZeroCrossingCells;
        public int MultiDirectionZeroCrossingCells;
        public int ThreePlusDirectionHypothesisCells;
        public int ThreePlusSurfaceClusterCells;
        public int MaximumDirectionHypothesesPerCell;
        public int MaximumSurfaceClustersPerCell;
        public int SecondaryLayerZeroCrossingCells;
        public int SuppressedOppositeDirectionCells;
        public int SuppressedCoincidentSecondaryCells;
        public int PreservedCloseParallelCells;
        public int InvalidDirectionHypotheses;
        public int ObservedNormalHypotheses;
        public int ParallelHypothesesCollapsed;
        public int IncompatibleWeakHypothesesDropped;
        public int EdgeCrossingsMerged;
        public int EdgeCrossingOverflowDropped;
        public int ConservativeConflictCells;
        public int NonManifoldTrianglesDropped;
        public int SurfaceCandidateNodes;
        public int SurfaceComponents;
        public int SurfaceComponentLinks;
        public int SurfaceSingleCellGapLinks;
        public int SurfaceComponentSingletons;
        public int SurfaceGapBridgeCandidates;
        public int SurfaceGapBridgeTriangles;
        public int SurfaceConsistentEdgeMerges;
        public int CreaseJunctionEdgeMerges;
        public int CrossSurfaceNearCrossingsPreserved;
        public int SurfaceContinuityEdgeDisagreements;
        public int PhysicalSurfaceLocalMerges;
        public int PhysicalSurfaceOppositeDirectionMerges;
        public int PhysicalSurfaceOneToOneLinks;
        public int PhysicalSurfaceCertificateCandidatePairs;
        public int PhysicalSurfaceCertificateLinks;
        public int PhysicalSurfaceCertificateRelaxedLinks;
        public int PhysicalSurfaceCertificateRejectedPairs;
        public int PhysicalSurfaceCertificateAmbiguousPairs;
        public int PhysicalSurfaceCertificateIndexedNodePairs;
        public int PhysicalSurfaceCertificateDirectCandidates;
        public int PhysicalSurfaceCertificateDirectLinks;
        public int PhysicalSurfaceCertificateDirectRejected;
        public int PhysicalSurfaceIdentityEdgeCandidates;
        public int PhysicalSurfaceIdentityEdgeMerges;
        public int PhysicalSurfaceIdentityEdgeRejected;
        public bool DirectionalMcShadowEnabled;
        public bool DirectionalMcShadowPaperReference;
        public int DirectionalMcShadowCellsEvaluated;
        public int DirectionalMcShadowRawHypotheses;
        public int DirectionalMcShadowIncompleteCornerHypotheses;
        public int DirectionalMcShadowIntraDirectionRejected;
        public int DirectionalMcShadowInterDirectionRejected;
        public int DirectionalMcShadowValidHypotheses;
        public int DirectionalMcShadowComponents;
        public int DirectionalMcShadowSingleSurfaceCells;
        public int DirectionalMcShadowDoubleSurfaceCells;
        public int DirectionalMcShadowOverflowDeferredComponents;
        public int DirectionalMcShadowEmptyAfterVotingCells;
        public int DirectionalMcShadowRawTransitionEdges;
        public int DirectionalMcShadowCombinedTransitionEdges;
        public int DirectionalMcShadowRegularizedTransitionEdges;
        public int DirectionalMcShadowOffsetClusters;
        public int DirectionalMcShadowDualOffsetEdges;
        public int DirectionalMcShadowOffsetOverflowEdges;
        public int DirectionalMcShadowNeighborFaceComparisons;
        public int DirectionalMcShadowNeighborDisagreementsBefore;
        public int DirectionalMcShadowNeighborDisagreementsAfter;
        public int DirectionalMcShadowRegularizedCorners;
        public int DirectionalMcShadowRegularizedCells;
        public int DirectionalMcShadowRegularizationDeferredPairs;
        public int DirectionalMcShadowPersistentDecisions;
        public int DirectionalMcShadowChangedDecisions;
        public int DirectionalMcShadowRetiredDecisions;
        public int DirectionalMcShadowUnknownDeferredCells;
        public int DirectionalMcShadowAmbiguousFaces;
        public int DirectionalMcShadowInteriorTests;
        public int DirectionalMcShadowTriangles;
        public int DirectionalMcShadowVertices;
        public int DirectionalMcShadowFineRefinementBlocks;
        public int DirectionalMcShadowFineCellsEvaluated;
        public int DirectionalMcShadowFineCellsAccepted;
        public int DirectionalMcShadowFineCoarseCellsPromoted;
        public int DirectionalMcShadowFineBoundaryDeferredCells;
        public int DirectionalMcShadowFineCoarsePriorCorners;
        public int DirectionalMcShadowFineIncompleteCells;
        public int DirectionalMcShadowFineVotingEmptyCells;
        public int DirectionalMcShadowFineVertices;
        public int DirectionalMcShadowFineTriangles;
        public int DirectionalMcShadowOverflowRefinementBlocks;
        public int DirectionalMcShadowOverflowRefinementActiveBlocks;
        public int DirectionalMcShadowOverflowResolvedByHalfVoxel;
        public int DirectionalMcShadowOverflowStillUnresolved;
        public int DirectionalMcShadowFaceDecisionCandidates;
        public int DirectionalMcShadowFaceDecisionConflicts;
        public int DirectionalMcShadowFaceDecisionStable;
        public int DirectionalMcShadowFaceDecisionAppliedPairs;
        public int DirectionalMcShadowFaceDecisionPersistent;
        public int DirectionalMcShadowFaceDecisionRetired;
        public int DirectionalMcShadowFaceDecisionFilledCorners;
        public int DirectionalMcShadowFaceDecisionChangedCells;
        public int DirectionalMcShadowFaceDecisionPrecompletedCorners;
        public int DirectionalMcShadowFaceDecisionRecoveredHypotheses;
        public int DirectionalMcShadowBoundaryEdges;
        public int DirectionalMcShadowNonManifoldEdges;
        public int DirectionalMcShadowDuplicateTriangles;
        public int DirectionalMcShadowDegenerateTriangles;
        public int DirectionalMcShadowCrackCandidateEdges;
        public int DirectionalMcShadowConflictFaceDeferredTriangles;
        public int DirectionalMcShadowSharedFaceTopologyComparisons;
        public int DirectionalMcShadowSharedFaceTopologyMismatches;
        public int DirectionalMcShadowUnmeasuredEdgeDeferredCells;
        public int DirectionalMcShadowUnmeasuredEdgeDeferredTriangles;
        public int DirectionalMcShadowWindingCorrectedTriangles;
        public int DirectionalMcShadowNormalMismatchTriangles;
        public int DirectionalMcShadowPendingTopologyChanges;
        public int DirectionalMcShadowAtomicCommittedCells;
        public int DirectionalMcShadowAtomicDeferredCells;
        public int DirectionalMcShadowPersistentEdges;
        public int DirectionalMcShadowPersistentSurfaceIdentities;
        public int DirectionalMcShadowCanonicalEdgeCorrections;
        public int DirectionalMcShadowTopologyStableBatches;
        public double DirectionalMcShadowMilliseconds;
        public bool HermiteQefShadowEnabled;
        public bool HermiteQefProductionEligible;
        public int HermiteQefScannedCells;
        public int HermiteQefHermiteSamples;
        public int HermiteQefRawCandidates;
        public int HermiteQefFrameRejected;
        public int HermiteQefViewRejected;
        public int HermiteQefSampleRejected;
        public int HermiteQefFamilyBalanceRejected;
        public int HermiteQefRankRejected;
        public int HermiteQefCellMarginRejected;
        public int HermiteQefDisplacementRejected;
        public int HermiteQefResidualRejected;
        public int HermiteQefCertified;
        public int HermiteQefMissingPatchRejected;
        public int HermiteQefMultiPatchRejected;
        public int HermiteQefOpenBoundaryRejected;
        public int HermiteQefOrientationRejected;
        public int HermiteQefProvisionalAppliedCells;
        public int HermiteQefAppliedCells;
        public int HermiteQefSourceTrianglesReplaced;
        public int HermiteQefFeatureTrianglesAdded;
        public int HermiteQefBoundaryEdges;
        public int HermiteQefNonManifoldEdges;
        public int HermiteQefDuplicateTriangles;
        public int HermiteQefBoundaryEdgeDelta;
        public int HermiteQefNonManifoldEdgeDelta;
        public int HermiteQefDuplicateTriangleDelta;
        public int HermiteQefPreRollbackBoundaryEdges;
        public int HermiteQefPreRollbackNonManifoldEdges;
        public int HermiteQefPreRollbackDuplicateTriangles;
        public int HermiteQefPreRollbackBoundaryEdgeDelta;
        public int HermiteQefPreRollbackNonManifoldEdgeDelta;
        public int HermiteQefPreRollbackDuplicateTriangleDelta;
        public int HermiteQefTopologyRollback;
        public double HermiteQefMilliseconds;
        public int BoundaryEdges;
        public int NonManifoldEdges;
        public int DuplicateTriangles;
        public int RefinementProbeEntries;
        public int RefinementSameDirectionSpreadCells;
        public int RefinementCreaseCells;
        public int RefinementDmcDoubleSurfaceCells;
        public int RefinementDmcDoubleSurfaceBlocks;
        public int RefinementHalfVoxelResolvableCells;
        public int RefinementHalfVoxelInsufficientCells;
        public int RefinementCandidateBlocks;
        public int RefinementPersistentBlocks;
        public int RefinementDirtyBlocks;
        public int RefinementCleanBlocks;
        public float RefinementProjectedVoxelMultiplier;
        public bool RefinementBoundsValid;
        public Vector3 RefinementMinimumWorld;
        public Vector3 RefinementMaximumWorld;
        public int[] RefinementDepthSpanBuckets = new int[4];
        public int HalfVoxelShadowActiveBlocks;
        public int HalfVoxelShadowAllocatedBlocks;
        public int HalfVoxelShadowAllocatedLayers;
        public int HalfVoxelShadowWeightedVoxels;
        public int HalfVoxelShadowCandidateCells;
        public int HalfVoxelShadowBufferedSamples;
        public int HalfVoxelShadowReplayedSamples;
        public int HalfVoxelShadowVoxelUpdates;
        public int HalfVoxelShadowZeroCrossingCells;
        public int HalfVoxelShadowPredictedCellsEvaluated;
        public int HalfVoxelShadowCoarseEndpointResolvedCells;
        public int HalfVoxelShadowFineEndpointResolvedCells;
        public int HalfVoxelShadowRecoveredCells;
        public int HalfVoxelShadowMissingCells;
        public int HalfVoxelShadowExtraEnvelopeCells;
        public double HalfVoxelShadowIntegrationMilliseconds;
        public double HalfVoxelShadowEvaluationMilliseconds;
        public int HalfVoxelScaledTruncationAllocatedLayers;
        public int HalfVoxelScaledTruncationWeightedVoxels;
        public int HalfVoxelScaledTruncationCandidateCells;
        public int HalfVoxelScaledTruncationReplayedSamples;
        public int HalfVoxelScaledTruncationVoxelUpdates;
        public int HalfVoxelScaledTruncationZeroCrossingCells;
        public int HalfVoxelScaledTruncationPredictedCellsEvaluated;
        public int HalfVoxelScaledTruncationFineEndpointResolvedCells;
        public int HalfVoxelScaledTruncationRecoveredCells;
        public int HalfVoxelScaledTruncationMissingCells;
        public int HalfVoxelScaledTruncationExtraEnvelopeCells;
        public double HalfVoxelScaledTruncationIntegrationMilliseconds;
        public double HalfVoxelScaledTruncationEvaluationMilliseconds;
        public bool Truncated;

        public int TriangleCount
        {
            get
            {
                int count = 0;
                for (int i = 0; i < DirectionCount; i++)
                    count += TrianglesByDirection[i].Count / 3;
                return count;
            }
        }

        private static List<int>[] CreateDirectionLists(int totalCapacity)
        {
            List<int>[] result = new List<int>[DirectionCount];
            int capacity = Mathf.Max(16, totalCapacity / DirectionCount);
            for (int i = 0; i < DirectionCount; i++)
                result[i] = new List<int>(capacity);
            return result;
        }
    }

    public struct Metrics
    {
        public bool Configured;
        public bool CanonicalSixDirectionSemantics;
        public bool RefinementProbeEnabled;
        public bool HalfVoxelShadowEnabled;
        public int AllocatedBlocks;
        public int AllocatedDirectionLayers;
        public int WeightedVoxels;
        public int CandidateCells;
        public int BatchSamples;
        public int BatchContestedSamples;
        public int BatchSecondarySamples;
        public int BatchCanonicalDirectionWrites;
        public int BatchCanonicalFanoutOne;
        public int BatchCanonicalFanoutTwo;
        public int BatchCanonicalFanoutThree;
        public int BatchCanonicalRejectedSamples;
        public int BatchVoxelUpdates;
        public int BatchOwnerMatched;
        public int BatchOwnerChallenged;
        public int BatchOwnerSwitched;
        public int BatchRetiredDirectionVoxels;
        public int BatchSaturatedRollingUpdates;
        public int BatchMatureDepthConflictCandidates;
        public int BatchMatureDepthConflictHeld;
        public int BatchMatureDepthCorrections;
        public int BatchMatureDepthSignFlips;
        public int BatchMatureDepthFastCandidates;
        public int BatchMatureDepthFastConfirmed;
        public int BatchMatureDepthFastHeld;
        public int BatchMatureDepthFastSharedNeighbors;
        public int BatchMatureDepthFastRiskDeferred;
        public int BatchMatureDepthPatchCandidates;
        public int BatchMatureDepthPatchConfirmed;
        public int BatchMatureDepthPatchSharedNeighbors;
        public int BatchMatureDepthPersistentCandidates;
        public int BatchMatureDepthPersistentMatched;
        public int BatchMatureDepthPersistentCrossVoxelMatched;
        public int BatchMatureDepthPersistentConfirmed;
        public int BatchMatureDepthPersistentRiskConfirmed;
        public int PersistentDepthChallengerStates;
        public double BatchIntegrationMilliseconds;
        public double LastExtractionMilliseconds;
        public int LastScannedCells;
        public int LastZeroCrossingCells;
        public int LastMultiDirectionZeroCrossingCells;
        public int LastThreePlusDirectionHypothesisCells;
        public int LastThreePlusSurfaceClusterCells;
        public int LastMaximumDirectionHypothesesPerCell;
        public int LastMaximumSurfaceClustersPerCell;
        public int LastSecondaryLayerZeroCrossingCells;
        public int LastSuppressedOppositeDirectionCells;
        public int LastSuppressedCoincidentSecondaryCells;
        public int LastPreservedCloseParallelCells;
        public int LastInvalidDirectionHypotheses;
        public int LastObservedNormalHypotheses;
        public int LastParallelHypothesesCollapsed;
        public int LastIncompatibleWeakHypothesesDropped;
        public int LastEdgeCrossingsMerged;
        public int LastEdgeCrossingOverflowDropped;
        public int LastConservativeConflictCells;
        public int LastNonManifoldTrianglesDropped;
        public int LastSurfaceCandidateNodes;
        public int LastSurfaceComponents;
        public int LastSurfaceComponentLinks;
        public int LastSurfaceSingleCellGapLinks;
        public int LastSurfaceComponentSingletons;
        public int LastSurfaceGapBridgeCandidates;
        public int LastSurfaceGapBridgeTriangles;
        public int LastSurfaceConsistentEdgeMerges;
        public int LastCreaseJunctionEdgeMerges;
        public int LastCrossSurfaceNearCrossingsPreserved;
        public int LastSurfaceContinuityEdgeDisagreements;
        public int LastPhysicalSurfaceLocalMerges;
        public int LastPhysicalSurfaceOppositeDirectionMerges;
        public int LastPhysicalSurfaceOneToOneLinks;
        public int LastPhysicalSurfaceCertificateCandidatePairs;
        public int LastPhysicalSurfaceCertificateLinks;
        public int LastPhysicalSurfaceCertificateRelaxedLinks;
        public int LastPhysicalSurfaceCertificateRejectedPairs;
        public int LastPhysicalSurfaceCertificateAmbiguousPairs;
        public int LastPhysicalSurfaceCertificateIndexedNodePairs;
        public int LastPhysicalSurfaceCertificateDirectCandidates;
        public int LastPhysicalSurfaceCertificateDirectLinks;
        public int LastPhysicalSurfaceCertificateDirectRejected;
        public int LastPhysicalSurfaceIdentityEdgeCandidates;
        public int LastPhysicalSurfaceIdentityEdgeMerges;
        public int LastPhysicalSurfaceIdentityEdgeRejected;
        public bool DirectionalMcShadowEnabled;
        public bool DirectionalMcShadowPaperReference;
        public int LastDirectionalMcShadowCellsEvaluated;
        public int LastDirectionalMcShadowRawHypotheses;
        public int LastDirectionalMcShadowIncompleteCornerHypotheses;
        public int LastDirectionalMcShadowIntraDirectionRejected;
        public int LastDirectionalMcShadowInterDirectionRejected;
        public int LastDirectionalMcShadowValidHypotheses;
        public int LastDirectionalMcShadowComponents;
        public int LastDirectionalMcShadowSingleSurfaceCells;
        public int LastDirectionalMcShadowDoubleSurfaceCells;
        public int LastDirectionalMcShadowOverflowDeferredComponents;
        public int LastDirectionalMcShadowEmptyAfterVotingCells;
        public int LastDirectionalMcShadowRawTransitionEdges;
        public int LastDirectionalMcShadowCombinedTransitionEdges;
        public int LastDirectionalMcShadowRegularizedTransitionEdges;
        public int LastDirectionalMcShadowOffsetClusters;
        public int LastDirectionalMcShadowDualOffsetEdges;
        public int LastDirectionalMcShadowOffsetOverflowEdges;
        public int LastDirectionalMcShadowNeighborFaceComparisons;
        public int LastDirectionalMcShadowNeighborDisagreementsBefore;
        public int LastDirectionalMcShadowNeighborDisagreementsAfter;
        public int LastDirectionalMcShadowRegularizedCorners;
        public int LastDirectionalMcShadowRegularizedCells;
        public int LastDirectionalMcShadowRegularizationDeferredPairs;
        public int LastDirectionalMcShadowPersistentDecisions;
        public int LastDirectionalMcShadowChangedDecisions;
        public int LastDirectionalMcShadowRetiredDecisions;
        public int LastDirectionalMcShadowUnknownDeferredCells;
        public int LastDirectionalMcShadowAmbiguousFaces;
        public int LastDirectionalMcShadowInteriorTests;
        public int LastDirectionalMcShadowTriangles;
        public int LastDirectionalMcShadowVertices;
        public int LastDirectionalMcShadowFineRefinementBlocks;
        public int LastDirectionalMcShadowFineCellsEvaluated;
        public int LastDirectionalMcShadowFineCellsAccepted;
        public int LastDirectionalMcShadowFineCoarseCellsPromoted;
        public int LastDirectionalMcShadowFineBoundaryDeferredCells;
        public int LastDirectionalMcShadowFineCoarsePriorCorners;
        public int LastDirectionalMcShadowFineIncompleteCells;
        public int LastDirectionalMcShadowFineVotingEmptyCells;
        public int LastDirectionalMcShadowFineVertices;
        public int LastDirectionalMcShadowFineTriangles;
        public int LastDirectionalMcShadowOverflowRefinementBlocks;
        public int LastDirectionalMcShadowOverflowRefinementActiveBlocks;
        public int LastDirectionalMcShadowOverflowResolvedByHalfVoxel;
        public int LastDirectionalMcShadowOverflowStillUnresolved;
        public int LastDirectionalMcShadowFaceDecisionCandidates;
        public int LastDirectionalMcShadowFaceDecisionConflicts;
        public int LastDirectionalMcShadowFaceDecisionStable;
        public int LastDirectionalMcShadowFaceDecisionAppliedPairs;
        public int LastDirectionalMcShadowFaceDecisionPersistent;
        public int LastDirectionalMcShadowFaceDecisionRetired;
        public int LastDirectionalMcShadowFaceDecisionFilledCorners;
        public int LastDirectionalMcShadowFaceDecisionChangedCells;
        public int LastDirectionalMcShadowFaceDecisionPrecompletedCorners;
        public int LastDirectionalMcShadowFaceDecisionRecoveredHypotheses;
        public int LastDirectionalMcShadowBoundaryEdges;
        public int LastDirectionalMcShadowNonManifoldEdges;
        public int LastDirectionalMcShadowDuplicateTriangles;
        public int LastDirectionalMcShadowDegenerateTriangles;
        public int LastDirectionalMcShadowCrackCandidateEdges;
        public int LastDirectionalMcShadowConflictFaceDeferredTriangles;
        public int LastDirectionalMcShadowSharedFaceTopologyComparisons;
        public int LastDirectionalMcShadowSharedFaceTopologyMismatches;
        public int LastDirectionalMcShadowUnmeasuredEdgeDeferredCells;
        public int LastDirectionalMcShadowUnmeasuredEdgeDeferredTriangles;
        public int LastDirectionalMcShadowWindingCorrectedTriangles;
        public int LastDirectionalMcShadowNormalMismatchTriangles;
        public int LastDirectionalMcShadowPendingTopologyChanges;
        public int LastDirectionalMcShadowAtomicCommittedCells;
        public int LastDirectionalMcShadowAtomicDeferredCells;
        public int LastDirectionalMcShadowPersistentEdges;
        public int LastDirectionalMcShadowPersistentSurfaceIdentities;
        public int LastDirectionalMcShadowCanonicalEdgeCorrections;
        public int LastDirectionalMcShadowTopologyStableBatches;
        public double LastDirectionalMcShadowMilliseconds;
        public bool HermiteQefShadowEnabled;
        public bool HermiteQefProductionEligible;
        public int LastHermiteQefScannedCells;
        public int LastHermiteQefHermiteSamples;
        public int LastHermiteQefRawCandidates;
        public int LastHermiteQefFrameRejected;
        public int LastHermiteQefViewRejected;
        public int LastHermiteQefSampleRejected;
        public int LastHermiteQefFamilyBalanceRejected;
        public int LastHermiteQefRankRejected;
        public int LastHermiteQefCellMarginRejected;
        public int LastHermiteQefDisplacementRejected;
        public int LastHermiteQefResidualRejected;
        public int LastHermiteQefCertified;
        public int LastHermiteQefMissingPatchRejected;
        public int LastHermiteQefMultiPatchRejected;
        public int LastHermiteQefOpenBoundaryRejected;
        public int LastHermiteQefOrientationRejected;
        public int LastHermiteQefProvisionalAppliedCells;
        public int LastHermiteQefAppliedCells;
        public int LastHermiteQefSourceTrianglesReplaced;
        public int LastHermiteQefFeatureTrianglesAdded;
        public int LastHermiteQefBoundaryEdges;
        public int LastHermiteQefNonManifoldEdges;
        public int LastHermiteQefDuplicateTriangles;
        public int LastHermiteQefBoundaryEdgeDelta;
        public int LastHermiteQefNonManifoldEdgeDelta;
        public int LastHermiteQefDuplicateTriangleDelta;
        public int LastHermiteQefPreRollbackBoundaryEdges;
        public int LastHermiteQefPreRollbackNonManifoldEdges;
        public int LastHermiteQefPreRollbackDuplicateTriangles;
        public int LastHermiteQefPreRollbackBoundaryEdgeDelta;
        public int LastHermiteQefPreRollbackNonManifoldEdgeDelta;
        public int LastHermiteQefPreRollbackDuplicateTriangleDelta;
        public int LastHermiteQefTopologyRollback;
        public double LastHermiteQefMilliseconds;
        public int LastVertices;
        public int LastTriangles;
        public int LastBoundaryEdges;
        public int LastNonManifoldEdges;
        public int LastDuplicateTriangles;
        public int LastRefinementProbeEntries;
        public int LastRefinementSameDirectionSpreadCells;
        public int LastRefinementCreaseCells;
        public int LastRefinementDmcDoubleSurfaceCells;
        public int LastRefinementDmcDoubleSurfaceBlocks;
        public int LastRefinementHalfVoxelResolvableCells;
        public int LastRefinementHalfVoxelInsufficientCells;
        public int LastRefinementCandidateBlocks;
        public int LastRefinementPersistentBlocks;
        public int LastRefinementDirtyBlocks;
        public int LastRefinementCleanBlocks;
        public float LastRefinementProjectedVoxelMultiplier;
        public bool LastRefinementBoundsValid;
        public Vector3 LastRefinementMinimumWorld;
        public Vector3 LastRefinementMaximumWorld;
        public int[] LastRefinementDepthSpanBuckets;
        public int LastHalfVoxelShadowActiveBlocks;
        public int LastHalfVoxelShadowAllocatedBlocks;
        public int LastHalfVoxelShadowAllocatedLayers;
        public int LastHalfVoxelShadowWeightedVoxels;
        public int LastHalfVoxelShadowCandidateCells;
        public int LastHalfVoxelShadowBufferedSamples;
        public int LastHalfVoxelShadowReplayedSamples;
        public int LastHalfVoxelShadowVoxelUpdates;
        public int LastHalfVoxelShadowZeroCrossingCells;
        public int LastHalfVoxelShadowPredictedCellsEvaluated;
        public int LastHalfVoxelShadowCoarseEndpointResolvedCells;
        public int LastHalfVoxelShadowFineEndpointResolvedCells;
        public int LastHalfVoxelShadowRecoveredCells;
        public int LastHalfVoxelShadowMissingCells;
        public int LastHalfVoxelShadowExtraEnvelopeCells;
        public double LastHalfVoxelShadowIntegrationMilliseconds;
        public double LastHalfVoxelShadowEvaluationMilliseconds;
        public int LastHalfVoxelScaledTruncationAllocatedLayers;
        public int LastHalfVoxelScaledTruncationWeightedVoxels;
        public int LastHalfVoxelScaledTruncationCandidateCells;
        public int LastHalfVoxelScaledTruncationReplayedSamples;
        public int LastHalfVoxelScaledTruncationVoxelUpdates;
        public int LastHalfVoxelScaledTruncationZeroCrossingCells;
        public int LastHalfVoxelScaledTruncationPredictedCellsEvaluated;
        public int LastHalfVoxelScaledTruncationFineEndpointResolvedCells;
        public int LastHalfVoxelScaledTruncationRecoveredCells;
        public int LastHalfVoxelScaledTruncationMissingCells;
        public int LastHalfVoxelScaledTruncationExtraEnvelopeCells;
        public double LastHalfVoxelScaledTruncationIntegrationMilliseconds;
        public double LastHalfVoxelScaledTruncationEvaluationMilliseconds;
        public bool LastTruncated;
        public int[] WeightedVoxelsByDirection;
        public int[] TrianglesByDirection;
    }

    private sealed class DirectionLayer
    {
        public readonly float[] Tsdf;
        public readonly float[] Weight;
        public readonly Vector3[] NormalSum;
        public readonly ulong[] PaperFrameMask;
        public readonly ulong[] PaperViewMask;
        public readonly byte[] DepthConflictHits;
        public readonly int[] LastDepthConflictBatch;

        public DirectionLayer(int voxelCount)
        {
            Tsdf = new float[voxelCount];
            Weight = new float[voxelCount];
            NormalSum = new Vector3[voxelCount];
            PaperFrameMask = new ulong[voxelCount];
            PaperViewMask = new ulong[voxelCount];
            DepthConflictHits = new byte[voxelCount];
            LastDepthConflictBatch = new int[voxelCount];
        }
    }

    private struct BatchDepthConflictEvidence
    {
        public int Batch;
        public ulong IndependentFrameMask;
        public Vector3 SurfacePointSum;
        public Vector3 NormalSum;
        public int IndependentFrames;
        public bool OpposingReliableSigns;
        public int PreviousBatch;
        public ulong PreviousIndependentFrameMask;
        public Vector3 PreviousSurfacePointSum;
        public Vector3 PreviousNormalSum;
        public int PreviousIndependentFrames;
        public bool PreviousOpposingReliableSigns;

        public Vector3 SurfacePoint
        {
            get
            {
                return IndependentFrames > 0
                    ? SurfacePointSum / IndependentFrames
                    : Vector3.zero;
            }
        }

        public Vector3 Normal
        {
            get
            {
                return NormalSum.sqrMagnitude > 0.00000001f
                    ? NormalSum.normalized
                    : Vector3.up;
            }
        }

    }

    private sealed class PersistentDepthChallenger
    {
        public int AnchorLayerChannel;
        public int AnchorX;
        public int AnchorY;
        public int AnchorZ;
        public int LastBatch;
        public int ConsecutiveBatches;
        public int AccumulatedIndependentFrames;
        public int PreviousSpatialVoxels;
        public ulong CurrentFrameMask;
        public Vector3 CurrentPointSum;
        public Vector3 CurrentNormalSum;
        public int CurrentIndependentFrames;
        public readonly HashSet<int> CurrentVoxels = new HashSet<int>();
        public Vector3 ReferencePoint;
        public Vector3 ReferenceNormal;

        public Vector3 CurrentPoint
        {
            get
            {
                return CurrentIndependentFrames > 0
                    ? CurrentPointSum / CurrentIndependentFrames
                    : ReferencePoint;
            }
        }

        public Vector3 CurrentNormal
        {
            get
            {
                return CurrentNormalSum.sqrMagnitude > 0.00000001f
                    ? CurrentNormalSum.normalized
                    : ReferenceNormal;
            }
        }
    }

    private struct PhysicalSurfaceCertificateKey : IEquatable<PhysicalSurfaceCertificateKey>
    {
        public long First;
        public long Second;

        public PhysicalSurfaceCertificateKey(long left, long right)
        {
            if (left <= right)
            {
                First = left;
                Second = right;
            }
            else
            {
                First = right;
                Second = left;
            }
        }

        public bool Equals(PhysicalSurfaceCertificateKey other)
        {
            return First == other.First && Second == other.Second;
        }

        public override bool Equals(object obj)
        {
            return obj is PhysicalSurfaceCertificateKey other && Equals(other);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return (First.GetHashCode() * 397) ^ Second.GetHashCode();
            }
        }
    }

    private struct PhysicalSurfaceCertificateState
    {
        public int LastConfirmedBatch;
        public int Confirmations;
    }

    private sealed class Block
    {
        public readonly DirectionLayer[] Layers = new DirectionLayer[LayerCount];
    }

    private sealed class HalfVoxelLayer
    {
        public readonly float[] Tsdf;
        public readonly float[] Weight;
        public int WeightedVoxels;

        public HalfVoxelLayer(int voxelCount)
        {
            Tsdf = new float[voxelCount];
            Weight = new float[voxelCount];
        }
    }

    private sealed class HalfVoxelBlock
    {
        public readonly int BlockKey;
        public readonly Vector3 Origin;
        public readonly int Dimension;
        public readonly HalfVoxelLayer[] Layers = new HalfVoxelLayer[DirectionCount];
        public readonly HashSet<int>[] CandidateCells = CreateHalfVoxelCandidateSets();
        public readonly HalfVoxelLayer[] ScaledTruncationLayers =
            new HalfVoxelLayer[DirectionCount];
        public readonly HashSet<int>[] ScaledTruncationCandidateCells =
            CreateHalfVoxelCandidateSets();
        public int LastEligibleBatch;

        public HalfVoxelBlock(int blockKey, Vector3 origin, int dimension)
        {
            BlockKey = blockKey;
            Origin = origin;
            Dimension = dimension;
        }
    }

    private struct DirectionOwnerState
    {
        public int Direction;
        public int Challenger;
        public int ChallengerHits;
    }

    private struct RefinementSpreadState
    {
        public int Samples;
        public float MinimumAxisCoordinate;
        public float MaximumAxisCoordinate;
        public int MinimumFineAxisBin;
        public int MaximumFineAxisBin;
        public byte FineSubcellMask;
    }

    private struct RefinementBlockPersistence
    {
        public int LastBatch;
        public int ConsecutiveBatches;
    }

    private struct RefinementBlockDiagnostic
    {
        public int SpreadCells;
        public int CreaseCells;
        public int DmcDoubleSurfaceCells;
        public int HalfResolvableCells;
        public int HalfInsufficientCells;
        public int DmcOverflowCells;
        public int ConsecutiveBatches;
    }

    private struct RefinementNormalFamilyState
    {
        public int FamilyCount;
        public int Samples0;
        public int Samples1;
        public int Samples2;
        public float Weight0;
        public float Weight1;
        public float Weight2;
        public Vector3 NormalSum0;
        public Vector3 NormalSum1;
        public Vector3 NormalSum2;
    }

    private struct HalfVoxelSample
    {
        public int Direction;
        public Vector3 SurfacePoint;
        public Vector3 Normal;
        public float Weight;
    }

    private struct HalfVoxelCrossingState
    {
        public int CrossingCells;
        public float MinimumSurfaceCoordinate;
        public float MaximumSurfaceCoordinate;
    }

    private struct HalfVoxelVariantMetrics
    {
        public int AllocatedLayers;
        public int WeightedVoxels;
        public int CandidateCells;
        public int ZeroCrossingCells;
        public int PredictedCellsEvaluated;
        public int CoarseEndpointResolvedCells;
        public int FineEndpointResolvedCells;
        public int RecoveredCells;
        public int MissingCells;
        public int ExtraEnvelopeCells;
    }

    private struct CellLayerEvaluation
    {
        public bool ZeroCrossing;
        public float SupportScore;
        public float SurfaceCoordinate;
    }

    private struct ComposedHypothesis
    {
        public int LayerChannel;
        public int Direction;
        public Vector3 Normal;
        public Vector3 Centroid;
        public float Support;
        public bool UsedObservedNormal;
    }

    private struct DirectionalMcShadowHypothesis
    {
        public int Direction;
        public byte Index;
        public byte KnownCorners;
        public ushort TransitionEdges;
        public float Support;
        public float Credibility;
        public Vector3 Gradient;
        public Vector3 Normal;
        public Vector3 Centroid;
    }

    private struct DirectionalMcShadowCombinedIndex
    {
        public byte Index;
        public byte KnownCorners;
        public float Support;
        public Vector3 NormalWeightedSum;
        public Vector3 CentroidWeightedSum;
        public int Members;

        public Vector3 Normal
        {
            get
            {
                return NormalWeightedSum.sqrMagnitude > 0.00000001f
                    ? NormalWeightedSum.normalized
                    : Vector3.up;
            }
        }

        public Vector3 Centroid
        {
            get
            {
                return Support > 0f
                    ? CentroidWeightedSum / Support
                    : Vector3.zero;
            }
        }
    }

    private struct DirectionalMcShadowCell
    {
        public int CellIndex;
        public int X;
        public int Y;
        public int Z;
        public int SurfaceCount;
        public byte CombinedIndex0;
        public byte CombinedIndex1;
        public byte RegularizedIndex0;
        public byte RegularizedIndex1;
        public byte KnownCornerMask0;
        public byte KnownCornerMask1;
        public float Support0;
        public float Support1;
        public Vector3 Normal0;
        public Vector3 Normal1;
        public Vector3 Centroid0;
        public Vector3 Centroid1;
        public byte DeferredFaceMask0;
        public byte DeferredFaceMask1;
    }

    private sealed class PaperDirectionalMcCell
    {
        public int CellIndex;
        public int X;
        public int Y;
        public int Z;
        public int SurfaceCount;
        public byte Index0;
        public byte Index1;
        public byte RegularizedIndex0;
        public byte RegularizedIndex1;
        public int SurfaceIdentity0;
        public int SurfaceIdentity1;
        // Bit 0/1 identify the two endpoint-facing intersection slots from the
        // paper. They are not spatial clusters and therefore never exchange
        // identity when two surfaces approach one another.
        public readonly byte[] EdgeSlotMask = new byte[12];
        public readonly float[] EdgeOffsets = new float[24];
        public readonly float[] EdgeWeights = new float[24];
    }

    private sealed class PaperDmcPersistentCellState
    {
        public PaperDirectionalMcCell Committed;
        public PaperDirectionalMcCell Pending;
        public int PendingStableBatches;
        public int PendingLastObservedBatch;
        public int LastObservedBatch;
        public int MissingBatches;
    }

    private struct PaperDmcCanonicalEdgeState
    {
        public float CommittedOffset;
        public float CommittedWeight;
        public int SurfaceIdentity;
        public float PendingOffset;
        public float PendingWeight;
        public int PendingSurfaceIdentity;
        public int PendingStableBatches;
        public int PendingLastObservedBatch;
        public int LastObservedBatch;
    }

    private struct PaperDmcEdgeObservation
    {
        public float WeightedOffsetSum;
        public float Weight;
        public int SurfaceIdentity;
    }

    private readonly struct PaperDmcSurfaceRef
    {
        public readonly PaperDirectionalMcCell Cell;
        public readonly int Surface;

        public PaperDmcSurfaceRef(PaperDirectionalMcCell cell, int surface)
        {
            Cell = cell;
            Surface = surface;
        }
    }

    private struct HermiteQefEvidence
    {
        public ulong FrameMask;
        public ulong ViewMask;
    }

    private struct HermiteQefSample
    {
        public Vector3 Point;
        public Vector3 Normal;
        public float Weight;
        public ulong FrameMask;
        public ulong ViewMask;
    }

    private sealed class HermiteQefFamily
    {
        public Vector3 NormalSum;
        public Vector3 PointSum;
        public float Weight;
        public int Samples;
        public readonly List<HermiteQefEvidence> Evidence =
            new List<HermiteQefEvidence>(8);
    }

    private struct HermiteQefFeature
    {
        public int CellIndex;
        public Vector3 Proposal;
    }

    private sealed class HermiteQefReplacement
    {
        public readonly List<int> SourceTriangles = new List<int>(8);
        public readonly List<int> Triangles = new List<int>(24);
    }

    private readonly struct HermiteQefTriangleKey :
        IEquatable<HermiteQefTriangleKey>
    {
        private readonly int _a;
        private readonly int _b;
        private readonly int _c;

        public HermiteQefTriangleKey(int a, int b, int c)
        {
            if (a > b) Swap(ref a, ref b);
            if (b > c) Swap(ref b, ref c);
            if (a > b) Swap(ref a, ref b);
            _a = a;
            _b = b;
            _c = c;
        }

        public bool Equals(HermiteQefTriangleKey other)
        {
            return _a == other._a && _b == other._b && _c == other._c;
        }

        public override bool Equals(object obj)
        {
            return obj is HermiteQefTriangleKey other && Equals(other);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = _a;
                hash = hash * 397 ^ _b;
                hash = hash * 397 ^ _c;
                return hash;
            }
        }
    }

    /// <summary>
    /// Persistent, read-only DMC decision for one coarse mesh cell.  It owns no
    /// production state.  The two edge entries are sorted by offset and the
    /// selected surface uses the matching slot when both combined indices cross
    /// the same grid edge.
    /// </summary>
    public sealed class DmcCellDecision
    {
        public int CellIndex;
        public int X;
        public int Y;
        public int Z;
        public int SurfaceCount;
        public byte Index0;
        public byte Index1;
        public byte KnownCornerMask0;
        public byte KnownCornerMask1;
        public byte UnknownCornerMask;
        public bool Overflow;
        public int LastUpdatedBatch;
        public int Revision;
        public int StableBatches;
        public int MissingBatches;
        public float Support0;
        public float Support1;
        public Vector3 Normal0;
        public Vector3 Normal1;
        public Vector3 Centroid0;
        public Vector3 Centroid1;
        public byte DeferredFaceMask0;
        public byte DeferredFaceMask1;
        public readonly byte[] DirectionIndices = new byte[DirectionCount];
        public readonly byte[] DirectionKnownMasks = new byte[DirectionCount];
        public readonly byte[] EdgeCounts = new byte[12];
        public readonly float[] EdgeOffsets = new float[24];
        public readonly float[] EdgeWeights = new float[24];
        public readonly float[] SurfaceValues = new float[16];

        public float EdgeOffset(int edge, int surface)
        {
            int count = EdgeCounts[edge];
            if (count <= 0)
                return 0.5f;
            if (count == 1)
                return EdgeOffsets[edge * 2];
            return EdgeOffsets[edge * 2 + Mathf.Clamp(surface, 0, 1)];
        }
    }

    private readonly struct DmcFaceKey : IEquatable<DmcFaceKey>
    {
        public readonly int MinimumCellIndex;
        public readonly byte Axis;

        public DmcFaceKey(int minimumCellIndex, byte axis)
        {
            MinimumCellIndex = minimumCellIndex;
            Axis = axis;
        }

        public bool Equals(DmcFaceKey other)
        {
            return MinimumCellIndex == other.MinimumCellIndex &&
                Axis == other.Axis;
        }

        public override bool Equals(object obj)
        {
            return obj is DmcFaceKey other && Equals(other);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return (MinimumCellIndex * 397) ^ Axis;
            }
        }
    }

    /// <summary>
    /// Canonical decision shared by the two cells incident to one grid face.
    /// It is deliberately restricted to single-surface, coplanar neighbors in
    /// this first shadow phase.  Reliable observed corner signs are immutable;
    /// the decision may only supply a sign where one side has no observation.
    /// </summary>
    private sealed class DmcFaceDecision
    {
        public DmcFaceKey Key;
        public byte KnownMask;
        public byte InsideMask;
        public byte ConflictMask;
        public byte TransitionMask;
        public Vector3 Normal;
        public Vector3 Centroid;
        public int LastUpdatedBatch;
        public int Revision;
        public int StableBatches;
        public int MissingBatches;
        public int LastMissingBatch = -1;
    }

    private readonly struct DmcFaceApplication
    {
        public readonly DmcFaceKey Key;
        public readonly int LeftCellIndex;
        public readonly int RightCellIndex;
        public readonly byte Axis;

        public DmcFaceApplication(
            DmcFaceKey key,
            int leftCellIndex,
            int rightCellIndex,
            byte axis)
        {
            Key = key;
            LeftCellIndex = leftCellIndex;
            RightCellIndex = rightCellIndex;
            Axis = axis;
        }
    }

    private readonly struct DmcEdgeVertexKey : IEquatable<DmcEdgeVertexKey>
    {
        public readonly int MinimumVoxelIndex;
        public readonly byte Axis;
        public readonly byte Slot;

        public DmcEdgeVertexKey(int minimumVoxelIndex, byte axis, byte slot)
        {
            MinimumVoxelIndex = minimumVoxelIndex;
            Axis = axis;
            Slot = slot;
        }

        public bool Equals(DmcEdgeVertexKey other)
        {
            return MinimumVoxelIndex == other.MinimumVoxelIndex &&
                Axis == other.Axis && Slot == other.Slot;
        }

        public override bool Equals(object obj)
        {
            return obj is DmcEdgeVertexKey other && Equals(other);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = MinimumVoxelIndex;
                hash = hash * 397 ^ Axis;
                hash = hash * 397 ^ Slot;
                return hash;
            }
        }
    }

    private struct ComposedCluster
    {
        public int LayerMask;
        public int MemberCount;
        public Vector3 Normal;
        public Vector3 Centroid;
        public float Support;
        public int OwnerDirection;
        public int OwnerLayerChannel;
        public float OwnerSupport;
    }

    private struct ComposedEdgePair
    {
        public int Count;
        public float T0;
        public int Vertex0;
        public int Surface0;
        public Vector3 Normal0;
        public Vector3 Centroid0;
        public float Support0;
        public float T1;
        public int Vertex1;
        public int Surface1;
        public Vector3 Normal1;
        public Vector3 Centroid1;
        public float Support1;
    }

    private struct SupportedTriangle
    {
        public int Direction;
        public int Surface;
        public int A;
        public int B;
        public int C;
        public float Support;
    }

    private struct ComposedSurfaceNode
    {
        public int CellIndex;
        public int X;
        public int Y;
        public int Z;
        public ComposedCluster Cluster;
        public bool SyntheticGapBridge;
    }

    private struct CertifiedSurfaceLinkCandidate
    {
        public ulong CellPair;
        public int LeftNode;
        public int RightNode;
        public float Score;
    }

    private struct ComposedCellNodes
    {
        public int Count;
        public int Node0;
        public int Node1;
        public int Node2;
        public int Node3;
        public int Node4;
        public int Node5;

        public int Get(int index)
        {
            switch (index)
            {
                case 0: return Node0;
                case 1: return Node1;
                case 2: return Node2;
                case 3: return Node3;
                case 4: return Node4;
                case 5: return Node5;
                default: return -1;
            }
        }

        public void Add(int node)
        {
            switch (Count)
            {
                case 0: Node0 = node; break;
                case 1: Node1 = node; break;
                case 2: Node2 = node; break;
                case 3: Node3 = node; break;
                case 4: Node4 = node; break;
                case 5: Node5 = node; break;
                default: return;
            }
            Count++;
        }
    }

    private struct ComposedSurfaceComponent
    {
        public int LayerMask;
        public int MemberCount;
        public int NodeCount;
        public Vector3 NormalWeightedSum;
        public Vector3 CentroidWeightedSum;
        public float Support;
        public float StrongestNodeSupport;
        public int OwnerDirection;
        public int OwnerLayerChannel;

        public Vector3 Normal
        {
            get
            {
                return NormalWeightedSum.sqrMagnitude > 0.00000001f
                    ? NormalWeightedSum.normalized
                    : Vector3.up;
            }
        }

        public Vector3 Centroid
        {
            get
            {
                return Support > 0f ? CentroidWeightedSum / Support : Vector3.zero;
            }
        }
    }

    private struct SurfaceUnionState
    {
        public Vector3 NormalWeightedSum;
        public Vector3 CentroidWeightedSum;
        public float Support;

        public Vector3 Normal
        {
            get
            {
                return NormalWeightedSum.sqrMagnitude > 0.00000001f
                    ? NormalWeightedSum.normalized
                    : Vector3.up;
            }
        }

        public Vector3 Centroid
        {
            get
            {
                return Support > 0f ? CentroidWeightedSum / Support : Vector3.zero;
            }
        }
    }

    // Same global body diagonal as ScanCoverReferenceVolumetricMesher.  It gives
    // every neighboring cube the same induced face diagonal and avoids cracks
    // while the directional extraction semantics are still under shadow test.
    private static readonly int[,] Tetrahedra =
    {
        { 0, 5, 1, 6 },
        { 0, 1, 2, 6 },
        { 0, 2, 3, 6 },
        { 0, 3, 7, 6 },
        { 0, 7, 4, 6 },
        { 0, 4, 5, 6 }
    };

    private static readonly int[,] CubeEdges =
    {
        { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
        { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
        { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }
    };

    private static readonly int[] CornerX = { 0, 1, 1, 0, 0, 1, 1, 0 };
    private static readonly int[] CornerY = { 0, 0, 1, 1, 0, 0, 1, 1 };
    private static readonly int[] CornerZ = { 0, 0, 0, 0, 1, 1, 1, 1 };
    private static readonly int[,] PositiveFaceCorners =
    {
        { 1, 2, 5, 6 },
        { 3, 2, 7, 6 },
        { 4, 5, 6, 7 }
    };
    private static readonly int[,] NegativeFaceCorners =
    {
        { 0, 3, 4, 7 },
        { 0, 1, 4, 5 },
        { 0, 1, 2, 3 }
    };
    private static readonly Vector3[] DirectionVectors =
    {
        Vector3.right, Vector3.left,
        Vector3.up, Vector3.down,
        Vector3.forward, Vector3.back
    };
    // Paper/source direction order is Y+,Y-,X+,X-,Z-,Z+. ScanCover stores
    // X+,X-,Y+,Y-,Z+,Z-, hence this explicit semantic mapping.
    private static readonly int[] PaperDirectionToScanCoverDirection =
        { 2, 3, 0, 1, 5, 4 };
    // kVtxOffset from the authors' MC tables. This numbering is not the same
    // as ScanCover's legacy/standard-MC CornerX/Y/Z ordering.
    private static readonly int[] PaperCornerX =
        { 0, 1, 1, 0, 0, 1, 1, 0 };
    private static readonly int[] PaperCornerY =
        { 1, 1, 1, 1, 0, 0, 0, 0 };
    private static readonly int[] PaperCornerZ =
        { 1, 1, 0, 0, 1, 1, 0, 0 };
    // Endpoint orientation copied from kEdgeEndpointVertices. It is globally
    // consistent (+X, -Y, -Z) and defines which of the two edge slots a
    // surface owns.
    private static readonly int[,] PaperEdgeEndpointCorners =
    {
        { 0, 1 }, { 2, 1 }, { 3, 2 }, { 3, 0 },
        { 4, 5 }, { 6, 5 }, { 7, 6 }, { 7, 4 },
        { 4, 0 }, { 5, 1 }, { 6, 2 }, { 7, 3 }
    };
    // Four oriented opposite-edge pairs for each paper direction, copied from
    // IsMCIndexDirectionCompatible in the authors' reference implementation.
    private static readonly int[,] PaperDirectionEdgesToCheck =
    {
        { 0, 4, 1, 5, 2, 6, 3, 7 },
        { 4, 0, 5, 1, 6, 2, 7, 3 },
        { 1, 3, 5, 7, 9, 8, 10, 11 },
        { 3, 1, 7, 5, 8, 9, 11, 10 },
        { 2, 0, 6, 4, 10, 9, 11, 8 },
        { 0, 2, 4, 6, 8, 11, 9, 10 }
    };

    private readonly Dictionary<int, Block> _blocks = new Dictionary<int, Block>(256);
    private readonly HashSet<int>[] _candidateCells = CreateCandidateSets();
    private readonly int[] _weightedVoxelsByDirection = new int[DirectionCount];
    private readonly Dictionary<long, DirectionOwnerState> _directionOwners =
        new Dictionary<long, DirectionOwnerState>(8192);
    private readonly Dictionary<long, RefinementSpreadState> _refinementSpreadByCellDirection =
        new Dictionary<long, RefinementSpreadState>(8192);
    private readonly Dictionary<int, RefinementNormalFamilyState>
        _refinementNormalFamiliesByCell =
            new Dictionary<int, RefinementNormalFamilyState>(4096);
    private readonly Dictionary<int, RefinementBlockPersistence> _refinementBlockPersistence =
        new Dictionary<int, RefinementBlockPersistence>(256);
    private readonly Dictionary<int, RefinementBlockDiagnostic> _lastRefinementBlockDiagnostics =
        new Dictionary<int, RefinementBlockDiagnostic>(128);
    private readonly HashSet<int> _dirtyExtractionBlocks = new HashSet<int>();
    private readonly Dictionary<int, HalfVoxelBlock> _halfVoxelShadowBlocks =
        new Dictionary<int, HalfVoxelBlock>(64);
    private readonly HashSet<int> _halfVoxelShadowActiveBlocks = new HashSet<int>();
    private readonly List<HalfVoxelSample> _halfVoxelShadowSamples =
        new List<HalfVoxelSample>(32768);
    private readonly Dictionary<long, BatchDepthConflictEvidence>
        _batchDepthConflictEvidence =
            new Dictionary<long, BatchDepthConflictEvidence>(8192);
    private readonly Dictionary<long, PersistentDepthChallenger>
        _persistentDepthChallengers =
            new Dictionary<long, PersistentDepthChallenger>(4096);
    private readonly HashSet<long> _batchFastCorrectedVoxels =
        new HashSet<long>();
    private readonly Dictionary<PhysicalSurfaceCertificateKey, PhysicalSurfaceCertificateState>
        _physicalSurfaceCertificates =
            new Dictionary<PhysicalSurfaceCertificateKey, PhysicalSurfaceCertificateState>(2048);
    private readonly Dictionary<long, List<long>> _physicalSurfaceCertificateAdjacency =
        new Dictionary<long, List<long>>(2048);
    private readonly Dictionary<int, DmcCellDecision> _dmcCellDecisions =
        new Dictionary<int, DmcCellDecision>(32768);
    private readonly Dictionary<DmcFaceKey, DmcFaceDecision> _dmcFaceDecisions =
        new Dictionary<DmcFaceKey, DmcFaceDecision>(65536);
    private readonly Dictionary<int, PaperDmcPersistentCellState>
        _paperDmcPersistentCells =
            new Dictionary<int, PaperDmcPersistentCellState>(32768);
    private readonly Dictionary<DmcEdgeVertexKey, PaperDmcCanonicalEdgeState>
        _paperDmcCanonicalEdges =
            new Dictionary<DmcEdgeVertexKey, PaperDmcCanonicalEdgeState>(65536);
    private readonly HashSet<int> _dmcOverflowRefinementBlocks = new HashSet<int>();
    private readonly HashSet<int> _dmcOverflowCells = new HashSet<int>();
    private Dictionary<int, PaperDirectionalMcCell> _lastPaperDmcCells;
    private int _lastPaperDmcTriangleLimit;
    private int _dmcDecisionRevision;
    private int _lastPaperDmcLedgerUpdateBatch;
    private int _nextPaperDmcSurfaceIdentity = 1;
    private int _paperDmcTopologyStableBatches;

    private int _dimX;
    private int _dimY;
    private int _dimZ;
    private int _blockSize;
    private int _blockDimX;
    private int _blockDimY;
    private int _blockDimZ;
    private Vector3 _origin;
    private float _voxelSize;
    private float _truncation;
    private float _maximumWeight;
    private bool _configured;
    private bool _canonicalSixDirectionSemantics;
    private bool _refinementProbeEnabled;
    private bool _halfVoxelShadowEnabled;
    private int _allocatedDirectionLayers;
    private int _weightedVoxels;
    private int _batchSamples;
    private int _batchContestedSamples;
    private int _batchSecondarySamples;
    private int _batchCanonicalDirectionWrites;
    private int _batchCanonicalFanoutOne;
    private int _batchCanonicalFanoutTwo;
    private int _batchCanonicalFanoutThree;
    private int _batchCanonicalRejectedSamples;
    private int _batchVoxelUpdates;
    private int _batchOwnerMatched;
    private int _batchOwnerChallenged;
    private int _batchOwnerSwitched;
    private int _batchRetiredDirectionVoxels;
    private int _batchSequence;
    private int _batchSaturatedRollingUpdates;
    private int _batchMatureDepthConflictCandidates;
    private int _batchMatureDepthConflictHeld;
    private int _batchMatureDepthCorrections;
    private int _batchMatureDepthSignFlips;
    private int _batchMatureDepthFastCandidates;
    private int _batchMatureDepthFastConfirmed;
    private int _batchMatureDepthFastHeld;
    private int _batchMatureDepthFastSharedNeighbors;
    private int _batchMatureDepthFastRiskDeferred;
    private int _batchMatureDepthPatchCandidates;
    private int _batchMatureDepthPatchConfirmed;
    private int _batchMatureDepthPatchSharedNeighbors;
    private int _batchMatureDepthPersistentCandidates;
    private int _batchMatureDepthPersistentMatched;
    private int _batchMatureDepthPersistentCrossVoxelMatched;
    private int _batchMatureDepthPersistentConfirmed;
    private int _batchMatureDepthPersistentRiskConfirmed;
    private long _batchIntegrationTicks;
    private MeshResult _lastMesh;
    private double _lastExtractionMilliseconds;
    private int _lastRefinementEvaluationBatch;
    private int _lastHalfVoxelShadowReplayBatch;
    private int _batchHalfVoxelShadowReplayedSamples;
    private int _batchHalfVoxelShadowVoxelUpdates;
    private int _batchHalfVoxelScaledTruncationReplayedSamples;
    private int _batchHalfVoxelScaledTruncationVoxelUpdates;
    private long _batchHalfVoxelScaledTruncationIntegrationTicks;
    private int _batchPaperNormalRays;
    private int _batchPaperTraversedVoxels;
    private int _batchPaperIntegratedVoxels;

    public int BatchPaperNormalRays => _batchPaperNormalRays;
    public int BatchPaperTraversedVoxels => _batchPaperTraversedVoxels;
    public int BatchPaperIntegratedVoxels => _batchPaperIntegratedVoxels;

    public static Vector3 DirectionVector(int direction)
    {
        return direction >= 0 && direction < DirectionCount
            ? DirectionVectors[direction]
            : Vector3.up;
    }

    public void Configure(
        int dimX,
        int dimY,
        int dimZ,
        Vector3 origin,
        float voxelSize,
        float truncation,
        float maximumWeight,
        int blockSize,
        bool canonicalSixDirectionSemantics = false,
        bool refinementProbeEnabled = false,
        bool halfVoxelShadowEnabled = false)
    {
        int safeBlockSize = Mathf.Clamp(blockSize, 4, 16);
        bool safeHalfVoxelShadowEnabled =
            halfVoxelShadowEnabled && refinementProbeEnabled;
        bool same = _configured &&
                    _dimX == dimX && _dimY == dimY && _dimZ == dimZ &&
                    _blockSize == safeBlockSize &&
                    _canonicalSixDirectionSemantics == canonicalSixDirectionSemantics &&
                    _refinementProbeEnabled == refinementProbeEnabled &&
                    _halfVoxelShadowEnabled == safeHalfVoxelShadowEnabled &&
                    Mathf.Abs(_voxelSize - voxelSize) <= 0.000001f &&
                    Mathf.Abs(_truncation - truncation) <= 0.000001f &&
                    (_origin - origin).sqrMagnitude <= 0.0000000001f;
        if (same)
        {
            _maximumWeight = Mathf.Max(1f, maximumWeight);
            return;
        }

        Clear();
        _dimX = Mathf.Max(2, dimX);
        _dimY = Mathf.Max(2, dimY);
        _dimZ = Mathf.Max(2, dimZ);
        _origin = origin;
        _voxelSize = Mathf.Max(0.0001f, voxelSize);
        _truncation = Mathf.Max(_voxelSize, truncation);
        _maximumWeight = Mathf.Max(1f, maximumWeight);
        _blockSize = safeBlockSize;
        _canonicalSixDirectionSemantics = canonicalSixDirectionSemantics;
        _refinementProbeEnabled = refinementProbeEnabled;
        _halfVoxelShadowEnabled = safeHalfVoxelShadowEnabled;
        _blockDimX = Mathf.CeilToInt(_dimX / (float)_blockSize);
        _blockDimY = Mathf.CeilToInt(_dimY / (float)_blockSize);
        _blockDimZ = Mathf.CeilToInt(_dimZ / (float)_blockSize);
        _configured = true;
    }

    public void Clear()
    {
        _blocks.Clear();
        _directionOwners.Clear();
        _refinementSpreadByCellDirection.Clear();
        _refinementNormalFamiliesByCell.Clear();
        _refinementBlockPersistence.Clear();
        _lastRefinementBlockDiagnostics.Clear();
        _dirtyExtractionBlocks.Clear();
        _halfVoxelShadowBlocks.Clear();
        _halfVoxelShadowActiveBlocks.Clear();
        _halfVoxelShadowSamples.Clear();
        _batchDepthConflictEvidence.Clear();
        _persistentDepthChallengers.Clear();
        _batchFastCorrectedVoxels.Clear();
        _physicalSurfaceCertificates.Clear();
        _physicalSurfaceCertificateAdjacency.Clear();
        _dmcCellDecisions.Clear();
        _dmcFaceDecisions.Clear();
        _paperDmcPersistentCells.Clear();
        _paperDmcCanonicalEdges.Clear();
        _dmcOverflowRefinementBlocks.Clear();
        _dmcOverflowCells.Clear();
        _lastPaperDmcCells = null;
        _lastPaperDmcTriangleLimit = 0;
        _dmcDecisionRevision = 0;
        _lastPaperDmcLedgerUpdateBatch = 0;
        _nextPaperDmcSurfaceIdentity = 1;
        _paperDmcTopologyStableBatches = 0;
        for (int i = 0; i < LayerCount; i++)
        {
            _candidateCells[i].Clear();
        }
        for (int i = 0; i < DirectionCount; i++)
        {
            _weightedVoxelsByDirection[i] = 0;
        }
        _configured = false;
        _canonicalSixDirectionSemantics = false;
        _refinementProbeEnabled = false;
        _halfVoxelShadowEnabled = false;
        _allocatedDirectionLayers = 0;
        _weightedVoxels = 0;
        _lastMesh = null;
        _lastExtractionMilliseconds = 0d;
        _lastRefinementEvaluationBatch = 0;
        _lastHalfVoxelShadowReplayBatch = 0;
        _batchSequence = 0;
        BeginBatch();
    }

    public void BeginBatch()
    {
        _batchSequence++;
        _refinementSpreadByCellDirection.Clear();
        _refinementNormalFamiliesByCell.Clear();
        _halfVoxelShadowSamples.Clear();
        _batchSamples = 0;
        _batchContestedSamples = 0;
        _batchSecondarySamples = 0;
        _batchCanonicalDirectionWrites = 0;
        _batchCanonicalFanoutOne = 0;
        _batchCanonicalFanoutTwo = 0;
        _batchCanonicalFanoutThree = 0;
        _batchCanonicalRejectedSamples = 0;
        _batchVoxelUpdates = 0;
        _batchOwnerMatched = 0;
        _batchOwnerChallenged = 0;
        _batchOwnerSwitched = 0;
        _batchRetiredDirectionVoxels = 0;
        _batchSaturatedRollingUpdates = 0;
        _batchMatureDepthConflictCandidates = 0;
        _batchMatureDepthConflictHeld = 0;
        _batchMatureDepthCorrections = 0;
        _batchMatureDepthSignFlips = 0;
        _batchMatureDepthFastCandidates = 0;
        _batchMatureDepthFastConfirmed = 0;
        _batchMatureDepthFastHeld = 0;
        _batchMatureDepthFastSharedNeighbors = 0;
        _batchMatureDepthFastRiskDeferred = 0;
        PruneDepthConflictEvidence();
        PrunePersistentDepthChallengers();
        PrunePhysicalSurfaceCertificates();
        _batchFastCorrectedVoxels.Clear();
        _batchMatureDepthPatchCandidates = 0;
        _batchMatureDepthPatchConfirmed = 0;
        _batchMatureDepthPatchSharedNeighbors = 0;
        _batchMatureDepthPersistentCandidates = 0;
        _batchMatureDepthPersistentMatched = 0;
        _batchMatureDepthPersistentCrossVoxelMatched = 0;
        _batchMatureDepthPersistentConfirmed = 0;
        _batchMatureDepthPersistentRiskConfirmed = 0;
        _batchIntegrationTicks = 0L;
        _batchHalfVoxelShadowReplayedSamples = 0;
        _batchHalfVoxelShadowVoxelUpdates = 0;
        _batchHalfVoxelScaledTruncationReplayedSamples = 0;
        _batchHalfVoxelScaledTruncationVoxelUpdates = 0;
        _batchHalfVoxelScaledTruncationIntegrationTicks = 0L;
        _batchPaperNormalRays = 0;
        _batchPaperTraversedVoxels = 0;
        _batchPaperIntegratedVoxels = 0;
    }

    public void Integrate(
        Vector3 cameraPosition,
        Vector3 surfacePoint,
        Vector3 surfaceNormal,
        float sampleWeight,
        int requestedDirectionChannel,
        int ownershipVoxelIndex,
        bool contested,
        bool secondary,
        float ownerSwitchMargin,
        int ownerSwitchConfirmations,
        int sourceFrameIndex = -1)
    {
        if (!_configured || !Finite(surfacePoint) || !Finite(surfaceNormal) ||
            surfaceNormal.sqrMagnitude <= 0.0001f || sampleWeight <= 0f)
            return;

        long start = Stopwatch.GetTimestamp();
        Vector3 normal = surfaceNormal.normalized;
        Vector3 towardCamera = cameraPosition - surfacePoint;
        if (towardCamera.sqrMagnitude > 0.0001f && Vector3.Dot(normal, towardCamera) < 0f)
            normal = -normal;

        if (_canonicalSixDirectionSemantics)
        {
            IntegrateCanonicalSixDirectionSample(
                surfacePoint, normal, sampleWeight, contested, secondary,
                sourceFrameIndex);
            _batchIntegrationTicks += Stopwatch.GetTimestamp() - start;
            return;
        }

        int suggestedDirection = requestedDirectionChannel >= 1 && requestedDirectionChannel <= DirectionCount
            ? requestedDirectionChannel - 1
            : DominantDirection(normal);
        if (ownershipVoxelIndex < 0)
        {
            WorldToVoxelClamped(surfacePoint, out int ownerX, out int ownerY, out int ownerZ);
            ownershipVoxelIndex = VoxelIndex(ownerX, ownerY, ownerZ);
        }
        int direction = ResolveStableDirection(
            ownershipVoxelIndex, secondary, suggestedDirection, normal,
            Mathf.Clamp(ownerSwitchMargin, 0f, 0.5f),
            Mathf.Clamp(ownerSwitchConfirmations, 2, 8),
            out int retiredDirection);
        if (retiredDirection >= 0 && retiredDirection != direction)
            RetirePreviousDirectionNearSurface(
                retiredDirection + (secondary ? DirectionCount : 0),
                surfacePoint, normal);
        float directionalConfidence = Mathf.Clamp01(Mathf.Abs(Vector3.Dot(normal, DirectionVectors[direction])));
        float weightedSample = sampleWeight * Mathf.Max(0.25f, directionalConfidence);
        if (weightedSample <= 0.0001f)
            return;

        _batchSamples++;
        if (contested)
            _batchContestedSamples++;
        if (secondary)
            _batchSecondarySamples++;

        Vector3 bandExtent = new Vector3(
            Mathf.Abs(normal.x) * _truncation + _voxelSize * 0.75f,
            Mathf.Abs(normal.y) * _truncation + _voxelSize * 0.75f,
            Mathf.Abs(normal.z) * _truncation + _voxelSize * 0.75f);
        Vector3 minimum = surfacePoint - bandExtent;
        Vector3 maximum = surfacePoint + bandExtent;
        WorldToVoxelClamped(minimum, out int minX, out int minY, out int minZ);
        WorldToVoxelClamped(maximum, out int maxX, out int maxY, out int maxZ);
        float lateralLimit = _voxelSize * 0.90f;
        float lateralLimitSq = lateralLimit * lateralLimit;

        for (int z = minZ; z <= maxZ; z++)
        for (int y = minY; y <= maxY; y++)
        for (int x = minX; x <= maxX; x++)
        {
            Vector3 center = VoxelCenter(x, y, z);
            Vector3 delta = center - surfacePoint;
            float signedDistance = Vector3.Dot(delta, normal);
            if (Mathf.Abs(signedDistance) > _truncation + _voxelSize * 0.25f)
                continue;
            Vector3 lateral = delta - normal * signedDistance;
            if (lateral.sqrMagnitude > lateralLimitSq)
                continue;

            float sampleTsdf = Mathf.Clamp(signedDistance / _truncation, -1f, 1f);
            int layerChannel = direction + (secondary ? DirectionCount : 0);
            IntegrateVoxel(
                layerChannel, x, y, z,
                sampleTsdf, weightedSample, normal,
                surfacePoint, sourceFrameIndex,
                !contested && !secondary);
        }

        _batchIntegrationTicks += Stopwatch.GetTimestamp() - start;
    }

    /// <summary>
    /// Frozen paper-side observation contract used by the offline Quest replay:
    /// raw projective surface point, neighbour-filtered normal, exact selected-eye
    /// position, depth-noise weight, view-angle weight and Eq. (3) direction sectors.
    /// This bypasses owner voting, geometry certificates and mature-voxel correction.
    /// </summary>
    public bool IntegratePaperNormalRaycast(
        Vector3 observingEyePosition,
        Vector3 surfacePoint,
        Vector3 neighbourNormal,
        float minimumDepthMeters,
        int sourceFrameIndex = -1)
    {
        if (!_configured || !Finite(observingEyePosition) || !Finite(surfacePoint) ||
            !Finite(neighbourNormal) || neighbourNormal.sqrMagnitude <= 0.00000001f)
            return false;

        long startTicks = Stopwatch.GetTimestamp();
        Vector3 normal = neighbourNormal.normalized;
        Vector3 toEye = observingEyePosition - surfacePoint;
        float depth = toEye.magnitude;
        if (!float.IsFinite(depth) || depth <= 0.000001f)
            return false;
        Vector3 viewDirection = toEye / depth;
        int viewBin = PaperSphericalViewBin(viewDirection);
        if (Vector3.Dot(normal, viewDirection) < 0f)
            normal = -normal;
        float angleWeight = Mathf.Max(0f, Vector3.Dot(normal, viewDirection));
        if (angleWeight <= 0.00000001f)
            return false;

        float safeMinimumDepth = Mathf.Max(0.001f, minimumDepthMeters);
        float referenceSigma = PaperDepthNoiseSigma(safeMinimumDepth);
        float depthSigma = PaperDepthNoiseSigma(depth);
        float depthWeight = Mathf.Clamp01(
            referenceSigma / Mathf.Max(0.000000000001f, depthSigma) *
            (safeMinimumDepth * safeMinimumDepth) /
            Mathf.Max(0.000000000001f, depth * depth));
        if (depthWeight <= 0.000000000001f)
            return false;

        bool hasDirection = false;
        for (int direction = 0; direction < DirectionCount; direction++)
        {
            if (Vector3.Dot(normal, DirectionVectors[direction]) > CanonicalDirectionThreshold)
            {
                hasDirection = true;
                break;
            }
        }
        if (!hasDirection)
            return false;

        _batchSamples++;
        _batchPaperNormalRays++;
        if (_refinementProbeEnabled)
        {
            RecordRefinementNormalFamilyProbe(
                surfacePoint, normal, depthWeight * angleWeight);
            for (int direction = 0; direction < DirectionCount; direction++)
            {
                float sectorWeight =
                    Vector3.Dot(normal, DirectionVectors[direction]);
                if (sectorWeight <= CanonicalDirectionThreshold)
                    continue;
                float combinedWeight =
                    depthWeight * angleWeight * sectorWeight;
                if (combinedWeight <= 0.000000000001f)
                    continue;
                RecordRefinementProbe(
                    direction, surfacePoint, normal, combinedWeight);
            }
        }
        Vector3 segmentStart = surfacePoint - normal * _truncation;
        Vector3 segmentEnd = surfacePoint + normal * _truncation;
        TraverseAndIntegratePaperNormalSegment(
            segmentStart, segmentEnd, surfacePoint, normal,
            depthWeight, angleWeight, sourceFrameIndex, viewBin);
        _batchIntegrationTicks += Stopwatch.GetTimestamp() - startTicks;
        return true;
    }

    /// <summary>
    /// Read-only audit of where the already fused Paper TSDF places its zero
    /// surface relative to one raw projective depth observation. A positive
    /// residual means the stored zero surface lies farther along the oriented
    /// sample normal than the current depth point; a negative value means it
    /// lies behind it. No voxel, evidence mask or topology decision is changed.
    /// </summary>
    public bool TryMeasurePaperSurfaceAlignment(
        Vector3 observingEyePosition,
        Vector3 surfacePoint,
        Vector3 neighbourNormal,
        out PaperSurfaceAlignmentSample sample)
    {
        sample = default;
        sample.DominantDirection = -1;
        if (!_configured || !Finite(observingEyePosition) || !Finite(surfacePoint) ||
            !Finite(neighbourNormal) || neighbourNormal.sqrMagnitude <= 0.00000001f)
            return false;

        Vector3 normal = neighbourNormal.normalized;
        Vector3 toEye = observingEyePosition - surfacePoint;
        if (toEye.sqrMagnitude <= 0.00000001f)
            return false;
        Vector3 viewDirection = toEye.normalized;
        if (Vector3.Dot(normal, viewDirection) < 0f)
            normal = -normal;
        sample.DominantDirection = DominantDirection(normal);

        Vector3 startGrid =
            (surfacePoint - normal * _truncation - _origin) / _voxelSize +
            Vector3.one * 0.5f;
        Vector3 endGrid =
            (surfacePoint + normal * _truncation - _origin) / _voxelSize +
            Vector3.one * 0.5f;
        Vector3 delta = endGrid - startGrid;
        int x = Mathf.FloorToInt(startGrid.x);
        int y = Mathf.FloorToInt(startGrid.y);
        int z = Mathf.FloorToInt(startGrid.z);
        int targetX = Mathf.FloorToInt(endGrid.x);
        int targetY = Mathf.FloorToInt(endGrid.y);
        int targetZ = Mathf.FloorToInt(endGrid.z);
        int stepX = delta.x > 0f ? 1 : delta.x < 0f ? -1 : 0;
        int stepY = delta.y > 0f ? 1 : delta.y < 0f ? -1 : 0;
        int stepZ = delta.z > 0f ? 1 : delta.z < 0f ? -1 : 0;
        float tMaxX = PaperTraversalTMax(startGrid.x, delta.x, x, stepX);
        float tMaxY = PaperTraversalTMax(startGrid.y, delta.y, y, stepY);
        float tMaxZ = PaperTraversalTMax(startGrid.z, delta.z, z, stepZ);
        float tDeltaX = stepX != 0 ? Mathf.Abs(1f / delta.x) : float.PositiveInfinity;
        float tDeltaY = stepY != 0 ? Mathf.Abs(1f / delta.y) : float.PositiveInfinity;
        float tDeltaZ = stepZ != 0 ? Mathf.Abs(1f / delta.z) : float.PositiveInfinity;
        int maximumSteps = Mathf.Abs(targetX - x) + Mathf.Abs(targetY - y) +
                           Mathf.Abs(targetZ - z) + 4;
        double weightedResidual = 0d;
        double supportWeight = 0d;

        for (int traversal = 0; traversal < maximumSteps; traversal++)
        {
            if (x >= 0 && y >= 0 && z >= 0 && x < _dimX && y < _dimY && z < _dimZ)
            {
                Vector3 center = VoxelCenter(x, y, z);
                float signedDistance = Vector3.Dot(center - surfacePoint, normal);
                if (Mathf.Abs(signedDistance) <= _truncation + 0.000000001f)
                {
                    for (int direction = 0; direction < DirectionCount; direction++)
                    {
                        float sectorWeight = Vector3.Dot(normal, DirectionVectors[direction]);
                        if (sectorWeight <= CanonicalDirectionThreshold ||
                            !TryReadVoxel(direction, x, y, z, out float tsdf, out float weight))
                            continue;
                        float auditWeight = weight * sectorWeight;
                        float residual = signedDistance - tsdf * _truncation;
                        weightedResidual += residual * auditWeight;
                        supportWeight += auditWeight;
                        sample.SupportVoxelCount++;
                        if (weight >= _maximumWeight - 0.00001f)
                            sample.SaturatedVoxelCount++;
                    }
                }
            }

            if (x == targetX && y == targetY && z == targetZ)
                break;
            float minimumT = Mathf.Min(tMaxX, Mathf.Min(tMaxY, tMaxZ));
            const float tieEpsilon = 0.000001f;
            if (Mathf.Abs(tMaxX - minimumT) <= tieEpsilon)
            {
                x += stepX;
                tMaxX += tDeltaX;
            }
            if (Mathf.Abs(tMaxY - minimumT) <= tieEpsilon)
            {
                y += stepY;
                tMaxY += tDeltaY;
            }
            if (Mathf.Abs(tMaxZ - minimumT) <= tieEpsilon)
            {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        }

        if (sample.SupportVoxelCount <= 0 || supportWeight <= 0.000000000001d)
            return false;
        sample.SupportWeight = (float)supportWeight;
        sample.SignedResidualMeters = (float)(weightedResidual / supportWeight);
        return float.IsFinite(sample.SignedResidualMeters);
    }

    private static float PaperDepthNoiseSigma(float distanceMeters)
    {
        float delta = distanceMeters - 0.4f;
        return 0.0012f + 0.0019f * delta * delta;
    }

    private void TraverseAndIntegratePaperNormalSegment(
        Vector3 start,
        Vector3 end,
        Vector3 surfacePoint,
        Vector3 normal,
        float depthWeight,
        float angleWeight,
        int sourceFrameIndex,
        int viewBin)
    {
        Vector3 startGrid = (start - _origin) / _voxelSize + Vector3.one * 0.5f;
        Vector3 endGrid = (end - _origin) / _voxelSize + Vector3.one * 0.5f;
        Vector3 delta = endGrid - startGrid;
        int x = Mathf.FloorToInt(startGrid.x);
        int y = Mathf.FloorToInt(startGrid.y);
        int z = Mathf.FloorToInt(startGrid.z);
        int targetX = Mathf.FloorToInt(endGrid.x);
        int targetY = Mathf.FloorToInt(endGrid.y);
        int targetZ = Mathf.FloorToInt(endGrid.z);
        int stepX = delta.x > 0f ? 1 : delta.x < 0f ? -1 : 0;
        int stepY = delta.y > 0f ? 1 : delta.y < 0f ? -1 : 0;
        int stepZ = delta.z > 0f ? 1 : delta.z < 0f ? -1 : 0;
        float tMaxX = PaperTraversalTMax(startGrid.x, delta.x, x, stepX);
        float tMaxY = PaperTraversalTMax(startGrid.y, delta.y, y, stepY);
        float tMaxZ = PaperTraversalTMax(startGrid.z, delta.z, z, stepZ);
        float tDeltaX = stepX != 0 ? Mathf.Abs(1f / delta.x) : float.PositiveInfinity;
        float tDeltaY = stepY != 0 ? Mathf.Abs(1f / delta.y) : float.PositiveInfinity;
        float tDeltaZ = stepZ != 0 ? Mathf.Abs(1f / delta.z) : float.PositiveInfinity;
        int maximumSteps = Mathf.Abs(targetX - x) + Mathf.Abs(targetY - y) +
                           Mathf.Abs(targetZ - z) + 4;

        for (int traversal = 0; traversal < maximumSteps; traversal++)
        {
            _batchPaperTraversedVoxels++;
            if (x >= 0 && y >= 0 && z >= 0 && x < _dimX && y < _dimY && z < _dimZ)
            {
                Vector3 center = VoxelCenter(x, y, z);
                float signedDistance = Vector3.Dot(center - surfacePoint, normal);
                if (Mathf.Abs(signedDistance) <= _truncation + 0.000000001f)
                {
                    float sampleTsdf = Mathf.Clamp(signedDistance / _truncation, -1f, 1f);
                    bool integrated = false;
                    for (int direction = 0; direction < DirectionCount; direction++)
                    {
                        float sectorWeight = Vector3.Dot(normal, DirectionVectors[direction]);
                        if (sectorWeight <= CanonicalDirectionThreshold)
                            continue;
                        float combinedWeight = depthWeight * angleWeight * sectorWeight;
                        if (combinedWeight <= 0.000000000001f)
                            continue;
                        IntegratePaperReferenceVoxel(
                            direction, x, y, z, sampleTsdf, combinedWeight, normal,
                            sourceFrameIndex, viewBin);
                        _batchCanonicalDirectionWrites++;
                        integrated = true;
                    }
                    if (integrated)
                        _batchPaperIntegratedVoxels++;
                }
            }

            if (x == targetX && y == targetY && z == targetZ)
                break;
            float minimumT = Mathf.Min(tMaxX, Mathf.Min(tMaxY, tMaxZ));
            const float tieEpsilon = 0.000001f;
            if (Mathf.Abs(tMaxX - minimumT) <= tieEpsilon)
            {
                x += stepX;
                tMaxX += tDeltaX;
            }
            if (Mathf.Abs(tMaxY - minimumT) <= tieEpsilon)
            {
                y += stepY;
                tMaxY += tDeltaY;
            }
            if (Mathf.Abs(tMaxZ - minimumT) <= tieEpsilon)
            {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        }
    }

    private static float PaperTraversalTMax(
        float startCoordinate,
        float delta,
        int current,
        int step)
    {
        if (step == 0 || Mathf.Abs(delta) <= 0.000000000001f)
            return float.PositiveInfinity;
        float nextBoundary = current + (step > 0 ? 1f : 0f);
        return (nextBoundary - startCoordinate) / delta;
    }

    private void IntegratePaperReferenceVoxel(
        int direction,
        int x,
        int y,
        int z,
        float sampleTsdf,
        float sampleWeight,
        Vector3 sampleNormal,
        int sourceFrameIndex,
        int viewBin)
    {
        DirectionLayer layer = GetOrCreateLayer(direction, x, y, z, out int localIndex);
        // Match the offline replay ledger: evidence describes a physical
        // observation even after the numeric TSDF weight has saturated.
        if (sampleWeight > 0.000000000001f)
        {
            if (sourceFrameIndex >= 0)
                layer.PaperFrameMask[localIndex] |=
                    1UL << (sourceFrameIndex & 63);
            if (viewBin >= 0)
                layer.PaperViewMask[localIndex] |= 1UL << (viewBin & 63);
        }
        float oldWeight = layer.Weight[localIndex];
        float acceptedWeight = Mathf.Min(
            sampleWeight, Mathf.Max(0f, _maximumWeight - oldWeight));
        if (acceptedWeight <= 0.00000001f)
            return;
        float newWeight = oldWeight + acceptedWeight;
        layer.Tsdf[localIndex] = oldWeight > 0f
            ? (layer.Tsdf[localIndex] * oldWeight + sampleTsdf * acceptedWeight) / newWeight
            : sampleTsdf;
        layer.NormalSum[localIndex] += sampleNormal * acceptedWeight;
        layer.Weight[localIndex] = newWeight;
        _batchVoxelUpdates++;
        if (oldWeight <= 0f)
        {
            _weightedVoxels++;
            _weightedVoxelsByDirection[direction]++;
        }
        MarkCandidateCells(direction, x, y, z);
    }

    private static int PaperSphericalViewBin(Vector3 towardEye)
    {
        if (!Finite(towardEye) || towardEye.sqrMagnitude <= 0.00000001f)
            return -1;
        Vector3 direction = towardEye.normalized;
        // Unity uses Y-up. Keep the offline 12 yaw x 3 elevation contract.
        float yaw = Mathf.Atan2(direction.z, direction.x);
        int yawBin = Mathf.FloorToInt(
            ((yaw + Mathf.PI) / (2f * Mathf.PI)) * 12f);
        yawBin = ((yawBin % 12) + 12) % 12;
        float elevation = Mathf.Asin(Mathf.Clamp(direction.y, -1f, 1f));
        int elevationBin = elevation < -25f * Mathf.Deg2Rad
            ? 0
            : elevation > 25f * Mathf.Deg2Rad
                ? 2
                : 1;
        return elevationBin * 12 + yawBin;
    }

    private void IntegrateCanonicalSixDirectionSample(
        Vector3 surfacePoint,
        Vector3 normal,
        float sampleWeight,
        bool contested,
        bool secondary,
        int sourceFrameIndex)
    {
        int fanout = 0;
        for (int direction = 0; direction < DirectionCount; direction++)
        {
            float correspondence = Vector3.Dot(normal, DirectionVectors[direction]);
            if (correspondence <= CanonicalDirectionThreshold)
                continue;
            fanout++;
        }
        if (fanout <= 0)
        {
            _batchCanonicalRejectedSamples++;
            return;
        }
        if (_refinementProbeEnabled)
            RecordRefinementNormalFamilyProbe(
                surfacePoint, normal, sampleWeight);

        _batchSamples++;
        if (contested)
            _batchContestedSamples++;
        if (secondary)
            _batchSecondarySamples++;
        if (fanout == 1)
            _batchCanonicalFanoutOne++;
        else if (fanout == 2)
            _batchCanonicalFanoutTwo++;
        else
            _batchCanonicalFanoutThree++;

        for (int direction = 0; direction < DirectionCount; direction++)
        {
            float correspondence = Vector3.Dot(normal, DirectionVectors[direction]);
            if (correspondence <= CanonicalDirectionThreshold)
                continue;
            _batchCanonicalDirectionWrites++;
            if (_refinementProbeEnabled)
                RecordRefinementProbe(
                    direction, surfacePoint, normal,
                    sampleWeight * correspondence);
            IntegrateCanonicalNormalTraversal(
                direction, surfacePoint, normal,
                sampleWeight * correspondence,
                sourceFrameIndex,
                !contested && !secondary);
        }
    }

    private void IntegrateCanonicalNormalTraversal(
        int direction,
        Vector3 surfacePoint,
        Vector3 normal,
        float weightedSample,
        int sourceFrameIndex,
        bool allowFastDepthCorrection)
    {
        if (weightedSample <= 0.0001f)
            return;

        // Directional TSDF integrates along the measured surface gradient,
        // rather than splatting a wide lateral cylinder into neighboring
        // surfaces.  A sub-voxel step plus consecutive-voxel suppression is a
        // deterministic, allocation-free traversal suitable for the managed
        // validation path; the same semantics can later move to a compute
        // kernel without changing the stored evidence.
        float stepLength = Mathf.Max(_voxelSize * 0.45f, 0.0001f);
        int stepCount = Mathf.Max(1, Mathf.CeilToInt((_truncation * 2f) / stepLength));
        int previousVoxelIndex = -1;
        for (int step = 0; step <= stepCount; step++)
        {
            float t = Mathf.Lerp(-_truncation, _truncation, step / (float)stepCount);
            Vector3 samplePosition = surfacePoint + normal * t;
            WorldToVoxelClamped(samplePosition, out int x, out int y, out int z);
            int voxelIndex = VoxelIndex(x, y, z);
            if (voxelIndex == previousVoxelIndex)
                continue;
            previousVoxelIndex = voxelIndex;

            Vector3 center = VoxelCenter(x, y, z);
            float signedDistance = Vector3.Dot(center - surfacePoint, normal);
            if (Mathf.Abs(signedDistance) > _truncation + _voxelSize * 0.25f)
                continue;
            float sampleTsdf = Mathf.Clamp(signedDistance / _truncation, -1f, 1f);
            IntegrateVoxel(
                direction, x, y, z,
                sampleTsdf, weightedSample, normal,
                surfacePoint, sourceFrameIndex,
                allowFastDepthCorrection);
        }
    }

    private void RecordRefinementProbe(
        int direction,
        Vector3 surfacePoint,
        Vector3 normal,
        float weightedSample)
    {
        Vector3 local = (surfacePoint - _origin) / _voxelSize;
        int cellX = Mathf.Clamp(Mathf.FloorToInt(local.x), 0, _dimX - 2);
        int cellY = Mathf.Clamp(Mathf.FloorToInt(local.y), 0, _dimY - 2);
        int cellZ = Mathf.Clamp(Mathf.FloorToInt(local.z), 0, _dimZ - 2);
        float fractionX = Mathf.Clamp01(local.x - cellX);
        float fractionY = Mathf.Clamp01(local.y - cellY);
        float fractionZ = Mathf.Clamp01(local.z - cellZ);
        int subcell = (fractionX >= 0.5f ? 1 : 0) |
                      (fractionY >= 0.5f ? 2 : 0) |
                      (fractionZ >= 0.5f ? 4 : 0);
        float axisCoordinate;
        switch (direction / 2)
        {
            case 0: axisCoordinate = local.x; break;
            case 1: axisCoordinate = local.y; break;
            default: axisCoordinate = local.z; break;
        }
        int fineAxisBin = Mathf.FloorToInt(axisCoordinate * 2f);
        int cellIndex = CellIndex(cellX, cellY, cellZ);
        long key = (long)cellIndex * DirectionCount + direction;
        _refinementSpreadByCellDirection.TryGetValue(
            key, out RefinementSpreadState state);
        if (state.Samples <= 0)
        {
            state.MinimumAxisCoordinate = axisCoordinate;
            state.MaximumAxisCoordinate = axisCoordinate;
            state.MinimumFineAxisBin = fineAxisBin;
            state.MaximumFineAxisBin = fineAxisBin;
        }
        else
        {
            state.MinimumAxisCoordinate = Mathf.Min(
                state.MinimumAxisCoordinate, axisCoordinate);
            state.MaximumAxisCoordinate = Mathf.Max(
                state.MaximumAxisCoordinate, axisCoordinate);
            state.MinimumFineAxisBin = Mathf.Min(
                state.MinimumFineAxisBin, fineAxisBin);
            state.MaximumFineAxisBin = Mathf.Max(
                state.MaximumFineAxisBin, fineAxisBin);
        }
        state.Samples++;
        state.FineSubcellMask |= (byte)(1 << subcell);
        _refinementSpreadByCellDirection[key] = state;
        if (_halfVoxelShadowEnabled && weightedSample > 0.000000000001f)
        {
            _halfVoxelShadowSamples.Add(new HalfVoxelSample
            {
                Direction = direction,
                SurfacePoint = surfacePoint,
                Normal = normal,
                Weight = weightedSample
            });
        }
    }

    private void RecordRefinementNormalFamilyProbe(
        Vector3 surfacePoint,
        Vector3 normal,
        float sampleWeight)
    {
        if (!Finite(surfacePoint) || !Finite(normal) ||
            normal.sqrMagnitude <= 0.00000001f)
            return;
        Vector3 local = (surfacePoint - _origin) / _voxelSize;
        int cellX = Mathf.Clamp(Mathf.FloorToInt(local.x), 0, _dimX - 2);
        int cellY = Mathf.Clamp(Mathf.FloorToInt(local.y), 0, _dimY - 2);
        int cellZ = Mathf.Clamp(Mathf.FloorToInt(local.z), 0, _dimZ - 2);
        int cellIndex = CellIndex(cellX, cellY, cellZ);
        Vector3 unitNormal = normal.normalized;
        float safeWeight = Mathf.Max(0.000001f, sampleWeight);
        _refinementNormalFamiliesByCell.TryGetValue(
            cellIndex, out RefinementNormalFamilyState state);

        int bestFamily = -1;
        float bestDot = -1f;
        for (int family = 0; family < state.FamilyCount; family++)
        {
            Vector3 familyNormal =
                RefinementFamilyNormal(state, family);
            if (familyNormal.sqrMagnitude <= 0.00000001f)
                continue;
            float dot = Mathf.Abs(Vector3.Dot(unitNormal, familyNormal));
            if (dot > bestDot)
            {
                bestDot = dot;
                bestFamily = family;
            }
        }

        if (bestFamily < 0 ||
            (bestDot < RefinementNormalFamilyMergeAbsDot &&
             state.FamilyCount < 3))
        {
            bestFamily = state.FamilyCount;
            state.FamilyCount++;
        }
        AddRefinementNormalFamilySample(
            ref state, bestFamily, unitNormal, safeWeight);
        _refinementNormalFamiliesByCell[cellIndex] = state;
    }

    private static Vector3 RefinementFamilyNormal(
        RefinementNormalFamilyState state,
        int family)
    {
        Vector3 sum = family == 0
            ? state.NormalSum0
            : family == 1 ? state.NormalSum1 : state.NormalSum2;
        return sum.sqrMagnitude > 0.00000001f
            ? sum.normalized
            : Vector3.zero;
    }

    private static int RefinementFamilySamples(
        RefinementNormalFamilyState state,
        int family)
    {
        return family == 0
            ? state.Samples0
            : family == 1 ? state.Samples1 : state.Samples2;
    }

    private static void AddRefinementNormalFamilySample(
        ref RefinementNormalFamilyState state,
        int family,
        Vector3 normal,
        float weight)
    {
        Vector3 familyNormal = RefinementFamilyNormal(state, family);
        Vector3 alignedNormal =
            familyNormal.sqrMagnitude > 0.00000001f &&
            Vector3.Dot(normal, familyNormal) < 0f
                ? -normal
                : normal;
        if (family == 0)
        {
            state.Samples0++;
            state.Weight0 += weight;
            state.NormalSum0 += alignedNormal * weight;
        }
        else if (family == 1)
        {
            state.Samples1++;
            state.Weight1 += weight;
            state.NormalSum1 += alignedNormal * weight;
        }
        else
        {
            state.Samples2++;
            state.Weight2 += weight;
            state.NormalSum2 += alignedNormal * weight;
        }
    }

    private void EvaluateRefinementProbe(MeshResult result, float minimumWeight)
    {
        result.RefinementProbeEntries = _refinementSpreadByCellDirection.Count;
        _lastRefinementBlockDiagnostics.Clear();
        foreach (KeyValuePair<long, RefinementSpreadState> pair in
                 _refinementSpreadByCellDirection)
        {
            RefinementSpreadState state = pair.Value;
            if (state.Samples < 4)
                continue;
            float spanVoxels = Mathf.Max(
                0f, state.MaximumAxisCoordinate - state.MinimumAxisCoordinate);
            int bucket = spanVoxels < 0.25f
                ? 0
                : spanVoxels < 0.50f
                    ? 1
                    : spanVoxels < 0.75f ? 2 : 3;
            result.RefinementDepthSpanBuckets[bucket]++;
            if (spanVoxels < 0.35f)
                continue;

            int cellIndex = (int)(pair.Key / DirectionCount);
            DecodeCell(cellIndex, out int cellX, out int cellY, out int cellZ);
            int blockKey = BlockKeyFromCell(cellX, cellY, cellZ);
            bool halfResolvable =
                state.MinimumFineAxisBin != state.MaximumFineAxisBin &&
                CountBits(state.FineSubcellMask) >= 2;
            result.RefinementSameDirectionSpreadCells++;
            if (halfResolvable)
                result.RefinementHalfVoxelResolvableCells++;
            else
                result.RefinementHalfVoxelInsufficientCells++;

            _lastRefinementBlockDiagnostics.TryGetValue(
                blockKey, out RefinementBlockDiagnostic block);
            block.SpreadCells++;
            if (halfResolvable)
                block.HalfResolvableCells++;
            else
                block.HalfInsufficientCells++;
            _lastRefinementBlockDiagnostics[blockKey] = block;
        }

        // Detect a crease from distinct measured normal families, not from the
        // number of directional TSDF sectors. One oblique plane legitimately
        // fans out to multiple sectors and must not be mistaken for a fold.
        foreach (KeyValuePair<int, RefinementNormalFamilyState> pair in
                 _refinementNormalFamiliesByCell)
        {
            RefinementNormalFamilyState state = pair.Value;
            bool stableCrease = false;
            for (int first = 0; first < state.FamilyCount && !stableCrease; first++)
            {
                if (RefinementFamilySamples(state, first) <
                    RefinementMinimumNormalFamilySamples)
                    continue;
                Vector3 firstNormal = RefinementFamilyNormal(state, first);
                for (int second = first + 1;
                     second < state.FamilyCount;
                     second++)
                {
                    if (RefinementFamilySamples(state, second) <
                        RefinementMinimumNormalFamilySamples)
                        continue;
                    Vector3 secondNormal =
                        RefinementFamilyNormal(state, second);
                    if (Mathf.Abs(Vector3.Dot(firstNormal, secondNormal)) <=
                        RefinementCreaseMaximumAbsNormalDot)
                    {
                        stableCrease = true;
                        break;
                    }
                }
            }
            if (!stableCrease)
                continue;
            DecodeCell(
                pair.Key, out int cellX, out int cellY, out int cellZ);
            int blockKey = BlockKeyFromCell(cellX, cellY, cellZ);
            _lastRefinementBlockDiagnostics.TryGetValue(
                blockKey, out RefinementBlockDiagnostic block);
            block.CreaseCells++;
            _lastRefinementBlockDiagnostics[blockKey] = block;
            result.RefinementCreaseCells++;
        }

        // A two-surface paper DMC decision is an independent, topology-backed
        // request for more spatial resolution. It catches folds whose two
        // measured faces land in adjacent coarse cells and therefore never
        // coexist in the raw normal-family cell above.
        HashSet<int> dmcDoubleSurfaceBlocks = new HashSet<int>();
        if (_lastPaperDmcCells != null)
        {
            foreach (PaperDirectionalMcCell cell in _lastPaperDmcCells.Values)
            {
                if (cell.SurfaceCount < 2)
                    continue;
                int blockKey = BlockKeyFromCell(cell.X, cell.Y, cell.Z);
                _lastRefinementBlockDiagnostics.TryGetValue(
                    blockKey, out RefinementBlockDiagnostic block);
                block.DmcDoubleSurfaceCells++;
                _lastRefinementBlockDiagnostics[blockKey] = block;
                dmcDoubleSurfaceBlocks.Add(blockKey);
                result.RefinementDmcDoubleSurfaceCells++;
            }
        }
        result.RefinementDmcDoubleSurfaceBlocks =
            dmcDoubleSurfaceBlocks.Count;

        // DMC overflow is a topology-backed request for more spatial
        // resolution.  Feed it into the existing bounded half-voxel lease
        // instead of inventing more directional slots in the coarse cell.
        foreach (int blockKey in _dmcOverflowRefinementBlocks)
        {
            _lastRefinementBlockDiagnostics.TryGetValue(
                blockKey, out RefinementBlockDiagnostic block);
            block.DmcOverflowCells++;
            _lastRefinementBlockDiagnostics[blockKey] = block;
        }

        bool advancePersistence = _lastRefinementEvaluationBatch != _batchSequence;
        if (advancePersistence)
            _lastRefinementEvaluationBatch = _batchSequence;
        List<int> blockKeys = new List<int>(_lastRefinementBlockDiagnostics.Keys);
        blockKeys.Sort();
        for (int i = 0; i < blockKeys.Count; i++)
        {
            int blockKey = blockKeys[i];
            _refinementBlockPersistence.TryGetValue(
                blockKey, out RefinementBlockPersistence persistence);
            if (advancePersistence)
            {
                persistence.ConsecutiveBatches =
                    persistence.LastBatch == _batchSequence - 1
                        ? persistence.ConsecutiveBatches + 1
                        : 1;
                persistence.LastBatch = _batchSequence;
                _refinementBlockPersistence[blockKey] = persistence;
            }
            RefinementBlockDiagnostic diagnostic =
                _lastRefinementBlockDiagnostics[blockKey];
            diagnostic.ConsecutiveBatches = persistence.ConsecutiveBatches;
            _lastRefinementBlockDiagnostics[blockKey] = diagnostic;
            if (diagnostic.ConsecutiveBatches >= 2)
                result.RefinementPersistentBlocks++;
            ExpandRefinementBounds(blockKey, result);
        }

        result.RefinementCandidateBlocks = blockKeys.Count;
        result.RefinementDirtyBlocks = _dirtyExtractionBlocks.Count;
        result.RefinementCleanBlocks = Mathf.Max(
            0, _blocks.Count - result.RefinementDirtyBlocks);
        float refinedFraction = _blocks.Count > 0
            ? result.RefinementCandidateBlocks / (float)_blocks.Count
            : 0f;
        // Replacing a coarse block with a 2x grid in each dimension costs
        // eight voxel samples for every original sample in that block.
        result.RefinementProjectedVoxelMultiplier = 1f + refinedFraction * 7f;
        EvaluateHalfVoxelShadow(result, Mathf.Max(0.0001f, minimumWeight));
        _dirtyExtractionBlocks.Clear();
    }

    private void EvaluateHalfVoxelShadow(MeshResult result, float minimumWeight)
    {
        result.HalfVoxelShadowBufferedSamples = _halfVoxelShadowSamples.Count;
        if (!_halfVoxelShadowEnabled)
            return;

        _halfVoxelShadowActiveBlocks.Clear();
        List<KeyValuePair<int, RefinementBlockDiagnostic>> eligibleBlocks =
            new List<KeyValuePair<int, RefinementBlockDiagnostic>>();
        foreach (KeyValuePair<int, RefinementBlockDiagnostic> pair in
                 _lastRefinementBlockDiagnostics)
        {
            RefinementBlockDiagnostic diagnostic = pair.Value;
            if (diagnostic.ConsecutiveBatches < 2 ||
                (diagnostic.DmcOverflowCells <= 0 &&
                 diagnostic.DmcDoubleSurfaceCells <= 0 &&
                 diagnostic.CreaseCells <= 0 &&
                 diagnostic.HalfResolvableCells <= diagnostic.HalfInsufficientCells))
                continue;
            eligibleBlocks.Add(pair);
        }
        eligibleBlocks.Sort((left, right) =>
        {
            int leftMargin = left.Value.HalfResolvableCells -
                             left.Value.HalfInsufficientCells +
                             left.Value.CreaseCells * 4 +
                             left.Value.DmcDoubleSurfaceCells * 4 +
                             left.Value.DmcOverflowCells * 4;
            int rightMargin = right.Value.HalfResolvableCells -
                              right.Value.HalfInsufficientCells +
                              right.Value.CreaseCells * 4 +
                              right.Value.DmcDoubleSurfaceCells * 4 +
                              right.Value.DmcOverflowCells * 4;
            int compare = rightMargin.CompareTo(leftMargin);
            if (compare != 0)
                return compare;
            compare = right.Value.HalfResolvableCells.CompareTo(
                left.Value.HalfResolvableCells);
            return compare != 0 ? compare : left.Key.CompareTo(right.Key);
        });
        // A scattered top-N selection leaves nearly every useful cell on an
        // active/inactive block seam.  Reserve the first lease for one complete
        // face-neighborhood around the strongest seed, then spend any
        // remaining slot on the next ranked candidate.  The active-block
        // budget is unchanged.
        if (eligibleBlocks.Count > 0)
        {
            int seedKey = eligibleBlocks[0].Key;
            ActivateHalfVoxelShadowBlock(seedKey);
            DecodeBlockKey(
                seedKey, out int seedX, out int seedY, out int seedZ);
            int[,] faceNeighbors =
            {
                { -1, 0, 0 }, { 1, 0, 0 },
                { 0, -1, 0 }, { 0, 1, 0 },
                { 0, 0, -1 }, { 0, 0, 1 }
            };
            for (int neighbor = 0;
                 neighbor < faceNeighbors.GetLength(0) &&
                 _halfVoxelShadowActiveBlocks.Count <
                 HalfVoxelShadowMaximumActiveBlocks;
                 neighbor++)
            {
                int x = seedX + faceNeighbors[neighbor, 0];
                int y = seedY + faceNeighbors[neighbor, 1];
                int z = seedZ + faceNeighbors[neighbor, 2];
                if (x < 0 || y < 0 || z < 0 ||
                    x >= _blockDimX ||
                    y >= _blockDimY ||
                    z >= _blockDimZ)
                {
                    continue;
                }
                ActivateHalfVoxelShadowBlock(
                    x + _blockDimX * (y + _blockDimY * z));
            }
        }
        for (int eligibleIndex = 1;
             eligibleIndex < eligibleBlocks.Count &&
             _halfVoxelShadowActiveBlocks.Count <
             HalfVoxelShadowMaximumActiveBlocks;
             eligibleIndex++)
        {
            ActivateHalfVoxelShadowBlock(
                eligibleBlocks[eligibleIndex].Key);
        }

        List<int> expired = null;
        foreach (KeyValuePair<int, HalfVoxelBlock> pair in _halfVoxelShadowBlocks)
        {
            if (_batchSequence - pair.Value.LastEligibleBatch <=
                HalfVoxelShadowLeaseBatches)
                continue;
            if (expired == null)
                expired = new List<int>();
            expired.Add(pair.Key);
        }
        if (expired != null)
        {
            for (int i = 0; i < expired.Count; i++)
                _halfVoxelShadowBlocks.Remove(expired[i]);
        }

        if (_lastHalfVoxelShadowReplayBatch != _batchSequence)
        {
            _lastHalfVoxelShadowReplayBatch = _batchSequence;
            long start = Stopwatch.GetTimestamp();
            for (int i = 0; i < _halfVoxelShadowSamples.Count; i++)
                ReplayHalfVoxelSample(_halfVoxelShadowSamples[i], true);
            _batchHalfVoxelScaledTruncationIntegrationTicks +=
                Stopwatch.GetTimestamp() - start;
        }

        result.HalfVoxelShadowActiveBlocks = _halfVoxelShadowActiveBlocks.Count;
        foreach (int blockKey in _dmcOverflowRefinementBlocks)
        {
            if (_halfVoxelShadowActiveBlocks.Contains(blockKey))
                result.DirectionalMcShadowOverflowRefinementActiveBlocks++;
        }
        result.HalfVoxelShadowAllocatedBlocks = _halfVoxelShadowBlocks.Count;
        result.HalfVoxelScaledTruncationReplayedSamples =
            _batchHalfVoxelScaledTruncationReplayedSamples;
        result.HalfVoxelScaledTruncationVoxelUpdates =
            _batchHalfVoxelScaledTruncationVoxelUpdates;
        result.HalfVoxelScaledTruncationIntegrationMilliseconds =
            _batchHalfVoxelScaledTruncationIntegrationTicks * 1000d /
            Stopwatch.Frequency;
        long evaluationStart = Stopwatch.GetTimestamp();
        HalfVoxelVariantMetrics fixedVoxel =
            EvaluateHalfVoxelShadowCrossings(minimumWeight, true);
        result.HalfVoxelScaledTruncationEvaluationMilliseconds =
            (Stopwatch.GetTimestamp() - evaluationStart) * 1000d /
            Stopwatch.Frequency;
        CopyHalfVoxelVariantMetrics(result, fixedVoxel, true);

        // The fixed-truncation-width-in-voxels branch won the device A/B test.
        // Keep the original metric names as aliases during the transition so
        // existing analysis scripts remain readable without paying for the
        // retired fixed-physical-truncation replay and extraction.
        result.HalfVoxelShadowReplayedSamples =
            result.HalfVoxelScaledTruncationReplayedSamples;
        result.HalfVoxelShadowVoxelUpdates =
            result.HalfVoxelScaledTruncationVoxelUpdates;
        result.HalfVoxelShadowIntegrationMilliseconds =
            result.HalfVoxelScaledTruncationIntegrationMilliseconds;
        result.HalfVoxelShadowEvaluationMilliseconds =
            result.HalfVoxelScaledTruncationEvaluationMilliseconds;
        CopyHalfVoxelVariantMetrics(result, fixedVoxel, false);
        EvaluateDmcOverflowHalfVoxelResolution(minimumWeight, result);
    }

    private void ActivateHalfVoxelShadowBlock(int blockKey)
    {
        if (!_halfVoxelShadowActiveBlocks.Add(blockKey))
            return;
        if (!_halfVoxelShadowBlocks.TryGetValue(
                blockKey, out HalfVoxelBlock block))
        {
            block = CreateHalfVoxelBlock(blockKey);
            _halfVoxelShadowBlocks.Add(blockKey, block);
        }
        block.LastEligibleBatch = _batchSequence;
    }

    private void EvaluateDmcOverflowHalfVoxelResolution(
        float minimumWeight,
        MeshResult result)
    {
        foreach (int coarseCellIndex in _dmcOverflowCells)
        {
            DecodeCell(
                coarseCellIndex,
                out int coarseX, out int coarseY, out int coarseZ);
            int blockKey = BlockKeyFromCell(coarseX, coarseY, coarseZ);
            if (!_halfVoxelShadowActiveBlocks.Contains(blockKey) ||
                !_halfVoxelShadowBlocks.TryGetValue(
                    blockKey, out HalfVoxelBlock block))
            {
                result.DirectionalMcShadowOverflowStillUnresolved++;
                continue;
            }
            DecodeBlockKey(
                blockKey, out int blockX, out int blockY, out int blockZ);
            int localBaseX =
                (coarseX - (blockX * _blockSize -
                            HalfVoxelShadowHaloCoarseVoxels)) * 2;
            int localBaseY =
                (coarseY - (blockY * _blockSize -
                            HalfVoxelShadowHaloCoarseVoxels)) * 2;
            int localBaseZ =
                (coarseZ - (blockZ * _blockSize -
                            HalfVoxelShadowHaloCoarseVoxels)) * 2;
            bool represented = false;
            bool fineOverflow = false;
            for (int childZ = 0; childZ < 2 && !fineOverflow; childZ++)
            for (int childY = 0; childY < 2 && !fineOverflow; childY++)
            for (int childX = 0; childX < 2 && !fineOverflow; childX++)
            {
                int x = localBaseX + childX;
                int y = localBaseY + childY;
                int z = localBaseZ + childZ;
                ushort group0 = 0;
                ushort group1 = 0;
                int groups = 0;
                for (int direction = 0;
                     direction < DirectionCount;
                     direction++)
                {
                    HalfVoxelLayer layer =
                        block.ScaledTruncationLayers[direction];
                    if (layer == null)
                        continue;
                    byte index = 0;
                    bool complete = true;
                    for (int corner = 0; corner < 8; corner++)
                    {
                        int vx = x + CornerX[corner];
                        int vy = y + CornerY[corner];
                        int vz = z + CornerZ[corner];
                        if (vx < 0 || vy < 0 || vz < 0 ||
                            vx >= block.Dimension ||
                            vy >= block.Dimension ||
                            vz >= block.Dimension)
                        {
                            complete = false;
                            break;
                        }
                        int voxel =
                            vx + block.Dimension *
                            (vy + block.Dimension * vz);
                        if (layer.Weight[voxel] < minimumWeight)
                        {
                            complete = false;
                            break;
                        }
                        if (layer.Tsdf[voxel] < 0f)
                            index |= (byte)(1 << corner);
                    }
                    if (!complete)
                        continue;
                    ushort transitions = TransitionEdgeMask(index);
                    if (transitions == 0)
                        continue;
                    represented = true;
                    if (groups == 0)
                    {
                        group0 = transitions;
                        groups = 1;
                    }
                    else if ((group0 & transitions) != 0)
                    {
                        group0 |= transitions;
                    }
                    else if (groups == 1)
                    {
                        group1 = transitions;
                        groups = 2;
                    }
                    else if ((group1 & transitions) != 0)
                    {
                        group1 |= transitions;
                    }
                    else
                    {
                        fineOverflow = true;
                        break;
                    }
                }
            }
            if (represented && !fineOverflow)
                result.DirectionalMcShadowOverflowResolvedByHalfVoxel++;
            else
                result.DirectionalMcShadowOverflowStillUnresolved++;
        }
    }

    private HalfVoxelBlock CreateHalfVoxelBlock(int blockKey)
    {
        DecodeBlockKey(blockKey, out int blockX, out int blockY, out int blockZ);
        Vector3 origin = _origin + new Vector3(
            blockX * _blockSize - HalfVoxelShadowHaloCoarseVoxels,
            blockY * _blockSize - HalfVoxelShadowHaloCoarseVoxels,
            blockZ * _blockSize - HalfVoxelShadowHaloCoarseVoxels) * _voxelSize;
        int dimension =
            (_blockSize + HalfVoxelShadowHaloCoarseVoxels * 2) * 2 + 1;
        return new HalfVoxelBlock(blockKey, origin, dimension);
    }

    private void ReplayHalfVoxelSample(
        HalfVoxelSample sample, bool scaledTruncation)
    {
        Vector3 coarseLocal = (sample.SurfacePoint - _origin) / _voxelSize;
        int coarseX = Mathf.Clamp(Mathf.FloorToInt(coarseLocal.x), 0, _dimX - 1);
        int coarseY = Mathf.Clamp(Mathf.FloorToInt(coarseLocal.y), 0, _dimY - 1);
        int coarseZ = Mathf.Clamp(Mathf.FloorToInt(coarseLocal.z), 0, _dimZ - 1);
        int centerBlockX = coarseX / _blockSize;
        int centerBlockY = coarseY / _blockSize;
        int centerBlockZ = coarseZ / _blockSize;
        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
        {
            int blockX = centerBlockX + dx;
            int blockY = centerBlockY + dy;
            int blockZ = centerBlockZ + dz;
            if (blockX < 0 || blockY < 0 || blockZ < 0 ||
                blockX >= _blockDimX || blockY >= _blockDimY ||
                blockZ >= _blockDimZ)
                continue;
            int blockKey = blockX +
                           _blockDimX * (blockY + _blockDimY * blockZ);
            if (!_halfVoxelShadowActiveBlocks.Contains(blockKey) ||
                !_halfVoxelShadowBlocks.TryGetValue(
                    blockKey, out HalfVoxelBlock block) ||
                !HalfVoxelBlockContainsSurface(block, sample.SurfacePoint))
                continue;
            IntegrateHalfVoxelSample(block, sample, scaledTruncation);
            if (scaledTruncation)
                _batchHalfVoxelScaledTruncationReplayedSamples++;
            else
                _batchHalfVoxelShadowReplayedSamples++;
        }
    }

    private bool HalfVoxelBlockContainsSurface(
        HalfVoxelBlock block, Vector3 surfacePoint)
    {
        DecodeBlockKey(
            block.BlockKey, out int blockX, out int blockY, out int blockZ);
        Vector3 minimum = _origin + new Vector3(
            blockX * _blockSize,
            blockY * _blockSize,
            blockZ * _blockSize) * _voxelSize -
            Vector3.one * (_voxelSize * HalfVoxelShadowHaloCoarseVoxels);
        Vector3 maximum = _origin + new Vector3(
            Mathf.Min(_dimX - 1, (blockX + 1) * _blockSize),
            Mathf.Min(_dimY - 1, (blockY + 1) * _blockSize),
            Mathf.Min(_dimZ - 1, (blockZ + 1) * _blockSize)) * _voxelSize +
            Vector3.one * (_voxelSize * HalfVoxelShadowHaloCoarseVoxels);
        return surfacePoint.x >= minimum.x && surfacePoint.x <= maximum.x &&
               surfacePoint.y >= minimum.y && surfacePoint.y <= maximum.y &&
               surfacePoint.z >= minimum.z && surfacePoint.z <= maximum.z;
    }

    private void IntegrateHalfVoxelSample(
        HalfVoxelBlock block,
        HalfVoxelSample sample,
        bool scaledTruncation)
    {
        float fineVoxelSize = _voxelSize * 0.5f;
        // nvblox-style fixed truncation width in voxel units: when the voxel
        // edge is halved, halve the physical truncation distance as well.
        float variantTruncation = scaledTruncation
            ? Mathf.Max(fineVoxelSize, _truncation * 0.5f)
            : _truncation;
        float stepLength = Mathf.Max(fineVoxelSize * 0.45f, 0.0001f);
        int stepCount = Mathf.Max(
            1, Mathf.CeilToInt((variantTruncation * 2f) / stepLength));
        int previousIndex = -1;
        for (int step = 0; step <= stepCount; step++)
        {
            float t = Mathf.Lerp(
                -variantTruncation, variantTruncation,
                step / (float)stepCount);
            Vector3 position = sample.SurfacePoint + sample.Normal * t;
            Vector3 local = (position - block.Origin) / fineVoxelSize;
            int x = Mathf.RoundToInt(local.x);
            int y = Mathf.RoundToInt(local.y);
            int z = Mathf.RoundToInt(local.z);
            int dimension = block.Dimension;
            if (x < 0 || y < 0 || z < 0 ||
                x >= dimension || y >= dimension || z >= dimension)
                continue;
            int index = x + dimension * (y + dimension * z);
            if (index == previousIndex)
                continue;
            previousIndex = index;
            Vector3 center = block.Origin +
                             new Vector3(x, y, z) * fineVoxelSize;
            float signedDistance =
                Vector3.Dot(center - sample.SurfacePoint, sample.Normal);
            if (Mathf.Abs(signedDistance) >
                variantTruncation + fineVoxelSize * 0.25f)
                continue;
            HalfVoxelLayer[] layers = scaledTruncation
                ? block.ScaledTruncationLayers
                : block.Layers;
            HalfVoxelLayer layer = layers[sample.Direction];
            if (layer == null)
            {
                layer = new HalfVoxelLayer(
                    dimension * dimension * dimension);
                layers[sample.Direction] = layer;
            }
            float sampleTsdf = Mathf.Clamp(
                signedDistance / variantTruncation, -1f, 1f);
            float oldWeight = layer.Weight[index];
            float acceptedWeight = Mathf.Min(
                sample.Weight, Mathf.Max(0f, _maximumWeight - oldWeight));
            if (acceptedWeight <= 0f)
            {
                acceptedWeight = Mathf.Min(
                    sample.Weight, Mathf.Max(0.0001f, _maximumWeight * 0.125f));
                float retainedWeight = Mathf.Max(0f, oldWeight - acceptedWeight);
                layer.Tsdf[index] =
                    (layer.Tsdf[index] * retainedWeight +
                     sampleTsdf * acceptedWeight) /
                    Mathf.Max(0.0001f, retainedWeight + acceptedWeight);
                layer.Weight[index] = retainedWeight + acceptedWeight;
            }
            else
            {
                float newWeight = oldWeight + acceptedWeight;
                layer.Tsdf[index] = oldWeight > 0f
                    ? (layer.Tsdf[index] * oldWeight +
                       sampleTsdf * acceptedWeight) / newWeight
                    : sampleTsdf;
                layer.Weight[index] = newWeight;
                if (oldWeight <= 0f)
                    layer.WeightedVoxels++;
            }
            if (scaledTruncation)
                _batchHalfVoxelScaledTruncationVoxelUpdates++;
            else
                _batchHalfVoxelShadowVoxelUpdates++;
            MarkHalfVoxelCandidateCells(
                block, sample.Direction, x, y, z, scaledTruncation);
        }
    }

    private static void MarkHalfVoxelCandidateCells(
        HalfVoxelBlock block,
        int direction,
        int x,
        int y,
        int z,
        bool scaledTruncation)
    {
        int dimension = block.Dimension;
        HashSet<int>[] candidateCells = scaledTruncation
            ? block.ScaledTruncationCandidateCells
            : block.CandidateCells;
        for (int dz = -1; dz <= 0; dz++)
        for (int dy = -1; dy <= 0; dy++)
        for (int dx = -1; dx <= 0; dx++)
        {
            int cx = x + dx;
            int cy = y + dy;
            int cz = z + dz;
            if (cx < 0 || cy < 0 || cz < 0 ||
                cx >= dimension - 1 || cy >= dimension - 1 ||
                cz >= dimension - 1)
                continue;
            candidateCells[direction].Add(
                cx + (dimension - 1) * (cy + (dimension - 1) * cz));
        }
    }

    private HalfVoxelVariantMetrics EvaluateHalfVoxelShadowCrossings(
        float minimumWeight, bool scaledTruncation)
    {
        HalfVoxelVariantMetrics metrics = default;
        Dictionary<long, HalfVoxelCrossingState> crossings =
            new Dictionary<long, HalfVoxelCrossingState>(4096);
        HashSet<long> uniqueFineCells = new HashSet<long>();
        float fineVoxelSize = _voxelSize * 0.5f;
        float variantTruncation = scaledTruncation
            ? Mathf.Max(fineVoxelSize, _truncation * 0.5f)
            : _truncation;
        foreach (KeyValuePair<int, HalfVoxelBlock> blockPair in
                 _halfVoxelShadowBlocks)
        {
            HalfVoxelBlock block = blockPair.Value;
            int dimension = block.Dimension;
            int cellDimension = dimension - 1;
            HalfVoxelLayer[] layers = scaledTruncation
                ? block.ScaledTruncationLayers
                : block.Layers;
            HashSet<int>[] candidateCells = scaledTruncation
                ? block.ScaledTruncationCandidateCells
                : block.CandidateCells;
            for (int direction = 0; direction < DirectionCount; direction++)
            {
                HalfVoxelLayer layer = layers[direction];
                if (layer == null)
                    continue;
                metrics.AllocatedLayers++;
                metrics.WeightedVoxels += layer.WeightedVoxels;
                metrics.CandidateCells += candidateCells[direction].Count;
                if (!_halfVoxelShadowActiveBlocks.Contains(blockPair.Key))
                    continue;
                foreach (int cellIndex in candidateCells[direction])
                {
                    int z = cellIndex / (cellDimension * cellDimension);
                    int remainder =
                        cellIndex - z * cellDimension * cellDimension;
                    int y = remainder / cellDimension;
                    int x = remainder - y * cellDimension;
                    bool anyPositive = false;
                    bool anyNegative = false;
                    float coordinateNumerator = 0f;
                    float coordinateWeight = 0f;
                    Vector3 axis = CanonicalAxis(direction);
                    float directionSign =
                        Vector3.Dot(DirectionVectors[direction], axis);
                    for (int corner = 0; corner < 8; corner++)
                    {
                        int vx = x + CornerX[corner];
                        int vy = y + CornerY[corner];
                        int vz = z + CornerZ[corner];
                        int voxelIndex =
                            vx + dimension * (vy + dimension * vz);
                        float weight = layer.Weight[voxelIndex];
                        if (weight < minimumWeight)
                            continue;
                        float value = layer.Tsdf[voxelIndex];
                        anyPositive |= value >= 0f;
                        anyNegative |= value < 0f;
                        float boundedWeight = Mathf.Min(
                            weight, minimumWeight * 4f);
                        Vector3 world = block.Origin +
                                        new Vector3(vx, vy, vz) * fineVoxelSize;
                        float coordinate = Vector3.Dot(world, axis) -
                                           value * variantTruncation * directionSign;
                        coordinateNumerator += coordinate * boundedWeight;
                        coordinateWeight += boundedWeight;
                    }
                    if (!anyPositive || !anyNegative || coordinateWeight <= 0f)
                        continue;
                    int globalFineX = Mathf.RoundToInt(
                        (block.Origin.x - _origin.x) / fineVoxelSize) + x;
                    int globalFineY = Mathf.RoundToInt(
                        (block.Origin.y - _origin.y) / fineVoxelSize) + y;
                    int globalFineZ = Mathf.RoundToInt(
                        (block.Origin.z - _origin.z) / fineVoxelSize) + z;
                    long fineCellKey = PackHalfVoxelCellKey(
                        direction, globalFineX, globalFineY, globalFineZ);
                    if (!uniqueFineCells.Add(fineCellKey))
                        continue;
                    metrics.ZeroCrossingCells++;
                    Vector3 center = block.Origin +
                        new Vector3(x + 0.5f, y + 0.5f, z + 0.5f) *
                        fineVoxelSize;
                    Vector3 coarseLocal = (center - _origin) / _voxelSize;
                    int coarseX = Mathf.Clamp(
                        Mathf.FloorToInt(coarseLocal.x), 0, _dimX - 2);
                    int coarseY = Mathf.Clamp(
                        Mathf.FloorToInt(coarseLocal.y), 0, _dimY - 2);
                    int coarseZ = Mathf.Clamp(
                        Mathf.FloorToInt(coarseLocal.z), 0, _dimZ - 2);
                    long coarseDirectionKey =
                        (long)CellIndex(coarseX, coarseY, coarseZ) *
                        DirectionCount + direction;
                    float surfaceCoordinate =
                        coordinateNumerator / coordinateWeight;
                    crossings.TryGetValue(
                        coarseDirectionKey, out HalfVoxelCrossingState crossing);
                    if (crossing.CrossingCells == 0)
                    {
                        crossing.MinimumSurfaceCoordinate = surfaceCoordinate;
                        crossing.MaximumSurfaceCoordinate = surfaceCoordinate;
                    }
                    else
                    {
                        crossing.MinimumSurfaceCoordinate = Mathf.Min(
                            crossing.MinimumSurfaceCoordinate, surfaceCoordinate);
                        crossing.MaximumSurfaceCoordinate = Mathf.Max(
                            crossing.MaximumSurfaceCoordinate, surfaceCoordinate);
                    }
                    crossing.CrossingCells++;
                    crossings[coarseDirectionKey] = crossing;
                }
            }
        }

        Dictionary<long, CellLayerEvaluation> coarseEvaluations =
            new Dictionary<long, CellLayerEvaluation>(1024);
        foreach (KeyValuePair<long, RefinementSpreadState> pair in
                 _refinementSpreadByCellDirection)
        {
            RefinementSpreadState spread = pair.Value;
            if (spread.Samples < 4 ||
                spread.MaximumAxisCoordinate - spread.MinimumAxisCoordinate < 0.35f ||
                spread.MinimumFineAxisBin == spread.MaximumFineAxisBin ||
                CountBits(spread.FineSubcellMask) < 2)
                continue;
            int cellIndex = (int)(pair.Key / DirectionCount);
            DecodeCell(cellIndex, out int cellX, out int cellY, out int cellZ);
            int blockKey = BlockKeyFromCell(cellX, cellY, cellZ);
            if (!_halfVoxelShadowActiveBlocks.Contains(blockKey))
                continue;
            int direction = (int)(pair.Key % DirectionCount);
            Vector3 axis = CanonicalAxis(direction);
            float originCoordinate = Vector3.Dot(_origin, axis);
            float targetMinimum = originCoordinate +
                                  spread.MinimumAxisCoordinate * _voxelSize;
            float targetMaximum = originCoordinate +
                                  spread.MaximumAxisCoordinate * _voxelSize;
            metrics.PredictedCellsEvaluated++;

            if (TryEvaluateCellLayerCached(
                    direction, cellIndex, minimumWeight, coarseEvaluations,
                    out _, out float coarseCoordinate))
            {
                float coarseTolerance = _voxelSize * 0.20f;
                if (Mathf.Abs(coarseCoordinate - targetMinimum) <= coarseTolerance &&
                    Mathf.Abs(coarseCoordinate - targetMaximum) <= coarseTolerance)
                    metrics.CoarseEndpointResolvedCells++;
            }

            if (!crossings.TryGetValue(pair.Key, out HalfVoxelCrossingState fine))
            {
                metrics.MissingCells++;
                continue;
            }
            float fineTolerance = fineVoxelSize * 0.75f;
            float requiredExtent = Mathf.Max(
                fineVoxelSize * 0.50f,
                (targetMaximum - targetMinimum) * 0.40f);
            bool endpointsResolved =
                fine.MaximumSurfaceCoordinate - fine.MinimumSurfaceCoordinate >=
                    requiredExtent &&
                Mathf.Abs(fine.MinimumSurfaceCoordinate - targetMinimum) <=
                    fineTolerance &&
                Mathf.Abs(fine.MaximumSurfaceCoordinate - targetMaximum) <=
                    fineTolerance;
            if (endpointsResolved)
            {
                metrics.FineEndpointResolvedCells++;
                metrics.RecoveredCells++;
            }
            else
            {
                metrics.MissingCells++;
            }
            if (fine.MinimumSurfaceCoordinate < targetMinimum - fineVoxelSize ||
                fine.MaximumSurfaceCoordinate > targetMaximum + fineVoxelSize)
                metrics.ExtraEnvelopeCells++;
        }
        return metrics;
    }

    private static void CopyHalfVoxelVariantMetrics(
        MeshResult result,
        HalfVoxelVariantMetrics metrics,
        bool scaledTruncation)
    {
        if (scaledTruncation)
        {
            result.HalfVoxelScaledTruncationAllocatedLayers = metrics.AllocatedLayers;
            result.HalfVoxelScaledTruncationWeightedVoxels = metrics.WeightedVoxels;
            result.HalfVoxelScaledTruncationCandidateCells = metrics.CandidateCells;
            result.HalfVoxelScaledTruncationZeroCrossingCells = metrics.ZeroCrossingCells;
            result.HalfVoxelScaledTruncationPredictedCellsEvaluated =
                metrics.PredictedCellsEvaluated;
            result.HalfVoxelScaledTruncationFineEndpointResolvedCells =
                metrics.FineEndpointResolvedCells;
            result.HalfVoxelScaledTruncationRecoveredCells = metrics.RecoveredCells;
            result.HalfVoxelScaledTruncationMissingCells = metrics.MissingCells;
            result.HalfVoxelScaledTruncationExtraEnvelopeCells =
                metrics.ExtraEnvelopeCells;
            return;
        }

        result.HalfVoxelShadowAllocatedLayers = metrics.AllocatedLayers;
        result.HalfVoxelShadowWeightedVoxels = metrics.WeightedVoxels;
        result.HalfVoxelShadowCandidateCells = metrics.CandidateCells;
        result.HalfVoxelShadowZeroCrossingCells = metrics.ZeroCrossingCells;
        result.HalfVoxelShadowPredictedCellsEvaluated =
            metrics.PredictedCellsEvaluated;
        result.HalfVoxelShadowCoarseEndpointResolvedCells =
            metrics.CoarseEndpointResolvedCells;
        result.HalfVoxelShadowFineEndpointResolvedCells =
            metrics.FineEndpointResolvedCells;
        result.HalfVoxelShadowRecoveredCells = metrics.RecoveredCells;
        result.HalfVoxelShadowMissingCells = metrics.MissingCells;
        result.HalfVoxelShadowExtraEnvelopeCells = metrics.ExtraEnvelopeCells;
    }

    private static long PackHalfVoxelCellKey(
        int direction, int x, int y, int z)
    {
        const int bias = 1 << 18;
        const long mask = (1L << 19) - 1L;
        long px = (x + bias) & mask;
        long py = (y + bias) & mask;
        long pz = (z + bias) & mask;
        return ((long)direction & 7L) |
               (px << 3) | (py << 22) | (pz << 41);
    }

    private void ExpandRefinementBounds(int blockKey, MeshResult result)
    {
        DecodeBlockKey(blockKey, out int blockX, out int blockY, out int blockZ);
        Vector3 minimum = _origin + new Vector3(
            blockX * _blockSize,
            blockY * _blockSize,
            blockZ * _blockSize) * _voxelSize;
        Vector3 maximum = _origin + new Vector3(
            Mathf.Min(_dimX, (blockX + 1) * _blockSize),
            Mathf.Min(_dimY, (blockY + 1) * _blockSize),
            Mathf.Min(_dimZ, (blockZ + 1) * _blockSize)) * _voxelSize;
        if (!result.RefinementBoundsValid)
        {
            result.RefinementBoundsValid = true;
            result.RefinementMinimumWorld = minimum;
            result.RefinementMaximumWorld = maximum;
            return;
        }
        result.RefinementMinimumWorld = Vector3.Min(
            result.RefinementMinimumWorld, minimum);
        result.RefinementMaximumWorld = Vector3.Max(
            result.RefinementMaximumWorld, maximum);
    }

    public string GetRefinementBlockDiagnostics(int maximumBlocks)
    {
        if (_lastRefinementBlockDiagnostics.Count == 0)
            return "none";
        List<KeyValuePair<int, RefinementBlockDiagnostic>> entries =
            new List<KeyValuePair<int, RefinementBlockDiagnostic>>(
                _lastRefinementBlockDiagnostics);
        entries.Sort((left, right) =>
        {
            int leftScore =
                left.Value.SpreadCells + left.Value.CreaseCells * 4 +
                left.Value.DmcDoubleSurfaceCells * 4 +
                left.Value.DmcOverflowCells * 4;
            int rightScore =
                right.Value.SpreadCells + right.Value.CreaseCells * 4 +
                right.Value.DmcDoubleSurfaceCells * 4 +
                right.Value.DmcOverflowCells * 4;
            int compare = rightScore.CompareTo(leftScore);
            return compare != 0 ? compare : left.Key.CompareTo(right.Key);
        });
        int count = Mathf.Min(Mathf.Max(1, maximumBlocks), entries.Count);
        StringBuilder builder = new StringBuilder(count * 32);
        for (int i = 0; i < count; i++)
        {
            if (i > 0)
                builder.Append('|');
            DecodeBlockKey(
                entries[i].Key, out int blockX, out int blockY, out int blockZ);
            RefinementBlockDiagnostic diagnostic = entries[i].Value;
            builder.Append(blockX).Append(',')
                .Append(blockY).Append(',')
                .Append(blockZ).Append(':')
                .Append(diagnostic.SpreadCells).Append(',')
                .Append(diagnostic.CreaseCells).Append(',')
                .Append(diagnostic.HalfResolvableCells).Append(',')
                .Append(diagnostic.HalfInsufficientCells).Append(',')
                .Append(diagnostic.DmcDoubleSurfaceCells).Append(',')
                .Append(diagnostic.DmcOverflowCells).Append(',')
                .Append(diagnostic.ConsecutiveBatches);
        }
        return builder.ToString();
    }

    public MeshResult BuildMesh(float minimumWeight, int maximumTriangles, int maximumCells)
    {
        Stopwatch stopwatch = Stopwatch.StartNew();
        MeshResult result = new MeshResult();
        if (!_configured || _blocks.Count == 0)
        {
            _lastMesh = result;
            _lastExtractionMilliseconds = 0d;
            return result;
        }

        float safeMinimumWeight = Mathf.Max(0.0001f, minimumWeight);
        int triangleLimit = Mathf.Max(1, maximumTriangles);
        int cellLimit = Mathf.Max(1, maximumCells);
        int[] cornerIndices = new int[8];
        float[] cornerValues = new float[8];
        float[] cornerWeights = new float[8];
        Vector3[] cornerPositions = new Vector3[8];
        int[] tetraCorners = new int[4];
        int[] inside = new int[4];
        int[] outside = new int[4];
        Dictionary<int, byte> cellDirectionMasks = new Dictionary<int, byte>(32768);
        Dictionary<long, CellLayerEvaluation> cellLayerEvaluations =
            new Dictionary<long, CellLayerEvaluation>(65536);

        int activeLayerCount = _canonicalSixDirectionSemantics
            ? DirectionCount
            : LayerCount;
        for (int layerChannel = 0; layerChannel < activeLayerCount && !result.Truncated; layerChannel++)
        {
            int direction = layerChannel % DirectionCount;
            bool secondaryLayer = layerChannel >= DirectionCount;
            Dictionary<ulong, int> edgeVertices = new Dictionary<ulong, int>(32768);
            foreach (int cellIndex in _candidateCells[layerChannel])
            {
                if (result.ScannedCells >= cellLimit || result.TriangleCount >= triangleLimit)
                {
                    result.Truncated = true;
                    break;
                }
                DecodeCell(cellIndex, out int x, out int y, out int z);
                if (x < 0 || y < 0 || z < 0 || x >= _dimX - 1 || y >= _dimY - 1 || z >= _dimZ - 1)
                    continue;
                result.ScannedCells++;

                bool anyPositive = false;
                bool anyNegative = false;
                float supportScore = 0f;
                float surfaceCoordinateSum = 0f;
                Vector3 canonicalAxis = CanonicalAxis(direction);
                float orientedAxisSign = Vector3.Dot(DirectionVectors[direction], canonicalAxis);
                for (int corner = 0; corner < 8; corner++)
                {
                    int vx = x + CornerX[corner];
                    int vy = y + CornerY[corner];
                    int vz = z + CornerZ[corner];
                    cornerIndices[corner] = VoxelIndex(vx, vy, vz);
                    cornerPositions[corner] = VoxelCenter(vx, vy, vz);
                    if (TryReadVoxel(layerChannel, vx, vy, vz, out float value, out float weight))
                    {
                        cornerValues[corner] = value;
                        cornerWeights[corner] = weight;
                        if (weight >= safeMinimumWeight)
                        {
                            anyPositive |= value >= 0f;
                            anyNegative |= value < 0f;
                            float boundedWeight = Mathf.Min(weight, safeMinimumWeight * 4f);
                            supportScore += boundedWeight;
                            float surfaceCoordinate = Vector3.Dot(cornerPositions[corner], canonicalAxis) -
                                                      value * _truncation * orientedAxisSign;
                            surfaceCoordinateSum += surfaceCoordinate * boundedWeight;
                        }
                    }
                    else
                    {
                        cornerValues[corner] = 1f;
                        cornerWeights[corner] = 0f;
                    }
                }
                bool zeroCrossing = anyPositive && anyNegative;
                float surfaceCoordinateEstimate = supportScore > 0f
                    ? surfaceCoordinateSum / supportScore
                    : 0f;
                cellLayerEvaluations[CellLayerEvaluationKey(layerChannel, cellIndex)] =
                    new CellLayerEvaluation
                    {
                        ZeroCrossing = zeroCrossing,
                        SupportScore = supportScore,
                        SurfaceCoordinate = surfaceCoordinateEstimate
                    };
                if (!zeroCrossing)
                    continue;
                if (!ShouldExtractCellLayer(
                        layerChannel, cellIndex, safeMinimumWeight,
                        supportScore, surfaceCoordinateEstimate,
                        cellLayerEvaluations, result))
                    continue;

                int trianglesBefore = result.TrianglesByDirection[direction].Count;
                for (int tetra = 0; tetra < 6 && result.TriangleCount < triangleLimit; tetra++)
                {
                    int a = Tetrahedra[tetra, 0];
                    int b = Tetrahedra[tetra, 1];
                    int c = Tetrahedra[tetra, 2];
                    int d = Tetrahedra[tetra, 3];
                    if (cornerWeights[a] < safeMinimumWeight || cornerWeights[b] < safeMinimumWeight ||
                        cornerWeights[c] < safeMinimumWeight || cornerWeights[d] < safeMinimumWeight)
                        continue;
                    PolygonizeTetrahedron(
                        direction, a, b, c, d,
                        cornerIndices, cornerValues, cornerPositions,
                        tetraCorners, inside, outside, edgeVertices, result);
                }

                if (result.TrianglesByDirection[direction].Count <= trianglesBefore)
                    continue;
                result.ZeroCrossingCells++;
                if (secondaryLayer)
                    result.SecondaryLayerZeroCrossingCells++;
                cellDirectionMasks.TryGetValue(cellIndex, out byte mask);
                cellDirectionMasks[cellIndex] = (byte)(mask | (1 << direction));
            }
        }

        foreach (byte mask in cellDirectionMasks.Values)
        {
            if (CountBits(mask) > 1)
                result.MultiDirectionZeroCrossingCells++;
        }
        BuildWireAndAudit(result);
        EvaluateRefinementProbe(result, safeMinimumWeight);
        PromoteActiveHalfVoxelPaperDmc(safeMinimumWeight, result);
        BuildHermiteQefFeatureShadow(safeMinimumWeight, result);
        stopwatch.Stop();
        _lastExtractionMilliseconds = stopwatch.Elapsed.TotalMilliseconds;
        _lastMesh = result;
        return result;
    }

    /// <summary>
    /// Conservatively composes compatible directional hypotheses before any
    /// triangle becomes visible.  Unlike BuildMesh, directions do not publish
    /// independent sheets.  Coincident hypotheses share edge vertices and weak
    /// conflicts are dropped; a small boundary loss is preferred over overlap.
    /// </summary>
    public MeshResult BuildComposedMesh(
        float minimumWeight,
        int maximumTriangles,
        int maximumCells,
        float minimumGradientDirectionDot,
        float parallelNormalDot,
        float edgeMergeVoxelRatio,
        bool enableSurfaceContinuity,
        bool enableSurfaceGapBridging,
        float surfaceLinkNormalDot,
        float surfaceLinkPlaneOffsetVoxelRatio)
    {
        Stopwatch stopwatch = Stopwatch.StartNew();
        MeshResult result = new MeshResult();
        if (!_configured || _blocks.Count == 0)
        {
            _lastMesh = result;
            _lastExtractionMilliseconds = 0d;
            return result;
        }

        float safeMinimumWeight = Mathf.Max(0.0001f, minimumWeight);
        int triangleLimit = Mathf.Max(1, maximumTriangles);
        int cellLimit = Mathf.Max(1, maximumCells);
        float safeDirectionDot = Mathf.Clamp(minimumGradientDirectionDot, -0.25f, 0.75f);
        float safeParallelDot = Mathf.Clamp(parallelNormalDot, 0.5f, 0.99f);
        float safeEdgeMergeRatio = Mathf.Clamp(edgeMergeVoxelRatio, 0.05f, 0.49f);
        float safeSurfaceLinkNormalDot = Mathf.Clamp(surfaceLinkNormalDot, safeParallelDot, 0.995f);
        float safeSurfaceLinkPlaneOffset =
            _voxelSize * Mathf.Clamp(surfaceLinkPlaneOffsetVoxelRatio, 0.1f, 0.75f);
        HashSet<int> candidateCells = new HashSet<int>();
        int activeLayerCount = _canonicalSixDirectionSemantics
            ? DirectionCount
            : LayerCount;
        for (int layerChannel = 0; layerChannel < activeLayerCount; layerChannel++)
            candidateCells.UnionWith(_candidateCells[layerChannel]);
        List<int> orderedCandidateCells = new List<int>(candidateCells);
        orderedCandidateCells.Sort();

        int[] cornerIndices = new int[8];
        float[] combinedValues = new float[8];
        float[] combinedWeights = new float[8];
        Vector3[] cornerPositions = new Vector3[8];
        int[] tetraCorners = new int[4];
        int[] inside = new int[4];
        int[] outside = new int[4];
        List<ComposedHypothesis> hypotheses = new List<ComposedHypothesis>(activeLayerCount);
        ComposedCluster[] clusters = new ComposedCluster[activeLayerCount];
        float[] hypothesisValues = new float[8];
        float[] hypothesisWeights = new float[8];
        List<ComposedSurfaceNode> surfaceNodes =
            new List<ComposedSurfaceNode>(Mathf.Min(
                cellLimit * (_canonicalSixDirectionSemantics ? DirectionCount : SurfaceSlotCount),
                orderedCandidateCells.Count *
                (_canonicalSixDirectionSemantics ? DirectionCount : SurfaceSlotCount)));
        Dictionary<int, ComposedCellNodes> nodesByCell =
            new Dictionary<int, ComposedCellNodes>(orderedCandidateCells.Count);
        Dictionary<ulong, ComposedEdgePair> edgeVertices =
            new Dictionary<ulong, ComposedEdgePair>(65536);
        List<SupportedTriangle> supportedTriangles = new List<SupportedTriangle>(98304);

        // Pass one only builds local candidates.  Polygonization waits until
        // adjacent cells have agreed on physical-surface continuity.
        for (int orderedCell = 0; orderedCell < orderedCandidateCells.Count; orderedCell++)
        {
            if (result.ScannedCells >= cellLimit)
            {
                result.Truncated = true;
                break;
            }
            int cellIndex = orderedCandidateCells[orderedCell];
            DecodeCell(cellIndex, out int x, out int y, out int z);
            if (x < 0 || y < 0 || z < 0 || x >= _dimX - 1 || y >= _dimY - 1 || z >= _dimZ - 1)
                continue;
            result.ScannedCells++;
            for (int corner = 0; corner < 8; corner++)
            {
                int vx = x + CornerX[corner];
                int vy = y + CornerY[corner];
                int vz = z + CornerZ[corner];
                cornerIndices[corner] = VoxelIndex(vx, vy, vz);
                cornerPositions[corner] = VoxelCenter(vx, vy, vz);
            }

            hypotheses.Clear();
            for (int layerChannel = 0; layerChannel < activeLayerCount; layerChannel++)
            {
                if (!TryBuildComposedHypothesis(
                        layerChannel, x, y, z, safeMinimumWeight,
                        cornerPositions, hypothesisValues, hypothesisWeights,
                        out ComposedHypothesis hypothesis))
                    continue;
                result.ZeroCrossingCells++;
                if (hypothesis.UsedObservedNormal)
                    result.ObservedNormalHypotheses++;
                float compliance = Vector3.Dot(
                    hypothesis.Normal, DirectionVectors[hypothesis.Direction]);
                if (compliance < safeDirectionDot)
                {
                    result.InvalidDirectionHypotheses++;
                    continue;
                }
                hypotheses.Add(hypothesis);
            }
            if (hypotheses.Count == 0)
                continue;
            if (hypotheses.Count > 1)
                result.MultiDirectionZeroCrossingCells++;
            if (hypotheses.Count >= 3)
                result.ThreePlusDirectionHypothesisCells++;
            result.MaximumDirectionHypothesesPerCell = Mathf.Max(
                result.MaximumDirectionHypothesesPerCell,
                hypotheses.Count);
            hypotheses.Sort((left, right) => right.Support.CompareTo(left.Support));

            int clusterCount = 0;
            for (int hypothesisIndex = 0; hypothesisIndex < hypotheses.Count; hypothesisIndex++)
            {
                ComposedHypothesis hypothesis = hypotheses[hypothesisIndex];
                int destinationIndex = -1;
                float bestMergeScore = float.PositiveInfinity;
                bool oppositeDirectionMerge = false;
                for (int clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++)
                {
                    ComposedCluster candidate = clusters[clusterIndex];
                    if (!TryPhysicalSurfaceMergeScore(
                            candidate.Normal, candidate.Centroid,
                            hypothesis.Normal, hypothesis.Centroid,
                            safeParallelDot,
                            _voxelSize * safeEdgeMergeRatio,
                            _voxelSize * 1.75f,
                            out float mergeScore,
                            out bool opposite) ||
                        mergeScore >= bestMergeScore)
                        continue;
                    destinationIndex = clusterIndex;
                    bestMergeScore = mergeScore;
                    oppositeDirectionMerge = opposite;
                }
                if (destinationIndex < 0)
                {
                    destinationIndex = clusterCount++;
                    clusters[destinationIndex] = new ComposedCluster
                    {
                        LayerMask = 0,
                        MemberCount = 0,
                        Normal = hypothesis.Normal,
                        Centroid = hypothesis.Centroid,
                        Support = 0f,
                        OwnerDirection = hypothesis.Direction,
                        OwnerLayerChannel = hypothesis.LayerChannel,
                        OwnerSupport = 0f
                    };
                }
                else
                {
                    result.ParallelHypothesesCollapsed++;
                    result.PhysicalSurfaceLocalMerges++;
                    if (oppositeDirectionMerge)
                        result.PhysicalSurfaceOppositeDirectionMerges++;
                }
                ComposedCluster destination = clusters[destinationIndex];
                float previousSupport = destination.Support;
                Vector3 alignedHypothesisNormal =
                    Vector3.Dot(destination.Normal, hypothesis.Normal) < 0f
                        ? -hypothesis.Normal
                        : hypothesis.Normal;
                destination.LayerMask |= 1 << hypothesis.LayerChannel;
                destination.MemberCount++;
                destination.Support += hypothesis.Support;
                destination.Normal = (destination.Normal * previousSupport +
                                      alignedHypothesisNormal * hypothesis.Support).normalized;
                destination.Centroid = previousSupport > 0f
                    ? (destination.Centroid * previousSupport +
                       hypothesis.Centroid * hypothesis.Support) / destination.Support
                    : hypothesis.Centroid;
                if (hypothesis.Support > destination.OwnerSupport)
                {
                    destination.OwnerDirection = hypothesis.Direction;
                    destination.OwnerLayerChannel = hypothesis.LayerChannel;
                    destination.OwnerSupport = hypothesis.Support;
                }
                clusters[destinationIndex] = destination;
            }
            for (int left = 0; left < clusterCount - 1; left++)
            {
                for (int right = left + 1; right < clusterCount; right++)
                {
                    if (clusters[left].Support >= clusters[right].Support)
                        continue;
                    ComposedCluster swap = clusters[left];
                    clusters[left] = clusters[right];
                    clusters[right] = swap;
                }
            }
            if (clusterCount >= 3)
                result.ThreePlusSurfaceClusterCells++;
            result.MaximumSurfaceClustersPerCell = Mathf.Max(
                result.MaximumSurfaceClustersPerCell,
                clusterCount);
            if (!_canonicalSixDirectionSemantics && clusterCount > SurfaceSlotCount)
            {
                for (int clusterIndex = SurfaceSlotCount; clusterIndex < clusterCount; clusterIndex++)
                    result.IncompatibleWeakHypothesesDropped += clusters[clusterIndex].MemberCount;
                clusterCount = SurfaceSlotCount;
            }
            if (!_canonicalSixDirectionSemantics && clusterCount > 1)
            {
                ComposedCluster first = clusters[0];
                ComposedCluster second = clusters[1];
                if (Mathf.Abs(Vector3.Dot(first.Normal, second.Normal)) >= safeParallelDot &&
                    Vector3.Distance(first.Centroid, second.Centroid) < _voxelSize)
                {
                    result.ConservativeConflictCells++;
                    result.IncompatibleWeakHypothesesDropped += second.MemberCount;
                    clusterCount = 1;
                }
            }

            ComposedCellNodes cellNodes = default;
            for (int clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++)
            {
                int nodeIndex = surfaceNodes.Count;
                surfaceNodes.Add(new ComposedSurfaceNode
                {
                    CellIndex = cellIndex,
                    X = x,
                    Y = y,
                    Z = z,
                    Cluster = clusters[clusterIndex]
                });
                cellNodes.Add(nodeIndex);
            }
            if (cellNodes.Count > 0)
                nodesByCell[cellIndex] = cellNodes;
        }

        // Evaluate the paper-style Directional Marching Cubes chain against
        // exactly the same sparse candidate cells.  This path deliberately
        // stops at regularized MC indices and edge offsets: it cannot publish
        // vertices or triangles until replay proves that its topology is safer
        // than the current production extractor.
        EvaluateDirectionalMarchingCubesShadow(
            orderedCandidateCells, cellLimit, triangleLimit,
            safeMinimumWeight, safeDirectionDot,
            cornerPositions, hypothesisValues, hypothesisWeights,
            result);

        if (enableSurfaceContinuity && enableSurfaceGapBridging)
        {
            AddSingleCellSurfaceGapBridges(
                surfaceNodes, nodesByCell,
                safeSurfaceLinkNormalDot, safeSurfaceLinkPlaneOffset,
                result);
        }

        result.SurfaceCandidateNodes = surfaceNodes.Count;
        // Index the sparse temporal certificates once.  The previous hot path
        // searched up to 8 corners x 12 layers for every candidate pairing;
        // apart from being expensive, that kept a certificate as an isolated
        // pairwise hint instead of a deterministic pre-triangulation identity.
        HashSet<ulong> certifiedNodePairs =
            BuildActivePhysicalSurfaceNodePairs(surfaceNodes);
        result.PhysicalSurfaceCertificateIndexedNodePairs =
            certifiedNodePairs.Count;
        int[] surfaceParents = new int[surfaceNodes.Count];
        SurfaceUnionState[] surfaceUnionStates = new SurfaceUnionState[surfaceNodes.Count];
        for (int nodeIndex = 0; nodeIndex < surfaceParents.Length; nodeIndex++)
        {
            surfaceParents[nodeIndex] = nodeIndex;
            ComposedCluster cluster = surfaceNodes[nodeIndex].Cluster;
            surfaceUnionStates[nodeIndex] = new SurfaceUnionState
            {
                NormalWeightedSum = cluster.Normal * cluster.Support,
                CentroidWeightedSum = cluster.Centroid * cluster.Support,
                Support = cluster.Support
            };
        }

        if (CertifiedSurfaceDirectComponentLinkProductionEnabled &&
            enableSurfaceContinuity && certifiedNodePairs.Count > 0)
        {
            LinkCertifiedPhysicalSurfaceNodes(
                surfaceNodes, surfaceParents, surfaceUnionStates,
                certifiedNodePairs, result);
        }

        if (enableSurfaceContinuity)
        {
            foreach (KeyValuePair<int, ComposedCellNodes> pair in nodesByCell)
            {
                DecodeCell(pair.Key, out int x, out int y, out int z);
                if (x + 1 < _dimX - 1 &&
                    nodesByCell.TryGetValue(CellIndex(x + 1, y, z), out ComposedCellNodes xNeighbor))
                {
                    LinkSurfaceCellPair(
                        pair.Value, xNeighbor, surfaceNodes, surfaceParents,
                        surfaceUnionStates,
                        safeSurfaceLinkNormalDot, safeSurfaceLinkPlaneOffset,
                        certifiedNodePairs,
                        result);
                }
                if (y + 1 < _dimY - 1 &&
                    nodesByCell.TryGetValue(CellIndex(x, y + 1, z), out ComposedCellNodes yNeighbor))
                {
                    LinkSurfaceCellPair(
                        pair.Value, yNeighbor, surfaceNodes, surfaceParents,
                        surfaceUnionStates,
                        safeSurfaceLinkNormalDot, safeSurfaceLinkPlaneOffset,
                        certifiedNodePairs,
                        result);
                }
                if (z + 1 < _dimZ - 1 &&
                    nodesByCell.TryGetValue(CellIndex(x, y, z + 1), out ComposedCellNodes zNeighbor))
                {
                    LinkSurfaceCellPair(
                        pair.Value, zNeighbor, surfaceNodes, surfaceParents,
                        surfaceUnionStates,
                        safeSurfaceLinkNormalDot, safeSurfaceLinkPlaneOffset,
                        certifiedNodePairs,
                        result);
                }
            }
            result.PhysicalSurfaceOneToOneLinks = result.SurfaceComponentLinks;
            LinkSurfaceComponentsAcrossSingleCellGaps(
                surfaceNodes, nodesByCell, surfaceParents, surfaceUnionStates,
                safeSurfaceLinkNormalDot, safeSurfaceLinkPlaneOffset,
                certifiedNodePairs,
                result);
        }

        ComposedSurfaceComponent[] surfaceComponents =
            new ComposedSurfaceComponent[surfaceNodes.Count];
        for (int nodeIndex = 0; nodeIndex < surfaceNodes.Count; nodeIndex++)
        {
            int root = FindSurfaceRoot(surfaceParents, nodeIndex);
            ComposedCluster cluster = surfaceNodes[nodeIndex].Cluster;
            ComposedSurfaceComponent component = surfaceComponents[root];
            Vector3 componentNormal = component.NormalWeightedSum;
            Vector3 alignedClusterNormal =
                componentNormal.sqrMagnitude > 0.00000001f &&
                Vector3.Dot(componentNormal, cluster.Normal) < 0f
                    ? -cluster.Normal
                    : cluster.Normal;
            component.LayerMask |= cluster.LayerMask;
            component.MemberCount += cluster.MemberCount;
            component.NodeCount++;
            component.NormalWeightedSum += alignedClusterNormal * cluster.Support;
            component.CentroidWeightedSum += cluster.Centroid * cluster.Support;
            component.Support += cluster.Support;
            if (cluster.Support > component.StrongestNodeSupport)
            {
                component.StrongestNodeSupport = cluster.Support;
                component.OwnerDirection = cluster.OwnerDirection;
                component.OwnerLayerChannel = cluster.OwnerLayerChannel;
            }
            surfaceComponents[root] = component;
        }
        for (int componentIndex = 0; componentIndex < surfaceComponents.Length; componentIndex++)
        {
            if (surfaceComponents[componentIndex].NodeCount <= 0)
                continue;
            result.SurfaceComponents++;
            if (surfaceComponents[componentIndex].NodeCount == 1)
                result.SurfaceComponentSingletons++;
        }

        // Pass two uses component-wide evidence and shared grid-edge ownership.
        for (int nodeIndex = 0;
             nodeIndex < surfaceNodes.Count && supportedTriangles.Count < triangleLimit;
             nodeIndex++)
        {
            ComposedSurfaceNode node = surfaceNodes[nodeIndex];
            int surfaceRoot = FindSurfaceRoot(surfaceParents, nodeIndex);
            int surface = enableSurfaceContinuity ? surfaceRoot : -1;
            ComposedSurfaceComponent component = surfaceComponents[surfaceRoot];
            ComposedCluster cluster = node.Cluster;
            if (enableSurfaceContinuity && component.NodeCount > 1)
            {
                cluster.Normal = component.Normal;
                cluster.OwnerDirection = component.OwnerDirection;
                cluster.OwnerLayerChannel = component.OwnerLayerChannel;
            }
            cluster.Support +=
                Mathf.Min(8, Mathf.Max(0, component.NodeCount - 1)) * safeMinimumWeight * 0.25f;

            for (int corner = 0; corner < 8; corner++)
            {
                int vx = node.X + CornerX[corner];
                int vy = node.Y + CornerY[corner];
                int vz = node.Z + CornerZ[corner];
                cornerIndices[corner] = VoxelIndex(vx, vy, vz);
                cornerPositions[corner] = VoxelCenter(vx, vy, vz);
            }
            bool combined;
            if (node.SyntheticGapBridge)
            {
                combined = TryBuildSurfaceGapBridgeValues(
                    component, node.X, node.Y, node.Z, safeMinimumWeight,
                    combinedValues, combinedWeights);
            }
            else
            {
                combined = TryCombineCluster(
                    cluster, node.X, node.Y, node.Z, safeMinimumWeight,
                    combinedValues, combinedWeights);
            }
            if (!combined)
            {
                result.IncompatibleWeakHypothesesDropped += node.Cluster.MemberCount;
                continue;
            }
            int trianglesBefore = supportedTriangles.Count;
            for (int tetra = 0;
                 tetra < 6 && supportedTriangles.Count < triangleLimit;
                 tetra++)
            {
                int a = Tetrahedra[tetra, 0];
                int b = Tetrahedra[tetra, 1];
                int c = Tetrahedra[tetra, 2];
                int d = Tetrahedra[tetra, 3];
                if (combinedWeights[a] < safeMinimumWeight ||
                    combinedWeights[b] < safeMinimumWeight ||
                    combinedWeights[c] < safeMinimumWeight ||
                    combinedWeights[d] < safeMinimumWeight)
                    continue;
                PolygonizeComposedTetrahedron(
                    cluster.OwnerDirection, surface, cluster.Normal,
                    component.Centroid, cluster.Support,
                    a, b, c, d,
                    cornerIndices, combinedValues, cornerPositions,
                    tetraCorners, inside, outside,
                    safeEdgeMergeRatio, safeParallelDot, edgeVertices,
                    supportedTriangles, result);
            }
            if (node.SyntheticGapBridge && supportedTriangles.Count > trianglesBefore)
                result.SurfaceGapBridgeTriangles += supportedTriangles.Count - trianglesBefore;
            if (supportedTriangles.Count > trianglesBefore &&
                node.Cluster.OwnerLayerChannel >= DirectionCount)
                result.SecondaryLayerZeroCrossingCells++;
        }
        if (supportedTriangles.Count >= triangleLimit && surfaceNodes.Count > 0)
            result.Truncated = true;

        CommitSupportedManifoldTriangles(supportedTriangles, result);
        BuildWireAndAudit(result);
        EvaluateRefinementProbe(result, safeMinimumWeight);
        PromoteActiveHalfVoxelPaperDmc(safeMinimumWeight, result);
        BuildHermiteQefFeatureShadow(safeMinimumWeight, result);
        stopwatch.Stop();
        _lastExtractionMilliseconds = stopwatch.Elapsed.TotalMilliseconds;
        _lastMesh = result;
        return result;
    }

    private bool TryBuildSurfaceGapBridgeValues(
        ComposedSurfaceComponent component,
        int x,
        int y,
        int z,
        float minimumWeight,
        float[] values,
        float[] weights)
    {
        if (component.NodeCount < 3 || component.Support <= 0f)
            return false;
        Vector3 normal = component.Normal;
        Vector3 centroid = component.Centroid;
        float bridgeWeight = Mathf.Clamp(
            component.StrongestNodeSupport / 8f,
            minimumWeight,
            minimumWeight * 2f);
        bool anyPositive = false;
        bool anyNegative = false;
        for (int corner = 0; corner < 8; corner++)
        {
            Vector3 position = VoxelCenter(
                x + CornerX[corner],
                y + CornerY[corner],
                z + CornerZ[corner]);
            float value = Mathf.Clamp(
                Vector3.Dot(position - centroid, normal) /
                Mathf.Max(_voxelSize, _truncation),
                -1f, 1f);
            values[corner] = value;
            weights[corner] = bridgeWeight;
            anyPositive |= value >= 0f;
            anyNegative |= value < 0f;
        }
        return anyPositive && anyNegative;
    }

    private void LinkSurfaceComponentsAcrossSingleCellGaps(
        List<ComposedSurfaceNode> nodes,
        Dictionary<int, ComposedCellNodes> nodesByCell,
        int[] parents,
        SurfaceUnionState[] unionStates,
        float minimumNormalDot,
        float maximumPlaneOffset,
        HashSet<ulong> certifiedNodePairs,
        MeshResult result)
    {
        List<int> sourceCells = new List<int>(nodesByCell.Keys);
        sourceCells.Sort();
        for (int sourceIndex = 0; sourceIndex < sourceCells.Count; sourceIndex++)
        {
            int sourceCell = sourceCells[sourceIndex];
            DecodeCell(sourceCell, out int x, out int y, out int z);
            TryLinkSurfaceAcrossSingleCellGap(
                sourceCell, x + 1, y, z, x + 2, y, z,
                nodes, nodesByCell, parents, unionStates,
                minimumNormalDot, maximumPlaneOffset,
                certifiedNodePairs,
                result);
            TryLinkSurfaceAcrossSingleCellGap(
                sourceCell, x, y + 1, z, x, y + 2, z,
                nodes, nodesByCell, parents, unionStates,
                minimumNormalDot, maximumPlaneOffset,
                certifiedNodePairs,
                result);
            TryLinkSurfaceAcrossSingleCellGap(
                sourceCell, x, y, z + 1, x, y, z + 2,
                nodes, nodesByCell, parents, unionStates,
                minimumNormalDot, maximumPlaneOffset,
                certifiedNodePairs,
                result);
        }
    }

    private void TryLinkSurfaceAcrossSingleCellGap(
        int sourceCell,
        int middleX,
        int middleY,
        int middleZ,
        int farX,
        int farY,
        int farZ,
        List<ComposedSurfaceNode> nodes,
        Dictionary<int, ComposedCellNodes> nodesByCell,
        int[] parents,
        SurfaceUnionState[] unionStates,
        float minimumNormalDot,
        float maximumPlaneOffset,
        HashSet<ulong> certifiedNodePairs,
        MeshResult result)
    {
        if (middleX < 0 || middleY < 0 || middleZ < 0 ||
            farX < 0 || farY < 0 || farZ < 0 ||
            middleX >= _dimX - 1 || middleY >= _dimY - 1 || middleZ >= _dimZ - 1 ||
            farX >= _dimX - 1 || farY >= _dimY - 1 || farZ >= _dimZ - 1)
            return;
        int middleCell = CellIndex(middleX, middleY, middleZ);
        if (nodesByCell.ContainsKey(middleCell))
            return;
        int farCell = CellIndex(farX, farY, farZ);
        if (!nodesByCell.TryGetValue(sourceCell, out ComposedCellNodes sourceNodes) ||
            !nodesByCell.TryGetValue(farCell, out ComposedCellNodes farNodes))
            return;

        int compatiblePairs = 0;
        int certifiedPairs = 0;
        int matchedSource = -1;
        int matchedFar = -1;
        bool matchedCertified = false;
        bool matchedRelaxed = false;
        for (int sourceSlot = 0; sourceSlot < sourceNodes.Count; sourceSlot++)
        for (int farSlot = 0; farSlot < farNodes.Count; farSlot++)
        {
            int sourceNode = sourceNodes.Get(sourceSlot);
            int farNode = farNodes.Get(farSlot);
            bool certified = certifiedNodePairs.Contains(
                SurfaceNodePairKey(sourceNode, farNode));
            bool standard = TrySurfaceLinkScore(
                nodes[sourceNode].Cluster, nodes[farNode].Cluster,
                minimumNormalDot, maximumPlaneOffset,
                out _);
            bool relaxed = !standard && certified &&
                           TryCertifiedSurfaceLinkScore(
                               nodes[sourceNode].Cluster,
                               nodes[farNode].Cluster,
                               out _);
            if (certified)
            {
                result.PhysicalSurfaceCertificateCandidatePairs++;
                if (!standard && !relaxed)
                    result.PhysicalSurfaceCertificateRejectedPairs++;
            }
            if (!standard && !relaxed)
                continue;
            compatiblePairs++;
            if (certified)
            {
                certifiedPairs++;
                matchedSource = sourceNode;
                matchedFar = farNode;
                matchedCertified = true;
                matchedRelaxed = relaxed;
            }
            else if (certifiedPairs == 0)
            {
                matchedSource = sourceNode;
                matchedFar = farNode;
                matchedCertified = false;
                matchedRelaxed = false;
            }
        }
        if (certifiedPairs > 1)
        {
            result.PhysicalSurfaceCertificateAmbiguousPairs++;
            return;
        }
        if ((certifiedPairs == 0 && compatiblePairs != 1) ||
            !TryUnionSurfaceRoots(
                parents, unionStates, matchedSource, matchedFar,
                matchedCertified
                    ? Mathf.Min(minimumNormalDot, CertifiedSurfaceLinkMinimumNormalDot)
                    : minimumNormalDot,
                matchedCertified
                    ? Mathf.Max(maximumPlaneOffset,
                        _voxelSize * CertifiedSurfaceLinkMaximumPlaneOffsetVoxels)
                    : maximumPlaneOffset))
            return;
        result.SurfaceComponentLinks++;
        result.SurfaceSingleCellGapLinks++;
        if (matchedCertified)
        {
            result.PhysicalSurfaceCertificateLinks++;
            if (matchedRelaxed)
                result.PhysicalSurfaceCertificateRelaxedLinks++;
        }
    }

    private void AddSingleCellSurfaceGapBridges(
        List<ComposedSurfaceNode> nodes,
        Dictionary<int, ComposedCellNodes> nodesByCell,
        float minimumNormalDot,
        float maximumPlaneOffset,
        MeshResult result)
    {
        List<int> sourceCells = new List<int>(nodesByCell.Keys);
        sourceCells.Sort();
        Dictionary<int, ComposedCluster> pendingBridges =
            new Dictionary<int, ComposedCluster>();
        for (int sourceIndex = 0; sourceIndex < sourceCells.Count; sourceIndex++)
        {
            int sourceCell = sourceCells[sourceIndex];
            DecodeCell(sourceCell, out int x, out int y, out int z);
            TryQueueSurfaceGapBridge(
                sourceCell, x + 1, y, z, x + 2, y, z,
                nodes, nodesByCell, pendingBridges,
                minimumNormalDot, maximumPlaneOffset);
            TryQueueSurfaceGapBridge(
                sourceCell, x, y + 1, z, x, y + 2, z,
                nodes, nodesByCell, pendingBridges,
                minimumNormalDot, maximumPlaneOffset);
            TryQueueSurfaceGapBridge(
                sourceCell, x, y, z + 1, x, y, z + 2,
                nodes, nodesByCell, pendingBridges,
                minimumNormalDot, maximumPlaneOffset);
        }

        List<int> bridgeCells = new List<int>(pendingBridges.Keys);
        bridgeCells.Sort();
        for (int bridgeIndex = 0; bridgeIndex < bridgeCells.Count; bridgeIndex++)
        {
            int cellIndex = bridgeCells[bridgeIndex];
            if (nodesByCell.ContainsKey(cellIndex))
                continue;
            DecodeCell(cellIndex, out int x, out int y, out int z);
            int nodeIndex = nodes.Count;
            nodes.Add(new ComposedSurfaceNode
            {
                CellIndex = cellIndex,
                X = x,
                Y = y,
                Z = z,
                Cluster = pendingBridges[cellIndex],
                SyntheticGapBridge = true
            });
            ComposedCellNodes cellNodes = default;
            cellNodes.Add(nodeIndex);
            nodesByCell[cellIndex] = cellNodes;
            result.SurfaceGapBridgeCandidates++;
        }
    }

    private void TryQueueSurfaceGapBridge(
        int sourceCell,
        int middleX,
        int middleY,
        int middleZ,
        int farX,
        int farY,
        int farZ,
        List<ComposedSurfaceNode> nodes,
        Dictionary<int, ComposedCellNodes> nodesByCell,
        Dictionary<int, ComposedCluster> pendingBridges,
        float minimumNormalDot,
        float maximumPlaneOffset)
    {
        if (middleX < 0 || middleY < 0 || middleZ < 0 ||
            farX < 0 || farY < 0 || farZ < 0 ||
            middleX >= _dimX - 1 || middleY >= _dimY - 1 || middleZ >= _dimZ - 1 ||
            farX >= _dimX - 1 || farY >= _dimY - 1 || farZ >= _dimZ - 1)
            return;
        int middleCell = CellIndex(middleX, middleY, middleZ);
        if (nodesByCell.ContainsKey(middleCell) || pendingBridges.ContainsKey(middleCell))
            return;
        int farCell = CellIndex(farX, farY, farZ);
        if (!nodesByCell.TryGetValue(sourceCell, out ComposedCellNodes sourceNodes) ||
            !nodesByCell.TryGetValue(farCell, out ComposedCellNodes farNodes))
            return;

        int compatiblePairs = 0;
        ComposedCluster leftMatch = default;
        ComposedCluster rightMatch = default;
        for (int leftSlot = 0; leftSlot < sourceNodes.Count; leftSlot++)
        for (int rightSlot = 0; rightSlot < farNodes.Count; rightSlot++)
        {
            ComposedCluster left = nodes[sourceNodes.Get(leftSlot)].Cluster;
            ComposedCluster right = nodes[farNodes.Get(rightSlot)].Cluster;
            if (!TrySurfaceLinkScore(
                    left, right, minimumNormalDot, maximumPlaneOffset,
                    out _))
                continue;
            compatiblePairs++;
            leftMatch = left;
            rightMatch = right;
        }
        // A missing cell is filled only when one unambiguous physical surface
        // crosses it.  Junctions and two-layer gaps remain open.
        if (compatiblePairs != 1)
            return;
        Vector3 normal =
            (leftMatch.Normal * leftMatch.Support + rightMatch.Normal * rightMatch.Support).normalized;
        pendingBridges[middleCell] = new ComposedCluster
        {
            LayerMask = leftMatch.LayerMask | rightMatch.LayerMask,
            MemberCount = leftMatch.MemberCount + rightMatch.MemberCount,
            Normal = normal,
            Centroid = (leftMatch.Centroid + rightMatch.Centroid) * 0.5f,
            Support = Mathf.Min(leftMatch.Support, rightMatch.Support) * 0.5f,
            OwnerDirection = leftMatch.Support >= rightMatch.Support
                ? leftMatch.OwnerDirection
                : rightMatch.OwnerDirection,
            OwnerLayerChannel = leftMatch.Support >= rightMatch.Support
                ? leftMatch.OwnerLayerChannel
                : rightMatch.OwnerLayerChannel,
            OwnerSupport = Mathf.Max(
                leftMatch.OwnerSupport, rightMatch.OwnerSupport)
        };
    }

    private void LinkSurfaceCellPair(
        ComposedCellNodes left,
        ComposedCellNodes right,
        List<ComposedSurfaceNode> nodes,
        int[] parents,
        SurfaceUnionState[] unionStates,
        float minimumNormalDot,
        float maximumPlaneOffset,
        HashSet<ulong> certifiedNodePairs,
        MeshResult result)
    {
        int leftUsedMask = 0;
        int rightUsedMask = 0;
        int maximumLinks = Mathf.Min(left.Count, right.Count);
        for (int pass = 0; pass < maximumLinks; pass++)
        {
            float bestScore = float.PositiveInfinity;
            int bestLeft = -1;
            int bestRight = -1;
            bool bestCertified = false;
            bool bestRelaxed = false;
            for (int leftSlot = 0; leftSlot < left.Count; leftSlot++)
            {
                if ((leftUsedMask & (1 << leftSlot)) != 0)
                    continue;
                int leftNode = left.Get(leftSlot);
                for (int rightSlot = 0; rightSlot < right.Count; rightSlot++)
                {
                    if ((rightUsedMask & (1 << rightSlot)) != 0)
                        continue;
                    int rightNode = right.Get(rightSlot);
                    bool certified = certifiedNodePairs.Contains(
                        SurfaceNodePairKey(leftNode, rightNode));
                    bool standard = TrySurfaceLinkScore(
                        nodes[leftNode].Cluster, nodes[rightNode].Cluster,
                        minimumNormalDot, maximumPlaneOffset,
                        out float score);
                    bool relaxed = !standard && certified &&
                                   TryCertifiedSurfaceLinkScore(
                                       nodes[leftNode].Cluster,
                                       nodes[rightNode].Cluster,
                                       out score);
                    if (certified && pass == 0)
                    {
                        result.PhysicalSurfaceCertificateCandidatePairs++;
                        if (!standard && !relaxed)
                            result.PhysicalSurfaceCertificateRejectedPairs++;
                    }
                    if (!standard && !relaxed)
                        continue;
                    float prioritizedScore = score - (certified ? 4f : 0f);
                    if (prioritizedScore >= bestScore)
                        continue;
                    bestScore = prioritizedScore;
                    bestLeft = leftSlot;
                    bestRight = rightSlot;
                    bestCertified = certified;
                    bestRelaxed = relaxed;
                }
            }
            if (bestLeft < 0 || bestRight < 0)
                break;
            int selectedLeft = left.Get(bestLeft);
            int selectedRight = right.Get(bestRight);
            if (TryUnionSurfaceRoots(
                    parents, unionStates, selectedLeft, selectedRight,
                    bestCertified
                        ? Mathf.Min(minimumNormalDot, CertifiedSurfaceLinkMinimumNormalDot)
                        : minimumNormalDot,
                    bestCertified
                        ? Mathf.Max(maximumPlaneOffset,
                            _voxelSize * CertifiedSurfaceLinkMaximumPlaneOffsetVoxels)
                        : maximumPlaneOffset))
            {
                result.SurfaceComponentLinks++;
                if (bestCertified)
                {
                    result.PhysicalSurfaceCertificateLinks++;
                    if (bestRelaxed)
                        result.PhysicalSurfaceCertificateRelaxedLinks++;
                }
            }
            leftUsedMask |= 1 << bestLeft;
            rightUsedMask |= 1 << bestRight;
        }
    }

    private bool TryPhysicalSurfaceMergeScore(
        Vector3 leftNormal,
        Vector3 leftCentroid,
        Vector3 rightNormal,
        Vector3 rightCentroid,
        float minimumNormalDot,
        float maximumPlaneOffset,
        float maximumTangentDistance,
        out float score,
        out bool oppositeDirection)
    {
        score = float.PositiveInfinity;
        oppositeDirection = false;
        float signedNormalDot = Vector3.Dot(leftNormal, rightNormal);
        float normalDot = Mathf.Abs(signedNormalDot);
        if (normalDot < minimumNormalDot)
            return false;
        oppositeDirection = signedNormalDot < 0f;
        Vector3 alignedRightNormal = oppositeDirection
            ? -rightNormal
            : rightNormal;
        Vector3 averageNormal = leftNormal + alignedRightNormal;
        if (averageNormal.sqrMagnitude <= 0.00000001f)
            return false;
        averageNormal.Normalize();
        Vector3 delta = rightCentroid - leftCentroid;
        float signedPlaneOffset = Vector3.Dot(delta, averageNormal);
        float planeOffset = Mathf.Abs(signedPlaneOffset);
        if (planeOffset > maximumPlaneOffset)
            return false;
        Vector3 tangent = delta - averageNormal * signedPlaneOffset;
        float tangentDistance = tangent.magnitude;
        if (tangentDistance > maximumTangentDistance)
            return false;
        float normalCost = (1f - normalDot) /
                           Mathf.Max(0.0001f, 1f - minimumNormalDot);
        float planeCost = planeOffset /
                          Mathf.Max(0.000001f, maximumPlaneOffset);
        float tangentCost = tangentDistance /
                            Mathf.Max(0.000001f, maximumTangentDistance);
        score = normalCost + planeCost + tangentCost * 0.1f;
        return true;
    }

    private bool TrySurfaceLinkScore(
        ComposedCluster left,
        ComposedCluster right,
        float minimumNormalDot,
        float maximumPlaneOffset,
        out float score)
    {
        score = float.PositiveInfinity;
        float maximumTangent = _voxelSize * 2.25f;
        if (!TryPhysicalSurfaceMergeScore(
                left.Normal, left.Centroid,
                right.Normal, right.Centroid,
                minimumNormalDot, maximumPlaneOffset, maximumTangent,
                out score, out _))
            return false;
        return true;
    }

    private bool TryCertifiedSurfaceLinkScore(
        ComposedCluster left,
        ComposedCluster right,
        out float score)
    {
        return TryPhysicalSurfaceMergeScore(
            left.Normal, left.Centroid,
            right.Normal, right.Centroid,
            CertifiedSurfaceLinkMinimumNormalDot,
            _voxelSize * CertifiedSurfaceLinkMaximumPlaneOffsetVoxels,
            _voxelSize * 2.25f,
            out score, out _);
    }

    private HashSet<ulong> BuildActivePhysicalSurfaceNodePairs(
        List<ComposedSurfaceNode> nodes)
    {
        HashSet<ulong> pairs = new HashSet<ulong>();
        if (nodes == null || nodes.Count == 0 ||
            _physicalSurfaceCertificates.Count == 0)
            return pairs;

        Dictionary<long, List<int>> endpointNodes =
            new Dictionary<long, List<int>>(
                Mathf.Min(nodes.Count * 4, 131072));
        for (int nodeIndex = 0; nodeIndex < nodes.Count; nodeIndex++)
        {
            ComposedSurfaceNode node = nodes[nodeIndex];
            for (int layerChannel = 0;
                 layerChannel < LayerCount;
                 layerChannel++)
            {
                if ((node.Cluster.LayerMask & (1 << layerChannel)) == 0)
                    continue;
                for (int corner = 0; corner < 8; corner++)
                {
                    int voxelIndex = VoxelIndex(
                        node.X + CornerX[corner],
                        node.Y + CornerY[corner],
                        node.Z + CornerZ[corner]);
                    long endpoint =
                        ((long)layerChannel << 32) | (uint)voxelIndex;
                    if (!endpointNodes.TryGetValue(
                            endpoint, out List<int> endpointList))
                    {
                        endpointList = new List<int>(4);
                        endpointNodes.Add(endpoint, endpointList);
                    }
                    endpointList.Add(nodeIndex);
                }
            }
        }

        int oldestActiveBatch =
            _batchSequence - PhysicalSurfaceCertificateLeaseBatches;
        foreach (KeyValuePair<PhysicalSurfaceCertificateKey, PhysicalSurfaceCertificateState>
                 certificate in _physicalSurfaceCertificates)
        {
            if (certificate.Value.LastConfirmedBatch < oldestActiveBatch ||
                !endpointNodes.TryGetValue(
                    certificate.Key.First, out List<int> firstNodes) ||
                !endpointNodes.TryGetValue(
                    certificate.Key.Second, out List<int> secondNodes))
                continue;
            for (int firstIndex = 0; firstIndex < firstNodes.Count; firstIndex++)
            for (int secondIndex = 0; secondIndex < secondNodes.Count; secondIndex++)
            {
                int firstNode = firstNodes[firstIndex];
                int secondNode = secondNodes[secondIndex];
                if (firstNode == secondNode)
                    continue;
                pairs.Add(SurfaceNodePairKey(firstNode, secondNode));
            }
        }
        return pairs;
    }

    private static ulong SurfaceNodePairKey(int left, int right)
    {
        uint minimum = (uint)Mathf.Min(left, right);
        uint maximum = (uint)Mathf.Max(left, right);
        return ((ulong)minimum << 32) | maximum;
    }

    private void LinkCertifiedPhysicalSurfaceNodes(
        List<ComposedSurfaceNode> nodes,
        int[] parents,
        SurfaceUnionState[] unionStates,
        HashSet<ulong> certifiedNodePairs,
        MeshResult result)
    {
        List<CertifiedSurfaceLinkCandidate> candidates =
            new List<CertifiedSurfaceLinkCandidate>(certifiedNodePairs.Count);
        foreach (ulong nodePair in certifiedNodePairs)
        {
            int firstNode = (int)(nodePair >> 32);
            int secondNode = (int)(nodePair & 0xffffffffu);
            if (firstNode < 0 || firstNode >= nodes.Count ||
                secondNode < 0 || secondNode >= nodes.Count)
                continue;
            ComposedSurfaceNode first = nodes[firstNode];
            ComposedSurfaceNode second = nodes[secondNode];
            if (first.CellIndex == second.CellIndex)
                continue;
            int dx = Mathf.Abs(first.X - second.X);
            int dy = Mathf.Abs(first.Y - second.Y);
            int dz = Mathf.Abs(first.Z - second.Z);
            if (Mathf.Max(dx, Mathf.Max(dy, dz)) > 2)
                continue;

            int leftNode = firstNode;
            int rightNode = secondNode;
            int leftCell = first.CellIndex;
            int rightCell = second.CellIndex;
            if (rightCell < leftCell)
            {
                int nodeSwap = leftNode;
                leftNode = rightNode;
                rightNode = nodeSwap;
                int cellSwap = leftCell;
                leftCell = rightCell;
                rightCell = cellSwap;
            }
            result.PhysicalSurfaceCertificateDirectCandidates++;
            if (!TryCertifiedSurfaceLinkScore(
                    nodes[leftNode].Cluster,
                    nodes[rightNode].Cluster,
                    out float score))
            {
                result.PhysicalSurfaceCertificateDirectRejected++;
                continue;
            }
            candidates.Add(new CertifiedSurfaceLinkCandidate
            {
                CellPair = ((ulong)(uint)leftCell << 32) | (uint)rightCell,
                LeftNode = leftNode,
                RightNode = rightNode,
                Score = score
            });
        }
        candidates.Sort((left, right) =>
        {
            int comparison = left.CellPair.CompareTo(right.CellPair);
            if (comparison != 0)
                return comparison;
            comparison = left.Score.CompareTo(right.Score);
            if (comparison != 0)
                return comparison;
            comparison = left.LeftNode.CompareTo(right.LeftNode);
            return comparison != 0
                ? comparison
                : left.RightNode.CompareTo(right.RightNode);
        });

        int groupStart = 0;
        while (groupStart < candidates.Count)
        {
            int groupEnd = groupStart + 1;
            ulong cellPair = candidates[groupStart].CellPair;
            while (groupEnd < candidates.Count &&
                   candidates[groupEnd].CellPair == cellPair)
                groupEnd++;

            HashSet<int> usedLeft = new HashSet<int>();
            HashSet<int> usedRight = new HashSet<int>();
            for (int candidateIndex = groupStart;
                 candidateIndex < groupEnd;
                 candidateIndex++)
            {
                CertifiedSurfaceLinkCandidate candidate =
                    candidates[candidateIndex];
                if (usedLeft.Contains(candidate.LeftNode) ||
                    usedRight.Contains(candidate.RightNode))
                {
                    result.PhysicalSurfaceCertificateAmbiguousPairs++;
                    continue;
                }
                usedLeft.Add(candidate.LeftNode);
                usedRight.Add(candidate.RightNode);
                if (FindSurfaceRoot(parents, candidate.LeftNode) ==
                    FindSurfaceRoot(parents, candidate.RightNode))
                    continue;
                if (TryUnionSurfaceRoots(
                        parents, unionStates,
                        candidate.LeftNode, candidate.RightNode,
                        CertifiedSurfaceLinkMinimumNormalDot,
                        _voxelSize *
                        CertifiedSurfaceLinkMaximumPlaneOffsetVoxels))
                {
                    result.SurfaceComponentLinks++;
                    result.PhysicalSurfaceCertificateLinks++;
                    result.PhysicalSurfaceCertificateDirectLinks++;
                }
                else
                {
                    result.PhysicalSurfaceCertificateDirectRejected++;
                }
            }
            groupStart = groupEnd;
        }
    }

    private static int FindSurfaceRoot(int[] parents, int node)
    {
        int root = node;
        while (parents[root] != root)
            root = parents[root];
        while (parents[node] != node)
        {
            int next = parents[node];
            parents[node] = root;
            node = next;
        }
        return root;
    }

    private static bool TryUnionSurfaceRoots(
        int[] parents,
        SurfaceUnionState[] states,
        int left,
        int right,
        float minimumNormalDot,
        float maximumPlaneOffset)
    {
        int leftRoot = FindSurfaceRoot(parents, left);
        int rightRoot = FindSurfaceRoot(parents, right);
        if (leftRoot == rightRoot)
            return false;
        SurfaceUnionState leftState = states[leftRoot];
        SurfaceUnionState rightState = states[rightRoot];
        Vector3 leftNormal = leftState.Normal;
        Vector3 rightNormal = rightState.Normal;
        float signedNormalDot = Vector3.Dot(leftNormal, rightNormal);
        if (Mathf.Abs(signedNormalDot) < minimumNormalDot)
            return false;
        bool oppositeDirection = signedNormalDot < 0f;
        Vector3 alignedRightNormal = oppositeDirection
            ? -rightNormal
            : rightNormal;
        Vector3 componentNormal = leftNormal + alignedRightNormal;
        if (componentNormal.sqrMagnitude <= 0.00000001f)
            return false;
        componentNormal.Normalize();
        float componentPlaneOffset = Mathf.Abs(Vector3.Dot(
            rightState.Centroid - leftState.Centroid,
            componentNormal));
        if (componentPlaneOffset > maximumPlaneOffset)
            return false;

        // Always retain the lower node index as the root.  Candidate cells are
        // sorted, making component identity deterministic across rebuilds.
        if (rightRoot < leftRoot)
        {
            int swap = leftRoot;
            leftRoot = rightRoot;
            rightRoot = swap;
        }
        parents[rightRoot] = leftRoot;
        if (oppositeDirection)
            rightState.NormalWeightedSum = -rightState.NormalWeightedSum;
        leftState.NormalWeightedSum += rightState.NormalWeightedSum;
        leftState.CentroidWeightedSum += rightState.CentroidWeightedSum;
        leftState.Support += rightState.Support;
        states[leftRoot] = leftState;
        states[rightRoot] = default;
        return true;
    }

    private void EvaluatePaperReferenceDirectionalMarchingCubes(
        List<int> orderedCandidateCells,
        int maximumCells,
        int maximumTriangles,
        float minimumWeight,
        Vector3[] cornerPositions,
        MeshResult result)
    {
        Stopwatch stopwatch = Stopwatch.StartNew();
        result.DirectionalMcShadowEnabled = _canonicalSixDirectionSemantics;
        result.DirectionalMcShadowPaperReference =
            _canonicalSixDirectionSemantics;
        if (!_canonicalSixDirectionSemantics)
        {
            stopwatch.Stop();
            result.DirectionalMcShadowMilliseconds =
                stopwatch.Elapsed.TotalMilliseconds;
            return;
        }

        Dictionary<int, PaperDirectionalMcCell> cells =
            new Dictionary<int, PaperDirectionalMcCell>(
                Mathf.Min(maximumCells, orderedCandidateCells.Count));
        int[] mcIndices = new int[DirectionCount];
        float[] sdfWeights = new float[DirectionCount];
        Vector3[] sdfGradients = new Vector3[DirectionCount];
        float[,] sdfValues = new float[DirectionCount, 8];

        for (int orderedCell = 0;
             orderedCell < orderedCandidateCells.Count &&
             result.DirectionalMcShadowCellsEvaluated < maximumCells;
             orderedCell++)
        {
            int cellIndex = orderedCandidateCells[orderedCell];
            DecodeCell(cellIndex, out int x, out int y, out int z);
            if (x < 0 || y < 0 || z < 0 ||
                x >= _dimX - 1 || y >= _dimY - 1 || z >= _dimZ - 1)
                continue;

            result.DirectionalMcShadowCellsEvaluated++;
            for (int corner = 0; corner < 8; corner++)
            {
                cornerPositions[corner] = VoxelCenter(
                    x + PaperCornerX[corner],
                    y + PaperCornerY[corner],
                    z + PaperCornerZ[corner]);
            }

            int validDirections = 0;
            for (int paperDirection = 0;
                 paperDirection < DirectionCount;
                 paperDirection++)
            {
                mcIndices[paperDirection] = -1;
                sdfWeights[paperDirection] = 0f;
                sdfGradients[paperDirection] = Vector3.zero;
                int scanCoverDirection =
                    PaperDirectionToScanCoverDirection[paperDirection];
                if (!TryReadPaperDirectionCell(
                        scanCoverDirection,
                        x, y, z,
                        minimumWeight,
                        sdfValues,
                        paperDirection,
                        out int mcIndex,
                        out float sdfWeight,
                        out Vector3 sdfGradient))
                {
                    continue;
                }

                validDirections++;
                sdfWeights[paperDirection] = sdfWeight;
                sdfGradients[paperDirection] = sdfGradient;
                if (mcIndex > 0 && mcIndex < byte.MaxValue)
                {
                    result.DirectionalMcShadowRawHypotheses++;
                    result.DirectionalMcShadowRawTransitionEdges +=
                        PopCount(PaperTransitionEdgeMask(mcIndex));
                }

                int filteredIndex = FilterPaperMcIndexDirection(
                    mcIndex, paperDirection, sdfValues);
                if (filteredIndex >= 0 && filteredIndex < byte.MaxValue)
                {
                    Vector3 gradient = sdfGradient.sqrMagnitude >
                        0.00000001f
                            ? sdfGradient.normalized
                            : Vector3.zero;
                    Vector3 directionVector =
                        DirectionVectors[scanCoverDirection];
                    float compliance =
                        Vector3.Dot(gradient, directionVector);
                    sdfWeights[paperDirection] *= compliance;
                    if (compliance < CanonicalDirectionThreshold)
                    {
                        filteredIndex = -1;
                        sdfWeights[paperDirection] = 0f;
                    }
                }
                if (mcIndex > 0 && mcIndex < byte.MaxValue &&
                    filteredIndex < 0)
                {
                    result.DirectionalMcShadowIntraDirectionRejected++;
                }
                mcIndices[paperDirection] = filteredIndex;
            }

            if (validDirections == 0)
            {
                result.DirectionalMcShadowUnknownDeferredCells++;
                continue;
            }

            // Algorithm 1, inter-directional filtering. Preserve the authors'
            // paper/source direction order and sequential update semantics.
            for (int direction = 0; direction < DirectionCount; direction++)
            {
                int mcIndex = mcIndices[direction];
                if (mcIndex <= 0 || mcIndex == byte.MaxValue)
                    continue;
                float directionWeight = sdfWeights[direction];
                if (directionWeight <= 0.00000001f)
                {
                    mcIndices[direction] = -1;
                    result.DirectionalMcShadowInterDirectionRejected++;
                    continue;
                }

                float supportWeight = 1f;
                for (int other = 0; other < DirectionCount; other++)
                {
                    if (mcIndices[other] < 0)
                        continue;
                    if (mcIndices[other] == 0)
                    {
                        supportWeight -=
                            sdfWeights[other] / directionWeight;
                        mcIndices[direction] = 0;
                        break;
                    }
                    if (other != direction &&
                        PaperMcIndexCompatible(
                            mcIndex, mcIndices[other]))
                    {
                        supportWeight +=
                            sdfWeights[other] / directionWeight;
                    }
                }
                if (supportWeight < 0f)
                    mcIndices[direction] = 0;
                if (mcIndices[direction] <= 0)
                    result.DirectionalMcShadowInterDirectionRejected++;
            }

            int keptDirections = 0;
            for (int direction = 0; direction < DirectionCount; direction++)
            {
                if (mcIndices[direction] > 0 &&
                    mcIndices[direction] < byte.MaxValue)
                {
                    keptDirections++;
                }
            }
            if (keptDirections == 0)
            {
                result.DirectionalMcShadowEmptyAfterVotingCells++;
                continue;
            }
            result.DirectionalMcShadowValidHypotheses += keptDirections;

            PaperDirectionalMcCell cell = new PaperDirectionalMcCell
            {
                CellIndex = cellIndex,
                X = x,
                Y = y,
                Z = z
            };
            EvaluatePaperEdgeOffsets(
                mcIndices, sdfValues, sdfWeights, cell, result);

            int combined0 = 0;
            int combined1 = 0;
            for (int direction = 0;
                 direction < DirectionCount;
                 direction++)
            {
                int mcIndex = mcIndices[direction];
                if (mcIndex <= 0 || mcIndex == byte.MaxValue)
                    continue;
                for (int componentIndex = 0;
                     componentIndex <
                         ScanCoverPaperDmcTables.ComponentsPerIndex;
                     componentIndex++)
                {
                    int component = ScanCoverPaperDmcTables.Component(
                        mcIndex, componentIndex);
                    if (component < 0)
                        break;
                    result.DirectionalMcShadowComponents++;
                    bool assigned = false;
                    if (combined0 == 0)
                    {
                        combined0 = component;
                        assigned = true;
                    }
                    else if (PaperMcIndexCompatible(
                                 mcIndex, combined0))
                    {
                        combined0 &= component;
                        assigned = true;
                    }
                    else if (combined1 == 0)
                    {
                        combined1 = component;
                        assigned = true;
                    }
                    else if (PaperMcIndexCompatible(
                                 mcIndex, combined1))
                    {
                        combined1 &= component;
                        assigned = true;
                    }
                    if (!assigned)
                    {
                        result
                            .DirectionalMcShadowOverflowDeferredComponents++;
                    }
                }
            }

            if (combined0 <= 0 || combined0 == byte.MaxValue)
            {
                if (combined1 > 0 && combined1 < byte.MaxValue)
                {
                    combined0 = combined1;
                    combined1 = 0;
                }
                else
                {
                    result.DirectionalMcShadowEmptyAfterVotingCells++;
                    continue;
                }
            }
            if (combined1 == byte.MaxValue)
                combined1 = 0;

            cell.Index0 = (byte)combined0;
            cell.Index1 = (byte)combined1;
            cell.RegularizedIndex0 = cell.Index0;
            cell.RegularizedIndex1 = cell.Index1;
            cell.SurfaceCount = combined1 > 0 ? 2 : 1;
            if (cell.SurfaceCount == 2)
                result.DirectionalMcShadowDoubleSurfaceCells++;
            else
                result.DirectionalMcShadowSingleSurfaceCells++;
            result.DirectionalMcShadowCombinedTransitionEdges +=
                PopCount(PaperTransitionEdgeMask(cell.Index0));
            if (cell.SurfaceCount == 2)
            {
                result.DirectionalMcShadowCombinedTransitionEdges +=
                    PopCount(PaperTransitionEdgeMask(cell.Index1));
            }
            cells.Add(cellIndex, cell);
        }

        AuditPaperNeighborDisagreements(
            cells,
            out int neighborComparisonsBefore,
            out int neighborDisagreementsBefore);
        result.DirectionalMcShadowNeighborFaceComparisons =
            neighborComparisonsBefore;
        result.DirectionalMcShadowNeighborDisagreementsBefore =
            neighborDisagreementsBefore;
        RegularizePaperMcIndices(cells, result);
        AuditPaperNeighborDisagreements(
            cells,
            out int neighborComparisonsAfter,
            out int neighborDisagreementsAfter);
        result.DirectionalMcShadowNeighborDisagreementsAfter =
            neighborDisagreementsAfter;
        result.DirectionalMcShadowNeighborFaceComparisons =
            Mathf.Max(neighborComparisonsBefore, neighborComparisonsAfter);

        // The paper algorithm above remains the sole topology authority.  The
        // persistent wrapper below only decides when a complete observed cell
        // decision may replace the previously published one, and canonicalizes
        // its already-measured edge intersections across cells and batches.
        cells = UpdatePersistentPaperDmcLedger(
            cells,
            orderedCandidateCells.Count <= maximumCells,
            result);

        foreach (PaperDirectionalMcCell cell in cells.Values)
        {
            result.DirectionalMcShadowRegularizedTransitionEdges +=
                PopCount(PaperTransitionEdgeMask(
                    cell.RegularizedIndex0));
            if (cell.SurfaceCount == 2)
            {
                result.DirectionalMcShadowRegularizedTransitionEdges +=
                    PopCount(PaperTransitionEdgeMask(
                        cell.RegularizedIndex1));
            }
        }

        BuildPaperDmcShadowMesh(cells, maximumTriangles, result);
        AuditDirectionalMcShadowMesh(result);
        _lastPaperDmcCells = cells;
        _lastPaperDmcTriangleLimit = maximumTriangles;
        stopwatch.Stop();
        result.DirectionalMcShadowMilliseconds =
            stopwatch.Elapsed.TotalMilliseconds;
    }

    private bool TryReadPaperDirectionCell(
        int scanCoverDirection,
        int x,
        int y,
        int z,
        float minimumWeight,
        float[,] sdfValues,
        int paperDirection,
        out int mcIndex,
        out float sdfWeight,
        out Vector3 sdfGradient)
    {
        mcIndex = 0;
        sdfWeight = 0f;
        sdfGradient = Vector3.zero;
        for (int corner = 0; corner < 8; corner++)
        {
            int vx = x + PaperCornerX[corner];
            int vy = y + PaperCornerY[corner];
            int vz = z + PaperCornerZ[corner];
            if (!TryReadVoxel(
                    scanCoverDirection, vx, vy, vz,
                    out float value, out float weight) ||
                weight < minimumWeight)
            {
                mcIndex = -1;
                sdfWeight = 0f;
                return false;
            }
            sdfValues[paperDirection, corner] = value;
            sdfWeight += weight * 0.125f;
            if (value < 0f)
                mcIndex |= 1 << corner;
        }

        float gx = 0f;
        float gy = 0f;
        float gz = 0f;
        for (int corner = 0; corner < 8; corner++)
        {
            float value = sdfValues[paperDirection, corner];
            gx += value * (PaperCornerX[corner] * 2f - 1f) * 0.25f;
            gy += value * (PaperCornerY[corner] * 2f - 1f) * 0.25f;
            gz += value * (PaperCornerZ[corner] * 2f - 1f) * 0.25f;
        }
        sdfGradient = new Vector3(gx, gy, gz);
        return true;
    }

    private static int FilterPaperMcIndexDirection(
        int mcIndex,
        int paperDirection,
        float[,] sdfValues)
    {
        if (mcIndex <= 0 || mcIndex == byte.MaxValue)
            return mcIndex;
        int filteredIndex = 0;
        for (int componentIndex = 0;
             componentIndex < ScanCoverPaperDmcTables.ComponentsPerIndex;
             componentIndex++)
        {
            int component = ScanCoverPaperDmcTables.Component(
                mcIndex, componentIndex);
            if (component < 0)
                break;
            if (IsPaperMcIndexDirectionCompatible(
                    component, paperDirection, sdfValues))
            {
                filteredIndex |= component;
            }
        }
        return filteredIndex == 0 ? -1 : filteredIndex;
    }

    private static bool IsPaperMcIndexDirectionCompatible(
        int mcIndex,
        int paperDirection,
        float[,] sdfValues)
    {
        int compatibility = ScanCoverPaperDmcTables.Compatibility(
            mcIndex, paperDirection);
        if (compatibility == 0)
            return false;
        if (compatibility != 2)
            return true;

        for (int pair = 0; pair < 4; pair++)
        {
            int edgeIndex =
                PaperDirectionEdgesToCheck[paperDirection, pair * 2];
            int oppositeEdgeIndex =
                PaperDirectionEdgesToCheck[paperDirection, pair * 2 + 1];
            int edgeA = PaperEdgeEndpointCorners[edgeIndex, 0];
            int edgeB = PaperEdgeEndpointCorners[edgeIndex, 1];
            int oppositeA =
                PaperEdgeEndpointCorners[oppositeEdgeIndex, 0];
            int oppositeB =
                PaperEdgeEndpointCorners[oppositeEdgeIndex, 1];
            bool edgeAInside = (mcIndex & (1 << edgeA)) != 0;
            bool edgeBInside = (mcIndex & (1 << edgeB)) != 0;
            if (edgeAInside == edgeBInside)
                continue;
            if (edgeBInside)
            {
                Swap(ref edgeA, ref edgeB);
                Swap(ref oppositeA, ref oppositeB);
            }
            float offset = PaperSurfaceOffset(
                sdfValues[paperDirection, edgeA],
                sdfValues[paperDirection, edgeB]);
            float oppositeOffset = PaperSurfaceOffset(
                sdfValues[paperDirection, oppositeA],
                sdfValues[paperDirection, oppositeB]);
            if (offset > oppositeOffset)
                return false;
        }
        return true;
    }

    private static bool PaperMcEdgeCompatible(
        int firstIndex,
        int secondIndex,
        int edge)
    {
        int a = PaperEdgeEndpointCorners[edge, 0];
        int b = PaperEdgeEndpointCorners[edge, 1];
        bool firstA = (firstIndex & (1 << a)) != 0;
        bool firstB = (firstIndex & (1 << b)) != 0;
        bool secondA = (secondIndex & (1 << a)) != 0;
        bool secondB = (secondIndex & (1 << b)) != 0;
        return !(
            (firstA && !firstB && !secondA && secondB) ||
            (!firstA && firstB && secondA && !secondB));
    }

    private static bool PaperMcIndexCompatible(
        int firstIndex,
        int secondIndex)
    {
        bool compatible = false;
        int intersection = firstIndex & secondIndex;
        if (intersection != 0)
        {
            for (int edge = 0; edge < 12; edge++)
            {
                int a = PaperEdgeEndpointCorners[edge, 0];
                int b = PaperEdgeEndpointCorners[edge, 1];
                if ((intersection & (1 << a)) == 0 ||
                    (intersection & (1 << b)) == 0)
                {
                    if (PaperMcEdgeCompatible(
                            firstIndex, secondIndex, edge))
                    {
                        compatible = true;
                    }
                }
            }
        }
        else
        {
            compatible = true;
            for (int edge = 0; edge < 12; edge++)
            {
                compatible &= PaperMcEdgeCompatible(
                    firstIndex, secondIndex, edge);
            }
        }
        return compatible;
    }

    private static void EvaluatePaperEdgeOffsets(
        int[] mcIndices,
        float[,] sdfValues,
        float[] sdfWeights,
        PaperDirectionalMcCell cell,
        MeshResult result)
    {
        for (int edge = 0; edge < 12; edge++)
        {
            int a = PaperEdgeEndpointCorners[edge, 0];
            int b = PaperEdgeEndpointCorners[edge, 1];
            float numerator0 = 0f;
            float numerator1 = 0f;
            float weight0 = 0f;
            float weight1 = 0f;
            for (int direction = 0;
                 direction < DirectionCount;
                 direction++)
            {
                int mcIndex = mcIndices[direction];
                if (mcIndex < 0 ||
                    (PaperTransitionEdgeMask(mcIndex) &
                     (1 << edge)) == 0)
                {
                    continue;
                }
                float weight = sdfWeights[direction];
                if (weight <= 0f)
                    continue;
                float offset = PaperSurfaceOffset(
                    sdfValues[direction, a],
                    sdfValues[direction, b]);
                if ((mcIndex & (1 << a)) != 0)
                {
                    numerator0 += weight * offset;
                    weight0 += weight;
                }
                else
                {
                    numerator1 += weight * offset;
                    weight1 += weight;
                }
            }

            if (weight0 > 0f)
            {
                cell.EdgeSlotMask[edge] |= 1;
                cell.EdgeOffsets[edge * 2] =
                    numerator0 / weight0;
                cell.EdgeWeights[edge * 2] = weight0;
                result.DirectionalMcShadowOffsetClusters++;
            }
            if (weight1 > 0f)
            {
                cell.EdgeSlotMask[edge] |= 2;
                cell.EdgeOffsets[edge * 2 + 1] =
                    numerator1 / weight1;
                cell.EdgeWeights[edge * 2 + 1] = weight1;
                result.DirectionalMcShadowOffsetClusters++;
            }
            if ((cell.EdgeSlotMask[edge] & 3) == 3)
            {
                result.DirectionalMcShadowDualOffsetEdges++;
                float first = cell.EdgeOffsets[edge * 2];
                float second = cell.EdgeOffsets[edge * 2 + 1];
                if (second < first)
                {
                    float mean = (first + second) * 0.5f;
                    cell.EdgeOffsets[edge * 2] = mean;
                    cell.EdgeOffsets[edge * 2 + 1] = mean;
                }
            }
        }
    }

    private void RegularizePaperMcIndices(
        Dictionary<int, PaperDirectionalMcCell> cells,
        MeshResult result)
    {
        Dictionary<int, byte> sourceIndices =
            new Dictionary<int, byte>(cells.Count);
        foreach (KeyValuePair<int, PaperDirectionalMcCell> pair in cells)
            sourceIndices.Add(pair.Key, pair.Value.Index0);

        foreach (PaperDirectionalMcCell cell in cells.Values)
        {
            int sourceIndex = cell.Index0;
            if (sourceIndex <= 0 || sourceIndex == byte.MaxValue)
                continue;
            int regularized = sourceIndex;
            for (int corner = 0; corner < 8; corner++)
            {
                int physicalX = cell.X + PaperCornerX[corner];
                int physicalY = cell.Y + PaperCornerY[corner];
                int physicalZ = cell.Z + PaperCornerZ[corner];
                int insideVotes =
                    (sourceIndex & (1 << corner)) != 0 ? 1 : 0;
                int votes = 1;
                for (int neighborCorner = 0;
                     neighborCorner < 8;
                     neighborCorner++)
                {
                    int neighborX =
                        physicalX - PaperCornerX[neighborCorner];
                    int neighborY =
                        physicalY - PaperCornerY[neighborCorner];
                    int neighborZ =
                        physicalZ - PaperCornerZ[neighborCorner];
                    if (neighborX == cell.X &&
                        neighborY == cell.Y &&
                        neighborZ == cell.Z)
                    {
                        continue;
                    }
                    if (neighborX < 0 || neighborY < 0 ||
                        neighborZ < 0 ||
                        neighborX >= _dimX - 1 ||
                        neighborY >= _dimY - 1 ||
                        neighborZ >= _dimZ - 1)
                    {
                        continue;
                    }
                    int neighborCellIndex =
                        CellIndex(neighborX, neighborY, neighborZ);
                    if (!sourceIndices.TryGetValue(
                            neighborCellIndex,
                            out byte neighborIndex) ||
                        neighborIndex <= 0)
                    {
                        continue;
                    }
                    if ((neighborIndex & (1 << neighborCorner)) != 0)
                        insideVotes++;
                    votes++;
                }
                bool inside = insideVotes > votes / 2;
                bool wasInside =
                    (sourceIndex & (1 << corner)) != 0;
                if (inside == wasInside)
                    continue;
                regularized = inside
                    ? regularized | (1 << corner)
                    : regularized & ~(1 << corner);
                result.DirectionalMcShadowRegularizedCorners++;
            }
            cell.RegularizedIndex0 = (byte)regularized;
            if (cell.RegularizedIndex0 != cell.Index0)
                result.DirectionalMcShadowRegularizedCells++;
        }
    }

    private void AuditPaperNeighborDisagreements(
        Dictionary<int, PaperDirectionalMcCell> cells,
        out int comparisons,
        out int disagreements)
    {
        comparisons = 0;
        disagreements = 0;
        foreach (PaperDirectionalMcCell left in cells.Values)
        {
            for (int axis = 0; axis < 3; axis++)
            {
                int nx = left.X + (axis == 0 ? 1 : 0);
                int ny = left.Y + (axis == 1 ? 1 : 0);
                int nz = left.Z + (axis == 2 ? 1 : 0);
                if (!cells.TryGetValue(
                        CellIndex(nx, ny, nz),
                        out PaperDirectionalMcCell right))
                {
                    continue;
                }
                comparisons++;
                disagreements += CountPaperMcFaceDisagreements(
                    left.RegularizedIndex0,
                    right.RegularizedIndex0,
                    axis);
            }
        }
    }

    private Dictionary<int, PaperDirectionalMcCell>
        UpdatePersistentPaperDmcLedger(
            Dictionary<int, PaperDirectionalMcCell> observedCells,
            bool completeTraversal,
            MeshResult result)
    {
        bool advanceBatch =
            _lastPaperDmcLedgerUpdateBatch != _batchSequence;
        if (advanceBatch)
            _lastPaperDmcLedgerUpdateBatch = _batchSequence;

        HashSet<int> seen = new HashSet<int>();
        HashSet<int> readyChanges = new HashSet<int>();
        List<int> orderedObserved =
            new List<int>(observedCells.Keys);
        orderedObserved.Sort();
        for (int i = 0; i < orderedObserved.Count; i++)
        {
            int cellIndex = orderedObserved[i];
            PaperDirectionalMcCell observed =
                observedCells[cellIndex];
            seen.Add(cellIndex);
            if (!_paperDmcPersistentCells.TryGetValue(
                    cellIndex,
                    out PaperDmcPersistentCellState state))
            {
                state = new PaperDmcPersistentCellState
                {
                    Committed = ClonePaperDmcCell(observed),
                    LastObservedBatch = _batchSequence
                };
                _paperDmcPersistentCells.Add(cellIndex, state);
                continue;
            }

            state.LastObservedBatch = _batchSequence;
            state.MissingBatches = 0;
            if (SamePaperDmcTopology(state.Committed, observed))
            {
                PaperDirectionalMcCell refreshed =
                    ClonePaperDmcCell(observed);
                PreserveMissingPaperDmcEdgeEvidence(
                    state.Committed, refreshed);
                state.Committed = refreshed;
                state.Pending = null;
                state.PendingStableBatches = 0;
                state.PendingLastObservedBatch = 0;
                continue;
            }

            if (!advanceBatch)
                continue;
            if (state.Pending != null &&
                SamePaperDmcTopology(state.Pending, observed) &&
                state.PendingLastObservedBatch == _batchSequence - 1)
            {
                state.Pending = ClonePaperDmcCell(observed);
                state.PendingStableBatches++;
            }
            else
            {
                state.Pending = ClonePaperDmcCell(observed);
                state.PendingStableBatches = 1;
            }
            state.PendingLastObservedBatch = _batchSequence;
            if (state.PendingStableBatches >=
                PaperPersistentTopologyConfirmationBatches)
            {
                readyChanges.Add(cellIndex);
            }
        }

        HashSet<int> acceptedChanges =
            SelectAtomicPaperDmcChanges(readyChanges);
        foreach (int cellIndex in acceptedChanges)
        {
            PaperDmcPersistentCellState state =
                _paperDmcPersistentCells[cellIndex];
            state.Committed = ClonePaperDmcCell(state.Pending);
            state.Pending = null;
            state.PendingStableBatches = 0;
            state.PendingLastObservedBatch = 0;
            result.DirectionalMcShadowChangedDecisions++;
            result.DirectionalMcShadowAtomicCommittedCells++;
        }
        result.DirectionalMcShadowAtomicDeferredCells +=
            Mathf.Max(0, readyChanges.Count - acceptedChanges.Count);

        if (completeTraversal && advanceBatch)
        {
            List<int> retire = null;
            foreach (KeyValuePair<int, PaperDmcPersistentCellState> pair in
                     _paperDmcPersistentCells)
            {
                if (seen.Contains(pair.Key))
                    continue;
                pair.Value.MissingBatches++;
                if (pair.Value.MissingBatches <
                    PaperPersistentMissingCellLeaseBatches)
                {
                    continue;
                }
                if (retire == null)
                    retire = new List<int>();
                retire.Add(pair.Key);
            }
            if (retire != null)
            {
                for (int i = 0; i < retire.Count; i++)
                    _paperDmcPersistentCells.Remove(retire[i]);
                result.DirectionalMcShadowRetiredDecisions = retire.Count;
            }
        }

        Dictionary<int, PaperDirectionalMcCell> committedCells =
            new Dictionary<int, PaperDirectionalMcCell>(
                _paperDmcPersistentCells.Count);
        int pendingChanges = 0;
        foreach (KeyValuePair<int, PaperDmcPersistentCellState> pair in
                 _paperDmcPersistentCells)
        {
            committedCells.Add(
                pair.Key,
                ClonePaperDmcCell(pair.Value.Committed));
            if (pair.Value.Pending != null)
                pendingChanges++;
        }

        AssignPersistentPaperDmcSurfaceIdentities(
            committedCells, result);
        CanonicalizePersistentPaperDmcEdges(
            committedCells, advanceBatch, result);

        result.DirectionalMcShadowPersistentDecisions =
            committedCells.Count;
        result.DirectionalMcShadowPendingTopologyChanges =
            pendingChanges;
        if (advanceBatch)
        {
            _paperDmcTopologyStableBatches = pendingChanges == 0
                ? _paperDmcTopologyStableBatches + 1
                : 0;
        }
        result.DirectionalMcShadowTopologyStableBatches =
            _paperDmcTopologyStableBatches;
        return committedCells;
    }

    private HashSet<int> SelectAtomicPaperDmcChanges(
        HashSet<int> readyChanges)
    {
        HashSet<int> accepted = new HashSet<int>(readyChanges);
        if (accepted.Count == 0)
            return accepted;

        bool removed;
        do
        {
            removed = false;
            List<int> reject = new List<int>();
            foreach (int cellIndex in accepted)
            {
                PaperDmcPersistentCellState state =
                    _paperDmcPersistentCells[cellIndex];
                PaperDirectionalMcCell before = state.Committed;
                PaperDirectionalMcCell after = state.Pending;
                for (int axis = 0; axis < 3; axis++)
                {
                    for (int side = -1; side <= 1; side += 2)
                    {
                        if (!TryPaperDmcNeighborCellIndex(
                                after, axis, side,
                                out int neighborIndex) ||
                            !_paperDmcPersistentCells.TryGetValue(
                                neighborIndex,
                                out PaperDmcPersistentCellState neighborState))
                        {
                            continue;
                        }
                        PaperDirectionalMcCell neighborBefore =
                            neighborState.Committed;
                        PaperDirectionalMcCell neighborAfter =
                            accepted.Contains(neighborIndex) &&
                            neighborState.Pending != null
                                ? neighborState.Pending
                                : neighborBefore;
                        int beforeDisagreements = side > 0
                            ? CountPaperMcFaceDisagreements(
                                before.RegularizedIndex0,
                                neighborBefore.RegularizedIndex0,
                                axis)
                            : CountPaperMcFaceDisagreements(
                                neighborBefore.RegularizedIndex0,
                                before.RegularizedIndex0,
                                axis);
                        int afterDisagreements = side > 0
                            ? CountPaperMcFaceDisagreements(
                                after.RegularizedIndex0,
                                neighborAfter.RegularizedIndex0,
                                axis)
                            : CountPaperMcFaceDisagreements(
                                neighborAfter.RegularizedIndex0,
                                after.RegularizedIndex0,
                                axis);
                        if (afterDisagreements <= beforeDisagreements)
                            continue;
                        reject.Add(cellIndex);
                        if (accepted.Contains(neighborIndex))
                            reject.Add(neighborIndex);
                    }
                }
            }
            for (int i = 0; i < reject.Count; i++)
                removed |= accepted.Remove(reject[i]);
        }
        while (removed && accepted.Count > 0);
        return accepted;
    }

    private bool TryPaperDmcNeighborCellIndex(
        PaperDirectionalMcCell cell,
        int axis,
        int side,
        out int cellIndex)
    {
        int x = cell.X + (axis == 0 ? side : 0);
        int y = cell.Y + (axis == 1 ? side : 0);
        int z = cell.Z + (axis == 2 ? side : 0);
        if (x < 0 || y < 0 || z < 0 ||
            x >= _dimX - 1 || y >= _dimY - 1 || z >= _dimZ - 1)
        {
            cellIndex = -1;
            return false;
        }
        cellIndex = CellIndex(x, y, z);
        return true;
    }

    private static bool SamePaperDmcTopology(
        PaperDirectionalMcCell first,
        PaperDirectionalMcCell second)
    {
        return first != null && second != null &&
            first.SurfaceCount == second.SurfaceCount &&
            first.RegularizedIndex0 == second.RegularizedIndex0 &&
            first.RegularizedIndex1 == second.RegularizedIndex1;
    }

    private static PaperDirectionalMcCell ClonePaperDmcCell(
        PaperDirectionalMcCell source)
    {
        if (source == null)
            return null;
        PaperDirectionalMcCell clone = new PaperDirectionalMcCell
        {
            CellIndex = source.CellIndex,
            X = source.X,
            Y = source.Y,
            Z = source.Z,
            SurfaceCount = source.SurfaceCount,
            Index0 = source.Index0,
            Index1 = source.Index1,
            RegularizedIndex0 = source.RegularizedIndex0,
            RegularizedIndex1 = source.RegularizedIndex1,
            SurfaceIdentity0 = source.SurfaceIdentity0,
            SurfaceIdentity1 = source.SurfaceIdentity1
        };
        Array.Copy(source.EdgeSlotMask, clone.EdgeSlotMask, 12);
        Array.Copy(source.EdgeOffsets, clone.EdgeOffsets, 24);
        Array.Copy(source.EdgeWeights, clone.EdgeWeights, 24);
        return clone;
    }

    private static void PreserveMissingPaperDmcEdgeEvidence(
        PaperDirectionalMcCell previous,
        PaperDirectionalMcCell current)
    {
        if (previous == null || current == null)
            return;
        for (int edge = 0; edge < 12; edge++)
        {
            for (int slot = 0; slot < 2; slot++)
            {
                byte bit = (byte)(1 << slot);
                if ((current.EdgeSlotMask[edge] & bit) != 0 ||
                    (previous.EdgeSlotMask[edge] & bit) == 0)
                {
                    continue;
                }
                current.EdgeSlotMask[edge] |= bit;
                int offsetIndex = edge * 2 + slot;
                current.EdgeOffsets[offsetIndex] =
                    previous.EdgeOffsets[offsetIndex];
                current.EdgeWeights[offsetIndex] =
                    previous.EdgeWeights[offsetIndex];
            }
        }
    }

    private void AssignPersistentPaperDmcSurfaceIdentities(
        Dictionary<int, PaperDirectionalMcCell> cells,
        MeshResult result)
    {
        List<PaperDmcSurfaceRef> surfaces =
            new List<PaperDmcSurfaceRef>(cells.Count * 2);
        List<int> ordered = new List<int>(cells.Keys);
        ordered.Sort();
        for (int i = 0; i < ordered.Count; i++)
        {
            PaperDirectionalMcCell cell = cells[ordered[i]];
            for (int surface = 0; surface < cell.SurfaceCount; surface++)
                surfaces.Add(new PaperDmcSurfaceRef(cell, surface));
        }
        int[] parents = new int[surfaces.Count];
        for (int i = 0; i < parents.Length; i++)
            parents[i] = i;
        Dictionary<DmcEdgeVertexKey, int> firstSurfaceByEdge =
            new Dictionary<DmcEdgeVertexKey, int>(surfaces.Count * 4);
        for (int node = 0; node < surfaces.Count; node++)
        {
            PaperDmcSurfaceRef surface = surfaces[node];
            int mcIndex = surface.Surface == 0
                ? surface.Cell.RegularizedIndex0
                : surface.Cell.RegularizedIndex1;
            ushort edgeMask = PaperTransitionEdgeMask(mcIndex);
            for (int edge = 0; edge < 12; edge++)
            {
                if ((edgeMask & (1 << edge)) == 0 ||
                    !TryPaperDmcEdgeKey(
                        surface.Cell, mcIndex, edge,
                        out DmcEdgeVertexKey key,
                        out int unusedSlot))
                {
                    continue;
                }
                if (firstSurfaceByEdge.TryGetValue(
                        key, out int previousNode))
                {
                    UnionPaperDmcSurfaceRoots(
                        parents, node, previousNode);
                }
                else
                {
                    firstSurfaceByEdge.Add(key, node);
                }
            }
        }

        Dictionary<int, int> identityByRoot =
            new Dictionary<int, int>();
        for (int node = 0; node < surfaces.Count; node++)
        {
            int root = FindPaperDmcSurfaceRoot(parents, node);
            PaperDmcSurfaceRef surface = surfaces[node];
            int mcIndex = surface.Surface == 0
                ? surface.Cell.RegularizedIndex0
                : surface.Cell.RegularizedIndex1;
            ushort edgeMask = PaperTransitionEdgeMask(mcIndex);
            for (int edge = 0; edge < 12; edge++)
            {
                if ((edgeMask & (1 << edge)) == 0 ||
                    !TryPaperDmcEdgeKey(
                        surface.Cell, mcIndex, edge,
                        out DmcEdgeVertexKey key,
                        out int unusedSlot) ||
                    !_paperDmcCanonicalEdges.TryGetValue(
                        key, out PaperDmcCanonicalEdgeState edgeState) ||
                    edgeState.SurfaceIdentity <= 0)
                {
                    continue;
                }
                if (!identityByRoot.TryGetValue(root, out int identity) ||
                    edgeState.SurfaceIdentity < identity)
                {
                    identityByRoot[root] = edgeState.SurfaceIdentity;
                }
            }
        }
        HashSet<int> activeIdentities = new HashSet<int>();
        for (int node = 0; node < surfaces.Count; node++)
        {
            int root = FindPaperDmcSurfaceRoot(parents, node);
            if (!identityByRoot.TryGetValue(root, out int identity))
            {
                identity = _nextPaperDmcSurfaceIdentity++;
                identityByRoot.Add(root, identity);
            }
            PaperDmcSurfaceRef surface = surfaces[node];
            if (surface.Surface == 0)
                surface.Cell.SurfaceIdentity0 = identity;
            else
                surface.Cell.SurfaceIdentity1 = identity;
            activeIdentities.Add(identity);
        }
        result.DirectionalMcShadowPersistentSurfaceIdentities =
            activeIdentities.Count;
    }

    private static int FindPaperDmcSurfaceRoot(int[] parents, int node)
    {
        int root = node;
        while (parents[root] != root)
            root = parents[root];
        while (parents[node] != node)
        {
            int next = parents[node];
            parents[node] = root;
            node = next;
        }
        return root;
    }

    private static void UnionPaperDmcSurfaceRoots(
        int[] parents,
        int first,
        int second)
    {
        int firstRoot = FindPaperDmcSurfaceRoot(parents, first);
        int secondRoot = FindPaperDmcSurfaceRoot(parents, second);
        if (firstRoot == secondRoot)
            return;
        if (secondRoot < firstRoot)
        {
            int swap = firstRoot;
            firstRoot = secondRoot;
            secondRoot = swap;
        }
        parents[secondRoot] = firstRoot;
    }

    private void CanonicalizePersistentPaperDmcEdges(
        Dictionary<int, PaperDirectionalMcCell> cells,
        bool advanceBatch,
        MeshResult result)
    {
        Dictionary<DmcEdgeVertexKey, PaperDmcEdgeObservation> observations =
            new Dictionary<DmcEdgeVertexKey, PaperDmcEdgeObservation>(
                cells.Count * 4);
        foreach (PaperDirectionalMcCell cell in cells.Values)
        {
            for (int surface = 0; surface < cell.SurfaceCount; surface++)
            {
                int mcIndex = surface == 0
                    ? cell.RegularizedIndex0
                    : cell.RegularizedIndex1;
                int identity = surface == 0
                    ? cell.SurfaceIdentity0
                    : cell.SurfaceIdentity1;
                ushort edgeMask = PaperTransitionEdgeMask(mcIndex);
                for (int edge = 0; edge < 12; edge++)
                {
                    if ((edgeMask & (1 << edge)) == 0 ||
                        !TryPaperDmcEdgeKey(
                            cell, mcIndex, edge,
                            out DmcEdgeVertexKey key,
                            out int slot) ||
                        (cell.EdgeSlotMask[edge] & (1 << slot)) == 0)
                    {
                        continue;
                    }
                    int offsetIndex = edge * 2 + slot;
                    float weight = Mathf.Max(
                        0.0001f, cell.EdgeWeights[offsetIndex]);
                    observations.TryGetValue(
                        key, out PaperDmcEdgeObservation observation);
                    observation.WeightedOffsetSum +=
                        cell.EdgeOffsets[offsetIndex] * weight;
                    observation.Weight += weight;
                    observation.SurfaceIdentity = identity;
                    observations[key] = observation;
                }
            }
        }

        foreach (KeyValuePair<DmcEdgeVertexKey, PaperDmcEdgeObservation> pair in
                 observations)
        {
            PaperDmcEdgeObservation observation = pair.Value;
            float observedOffset = Mathf.Clamp01(
                observation.WeightedOffsetSum /
                Mathf.Max(0.0001f, observation.Weight));
            if (!_paperDmcCanonicalEdges.TryGetValue(
                    pair.Key,
                    out PaperDmcCanonicalEdgeState state))
            {
                state = new PaperDmcCanonicalEdgeState
                {
                    CommittedOffset = observedOffset,
                    CommittedWeight = observation.Weight,
                    SurfaceIdentity = observation.SurfaceIdentity,
                    LastObservedBatch = _batchSequence
                };
                _paperDmcCanonicalEdges.Add(pair.Key, state);
                continue;
            }
            if (!advanceBatch || state.LastObservedBatch == _batchSequence)
                continue;

            float delta = Mathf.Abs(
                observedOffset - state.CommittedOffset);
            bool sameIdentity =
                state.SurfaceIdentity == observation.SurfaceIdentity;
            if (sameIdentity &&
                delta <= PaperCanonicalEdgeJitterVoxels)
            {
                float retainedWeight = Mathf.Min(
                    state.CommittedWeight,
                    observation.Weight * 4f);
                float totalWeight =
                    Mathf.Max(0.0001f, retainedWeight + observation.Weight);
                state.CommittedOffset =
                    (state.CommittedOffset * retainedWeight +
                     observedOffset * observation.Weight) / totalWeight;
                state.CommittedWeight = totalWeight;
                state.PendingStableBatches = 0;
                state.PendingLastObservedBatch = 0;
            }
            else
            {
                bool samePending =
                    state.PendingStableBatches > 0 &&
                    state.PendingSurfaceIdentity == observation.SurfaceIdentity &&
                    Mathf.Abs(state.PendingOffset - observedOffset) <=
                        PaperCanonicalEdgePendingMatchVoxels &&
                    state.PendingLastObservedBatch == _batchSequence - 1;
                if (samePending)
                {
                    state.PendingStableBatches++;
                    state.PendingOffset =
                        (state.PendingOffset * state.PendingWeight +
                         observedOffset * observation.Weight) /
                        Mathf.Max(
                            0.0001f,
                            state.PendingWeight + observation.Weight);
                    state.PendingWeight += observation.Weight;
                }
                else
                {
                    state.PendingOffset = observedOffset;
                    state.PendingWeight = observation.Weight;
                    state.PendingSurfaceIdentity =
                        observation.SurfaceIdentity;
                    state.PendingStableBatches = 1;
                }
                state.PendingLastObservedBatch = _batchSequence;
                if (state.PendingStableBatches >=
                    PaperCanonicalEdgeConfirmationBatches)
                {
                    state.CommittedOffset = state.PendingOffset;
                    state.CommittedWeight = state.PendingWeight;
                    state.SurfaceIdentity = state.PendingSurfaceIdentity;
                    state.PendingStableBatches = 0;
                    state.PendingLastObservedBatch = 0;
                    result
                        .DirectionalMcShadowCanonicalEdgeCorrections++;
                }
            }
            state.LastObservedBatch = _batchSequence;
            _paperDmcCanonicalEdges[pair.Key] = state;
        }

        if (advanceBatch)
        {
            List<DmcEdgeVertexKey> retire = null;
            foreach (KeyValuePair<DmcEdgeVertexKey, PaperDmcCanonicalEdgeState>
                     pair in _paperDmcCanonicalEdges)
            {
                if (_batchSequence - pair.Value.LastObservedBatch <=
                    PaperCanonicalEdgeLeaseBatches)
                {
                    continue;
                }
                if (retire == null)
                    retire = new List<DmcEdgeVertexKey>();
                retire.Add(pair.Key);
            }
            if (retire != null)
            {
                for (int i = 0; i < retire.Count; i++)
                    _paperDmcCanonicalEdges.Remove(retire[i]);
            }
        }

        foreach (PaperDirectionalMcCell cell in cells.Values)
        {
            for (int surface = 0; surface < cell.SurfaceCount; surface++)
            {
                int mcIndex = surface == 0
                    ? cell.RegularizedIndex0
                    : cell.RegularizedIndex1;
                ushort edgeMask = PaperTransitionEdgeMask(mcIndex);
                for (int edge = 0; edge < 12; edge++)
                {
                    if ((edgeMask & (1 << edge)) == 0 ||
                        !TryPaperDmcEdgeKey(
                            cell, mcIndex, edge,
                            out DmcEdgeVertexKey key,
                            out int slot) ||
                        !_paperDmcCanonicalEdges.TryGetValue(
                            key, out PaperDmcCanonicalEdgeState state))
                    {
                        continue;
                    }
                    cell.EdgeSlotMask[edge] |= (byte)(1 << slot);
                    int offsetIndex = edge * 2 + slot;
                    cell.EdgeOffsets[offsetIndex] =
                        state.CommittedOffset;
                    cell.EdgeWeights[offsetIndex] = Mathf.Max(
                        cell.EdgeWeights[offsetIndex],
                        state.CommittedWeight);
                }
            }
        }
        result.DirectionalMcShadowPersistentEdges =
            _paperDmcCanonicalEdges.Count;
    }

    private bool TryPaperDmcEdgeKey(
        PaperDirectionalMcCell cell,
        int mcIndex,
        int edge,
        out DmcEdgeVertexKey key,
        out int slot)
    {
        key = default;
        slot = 0;
        if (cell == null || edge < 0 || edge >= 12)
            return false;
        int endpointA = PaperEdgeEndpointCorners[edge, 0];
        int endpointB = PaperEdgeEndpointCorners[edge, 1];
        slot = (mcIndex & (1 << endpointA)) != 0 ? 0 : 1;
        int ax = cell.X + PaperCornerX[endpointA];
        int ay = cell.Y + PaperCornerY[endpointA];
        int az = cell.Z + PaperCornerZ[endpointA];
        int bx = cell.X + PaperCornerX[endpointB];
        int by = cell.Y + PaperCornerY[endpointB];
        int bz = cell.Z + PaperCornerZ[endpointB];
        int voxelA = ax + _dimX * (ay + _dimY * az);
        int voxelB = bx + _dimX * (by + _dimY * bz);
        int minimumVoxel = Mathf.Min(voxelA, voxelB);
        byte axis = (byte)(ax != bx ? 0 : ay != by ? 1 : 2);
        key = new DmcEdgeVertexKey(
            minimumVoxel, axis, (byte)slot);
        return true;
    }

    private void BuildPaperDmcShadowMesh(
        Dictionary<int, PaperDirectionalMcCell> cells,
        int maximumTriangles,
        MeshResult result,
        HashSet<int> suppressedCells = null)
    {
        result.DmcShadowVertices.Clear();
        result.DmcShadowTriangles.Clear();
        result.DmcShadowLines.Clear();
        result.DmcShadowTriangleCells.Clear();
        Dictionary<DmcEdgeVertexKey, int> edgeVertices =
            new Dictionary<DmcEdgeVertexKey, int>(32768);
        int[] localVertices = new int[12];
        int triangleLimit = Mathf.Max(1, maximumTriangles);
        List<int> ordered = new List<int>(cells.Keys);
        ordered.Sort();

        for (int cellPosition = 0;
             cellPosition < ordered.Count &&
             result.DmcShadowTriangles.Count / 3 < triangleLimit;
             cellPosition++)
        {
            PaperDirectionalMcCell cell = cells[ordered[cellPosition]];
            if (suppressedCells != null &&
                suppressedCells.Contains(cell.CellIndex))
            {
                continue;
            }
            for (int surface = 0;
                 surface < cell.SurfaceCount &&
                 result.DmcShadowTriangles.Count / 3 < triangleLimit;
                 surface++)
            {
                int mcIndex = surface == 0
                    ? cell.RegularizedIndex0
                    : cell.RegularizedIndex1;
                if (mcIndex <= 0 || mcIndex == byte.MaxValue)
                    continue;
                for (int i = 0; i < localVertices.Length; i++)
                    localVertices[i] = -1;
                for (int entry = 0;
                     entry <
                         ScanCoverPaperDmcTables.TriangleEntriesPerIndex;
                     entry += 3)
                {
                    int edgeA = ScanCoverPaperDmcTables.TriangleEdge(
                        mcIndex, entry);
                    if (edgeA < 0)
                        break;
                    if (result.DmcShadowTriangles.Count / 3 >=
                        triangleLimit)
                    {
                        break;
                    }
                    int edgeB = ScanCoverPaperDmcTables.TriangleEdge(
                        mcIndex, entry + 1);
                    int edgeC = ScanCoverPaperDmcTables.TriangleEdge(
                        mcIndex, entry + 2);
                    int a = ResolvePaperDmcEdgeVertex(
                        cell, mcIndex, edgeA,
                        localVertices, edgeVertices,
                        result.DmcShadowVertices);
                    int b = ResolvePaperDmcEdgeVertex(
                        cell, mcIndex, edgeB,
                        localVertices, edgeVertices,
                        result.DmcShadowVertices);
                    int c = ResolvePaperDmcEdgeVertex(
                        cell, mcIndex, edgeC,
                        localVertices, edgeVertices,
                        result.DmcShadowVertices);
                    if (a < 0 || b < 0 || c < 0 ||
                        a == b || b == c || c == a)
                    {
                        // Reference behavior: a missing shared edge suppresses
                        // this triangle only, never the complete cell surface.
                        result
                            .DirectionalMcShadowUnmeasuredEdgeDeferredTriangles++;
                        continue;
                    }
                    result.DmcShadowTriangles.Add(a);
                    result.DmcShadowTriangles.Add(b);
                    result.DmcShadowTriangles.Add(c);
                    result.DmcShadowTriangleCells.Add(cell.CellIndex);
                    result.DmcShadowLines.Add(a);
                    result.DmcShadowLines.Add(b);
                    result.DmcShadowLines.Add(b);
                    result.DmcShadowLines.Add(c);
                    result.DmcShadowLines.Add(c);
                    result.DmcShadowLines.Add(a);
                }
            }
        }
        result.DirectionalMcShadowVertices =
            result.DmcShadowVertices.Count;
        result.DirectionalMcShadowTriangles =
            result.DmcShadowTriangles.Count / 3;
    }

    private void PromoteActiveHalfVoxelPaperDmc(
        float minimumWeight,
        MeshResult result)
    {
        long startTicks = Stopwatch.GetTimestamp();
        if (!PaperReferenceDirectionalMcEnabled ||
            !_canonicalSixDirectionSemantics ||
            _lastPaperDmcCells == null ||
            _lastPaperDmcCells.Count == 0 ||
            _halfVoxelShadowActiveBlocks.Count == 0)
        {
            return;
        }

        int fineCellDimX = Mathf.Max(1, (_dimX - 1) * 2);
        int fineCellDimY = Mathf.Max(1, (_dimY - 1) * 2);
        int fineCellDimZ = Mathf.Max(1, (_dimZ - 1) * 2);
        Dictionary<int, PaperDirectionalMcCell> fineCells =
            new Dictionary<int, PaperDirectionalMcCell>(8192);
        HashSet<int> coarseCellsWithFineSurface = new HashSet<int>();
        HashSet<int> coarseCellsWithIncompleteFineEvidence =
            new HashSet<int>();
        List<int> orderedBlocks =
            new List<int>(_halfVoxelShadowActiveBlocks);
        orderedBlocks.Sort();

        for (int blockPosition = 0;
             blockPosition < orderedBlocks.Count;
             blockPosition++)
        {
            int blockKey = orderedBlocks[blockPosition];
            if (!_halfVoxelShadowBlocks.TryGetValue(
                    blockKey, out HalfVoxelBlock block))
            {
                continue;
            }
            result.DirectionalMcShadowFineRefinementBlocks++;
            int localCellDimension = block.Dimension - 1;
            HashSet<int> candidates = new HashSet<int>();
            for (int direction = 0; direction < DirectionCount; direction++)
            {
                candidates.UnionWith(
                    block.ScaledTruncationCandidateCells[direction]);
            }
            List<int> orderedCandidates = new List<int>(candidates);
            orderedCandidates.Sort();
            int globalFineStartX = Mathf.RoundToInt(
                (block.Origin.x - _origin.x) / (_voxelSize * 0.5f));
            int globalFineStartY = Mathf.RoundToInt(
                (block.Origin.y - _origin.y) / (_voxelSize * 0.5f));
            int globalFineStartZ = Mathf.RoundToInt(
                (block.Origin.z - _origin.z) / (_voxelSize * 0.5f));
            for (int candidatePosition = 0;
                 candidatePosition < orderedCandidates.Count;
                 candidatePosition++)
            {
                int localCellIndex = orderedCandidates[candidatePosition];
                int localZ =
                    localCellIndex /
                    (localCellDimension * localCellDimension);
                int remainder =
                    localCellIndex -
                    localZ * localCellDimension * localCellDimension;
                int localY = remainder / localCellDimension;
                int localX = remainder - localY * localCellDimension;
                int globalFineX = globalFineStartX + localX;
                int globalFineY = globalFineStartY + localY;
                int globalFineZ = globalFineStartZ + localZ;
                if (globalFineX < 0 || globalFineY < 0 ||
                    globalFineZ < 0 ||
                    globalFineX >= fineCellDimX ||
                    globalFineY >= fineCellDimY ||
                    globalFineZ >= fineCellDimZ)
                {
                    continue;
                }

                int coarseX = Mathf.FloorToInt(globalFineX * 0.5f);
                int coarseY = Mathf.FloorToInt(globalFineY * 0.5f);
                int coarseZ = Mathf.FloorToInt(globalFineZ * 0.5f);
                if (BlockKeyFromCell(coarseX, coarseY, coarseZ) != blockKey)
                    continue;
                if (!CanPromoteHalfVoxelCoarseCell(
                        coarseX, coarseY, coarseZ, blockKey))
                {
                    result.DirectionalMcShadowFineBoundaryDeferredCells++;
                    continue;
                }

                int globalFineCellIndex =
                    globalFineX +
                    fineCellDimX *
                    (globalFineY + fineCellDimY * globalFineZ);
                if (fineCells.ContainsKey(globalFineCellIndex))
                    continue;
                result.DirectionalMcShadowFineCellsEvaluated++;
                int coarseCellIndex =
                    CellIndex(coarseX, coarseY, coarseZ);
                if (!TryCreateHalfVoxelPaperDmcCell(
                        block,
                        localX, localY, localZ,
                        globalFineCellIndex,
                        globalFineX, globalFineY, globalFineZ,
                        minimumWeight,
                        result,
                        out PaperDirectionalMcCell cell,
                        out bool incompleteEvidence))
                {
                    if (incompleteEvidence)
                    {
                        coarseCellsWithIncompleteFineEvidence.Add(
                            coarseCellIndex);
                    }
                    continue;
                }
                fineCells.Add(globalFineCellIndex, cell);
                result.DirectionalMcShadowFineCellsAccepted++;
                coarseCellsWithFineSurface.Add(
                    coarseCellIndex);
            }
        }

        coarseCellsWithFineSurface.ExceptWith(
            coarseCellsWithIncompleteFineEvidence);
        if (fineCells.Count == 0 || coarseCellsWithFineSurface.Count == 0)
        {
            result.DirectionalMcShadowMilliseconds +=
                (Stopwatch.GetTimestamp() - startTicks) * 1000d /
                Stopwatch.Frequency;
            return;
        }

        RegularizePaperMcIndices(
            fineCells,
            fineCellDimX, fineCellDimY, fineCellDimZ,
            result);
        // Fine cells belonging to a coarse cell that failed the complete-
        // evidence gate remain diagnostic only; do not overlay them on the
        // retained coarse surface.
        List<int> fineKeys = new List<int>(fineCells.Keys);
        for (int i = 0; i < fineKeys.Count; i++)
        {
            PaperDirectionalMcCell fineCell = fineCells[fineKeys[i]];
            int coarseX = fineCell.X / 2;
            int coarseY = fineCell.Y / 2;
            int coarseZ = fineCell.Z / 2;
            if (!coarseCellsWithFineSurface.Contains(
                    CellIndex(coarseX, coarseY, coarseZ)))
            {
                fineCells.Remove(fineKeys[i]);
            }
        }
        if (!PaperHalfVoxelProductionPromotionEnabled)
        {
            // Keep the refinement evidence and accepted-cell counters alive,
            // but never punch partial holes in the committed coarse mesh.
            result.DirectionalMcShadowFineCoarseCellsPromoted = 0;
            result.DirectionalMcShadowFineVertices = 0;
            result.DirectionalMcShadowFineTriangles = 0;
            result.DirectionalMcShadowMilliseconds +=
                (Stopwatch.GetTimestamp() - startTicks) * 1000d /
                Stopwatch.Frequency;
            return;
        }
        BuildPaperDmcShadowMesh(
            _lastPaperDmcCells,
            _lastPaperDmcTriangleLimit,
            result,
            coarseCellsWithFineSurface);
        int coarseVertexCount = result.DmcShadowVertices.Count;
        int coarseTriangleCount = result.DmcShadowTriangles.Count / 3;
        AppendFinePaperDmcShadowMesh(
            fineCells,
            fineCellDimX, fineCellDimY,
            _lastPaperDmcTriangleLimit,
            result);
        result.DirectionalMcShadowFineCoarseCellsPromoted =
            coarseCellsWithFineSurface.Count;
        result.DirectionalMcShadowFineVertices =
            Mathf.Max(0, result.DmcShadowVertices.Count - coarseVertexCount);
        result.DirectionalMcShadowFineTriangles =
            Mathf.Max(
                0,
                result.DmcShadowTriangles.Count / 3 - coarseTriangleCount);
        result.DirectionalMcShadowVertices =
            result.DmcShadowVertices.Count;
        result.DirectionalMcShadowTriangles =
            result.DmcShadowTriangles.Count / 3;

        ResetDirectionalMcMeshAudit(result);
        AuditDirectionalMcShadowMesh(result);
        result.DirectionalMcShadowMilliseconds +=
            (Stopwatch.GetTimestamp() - startTicks) * 1000d /
            Stopwatch.Frequency;
    }

    private void BuildHermiteQefFeatureShadow(
        float minimumWeight,
        MeshResult result)
    {
        Stopwatch stopwatch = Stopwatch.StartNew();
        result.HermiteQefShadowEnabled =
            PaperReferenceDirectionalMcEnabled &&
            _canonicalSixDirectionSemantics;
        CopyPaperDmcToHermiteQefShadow(result);
        if (!result.HermiteQefShadowEnabled ||
            _lastPaperDmcCells == null ||
            _lastPaperDmcCells.Count == 0 ||
            result.DmcShadowTriangles.Count == 0 ||
            result.DmcShadowTriangleCells.Count !=
                result.DmcShadowTriangles.Count / 3)
        {
            stopwatch.Stop();
            result.HermiteQefMilliseconds =
                stopwatch.Elapsed.TotalMilliseconds;
            return;
        }

        List<HermiteQefFeature> features =
            new List<HermiteQefFeature>(256);
        List<int> orderedCells =
            new List<int>(_lastPaperDmcCells.Keys);
        orderedCells.Sort();
        int[] indices = new int[DirectionCount];
        bool[] validHypotheses = new bool[DirectionCount];
        Vector3[] hypothesisNormals = new Vector3[DirectionCount];
        float[,] values = new float[DirectionCount, 8];
        List<HermiteQefSample> samples =
            new List<HermiteQefSample>(72);
        for (int position = 0; position < orderedCells.Count; position++)
        {
            PaperDirectionalMcCell cell =
                _lastPaperDmcCells[orderedCells[position]];
            result.HermiteQefScannedCells++;
            if (TryCreateHermiteQefFeature(
                    cell, minimumWeight,
                    indices, validHypotheses, hypothesisNormals,
                    values, samples, result,
                    out HermiteQefFeature feature))
            {
                features.Add(feature);
            }
        }

        ApplyHermiteQefFeatures(features, result);
        AuditHermiteQefShadowMesh(
            result.HermiteQefShadowVertices,
            result.HermiteQefShadowTriangles,
            out result.HermiteQefBoundaryEdges,
            out result.HermiteQefNonManifoldEdges,
            out result.HermiteQefDuplicateTriangles);
        result.HermiteQefBoundaryEdgeDelta =
            result.HermiteQefBoundaryEdges -
            result.DirectionalMcShadowBoundaryEdges;
        result.HermiteQefNonManifoldEdgeDelta =
            result.HermiteQefNonManifoldEdges -
            result.DirectionalMcShadowNonManifoldEdges;
        result.HermiteQefDuplicateTriangleDelta =
            result.HermiteQefDuplicateTriangles -
            result.DirectionalMcShadowDuplicateTriangles;
        result.HermiteQefPreRollbackBoundaryEdges =
            result.HermiteQefBoundaryEdges;
        result.HermiteQefPreRollbackNonManifoldEdges =
            result.HermiteQefNonManifoldEdges;
        result.HermiteQefPreRollbackDuplicateTriangles =
            result.HermiteQefDuplicateTriangles;
        result.HermiteQefPreRollbackBoundaryEdgeDelta =
            result.HermiteQefBoundaryEdgeDelta;
        result.HermiteQefPreRollbackNonManifoldEdgeDelta =
            result.HermiteQefNonManifoldEdgeDelta;
        result.HermiteQefPreRollbackDuplicateTriangleDelta =
            result.HermiteQefDuplicateTriangleDelta;

        // A feature placement is never allowed to change the paper topology
        // audit. Roll the whole visual shadow back to byte-for-byte DMC lists if
        // an implementation mistake escapes the per-cell boundary signature.
        if (result.HermiteQefBoundaryEdgeDelta != 0 ||
            result.HermiteQefNonManifoldEdgeDelta != 0 ||
            result.HermiteQefDuplicateTriangleDelta != 0)
        {
            result.HermiteQefTopologyRollback = 1;
            result.HermiteQefAppliedCells = 0;
            result.HermiteQefSourceTrianglesReplaced = 0;
            result.HermiteQefFeatureTrianglesAdded = 0;
            CopyPaperDmcToHermiteQefShadow(result);
            result.HermiteQefBoundaryEdges =
                result.DirectionalMcShadowBoundaryEdges;
            result.HermiteQefNonManifoldEdges =
                result.DirectionalMcShadowNonManifoldEdges;
            result.HermiteQefDuplicateTriangles =
                result.DirectionalMcShadowDuplicateTriangles;
            result.HermiteQefBoundaryEdgeDelta = 0;
            result.HermiteQefNonManifoldEdgeDelta = 0;
            result.HermiteQefDuplicateTriangleDelta = 0;
        }
        result.HermiteQefProductionEligible =
            result.HermiteQefAppliedCells > 0 &&
            result.HermiteQefTopologyRollback == 0 &&
            result.DirectionalMcShadowPendingTopologyChanges == 0 &&
            result.DirectionalMcShadowTopologyStableBatches >=
                PaperQefMinimumStableTopologyBatches &&
            result.HermiteQefBoundaryEdgeDelta <= 0 &&
            result.HermiteQefNonManifoldEdgeDelta <= 0 &&
            result.HermiteQefDuplicateTriangleDelta <= 0;
        stopwatch.Stop();
        result.HermiteQefMilliseconds =
            stopwatch.Elapsed.TotalMilliseconds;
    }

    private static void CopyPaperDmcToHermiteQefShadow(
        MeshResult result)
    {
        result.HermiteQefShadowVertices.Clear();
        result.HermiteQefShadowTriangles.Clear();
        result.HermiteQefShadowLines.Clear();
        result.HermiteQefShadowVertices.AddRange(
            result.DmcShadowVertices);
        result.HermiteQefShadowTriangles.AddRange(
            result.DmcShadowTriangles);
        result.HermiteQefShadowLines.AddRange(
            result.DmcShadowLines);
    }

    private bool TryCreateHermiteQefFeature(
        PaperDirectionalMcCell cell,
        float minimumWeight,
        int[] indices,
        bool[] validHypotheses,
        Vector3[] hypothesisNormals,
        float[,] values,
        List<HermiteQefSample> samples,
        MeshResult result,
        out HermiteQefFeature feature)
    {
        feature = default;
        samples.Clear();
        int hypothesisCount = 0;
        for (int paperDirection = 0;
             paperDirection < DirectionCount;
             paperDirection++)
        {
            validHypotheses[paperDirection] = false;
            indices[paperDirection] = -1;
            int scanDirection =
                PaperDirectionToScanCoverDirection[paperDirection];
            if (!TryReadPaperDirectionCell(
                    scanDirection,
                    cell.X, cell.Y, cell.Z,
                    minimumWeight,
                    values, paperDirection,
                    out int mcIndex,
                    out float unusedWeight,
                    out Vector3 gradient) ||
                mcIndex <= 0 || mcIndex == byte.MaxValue ||
                gradient.sqrMagnitude <= 0.00000001f)
            {
                continue;
            }
            Vector3 normal = gradient.normalized;
            if (Vector3.Dot(
                    normal, DirectionVectors[scanDirection]) <
                CanonicalDirectionThreshold)
            {
                continue;
            }
            indices[paperDirection] = mcIndex;
            validHypotheses[paperDirection] = true;
            hypothesisNormals[paperDirection] = normal;
            hypothesisCount++;
        }
        if (hypothesisCount < 2)
            return false;

        Vector3 cellOrigin = VoxelCenter(
            cell.X, cell.Y, cell.Z);
        for (int paperDirection = 0;
             paperDirection < DirectionCount;
             paperDirection++)
        {
            if (!validHypotheses[paperDirection])
                continue;
            int scanDirection =
                PaperDirectionToScanCoverDirection[paperDirection];
            for (int edge = 0; edge < 12; edge++)
            {
                int cornerA = PaperEdgeEndpointCorners[edge, 0];
                int cornerB = PaperEdgeEndpointCorners[edge, 1];
                float valueA = values[paperDirection, cornerA];
                float valueB = values[paperDirection, cornerB];
                if ((valueA < 0f) == (valueB < 0f))
                    continue;
                int ax = cell.X + PaperCornerX[cornerA];
                int ay = cell.Y + PaperCornerY[cornerA];
                int az = cell.Z + PaperCornerZ[cornerA];
                int bx = cell.X + PaperCornerX[cornerB];
                int by = cell.Y + PaperCornerY[cornerB];
                int bz = cell.Z + PaperCornerZ[cornerB];
                if (!TryReadVoxel(
                        scanDirection, ax, ay, az,
                        out float unusedA, out float weightA) ||
                    !TryReadVoxel(
                        scanDirection, bx, by, bz,
                        out float unusedB, out float weightB) ||
                    weightA < minimumWeight ||
                    weightB < minimumWeight)
                {
                    continue;
                }

                float denominator = valueA - valueB;
                float t = Mathf.Clamp01(
                    Mathf.Abs(denominator) > 0.000000001f
                        ? valueA / denominator
                        : 0.5f);
                Vector3 localA = new Vector3(
                    PaperCornerX[cornerA],
                    PaperCornerY[cornerA],
                    PaperCornerZ[cornerA]);
                Vector3 localB = new Vector3(
                    PaperCornerX[cornerB],
                    PaperCornerY[cornerB],
                    PaperCornerZ[cornerB]);
                Vector3 local = Vector3.LerpUnclamped(
                    localA, localB, t);
                Vector3 normal = PaperTrilinearGradient(
                    values, paperDirection, local);
                if (normal.sqrMagnitude <= 0.00000001f)
                    continue;
                normal.Normalize();
                if (Vector3.Dot(
                        normal,
                        hypothesisNormals[paperDirection]) < 0f)
                {
                    normal = -normal;
                }
                TryReadPaperEvidence(
                    scanDirection, ax, ay, az,
                    out ulong frameA, out ulong viewA);
                TryReadPaperEvidence(
                    scanDirection, bx, by, bz,
                    out ulong frameB, out ulong viewB);
                samples.Add(new HermiteQefSample
                {
                    Point = cellOrigin + local * _voxelSize,
                    Normal = normal,
                    Weight = Mathf.Min(weightA, weightB),
                    FrameMask = frameA & frameB,
                    ViewMask = viewA & viewB
                });
            }
        }
        result.HermiteQefHermiteSamples += samples.Count;
        if (samples.Count < 4)
            return false;

        samples.Sort((left, right) =>
            right.Weight.CompareTo(left.Weight));
        float familyMergeDot = Mathf.Cos(
            Mathf.Max(
                8f,
                HermiteQefFeatureAngleDegrees * 0.55f) *
            Mathf.Deg2Rad);
        List<HermiteQefFamily> families =
            new List<HermiteQefFamily>(8);
        for (int sampleIndex = 0;
             sampleIndex < samples.Count;
             sampleIndex++)
        {
            HermiteQefSample sample = samples[sampleIndex];
            int bestFamily = -1;
            float bestDot = -1f;
            float bestSign = 1f;
            for (int familyIndex = 0;
                 familyIndex < families.Count;
                 familyIndex++)
            {
                HermiteQefFamily family = families[familyIndex];
                Vector3 familyNormal =
                    family.NormalSum.sqrMagnitude > 0.00000001f
                        ? family.NormalSum.normalized
                        : Vector3.zero;
                float normalDot =
                    Vector3.Dot(sample.Normal, familyNormal);
                if (Mathf.Abs(normalDot) <= bestDot)
                    continue;
                bestDot = Mathf.Abs(normalDot);
                bestSign = normalDot >= 0f ? 1f : -1f;
                bestFamily = familyIndex;
            }
            if (bestFamily >= 0 && bestDot >= familyMergeDot)
            {
                HermiteQefFamily family = families[bestFamily];
                family.NormalSum +=
                    sample.Normal * bestSign * sample.Weight;
                family.PointSum += sample.Point * sample.Weight;
                family.Weight += sample.Weight;
                family.Samples++;
                family.Evidence.Add(new HermiteQefEvidence
                {
                    FrameMask = sample.FrameMask,
                    ViewMask = sample.ViewMask
                });
            }
            else
            {
                HermiteQefFamily family =
                    new HermiteQefFamily
                    {
                        NormalSum =
                            sample.Normal * sample.Weight,
                        PointSum =
                            sample.Point * sample.Weight,
                        Weight = sample.Weight,
                        Samples = 1
                    };
                family.Evidence.Add(new HermiteQefEvidence
                {
                    FrameMask = sample.FrameMask,
                    ViewMask = sample.ViewMask
                });
                families.Add(family);
            }
        }

        float totalFamilyWeight = 0f;
        for (int i = 0; i < families.Count; i++)
            totalFamilyWeight += families[i].Weight;
        for (int i = families.Count - 1; i >= 0; i--)
        {
            HermiteQefFamily family = families[i];
            if (family.Samples < 2 ||
                family.Weight <
                totalFamilyWeight *
                HermiteQefMinimumFamilySupportRatio)
            {
                families.RemoveAt(i);
            }
        }
        families.Sort((left, right) =>
            right.Weight.CompareTo(left.Weight));
        float featureDot = Mathf.Cos(
            HermiteQefFeatureAngleDegrees * Mathf.Deg2Rad);
        List<HermiteQefFamily> selected =
            new List<HermiteQefFamily>(3);
        for (int familyIndex = 0;
             familyIndex < families.Count &&
             selected.Count < 3;
             familyIndex++)
        {
            HermiteQefFamily family = families[familyIndex];
            Vector3 familyNormal = family.NormalSum.normalized;
            bool distinct = true;
            for (int previous = 0;
                 previous < selected.Count;
                 previous++)
            {
                if (Mathf.Abs(Vector3.Dot(
                        familyNormal,
                        selected[previous].NormalSum.normalized)) >=
                    featureDot)
                {
                    distinct = false;
                    break;
                }
            }
            if (distinct)
                selected.Add(family);
        }
        if (selected.Count < 2)
            return false;
        result.HermiteQefRawCandidates++;

        int minimumPersistentSamples = int.MaxValue;
        for (int familyIndex = 0;
             familyIndex < selected.Count;
             familyIndex++)
        {
            HermiteQefFamily family = selected[familyIndex];
            bool hasFrameStableEdge = false;
            int persistentSamples = 0;
            for (int evidenceIndex = 0;
                 evidenceIndex < family.Evidence.Count;
                 evidenceIndex++)
            {
                HermiteQefEvidence evidence =
                    family.Evidence[evidenceIndex];
                if (CountBits64(evidence.FrameMask) <
                    HermiteQefMinimumFramesPerFamily)
                {
                    continue;
                }
                hasFrameStableEdge = true;
                if (CountBits64(evidence.ViewMask) >=
                    HermiteQefMinimumViewsPerFamily)
                {
                    persistentSamples++;
                }
            }
            if (!hasFrameStableEdge)
            {
                result.HermiteQefFrameRejected++;
                return false;
            }
            if (persistentSamples <= 0)
            {
                result.HermiteQefViewRejected++;
                return false;
            }
            minimumPersistentSamples =
                Mathf.Min(minimumPersistentSamples, persistentSamples);
        }
        if (minimumPersistentSamples <
            HermiteQefMinimumPersistentSamplesPerFamily)
        {
            result.HermiteQefSampleRejected++;
            return false;
        }

        float maximumFamilyWeight = selected[0].Weight;
        float minimumFamilyWeight = selected[0].Weight;
        for (int i = 1; i < selected.Count; i++)
        {
            maximumFamilyWeight =
                Mathf.Max(maximumFamilyWeight, selected[i].Weight);
            minimumFamilyWeight =
                Mathf.Min(minimumFamilyWeight, selected[i].Weight);
        }
        if (minimumFamilyWeight /
            Mathf.Max(0.00000001f, maximumFamilyWeight) <
            HermiteQefMinimumFamilyWeightRatio)
        {
            result.HermiteQefFamilyBalanceRejected++;
            return false;
        }

        if (!SolveHermiteQef(
                selected,
                out Vector3 baseline,
                out Vector3 proposal,
                out int rank,
                out float retainedRankRatio,
                out float beforeResidual,
                out float afterResidual) ||
            rank < 2 ||
            retainedRankRatio <
            HermiteQefMinimumCertificateRankRatio)
        {
            result.HermiteQefRankRejected++;
            return false;
        }
        float displacementRatio =
            Vector3.Distance(proposal, baseline) /
            Mathf.Max(0.00000001f, _voxelSize);
        if (displacementRatio <
            HermiteQefMinimumDisplacementRatio)
        {
            result.HermiteQefDisplacementRejected++;
            return false;
        }
        Vector3 cellMaximum =
            cellOrigin + Vector3.one * _voxelSize;
        float expanded = _voxelSize * 0.25f;
        if (!Finite(proposal) ||
            proposal.x < cellOrigin.x - expanded ||
            proposal.y < cellOrigin.y - expanded ||
            proposal.z < cellOrigin.z - expanded ||
            proposal.x > cellMaximum.x + expanded ||
            proposal.y > cellMaximum.y + expanded ||
            proposal.z > cellMaximum.z + expanded)
        {
            result.HermiteQefCellMarginRejected++;
            return false;
        }
        float margin = Mathf.Min(
            Mathf.Min(
                proposal.x - cellOrigin.x,
                proposal.y - cellOrigin.y),
            Mathf.Min(
                proposal.z - cellOrigin.z,
                Mathf.Min(
                    cellMaximum.x - proposal.x,
                    Mathf.Min(
                        cellMaximum.y - proposal.y,
                        cellMaximum.z - proposal.z))));
        if (margin /
            Mathf.Max(0.00000001f, _voxelSize) <
            HermiteQefMinimumCellMarginRatio)
        {
            result.HermiteQefCellMarginRejected++;
            return false;
        }
        if (beforeResidual > 0.000000000001f &&
            afterResidual >= beforeResidual * 0.98f)
        {
            result.HermiteQefResidualRejected++;
            return false;
        }

        feature = new HermiteQefFeature
        {
            CellIndex = cell.CellIndex,
            Proposal = proposal
        };
        result.HermiteQefCertified++;
        return true;
    }

    private Vector3 PaperTrilinearGradient(
        float[,] values,
        int direction,
        Vector3 local)
    {
        float x = Mathf.Clamp01(local.x);
        float y = Mathf.Clamp01(local.y);
        float z = Mathf.Clamp01(local.z);
        Vector3 gradient = Vector3.zero;
        for (int corner = 0; corner < 8; corner++)
        {
            bool highX = PaperCornerX[corner] != 0;
            bool highY = PaperCornerY[corner] != 0;
            bool highZ = PaperCornerZ[corner] != 0;
            float wx = highX ? x : 1f - x;
            float wy = highY ? y : 1f - y;
            float wz = highZ ? z : 1f - z;
            float value = values[direction, corner];
            gradient.x +=
                value * (highX ? 1f : -1f) * wy * wz;
            gradient.y +=
                value * (highY ? 1f : -1f) * wx * wz;
            gradient.z +=
                value * (highZ ? 1f : -1f) * wx * wy;
        }
        return gradient /
            Mathf.Max(0.00000001f, _voxelSize);
    }

    private bool TryReadPaperEvidence(
        int direction,
        int x,
        int y,
        int z,
        out ulong frameMask,
        out ulong viewMask)
    {
        frameMask = 0UL;
        viewMask = 0UL;
        if (!TryGetLayer(
                direction, x, y, z,
                out DirectionLayer layer,
                out int localIndex))
        {
            return false;
        }
        frameMask = layer.PaperFrameMask[localIndex];
        viewMask = layer.PaperViewMask[localIndex];
        return frameMask != 0UL || viewMask != 0UL;
    }

    private static int CountBits64(ulong value)
    {
        int count = 0;
        while (value != 0UL)
        {
            value &= value - 1UL;
            count++;
        }
        return count;
    }

    private static bool SolveHermiteQef(
        List<HermiteQefFamily> families,
        out Vector3 baseline,
        out Vector3 proposal,
        out int rank,
        out float retainedRankRatio,
        out float beforeResidual,
        out float afterResidual)
    {
        baseline = Vector3.zero;
        proposal = Vector3.zero;
        rank = 0;
        retainedRankRatio = 0f;
        beforeResidual = 0f;
        afterResidual = 0f;
        if (families == null || families.Count < 2)
            return false;

        float maximumWeight = 0f;
        for (int i = 0; i < families.Count; i++)
            maximumWeight =
                Mathf.Max(maximumWeight, families[i].Weight);
        if (maximumWeight <= 0.00000001f)
            return false;
        float normalizedWeightSum = 0f;
        Vector3[] normals = new Vector3[families.Count];
        Vector3[] points = new Vector3[families.Count];
        float[] weights = new float[families.Count];
        for (int i = 0; i < families.Count; i++)
        {
            HermiteQefFamily family = families[i];
            normals[i] = family.NormalSum.normalized;
            points[i] =
                family.PointSum /
                Mathf.Max(0.00000001f, family.Weight);
            weights[i] = family.Weight / maximumWeight;
            normalizedWeightSum += weights[i];
            baseline += points[i] * weights[i];
        }
        baseline /=
            Mathf.Max(0.00000001f, normalizedWeightSum);

        float[,] matrix = new float[3, 3];
        Vector3 rightHandSide = Vector3.zero;
        for (int i = 0; i < families.Count; i++)
        {
            Vector3 n = normals[i];
            float w = weights[i];
            matrix[0, 0] += w * n.x * n.x;
            matrix[0, 1] += w * n.x * n.y;
            matrix[0, 2] += w * n.x * n.z;
            matrix[1, 1] += w * n.y * n.y;
            matrix[1, 2] += w * n.y * n.z;
            matrix[2, 2] += w * n.z * n.z;
            float distance =
                Vector3.Dot(n, points[i] - baseline);
            rightHandSide += n * (w * distance);
        }
        matrix[1, 0] = matrix[0, 1];
        matrix[2, 0] = matrix[0, 2];
        matrix[2, 1] = matrix[1, 2];
        JacobiEigenDecomposition(
            matrix,
            out float[] eigenvalues,
            out Vector3[] axes);
        for (int left = 0; left < 2; left++)
        {
            for (int right = left + 1; right < 3; right++)
            {
                if (eigenvalues[left] >= eigenvalues[right])
                    continue;
                float value = eigenvalues[left];
                eigenvalues[left] = eigenvalues[right];
                eigenvalues[right] = value;
                Vector3 axis = axes[left];
                axes[left] = axes[right];
                axes[right] = axis;
            }
        }
        float maximumEigenvalue =
            Mathf.Max(0f, eigenvalues[0]);
        if (maximumEigenvalue <= 0.00000001f)
            return false;
        float solveThreshold =
            maximumEigenvalue *
            HermiteQefSolveRankRatio *
            HermiteQefSolveRankRatio;
        Vector3 delta = Vector3.zero;
        for (int i = 0; i < 3; i++)
        {
            float eigenvalue = Mathf.Max(0f, eigenvalues[i]);
            if (eigenvalue < solveThreshold)
                continue;
            rank++;
            delta += axes[i] *
                (Vector3.Dot(axes[i], rightHandSide) /
                 Mathf.Max(0.00000001f, eigenvalue));
        }
        if (rank <= 0)
            return false;
        retainedRankRatio = Mathf.Sqrt(
            Mathf.Max(0f, eigenvalues[rank - 1]) /
            maximumEigenvalue);
        proposal = baseline + delta;
        for (int i = 0; i < families.Count; i++)
        {
            float before =
                Vector3.Dot(normals[i], baseline - points[i]);
            float after =
                Vector3.Dot(normals[i], proposal - points[i]);
            beforeResidual += weights[i] * before * before;
            afterResidual += weights[i] * after * after;
        }
        return Finite(proposal);
    }

    private static void JacobiEigenDecomposition(
        float[,] source,
        out float[] eigenvalues,
        out Vector3[] axes)
    {
        float[,] a = (float[,])source.Clone();
        float[,] v =
        {
            { 1f, 0f, 0f },
            { 0f, 1f, 0f },
            { 0f, 0f, 1f }
        };
        for (int iteration = 0; iteration < 16; iteration++)
        {
            int p = 0;
            int q = 1;
            float maximum = Mathf.Abs(a[0, 1]);
            if (Mathf.Abs(a[0, 2]) > maximum)
            {
                p = 0;
                q = 2;
                maximum = Mathf.Abs(a[0, 2]);
            }
            if (Mathf.Abs(a[1, 2]) > maximum)
            {
                p = 1;
                q = 2;
                maximum = Mathf.Abs(a[1, 2]);
            }
            if (maximum <= 0.00000001f)
                break;
            float angle =
                0.5f * Mathf.Atan2(
                    2f * a[p, q],
                    a[q, q] - a[p, p]);
            float cosine = Mathf.Cos(angle);
            float sine = Mathf.Sin(angle);
            for (int row = 0; row < 3; row++)
            {
                if (row == p || row == q)
                    continue;
                float arp = a[row, p];
                float arq = a[row, q];
                a[row, p] =
                    a[p, row] =
                        cosine * arp - sine * arq;
                a[row, q] =
                    a[q, row] =
                        sine * arp + cosine * arq;
            }
            float app = a[p, p];
            float aqq = a[q, q];
            float apq = a[p, q];
            a[p, p] =
                cosine * cosine * app -
                2f * sine * cosine * apq +
                sine * sine * aqq;
            a[q, q] =
                sine * sine * app +
                2f * sine * cosine * apq +
                cosine * cosine * aqq;
            a[p, q] = a[q, p] = 0f;
            for (int row = 0; row < 3; row++)
            {
                float vrp = v[row, p];
                float vrq = v[row, q];
                v[row, p] =
                    cosine * vrp - sine * vrq;
                v[row, q] =
                    sine * vrp + cosine * vrq;
            }
        }
        eigenvalues = new[]
        {
            a[0, 0], a[1, 1], a[2, 2]
        };
        axes = new[]
        {
            new Vector3(v[0, 0], v[1, 0], v[2, 0]).normalized,
            new Vector3(v[0, 1], v[1, 1], v[2, 1]).normalized,
            new Vector3(v[0, 2], v[1, 2], v[2, 2]).normalized
        };
    }

    private void ApplyHermiteQefFeatures(
        List<HermiteQefFeature> features,
        MeshResult result)
    {
        if (features == null || features.Count == 0)
            return;
        Dictionary<int, List<int>> cellTriangles =
            new Dictionary<int, List<int>>(features.Count);
        int sourceTriangleCount =
            result.DmcShadowTriangles.Count / 3;
        for (int triangleIndex = 0;
             triangleIndex < sourceTriangleCount;
             triangleIndex++)
        {
            int cellIndex =
                result.DmcShadowTriangleCells[triangleIndex];
            if (cellIndex < 0)
                continue;
            if (!cellTriangles.TryGetValue(
                    cellIndex, out List<int> triangles))
            {
                triangles = new List<int>(8);
                cellTriangles.Add(cellIndex, triangles);
            }
            triangles.Add(triangleIndex);
        }

        Dictionary<int, HermiteQefReplacement> replacements =
            new Dictionary<int, HermiteQefReplacement>();
        HashSet<int> replacedTriangles = new HashSet<int>();
        for (int featureIndex = 0;
             featureIndex < features.Count;
             featureIndex++)
        {
            HermiteQefFeature feature = features[featureIndex];
            if (!cellTriangles.TryGetValue(
                    feature.CellIndex,
                    out List<int> localTriangles) ||
                localTriangles.Count == 0)
            {
                result.HermiteQefMissingPatchRejected++;
                continue;
            }

            Dictionary<ulong, List<int>> edgeFaces =
                new Dictionary<ulong, List<int>>(32);
            Dictionary<int, HashSet<int>> triangleNeighbors =
                new Dictionary<int, HashSet<int>>();
            for (int local = 0;
                 local < localTriangles.Count;
                 local++)
            {
                int triangleIndex = localTriangles[local];
                triangleNeighbors[triangleIndex] =
                    new HashSet<int>();
                int offset = triangleIndex * 3;
                int a = result.DmcShadowTriangles[offset];
                int b = result.DmcShadowTriangles[offset + 1];
                int c = result.DmcShadowTriangles[offset + 2];
                AddHermiteQefEdgeFace(
                    edgeFaces, EdgeKey(a, b), triangleIndex);
                AddHermiteQefEdgeFace(
                    edgeFaces, EdgeKey(b, c), triangleIndex);
                AddHermiteQefEdgeFace(
                    edgeFaces, EdgeKey(c, a), triangleIndex);
            }
            foreach (List<int> users in edgeFaces.Values)
            {
                if (users.Count != 2)
                    continue;
                triangleNeighbors[users[0]].Add(users[1]);
                triangleNeighbors[users[1]].Add(users[0]);
            }
            HashSet<int> visited = new HashSet<int>();
            Stack<int> stack = new Stack<int>();
            stack.Push(localTriangles[0]);
            while (stack.Count > 0)
            {
                int current = stack.Pop();
                if (!visited.Add(current))
                    continue;
                foreach (int neighbor in triangleNeighbors[current])
                    stack.Push(neighbor);
            }
            if (visited.Count != localTriangles.Count)
            {
                result.HermiteQefMultiPatchRejected++;
                continue;
            }

            List<ulong> boundaryEdges = new List<ulong>();
            Dictionary<int, HashSet<int>> boundaryNeighbors =
                new Dictionary<int, HashSet<int>>();
            foreach (KeyValuePair<ulong, List<int>> pair in edgeFaces)
            {
                if (pair.Value.Count != 1)
                    continue;
                boundaryEdges.Add(pair.Key);
                DecodeEdgeKey(
                    pair.Key, out int left, out int right);
                AddBoundaryNeighbor(
                    boundaryNeighbors, left, right);
                AddBoundaryNeighbor(
                    boundaryNeighbors, right, left);
            }
            bool closedBoundary = boundaryEdges.Count >= 3;
            foreach (HashSet<int> neighbors in
                     boundaryNeighbors.Values)
            {
                if (neighbors.Count != 2)
                {
                    closedBoundary = false;
                    break;
                }
            }
            if (closedBoundary)
            {
                HashSet<int> boundaryVisited =
                    new HashSet<int>();
                Stack<int> boundaryStack = new Stack<int>();
                foreach (int vertex in boundaryNeighbors.Keys)
                {
                    boundaryStack.Push(vertex);
                    break;
                }
                while (boundaryStack.Count > 0)
                {
                    int current = boundaryStack.Pop();
                    if (!boundaryVisited.Add(current))
                        continue;
                    foreach (int neighbor in
                             boundaryNeighbors[current])
                    {
                        boundaryStack.Push(neighbor);
                    }
                }
                closedBoundary =
                    boundaryVisited.Count ==
                    boundaryNeighbors.Count;
            }
            if (!closedBoundary)
            {
                result.HermiteQefOpenBoundaryRejected++;
                continue;
            }

            int candidateVertex =
                result.HermiteQefShadowVertices.Count;
            HermiteQefReplacement replacement =
                new HermiteQefReplacement();
            bool valid = true;
            for (int edgeIndex = 0;
                 edgeIndex < boundaryEdges.Count;
                 edgeIndex++)
            {
                ulong edgeKey = boundaryEdges[edgeIndex];
                DecodeEdgeKey(
                    edgeKey, out int left, out int right);
                int referenceTriangle =
                    edgeFaces[edgeKey][0];
                int referenceOffset = referenceTriangle * 3;
                Vector3 referenceA =
                    result.DmcShadowVertices[
                        result.DmcShadowTriangles[
                            referenceOffset]];
                Vector3 referenceB =
                    result.DmcShadowVertices[
                        result.DmcShadowTriangles[
                            referenceOffset + 1]];
                Vector3 referenceC =
                    result.DmcShadowVertices[
                        result.DmcShadowTriangles[
                            referenceOffset + 2]];
                Vector3 referenceCross =
                    Vector3.Cross(
                        referenceB - referenceA,
                        referenceC - referenceA);
                Vector3 fanCross =
                    Vector3.Cross(
                        result.DmcShadowVertices[right] -
                        result.DmcShadowVertices[left],
                        feature.Proposal -
                        result.DmcShadowVertices[left]);
                if (referenceCross.sqrMagnitude <=
                        _voxelSize * _voxelSize *
                        _voxelSize * _voxelSize *
                        0.000001f ||
                    fanCross.sqrMagnitude <=
                        _voxelSize * _voxelSize *
                        _voxelSize * _voxelSize *
                        0.000001f)
                {
                    valid = false;
                    break;
                }
                float normalDot = Vector3.Dot(
                    referenceCross.normalized,
                    fanCross.normalized);
                if (normalDot < 0f)
                {
                    int swap = left;
                    left = right;
                    right = swap;
                    normalDot = -normalDot;
                }
                if (normalDot < 0.05f)
                {
                    valid = false;
                    break;
                }
                replacement.Triangles.Add(left);
                replacement.Triangles.Add(right);
                replacement.Triangles.Add(candidateVertex);
            }
            if (!valid)
            {
                result.HermiteQefOrientationRejected++;
                continue;
            }
            replacement.SourceTriangles.AddRange(localTriangles);
            result.HermiteQefShadowVertices.Add(
                feature.Proposal);
            replacements[feature.CellIndex] = replacement;
            for (int i = 0;
                 i < localTriangles.Count;
                 i++)
            {
                replacedTriangles.Add(localTriangles[i]);
            }
            result.HermiteQefProvisionalAppliedCells++;
            result.HermiteQefSourceTrianglesReplaced +=
                localTriangles.Count;
            result.HermiteQefFeatureTrianglesAdded +=
                replacement.Triangles.Count / 3;
        }

        if (replacements.Count == 0)
            return;
        result.HermiteQefShadowTriangles.Clear();
        for (int triangleIndex = 0;
             triangleIndex < sourceTriangleCount;
             triangleIndex++)
        {
            if (replacedTriangles.Contains(triangleIndex))
                continue;
            int offset = triangleIndex * 3;
            result.HermiteQefShadowTriangles.Add(
                result.DmcShadowTriangles[offset]);
            result.HermiteQefShadowTriangles.Add(
                result.DmcShadowTriangles[offset + 1]);
            result.HermiteQefShadowTriangles.Add(
                result.DmcShadowTriangles[offset + 2]);
        }
        List<int> replacementCells =
            new List<int>(replacements.Keys);
        replacementCells.Sort();
        for (int i = 0; i < replacementCells.Count; i++)
        {
            result.HermiteQefShadowTriangles.AddRange(
                replacements[replacementCells[i]].Triangles);
        }
        result.HermiteQefShadowLines.Clear();
        for (int i = 0;
             i + 2 < result.HermiteQefShadowTriangles.Count;
             i += 3)
        {
            int a = result.HermiteQefShadowTriangles[i];
            int b = result.HermiteQefShadowTriangles[i + 1];
            int c = result.HermiteQefShadowTriangles[i + 2];
            result.HermiteQefShadowLines.Add(a);
            result.HermiteQefShadowLines.Add(b);
            result.HermiteQefShadowLines.Add(b);
            result.HermiteQefShadowLines.Add(c);
            result.HermiteQefShadowLines.Add(c);
            result.HermiteQefShadowLines.Add(a);
        }
        result.HermiteQefAppliedCells = replacements.Count;
    }

    private static void AddHermiteQefEdgeFace(
        Dictionary<ulong, List<int>> edgeFaces,
        ulong edge,
        int triangle)
    {
        if (!edgeFaces.TryGetValue(
                edge, out List<int> users))
        {
            users = new List<int>(2);
            edgeFaces.Add(edge, users);
        }
        users.Add(triangle);
    }

    private static void AddBoundaryNeighbor(
        Dictionary<int, HashSet<int>> neighbors,
        int vertex,
        int neighbor)
    {
        if (!neighbors.TryGetValue(
                vertex, out HashSet<int> values))
        {
            values = new HashSet<int>();
            neighbors.Add(vertex, values);
        }
        values.Add(neighbor);
    }

    private static void DecodeEdgeKey(
        ulong key,
        out int minimum,
        out int maximum)
    {
        minimum = (int)(key >> 32);
        maximum = (int)(key & uint.MaxValue);
    }

    private static void AuditHermiteQefShadowMesh(
        List<Vector3> vertices,
        List<int> triangles,
        out int boundaryEdges,
        out int nonManifoldEdges,
        out int duplicateTriangles)
    {
        boundaryEdges = 0;
        nonManifoldEdges = 0;
        duplicateTriangles = 0;
        Dictionary<ulong, int> edgeUse =
            new Dictionary<ulong, int>(triangles.Count);
        HashSet<HermiteQefTriangleKey> unique =
            new HashSet<HermiteQefTriangleKey>();
        for (int i = 0; i + 2 < triangles.Count; i += 3)
        {
            int a = triangles[i];
            int b = triangles[i + 1];
            int c = triangles[i + 2];
            if (a < 0 || b < 0 || c < 0 ||
                a >= vertices.Count ||
                b >= vertices.Count ||
                c >= vertices.Count ||
                a == b || b == c || c == a)
            {
                continue;
            }
            if (!unique.Add(
                    new HermiteQefTriangleKey(a, b, c)))
            {
                duplicateTriangles++;
            }
            IncrementEdgeUse(edgeUse, EdgeKey(a, b));
            IncrementEdgeUse(edgeUse, EdgeKey(b, c));
            IncrementEdgeUse(edgeUse, EdgeKey(c, a));
        }
        foreach (int useCount in edgeUse.Values)
        {
            if (useCount == 1)
                boundaryEdges++;
            else if (useCount > 2)
                nonManifoldEdges++;
        }
    }

    private static void IncrementEdgeUse(
        Dictionary<ulong, int> edgeUse,
        ulong edge)
    {
        edgeUse.TryGetValue(edge, out int count);
        edgeUse[edge] = count + 1;
    }

    private bool CanPromoteHalfVoxelCoarseCell(
        int coarseX,
        int coarseY,
        int coarseZ,
        int blockKey)
    {
        DecodeBlockKey(
            blockKey, out int blockX, out int blockY, out int blockZ);
        int localX = coarseX - blockX * _blockSize;
        int localY = coarseY - blockY * _blockSize;
        int localZ = coarseZ - blockZ * _blockSize;
        if (localX == 0 && coarseX > 0 &&
            !IsHalfVoxelNeighborBlockActive(blockX - 1, blockY, blockZ))
            return false;
        if (localY == 0 && coarseY > 0 &&
            !IsHalfVoxelNeighborBlockActive(blockX, blockY - 1, blockZ))
            return false;
        if (localZ == 0 && coarseZ > 0 &&
            !IsHalfVoxelNeighborBlockActive(blockX, blockY, blockZ - 1))
            return false;
        if (localX == _blockSize - 1 && coarseX < _dimX - 2 &&
            !IsHalfVoxelNeighborBlockActive(blockX + 1, blockY, blockZ))
            return false;
        if (localY == _blockSize - 1 && coarseY < _dimY - 2 &&
            !IsHalfVoxelNeighborBlockActive(blockX, blockY + 1, blockZ))
            return false;
        if (localZ == _blockSize - 1 && coarseZ < _dimZ - 2 &&
            !IsHalfVoxelNeighborBlockActive(blockX, blockY, blockZ + 1))
            return false;
        return true;
    }

    private bool IsHalfVoxelNeighborBlockActive(
        int blockX, int blockY, int blockZ)
    {
        if (blockX < 0 || blockY < 0 || blockZ < 0 ||
            blockX >= _blockDimX ||
            blockY >= _blockDimY ||
            blockZ >= _blockDimZ)
        {
            return false;
        }
        int blockKey =
            blockX + _blockDimX * (blockY + _blockDimY * blockZ);
        return _halfVoxelShadowActiveBlocks.Contains(blockKey);
    }

    private bool TryCreateHalfVoxelPaperDmcCell(
        HalfVoxelBlock block,
        int localX,
        int localY,
        int localZ,
        int globalCellIndex,
        int globalX,
        int globalY,
        int globalZ,
        float minimumWeight,
        MeshResult result,
        out PaperDirectionalMcCell cell,
        out bool incompleteEvidence)
    {
        cell = null;
        incompleteEvidence = false;
        int[] mcIndices = new int[DirectionCount];
        float[] sdfWeights = new float[DirectionCount];
        Vector3[] sdfGradients = new Vector3[DirectionCount];
        float[,] sdfValues = new float[DirectionCount, 8];
        int validDirections = 0;
        for (int paperDirection = 0;
             paperDirection < DirectionCount;
             paperDirection++)
        {
            mcIndices[paperDirection] = -1;
            int scanCoverDirection =
                PaperDirectionToScanCoverDirection[paperDirection];
            if (!TryReadHalfVoxelPaperDirectionCell(
                    block,
                    scanCoverDirection,
                    localX, localY, localZ,
                    minimumWeight,
                    sdfValues,
                    paperDirection,
                    out int mcIndex,
                    out float sdfWeight,
                    out Vector3 sdfGradient,
                    out int coarsePriorCorners))
            {
                continue;
            }
            result.DirectionalMcShadowFineCoarsePriorCorners +=
                coarsePriorCorners;
            validDirections++;
            sdfWeights[paperDirection] = sdfWeight;
            sdfGradients[paperDirection] = sdfGradient;
            int filteredIndex = FilterPaperMcIndexDirection(
                mcIndex, paperDirection, sdfValues);
            if (filteredIndex >= 0 && filteredIndex < byte.MaxValue)
            {
                Vector3 gradient =
                    sdfGradient.sqrMagnitude > 0.00000001f
                        ? sdfGradient.normalized
                        : Vector3.zero;
                float compliance = Vector3.Dot(
                    gradient,
                    DirectionVectors[scanCoverDirection]);
                sdfWeights[paperDirection] *= compliance;
                if (compliance < CanonicalDirectionThreshold)
                {
                    filteredIndex = -1;
                    sdfWeights[paperDirection] = 0f;
                }
            }
            mcIndices[paperDirection] = filteredIndex;
        }
        if (validDirections == 0)
        {
            result.DirectionalMcShadowFineIncompleteCells++;
            incompleteEvidence = true;
            return false;
        }

        // Exact Algorithm 1 ordering and sequential update semantics from the
        // reference branch above; only the TSDF sampling lattice is finer.
        for (int direction = 0; direction < DirectionCount; direction++)
        {
            int mcIndex = mcIndices[direction];
            if (mcIndex <= 0 || mcIndex == byte.MaxValue)
                continue;
            float directionWeight = sdfWeights[direction];
            if (directionWeight <= 0.00000001f)
            {
                mcIndices[direction] = -1;
                continue;
            }
            float supportWeight = 1f;
            for (int other = 0; other < DirectionCount; other++)
            {
                if (mcIndices[other] < 0)
                    continue;
                if (mcIndices[other] == 0)
                {
                    supportWeight -=
                        sdfWeights[other] / directionWeight;
                    mcIndices[direction] = 0;
                    break;
                }
                if (other != direction &&
                    PaperMcIndexCompatible(mcIndex, mcIndices[other]))
                {
                    supportWeight +=
                        sdfWeights[other] / directionWeight;
                }
            }
            if (supportWeight < 0f)
                mcIndices[direction] = 0;
        }

        int combined0 = 0;
        int combined1 = 0;
        for (int direction = 0; direction < DirectionCount; direction++)
        {
            int mcIndex = mcIndices[direction];
            if (mcIndex <= 0 || mcIndex == byte.MaxValue)
                continue;
            for (int componentIndex = 0;
                 componentIndex <
                 ScanCoverPaperDmcTables.ComponentsPerIndex;
                 componentIndex++)
            {
                int component = ScanCoverPaperDmcTables.Component(
                    mcIndex, componentIndex);
                if (component < 0)
                    break;
                bool assigned = false;
                if (combined0 == 0)
                {
                    combined0 = component;
                    assigned = true;
                }
                else if (PaperMcIndexCompatible(mcIndex, combined0))
                {
                    combined0 &= component;
                    assigned = true;
                }
                else if (combined1 == 0)
                {
                    combined1 = component;
                    assigned = true;
                }
                else if (PaperMcIndexCompatible(mcIndex, combined1))
                {
                    combined1 &= component;
                    assigned = true;
                }
                if (!assigned)
                {
                    result
                        .DirectionalMcShadowOverflowDeferredComponents++;
                }
            }
        }
        if (combined0 <= 0 || combined0 == byte.MaxValue)
        {
            if (combined1 <= 0 || combined1 == byte.MaxValue)
            {
                result.DirectionalMcShadowFineVotingEmptyCells++;
                return false;
            }
            combined0 = combined1;
            combined1 = 0;
        }
        if (combined1 == byte.MaxValue)
            combined1 = 0;

        cell = new PaperDirectionalMcCell
        {
            CellIndex = globalCellIndex,
            X = globalX,
            Y = globalY,
            Z = globalZ,
            Index0 = (byte)combined0,
            Index1 = (byte)combined1,
            RegularizedIndex0 = (byte)combined0,
            RegularizedIndex1 = (byte)combined1,
            SurfaceCount = combined1 > 0 ? 2 : 1
        };
        EvaluatePaperEdgeOffsets(
            mcIndices, sdfValues, sdfWeights, cell, result);
        return true;
    }

    private bool TryReadHalfVoxelPaperDirectionCell(
        HalfVoxelBlock block,
        int scanCoverDirection,
        int x,
        int y,
        int z,
        float minimumWeight,
        float[,] sdfValues,
        int paperDirection,
        out int mcIndex,
        out float sdfWeight,
        out Vector3 sdfGradient,
        out int coarsePriorCorners)
    {
        mcIndex = 0;
        sdfWeight = 0f;
        sdfGradient = Vector3.zero;
        coarsePriorCorners = 0;
        HalfVoxelLayer layer =
            block.ScaledTruncationLayers[scanCoverDirection];
        int dimension = block.Dimension;
        for (int corner = 0; corner < 8; corner++)
        {
            int vx = x + PaperCornerX[corner];
            int vy = y + PaperCornerY[corner];
            int vz = z + PaperCornerZ[corner];
            if (vx < 0 || vy < 0 || vz < 0 ||
                vx >= dimension || vy >= dimension || vz >= dimension)
            {
                mcIndex = -1;
                return false;
            }
            int voxel = vx + dimension * (vy + dimension * vz);
            float weight = layer != null ? layer.Weight[voxel] : 0f;
            float value;
            if (layer != null && weight >= minimumWeight)
            {
                value = layer.Tsdf[voxel];
            }
            else
            {
                Vector3 world = block.Origin +
                    new Vector3(vx, vy, vz) * (_voxelSize * 0.5f);
                Vector3 coarseCoordinate =
                    (world - _origin) / _voxelSize;
                if (!TryInterpolateCoarseDirectionalTsdf(
                        scanCoverDirection,
                        coarseCoordinate,
                        minimumWeight,
                        out float coarseValue,
                        out weight))
                {
                    mcIndex = -1;
                    sdfWeight = 0f;
                    return false;
                }
                float fineTruncation = Mathf.Max(
                    _voxelSize * 0.5f,
                    _truncation * 0.5f);
                value = Mathf.Clamp(
                    coarseValue * _truncation / fineTruncation,
                    -1f, 1f);
                coarsePriorCorners++;
            }
            sdfValues[paperDirection, corner] = value;
            sdfWeight += weight * 0.125f;
            if (value < 0f)
                mcIndex |= 1 << corner;
        }
        float gx = 0f;
        float gy = 0f;
        float gz = 0f;
        for (int corner = 0; corner < 8; corner++)
        {
            float value = sdfValues[paperDirection, corner];
            gx += value *
                  (PaperCornerX[corner] * 2f - 1f) * 0.25f;
            gy += value *
                  (PaperCornerY[corner] * 2f - 1f) * 0.25f;
            gz += value *
                  (PaperCornerZ[corner] * 2f - 1f) * 0.25f;
        }
        sdfGradient = new Vector3(gx, gy, gz);
        return true;
    }

    private bool TryInterpolateCoarseDirectionalTsdf(
        int direction,
        Vector3 coordinate,
        float minimumWeight,
        out float value,
        out float weight)
    {
        value = 0f;
        weight = 0f;
        int x0 = Mathf.FloorToInt(coordinate.x);
        int y0 = Mathf.FloorToInt(coordinate.y);
        int z0 = Mathf.FloorToInt(coordinate.z);
        float fx = coordinate.x - x0;
        float fy = coordinate.y - y0;
        float fz = coordinate.z - z0;
        float availableBasis = 0f;
        float weightedBasis = 0f;
        float valueNumerator = 0f;
        float weightNumerator = 0f;
        for (int dz = 0; dz <= 1; dz++)
        for (int dy = 0; dy <= 1; dy++)
        for (int dx = 0; dx <= 1; dx++)
        {
            float basis =
                (dx == 0 ? 1f - fx : fx) *
                (dy == 0 ? 1f - fy : fy) *
                (dz == 0 ? 1f - fz : fz);
            if (basis <= 0.000001f)
                continue;
            int x = x0 + dx;
            int y = y0 + dy;
            int z = z0 + dz;
            if (x < 0 || y < 0 || z < 0 ||
                x >= _dimX || y >= _dimY || z >= _dimZ ||
                !TryReadVoxel(
                    direction, x, y, z,
                    out float coarseValue, out float coarseWeight) ||
                coarseWeight < minimumWeight)
            {
                continue;
            }
            availableBasis += basis;
            float confidence =
                Mathf.Min(coarseWeight, minimumWeight * 4f);
            float contribution = basis * confidence;
            valueNumerator += coarseValue * contribution;
            weightNumerator += coarseWeight * basis;
            weightedBasis += contribution;
        }
        // Never bridge a largely unobserved coarse neighborhood. At a
        // half-grid location this permits one missing eighth/quarter sample,
        // but rejects a one-sided extrapolation across a surface boundary.
        if (availableBasis < 0.75f ||
            weightedBasis <= 0.000001f)
        {
            return false;
        }
        value = valueNumerator / weightedBasis;
        weight = weightNumerator / availableBasis;
        return weight >= minimumWeight;
    }

    private void RegularizePaperMcIndices(
        Dictionary<int, PaperDirectionalMcCell> cells,
        int cellDimX,
        int cellDimY,
        int cellDimZ,
        MeshResult result)
    {
        Dictionary<int, byte> sourceIndices =
            new Dictionary<int, byte>(cells.Count);
        foreach (KeyValuePair<int, PaperDirectionalMcCell> pair in cells)
            sourceIndices[pair.Key] = pair.Value.Index0;
        foreach (PaperDirectionalMcCell cell in cells.Values)
        {
            int sourceIndex = cell.Index0;
            if (sourceIndex <= 0 || sourceIndex == byte.MaxValue)
                continue;
            int regularized = sourceIndex;
            for (int corner = 0; corner < 8; corner++)
            {
                int physicalX = cell.X + PaperCornerX[corner];
                int physicalY = cell.Y + PaperCornerY[corner];
                int physicalZ = cell.Z + PaperCornerZ[corner];
                int insideVotes =
                    (sourceIndex & (1 << corner)) != 0 ? 1 : 0;
                int votes = 1;
                for (int neighborCorner = 0;
                     neighborCorner < 8;
                     neighborCorner++)
                {
                    int neighborX =
                        physicalX - PaperCornerX[neighborCorner];
                    int neighborY =
                        physicalY - PaperCornerY[neighborCorner];
                    int neighborZ =
                        physicalZ - PaperCornerZ[neighborCorner];
                    if (neighborX == cell.X &&
                        neighborY == cell.Y &&
                        neighborZ == cell.Z)
                    {
                        continue;
                    }
                    if (neighborX < 0 || neighborY < 0 ||
                        neighborZ < 0 ||
                        neighborX >= cellDimX ||
                        neighborY >= cellDimY ||
                        neighborZ >= cellDimZ)
                    {
                        continue;
                    }
                    int neighborIndexKey =
                        neighborX +
                        cellDimX *
                        (neighborY + cellDimY * neighborZ);
                    if (!sourceIndices.TryGetValue(
                            neighborIndexKey,
                            out byte neighborIndex) ||
                        neighborIndex <= 0)
                    {
                        continue;
                    }
                    if ((neighborIndex & (1 << neighborCorner)) != 0)
                        insideVotes++;
                    votes++;
                }
                bool inside = insideVotes > votes / 2;
                bool wasInside =
                    (sourceIndex & (1 << corner)) != 0;
                if (inside == wasInside)
                    continue;
                regularized = inside
                    ? regularized | (1 << corner)
                    : regularized & ~(1 << corner);
                result.DirectionalMcShadowRegularizedCorners++;
            }
            cell.RegularizedIndex0 = (byte)regularized;
            if (cell.RegularizedIndex0 != cell.Index0)
                result.DirectionalMcShadowRegularizedCells++;
        }
    }

    private void AppendFinePaperDmcShadowMesh(
        Dictionary<int, PaperDirectionalMcCell> cells,
        int fineCellDimX,
        int fineCellDimY,
        int maximumTriangles,
        MeshResult result)
    {
        Dictionary<DmcEdgeVertexKey, int> edgeVertices =
            new Dictionary<DmcEdgeVertexKey, int>(16384);
        int[] localVertices = new int[12];
        int triangleLimit = Mathf.Max(1, maximumTriangles);
        List<int> ordered = new List<int>(cells.Keys);
        ordered.Sort();
        for (int position = 0;
             position < ordered.Count &&
             result.DmcShadowTriangles.Count / 3 < triangleLimit;
             position++)
        {
            PaperDirectionalMcCell cell = cells[ordered[position]];
            for (int surface = 0;
                 surface < cell.SurfaceCount &&
                 result.DmcShadowTriangles.Count / 3 < triangleLimit;
                 surface++)
            {
                int mcIndex = surface == 0
                    ? cell.RegularizedIndex0
                    : cell.RegularizedIndex1;
                if (mcIndex <= 0 || mcIndex == byte.MaxValue)
                    continue;
                for (int local = 0;
                     local < localVertices.Length;
                     local++)
                {
                    localVertices[local] = -1;
                }
                for (int entry = 0;
                     entry <
                     ScanCoverPaperDmcTables.TriangleEntriesPerIndex;
                     entry += 3)
                {
                    int edgeA =
                        ScanCoverPaperDmcTables.TriangleEdge(
                            mcIndex, entry);
                    if (edgeA < 0 ||
                        result.DmcShadowTriangles.Count / 3 >=
                        triangleLimit)
                    {
                        break;
                    }
                    int edgeB =
                        ScanCoverPaperDmcTables.TriangleEdge(
                            mcIndex, entry + 1);
                    int edgeC =
                        ScanCoverPaperDmcTables.TriangleEdge(
                            mcIndex, entry + 2);
                    int a = ResolvePaperFineDmcEdgeVertex(
                        cell, mcIndex, edgeA,
                        fineCellDimX, fineCellDimY,
                        localVertices, edgeVertices,
                        result.DmcShadowVertices);
                    int b = ResolvePaperFineDmcEdgeVertex(
                        cell, mcIndex, edgeB,
                        fineCellDimX, fineCellDimY,
                        localVertices, edgeVertices,
                        result.DmcShadowVertices);
                    int c = ResolvePaperFineDmcEdgeVertex(
                        cell, mcIndex, edgeC,
                        fineCellDimX, fineCellDimY,
                        localVertices, edgeVertices,
                        result.DmcShadowVertices);
                    if (a < 0 || b < 0 || c < 0 ||
                        a == b || b == c || c == a)
                    {
                        result
                            .DirectionalMcShadowUnmeasuredEdgeDeferredTriangles++;
                        continue;
                    }
                    result.DmcShadowTriangles.Add(a);
                    result.DmcShadowTriangles.Add(b);
                    result.DmcShadowTriangles.Add(c);
                    // Fine-cell indices live in a different grid and are not
                    // eligible for the coarse QEF interior replacement.
                    result.DmcShadowTriangleCells.Add(-1);
                    result.DmcShadowLines.Add(a);
                    result.DmcShadowLines.Add(b);
                    result.DmcShadowLines.Add(b);
                    result.DmcShadowLines.Add(c);
                    result.DmcShadowLines.Add(c);
                    result.DmcShadowLines.Add(a);
                }
            }
        }
    }

    private int ResolvePaperFineDmcEdgeVertex(
        PaperDirectionalMcCell cell,
        int mcIndex,
        int edge,
        int fineCellDimX,
        int fineCellDimY,
        int[] localVertices,
        Dictionary<DmcEdgeVertexKey, int> edgeVertices,
        List<Vector3> vertices)
    {
        if (edge < 0 || edge >= 12)
            return -1;
        if (localVertices[edge] >= 0)
            return localVertices[edge];
        int endpointA = PaperEdgeEndpointCorners[edge, 0];
        int endpointB = PaperEdgeEndpointCorners[edge, 1];
        int slot = (mcIndex & (1 << endpointA)) != 0 ? 0 : 1;
        if ((cell.EdgeSlotMask[edge] & (1 << slot)) == 0)
            return -1;
        int ax = cell.X + PaperCornerX[endpointA];
        int ay = cell.Y + PaperCornerY[endpointA];
        int az = cell.Z + PaperCornerZ[endpointA];
        int bx = cell.X + PaperCornerX[endpointB];
        int by = cell.Y + PaperCornerY[endpointB];
        int bz = cell.Z + PaperCornerZ[endpointB];
        int fineVoxelDimX = fineCellDimX + 1;
        int fineVoxelDimY = fineCellDimY + 1;
        int voxelA =
            ax + fineVoxelDimX * (ay + fineVoxelDimY * az);
        int voxelB =
            bx + fineVoxelDimX * (by + fineVoxelDimY * bz);
        int minimumVoxel = Mathf.Min(voxelA, voxelB);
        byte axis = (byte)(ax != bx ? 0 : ay != by ? 1 : 2);
        DmcEdgeVertexKey key = new DmcEdgeVertexKey(
            minimumVoxel, axis, (byte)slot);
        if (!edgeVertices.TryGetValue(key, out int vertexIndex))
        {
            float fineVoxelSize = _voxelSize * 0.5f;
            Vector3 a = _origin +
                        new Vector3(ax, ay, az) * fineVoxelSize;
            Vector3 b = _origin +
                        new Vector3(bx, by, bz) * fineVoxelSize;
            float t = cell.EdgeOffsets[edge * 2 + slot];
            vertexIndex = vertices.Count;
            vertices.Add(Vector3.LerpUnclamped(a, b, t));
            edgeVertices.Add(key, vertexIndex);
        }
        localVertices[edge] = vertexIndex;
        return vertexIndex;
    }

    private static void ResetDirectionalMcMeshAudit(MeshResult result)
    {
        result.DirectionalMcShadowBoundaryEdges = 0;
        result.DirectionalMcShadowNonManifoldEdges = 0;
        result.DirectionalMcShadowDuplicateTriangles = 0;
        result.DirectionalMcShadowDegenerateTriangles = 0;
        result.DirectionalMcShadowCrackCandidateEdges = 0;
    }

    private int ResolvePaperDmcEdgeVertex(
        PaperDirectionalMcCell cell,
        int mcIndex,
        int edge,
        int[] localVertices,
        Dictionary<DmcEdgeVertexKey, int> edgeVertices,
        List<Vector3> vertices)
    {
        if (edge < 0 || edge >= 12)
            return -1;
        if (localVertices[edge] >= 0)
            return localVertices[edge];
        int endpointA = PaperEdgeEndpointCorners[edge, 0];
        int endpointB = PaperEdgeEndpointCorners[edge, 1];
        int slot = (mcIndex & (1 << endpointA)) != 0 ? 0 : 1;
        if ((cell.EdgeSlotMask[edge] & (1 << slot)) == 0)
            return -1;

        int ax = cell.X + PaperCornerX[endpointA];
        int ay = cell.Y + PaperCornerY[endpointA];
        int az = cell.Z + PaperCornerZ[endpointA];
        int bx = cell.X + PaperCornerX[endpointB];
        int by = cell.Y + PaperCornerY[endpointB];
        int bz = cell.Z + PaperCornerZ[endpointB];
        int voxelA = ax + _dimX * (ay + _dimY * az);
        int voxelB = bx + _dimX * (by + _dimY * bz);
        int minimumVoxel = Mathf.Min(voxelA, voxelB);
        byte axis = (byte)(ax != bx ? 0 : ay != by ? 1 : 2);
        DmcEdgeVertexKey key = new DmcEdgeVertexKey(
            minimumVoxel, axis, (byte)slot);
        if (!edgeVertices.TryGetValue(key, out int vertexIndex))
        {
            float t = cell.EdgeOffsets[edge * 2 + slot];
            Vector3 a = VoxelCenter(ax, ay, az);
            Vector3 b = VoxelCenter(bx, by, bz);
            vertexIndex = vertices.Count;
            vertices.Add(Vector3.LerpUnclamped(a, b, t));
            edgeVertices.Add(key, vertexIndex);
        }
        localVertices[edge] = vertexIndex;
        return vertexIndex;
    }

    private static float PaperSurfaceOffset(float first, float second)
    {
        float denominator = second - first;
        return Mathf.Abs(denominator) > 0.00000001f
            ? -first / denominator
            : 0.5f;
    }

    private static ushort PaperTransitionEdgeMask(int mcIndex)
    {
        ushort mask = 0;
        for (int edge = 0; edge < 12; edge++)
        {
            int a = PaperEdgeEndpointCorners[edge, 0];
            int b = PaperEdgeEndpointCorners[edge, 1];
            bool insideA = (mcIndex & (1 << a)) != 0;
            bool insideB = (mcIndex & (1 << b)) != 0;
            if (insideA != insideB)
                mask |= (ushort)(1 << edge);
        }
        return mask;
    }

    private static int CountPaperMcFaceDisagreements(
        int leftIndex,
        int rightIndex,
        int axis)
    {
        int disagreements = 0;
        for (int leftCorner = 0; leftCorner < 8; leftCorner++)
        {
            if (PaperCornerCoordinate(leftCorner, axis) != 1)
                continue;
            int x = PaperCornerX[leftCorner];
            int y = PaperCornerY[leftCorner];
            int z = PaperCornerZ[leftCorner];
            if (axis == 0)
                x = 0;
            else if (axis == 1)
                y = 0;
            else
                z = 0;
            int rightCorner = FindPaperCorner(x, y, z);
            if (rightCorner < 0)
                continue;
            bool leftInside =
                (leftIndex & (1 << leftCorner)) != 0;
            bool rightInside =
                (rightIndex & (1 << rightCorner)) != 0;
            if (leftInside != rightInside)
                disagreements++;
        }
        return disagreements;
    }

    private static int PaperCornerCoordinate(int corner, int axis)
    {
        if (axis == 0)
            return PaperCornerX[corner];
        if (axis == 1)
            return PaperCornerY[corner];
        return PaperCornerZ[corner];
    }

    private static int FindPaperCorner(int x, int y, int z)
    {
        for (int corner = 0; corner < 8; corner++)
        {
            if (PaperCornerX[corner] == x &&
                PaperCornerY[corner] == y &&
                PaperCornerZ[corner] == z)
            {
                return corner;
            }
        }
        return -1;
    }

    private static void Swap(ref int first, ref int second)
    {
        int temporary = first;
        first = second;
        second = temporary;
    }

    private void EvaluateDirectionalMarchingCubesShadow(
        List<int> orderedCandidateCells,
        int maximumCells,
        int maximumTriangles,
        float minimumWeight,
        float minimumDirectionDot,
        Vector3[] cornerPositions,
        float[] values,
        float[] weights,
        MeshResult result)
    {
        if (PaperReferenceDirectionalMcEnabled)
        {
            EvaluatePaperReferenceDirectionalMarchingCubes(
                orderedCandidateCells,
                maximumCells,
                maximumTriangles,
                minimumWeight,
                cornerPositions,
                result);
            return;
        }

        Stopwatch stopwatch = Stopwatch.StartNew();
        result.DirectionalMcShadowEnabled = _canonicalSixDirectionSemantics;
        if (!_canonicalSixDirectionSemantics)
        {
            stopwatch.Stop();
            result.DirectionalMcShadowMilliseconds = stopwatch.Elapsed.TotalMilliseconds;
            return;
        }

        Dictionary<int, DirectionalMcShadowCell> cells =
            new Dictionary<int, DirectionalMcShadowCell>(
                Mathf.Min(maximumCells, orderedCandidateCells.Count));
        DirectionalMcShadowHypothesis[] hypotheses =
            new DirectionalMcShadowHypothesis[DirectionCount];
        DirectionalMcShadowCombinedIndex[] combined =
            new DirectionalMcShadowCombinedIndex[2];
        bool[] keep = new bool[DirectionCount];
        byte[] components = new byte[4];
        float[,] directionalValues = new float[DirectionCount, 8];
        float[,] directionalWeights = new float[DirectionCount, 8];
        float[] edgeOffsetSamples = new float[DirectionCount];
        float[] edgeOffsetWeightSamples = new float[DirectionCount];
        byte[] cellEdgeCounts = new byte[12];
        float[] cellEdgeOffsets = new float[24];
        float[] cellEdgeWeights = new float[24];
        byte[] directionIndices = new byte[DirectionCount];
        byte[] directionKnownMasks = new byte[DirectionCount];
        float[] mcValues = new float[8];
        bool[] mcActiveEdges = new bool[12];
        int[] mcEdgeComponents = new int[12];
        HashSet<int> seenDecisions = new HashSet<int>();
        _dmcOverflowCells.Clear();
        _dmcOverflowRefinementBlocks.Clear();
        _dmcDecisionRevision++;

        for (int orderedCell = 0;
             orderedCell < orderedCandidateCells.Count &&
             result.DirectionalMcShadowCellsEvaluated < maximumCells;
             orderedCell++)
        {
            int cellIndex = orderedCandidateCells[orderedCell];
            DecodeCell(cellIndex, out int x, out int y, out int z);
            if (x < 0 || y < 0 || z < 0 ||
                x >= _dimX - 1 || y >= _dimY - 1 || z >= _dimZ - 1)
                continue;
            result.DirectionalMcShadowCellsEvaluated++;
            for (int corner = 0; corner < 8; corner++)
            {
                cornerPositions[corner] = VoxelCenter(
                    x + CornerX[corner],
                    y + CornerY[corner],
                    z + CornerZ[corner]);
            }

            int hypothesisCount = 0;
            byte unknownCornerMask = 0;
            Array.Clear(directionIndices, 0, directionIndices.Length);
            Array.Clear(directionKnownMasks, 0, directionKnownMasks.Length);
            for (int direction = 0; direction < DirectionCount; direction++)
            {
                if (!TryBuildComposedHypothesis(
                        direction, x, y, z, minimumWeight,
                        cornerPositions, values, weights,
                        out ComposedHypothesis source))
                    continue;
                bool builtMcIndex = TryBuildDirectionalMcIndex(
                    values, weights, minimumWeight,
                    out byte mcIndex, out ushort transitionEdges,
                    out int supportedCorners, out byte knownCorners,
                    out bool topologyDeferred);
                if (!builtMcIndex && topologyDeferred &&
                    DmcSyntheticFacePrecompletionEnabled)
                {
                    int precompletedCorners =
                        ApplyStableDmcFaceEvidenceToUnknownCorners(
                            cellIndex, x, y, z, source,
                            values, weights, minimumWeight);
                    if (precompletedCorners > 0)
                    {
                        result.DirectionalMcShadowFaceDecisionPrecompletedCorners +=
                            precompletedCorners;
                        builtMcIndex = TryBuildDirectionalMcIndex(
                            values, weights, minimumWeight,
                            out mcIndex, out transitionEdges,
                            out supportedCorners, out knownCorners,
                            out topologyDeferred);
                        if (builtMcIndex)
                        {
                            result.DirectionalMcShadowFaceDecisionRecoveredHypotheses++;
                        }
                    }
                }
                if (!builtMcIndex)
                {
                    if (topologyDeferred)
                        result.DirectionalMcShadowUnknownDeferredCells++;
                    continue;
                }

                result.DirectionalMcShadowRawHypotheses++;
                result.DirectionalMcShadowRawTransitionEdges +=
                    PopCount(transitionEdges);
                if (supportedCorners < 8)
                    result.DirectionalMcShadowIncompleteCornerHypotheses++;
                for (int corner = 0; corner < 8; corner++)
                {
                    directionalValues[direction, corner] = values[corner];
                    directionalWeights[direction, corner] = weights[corner];
                }
                directionIndices[direction] = mcIndex;
                directionKnownMasks[direction] = knownCorners;
                unknownCornerMask |= (byte)~knownCorners;

                Vector3 directionVector = DirectionVectors[direction];
                float compliance = Mathf.Abs(Vector3.Dot(
                    source.Normal, directionVector));
                if (compliance < minimumDirectionDot)
                {
                    result.DirectionalMcShadowIntraDirectionRejected++;
                    continue;
                }
                hypotheses[hypothesisCount++] =
                    new DirectionalMcShadowHypothesis
                    {
                        Direction = direction,
                        Index = mcIndex,
                        KnownCorners = knownCorners,
                        TransitionEdges = transitionEdges,
                        Support = source.Support,
                        Credibility = source.Support * Mathf.Max(0.05f, compliance),
                        Gradient = source.Normal,
                        Normal = source.Normal,
                        Centroid = source.Centroid
                    };
            }

            if (hypothesisCount == 0)
                continue;
            for (int i = 0; i < hypothesisCount - 1; i++)
            {
                for (int j = i + 1; j < hypothesisCount; j++)
                {
                    if (hypotheses[i].Credibility >= hypotheses[j].Credibility)
                        continue;
                    DirectionalMcShadowHypothesis swap = hypotheses[i];
                    hypotheses[i] = hypotheses[j];
                    hypotheses[j] = swap;
                }
            }

            int keptCount = 0;
            for (int i = 0; i < hypothesisCount; i++)
            {
                float consensus = 0f;
                for (int j = 0; j < hypothesisCount; j++)
                {
                    bool agrees = i == j ||
                        (hypotheses[i].TransitionEdges &
                         hypotheses[j].TransitionEdges) != 0;
                    consensus += hypotheses[j].Credibility *
                        (agrees ? 1f : -1f);
                }
                keep[i] = consensus >= 0f;
                if (keep[i])
                    keptCount++;
                else
                    result.DirectionalMcShadowInterDirectionRejected++;
            }
            if (keptCount == 0)
            {
                result.DirectionalMcShadowEmptyAfterVotingCells++;
                continue;
            }
            result.DirectionalMcShadowValidHypotheses += keptCount;
            int offsetOverflowBefore =
                result.DirectionalMcShadowOffsetOverflowEdges;
            EvaluateDirectionalMcShadowEdgeOffsets(
                hypotheses, hypothesisCount, keep,
                directionalValues, directionalWeights,
                edgeOffsetSamples, edgeOffsetWeightSamples,
                cellEdgeCounts, cellEdgeOffsets, cellEdgeWeights,
                minimumWeight, result);
            if (result.DirectionalMcShadowOffsetOverflowEdges >
                offsetOverflowBefore)
            {
                _dmcOverflowCells.Add(cellIndex);
            }

            combined[0] = default;
            combined[1] = default;
            int combinedCount = 0;
            for (int hypothesisIndex = 0;
                 hypothesisIndex < hypothesisCount;
                 hypothesisIndex++)
            {
                if (!keep[hypothesisIndex])
                    continue;
                DirectionalMcShadowHypothesis hypothesis =
                    hypotheses[hypothesisIndex];
                for (int corner = 0; corner < 8; corner++)
                    mcValues[corner] =
                        directionalValues[hypothesis.Direction, corner];
                int componentCount = SplitDirectionalMcIndexComponentsMc33(
                    hypothesis.Index, mcValues, components,
                    mcActiveEdges, mcEdgeComponents,
                    result);
                result.DirectionalMcShadowComponents += componentCount;
                float componentSupport = hypothesis.Credibility /
                    Mathf.Max(1, componentCount);
                for (int componentIndex = 0;
                     componentIndex < componentCount;
                     componentIndex++)
                {
                    byte component = components[componentIndex];
                    int destination = -1;
                    for (int slot = 0; slot < combinedCount; slot++)
                    {
                        if ((combined[slot].Index & component) == 0)
                            continue;
                        destination = slot;
                        break;
                    }
                    if (destination < 0 && combinedCount < 2)
                        destination = combinedCount++;
                    if (destination < 0)
                    {
                        result.DirectionalMcShadowOverflowDeferredComponents++;
                        _dmcOverflowCells.Add(cellIndex);
                        continue;
                    }

                    DirectionalMcShadowCombinedIndex accumulator =
                        combined[destination];
                    if (accumulator.Members == 0)
                        accumulator.Index = component;
                    else
                        accumulator.Index = (byte)(accumulator.Index & component);
                    accumulator.KnownCorners |= hypothesis.KnownCorners;
                    Vector3 alignedNormal =
                        accumulator.NormalWeightedSum.sqrMagnitude > 0.00000001f &&
                        Vector3.Dot(accumulator.NormalWeightedSum, hypothesis.Normal) < 0f
                            ? -hypothesis.Normal
                            : hypothesis.Normal;
                    accumulator.Support += componentSupport;
                    accumulator.NormalWeightedSum += alignedNormal * componentSupport;
                    accumulator.CentroidWeightedSum +=
                        hypothesis.Centroid * componentSupport;
                    accumulator.Members++;
                    combined[destination] = accumulator;
                }
            }

            int writeCount = 0;
            for (int slot = 0; slot < combinedCount; slot++)
            {
                if (combined[slot].Index == 0 || combined[slot].Index == byte.MaxValue)
                    continue;
                if (slot != writeCount)
                    combined[writeCount] = combined[slot];
                writeCount++;
            }
            combinedCount = writeCount;
            if (combinedCount == 0)
            {
                result.DirectionalMcShadowEmptyAfterVotingCells++;
                continue;
            }
            if (combinedCount > 1 && combined[1].Support > combined[0].Support)
            {
                DirectionalMcShadowCombinedIndex swap = combined[0];
                combined[0] = combined[1];
                combined[1] = swap;
            }

            DirectionalMcShadowCell cell = new DirectionalMcShadowCell
            {
                CellIndex = cellIndex,
                X = x,
                Y = y,
                Z = z,
                SurfaceCount = combinedCount,
                CombinedIndex0 = combined[0].Index,
                RegularizedIndex0 = combined[0].Index,
                KnownCornerMask0 = combined[0].KnownCorners,
                Support0 = combined[0].Support,
                Normal0 = combined[0].Normal,
                Centroid0 = combined[0].Centroid
            };
            if (combinedCount > 1)
            {
                cell.CombinedIndex1 = combined[1].Index;
                cell.RegularizedIndex1 = combined[1].Index;
                cell.KnownCornerMask1 = combined[1].KnownCorners;
                cell.Support1 = combined[1].Support;
                cell.Normal1 = combined[1].Normal;
                cell.Centroid1 = combined[1].Centroid;
                result.DirectionalMcShadowDoubleSurfaceCells++;
            }
            else
            {
                result.DirectionalMcShadowSingleSurfaceCells++;
            }
            result.DirectionalMcShadowCombinedTransitionEdges +=
                PopCount(TransitionEdgeMask(cell.CombinedIndex0));
            if (combinedCount > 1)
            {
                result.DirectionalMcShadowCombinedTransitionEdges +=
                    PopCount(TransitionEdgeMask(cell.CombinedIndex1));
            }
            cells[cellIndex] = cell;

            UpdatePersistentDmcDecision(
                cell, unknownCornerMask,
                directionIndices, directionKnownMasks,
                cellEdgeCounts, cellEdgeOffsets, cellEdgeWeights,
                result);
            seenDecisions.Add(cellIndex);
        }

        // MC33 owns per-cell topology.  The canonical face registry below is
        // allowed to complete only genuinely unknown signs; unlike the retired
        // stronger-cell regularizer, it can never overwrite observed evidence.
        AuditDirectionalMcShadowNeighborsBefore(cells, result);
        UpdateAndApplyPersistentDmcFaceDecisions(
            cells,
            orderedCandidateCells.Count <= maximumCells,
            result);
        AuditDirectionalMcShadowNeighborsAfter(cells, result);
        foreach (DirectionalMcShadowCell cell in cells.Values)
        {
            result.DirectionalMcShadowRegularizedTransitionEdges +=
                PopCount(TransitionEdgeMask(cell.RegularizedIndex0));
            if (cell.SurfaceCount > 1)
            {
                result.DirectionalMcShadowRegularizedTransitionEdges +=
                    PopCount(TransitionEdgeMask(cell.RegularizedIndex1));
            }
            if (_dmcCellDecisions.TryGetValue(cell.CellIndex,
                    out DmcCellDecision decision))
            {
                decision.Index0 = cell.RegularizedIndex0;
                decision.Index1 = cell.RegularizedIndex1;
                decision.KnownCornerMask0 = cell.KnownCornerMask0;
                decision.KnownCornerMask1 = cell.KnownCornerMask1;
                decision.DeferredFaceMask0 = cell.DeferredFaceMask0;
                decision.DeferredFaceMask1 = cell.DeferredFaceMask1;
                FillDmcSurfaceValues(decision);
            }
        }
        AuditAndQuarantineDmcSharedFaceTopology(result);
        if (orderedCandidateCells.Count <= maximumCells)
            RetireMissingDmcDecisions(seenDecisions, result);
        else
            result.DirectionalMcShadowPersistentDecisions =
                _dmcCellDecisions.Count;
        BuildPersistentDmcShadowMesh(maximumTriangles, result);
        AuditDirectionalMcShadowMesh(result);
        CollectDmcOverflowRefinementBlocks(result);
        stopwatch.Stop();
        result.DirectionalMcShadowMilliseconds = stopwatch.Elapsed.TotalMilliseconds;
    }

    private int ApplyStableDmcFaceEvidenceToUnknownCorners(
        int cellIndex,
        int x,
        int y,
        int z,
        ComposedHypothesis source,
        float[] values,
        float[] weights,
        float minimumWeight)
    {
        byte proposedMask = 0;
        byte proposedInsideMask = 0;
        byte conflictMask = 0;
        float maximumPlaneOffset =
            _voxelSize * DmcFaceDecisionMaximumPlaneOffsetVoxels;

        for (int axis = 0; axis < 3; axis++)
        {
            for (int side = 0; side < 2; side++)
            {
                bool positiveFace = side == 1;
                int ownerX = x - (!positiveFace && axis == 0 ? 1 : 0);
                int ownerY = y - (!positiveFace && axis == 1 ? 1 : 0);
                int ownerZ = z - (!positiveFace && axis == 2 ? 1 : 0);
                if (ownerX < 0 || ownerY < 0 || ownerZ < 0 ||
                    ownerX >= _dimX - 1 ||
                    ownerY >= _dimY - 1 ||
                    ownerZ >= _dimZ - 1)
                    continue;
                int ownerCellIndex = positiveFace
                    ? cellIndex
                    : CellIndex(ownerX, ownerY, ownerZ);
                DmcFaceKey key =
                    new DmcFaceKey(ownerCellIndex, (byte)axis);
                if (!_dmcFaceDecisions.TryGetValue(
                        key, out DmcFaceDecision faceDecision) ||
                    faceDecision.StableBatches <
                        DmcFaceDecisionMinimumStableBatches ||
                    faceDecision.MissingBatches != 0 ||
                    faceDecision.ConflictMask != 0 ||
                    faceDecision.Normal.sqrMagnitude <= 0.00000001f)
                    continue;

                float normalDot = Mathf.Abs(Vector3.Dot(
                    source.Normal, faceDecision.Normal));
                if (normalDot < DmcFaceDecisionMinimumNormalDot)
                    continue;
                Vector3 centroidDelta =
                    source.Centroid - faceDecision.Centroid;
                if (Mathf.Abs(Vector3.Dot(
                        centroidDelta, faceDecision.Normal)) >
                        maximumPlaneOffset ||
                    Mathf.Abs(Vector3.Dot(
                        centroidDelta, source.Normal)) >
                        maximumPlaneOffset)
                    continue;

                for (int faceCorner = 0; faceCorner < 4; faceCorner++)
                {
                    byte faceBit = (byte)(1 << faceCorner);
                    if ((faceDecision.KnownMask & faceBit) == 0)
                        continue;
                    int corner = DirectionalMcFaceCorner(
                        axis, positiveFace, faceCorner);
                    if (weights[corner] >= minimumWeight)
                        continue;
                    byte cornerBit = (byte)(1 << corner);
                    bool inside =
                        (faceDecision.InsideMask & faceBit) != 0;
                    if ((proposedMask & cornerBit) != 0)
                    {
                        bool previousInside =
                            (proposedInsideMask & cornerBit) != 0;
                        if (previousInside != inside)
                            conflictMask |= cornerBit;
                        continue;
                    }
                    proposedMask |= cornerBit;
                    if (inside)
                        proposedInsideMask |= cornerBit;
                }
            }
        }

        proposedMask &= (byte)~conflictMask;
        if (proposedMask == 0)
            return 0;
        float syntheticMagnitude = Mathf.Max(
            0.00001f,
            Mathf.Min(_truncation, _voxelSize * 0.5f));
        int completed = 0;
        for (int corner = 0; corner < 8; corner++)
        {
            byte cornerBit = (byte)(1 << corner);
            if ((proposedMask & cornerBit) == 0 ||
                weights[corner] >= minimumWeight)
                continue;
            bool inside = (proposedInsideMask & cornerBit) != 0;
            values[corner] = inside
                ? -syntheticMagnitude
                : syntheticMagnitude;
            weights[corner] = minimumWeight;
            completed++;
        }
        return completed;
    }

    private static bool TryBuildDirectionalMcIndex(
        float[] values,
        float[] weights,
        float minimumWeight,
        out byte index,
        out ushort transitionEdges,
        out int supportedCorners,
        out byte knownCorners,
        out bool topologyDeferred)
    {
        index = 0;
        transitionEdges = 0;
        supportedCorners = 0;
        knownCorners = 0;
        topologyDeferred = false;
        for (int corner = 0; corner < 8; corner++)
        {
            if (weights[corner] < minimumWeight)
                continue;
            supportedCorners++;
            knownCorners |= (byte)(1 << corner);
            if (values[corner] < 0f)
                index |= (byte)(1 << corner);
        }
        if (supportedCorners < 4)
            return false;

        // Missing samples are a third state, not implicit free space.  Accept
        // the cell only when assigning every unknown corner either sign gives
        // the same crossing topology.  Otherwise wait for more observations;
        // this deliberately shrinks an uncertain edge instead of inventing a
        // permanent sheet there.
        byte unknownCorners = (byte)~knownCorners;
        byte alternativeIndex = (byte)(index | unknownCorners);
        ushort knownOutsideTransitions = TransitionEdgeMask(index);
        ushort knownInsideTransitions = TransitionEdgeMask(alternativeIndex);
        if (unknownCorners != 0 &&
            knownOutsideTransitions != knownInsideTransitions)
        {
            topologyDeferred = true;
            return false;
        }
        if (index == 0 || index == byte.MaxValue)
            return false;
        transitionEdges = TransitionEdgeMask(index);
        return transitionEdges != 0;
    }

    private static ushort TransitionEdgeMask(byte index)
    {
        ushort mask = 0;
        for (int edge = 0; edge < 12; edge++)
        {
            bool a = (index & (1 << CubeEdges[edge, 0])) != 0;
            bool b = (index & (1 << CubeEdges[edge, 1])) != 0;
            if (a != b)
                mask |= (ushort)(1 << edge);
        }
        return mask;
    }

    private static int SplitDirectionalMcIndexComponents(
        byte index,
        byte[] components)
    {
        byte remaining = index;
        int count = 0;
        while (remaining != 0 && count < components.Length)
        {
            int seed = 0;
            while ((remaining & (1 << seed)) == 0)
                seed++;
            byte component = 0;
            byte frontier = (byte)(1 << seed);
            remaining = (byte)(remaining & ~frontier);
            while (frontier != 0)
            {
                int corner = 0;
                while ((frontier & (1 << corner)) == 0)
                    corner++;
                frontier = (byte)(frontier & ~(1 << corner));
                component |= (byte)(1 << corner);
                for (int edge = 0; edge < 12; edge++)
                {
                    int neighbor = -1;
                    if (CubeEdges[edge, 0] == corner)
                        neighbor = CubeEdges[edge, 1];
                    else if (CubeEdges[edge, 1] == corner)
                        neighbor = CubeEdges[edge, 0];
                    if (neighbor < 0 || (remaining & (1 << neighbor)) == 0)
                        continue;
                    byte neighborBit = (byte)(1 << neighbor);
                    frontier |= neighborBit;
                    remaining = (byte)(remaining & ~neighborBit);
                }
            }
            components[count++] = component;
        }
        return count;
    }

    private static int SplitDirectionalMcIndexComponentsMc33(
        byte index,
        float[] values,
        byte[] components,
        bool[] activeEdges,
        int[] edgeComponents,
        MeshResult result)
    {
        for (int edge = 0; edge < 12; edge++)
        {
            int a = CubeEdges[edge, 0];
            int b = CubeEdges[edge, 1];
            activeEdges[edge] =
                ((index & (1 << a)) != 0) != ((index & (1 << b)) != 0);
            edgeComponents[edge] = -1;
        }
        if (!ScanCoverMc33Topology.TryBuildComponents(
                values, activeEdges, edgeComponents,
                out ScanCoverMc33Topology.Result topology) ||
            topology.ComponentCount <= 0)
        {
            return SplitDirectionalMcIndexComponents(index, components);
        }

        result.DirectionalMcShadowAmbiguousFaces += topology.AmbiguousFaces;
        result.DirectionalMcShadowInteriorTests += topology.InteriorTests;
        int count = Mathf.Min(components.Length, topology.ComponentCount);
        Array.Clear(components, 0, components.Length);
        for (int edge = 0; edge < 12; edge++)
        {
            int component = edgeComponents[edge];
            if (component < 0 || component >= count)
                continue;
            int a = CubeEdges[edge, 0];
            int b = CubeEdges[edge, 1];
            int inside = (index & (1 << a)) != 0 ? a : b;
            components[component] |= (byte)(1 << inside);
        }
        for (int component = 0; component < count; component++)
        {
            if (components[component] == 0)
                components[component] = index;
        }
        return count;
    }

    private void UpdatePersistentDmcDecision(
        DirectionalMcShadowCell cell,
        byte unknownCornerMask,
        byte[] directionIndices,
        byte[] directionKnownMasks,
        byte[] edgeCounts,
        float[] edgeOffsets,
        float[] edgeWeights,
        MeshResult result)
    {
        bool existed = _dmcCellDecisions.TryGetValue(
            cell.CellIndex, out DmcCellDecision decision);
        if (!existed)
        {
            decision = new DmcCellDecision
            {
                CellIndex = cell.CellIndex,
                X = cell.X,
                Y = cell.Y,
                Z = cell.Z
            };
            _dmcCellDecisions.Add(cell.CellIndex, decision);
        }

        bool changed = existed &&
            (decision.SurfaceCount != cell.SurfaceCount ||
             decision.Index0 != cell.RegularizedIndex0 ||
             decision.Index1 != cell.RegularizedIndex1);
        decision.SurfaceCount = cell.SurfaceCount;
        decision.Index0 = cell.RegularizedIndex0;
        decision.Index1 = cell.RegularizedIndex1;
        decision.KnownCornerMask0 = cell.KnownCornerMask0;
        decision.KnownCornerMask1 = cell.KnownCornerMask1;
        byte selectedUnknownMask = (byte)~cell.KnownCornerMask0;
        if (cell.SurfaceCount > 1)
            selectedUnknownMask |= (byte)~cell.KnownCornerMask1;
        // Keep the broader directional diagnostic in the record, but do not
        // use it to authorize completion.  Completion is controlled solely by
        // the per-selected-surface known masks above.
        decision.UnknownCornerMask =
            (byte)(selectedUnknownMask | unknownCornerMask);
        decision.Overflow = _dmcOverflowCells.Contains(cell.CellIndex);
        decision.LastUpdatedBatch = _batchSequence;
        decision.Revision = _dmcDecisionRevision;
        decision.MissingBatches = 0;
        decision.StableBatches = changed ? 0 : decision.StableBatches + 1;
        decision.Support0 = cell.Support0;
        decision.Support1 = cell.Support1;
        decision.Normal0 = cell.Normal0;
        decision.Normal1 = cell.Normal1;
        decision.Centroid0 = cell.Centroid0;
        decision.Centroid1 = cell.Centroid1;
        // Face conflicts are recomputed from the current raw observations after
        // every cell has been evaluated.  Never carry a previous quarantine
        // into the new decision before that audit has run.
        decision.DeferredFaceMask0 = 0;
        decision.DeferredFaceMask1 = 0;
        Array.Copy(directionIndices, decision.DirectionIndices, DirectionCount);
        Array.Copy(directionKnownMasks, decision.DirectionKnownMasks, DirectionCount);
        Array.Copy(edgeCounts, decision.EdgeCounts, 12);
        AssignDmcEdgeOffsetsToSurfaces(
            decision, edgeOffsets, edgeWeights);
        FillDmcSurfaceValues(decision);
        if (changed)
            result.DirectionalMcShadowChangedDecisions++;
    }

    private void AssignDmcEdgeOffsetsToSurfaces(
        DmcCellDecision decision,
        float[] sortedOffsets,
        float[] sortedWeights)
    {
        for (int edge = 0; edge < 12; edge++)
        {
            int baseIndex = edge * 2;
            int count = decision.EdgeCounts[edge];
            if (count <= 0)
                continue;
            float first = sortedOffsets[baseIndex];
            float second = sortedOffsets[baseIndex + 1];
            float firstWeight = sortedWeights[baseIndex];
            float secondWeight = sortedWeights[baseIndex + 1];
            int cornerA = CubeEdges[edge, 0];
            int cornerB = CubeEdges[edge, 1];
            Vector3 a = VoxelCenter(
                decision.X + CornerX[cornerA],
                decision.Y + CornerY[cornerA],
                decision.Z + CornerZ[cornerA]);
            Vector3 b = VoxelCenter(
                decision.X + CornerX[cornerB],
                decision.Y + CornerY[cornerB],
                decision.Z + CornerZ[cornerB]);
            Vector3 edgeVector = b - a;
            float inverseLengthSquared =
                1f / Mathf.Max(0.00000001f, edgeVector.sqrMagnitude);
            float surface0Projection = Mathf.Clamp01(
                Vector3.Dot(decision.Centroid0 - a, edgeVector) *
                inverseLengthSquared);
            bool swap;
            if (count == 1)
            {
                swap = false;
            }
            else if (decision.SurfaceCount <= 1)
            {
                swap = Mathf.Abs(second - surface0Projection) <
                       Mathf.Abs(first - surface0Projection);
            }
            else
            {
                float surface1Projection = Mathf.Clamp01(
                    Vector3.Dot(decision.Centroid1 - a, edgeVector) *
                    inverseLengthSquared);
                float directCost =
                    Mathf.Abs(first - surface0Projection) +
                    Mathf.Abs(second - surface1Projection);
                float swappedCost =
                    Mathf.Abs(second - surface0Projection) +
                    Mathf.Abs(first - surface1Projection);
                swap = swappedCost < directCost;
            }
            decision.EdgeOffsets[baseIndex] = swap ? second : first;
            decision.EdgeOffsets[baseIndex + 1] = swap ? first : second;
            decision.EdgeWeights[baseIndex] =
                swap ? secondWeight : firstWeight;
            decision.EdgeWeights[baseIndex + 1] =
                swap ? firstWeight : secondWeight;
        }
    }

    private void FillDmcSurfaceValues(DmcCellDecision decision)
    {
        for (int surface = 0; surface < 2; surface++)
        {
            byte index = surface == 0 ? decision.Index0 : decision.Index1;
            for (int corner = 0; corner < 8; corner++)
            {
                // The combined DMC record currently retains signs and measured
                // edge offsets, not the original eight scalar magnitudes.  A
                // uniform +/-1 field makes every checkerboard face an exact
                // asymptotic-decider tie, so the two incident cells can select
                // different MC33 branches.  Use a tiny global-corner-stable
                // magnitude only as a deterministic tie breaker.  It does not
                // move geometry; measured edge offsets still own vertex position.
                float magnitude = DmcCanonicalCornerMagnitude(decision, corner);
                decision.SurfaceValues[surface * 8 + corner] =
                    (index & (1 << corner)) != 0 ? -magnitude : magnitude;
            }
        }
    }

    private static float DmcCanonicalCornerMagnitude(
        DmcCellDecision decision,
        int corner)
    {
        int x = decision.X + CornerX[corner];
        int y = decision.Y + CornerY[corner];
        int z = decision.Z + CornerZ[corner];
        int hash;
        unchecked
        {
            hash = x * 73856093 ^ y * 19349663 ^ z * 83492791;
        }
        float unit = ((hash & 255) + 1) / 256f;
        return 1f + unit * 0.01f;
    }

    private void AuditAndQuarantineDmcSharedFaceTopology(
        MeshResult result)
    {
        List<int> ordered = new List<int>(_dmcCellDecisions.Keys);
        ordered.Sort();
        List<int> leftPattern = new List<int>(48);
        List<int> rightPattern = new List<int>(48);
        HashSet<ulong> leftSegments = new HashSet<ulong>();
        HashSet<ulong> rightSegments = new HashSet<ulong>();
        float[] faceValues = new float[8];
        long[] faceEdges = new long[3];
        float maximumPlaneOffset =
            _voxelSize * DmcFaceDecisionMaximumPlaneOffsetVoxels;

        for (int decisionIndex = 0;
             decisionIndex < ordered.Count;
             decisionIndex++)
        {
            DmcCellDecision left = _dmcCellDecisions[ordered[decisionIndex]];
            if (left.MissingBatches != 0 || left.SurfaceCount != 1)
                continue;
            for (int axis = 0; axis < 3; axis++)
            {
                int nx = left.X + (axis == 0 ? 1 : 0);
                int ny = left.Y + (axis == 1 ? 1 : 0);
                int nz = left.Z + (axis == 2 ? 1 : 0);
                if (nx >= _dimX - 1 ||
                    ny >= _dimY - 1 ||
                    nz >= _dimZ - 1 ||
                    !_dmcCellDecisions.TryGetValue(
                        CellIndex(nx, ny, nz),
                        out DmcCellDecision right) ||
                    right.MissingBatches != 0 ||
                    right.SurfaceCount != 1)
                    continue;

                float signedNormalDot =
                    Vector3.Dot(left.Normal0, right.Normal0);
                if (Mathf.Abs(signedNormalDot) <
                    DmcFaceDecisionMinimumNormalDot)
                    continue;
                Vector3 alignedRightNormal =
                    signedNormalDot < 0f ? -right.Normal0 : right.Normal0;
                Vector3 centroidDelta = right.Centroid0 - left.Centroid0;
                if (Mathf.Abs(Vector3.Dot(
                        centroidDelta, left.Normal0)) >
                        maximumPlaneOffset ||
                    Mathf.Abs(Vector3.Dot(
                        centroidDelta, alignedRightNormal)) >
                        maximumPlaneOffset)
                    continue;

                result.DirectionalMcShadowSharedFaceTopologyComparisons++;
                bool leftBuilt = TryCollectDmcFaceTopologySegments(
                    left, 0, axis, true,
                    leftPattern, leftSegments,
                    faceValues, faceEdges);
                bool rightBuilt = TryCollectDmcFaceTopologySegments(
                    right, 0, axis, false,
                    rightPattern, rightSegments,
                    faceValues, faceEdges);
                bool matches =
                    leftBuilt == rightBuilt &&
                    (!leftBuilt || leftSegments.SetEquals(rightSegments));
                if (matches)
                    continue;

                result.DirectionalMcShadowSharedFaceTopologyMismatches++;
                left.DeferredFaceMask0 |= DmcFaceMaskBit(axis, true);
                right.DeferredFaceMask0 |= DmcFaceMaskBit(axis, false);
                _dmcCellDecisions[left.CellIndex] = left;
                _dmcCellDecisions[right.CellIndex] = right;
            }
        }
    }

    private bool TryCollectDmcFaceTopologySegments(
        DmcCellDecision decision,
        int surface,
        int axis,
        bool positiveFace,
        List<int> pattern,
        HashSet<ulong> segments,
        float[] values,
        long[] faceEdges)
    {
        segments.Clear();
        Array.Copy(
            decision.SurfaceValues, surface * 8,
            values, 0, 8);
        if (!ScanCoverMc33Topology.TryBuildTriangles(
                values, pattern, out ScanCoverMc33Topology.Result unused))
            return false;

        for (int i = 0; i + 2 < pattern.Count; i += 3)
        {
            int count = 0;
            for (int vertex = 0; vertex < 3; vertex++)
            {
                int patternVertex = pattern[i + vertex];
                if (!DmcPatternVertexOnFace(
                        patternVertex, axis, positiveFace))
                    continue;
                faceEdges[count++] =
                    DmcGlobalGridEdgeId(decision, patternVertex);
            }
            if (count == 2)
            {
                AddDmcFaceSegment(
                    faceEdges[0], faceEdges[1], segments);
            }
            else if (count == 3)
            {
                AddDmcFaceSegment(
                    faceEdges[0], faceEdges[1], segments);
                AddDmcFaceSegment(
                    faceEdges[1], faceEdges[2], segments);
                AddDmcFaceSegment(
                    faceEdges[2], faceEdges[0], segments);
            }
        }
        return true;
    }

    private long DmcGlobalGridEdgeId(
        DmcCellDecision decision,
        int edge)
    {
        int cornerA = CubeEdges[edge, 0];
        int cornerB = CubeEdges[edge, 1];
        int ax = decision.X + CornerX[cornerA];
        int ay = decision.Y + CornerY[cornerA];
        int az = decision.Z + CornerZ[cornerA];
        int bx = decision.X + CornerX[cornerB];
        int by = decision.Y + CornerY[cornerB];
        int bz = decision.Z + CornerZ[cornerB];
        int voxelA = ax + _dimX * (ay + _dimY * az);
        int voxelB = bx + _dimX * (by + _dimY * bz);
        int minimumVoxel = Mathf.Min(voxelA, voxelB);
        int edgeAxis = ax != bx ? 0 : ay != by ? 1 : 2;
        return (long)minimumVoxel * 3L + edgeAxis;
    }

    private static void AddDmcFaceSegment(
        long a,
        long b,
        HashSet<ulong> segments)
    {
        if (a == b || a < 0L || b < 0L ||
            a > uint.MaxValue || b > uint.MaxValue)
            return;
        uint first = (uint)Math.Min(a, b);
        uint second = (uint)Math.Max(a, b);
        segments.Add(((ulong)first << 32) | second);
    }

    private void RetireMissingDmcDecisions(
        HashSet<int> seenDecisions,
        MeshResult result)
    {
        List<int> retire = null;
        foreach (KeyValuePair<int, DmcCellDecision> pair in _dmcCellDecisions)
        {
            if (seenDecisions.Contains(pair.Key))
                continue;
            DmcCellDecision decision = pair.Value;
            decision.MissingBatches++;
            // Persistent decisions bridge a single incomplete scan, but an
            // absent surface is still allowed to correct itself promptly.
            if (decision.MissingBatches <= 2)
                continue;
            if (retire == null)
                retire = new List<int>();
            retire.Add(pair.Key);
        }
        if (retire != null)
        {
            for (int i = 0; i < retire.Count; i++)
                _dmcCellDecisions.Remove(retire[i]);
            result.DirectionalMcShadowRetiredDecisions = retire.Count;
        }
        result.DirectionalMcShadowPersistentDecisions =
            _dmcCellDecisions.Count;
    }

    private void CollectDmcOverflowRefinementBlocks(MeshResult result)
    {
        foreach (int cellIndex in _dmcOverflowCells)
        {
            DecodeCell(cellIndex, out int x, out int y, out int z);
            int blockX = Mathf.Clamp(x / _blockSize, 0, _blockDimX - 1);
            int blockY = Mathf.Clamp(y / _blockSize, 0, _blockDimY - 1);
            int blockZ = Mathf.Clamp(z / _blockSize, 0, _blockDimZ - 1);
            _dmcOverflowRefinementBlocks.Add(
                blockX + _blockDimX * (blockY + _blockDimY * blockZ));
        }
        result.DirectionalMcShadowOverflowRefinementBlocks =
            _dmcOverflowRefinementBlocks.Count;
    }

    private void BuildPersistentDmcShadowMesh(
        int maximumTriangles,
        MeshResult result)
    {
        result.DmcShadowVertices.Clear();
        result.DmcShadowTriangles.Clear();
        result.DmcShadowLines.Clear();
        Dictionary<DmcEdgeVertexKey, int> edgeVertices =
            new Dictionary<DmcEdgeVertexKey, int>(32768);
        List<int> pattern = new List<int>(48);
        float[] surfaceValues = new float[8];
        int[] localVertices = new int[13];
        int triangleLimit = Mathf.Max(1, maximumTriangles);
        HashSet<int> unmeasuredEdgeCells = new HashSet<int>();

        List<int> ordered = new List<int>(_dmcCellDecisions.Keys);
        ordered.Sort();
        for (int decisionIndex = 0;
             decisionIndex < ordered.Count &&
             result.DmcShadowTriangles.Count / 3 < triangleLimit;
             decisionIndex++)
        {
            DmcCellDecision decision = _dmcCellDecisions[ordered[decisionIndex]];
            for (int surface = 0;
                 surface < decision.SurfaceCount &&
                 result.DmcShadowTriangles.Count / 3 < triangleLimit;
                 surface++)
            {
                byte surfaceIndex =
                    surface == 0 ? decision.Index0 : decision.Index1;
                if (!DmcSurfaceHasMeasuredCrossings(
                        decision, surfaceIndex))
                {
                    unmeasuredEdgeCells.Add(decision.CellIndex);
                    continue;
                }
                Array.Copy(
                    decision.SurfaceValues, surface * 8,
                    surfaceValues, 0, 8);
                if (!ScanCoverMc33Topology.TryBuildTriangles(
                        surfaceValues, pattern,
                        out ScanCoverMc33Topology.Result topology))
                    continue;
                result.DirectionalMcShadowAmbiguousFaces +=
                    topology.AmbiguousFaces;
                result.DirectionalMcShadowInteriorTests +=
                    topology.InteriorTests;

                for (int i = 0; i < localVertices.Length; i++)
                    localVertices[i] = -1;
                for (int i = 0; i < pattern.Count; i += 3)
                {
                    if (result.DmcShadowTriangles.Count / 3 >= triangleLimit)
                        break;
                    if (DmcTriangleTouchesDeferredFace(
                            decision, surface,
                            pattern[i], pattern[i + 1], pattern[i + 2]))
                    {
                        result.DirectionalMcShadowConflictFaceDeferredTriangles++;
                        continue;
                    }
                    int a = ResolveDmcPatternVertex(
                        decision, surface, pattern[i],
                        localVertices, edgeVertices, result.DmcShadowVertices);
                    int b = ResolveDmcPatternVertex(
                        decision, surface, pattern[i + 1],
                        localVertices, edgeVertices, result.DmcShadowVertices);
                    int c = ResolveDmcPatternVertex(
                        decision, surface, pattern[i + 2],
                        localVertices, edgeVertices, result.DmcShadowVertices);
                    if (a < 0 || b < 0 || c < 0 ||
                        a == b || b == c || c == a)
                    {
                        result.DirectionalMcShadowUnmeasuredEdgeDeferredTriangles++;
                        continue;
                    }
                    Vector3 pa = result.DmcShadowVertices[a];
                    Vector3 pb = result.DmcShadowVertices[b];
                    Vector3 pc = result.DmcShadowVertices[c];
                    Vector3 cross = Vector3.Cross(pb - pa, pc - pa);
                    float crossMagnitude = cross.magnitude;
                    if (crossMagnitude < 0.00001f)
                        continue;
                    Vector3 surfaceNormal =
                        surface == 0 ? decision.Normal0 : decision.Normal1;
                    if (surfaceNormal.sqrMagnitude > 0.00000001f)
                    {
                        surfaceNormal.Normalize();
                        float signedAlignment =
                            Vector3.Dot(cross / crossMagnitude, surfaceNormal);
                        if (Mathf.Abs(signedAlignment) < 0.35f)
                        {
                            result.DirectionalMcShadowNormalMismatchTriangles++;
                        }
                        if (signedAlignment < 0f)
                        {
                            int swap = b;
                            b = c;
                            c = swap;
                            result.DirectionalMcShadowWindingCorrectedTriangles++;
                        }
                    }
                    result.DmcShadowTriangles.Add(a);
                    result.DmcShadowTriangles.Add(b);
                    result.DmcShadowTriangles.Add(c);
                    result.DmcShadowLines.Add(a);
                    result.DmcShadowLines.Add(b);
                    result.DmcShadowLines.Add(b);
                    result.DmcShadowLines.Add(c);
                    result.DmcShadowLines.Add(c);
                    result.DmcShadowLines.Add(a);
                }
            }
        }
        result.DirectionalMcShadowUnmeasuredEdgeDeferredCells =
            unmeasuredEdgeCells.Count;
        result.DirectionalMcShadowVertices = result.DmcShadowVertices.Count;
        result.DirectionalMcShadowTriangles =
            result.DmcShadowTriangles.Count / 3;
    }

    private static bool DmcSurfaceHasMeasuredCrossings(
        DmcCellDecision decision,
        byte index)
    {
        ushort transitionMask = TransitionEdgeMask(index);
        for (int edge = 0; edge < 12; edge++)
        {
            if ((transitionMask & (1 << edge)) != 0 &&
                decision.EdgeCounts[edge] <= 0)
                return false;
        }
        return true;
    }

    private static bool DmcTriangleTouchesDeferredFace(
        DmcCellDecision decision,
        int surface,
        int a,
        int b,
        int c)
    {
        byte deferredMask =
            surface == 0
                ? decision.DeferredFaceMask0
                : decision.DeferredFaceMask1;
        if (deferredMask == 0)
            return false;
        for (int axis = 0; axis < 3; axis++)
        {
            for (int side = 0; side < 2; side++)
            {
                bool positiveFace = side == 1;
                if ((deferredMask & DmcFaceMaskBit(axis, positiveFace)) == 0)
                    continue;
                int verticesOnFace = 0;
                if (DmcPatternVertexOnFace(a, axis, positiveFace))
                    verticesOnFace++;
                if (DmcPatternVertexOnFace(b, axis, positiveFace))
                    verticesOnFace++;
                if (DmcPatternVertexOnFace(c, axis, positiveFace))
                    verticesOnFace++;
                if (verticesOnFace >= 2)
                    return true;
            }
        }
        return false;
    }

    private static bool DmcPatternVertexOnFace(
        int patternVertex,
        int axis,
        bool positiveFace)
    {
        if (patternVertex < 0 || patternVertex >= 12)
            return false;
        int cornerA = CubeEdges[patternVertex, 0];
        int cornerB = CubeEdges[patternVertex, 1];
        int expected = positiveFace ? 1 : 0;
        return DmcCornerAxisCoordinate(cornerA, axis) == expected &&
               DmcCornerAxisCoordinate(cornerB, axis) == expected;
    }

    private static int DmcCornerAxisCoordinate(int corner, int axis)
    {
        if (axis == 0)
            return CornerX[corner];
        if (axis == 1)
            return CornerY[corner];
        return CornerZ[corner];
    }

    private void AuditDirectionalMcShadowMesh(MeshResult result)
    {
        Dictionary<ulong, int> edgeUse =
            new Dictionary<ulong, int>(result.DmcShadowTriangles.Count);
        HashSet<TriangleKey> uniqueTriangles =
            new HashSet<TriangleKey>();
        for (int i = 0; i + 2 < result.DmcShadowTriangles.Count; i += 3)
        {
            int a = result.DmcShadowTriangles[i];
            int b = result.DmcShadowTriangles[i + 1];
            int c = result.DmcShadowTriangles[i + 2];
            if (a < 0 || b < 0 || c < 0 ||
                a >= result.DmcShadowVertices.Count ||
                b >= result.DmcShadowVertices.Count ||
                c >= result.DmcShadowVertices.Count ||
                a == b || b == c || c == a)
            {
                result.DirectionalMcShadowDegenerateTriangles++;
                continue;
            }
            Vector3 pa = result.DmcShadowVertices[a];
            Vector3 pb = result.DmcShadowVertices[b];
            Vector3 pc = result.DmcShadowVertices[c];
            if (!Finite(pa) || !Finite(pb) || !Finite(pc) ||
                Vector3.Cross(pb - pa, pc - pa).sqrMagnitude <
                    0.0000000001f)
            {
                result.DirectionalMcShadowDegenerateTriangles++;
                continue;
            }
            if (!uniqueTriangles.Add(new TriangleKey(-1, a, b, c)))
            {
                result.DirectionalMcShadowDuplicateTriangles++;
                continue;
            }
            IncrementDmcAuditEdge(edgeUse, a, b);
            IncrementDmcAuditEdge(edgeUse, b, c);
            IncrementDmcAuditEdge(edgeUse, c, a);
        }

        Vector3 volumeMinimum = VoxelCenter(0, 0, 0);
        Vector3 volumeMaximum = VoxelCenter(
            _dimX - 1, _dimY - 1, _dimZ - 1);
        float boundaryMargin = _voxelSize * 1.1f;
        foreach (KeyValuePair<ulong, int> pair in edgeUse)
        {
            if (pair.Value > 2)
            {
                result.DirectionalMcShadowNonManifoldEdges++;
                continue;
            }
            if (pair.Value != 1)
                continue;
            result.DirectionalMcShadowBoundaryEdges++;
            int a = (int)(pair.Key >> 32);
            int b = (int)(pair.Key & uint.MaxValue);
            Vector3 midpoint =
                (result.DmcShadowVertices[a] +
                 result.DmcShadowVertices[b]) * 0.5f;
            if (!NearDmcVolumeBoundary(
                    midpoint, volumeMinimum, volumeMaximum, boundaryMargin))
            {
                result.DirectionalMcShadowCrackCandidateEdges++;
            }
        }
    }

    private static void IncrementDmcAuditEdge(
        Dictionary<ulong, int> edgeUse,
        int a,
        int b)
    {
        ulong key = EdgeKey(a, b);
        edgeUse.TryGetValue(key, out int count);
        edgeUse[key] = count + 1;
    }

    private static bool NearDmcVolumeBoundary(
        Vector3 point,
        Vector3 minimum,
        Vector3 maximum,
        float margin)
    {
        return point.x <= minimum.x + margin ||
               point.y <= minimum.y + margin ||
               point.z <= minimum.z + margin ||
               point.x >= maximum.x - margin ||
               point.y >= maximum.y - margin ||
               point.z >= maximum.z - margin;
    }

    private int ResolveDmcPatternVertex(
        DmcCellDecision decision,
        int surface,
        int patternVertex,
        int[] localVertices,
        Dictionary<DmcEdgeVertexKey, int> edgeVertices,
        List<Vector3> vertices)
    {
        if (patternVertex < 0 || patternVertex > 12)
            return -1;
        if (localVertices[patternVertex] >= 0)
            return localVertices[patternVertex];
        if (patternVertex == 12)
        {
            Vector3 sum = Vector3.zero;
            int count = 0;
            for (int edge = 0; edge < 12; edge++)
            {
                byte index = surface == 0 ? decision.Index0 : decision.Index1;
                if ((TransitionEdgeMask(index) & (1 << edge)) == 0)
                    continue;
                int vertex = ResolveDmcPatternVertex(
                    decision, surface, edge,
                    localVertices, edgeVertices, vertices);
                if (vertex < 0)
                    continue;
                sum += vertices[vertex];
                count++;
            }
            if (count == 0)
                return -1;
            int center = vertices.Count;
            vertices.Add(sum / count);
            localVertices[12] = center;
            return center;
        }

        int edgeIndex = patternVertex;
        int cornerA = CubeEdges[edgeIndex, 0];
        int cornerB = CubeEdges[edgeIndex, 1];
        int ax = decision.X + CornerX[cornerA];
        int ay = decision.Y + CornerY[cornerA];
        int az = decision.Z + CornerZ[cornerA];
        int bx = decision.X + CornerX[cornerB];
        int by = decision.Y + CornerY[cornerB];
        int bz = decision.Z + CornerZ[cornerB];
        int voxelA = ax + _dimX * (ay + _dimY * az);
        int voxelB = bx + _dimX * (by + _dimY * bz);
        float t = decision.EdgeOffset(edgeIndex, surface);
        int countOnEdge = decision.EdgeCounts[edgeIndex];
        float otherT = countOnEdge > 1 && decision.SurfaceCount > 1
            ? decision.EdgeOffset(edgeIndex, 1 - Mathf.Clamp(surface, 0, 1))
            : t;
        if (voxelA > voxelB)
        {
            int swap = voxelA;
            voxelA = voxelB;
            voxelB = swap;
            t = 1f - t;
            otherT = 1f - otherT;
        }
        byte slot = (byte)(countOnEdge > 1 && decision.SurfaceCount > 1
            ? (t <= otherT ? 0 : 1)
            : 0);
        byte axis = (byte)(ax != bx ? 0 : ay != by ? 1 : 2);
        DmcEdgeVertexKey key = new DmcEdgeVertexKey(voxelA, axis, slot);
        if (!edgeVertices.TryGetValue(key, out int vertexIndex))
        {
            Vector3 a = VoxelCenter(ax, ay, az);
            Vector3 b = VoxelCenter(bx, by, bz);
            vertexIndex = vertices.Count;
            vertices.Add(Vector3.Lerp(a, b, Mathf.Clamp01(t)));
            edgeVertices.Add(key, vertexIndex);
        }
        localVertices[patternVertex] = vertexIndex;
        return vertexIndex;
    }

    private static void EvaluateDirectionalMcShadowEdgeOffsets(
        DirectionalMcShadowHypothesis[] hypotheses,
        int hypothesisCount,
        bool[] keep,
        float[,] values,
        float[,] weights,
        float[] offsets,
        float[] offsetWeights,
        byte[] edgeCounts,
        float[] resolvedOffsets,
        float[] resolvedWeights,
        float minimumWeight,
        MeshResult result)
    {
        Array.Clear(edgeCounts, 0, edgeCounts.Length);
        Array.Clear(resolvedOffsets, 0, resolvedOffsets.Length);
        Array.Clear(resolvedWeights, 0, resolvedWeights.Length);
        for (int edge = 0; edge < 12; edge++)
        {
            int offsetCount = 0;
            int a = CubeEdges[edge, 0];
            int b = CubeEdges[edge, 1];
            for (int hypothesisIndex = 0;
                 hypothesisIndex < hypothesisCount;
                 hypothesisIndex++)
            {
                if (!keep[hypothesisIndex])
                    continue;
                DirectionalMcShadowHypothesis hypothesis =
                    hypotheses[hypothesisIndex];
                if ((hypothesis.TransitionEdges & (1 << edge)) == 0)
                    continue;
                int direction = hypothesis.Direction;
                float wa = weights[direction, a];
                float wb = weights[direction, b];
                if (wa < minimumWeight || wb < minimumWeight)
                    continue;
                float va = values[direction, a];
                float vb = values[direction, b];
                float denominator = va - vb;
                offsets[offsetCount] = Mathf.Abs(denominator) > 0.000001f
                    ? Mathf.Clamp01(va / denominator)
                    : 0.5f;
                offsetWeights[offsetCount] =
                    Mathf.Min(wa, wb) * hypothesis.Credibility;
                offsetCount++;
            }
            if (offsetCount == 0)
                continue;
            for (int i = 0; i < offsetCount - 1; i++)
            {
                for (int j = i + 1; j < offsetCount; j++)
                {
                    if (offsets[i] <= offsets[j])
                        continue;
                    float offsetSwap = offsets[i];
                    offsets[i] = offsets[j];
                    offsets[j] = offsetSwap;
                    float weightSwap = offsetWeights[i];
                    offsetWeights[i] = offsetWeights[j];
                    offsetWeights[j] = weightSwap;
                }
            }
            int clusters = 1;
            float clusterNumerator = offsets[0] * offsetWeights[0];
            float clusterDenominator = offsetWeights[0];
            float clusterCenter = offsets[0];
            int storedClusters = 0;
            for (int i = 1; i < offsetCount; i++)
            {
                if (Mathf.Abs(offsets[i] - clusterCenter) >
                    DirectionalMcShadowOffsetSeparation)
                {
                    if (storedClusters < 2)
                    {
                        int target = edge * 2 + storedClusters;
                        resolvedOffsets[target] = clusterCenter;
                        resolvedWeights[target] = clusterDenominator;
                        storedClusters++;
                    }
                    clusters++;
                    clusterNumerator = offsets[i] * offsetWeights[i];
                    clusterDenominator = offsetWeights[i];
                    clusterCenter = offsets[i];
                    continue;
                }
                clusterNumerator += offsets[i] * offsetWeights[i];
                clusterDenominator += offsetWeights[i];
                clusterCenter = clusterDenominator > 0f
                    ? clusterNumerator / clusterDenominator
                    : clusterCenter;
            }
            if (storedClusters < 2)
            {
                int target = edge * 2 + storedClusters;
                resolvedOffsets[target] = clusterCenter;
                resolvedWeights[target] = clusterDenominator;
                storedClusters++;
            }
            edgeCounts[edge] = (byte)Mathf.Min(2, storedClusters);
            result.DirectionalMcShadowOffsetClusters += Mathf.Min(2, clusters);
            if (clusters >= 2)
                result.DirectionalMcShadowDualOffsetEdges++;
            if (clusters > 2)
                result.DirectionalMcShadowOffsetOverflowEdges++;
        }
    }

    private void RegularizeDirectionalMcShadowCells(
        Dictionary<int, DirectionalMcShadowCell> cells,
        MeshResult result)
    {
        List<int> ordered = new List<int>(cells.Keys);
        ordered.Sort();
        HashSet<int> changed = new HashSet<int>();
        for (int keyIndex = 0; keyIndex < ordered.Count; keyIndex++)
        {
            int key = ordered[keyIndex];
            for (int axis = 0; axis < 3; axis++)
            {
                DirectionalMcShadowCell left = cells[key];
                int nx = left.X + (axis == 0 ? 1 : 0);
                int ny = left.Y + (axis == 1 ? 1 : 0);
                int nz = left.Z + (axis == 2 ? 1 : 0);
                if (!cells.TryGetValue(CellIndex(nx, ny, nz),
                        out DirectionalMcShadowCell right))
                    continue;
                result.DirectionalMcShadowNeighborFaceComparisons++;
                int before = CountDirectionalMcFaceDisagreements(
                    left.RegularizedIndex0, right.RegularizedIndex0, axis);
                result.DirectionalMcShadowNeighborDisagreementsBefore += before;
                if (before == 0)
                    continue;
                if (left.SurfaceCount != 1 || right.SurfaceCount != 1 ||
                    Vector3.Dot(left.Normal0, right.Normal0) <
                    DirectionalMcShadowRegularizationNormalDot)
                {
                    result.DirectionalMcShadowRegularizationDeferredPairs++;
                    continue;
                }

                bool leftStronger = left.Support0 >=
                    right.Support0 * DirectionalMcShadowRegularizationSupportRatio;
                bool rightStronger = right.Support0 >=
                    left.Support0 * DirectionalMcShadowRegularizationSupportRatio;
                if (!leftStronger && !rightStronger)
                {
                    result.DirectionalMcShadowRegularizationDeferredPairs++;
                    continue;
                }
                for (int faceCorner = 0; faceCorner < 4; faceCorner++)
                {
                    int leftCorner = DirectionalMcFaceCorner(axis, true, faceCorner);
                    int rightCorner = DirectionalMcFaceCorner(axis, false, faceCorner);
                    bool leftInside =
                        (left.RegularizedIndex0 & (1 << leftCorner)) != 0;
                    bool rightInside =
                        (right.RegularizedIndex0 & (1 << rightCorner)) != 0;
                    if (leftInside == rightInside)
                        continue;
                    if (leftStronger)
                    {
                        byte candidate = SetDirectionalMcCorner(
                            right.RegularizedIndex0, rightCorner, leftInside);
                        if (candidate == 0 || candidate == byte.MaxValue)
                            continue;
                        right.RegularizedIndex0 = candidate;
                        changed.Add(right.CellIndex);
                    }
                    else
                    {
                        byte candidate = SetDirectionalMcCorner(
                            left.RegularizedIndex0, leftCorner, rightInside);
                        if (candidate == 0 || candidate == byte.MaxValue)
                            continue;
                        left.RegularizedIndex0 = candidate;
                        changed.Add(left.CellIndex);
                    }
                    result.DirectionalMcShadowRegularizedCorners++;
                }
                cells[left.CellIndex] = left;
                cells[right.CellIndex] = right;
            }
        }
        result.DirectionalMcShadowRegularizedCells = changed.Count;
        for (int keyIndex = 0; keyIndex < ordered.Count; keyIndex++)
        {
            DirectionalMcShadowCell left = cells[ordered[keyIndex]];
            for (int axis = 0; axis < 3; axis++)
            {
                int nx = left.X + (axis == 0 ? 1 : 0);
                int ny = left.Y + (axis == 1 ? 1 : 0);
                int nz = left.Z + (axis == 2 ? 1 : 0);
                if (!cells.TryGetValue(CellIndex(nx, ny, nz),
                        out DirectionalMcShadowCell right))
                    continue;
                result.DirectionalMcShadowNeighborDisagreementsAfter +=
                    CountDirectionalMcFaceDisagreements(
                        left.RegularizedIndex0, right.RegularizedIndex0, axis);
            }
        }
        if (result.DirectionalMcShadowNeighborDisagreementsAfter >
            result.DirectionalMcShadowNeighborDisagreementsBefore)
        {
            for (int keyIndex = 0; keyIndex < ordered.Count; keyIndex++)
            {
                DirectionalMcShadowCell cell = cells[ordered[keyIndex]];
                cell.RegularizedIndex0 = cell.CombinedIndex0;
                cell.RegularizedIndex1 = cell.CombinedIndex1;
                cells[cell.CellIndex] = cell;
            }
            result.DirectionalMcShadowRegularizationDeferredPairs++;
            result.DirectionalMcShadowRegularizedCorners = 0;
            result.DirectionalMcShadowRegularizedCells = 0;
            result.DirectionalMcShadowNeighborDisagreementsAfter =
                result.DirectionalMcShadowNeighborDisagreementsBefore;
        }
    }

    private void AuditDirectionalMcShadowNeighborsBefore(
        Dictionary<int, DirectionalMcShadowCell> cells,
        MeshResult result)
    {
        foreach (DirectionalMcShadowCell left in cells.Values)
        {
            for (int axis = 0; axis < 3; axis++)
            {
                int nx = left.X + (axis == 0 ? 1 : 0);
                int ny = left.Y + (axis == 1 ? 1 : 0);
                int nz = left.Z + (axis == 2 ? 1 : 0);
                if (!cells.TryGetValue(
                        CellIndex(nx, ny, nz),
                        out DirectionalMcShadowCell right))
                    continue;
                result.DirectionalMcShadowNeighborFaceComparisons++;
                int disagreements = CountDirectionalMcFaceDisagreements(
                    left.RegularizedIndex0,
                    right.RegularizedIndex0,
                    axis);
                result.DirectionalMcShadowNeighborDisagreementsBefore +=
                    disagreements;
            }
        }
    }

    private void UpdateAndApplyPersistentDmcFaceDecisions(
        Dictionary<int, DirectionalMcShadowCell> cells,
        bool completeEvaluation,
        MeshResult result)
    {
        List<int> ordered = new List<int>(cells.Keys);
        ordered.Sort();
        HashSet<DmcFaceKey> seenFaces = new HashSet<DmcFaceKey>();
        HashSet<int> changedCells = new HashSet<int>();
        List<DmcFaceApplication> applications =
            new List<DmcFaceApplication>();

        for (int cellPosition = 0; cellPosition < ordered.Count; cellPosition++)
        {
            DirectionalMcShadowCell left = cells[ordered[cellPosition]];
            for (int axis = 0; axis < 3; axis++)
            {
                int nx = left.X + (axis == 0 ? 1 : 0);
                int ny = left.Y + (axis == 1 ? 1 : 0);
                int nz = left.Z + (axis == 2 ? 1 : 0);
                if (!cells.TryGetValue(
                        CellIndex(nx, ny, nz),
                        out DirectionalMcShadowCell right))
                    continue;

                // Phase one intentionally excludes creases, parallel layers and
                // dual-surface cells.  They require SurfaceId/global-edge work,
                // not a broader face-completion rule.
                if (left.SurfaceCount != 1 || right.SurfaceCount != 1)
                {
                    result.DirectionalMcShadowRegularizationDeferredPairs++;
                    continue;
                }
                float signedNormalDot = Vector3.Dot(left.Normal0, right.Normal0);
                if (Mathf.Abs(signedNormalDot) <
                    DmcFaceDecisionMinimumNormalDot)
                {
                    result.DirectionalMcShadowRegularizationDeferredPairs++;
                    continue;
                }
                Vector3 alignedRightNormal =
                    signedNormalDot < 0f ? -right.Normal0 : right.Normal0;
                Vector3 centroidDelta = right.Centroid0 - left.Centroid0;
                float maximumPlaneOffset =
                    _voxelSize * DmcFaceDecisionMaximumPlaneOffsetVoxels;
                if (Mathf.Abs(Vector3.Dot(centroidDelta, left.Normal0)) >
                        maximumPlaneOffset ||
                    Mathf.Abs(Vector3.Dot(centroidDelta, alignedRightNormal)) >
                        maximumPlaneOffset)
                {
                    result.DirectionalMcShadowRegularizationDeferredPairs++;
                    continue;
                }

                result.DirectionalMcShadowFaceDecisionCandidates++;
                byte knownMask = 0;
                byte insideMask = 0;
                byte conflictMask = 0;
                for (int faceCorner = 0; faceCorner < 4; faceCorner++)
                {
                    int leftCorner =
                        DirectionalMcFaceCorner(axis, true, faceCorner);
                    int rightCorner =
                        DirectionalMcFaceCorner(axis, false, faceCorner);
                    bool leftKnown =
                        (left.KnownCornerMask0 & (1 << leftCorner)) != 0;
                    bool rightKnown =
                        (right.KnownCornerMask0 & (1 << rightCorner)) != 0;
                    if (!leftKnown && !rightKnown)
                        continue;

                    bool leftInside =
                        (left.CombinedIndex0 & (1 << leftCorner)) != 0;
                    bool rightInside =
                        (right.CombinedIndex0 & (1 << rightCorner)) != 0;
                    byte faceBit = (byte)(1 << faceCorner);
                    knownMask |= faceBit;
                    if (leftKnown && rightKnown &&
                        leftInside != rightInside)
                    {
                        conflictMask |= faceBit;
                        continue;
                    }
                    bool resolvedInside =
                        leftKnown ? leftInside : rightInside;
                    if (resolvedInside)
                        insideMask |= faceBit;
                }

                if (knownMask == 0)
                {
                    result.DirectionalMcShadowRegularizationDeferredPairs++;
                    continue;
                }

                DmcFaceKey key =
                    new DmcFaceKey(left.CellIndex, (byte)axis);
                seenFaces.Add(key);
                bool existed = _dmcFaceDecisions.TryGetValue(
                    key, out DmcFaceDecision faceDecision);
                if (!existed)
                {
                    faceDecision = new DmcFaceDecision { Key = key };
                    _dmcFaceDecisions.Add(key, faceDecision);
                }
                bool sameDecision = existed &&
                    faceDecision.KnownMask == knownMask &&
                    faceDecision.InsideMask == insideMask &&
                    faceDecision.ConflictMask == conflictMask;
                bool newEvidenceBatch =
                    !existed ||
                    faceDecision.LastUpdatedBatch != _batchSequence;
                faceDecision.KnownMask = knownMask;
                faceDecision.InsideMask = insideMask;
                faceDecision.ConflictMask = conflictMask;
                faceDecision.TransitionMask =
                    DmcFaceTransitionMask(knownMask, insideMask);
                Vector3 faceNormal = left.Normal0 + alignedRightNormal;
                faceDecision.Normal =
                    faceNormal.sqrMagnitude > 0.00000001f
                        ? faceNormal.normalized
                        : left.Normal0;
                faceDecision.Centroid =
                    (left.Centroid0 + right.Centroid0) * 0.5f;
                faceDecision.LastUpdatedBatch = _batchSequence;
                faceDecision.Revision = _dmcDecisionRevision;
                faceDecision.MissingBatches = 0;
                if (!sameDecision)
                    faceDecision.StableBatches = 1;
                else if (newEvidenceBatch)
                    faceDecision.StableBatches++;

                if (conflictMask != 0)
                {
                    // Both sides have reliable but incompatible signs on this
                    // physical face.  Do not overwrite either observation and
                    // do not let the visible shadow bridge across it.  The mesh
                    // builder will trim only triangles that touch this face.
                    left.DeferredFaceMask0 |= DmcFaceMaskBit(axis, true);
                    right.DeferredFaceMask0 |= DmcFaceMaskBit(axis, false);
                    cells[left.CellIndex] = left;
                    cells[right.CellIndex] = right;
                    result.DirectionalMcShadowFaceDecisionConflicts++;
                    result.DirectionalMcShadowRegularizationDeferredPairs++;
                    continue;
                }
                if (faceDecision.StableBatches <
                    DmcFaceDecisionMinimumStableBatches)
                {
                    result.DirectionalMcShadowRegularizationDeferredPairs++;
                    continue;
                }

                result.DirectionalMcShadowFaceDecisionStable++;
                applications.Add(new DmcFaceApplication(
                    key, left.CellIndex, right.CellIndex, (byte)axis));
            }
        }

        // Apply only after every face has been evaluated from raw observations.
        // A synthetic completion made on one face therefore cannot cascade and
        // become "evidence" for another face in the same rebuild.
        for (int applicationIndex = 0;
             applicationIndex < applications.Count;
             applicationIndex++)
        {
            DmcFaceApplication application = applications[applicationIndex];
            if (!_dmcFaceDecisions.TryGetValue(
                    application.Key, out DmcFaceDecision faceDecision) ||
                !cells.TryGetValue(
                    application.LeftCellIndex,
                    out DirectionalMcShadowCell left) ||
                !cells.TryGetValue(
                    application.RightCellIndex,
                    out DirectionalMcShadowCell right))
                continue;
            int filled = ApplyDmcFaceDecision(
                faceDecision, application.Axis, ref left, ref right,
                out bool leftChanged, out bool rightChanged);
            if (filled <= 0)
                continue;

            result.DirectionalMcShadowFaceDecisionAppliedPairs++;
            result.DirectionalMcShadowFaceDecisionFilledCorners += filled;
            if (leftChanged)
                changedCells.Add(left.CellIndex);
            if (rightChanged)
                changedCells.Add(right.CellIndex);
            cells[left.CellIndex] = left;
            cells[right.CellIndex] = right;
            if (_dmcCellDecisions.TryGetValue(
                    left.CellIndex, out DmcCellDecision leftDecision))
            {
                leftDecision.Index0 = left.RegularizedIndex0;
                leftDecision.KnownCornerMask0 = left.KnownCornerMask0;
            }
            if (_dmcCellDecisions.TryGetValue(
                    right.CellIndex, out DmcCellDecision rightDecision))
            {
                rightDecision.Index0 = right.RegularizedIndex0;
                rightDecision.KnownCornerMask0 = right.KnownCornerMask0;
            }
        }

        result.DirectionalMcShadowFaceDecisionChangedCells =
            changedCells.Count;
        result.DirectionalMcShadowRegularizedCorners =
            result.DirectionalMcShadowFaceDecisionFilledCorners;
        result.DirectionalMcShadowRegularizedCells = changedCells.Count;
        if (completeEvaluation)
            RetireMissingDmcFaceDecisions(seenFaces, result);
        result.DirectionalMcShadowFaceDecisionPersistent =
            _dmcFaceDecisions.Count;
    }

    private static int ApplyDmcFaceDecision(
        DmcFaceDecision decision,
        int axis,
        ref DirectionalMcShadowCell left,
        ref DirectionalMcShadowCell right,
        out bool leftChanged,
        out bool rightChanged)
    {
        leftChanged = false;
        rightChanged = false;
        byte leftIndex = left.RegularizedIndex0;
        byte rightIndex = right.RegularizedIndex0;
        byte leftKnownMask = left.KnownCornerMask0;
        byte rightKnownMask = right.KnownCornerMask0;
        int leftFilled = 0;
        int rightFilled = 0;

        for (int faceCorner = 0; faceCorner < 4; faceCorner++)
        {
            byte faceBit = (byte)(1 << faceCorner);
            if ((decision.KnownMask & faceBit) == 0 ||
                (decision.ConflictMask & faceBit) != 0)
                continue;
            bool inside = (decision.InsideMask & faceBit) != 0;
            int leftCorner =
                DirectionalMcFaceCorner(axis, true, faceCorner);
            int rightCorner =
                DirectionalMcFaceCorner(axis, false, faceCorner);
            if ((leftKnownMask & (1 << leftCorner)) == 0)
            {
                leftIndex = SetDirectionalMcCorner(
                    leftIndex, leftCorner, inside);
                leftKnownMask |= (byte)(1 << leftCorner);
                leftFilled++;
            }
            if ((rightKnownMask & (1 << rightCorner)) == 0)
            {
                rightIndex = SetDirectionalMcCorner(
                    rightIndex, rightCorner, inside);
                rightKnownMask |= (byte)(1 << rightCorner);
                rightFilled++;
            }
        }

        // Completion may close an uncertain boundary, but phase one must not
        // turn a cell wholly inside/outside and fabricate or erase a sheet.
        if (leftFilled > 0 &&
            leftIndex != 0 && leftIndex != byte.MaxValue)
        {
            left.RegularizedIndex0 = leftIndex;
            left.KnownCornerMask0 = leftKnownMask;
            leftChanged = true;
        }
        else
        {
            leftFilled = 0;
        }
        if (rightFilled > 0 &&
            rightIndex != 0 && rightIndex != byte.MaxValue)
        {
            right.RegularizedIndex0 = rightIndex;
            right.KnownCornerMask0 = rightKnownMask;
            rightChanged = true;
        }
        else
        {
            rightFilled = 0;
        }
        return leftFilled + rightFilled;
    }

    private static byte DmcFaceTransitionMask(
        byte knownMask,
        byte insideMask)
    {
        // Positive/negative face corner arrays use the same 2x2 layout:
        // 0--1
        // |  |
        // 2--3
        // Traverse 0,1,3,2 so the signature is canonical on both cells.
        int[] cycle = { 0, 1, 3, 2 };
        byte transitions = 0;
        for (int edge = 0; edge < 4; edge++)
        {
            int a = cycle[edge];
            int b = cycle[(edge + 1) & 3];
            if ((knownMask & (1 << a)) == 0 ||
                (knownMask & (1 << b)) == 0)
                continue;
            bool aInside = (insideMask & (1 << a)) != 0;
            bool bInside = (insideMask & (1 << b)) != 0;
            if (aInside != bInside)
                transitions |= (byte)(1 << edge);
        }
        return transitions;
    }

    private void RetireMissingDmcFaceDecisions(
        HashSet<DmcFaceKey> seenFaces,
        MeshResult result)
    {
        List<DmcFaceKey> retire = null;
        foreach (KeyValuePair<DmcFaceKey, DmcFaceDecision> pair
                 in _dmcFaceDecisions)
        {
            if (seenFaces.Contains(pair.Key))
                continue;
            if (pair.Value.LastMissingBatch == _batchSequence)
                continue;
            pair.Value.LastMissingBatch = _batchSequence;
            pair.Value.MissingBatches++;
            if (pair.Value.MissingBatches <= DmcFaceDecisionLeaseBatches)
                continue;
            if (retire == null)
                retire = new List<DmcFaceKey>();
            retire.Add(pair.Key);
        }
        if (retire == null)
            return;
        for (int i = 0; i < retire.Count; i++)
            _dmcFaceDecisions.Remove(retire[i]);
        result.DirectionalMcShadowFaceDecisionRetired = retire.Count;
    }

    private void AuditDirectionalMcShadowNeighborsAfter(
        Dictionary<int, DirectionalMcShadowCell> cells,
        MeshResult result)
    {
        foreach (DirectionalMcShadowCell left in cells.Values)
        {
            for (int axis = 0; axis < 3; axis++)
            {
                int nx = left.X + (axis == 0 ? 1 : 0);
                int ny = left.Y + (axis == 1 ? 1 : 0);
                int nz = left.Z + (axis == 2 ? 1 : 0);
                if (!cells.TryGetValue(
                        CellIndex(nx, ny, nz),
                        out DirectionalMcShadowCell right))
                    continue;
                result.DirectionalMcShadowNeighborDisagreementsAfter +=
                    CountDirectionalMcFaceDisagreements(
                        left.RegularizedIndex0,
                        right.RegularizedIndex0,
                        axis);
            }
        }
    }

    private static int CountDirectionalMcFaceDisagreements(
        byte left,
        byte right,
        int axis)
    {
        int disagreements = 0;
        for (int faceCorner = 0; faceCorner < 4; faceCorner++)
        {
            int leftCorner = DirectionalMcFaceCorner(axis, true, faceCorner);
            int rightCorner = DirectionalMcFaceCorner(axis, false, faceCorner);
            bool leftInside = (left & (1 << leftCorner)) != 0;
            bool rightInside = (right & (1 << rightCorner)) != 0;
            if (leftInside != rightInside)
                disagreements++;
        }
        return disagreements;
    }

    private static int DirectionalMcFaceCorner(
        int axis,
        bool positiveFace,
        int faceCorner)
    {
        return positiveFace
            ? PositiveFaceCorners[axis, faceCorner]
            : NegativeFaceCorners[axis, faceCorner];
    }

    private static byte DmcFaceMaskBit(int axis, bool positiveFace)
    {
        return (byte)(1 << (axis * 2 + (positiveFace ? 1 : 0)));
    }

    private static byte SetDirectionalMcCorner(
        byte index,
        int corner,
        bool inside)
    {
        return inside
            ? (byte)(index | (1 << corner))
            : (byte)(index & ~(1 << corner));
    }

    private static int PopCount(ushort value)
    {
        int count = 0;
        while (value != 0)
        {
            value &= (ushort)(value - 1);
            count++;
        }
        return count;
    }

    private bool TryBuildComposedHypothesis(
        int layerChannel,
        int x,
        int y,
        int z,
        float minimumWeight,
        Vector3[] cornerPositions,
        float[] values,
        float[] weights,
        out ComposedHypothesis hypothesis)
    {
        hypothesis = default;
        bool anyPositive = false;
        bool anyNegative = false;
        int validCorners = 0;
        Vector3 centroid = Vector3.zero;
        int centroidCount = 0;
        float support = 0f;
        for (int corner = 0; corner < 8; corner++)
        {
            int vx = x + CornerX[corner];
            int vy = y + CornerY[corner];
            int vz = z + CornerZ[corner];
            if (!TryReadVoxel(layerChannel, vx, vy, vz, out float value, out float weight))
            {
                values[corner] = 1f;
                weights[corner] = 0f;
                continue;
            }
            values[corner] = value;
            weights[corner] = weight;
            if (weight < minimumWeight)
                continue;
            validCorners++;
            anyPositive |= value >= 0f;
            anyNegative |= value < 0f;
            support += Mathf.Min(weight, minimumWeight * 4f);
            if (Mathf.Abs(value) <= 0.5f)
            {
                centroid += cornerPositions[corner];
                centroidCount++;
            }
        }
        if (validCorners < 4 || !anyPositive || !anyNegative)
            return false;
        Vector3 crossingCentroid = Vector3.zero;
        int crossingCount = 0;
        for (int edge = 0; edge < 12; edge++)
        {
            int a = CubeEdges[edge, 0];
            int b = CubeEdges[edge, 1];
            if (weights[a] < minimumWeight || weights[b] < minimumWeight ||
                (values[a] < 0f) == (values[b] < 0f))
                continue;
            float denominator = values[a] - values[b];
            float t = Mathf.Abs(denominator) > 0.000001f
                ? Mathf.Clamp01(values[a] / denominator)
                : 0.5f;
            crossingCentroid += Vector3.Lerp(cornerPositions[a], cornerPositions[b], t);
            crossingCount++;
        }
        if (crossingCount > 0)
        {
            centroid = crossingCentroid;
            centroidCount = crossingCount;
        }
        if (centroidCount == 0)
        {
            for (int corner = 0; corner < 8; corner++)
            {
                if (weights[corner] < minimumWeight)
                    continue;
                centroid += cornerPositions[corner];
                centroidCount++;
            }
        }

        float gx = SupportedAxisDifference(
            values, weights, minimumWeight,
            0, 1, 3, 2, 4, 5, 7, 6);
        float gy = SupportedAxisDifference(
            values, weights, minimumWeight,
            0, 3, 1, 2, 4, 7, 5, 6);
        float gz = SupportedAxisDifference(
            values, weights, minimumWeight,
            0, 4, 1, 5, 2, 6, 3, 7);
        Vector3 gradient = new Vector3(gx, gy, gz);
        int direction = layerChannel % DirectionCount;
        Vector3 observedNormalSum = Vector3.zero;
        float observedNormalWeight = 0f;
        for (int corner = 0; corner < 8; corner++)
        {
            int vx = x + CornerX[corner];
            int vy = y + CornerY[corner];
            int vz = z + CornerZ[corner];
            if (!TryReadVoxelNormal(
                    layerChannel, vx, vy, vz,
                    out Vector3 observedNormal, out float observedWeight) ||
                observedWeight < minimumWeight)
                continue;
            float boundedWeight = Mathf.Min(observedWeight, minimumWeight * 4f);
            observedNormalSum += observedNormal * boundedWeight;
            observedNormalWeight += boundedWeight;
        }
        bool usedObservedNormal =
            observedNormalWeight > 0f && observedNormalSum.sqrMagnitude > 0.00000001f;
        Vector3 resolvedNormal;
        if (usedObservedNormal)
        {
            resolvedNormal = observedNormalSum.normalized;
            if (gradient.sqrMagnitude > 0.00000001f)
            {
                Vector3 gradientNormal = gradient.normalized;
                if (Vector3.Dot(gradientNormal, resolvedNormal) > 0.5f)
                    resolvedNormal = (resolvedNormal * 0.8f + gradientNormal * 0.2f).normalized;
            }
        }
        else
        {
            resolvedNormal = gradient.sqrMagnitude > 0.00000001f
                ? gradient.normalized
                : DirectionVectors[direction];
        }
        hypothesis = new ComposedHypothesis
        {
            LayerChannel = layerChannel,
            Direction = direction,
            Normal = resolvedNormal,
            Centroid = centroid / Mathf.Max(1, centroidCount),
            Support = support,
            UsedObservedNormal = usedObservedNormal
        };
        return true;
    }

    private static float SupportedAxisDifference(
        float[] values,
        float[] weights,
        float minimumWeight,
        int a0, int b0,
        int a1, int b1,
        int a2, int b2,
        int a3, int b3)
    {
        float sum = 0f;
        int count = 0;
        AddSupportedPairDifference(values, weights, minimumWeight, a0, b0, ref sum, ref count);
        AddSupportedPairDifference(values, weights, minimumWeight, a1, b1, ref sum, ref count);
        AddSupportedPairDifference(values, weights, minimumWeight, a2, b2, ref sum, ref count);
        AddSupportedPairDifference(values, weights, minimumWeight, a3, b3, ref sum, ref count);
        return count > 0 ? sum / count : 0f;
    }

    private static void AddSupportedPairDifference(
        float[] values,
        float[] weights,
        float minimumWeight,
        int a,
        int b,
        ref float sum,
        ref int count)
    {
        if (weights[a] < minimumWeight || weights[b] < minimumWeight)
            return;
        sum += values[b] - values[a];
        count++;
    }

    private bool TryCombineCluster(
        ComposedCluster cluster,
        int x,
        int y,
        int z,
        float minimumWeight,
        float[] combinedValues,
        float[] combinedWeights)
    {
        bool anyPositive = false;
        bool anyNegative = false;
        int validCorners = 0;
        for (int corner = 0; corner < 8; corner++)
        {
            int vx = x + CornerX[corner];
            int vy = y + CornerY[corner];
            int vz = z + CornerZ[corner];
            float numerator = 0f;
            float denominator = 0f;
            for (int memberIndex = 0; memberIndex < LayerCount; memberIndex++)
            {
                if ((cluster.LayerMask & (1 << memberIndex)) == 0)
                    continue;
                int direction = memberIndex % DirectionCount;
                if (!TryReadVoxel(
                        memberIndex, vx, vy, vz,
                        out float value, out float weight) ||
                    weight < minimumWeight)
                    continue;
                float alignment = Mathf.Max(
                    0.15f,
                    Mathf.Abs(Vector3.Dot(
                        cluster.Normal, DirectionVectors[direction])));
                float effectiveWeight = weight * alignment;
                numerator += value * effectiveWeight;
                denominator += effectiveWeight;
            }
            if (denominator <= 0f)
            {
                combinedValues[corner] = 1f;
                combinedWeights[corner] = 0f;
                continue;
            }
            combinedValues[corner] = numerator / denominator;
            combinedWeights[corner] = denominator;
            if (denominator < minimumWeight)
                continue;
            validCorners++;
            anyPositive |= combinedValues[corner] >= 0f;
            anyNegative |= combinedValues[corner] < 0f;
        }
        return validCorners >= 4 && anyPositive && anyNegative;
    }

    private void PolygonizeComposedTetrahedron(
        int direction,
        int surface,
        Vector3 surfaceNormal,
        Vector3 surfaceCentroid,
        float support,
        int a,
        int b,
        int c,
        int d,
        int[] indices,
        float[] values,
        Vector3[] positions,
        int[] tetra,
        int[] inside,
        int[] outside,
        float mergeRatio,
        float parallelNormalDot,
        Dictionary<ulong, ComposedEdgePair> edgeVertices,
        List<SupportedTriangle> triangles,
        MeshResult result)
    {
        tetra[0] = a;
        tetra[1] = b;
        tetra[2] = c;
        tetra[3] = d;
        int insideCount = 0;
        int outsideCount = 0;
        for (int i = 0; i < 4; i++)
        {
            if (values[tetra[i]] < 0f)
                inside[insideCount++] = tetra[i];
            else
                outside[outsideCount++] = tetra[i];
        }
        if (insideCount == 0 || insideCount == 4)
            return;
        Vector3 outward = DirectionVectors[direction];
        if (insideCount == 1 || insideCount == 3)
        {
            bool invert = insideCount == 3;
            int pivot = invert ? outside[0] : inside[0];
            int q0 = invert ? inside[0] : outside[0];
            int q1 = invert ? inside[1] : outside[1];
            int q2 = invert ? inside[2] : outside[2];
            int v0 = ComposedEdgeVertex(
                surface, surfaceNormal, surfaceCentroid, support,
                pivot, q0, indices, values, positions,
                mergeRatio, parallelNormalDot, edgeVertices, result);
            int v1 = ComposedEdgeVertex(
                surface, surfaceNormal, surfaceCentroid, support,
                pivot, q1, indices, values, positions,
                mergeRatio, parallelNormalDot, edgeVertices, result);
            int v2 = ComposedEdgeVertex(
                surface, surfaceNormal, surfaceCentroid, support,
                pivot, q2, indices, values, positions,
                mergeRatio, parallelNormalDot, edgeVertices, result);
            AddSupportedTriangle(
                direction, surface, v0, v1, v2, support, outward, triangles, result);
            return;
        }
        int i0 = inside[0];
        int i1 = inside[1];
        int o0 = outside[0];
        int o1 = outside[1];
        int v00 = ComposedEdgeVertex(
            surface, surfaceNormal, surfaceCentroid, support,
            i0, o0, indices, values, positions,
            mergeRatio, parallelNormalDot, edgeVertices, result);
        int v01 = ComposedEdgeVertex(
            surface, surfaceNormal, surfaceCentroid, support,
            i0, o1, indices, values, positions,
            mergeRatio, parallelNormalDot, edgeVertices, result);
        int v11 = ComposedEdgeVertex(
            surface, surfaceNormal, surfaceCentroid, support,
            i1, o1, indices, values, positions,
            mergeRatio, parallelNormalDot, edgeVertices, result);
        int v10 = ComposedEdgeVertex(
            surface, surfaceNormal, surfaceCentroid, support,
            i1, o0, indices, values, positions,
            mergeRatio, parallelNormalDot, edgeVertices, result);
        AddSupportedTriangle(
            direction, surface, v00, v01, v11, support, outward, triangles, result);
        AddSupportedTriangle(
            direction, surface, v00, v11, v10, support, outward, triangles, result);
    }

    private int ComposedEdgeVertex(
        int surface,
        Vector3 surfaceNormal,
        Vector3 surfaceCentroid,
        float support,
        int localA,
        int localB,
        int[] indices,
        float[] values,
        Vector3[] positions,
        float mergeRatio,
        float parallelNormalDot,
        Dictionary<ulong, ComposedEdgePair> edgeVertices,
        MeshResult result)
    {
        int indexA = indices[localA];
        int indexB = indices[localB];
        float valueA = values[localA];
        float valueB = values[localB];
        Vector3 positionA = positions[localA];
        Vector3 positionB = positions[localB];
        float denominator = valueA - valueB;
        float t = Mathf.Abs(denominator) > 0.000001f
            ? Mathf.Clamp01(valueA / denominator)
            : 0.5f;
        if (indexB < indexA)
        {
            int indexSwap = indexA;
            indexA = indexB;
            indexB = indexSwap;
            float valueSwap = valueA;
            valueA = valueB;
            valueB = valueSwap;
            Vector3 positionSwap = positionA;
            positionA = positionB;
            positionB = positionSwap;
            t = 1f - t;
        }
        ulong key = ((ulong)(uint)indexA << 32) | (uint)indexB;
        edgeVertices.TryGetValue(key, out ComposedEdgePair crossings);
        Vector3 crossingPosition = Vector3.Lerp(positionA, positionB, t);

        if (surface < 0)
        {
            if (crossings.Count >= 1 && Mathf.Abs(crossings.T0 - t) <= mergeRatio)
            {
                result.EdgeCrossingsMerged++;
                return crossings.Vertex0;
            }
            if (crossings.Count >= 2 && Mathf.Abs(crossings.T1 - t) <= mergeRatio)
            {
                result.EdgeCrossingsMerged++;
                return crossings.Vertex1;
            }
        }

        if (surface >= 0 && crossings.Count >= 1 && crossings.Surface0 == surface)
        {
            if (Mathf.Abs(crossings.T0 - t) > mergeRatio)
            {
                result.SurfaceContinuityEdgeDisagreements++;
            }
            else
            {
                result.EdgeCrossingsMerged++;
                result.SurfaceConsistentEdgeMerges++;
                float previousSupport = crossings.Support0;
                float combinedSupport = Mathf.Max(0.0001f, previousSupport + support);
                result.Vertices[crossings.Vertex0] =
                    (result.Vertices[crossings.Vertex0] * previousSupport + crossingPosition * support) /
                    combinedSupport;
                crossings.T0 = (crossings.T0 * previousSupport + t * support) / combinedSupport;
                crossings.Normal0 =
                    (crossings.Normal0 * previousSupport + surfaceNormal * support).normalized;
                crossings.Centroid0 =
                    (crossings.Centroid0 * previousSupport + surfaceCentroid * support) /
                    combinedSupport;
                crossings.Support0 = combinedSupport;
                edgeVertices[key] = crossings;
                return crossings.Vertex0;
            }
        }
        if (surface >= 0 && crossings.Count >= 2 && crossings.Surface1 == surface)
        {
            if (Mathf.Abs(crossings.T1 - t) > mergeRatio)
            {
                result.SurfaceContinuityEdgeDisagreements++;
            }
            else
            {
                result.EdgeCrossingsMerged++;
                result.SurfaceConsistentEdgeMerges++;
                float previousSupport = crossings.Support1;
                float combinedSupport = Mathf.Max(0.0001f, previousSupport + support);
                result.Vertices[crossings.Vertex1] =
                    (result.Vertices[crossings.Vertex1] * previousSupport + crossingPosition * support) /
                    combinedSupport;
                crossings.T1 = (crossings.T1 * previousSupport + t * support) / combinedSupport;
                crossings.Normal1 =
                    (crossings.Normal1 * previousSupport + surfaceNormal * support).normalized;
                crossings.Centroid1 =
                    (crossings.Centroid1 * previousSupport + surfaceCentroid * support) /
                    combinedSupport;
                crossings.Support1 = combinedSupport;
                edgeVertices[key] = crossings;
                return crossings.Vertex1;
            }
        }

        // Establish a physical owner on the grid edge before triangles are
        // committed.  Face-only component linking cannot connect every pair
        // of cells incident on an edge; without this hand-off, two roots that
        // describe the same crossing publish coincident sheets.  Only a tight
        // crossing, normal and bidirectional plane agreement may share the
        // owner.  A small real step therefore remains two crossings.
        bool firstCandidate = false;
        bool firstIdentityMerge =
            PhysicalIdentityEdgeOwnerProductionEnabled &&
            surface >= 0 && crossings.Count >= 1 &&
            crossings.Surface0 != surface &&
            TryMergePhysicalIdentityEdgeCrossing(
                crossings.T0, t,
                crossings.Normal0, crossings.Centroid0,
                surfaceNormal, surfaceCentroid,
                mergeRatio, out firstCandidate);
        if (firstIdentityMerge)
        {
            result.PhysicalSurfaceIdentityEdgeCandidates++;
            result.PhysicalSurfaceIdentityEdgeMerges++;
            result.EdgeCrossingsMerged++;
            result.SurfaceConsistentEdgeMerges++;
            float previousSupport = crossings.Support0;
            float combinedSupport = Mathf.Max(0.0001f, previousSupport + support);
            result.Vertices[crossings.Vertex0] =
                (result.Vertices[crossings.Vertex0] * previousSupport +
                 crossingPosition * support) / combinedSupport;
            crossings.T0 =
                (crossings.T0 * previousSupport + t * support) / combinedSupport;
            Vector3 alignedNormal =
                Vector3.Dot(crossings.Normal0, surfaceNormal) < 0f
                    ? -surfaceNormal
                    : surfaceNormal;
            crossings.Normal0 =
                (crossings.Normal0 * previousSupport + alignedNormal * support).normalized;
            crossings.Centroid0 =
                (crossings.Centroid0 * previousSupport + surfaceCentroid * support) /
                combinedSupport;
            crossings.Support0 = combinedSupport;
            edgeVertices[key] = crossings;
            return crossings.Vertex0;
        }
        else if (firstCandidate)
        {
            result.PhysicalSurfaceIdentityEdgeCandidates++;
            result.PhysicalSurfaceIdentityEdgeRejected++;
        }
        bool secondCandidate = false;
        bool secondIdentityMerge =
            PhysicalIdentityEdgeOwnerProductionEnabled &&
            surface >= 0 && crossings.Count >= 2 &&
            crossings.Surface1 != surface &&
            TryMergePhysicalIdentityEdgeCrossing(
                crossings.T1, t,
                crossings.Normal1, crossings.Centroid1,
                surfaceNormal, surfaceCentroid,
                mergeRatio, out secondCandidate);
        if (secondIdentityMerge)
        {
            result.PhysicalSurfaceIdentityEdgeCandidates++;
            result.PhysicalSurfaceIdentityEdgeMerges++;
            result.EdgeCrossingsMerged++;
            result.SurfaceConsistentEdgeMerges++;
            float previousSupport = crossings.Support1;
            float combinedSupport = Mathf.Max(0.0001f, previousSupport + support);
            result.Vertices[crossings.Vertex1] =
                (result.Vertices[crossings.Vertex1] * previousSupport +
                 crossingPosition * support) / combinedSupport;
            crossings.T1 =
                (crossings.T1 * previousSupport + t * support) / combinedSupport;
            Vector3 alignedNormal =
                Vector3.Dot(crossings.Normal1, surfaceNormal) < 0f
                    ? -surfaceNormal
                    : surfaceNormal;
            crossings.Normal1 =
                (crossings.Normal1 * previousSupport + alignedNormal * support).normalized;
            crossings.Centroid1 =
                (crossings.Centroid1 * previousSupport + surfaceCentroid * support) /
                combinedSupport;
            crossings.Support1 = combinedSupport;
            edgeVertices[key] = crossings;
            return crossings.Vertex1;
        }
        else if (secondCandidate)
        {
            result.PhysicalSurfaceIdentityEdgeCandidates++;
            result.PhysicalSurfaceIdentityEdgeRejected++;
        }

        if (surface >= 0 && crossings.Count >= 1 && Mathf.Abs(crossings.T0 - t) <= mergeRatio)
        {
            float normalDot = Mathf.Abs(Vector3.Dot(crossings.Normal0, surfaceNormal));
            if (normalDot < parallelNormalDot)
            {
                result.EdgeCrossingsMerged++;
                result.CreaseJunctionEdgeMerges++;
                return crossings.Vertex0;
            }
            result.CrossSurfaceNearCrossingsPreserved++;
        }
        if (surface >= 0 && crossings.Count >= 2 && Mathf.Abs(crossings.T1 - t) <= mergeRatio)
        {
            float normalDot = Mathf.Abs(Vector3.Dot(crossings.Normal1, surfaceNormal));
            if (normalDot < parallelNormalDot)
            {
                result.EdgeCrossingsMerged++;
                result.CreaseJunctionEdgeMerges++;
                return crossings.Vertex1;
            }
            result.CrossSurfaceNearCrossingsPreserved++;
        }
        if (crossings.Count >= 2)
        {
            result.EdgeCrossingOverflowDropped++;
            return -1;
        }
        int vertex = result.Vertices.Count;
        result.Vertices.Add(crossingPosition);
        if (crossings.Count == 0)
        {
            crossings.Count = 1;
            crossings.T0 = t;
            crossings.Vertex0 = vertex;
            crossings.Surface0 = surface;
            crossings.Normal0 = surfaceNormal;
            crossings.Centroid0 = surfaceCentroid;
            crossings.Support0 = support;
        }
        else if (t < crossings.T0)
        {
            crossings.Count = 2;
            crossings.T1 = crossings.T0;
            crossings.Vertex1 = crossings.Vertex0;
            crossings.Surface1 = crossings.Surface0;
            crossings.Normal1 = crossings.Normal0;
            crossings.Centroid1 = crossings.Centroid0;
            crossings.Support1 = crossings.Support0;
            crossings.T0 = t;
            crossings.Vertex0 = vertex;
            crossings.Surface0 = surface;
            crossings.Normal0 = surfaceNormal;
            crossings.Centroid0 = surfaceCentroid;
            crossings.Support0 = support;
        }
        else
        {
            crossings.Count = 2;
            crossings.T1 = t;
            crossings.Vertex1 = vertex;
            crossings.Surface1 = surface;
            crossings.Normal1 = surfaceNormal;
            crossings.Centroid1 = surfaceCentroid;
            crossings.Support1 = support;
        }
        edgeVertices[key] = crossings;
        return vertex;
    }

    private bool TryMergePhysicalIdentityEdgeCrossing(
        float existingT,
        float candidateT,
        Vector3 existingNormal,
        Vector3 existingCentroid,
        Vector3 candidateNormal,
        Vector3 candidateCentroid,
        float ordinaryMergeRatio,
        out bool candidate)
    {
        candidate = false;
        float crossingDelta = Mathf.Abs(existingT - candidateT);
        if (crossingDelta > ordinaryMergeRatio)
            return false;
        float signedNormalDot = Vector3.Dot(existingNormal, candidateNormal);
        if (Mathf.Abs(signedNormalDot) < PhysicalIdentityEdgeMinimumNormalDot)
            return false;
        candidate = true;
        if (crossingDelta > PhysicalIdentityEdgeMaximumCrossingDeltaVoxels)
            return false;
        Vector3 alignedCandidateNormal = signedNormalDot < 0f
            ? -candidateNormal
            : candidateNormal;
        Vector3 delta = candidateCentroid - existingCentroid;
        float maximumPlaneOffset =
            _voxelSize * PhysicalIdentityEdgeMaximumPlaneOffsetVoxels;
        if (Mathf.Abs(Vector3.Dot(delta, existingNormal)) > maximumPlaneOffset ||
            Mathf.Abs(Vector3.Dot(delta, alignedCandidateNormal)) > maximumPlaneOffset)
            return false;
        return true;
    }

    private static void AddSupportedTriangle(
        int direction,
        int surface,
        int a,
        int b,
        int c,
        float support,
        Vector3 outward,
        List<SupportedTriangle> triangles,
        MeshResult result)
    {
        if (a < 0 || b < 0 || c < 0 || a == b || b == c || c == a)
            return;
        Vector3 cross = Vector3.Cross(
            result.Vertices[b] - result.Vertices[a],
            result.Vertices[c] - result.Vertices[a]);
        if (cross.sqrMagnitude <= 0.0000000001f)
            return;
        if (Vector3.Dot(cross, outward) < 0f)
        {
            int swap = b;
            b = c;
            c = swap;
        }
        triangles.Add(new SupportedTriangle
        {
            Direction = direction,
            Surface = surface,
            A = a,
            B = b,
            C = c,
            Support = support
        });
    }

    private static void CommitSupportedManifoldTriangles(
        List<SupportedTriangle> triangles,
        MeshResult result)
    {
        triangles.Sort((left, right) => right.Support.CompareTo(left.Support));
        Dictionary<ulong, int> edgeUse = new Dictionary<ulong, int>(131072);
        HashSet<TriangleKey> unique = new HashSet<TriangleKey>();
        for (int i = 0; i < triangles.Count; i++)
        {
            SupportedTriangle triangle = triangles[i];
            if (!unique.Add(new TriangleKey(
                    -1, triangle.A, triangle.B, triangle.C)))
            {
                result.DuplicateTriangles++;
                continue;
            }
            ulong ab = EdgeKey(triangle.A, triangle.B);
            ulong bc = EdgeKey(triangle.B, triangle.C);
            ulong ca = EdgeKey(triangle.C, triangle.A);
            edgeUse.TryGetValue(ab, out int abUse);
            edgeUse.TryGetValue(bc, out int bcUse);
            edgeUse.TryGetValue(ca, out int caUse);
            if (abUse >= 2 || bcUse >= 2 || caUse >= 2)
            {
                result.NonManifoldTrianglesDropped++;
                continue;
            }
            result.TrianglesByDirection[triangle.Direction].Add(triangle.A);
            result.TrianglesByDirection[triangle.Direction].Add(triangle.B);
            result.TrianglesByDirection[triangle.Direction].Add(triangle.C);
            edgeUse[ab] = abUse + 1;
            edgeUse[bc] = bcUse + 1;
            edgeUse[ca] = caUse + 1;
        }
    }

    private static ulong EdgeKey(int a, int b)
    {
        uint minimum = (uint)Mathf.Min(a, b);
        uint maximum = (uint)Mathf.Max(a, b);
        return ((ulong)minimum << 32) | maximum;
    }

    public Metrics GetMetrics()
    {
        int candidateCount = 0;
        int activeLayerCount = _canonicalSixDirectionSemantics
            ? DirectionCount
            : LayerCount;
        for (int i = 0; i < activeLayerCount; i++)
            candidateCount += _candidateCells[i].Count;
        int[] weighted = (int[])_weightedVoxelsByDirection.Clone();
        int[] triangles = new int[DirectionCount];
        if (_lastMesh != null)
        {
            for (int i = 0; i < DirectionCount; i++)
                triangles[i] = _lastMesh.TrianglesByDirection[i].Count / 3;
        }
        return new Metrics
        {
            Configured = _configured,
            CanonicalSixDirectionSemantics = _canonicalSixDirectionSemantics,
            RefinementProbeEnabled = _refinementProbeEnabled,
            HalfVoxelShadowEnabled = _halfVoxelShadowEnabled,
            AllocatedBlocks = _blocks.Count,
            AllocatedDirectionLayers = _allocatedDirectionLayers,
            WeightedVoxels = _weightedVoxels,
            CandidateCells = candidateCount,
            BatchSamples = _batchSamples,
            BatchContestedSamples = _batchContestedSamples,
            BatchSecondarySamples = _batchSecondarySamples,
            BatchCanonicalDirectionWrites = _batchCanonicalDirectionWrites,
            BatchCanonicalFanoutOne = _batchCanonicalFanoutOne,
            BatchCanonicalFanoutTwo = _batchCanonicalFanoutTwo,
            BatchCanonicalFanoutThree = _batchCanonicalFanoutThree,
            BatchCanonicalRejectedSamples = _batchCanonicalRejectedSamples,
            BatchVoxelUpdates = _batchVoxelUpdates,
            BatchOwnerMatched = _batchOwnerMatched,
            BatchOwnerChallenged = _batchOwnerChallenged,
            BatchOwnerSwitched = _batchOwnerSwitched,
            BatchRetiredDirectionVoxels = _batchRetiredDirectionVoxels,
            BatchSaturatedRollingUpdates = _batchSaturatedRollingUpdates,
            BatchMatureDepthConflictCandidates = _batchMatureDepthConflictCandidates,
            BatchMatureDepthConflictHeld = _batchMatureDepthConflictHeld,
            BatchMatureDepthCorrections = _batchMatureDepthCorrections,
            BatchMatureDepthSignFlips = _batchMatureDepthSignFlips,
            BatchMatureDepthFastCandidates = _batchMatureDepthFastCandidates,
            BatchMatureDepthFastConfirmed = _batchMatureDepthFastConfirmed,
            BatchMatureDepthFastHeld = _batchMatureDepthFastHeld,
            BatchMatureDepthFastSharedNeighbors =
                _batchMatureDepthFastSharedNeighbors,
            BatchMatureDepthFastRiskDeferred =
                _batchMatureDepthFastRiskDeferred,
            BatchMatureDepthPatchCandidates =
                _batchMatureDepthPatchCandidates,
            BatchMatureDepthPatchConfirmed =
                _batchMatureDepthPatchConfirmed,
            BatchMatureDepthPatchSharedNeighbors =
                _batchMatureDepthPatchSharedNeighbors,
            BatchMatureDepthPersistentCandidates =
                _batchMatureDepthPersistentCandidates,
            BatchMatureDepthPersistentMatched =
                _batchMatureDepthPersistentMatched,
            BatchMatureDepthPersistentCrossVoxelMatched =
                _batchMatureDepthPersistentCrossVoxelMatched,
            BatchMatureDepthPersistentConfirmed =
                _batchMatureDepthPersistentConfirmed,
            BatchMatureDepthPersistentRiskConfirmed =
                _batchMatureDepthPersistentRiskConfirmed,
            PersistentDepthChallengerStates =
                _persistentDepthChallengers.Count,
            BatchIntegrationMilliseconds = _batchIntegrationTicks * 1000d / Stopwatch.Frequency,
            LastExtractionMilliseconds = _lastExtractionMilliseconds,
            LastScannedCells = _lastMesh != null ? _lastMesh.ScannedCells : 0,
            LastZeroCrossingCells = _lastMesh != null ? _lastMesh.ZeroCrossingCells : 0,
            LastMultiDirectionZeroCrossingCells = _lastMesh != null ? _lastMesh.MultiDirectionZeroCrossingCells : 0,
            LastThreePlusDirectionHypothesisCells =
                _lastMesh != null ? _lastMesh.ThreePlusDirectionHypothesisCells : 0,
            LastThreePlusSurfaceClusterCells =
                _lastMesh != null ? _lastMesh.ThreePlusSurfaceClusterCells : 0,
            LastMaximumDirectionHypothesesPerCell =
                _lastMesh != null ? _lastMesh.MaximumDirectionHypothesesPerCell : 0,
            LastMaximumSurfaceClustersPerCell =
                _lastMesh != null ? _lastMesh.MaximumSurfaceClustersPerCell : 0,
            LastSecondaryLayerZeroCrossingCells = _lastMesh != null ? _lastMesh.SecondaryLayerZeroCrossingCells : 0,
            LastSuppressedOppositeDirectionCells = _lastMesh != null ? _lastMesh.SuppressedOppositeDirectionCells : 0,
            LastSuppressedCoincidentSecondaryCells = _lastMesh != null ? _lastMesh.SuppressedCoincidentSecondaryCells : 0,
            LastPreservedCloseParallelCells = _lastMesh != null ? _lastMesh.PreservedCloseParallelCells : 0,
            LastInvalidDirectionHypotheses = _lastMesh != null ? _lastMesh.InvalidDirectionHypotheses : 0,
            LastObservedNormalHypotheses = _lastMesh != null ? _lastMesh.ObservedNormalHypotheses : 0,
            LastParallelHypothesesCollapsed = _lastMesh != null ? _lastMesh.ParallelHypothesesCollapsed : 0,
            LastIncompatibleWeakHypothesesDropped = _lastMesh != null ? _lastMesh.IncompatibleWeakHypothesesDropped : 0,
            LastEdgeCrossingsMerged = _lastMesh != null ? _lastMesh.EdgeCrossingsMerged : 0,
            LastEdgeCrossingOverflowDropped = _lastMesh != null ? _lastMesh.EdgeCrossingOverflowDropped : 0,
            LastConservativeConflictCells = _lastMesh != null ? _lastMesh.ConservativeConflictCells : 0,
            LastNonManifoldTrianglesDropped = _lastMesh != null ? _lastMesh.NonManifoldTrianglesDropped : 0,
            LastSurfaceCandidateNodes = _lastMesh != null ? _lastMesh.SurfaceCandidateNodes : 0,
            LastSurfaceComponents = _lastMesh != null ? _lastMesh.SurfaceComponents : 0,
            LastSurfaceComponentLinks = _lastMesh != null ? _lastMesh.SurfaceComponentLinks : 0,
            LastSurfaceSingleCellGapLinks = _lastMesh != null ? _lastMesh.SurfaceSingleCellGapLinks : 0,
            LastSurfaceComponentSingletons = _lastMesh != null ? _lastMesh.SurfaceComponentSingletons : 0,
            LastSurfaceGapBridgeCandidates = _lastMesh != null ? _lastMesh.SurfaceGapBridgeCandidates : 0,
            LastSurfaceGapBridgeTriangles = _lastMesh != null ? _lastMesh.SurfaceGapBridgeTriangles : 0,
            LastSurfaceConsistentEdgeMerges = _lastMesh != null ? _lastMesh.SurfaceConsistentEdgeMerges : 0,
            LastCreaseJunctionEdgeMerges = _lastMesh != null ? _lastMesh.CreaseJunctionEdgeMerges : 0,
            LastCrossSurfaceNearCrossingsPreserved =
                _lastMesh != null ? _lastMesh.CrossSurfaceNearCrossingsPreserved : 0,
            LastSurfaceContinuityEdgeDisagreements =
                _lastMesh != null ? _lastMesh.SurfaceContinuityEdgeDisagreements : 0,
            LastPhysicalSurfaceLocalMerges =
                _lastMesh != null ? _lastMesh.PhysicalSurfaceLocalMerges : 0,
            LastPhysicalSurfaceOppositeDirectionMerges =
                _lastMesh != null ? _lastMesh.PhysicalSurfaceOppositeDirectionMerges : 0,
            LastPhysicalSurfaceOneToOneLinks =
                _lastMesh != null ? _lastMesh.PhysicalSurfaceOneToOneLinks : 0,
            LastPhysicalSurfaceCertificateCandidatePairs =
                _lastMesh != null ? _lastMesh.PhysicalSurfaceCertificateCandidatePairs : 0,
            LastPhysicalSurfaceCertificateLinks =
                _lastMesh != null ? _lastMesh.PhysicalSurfaceCertificateLinks : 0,
            LastPhysicalSurfaceCertificateRelaxedLinks =
                _lastMesh != null ? _lastMesh.PhysicalSurfaceCertificateRelaxedLinks : 0,
            LastPhysicalSurfaceCertificateRejectedPairs =
                _lastMesh != null ? _lastMesh.PhysicalSurfaceCertificateRejectedPairs : 0,
            LastPhysicalSurfaceCertificateAmbiguousPairs =
                _lastMesh != null ? _lastMesh.PhysicalSurfaceCertificateAmbiguousPairs : 0,
            LastPhysicalSurfaceCertificateIndexedNodePairs =
                _lastMesh != null ? _lastMesh.PhysicalSurfaceCertificateIndexedNodePairs : 0,
            LastPhysicalSurfaceCertificateDirectCandidates =
                _lastMesh != null ? _lastMesh.PhysicalSurfaceCertificateDirectCandidates : 0,
            LastPhysicalSurfaceCertificateDirectLinks =
                _lastMesh != null ? _lastMesh.PhysicalSurfaceCertificateDirectLinks : 0,
            LastPhysicalSurfaceCertificateDirectRejected =
                _lastMesh != null ? _lastMesh.PhysicalSurfaceCertificateDirectRejected : 0,
            LastPhysicalSurfaceIdentityEdgeCandidates =
                _lastMesh != null ? _lastMesh.PhysicalSurfaceIdentityEdgeCandidates : 0,
            LastPhysicalSurfaceIdentityEdgeMerges =
                _lastMesh != null ? _lastMesh.PhysicalSurfaceIdentityEdgeMerges : 0,
            LastPhysicalSurfaceIdentityEdgeRejected =
                _lastMesh != null ? _lastMesh.PhysicalSurfaceIdentityEdgeRejected : 0,
            DirectionalMcShadowEnabled =
                _lastMesh != null && _lastMesh.DirectionalMcShadowEnabled,
            DirectionalMcShadowPaperReference =
                _lastMesh != null &&
                _lastMesh.DirectionalMcShadowPaperReference,
            LastDirectionalMcShadowCellsEvaluated =
                _lastMesh != null ? _lastMesh.DirectionalMcShadowCellsEvaluated : 0,
            LastDirectionalMcShadowRawHypotheses =
                _lastMesh != null ? _lastMesh.DirectionalMcShadowRawHypotheses : 0,
            LastDirectionalMcShadowIncompleteCornerHypotheses =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowIncompleteCornerHypotheses
                    : 0,
            LastDirectionalMcShadowIntraDirectionRejected =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowIntraDirectionRejected
                    : 0,
            LastDirectionalMcShadowInterDirectionRejected =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowInterDirectionRejected
                    : 0,
            LastDirectionalMcShadowValidHypotheses =
                _lastMesh != null ? _lastMesh.DirectionalMcShadowValidHypotheses : 0,
            LastDirectionalMcShadowComponents =
                _lastMesh != null ? _lastMesh.DirectionalMcShadowComponents : 0,
            LastDirectionalMcShadowSingleSurfaceCells =
                _lastMesh != null ? _lastMesh.DirectionalMcShadowSingleSurfaceCells : 0,
            LastDirectionalMcShadowDoubleSurfaceCells =
                _lastMesh != null ? _lastMesh.DirectionalMcShadowDoubleSurfaceCells : 0,
            LastDirectionalMcShadowOverflowDeferredComponents =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowOverflowDeferredComponents
                    : 0,
            LastDirectionalMcShadowEmptyAfterVotingCells =
                _lastMesh != null ? _lastMesh.DirectionalMcShadowEmptyAfterVotingCells : 0,
            LastDirectionalMcShadowRawTransitionEdges =
                _lastMesh != null ? _lastMesh.DirectionalMcShadowRawTransitionEdges : 0,
            LastDirectionalMcShadowCombinedTransitionEdges =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowCombinedTransitionEdges
                    : 0,
            LastDirectionalMcShadowRegularizedTransitionEdges =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowRegularizedTransitionEdges
                    : 0,
            LastDirectionalMcShadowOffsetClusters =
                _lastMesh != null ? _lastMesh.DirectionalMcShadowOffsetClusters : 0,
            LastDirectionalMcShadowDualOffsetEdges =
                _lastMesh != null ? _lastMesh.DirectionalMcShadowDualOffsetEdges : 0,
            LastDirectionalMcShadowOffsetOverflowEdges =
                _lastMesh != null ? _lastMesh.DirectionalMcShadowOffsetOverflowEdges : 0,
            LastDirectionalMcShadowNeighborFaceComparisons =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowNeighborFaceComparisons
                    : 0,
            LastDirectionalMcShadowNeighborDisagreementsBefore =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowNeighborDisagreementsBefore
                    : 0,
            LastDirectionalMcShadowNeighborDisagreementsAfter =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowNeighborDisagreementsAfter
                    : 0,
            LastDirectionalMcShadowRegularizedCorners =
                _lastMesh != null ? _lastMesh.DirectionalMcShadowRegularizedCorners : 0,
            LastDirectionalMcShadowRegularizedCells =
                _lastMesh != null ? _lastMesh.DirectionalMcShadowRegularizedCells : 0,
            LastDirectionalMcShadowRegularizationDeferredPairs =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowRegularizationDeferredPairs
                    : 0,
            LastDirectionalMcShadowPersistentDecisions =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowPersistentDecisions
                    : 0,
            LastDirectionalMcShadowChangedDecisions =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowChangedDecisions
                    : 0,
            LastDirectionalMcShadowRetiredDecisions =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowRetiredDecisions
                    : 0,
            LastDirectionalMcShadowUnknownDeferredCells =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowUnknownDeferredCells
                    : 0,
            LastDirectionalMcShadowAmbiguousFaces =
                _lastMesh != null ? _lastMesh.DirectionalMcShadowAmbiguousFaces : 0,
            LastDirectionalMcShadowInteriorTests =
                _lastMesh != null ? _lastMesh.DirectionalMcShadowInteriorTests : 0,
            LastDirectionalMcShadowTriangles =
                _lastMesh != null ? _lastMesh.DirectionalMcShadowTriangles : 0,
            LastDirectionalMcShadowVertices =
                _lastMesh != null ? _lastMesh.DirectionalMcShadowVertices : 0,
            LastDirectionalMcShadowFineRefinementBlocks =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFineRefinementBlocks
                    : 0,
            LastDirectionalMcShadowFineCellsEvaluated =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFineCellsEvaluated
                    : 0,
            LastDirectionalMcShadowFineCellsAccepted =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFineCellsAccepted
                    : 0,
            LastDirectionalMcShadowFineCoarseCellsPromoted =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFineCoarseCellsPromoted
                    : 0,
            LastDirectionalMcShadowFineBoundaryDeferredCells =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFineBoundaryDeferredCells
                    : 0,
            LastDirectionalMcShadowFineCoarsePriorCorners =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFineCoarsePriorCorners
                    : 0,
            LastDirectionalMcShadowFineIncompleteCells =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFineIncompleteCells
                    : 0,
            LastDirectionalMcShadowFineVotingEmptyCells =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFineVotingEmptyCells
                    : 0,
            LastDirectionalMcShadowFineVertices =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFineVertices
                    : 0,
            LastDirectionalMcShadowFineTriangles =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFineTriangles
                    : 0,
            LastDirectionalMcShadowOverflowRefinementBlocks =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowOverflowRefinementBlocks
                    : 0,
            LastDirectionalMcShadowOverflowRefinementActiveBlocks =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowOverflowRefinementActiveBlocks
                    : 0,
            LastDirectionalMcShadowOverflowResolvedByHalfVoxel =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowOverflowResolvedByHalfVoxel
                    : 0,
            LastDirectionalMcShadowOverflowStillUnresolved =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowOverflowStillUnresolved
                    : 0,
            LastDirectionalMcShadowFaceDecisionCandidates =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFaceDecisionCandidates
                    : 0,
            LastDirectionalMcShadowFaceDecisionConflicts =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFaceDecisionConflicts
                    : 0,
            LastDirectionalMcShadowFaceDecisionStable =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFaceDecisionStable
                    : 0,
            LastDirectionalMcShadowFaceDecisionAppliedPairs =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFaceDecisionAppliedPairs
                    : 0,
            LastDirectionalMcShadowFaceDecisionPersistent =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFaceDecisionPersistent
                    : 0,
            LastDirectionalMcShadowFaceDecisionRetired =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFaceDecisionRetired
                    : 0,
            LastDirectionalMcShadowFaceDecisionFilledCorners =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFaceDecisionFilledCorners
                    : 0,
            LastDirectionalMcShadowFaceDecisionChangedCells =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFaceDecisionChangedCells
                    : 0,
            LastDirectionalMcShadowFaceDecisionPrecompletedCorners =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFaceDecisionPrecompletedCorners
                    : 0,
            LastDirectionalMcShadowFaceDecisionRecoveredHypotheses =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowFaceDecisionRecoveredHypotheses
                    : 0,
            LastDirectionalMcShadowBoundaryEdges =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowBoundaryEdges
                    : 0,
            LastDirectionalMcShadowNonManifoldEdges =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowNonManifoldEdges
                    : 0,
            LastDirectionalMcShadowDuplicateTriangles =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowDuplicateTriangles
                    : 0,
            LastDirectionalMcShadowDegenerateTriangles =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowDegenerateTriangles
                    : 0,
            LastDirectionalMcShadowCrackCandidateEdges =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowCrackCandidateEdges
                    : 0,
            LastDirectionalMcShadowConflictFaceDeferredTriangles =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowConflictFaceDeferredTriangles
                    : 0,
            LastDirectionalMcShadowSharedFaceTopologyComparisons =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowSharedFaceTopologyComparisons
                    : 0,
            LastDirectionalMcShadowSharedFaceTopologyMismatches =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowSharedFaceTopologyMismatches
                    : 0,
            LastDirectionalMcShadowUnmeasuredEdgeDeferredCells =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowUnmeasuredEdgeDeferredCells
                    : 0,
            LastDirectionalMcShadowUnmeasuredEdgeDeferredTriangles =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowUnmeasuredEdgeDeferredTriangles
                    : 0,
            LastDirectionalMcShadowWindingCorrectedTriangles =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowWindingCorrectedTriangles
                    : 0,
            LastDirectionalMcShadowNormalMismatchTriangles =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowNormalMismatchTriangles
                    : 0,
            LastDirectionalMcShadowPendingTopologyChanges =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowPendingTopologyChanges
                    : 0,
            LastDirectionalMcShadowAtomicCommittedCells =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowAtomicCommittedCells
                    : 0,
            LastDirectionalMcShadowAtomicDeferredCells =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowAtomicDeferredCells
                    : 0,
            LastDirectionalMcShadowPersistentEdges =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowPersistentEdges
                    : 0,
            LastDirectionalMcShadowPersistentSurfaceIdentities =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowPersistentSurfaceIdentities
                    : 0,
            LastDirectionalMcShadowCanonicalEdgeCorrections =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowCanonicalEdgeCorrections
                    : 0,
            LastDirectionalMcShadowTopologyStableBatches =
                _lastMesh != null
                    ? _lastMesh.DirectionalMcShadowTopologyStableBatches
                    : 0,
            LastDirectionalMcShadowMilliseconds =
                _lastMesh != null ? _lastMesh.DirectionalMcShadowMilliseconds : 0d,
            HermiteQefShadowEnabled =
                _lastMesh != null && _lastMesh.HermiteQefShadowEnabled,
            HermiteQefProductionEligible =
                _lastMesh != null &&
                _lastMesh.HermiteQefProductionEligible,
            LastHermiteQefScannedCells =
                _lastMesh != null ? _lastMesh.HermiteQefScannedCells : 0,
            LastHermiteQefHermiteSamples =
                _lastMesh != null ? _lastMesh.HermiteQefHermiteSamples : 0,
            LastHermiteQefRawCandidates =
                _lastMesh != null ? _lastMesh.HermiteQefRawCandidates : 0,
            LastHermiteQefFrameRejected =
                _lastMesh != null ? _lastMesh.HermiteQefFrameRejected : 0,
            LastHermiteQefViewRejected =
                _lastMesh != null ? _lastMesh.HermiteQefViewRejected : 0,
            LastHermiteQefSampleRejected =
                _lastMesh != null ? _lastMesh.HermiteQefSampleRejected : 0,
            LastHermiteQefFamilyBalanceRejected =
                _lastMesh != null ? _lastMesh.HermiteQefFamilyBalanceRejected : 0,
            LastHermiteQefRankRejected =
                _lastMesh != null ? _lastMesh.HermiteQefRankRejected : 0,
            LastHermiteQefCellMarginRejected =
                _lastMesh != null ? _lastMesh.HermiteQefCellMarginRejected : 0,
            LastHermiteQefDisplacementRejected =
                _lastMesh != null ? _lastMesh.HermiteQefDisplacementRejected : 0,
            LastHermiteQefResidualRejected =
                _lastMesh != null ? _lastMesh.HermiteQefResidualRejected : 0,
            LastHermiteQefCertified =
                _lastMesh != null ? _lastMesh.HermiteQefCertified : 0,
            LastHermiteQefMissingPatchRejected =
                _lastMesh != null ? _lastMesh.HermiteQefMissingPatchRejected : 0,
            LastHermiteQefMultiPatchRejected =
                _lastMesh != null ? _lastMesh.HermiteQefMultiPatchRejected : 0,
            LastHermiteQefOpenBoundaryRejected =
                _lastMesh != null ? _lastMesh.HermiteQefOpenBoundaryRejected : 0,
            LastHermiteQefOrientationRejected =
                _lastMesh != null ? _lastMesh.HermiteQefOrientationRejected : 0,
            LastHermiteQefProvisionalAppliedCells =
                _lastMesh != null ? _lastMesh.HermiteQefProvisionalAppliedCells : 0,
            LastHermiteQefAppliedCells =
                _lastMesh != null ? _lastMesh.HermiteQefAppliedCells : 0,
            LastHermiteQefSourceTrianglesReplaced =
                _lastMesh != null ? _lastMesh.HermiteQefSourceTrianglesReplaced : 0,
            LastHermiteQefFeatureTrianglesAdded =
                _lastMesh != null ? _lastMesh.HermiteQefFeatureTrianglesAdded : 0,
            LastHermiteQefBoundaryEdges =
                _lastMesh != null ? _lastMesh.HermiteQefBoundaryEdges : 0,
            LastHermiteQefNonManifoldEdges =
                _lastMesh != null ? _lastMesh.HermiteQefNonManifoldEdges : 0,
            LastHermiteQefDuplicateTriangles =
                _lastMesh != null ? _lastMesh.HermiteQefDuplicateTriangles : 0,
            LastHermiteQefBoundaryEdgeDelta =
                _lastMesh != null ? _lastMesh.HermiteQefBoundaryEdgeDelta : 0,
            LastHermiteQefNonManifoldEdgeDelta =
                _lastMesh != null ? _lastMesh.HermiteQefNonManifoldEdgeDelta : 0,
            LastHermiteQefDuplicateTriangleDelta =
                _lastMesh != null ? _lastMesh.HermiteQefDuplicateTriangleDelta : 0,
            LastHermiteQefPreRollbackBoundaryEdges =
                _lastMesh != null ? _lastMesh.HermiteQefPreRollbackBoundaryEdges : 0,
            LastHermiteQefPreRollbackNonManifoldEdges =
                _lastMesh != null ? _lastMesh.HermiteQefPreRollbackNonManifoldEdges : 0,
            LastHermiteQefPreRollbackDuplicateTriangles =
                _lastMesh != null ? _lastMesh.HermiteQefPreRollbackDuplicateTriangles : 0,
            LastHermiteQefPreRollbackBoundaryEdgeDelta =
                _lastMesh != null ? _lastMesh.HermiteQefPreRollbackBoundaryEdgeDelta : 0,
            LastHermiteQefPreRollbackNonManifoldEdgeDelta =
                _lastMesh != null ? _lastMesh.HermiteQefPreRollbackNonManifoldEdgeDelta : 0,
            LastHermiteQefPreRollbackDuplicateTriangleDelta =
                _lastMesh != null ? _lastMesh.HermiteQefPreRollbackDuplicateTriangleDelta : 0,
            LastHermiteQefTopologyRollback =
                _lastMesh != null ? _lastMesh.HermiteQefTopologyRollback : 0,
            LastHermiteQefMilliseconds =
                _lastMesh != null ? _lastMesh.HermiteQefMilliseconds : 0d,
            LastVertices = _lastMesh != null ? _lastMesh.Vertices.Count : 0,
            LastTriangles = _lastMesh != null ? _lastMesh.TriangleCount : 0,
            LastBoundaryEdges = _lastMesh != null ? _lastMesh.BoundaryEdges : 0,
            LastNonManifoldEdges = _lastMesh != null ? _lastMesh.NonManifoldEdges : 0,
            LastDuplicateTriangles = _lastMesh != null ? _lastMesh.DuplicateTriangles : 0,
            LastRefinementProbeEntries =
                _lastMesh != null ? _lastMesh.RefinementProbeEntries : 0,
            LastRefinementSameDirectionSpreadCells =
                _lastMesh != null ? _lastMesh.RefinementSameDirectionSpreadCells : 0,
            LastRefinementCreaseCells =
                _lastMesh != null ? _lastMesh.RefinementCreaseCells : 0,
            LastRefinementDmcDoubleSurfaceCells =
                _lastMesh != null
                    ? _lastMesh.RefinementDmcDoubleSurfaceCells
                    : 0,
            LastRefinementDmcDoubleSurfaceBlocks =
                _lastMesh != null
                    ? _lastMesh.RefinementDmcDoubleSurfaceBlocks
                    : 0,
            LastRefinementHalfVoxelResolvableCells =
                _lastMesh != null ? _lastMesh.RefinementHalfVoxelResolvableCells : 0,
            LastRefinementHalfVoxelInsufficientCells =
                _lastMesh != null ? _lastMesh.RefinementHalfVoxelInsufficientCells : 0,
            LastRefinementCandidateBlocks =
                _lastMesh != null ? _lastMesh.RefinementCandidateBlocks : 0,
            LastRefinementPersistentBlocks =
                _lastMesh != null ? _lastMesh.RefinementPersistentBlocks : 0,
            LastRefinementDirtyBlocks =
                _lastMesh != null ? _lastMesh.RefinementDirtyBlocks : 0,
            LastRefinementCleanBlocks =
                _lastMesh != null ? _lastMesh.RefinementCleanBlocks : 0,
            LastRefinementProjectedVoxelMultiplier =
                _lastMesh != null ? _lastMesh.RefinementProjectedVoxelMultiplier : 1f,
            LastRefinementBoundsValid =
                _lastMesh != null && _lastMesh.RefinementBoundsValid,
            LastRefinementMinimumWorld =
                _lastMesh != null ? _lastMesh.RefinementMinimumWorld : Vector3.zero,
            LastRefinementMaximumWorld =
                _lastMesh != null ? _lastMesh.RefinementMaximumWorld : Vector3.zero,
            LastRefinementDepthSpanBuckets =
                _lastMesh != null
                    ? (int[])_lastMesh.RefinementDepthSpanBuckets.Clone()
                    : new int[4],
            LastHalfVoxelShadowActiveBlocks =
                _lastMesh != null ? _lastMesh.HalfVoxelShadowActiveBlocks : 0,
            LastHalfVoxelShadowAllocatedBlocks =
                _lastMesh != null ? _lastMesh.HalfVoxelShadowAllocatedBlocks : 0,
            LastHalfVoxelShadowAllocatedLayers =
                _lastMesh != null ? _lastMesh.HalfVoxelShadowAllocatedLayers : 0,
            LastHalfVoxelShadowWeightedVoxels =
                _lastMesh != null ? _lastMesh.HalfVoxelShadowWeightedVoxels : 0,
            LastHalfVoxelShadowCandidateCells =
                _lastMesh != null ? _lastMesh.HalfVoxelShadowCandidateCells : 0,
            LastHalfVoxelShadowBufferedSamples =
                _lastMesh != null ? _lastMesh.HalfVoxelShadowBufferedSamples : 0,
            LastHalfVoxelShadowReplayedSamples =
                _lastMesh != null ? _lastMesh.HalfVoxelShadowReplayedSamples : 0,
            LastHalfVoxelShadowVoxelUpdates =
                _lastMesh != null ? _lastMesh.HalfVoxelShadowVoxelUpdates : 0,
            LastHalfVoxelShadowZeroCrossingCells =
                _lastMesh != null ? _lastMesh.HalfVoxelShadowZeroCrossingCells : 0,
            LastHalfVoxelShadowPredictedCellsEvaluated =
                _lastMesh != null ? _lastMesh.HalfVoxelShadowPredictedCellsEvaluated : 0,
            LastHalfVoxelShadowCoarseEndpointResolvedCells =
                _lastMesh != null ? _lastMesh.HalfVoxelShadowCoarseEndpointResolvedCells : 0,
            LastHalfVoxelShadowFineEndpointResolvedCells =
                _lastMesh != null ? _lastMesh.HalfVoxelShadowFineEndpointResolvedCells : 0,
            LastHalfVoxelShadowRecoveredCells =
                _lastMesh != null ? _lastMesh.HalfVoxelShadowRecoveredCells : 0,
            LastHalfVoxelShadowMissingCells =
                _lastMesh != null ? _lastMesh.HalfVoxelShadowMissingCells : 0,
            LastHalfVoxelShadowExtraEnvelopeCells =
                _lastMesh != null ? _lastMesh.HalfVoxelShadowExtraEnvelopeCells : 0,
            LastHalfVoxelShadowIntegrationMilliseconds =
                _lastMesh != null ? _lastMesh.HalfVoxelShadowIntegrationMilliseconds : 0d,
            LastHalfVoxelShadowEvaluationMilliseconds =
                _lastMesh != null ? _lastMesh.HalfVoxelShadowEvaluationMilliseconds : 0d,
            LastHalfVoxelScaledTruncationAllocatedLayers =
                _lastMesh != null ? _lastMesh.HalfVoxelScaledTruncationAllocatedLayers : 0,
            LastHalfVoxelScaledTruncationWeightedVoxels =
                _lastMesh != null ? _lastMesh.HalfVoxelScaledTruncationWeightedVoxels : 0,
            LastHalfVoxelScaledTruncationCandidateCells =
                _lastMesh != null ? _lastMesh.HalfVoxelScaledTruncationCandidateCells : 0,
            LastHalfVoxelScaledTruncationReplayedSamples =
                _lastMesh != null ? _lastMesh.HalfVoxelScaledTruncationReplayedSamples : 0,
            LastHalfVoxelScaledTruncationVoxelUpdates =
                _lastMesh != null ? _lastMesh.HalfVoxelScaledTruncationVoxelUpdates : 0,
            LastHalfVoxelScaledTruncationZeroCrossingCells =
                _lastMesh != null ? _lastMesh.HalfVoxelScaledTruncationZeroCrossingCells : 0,
            LastHalfVoxelScaledTruncationPredictedCellsEvaluated =
                _lastMesh != null
                    ? _lastMesh.HalfVoxelScaledTruncationPredictedCellsEvaluated
                    : 0,
            LastHalfVoxelScaledTruncationFineEndpointResolvedCells =
                _lastMesh != null
                    ? _lastMesh.HalfVoxelScaledTruncationFineEndpointResolvedCells
                    : 0,
            LastHalfVoxelScaledTruncationRecoveredCells =
                _lastMesh != null ? _lastMesh.HalfVoxelScaledTruncationRecoveredCells : 0,
            LastHalfVoxelScaledTruncationMissingCells =
                _lastMesh != null ? _lastMesh.HalfVoxelScaledTruncationMissingCells : 0,
            LastHalfVoxelScaledTruncationExtraEnvelopeCells =
                _lastMesh != null
                    ? _lastMesh.HalfVoxelScaledTruncationExtraEnvelopeCells
                    : 0,
            LastHalfVoxelScaledTruncationIntegrationMilliseconds =
                _lastMesh != null
                    ? _lastMesh.HalfVoxelScaledTruncationIntegrationMilliseconds
                    : 0d,
            LastHalfVoxelScaledTruncationEvaluationMilliseconds =
                _lastMesh != null
                    ? _lastMesh.HalfVoxelScaledTruncationEvaluationMilliseconds
                    : 0d,
            LastTruncated = _lastMesh != null && _lastMesh.Truncated,
            WeightedVoxelsByDirection = weighted,
            TrianglesByDirection = triangles
        };
    }

    private bool ShouldExtractCellLayer(
        int layerChannel,
        int cellIndex,
        float minimumWeight,
        float supportScore,
        float surfaceCoordinate,
        Dictionary<long, CellLayerEvaluation> evaluations,
        MeshResult result)
    {
        int direction = layerChannel % DirectionCount;
        int slot = layerChannel / DirectionCount;
        float coincidentDistance = _voxelSize * 0.35f;

        // Primary and secondary certificates must remain independent when they
        // describe two nearby parallel surfaces (door frame versus wall, trim, thin
        // recess).  Only collapse them when their fitted zero crossings are genuinely
        // coincident at sub-voxel scale.
        int siblingLayer = direction + (slot == 0 ? DirectionCount : 0);
        if (TryEvaluateCellLayerCached(
                siblingLayer, cellIndex, minimumWeight, evaluations,
                out float siblingSupport, out float siblingCoordinate))
        {
            float separation = Mathf.Abs(surfaceCoordinate - siblingCoordinate);
            if (separation < coincidentDistance)
            {
                if (StrongerOrStableTie(siblingSupport, siblingLayer, supportScore, layerChannel))
                {
                    result.SuppressedCoincidentSecondaryCells++;
                    return false;
                }
            }
            else if (slot == 1)
            {
                result.PreservedCloseParallelCells++;
            }
        }

        // The same physical sheet can receive opposite direction labels while the
        // normal is unstable around an oblique/exterior corner.  Suppress only an
        // opposite-layer crossing at nearly the same depth; orthogonal directions
        // remain independent so real corners are retained.
        int oppositeDirection = direction ^ 1;
        float oppositeCoincidentDistance = _voxelSize * 0.55f;
        for (int oppositeSlot = 0; oppositeSlot < SurfaceSlotCount; oppositeSlot++)
        {
            int oppositeLayer = oppositeDirection + oppositeSlot * DirectionCount;
            if (!TryEvaluateCellLayerCached(
                    oppositeLayer, cellIndex, minimumWeight, evaluations,
                    out float oppositeSupport, out float oppositeCoordinate))
                continue;
            if (Mathf.Abs(surfaceCoordinate - oppositeCoordinate) >= oppositeCoincidentDistance)
                continue;
            if (!StrongerOrStableTie(
                    oppositeSupport, oppositeLayer, supportScore, layerChannel))
                continue;
            result.SuppressedOppositeDirectionCells++;
            return false;
        }
        return true;
    }

    private bool TryEvaluateCellLayerCached(
        int layerChannel,
        int cellIndex,
        float minimumWeight,
        Dictionary<long, CellLayerEvaluation> evaluations,
        out float supportScore,
        out float surfaceCoordinate)
    {
        supportScore = 0f;
        surfaceCoordinate = 0f;
        long evaluationKey = CellLayerEvaluationKey(layerChannel, cellIndex);
        if (evaluations.TryGetValue(evaluationKey, out CellLayerEvaluation cached))
        {
            supportScore = cached.SupportScore;
            surfaceCoordinate = cached.SurfaceCoordinate;
            return cached.ZeroCrossing;
        }
        if (layerChannel < 0 || layerChannel >= LayerCount ||
            !_candidateCells[layerChannel].Contains(cellIndex))
        {
            evaluations[evaluationKey] = default;
            return false;
        }

        DecodeCell(cellIndex, out int x, out int y, out int z);
        int direction = layerChannel % DirectionCount;
        Vector3 canonicalAxis = CanonicalAxis(direction);
        float orientedAxisSign = Vector3.Dot(DirectionVectors[direction], canonicalAxis);
        bool anyPositive = false;
        bool anyNegative = false;
        float coordinateSum = 0f;
        for (int corner = 0; corner < 8; corner++)
        {
            int vx = x + CornerX[corner];
            int vy = y + CornerY[corner];
            int vz = z + CornerZ[corner];
            if (!TryReadVoxel(layerChannel, vx, vy, vz, out float value, out float weight) ||
                weight < minimumWeight)
                continue;
            anyPositive |= value >= 0f;
            anyNegative |= value < 0f;
            float boundedWeight = Mathf.Min(weight, minimumWeight * 4f);
            float coordinate = Vector3.Dot(VoxelCenter(vx, vy, vz), canonicalAxis) -
                               value * _truncation * orientedAxisSign;
            coordinateSum += coordinate * boundedWeight;
            supportScore += boundedWeight;
        }
        if (!anyPositive || !anyNegative || supportScore <= 0f)
        {
            evaluations[evaluationKey] = new CellLayerEvaluation
            {
                ZeroCrossing = false,
                SupportScore = supportScore,
                SurfaceCoordinate = 0f
            };
            return false;
        }
        surfaceCoordinate = coordinateSum / supportScore;
        evaluations[evaluationKey] = new CellLayerEvaluation
        {
            ZeroCrossing = true,
            SupportScore = supportScore,
            SurfaceCoordinate = surfaceCoordinate
        };
        return true;
    }

    private static long CellLayerEvaluationKey(int layerChannel, int cellIndex)
    {
        return ((long)layerChannel << 32) | (uint)cellIndex;
    }

    private static bool StrongerOrStableTie(
        float challengerSupport,
        int challengerLayer,
        float currentSupport,
        int currentLayer)
    {
        const float meaningfulSupportRatio = 1.05f;
        if (challengerSupport > currentSupport * meaningfulSupportRatio)
            return true;
        bool nearTie = challengerSupport >= currentSupport / meaningfulSupportRatio;
        return nearTie && challengerLayer < currentLayer;
    }

    private static Vector3 CanonicalAxis(int direction)
    {
        if (direction < 2) return Vector3.right;
        if (direction < 4) return Vector3.up;
        return Vector3.forward;
    }

    private int ResolveStableDirection(
        int ownershipVoxelIndex,
        bool secondary,
        int suggestedDirection,
        Vector3 normal,
        float switchMargin,
        int switchConfirmations,
        out int retiredDirection)
    {
        retiredDirection = -1;
        long key = ((long)(ownershipVoxelIndex + 1) << 1) | (secondary ? 1L : 0L);
        if (!_directionOwners.TryGetValue(key, out DirectionOwnerState state))
        {
            state.Direction = suggestedDirection;
            state.Challenger = -1;
            state.ChallengerHits = 0;
            _directionOwners[key] = state;
            return suggestedDirection;
        }

        if (state.Direction == suggestedDirection)
        {
            state.Challenger = -1;
            state.ChallengerHits = 0;
            _directionOwners[key] = state;
            _batchOwnerMatched++;
            return state.Direction;
        }

        _batchOwnerChallenged++;
        float currentScore = Mathf.Max(0f, Vector3.Dot(normal, DirectionVectors[state.Direction]));
        float suggestedScore = Mathf.Max(0f, Vector3.Dot(normal, DirectionVectors[suggestedDirection]));
        if (suggestedScore < currentScore + switchMargin)
        {
            state.Challenger = -1;
            state.ChallengerHits = 0;
            _directionOwners[key] = state;
            return state.Direction;
        }

        if (state.Challenger == suggestedDirection)
            state.ChallengerHits++;
        else
        {
            state.Challenger = suggestedDirection;
            state.ChallengerHits = 1;
        }
        if (state.ChallengerHits < switchConfirmations)
        {
            _directionOwners[key] = state;
            return state.Direction;
        }

        retiredDirection = state.Direction;
        state.Direction = suggestedDirection;
        state.Challenger = -1;
        state.ChallengerHits = 0;
        _directionOwners[key] = state;
        _batchOwnerSwitched++;
        return state.Direction;
    }

    private void RetirePreviousDirectionNearSurface(
        int layerChannel,
        Vector3 surfacePoint,
        Vector3 normal)
    {
        Vector3 extent = Vector3.one * (_voxelSize * 1.25f);
        WorldToVoxelClamped(surfacePoint - extent, out int minX, out int minY, out int minZ);
        WorldToVoxelClamped(surfacePoint + extent, out int maxX, out int maxY, out int maxZ);
        float maximumNormalDistance = _voxelSize * 0.85f;
        float maximumLateralDistanceSq = _voxelSize * _voxelSize * 1.25f;
        for (int z = minZ; z <= maxZ; z++)
        for (int y = minY; y <= maxY; y++)
        for (int x = minX; x <= maxX; x++)
        {
            Vector3 delta = VoxelCenter(x, y, z) - surfacePoint;
            float normalDistance = Vector3.Dot(delta, normal);
            if (Mathf.Abs(normalDistance) > maximumNormalDistance)
                continue;
            Vector3 lateral = delta - normal * normalDistance;
            if (lateral.sqrMagnitude > maximumLateralDistanceSq)
                continue;
            if (!TryGetLayer(layerChannel, x, y, z, out DirectionLayer layer, out int localIndex) ||
                layer.Weight[localIndex] <= 0f)
                continue;
            layer.Weight[localIndex] = 0f;
            layer.Tsdf[localIndex] = 0f;
            layer.NormalSum[localIndex] = Vector3.zero;
            layer.PaperFrameMask[localIndex] = 0UL;
            layer.PaperViewMask[localIndex] = 0UL;
            layer.DepthConflictHits[localIndex] = 0;
            layer.LastDepthConflictBatch[localIndex] = 0;
            _weightedVoxels = Mathf.Max(0, _weightedVoxels - 1);
            int direction = layerChannel % DirectionCount;
            _weightedVoxelsByDirection[direction] =
                Mathf.Max(0, _weightedVoxelsByDirection[direction] - 1);
            _batchRetiredDirectionVoxels++;
        }
    }

    private void IntegrateVoxel(
        int layerChannel,
        int x,
        int y,
        int z,
        float sampleTsdf,
        float sampleWeight,
        Vector3 sampleNormal,
        Vector3 surfacePoint,
        int sourceFrameIndex,
        bool allowFastDepthCorrection)
    {
        DirectionLayer layer = GetOrCreateLayer(layerChannel, x, y, z, out int localIndex);
        float oldWeight = layer.Weight[localIndex];
        float oldTsdf = layer.Tsdf[localIndex];
        bool mature = oldWeight >= _maximumWeight * 0.75f;
        float depthDelta = Mathf.Abs(oldTsdf - sampleTsdf);
        bool opposingReliableSigns =
            oldTsdf * sampleTsdf < 0f && depthDelta >= 0.20f;
        bool matureDepthConflict =
            mature && (opposingReliableSigns || depthDelta >= 0.45f);
        bool confirmedDepthCorrection = false;
        if (matureDepthConflict)
        {
            _batchMatureDepthConflictCandidates++;
            int lastConflictBatch = layer.LastDepthConflictBatch[localIndex];
            if (lastConflictBatch != _batchSequence)
            {
                layer.DepthConflictHits[localIndex] =
                    lastConflictBatch == _batchSequence - 1
                        ? (byte)Mathf.Min(byte.MaxValue,
                            layer.DepthConflictHits[localIndex] + 1)
                        : (byte)1;
                layer.LastDepthConflictBatch[localIndex] = _batchSequence;
            }
            confirmedDepthCorrection = layer.DepthConflictHits[localIndex] >= 2;
            int direction = layerChannel % DirectionCount;
            float directionAlignment = Mathf.Abs(Vector3.Dot(
                sampleNormal, DirectionVectors[direction]));
            bool safePatchCandidate =
                allowFastDepthCorrection &&
                sourceFrameIndex >= 0 &&
                sampleWeight >= 0.50f &&
                directionAlignment >= 0.85f;
            bool persistentPatchCandidate =
                sourceFrameIndex >= 0 &&
                sampleWeight >= 0.50f &&
                directionAlignment >= 0.65f;
            long correctionKey =
                ((long)layerChannel << 32) |
                (uint)VoxelIndex(x, y, z);
            if (!confirmedDepthCorrection &&
                persistentPatchCandidate)
            {
                _batchMatureDepthPersistentCandidates++;
                bool persistentConfirmed =
                    TryConfirmPersistentDepthChallenger(
                        layerChannel, x, y, z,
                        surfacePoint, sampleNormal,
                        sourceFrameIndex,
                        !safePatchCandidate,
                        out bool persistentMatched,
                        out bool persistentCrossVoxelMatched);
                if (persistentMatched)
                    _batchMatureDepthPersistentMatched++;
                if (persistentCrossVoxelMatched)
                    _batchMatureDepthPersistentCrossVoxelMatched++;
                if (persistentConfirmed &&
                    _batchFastCorrectedVoxels.Add(correctionKey))
                {
                    confirmedDepthCorrection = true;
                    _batchMatureDepthPersistentConfirmed++;
                    if (!safePatchCandidate)
                        _batchMatureDepthPersistentRiskConfirmed++;
                }
            }
            long evidenceKey = 0L;
            if (safePatchCandidate)
            {
                _batchMatureDepthPatchCandidates++;
                evidenceKey = RecordBatchDepthConflictEvidence(
                    layerChannel, x, y, z,
                    surfacePoint, sampleNormal,
                    sourceFrameIndex, opposingReliableSigns);
                if (!confirmedDepthCorrection &&
                    TryConfirmPreviousBatchPhysicalSurface(
                        layerChannel, x, y, z,
                        surfacePoint, sampleNormal,
                        out int patchSharedNeighbors))
                {
                    confirmedDepthCorrection = true;
                    _batchMatureDepthPatchConfirmed++;
                    _batchMatureDepthPatchSharedNeighbors +=
                        patchSharedNeighbors;
                }
            }
            if (!confirmedDepthCorrection)
            {
                if (safePatchCandidate)
                {
                    _batchMatureDepthFastCandidates++;
                    bool fastConfirmed = TryConfirmFastDepthCorrection(
                        evidenceKey, layerChannel, x, y, z,
                        out int sharedNeighbors);
                    if (fastConfirmed &&
                        (!opposingReliableSigns || sharedNeighbors > 0) &&
                        _batchFastCorrectedVoxels.Add(evidenceKey))
                    {
                        confirmedDepthCorrection = true;
                        _batchMatureDepthFastConfirmed++;
                        _batchMatureDepthFastSharedNeighbors += sharedNeighbors;
                    }
                    else
                    {
                        _batchMatureDepthFastHeld++;
                    }
                }
                else if (sourceFrameIndex >= 0)
                {
                    // Contested/secondary evidence and oblique boundary
                    // channels remain on the original two-batch path.
                    _batchMatureDepthFastRiskDeferred++;
                }
            }
            if (!confirmedDepthCorrection)
            {
                _batchMatureDepthConflictHeld++;
                return;
            }
            _batchMatureDepthCorrections++;
        }
        else if (layer.LastDepthConflictBatch[localIndex] != _batchSequence)
        {
            // Compatible evidence cancels an old, non-consecutive challenge.
            layer.DepthConflictHits[localIndex] = 0;
            layer.LastDepthConflictBatch[localIndex] = 0;
        }

        bool saturated = oldWeight >= _maximumWeight - 0.0001f;
        if (saturated || confirmedDepthCorrection)
        {
            // A capped TSDF must remain correctable.  Replace a bounded part of
            // the old evidence while keeping confidence capped instead of
            // freezing forever at maximum weight.
            float replacementWeight = Mathf.Min(
                Mathf.Max(0.0001f, sampleWeight),
                Mathf.Max(
                    0.0001f,
                    _maximumWeight *
                    0.125f));
            float retainedWeight = Mathf.Max(0f, oldWeight - replacementWeight);
            float correctedWeight = retainedWeight + replacementWeight;
            float correctedTsdf =
                (oldTsdf * retainedWeight + sampleTsdf * replacementWeight) /
                Mathf.Max(0.0001f, correctedWeight);
            float retainedNormalScale = oldWeight > 0f ? retainedWeight / oldWeight : 0f;
            layer.Tsdf[localIndex] = correctedTsdf;
            layer.NormalSum[localIndex] =
                layer.NormalSum[localIndex] * retainedNormalScale +
                sampleNormal * replacementWeight;
            layer.Weight[localIndex] = correctedWeight;
            if (saturated)
                _batchSaturatedRollingUpdates++;
            if (confirmedDepthCorrection && oldTsdf * correctedTsdf < 0f)
                _batchMatureDepthSignFlips++;
            _batchVoxelUpdates++;
            MarkCandidateCells(layerChannel, x, y, z);
            return;
        }

        float acceptedWeight = Mathf.Min(sampleWeight, Mathf.Max(0f, _maximumWeight - oldWeight));
        if (acceptedWeight <= 0f)
            return;
        float newWeight = oldWeight + acceptedWeight;
        layer.Tsdf[localIndex] = oldWeight > 0f
            ? (layer.Tsdf[localIndex] * oldWeight + sampleTsdf * acceptedWeight) / newWeight
            : sampleTsdf;
        layer.NormalSum[localIndex] += sampleNormal * acceptedWeight;
        layer.Weight[localIndex] = newWeight;
        _batchVoxelUpdates++;
        if (oldWeight <= 0f)
        {
            _weightedVoxels++;
            int direction = layerChannel % DirectionCount;
            _weightedVoxelsByDirection[direction]++;
        }
        if (Mathf.Abs(sampleTsdf) <= 1f)
            MarkCandidateCells(layerChannel, x, y, z);
    }

    private long RecordBatchDepthConflictEvidence(
        int layerChannel,
        int x,
        int y,
        int z,
        Vector3 surfacePoint,
        Vector3 sampleNormal,
        int sourceFrameIndex,
        bool opposingReliableSigns)
    {
        int voxelIndex = VoxelIndex(x, y, z);
        long key = ((long)layerChannel << 32) | (uint)voxelIndex;
        _batchDepthConflictEvidence.TryGetValue(
            key, out BatchDepthConflictEvidence evidence);
        if (evidence.Batch != _batchSequence)
        {
            if (evidence.Batch == _batchSequence - 1)
            {
                evidence.PreviousBatch = evidence.Batch;
                evidence.PreviousIndependentFrameMask =
                    evidence.IndependentFrameMask;
                evidence.PreviousSurfacePointSum = evidence.SurfacePointSum;
                evidence.PreviousNormalSum = evidence.NormalSum;
                evidence.PreviousIndependentFrames =
                    evidence.IndependentFrames;
                evidence.PreviousOpposingReliableSigns =
                    evidence.OpposingReliableSigns;
            }
            else if (evidence.PreviousBatch != _batchSequence - 1)
            {
                evidence.PreviousBatch = 0;
                evidence.PreviousIndependentFrameMask = 0UL;
                evidence.PreviousSurfacePointSum = Vector3.zero;
                evidence.PreviousNormalSum = Vector3.zero;
                evidence.PreviousIndependentFrames = 0;
                evidence.PreviousOpposingReliableSigns = false;
            }
            evidence.Batch = _batchSequence;
            evidence.IndependentFrameMask = 0UL;
            evidence.SurfacePointSum = Vector3.zero;
            evidence.NormalSum = Vector3.zero;
            evidence.IndependentFrames = 0;
            evidence.OpposingReliableSigns = false;
        }
        ulong frameBit = 1UL << (sourceFrameIndex & 63);
        if ((evidence.IndependentFrameMask & frameBit) == 0UL)
        {
            evidence.IndependentFrameMask |= frameBit;
            evidence.IndependentFrames++;
            evidence.SurfacePointSum += surfacePoint;
            Vector3 alignedNormal =
                evidence.NormalSum.sqrMagnitude > 0.00000001f &&
                Vector3.Dot(evidence.NormalSum, sampleNormal) < 0f
                    ? -sampleNormal
                    : sampleNormal;
            evidence.NormalSum += alignedNormal;
        }
        evidence.OpposingReliableSigns |= opposingReliableSigns;
        _batchDepthConflictEvidence[key] = evidence;
        return key;
    }

    private bool TryConfirmFastDepthCorrection(
        long currentKey,
        int layerChannel,
        int x,
        int y,
        int z,
        out int sharedNeighbors)
    {
        sharedNeighbors = 0;
        if (!_batchDepthConflictEvidence.TryGetValue(
                currentKey, out BatchDepthConflictEvidence current) ||
            current.Batch != _batchSequence ||
            current.IndependentFrames <= 0)
            return false;

        Vector3 currentNormal = current.Normal;
        Vector3 currentPoint = current.SurfacePoint;
        float maximumPlaneOffset =
            _voxelSize * FastDepthCorrectionMaximumPlaneOffsetVoxels;
        float maximumTangentDistance =
            _voxelSize * FastDepthCorrectionMaximumTangentDistanceVoxels;
        int slot = layerChannel / DirectionCount;
        int firstLayer = slot * DirectionCount;
        int lastLayer = firstLayer + DirectionCount;
        ulong independentFrames = 0UL;

        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
        {
            int neighborX = x + dx;
            int neighborY = y + dy;
            int neighborZ = z + dz;
            if (neighborX < 0 || neighborY < 0 || neighborZ < 0 ||
                neighborX >= _dimX || neighborY >= _dimY ||
                neighborZ >= _dimZ)
                continue;
            int neighborVoxel = VoxelIndex(neighborX, neighborY, neighborZ);
            for (int neighborLayer = firstLayer;
                 neighborLayer < lastLayer;
                 neighborLayer++)
            {
                long neighborKey =
                    ((long)neighborLayer << 32) | (uint)neighborVoxel;
                if (!_batchDepthConflictEvidence.TryGetValue(
                        neighborKey, out BatchDepthConflictEvidence neighbor) ||
                    neighbor.Batch != _batchSequence ||
                    neighbor.IndependentFrames <= 0)
                    continue;
                if (neighbor.OpposingReliableSigns !=
                    current.OpposingReliableSigns)
                    continue;
                Vector3 neighborNormal = neighbor.Normal;
                float signedNormalDot = Vector3.Dot(
                    currentNormal, neighborNormal);
                if (Mathf.Abs(signedNormalDot) <
                    FastDepthCorrectionMinimumNormalDot)
                    continue;
                if (signedNormalDot < 0f)
                    neighborNormal = -neighborNormal;
                Vector3 neighborPoint = neighbor.SurfacePoint;
                Vector3 delta = neighborPoint - currentPoint;
                float currentPlaneOffset = Mathf.Abs(Vector3.Dot(
                    delta, currentNormal));
                float neighborPlaneOffset = Mathf.Abs(Vector3.Dot(
                    delta, neighborNormal));
                if (currentPlaneOffset > maximumPlaneOffset ||
                    neighborPlaneOffset > maximumPlaneOffset)
                    continue;
                Vector3 averageNormal = currentNormal + neighborNormal;
                if (averageNormal.sqrMagnitude <= 0.00000001f)
                    continue;
                averageNormal.Normalize();
                float signedPlaneOffset = Vector3.Dot(delta, averageNormal);
                Vector3 tangent = delta - averageNormal * signedPlaneOffset;
                if (tangent.magnitude > maximumTangentDistance)
                    continue;
                independentFrames |= neighbor.IndependentFrameMask;
                if (neighborKey != currentKey)
                    sharedNeighbors++;
            }
        }

        int requiredFrames = current.OpposingReliableSigns
            ? FastDepthCorrectionMinimumIndependentFrames + 1
            : FastDepthCorrectionMinimumIndependentFrames;
        return CountBits(independentFrames) >= requiredFrames;
    }

    private bool TryConfirmPreviousBatchPhysicalSurface(
        int layerChannel,
        int x,
        int y,
        int z,
        Vector3 surfacePoint,
        Vector3 sampleNormal,
        out int sharedNeighbors)
    {
        sharedNeighbors = 0;
        float maximumPlaneOffset =
            _voxelSize * FastDepthCorrectionMaximumPlaneOffsetVoxels;
        float maximumTangentDistance =
            _voxelSize * FastDepthCorrectionMaximumTangentDistanceVoxels;
        int slot = layerChannel / DirectionCount;
        int firstLayer = slot * DirectionCount;
        int lastLayer = firstLayer + DirectionCount;
        int currentVoxel = VoxelIndex(x, y, z);
        long currentKey = ((long)layerChannel << 32) | (uint)currentVoxel;

        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
        {
            int neighborX = x + dx;
            int neighborY = y + dy;
            int neighborZ = z + dz;
            if (neighborX < 0 || neighborY < 0 || neighborZ < 0 ||
                neighborX >= _dimX || neighborY >= _dimY ||
                neighborZ >= _dimZ)
                continue;
            int neighborVoxel = VoxelIndex(neighborX, neighborY, neighborZ);
            for (int neighborLayer = firstLayer;
                 neighborLayer < lastLayer;
                 neighborLayer++)
            {
                long neighborKey =
                    ((long)neighborLayer << 32) | (uint)neighborVoxel;
                if (!_batchDepthConflictEvidence.TryGetValue(
                        neighborKey, out BatchDepthConflictEvidence evidence) ||
                    !TryGetPreviousBatchEvidence(
                        evidence,
                        out Vector3 previousPoint,
                        out Vector3 previousNormal))
                    continue;
                float signedNormalDot = Vector3.Dot(
                    sampleNormal, previousNormal);
                if (Mathf.Abs(signedNormalDot) <
                    FastDepthCorrectionMinimumNormalDot)
                    continue;
                if (signedNormalDot < 0f)
                    previousNormal = -previousNormal;
                Vector3 delta = previousPoint - surfacePoint;
                float currentPlaneOffset = Mathf.Abs(Vector3.Dot(
                    delta, sampleNormal));
                float previousPlaneOffset = Mathf.Abs(Vector3.Dot(
                    delta, previousNormal));
                if (currentPlaneOffset > maximumPlaneOffset ||
                    previousPlaneOffset > maximumPlaneOffset)
                    continue;
                Vector3 averageNormal = sampleNormal + previousNormal;
                if (averageNormal.sqrMagnitude <= 0.00000001f)
                    continue;
                averageNormal.Normalize();
                float signedPlaneOffset = Vector3.Dot(delta, averageNormal);
                Vector3 tangent = delta - averageNormal * signedPlaneOffset;
                if (tangent.magnitude > maximumTangentDistance)
                    continue;
                if (neighborVoxel != currentVoxel)
                {
                    RecordPhysicalSurfaceCertificate(
                        currentKey, neighborKey);
                }
                sharedNeighbors++;
            }
        }
        return sharedNeighbors > 0;
    }

    private bool TryConfirmPersistentDepthChallenger(
        int layerChannel,
        int x,
        int y,
        int z,
        Vector3 surfacePoint,
        Vector3 sampleNormal,
        int sourceFrameIndex,
        bool riskPath,
        out bool matched,
        out bool crossVoxelMatched)
    {
        matched = false;
        crossVoxelMatched = false;
        int slot = layerChannel / DirectionCount;
        int firstLayer = slot * DirectionCount;
        int lastLayer = firstLayer + DirectionCount;
        float maximumPlaneOffset =
            _voxelSize *
            PersistentDepthChallengerMaximumPlaneOffsetVoxels;
        float maximumTangentDistance =
            _voxelSize *
            PersistentDepthChallengerMaximumTangentDistanceVoxels;
        long bestKey = 0L;
        PersistentDepthChallenger best = null;
        float bestScore = float.PositiveInfinity;

        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
        {
            int neighborX = x + dx;
            int neighborY = y + dy;
            int neighborZ = z + dz;
            if (neighborX < 0 || neighborY < 0 || neighborZ < 0 ||
                neighborX >= _dimX || neighborY >= _dimY ||
                neighborZ >= _dimZ)
                continue;
            int neighborVoxel =
                VoxelIndex(neighborX, neighborY, neighborZ);
            for (int candidateLayer = firstLayer;
                 candidateLayer < lastLayer;
                 candidateLayer++)
            {
                long candidateKey =
                    ((long)candidateLayer << 32) |
                    (uint)neighborVoxel;
                if (!_persistentDepthChallengers.TryGetValue(
                        candidateKey,
                        out PersistentDepthChallenger challenger) ||
                    challenger.LastBatch <
                        _batchSequence -
                        PersistentDepthChallengerLeaseBatches)
                {
                    continue;
                }
                Vector3 challengerNormal =
                    challenger.ReferenceNormal.sqrMagnitude >
                    0.00000001f
                        ? challenger.ReferenceNormal.normalized
                        : challenger.CurrentNormal;
                float signedNormalDot =
                    Vector3.Dot(sampleNormal, challengerNormal);
                if (Mathf.Abs(signedNormalDot) <
                    PersistentDepthChallengerMinimumNormalDot)
                {
                    continue;
                }
                if (signedNormalDot < 0f)
                    challengerNormal = -challengerNormal;
                Vector3 delta =
                    surfacePoint - challenger.ReferencePoint;
                float currentPlaneOffset =
                    Mathf.Abs(Vector3.Dot(delta, sampleNormal));
                float challengerPlaneOffset =
                    Mathf.Abs(Vector3.Dot(
                        delta, challengerNormal));
                if (currentPlaneOffset > maximumPlaneOffset ||
                    challengerPlaneOffset > maximumPlaneOffset)
                {
                    continue;
                }
                Vector3 averageNormal =
                    sampleNormal + challengerNormal;
                if (averageNormal.sqrMagnitude <= 0.00000001f)
                    continue;
                averageNormal.Normalize();
                float signedPlaneOffset =
                    Vector3.Dot(delta, averageNormal);
                Vector3 tangent =
                    delta - averageNormal * signedPlaneOffset;
                float tangentDistance = tangent.magnitude;
                if (tangentDistance > maximumTangentDistance)
                    continue;
                float score =
                    Mathf.Max(currentPlaneOffset,
                        challengerPlaneOffset) /
                    Mathf.Max(0.0001f, maximumPlaneOffset) +
                    tangentDistance /
                    Mathf.Max(0.0001f, maximumTangentDistance) *
                    0.10f +
                    (1f - Mathf.Abs(signedNormalDot));
                if (score >= bestScore)
                    continue;
                bestScore = score;
                bestKey = candidateKey;
                best = challenger;
            }
        }

        int currentVoxel = VoxelIndex(x, y, z);
        long currentKey =
            ((long)layerChannel << 32) | (uint)currentVoxel;
        if (best == null)
        {
            if (_persistentDepthChallengers.ContainsKey(currentKey))
                return false;
            best = new PersistentDepthChallenger
            {
                AnchorLayerChannel = layerChannel,
                AnchorX = x,
                AnchorY = y,
                AnchorZ = z,
                LastBatch = _batchSequence,
                ConsecutiveBatches = 1,
                ReferencePoint = surfacePoint,
                ReferenceNormal = sampleNormal
            };
            bestKey = currentKey;
            _persistentDepthChallengers.Add(bestKey, best);
        }
        else
        {
            matched = true;
            crossVoxelMatched =
                best.AnchorLayerChannel != layerChannel ||
                best.AnchorX != x ||
                best.AnchorY != y ||
                best.AnchorZ != z;
        }

        bool enteredNewBatch = best.LastBatch != _batchSequence;
        if (enteredNewBatch)
        {
            bool consecutive =
                best.LastBatch == _batchSequence - 1;
            best.ConsecutiveBatches = consecutive
                ? best.ConsecutiveBatches + 1
                : 1;
            best.AccumulatedIndependentFrames = consecutive
                ? best.AccumulatedIndependentFrames
                : 0;
            best.PreviousSpatialVoxels = consecutive
                ? best.CurrentVoxels.Count
                : 0;
            best.CurrentFrameMask = 0UL;
            best.CurrentPointSum = Vector3.zero;
            best.CurrentNormalSum = Vector3.zero;
            best.CurrentIndependentFrames = 0;
            best.CurrentVoxels.Clear();
            best.LastBatch = _batchSequence;

            // Follow a slowly drifting physical patch instead of leaving its
            // identity pinned forever to the first observed voxel.
            if (bestKey != currentKey &&
                !_persistentDepthChallengers.ContainsKey(currentKey))
            {
                _persistentDepthChallengers.Remove(bestKey);
                bestKey = currentKey;
                best.AnchorLayerChannel = layerChannel;
                best.AnchorX = x;
                best.AnchorY = y;
                best.AnchorZ = z;
                _persistentDepthChallengers.Add(bestKey, best);
            }
        }

        best.CurrentVoxels.Add(currentVoxel);
        ulong frameBit = 1UL << (sourceFrameIndex & 63);
        if ((best.CurrentFrameMask & frameBit) == 0UL)
        {
            best.CurrentFrameMask |= frameBit;
            best.CurrentIndependentFrames++;
            best.AccumulatedIndependentFrames++;
            best.CurrentPointSum += surfacePoint;
            Vector3 alignedNormal =
                best.CurrentNormalSum.sqrMagnitude >
                    0.00000001f &&
                Vector3.Dot(best.CurrentNormalSum, sampleNormal) < 0f
                    ? -sampleNormal
                    : sampleNormal;
            best.CurrentNormalSum += alignedNormal;
        }

        Vector3 currentNormal = best.CurrentNormal;
        if (Vector3.Dot(best.ReferenceNormal, currentNormal) < 0f)
            currentNormal = -currentNormal;
        float referenceBlend =
            best.ConsecutiveBatches > 1 ? 0.25f : 1f;
        best.ReferencePoint = Vector3.Lerp(
            best.ReferencePoint,
            best.CurrentPoint,
            referenceBlend);
        Vector3 blendedNormal = Vector3.Lerp(
            best.ReferenceNormal,
            currentNormal,
            referenceBlend);
        if (blendedNormal.sqrMagnitude > 0.00000001f)
            best.ReferenceNormal = blendedNormal.normalized;

        int requiredBatches = riskPath
            ? PersistentDepthChallengerRiskMinimumStableBatches
            : PersistentDepthChallengerMinimumStableBatches;
        int requiredFrames = riskPath
            ? PersistentDepthChallengerRiskMinimumIndependentFrames
            : PersistentDepthChallengerMinimumIndependentFrames;
        int requiredSpatialVoxels = riskPath
            ? PersistentDepthChallengerRiskMinimumSpatialVoxels
            : PersistentDepthChallengerMinimumSpatialVoxels;
        return best.ConsecutiveBatches >= requiredBatches &&
            best.AccumulatedIndependentFrames >= requiredFrames &&
            best.PreviousSpatialVoxels >= requiredSpatialVoxels &&
            best.CurrentVoxels.Count >= requiredSpatialVoxels;
    }

    private void RecordPhysicalSurfaceCertificate(long left, long right)
    {
        if (left == right)
            return;
        PhysicalSurfaceCertificateKey key =
            new PhysicalSurfaceCertificateKey(left, right);
        bool existed = _physicalSurfaceCertificates.TryGetValue(
            key, out PhysicalSurfaceCertificateState state);
        state.LastConfirmedBatch = _batchSequence;
        state.Confirmations = Mathf.Min(int.MaxValue, state.Confirmations + 1);
        _physicalSurfaceCertificates[key] = state;
        if (existed)
            return;
        AddPhysicalSurfaceCertificateAdjacency(key.First, key.Second);
        AddPhysicalSurfaceCertificateAdjacency(key.Second, key.First);
    }

    private void AddPhysicalSurfaceCertificateAdjacency(long endpoint, long linkedEndpoint)
    {
        if (!_physicalSurfaceCertificateAdjacency.TryGetValue(
                endpoint, out List<long> links))
        {
            links = new List<long>(4);
            _physicalSurfaceCertificateAdjacency.Add(endpoint, links);
        }
        links.Add(linkedEndpoint);
    }

    private bool TryGetPreviousBatchEvidence(
        BatchDepthConflictEvidence evidence,
        out Vector3 surfacePoint,
        out Vector3 normal)
    {
        surfacePoint = Vector3.zero;
        normal = Vector3.up;
        Vector3 pointSum;
        Vector3 normalSum;
        int frames;
        if (evidence.Batch == _batchSequence - 1)
        {
            pointSum = evidence.SurfacePointSum;
            normalSum = evidence.NormalSum;
            frames = evidence.IndependentFrames;
        }
        else if (evidence.PreviousBatch == _batchSequence - 1)
        {
            pointSum = evidence.PreviousSurfacePointSum;
            normalSum = evidence.PreviousNormalSum;
            frames = evidence.PreviousIndependentFrames;
        }
        else
        {
            return false;
        }
        if (frames <= 0 || normalSum.sqrMagnitude <= 0.00000001f)
            return false;
        surfacePoint = pointSum / frames;
        normal = normalSum.normalized;
        return true;
    }

    private void PruneDepthConflictEvidence()
    {
        if (_batchDepthConflictEvidence.Count <= 0)
            return;
        List<long> expired = null;
        foreach (KeyValuePair<long, BatchDepthConflictEvidence> pair in
                 _batchDepthConflictEvidence)
        {
            BatchDepthConflictEvidence evidence = pair.Value;
            if (evidence.Batch >= _batchSequence - 2 ||
                evidence.PreviousBatch >= _batchSequence - 2)
                continue;
            if (expired == null)
                expired = new List<long>();
            expired.Add(pair.Key);
        }
        if (expired == null)
            return;
        for (int i = 0; i < expired.Count; i++)
            _batchDepthConflictEvidence.Remove(expired[i]);
    }

    private void PrunePersistentDepthChallengers()
    {
        if (_persistentDepthChallengers.Count <= 0)
            return;
        List<long> expired = null;
        foreach (KeyValuePair<long, PersistentDepthChallenger> pair in
                 _persistentDepthChallengers)
        {
            if (pair.Value.LastBatch >=
                _batchSequence -
                PersistentDepthChallengerLeaseBatches)
            {
                continue;
            }
            if (expired == null)
                expired = new List<long>();
            expired.Add(pair.Key);
        }
        if (expired == null)
            return;
        for (int index = 0; index < expired.Count; index++)
            _persistentDepthChallengers.Remove(expired[index]);
    }

    private void PrunePhysicalSurfaceCertificates()
    {
        if (_physicalSurfaceCertificates.Count == 0)
            return;
        List<PhysicalSurfaceCertificateKey> expired = null;
        foreach (KeyValuePair<PhysicalSurfaceCertificateKey, PhysicalSurfaceCertificateState> pair
                 in _physicalSurfaceCertificates)
        {
            if (pair.Value.LastConfirmedBatch >=
                _batchSequence - PhysicalSurfaceCertificateLeaseBatches)
                continue;
            if (expired == null)
                expired = new List<PhysicalSurfaceCertificateKey>();
            expired.Add(pair.Key);
        }
        if (expired == null)
            return;
        for (int index = 0; index < expired.Count; index++)
        {
            PhysicalSurfaceCertificateKey key = expired[index];
            _physicalSurfaceCertificates.Remove(key);
            RemovePhysicalSurfaceCertificateAdjacency(key.First, key.Second);
            RemovePhysicalSurfaceCertificateAdjacency(key.Second, key.First);
        }
    }

    private void RemovePhysicalSurfaceCertificateAdjacency(
        long endpoint,
        long linkedEndpoint)
    {
        if (!_physicalSurfaceCertificateAdjacency.TryGetValue(
                endpoint, out List<long> links))
            return;
        links.Remove(linkedEndpoint);
        if (links.Count == 0)
            _physicalSurfaceCertificateAdjacency.Remove(endpoint);
    }

    private DirectionLayer GetOrCreateLayer(int layerChannel, int x, int y, int z, out int localIndex)
    {
        int bx = x / _blockSize;
        int by = y / _blockSize;
        int bz = z / _blockSize;
        int blockKey = bx + _blockDimX * (by + _blockDimY * bz);
        if (!_blocks.TryGetValue(blockKey, out Block block))
        {
            block = new Block();
            _blocks.Add(blockKey, block);
        }
        DirectionLayer layer = block.Layers[layerChannel];
        if (layer == null)
        {
            layer = new DirectionLayer(_blockSize * _blockSize * _blockSize);
            block.Layers[layerChannel] = layer;
            _allocatedDirectionLayers++;
        }
        int lx = x - bx * _blockSize;
        int ly = y - by * _blockSize;
        int lz = z - bz * _blockSize;
        localIndex = lx + _blockSize * (ly + _blockSize * lz);
        return layer;
    }

    private bool TryGetLayer(
        int layerChannel,
        int x,
        int y,
        int z,
        out DirectionLayer layer,
        out int localIndex)
    {
        layer = null;
        localIndex = -1;
        if (layerChannel < 0 || layerChannel >= LayerCount ||
            x < 0 || y < 0 || z < 0 ||
            x >= _dimX || y >= _dimY || z >= _dimZ)
            return false;

        int bx = x / _blockSize;
        int by = y / _blockSize;
        int bz = z / _blockSize;
        int blockKey = bx + _blockDimX * (by + _blockDimY * bz);
        if (!_blocks.TryGetValue(blockKey, out Block block))
            return false;

        layer = block.Layers[layerChannel];
        if (layer == null)
            return false;

        int lx = x - bx * _blockSize;
        int ly = y - by * _blockSize;
        int lz = z - bz * _blockSize;
        localIndex = lx + _blockSize * (ly + _blockSize * lz);
        return true;
    }

    private bool TryReadVoxel(int layerChannel, int x, int y, int z, out float tsdf, out float weight)
    {
        tsdf = 1f;
        weight = 0f;
        if (!TryGetLayer(layerChannel, x, y, z, out DirectionLayer layer, out int localIndex))
            return false;
        weight = layer.Weight[localIndex];
        if (weight <= 0f)
            return false;
        tsdf = layer.Tsdf[localIndex];
        return true;
    }

    private bool TryReadVoxelNormal(
        int layerChannel,
        int x,
        int y,
        int z,
        out Vector3 normal,
        out float weight)
    {
        normal = Vector3.zero;
        weight = 0f;
        if (!TryGetLayer(layerChannel, x, y, z, out DirectionLayer layer, out int localIndex))
            return false;
        weight = layer.Weight[localIndex];
        Vector3 sum = layer.NormalSum[localIndex];
        if (weight <= 0f || sum.sqrMagnitude <= 0.00000001f)
            return false;
        normal = sum.normalized;
        return true;
    }

    private void MarkCandidateCells(int layerChannel, int x, int y, int z)
    {
        for (int dz = -1; dz <= 0; dz++)
        for (int dy = -1; dy <= 0; dy++)
        for (int dx = -1; dx <= 0; dx++)
        {
            int cx = x + dx;
            int cy = y + dy;
            int cz = z + dz;
            if (cx < 0 || cy < 0 || cz < 0 || cx >= _dimX - 1 || cy >= _dimY - 1 || cz >= _dimZ - 1)
                continue;
            _candidateCells[layerChannel].Add(CellIndex(cx, cy, cz));
            _dirtyExtractionBlocks.Add(BlockKeyFromCell(cx, cy, cz));
        }
    }

    private int BlockKeyFromCell(int x, int y, int z)
    {
        int blockX = Mathf.Clamp(x / _blockSize, 0, _blockDimX - 1);
        int blockY = Mathf.Clamp(y / _blockSize, 0, _blockDimY - 1);
        int blockZ = Mathf.Max(0, z / _blockSize);
        return blockX + _blockDimX * (blockY + _blockDimY * blockZ);
    }

    private void DecodeBlockKey(
        int blockKey, out int blockX, out int blockY, out int blockZ)
    {
        blockX = blockKey % _blockDimX;
        int remainder = blockKey / _blockDimX;
        blockY = remainder % _blockDimY;
        blockZ = remainder / _blockDimY;
    }

    private void PolygonizeTetrahedron(
        int direction, int a, int b, int c, int d,
        int[] indices, float[] values, Vector3[] positions,
        int[] tetra, int[] inside, int[] outside,
        Dictionary<ulong, int> edgeVertices, MeshResult result)
    {
        tetra[0] = a;
        tetra[1] = b;
        tetra[2] = c;
        tetra[3] = d;
        int insideCount = 0;
        int outsideCount = 0;
        for (int i = 0; i < 4; i++)
        {
            if (values[tetra[i]] < 0f)
                inside[insideCount++] = tetra[i];
            else
                outside[outsideCount++] = tetra[i];
        }
        if (insideCount == 0 || insideCount == 4)
            return;

        Vector3 outward = DirectionVectors[direction];
        if (insideCount == 1 || insideCount == 3)
        {
            bool invert = insideCount == 3;
            int pivot = invert ? outside[0] : inside[0];
            int q0 = invert ? inside[0] : outside[0];
            int q1 = invert ? inside[1] : outside[1];
            int q2 = invert ? inside[2] : outside[2];
            int v0 = EdgeVertex(pivot, q0, indices, values, positions, edgeVertices, result.Vertices);
            int v1 = EdgeVertex(pivot, q1, indices, values, positions, edgeVertices, result.Vertices);
            int v2 = EdgeVertex(pivot, q2, indices, values, positions, edgeVertices, result.Vertices);
            AddOrientedTriangle(direction, v0, v1, v2, outward, result);
            return;
        }

        int i0 = inside[0];
        int i1 = inside[1];
        int o0 = outside[0];
        int o1 = outside[1];
        int v00 = EdgeVertex(i0, o0, indices, values, positions, edgeVertices, result.Vertices);
        int v01 = EdgeVertex(i0, o1, indices, values, positions, edgeVertices, result.Vertices);
        int v11 = EdgeVertex(i1, o1, indices, values, positions, edgeVertices, result.Vertices);
        int v10 = EdgeVertex(i1, o0, indices, values, positions, edgeVertices, result.Vertices);
        AddOrientedTriangle(direction, v00, v01, v11, outward, result);
        AddOrientedTriangle(direction, v00, v11, v10, outward, result);
    }

    private static int EdgeVertex(
        int localA, int localB,
        int[] indices, float[] values, Vector3[] positions,
        Dictionary<ulong, int> edgeVertices, List<Vector3> vertices)
    {
        int indexA = indices[localA];
        int indexB = indices[localB];
        uint minimum = (uint)Mathf.Min(indexA, indexB);
        uint maximum = (uint)Mathf.Max(indexA, indexB);
        ulong key = ((ulong)minimum << 32) | maximum;
        if (edgeVertices.TryGetValue(key, out int vertexIndex))
            return vertexIndex;
        float valueA = values[localA];
        float valueB = values[localB];
        float denominator = valueA - valueB;
        float t = Mathf.Abs(denominator) > 0.000001f
            ? Mathf.Clamp01(valueA / denominator)
            : 0.5f;
        vertexIndex = vertices.Count;
        vertices.Add(Vector3.Lerp(positions[localA], positions[localB], t));
        edgeVertices[key] = vertexIndex;
        return vertexIndex;
    }

    private static void AddOrientedTriangle(
        int direction, int a, int b, int c, Vector3 outward, MeshResult result)
    {
        if (a == b || b == c || c == a)
            return;
        Vector3 cross = Vector3.Cross(result.Vertices[b] - result.Vertices[a], result.Vertices[c] - result.Vertices[a]);
        if (cross.sqrMagnitude <= 0.0000000001f)
            return;
        if (Vector3.Dot(cross, outward) < 0f)
        {
            int swap = b;
            b = c;
            c = swap;
        }
        result.TrianglesByDirection[direction].Add(a);
        result.TrianglesByDirection[direction].Add(b);
        result.TrianglesByDirection[direction].Add(c);
    }

    private static void BuildWireAndAudit(MeshResult result)
    {
        Dictionary<ulong, int> edgeUse = new Dictionary<ulong, int>(131072);
        HashSet<TriangleKey> uniqueTriangles = new HashSet<TriangleKey>();
        for (int direction = 0; direction < DirectionCount; direction++)
        {
            List<int> triangles = result.TrianglesByDirection[direction];
            HashSet<ulong> directionEdges = new HashSet<ulong>();
            List<int> lines = result.LinesByDirection[direction];
            for (int i = 0; i + 2 < triangles.Count; i += 3)
            {
                int a = triangles[i];
                int b = triangles[i + 1];
                int c = triangles[i + 2];
                if (!uniqueTriangles.Add(new TriangleKey(direction, a, b, c)))
                    result.DuplicateTriangles++;
                AddWireEdge(a, b, directionEdges, lines, edgeUse);
                AddWireEdge(b, c, directionEdges, lines, edgeUse);
                AddWireEdge(c, a, directionEdges, lines, edgeUse);
            }
        }
        foreach (int use in edgeUse.Values)
        {
            if (use == 1)
                result.BoundaryEdges++;
            else if (use > 2)
                result.NonManifoldEdges++;
        }
    }

    private static void AddWireEdge(
        int a, int b, HashSet<ulong> directionEdges, List<int> lines,
        Dictionary<ulong, int> globalEdgeUse)
    {
        uint minimum = (uint)Mathf.Min(a, b);
        uint maximum = (uint)Mathf.Max(a, b);
        ulong key = ((ulong)minimum << 32) | maximum;
        globalEdgeUse.TryGetValue(key, out int use);
        globalEdgeUse[key] = use + 1;
        if (!directionEdges.Add(key))
            return;
        lines.Add(a);
        lines.Add(b);
    }

    private int DominantDirection(Vector3 normal)
    {
        float ax = Mathf.Abs(normal.x);
        float ay = Mathf.Abs(normal.y);
        float az = Mathf.Abs(normal.z);
        if (ax >= ay && ax >= az)
            return normal.x >= 0f ? 0 : 1;
        if (ay >= az)
            return normal.y >= 0f ? 2 : 3;
        return normal.z >= 0f ? 4 : 5;
    }

    private void WorldToVoxelClamped(Vector3 world, out int x, out int y, out int z)
    {
        Vector3 local = (world - _origin) / _voxelSize;
        x = Mathf.Clamp(Mathf.RoundToInt(local.x), 0, _dimX - 1);
        y = Mathf.Clamp(Mathf.RoundToInt(local.y), 0, _dimY - 1);
        z = Mathf.Clamp(Mathf.RoundToInt(local.z), 0, _dimZ - 1);
    }

    private Vector3 VoxelCenter(int x, int y, int z)
    {
        return _origin + new Vector3(x * _voxelSize, y * _voxelSize, z * _voxelSize);
    }

    private int VoxelIndex(int x, int y, int z)
    {
        return x + _dimX * (y + _dimY * z);
    }

    private void DecodeVoxel(int index, out int x, out int y, out int z)
    {
        int plane = _dimX * _dimY;
        z = index / plane;
        int remainder = index - z * plane;
        y = remainder / _dimX;
        x = remainder - y * _dimX;
    }

    private int CellIndex(int x, int y, int z)
    {
        return x + (_dimX - 1) * (y + (_dimY - 1) * z);
    }

    private void DecodeCell(int index, out int x, out int y, out int z)
    {
        int cellX = _dimX - 1;
        int cellY = _dimY - 1;
        int plane = cellX * cellY;
        z = index / plane;
        int remainder = index - z * plane;
        y = remainder / cellX;
        x = remainder - y * cellX;
    }

    private static HashSet<int>[] CreateCandidateSets()
    {
        HashSet<int>[] result = new HashSet<int>[LayerCount];
        for (int i = 0; i < LayerCount; i++)
            result[i] = new HashSet<int>();
        return result;
    }

    private static HashSet<int>[] CreateHalfVoxelCandidateSets()
    {
        HashSet<int>[] result = new HashSet<int>[DirectionCount];
        for (int i = 0; i < DirectionCount; i++)
            result[i] = new HashSet<int>();
        return result;
    }

    private static int CountBits(byte value)
    {
        int count = 0;
        while (value != 0)
        {
            count += value & 1;
            value >>= 1;
        }
        return count;
    }

    private static int CountBits(ulong value)
    {
        int count = 0;
        while (value != 0UL)
        {
            value &= value - 1UL;
            count++;
        }
        return count;
    }

    private static bool Finite(Vector3 value)
    {
        return !float.IsNaN(value.x) && !float.IsInfinity(value.x) &&
               !float.IsNaN(value.y) && !float.IsInfinity(value.y) &&
               !float.IsNaN(value.z) && !float.IsInfinity(value.z);
    }

    private struct TriangleKey : IEquatable<TriangleKey>
    {
        private readonly int _direction;
        private readonly int _a;
        private readonly int _b;
        private readonly int _c;

        public TriangleKey(int direction, int a, int b, int c)
        {
            _direction = direction;
            _a = Mathf.Min(a, Mathf.Min(b, c));
            _c = Mathf.Max(a, Mathf.Max(b, c));
            _b = a + b + c - _a - _c;
        }

        public bool Equals(TriangleKey other)
        {
            return _direction == other._direction && _a == other._a && _b == other._b && _c == other._c;
        }

        public override bool Equals(object obj)
        {
            return obj is TriangleKey other && Equals(other);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = _direction;
                hash = hash * 397 ^ _a;
                hash = hash * 397 ^ _b;
                hash = hash * 397 ^ _c;
                return hash;
            }
        }
    }
}
