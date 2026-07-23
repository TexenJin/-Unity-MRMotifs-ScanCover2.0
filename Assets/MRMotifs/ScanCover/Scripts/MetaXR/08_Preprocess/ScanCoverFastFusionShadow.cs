using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using UnityEngine;

/// <summary>
/// Isolated performance shadow for ScanCover's future fusion route.
///
/// The shadow owns every voxel it writes. It only receives cloned raw-depth snapshots and
/// cloned authoritative arrays for comparison, so it cannot publish, promote, roll back, or
/// otherwise mutate the production TSDF/result pipeline.
/// </summary>
public sealed class ScanCoverFastFusionShadow
{
    private static readonly int[] PlaneNeighborOffsetX = { -1, 1, 0, 0, -1, 1, -1, 1 };
    private static readonly int[] PlaneNeighborOffsetY = { 0, 0, -1, 1, -1, -1, 1, 1 };

    public struct Config
    {
        public int BatchId;
        public int DimX;
        public int DimY;
        public int DimZ;
        public Vector3 Origin;
        public float VoxelSize;
        public float Truncation;
        public float MinDepth;
        public float MaxDepth;
        public float MaxWeight;
        public float MinimumAngleWeight;
        public float MinimumDistanceWeight;
        public float ProjectiveLateralRadiusScale;
        public int PlaneStride;
        public int DetailStride;
        public int BlockVoxels;
        public float PlaneNormalDot;
        public float PlaneResidualMeters;
        public int MinimumPlaneNeighbors;
        public float ShadowComparisonMinWeight;
        public int AuthorityComparisonMinWeight;
        public float BlockAuditMinimumWeightedCoverage;
        public float BlockAuditMinimumSignAgreement;
        public float BlockAuditMinimumZeroCrossJaccard;
        public float BlockAuditMaximumMeanSurfaceDeltaMeters;
        public bool EnableMismatchBlockReconciliation;
        public bool EnableDepthSupportGate;
        public int DepthSupportRadiusPixels;
        public int MinimumDepthConsistentNeighbors;
        public float MaximumDepthDiscontinuityMeters;
        public float MaximumDepthDiscontinuityRatio;
        public float MinimumDepthConsistencyRatio;
        public bool EnableConservativeFreeSpaceCarving;
        public float FreeSpaceWeightScale;
        public float FreeSpaceMaximumDistanceMeters;
        public float FreeSpaceMaximumExtraEdgeGrowthRatio;
        public int FreeSpaceMaximumExtraEdgeGrowthAbsolute;
        public bool EnableStableProjectiveOwnership;
        public int StableOwnershipSwitchConfirmations;
        public int StableOwnershipPendingMaxGapFrames;
        public float StableOwnershipLateralImprovementRatio;
        public float StableOwnershipPendingTsdfTolerance;
        public float StableOwnershipPendingSurfaceToleranceVoxels;
        public bool ResetStateAtBatchBoundary;
    }

    /// <summary>
    /// Immutable result of one authoritative fusion decision. The performance shadow replays
    /// these final voxel states into sparse blocks; it never reads or mutates live authority
    /// storage. This separates decision equivalence from sparse-storage equivalence.
    /// </summary>
    public struct VoxelWrite
    {
        public int DenseIndex;
        public float Tsdf;
        public byte Weight;
        public int OperationCode;
    }

    public struct FrameMetrics
    {
        public int BatchId;
        public int FrameOrdinal;
        public int RawFrameIndex;
        public int InputPixels;
        public int PlaneLikePixels;
        public int DetailPixels;
        public int PlaneSkippedPixels;
        public int SampledPixels;
        public int ValidSamples;
        public int DepthSupportRejectedPixels;
        public int ProjectiveCandidateAssignments;
        public int ProjectiveUniqueAssignments;
        public int ProjectiveAssignmentCollisions;
        public int ProjectiveAssignmentReplacements;
        public int ProjectiveConflictingSignCandidates;
        public int StableOwnershipSeeds;
        public int StableOwnershipCompatibleWrites;
        public int StableOwnershipConflictCandidates;
        public int StableOwnershipHeldWrites;
        public int StableOwnershipPendingConfirmations;
        public int StableOwnershipConfirmedSwitches;
        public int StableOwnershipStates;
        public int BandWrites;
        public int DirtyBlocks;
        public int NewBlocks;
        public int ResidentBlocks;
        public int WeightedVoxels;
        public int DirtyZeroCrossEdges;
        public int SaturatedVoxels;
        public int SaturatedWriteSkips;
        public int SaturatedConflictSkips;
        public int WeightLimitedWrites;
        public int CappedRecencyWrites;
        public int CappedConflictRecencyWrites;
        public int CappedConflictSignFlips;
        public int FreeSpaceCandidates;
        public int FreeSpaceWrites;
        public int FreeSpaceNewWeightedVoxels;
        public int FreeSpaceNegativeUpdates;
        public int FreeSpaceSignFlips;
        public int FreeSpaceRejectedNoSurfaceNeighbor;
        public int DecisionJournalWrites;
        public int DecisionJournalUniqueVoxels;
        public int DecisionJournalClears;
        public double CpuMilliseconds;
    }

    public sealed class BatchMetrics
    {
        public int BatchId;
        public bool Complete;
        public string Error = string.Empty;
        public int Frames;
        public int InputPixels;
        public int PlaneLikePixels;
        public int DetailPixels;
        public int PlaneSkippedPixels;
        public int SampledPixels;
        public int ValidSamples;
        public int DepthSupportRejectedPixels;
        public int ProjectiveCandidateAssignments;
        public int ProjectiveUniqueAssignments;
        public int ProjectiveAssignmentCollisions;
        public int ProjectiveAssignmentReplacements;
        public int ProjectiveConflictingSignCandidates;
        public int StableOwnershipSeeds;
        public int StableOwnershipCompatibleWrites;
        public int StableOwnershipConflictCandidates;
        public int StableOwnershipHeldWrites;
        public int StableOwnershipPendingConfirmations;
        public int StableOwnershipConfirmedSwitches;
        public int StableOwnershipStates;
        public int BandWrites;
        public int DecisionJournalWrites;
        public int DecisionJournalUniqueVoxels;
        public int DecisionJournalClears;
        public int ResidentBlocks;
        public int DirtyBlocks;
        public int WeightedVoxels;
        public int DirtyZeroCrossEdges;
        public int SaturatedVoxels;
        public int SaturatedWriteSkips;
        public int SaturatedConflictSkips;
        public int WeightLimitedWrites;
        public int CappedRecencyWrites;
        public int CappedConflictRecencyWrites;
        public int CappedConflictSignFlips;
        public int FreeSpaceCandidates;
        public int FreeSpaceWrites;
        public int FreeSpaceNewWeightedVoxels;
        public int FreeSpaceNegativeUpdates;
        public int FreeSpaceSignFlips;
        public int FreeSpaceRejectedNoSurfaceNeighbor;
        public int ShadowZeroCrossEdges;
        public int AuthorityWeightedVoxels;
        public int AuthorityZeroCrossEdges;
        public int CommonWeightedVoxels;
        public int CommonSignAgreementVoxels;
        public int CommonZeroCrossEdges;
        public int UnionZeroCrossEdges;
        public long EstimatedSparseBytes;
        public long DenseEquivalentBytes;
        public double IntegrationCpuMilliseconds;
        public double FinalizeCpuMilliseconds;
        public double MainThreadTailWaitMilliseconds;
        public readonly List<FrameMetrics> FrameRows = new List<FrameMetrics>(16);
        public readonly List<BlockEquivalenceRow> BlockEquivalenceRows =
            new List<BlockEquivalenceRow>(128);
        public int BlockAuditSurfaceBlocks;
        public int BlockAuditEquivalentBlocks;
        public int BlockAuditExactTopologyBlocks;
        public int BlockAuditTopologyMismatchBlocks;
        public int BlockAuditGeometryMismatchBlocks;
        public int BlockAuditCoverageMismatchBlocks;
        public int BlockAuditShadowOnlyBlocks;
        public int BlockAuditAuthorityOnlyBlocks;
        public int BlockAuditCommonEdges;
        public int BlockAuditMissingEdges;
        public int BlockAuditExtraEdges;
        public float BlockAuditMeanZeroCrossJaccard;
        public float BlockAuditMeanSurfaceDeltaMeters;
        public float BlockAuditMaxSurfaceDeltaMeters;
        public int PreRepairSurfaceBlocks;
        public int PreRepairEquivalentBlocks;
        public int PreRepairSignMismatchBlocks;
        public int PreRepairTopologyMismatchBlocks;
        public float PreRepairMeanZeroCrossJaccard;
        public int ReconciliationSeedBlocks;
        public int ReconciliationTouchedBlocks;
        public int ReconciliationCopiedVoxels;
        public int ReconciliationClearedVoxels;
        public bool SpatialFusionTrendGuardEvaluated;
        public bool SpatialFusionTrendGuardPass;
        public int PreviousMissingEdges;
        public int PreviousExtraEdges;
        public int MissingEdgeDeltaFromPreviousBatch;
        public int ExtraEdgeDeltaFromPreviousBatch;
        public int MaximumAllowedExtraEdges;

        public float SampleRetentionRatio => InputPixels > 0 ? (float)SampledPixels / InputPixels : 0f;
        public float PlaneSkipRatio => PlaneLikePixels > 0 ? (float)PlaneSkippedPixels / PlaneLikePixels : 0f;
        public float SparseMemoryRatio => DenseEquivalentBytes > 0 ? (float)EstimatedSparseBytes / DenseEquivalentBytes : 0f;
        public float AuthorityWeightedCoverage => AuthorityWeightedVoxels > 0 ? (float)CommonWeightedVoxels / AuthorityWeightedVoxels : 0f;
        public float ShadowWeightedPrecision => WeightedVoxels > 0 ? (float)CommonWeightedVoxels / WeightedVoxels : 0f;
        public float CommonSignAgreement => CommonWeightedVoxels > 0 ? (float)CommonSignAgreementVoxels / CommonWeightedVoxels : 0f;
        public float ZeroCrossJaccard => UnionZeroCrossEdges > 0 ? (float)CommonZeroCrossEdges / UnionZeroCrossEdges : 0f;

        public string BuildFrameCsv()
        {
            StringBuilder csv = new StringBuilder(2048);
            csv.AppendLine("batch,frame_ordinal,raw_frame,input_pixels,plane_like_pixels,detail_pixels,plane_skipped_pixels,sampled_pixels,valid_samples,depth_support_rejected_pixels,projective_candidate_assignments,projective_unique_assignments,projective_assignment_collisions,projective_assignment_replacements,projective_conflicting_sign_candidates,stable_ownership_seeds,stable_ownership_compatible_writes,stable_ownership_conflict_candidates,stable_ownership_held_writes,stable_ownership_pending_confirmations,stable_ownership_confirmed_switches,stable_ownership_states,band_writes,free_space_candidates,free_space_writes,free_space_new_weighted_voxels,free_space_negative_updates,free_space_sign_flips,free_space_rejected_no_surface_neighbor,dirty_blocks,new_blocks,resident_blocks,weighted_voxels,dirty_zero_cross_edges,saturated_voxels,saturated_write_skips,saturated_conflict_skips,weight_limited_writes,capped_recency_writes,capped_conflict_recency_writes,capped_conflict_sign_flips,decision_journal_writes,decision_journal_unique_voxels,decision_journal_clears,cpu_ms");
            for (int i = 0; i < FrameRows.Count; i++)
            {
                FrameMetrics row = FrameRows[i];
                csv.Append(row.BatchId).Append(',')
                    .Append(row.FrameOrdinal).Append(',')
                    .Append(row.RawFrameIndex).Append(',')
                    .Append(row.InputPixels).Append(',')
                    .Append(row.PlaneLikePixels).Append(',')
                    .Append(row.DetailPixels).Append(',')
                    .Append(row.PlaneSkippedPixels).Append(',')
                    .Append(row.SampledPixels).Append(',')
                    .Append(row.ValidSamples).Append(',')
                    .Append(row.DepthSupportRejectedPixels).Append(',')
                    .Append(row.ProjectiveCandidateAssignments).Append(',')
                    .Append(row.ProjectiveUniqueAssignments).Append(',')
                    .Append(row.ProjectiveAssignmentCollisions).Append(',')
                    .Append(row.ProjectiveAssignmentReplacements).Append(',')
                    .Append(row.ProjectiveConflictingSignCandidates).Append(',')
                    .Append(row.StableOwnershipSeeds).Append(',')
                    .Append(row.StableOwnershipCompatibleWrites).Append(',')
                    .Append(row.StableOwnershipConflictCandidates).Append(',')
                    .Append(row.StableOwnershipHeldWrites).Append(',')
                    .Append(row.StableOwnershipPendingConfirmations).Append(',')
                    .Append(row.StableOwnershipConfirmedSwitches).Append(',')
                    .Append(row.StableOwnershipStates).Append(',')
                    .Append(row.BandWrites).Append(',')
                    .Append(row.FreeSpaceCandidates).Append(',')
                    .Append(row.FreeSpaceWrites).Append(',')
                    .Append(row.FreeSpaceNewWeightedVoxels).Append(',')
                    .Append(row.FreeSpaceNegativeUpdates).Append(',')
                    .Append(row.FreeSpaceSignFlips).Append(',')
                    .Append(row.FreeSpaceRejectedNoSurfaceNeighbor).Append(',')
                    .Append(row.DirtyBlocks).Append(',')
                    .Append(row.NewBlocks).Append(',')
                    .Append(row.ResidentBlocks).Append(',')
                    .Append(row.WeightedVoxels).Append(',')
                    .Append(row.DirtyZeroCrossEdges).Append(',')
                    .Append(row.SaturatedVoxels).Append(',')
                    .Append(row.SaturatedWriteSkips).Append(',')
                    .Append(row.SaturatedConflictSkips).Append(',')
                    .Append(row.WeightLimitedWrites).Append(',')
                    .Append(row.CappedRecencyWrites).Append(',')
                    .Append(row.CappedConflictRecencyWrites).Append(',')
                    .Append(row.CappedConflictSignFlips).Append(',')
                    .Append(row.DecisionJournalWrites).Append(',')
                    .Append(row.DecisionJournalUniqueVoxels).Append(',')
                    .Append(row.DecisionJournalClears).Append(',')
                    .Append(row.CpuMilliseconds.ToString("F3", CultureInfo.InvariantCulture))
                    .AppendLine();
            }
            return csv.ToString();
        }

        public string BuildBlockEquivalenceCsv()
        {
            StringBuilder csv = new StringBuilder(Math.Max(2048, BlockEquivalenceRows.Count * 192));
            csv.AppendLine("batch,stage,block_x,block_y,block_z,shadow_weighted_voxels,authority_weighted_voxels,common_weighted_voxels,common_sign_agreement_voxels,weighted_coverage,sign_agreement,shadow_zero_cross_edges,authority_zero_cross_edges,common_zero_cross_edges,union_zero_cross_edges,missing_authority_edges,extra_shadow_edges,zero_cross_jaccard,common_surface_samples,mean_surface_delta_m,max_surface_delta_m,exact_topology,coverage_pass,sign_pass,topology_pass,geometry_pass,equivalent,reason");
            for (int i = 0; i < BlockEquivalenceRows.Count; i++)
            {
                BlockEquivalenceRow row = BlockEquivalenceRows[i];
                csv.Append(row.BatchId).Append(',').Append(row.Stage).Append(',')
                    .Append(row.Block.x).Append(',').Append(row.Block.y).Append(',').Append(row.Block.z).Append(',')
                    .Append(row.ShadowWeightedVoxels).Append(',').Append(row.AuthorityWeightedVoxels).Append(',')
                    .Append(row.CommonWeightedVoxels).Append(',').Append(row.CommonSignAgreementVoxels).Append(',')
                    .Append(row.WeightedCoverage.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                    .Append(row.SignAgreement.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                    .Append(row.ShadowZeroCrossEdges).Append(',').Append(row.AuthorityZeroCrossEdges).Append(',')
                    .Append(row.CommonZeroCrossEdges).Append(',').Append(row.UnionZeroCrossEdges).Append(',')
                    .Append(row.MissingAuthorityEdges).Append(',').Append(row.ExtraShadowEdges).Append(',')
                    .Append(row.ZeroCrossJaccard.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                    .Append(row.CommonSurfaceSamples).Append(',')
                    .Append(row.MeanSurfaceDeltaMeters.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                    .Append(row.MaxSurfaceDeltaMeters.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                    .Append(row.ExactTopology ? 1 : 0).Append(',')
                    .Append(row.CoveragePass ? 1 : 0).Append(',')
                    .Append(row.SignPass ? 1 : 0).Append(',')
                    .Append(row.TopologyPass ? 1 : 0).Append(',')
                    .Append(row.GeometryPass ? 1 : 0).Append(',')
                    .Append(row.Equivalent ? 1 : 0).Append(',')
                    .Append(row.Reason).AppendLine();
            }
            return csv.ToString();
        }
    }

    public sealed class BlockEquivalenceRow
    {
        public int BatchId;
        public string Stage = "final";
        public Vector3Int Block;
        public int ShadowWeightedVoxels;
        public int AuthorityWeightedVoxels;
        public int CommonWeightedVoxels;
        public int CommonSignAgreementVoxels;
        public int ShadowZeroCrossEdges;
        public int AuthorityZeroCrossEdges;
        public int CommonZeroCrossEdges;
        public int UnionZeroCrossEdges;
        public int MissingAuthorityEdges;
        public int ExtraShadowEdges;
        public int CommonSurfaceSamples;
        public float MeanSurfaceDeltaMeters;
        public float MaxSurfaceDeltaMeters;
        public bool ExactTopology;
        public bool CoveragePass;
        public bool SignPass;
        public bool TopologyPass;
        public bool GeometryPass;
        public bool Equivalent;
        public string Reason = "none";

        public float WeightedCoverage => AuthorityWeightedVoxels > 0
            ? (float)CommonWeightedVoxels / AuthorityWeightedVoxels
            : ShadowWeightedVoxels == 0 ? 1f : 0f;
        public float SignAgreement => CommonWeightedVoxels > 0
            ? (float)CommonSignAgreementVoxels / CommonWeightedVoxels
            : AuthorityWeightedVoxels == 0 && ShadowWeightedVoxels == 0 ? 1f : 0f;
        public float ZeroCrossJaccard => UnionZeroCrossEdges > 0
            ? (float)CommonZeroCrossEdges / UnionZeroCrossEdges
            : 1f;
    }

    private sealed class BlockEdgeAuditAccumulator
    {
        public readonly HashSet<long> ShadowEdges = new HashSet<long>();
        public readonly HashSet<long> AuthorityEdges = new HashSet<long>();
    }

    private sealed class SparseBlock
    {
        public readonly float[] Tsdf;
        public readonly float[] Weights;

        public SparseBlock(int voxelCount)
        {
            Tsdf = new float[voxelCount];
            Weights = new float[voxelCount];
            for (int i = 0; i < voxelCount; i++)
                Tsdf[i] = 1f;
        }
    }

    private struct ProjectiveAssignment
    {
        public int X;
        public int Y;
        public int Z;
        public Vector3 SurfacePoint;
        public float SampleTsdf;
        public float ObservationWeight;
        public float LateralDistanceSquared;
        public bool HasRunnerUp;
        public float RunnerUpLateralDistanceSquared;
    }

    /// <summary>
    /// Persistent per-voxel ownership memory. Same-sign observations keep updating normally.
    /// A sign-changing challenger must remain spatially and numerically coherent across
    /// multiple raw frames before it may replace the accepted owner.
    /// </summary>
    private struct StableProjectiveOwnership
    {
        public Vector3 AcceptedSurfacePoint;
        public float AcceptedTsdf;
        public float AcceptedLateralDistanceSquared;
        public int LastAcceptedFrameSequence;
        public bool HasPending;
        public bool PendingNegative;
        public Vector3 PendingSurfacePoint;
        public float PendingTsdf;
        public float PendingBestLateralDistanceSquared;
        public int PendingHits;
        public int LastPendingFrameSequence;
    }

    private enum StableOwnershipDecision
    {
        Seed,
        Compatible,
        Hold,
        Switch
    }

    private struct FreeSpaceObservation
    {
        public Vector3 Camera;
        public Vector3 Ray;
        public float SurfaceDepth;
        public float ObservationWeight;
    }

    private readonly Dictionary<Vector3Int, SparseBlock> _blocks = new Dictionary<Vector3Int, SparseBlock>();
    private readonly HashSet<Vector3Int> _batchDirtyBlocks = new HashSet<Vector3Int>();
    private readonly List<FrameMetrics> _batchFrames = new List<FrameMetrics>(16);
    private readonly Dictionary<int, ProjectiveAssignment> _frameProjectiveAssignments =
        new Dictionary<int, ProjectiveAssignment>(8192);
    private readonly Dictionary<int, StableProjectiveOwnership> _stableProjectiveOwnership =
        new Dictionary<int, StableProjectiveOwnership>(16384);
    private readonly List<FreeSpaceObservation> _frameFreeSpaceObservations =
        new List<FreeSpaceObservation>(32768);
    private int _dimX;
    private int _dimY;
    private int _dimZ;
    private int _blockVoxels;
    private int _blockVoxelCount;
    private int _weightedVoxelCount;
    private float _weightedVoxelThreshold;
    private float _maximumWeight;
    private int _batchId = int.MinValue;
    private Vector3 _origin;
    private float _voxelSize;
    private double _batchIntegrationMilliseconds;
    private int _batchInputPixels;
    private int _batchPlaneLikePixels;
    private int _batchDetailPixels;
    private int _batchPlaneSkippedPixels;
    private int _batchSampledPixels;
    private int _batchValidSamples;
    private int _batchDepthSupportRejectedPixels;
    private int _batchProjectiveCandidateAssignments;
    private int _batchProjectiveUniqueAssignments;
    private int _batchProjectiveAssignmentCollisions;
    private int _batchProjectiveAssignmentReplacements;
    private int _batchProjectiveConflictingSignCandidates;
    private int _batchStableOwnershipSeeds;
    private int _batchStableOwnershipCompatibleWrites;
    private int _batchStableOwnershipConflictCandidates;
    private int _batchStableOwnershipHeldWrites;
    private int _batchStableOwnershipPendingConfirmations;
    private int _batchStableOwnershipConfirmedSwitches;
    private int _projectiveFrameSequence;
    private int _batchBandWrites;
    private int _batchSaturatedWriteSkips;
    private int _batchSaturatedConflictSkips;
    private int _batchWeightLimitedWrites;
    private int _batchCappedRecencyWrites;
    private int _batchCappedConflictRecencyWrites;
    private int _batchCappedConflictSignFlips;
    private int _batchFreeSpaceCandidates;
    private int _batchFreeSpaceWrites;
    private int _batchFreeSpaceNewWeightedVoxels;
    private int _batchFreeSpaceNegativeUpdates;
    private int _batchFreeSpaceSignFlips;
    private int _batchFreeSpaceRejectedNoSurfaceNeighbor;
    private int _batchDecisionJournalWrites;
    private int _batchDecisionJournalClears;
    private readonly HashSet<int> _batchDecisionJournalVoxelIds = new HashSet<int>();
    private bool _hasPreviousSpatialFusionAudit;
    private int _previousSpatialFusionMissingEdges;
    private int _previousSpatialFusionExtraEdges;

    /// <summary>
    /// Returns an immutable copy of the sparse blocks touched by the current batch.
    /// The promoted phase-one route consumes only these coordinates for local-update
    /// scheduling; voxel decisions and publication remain outside this class.
    /// </summary>
    public Vector3Int[] GetDirtyBlocksSnapshot()
    {
        Vector3Int[] result = new Vector3Int[_batchDirtyBlocks.Count];
        _batchDirtyBlocks.CopyTo(result);
        return result;
    }

    public FrameMetrics ApplyDecisionJournalFrame(
        Config config,
        VoxelWrite[] writes,
        int rawFrameIndex,
        int inputPixels,
        int frameOrdinal)
    {
        EnsureBatch(config);
        Stopwatch stopwatch = Stopwatch.StartNew();
        FrameMetrics metrics = new FrameMetrics
        {
            BatchId = config.BatchId,
            FrameOrdinal = frameOrdinal,
            RawFrameIndex = rawFrameIndex,
            InputPixels = Math.Max(0, inputPixels)
        };
        int blocksBefore = _blocks.Count;
        HashSet<Vector3Int> frameDirty = new HashSet<Vector3Int>();
        HashSet<int> uniqueVoxels = new HashSet<int>();
        float threshold = Math.Max(0.0001f, config.ShadowComparisonMinWeight);

        if (writes != null)
        {
            for (int i = 0; i < writes.Length; i++)
            {
                VoxelWrite write = writes[i];
                if (write.DenseIndex < 0 || write.DenseIndex >= _dimX * _dimY * _dimZ ||
                    float.IsNaN(write.Tsdf) || float.IsInfinity(write.Tsdf))
                    continue;

                IndexToCoordinates(write.DenseIndex, out int x, out int y, out int z);
                bool hadVoxel = TryGetVoxel(x, y, z, out _, out float oldWeight);
                bool oldWeighted = hadVoxel && oldWeight >= threshold;
                bool newWeighted = write.Weight >= threshold;
                if (!newWeighted && !hadVoxel)
                    continue;

                SparseBlock block = GetOrCreateBlock(
                    x, y, z, out Vector3Int key, out int localIndex);
                block.Tsdf[localIndex] = newWeighted
                    ? Clamp(write.Tsdf, -1f, 1f)
                    : 1f;
                block.Weights[localIndex] = newWeighted ? write.Weight : 0f;
                if (!oldWeighted && newWeighted) _weightedVoxelCount++;
                else if (oldWeighted && !newWeighted) _weightedVoxelCount--;

                frameDirty.Add(key);
                uniqueVoxels.Add(write.DenseIndex);
                _batchDecisionJournalVoxelIds.Add(write.DenseIndex);
                metrics.DecisionJournalWrites++;
                if (!newWeighted) metrics.DecisionJournalClears++;
            }
        }

        foreach (Vector3Int key in frameDirty)
            _batchDirtyBlocks.Add(key);
        metrics.DecisionJournalUniqueVoxels = uniqueVoxels.Count;
        metrics.SampledPixels = metrics.DecisionJournalUniqueVoxels;
        metrics.ValidSamples = metrics.DecisionJournalUniqueVoxels;
        metrics.BandWrites = metrics.DecisionJournalWrites;
        metrics.DirtyBlocks = frameDirty.Count;
        metrics.NewBlocks = _blocks.Count - blocksBefore;
        metrics.ResidentBlocks = _blocks.Count;
        metrics.WeightedVoxels = _weightedVoxelCount;
        metrics.DirtyZeroCrossEdges = CountZeroCrossEdges(threshold, frameDirty);
        stopwatch.Stop();
        metrics.CpuMilliseconds = stopwatch.Elapsed.TotalMilliseconds;

        _batchIntegrationMilliseconds += metrics.CpuMilliseconds;
        _batchInputPixels += metrics.InputPixels;
        _batchSampledPixels += metrics.SampledPixels;
        _batchValidSamples += metrics.ValidSamples;
        _batchBandWrites += metrics.BandWrites;
        _batchDecisionJournalWrites += metrics.DecisionJournalWrites;
        _batchDecisionJournalClears += metrics.DecisionJournalClears;
        _batchFrames.Add(metrics);
        return metrics;
    }

    public FrameMetrics IntegrateFrame(
        Config config,
        ScanCoverDepthGridPointCloud.RawDepthFrameSnapshot snapshot,
        int frameOrdinal)
    {
        EnsureBatch(config);
        Stopwatch stopwatch = Stopwatch.StartNew();
        FrameMetrics metrics = new FrameMetrics
        {
            BatchId = config.BatchId,
            FrameOrdinal = frameOrdinal,
            RawFrameIndex = snapshot != null ? snapshot.frameIndex : -1
        };

        if (snapshot == null || snapshot.worldPositions == null ||
            snapshot.resolutionWidth <= 1 || snapshot.resolutionHeight <= 1)
        {
            stopwatch.Stop();
            metrics.CpuMilliseconds = stopwatch.Elapsed.TotalMilliseconds;
            _batchFrames.Add(metrics);
            return metrics;
        }

        Vector3[] positions = snapshot.worldPositions;
        Vector3[] normals = snapshot.worldNormals;
        Color[] meta = snapshot.observationMeta;
        _projectiveFrameSequence++;
        int width = snapshot.resolutionWidth;
        int height = snapshot.resolutionHeight;
        int expected = Math.Min(width * height, positions.Length);
        int planeStride = Math.Max(1, config.PlaneStride);
        int detailStride = Math.Max(1, config.DetailStride);
        Vector3 camera = snapshot.hasSnapshotCameraPose ? snapshot.snapshotCameraPosition : Vector3.zero;
        int blocksBefore = _blocks.Count;
        HashSet<Vector3Int> frameDirty = new HashSet<Vector3Int>();
        // Correlated neighboring pixels must not repeatedly strengthen the same free-space
        // voxel in one frame. Temporal confirmation should come from distinct raw frames.
        HashSet<int> frameFreeSpaceVoxels = new HashSet<int>();
        _frameProjectiveAssignments.Clear();
        _frameFreeSpaceObservations.Clear();

        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;
            if (index < 0 || index >= expected)
                continue;
            metrics.InputPixels++;

            bool planeLike = IsPlaneLike(
                x, y, width, height, expected, positions, normals,
                config.PlaneNormalDot, config.PlaneResidualMeters,
                config.MinimumPlaneNeighbors);
            if (planeLike)
                metrics.PlaneLikePixels++;
            else
                metrics.DetailPixels++;

            int stride = planeLike ? planeStride : detailStride;
            if (stride > 1 && (x % stride != 0 || y % stride != 0))
            {
                if (planeLike)
                    metrics.PlaneSkippedPixels++;
                continue;
            }
            metrics.SampledPixels++;

            Vector3 point = positions[index];
            if (!Finite(point))
                continue;
            Vector3 toSurface = point - camera;
            float depth = toSurface.magnitude;
            if (!Finite(toSurface) || depth < config.MinDepth || depth > config.MaxDepth || depth <= 0.0001f)
                continue;

            float confidence = meta != null && index < meta.Length ? Clamp01(meta[index].g) : 1f;
            if (confidence <= 0.001f)
                continue;
            Vector3 ray = toSurface / depth;
            if (config.EnableDepthSupportGate &&
                !PassDepthSupportGate(
                    x, y, width, height, expected,
                    positions, normals, camera, point, depth, config))
            {
                metrics.DepthSupportRejectedPixels++;
                continue;
            }
            float facing = 1f;
            if (normals != null && index < normals.Length && Finite(normals[index]) && normals[index].sqrMagnitude > 0.000001f)
                facing = Math.Abs(Vector3.Dot(normals[index].normalized, -ray));
            float angleWeight = Lerp(Clamp01(config.MinimumAngleWeight), 1f, facing * facing);
            float rangeT = InverseLerp(config.MinDepth, Math.Max(config.MinDepth + 0.001f, config.MaxDepth), depth);
            float distanceWeight = Lerp(1f, Clamp01(config.MinimumDistanceWeight), rangeT * rangeT);
            float observationWeight = Math.Max(0.001f, confidence * angleWeight * distanceWeight);
            metrics.ValidSamples++;
            metrics.ProjectiveCandidateAssignments +=
                AccumulateProjectiveBandAssignments(
                camera, ray, depth, observationWeight,
                Math.Max(config.VoxelSize, config.Truncation),
                Clamp(config.ProjectiveLateralRadiusScale, 0.5f, 1.25f),
                out int assignmentCollisions,
                out int assignmentReplacements,
                out int conflictingSignCandidates);
            metrics.ProjectiveAssignmentCollisions += assignmentCollisions;
            metrics.ProjectiveAssignmentReplacements += assignmentReplacements;
            metrics.ProjectiveConflictingSignCandidates +=
                conflictingSignCandidates;
            if (config.EnableConservativeFreeSpaceCarving)
            {
                _frameFreeSpaceObservations.Add(new FreeSpaceObservation
                {
                    Camera = camera,
                    Ray = ray,
                    SurfaceDepth = depth,
                    ObservationWeight = observationWeight
                });
            }
        }

        metrics.ProjectiveUniqueAssignments =
            _frameProjectiveAssignments.Count;
        metrics.BandWrites = ApplyProjectiveBandAssignments(
            Math.Max(1f, config.MaxWeight),
            Math.Max(0.0001f, config.ShadowComparisonMinWeight),
            config.EnableStableProjectiveOwnership,
            Math.Max(2, config.StableOwnershipSwitchConfirmations),
            Math.Max(1, config.StableOwnershipPendingMaxGapFrames),
            Clamp(config.StableOwnershipLateralImprovementRatio, 0.1f, 1f),
            Clamp(config.StableOwnershipPendingTsdfTolerance, 0.05f, 1f),
            Math.Max(0.5f, config.StableOwnershipPendingSurfaceToleranceVoxels),
            frameDirty,
            out int stableOwnershipSeeds,
            out int stableOwnershipCompatibleWrites,
            out int stableOwnershipConflictCandidates,
            out int stableOwnershipHeldWrites,
            out int stableOwnershipPendingConfirmations,
            out int stableOwnershipConfirmedSwitches,
            out int saturatedWriteSkips,
            out int saturatedConflictSkips,
            out int weightLimitedWrites,
            out int cappedRecencyWrites,
            out int cappedConflictRecencyWrites,
            out int cappedConflictSignFlips);
        metrics.StableOwnershipSeeds = stableOwnershipSeeds;
        metrics.StableOwnershipCompatibleWrites =
            stableOwnershipCompatibleWrites;
        metrics.StableOwnershipConflictCandidates =
            stableOwnershipConflictCandidates;
        metrics.StableOwnershipHeldWrites = stableOwnershipHeldWrites;
        metrics.StableOwnershipPendingConfirmations =
            stableOwnershipPendingConfirmations;
        metrics.StableOwnershipConfirmedSwitches =
            stableOwnershipConfirmedSwitches;
        metrics.StableOwnershipStates = _stableProjectiveOwnership.Count;
        metrics.SaturatedWriteSkips = saturatedWriteSkips;
        metrics.SaturatedConflictSkips = saturatedConflictSkips;
        metrics.WeightLimitedWrites = weightLimitedWrites;
        metrics.CappedRecencyWrites = cappedRecencyWrites;
        metrics.CappedConflictRecencyWrites =
            cappedConflictRecencyWrites;
        metrics.CappedConflictSignFlips = cappedConflictSignFlips;

        for (int i = 0; i < _frameFreeSpaceObservations.Count; i++)
        {
            FreeSpaceObservation observation =
                _frameFreeSpaceObservations[i];
            metrics.FreeSpaceWrites += IntegrateConservativeFreeSpace(
                observation.Camera,
                observation.Ray,
                observation.SurfaceDepth,
                observation.ObservationWeight,
                Math.Max(config.VoxelSize, config.Truncation),
                config.MinDepth,
                Math.Max(1f, config.MaxWeight),
                Math.Max(0.0001f, config.ShadowComparisonMinWeight),
                Clamp01(config.FreeSpaceWeightScale),
                Math.Max(
                    config.VoxelSize,
                    config.FreeSpaceMaximumDistanceMeters),
                frameDirty,
                frameFreeSpaceVoxels,
                out int freeSpaceCandidates,
                out int freeSpaceNewWeightedVoxels,
                out int freeSpaceNegativeUpdates,
                out int freeSpaceSignFlips,
                out int freeSpaceRejectedNoSurfaceNeighbor);
            metrics.FreeSpaceCandidates += freeSpaceCandidates;
            metrics.FreeSpaceNewWeightedVoxels +=
                freeSpaceNewWeightedVoxels;
            metrics.FreeSpaceNegativeUpdates += freeSpaceNegativeUpdates;
            metrics.FreeSpaceSignFlips += freeSpaceSignFlips;
            metrics.FreeSpaceRejectedNoSurfaceNeighbor +=
                freeSpaceRejectedNoSurfaceNeighbor;
        }

        foreach (Vector3Int key in frameDirty)
            _batchDirtyBlocks.Add(key);
        metrics.DirtyBlocks = frameDirty.Count;
        metrics.NewBlocks = _blocks.Count - blocksBefore;
        metrics.ResidentBlocks = _blocks.Count;
        metrics.WeightedVoxels = _weightedVoxelCount;
        metrics.DirtyZeroCrossEdges = CountZeroCrossEdges(config.ShadowComparisonMinWeight, frameDirty);
        metrics.SaturatedVoxels = CountSaturatedVoxels(config.MaxWeight);
        stopwatch.Stop();
        metrics.CpuMilliseconds = stopwatch.Elapsed.TotalMilliseconds;
        _batchIntegrationMilliseconds += metrics.CpuMilliseconds;
        _batchInputPixels += metrics.InputPixels;
        _batchPlaneLikePixels += metrics.PlaneLikePixels;
        _batchDetailPixels += metrics.DetailPixels;
        _batchPlaneSkippedPixels += metrics.PlaneSkippedPixels;
        _batchSampledPixels += metrics.SampledPixels;
        _batchValidSamples += metrics.ValidSamples;
        _batchDepthSupportRejectedPixels += metrics.DepthSupportRejectedPixels;
        _batchProjectiveCandidateAssignments +=
            metrics.ProjectiveCandidateAssignments;
        _batchProjectiveUniqueAssignments +=
            metrics.ProjectiveUniqueAssignments;
        _batchProjectiveAssignmentCollisions +=
            metrics.ProjectiveAssignmentCollisions;
        _batchProjectiveAssignmentReplacements +=
            metrics.ProjectiveAssignmentReplacements;
        _batchProjectiveConflictingSignCandidates +=
            metrics.ProjectiveConflictingSignCandidates;
        _batchStableOwnershipSeeds += metrics.StableOwnershipSeeds;
        _batchStableOwnershipCompatibleWrites +=
            metrics.StableOwnershipCompatibleWrites;
        _batchStableOwnershipConflictCandidates +=
            metrics.StableOwnershipConflictCandidates;
        _batchStableOwnershipHeldWrites +=
            metrics.StableOwnershipHeldWrites;
        _batchStableOwnershipPendingConfirmations +=
            metrics.StableOwnershipPendingConfirmations;
        _batchStableOwnershipConfirmedSwitches +=
            metrics.StableOwnershipConfirmedSwitches;
        _batchBandWrites += metrics.BandWrites;
        _batchSaturatedWriteSkips += metrics.SaturatedWriteSkips;
        _batchSaturatedConflictSkips += metrics.SaturatedConflictSkips;
        _batchWeightLimitedWrites += metrics.WeightLimitedWrites;
        _batchCappedRecencyWrites += metrics.CappedRecencyWrites;
        _batchCappedConflictRecencyWrites += metrics.CappedConflictRecencyWrites;
        _batchCappedConflictSignFlips += metrics.CappedConflictSignFlips;
        _batchFreeSpaceCandidates += metrics.FreeSpaceCandidates;
        _batchFreeSpaceWrites += metrics.FreeSpaceWrites;
        _batchFreeSpaceNewWeightedVoxels += metrics.FreeSpaceNewWeightedVoxels;
        _batchFreeSpaceNegativeUpdates += metrics.FreeSpaceNegativeUpdates;
        _batchFreeSpaceSignFlips += metrics.FreeSpaceSignFlips;
        _batchFreeSpaceRejectedNoSurfaceNeighbor +=
            metrics.FreeSpaceRejectedNoSurfaceNeighbor;
        _batchFrames.Add(metrics);
        return metrics;
    }

    public BatchMetrics FinalizeBatch(Config config, float[] authorityTsdf, byte[] authorityWeights)
    {
        EnsureBatch(config);
        Stopwatch stopwatch = Stopwatch.StartNew();
        BatchMetrics result = new BatchMetrics
        {
            BatchId = config.BatchId,
            Complete = true,
            Frames = _batchFrames.Count,
            InputPixels = _batchInputPixels,
            PlaneLikePixels = _batchPlaneLikePixels,
            DetailPixels = _batchDetailPixels,
            PlaneSkippedPixels = _batchPlaneSkippedPixels,
            SampledPixels = _batchSampledPixels,
            ValidSamples = _batchValidSamples,
            DepthSupportRejectedPixels = _batchDepthSupportRejectedPixels,
            ProjectiveCandidateAssignments =
                _batchProjectiveCandidateAssignments,
            ProjectiveUniqueAssignments =
                _batchProjectiveUniqueAssignments,
            ProjectiveAssignmentCollisions =
                _batchProjectiveAssignmentCollisions,
            ProjectiveAssignmentReplacements =
                _batchProjectiveAssignmentReplacements,
            ProjectiveConflictingSignCandidates =
                _batchProjectiveConflictingSignCandidates,
            StableOwnershipSeeds = _batchStableOwnershipSeeds,
            StableOwnershipCompatibleWrites =
                _batchStableOwnershipCompatibleWrites,
            StableOwnershipConflictCandidates =
                _batchStableOwnershipConflictCandidates,
            StableOwnershipHeldWrites = _batchStableOwnershipHeldWrites,
            StableOwnershipPendingConfirmations =
                _batchStableOwnershipPendingConfirmations,
            StableOwnershipConfirmedSwitches =
                _batchStableOwnershipConfirmedSwitches,
            StableOwnershipStates = _stableProjectiveOwnership.Count,
            BandWrites = _batchBandWrites,
            SaturatedWriteSkips = _batchSaturatedWriteSkips,
            SaturatedConflictSkips = _batchSaturatedConflictSkips,
            WeightLimitedWrites = _batchWeightLimitedWrites,
            CappedRecencyWrites = _batchCappedRecencyWrites,
            CappedConflictRecencyWrites = _batchCappedConflictRecencyWrites,
            CappedConflictSignFlips = _batchCappedConflictSignFlips,
            FreeSpaceCandidates = _batchFreeSpaceCandidates,
            FreeSpaceWrites = _batchFreeSpaceWrites,
            FreeSpaceNewWeightedVoxels = _batchFreeSpaceNewWeightedVoxels,
            FreeSpaceNegativeUpdates = _batchFreeSpaceNegativeUpdates,
            FreeSpaceSignFlips = _batchFreeSpaceSignFlips,
            FreeSpaceRejectedNoSurfaceNeighbor =
                _batchFreeSpaceRejectedNoSurfaceNeighbor,
            DecisionJournalWrites = _batchDecisionJournalWrites,
            DecisionJournalUniqueVoxels = _batchDecisionJournalVoxelIds.Count,
            DecisionJournalClears = _batchDecisionJournalClears,
            ResidentBlocks = _blocks.Count,
            DirtyBlocks = _batchDirtyBlocks.Count,
            IntegrationCpuMilliseconds = _batchIntegrationMilliseconds
        };
        result.FrameRows.AddRange(_batchFrames);
        result.WeightedVoxels = _weightedVoxelCount;
        result.SaturatedVoxels = CountSaturatedVoxels(config.MaxWeight);
        result.DirtyZeroCrossEdges = CountZeroCrossEdges(config.ShadowComparisonMinWeight, _batchDirtyBlocks);
        HashSet<long> shadowCrossings = BuildShadowZeroCrossEdges(config.ShadowComparisonMinWeight);
        HashSet<long> authorityCrossings = BuildAuthorityZeroCrossEdges(
            authorityTsdf, authorityWeights, config.AuthorityComparisonMinWeight);
        HashSet<Vector3Int> mismatchBlocks = new HashSet<Vector3Int>();
        BuildBlockEquivalenceAudit(
            config, authorityTsdf, authorityWeights,
            shadowCrossings, authorityCrossings, result,
            "pre_repair", false, mismatchBlocks);
        result.ReconciliationSeedBlocks = mismatchBlocks.Count;
        if (config.EnableMismatchBlockReconciliation && mismatchBlocks.Count > 0)
        {
            ReconcileMismatchBlocks(
                mismatchBlocks, authorityTsdf, authorityWeights, config, result);
            shadowCrossings = BuildShadowZeroCrossEdges(config.ShadowComparisonMinWeight);
        }
        result.ShadowZeroCrossEdges = shadowCrossings.Count;
        result.AuthorityZeroCrossEdges = authorityCrossings.Count;
        foreach (long edge in shadowCrossings)
        {
            if (authorityCrossings.Contains(edge))
                result.CommonZeroCrossEdges++;
        }
        result.UnionZeroCrossEdges = result.ShadowZeroCrossEdges + result.AuthorityZeroCrossEdges - result.CommonZeroCrossEdges;
        BuildBlockEquivalenceAudit(
            config, authorityTsdf, authorityWeights,
            shadowCrossings, authorityCrossings, result,
            "post_repair", true, null);
        UpdateSpatialFusionTrendGuard(config, result);

        int authorityCount = Math.Min(
            authorityTsdf != null ? authorityTsdf.Length : 0,
            authorityWeights != null ? authorityWeights.Length : 0);
        float authorityMinWeight = Math.Max(1, config.AuthorityComparisonMinWeight);
        float shadowMinWeight = Math.Max(0.0001f, config.ShadowComparisonMinWeight);
        for (int index = 0; index < authorityCount; index++)
        {
            if (authorityWeights[index] < authorityMinWeight)
                continue;
            result.AuthorityWeightedVoxels++;
            IndexToCoordinates(index, out int x, out int y, out int z);
            if (!TryGetVoxel(x, y, z, out float shadowTsdf, out float shadowWeight) || shadowWeight < shadowMinWeight)
                continue;
            result.CommonWeightedVoxels++;
            if ((shadowTsdf < 0f) == (authorityTsdf[index] < 0f))
                result.CommonSignAgreementVoxels++;
        }

        long bytesPerBlock = (long)_blockVoxelCount * sizeof(float) * 2L;
        result.EstimatedSparseBytes = bytesPerBlock * _blocks.Count;
        result.DenseEquivalentBytes = (long)_dimX * _dimY * _dimZ * sizeof(float) * 2L;
        stopwatch.Stop();
        result.FinalizeCpuMilliseconds = stopwatch.Elapsed.TotalMilliseconds;
        return result;
    }

    private void UpdateSpatialFusionTrendGuard(
        Config config,
        BatchMetrics result)
    {
        if (!config.EnableConservativeFreeSpaceCarving)
        {
            _hasPreviousSpatialFusionAudit = false;
            return;
        }

        if (_hasPreviousSpatialFusionAudit)
        {
            result.SpatialFusionTrendGuardEvaluated = true;
            result.PreviousMissingEdges = _previousSpatialFusionMissingEdges;
            result.PreviousExtraEdges = _previousSpatialFusionExtraEdges;
            result.MissingEdgeDeltaFromPreviousBatch =
                result.BlockAuditMissingEdges -
                _previousSpatialFusionMissingEdges;
            result.ExtraEdgeDeltaFromPreviousBatch =
                result.BlockAuditExtraEdges -
                _previousSpatialFusionExtraEdges;
            int relativeAllowance = (int)Math.Ceiling(
                _previousSpatialFusionExtraEdges *
                Clamp(
                    config.FreeSpaceMaximumExtraEdgeGrowthRatio,
                    0f, 0.25f));
            int allowance = Math.Max(
                Math.Max(0, config.FreeSpaceMaximumExtraEdgeGrowthAbsolute),
                relativeAllowance);
            result.MaximumAllowedExtraEdges =
                _previousSpatialFusionExtraEdges + allowance;
            result.SpatialFusionTrendGuardPass =
                result.BlockAuditMissingEdges <
                    _previousSpatialFusionMissingEdges &&
                result.BlockAuditExtraEdges <=
                    result.MaximumAllowedExtraEdges;
        }

        _previousSpatialFusionMissingEdges = result.BlockAuditMissingEdges;
        _previousSpatialFusionExtraEdges = result.BlockAuditExtraEdges;
        _hasPreviousSpatialFusionAudit = true;
    }

    private void BuildBlockEquivalenceAudit(
        Config config,
        float[] authorityTsdf,
        byte[] authorityWeights,
        HashSet<long> shadowCrossings,
        HashSet<long> authorityCrossings,
        BatchMetrics result,
        string stage,
        bool finalStage,
        HashSet<Vector3Int> mismatchBlocks)
    {
        Dictionary<Vector3Int, BlockEdgeAuditAccumulator> blocks =
            new Dictionary<Vector3Int, BlockEdgeAuditAccumulator>();
        AddEdgesByBlock(shadowCrossings, blocks, true);
        AddEdgesByBlock(authorityCrossings, blocks, false);
        List<Vector3Int> keys = new List<Vector3Int>(blocks.Keys);
        keys.Sort((a, b) =>
        {
            int z = a.z.CompareTo(b.z);
            if (z != 0) return z;
            int y = a.y.CompareTo(b.y);
            return y != 0 ? y : a.x.CompareTo(b.x);
        });

        double jaccardSum = 0d;
        double surfaceDeltaSum = 0d;
        int surfaceDeltaSamples = 0;
        float shadowMinWeight = Math.Max(0.0001f, config.ShadowComparisonMinWeight);
        int authorityMinWeight = Math.Max(1, config.AuthorityComparisonMinWeight);
        float maximumMeanDelta = Math.Max(0.0001f, config.BlockAuditMaximumMeanSurfaceDeltaMeters);

        for (int keyIndex = 0; keyIndex < keys.Count; keyIndex++)
        {
            Vector3Int key = keys[keyIndex];
            BlockEdgeAuditAccumulator accumulator = blocks[key];
            BlockEquivalenceRow row = new BlockEquivalenceRow
            {
                BatchId = config.BatchId,
                Stage = stage,
                Block = key,
                ShadowZeroCrossEdges = accumulator.ShadowEdges.Count,
                AuthorityZeroCrossEdges = accumulator.AuthorityEdges.Count
            };
            foreach (long edge in accumulator.ShadowEdges)
            {
                if (!accumulator.AuthorityEdges.Contains(edge))
                {
                    row.ExtraShadowEdges++;
                    continue;
                }
                row.CommonZeroCrossEdges++;
                if (TryMeasureCommonEdgeSurfaceDelta(
                        edge, authorityTsdf, authorityWeights,
                        shadowMinWeight, authorityMinWeight, out float delta))
                {
                    row.CommonSurfaceSamples++;
                    row.MeanSurfaceDeltaMeters += delta;
                    row.MaxSurfaceDeltaMeters = Math.Max(row.MaxSurfaceDeltaMeters, delta);
                }
            }
            foreach (long edge in accumulator.AuthorityEdges)
            {
                if (!accumulator.ShadowEdges.Contains(edge))
                    row.MissingAuthorityEdges++;
            }
            row.UnionZeroCrossEdges = row.ShadowZeroCrossEdges + row.AuthorityZeroCrossEdges -
                                      row.CommonZeroCrossEdges;
            if (row.CommonSurfaceSamples > 0)
                row.MeanSurfaceDeltaMeters /= row.CommonSurfaceSamples;

            AccumulateBlockVoxelAgreement(
                key, authorityTsdf, authorityWeights,
                shadowMinWeight, authorityMinWeight, row);
            row.ExactTopology = row.MissingAuthorityEdges == 0 && row.ExtraShadowEdges == 0;
            row.CoveragePass = row.WeightedCoverage >= config.BlockAuditMinimumWeightedCoverage;
            row.SignPass = row.SignAgreement >= config.BlockAuditMinimumSignAgreement;
            row.TopologyPass = row.ZeroCrossJaccard >= config.BlockAuditMinimumZeroCrossJaccard;
            row.GeometryPass = row.CommonSurfaceSamples > 0 &&
                               row.MeanSurfaceDeltaMeters <= maximumMeanDelta;
            row.Equivalent = row.CoveragePass && row.SignPass && row.TopologyPass && row.GeometryPass;
            row.Reason = BuildBlockAuditReason(row);
            result.BlockEquivalenceRows.Add(row);
            if (!row.SignPass || !row.TopologyPass)
                mismatchBlocks?.Add(key);
            if (finalStage)
            {
                result.BlockAuditSurfaceBlocks++;
                if (row.Equivalent) result.BlockAuditEquivalentBlocks++;
                if (row.ExactTopology) result.BlockAuditExactTopologyBlocks++;
                if (!row.TopologyPass) result.BlockAuditTopologyMismatchBlocks++;
                if (!row.GeometryPass) result.BlockAuditGeometryMismatchBlocks++;
                if (!row.CoveragePass || !row.SignPass) result.BlockAuditCoverageMismatchBlocks++;
                if (row.AuthorityZeroCrossEdges == 0 && row.ShadowZeroCrossEdges > 0)
                    result.BlockAuditShadowOnlyBlocks++;
                if (row.ShadowZeroCrossEdges == 0 && row.AuthorityZeroCrossEdges > 0)
                    result.BlockAuditAuthorityOnlyBlocks++;
                result.BlockAuditCommonEdges += row.CommonZeroCrossEdges;
                result.BlockAuditMissingEdges += row.MissingAuthorityEdges;
                result.BlockAuditExtraEdges += row.ExtraShadowEdges;
            }
            else
            {
                result.PreRepairSurfaceBlocks++;
                if (row.Equivalent) result.PreRepairEquivalentBlocks++;
                if (!row.SignPass) result.PreRepairSignMismatchBlocks++;
                if (!row.TopologyPass) result.PreRepairTopologyMismatchBlocks++;
            }
            jaccardSum += row.ZeroCrossJaccard;
            surfaceDeltaSum += row.MeanSurfaceDeltaMeters * row.CommonSurfaceSamples;
            surfaceDeltaSamples += row.CommonSurfaceSamples;
            if (finalStage)
                result.BlockAuditMaxSurfaceDeltaMeters =
                    Math.Max(result.BlockAuditMaxSurfaceDeltaMeters, row.MaxSurfaceDeltaMeters);
        }
        if (finalStage)
        {
            result.BlockAuditMeanZeroCrossJaccard = result.BlockAuditSurfaceBlocks > 0
                ? (float)(jaccardSum / result.BlockAuditSurfaceBlocks) : 0f;
            result.BlockAuditMeanSurfaceDeltaMeters = surfaceDeltaSamples > 0
                ? (float)(surfaceDeltaSum / surfaceDeltaSamples) : 0f;
        }
        else
        {
            result.PreRepairMeanZeroCrossJaccard = result.PreRepairSurfaceBlocks > 0
                ? (float)(jaccardSum / result.PreRepairSurfaceBlocks) : 0f;
        }
    }

    private void AddEdgesByBlock(
        HashSet<long> edges,
        Dictionary<Vector3Int, BlockEdgeAuditAccumulator> blocks,
        bool shadow)
    {
        if (edges == null) return;
        foreach (long edge in edges)
        {
            int denseIndex = (int)(edge / 3L);
            IndexToCoordinates(denseIndex, out int x, out int y, out int z);
            Vector3Int key = new Vector3Int(
                x / _blockVoxels, y / _blockVoxels, z / _blockVoxels);
            if (!blocks.TryGetValue(key, out BlockEdgeAuditAccumulator accumulator))
            {
                accumulator = new BlockEdgeAuditAccumulator();
                blocks.Add(key, accumulator);
            }
            if (shadow) accumulator.ShadowEdges.Add(edge);
            else accumulator.AuthorityEdges.Add(edge);
        }
    }

    private void ReconcileMismatchBlocks(
        HashSet<Vector3Int> seedBlocks,
        float[] authorityTsdf,
        byte[] authorityWeights,
        Config config,
        BatchMetrics result)
    {
        int authorityCount = Math.Min(
            authorityTsdf != null ? authorityTsdf.Length : 0,
            authorityWeights != null ? authorityWeights.Length : 0);
        int authorityMinimumWeight = Math.Max(1, config.AuthorityComparisonMinWeight);
        float shadowMinimumWeight = Math.Max(0.0001f, config.ShadowComparisonMinWeight);
        HashSet<Vector3Int> touched = new HashSet<Vector3Int>();
        foreach (Vector3Int key in seedBlocks)
        {
            int minX = Math.Max(0, key.x * _blockVoxels - 1);
            int minY = Math.Max(0, key.y * _blockVoxels - 1);
            int minZ = Math.Max(0, key.z * _blockVoxels - 1);
            int maxX = Math.Min(_dimX - 1, (key.x + 1) * _blockVoxels);
            int maxY = Math.Min(_dimY - 1, (key.y + 1) * _blockVoxels);
            int maxZ = Math.Min(_dimZ - 1, (key.z + 1) * _blockVoxels);
            for (int z = minZ; z <= maxZ; z++)
            for (int y = minY; y <= maxY; y++)
            for (int x = minX; x <= maxX; x++)
            {
                int denseIndex = DenseIndex(x, y, z);
                bool authorityValid = denseIndex >= 0 && denseIndex < authorityCount &&
                                      authorityWeights[denseIndex] >= authorityMinimumWeight;
                bool shadowValid = TryGetVoxel(x, y, z, out _, out float oldWeight) &&
                                   oldWeight >= shadowMinimumWeight;
                if (!authorityValid && !shadowValid) continue;
                SparseBlock block = GetOrCreateBlock(x, y, z, out Vector3Int touchedKey, out int localIndex);
                touched.Add(touchedKey);
                if (authorityValid)
                {
                    block.Tsdf[localIndex] = authorityTsdf[denseIndex];
                    block.Weights[localIndex] = Math.Max(shadowMinimumWeight, authorityWeights[denseIndex]);
                    if (!shadowValid) _weightedVoxelCount++;
                    result.ReconciliationCopiedVoxels++;
                }
                else
                {
                    block.Tsdf[localIndex] = 1f;
                    block.Weights[localIndex] = 0f;
                    if (shadowValid) _weightedVoxelCount--;
                    result.ReconciliationClearedVoxels++;
                }
            }
        }
        result.ReconciliationTouchedBlocks = touched.Count;
        result.WeightedVoxels = _weightedVoxelCount;
        result.ResidentBlocks = _blocks.Count;
    }

    private void AccumulateBlockVoxelAgreement(
        Vector3Int key,
        float[] authorityTsdf,
        byte[] authorityWeights,
        float shadowMinimumWeight,
        int authorityMinimumWeight,
        BlockEquivalenceRow row)
    {
        int minX = key.x * _blockVoxels;
        int minY = key.y * _blockVoxels;
        int minZ = key.z * _blockVoxels;
        int maxX = Math.Min(_dimX, minX + _blockVoxels);
        int maxY = Math.Min(_dimY, minY + _blockVoxels);
        int maxZ = Math.Min(_dimZ, minZ + _blockVoxels);
        int authorityCount = Math.Min(
            authorityTsdf != null ? authorityTsdf.Length : 0,
            authorityWeights != null ? authorityWeights.Length : 0);
        for (int z = Math.Max(0, minZ); z < maxZ; z++)
        for (int y = Math.Max(0, minY); y < maxY; y++)
        for (int x = Math.Max(0, minX); x < maxX; x++)
        {
            bool hasShadow = TryGetVoxel(x, y, z, out float shadowTsdf, out float shadowWeight) &&
                             shadowWeight >= shadowMinimumWeight;
            int denseIndex = DenseIndex(x, y, z);
            bool hasAuthority = denseIndex >= 0 && denseIndex < authorityCount &&
                                authorityWeights[denseIndex] >= authorityMinimumWeight;
            if (hasShadow) row.ShadowWeightedVoxels++;
            if (!hasAuthority) continue;
            row.AuthorityWeightedVoxels++;
            if (!hasShadow) continue;
            row.CommonWeightedVoxels++;
            if ((shadowTsdf < 0f) == (authorityTsdf[denseIndex] < 0f))
                row.CommonSignAgreementVoxels++;
        }
    }

    private bool TryMeasureCommonEdgeSurfaceDelta(
        long edge,
        float[] authorityTsdf,
        byte[] authorityWeights,
        float shadowMinimumWeight,
        int authorityMinimumWeight,
        out float deltaMeters)
    {
        deltaMeters = 0f;
        int axis = (int)(edge % 3L);
        int denseIndex = (int)(edge / 3L);
        IndexToCoordinates(denseIndex, out int x, out int y, out int z);
        int nx = x + (axis == 0 ? 1 : 0);
        int ny = y + (axis == 1 ? 1 : 0);
        int nz = z + (axis == 2 ? 1 : 0);
        if (!TryGetVoxel(x, y, z, out float shadowA, out float shadowWeightA) ||
            !TryGetVoxel(nx, ny, nz, out float shadowB, out float shadowWeightB) ||
            shadowWeightA < shadowMinimumWeight || shadowWeightB < shadowMinimumWeight)
            return false;
        int authorityCount = Math.Min(
            authorityTsdf != null ? authorityTsdf.Length : 0,
            authorityWeights != null ? authorityWeights.Length : 0);
        int neighborIndex = DenseIndex(nx, ny, nz);
        if (denseIndex < 0 || neighborIndex < 0 ||
            denseIndex >= authorityCount || neighborIndex >= authorityCount ||
            authorityWeights[denseIndex] < authorityMinimumWeight ||
            authorityWeights[neighborIndex] < authorityMinimumWeight)
            return false;
        float shadowT = ZeroCrossFraction(shadowA, shadowB);
        float authorityT = ZeroCrossFraction(authorityTsdf[denseIndex], authorityTsdf[neighborIndex]);
        deltaMeters = Math.Abs(shadowT - authorityT) * _voxelSize;
        return true;
    }

    private static float ZeroCrossFraction(float a, float b)
    {
        float denominator = Math.Abs(a) + Math.Abs(b);
        return denominator > 0.000001f ? Math.Abs(a) / denominator : 0.5f;
    }

    private static string BuildBlockAuditReason(BlockEquivalenceRow row)
    {
        if (row.Equivalent) return row.ExactTopology ? "equivalent_exact_topology" : "equivalent_threshold";
        StringBuilder reason = new StringBuilder(64);
        if (!row.CoveragePass) reason.Append("coverage|");
        if (!row.SignPass) reason.Append("sign|");
        if (!row.TopologyPass) reason.Append("topology|");
        if (!row.GeometryPass) reason.Append("geometry|");
        if (row.AuthorityZeroCrossEdges == 0 && row.ShadowZeroCrossEdges > 0) reason.Append("shadow_only|");
        if (row.ShadowZeroCrossEdges == 0 && row.AuthorityZeroCrossEdges > 0) reason.Append("authority_only|");
        if (reason.Length > 0) reason.Length--;
        return reason.Length > 0 ? reason.ToString() : "unknown";
    }

    public void Clear()
    {
        _blocks.Clear();
        _batchDirtyBlocks.Clear();
        _batchFrames.Clear();
        _frameProjectiveAssignments.Clear();
        _stableProjectiveOwnership.Clear();
        _frameFreeSpaceObservations.Clear();
        _batchId = int.MinValue;
        _dimX = _dimY = _dimZ = 0;
        _blockVoxels = _blockVoxelCount = 0;
        _weightedVoxelCount = 0;
        _weightedVoxelThreshold = 0f;
        _maximumWeight = 0f;
        _projectiveFrameSequence = 0;
        _hasPreviousSpatialFusionAudit = false;
        _previousSpatialFusionMissingEdges = 0;
        _previousSpatialFusionExtraEdges = 0;
        ResetBatchCounters();
    }

    private void EnsureBatch(Config config)
    {
        int blockVoxels = Math.Max(2, config.BlockVoxels);
        bool compatible =
            _dimX == config.DimX && _dimY == config.DimY && _dimZ == config.DimZ &&
            _blockVoxels == blockVoxels &&
            Vector3.SqrMagnitude(_origin - config.Origin) <= 0.00000001f &&
            Math.Abs(_voxelSize - config.VoxelSize) <= 0.000001f &&
            Math.Abs(_weightedVoxelThreshold - config.ShadowComparisonMinWeight) <= 0.000001f &&
            Math.Abs(_maximumWeight - config.MaxWeight) <= 0.000001f;
        if (!compatible)
        {
            _blocks.Clear();
            _stableProjectiveOwnership.Clear();
            _projectiveFrameSequence = 0;
            _hasPreviousSpatialFusionAudit = false;
            _previousSpatialFusionMissingEdges = 0;
            _previousSpatialFusionExtraEdges = 0;
            _dimX = Math.Max(0, config.DimX);
            _dimY = Math.Max(0, config.DimY);
            _dimZ = Math.Max(0, config.DimZ);
            _origin = config.Origin;
            _voxelSize = Math.Max(0.0001f, config.VoxelSize);
            _blockVoxels = blockVoxels;
            _blockVoxelCount = blockVoxels * blockVoxels * blockVoxels;
            _weightedVoxelCount = 0;
            _weightedVoxelThreshold = Math.Max(0.0001f, config.ShadowComparisonMinWeight);
            _maximumWeight = Math.Max(1f, config.MaxWeight);
        }
        if (_batchId == config.BatchId)
            return;
        if (compatible && config.ResetStateAtBatchBoundary)
        {
            _blocks.Clear();
            _stableProjectiveOwnership.Clear();
            _projectiveFrameSequence = 0;
            _weightedVoxelCount = 0;
        }
        _batchId = config.BatchId;
        _batchDirtyBlocks.Clear();
        _batchFrames.Clear();
        ResetBatchCounters();
    }

    private void ResetBatchCounters()
    {
        _batchIntegrationMilliseconds = 0d;
        _batchInputPixels = 0;
        _batchPlaneLikePixels = 0;
        _batchDetailPixels = 0;
        _batchPlaneSkippedPixels = 0;
        _batchSampledPixels = 0;
        _batchValidSamples = 0;
        _batchDepthSupportRejectedPixels = 0;
        _batchProjectiveCandidateAssignments = 0;
        _batchProjectiveUniqueAssignments = 0;
        _batchProjectiveAssignmentCollisions = 0;
        _batchProjectiveAssignmentReplacements = 0;
        _batchProjectiveConflictingSignCandidates = 0;
        _batchStableOwnershipSeeds = 0;
        _batchStableOwnershipCompatibleWrites = 0;
        _batchStableOwnershipConflictCandidates = 0;
        _batchStableOwnershipHeldWrites = 0;
        _batchStableOwnershipPendingConfirmations = 0;
        _batchStableOwnershipConfirmedSwitches = 0;
        _batchBandWrites = 0;
        _batchSaturatedWriteSkips = 0;
        _batchSaturatedConflictSkips = 0;
        _batchWeightLimitedWrites = 0;
        _batchCappedRecencyWrites = 0;
        _batchCappedConflictRecencyWrites = 0;
        _batchCappedConflictSignFlips = 0;
        _batchFreeSpaceCandidates = 0;
        _batchFreeSpaceWrites = 0;
        _batchFreeSpaceNewWeightedVoxels = 0;
        _batchFreeSpaceNegativeUpdates = 0;
        _batchFreeSpaceSignFlips = 0;
        _batchFreeSpaceRejectedNoSurfaceNeighbor = 0;
        _batchDecisionJournalWrites = 0;
        _batchDecisionJournalClears = 0;
        _batchDecisionJournalVoxelIds.Clear();
    }

    /// <summary>
    /// Mirrors the authority's first-line depth support gate before independent writes.
    /// Mature projective TSDF pipelines integrate only depth observations with coherent local
    /// support; otherwise silhouette noise and mixed foreground/background pixels create
    /// persistent sign flips and duplicate zero crossings. A normal-consistent planar neighbor
    /// may rescue a larger radial depth delta, which protects grazing walls.
    /// </summary>
    private static bool PassDepthSupportGate(
        int x,
        int y,
        int width,
        int height,
        int expected,
        Vector3[] positions,
        Vector3[] normals,
        Vector3 camera,
        Vector3 centerPoint,
        float centerDepth,
        Config config)
    {
        int radius = Clamp(config.DepthSupportRadiusPixels, 1, 3);
        float maximumDelta = Math.Max(
            Math.Max(0.001f, config.MaximumDepthDiscontinuityMeters),
            centerDepth * Math.Max(0f, config.MaximumDepthDiscontinuityRatio));
        int checkedNeighbors = 0;
        int consistentNeighbors = 0;
        bool centerNormalValid =
            normals != null &&
            y * width + x >= 0 &&
            y * width + x < normals.Length &&
            Finite(normals[y * width + x]) &&
            normals[y * width + x].sqrMagnitude > 0.000001f;
        Vector3 centerNormal = centerNormalValid
            ? normals[y * width + x].normalized
            : Vector3.zero;

        for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
        {
            if (dx == 0 && dy == 0)
                continue;
            int nx = x + dx;
            int ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height)
                continue;
            int neighborIndex = ny * width + nx;
            if (neighborIndex < 0 || neighborIndex >= expected)
                continue;
            Vector3 neighbor = positions[neighborIndex];
            if (!Finite(neighbor))
                continue;
            float neighborDepth = (neighbor - camera).magnitude;
            if (neighborDepth < config.MinDepth || neighborDepth > config.MaxDepth)
                continue;

            checkedNeighbors++;
            bool consistent = Math.Abs(neighborDepth - centerDepth) <= maximumDelta;
            if (!consistent && centerNormalValid &&
                normals != null && neighborIndex < normals.Length &&
                Finite(normals[neighborIndex]) &&
                normals[neighborIndex].sqrMagnitude > 0.000001f)
            {
                Vector3 neighborNormal = normals[neighborIndex].normalized;
                float normalDot = Math.Abs(Vector3.Dot(centerNormal, neighborNormal));
                float planeResidual = Math.Abs(
                    Vector3.Dot(neighbor - centerPoint, centerNormal));
                consistent =
                    normalDot >= Clamp(config.PlaneNormalDot, 0.5f, 1f) &&
                    planeResidual <= Math.Max(0.001f, config.PlaneResidualMeters);
            }
            if (consistent)
                consistentNeighbors++;
        }

        int minimumNeighbors = Math.Max(1, config.MinimumDepthConsistentNeighbors);
        if (checkedNeighbors < minimumNeighbors ||
            consistentNeighbors < minimumNeighbors)
            return false;
        float ratio = (float)consistentNeighbors / checkedNeighbors;
        return ratio >= Clamp01(config.MinimumDepthConsistencyRatio);
    }

    /// <summary>
    /// Collects ray-to-voxel candidates without mutating TSDF state. A mature projective
    /// integrator updates a voxel from the depth sample at that voxel's projected pixel,
    /// rather than allowing every nearby input ray to repeatedly vote into it. The raw
    /// snapshot does not expose camera intrinsics, so the closest sampled ray to the voxel
    /// center is the equivalent geometric assignment available here.
    /// </summary>
    private int AccumulateProjectiveBandAssignments(
        Vector3 camera,
        Vector3 ray,
        float surfaceDepth,
        float observationWeight,
        float truncation,
        float lateralRadiusScale,
        out int collisions,
        out int replacements,
        out int conflictingSignCandidates)
    {
        collisions = 0;
        replacements = 0;
        conflictingSignCandidates = 0;
        float startDepth = Math.Max(0.01f, surfaceDepth - truncation);
        float endDepth = surfaceDepth + truncation;
        Vector3 segmentStart = camera + ray * startDepth;
        Vector3 segmentEnd = camera + ray * endDepth;
        float radius = _voxelSize * Clamp(lateralRadiusScale, 0.5f, 1.25f);
        Vector3 minimumGrid = (Vector3.Min(segmentStart, segmentEnd) - Vector3.one * radius - _origin) / _voxelSize;
        Vector3 maximumGrid = (Vector3.Max(segmentStart, segmentEnd) + Vector3.one * radius - _origin) / _voxelSize;
        int minX = Clamp((int)Math.Floor(minimumGrid.x), 0, _dimX - 1);
        int minY = Clamp((int)Math.Floor(minimumGrid.y), 0, _dimY - 1);
        int minZ = Clamp((int)Math.Floor(minimumGrid.z), 0, _dimZ - 1);
        int maxX = Clamp((int)Math.Ceiling(maximumGrid.x), 0, _dimX - 1);
        int maxY = Clamp((int)Math.Ceiling(maximumGrid.y), 0, _dimY - 1);
        int maxZ = Clamp((int)Math.Ceiling(maximumGrid.z), 0, _dimZ - 1);
        float radiusSq = radius * radius;
        int candidates = 0;

        for (int z = minZ; z <= maxZ; z++)
        for (int y = minY; y <= maxY; y++)
        for (int x = minX; x <= maxX; x++)
        {
            Vector3 center = _origin + new Vector3(x * _voxelSize, y * _voxelSize, z * _voxelSize);
            float centerDepth = Vector3.Dot(center - camera, ray);
            if (centerDepth < startDepth - _voxelSize * 0.5f || centerDepth > endDepth + _voxelSize * 0.5f)
                continue;
            Vector3 projected = camera + ray * centerDepth;
            float lateralDistanceSquared =
                (center - projected).sqrMagnitude;
            if (lateralDistanceSquared > radiusSq)
                continue;

            int denseIndex = x + _dimX * (y + _dimY * z);
            ProjectiveAssignment candidate = new ProjectiveAssignment
            {
                X = x,
                Y = y,
                Z = z,
                SurfacePoint = camera + ray * surfaceDepth,
                SampleTsdf = Clamp(
                    (surfaceDepth - centerDepth) / truncation,
                    -1f, 1f),
                ObservationWeight = observationWeight,
                LateralDistanceSquared = lateralDistanceSquared,
                HasRunnerUp = false,
                RunnerUpLateralDistanceSquared = float.PositiveInfinity
            };
            candidates++;
            if (_frameProjectiveAssignments.TryGetValue(
                    denseIndex, out ProjectiveAssignment current))
            {
                collisions++;
                if ((current.SampleTsdf < 0f) !=
                    (candidate.SampleTsdf < 0f))
                    conflictingSignCandidates++;
                const float assignmentEpsilon = 0.0000001f;
                bool closer =
                    candidate.LateralDistanceSquared + assignmentEpsilon <
                    current.LateralDistanceSquared;
                bool equalDistanceHigherConfidence =
                    Math.Abs(
                        candidate.LateralDistanceSquared -
                        current.LateralDistanceSquared) <=
                    assignmentEpsilon &&
                    candidate.ObservationWeight >
                    current.ObservationWeight;
                if (closer || equalDistanceHigherConfidence)
                {
                    candidate.HasRunnerUp = true;
                    candidate.RunnerUpLateralDistanceSquared =
                        current.LateralDistanceSquared;
                    _frameProjectiveAssignments[denseIndex] =
                        candidate;
                    replacements++;
                }
                else if (!current.HasRunnerUp ||
                         candidate.LateralDistanceSquared <
                         current.RunnerUpLateralDistanceSquared)
                {
                    current.HasRunnerUp = true;
                    current.RunnerUpLateralDistanceSquared =
                        candidate.LateralDistanceSquared;
                    _frameProjectiveAssignments[denseIndex] =
                        current;
                }
            }
            else
            {
                _frameProjectiveAssignments.Add(
                    denseIndex, candidate);
            }
        }
        return candidates;
    }

    /// <summary>
    /// Applies exactly one selected projective observation per voxel for the current raw
    /// frame. Capped-recency behavior is unchanged; only spatial ownership is corrected.
    /// </summary>
    private int ApplyProjectiveBandAssignments(
        float maximumWeight,
        float weightedVoxelThreshold,
        bool enableStableOwnership,
        int stableSwitchConfirmations,
        int stablePendingMaxGapFrames,
        float stableLateralImprovementRatio,
        float stablePendingTsdfTolerance,
        float stablePendingSurfaceToleranceVoxels,
        HashSet<Vector3Int> frameDirty,
        out int stableOwnershipSeeds,
        out int stableOwnershipCompatibleWrites,
        out int stableOwnershipConflictCandidates,
        out int stableOwnershipHeldWrites,
        out int stableOwnershipPendingConfirmations,
        out int stableOwnershipConfirmedSwitches,
        out int saturatedWriteSkips,
        out int saturatedConflictSkips,
        out int weightLimitedWrites,
        out int cappedRecencyWrites,
        out int cappedConflictRecencyWrites,
        out int cappedConflictSignFlips)
    {
        stableOwnershipSeeds = 0;
        stableOwnershipCompatibleWrites = 0;
        stableOwnershipConflictCandidates = 0;
        stableOwnershipHeldWrites = 0;
        stableOwnershipPendingConfirmations = 0;
        stableOwnershipConfirmedSwitches = 0;
        saturatedWriteSkips = 0;
        saturatedConflictSkips = 0;
        weightLimitedWrites = 0;
        cappedRecencyWrites = 0;
        cappedConflictRecencyWrites = 0;
        cappedConflictSignFlips = 0;
        int writes = 0;
        foreach (KeyValuePair<int, ProjectiveAssignment> pair in
                 _frameProjectiveAssignments)
        {
            ProjectiveAssignment assignment = pair.Value;
            if (enableStableOwnership)
            {
                StableOwnershipDecision ownershipDecision =
                    EvaluateStableProjectiveOwnership(
                        pair.Key,
                        assignment,
                        stableSwitchConfirmations,
                        stablePendingMaxGapFrames,
                        stableLateralImprovementRatio,
                        stablePendingTsdfTolerance,
                        stablePendingSurfaceToleranceVoxels,
                        out bool continuedPending);
                if (ownershipDecision == StableOwnershipDecision.Seed)
                    stableOwnershipSeeds++;
                else if (ownershipDecision ==
                         StableOwnershipDecision.Compatible)
                    stableOwnershipCompatibleWrites++;
                else
                {
                    stableOwnershipConflictCandidates++;
                    if (continuedPending)
                        stableOwnershipPendingConfirmations++;
                    if (ownershipDecision == StableOwnershipDecision.Hold)
                    {
                        stableOwnershipHeldWrites++;
                        continue;
                    }
                    stableOwnershipConfirmedSwitches++;
                }
            }
            SparseBlock block = GetOrCreateBlock(
                assignment.X,
                assignment.Y,
                assignment.Z,
                out Vector3Int key,
                out int localIndex);
            float oldWeight = block.Weights[localIndex];
            float appliedWeight = Math.Min(
                assignment.ObservationWeight,
                maximumWeight);
            float newWeight;
            if (oldWeight + appliedWeight > maximumWeight)
            {
                // Mature capped TSDF semantics: confidence stops growing, but the estimate
                // continues to absorb recent observations. Make room for the full new sample
                // by forgetting the same amount of old confidence instead of rejecting it.
                float retainedOldWeight = Math.Max(
                    0f, maximumWeight - appliedWeight);
                bool oldNegative = block.Tsdf[localIndex] < 0f;
                bool signConflict =
                    oldNegative != (assignment.SampleTsdf < 0f);
                block.Tsdf[localIndex] =
                    (block.Tsdf[localIndex] * retainedOldWeight +
                     assignment.SampleTsdf * appliedWeight) /
                    Math.Max(
                        0.000001f,
                        retainedOldWeight + appliedWeight);
                newWeight = maximumWeight;
                weightLimitedWrites++;
                cappedRecencyWrites++;
                if (signConflict)
                {
                    cappedConflictRecencyWrites++;
                    if ((block.Tsdf[localIndex] < 0f) !=
                        oldNegative)
                        cappedConflictSignFlips++;
                }
            }
            else
            {
                newWeight = oldWeight + appliedWeight;
                block.Tsdf[localIndex] =
                    (block.Tsdf[localIndex] * oldWeight +
                     assignment.SampleTsdf * appliedWeight) /
                    Math.Max(0.000001f, newWeight);
            }
            block.Weights[localIndex] = newWeight;
            if (oldWeight < weightedVoxelThreshold &&
                newWeight >= weightedVoxelThreshold)
                _weightedVoxelCount++;
            frameDirty.Add(key);
            writes++;
        }
        return writes;
    }

    /// <summary>
    /// Prevents a single raw frame from changing a voxel's accepted TSDF sign. The challenger
    /// remains pending until repeated frames agree on its sign, normalized TSDF and world-space
    /// surface location. A clearly better ray needs the configured confirmations; an ambiguous
    /// ray needs one additional confirmation.
    /// </summary>
    private StableOwnershipDecision EvaluateStableProjectiveOwnership(
        int denseIndex,
        ProjectiveAssignment candidate,
        int switchConfirmations,
        int pendingMaxGapFrames,
        float lateralImprovementRatio,
        float pendingTsdfTolerance,
        float pendingSurfaceToleranceVoxels,
        out bool continuedPending)
    {
        continuedPending = false;
        if (!_stableProjectiveOwnership.TryGetValue(
                denseIndex, out StableProjectiveOwnership ownership))
        {
            _stableProjectiveOwnership.Add(
                denseIndex,
                CreateStableProjectiveOwnership(candidate));
            return StableOwnershipDecision.Seed;
        }

        bool candidateNegative = candidate.SampleTsdf < 0f;
        bool acceptedNegative = ownership.AcceptedTsdf < 0f;
        if (candidateNegative == acceptedNegative)
        {
            ownership.AcceptedSurfacePoint = candidate.SurfacePoint;
            ownership.AcceptedTsdf = candidate.SampleTsdf;
            ownership.AcceptedLateralDistanceSquared =
                candidate.LateralDistanceSquared;
            ownership.LastAcceptedFrameSequence = _projectiveFrameSequence;
            ownership.HasPending = false;
            ownership.PendingHits = 0;
            _stableProjectiveOwnership[denseIndex] = ownership;
            return StableOwnershipDecision.Compatible;
        }

        int frameGap = _projectiveFrameSequence -
                       ownership.LastPendingFrameSequence;
        float surfaceTolerance = _voxelSize *
                                 Math.Max(
                                     0.5f,
                                     pendingSurfaceToleranceVoxels);
        bool pendingCompatible =
            ownership.HasPending &&
            ownership.PendingNegative == candidateNegative &&
            frameGap >= 1 &&
            frameGap <= Math.Max(1, pendingMaxGapFrames) &&
            Math.Abs(ownership.PendingTsdf - candidate.SampleTsdf) <=
            Math.Max(0.05f, pendingTsdfTolerance) &&
            Vector3.SqrMagnitude(
                ownership.PendingSurfacePoint -
                candidate.SurfacePoint) <=
            surfaceTolerance * surfaceTolerance;
        if (pendingCompatible)
        {
            int oldHits = Math.Max(1, ownership.PendingHits);
            int newHits = oldHits + 1;
            float candidateShare = 1f / newHits;
            ownership.PendingSurfacePoint = Vector3.Lerp(
                ownership.PendingSurfacePoint,
                candidate.SurfacePoint,
                candidateShare);
            ownership.PendingTsdf = Lerp(
                ownership.PendingTsdf,
                candidate.SampleTsdf,
                candidateShare);
            ownership.PendingBestLateralDistanceSquared = Math.Min(
                ownership.PendingBestLateralDistanceSquared,
                candidate.LateralDistanceSquared);
            ownership.PendingHits = newHits;
            continuedPending = true;
        }
        else
        {
            ownership.HasPending = true;
            ownership.PendingNegative = candidateNegative;
            ownership.PendingSurfacePoint = candidate.SurfacePoint;
            ownership.PendingTsdf = candidate.SampleTsdf;
            ownership.PendingBestLateralDistanceSquared =
                candidate.LateralDistanceSquared;
            ownership.PendingHits = 1;
        }
        ownership.LastPendingFrameSequence = _projectiveFrameSequence;

        float improvementRatio = Clamp(
            lateralImprovementRatio, 0.1f, 1f);
        // Compare candidates observed by the same camera in the same raw frame. Comparing the
        // new ray's absolute lateral distance with a previous frame is not stable when the
        // camera moves. A missing runner-up means the winner was uncontested.
        bool clearlyBetterRay =
            !candidate.HasRunnerUp ||
            candidate.LateralDistanceSquared + 0.0000001f <
            candidate.RunnerUpLateralDistanceSquared *
            improvementRatio * improvementRatio;
        int requiredConfirmations =
            Math.Max(2, switchConfirmations) +
            (clearlyBetterRay ? 0 : 1);
        if (ownership.PendingHits < requiredConfirmations)
        {
            _stableProjectiveOwnership[denseIndex] = ownership;
            return StableOwnershipDecision.Hold;
        }

        ownership.AcceptedSurfacePoint = candidate.SurfacePoint;
        ownership.AcceptedTsdf = candidate.SampleTsdf;
        ownership.AcceptedLateralDistanceSquared =
            candidate.LateralDistanceSquared;
        ownership.LastAcceptedFrameSequence = _projectiveFrameSequence;
        ownership.HasPending = false;
        ownership.PendingHits = 0;
        _stableProjectiveOwnership[denseIndex] = ownership;
        return StableOwnershipDecision.Switch;
    }

    private StableProjectiveOwnership CreateStableProjectiveOwnership(
        ProjectiveAssignment assignment)
    {
        return new StableProjectiveOwnership
        {
            AcceptedSurfacePoint = assignment.SurfacePoint,
            AcceptedTsdf = assignment.SampleTsdf,
            AcceptedLateralDistanceSquared =
                assignment.LateralDistanceSquared,
            LastAcceptedFrameSequence = _projectiveFrameSequence,
            HasPending = false,
            PendingHits = 0
        };
    }

    /// <summary>
    /// Adds the missing projective free-space half of independent TSDF fusion without
    /// turning the experiment into aggressive whole-ray carving.
    ///
    /// The carve is deliberately conservative:
    /// - it stops one truncation band before the observed surface;
    /// - it only examines a bounded near-surface ray segment;
    /// - a voxel is updated at most once per raw frame;
    /// - unknown voxels are admitted only next to existing weighted surface state;
    /// - evidence uses a fractional weight, so a new publishable positive voxel needs
    ///   confirmation across multiple frames.
    ///
    /// This remains isolated shadow state. It cannot clear or publish authority voxels.
    /// </summary>
    private int IntegrateConservativeFreeSpace(
        Vector3 camera,
        Vector3 ray,
        float surfaceDepth,
        float observationWeight,
        float truncation,
        float minimumDepth,
        float maximumWeight,
        float weightedVoxelThreshold,
        float weightScale,
        float maximumDistanceMeters,
        HashSet<Vector3Int> frameDirty,
        HashSet<int> frameFreeSpaceVoxels,
        out int candidates,
        out int newWeightedVoxels,
        out int negativeUpdates,
        out int signFlips,
        out int rejectedNoSurfaceNeighbor)
    {
        candidates = 0;
        newWeightedVoxels = 0;
        negativeUpdates = 0;
        signFlips = 0;
        rejectedNoSurfaceNeighbor = 0;
        if (weightScale <= 0.0001f || maximumDistanceMeters <= 0f)
            return 0;

        float guard = Math.Max(truncation, _voxelSize * 1.5f);
        float endDepth = surfaceDepth - guard;
        float startDepth = Math.Max(
            Math.Max(0.01f, minimumDepth),
            endDepth - maximumDistanceMeters);
        if (endDepth <= startDepth)
            return 0;

        float step = Math.Max(_voxelSize * 0.75f, 0.001f);
        float appliedWeight = Math.Max(
            0.001f,
            Math.Min(maximumWeight, observationWeight * weightScale));
        int writes = 0;
        for (float depth = endDepth; depth >= startDepth; depth -= step)
        {
            Vector3 position = camera + ray * depth;
            Vector3 grid = (position - _origin) / _voxelSize;
            int x = (int)Math.Round(grid.x);
            int y = (int)Math.Round(grid.y);
            int z = (int)Math.Round(grid.z);
            if (x < 0 || y < 0 || z < 0 ||
                x >= _dimX || y >= _dimY || z >= _dimZ)
                continue;

            int denseIndex = x + _dimX * (y + _dimY * z);
            if (!frameFreeSpaceVoxels.Add(denseIndex))
                continue;
            candidates++;

            bool hadVoxel = TryGetVoxel(
                x, y, z, out float oldTsdf, out float oldWeight);
            if (oldWeight < weightedVoxelThreshold &&
                !HasWeightedSurfaceNeighbor(
                    x, y, z, weightedVoxelThreshold))
            {
                rejectedNoSurfaceNeighbor++;
                continue;
            }

            SparseBlock block = GetOrCreateBlock(
                x, y, z, out Vector3Int key, out int localIndex);
            bool oldWeighted = oldWeight >= weightedVoxelThreshold;
            bool oldNegative = oldWeighted && oldTsdf < 0f;
            float newWeight;
            float newTsdf;
            if (oldWeight + appliedWeight > maximumWeight)
            {
                float retainedOldWeight =
                    Math.Max(0f, maximumWeight - appliedWeight);
                newTsdf =
                    (oldTsdf * retainedOldWeight + appliedWeight) /
                    Math.Max(0.000001f, retainedOldWeight + appliedWeight);
                newWeight = maximumWeight;
            }
            else
            {
                newWeight = oldWeight + appliedWeight;
                newTsdf =
                    (oldTsdf * oldWeight + appliedWeight) /
                    Math.Max(0.000001f, newWeight);
            }
            block.Tsdf[localIndex] = Clamp(newTsdf, -1f, 1f);
            block.Weights[localIndex] = newWeight;
            bool newWeighted = newWeight >= weightedVoxelThreshold;
            if (!oldWeighted && newWeighted)
            {
                _weightedVoxelCount++;
                newWeightedVoxels++;
            }
            if (oldNegative)
            {
                negativeUpdates++;
                if (block.Tsdf[localIndex] >= 0f)
                    signFlips++;
            }
            frameDirty.Add(key);
            writes++;
        }
        return writes;
    }

    private bool HasWeightedSurfaceNeighbor(
        int x,
        int y,
        int z,
        float minimumWeight)
    {
        float threshold = Math.Max(0.0001f, minimumWeight);
        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
        {
            if (dx == 0 && dy == 0 && dz == 0)
                continue;
            if (TryGetVoxel(
                    x + dx, y + dy, z + dz,
                    out _, out float neighborWeight) &&
                neighborWeight >= threshold)
                return true;
        }
        return false;
    }

    private int CountSaturatedVoxels(float maximumWeight)
    {
        int count = 0;
        float threshold = Math.Max(1f, maximumWeight) - 0.000001f;
        foreach (SparseBlock block in _blocks.Values)
        for (int i = 0; i < block.Weights.Length; i++)
        {
            if (block.Weights[i] >= threshold)
                count++;
        }
        return count;
    }

    private SparseBlock GetOrCreateBlock(int x, int y, int z, out Vector3Int key, out int localIndex)
    {
        int bx = x / _blockVoxels;
        int by = y / _blockVoxels;
        int bz = z / _blockVoxels;
        key = new Vector3Int(bx, by, bz);
        if (!_blocks.TryGetValue(key, out SparseBlock block))
        {
            block = new SparseBlock(_blockVoxelCount);
            _blocks.Add(key, block);
        }
        int lx = x - bx * _blockVoxels;
        int ly = y - by * _blockVoxels;
        int lz = z - bz * _blockVoxels;
        localIndex = lx + _blockVoxels * (ly + _blockVoxels * lz);
        return block;
    }

    private bool TryGetVoxel(int x, int y, int z, out float tsdf, out float weight)
    {
        tsdf = 1f;
        weight = 0f;
        if (x < 0 || y < 0 || z < 0 || x >= _dimX || y >= _dimY || z >= _dimZ || _blockVoxels <= 0)
            return false;
        int bx = x / _blockVoxels;
        int by = y / _blockVoxels;
        int bz = z / _blockVoxels;
        if (!_blocks.TryGetValue(new Vector3Int(bx, by, bz), out SparseBlock block))
            return false;
        int lx = x - bx * _blockVoxels;
        int ly = y - by * _blockVoxels;
        int lz = z - bz * _blockVoxels;
        int localIndex = lx + _blockVoxels * (ly + _blockVoxels * lz);
        tsdf = block.Tsdf[localIndex];
        weight = block.Weights[localIndex];
        return true;
    }

    private int CountWeightedVoxels(float minimumWeight)
    {
        int count = 0;
        float threshold = Math.Max(0.0001f, minimumWeight);
        foreach (SparseBlock block in _blocks.Values)
        for (int i = 0; i < block.Weights.Length; i++)
        {
            if (block.Weights[i] >= threshold)
                count++;
        }
        return count;
    }

    private int CountZeroCrossEdges(float minimumWeight, HashSet<Vector3Int> restrictedBlocks)
    {
        HashSet<long> edges = BuildShadowZeroCrossEdges(minimumWeight, restrictedBlocks);
        return edges.Count;
    }

    private HashSet<long> BuildShadowZeroCrossEdges(float minimumWeight, HashSet<Vector3Int> restrictedBlocks = null)
    {
        HashSet<long> edges = new HashSet<long>();
        float threshold = Math.Max(0.0001f, minimumWeight);
        foreach (KeyValuePair<Vector3Int, SparseBlock> pair in _blocks)
        {
            if (restrictedBlocks != null && !restrictedBlocks.Contains(pair.Key))
                continue;
            Vector3Int key = pair.Key;
            SparseBlock block = pair.Value;
            for (int lz = 0; lz < _blockVoxels; lz++)
            for (int ly = 0; ly < _blockVoxels; ly++)
            for (int lx = 0; lx < _blockVoxels; lx++)
            {
                int x = key.x * _blockVoxels + lx;
                int y = key.y * _blockVoxels + ly;
                int z = key.z * _blockVoxels + lz;
                if (x >= _dimX || y >= _dimY || z >= _dimZ)
                    continue;
                int local = lx + _blockVoxels * (ly + _blockVoxels * lz);
                if (block.Weights[local] < threshold)
                    continue;
                float value = block.Tsdf[local];
                AddShadowCrossing(edges, x, y, z, 0, x + 1, y, z, value, threshold);
                AddShadowCrossing(edges, x, y, z, 1, x, y + 1, z, value, threshold);
                AddShadowCrossing(edges, x, y, z, 2, x, y, z + 1, value, threshold);
            }
        }
        return edges;
    }

    private void AddShadowCrossing(
        HashSet<long> edges, int x, int y, int z, int axis,
        int nx, int ny, int nz, float value, float minimumWeight)
    {
        if (!TryGetVoxel(nx, ny, nz, out float neighborValue, out float neighborWeight) || neighborWeight < minimumWeight)
            return;
        if ((value < 0f) == (neighborValue < 0f))
            return;
        edges.Add(EdgeKey(x, y, z, axis));
    }

    private HashSet<long> BuildAuthorityZeroCrossEdges(float[] tsdf, byte[] weights, int minimumWeight)
    {
        HashSet<long> edges = new HashSet<long>();
        if (tsdf == null || weights == null)
            return edges;
        int count = Math.Min(tsdf.Length, weights.Length);
        int threshold = Math.Max(1, minimumWeight);
        for (int z = 0; z < _dimZ; z++)
        for (int y = 0; y < _dimY; y++)
        for (int x = 0; x < _dimX; x++)
        {
            int index = DenseIndex(x, y, z);
            if (index < 0 || index >= count || weights[index] < threshold)
                continue;
            float value = tsdf[index];
            AddAuthorityCrossing(edges, tsdf, weights, count, threshold, x, y, z, 0, x + 1, y, z, value);
            AddAuthorityCrossing(edges, tsdf, weights, count, threshold, x, y, z, 1, x, y + 1, z, value);
            AddAuthorityCrossing(edges, tsdf, weights, count, threshold, x, y, z, 2, x, y, z + 1, value);
        }
        return edges;
    }

    private void AddAuthorityCrossing(
        HashSet<long> edges, float[] tsdf, byte[] weights, int count, int minimumWeight,
        int x, int y, int z, int axis, int nx, int ny, int nz, float value)
    {
        if (nx < 0 || ny < 0 || nz < 0 || nx >= _dimX || ny >= _dimY || nz >= _dimZ)
            return;
        int neighbor = DenseIndex(nx, ny, nz);
        if (neighbor < 0 || neighbor >= count || weights[neighbor] < minimumWeight)
            return;
        if ((value < 0f) != (tsdf[neighbor] < 0f))
            edges.Add(EdgeKey(x, y, z, axis));
    }

    private long EdgeKey(int x, int y, int z, int axis)
    {
        return ((long)DenseIndex(x, y, z) * 3L) + axis;
    }

    private int DenseIndex(int x, int y, int z)
    {
        return x + _dimX * (y + _dimY * z);
    }

    private void IndexToCoordinates(int index, out int x, out int y, out int z)
    {
        x = index % _dimX;
        int yz = index / _dimX;
        y = yz % _dimY;
        z = yz / _dimY;
    }

    private static bool IsPlaneLike(
        int x,
        int y,
        int width,
        int height,
        int expected,
        Vector3[] positions,
        Vector3[] normals,
        float minimumNormalDot,
        float maximumResidual,
        int minimumNeighbors)
    {
        int centerIndex = y * width + x;
        if (centerIndex < 0 || centerIndex >= expected || normals == null || centerIndex >= normals.Length)
            return false;
        Vector3 center = positions[centerIndex];
        Vector3 normal = normals[centerIndex];
        if (!Finite(center) || !Finite(normal) || normal.sqrMagnitude <= 0.000001f)
            return false;
        normal.Normalize();
        int compatible = 0;
        for (int i = 0; i < PlaneNeighborOffsetX.Length; i++)
        {
            int nx = x + PlaneNeighborOffsetX[i];
            int ny = y + PlaneNeighborOffsetY[i];
            if (nx < 0 || ny < 0 || nx >= width || ny >= height)
                continue;
            int neighborIndex = ny * width + nx;
            if (neighborIndex < 0 || neighborIndex >= expected || neighborIndex >= normals.Length)
                continue;
            Vector3 neighborPoint = positions[neighborIndex];
            Vector3 neighborNormal = normals[neighborIndex];
            if (!Finite(neighborPoint) || !Finite(neighborNormal) || neighborNormal.sqrMagnitude <= 0.000001f)
                continue;
            float normalDot = Math.Abs(Vector3.Dot(normal, neighborNormal.normalized));
            float residual = Math.Abs(Vector3.Dot(neighborPoint - center, normal));
            if (normalDot >= minimumNormalDot && residual <= maximumResidual)
                compatible++;
        }
        return compatible >= Math.Max(1, minimumNeighbors);
    }

    private static bool Finite(Vector3 value)
    {
        return !(float.IsNaN(value.x) || float.IsInfinity(value.x) ||
                 float.IsNaN(value.y) || float.IsInfinity(value.y) ||
                 float.IsNaN(value.z) || float.IsInfinity(value.z));
    }

    private static float Clamp01(float value) => Clamp(value, 0f, 1f);
    private static float Clamp(float value, float minimum, float maximum) => Math.Max(minimum, Math.Min(maximum, value));
    private static int Clamp(int value, int minimum, int maximum) => Math.Max(minimum, Math.Min(maximum, value));
    private static float Lerp(float a, float b, float t) => a + (b - a) * Clamp01(t);
    private static float InverseLerp(float a, float b, float value) => Math.Abs(b - a) <= 0.000001f ? 0f : Clamp01((value - a) / (b - a));
}
