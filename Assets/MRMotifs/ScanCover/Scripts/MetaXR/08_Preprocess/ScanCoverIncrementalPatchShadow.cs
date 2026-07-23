using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;

/// <summary>
/// Parallel, display-only patch publisher. It observes the existing global mesh candidate,
/// keeps local geometry across rebuilds, and publishes only patches that survive probation.
/// It never writes the production TSDF or production mesh.
/// </summary>
public sealed class ScanCoverIncrementalPatchShadow
{
    public enum PatchLifecycle
    {
        Accumulating,
        Provisional,
        CorrectionPending,
        Mature,
        Retired
    }

    public enum PatchSurfaceKind
    {
        Uncertain,
        Planar,
        Curved
    }

    public enum PatchDirtyReason
    {
        None,
        New,
        Provisional,
        GeometryChanged,
        Missing
    }

    public struct Settings
    {
        public bool DoubleSidedTriangles;
        // The TSDF zero-crossing tracker owns dirty-region discovery in the mature route.
        // Patch probation remains diagnostic state and must not recursively dirty itself.
        public bool ExternalDirtyIsAuthoritative;
        // Publish or roll back a TSDF-dirty spatial block as one unit. Triangle identity,
        // family migration, and cross-round probation do not participate in this mode.
        public bool DirectAtomicBlockPublication;
        public int LocalExtractionPaddingPatches;
        public int CorrectionConfirmRebuilds;
        public int CorrectionTrialRebuilds;
        public float CorrectionMaxResidualRegressionMeters;
        public float CorrectionMaxCoherenceRegression;
        public float PatchSizeMeters;
        public int ConfirmRebuilds;
        public int ReplacementConfirmRebuilds;
        public int RetireMissingRebuilds;
        public int MaxPatches;
        public int MaxTrianglesPerPatch;
        public float StableCentroidDistanceMeters;
        public float StableNormalDot;
        public float StableTriangleRatio;
        public float PlanarResidualMeters;
        public float PlanarNormalCoherence;
        public float ReplacementResidualSlackMeters;
        public float ReplacementBoundarySlackRatio;
        public IReadOnlyList<ScanCoverStablePlaneRegistry.PlaneConstraint> MaturePlaneConstraints;
        public float PlanePatchMinResidualMeters;
        public float PlanePatchMinNormalDot;
        public float PlanePatchMaxCentroidDistanceMeters;
        public float PlanePatchTangentialPaddingMeters;
        public float PlanePatchMaxVertexMoveMeters;
        public float PlanePatchMinTriangleAreaRatio;
        public float PlanePatchMinTriangleNormalDot;
        public float PlanePatchAmbiguityScoreMargin;
        public int PlaneRejectMaxHoldRebuilds;
        public float PlaneRejectWithdrawResidualMeters;
    }

    public struct Result
    {
        public int RebuildSequence;
        public int CandidatePatches;
        public int CandidateTriangles;
        public int NewPatches;
        public int ProvisionalPatches;
        public int MaturePatches;
        public int MaturePlanarPatches;
        public int MatureCurvedPatches;
        public int MatureUncertainPatches;
        public int PublishedTriangles;
        public int PublishedThisRebuild;
        public int ReplacedThisRebuild;
        public int HeldMaturePatches;
        public int RetiredThisRebuild;
        public int AtomicRollbackCount;
        public int CandidateBoundaryEdges;
        public int PublishedBoundaryEdges;
        public int CandidateNonManifoldEdges;
        public int PublishedNonManifoldEdges;
        public int DirtyPatches;
        public int DirtyNewPatches;
        public int DirtyProvisionalPatches;
        public int DirtyChangedPatches;
        public int DirtyMissingPatches;
        public int DirtyTriangles;
        public int LocalExtractionPatches;
        public int LocalExtractionTriangles;
        public int ReadOnlySeamAnchorPatches;
        public int ReadOnlySeamAnchorTriangles;
        public int CleanReusedPatches;
        public int DirtyClusterCount;
        public int LargestDirtyClusterPatches;
        public int LargestLocalClusterPatches;
        public int CorrectionPendingPatches;
        public int CorrectionUpgradedThisRebuild;
        public int CorrectionCommittedThisRebuild;
        public int CorrectionRollbackThisRebuild;
        public int PlaneConstraintCandidatePatches;
        public int PlaneConstraintAppliedPatches;
        public int PlaneConstraintAppliedTriangles;
        public int PlaneConstraintRejectedNormal;
        public int PlaneConstraintRejectedDistance;
        public int PlaneConstraintRejectedExtent;
        public int PlaneConstraintRejectedAmbiguity;
        public int PlaneConstraintRejectedMove;
        public int PlaneConstraintRejectedTopology;
        public int PlaneConstraintMoveSplitRescuedPatches;
        public int PlaneConstraintQuarantinedTriangles;
        public int PlaneConstraintMultiIslandRescuedPatches;
        public int PlaneConstraintRetainedIslands;
        public int PlaneRejectLeasePatches;
        public int PlaneRejectForcedLocalPatches;
        public int PlaneRejectWithheldNewPatches;
        public int PlaneRejectWithdrawnPatches;
        public int PlaneRejectRecoveredPatches;
        public int NormalDecompositionCandidatePatches;
        public int NormalDecompositionSinglePlanePatches;
        public int NormalDecompositionMultiPlanePatches;
        public int NormalDecompositionFragmentedPatches;
        public int NormalDecompositionUnresolvedPatches;
        public int NormalDecompositionMatchedTriangles;
        public int NormalDecompositionUnresolvedTriangles;
        public int NormalDecompositionWindingMinorityTriangles;
        public int NormalDecompositionIslands;
        public int NormalDecompositionParallelLayerPatches;
        public int NormalDecompositionCreasePatches;
        public int NormalDecompositionMixedMultiPatches;
        public int NormalDecompositionUnmatchedNormalTriangles;
        public int NormalDecompositionUnmatchedDistanceTriangles;
        public int NormalDecompositionUnmatchedExtentTriangles;
        public int NormalDecompositionUnmatchedMoveTriangles;
        public int NormalDecompositionUnmatchedAmbiguityTriangles;
        public int CreaseAtomicCandidatePatches;
        public int CreaseAtomicAcceptedPatches;
        public int CreaseAtomicRejectedCoveragePatches;
        public int CreaseAtomicRejectedFamilyPatches;
        public int CreaseAtomicRejectedTopologyPatches;
        public int CreaseAtomicRetainedTriangles;
        public int CreaseAtomicQuarantinedTriangles;
        public int CreaseAtomicTopologyRejectedTriangles;
        public int CreaseAtomicRetainedFamilies;
        public int CreaseAtomicRetainedIslands;
        public int CreaseAtomicTinyIslandTriangles;
        public int CreaseAtomicTinyIslands;
        public int CreaseAtomicNearNormalTriangles;
        public int CreaseAtomicNovelNormalTriangles;
        public int CreaseAtomicNearDistanceTriangles;
        public int CreaseAtomicFarDistanceTriangles;
        public int CreaseAtomicNearMoveTriangles;
        public int CreaseAtomicFarMoveTriangles;
        public int CreaseAtomicPersistentTinyIslandPatches;
        public int CreaseAtomicProbationPendingPatches;
        public int CreaseAtomicProbationReadyPatches;
        public int CreaseAtomicProbationResetPatches;
        public int CreaseTransitionCandidatePatches;
        public int CreaseTransitionStablePatches;
        public int CreaseTransitionSameFamilyDistancePatches;
        public int CreaseTransitionNewFamilyDistancePatches;
        public int CreaseTransitionNormalGatePatches;
        public int CreaseTransitionFamilyChangedPatches;
        public int CreaseTransitionFamilyLostPatches;
        public int CreaseTransitionCoverageDropPatches;
        public int CreaseTransitionContentShiftPatches;
        public int CreaseTransitionOtherPatches;
        public int CreaseTransitionCanonicalRemapPatches;
        public int CreaseRegionalCandidatePatches;
        public int CreaseRegionalAggregatedPatches;
        public int CreaseRegionalMigratedPatches;
        public int CreaseRegionalReadyPatches;
        public bool HasDirtyBounds;
        public Vector3Int DirtyBoundsMin;
        public Vector3Int DirtyBoundsMax;
    }

    public struct Snapshot
    {
        public Vector3Int Key;
        public PatchLifecycle Lifecycle;
        public PatchSurfaceKind SurfaceKind;
        public int StableHits;
        public int ReplacementHits;
        public int MissingRebuilds;
        public int CandidateTriangles;
        public int PublishedTriangles;
        public int CandidateBoundaryEdges;
        public int PublishedBoundaryEdges;
        public int CandidateNonManifoldEdges;
        public int PublishedNonManifoldEdges;
        public float CandidatePlaneResidualMeters;
        public float PublishedPlaneResidualMeters;
        public float CandidateNormalCoherence;
        public float PublishedNormalCoherence;
        public PatchDirtyReason DirtyReason;
        public bool InLocalExtractionBounds;
        public bool ReusedCleanPublished;
        public bool ReadOnlySeamAnchor;
        public int DirtyClusterId;
        public int CorrectionHits;
        public int CorrectionTrialRemaining;
        public bool HasRollbackBackup;
        public int PlaneConstraintId;
        public bool PlaneConstraintApplied;
        public string PlaneConstraintRejectReason;
        public float PlaneConstraintNormalDot;
        public float PlaneConstraintCentroidDistanceMeters;
        public int PlaneConstraintNearestNormalPlaneId;
        public float PlaneConstraintBestNormalDot;
        public float PlaneConstraintNearestNormalDistanceMeters;
        public float PlaneConstraintDistanceExcessMeters;
        public float PlaneConstraintMaxMoveMeters;
        public float PlaneResidualBeforeConstraintMeters;
        public int PlaneConstraintQuarantinedTriangles;
        public int PlaneConstraintRetainedIslands;
        public int PlaneRejectHits;
        public string PlaneRejectLeaseReason;
        public bool PlaneRejectForcedLocal;
        public bool PlaneRejectWithheld;
        public string NormalDecompositionKind;
        public int NormalDecompositionDominantPlaneId;
        public int NormalDecompositionPlaneFamilies;
        public int NormalDecompositionIslands;
        public int NormalDecompositionMatchedTriangles;
        public int NormalDecompositionUnresolvedTriangles;
        public int NormalDecompositionLargestIslandTriangles;
        public int NormalDecompositionWindingMinorityTriangles;
        public string NormalDecompositionSubtype;
        public string NormalDecompositionPlaneIds;
        public float NormalDecompositionMinFamilyNormalDot;
        public float NormalDecompositionMaxPlaneSeparationMeters;
        public int NormalDecompositionUnmatchedNormalTriangles;
        public int NormalDecompositionUnmatchedDistanceTriangles;
        public int NormalDecompositionUnmatchedExtentTriangles;
        public int NormalDecompositionUnmatchedMoveTriangles;
        public int NormalDecompositionUnmatchedAmbiguityTriangles;
        public string CreaseAtomicStatus;
        public int CreaseAtomicRetainedTriangles;
        public int CreaseAtomicQuarantinedTriangles;
        public int CreaseAtomicTopologyRejectedTriangles;
        public int CreaseAtomicRetainedFamilies;
        public int CreaseAtomicRequiredFamilies;
        public int CreaseAtomicRetainedIslands;
        public float CreaseAtomicRetainedRatio;
        public float CreaseAtomicMaxMoveMeters;
        public int CreaseAtomicTinyIslandTriangles;
        public int CreaseAtomicTinyIslands;
        public int CreaseAtomicTinyIslandStableHits;
        public string CreaseAtomicFamilyIds;
        public int CreaseAtomicProbationHits;
        public bool CreaseAtomicProbationReady;
        public string CreaseTransitionKind;
        public string CreaseTransitionFromStatus;
        public string CreaseTransitionFromFamilyIds;
        public bool CreaseTransitionNearestInPriorFamily;
        public float CreaseTransitionTriangleRatio;
        public float CreaseTransitionCentroidShiftMeters;
        public float CreaseTransitionNormalDot;
        public float CreaseTransitionRetainedRatioDelta;
        public bool CreaseTransitionContentCompatible;
        public float CreaseTransitionFamilyMinNormalDot;
        public float CreaseTransitionFamilyMaxDistanceDeltaMeters;
        public float CreaseTransitionFamilyMinTriangleRatio;
        public int CreaseTransitionFamilyRemappedCount;
        public int CreaseRegionalMemberPatches;
        public float CreaseRegionalRetainedRatio;
        public string CreaseRegionalStatus;
        public int CreaseRegionalRetainedFamilies;
        public int CreaseRegionalRequiredFamilies;
        public bool CreaseRegionalMigrated;
        public Vector3Int CreaseRegionalSourceKey;
        public int NormalDecompositionNearNormalTriangles;
        public int NormalDecompositionNovelNormalTriangles;
        public int NormalDecompositionNearDistanceTriangles;
        public int NormalDecompositionFarDistanceTriangles;
        public int NormalDecompositionNearMoveTriangles;
        public int NormalDecompositionFarMoveTriangles;
    }

    private sealed class PatchGeometry
    {
        public readonly List<Vector3> Vertices = new List<Vector3>(256);
        public readonly List<int> Triangles = new List<int>(768);
        public readonly List<Color> Colors = new List<Color>(256);
        public readonly Dictionary<int, int> Remap = new Dictionary<int, int>(256);
        public Vector3 Centroid;
        public Vector3 AverageNormal;
        public Vector3 BoundsMin;
        public Vector3 BoundsMax;
        public float PlaneResidualMeters;
        public float NormalCoherence;
        public int BoundaryEdges;
        public int NonManifoldEdges;
        public PatchSurfaceKind SurfaceKind;
        public int PlaneConstraintId = -1;
        public bool PlaneConstraintApplied;
        public string PlaneConstraintRejectReason = "none";
        public float PlaneConstraintNormalDot;
        public float PlaneConstraintCentroidDistanceMeters;
        public int PlaneConstraintNearestNormalPlaneId = -1;
        public float PlaneConstraintBestNormalDot;
        public float PlaneConstraintNearestNormalDistanceMeters;
        public float PlaneConstraintDistanceExcessMeters;
        public float PlaneConstraintMaxMoveMeters;
        public float PlaneResidualBeforeConstraintMeters;
        public int PlaneConstraintQuarantinedTriangles;
        public int PlaneConstraintOriginalTriangles;
        public int PlaneConstraintRetainedIslands;
        public string NormalDecompositionKind = "none";
        public int NormalDecompositionDominantPlaneId = -1;
        public int NormalDecompositionPlaneFamilies;
        public int NormalDecompositionIslands;
        public int NormalDecompositionMatchedTriangles;
        public int NormalDecompositionUnresolvedTriangles;
        public int NormalDecompositionLargestIslandTriangles;
        public int NormalDecompositionWindingMinorityTriangles;
        public string NormalDecompositionSubtype = "none";
        public string NormalDecompositionPlaneIds = "";
        public float NormalDecompositionMinFamilyNormalDot = 1f;
        public float NormalDecompositionMaxPlaneSeparationMeters;
        public int NormalDecompositionUnmatchedNormalTriangles;
        public int NormalDecompositionUnmatchedDistanceTriangles;
        public int NormalDecompositionUnmatchedExtentTriangles;
        public int NormalDecompositionUnmatchedMoveTriangles;
        public int NormalDecompositionUnmatchedAmbiguityTriangles;
        public string CreaseAtomicStatus = "none";
        public int CreaseAtomicRetainedTriangles;
        public int CreaseAtomicQuarantinedTriangles;
        public int CreaseAtomicTopologyRejectedTriangles;
        public int CreaseAtomicRetainedFamilies;
        public int CreaseAtomicRequiredFamilies;
        public int CreaseAtomicRetainedIslands;
        public float CreaseAtomicRetainedRatio;
        public float CreaseAtomicMaxMoveMeters;
        public int CreaseAtomicTinyIslandTriangles;
        public int CreaseAtomicTinyIslands;
        public string CreaseAtomicFamilyIds = "";
        public string CreaseTransitionKind = "none";
        public string CreaseTransitionFromStatus = "none";
        public string CreaseTransitionFromFamilyIds = "";
        public bool CreaseTransitionNearestInPriorFamily;
        public float CreaseTransitionTriangleRatio;
        public float CreaseTransitionCentroidShiftMeters;
        public float CreaseTransitionNormalDot;
        public float CreaseTransitionRetainedRatioDelta;
        public bool CreaseTransitionContentCompatible;
        public float CreaseTransitionFamilyMinNormalDot;
        public float CreaseTransitionFamilyMaxDistanceDeltaMeters;
        public float CreaseTransitionFamilyMinTriangleRatio;
        public int CreaseTransitionFamilyRemappedCount;
        public int CreaseRegionalMemberPatches = 1;
        public float CreaseRegionalRetainedRatio;
        public string CreaseRegionalStatus = "none";
        public int CreaseRegionalRetainedFamilies;
        public int CreaseRegionalRequiredFamilies;
        public bool CreaseRegionalMigrated;
        public Vector3Int CreaseRegionalSourceKey;
        public readonly List<CreaseFamilyEvidence> CreaseAtomicFamilies =
            new List<CreaseFamilyEvidence>(4);
        public int NormalDecompositionNearNormalTriangles;
        public int NormalDecompositionNovelNormalTriangles;
        public int NormalDecompositionNearDistanceTriangles;
        public int NormalDecompositionFarDistanceTriangles;
        public int NormalDecompositionNearMoveTriangles;
        public int NormalDecompositionFarMoveTriangles;

        public int TriangleCount => Triangles.Count / 3;
    }

    private sealed class CreaseFamilyEvidence
    {
        public int PlaneId;
        public Vector3 Normal;
        public float Offset;
        public int RetainedTriangles;
    }

    private struct CreaseFamilyMatch
    {
        public int PriorIndex;
        public int CandidateIndex;
        public float NormalDot;
        public float DistanceDeltaMeters;
        public float Score;
    }

    private sealed class Track
    {
        public Vector3Int Key;
        public PatchLifecycle Lifecycle;
        public PatchGeometry LastCandidate;
        public PatchGeometry Published;
        public int StableHits;
        public int ReplacementHits;
        public int MissingRebuilds;
        public int FirstSeenRebuild;
        public int LastSeenRebuild;
        public PatchGeometry CorrectionCandidate;
        public PatchGeometry RollbackPublished;
        public int CorrectionTrialRemaining;
        public int PlaneRejectHits;
        public string PlaneRejectLeaseReason = "none";
        public bool PlaneRejectWithheld;
        public int CreaseAtomicTinyIslandStableHits;
        public string CreaseAtomicProbationFamilyIds = "";
        public int CreaseAtomicProbationHits;
        public int CreaseAtomicProbationRetainedTriangles;
        public PatchGeometry CreaseAtomicProbationGeometry;
        public Vector3Int CreaseAtomicProbationOriginKey;
        public PatchGeometry LastCreaseRegionalEvidence;
    }

    private sealed class CreaseProbationSeed
    {
        public Vector3Int Key;
        public PatchGeometry Geometry;
        public string FamilyIds;
        public int Hits;
        public int RetainedTriangles;
        public bool Claimed;
    }

    private sealed class CreaseProbationAssignment
    {
        public Vector3Int SourceKey;
        public PatchGeometry Geometry;
        public string FamilyIds;
        public int Hits;
        public int RetainedTriangles;
    }

    private readonly Dictionary<Vector3Int, Track> _tracks = new Dictionary<Vector3Int, Track>(256);
    private readonly Dictionary<Vector3Int, PatchGeometry> _candidates = new Dictionary<Vector3Int, PatchGeometry>(256);
    private readonly HashSet<Vector3Int> _observed = new HashSet<Vector3Int>();
    private readonly Dictionary<Vector3Int, PatchDirtyReason> _dirtyReasons =
        new Dictionary<Vector3Int, PatchDirtyReason>(128);
    private readonly HashSet<Vector3Int> _localExtractionKeys = new HashSet<Vector3Int>();
    private readonly HashSet<Vector3Int> _cleanReusedKeys = new HashSet<Vector3Int>();
    private readonly HashSet<Vector3Int> _readOnlySeamAnchorKeys = new HashSet<Vector3Int>();
    private readonly Dictionary<Vector3Int, int> _localClusterIds = new Dictionary<Vector3Int, int>(256);
    private readonly Dictionary<Vector3Int, PatchGeometry> _creaseRegionalEvidence =
        new Dictionary<Vector3Int, PatchGeometry>(64);
    private readonly Dictionary<Vector3Int, CreaseProbationAssignment> _creaseProbationAssignments =
        new Dictionary<Vector3Int, CreaseProbationAssignment>(64);
    private static readonly Vector3Int[] SixConnectedOffsets =
    {
        new Vector3Int(1, 0, 0), new Vector3Int(-1, 0, 0),
        new Vector3Int(0, 1, 0), new Vector3Int(0, -1, 0),
        new Vector3Int(0, 0, 1), new Vector3Int(0, 0, -1)
    };
    private int _rebuildSequence;

    public Result Update(
        List<Vector3> sourceVertices,
        List<int> sourceTriangles,
        List<Color> sourceColors,
        Vector3 origin,
        Settings settings,
        IReadOnlyDictionary<Vector3Int, PatchDirtyReason> externalSurfaceDirtyReasons,
        out List<Vector3> publishedVertices,
        out List<int> publishedTriangles,
        out List<Color> publishedColors)
    {
        Sanitize(ref settings);
        _rebuildSequence++;
        _candidates.Clear();
        _observed.Clear();
        _dirtyReasons.Clear();
        _localExtractionKeys.Clear();
        _cleanReusedKeys.Clear();
        _readOnlySeamAnchorKeys.Clear();
        _localClusterIds.Clear();
        _creaseRegionalEvidence.Clear();
        _creaseProbationAssignments.Clear();

        Result result = new Result { RebuildSequence = _rebuildSequence };
        BuildCandidates(sourceVertices, sourceTriangles, sourceColors, origin, settings, ref result);
        BuildDirtyAndLocalExtractionSets(settings, externalSurfaceDirtyReasons, ref result);
        if (settings.DirectAtomicBlockPublication)
            return UpdateDirectAtomicBlocks(settings, externalSurfaceDirtyReasons,
                out publishedVertices, out publishedTriangles, out publishedColors, ref result);
        BuildCreaseRegionalEvidence(settings, ref result);
        BuildCreaseProbationAssignments(settings, ref result);

        foreach (KeyValuePair<Vector3Int, PatchGeometry> pair in _candidates)
        {
            Vector3Int key = pair.Key;
            PatchGeometry candidate = pair.Value;
            bool hasRegionalEvidence = _creaseRegionalEvidence.TryGetValue(key, out PatchGeometry regional);
            PatchGeometry probationCandidate = hasRegionalEvidence
                ? regional
                : candidate;
            _observed.Add(key);
            if (!_tracks.TryGetValue(key, out Track track))
            {
                track = new Track
                {
                    Key = key,
                    Lifecycle = PatchLifecycle.Accumulating,
                    StableHits = 1,
                    FirstSeenRebuild = _rebuildSequence,
                    LastSeenRebuild = _rebuildSequence,
                    LastCandidate = CloneGeometry(candidate),
                    CreaseAtomicTinyIslandStableHits = candidate.CreaseAtomicTinyIslandTriangles > 0 ? 1 : 0,
                    CreaseAtomicProbationFamilyIds = probationCandidate.CreaseAtomicStatus == "accepted"
                        ? probationCandidate.CreaseAtomicFamilyIds
                        : "",
                    CreaseAtomicProbationHits = 0,
                    CreaseAtomicProbationRetainedTriangles = 0,
                    CreaseAtomicProbationOriginKey = key
                };
                _tracks.Add(key, track);
                ApplyCreaseProbationAssignment(track, key, probationCandidate, ref result);
                CaptureCreaseRegionalDiagnostics(track, probationCandidate, hasRegionalEvidence);
                UpdateCreaseAtomicProbation(track, probationCandidate, settings, ref result);
                UpdatePlaneRejectLease(track, candidate, settings, ref result);
                result.NewPatches++;
                continue;
            }

            AnalyzeCreaseTransition(track.LastCandidate, candidate, settings, ref result);
            UpdateCreaseTinyIslandEvidence(track, candidate, settings, ref result);
            ApplyCreaseProbationAssignment(track, key, probationCandidate, ref result);
            CaptureCreaseRegionalDiagnostics(track, probationCandidate, hasRegionalEvidence);
            UpdateCreaseAtomicProbation(track, probationCandidate, settings, ref result);
            UpdatePlaneRejectLease(track, candidate, settings, ref result);

            if (track.CorrectionTrialRemaining > 0)
            {
                track.MissingRebuilds = 0;
                track.LastSeenRebuild = _rebuildSequence;
                track.StableHits = GeometryCompatible(track.LastCandidate, candidate, settings)
                    ? track.StableHits + 1
                    : 1;
                track.LastCandidate = CloneGeometry(candidate);
                bool trialSupported = GeometrySafeForPublication(candidate) &&
                                      GeometryCompatible(track.Published, candidate, settings);
                if (trialSupported)
                {
                    track.CorrectionTrialRemaining--;
                    if (track.CorrectionTrialRemaining <= 0)
                    {
                        track.RollbackPublished = null;
                        track.CorrectionCandidate = null;
                        track.ReplacementHits = 0;
                        track.Lifecycle = PatchLifecycle.Mature;
                        result.CorrectionCommittedThisRebuild++;
                    }
                    else
                    {
                        track.Lifecycle = PatchLifecycle.CorrectionPending;
                    }
                }
                else
                {
                    RollbackCorrection(track, ref result);
                }
                result.HeldMaturePatches++;
                continue;
            }

            // Clean and seam-anchor patches remain immutable for this rebuild, but
            // a persistent, safe mismatch can revoke that lease and promote the
            // patch into the next rebuild's dirty core.
            if (!_localExtractionKeys.Contains(key) && track.Published != null)
            {
                track.MissingRebuilds = 0;
                track.LastSeenRebuild = _rebuildSequence;
                track.StableHits = GeometryCompatible(track.LastCandidate, candidate, settings)
                    ? track.StableHits + 1
                    : 1;
                track.LastCandidate = CloneGeometry(candidate);
                bool correctionCandidate = !GeometryCompatible(track.Published, candidate, settings) &&
                                           CorrectionCandidateAcceptable(track.Published, candidate, settings);
                UpdateCorrectionEvidence(track, candidate, correctionCandidate, settings);
                track.Lifecycle = track.ReplacementHits > 0
                    ? PatchLifecycle.CorrectionPending
                    : PatchLifecycle.Mature;
                if (!_readOnlySeamAnchorKeys.Contains(key))
                {
                    _cleanReusedKeys.Add(key);
                    result.CleanReusedPatches++;
                }
                result.HeldMaturePatches++;
                continue;
            }

            track.MissingRebuilds = 0;
            track.LastSeenRebuild = _rebuildSequence;
            bool stableWithPrior = GeometryCompatible(track.LastCandidate, candidate, settings);
            track.StableHits = stableWithPrior ? track.StableHits + 1 : 1;
            track.LastCandidate = CloneGeometry(candidate);

            if (ShouldWithdrawRejectedPublication(track, candidate, settings))
            {
                track.Published = null;
                track.RollbackPublished = null;
                track.CorrectionCandidate = null;
                track.CorrectionTrialRemaining = 0;
                track.ReplacementHits = 0;
                track.PlaneRejectWithheld = true;
                track.Lifecycle = PatchLifecycle.Provisional;
                result.PlaneRejectWithdrawnPatches++;
                continue;
            }

            if (track.Published == null)
            {
                track.Lifecycle = track.StableHits >= 2
                    ? PatchLifecycle.Provisional
                    : PatchLifecycle.Accumulating;
                bool rejectedHighResidual = RequiresRejectedPlaneHold(candidate, settings);
                if (track.StableHits >= settings.ConfirmRebuilds &&
                    GeometrySafeForPublication(candidate) &&
                    !rejectedHighResidual)
                {
                    track.Published = CloneGeometry(candidate);
                    track.PlaneRejectWithheld = false;
                    track.Lifecycle = PatchLifecycle.Mature;
                    track.ReplacementHits = 0;
                    result.PublishedThisRebuild++;
                }
                else if (rejectedHighResidual)
                {
                    track.PlaneRejectWithheld = true;
                    result.PlaneRejectWithheldNewPatches++;
                }
                continue;
            }

            track.Lifecycle = PatchLifecycle.Mature;
            if (GeometryCompatible(track.Published, candidate, settings))
            {
                track.ReplacementHits = 0;
                track.CorrectionCandidate = null;
                result.HeldMaturePatches++;
                continue;
            }

            bool replacementSafe = CorrectionCandidateAcceptable(
                track.Published,
                candidate,
                settings);
            UpdateCorrectionEvidence(track, candidate, replacementSafe, settings);
            if (replacementSafe && track.ReplacementHits >= settings.CorrectionConfirmRebuilds)
            {
                StartCorrectionTrial(track, candidate, settings, ref result);
                result.ReplacedThisRebuild++;
            }
            else
            {
                track.Lifecycle = track.ReplacementHits > 0
                    ? PatchLifecycle.CorrectionPending
                    : PatchLifecycle.Mature;
                result.HeldMaturePatches++;
                if (!replacementSafe)
                    result.AtomicRollbackCount++;
            }
        }

        List<Vector3Int> retiredKeys = null;
        foreach (KeyValuePair<Vector3Int, Track> pair in _tracks)
        {
            if (_observed.Contains(pair.Key))
                continue;
            Track track = pair.Value;
            track.MissingRebuilds++;
            if (track.CreaseAtomicProbationHits > 0)
            {
                ResetCreaseAtomicProbation(track);
                result.CreaseAtomicProbationResetPatches++;
            }
            if (track.CorrectionTrialRemaining > 0)
            {
                RollbackCorrection(track, ref result);
                result.HeldMaturePatches++;
                continue;
            }
            if (track.Published != null)
            {
                track.Lifecycle = PatchLifecycle.Mature;
                result.HeldMaturePatches++;
                continue;
            }
            if (track.MissingRebuilds < settings.RetireMissingRebuilds)
                continue;
            if (retiredKeys == null)
                retiredKeys = new List<Vector3Int>();
            retiredKeys.Add(pair.Key);
        }
        if (retiredKeys != null)
        {
            for (int i = 0; i < retiredKeys.Count; i++)
            {
                _tracks.Remove(retiredKeys[i]);
                result.RetiredThisRebuild++;
            }
        }

        BuildPublishedMesh(out publishedVertices, out publishedTriangles, out publishedColors, ref result);
        CountLifecycle(ref result);
        return result;
    }

    private Result UpdateDirectAtomicBlocks(
        Settings settings,
        IReadOnlyDictionary<Vector3Int, PatchDirtyReason> externalSurfaceDirtyReasons,
        out List<Vector3> publishedVertices,
        out List<int> publishedTriangles,
        out List<Color> publishedColors,
        ref Result result)
    {
        foreach (KeyValuePair<Vector3Int, PatchGeometry> pair in _candidates)
        {
            Vector3Int key = pair.Key;
            PatchGeometry candidate = pair.Value;
            _observed.Add(key);
            if (!_tracks.TryGetValue(key, out Track track))
            {
                track = new Track
                {
                    Key = key,
                    FirstSeenRebuild = _rebuildSequence,
                    LastSeenRebuild = _rebuildSequence,
                    StableHits = 1,
                    LastCandidate = CloneGeometry(candidate),
                    Lifecycle = PatchLifecycle.Accumulating
                };
                _tracks.Add(key, track);
                result.NewPatches++;
                if (GeometrySafeForPublication(candidate))
                {
                    track.Published = CloneGeometry(candidate);
                    track.Lifecycle = PatchLifecycle.Mature;
                    result.PublishedThisRebuild++;
                }
                else
                {
                    track.Lifecycle = PatchLifecycle.Provisional;
                    result.AtomicRollbackCount++;
                }
                continue;
            }

            track.MissingRebuilds = 0;
            track.LastSeenRebuild = _rebuildSequence;
            track.LastCandidate = CloneGeometry(candidate);
            if (!_localExtractionKeys.Contains(key))
            {
                if (track.Published != null)
                {
                    track.Lifecycle = PatchLifecycle.Mature;
                    _cleanReusedKeys.Add(key);
                    result.CleanReusedPatches++;
                    result.HeldMaturePatches++;
                }
                continue;
            }

            if (!GeometrySafeForPublication(candidate))
            {
                track.Lifecycle = track.Published != null
                    ? PatchLifecycle.Mature
                    : PatchLifecycle.Provisional;
                if (track.Published != null)
                    result.HeldMaturePatches++;
                result.AtomicRollbackCount++;
                continue;
            }

            bool replacing = track.Published != null;
            track.Published = CloneGeometry(candidate);
            track.Lifecycle = PatchLifecycle.Mature;
            if (replacing)
                result.ReplacedThisRebuild++;
            else
                result.PublishedThisRebuild++;
        }

        List<Vector3Int> retired = null;
        foreach (KeyValuePair<Vector3Int, Track> pair in _tracks)
        {
            if (_observed.Contains(pair.Key))
                continue;
            Track track = pair.Value;
            bool tsdfMissing = externalSurfaceDirtyReasons != null &&
                               externalSurfaceDirtyReasons.TryGetValue(pair.Key, out PatchDirtyReason reason) &&
                               reason == PatchDirtyReason.Missing;
            if (!tsdfMissing)
            {
                if (track.Published != null)
                    result.HeldMaturePatches++;
                continue;
            }
            track.MissingRebuilds++;
            if (track.MissingRebuilds < settings.RetireMissingRebuilds)
            {
                if (track.Published != null)
                    result.HeldMaturePatches++;
                continue;
            }
            if (retired == null)
                retired = new List<Vector3Int>();
            retired.Add(pair.Key);
        }
        if (retired != null)
        {
            for (int i = 0; i < retired.Count; i++)
            {
                _tracks.Remove(retired[i]);
                result.RetiredThisRebuild++;
            }
        }

        BuildPublishedMesh(out publishedVertices, out publishedTriangles, out publishedColors, ref result);
        CountLifecycle(ref result);
        return result;
    }

    public void Clear()
    {
        _tracks.Clear();
        _candidates.Clear();
        _observed.Clear();
        _dirtyReasons.Clear();
        _localExtractionKeys.Clear();
        _cleanReusedKeys.Clear();
        _readOnlySeamAnchorKeys.Clear();
        _localClusterIds.Clear();
        _creaseRegionalEvidence.Clear();
        _creaseProbationAssignments.Clear();
        _rebuildSequence = 0;
    }

    public List<Snapshot> GetSnapshots()
    {
        List<Snapshot> snapshots = new List<Snapshot>(_tracks.Count);
        foreach (KeyValuePair<Vector3Int, Track> pair in _tracks)
        {
            Track track = pair.Value;
            snapshots.Add(new Snapshot
            {
                Key = pair.Key,
                Lifecycle = track.Lifecycle,
                SurfaceKind = track.Published != null
                    ? track.Published.SurfaceKind
                    : (track.LastCandidate != null ? track.LastCandidate.SurfaceKind : PatchSurfaceKind.Uncertain),
                StableHits = track.StableHits,
                ReplacementHits = track.ReplacementHits,
                MissingRebuilds = track.MissingRebuilds,
                CandidateTriangles = track.LastCandidate != null ? track.LastCandidate.TriangleCount : 0,
                PublishedTriangles = track.Published != null ? track.Published.TriangleCount : 0,
                CandidateBoundaryEdges = track.LastCandidate != null ? track.LastCandidate.BoundaryEdges : 0,
                PublishedBoundaryEdges = track.Published != null ? track.Published.BoundaryEdges : 0,
                CandidateNonManifoldEdges = track.LastCandidate != null ? track.LastCandidate.NonManifoldEdges : 0,
                PublishedNonManifoldEdges = track.Published != null ? track.Published.NonManifoldEdges : 0,
                CandidatePlaneResidualMeters = track.LastCandidate != null ? track.LastCandidate.PlaneResidualMeters : 0f,
                PublishedPlaneResidualMeters = track.Published != null ? track.Published.PlaneResidualMeters : 0f,
                CandidateNormalCoherence = track.LastCandidate != null ? track.LastCandidate.NormalCoherence : 0f,
                PublishedNormalCoherence = track.Published != null ? track.Published.NormalCoherence : 0f,
                DirtyReason = _dirtyReasons.TryGetValue(pair.Key, out PatchDirtyReason dirtyReason)
                    ? dirtyReason
                    : PatchDirtyReason.None,
                InLocalExtractionBounds = _localExtractionKeys.Contains(pair.Key),
                ReusedCleanPublished = _cleanReusedKeys.Contains(pair.Key),
                ReadOnlySeamAnchor = _readOnlySeamAnchorKeys.Contains(pair.Key),
                DirtyClusterId = _localClusterIds.TryGetValue(pair.Key, out int clusterId) ? clusterId : -1,
                CorrectionHits = track.ReplacementHits,
                CorrectionTrialRemaining = track.CorrectionTrialRemaining,
                HasRollbackBackup = track.RollbackPublished != null,
                PlaneConstraintId = track.LastCandidate != null ? track.LastCandidate.PlaneConstraintId : -1,
                PlaneConstraintApplied = track.LastCandidate != null && track.LastCandidate.PlaneConstraintApplied,
                PlaneConstraintRejectReason = track.LastCandidate != null ? track.LastCandidate.PlaneConstraintRejectReason : "none",
                PlaneConstraintNormalDot = track.LastCandidate != null ? track.LastCandidate.PlaneConstraintNormalDot : 0f,
                PlaneConstraintCentroidDistanceMeters = track.LastCandidate != null ? track.LastCandidate.PlaneConstraintCentroidDistanceMeters : 0f,
                PlaneConstraintNearestNormalPlaneId = track.LastCandidate != null ? track.LastCandidate.PlaneConstraintNearestNormalPlaneId : -1,
                PlaneConstraintBestNormalDot = track.LastCandidate != null ? track.LastCandidate.PlaneConstraintBestNormalDot : 0f,
                PlaneConstraintNearestNormalDistanceMeters = track.LastCandidate != null ? track.LastCandidate.PlaneConstraintNearestNormalDistanceMeters : 0f,
                PlaneConstraintDistanceExcessMeters = track.LastCandidate != null ? track.LastCandidate.PlaneConstraintDistanceExcessMeters : 0f,
                PlaneConstraintMaxMoveMeters = track.LastCandidate != null ? track.LastCandidate.PlaneConstraintMaxMoveMeters : 0f,
                PlaneResidualBeforeConstraintMeters = track.LastCandidate != null ? track.LastCandidate.PlaneResidualBeforeConstraintMeters : 0f,
                PlaneConstraintQuarantinedTriangles = track.LastCandidate != null ? track.LastCandidate.PlaneConstraintQuarantinedTriangles : 0,
                PlaneConstraintRetainedIslands = track.LastCandidate != null ? track.LastCandidate.PlaneConstraintRetainedIslands : 0,
                PlaneRejectHits = track.PlaneRejectHits,
                PlaneRejectLeaseReason = track.PlaneRejectLeaseReason,
                PlaneRejectForcedLocal = track.PlaneRejectHits > 0 && _localExtractionKeys.Contains(pair.Key),
                PlaneRejectWithheld = track.PlaneRejectWithheld,
                NormalDecompositionKind = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionKind : "none",
                NormalDecompositionDominantPlaneId = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionDominantPlaneId : -1,
                NormalDecompositionPlaneFamilies = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionPlaneFamilies : 0,
                NormalDecompositionIslands = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionIslands : 0,
                NormalDecompositionMatchedTriangles = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionMatchedTriangles : 0,
                NormalDecompositionUnresolvedTriangles = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionUnresolvedTriangles : 0,
                NormalDecompositionLargestIslandTriangles = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionLargestIslandTriangles : 0,
                NormalDecompositionWindingMinorityTriangles = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionWindingMinorityTriangles : 0,
                NormalDecompositionSubtype = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionSubtype : "none",
                NormalDecompositionPlaneIds = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionPlaneIds : "",
                NormalDecompositionMinFamilyNormalDot = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionMinFamilyNormalDot : 1f,
                NormalDecompositionMaxPlaneSeparationMeters = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionMaxPlaneSeparationMeters : 0f,
                NormalDecompositionUnmatchedNormalTriangles = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionUnmatchedNormalTriangles : 0,
                NormalDecompositionUnmatchedDistanceTriangles = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionUnmatchedDistanceTriangles : 0,
                NormalDecompositionUnmatchedExtentTriangles = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionUnmatchedExtentTriangles : 0,
                NormalDecompositionUnmatchedMoveTriangles = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionUnmatchedMoveTriangles : 0,
                NormalDecompositionUnmatchedAmbiguityTriangles = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionUnmatchedAmbiguityTriangles : 0,
                CreaseAtomicStatus = track.LastCandidate != null ? track.LastCandidate.CreaseAtomicStatus : "none",
                CreaseAtomicRetainedTriangles = track.LastCandidate != null ? track.LastCandidate.CreaseAtomicRetainedTriangles : 0,
                CreaseAtomicQuarantinedTriangles = track.LastCandidate != null ? track.LastCandidate.CreaseAtomicQuarantinedTriangles : 0,
                CreaseAtomicTopologyRejectedTriangles = track.LastCandidate != null ? track.LastCandidate.CreaseAtomicTopologyRejectedTriangles : 0,
                CreaseAtomicRetainedFamilies = track.LastCandidate != null ? track.LastCandidate.CreaseAtomicRetainedFamilies : 0,
                CreaseAtomicRequiredFamilies = track.LastCandidate != null ? track.LastCandidate.CreaseAtomicRequiredFamilies : 0,
                CreaseAtomicRetainedIslands = track.LastCandidate != null ? track.LastCandidate.CreaseAtomicRetainedIslands : 0,
                CreaseAtomicRetainedRatio = track.LastCandidate != null ? track.LastCandidate.CreaseAtomicRetainedRatio : 0f,
                CreaseAtomicMaxMoveMeters = track.LastCandidate != null ? track.LastCandidate.CreaseAtomicMaxMoveMeters : 0f,
                CreaseAtomicTinyIslandTriangles = track.LastCandidate != null ? track.LastCandidate.CreaseAtomicTinyIslandTriangles : 0,
                CreaseAtomicTinyIslands = track.LastCandidate != null ? track.LastCandidate.CreaseAtomicTinyIslands : 0,
                CreaseAtomicTinyIslandStableHits = track.CreaseAtomicTinyIslandStableHits,
                CreaseAtomicFamilyIds = track.LastCandidate != null ? track.LastCandidate.CreaseAtomicFamilyIds : "",
                CreaseAtomicProbationHits = track.CreaseAtomicProbationHits,
                CreaseAtomicProbationReady = track.CreaseAtomicProbationHits >= 2,
                CreaseTransitionKind = track.LastCandidate != null ? track.LastCandidate.CreaseTransitionKind : "none",
                CreaseTransitionFromStatus = track.LastCandidate != null ? track.LastCandidate.CreaseTransitionFromStatus : "none",
                CreaseTransitionFromFamilyIds = track.LastCandidate != null ? track.LastCandidate.CreaseTransitionFromFamilyIds : "",
                CreaseTransitionNearestInPriorFamily = track.LastCandidate != null && track.LastCandidate.CreaseTransitionNearestInPriorFamily,
                CreaseTransitionTriangleRatio = track.LastCandidate != null ? track.LastCandidate.CreaseTransitionTriangleRatio : 0f,
                CreaseTransitionCentroidShiftMeters = track.LastCandidate != null ? track.LastCandidate.CreaseTransitionCentroidShiftMeters : 0f,
                CreaseTransitionNormalDot = track.LastCandidate != null ? track.LastCandidate.CreaseTransitionNormalDot : 0f,
                CreaseTransitionRetainedRatioDelta = track.LastCandidate != null ? track.LastCandidate.CreaseTransitionRetainedRatioDelta : 0f,
                CreaseTransitionContentCompatible = track.LastCandidate != null && track.LastCandidate.CreaseTransitionContentCompatible,
                CreaseTransitionFamilyMinNormalDot = track.LastCandidate != null ? track.LastCandidate.CreaseTransitionFamilyMinNormalDot : 0f,
                CreaseTransitionFamilyMaxDistanceDeltaMeters = track.LastCandidate != null ? track.LastCandidate.CreaseTransitionFamilyMaxDistanceDeltaMeters : 0f,
                CreaseTransitionFamilyMinTriangleRatio = track.LastCandidate != null ? track.LastCandidate.CreaseTransitionFamilyMinTriangleRatio : 0f,
                CreaseTransitionFamilyRemappedCount = track.LastCandidate != null ? track.LastCandidate.CreaseTransitionFamilyRemappedCount : 0,
                CreaseRegionalMemberPatches = track.LastCreaseRegionalEvidence != null ? track.LastCreaseRegionalEvidence.CreaseRegionalMemberPatches : 0,
                CreaseRegionalRetainedRatio = track.LastCreaseRegionalEvidence != null ? track.LastCreaseRegionalEvidence.CreaseRegionalRetainedRatio : 0f,
                CreaseRegionalStatus = track.LastCreaseRegionalEvidence != null ? track.LastCreaseRegionalEvidence.CreaseRegionalStatus : "none",
                CreaseRegionalRetainedFamilies = track.LastCreaseRegionalEvidence != null ? track.LastCreaseRegionalEvidence.CreaseRegionalRetainedFamilies : 0,
                CreaseRegionalRequiredFamilies = track.LastCreaseRegionalEvidence != null ? track.LastCreaseRegionalEvidence.CreaseRegionalRequiredFamilies : 0,
                CreaseRegionalMigrated = track.LastCreaseRegionalEvidence != null && track.LastCreaseRegionalEvidence.CreaseRegionalMigrated,
                CreaseRegionalSourceKey = track.LastCreaseRegionalEvidence != null ? track.LastCreaseRegionalEvidence.CreaseRegionalSourceKey : track.Key,
                NormalDecompositionNearNormalTriangles = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionNearNormalTriangles : 0,
                NormalDecompositionNovelNormalTriangles = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionNovelNormalTriangles : 0,
                NormalDecompositionNearDistanceTriangles = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionNearDistanceTriangles : 0,
                NormalDecompositionFarDistanceTriangles = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionFarDistanceTriangles : 0,
                NormalDecompositionNearMoveTriangles = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionNearMoveTriangles : 0,
                NormalDecompositionFarMoveTriangles = track.LastCandidate != null ? track.LastCandidate.NormalDecompositionFarMoveTriangles : 0
            });
        }
        snapshots.Sort((a, b) =>
        {
            int compare = a.Key.x.CompareTo(b.Key.x);
            if (compare != 0) return compare;
            compare = a.Key.y.CompareTo(b.Key.y);
            return compare != 0 ? compare : a.Key.z.CompareTo(b.Key.z);
        });
        return snapshots;
    }

    public StringBuilder BuildCsv()
    {
        StringBuilder csv = new StringBuilder(4096);
        csv.AppendLine("rebuild,patch_x,patch_y,patch_z,lifecycle,surface_kind,dirty_reason,dirty_cluster,in_local_extraction,read_only_seam_anchor,reused_clean,stable_hits,replacement_hits,correction_trial_remaining,has_rollback_backup,missing_rebuilds,candidate_triangles,published_triangles,candidate_boundary,published_boundary,candidate_nonmanifold,published_nonmanifold,candidate_plane_residual_m,published_plane_residual_m,candidate_normal_coherence,published_normal_coherence,plane_constraint_id,plane_constraint_applied,plane_constraint_reject,plane_constraint_normal_dot,plane_constraint_centroid_distance_m,plane_constraint_max_move_m,plane_residual_before_constraint_m,plane_constraint_quarantined_triangles,plane_constraint_retained_islands,plane_reject_hits,plane_reject_lease_reason,plane_reject_forced_local,plane_reject_withheld,normal_decomp_kind,normal_decomp_subtype,normal_decomp_plane_ids,normal_decomp_dominant_plane_id,normal_decomp_plane_families,normal_decomp_islands,normal_decomp_matched_triangles,normal_decomp_unresolved_triangles,normal_decomp_largest_island_triangles,normal_decomp_winding_minority_triangles,normal_decomp_min_family_normal_dot,normal_decomp_max_plane_separation_m,normal_decomp_unmatched_normal,normal_decomp_unmatched_distance,normal_decomp_unmatched_extent,normal_decomp_unmatched_move,normal_decomp_unmatched_ambiguity,normal_decomp_near_normal,normal_decomp_novel_normal,normal_decomp_near_distance,normal_decomp_far_distance,normal_decomp_near_move,normal_decomp_far_move,crease_atomic_status,crease_atomic_retained_triangles,crease_atomic_quarantined_triangles,crease_atomic_topology_rejected_triangles,crease_atomic_retained_families,crease_atomic_required_families,crease_atomic_retained_islands,crease_atomic_retained_ratio,crease_atomic_max_move_m,crease_atomic_tiny_island_triangles,crease_atomic_tiny_islands,crease_atomic_tiny_island_stable_hits,crease_atomic_family_ids,crease_atomic_probation_hits,crease_atomic_probation_ready");
        int headerLineEnd = csv.Length - Environment.NewLine.Length;
        csv.Insert(headerLineEnd,
            ",plane_constraint_nearest_normal_plane_id,plane_constraint_best_normal_dot,plane_constraint_nearest_normal_distance_m,plane_constraint_distance_excess_m" +
            ",crease_transition_kind,crease_transition_from_status,crease_transition_from_family_ids,crease_transition_nearest_in_prior_family" +
            ",crease_transition_triangle_ratio,crease_transition_centroid_shift_m,crease_transition_normal_dot,crease_transition_retained_ratio_delta,crease_transition_content_compatible" +
            ",crease_transition_family_min_normal_dot,crease_transition_family_max_distance_delta_m,crease_transition_family_min_triangle_ratio,crease_transition_family_remapped_count" +
            ",crease_region_member_patches,crease_region_retained_ratio,crease_region_status,crease_region_retained_families,crease_region_required_families,crease_region_migrated,crease_region_source_x,crease_region_source_y,crease_region_source_z");
        List<Snapshot> snapshots = GetSnapshots();
        for (int i = 0; i < snapshots.Count; i++)
        {
            Snapshot row = snapshots[i];
            csv.Append(_rebuildSequence).Append(',')
                .Append(row.Key.x).Append(',').Append(row.Key.y).Append(',').Append(row.Key.z).Append(',')
                .Append(row.Lifecycle).Append(',').Append(row.SurfaceKind).Append(',')
                .Append(row.DirtyReason).Append(',')
                .Append(row.DirtyClusterId).Append(',')
                .Append(row.InLocalExtractionBounds ? 1 : 0).Append(',')
                .Append(row.ReadOnlySeamAnchor ? 1 : 0).Append(',')
                .Append(row.ReusedCleanPublished ? 1 : 0).Append(',')
                .Append(row.StableHits).Append(',').Append(row.ReplacementHits).Append(',')
                .Append(row.CorrectionTrialRemaining).Append(',')
                .Append(row.HasRollbackBackup ? 1 : 0).Append(',')
                .Append(row.MissingRebuilds).Append(',')
                .Append(row.CandidateTriangles).Append(',').Append(row.PublishedTriangles).Append(',')
                .Append(row.CandidateBoundaryEdges).Append(',').Append(row.PublishedBoundaryEdges).Append(',')
                .Append(row.CandidateNonManifoldEdges).Append(',').Append(row.PublishedNonManifoldEdges).Append(',')
                .Append(row.CandidatePlaneResidualMeters.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.PublishedPlaneResidualMeters.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.CandidateNormalCoherence.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.PublishedNormalCoherence.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.PlaneConstraintId).Append(',')
                .Append(row.PlaneConstraintApplied ? 1 : 0).Append(',')
                .Append(row.PlaneConstraintRejectReason ?? "none").Append(',')
                .Append(row.PlaneConstraintNormalDot.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.PlaneConstraintCentroidDistanceMeters.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.PlaneConstraintMaxMoveMeters.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.PlaneResidualBeforeConstraintMeters.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.PlaneConstraintQuarantinedTriangles).Append(',')
                .Append(row.PlaneConstraintRetainedIslands).Append(',')
                .Append(row.PlaneRejectHits).Append(',')
                .Append(row.PlaneRejectLeaseReason ?? "none").Append(',')
                .Append(row.PlaneRejectForcedLocal ? 1 : 0).Append(',')
                .Append(row.PlaneRejectWithheld ? 1 : 0).Append(',')
                .Append(row.NormalDecompositionKind ?? "none").Append(',')
                .Append(row.NormalDecompositionSubtype ?? "none").Append(',')
                .Append(row.NormalDecompositionPlaneIds ?? "").Append(',')
                .Append(row.NormalDecompositionDominantPlaneId).Append(',')
                .Append(row.NormalDecompositionPlaneFamilies).Append(',')
                .Append(row.NormalDecompositionIslands).Append(',')
                .Append(row.NormalDecompositionMatchedTriangles).Append(',')
                .Append(row.NormalDecompositionUnresolvedTriangles).Append(',')
                .Append(row.NormalDecompositionLargestIslandTriangles).Append(',')
                .Append(row.NormalDecompositionWindingMinorityTriangles).Append(',')
                .Append(row.NormalDecompositionMinFamilyNormalDot.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.NormalDecompositionMaxPlaneSeparationMeters.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.NormalDecompositionUnmatchedNormalTriangles).Append(',')
                .Append(row.NormalDecompositionUnmatchedDistanceTriangles).Append(',')
                .Append(row.NormalDecompositionUnmatchedExtentTriangles).Append(',')
                .Append(row.NormalDecompositionUnmatchedMoveTriangles).Append(',')
                .Append(row.NormalDecompositionUnmatchedAmbiguityTriangles).Append(',')
                .Append(row.NormalDecompositionNearNormalTriangles).Append(',')
                .Append(row.NormalDecompositionNovelNormalTriangles).Append(',')
                .Append(row.NormalDecompositionNearDistanceTriangles).Append(',')
                .Append(row.NormalDecompositionFarDistanceTriangles).Append(',')
                .Append(row.NormalDecompositionNearMoveTriangles).Append(',')
                .Append(row.NormalDecompositionFarMoveTriangles).Append(',')
                .Append(row.CreaseAtomicStatus ?? "none").Append(',')
                .Append(row.CreaseAtomicRetainedTriangles).Append(',')
                .Append(row.CreaseAtomicQuarantinedTriangles).Append(',')
                .Append(row.CreaseAtomicTopologyRejectedTriangles).Append(',')
                .Append(row.CreaseAtomicRetainedFamilies).Append(',')
                .Append(row.CreaseAtomicRequiredFamilies).Append(',')
                .Append(row.CreaseAtomicRetainedIslands).Append(',')
                .Append(row.CreaseAtomicRetainedRatio.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.CreaseAtomicMaxMoveMeters.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.CreaseAtomicTinyIslandTriangles).Append(',')
                .Append(row.CreaseAtomicTinyIslands).Append(',')
                .Append(row.CreaseAtomicTinyIslandStableHits).Append(',')
                .Append(row.CreaseAtomicFamilyIds ?? "").Append(',')
                .Append(row.CreaseAtomicProbationHits).Append(',')
                .Append(row.CreaseAtomicProbationReady ? 1 : 0).Append(',')
                .Append(row.PlaneConstraintNearestNormalPlaneId).Append(',')
                .Append(row.PlaneConstraintBestNormalDot.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.PlaneConstraintNearestNormalDistanceMeters.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.PlaneConstraintDistanceExcessMeters.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.CreaseTransitionKind ?? "none").Append(',')
                .Append(row.CreaseTransitionFromStatus ?? "none").Append(',')
                .Append(row.CreaseTransitionFromFamilyIds ?? "").Append(',')
                .Append(row.CreaseTransitionNearestInPriorFamily ? 1 : 0).Append(',')
                .Append(row.CreaseTransitionTriangleRatio.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.CreaseTransitionCentroidShiftMeters.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.CreaseTransitionNormalDot.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.CreaseTransitionRetainedRatioDelta.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.CreaseTransitionContentCompatible ? 1 : 0).Append(',')
                .Append(row.CreaseTransitionFamilyMinNormalDot.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.CreaseTransitionFamilyMaxDistanceDeltaMeters.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.CreaseTransitionFamilyMinTriangleRatio.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.CreaseTransitionFamilyRemappedCount).Append(',')
                .Append(row.CreaseRegionalMemberPatches).Append(',')
                .Append(row.CreaseRegionalRetainedRatio.ToString("F6", CultureInfo.InvariantCulture)).Append(',')
                .Append(row.CreaseRegionalStatus ?? "none").Append(',')
                .Append(row.CreaseRegionalRetainedFamilies).Append(',')
                .Append(row.CreaseRegionalRequiredFamilies).Append(',')
                .Append(row.CreaseRegionalMigrated ? 1 : 0).Append(',')
                .Append(row.CreaseRegionalSourceKey.x).Append(',')
                .Append(row.CreaseRegionalSourceKey.y).Append(',')
                .Append(row.CreaseRegionalSourceKey.z).AppendLine();
        }
        return csv;
    }

    private void BuildDirtyAndLocalExtractionSets(
        Settings settings,
        IReadOnlyDictionary<Vector3Int, PatchDirtyReason> externalSurfaceDirtyReasons,
        ref Result result)
    {
        bool useExternalSurfaceDirty = externalSurfaceDirtyReasons != null;
        bool externalDirtyIsAuthoritative = useExternalSurfaceDirty && settings.ExternalDirtyIsAuthoritative;
        foreach (KeyValuePair<Vector3Int, PatchGeometry> pair in _candidates)
        {
            Vector3Int key = pair.Key;
            PatchDirtyReason reason;
            if (!_tracks.TryGetValue(key, out Track track))
                reason = PatchDirtyReason.New;
            else if (externalDirtyIsAuthoritative)
                reason = externalSurfaceDirtyReasons.TryGetValue(key, out PatchDirtyReason authoritativeReason)
                    ? authoritativeReason
                    : PatchDirtyReason.None;
            else if (track.Published == null)
                reason = PatchDirtyReason.Provisional;
            else if (track.PlaneRejectHits > 0)
                reason = PatchDirtyReason.GeometryChanged;
            else if (track.Lifecycle == PatchLifecycle.CorrectionPending ||
                     track.ReplacementHits > 0 ||
                     track.CorrectionTrialRemaining > 0)
                reason = PatchDirtyReason.GeometryChanged;
            else if (useExternalSurfaceDirty &&
                     externalSurfaceDirtyReasons.TryGetValue(key, out PatchDirtyReason externalReason))
                reason = externalReason;
            else if (!useExternalSurfaceDirty && !GeometryCompatible(track.Published, pair.Value, settings))
                reason = PatchDirtyReason.GeometryChanged;
            else
                reason = PatchDirtyReason.None;

            if (reason == PatchDirtyReason.None)
                continue;
            _dirtyReasons[key] = reason;
            result.DirtyPatches++;
            result.DirtyTriangles += pair.Value.TriangleCount;
            CountDirtyReason(reason, ref result);
        }

        if (useExternalSurfaceDirty)
        {
            foreach (KeyValuePair<Vector3Int, PatchDirtyReason> pair in externalSurfaceDirtyReasons)
            {
                if (pair.Value == PatchDirtyReason.None || _dirtyReasons.ContainsKey(pair.Key))
                    continue;
                _dirtyReasons[pair.Key] = pair.Value;
                result.DirtyPatches++;
                CountDirtyReason(pair.Value, ref result);
                if (pair.Value == PatchDirtyReason.Missing)
                    result.DirtyMissingPatches++;
            }
        }

        foreach (KeyValuePair<Vector3Int, Track> pair in _tracks)
        {
            if (_candidates.ContainsKey(pair.Key))
                continue;
            if (useExternalSurfaceDirty &&
                (!externalSurfaceDirtyReasons.TryGetValue(pair.Key, out PatchDirtyReason missingReason) ||
                 missingReason != PatchDirtyReason.Missing))
                continue;
            if (_dirtyReasons.ContainsKey(pair.Key))
                continue;
            _dirtyReasons[pair.Key] = PatchDirtyReason.Missing;
            result.DirtyPatches++;
            result.DirtyMissingPatches++;
        }

        int padding = Mathf.Clamp(settings.LocalExtractionPaddingPatches, 0, 3);
        foreach (KeyValuePair<Vector3Int, PatchDirtyReason> pair in _dirtyReasons)
        {
            Vector3Int key = pair.Key;
            if (!result.HasDirtyBounds)
            {
                result.HasDirtyBounds = true;
                result.DirtyBoundsMin = key;
                result.DirtyBoundsMax = key;
            }
            else
            {
                result.DirtyBoundsMin = Vector3Int.Min(result.DirtyBoundsMin, key);
                result.DirtyBoundsMax = Vector3Int.Max(result.DirtyBoundsMax, key);
            }
        }

        // Build independent six-connected dirty components. Each component is an
        // extraction work unit; diagonal/corner contact must not merge two regions.
        HashSet<Vector3Int> unvisited = new HashSet<Vector3Int>(_dirtyReasons.Keys);
        Queue<Vector3Int> queue = new Queue<Vector3Int>();
        int clusterId = 0;
        while (unvisited.Count > 0)
        {
            Vector3Int seed = default(Vector3Int);
            foreach (Vector3Int key in unvisited)
            {
                seed = key;
                break;
            }
            unvisited.Remove(seed);
            queue.Enqueue(seed);
            List<Vector3Int> dirtyCluster = new List<Vector3Int>();
            while (queue.Count > 0)
            {
                Vector3Int key = queue.Dequeue();
                dirtyCluster.Add(key);
                _localClusterIds[key] = clusterId;
                for (int i = 0; i < SixConnectedOffsets.Length; i++)
                {
                    Vector3Int neighbor = key + SixConnectedOffsets[i];
                    if (!unvisited.Remove(neighbor))
                        continue;
                    queue.Enqueue(neighbor);
                }
            }

            HashSet<Vector3Int> localCluster = new HashSet<Vector3Int>();
            for (int i = 0; i < dirtyCluster.Count; i++)
            {
                Vector3Int key = dirtyCluster[i];
                for (int z = -padding; z <= padding; z++)
                {
                    for (int y = -padding; y <= padding; y++)
                    {
                        for (int x = -padding; x <= padding; x++)
                        {
                            if (Mathf.Abs(x) + Mathf.Abs(y) + Mathf.Abs(z) > padding)
                                continue;
                            Vector3Int localKey = key + new Vector3Int(x, y, z);
                            localCluster.Add(localKey);
                            if (_dirtyReasons.ContainsKey(localKey))
                                _localExtractionKeys.Add(localKey);
                            else
                                _readOnlySeamAnchorKeys.Add(localKey);
                            if (!_localClusterIds.ContainsKey(localKey))
                                _localClusterIds[localKey] = clusterId;
                        }
                    }
                }
            }

            int localCandidateCount = 0;
            foreach (Vector3Int localKey in localCluster)
            {
                if (_candidates.ContainsKey(localKey))
                    localCandidateCount++;
            }
            result.DirtyClusterCount++;
            result.LargestDirtyClusterPatches = Mathf.Max(
                result.LargestDirtyClusterPatches,
                dirtyCluster.Count);
            result.LargestLocalClusterPatches = Mathf.Max(
                result.LargestLocalClusterPatches,
                localCandidateCount);
            clusterId++;
        }

        foreach (KeyValuePair<Vector3Int, PatchGeometry> pair in _candidates)
        {
            if (_localExtractionKeys.Contains(pair.Key))
            {
                result.LocalExtractionPatches++;
                result.LocalExtractionTriangles += pair.Value.TriangleCount;
                if (_tracks.TryGetValue(pair.Key, out Track track) && track.PlaneRejectHits > 0)
                    result.PlaneRejectForcedLocalPatches++;
            }
            else if (_readOnlySeamAnchorKeys.Contains(pair.Key))
            {
                result.ReadOnlySeamAnchorPatches++;
                result.ReadOnlySeamAnchorTriangles += pair.Value.TriangleCount;
            }
        }
    }

    private static void CountDirtyReason(PatchDirtyReason reason, ref Result result)
    {
        switch (reason)
        {
            case PatchDirtyReason.New:
                result.DirtyNewPatches++;
                break;
            case PatchDirtyReason.Provisional:
                result.DirtyProvisionalPatches++;
                break;
            case PatchDirtyReason.GeometryChanged:
                result.DirtyChangedPatches++;
                break;
        }
    }

    private void BuildCandidates(
        List<Vector3> vertices,
        List<int> triangles,
        List<Color> colors,
        Vector3 origin,
        Settings settings,
        ref Result result)
    {
        if (vertices == null || triangles == null)
            return;
        for (int i = 0; i + 2 < triangles.Count; i += 3)
        {
            int ia = triangles[i];
            int ib = triangles[i + 1];
            int ic = triangles[i + 2];
            if (!ValidTriangle(ia, ib, ic, vertices.Count))
                continue;
            Vector3 centroid = (vertices[ia] + vertices[ib] + vertices[ic]) / 3f;
            Vector3Int key = PatchKey(centroid, origin, settings.PatchSizeMeters);
            if (!_candidates.TryGetValue(key, out PatchGeometry patch))
            {
                if (_candidates.Count >= settings.MaxPatches)
                    continue;
                patch = new PatchGeometry();
                _candidates.Add(key, patch);
            }
            if (patch.TriangleCount >= settings.MaxTrianglesPerPatch)
                continue;
            patch.Triangles.Add(RemapVertex(ia, vertices, colors, patch));
            patch.Triangles.Add(RemapVertex(ib, vertices, colors, patch));
            patch.Triangles.Add(RemapVertex(ic, vertices, colors, patch));
            result.CandidateTriangles++;
        }

        foreach (KeyValuePair<Vector3Int, PatchGeometry> pair in _candidates)
        {
            AnalyzeGeometry(pair.Value, settings);
            TryApplyMaturePlaneConstraint(pair.Value, settings, ref result);
            result.CandidateBoundaryEdges += pair.Value.BoundaryEdges;
            result.CandidateNonManifoldEdges += pair.Value.NonManifoldEdges;
        }
        result.CandidatePatches = _candidates.Count;
    }

    private static void TryApplyMaturePlaneConstraint(
        PatchGeometry geometry,
        Settings settings,
        ref Result result)
    {
        IReadOnlyList<ScanCoverStablePlaneRegistry.PlaneConstraint> constraints = settings.MaturePlaneConstraints;
        if (geometry == null || constraints == null || constraints.Count == 0 ||
            geometry.PlaneResidualMeters < settings.PlanePatchMinResidualMeters)
            return;

        result.PlaneConstraintCandidatePatches++;
        geometry.PlaneResidualBeforeConstraintMeters = geometry.PlaneResidualMeters;
        int bestIndex = -1;
        int secondIndex = -1;
        float bestScore = float.PositiveInfinity;
        float secondScore = float.PositiveInfinity;
        bool passedNormal = false;
        bool passedDistance = false;
        float bestAnyNormalDot = 0f;
        float nearestNormalDistance = float.PositiveInfinity;
        int nearestNormalPlaneId = -1;

        for (int i = 0; i < constraints.Count; i++)
        {
            ScanCoverStablePlaneRegistry.PlaneConstraint plane = constraints[i];
            if (plane.Normal.sqrMagnitude < 0.5f)
                continue;
            Vector3 normal = plane.Normal.normalized;
            float normalDot = Mathf.Abs(Vector3.Dot(geometry.AverageNormal, normal));
            bestAnyNormalDot = Mathf.Max(bestAnyNormalDot, normalDot);
            if (normalDot < settings.PlanePatchMinNormalDot)
                continue;
            passedNormal = true;
            float signedDistance = Vector3.Dot(normal, geometry.Centroid) + plane.Offset;
            float distance = Mathf.Abs(signedDistance);
            if (distance < nearestNormalDistance)
            {
                nearestNormalDistance = distance;
                nearestNormalPlaneId = plane.Id;
            }
            if (distance > settings.PlanePatchMaxCentroidDistanceMeters)
                continue;
            passedDistance = true;
            Vector3 fromCenter = geometry.Centroid - plane.Center;
            Vector3 tangent = fromCenter - normal * Vector3.Dot(normal, fromCenter);
            if (tangent.magnitude > plane.Radius + settings.PlanePatchTangentialPaddingMeters)
                continue;
            float score = distance / Mathf.Max(0.001f, settings.PlanePatchMaxCentroidDistanceMeters) +
                          (1f - normalDot);
            if (score < bestScore)
            {
                secondScore = bestScore;
                secondIndex = bestIndex;
                bestScore = score;
                bestIndex = i;
            }
            else if (score < secondScore)
            {
                secondScore = score;
                secondIndex = i;
            }
        }

        geometry.PlaneConstraintBestNormalDot = bestAnyNormalDot;
        geometry.PlaneConstraintNearestNormalPlaneId = nearestNormalPlaneId;
        geometry.PlaneConstraintNearestNormalDistanceMeters = float.IsInfinity(nearestNormalDistance)
            ? 0f
            : nearestNormalDistance;
        geometry.PlaneConstraintDistanceExcessMeters = float.IsInfinity(nearestNormalDistance)
            ? 0f
            : Mathf.Max(0f, nearestNormalDistance - settings.PlanePatchMaxCentroidDistanceMeters);

        if (bestIndex < 0)
        {
            if (!passedNormal)
            {
                geometry.PlaneConstraintRejectReason = "normal";
                AnalyzeNormalRejectDecomposition(geometry, constraints, settings, ref result);
                result.PlaneConstraintRejectedNormal++;
            }
            else if (!passedDistance)
            {
                geometry.PlaneConstraintRejectReason = "distance";
                result.PlaneConstraintRejectedDistance++;
            }
            else
            {
                geometry.PlaneConstraintRejectReason = "extent";
                result.PlaneConstraintRejectedExtent++;
            }
            return;
        }

        ScanCoverStablePlaneRegistry.PlaneConstraint best = constraints[bestIndex];
        if (secondIndex >= 0 && secondScore - bestScore < settings.PlanePatchAmbiguityScoreMargin &&
            Mathf.Abs(Vector3.Dot(best.Normal.normalized, constraints[secondIndex].Normal.normalized)) < 0.95f)
        {
            geometry.PlaneConstraintRejectReason = "ambiguous";
            result.PlaneConstraintRejectedAmbiguity++;
            return;
        }

        Vector3 planeNormal = best.Normal.normalized;
        List<Vector3> projected = new List<Vector3>(geometry.Vertices.Count);
        float maxMove = 0f;
        bool moveOverflow = false;
        for (int i = 0; i < geometry.Vertices.Count; i++)
        {
            Vector3 point = geometry.Vertices[i];
            float signed = Vector3.Dot(planeNormal, point) + best.Offset;
            Vector3 snapped = point - planeNormal * signed;
            float move = Mathf.Abs(signed);
            maxMove = Mathf.Max(maxMove, move);
            if (!Finite(snapped))
            {
                geometry.PlaneConstraintId = best.Id;
                geometry.PlaneConstraintMaxMoveMeters = maxMove;
                geometry.PlaneConstraintRejectReason = "move";
                result.PlaneConstraintRejectedMove++;
                return;
            }
            if (move > settings.PlanePatchMaxVertexMoveMeters)
                moveOverflow = true;
            projected.Add(snapped);
        }

        if (moveOverflow)
        {
            if (TryRescueMoveOverflowComponent(geometry, best, planeNormal, settings, maxMove, ref result))
                return;
            geometry.PlaneConstraintId = best.Id;
            geometry.PlaneConstraintMaxMoveMeters = maxMove;
            geometry.PlaneConstraintRejectReason = "move";
            result.PlaneConstraintRejectedMove++;
            return;
        }

        for (int i = 0; i + 2 < geometry.Triangles.Count; i += 3)
        {
            int a = geometry.Triangles[i];
            int b = geometry.Triangles[i + 1];
            int c = geometry.Triangles[i + 2];
            Vector3 oldCross = Vector3.Cross(
                geometry.Vertices[b] - geometry.Vertices[a],
                geometry.Vertices[c] - geometry.Vertices[a]);
            Vector3 newCross = Vector3.Cross(projected[b] - projected[a], projected[c] - projected[a]);
            float oldArea = oldCross.magnitude;
            float newArea = newCross.magnitude;
            float normalDot = oldArea > 0.000001f && newArea > 0.000001f
                ? Vector3.Dot(oldCross / oldArea, newCross / newArea)
                : -1f;
            if (newArea < oldArea * settings.PlanePatchMinTriangleAreaRatio ||
                normalDot < settings.PlanePatchMinTriangleNormalDot)
            {
                geometry.PlaneConstraintId = best.Id;
                geometry.PlaneConstraintMaxMoveMeters = maxMove;
                geometry.PlaneConstraintRejectReason = "topology";
                result.PlaneConstraintRejectedTopology++;
                return;
            }
        }

        geometry.Vertices.Clear();
        geometry.Vertices.AddRange(projected);
        geometry.PlaneConstraintId = best.Id;
        geometry.PlaneConstraintApplied = true;
        geometry.PlaneConstraintRejectReason = "none";
        geometry.PlaneConstraintNormalDot = Mathf.Abs(Vector3.Dot(geometry.AverageNormal, planeNormal));
        geometry.PlaneConstraintCentroidDistanceMeters =
            Mathf.Abs(Vector3.Dot(planeNormal, geometry.Centroid) + best.Offset);
        geometry.PlaneConstraintMaxMoveMeters = maxMove;
        AnalyzeGeometry(geometry, settings);
        result.PlaneConstraintAppliedPatches++;
        result.PlaneConstraintAppliedTriangles += geometry.TriangleCount;
    }

    private static void AnalyzeNormalRejectDecomposition(
        PatchGeometry geometry,
        IReadOnlyList<ScanCoverStablePlaneRegistry.PlaneConstraint> constraints,
        Settings settings,
        ref Result result)
    {
        if (geometry == null || constraints == null || constraints.Count == 0 || geometry.TriangleCount <= 0)
            return;

        int triangleCount = geometry.TriangleCount;
        int[] planeByTriangle = new int[triangleCount];
        int[] parent = new int[triangleCount];
        int[] windingByTriangle = new int[triangleCount];
        for (int i = 0; i < triangleCount; i++)
        {
            planeByTriangle[i] = -1;
            parent[i] = i;
        }

        Dictionary<ulong, int> firstTriangleByEdge = new Dictionary<ulong, int>(triangleCount * 2);
        Dictionary<int, int> familyCounts = new Dictionary<int, int>();
        Dictionary<int, Vector2Int> familyWindingCounts = new Dictionary<int, Vector2Int>();
        Dictionary<int, ScanCoverStablePlaneRegistry.PlaneConstraint> constraintById =
            new Dictionary<int, ScanCoverStablePlaneRegistry.PlaneConstraint>();
        for (int i = 0; i < constraints.Count; i++)
            constraintById[constraints[i].Id] = constraints[i];
        int matched = 0;
        int unmatchedNormal = 0;
        int unmatchedDistance = 0;
        int unmatchedExtent = 0;
        int unmatchedMove = 0;
        int unmatchedAmbiguity = 0;
        int nearNormal = 0;
        int novelNormal = 0;
        int nearDistance = 0;
        int farDistance = 0;
        int nearMove = 0;
        int farMove = 0;
        for (int triangle = 0; triangle < triangleCount; triangle++)
        {
            int offset = triangle * 3;
            int a = geometry.Triangles[offset];
            int b = geometry.Triangles[offset + 1];
            int c = geometry.Triangles[offset + 2];
            Vector3 va = geometry.Vertices[a];
            Vector3 vb = geometry.Vertices[b];
            Vector3 vc = geometry.Vertices[c];
            Vector3 cross = Vector3.Cross(vb - va, vc - va);
            float area = cross.magnitude;
            if (area <= 0.000001f)
                continue;
            Vector3 triangleNormal = cross / area;
            Vector3 centroid = (va + vb + vc) / 3f;
            int bestIndex = -1;
            int secondIndex = -1;
            float bestScore = float.PositiveInfinity;
            float secondScore = float.PositiveInfinity;
            bool passedTriangleNormal = false;
            bool passedTriangleDistance = false;
            bool passedTriangleExtent = false;
            bool passedTriangleMove = false;
            float bestAnyNormalDot = 0f;
            float nearestNormalDistance = float.PositiveInfinity;
            float smallestRequiredMove = float.PositiveInfinity;
            for (int planeIndex = 0; planeIndex < constraints.Count; planeIndex++)
            {
                ScanCoverStablePlaneRegistry.PlaneConstraint plane = constraints[planeIndex];
                if (plane.Normal.sqrMagnitude < 0.5f)
                    continue;
                Vector3 normal = plane.Normal.normalized;
                float normalDot = Mathf.Abs(Vector3.Dot(triangleNormal, normal));
                bestAnyNormalDot = Mathf.Max(bestAnyNormalDot, normalDot);
                if (normalDot < settings.PlanePatchMinNormalDot)
                    continue;
                passedTriangleNormal = true;
                float centroidDistance = Mathf.Abs(Vector3.Dot(normal, centroid) + plane.Offset);
                nearestNormalDistance = Mathf.Min(nearestNormalDistance, centroidDistance);
                if (centroidDistance > settings.PlanePatchMaxCentroidDistanceMeters)
                    continue;
                passedTriangleDistance = true;
                Vector3 fromCenter = centroid - plane.Center;
                Vector3 tangent = fromCenter - normal * Vector3.Dot(normal, fromCenter);
                if (tangent.magnitude > plane.Radius + settings.PlanePatchTangentialPaddingMeters)
                    continue;
                passedTriangleExtent = true;
                float da = Vector3.Dot(normal, va) + plane.Offset;
                float db = Vector3.Dot(normal, vb) + plane.Offset;
                float dc = Vector3.Dot(normal, vc) + plane.Offset;
                float requiredMove = Mathf.Max(Mathf.Abs(da), Mathf.Max(Mathf.Abs(db), Mathf.Abs(dc)));
                smallestRequiredMove = Mathf.Min(smallestRequiredMove, requiredMove);
                if (requiredMove > settings.PlanePatchMaxVertexMoveMeters)
                    continue;
                passedTriangleMove = true;
                float score = centroidDistance / Mathf.Max(0.001f, settings.PlanePatchMaxCentroidDistanceMeters) +
                              (1f - normalDot);
                if (score < bestScore)
                {
                    secondScore = bestScore;
                    secondIndex = bestIndex;
                    bestScore = score;
                    bestIndex = planeIndex;
                }
                else if (score < secondScore)
                {
                    secondScore = score;
                    secondIndex = planeIndex;
                }
            }

            if (bestIndex < 0)
            {
                if (!passedTriangleNormal)
                {
                    unmatchedNormal++;
                    if (bestAnyNormalDot >= Mathf.Max(0.5f, settings.PlanePatchMinNormalDot - 0.10f)) nearNormal++;
                    else novelNormal++;
                }
                else if (!passedTriangleDistance)
                {
                    unmatchedDistance++;
                    if (nearestNormalDistance <= settings.PlanePatchMaxCentroidDistanceMeters * 1.5f) nearDistance++;
                    else farDistance++;
                }
                else if (!passedTriangleExtent) unmatchedExtent++;
                else if (!passedTriangleMove)
                {
                    unmatchedMove++;
                    if (smallestRequiredMove <= settings.PlanePatchMaxVertexMoveMeters * 1.5f) nearMove++;
                    else farMove++;
                }
                else unmatchedAmbiguity++;
                continue;
            }
            if (secondIndex >= 0 && secondScore - bestScore < settings.PlanePatchAmbiguityScoreMargin &&
                Mathf.Abs(Vector3.Dot(constraints[bestIndex].Normal.normalized,
                                      constraints[secondIndex].Normal.normalized)) < 0.95f)
            {
                unmatchedAmbiguity++;
                continue;
            }

            ScanCoverStablePlaneRegistry.PlaneConstraint best = constraints[bestIndex];
            int planeId = best.Id;
            planeByTriangle[triangle] = planeId;
            windingByTriangle[triangle] = Vector3.Dot(triangleNormal, best.Normal.normalized) >= 0f ? 1 : -1;
            matched++;
            familyCounts.TryGetValue(planeId, out int familyCount);
            familyCounts[planeId] = familyCount + 1;
            familyWindingCounts.TryGetValue(planeId, out Vector2Int windingCounts);
            if (windingByTriangle[triangle] > 0)
                windingCounts.x++;
            else
                windingCounts.y++;
            familyWindingCounts[planeId] = windingCounts;
            UnionNormalTriangleEdge(firstTriangleByEdge, parent, planeByTriangle, triangle, a, b);
            UnionNormalTriangleEdge(firstTriangleByEdge, parent, planeByTriangle, triangle, b, c);
            UnionNormalTriangleEdge(firstTriangleByEdge, parent, planeByTriangle, triangle, c, a);
        }

        Dictionary<int, int> componentSizes = new Dictionary<int, int>();
        int largestIsland = 0;
        for (int triangle = 0; triangle < triangleCount; triangle++)
        {
            if (planeByTriangle[triangle] < 0)
                continue;
            int root = FindRoot(parent, triangle);
            componentSizes.TryGetValue(root, out int count);
            count++;
            componentSizes[root] = count;
            largestIsland = Mathf.Max(largestIsland, count);
        }

        int dominantPlaneId = -1;
        int dominantTriangles = 0;
        int significantFamilies = 0;
        List<int> significantPlaneIds = new List<int>();
        List<int> allPlaneIds = new List<int>(familyCounts.Keys);
        allPlaneIds.Sort();
        int significantMinimum = Mathf.Max(2, Mathf.CeilToInt(triangleCount * 0.15f));
        foreach (KeyValuePair<int, int> family in familyCounts)
        {
            if (family.Value > dominantTriangles)
            {
                dominantTriangles = family.Value;
                dominantPlaneId = family.Key;
            }
            if (family.Value >= significantMinimum)
            {
                significantFamilies++;
                significantPlaneIds.Add(family.Key);
            }
        }

        significantPlaneIds.Sort();
        bool hasParallelPair = false;
        bool hasCreasePair = false;
        float minFamilyNormalDot = 1f;
        float maxPlaneSeparation = 0f;
        for (int i = 0; i < significantPlaneIds.Count; i++)
        {
            if (!constraintById.TryGetValue(significantPlaneIds[i], out ScanCoverStablePlaneRegistry.PlaneConstraint a))
                continue;
            for (int j = i + 1; j < significantPlaneIds.Count; j++)
            {
                if (!constraintById.TryGetValue(significantPlaneIds[j], out ScanCoverStablePlaneRegistry.PlaneConstraint b))
                    continue;
                float signedDot = Vector3.Dot(a.Normal.normalized, b.Normal.normalized);
                float normalDot = Mathf.Abs(signedDot);
                minFamilyNormalDot = Mathf.Min(minFamilyNormalDot, normalDot);
                if (normalDot >= 0.95f)
                {
                    hasParallelPair = true;
                    float alignedOffset = signedDot >= 0f ? b.Offset : -b.Offset;
                    maxPlaneSeparation = Mathf.Max(maxPlaneSeparation, Mathf.Abs(a.Offset - alignedOffset));
                }
                else
                {
                    hasCreasePair = true;
                }
            }
        }

        int dominantLargestIsland = 0;
        foreach (KeyValuePair<int, int> component in componentSizes)
        {
            int root = component.Key;
            if (root >= 0 && root < planeByTriangle.Length && planeByTriangle[root] == dominantPlaneId)
                dominantLargestIsland = Mathf.Max(dominantLargestIsland, component.Value);
        }
        int windingMinority = 0;
        foreach (KeyValuePair<int, Vector2Int> winding in familyWindingCounts)
            windingMinority += Mathf.Min(winding.Value.x, winding.Value.y);

        float matchedRatio = matched / (float)Mathf.Max(1, triangleCount);
        bool dominantEnough = dominantTriangles >= triangleCount * settings.StableTriangleRatio;
        bool dominantConnected = dominantLargestIsland >= dominantTriangles * settings.StableTriangleRatio;
        string kind;
        if (matched <= 0 || matchedRatio < 0.25f)
            kind = "unresolved";
        else if (significantFamilies >= 2)
            kind = "multi_plane";
        else if (dominantEnough && dominantConnected)
            kind = "single_plane_outliers";
        else if (componentSizes.Count > 1)
            kind = "fragmented";
        else
            kind = "unresolved";

        string subtype;
        if (kind == "multi_plane")
            subtype = hasParallelPair && hasCreasePair
                ? "mixed_parallel_crease"
                : (hasParallelPair ? "parallel_layers" : "crease");
        else if (kind == "fragmented")
            subtype = familyCounts.Count <= 1 ? "same_plane_islands" : "dominant_plane_islands";
        else if (kind == "single_plane_outliers")
            subtype = "dominant_plane_outliers";
        else
        {
            int largestReject = Mathf.Max(unmatchedNormal,
                Mathf.Max(unmatchedDistance, Mathf.Max(unmatchedExtent,
                Mathf.Max(unmatchedMove, unmatchedAmbiguity))));
            subtype = largestReject == unmatchedNormal ? "missing_normal" :
                (largestReject == unmatchedDistance ? "missing_distance" :
                (largestReject == unmatchedExtent ? "missing_extent" :
                (largestReject == unmatchedMove ? "missing_move" : "ambiguous")));
        }

        StringBuilder planeIds = new StringBuilder();
        for (int i = 0; i < allPlaneIds.Count; i++)
        {
            if (i > 0) planeIds.Append(';');
            planeIds.Append(allPlaneIds[i]);
        }

        geometry.NormalDecompositionKind = kind;
        geometry.NormalDecompositionSubtype = subtype;
        geometry.NormalDecompositionPlaneIds = planeIds.ToString();
        geometry.NormalDecompositionDominantPlaneId = dominantPlaneId;
        geometry.NormalDecompositionPlaneFamilies = familyCounts.Count;
        geometry.NormalDecompositionIslands = componentSizes.Count;
        geometry.NormalDecompositionMatchedTriangles = matched;
        geometry.NormalDecompositionUnresolvedTriangles = triangleCount - matched;
        geometry.NormalDecompositionLargestIslandTriangles = largestIsland;
        geometry.NormalDecompositionWindingMinorityTriangles = windingMinority;
        geometry.NormalDecompositionMinFamilyNormalDot = minFamilyNormalDot;
        geometry.NormalDecompositionMaxPlaneSeparationMeters = maxPlaneSeparation;
        geometry.NormalDecompositionUnmatchedNormalTriangles = unmatchedNormal;
        geometry.NormalDecompositionUnmatchedDistanceTriangles = unmatchedDistance;
        geometry.NormalDecompositionUnmatchedExtentTriangles = unmatchedExtent;
        geometry.NormalDecompositionUnmatchedMoveTriangles = unmatchedMove;
        geometry.NormalDecompositionUnmatchedAmbiguityTriangles = unmatchedAmbiguity;
        geometry.NormalDecompositionNearNormalTriangles = nearNormal;
        geometry.NormalDecompositionNovelNormalTriangles = novelNormal;
        geometry.NormalDecompositionNearDistanceTriangles = nearDistance;
        geometry.NormalDecompositionFarDistanceTriangles = farDistance;
        geometry.NormalDecompositionNearMoveTriangles = nearMove;
        geometry.NormalDecompositionFarMoveTriangles = farMove;

        result.NormalDecompositionCandidatePatches++;
        result.NormalDecompositionMatchedTriangles += matched;
        result.NormalDecompositionUnresolvedTriangles += triangleCount - matched;
        result.NormalDecompositionWindingMinorityTriangles += windingMinority;
        result.NormalDecompositionIslands += componentSizes.Count;
        result.NormalDecompositionUnmatchedNormalTriangles += unmatchedNormal;
        result.NormalDecompositionUnmatchedDistanceTriangles += unmatchedDistance;
        result.NormalDecompositionUnmatchedExtentTriangles += unmatchedExtent;
        result.NormalDecompositionUnmatchedMoveTriangles += unmatchedMove;
        result.NormalDecompositionUnmatchedAmbiguityTriangles += unmatchedAmbiguity;
        if (subtype == "parallel_layers") result.NormalDecompositionParallelLayerPatches++;
        else if (subtype == "crease") result.NormalDecompositionCreasePatches++;
        else if (subtype == "mixed_parallel_crease") result.NormalDecompositionMixedMultiPatches++;
        if (subtype == "crease")
        {
            EvaluateCreaseAtomicSplitShadow(
                geometry,
                planeByTriangle,
                significantPlaneIds,
                constraintById,
                settings,
                ref result);
        }
        if (kind == "single_plane_outliers") result.NormalDecompositionSinglePlanePatches++;
        else if (kind == "multi_plane") result.NormalDecompositionMultiPlanePatches++;
        else if (kind == "fragmented") result.NormalDecompositionFragmentedPatches++;
        else result.NormalDecompositionUnresolvedPatches++;
    }

    private static void EvaluateCreaseAtomicSplitShadow(
        PatchGeometry geometry,
        int[] planeByTriangle,
        List<int> significantPlaneIds,
        Dictionary<int, ScanCoverStablePlaneRegistry.PlaneConstraint> constraintById,
        Settings settings,
        ref Result result)
    {
        int triangleCount = geometry != null ? geometry.TriangleCount : 0;
        if (triangleCount <= 0 || significantPlaneIds == null || significantPlaneIds.Count < 2)
            return;

        result.CreaseAtomicCandidatePatches++;
        HashSet<int> requiredFamilies = new HashSet<int>(significantPlaneIds);
        List<int> sortedRequiredFamilies = new List<int>(requiredFamilies);
        sortedRequiredFamilies.Sort();
        StringBuilder familyIds = new StringBuilder(sortedRequiredFamilies.Count * 4);
        for (int i = 0; i < sortedRequiredFamilies.Count; i++)
        {
            if (i > 0) familyIds.Append(';');
            familyIds.Append(sortedRequiredFamilies[i]);
        }
        int[] safePlaneByTriangle = new int[triangleCount];
        int[] parent = new int[triangleCount];
        for (int i = 0; i < triangleCount; i++)
        {
            safePlaneByTriangle[i] = -1;
            parent[i] = i;
        }
        Dictionary<ulong, int> firstTriangleByEdge = new Dictionary<ulong, int>(triangleCount * 2);
        int topologyRejected = 0;
        float maxMove = 0f;
        for (int triangle = 0; triangle < triangleCount; triangle++)
        {
            int planeId = planeByTriangle[triangle];
            if (!requiredFamilies.Contains(planeId) ||
                !constraintById.TryGetValue(planeId, out ScanCoverStablePlaneRegistry.PlaneConstraint plane))
                continue;
            Vector3 normal = plane.Normal.normalized;
            int offset = triangle * 3;
            int a = geometry.Triangles[offset];
            int b = geometry.Triangles[offset + 1];
            int c = geometry.Triangles[offset + 2];
            Vector3 va = geometry.Vertices[a];
            Vector3 vb = geometry.Vertices[b];
            Vector3 vc = geometry.Vertices[c];
            float da = Vector3.Dot(normal, va) + plane.Offset;
            float db = Vector3.Dot(normal, vb) + plane.Offset;
            float dc = Vector3.Dot(normal, vc) + plane.Offset;
            Vector3 projectedA = va - normal * da;
            Vector3 projectedB = vb - normal * db;
            Vector3 projectedC = vc - normal * dc;
            Vector3 oldCross = Vector3.Cross(vb - va, vc - va);
            Vector3 newCross = Vector3.Cross(projectedB - projectedA, projectedC - projectedA);
            float oldArea = oldCross.magnitude;
            float newArea = newCross.magnitude;
            float normalDot = oldArea > 0.000001f && newArea > 0.000001f
                ? Vector3.Dot(oldCross / oldArea, newCross / newArea)
                : -1f;
            if (!Finite(projectedA) || !Finite(projectedB) || !Finite(projectedC) ||
                newArea < oldArea * settings.PlanePatchMinTriangleAreaRatio ||
                normalDot < settings.PlanePatchMinTriangleNormalDot)
            {
                topologyRejected++;
                continue;
            }
            safePlaneByTriangle[triangle] = planeId;
            maxMove = Mathf.Max(maxMove, Mathf.Max(Mathf.Abs(da), Mathf.Max(Mathf.Abs(db), Mathf.Abs(dc))));
            UnionNormalTriangleEdge(firstTriangleByEdge, parent, safePlaneByTriangle, triangle, a, b);
            UnionNormalTriangleEdge(firstTriangleByEdge, parent, safePlaneByTriangle, triangle, b, c);
            UnionNormalTriangleEdge(firstTriangleByEdge, parent, safePlaneByTriangle, triangle, c, a);
        }

        Dictionary<int, int> componentSizes = new Dictionary<int, int>();
        Dictionary<int, int> componentPlaneIds = new Dictionary<int, int>();
        for (int triangle = 0; triangle < triangleCount; triangle++)
        {
            if (safePlaneByTriangle[triangle] < 0)
                continue;
            int root = FindRoot(parent, triangle);
            componentSizes.TryGetValue(root, out int count);
            componentSizes[root] = count + 1;
            componentPlaneIds[root] = safePlaneByTriangle[triangle];
        }

        int minimumIslandTriangles = Mathf.Max(settings.DoubleSidedTriangles ? 4 : 2,
            Mathf.CeilToInt(triangleCount * 0.05f));
        int retainedTriangles = 0;
        int retainedIslands = 0;
        int tinyIslandTriangles = 0;
        int tinyIslands = 0;
        HashSet<int> retainedFamilies = new HashSet<int>();
        Dictionary<int, int> retainedTrianglesByFamily = new Dictionary<int, int>();
        foreach (KeyValuePair<int, int> component in componentSizes)
        {
            if (component.Value < minimumIslandTriangles)
            {
                tinyIslandTriangles += component.Value;
                tinyIslands++;
                continue;
            }
            retainedTriangles += component.Value;
            retainedIslands++;
            int retainedPlaneId = componentPlaneIds[component.Key];
            retainedFamilies.Add(retainedPlaneId);
            retainedTrianglesByFamily.TryGetValue(retainedPlaneId, out int retainedFamilyTriangles);
            retainedTrianglesByFamily[retainedPlaneId] = retainedFamilyTriangles + component.Value;
        }

        float retainedRatio = retainedTriangles / (float)Mathf.Max(1, triangleCount);
        string status;
        if (retainedTriangles <= 0 && topologyRejected > 0)
            status = "topology";
        else if (retainedFamilies.Count < requiredFamilies.Count)
            status = "family";
        else if (retainedRatio < settings.StableTriangleRatio)
            status = "coverage";
        else
            status = "accepted";

        geometry.CreaseAtomicStatus = status;
        geometry.CreaseAtomicRetainedTriangles = retainedTriangles;
        geometry.CreaseAtomicQuarantinedTriangles = triangleCount - retainedTriangles;
        geometry.CreaseAtomicTopologyRejectedTriangles = topologyRejected;
        geometry.CreaseAtomicRetainedFamilies = retainedFamilies.Count;
        geometry.CreaseAtomicRequiredFamilies = requiredFamilies.Count;
        geometry.CreaseAtomicRetainedIslands = retainedIslands;
        geometry.CreaseAtomicRetainedRatio = retainedRatio;
        geometry.CreaseAtomicMaxMoveMeters = maxMove;
        geometry.CreaseAtomicTinyIslandTriangles = tinyIslandTriangles;
        geometry.CreaseAtomicTinyIslands = tinyIslands;
        geometry.CreaseAtomicFamilyIds = familyIds.ToString();
        geometry.CreaseAtomicFamilies.Clear();
        for (int i = 0; i < sortedRequiredFamilies.Count; i++)
        {
            int planeId = sortedRequiredFamilies[i];
            if (!constraintById.TryGetValue(planeId, out ScanCoverStablePlaneRegistry.PlaneConstraint plane))
                continue;
            retainedTrianglesByFamily.TryGetValue(planeId, out int familyTriangles);
            geometry.CreaseAtomicFamilies.Add(new CreaseFamilyEvidence
            {
                PlaneId = planeId,
                Normal = plane.Normal.normalized,
                Offset = plane.Offset,
                RetainedTriangles = familyTriangles
            });
        }

        result.CreaseAtomicRetainedTriangles += retainedTriangles;
        result.CreaseAtomicQuarantinedTriangles += triangleCount - retainedTriangles;
        result.CreaseAtomicTopologyRejectedTriangles += topologyRejected;
        result.CreaseAtomicRetainedFamilies += retainedFamilies.Count;
        result.CreaseAtomicRetainedIslands += retainedIslands;
        result.CreaseAtomicTinyIslandTriangles += tinyIslandTriangles;
        result.CreaseAtomicTinyIslands += tinyIslands;
        result.CreaseAtomicNearNormalTriangles += geometry.NormalDecompositionNearNormalTriangles;
        result.CreaseAtomicNovelNormalTriangles += geometry.NormalDecompositionNovelNormalTriangles;
        result.CreaseAtomicNearDistanceTriangles += geometry.NormalDecompositionNearDistanceTriangles;
        result.CreaseAtomicFarDistanceTriangles += geometry.NormalDecompositionFarDistanceTriangles;
        result.CreaseAtomicNearMoveTriangles += geometry.NormalDecompositionNearMoveTriangles;
        result.CreaseAtomicFarMoveTriangles += geometry.NormalDecompositionFarMoveTriangles;
        if (status == "accepted") result.CreaseAtomicAcceptedPatches++;
        else if (status == "coverage") result.CreaseAtomicRejectedCoveragePatches++;
        else if (status == "family") result.CreaseAtomicRejectedFamilyPatches++;
        else result.CreaseAtomicRejectedTopologyPatches++;
    }

    private static void UnionNormalTriangleEdge(
        Dictionary<ulong, int> firstTriangleByEdge,
        int[] parent,
        int[] planeByTriangle,
        int triangle,
        int a,
        int b)
    {
        uint min = (uint)Mathf.Min(a, b);
        uint max = (uint)Mathf.Max(a, b);
        ulong edge = ((ulong)min << 32) | max;
        if (firstTriangleByEdge.TryGetValue(edge, out int other))
        {
            if (planeByTriangle[other] == planeByTriangle[triangle])
                UnionRoots(parent, triangle, other);
        }
        else
        {
            firstTriangleByEdge.Add(edge, triangle);
        }
    }

    private static bool TryRescueMoveOverflowComponent(
        PatchGeometry geometry,
        ScanCoverStablePlaneRegistry.PlaneConstraint plane,
        Vector3 planeNormal,
        Settings settings,
        float overflowMaxMove,
        ref Result result)
    {
        int triangleCount = geometry.TriangleCount;
        if (triangleCount < 4)
            return false;
        bool[] eligible = new bool[triangleCount];
        int[] parent = new int[triangleCount];
        Dictionary<ulong, int> firstTriangleByEdge = new Dictionary<ulong, int>(triangleCount * 2);
        for (int triangle = 0; triangle < triangleCount; triangle++)
        {
            parent[triangle] = triangle;
            int offset = triangle * 3;
            int a = geometry.Triangles[offset];
            int b = geometry.Triangles[offset + 1];
            int c = geometry.Triangles[offset + 2];
            Vector3 oldA = geometry.Vertices[a];
            Vector3 oldB = geometry.Vertices[b];
            Vector3 oldC = geometry.Vertices[c];
            float da = Vector3.Dot(planeNormal, oldA) + plane.Offset;
            float db = Vector3.Dot(planeNormal, oldB) + plane.Offset;
            float dc = Vector3.Dot(planeNormal, oldC) + plane.Offset;
            if (Mathf.Abs(da) > settings.PlanePatchMaxVertexMoveMeters ||
                Mathf.Abs(db) > settings.PlanePatchMaxVertexMoveMeters ||
                Mathf.Abs(dc) > settings.PlanePatchMaxVertexMoveMeters)
                continue;
            Vector3 oldCross = Vector3.Cross(oldB - oldA, oldC - oldA);
            float oldArea = oldCross.magnitude;
            if (oldArea <= 0.000001f ||
                Mathf.Abs(Vector3.Dot(oldCross / oldArea, planeNormal)) < settings.PlanePatchMinNormalDot)
                continue;
            Vector3 newA = oldA - planeNormal * da;
            Vector3 newB = oldB - planeNormal * db;
            Vector3 newC = oldC - planeNormal * dc;
            Vector3 newCross = Vector3.Cross(newB - newA, newC - newA);
            float newArea = newCross.magnitude;
            float normalDot = newArea > 0.000001f
                ? Vector3.Dot(oldCross / oldArea, newCross / newArea)
                : -1f;
            if (newArea < oldArea * settings.PlanePatchMinTriangleAreaRatio ||
                normalDot < settings.PlanePatchMinTriangleNormalDot)
                continue;
            eligible[triangle] = true;
            UnionTriangleEdge(firstTriangleByEdge, parent, triangle, a, b);
            UnionTriangleEdge(firstTriangleByEdge, parent, triangle, b, c);
            UnionTriangleEdge(firstTriangleByEdge, parent, triangle, c, a);
        }

        Dictionary<int, int> componentSizes = new Dictionary<int, int>();
        for (int triangle = 0; triangle < triangleCount; triangle++)
        {
            if (!eligible[triangle])
                continue;
            int root = FindRoot(parent, triangle);
            int size = componentSizes.TryGetValue(root, out int count) ? count + 1 : 1;
            componentSizes[root] = size;
        }

        int minimumIslandTriangles = Mathf.Max(settings.DoubleSidedTriangles ? 4 : 2,
            Mathf.CeilToInt(triangleCount * 0.05f));
        HashSet<int> retainedRoots = new HashSet<int>();
        int retainedTriangleCount = 0;
        foreach (KeyValuePair<int, int> component in componentSizes)
        {
            if (component.Value < minimumIslandTriangles)
                continue;
            retainedRoots.Add(component.Key);
            retainedTriangleCount += component.Value;
        }
        float retainedRatio = retainedTriangleCount / (float)Mathf.Max(1, triangleCount);
        if (retainedRoots.Count <= 0 || retainedRatio < settings.StableTriangleRatio)
            return false;

        List<Vector3> retainedVertices = new List<Vector3>(geometry.Vertices.Count);
        List<Color> retainedColors = new List<Color>(geometry.Vertices.Count);
        List<int> retainedTriangles = new List<int>(retainedTriangleCount * 3);
        Dictionary<int, int> remap = new Dictionary<int, int>(geometry.Vertices.Count);
        float appliedMaxMove = 0f;
        for (int triangle = 0; triangle < triangleCount; triangle++)
        {
            if (!eligible[triangle] || !retainedRoots.Contains(FindRoot(parent, triangle)))
                continue;
            int offset = triangle * 3;
            for (int corner = 0; corner < 3; corner++)
            {
                int sourceIndex = geometry.Triangles[offset + corner];
                if (!remap.TryGetValue(sourceIndex, out int retainedIndex))
                {
                    Vector3 source = geometry.Vertices[sourceIndex];
                    float signed = Vector3.Dot(planeNormal, source) + plane.Offset;
                    retainedIndex = retainedVertices.Count;
                    remap.Add(sourceIndex, retainedIndex);
                    retainedVertices.Add(source - planeNormal * signed);
                    retainedColors.Add(sourceIndex < geometry.Colors.Count ? geometry.Colors[sourceIndex] : Color.white);
                    appliedMaxMove = Mathf.Max(appliedMaxMove, Mathf.Abs(signed));
                }
                retainedTriangles.Add(retainedIndex);
            }
        }

        int quarantined = triangleCount - retainedTriangleCount;
        geometry.Vertices.Clear();
        geometry.Vertices.AddRange(retainedVertices);
        geometry.Colors.Clear();
        geometry.Colors.AddRange(retainedColors);
        geometry.Triangles.Clear();
        geometry.Triangles.AddRange(retainedTriangles);
        geometry.Remap.Clear();
        geometry.PlaneConstraintId = plane.Id;
        geometry.PlaneConstraintApplied = true;
        geometry.PlaneConstraintRejectReason = "move_split";
        geometry.PlaneConstraintNormalDot = Mathf.Abs(Vector3.Dot(geometry.AverageNormal, planeNormal));
        geometry.PlaneConstraintCentroidDistanceMeters =
            Mathf.Abs(Vector3.Dot(planeNormal, geometry.Centroid) + plane.Offset);
        geometry.PlaneConstraintMaxMoveMeters = appliedMaxMove;
        geometry.PlaneConstraintQuarantinedTriangles = quarantined;
        geometry.PlaneConstraintOriginalTriangles = triangleCount;
        geometry.PlaneConstraintRetainedIslands = retainedRoots.Count;
        AnalyzeGeometry(geometry, settings);
        result.PlaneConstraintAppliedPatches++;
        result.PlaneConstraintAppliedTriangles += geometry.TriangleCount;
        result.PlaneConstraintMoveSplitRescuedPatches++;
        result.PlaneConstraintQuarantinedTriangles += quarantined;
        result.PlaneConstraintRetainedIslands += retainedRoots.Count;
        if (retainedRoots.Count > 1)
            result.PlaneConstraintMultiIslandRescuedPatches++;
        return true;
    }

    private static void UnionTriangleEdge(
        Dictionary<ulong, int> firstTriangleByEdge,
        int[] parent,
        int triangle,
        int a,
        int b)
    {
        uint min = (uint)Mathf.Min(a, b);
        uint max = (uint)Mathf.Max(a, b);
        ulong key = ((ulong)min << 32) | max;
        if (firstTriangleByEdge.TryGetValue(key, out int other))
            UnionRoots(parent, triangle, other);
        else
            firstTriangleByEdge.Add(key, triangle);
    }

    private static int FindRoot(int[] parent, int value)
    {
        int root = value;
        while (parent[root] != root)
            root = parent[root];
        while (parent[value] != value)
        {
            int next = parent[value];
            parent[value] = root;
            value = next;
        }
        return root;
    }

    private static void UnionRoots(int[] parent, int a, int b)
    {
        int rootA = FindRoot(parent, a);
        int rootB = FindRoot(parent, b);
        if (rootA != rootB)
            parent[rootB] = rootA;
    }

    private static int RemapVertex(
        int sourceIndex,
        List<Vector3> vertices,
        List<Color> colors,
        PatchGeometry patch)
    {
        if (patch.Remap.TryGetValue(sourceIndex, out int localIndex))
            return localIndex;
        localIndex = patch.Vertices.Count;
        patch.Remap.Add(sourceIndex, localIndex);
        patch.Vertices.Add(vertices[sourceIndex]);
        patch.Colors.Add(colors != null && sourceIndex < colors.Count ? colors[sourceIndex] : Color.white);
        return localIndex;
    }

    private static void AnalyzeGeometry(PatchGeometry geometry, Settings settings)
    {
        if (geometry.Vertices.Count <= 0 || geometry.Triangles.Count < 3)
            return;
        Vector3 center = Vector3.zero;
        Vector3 min = geometry.Vertices[0];
        Vector3 max = geometry.Vertices[0];
        for (int i = 0; i < geometry.Vertices.Count; i++)
        {
            Vector3 point = geometry.Vertices[i];
            center += point;
            min = Vector3.Min(min, point);
            max = Vector3.Max(max, point);
        }
        center /= geometry.Vertices.Count;

        Vector3 normalSum = Vector3.zero;
        float areaSum = 0f;
        List<Vector3> triangleNormals = new List<Vector3>(geometry.TriangleCount);
        for (int i = 0; i + 2 < geometry.Triangles.Count; i += 3)
        {
            Vector3 a = geometry.Vertices[geometry.Triangles[i]];
            Vector3 b = geometry.Vertices[geometry.Triangles[i + 1]];
            Vector3 c = geometry.Vertices[geometry.Triangles[i + 2]];
            Vector3 cross = Vector3.Cross(b - a, c - a);
            float area = cross.magnitude;
            if (area <= 0.000001f)
                continue;
            Vector3 normal = cross / area;
            if (normalSum.sqrMagnitude > 0.000001f && Vector3.Dot(normal, normalSum) < 0f)
                normal = -normal;
            normalSum += normal * area;
            areaSum += area;
            triangleNormals.Add(normal);
        }
        Vector3 averageNormal = normalSum.sqrMagnitude > 0.000001f ? normalSum.normalized : Vector3.up;
        float residual = 0f;
        for (int i = 0; i < geometry.Vertices.Count; i++)
            residual += Mathf.Abs(Vector3.Dot(averageNormal, geometry.Vertices[i] - center));
        residual /= Mathf.Max(1, geometry.Vertices.Count);

        float coherence = 0f;
        for (int i = 0; i < triangleNormals.Count; i++)
            coherence += Mathf.Abs(Vector3.Dot(averageNormal, triangleNormals[i]));
        coherence /= Mathf.Max(1, triangleNormals.Count);

        CountEdges(
            geometry.Triangles,
            settings.DoubleSidedTriangles,
            out int boundary,
            out int nonManifold);
        geometry.Centroid = center;
        geometry.AverageNormal = averageNormal;
        geometry.BoundsMin = min;
        geometry.BoundsMax = max;
        geometry.PlaneResidualMeters = residual;
        geometry.NormalCoherence = coherence;
        geometry.BoundaryEdges = boundary;
        geometry.NonManifoldEdges = nonManifold;
        geometry.SurfaceKind = residual <= settings.PlanarResidualMeters &&
                               coherence >= settings.PlanarNormalCoherence
            ? PatchSurfaceKind.Planar
            : (coherence >= 0.55f ? PatchSurfaceKind.Curved : PatchSurfaceKind.Uncertain);
    }

    private static bool GeometryCompatible(PatchGeometry a, PatchGeometry b, Settings settings)
    {
        if (a == null || b == null || a.TriangleCount <= 0 || b.TriangleCount <= 0)
            return false;
        float ratio = b.TriangleCount / (float)Mathf.Max(1, a.TriangleCount);
        if (ratio < settings.StableTriangleRatio || ratio > 1f / settings.StableTriangleRatio)
            return false;
        if (Vector3.Distance(a.Centroid, b.Centroid) > settings.StableCentroidDistanceMeters)
            return false;
        if (Mathf.Abs(Vector3.Dot(a.AverageNormal, b.AverageNormal)) < settings.StableNormalDot)
            return false;
        return a.NonManifoldEdges == b.NonManifoldEdges;
    }

    private static bool GeometrySafeForPublication(PatchGeometry geometry)
    {
        return geometry != null &&
               geometry.TriangleCount > 0 &&
               geometry.NonManifoldEdges <= 0 &&
               Finite(geometry.Centroid) &&
               Finite(geometry.AverageNormal);
    }

    private static bool HasActionablePlaneReject(PatchGeometry geometry)
    {
        if (geometry == null || geometry.PlaneConstraintApplied ||
            geometry.PlaneResidualBeforeConstraintMeters <= 0f)
            return false;
        string reason = geometry.PlaneConstraintRejectReason;
        return reason == "normal" || reason == "distance" || reason == "extent" ||
               reason == "ambiguous" || reason == "move" || reason == "topology";
    }

    private void BuildCreaseRegionalEvidence(Settings settings, ref Result result)
    {
        _creaseRegionalEvidence.Clear();
        foreach (KeyValuePair<Vector3Int, PatchGeometry> pair in _candidates)
        {
            PatchGeometry center = pair.Value;
            if (!CanContributeCreaseRegionalEvidence(center))
                continue;

            result.CreaseRegionalCandidatePatches++;
            PatchGeometry regional = CloneGeometry(center);
            regional.CreaseRegionalMemberPatches = 1;
            regional.CreaseRegionalSourceKey = pair.Key;
            int retained = center.CreaseAtomicRetainedTriangles;
            int original = center.CreaseAtomicRetainedTriangles + center.CreaseAtomicQuarantinedTriangles;

            for (int offsetIndex = 0; offsetIndex < SixConnectedOffsets.Length; offsetIndex++)
            {
                Vector3Int neighborKey = pair.Key + SixConnectedOffsets[offsetIndex];
                if (!_candidates.TryGetValue(neighborKey, out PatchGeometry neighbor) ||
                    !CanContributeCreaseRegionalEvidence(neighbor) ||
                    !CreaseFamiliesCompatible(
                        regional, neighbor, settings,
                        out _, out _, out _, out _, false))
                    continue;

                MergeCreaseFamilyTriangleCounts(regional, neighbor, settings);
                retained += neighbor.CreaseAtomicRetainedTriangles;
                original += neighbor.CreaseAtomicRetainedTriangles + neighbor.CreaseAtomicQuarantinedTriangles;
                regional.CreaseRegionalMemberPatches++;
            }

            regional.CreaseAtomicRetainedTriangles = retained;
            regional.CreaseAtomicQuarantinedTriangles = Mathf.Max(0, original - retained);
            regional.CreaseAtomicRetainedRatio = retained / (float)Mathf.Max(1, original);
            regional.CreaseRegionalRetainedRatio = regional.CreaseAtomicRetainedRatio;
            int retainedFamilies = CountRetainedCreaseFamilies(regional);
            regional.CreaseAtomicRetainedFamilies = retainedFamilies;
            regional.CreaseRegionalRetainedFamilies = retainedFamilies;
            regional.CreaseRegionalRequiredFamilies = regional.CreaseAtomicRequiredFamilies;
            if (regional.CreaseAtomicTopologyRejectedTriangles > 0)
                regional.CreaseAtomicStatus = "topology";
            else if (retainedFamilies < regional.CreaseAtomicRequiredFamilies)
                regional.CreaseAtomicStatus = "family";
            else if (regional.CreaseAtomicRetainedRatio < settings.StableTriangleRatio)
                regional.CreaseAtomicStatus = "coverage";
            else
                regional.CreaseAtomicStatus = "accepted";
            regional.CreaseRegionalStatus = regional.CreaseAtomicStatus;
            if (regional.CreaseRegionalMemberPatches > 1)
                result.CreaseRegionalAggregatedPatches++;
            _creaseRegionalEvidence[pair.Key] = regional;
        }
    }

    private void BuildCreaseProbationAssignments(Settings settings, ref Result result)
    {
        _creaseProbationAssignments.Clear();
        List<CreaseProbationSeed> seeds = new List<CreaseProbationSeed>(_tracks.Count);
        foreach (KeyValuePair<Vector3Int, Track> pair in _tracks)
        {
            Track track = pair.Value;
            if (track.CreaseAtomicProbationHits <= 0 || track.CreaseAtomicProbationGeometry == null)
                continue;
            seeds.Add(new CreaseProbationSeed
            {
                Key = pair.Key,
                Geometry = CloneGeometry(track.CreaseAtomicProbationGeometry),
                FamilyIds = track.CreaseAtomicProbationFamilyIds,
                Hits = track.CreaseAtomicProbationHits,
                RetainedTriangles = track.CreaseAtomicProbationRetainedTriangles
            });
        }

        List<Vector3Int> candidateKeys = new List<Vector3Int>(_creaseRegionalEvidence.Keys);
        candidateKeys.Sort(ComparePatchKeys);
        for (int candidateIndex = 0; candidateIndex < candidateKeys.Count; candidateIndex++)
        {
            Vector3Int key = candidateKeys[candidateIndex];
            PatchGeometry candidate = _creaseRegionalEvidence[key];
            if (candidate.CreaseAtomicStatus != "accepted")
                continue;

            CreaseProbationSeed best = null;
            float bestScore = float.PositiveInfinity;
            float secondScore = float.PositiveInfinity;
            for (int seedIndex = 0; seedIndex < seeds.Count; seedIndex++)
            {
                CreaseProbationSeed seed = seeds[seedIndex];
                if (seed.Claimed || PatchManhattanDistance(seed.Key, key) > 1 ||
                    !CreaseFamiliesCompatible(
                        seed.Geometry, candidate, settings,
                        out float minNormalDot, out float maxDistanceDelta,
                        out _, out _, false))
                    continue;
                float score = (seed.Key == key ? -10f : 0f) +
                              (1f - minNormalDot) +
                              maxDistanceDelta / Mathf.Max(0.001f, settings.StableCentroidDistanceMeters);
                if (score < bestScore)
                {
                    secondScore = bestScore;
                    bestScore = score;
                    best = seed;
                }
                else if (score < secondScore)
                {
                    secondScore = score;
                }
            }

            // Same-key continuity is authoritative.  A migrated neighbor must be
            // uniquely better than any competing regional trial.
            if (best == null || (best.Key != key &&
                secondScore - bestScore < settings.PlanePatchAmbiguityScoreMargin))
                continue;

            best.Claimed = true;
            candidate.CreaseRegionalMigrated = best.Key != key;
            candidate.CreaseRegionalSourceKey = best.Key;
            _creaseProbationAssignments[key] = new CreaseProbationAssignment
            {
                SourceKey = best.Key,
                Geometry = best.Geometry,
                FamilyIds = best.FamilyIds,
                Hits = best.Hits,
                RetainedTriangles = best.RetainedTriangles
            };
            if (best.Key != key)
            {
                result.CreaseRegionalMigratedPatches++;
                if (_tracks.TryGetValue(best.Key, out Track donor))
                    ResetCreaseAtomicProbation(donor);
            }
        }
    }

    private void ApplyCreaseProbationAssignment(
        Track track,
        Vector3Int key,
        PatchGeometry candidate,
        ref Result result)
    {
        if (track == null || candidate == null ||
            !_creaseProbationAssignments.TryGetValue(key, out CreaseProbationAssignment assignment))
            return;
        track.CreaseAtomicProbationGeometry = CloneGeometry(assignment.Geometry);
        track.CreaseAtomicProbationFamilyIds = assignment.FamilyIds;
        track.CreaseAtomicProbationHits = assignment.Hits;
        track.CreaseAtomicProbationRetainedTriangles = assignment.RetainedTriangles;
        track.CreaseAtomicProbationOriginKey = assignment.SourceKey;
        candidate.CreaseRegionalMigrated = assignment.SourceKey != key;
        candidate.CreaseRegionalSourceKey = assignment.SourceKey;
    }

    private static void CaptureCreaseRegionalDiagnostics(
        Track track,
        PatchGeometry candidate,
        bool hasRegionalEvidence)
    {
        if (track == null)
            return;
        track.LastCreaseRegionalEvidence = hasRegionalEvidence && candidate != null
            ? CloneGeometry(candidate)
            : null;
    }

    private static bool CanContributeCreaseRegionalEvidence(PatchGeometry geometry)
    {
        return geometry != null &&
               (geometry.CreaseAtomicStatus == "accepted" ||
                geometry.CreaseAtomicStatus == "coverage" ||
                geometry.CreaseAtomicStatus == "family") &&
               geometry.CreaseAtomicTopologyRejectedTriangles == 0 &&
               geometry.CreaseAtomicFamilies.Count >= 2;
    }

    private static int CountRetainedCreaseFamilies(PatchGeometry geometry)
    {
        if (geometry == null) return 0;
        int count = 0;
        for (int i = 0; i < geometry.CreaseAtomicFamilies.Count; i++)
        {
            if (geometry.CreaseAtomicFamilies[i].RetainedTriangles > 0)
                count++;
        }
        return count;
    }

    private static void MergeCreaseFamilyTriangleCounts(
        PatchGeometry target,
        PatchGeometry source,
        Settings settings)
    {
        bool[] used = new bool[source.CreaseAtomicFamilies.Count];
        for (int i = 0; i < target.CreaseAtomicFamilies.Count; i++)
        {
            CreaseFamilyEvidence a = target.CreaseAtomicFamilies[i];
            int bestIndex = -1;
            float bestScore = float.PositiveInfinity;
            for (int j = 0; j < source.CreaseAtomicFamilies.Count; j++)
            {
                if (used[j]) continue;
                CreaseFamilyEvidence b = source.CreaseAtomicFamilies[j];
                if (!TryMeasureCreaseFamilyPair(a, b, settings, out CreaseFamilyMatch match))
                    continue;
                float score = (a.PlaneId == b.PlaneId ? -10f : 0f) + match.Score;
                if (score < bestScore)
                {
                    bestScore = score;
                    bestIndex = j;
                }
            }
            if (bestIndex < 0) continue;
            used[bestIndex] = true;
            a.RetainedTriangles += source.CreaseAtomicFamilies[bestIndex].RetainedTriangles;
        }
    }

    private static int PatchManhattanDistance(Vector3Int a, Vector3Int b)
    {
        return Mathf.Abs(a.x - b.x) + Mathf.Abs(a.y - b.y) + Mathf.Abs(a.z - b.z);
    }

    private static int ComparePatchKeys(Vector3Int a, Vector3Int b)
    {
        int compare = a.x.CompareTo(b.x);
        if (compare != 0) return compare;
        compare = a.y.CompareTo(b.y);
        return compare != 0 ? compare : a.z.CompareTo(b.z);
    }

    private static void UpdateCreaseTinyIslandEvidence(
        Track track,
        PatchGeometry candidate,
        Settings settings,
        ref Result result)
    {
        if (track == null || candidate == null || candidate.CreaseAtomicTinyIslandTriangles <= 0)
        {
            if (track != null)
                track.CreaseAtomicTinyIslandStableHits = 0;
            return;
        }
        PatchGeometry prior = track.LastCandidate;
        bool sameFamilyType = prior != null && prior.CreaseAtomicTinyIslandTriangles > 0 &&
                              prior.NormalDecompositionPlaneIds == candidate.NormalDecompositionPlaneIds;
        bool compatibleCount = false;
        if (sameFamilyType)
        {
            float ratio = candidate.CreaseAtomicTinyIslandTriangles /
                          (float)Mathf.Max(1, prior.CreaseAtomicTinyIslandTriangles);
            compatibleCount = ratio >= settings.StableTriangleRatio &&
                              ratio <= 1f / settings.StableTriangleRatio;
        }
        track.CreaseAtomicTinyIslandStableHits = sameFamilyType && compatibleCount
            ? track.CreaseAtomicTinyIslandStableHits + 1
            : 1;
        if (track.CreaseAtomicTinyIslandStableHits >= 2)
            result.CreaseAtomicPersistentTinyIslandPatches++;
    }

    private static void UpdateCreaseAtomicProbation(
        Track track,
        PatchGeometry candidate,
        Settings settings,
        ref Result result)
    {
        if (track == null)
            return;

        bool accepted = candidate != null &&
                        candidate.CreaseAtomicStatus == "accepted" &&
                        candidate.CreaseAtomicTopologyRejectedTriangles == 0 &&
                        candidate.CreaseAtomicRetainedRatio >= settings.StableTriangleRatio &&
                        !string.IsNullOrEmpty(candidate.CreaseAtomicFamilyIds);
        if (!accepted)
        {
            if (track.CreaseAtomicProbationHits > 0)
                result.CreaseAtomicProbationResetPatches++;
            ResetCreaseAtomicProbation(track);
            return;
        }

        bool compatibleFamilies = track.CreaseAtomicProbationHits > 0 &&
                                  CreaseFamiliesCompatible(
                                      track.CreaseAtomicProbationGeometry,
                                      candidate,
                                      settings,
                                      out _, out _, out _, out _);

        if (compatibleFamilies)
        {
            track.CreaseAtomicProbationHits++;
            track.CreaseAtomicProbationFamilyIds = candidate.CreaseAtomicFamilyIds;
        }
        else
        {
            if (track.CreaseAtomicProbationHits > 0)
                result.CreaseAtomicProbationResetPatches++;
            track.CreaseAtomicProbationHits = 1;
            track.CreaseAtomicProbationFamilyIds = candidate.CreaseAtomicFamilyIds;
        }
        track.CreaseAtomicProbationRetainedTriangles = candidate.CreaseAtomicRetainedTriangles;
        track.CreaseAtomicProbationGeometry = CloneGeometry(candidate);
        CountCreaseAtomicProbation(track, ref result);
        if (track.CreaseAtomicProbationHits >= 2 && candidate.CreaseRegionalMemberPatches > 1)
            result.CreaseRegionalReadyPatches++;
    }

    private static void AnalyzeCreaseTransition(
        PatchGeometry prior,
        PatchGeometry candidate,
        Settings settings,
        ref Result result)
    {
        if (prior == null || candidate == null || prior.CreaseAtomicStatus == "none")
            return;

        result.CreaseTransitionCandidatePatches++;
        float triangleRatio = candidate.TriangleCount /
                              (float)Mathf.Max(1, prior.TriangleCount);
        float centroidShift = Vector3.Distance(prior.Centroid, candidate.Centroid);
        float normalDot = Mathf.Abs(Vector3.Dot(prior.AverageNormal, candidate.AverageNormal));
        bool triangleCompatible = triangleRatio >= settings.StableTriangleRatio &&
                                  triangleRatio <= 1f / settings.StableTriangleRatio;
        float familyMinNormalDot = 0f;
        float familyMaxDistanceDelta = 0f;
        float familyMinTriangleRatio = 0f;
        int familyRemappedCount = 0;
        bool familyCompatible = CreaseFamiliesCompatible(
            prior,
            candidate,
            settings,
            out familyMinNormalDot,
            out familyMaxDistanceDelta,
            out familyMinTriangleRatio,
            out familyRemappedCount);
        bool contentCompatible = triangleCompatible &&
                                 centroidShift <= settings.StableCentroidDistanceMeters &&
                                 familyCompatible;
        bool nearestInPriorFamily = FamilyContainsPlane(
            prior.CreaseAtomicFamilyIds,
            candidate.PlaneConstraintNearestNormalPlaneId);

        candidate.CreaseTransitionFromStatus = prior.CreaseAtomicStatus;
        candidate.CreaseTransitionFromFamilyIds = prior.CreaseAtomicFamilyIds;
        candidate.CreaseTransitionNearestInPriorFamily = nearestInPriorFamily;
        candidate.CreaseTransitionTriangleRatio = triangleRatio;
        candidate.CreaseTransitionCentroidShiftMeters = centroidShift;
        candidate.CreaseTransitionNormalDot = normalDot;
        candidate.CreaseTransitionRetainedRatioDelta =
            candidate.CreaseAtomicRetainedRatio - prior.CreaseAtomicRetainedRatio;
        candidate.CreaseTransitionContentCompatible = contentCompatible;
        candidate.CreaseTransitionFamilyMinNormalDot = familyMinNormalDot;
        candidate.CreaseTransitionFamilyMaxDistanceDeltaMeters = familyMaxDistanceDelta;
        candidate.CreaseTransitionFamilyMinTriangleRatio = familyMinTriangleRatio;
        candidate.CreaseTransitionFamilyRemappedCount = familyRemappedCount;
        if (familyCompatible && familyRemappedCount > 0)
            result.CreaseTransitionCanonicalRemapPatches++;

        string kind;
        if (candidate.CreaseAtomicStatus == "accepted" && contentCompatible)
        {
            kind = "stable";
            result.CreaseTransitionStablePatches++;
        }
        else if (candidate.PlaneConstraintRejectReason == "distance")
        {
            if (nearestInPriorFamily)
            {
                kind = "same_family_distance";
                result.CreaseTransitionSameFamilyDistancePatches++;
            }
            else
            {
                kind = "new_family_distance";
                result.CreaseTransitionNewFamilyDistancePatches++;
            }
        }
        else if (candidate.CreaseAtomicStatus == "coverage" && familyCompatible)
        {
            kind = "coverage_drop";
            result.CreaseTransitionCoverageDropPatches++;
        }
        else if (candidate.CreaseAtomicStatus != "none" && !familyCompatible)
        {
            kind = "family_changed";
            result.CreaseTransitionFamilyChangedPatches++;
        }
        else if (candidate.PlaneConstraintApplied)
        {
            kind = "resolved_single_plane";
            result.CreaseTransitionOtherPatches++;
        }
        else if (candidate.PlaneConstraintRejectReason == "normal")
        {
            kind = "normal_gate";
            result.CreaseTransitionNormalGatePatches++;
        }
        else if (candidate.CreaseAtomicStatus == "none")
        {
            kind = "family_lost";
            result.CreaseTransitionFamilyLostPatches++;
        }
        else
        {
            kind = candidate.CreaseAtomicStatus;
            result.CreaseTransitionOtherPatches++;
        }
        candidate.CreaseTransitionKind = kind;
        if (!contentCompatible)
            result.CreaseTransitionContentShiftPatches++;
    }

    private static bool FamilyContainsPlane(string familyIds, int planeId)
    {
        if (planeId < 0 || string.IsNullOrEmpty(familyIds))
            return false;
        string target = planeId.ToString(CultureInfo.InvariantCulture);
        int start = 0;
        while (start < familyIds.Length)
        {
            int end = familyIds.IndexOf(';', start);
            if (end < 0) end = familyIds.Length;
            if (end - start == target.Length &&
                string.CompareOrdinal(familyIds, start, target, 0, target.Length) == 0)
                return true;
            start = end + 1;
        }
        return false;
    }

    private static bool CreaseFamiliesCompatible(
        PatchGeometry prior,
        PatchGeometry candidate,
        Settings settings,
        out float minNormalDot,
        out float maxDistanceDeltaMeters,
        out float minTriangleRatio,
        out int remappedCount,
        bool enforceTriangleRatio = true)
    {
        minNormalDot = 1f;
        maxDistanceDeltaMeters = 0f;
        minTriangleRatio = 1f;
        remappedCount = 0;
        if (prior == null || candidate == null ||
            prior.CreaseAtomicFamilies.Count <= 0 ||
            prior.CreaseAtomicFamilies.Count != candidate.CreaseAtomicFamilies.Count)
            return false;

        int familyCount = prior.CreaseAtomicFamilies.Count;
        bool[] priorMatched = new bool[familyCount];
        bool[] candidateMatched = new bool[familyCount];
        List<CreaseFamilyMatch> matches = new List<CreaseFamilyMatch>(familyCount);

        // Preserve a canonical ID when it remains geometrically valid.
        for (int i = 0; i < familyCount; i++)
        {
            CreaseFamilyEvidence a = prior.CreaseAtomicFamilies[i];
            for (int j = 0; j < familyCount; j++)
            {
                CreaseFamilyEvidence b = candidate.CreaseAtomicFamilies[j];
                if (a.PlaneId != b.PlaneId ||
                    !TryMeasureCreaseFamilyPair(a, b, settings, out CreaseFamilyMatch match))
                    continue;
                match.PriorIndex = i;
                match.CandidateIndex = j;
                matches.Add(match);
                priorMatched[i] = true;
                candidateMatched[j] = true;
                break;
            }
        }

        // If the registry merged an alias into a new canonical ID, recover identity
        // only through an unambiguous geometric one-to-one match.
        List<CreaseFamilyMatch> remapCandidates = new List<CreaseFamilyMatch>(familyCount * familyCount);
        for (int i = 0; i < familyCount; i++)
        {
            if (priorMatched[i]) continue;
            for (int j = 0; j < familyCount; j++)
            {
                if (candidateMatched[j] ||
                    !TryMeasureCreaseFamilyPair(
                        prior.CreaseAtomicFamilies[i],
                        candidate.CreaseAtomicFamilies[j],
                        settings,
                        out CreaseFamilyMatch match))
                    continue;
                match.PriorIndex = i;
                match.CandidateIndex = j;
                remapCandidates.Add(match);
            }
        }
        remapCandidates.Sort((a, b) => a.Score.CompareTo(b.Score));
        for (int index = 0; index < remapCandidates.Count; index++)
        {
            CreaseFamilyMatch match = remapCandidates[index];
            if (priorMatched[match.PriorIndex] || candidateMatched[match.CandidateIndex])
                continue;
            bool ambiguous = false;
            for (int otherIndex = index + 1; otherIndex < remapCandidates.Count; otherIndex++)
            {
                CreaseFamilyMatch other = remapCandidates[otherIndex];
                if (priorMatched[other.PriorIndex] || candidateMatched[other.CandidateIndex])
                    continue;
                if ((other.PriorIndex == match.PriorIndex ||
                     other.CandidateIndex == match.CandidateIndex) &&
                    other.Score - match.Score < settings.PlanePatchAmbiguityScoreMargin)
                {
                    ambiguous = true;
                    break;
                }
            }
            if (ambiguous)
                continue;
            matches.Add(match);
            priorMatched[match.PriorIndex] = true;
            candidateMatched[match.CandidateIndex] = true;
            remappedCount++;
        }

        if (matches.Count != familyCount)
        {
            minTriangleRatio = 0f;
            return false;
        }

        bool compatible = true;
        for (int i = 0; i < matches.Count; i++)
        {
            CreaseFamilyMatch match = matches[i];
            CreaseFamilyEvidence a = prior.CreaseAtomicFamilies[match.PriorIndex];
            CreaseFamilyEvidence b = candidate.CreaseAtomicFamilies[match.CandidateIndex];
            if (a.RetainedTriangles <= 0 || b.RetainedTriangles <= 0)
            {
                if (enforceTriangleRatio)
                    compatible = false;
                minTriangleRatio = 0f;
            }
            float symmetricTriangleRatio = 0f;
            if (a.RetainedTriangles > 0 && b.RetainedTriangles > 0)
            {
                float triangleRatio = b.RetainedTriangles / (float)Mathf.Max(1, a.RetainedTriangles);
                symmetricTriangleRatio = Mathf.Min(triangleRatio, 1f / Mathf.Max(0.000001f, triangleRatio));
                minTriangleRatio = Mathf.Min(minTriangleRatio, symmetricTriangleRatio);
            }
            minNormalDot = Mathf.Min(minNormalDot, match.NormalDot);
            maxDistanceDeltaMeters = Mathf.Max(maxDistanceDeltaMeters, match.DistanceDeltaMeters);
            if (enforceTriangleRatio && symmetricTriangleRatio < settings.StableTriangleRatio)
                compatible = false;
        }
        return compatible;
    }

    private static bool TryMeasureCreaseFamilyPair(
        CreaseFamilyEvidence a,
        CreaseFamilyEvidence b,
        Settings settings,
        out CreaseFamilyMatch match)
    {
        match = new CreaseFamilyMatch();
        if (a == null || b == null || a.Normal.sqrMagnitude < 0.5f || b.Normal.sqrMagnitude < 0.5f)
            return false;
        float signedNormalDot = Vector3.Dot(a.Normal.normalized, b.Normal.normalized);
        float normalDot = Mathf.Abs(signedNormalDot);
        float alignedOffset = signedNormalDot >= 0f ? b.Offset : -b.Offset;
        float distanceDelta = Mathf.Abs(a.Offset - alignedOffset);
        if (normalDot < settings.StableNormalDot ||
            distanceDelta > settings.StableCentroidDistanceMeters)
            return false;
        match.NormalDot = normalDot;
        match.DistanceDeltaMeters = distanceDelta;
        match.Score = (1f - normalDot) +
                      distanceDelta / Mathf.Max(0.001f, settings.StableCentroidDistanceMeters);
        return true;
    }

    private static void CountCreaseAtomicProbation(Track track, ref Result result)
    {
        if (track == null || track.CreaseAtomicProbationHits <= 0)
            return;
        if (track.CreaseAtomicProbationHits >= 2)
            result.CreaseAtomicProbationReadyPatches++;
        else
            result.CreaseAtomicProbationPendingPatches++;
    }

    private static void ResetCreaseAtomicProbation(Track track)
    {
        track.CreaseAtomicProbationFamilyIds = "";
        track.CreaseAtomicProbationHits = 0;
        track.CreaseAtomicProbationRetainedTriangles = 0;
        track.CreaseAtomicProbationGeometry = null;
    }

    private static bool RequiresRejectedPlaneHold(PatchGeometry geometry, Settings settings)
    {
        return HasActionablePlaneReject(geometry) &&
               geometry.PlaneResidualBeforeConstraintMeters >= settings.PlaneRejectWithdrawResidualMeters;
    }

    private static void UpdatePlaneRejectLease(
        Track track,
        PatchGeometry candidate,
        Settings settings,
        ref Result result)
    {
        if (HasActionablePlaneReject(candidate))
        {
            string reason = candidate.PlaneConstraintRejectReason;
            track.PlaneRejectHits = track.PlaneRejectLeaseReason == reason
                ? track.PlaneRejectHits + 1
                : 1;
            track.PlaneRejectLeaseReason = reason;
            result.PlaneRejectLeasePatches++;
            return;
        }

        bool positiveRecovery = candidate != null &&
                                (candidate.PlaneConstraintApplied ||
                                 candidate.PlaneResidualMeters < settings.PlaneRejectWithdrawResidualMeters);
        if (track.PlaneRejectHits > 0 && !positiveRecovery)
        {
            result.PlaneRejectLeasePatches++;
            return;
        }
        if (track.PlaneRejectHits > 0)
            result.PlaneRejectRecoveredPatches++;
        track.PlaneRejectHits = 0;
        track.PlaneRejectLeaseReason = "none";
        track.PlaneRejectWithheld = false;
    }

    private static bool ShouldWithdrawRejectedPublication(
        Track track,
        PatchGeometry candidate,
        Settings settings)
    {
        return track != null && track.Published != null &&
               track.PlaneRejectHits >= settings.PlaneRejectMaxHoldRebuilds &&
               track.Published.PlaneResidualMeters >= settings.PlaneRejectWithdrawResidualMeters &&
               RequiresRejectedPlaneHold(candidate, settings);
    }

    private static bool ReplacementDoesNotRegress(
        PatchGeometry published,
        PatchGeometry candidate,
        Settings settings)
    {
        if (candidate.NonManifoldEdges > published.NonManifoldEdges)
            return false;
        float boundaryLimit = published.BoundaryEdges * settings.ReplacementBoundarySlackRatio + 2f;
        bool planeSplitSafe = candidate.PlaneConstraintApplied &&
                              candidate.PlaneConstraintQuarantinedTriangles > 0 &&
                              candidate.PlaneConstraintOriginalTriangles > 0 &&
                              candidate.TriangleCount >= candidate.PlaneConstraintOriginalTriangles * settings.StableTriangleRatio;
        if (!planeSplitSafe && candidate.BoundaryEdges > boundaryLimit)
            return false;
        if (published.SurfaceKind == PatchSurfaceKind.Planar &&
            candidate.PlaneResidualMeters >
            published.PlaneResidualMeters + settings.ReplacementResidualSlackMeters)
            return false;
        return true;
    }

    private static bool CorrectionCandidateAcceptable(
        PatchGeometry published,
        PatchGeometry candidate,
        Settings settings)
    {
        if (!GeometrySafeForPublication(candidate) || published == null)
            return false;
        if (candidate.NonManifoldEdges > published.NonManifoldEdges)
            return false;
        float boundaryLimit = published.BoundaryEdges * settings.ReplacementBoundarySlackRatio + 2f;
        bool planeSplitSafe = candidate.PlaneConstraintApplied &&
                              candidate.PlaneConstraintQuarantinedTriangles > 0 &&
                              candidate.PlaneConstraintOriginalTriangles > 0 &&
                              candidate.TriangleCount >= candidate.PlaneConstraintOriginalTriangles * settings.StableTriangleRatio;
        if (!planeSplitSafe && candidate.BoundaryEdges > boundaryLimit)
            return false;
        if (candidate.PlaneResidualMeters >
            published.PlaneResidualMeters + settings.CorrectionMaxResidualRegressionMeters)
            return false;
        if (candidate.NormalCoherence + settings.CorrectionMaxCoherenceRegression <
            published.NormalCoherence)
            return false;
        return true;
    }

    private static void UpdateCorrectionEvidence(
        Track track,
        PatchGeometry candidate,
        bool acceptable,
        Settings settings)
    {
        if (!acceptable)
        {
            track.ReplacementHits = 0;
            track.CorrectionCandidate = null;
            return;
        }
        bool consistent = track.CorrectionCandidate == null ||
                          GeometryCompatible(track.CorrectionCandidate, candidate, settings);
        track.ReplacementHits = consistent ? track.ReplacementHits + 1 : 1;
        track.CorrectionCandidate = CloneGeometry(candidate);
    }

    private static void StartCorrectionTrial(
        Track track,
        PatchGeometry candidate,
        Settings settings,
        ref Result result)
    {
        track.RollbackPublished = CloneGeometry(track.Published);
        track.Published = CloneGeometry(candidate);
        track.CorrectionCandidate = CloneGeometry(candidate);
        track.ReplacementHits = 0;
        track.CorrectionTrialRemaining = settings.CorrectionTrialRebuilds;
        track.Lifecycle = PatchLifecycle.CorrectionPending;
        result.CorrectionUpgradedThisRebuild++;
    }

    private static void RollbackCorrection(Track track, ref Result result)
    {
        if (track.RollbackPublished != null)
            track.Published = track.RollbackPublished;
        track.RollbackPublished = null;
        track.CorrectionCandidate = null;
        track.ReplacementHits = 0;
        track.CorrectionTrialRemaining = 0;
        track.Lifecycle = PatchLifecycle.Mature;
        result.CorrectionRollbackThisRebuild++;
        result.AtomicRollbackCount++;
    }

    private void BuildPublishedMesh(
        out List<Vector3> vertices,
        out List<int> triangles,
        out List<Color> colors,
        ref Result result)
    {
        vertices = new List<Vector3>(4096);
        triangles = new List<int>(12288);
        colors = new List<Color>(4096);
        foreach (KeyValuePair<Vector3Int, Track> pair in _tracks)
        {
            PatchGeometry patch = pair.Value.Published;
            if (patch == null)
                continue;
            int baseVertex = vertices.Count;
            vertices.AddRange(patch.Vertices);
            colors.AddRange(patch.Colors);
            for (int i = 0; i < patch.Triangles.Count; i++)
                triangles.Add(baseVertex + patch.Triangles[i]);
            result.PublishedTriangles += patch.TriangleCount;
            result.PublishedBoundaryEdges += patch.BoundaryEdges;
            result.PublishedNonManifoldEdges += patch.NonManifoldEdges;
        }
    }

    private void CountLifecycle(ref Result result)
    {
        foreach (KeyValuePair<Vector3Int, Track> pair in _tracks)
        {
            Track track = pair.Value;
            if (track.Published == null)
            {
                result.ProvisionalPatches++;
                continue;
            }
            if (track.Lifecycle == PatchLifecycle.CorrectionPending)
                result.CorrectionPendingPatches++;
            result.MaturePatches++;
            switch (track.Published.SurfaceKind)
            {
                case PatchSurfaceKind.Planar:
                    result.MaturePlanarPatches++;
                    break;
                case PatchSurfaceKind.Curved:
                    result.MatureCurvedPatches++;
                    break;
                default:
                    result.MatureUncertainPatches++;
                    break;
            }
        }
    }

    private static PatchGeometry CloneGeometry(PatchGeometry source)
    {
        PatchGeometry clone = new PatchGeometry
        {
            Centroid = source.Centroid,
            AverageNormal = source.AverageNormal,
            BoundsMin = source.BoundsMin,
            BoundsMax = source.BoundsMax,
            PlaneResidualMeters = source.PlaneResidualMeters,
            NormalCoherence = source.NormalCoherence,
            BoundaryEdges = source.BoundaryEdges,
            NonManifoldEdges = source.NonManifoldEdges,
            SurfaceKind = source.SurfaceKind,
            PlaneConstraintId = source.PlaneConstraintId,
            PlaneConstraintApplied = source.PlaneConstraintApplied,
            PlaneConstraintRejectReason = source.PlaneConstraintRejectReason,
            PlaneConstraintNormalDot = source.PlaneConstraintNormalDot,
            PlaneConstraintCentroidDistanceMeters = source.PlaneConstraintCentroidDistanceMeters,
            PlaneConstraintNearestNormalPlaneId = source.PlaneConstraintNearestNormalPlaneId,
            PlaneConstraintBestNormalDot = source.PlaneConstraintBestNormalDot,
            PlaneConstraintNearestNormalDistanceMeters = source.PlaneConstraintNearestNormalDistanceMeters,
            PlaneConstraintDistanceExcessMeters = source.PlaneConstraintDistanceExcessMeters,
            PlaneConstraintMaxMoveMeters = source.PlaneConstraintMaxMoveMeters,
            PlaneResidualBeforeConstraintMeters = source.PlaneResidualBeforeConstraintMeters,
            PlaneConstraintQuarantinedTriangles = source.PlaneConstraintQuarantinedTriangles,
            PlaneConstraintOriginalTriangles = source.PlaneConstraintOriginalTriangles,
            PlaneConstraintRetainedIslands = source.PlaneConstraintRetainedIslands,
            NormalDecompositionKind = source.NormalDecompositionKind,
            NormalDecompositionDominantPlaneId = source.NormalDecompositionDominantPlaneId,
            NormalDecompositionPlaneFamilies = source.NormalDecompositionPlaneFamilies,
            NormalDecompositionIslands = source.NormalDecompositionIslands,
            NormalDecompositionMatchedTriangles = source.NormalDecompositionMatchedTriangles,
            NormalDecompositionUnresolvedTriangles = source.NormalDecompositionUnresolvedTriangles,
            NormalDecompositionLargestIslandTriangles = source.NormalDecompositionLargestIslandTriangles,
            NormalDecompositionWindingMinorityTriangles = source.NormalDecompositionWindingMinorityTriangles,
            NormalDecompositionSubtype = source.NormalDecompositionSubtype,
            NormalDecompositionPlaneIds = source.NormalDecompositionPlaneIds,
            NormalDecompositionMinFamilyNormalDot = source.NormalDecompositionMinFamilyNormalDot,
            NormalDecompositionMaxPlaneSeparationMeters = source.NormalDecompositionMaxPlaneSeparationMeters,
            NormalDecompositionUnmatchedNormalTriangles = source.NormalDecompositionUnmatchedNormalTriangles,
            NormalDecompositionUnmatchedDistanceTriangles = source.NormalDecompositionUnmatchedDistanceTriangles,
            NormalDecompositionUnmatchedExtentTriangles = source.NormalDecompositionUnmatchedExtentTriangles,
            NormalDecompositionUnmatchedMoveTriangles = source.NormalDecompositionUnmatchedMoveTriangles,
            NormalDecompositionUnmatchedAmbiguityTriangles = source.NormalDecompositionUnmatchedAmbiguityTriangles,
            CreaseAtomicStatus = source.CreaseAtomicStatus,
            CreaseAtomicRetainedTriangles = source.CreaseAtomicRetainedTriangles,
            CreaseAtomicQuarantinedTriangles = source.CreaseAtomicQuarantinedTriangles,
            CreaseAtomicTopologyRejectedTriangles = source.CreaseAtomicTopologyRejectedTriangles,
            CreaseAtomicRetainedFamilies = source.CreaseAtomicRetainedFamilies,
            CreaseAtomicRequiredFamilies = source.CreaseAtomicRequiredFamilies,
            CreaseAtomicRetainedIslands = source.CreaseAtomicRetainedIslands,
            CreaseAtomicRetainedRatio = source.CreaseAtomicRetainedRatio,
            CreaseAtomicMaxMoveMeters = source.CreaseAtomicMaxMoveMeters,
            CreaseAtomicTinyIslandTriangles = source.CreaseAtomicTinyIslandTriangles,
            CreaseAtomicTinyIslands = source.CreaseAtomicTinyIslands,
            CreaseAtomicFamilyIds = source.CreaseAtomicFamilyIds,
            CreaseTransitionKind = source.CreaseTransitionKind,
            CreaseTransitionFromStatus = source.CreaseTransitionFromStatus,
            CreaseTransitionFromFamilyIds = source.CreaseTransitionFromFamilyIds,
            CreaseTransitionNearestInPriorFamily = source.CreaseTransitionNearestInPriorFamily,
            CreaseTransitionTriangleRatio = source.CreaseTransitionTriangleRatio,
            CreaseTransitionCentroidShiftMeters = source.CreaseTransitionCentroidShiftMeters,
            CreaseTransitionNormalDot = source.CreaseTransitionNormalDot,
            CreaseTransitionRetainedRatioDelta = source.CreaseTransitionRetainedRatioDelta,
            CreaseTransitionContentCompatible = source.CreaseTransitionContentCompatible,
            CreaseTransitionFamilyMinNormalDot = source.CreaseTransitionFamilyMinNormalDot,
            CreaseTransitionFamilyMaxDistanceDeltaMeters = source.CreaseTransitionFamilyMaxDistanceDeltaMeters,
            CreaseTransitionFamilyMinTriangleRatio = source.CreaseTransitionFamilyMinTriangleRatio,
            CreaseTransitionFamilyRemappedCount = source.CreaseTransitionFamilyRemappedCount,
            CreaseRegionalMemberPatches = source.CreaseRegionalMemberPatches,
            CreaseRegionalRetainedRatio = source.CreaseRegionalRetainedRatio,
            CreaseRegionalStatus = source.CreaseRegionalStatus,
            CreaseRegionalRetainedFamilies = source.CreaseRegionalRetainedFamilies,
            CreaseRegionalRequiredFamilies = source.CreaseRegionalRequiredFamilies,
            CreaseRegionalMigrated = source.CreaseRegionalMigrated,
            CreaseRegionalSourceKey = source.CreaseRegionalSourceKey,
            NormalDecompositionNearNormalTriangles = source.NormalDecompositionNearNormalTriangles,
            NormalDecompositionNovelNormalTriangles = source.NormalDecompositionNovelNormalTriangles,
            NormalDecompositionNearDistanceTriangles = source.NormalDecompositionNearDistanceTriangles,
            NormalDecompositionFarDistanceTriangles = source.NormalDecompositionFarDistanceTriangles,
            NormalDecompositionNearMoveTriangles = source.NormalDecompositionNearMoveTriangles,
            NormalDecompositionFarMoveTriangles = source.NormalDecompositionFarMoveTriangles
        };
        clone.Vertices.AddRange(source.Vertices);
        clone.Triangles.AddRange(source.Triangles);
        clone.Colors.AddRange(source.Colors);
        for (int i = 0; i < source.CreaseAtomicFamilies.Count; i++)
        {
            CreaseFamilyEvidence family = source.CreaseAtomicFamilies[i];
            clone.CreaseAtomicFamilies.Add(new CreaseFamilyEvidence
            {
                PlaneId = family.PlaneId,
                Normal = family.Normal,
                Offset = family.Offset,
                RetainedTriangles = family.RetainedTriangles
            });
        }
        return clone;
    }

    private static void CountEdges(
        List<int> triangles,
        bool doubleSidedTriangles,
        out int boundary,
        out int nonManifold)
    {
        Dictionary<ulong, int> edgeUse = new Dictionary<ulong, int>(triangles.Count);
        for (int i = 0; i + 2 < triangles.Count; i += 3)
        {
            CountEdge(edgeUse, triangles[i], triangles[i + 1]);
            CountEdge(edgeUse, triangles[i + 1], triangles[i + 2]);
            CountEdge(edgeUse, triangles[i + 2], triangles[i]);
        }
        boundary = 0;
        nonManifold = 0;
        // The production mesh can contain both windings of every logical triangle.
        // In that mode a logical boundary edge has two index uses and a logical
        // manifold interior edge has four.  Applying the single-sided 1/2 limits
        // here incorrectly quarantines every ordinary quad diagonal as non-manifold.
        int boundaryUseLimit = doubleSidedTriangles ? 2 : 1;
        int manifoldUseLimit = doubleSidedTriangles ? 4 : 2;
        foreach (KeyValuePair<ulong, int> pair in edgeUse)
        {
            if (pair.Value <= boundaryUseLimit)
                boundary++;
            else if (pair.Value > manifoldUseLimit)
                nonManifold++;
        }
    }

    private static void CountEdge(Dictionary<ulong, int> edgeUse, int a, int b)
    {
        uint min = (uint)Mathf.Min(a, b);
        uint max = (uint)Mathf.Max(a, b);
        ulong key = ((ulong)min << 32) | max;
        edgeUse.TryGetValue(key, out int count);
        edgeUse[key] = count + 1;
    }

    private static Vector3Int PatchKey(Vector3 point, Vector3 origin, float patchSize)
    {
        Vector3 local = (point - origin) / Mathf.Max(0.001f, patchSize);
        return new Vector3Int(
            Mathf.FloorToInt(local.x),
            Mathf.FloorToInt(local.y),
            Mathf.FloorToInt(local.z));
    }

    private static bool ValidTriangle(int a, int b, int c, int vertexCount)
    {
        return a >= 0 && b >= 0 && c >= 0 &&
               a < vertexCount && b < vertexCount && c < vertexCount &&
               a != b && b != c && c != a;
    }

    private static bool Finite(Vector3 value)
    {
        return !float.IsNaN(value.x) && !float.IsInfinity(value.x) &&
               !float.IsNaN(value.y) && !float.IsInfinity(value.y) &&
               !float.IsNaN(value.z) && !float.IsInfinity(value.z);
    }

    private static void Sanitize(ref Settings settings)
    {
        settings.LocalExtractionPaddingPatches = Mathf.Clamp(settings.LocalExtractionPaddingPatches, 0, 3);
        settings.CorrectionConfirmRebuilds = Mathf.Clamp(settings.CorrectionConfirmRebuilds, 2, 8);
        settings.CorrectionTrialRebuilds = Mathf.Clamp(settings.CorrectionTrialRebuilds, 1, 4);
        settings.CorrectionMaxResidualRegressionMeters = Mathf.Clamp(
            settings.CorrectionMaxResidualRegressionMeters, 0f, 0.02f);
        settings.CorrectionMaxCoherenceRegression = Mathf.Clamp(
            settings.CorrectionMaxCoherenceRegression, 0f, 0.2f);
        settings.PatchSizeMeters = Mathf.Clamp(settings.PatchSizeMeters, 0.15f, 2f);
        settings.ConfirmRebuilds = Mathf.Clamp(settings.ConfirmRebuilds, 2, 8);
        settings.ReplacementConfirmRebuilds = Mathf.Clamp(settings.ReplacementConfirmRebuilds, 2, 8);
        settings.RetireMissingRebuilds = Mathf.Clamp(settings.RetireMissingRebuilds, 2, 16);
        settings.MaxPatches = Mathf.Clamp(settings.MaxPatches, 16, 2048);
        settings.MaxTrianglesPerPatch = Mathf.Clamp(settings.MaxTrianglesPerPatch, 32, 8192);
        settings.StableCentroidDistanceMeters = Mathf.Clamp(settings.StableCentroidDistanceMeters, 0.005f, 0.25f);
        settings.StableNormalDot = Mathf.Clamp(settings.StableNormalDot, 0.5f, 1f);
        settings.StableTriangleRatio = Mathf.Clamp(settings.StableTriangleRatio, 0.25f, 1f);
        settings.PlanarResidualMeters = Mathf.Clamp(settings.PlanarResidualMeters, 0.005f, 0.2f);
        settings.PlanarNormalCoherence = Mathf.Clamp(settings.PlanarNormalCoherence, 0.5f, 1f);
        settings.ReplacementResidualSlackMeters = Mathf.Clamp(settings.ReplacementResidualSlackMeters, 0f, 0.1f);
        settings.ReplacementBoundarySlackRatio = Mathf.Clamp(settings.ReplacementBoundarySlackRatio, 1f, 2f);
        settings.PlanePatchMinResidualMeters = Mathf.Clamp(settings.PlanePatchMinResidualMeters, 0.005f, 0.1f);
        settings.PlanePatchMinNormalDot = Mathf.Clamp(settings.PlanePatchMinNormalDot, 0.5f, 1f);
        settings.PlanePatchMaxCentroidDistanceMeters = Mathf.Clamp(settings.PlanePatchMaxCentroidDistanceMeters, 0.01f, 0.2f);
        settings.PlanePatchTangentialPaddingMeters = Mathf.Clamp(settings.PlanePatchTangentialPaddingMeters, 0f, 1f);
        settings.PlanePatchMaxVertexMoveMeters = Mathf.Clamp(settings.PlanePatchMaxVertexMoveMeters, 0.005f, 0.2f);
        settings.PlanePatchMinTriangleAreaRatio = Mathf.Clamp(settings.PlanePatchMinTriangleAreaRatio, 0.05f, 0.95f);
        settings.PlanePatchMinTriangleNormalDot = Mathf.Clamp(settings.PlanePatchMinTriangleNormalDot, -0.5f, 0.999f);
        settings.PlanePatchAmbiguityScoreMargin = Mathf.Clamp(settings.PlanePatchAmbiguityScoreMargin, 0f, 0.5f);
        settings.PlaneRejectMaxHoldRebuilds = Mathf.Clamp(settings.PlaneRejectMaxHoldRebuilds, 1, 8);
        settings.PlaneRejectWithdrawResidualMeters = Mathf.Clamp(
            settings.PlaneRejectWithdrawResidualMeters, 0.02f, 0.1f);
    }
}

/// <summary>
/// Builds a read-only, single-sided logical triangle view from a render index buffer.
/// Reverse-winding render copies are collapsed without changing the production mesh.
/// </summary>
public static class ScanCoverLogicalTriangleView
{
    public struct Result
    {
        public List<int> Triangles;
        public int RenderTriangles;
        public int ValidTriangles;
        public int LogicalTriangles;
        public int ReversePairsRemoved;
        public int SameWindingDuplicatesRemoved;
        public int UnpairedLogicalTriangles;
        public int WindingConflictKeys;
        public int InvalidTriangles;
    }

    private struct TriangleKey : IEquatable<TriangleKey>
    {
        public int A;
        public int B;
        public int C;

        public TriangleKey(int a, int b, int c)
        {
            if (a > b) Swap(ref a, ref b);
            if (b > c) Swap(ref b, ref c);
            if (a > b) Swap(ref a, ref b);
            A = a;
            B = b;
            C = c;
        }

        public bool Equals(TriangleKey other) => A == other.A && B == other.B && C == other.C;
        public override bool Equals(object obj) => obj is TriangleKey && Equals((TriangleKey)obj);
        public override int GetHashCode()
        {
            unchecked
            {
                int hash = A;
                hash = hash * 397 ^ B;
                return hash * 397 ^ C;
            }
        }

        private static void Swap(ref int a, ref int b)
        {
            int value = a;
            a = b;
            b = value;
        }
    }

    private sealed class LogicalTriangle
    {
        public int A;
        public int B;
        public int C;
        public int ForwardCount;
        public int ReverseCount;
    }

    public static Result Build(List<Vector3> vertices, List<int> renderTriangles)
    {
        Result result = new Result
        {
            Triangles = new List<int>(renderTriangles != null ? renderTriangles.Count / 2 : 0),
            RenderTriangles = renderTriangles != null ? renderTriangles.Count / 3 : 0
        };
        if (vertices == null || renderTriangles == null)
            return result;

        Dictionary<TriangleKey, LogicalTriangle> logical =
            new Dictionary<TriangleKey, LogicalTriangle>(Mathf.Max(16, result.RenderTriangles / 2));
        for (int i = 0; i + 2 < renderTriangles.Count; i += 3)
        {
            int a = renderTriangles[i];
            int b = renderTriangles[i + 1];
            int c = renderTriangles[i + 2];
            if (!ValidTriangle(vertices, a, b, c))
            {
                result.InvalidTriangles++;
                continue;
            }
            result.ValidTriangles++;
            TriangleKey key = new TriangleKey(a, b, c);
            if (!logical.TryGetValue(key, out LogicalTriangle entry))
            {
                entry = new LogicalTriangle { A = a, B = b, C = c, ForwardCount = 1 };
                logical.Add(key, entry);
                continue;
            }
            if (SameOrientation(entry.A, entry.B, entry.C, a, b, c))
                entry.ForwardCount++;
            else
                entry.ReverseCount++;
        }

        foreach (LogicalTriangle entry in logical.Values)
        {
            result.Triangles.Add(entry.A);
            result.Triangles.Add(entry.B);
            result.Triangles.Add(entry.C);
            result.LogicalTriangles++;
            int reversePairs = Mathf.Min(entry.ForwardCount, entry.ReverseCount);
            result.ReversePairsRemoved += reversePairs;
            result.SameWindingDuplicatesRemoved += Mathf.Max(
                0, entry.ForwardCount + entry.ReverseCount - 1 - reversePairs);
            if (entry.ReverseCount == 0)
                result.UnpairedLogicalTriangles++;
            if (entry.ForwardCount > 1 || entry.ReverseCount > 1 ||
                Mathf.Abs(entry.ForwardCount - entry.ReverseCount) > 1)
                result.WindingConflictKeys++;
        }
        return result;
    }

    private static bool SameOrientation(int a, int b, int c, int x, int y, int z)
    {
        return (a == x && b == y && c == z) ||
               (a == y && b == z && c == x) ||
               (a == z && b == x && c == y);
    }

    private static bool ValidTriangle(List<Vector3> vertices, int a, int b, int c)
    {
        if (a < 0 || b < 0 || c < 0 || a >= vertices.Count || b >= vertices.Count || c >= vertices.Count ||
            a == b || b == c || c == a)
            return false;
        Vector3 cross = Vector3.Cross(vertices[b] - vertices[a], vertices[c] - vertices[a]);
        return cross.sqrMagnitude > 0.000000000001f;
    }
}
