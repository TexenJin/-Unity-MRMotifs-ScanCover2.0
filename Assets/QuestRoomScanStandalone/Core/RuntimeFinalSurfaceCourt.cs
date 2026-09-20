using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Unity.Mathematics;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Runtime counterpart of ScanCoverOfflineFinalCourtReplay.py. GunGel's
    /// stable id is an address only. Each time-separated observation contributes
    /// one median depth testimony; pose diversity strengthens the evidence but is
    /// not required, because a stationary user must still be able to establish a
    /// temporally stable surface. Interleaved proposal/selection/audit roles
    /// prevent one time slice from both inventing and approving its own layer.
    /// The court never constructs a mesh or writes TSDF storage. It publishes
    /// a stable-id verdict plus the independently selected surface plane. The
    /// production TSDF deliberately does not consume that plane: approved
    /// planes are product constraints for already-extracted candidate chunks.
    /// This keeps reconstruction evidence and delivered-mesh policy separate.
    /// </summary>
    internal sealed class RuntimeFinalSurfaceCourt : IDisposable
    {
        private const uint ObservationValid = 1u << 0;
        private const uint RawAvailable = 1u << 1;
        private const uint DualAgree = 1u << 2;
        private const uint StableFound = 1u << 6;
        private const uint NormalPass = 1u << 8;

        private const float SafeMotionQuality = 0.70f;
        private const float CandidateBandMetres = 0.150f;
        private const float MinimumIncidence = 0.30f;
        private const float ModeRadiusMetres = 0.020f;
        private const float MinimumProposalModeShare = 0.15f;
        private const float MaximumSelectionResidualMetres = 0.015f;
        private const float MinimumSelectionModeShare = 0.60f;
        private const float MinimumWinnerMarginMetres = 0.010f;
        private const int MinimumRoleViews = 5;
        private const int MaximumHypotheses = 2;
        private const int MaximumIndependentViews = 48;
        private const int ReplacementConfirmations = 3;
        private const int RevocationConfirmations = 3;
        private const int ReplacementCooldownFrames = 30;
        private const float InvalidationRadiusMetres = 0.125f;

        private enum CourtVerdict
        {
            Pending,
            Approved,
            Abstain
        }

        private struct Testimony
        {
            internal Vector3 Camera;
            internal Vector3 Direction;
            internal Vector3 Source;
            internal Vector3 Target;
            internal Vector3 Normal;
            internal float PlaneCoordinate;
            internal int Frame;
            internal int SourceFrame;
            internal int AttemptIndex;
        }

        private sealed class FrameBin
        {
            internal readonly List<Vector3> Sources = new List<Vector3>(8);
            internal readonly List<Vector3> Targets = new List<Vector3>(8);
            internal readonly List<Vector3> Normals = new List<Vector3>(8);
        }

        private sealed class AddressState
        {
            internal readonly uint StableId;
            internal readonly List<Testimony> Views = new List<Testimony>(24);
            internal CourtVerdict Verdict;
            internal Vector3 PublishedCenter;
            internal Vector3 PublishedNormal;
            internal float PublishedLayerCoordinate;
            internal float PublishedPlaneCoordinate;
            internal Vector3 ChallengeCenter;
            internal float ChallengePlaneCoordinate;
            internal int ChallengeConfirmations;
            internal int FailureConfirmations;
            internal Vector3 ReferenceNormal;
            internal bool ReferenceInitialized;
            internal uint Generation;
            internal int LastPublishFrame = int.MinValue / 2;

            internal AddressState(uint stableId)
            {
                StableId = stableId;
            }
        }

        private readonly struct Hypothesis
        {
            internal readonly float Center;
            internal readonly int Support;

            internal Hypothesis(float center, int support)
            {
                Center = center;
                Support = support;
            }
        }

        internal readonly struct ProductPlane
        {
            internal readonly Vector4 Equation;
            internal readonly Vector4 CenterRadius;
            internal readonly uint Generation;

            internal ProductPlane(Vector4 equation, Vector4 centerRadius,
                uint generation)
            {
                Equation = equation;
                CenterRadius = centerRadius;
                Generation = generation;
            }
        }

        private readonly object _stateLock = new object();
        private readonly Dictionary<uint, AddressState> _addresses =
            new Dictionary<uint, AddressState>(4096);
        private readonly Dictionary<uint, FrameBin> _frameBins =
            new Dictionary<uint, FrameBin>(1024);
        private readonly List<float> _scratch = new List<float>(64);
        private readonly Queue<Vector4> _pendingInvalidations =
            new Queue<Vector4>(64);
        // CopyProductPlanes runs once per rebuilt mesh block.  Reallocating four
        // clustering arrays for every block creates avoidable Quest GC pressure
        // exactly while the GPU readback queue is busiest.  The method is
        // serialized by _stateLock, so one reusable scratch set is sufficient.
        private Vector3[] _productNormalSums = Array.Empty<Vector3>();
        private Vector3[] _productCenterSums = Array.Empty<Vector3>();
        private int[] _productMembers = Array.Empty<int>();
        private uint[] _productGenerations = Array.Empty<uint>();
        private string _eventsPath;
        private string _testimoniesPath;
        private string _frameGatesPath;
        private string _decisionChecksPath;
        private readonly List<string> _pendingEvents = new List<string>(128);
        private readonly List<string> _pendingTestimonies = new List<string>(256);
        private readonly List<string> _pendingFrameGates = new List<string>(64);
        private readonly List<string> _pendingDecisionChecks = new List<string>(256);
        private int _revision;
        private int _writeErrors;
        private bool _active;

        internal int Revision
        {
            get { lock (_stateLock) return _revision; }
        }

        internal int WriteErrors
        {
            get { lock (_stateLock) return _writeErrors; }
        }

        internal void Begin(string directory)
        {
            lock (_stateLock)
            {
                ResetLocked();
                if (string.IsNullOrEmpty(directory)) return;
                string courtDirectory = Path.Combine(directory,
                    "final_surface_court");
                Directory.CreateDirectory(courtDirectory);
                _eventsPath = Path.Combine(courtDirectory, "runtime_events.csv");
                File.WriteAllText(_eventsPath,
                    "gunGelFrame,stableId,independentViews,proposalViews,selectionViews,auditViews,oldVerdict,newVerdict,generation,publishedX,publishedY,publishedZ,publishedNormalX,publishedNormalY,publishedNormalZ,publishedLayerCoordinateM,publishedPlaneCoordinateM,reason\n",
                    new UTF8Encoding(false));
                _testimoniesPath = Path.Combine(courtDirectory,
                    "independent_testimonies.csv");
                File.WriteAllText(_testimoniesPath,
                    "gunGelFrame,sourceFrame,attemptIndex,stableId,archiveIndex,role,cameraX,cameraY,cameraZ,sourceX,sourceY,sourceZ,targetX,targetY,targetZ,normalX,normalY,normalZ,referenceNormalX,referenceNormalY,referenceNormalZ,rangeM,incidenceAbs,targetNormalResidualMm,planeCoordinateM,angularDegPerSec,linearMps,motionQuality,headPitchDeg,headYawDeg,headRollDeg\n",
                    new UTF8Encoding(false));
                _frameGatesPath = Path.Combine(courtDirectory,
                    "frame_gate_summary.csv");
                File.WriteAllText(_frameGatesPath,
                    "gunGelFrame,sourceFrame,attemptIndex,totalRows,motionRejected,stableIdMissing,evidenceRejected,observationInvalid,rawMissing,dualDisagree,stableMissing,normalRejected,nonFinite,incidenceRejected,candidateBandRejected,eligibleRows,stableBins,referenceNormalRejected,temporalDuplicateRejected,independentAccepted,angularDegPerSec,linearMps,motionQuality,headPitchDeg,headYawDeg,headRollDeg\n",
                    new UTF8Encoding(false));
                _decisionChecksPath = Path.Combine(courtDirectory,
                    "decision_checks.csv");
                File.WriteAllText(_decisionChecksPath,
                    "gunGelFrame,stableId,generation,currentVerdict,viewCount,proposalViews,selectionViews,auditViews,hypothesisCount,hypothesisCentersMm,hypothesisSupports,selectionCenterMm,selectionMembers,selectionShare,auditCenterMm,auditMembers,auditShare,winnerCenterMm,winnerResidualMm,runnerResidualMm,winnerMarginMm,challengeConfirmations,failureConfirmations,framesSincePublish,outcome\n",
                    new UTF8Encoding(false));
                File.WriteAllText(Path.Combine(courtDirectory, "schema.json"),
                    BuildSchema(), new UTF8Encoding(false));
                _active = true;
            }
        }

        /// <summary>
        /// Starts the production court without opening a diagnostic session or
        /// writing files.  The verdict state is authoritative in memory; disk
        /// ledgers remain an optional observer owned by ScanReplaySessionPackage.
        /// </summary>
        internal void BeginInMemory()
        {
            lock (_stateLock)
            {
                ResetLocked();
                _active = true;
            }
        }

        internal void Record(int gunGelFrame, int sourceFrame, int attemptIndex,
            GunGelEvidenceShadow.Correspondence[] correspondences,
            uint4[] identities, Matrix4x4 sourceViewInverse,
            Matrix4x4 fusionCorrection, float angularSpeed, float linearSpeed,
            float motionQuality, Vector3 headEuler)
        {
            lock (_stateLock)
            {
                if (!_active) return;
                try
                {
                    int totalRows = correspondences == null || identities == null
                        ? 0 : Mathf.Min(correspondences.Length, identities.Length);
                    if (totalRows == 0 || motionQuality < SafeMotionQuality)
                    {
                        AppendFrameGate(gunGelFrame, sourceFrame, attemptIndex,
                            totalRows,
                            motionRejected: motionQuality < SafeMotionQuality ? totalRows : 0,
                            stableIdMissing: 0, evidenceRejected: 0,
                            observationInvalid: 0, rawMissing: 0,
                            dualDisagree: 0, stableMissing: 0,
                            normalRejected: 0, nonFinite: 0,
                            incidenceRejected: 0, candidateBandRejected: 0,
                            eligibleRows: 0, stableBins: 0,
                            referenceNormalRejected: 0,
                            temporalDuplicateRejected: 0, independentAccepted: 0,
                            angularSpeed: angularSpeed, linearSpeed: linearSpeed,
                            motionQuality: motionQuality, headEuler: headEuler);
                        FlushEventsLocked();
                        return;
                    }
                    Vector3 camera = sourceViewInverse.MultiplyPoint3x4(Vector3.zero);
                    _frameBins.Clear();
                    int stableIdMissing = 0, evidenceRejected = 0;
                    int observationInvalid = 0, rawMissing = 0, dualDisagree = 0;
                    int stableMissing = 0, normalRejected = 0, nonFinite = 0;
                    int incidenceRejected = 0, candidateBandRejected = 0;
                    int eligibleRows = 0;
                    int count = totalRows;
                    for (int i = 0; i < count; i++)
                    {
                        GunGelEvidenceShadow.Correspondence item = correspondences[i];
                        uint flags = (uint)Mathf.Max(0,
                            Mathf.RoundToInt(item.SourceValid.w));
                        uint stableId = identities[i].y;
                        if (stableId == 0u)
                        {
                            stableIdMissing++;
                            continue;
                        }
                        bool evidencePass =
                            (flags & (ObservationValid | RawAvailable | DualAgree |
                                      StableFound | NormalPass)) ==
                            (ObservationValid | RawAvailable | DualAgree |
                             StableFound | NormalPass);
                        if (!evidencePass)
                        {
                            evidenceRejected++;
                            if ((flags & ObservationValid) == 0u) observationInvalid++;
                            if ((flags & RawAvailable) == 0u) rawMissing++;
                            if ((flags & DualAgree) == 0u) dualDisagree++;
                            if ((flags & StableFound) == 0u) stableMissing++;
                            if ((flags & NormalPass) == 0u) normalRejected++;
                            continue;
                        }

                        Vector3 source = fusionCorrection.MultiplyPoint3x4(
                            new Vector3(item.SourceValid.x,
                                item.SourceValid.y, item.SourceValid.z));
                        Vector3 target = new Vector3(item.TargetSigma.x,
                            item.TargetSigma.y, item.TargetSigma.z);
                        Vector3 normal = new Vector3(item.NormalAngle.x,
                            item.NormalAngle.y, item.NormalAngle.z);
                        if (!Finite(source) || !Finite(target) || !Finite(normal) ||
                            normal.sqrMagnitude < 1e-8f)
                        {
                            nonFinite++;
                            continue;
                        }
                        normal.Normalize();
                        Vector3 ray = source - camera;
                        float range = ray.magnitude;
                        if (range < 1e-4f)
                        {
                            nonFinite++;
                            continue;
                        }
                        ray /= range;
                        if (Mathf.Abs(Vector3.Dot(ray, normal)) < MinimumIncidence)
                        {
                            incidenceRejected++;
                            continue;
                        }
                        if (Mathf.Abs(Vector3.Dot(source - target, normal)) >
                            CandidateBandMetres)
                        {
                            candidateBandRejected++;
                            continue;
                        }
                        eligibleRows++;

                        if (!_frameBins.TryGetValue(stableId, out FrameBin bin))
                        {
                            bin = new FrameBin();
                            _frameBins.Add(stableId, bin);
                        }
                        bin.Sources.Add(source);
                        bin.Targets.Add(target);
                        bin.Normals.Add(normal);
                    }

                    int temporalDuplicateRejected = 0;
                    int independentAccepted = 0;
                    int referenceNormalRejected = 0;
                    foreach (KeyValuePair<uint, FrameBin> pair in _frameBins)
                    {
                        FrameBin bin = pair.Value;
                        if (bin.Sources.Count == 0) continue;
                        Vector3 source = MedianVector(bin.Sources);
                        Vector3 target = MedianVector(bin.Targets);
                        Vector3 normal = AlignedMean(bin.Normals);
                        Vector3 direction = SafeDirection(target - camera);
                        if (!_addresses.TryGetValue(pair.Key,
                                out AddressState state))
                        {
                            state = new AddressState(pair.Key);
                            _addresses.Add(pair.Key, state);
                        }
                        // Plane coordinates from different per-frame normals are
                        // not comparable: a small normal wobble changes dot(p,n)
                        // with the world origin and can manufacture a false layer.
                        // StableId supplies only the address; the first finite
                        // normal fixes a coordinate axis for every later testimony.
                        if (!state.ReferenceInitialized)
                        {
                            state.ReferenceNormal = normal;
                            state.ReferenceInitialized = true;
                        }
                        else
                        {
                            if (Vector3.Dot(normal, state.ReferenceNormal) < 0f)
                                normal = -normal;
                            if (Vector3.Dot(normal, state.ReferenceNormal) < 0.5f)
                            {
                                referenceNormalRejected++;
                                continue;
                            }
                        }
                        float coordinate = Vector3.Dot(source,
                            state.ReferenceNormal);
                        if (!TryAddIndependent(state, camera, direction, source,
                                target, normal, coordinate, gunGelFrame,
                                sourceFrame, attemptIndex))
                        {
                            temporalDuplicateRejected++;
                            continue;
                        }
                        independentAccepted++;
                        int archiveIndex = state.Views.Count - 1;
                        AppendTestimony(gunGelFrame, sourceFrame, attemptIndex,
                            state, archiveIndex, camera, source, target, normal,
                            range: Vector3.Distance(source, camera),
                            incidence: Mathf.Abs(Vector3.Dot(
                                SafeDirection(source - camera), normal)),
                            targetResidualMm: Vector3.Dot(source - target, normal) * 1000f,
                            coordinate: coordinate, angularSpeed: angularSpeed,
                            linearSpeed: linearSpeed, motionQuality: motionQuality,
                            headEuler: headEuler);
                        Rejudge(state, gunGelFrame);
                    }
                    AppendFrameGate(gunGelFrame, sourceFrame, attemptIndex,
                        totalRows, 0, stableIdMissing, evidenceRejected,
                        observationInvalid, rawMissing, dualDisagree, stableMissing,
                        normalRejected, nonFinite, incidenceRejected,
                        candidateBandRejected, eligibleRows, _frameBins.Count,
                        referenceNormalRejected,
                        temporalDuplicateRejected, independentAccepted,
                        angularSpeed, linearSpeed, motionQuality, headEuler);
                    FlushEventsLocked();
                }
                catch
                {
                    _writeErrors++;
                }
            }
        }

        internal VirtualProbeShadowAdjudicator.VerdictCounts GetCounts()
        {
            lock (_stateLock)
            {
                int pending = 0, approved = 0, abstain = 0;
                foreach (AddressState state in _addresses.Values)
                {
                    if (state.Verdict == CourtVerdict.Approved) approved++;
                    else if (state.Verdict == CourtVerdict.Abstain) abstain++;
                    else pending++;
                }
                return new VirtualProbeShadowAdjudicator.VerdictCounts(
                    pending, approved, abstain);
            }
        }

        /// <summary>
        /// Copies the active address prefix used by the single production
        /// integration path. The plane has two deliberately separate jobs on
        /// the GPU: the current raw ray must first prove membership in its thin
        /// band, then this independently selected plane supplies the zero-level
        /// constraint to the one and only production TSDF.
        /// </summary>
        internal int CopyAdmissionTables(uint[] verdicts, Vector4[] planes,
            uint[] generations, int clearCount)
        {
            if (verdicts == null || planes == null || generations == null)
                return 0;
            lock (_stateLock)
            {
                int capacity = Mathf.Min(verdicts.Length,
                    Mathf.Min(planes.Length, generations.Length));
                int cleared = Mathf.Clamp(clearCount, 0, capacity);
                if (cleared > 0)
                {
                    Array.Clear(verdicts, 0, cleared);
                    Array.Clear(planes, 0, cleared);
                    Array.Clear(generations, 0, cleared);
                }
                int usedPrefix = 1;
                foreach (AddressState state in _addresses.Values)
                {
                    if (state.StableId >= (uint)capacity) continue;
                    int index = (int)state.StableId;
                    verdicts[index] = state.Verdict == CourtVerdict.Approved
                        ? 1u
                        : state.Verdict == CourtVerdict.Abstain ? 2u : 0u;
                    if (state.Verdict == CourtVerdict.Approved &&
                        state.PublishedNormal.sqrMagnitude > 1e-8f)
                    {
                        Vector3 normal = state.PublishedNormal.normalized;
                        planes[index] = new Vector4(normal.x, normal.y, normal.z,
                            state.PublishedPlaneCoordinate);
                        generations[index] = state.Generation == 0u
                            ? 1u : state.Generation;
                    }
                    usedPrefix = Mathf.Max(usedPrefix, index + 1);
                }
                return usedPrefix;
            }
        }

        /// <summary>
        /// Copies only approved planes whose finite support overlaps one output
        /// chunk.  Unlike the legacy admission table this is a mesh-product
        /// query: it is consumed after Surface Nets extraction and can never
        /// alter TSDF distance or weight.
        /// </summary>
        internal int CopyProductPlanes(Bounds worldBounds,
            ProductPlane[] destination)
        {
            if (destination == null || destination.Length == 0)
                return 0;

            lock (_stateLock)
            {
                EnsureProductScratch(destination.Length);
                Array.Clear(_productNormalSums, 0, destination.Length);
                Array.Clear(_productCenterSums, 0, destination.Length);
                Array.Clear(_productMembers, 0, destination.Length);
                Array.Clear(_productGenerations, 0, destination.Length);
                Vector3[] normalSums = _productNormalSums;
                Vector3[] centerSums = _productCenterSums;
                int[] members = _productMembers;
                uint[] generations = _productGenerations;
                int count = 0;
                foreach (AddressState state in _addresses.Values)
                {
                    if (state.Verdict != CourtVerdict.Approved ||
                        state.PublishedNormal.sqrMagnitude <= 1e-8f ||
                        state.Generation == 0u)
                        continue;

                    float radius = InvalidationRadiusMetres;
                    if (worldBounds.SqrDistance(state.PublishedCenter) >
                        radius * radius)
                        continue;

                    Vector3 normal = state.PublishedNormal.normalized;
                    int cluster = -1;
                    for (int i = 0; i < count; i++)
                    {
                        Vector3 clusterNormal = normalSums[i].normalized;
                        float alignment = Vector3.Dot(normal, clusterNormal);
                        if (Mathf.Abs(alignment) < 0.985f)
                            continue;
                        Vector3 candidateNormal = alignment < 0f
                            ? -normal : normal;
                        Vector3 clusterCenter = centerSums[i] /
                            Mathf.Max(1, members[i]);
                        float separation = Mathf.Abs(Vector3.Dot(
                            state.PublishedCenter - clusterCenter,
                            clusterNormal));
                        if (separation > ModeRadiusMetres)
                            continue;
                        normal = candidateNormal;
                        cluster = i;
                        break;
                    }

                    if (cluster < 0)
                    {
                        if (count >= destination.Length)
                            continue;
                        cluster = count++;
                    }
                    normalSums[cluster] += normal;
                    centerSums[cluster] += state.PublishedCenter;
                    members[cluster]++;
                    if (state.Generation > generations[cluster])
                        generations[cluster] = state.Generation;
                }

                // Many 10 cm GunGel addresses on one wall become one bounded
                // product constraint.  This is both cheaper than looping every
                // address per vertex and more continuous than arbitrarily taking
                // the first N dictionary entries.
                float supportRadius = worldBounds.extents.magnitude +
                                      InvalidationRadiusMetres;
                for (int i = 0; i < count; i++)
                {
                    Vector3 normal = normalSums[i].normalized;
                    Vector3 center = centerSums[i] / Mathf.Max(1, members[i]);
                    float coordinate = Vector3.Dot(center, normal);
                    destination[i] = new ProductPlane(
                        new Vector4(normal.x, normal.y, normal.z, coordinate),
                        new Vector4(center.x, center.y, center.z, supportRadius),
                        generations[i]);
                }
                return count;
            }
        }

        private void EnsureProductScratch(int capacity)
        {
            if (_productNormalSums.Length >= capacity)
                return;
            _productNormalSums = new Vector3[capacity];
            _productCenterSums = new Vector3[capacity];
            _productMembers = new int[capacity];
            _productGenerations = new uint[capacity];
        }

        /// <summary>
        /// Returns local regions whose published product constraint changed.
        /// The mesh pipeline re-productizes only intersecting chunks and keeps
        /// their old front visible until the replacement atomically commits.
        /// Neither this queue nor its consumer clears or rewrites TSDF.
        /// </summary>
        internal int DrainInvalidationRegions(Vector4[] destination)
        {
            if (destination == null || destination.Length == 0) return 0;
            lock (_stateLock)
            {
                int count = 0;
                while (count < destination.Length &&
                       _pendingInvalidations.Count > 0)
                    destination[count++] = _pendingInvalidations.Dequeue();
                return count;
            }
        }

        internal void End()
        {
            lock (_stateLock) EndLocked();
        }

        public void Dispose()
        {
            lock (_stateLock)
            {
                EndLocked();
                _addresses.Clear();
                _frameBins.Clear();
                _pendingInvalidations.Clear();
            }
        }

        private void Rejudge(AddressState state, int frame)
        {
            int count = state.Views.Count;
            if (count < MinimumRoleViews * 3)
            {
                AppendDecisionCheck(state, frame, (count + 2) / 3,
                    (count + 1) / 3, count / 3, null,
                    float.NaN, 0, float.NaN, 0, -1,
                    float.NaN, float.NaN, float.NaN,
                    "insufficient_role_views");
                return;
            }
            var proposal = new List<float>((count + 2) / 3);
            var selection = new List<float>((count + 1) / 3);
            var audit = new List<float>(count / 3);
            for (int i = 0; i < count; i++)
            {
                if (i % 3 == 0) proposal.Add(state.Views[i].PlaneCoordinate);
                else if (i % 3 == 1) selection.Add(state.Views[i].PlaneCoordinate);
                else audit.Add(state.Views[i].PlaneCoordinate);
            }
            if (proposal.Count < MinimumRoleViews ||
                selection.Count < MinimumRoleViews || audit.Count < MinimumRoleViews)
            {
                AppendDecisionCheck(state, frame, proposal.Count,
                    selection.Count, audit.Count, null,
                    float.NaN, 0, float.NaN, 0, -1,
                    float.NaN, float.NaN, float.NaN,
                    "insufficient_role_views");
                return;
            }

            List<Hypothesis> hypotheses = BuildHypotheses(proposal);
            Mode(selection, out float selectionCenter, out int selectionMembers);
            Mode(audit, out float auditCenter, out int auditMembers);
            float selectionShare = selectionMembers / (float)selection.Count;
            float auditShare = auditMembers / (float)audit.Count;
            int winnerIndex = -1;
            float winnerResidual = float.PositiveInfinity;
            float runnerResidual = float.PositiveInfinity;
            for (int i = 0; i < hypotheses.Count; i++)
            {
                float residual = Mathf.Abs(hypotheses[i].Center - selectionCenter);
                if (residual < winnerResidual)
                {
                    runnerResidual = winnerResidual;
                    winnerResidual = residual;
                    winnerIndex = i;
                }
                else if (residual < runnerResidual) runnerResidual = residual;
            }
            float margin = float.IsInfinity(runnerResidual)
                ? float.PositiveInfinity
                : runnerResidual - winnerResidual;
            bool approved = winnerIndex >= 0 &&
                            winnerResidual <= MaximumSelectionResidualMetres &&
                            selectionShare >= MinimumSelectionModeShare &&
                            auditShare >= MinimumSelectionModeShare &&
                            Mathf.Abs(hypotheses[winnerIndex].Center - auditCenter) <=
                                MaximumSelectionResidualMetres &&
                            margin >= MinimumWinnerMarginMetres;
            if (!approved)
            {
                state.ChallengeConfirmations = 0;
                if (state.Verdict == CourtVerdict.Approved)
                {
                    state.FailureConfirmations++;
                    if (state.FailureConfirmations < RevocationConfirmations)
                    {
                        AppendDecisionCheck(state, frame, proposal.Count,
                            selection.Count, audit.Count, hypotheses,
                            selectionCenter, selectionMembers, auditCenter,
                            auditMembers, winnerIndex, winnerResidual,
                            runnerResidual, margin, "revocation_confirming");
                        return;
                    }
                    QueueInvalidation(state);
                }
                AppendDecisionCheck(state, frame, proposal.Count,
                    selection.Count, audit.Count, hypotheses,
                    selectionCenter, selectionMembers, auditCenter,
                    auditMembers, winnerIndex, winnerResidual,
                    runnerResidual, margin, "selection_abstain_confirmed");
                Transition(state, CourtVerdict.Abstain, frame,
                    "selection_abstain_confirmed", proposal.Count,
                    selection.Count, audit.Count);
                state.FailureConfirmations = 0;
                state.ChallengeConfirmations = 0;
                return;
            }

            Hypothesis winner = hypotheses[winnerIndex];
            Vector3 normal = WinnerNormal(state, winner.Center);
            Vector3 center = WinnerCenter(state, winner.Center);
            center += state.ReferenceNormal *
                (winner.Center - Vector3.Dot(center,
                    state.ReferenceNormal));
            float planeCoordinate = Vector3.Dot(center, normal);
            if (state.Verdict == CourtVerdict.Approved &&
                Mathf.Abs(winner.Center - state.PublishedLayerCoordinate) <=
                MaximumSelectionResidualMetres)
            {
                // The archive does not rewrite a formal plane for ordinary
                // sub-band jitter. This is the anti-thrashing invariant.
                state.ChallengeConfirmations = 0;
                state.FailureConfirmations = 0;
                AppendDecisionCheck(state, frame, proposal.Count,
                    selection.Count, audit.Count, hypotheses,
                    selectionCenter, selectionMembers, auditCenter,
                    auditMembers, winnerIndex, winnerResidual,
                    runnerResidual, margin, "within_published_deadband");
                return;
            }

            if (state.Verdict == CourtVerdict.Approved)
            {
                if (frame - state.LastPublishFrame < ReplacementCooldownFrames)
                {
                    AppendDecisionCheck(state, frame, proposal.Count,
                        selection.Count, audit.Count, hypotheses,
                        selectionCenter, selectionMembers, auditCenter,
                        auditMembers, winnerIndex, winnerResidual,
                        runnerResidual, margin, "replacement_cooldown");
                    return;
                }
                bool sameChallenge = Mathf.Abs(winner.Center -
                    state.ChallengePlaneCoordinate) <=
                    MaximumSelectionResidualMetres;
                state.ChallengeConfirmations = sameChallenge
                    ? state.ChallengeConfirmations + 1
                    : 1;
                state.ChallengeCenter = center;
                state.ChallengePlaneCoordinate = winner.Center;
                state.FailureConfirmations = 0;
                if (state.ChallengeConfirmations < ReplacementConfirmations)
                {
                    AppendDecisionCheck(state, frame, proposal.Count,
                        selection.Count, audit.Count, hypotheses,
                        selectionCenter, selectionMembers, auditCenter,
                        auditMembers, winnerIndex, winnerResidual,
                        runnerResidual, margin, "replacement_confirming");
                    return;
                }
                QueueInvalidation(state);
                AppendDecisionCheck(state, frame, proposal.Count,
                    selection.Count, audit.Count, hypotheses,
                    selectionCenter, selectionMembers, auditCenter,
                    auditMembers, winnerIndex, winnerResidual,
                    runnerResidual, margin, "replacement_published");
                Publish(state, center, normal, winner.Center, planeCoordinate,
                    frame,
                    "replacement_published",
                    proposal.Count, selection.Count, audit.Count);
                return;
            }

            bool confirming = state.ChallengeConfirmations == 0 ||
                              Mathf.Abs(winner.Center -
                                  state.ChallengePlaneCoordinate) <=
                                  MaximumSelectionResidualMetres;
            state.ChallengeConfirmations = confirming
                ? state.ChallengeConfirmations + 1
                : 1;
            state.ChallengeCenter = center;
            state.ChallengePlaneCoordinate = winner.Center;
            if (state.ChallengeConfirmations < ReplacementConfirmations)
            {
                AppendDecisionCheck(state, frame, proposal.Count,
                    selection.Count, audit.Count, hypotheses,
                    selectionCenter, selectionMembers, auditCenter,
                    auditMembers, winnerIndex, winnerResidual,
                    runnerResidual, margin, "winner_confirming");
                Transition(state, CourtVerdict.Abstain, frame,
                    "winner_confirming", proposal.Count, selection.Count, audit.Count);
                return;
            }

            AppendDecisionCheck(state, frame, proposal.Count,
                selection.Count, audit.Count, hypotheses,
                selectionCenter, selectionMembers, auditCenter,
                auditMembers, winnerIndex, winnerResidual,
                runnerResidual, margin, "winner_published");
            Publish(state, center, normal, winner.Center, planeCoordinate, frame,
                "winner_published",
                proposal.Count, selection.Count, audit.Count);
        }

        private void Publish(AddressState state, Vector3 center, Vector3 normal,
            float layerCoordinate, float planeCoordinate, int frame,
            string reason, int proposal, int selection, int audit)
        {
            CourtVerdict previous = state.Verdict;
            state.PublishedCenter = center;
            state.PublishedNormal = normal;
            state.PublishedLayerCoordinate = layerCoordinate;
            state.PublishedPlaneCoordinate = planeCoordinate;
            state.Generation = state.Generation == uint.MaxValue
                ? 1u : state.Generation + 1u;
            state.LastPublishFrame = frame;
            state.ChallengeConfirmations = 0;
            state.FailureConfirmations = 0;
            state.Verdict = CourtVerdict.Approved;
            _revision++;
            QueueInvalidation(state);
            AppendEvent(state, previous, CourtVerdict.Approved, frame, reason,
                proposal, selection, audit);
        }

        private void QueueInvalidation(AddressState state)
        {
            if (state.PublishedNormal.sqrMagnitude < 1e-8f) return;
            while (_pendingInvalidations.Count >= 64)
                _pendingInvalidations.Dequeue();
            _pendingInvalidations.Enqueue(new Vector4(state.PublishedCenter.x,
                state.PublishedCenter.y, state.PublishedCenter.z,
                InvalidationRadiusMetres));
        }

        private void Transition(AddressState state, CourtVerdict verdict, int frame,
            string reason, int proposal, int selection, int audit)
        {
            CourtVerdict previous = state.Verdict;
            if (previous == verdict) return;
            state.Verdict = verdict;
            _revision++;
            AppendEvent(state, previous, verdict, frame, reason, proposal,
                selection, audit);
        }

        private void AppendEvent(AddressState state, CourtVerdict previous,
            CourtVerdict verdict, int frame, string reason, int proposal,
            int selection, int audit)
        {
            _pendingEvents.Add(string.Join(",",
                frame.ToString(CultureInfo.InvariantCulture),
                state.StableId.ToString(CultureInfo.InvariantCulture),
                state.Views.Count.ToString(CultureInfo.InvariantCulture),
                proposal.ToString(CultureInfo.InvariantCulture),
                selection.ToString(CultureInfo.InvariantCulture),
                audit.ToString(CultureInfo.InvariantCulture),
                previous.ToString(), verdict.ToString(),
                state.Generation.ToString(CultureInfo.InvariantCulture),
                F(state.PublishedCenter.x), F(state.PublishedCenter.y),
                F(state.PublishedCenter.z), F(state.PublishedNormal.x),
                F(state.PublishedNormal.y), F(state.PublishedNormal.z),
                F(state.PublishedLayerCoordinate),
                F(state.PublishedPlaneCoordinate), reason));
        }

        private void AppendTestimony(int gunGelFrame, int sourceFrame,
            int attemptIndex, AddressState state, int archiveIndex,
            Vector3 camera, Vector3 source, Vector3 target, Vector3 normal,
            float range, float incidence, float targetResidualMm,
            float coordinate, float angularSpeed, float linearSpeed,
            float motionQuality, Vector3 headEuler)
        {
            string role = archiveIndex % 3 == 0 ? "proposal" :
                archiveIndex % 3 == 1 ? "selection" : "audit";
            _pendingTestimonies.Add(string.Join(",",
                gunGelFrame.ToString(CultureInfo.InvariantCulture),
                sourceFrame.ToString(CultureInfo.InvariantCulture),
                attemptIndex.ToString(CultureInfo.InvariantCulture),
                state.StableId.ToString(CultureInfo.InvariantCulture),
                archiveIndex.ToString(CultureInfo.InvariantCulture), role,
                F(camera.x), F(camera.y), F(camera.z),
                F(source.x), F(source.y), F(source.z),
                F(target.x), F(target.y), F(target.z),
                F(normal.x), F(normal.y), F(normal.z),
                F(state.ReferenceNormal.x), F(state.ReferenceNormal.y),
                F(state.ReferenceNormal.z), F(range), F(incidence),
                F(targetResidualMm), F(coordinate), F(angularSpeed),
                F(linearSpeed), F(motionQuality), F(headEuler.x),
                F(headEuler.y), F(headEuler.z)));
        }

        private void AppendFrameGate(int gunGelFrame, int sourceFrame,
            int attemptIndex, int totalRows, int motionRejected,
            int stableIdMissing, int evidenceRejected, int observationInvalid,
            int rawMissing, int dualDisagree, int stableMissing,
            int normalRejected, int nonFinite, int incidenceRejected,
            int candidateBandRejected, int eligibleRows, int stableBins,
            int referenceNormalRejected, int temporalDuplicateRejected,
            int independentAccepted,
            float angularSpeed, float linearSpeed, float motionQuality,
            Vector3 headEuler)
        {
            _pendingFrameGates.Add(string.Join(",",
                gunGelFrame.ToString(CultureInfo.InvariantCulture),
                sourceFrame.ToString(CultureInfo.InvariantCulture),
                attemptIndex.ToString(CultureInfo.InvariantCulture),
                totalRows.ToString(CultureInfo.InvariantCulture),
                motionRejected.ToString(CultureInfo.InvariantCulture),
                stableIdMissing.ToString(CultureInfo.InvariantCulture),
                evidenceRejected.ToString(CultureInfo.InvariantCulture),
                observationInvalid.ToString(CultureInfo.InvariantCulture),
                rawMissing.ToString(CultureInfo.InvariantCulture),
                dualDisagree.ToString(CultureInfo.InvariantCulture),
                stableMissing.ToString(CultureInfo.InvariantCulture),
                normalRejected.ToString(CultureInfo.InvariantCulture),
                nonFinite.ToString(CultureInfo.InvariantCulture),
                incidenceRejected.ToString(CultureInfo.InvariantCulture),
                candidateBandRejected.ToString(CultureInfo.InvariantCulture),
                eligibleRows.ToString(CultureInfo.InvariantCulture),
                stableBins.ToString(CultureInfo.InvariantCulture),
                referenceNormalRejected.ToString(CultureInfo.InvariantCulture),
                temporalDuplicateRejected.ToString(CultureInfo.InvariantCulture),
                independentAccepted.ToString(CultureInfo.InvariantCulture),
                F(angularSpeed), F(linearSpeed), F(motionQuality),
                F(headEuler.x), F(headEuler.y), F(headEuler.z)));
        }

        private void AppendDecisionCheck(AddressState state, int frame,
            int proposalViews, int selectionViews, int auditViews,
            List<Hypothesis> hypotheses, float selectionCenter,
            int selectionMembers, float auditCenter, int auditMembers,
            int winnerIndex, float winnerResidual, float runnerResidual,
            float margin, string outcome)
        {
            var centers = new StringBuilder(48);
            var supports = new StringBuilder(24);
            if (hypotheses != null)
            {
                for (int i = 0; i < hypotheses.Count; i++)
                {
                    if (i > 0) { centers.Append('|'); supports.Append('|'); }
                    centers.Append((hypotheses[i].Center * 1000f).ToString(
                        "R", CultureInfo.InvariantCulture));
                    supports.Append(hypotheses[i].Support.ToString(
                        CultureInfo.InvariantCulture));
                }
            }
            float selectionShare = selectionViews > 0
                ? selectionMembers / (float)selectionViews : float.NaN;
            float auditShare = auditViews > 0
                ? auditMembers / (float)auditViews : float.NaN;
            float winnerCenter = hypotheses != null && winnerIndex >= 0 &&
                                 winnerIndex < hypotheses.Count
                ? hypotheses[winnerIndex].Center : float.NaN;
            long framesSincePublish = state.LastPublishFrame <= int.MinValue / 4
                ? -1L : (long)frame - state.LastPublishFrame;
            _pendingDecisionChecks.Add(string.Join(",",
                frame.ToString(CultureInfo.InvariantCulture),
                state.StableId.ToString(CultureInfo.InvariantCulture),
                state.Generation.ToString(CultureInfo.InvariantCulture),
                state.Verdict.ToString(),
                state.Views.Count.ToString(CultureInfo.InvariantCulture),
                proposalViews.ToString(CultureInfo.InvariantCulture),
                selectionViews.ToString(CultureInfo.InvariantCulture),
                auditViews.ToString(CultureInfo.InvariantCulture),
                (hypotheses?.Count ?? 0).ToString(CultureInfo.InvariantCulture),
                centers.ToString(), supports.ToString(),
                FMm(selectionCenter),
                selectionMembers.ToString(CultureInfo.InvariantCulture),
                F(selectionShare), FMm(auditCenter),
                auditMembers.ToString(CultureInfo.InvariantCulture),
                F(auditShare), FMm(winnerCenter), FMm(winnerResidual),
                FMm(runnerResidual), FMm(margin),
                state.ChallengeConfirmations.ToString(CultureInfo.InvariantCulture),
                state.FailureConfirmations.ToString(CultureInfo.InvariantCulture),
                framesSincePublish.ToString(CultureInfo.InvariantCulture), outcome));
        }

        private static bool TryAddIndependent(AddressState state, Vector3 camera,
            Vector3 direction, Vector3 source, Vector3 target, Vector3 normal,
            float coordinate, int frame, int sourceFrame, int attemptIndex)
        {
            // Requiring every accepted pose to be 8 cm / 3 degrees away from
            // every older pose made a stationary surface impossible to approve.
            // The court selects a persistent depth layer, so temporal separation
            // is the mandatory independence axis. Camera diversity remains in
            // the testimony and naturally exposes view-dependent disagreement.
            if (state.Views.Count > 0 &&
                frame - state.Views[state.Views.Count - 1].Frame <
                IndependentViewWitnessPolicy.FrameGap)
                return false;
            // Keep a rolling, role-stable archive. Removing one row would rotate
            // proposal/selection/audit ownership for every surviving testimony;
            // retire a complete triplet so no old witness silently changes job.
            if (state.Views.Count >= MaximumIndependentViews)
                state.Views.RemoveRange(0, Mathf.Min(3, state.Views.Count));
            state.Views.Add(new Testimony
            {
                Camera = camera,
                Direction = direction,
                Source = source,
                Target = target,
                Normal = normal,
                PlaneCoordinate = coordinate,
                Frame = frame,
                SourceFrame = sourceFrame,
                AttemptIndex = attemptIndex
            });
            return true;
        }

        private List<Hypothesis> BuildHypotheses(List<float> values)
        {
            var remaining = new List<float>(values);
            var result = new List<Hypothesis>(MaximumHypotheses);
            int minimum = Mathf.Max(2,
                Mathf.CeilToInt(values.Count * MinimumProposalModeShare));
            while (remaining.Count > 0 && result.Count < MaximumHypotheses)
            {
                float seed = remaining[0];
                int bestSupport = -1;
                for (int i = 0; i < remaining.Count; i++)
                {
                    int support = 0;
                    for (int j = 0; j < remaining.Count; j++)
                        if (Mathf.Abs(remaining[j] - remaining[i]) <= ModeRadiusMetres)
                            support++;
                    if (support > bestSupport)
                    {
                        bestSupport = support;
                        seed = remaining[i];
                    }
                }
                _scratch.Clear();
                for (int i = 0; i < remaining.Count; i++)
                    if (Mathf.Abs(remaining[i] - seed) <= ModeRadiusMetres)
                        _scratch.Add(remaining[i]);
                if (_scratch.Count < minimum) break;
                float center = Median(_scratch);
                _scratch.Clear();
                for (int i = 0; i < remaining.Count; i++)
                    if (Mathf.Abs(remaining[i] - center) <= ModeRadiusMetres)
                        _scratch.Add(remaining[i]);
                if (_scratch.Count < minimum) break;
                result.Add(new Hypothesis(center, _scratch.Count));
                for (int i = remaining.Count - 1; i >= 0; i--)
                    if (Mathf.Abs(remaining[i] - center) <= ModeRadiusMetres)
                        remaining.RemoveAt(i);
            }
            if (result.Count == 0)
                result.Add(new Hypothesis(Median(values), values.Count));
            result.Sort((a, b) => b.Support != a.Support
                ? b.Support.CompareTo(a.Support)
                : a.Center.CompareTo(b.Center));
            return result;
        }

        private static Vector3 WinnerNormal(AddressState state,
            float winnerCoordinate)
        {
            Vector3 sum = Vector3.zero;
            int count = 0;
            for (int i = 0; i < state.Views.Count; i++)
            {
                Testimony view = state.Views[i];
                if (Mathf.Abs(view.PlaneCoordinate - winnerCoordinate) >
                    ModeRadiusMetres)
                    continue;
                Vector3 normal = view.Normal;
                if (normal.sqrMagnitude < 1e-8f) continue;
                normal.Normalize();
                if (Vector3.Dot(normal, state.ReferenceNormal) < 0f)
                    normal = -normal;
                sum += normal;
                count++;
            }
            if (count == 0 || sum.sqrMagnitude < 1e-8f)
                return state.ReferenceNormal;
            return sum.normalized;
        }

        private static Vector3 WinnerCenter(AddressState state,
            float winnerCoordinate)
        {
            var sources = new List<Vector3>(state.Views.Count);
            for (int i = 0; i < state.Views.Count; i++)
            {
                Testimony view = state.Views[i];
                if (Mathf.Abs(view.PlaneCoordinate - winnerCoordinate) <=
                    ModeRadiusMetres)
                    sources.Add(view.Source);
            }
            return sources.Count > 0
                ? MedianVector(sources)
                : state.Views[state.Views.Count - 1].Source;
        }

        private static void Mode(List<float> values, out float center,
            out int members)
        {
            float seed = values[0];
            members = -1;
            for (int i = 0; i < values.Count; i++)
            {
                int support = 0;
                for (int j = 0; j < values.Count; j++)
                    if (Mathf.Abs(values[j] - values[i]) <= ModeRadiusMetres)
                        support++;
                if (support > members)
                {
                    members = support;
                    seed = values[i];
                }
            }
            var local = new List<float>(values.Count);
            for (int i = 0; i < values.Count; i++)
                if (Mathf.Abs(values[i] - seed) <= ModeRadiusMetres)
                    local.Add(values[i]);
            center = Median(local);
            members = local.Count;
        }

        private static float Median(List<float> values)
        {
            values.Sort();
            int middle = values.Count / 2;
            return (values.Count & 1) != 0
                ? values[middle]
                : 0.5f * (values[middle - 1] + values[middle]);
        }

        private static Vector3 MedianVector(List<Vector3> values)
        {
            var x = new List<float>(values.Count);
            var y = new List<float>(values.Count);
            var z = new List<float>(values.Count);
            for (int i = 0; i < values.Count; i++)
            {
                x.Add(values[i].x); y.Add(values[i].y); z.Add(values[i].z);
            }
            return new Vector3(Median(x), Median(y), Median(z));
        }

        private static Vector3 AlignedMean(List<Vector3> values)
        {
            Vector3 reference = values[0].normalized;
            Vector3 sum = Vector3.zero;
            for (int i = 0; i < values.Count; i++)
            {
                Vector3 value = values[i].normalized;
                sum += Vector3.Dot(value, reference) < 0f ? -value : value;
            }
            return sum.sqrMagnitude > 1e-8f ? sum.normalized : reference;
        }

        private void FlushEventsLocked(bool force = false)
        {
            int pending = _pendingEvents.Count + _pendingTestimonies.Count +
                          _pendingFrameGates.Count + _pendingDecisionChecks.Count;
            if (!force && pending < 512) return;
            try
            {
                AppendPending(_eventsPath, _pendingEvents);
                AppendPending(_testimoniesPath, _pendingTestimonies);
                AppendPending(_frameGatesPath, _pendingFrameGates);
                AppendPending(_decisionChecksPath, _pendingDecisionChecks);
            }
            catch
            {
                _writeErrors++;
            }
        }

        private static void AppendPending(string path, List<string> rows)
        {
            if (rows.Count == 0) return;
            // The production court deliberately has no files.  Discard its
            // optional CSV rows after the same batching point so an hour-long
            // scan cannot turn diagnostics into an unbounded memory queue.
            if (string.IsNullOrEmpty(path))
            {
                rows.Clear();
                return;
            }
            File.AppendAllLines(path, rows, new UTF8Encoding(false));
            rows.Clear();
        }

        private void ResetLocked()
        {
            EndLocked();
            _addresses.Clear();
            _frameBins.Clear();
            _pendingInvalidations.Clear();
            _pendingEvents.Clear();
            _pendingTestimonies.Clear();
            _pendingFrameGates.Clear();
            _pendingDecisionChecks.Clear();
            _eventsPath = string.Empty;
            _testimoniesPath = string.Empty;
            _frameGatesPath = string.Empty;
            _decisionChecksPath = string.Empty;
            _revision = 0;
            _writeErrors = 0;
        }

        private void EndLocked()
        {
            if (!_active) return;
            FlushEventsLocked(true);
            _active = false;
        }

        private static Vector3 SafeDirection(Vector3 value) =>
            value.sqrMagnitude > 1e-8f ? value.normalized : Vector3.forward;

        private static bool Finite(Vector3 value) =>
            !float.IsNaN(value.x) && !float.IsInfinity(value.x) &&
            !float.IsNaN(value.y) && !float.IsInfinity(value.y) &&
            !float.IsNaN(value.z) && !float.IsInfinity(value.z);

        private static string F(float value) =>
            value.ToString("R", CultureInfo.InvariantCulture);

        private static string FMm(float metres) =>
            float.IsNaN(metres) || float.IsInfinity(metres)
                ? string.Empty
                : (metres * 1000f).ToString("R", CultureInfo.InvariantCulture);

        private static string BuildSchema() =>
            "{\n" +
            "  \"schema\": \"scan-cover-runtime-final-surface-court-v3\",\n" +
            "  \"authority\": \"publishes a held-out-audited plane constraint; never owns a mesh, volume or TSDF write\",\n" +
            "  \"address\": \"GunGel stableId is association only, never truth\",\n" +
            "  \"testimony\": \"one median coordinate per time-separated observation, projected onto one stable-id reference normal; pose diversity is evidence, not a startup requirement\",\n" +
            "  \"roles\": \"interleaved proposal selection held-out audit; selection and audit must independently agree with the proposed winner\",\n" +
            "  \"publication\": \"strict winner plane only; raw depth proves membership, the plane constrains only the extracted mesh product, and replacement atomically supersedes the prior visible local generation without rewriting TSDF\",\n" +
            "  \"responsibilityLedger\": {\"frame_gate_summary.csv\":\"overlapping gate counts per recorded fusion frame\",\"independent_testimonies.csv\":\"one row only when a time-separated stable-id testimony enters the archive\",\"decision_checks.csv\":\"every rejudge attempt including insufficient evidence, abstention, hysteresis, cooldown and publication\",\"runtime_events.csv\":\"formal verdict and generation transitions\"},\n" +
            "  \"performanceBoundary\": \"no per-pixel disk log; rejected correspondences are counted per frame, while exact spatial rejection remains recoverable from sealed fusion frame payloads\",\n" +
            "  \"thresholds\": {\"minimumRoleViews\":5,\"candidateBandMm\":150,\"minimumIncidence\":0.30,\"modeRadiusMm\":20,\"maximumHypotheses\":2,\"selectionResidualMm\":15,\"selectionModeShare\":0.60,\"winnerMarginMm\":10,\"replacementConfirmations\":3,\"revocationConfirmations\":3,\"replacementCooldownFrames\":30,\"invalidationRadiusMm\":125}\n" +
            "}\n";
    }
}
