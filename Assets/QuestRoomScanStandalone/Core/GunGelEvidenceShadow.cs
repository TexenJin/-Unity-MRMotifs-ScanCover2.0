using System;
using System.Collections;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// “校枪—分层凝胶—定型”GPU 证据与校枪层。
    /// 它异步估计融合专用整帧小修正，并用修正后的观测维护最多三层局部候选；
    /// 本类自身不写 TSDF；受保护实验可由 VolumeIntegrator 延迟消费同帧校枪结果。
    /// </summary>
    internal sealed class GunGelEvidenceShadow : IDisposable
    {
        internal readonly struct PaperAuthorityBuffers
        {
            internal readonly ComputeBuffer CellKeys;
            internal readonly ComputeBuffer CellState;
            internal readonly ComputeBuffer CandidateCenterSigma;
            internal readonly ComputeBuffer CandidateNormalSupport;
            internal readonly ComputeBuffer CandidateMeta;
            internal readonly ComputeBuffer CandidateEvidence;
            internal readonly int TableCapacity;
            internal readonly int TableMask;
            internal readonly int ProbeCount;
            internal readonly int CandidateCapacity;
            internal readonly float CellSize;

            internal PaperAuthorityBuffers(ComputeBuffer cellKeys,
                ComputeBuffer cellState, ComputeBuffer candidateCenterSigma,
                ComputeBuffer candidateNormalSupport, ComputeBuffer candidateMeta,
                ComputeBuffer candidateEvidence, float cellSize)
            {
                CellKeys = cellKeys;
                CellState = cellState;
                CandidateCenterSigma = candidateCenterSigma;
                CandidateNormalSupport = candidateNormalSupport;
                CandidateMeta = candidateMeta;
                CandidateEvidence = candidateEvidence;
                TableCapacity = GunGelEvidenceShadow.TableCapacity;
                TableMask = GunGelEvidenceShadow.TableCapacity - 1;
                ProbeCount = GunGelEvidenceShadow.ProbeCount;
                CandidateCapacity = GunGelEvidenceShadow.CandidateCapacity;
                CellSize = cellSize;
            }

            internal bool IsValid => CellKeys != null && CellState != null &&
                CandidateCenterSigma != null && CandidateNormalSupport != null &&
                CandidateMeta != null && CandidateEvidence != null;
        }

        /// <summary>
        /// “枪胶裁决海面”只读GPU视图。候选缓冲提供绿色静海；Wave缓冲按稳定ID
        /// 保存扫描期间偏离最严重的红色浪头及其当时裁决。任何生产内核都不读取后者。
        /// </summary>
        internal readonly struct CourtBuffers
        {
            internal readonly ComputeBuffer CandidateCenterSigma;
            internal readonly ComputeBuffer CandidateNormalSupport;
            internal readonly ComputeBuffer CandidateMeta;
            internal readonly ComputeBuffer CandidateEvidence;
            internal readonly ComputeBuffer WavePeakBits;
            internal readonly ComputeBuffer WavePositionResidual;
            internal readonly ComputeBuffer WaveTargetSigma;
            internal readonly ComputeBuffer WaveMeta;
            internal readonly ComputeBuffer WaveCounters;
            internal readonly int CandidateCount;
            internal readonly int WaveCapacity;

            internal CourtBuffers(ComputeBuffer candidateCenterSigma,
                ComputeBuffer candidateNormalSupport, ComputeBuffer candidateMeta,
                ComputeBuffer candidateEvidence, ComputeBuffer wavePeakBits,
                ComputeBuffer wavePositionResidual, ComputeBuffer waveTargetSigma,
                ComputeBuffer waveMeta, ComputeBuffer waveCounters)
            {
                CandidateCenterSigma = candidateCenterSigma;
                CandidateNormalSupport = candidateNormalSupport;
                CandidateMeta = candidateMeta;
                CandidateEvidence = candidateEvidence;
                WavePeakBits = wavePeakBits;
                WavePositionResidual = wavePositionResidual;
                WaveTargetSigma = waveTargetSigma;
                WaveMeta = waveMeta;
                WaveCounters = waveCounters;
                CandidateCount = TableCapacity * CandidateCapacity;
                WaveCapacity = CourtWaveCapacity;
            }

            internal bool IsValid => CandidateCenterSigma != null &&
                CandidateNormalSupport != null && CandidateMeta != null &&
                CandidateEvidence != null && WavePeakBits != null &&
                WavePositionResidual != null && WaveTargetSigma != null &&
                WaveMeta != null && WaveCounters != null;
        }

        internal readonly struct FrameDecision
        {
            public readonly int FrameIndex;
            public readonly Matrix4x4 Correction;
            public readonly int CorrespondenceCount;
            public readonly int EffectiveRank;
            public readonly float TranslationMm;
            public readonly float RotationDeg;
            public readonly bool TranslationClamped;
            public readonly bool RotationClamped;
            public readonly bool ReadbackSucceeded;
            internal readonly ComputeBuffer FusionObservations;
            internal readonly ComputeBuffer FusionCorrespondences;
            internal readonly int FusionObservationGridX;
            internal readonly int FusionObservationGridY;
            internal readonly int FusionPixelStride;
            internal readonly int FusionSlotIndex;
            internal readonly int FusionSlotToken;

            internal bool HasFusionAdmissionBuffers =>
                FusionObservations != null && FusionCorrespondences != null &&
                FusionObservationGridX > 0 && FusionObservationGridY > 0 &&
                FusionPixelStride > 0 && FusionSlotIndex >= 0 && FusionSlotToken != 0;

            public FrameDecision(int frameIndex, Matrix4x4 correction,
                int correspondenceCount, int effectiveRank,
                float translationMm, float rotationDeg,
                bool translationClamped, bool rotationClamped,
                bool readbackSucceeded,
                ComputeBuffer fusionObservations, ComputeBuffer fusionCorrespondences,
                int fusionObservationGridX, int fusionObservationGridY,
                int fusionPixelStride, int fusionSlotIndex, int fusionSlotToken)
            {
                FrameIndex = frameIndex;
                Correction = correction;
                CorrespondenceCount = correspondenceCount;
                EffectiveRank = effectiveRank;
                TranslationMm = translationMm;
                RotationDeg = rotationDeg;
                TranslationClamped = translationClamped;
                RotationClamped = rotationClamped;
                ReadbackSucceeded = readbackSucceeded;
                FusionObservations = fusionObservations;
                FusionCorrespondences = fusionCorrespondences;
                FusionObservationGridX = fusionObservationGridX;
                FusionObservationGridY = fusionObservationGridY;
                FusionPixelStride = fusionPixelStride;
                FusionSlotIndex = fusionSlotIndex;
                FusionSlotToken = fusionSlotToken;
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct Correspondence
        {
            public Vector4 SourceValid;
            public Vector4 TargetSigma;
            public Vector4 NormalAngle;
        }

        private sealed class FrameSlot
        {
            public ComputeBuffer Observations;
            public ComputeBuffer Correspondences;
            public bool Pending;
            public bool ConsumerHeld;
            public bool Adjudicated;
            public bool CandidateCommitted;
            public int ObservationCount;
            public int FrameIndex;
            public int GridX;
            public int GridY;
            public int DepthWidth;
            public int DepthHeight;
            public float MotionQuality;
            public float AngularSpeed;
            public float LinearSpeed;
            public bool HasRawDepth;
            public bool HasTemporalReason;
            public bool ReportFrame;
            public Matrix4x4[] ViewInv;
            public int Generation;
            public int SlotIndex;
            public int ConsumerToken;
            public Action<FrameDecision> Completion;
        }

        private const int TableCapacity = 65536;
        private const int CandidateCapacity = 3;
        private const int ProbeCount = 12;
        private const uint FusionAuthorityMask = 0x3Fu;
        private const int RingSize = 3;
        private const int StatsCount = 25;
        private const int AuditTotalsCount = 9;
        private const int RetireAuditCapacity = 32768;
        // stableID 从 1 单调递增；该容量覆盖完整候选表容量并给淘汰后重生留余量。
        // 超出时只计 overflow，不回绕覆盖旧浪头。
        private const int CourtWaveCapacity = 262144;
        // GPU Observation = 3 x float4 + 1 x uint4. Keep replay metadata in sync.
        private const int ObservationStride = sizeof(float) * 12 + sizeof(uint) * 4;
        private const int CorrespondenceStride = sizeof(float) * 12;
        private const int CandidateFloat4Stride = sizeof(float) * 4;
        private const int CandidateMetaStride = sizeof(uint) * 4;

        private static readonly int DepthTexID = Shader.PropertyToID("_DepthTex");
        private static readonly int RawDepthTexID = Shader.PropertyToID("_RawDepthTex");
        private static readonly int NormalTexID = Shader.PropertyToID("_NormalTex");
        private static readonly int EdgeReasonTexID = Shader.PropertyToID("_EdgeReasonTex");
        private static readonly int TemporalReasonTexID = Shader.PropertyToID("_TemporalReasonTex");
        private static readonly int DepthProjInvID = Shader.PropertyToID("_DepthProjInv");
        private static readonly int DepthViewInvID = Shader.PropertyToID("_DepthViewInv");
        private static readonly int CorrectionID = Shader.PropertyToID("_Correction");
        private static readonly int DepthSizeID = Shader.PropertyToID("_DepthSize");
        private static readonly int ObservationGridID = Shader.PropertyToID("_ObservationGrid");
        private static readonly int PixelStrideID = Shader.PropertyToID("_PixelStride");
        private static readonly int ObservationCountID = Shader.PropertyToID("_ObservationCount");
        private static readonly int TableCapacityID = Shader.PropertyToID("_TableCapacity");
        private static readonly int TableMaskID = Shader.PropertyToID("_TableMask");
        private static readonly int ProbeCountID = Shader.PropertyToID("_ProbeCount");
        private static readonly int CandidateCapacityID = Shader.PropertyToID("_CandidateCapacity");
        private static readonly int FrameIndexID = Shader.PropertyToID("_FrameIndex");
        private static readonly int RawDepthEyeID = Shader.PropertyToID("_RawDepthEye");
        private static readonly int HasRawDepthID = Shader.PropertyToID("_HasRawDepth");
        private static readonly int HasTemporalReasonID = Shader.PropertyToID("_HasTemporalReason");
        private static readonly int PlatformFrameID = Shader.PropertyToID("_PlatformFrame");
        private static readonly int CellSizeID = Shader.PropertyToID("_CellSize");
        private static readonly int MotionQualityID = Shader.PropertyToID("_MotionQuality");
        private static readonly int AngularSpeedID = Shader.PropertyToID("_AngularSpeed");
        private static readonly int LinearSpeedID = Shader.PropertyToID("_LinearSpeed");
        private static readonly int MaxDepthID = Shader.PropertyToID("_MaxDepth");
        private static readonly int ObservationsID = Shader.PropertyToID("_Observations");
        private static readonly int CorrespondencesID = Shader.PropertyToID("_Correspondences");
        private static readonly int CellKeysID = Shader.PropertyToID("_CellKeys");
        private static readonly int CellStateID = Shader.PropertyToID("_CellState");
        private static readonly int CellLockID = Shader.PropertyToID("_CellLock");
        private static readonly int CandidateCenterSigmaID = Shader.PropertyToID("_CandidateCenterSigma");
        private static readonly int CandidateNormalSupportID = Shader.PropertyToID("_CandidateNormalSupport");
        private static readonly int CandidateFirstViewAngleID = Shader.PropertyToID("_CandidateFirstViewAngle");
        private static readonly int CandidateMetaID = Shader.PropertyToID("_CandidateMeta");
        private static readonly int CandidateEvidenceID = Shader.PropertyToID("_CandidateEvidence");
        private static readonly int CandidateSuccessorID = Shader.PropertyToID("_CandidateSuccessor");
        private static readonly int CandidatePredecessorID = Shader.PropertyToID("_CandidatePredecessor");
        private static readonly int CandidateConsensusID = Shader.PropertyToID("_CandidateConsensus");
        private static readonly int NextStableIdID = Shader.PropertyToID("_NextStableId");
        private static readonly int StatsID = Shader.PropertyToID("_Stats");
        private static readonly int AuditTotalsID = Shader.PropertyToID("_AuditTotals");
        private static readonly int RetireAuditCountersID = Shader.PropertyToID("_RetireAuditCounters");
        private static readonly int RetireAuditMetaID = Shader.PropertyToID("_RetireAuditMeta");
        private static readonly int RetireAuditGeometryID = Shader.PropertyToID("_RetireAuditGeometry");
        private static readonly int RetireAuditEvidenceID = Shader.PropertyToID("_RetireAuditEvidence");
        private static readonly int RetireAuditTimingID = Shader.PropertyToID("_RetireAuditTiming");
        private static readonly int RetireAuditCapacityID = Shader.PropertyToID("_RetireAuditCapacity");
        private static readonly int CourtWaveLockID = Shader.PropertyToID("_CourtWaveLock");
        private static readonly int CourtWavePeakBitsID = Shader.PropertyToID("_CourtWavePeakBits");
        private static readonly int CourtWavePositionResidualID = Shader.PropertyToID("_CourtWavePositionResidual");
        private static readonly int CourtWaveTargetSigmaID = Shader.PropertyToID("_CourtWaveTargetSigma");
        private static readonly int CourtWaveMetaID = Shader.PropertyToID("_CourtWaveMeta");
        private static readonly int CourtWaveSourceReasonID = Shader.PropertyToID("_CourtWaveSourceReason");
        private static readonly int CourtWaveRawPositionDeltaID = Shader.PropertyToID("_CourtWaveRawPositionDelta");
        private static readonly int CourtWaveMotionViewID = Shader.PropertyToID("_CourtWaveMotionView");
        private static readonly int CourtWaveCountersID = Shader.PropertyToID("_CourtWaveCounters");
        private static readonly int CourtWaveCapacityID = Shader.PropertyToID("_CourtWaveCapacity");

        private readonly ComputeShader _shader;
        private readonly ComputeKernelHelper _clearCells;
        private readonly ComputeKernelHelper _clearStats;
        private readonly ComputeKernelHelper _buildObservations;
        private readonly ComputeKernelHelper _buildCorrespondences;
        private readonly ComputeKernelHelper _captureCourtWaves;
        private readonly ComputeKernelHelper _updateCandidates;
        private readonly ComputeKernelHelper _computeCandidateConsensus;
        private readonly ComputeKernelHelper _markCandidateSuccessions;
        private readonly ComputeKernelHelper _applyCandidateSuccessions;
        private readonly ComputeKernelHelper _rebuildProductionCorrespondences;
        private readonly ComputeKernelHelper _challengeCandidates;
        private readonly ComputeKernelHelper _retireCandidates;
        private readonly ComputeKernelHelper _countCensus;
        private readonly FrameSlot[] _slots = new FrameSlot[RingSize];
        private readonly int _pixelStride;
        private readonly float _cellSize;
        private readonly int _reportInterval;

        private ComputeBuffer _cellKeys;
        private ComputeBuffer _cellState;
        private ComputeBuffer _cellLock;
        private ComputeBuffer _candidateCenterSigma;
        private ComputeBuffer _candidateNormalSupport;
        private ComputeBuffer _candidateFirstViewAngle;
        private ComputeBuffer _candidateMeta;
        private ComputeBuffer _candidateEvidence;
        private ComputeBuffer _candidateSuccessor;
        private ComputeBuffer _candidatePredecessor;
        private ComputeBuffer _candidateConsensus;
        private ComputeBuffer _nextStableId;
        private ComputeBuffer _stats;
        private ComputeBuffer _auditTotals;
        private ComputeBuffer _retireAuditCounters;
        private ComputeBuffer _retireAuditMeta;
        private ComputeBuffer _retireAuditGeometry;
        private ComputeBuffer _retireAuditEvidence;
        private ComputeBuffer _retireAuditTiming;
        private ComputeBuffer _courtWaveLock;
        private ComputeBuffer _courtWavePeakBits;
        private ComputeBuffer _courtWavePositionResidual;
        private ComputeBuffer _courtWaveTargetSigma;
        private ComputeBuffer _courtWaveMeta;
        private ComputeBuffer _courtWaveSourceReason;
        private ComputeBuffer _courtWaveRawPositionDelta;
        private ComputeBuffer _courtWaveMotionView;
        private ComputeBuffer _courtWaveCounters;
        private bool _disposed;
        private bool _auditExportPending;
        private bool _courtSealPending;
        private bool _courtSealed;
        private int _nextSlot;
        private int _nextConsumerToken;
        private int _generation;
        private uint _candidateTransactionsCommitted;
        private uint _candidateBootstrapTransactions;
        private uint _candidateTransactionsDiscarded;
        private uint _candidateTransactionDuplicateAttempts;

        private int _lastCorrespondences;
        private int _lastEffectiveRank;
        private float _lastTranslationMm;
        private float _lastRotationDeg;
        private uint _lastValidObservations;
        private uint _lastBirths;
        private uint _lastUpdates;
        private uint _lastPromotions;
        private uint _lastCapacityAbstentions;
        private uint _lastContentions;
        private uint _lastActiveCandidates;
        private uint _lastStableCandidates;
        private uint _lastMultiStableCells;
        private uint _lastStaleStableCandidates;
        private uint _lastDualAgree;
        private uint _lastDualDisagree;
        private uint _lastChallengeVotes;
        private uint _lastContradictionRetirements;
        private uint _lastEmbryoExpirations;
        private uint _lastStableIdsAssigned;
        private uint _lastRawUnavailable;
        private uint _lastStableIdsAlive;
        private uint _lastCandidateSuccessions;
        private bool _hasReport;

        public bool AuditExportPending => _auditExportPending;
        public bool CourtSealPending => _courtSealPending;
        public bool CourtSealed => _courtSealed;

        internal bool TryGetPaperAuthority(out PaperAuthorityBuffers authority)
        {
            authority = default;
            if (_disposed || _cellKeys == null || _cellState == null ||
                _candidateCenterSigma == null || _candidateNormalSupport == null ||
                _candidateMeta == null || _candidateEvidence == null)
                return false;

            authority = new PaperAuthorityBuffers(_cellKeys, _cellState,
                _candidateCenterSigma, _candidateNormalSupport,
                _candidateMeta, _candidateEvidence, _cellSize);
            return authority.IsValid;
        }

        internal bool TryGetCourtBuffers(out CourtBuffers court)
        {
            court = default;
            if (_disposed) return false;
            court = new CourtBuffers(_candidateCenterSigma, _candidateNormalSupport,
                _candidateMeta, _candidateEvidence, _courtWavePeakBits,
                _courtWavePositionResidual, _courtWaveTargetSigma,
                _courtWaveMeta, _courtWaveCounters);
            return court.IsValid;
        }

        public GunGelEvidenceShadow(int pixelStride, float cellSize, int reportInterval)
        {
            _pixelStride = Mathf.Max(2, pixelStride);
            _cellSize = Mathf.Max(0.05f, cellSize);
            _reportInterval = Mathf.Max(10, reportInterval);
            _shader = Resources.Load<ComputeShader>("GunGelEvidenceShadow");
            if (_shader == null)
                throw new InvalidOperationException("Resources/GunGelEvidenceShadow.compute not found");

            _clearCells = new ComputeKernelHelper(_shader, "ClearCells");
            _clearStats = new ComputeKernelHelper(_shader, "ClearStats");
            _buildObservations = new ComputeKernelHelper(_shader, "BuildObservations");
            _buildCorrespondences = new ComputeKernelHelper(_shader, "BuildCorrespondences");
            _captureCourtWaves = new ComputeKernelHelper(_shader, "CaptureCourtWaves");
            _updateCandidates = new ComputeKernelHelper(_shader, "UpdateCandidates");
            _computeCandidateConsensus = new ComputeKernelHelper(_shader, "ComputeCandidateConsensus");
            _markCandidateSuccessions = new ComputeKernelHelper(_shader, "MarkCandidateSuccessions");
            _applyCandidateSuccessions = new ComputeKernelHelper(_shader, "ApplyCandidateSuccessions");
            _rebuildProductionCorrespondences = new ComputeKernelHelper(_shader, "RebuildProductionCorrespondences");
            _challengeCandidates = new ComputeKernelHelper(_shader, "ChallengeCandidates");
            _retireCandidates = new ComputeKernelHelper(_shader, "RetireCandidates");
            _countCensus = new ComputeKernelHelper(_shader, "CountCensus");
            AllocatePersistentBuffers();
            BindPersistentBuffers();
            SetSharedConstants();
            Clear();
        }

        private void AllocatePersistentBuffers()
        {
            int candidateCount = TableCapacity * CandidateCapacity;
            _cellKeys = new ComputeBuffer(TableCapacity, sizeof(int) * 4);
            _cellState = new ComputeBuffer(TableCapacity, sizeof(uint));
            _cellLock = new ComputeBuffer(TableCapacity, sizeof(uint));
            _candidateCenterSigma = new ComputeBuffer(candidateCount, CandidateFloat4Stride);
            _candidateNormalSupport = new ComputeBuffer(candidateCount, CandidateFloat4Stride);
            _candidateFirstViewAngle = new ComputeBuffer(candidateCount, CandidateFloat4Stride);
            _candidateMeta = new ComputeBuffer(candidateCount, CandidateMetaStride);
            _candidateEvidence = new ComputeBuffer(candidateCount, CandidateMetaStride);
            _candidateSuccessor = new ComputeBuffer(candidateCount, sizeof(uint));
            _candidatePredecessor = new ComputeBuffer(candidateCount, sizeof(uint));
            _candidateConsensus = new ComputeBuffer(candidateCount, sizeof(uint));
            _nextStableId = new ComputeBuffer(1, sizeof(uint));
            _nextStableId.SetData(new uint[] { 1u });
            _stats = new ComputeBuffer(StatsCount, sizeof(uint));
            _stats.SetData(new uint[StatsCount]);
            _auditTotals = new ComputeBuffer(AuditTotalsCount, sizeof(uint));
            _auditTotals.SetData(new uint[AuditTotalsCount]);
            _retireAuditCounters = new ComputeBuffer(1, sizeof(uint));
            _retireAuditCounters.SetData(new uint[1]);
            _retireAuditMeta = new ComputeBuffer(RetireAuditCapacity, CandidateMetaStride);
            _retireAuditGeometry = new ComputeBuffer(RetireAuditCapacity, CandidateFloat4Stride);
            _retireAuditEvidence = new ComputeBuffer(RetireAuditCapacity, CandidateMetaStride);
            _retireAuditTiming = new ComputeBuffer(RetireAuditCapacity, CandidateMetaStride);
            _courtWaveLock = new ComputeBuffer(CourtWaveCapacity, sizeof(uint));
            _courtWavePeakBits = new ComputeBuffer(CourtWaveCapacity, sizeof(uint));
            _courtWavePositionResidual = new ComputeBuffer(CourtWaveCapacity, CandidateFloat4Stride);
            _courtWaveTargetSigma = new ComputeBuffer(CourtWaveCapacity, CandidateFloat4Stride);
            _courtWaveMeta = new ComputeBuffer(CourtWaveCapacity, CandidateMetaStride);
            _courtWaveSourceReason = new ComputeBuffer(CourtWaveCapacity, CandidateMetaStride);
            _courtWaveRawPositionDelta = new ComputeBuffer(CourtWaveCapacity, CandidateFloat4Stride);
            _courtWaveMotionView = new ComputeBuffer(CourtWaveCapacity, CandidateFloat4Stride);
            _courtWaveCounters = new ComputeBuffer(3, sizeof(uint));
            for (int i = 0; i < _slots.Length; i++)
                _slots[i] = new FrameSlot { SlotIndex = i };
        }

        private void BindPersistentBuffers()
        {
            var kernels = new[]
            {
                _clearCells, _clearStats, _buildObservations, _buildCorrespondences,
                _captureCourtWaves,
                _updateCandidates, _computeCandidateConsensus,
                _markCandidateSuccessions, _applyCandidateSuccessions,
                _rebuildProductionCorrespondences,
                _challengeCandidates, _retireCandidates,
                _countCensus
            };
            foreach (var kernel in kernels)
            {
                kernel.Set(CellKeysID, _cellKeys);
                kernel.Set(CellStateID, _cellState);
                kernel.Set(CellLockID, _cellLock);
                kernel.Set(CandidateCenterSigmaID, _candidateCenterSigma);
                kernel.Set(CandidateNormalSupportID, _candidateNormalSupport);
                kernel.Set(CandidateFirstViewAngleID, _candidateFirstViewAngle);
                kernel.Set(CandidateMetaID, _candidateMeta);
                kernel.Set(CandidateEvidenceID, _candidateEvidence);
                kernel.Set(CandidateSuccessorID, _candidateSuccessor);
                kernel.Set(CandidatePredecessorID, _candidatePredecessor);
                kernel.Set(CandidateConsensusID, _candidateConsensus);
                kernel.Set(NextStableIdID, _nextStableId);
                kernel.Set(StatsID, _stats);
                kernel.Set(AuditTotalsID, _auditTotals);
                kernel.Set(RetireAuditCountersID, _retireAuditCounters);
                kernel.Set(RetireAuditMetaID, _retireAuditMeta);
                kernel.Set(RetireAuditGeometryID, _retireAuditGeometry);
                kernel.Set(RetireAuditEvidenceID, _retireAuditEvidence);
                kernel.Set(RetireAuditTimingID, _retireAuditTiming);
                kernel.Set(CourtWaveLockID, _courtWaveLock);
                kernel.Set(CourtWavePeakBitsID, _courtWavePeakBits);
                kernel.Set(CourtWavePositionResidualID, _courtWavePositionResidual);
                kernel.Set(CourtWaveTargetSigmaID, _courtWaveTargetSigma);
                kernel.Set(CourtWaveMetaID, _courtWaveMeta);
                kernel.Set(CourtWaveSourceReasonID, _courtWaveSourceReason);
                kernel.Set(CourtWaveRawPositionDeltaID, _courtWaveRawPositionDelta);
                kernel.Set(CourtWaveMotionViewID, _courtWaveMotionView);
                kernel.Set(CourtWaveCountersID, _courtWaveCounters);
            }
        }

        private void SetSharedConstants()
        {
            _shader.SetInt(TableCapacityID, TableCapacity);
            _shader.SetInt(TableMaskID, TableCapacity - 1);
            _shader.SetInt(ProbeCountID, ProbeCount);
            _shader.SetInt(CandidateCapacityID, CandidateCapacity);
            _shader.SetInt(PixelStrideID, _pixelStride);
            _shader.SetFloat(CellSizeID, _cellSize);
            _shader.SetFloat(MaxDepthID, 5f);
            _shader.SetInt(RetireAuditCapacityID, RetireAuditCapacity);
            _shader.SetInt(CourtWaveCapacityID, CourtWaveCapacity);
        }

        private bool EnsureFrameBuffers(int observationCount)
        {
            if (_slots[0].Observations != null && _slots[0].ObservationCount == observationCount)
                return true;
            foreach (var slot in _slots)
                if (slot.Pending || slot.ConsumerHeld) return false;
            foreach (var slot in _slots)
            {
                slot.Observations?.Release();
                slot.Correspondences?.Release();
                slot.Observations = new ComputeBuffer(observationCount, ObservationStride);
                slot.Correspondences = new ComputeBuffer(observationCount, CorrespondenceStride);
                slot.ObservationCount = observationCount;
            }
            return true;
        }

        public bool Dispatch(DepthCapture depth, float motionQuality, int frameIndex,
            Action<FrameDecision> completion = null)
        {
            if (_disposed || depth == null || depth.DepthTex == null || depth.NormTex == null ||
                depth.DepthWidth <= 0 || depth.DepthHeight <= 0) return false;

            return Dispatch(depth.PlatformDepthWitnessTex, depth.DepthTex, depth.NormTex,
                depth.EdgeReasonTex, depth.TemporalReasonTex,
                depth.DepthWidth, depth.DepthHeight,
                depth.ProjInv, depth.ViewInv,
                motionQuality, depth.SmoothedDepthAngularSpeed,
                depth.SmoothedDepthLinearSpeed, depth.CurrentPlatformFrame,
                frameIndex, completion);
        }

        public bool Dispatch(Texture rawDepthTexture, Texture depthTexture, Texture normalTexture,
            Texture edgeReasonTexture, Texture temporalReasonTexture,
            int depthWidth, int depthHeight,
            Matrix4x4[] projectionInverse, Matrix4x4[] viewInverse,
            float motionQuality, float angularSpeed, float linearSpeed,
            int platformFrame, int frameIndex,
            Action<FrameDecision> completion = null)
        {
            if (_disposed || _courtSealed || _courtSealPending ||
                depthTexture == null || normalTexture == null || edgeReasonTexture == null ||
                depthWidth <= 0 || depthHeight <= 0 ||
                projectionInverse == null || projectionInverse.Length == 0 ||
                viewInverse == null || viewInverse.Length == 0) return false;

            int gridX = Mathf.CeilToInt(depthWidth / (float)_pixelStride);
            int gridY = Mathf.CeilToInt(depthHeight / (float)_pixelStride);
            int observationCount = gridX * gridY;
            if (!EnsureFrameBuffers(observationCount)) return false;

            FrameSlot slot = null;
            for (int offset = 0; offset < _slots.Length; offset++)
            {
                int index = (_nextSlot + offset) % _slots.Length;
                if (_slots[index].Pending || _slots[index].ConsumerHeld) continue;
                slot = _slots[index];
                _nextSlot = (index + 1) % _slots.Length;
                break;
            }
            if (slot == null) return false;

            slot.Pending = true;
            slot.Adjudicated = false;
            slot.CandidateCommitted = false;
            slot.FrameIndex = frameIndex;
            slot.GridX = gridX;
            slot.GridY = gridY;
            slot.DepthWidth = depthWidth;
            slot.DepthHeight = depthHeight;
            slot.MotionQuality = Mathf.Clamp01(motionQuality);
            slot.AngularSpeed = Mathf.Max(0f, angularSpeed);
            slot.LinearSpeed = Mathf.Max(0f, linearSpeed);
            slot.HasRawDepth = rawDepthTexture != null;
            slot.HasTemporalReason = temporalReasonTexture != null;
            slot.ReportFrame = frameIndex % _reportInterval == 0;
            slot.ViewInv = (Matrix4x4[])viewInverse.Clone();
            slot.Generation = _generation;
            slot.ConsumerToken = NextConsumerToken();
            slot.Completion = completion;

            _shader.SetInts(DepthSizeID, depthWidth, depthHeight);
            _shader.SetInts(ObservationGridID, gridX, gridY);
            _shader.SetInt(ObservationCountID, observationCount);
            _shader.SetInt(FrameIndexID, frameIndex);
            _shader.SetFloat(MotionQualityID, slot.MotionQuality);
            _shader.SetFloat(AngularSpeedID, slot.AngularSpeed);
            _shader.SetFloat(LinearSpeedID, slot.LinearSpeed);
            _shader.SetInt(PlatformFrameID, platformFrame);
            _shader.SetInt(RawDepthEyeID, 0);
            _shader.SetInt(HasRawDepthID, slot.HasRawDepth ? 1 : 0);
            _shader.SetInt(HasTemporalReasonID, slot.HasTemporalReason ? 1 : 0);
            _shader.SetMatrixArray(DepthProjInvID, projectionInverse);
            _shader.SetMatrixArray(DepthViewInvID, viewInverse);
            _buildObservations.Set(DepthTexID, depthTexture);
            _buildObservations.Set(RawDepthTexID,
                rawDepthTexture != null ? rawDepthTexture : depthTexture);
            _buildObservations.Set(NormalTexID, normalTexture);
            _buildObservations.Set(EdgeReasonTexID, edgeReasonTexture);
            _buildObservations.Set(TemporalReasonTexID,
                temporalReasonTexture != null ? temporalReasonTexture : edgeReasonTexture);
            _buildObservations.Set(ObservationsID, slot.Observations);
            _buildCorrespondences.Set(ObservationsID, slot.Observations);
            _buildCorrespondences.Set(CorrespondencesID, slot.Correspondences);

            if (slot.ReportFrame) _clearStats.DispatchFit(StatsCount, 1, 1);
            _buildObservations.DispatchFit(gridX, gridY, 1);
            _buildCorrespondences.DispatchFit(observationCount, 1, 1);
            AsyncGPUReadback.Request(slot.Correspondences,
                request => OnCorrespondences(request, slot));
            return true;
        }

        private void OnCorrespondences(AsyncGPUReadbackRequest request, FrameSlot slot)
        {
            if (_disposed || slot == null) return;
            if (slot.Generation != _generation)
            {
                slot.Pending = false;
                return;
            }
            Matrix4x4 correction = Matrix4x4.identity;
            int correspondenceCount = 0;
            int effectiveRank = 0;
            float translationMm = 0f;
            float rotationDeg = 0f;
            bool translationClamped = false;
            bool rotationClamped = false;
            if (!request.hasError)
            {
                var data = request.GetData<Correspondence>();
                correction = SolveCorrection(data, out correspondenceCount, out effectiveRank,
                    out translationMm, out rotationDeg,
                    out translationClamped, out rotationClamped);
            }

            _lastCorrespondences = correspondenceCount;
            _lastEffectiveRank = effectiveRank;
            _lastTranslationMm = translationMm;
            _lastRotationDeg = rotationDeg;

            bool consumerWillHold = slot.Completion != null;
            var decision = new FrameDecision(slot.FrameIndex, correction,
                correspondenceCount, effectiveRank, translationMm, rotationDeg,
                translationClamped, rotationClamped, !request.hasError,
                consumerWillHold ? slot.Observations : null,
                consumerWillHold ? slot.Correspondences : null,
                slot.GridX, slot.GridY, _pixelStride,
                consumerWillHold ? slot.SlotIndex : -1,
                consumerWillHold ? slot.ConsumerToken : 0);

            var completion = slot.Completion;
            slot.Completion = null;
            slot.ConsumerHeld = consumerWillHold;
            slot.Pending = false;
            // 有生产消费者时，候选事务必须等整帧准入结果。旧顺序在这里已经先
            // Update/Challenge/Retire，随后 VolumeIntegrator 才可能以撞顶/快角拒帧，
            // 导致“没写 TSDF 的帧却改了稳定候选”。无消费者的纯影子轨仍立即提交。
            if (!consumerWillHold)
                AdjudicateSlot(slot, correction, true);
            completion?.Invoke(decision);
        }

        /// <summary>
        /// 冷启动观察事务。候选账本尚不足以提供满秩校枪时，只允许当前可靠帧
        /// 以原始世界位姿养出胚胎/稳定 ID；不投反对票、不淘汰旧候选，也不因此
        /// 取得 TSDF 写入权。它打破“无候选→无配准→永远无候选”的启动自锁，
        /// 同时不把欠约束校枪结果施加到候选中心。
        /// </summary>
        internal bool BootstrapFrameDecision(FrameDecision decision)
        {
            int slotIndex = decision.FusionSlotIndex;
            if (slotIndex < 0 || slotIndex >= _slots.Length) return false;
            FrameSlot slot = _slots[slotIndex];
            if (!slot.ConsumerHeld || slot.ConsumerToken != decision.FusionSlotToken)
                return false;
            if (slot.Adjudicated)
            {
                _candidateTransactionDuplicateAttempts++;
                return slot.CandidateCommitted;
            }

            AdjudicateSlot(slot, Matrix4x4.identity, true, false);
            return true;
        }

        /// <summary>
        /// 整帧准入后的候选事务封口。无论接受还是拒绝，都先按当前稳定账本重建
        /// 严格逐点证据并记录黑匣子浪头；只有接受帧可以更新、反对和淘汰候选。
        /// 缓冲仍由 ReleaseFrameDecision 在 TSDF 消费结束后释放。
        /// </summary>
        internal bool AdjudicateFrameDecision(FrameDecision decision, bool commitCandidates)
        {
            int slotIndex = decision.FusionSlotIndex;
            if (slotIndex < 0 || slotIndex >= _slots.Length) return false;
            FrameSlot slot = _slots[slotIndex];
            if (!slot.ConsumerHeld || slot.ConsumerToken != decision.FusionSlotToken)
                return false;
            if (slot.Adjudicated)
            {
                _candidateTransactionDuplicateAttempts++;
                return slot.CandidateCommitted == commitCandidates;
            }

            AdjudicateSlot(slot, decision.Correction, commitCandidates);
            return true;
        }

        private void AdjudicateSlot(FrameSlot slot, Matrix4x4 correction,
            bool commitCandidates, bool includeOpposition = true)
        {
            if (slot == null || slot.Adjudicated) return;

            // GPU 回读异步完成；在最终裁决处恢复该槽自有的全部输入，禁止较新帧
            // 把运动、网格或姿态状态借给较旧帧。生产消费者按最老帧顺序调用本方法，
            // 因而候选账本也严格按扫描顺序提交。
            _shader.SetInt(ObservationCountID, slot.ObservationCount);
            _shader.SetInt(FrameIndexID, slot.FrameIndex);
            _shader.SetInts(ObservationGridID, slot.GridX, slot.GridY);
            _shader.SetInts(DepthSizeID, slot.DepthWidth, slot.DepthHeight);
            _shader.SetFloat(MotionQualityID, slot.MotionQuality);
            _shader.SetFloat(AngularSpeedID, slot.AngularSpeed);
            _shader.SetFloat(LinearSpeedID, slot.LinearSpeed);
            _shader.SetInt(HasRawDepthID, slot.HasRawDepth ? 1 : 0);
            _shader.SetInt(HasTemporalReasonID, slot.HasTemporalReason ? 1 : 0);
            _shader.SetMatrix(CorrectionID, correction);
            _shader.SetMatrixArray(DepthViewInvID, slot.ViewInv);

            _captureCourtWaves.Set(ObservationsID, slot.Observations);
            _captureCourtWaves.Set(CorrespondencesID, slot.Correspondences);
            _captureCourtWaves.DispatchFit(slot.ObservationCount, 1, 1);

            if (commitCandidates)
            {
                _updateCandidates.Set(ObservationsID, slot.Observations);
                _updateCandidates.DispatchFit(slot.ObservationCount, 1, 1);
                if (includeOpposition)
                {
                    int candidateCount = TableCapacity * CandidateCapacity;
                    _computeCandidateConsensus.DispatchFit(candidateCount, 1, 1);
                    _markCandidateSuccessions.DispatchFit(candidateCount, 1, 1);
                    _applyCandidateSuccessions.DispatchFit(candidateCount, 1, 1);
                    _rebuildProductionCorrespondences.Set(ObservationsID, slot.Observations);
                    _rebuildProductionCorrespondences.Set(CorrespondencesID, slot.Correspondences);
                    _rebuildProductionCorrespondences.DispatchFit(slot.ObservationCount, 1, 1);
                    _challengeCandidates.Set(ObservationsID, slot.Observations);
                    _challengeCandidates.DispatchFit(slot.ObservationCount, 1, 1);
                    _retireCandidates.DispatchFit(candidateCount, 1, 1);
                    _candidateTransactionsCommitted++;
                }
                else
                {
                    _candidateBootstrapTransactions++;
                }
            }
            else
            {
                _candidateTransactionsDiscarded++;
            }

            slot.CandidateCommitted = commitCandidates;
            slot.Adjudicated = true;
            if (slot.ReportFrame)
            {
                _countCensus.DispatchFit(TableCapacity, 1, 1);
                int reportGeneration = _generation;
                AsyncGPUReadback.Request(_stats,
                    statsRequest => OnStats(statsRequest, reportGeneration));
            }
        }

        private int NextConsumerToken()
        {
            _nextConsumerToken++;
            if (_nextConsumerToken == 0) _nextConsumerToken++;
            return _nextConsumerToken;
        }

        /// <summary>
        /// Releases the exact per-frame evidence buffers after the deferred TSDF
        /// consumer has either integrated or rejected that frame. A token prevents
        /// a late callback from releasing a slot that has already been recycled.
        /// </summary>
        internal void ReleaseFrameDecision(FrameDecision decision)
        {
            int slotIndex = decision.FusionSlotIndex;
            if (slotIndex < 0 || slotIndex >= _slots.Length) return;
            FrameSlot slot = _slots[slotIndex];
            if (!slot.ConsumerHeld || slot.ConsumerToken != decision.FusionSlotToken) return;
            slot.ConsumerHeld = false;
            slot.Adjudicated = false;
            slot.CandidateCommitted = false;
        }

        /// <summary>
        /// A 键封存契约：禁止新枪弹进入，并等待所有已经提交的 GPU 裁决回调完成。
        /// 候选与浪头缓冲随后原地保持；ConsumerHeld 只属于后续 TSDF 消费，
        /// 不影响本诊断层已经完成的裁决，故不把它纳入封存等待以免暂停后死锁。
        /// </summary>
        public IEnumerator SealCourtAsync(Action<bool> completed)
        {
            if (_disposed)
            {
                completed?.Invoke(false);
                yield break;
            }
            if (_courtSealed)
            {
                completed?.Invoke(true);
                yield break;
            }

            _courtSealPending = true;
            int sealGeneration = _generation;
            bool pending;
            do
            {
                pending = false;
                for (int i = 0; i < _slots.Length; i++)
                    pending |= _slots[i].Pending;
                if (pending) yield return null;
            }
            while (pending && !_disposed && sealGeneration == _generation);

            bool sealedSuccessfully = !_disposed && sealGeneration == _generation;
            _courtSealed = sealedSuccessfully;
            _courtSealPending = false;
            completed?.Invoke(sealedSuccessfully);
        }

        private Matrix4x4 SolveCorrection(Unity.Collections.NativeArray<Correspondence> data,
            out int correspondenceCount, out int effectiveRank,
            out float translationMm, out float rotationDeg,
            out bool translationClamped, out bool rotationClamped)
        {
            var normal = new double[6, 6];
            var rhs = new double[6];
            correspondenceCount = 0;
            for (int i = 0; i < data.Length; i++)
            {
                var item = data[i];
                uint evidenceFlags = (uint)Mathf.RoundToInt(item.SourceValid.w);
                if ((evidenceFlags & FusionAuthorityMask) != FusionAuthorityMask ||
                    item.TargetSigma.w <= 0f) continue;
                Vector3 source = new(item.SourceValid.x, item.SourceValid.y, item.SourceValid.z);
                Vector3 target = new(item.TargetSigma.x, item.TargetSigma.y, item.TargetSigma.z);
                Vector3 n = ((Vector3)item.NormalAngle).normalized;
                double residual = Vector3.Dot(source - target, n);
                double sigma = Math.Max(item.TargetSigma.w, 1e-5f);
                double standardized = Math.Abs(residual) / sigma;
                double robust = 1.0 / (1.0 + Math.Pow(standardized / 3.0, 2.0));
                double facing = Math.Max(Math.Cos(item.NormalAngle.w * Mathf.Deg2Rad), 0.1);
                double weight = robust * facing / Math.Max(sigma * sigma, 1e-8);
                Vector3 cross = Vector3.Cross(source, n);
                double[] jacobian = { cross.x, cross.y, cross.z, n.x, n.y, n.z };
                for (int row = 0; row < 6; row++)
                {
                    rhs[row] += -weight * jacobian[row] * residual;
                    for (int column = 0; column < 6; column++)
                        normal[row, column] += weight * jacobian[row] * jacobian[column];
                }
                correspondenceCount++;
            }

            effectiveRank = 0;
            translationMm = 0f;
            rotationDeg = 0f;
            translationClamped = false;
            rotationClamped = false;
            if (correspondenceCount < 64) return Matrix4x4.identity;
            double maxDiagonal = 0.0;
            for (int i = 0; i < 6; i++) maxDiagonal = Math.Max(maxDiagonal, normal[i, i]);
            if (maxDiagonal <= 1e-9) return Matrix4x4.identity;
            for (int i = 0; i < 6; i++)
                if (normal[i, i] > maxDiagonal * 1e-4) effectiveRank++;
            if (effectiveRank < 3) return Matrix4x4.identity;

            double damping = maxDiagonal * 1e-4;
            for (int i = 0; i < 6; i++) normal[i, i] += damping;
            if (!SolveLinearSystem(normal, rhs, out double[] delta)) return Matrix4x4.identity;
            Vector3 rotationVector = new((float)delta[0], (float)delta[1], (float)delta[2]);
            Vector3 translation = new((float)delta[3], (float)delta[4], (float)delta[5]);
            float rotationLength = rotationVector.magnitude;
            float rotationLimit = 1.5f * Mathf.Deg2Rad;
            if (rotationLength > rotationLimit)
            {
                rotationClamped = true;
                rotationVector *= rotationLimit / rotationLength;
            }
            if (translation.magnitude > 0.03f)
            {
                translationClamped = true;
                translation = translation.normalized * 0.03f;
            }

            rotationDeg = rotationVector.magnitude * Mathf.Rad2Deg;
            translationMm = translation.magnitude * 1000f;
            Quaternion rotation = rotationVector.sqrMagnitude > 1e-12f
                ? Quaternion.AngleAxis(rotationDeg, rotationVector.normalized)
                : Quaternion.identity;
            return Matrix4x4.TRS(translation, rotation, Vector3.one);
        }

        private static bool SolveLinearSystem(double[,] matrix, double[] vector, out double[] result)
        {
            const int size = 6;
            var augmented = new double[size, size + 1];
            for (int row = 0; row < size; row++)
            {
                for (int column = 0; column < size; column++)
                    augmented[row, column] = matrix[row, column];
                augmented[row, size] = vector[row];
            }
            for (int pivot = 0; pivot < size; pivot++)
            {
                int best = pivot;
                for (int row = pivot + 1; row < size; row++)
                    if (Math.Abs(augmented[row, pivot]) > Math.Abs(augmented[best, pivot])) best = row;
                if (Math.Abs(augmented[best, pivot]) < 1e-12)
                {
                    result = null;
                    return false;
                }
                if (best != pivot)
                    for (int column = pivot; column <= size; column++)
                    {
                        double temporary = augmented[pivot, column];
                        augmented[pivot, column] = augmented[best, column];
                        augmented[best, column] = temporary;
                    }
                double divisor = augmented[pivot, pivot];
                for (int column = pivot; column <= size; column++) augmented[pivot, column] /= divisor;
                for (int row = 0; row < size; row++)
                {
                    if (row == pivot) continue;
                    double factor = augmented[row, pivot];
                    for (int column = pivot; column <= size; column++)
                        augmented[row, column] -= factor * augmented[pivot, column];
                }
            }
            result = new double[size];
            for (int i = 0; i < size; i++) result[i] = augmented[i, size];
            return true;
        }

        private void OnStats(AsyncGPUReadbackRequest request, int reportGeneration)
        {
            if (_disposed || reportGeneration != _generation || request.hasError) return;
            var data = request.GetData<uint>();
            if (data.Length < StatsCount) return;
            _lastValidObservations = data[0];
            _lastBirths = data[2];
            _lastUpdates = data[3];
            _lastPromotions = data[4];
            _lastCapacityAbstentions = data[5];
            _lastContentions = data[6];
            _lastActiveCandidates = data[8];
            _lastStableCandidates = data[9];
            _lastMultiStableCells = data[10];
            _lastStaleStableCandidates = data[11];
            _lastDualAgree = data[13];
            _lastDualDisagree = data[14];
            _lastChallengeVotes = data[15];
            _lastContradictionRetirements = data[16];
            _lastEmbryoExpirations = data[17];
            _lastStableIdsAssigned = data[18];
            _lastRawUnavailable = data[19];
            _lastStableIdsAlive = data[20];
            _lastCandidateSuccessions = data[24];
            _hasReport = true;
        }

        public string GetCompact()
        {
            if (!_hasReport) return "影子预热中";
            return $"校枪{_lastTranslationMm:F1}mm/{_lastRotationDeg:F2}度 " +
                   $"配{_lastCorrespondences}秩{_lastEffectiveRank} " +
                   $"胶{_lastActiveCandidates}稳{_lastStableCandidates}" +
                   $"多{_lastMultiStableCells}陈{_lastStaleStableCandidates} " +
                   $"双{_lastDualAgree}/{_lastDualDisagree}无{_lastRawUnavailable} " +
                   $"反{_lastChallengeVotes}淘{_lastContradictionRetirements}+{_lastEmbryoExpirations} " +
                   $"换{_lastCandidateSuccessions} " +
                   $"ID{_lastStableIdsAlive}生{_lastStableIdsAssigned} " +
                   $"本帧观{_lastValidObservations}生{_lastBirths}更{_lastUpdates}" +
                   $"晋{_lastPromotions}满{_lastCapacityAbstentions}锁{_lastContentions}";
        }

        /// <summary>
        /// 冻结前导出数据层候选黑匣子。等待在途枪胶帧自然排空后，一次性回读
        /// 存活候选、累计证词与淘汰环；只读，不改变候选或生产 TSDF。
        /// </summary>
        public IEnumerator ExportAuditAsync(string reason, Action<string> completed)
        {
            if (_disposed || _auditExportPending)
            {
                completed?.Invoke("");
                yield break;
            }

            _auditExportPending = true;
            int exportGeneration = _generation;
            bool pending;
            do
            {
                pending = false;
                for (int i = 0; i < _slots.Length; i++)
                    pending |= _slots[i].Pending;
                if (pending) yield return null;
            }
            while (pending && !_disposed && exportGeneration == _generation);

            if (_disposed || exportGeneration != _generation)
            {
                _auditExportPending = false;
                completed?.Invoke("");
                yield break;
            }

            var cellKeysRequest = AsyncGPUReadback.Request(_cellKeys);
            var cellStateRequest = AsyncGPUReadback.Request(_cellState);
            var centerRequest = AsyncGPUReadback.Request(_candidateCenterSigma);
            var normalRequest = AsyncGPUReadback.Request(_candidateNormalSupport);
            var viewRequest = AsyncGPUReadback.Request(_candidateFirstViewAngle);
            var metaRequest = AsyncGPUReadback.Request(_candidateMeta);
            var evidenceRequest = AsyncGPUReadback.Request(_candidateEvidence);
            var nextIdRequest = AsyncGPUReadback.Request(_nextStableId);
            var totalsRequest = AsyncGPUReadback.Request(_auditTotals);
            var retireCounterRequest = AsyncGPUReadback.Request(_retireAuditCounters);
            var retireMetaRequest = AsyncGPUReadback.Request(_retireAuditMeta);
            var retireGeometryRequest = AsyncGPUReadback.Request(_retireAuditGeometry);
            var retireEvidenceRequest = AsyncGPUReadback.Request(_retireAuditEvidence);
            var retireTimingRequest = AsyncGPUReadback.Request(_retireAuditTiming);
            var courtPeakRequest = AsyncGPUReadback.Request(_courtWavePeakBits);
            var courtPositionRequest = AsyncGPUReadback.Request(_courtWavePositionResidual);
            var courtTargetRequest = AsyncGPUReadback.Request(_courtWaveTargetSigma);
            var courtMetaRequest = AsyncGPUReadback.Request(_courtWaveMeta);
            var courtSourceReasonRequest = AsyncGPUReadback.Request(_courtWaveSourceReason);
            var courtRawRequest = AsyncGPUReadback.Request(_courtWaveRawPositionDelta);
            var courtMotionViewRequest = AsyncGPUReadback.Request(_courtWaveMotionView);
            var courtCountersRequest = AsyncGPUReadback.Request(_courtWaveCounters);

            while (!cellKeysRequest.done || !cellStateRequest.done ||
                   !centerRequest.done || !normalRequest.done || !viewRequest.done ||
                   !metaRequest.done || !evidenceRequest.done || !nextIdRequest.done ||
                   !totalsRequest.done || !retireCounterRequest.done ||
                   !retireMetaRequest.done || !retireGeometryRequest.done ||
                   !retireEvidenceRequest.done || !retireTimingRequest.done ||
                   !courtPeakRequest.done || !courtPositionRequest.done ||
                   !courtTargetRequest.done || !courtMetaRequest.done ||
                   !courtSourceReasonRequest.done || !courtRawRequest.done ||
                   !courtMotionViewRequest.done ||
                   !courtCountersRequest.done)
                yield return null;

            if (cellKeysRequest.hasError || cellStateRequest.hasError ||
                centerRequest.hasError || normalRequest.hasError || viewRequest.hasError ||
                metaRequest.hasError || evidenceRequest.hasError || nextIdRequest.hasError ||
                totalsRequest.hasError || retireCounterRequest.hasError ||
                retireMetaRequest.hasError || retireGeometryRequest.hasError ||
                retireEvidenceRequest.hasError || retireTimingRequest.hasError ||
                courtPeakRequest.hasError || courtPositionRequest.hasError ||
                courtTargetRequest.hasError || courtMetaRequest.hasError ||
                courtSourceReasonRequest.hasError || courtRawRequest.hasError ||
                courtMotionViewRequest.hasError ||
                courtCountersRequest.hasError)
            {
                _auditExportPending = false;
                Logger.Error("枪胶候选黑匣子GPU快照失败：至少一个缓冲回读报错");
                completed?.Invoke("");
                yield break;
            }
            if (_disposed || exportGeneration != _generation)
            {
                _auditExportPending = false;
                completed?.Invoke("");
                yield break;
            }

            var cellKeys = cellKeysRequest.GetData<int4>();
            var cellStates = cellStateRequest.GetData<uint>();
            var centers = centerRequest.GetData<float4>();
            var normals = normalRequest.GetData<float4>();
            var views = viewRequest.GetData<float4>();
            var metas = metaRequest.GetData<uint4>();
            var evidence = evidenceRequest.GetData<uint4>();
            var nextId = nextIdRequest.GetData<uint>();
            var totals = totalsRequest.GetData<uint>();
            var retireCounters = retireCounterRequest.GetData<uint>();
            var retireMeta = retireMetaRequest.GetData<uint4>();
            var retireGeometry = retireGeometryRequest.GetData<float4>();
            var retireEvidence = retireEvidenceRequest.GetData<uint4>();
            var retireTiming = retireTimingRequest.GetData<uint4>();
            var courtPeaks = courtPeakRequest.GetData<uint>();
            var courtPositions = courtPositionRequest.GetData<float4>();
            var courtTargets = courtTargetRequest.GetData<float4>();
            var courtMetas = courtMetaRequest.GetData<uint4>();
            var courtSourceReasons = courtSourceReasonRequest.GetData<uint4>();
            var courtRaw = courtRawRequest.GetData<float4>();
            var courtMotionView = courtMotionViewRequest.GetData<float4>();
            var courtCounters = courtCountersRequest.GetData<uint>();

            DateTime capturedUtc = DateTime.UtcNow;
            string stamp = capturedUtc.ToString("yyyyMMdd_HHmmss_fff",
                CultureInfo.InvariantCulture);
            string directory = Path.Combine(Application.persistentDataPath,
                "ScanCoverDiagnostics", "gungel_candidate_audit", stamp);
            string candidatesPath = Path.Combine(directory, "candidates.csv");
            string retirementsPath = Path.Combine(directory, "retirements.csv");
            string courtWavesPath = Path.Combine(directory, "court_waves.csv");
            string summaryPath = Path.Combine(directory, "candidate_summary.json");

            int activeCandidates = 0;
            int stableCandidates = 0;
            int embryoCandidates = 0;
            int activeCells = 0;
            int multiCandidateCells = 0;
            int stableIdMissing = 0;
            ulong aliveDualSupport = 0;
            ulong aliveOpposition = 0;
            uint latestFrame = 0u;
            bool haveBounds = false;
            Vector3 boundsMin = Vector3.zero;
            Vector3 boundsMax = Vector3.zero;
            var candidatesCsv = new StringBuilder(256 * 1024);
            candidatesCsv.AppendLine("candidate_index,cell_slot,layer_slot,cell_x,cell_y,cell_z,stable_id,state,observation_count,last_seen_frame,center_x_m,center_y_m,center_z_m,sigma_mm,normal_x,normal_y,normal_z,effective_support,dual_agree_support,opposition_votes,last_challenge_frame,first_view_x,first_view_y,first_view_z,view_spread_deg");

            for (int cell = 0; cell < TableCapacity; cell++)
            {
                int cellActive = 0;
                int4 key = cellStates[cell] == 2u
                    ? cellKeys[cell] : new int4(0, 0, 0, 0);
                for (int layer = 0; layer < CandidateCapacity; layer++)
                {
                    int index = cell * CandidateCapacity + layer;
                    uint4 meta = metas[index];
                    if (meta.x == 0u) continue;
                    uint4 itemEvidence = evidence[index];
                    float4 center = centers[index];
                    float4 normal = normals[index];
                    float4 view = views[index];
                    cellActive++;
                    activeCandidates++;
                    latestFrame = Math.Max(latestFrame, meta.z);
                    aliveDualSupport += itemEvidence.x;
                    aliveOpposition += itemEvidence.y;
                    if (meta.y != 0u)
                    {
                        stableCandidates++;
                        if (itemEvidence.z == 0u) stableIdMissing++;
                    }
                    else embryoCandidates++;

                    Vector3 position = new(center.x, center.y, center.z);
                    if (!haveBounds)
                    {
                        boundsMin = boundsMax = position;
                        haveBounds = true;
                    }
                    else
                    {
                        boundsMin = Vector3.Min(boundsMin, position);
                        boundsMax = Vector3.Max(boundsMax, position);
                    }

                    candidatesCsv.Append(index).Append(',').Append(cell).Append(',').Append(layer).Append(',')
                        .Append(key.x).Append(',').Append(key.y).Append(',').Append(key.z).Append(',')
                        .Append(itemEvidence.z).Append(',').Append(meta.y != 0u ? "stable" : "embryo").Append(',')
                        .Append(meta.x).Append(',').Append(meta.z).Append(',')
                        .Append(FloatText(center.x)).Append(',').Append(FloatText(center.y)).Append(',')
                        .Append(FloatText(center.z)).Append(',')
                        .Append(FloatText(Mathf.Sqrt(Mathf.Max(center.w, 0f)) * 1000f)).Append(',')
                        .Append(FloatText(normal.x)).Append(',').Append(FloatText(normal.y)).Append(',')
                        .Append(FloatText(normal.z)).Append(',').Append(FloatText(normal.w)).Append(',')
                        .Append(itemEvidence.x).Append(',').Append(itemEvidence.y).Append(',')
                        .Append(itemEvidence.w == 0u ? 0u : itemEvidence.w - 1u).Append(',')
                        .Append(FloatText(view.x)).Append(',').Append(FloatText(view.y)).Append(',')
                        .Append(FloatText(view.z)).Append(',').Append(FloatText(view.w)).AppendLine();
                }
                if (cellActive > 0)
                {
                    activeCells++;
                    if (cellActive > 1) multiCandidateCells++;
                }
            }

            uint retireTotal = retireCounters.Length > 0 ? retireCounters[0] : 0u;
            int retireRetained = (int)Math.Min(retireTotal, (uint)RetireAuditCapacity);
            uint firstRetireSequence = retireTotal - (uint)retireRetained;
            int contradictionRetained = 0;
            int embryoRetained = 0;
            int successionRetained = 0;
            int retiredStable = 0;
            var retirementsCsv = new StringBuilder(Math.Max(4096, retireRetained * 160));
            retirementsCsv.AppendLine("sequence,candidate_index,cell_slot,layer_slot,cell_x,cell_y,cell_z,stable_id,reason,retired_frame,last_seen_frame,last_challenge_frame,successor_index,successor_previous_stable_id,center_x_m,center_y_m,center_z_m,sigma_mm,observation_count,was_stable,dual_agree_support,opposition_votes");
            for (uint sequence = firstRetireSequence; sequence < retireTotal; sequence++)
            {
                int ringIndex = (int)(sequence % (uint)RetireAuditCapacity);
                uint4 itemMeta = retireMeta[ringIndex];
                float4 geometry = retireGeometry[ringIndex];
                uint4 itemEvidence = retireEvidence[ringIndex];
                uint4 timing = retireTiming[ringIndex];
                int candidateIndex = (int)itemMeta.x;
                int cell = candidateIndex / CandidateCapacity;
                int layer = candidateIndex % CandidateCapacity;
                int4 key = cell >= 0 && cell < cellKeys.Length && cellStates[cell] == 2u
                    ? cellKeys[cell] : new int4(0, 0, 0, 0);
                string retireReason = itemMeta.z == 1u
                    ? "contradicted" : (itemMeta.z == 2u
                    ? "embryo_expired" : "stable_successor");
                if (itemMeta.z == 1u) contradictionRetained++;
                else if (itemMeta.z == 2u) embryoRetained++;
                else successionRetained++;
                if (itemEvidence.y != 0u) retiredStable++;
                latestFrame = Math.Max(latestFrame, itemMeta.w);
                retirementsCsv.Append(sequence).Append(',').Append(candidateIndex).Append(',')
                    .Append(cell).Append(',').Append(layer).Append(',')
                    .Append(key.x).Append(',').Append(key.y).Append(',').Append(key.z).Append(',')
                    .Append(itemMeta.y).Append(',').Append(retireReason).Append(',')
                    .Append(itemMeta.w).Append(',').Append(timing.x).Append(',')
                    .Append(timing.y == 0u ? 0u : timing.y - 1u).Append(',')
                    .Append(itemMeta.z == 3u ? timing.z : 0u).Append(',')
                    .Append(itemMeta.z == 3u ? timing.w : 0u).Append(',')
                    .Append(FloatText(geometry.x)).Append(',').Append(FloatText(geometry.y)).Append(',')
                    .Append(FloatText(geometry.z)).Append(',')
                    .Append(FloatText(Mathf.Sqrt(Mathf.Max(geometry.w, 0f)) * 1000f)).Append(',')
                    .Append(itemEvidence.x).Append(',').Append(itemEvidence.y).Append(',')
                    .Append(itemEvidence.z).Append(',').Append(itemEvidence.w).AppendLine();
            }

            int retainedCourtWaves = 0;
            int acceptedCourtWaves = 0;
            int pendingCourtWaves = 0;
            int rejectedCourtWaves = 0;
            var courtWavesCsv = new StringBuilder(64 * 1024);
            courtWavesCsv.AppendLine("stable_id,source_frame,platform_frame,pixel_x,pixel_y,view_radius,angular_deg_s,linear_m_s,motion_quality,edge_reason_bits,edge_reason_labels,temporal_reason,temporal_reason_label,adjudication,evidence_flags,accepted,raw_available,dual_agree,stable_match,stable_dual_mature,unopposed,raw_x_m,raw_y_m,raw_z_m,raw_processed_delta_mm,observation_x_m,observation_y_m,observation_z_m,candidate_x_m,candidate_y_m,candidate_z_m,normal_residual_mm,combined_sigma_mm,severity_mm");
            int courtLength = Math.Min(courtPeaks.Length,
                Math.Min(courtPositions.Length, Math.Min(courtTargets.Length,
                Math.Min(courtMetas.Length, Math.Min(courtSourceReasons.Length,
                Math.Min(courtRaw.Length, courtMotionView.Length))))));
            for (int i = 0; i < courtLength; i++)
            {
                uint4 waveMeta = courtMetas[i];
                uint peakBits = courtPeaks[i];
                if (waveMeta.x == 0u || peakBits == 0u) continue;
                retainedCourtWaves++;
                if (waveMeta.z == 1u) acceptedCourtWaves++;
                else if (waveMeta.z == 2u || waveMeta.z == 3u) pendingCourtWaves++;
                else rejectedCourtWaves++;
                uint flags = waveMeta.w;
                float4 observation = courtPositions[i];
                float4 target = courtTargets[i];
                uint4 sourceReason = courtSourceReasons[i];
                uint packedPixel = sourceReason.x;
                uint pixelX = packedPixel & 0xffffu;
                uint pixelY = packedPixel >> 16;
                float4 raw = courtRaw[i];
                float4 motionView = courtMotionView[i];
                courtWavesCsv.Append(waveMeta.x).Append(',').Append(waveMeta.y).Append(',')
                    .Append(sourceReason.w).Append(',').Append(pixelX).Append(',')
                    .Append(pixelY).Append(',').Append(FloatText(motionView.w)).Append(',')
                    .Append(FloatText(motionView.x)).Append(',')
                    .Append(FloatText(motionView.y)).Append(',')
                    .Append(FloatText(motionView.z)).Append(',')
                    .Append(sourceReason.y).Append(',').Append(EdgeReasonLabels(sourceReason.y)).Append(',')
                    .Append(sourceReason.z).Append(',').Append(TemporalReasonName(sourceReason.z)).Append(',')
                    .Append(CourtVerdictName(waveMeta.z)).Append(',').Append(flags).Append(',')
                    .Append(waveMeta.z == 1u ? 1 : 0).Append(',')
                    .Append((flags & 2u) != 0u ? 1 : 0).Append(',')
                    .Append((flags & 4u) != 0u ? 1 : 0).Append(',')
                    .Append((flags & 8u) != 0u ? 1 : 0).Append(',')
                    .Append((flags & 16u) != 0u ? 1 : 0).Append(',')
                    .Append((flags & 32u) != 0u ? 1 : 0).Append(',')
                    .Append(FloatText(raw.x)).Append(',')
                    .Append(FloatText(raw.y)).Append(',')
                    .Append(FloatText(raw.z)).Append(',')
                    .Append(FloatText(raw.w * 1000f)).Append(',')
                    .Append(FloatText(observation.x)).Append(',')
                    .Append(FloatText(observation.y)).Append(',')
                    .Append(FloatText(observation.z)).Append(',')
                    .Append(FloatText(target.x)).Append(',')
                    .Append(FloatText(target.y)).Append(',')
                    .Append(FloatText(target.z)).Append(',')
                    .Append(FloatText(observation.w * 1000f)).Append(',')
                    .Append(FloatText(target.w * 1000f)).Append(',')
                    .Append(FloatText(math.asfloat(peakBits) * 1000f)).AppendLine();
            }

            uint courtUniqueCount = courtCounters.Length > 0 ? courtCounters[0] : 0u;
            uint courtPeakUpdates = courtCounters.Length > 1 ? courtCounters[1] : 0u;
            uint courtOverflow = courtCounters.Length > 2 ? courtCounters[2] : 0u;

            uint totalValid = AuditTotal(totals, 0);
            uint totalDualAgree = AuditTotal(totals, 1);
            uint totalDualDisagree = AuditTotal(totals, 2);
            uint totalRawUnavailable = AuditTotal(totals, 3);
            uint totalChallengeVotes = AuditTotal(totals, 4);
            uint totalContradictionRetired = AuditTotal(totals, 5);
            uint totalEmbryoExpired = AuditTotal(totals, 6);
            uint totalStableIdsAssigned = AuditTotal(totals, 7);
            uint totalCandidateSuccessions = AuditTotal(totals, 8);
            var summary = new StringBuilder(4096);
            summary.AppendLine("{")
                .Append("  \"schema\": \"scancover.gungel_candidate_audit.v5\",\n")
                .Append("  \"reason\": \"").Append(JsonEscape(reason)).Append("\",\n")
                .Append("  \"captured_utc\": \"").Append(capturedUtc.ToString("O", CultureInfo.InvariantCulture)).Append("\",\n")
                .Append("  \"latest_frame\": ").Append(latestFrame).Append(",\n")
                .Append("  \"cell_size_m\": ").Append(FloatText(_cellSize)).Append(",\n")
                .Append("  \"semantics\": {\n")
                .Append("    \"coordinates\": \"world_meters\",\n")
                .Append("    \"cumulative_dual_disagree_scope\": \"global_counter\",\n")
                .Append("    \"court_wave_scope\": \"nearest_stable_id_before_candidate_update\",\n")
                .Append("    \"court_wave_provenance\": \"latched_pixel_platform_frame_raw_processed_motion_edge_temporal\",\n")
                .Append("    \"edge_reason_bits\": \"bit0 span_jump; bit1 gap; bit2 skirt; bit3 plane_trusted; bit4 grazing; bit5 dual_cluster; bit6 cross_eye; bit7 rejected_edge; bit8 grazing_span; bit9 grazing_plane_supported; bit10 grazing_rescued\",\n")
                .Append("    \"temporal_reason_codes\": \"0 unavailable; 1 first_frame; 2 current_invalid; 3 previous_fov_miss; 4 history_invalid; 5 changed; 6 stable\",\n")
                .Append("    \"fusion_residual_limit\": \"max(0.012m,2.5*combined_sigma)\",\n")
                .Append("    \"fusion_contract\": \"court_wave_must_not_hold_fusion_authority\",\n")
                .Append("    \"candidate_transaction_contract\": \"frame_adjudication_before_candidate_mutation; underconstrained_bootstrap_updates_only; other_rejected_frames_diagnostic_only\",\n")
                .Append("    \"stable_update_contract\": \"stable_candidate_accepts_ema_only_inside_fusion_residual_limit; outside_band_may_only_grow_independent_challenger\",\n")
                .Append("    \"retire_reason_1\": \"three_consecutive_reliable_free_space_votes\",\n")
                .Append("    \"retire_reason_2\": \"unpromoted_embryo_expired\",\n")
                .Append("    \"retire_reason_3\": \"local_coplanar_consensus_successor_inherits_stable_id\",\n")
                .Append("    \"succession_contract\": \"current_dual_mature_candidate; stale_incumbent; same_normal_and_local_footprint; challenger_consensus_at_least_3_and_margin_2\"\n")
                .Append("  },\n")
                .Append("  \"candidate_snapshot\": {\n")
                .Append("    \"active_cells\": ").Append(activeCells).Append(",\n")
                .Append("    \"multi_candidate_cells\": ").Append(multiCandidateCells).Append(",\n")
                .Append("    \"active_candidates\": ").Append(activeCandidates).Append(",\n")
                .Append("    \"stable_candidates\": ").Append(stableCandidates).Append(",\n")
                .Append("    \"embryo_candidates\": ").Append(embryoCandidates).Append(",\n")
                .Append("    \"stable_id_missing\": ").Append(stableIdMissing).Append(",\n")
                .Append("    \"next_stable_id\": ").Append(nextId.Length > 0 ? nextId[0] : 0u).Append(",\n")
                .Append("    \"alive_dual_agree_support\": ").Append(aliveDualSupport).Append(",\n")
                .Append("    \"alive_opposition_votes\": ").Append(aliveOpposition).Append("\n")
                .Append("  },\n")
                .Append("  \"cumulative_evidence\": {\n")
                .Append("    \"valid_observations\": ").Append(totalValid).Append(",\n")
                .Append("    \"dual_agree\": ").Append(totalDualAgree).Append(",\n")
                .Append("    \"dual_disagree\": ").Append(totalDualDisagree).Append(",\n")
                .Append("    \"raw_unavailable\": ").Append(totalRawUnavailable).Append(",\n")
                .Append("    \"challenge_votes\": ").Append(totalChallengeVotes).Append(",\n")
                .Append("    \"contradiction_retirements\": ").Append(totalContradictionRetired).Append(",\n")
                .Append("    \"embryo_expirations\": ").Append(totalEmbryoExpired).Append(",\n")
                .Append("    \"stable_ids_assigned\": ").Append(totalStableIdsAssigned).Append(",\n")
                .Append("    \"candidate_successions\": ").Append(totalCandidateSuccessions).Append("\n")
                .Append("  },\n")
                .Append("  \"candidate_transactions\": {\n")
                .Append("    \"committed_frames\": ").Append(_candidateTransactionsCommitted).Append(",\n")
                .Append("    \"bootstrap_observation_frames\": ").Append(_candidateBootstrapTransactions).Append(",\n")
                .Append("    \"discarded_frames\": ").Append(_candidateTransactionsDiscarded).Append(",\n")
                .Append("    \"duplicate_or_conflicting_attempts\": ").Append(_candidateTransactionDuplicateAttempts).Append(",\n")
                .Append("    \"rejected_frame_full_candidate_mutations\": 0\n")
                .Append("  },\n")
                .Append("  \"retirement_ring\": {\n")
                .Append("    \"total_events\": ").Append(retireTotal).Append(",\n")
                .Append("    \"retained_events\": ").Append(retireRetained).Append(",\n")
                .Append("    \"overwritten_events\": ").Append(Math.Max(0L, (long)retireTotal - retireRetained)).Append(",\n")
                .Append("    \"retained_contradicted\": ").Append(contradictionRetained).Append(",\n")
                .Append("    \"retained_embryo_expired\": ").Append(embryoRetained).Append(",\n")
                .Append("    \"retained_stable_successions\": ").Append(successionRetained).Append(",\n")
                .Append("    \"retained_stable_candidates\": ").Append(retiredStable).Append("\n")
                .Append("  },\n")
                .Append("  \"court_waves\": {\n")
                .Append("    \"meaning\": \"worst_latched_wave_per_stable_id_before_candidate_update\",\n")
                .Append("    \"unique_count_gpu\": ").Append(courtUniqueCount).Append(",\n")
                .Append("    \"retained_rows\": ").Append(retainedCourtWaves).Append(",\n")
                .Append("    \"peak_updates\": ").Append(courtPeakUpdates).Append(",\n")
                .Append("    \"capacity_overflow\": ").Append(courtOverflow).Append(",\n")
                .Append("    \"accepted\": ").Append(acceptedCourtWaves).Append(",\n")
                .Append("    \"fusion_contract_violations\": ").Append(acceptedCourtWaves).Append(",\n")
                .Append("    \"pending\": ").Append(pendingCourtWaves).Append(",\n")
                .Append("    \"rejected\": ").Append(rejectedCourtWaves).Append("\n")
                .Append("  },\n")
                .Append("  \"world_bounds_m\": {\n")
                .Append("    \"valid\": ").Append(haveBounds ? "true" : "false").Append(",\n")
                .Append("    \"min\": [").Append(FloatText(boundsMin.x)).Append(',').Append(FloatText(boundsMin.y)).Append(',').Append(FloatText(boundsMin.z)).Append("],\n")
                .Append("    \"max\": [").Append(FloatText(boundsMax.x)).Append(',').Append(FloatText(boundsMax.y)).Append(',').Append(FloatText(boundsMax.z)).Append("]\n")
                .Append("  },\n")
                .Append("  \"files\": {\"candidates\": \"candidates.csv\", \"retirements\": \"retirements.csv\", \"court_waves\": \"court_waves.csv\"}\n")
                .AppendLine("}");

            try
            {
                Directory.CreateDirectory(directory);
                File.WriteAllText(candidatesPath, candidatesCsv.ToString(), Encoding.UTF8);
                File.WriteAllText(retirementsPath, retirementsCsv.ToString(), Encoding.UTF8);
                File.WriteAllText(courtWavesPath, courtWavesCsv.ToString(), Encoding.UTF8);
                File.WriteAllText(summaryPath, summary.ToString(), Encoding.UTF8);
                _auditExportPending = false;
                completed?.Invoke(summaryPath);
            }
            catch (Exception ex)
            {
                _auditExportPending = false;
                Logger.Error($"枪胶候选黑匣子写盘失败: {ex.Message}");
                completed?.Invoke("");
            }
        }

        private static uint AuditTotal(Unity.Collections.NativeArray<uint> totals, int index)
        {
            return index >= 0 && index < totals.Length ? totals[index] : 0u;
        }

        private static string CourtVerdictName(uint verdict)
        {
            return verdict switch
            {
                1u => "accepted",
                2u => "pending_stable_mismatch",
                3u => "pending_dual_immature",
                4u => "rejected_raw_missing",
                5u => "rejected_dual_disagree",
                6u => "rejected_opposed",
                _ => "unknown"
            };
        }

        private static string TemporalReasonName(uint reason)
        {
            return reason switch
            {
                1u => "first_frame",
                2u => "current_invalid",
                3u => "previous_fov_miss",
                4u => "history_invalid",
                5u => "changed",
                6u => "stable",
                _ => "unavailable"
            };
        }

        private static string EdgeReasonLabels(uint bits)
        {
            if (bits == 0u) return "none";
            string[] names =
            {
                "span_jump", "gap", "skirt", "plane_trusted", "grazing",
                "dual_cluster", "cross_eye", "rejected_edge", "grazing_span",
                "grazing_plane_supported", "grazing_rescued"
            };
            var result = new StringBuilder(96);
            for (int bit = 0; bit < names.Length; bit++)
            {
                if ((bits & (1u << bit)) == 0u) continue;
                if (result.Length > 0) result.Append('|');
                result.Append(names[bit]);
            }
            return result.ToString();
        }

        private static string FloatText(float value)
        {
            return value.ToString("R", CultureInfo.InvariantCulture);
        }

        private static string JsonEscape(string value)
        {
            return (value ?? string.Empty).Replace("\\", "\\\\").Replace("\"", "\\\"")
                .Replace("\r", "\\r").Replace("\n", "\\n");
        }

        public void Clear()
        {
            if (_disposed) return;
            _generation++;
            _courtSealPending = false;
            _courtSealed = false;
            SetSharedConstants();
            _clearCells.DispatchFit(Mathf.Max(TableCapacity * CandidateCapacity,
                CourtWaveCapacity), 1, 1);
            _clearStats.DispatchFit(StatsCount, 1, 1);
            _candidateTransactionsCommitted = 0u;
            _candidateBootstrapTransactions = 0u;
            _candidateTransactionsDiscarded = 0u;
            _candidateTransactionDuplicateAttempts = 0u;
            _hasReport = false;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var slot in _slots)
            {
                slot.Observations?.Release();
                slot.Correspondences?.Release();
                slot.Observations = null;
                slot.Correspondences = null;
                slot.Completion = null;
                slot.Pending = false;
                slot.ConsumerHeld = false;
                slot.Adjudicated = false;
                slot.CandidateCommitted = false;
            }
            _cellKeys?.Release();
            _cellState?.Release();
            _cellLock?.Release();
            _candidateCenterSigma?.Release();
            _candidateNormalSupport?.Release();
            _candidateFirstViewAngle?.Release();
            _candidateMeta?.Release();
            _candidateEvidence?.Release();
            _candidateSuccessor?.Release();
            _candidatePredecessor?.Release();
            _candidateConsensus?.Release();
            _nextStableId?.Release();
            _stats?.Release();
            _auditTotals?.Release();
            _retireAuditCounters?.Release();
            _retireAuditMeta?.Release();
            _retireAuditGeometry?.Release();
            _retireAuditEvidence?.Release();
            _retireAuditTiming?.Release();
            _courtWaveLock?.Release();
            _courtWavePeakBits?.Release();
            _courtWavePositionResidual?.Release();
            _courtWaveTargetSigma?.Release();
            _courtWaveMeta?.Release();
            _courtWaveSourceReason?.Release();
            _courtWaveRawPositionDelta?.Release();
            _courtWaveMotionView?.Release();
            _courtWaveCounters?.Release();
        }
    }
}
