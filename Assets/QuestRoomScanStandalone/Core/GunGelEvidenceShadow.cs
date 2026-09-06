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
            internal readonly ComputeBuffer FusionPreTransactionCorrespondenceIdentity;
            internal readonly ComputeBuffer FusionCorrespondenceIdentity;
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
                ComputeBuffer fusionPreTransactionCorrespondenceIdentity,
                ComputeBuffer fusionCorrespondenceIdentity,
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
                FusionPreTransactionCorrespondenceIdentity =
                    fusionPreTransactionCorrespondenceIdentity;
                FusionCorrespondenceIdentity = fusionCorrespondenceIdentity;
                FusionObservationGridX = fusionObservationGridX;
                FusionObservationGridY = fusionObservationGridY;
                FusionPixelStride = fusionPixelStride;
                FusionSlotIndex = fusionSlotIndex;
                FusionSlotToken = fusionSlotToken;
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct Correspondence
        {
            public Vector4 SourceValid;
            public Vector4 TargetSigma;
            public Vector4 NormalAngle;
        }

        private sealed class FrameSlot
        {
            public ComputeBuffer Observations;
            public ComputeBuffer Correspondences;
            public ComputeBuffer PreTransactionCorrespondenceIdentity;
            public ComputeBuffer CorrespondenceIdentity;
            public ComputeBuffer DirectProjectionAudit;
            public bool Pending;
            public bool CorrespondenceReadbackDone;
            public bool DirectProjectionReadbackDone;
            public bool DirectProjectionScheduled;
            public bool ConsumerHeld;
            public bool Adjudicated;
            public bool CandidateCommitted;
            public int ObservationCount;
            public int FrameIndex;
            public int GridX;
            public int GridY;
            public int DepthWidth;
            public int DepthHeight;
            public int PlatformFrame;
            public float MotionQuality;
            public float AngularSpeed;
            public float LinearSpeed;
            public bool HasRawDepth;
            public bool HasTemporalReason;
            public bool ReportFrame;
            public Matrix4x4[] ProjectionInv;
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
        // 与 GPU 的 HasPromotionHardEdgeRisk 同步。跨度、间隙和双簇只是复核上下文；
        // 只有最终边缘、裙边、跨眼证伪和未被局部平面救回的掠射才失去晋升权。
        private static bool HasPromotionHardEdgeRisk(uint edgeReason)
        {
            bool skirt = (edgeReason & (1u << 2)) != 0u;
            bool crossEye = (edgeReason & (1u << 6)) != 0u;
            bool rejectedEdge = (edgeReason & (1u << 7)) != 0u;
            bool grazing = (edgeReason & (1u << 4)) != 0u;
            bool grazingRescued = (edgeReason & (1u << 10)) != 0u;
            return skirt || crossEye || rejectedEdge || (grazing && !grazingRescued);
        }
        private const int RingSize = 3;
        private const int StatsCount = 25;
        private const int AuditTotalsCount = 13;
        private const int RetireAuditCapacity = 32768;
        private const int PromotionAuditCapacity = 32768;
        // stableID 从 1 单调递增；该容量覆盖完整候选表容量并给淘汰后重生留余量。
        // 超出时只计 overflow，不回绕覆盖旧浪头。
        private const int CourtWaveCapacity = 262144;
        // 每四帧对完整稳定候选账本做一次候选不确定足迹直投，避免把 Quest GPU
        // 压成瓶颈。记录不参与融合、塑形或纸皮；只在候选已进入待退场后，给其
        // 删除/保留裁决提供足迹证词。
        private const int DirectProjectionFrameStride = 4;
        private const int DirectProjectionAuditCapacity = 4096;
        private const int DirectProjectionHeaderRows = 3;
        private const int DirectProjectionRecordRows = 9;
        private const int DirectProjectionAuditRows = DirectProjectionHeaderRows +
            DirectProjectionAuditCapacity * DirectProjectionRecordRows;
        private const int RetirementGhostCapacity = 4096;
        private const int RetirementGhostObservationCapacity = 32;
        private const int RetirementGateCounterCount = 11;
        private const float NearStationaryTranslationMm = 10f;
        private const float NearStationaryRaySeparationDeg = 0.25f;
        private const float SameViewTemporalConsistencyConeDeg = 0.25f;
        private const float RetirementGateCrossViewDeg = 1f;
        private static readonly float[] RetirementContractShadowAnglesDeg =
            { 0.5f, 1.0f, 2.0f };
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
        private static readonly int DepthProjID = Shader.PropertyToID("_DepthProj");
        private static readonly int DepthViewID = Shader.PropertyToID("_DepthView");
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
        private static readonly int CorrespondenceIdentityID =
            Shader.PropertyToID("_CorrespondenceIdentity");
        private static readonly int CellKeysID = Shader.PropertyToID("_CellKeys");
        private static readonly int CellStateID = Shader.PropertyToID("_CellState");
        private static readonly int CellLockID = Shader.PropertyToID("_CellLock");
        private static readonly int CandidateCenterSigmaID = Shader.PropertyToID("_CandidateCenterSigma");
        private static readonly int CandidateNormalSupportID = Shader.PropertyToID("_CandidateNormalSupport");
        private static readonly int CandidateFirstViewAngleID = Shader.PropertyToID("_CandidateFirstViewAngle");
        private static readonly int CandidateMetaID = Shader.PropertyToID("_CandidateMeta");
        private static readonly int CandidateEvidenceID = Shader.PropertyToID("_CandidateEvidence");
        private static readonly int CandidateDirectSupportReceiptID =
            Shader.PropertyToID("_CandidateDirectSupportReceipt");
        private static readonly int CandidateRetirementGateStateID =
            Shader.PropertyToID("_CandidateRetirementGateState");
        private static readonly int CandidateRetirementSupportAID =
            Shader.PropertyToID("_CandidateRetirementSupportA");
        private static readonly int CandidateRetirementSupportBID =
            Shader.PropertyToID("_CandidateRetirementSupportB");
        private static readonly int CandidateRetirementFreeAID =
            Shader.PropertyToID("_CandidateRetirementFreeA");
        private static readonly int CandidateRetirementFreeBID =
            Shader.PropertyToID("_CandidateRetirementFreeB");
        private static readonly int RetirementGateCountersID =
            Shader.PropertyToID("_RetirementGateCounters");
        private static readonly int CandidateBirthSourceReasonID = Shader.PropertyToID("_CandidateBirthSourceReason");
        private static readonly int CandidateBirthMotionViewID = Shader.PropertyToID("_CandidateBirthMotionView");
        private static readonly int CandidateBirthDeltaQualityID = Shader.PropertyToID("_CandidateBirthDeltaQuality");
        private static readonly int CandidateEvidenceClassCountsID = Shader.PropertyToID("_CandidateEvidenceClassCounts");
        private static readonly int CandidatePromotionAuthorityID = Shader.PropertyToID("_CandidatePromotionAuthority");
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
        private static readonly int RetireAuditGateStateID =
            Shader.PropertyToID("_RetireAuditGateState");
        private static readonly int RetireAuditGateWitnessFramesID =
            Shader.PropertyToID("_RetireAuditGateWitnessFrames");
        private static readonly int RetireAuditGateWitnessAnglesID =
            Shader.PropertyToID("_RetireAuditGateWitnessAngles");
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
        private static readonly int PromotionAuditCountersID = Shader.PropertyToID("_PromotionAuditCounters");
        private static readonly int PromotionAuditMetaID = Shader.PropertyToID("_PromotionAuditMeta");
        private static readonly int PromotionAuditGeometryID = Shader.PropertyToID("_PromotionAuditGeometry");
        private static readonly int PromotionAuditBirthSourceReasonID = Shader.PropertyToID("_PromotionAuditBirthSourceReason");
        private static readonly int PromotionAuditSourceReasonID = Shader.PropertyToID("_PromotionAuditSourceReason");
        private static readonly int PromotionAuditBirthMotionViewID = Shader.PropertyToID("_PromotionAuditBirthMotionView");
        private static readonly int PromotionAuditMotionViewID = Shader.PropertyToID("_PromotionAuditMotionView");
        private static readonly int PromotionAuditDeltaSigmaID = Shader.PropertyToID("_PromotionAuditDeltaSigma");
        private static readonly int PromotionAuditQualityRangeID = Shader.PropertyToID("_PromotionAuditQualityRange");
        private static readonly int PromotionAuditSupportID = Shader.PropertyToID("_PromotionAuditSupport");
        private static readonly int PromotionAuditAuthorityID = Shader.PropertyToID("_PromotionAuditAuthority");
        private static readonly int PromotionAuditCapacityID = Shader.PropertyToID("_PromotionAuditCapacity");
        private static readonly int DirectProjectionAuditID =
            Shader.PropertyToID("_DirectProjectionAudit");
        private static readonly int DirectProjectionAuditCapacityID =
            Shader.PropertyToID("_DirectProjectionAuditCapacity");
        private static readonly int RetirementGhostCountersID =
            Shader.PropertyToID("_RetirementGhostCounters");
        private static readonly int RetirementGhostIdentityID =
            Shader.PropertyToID("_RetirementGhostIdentity");
        private static readonly int RetirementGhostCenterSigmaID =
            Shader.PropertyToID("_RetirementGhostCenterSigma");
        private static readonly int RetirementGhostNormalSupportID =
            Shader.PropertyToID("_RetirementGhostNormalSupport");
        private static readonly int RetirementGhostOriginCameraID =
            Shader.PropertyToID("_RetirementGhostOriginCamera");
        private static readonly int RetirementGhostStateID =
            Shader.PropertyToID("_RetirementGhostState");
        private static readonly int RetirementGhostClassesID =
            Shader.PropertyToID("_RetirementGhostClasses");
        private static readonly int RetirementGhostObservationStateID =
            Shader.PropertyToID("_RetirementGhostObservationState");
        private static readonly int RetirementGhostObservationMetaID =
            Shader.PropertyToID("_RetirementGhostObservationMeta");
        private static readonly int RetirementGhostObservationCameraID =
            Shader.PropertyToID("_RetirementGhostObservationCamera");
        private static readonly int RetirementGhostObservationViewID =
            Shader.PropertyToID("_RetirementGhostObservationView");
        private static readonly int RetirementGhostObservationResidualID =
            Shader.PropertyToID("_RetirementGhostObservationResidual");
        private static readonly int RetirementGhostCapacityID =
            Shader.PropertyToID("_RetirementGhostCapacity");
        private static readonly int RetirementGhostObservationCapacityID =
            Shader.PropertyToID("_RetirementGhostObservationCapacity");

        private readonly ComputeShader _shader;
        private readonly ComputeKernelHelper _clearCells;
        private readonly ComputeKernelHelper _clearStats;
        private readonly ComputeKernelHelper _clearDirectProjectionAudit;
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
        private readonly ComputeKernelHelper _captureDirectProjectionAudit;
        private readonly ComputeKernelHelper _commitRetirementGateWitnesses;
        private readonly ComputeKernelHelper _captureRetirementGhostAudit;
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
        private ComputeBuffer _candidateDirectSupportReceipt;
        private ComputeBuffer _candidateRetirementGateState;
        private ComputeBuffer _candidateRetirementSupportA;
        private ComputeBuffer _candidateRetirementSupportB;
        private ComputeBuffer _candidateRetirementFreeA;
        private ComputeBuffer _candidateRetirementFreeB;
        private ComputeBuffer _retirementGateCounters;
        private ComputeBuffer _candidateBirthSourceReason;
        private ComputeBuffer _candidateBirthMotionView;
        private ComputeBuffer _candidateBirthDeltaQuality;
        private ComputeBuffer _candidateEvidenceClassCounts;
        private ComputeBuffer _candidatePromotionAuthority;
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
        private ComputeBuffer _retireAuditGateState;
        private ComputeBuffer _retireAuditGateWitnessFrames;
        private ComputeBuffer _retireAuditGateWitnessAngles;
        private ComputeBuffer _courtWaveLock;
        private ComputeBuffer _courtWavePeakBits;
        private ComputeBuffer _courtWavePositionResidual;
        private ComputeBuffer _courtWaveTargetSigma;
        private ComputeBuffer _courtWaveMeta;
        private ComputeBuffer _courtWaveSourceReason;
        private ComputeBuffer _courtWaveRawPositionDelta;
        private ComputeBuffer _courtWaveMotionView;
        private ComputeBuffer _courtWaveCounters;
        private ComputeBuffer _promotionAuditCounters;
        private ComputeBuffer _promotionAuditMeta;
        private ComputeBuffer _promotionAuditGeometry;
        private ComputeBuffer _promotionAuditBirthSourceReason;
        private ComputeBuffer _promotionAuditSourceReason;
        private ComputeBuffer _promotionAuditBirthMotionView;
        private ComputeBuffer _promotionAuditMotionView;
        private ComputeBuffer _promotionAuditDeltaSigma;
        private ComputeBuffer _promotionAuditQualityRange;
        private ComputeBuffer _promotionAuditSupport;
        private ComputeBuffer _promotionAuditAuthority;
        private ComputeBuffer _retirementGhostCounters;
        private ComputeBuffer _retirementGhostIdentity;
        private ComputeBuffer _retirementGhostCenterSigma;
        private ComputeBuffer _retirementGhostNormalSupport;
        private ComputeBuffer _retirementGhostOriginCamera;
        private ComputeBuffer _retirementGhostState;
        private ComputeBuffer _retirementGhostClasses;
        private ComputeBuffer _retirementGhostObservationState;
        private ComputeBuffer _retirementGhostObservationMeta;
        private ComputeBuffer _retirementGhostObservationCamera;
        private ComputeBuffer _retirementGhostObservationView;
        private ComputeBuffer _retirementGhostObservationResidual;
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
            _clearDirectProjectionAudit = new ComputeKernelHelper(_shader,
                "ClearDirectProjectionAudit");
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
            _captureDirectProjectionAudit = new ComputeKernelHelper(_shader,
                "CaptureDirectProjectionAudit");
            _commitRetirementGateWitnesses = new ComputeKernelHelper(_shader,
                "CommitRetirementGateWitnesses");
            _captureRetirementGhostAudit = new ComputeKernelHelper(_shader,
                "CaptureRetirementGhostAudit");
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
            _candidateDirectSupportReceipt = new ComputeBuffer(candidateCount,
                CandidateMetaStride);
            _candidateRetirementGateState = new ComputeBuffer(candidateCount,
                CandidateMetaStride);
            _candidateRetirementSupportA = new ComputeBuffer(candidateCount,
                CandidateFloat4Stride);
            _candidateRetirementSupportB = new ComputeBuffer(candidateCount,
                CandidateFloat4Stride);
            _candidateRetirementFreeA = new ComputeBuffer(candidateCount,
                CandidateFloat4Stride);
            _candidateRetirementFreeB = new ComputeBuffer(candidateCount,
                CandidateFloat4Stride);
            _retirementGateCounters = new ComputeBuffer(
                RetirementGateCounterCount, sizeof(uint));
            _retirementGateCounters.SetData(new uint[RetirementGateCounterCount]);
            _candidateBirthSourceReason = new ComputeBuffer(candidateCount, CandidateMetaStride);
            _candidateBirthMotionView = new ComputeBuffer(candidateCount, CandidateFloat4Stride);
            _candidateBirthDeltaQuality = new ComputeBuffer(candidateCount, CandidateFloat4Stride);
            _candidateEvidenceClassCounts = new ComputeBuffer(candidateCount, sizeof(uint));
            _candidatePromotionAuthority = new ComputeBuffer(candidateCount, CandidateMetaStride);
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
            _retireAuditGateState = new ComputeBuffer(RetireAuditCapacity,
                CandidateMetaStride);
            _retireAuditGateWitnessFrames = new ComputeBuffer(RetireAuditCapacity,
                CandidateMetaStride);
            _retireAuditGateWitnessAngles = new ComputeBuffer(RetireAuditCapacity,
                CandidateFloat4Stride);
            _courtWaveLock = new ComputeBuffer(CourtWaveCapacity, sizeof(uint));
            _courtWavePeakBits = new ComputeBuffer(CourtWaveCapacity, sizeof(uint));
            _courtWavePositionResidual = new ComputeBuffer(CourtWaveCapacity, CandidateFloat4Stride);
            _courtWaveTargetSigma = new ComputeBuffer(CourtWaveCapacity, CandidateFloat4Stride);
            _courtWaveMeta = new ComputeBuffer(CourtWaveCapacity, CandidateMetaStride);
            _courtWaveSourceReason = new ComputeBuffer(CourtWaveCapacity, CandidateMetaStride);
            _courtWaveRawPositionDelta = new ComputeBuffer(CourtWaveCapacity, CandidateFloat4Stride);
            _courtWaveMotionView = new ComputeBuffer(CourtWaveCapacity, CandidateFloat4Stride);
            _courtWaveCounters = new ComputeBuffer(3, sizeof(uint));
            _promotionAuditCounters = new ComputeBuffer(1, sizeof(uint));
            _promotionAuditCounters.SetData(new uint[1]);
            _promotionAuditMeta = new ComputeBuffer(PromotionAuditCapacity, CandidateMetaStride);
            _promotionAuditGeometry = new ComputeBuffer(PromotionAuditCapacity, CandidateFloat4Stride);
            _promotionAuditBirthSourceReason = new ComputeBuffer(PromotionAuditCapacity, CandidateMetaStride);
            _promotionAuditSourceReason = new ComputeBuffer(PromotionAuditCapacity, CandidateMetaStride);
            _promotionAuditBirthMotionView = new ComputeBuffer(PromotionAuditCapacity, CandidateFloat4Stride);
            _promotionAuditMotionView = new ComputeBuffer(PromotionAuditCapacity, CandidateFloat4Stride);
            _promotionAuditDeltaSigma = new ComputeBuffer(PromotionAuditCapacity, CandidateFloat4Stride);
            _promotionAuditQualityRange = new ComputeBuffer(PromotionAuditCapacity, CandidateFloat4Stride);
            _promotionAuditSupport = new ComputeBuffer(PromotionAuditCapacity, CandidateMetaStride);
            _promotionAuditAuthority = new ComputeBuffer(PromotionAuditCapacity, CandidateMetaStride);
            _retirementGhostCounters = new ComputeBuffer(2, sizeof(uint));
            _retirementGhostCounters.SetData(new uint[2]);
            _retirementGhostIdentity = new ComputeBuffer(RetirementGhostCapacity,
                CandidateMetaStride);
            _retirementGhostCenterSigma = new ComputeBuffer(RetirementGhostCapacity,
                CandidateFloat4Stride);
            _retirementGhostNormalSupport = new ComputeBuffer(RetirementGhostCapacity,
                CandidateFloat4Stride);
            _retirementGhostOriginCamera = new ComputeBuffer(RetirementGhostCapacity,
                CandidateFloat4Stride);
            _retirementGhostState = new ComputeBuffer(RetirementGhostCapacity,
                CandidateMetaStride);
            _retirementGhostClasses = new ComputeBuffer(RetirementGhostCapacity,
                CandidateMetaStride);
            _retirementGhostObservationState = new ComputeBuffer(
                RetirementGhostCapacity, CandidateMetaStride);
            int ghostObservationCount = RetirementGhostCapacity *
                RetirementGhostObservationCapacity;
            _retirementGhostObservationMeta = new ComputeBuffer(
                ghostObservationCount, CandidateMetaStride);
            _retirementGhostObservationCamera = new ComputeBuffer(
                ghostObservationCount, CandidateFloat4Stride);
            _retirementGhostObservationView = new ComputeBuffer(
                ghostObservationCount, CandidateFloat4Stride);
            _retirementGhostObservationResidual = new ComputeBuffer(
                ghostObservationCount, CandidateFloat4Stride);
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
                _countCensus, _captureDirectProjectionAudit,
                _commitRetirementGateWitnesses,
                _captureRetirementGhostAudit
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
                kernel.Set(CandidateDirectSupportReceiptID,
                    _candidateDirectSupportReceipt);
                kernel.Set(CandidateRetirementGateStateID,
                    _candidateRetirementGateState);
                kernel.Set(CandidateRetirementSupportAID,
                    _candidateRetirementSupportA);
                kernel.Set(CandidateRetirementSupportBID,
                    _candidateRetirementSupportB);
                kernel.Set(CandidateRetirementFreeAID,
                    _candidateRetirementFreeA);
                kernel.Set(CandidateRetirementFreeBID,
                    _candidateRetirementFreeB);
                kernel.Set(RetirementGateCountersID, _retirementGateCounters);
                kernel.Set(CandidateBirthSourceReasonID, _candidateBirthSourceReason);
                kernel.Set(CandidateBirthMotionViewID, _candidateBirthMotionView);
                kernel.Set(CandidateBirthDeltaQualityID, _candidateBirthDeltaQuality);
                kernel.Set(CandidateEvidenceClassCountsID, _candidateEvidenceClassCounts);
                kernel.Set(CandidatePromotionAuthorityID, _candidatePromotionAuthority);
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
                kernel.Set(RetireAuditGateStateID, _retireAuditGateState);
                kernel.Set(RetireAuditGateWitnessFramesID,
                    _retireAuditGateWitnessFrames);
                kernel.Set(RetireAuditGateWitnessAnglesID,
                    _retireAuditGateWitnessAngles);
                kernel.Set(CourtWaveLockID, _courtWaveLock);
                kernel.Set(CourtWavePeakBitsID, _courtWavePeakBits);
                kernel.Set(CourtWavePositionResidualID, _courtWavePositionResidual);
                kernel.Set(CourtWaveTargetSigmaID, _courtWaveTargetSigma);
                kernel.Set(CourtWaveMetaID, _courtWaveMeta);
                kernel.Set(CourtWaveSourceReasonID, _courtWaveSourceReason);
                kernel.Set(CourtWaveRawPositionDeltaID, _courtWaveRawPositionDelta);
                kernel.Set(CourtWaveMotionViewID, _courtWaveMotionView);
                kernel.Set(CourtWaveCountersID, _courtWaveCounters);
                kernel.Set(PromotionAuditCountersID, _promotionAuditCounters);
                kernel.Set(PromotionAuditMetaID, _promotionAuditMeta);
                kernel.Set(PromotionAuditGeometryID, _promotionAuditGeometry);
                kernel.Set(PromotionAuditBirthSourceReasonID, _promotionAuditBirthSourceReason);
                kernel.Set(PromotionAuditSourceReasonID, _promotionAuditSourceReason);
                kernel.Set(PromotionAuditBirthMotionViewID, _promotionAuditBirthMotionView);
                kernel.Set(PromotionAuditMotionViewID, _promotionAuditMotionView);
                kernel.Set(PromotionAuditDeltaSigmaID, _promotionAuditDeltaSigma);
                kernel.Set(PromotionAuditQualityRangeID, _promotionAuditQualityRange);
                kernel.Set(PromotionAuditSupportID, _promotionAuditSupport);
                kernel.Set(PromotionAuditAuthorityID, _promotionAuditAuthority);
                kernel.Set(RetirementGhostCountersID, _retirementGhostCounters);
                kernel.Set(RetirementGhostIdentityID, _retirementGhostIdentity);
                kernel.Set(RetirementGhostCenterSigmaID,
                    _retirementGhostCenterSigma);
                kernel.Set(RetirementGhostNormalSupportID,
                    _retirementGhostNormalSupport);
                kernel.Set(RetirementGhostOriginCameraID,
                    _retirementGhostOriginCamera);
                kernel.Set(RetirementGhostStateID, _retirementGhostState);
                kernel.Set(RetirementGhostClassesID, _retirementGhostClasses);
                kernel.Set(RetirementGhostObservationStateID,
                    _retirementGhostObservationState);
                kernel.Set(RetirementGhostObservationMetaID,
                    _retirementGhostObservationMeta);
                kernel.Set(RetirementGhostObservationCameraID,
                    _retirementGhostObservationCamera);
                kernel.Set(RetirementGhostObservationViewID,
                    _retirementGhostObservationView);
                kernel.Set(RetirementGhostObservationResidualID,
                    _retirementGhostObservationResidual);
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
            _shader.SetInt(PromotionAuditCapacityID, PromotionAuditCapacity);
            _shader.SetInt(DirectProjectionAuditCapacityID,
                DirectProjectionAuditCapacity);
            _shader.SetInt(RetirementGhostCapacityID, RetirementGhostCapacity);
            _shader.SetInt(RetirementGhostObservationCapacityID,
                RetirementGhostObservationCapacity);
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
                slot.PreTransactionCorrespondenceIdentity?.Release();
                slot.CorrespondenceIdentity?.Release();
                slot.DirectProjectionAudit?.Release();
                slot.Observations = new ComputeBuffer(observationCount, ObservationStride);
                slot.Correspondences = new ComputeBuffer(observationCount, CorrespondenceStride);
                slot.PreTransactionCorrespondenceIdentity = new ComputeBuffer(observationCount,
                    CandidateMetaStride);
                slot.CorrespondenceIdentity = new ComputeBuffer(observationCount,
                    CandidateMetaStride);
                slot.DirectProjectionAudit = new ComputeBuffer(
                    DirectProjectionAuditRows, CandidateMetaStride);
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
            slot.CorrespondenceReadbackDone = false;
            slot.DirectProjectionReadbackDone = false;
            slot.DirectProjectionScheduled = false;
            slot.Adjudicated = false;
            slot.CandidateCommitted = false;
            slot.FrameIndex = frameIndex;
            slot.GridX = gridX;
            slot.GridY = gridY;
            slot.DepthWidth = depthWidth;
            slot.DepthHeight = depthHeight;
            slot.PlatformFrame = platformFrame;
            slot.MotionQuality = Mathf.Clamp01(motionQuality);
            slot.AngularSpeed = Mathf.Max(0f, angularSpeed);
            slot.LinearSpeed = Mathf.Max(0f, linearSpeed);
            slot.HasRawDepth = rawDepthTexture != null;
            slot.HasTemporalReason = temporalReasonTexture != null;
            slot.ReportFrame = frameIndex % _reportInterval == 0;
            slot.ProjectionInv = (Matrix4x4[])projectionInverse.Clone();
            slot.ViewInv = (Matrix4x4[])viewInverse.Clone();
            slot.Generation = _generation;
            slot.ConsumerToken = NextConsumerToken();
            slot.Completion = completion;

            var projection = new Matrix4x4[2];
            var view = new Matrix4x4[2];
            for (int eye = 0; eye < 2; eye++)
            {
                int projectionIndex = Mathf.Min(eye, projectionInverse.Length - 1);
                int viewIndex = Mathf.Min(eye, viewInverse.Length - 1);
                projection[eye] = projectionInverse[projectionIndex].inverse;
                view[eye] = viewInverse[viewIndex].inverse;
            }

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
            _shader.SetMatrixArray(DepthProjID, projection);
            _shader.SetMatrixArray(DepthViewID, view);
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
            _buildCorrespondences.Set(CorrespondenceIdentityID,
                slot.PreTransactionCorrespondenceIdentity);

            // 幽灵复核每张原料完整且运动合格的后续帧，不受 direct45 每四帧
            // 记录节流影响。它只写独立影子缓冲，不能改变实际候选退场。
            if (slot.HasRawDepth && slot.MotionQuality >= 0.70f)
            {
                _captureRetirementGhostAudit.Set(DepthTexID, depthTexture);
                _captureRetirementGhostAudit.Set(RawDepthTexID, rawDepthTexture);
                _captureRetirementGhostAudit.Set(EdgeReasonTexID, edgeReasonTexture);
                _captureRetirementGhostAudit.Set(TemporalReasonTexID,
                    temporalReasonTexture != null ? temporalReasonTexture : edgeReasonTexture);
                _captureRetirementGhostAudit.DispatchFit(
                    RetirementGhostCapacity, 1, 1);
            }

            bool directProjectionScheduled = slot.HasRawDepth &&
                slot.MotionQuality >= 0.70f &&
                frameIndex % DirectProjectionFrameStride == 0;
            slot.DirectProjectionScheduled = directProjectionScheduled;
            if (directProjectionScheduled)
            {
                _clearDirectProjectionAudit.Set(DirectProjectionAuditID,
                    slot.DirectProjectionAudit);
                _captureDirectProjectionAudit.Set(DirectProjectionAuditID,
                    slot.DirectProjectionAudit);
                _captureDirectProjectionAudit.Set(DepthTexID, depthTexture);
                _captureDirectProjectionAudit.Set(RawDepthTexID, rawDepthTexture);
                _captureDirectProjectionAudit.Set(EdgeReasonTexID, edgeReasonTexture);
                _captureDirectProjectionAudit.Set(TemporalReasonTexID,
                    temporalReasonTexture != null ? temporalReasonTexture : edgeReasonTexture);
                _clearDirectProjectionAudit.DispatchFit(DirectProjectionHeaderRows, 1, 1);
                _captureDirectProjectionAudit.DispatchFit(
                    TableCapacity * CandidateCapacity, 1, 1);
                AsyncGPUReadback.Request(slot.DirectProjectionAudit,
                    request => OnDirectProjectionAudit(request, slot));
            }
            else
            {
                slot.DirectProjectionReadbackDone = true;
            }

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
                slot.CorrespondenceReadbackDone = true;
                TryCompleteSlotReadbacks(slot);
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
                Matrix4x4 sourceViewInverse = slot.ViewInv != null &&
                    slot.ViewInv.Length > DepthCapture.FusionEyeIndex
                    ? slot.ViewInv[DepthCapture.FusionEyeIndex]
                    : Matrix4x4.identity;
                Matrix4x4 sourceProjectionInverse = slot.ProjectionInv != null &&
                    slot.ProjectionInv.Length > DepthCapture.FusionEyeIndex
                    ? slot.ProjectionInv[DepthCapture.FusionEyeIndex]
                    : Matrix4x4.identity;
                ScanReplaySessionPackage.Active?.RecordVirtualProbeShadow(
                    slot.FrameIndex, slot.PlatformFrame, data,
                    slot.GridX, slot.GridY, _pixelStride,
                    slot.DepthWidth, slot.DepthHeight, sourceProjectionInverse,
                    sourceViewInverse,
                    slot.AngularSpeed, slot.LinearSpeed, slot.MotionQuality);
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
                consumerWillHold ? slot.PreTransactionCorrespondenceIdentity : null,
                consumerWillHold ? slot.CorrespondenceIdentity : null,
                slot.GridX, slot.GridY, _pixelStride,
                consumerWillHold ? slot.SlotIndex : -1,
                consumerWillHold ? slot.ConsumerToken : 0);

            var completion = slot.Completion;
            slot.Completion = null;
            slot.ConsumerHeld = consumerWillHold;
            slot.CorrespondenceReadbackDone = true;
            TryCompleteSlotReadbacks(slot);
            // 有生产消费者时，候选事务必须等整帧准入结果。旧顺序在这里已经先
            // Update/Challenge/Retire，随后 VolumeIntegrator 才可能以撞顶/快角拒帧，
            // 导致“没写 TSDF 的帧却改了稳定候选”。无消费者的纯影子轨仍立即提交。
            if (!consumerWillHold)
                AdjudicateSlot(slot, correction, true);
            completion?.Invoke(decision);
        }

        private void OnDirectProjectionAudit(AsyncGPUReadbackRequest request,
            FrameSlot slot)
        {
            if (_disposed || slot == null) return;
            if (slot.Generation == _generation)
            {
                if (request.hasError)
                {
                    ScanReplaySessionPackage.Active?
                        .RecordDirectProjectionShadowReadbackError();
                }
                else
                {
                    ScanReplaySessionPackage.Active?.RecordDirectProjectionShadow(
                        slot.FrameIndex, slot.PlatformFrame,
                        request.GetData<uint4>(), slot.AngularSpeed,
                        slot.LinearSpeed, slot.MotionQuality);
                }
            }
            slot.DirectProjectionReadbackDone = true;
            TryCompleteSlotReadbacks(slot);
        }

        private static void TryCompleteSlotReadbacks(FrameSlot slot)
        {
            if (slot == null) return;
            slot.Pending = !(slot.CorrespondenceReadbackDone &&
                             slot.DirectProjectionReadbackDone);
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
            _captureCourtWaves.Set(CorrespondenceIdentityID,
                slot.CorrespondenceIdentity);
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
                    _rebuildProductionCorrespondences.Set(CorrespondenceIdentityID,
                        slot.CorrespondenceIdentity);
                    _rebuildProductionCorrespondences.DispatchFit(slot.ObservationCount, 1, 1);
                    _challengeCandidates.Set(ObservationsID, slot.Observations);
                    _challengeCandidates.DispatchFit(slot.ObservationCount, 1, 1);
                    if (slot.DirectProjectionScheduled)
                    {
                        _commitRetirementGateWitnesses.Set(
                            DirectProjectionAuditID, slot.DirectProjectionAudit);
                        _commitRetirementGateWitnesses.DispatchFit(
                            DirectProjectionAuditCapacity, 1, 1);
                    }
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
            var retirementGateStateRequest = AsyncGPUReadback.Request(
                _candidateRetirementGateState);
            var retirementSupportARequest = AsyncGPUReadback.Request(
                _candidateRetirementSupportA);
            var retirementSupportBRequest = AsyncGPUReadback.Request(
                _candidateRetirementSupportB);
            var retirementFreeARequest = AsyncGPUReadback.Request(
                _candidateRetirementFreeA);
            var retirementFreeBRequest = AsyncGPUReadback.Request(
                _candidateRetirementFreeB);
            var retirementGateCountersRequest = AsyncGPUReadback.Request(
                _retirementGateCounters);
            var candidatePromotionAuthorityRequest = AsyncGPUReadback.Request(
                _candidatePromotionAuthority);
            var nextIdRequest = AsyncGPUReadback.Request(_nextStableId);
            var totalsRequest = AsyncGPUReadback.Request(_auditTotals);
            var retireCounterRequest = AsyncGPUReadback.Request(_retireAuditCounters);
            var retireMetaRequest = AsyncGPUReadback.Request(_retireAuditMeta);
            var retireGeometryRequest = AsyncGPUReadback.Request(_retireAuditGeometry);
            var retireEvidenceRequest = AsyncGPUReadback.Request(_retireAuditEvidence);
            var retireTimingRequest = AsyncGPUReadback.Request(_retireAuditTiming);
            var retireGateStateRequest = AsyncGPUReadback.Request(
                _retireAuditGateState);
            var retireGateWitnessFramesRequest = AsyncGPUReadback.Request(
                _retireAuditGateWitnessFrames);
            var retireGateWitnessAnglesRequest = AsyncGPUReadback.Request(
                _retireAuditGateWitnessAngles);
            var courtPeakRequest = AsyncGPUReadback.Request(_courtWavePeakBits);
            var courtPositionRequest = AsyncGPUReadback.Request(_courtWavePositionResidual);
            var courtTargetRequest = AsyncGPUReadback.Request(_courtWaveTargetSigma);
            var courtMetaRequest = AsyncGPUReadback.Request(_courtWaveMeta);
            var courtSourceReasonRequest = AsyncGPUReadback.Request(_courtWaveSourceReason);
            var courtRawRequest = AsyncGPUReadback.Request(_courtWaveRawPositionDelta);
            var courtMotionViewRequest = AsyncGPUReadback.Request(_courtWaveMotionView);
            var courtCountersRequest = AsyncGPUReadback.Request(_courtWaveCounters);
            var promotionCountersRequest = AsyncGPUReadback.Request(_promotionAuditCounters);
            var promotionMetaRequest = AsyncGPUReadback.Request(_promotionAuditMeta);
            var promotionGeometryRequest = AsyncGPUReadback.Request(_promotionAuditGeometry);
            var promotionBirthSourceRequest = AsyncGPUReadback.Request(_promotionAuditBirthSourceReason);
            var promotionSourceRequest = AsyncGPUReadback.Request(_promotionAuditSourceReason);
            var promotionBirthMotionRequest = AsyncGPUReadback.Request(_promotionAuditBirthMotionView);
            var promotionMotionRequest = AsyncGPUReadback.Request(_promotionAuditMotionView);
            var promotionDeltaRequest = AsyncGPUReadback.Request(_promotionAuditDeltaSigma);
            var promotionQualityRangeRequest = AsyncGPUReadback.Request(_promotionAuditQualityRange);
            var promotionSupportRequest = AsyncGPUReadback.Request(_promotionAuditSupport);
            var promotionAuthorityRequest = AsyncGPUReadback.Request(
                _promotionAuditAuthority);
            var ghostCountersRequest = AsyncGPUReadback.Request(_retirementGhostCounters);
            var ghostIdentityRequest = AsyncGPUReadback.Request(_retirementGhostIdentity);
            var ghostCenterRequest = AsyncGPUReadback.Request(_retirementGhostCenterSigma);
            var ghostNormalRequest = AsyncGPUReadback.Request(_retirementGhostNormalSupport);
            var ghostOriginCameraRequest = AsyncGPUReadback.Request(
                _retirementGhostOriginCamera);
            var ghostStateRequest = AsyncGPUReadback.Request(_retirementGhostState);
            var ghostClassesRequest = AsyncGPUReadback.Request(_retirementGhostClasses);
            var ghostObservationStateRequest = AsyncGPUReadback.Request(
                _retirementGhostObservationState);
            var ghostObservationMetaRequest = AsyncGPUReadback.Request(
                _retirementGhostObservationMeta);
            var ghostObservationCameraRequest = AsyncGPUReadback.Request(
                _retirementGhostObservationCamera);
            var ghostObservationViewRequest = AsyncGPUReadback.Request(
                _retirementGhostObservationView);
            var ghostObservationResidualRequest = AsyncGPUReadback.Request(
                _retirementGhostObservationResidual);

            while (!cellKeysRequest.done || !cellStateRequest.done ||
                   !centerRequest.done || !normalRequest.done || !viewRequest.done ||
                   !metaRequest.done || !evidenceRequest.done ||
                   !retirementGateStateRequest.done ||
                   !retirementSupportARequest.done ||
                   !retirementSupportBRequest.done ||
                   !retirementFreeARequest.done ||
                   !retirementFreeBRequest.done ||
                   !retirementGateCountersRequest.done ||
                   !candidatePromotionAuthorityRequest.done || !nextIdRequest.done ||
                   !totalsRequest.done || !retireCounterRequest.done ||
                   !retireMetaRequest.done || !retireGeometryRequest.done ||
                   !retireEvidenceRequest.done || !retireTimingRequest.done ||
                   !retireGateStateRequest.done ||
                   !retireGateWitnessFramesRequest.done ||
                   !retireGateWitnessAnglesRequest.done ||
                   !courtPeakRequest.done || !courtPositionRequest.done ||
                   !courtTargetRequest.done || !courtMetaRequest.done ||
                   !courtSourceReasonRequest.done || !courtRawRequest.done ||
                   !courtMotionViewRequest.done ||
                   !courtCountersRequest.done || !promotionCountersRequest.done ||
                   !promotionMetaRequest.done || !promotionGeometryRequest.done ||
                   !promotionBirthSourceRequest.done || !promotionSourceRequest.done ||
                   !promotionBirthMotionRequest.done || !promotionMotionRequest.done ||
                   !promotionDeltaRequest.done || !promotionQualityRangeRequest.done ||
                   !promotionSupportRequest.done || !promotionAuthorityRequest.done ||
                   !ghostCountersRequest.done || !ghostIdentityRequest.done ||
                   !ghostCenterRequest.done || !ghostNormalRequest.done ||
                   !ghostOriginCameraRequest.done || !ghostStateRequest.done ||
                   !ghostClassesRequest.done || !ghostObservationStateRequest.done ||
                   !ghostObservationMetaRequest.done ||
                   !ghostObservationCameraRequest.done ||
                   !ghostObservationViewRequest.done ||
                   !ghostObservationResidualRequest.done)
                yield return null;

            if (cellKeysRequest.hasError || cellStateRequest.hasError ||
                centerRequest.hasError || normalRequest.hasError || viewRequest.hasError ||
                metaRequest.hasError || evidenceRequest.hasError ||
                retirementGateStateRequest.hasError ||
                retirementSupportARequest.hasError ||
                retirementSupportBRequest.hasError ||
                retirementFreeARequest.hasError ||
                retirementFreeBRequest.hasError ||
                retirementGateCountersRequest.hasError ||
                candidatePromotionAuthorityRequest.hasError || nextIdRequest.hasError ||
                totalsRequest.hasError || retireCounterRequest.hasError ||
                retireMetaRequest.hasError || retireGeometryRequest.hasError ||
                retireEvidenceRequest.hasError || retireTimingRequest.hasError ||
                retireGateStateRequest.hasError ||
                retireGateWitnessFramesRequest.hasError ||
                retireGateWitnessAnglesRequest.hasError ||
                courtPeakRequest.hasError || courtPositionRequest.hasError ||
                courtTargetRequest.hasError || courtMetaRequest.hasError ||
                courtSourceReasonRequest.hasError || courtRawRequest.hasError ||
                courtMotionViewRequest.hasError ||
                courtCountersRequest.hasError || promotionCountersRequest.hasError ||
                promotionMetaRequest.hasError || promotionGeometryRequest.hasError ||
                promotionBirthSourceRequest.hasError || promotionSourceRequest.hasError ||
                promotionBirthMotionRequest.hasError || promotionMotionRequest.hasError ||
                promotionDeltaRequest.hasError || promotionQualityRangeRequest.hasError ||
                promotionSupportRequest.hasError || promotionAuthorityRequest.hasError ||
                 ghostCountersRequest.hasError || ghostIdentityRequest.hasError ||
                 ghostCenterRequest.hasError || ghostNormalRequest.hasError ||
                 ghostOriginCameraRequest.hasError || ghostStateRequest.hasError ||
                 ghostClassesRequest.hasError || ghostObservationStateRequest.hasError ||
                 ghostObservationMetaRequest.hasError ||
                 ghostObservationCameraRequest.hasError ||
                  ghostObservationViewRequest.hasError ||
                  ghostObservationResidualRequest.hasError)
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
            var retirementGateStates = retirementGateStateRequest.GetData<uint4>();
            var retirementSupportA = retirementSupportARequest.GetData<float4>();
            var retirementSupportB = retirementSupportBRequest.GetData<float4>();
            var retirementFreeA = retirementFreeARequest.GetData<float4>();
            var retirementFreeB = retirementFreeBRequest.GetData<float4>();
            var retirementGateCounters =
                retirementGateCountersRequest.GetData<uint>();
            var candidatePromotionAuthority =
                candidatePromotionAuthorityRequest.GetData<uint4>();
            var nextId = nextIdRequest.GetData<uint>();
            var totals = totalsRequest.GetData<uint>();
            var retireCounters = retireCounterRequest.GetData<uint>();
            var retireMeta = retireMetaRequest.GetData<uint4>();
            var retireGeometry = retireGeometryRequest.GetData<float4>();
            var retireEvidence = retireEvidenceRequest.GetData<uint4>();
            var retireTiming = retireTimingRequest.GetData<uint4>();
            var retireGateState = retireGateStateRequest.GetData<uint4>();
            var retireGateWitnessFrames =
                retireGateWitnessFramesRequest.GetData<uint4>();
            var retireGateWitnessAngles =
                retireGateWitnessAnglesRequest.GetData<float4>();
            var courtPeaks = courtPeakRequest.GetData<uint>();
            var courtPositions = courtPositionRequest.GetData<float4>();
            var courtTargets = courtTargetRequest.GetData<float4>();
            var courtMetas = courtMetaRequest.GetData<uint4>();
            var courtSourceReasons = courtSourceReasonRequest.GetData<uint4>();
            var courtRaw = courtRawRequest.GetData<float4>();
            var courtMotionView = courtMotionViewRequest.GetData<float4>();
            var courtCounters = courtCountersRequest.GetData<uint>();
            var promotionCounters = promotionCountersRequest.GetData<uint>();
            var promotionMetas = promotionMetaRequest.GetData<uint4>();
            var promotionGeometry = promotionGeometryRequest.GetData<float4>();
            var promotionBirthSources = promotionBirthSourceRequest.GetData<uint4>();
            var promotionSources = promotionSourceRequest.GetData<uint4>();
            var promotionBirthMotion = promotionBirthMotionRequest.GetData<float4>();
            var promotionMotion = promotionMotionRequest.GetData<float4>();
            var promotionDelta = promotionDeltaRequest.GetData<float4>();
            var promotionQualityRange = promotionQualityRangeRequest.GetData<float4>();
            var promotionSupport = promotionSupportRequest.GetData<uint4>();
            var promotionAuthority = promotionAuthorityRequest.GetData<uint4>();
            var ghostCounters = ghostCountersRequest.GetData<uint>();
            var ghostIdentity = ghostIdentityRequest.GetData<uint4>();
            var ghostCenters = ghostCenterRequest.GetData<float4>();
            var ghostNormals = ghostNormalRequest.GetData<float4>();
            var ghostOriginCameras = ghostOriginCameraRequest.GetData<float4>();
            var ghostStates = ghostStateRequest.GetData<uint4>();
            var ghostClasses = ghostClassesRequest.GetData<uint4>();
            var ghostObservationStates = ghostObservationStateRequest.GetData<uint4>();
            var ghostObservationMeta = ghostObservationMetaRequest.GetData<uint4>();
            var ghostObservationCamera = ghostObservationCameraRequest.GetData<float4>();
            var ghostObservationView = ghostObservationViewRequest.GetData<float4>();
            var ghostObservationResidual =
                ghostObservationResidualRequest.GetData<float4>();

            DateTime capturedUtc = DateTime.UtcNow;
            string stamp = capturedUtc.ToString("yyyyMMdd_HHmmss_fff",
                CultureInfo.InvariantCulture);
            string directory = Path.Combine(Application.persistentDataPath,
                "ScanCoverDiagnostics", "gungel_candidate_audit", stamp);
            string candidatesPath = Path.Combine(directory, "candidates.csv");
            string retirementsPath = Path.Combine(directory, "retirements.csv");
            string retirementGhostsPath = Path.Combine(directory,
                "retirement_ghosts.csv");
            string retirementGhostObservationsPath = Path.Combine(directory,
                "retirement_ghost_observations.csv");
            string retirementContractShadowPath = Path.Combine(directory,
                "retirement_contract_shadow.csv");
            string courtWavesPath = Path.Combine(directory, "court_waves.csv");
            string promotionsPath = Path.Combine(directory, "promotions.csv");
            string summaryPath = Path.Combine(directory, "candidate_summary.json");

            int activeCandidates = 0;
            int stableCandidates = 0;
            int embryoCandidates = 0;
            int activeCells = 0;
            int multiCandidateCells = 0;
            int stableIdMissing = 0;
            int pendingRetirementCandidates = 0;
            ulong aliveDualSupport = 0;
            ulong aliveOpposition = 0;
            uint latestFrame = 0u;
            bool haveBounds = false;
            Vector3 boundsMin = Vector3.zero;
            Vector3 boundsMax = Vector3.zero;
            var candidatesCsv = new StringBuilder(256 * 1024);
            candidatesCsv.AppendLine("candidate_index,cell_slot,layer_slot,cell_x,cell_y,cell_z,stable_id,state,observation_count,last_seen_frame,last_observed_frame,center_x_m,center_y_m,center_z_m,sigma_mm,normal_x,normal_y,normal_z,effective_support,dual_agree_support,promotion_authority_frames,observed_frames,required_authority_frames,promotion_authority_ready,opposition_votes,last_challenge_frame,first_view_x,first_view_y,first_view_z,view_spread_deg,retirement_gate_state,pending_since_frame,last_gate_decision,same_view_conflict_pairs,support_witness_a_frame,support_witness_b_frame,support_pair_span_deg,free_witness_a_frame,free_witness_b_frame,free_pair_span_deg");

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
                    uint4 itemPromotionAuthority = candidatePromotionAuthority[index];
                    uint requiredAuthority = RequiredPromotionAuthorityFrames(
                        itemPromotionAuthority.y);
                    float4 center = centers[index];
                    float4 normal = normals[index];
                    float4 view = views[index];
                    uint4 gateState = retirementGateStates[index];
                    bool gateIdentityValid = gateState.x == itemEvidence.z &&
                        gateState.y != 0u && itemEvidence.z != 0u;
                    float4 supportA = gateIdentityValid
                        ? retirementSupportA[index] : default;
                    float4 supportB = gateIdentityValid
                        ? retirementSupportB[index] : default;
                    float4 freeA = gateIdentityValid
                        ? retirementFreeA[index] : default;
                    float4 freeB = gateIdentityValid
                        ? retirementFreeB[index] : default;
                    if (gateIdentityValid) pendingRetirementCandidates++;
                    cellActive++;
                    activeCandidates++;
                    uint lastObservedFrame = itemPromotionAuthority.w == 0u
                        ? meta.z : itemPromotionAuthority.w - 1u;
                    latestFrame = Math.Max(latestFrame,
                        Math.Max(meta.z, lastObservedFrame));
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
                        .Append(lastObservedFrame).Append(',')
                        .Append(FloatText(center.x)).Append(',').Append(FloatText(center.y)).Append(',')
                        .Append(FloatText(center.z)).Append(',')
                        .Append(FloatText(Mathf.Sqrt(Mathf.Max(center.w, 0f)) * 1000f)).Append(',')
                        .Append(FloatText(normal.x)).Append(',').Append(FloatText(normal.y)).Append(',')
                        .Append(FloatText(normal.z)).Append(',').Append(FloatText(normal.w)).Append(',')
                        .Append(itemEvidence.x).Append(',')
                        .Append(itemPromotionAuthority.x).Append(',')
                        .Append(itemPromotionAuthority.y).Append(',')
                        .Append(requiredAuthority).Append(',')
                        .Append(itemPromotionAuthority.x >= requiredAuthority ? 1 : 0).Append(',')
                        .Append(itemEvidence.y).Append(',')
                        .Append(itemEvidence.w == 0u ? 0u : itemEvidence.w - 1u).Append(',')
                        .Append(FloatText(view.x)).Append(',').Append(FloatText(view.y)).Append(',')
                        .Append(FloatText(view.z)).Append(',').Append(FloatText(view.w)).Append(',')
                        .Append(gateIdentityValid ? "pending" : "none").Append(',')
                        .Append(gateIdentityValid ? gateState.y - 1u : 0u).Append(',')
                        .Append(gateIdentityValid
                            ? RetirementGateDecisionName(gateState.z) : "none").Append(',')
                        .Append(gateIdentityValid ? gateState.w : 0u).Append(',')
                        .Append(RetirementWitnessFrame(supportA)).Append(',')
                        .Append(RetirementWitnessFrame(supportB)).Append(',')
                        .Append(FloatText(RetirementWitnessPairAngleDeg(
                            supportA, supportB))).Append(',')
                        .Append(RetirementWitnessFrame(freeA)).Append(',')
                        .Append(RetirementWitnessFrame(freeB)).Append(',')
                        .Append(FloatText(RetirementWitnessPairAngleDeg(
                            freeA, freeB))).AppendLine();
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
            int stableContradictionShadowRows = 0;
            int stableRecentSupportVeto16 = 0;
            int stableWouldAllowNoRecentSupport16 = 0;
            int stableNoDirectSupportReceipt = 0;
            int stableInvalidSupportIdentity = 0;
            int stableFutureOnlySupportReceipt = 0;
            int stableFutureReceiptWithOlderValid = 0;
            int confirmedFreeDeleteAuditRows = 0;
            int confirmedFreeDeleteMissingWitness = 0;
            int confirmedFreeDeleteBelowThreshold = 0;
            float confirmedFreeDeleteMinAngleDeg = float.PositiveInfinity;
            float confirmedFreeDeleteMaxAngleDeg = 0f;
            var retirementsCsv = new StringBuilder(Math.Max(4096, retireRetained * 160));
            retirementsCsv.AppendLine("sequence,candidate_index,cell_slot,layer_slot,cell_x,cell_y,cell_z,stable_id,reason,retired_frame,last_seen_frame,last_challenge_frame,direct_footprint_support_frame,direct_footprint_support_age_frames,direct_footprint_support_receipt_seen,direct_footprint_support_identity_valid,direct_footprint_support_frame_order_valid,direct_footprint_future_receipt_seen,legacy_guard16_shadow,successor_index,successor_previous_stable_id,gate_pending_since_frame,gate_decision,gate_same_view_conflict_pairs,gate_identity_valid,support_witness_a_frame,support_witness_b_frame,support_pair_span_deg,free_witness_a_frame,free_witness_b_frame,free_pair_span_deg,center_x_m,center_y_m,center_z_m,sigma_mm,observation_count,was_stable,dual_agree_support,opposition_votes");
            for (uint sequence = firstRetireSequence; sequence < retireTotal; sequence++)
            {
                int ringIndex = (int)(sequence % (uint)RetireAuditCapacity);
                uint4 itemMeta = retireMeta[ringIndex];
                float4 geometry = retireGeometry[ringIndex];
                uint4 itemEvidence = retireEvidence[ringIndex];
                uint4 timing = retireTiming[ringIndex];
                uint4 gateAudit = retireGateState[ringIndex];
                uint4 gateFrames = retireGateWitnessFrames[ringIndex];
                float4 gateAngles = retireGateWitnessAngles[ringIndex];
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

                uint directSupportToken = itemMeta.z == 3u ? 0u : timing.z;
                int directSupportFrame = directSupportToken == 0u
                    ? -1 : (int)(directSupportToken - 1u);
                int directSupportAge = directSupportToken == 0u ||
                    itemMeta.w < directSupportToken - 1u
                    ? -1 : (int)(itemMeta.w - (directSupportToken - 1u));
                bool stableContradiction = itemMeta.z == 1u && itemEvidence.y != 0u;
                uint receiptFlags = stableContradiction ? timing.w : 0u;
                bool recentDirectSupport = (receiptFlags & 1u) != 0u;
                bool supportIdentityValid = (receiptFlags & 2u) != 0u;
                bool supportFrameOrderValid = (receiptFlags & 4u) != 0u;
                bool futureReceiptSeen = (receiptFlags & 8u) != 0u;
                bool anySupportReceiptSeen = (receiptFlags & 16u) != 0u;
                string legacyGuard16Shadow;
                if (!stableContradiction)
                    legacyGuard16Shadow = itemMeta.z == 1u
                        ? "not_stable" : "not_contradiction";
                else if (recentDirectSupport)
                    legacyGuard16Shadow = "veto_recent_support";
                else if (!anySupportReceiptSeen)
                    legacyGuard16Shadow = "no_support_receipt";
                else if (!supportIdentityValid)
                    legacyGuard16Shadow = "invalid_support_identity";
                else if (!supportFrameOrderValid && futureReceiptSeen)
                    legacyGuard16Shadow = "invalid_future_frame";
                else
                    legacyGuard16Shadow = "would_allow_no_recent_support";
                if (stableContradiction)
                {
                    stableContradictionShadowRows++;
                    if (recentDirectSupport) stableRecentSupportVeto16++;
                    else stableWouldAllowNoRecentSupport16++;
                    if (!anySupportReceiptSeen) stableNoDirectSupportReceipt++;
                    else if (!supportIdentityValid) stableInvalidSupportIdentity++;
                    else if (!supportFrameOrderValid && futureReceiptSeen)
                        stableFutureOnlySupportReceipt++;
                    if (futureReceiptSeen && supportFrameOrderValid)
                        stableFutureReceiptWithOlderValid++;
                }
                bool confirmedFreeDelete = stableContradiction &&
                    gateAudit.w != 0u && gateAudit.y == 4u;
                if (confirmedFreeDelete)
                {
                    confirmedFreeDeleteAuditRows++;
                    bool hasFreePair = gateFrames.z != 0u && gateFrames.w != 0u;
                    if (!hasFreePair)
                        confirmedFreeDeleteMissingWitness++;
                    else
                    {
                        confirmedFreeDeleteMinAngleDeg = Mathf.Min(
                            confirmedFreeDeleteMinAngleDeg, gateAngles.y);
                        confirmedFreeDeleteMaxAngleDeg = Mathf.Max(
                            confirmedFreeDeleteMaxAngleDeg, gateAngles.y);
                        if (gateAngles.y + 1e-4f < RetirementGateCrossViewDeg)
                            confirmedFreeDeleteBelowThreshold++;
                    }
                }

                latestFrame = Math.Max(latestFrame, itemMeta.w);
                retirementsCsv.Append(sequence).Append(',').Append(candidateIndex).Append(',')
                    .Append(cell).Append(',').Append(layer).Append(',')
                    .Append(key.x).Append(',').Append(key.y).Append(',').Append(key.z).Append(',')
                    .Append(itemMeta.y).Append(',').Append(retireReason).Append(',')
                    .Append(itemMeta.w).Append(',').Append(timing.x).Append(',')
                    .Append(timing.y == 0u ? 0u : timing.y - 1u).Append(',')
                    .Append(directSupportFrame).Append(',')
                    .Append(directSupportAge).Append(',')
                    .Append(anySupportReceiptSeen ? 1 : 0).Append(',')
                    .Append(supportIdentityValid ? 1 : 0).Append(',')
                    .Append(supportFrameOrderValid ? 1 : 0).Append(',')
                    .Append(futureReceiptSeen ? 1 : 0).Append(',')
                    .Append(legacyGuard16Shadow).Append(',')
                    .Append(itemMeta.z == 3u ? timing.z : 0u).Append(',')
                    .Append(itemMeta.z == 3u ? timing.w : 0u).Append(',')
                    .Append(gateAudit.x == 0u ? -1 : (int)gateAudit.x - 1).Append(',')
                    .Append(gateAudit.w == 0u
                        ? "none" : RetirementGateDecisionName(gateAudit.y)).Append(',')
                    .Append(gateAudit.z).Append(',').Append(gateAudit.w != 0u ? 1 : 0).Append(',')
                    .Append(gateFrames.x == 0u ? -1 : (int)gateFrames.x - 1).Append(',')
                    .Append(gateFrames.y == 0u ? -1 : (int)gateFrames.y - 1).Append(',')
                    .Append(FloatText(gateAngles.x)).Append(',')
                    .Append(gateFrames.z == 0u ? -1 : (int)gateFrames.z - 1).Append(',')
                    .Append(gateFrames.w == 0u ? -1 : (int)gateFrames.w - 1).Append(',')
                    .Append(FloatText(gateAngles.y)).Append(',')
                    .Append(FloatText(geometry.x)).Append(',').Append(FloatText(geometry.y)).Append(',')
                    .Append(FloatText(geometry.z)).Append(',')
                    .Append(FloatText(Mathf.Sqrt(Mathf.Max(geometry.w, 0f)) * 1000f)).Append(',')
                    .Append(itemEvidence.x).Append(',').Append(itemEvidence.y).Append(',')
                    .Append(itemEvidence.z).Append(',').Append(itemEvidence.w).AppendLine();
            }

            uint ghostCreated = ghostCounters.Length > 0 ? ghostCounters[0] : 0u;
            uint ghostOverflow = ghostCounters.Length > 1 ? ghostCounters[1] : 0u;
            int ghostLength = Math.Min(RetirementGhostCapacity,
                Math.Min(ghostIdentity.Length, Math.Min(ghostCenters.Length,
                Math.Min(ghostNormals.Length, Math.Min(ghostOriginCameras.Length,
                Math.Min(ghostStates.Length, Math.Min(ghostClasses.Length,
                ghostObservationStates.Length)))))));
            int ghostObservationLength = Math.Min(ghostObservationMeta.Length,
                Math.Min(ghostObservationCamera.Length, Math.Min(
                    ghostObservationView.Length, ghostObservationResidual.Length)));
            int ghostRetained = (int)Math.Min(ghostCreated, (uint)ghostLength);
            int ghostRows = 0;
            int ghostSupportReappeared = 0;
            int ghostSustainedFree = 0;
            int ghostInconclusive = 0;
            int ghostPendingAtSeal = 0;
            int ghostIndependentObservationRows = 0;
            int ghostIndependentSupportRows = 0;
            int ghostIndependentFreeRows = 0;
            int ghostIndependentMixedRows = 0;
            int ghostIndependentIncompleteRows = 0;
            int ghostIndependentNeutralRows = 0;
            int nearStationaryObservationRows = 0;
            int nearStationarySupportRows = 0;
            int nearStationaryFreeRows = 0;
            int nearStationaryMixedRows = 0;
            int nearStationarySupportFreeFlipGhosts = 0;
            ulong nearStationaryValidSamples = 0;
            double nearStationarySignedResidualSumM = 0.0;
            double nearStationaryAbsoluteResidualSumM = 0.0;
            double nearStationarySquaredResidualSumM2 = 0.0;
            float nearStationaryMaxAbsoluteResidualM = 0f;
            ulong nearStationarySupportValidSamples = 0;
            double nearStationarySupportSignedResidualSumM = 0.0;
            double nearStationarySupportAbsoluteResidualSumM = 0.0;
            double nearStationarySupportSquaredResidualSumM2 = 0.0;
            float nearStationarySupportMaxAbsoluteResidualM = 0f;
            int sameViewTemporalConflictGhosts = 0;
            int sameViewTemporalConflictPairs = 0;
            int sameViewSupportObservationsInvalidated = 0;
            int sameViewFreeObservationsInvalidated = 0;
            int[] retirementLaneRetain =
                new int[RetirementContractShadowAnglesDeg.Length];
            int[] retirementLaneDelete =
                new int[RetirementContractShadowAnglesDeg.Length];
            int[] retirementLaneConflict =
                new int[RetirementContractShadowAnglesDeg.Length];
            int[] retirementLaneInsufficient =
                new int[RetirementContractShadowAnglesDeg.Length];
            var retirementGhostsCsv = new StringBuilder(
                Math.Max(4096, ghostRetained * 256));
            var retirementGhostObservationsCsv = new StringBuilder(
                Math.Max(4096, ghostRetained * 1024));
            var retirementContractShadowCsv = new StringBuilder(
                Math.Max(4096, ghostRetained * 384));
            retirementGhostsCsv.AppendLine("ghost_index,stable_id,candidate_index,retired_frame,last_support_frame,center_x_m,center_y_m,center_z_m,sigma_mm,normal_x,normal_y,normal_z,effective_support,status,evaluated_visible_frames,support_frames,free_frames,mixed_frames,abstain_frames,last_evaluated_frame,origin_camera_x_m,origin_camera_y_m,origin_camera_z_m,independent_observation_frames,max_head_translation_mm,max_lateral_baseline_mm,max_ray_separation_deg,max_support_ray_separation_deg,max_free_ray_separation_deg,support_pair_span_deg,free_pair_span_deg,consistent_support_pair_span_deg,consistent_free_pair_span_deg,same_view_temporal_conflict_pairs,same_view_support_observations_invalidated,same_view_free_observations_invalidated,near_stationary_observations,near_stationary_valid_samples,near_stationary_mean_signed_residual_mm,near_stationary_mean_abs_residual_mm,near_stationary_rms_residual_mm,near_stationary_max_abs_residual_mm,near_stationary_support_free_flip,authority");
            retirementGhostObservationsCsv.AppendLine("ghost_index,stable_id,candidate_index,retired_frame,observation_index,frame,classification,camera_x_m,camera_y_m,camera_z_m,head_translation_mm,lateral_baseline_mm,ray_separation_deg,candidate_range_m,incidence_cosine,motion_quality,valid_samples,total_samples,mean_signed_residual_mm,mean_abs_residual_mm,rms_residual_mm,max_abs_residual_mm,near_stationary_bin,same_view_temporal_conflict,contract_eligible,authority");
            retirementContractShadowCsv.AppendLine("ghost_index,stable_id,retired_frame,legacy_status,angle_threshold_deg,raw_support_pair_span_deg,raw_free_pair_span_deg,consistent_support_pair_span_deg,consistent_free_pair_span_deg,same_view_cone_deg,same_view_temporal_conflict_pairs,support_observations_invalidated,free_observations_invalidated,support_cross_view,free_cross_view,outcome,authority");
            for (int i = 0; i < ghostRetained; i++)
            {
                uint4 identity = ghostIdentity[i];
                if (identity.x == 0u) continue;
                float4 center = ghostCenters[i];
                float4 normal = ghostNormals[i];
                float4 originCamera = ghostOriginCameras[i];
                uint4 state = ghostStates[i];
                uint4 classes = ghostClasses[i];
                uint4 observationState = ghostObservationStates[i];
                int observationCount = Math.Min((int)observationState.x,
                    RetirementGhostObservationCapacity);
                int observationBase = i * RetirementGhostObservationCapacity;
                observationCount = Math.Min(observationCount,
                    Math.Max(0, ghostObservationLength - observationBase));
                float maxHeadTranslationMm = 0f;
                float maxLateralBaselineMm = 0f;
                float maxRaySeparationDeg = 0f;
                float maxSupportRaySeparationDeg = 0f;
                float maxFreeRaySeparationDeg = 0f;
                var supportRays = new Vector3[RetirementGhostObservationCapacity];
                var freeRays = new Vector3[RetirementGhostObservationCapacity];
                var observationRays = new Vector3[RetirementGhostObservationCapacity];
                var observationClassifications =
                    new uint[RetirementGhostObservationCapacity];
                var observationHasRay = new bool[RetirementGhostObservationCapacity];
                var observationTemporalConflict =
                    new bool[RetirementGhostObservationCapacity];
                int supportRayCount = 0;
                int freeRayCount = 0;
                int ghostNearStationaryObservations = 0;
                ulong ghostNearStationaryValidSamples = 0;
                double ghostNearStationarySignedResidualSumM = 0.0;
                double ghostNearStationaryAbsoluteResidualSumM = 0.0;
                double ghostNearStationarySquaredResidualSumM2 = 0.0;
                float ghostNearStationaryMaxAbsoluteResidualM = 0f;
                bool ghostNearStationarySupport = false;
                bool ghostNearStationaryFree = false;
                for (int observation = 0; observation < observationCount; observation++)
                {
                    int observationIndex = observationBase + observation;
                    uint4 observationMeta = ghostObservationMeta[observationIndex];
                    float4 observationCamera = ghostObservationCamera[observationIndex];
                    float4 observationView = ghostObservationView[observationIndex];
                    float4 observationResidual = ghostObservationResidual[observationIndex];
                    uint classification = observationMeta.y;
                    if (classification == 0u) continue;
                    float headTranslationMm = Mathf.Max(0f, observationCamera.w) * 1000f;
                    float lateralBaselineMm = Mathf.Max(0f, observationView.x) * 1000f;
                    float raySeparationDeg = Mathf.Max(0f, observationView.y);
                    float candidateRange = math.distance(
                        new float3(center.x, center.y, center.z),
                        new float3(observationCamera.x, observationCamera.y,
                            observationCamera.z));
                    Vector3 currentRay = new Vector3(center.x - observationCamera.x,
                        center.y - observationCamera.y,
                        center.z - observationCamera.z);
                    if (currentRay.sqrMagnitude > 1e-10f)
                    {
                        currentRay.Normalize();
                        observationRays[observation] = currentRay;
                        observationClassifications[observation] = classification;
                        observationHasRay[observation] = true;
                        if (classification == 1u)
                            supportRays[supportRayCount++] = currentRay;
                        else if (classification == 2u)
                            freeRays[freeRayCount++] = currentRay;
                    }
                    maxHeadTranslationMm = Mathf.Max(maxHeadTranslationMm,
                        headTranslationMm);
                    maxLateralBaselineMm = Mathf.Max(maxLateralBaselineMm,
                        lateralBaselineMm);
                    maxRaySeparationDeg = Mathf.Max(maxRaySeparationDeg,
                        raySeparationDeg);
                    if (classification == 1u)
                    {
                        ghostIndependentSupportRows++;
                        maxSupportRaySeparationDeg = Mathf.Max(
                            maxSupportRaySeparationDeg, raySeparationDeg);
                    }
                    else if (classification == 2u)
                    {
                        ghostIndependentFreeRows++;
                        maxFreeRaySeparationDeg = Mathf.Max(
                            maxFreeRaySeparationDeg, raySeparationDeg);
                    }
                    else if (classification == 3u) ghostIndependentMixedRows++;
                    else if (classification == 4u) ghostIndependentIncompleteRows++;
                    else ghostIndependentNeutralRows++;
                    bool nearStationary = headTranslationMm <=
                        NearStationaryTranslationMm && raySeparationDeg <=
                        NearStationaryRaySeparationDeg;
                    uint validSamples = observationMeta.z;
                    uint totalSamples = observationMeta.w;
                    if (nearStationary)
                    {
                        ghostNearStationaryObservations++;
                        nearStationaryObservationRows++;
                        if (classification == 1u)
                        {
                            ghostNearStationarySupport = true;
                            nearStationarySupportRows++;
                        }
                        else if (classification == 2u)
                        {
                            ghostNearStationaryFree = true;
                            nearStationaryFreeRows++;
                        }
                        else if (classification == 3u)
                            nearStationaryMixedRows++;
                        if (validSamples > 0u)
                        {
                            ghostNearStationaryValidSamples += validSamples;
                            nearStationaryValidSamples += validSamples;
                            ghostNearStationarySignedResidualSumM +=
                                observationResidual.x * validSamples;
                            ghostNearStationaryAbsoluteResidualSumM +=
                                observationResidual.y * validSamples;
                            ghostNearStationarySquaredResidualSumM2 +=
                                observationResidual.z * observationResidual.z * validSamples;
                            nearStationarySignedResidualSumM +=
                                observationResidual.x * validSamples;
                            nearStationaryAbsoluteResidualSumM +=
                                observationResidual.y * validSamples;
                            nearStationarySquaredResidualSumM2 +=
                                observationResidual.z * observationResidual.z * validSamples;
                            ghostNearStationaryMaxAbsoluteResidualM = Mathf.Max(
                                ghostNearStationaryMaxAbsoluteResidualM,
                                observationResidual.w);
                            nearStationaryMaxAbsoluteResidualM = Mathf.Max(
                                nearStationaryMaxAbsoluteResidualM,
                                observationResidual.w);
                            if (classification == 1u)
                            {
                                nearStationarySupportValidSamples += validSamples;
                                nearStationarySupportSignedResidualSumM +=
                                    observationResidual.x * validSamples;
                                nearStationarySupportAbsoluteResidualSumM +=
                                    observationResidual.y * validSamples;
                                nearStationarySupportSquaredResidualSumM2 +=
                                    observationResidual.z * observationResidual.z *
                                    validSamples;
                                nearStationarySupportMaxAbsoluteResidualM = Mathf.Max(
                                    nearStationarySupportMaxAbsoluteResidualM,
                                    observationResidual.w);
                            }
                        }
                    }
                    ghostIndependentObservationRows++;
                    latestFrame = Math.Max(latestFrame, observationMeta.x);
                }
                int ghostSameViewConflictPairs = MarkSameViewTemporalConflicts(
                    observationRays, observationClassifications, observationHasRay,
                    observationCount, observationTemporalConflict,
                    SameViewTemporalConsistencyConeDeg);
                var consistentSupportRays =
                    new Vector3[RetirementGhostObservationCapacity];
                var consistentFreeRays =
                    new Vector3[RetirementGhostObservationCapacity];
                int consistentSupportRayCount = 0;
                int consistentFreeRayCount = 0;
                int ghostSupportInvalidated = 0;
                int ghostFreeInvalidated = 0;
                for (int observation = 0; observation < observationCount; observation++)
                {
                    uint classification = observationClassifications[observation];
                    if (!observationHasRay[observation]) continue;
                    if (observationTemporalConflict[observation])
                    {
                        if (classification == 1u) ghostSupportInvalidated++;
                        else if (classification == 2u) ghostFreeInvalidated++;
                        continue;
                    }
                    if (classification == 1u)
                        consistentSupportRays[consistentSupportRayCount++] =
                            observationRays[observation];
                    else if (classification == 2u)
                        consistentFreeRays[consistentFreeRayCount++] =
                            observationRays[observation];
                }
                if (ghostSameViewConflictPairs > 0)
                    sameViewTemporalConflictGhosts++;
                sameViewTemporalConflictPairs += ghostSameViewConflictPairs;
                sameViewSupportObservationsInvalidated += ghostSupportInvalidated;
                sameViewFreeObservationsInvalidated += ghostFreeInvalidated;

                for (int observation = 0; observation < observationCount; observation++)
                {
                    int observationIndex = observationBase + observation;
                    uint4 observationMeta = ghostObservationMeta[observationIndex];
                    uint classification = observationMeta.y;
                    if (classification == 0u) continue;
                    float4 observationCamera = ghostObservationCamera[observationIndex];
                    float4 observationView = ghostObservationView[observationIndex];
                    float4 observationResidual = ghostObservationResidual[observationIndex];
                    float headTranslationMm = Mathf.Max(0f, observationCamera.w) *
                        1000f;
                    float lateralBaselineMm = Mathf.Max(0f, observationView.x) *
                        1000f;
                    float raySeparationDeg = Mathf.Max(0f, observationView.y);
                    float candidateRange = math.distance(
                        new float3(center.x, center.y, center.z),
                        new float3(observationCamera.x, observationCamera.y,
                            observationCamera.z));
                    bool nearStationary = headTranslationMm <=
                        NearStationaryTranslationMm && raySeparationDeg <=
                        NearStationaryRaySeparationDeg;
                    bool contractEligible = observationHasRay[observation] &&
                        !observationTemporalConflict[observation] &&
                        (classification == 1u || classification == 2u);
                    retirementGhostObservationsCsv.Append(i).Append(',')
                        .Append(identity.x).Append(',').Append(identity.y).Append(',')
                        .Append(identity.z).Append(',').Append(observation).Append(',')
                        .Append(observationMeta.x).Append(',')
                        .Append(RetirementGhostClassificationName(classification)).Append(',')
                        .Append(FloatText(observationCamera.x)).Append(',')
                        .Append(FloatText(observationCamera.y)).Append(',')
                        .Append(FloatText(observationCamera.z)).Append(',')
                        .Append(FloatText(headTranslationMm)).Append(',')
                        .Append(FloatText(lateralBaselineMm)).Append(',')
                        .Append(FloatText(raySeparationDeg)).Append(',')
                        .Append(FloatText(candidateRange)).Append(',')
                        .Append(FloatText(observationView.z)).Append(',')
                        .Append(FloatText(observationView.w)).Append(',')
                        .Append(observationMeta.z).Append(',')
                        .Append(observationMeta.w).Append(',')
                        .Append(FloatText(observationResidual.x * 1000f)).Append(',')
                        .Append(FloatText(observationResidual.y * 1000f)).Append(',')
                        .Append(FloatText(observationResidual.z * 1000f)).Append(',')
                        .Append(FloatText(observationResidual.w * 1000f)).Append(',')
                        .Append(nearStationary ? 1 : 0).Append(',')
                        .Append(observationTemporalConflict[observation] ? 1 : 0)
                        .Append(',').Append(contractEligible ? 1 : 0).Append(',')
                        .Append("shadow_only_zero_production_authority").AppendLine();
                }
                float supportPairSpanDeg = MaximumPairwiseRayAngleDeg(
                    supportRays, supportRayCount);
                float freePairSpanDeg = MaximumPairwiseRayAngleDeg(
                    freeRays, freeRayCount);
                float consistentSupportPairSpanDeg = MaximumPairwiseRayAngleDeg(
                    consistentSupportRays, consistentSupportRayCount);
                float consistentFreePairSpanDeg = MaximumPairwiseRayAngleDeg(
                    consistentFreeRays, consistentFreeRayCount);
                bool nearStationarySupportFreeFlip = ghostNearStationarySupport &&
                    ghostNearStationaryFree;
                if (nearStationarySupportFreeFlip)
                    nearStationarySupportFreeFlipGhosts++;
                float ghostNearStationaryMeanSignedResidualMm =
                    ghostNearStationaryValidSamples == 0 ? 0f : (float)(
                        ghostNearStationarySignedResidualSumM /
                        ghostNearStationaryValidSamples * 1000.0);
                float ghostNearStationaryMeanAbsResidualMm =
                    ghostNearStationaryValidSamples == 0 ? 0f : (float)(
                        ghostNearStationaryAbsoluteResidualSumM /
                        ghostNearStationaryValidSamples * 1000.0);
                float ghostNearStationaryRmsResidualMm =
                    ghostNearStationaryValidSamples == 0 ? 0f : (float)(
                        Math.Sqrt(ghostNearStationarySquaredResidualSumM2 /
                        ghostNearStationaryValidSamples) * 1000.0);
                string status = state.x == 2u ? "support_reappeared" :
                    (state.x == 3u ? "sustained_free" :
                    (state.x == 4u ? "inconclusive" : "pending_at_seal"));
                if (state.x == 2u) ghostSupportReappeared++;
                else if (state.x == 3u) ghostSustainedFree++;
                else if (state.x == 4u) ghostInconclusive++;
                else ghostPendingAtSeal++;
                ghostRows++;
                latestFrame = Math.Max(latestFrame, identity.z);
                if (state.z != 0u) latestFrame = Math.Max(latestFrame, state.z - 1u);
                retirementGhostsCsv.Append(i).Append(',')
                    .Append(identity.x).Append(',').Append(identity.y).Append(',')
                    .Append(identity.z).Append(',').Append(identity.w).Append(',')
                    .Append(FloatText(center.x)).Append(',')
                    .Append(FloatText(center.y)).Append(',')
                    .Append(FloatText(center.z)).Append(',')
                    .Append(FloatText(Mathf.Sqrt(Mathf.Max(center.w, 0f)) * 1000f)).Append(',')
                    .Append(FloatText(normal.x)).Append(',')
                    .Append(FloatText(normal.y)).Append(',')
                    .Append(FloatText(normal.z)).Append(',')
                    .Append(FloatText(normal.w)).Append(',')
                    .Append(status).Append(',').Append(state.y).Append(',')
                    .Append(classes.x).Append(',').Append(classes.y).Append(',')
                    .Append(classes.z).Append(',').Append(classes.w).Append(',')
                    .Append(state.z == 0u ? -1 : (long)state.z - 1L).Append(',')
                    .Append(FloatText(originCamera.x)).Append(',')
                    .Append(FloatText(originCamera.y)).Append(',')
                    .Append(FloatText(originCamera.z)).Append(',')
                    .Append(observationCount).Append(',')
                    .Append(FloatText(maxHeadTranslationMm)).Append(',')
                    .Append(FloatText(maxLateralBaselineMm)).Append(',')
                    .Append(FloatText(maxRaySeparationDeg)).Append(',')
                    .Append(FloatText(maxSupportRaySeparationDeg)).Append(',')
                    .Append(FloatText(maxFreeRaySeparationDeg)).Append(',')
                    .Append(FloatText(supportPairSpanDeg)).Append(',')
                    .Append(FloatText(freePairSpanDeg)).Append(',')
                    .Append(FloatText(consistentSupportPairSpanDeg)).Append(',')
                    .Append(FloatText(consistentFreePairSpanDeg)).Append(',')
                    .Append(ghostSameViewConflictPairs).Append(',')
                    .Append(ghostSupportInvalidated).Append(',')
                    .Append(ghostFreeInvalidated).Append(',')
                    .Append(ghostNearStationaryObservations).Append(',')
                    .Append(ghostNearStationaryValidSamples).Append(',')
                    .Append(FloatText(ghostNearStationaryMeanSignedResidualMm)).Append(',')
                    .Append(FloatText(ghostNearStationaryMeanAbsResidualMm)).Append(',')
                    .Append(FloatText(ghostNearStationaryRmsResidualMm)).Append(',')
                    .Append(FloatText(ghostNearStationaryMaxAbsoluteResidualM * 1000f)).Append(',')
                    .Append(nearStationarySupportFreeFlip ? 1 : 0).Append(',')
                    .Append("shadow_only_zero_production_authority").AppendLine();

                for (int lane = 0; lane < RetirementContractShadowAnglesDeg.Length;
                     lane++)
                {
                    float angleThresholdDeg = RetirementContractShadowAnglesDeg[lane];
                    bool supportCrossView = consistentSupportPairSpanDeg >=
                        angleThresholdDeg;
                    bool freeCrossView = consistentFreePairSpanDeg >=
                        angleThresholdDeg;
                    string outcome = RetirementContractShadowOutcome(
                        supportCrossView, freeCrossView);
                    if (supportCrossView && freeCrossView)
                        retirementLaneConflict[lane]++;
                    else if (supportCrossView)
                        retirementLaneRetain[lane]++;
                    else if (freeCrossView)
                        retirementLaneDelete[lane]++;
                    else
                        retirementLaneInsufficient[lane]++;
                    retirementContractShadowCsv.Append(i).Append(',')
                        .Append(identity.x).Append(',').Append(identity.z).Append(',')
                        .Append(status).Append(',')
                        .Append(FloatText(angleThresholdDeg)).Append(',')
                        .Append(FloatText(supportPairSpanDeg)).Append(',')
                        .Append(FloatText(freePairSpanDeg)).Append(',')
                        .Append(FloatText(consistentSupportPairSpanDeg)).Append(',')
                        .Append(FloatText(consistentFreePairSpanDeg)).Append(',')
                        .Append(FloatText(SameViewTemporalConsistencyConeDeg)).Append(',')
                        .Append(ghostSameViewConflictPairs).Append(',')
                        .Append(ghostSupportInvalidated).Append(',')
                        .Append(ghostFreeInvalidated).Append(',')
                        .Append(supportCrossView ? 1 : 0).Append(',')
                        .Append(freeCrossView ? 1 : 0).Append(',')
                        .Append(outcome).Append(',')
                        .Append("shadow_only_zero_production_authority").AppendLine();
                }
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

            uint promotionTotal = promotionCounters.Length > 0 ? promotionCounters[0] : 0u;
            int promotionRetained = (int)Math.Min(promotionTotal,
                (uint)PromotionAuditCapacity);
            uint firstPromotionSequence = promotionTotal - (uint)promotionRetained;
            int birthEdgeRisk = 0;
            int promotionEdgeRisk = 0;
            int birthTemporalRisk = 0;
            int promotionTemporalRisk = 0;
            int birthOuterView = 0;
            int promotionOuterView = 0;
            int birthLowMotionQuality = 0;
            int promotionLowMotionQuality = 0;
            int birthAndPromotionEdgeRisk = 0;
            var promotionsCsv = new StringBuilder(Math.Max(8192,
                promotionRetained * 320));
            promotionsCsv.AppendLine("sequence,candidate_index,stable_id,birth_frame,promotion_frame,center_x_m,center_y_m,center_z_m,promotion_sigma_mm,birth_platform_frame,birth_pixel_x,birth_pixel_y,birth_view_radius,birth_angular_deg_s,birth_linear_m_s,birth_motion_quality,birth_range_m,birth_observation_quality,birth_edge_reason_bits,birth_edge_reason_labels,birth_temporal_reason,birth_temporal_reason_label,birth_raw_processed_delta_mm,birth_observation_sigma_mm,promotion_platform_frame,promotion_pixel_x,promotion_pixel_y,promotion_view_radius,promotion_angular_deg_s,promotion_linear_m_s,promotion_motion_quality,promotion_range_m,promotion_observation_quality,promotion_edge_reason_bits,promotion_edge_reason_labels,promotion_temporal_reason,promotion_temporal_reason_label,promotion_raw_processed_delta_mm,promotion_observation_sigma_mm,promotion_authority_support_count,hard_edge_or_unrescued_grazing_support_count,temporal_risk_support_count,outer_or_motion_risk_support_count,observation_count,dual_agree_support,view_spread_deg,promotion_authority_frames,observed_frames,required_authority_frames");
            for (uint sequence = firstPromotionSequence;
                 sequence < promotionTotal; sequence++)
            {
                int ringIndex = (int)(sequence % (uint)PromotionAuditCapacity);
                uint4 itemMeta = promotionMetas[ringIndex];
                float4 geometry = promotionGeometry[ringIndex];
                uint4 birthSource = promotionBirthSources[ringIndex];
                uint4 itemSource = promotionSources[ringIndex];
                float4 birthMotion = promotionBirthMotion[ringIndex];
                float4 itemMotion = promotionMotion[ringIndex];
                float4 deltaSigma = promotionDelta[ringIndex];
                float4 qualityRange = promotionQualityRange[ringIndex];
                uint4 support = promotionSupport[ringIndex];
                uint4 authority = promotionAuthority[ringIndex];
                uint birthPackedPixel = birthSource.x;
                uint itemPackedPixel = itemSource.x;
                bool birthEdge = HasPromotionHardEdgeRisk(birthSource.y);
                bool itemEdge = HasPromotionHardEdgeRisk(itemSource.y);
                bool birthTemporal = birthSource.z != 0u && birthSource.z != 6u;
                bool itemTemporal = itemSource.z != 0u && itemSource.z != 6u;
                if (birthEdge) birthEdgeRisk++;
                if (itemEdge) promotionEdgeRisk++;
                if (birthEdge && itemEdge) birthAndPromotionEdgeRisk++;
                if (birthTemporal) birthTemporalRisk++;
                if (itemTemporal) promotionTemporalRisk++;
                if (birthMotion.w >= 0.70f) birthOuterView++;
                if (itemMotion.w >= 0.70f) promotionOuterView++;
                if (birthMotion.z < 0.70f) birthLowMotionQuality++;
                if (itemMotion.z < 0.70f) promotionLowMotionQuality++;
                uint packedClasses = support.x;
                latestFrame = Math.Max(latestFrame, itemMeta.w);
                promotionsCsv.Append(sequence).Append(',')
                    .Append(itemMeta.x).Append(',').Append(itemMeta.y).Append(',')
                    .Append(itemMeta.z).Append(',').Append(itemMeta.w).Append(',')
                    .Append(FloatText(geometry.x)).Append(',')
                    .Append(FloatText(geometry.y)).Append(',')
                    .Append(FloatText(geometry.z)).Append(',')
                    .Append(FloatText(Mathf.Sqrt(Mathf.Max(geometry.w, 0f)) * 1000f)).Append(',')
                    .Append(birthSource.w).Append(',')
                    .Append(birthPackedPixel & 0xffffu).Append(',')
                    .Append(birthPackedPixel >> 16).Append(',')
                    .Append(FloatText(birthMotion.w)).Append(',')
                    .Append(FloatText(birthMotion.x)).Append(',')
                    .Append(FloatText(birthMotion.y)).Append(',')
                    .Append(FloatText(birthMotion.z)).Append(',')
                    .Append(FloatText(qualityRange.x)).Append(',')
                    .Append(FloatText(qualityRange.z)).Append(',')
                    .Append(birthSource.y).Append(',')
                    .Append(EdgeReasonLabels(birthSource.y)).Append(',')
                    .Append(birthSource.z).Append(',')
                    .Append(TemporalReasonName(birthSource.z)).Append(',')
                    .Append(FloatText(deltaSigma.x * 1000f)).Append(',')
                    .Append(FloatText(deltaSigma.z * 1000f)).Append(',')
                    .Append(itemSource.w).Append(',')
                    .Append(itemPackedPixel & 0xffffu).Append(',')
                    .Append(itemPackedPixel >> 16).Append(',')
                    .Append(FloatText(itemMotion.w)).Append(',')
                    .Append(FloatText(itemMotion.x)).Append(',')
                    .Append(FloatText(itemMotion.y)).Append(',')
                    .Append(FloatText(itemMotion.z)).Append(',')
                    .Append(FloatText(qualityRange.y)).Append(',')
                    .Append(FloatText(qualityRange.w)).Append(',')
                    .Append(itemSource.y).Append(',')
                    .Append(EdgeReasonLabels(itemSource.y)).Append(',')
                    .Append(itemSource.z).Append(',')
                    .Append(TemporalReasonName(itemSource.z)).Append(',')
                    .Append(FloatText(deltaSigma.y * 1000f)).Append(',')
                    .Append(FloatText(deltaSigma.w * 1000f)).Append(',')
                    .Append(packedClasses & 0xffu).Append(',')
                    .Append((packedClasses >> 8) & 0xffu).Append(',')
                    .Append((packedClasses >> 16) & 0xffu).Append(',')
                    .Append((packedClasses >> 24) & 0xffu).Append(',')
                    .Append(support.y).Append(',').Append(support.z).Append(',')
                    .Append(FloatText(math.asfloat(support.w))).Append(',')
                    .Append(authority.x).Append(',').Append(authority.y).Append(',')
                    .Append(authority.z).AppendLine();
            }

            uint totalValid = AuditTotal(totals, 0);
            uint totalDualAgree = AuditTotal(totals, 1);
            uint totalDualDisagree = AuditTotal(totals, 2);
            uint totalRawUnavailable = AuditTotal(totals, 3);
            uint totalChallengeVotes = AuditTotal(totals, 4);
            uint totalContradictionRetired = AuditTotal(totals, 5);
            uint totalEmbryoExpired = AuditTotal(totals, 6);
            uint totalStableIdsAssigned = AuditTotal(totals, 7);
            uint totalCandidateSuccessions = AuditTotal(totals, 8);
            uint totalPromotionAuthorityDeferred = AuditTotal(totals, 9);
            uint totalRiskGeometryQuarantined = AuditTotal(totals, 10);
            uint totalSafeGeometryReanchors = AuditTotal(totals, 11);
            uint totalStableRiskGeometryBlocked = AuditTotal(totals, 12);
            uint retirementPendingStarts = AuditTotal(retirementGateCounters, 0);
            uint retirementSupportVetoes = AuditTotal(retirementGateCounters, 1);
            uint retirementConfirmedFreeDeletes = AuditTotal(
                retirementGateCounters, 2);
            uint retirementConflictWaitEvaluations = AuditTotal(
                retirementGateCounters, 3);
            uint retirementInsufficientWaitEvaluations = AuditTotal(
                retirementGateCounters, 4);
            uint retirementSameViewConflictPairs = AuditTotal(
                retirementGateCounters, 5);
            uint retirementSupportReceipts = AuditTotal(retirementGateCounters, 6);
            uint retirementFreeReceipts = AuditTotal(retirementGateCounters, 7);
            uint retirementOrdinarySupportCancellations = AuditTotal(
                retirementGateCounters, 8);
            uint retirementSuccessionCancellations = AuditTotal(
                retirementGateCounters, 9);
            uint retirementIdentityRestartCancellations = AuditTotal(
                retirementGateCounters, 10);
            ulong retirementAccountedGateExits =
                (ulong)retirementSupportVetoes + retirementConfirmedFreeDeletes +
                retirementOrdinarySupportCancellations +
                retirementSuccessionCancellations +
                retirementIdentityRestartCancellations;
            long retirementGateBalanceDelta = (long)retirementPendingStarts -
                (long)retirementAccountedGateExits - pendingRetirementCandidates;
            float confirmedFreeDeleteMinAngleOutput =
                confirmedFreeDeleteAuditRows == 0 ||
                float.IsPositiveInfinity(confirmedFreeDeleteMinAngleDeg)
                ? 0f : confirmedFreeDeleteMinAngleDeg;
            float nearStationaryMeanSignedResidualMm = nearStationaryValidSamples == 0
                ? 0f : (float)(nearStationarySignedResidualSumM /
                    nearStationaryValidSamples * 1000.0);
            float nearStationaryMeanAbsResidualMm = nearStationaryValidSamples == 0
                ? 0f : (float)(nearStationaryAbsoluteResidualSumM /
                    nearStationaryValidSamples * 1000.0);
            float nearStationaryRmsResidualMm = nearStationaryValidSamples == 0
                ? 0f : (float)(Math.Sqrt(nearStationarySquaredResidualSumM2 /
                    nearStationaryValidSamples) * 1000.0);
            float nearStationarySupportMeanSignedResidualMm =
                nearStationarySupportValidSamples == 0 ? 0f : (float)(
                    nearStationarySupportSignedResidualSumM /
                    nearStationarySupportValidSamples * 1000.0);
            float nearStationarySupportMeanAbsResidualMm =
                nearStationarySupportValidSamples == 0 ? 0f : (float)(
                    nearStationarySupportAbsoluteResidualSumM /
                    nearStationarySupportValidSamples * 1000.0);
            float nearStationarySupportRmsResidualMm =
                nearStationarySupportValidSamples == 0 ? 0f : (float)(
                    Math.Sqrt(nearStationarySupportSquaredResidualSumM2 /
                    nearStationarySupportValidSamples) * 1000.0);
            var retirementContractShadowSummary = new StringBuilder(512);
            for (int lane = 0; lane < RetirementContractShadowAnglesDeg.Length;
                 lane++)
            {
                if (lane > 0) retirementContractShadowSummary.Append(",\n");
                retirementContractShadowSummary.Append("      {\"angle_threshold_deg\": ")
                    .Append(FloatText(RetirementContractShadowAnglesDeg[lane]))
                    .Append(", \"retain_cross_view_support\": ")
                    .Append(retirementLaneRetain[lane])
                    .Append(", \"delete_cross_view_free\": ")
                    .Append(retirementLaneDelete[lane])
                    .Append(", \"defer_conflict\": ")
                    .Append(retirementLaneConflict[lane])
                    .Append(", \"defer_insufficient_view_diversity\": ")
                    .Append(retirementLaneInsufficient[lane]).Append('}');
            }
            var summary = new StringBuilder(6144);
            summary.AppendLine("{")
                .Append("  \"schema\": \"scancover.gungel_candidate_audit.v16\",\n")
                .Append("  \"reason\": \"").Append(JsonEscape(reason)).Append("\",\n")
                .Append("  \"captured_utc\": \"").Append(capturedUtc.ToString("O", CultureInfo.InvariantCulture)).Append("\",\n")
                .Append("  \"latest_frame\": ").Append(latestFrame).Append(",\n")
                .Append("  \"cell_size_m\": ").Append(FloatText(_cellSize)).Append(",\n")
                .Append("  \"semantics\": {\n")
                .Append("    \"coordinates\": \"world_meters\",\n")
                .Append("    \"cumulative_dual_disagree_scope\": \"global_counter\",\n")
                .Append("    \"court_wave_scope\": \"nearest_stable_id_before_candidate_update\",\n")
                .Append("    \"court_wave_provenance\": \"latched_pixel_platform_frame_raw_processed_motion_edge_temporal\",\n")
                .Append("    \"promotion_provenance\": \"candidate_birth_and_first_stable_promotion_pixel_platform_frame_raw_processed_motion_view_edge_temporal_support_mix\",\n")
                .Append("    \"promotion_provenance_contract\": \"audit receipts remain write-only; the per-candidate distinct-frame authority ledger controls safe geometry writes and first stable promotion\",\n")
                .Append("    \"stable_promotion_contract\": \"legacy safe-geometry and dual maturity plus at least two distinct non-edge non-temporal-risk non-outer motion-qualified frames; no lifetime percentage veto; the promotion observation itself must be safe\",\n")
                .Append("    \"geometry_write_contract\": \"dual-agree risky evidence may prove existence but cannot write center normal thickness or refresh the geometry lease; the first safe observation atomically reanchors provisional geometry before any EMA\",\n")
                .Append("    \"candidate_last_seen_frame_semantics\": \"last safe geometry lease frame; a risky provisional birth uses its birth frame until the first safe reanchor\",\n")
                .Append("    \"edge_reason_bits\": \"bit0 span_jump; bit1 gap; bit2 skirt; bit3 plane_trusted; bit4 grazing; bit5 dual_cluster; bit6 cross_eye; bit7 rejected_edge; bit8 grazing_span; bit9 grazing_plane_supported; bit10 grazing_rescued\",\n")
                .Append("    \"promotion_edge_contract\": \"span_jump gap dual_cluster and grazing_span are review context only; hard veto is skirt cross_eye rejected_edge or grazing without grazing_rescued\",\n")
                .Append("    \"temporal_reason_codes\": \"0 unavailable; 1 first_frame; 2 current_invalid; 3 previous_fov_miss; 4 history_invalid; 5 changed; 6 stable\",\n")
                .Append("    \"fusion_residual_limit\": \"max(0.012m,2.5*combined_sigma)\",\n")
                .Append("    \"fusion_contract\": \"court_wave_must_not_hold_fusion_authority\",\n")
                .Append("    \"candidate_transaction_contract\": \"frame_adjudication_before_candidate_mutation; underconstrained_bootstrap_updates_only; other_rejected_frames_diagnostic_only\",\n")
                .Append("    \"stable_update_contract\": \"stable_candidate_accepts_geometry_ema_only from safe observations inside fusion residual limit; risky or outside-band evidence may only maintain existence or grow an independent challenger\",\n")
                .Append("    \"retire_reason_1\": \"three consecutive reliable free-space votes opened pending retirement, then two post-pending universal-free direct45 witnesses separated by at least 1deg authorized deletion\",\n")
                .Append("    \"retire_reason_2\": \"unpromoted_embryo_expired\",\n")
                .Append("    \"retire_reason_3\": \"local_coplanar_consensus_successor_inherits_stable_id\",\n")
                .Append("    \"succession_contract\": \"current_dual_mature_candidate; stale_incumbent; same_normal_and_local_footprint; challenger_consensus_at_least_3_and_margin_2\",\n")
                .Append("    \"legacy_retirement_guard16_shadow\": \"at each actual stable contradiction retirement, select the newest of two identity-stamped direct45 support receipts whose frame is not newer than the retirement frame; support within the prior 16 fusion frames records would-veto; all receipts have zero production authority\",\n")
                .Append("    \"retirement_ghost_shadow\": \"actual stable contradiction retirements with valid recent support are copied into a separate zero-authority ghost buffer; the legacy result still resolves from up to eight later valid visible frames, while a separate fixed ledger continues for up to 32 classified observations and records camera geometry, incidence, motion, valid footprint samples and candidate-relative depth residuals\",\n")
                .Append("    \"near_stationary_repeatability\": \"camera displacement no more than 10mm and candidate-ray separation no more than 0.25deg; all-class residual includes intentional free-space separation, while support-only residual is the candidate-relative repeat precision; neither is absolute geometric accuracy or a production threshold\",\n")
                .Append("    \"same_view_temporal_consistency\": \"inside the production pending-retirement gate, opposite support/free receipts whose candidate rays differ by no more than 0.25deg invalidate each other; the legacy ghost and 0.5/1/2deg shadow lanes apply the same principle without production authority\",\n")
                .Append("    \"retirement_production_gate\": \"three opposition votes only open pending retirement; post-pending direct45 support/free receipts are sampled every fourth fusion frame but committed only when that frame is accepted; cross-view support at least 1deg cancels retirement and clears opposition, cross-view universal-free at least 1deg authorizes deletion, conflict or insufficient diversity keeps the candidate; every gate exit reason is counted and every actual retirement snapshots its gate witnesses\",\n")
                .Append("    \"retirement_contract_shadow\": \"0.5deg 1deg and 2deg zero-authority lanes; only observations that survive same-view temporal consistency may form two later same-class mutually separated witnesses; no room-specific positions or stable IDs are used\"\n")
                .Append("  },\n")
                .Append("  \"candidate_snapshot\": {\n")
                .Append("    \"active_cells\": ").Append(activeCells).Append(",\n")
                .Append("    \"multi_candidate_cells\": ").Append(multiCandidateCells).Append(",\n")
                .Append("    \"active_candidates\": ").Append(activeCandidates).Append(",\n")
                .Append("    \"stable_candidates\": ").Append(stableCandidates).Append(",\n")
                .Append("    \"embryo_candidates\": ").Append(embryoCandidates).Append(",\n")
                .Append("    \"stable_id_missing\": ").Append(stableIdMissing).Append(",\n")
                .Append("    \"pending_retirement_candidates\": ").Append(pendingRetirementCandidates).Append(",\n")
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
                .Append("    \"candidate_successions\": ").Append(totalCandidateSuccessions).Append(",\n")
                .Append("    \"promotion_authority_deferred_evaluations\": ").Append(totalPromotionAuthorityDeferred).Append(",\n")
                .Append("    \"risk_geometry_quarantined_observations\": ").Append(totalRiskGeometryQuarantined).Append(",\n")
                .Append("    \"safe_geometry_reanchors\": ").Append(totalSafeGeometryReanchors).Append(",\n")
                .Append("    \"stable_risk_geometry_updates_blocked\": ").Append(totalStableRiskGeometryBlocked).Append("\n")
                .Append("  },\n")
                .Append("  \"candidate_transactions\": {\n")
                .Append("    \"committed_frames\": ").Append(_candidateTransactionsCommitted).Append(",\n")
                .Append("    \"bootstrap_observation_frames\": ").Append(_candidateBootstrapTransactions).Append(",\n")
                .Append("    \"discarded_frames\": ").Append(_candidateTransactionsDiscarded).Append(",\n")
                .Append("    \"duplicate_or_conflicting_attempts\": ").Append(_candidateTransactionDuplicateAttempts).Append(",\n")
                .Append("    \"rejected_frame_full_candidate_mutations\": 0\n")
                .Append("  },\n")
                .Append("  \"retirement_production_gate\": {\n")
                .Append("    \"authority\": \"production_candidate_retirement_only\",\n")
                .Append("    \"same_view_cone_deg\": ").Append(FloatText(SameViewTemporalConsistencyConeDeg)).Append(",\n")
                .Append("    \"cross_view_threshold_deg\": ").Append(FloatText(RetirementGateCrossViewDeg)).Append(",\n")
                .Append("    \"direct_projection_stride_frames\": ").Append(DirectProjectionFrameStride).Append(",\n")
                .Append("    \"pending_candidates_at_seal\": ").Append(pendingRetirementCandidates).Append(",\n")
                .Append("    \"pending_starts\": ").Append(retirementPendingStarts).Append(",\n")
                .Append("    \"support_vetoes\": ").Append(retirementSupportVetoes).Append(",\n")
                .Append("    \"confirmed_free_deletes\": ").Append(retirementConfirmedFreeDeletes).Append(",\n")
                .Append("    \"conflict_wait_evaluations\": ").Append(retirementConflictWaitEvaluations).Append(",\n")
                .Append("    \"insufficient_wait_evaluations\": ").Append(retirementInsufficientWaitEvaluations).Append(",\n")
                .Append("    \"same_view_conflict_pairs\": ").Append(retirementSameViewConflictPairs).Append(",\n")
                .Append("    \"support_receipts\": ").Append(retirementSupportReceipts).Append(",\n")
                .Append("    \"free_receipts\": ").Append(retirementFreeReceipts).Append(",\n")
                .Append("    \"ordinary_support_cancellations\": ").Append(retirementOrdinarySupportCancellations).Append(",\n")
                .Append("    \"succession_cancellations\": ").Append(retirementSuccessionCancellations).Append(",\n")
                .Append("    \"identity_restart_cancellations\": ").Append(retirementIdentityRestartCancellations).Append(",\n")
                .Append("    \"accounted_gate_exits\": ").Append(retirementAccountedGateExits).Append(",\n")
                .Append("    \"gate_balance_delta\": ").Append(retirementGateBalanceDelta).Append(",\n")
                .Append("    \"confirmed_free_delete_audit_rows\": ").Append(confirmedFreeDeleteAuditRows).Append(",\n")
                .Append("    \"confirmed_free_delete_missing_witness_pair\": ").Append(confirmedFreeDeleteMissingWitness).Append(",\n")
                .Append("    \"confirmed_free_delete_below_angle_threshold\": ").Append(confirmedFreeDeleteBelowThreshold).Append(",\n")
                .Append("    \"confirmed_free_delete_min_pair_angle_deg\": ").Append(FloatText(confirmedFreeDeleteMinAngleOutput)).Append(",\n")
                .Append("    \"confirmed_free_delete_max_pair_angle_deg\": ").Append(FloatText(confirmedFreeDeleteMaxAngleDeg)).Append("\n")
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
                .Append("  \"legacy_retirement_guard16_shadow\": {\n")
                .Append("    \"authority\": \"shadow_only_zero_production_authority\",\n")
                .Append("    \"support_window_frames\": 16,\n")
                .Append("    \"stable_contradiction_rows\": ").Append(stableContradictionShadowRows).Append(",\n")
                .Append("    \"veto_recent_support\": ").Append(stableRecentSupportVeto16).Append(",\n")
                .Append("    \"would_allow_no_recent_support\": ").Append(stableWouldAllowNoRecentSupport16).Append(",\n")
                .Append("    \"no_direct_support_receipt\": ").Append(stableNoDirectSupportReceipt).Append(",\n")
                .Append("    \"invalid_support_identity\": ").Append(stableInvalidSupportIdentity).Append(",\n")
                .Append("    \"invalid_future_only_receipt\": ").Append(stableFutureOnlySupportReceipt).Append(",\n")
                .Append("    \"future_receipt_with_older_valid_selected\": ").Append(stableFutureReceiptWithOlderValid).Append("\n")
                .Append("  },\n")
                .Append("  \"retirement_ghost_shadow\": {\n")
                .Append("    \"authority\": \"shadow_only_zero_production_authority\",\n")
                .Append("    \"settle_visible_frames\": 8,\n")
                .Append("    \"independent_observation_capacity\": ").Append(RetirementGhostObservationCapacity).Append(",\n")
                .Append("    \"independent_view_threshold\": \"none_feature_ledger_only\",\n")
                .Append("    \"capacity\": ").Append(RetirementGhostCapacity).Append(",\n")
                .Append("    \"total_created\": ").Append(ghostCreated).Append(",\n")
                .Append("    \"retained_rows\": ").Append(ghostRows).Append(",\n")
                .Append("    \"overflow\": ").Append(ghostOverflow).Append(",\n")
                .Append("    \"support_reappeared\": ").Append(ghostSupportReappeared).Append(",\n")
                .Append("    \"sustained_free\": ").Append(ghostSustainedFree).Append(",\n")
                .Append("    \"inconclusive\": ").Append(ghostInconclusive).Append(",\n")
                .Append("    \"pending_at_seal\": ").Append(ghostPendingAtSeal).Append(",\n")
                .Append("    \"independent_observation_rows\": ").Append(ghostIndependentObservationRows).Append(",\n")
                .Append("    \"independent_support_rows\": ").Append(ghostIndependentSupportRows).Append(",\n")
                .Append("    \"independent_free_rows\": ").Append(ghostIndependentFreeRows).Append(",\n")
                .Append("    \"independent_mixed_rows\": ").Append(ghostIndependentMixedRows).Append(",\n")
                .Append("    \"independent_incomplete_rows\": ").Append(ghostIndependentIncompleteRows).Append(",\n")
                .Append("    \"independent_neutral_rows\": ").Append(ghostIndependentNeutralRows).Append(",\n")
                .Append("    \"near_stationary_translation_limit_mm\": ").Append(FloatText(NearStationaryTranslationMm)).Append(",\n")
                .Append("    \"near_stationary_ray_separation_limit_deg\": ").Append(FloatText(NearStationaryRaySeparationDeg)).Append(",\n")
                .Append("    \"near_stationary_observation_rows\": ").Append(nearStationaryObservationRows).Append(",\n")
                .Append("    \"near_stationary_support_rows\": ").Append(nearStationarySupportRows).Append(",\n")
                .Append("    \"near_stationary_free_rows\": ").Append(nearStationaryFreeRows).Append(",\n")
                .Append("    \"near_stationary_mixed_rows\": ").Append(nearStationaryMixedRows).Append(",\n")
                .Append("    \"near_stationary_support_free_flip_ghosts\": ").Append(nearStationarySupportFreeFlipGhosts).Append(",\n")
                .Append("    \"near_stationary_valid_samples\": ").Append(nearStationaryValidSamples).Append(",\n")
                .Append("    \"near_stationary_mean_signed_residual_mm\": ").Append(FloatText(nearStationaryMeanSignedResidualMm)).Append(",\n")
                .Append("    \"near_stationary_mean_abs_residual_mm\": ").Append(FloatText(nearStationaryMeanAbsResidualMm)).Append(",\n")
                .Append("    \"near_stationary_rms_residual_mm\": ").Append(FloatText(nearStationaryRmsResidualMm)).Append(",\n")
                .Append("    \"near_stationary_max_abs_residual_mm\": ").Append(FloatText(nearStationaryMaxAbsoluteResidualM * 1000f)).Append(",\n")
                .Append("    \"near_stationary_support_valid_samples\": ").Append(nearStationarySupportValidSamples).Append(",\n")
                .Append("    \"near_stationary_support_mean_signed_residual_mm\": ").Append(FloatText(nearStationarySupportMeanSignedResidualMm)).Append(",\n")
                .Append("    \"near_stationary_support_mean_abs_residual_mm\": ").Append(FloatText(nearStationarySupportMeanAbsResidualMm)).Append(",\n")
                .Append("    \"near_stationary_support_rms_residual_mm\": ").Append(FloatText(nearStationarySupportRmsResidualMm)).Append(",\n")
                .Append("    \"near_stationary_support_max_abs_residual_mm\": ").Append(FloatText(nearStationarySupportMaxAbsoluteResidualM * 1000f)).Append(",\n")
                .Append("    \"same_view_temporal_consistency_cone_deg\": ").Append(FloatText(SameViewTemporalConsistencyConeDeg)).Append(",\n")
                .Append("    \"same_view_temporal_conflict_ghosts\": ").Append(sameViewTemporalConflictGhosts).Append(",\n")
                .Append("    \"same_view_temporal_conflict_pairs\": ").Append(sameViewTemporalConflictPairs).Append(",\n")
                .Append("    \"same_view_support_observations_invalidated\": ").Append(sameViewSupportObservationsInvalidated).Append(",\n")
                .Append("    \"same_view_free_observations_invalidated\": ").Append(sameViewFreeObservationsInvalidated).Append("\n")
                .Append("  },\n")
                .Append("  \"retirement_contract_shadow\": {\n")
                .Append("    \"authority\": \"shadow_only_zero_production_authority\",\n")
                .Append("    \"same_class_pair_required\": 1,\n")
                .Append("    \"same_view_temporal_consistency_required\": 1,\n")
                .Append("    \"same_view_cone_deg\": ").Append(FloatText(SameViewTemporalConsistencyConeDeg)).Append(",\n")
                .Append("    \"lanes\": [\n")
                .Append(retirementContractShadowSummary).Append("\n")
                .Append("    ]\n")
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
                .Append("  \"promotion_provenance\": {\n")
                .Append("    \"meaning\": \"birth_to_first_stable_promotion_receipt\",\n")
                .Append("    \"total_events\": ").Append(promotionTotal).Append(",\n")
                .Append("    \"retained_events\": ").Append(promotionRetained).Append(",\n")
                .Append("    \"overwritten_events\": ").Append(Math.Max(0L, (long)promotionTotal - promotionRetained)).Append(",\n")
                .Append("    \"birth_hard_edge_or_unrescued_grazing\": ").Append(birthEdgeRisk).Append(",\n")
                .Append("    \"promotion_hard_edge_or_unrescued_grazing\": ").Append(promotionEdgeRisk).Append(",\n")
                .Append("    \"birth_and_promotion_hard_edge_or_unrescued_grazing\": ").Append(birthAndPromotionEdgeRisk).Append(",\n")
                .Append("    \"birth_temporal_risk\": ").Append(birthTemporalRisk).Append(",\n")
                .Append("    \"promotion_temporal_risk\": ").Append(promotionTemporalRisk).Append(",\n")
                .Append("    \"birth_outer_view_ge_0_70\": ").Append(birthOuterView).Append(",\n")
                .Append("    \"promotion_outer_view_ge_0_70\": ").Append(promotionOuterView).Append(",\n")
                .Append("    \"birth_motion_quality_lt_0_70\": ").Append(birthLowMotionQuality).Append(",\n")
                .Append("    \"promotion_motion_quality_lt_0_70\": ").Append(promotionLowMotionQuality).Append("\n")
                .Append("  },\n")
                .Append("  \"world_bounds_m\": {\n")
                .Append("    \"valid\": ").Append(haveBounds ? "true" : "false").Append(",\n")
                .Append("    \"min\": [").Append(FloatText(boundsMin.x)).Append(',').Append(FloatText(boundsMin.y)).Append(',').Append(FloatText(boundsMin.z)).Append("],\n")
                .Append("    \"max\": [").Append(FloatText(boundsMax.x)).Append(',').Append(FloatText(boundsMax.y)).Append(',').Append(FloatText(boundsMax.z)).Append("]\n")
                .Append("  },\n")
                .Append("  \"files\": {\"candidates\": \"candidates.csv\", \"retirements\": \"retirements.csv\", \"retirement_ghosts\": \"retirement_ghosts.csv\", \"retirement_ghost_observations\": \"retirement_ghost_observations.csv\", \"retirement_contract_shadow\": \"retirement_contract_shadow.csv\", \"court_waves\": \"court_waves.csv\", \"promotions\": \"promotions.csv\"}\n")
                .AppendLine("}");

            try
            {
                Directory.CreateDirectory(directory);
                File.WriteAllText(candidatesPath, candidatesCsv.ToString(), Encoding.UTF8);
                File.WriteAllText(retirementsPath, retirementsCsv.ToString(), Encoding.UTF8);
                File.WriteAllText(retirementGhostsPath,
                    retirementGhostsCsv.ToString(), Encoding.UTF8);
                File.WriteAllText(retirementGhostObservationsPath,
                    retirementGhostObservationsCsv.ToString(), Encoding.UTF8);
                File.WriteAllText(retirementContractShadowPath,
                    retirementContractShadowCsv.ToString(), Encoding.UTF8);
                File.WriteAllText(courtWavesPath, courtWavesCsv.ToString(), Encoding.UTF8);
                File.WriteAllText(promotionsPath, promotionsCsv.ToString(), Encoding.UTF8);
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

        private static uint RequiredPromotionAuthorityFrames(uint observedFrames)
        {
            return 2u;
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

        private static float MaximumPairwiseRayAngleDeg(Vector3[] rays, int count)
        {
            float maximum = 0f;
            int safeCount = Mathf.Min(count, rays?.Length ?? 0);
            for (int first = 0; first < safeCount; first++)
            {
                for (int second = first + 1; second < safeCount; second++)
                    maximum = Mathf.Max(maximum,
                        Vector3.Angle(rays[first], rays[second]));
            }
            return maximum;
        }

        private static int MarkSameViewTemporalConflicts(
            Vector3[] rays, uint[] classifications, bool[] hasRay, int count,
            bool[] conflicts, float coneDeg)
        {
            int safeCount = Mathf.Min(count, Mathf.Min(rays?.Length ?? 0,
                Mathf.Min(classifications?.Length ?? 0, Mathf.Min(
                    hasRay?.Length ?? 0, conflicts?.Length ?? 0))));
            int conflictPairs = 0;
            for (int first = 0; first < safeCount; first++)
            {
                if (!hasRay[first] || (classifications[first] != 1u &&
                    classifications[first] != 2u)) continue;
                for (int second = first + 1; second < safeCount; second++)
                {
                    if (!hasRay[second] || classifications[first] ==
                        classifications[second] || (classifications[second] != 1u &&
                        classifications[second] != 2u)) continue;
                    if (Vector3.Angle(rays[first], rays[second]) > coneDeg) continue;
                    conflicts[first] = true;
                    conflicts[second] = true;
                    conflictPairs++;
                }
            }
            return conflictPairs;
        }

        private static string RetirementContractShadowOutcome(
            bool supportCrossView, bool freeCrossView)
        {
            if (supportCrossView && freeCrossView) return "defer_conflict";
            if (supportCrossView) return "retain_cross_view_support";
            if (freeCrossView) return "delete_cross_view_free";
            return "defer_insufficient_view_diversity";
        }

        private static int RetirementWitnessFrame(float4 witness)
        {
            uint token = witness.w > 0f
                ? (uint)Mathf.RoundToInt(witness.w) : 0u;
            return token == 0u ? -1 : (int)(token - 1u);
        }

        private static float RetirementWitnessPairAngleDeg(float4 first,
            float4 second)
        {
            if (first.w <= 0f || second.w <= 0f) return 0f;
            var firstRay = new Vector3(first.x, first.y, first.z);
            var secondRay = new Vector3(second.x, second.y, second.z);
            if (firstRay.sqrMagnitude <= 1e-8f || secondRay.sqrMagnitude <= 1e-8f)
                return 0f;
            return Vector3.Angle(firstRay, secondRay);
        }

        private static string RetirementGateDecisionName(uint decision)
        {
            switch (decision)
            {
                case 1u: return "waiting_insufficient";
                case 2u: return "waiting_conflict";
                case 3u: return "support_veto";
                case 4u: return "confirmed_free_delete";
                default: return "none";
            }
        }

        private static string RetirementGhostClassificationName(uint classification)
        {
            switch (classification)
            {
                case 1u: return "coherent_support";
                case 2u: return "universal_free";
                case 3u: return "mixed_boundary";
                case 4u: return "incomplete";
                case 5u: return "neutral";
                default: return "unknown";
            }
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
            int ghostObservationCount = RetirementGhostCapacity *
                RetirementGhostObservationCapacity;
            int clearCount = Mathf.Max(TableCapacity * CandidateCapacity,
                Mathf.Max(CourtWaveCapacity, ghostObservationCount));
            _clearCells.DispatchFit(clearCount, 1, 1);
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
                slot.PreTransactionCorrespondenceIdentity?.Release();
                slot.CorrespondenceIdentity?.Release();
                slot.DirectProjectionAudit?.Release();
                slot.Observations = null;
                slot.Correspondences = null;
                slot.PreTransactionCorrespondenceIdentity = null;
                slot.CorrespondenceIdentity = null;
                slot.DirectProjectionAudit = null;
                slot.Completion = null;
                slot.Pending = false;
                slot.CorrespondenceReadbackDone = false;
                slot.DirectProjectionReadbackDone = false;
                slot.DirectProjectionScheduled = false;
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
            _candidateDirectSupportReceipt?.Release();
            _candidateRetirementGateState?.Release();
            _candidateRetirementSupportA?.Release();
            _candidateRetirementSupportB?.Release();
            _candidateRetirementFreeA?.Release();
            _candidateRetirementFreeB?.Release();
            _retirementGateCounters?.Release();
            _candidateBirthSourceReason?.Release();
            _candidateBirthMotionView?.Release();
            _candidateBirthDeltaQuality?.Release();
            _candidateEvidenceClassCounts?.Release();
            _candidatePromotionAuthority?.Release();
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
            _retireAuditGateState?.Release();
            _retireAuditGateWitnessFrames?.Release();
            _retireAuditGateWitnessAngles?.Release();
            _courtWaveLock?.Release();
            _courtWavePeakBits?.Release();
            _courtWavePositionResidual?.Release();
            _courtWaveTargetSigma?.Release();
            _courtWaveMeta?.Release();
            _courtWaveSourceReason?.Release();
            _courtWaveRawPositionDelta?.Release();
            _courtWaveMotionView?.Release();
            _courtWaveCounters?.Release();
            _promotionAuditCounters?.Release();
            _promotionAuditMeta?.Release();
            _promotionAuditGeometry?.Release();
            _promotionAuditBirthSourceReason?.Release();
            _promotionAuditSourceReason?.Release();
            _promotionAuditBirthMotionView?.Release();
            _promotionAuditMotionView?.Release();
            _promotionAuditDeltaSigma?.Release();
            _promotionAuditQualityRange?.Release();
            _promotionAuditSupport?.Release();
            _promotionAuditAuthority?.Release();
            _retirementGhostCounters?.Release();
            _retirementGhostIdentity?.Release();
            _retirementGhostCenterSigma?.Release();
            _retirementGhostNormalSupport?.Release();
            _retirementGhostOriginCamera?.Release();
            _retirementGhostState?.Release();
            _retirementGhostClasses?.Release();
            _retirementGhostObservationState?.Release();
            _retirementGhostObservationMeta?.Release();
            _retirementGhostObservationCamera?.Release();
            _retirementGhostObservationView?.Release();
            _retirementGhostObservationResidual?.Release();
        }
    }
}
