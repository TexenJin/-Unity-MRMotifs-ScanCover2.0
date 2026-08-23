using System;
using System.Runtime.InteropServices;
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

            public FrameDecision(int frameIndex, Matrix4x4 correction,
                int correspondenceCount, int effectiveRank,
                float translationMm, float rotationDeg,
                bool translationClamped, bool rotationClamped,
                bool readbackSucceeded)
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
            public int ObservationCount;
            public int FrameIndex;
            public bool ReportFrame;
            public Matrix4x4[] ViewInv;
            public int Generation;
            public Action<FrameDecision> Completion;
        }

        private const int TableCapacity = 65536;
        private const int CandidateCapacity = 3;
        private const int ProbeCount = 12;
        private const int RingSize = 3;
        private const int StatsCount = 16;
        private const int ObservationStride = sizeof(float) * 8;
        private const int CorrespondenceStride = sizeof(float) * 12;
        private const int CandidateFloat4Stride = sizeof(float) * 4;
        private const int CandidateMetaStride = sizeof(uint) * 4;

        private static readonly int DepthTexID = Shader.PropertyToID("_DepthTex");
        private static readonly int NormalTexID = Shader.PropertyToID("_NormalTex");
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
        private static readonly int CellSizeID = Shader.PropertyToID("_CellSize");
        private static readonly int MotionQualityID = Shader.PropertyToID("_MotionQuality");
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
        private static readonly int StatsID = Shader.PropertyToID("_Stats");

        private readonly ComputeShader _shader;
        private readonly ComputeKernelHelper _clearCells;
        private readonly ComputeKernelHelper _clearStats;
        private readonly ComputeKernelHelper _buildObservations;
        private readonly ComputeKernelHelper _buildCorrespondences;
        private readonly ComputeKernelHelper _updateCandidates;
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
        private ComputeBuffer _stats;
        private bool _disposed;
        private int _nextSlot;
        private int _generation;

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
        private bool _hasReport;

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
            _updateCandidates = new ComputeKernelHelper(_shader, "UpdateCandidates");
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
            _stats = new ComputeBuffer(StatsCount, sizeof(uint));
            _stats.SetData(new uint[StatsCount]);
            for (int i = 0; i < _slots.Length; i++) _slots[i] = new FrameSlot();
        }

        private void BindPersistentBuffers()
        {
            var kernels = new[]
            {
                _clearCells, _clearStats, _buildObservations, _buildCorrespondences,
                _updateCandidates, _countCensus
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
                kernel.Set(StatsID, _stats);
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
        }

        private bool EnsureFrameBuffers(int observationCount)
        {
            if (_slots[0].Observations != null && _slots[0].ObservationCount == observationCount)
                return true;
            foreach (var slot in _slots)
                if (slot.Pending) return false;
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

            return Dispatch(depth.DepthTex, depth.NormTex,
                depth.DepthWidth, depth.DepthHeight,
                depth.ProjInv, depth.ViewInv,
                motionQuality, frameIndex, completion);
        }

        public bool Dispatch(Texture depthTexture, Texture normalTexture,
            int depthWidth, int depthHeight,
            Matrix4x4[] projectionInverse, Matrix4x4[] viewInverse,
            float motionQuality, int frameIndex,
            Action<FrameDecision> completion = null)
        {
            if (_disposed || depthTexture == null || normalTexture == null ||
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
                if (_slots[index].Pending) continue;
                slot = _slots[index];
                _nextSlot = (index + 1) % _slots.Length;
                break;
            }
            if (slot == null) return false;

            slot.Pending = true;
            slot.FrameIndex = frameIndex;
            slot.ReportFrame = frameIndex % _reportInterval == 0;
            slot.ViewInv = (Matrix4x4[])viewInverse.Clone();
            slot.Generation = _generation;
            slot.Completion = completion;

            _shader.SetInts(DepthSizeID, depthWidth, depthHeight);
            _shader.SetInts(ObservationGridID, gridX, gridY);
            _shader.SetInt(ObservationCountID, observationCount);
            _shader.SetInt(FrameIndexID, frameIndex);
            _shader.SetFloat(MotionQualityID, Mathf.Clamp01(motionQuality));
            _shader.SetMatrixArray(DepthProjInvID, projectionInverse);
            _shader.SetMatrixArray(DepthViewInvID, viewInverse);
            _buildObservations.Set(DepthTexID, depthTexture);
            _buildObservations.Set(NormalTexID, normalTexture);
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

            var decision = new FrameDecision(slot.FrameIndex, correction,
                correspondenceCount, effectiveRank, translationMm, rotationDeg,
                translationClamped, rotationClamped, !request.hasError);

            _shader.SetInt(ObservationCountID, slot.ObservationCount);
            _shader.SetInt(FrameIndexID, slot.FrameIndex);
            _shader.SetMatrix(CorrectionID, correction);
            _shader.SetMatrixArray(DepthViewInvID, slot.ViewInv);
            _updateCandidates.Set(ObservationsID, slot.Observations);
            _updateCandidates.DispatchFit(slot.ObservationCount, 1, 1);

            if (slot.ReportFrame)
            {
                _countCensus.DispatchFit(TableCapacity, 1, 1);
                int reportGeneration = _generation;
                AsyncGPUReadback.Request(_stats,
                    statsRequest => OnStats(statsRequest, reportGeneration));
            }
            var completion = slot.Completion;
            slot.Completion = null;
            slot.Pending = false;
            completion?.Invoke(decision);
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
                if (item.SourceValid.w < 0.5f || item.TargetSigma.w <= 0f) continue;
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
            _hasReport = true;
        }

        public string GetCompact()
        {
            if (!_hasReport) return "影子预热中";
            return $"校枪{_lastTranslationMm:F1}mm/{_lastRotationDeg:F2}度 " +
                   $"配{_lastCorrespondences}秩{_lastEffectiveRank} " +
                   $"胶{_lastActiveCandidates}稳{_lastStableCandidates}" +
                   $"多{_lastMultiStableCells}陈{_lastStaleStableCandidates} " +
                   $"本帧观{_lastValidObservations}生{_lastBirths}更{_lastUpdates}" +
                   $"晋{_lastPromotions}满{_lastCapacityAbstentions}锁{_lastContentions}";
        }

        public void Clear()
        {
            if (_disposed) return;
            _generation++;
            SetSharedConstants();
            _clearCells.DispatchFit(TableCapacity * CandidateCapacity, 1, 1);
            _clearStats.DispatchFit(StatsCount, 1, 1);
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
            }
            _cellKeys?.Release();
            _cellState?.Release();
            _cellLock?.Release();
            _candidateCenterSigma?.Release();
            _candidateNormalSupport?.Release();
            _candidateFirstViewAngle?.Release();
            _candidateMeta?.Release();
            _stats?.Release();
        }
    }
}
