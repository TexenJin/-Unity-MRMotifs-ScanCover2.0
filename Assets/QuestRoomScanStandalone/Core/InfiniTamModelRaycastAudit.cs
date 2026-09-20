using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Model prediction and frame-to-model tracking for the isolated InfiniTAM
    /// baseline.  Diagnostic dispatches remain read-only.  Production tracking
    /// returns a decision for the exact retained depth frame; VolumeIntegrator
    /// alone owns the later quality gate and may choose to fuse that frame.
    /// </summary>
    internal sealed class InfiniTamModelRaycastAudit : IDisposable
    {
        private const int StatCount = 40;
        private const float DispatchIntervalSeconds = 0.25f;
        private const float ReadbackIntervalSeconds = 1f;
        // Coarse-to-fine production schedule.  Every entry re-renders the
        // model at the pose produced by the preceding pass; this is not five
        // solves over one stale correspondence set.
        // Quest already supplies a world-tracked pose for the retained depth
        // frame.  Re-running the five-pass no-pose-camera schedule through five
        // serial AsyncGPUReadback round trips drops production tracking to only
        // a few hertz on-device; a normal head turn then outruns model overlap.
        // Use that external pose as the strong prior and keep only one medium
        // and one fine point-to-plane refinement (8 / 2 => strides 4, 2).
        // This is still track-before-fuse: both passes re-raycast the sole TSDF
        // and the exact retained frame remains blocked until the final verdict.
        private static readonly int[] ProductionStrideDivisors = { 2, 4 };
        private static readonly uint[] ZeroStats = new uint[StatCount];

        private static readonly int DepthTexID = Shader.PropertyToID("gsDepthTex");
        private static readonly int ModelTsdfID = Shader.PropertyToID("gsModelTsdf");
        private static readonly int ModelVoteWeightID = Shader.PropertyToID("gsModelVoteWeight");
        private static readonly int ModelDepthRWID = Shader.PropertyToID("gsModelDepthRW");
        private static readonly int ModelResidualMmRWID = Shader.PropertyToID("gsModelResidualMmRW");
        private static readonly int StatsID = Shader.PropertyToID("gsModelRaycastStats");
        private static readonly int TrackingPairsID = Shader.PropertyToID("gsModelTrackingPairs");
        private static readonly int DepthSizeID = Shader.PropertyToID("gsDepthTexSize");
        private static readonly int OutputSizeID = Shader.PropertyToID("gsModelOutputSize");
        private static readonly int VoxelCountID = Shader.PropertyToID("gsVoxCount");
        private static readonly int VoxelSizeID = Shader.PropertyToID("gsVoxSize");
        private static readonly int VoxelDistanceID = Shader.PropertyToID("gsVoxDist");
        private static readonly int MinWeightID = Shader.PropertyToID("gsModelMinWeight");
        private static readonly int MinObservedRangeID = Shader.PropertyToID("gsModelMinObservedRange");
        private static readonly int MaxObservedRangeID = Shader.PropertyToID("gsModelMaxObservedRange");
        private static readonly int PixelStrideID = Shader.PropertyToID("gsModelPixelStride");

        private readonly int _pixelStride;
        private ComputeShader _shader;
        private ComputeKernelHelper _clearKernel;
        private ComputeKernelHelper _raycastKernel;
        private ComputeBuffer _stats;
        private ComputeBuffer _trackingPairs;
        private RenderTexture _modelDepth;
        private RenderTexture _modelResidualMm;
        private bool _initializationFailed;
        private bool _readbackPending;
        private bool _trackingReadbackPending;
        private bool _productionTrackingPending;
        private float _nextDispatchTime;
        private float _nextReadbackTime;
        private int _generation;
        private int _lastSourceFrame = -1;
        private int _sampleCount;
        private float _lastMinObservedRange;
        private float _lastMaxObservedRange;
        private readonly uint[] _last = new uint[StatCount];
        private readonly ulong[] _cumulative = new ulong[StatCount];
        private int _trackingSampleCount;
        private int _trackingPairCount;
        private int _trackingRank;
        private float _trackingTranslationMm;
        private float _trackingRotationDeg;
        private float _trackingMeanBeforeMm;
        private float _trackingMeanAfterMm;
        private float _trackingWithin30BeforePercent;
        private float _trackingWithin30AfterPercent;
        private bool _trackingTranslationClamped;
        private bool _trackingRotationClamped;
        private int _trackingIterationCount;
        private int _trackingLevelCount;
        private ProductionTrackingState _productionState;

        private sealed class ProductionTrackingState
        {
            public int Generation;
            public int SourceFrame;
            public Texture Depth;
            public Matrix4x4[] Projection;
            public Matrix4x4[] View;
            public Matrix4x4[] ProjectionInverse;
            public Matrix4x4[] ViewInverse;
            public RenderTexture Tsdf;
            public RenderTexture VoteWeight;
            public int3 VoxelCount;
            public float VoxelSize;
            public float VoxelDistance;
            public float MinWeight;
            public float MinObservedRange;
            public float MaxObservedRange;
            public Action<TrackingDecision> Completed;
            public int[] Strides;
            public int PassIndex;
            public int LevelCount;
            public int MinimumRank = 6;
            public int MinimumCorrespondenceCount = int.MaxValue;
            public float InitialMeanBeforeMm;
            public float InitialWithin30Percent;
            public float FinalMeanAfterMm;
            public float FinalWithin30Percent;
            public bool HasInitialMetrics;
            public bool AnyTranslationClamped;
            public bool AnyRotationClamped;
            public Matrix4x4 AccumulatedCorrection = Matrix4x4.identity;
            public TrackingDecision FinalDecision;
            public bool FinalPairsDone;
            public bool FinalStatsDone;
        }

        internal readonly struct TrackingDecision
        {
            public readonly int SourceFrame;
            public readonly bool ReadbackSucceeded;
            public readonly Matrix4x4 Correction;
            public readonly int CorrespondenceCount;
            public readonly int EffectiveRank;
            public readonly float TranslationMm;
            public readonly float RotationDeg;
            public readonly float MeanBeforeMm;
            public readonly float MeanAfterMm;
            public readonly float Within30BeforePercent;
            public readonly float Within30AfterPercent;
            public readonly bool TranslationClamped;
            public readonly bool RotationClamped;
            public readonly int IterationCount;
            public readonly int PyramidLevelCount;

            public TrackingDecision(int sourceFrame, bool readbackSucceeded,
                Matrix4x4 correction, int correspondenceCount, int effectiveRank,
                float translationMm, float rotationDeg, float meanBeforeMm,
                float meanAfterMm, float within30BeforePercent,
                float within30AfterPercent, bool translationClamped,
                bool rotationClamped, int iterationCount = 1,
                int pyramidLevelCount = 1)
            {
                SourceFrame = sourceFrame;
                ReadbackSucceeded = readbackSucceeded;
                Correction = correction;
                CorrespondenceCount = correspondenceCount;
                EffectiveRank = effectiveRank;
                TranslationMm = translationMm;
                RotationDeg = rotationDeg;
                MeanBeforeMm = meanBeforeMm;
                MeanAfterMm = meanAfterMm;
                Within30BeforePercent = within30BeforePercent;
                Within30AfterPercent = within30AfterPercent;
                TranslationClamped = translationClamped;
                RotationClamped = rotationClamped;
                IterationCount = iterationCount;
                PyramidLevelCount = pyramidLevelCount;
            }

            public static TrackingDecision Failed(int sourceFrame) =>
                new(sourceFrame, false, Matrix4x4.identity, 0, 0,
                    0f, 0f, 0f, 0f, 0f, 0f, false, false, 0, 0);

            // The GPU readback succeeded, but this pyramid pass did not have
            // enough independent constraints to solve a correction.  Keep it
            // distinct from a readback failure so the production gate can
            // report "少配" or "欠秩" instead of the misleading "回读".
            public static TrackingDecision Underdetermined(int sourceFrame,
                int correspondenceCount, int effectiveRank) =>
                new(sourceFrame, true, Matrix4x4.identity,
                    correspondenceCount, effectiveRank,
                    0f, 0f, 0f, 0f, 0f, 0f, false, false, 1, 1);
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct ModelTrackingPair
        {
            public Vector4 ObservedValid;
            public Vector4 ModelResidual;
            public Vector4 NormalVote;
        }

        public InfiniTamModelRaycastAudit(int pixelStride = 4)
        {
            _pixelStride = Mathf.Clamp(pixelStride, 1, 16);
        }

        private int[] BuildProductionStrideSchedule()
        {
            int coarseStride = Mathf.Clamp(_pixelStride, 4, 16);
            var strides = new int[ProductionStrideDivisors.Length];
            for (int i = 0; i < strides.Length; i++)
                strides[i] = Mathf.Max(1,
                    coarseStride / ProductionStrideDivisors[i]);
            return strides;
        }

        public RenderTexture ModelDepth => _modelDepth;
        public RenderTexture ModelResidualMm => _modelResidualMm;
        public bool HasSample { get; private set; }
        public bool ProductionTrackingPending => _productionTrackingPending;

        /// <summary>
        /// Runs prediction for one retained production frame.  Only one exact
        /// frame may be in flight because the GPU pair buffer is shared with
        /// its asynchronous readback.  Returning false means the caller must
        /// not enqueue or fuse the frame.
        /// </summary>
        public bool TryDispatchProductionTracking(Texture acceptedDepth,
            Matrix4x4[] projection, Matrix4x4[] view,
            Matrix4x4[] projectionInverse, Matrix4x4[] viewInverse,
            RenderTexture soleTsdf, RenderTexture soleVoteWeight,
            int3 voxelCount, float voxelSize, float voxelDistance,
            float minWeight, float minObservedRange, float maxObservedRange,
            int sourceFrame, Action<TrackingDecision> completed)
        {
            if (_productionTrackingPending || _trackingReadbackPending ||
                _readbackPending || acceptedDepth == null || soleTsdf == null ||
                soleVoteWeight == null || projection == null || view == null ||
                projectionInverse == null || viewInverse == null ||
                projection.Length < 2 || view.Length < 2 ||
                projectionInverse.Length < 2 || viewInverse.Length < 2)
                return false;
            int[] strides = BuildProductionStrideSchedule();
            int levelCount = 0;
            int previousStride = -1;
            for (int i = 0; i < strides.Length; i++)
            {
                if (strides[i] != previousStride)
                {
                    levelCount++;
                    previousStride = strides[i];
                }
            }
            var state = new ProductionTrackingState
            {
                Generation = _generation,
                SourceFrame = sourceFrame,
                Depth = acceptedDepth,
                Projection = (Matrix4x4[])projection.Clone(),
                View = (Matrix4x4[])view.Clone(),
                ProjectionInverse = (Matrix4x4[])projectionInverse.Clone(),
                ViewInverse = (Matrix4x4[])viewInverse.Clone(),
                Tsdf = soleTsdf,
                VoteWeight = soleVoteWeight,
                VoxelCount = voxelCount,
                VoxelSize = voxelSize,
                VoxelDistance = voxelDistance,
                MinWeight = minWeight,
                MinObservedRange = minObservedRange,
                MaxObservedRange = maxObservedRange,
                Completed = completed,
                Strides = strides,
                LevelCount = levelCount
            };
            _productionState = state;
            _productionTrackingPending = true;
            if (DispatchProductionPass(state)) return true;
            _productionState = null;
            _productionTrackingPending = false;
            return false;
        }

        private bool DispatchProductionPass(ProductionTrackingState state)
        {
            if (state == null || state != _productionState ||
                state.Generation != _generation ||
                state.PassIndex < 0 || state.PassIndex >= state.Strides.Length)
                return false;
            int stride = state.Strides[state.PassIndex];
            if (!EnsureResources(state.Depth.width, state.Depth.height, stride))
                return false;

            ConfigureAndDispatch(state.Depth, state.Projection, state.View,
                state.ProjectionInverse, state.ViewInverse, state.Tsdf,
                state.VoteWeight, state.VoxelCount, state.VoxelSize,
                state.VoxelDistance, state.MinWeight,
                state.MinObservedRange, state.MaxObservedRange,
                state.SourceFrame, stride);

            bool finalPass = state.PassIndex == state.Strides.Length - 1;
            _trackingReadbackPending = true;
            if (finalPass)
            {
                _readbackPending = true;
                AsyncGPUReadback.Request(_stats, request =>
                {
                    if (state.Generation != _generation ||
                        state != _productionState) return;
                    CompleteReadback(request, state.Generation);
                    state.FinalStatsDone = true;
                    TryCompleteFinalProductionPass(state);
                });
            }
            AsyncGPUReadback.Request(_trackingPairs, request =>
            {
                if (state.Generation != _generation ||
                    state != _productionState) return;
                _trackingReadbackPending = false;
                TrackingDecision incremental = request.hasError
                    ? TrackingDecision.Failed(state.SourceFrame)
                    : SolveTrackingCorrection(
                        request.GetData<ModelTrackingPair>(), state.SourceFrame);
                if (!incremental.ReadbackSucceeded)
                {
                    state.FinalDecision = incremental;
                    state.FinalPairsDone = true;
                    if (finalPass) TryCompleteFinalProductionPass(state);
                    else CompleteProductionTracking(state, incremental);
                    return;
                }

                if (!state.HasInitialMetrics)
                {
                    state.HasInitialMetrics = true;
                    state.InitialMeanBeforeMm = incremental.MeanBeforeMm;
                    state.InitialWithin30Percent =
                        incremental.Within30BeforePercent;
                }
                state.MinimumRank = Mathf.Min(state.MinimumRank,
                    incremental.EffectiveRank);
                state.MinimumCorrespondenceCount = Mathf.Min(
                    state.MinimumCorrespondenceCount,
                    incremental.CorrespondenceCount);
                state.FinalMeanAfterMm = incremental.MeanAfterMm;
                state.FinalWithin30Percent =
                    incremental.Within30AfterPercent;
                state.AnyTranslationClamped |=
                    incremental.TranslationClamped;
                state.AnyRotationClamped |= incremental.RotationClamped;
                state.AccumulatedCorrection = incremental.Correction *
                                              state.AccumulatedCorrection;
                ApplyCorrection(state.View, state.ViewInverse,
                    incremental.Correction);
                state.PassIndex++;

                if (finalPass)
                {
                    state.FinalDecision = BuildAggregateDecision(state);
                    state.FinalPairsDone = true;
                    TryCompleteFinalProductionPass(state);
                    return;
                }
                if (!DispatchProductionPass(state))
                    CompleteProductionTracking(state,
                        TrackingDecision.Failed(state.SourceFrame));
            });
            return true;
        }

        private static void ApplyCorrection(Matrix4x4[] view,
            Matrix4x4[] viewInverse, Matrix4x4 correction)
        {
            int eyeCount = Mathf.Min(view.Length, viewInverse.Length);
            for (int eye = 0; eye < eyeCount; eye++)
            {
                viewInverse[eye] = correction * viewInverse[eye];
                view[eye] = viewInverse[eye].inverse;
            }
        }

        private static int CountCompletedLevels(ProductionTrackingState state)
        {
            int levels = 0;
            int previousStride = -1;
            int count = Mathf.Min(state.PassIndex, state.Strides.Length);
            for (int i = 0; i < count; i++)
            {
                if (state.Strides[i] == previousStride) continue;
                previousStride = state.Strides[i];
                levels++;
            }
            return levels;
        }

        private static TrackingDecision BuildAggregateDecision(
            ProductionTrackingState state)
        {
            Vector3 translation = state.AccumulatedCorrection.GetColumn(3);
            float translationMm = translation.magnitude * 1000f;
            float rotationDeg = Quaternion.Angle(Quaternion.identity,
                state.AccumulatedCorrection.rotation);
            return new TrackingDecision(state.SourceFrame, true,
                state.AccumulatedCorrection,
                state.MinimumCorrespondenceCount == int.MaxValue
                    ? 0 : state.MinimumCorrespondenceCount,
                state.MinimumRank, translationMm, rotationDeg,
                state.InitialMeanBeforeMm, state.FinalMeanAfterMm,
                state.InitialWithin30Percent, state.FinalWithin30Percent,
                state.AnyTranslationClamped, state.AnyRotationClamped,
                state.PassIndex, CountCompletedLevels(state));
        }

        private void TryCompleteFinalProductionPass(
            ProductionTrackingState state)
        {
            if (state == null || state != _productionState ||
                !state.FinalPairsDone || !state.FinalStatsDone) return;
            CompleteProductionTracking(state, state.FinalDecision);
        }

        private void CompleteProductionTracking(ProductionTrackingState state,
            TrackingDecision decision)
        {
            if (state == null || state != _productionState ||
                state.Generation != _generation) return;
            _trackingReadbackPending = false;
            _productionTrackingPending = false;
            _productionState = null;
            _trackingPairCount = decision.CorrespondenceCount;
            _trackingRank = decision.EffectiveRank;
            _trackingTranslationMm = decision.TranslationMm;
            _trackingRotationDeg = decision.RotationDeg;
            _trackingMeanBeforeMm = decision.MeanBeforeMm;
            _trackingMeanAfterMm = decision.MeanAfterMm;
            _trackingWithin30BeforePercent =
                decision.Within30BeforePercent;
            _trackingWithin30AfterPercent = decision.Within30AfterPercent;
            _trackingTranslationClamped = decision.TranslationClamped;
            _trackingRotationClamped = decision.RotationClamped;
            _trackingIterationCount = decision.IterationCount;
            _trackingLevelCount = decision.PyramidLevelCount;
            try { state.Completed?.Invoke(decision); }
            catch (Exception e)
            {
                Logger.Warning("InfiniTAM tracking callback failed: " + e.Message);
            }
        }

        public void Dispatch(Texture acceptedDepth, Matrix4x4[] projection,
            Matrix4x4[] view, Matrix4x4[] projectionInverse,
            Matrix4x4[] viewInverse, RenderTexture soleTsdf,
            RenderTexture soleVoteWeight,
            int3 voxelCount, float voxelSize, float voxelDistance,
            float minWeight, float minObservedRange, float maxObservedRange,
            int sourceFrame)
        {
            if (acceptedDepth == null || soleTsdf == null || soleVoteWeight == null ||
                projection == null ||
                view == null || projectionInverse == null || viewInverse == null)
                return;
            // Never clear/reuse the shared stats or correspondence buffers while
            // either asynchronous snapshot is still reading them. The previous
            // diagnostic cadence could tear one receipt; production tracking
            // requires the same strict buffer ownership.
            if (_readbackPending || _trackingReadbackPending ||
                _productionTrackingPending) return;
            // This overload is the throttled diagnostic receipt used during
            // bootstrap. Production authority uses the exact-frame overload
            // above and is never throttled by this display cadence.
            if (Time.unscaledTime < _nextDispatchTime) return;
            if (!EnsureResources(acceptedDepth.width, acceptedDepth.height,
                _pixelStride)) return;
            _nextDispatchTime = Time.unscaledTime + DispatchIntervalSeconds;

            ConfigureAndDispatch(acceptedDepth, projection, view,
                projectionInverse, viewInverse, soleTsdf, soleVoteWeight,
                voxelCount, voxelSize, voxelDistance, minWeight,
                minObservedRange, maxObservedRange, sourceFrame,
                _pixelStride);

            if (!_readbackPending && Time.unscaledTime >= _nextReadbackTime)
            {
                _readbackPending = true;
                _nextReadbackTime = Time.unscaledTime + ReadbackIntervalSeconds;
                int requestGeneration = _generation;
                AsyncGPUReadback.Request(_stats,
                    request => CompleteReadback(request, requestGeneration));
                if (!_trackingReadbackPending)
                {
                    _trackingReadbackPending = true;
                    AsyncGPUReadback.Request(_trackingPairs,
                        request => CompleteTrackingReadback(request,
                            requestGeneration));
                }
            }
        }

        private void ConfigureAndDispatch(Texture acceptedDepth,
            Matrix4x4[] projection, Matrix4x4[] view,
            Matrix4x4[] projectionInverse, Matrix4x4[] viewInverse,
            RenderTexture soleTsdf, RenderTexture soleVoteWeight,
            int3 voxelCount, float voxelSize, float voxelDistance,
            float minWeight, float minObservedRange, float maxObservedRange,
            int sourceFrame, int pixelStride)
        {
            _lastSourceFrame = sourceFrame;
            _shader.SetInts(DepthSizeID, acceptedDepth.width, acceptedDepth.height);
            _shader.SetInts(OutputSizeID, _modelDepth.width, _modelDepth.height);
            _shader.SetInts(VoxelCountID, voxelCount.x, voxelCount.y, voxelCount.z);
            _shader.SetFloat(VoxelSizeID, voxelSize);
            _shader.SetFloat(VoxelDistanceID, voxelDistance);
            _shader.SetFloat(MinWeightID, minWeight);
            _lastMinObservedRange = Mathf.Max(0f, minObservedRange);
            _lastMaxObservedRange = Mathf.Max(
                _lastMinObservedRange + 0.01f, maxObservedRange);
            _shader.SetFloat(MinObservedRangeID, _lastMinObservedRange);
            _shader.SetFloat(MaxObservedRangeID, _lastMaxObservedRange);
            _shader.SetInt(PixelStrideID, Mathf.Max(1, pixelStride));
            _shader.SetMatrixArray(DepthCapture.ProjID, projection);
            _shader.SetMatrixArray(DepthCapture.ViewID, view);
            _shader.SetMatrixArray(DepthCapture.ProjInvID, projectionInverse);
            _shader.SetMatrixArray(DepthCapture.ViewInvID, viewInverse);

            _clearKernel.Set(StatsID, _stats);
            _clearKernel.DispatchFit(StatCount, 1, 1);
            _raycastKernel.Set(DepthTexID, acceptedDepth);
            _raycastKernel.Set(ModelTsdfID, soleTsdf);
            _raycastKernel.Set(ModelVoteWeightID, soleVoteWeight);
            _raycastKernel.Set(ModelDepthRWID, _modelDepth);
            _raycastKernel.Set(ModelResidualMmRWID, _modelResidualMm);
            _raycastKernel.Set(StatsID, _stats);
            _raycastKernel.Set(TrackingPairsID, _trackingPairs);
            _raycastKernel.DispatchFit(_modelDepth.width, _modelDepth.height, 1);
        }

        public string GetCompact()
        {
            if (!HasSample) return "模[统计中]";
            uint observed = _last[1];
            uint compared = _last[3];
            uint within30 = _last[4] + _last[5];
            float hitPercent = observed > 0 ? 100f * compared / observed : 0f;
            float within30Percent = compared > 0 ? 100f * within30 / compared : 0f;
            float nearestWithin30Percent = compared > 0 ? 100f * _last[26] / compared : 0f;
            float meanAbsMm = compared > 0 ? (float)_last[13] / compared : 0f;
            float nearestMeanAbsMm = compared > 0 ? (float)_last[27] / compared : 0f;
            uint rejected = _last[20] + _last[21] + _last[22] + _last[23];
            float meanModelVotes = compared > 0 ? (float)_last[36] / (10f * compared) : 0f;
            return $"模 命{hitPercent:F0}% 首3c{within30Percent:F0}%/近{nearestWithin30Percent:F0}% " +
                   $"均{meanAbsMm:F0}/{nearestMeanAbsMm:F0} 峰{_last[16]}mm\n" +
                   $"坏{rejected}(深{_last[20]}/算{_last[21]}/近{_last[22]}/远{_last[23]}) " +
                   $"多{_last[24]}换{_last[25]} " +
                   $"漏{_last[9]}(界{_last[17]}/支{_last[18]}/叉{_last[19]})\n" +
                   $"偏票 青{_last[32]}/中{_last[33]}/熟{_last[34]}/满{_last[35]} " +
                   $"面票均{meanModelVotes:F0} 原位有{_last[38]}/空{_last[39]}\n" +
                   $"轨 配{_trackingPairCount}秩{_trackingRank} " +
                   $"迭{_trackingIterationCount}/{_trackingLevelCount}层 " +
                   $"修{_trackingTranslationMm:F1}mm/{_trackingRotationDeg:F2}° " +
                   $"均{_trackingMeanBeforeMm:F0}>{_trackingMeanAfterMm:F0} " +
                   $"准3c{_trackingWithin30BeforePercent:F0}>{_trackingWithin30AfterPercent:F0}%" +
                   $"{(_trackingTranslationClamped || _trackingRotationClamped ? "!" : string.Empty)}";
        }

        public void AppendReport(StringBuilder sb)
        {
            if (sb == null) return;
            static string U(ulong value) => value.ToString(CultureInfo.InvariantCulture);
            uint compared = _last[3];
            uint observed = _last[1];
            ulong cumulativeCompared = _cumulative[3];
            sb.AppendLine();
            sb.AppendLine("infinitam_model_raycast_ticket:");
            sb.AppendLine("tracking_architecture=coarse_to_fine_reraycast_8_8_4_4_2");
            sb.AppendLine("tsdf_binding=read_only;prediction_textures=diagnostic_only;tracking_decision=quality_gated_by_volume_integrator");
            sb.AppendLine("timing=before_retained_same_frame_integration;pose=retained_frame_exact");
            sb.AppendLine($"pixel_stride={_pixelStride.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"dispatch_interval_s={DispatchIntervalSeconds.ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"valid_observed_range_m={_lastMinObservedRange.ToString("F3", CultureInfo.InvariantCulture)}..{_lastMaxObservedRange.ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine("residual_authority=first_visible_positive_to_negative_crossing;nearest_crossing=diagnostic_only");
            sb.AppendLine($"sample_ready={HasSample.ToString().ToLowerInvariant()}");
            sb.AppendLine($"readback_pending={_readbackPending.ToString().ToLowerInvariant()}");
            sb.AppendLine($"sample_count={_sampleCount.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_source_frame={_lastSourceFrame.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_rays={_last[0].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_observed_depth={observed.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_model_hits={_last[2].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_compared={compared.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_model_hit_percent={(observed > 0 ? 100.0 * compared / observed : 0.0).ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_abs_le_10mm={_last[4].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_abs_10_30mm={_last[5].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_abs_30_60mm={_last[6].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_abs_60_120mm={_last[7].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_abs_gt_120mm={_last[8].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_observed_without_model={_last[9].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_model_without_observation={_last[10].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_model_in_front_gt_10mm={_last[11].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_model_behind_gt_10mm={_last[12].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_mean_abs_residual_mm={(compared > 0 ? (double)_last[13] / compared : 0.0).ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_max_abs_residual_mm={_last[16].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_observed_outside_volume={_last[17].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_observed_without_supported_tsdf={_last[18].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_supported_tsdf_without_zero_crossing={_last[19].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_rejected_raw_depth_domain={_last[20].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_rejected_unprojection_nonfinite={_last[21].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_rejected_observed_too_near={_last[22].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_rejected_observed_too_far={_last[23].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_multiple_zero_crossing_rays={_last[24].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_nearest_crossing_improves_gt_10mm={_last[25].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_nearest_crossing_abs_le_30mm={_last[26].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_nearest_crossing_mean_abs_residual_mm={(compared > 0 ? (double)_last[27] / compared : 0.0).ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_nearest_crossing_max_abs_residual_mm={_last[28].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_max_observed_range_mm={_last[29].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_max_first_model_range_mm={_last[30].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_max_positive_to_negative_crossings_per_ray={_last[31].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_gt_30mm_model_votes_lt_10={_last[32].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_gt_30mm_model_votes_10_50={_last[33].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_gt_30mm_model_votes_50_99_5={_last[34].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_gt_30mm_model_votes_ge_99_5={_last[35].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_model_crossing_mean_vote_weight={(compared > 0 ? (double)_last[36] / (10.0 * compared) : 0.0).ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_model_crossing_max_vote_weight={((double)_last[37] / 10.0).ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_gt_30mm_observation_position_supported={_last[38].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"last_gt_30mm_observation_position_unsupported={_last[39].ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_sample_count={_trackingSampleCount.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_min_correspondence_count_across_pyramid={_trackingPairCount.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_effective_rank={_trackingRank.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_iteration_count={_trackingIterationCount.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_pyramid_level_count={_trackingLevelCount.ToString(CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_pyramid_schedule={string.Join(",", BuildProductionStrideSchedule())}");
            sb.AppendLine($"tracking_candidate_translation_mm={_trackingTranslationMm.ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_candidate_rotation_deg={_trackingRotationDeg.ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_mean_abs_point_plane_before_mm={_trackingMeanBeforeMm.ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_mean_abs_point_plane_after_mm={_trackingMeanAfterMm.ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_within_30mm_before_percent={_trackingWithin30BeforePercent.ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_within_30mm_after_percent={_trackingWithin30AfterPercent.ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"tracking_translation_clamped={_trackingTranslationClamped.ToString().ToLowerInvariant()}");
            sb.AppendLine($"tracking_rotation_clamped={_trackingRotationClamped.ToString().ToLowerInvariant()}");
            sb.AppendLine($"cumulative_compared={U(cumulativeCompared)}");
            sb.AppendLine($"cumulative_observed_without_model={U(_cumulative[9])}");
            sb.AppendLine($"cumulative_observed_outside_volume={U(_cumulative[17])}");
            sb.AppendLine($"cumulative_observed_without_supported_tsdf={U(_cumulative[18])}");
            sb.AppendLine($"cumulative_supported_tsdf_without_zero_crossing={U(_cumulative[19])}");
            sb.AppendLine($"cumulative_rejected_raw_depth_domain={U(_cumulative[20])}");
            sb.AppendLine($"cumulative_rejected_unprojection_nonfinite={U(_cumulative[21])}");
            sb.AppendLine($"cumulative_rejected_observed_too_near={U(_cumulative[22])}");
            sb.AppendLine($"cumulative_rejected_observed_too_far={U(_cumulative[23])}");
            sb.AppendLine($"cumulative_multiple_zero_crossing_rays={U(_cumulative[24])}");
            sb.AppendLine($"cumulative_nearest_crossing_improves_gt_10mm={U(_cumulative[25])}");
            sb.AppendLine($"cumulative_nearest_crossing_abs_le_30mm={U(_cumulative[26])}");
            sb.AppendLine($"cumulative_nearest_crossing_mean_abs_residual_mm={(cumulativeCompared > 0 ? (double)_cumulative[27] / cumulativeCompared : 0.0).ToString("F3", CultureInfo.InvariantCulture)}");
            sb.AppendLine($"cumulative_gt_30mm_model_votes_lt_10={U(_cumulative[32])}");
            sb.AppendLine($"cumulative_gt_30mm_model_votes_10_50={U(_cumulative[33])}");
            sb.AppendLine($"cumulative_gt_30mm_model_votes_50_99_5={U(_cumulative[34])}");
            sb.AppendLine($"cumulative_gt_30mm_model_votes_ge_99_5={U(_cumulative[35])}");
            sb.AppendLine($"cumulative_gt_30mm_observation_position_supported={U(_cumulative[38])}");
            sb.AppendLine($"cumulative_gt_30mm_observation_position_unsupported={U(_cumulative[39])}");
            sb.AppendLine($"cumulative_model_in_front_gt_10mm={U(_cumulative[11])}");
            sb.AppendLine($"cumulative_model_behind_gt_10mm={U(_cumulative[12])}");
            sb.AppendLine($"cumulative_mean_abs_residual_mm={(cumulativeCompared > 0 ? (double)_cumulative[13] / cumulativeCompared : 0.0).ToString("F3", CultureInfo.InvariantCulture)}");
        }

        public void Reset()
        {
            _generation++;
            _readbackPending = false;
            _trackingReadbackPending = false;
            _productionTrackingPending = false;
            _nextDispatchTime = 0f;
            _nextReadbackTime = 0f;
            _lastSourceFrame = -1;
            _sampleCount = 0;
            _trackingSampleCount = 0;
            _trackingPairCount = 0;
            _trackingRank = 0;
            _trackingTranslationMm = 0f;
            _trackingRotationDeg = 0f;
            _trackingMeanBeforeMm = 0f;
            _trackingMeanAfterMm = 0f;
            _trackingWithin30BeforePercent = 0f;
            _trackingWithin30AfterPercent = 0f;
            _trackingTranslationClamped = false;
            _trackingRotationClamped = false;
            _trackingIterationCount = 0;
            _trackingLevelCount = 0;
            _productionState = null;
            HasSample = false;
            Array.Clear(_last, 0, _last.Length);
            Array.Clear(_cumulative, 0, _cumulative.Length);
            _stats?.SetData(ZeroStats);
        }

        public void Dispose()
        {
            _generation++;
            _readbackPending = false;
            _trackingReadbackPending = false;
            _productionTrackingPending = false;
            _productionState = null;
            _stats?.Release();
            _stats = null;
            _trackingPairs?.Release();
            _trackingPairs = null;
            if (_modelDepth != null) UnityEngine.Object.Destroy(_modelDepth);
            if (_modelResidualMm != null) UnityEngine.Object.Destroy(_modelResidualMm);
            _modelDepth = null;
            _modelResidualMm = null;
            _shader = null;
        }

        private bool EnsureResources(int depthWidth, int depthHeight,
            int pixelStride)
        {
            if (_initializationFailed) return false;
            if (_shader == null)
            {
                _shader = Resources.Load<ComputeShader>("InfiniTamModelRaycast");
                if (_shader == null || !_shader.HasKernel("ClearModelRaycastStats") ||
                    !_shader.HasKernel("RaycastModel"))
                {
                    Logger.Warning("InfiniTAM 模型预测内核缺失；bootstrap 后严格停笔，绝不绕回原始位姿污染 TSDF");
                    _shader = null;
                    _initializationFailed = true;
                    return false;
                }
                _clearKernel = new ComputeKernelHelper(_shader, "ClearModelRaycastStats");
                _raycastKernel = new ComputeKernelHelper(_shader, "RaycastModel");
                _stats = new ComputeBuffer(StatCount, sizeof(uint));
                _stats.SetData(ZeroStats);
            }

            int safeStride = Mathf.Max(1, pixelStride);
            int width = Mathf.Max(1,
                Mathf.CeilToInt((float)depthWidth / safeStride));
            int height = Mathf.Max(1,
                Mathf.CeilToInt((float)depthHeight / safeStride));
            if (_modelDepth != null && _modelDepth.width == width &&
                _modelDepth.height == height && _trackingPairs != null &&
                _trackingPairs.count == width * height)
                return true;

            if (_modelDepth != null) UnityEngine.Object.Destroy(_modelDepth);
            if (_modelResidualMm != null) UnityEngine.Object.Destroy(_modelResidualMm);
            _trackingPairs?.Release();
            _trackingPairs = null;
            GraphicsFormat format = SystemInfo.IsFormatSupported(
                GraphicsFormat.R16_SFloat, FormatUsage.LoadStore)
                ? GraphicsFormat.R16_SFloat
                : GraphicsFormat.R32_SFloat;
            _modelDepth = CreateOutput(width, height, format, "InfiniTamModelDepth");
            _modelResidualMm = CreateOutput(width, height, format,
                "InfiniTamModelResidualMm");
            _trackingPairs = new ComputeBuffer(width * height,
                Marshal.SizeOf<ModelTrackingPair>());
            bool ready = _modelDepth != null && _modelDepth.IsCreated() &&
                         _modelResidualMm != null && _modelResidualMm.IsCreated() &&
                         _trackingPairs != null;
            if (!ready)
            {
                Logger.Warning("InfiniTAM 模型预测资源创建失败；bootstrap 后严格停笔，绝不绕回原始位姿污染 TSDF");
                _initializationFailed = true;
            }
            return ready;
        }

        private static RenderTexture CreateOutput(int width, int height,
            GraphicsFormat format, string name)
        {
            var texture = new RenderTexture(width, height, 0, format, 0)
            {
                enableRandomWrite = true,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                name = name
            };
            texture.Create();
            return texture;
        }

        private void CompleteReadback(AsyncGPUReadbackRequest request, int requestGeneration)
        {
            if (requestGeneration != _generation) return;
            _readbackPending = false;
            if (request.hasError) return;
            var data = request.GetData<uint>();
            if (data.Length < StatCount) return;
            for (int i = 0; i < StatCount; i++)
            {
                _last[i] = data[i];
                _cumulative[i] += data[i];
            }
            _sampleCount++;
            HasSample = true;
        }

        private void CompleteTrackingReadback(AsyncGPUReadbackRequest request,
            int requestGeneration)
        {
            if (requestGeneration != _generation) return;
            _trackingReadbackPending = false;
            if (request.hasError) return;
            SolveTrackingCorrection(request.GetData<ModelTrackingPair>(),
                _lastSourceFrame);
        }

        private TrackingDecision SolveTrackingCorrection(
            Unity.Collections.NativeArray<ModelTrackingPair> data,
            int sourceFrame)
        {
            var normalEquation = new double[6, 6];
            var rhs = new double[6];
            int count = 0;
            for (int i = 0; i < data.Length; i++)
            {
                ModelTrackingPair pair = data[i];
                if (pair.ObservedValid.w < 0.5f) continue;
                Vector3 source = pair.ObservedValid;
                Vector3 target = pair.ModelResidual;
                Vector3 n = pair.NormalVote;
                float normalLength = n.magnitude;
                if (normalLength < 0.5f) continue;
                n /= normalLength;
                double residual = Vector3.Dot(source - target, n);
                double standardized = Math.Abs(residual) / 0.03;
                double weight = 1.0 / (1.0 + standardized * standardized);
                Vector3 cross = Vector3.Cross(source, n);
                double[] jacobian = { cross.x, cross.y, cross.z, n.x, n.y, n.z };
                for (int row = 0; row < 6; row++)
                {
                    rhs[row] += -weight * jacobian[row] * residual;
                    for (int column = 0; column < 6; column++)
                        normalEquation[row, column] +=
                            weight * jacobian[row] * jacobian[column];
                }
                count++;
            }

            _trackingPairCount = count;
            _trackingRank = 0;
            _trackingTranslationMm = 0f;
            _trackingRotationDeg = 0f;
            _trackingMeanBeforeMm = 0f;
            _trackingMeanAfterMm = 0f;
            _trackingWithin30BeforePercent = 0f;
            _trackingWithin30AfterPercent = 0f;
            _trackingTranslationClamped = false;
            _trackingRotationClamped = false;
            _trackingSampleCount++;
            if (count < 64)
                return TrackingDecision.Underdetermined(sourceFrame, count, 0);

            double maxDiagonal = 0.0;
            for (int i = 0; i < 6; i++)
                maxDiagonal = Math.Max(maxDiagonal, normalEquation[i, i]);
            if (maxDiagonal <= 1e-9)
                return TrackingDecision.Underdetermined(sourceFrame, count, 0);
            _trackingRank = EstimateNormalizedRank(normalEquation);
            if (_trackingRank < 3)
                return TrackingDecision.Underdetermined(sourceFrame, count,
                    _trackingRank);

            double damping = maxDiagonal * 1e-4;
            for (int i = 0; i < 6; i++) normalEquation[i, i] += damping;
            if (!SolveLinearSystem(normalEquation, rhs, out double[] delta))
                return TrackingDecision.Underdetermined(sourceFrame, count,
                    _trackingRank);

            Vector3 rotationVector = new((float)delta[0], (float)delta[1],
                                         (float)delta[2]);
            Vector3 translation = new((float)delta[3], (float)delta[4],
                                      (float)delta[5]);
            float rotationLength = rotationVector.magnitude;
            float rotationLimit = 1.5f * Mathf.Deg2Rad;
            if (rotationLength > rotationLimit)
            {
                _trackingRotationClamped = true;
                rotationVector *= rotationLimit / rotationLength;
            }
            if (translation.magnitude > 0.03f)
            {
                _trackingTranslationClamped = true;
                translation = translation.normalized * 0.03f;
            }
            _trackingRotationDeg = rotationVector.magnitude * Mathf.Rad2Deg;
            _trackingTranslationMm = translation.magnitude * 1000f;
            Quaternion rotation = rotationVector.sqrMagnitude > 1e-12f
                ? Quaternion.AngleAxis(_trackingRotationDeg,
                    rotationVector.normalized)
                : Quaternion.identity;
            Matrix4x4 correction = Matrix4x4.TRS(translation, rotation,
                                                  Vector3.one);

            double beforeSum = 0.0;
            double afterSum = 0.0;
            int before30 = 0;
            int after30 = 0;
            int evaluated = 0;
            for (int i = 0; i < data.Length; i++)
            {
                ModelTrackingPair pair = data[i];
                if (pair.ObservedValid.w < 0.5f) continue;
                Vector3 source = pair.ObservedValid;
                Vector3 target = pair.ModelResidual;
                Vector3 n = ((Vector3)pair.NormalVote).normalized;
                double before = Math.Abs(Vector3.Dot(source - target, n));
                Vector3 corrected = correction.MultiplyPoint3x4(source);
                double after = Math.Abs(Vector3.Dot(corrected - target, n));
                beforeSum += before;
                afterSum += after;
                if (before <= 0.03) before30++;
                if (after <= 0.03) after30++;
                evaluated++;
            }
            if (evaluated <= 0) return TrackingDecision.Failed(sourceFrame);
            _trackingMeanBeforeMm = (float)(beforeSum * 1000.0 / evaluated);
            _trackingMeanAfterMm = (float)(afterSum * 1000.0 / evaluated);
            _trackingWithin30BeforePercent = 100f * before30 / evaluated;
            _trackingWithin30AfterPercent = 100f * after30 / evaluated;
            return new TrackingDecision(sourceFrame, true, correction,
                _trackingPairCount, _trackingRank, _trackingTranslationMm,
                _trackingRotationDeg, _trackingMeanBeforeMm,
                _trackingMeanAfterMm, _trackingWithin30BeforePercent,
                _trackingWithin30AfterPercent, _trackingTranslationClamped,
                _trackingRotationClamped);
        }

        private static int EstimateNormalizedRank(double[,] matrix)
        {
            const int size = 6;
            var normalized = new double[size, size];
            var scale = new double[size];
            for (int i = 0; i < size; i++)
                scale[i] = matrix[i, i] > 1e-12
                    ? 1.0 / Math.Sqrt(matrix[i, i]) : 0.0;
            for (int row = 0; row < size; row++)
                for (int column = 0; column < size; column++)
                    normalized[row, column] = matrix[row, column] *
                                              scale[row] * scale[column];

            int rank = 0;
            const double threshold = 1e-4;
            for (int column = 0; column < size && rank < size; column++)
            {
                int pivot = rank;
                for (int row = rank + 1; row < size; row++)
                    if (Math.Abs(normalized[row, column]) >
                        Math.Abs(normalized[pivot, column])) pivot = row;
                if (Math.Abs(normalized[pivot, column]) <= threshold) continue;
                if (pivot != rank)
                    for (int c = column; c < size; c++)
                    {
                        double temporary = normalized[rank, c];
                        normalized[rank, c] = normalized[pivot, c];
                        normalized[pivot, c] = temporary;
                    }
                double divisor = normalized[rank, column];
                for (int c = column; c < size; c++)
                    normalized[rank, c] /= divisor;
                for (int row = 0; row < size; row++)
                {
                    if (row == rank) continue;
                    double factor = normalized[row, column];
                    for (int c = column; c < size; c++)
                        normalized[row, c] -= factor * normalized[rank, c];
                }
                rank++;
            }
            return rank;
        }

        private static bool SolveLinearSystem(double[,] matrix, double[] vector,
            out double[] result)
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
                    if (Math.Abs(augmented[row, pivot]) >
                        Math.Abs(augmented[best, pivot])) best = row;
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
                for (int column = pivot; column <= size; column++)
                    augmented[pivot, column] /= divisor;
                for (int row = 0; row < size; row++)
                {
                    if (row == pivot) continue;
                    double factor = augmented[row, pivot];
                    for (int column = pivot; column <= size; column++)
                        augmented[row, column] -= factor *
                                                   augmented[pivot, column];
                }
            }
            result = new double[size];
            for (int i = 0; i < size; i++) result[i] = augmented[i, size];
            return true;
        }
    }
}
