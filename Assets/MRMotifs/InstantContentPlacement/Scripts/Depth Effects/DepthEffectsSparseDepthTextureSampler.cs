using System;
using Meta.XR.EnvironmentDepth;
using Meta.XR.Samples;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

namespace MRMotifs.InstantContentPlacement.DepthEffects
{
    [DisallowMultipleComponent]
    [MetaCodeSample("MRMotifs-InstantContentPlacement")]
    public class DepthEffectsSparseDepthTextureSampler : MonoBehaviour
    {
        private const int CopyTextureSize = 128;
        private const int NumEyes = 2;
        private static readonly int EnvironmentDepthTextureId = Shader.PropertyToID("_EnvironmentDepthTexture");
        private static readonly int EnvironmentDepthTextureSizeId = Shader.PropertyToID("_EnvironmentDepthTextureSize");
        private static readonly int EnvironmentDepthZBufferParamsId = Shader.PropertyToID("_EnvironmentDepthZBufferParams");
        private static readonly int EnvironmentDepthInverseReprojectionMatricesId =
            Shader.PropertyToID("_EnvironmentDepthInverseReprojectionMatrices");
        private static readonly int EnvironmentDepthReprojectionMatricesId =
            Shader.PropertyToID("_EnvironmentDepthReprojectionMatrices");
        private static readonly int CopiedDepthTextureId = Shader.PropertyToID("_CopiedDepthTexture");

        [Header("Refs")]
        [SerializeField]
        private EnvironmentDepthManager environmentDepthManager;

        [SerializeField]
        private Camera sampleCamera;

        [SerializeField]
        private ScanCoverVoxelPointAccumulator voxelPointAccumulator;

        [Header("Sampling")]
        [SerializeField]
        private int sampleColumns = 18;

        [SerializeField]
        private int sampleRows = 12;

        [SerializeField]
        private float viewportCenterX = 0.5f;

        [SerializeField]
        private float viewportCenterY = 0.56f;

        [SerializeField]
        private float viewportWidth = 0.72f;

        [SerializeField]
        private float viewportHeight = 0.58f;

        [SerializeField]
        private float sampleIntervalSeconds = 0.08f;

        [SerializeField]
        private float minLinearDepthMeters = 0.25f;

        [SerializeField]
        private float maxLinearDepthMeters = 3.5f;

        [SerializeField]
        private float minConfidence = 0.18f;

        [SerializeField]
        private bool debugLog;

        public int LastSampleCount { get; private set; }
        public int LastAcceptedCount { get; private set; }
        public int LastRejectedDepthCount { get; private set; }
        public int LastRejectedInvalidCount { get; private set; }

        private ComputeShader m_copyShader;
        private ComputeBuffer m_computeBuffer;
        private NativeArray<float> m_depthPixels;
        private NativeArray<float> m_gpuReadbackBuffer;
        private AsyncGPUReadbackRequest m_pendingReadback;
        private bool m_hasPendingReadback;
        private float m_nextSampleTime;
        private float m_sampleScale = 1f;
        private float m_intervalScale = 1f;
        private float m_rayDistanceScale = 1f;

        public void Configure(EnvironmentDepthManager depthManager, Camera camera, ScanCoverVoxelPointAccumulator accumulator)
        {
            environmentDepthManager = depthManager;
            sampleCamera = camera;
            voxelPointAccumulator = accumulator;
        }

        public void ApplyPerformanceSlice(float sampleScale, float intervalScale, float rayDistanceScale)
        {
            m_sampleScale = Mathf.Max(0.1f, sampleScale);
            m_intervalScale = Mathf.Max(0.1f, intervalScale);
            m_rayDistanceScale = Mathf.Max(0.1f, rayDistanceScale);
        }

        private void OnEnable()
        {
            EnsureResources();
            m_nextSampleTime = Time.unscaledTime;
        }

        private void OnDisable()
        {
            DisposeResources();
        }

        private void Update()
        {
            ResolveRefs();
            UpdatePendingReadback();

            if (environmentDepthManager == null || sampleCamera == null || voxelPointAccumulator == null)
                return;

            if (Time.unscaledTime < m_nextSampleTime || m_hasPendingReadback)
                return;

            m_nextSampleTime = Time.unscaledTime + ResolveSampleIntervalSeconds();
            RequestDepthCopy();
        }

        private void ResolveRefs()
        {
            if (environmentDepthManager == null)
                environmentDepthManager = FindAnyObjectByType<EnvironmentDepthManager>(FindObjectsInactive.Include);
            if (sampleCamera == null)
                sampleCamera = Camera.main;
            if (voxelPointAccumulator == null)
                voxelPointAccumulator = GetComponent<ScanCoverVoxelPointAccumulator>();
        }

        private void EnsureResources()
        {
            if (m_computeBuffer != null)
                return;

            m_copyShader = Resources.Load<ComputeShader>("CopyDepthTexture");
            if (m_copyShader == null)
            {
                Debug.LogWarning("[DepthEffectsSparseDepthTextureSampler] CopyDepthTexture compute shader not found.");
                return;
            }

            int numPixels = CopyTextureSize * CopyTextureSize * NumEyes;
            m_computeBuffer = new ComputeBuffer(numPixels, sizeof(float));
            m_depthPixels = new NativeArray<float>(numPixels, Allocator.Persistent);
            m_gpuReadbackBuffer = new NativeArray<float>(numPixels, Allocator.Persistent);
        }

        private void DisposeResources()
        {
            if (m_hasPendingReadback)
                m_hasPendingReadback = false;

            if (m_computeBuffer != null)
            {
                m_computeBuffer.Dispose();
                m_computeBuffer = null;
            }

            if (m_depthPixels.IsCreated)
                m_depthPixels.Dispose();
            if (m_gpuReadbackBuffer.IsCreated)
                m_gpuReadbackBuffer.Dispose();
        }

        private void RequestDepthCopy()
        {
            if (m_copyShader == null || m_computeBuffer == null || !m_gpuReadbackBuffer.IsCreated)
                return;

            if (environmentDepthManager == null || !environmentDepthManager.isActiveAndEnabled || !environmentDepthManager.IsDepthAvailable)
                return;

            RenderTexture depthTexture = Shader.GetGlobalTexture(EnvironmentDepthTextureId) as RenderTexture;
            if (depthTexture == null)
                return;

            Vector4 zParams = Shader.GetGlobalVector(EnvironmentDepthZBufferParamsId);
            m_copyShader.SetTexture(0, EnvironmentDepthTextureId, depthTexture);
            m_copyShader.SetFloat(EnvironmentDepthTextureSizeId, depthTexture.width);
            m_copyShader.SetVector(EnvironmentDepthZBufferParamsId, zParams);
            m_copyShader.SetBuffer(0, CopiedDepthTextureId, m_computeBuffer);
            m_copyShader.Dispatch(0, 1, 1, 1);

            m_pendingReadback = AsyncGPUReadback.RequestIntoNativeArray(ref m_gpuReadbackBuffer, m_computeBuffer);
            m_hasPendingReadback = true;
        }

        private void UpdatePendingReadback()
        {
            if (!m_hasPendingReadback || !m_pendingReadback.done)
                return;

            m_hasPendingReadback = false;
            if (m_pendingReadback.hasError)
            {
                if (debugLog)
                    Debug.LogWarning("[DepthEffectsSparseDepthTextureSampler] AsyncGPUReadback failed.");
                return;
            }

            (m_depthPixels, m_gpuReadbackBuffer) = (m_gpuReadbackBuffer, m_depthPixels);
            ConsumeDepthSamples();
        }

        private void ConsumeDepthSamples()
        {
            Matrix4x4[] inverseMatrices = Shader.GetGlobalMatrixArray(EnvironmentDepthInverseReprojectionMatricesId);
            if (inverseMatrices == null || inverseMatrices.Length <= 0)
            {
                Matrix4x4[] reprojectionMatrices = Shader.GetGlobalMatrixArray(EnvironmentDepthReprojectionMatricesId);
                if (reprojectionMatrices == null || reprojectionMatrices.Length <= 0)
                    return;

                inverseMatrices = new Matrix4x4[reprojectionMatrices.Length];
                for (int i = 0; i < reprojectionMatrices.Length; i++)
                    inverseMatrices[i] = reprojectionMatrices[i].inverse;
            }

            int eyeIndex = 0;
            float maxDepth = Mathf.Max(minLinearDepthMeters + 0.1f, maxLinearDepthMeters * m_rayDistanceScale);
            float minDepth = Mathf.Max(0.05f, minLinearDepthMeters);
            int columns = Mathf.Max(4, Mathf.RoundToInt(sampleColumns * m_sampleScale));
            int rows = Mathf.Max(4, Mathf.RoundToInt(sampleRows * m_sampleScale));
            LastSampleCount = columns * rows;
            LastAcceptedCount = 0;
            LastRejectedDepthCount = 0;
            LastRejectedInvalidCount = 0;

            for (int row = 0; row < rows; row++)
            {
                float v = rows > 1 ? (float)row / (rows - 1) : 0.5f;
                float viewportY = viewportCenterY + (v - 0.5f) * viewportHeight;
                for (int column = 0; column < columns; column++)
                {
                    float u = columns > 1 ? (float)column / (columns - 1) : 0.5f;
                    float viewportX = viewportCenterX + (u - 0.5f) * viewportWidth;

                    Vector2Int texCoord = ViewportToDepthCoord(viewportX, viewportY);
                    float linearDepth = SampleLinearDepth(texCoord, eyeIndex);
                    if (linearDepth < minDepth || linearDepth > maxDepth)
                    {
                        LastRejectedDepthCount++;
                        continue;
                    }

                    if (!TryReconstructWorld(texCoord, linearDepth, inverseMatrices[Mathf.Min(eyeIndex, inverseMatrices.Length - 1)], out Vector3 worldPos))
                    {
                        LastRejectedInvalidCount++;
                        continue;
                    }

                    Vector3 normal = ReconstructNormal(texCoord, eyeIndex, inverseMatrices[Mathf.Min(eyeIndex, inverseMatrices.Length - 1)]);
                    float confidence = Mathf.Clamp01(1f - (linearDepth / maxDepth));
                    confidence = Mathf.Max(minConfidence, confidence);
                    voxelPointAccumulator.AddObservation(worldPos, normal, confidence, Time.unscaledTime);
                    LastAcceptedCount++;
                }
            }

            voxelPointAccumulator.Prune(Time.unscaledTime);
            if (debugLog)
            {
                Debug.Log(
                    $"[DepthEffectsSparseDepthTextureSampler] samples={LastSampleCount}, accepted={LastAcceptedCount}, rejectDepth={LastRejectedDepthCount}, rejectInvalid={LastRejectedInvalidCount}");
            }
        }

        private static bool TryReconstructWorld(Vector2Int texCoord, float linearDepth, Matrix4x4 inverseMatrix, out Vector3 worldPos)
        {
            worldPos = Vector3.zero;
            if (linearDepth <= 0f)
                return false;

            Vector4 zParams = Shader.GetGlobalVector(EnvironmentDepthZBufferParamsId);
            float clipDepth = zParams.x / linearDepth - zParams.y;
            float oneOverSize = 1f / CopyTextureSize;
            Vector4 clipPos = new Vector4(
                texCoord.x * oneOverSize * 2f - 1f,
                texCoord.y * oneOverSize * 2f - 1f,
                clipDepth,
                1f);

            Vector4 worldH = inverseMatrix * clipPos;
            if (Mathf.Abs(worldH.w) <= 1e-5f || !IsFinite(worldH))
                return false;

            Vector4 resolvedWorld = worldH / worldH.w;
            worldPos = new Vector3(resolvedWorld.x, resolvedWorld.y, resolvedWorld.z);
            return IsFinite(worldPos);
        }

        private Vector3 ReconstructNormal(Vector2Int texCoord, int eyeIndex, Matrix4x4 inverseMatrix)
        {
            Vector3 center = default;
            float centerDepth = SampleLinearDepth(texCoord, eyeIndex);
            if (!TryReconstructWorld(texCoord, centerDepth, inverseMatrix, out center))
                return Vector3.up;

            Vector2Int rightCoord = new Vector2Int(Mathf.Min(CopyTextureSize - 1, texCoord.x + 1), texCoord.y);
            Vector2Int upCoord = new Vector2Int(texCoord.x, Mathf.Min(CopyTextureSize - 1, texCoord.y + 1));
            if (!TryReconstructWorld(rightCoord, SampleLinearDepth(rightCoord, eyeIndex), inverseMatrix, out Vector3 right))
                return Vector3.up;
            if (!TryReconstructWorld(upCoord, SampleLinearDepth(upCoord, eyeIndex), inverseMatrix, out Vector3 up))
                return Vector3.up;

            Vector3 horizontal = right - center;
            Vector3 vertical = up - center;
            if (horizontal.sqrMagnitude <= 1e-6f || vertical.sqrMagnitude <= 1e-6f)
                return Vector3.up;

            Vector3 normal = -Vector3.Cross(horizontal, vertical).normalized;
            return normal.sqrMagnitude > 1e-5f ? normal : Vector3.up;
        }

        private float SampleLinearDepth(Vector2Int texCoord, int eyeIndex)
        {
            if (!m_depthPixels.IsCreated)
                return 0f;

            int clampedEye = Mathf.Clamp(eyeIndex, 0, NumEyes - 1);
            int index = texCoord.x + texCoord.y * CopyTextureSize + CopyTextureSize * CopyTextureSize * clampedEye;
            if (index < 0 || index >= m_depthPixels.Length)
                return 0f;
            return m_depthPixels[index];
        }

        private static Vector2Int ViewportToDepthCoord(float viewportX, float viewportY)
        {
            int x = Mathf.Clamp(Mathf.RoundToInt(Mathf.Clamp01(viewportX) * (CopyTextureSize - 1)), 0, CopyTextureSize - 1);
            int y = Mathf.Clamp(Mathf.RoundToInt((1f - Mathf.Clamp01(viewportY)) * (CopyTextureSize - 1)), 0, CopyTextureSize - 1);
            return new Vector2Int(x, y);
        }

        private float ResolveSampleIntervalSeconds()
        {
            return Mathf.Max(0.03f, sampleIntervalSeconds * m_intervalScale);
        }

        private static bool IsFinite(Vector4 value)
        {
            return IsFinite((Vector3)value) && !float.IsNaN(value.w) && !float.IsInfinity(value.w);
        }

        private static bool IsFinite(Vector3 value)
        {
            return
                !float.IsNaN(value.x) && !float.IsInfinity(value.x) &&
                !float.IsNaN(value.y) && !float.IsInfinity(value.y) &&
                !float.IsNaN(value.z) && !float.IsInfinity(value.z);
        }
    }
}
