using System;
using System.Collections.Generic;
using Meta.XR.EnvironmentDepth;
using Meta.XR.Samples;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

namespace MRMotifs.InstantContentPlacement.DepthEffects
{
    [DisallowMultipleComponent]
    [MetaCodeSample("MRMotifs-InstantContentPlacement")]
    public class DepthEffectsSurfaceSurfelAccumulator : MonoBehaviour
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

        public struct SurfelData
        {
            public Vector3 position;
            public Vector3 normal;
            public float radius;
            public float confidence;
            public int supportCount;
            public float lastSeenTime;
            public bool retained;
            public bool activeSupport;
            public bool staleSupport;
            public bool anchored;
        }

        private struct SurfelState
        {
            public bool valid;
            public Vector3 position;
            public Vector3 normal;
            public float radius;
            public float confidence;
            public int supportCount;
            public float lastSeenTime;
            public bool retained;
            public bool anchored;
            public Vector3 anchorPosition;
            public Vector3 anchorNormal;
            public int anchorSupportFrames;
        }

        [SerializeField]
        private EnvironmentDepthManager environmentDepthManager;

        [SerializeField]
        private Camera sampleCamera;

        [Header("Sampling")]
        [SerializeField]
        private int sampleColumns = 28;

        [SerializeField]
        private int sampleRows = 20;

        [SerializeField]
        private float viewportCenterX = 0.5f;

        [SerializeField]
        private float viewportCenterY = 0.58f;

        [SerializeField]
        private float viewportWidth = 0.92f;

        [SerializeField]
        private float viewportHeight = 0.84f;

        [SerializeField]
        private float sampleIntervalSeconds = 0.05f;

        [SerializeField]
        private float minLinearDepthMeters = 0.25f;

        [SerializeField]
        private float maxLinearDepthMeters = 4.5f;

        [Header("Surfel Fusion")]
        [SerializeField]
        private float surfelRadiusMeters = 0.12f;

        [SerializeField]
        private float surfelMergeDistanceMeters = 0.14f;

        [SerializeField]
        private float surfelMergeNormalDot = 0.78f;

        [SerializeField]
        private float surfelPositionBlend = 0.28f;

        [SerializeField]
        private float surfelNormalBlend = 0.22f;

        [SerializeField]
        private float surfelConfidenceFloor = 0.18f;

        [SerializeField]
        private float surfelMaxUnseenSeconds = 18.0f;

        [SerializeField]
        private int maxSurfels = 3500;

        [Header("Surfel Memory")]
        [SerializeField]
        private int retainedMinSupportCount = 5;

        [SerializeField]
        private float retainedMinConfidence = 0.28f;

        [SerializeField]
        private float activeSupportMaxUnseenSeconds = 3.0f;

        [SerializeField]
        private float staleSupportMaxUnseenSeconds = 20.0f;

        [SerializeField]
        private float matureSurfelMatchDistanceMeters = 0.10f;

        [SerializeField]
        private float matureSurfelMatchNormalDot = 0.84f;

        [Header("Surfel Anchors")]
        [SerializeField]
        private int anchorMinSupportCount = 7;

        [SerializeField]
        private float anchorMinConfidence = 0.34f;

        [SerializeField]
        private float anchorMatchDistanceMeters = 0.08f;

        [SerializeField]
        private float anchorMatchNormalDot = 0.88f;

        [SerializeField]
        private float anchorUpdateDistanceMeters = 0.06f;

        [SerializeField]
        private int anchorUpdateSupportFrames = 3;

        [SerializeField]
        private float anchorPositionBlend = 0.14f;

        [SerializeField]
        private float anchorNormalBlend = 0.12f;

        [SerializeField]
        private bool debugLog;

        private readonly List<SurfelState> m_surfels = new();
        private readonly List<int> m_removeIndices = new();
        private ComputeShader m_copyShader;
        private ComputeBuffer m_computeBuffer;
        private NativeArray<float> m_depthPixels;
        private NativeArray<float> m_gpuReadbackBuffer;
        private AsyncGPUReadbackRequest m_pendingReadback;
        private bool m_hasPendingReadback;
        private float m_nextSampleTime;

        public int FillSurfels(List<SurfelData> output, float now)
        {
            output.Clear();
            for (int i = 0; i < m_surfels.Count; i++)
            {
                SurfelState surfel = m_surfels[i];
                if (!surfel.valid)
                    continue;
                float unseenSeconds = now - surfel.lastSeenTime;
                float maxUnseenSeconds = surfel.retained
                    ? Mathf.Max(surfelMaxUnseenSeconds, staleSupportMaxUnseenSeconds)
                    : surfelMaxUnseenSeconds;
                if (unseenSeconds > maxUnseenSeconds)
                    continue;

                output.Add(new SurfelData
                {
                    position = surfel.anchored ? surfel.anchorPosition : surfel.position,
                    normal = surfel.anchored ? surfel.anchorNormal : surfel.normal,
                    radius = surfel.radius,
                    confidence = surfel.confidence,
                    supportCount = surfel.supportCount,
                    lastSeenTime = surfel.lastSeenTime,
                    retained = surfel.retained,
                    activeSupport = unseenSeconds <= Mathf.Max(0.1f, activeSupportMaxUnseenSeconds),
                    staleSupport = unseenSeconds > Mathf.Max(0.1f, activeSupportMaxUnseenSeconds),
                    anchored = surfel.anchored
                });
            }

            return output.Count;
        }

        public void Configure(Camera camera, EnvironmentDepthManager depthManager)
        {
            sampleCamera = camera;
            environmentDepthManager = depthManager;
        }

        private void OnEnable()
        {
            ResolveRefs();
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

            if (sampleCamera == null || environmentDepthManager == null)
                return;

            if (Time.unscaledTime >= m_nextSampleTime && !m_hasPendingReadback)
            {
                m_nextSampleTime = Time.unscaledTime + Mathf.Max(0.02f, sampleIntervalSeconds);
                RequestDepthCopy();
            }
        }

        private void ResolveRefs()
        {
            if (sampleCamera == null)
                sampleCamera = Camera.main;
            if (environmentDepthManager == null)
                environmentDepthManager = FindAnyObjectByType<EnvironmentDepthManager>(FindObjectsInactive.Include);
        }

        private void EnsureResources()
        {
            if (m_copyShader == null)
                m_copyShader = Resources.Load<ComputeShader>("CopyDepthTexture");

            if (m_computeBuffer == null)
            {
                int numPixels = CopyTextureSize * CopyTextureSize * NumEyes;
                m_computeBuffer = new ComputeBuffer(numPixels, sizeof(float));
                m_depthPixels = new NativeArray<float>(numPixels, Allocator.Persistent);
                m_gpuReadbackBuffer = new NativeArray<float>(numPixels, Allocator.Persistent);
            }
        }

        private void DisposeResources()
        {
            if (m_computeBuffer != null)
            {
                m_computeBuffer.Dispose();
                m_computeBuffer = null;
            }

            if (m_depthPixels.IsCreated)
                m_depthPixels.Dispose();
            if (m_gpuReadbackBuffer.IsCreated)
                m_gpuReadbackBuffer.Dispose();

            m_hasPendingReadback = false;
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
                return;

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

            Matrix4x4 inverseMatrix = inverseMatrices[0];
            float now = Time.unscaledTime;

            for (int row = 0; row < sampleRows; row++)
            {
                float v = sampleRows > 1 ? (float)row / (sampleRows - 1) : 0.5f;
                float viewportY = viewportCenterY + (v - 0.5f) * viewportHeight;

                for (int column = 0; column < sampleColumns; column++)
                {
                    float u = sampleColumns > 1 ? (float)column / (sampleColumns - 1) : 0.5f;
                    float viewportX = viewportCenterX + (u - 0.5f) * viewportWidth;

                    Vector2Int texCoord = ViewportToDepthCoord(viewportX, viewportY);
                    float linearDepth = SampleLinearDepth(texCoord, 0);
                    if (linearDepth < minLinearDepthMeters || linearDepth > maxLinearDepthMeters)
                        continue;

                    if (!TryReconstructWorld(texCoord, linearDepth, inverseMatrix, out Vector3 worldPos))
                        continue;

                    Vector3 normal = ReconstructNormal(texCoord, 0, inverseMatrix);
                    float confidence = Mathf.Clamp01(1f - (linearDepth / Mathf.Max(minLinearDepthMeters + 0.1f, maxLinearDepthMeters)));
                    confidence = Mathf.Max(surfelConfidenceFloor, confidence);

                    int surfelIndex = FindBestSurfel(worldPos, normal, now);
                    if (surfelIndex >= 0)
                    {
                        SurfelState surfel = m_surfels[surfelIndex];
                        surfel.position = Vector3.Lerp(surfel.position, worldPos, Mathf.Clamp01(surfelPositionBlend));
                        Vector3 blendedNormal = Vector3.Lerp(surfel.normal, normal, Mathf.Clamp01(surfelNormalBlend));
                        surfel.normal = blendedNormal.sqrMagnitude > 1e-6f ? blendedNormal.normalized : Vector3.up;
                        surfel.confidence = Mathf.Lerp(surfel.confidence, confidence, 0.2f);
                        surfel.supportCount = Mathf.Min(9999, surfel.supportCount + 1);
                        surfel.lastSeenTime = now;
                        surfel.radius = surfelRadiusMeters;
                        if (ShouldRetainSurfel(surfel))
                            surfel.retained = true;
                        UpdateAnchorState(ref surfel, worldPos, normal);
                        m_surfels[surfelIndex] = surfel;
                    }
                    else if (m_surfels.Count < Mathf.Max(64, maxSurfels))
                    {
                        SurfelState surfel = new SurfelState
                        {
                            valid = true,
                            position = worldPos,
                            normal = normal.sqrMagnitude > 1e-6f ? normal.normalized : Vector3.up,
                            radius = surfelRadiusMeters,
                            confidence = confidence,
                            supportCount = 1,
                            lastSeenTime = now
                        };
                        surfel.retained = ShouldRetainSurfel(surfel);
                        surfel.anchored = false;
                        surfel.anchorPosition = surfel.position;
                        surfel.anchorNormal = surfel.normal;
                        surfel.anchorSupportFrames = 0;
                        m_surfels.Add(surfel);
                    }
                }
            }

            PruneStaleSurfels(now);

            if (debugLog)
                Debug.Log($"[DepthEffectsSurfaceSurfelAccumulator] surfels={m_surfels.Count}");
        }

        private int FindBestSurfel(Vector3 worldPos, Vector3 normal, float now)
        {
            int bestIndex = -1;
            float bestScore = float.MaxValue;

            for (int i = 0; i < m_surfels.Count; i++)
            {
                SurfelState surfel = m_surfels[i];
                if (!surfel.valid)
                    continue;
                float unseenSeconds = now - surfel.lastSeenTime;
                float maxUnseenSeconds = surfel.retained
                    ? Mathf.Max(surfelMaxUnseenSeconds, staleSupportMaxUnseenSeconds)
                    : surfelMaxUnseenSeconds;
                if (unseenSeconds > maxUnseenSeconds)
                    continue;

                Vector3 matchPosition = surfel.anchored ? surfel.anchorPosition : surfel.position;
                Vector3 matchNormal = surfel.anchored ? surfel.anchorNormal : surfel.normal;

                float distance = Vector3.Distance(matchPosition, worldPos);
                float maxDistance = surfel.retained || surfel.supportCount >= retainedMinSupportCount
                    ? Mathf.Min(Mathf.Max(0.01f, surfelMergeDistanceMeters), Mathf.Max(0.01f, matureSurfelMatchDistanceMeters))
                    : Mathf.Max(0.01f, surfelMergeDistanceMeters);
                if (surfel.anchored)
                    maxDistance = Mathf.Min(maxDistance, Mathf.Max(0.01f, anchorMatchDistanceMeters));
                if (distance > maxDistance)
                    continue;

                float normalDot = Mathf.Abs(Vector3.Dot(
                    matchNormal.sqrMagnitude > 1e-6f ? matchNormal.normalized : Vector3.up,
                    normal.sqrMagnitude > 1e-6f ? normal.normalized : Vector3.up));
                float minNormalDot = surfel.retained || surfel.supportCount >= retainedMinSupportCount
                    ? Mathf.Max(surfelMergeNormalDot, matureSurfelMatchNormalDot)
                    : surfelMergeNormalDot;
                if (surfel.anchored)
                    minNormalDot = Mathf.Max(minNormalDot, anchorMatchNormalDot);
                if (normalDot < Mathf.Clamp(minNormalDot, 0f, 0.9999f))
                    continue;

                float score = distance - normalDot * 0.02f + unseenSeconds * 0.0025f;
                if (score < bestScore)
                {
                    bestScore = score;
                    bestIndex = i;
                }
            }

            return bestIndex;
        }

        private void PruneStaleSurfels(float now)
        {
            m_removeIndices.Clear();
            for (int i = 0; i < m_surfels.Count; i++)
            {
                SurfelState surfel = m_surfels[i];
                float maxUnseenSeconds = surfel.retained
                    ? Mathf.Max(surfelMaxUnseenSeconds, staleSupportMaxUnseenSeconds)
                    : surfelMaxUnseenSeconds;
                if (now - surfel.lastSeenTime > maxUnseenSeconds)
                    m_removeIndices.Add(i);
            }

            for (int i = m_removeIndices.Count - 1; i >= 0; i--)
                m_surfels.RemoveAt(m_removeIndices[i]);
        }

        private bool ShouldRetainSurfel(SurfelState surfel)
        {
            return surfel.supportCount >= Mathf.Max(1, retainedMinSupportCount) &&
                   surfel.confidence >= retainedMinConfidence;
        }

        private void UpdateAnchorState(ref SurfelState surfel, Vector3 observedPosition, Vector3 observedNormal)
        {
            if (!surfel.anchored)
            {
                if (surfel.supportCount >= Mathf.Max(1, anchorMinSupportCount) &&
                    surfel.confidence >= anchorMinConfidence)
                {
                    surfel.anchored = true;
                    surfel.anchorPosition = surfel.position;
                    surfel.anchorNormal = surfel.normal.sqrMagnitude > 1e-6f ? surfel.normal.normalized : Vector3.up;
                    surfel.anchorSupportFrames = 0;
                }
                return;
            }

            Vector3 anchorNormal = surfel.anchorNormal.sqrMagnitude > 1e-6f ? surfel.anchorNormal.normalized : Vector3.up;
            Vector3 observedResolvedNormal = observedNormal.sqrMagnitude > 1e-6f ? observedNormal.normalized : Vector3.up;
            float distance = Vector3.Distance(surfel.anchorPosition, observedPosition);
            float normalDot = Mathf.Abs(Vector3.Dot(anchorNormal, observedResolvedNormal));

            bool supportsAnchorUpdate =
                distance <= Mathf.Max(0.01f, anchorUpdateDistanceMeters) &&
                normalDot >= Mathf.Clamp(anchorMatchNormalDot, 0f, 0.9999f);

            if (!supportsAnchorUpdate)
            {
                surfel.anchorSupportFrames = 0;
                return;
            }

            surfel.anchorSupportFrames = Mathf.Min(999, surfel.anchorSupportFrames + 1);
            if (surfel.anchorSupportFrames < Mathf.Max(1, anchorUpdateSupportFrames))
                return;

            surfel.anchorPosition = Vector3.Lerp(surfel.anchorPosition, observedPosition, Mathf.Clamp01(anchorPositionBlend));
            Vector3 blendedNormal = Vector3.Lerp(surfel.anchorNormal, observedResolvedNormal, Mathf.Clamp01(anchorNormalBlend));
            surfel.anchorNormal = blendedNormal.sqrMagnitude > 1e-6f ? blendedNormal.normalized : Vector3.up;
            surfel.anchorSupportFrames = 0;
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

        private Vector3 ReconstructNormal(Vector2Int texCoord, int eyeIndex, Matrix4x4 inverseMatrix)
        {
            float centerDepth = SampleLinearDepth(texCoord, eyeIndex);
            if (!TryReconstructWorld(texCoord, centerDepth, inverseMatrix, out Vector3 center))
                return Vector3.up;

            Vector2Int rightCoord = new(Mathf.Min(CopyTextureSize - 1, texCoord.x + 1), texCoord.y);
            Vector2Int upCoord = new(texCoord.x, Mathf.Min(CopyTextureSize - 1, texCoord.y + 1));
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

        private static Vector2Int ViewportToDepthCoord(float viewportX, float viewportY)
        {
            int x = Mathf.Clamp(Mathf.RoundToInt(Mathf.Clamp01(viewportX) * (CopyTextureSize - 1)), 0, CopyTextureSize - 1);
            int y = Mathf.Clamp(Mathf.RoundToInt((1f - Mathf.Clamp01(viewportY)) * (CopyTextureSize - 1)), 0, CopyTextureSize - 1);
            return new Vector2Int(x, y);
        }

        private static bool TryReconstructWorld(Vector2Int texCoord, float linearDepth, Matrix4x4 inverseMatrix, out Vector3 worldPos)
        {
            worldPos = Vector3.zero;
            if (linearDepth <= 0f)
                return false;

            Vector4 zParams = Shader.GetGlobalVector(EnvironmentDepthZBufferParamsId);
            float clipDepth = zParams.x / linearDepth - zParams.y;
            float oneOverSize = 1f / CopyTextureSize;
            Vector4 clipPos = new(
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

        private static bool IsFinite(Vector4 value)
        {
            return !(float.IsNaN(value.x) || float.IsInfinity(value.x) ||
                     float.IsNaN(value.y) || float.IsInfinity(value.y) ||
                     float.IsNaN(value.z) || float.IsInfinity(value.z) ||
                     float.IsNaN(value.w) || float.IsInfinity(value.w));
        }

        private static bool IsFinite(Vector3 value)
        {
            return !(float.IsNaN(value.x) || float.IsInfinity(value.x) ||
                     float.IsNaN(value.y) || float.IsInfinity(value.y) ||
                     float.IsNaN(value.z) || float.IsInfinity(value.z));
        }
    }
}
