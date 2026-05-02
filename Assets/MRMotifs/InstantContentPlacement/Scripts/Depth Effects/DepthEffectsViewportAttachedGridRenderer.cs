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
    public class DepthEffectsViewportAttachedGridRenderer : MonoBehaviour
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

        [SerializeField]
        private EnvironmentDepthManager environmentDepthManager;

        [SerializeField]
        private Camera sampleCamera;

        [SerializeField]
        private Material materialOverride;

        [Header("Sampling")]
        [SerializeField]
        private int gridColumns = 26;

        [SerializeField]
        private int gridRows = 20;

        [SerializeField]
        private float viewportCenterX = 0.5f;

        [SerializeField]
        private float viewportCenterY = 0.58f;

        [SerializeField]
        private float viewportWidth = 0.9f;

        [SerializeField]
        private float viewportHeight = 0.82f;

        [SerializeField]
        private float sampleIntervalSeconds = 0.04f;

        [SerializeField]
        private float minLinearDepthMeters = 0.25f;

        [SerializeField]
        private float maxLinearDepthMeters = 4.5f;

        [Header("Display")]
        [SerializeField]
        private bool showGridPoints = true;

        [SerializeField]
        private PrimitiveType pointPrimitive = PrimitiveType.Cube;

        [SerializeField]
        private float pointScaleMeters = 0.018f;

        [SerializeField]
        private float surfaceOffsetMeters = 0.008f;

        [SerializeField]
        private float holdSeconds = 0.35f;

        [SerializeField]
        private float positionBlend = 0.35f;

        [SerializeField]
        private Color pointColor = new(0.92f, 0.96f, 0.99f, 0.96f);

        private struct GridPointState
        {
            public bool valid;
            public float lastSeenTime;
            public Vector3 worldPos;
        }

        private ComputeShader m_copyShader;
        private ComputeBuffer m_computeBuffer;
        private NativeArray<float> m_depthPixels;
        private NativeArray<float> m_gpuReadbackBuffer;
        private AsyncGPUReadbackRequest m_pendingReadback;
        private bool m_hasPendingReadback;
        private float m_nextSampleTime;
        private Mesh m_pointMesh;
        private Material m_runtimeMaterial;
        private MaterialPropertyBlock m_propertyBlock;
        private GridPointState[] m_gridPoints;
        private Matrix4x4[] m_matrices = Array.Empty<Matrix4x4>();

        public void Configure(Camera camera, EnvironmentDepthManager depthManager, Material overrideMaterial)
        {
            sampleCamera = camera;
            environmentDepthManager = depthManager;
            materialOverride = overrideMaterial;
        }

        private void OnEnable()
        {
            ResolveRefs();
            EnsureResources();
            EnsureGridState();
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

            if (!showGridPoints || sampleCamera == null || environmentDepthManager == null)
                return;

            if (Time.unscaledTime >= m_nextSampleTime && !m_hasPendingReadback)
            {
                m_nextSampleTime = Time.unscaledTime + Mathf.Max(0.02f, sampleIntervalSeconds);
                RequestDepthCopy();
            }
        }

        private void LateUpdate()
        {
            if (!showGridPoints)
                return;

            RenderAttachedGrid();
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

            if (m_pointMesh == null)
                m_pointMesh = ResolvePrimitiveMesh(pointPrimitive);

            if (m_runtimeMaterial == null)
            {
                Shader shader = Shader.Find("Universal Render Pipeline/Lit");
                if (shader == null)
                    shader = Shader.Find("Standard");
                if (shader != null)
                {
                    m_runtimeMaterial = new Material(shader)
                    {
                        name = "DepthEffectsViewportAttachedGridRenderer_Runtime"
                    };
                    m_runtimeMaterial.enableInstancing = true;
                    m_runtimeMaterial.color = pointColor;
                    if (m_runtimeMaterial.HasProperty("_Surface"))
                        m_runtimeMaterial.SetFloat("_Surface", 0f);
                    if (m_runtimeMaterial.HasProperty("_ZWrite"))
                        m_runtimeMaterial.SetFloat("_ZWrite", 1f);
                    if (m_runtimeMaterial.HasProperty("_Cull"))
                        m_runtimeMaterial.SetFloat("_Cull", (float)CullMode.Off);
                }
            }

            m_propertyBlock ??= new MaterialPropertyBlock();
        }

        private void EnsureGridState()
        {
            int count = Mathf.Max(1, gridColumns * gridRows);
            if (m_gridPoints == null || m_gridPoints.Length != count)
                m_gridPoints = new GridPointState[count];
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
            EnsureGridState();

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

            for (int row = 0; row < gridRows; row++)
            {
                float v = gridRows > 1 ? (float)row / (gridRows - 1) : 0.5f;
                float viewportY = viewportCenterY + (v - 0.5f) * viewportHeight;

                for (int column = 0; column < gridColumns; column++)
                {
                    float u = gridColumns > 1 ? (float)column / (gridColumns - 1) : 0.5f;
                    float viewportX = viewportCenterX + (u - 0.5f) * viewportWidth;
                    int index = column + row * gridColumns;

                    Vector2Int texCoord = ViewportToDepthCoord(viewportX, viewportY);
                    float linearDepth = SampleLinearDepth(texCoord, 0);
                    if (linearDepth < minLinearDepthMeters || linearDepth > maxLinearDepthMeters)
                        continue;

                    if (!TryReconstructWorld(texCoord, linearDepth, inverseMatrix, out Vector3 worldPos))
                        continue;

                    GridPointState state = m_gridPoints[index];
                    if (!state.valid)
                    {
                        state.valid = true;
                        state.worldPos = worldPos;
                    }
                    else
                    {
                        state.worldPos = Vector3.Lerp(state.worldPos, worldPos, Mathf.Clamp01(positionBlend));
                    }

                    state.lastSeenTime = now;
                    m_gridPoints[index] = state;
                }
            }
        }

        private void RenderAttachedGrid()
        {
            if (m_pointMesh == null)
                return;

            Material material = materialOverride != null ? materialOverride : m_runtimeMaterial;
            if (material == null)
                return;

            float now = Time.unscaledTime;
            int visibleCount = 0;
            float pointScale = Mathf.Max(0.001f, pointScaleMeters);
            Vector3 scale = Vector3.one * pointScale;

            EnsureMatrixCapacity(Mathf.Max(1, gridColumns * gridRows));

            for (int i = 0; i < m_gridPoints.Length; i++)
            {
                GridPointState state = m_gridPoints[i];
                if (!state.valid)
                    continue;
                if (now - state.lastSeenTime > Mathf.Max(0.05f, holdSeconds))
                    continue;

                Vector3 offsetPos = state.worldPos + sampleCamera.transform.forward * Mathf.Max(0f, surfaceOffsetMeters);
                m_matrices[visibleCount++] = Matrix4x4.TRS(offsetPos, Quaternion.identity, scale);
            }

            if (visibleCount <= 0)
                return;

            m_propertyBlock.Clear();
            m_propertyBlock.SetColor("_BaseColor", pointColor);
            m_propertyBlock.SetColor("_Color", pointColor);

            const int batchSize = 1023;
            int drawn = 0;
            while (drawn < visibleCount)
            {
                int count = Mathf.Min(batchSize, visibleCount - drawn);
                Graphics.DrawMeshInstanced(
                    m_pointMesh,
                    0,
                    material,
                    m_matrices,
                    count,
                    m_propertyBlock,
                    ShadowCastingMode.Off,
                    false,
                    gameObject.layer,
                    sampleCamera);
                drawn += count;
            }
        }

        private void EnsureMatrixCapacity(int count)
        {
            int required = Mathf.Max(1, Mathf.NextPowerOfTwo(count));
            if (m_matrices.Length < required)
                m_matrices = new Matrix4x4[required];
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

        private static Mesh ResolvePrimitiveMesh(PrimitiveType primitiveType)
        {
            string resourceName = primitiveType switch
            {
                PrimitiveType.Cube => "Cube.fbx",
                PrimitiveType.Sphere => "Sphere.fbx",
                PrimitiveType.Capsule => "Capsule.fbx",
                PrimitiveType.Cylinder => "Cylinder.fbx",
                PrimitiveType.Plane => "Plane.fbx",
                PrimitiveType.Quad => "Quad.fbx",
                _ => "Cube.fbx"
            };

            Mesh mesh = Resources.GetBuiltinResource<Mesh>(resourceName);
            if (mesh != null)
                return mesh;

            GameObject temp = GameObject.CreatePrimitive(primitiveType);
            Mesh resolved = temp.GetComponent<MeshFilter>()?.sharedMesh;
            if (Application.isPlaying)
                Destroy(temp);
            else
                DestroyImmediate(temp);
            return resolved;
        }
    }
}
