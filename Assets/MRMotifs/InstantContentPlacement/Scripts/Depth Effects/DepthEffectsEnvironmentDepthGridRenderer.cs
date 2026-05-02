using Meta.XR.EnvironmentDepth;
using Meta.XR.Samples;
using UnityEngine;
using UnityEngine.Rendering;

namespace MRMotifs.InstantContentPlacement.DepthEffects
{
    [DisallowMultipleComponent]
    [MetaCodeSample("MRMotifs-InstantContentPlacement")]
    public class DepthEffectsEnvironmentDepthGridRenderer : MonoBehaviour
    {
        private enum SurfaceDisplayMode
        {
            RepresentativeDiscs,
            StablePatchMask
        }

        private enum DebugGeometryMode
        {
            WarpedDepthSurface,
            ScreenGridPlane
        }

        [SerializeField]
        private bool showGrid = true;

        [SerializeField]
        private Camera sampleCamera;

        [SerializeField]
        private EnvironmentDepthManager environmentDepthManager;

        [SerializeField]
        private Material materialOverride;

        [Header("Sampling Window")]
        [SerializeField]
        private int sampleColumns = 22;

        [SerializeField]
        private int sampleRows = 16;

        [SerializeField]
        private float viewportCenterX = 0.5f;

        [SerializeField]
        private float viewportCenterY = 0.56f;

        [SerializeField]
        private float viewportWidth = 0.72f;

        [SerializeField]
        private float viewportHeight = 0.58f;

        [Header("Grid")]
        [SerializeField]
        private Color gridColor = new Color(0.92f, 0.96f, 0.99f, 0.88f);

        [SerializeField]
        private float cellSizeMeters = 0.22f;

        [SerializeField]
        private float representativeRadiusRatio = 0.07f;

        [SerializeField]
        private float lineHalfWidthMeters = 0.012f;

        [SerializeField]
        private SurfaceDisplayMode surfaceDisplayMode = SurfaceDisplayMode.StablePatchMask;

        [SerializeField]
        private float patchFillRatio = 0.94f;

        [SerializeField]
        private float patchConfidenceThreshold = 0.55f;

        [SerializeField]
        private int patchCellsX = 4;

        [SerializeField]
        private int patchCellsY = 4;

        [SerializeField]
        private float patchPlanarityThresholdMeters = 0.08f;

        [SerializeField]
        private int patchMinValidSupportPoints = 3;

        [SerializeField]
        private float surfaceOffsetMeters = 0.01f;

        [SerializeField]
        private float minLinearDepthMeters = 0.2f;

        [SerializeField]
        private float maxLinearDepthMeters = 3.5f;

        [SerializeField]
        private float axisAlignmentThreshold = 0.7f;

        [SerializeField]
        private float depthEdgeSuppressStartMeters = 0.06f;

        [SerializeField]
        private float depthEdgeSuppressEndMeters = 0.18f;

        [SerializeField]
        private float surfaceStretchSuppressStartMeters = 0.18f;

        [SerializeField]
        private float surfaceStretchSuppressEndMeters = 0.45f;

        [SerializeField]
        private bool debugLog;

        [Header("Debug Geometry")]
        [SerializeField]
        private DebugGeometryMode debugGeometryMode = DebugGeometryMode.WarpedDepthSurface;

        [SerializeField]
        private float debugScreenGridDistanceMeters = 1.0f;

        [SerializeField]
        private float debugScreenGridWidthMeters = 2.4f;

        [SerializeField]
        private float debugScreenGridHeightMeters = 1.8f;

        private MeshRenderer m_meshRenderer;
        private MeshFilter m_meshFilter;
        private Mesh m_gridMesh;
        private Material m_runtimeMaterial;
        private bool m_loggedMaterialFailure;

        public void Configure(Camera camera, EnvironmentDepthManager depthManager, Material overrideMaterial)
        {
            sampleCamera = camera;
            environmentDepthManager = depthManager;
            materialOverride = overrideMaterial;
        }

        private void Awake()
        {
            ResolveRefs();
            EnsureGridObject();
            RebuildMesh();
        }

        private void OnValidate()
        {
            if (!Application.isPlaying)
                return;

            RebuildMesh();
        }

        private void LateUpdate()
        {
            ResolveRefs();
            EnsureGridObject();

            bool shouldRender = showGrid && sampleCamera != null && environmentDepthManager != null &&
                                environmentDepthManager.isActiveAndEnabled && environmentDepthManager.IsDepthAvailable;
            if (m_meshRenderer != null)
                m_meshRenderer.enabled = shouldRender;

            if (!shouldRender)
                return;

            Material material = ResolveMaterial();
            if (material == null)
                return;

            ApplyMaterialProperties(material);
            if (debugLog && Time.frameCount % 180 == 0)
                Debug.Log("[DepthEffectsEnvironmentDepthGridRenderer] Rendering world-space GPU depth grid.");
        }

        private void ResolveRefs()
        {
            if (sampleCamera == null)
                sampleCamera = Camera.main;
            if (environmentDepthManager == null)
                environmentDepthManager = FindAnyObjectByType<EnvironmentDepthManager>(FindObjectsInactive.Include);
        }

        private void EnsureGridObject()
        {
            if (m_meshRenderer != null && m_meshFilter != null)
                return;

            GameObject gridObject = new GameObject("DepthEffectsEnvironmentDepthGrid");
            gridObject.transform.SetParent(transform, false);
            gridObject.transform.localPosition = Vector3.zero;
            gridObject.transform.localRotation = Quaternion.identity;
            gridObject.transform.localScale = Vector3.one;

            m_meshFilter = gridObject.AddComponent<MeshFilter>();
            m_meshRenderer = gridObject.AddComponent<MeshRenderer>();
            m_meshRenderer.shadowCastingMode = ShadowCastingMode.Off;
            m_meshRenderer.receiveShadows = false;
            m_meshRenderer.lightProbeUsage = LightProbeUsage.Off;
            m_meshRenderer.reflectionProbeUsage = ReflectionProbeUsage.Off;
            m_meshRenderer.allowOcclusionWhenDynamic = false;
        }

        private void RebuildMesh()
        {
            if (m_meshFilter == null)
                return;

            if (m_gridMesh != null)
                DestroyImmediate(m_gridMesh);

            int columns = Mathf.Max(8, sampleColumns);
            int rows = Mathf.Max(6, sampleRows);
            int vertexCount = columns * rows * 4;
            int quadCount = columns * rows;

            Vector3[] vertices = new Vector3[vertexCount];
            Vector2[] uv = new Vector2[vertexCount];
            Vector2[] uv1 = new Vector2[vertexCount];
            int[] triangles = new int[quadCount * 6];

            int vertexIndex = 0;
            int triangleOffset = 0;
            for (int row = 0; row < rows; row++)
            {
                float v = rows > 0 ? ((float)row + 0.5f) / rows : 0.5f;
                float sampleY = viewportCenterY + (v - 0.5f) * viewportHeight;
                for (int column = 0; column < columns; column++)
                {
                    float uCoord = columns > 0 ? ((float)column + 0.5f) / columns : 0.5f;
                    float sampleX = viewportCenterX + (uCoord - 0.5f) * viewportWidth;
                    Vector2 centerUv = new Vector2(sampleX, sampleY);

                    if (debugGeometryMode == DebugGeometryMode.ScreenGridPlane)
                    {
                        float x = (uCoord - 0.5f) * debugScreenGridWidthMeters;
                        float y = (v - 0.5f) * debugScreenGridHeightMeters;
                        Vector3 center = new Vector3(x, y, debugScreenGridDistanceMeters);
                        vertices[vertexIndex + 0] = center;
                        vertices[vertexIndex + 1] = center;
                        vertices[vertexIndex + 2] = center;
                        vertices[vertexIndex + 3] = center;
                    }
                    else
                    {
                        vertices[vertexIndex + 0] = Vector3.zero;
                        vertices[vertexIndex + 1] = Vector3.zero;
                        vertices[vertexIndex + 2] = Vector3.zero;
                        vertices[vertexIndex + 3] = Vector3.zero;
                    }

                    uv[vertexIndex + 0] = centerUv;
                    uv[vertexIndex + 1] = centerUv;
                    uv[vertexIndex + 2] = centerUv;
                    uv[vertexIndex + 3] = centerUv;

                    uv1[vertexIndex + 0] = new Vector2(-0.5f, -0.5f);
                    uv1[vertexIndex + 1] = new Vector2(0.5f, -0.5f);
                    uv1[vertexIndex + 2] = new Vector2(0.5f, 0.5f);
                    uv1[vertexIndex + 3] = new Vector2(-0.5f, 0.5f);

                    int i0 = vertexIndex + 0;
                    int i1 = i0 + 1;
                    int i2 = i0 + 2;
                    int i3 = i0 + 3;

                    triangles[triangleOffset + 0] = i0;
                    triangles[triangleOffset + 1] = i1;
                    triangles[triangleOffset + 2] = i2;
                    triangles[triangleOffset + 3] = i0;
                    triangles[triangleOffset + 4] = i2;
                    triangles[triangleOffset + 5] = i3;
                    triangleOffset += 6;
                    vertexIndex += 4;
                }
            }

            m_gridMesh = new Mesh
            {
                name = "DepthEffectsEnvironmentDepthGridMesh"
            };
            m_gridMesh.SetVertices(vertices);
            m_gridMesh.SetUVs(0, uv);
            m_gridMesh.SetUVs(1, uv1);
            m_gridMesh.SetTriangles(triangles, 0);
            // Vertex shader reconstructs world-space positions from environment depth,
            // so the local mesh bounds are meaningless for culling. Keep them huge.
            m_gridMesh.bounds = new Bounds(Vector3.zero, Vector3.one * 1000f);
            m_meshFilter.sharedMesh = m_gridMesh;
        }

        private Material ResolveMaterial()
        {
            Material material = materialOverride != null ? materialOverride : m_runtimeMaterial;
            if (material == null)
            {
                Shader shader = Resources.Load<Shader>("DepthEffectsEnvironmentDepthGrid");
                if (shader == null)
                {
                    if (!m_loggedMaterialFailure)
                    {
                        Debug.LogWarning("[DepthEffectsEnvironmentDepthGridRenderer] Shader resource not found.");
                        m_loggedMaterialFailure = true;
                    }
                    return null;
                }

                m_runtimeMaterial = new Material(shader)
                {
                    name = "DepthEffectsEnvironmentDepthGridRenderer_Runtime"
                };
                material = m_runtimeMaterial;
            }

            if (m_meshRenderer != null && m_meshRenderer.sharedMaterial != material)
                m_meshRenderer.sharedMaterial = material;
            return material;
        }

        private void ApplyMaterialProperties(Material material)
        {
            material.SetColor("_GridColor", gridColor);
            material.SetFloat("_CellSizeMeters", Mathf.Max(0.02f, cellSizeMeters));
            material.SetFloat("_RepresentativeRadiusRatio", Mathf.Clamp(representativeRadiusRatio, 0.02f, 0.48f));
            material.SetFloat("_LineHalfWidthMeters", Mathf.Max(0.0015f, lineHalfWidthMeters));
            material.SetFloat("_DisplayMode", surfaceDisplayMode == SurfaceDisplayMode.StablePatchMask ? 1f : 0f);
            material.SetFloat("_PatchFillRatio", Mathf.Clamp(patchFillRatio, 0.1f, 1f));
            material.SetFloat("_PatchConfidenceThreshold", Mathf.Clamp01(patchConfidenceThreshold));
            material.SetVector("_ViewportRect", new Vector4(viewportCenterX, viewportCenterY, viewportWidth, viewportHeight));
            material.SetVector("_SampleCounts", new Vector4(Mathf.Max(1, sampleColumns), Mathf.Max(1, sampleRows), 0f, 0f));
            material.SetVector("_PatchCounts", new Vector4(Mathf.Max(1, patchCellsX), Mathf.Max(1, patchCellsY), Mathf.Max(1, patchMinValidSupportPoints), 0f));
            material.SetFloat("_PatchPlanarityThreshold", Mathf.Max(0.005f, patchPlanarityThresholdMeters));
            material.SetFloat("_SurfaceOffset", Mathf.Max(0f, surfaceOffsetMeters));
            material.SetFloat("_MinLinearDepth", Mathf.Max(0.05f, minLinearDepthMeters));
            material.SetFloat("_MaxLinearDepth", Mathf.Max(minLinearDepthMeters + 0.1f, maxLinearDepthMeters));
            material.SetFloat("_AxisAlignmentThreshold", Mathf.Clamp01(axisAlignmentThreshold));
            material.SetFloat("_DepthEdgeSuppressStart", Mathf.Max(0.001f, depthEdgeSuppressStartMeters));
            material.SetFloat("_DepthEdgeSuppressEnd", Mathf.Max(depthEdgeSuppressStartMeters + 0.001f, depthEdgeSuppressEndMeters));
            material.SetFloat("_SurfaceStretchSuppressStart", Mathf.Max(0.001f, surfaceStretchSuppressStartMeters));
            material.SetFloat("_SurfaceStretchSuppressEnd", Mathf.Max(surfaceStretchSuppressStartMeters + 0.001f, surfaceStretchSuppressEndMeters));
            material.SetFloat("_DebugUseScreenGridPlane", debugGeometryMode == DebugGeometryMode.ScreenGridPlane ? 1f : 0f);
        }

        private void OnDestroy()
        {
            if (m_runtimeMaterial != null)
                Destroy(m_runtimeMaterial);
            if (m_gridMesh != null)
                Destroy(m_gridMesh);
        }
    }
}
