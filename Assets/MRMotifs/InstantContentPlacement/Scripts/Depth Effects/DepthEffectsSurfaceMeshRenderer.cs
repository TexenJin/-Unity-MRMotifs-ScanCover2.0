using System.Collections.Generic;
using Meta.XR.Samples;
using UnityEngine;
using UnityEngine.Rendering;

namespace MRMotifs.InstantContentPlacement.DepthEffects
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(MeshFilter))]
    [RequireComponent(typeof(MeshRenderer))]
    [MetaCodeSample("MRMotifs-InstantContentPlacement")]
    public class DepthEffectsSurfaceMeshRenderer : MonoBehaviour
    {
        private enum SurfaceDisplayMode
        {
            Nodes = 0,
            Wireframe = 1,
            NodesAndWireframe = 2
        }

        [SerializeField]
        private DepthEffectsSurfaceSurfelMesher surfelMesher;

        [SerializeField]
        private SurfaceDisplayMode displayMode = SurfaceDisplayMode.Nodes;

        [SerializeField]
        private Material materialOverride;

        [SerializeField]
        private Color lineColor = new(0.92f, 0.96f, 0.99f, 0.92f);

        [SerializeField]
        private Color nodeColor = new(0.82f, 0.9f, 0.99f, 0.95f);

        [SerializeField]
        private float nodeScaleMeters = 0.035f;

        [SerializeField]
        private float refreshIntervalSeconds = 0.12f;

        private readonly List<Vector3> m_vertices = new();
        private readonly List<int> m_lineIndices = new();
        private readonly List<Vector3> m_nodePositions = new();
        private readonly List<GameObject> m_nodeObjects = new();
        private MeshFilter m_meshFilter;
        private MeshRenderer m_meshRenderer;
        private Mesh m_runtimeMesh;
        private Material m_runtimeMaterial;
        private Material m_nodeMaterial;
        private float m_nextRefreshTime;

        public void Configure(DepthEffectsSurfaceSurfelMesher mesher, Material overrideMaterial)
        {
            surfelMesher = mesher;
            materialOverride = overrideMaterial;
        }

        private void OnEnable()
        {
            EnsureResources();
            m_nextRefreshTime = Time.unscaledTime;
        }

        private void LateUpdate()
        {
            EnsureResources();
            if (surfelMesher == null)
                return;

            if (Time.unscaledTime < m_nextRefreshTime)
                return;

            m_nextRefreshTime = Time.unscaledTime + Mathf.Max(0.05f, refreshIntervalSeconds);
            RebuildMesh();
        }

        private void EnsureResources()
        {
            if (m_meshFilter == null)
                m_meshFilter = GetComponent<MeshFilter>();
            if (m_meshRenderer == null)
                m_meshRenderer = GetComponent<MeshRenderer>();
            if (surfelMesher == null)
                surfelMesher = GetComponent<DepthEffectsSurfaceSurfelMesher>();

            if (m_runtimeMesh == null)
            {
                m_runtimeMesh = new Mesh
                {
                    name = "DepthEffectsSurfaceSurfelWireframe"
                };
                m_runtimeMesh.MarkDynamic();
                m_meshFilter.sharedMesh = m_runtimeMesh;
            }

            if (m_runtimeMaterial == null)
            {
                Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
                if (shader == null)
                    shader = Shader.Find("Unlit/Color");
                if (shader != null)
                {
                    m_runtimeMaterial = new Material(shader)
                    {
                        name = "DepthEffectsSurfaceSurfelWireframe_Runtime"
                    };
                    if (m_runtimeMaterial.HasProperty("_BaseColor"))
                        m_runtimeMaterial.SetColor("_BaseColor", lineColor);
                    if (m_runtimeMaterial.HasProperty("_Color"))
                        m_runtimeMaterial.SetColor("_Color", lineColor);
                    if (m_runtimeMaterial.HasProperty("_Cull"))
                        m_runtimeMaterial.SetFloat("_Cull", (float)CullMode.Off);
                    m_runtimeMaterial.enableInstancing = true;
                }
            }

            if (m_nodeMaterial == null && m_runtimeMaterial != null)
            {
                m_nodeMaterial = new Material(m_runtimeMaterial)
                {
                    name = "DepthEffectsSurfaceSurfelNodes_Runtime"
                };
                if (m_nodeMaterial.HasProperty("_BaseColor"))
                    m_nodeMaterial.SetColor("_BaseColor", nodeColor);
                if (m_nodeMaterial.HasProperty("_Color"))
                    m_nodeMaterial.SetColor("_Color", nodeColor);
                m_nodeMaterial.enableInstancing = true;
            }

            Material lineMaterial = materialOverride != null ? materialOverride : m_runtimeMaterial;
            m_meshRenderer.sharedMaterial = displayMode == SurfaceDisplayMode.Nodes && m_nodeMaterial != null
                ? m_nodeMaterial
                : lineMaterial;
            m_meshRenderer.shadowCastingMode = ShadowCastingMode.Off;
            m_meshRenderer.receiveShadows = false;
        }

        private void RebuildMesh()
        {
            if (displayMode == SurfaceDisplayMode.Nodes)
            {
                RebuildNodeMesh();
                return;
            }

            RebuildWireframeMesh();
            RebuildNodeMatrices();
        }

        private void RebuildWireframeMesh()
        {
            m_runtimeMesh.Clear();
            SetNodeObjectsVisible(false);
            if (displayMode == SurfaceDisplayMode.Nodes)
                return;

            int lineIndexCount = surfelMesher.CopyWireframe(m_vertices, m_lineIndices);
            if (lineIndexCount <= 0 || m_vertices.Count <= 0)
                return;

            m_runtimeMesh.SetVertices(m_vertices);
            m_runtimeMesh.SetIndices(m_lineIndices, MeshTopology.Lines, 0);
            m_runtimeMesh.RecalculateBounds();
        }

        private void RebuildNodeMatrices()
        {
            m_nodePositions.Clear();
            SetNodeObjectsVisible(false);
            if (displayMode == SurfaceDisplayMode.Wireframe)
                return;

            surfelMesher.CopyNodes(m_nodePositions);
        }

        private void RebuildNodeMesh()
        {
            m_runtimeMesh.Clear();
            SetNodeObjectsVisible(true);
            surfelMesher.CopyNodes(m_nodePositions);
            SyncNodeObjects();
        }

        private void SyncNodeObjects()
        {
            int targetCount = m_nodePositions.Count;
            while (m_nodeObjects.Count < targetCount)
                m_nodeObjects.Add(CreateNodeObject(m_nodeObjects.Count));

            Vector3 scale = Vector3.one * Mathf.Max(0.005f, nodeScaleMeters);
            for (int i = 0; i < m_nodeObjects.Count; i++)
            {
                bool active = i < targetCount;
                GameObject node = m_nodeObjects[i];
                if (node.activeSelf != active)
                    node.SetActive(active);
                if (!active)
                    continue;

                Transform nodeTransform = node.transform;
                nodeTransform.position = m_nodePositions[i];
                nodeTransform.rotation = Quaternion.identity;
                nodeTransform.localScale = scale;
            }
        }

        private GameObject CreateNodeObject(int index)
        {
            GameObject node = GameObject.CreatePrimitive(PrimitiveType.Cube);
            node.name = $"SurfaceNode_{index}";
            node.transform.SetParent(transform, worldPositionStays: false);
            Collider collider = node.GetComponent<Collider>();
            if (collider != null)
            {
                if (Application.isPlaying)
                    Destroy(collider);
                else
                    DestroyImmediate(collider);
            }

            MeshRenderer renderer = node.GetComponent<MeshRenderer>();
            if (renderer != null)
            {
                renderer.sharedMaterial = m_nodeMaterial;
                renderer.shadowCastingMode = ShadowCastingMode.Off;
                renderer.receiveShadows = false;
                renderer.motionVectorGenerationMode = MotionVectorGenerationMode.ForceNoMotion;
            }

            return node;
        }

        private void SetNodeObjectsVisible(bool visible)
        {
            for (int i = 0; i < m_nodeObjects.Count; i++)
            {
                if (m_nodeObjects[i] != null && m_nodeObjects[i].activeSelf != visible)
                    m_nodeObjects[i].SetActive(visible);
            }
        }
    }
}
