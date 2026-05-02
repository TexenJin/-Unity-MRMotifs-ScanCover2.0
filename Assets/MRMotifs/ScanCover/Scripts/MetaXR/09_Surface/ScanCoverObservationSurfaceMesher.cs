using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

[DefaultExecutionOrder(-41)]
[DisallowMultipleComponent]
public sealed class ScanCoverObservationSurfaceMesher : MonoBehaviour
{
    private struct Node
    {
        public ScanCoverDepthObservationGridProvider.Observation observation;
        public int vertexIndex;
    }

    private struct CachedNode
    {
        public ScanCoverDepthObservationGridProvider.Observation observation;
        public float lastSeenTime;
        public bool seenThisFrame;
    }

    [Header("Refs")]
    [SerializeField] private ScanCoverDepthObservationGridProvider provider;
    [SerializeField] private Transform surfaceRoot;

    [Header("Build")]
    [SerializeField] private bool updateEveryFrame = true;
    [SerializeField, Range(0f, 1f)] private float minConfidence = 0.2f;
    [SerializeField, Min(0f)] private float maxEdgeLengthMeters = 0.25f;
    [SerializeField, Range(-1f, 1f)] private float minNormalDot = 0.35f;
    [SerializeField] private bool useObservationNormals = true;

    [Header("Temporal")]
    [SerializeField] private bool enableTemporalStabilization = true;
    [SerializeField, Range(0f, 1f)] private float positionBlend = 0.35f;
    [SerializeField, Range(0f, 1f)] private float normalBlend = 0.35f;
    [SerializeField, Range(0f, 1f)] private float confidenceBlend = 0.4f;
    [SerializeField, Min(0f)] private float holdMissingSeconds = 0.2f;
    [SerializeField, Min(0f)] private float maxSnapDistanceMeters = 0.25f;

    [Header("Render")]
    [SerializeField] private bool rendererVisible = true;
    [SerializeField] private Color surfaceColor = new Color(0.18f, 0.95f, 0.98f, 0.28f);
    [SerializeField] private Color fresnelColor = new Color(0.95f, 1.0f, 1.0f, 0.9f);
    [SerializeField, Min(0.1f)] private float fresnelPower = 2.5f;
    [SerializeField, Range(0f, 3f)] private float fresnelStrength = 1.0f;
    [SerializeField, Min(0.1f)] private float gridScale = 4.5f;
    [SerializeField, Range(0.001f, 0.2f)] private float gridThickness = 0.035f;
    [SerializeField, Range(0f, 3f)] private float gridIntensity = 1.1f;
    [SerializeField] private bool doubleSided = true;
    [SerializeField] private bool receiveShadows;

    [Header("Debug")]
    [SerializeField] private bool debugLog;

    public int VertexCount => _mesh != null ? _mesh.vertexCount : 0;
    public int TriangleCount => _mesh != null ? _mesh.triangles.Length / 3 : 0;
    public string LastIssue { get; private set; }

    private readonly Dictionary<Vector2Int, Node> _nodes = new Dictionary<Vector2Int, Node>(4096);
    private readonly Dictionary<Vector2Int, CachedNode> _temporalNodes = new Dictionary<Vector2Int, CachedNode>(4096);
    private readonly List<Vector3> _vertices = new List<Vector3>(8192);
    private readonly List<Vector3> _normals = new List<Vector3>(8192);
    private readonly List<int> _triangles = new List<int>(16384);
    private readonly List<Vector2Int> _pixels = new List<Vector2Int>(4096);
    private readonly List<Vector2Int> _stalePixels = new List<Vector2Int>(1024);
    private Mesh _mesh;
    private MeshFilter _meshFilter;
    private MeshRenderer _meshRenderer;
    private Material _surfaceMaterial;

    private void Awake()
    {
        ResolveRefs();
        EnsureSurfaceObjects();
    }

    private void OnEnable()
    {
        ResolveRefs();
        EnsureSurfaceObjects();
    }

    private void Update()
    {
        if (!updateEveryFrame)
            return;

        RefreshNow();
    }

    private void OnDestroy()
    {
        if (_mesh != null)
        {
            Object.Destroy(_mesh);
            _mesh = null;
        }

        if (_surfaceMaterial != null)
        {
            Object.Destroy(_surfaceMaterial);
            _surfaceMaterial = null;
        }
    }

    [ContextMenu("Refresh Observation Surface")]
    public bool RefreshNow()
    {
        ResolveRefs();
        EnsureSurfaceObjects();

        if (provider == null)
            return SetIssue("ScanCoverDepthObservationGridProvider is missing.");

        if (_mesh == null || _meshFilter == null || _meshRenderer == null || surfaceRoot == null)
            return SetIssue("Surface mesh objects are not ready.");

        var observations = provider.CurrentObservations;
        if (observations == null || observations.Count <= 0)
        {
            ClearMesh();
            return SetIssue("Observation list is empty.");
        }

        int step = provider.Stride;
        _nodes.Clear();
        _pixels.Clear();
        _vertices.Clear();
        _normals.Clear();
        _triangles.Clear();

        BuildTemporalNodes(observations);
        RebuildSurfaceNodes();

        for (int i = 0; i < _pixels.Count; i++)
        {
            Vector2Int p00 = _pixels[i];
            Vector2Int p10 = new Vector2Int(p00.x + step, p00.y);
            Vector2Int p01 = new Vector2Int(p00.x, p00.y + step);
            Vector2Int p11 = new Vector2Int(p00.x + step, p00.y + step);

            if (!_nodes.TryGetValue(p00, out Node n00) ||
                !_nodes.TryGetValue(p10, out Node n10) ||
                !_nodes.TryGetValue(p01, out Node n01) ||
                !_nodes.TryGetValue(p11, out Node n11))
            {
                continue;
            }

            if (!IsQuadCoherent(n00.observation, n10.observation, n01.observation, n11.observation))
                continue;

            AddTriangle(n00, n01, n10);
            AddTriangle(n10, n01, n11);

            if (doubleSided)
            {
                AddTriangle(n10, n01, n00);
                AddTriangle(n11, n01, n10);
            }
        }

        _mesh.Clear();
        if (_vertices.Count > 0 && _triangles.Count > 0)
        {
            _mesh.SetVertices(_vertices);
            if (useObservationNormals && _normals.Count == _vertices.Count)
                _mesh.SetNormals(_normals);
            else
                _mesh.RecalculateNormals();
            _mesh.SetTriangles(_triangles, 0);
            _mesh.RecalculateBounds();
        }

        _meshFilter.sharedMesh = _mesh;
        _meshRenderer.enabled = rendererVisible && _triangles.Count > 0;
        LastIssue = null;

        if (debugLog)
        {
            Debug.Log(
                $"[ScanCoverObservationSurfaceMesher] observations={observations.Count}, " +
                $"vertices={_vertices.Count}, triangles={_triangles.Count / 3}, retained={_temporalNodes.Count}");
        }

        return _triangles.Count > 0;
    }

    public void ClearMesh()
    {
        if (_mesh != null)
            _mesh.Clear();
        if (_meshFilter != null)
            _meshFilter.sharedMesh = _mesh;
        if (_meshRenderer != null)
            _meshRenderer.enabled = false;
    }

    private void BuildTemporalNodes(IReadOnlyList<ScanCoverDepthObservationGridProvider.Observation> observations)
    {
        if (!enableTemporalStabilization)
        {
            _temporalNodes.Clear();
            for (int i = 0; i < observations.Count; i++)
            {
                var observation = observations[i];
                if (!observation.valid || observation.confidence < minConfidence)
                    continue;

                _temporalNodes[observation.sourcePixel] = new CachedNode
                {
                    observation = observation,
                    lastSeenTime = Time.time,
                    seenThisFrame = true,
                };
            }
            return;
        }

        float now = Time.time;
        _stalePixels.Clear();
        _pixels.Clear();
        foreach (var pair in _temporalNodes)
            _pixels.Add(pair.Key);
        for (int i = 0; i < _pixels.Count; i++)
        {
            Vector2Int key = _pixels[i];
            CachedNode node = _temporalNodes[key];
            node.seenThisFrame = false;
            _temporalNodes[key] = node;
        }
        _pixels.Clear();

        for (int i = 0; i < observations.Count; i++)
        {
            var observation = observations[i];
            if (!observation.valid)
                continue;

            Vector2Int pixel = observation.sourcePixel;
            if (_temporalNodes.TryGetValue(pixel, out CachedNode cached))
            {
                cached.observation = BlendObservation(cached.observation, observation);
                cached.lastSeenTime = now;
                cached.seenThisFrame = true;
                _temporalNodes[pixel] = cached;
            }
            else
            {
                _temporalNodes[pixel] = new CachedNode
                {
                    observation = observation,
                    lastSeenTime = now,
                    seenThisFrame = true,
                };
            }
        }

        foreach (var pair in _temporalNodes)
        {
            if (now - pair.Value.lastSeenTime > holdMissingSeconds)
                _stalePixels.Add(pair.Key);
        }

        for (int i = 0; i < _stalePixels.Count; i++)
            _temporalNodes.Remove(_stalePixels[i]);
    }

    private void RebuildSurfaceNodes()
    {
        foreach (var pair in _temporalNodes)
        {
            var observation = pair.Value.observation;
            if (!observation.valid || observation.confidence < minConfidence)
                continue;

            int vertexIndex = _vertices.Count;
            _vertices.Add(surfaceRoot.InverseTransformPoint(observation.worldPos));

            Vector3 normal = observation.worldNormal.sqrMagnitude > 1e-6f
                ? observation.worldNormal.normalized
                : Vector3.up;
            _normals.Add(surfaceRoot.InverseTransformDirection(normal).normalized);

            _nodes[pair.Key] = new Node
            {
                observation = observation,
                vertexIndex = vertexIndex,
            };
            _pixels.Add(pair.Key);
        }
    }

    private ScanCoverDepthObservationGridProvider.Observation BlendObservation(
        ScanCoverDepthObservationGridProvider.Observation previous,
        ScanCoverDepthObservationGridProvider.Observation current)
    {
        ScanCoverDepthObservationGridProvider.Observation blended = current;

        float distance = Vector3.Distance(previous.worldPos, current.worldPos);
        if (distance > maxSnapDistanceMeters)
        {
            blended.worldPos = current.worldPos;
            blended.worldNormal = current.worldNormal;
            blended.confidence = current.confidence;
            blended.linearDepth = current.linearDepth;
            return blended;
        }

        blended.worldPos = Vector3.Lerp(previous.worldPos, current.worldPos, positionBlend);

        Vector3 prevNormal = previous.worldNormal.sqrMagnitude > 1e-6f ? previous.worldNormal.normalized : Vector3.up;
        Vector3 currNormal = current.worldNormal.sqrMagnitude > 1e-6f ? current.worldNormal.normalized : prevNormal;
        Vector3 lerpedNormal = Vector3.Lerp(prevNormal, currNormal, normalBlend);
        blended.worldNormal = lerpedNormal.sqrMagnitude > 1e-6f ? lerpedNormal.normalized : currNormal;

        blended.confidence = Mathf.Lerp(previous.confidence, current.confidence, confidenceBlend);
        blended.linearDepth = Mathf.Lerp(previous.linearDepth, current.linearDepth, positionBlend);
        blended.valid = current.valid || previous.valid;
        return blended;
    }

    private void ResolveRefs()
    {
        if (provider == null)
            provider = FindAnyObjectByType<ScanCoverDepthObservationGridProvider>();
        if (surfaceRoot == null)
            surfaceRoot = transform;
    }

    private void EnsureSurfaceObjects()
    {
        if (surfaceRoot == null)
            surfaceRoot = transform;

        if (_meshFilter == null)
            _meshFilter = surfaceRoot.GetComponent<MeshFilter>();
        if (_meshFilter == null)
            _meshFilter = surfaceRoot.gameObject.AddComponent<MeshFilter>();

        if (_meshRenderer == null)
            _meshRenderer = surfaceRoot.GetComponent<MeshRenderer>();
        if (_meshRenderer == null)
            _meshRenderer = surfaceRoot.gameObject.AddComponent<MeshRenderer>();

        if (_mesh == null)
        {
            _mesh = new Mesh
            {
                name = "ScanCover_ObservationSurface"
            };
            _mesh.MarkDynamic();
        }

        if (_surfaceMaterial == null)
        {
            Shader shader = Shader.Find("MRMotifs/ScanCover/ObservationSurface");
            if (shader == null)
                shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
                shader = Shader.Find("Unlit/Color");
            _surfaceMaterial = new Material(shader)
            {
                name = "ScanCover_ObservationSurface_Mat"
            };
        }

        ApplyMaterialSettings();
        _meshFilter.sharedMesh = _mesh;
        if (_meshRenderer.sharedMaterial != _surfaceMaterial)
            _meshRenderer.sharedMaterial = _surfaceMaterial;
        _meshRenderer.receiveShadows = receiveShadows;
        _meshRenderer.shadowCastingMode = receiveShadows ? ShadowCastingMode.On : ShadowCastingMode.Off;
    }

    private void ApplyMaterialSettings()
    {
        if (_surfaceMaterial == null)
            return;

        if (_surfaceMaterial.HasProperty("_BaseColor"))
            _surfaceMaterial.SetColor("_BaseColor", surfaceColor);
        if (_surfaceMaterial.HasProperty("_Color"))
            _surfaceMaterial.SetColor("_Color", surfaceColor);
        if (_surfaceMaterial.HasProperty("_FresnelColor"))
            _surfaceMaterial.SetColor("_FresnelColor", fresnelColor);
        if (_surfaceMaterial.HasProperty("_GridColor"))
            _surfaceMaterial.SetColor("_GridColor", fresnelColor);
        if (_surfaceMaterial.HasProperty("_BaseAlpha"))
            _surfaceMaterial.SetFloat("_BaseAlpha", surfaceColor.a);
        if (_surfaceMaterial.HasProperty("_FresnelPower"))
            _surfaceMaterial.SetFloat("_FresnelPower", fresnelPower);
        if (_surfaceMaterial.HasProperty("_FresnelStrength"))
            _surfaceMaterial.SetFloat("_FresnelStrength", fresnelStrength);
        if (_surfaceMaterial.HasProperty("_GridScale"))
            _surfaceMaterial.SetFloat("_GridScale", gridScale);
        if (_surfaceMaterial.HasProperty("_GridThickness"))
            _surfaceMaterial.SetFloat("_GridThickness", gridThickness);
        if (_surfaceMaterial.HasProperty("_GridIntensity"))
            _surfaceMaterial.SetFloat("_GridIntensity", gridIntensity);
        if (_surfaceMaterial.HasProperty("_Surface"))
            _surfaceMaterial.SetFloat("_Surface", 1f);
        if (_surfaceMaterial.HasProperty("_Blend"))
            _surfaceMaterial.SetFloat("_Blend", 0f);
        if (_surfaceMaterial.HasProperty("_ZWrite"))
            _surfaceMaterial.SetFloat("_ZWrite", 0f);
        if (_surfaceMaterial.HasProperty("_Cull"))
            _surfaceMaterial.SetFloat("_Cull", doubleSided ? (float)CullMode.Off : (float)CullMode.Back);
        if (_surfaceMaterial.HasProperty("_SrcBlend"))
            _surfaceMaterial.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
        if (_surfaceMaterial.HasProperty("_DstBlend"))
            _surfaceMaterial.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
        _surfaceMaterial.renderQueue = (int)RenderQueue.Transparent;
    }

    private bool IsQuadCoherent(
        ScanCoverDepthObservationGridProvider.Observation a,
        ScanCoverDepthObservationGridProvider.Observation b,
        ScanCoverDepthObservationGridProvider.Observation c,
        ScanCoverDepthObservationGridProvider.Observation d)
    {
        return
            AreNeighborsCoherent(a, b) &&
            AreNeighborsCoherent(a, c) &&
            AreNeighborsCoherent(b, d) &&
            AreNeighborsCoherent(c, d) &&
            AreNeighborsCoherent(a, d) &&
            AreNeighborsCoherent(b, c);
    }

    private bool AreNeighborsCoherent(
        ScanCoverDepthObservationGridProvider.Observation a,
        ScanCoverDepthObservationGridProvider.Observation b)
    {
        if (Vector3.Distance(a.worldPos, b.worldPos) > maxEdgeLengthMeters)
            return false;

        Vector3 na = a.worldNormal.sqrMagnitude > 1e-6f ? a.worldNormal.normalized : Vector3.up;
        Vector3 nb = b.worldNormal.sqrMagnitude > 1e-6f ? b.worldNormal.normalized : Vector3.up;
        return Vector3.Dot(na, nb) >= minNormalDot;
    }

    private void AddTriangle(Node a, Node b, Node c)
    {
        _triangles.Add(a.vertexIndex);
        _triangles.Add(b.vertexIndex);
        _triangles.Add(c.vertexIndex);
    }

    private bool SetIssue(string issue)
    {
        LastIssue = issue;
        if (debugLog)
            Debug.LogWarning($"[ScanCoverObservationSurfaceMesher] {issue}");
        return false;
    }
}
