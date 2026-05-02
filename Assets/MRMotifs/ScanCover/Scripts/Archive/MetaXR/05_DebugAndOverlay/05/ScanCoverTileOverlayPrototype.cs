using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

[DisallowMultipleComponent]
public class ScanCoverTileOverlayPrototype : MonoBehaviour
{
    [Header("Refs")]
    public ScanCoverSkeletonSessionController sessionController;
    public ScanCoverSkeletonMesher_B mesher;
    public Transform rayOrigin;

    [Header("Sampling")]
    [Tooltip("Master switch for tile overlay logic (sampling + rendering).")]
    public bool enableTileOverlay = true;
    [Tooltip("When turning off overlay, clear existing tiles immediately.")]
    public bool clearTilesWhenDisabled = true;
    public bool sampleWhileScanning = true;
    public bool sampleWhenFrozen = true;
    [Min(1)] public int raysPerStep = 24;
    [Min(0.01f)] public float sampleIntervalSeconds = 0.03f;
    [Range(0.05f, 0.95f)] public float sampleViewportRadius = 0.45f;
    [Min(0.1f)] public float maxRayDistance = 8.0f;
    public LayerMask hitMask = ~0;
    public bool restrictToMesherChunks = true;
    [Tooltip("Only keep hits whose surface normal faces the room side (towards this anchor). If null, use Ray Origin.")]
    public Transform interiorAnchor;
    [Range(-1f, 1f)] public float minInwardDot = 0.15f;
    [Tooltip("Only keep front-facing surfaces relative to camera rays.")]
    public bool cullBackFacingHits = true;
    [Range(-1f, 1f)] public float minViewFacingDot = 0.05f;

    [Header("Tile Instances")]
    [Min(0.01f)] public float tileSizeMeters = 0.035f;
    [Min(0f)] public float normalOffsetMeters = 0.003f;
    [Min(1)] public int maxTiles = 20000;
    [Min(0.002f)] public float dedupCellMeters = 0.02f;
    public bool randomRoll = true;
    public bool clearOnEnterScanning = true;

    [Header("Render")]
    public Material tileMaterial;
    public Color tileColor = new Color(0.24f, 0.92f, 0.98f, 0.38f);
    public Mesh tileMesh;
    public bool renderInstances = true;
    public bool renderShadows = false;

    [Header("Debug")]
    public bool debugLog = false;

    private readonly Dictionary<Vector3Int, int> _tileByCell = new Dictionary<Vector3Int, int>(32768);
    private readonly List<Matrix4x4> _matrices = new List<Matrix4x4>(32768);
    private readonly Matrix4x4[] _drawBatch = new Matrix4x4[1023];
    private readonly System.Random _rng = new System.Random(1337);
    private Transform _chunksRoot;
    private float _nextSampleTime;
    private int _sampleIndex;
    private ScanCoverSkeletonSessionController.SessionState _lastState = ScanCoverSkeletonSessionController.SessionState.Idle;
    private bool _lastOverlayEnabled = true;

    private void Awake()
    {
        ResolveRefs();
        EnsureResources();
        if (sessionController) _lastState = sessionController.State;
    }

    private void OnEnable()
    {
        ResolveRefs();
        EnsureResources();
        _nextSampleTime = Time.time;
        _lastOverlayEnabled = enableTileOverlay;
    }

    private void Update()
    {
        ResolveRefs();

        if (_lastOverlayEnabled != enableTileOverlay)
        {
            if (!enableTileOverlay && clearTilesWhenDisabled)
                ClearTiles();
            _lastOverlayEnabled = enableTileOverlay;
        }

        if (!enableTileOverlay)
            return;

        var state = sessionController ? sessionController.State : ScanCoverSkeletonSessionController.SessionState.Idle;
        if (clearOnEnterScanning && state != _lastState && state == ScanCoverSkeletonSessionController.SessionState.Scanning)
            ClearTiles();
        _lastState = state;

        if (!ShouldSample(state)) return;
        if (Time.time < _nextSampleTime) return;
        _nextSampleTime = Time.time + Mathf.Max(0.01f, sampleIntervalSeconds);

        SampleHits(Mathf.Max(1, raysPerStep));
    }

    private void LateUpdate()
    {
        if (!enableTileOverlay) return;
        if (!renderInstances || tileMaterial == null || tileMesh == null || _matrices.Count == 0) return;
        if (!tileMaterial.enableInstancing) return;

        ShadowCastingMode cast = renderShadows ? ShadowCastingMode.On : ShadowCastingMode.Off;
        for (int i = 0; i < _matrices.Count; i += 1023)
        {
            int count = Mathf.Min(1023, _matrices.Count - i);
            for (int k = 0; k < count; k++)
                _drawBatch[k] = _matrices[i + k];
            Graphics.DrawMeshInstanced(
                tileMesh,
                0,
                tileMaterial,
                _drawBatch,
                count,
                null,
                cast,
                false,
                gameObject.layer,
                null,
                LightProbeUsage.Off,
                null);
        }
    }

    public void ClearTiles()
    {
        _tileByCell.Clear();
        _matrices.Clear();
        _sampleIndex = 0;
        if (debugLog) Debug.Log("[ScanCoverTileOverlayPrototype] ClearTiles");
    }

    private void ResolveRefs()
    {
        if (!sessionController) sessionController = GetComponent<ScanCoverSkeletonSessionController>();
        if (!mesher) mesher = GetComponent<ScanCoverSkeletonMesher_B>();
        if (!rayOrigin && Camera.main) rayOrigin = Camera.main.transform;
        if (!interiorAnchor) interiorAnchor = rayOrigin;
        if (restrictToMesherChunks && mesher != null && _chunksRoot == null)
            _chunksRoot = mesher.chunksRoot;
    }

    private void EnsureResources()
    {
        if (tileMesh == null)
            tileMesh = CreateUnitQuadMesh();

        if (tileMaterial == null)
        {
            Shader sh = Shader.Find("Universal Render Pipeline/Unlit");
            if (!sh) sh = Shader.Find("Unlit/Color");
            if (sh) tileMaterial = new Material(sh);
        }

        if (tileMaterial != null)
        {
            ConfigureTransparentMaterial(tileMaterial);
            tileMaterial.enableInstancing = true;
            if (tileMaterial.HasProperty("_BaseColor"))
                tileMaterial.SetColor("_BaseColor", tileColor);
            else if (tileMaterial.HasProperty("_Color"))
                tileMaterial.SetColor("_Color", tileColor);
        }
    }

    private bool ShouldSample(ScanCoverSkeletonSessionController.SessionState state)
    {
        if (!enableTileOverlay) return false;
        if (rayOrigin == null || mesher == null) return false;
        if (restrictToMesherChunks && (mesher.chunksRoot == null || mesher.ChunkCount <= 0)) return false;
        if (state == ScanCoverSkeletonSessionController.SessionState.Scanning) return sampleWhileScanning;
        if (state == ScanCoverSkeletonSessionController.SessionState.Frozen) return sampleWhenFrozen;
        return false;
    }

    private void SampleHits(int count)
    {
        Vector3 origin = rayOrigin.position;
        Vector3 fwd = rayOrigin.forward;
        Vector3 right = rayOrigin.right;
        Vector3 up = rayOrigin.up;
        float radius = Mathf.Clamp01(sampleViewportRadius);

        for (int i = 0; i < count; i++)
        {
            Vector2 uv = NextHaltonDiskPoint(radius);
            Vector3 dir = (fwd + right * uv.x + up * uv.y).normalized;
            if (!Physics.Raycast(origin, dir, out RaycastHit hit, maxRayDistance, hitMask, QueryTriggerInteraction.Ignore))
                continue;
            if (restrictToMesherChunks && _chunksRoot != null && !hit.collider.transform.IsChildOf(_chunksRoot))
                continue;
            if (!AcceptHit(hit, origin, dir))
                continue;

            AddTile(hit.point, hit.normal);
            if (_matrices.Count >= maxTiles)
                return;
        }
    }

    private bool AcceptHit(RaycastHit hit, Vector3 rayStart, Vector3 rayDir)
    {
        Vector3 n = hit.normal.sqrMagnitude > 1e-8f ? hit.normal.normalized : Vector3.up;

        if (cullBackFacingHits)
        {
            float facing = Vector3.Dot(n, -rayDir);
            if (facing < minViewFacingDot) return false;
        }

        Vector3 anchorPos = interiorAnchor ? interiorAnchor.position : rayStart;
        Vector3 toAnchor = anchorPos - hit.point;
        if (toAnchor.sqrMagnitude > 1e-8f)
        {
            float inward = Vector3.Dot(n, toAnchor.normalized);
            if (inward < minInwardDot) return false;
        }

        return true;
    }

    private void AddTile(Vector3 point, Vector3 normal)
    {
        if (_matrices.Count >= maxTiles) return;
        Vector3Int cell = Quantize(point, Mathf.Max(0.002f, dedupCellMeters));
        if (_tileByCell.ContainsKey(cell)) return;

        Vector3 n = normal.sqrMagnitude > 1e-8f ? normal.normalized : Vector3.up;
        Vector3 pos = point + n * Mathf.Max(0f, normalOffsetMeters);
        Quaternion rot = Quaternion.LookRotation(n);
        if (randomRoll)
            rot = rot * Quaternion.AngleAxis((float)(_rng.NextDouble() * 360.0), Vector3.forward);

        Matrix4x4 m = Matrix4x4.TRS(pos, rot, Vector3.one * Mathf.Max(0.01f, tileSizeMeters));
        int idx = _matrices.Count;
        _matrices.Add(m);
        _tileByCell[cell] = idx;
    }

    private static Vector3Int Quantize(Vector3 p, float cell)
    {
        float inv = 1f / Mathf.Max(1e-6f, cell);
        return new Vector3Int(
            Mathf.RoundToInt(p.x * inv),
            Mathf.RoundToInt(p.y * inv),
            Mathf.RoundToInt(p.z * inv));
    }

    private Vector2 NextHaltonDiskPoint(float radius)
    {
        _sampleIndex++;
        float x = Halton(_sampleIndex, 2) * 2f - 1f;
        float y = Halton(_sampleIndex, 3) * 2f - 1f;
        Vector2 v = new Vector2(x, y);
        if (v.sqrMagnitude > 1f)
            v = v.normalized * Mathf.Sqrt(Halton(_sampleIndex, 5));
        return v * radius;
    }

    private static float Halton(int index, int b)
    {
        float f = 1f;
        float r = 0f;
        int i = index;
        while (i > 0)
        {
            f /= b;
            r += f * (i % b);
            i /= b;
        }
        return r;
    }

    private static Mesh CreateUnitQuadMesh()
    {
        Mesh m = new Mesh();
        m.name = "ScanCover_TileQuad";
        m.vertices = new[]
        {
            new Vector3(-0.5f, -0.5f, 0f),
            new Vector3( 0.5f, -0.5f, 0f),
            new Vector3( 0.5f,  0.5f, 0f),
            new Vector3(-0.5f,  0.5f, 0f)
        };
        m.normals = new[] { Vector3.forward, Vector3.forward, Vector3.forward, Vector3.forward };
        m.uv = new[]
        {
            new Vector2(0f, 0f),
            new Vector2(1f, 0f),
            new Vector2(1f, 1f),
            new Vector2(0f, 1f)
        };
        m.triangles = new[] { 0, 1, 2, 0, 2, 3 };
        m.RecalculateBounds();
        return m;
    }

    private static void ConfigureTransparentMaterial(Material mat)
    {
        if (mat == null) return;
        mat.enableInstancing = true;

        if (mat.HasProperty("_Surface"))
            mat.SetFloat("_Surface", 1f); // Transparent
        if (mat.HasProperty("_Blend"))
            mat.SetFloat("_Blend", 0f); // Alpha
        if (mat.HasProperty("_ZWrite"))
            mat.SetFloat("_ZWrite", 0f);
        if (mat.HasProperty("_SrcBlend"))
            mat.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
        if (mat.HasProperty("_DstBlend"))
            mat.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
        mat.renderQueue = (int)RenderQueue.Transparent;
    }
}
