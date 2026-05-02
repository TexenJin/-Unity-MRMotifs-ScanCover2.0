using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

[DisallowMultipleComponent]
public class ScanCoverDepthSurfaceDebugRenderer_P1 : MonoBehaviour
{
    private struct ShellCellKey
    {
        public int x;
        public int y;

        public ShellCellKey(int x, int y)
        {
            this.x = x;
            this.y = y;
        }
    }

    private sealed class ShellCellData
    {
        public Vector3 localSum;
        public float confidenceSum;
        public float stabilitySum;
        public int hitCount;
        public bool filled;
    }

    [Header("Refs")]
    public ScanCoverDepthSurfaceField_P1 surfaceField;

    [Header("Render")]
    public bool renderSamples = true;
    public bool renderPatches = false;
    public bool renderShells = false;
    public Material sampleMaterial;
    public Material patchMaterial;
    public Material shellMaterial;
    public Mesh sampleMesh;
    public Mesh patchMesh;
    public Color sampleColor = new Color(0.24f, 0.92f, 0.98f, 0.45f);
    public Color patchColor = new Color(1.0f, 0.72f, 0.22f, 0.18f);
    public Color shellColor = new Color(0.30f, 0.86f, 1.0f, 0.18f);
    [Min(0.002f)] public float sampleSizeMeters = 0.025f;
    [Min(0.05f)] public float patchMarkerSizeMeters = 0.12f;
    [Min(1)] public int minPatchSamples = 8;
    [Range(0f, 1f)] public float minPatchCoverageRatio = 0.12f;
    [Min(0f)] public float maxPatchMeanHeightDeviationMeters = 0.08f;
    [Min(0.01f)] public float shellCellSizeMeters = 0.03f;
    [Min(1)] public int minShellCellHits = 2;
    [Range(0f, 1f)] public float shellHoleFillNeighborRatio = 0.6f;
    [Min(0f)] public float shellNormalBiasMeters = 0.0015f;
    [Min(0f)] public float shellMaxEdgeHeightDeltaMeters = 0.04f;
    [Min(1)] public int maxRenderedSamples = 6000;
    [Min(1)] public int maxRenderedPatches = 256;
    public bool renderShadows = false;

    [Header("Debug")]
    public bool debugLog = false;

    private readonly List<ScanCoverDepthSurfaceField_P1.SurfaceSampleInfo> _sampleScratch = new List<ScanCoverDepthSurfaceField_P1.SurfaceSampleInfo>(8192);
    private readonly List<ScanCoverDepthSurfaceField_P1.SurfacePatchInfo> _patchScratch = new List<ScanCoverDepthSurfaceField_P1.SurfacePatchInfo>(512);
    private readonly Matrix4x4[] _batchMatrices = new Matrix4x4[1023];
    private readonly Dictionary<ScanCoverDepthSurfaceField_P1.SurfacePatchKey, ScanCoverDepthSurfaceField_P1.SurfacePatchInfo> _selectedPatches = new Dictionary<ScanCoverDepthSurfaceField_P1.SurfacePatchKey, ScanCoverDepthSurfaceField_P1.SurfacePatchInfo>(256);
    private readonly Dictionary<ShellCellKey, ShellCellData> _shellCells = new Dictionary<ShellCellKey, ShellCellData>(512);
    private readonly Dictionary<ShellCellKey, int> _shellIndices = new Dictionary<ShellCellKey, int>(512);
    private readonly List<Vector3> _shellVertices = new List<Vector3>(8192);
    private readonly List<Vector3> _shellNormals = new List<Vector3>(8192);
    private readonly List<Vector2> _shellUvs = new List<Vector2>(8192);
    private readonly List<int> _shellTriangles = new List<int>(16384);
    private Mesh _shellMesh;

    private void Awake()
    {
        ResolveRefs();
        EnsureResources();
    }

    private void OnEnable()
    {
        ResolveRefs();
        EnsureResources();
    }

    private void LateUpdate()
    {
        ResolveRefs();
        EnsureResources();

        if (surfaceField == null)
            return;

        if (renderSamples)
            DrawSamples();

        if (renderPatches)
            DrawPatches();

        if (renderShells)
            DrawShells();
    }

    private void ResolveRefs()
    {
        if (surfaceField == null)
            surfaceField = GetComponent<ScanCoverDepthSurfaceField_P1>();
    }

    private void EnsureResources()
    {
        if (sampleMesh == null)
            sampleMesh = CreateUnitQuadMesh("ScanCover_DepthSurfaceSampleQuad");
        if (patchMesh == null)
            patchMesh = CreateUnitQuadMesh("ScanCover_DepthSurfacePatchQuad");
        if (_shellMesh == null)
        {
            _shellMesh = new Mesh();
            _shellMesh.name = "ScanCover_DepthSurfaceShellMesh";
            _shellMesh.MarkDynamic();
        }

        if (sampleMaterial == null)
            sampleMaterial = CreateTransparentUnlitMaterial("ScanCover_DepthSurfaceSampleMat", sampleColor);
        if (patchMaterial == null)
            patchMaterial = CreateTransparentUnlitMaterial("ScanCover_DepthSurfacePatchMat", patchColor);
        if (shellMaterial == null)
            shellMaterial = CreateTransparentUnlitMaterial("ScanCover_DepthSurfaceShellMat", shellColor);
    }

    private void DrawSamples()
    {
        if (sampleMaterial == null || sampleMesh == null || !sampleMaterial.enableInstancing)
            return;

        _sampleScratch.Clear();
        surfaceField.GetSamplesSnapshot(_sampleScratch);
        if (_sampleScratch.Count <= 0)
            return;

        int limit = Mathf.Min(maxRenderedSamples, _sampleScratch.Count);
        ShadowCastingMode castMode = renderShadows ? ShadowCastingMode.On : ShadowCastingMode.Off;
        int batchFill = 0;

        for (int i = 0; i < limit; i++)
        {
            var sample = _sampleScratch[i];
            Vector3 n = sample.worldNormal.sqrMagnitude > 1e-6f ? sample.worldNormal.normalized : Vector3.up;
            Quaternion rot = Quaternion.LookRotation(n);
            float scale = sampleSizeMeters * Mathf.Lerp(0.85f, 1.15f, Mathf.Clamp01(sample.confidence));
            _batchMatrices[batchFill++] = Matrix4x4.TRS(sample.worldPos, rot, Vector3.one * scale);

            if (batchFill < 1023)
                continue;

            Graphics.DrawMeshInstanced(sampleMesh, 0, sampleMaterial, _batchMatrices, batchFill, null, castMode, false, gameObject.layer, null, LightProbeUsage.Off, null);
            batchFill = 0;
        }

        if (batchFill > 0)
            Graphics.DrawMeshInstanced(sampleMesh, 0, sampleMaterial, _batchMatrices, batchFill, null, castMode, false, gameObject.layer, null, LightProbeUsage.Off, null);
    }

    private void DrawPatches()
    {
        if (patchMaterial == null || patchMesh == null || !patchMaterial.enableInstancing)
            return;

        _patchScratch.Clear();
        surfaceField.GetPatchSnapshot(_patchScratch);
        if (_patchScratch.Count <= 0)
            return;

        _patchScratch.Sort((a, b) =>
        {
            int scoreCmp = PatchScore(b).CompareTo(PatchScore(a));
            if (scoreCmp != 0)
                return scoreCmp;

            int sampleCmp = b.sampleCount.CompareTo(a.sampleCount);
            if (sampleCmp != 0)
                return sampleCmp;
            return b.meanStability.CompareTo(a.meanStability);
        });

        ShadowCastingMode castMode = renderShadows ? ShadowCastingMode.On : ShadowCastingMode.Off;
        int batchFill = 0;
        int rendered = 0;

        for (int i = 0; i < _patchScratch.Count && rendered < maxRenderedPatches; i++)
        {
            var patch = _patchScratch[i];
            if (!IsPatchRenderable(patch))
                continue;

            Vector3 n = patch.meanNormalWS.sqrMagnitude > 1e-6f ? patch.meanNormalWS.normalized : Vector3.up;
            Vector3 t = patch.tangentWS.sqrMagnitude > 1e-6f ? patch.tangentWS.normalized : Vector3.right;
            Vector3 b = patch.bitangentWS.sqrMagnitude > 1e-6f ? patch.bitangentWS.normalized : Vector3.forward;
            float width = Mathf.Max(patchMarkerSizeMeters, patch.footprintMeters.x * Mathf.Lerp(0.8f, 1.0f, Mathf.Clamp01(patch.coverageRatio)));
            float height = Mathf.Max(patchMarkerSizeMeters, patch.footprintMeters.y * Mathf.Lerp(0.8f, 1.0f, Mathf.Clamp01(patch.coverageRatio)));
            float bias = Mathf.Lerp(0.001f, 0.003f, Mathf.Clamp01(patch.meanStability));
            _batchMatrices[batchFill++] = BuildQuadMatrix(patch.centerWS + n * bias, t, b, n, width, height);
            rendered++;

            if (batchFill < 1023)
                continue;

            Graphics.DrawMeshInstanced(patchMesh, 0, patchMaterial, _batchMatrices, batchFill, null, castMode, false, gameObject.layer, null, LightProbeUsage.Off, null);
            batchFill = 0;
        }

        if (batchFill > 0)
            Graphics.DrawMeshInstanced(patchMesh, 0, patchMaterial, _batchMatrices, batchFill, null, castMode, false, gameObject.layer, null, LightProbeUsage.Off, null);
    }

    private void DrawShells()
    {
        if (shellMaterial == null || _shellMesh == null)
            return;

        _patchScratch.Clear();
        surfaceField.GetPatchSnapshot(_patchScratch);
        if (_patchScratch.Count <= 0)
            return;

        _patchScratch.Sort((a, b) => PatchScore(b).CompareTo(PatchScore(a)));
        _selectedPatches.Clear();
        for (int i = 0; i < _patchScratch.Count && _selectedPatches.Count < maxRenderedPatches; i++)
        {
            var patch = _patchScratch[i];
            if (!IsPatchRenderable(patch))
                continue;
            _selectedPatches[patch.key] = patch;
        }

        if (_selectedPatches.Count <= 0)
            return;

        _sampleScratch.Clear();
        surfaceField.GetSamplesSnapshot(_sampleScratch);
        if (_sampleScratch.Count <= 0)
            return;

        BuildShellMesh();
        if (_shellMesh.vertexCount <= 0)
            return;

        ShadowCastingMode castMode = renderShadows ? ShadowCastingMode.On : ShadowCastingMode.Off;
        Graphics.DrawMesh(_shellMesh, Matrix4x4.identity, shellMaterial, gameObject.layer, null, 0, null, castMode, false, null, LightProbeUsage.Off, null);
    }

    private static Mesh CreateUnitQuadMesh(string name)
    {
        Mesh mesh = new Mesh();
        mesh.name = name;
        mesh.vertices = new[]
        {
            new Vector3(-0.5f, -0.5f, 0f),
            new Vector3( 0.5f, -0.5f, 0f),
            new Vector3( 0.5f,  0.5f, 0f),
            new Vector3(-0.5f,  0.5f, 0f)
        };
        mesh.normals = new[] { Vector3.forward, Vector3.forward, Vector3.forward, Vector3.forward };
        mesh.uv = new[]
        {
            new Vector2(0f, 0f),
            new Vector2(1f, 0f),
            new Vector2(1f, 1f),
            new Vector2(0f, 1f)
        };
        mesh.triangles = new[] { 0, 1, 2, 0, 2, 3 };
        mesh.RecalculateBounds();
        return mesh;
    }

    private static Material CreateTransparentUnlitMaterial(string name, Color color)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (!shader)
            shader = Shader.Find("Unlit/Color");
        if (!shader)
            return null;

        Material material = new Material(shader);
        material.name = name;
        ConfigureTransparentMaterial(material, color);
        return material;
    }

    private static void ConfigureTransparentMaterial(Material material, Color color)
    {
        if (material == null)
            return;

        material.enableInstancing = true;
        if (material.HasProperty("_BaseColor"))
            material.SetColor("_BaseColor", color);
        if (material.HasProperty("_Color"))
            material.SetColor("_Color", color);
        if (material.HasProperty("_Surface"))
            material.SetFloat("_Surface", 1f);
        if (material.HasProperty("_Blend"))
            material.SetFloat("_Blend", 0f);
        if (material.HasProperty("_ZWrite"))
            material.SetFloat("_ZWrite", 0f);
        if (material.HasProperty("_SrcBlend"))
            material.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
        if (material.HasProperty("_DstBlend"))
            material.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
        material.renderQueue = (int)RenderQueue.Transparent;
    }

    private static Matrix4x4 BuildQuadMatrix(Vector3 center, Vector3 tangent, Vector3 bitangent, Vector3 normal, float width, float height)
    {
        Matrix4x4 m = Matrix4x4.identity;
        m.SetColumn(0, new Vector4(tangent.x * width, tangent.y * width, tangent.z * width, 0f));
        m.SetColumn(1, new Vector4(bitangent.x * height, bitangent.y * height, bitangent.z * height, 0f));
        m.SetColumn(2, new Vector4(normal.x, normal.y, normal.z, 0f));
        m.SetColumn(3, new Vector4(center.x, center.y, center.z, 1f));
        return m;
    }

    private static float PatchScore(ScanCoverDepthSurfaceField_P1.SurfacePatchInfo patch)
    {
        float density = patch.projectedCellSpan > 0 ? (float)patch.sampleCount / patch.projectedCellSpan : 0f;
        float flatness = 1f / (1f + Mathf.Max(0f, patch.meanHeightDeviationMeters) * 20f);
        return patch.coverageRatio * 2.0f + density * 0.5f + patch.meanStability * 0.8f + flatness * 0.7f;
    }

    private bool IsPatchRenderable(ScanCoverDepthSurfaceField_P1.SurfacePatchInfo patch)
    {
        if (patch.sampleCount < minPatchSamples)
            return false;
        if (patch.coverageRatio < minPatchCoverageRatio)
            return false;
        if (maxPatchMeanHeightDeviationMeters > 0f && patch.meanHeightDeviationMeters > maxPatchMeanHeightDeviationMeters)
            return false;
        return true;
    }

    private void BuildShellMesh()
    {
        _shellVertices.Clear();
        _shellNormals.Clear();
        _shellUvs.Clear();
        _shellTriangles.Clear();

        foreach (var pair in _selectedPatches)
            BuildShellForPatch(pair.Value);

        _shellMesh.Clear();
        if (_shellVertices.Count <= 0)
            return;

        _shellMesh.SetVertices(_shellVertices);
        _shellMesh.SetNormals(_shellNormals);
        _shellMesh.SetUVs(0, _shellUvs);
        _shellMesh.SetTriangles(_shellTriangles, 0, true);
        _shellMesh.RecalculateBounds();
    }

    private void BuildShellForPatch(ScanCoverDepthSurfaceField_P1.SurfacePatchInfo patch)
    {
        _shellCells.Clear();
        _shellIndices.Clear();

        float cellSize = Mathf.Max(0.01f, shellCellSizeMeters);
        Vector3 n = patch.meanNormalWS.sqrMagnitude > 1e-6f ? patch.meanNormalWS.normalized : Vector3.up;
        Vector3 t = patch.tangentWS.sqrMagnitude > 1e-6f ? patch.tangentWS.normalized : Vector3.right;
        Vector3 b = patch.bitangentWS.sqrMagnitude > 1e-6f ? patch.bitangentWS.normalized : Vector3.forward;

        for (int i = 0; i < _sampleScratch.Count; i++)
        {
            var sample = _sampleScratch[i];
            if (!sample.patchKey.Equals(patch.key))
                continue;

            Vector3 delta = sample.worldPos - patch.centerWS;
            float u = Vector3.Dot(delta, t);
            float v = Vector3.Dot(delta, b);
            float h = Vector3.Dot(delta, n);
            var key = new ShellCellKey(
                Mathf.RoundToInt(u / cellSize),
                Mathf.RoundToInt(v / cellSize));

            if (!_shellCells.TryGetValue(key, out var cell))
            {
                cell = new ShellCellData();
                _shellCells.Add(key, cell);
            }

            cell.localSum += new Vector3(u, v, h);
            cell.confidenceSum += sample.confidence;
            cell.stabilitySum += sample.stability;
            cell.hitCount++;
        }

        if (_shellCells.Count <= 0)
            return;

        FillPatchShellHoles();

        foreach (var pair in _shellCells)
        {
            var cell = pair.Value;
            if (cell.hitCount < minShellCellHits && !cell.filled)
                continue;

            Vector3 local = cell.localSum / Mathf.Max(1, cell.hitCount);
            int idx = _shellVertices.Count;
            _shellIndices[pair.Key] = idx;
            _shellVertices.Add(patch.centerWS + t * local.x + b * local.y + n * (local.z + shellNormalBiasMeters));
            _shellNormals.Add(n);
            _shellUvs.Add(new Vector2(local.x, local.y));
        }

        foreach (var pair in _shellIndices)
        {
            var k = pair.Key;
            TryAddQuadTriangles(k, new ShellCellKey(k.x + 1, k.y), new ShellCellKey(k.x, k.y + 1), new ShellCellKey(k.x + 1, k.y + 1), n);
        }
    }

    private void FillPatchShellHoles()
    {
        List<ShellCellKey> fillKeys = null;
        List<Vector3> fillValues = null;

        foreach (var pair in _shellCells)
        {
            var center = pair.Key;
            for (int oy = -1; oy <= 1; oy++)
            {
                for (int ox = -1; ox <= 1; ox++)
                {
                    if (ox == 0 && oy == 0)
                        continue;

                    var candidate = new ShellCellKey(center.x + ox, center.y + oy);
                    if (_shellCells.ContainsKey(candidate))
                        continue;

                    int neighborCount = 0;
                    Vector3 sum = Vector3.zero;
                    for (int ny = -1; ny <= 1; ny++)
                    {
                        for (int nx = -1; nx <= 1; nx++)
                        {
                            var nk = new ShellCellKey(candidate.x + nx, candidate.y + ny);
                            if (_shellCells.TryGetValue(nk, out var neighbor))
                            {
                                sum += neighbor.localSum / Mathf.Max(1, neighbor.hitCount);
                                neighborCount++;
                            }
                        }
                    }

                    if (neighborCount < 4)
                        continue;
                    if (neighborCount / 9f < shellHoleFillNeighborRatio)
                        continue;

                    fillKeys ??= new List<ShellCellKey>(64);
                    fillValues ??= new List<Vector3>(64);
                    fillKeys.Add(candidate);
                    fillValues.Add(sum / neighborCount);
                }
            }
        }

        if (fillKeys == null)
            return;

        for (int i = 0; i < fillKeys.Count; i++)
        {
            if (_shellCells.ContainsKey(fillKeys[i]))
                continue;

            _shellCells.Add(fillKeys[i], new ShellCellData
            {
                localSum = fillValues[i],
                confidenceSum = 0.5f,
                stabilitySum = 0.5f,
                hitCount = 1,
                filled = true,
            });
        }
    }

    private void TryAddQuadTriangles(ShellCellKey k00, ShellCellKey k10, ShellCellKey k01, ShellCellKey k11, Vector3 normal)
    {
        if (!_shellIndices.TryGetValue(k00, out int i00))
            return;
        if (!_shellIndices.TryGetValue(k10, out int i10))
            return;
        if (!_shellIndices.TryGetValue(k01, out int i01))
            return;
        if (!_shellIndices.TryGetValue(k11, out int i11))
            return;

        if (IsTriangleCompatible(i00, i10, i11, normal))
        {
            _shellTriangles.Add(i00);
            _shellTriangles.Add(i10);
            _shellTriangles.Add(i11);
        }

        if (IsTriangleCompatible(i00, i11, i01, normal))
        {
            _shellTriangles.Add(i00);
            _shellTriangles.Add(i11);
            _shellTriangles.Add(i01);
        }
    }

    private bool IsTriangleCompatible(int ia, int ib, int ic, Vector3 normal)
    {
        Vector3 a = _shellVertices[ia];
        Vector3 b = _shellVertices[ib];
        Vector3 c = _shellVertices[ic];

        float ab = Mathf.Abs(Vector3.Dot(b - a, normal));
        float bc = Mathf.Abs(Vector3.Dot(c - b, normal));
        float ca = Mathf.Abs(Vector3.Dot(a - c, normal));
        if (Mathf.Max(ab, Mathf.Max(bc, ca)) > shellMaxEdgeHeightDeltaMeters)
            return false;

        Vector3 cross = Vector3.Cross(b - a, c - a);
        if (cross.sqrMagnitude <= 1e-8f)
            return false;
        return Vector3.Dot(cross.normalized, normal) > 0.15f;
    }
}
