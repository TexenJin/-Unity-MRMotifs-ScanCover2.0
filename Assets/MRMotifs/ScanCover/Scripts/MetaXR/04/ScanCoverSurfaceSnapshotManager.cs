using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering;

[DisallowMultipleComponent]
public sealed class ScanCoverSurfaceSnapshotManager : MonoBehaviour
{
    private struct ProjectedVertex
    {
        public Vector3 position;
        public Vector3 normal;
        public float weight;
    }

    private struct IndexedTriangle
    {
        public Vector3 centroid;
        public Vector3 normal;
    }

    [Header("Capture")]
    public Transform captureRoot;
    public ScanCoverDepthGridPointCloud depthGridPointCloud;
    public bool captureOnlyVisibleRenderers = true;
    public bool excludeMarkerLikeObjects = true;
    public bool includeInactiveChildren = false;

    [Header("Lifecycle")]
    public bool clearSnapshotsOnControllerClearAll = true;
    public bool debugLog = false;

    [Header("OBJ Export")]
    public bool autoExportLatestSnapshotAsObjOnCapture = false;
    public bool autoExportAllSnapshotsAsSingleObjOnCapture = false;
    public string exportDirectoryOverride = "";
    public bool exportNormals = true;

    [Header("Snapshot Display")]
    public Material snapshotMaterialOverride;
    public Color snapshotColor = new Color(0.92f, 0.94f, 0.98f, 1f);
    public bool snapshotDoubleSided = true;

    [Header("Regular Triangle Skin")]
    public bool generateRegularTriangleSkinOnCapture = false;
    [Min(0.01f)] public float regularSkinGridMeters = 0.08f;
    [Min(0.01f)] public float regularSkinMaxEdgeMeters = 0.18f;
    [Min(0f)] public float regularSkinOffsetMeters = 0.0025f;
    public Material regularSkinMaterialOverride;
    public Color regularSkinColor = new Color(0.30f, 0.82f, 1f, 1f);
    public bool hideCapturedSurfaceMeshWhenSkinGenerated = false;

    [Header("Incremental Reference Shell")]
    public bool maintainIncrementalReferenceShell = true;
    public bool integrateLatestSnapshotIntoReferenceShellOnCapture = true;
    public bool hideSnapshotsAfterIntegration = false;
    [Min(0.005f)] public float incrementalDuplicateDistance = 0.02f;
    [Min(0.010f)] public float incrementalNeighborRadius = 0.06f;
    [Min(0.005f)] public float incrementalInsideDistance = 0.03f;
    [Range(-1f, 1f)] public float incrementalMinNormalDot = 0.20f;
    public bool debugLogIncrementalStats = false;

    public int SnapshotCount => _snapshotCount;
    public string LastIssue { get; private set; }
    public string LastExportPath { get; private set; }
    public bool HasIncrementalReferenceShell => _incrementalShellRoot != null && _incrementalShellRoot.transform.childCount > 0;

    private GameObject _snapshotRoot;
    private GameObject _incrementalShellRoot;
    private int _snapshotCount;
    private Material _runtimeSnapshotMaterial;
    private Material _runtimeRegularSkinMaterial;

    public void EnsureInitialized()
    {
        if (captureRoot == null)
            captureRoot = transform;
        if (depthGridPointCloud == null)
            depthGridPointCloud = GetComponentInChildren<ScanCoverDepthGridPointCloud>(true);
        EnsureSnapshotRoot();
        EnsureIncrementalShellRoot();
    }

    public int CaptureVisibleSurfaces()
    {
        EnsureInitialized();
        if (captureRoot == null)
        {
            LastIssue = "Capture root is missing.";
            return 0;
        }

        List<MeshFilter> filterList = new List<MeshFilter>(128);
        HashSet<MeshFilter> uniqueFilters = new HashSet<MeshFilter>();
        AppendMeshFilters(captureRoot, uniqueFilters, filterList);

        Transform previewRoot = depthGridPointCloud != null ? depthGridPointCloud.SnapshotCaptureRoot : null;
        if (previewRoot != null && previewRoot != captureRoot)
            AppendMeshFilters(previewRoot, uniqueFilters, filterList);

        if (filterList.Count == 0)
        {
            LastIssue = "No mesh filters found under capture root.";
            return 0;
        }

        int snapshotIndex = ++_snapshotCount;
        GameObject snapshotGroup = new GameObject($"Snapshot_{snapshotIndex:000}");
        snapshotGroup.transform.SetParent(_snapshotRoot.transform, false);
        snapshotGroup.transform.localPosition = Vector3.zero;
        snapshotGroup.transform.localRotation = Quaternion.identity;
        snapshotGroup.transform.localScale = Vector3.one;

        int captured = 0;
        for (int i = 0; i < filterList.Count; i++)
        {
            MeshFilter filter = filterList[i];
            if (filter == null)
                continue;

            if (!ShouldCapture(filter, out MeshRenderer sourceRenderer))
                continue;

            Mesh sourceMesh = filter.sharedMesh;
            if (sourceMesh == null || sourceMesh.vertexCount <= 0 || !HasSnapshotGeometry(sourceMesh))
                continue;

            Mesh bakedMesh = BakeWorldSpaceMesh(filter.transform, sourceMesh);
            if (bakedMesh == null)
                continue;

            GameObject go = new GameObject(filter.gameObject.name, typeof(MeshFilter), typeof(MeshRenderer));
            go.transform.SetParent(snapshotGroup.transform, false);
            go.transform.localPosition = Vector3.zero;
            go.transform.localRotation = Quaternion.identity;
            go.transform.localScale = Vector3.one;

            MeshFilter targetFilter = go.GetComponent<MeshFilter>();
            MeshRenderer targetRenderer = go.GetComponent<MeshRenderer>();
            targetFilter.sharedMesh = bakedMesh;
            ApplyRendererCopy(sourceRenderer, targetRenderer, bakedMesh);
            bool generatedSkin = false;
            if (generateRegularTriangleSkinOnCapture && HasTriangleGeometry(bakedMesh))
                generatedSkin = TryCreateRegularTriangleSkin(snapshotGroup.transform, filter.gameObject.name, bakedMesh);
            if (generatedSkin && hideCapturedSurfaceMeshWhenSkinGenerated)
                targetRenderer.enabled = false;
            captured++;
        }

        if (captured <= 0)
        {
            DestroySnapshotGroup(snapshotGroup);
            _snapshotCount--;
            LastIssue = "No eligible visible surfaces were found to snapshot.";
            return 0;
        }

        LastIssue = null;
        if (debugLog)
            Debug.Log($"[ScanCoverSurfaceSnapshotManager] Captured snapshot #{snapshotIndex} with {captured} surfaces.");

        if (maintainIncrementalReferenceShell && integrateLatestSnapshotIntoReferenceShellOnCapture)
        {
            IntegrateSnapshotGroupIntoReferenceShell(snapshotGroup.transform, out int keptSurfaces, out int keptTriangles, out int rejectedTriangles);
            if (hideSnapshotsAfterIntegration)
                SetSnapshotGroupVisible(snapshotGroup.transform, false);

            if (debugLog || debugLogIncrementalStats)
            {
                Debug.Log(
                    $"[ScanCoverSurfaceSnapshotManager] Incremental shell integrate snapshot #{snapshotIndex} => keptSurfaces={keptSurfaces}, keptTriangles={keptTriangles}, rejectedTriangles={rejectedTriangles}, shellChildren={(_incrementalShellRoot != null ? _incrementalShellRoot.transform.childCount : 0)}");
            }
        }

        if (autoExportAllSnapshotsAsSingleObjOnCapture)
            ExportAllSnapshotsAsSingleObj();
        else if (autoExportLatestSnapshotAsObjOnCapture)
            ExportLatestSnapshotAsObj();

        return captured;
    }

    private void AppendMeshFilters(Transform root, HashSet<MeshFilter> uniqueFilters, List<MeshFilter> target)
    {
        if (root == null)
            return;

        MeshFilter[] filters = root.GetComponentsInChildren<MeshFilter>(includeInactiveChildren);
        if (filters == null || filters.Length == 0)
            return;

        for (int i = 0; i < filters.Length; i++)
        {
            MeshFilter filter = filters[i];
            if (filter == null || !uniqueFilters.Add(filter))
                continue;

            target.Add(filter);
        }
    }

    [ContextMenu("Export Latest Snapshot As OBJ")]
    public bool ExportLatestSnapshotAsObj()
    {
        EnsureInitialized();
        if (_snapshotRoot == null || _snapshotRoot.transform.childCount <= 0)
        {
            LastIssue = "No snapshot is available to export.";
            return false;
        }

        Transform latestSnapshot = _snapshotRoot.transform.GetChild(_snapshotRoot.transform.childCount - 1);
        return ExportSnapshotGroupAsObj(latestSnapshot, out _);
    }

    [ContextMenu("Export All Snapshots As Single OBJ")]
    public bool ExportAllSnapshotsAsSingleObj()
    {
        EnsureInitialized();
        if (_snapshotRoot == null || _snapshotRoot.transform.childCount <= 0)
        {
            LastIssue = "No snapshots are available to export.";
            return false;
        }

        string exportDirectory = ResolveExportDirectory();
        Directory.CreateDirectory(exportDirectory);
        string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
        string exportPath = Path.Combine(exportDirectory, $"Snapshots_All_{timestamp}.obj");

        StringBuilder builder = new StringBuilder(1024 * 128);
        builder.AppendLine("# ScanCover Combined Surface Snapshots OBJ");

        int vertexOffset = 1;
        int normalOffset = 1;
        int exportedMeshCount = 0;
        for (int i = 0; i < _snapshotRoot.transform.childCount; i++)
        {
            Transform snapshot = _snapshotRoot.transform.GetChild(i);
            if (snapshot == null)
                continue;

            AppendSnapshotGroupToObj(snapshot, builder, ref vertexOffset, ref normalOffset, ref exportedMeshCount);
        }

        if (exportedMeshCount <= 0)
        {
            LastIssue = "No exportable meshes were found across snapshots.";
            return false;
        }

        File.WriteAllText(exportPath, builder.ToString(), Encoding.UTF8);
        LastIssue = null;
        LastExportPath = exportPath;
        if (debugLog)
            Debug.Log($"[ScanCoverSurfaceSnapshotManager] Exported combined OBJ => {exportPath}");
        return true;
    }

    [ContextMenu("Export Incremental Reference Shell As OBJ")]
    public bool ExportIncrementalReferenceShellAsObj()
    {
        EnsureInitialized();
        if (_incrementalShellRoot == null || _incrementalShellRoot.transform.childCount <= 0)
        {
            LastIssue = "Incremental reference shell is empty.";
            return false;
        }

        string exportDirectory = ResolveExportDirectory();
        Directory.CreateDirectory(exportDirectory);
        string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
        string exportPath = Path.Combine(exportDirectory, $"ReferenceShell_{timestamp}.obj");

        StringBuilder builder = new StringBuilder(1024 * 128);
        builder.AppendLine("# ScanCover Incremental Reference Shell OBJ");

        int vertexOffset = 1;
        int normalOffset = 1;
        int exportedMeshCount = 0;
        MeshFilter[] filters = _incrementalShellRoot.GetComponentsInChildren<MeshFilter>(true);
        AppendMeshFiltersToObj(_incrementalShellRoot.name, filters, builder, ref vertexOffset, ref normalOffset, ref exportedMeshCount);

        if (exportedMeshCount <= 0)
        {
            LastIssue = "Incremental reference shell has no exportable meshes.";
            return false;
        }

        File.WriteAllText(exportPath, builder.ToString(), Encoding.UTF8);
        LastIssue = null;
        LastExportPath = exportPath;
        if (debugLog)
            Debug.Log($"[ScanCoverSurfaceSnapshotManager] Exported reference shell OBJ => {exportPath}");
        return true;
    }

    public void ClearAll()
    {
        LastIssue = null;
        LastExportPath = null;
        _snapshotCount = 0;

        if (_snapshotRoot != null)
        {
            for (int i = _snapshotRoot.transform.childCount - 1; i >= 0; i--)
                DestroySnapshotGroup(_snapshotRoot.transform.GetChild(i).gameObject);
        }

        if (_incrementalShellRoot != null)
        {
            for (int i = _incrementalShellRoot.transform.childCount - 1; i >= 0; i--)
                DestroySnapshotGroup(_incrementalShellRoot.transform.GetChild(i).gameObject);
        }
    }

    private void OnDestroy()
    {
        if (_snapshotRoot != null || _incrementalShellRoot != null)
        {
            ClearAll();
        }

        if (_snapshotRoot != null)
        {
            SafeDestroy(_snapshotRoot);
        }
        _snapshotRoot = null;

        if (_incrementalShellRoot != null)
        {
            SafeDestroy(_incrementalShellRoot);
        }
        _incrementalShellRoot = null;

        if (_runtimeSnapshotMaterial != null)
        {
            SafeDestroy(_runtimeSnapshotMaterial);
        }
        _runtimeSnapshotMaterial = null;

        if (_runtimeRegularSkinMaterial != null)
        {
            SafeDestroy(_runtimeRegularSkinMaterial);
        }
        _runtimeRegularSkinMaterial = null;
    }

    private bool ShouldCapture(MeshFilter filter, out MeshRenderer renderer)
    {
        renderer = filter.GetComponent<MeshRenderer>();
        if (renderer == null)
            return false;

        if (_snapshotRoot != null && filter.transform.IsChildOf(_snapshotRoot.transform))
            return false;

        if (excludeMarkerLikeObjects)
        {
            string lowerName = filter.gameObject.name.ToLowerInvariant();
            if (lowerName.Contains("marker") || lowerName.Contains("point"))
                return false;
        }

        if (captureOnlyVisibleRenderers)
        {
            if (!renderer.enabled || !renderer.gameObject.activeInHierarchy)
                return false;
        }

        return true;
    }

    private void EnsureSnapshotRoot()
    {
        if (_snapshotRoot != null)
            return;

        _snapshotRoot = new GameObject("[ScanCover] Surface Snapshots");
        _snapshotRoot.transform.SetParent(null, false);
        _snapshotRoot.transform.position = Vector3.zero;
        _snapshotRoot.transform.rotation = Quaternion.identity;
        _snapshotRoot.transform.localScale = Vector3.one;
    }

    private void EnsureIncrementalShellRoot()
    {
        if (_incrementalShellRoot != null || !maintainIncrementalReferenceShell)
            return;

        _incrementalShellRoot = new GameObject("[ScanCover] Incremental Reference Shell");
        _incrementalShellRoot.transform.SetParent(null, false);
        _incrementalShellRoot.transform.position = Vector3.zero;
        _incrementalShellRoot.transform.rotation = Quaternion.identity;
        _incrementalShellRoot.transform.localScale = Vector3.one;
    }

    private bool ExportSnapshotGroupAsObj(Transform snapshotGroup, out string exportPath)
    {
        exportPath = null;
        if (snapshotGroup == null)
        {
            LastIssue = "Snapshot group is missing.";
            return false;
        }

        MeshFilter[] filters = snapshotGroup.GetComponentsInChildren<MeshFilter>(true);
        if (filters == null || filters.Length == 0)
        {
            LastIssue = "Snapshot group has no mesh filters to export.";
            return false;
        }

        string exportDirectory = ResolveExportDirectory();
        Directory.CreateDirectory(exportDirectory);
        string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
        exportPath = Path.Combine(exportDirectory, $"{snapshotGroup.name}_{timestamp}.obj");

        StringBuilder builder = new StringBuilder(1024 * 64);
        builder.AppendLine("# ScanCover Surface Snapshot OBJ");
        builder.AppendLine($"# Source {snapshotGroup.name}");

        int vertexOffset = 1;
        int normalOffset = 1;
        int exportedMeshCount = 0;
        AppendMeshFiltersToObj(snapshotGroup.name, filters, builder, ref vertexOffset, ref normalOffset, ref exportedMeshCount);

        if (exportedMeshCount <= 0)
        {
            LastIssue = "Snapshot group did not contain any exportable meshes.";
            return false;
        }

        File.WriteAllText(exportPath, builder.ToString(), Encoding.UTF8);
        LastIssue = null;
        LastExportPath = exportPath;
        if (debugLog)
            Debug.Log($"[ScanCoverSurfaceSnapshotManager] Exported OBJ => {exportPath}");
        return true;
    }

    private void AppendSnapshotGroupToObj(Transform snapshotGroup, StringBuilder builder, ref int vertexOffset, ref int normalOffset, ref int exportedMeshCount)
    {
        MeshFilter[] filters = snapshotGroup.GetComponentsInChildren<MeshFilter>(true);
        if (filters == null || filters.Length == 0)
            return;

        builder.AppendLine($"g {MakeObjSafeName(snapshotGroup.name)}");
        AppendMeshFiltersToObj(snapshotGroup.name, filters, builder, ref vertexOffset, ref normalOffset, ref exportedMeshCount);
    }

    private void AppendMeshFiltersToObj(string groupName, MeshFilter[] filters, StringBuilder builder, ref int vertexOffset, ref int normalOffset, ref int exportedMeshCount)
    {
        for (int i = 0; i < filters.Length; i++)
        {
            MeshFilter filter = filters[i];
            if (filter == null || filter.sharedMesh == null)
                continue;

            Mesh mesh = filter.sharedMesh;
            Vector3[] vertices = mesh.vertices;
            int[] triangles = mesh.triangles;
            if (vertices == null || triangles == null || vertices.Length == 0 || triangles.Length < 3)
                continue;

            Vector3[] normals = mesh.normals;
            bool hasNormals = exportNormals && normals != null && normals.Length == vertices.Length;
            string objectName = MakeObjSafeName(filter.gameObject.name);
            builder.AppendLine($"o {objectName}");

            for (int v = 0; v < vertices.Length; v++)
            {
                Vector3 vertex = vertices[v];
                builder.Append("v ");
                builder.Append(vertex.x.ToString("R", CultureInfo.InvariantCulture)).Append(' ');
                builder.Append(vertex.y.ToString("R", CultureInfo.InvariantCulture)).Append(' ');
                builder.Append(vertex.z.ToString("R", CultureInfo.InvariantCulture)).AppendLine();
            }

            if (hasNormals)
            {
                for (int n = 0; n < normals.Length; n++)
                {
                    Vector3 normal = normals[n].normalized;
                    builder.Append("vn ");
                    builder.Append(normal.x.ToString("R", CultureInfo.InvariantCulture)).Append(' ');
                    builder.Append(normal.y.ToString("R", CultureInfo.InvariantCulture)).Append(' ');
                    builder.Append(normal.z.ToString("R", CultureInfo.InvariantCulture)).AppendLine();
                }
            }

            for (int t = 0; t <= triangles.Length - 3; t += 3)
            {
                int a = triangles[t] + vertexOffset;
                int b = triangles[t + 1] + vertexOffset;
                int c = triangles[t + 2] + vertexOffset;
                if (hasNormals)
                {
                    int na = triangles[t] + normalOffset;
                    int nb = triangles[t + 1] + normalOffset;
                    int nc = triangles[t + 2] + normalOffset;
                    builder.Append("f ")
                        .Append(a).Append("//").Append(na).Append(' ')
                        .Append(b).Append("//").Append(nb).Append(' ')
                        .Append(c).Append("//").Append(nc).AppendLine();
                }
                else
                {
                    builder.Append("f ")
                        .Append(a).Append(' ')
                        .Append(b).Append(' ')
                        .Append(c).AppendLine();
                }
            }

            vertexOffset += vertices.Length;
            if (hasNormals)
                normalOffset += normals.Length;
            exportedMeshCount++;
        }
    }

    private string ResolveExportDirectory()
    {
        if (!string.IsNullOrWhiteSpace(exportDirectoryOverride))
            return Path.GetFullPath(exportDirectoryOverride);

        string projectRoot = Directory.GetParent(Application.dataPath)?.FullName ?? Application.dataPath;
        return Path.Combine(projectRoot, "ScanCoverExports");
    }

    private static string MakeObjSafeName(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "SnapshotMesh";

        StringBuilder safe = new StringBuilder(value.Length);
        for (int i = 0; i < value.Length; i++)
        {
            char c = value[i];
            safe.Append(char.IsWhiteSpace(c) ? '_' : c);
        }
        return safe.ToString();
    }

    private Mesh BakeWorldSpaceMesh(Transform sourceTransform, Mesh sourceMesh)
    {
        if (sourceTransform == null || sourceMesh == null)
            return null;

        Mesh baked = Instantiate(sourceMesh);
        baked.name = $"{sourceMesh.name}_Snapshot";

        Vector3[] vertices = baked.vertices;
        Vector3[] normals = baked.normals;
        Matrix4x4 localToWorld = sourceTransform.localToWorldMatrix;
        Matrix4x4 normalMatrix = localToWorld.inverse.transpose;

        for (int i = 0; i < vertices.Length; i++)
            vertices[i] = localToWorld.MultiplyPoint3x4(vertices[i]);
        baked.vertices = vertices;

        if (normals != null && normals.Length == vertices.Length)
        {
            for (int i = 0; i < normals.Length; i++)
                normals[i] = normalMatrix.MultiplyVector(normals[i]).normalized;
            baked.normals = normals;
        }
        else
        {
            baked.RecalculateNormals();
        }

        baked.RecalculateBounds();
        return baked;
    }

    private static bool HasSnapshotGeometry(Mesh mesh)
    {
        if (mesh == null || mesh.vertexCount <= 0 || mesh.subMeshCount <= 0)
            return false;

        for (int subMesh = 0; subMesh < mesh.subMeshCount; subMesh++)
        {
            MeshTopology topology = mesh.GetTopology(subMesh);
            int indexCount = (int)mesh.GetIndexCount(subMesh);
            if (topology == MeshTopology.Triangles && indexCount >= 3)
                return true;
            if (topology == MeshTopology.Lines && indexCount >= 2)
                return true;
        }

        return false;
    }

    private static bool HasTriangleGeometry(Mesh mesh)
    {
        if (mesh == null || mesh.subMeshCount <= 0)
            return false;

        for (int subMesh = 0; subMesh < mesh.subMeshCount; subMesh++)
        {
            if (mesh.GetTopology(subMesh) == MeshTopology.Triangles && mesh.GetIndexCount(subMesh) >= 3)
                return true;
        }

        return false;
    }

    private bool TryCreateRegularTriangleSkin(Transform snapshotGroup, string sourceName, Mesh sourceMesh)
    {
        if (snapshotGroup == null || sourceMesh == null)
            return false;

        Mesh skinMesh = BuildRegularTriangleSkinMesh(sourceMesh);
        if (skinMesh == null || skinMesh.vertexCount <= 0 || skinMesh.GetIndexCount(0) <= 0)
        {
            if (skinMesh != null)
                SafeDestroy(skinMesh);
            return false;
        }

        GameObject go = new GameObject($"{sourceName}_RegularSkin", typeof(MeshFilter), typeof(MeshRenderer));
        go.transform.SetParent(snapshotGroup, false);
        go.transform.localPosition = Vector3.zero;
        go.transform.localRotation = Quaternion.identity;
        go.transform.localScale = Vector3.one;

        MeshFilter filter = go.GetComponent<MeshFilter>();
        MeshRenderer renderer = go.GetComponent<MeshRenderer>();
        filter.sharedMesh = skinMesh;
        renderer.sharedMaterial = ResolveRegularSkinMaterial();
        renderer.shadowCastingMode = ShadowCastingMode.Off;
        renderer.receiveShadows = false;
        return true;
    }

    private Mesh BuildRegularTriangleSkinMesh(Mesh sourceMesh)
    {
        Vector3[] vertices = sourceMesh.vertices;
        int[] triangles = sourceMesh.triangles;
        if (vertices == null || triangles == null || vertices.Length < 3 || triangles.Length < 3)
            return null;

        Vector3[] sourceNormals = sourceMesh.normals;
        bool hasSourceNormals = sourceNormals != null && sourceNormals.Length == vertices.Length;

        Vector3 averageNormal = Vector3.zero;
        for (int i = 0; i <= triangles.Length - 3; i += 3)
        {
            Vector3 a = vertices[triangles[i]];
            Vector3 b = vertices[triangles[i + 1]];
            Vector3 c = vertices[triangles[i + 2]];
            Vector3 n = Vector3.Cross(b - a, c - a);
            float area = n.magnitude;
            if (area <= 1e-6f)
                continue;
            averageNormal += n.normalized * area;
        }

        if (averageNormal.sqrMagnitude <= 1e-8f)
            return null;

        averageNormal.Normalize();
        Vector3 referenceAxis = Mathf.Abs(Vector3.Dot(averageNormal, Vector3.up)) > 0.85f ? Vector3.right : Vector3.up;
        Vector3 tangent = Vector3.Cross(referenceAxis, averageNormal).normalized;
        if (tangent.sqrMagnitude <= 1e-8f)
            tangent = Vector3.Cross(Vector3.forward, averageNormal).normalized;
        Vector3 bitangent = Vector3.Cross(averageNormal, tangent).normalized;

        Vector2[] projected = new Vector2[vertices.Length];
        Vector2 min = new Vector2(float.PositiveInfinity, float.PositiveInfinity);
        Vector2 max = new Vector2(float.NegativeInfinity, float.NegativeInfinity);
        for (int i = 0; i < vertices.Length; i++)
        {
            Vector2 uv = new Vector2(Vector3.Dot(vertices[i], tangent), Vector3.Dot(vertices[i], bitangent));
            projected[i] = uv;
            min = Vector2.Min(min, uv);
            max = Vector2.Max(max, uv);
        }

        float step = Mathf.Max(0.01f, regularSkinGridMeters);
        int cols = Mathf.Clamp(Mathf.CeilToInt((max.x - min.x) / step) + 1, 2, 320);
        int rows = Mathf.Clamp(Mathf.CeilToInt((max.y - min.y) / step) + 1, 2, 320);
        int sampleCount = cols * rows;
        ProjectedVertex[] samples = new ProjectedVertex[sampleCount];
        bool[] valid = new bool[sampleCount];

        for (int i = 0; i <= triangles.Length - 3; i += 3)
        {
            int ia = triangles[i];
            int ib = triangles[i + 1];
            int ic = triangles[i + 2];

            Vector2 pa = projected[ia];
            Vector2 pb = projected[ib];
            Vector2 pc = projected[ic];
            float det = Cross2D(pb - pa, pc - pa);
            if (Mathf.Abs(det) <= 1e-7f)
                continue;

            Vector3 va = vertices[ia];
            Vector3 vb = vertices[ib];
            Vector3 vc = vertices[ic];
            Vector3 na = hasSourceNormals ? sourceNormals[ia] : averageNormal;
            Vector3 nb = hasSourceNormals ? sourceNormals[ib] : averageNormal;
            Vector3 nc = hasSourceNormals ? sourceNormals[ic] : averageNormal;

            float minX = Mathf.Min(pa.x, Mathf.Min(pb.x, pc.x));
            float minY = Mathf.Min(pa.y, Mathf.Min(pb.y, pc.y));
            float maxX = Mathf.Max(pa.x, Mathf.Max(pb.x, pc.x));
            float maxY = Mathf.Max(pa.y, Mathf.Max(pb.y, pc.y));

            int xMin = Mathf.Clamp(Mathf.FloorToInt((minX - min.x) / step), 0, cols - 1);
            int xMax = Mathf.Clamp(Mathf.CeilToInt((maxX - min.x) / step), 0, cols - 1);
            int yMin = Mathf.Clamp(Mathf.FloorToInt((minY - min.y) / step), 0, rows - 1);
            int yMax = Mathf.Clamp(Mathf.CeilToInt((maxY - min.y) / step), 0, rows - 1);

            for (int y = yMin; y <= yMax; y++)
            {
                float sy = min.y + y * step;
                for (int x = xMin; x <= xMax; x++)
                {
                    float sx = min.x + x * step;
                    Vector2 sp = new Vector2(sx, sy);
                    if (!TryGetBarycentric(pa, pb, pc, sp, out float u, out float v, out float w))
                        continue;

                    int sampleIndex = y * cols + x;
                    Vector3 samplePosition = va * u + vb * v + vc * w;
                    Vector3 sampleNormal = (na * u + nb * v + nc * w).normalized;
                    ProjectedVertex sample = samples[sampleIndex];
                    sample.position += samplePosition;
                    sample.normal += sampleNormal;
                    sample.weight += 1f;
                    samples[sampleIndex] = sample;
                    valid[sampleIndex] = true;
                }
            }
        }

        List<Vector3> skinVertices = new List<Vector3>(sampleCount);
        Dictionary<int, int> compactMap = new Dictionary<int, int>(sampleCount);
        for (int i = 0; i < sampleCount; i++)
        {
            if (!valid[i] || samples[i].weight <= 0f)
                continue;

            Vector3 position = samples[i].position / samples[i].weight;
            Vector3 normal = samples[i].normal.sqrMagnitude > 1e-8f ? samples[i].normal.normalized : averageNormal;
            position += normal * regularSkinOffsetMeters;
            compactMap[i] = skinVertices.Count;
            skinVertices.Add(position);
        }

        if (skinVertices.Count < 3)
            return null;

        float maxEdge = Mathf.Max(step * 2.5f, regularSkinMaxEdgeMeters);
        float maxEdgeSqr = maxEdge * maxEdge;
        HashSet<ulong> edgeSet = new HashSet<ulong>();
        List<int> lineIndices = new List<int>(skinVertices.Count * 6);

        for (int y = 0; y < rows - 1; y++)
        {
            for (int x = 0; x < cols - 1; x++)
            {
                int i00 = y * cols + x;
                int i10 = i00 + 1;
                int i01 = i00 + cols;
                int i11 = i01 + 1;

                bool v00 = compactMap.TryGetValue(i00, out int c00);
                bool v10 = compactMap.TryGetValue(i10, out int c10);
                bool v01 = compactMap.TryGetValue(i01, out int c01);
                bool v11 = compactMap.TryGetValue(i11, out int c11);

                if (v00 && v10 && v11 && ValidEdgeTriplet(skinVertices, c00, c10, c11, maxEdgeSqr))
                    AddTriangleEdges(edgeSet, lineIndices, c00, c10, c11);
                if (v00 && v11 && v01 && ValidEdgeTriplet(skinVertices, c00, c11, c01, maxEdgeSqr))
                    AddTriangleEdges(edgeSet, lineIndices, c00, c11, c01);
                else if (v00 && v10 && v01 && ValidEdgeTriplet(skinVertices, c00, c10, c01, maxEdgeSqr))
                    AddTriangleEdges(edgeSet, lineIndices, c00, c10, c01);
                else if (v10 && v11 && v01 && ValidEdgeTriplet(skinVertices, c10, c11, c01, maxEdgeSqr))
                    AddTriangleEdges(edgeSet, lineIndices, c10, c11, c01);
            }
        }

        if (lineIndices.Count <= 0)
            return null;

        Mesh skinMesh = new Mesh
        {
            name = $"{sourceMesh.name}_RegularSkin"
        };
        if (skinVertices.Count > 65535)
            skinMesh.indexFormat = IndexFormat.UInt32;
        skinMesh.SetVertices(skinVertices);
        skinMesh.SetIndices(lineIndices, MeshTopology.Lines, 0, true);
        skinMesh.RecalculateBounds();
        return skinMesh;
    }

    private static float Cross2D(Vector2 a, Vector2 b)
    {
        return a.x * b.y - a.y * b.x;
    }

    private static bool TryGetBarycentric(Vector2 a, Vector2 b, Vector2 c, Vector2 p, out float u, out float v, out float w)
    {
        Vector2 v0 = b - a;
        Vector2 v1 = c - a;
        Vector2 v2 = p - a;
        float denom = Cross2D(v0, v1);
        if (Mathf.Abs(denom) <= 1e-7f)
        {
            u = v = w = 0f;
            return false;
        }

        v = Cross2D(v2, v1) / denom;
        w = Cross2D(v0, v2) / denom;
        u = 1f - v - w;
        const float tolerance = -0.001f;
        return u >= tolerance && v >= tolerance && w >= tolerance;
    }

    private static bool ValidEdgeTriplet(List<Vector3> vertices, int a, int b, int c, float maxEdgeSqr)
    {
        return (vertices[a] - vertices[b]).sqrMagnitude <= maxEdgeSqr
               && (vertices[b] - vertices[c]).sqrMagnitude <= maxEdgeSqr
               && (vertices[c] - vertices[a]).sqrMagnitude <= maxEdgeSqr;
    }

    private static void AddTriangleEdges(HashSet<ulong> edgeSet, List<int> indices, int a, int b, int c)
    {
        AddEdge(edgeSet, indices, a, b);
        AddEdge(edgeSet, indices, b, c);
        AddEdge(edgeSet, indices, c, a);
    }

    private static void AddEdge(HashSet<ulong> edgeSet, List<int> indices, int a, int b)
    {
        if (a == b)
            return;
        uint min = (uint)Mathf.Min(a, b);
        uint max = (uint)Mathf.Max(a, b);
        ulong key = ((ulong)min << 32) | max;
        if (!edgeSet.Add(key))
            return;
        indices.Add(a);
        indices.Add(b);
    }

    private Material ResolveRegularSkinMaterial()
    {
        if (regularSkinMaterialOverride != null)
            return regularSkinMaterialOverride;

        if (_runtimeRegularSkinMaterial != null)
            return _runtimeRegularSkinMaterial;

        Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null)
            shader = Shader.Find("Sprites/Default");
        if (shader == null)
            shader = Shader.Find("Unlit/Color");
        if (shader == null)
            shader = Shader.Find("Standard");

        _runtimeRegularSkinMaterial = new Material(shader)
        {
            name = "ScanCover_RuntimeRegularSkin"
        };
        if (_runtimeRegularSkinMaterial.HasProperty("_BaseColor"))
            _runtimeRegularSkinMaterial.SetColor("_BaseColor", regularSkinColor);
        if (_runtimeRegularSkinMaterial.HasProperty("_Color"))
            _runtimeRegularSkinMaterial.SetColor("_Color", regularSkinColor);
        _runtimeRegularSkinMaterial.renderQueue = (int)RenderQueue.Transparent;
        return _runtimeRegularSkinMaterial;
    }

    private void IntegrateSnapshotGroupIntoReferenceShell(Transform snapshotGroup, out int keptSurfaces, out int keptTriangles, out int rejectedTriangles)
    {
        keptSurfaces = 0;
        keptTriangles = 0;
        rejectedTriangles = 0;

        if (snapshotGroup == null || !maintainIncrementalReferenceShell)
            return;

        EnsureIncrementalShellRoot();
        if (_incrementalShellRoot == null)
            return;

        MeshFilter[] sourceFilters = snapshotGroup.GetComponentsInChildren<MeshFilter>(true);
        if (sourceFilters == null || sourceFilters.Length == 0)
            return;

        MeshFilter[] referenceFilters = _incrementalShellRoot.GetComponentsInChildren<MeshFilter>(true);
        if (referenceFilters == null || referenceFilters.Length == 0)
        {
            CloneSnapshotGroupIntoIncrementalShell(snapshotGroup);
            keptSurfaces = sourceFilters.Length;
            for (int i = 0; i < sourceFilters.Length; i++)
            {
                Mesh mesh = sourceFilters[i] != null ? sourceFilters[i].sharedMesh : null;
                if (mesh != null && mesh.triangles != null)
                    keptTriangles += mesh.triangles.Length / 3;
            }
            return;
        }

        Dictionary<Vector3Int, List<IndexedTriangle>> triangleIndex = BuildReferenceTriangleIndex(referenceFilters, out Vector3 referenceCenter);
        float cellSize = Mathf.Max(0.01f, incrementalNeighborRadius);

        for (int i = 0; i < sourceFilters.Length; i++)
        {
            MeshFilter sourceFilter = sourceFilters[i];
            if (sourceFilter == null || sourceFilter.sharedMesh == null)
                continue;

            Mesh filteredMesh = FilterMeshAgainstReference(
                sourceFilter.sharedMesh,
                triangleIndex,
                referenceCenter,
                cellSize,
                out int filterKeptTriangles,
                out int filterRejectedTriangles);

            keptTriangles += filterKeptTriangles;
            rejectedTriangles += filterRejectedTriangles;
            if (filteredMesh == null || filterKeptTriangles <= 0)
                continue;

            MeshRenderer sourceRenderer = sourceFilter.GetComponent<MeshRenderer>();
            GameObject go = new GameObject(sourceFilter.gameObject.name, typeof(MeshFilter), typeof(MeshRenderer));
            go.transform.SetParent(_incrementalShellRoot.transform, false);
            go.transform.localPosition = Vector3.zero;
            go.transform.localRotation = Quaternion.identity;
            go.transform.localScale = Vector3.one;

            go.GetComponent<MeshFilter>().sharedMesh = filteredMesh;
            ApplyRendererCopy(sourceRenderer, go.GetComponent<MeshRenderer>(), filteredMesh);
            keptSurfaces++;
        }
    }

    private void CloneSnapshotGroupIntoIncrementalShell(Transform snapshotGroup)
    {
        MeshFilter[] sourceFilters = snapshotGroup.GetComponentsInChildren<MeshFilter>(true);
        for (int i = 0; i < sourceFilters.Length; i++)
        {
            MeshFilter sourceFilter = sourceFilters[i];
            if (sourceFilter == null || sourceFilter.sharedMesh == null)
                continue;

            GameObject go = new GameObject(sourceFilter.gameObject.name, typeof(MeshFilter), typeof(MeshRenderer));
            go.transform.SetParent(_incrementalShellRoot.transform, false);
            go.transform.localPosition = Vector3.zero;
            go.transform.localRotation = Quaternion.identity;
            go.transform.localScale = Vector3.one;

            go.GetComponent<MeshFilter>().sharedMesh = Instantiate(sourceFilter.sharedMesh);
            ApplyRendererCopy(sourceFilter.GetComponent<MeshRenderer>(), go.GetComponent<MeshRenderer>(), sourceFilter.sharedMesh);
        }
    }

    private Dictionary<Vector3Int, List<IndexedTriangle>> BuildReferenceTriangleIndex(MeshFilter[] filters, out Vector3 referenceCenter)
    {
        Dictionary<Vector3Int, List<IndexedTriangle>> grid = new Dictionary<Vector3Int, List<IndexedTriangle>>(4096);
        float cellSize = Mathf.Max(0.01f, incrementalNeighborRadius);
        referenceCenter = ComputeMeshCenter(filters);

        for (int i = 0; i < filters.Length; i++)
        {
            MeshFilter filter = filters[i];
            Mesh mesh = filter != null ? filter.sharedMesh : null;
            if (mesh == null)
                continue;

            Vector3[] vertices = mesh.vertices;
            int[] triangles = mesh.triangles;
            if (vertices == null || triangles == null || triangles.Length < 3)
                continue;

            for (int t = 0; t <= triangles.Length - 3; t += 3)
            {
                Vector3 a = vertices[triangles[t]];
                Vector3 b = vertices[triangles[t + 1]];
                Vector3 c = vertices[triangles[t + 2]];
                Vector3 normal = Vector3.Cross(b - a, c - a);
                if (normal.sqrMagnitude < 1e-10f)
                    continue;

                Vector3 centroid = (a + b + c) / 3f;
                if (Vector3.Dot(normal, centroid - referenceCenter) < 0f)
                    normal = -normal;

                IndexedTriangle tri = new IndexedTriangle
                {
                    centroid = centroid,
                    normal = normal.normalized
                };

                Vector3Int key = Quantize(centroid, cellSize);
                if (!grid.TryGetValue(key, out List<IndexedTriangle> bucket))
                {
                    bucket = new List<IndexedTriangle>(8);
                    grid.Add(key, bucket);
                }
                bucket.Add(tri);
            }
        }

        return grid;
    }

    private Mesh FilterMeshAgainstReference(
        Mesh sourceMesh,
        Dictionary<Vector3Int, List<IndexedTriangle>> referenceIndex,
        Vector3 referenceCenter,
        float cellSize,
        out int keptTriangles,
        out int rejectedTriangles)
    {
        keptTriangles = 0;
        rejectedTriangles = 0;

        if (sourceMesh == null)
            return null;

        Vector3[] vertices = sourceMesh.vertices;
        Vector3[] normals = sourceMesh.normals;
        int[] triangles = sourceMesh.triangles;
        if (vertices == null || triangles == null || triangles.Length < 3)
            return null;

        bool hasNormals = normals != null && normals.Length == vertices.Length;
        Dictionary<int, int> remap = new Dictionary<int, int>(vertices.Length);
        List<Vector3> keptVertices = new List<Vector3>(vertices.Length);
        List<Vector3> keptNormals = hasNormals ? new List<Vector3>(vertices.Length) : null;
        List<int> keptIndices = new List<int>(triangles.Length);

        for (int t = 0; t <= triangles.Length - 3; t += 3)
        {
            int i0 = triangles[t];
            int i1 = triangles[t + 1];
            int i2 = triangles[t + 2];
            Vector3 a = vertices[i0];
            Vector3 b = vertices[i1];
            Vector3 c = vertices[i2];
            Vector3 centroid = (a + b + c) / 3f;
            Vector3 normal = Vector3.Cross(b - a, c - a);
            if (normal.sqrMagnitude < 1e-10f)
            {
                rejectedTriangles++;
                continue;
            }

            if (Vector3.Dot(normal, centroid - referenceCenter) < 0f)
                normal = -normal;
            normal.Normalize();

            if (ShouldRejectTriangle(centroid, normal, referenceIndex, cellSize))
            {
                rejectedTriangles++;
                continue;
            }

            keptIndices.Add(GetOrAddVertex(i0, vertices, normals, hasNormals, remap, keptVertices, keptNormals));
            keptIndices.Add(GetOrAddVertex(i1, vertices, normals, hasNormals, remap, keptVertices, keptNormals));
            keptIndices.Add(GetOrAddVertex(i2, vertices, normals, hasNormals, remap, keptVertices, keptNormals));
            keptTriangles++;
        }

        if (keptTriangles <= 0)
            return null;

        Mesh filtered = new Mesh
        {
            name = $"{sourceMesh.name}_Incremental"
        };
        filtered.indexFormat = keptVertices.Count > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
        filtered.SetVertices(keptVertices);
        filtered.SetTriangles(keptIndices, 0, true);
        if (hasNormals && keptNormals != null && keptNormals.Count == keptVertices.Count)
            filtered.SetNormals(keptNormals);
        else
            filtered.RecalculateNormals();
        filtered.RecalculateBounds();
        return filtered;
    }

    private bool ShouldRejectTriangle(
        Vector3 centroid,
        Vector3 normal,
        Dictionary<Vector3Int, List<IndexedTriangle>> referenceIndex,
        float cellSize)
    {
        Vector3Int key = Quantize(centroid, cellSize);
        float duplicateDistanceSqr = incrementalDuplicateDistance * incrementalDuplicateDistance;
        float neighborDistanceSqr = incrementalNeighborRadius * incrementalNeighborRadius;

        for (int x = -1; x <= 1; x++)
        {
            for (int y = -1; y <= 1; y++)
            {
                for (int z = -1; z <= 1; z++)
                {
                    if (!referenceIndex.TryGetValue(key + new Vector3Int(x, y, z), out List<IndexedTriangle> bucket))
                        continue;

                    for (int i = 0; i < bucket.Count; i++)
                    {
                        IndexedTriangle other = bucket[i];
                        Vector3 delta = centroid - other.centroid;
                        float distanceSqr = delta.sqrMagnitude;
                        float normalDot = Vector3.Dot(normal, other.normal);
                        if (distanceSqr <= duplicateDistanceSqr && Mathf.Abs(normalDot) >= incrementalMinNormalDot)
                            return true;

                        float planeDelta = Vector3.Dot(delta, other.normal);
                        Vector3 lateral = delta - other.normal * planeDelta;
                        if (lateral.sqrMagnitude <= neighborDistanceSqr &&
                            normalDot >= incrementalMinNormalDot &&
                            planeDelta <= -incrementalInsideDistance)
                        {
                            return true;
                        }
                    }
                }
            }
        }

        return false;
    }

    private static int GetOrAddVertex(
        int sourceIndex,
        Vector3[] sourceVertices,
        Vector3[] sourceNormals,
        bool hasNormals,
        Dictionary<int, int> remap,
        List<Vector3> keptVertices,
        List<Vector3> keptNormals)
    {
        if (remap.TryGetValue(sourceIndex, out int existing))
            return existing;

        int next = keptVertices.Count;
        remap.Add(sourceIndex, next);
        keptVertices.Add(sourceVertices[sourceIndex]);
        if (hasNormals && keptNormals != null)
            keptNormals.Add(sourceNormals[sourceIndex]);
        return next;
    }

    private static Vector3 ComputeMeshCenter(MeshFilter[] filters)
    {
        Vector3 sum = Vector3.zero;
        int count = 0;
        for (int i = 0; i < filters.Length; i++)
        {
            Mesh mesh = filters[i] != null ? filters[i].sharedMesh : null;
            if (mesh == null || mesh.vertices == null)
                continue;

            Vector3[] vertices = mesh.vertices;
            for (int v = 0; v < vertices.Length; v++)
            {
                sum += vertices[v];
                count++;
            }
        }

        return count > 0 ? (sum / count) : Vector3.zero;
    }

    private static Vector3Int Quantize(Vector3 value, float cellSize)
    {
        float safe = Mathf.Max(0.0001f, cellSize);
        return new Vector3Int(
            Mathf.RoundToInt(value.x / safe),
            Mathf.RoundToInt(value.y / safe),
            Mathf.RoundToInt(value.z / safe));
    }

    private static void SetSnapshotGroupVisible(Transform snapshotGroup, bool visible)
    {
        if (snapshotGroup == null)
            return;

        MeshRenderer[] renderers = snapshotGroup.GetComponentsInChildren<MeshRenderer>(true);
        for (int i = 0; i < renderers.Length; i++)
        {
            if (renderers[i] != null)
                renderers[i].enabled = visible;
        }
    }

    private void ApplyRendererCopy(MeshRenderer source, MeshRenderer target, Mesh targetMesh)
    {
        if (source == null || target == null)
            return;

        if (IsLineOnlyGeometry(targetMesh) && source.sharedMaterial != null)
            target.sharedMaterial = source.sharedMaterial;
        else
        {
            Material snapshotMaterial = ResolveSnapshotMaterial();
            if (snapshotMaterial != null)
                target.sharedMaterial = snapshotMaterial;
        }

        target.shadowCastingMode = source.shadowCastingMode;
        target.receiveShadows = true;
        target.motionVectorGenerationMode = source.motionVectorGenerationMode;
        target.lightProbeUsage = source.lightProbeUsage;
        target.reflectionProbeUsage = source.reflectionProbeUsage;
        target.allowOcclusionWhenDynamic = source.allowOcclusionWhenDynamic;
        target.renderingLayerMask = source.renderingLayerMask;
        target.sortingLayerID = source.sortingLayerID;
        target.sortingOrder = source.sortingOrder;
        target.enabled = true;
    }

    private static bool IsLineOnlyGeometry(Mesh mesh)
    {
        if (mesh == null || mesh.subMeshCount <= 0)
            return false;

        bool hasLines = false;
        for (int subMesh = 0; subMesh < mesh.subMeshCount; subMesh++)
        {
            MeshTopology topology = mesh.GetTopology(subMesh);
            int indexCount = (int)mesh.GetIndexCount(subMesh);
            if (topology == MeshTopology.Triangles && indexCount >= 3)
                return false;
            if (topology == MeshTopology.Lines && indexCount >= 2)
                hasLines = true;
        }

        return hasLines;
    }

    private Material ResolveSnapshotMaterial()
    {
        if (snapshotMaterialOverride != null)
            return snapshotMaterialOverride;

        if (_runtimeSnapshotMaterial == null)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null)
                shader = Shader.Find("Standard");

            if (shader != null)
                _runtimeSnapshotMaterial = new Material(shader) { name = "ScanCover_SnapshotSurfaceMaterial" };
        }

        if (_runtimeSnapshotMaterial != null)
        {
            if (_runtimeSnapshotMaterial.HasProperty("_BaseColor"))
                _runtimeSnapshotMaterial.SetColor("_BaseColor", snapshotColor);
            if (_runtimeSnapshotMaterial.HasProperty("_Color"))
                _runtimeSnapshotMaterial.SetColor("_Color", snapshotColor);
            if (_runtimeSnapshotMaterial.HasProperty("_Smoothness"))
                _runtimeSnapshotMaterial.SetFloat("_Smoothness", 0.18f);
            if (_runtimeSnapshotMaterial.HasProperty("_Metallic"))
                _runtimeSnapshotMaterial.SetFloat("_Metallic", 0f);
            if (_runtimeSnapshotMaterial.HasProperty("_Surface"))
                _runtimeSnapshotMaterial.SetFloat("_Surface", 0f);
            if (_runtimeSnapshotMaterial.HasProperty("_Blend"))
                _runtimeSnapshotMaterial.SetFloat("_Blend", 0f);
            if (_runtimeSnapshotMaterial.HasProperty("_ZWrite"))
                _runtimeSnapshotMaterial.SetFloat("_ZWrite", 1f);
            if (_runtimeSnapshotMaterial.HasProperty("_Cull"))
                _runtimeSnapshotMaterial.SetFloat("_Cull", snapshotDoubleSided ? (float)CullMode.Off : (float)CullMode.Back);
            if (_runtimeSnapshotMaterial.HasProperty("_SrcBlend"))
                _runtimeSnapshotMaterial.SetFloat("_SrcBlend", (float)BlendMode.One);
            if (_runtimeSnapshotMaterial.HasProperty("_DstBlend"))
                _runtimeSnapshotMaterial.SetFloat("_DstBlend", (float)BlendMode.Zero);
            if (_runtimeSnapshotMaterial.HasProperty("_Mode"))
                _runtimeSnapshotMaterial.SetFloat("_Mode", 0f);
            _runtimeSnapshotMaterial.renderQueue = (int)RenderQueue.Geometry;
        }

        return _runtimeSnapshotMaterial;
    }

    private void DestroySnapshotGroup(GameObject group)
    {
        if (group == null)
            return;
        MeshFilter[] filters = group.GetComponentsInChildren<MeshFilter>(true);
        for (int i = 0; i < filters.Length; i++)
        {
            if (filters[i] != null && filters[i].sharedMesh != null)
                SafeDestroy(filters[i].sharedMesh);
        }
        MeshRenderer[] renderers = group.GetComponentsInChildren<MeshRenderer>(true);
        for (int i = 0; i < renderers.Length; i++)
        {
            if (renderers[i] == null)
                continue;
            Material[] materials = renderers[i].sharedMaterials;
            if (materials == null)
                continue;
            for (int m = 0; m < materials.Length; m++)
            {
                if (materials[m] != null)
                    SafeDestroy(materials[m]);
            }
        }
        SafeDestroy(group);
    }

    private static void SafeDestroy(Object target)
    {
        if (target == null)
            return;

        if (Application.isPlaying)
            Object.Destroy(target);
        else
            Object.DestroyImmediate(target);
    }
}
