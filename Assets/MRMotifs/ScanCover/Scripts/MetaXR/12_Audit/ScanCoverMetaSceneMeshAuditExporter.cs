using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

[DisallowMultipleComponent]
public sealed class ScanCoverMetaSceneMeshAuditExporter : MonoBehaviour
{
    [Header("Source")]
    public Transform sceneMeshRoot;
    public bool searchWholeSceneWhenRootMissing = true;
    public bool includeInactiveChildren = true;
    public bool requireRenderer = false;
    public bool requireRendererEnabled = false;

    [Header("Export")]
    public string exportDirectoryOverride = "";
    public string sessionPrefix = "ScanCover_MetaSceneMeshAudit";
    public bool exportLocalRawObjects = true;
    public bool exportWorldAlignedObjects = true;
    public bool exportCombinedWorldObj = true;
    public bool exportComponentInventory = true;
    public bool debugLog = true;

    public string LastExportDirectory { get; private set; }
    public string LastIssue { get; private set; }
    public int LastExportedMeshCount { get; private set; }
    public int LastExportedVertexCount { get; private set; }
    public int LastExportedTriangleCount { get; private set; }

    private sealed class MeshRecord
    {
        public int id;
        public MeshFilter filter;
        public MeshRenderer renderer;
        public Mesh mesh;
        public string path;
        public Bounds worldBounds;
        public int vertexCount;
        public int triangleCount;
    }

    [ContextMenu("Export Meta Scene Mesh Audit Package")]
    public void ExportFromContextMenu()
    {
        ExportAuditPackage();
    }

    public bool ExportAuditPackage()
    {
        return ExportAuditPackageToDirectory(CreateSessionDirectory());
    }

    public bool ExportAuditPackageToDirectory(string sessionDir)
    {
        LastIssue = null;
        LastExportDirectory = "";
        LastExportedMeshCount = 0;
        LastExportedVertexCount = 0;
        LastExportedTriangleCount = 0;

        List<MeshRecord> records = CollectMeshRecords();
        if (records.Count == 0)
        {
            LastIssue = "No eligible MeshFilter was found for Meta Scene Mesh audit export.";
            if (debugLog)
                Debug.LogWarning($"[ScanCoverMetaSceneMeshAuditExporter] {LastIssue}", this);
            return false;
        }

        if (string.IsNullOrWhiteSpace(sessionDir))
        {
            LastIssue = "The requested Meta Scene Mesh export directory is empty.";
            if (debugLog)
                Debug.LogWarning($"[ScanCoverMetaSceneMeshAuditExporter] {LastIssue}", this);
            return false;
        }

        sessionDir = Path.GetFullPath(sessionDir);
        Directory.CreateDirectory(sessionDir);

        string localDir = Path.Combine(sessionDir, "raw_local_meshes");
        string worldDir = Path.Combine(sessionDir, "aligned_world_meshes");
        if (exportLocalRawObjects)
            Directory.CreateDirectory(localDir);
        if (exportWorldAlignedObjects)
            Directory.CreateDirectory(worldDir);

        StringBuilder combinedObj = new StringBuilder(1024 * 64);
        combinedObj.AppendLine("# ScanCover Meta Scene Mesh aligned combined OBJ");
        combinedObj.AppendLine("# Vertices are baked to Unity world space.");
        int combinedVertexOffset = 0;
        int exportedMeshes = 0;
        int exportedVertices = 0;
        int exportedTriangles = 0;

        for (int i = 0; i < records.Count; i++)
        {
            MeshRecord record = records[i];
            string safeName = MakeSafeFileName($"{record.id:000}_{record.filter.gameObject.name}");

            if (exportLocalRawObjects)
            {
                string path = Path.Combine(localDir, safeName + "_local.obj");
                File.WriteAllText(path, BuildObj(record, bakeWorld: false, includeObjectHeader: true, ref combinedVertexOffset, mutateOffset: false), Encoding.UTF8);
            }

            if (exportWorldAlignedObjects)
            {
                string path = Path.Combine(worldDir, safeName + "_world.obj");
                File.WriteAllText(path, BuildObj(record, bakeWorld: true, includeObjectHeader: true, ref combinedVertexOffset, mutateOffset: false), Encoding.UTF8);
            }

            if (exportCombinedWorldObj)
                combinedObj.Append(BuildObj(record, bakeWorld: true, includeObjectHeader: true, ref combinedVertexOffset, mutateOffset: true));

            exportedMeshes++;
            exportedVertices += record.vertexCount;
            exportedTriangles += record.triangleCount;
        }

        if (exportCombinedWorldObj)
            File.WriteAllText(Path.Combine(sessionDir, "meta_scene_mesh_aligned_all.obj"), combinedObj.ToString(), Encoding.UTF8);

        File.WriteAllText(Path.Combine(sessionDir, "mesh_filters.csv"), BuildMeshFilterCsv(records), Encoding.UTF8);
        if (exportComponentInventory)
            File.WriteAllText(Path.Combine(sessionDir, "component_inventory.csv"), BuildComponentInventoryCsv(records), Encoding.UTF8);
        File.WriteAllText(Path.Combine(sessionDir, "session_info.json"), BuildSessionInfoJson(records, exportedMeshes, exportedVertices, exportedTriangles), Encoding.UTF8);

        LastExportDirectory = sessionDir;
        LastExportedMeshCount = exportedMeshes;
        LastExportedVertexCount = exportedVertices;
        LastExportedTriangleCount = exportedTriangles;

        if (debugLog)
        {
            Debug.Log(
                $"[ScanCoverMetaSceneMeshAuditExporter] Exported meshes={exportedMeshes}, vertices={exportedVertices}, triangles={exportedTriangles} to {sessionDir}",
                this);
        }

        return true;
    }

    private List<MeshRecord> CollectMeshRecords()
    {
        MeshFilter[] filters;
        if (sceneMeshRoot != null)
        {
            filters = sceneMeshRoot.GetComponentsInChildren<MeshFilter>(includeInactiveChildren);
        }
        else if (searchWholeSceneWhenRootMissing)
        {
            filters = FindObjectsOfType<MeshFilter>(includeInactiveChildren);
        }
        else
        {
            filters = Array.Empty<MeshFilter>();
        }

        List<MeshRecord> records = new List<MeshRecord>(filters.Length);
        HashSet<MeshFilter> seen = new HashSet<MeshFilter>();
        for (int i = 0; i < filters.Length; i++)
        {
            MeshFilter filter = filters[i];
            if (filter == null || !seen.Add(filter))
                continue;

            Mesh mesh = filter.sharedMesh;
            if (mesh == null || mesh.vertexCount <= 0 || mesh.triangles == null || mesh.triangles.Length < 3)
                continue;

            MeshRenderer renderer = filter.GetComponent<MeshRenderer>();
            if (requireRenderer && renderer == null)
                continue;
            if (requireRendererEnabled && (renderer == null || !renderer.enabled))
                continue;

            MeshRecord record = new MeshRecord
            {
                id = records.Count,
                filter = filter,
                renderer = renderer,
                mesh = mesh,
                path = BuildTransformPath(filter.transform),
                vertexCount = mesh.vertexCount,
                triangleCount = mesh.triangles.Length / 3,
                worldBounds = ComputeWorldBounds(filter.transform, mesh)
            };
            records.Add(record);
        }

        return records;
    }

    private string CreateSessionDirectory()
    {
        string root = string.IsNullOrWhiteSpace(exportDirectoryOverride)
            ? Path.Combine(ProjectRoot(), "ScanCoverExports", "MetaSceneMeshAuditSessions")
            : exportDirectoryOverride;
        string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture);
        return Path.Combine(root, $"{sessionPrefix}_{stamp}");
    }

    private static string ProjectRoot()
    {
        string dataPath = Application.dataPath.Replace('\\', '/');
        return Directory.GetParent(dataPath)?.FullName ?? Application.dataPath;
    }

    private static string BuildObj(MeshRecord record, bool bakeWorld, bool includeObjectHeader, ref int vertexOffset, bool mutateOffset)
    {
        Mesh mesh = record.mesh;
        Vector3[] vertices = mesh.vertices;
        Vector3[] normals = mesh.normals;
        int[] triangles = mesh.triangles;
        Matrix4x4 localToWorld = record.filter.transform.localToWorldMatrix;
        Matrix4x4 normalMatrix = localToWorld.inverse.transpose;

        StringBuilder builder = new StringBuilder(vertices.Length * 48 + triangles.Length * 16);
        if (includeObjectHeader)
        {
            builder.Append("o ").Append(MakeObjName(record.path)).AppendLine();
            builder.Append("# path ").Append(record.path).AppendLine();
        }

        for (int i = 0; i < vertices.Length; i++)
        {
            Vector3 v = bakeWorld ? localToWorld.MultiplyPoint3x4(vertices[i]) : vertices[i];
            builder.Append("v ")
                .Append(F(v.x)).Append(' ')
                .Append(F(v.y)).Append(' ')
                .Append(F(v.z)).AppendLine();
        }

        bool hasNormals = normals != null && normals.Length == vertices.Length;
        if (hasNormals)
        {
            for (int i = 0; i < normals.Length; i++)
            {
                Vector3 n = bakeWorld ? normalMatrix.MultiplyVector(normals[i]).normalized : normals[i].normalized;
                builder.Append("vn ")
                    .Append(F(n.x)).Append(' ')
                    .Append(F(n.y)).Append(' ')
                    .Append(F(n.z)).AppendLine();
            }
        }

        int offset = mutateOffset ? vertexOffset : 0;
        for (int i = 0; i < triangles.Length; i += 3)
        {
            int a = triangles[i] + 1 + offset;
            int b = triangles[i + 1] + 1 + offset;
            int c = triangles[i + 2] + 1 + offset;
            if (hasNormals)
                builder.Append("f ").Append(a).Append("//").Append(a).Append(' ').Append(b).Append("//").Append(b).Append(' ').Append(c).Append("//").Append(c).AppendLine();
            else
                builder.Append("f ").Append(a).Append(' ').Append(b).Append(' ').Append(c).AppendLine();
        }

        if (mutateOffset)
            vertexOffset += vertices.Length;
        return builder.ToString();
    }

    private static string BuildMeshFilterCsv(List<MeshRecord> records)
    {
        StringBuilder builder = new StringBuilder(1024 * 8);
        builder.AppendLine("id,name,path,activeInHierarchy,rendererEnabled,layer,vertexCount,triangleCount,worldBoundsCenterX,worldBoundsCenterY,worldBoundsCenterZ,worldBoundsSizeX,worldBoundsSizeY,worldBoundsSizeZ,worldPosX,worldPosY,worldPosZ,worldRotX,worldRotY,worldRotZ,worldScaleX,worldScaleY,worldScaleZ,meshName,materialCount");
        for (int i = 0; i < records.Count; i++)
        {
            MeshRecord r = records[i];
            Transform t = r.filter.transform;
            Vector3 pos = t.position;
            Vector3 rot = t.rotation.eulerAngles;
            Vector3 scale = t.lossyScale;
            Bounds b = r.worldBounds;
            int materialCount = r.renderer != null && r.renderer.sharedMaterials != null ? r.renderer.sharedMaterials.Length : 0;
            builder.Append(r.id).Append(',')
                .Append(Csv(r.filter.gameObject.name)).Append(',')
                .Append(Csv(r.path)).Append(',')
                .Append(r.filter.gameObject.activeInHierarchy).Append(',')
                .Append(r.renderer != null && r.renderer.enabled).Append(',')
                .Append(r.filter.gameObject.layer).Append(',')
                .Append(r.vertexCount).Append(',')
                .Append(r.triangleCount).Append(',')
                .Append(F(b.center.x)).Append(',').Append(F(b.center.y)).Append(',').Append(F(b.center.z)).Append(',')
                .Append(F(b.size.x)).Append(',').Append(F(b.size.y)).Append(',').Append(F(b.size.z)).Append(',')
                .Append(F(pos.x)).Append(',').Append(F(pos.y)).Append(',').Append(F(pos.z)).Append(',')
                .Append(F(rot.x)).Append(',').Append(F(rot.y)).Append(',').Append(F(rot.z)).Append(',')
                .Append(F(scale.x)).Append(',').Append(F(scale.y)).Append(',').Append(F(scale.z)).Append(',')
                .Append(Csv(r.mesh.name)).Append(',')
                .Append(materialCount)
                .AppendLine();
        }
        return builder.ToString();
    }

    private static string BuildComponentInventoryCsv(List<MeshRecord> records)
    {
        StringBuilder builder = new StringBuilder(1024 * 8);
        builder.AppendLine("meshId,path,componentType,enabled");
        HashSet<Component> seen = new HashSet<Component>();
        for (int i = 0; i < records.Count; i++)
        {
            MeshRecord r = records[i];
            Component[] components = r.filter.GetComponents<Component>();
            for (int c = 0; c < components.Length; c++)
            {
                Component component = components[c];
                if (component == null || !seen.Add(component))
                    continue;
                bool enabled = component is Behaviour behaviour ? behaviour.enabled : true;
                builder.Append(r.id).Append(',')
                    .Append(Csv(r.path)).Append(',')
                    .Append(Csv(component.GetType().FullName)).Append(',')
                    .Append(enabled)
                    .AppendLine();
            }
        }
        return builder.ToString();
    }

    private string BuildSessionInfoJson(List<MeshRecord> records, int meshCount, int vertexCount, int triangleCount)
    {
        Bounds bounds = ComputeAggregateBounds(records);
        StringBuilder builder = new StringBuilder(2048);
        builder.AppendLine("{");
        builder.Append("  \"sessionPrefix\": ").Append(Json(sessionPrefix)).AppendLine(",");
        builder.Append("  \"exportedAtLocal\": ").Append(Json(DateTime.Now.ToString("o", CultureInfo.InvariantCulture))).AppendLine(",");
        builder.Append("  \"sceneName\": ").Append(Json(UnityEngine.SceneManagement.SceneManager.GetActiveScene().name)).AppendLine(",");
        builder.Append("  \"sourceRoot\": ").Append(Json(sceneMeshRoot != null ? BuildTransformPath(sceneMeshRoot) : "<whole-scene-search>")).AppendLine(",");
        builder.Append("  \"meshCount\": ").Append(meshCount).AppendLine(",");
        builder.Append("  \"vertexCount\": ").Append(vertexCount).AppendLine(",");
        builder.Append("  \"triangleCount\": ").Append(triangleCount).AppendLine(",");
        builder.AppendLine("  \"worldBounds\": {");
        builder.Append("    \"center\": [").Append(F(bounds.center.x)).Append(", ").Append(F(bounds.center.y)).Append(", ").Append(F(bounds.center.z)).AppendLine("],");
        builder.Append("    \"size\": [").Append(F(bounds.size.x)).Append(", ").Append(F(bounds.size.y)).Append(", ").Append(F(bounds.size.z)).AppendLine("]");
        builder.AppendLine("  }");
        builder.AppendLine("}");
        return builder.ToString();
    }

    private static Bounds ComputeWorldBounds(Transform transform, Mesh mesh)
    {
        Vector3[] vertices = mesh.vertices;
        if (vertices == null || vertices.Length == 0)
            return new Bounds(transform.position, Vector3.zero);

        Matrix4x4 matrix = transform.localToWorldMatrix;
        Bounds bounds = new Bounds(matrix.MultiplyPoint3x4(vertices[0]), Vector3.zero);
        for (int i = 1; i < vertices.Length; i++)
            bounds.Encapsulate(matrix.MultiplyPoint3x4(vertices[i]));
        return bounds;
    }

    private static Bounds ComputeAggregateBounds(List<MeshRecord> records)
    {
        if (records == null || records.Count == 0)
            return new Bounds(Vector3.zero, Vector3.zero);
        Bounds bounds = records[0].worldBounds;
        for (int i = 1; i < records.Count; i++)
            bounds.Encapsulate(records[i].worldBounds);
        return bounds;
    }

    private static string BuildTransformPath(Transform transform)
    {
        if (transform == null)
            return "";
        Stack<string> names = new Stack<string>();
        Transform t = transform;
        while (t != null)
        {
            names.Push(t.name);
            t = t.parent;
        }
        return string.Join("/", names.ToArray());
    }

    private static string MakeSafeFileName(string value)
    {
        if (string.IsNullOrEmpty(value))
            return "unnamed";
        char[] invalid = Path.GetInvalidFileNameChars();
        StringBuilder builder = new StringBuilder(value.Length);
        for (int i = 0; i < value.Length; i++)
        {
            char c = value[i];
            builder.Append(Array.IndexOf(invalid, c) >= 0 ? '_' : c);
        }
        return builder.ToString();
    }

    private static string MakeObjName(string value)
    {
        if (string.IsNullOrEmpty(value))
            return "unnamed";
        return value.Replace(' ', '_').Replace('/', '_').Replace('\\', '_').Replace(':', '_');
    }

    private static string Csv(string value)
    {
        if (value == null)
            return "";
        return "\"" + value.Replace("\"", "\"\"") + "\"";
    }

    private static string Json(string value)
    {
        if (value == null)
            return "null";
        return "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
    }

    private static string F(float value)
    {
        return value.ToString("R", CultureInfo.InvariantCulture);
    }
}
