using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(ScanCoverMetaSceneMeshAuditExporter))]
public sealed class ScanCoverMetaSceneMeshAuditExporterEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        ScanCoverMetaSceneMeshAuditExporter exporter = (ScanCoverMetaSceneMeshAuditExporter)target;
        EditorGUILayout.Space();
        if (GUILayout.Button("Export Meta Scene Mesh Audit Package"))
        {
            if (Application.isPlaying)
            {
                bool ok = exporter.ExportAuditPackage();
                if (ok)
                    EditorUtility.RevealInFinder(exporter.LastExportDirectory);
            }
            else
            {
                Debug.LogWarning("[ScanCoverMetaSceneMeshAuditExporter] Enter Play Mode first so Meta Scene Mesh can be loaded before export.", exporter);
            }
        }

        if (!string.IsNullOrEmpty(exporter.LastIssue))
            EditorGUILayout.HelpBox(exporter.LastIssue, MessageType.Warning);

        if (!string.IsNullOrEmpty(exporter.LastExportDirectory))
        {
            EditorGUILayout.HelpBox(
                $"Last export: {exporter.LastExportedMeshCount} meshes, {exporter.LastExportedVertexCount} vertices, {exporter.LastExportedTriangleCount} triangles\n{exporter.LastExportDirectory}",
                MessageType.Info);
        }
    }
}
