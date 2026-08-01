using System;
using System.Globalization;
using System.IO;
using System.Text;
using ScanCover.EvidenceCapture;
using UnityEngine;

[DefaultExecutionOrder(650)]
[DisallowMultipleComponent]
public sealed class ScanCoverProductionSystemRoomMeshExportController : MonoBehaviour
{
    [Header("Input")]
    [SerializeField] private bool enableRightControllerExportInput = true;
    [SerializeField] private OVRInput.RawButton exportButton = OVRInput.RawButton.B;
    [SerializeField, Min(0.1f)] private float inputDebounceSeconds = 0.75f;

    [Header("Output")]
    [SerializeField] private QuestSystemRoomMeshCapture roomMeshCapture;
    [SerializeField] private string diagnosticsFolderName = "ScanCoverDiagnostics";
    [SerializeField] private string exportSessionPrefix = "SystemRoomMesh";

    private float _nextAllowedExportTime;

    public string LastStatus { get; private set; } = "not_started";
    public string LastIssue { get; private set; } = string.Empty;
    public string LastExportDirectory { get; private set; } = string.Empty;

    private void Awake()
    {
        ResolveDependencies();
    }

    private void Update()
    {
        if (!enableRightControllerExportInput || Time.unscaledTime < _nextAllowedExportTime)
            return;

        if (OVRInput.GetDown(exportButton, OVRInput.Controller.RTouch))
            ExportSystemRoomMeshNow();
    }

    [ContextMenu("Export Invisible Quest System Room Mesh Now")]
    public void ExportSystemRoomMeshNow()
    {
        _nextAllowedExportTime = Time.unscaledTime + Mathf.Max(0.1f, inputDebounceSeconds);
        ResolveDependencies();

        string diagnosticsRoot = Path.Combine(
            Application.persistentDataPath,
            SanitizeFolderName(diagnosticsFolderName, "ScanCoverDiagnostics"));
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture);
        string sessionName = SanitizeFolderName(exportSessionPrefix, "SystemRoomMesh") + "_" + timestamp;
        string sessionDirectory = Path.Combine(diagnosticsRoot, sessionName);
        LastExportDirectory = sessionDirectory;

        try
        {
            Directory.CreateDirectory(sessionDirectory);
            if (roomMeshCapture == null)
            {
                LastStatus = "failed";
                LastIssue = "QuestSystemRoomMeshCapture is missing from the production scene.";
                WriteLatestPointer(diagnosticsRoot);
                Debug.LogWarning("[ScanCover] System room mesh export failed: " + LastIssue, this);
                return;
            }

            roomMeshCapture.BeginSession(sessionDirectory);
            bool success = roomMeshCapture.ExportForSession(sessionDirectory);
            LastStatus = roomMeshCapture.LastStatus;
            LastIssue = roomMeshCapture.LastIssue;
            WriteLatestPointer(diagnosticsRoot);

            if (success)
            {
                Debug.Log(
                    $"[ScanCover] Invisible system room mesh exported: {sessionDirectory} " +
                    $"meshes={roomMeshCapture.LastExportedMeshCount} " +
                    $"vertices={roomMeshCapture.LastExportedVertexCount} " +
                    $"triangles={roomMeshCapture.LastExportedTriangleCount}",
                    this);
            }
            else
            {
                Debug.LogWarning(
                    $"[ScanCover] System room mesh export status={LastStatus}: {LastIssue}. " +
                    $"Details: {Path.Combine(sessionDirectory, "system_room_mesh_status.json")}",
                    this);
            }
        }
        catch (Exception ex)
        {
            LastStatus = "failed";
            LastIssue = ex.GetType().Name + ": " + ex.Message;
            try
            {
                Directory.CreateDirectory(diagnosticsRoot);
                WriteLatestPointer(diagnosticsRoot);
            }
            catch (Exception)
            {
                // Preserve the original export error.
            }
            Debug.LogWarning("[ScanCover] System room mesh export failed: " + LastIssue, this);
        }
    }

    private void ResolveDependencies()
    {
        if (roomMeshCapture == null)
            roomMeshCapture = GetComponent<QuestSystemRoomMeshCapture>();
        if (roomMeshCapture == null)
            roomMeshCapture = FindAnyObjectByType<QuestSystemRoomMeshCapture>(FindObjectsInactive.Include);
    }

    private void WriteLatestPointer(string diagnosticsRoot)
    {
        Directory.CreateDirectory(diagnosticsRoot);
        var json = new StringBuilder(768);
        json.AppendLine("{");
        AppendJsonString(json, "schema", "ScanCoverProductionSystemRoomMeshExport/v1", true);
        AppendJsonString(json, "writtenUtc", DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture), true);
        AppendJsonString(json, "status", LastStatus, true);
        AppendJsonString(json, "issue", LastIssue, true);
        AppendJsonString(json, "exportDirectory", LastExportDirectory.Replace('\\', '/'), true);
        AppendJsonBool(json, "renderedInProductionScene", false, true);
        AppendJsonNumber(json, "meshCount", roomMeshCapture != null ? roomMeshCapture.LastExportedMeshCount : 0, true);
        AppendJsonNumber(json, "vertexCount", roomMeshCapture != null ? roomMeshCapture.LastExportedVertexCount : 0, true);
        AppendJsonNumber(json, "triangleCount", roomMeshCapture != null ? roomMeshCapture.LastExportedTriangleCount : 0, false);
        json.AppendLine("}");
        File.WriteAllText(
            Path.Combine(diagnosticsRoot, "latest_system_room_mesh_export.json"),
            json.ToString(),
            new UTF8Encoding(false));
    }

    private static string SanitizeFolderName(string value, string fallback)
    {
        string result = string.IsNullOrWhiteSpace(value) ? fallback : value.Trim();
        foreach (char invalid in Path.GetInvalidFileNameChars())
            result = result.Replace(invalid, '_');
        return result;
    }

    private static void AppendJsonString(StringBuilder builder, string name, string value, bool comma)
    {
        builder.Append("  \"").Append(name).Append("\": \"").Append(JsonEscape(value ?? string.Empty)).Append('"');
        if (comma) builder.Append(',');
        builder.AppendLine();
    }

    private static void AppendJsonBool(StringBuilder builder, string name, bool value, bool comma)
    {
        builder.Append("  \"").Append(name).Append("\": ").Append(value ? "true" : "false");
        if (comma) builder.Append(',');
        builder.AppendLine();
    }

    private static void AppendJsonNumber(StringBuilder builder, string name, int value, bool comma)
    {
        builder.Append("  \"").Append(name).Append("\": ").Append(value.ToString(CultureInfo.InvariantCulture));
        if (comma) builder.Append(',');
        builder.AppendLine();
    }

    private static string JsonEscape(string value)
    {
        return value.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\r", "\\r").Replace("\n", "\\n");
    }
}
