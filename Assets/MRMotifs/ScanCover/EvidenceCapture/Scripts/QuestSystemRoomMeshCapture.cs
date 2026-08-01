using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using Meta.XR.MRUtilityKit;
using UnityEngine;

namespace ScanCover.EvidenceCapture
{
    /// <summary>
    /// Captures Quest's already-scanned MRUK global room mesh as a companion artifact
    /// for a depth-evidence session. The source is represented by MeshFilters only;
    /// no Renderer is created and any renderer found on the global-mesh anchor is disabled.
    /// </summary>
    [DefaultExecutionOrder(550)]
    [DisallowMultipleComponent]
    public sealed class QuestSystemRoomMeshCapture : MonoBehaviour
    {
        private const string StatusSchema = "ScanCoverSystemRoomMeshCompanion/v1";

        [Header("Source")]
        [SerializeField] private MRUK mruk;
        [SerializeField] private bool includeAllLoadedRooms = true;
        [SerializeField] private bool disableSourceRenderers = true;

        [Header("Export")]
        [SerializeField] private ScanCoverMetaSceneMeshAuditExporter exporter;
        [SerializeField] private string outputFolderName = "system_room_mesh";
        [SerializeField] private bool writeChecksums = true;

        private Transform _proxyRoot;
        private bool _registeredForSceneLoad;
        private bool _exportAttempted;
        private string _sessionDirectory = string.Empty;

        public string LastStatus { get; private set; } = "not_started";
        public string LastIssue { get; private set; } = string.Empty;
        public int ReadyMeshCount { get; private set; }
        public int LastExportedMeshCount => exporter != null ? exporter.LastExportedMeshCount : 0;
        public int LastExportedVertexCount => exporter != null ? exporter.LastExportedVertexCount : 0;
        public int LastExportedTriangleCount => exporter != null ? exporter.LastExportedTriangleCount : 0;
        public string OutputFolderName => string.IsNullOrWhiteSpace(outputFolderName) ? "system_room_mesh" : outputFolderName;

        public void ConfigureForScene(MRUK mrukInstance, ScanCoverMetaSceneMeshAuditExporter meshExporter)
        {
            mruk = mrukInstance;
            exporter = meshExporter;
            ConfigureExporter();
        }

        private void Awake()
        {
            ResolveDependencies();
            ConfigureExporter();
        }

        private IEnumerator Start()
        {
            // MRUK can be created earlier in the same frame but initialize after this
            // component. Polling only resolves the singleton; mesh export remains event-driven.
            float deadline = Time.realtimeSinceStartup + 15f;
            while (mruk == null && Time.realtimeSinceStartup < deadline)
            {
                ResolveDependencies();
                if (mruk == null)
                    yield return null;
            }

            RegisterSceneLoadedCallback();
            RefreshHiddenMeshSources();
        }

        private void OnDisable()
        {
            if (_registeredForSceneLoad && mruk != null)
            {
                mruk.SceneLoadedEvent.RemoveListener(OnSceneLoaded);
            }
            _registeredForSceneLoad = false;
        }

        public void BeginSession(string sessionDirectory)
        {
            _sessionDirectory = sessionDirectory ?? string.Empty;
            _exportAttempted = false;
            LastStatus = "waiting_for_system_room_mesh";
            LastIssue = string.Empty;
            ResolveDependencies();
            RegisterSceneLoadedCallback();
            RefreshHiddenMeshSources();
        }

        public bool ExportForSession(string sessionDirectory)
        {
            if (!string.IsNullOrWhiteSpace(sessionDirectory))
                _sessionDirectory = sessionDirectory;

            if (_exportAttempted)
                return LastStatus == "exported";
            _exportAttempted = true;

            ResolveDependencies();
            RegisterSceneLoadedCallback();
            RefreshHiddenMeshSources();

            if (string.IsNullOrWhiteSpace(_sessionDirectory))
            {
                LastStatus = "failed";
                LastIssue = "Evidence session directory is empty.";
                return false;
            }

            if (exporter == null)
            {
                LastStatus = "failed";
                LastIssue = "ScanCoverMetaSceneMeshAuditExporter is missing.";
                WriteStatusFile();
                return false;
            }

            if (ReadyMeshCount <= 0 || _proxyRoot == null)
            {
                LastStatus = "unavailable";
                LastIssue = mruk == null
                    ? "MRUK is unavailable; no system room mesh was loaded."
                    : "MRUK loaded no GlobalMeshAnchor with triangle data.";
                WriteStatusFile();
                return false;
            }

            ConfigureExporter();
            string outputDirectory = Path.Combine(_sessionDirectory, OutputFolderName);
            bool success;
            try
            {
                success = exporter.ExportAuditPackageToDirectory(outputDirectory);
            }
            catch (Exception ex)
            {
                success = false;
                LastIssue = ex.GetType().Name + ": " + ex.Message;
            }

            if (success)
            {
                LastStatus = "exported";
                LastIssue = string.Empty;
                if (writeChecksums)
                    WriteChecksumFile(outputDirectory);
            }
            else
            {
                LastStatus = "failed";
                if (string.IsNullOrEmpty(LastIssue))
                    LastIssue = exporter.LastIssue ?? "System room mesh exporter returned false.";
            }

            WriteStatusFile();
            return success;
        }

        private void OnSceneLoaded()
        {
            RefreshHiddenMeshSources();
        }

        private void ResolveDependencies()
        {
            if (mruk == null)
                mruk = MRUK.Instance != null
                    ? MRUK.Instance
                    : FindAnyObjectByType<MRUK>(FindObjectsInactive.Include);
            if (exporter == null)
                exporter = GetComponent<ScanCoverMetaSceneMeshAuditExporter>();
        }

        private void RegisterSceneLoadedCallback()
        {
            if (_registeredForSceneLoad || mruk == null)
                return;
            mruk.RegisterSceneLoadedCallback(OnSceneLoaded);
            _registeredForSceneLoad = true;
        }

        private void ConfigureExporter()
        {
            if (exporter == null)
                return;
            exporter.sceneMeshRoot = _proxyRoot;
            exporter.searchWholeSceneWhenRootMissing = false;
            exporter.includeInactiveChildren = true;
            exporter.requireRenderer = false;
            exporter.requireRendererEnabled = false;
            exporter.sessionPrefix = "ScanCover_SystemRoomMesh";
            exporter.exportLocalRawObjects = false;
            exporter.exportWorldAlignedObjects = false;
            exporter.exportCombinedWorldObj = true;
            exporter.exportComponentInventory = true;
        }

        private void RefreshHiddenMeshSources()
        {
            ReadyMeshCount = 0;
            ClearProxyRoot();

            if (mruk == null)
            {
                LastStatus = "waiting_for_mruk";
                return;
            }

            IReadOnlyList<MRUKRoom> rooms = mruk.Rooms;
            if (rooms == null || rooms.Count == 0)
            {
                LastStatus = "waiting_for_system_room_mesh";
                return;
            }

            GameObject rootObject = new GameObject("[ScanCover] Hidden System Room Mesh Sources");
            rootObject.transform.SetParent(transform, false);
            _proxyRoot = rootObject.transform;

            MRUKRoom currentRoom = mruk.GetCurrentRoom();
            for (int roomIndex = 0; roomIndex < rooms.Count; roomIndex++)
            {
                MRUKRoom room = rooms[roomIndex];
                if (room == null || (!includeAllLoadedRooms && currentRoom != null && room != currentRoom))
                    continue;

                MRUKAnchor anchor = room.GlobalMeshAnchor;
                if (anchor == null)
                    continue;

                if (disableSourceRenderers)
                {
                    Renderer[] renderers = anchor.GetComponentsInChildren<Renderer>(true);
                    for (int rendererIndex = 0; rendererIndex < renderers.Length; rendererIndex++)
                    {
                        if (renderers[rendererIndex] != null)
                            renderers[rendererIndex].enabled = false;
                    }
                }

                Mesh mesh;
                try
                {
                    mesh = anchor.GlobalMesh;
                }
                catch (Exception ex)
                {
                    LastIssue = ex.GetType().Name + ": " + ex.Message;
                    continue;
                }

                if (mesh == null || mesh.vertexCount <= 0 || mesh.triangles == null || mesh.triangles.Length < 3)
                    continue;

                GameObject proxy = new GameObject($"SystemRoomMesh_{roomIndex:00}");
                proxy.transform.SetParent(_proxyRoot, false);
                proxy.transform.SetPositionAndRotation(anchor.transform.position, anchor.transform.rotation);
                proxy.transform.localScale = anchor.transform.lossyScale;
                MeshFilter filter = proxy.AddComponent<MeshFilter>();
                filter.sharedMesh = mesh;
                // Deliberately no MeshRenderer: this data can be exported but never displayed.
                ReadyMeshCount++;
            }

            if (ReadyMeshCount > 0)
            {
                LastStatus = "ready";
                LastIssue = string.Empty;
            }
            else
            {
                LastStatus = "waiting_for_system_room_mesh";
            }

            ConfigureExporter();
        }

        private void ClearProxyRoot()
        {
            if (_proxyRoot == null)
                return;
            Destroy(_proxyRoot.gameObject);
            _proxyRoot = null;
            ConfigureExporter();
        }

        private void WriteStatusFile()
        {
            if (string.IsNullOrWhiteSpace(_sessionDirectory))
                return;
            try
            {
                Directory.CreateDirectory(_sessionDirectory);
                var builder = new StringBuilder(1024);
                builder.AppendLine("{");
                AppendJsonString(builder, "schema", StatusSchema, true);
                AppendJsonString(builder, "writtenUtc", DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture), true);
                AppendJsonString(builder, "status", LastStatus, true);
                AppendJsonString(builder, "issue", LastIssue, true);
                AppendJsonString(builder, "source", "MRUK.Rooms[].GlobalMeshAnchor.GlobalMesh", true);
                AppendJsonString(builder, "relativeDirectory", OutputFolderName.Replace('\\', '/'), true);
                AppendJsonBool(builder, "renderedInCaptureScene", false, true);
                AppendJsonNumber(builder, "sourceMeshCount", ReadyMeshCount, true);
                AppendJsonNumber(builder, "exportedMeshCount", LastExportedMeshCount, true);
                AppendJsonNumber(builder, "exportedVertexCount", LastExportedVertexCount, true);
                AppendJsonNumber(builder, "exportedTriangleCount", LastExportedTriangleCount, false);
                builder.AppendLine("}");
                File.WriteAllText(
                    Path.Combine(_sessionDirectory, "system_room_mesh_status.json"),
                    builder.ToString(),
                    new UTF8Encoding(false));
            }
            catch (Exception ex)
            {
                Debug.LogWarning("[System Room Mesh Capture] Failed to write status: " + ex.Message, this);
            }
        }

        private static void WriteChecksumFile(string directory)
        {
            string checksumPath = Path.Combine(directory, "checksums.sha256");
            string[] files = Directory.GetFiles(directory, "*", SearchOption.AllDirectories);
            Array.Sort(files, StringComparer.Ordinal);
            var builder = new StringBuilder(files.Length * 96);
            using (SHA256 sha = SHA256.Create())
            {
                for (int index = 0; index < files.Length; index++)
                {
                    string file = files[index];
                    if (string.Equals(file, checksumPath, StringComparison.OrdinalIgnoreCase))
                        continue;
                    using (FileStream stream = File.OpenRead(file))
                    {
                        byte[] hash = sha.ComputeHash(stream);
                        builder.Append(BitConverter.ToString(hash).Replace("-", string.Empty).ToLowerInvariant())
                            .Append("  ")
                            .Append(Path.GetRelativePath(directory, file).Replace('\\', '/'))
                            .AppendLine();
                    }
                }
            }
            File.WriteAllText(checksumPath, builder.ToString(), new UTF8Encoding(false));
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
}
