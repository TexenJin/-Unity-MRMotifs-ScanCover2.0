using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;

[DefaultExecutionOrder(30)]
[DisallowMultipleComponent]
[RequireComponent(typeof(RoomSetupReplicaV2Reconstruction))]
[RequireComponent(typeof(RoomSetupReplicaV2StructuralExtractor))]
public sealed class RoomSetupReplicaV2CaptureExporter : MonoBehaviour
{
    private const OVRInput.RawButton ExportButton =
        OVRInput.RawButton.LThumbstick;
    [SerializeField] private string exportDirectoryOverride = "";
    [SerializeField] private bool hideStructuralPreview = true;
    [SerializeField, Range(1, 10)]
    private int stabilityRecordingIntegrationInterval = 4;
    [SerializeField, Range(0.5f, 10f)]
    private float stabilitySnapshotTimeoutSeconds = 2f;

    private RoomSetupReplicaV2Reconstruction _reconstruction;
    private RoomSetupReplicaV2StructuralExtractor _structural;
    private RoomSetupReplicaV2EnvironmentDepthScanner _scanner;
    private bool _exportInProgress;
    private string _sessionDirectory;
    private Vector3[] _meshVertices;
    private float[] _meshStability;
    private uint[] _meshCounters;
    private GameObject _uiRoot;
    private Text _statusText;
    private Image _statusBackground;
    private Coroutine _pulseStop;
    private bool _exportButtonWasPressed;
    private string _lastInputSource = "none";
    private bool _wasScanning;
    private bool _burstInProgress;
    private bool _burstSnapshotPending;
    private bool _burstFinalizing;
    private bool _burstStopRequested;
    private bool _exportAfterBurst;
    private int _burstFramesRequested;
    private int _burstFramesCompleted;
    private int _burstNextIntegration;
    private int _burstStartIntegration;
    private float _burstSnapshotStartedAt;
    private string _burstDirectory;
    private string _lastCompletedBurstDirectory = "";
    private Task _burstWriteTask = Task.CompletedTask;
    private readonly List<int> _burstCapturedIntegrations = new();

    public string LastExportDirectory { get; private set; } = "";
    public string LastStatus { get; private set; } = "ready";
    public bool ExportInProgress => _exportInProgress;
    public bool StabilityBurstInProgress => _burstInProgress;

    private void Awake()
    {
        _reconstruction = GetComponent<RoomSetupReplicaV2Reconstruction>();
        _structural = GetComponent<RoomSetupReplicaV2StructuralExtractor>();
        _scanner = GetComponent<RoomSetupReplicaV2EnvironmentDepthScanner>();
        if (hideStructuralPreview)
            _structural.SetPreviewVisible(false);
    }

    private void Start()
    {
        EnsureStatusUi();
        _exportButtonWasPressed = ReadExportButtonState(out _);
        _wasScanning = _scanner != null && _scanner.IsScanning;
        SetStatus(_reconstruction != null &&
                  _reconstruction.CapturesFusionStability
                ? "稳定度：洋红变化 / 黄色收敛 / 青色稳定\n" +
                  "扳机：扫描并记录 / 左摇杆：停止并导出"
                : "按下左摇杆：导出房间数据",
            new Color(0.08f, 0.12f, 0.16f, 0.78f));
        if (_wasScanning && _reconstruction != null &&
            _reconstruction.CapturesFusionStability)
            BeginStabilityBurst();
        Debug.Log("[RoomSetupReplicaV2CaptureExporter] ready; " +
                  "left-thumbstick input armed (OVR + Unity XR)", this);
    }

    private void Update()
    {
        EnsureStatusUi();
        bool scanning = _scanner != null && _scanner.IsScanning;
        if (scanning && !_wasScanning && !_exportInProgress &&
            !_burstInProgress && _reconstruction != null &&
            _reconstruction.CapturesFusionStability)
        {
            BeginStabilityBurst();
        }
        _wasScanning = scanning;

        bool buttonDown = ReadExportButtonDown(out string source);
        if (!_exportInProgress && buttonDown)
        {
            _lastInputSource = source;
            SetStatus("已收到左摇杆，正在停止记录并导出…",
                new Color(0.10f, 0.24f, 0.40f, 0.82f));
            Debug.Log("[RoomSetupReplicaV2CaptureExporter] left thumbstick " +
                      "pressed via " + source, this);
            if (_burstInProgress)
                RequestStopStabilityRecordingAndExport();
            else
                BeginExport();
        }
        if (_burstInProgress)
            PumpStabilityBurst();
        if (_exportInProgress || buttonDown)
            return;
    }

    private bool ReadExportButtonDown(out string source)
    {
        bool ovrDown =
            OVRInput.GetDown(ExportButton, OVRInput.Controller.LTouch) ||
            OVRInput.GetDown(ExportButton, OVRInput.Controller.Touch) ||
            OVRInput.GetDown(ExportButton, OVRInput.Controller.Active);
        bool pressed = ReadExportButtonState(out source);
        bool stateDown = pressed && !_exportButtonWasPressed;
        _exportButtonWasPressed = pressed;
        if (!ovrDown && !stateDown)
        {
            source = "none";
            return false;
        }

        if (source == "none")
            source = "OVR.GetDown";
        return true;
    }

    private static bool ReadExportButtonState(out string source)
    {
        bool ovrPressed =
            OVRInput.Get(ExportButton, OVRInput.Controller.LTouch) ||
            OVRInput.Get(ExportButton, OVRInput.Controller.Touch) ||
            OVRInput.Get(ExportButton, OVRInput.Controller.Active) ||
            OVRInput.Get(OVRInput.Button.PrimaryThumbstick,
                OVRInput.Controller.LTouch);

        InputDevice leftHand =
            InputDevices.GetDeviceAtXRNode(XRNode.LeftHand);
        bool xrPressed = leftHand.isValid &&
                         leftHand.TryGetFeatureValue(
                             CommonUsages.primary2DAxisClick,
                             out bool click) && click;

        if (ovrPressed && xrPressed)
            source = "OVR+UnityXR";
        else if (ovrPressed)
            source = "OVR";
        else if (xrPressed)
            source = "UnityXR";
        else
            source = "none";
        return ovrPressed || xrPressed;
    }

    [ContextMenu("Begin Stability Recording")]
    public void BeginStabilityBurst()
    {
        if (_burstInProgress || _exportInProgress)
            return;
        if (_reconstruction == null ||
            !_reconstruction.CapturesFusionStability)
        {
            SetStatus("连续抓帧仅用于带稳定度的融合验证模式",
                new Color(0.48f, 0.20f, 0.04f, 0.88f));
            return;
        }
        if (_scanner == null || !_scanner.IsScanning)
        {
            SetStatus("抓帧失败：请先保持扫描运行",
                new Color(0.48f, 0.06f, 0.06f, 0.88f));
            return;
        }
        try
        {
            _burstDirectory = CreateStabilityRecordingDirectory();
            Directory.CreateDirectory(_burstDirectory);
        }
        catch (Exception exception)
        {
            SetStatus("抓帧失败：" + exception.GetType().Name,
                new Color(0.48f, 0.06f, 0.06f, 0.88f));
            return;
        }

        _burstInProgress = true;
        _burstSnapshotPending = false;
        _burstFinalizing = false;
        _burstStopRequested = false;
        _exportAfterBurst = false;
        _burstFramesRequested = 0;
        _burstFramesCompleted = 0;
        _burstStartIntegration = _reconstruction.IntegrationCount;
        _burstNextIntegration = Mathf.Max(1, _burstStartIntegration);
        _burstWriteTask = Task.CompletedTask;
        _burstCapturedIntegrations.Clear();
        WriteStabilityBurstStatus("capturing", null);
        SetStatus("稳定度序列记录中  0 帧",
            new Color(0.30f, 0.16f, 0.42f, 0.86f));
        Debug.Log("[RoomSetupReplicaV2CaptureExporter] stability recording " +
                  "started; minimumIntegrationInterval=" +
                  stabilityRecordingIntegrationInterval, this);
    }

    private void PumpStabilityBurst()
    {
        if (!_burstInProgress || _burstFinalizing)
            return;
        if (_scanner == null)
        {
            FailStabilityBurst("抓帧失败：扫描器缺失");
            return;
        }
        if (_burstSnapshotPending &&
            Time.realtimeSinceStartup - _burstSnapshotStartedAt >=
            stabilitySnapshotTimeoutSeconds)
        {
            _reconstruction?.CancelMeshSnapshot();
            FailStabilityBurst("抓帧失败：网格快照超时，扫描已继续");
            return;
        }
        if (_burstWriteTask != null && !_burstWriteTask.IsCompleted)
            return;
        if (_burstWriteTask != null && _burstWriteTask.IsFaulted)
        {
            FailStabilityBurst("抓帧失败：帧文件写入异常");
            return;
        }
        if (_burstStopRequested && !_burstSnapshotPending)
        {
            FinalizeStabilityBurstAsync();
            return;
        }
        if (!_scanner.IsScanning || _burstSnapshotPending ||
            _burstStopRequested)
            return;

        int integration = _reconstruction.IntegrationCount;
        if (integration <= 0 || integration < _burstNextIntegration)
            return;
        Camera camera = Camera.main;
        if (camera == null) camera = FindAnyObjectByType<Camera>();
        Vector3 cameraPosition = camera != null
            ? camera.transform.position : Vector3.zero;
        Quaternion cameraRotation = camera != null
            ? camera.transform.rotation : Quaternion.identity;
        float realtime = Time.realtimeSinceStartup;
        string capturedAt = DateTime.Now.ToString(
            "o", CultureInfo.InvariantCulture);
        int frameIndex = _burstFramesRequested;
        bool accepted = _reconstruction.TryRequestMeshSnapshot(
            (vertices, stability, counters) => OnStabilityBurstSnapshot(
                frameIndex, integration, realtime, capturedAt,
                cameraPosition, cameraRotation,
                vertices, stability, counters));
        if (!accepted)
            return;
        _burstSnapshotPending = true;
        _burstSnapshotStartedAt = Time.realtimeSinceStartup;
        _burstFramesRequested++;
        _burstNextIntegration = integration +
                                stabilityRecordingIntegrationInterval;
    }

    private void OnStabilityBurstSnapshot(
        int frameIndex, int integration, float realtime,
        string capturedAt, Vector3 cameraPosition,
        Quaternion cameraRotation, Vector3[] vertices,
        float[] stability, uint[] counters)
    {
        _burstSnapshotPending = false;
        if (!_burstInProgress)
            return;
        if (vertices == null || vertices.Length < 3 ||
            stability == null || stability.Length != vertices.Length)
        {
            FailStabilityBurst(
                "抓帧失败：网格与稳定度数据未对齐");
            return;
        }

        uint[] safeCounters = counters ?? Array.Empty<uint>();
        string binaryPath = Path.Combine(_burstDirectory,
            "frame_" + frameIndex.ToString("D3") + ".bin");
        string metadataPath = Path.Combine(_burstDirectory,
            "frame_" + frameIndex.ToString("D3") + ".json");
        _burstWriteTask = Task.Run(() =>
        {
            WriteStabilityBurstFrameBinary(
                binaryPath, frameIndex, integration, realtime,
                cameraPosition, cameraRotation, vertices,
                stability, safeCounters);
            File.WriteAllText(metadataPath,
                BuildStabilityBurstFrameMetadataJson(
                    frameIndex, integration, realtime, capturedAt,
                    cameraPosition, cameraRotation, vertices.Length,
                    safeCounters), new UTF8Encoding(false));
        });
        _burstCapturedIntegrations.Add(integration);
        _burstFramesCompleted++;
        WriteStabilityBurstStatus("capturing", null);
        SetStatus("稳定度序列记录中  " + _burstFramesCompleted + " 帧",
            new Color(0.30f, 0.16f, 0.42f, 0.86f));
    }

    private void RequestStopStabilityRecordingAndExport()
    {
        if (!_burstInProgress || _burstStopRequested)
            return;
        _burstStopRequested = true;
        _exportAfterBurst = true;
        if (_scanner != null && _scanner.IsScanning)
            _scanner.PauseScanning();
        SetStatus("正在封口 " + _burstFramesCompleted +
                  " 帧记录并准备最终导出…",
            new Color(0.10f, 0.24f, 0.40f, 0.82f));
        WriteStabilityBurstStatus("finalizing", null);
    }

    private async void FinalizeStabilityBurstAsync()
    {
        if (_burstFinalizing)
            return;
        _burstFinalizing = true;
        try
        {
            if (_burstWriteTask != null)
                await _burstWriteTask;
            string manifest = BuildStabilityBurstManifestJson();
            await Task.Run(() => File.WriteAllText(
                Path.Combine(_burstDirectory, "manifest.json"),
                manifest, new UTF8Encoding(false)));
            _lastCompletedBurstDirectory = _burstDirectory;
            _burstInProgress = false;
            _burstFinalizing = false;
            WriteStabilityBurstStatus("success", null);
            Debug.Log("[RoomSetupReplicaV2CaptureExporter] stability " +
                      "recording completed at " + _burstDirectory, this);
            if (_exportAfterBurst)
            {
                _exportAfterBurst = false;
                BeginExport();
                return;
            }
            SetStatus("稳定度序列已封口  " +
                      Path.GetFileName(_burstDirectory),
                new Color(0.05f, 0.38f, 0.18f, 0.86f));
        }
        catch (Exception exception)
        {
            FailStabilityBurst(
                "抓帧失败：" + exception.GetType().Name);
            Debug.LogException(exception, this);
        }
    }

    private void FailStabilityBurst(string message)
    {
        _reconstruction?.CancelMeshSnapshot();
        _burstInProgress = false;
        _burstSnapshotPending = false;
        _burstFinalizing = false;
        WriteStabilityBurstStatus("failed", message);
        SetStatus(message, new Color(0.48f, 0.06f, 0.06f, 0.88f));
        Debug.LogWarning("[RoomSetupReplicaV2CaptureExporter] " + message,
            this);
    }

    [ContextMenu("Export V2 Room Capture")]
    public void BeginExport()
    {
        if (_exportInProgress || _burstInProgress)
            return;
        LastExportDirectory = "";
        if (_reconstruction == null || _structural == null)
        {
            Fail("导出失败：V2 数据源缺失");
            return;
        }
        if (_reconstruction.IntegrationCount <= 0)
        {
            Fail("导出失败：当前没有扫描数据");
            return;
        }

        _exportInProgress = true;
        _meshVertices = null;
        _meshStability = null;
        _meshCounters = null;
        if (_scanner != null && _scanner.IsScanning)
            _scanner.PauseScanning();
        _sessionDirectory = CreateSessionDirectory();
        LastExportDirectory = _sessionDirectory;
        WriteRootStatus(null, "exporting", "export_started");
        SetStatus("正在锁定并读取最终网格…",
            new Color(0.10f, 0.24f, 0.40f, 0.82f));
        if (!_reconstruction.TryRequestMeshSnapshot(OnMeshSnapshotReady))
            Fail("导出失败：GPU 网格尚未就绪");
    }

    private void OnMeshSnapshotReady(
        Vector3[] vertices, float[] stability, uint[] counters)
    {
        if (!_exportInProgress)
            return;
        if (vertices == null || vertices.Length < 3)
        {
            Fail("导出失败：最终网格读取为空");
            return;
        }
        if (_reconstruction.CapturesFusionStability &&
            (stability == null || stability.Length != vertices.Length))
        {
            Fail("导出失败：稳定度与最终网格未对齐");
            return;
        }
        _meshVertices = vertices;
        _meshStability = stability ?? Array.Empty<float>();
        _meshCounters = counters ?? Array.Empty<uint>();
        SetStatus("正在刷新结构采样…",
            new Color(0.10f, 0.24f, 0.40f, 0.82f));
        StartCoroutine(RequestFreshStructuralSnapshot());
    }

    private IEnumerator RequestFreshStructuralSnapshot()
    {
        float deadline = Time.realtimeSinceStartup + 6f;
        while (Time.realtimeSinceStartup < deadline)
        {
            if (_structural.TryRefreshForExport(
                    OnStructuralSnapshotReady))
                yield break;
            yield return new WaitForSecondsRealtime(0.10f);
        }
        Fail("导出失败：结构采样读取超时");
    }

    private void OnStructuralSnapshotReady(bool success)
    {
        if (!_exportInProgress)
            return;
        if (!success)
        {
            Fail("导出失败：结构采样无有效数据");
            return;
        }
        WriteExportPackageAsync();
    }

    private async void WriteExportPackageAsync()
    {
        SetStatus("正在写入 OBJ 和 JSON…",
            new Color(0.10f, 0.24f, 0.40f, 0.82f));
        string samples = _structural.BuildStructuralSamplesJson();
        string clusters = _structural.BuildPlaneClustersJson();
        string grids = _structural.BuildSupportGridsJson();
        string before = _structural.BuildPlanesJson(true);
        string after = _structural.BuildPlanesJson(false);
        string extractorMetadata = _structural.BuildExtractorMetadataJson();
        string simplifiedRoom = _structural.BuildSimplifiedRoomJson();
        string captureMetadata = BuildCaptureMetadataJson();
        Vector3[] vertices = _meshVertices;
        float[] stability = _meshStability ?? Array.Empty<float>();
        uint[] counters = _meshCounters;
        string directory = _sessionDirectory;
        try
        {
            await Task.Run(() =>
            {
                Directory.CreateDirectory(directory);
                WriteMeshObj(Path.Combine(directory, "final_mesh.obj"),
                    vertices);
                WriteMeshJson(Path.Combine(directory, "final_mesh.json"),
                    vertices);
                WriteMeshStabilityJson(Path.Combine(directory,
                    "mesh_stability.json"), stability);
                File.WriteAllText(Path.Combine(directory,
                    "mesh_counters.json"), BuildCountersJson(counters),
                    new UTF8Encoding(false));
                File.WriteAllText(Path.Combine(directory,
                    "structural_samples.json"), samples,
                    new UTF8Encoding(false));
                File.WriteAllText(Path.Combine(directory,
                    "plane_clusters.json"), clusters,
                    new UTF8Encoding(false));
                File.WriteAllText(Path.Combine(directory,
                    "support_grids.json"), grids,
                    new UTF8Encoding(false));
                File.WriteAllText(Path.Combine(directory,
                    "planes_before_topology.json"), before,
                    new UTF8Encoding(false));
                File.WriteAllText(Path.Combine(directory,
                    "planes_after_topology.json"), after,
                    new UTF8Encoding(false));
                File.WriteAllText(Path.Combine(directory,
                    "extractor_meta.json"), extractorMetadata,
                    new UTF8Encoding(false));
                File.WriteAllText(Path.Combine(directory,
                    "simplified_room_online.json"), simplifiedRoom,
                    new UTF8Encoding(false));
                File.WriteAllText(Path.Combine(directory,
                    "capture_meta.json"), captureMetadata,
                    new UTF8Encoding(false));
                File.WriteAllText(Path.Combine(directory, "README.txt"),
                    "RoomSetupReplicaV2 independent Environment Depth " +
                    "capture. final_mesh coordinates are Unity world " +
                    "coordinates in meters. OBJ vertices are non-indexed " +
                    "Marching Cubes triangle vertices. " +
                    "When stabilityProbe is true, mesh_stability.json " +
                    "contains one aligned normalized value per vertex: " +
                    "0=changing/magenta, intermediate=settling/yellow, " +
                    "1=stable/cyan. Stability means temporal depth " +
                    "consistency, not semantic correctness.\n",
                    new UTF8Encoding(false));
            });
            LastExportDirectory = directory;
            _exportInProgress = false;
            LastStatus = "success";
            WriteRootStatus(true, "success", "success");
            SetStatus("导出成功  " + Path.GetFileName(directory) +
                      "\n扫描已暂停，可通过 ADB 读取",
                new Color(0.05f, 0.38f, 0.18f, 0.86f));
            Pulse(0.55f, 0.16f);
            Debug.Log("[RoomSetupReplicaV2CaptureExporter] exported " +
                      vertices.Length + " vertices to " + directory, this);
        }
        catch (Exception exception)
        {
            Fail("导出失败：" + exception.GetType().Name);
            Debug.LogException(exception, this);
        }
    }

    private string CreateSessionDirectory()
    {
        string root = GetExportRoot();
        return Path.Combine(root,
            DateTime.Now.ToString("yyyyMMdd_HHmmss",
                CultureInfo.InvariantCulture));
    }

    private string CreateStabilityRecordingDirectory()
    {
        return Path.Combine(GetExportRoot(), "StabilityRecordings",
            DateTime.Now.ToString("yyyyMMdd_HHmmss_fff",
                CultureInfo.InvariantCulture));
    }

    private string GetExportRoot()
    {
        return string.IsNullOrWhiteSpace(exportDirectoryOverride)
            ? Path.Combine(Application.persistentDataPath,
                "RoomSetupReplicaV2Exports")
            : exportDirectoryOverride;
    }

    private string BuildCaptureMetadataJson()
    {
        Camera camera = Camera.main;
        if (camera == null) camera = FindAnyObjectByType<Camera>();
        Vector3 position = camera != null
            ? camera.transform.position : Vector3.zero;
        Quaternion rotation = camera != null
            ? camera.transform.rotation : Quaternion.identity;
        return "{\n" +
               "  \"exportedAtLocal\": " +
               Json(DateTime.Now.ToString("o",
                   CultureInfo.InvariantCulture)) + ",\n" +
               "  \"applicationVersion\": " +
               Json(Application.version) + ",\n" +
               "  \"route\": " + Json(_reconstruction.RouteLabel) +
               ",\n  \"stabilityProbe\": " +
               (_reconstruction.LatestSampleStabilityProbe
                   ? "true" : "false") +
               ",\n  \"fusionMode\": " +
               Json(_reconstruction.FusionMode.ToString()) +
               ",\n  \"fusionStabilityCaptured\": " +
               (_reconstruction.CapturesFusionStability
                   ? "true" : "false") +
               ",\n  \"stabilityRecordingDirectory\": " +
               Json(_lastCompletedBurstDirectory) +
               ",\n  \"stabilityRecordingFrameCount\": " +
               _burstFramesCompleted +
               ",\n  \"coordinateSystem\": " +
               Json("Unity world, meters, +Y up") + ",\n" +
               "  \"exportButton\": " + Json("LThumbstick") + ",\n" +
               "  \"inputSource\": " + Json(_lastInputSource) + ",\n" +
               "  \"structuralPreviewHidden\": true,\n" +
               "  \"integrationCount\": " +
               _reconstruction.IntegrationCount + ",\n" +
               "  \"acceptedPairCount\": " +
               _reconstruction.AcceptedPairCount + ",\n" +
               "  \"activeChunkCount\": " +
               _reconstruction.ActiveChunkCount + ",\n" +
               "  \"meshVertexCount\": " + _meshVertices.Length +
               ",\n  \"stabilityVertexCount\": " +
               (_meshStability != null ? _meshStability.Length : 0) +
               ",\n  \"meshTriangleCount\": " +
               (_meshVertices.Length / 3) + ",\n" +
               "  \"cameraPosition\": " + VectorJson(position) +
               ",\n  \"cameraRotation\": [" +
               Number(rotation.x) + "," + Number(rotation.y) + "," +
               Number(rotation.z) + "," + Number(rotation.w) + "]\n" +
               "}\n";
    }

    private static void WriteMeshObj(string path, Vector3[] vertices)
    {
        using var writer = new StreamWriter(
            path, false, new UTF8Encoding(false), 1 << 20);
        writer.WriteLine("# RoomSetupReplicaV2 final GPU mesh");
        writer.WriteLine("# Unity world coordinates, meters");
        for (int i = 0; i < vertices.Length; i++)
        {
            Vector3 vertex = vertices[i];
            writer.Write("v ");
            writer.Write(Number(vertex.x));
            writer.Write(' ');
            writer.Write(Number(vertex.y));
            writer.Write(' ');
            writer.WriteLine(Number(vertex.z));
        }
        for (int i = 0; i + 2 < vertices.Length; i += 3)
            writer.WriteLine("f " + (i + 1) + " " + (i + 2) + " " +
                             (i + 3));
    }

    private static void WriteMeshJson(string path, Vector3[] vertices)
    {
        using var writer = new StreamWriter(
            path, false, new UTF8Encoding(false), 1 << 20);
        writer.Write("{\n  \"vertexCount\": ");
        writer.Write(vertices.Length);
        writer.Write(",\n  \"triangleCount\": ");
        writer.Write(vertices.Length / 3);
        writer.Write(",\n  \"vertices\": [\n");
        for (int i = 0; i < vertices.Length; i++)
        {
            writer.Write("    ");
            writer.Write(VectorJson(vertices[i]));
            writer.WriteLine(i + 1 < vertices.Length ? "," : "");
        }
        writer.Write("  ],\n  \"indices\": [");
        for (int i = 0; i < vertices.Length; i++)
        {
            if (i > 0) writer.Write(',');
            writer.Write(i);
        }
        writer.Write("]\n}\n");
    }

    private static void WriteMeshStabilityJson(
        string path, float[] stability)
    {
        using var writer = new StreamWriter(
            path, false, new UTF8Encoding(false), 1 << 20);
        writer.Write("{\n  \"mode\": ");
        writer.Write(Json("latest-sample temporal consistency"));
        writer.Write(",\n  \"valueRange\": [0,1],\n  \"legend\": {");
        writer.Write("\n    \"changing\": \"magenta\",");
        writer.Write("\n    \"settling\": \"yellow\",");
        writer.Write("\n    \"stable\": \"cyan\"");
        writer.Write("\n  },\n  \"vertexCount\": ");
        writer.Write(stability.Length);
        writer.Write(",\n  \"values\": [");
        for (int i = 0; i < stability.Length; i++)
        {
            if (i > 0) writer.Write(',');
            writer.Write(Number(stability[i]));
        }
        writer.Write("]\n}\n");
    }

    private static void WriteStabilityBurstFrameBinary(
        string path, int frameIndex, int integration, float realtime,
        Vector3 cameraPosition, Quaternion cameraRotation,
        Vector3[] vertices, float[] stability, uint[] counters)
    {
        using var stream = new FileStream(
            path, FileMode.Create, FileAccess.Write, FileShare.None,
            1 << 20);
        using var writer = new BinaryWriter(
            stream, new UTF8Encoding(false));
        writer.Write(Encoding.ASCII.GetBytes("RSV2SB01"));
        writer.Write(1); // little-endian format version
        writer.Write(frameIndex);
        writer.Write(integration);
        writer.Write(realtime);
        writer.Write(cameraPosition.x);
        writer.Write(cameraPosition.y);
        writer.Write(cameraPosition.z);
        writer.Write(cameraRotation.x);
        writer.Write(cameraRotation.y);
        writer.Write(cameraRotation.z);
        writer.Write(cameraRotation.w);
        writer.Write(vertices.Length);
        writer.Write(counters.Length);
        for (int i = 0; i < counters.Length; i++)
            writer.Write(counters[i]);
        for (int i = 0; i < vertices.Length; i++)
        {
            writer.Write(vertices[i].x);
            writer.Write(vertices[i].y);
            writer.Write(vertices[i].z);
            writer.Write(stability[i]);
        }
    }

    private static string BuildStabilityBurstFrameMetadataJson(
        int frameIndex, int integration, float realtime,
        string capturedAt, Vector3 cameraPosition,
        Quaternion cameraRotation, int vertexCount, uint[] counters)
    {
        var counterText = new StringBuilder();
        for (int i = 0; i < counters.Length; i++)
        {
            if (i > 0) counterText.Append(',');
            counterText.Append(counters[i]);
        }
        return "{\n" +
               "  \"frameIndex\": " + frameIndex + ",\n" +
               "  \"integrationCount\": " + integration + ",\n" +
               "  \"realtimeSinceStartup\": " + Number(realtime) +
               ",\n  \"capturedAtLocal\": " + Json(capturedAt) +
               ",\n  \"binaryFile\": " +
               Json("frame_" + frameIndex.ToString("D3") + ".bin") +
               ",\n  \"vertexCount\": " + vertexCount +
               ",\n  \"triangleCount\": " + vertexCount / 3 +
               ",\n  \"cameraPosition\": " +
               VectorJson(cameraPosition) +
               ",\n  \"cameraRotation\": [" +
               Number(cameraRotation.x) + "," +
               Number(cameraRotation.y) + "," +
               Number(cameraRotation.z) + "," +
               Number(cameraRotation.w) + "],\n" +
               "  \"counters\": [" + counterText + "]\n" +
               "}\n";
    }

    private string BuildStabilityBurstManifestJson()
    {
        var integrations = new StringBuilder();
        var files = new StringBuilder();
        for (int i = 0; i < _burstCapturedIntegrations.Count; i++)
        {
            if (i > 0)
            {
                integrations.Append(',');
                files.Append(',');
            }
            integrations.Append(_burstCapturedIntegrations[i]);
            files.Append(Json("frame_" + i.ToString("D3") + ".bin"));
        }
        return "{\n" +
               "  \"format\": \"RSV2SB01\",\n" +
               "  \"formatVersion\": 1,\n" +
               "  \"endianness\": \"little\",\n" +
               "  \"route\": " + Json(_reconstruction.RouteLabel) +
               ",\n  \"frameCount\": " + _burstFramesCompleted +
               ",\n  \"startIntegration\": " +
               _burstStartIntegration +
               ",\n  \"minimumIntegrationInterval\": " +
               stabilityRecordingIntegrationInterval +
               ",\n  \"capturedIntegrations\": [" + integrations +
               "],\n  \"files\": [" + files + "],\n" +
               "  \"recordLayout\": " +
               Json("magic[8],version:i32,frame:i32,integration:i32," +
                    "realtime:f32,cameraPosition:3xf32," +
                    "cameraRotation:4xf32,vertexCount:i32," +
                    "counterCount:i32,counters:u32[]," +
                    "vertices:[x:f32,y:f32,z:f32,stability:f32][]") +
               ",\n  \"stabilityMeaning\": " +
               Json("temporal consistency only; 0 changing, 1 stable") +
               "\n}\n";
    }

    private static string BuildCountersJson(uint[] counters)
    {
        var builder = new StringBuilder("{\n  \"counters\": [");
        for (int i = 0; i < counters.Length; i++)
        {
            if (i > 0) builder.Append(',');
            builder.Append(counters[i]);
        }
        builder.Append("],\n  \"meaning\": [")
            .Append(Json("requestedCandidateVertices")).Append(',')
            .Append(Json("overflowTriangles")).Append(',')
            .Append(Json("rejectedCrossLayerCells")).Append(',')
            .Append(Json("emittedCandidateTriangles")).Append(',')
            .Append(Json("survivingCompactedVertices")).Append(',')
            .Append(Json("removedSmallComponentTriangles")).Append(',')
            .Append(Json("cleanupApplied"))
            .Append(',').Append(Json("probeChangingTriangles"))
            .Append(',').Append(Json("probeSettlingTriangles"))
            .Append(',').Append(Json("probeStableTriangles"))
            .Append("]\n}\n");
        return builder.ToString();
    }

    private void EnsureStatusUi()
    {
        if (_uiRoot != null)
            return;
        Camera camera = Camera.main;
        if (camera == null) camera = FindAnyObjectByType<Camera>();
        if (camera == null)
            return;
        _uiRoot = new GameObject("V2 Export Status UI");
        _uiRoot.transform.SetParent(camera.transform, false);
        _uiRoot.transform.localPosition = new Vector3(0f, -0.23f, 0.85f);
        _uiRoot.transform.localRotation = Quaternion.identity;
        _uiRoot.transform.localScale = Vector3.one * 0.00075f;
        Canvas canvas = _uiRoot.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;
        canvas.worldCamera = camera;
        canvas.sortingOrder = 200;
        RectTransform rootRect = _uiRoot.GetComponent<RectTransform>();
        rootRect.sizeDelta = new Vector2(520f, 86f);
        _statusBackground = _uiRoot.AddComponent<Image>();
        _statusBackground.color =
            new Color(0.08f, 0.12f, 0.16f, 0.78f);

        var textObject = new GameObject("Status");
        textObject.transform.SetParent(_uiRoot.transform, false);
        RectTransform textRect = textObject.AddComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = new Vector2(12f, 6f);
        textRect.offsetMax = new Vector2(-12f, -6f);
        _statusText = textObject.AddComponent<Text>();
        _statusText.font = Resources.GetBuiltinResource<Font>(
            "LegacyRuntime.ttf");
        _statusText.fontSize = 25;
        _statusText.alignment = TextAnchor.MiddleCenter;
        _statusText.color = Color.white;
        _statusText.horizontalOverflow = HorizontalWrapMode.Wrap;
        _statusText.verticalOverflow = VerticalWrapMode.Truncate;
    }

    private void SetStatus(string message, Color background)
    {
        LastStatus = message;
        EnsureStatusUi();
        if (_statusText != null) _statusText.text = message;
        if (_statusBackground != null) _statusBackground.color = background;
    }

    private void Fail(string message)
    {
        _exportInProgress = false;
        LastStatus = message;
        WriteRootStatus(false, "failed", message);
        SetStatus(message, new Color(0.48f, 0.06f, 0.06f, 0.88f));
        Pulse(0.25f, 0.25f);
        Debug.LogWarning("[RoomSetupReplicaV2CaptureExporter] " + message,
            this);
    }

    private void WriteRootStatus(bool? success, string state,
                                 string message)
    {
        try
        {
            string root = GetExportRoot();
            Directory.CreateDirectory(root);
            string successJson = success.HasValue
                ? (success.Value ? "true" : "false")
                : "null";
            string json = "{\n  \"success\": " + successJson +
                          ",\n  \"state\": " + Json(state) +
                          ",\n  \"message\": " + Json(message) +
                          ",\n  \"inputSource\": " +
                          Json(_lastInputSource) +
                          ",\n  \"lastExportDirectory\": " +
                          Json(LastExportDirectory) + "\n}\n";
            File.WriteAllText(Path.Combine(root, "export_status.json"),
                json, new UTF8Encoding(false));
        }
        catch (Exception exception)
        {
            Debug.LogWarning("[RoomSetupReplicaV2CaptureExporter] " +
                             "could not write export_status.json: " +
                             exception.Message, this);
        }
    }

    private void WriteStabilityBurstStatus(
        string state, string message)
    {
        try
        {
            string root = GetExportRoot();
            Directory.CreateDirectory(root);
            string json = "{\n  \"state\": " + Json(state) +
                          ",\n  \"message\": " + Json(message) +
                          ",\n  \"inputSource\": " +
                          Json(_lastInputSource) +
                          ",\n  \"frameCount\": " +
                          _burstFramesCompleted +
                          ",\n  \"directory\": " +
                          Json(_burstDirectory) + "\n}\n";
            File.WriteAllText(Path.Combine(root,
                    "stability_recording_status.json"),
                json, new UTF8Encoding(false));
        }
        catch (Exception exception)
        {
            Debug.LogWarning("[RoomSetupReplicaV2CaptureExporter] " +
                             "could not write stability recording status: " +
                             exception.Message, this);
        }
    }

    private void Pulse(float amplitude, float seconds)
    {
        OVRInput.SetControllerVibration(1f, amplitude,
            OVRInput.Controller.LTouch);
        if (_pulseStop != null) StopCoroutine(_pulseStop);
        _pulseStop = StartCoroutine(StopPulse(seconds));
    }

    private IEnumerator StopPulse(float seconds)
    {
        yield return new WaitForSecondsRealtime(seconds);
        OVRInput.SetControllerVibration(0f, 0f,
            OVRInput.Controller.LTouch);
        _pulseStop = null;
    }

    private static string VectorJson(Vector3 value)
    {
        return "[" + Number(value.x) + "," + Number(value.y) + "," +
               Number(value.z) + "]";
    }

    private static string Number(float value)
    {
        return float.IsFinite(value)
            ? value.ToString("R", CultureInfo.InvariantCulture)
            : "null";
    }

    private static string Json(string value)
    {
        if (value == null) return "null";
        return "\"" + value.Replace("\\", "\\\\")
            .Replace("\"", "\\\"")
            .Replace("\r", "\\r")
            .Replace("\n", "\\n") + "\"";
    }

    private void OnDestroy()
    {
        _reconstruction?.CancelMeshSnapshot();
        if (_uiRoot != null) Destroy(_uiRoot);
    }
}
