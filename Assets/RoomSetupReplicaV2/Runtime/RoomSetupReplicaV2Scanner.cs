using System;
using System.IO;
using Genesis.RoomScan;
using UnityEngine;

/// <summary>
/// Minimal capture loop for the isolated Room Setup replica scene. No
/// ScanCover VolumeIntegrator or MeshExtractor is instantiated.
/// </summary>
[DefaultExecutionOrder(-20)]
[DisallowMultipleComponent]
[RequireComponent(
    typeof(DepthCapture),
    typeof(RoomSetupReplicaV2Reconstruction))]
public sealed class RoomSetupReplicaV2Scanner : MonoBehaviour
{
    // Production trial route: keep the stable A reconstruction as the only
    // volume. The synchronized B experiment fragmented the principal room
    // surface and doubled reconstruction/meshing work, so it must not be
    // created or fed in this build. Keep this compile-time rather than
    // serialized so an old Unity scene value cannot silently restore B.
    private const bool SynchronizedAtomicABCapture = false;

    [Serializable]
    private sealed class LiveAbSessionReceipt
    {
        public string schema;
        public string createdUtc;
        public string coordinateSpace;
        public string captureMode;
        public string inputContract;
        public string armA;
        public string armB;
        public string displayedArm;
        public int synchronizedSubmittedPairs;
        public int oneArmSubmissionFailures;
        public int skippedPlatformFramesWhileBusy;
        public int acceptedPairsAWhenQueued;
        public int acceptedPairsBWhenQueued;
        public string completionContract;
    }

    public static RoomSetupReplicaV2Scanner Instance { get; private set; }

    [SerializeField, Range(1f, 30f)] private float captureHz = 10f;
    [SerializeField] private bool startAutomatically = false;

    public bool IsScanning { get; private set; }
    public RoomSetupReplicaV2Reconstruction Reconstruction =>
        _atomicReconstruction != null
            ? _atomicReconstruction
            : _reconstruction;

    private DepthCapture _depth;
    private RoomSetupReplicaV2Reconstruction _reconstruction;
    private RoomSetupReplicaV2Reconstruction _atomicReconstruction;
    private DepthPointCloudOverlay _depthPointCloudOverlay;
    private float _nextCaptureTime;
    private int _lastSubmittedPlatformFrame = -1;
    private int _lastDisplayPlatformFrame = -1;
    private int _lastBusyPlatformFrame = -1;
    private int _synchronizedSubmittedPairCount;
    private int _oneArmSubmissionFailureCount;
    private int _skippedPlatformFramesWhileBusy;
    private bool _recordWhenDepthReady;
    private bool _coverageRecordingSession;
    private bool _recordingStarted;

    private void Awake()
    {
        Instance = this;
        _depth = GetComponent<DepthCapture>();
        _reconstruction = GetComponent<RoomSetupReplicaV2Reconstruction>();
        _depthPointCloudOverlay = GetComponent<DepthPointCloudOverlay>();
        if (_depthPointCloudOverlay != null)
        {
            _depthPointCloudOverlay.SetVisible(false);
            _depthPointCloudOverlay.SetAcquiring(false);
        }
        if (SynchronizedAtomicABCapture)
            CreateSynchronizedAtomicArm();
        SetLivePreviewState();
        Debug.Log(SynchronizedAtomicABCapture
            ? "[RoomSetupReplicaV2] synchronized live A/B enabled: " +
              $"A={_reconstruction.RouteLabel} " +
              $"B={_atomicReconstruction.RouteLabel} displayed=B"
            : "[RoomSetupReplicaV2] single live route=" +
              _reconstruction.RouteLabel);
    }

    private void Start()
    {
        if (startAutomatically)
            StartScanning();
    }

    private void Update()
    {
        if (_coverageRecordingSession && _recordingStarted &&
            _reconstruction != null && !_reconstruction.IsRecording)
        {
            StopCoverageAndRecording();
            return;
        }
        if (!IsScanning || !DepthCapture.DepthAvailable ||
            Time.unscaledTime < _nextCaptureTime || _depth == null ||
            _reconstruction == null)
            return;
        int platformFrame = _depth.CurrentPlatformFrame;
        if (platformFrame <= 0)
            return;
        if (platformFrame != _lastDisplayPlatformFrame &&
            _depth.PublishCurrentRawFrameGlobalsForDisplay())
        {
            _lastDisplayPlatformFrame = platformFrame;
            _depthPointCloudOverlay?.CaptureCurrentFrameForDisplay();
        }
        if (platformFrame == _lastSubmittedPlatformFrame)
            return;
        if (!AllArmsReadyForSynchronizedPair)
        {
            if (platformFrame != _lastBusyPlatformFrame)
            {
                _lastBusyPlatformFrame = platformFrame;
                _skippedPlatformFramesWhileBusy++;
            }
            return;
        }
        _nextCaptureTime = Time.unscaledTime + 1f / Mathf.Max(1f, captureHz);
        _lastSubmittedPlatformFrame = platformFrame;
        _lastBusyPlatformFrame = -1;
        // Deliberately do not call DepthCapture.PreprocessLatestFrame here.
        // ChiselSourceDepthTex falls back to the exact callback-latched raw
        // stereo depth, keeping ScanCover temporal/bilateral/edge filters out
        // of this replica scene.
        bool submittedA = _reconstruction.TryObserveDepthFrame(_depth);
        bool submittedB = _atomicReconstruction == null ||
                          _atomicReconstruction.TryObserveDepthFrame(_depth);
        if (submittedA && submittedB)
        {
            _synchronizedSubmittedPairCount++;
        }
        else
        {
            _oneArmSubmissionFailureCount++;
            Debug.LogError(
                "[RoomSetupReplicaV2] synchronized pair submission failed " +
                $"frame={platformFrame} A={submittedA} B={submittedB}; " +
                "the mismatch is retained in the session receipt.");
        }
        if (_recordWhenDepthReady && !_reconstruction.IsRecording)
        {
            if (_reconstruction.StartRecording())
            {
                _recordWhenDepthReady = false;
                _recordingStarted = true;
            }
            else
            {
                _recordWhenDepthReady = false;
                _coverageRecordingSession = false;
                PauseScanning();
                ClearReplica();
                Debug.LogError(
                    "[RoomSetupReplicaV2] coverage cancelled because its " +
                    "matching recording could not start");
            }
        }
    }

    public void StartScanning()
    {
        if (IsScanning)
            return;
        _reconstruction?.StopReplay();
        _reconstruction?.PrepareForLiveCapture();
        _atomicReconstruction?.StopReplay();
        _atomicReconstruction?.PrepareForLiveCapture();
        SetLivePreviewState();
        IsScanning = true;
        _lastSubmittedPlatformFrame = -1;
        _lastDisplayPlatformFrame = -1;
        _lastBusyPlatformFrame = -1;
        _synchronizedSubmittedPairCount = 0;
        _oneArmSubmissionFailureCount = 0;
        _skippedPlatformFramesWhileBusy = 0;
        _nextCaptureTime = 0f;
        _depth?.StartDepthCapture();
        _depthPointCloudOverlay?.SetAcquiring(
            _depthPointCloudOverlay.Visible);
        Debug.Log("[RoomSetupReplicaV2] capture started");
    }

    public void PauseScanning()
    {
        if (!IsScanning)
            return;
        IsScanning = false;
        _depthPointCloudOverlay?.SetAcquiring(false);
        _depth?.StopDepthCapture();
        Debug.Log("[RoomSetupReplicaV2] capture paused");
    }

    public void ToggleScanning()
    {
        if (IsScanning) PauseScanning();
        else StartScanning();
    }

    public void ToggleCoverageAndRecording()
    {
        if (IsScanning || _recordWhenDepthReady ||
            (_reconstruction != null && _reconstruction.IsRecording))
        {
            StopCoverageAndRecording();
            return;
        }

        _recordWhenDepthReady = true;
        _coverageRecordingSession = true;
        _recordingStarted = false;
        StartScanning();
        Debug.Log(
            "[RoomSetupReplicaV2] coverage + recording requested");
    }

    public void StopCoverageAndRecording()
    {
        bool exportMesh = _recordingStarted ||
                          (_reconstruction != null &&
                           _reconstruction.IsRecording);
        _recordWhenDepthReady = false;
        _coverageRecordingSession = false;
        _recordingStarted = false;
        _reconstruction?.StopRecording();
        PauseScanning();
        if (exportMesh)
            QueueLiveExport();
        Debug.Log("[RoomSetupReplicaV2] coverage + recording stopped");
    }

    /// <summary>
    /// Stops accepting new depth frames and keeps the current reconstructed
    /// front visible. Already-submitted reconstruction work is allowed to
    /// finish; nothing is cleared and replay is not started.
    /// </summary>
    public bool FreezeCurrentReplica()
    {
        bool active = IsScanning || _recordWhenDepthReady ||
                      _coverageRecordingSession || _recordingStarted ||
                      (_reconstruction != null &&
                       _reconstruction.IsRecording);
        if (!active)
            return false;

        StopCoverageAndRecording();
        Debug.Log(
            "[RoomSetupReplicaV2] current replica frozen; visible front retained");
        return true;
    }

    /// <summary>
    /// Toggles the classic v2.4 world-locked multicolour BB depth-point trail.
    /// </summary>
    public bool ToggleDepthPointCloudOverlay()
    {
        if (_depthPointCloudOverlay == null)
        {
            _depthPointCloudOverlay =
                gameObject.AddComponent<DepthPointCloudOverlay>();
            _depthPointCloudOverlay.SetVisible(true);
            _depthPointCloudOverlay.SetAcquiring(IsScanning);
            Debug.Log(
                "[RoomSetupReplicaV2] BB depth-point trail created and visible");
            return true;
        }

        bool visible = !_depthPointCloudOverlay.Visible;
        _depthPointCloudOverlay.SetVisible(visible);
        _depthPointCloudOverlay.SetAcquiring(visible && IsScanning);
        Debug.Log(visible
            ? "[RoomSetupReplicaV2] BB depth-point trail visible"
            : "[RoomSetupReplicaV2] BB depth-point trail hidden");
        return visible;
    }

    private void QueueLiveExport()
    {
        string directory = Path.Combine(
            Application.persistentDataPath,
            "RoomSetupReplicaV2",
            SynchronizedAtomicABCapture ? "live-atomic-ab" : "live-single",
            DateTime.UtcNow.ToString("yyyyMMddTHHmmssfffZ"));
        Directory.CreateDirectory(directory);
        var session = new LiveAbSessionReceipt
        {
            schema = SynchronizedAtomicABCapture
                ? "roomsetup-replica-v2-live-atomic-ab-session-v1"
                : "roomsetup-replica-v2-live-single-session-v2",
            createdUtc = DateTime.UtcNow.ToString("O"),
            coordinateSpace = "QuestStage",
            captureMode = SynchronizedAtomicABCapture
                ? "synchronized-isolated-volumes"
                : "single-volume",
            inputContract = SynchronizedAtomicABCapture
                ? "A and B accept only the same callback-latched stereo platform frame while both arms are idle"
                : "one callback-latched stereo pair feeds one reconstruction",
            armA = _reconstruction != null
                ? _reconstruction.RouteLabel
                : "missing",
            armB = _atomicReconstruction != null
                ? _atomicReconstruction.RouteLabel
                : "not-enabled",
            displayedArm = _atomicReconstruction != null ? "B" : "A",
            synchronizedSubmittedPairs = _synchronizedSubmittedPairCount,
            oneArmSubmissionFailures = _oneArmSubmissionFailureCount,
            skippedPlatformFramesWhileBusy =
                _skippedPlatformFramesWhileBusy,
            acceptedPairsAWhenQueued = _reconstruction != null
                ? _reconstruction.AcceptedPairCount
                : 0,
            acceptedPairsBWhenQueued = _atomicReconstruction != null
                ? _atomicReconstruction.AcceptedPairCount
                : 0,
            completionContract =
                "Each arm writes its own mesh and receipt only after readback, integration, meshing, and simplification queues are empty."
        };
        File.WriteAllText(
            Path.Combine(directory, "live-session.json"),
            JsonUtility.ToJson(session, true) + "\n");
        _reconstruction?.RequestLiveExport(
            directory, "mesh-a-stable-baseline.ply");
        _atomicReconstruction?.RequestLiveExport(
            directory, "mesh-b-atomic-replacement.ply");
        Debug.Log("[RoomSetupReplicaV2] live session queued: " + directory);
    }

    public void ClearReplica()
    {
        _reconstruction?.ClearReplica();
        _atomicReconstruction?.ClearReplica();
        _depthPointCloudOverlay?.ClearTrail();
        _lastSubmittedPlatformFrame = -1;
        _lastDisplayPlatformFrame = -1;
        _lastBusyPlatformFrame = -1;
    }

    public bool ToggleRecording()
    {
        return _reconstruction != null &&
               _reconstruction.ToggleRecording();
    }

    public bool ReplayLastTape()
    {
        _recordWhenDepthReady = false;
        _coverageRecordingSession = false;
        _recordingStarted = false;
        PauseScanning();
        RoomSetupReplicaV2Reconstruction replay = Reconstruction;
        if (_reconstruction != null)
            _reconstruction.SetPreviewVisible(replay == _reconstruction);
        if (_atomicReconstruction != null)
            _atomicReconstruction.SetPreviewVisible(
                replay == _atomicReconstruction);
        return replay != null && replay.BeginReplayLast();
    }

    private bool AllArmsReadyForSynchronizedPair =>
        _reconstruction != null &&
        _reconstruction.ReadyForSynchronizedDepthPair &&
        (_atomicReconstruction == null ||
         _atomicReconstruction.ReadyForSynchronizedDepthPair);

    private void CreateSynchronizedAtomicArm()
    {
        var armObject = new GameObject(
            "Room Setup Replica V2 - B Atomic Replacement");
        armObject.transform.SetParent(transform, false);
        _atomicReconstruction =
            armObject.AddComponent<RoomSetupReplicaV2Reconstruction>();
        _atomicReconstruction.CopyRuntimeConfigurationFrom(_reconstruction);
        _atomicReconstruction.ConfigureAsExperimentalB();
    }

    private void SetLivePreviewState()
    {
        bool hasAtomicArm = _atomicReconstruction != null;
        _reconstruction?.SetPreviewVisible(!hasAtomicArm);
        _atomicReconstruction?.SetPreviewVisible(true);
    }

    private void OnDestroy()
    {
        if (Instance == this)
            Instance = null;
        if (IsScanning)
        {
            _depthPointCloudOverlay?.SetAcquiring(false);
            _depth?.StopDepthCapture();
        }
    }
}
