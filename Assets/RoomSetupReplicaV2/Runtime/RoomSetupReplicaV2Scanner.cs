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
    typeof(PassthroughCameraProvider),
    typeof(RoomSetupReplicaV2Reconstruction))]
public sealed class RoomSetupReplicaV2Scanner : MonoBehaviour
{
    public static RoomSetupReplicaV2Scanner Instance { get; private set; }

    [SerializeField, Range(1f, 30f)] private float captureHz = 10f;
    [SerializeField] private bool startAutomatically = false;

    public bool IsScanning { get; private set; }
    public RoomSetupReplicaV2Reconstruction Reconstruction => _reconstruction;

    private DepthCapture _depth;
    private PassthroughCameraProvider _cameraProvider;
    private RoomSetupReplicaV2Reconstruction _reconstruction;
    private float _nextCaptureTime;
    private int _lastSubmittedPlatformFrame = -1;
    private bool _recordWhenDepthReady;
    private bool _coverageRecordingSession;
    private bool _recordingStarted;

    private void Awake()
    {
        Instance = this;
        _depth = GetComponent<DepthCapture>();
        _cameraProvider = GetComponent<PassthroughCameraProvider>();
        _reconstruction = GetComponent<RoomSetupReplicaV2Reconstruction>();
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
        if (platformFrame <= 0 || platformFrame == _lastSubmittedPlatformFrame)
            return;
        _nextCaptureTime = Time.unscaledTime + 1f / Mathf.Max(1f, captureHz);
        _lastSubmittedPlatformFrame = platformFrame;
        // Deliberately do not call DepthCapture.PreprocessLatestFrame here.
        // ChiselSourceDepthTex falls back to the exact callback-latched raw
        // stereo depth, keeping ScanCover temporal/bilateral/edge filters out
        // of this replica scene.
        _reconstruction.ObserveDepthFrame(_depth);
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
                _reconstruction.ClearReplica();
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
        IsScanning = true;
        _lastSubmittedPlatformFrame = -1;
        _nextCaptureTime = 0f;
        _cameraProvider?.StartCapture();
        _depth?.StartDepthCapture();
        Debug.Log("[RoomSetupReplicaV2] capture started");
    }

    public void PauseScanning()
    {
        if (!IsScanning)
            return;
        IsScanning = false;
        _cameraProvider?.StopCapture();
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
        _recordWhenDepthReady = false;
        _coverageRecordingSession = false;
        _recordingStarted = false;
        _reconstruction?.StopRecording();
        PauseScanning();
        Debug.Log("[RoomSetupReplicaV2] coverage + recording stopped");
    }

    public void ClearReplica()
    {
        _reconstruction?.ClearReplica();
        _lastSubmittedPlatformFrame = -1;
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
        return _reconstruction != null &&
               _reconstruction.BeginReplayLast();
    }

    private void OnDestroy()
    {
        if (Instance == this)
            Instance = null;
        if (IsScanning)
        {
            _cameraProvider?.StopCapture();
            _depth?.StopDepthCapture();
        }
    }
}
