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
    typeof(RoomSetupReplicaReconstruction))]
public sealed class RoomSetupReplicaScanner : MonoBehaviour
{
    public static RoomSetupReplicaScanner Instance { get; private set; }

    [SerializeField, Range(1f, 30f)] private float captureHz = 10f;
    [SerializeField] private bool startAutomatically = true;

    public bool IsScanning { get; private set; }
    public RoomSetupReplicaReconstruction Reconstruction => _reconstruction;

    private DepthCapture _depth;
    private PassthroughCameraProvider _cameraProvider;
    private RoomSetupReplicaReconstruction _reconstruction;
    private float _nextCaptureTime;
    private int _lastSubmittedPlatformFrame = -1;

    private void Awake()
    {
        Instance = this;
        _depth = GetComponent<DepthCapture>();
        _cameraProvider = GetComponent<PassthroughCameraProvider>();
        _reconstruction = GetComponent<RoomSetupReplicaReconstruction>();
    }

    private void Start()
    {
        if (startAutomatically)
            StartScanning();
    }

    private void Update()
    {
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
    }

    public void StartScanning()
    {
        if (IsScanning)
            return;
        IsScanning = true;
        _lastSubmittedPlatformFrame = -1;
        _nextCaptureTime = 0f;
        _cameraProvider?.StartCapture();
        _depth?.StartDepthCapture();
        Debug.Log("[RoomSetupReplica] capture started");
    }

    public void PauseScanning()
    {
        if (!IsScanning)
            return;
        IsScanning = false;
        _cameraProvider?.StopCapture();
        _depth?.StopDepthCapture();
        Debug.Log("[RoomSetupReplica] capture paused");
    }

    public void ToggleScanning()
    {
        if (IsScanning) PauseScanning();
        else StartScanning();
    }

    public void ClearReplica()
    {
        _reconstruction?.ClearReplica();
        _lastSubmittedPlatformFrame = -1;
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
