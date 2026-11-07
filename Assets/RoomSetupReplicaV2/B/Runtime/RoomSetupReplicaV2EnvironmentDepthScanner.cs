using Genesis.RoomScan;
using Meta.XR.EnvironmentDepth;
using UnityEngine;

/// <summary>
/// Sequential-B capture loop. EnvironmentDepthManager is the sole owner of
/// depth acquisition in this scene; no DepthCapture/AROcclusionManager chain
/// may coexist with it.
/// </summary>
[DefaultExecutionOrder(-20)]
[DisallowMultipleComponent]
[RequireComponent(
    typeof(EnvironmentDepthManager),
    typeof(RoomSetupReplicaV2Reconstruction))]
public sealed class RoomSetupReplicaV2EnvironmentDepthScanner : MonoBehaviour
{
    public static RoomSetupReplicaV2EnvironmentDepthScanner Instance
        { get; private set; }

    [SerializeField, Range(1f, 30f)] private float captureHz = 10f;
    [SerializeField] private bool startAutomatically = false;
    [SerializeField] private Camera xrCamera;

    public bool IsScanning { get; private set; }
    public RoomSetupReplicaV2Reconstruction Reconstruction =>
        _reconstruction;

    private EnvironmentDepthManager _environmentDepth;
    private RoomSetupReplicaV2Reconstruction _reconstruction;
    private float _nextCaptureTime;
    private bool _routeLogged;

    private void Awake()
    {
        Instance = this;
        _environmentDepth = GetComponent<EnvironmentDepthManager>();
        _reconstruction = GetComponent<RoomSetupReplicaV2Reconstruction>();
        _reconstruction.ConfigureAsEnvironmentDepthGlobalSource();

        _environmentDepth.OcclusionShadersMode =
            OcclusionShadersMode.None;
        _environmentDepth.RemoveHands = false;
        _environmentDepth.MaskMeshFilters.Clear();
        // Acquisition starts only when the user pulls the trigger.
        _environmentDepth.enabled = false;

        if (FindObjectsByType<DepthCapture>(
                FindObjectsInactive.Include,
                FindObjectsSortMode.None).Length > 0)
        {
            Debug.LogError(
                "[RoomSetupReplicaV2 B] DepthCapture is present. " +
                "Sequential B requires exclusive EnvironmentDepthManager " +
                "ownership; rebuild the B scene before testing.");
            enabled = false;
        }
    }

    private void Start()
    {
        ResolveCamera();
        if (startAutomatically)
            StartScanning();
    }

    private void Update()
    {
        if (!IsScanning || _environmentDepth == null ||
            !_environmentDepth.IsDepthAvailable ||
            Time.unscaledTime < _nextCaptureTime ||
            _reconstruction == null)
            return;
        if (xrCamera == null)
            ResolveCamera();
        if (xrCamera == null)
            return;

        _nextCaptureTime = Time.unscaledTime +
                           1f / Mathf.Max(1f, captureHz);
        if (!_reconstruction.TryObserveEnvironmentDepthGlobal(xrCamera))
            return;
        if (_routeLogged)
            return;
        _routeLogged = true;
        Debug.Log(
            "[RoomSetupReplicaV2 B] first global Environment Depth frame " +
            "accepted; route=" + _reconstruction.RouteLabel);
    }

    public void StartScanning()
    {
        if (IsScanning || !enabled)
            return;
        if (!EnvironmentDepthManager.IsSupported)
        {
            Debug.LogError(
                "[RoomSetupReplicaV2 B] Environment Depth is not supported " +
                "on this runtime/provider.");
            return;
        }
        _reconstruction.StopReplay();
        _reconstruction.PrepareForLiveCapture();
        ResolveCamera();
        _nextCaptureTime = 0f;
        _environmentDepth.enabled = true;
        IsScanning = true;
        Debug.Log(
            "[RoomSetupReplicaV2 B] capture started; source=" +
            "_EnvironmentDepthTexture, hardRange=3.00m, masks=off");
    }

    public void PauseScanning()
    {
        if (!IsScanning)
            return;
        IsScanning = false;
        if (_environmentDepth != null)
            _environmentDepth.enabled = false;
        Debug.Log("[RoomSetupReplicaV2 B] capture paused");
    }

    public void ToggleScanning()
    {
        if (IsScanning) PauseScanning();
        else StartScanning();
    }

    public void ClearReplica()
    {
        _reconstruction?.ClearReplica();
    }

    public void ToggleReplicaPreview()
    {
        if (_reconstruction != null)
            _reconstruction.SetPreviewVisible(
                !_reconstruction.PreviewVisible);
    }

    public bool ToggleUniqueEdgeDisplay()
    {
        return _reconstruction != null &&
               _reconstruction.ToggleUniqueEdgeDisplay();
    }

    public bool FreezeCurrentReplica()
    {
        if (!IsScanning)
            return false;
        PauseScanning();
        return true;
    }

    private void ResolveCamera()
    {
        if (xrCamera != null)
            return;
        OVRCameraRig rig = FindAnyObjectByType<OVRCameraRig>();
        if (rig != null && rig.centerEyeAnchor != null)
            xrCamera = rig.centerEyeAnchor.GetComponent<Camera>();
        if (xrCamera == null)
            xrCamera = Camera.main;
        if (xrCamera == null)
            xrCamera = FindAnyObjectByType<Camera>();
    }

    private void OnDestroy()
    {
        if (Instance == this)
            Instance = null;
    }
}
