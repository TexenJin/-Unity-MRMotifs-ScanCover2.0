using MRMotifs.InstantContentPlacement.DepthEffects;
using System.Collections;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem.XR;
using UnityEngine.Serialization;

[DisallowMultipleComponent]
public sealed class ScanCoverSkeletonSessionController : MonoBehaviour
{
    public enum ControllerButton
    {
        None,
        PrimaryButton,
        SecondaryButton,
        Trigger,
        Grip,
        ThumbstickClick,
        Menu
    }

    public enum SessionState
    {
        Idle = 0,
        Scanning = 1,
        Frozen = 2,
    }

    [Header("Refs")]
    public ScanCoverSkeletonRuntimeCore runtimeCore;
    public ScanCoverSurfaceSnapshotManager surfaceSnapshotManager;
    public ScanCoverDepthGridPointCloud depthGridPointCloud;
    public DepthEffectsDepthProbeRowRenderer[] depthProbeRows;

    [Header("Startup")]
    public bool autoStartOnEnable = true;
    public bool clearOnAutoStart = true;

    [Header("Input (XR Controller / OVRInput)")]
    public bool preferRightHandController = true;
    public bool useOvrInput = true;
    public bool useXrControllerInput = false;
    public bool useGamepadFallback = false;
    [InspectorName("Start Scan Button (Unused)")]
    public ControllerButton startScanButton = ControllerButton.None;
    [InspectorName("Capture Snapshot Button (Unused)")]
    public ControllerButton captureSnapshotButton = ControllerButton.None;
    [InspectorName("Toggle Snapshot Freeze Button (Unused)")]
    public ControllerButton toggleSnapshotFreezeButton = ControllerButton.None;
    [InspectorName("Freeze / Capture Snapshot Button")]
    public ControllerButton freezeBuildButton = ControllerButton.Trigger;
    [InspectorName("Clear All Runtime State Button")]
    public ControllerButton clearAllButton = ControllerButton.ThumbstickClick;
    [InspectorName("Deprecated Preview Visibility Button")]
    [FormerlySerializedAs("toggleRenderButton")]
    public ControllerButton deprecatedPreviewVisibilityButton = ControllerButton.None;
    [InspectorName("Allow Deprecated Preview Visibility Button")]
    public bool allowDeprecatedPreviewToggleButton = false;

    [Header("Behavior")]
    public bool startScanClearsExisting = false;
    public bool captureSurfaceSnapshotOnFreeze = false;
    public bool freezeProbeRowsAfterSnapshotCapture = true;
    public bool hideDepthPreviewOnClearAll = false;
    public bool forceClearSurfaceSnapshotsOnClearAll = true;
    public bool enableLegacyVisualChain = false;
    public bool debugLog = false;
    [Min(0f)]
    public float inputDebounceSeconds = 0.35f;
    public bool ignoreInputWhileCaptureBurstRunning = true;

    [Header("Capture Burst")]
    public bool useCaptureBurst = true;
    [Min(1)] public int captureBurstRefreshCount = 1;
    [Min(0.05f)] public float captureBurstTimeoutSeconds = 0.6f;
    public bool captureSurfaceSnapshotOnManualCapture = false;
    public bool stopRealtimeAfterCaptureBurst = true;

    [Header("Optional Export")]
    public bool exportSnapshotObjOnFreeze = false;
    public bool exportAllSnapshotsAsSingleObjOnFreeze = false;
    public bool exportIncrementalReferenceShellOnFreeze = false;
    public bool exportGridStateJsonOnFreeze = false;

    public SessionState State { get; private set; } = SessionState.Idle;
    public bool HasCurrentTile { get; private set; }
    public ScanCoverSkeletonMesher_B.ChunkKey CurrentTile { get; private set; }
    public int CurrentTileTotalCells { get; private set; }
    public int CurrentTileConfirmedCells { get; private set; }
    public int CurrentTileRecentConfirmedCells { get; private set; }
    public float CurrentTileRecentGrowthRatio { get; private set; }
    public bool CurrentTileReady { get; private set; }
    public int StableUncommittedTileCount { get; private set; }
    public int CommittedTileCount { get; private set; }
    public ScanCoverSkeletonBuilder_A.SummaryStats LastSummary { get; private set; }

    private bool _autoStarted;
    private Coroutine _captureBurstRoutine;
    private float _nextAcceptedInputTime;

    private void Awake()
    {
        ResolveRefs();
    }

    private void OnEnable()
    {
        ResolveRefs();
        if (autoStartOnEnable && !_autoStarted)
        {
            _autoStarted = true;
            StartNewScan(clearOnAutoStart);
        }
    }

    private void Update()
    {
        ConsumeOneInputAction();
    }

    public string GetStateLabel()
    {
        if (runtimeCore == null)
            return State.ToString();

        return $"{State} grid={runtimeCore.VisibleGridCount} snapshots={runtimeCore.SnapshotCount}";
    }

    [ContextMenu("Start New Scan")]
    public void StartNewScan()
    {
        StartNewScan(startScanClearsExisting);
    }

    public void StartNewScan(bool clearExisting)
    {
        ResolveRefs();
        if (runtimeCore == null)
            return;

        runtimeCore.BeginNewSession(clearExisting);
        ClearHudStats();
        State = SessionState.Scanning;
    }

    [ContextMenu("Capture Snapshot Now")]
    public void CaptureSnapshotNow()
    {
        ResolveRefs();
        if (runtimeCore == null)
            return;

        if (useCaptureBurst && isActiveAndEnabled)
        {
            StartCaptureBurst(captureSurfaceSnapshotOnManualCapture, freezeAfterCapture: false, runFreezeExports: false);
            return;
        }

        runtimeCore.CaptureProbeRowSnapshot(freezeProbeRowsAfterSnapshotCapture);
        State = SessionState.Scanning;
    }

    [ContextMenu("Toggle Probe Row Snapshot Freeze")]
    public void ToggleSnapshotFreeze()
    {
        ResolveRefs();
        runtimeCore?.ToggleProbeRowSnapshotFreeze();
    }

    [ContextMenu("Freeze And Build")]
    public void FreezeAndBuild()
    {
        ResolveRefs();
        if (runtimeCore == null)
            return;

        runtimeCore.RestoreMainlineVisibility();
        if (useCaptureBurst && isActiveAndEnabled)
        {
            StartCaptureBurst(captureSurfaceSnapshotOnFreeze, freezeAfterCapture: true, runFreezeExports: true);
            return;
        }

        runtimeCore.CaptureProbeRowSnapshot(freezeProbeRowsAfterSnapshotCapture);

        if (captureSurfaceSnapshotOnFreeze)
            runtimeCore.CaptureSurfaceSnapshot();

        if (surfaceSnapshotManager != null)
        {
            if (exportAllSnapshotsAsSingleObjOnFreeze)
                surfaceSnapshotManager.ExportAllSnapshotsAsSingleObj();
            else if (exportSnapshotObjOnFreeze)
                surfaceSnapshotManager.ExportLatestSnapshotAsObj();

            if (exportIncrementalReferenceShellOnFreeze)
                surfaceSnapshotManager.ExportIncrementalReferenceShellAsObj();
        }

        if (exportGridStateJsonOnFreeze && depthGridPointCloud != null)
            depthGridPointCloud.ExportCurrentGridStateAsJson(out _);

        runtimeCore.FreezeSession();
        State = SessionState.Frozen;
    }

    [ContextMenu("Clear All Runtime State")]
    public void ClearAll()
    {
        ResolveRefs();
        if (_captureBurstRoutine != null)
        {
            StopCoroutine(_captureBurstRoutine);
            _captureBurstRoutine = null;
        }

        runtimeCore?.ClearAll(hideDepthPreviewOnClearAll, forceClearSurfaceSnapshotsOnClearAll);
        ClearHudStats();
        State = SessionState.Idle;
    }

    [ContextMenu("Toggle Preview Visible")]
    public void TogglePreview()
    {
        ResolveRefs();
        runtimeCore?.TogglePreviewVisible();
    }

    private void StartCaptureBurst(bool captureSurfaceSnapshot, bool freezeAfterCapture, bool runFreezeExports)
    {
        if (_captureBurstRoutine != null)
            StopCoroutine(_captureBurstRoutine);

        _captureBurstRoutine = StartCoroutine(CaptureBurstRoutine(
            captureSurfaceSnapshot,
            freezeAfterCapture,
            runFreezeExports));
    }

    private IEnumerator CaptureBurstRoutine(bool captureSurfaceSnapshot, bool freezeAfterCapture, bool runFreezeExports)
    {
        ResolveRefs();
        if (runtimeCore == null)
        {
            _captureBurstRoutine = null;
            yield break;
        }

        State = SessionState.Scanning;
        runtimeCore.SetRealtimeDepthConsumption(true);
        runtimeCore.RestoreMainlineVisibility();

        float depthDeadline = Time.unscaledTime + Mathf.Max(0.05f, captureBurstTimeoutSeconds);
        while (!runtimeCore.IsRealtimeDepthAvailable && Time.unscaledTime < depthDeadline)
            yield return null;

        int startFrame = depthGridPointCloud != null ? depthGridPointCloud.FrameIndex : -1;
        bool requestedAnyRefresh = false;

        int refreshCount = Mathf.Max(1, captureBurstRefreshCount);
        for (int i = 0; i < refreshCount; i++)
        {
            requestedAnyRefresh |= runtimeCore.RequestDepthGridBurstRefresh();
            float deadline = Time.unscaledTime + Mathf.Max(0.05f, captureBurstTimeoutSeconds);

            while (depthGridPointCloud != null &&
                   depthGridPointCloud.HasPendingReadback &&
                   Time.unscaledTime < deadline)
            {
                yield return null;
            }

            if (depthGridPointCloud == null ||
                !depthGridPointCloud.HasPendingReadback ||
                depthGridPointCloud.FrameIndex != startFrame)
            {
                break;
            }

            yield return null;
        }

        if (!requestedAnyRefresh && debugLog)
            Debug.LogWarning($"[ScanCoverSkeletonSessionController] Capture burst did not request a depth refresh: {runtimeCore.LastIssue}");

        runtimeCore.RestoreMainlineVisibility();
        runtimeCore.CaptureProbeRowSnapshot(freezeProbeRowsAfterSnapshotCapture);
        yield return null;

        if (captureSurfaceSnapshot)
            runtimeCore.CaptureSurfaceSnapshot();

        if (runFreezeExports)
            RunFreezeExports();

        if (exportGridStateJsonOnFreeze && runFreezeExports && depthGridPointCloud != null)
            depthGridPointCloud.ExportCurrentGridStateAsJson(out _);

        if (freezeAfterCapture)
        {
            runtimeCore.FreezeSession();
            State = SessionState.Frozen;
        }
        else
        {
            State = SessionState.Scanning;
        }

        if (stopRealtimeAfterCaptureBurst)
            runtimeCore.SetRealtimeDepthConsumption(false);

        _captureBurstRoutine = null;
    }

    private void RunFreezeExports()
    {
        if (surfaceSnapshotManager == null)
            return;

        if (exportAllSnapshotsAsSingleObjOnFreeze)
            surfaceSnapshotManager.ExportAllSnapshotsAsSingleObj();
        else if (exportSnapshotObjOnFreeze)
            surfaceSnapshotManager.ExportLatestSnapshotAsObj();

        if (exportIncrementalReferenceShellOnFreeze)
            surfaceSnapshotManager.ExportIncrementalReferenceShellAsObj();
    }

    private bool ConsumeOneInputAction()
    {
        if (Time.unscaledTime < _nextAcceptedInputTime)
            return false;
        if (ignoreInputWhileCaptureBurstRunning && _captureBurstRoutine != null)
            return false;

        if (GetActionDown(clearAllButton))
        {
            if (debugLog) Debug.Log("[ScanCoverSkeletonSessionController] ClearAll input");
            MarkInputConsumed();
            ClearAll();
            return true;
        }

        if (allowDeprecatedPreviewToggleButton && GetActionDown(deprecatedPreviewVisibilityButton))
        {
            if (debugLog) Debug.Log("[ScanCoverSkeletonSessionController] TogglePreview input");
            MarkInputConsumed();
            TogglePreview();
            return true;
        }

        if (GetActionDown(freezeBuildButton))
        {
            if (debugLog) Debug.Log("[ScanCoverSkeletonSessionController] FreezeAndBuild input");
            MarkInputConsumed();
            FreezeAndBuild();
            return true;
        }

        if (GetActionDown(captureSnapshotButton))
        {
            if (debugLog) Debug.Log("[ScanCoverSkeletonSessionController] CaptureSnapshotNow input");
            MarkInputConsumed();
            CaptureSnapshotNow();
            return true;
        }

        if (GetActionDown(toggleSnapshotFreezeButton))
        {
            if (debugLog) Debug.Log("[ScanCoverSkeletonSessionController] ToggleSnapshotFreeze input");
            MarkInputConsumed();
            ToggleSnapshotFreeze();
            return true;
        }

        if (GetActionDown(startScanButton))
        {
            if (debugLog) Debug.Log("[ScanCoverSkeletonSessionController] StartNewScan input");
            MarkInputConsumed();
            StartNewScan(startScanClearsExisting);
            return true;
        }

        return false;
    }

    private void MarkInputConsumed()
    {
        _nextAcceptedInputTime = Time.unscaledTime + Mathf.Max(0f, inputDebounceSeconds);
    }

    private void ResolveRefs()
    {
        if (runtimeCore == null)
            runtimeCore = GetComponent<ScanCoverSkeletonRuntimeCore>();
        if (runtimeCore == null)
            runtimeCore = gameObject.AddComponent<ScanCoverSkeletonRuntimeCore>();

        runtimeCore.ResolveRefs();

        if (surfaceSnapshotManager == null)
            surfaceSnapshotManager = runtimeCore.surfaceSnapshotManager;
        if (depthGridPointCloud == null)
            depthGridPointCloud = runtimeCore.depthGridPointCloud;
        if (depthProbeRows == null || depthProbeRows.Length == 0)
            depthProbeRows = runtimeCore.depthProbeRows;
    }

    private void ClearHudStats()
    {
        HasCurrentTile = false;
        CurrentTile = default;
        CurrentTileTotalCells = 0;
        CurrentTileConfirmedCells = 0;
        CurrentTileRecentConfirmedCells = 0;
        CurrentTileRecentGrowthRatio = 0f;
        CurrentTileReady = false;
        StableUncommittedTileCount = 0;
        CommittedTileCount = 0;
        LastSummary = default;
    }

    private bool GetActionDown(ControllerButton button)
    {
        if (button == ControllerButton.None)
            return false;

        if (useOvrInput && GetOvrButtonDown(button))
            return true;

        if (useXrControllerInput)
        {
            XRController xr = GetPreferredXRController();
            ButtonControl xrButton = ResolveXRButton(xr, button);
            if (xrButton != null && xrButton.wasPressedThisFrame)
                return true;
        }

        return useGamepadFallback && GetGamepadButtonDown(button);
    }

    private XRController GetPreferredXRController()
    {
        XRController fallback = null;

        foreach (InputDevice device in InputSystem.devices)
        {
            if (device is not XRController xr || !device.enabled)
                continue;

            fallback ??= xr;

            bool isRight = HasUsage(device, "RightHand");
            bool isLeft = HasUsage(device, "LeftHand");
            if (preferRightHandController && isRight)
                return xr;
            if (!preferRightHandController && isLeft)
                return xr;
        }

        return fallback;
    }

    private static bool HasUsage(InputDevice device, string usageName)
    {
        for (int i = 0; i < device.usages.Count; i++)
        {
            if (device.usages[i].ToString() == usageName)
                return true;
        }
        return false;
    }

    private static ButtonControl ResolveXRButton(XRController controller, ControllerButton button)
    {
        if (controller == null)
            return null;

        string controlName = button switch
        {
            ControllerButton.PrimaryButton => "primaryButton",
            ControllerButton.SecondaryButton => "secondaryButton",
            ControllerButton.Trigger => "triggerPressed",
            ControllerButton.Grip => "gripPressed",
            ControllerButton.ThumbstickClick => "primary2DAxisClick",
            ControllerButton.Menu => "menuButton",
            _ => null
        };

        return string.IsNullOrEmpty(controlName) ? null : controller.TryGetChildControl<ButtonControl>(controlName);
    }

    private bool GetOvrButtonDown(ControllerButton button)
    {
        return TryResolveOvrRawButton(button, preferRightHandController, out OVRInput.RawButton rawButton)
            && OVRInput.GetDown(rawButton);
    }

    private static bool TryResolveOvrRawButton(ControllerButton button, bool preferRightHand, out OVRInput.RawButton rawButton)
    {
        rawButton = OVRInput.RawButton.None;
        switch (button)
        {
            case ControllerButton.PrimaryButton:
                rawButton = preferRightHand ? OVRInput.RawButton.A : OVRInput.RawButton.X;
                return true;
            case ControllerButton.SecondaryButton:
                rawButton = preferRightHand ? OVRInput.RawButton.B : OVRInput.RawButton.Y;
                return true;
            case ControllerButton.Trigger:
                rawButton = preferRightHand ? OVRInput.RawButton.RIndexTrigger : OVRInput.RawButton.LIndexTrigger;
                return true;
            case ControllerButton.Grip:
                rawButton = preferRightHand ? OVRInput.RawButton.RHandTrigger : OVRInput.RawButton.LHandTrigger;
                return true;
            case ControllerButton.ThumbstickClick:
                rawButton = preferRightHand ? OVRInput.RawButton.RThumbstick : OVRInput.RawButton.LThumbstick;
                return true;
            case ControllerButton.Menu:
                rawButton = OVRInput.RawButton.Start;
                return true;
            default:
                return false;
        }
    }

    private static bool GetGamepadButtonDown(ControllerButton button)
    {
        Gamepad pad = Gamepad.current;
        if (pad == null)
            return false;

        return button switch
        {
            ControllerButton.PrimaryButton => pad.buttonSouth.wasPressedThisFrame,
            ControllerButton.SecondaryButton => pad.buttonEast.wasPressedThisFrame,
            ControllerButton.Trigger => pad.rightTrigger.wasPressedThisFrame,
            ControllerButton.Grip => pad.leftTrigger.wasPressedThisFrame,
            ControllerButton.ThumbstickClick => pad.rightStickButton.wasPressedThisFrame,
            ControllerButton.Menu => pad.startButton.wasPressedThisFrame,
            _ => false
        };
    }
}
