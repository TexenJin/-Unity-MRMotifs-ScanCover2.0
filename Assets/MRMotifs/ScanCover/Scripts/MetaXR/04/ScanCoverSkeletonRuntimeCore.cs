using MRMotifs.InstantContentPlacement.DepthEffects;
using Meta.XR.EnvironmentDepth;
using UnityEngine;

[DisallowMultipleComponent]
public sealed class ScanCoverSkeletonRuntimeCore : MonoBehaviour
{
    [Header("Refs")]
    public Transform referenceFrame;
    public EnvironmentDepthManager environmentDepthManager;
    public ScanCoverDepthGridPointCloud depthGridPointCloud;
    public ScanCoverSurfaceSnapshotManager surfaceSnapshotManager;
    public DepthEffectsDepthProbeRowRenderer[] depthProbeRows;

    [Header("Runtime")]
    public bool scanEnabled = true;
    public bool showPreviewOnStart = true;
    public bool freezeRowsAfterManualCapture = true;
    public bool disableEnvironmentDepthWhileIdle = false;
    public bool debugLog;

    public bool IsFrozen { get; private set; }
    public bool PreviewVisible => depthGridPointCloud == null || depthGridPointCloud.PreviewVisible;
    public bool IsRealtimeDepthAvailable => environmentDepthManager == null || environmentDepthManager.IsDepthAvailable;
    public int VisibleGridCount => depthGridPointCloud != null ? depthGridPointCloud.VisibleCount : 0;
    public int SnapshotCount => surfaceSnapshotManager != null ? surfaceSnapshotManager.SnapshotCount : 0;
    public string LastIssue { get; private set; }

    private bool _resolved;

    private void Awake()
    {
        ResolveRefs();
    }

    private void OnEnable()
    {
        ResolveRefs();
        if (depthGridPointCloud != null)
            depthGridPointCloud.SetPreviewVisible(showPreviewOnStart);
    }

    public void ResolveRefs()
    {
        if (referenceFrame == null)
            referenceFrame = transform;

        if (depthGridPointCloud == null)
            depthGridPointCloud = GetComponentInChildren<ScanCoverDepthGridPointCloud>(true);
        if (depthGridPointCloud == null && referenceFrame != null)
            depthGridPointCloud = referenceFrame.GetComponentInChildren<ScanCoverDepthGridPointCloud>(true);

        if (environmentDepthManager == null)
            environmentDepthManager = FindAnyObjectByType<EnvironmentDepthManager>(FindObjectsInactive.Include);

        if (surfaceSnapshotManager == null)
            surfaceSnapshotManager = GetComponent<ScanCoverSurfaceSnapshotManager>();
        if (surfaceSnapshotManager == null)
            surfaceSnapshotManager = gameObject.AddComponent<ScanCoverSurfaceSnapshotManager>();

        ResolveDepthProbeRows();

        if (surfaceSnapshotManager != null)
        {
            if (depthGridPointCloud != null)
                surfaceSnapshotManager.captureRoot = depthGridPointCloud.transform;
            else if (surfaceSnapshotManager.captureRoot == null)
                surfaceSnapshotManager.captureRoot = referenceFrame != null ? referenceFrame : transform;
            surfaceSnapshotManager.depthGridPointCloud = depthGridPointCloud;
            surfaceSnapshotManager.excludeMarkerLikeObjects = false;
            surfaceSnapshotManager.maintainIncrementalReferenceShell = false;
            surfaceSnapshotManager.integrateLatestSnapshotIntoReferenceShellOnCapture = false;
            surfaceSnapshotManager.generateRegularTriangleSkinOnCapture = false;
            surfaceSnapshotManager.hideCapturedSurfaceMeshWhenSkinGenerated = false;
            surfaceSnapshotManager.EnsureInitialized();
        }

        _resolved = true;
        LastIssue = null;
    }

    public void BeginNewSession(bool clearExisting)
    {
        ResolveRefs();
        if (clearExisting)
            ClearAll(hidePreview: false, clearSurfaceSnapshots: true);

        scanEnabled = true;
        IsFrozen = false;
        SetRealtimeDepthConsumption(false);
        SetPreviewVisible(true);
        SetProbeRowSnapshotFrozen(false);
        LastIssue = null;
        if (debugLog)
            Debug.Log("[ScanCoverSkeletonRuntimeCore] BeginNewSession");
    }

    public void FreezeSession()
    {
        ResolveRefs();
        IsFrozen = true;
        scanEnabled = false;
        SetRealtimeDepthConsumption(false);
        RestoreMainlineVisibility();
        SetProbeRowSnapshotFrozen(true);
        if (debugLog)
            Debug.Log("[ScanCoverSkeletonRuntimeCore] FreezeSession");
    }

    public void ClearAll(bool hidePreview)
    {
        ClearAll(hidePreview, clearSurfaceSnapshots: true);
    }

    public void ClearAll(bool hidePreview, bool clearSurfaceSnapshots)
    {
        ResolveRefs();
        scanEnabled = false;
        IsFrozen = false;

        if (clearSurfaceSnapshots && surfaceSnapshotManager != null)
            surfaceSnapshotManager.ClearAll();

        if (depthGridPointCloud != null)
            depthGridPointCloud.ClearRuntimeState(hidePreview);
        SetRealtimeDepthConsumption(false);

        ResolveDepthProbeRows();
        for (int i = 0; i < depthProbeRows.Length; i++)
        {
            if (depthProbeRows[i] == null)
                continue;
            depthProbeRows[i].ClearRuntimeState(requestFreshSnapshot: false);
            depthProbeRows[i].SetSnapshotUpdatesFrozen(true);
        }

        LastIssue = null;
        if (debugLog)
            Debug.Log("[ScanCoverSkeletonRuntimeCore] ClearAll");
    }

    public int CaptureSurfaceSnapshot()
    {
        ResolveRefs();
        RestoreMainlineVisibility();
        if (surfaceSnapshotManager == null)
        {
            LastIssue = "Surface snapshot manager is missing.";
            return 0;
        }

        int captured = surfaceSnapshotManager.CaptureVisibleSurfaces();
        LastIssue = surfaceSnapshotManager.LastIssue;
        if (debugLog)
            Debug.Log($"[ScanCoverSkeletonRuntimeCore] CaptureSurfaceSnapshot captured={captured}");
        return captured;
    }

    public void CaptureProbeRowSnapshot(bool freezeAfterCapture)
    {
        ResolveRefs();
        RestoreMainlineVisibility();
        ResolveDepthProbeRows();
        for (int i = 0; i < depthProbeRows.Length; i++)
        {
            if (depthProbeRows[i] == null)
                continue;
            depthProbeRows[i].RequestSnapshotCapture();
            if (freezeAfterCapture)
                depthProbeRows[i].SetSnapshotUpdatesFrozen(true);
        }

        if (debugLog)
            Debug.Log($"[ScanCoverSkeletonRuntimeCore] CaptureProbeRowSnapshot rows={depthProbeRows.Length}, freeze={freezeAfterCapture}");
    }

    public void SetRealtimeDepthConsumption(bool enabled)
    {
        ResolveRefs();
        if (depthGridPointCloud != null)
        {
            depthGridPointCloud.SetUpdateEveryFrame(enabled);
            ScanCoverDepthPreprocessor preprocessor = depthGridPointCloud.Preprocessor;
            if (preprocessor != null)
                preprocessor.SetRefreshEveryFrame(enabled);
        }

        // Keep Meta's EnvironmentDepthManager alive. Toggling the provider at runtime is
        // unreliable on device and can destabilize editor Play-mode shutdown.
    }

    public bool RequestDepthGridBurstRefresh()
    {
        ResolveRefs();
        if (depthGridPointCloud == null)
        {
            LastIssue = "Depth grid point cloud is missing.";
            return false;
        }

        bool requested = depthGridPointCloud.RefreshNow(forcePreprocessorRefresh: true);
        LastIssue = depthGridPointCloud.LastIssue;
        return requested;
    }

    public void SetProbeRowSnapshotFrozen(bool frozen)
    {
        ResolveRefs();
        ResolveDepthProbeRows();
        for (int i = 0; i < depthProbeRows.Length; i++)
        {
            if (depthProbeRows[i] != null)
                depthProbeRows[i].SetSnapshotUpdatesFrozen(frozen);
        }
    }

    public void ToggleProbeRowSnapshotFreeze()
    {
        ResolveRefs();
        ResolveDepthProbeRows();
        bool frozen = true;
        for (int i = 0; i < depthProbeRows.Length; i++)
        {
            if (depthProbeRows[i] == null)
                continue;
            frozen = !depthProbeRows[i].SnapshotUpdatesFrozen;
            break;
        }
        SetProbeRowSnapshotFrozen(frozen);
    }

    public void SetPreviewVisible(bool visible)
    {
        ResolveRefs();
        if (depthGridPointCloud != null)
            depthGridPointCloud.SetPreviewVisible(visible);
    }

    public void RestoreMainlineVisibility()
    {
        ResolveRefs();
        if (depthGridPointCloud != null && !depthGridPointCloud.PreviewVisible)
            depthGridPointCloud.SetPreviewVisible(true);
    }

    public void TogglePreviewVisible()
    {
        ResolveRefs();
        bool next = depthGridPointCloud == null || !depthGridPointCloud.PreviewVisible;
        SetPreviewVisible(next);
    }

    private void ResolveDepthProbeRows()
    {
        if (depthProbeRows != null && depthProbeRows.Length > 0)
            return;

        depthProbeRows = GetComponentsInChildren<DepthEffectsDepthProbeRowRenderer>(true);
        if (depthProbeRows == null || depthProbeRows.Length == 0)
            depthProbeRows = Object.FindObjectsOfType<DepthEffectsDepthProbeRowRenderer>(true);
    }
}
