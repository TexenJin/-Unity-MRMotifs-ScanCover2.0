using UnityEngine;

[DisallowMultipleComponent]
public class ScanCoverDisplaySurfaceRenderer : MonoBehaviour
{
    [Header("Refs")]
    public ScanCoverDisplaySurfaceBuilder surfaceBuilder;
    public ScanCoverSkeletonSessionController sessionController;
    public ScanCoverSkeletonMesher_B mesher;

    [Header("Behavior")]
    public bool showDisplayWhenFrozen = true;
    public bool hideDisplayWhileScanning = true;
    public bool autoRebuildOnFreeze = true;
    public bool rebuildWhenFrozenChunkCountChanges = true;
    public bool hideSourceChunksWhenDisplayVisible = true;
    public bool restoreSourceChunksWhenDisplayHidden = false;

    [Header("Debug")]
    public bool debugLog = false;

    private ScanCoverSkeletonSessionController.SessionState _lastState;
    private int _lastChunkCount = -1;

    private void Awake()
    {
        if (!surfaceBuilder) surfaceBuilder = GetComponent<ScanCoverDisplaySurfaceBuilder>();
        if (!sessionController) sessionController = GetComponent<ScanCoverSkeletonSessionController>();
        if (!mesher) mesher = GetComponent<ScanCoverSkeletonMesher_B>();
        if (sessionController != null && !sessionController.enableLegacyVisualChain)
        {
            enabled = false;
            return;
        }
        _lastState = sessionController ? sessionController.State : ScanCoverSkeletonSessionController.SessionState.Idle;
    }

    private void Update()
    {
        if (!surfaceBuilder) surfaceBuilder = GetComponent<ScanCoverDisplaySurfaceBuilder>();
        if (!sessionController) sessionController = GetComponent<ScanCoverSkeletonSessionController>();
        if (!mesher) mesher = GetComponent<ScanCoverSkeletonMesher_B>();
        if (sessionController != null && !sessionController.enableLegacyVisualChain)
        {
            if (surfaceBuilder != null)
                surfaceBuilder.SetVisible(false);
            enabled = false;
            return;
        }
        if (!surfaceBuilder || !sessionController) return;

        var state = sessionController.State;
        bool stateChanged = state != _lastState;
        bool frozen = state == ScanCoverSkeletonSessionController.SessionState.Frozen;

        if (stateChanged)
        {
            if (debugLog)
                Debug.Log($"[ScanCoverDisplaySurfaceRenderer] state {_lastState} -> {state}");

            if (state == ScanCoverSkeletonSessionController.SessionState.Scanning)
            {
                if (hideDisplayWhileScanning)
                    surfaceBuilder.SetVisible(false);
                if (restoreSourceChunksWhenDisplayHidden && hideSourceChunksWhenDisplayVisible)
                    surfaceBuilder.SetSourceChunkRenderersVisible(true);
            }
            else if (state == ScanCoverSkeletonSessionController.SessionState.Frozen)
            {
                if (autoRebuildOnFreeze)
                    surfaceBuilder.RebuildFromSource();
                surfaceBuilder.SetVisible(showDisplayWhenFrozen);
                if (hideSourceChunksWhenDisplayVisible && showDisplayWhenFrozen)
                    surfaceBuilder.SetSourceChunkRenderersVisible(false);
            }
            else
            {
                surfaceBuilder.SetVisible(false);
                if (restoreSourceChunksWhenDisplayHidden && hideSourceChunksWhenDisplayVisible)
                    surfaceBuilder.SetSourceChunkRenderersVisible(true);
            }
        }

        if (frozen && rebuildWhenFrozenChunkCountChanges && mesher)
        {
            int cc = mesher.ChunkCount;
            if (cc != _lastChunkCount)
            {
                surfaceBuilder.RebuildFromSource();
                surfaceBuilder.SetVisible(showDisplayWhenFrozen);
                if (hideSourceChunksWhenDisplayVisible && showDisplayWhenFrozen)
                    surfaceBuilder.SetSourceChunkRenderersVisible(false);
            }
            _lastChunkCount = cc;
        }
        else if (mesher)
        {
            _lastChunkCount = mesher.ChunkCount;
        }

        _lastState = state;
    }
}
