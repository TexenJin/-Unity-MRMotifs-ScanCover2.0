using System;
using UnityEngine;

// Kept as a compatibility label for existing exports. Production is pinned to
// V013Weighted; the experimental implementations are archived outside the
// compiled runtime instead of being selectable here.
public enum RoomSetupReplicaV2FusionMode
{
    V013Weighted = 0,
    LatestSampleProbe = 1,
    RobustCandidateV1 = 2,
    RobustCandidateV2 = 3
}

/// <summary>
/// Production owner. EnvironmentDepthManager publishes the current
/// stereo depth globals; this component forwards them directly to the GPU
/// TSDF and GPU Marching Cubes backend.
/// </summary>
[DisallowMultipleComponent]
public sealed class RoomSetupReplicaV2Reconstruction : MonoBehaviour
{
    [SerializeField] private bool showReplica = true;
    [SerializeField] private Color lineColor =
        new Color(1f, 1f, 1f, 0.92f);
    [SerializeField, Range(0.75f, 3f)]
    private float lineWidthPixels = 1.5f;

    private RoomSetupReplicaV2GpuBackend _gpuBackend;

    public int AcceptedPairCount { get; private set; }
    public int DroppedBusyPairCount { get; private set; }
    public int ActiveChunkCount =>
        _gpuBackend != null && _gpuBackend.Ready
            ? _gpuBackend.TileCount
            : 0;
    public bool PreviewVisible => showReplica;
    public bool UniqueEdgeDisplay =>
        _gpuBackend != null && _gpuBackend.UniqueEdgesOnly;
    public bool HasPendingWork => false;
    public int IntegrationCount =>
        _gpuBackend != null ? _gpuBackend.IntegrationCount : 0;
    public bool LatestSampleStabilityProbe => false;
    public bool CapturesFusionStability => false;
    public RoomSetupReplicaV2FusionMode FusionMode =>
        RoomSetupReplicaV2FusionMode.V013Weighted;
    public string RouteLabel =>
        "environment-depth-global-v0.1.3-gpu-mc-hard3m-v1";

    public void ConfigureAsEnvironmentDepthGlobalSource()
    {
        EnsureGpuBackend();
    }

    public void PrepareForLiveCapture()
    {
        EnsureGpuBackend().SetVisible(showReplica);
    }

    public bool TryObserveEnvironmentDepthGlobal(Camera camera)
    {
        if (camera == null)
            return false;
        bool submitted =
            EnsureGpuBackend().TryObserveEnvironmentDepthGlobal(camera);
        if (submitted)
            AcceptedPairCount++;
        return submitted;
    }

    public void StopReplay()
    {
        // Production has no replay route. Kept as an explicit no-op so the capture
        // controller cannot accidentally reintroduce a second producer.
    }

    public void ClearReplica()
    {
        _gpuBackend?.Clear();
        GetComponent<RoomSetupReplicaV2StructuralExtractor>()?
            .ClearExtractedPlanes();
        AcceptedPairCount = 0;
        DroppedBusyPairCount = 0;
    }

    public void SetPreviewVisible(bool visible)
    {
        showReplica = visible;
        _gpuBackend?.SetVisible(visible);
    }

    public bool ToggleUniqueEdgeDisplay()
    {
        RoomSetupReplicaV2GpuBackend backend = EnsureGpuBackend();
        backend.SetUniqueEdgeDisplay(!backend.UniqueEdgesOnly);
        return backend.UniqueEdgesOnly;
    }

    public bool TryRequestStructuralSamples(Action<Vector4[]> completed)
    {
        return _gpuBackend != null &&
               _gpuBackend.TryRequestStructuralSamples(completed);
    }

    public bool TryRequestMeshSnapshot(
        Action<Vector3[], float[], uint[]> completed)
    {
        return _gpuBackend != null &&
               _gpuBackend.TryRequestMeshSnapshot(completed);
    }

    public void CancelMeshSnapshot()
    {
        _gpuBackend?.CancelMeshSnapshot();
    }

    private RoomSetupReplicaV2GpuBackend EnsureGpuBackend()
    {
        if (_gpuBackend == null)
        {
            _gpuBackend = new RoomSetupReplicaV2GpuBackend(
                gameObject, lineColor, lineWidthPixels);
            _gpuBackend.SetVisible(showReplica);
        }
        return _gpuBackend;
    }

    private void LateUpdate()
    {
        _gpuBackend?.Render();
    }

    private void OnDestroy()
    {
        _gpuBackend?.Dispose();
        _gpuBackend = null;
    }
}
