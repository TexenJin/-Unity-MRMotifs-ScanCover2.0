using Unity.Collections;
using UnityEngine;
using UnityEngine.XR.ARSubsystems;
using Meta.XR.EnvironmentDepth;

namespace MRMotifs.ScanCover.OpenXR02
{
    /// <summary>
    /// Diagnostics-only bridge probe.
    /// Official ScanCover depth consumption remains on EnvironmentDepthManager.
    /// This component only verifies whether MetaOpenXROcclusionSubsystem is available
    /// and whether EnvironmentDepthManager is successfully publishing the bridged depth texture.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class ScanCoverOpenXRDepthFrameSource : MonoBehaviour
    {
        [Header("Refs")]
        [SerializeField] private ScanCoverOpenXROcclusionContext context;
        [SerializeField] private EnvironmentDepthManager environmentDepthManager;

        [Header("Refresh")]
        [SerializeField] private bool refreshEveryFrame = true;
        [SerializeField, Min(0f)] private float refreshIntervalSeconds = 0f;
        [SerializeField] private bool requireRunningSubsystem = true;
        [SerializeField] private bool preferEnvironmentDepthBridge = true;
        [SerializeField] private bool requireEnvironmentDepthManager = true;
        [SerializeField, Min(0f)] private float bridgeWarmupSeconds = 2f;

        [Header("Debug")]
        [SerializeField] private bool debugLog;

        public XRResultStatus LastFrameStatus { get; private set; } = XRResultStatus.unqualifiedSuccess;
        public bool HasFrame { get; private set; }
        public bool HasTextureDescriptor { get; private set; }
        public string LastIssue { get; private set; }
        public long LastTimestampNs { get; private set; }
        public XRNearFarPlanes LastNearFarPlanes { get; private set; }
        public int LastPoseCount => _poseCount;
        public int LastFovCount => _fovCount;
        public XRTextureDescriptor LastTextureDescriptor { get; private set; }
        public Texture CurrentEnvironmentDepthTexture =>
            HasTextureDescriptor ? Shader.GetGlobalTexture(LastTextureDescriptor.propertyNameId) : null;
        public Texture CurrentBridgeDepthTexture => Shader.GetGlobalTexture(EnvironmentDepthTextureId);

        private readonly Pose[] _poses = new Pose[2];
        private readonly XRFov[] _fovs = new XRFov[2];
        private static readonly int EnvironmentDepthTextureId = Shader.PropertyToID("_EnvironmentDepthTexture");
        private int _poseCount;
        private int _fovCount;
        private float _nextRefreshTime;
        private bool _wasSubsystemRunning;
        private float _subsystemRunningSince;

        private void Awake()
        {
            ResolveRefs();
        }

        private void OnEnable()
        {
            ResolveRefs();
        }

        private void Update()
        {
            if (!refreshEveryFrame)
                return;

            if (refreshIntervalSeconds > 0f && Time.unscaledTime < _nextRefreshTime)
                return;

            _nextRefreshTime = Time.unscaledTime + Mathf.Max(0f, refreshIntervalSeconds);
            RefreshNow();
        }

        public bool TryGetPose(int index, out Pose pose)
        {
            if (index >= 0 && index < _poseCount)
            {
                pose = _poses[index];
                return true;
            }

            pose = default;
            return false;
        }

        public bool TryGetFov(int index, out XRFov fov)
        {
            if (index >= 0 && index < _fovCount)
            {
                fov = _fovs[index];
                return true;
            }

            fov = default;
            return false;
        }

        [ContextMenu("Refresh OpenXR Depth Frame")]
        public bool RefreshNow()
        {
            ResolveRefs();
            if (context == null)
            {
                SetIssue("ScanCoverOpenXROcclusionContext is missing.");
                return false;
            }

            bool ready = requireRunningSubsystem
                ? context.TryEnsureReadyAndRunning()
                : context.ResolveNow();
            if (!ready || context.OcclusionSubsystem == null)
            {
                SetIssue(context.LastIssue ?? "OpenXR occlusion context is not ready.");
                return false;
            }

            UpdateRunningWindow();

            if (!PrepareDepthOwner())
                return false;

            using XROcclusionFrame frame = FetchFrame(context.OcclusionSubsystem, out XRResultStatus frameStatus);
            LastFrameStatus = frameStatus;
            if (!frameStatus.IsSuccess())
            {
                HasFrame = false;
                if (TryCaptureEnvironmentDepthBridgeFallback())
                    return true;

                SetIssue(
                    $"TryGetFrame failed with status={frameStatus.statusCode}, native={frameStatus.nativeStatusCode}.");
                CaptureTextureDescriptor(context.OcclusionSubsystem);
                return false;
            }

            CopyFrame(frame);
            CaptureTextureDescriptor(context.OcclusionSubsystem);
            LastIssue = null;

            if (debugLog)
            {
                Debug.Log(
                    $"[ScanCoverOpenXRDepthFrameSource] frameTs={LastTimestampNs}, " +
                    $"poses={_poseCount}, fovs={_fovCount}, textureValid={HasTextureDescriptor}");
            }
            return true;
        }

        private void ResolveRefs()
        {
            if (context == null)
                context = GetComponent<ScanCoverOpenXROcclusionContext>();

            if (environmentDepthManager == null)
                environmentDepthManager = FindAnyObjectByType<EnvironmentDepthManager>();
        }

        private bool PrepareDepthOwner()
        {
            if (!preferEnvironmentDepthBridge)
            {
                SetIssue(
                    "preferEnvironmentDepthBridge is disabled. Current project decision is to keep " +
                    "EnvironmentDepthManager as the official consumption layer.");
                return false;
            }

            return PrepareEnvironmentDepthBridge();
        }

        private bool PrepareEnvironmentDepthBridge()
        {
            if (environmentDepthManager == null)
            {
                if (!requireEnvironmentDepthManager)
                    return true;

                SetIssue("EnvironmentDepthManager is missing.");
                return false;
            }

            if (!environmentDepthManager.isActiveAndEnabled)
            {
                SetIssue("EnvironmentDepthManager is present but disabled.");
                return false;
            }

            bool bridgeWarmupActive = context != null &&
                context.IsRunning &&
                Time.unscaledTime < _subsystemRunningSince + Mathf.Max(0f, bridgeWarmupSeconds);

            if (debugLog)
            {
                Texture bridgeTexture = CurrentBridgeDepthTexture;
                Debug.Log(
                    $"[ScanCoverOpenXRDepthFrameSource] EnvironmentDepth bridge supported=" +
                    $"{EnvironmentDepthManager.IsSupported}, enabled={environmentDepthManager.enabled}, " +
                    $"depthAvailable={environmentDepthManager.IsDepthAvailable}, globalTextureValid={bridgeTexture != null}, " +
                    $"bridgeWarmupActive={bridgeWarmupActive}");
            }

            if (bridgeWarmupActive &&
                !environmentDepthManager.IsDepthAvailable &&
                CurrentBridgeDepthTexture == null)
            {
                LastIssue = null;
                return true;
            }

            return true;
        }

        private bool TryCaptureEnvironmentDepthBridgeFallback()
        {
            if (environmentDepthManager == null || !environmentDepthManager.isActiveAndEnabled)
                return false;

            Texture bridgeTexture = CurrentBridgeDepthTexture;
            if (!environmentDepthManager.IsDepthAvailable || bridgeTexture == null)
                return false;

            HasTextureDescriptor = false;
            LastIssue = null;

            if (debugLog)
            {
                Debug.Log(
                    "[ScanCoverOpenXRDepthFrameSource] Direct TryGetFrame failed, but EnvironmentDepthManager " +
                    "bridge is active and publishing _EnvironmentDepthTexture.");
            }

            return true;
        }

        private static XROcclusionFrame FetchFrame(
            XROcclusionSubsystem subsystem,
            out XRResultStatus status)
        {
            status = subsystem.TryGetFrame(Allocator.Temp, out XROcclusionFrame frame);
            return frame;
        }

        private void UpdateRunningWindow()
        {
            bool running = context != null && context.IsRunning;
            if (running && !_wasSubsystemRunning)
                _subsystemRunningSince = Time.unscaledTime;

            _wasSubsystemRunning = running;
        }

        private void CopyFrame(XROcclusionFrame frame)
        {
            HasFrame = true;
            LastTimestampNs = 0L;
            LastNearFarPlanes = default;
            _poseCount = 0;
            _fovCount = 0;

            frame.TryGetTimestamp(out long timestampNs);
            LastTimestampNs = timestampNs;

            frame.TryGetNearFarPlanes(out XRNearFarPlanes nearFarPlanes);
            LastNearFarPlanes = nearFarPlanes;

            if (frame.TryGetPoses(out NativeArray<Pose> poses))
            {
                _poseCount = Mathf.Min(_poses.Length, poses.Length);
                for (int i = 0; i < _poseCount; i++)
                    _poses[i] = poses[i];
            }

            if (frame.TryGetFovs(out NativeArray<XRFov> fovs))
            {
                _fovCount = Mathf.Min(_fovs.Length, fovs.Length);
                for (int i = 0; i < _fovCount; i++)
                    _fovs[i] = fovs[i];
            }
        }

        private void CaptureTextureDescriptor(XROcclusionSubsystem subsystem)
        {
            HasTextureDescriptor = subsystem.TryGetEnvironmentDepth(out XRTextureDescriptor descriptor) && descriptor.valid;
            LastTextureDescriptor = descriptor;
        }

        private void SetIssue(string message)
        {
            LastIssue = message;
            if (debugLog)
                Debug.LogWarning($"[ScanCoverOpenXRDepthFrameSource] {message}");
        }
    }
}
