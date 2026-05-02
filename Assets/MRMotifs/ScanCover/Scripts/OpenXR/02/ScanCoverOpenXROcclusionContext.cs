using System.Reflection;
using UnityEngine;
using UnityEngine.XR.ARSubsystems;
using UnityEngine.XR.Management;
using UnityEngine.XR.OpenXR.Features.Meta;

namespace MRMotifs.ScanCover.OpenXR02
{
    [DisallowMultipleComponent]
    public sealed class ScanCoverOpenXROcclusionContext : MonoBehaviour
    {
        [Header("Lifecycle")]
        [SerializeField] private bool autoResolveOnEnable = true;
        [SerializeField] private bool autoStartSubsystem = true;
        [SerializeField] private bool keepTryingWhileUnready = true;
        [SerializeField] private bool stopSubsystemOnDisable = false;
        [SerializeField, Min(0.1f)] private float retryStartIntervalSeconds = 1.0f;

        [Header("Debug")]
        [SerializeField] private bool debugLog;

        public XRManagerSettings XRManager { get; private set; }
        public XRLoader ActiveLoader { get; private set; }
        public XROcclusionSubsystem OcclusionSubsystem { get; private set; }
        public MetaOpenXROcclusionSubsystem MetaOcclusionSubsystem { get; private set; }
        public string LastIssue { get; private set; }
        public string ActiveLoaderTypeName => ActiveLoader != null ? ActiveLoader.GetType().FullName : "<null>";
        public string OcclusionSubsystemTypeName => OcclusionSubsystem != null ? OcclusionSubsystem.GetType().FullName : "<null>";
        public string OcclusionDescriptorId => TryGetDescriptorId(OcclusionSubsystem);
        public bool HasActiveLoader => ActiveLoader != null;
        public bool HasOcclusionSubsystem => OcclusionSubsystem != null;
        public bool IsRunning => OcclusionSubsystem != null && OcclusionSubsystem.running;

        private float _nextRetryTime;
        private XRLoader _lastBoundLoader;

        private void OnEnable()
        {
            if (autoResolveOnEnable)
                ResolveNow();
        }

        private void Update()
        {
            if (!keepTryingWhileUnready && (!autoStartSubsystem || IsRunning))
                return;

            if (Time.unscaledTime < _nextRetryTime)
                return;

            _nextRetryTime = Time.unscaledTime + Mathf.Max(0.1f, retryStartIntervalSeconds);

            if (!ResolveNow())
                return;

            if (autoStartSubsystem && !IsRunning)
                TryStartSubsystem();
        }

        private void OnDisable()
        {
            if (stopSubsystemOnDisable && IsRunning)
                StopSubsystem();
        }

        [ContextMenu("Resolve OpenXR Occlusion")]
        public bool ResolveNow()
        {
            XRManagerSettings xrManager = XRGeneralSettings.Instance?.Manager;
            XRManager = xrManager;
            if (xrManager == null)
            {
                SetIssue("XRGeneralSettings.Manager is null.");
                ClearSubsystemBinding();
                return false;
            }

            XRLoader loader = xrManager.activeLoader;
            if (loader == null)
            {
                SetIssue("XR active loader is null.");
                ClearSubsystemBinding();
                return false;
            }

            ActiveLoader = loader;
            if (_lastBoundLoader == loader && OcclusionSubsystem != null)
                return true;

            _lastBoundLoader = loader;
            OcclusionSubsystem = loader.GetLoadedSubsystem<XROcclusionSubsystem>();
            MetaOcclusionSubsystem = OcclusionSubsystem as MetaOpenXROcclusionSubsystem;
            if (OcclusionSubsystem == null)
            {
                SetIssue($"Active loader '{loader.GetType().FullName}' has no loaded XROcclusionSubsystem.");
                return false;
            }

            LastIssue = null;
            if (debugLog)
            {
                Debug.Log(
                    $"[ScanCoverOpenXROcclusionContext] loader={ActiveLoaderTypeName}, " +
                    $"occlusion={OcclusionSubsystemTypeName}, descriptor={OcclusionDescriptorId}");
            }
            return true;
        }

        [ContextMenu("Start OpenXR Occlusion")]
        public bool TryStartSubsystem()
        {
            if (!ResolveNow())
                return false;

            if (OcclusionSubsystem.running)
                return true;

            OcclusionSubsystem.Start();
            bool running = OcclusionSubsystem.running;
            if (!running)
            {
                SetIssue("Occlusion subsystem Start() returned without entering running state.");
                return false;
            }

            LastIssue = null;
            if (debugLog)
                Debug.Log("[ScanCoverOpenXROcclusionContext] Occlusion subsystem started.");
            return true;
        }

        [ContextMenu("Stop OpenXR Occlusion")]
        public void StopSubsystem()
        {
            if (OcclusionSubsystem == null || !OcclusionSubsystem.running)
                return;

            OcclusionSubsystem.Stop();
            if (debugLog)
                Debug.Log("[ScanCoverOpenXROcclusionContext] Occlusion subsystem stopped.");
        }

        public bool TryEnsureReadyAndRunning()
        {
            if (!ResolveNow())
                return false;

            if (IsRunning)
                return true;

            if (!autoStartSubsystem)
            {
                SetIssue("Occlusion subsystem is resolved but not running.");
                return false;
            }

            return TryStartSubsystem();
        }

        public string BuildSummary()
        {
            return
                $"loader={ActiveLoaderTypeName}, " +
                $"occlusion={OcclusionSubsystemTypeName}, " +
                $"descriptor={OcclusionDescriptorId}, " +
                $"running={IsRunning}, " +
                $"issue={(string.IsNullOrEmpty(LastIssue) ? "<none>" : LastIssue)}";
        }

        private void ClearSubsystemBinding()
        {
            ActiveLoader = null;
            OcclusionSubsystem = null;
            MetaOcclusionSubsystem = null;
            _lastBoundLoader = null;
        }

        private void SetIssue(string message)
        {
            LastIssue = message;
            if (debugLog)
                Debug.LogWarning($"[ScanCoverOpenXROcclusionContext] {message}");
        }

        private static string TryGetDescriptorId(XROcclusionSubsystem occlusion)
        {
            if (occlusion == null)
                return "<null>";

            PropertyInfo descriptorProperty = occlusion.GetType().GetProperty(
                "subsystemDescriptor",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);

            object descriptor = descriptorProperty?.GetValue(occlusion);
            if (descriptor == null)
                return "<unavailable>";

            PropertyInfo idProperty = descriptor.GetType().GetProperty(
                "id",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);

            object id = idProperty?.GetValue(descriptor);
            return id?.ToString() ?? "<unavailable>";
        }
    }
}
