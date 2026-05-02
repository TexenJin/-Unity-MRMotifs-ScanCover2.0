using System.Reflection;
using UnityEngine;
using UnityEngine.XR.ARSubsystems;
using UnityEngine.XR.Management;

namespace MRMotifs.ScanCover.OpenXR01
{
    public sealed class ScanCoverOpenXROcclusionProbe : MonoBehaviour
    {
        [Header("Probe")]
        [SerializeField] private bool probeOnStart = true;
        [SerializeField] private bool repeatProbe = false;
        [SerializeField] private float repeatIntervalSeconds = 2f;
        [SerializeField] private bool logWarningsAsErrors = false;

        private float _nextProbeTime;

        private void Start()
        {
            if (!probeOnStart)
                return;

            ProbeOnce();
            _nextProbeTime = Time.unscaledTime + Mathf.Max(0.1f, repeatIntervalSeconds);
        }

        private void Update()
        {
            if (!repeatProbe)
                return;

            if (Time.unscaledTime < _nextProbeTime)
                return;

            ProbeOnce();
            _nextProbeTime = Time.unscaledTime + Mathf.Max(0.1f, repeatIntervalSeconds);
        }

        [ContextMenu("Probe OpenXR Occlusion Once")]
        public void ProbeOnce()
        {
            XRManagerSettings xrManager = XRGeneralSettings.Instance?.Manager;
            if (xrManager == null)
            {
                LogIssue("XRGeneralSettings.Manager is null.");
                return;
            }

            XRLoader activeLoader = xrManager.activeLoader;
            if (activeLoader == null)
            {
                LogIssue("No active XR loader. Check XR Plug-in Management Android settings.");
                return;
            }

            string loaderName = activeLoader.GetType().FullName;
            XROcclusionSubsystem occlusion = activeLoader.GetLoadedSubsystem<XROcclusionSubsystem>();

            if (occlusion == null)
            {
                Debug.Log(
                    $"[ScanCoverOpenXROcclusionProbe] Active loader: {loaderName}. " +
                    "XROcclusionSubsystem is null. OpenXR may be active without a loaded occlusion provider.");
                return;
            }

            string subsystemType = occlusion.GetType().FullName;
            string descriptorId = TryGetDescriptorId(occlusion);
            string providerType = TryGetProviderTypeName(occlusion);

            Debug.Log(
                $"[ScanCoverOpenXROcclusionProbe] Active loader: {loaderName}\n" +
                $"Occlusion subsystem type: {subsystemType}\n" +
                $"Occlusion descriptor id: {descriptorId}\n" +
                $"Occlusion provider type: {providerType}");
        }

        private void LogIssue(string message)
        {
            if (logWarningsAsErrors)
                Debug.LogError($"[ScanCoverOpenXROcclusionProbe] {message}");
            else
                Debug.LogWarning($"[ScanCoverOpenXROcclusionProbe] {message}");
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

        private static string TryGetProviderTypeName(XROcclusionSubsystem occlusion)
        {
            if (occlusion == null)
                return "<null>";

            PropertyInfo providerProperty = occlusion.GetType().GetProperty(
                "provider",
                BindingFlags.Instance | BindingFlags.NonPublic);

            object provider = providerProperty?.GetValue(occlusion);
            return provider?.GetType().FullName ?? "<unavailable>";
        }
    }
}
