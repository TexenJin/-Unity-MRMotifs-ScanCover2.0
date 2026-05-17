using UnityEngine;
using UnityEngine.XR.Management;

[DefaultExecutionOrder(-32000)]
[DisallowMultipleComponent]
public sealed class ScanCoverXRRuntimeBootstrap : MonoBehaviour
{
    private static bool s_attemptedStart;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void AutoCreate()
    {
        if (FindAnyObjectByType<ScanCoverXRRuntimeBootstrap>(FindObjectsInactive.Include) != null)
            return;

        new GameObject("[ScanCover] XR Runtime Bootstrap")
            .AddComponent<ScanCoverXRRuntimeBootstrap>();
    }

    private void Awake()
    {
        TryStartXR("Awake");
    }

    private void Start()
    {
        TryStartXR("Start");
    }

    private static void TryStartXR(string reason)
    {
        if (s_attemptedStart)
            return;

        XRGeneralSettings settings = XRGeneralSettings.Instance;
        XRManagerSettings manager = settings != null ? settings.Manager : null;
        if (manager == null)
        {
            Debug.LogWarning($"[ScanCoverXRRuntimeBootstrap] {reason}: XRManager missing");
            return;
        }

        if (manager.activeLoader != null)
        {
            Debug.Log($"[ScanCoverXRRuntimeBootstrap] {reason}: activeLoader already started: {manager.activeLoader.name}");
            s_attemptedStart = true;
            return;
        }

        if (manager.activeLoaders.Count <= 0)
        {
            Debug.LogWarning($"[ScanCoverXRRuntimeBootstrap] {reason}: no XR loaders configured");
            s_attemptedStart = true;
            return;
        }

        s_attemptedStart = true;
        Debug.Log($"[ScanCoverXRRuntimeBootstrap] {reason}: initializing XR loader manually. automaticLoading={manager.automaticLoading} automaticRunning={manager.automaticRunning} loaders={manager.activeLoaders.Count}");

        manager.InitializeLoaderSync();
        if (manager.activeLoader == null)
        {
            Debug.LogError("[ScanCoverXRRuntimeBootstrap] InitializeLoaderSync failed: activeLoader is still null. Check Meta Quest Link active OpenXR runtime and Project Settings > XR Plug-in Management > Standalone.");
            return;
        }

        manager.StartSubsystems();
        Debug.Log($"[ScanCoverXRRuntimeBootstrap] XR started with loader={manager.activeLoader.name}");
    }
}
