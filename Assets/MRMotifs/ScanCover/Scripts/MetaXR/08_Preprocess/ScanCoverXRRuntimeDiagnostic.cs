using System.Collections.Generic;
using System.Text;
using Meta.XR.EnvironmentDepth;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.XR.Management;

[DisallowMultipleComponent]
public sealed class ScanCoverXRRuntimeDiagnostic : MonoBehaviour
{
    [SerializeField, Min(0.5f)] private float logIntervalSeconds = 2f;

    private float _nextLogTime;
    private bool _attemptedStart;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void AutoCreate()
    {
        if (FindAnyObjectByType<ScanCoverXRRuntimeDiagnostic>(FindObjectsInactive.Include) != null)
            return;

        new GameObject("[ScanCover] XR Runtime Diagnostic")
            .AddComponent<ScanCoverXRRuntimeDiagnostic>();
    }

    private void Update()
    {
        TryStartXR();

        if (Time.unscaledTime < _nextLogTime)
            return;

        _nextLogTime = Time.unscaledTime + logIntervalSeconds;
        LogStatus();
    }

    private void TryStartXR()
    {
        if (_attemptedStart)
            return;

        XRGeneralSettings settings = XRGeneralSettings.Instance;
        XRManagerSettings manager = settings != null ? settings.Manager : null;
        if (manager == null)
            return;

        if (manager.activeLoader != null)
        {
            _attemptedStart = true;
            return;
        }

        if (manager.activeLoaders.Count <= 0)
        {
            _attemptedStart = true;
            Debug.LogWarning("[ScanCoverXRRuntimeDiagnostic] Cannot start XR: no active loaders configured.");
            return;
        }

        _attemptedStart = true;
        Debug.Log($"[ScanCoverXRRuntimeDiagnostic] activeLoader is null; manually starting XR. automaticLoading={manager.automaticLoading} automaticRunning={manager.automaticRunning} loaders={manager.activeLoaders.Count}");

        manager.InitializeLoaderSync();
        if (manager.activeLoader == null)
        {
            Debug.LogError("[ScanCoverXRRuntimeDiagnostic] InitializeLoaderSync failed: activeLoader is still null. Meta Quest Link may not be the active OpenXR runtime, or OpenXR failed to initialize.");
            return;
        }

        manager.StartSubsystems();
        Debug.Log($"[ScanCoverXRRuntimeDiagnostic] XR started with loader={manager.activeLoader.name}");
    }

    private static void LogStatus()
    {
        XRGeneralSettings settings = XRGeneralSettings.Instance;
        XRManagerSettings manager = settings != null ? settings.Manager : null;

        var displaySubsystems = new List<XRDisplaySubsystem>();
        SubsystemManager.GetSubsystems(displaySubsystems);

        var inputSubsystems = new List<XRInputSubsystem>();
        SubsystemManager.GetSubsystems(inputSubsystems);

        StringBuilder sb = new StringBuilder(1024);
        sb.AppendLine("[ScanCoverXRRuntimeDiagnostic]");
        sb.AppendLine($"  XRGeneralSettings={(settings != null ? "present" : "missing")}");
        sb.AppendLine($"  XRManager={(manager != null ? "present" : "missing")}");
        if (manager != null)
        {
            sb.AppendLine($"  automaticLoading={manager.automaticLoading} automaticRunning={manager.automaticRunning}");
            sb.AppendLine($"  activeLoader={(manager.activeLoader != null ? manager.activeLoader.name : "null")}");
            sb.AppendLine($"  activeLoaders={manager.activeLoaders.Count}");
            for (int i = 0; i < manager.activeLoaders.Count; i++)
            {
                var loader = manager.activeLoaders[i];
                sb.AppendLine($"    loader[{i}]={(loader != null ? loader.name : "null")}");
            }
        }

        sb.AppendLine($"  XRDisplaySubsystems={displaySubsystems.Count}");
        for (int i = 0; i < displaySubsystems.Count; i++)
        {
            var display = displaySubsystems[i];
            sb.AppendLine($"    display[{i}] running={display.running} id={display.SubsystemDescriptor.id}");
        }

        sb.AppendLine($"  XRInputSubsystems={inputSubsystems.Count}");
        for (int i = 0; i < inputSubsystems.Count; i++)
        {
            var input = inputSubsystems[i];
            sb.AppendLine($"    input[{i}] running={input.running} id={input.SubsystemDescriptor.id}");
        }

        sb.AppendLine($"  EnvironmentDepthManager.IsSupported={EnvironmentDepthManager.IsSupported}");
        Debug.Log(sb.ToString());
    }
}
