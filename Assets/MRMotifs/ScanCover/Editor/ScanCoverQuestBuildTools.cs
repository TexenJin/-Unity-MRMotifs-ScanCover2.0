using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.XR.Management;
using UnityEngine;
using UnityEngine.XR.Management;
using UnityEngine.XR.OpenXR;
using UnityEngine.XR.OpenXR.Features;

public static class ScanCoverQuestBuildTools
{
    private const string ScenePath = "Assets/MRMotifs/ScanCover/Scene/DepthEffects_ScanCover.unity";
    private const string ApkPath = "Builds/Quest3/ScanCoverQuest3.apk";
    private const string PackageName = "com.pcaii.scancover.quest3";

    [MenuItem("ScanCover/Quest 3/Configure Project For Device Build")]
    public static void ConfigureProjectForDeviceBuild()
    {
        EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Android, BuildTarget.Android);
        EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };

        PlayerSettings.companyName = "PCAII";
        PlayerSettings.productName = "ScanCover Quest3";
        PlayerSettings.bundleVersion = "0.1.0";
        PlayerSettings.SetApplicationIdentifier(BuildTargetGroup.Android, PackageName);
        PlayerSettings.Android.bundleVersionCode = 1;
        PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel32;
        PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevelAuto;
        PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
        PlayerSettings.Android.applicationEntry = AndroidApplicationEntry.GameActivity;
        PlayerSettings.SetScriptingBackend(BuildTargetGroup.Android, ScriptingImplementation.IL2CPP);
        PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { UnityEngine.Rendering.GraphicsDeviceType.Vulkan });
        PlayerSettings.Android.startInFullscreen = true;
        PlayerSettings.Android.renderOutsideSafeArea = true;
        PlayerSettings.defaultInterfaceOrientation = UIOrientation.LandscapeLeft;

        EnsureAndroidOpenXRLoader();
        EnableAndroidOpenXRFeature("com.meta.openxr.feature.metaxr");
        EnableAndroidOpenXRFeature("com.unity.openxr.feature.input.oculustouch");
        EnableAndroidOpenXRFeature("com.unity.openxr.feature.input.metaquestplus");
        EnableAndroidOpenXRFeature("com.unity.openxr.feature.compositionlayers");
        EnableAndroidOpenXRFeature("com.unity.openxr.feature.arfoundation-meta-camera");
        EnableAndroidOpenXRFeature("com.unity.openxr.feature.arfoundation-meta-session");
        EnableAndroidOpenXRFeature("com.unity.openxr.feature.arfoundation-meta-occlusion");
        EnableAndroidOpenXRFeature("com.unity.openxr.feature.arfoundation-meta-plane");
        EnableAndroidOpenXRFeature("com.unity.openxr.feature.arfoundation-meta-bounding-boxes");
        EnableAndroidOpenXRFeature("com.unity.openxr.feature.meta-boundary-visibility");

        AssetDatabase.SaveAssets();
        Debug.Log("[ScanCoverQuestBuildTools] Quest 3 device build configuration applied.");
    }

    [MenuItem("ScanCover/Quest 3/Build APK")]
    public static void BuildApk()
    {
        ConfigureProjectForDeviceBuild();

        string fullApkPath = Path.GetFullPath(Path.Combine(Directory.GetParent(Application.dataPath)!.FullName, ApkPath));
        Directory.CreateDirectory(Path.GetDirectoryName(fullApkPath)!);

        BuildPlayerOptions options = new BuildPlayerOptions
        {
            scenes = new[] { ScenePath },
            locationPathName = fullApkPath,
            target = BuildTarget.Android,
            targetGroup = BuildTargetGroup.Android,
            options = BuildOptions.Development
        };

        var report = BuildPipeline.BuildPlayer(options);
        Debug.Log($"[ScanCoverQuestBuildTools] APK build result: {report.summary.result} => {fullApkPath}");
    }

    private static void EnsureAndroidOpenXRLoader()
    {
        var general = XRGeneralSettingsPerBuildTarget.XRGeneralSettingsForBuildTarget(BuildTargetGroup.Android);
        if (general == null)
        {
            if (!EditorBuildSettings.TryGetConfigObject(XRGeneralSettings.k_SettingsKey, out XRGeneralSettingsPerBuildTarget perBuildTarget) || perBuildTarget == null)
            {
                Debug.LogError("[ScanCoverQuestBuildTools] XRGeneralSettingsPerBuildTarget asset is not registered.");
                return;
            }

            perBuildTarget.CreateDefaultSettingsForBuildTarget(BuildTargetGroup.Android);
            general = perBuildTarget.SettingsForBuildTarget(BuildTargetGroup.Android);
        }

        if (general.Manager == null)
        {
            var manager = ScriptableObject.CreateInstance<XRManagerSettings>();
            manager.name = "Android Providers";
            AssetDatabase.AddObjectToAsset(manager, "Assets/XR/XRGeneralSettingsPerBuildTarget.asset");
            general.AssignedSettings = manager;
        }

        general.InitManagerOnStart = true;
        general.Manager.automaticLoading = true;
        general.Manager.automaticRunning = true;

        bool hasOpenXR = general.Manager.activeLoaders.Any(loader => loader is OpenXRLoader);
        if (!hasOpenXR)
        {
            var loader = AssetDatabase.LoadAssetAtPath<OpenXRLoader>("Assets/XR/Loaders/OpenXRLoader.asset");
            if (loader == null)
            {
                loader = ScriptableObject.CreateInstance<OpenXRLoader>();
                AssetDatabase.CreateAsset(loader, "Assets/XR/Loaders/OpenXRLoader.asset");
            }
            general.Manager.TryAddLoader(loader);
        }

        EditorUtility.SetDirty(general);
        EditorUtility.SetDirty(general.Manager);
    }

    private static void EnableAndroidOpenXRFeature(string featureId)
    {
        OpenXRSettings settings = OpenXRSettings.GetSettingsForBuildTargetGroup(BuildTargetGroup.Android);
        if (settings == null)
            return;

        OpenXRFeature feature = settings.GetFeatures().FirstOrDefault(f => string.Equals(ReadFeatureId(f), featureId, StringComparison.Ordinal));
        if (feature == null)
        {
            Debug.LogWarning($"[ScanCoverQuestBuildTools] OpenXR feature not found for Android: {featureId}");
            return;
        }

        feature.enabled = true;
        EditorUtility.SetDirty(feature);
        EditorUtility.SetDirty(settings);
    }

    private static string ReadFeatureId(OpenXRFeature feature)
    {
        SerializedObject serialized = new SerializedObject(feature);
        SerializedProperty property = serialized.FindProperty("featureIdInternal");
        return property == null ? string.Empty : property.stringValue;
    }
}
