using System;
using System.IO;
using System.Xml;
using Genesis.RoomScan;
using Meta.XR.EnvironmentDepth;
using UnityEditor;
using UnityEditor.Android;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.XR.ARFoundation;

internal static class RoomSetupReplicaV2BBuilder
{
    public const string PackageName =
        "com.pcaii.roomsetupreplicav2environmentteacher";
    public const string ProductName = "RoomSetupReplicaV2 B Final v0.1.2";
    public const string ScenePath =
        "Assets/RoomSetupReplicaV2/B/Scenes/" +
        "RoomSetupReplicaV2EnvironmentDepthTeacher.unity";
    private const string SourceScenePath =
        "Assets/RoomSetupReplicaV2/Scenes/RoomSetupReplicaV2.unity";

    [MenuItem("Codex/RoomSetup Replica V2 B/Prepare Scene")]
    public static void PrepareScene()
    {
        string directory = Path.GetDirectoryName(ScenePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            Directory.CreateDirectory(directory);

        AssetDatabase.Refresh();
        if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(ScenePath) != null)
        {
            File.Copy(
                Path.GetFullPath(SourceScenePath),
                Path.GetFullPath(ScenePath), true);
            AssetDatabase.ImportAsset(
                ScenePath, ImportAssetOptions.ForceUpdate);
        }
        else if (!AssetDatabase.CopyAsset(SourceScenePath, ScenePath))
        {
            throw new InvalidOperationException(
                "Could not copy the RoomSetupReplicaV2 source scene for B.");
        }

        var scene = EditorSceneManager.OpenScene(
            ScenePath, OpenSceneMode.Single);

        DestroyAll<RoomSetupReplicaV2Input>();
        DestroyAll<RoomSetupReplicaV2Scanner>();
        DestroyAll<RoomSetupReplicaV2ReplayProbe>();
        DestroyAll<DepthPointCloudOverlay>();
        DestroyAll<RoomSetupReplicaV2AdmissionPointOverlay>();
        DestroyAll<RoomSetupReplicaV2PersistentBoundaryOverlay>();
        DestroyAll<DepthCapture>();
        DestroyAll<AROcclusionManager>();

        RoomSetupReplicaV2Reconstruction[] reconstructions =
            UnityEngine.Object.FindObjectsByType<
                RoomSetupReplicaV2Reconstruction>(
                FindObjectsInactive.Include, FindObjectsSortMode.None);
        if (reconstructions.Length != 1)
            throw new InvalidOperationException(
                "RoomSetupReplicaV2 B requires exactly one reconstruction; " +
                "found " + reconstructions.Length + ".");

        GameObject host = reconstructions[0].gameObject;
        host.name = "[Room Setup Replica V2] Environment Depth B";
        var scanner = host.AddComponent<
            RoomSetupReplicaV2EnvironmentDepthScanner>();
        host.AddComponent<RoomSetupReplicaV2EnvironmentDepthInput>();

        EnvironmentDepthManager manager =
            host.GetComponent<EnvironmentDepthManager>();
        if (manager == null)
            throw new InvalidOperationException(
                "RoomSetupReplicaV2 B source scene has no " +
                "EnvironmentDepthManager.");
        manager.OcclusionShadersMode = OcclusionShadersMode.None;
        manager.RemoveHands = false;
        manager.MaskMeshFilters.Clear();
        manager.enabled = false;
        EditorUtility.SetDirty(scanner);
        EditorUtility.SetDirty(manager);

        if (UnityEngine.Object.FindObjectsByType<DepthCapture>(
                FindObjectsInactive.Include,
                FindObjectsSortMode.None).Length != 0 ||
            UnityEngine.Object.FindObjectsByType<AROcclusionManager>(
                FindObjectsInactive.Include,
                FindObjectsSortMode.None).Length != 0)
            throw new InvalidOperationException(
                "RoomSetupReplicaV2 B still contains a competing depth owner.");

        EditorSceneManager.MarkSceneDirty(scene);
        if (!EditorSceneManager.SaveScene(scene))
            throw new InvalidOperationException(
                "Could not save the RoomSetupReplicaV2 B scene.");
        AssetDatabase.SaveAssets();
        Debug.Log("[RoomSetupReplicaV2BBuilder] prepared " + ScenePath);
    }

    [MenuItem("Codex/RoomSetup Replica V2 B/Build APK")]
    public static void BuildApk()
    {
        PrepareScene();
        string output = Environment.GetEnvironmentVariable(
            "ROOM_SETUP_REPLICA_V2_B_APK");
        if (string.IsNullOrWhiteSpace(output))
            output = Path.GetFullPath(
                "Builds/RoomSetupReplicaV2B/RoomSetupReplicaV2B.apk");
        else
            output = Path.GetFullPath(output);
        Directory.CreateDirectory(
            Path.GetDirectoryName(output) ??
            throw new InvalidOperationException("Output directory missing."));

        const BuildTargetGroup group = BuildTargetGroup.Android;
        string oldIdentifier =
            PlayerSettings.GetApplicationIdentifier(group);
        string oldProductName = PlayerSettings.productName;
        string oldVersion = PlayerSettings.bundleVersion;
        int oldVersionCode = PlayerSettings.Android.bundleVersionCode;
        ScriptingImplementation oldBackend =
            PlayerSettings.GetScriptingBackend(group);
        AndroidArchitecture oldArchitectures =
            PlayerSettings.Android.targetArchitectures;
        try
        {
            if (EditorUserBuildSettings.activeBuildTarget !=
                BuildTarget.Android)
                EditorUserBuildSettings.SwitchActiveBuildTarget(
                    group, BuildTarget.Android);
            PlayerSettings.SetApplicationIdentifier(group, PackageName);
            PlayerSettings.productName = ProductName;
            PlayerSettings.bundleVersion = "0.1.2";
            PlayerSettings.Android.bundleVersionCode = 12;
            PlayerSettings.SetScriptingBackend(
                group, ScriptingImplementation.IL2CPP);
            PlayerSettings.Android.targetArchitectures =
                AndroidArchitecture.ARM64;

            var options = new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = output,
                target = BuildTarget.Android,
                targetGroup = group,
                options = BuildOptions.CompressWithLz4HC
            };
            BuildReport report = BuildPipeline.BuildPlayer(options);
            if (report.summary.result != BuildResult.Succeeded)
                throw new InvalidOperationException(
                    "RoomSetupReplicaV2 B build failed: " +
                    report.summary.result);
            Debug.Log(
                "[RoomSetupReplicaV2BBuilder] package=" + PackageName +
                " output=" + output);
        }
        finally
        {
            PlayerSettings.SetApplicationIdentifier(group, oldIdentifier);
            PlayerSettings.productName = oldProductName;
            PlayerSettings.bundleVersion = oldVersion;
            PlayerSettings.Android.bundleVersionCode = oldVersionCode;
            PlayerSettings.SetScriptingBackend(group, oldBackend);
            PlayerSettings.Android.targetArchitectures = oldArchitectures;
        }
    }

    private static void DestroyAll<T>() where T : Component
    {
        foreach (T component in
                 UnityEngine.Object.FindObjectsByType<T>(
                     FindObjectsInactive.Include,
                     FindObjectsSortMode.None))
            UnityEngine.Object.DestroyImmediate(component);
    }
}

internal sealed class RoomSetupReplicaV2BManifestPostprocessor :
    IPostGenerateGradleAndroidProject
{
    private const string AndroidNamespace =
        "http://schemas.android.com/apk/res/android";

    public int callbackOrder => 10010;

    public void OnPostGenerateGradleAndroidProject(string path)
    {
        string identifier =
            PlayerSettings.GetApplicationIdentifier(BuildTargetGroup.Android);
        if (!string.Equals(
                identifier, RoomSetupReplicaV2BBuilder.PackageName,
                StringComparison.Ordinal))
            return;

        string[] candidates =
        {
            Path.Combine(path, "src", "main", "AndroidManifest.xml"),
            Path.Combine(
                path, "launcher", "src", "main", "AndroidManifest.xml"),
            Path.Combine(
                path, "unityLibrary", "src", "main", "AndroidManifest.xml")
        };
        foreach (string manifestPath in candidates)
        {
            if (File.Exists(manifestPath))
                PatchManifest(manifestPath);
        }
    }

    private static void PatchManifest(string path)
    {
        var document = new XmlDocument { PreserveWhitespace = true };
        document.Load(path);
        XmlElement root = document.DocumentElement;
        if (root == null) return;

        var namespaceManager =
            new XmlNamespaceManager(document.NameTable);
        namespaceManager.AddNamespace("android", AndroidNamespace);
        foreach (string permission in new[]
                 {
                     "android.permission.RECORD_AUDIO",
                     "horizonos.permission.HEADSET_CAMERA"
                 })
        {
            XmlNodeList nodes = root.SelectNodes(
                "uses-permission[@android:name='" + permission + "']",
                namespaceManager);
            if (nodes == null) continue;
            for (int i = nodes.Count - 1; i >= 0; i--)
                root.RemoveChild(nodes[i]);
        }

        XmlElement application =
            root.SelectSingleNode("application") as XmlElement;
        if (application != null)
        {
            XmlElement skipDialog = application.SelectSingleNode(
                "meta-data[@android:name='unityplayer.SkipPermissionsDialog']",
                namespaceManager) as XmlElement;
            if (skipDialog == null)
            {
                skipDialog = document.CreateElement("meta-data");
                application.AppendChild(skipDialog);
            }
            skipDialog.SetAttribute(
                "name", AndroidNamespace, "unityplayer.SkipPermissionsDialog");
            skipDialog.SetAttribute("value", AndroidNamespace, "true");
        }

        document.Save(path);
    }
}
