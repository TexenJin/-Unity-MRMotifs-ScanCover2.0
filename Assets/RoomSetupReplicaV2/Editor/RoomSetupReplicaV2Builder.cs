using System;
using System.IO;
using System.Xml;
using Meta.XR.EnvironmentDepth;
using UnityEditor;
using UnityEditor.Android;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.XR.ARFoundation;

internal static class RoomSetupReplicaV2Builder
{
    public const string PackageName =
        "com.pcaii.roomsetupreplicav2environmentteacher";
    public const string ProductName = "RoomSetup Replica V2 v0.1.3";
    public const string ScenePath =
        "Assets/RoomSetupReplicaV2/Scenes/" +
        "RoomSetupReplicaV2EnvironmentDepth.unity";
    [MenuItem("Codex/RoomSetup Replica V2/Prepare Scene")]
    public static void PrepareScene()
    {
        AssetDatabase.Refresh();
        if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(ScenePath) == null)
            throw new InvalidOperationException(
                "The production scene is missing: " + ScenePath);

        var scene = EditorSceneManager.OpenScene(
            ScenePath, OpenSceneMode.Single);

        RoomSetupReplicaV2Reconstruction[] reconstructions =
            UnityEngine.Object.FindObjectsByType<
                RoomSetupReplicaV2Reconstruction>(
                FindObjectsInactive.Include, FindObjectsSortMode.None);
        if (reconstructions.Length != 1)
            throw new InvalidOperationException(
                "RoomSetupReplicaV2 requires exactly one reconstruction; " +
                "found " + reconstructions.Length + ".");

        GameObject host = reconstructions[0].gameObject;
        RoomSetupReplicaV2EnvironmentDepthScanner scanner =
            host.GetComponent<RoomSetupReplicaV2EnvironmentDepthScanner>();
        RoomSetupReplicaV2EnvironmentDepthInput input =
            host.GetComponent<RoomSetupReplicaV2EnvironmentDepthInput>();
        if (scanner == null || input == null)
            throw new InvalidOperationException(
                "The production scene is missing its scanner or controls.");
        if (UnityEngine.Object.FindObjectsByType<
                RoomSetupReplicaV2EnvironmentDepthScanner>(
                FindObjectsInactive.Include,
                FindObjectsSortMode.None).Length != 1 ||
            UnityEngine.Object.FindObjectsByType<
                RoomSetupReplicaV2EnvironmentDepthInput>(
                FindObjectsInactive.Include,
                FindObjectsSortMode.None).Length != 1)
            throw new InvalidOperationException(
                "The production scene must contain exactly one producer.");

        EnvironmentDepthManager manager =
            host.GetComponent<EnvironmentDepthManager>();
        if (manager == null)
            throw new InvalidOperationException(
                "RoomSetupReplicaV2 source scene has no " +
                "EnvironmentDepthManager.");
        if (UnityEngine.Object.FindObjectsByType<EnvironmentDepthManager>(
                FindObjectsInactive.Include,
                FindObjectsSortMode.None).Length != 1 ||
            manager.enabled ||
            manager.OcclusionShadersMode != OcclusionShadersMode.None ||
            manager.RemoveHands || manager.MaskMeshFilters.Count != 0)
            throw new InvalidOperationException(
                "The production depth owner no longer matches v0.1.3.");
        if (UnityEngine.Object.FindObjectsByType<AROcclusionManager>(
                FindObjectsInactive.Include,
                FindObjectsSortMode.None).Length != 0)
            throw new InvalidOperationException(
                "RoomSetupReplicaV2 still contains a competing depth owner.");

        Debug.Log("[RoomSetupReplicaV2Builder] validated " + scene.path);
    }

    [MenuItem("Codex/RoomSetup Replica V2/Build APK")]
    public static void BuildApk()
    {
        PrepareScene();
        string output = Environment.GetEnvironmentVariable(
            "ROOM_SETUP_REPLICA_V2_APK");
        if (string.IsNullOrWhiteSpace(output))
            output = Path.GetFullPath(
                "Builds/RoomSetupReplicaV2/RoomSetupReplicaV2.apk");
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
            PlayerSettings.bundleVersion = "0.1.3";
            PlayerSettings.Android.bundleVersionCode = 13;
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
                    "RoomSetupReplicaV2 build failed: " +
                    report.summary.result);
            Debug.Log(
                "[RoomSetupReplicaV2Builder] package=" + PackageName +
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
}

internal sealed class RoomSetupReplicaV2ManifestPostprocessor :
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
                identifier, RoomSetupReplicaV2Builder.PackageName,
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
