using System;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using Genesis.RoomScan;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using Object = UnityEngine.Object;

/// <summary>Creates a scene containing only capture plus the Room Setup replica.</summary>
public static class RoomSetupReplicaV2SceneSetup
{
    private const string ScenePath =
        "Assets/RoomSetupReplicaV2/Scenes/RoomSetupReplicaV2.unity";
    private const string ShaderDirectory =
        "Assets/QuestRoomScanStandalone/Shaders/";
    private const string CameraRigBlock =
        "e47682b9-c270-40b1-b16d-90b627a5ce1b";
    private const string PassthroughBlock =
        "f0540b20-dfd6-420e-b20d-c270f88dc77e";
    private const string CameraAccessBlock =
        "0792d3af-c7d9-4f9c-a6f0-fd580a051e48";

    [MenuItem("Room Setup Replica V2/重建独立测试场景")]
    public static async void SetupScene()
    {
        try
        {
            EditorSceneManager.NewScene(
                NewSceneSetup.EmptyScene, NewSceneMode.Single);
            await EnsureBuildingBlock(CameraRigBlock, "OVRCameraRig");
            await EnsureBuildingBlock(PassthroughBlock, "Passthrough");
            await EnsureBuildingBlock(CameraAccessBlock,
                "PassthroughCameraAccess");
            ConfigurePassthrough();
            EnsureArStack();
            GameObject root = EnsureReplicaRoot();
            ConfigureDepthCapture(root.GetComponent<DepthCapture>());
            EditorSceneManager.SaveScene(
                EditorSceneManager.GetActiveScene(), ScenePath);
            EditorBuildSettings.scenes = new[]
            {
                new EditorBuildSettingsScene(ScenePath, true)
            };
            AssetDatabase.SaveAssets();
            Debug.Log(
                "[RoomSetupReplicaV2] independent scene ready: " + ScenePath);
            EditorUtility.DisplayDialog(
                "Room Setup Replica V2",
                "独立场景已建立并设为唯一 Build Scene。\n" +
                "运行后会自动开始；右扳机暂停/继续，A 录制/停止，" +
                "Y 回放，B 清空，X 显示/隐藏。",
                "好");
        }
        catch (Exception exception)
        {
            Debug.LogError(
                "[RoomSetupReplicaV2] scene setup failed: " + exception);
            EditorUtility.DisplayDialog(
                "Room Setup Replica V2",
                "搭建失败，详情见 Console：\n" + exception.Message,
                "好");
        }
    }

    private static async Task EnsureBuildingBlock(
        string blockId, string label)
    {
        Assembly assembly = AppDomain.CurrentDomain.GetAssemblies()
            .FirstOrDefault(a =>
                a.GetName().Name == "Meta.XR.BuildingBlocks.Editor");
        if (assembly == null)
        {
            Debug.LogWarning(
                "[RoomSetupReplicaV2] Building Blocks assembly missing; " +
                "add " + label + " manually.");
            return;
        }
        Type utils = assembly.GetType(
            "Meta.XR.BuildingBlocks.Editor.Utils");
        MethodInfo getBlock = utils?.GetMethod(
            "GetBlockData",
            BindingFlags.Public | BindingFlags.NonPublic |
            BindingFlags.Static,
            null, new[] { typeof(string) }, null);
        object block = getBlock?.Invoke(null, new object[] { blockId });
        Type blockType = assembly.GetType(
            "Meta.XR.BuildingBlocks.Editor.BlockData");
        if (block == null || blockType == null)
        {
            Debug.LogWarning(
                "[RoomSetupReplicaV2] block unavailable: " + label);
            return;
        }
        PropertyInfo present = blockType.GetProperty(
            "IsSingletonAndAlreadyPresent",
            BindingFlags.Public | BindingFlags.NonPublic |
            BindingFlags.Instance);
        if (present != null && (bool)present.GetValue(block))
            return;
        MethodInfo install = blockType.GetMethods(
                BindingFlags.Public | BindingFlags.NonPublic |
                BindingFlags.Instance)
            .FirstOrDefault(method =>
                method.Name == "InstallWithDependencies" &&
                method.GetParameters().Length == 1 &&
                method.GetParameters()[0].ParameterType ==
                typeof(GameObject));
        if (install == null)
            return;
        try
        {
            if (install.Invoke(block, new object[] { null }) is Task task)
                await task;
        }
        catch (TargetInvocationException exception)
            when (exception.InnerException != null &&
                  exception.InnerException.GetType().Name ==
                  "InstallationCancelledException")
        {
            // Meta reports an already-present singleton as cancellation.
        }
    }

    private static void ConfigurePassthrough()
    {
        OVRManager manager = Object.FindAnyObjectByType<OVRManager>();
        if (manager != null)
        {
            manager.isInsightPassthroughEnabled = true;
            manager.shouldBoundaryVisibilityBeSuppressed = true;
            using var serialized = new SerializedObject(manager);
            SerializedProperty permission = serialized.FindProperty(
                "requestPassthroughCameraAccessPermissionOnStartup");
            if (permission != null)
                permission.boolValue = true;
            serialized.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(manager);
        }
        OVRCameraRig rig = Object.FindAnyObjectByType<OVRCameraRig>();
        Camera camera = rig != null && rig.centerEyeAnchor != null
            ? rig.centerEyeAnchor.GetComponent<Camera>()
            : Object.FindAnyObjectByType<Camera>();
        if (camera != null)
        {
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = Color.clear;
            EditorUtility.SetDirty(camera);
        }
    }

    private static void EnsureArStack()
    {
        if (Object.FindAnyObjectByType<ARSession>() == null)
            new GameObject("AR Session").AddComponent<ARSession>();
        OVRCameraRig rig = Object.FindAnyObjectByType<OVRCameraRig>();
        Camera camera = rig != null && rig.centerEyeAnchor != null
            ? rig.centerEyeAnchor.GetComponent<Camera>()
            : Object.FindAnyObjectByType<Camera>();
        if (camera == null)
            throw new InvalidOperationException(
                "No XR camera was created by Building Blocks.");
        if (camera.GetComponent<ARCameraManager>() == null)
            camera.gameObject.AddComponent<ARCameraManager>();
        if (camera.GetComponent<AROcclusionManager>() == null)
            camera.gameObject.AddComponent<AROcclusionManager>();
    }

    private static GameObject EnsureReplicaRoot()
    {
        RoomSetupReplicaV2Scanner existing =
            Object.FindAnyObjectByType<RoomSetupReplicaV2Scanner>();
        GameObject root = existing != null
            ? existing.gameObject
            : new GameObject("[Room Setup Replica V2] Scanner");
        if (existing == null)
            root.AddComponent<RoomSetupReplicaV2Scanner>();
        if (root.GetComponent<RoomSetupReplicaV2Input>() == null)
            root.AddComponent<RoomSetupReplicaV2Input>();
        return root;
    }

    private static void ConfigureDepthCapture(DepthCapture capture)
    {
        if (capture == null)
            throw new InvalidOperationException(
                "DepthCapture was not created by the replica scanner.");
        using var serialized = new SerializedObject(capture);
        AssignCompute(serialized, "depthNormalCompute",
            ShaderDirectory + "DepthNormals.compute");
        AssignCompute(serialized, "depthDilationCompute",
            ShaderDirectory + "DepthDilation.compute");
        SetBool(serialized, "enableSeedPlaneProductionExperiment", false);
        SetBool(serialized, "enableGunGelDualTestimony", false);
        SetBool(serialized, "enableTemporalFilter", false);
        SetBool(serialized, "enableHandMask", false);
        SetBool(serialized, "enableBilateralFilter", false);
        SetBool(serialized, "enableDepthEdgeClean", false);
        serialized.ApplyModifiedPropertiesWithoutUndo();
        EditorUtility.SetDirty(capture);
    }

    private static void AssignCompute(
        SerializedObject serialized, string field, string path)
    {
        SerializedProperty property = serialized.FindProperty(field);
        if (property == null)
            throw new MissingFieldException(typeof(DepthCapture).Name, field);
        ComputeShader shader =
            AssetDatabase.LoadAssetAtPath<ComputeShader>(path);
        if (shader == null)
            throw new InvalidOperationException("Missing compute shader: " + path);
        property.objectReferenceValue = shader;
    }

    private static void SetBool(
        SerializedObject serialized, string field, bool value)
    {
        SerializedProperty property = serialized.FindProperty(field);
        if (property != null)
            property.boolValue = value;
    }
}
