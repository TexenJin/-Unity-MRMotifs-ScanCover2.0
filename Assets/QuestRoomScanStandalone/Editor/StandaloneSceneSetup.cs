using System;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using Object = UnityEngine.Object;

namespace Genesis.RoomScan.Editor
{
    /// <summary>
    /// QRS 独立测试链的一键场景搭建向导（移植自 QuestRoomScan 官方
    /// RoomScanSetupWizard 的最小链路部分）。
    ///
    /// 菜单：QRS独立链 → 一键搭建测试场景
    ///
    /// 做的事（与官方 Quick Start 完全同构）：
    ///   1. 新建空白场景（覆盖掉之前手写的 YAML 场景，杜绝序列化陷阱）
    ///   2. 通过 Meta Building Blocks 管线安装：OVRCameraRig / 透视层 / PassthroughCameraAccess
    ///   3. 透视三件套修正：OVRManager.isInsightPassthroughEnabled、
    ///      中心眼相机透明清屏、启动时请求 HEADSET_CAMERA 权限
    ///   4. AR Session + 中心眼相机上的 ARCameraManager + AROcclusionManager
    ///   5. 根节点挂 StandaloneRoomScanner（RequireComponent 自动带 DepthCapture/
    ///      VolumeIntegrator/MeshExtractor）+ PassthroughCameraProvider + StandaloneScanInput
    ///   6. 用 SerializedObject 接线全部 compute shader 与 ScanMesh 材质
    ///   7. 保存场景、确保在 Build Settings 里
    /// </summary>
    public static class StandaloneSceneSetup
    {
        const string SCENE_PATH = "Assets/QuestRoomScanStandalone/Scenes/QuestRoomScanStandalone.unity";
        const string SHADER_DIR = "Assets/QuestRoomScanStandalone/Shaders/";
        const string MAT_PATH = "Assets/QuestRoomScanStandalone/Materials/ScanMesh.mat";

        // Meta XR Building Blocks 的块 ID（与 Meta SDK BlockDataIds 一致，
        // 抄自 QRS RoomScanSetupWizard.BuildingBlocks.cs）
        const string BB_CAMERA_RIG = "e47682b9-c270-40b1-b16d-90b627a5ce1b";
        const string BB_PASSTHROUGH = "f0540b20-dfd6-420e-b20d-c270f88dc77e";
        const string BB_PCA = "0792d3af-c7d9-4f9c-a6f0-fd580a051e48";

        [MenuItem("QRS独立链/一键搭建测试场景")]
        public static async void SetupScene()
        {
            try
            {
                Debug.Log("[QRS独立链] 开始搭建场景…");

                // 1. 全新空白场景——覆盖手写 YAML，之后由 Unity 自己序列化
                EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

                // 2. Building Blocks：相机装置 + 透视 + PCA（幂等）
                await EnsureBuildingBlock(BB_CAMERA_RIG, "OVRCameraRig");
                await EnsureBuildingBlock(BB_PASSTHROUGH, "Passthrough");
                await EnsureBuildingBlock(BB_PCA, "PassthroughCameraAccess");

                // 3. 透现场景修正（Meta 自己的 Outstanding Issues 检查项）
                EnsurePassthroughConfig();

                // 4. AR Session + 遮挡（深度）管理器
                EnsureARStack();

                // 5. 扫描链根节点
                var root = EnsureScannerRoot();

                // 6. 接线 compute shader / 材质
                WireComponents(root);

                // 7. 保存 + Build Settings
                SaveAndRegisterScene();

                Debug.Log("[QRS独立链] ✅ 场景搭建完成：" + SCENE_PATH +
                          "\n下一步：Build Settings 里把该场景拖到最顶（或只勾它），然后打包。");
                EditorUtility.DisplayDialog("QRS独立链",
                    "场景搭建完成。\n\n请在 Build Settings 把 QuestRoomScanStandalone 拖到最顶（或只勾它），然后打包。",
                    "好");
            }
            catch (Exception e)
            {
                Debug.LogError("[QRS独立链] 场景搭建失败：" + e);
                EditorUtility.DisplayDialog("QRS独立链", "搭建失败，详情见 Console：\n" + e.Message, "好");
            }
        }

        // ── 2. Building Blocks（反射驱动 Meta 的内部安装 API，与 QRS 向导同法）──

        static async Task EnsureBuildingBlock(string blockId, string label)
        {
            var asm = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetName().Name == "Meta.XR.BuildingBlocks.Editor");
            if (asm == null)
            {
                Debug.LogWarning($"[QRS独立链] Meta.XR.BuildingBlocks.Editor 程序集未加载，跳过 {label}。" +
                                 "请手动用 Menu > Meta > Building Blocks 添加。");
                return;
            }

            var utilsType = asm.GetType("Meta.XR.BuildingBlocks.Editor.Utils");
            var getBlockData = utilsType?.GetMethod("GetBlockData",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static,
                null, new[] { typeof(string) }, null);
            var blockData = getBlockData?.Invoke(null, new object[] { blockId });
            if (blockData == null)
            {
                Debug.LogWarning($"[QRS独立链] Building Block '{label}' 未在注册表找到，跳过。");
                return;
            }

            var blockDataType = asm.GetType("Meta.XR.BuildingBlocks.Editor.BlockData");
            var isPresent = blockDataType?.GetProperty("IsSingletonAndAlreadyPresent",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (isPresent != null && (bool)isPresent.GetValue(blockData))
            {
                Debug.Log($"[QRS独立链] {label} 已存在，跳过安装。");
                return;
            }

            var install = blockDataType?.GetMethods(
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                .FirstOrDefault(m => m.Name == "InstallWithDependencies"
                                     && m.GetParameters().Length == 1
                                     && m.GetParameters()[0].ParameterType == typeof(GameObject));
            if (install == null)
            {
                Debug.LogWarning($"[QRS独立链] InstallWithDependencies 解析失败，跳过 {label}。");
                return;
            }

            try
            {
                if (install.Invoke(blockData, new object[] { null }) is Task t) await t;
                Debug.Log($"[QRS独立链] 已安装 Building Block：{label}");
            }
            catch (TargetInvocationException tex) when (tex.InnerException != null
                && tex.InnerException.GetType().Name == "InstallationCancelledException")
            {
                Debug.Log($"[QRS独立链] {label} 安装被取消（通常=单例已存在），视为成功。");
            }
        }

        // ── 3. 透视三件套 ──

        static void EnsurePassthroughConfig()
        {
            var ovrManager = Object.FindAnyObjectByType<OVRManager>();
            if (ovrManager == null)
            {
                Debug.LogWarning("[QRS独立链] 场景里没有 OVRManager（相机装置块可能没装上）。");
                return;
            }

            if (!ovrManager.isInsightPassthroughEnabled)
            {
                ovrManager.isInsightPassthroughEnabled = true;
                EditorUtility.SetDirty(ovrManager);
                Debug.Log("[QRS独立链] 已开启 OVRManager.isInsightPassthroughEnabled。");
            }

            // 扫描要满屋走动：抑制边界蓝网（走近 Guardian 不再弹网格墙）。
            // 注意：这只是应用侧能关的部分——走出安全区时系统强制切透视是
            // OS 级安全行为，应用关不掉；要真正自由行走需在头显设置里
            // 改大房间级边界或开发者模式关闭边界（见 HUD/文档说明）。
            if (!ovrManager.shouldBoundaryVisibilityBeSuppressed)
            {
                ovrManager.shouldBoundaryVisibilityBeSuppressed = true;
                EditorUtility.SetDirty(ovrManager);
                Debug.Log("[QRS独立链] 已开启 OVRManager.shouldBoundaryVisibilityBeSuppressed（边界蓝网抑制）。");
            }

            // 启动时请求 HEADSET_CAMERA 权限（该字段是 internal，走序列化层写）
            using (var so = new SerializedObject(ovrManager))
            {
                var prop = so.FindProperty("requestPassthroughCameraAccessPermissionOnStartup");
                if (prop != null && !prop.boolValue)
                {
                    prop.boolValue = true;
                    so.ApplyModifiedPropertiesWithoutUndo();
                    EditorUtility.SetDirty(ovrManager);
                    Debug.Log("[QRS独立链] 已开启 OVRManager.requestPassthroughCameraAccessPermissionOnStartup。");
                }
            }

            var rig = Object.FindAnyObjectByType<OVRCameraRig>();
            if (rig != null && rig.centerEyeAnchor != null)
            {
                var cam = rig.centerEyeAnchor.GetComponent<Camera>();
                if (cam != null && (cam.clearFlags != CameraClearFlags.SolidColor || cam.backgroundColor.a >= 1f))
                {
                    cam.clearFlags = CameraClearFlags.SolidColor;
                    cam.backgroundColor = Color.clear;
                    EditorUtility.SetDirty(cam);
                    Debug.Log("[QRS独立链] 中心眼相机已设为透明清屏（透视 Underlay 前提）。");
                }
            }
        }

        // ── 4. AR Session + 遮挡管理器（挂在中心眼相机上，与 QRS 向导一致）──

        static void EnsureARStack()
        {
            if (Object.FindAnyObjectByType<ARSession>() == null)
            {
                var go = new GameObject("AR Session");
                go.AddComponent<ARSession>();
                Debug.Log("[QRS独立链] 已添加 ARSession。");
            }

            var rig = Object.FindAnyObjectByType<OVRCameraRig>();
            Camera cam = rig != null && rig.centerEyeAnchor != null
                ? rig.centerEyeAnchor.GetComponent<Camera>()
                : Object.FindAnyObjectByType<Camera>();
            if (cam == null)
            {
                Debug.LogWarning("[QRS独立链] 找不到相机，AROcclusionManager 未添加。");
                return;
            }

            // QRS 向导注释：AROcclusionManager 需要 ARCameraManager 才能工作
            if (cam.GetComponent<ARCameraManager>() == null)
            {
                cam.gameObject.AddComponent<ARCameraManager>();
                Debug.Log("[QRS独立链] 已在中心眼相机添加 ARCameraManager。");
            }
            if (cam.GetComponent<AROcclusionManager>() == null)
            {
                // 与 QRS 真实场景一致：初始保持启用（环境深度时域平滑默认开），
                // DepthCapture.Start 会自己禁用、待 USE_SCENE 权限确认后再开。
                // 不要预先禁用——那是我们手写 YAML 时代的猜法。
                cam.gameObject.AddComponent<AROcclusionManager>();
                Debug.Log("[QRS独立链] 已在中心眼相机添加 AROcclusionManager（初始启用，DepthCapture 自管开关）。");
            }
        }

        // ── 5. 扫描链根节点 ──

        static GameObject EnsureScannerRoot()
        {
            var root = Object.FindAnyObjectByType<StandaloneRoomScanner>()?.gameObject;
            if (root == null)
            {
                root = new GameObject("[QRS Standalone] Room Scan");
                // AddComponent<StandaloneRoomScanner> 会经 RequireComponent 自动带上
                // DepthCapture / VolumeIntegrator / MeshExtractor，序列化默认值全部生效
                root.AddComponent<StandaloneRoomScanner>();
                Debug.Log("[QRS独立链] 已创建扫描链根节点（StandaloneRoomScanner + 核心三件套）。");
            }

            if (root.GetComponent<PassthroughCameraProvider>() == null)
            {
                root.AddComponent<PassthroughCameraProvider>();
                Debug.Log("[QRS独立链] 已添加 PassthroughCameraProvider。");
            }
            if (root.GetComponent<StandaloneScanInput>() == null)
            {
                root.AddComponent<StandaloneScanInput>(); // controller 字段默认 RTouch，代码生效无 YAML 陷阱
                Debug.Log("[QRS独立链] 已添加 StandaloneScanInput（右手柄）。");
            }
            return root;
        }

        // ── 6. 接线（SerializedObject，与 QRS WireComponent 同法；只填空槽）──

        static void WireComponents(GameObject root)
        {
            var dc = root.GetComponent<DepthCapture>();
            if (dc != null)
            {
                var so = new SerializedObject(dc);
                AssignCompute(so, "depthNormalCompute", SHADER_DIR + "DepthNormals.compute");
                AssignCompute(so, "depthDilationCompute", SHADER_DIR + "DepthDilation.compute");
                AssignCompute(so, "bilateralFilterCompute", SHADER_DIR + "BilateralDepthFilter.compute");
                so.ApplyModifiedPropertiesWithoutUndo();
                EditorUtility.SetDirty(dc);
            }

            var vi = root.GetComponent<VolumeIntegrator>();
            if (vi != null)
            {
                var so = new SerializedObject(vi);
                AssignCompute(so, "compute", SHADER_DIR + "VolumeIntegration.compute");
                so.ApplyModifiedPropertiesWithoutUndo();
                EditorUtility.SetDirty(vi);
            }

            var me = root.GetComponent<MeshExtractor>();
            if (me != null)
            {
                var so = new SerializedObject(me);
                AssignCompute(so, "surfaceNetsCompute", SHADER_DIR + "SurfaceNetsExtract.compute");
                var matProp = so.FindProperty("scanMeshMaterial");
                if (matProp != null && matProp.objectReferenceValue == null)
                {
                    var mat = AssetDatabase.LoadAssetAtPath<Material>(MAT_PATH);
                    if (mat != null) matProp.objectReferenceValue = mat;
                    else Debug.LogWarning("[QRS独立链] 找不到材质 " + MAT_PATH);
                }
                so.ApplyModifiedPropertiesWithoutUndo();
                EditorUtility.SetDirty(me);
            }

            Debug.Log("[QRS独立链] compute shader / 材质接线完成。");
        }

        static void AssignCompute(SerializedObject so, string fieldName, string assetPath)
        {
            var prop = so.FindProperty(fieldName);
            if (prop == null)
            {
                Debug.LogWarning($"[QRS独立链] 字段 {fieldName} 不存在，跳过。");
                return;
            }
            var asset = AssetDatabase.LoadAssetAtPath<ComputeShader>(assetPath);
            if (asset == null)
            {
                Debug.LogWarning("[QRS独立链] 找不到 " + assetPath);
                return;
            }
            prop.objectReferenceValue = asset; // 强制重写，不信旧值
        }

        // ── 7. 保存 + Build Settings ──

        static void SaveAndRegisterScene()
        {
            var scene = EditorSceneManager.GetActiveScene();
            var dir = System.IO.Path.GetDirectoryName(SCENE_PATH);
            if (!AssetDatabase.IsValidFolder(dir))
            {
                var parent = System.IO.Path.GetDirectoryName(dir).Replace('\\', '/');
                AssetDatabase.CreateFolder(parent, System.IO.Path.GetFileName(dir));
            }
            EditorSceneManager.SaveScene(scene, SCENE_PATH);

            var scenes = EditorBuildSettings.scenes.ToList();
            if (!scenes.Any(s => s.path == SCENE_PATH))
            {
                scenes.Add(new EditorBuildSettingsScene(SCENE_PATH, true));
                EditorBuildSettings.scenes = scenes.ToArray();
                Debug.Log("[QRS独立链] 场景已加入 Build Settings。");
            }
        }
    }
}
