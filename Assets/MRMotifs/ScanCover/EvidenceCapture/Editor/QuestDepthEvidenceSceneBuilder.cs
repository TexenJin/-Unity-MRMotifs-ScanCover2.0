#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Meta.XR.EnvironmentDepth;
using Meta.XR.MRUtilityKit;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace ScanCover.EvidenceCapture.Editor
{
    public static class QuestDepthEvidenceSceneBuilder
    {
        private const string SourceScene = "Assets/MRMotifs/ScanCover/Scene/DepthEffects_ScanCover.unity";
        private const string TargetScene = "Assets/MRMotifs/ScanCover/EvidenceCapture/Scene/QuestDepthEvidenceCapture.unity";
        private const string ComputePath = "Assets/MRMotifs/ScanCover/EvidenceCapture/Shaders/QuestDepthEvidenceCapture.compute";
        private const string CaptureObjectName = "[ScanCover] 深度证据采集";
        private const string SystemRoomMeshObjectName = "[ScanCover] 系统房间网格伴随采集";

        private static readonly HashSet<string> RequiredRootNames = new HashSet<string>(StringComparer.Ordinal)
        {
            "[BuildingBlock] Camera Rig",
            "[BuildingBlock] Passthrough",
            "[MR Motif] EnvironmentDepth",
            "Directional Light"
        };

        [MenuItem("ScanCover/深度证据/构建或刷新采集 Scene")]
        public static void BuildOrRefreshScene()
        {
            if (AssetDatabase.LoadAssetAtPath<SceneAsset>(SourceScene) == null)
            {
                throw new FileNotFoundException("找不到基础 XR Scene", SourceScene);
            }

            EnsureAssetFolder("Assets/MRMotifs/ScanCover/EvidenceCapture");
            EnsureAssetFolder("Assets/MRMotifs/ScanCover/EvidenceCapture/Scene");

            if (AssetDatabase.LoadAssetAtPath<SceneAsset>(TargetScene) == null)
            {
                if (!AssetDatabase.CopyAsset(SourceScene, TargetScene))
                {
                    throw new InvalidOperationException("无法建立采集 Scene 副本：" + TargetScene);
                }
                AssetDatabase.ImportAsset(TargetScene, ImportAssetOptions.ForceSynchronousImport);
            }

            Scene scene = EditorSceneManager.OpenScene(TargetScene, OpenSceneMode.Single);
            foreach (GameObject root in scene.GetRootGameObjects())
            {
                if (!RequiredRootNames.Contains(root.name) && root.name != CaptureObjectName && root.name != SystemRoomMeshObjectName)
                {
                    UnityEngine.Object.DestroyImmediate(root);
                }
            }

            GameObject captureRoot = scene.GetRootGameObjects().FirstOrDefault(root => root.name == CaptureObjectName);
            if (captureRoot == null)
            {
                captureRoot = new GameObject(CaptureObjectName);
            }

            QuestDepthEvidenceCaptureController controller = captureRoot.GetComponent<QuestDepthEvidenceCaptureController>();
            if (controller == null)
            {
                controller = captureRoot.AddComponent<QuestDepthEvidenceCaptureController>();
            }

            ComputeShader compute = AssetDatabase.LoadAssetAtPath<ComputeShader>(ComputePath);
            if (compute == null)
            {
                throw new InvalidOperationException("找不到深度证据 ComputeShader：" + ComputePath);
            }
            controller.ConfigureForScene(compute);

            GameObject roomMeshRoot = scene.GetRootGameObjects().FirstOrDefault(root => root.name == SystemRoomMeshObjectName);
            if (roomMeshRoot == null)
            {
                roomMeshRoot = new GameObject(SystemRoomMeshObjectName);
            }
            roomMeshRoot.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
            roomMeshRoot.transform.localScale = Vector3.one;

            MRUK mruk = roomMeshRoot.GetComponent<MRUK>();
            if (mruk == null)
            {
                mruk = roomMeshRoot.AddComponent<MRUK>();
            }
            if (mruk.SceneSettings == null)
            {
                mruk.SceneSettings = new MRUK.MRUKSettings();
            }
            mruk.SceneSettings.DataSource = MRUK.SceneDataSource.Device;
            mruk.SceneSettings.LoadSceneOnStartup = true;
            mruk.SceneSettings.EnableHighFidelityScene = false;

            ScanCoverMetaSceneMeshAuditExporter meshExporter = roomMeshRoot.GetComponent<ScanCoverMetaSceneMeshAuditExporter>();
            if (meshExporter == null)
            {
                meshExporter = roomMeshRoot.AddComponent<ScanCoverMetaSceneMeshAuditExporter>();
            }
            meshExporter.sceneMeshRoot = null;
            meshExporter.searchWholeSceneWhenRootMissing = false;
            meshExporter.includeInactiveChildren = true;
            meshExporter.requireRenderer = false;
            meshExporter.requireRendererEnabled = false;
            meshExporter.sessionPrefix = "ScanCover_SystemRoomMesh";
            meshExporter.exportLocalRawObjects = false;
            meshExporter.exportWorldAlignedObjects = false;
            meshExporter.exportCombinedWorldObj = true;
            meshExporter.exportComponentInventory = true;

            QuestSystemRoomMeshCapture roomMeshCapture = roomMeshRoot.GetComponent<QuestSystemRoomMeshCapture>();
            if (roomMeshCapture == null)
            {
                roomMeshCapture = roomMeshRoot.AddComponent<QuestSystemRoomMeshCapture>();
            }
            roomMeshCapture.ConfigureForScene(mruk, meshExporter);

            RemoveProductionScanCoverComponents(scene, controller);
            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene);
            PutCaptureSceneFirstInBuildSettings();
            ValidateScene(scene);

            Selection.activeGameObject = captureRoot;
            EditorGUIUtility.PingObject(captureRoot);
            Debug.Log("[深度证据] 专用采集 Scene 已构建并设为构建入口：" + TargetScene);
        }

        [MenuItem("ScanCover/深度证据/打开采集 Scene")]
        public static void OpenCaptureScene()
        {
            if (AssetDatabase.LoadAssetAtPath<SceneAsset>(TargetScene) == null)
            {
                BuildOrRefreshScene();
                return;
            }
            EditorSceneManager.OpenScene(TargetScene, OpenSceneMode.Single);
        }

        [MenuItem("ScanCover/深度证据/验证采集 Scene")]
        public static void ValidateOpenCaptureScene()
        {
            Scene scene = SceneManager.GetActiveScene();
            if (scene.path != TargetScene)
            {
                scene = EditorSceneManager.OpenScene(TargetScene, OpenSceneMode.Single);
            }
            ValidateScene(scene);
        }

        private static void RemoveProductionScanCoverComponents(Scene scene, QuestDepthEvidenceCaptureController keepController)
        {
            foreach (GameObject root in scene.GetRootGameObjects())
            {
                MonoBehaviour[] behaviours = root.GetComponentsInChildren<MonoBehaviour>(true);
                foreach (MonoBehaviour behaviour in behaviours)
                {
                    if (behaviour == null || behaviour == keepController) continue;
                    if (behaviour is QuestSystemRoomMeshCapture || behaviour is ScanCoverMetaSceneMeshAuditExporter) continue;
                    Type type = behaviour.GetType();
                    if (type.Namespace != null && type.Namespace.StartsWith("ScanCover.EvidenceCapture", StringComparison.Ordinal)) continue;

                    string typeName = type.Name;
                    bool isProductionScanCover = typeName.StartsWith("ScanCover", StringComparison.Ordinal) &&
                                                 !(behaviour is EnvironmentDepthManager);
                    if (isProductionScanCover)
                    {
                        UnityEngine.Object.DestroyImmediate(behaviour);
                    }
                }
            }
        }

        private static void PutCaptureSceneFirstInBuildSettings()
        {
            var scenes = new List<EditorBuildSettingsScene>
            {
                new EditorBuildSettingsScene(TargetScene, true)
            };
            foreach (EditorBuildSettingsScene existing in EditorBuildSettings.scenes)
            {
                if (existing.path == TargetScene) continue;
                scenes.Add(existing);
            }
            EditorBuildSettings.scenes = scenes.ToArray();
        }

        private static void ValidateScene(Scene scene)
        {
            var errors = new List<string>();
            GameObject[] roots = scene.GetRootGameObjects();
            foreach (string required in RequiredRootNames)
            {
                if (!roots.Any(root => root.name == required)) errors.Add("缺少根对象：" + required);
            }

            QuestDepthEvidenceCaptureController controller = UnityEngine.Object.FindAnyObjectByType<QuestDepthEvidenceCaptureController>(FindObjectsInactive.Include);
            if (controller == null) errors.Add("缺少 QuestDepthEvidenceCaptureController");
            if (UnityEngine.Object.FindAnyObjectByType<MRUK>(FindObjectsInactive.Include) == null)
                errors.Add("缺少 MRUK 系统房间网格加载器");
            if (UnityEngine.Object.FindAnyObjectByType<QuestSystemRoomMeshCapture>(FindObjectsInactive.Include) == null)
                errors.Add("缺少系统房间网格伴随采集器");
            if (UnityEngine.Object.FindAnyObjectByType<EnvironmentDepthManager>(FindObjectsInactive.Include) == null)
                errors.Add("缺少 EnvironmentDepthManager");
            if (Camera.main == null) errors.Add("缺少 MainCamera");

            MonoBehaviour[] behaviours = UnityEngine.Object.FindObjectsByType<MonoBehaviour>(FindObjectsInactive.Include, FindObjectsSortMode.None);
            foreach (MonoBehaviour behaviour in behaviours)
            {
                if (behaviour == null || behaviour == controller) continue;
                if (behaviour is QuestSystemRoomMeshCapture || behaviour is ScanCoverMetaSceneMeshAuditExporter) continue;
                string typeName = behaviour.GetType().Name;
                if (typeName.StartsWith("ScanCover", StringComparison.Ordinal))
                {
                    errors.Add("仍有生产组件：" + typeName);
                }
            }

            if (errors.Count > 0)
            {
                throw new InvalidOperationException("采集 Scene 验证失败：\n- " + string.Join("\n- ", errors));
            }

            Debug.Log("[深度证据] Scene 验证通过：保留 XR 相机、透视、环境深度、独立采集器与不可见系统房间网格伴随导出器。");
        }

        private static void EnsureAssetFolder(string path)
        {
            if (AssetDatabase.IsValidFolder(path)) return;
            string parent = Path.GetDirectoryName(path)?.Replace('\\', '/');
            string name = Path.GetFileName(path);
            if (!string.IsNullOrEmpty(parent)) EnsureAssetFolder(parent);
            AssetDatabase.CreateFolder(parent, name);
        }
    }
}
#endif
