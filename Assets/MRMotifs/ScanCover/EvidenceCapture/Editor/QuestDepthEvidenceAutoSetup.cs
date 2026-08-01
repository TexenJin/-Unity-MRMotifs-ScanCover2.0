#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.Callbacks;
using UnityEngine;

namespace ScanCover.EvidenceCapture.Editor
{
    internal static class QuestDepthEvidenceAutoSetup
    {
        private const string TargetScene = "Assets/MRMotifs/ScanCover/EvidenceCapture/Scene/QuestDepthEvidenceCapture.unity";

        [DidReloadScripts]
        private static void AfterScriptsReloaded()
        {
            if (AssetDatabase.LoadAssetAtPath<SceneAsset>(TargetScene) != null) return;
            EditorApplication.delayCall += () =>
            {
                if (AssetDatabase.LoadAssetAtPath<SceneAsset>(TargetScene) == null)
                    QuestDepthEvidenceSceneBuilder.BuildOrRefreshScene();
            };
        }
    }
}
#endif
