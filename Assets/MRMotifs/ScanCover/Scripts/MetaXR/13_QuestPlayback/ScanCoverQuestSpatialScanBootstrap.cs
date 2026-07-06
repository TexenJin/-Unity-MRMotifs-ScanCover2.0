using UnityEngine;

public static class ScanCoverQuestSpatialScanBootstrap
{
    private const string ResourcePath = "QuestSpatialScanDemo/office0_observer_scan_lite";
    private const bool AutoCreatePlayback = false;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void CreatePlaybackIfAvailable()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        if (!AutoCreatePlayback)
            return;

        TextAsset asset = Resources.Load<TextAsset>(ResourcePath);
        if (asset == null)
            return;

        if (Object.FindAnyObjectByType<ScanCoverQuestSpatialScanPlayback>() != null)
            return;

        GameObject root = new GameObject("[ScanCover] Quest Spatial Scan Playback");
        ScanCoverQuestSpatialScanPlayback playback = root.AddComponent<ScanCoverQuestSpatialScanPlayback>();
        root.transform.position = Vector3.zero;
        root.transform.rotation = Quaternion.identity;
        root.transform.localScale = Vector3.one;
        playback.RestartPlayback();
#endif
    }
}
