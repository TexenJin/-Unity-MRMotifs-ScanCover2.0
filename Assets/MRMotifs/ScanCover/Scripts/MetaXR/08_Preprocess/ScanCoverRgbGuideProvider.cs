using System;
using Meta.XR;
using UnityEngine;
#if UNITY_ANDROID && !UNITY_EDITOR
using UnityEngine.Android;
#endif

/// <summary>
/// Supplies passthrough RGB frames to the depth preprocessor for the
/// QRS-style RGB-guided joint bilateral filter.  Mirrors the proven
/// QuestRoomScanStandalone PassthroughCameraProvider: scene-wide PCA
/// adoption (never a second instance racing for the camera handle),
/// permission-first startup, graceful degradation — when no frame is
/// available the preprocessor simply falls back to the depth-only term.
/// Runtime-created by ScanCoverDepthPreprocessor; no scene wiring needed.
/// </summary>
[DisallowMultipleComponent]
public sealed class ScanCoverRgbGuideProvider : MonoBehaviour
{
    public const string CameraPermissionId = "horizonos.permission.HEADSET_CAMERA";

    [SerializeField] private PassthroughCameraAccess.CameraPositionType cameraPosition =
        PassthroughCameraAccess.CameraPositionType.Right;
    [SerializeField] private Vector2Int requestedResolution = new Vector2Int(1280, 960);
    [SerializeField] private int maxFramerate = 30;

    private PassthroughCameraAccess _pca;
    private bool _startRequested;

    public string LastIssue { get; private set; }

    public PassthroughCameraAccess.CameraPositionType CameraPosition => cameraPosition;

    public bool IsReady
    {
        get
        {
            try
            {
                return _pca != null && _pca.IsPlaying && _pca.IsUpdatedThisFrame;
            }
            catch (Exception)
            {
                return false;
            }
        }
    }

    public Texture CurrentFrame
    {
        get
        {
            try
            {
                return _pca != null && _pca.IsPlaying ? _pca.GetTexture() : null;
            }
            catch (Exception)
            {
                return null;
            }
        }
    }

    public void SetCameraPosition(PassthroughCameraAccess.CameraPositionType position)
    {
        if (cameraPosition == position)
            return;
        cameraPosition = position;
        if (_pca != null && !_pca.IsPlaying)
            _pca.CameraPosition = position;
    }

    public void StartCapture()
    {
        if (_startRequested)
            return;
        _startRequested = true;

#if UNITY_ANDROID && !UNITY_EDITOR
        if (!Permission.HasUserAuthorizedPermission(CameraPermissionId))
            Permission.RequestUserPermission(CameraPermissionId);
#endif

        try
        {
            // Scene-wide find: never create a second PCA for the same camera
            // position — the native side allows exactly one instance and a
            // racer self-disables both (learned in the standalone chain).
            if (_pca == null)
            {
                _pca = FindAnyObjectByType<PassthroughCameraAccess>(FindObjectsInactive.Include);
                if (_pca == null)
                    _pca = gameObject.AddComponent<PassthroughCameraAccess>();
            }

            bool wasEnabled = _pca.enabled;
            if (wasEnabled)
                _pca.enabled = false;
            _pca.CameraPosition = cameraPosition;
            _pca.RequestedResolution = requestedResolution;
            _pca.MaxFramerate = maxFramerate;
            _pca.enabled = true;
            LastIssue = null;
        }
        catch (Exception ex)
        {
            LastIssue = $"RGB guide PCA startup failed: {ex.Message}";
        }
    }

    private void OnDestroy()
    {
        if (_pca != null)
            _pca.enabled = false;
    }
}
