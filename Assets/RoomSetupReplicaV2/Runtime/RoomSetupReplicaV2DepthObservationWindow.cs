using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Head-locked, read-only view of Meta's global stereo Environment Depth.
/// This component never owns, copies, masks, or writes depth data.
/// </summary>
[DefaultExecutionOrder(-10)]
[DisallowMultipleComponent]
public sealed class RoomSetupReplicaV2DepthObservationWindow : MonoBehaviour
{
    private const string ObservationShaderName =
        "RoomSetupReplicaV2/ReadOnlyDepthObservation";
    private static readonly int HasDepthId =
        Shader.PropertyToID("_HasDepth");

    [SerializeField] private bool visibleOnStart = true;
    [SerializeField] private Vector3 localPosition =
        new Vector3(0.27f, -0.14f, 0.70f);
    [SerializeField] private Vector2 windowSize =
        new Vector2(0.34f, 0.19f);

    public bool IsVisible => _visible;

    private Camera _camera;
    private GameObject _window;
    private Material _material;
    private bool _visible;
    private bool _missingShaderLogged;

    private void Awake()
    {
        _visible = visibleOnStart;
        ResolveCamera();
        EnsureWindow();
    }

    private void LateUpdate()
    {
        if (_camera == null)
            ResolveCamera();
        if (_window == null)
            EnsureWindow();

        if (_material != null)
        {
            Texture depth = Shader.GetGlobalTexture(
                "_EnvironmentDepthTexture");
            Vector4 zBuffer = Shader.GetGlobalVector(
                "_EnvironmentDepthZBufferParams");
            bool available = depth != null &&
                             depth.dimension == TextureDimension.Tex2DArray &&
                             Mathf.Abs(zBuffer.x) > 0.000001f;
            _material.SetFloat(HasDepthId, available ? 1f : 0f);
        }
    }

    public bool ToggleVisibility()
    {
        SetVisible(!_visible);
        return _visible;
    }

    public void SetVisible(bool visible)
    {
        _visible = visible;
        if (_window != null)
            _window.SetActive(visible);
    }

    private void ResolveCamera()
    {
        OVRCameraRig rig = FindAnyObjectByType<OVRCameraRig>();
        if (rig != null && rig.centerEyeAnchor != null)
            _camera = rig.centerEyeAnchor.GetComponent<Camera>();
        if (_camera == null)
            _camera = Camera.main;
        if (_camera == null)
            _camera = FindAnyObjectByType<Camera>();

        if (_camera != null && _window != null)
            AttachToCamera();
    }

    private void EnsureWindow()
    {
        if (_window != null || _camera == null)
            return;

        Shader shader = Shader.Find(ObservationShaderName);
        if (shader == null)
        {
            if (!_missingShaderLogged)
            {
                _missingShaderLogged = true;
                Debug.LogError(
                    "[RoomSetupReplicaV2] read-only depth observation " +
                    "shader is unavailable: " + ObservationShaderName);
            }
            return;
        }

        _material = new Material(shader)
        {
            name = "RoomSetup Read-Only Depth Observation",
            hideFlags = HideFlags.DontSave
        };
        _material.SetFloat(HasDepthId, 0f);

        _window = GameObject.CreatePrimitive(PrimitiveType.Quad);
        _window.name = "[RoomSetupReplicaV2] Read-Only Depth Observation";
        _window.hideFlags = HideFlags.DontSave;
        Collider collider = _window.GetComponent<Collider>();
        if (collider != null)
            Destroy(collider);

        MeshRenderer renderer = _window.GetComponent<MeshRenderer>();
        renderer.sharedMaterial = _material;
        renderer.shadowCastingMode = ShadowCastingMode.Off;
        renderer.receiveShadows = false;
        renderer.lightProbeUsage = LightProbeUsage.Off;
        renderer.reflectionProbeUsage = ReflectionProbeUsage.Off;
        renderer.motionVectorGenerationMode = MotionVectorGenerationMode.ForceNoMotion;

        AttachToCamera();
        _window.SetActive(_visible);
        Debug.Log(
            "[RoomSetupReplicaV2] read-only stereo depth observation " +
            "window ready; A toggles visibility; TSDF writes unchanged");
    }

    private void AttachToCamera()
    {
        Transform windowTransform = _window.transform;
        windowTransform.SetParent(_camera.transform, false);
        windowTransform.localPosition = localPosition;
        windowTransform.localRotation = Quaternion.identity;
        windowTransform.localScale =
            new Vector3(windowSize.x, windowSize.y, 1f);
    }

    private void OnDestroy()
    {
        if (_window != null)
            Destroy(_window);
        if (_material != null)
            Destroy(_material);
    }
}
