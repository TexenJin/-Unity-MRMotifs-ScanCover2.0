using UnityEngine;

/// <summary>
/// 稳定的扫描工作中心（ScanWorkCenter）跟随器。
/// 作用：提供“调度中心”，不一定贴真实表面。
/// 建议挂在：SessionReferenceFrame/ScanWorkCenter
/// </summary>
public class ScanWorkCenterFollower : MonoBehaviour
{
    public enum FollowMode
    {
        HeadForward,     // 头前方固定距离
        FootProjection,  // 头显位置投影到参考平面（固定Y）
        Hybrid           // 位置=脚下投影，朝向=头部水平朝向
    }

    [Header("Mode")]
    public FollowMode followMode = FollowMode.Hybrid;

    [Header("References")]
    [Tooltip("稳定参考系根（建议 SessionReferenceFrame）")]
    public Transform sessionReferenceFrame;

    [Tooltip("头显（CenterEyeAnchor）；留空将尝试 Camera.main")]
    public Transform headTransform;

    [Header("HeadForward")]
    [Min(0f)] public float forwardDistance = 1.0f;
    public bool usePlanarForward = true;
    public bool followHeadHeightInHeadForward = false;
    [Tooltip("HeadForward: 若 followHeadHeightInHeadForward=false，则作为固定 localY；否则作为头高偏移")]
    public float headForwardYOffsetOrFixedY = 0.0f;

    [Header("FootProjection / Hybrid")]
    [Tooltip("投影到参考系平面（一般保持 true）")]
    public bool projectToReferencePlane = true;
    [Tooltip("投影目标平面 localY（通常 0）")]
    public float footOrHybridFixedLocalY = 0.0f;
    [Tooltip("平面偏移（local XZ）")]
    public Vector2 planarOffsetXZ = Vector2.zero;

    [Header("Clamp (optional)")]
    public bool clampPlanarRadius = false;
    [Min(0f)] public float minPlanarRadius = 0.15f;
    [Min(0f)] public float maxPlanarRadius = 3.0f;

    [Header("Smoothing")]
    public bool smoothPosition = true;
    [Min(0.001f)] public float positionSmoothTime = 0.08f;
    public bool smoothRotation = true;
    [Min(0f)] public float rotationLerpSpeed = 12f;

    [Header("Rotation")]
    public bool orientToForward = true;
    public bool keepUprightToReference = true;

    [Header("Visual Marker")]
    public bool enableMarker = true;
    public bool spawnMarkerAtRuntime = true;
    public GameObject markerPrefab;
    public bool autoCreateFallbackMarker = true;
    public bool markerParentToWorkCenter = true;
    public Vector3 markerLocalOffset = new Vector3(0f, 0.08f, 0f);
    public Vector3 markerWorldOffset = new Vector3(0f, 0.08f, 0f);
    public Vector3 markerLocalScale = new Vector3(0.08f, 0.08f, 0.08f);
    public bool colorizeMarkerByMode = true;
    public Color markerColorHeadForward = new Color(1.0f, 0.75f, 0.1f);
    public Color markerColorFootProjection = new Color(0.2f, 1.0f, 0.35f);
    public Color markerColorHybrid = new Color(0.1f, 0.9f, 1.0f);
    public bool showMarkerStatusInName = true;

    [Header("Debug")]
    public bool drawDebugRay = true;
    public Color debugRayColor = Color.cyan;
    public bool logWarnings = true;
    public bool drawGizmosSelected = true;
    public float gizmoRadius = 0.04f;

    private Vector3 _positionVelocity;
    private bool _warnedMissingRefs;

    private GameObject _markerInstance;
    private Renderer[] _markerRenderers;

    private void Reset()
    {
        if (transform.parent != null) sessionReferenceFrame = transform.parent;
        if (Camera.main != null) headTransform = Camera.main.transform;
    }

    private void Awake()
    {
        TryResolveReferences();
        EnsureMarker();
    }

    private void OnEnable()
    {
        TryResolveReferences();
        EnsureMarker();
        UpdateMarkerVisual(true);
    }

    private void OnValidate()
    {
        if (maxPlanarRadius < minPlanarRadius) maxPlanarRadius = minPlanarRadius;
    }

    private void LateUpdate()
    {
        bool ok = TryResolveReferences();
        if (!ok)
        {
            UpdateMarkerActive(false);
            return;
        }

        UpdateWorkCenter(Time.deltaTime);
        EnsureMarker();
        UpdateMarkerVisual(false);
        UpdateMarkerActive(true);
    }

    private bool TryResolveReferences()
    {
        if (sessionReferenceFrame == null && transform.parent != null)
            sessionReferenceFrame = transform.parent;

        if (headTransform == null && Camera.main != null)
            headTransform = Camera.main.transform;

        bool ok = sessionReferenceFrame != null && headTransform != null;
        if (!ok && logWarnings && !_warnedMissingRefs)
        {
            Debug.LogWarning(
                $"[{nameof(ScanWorkCenterFollower)}] Missing refs. sessionReferenceFrame={(sessionReferenceFrame ? sessionReferenceFrame.name : "null")}, headTransform={(headTransform ? headTransform.name : "null")}",
                this);
            _warnedMissingRefs = true;
        }
        if (ok) _warnedMissingRefs = false;
        return ok;
    }

    private void UpdateWorkCenter(float dt)
    {
        Transform rf = sessionReferenceFrame;
        Transform h = headTransform;

        Vector3 headLocalPos = rf.InverseTransformPoint(h.position);
        Vector3 headLocalForward = rf.InverseTransformDirection(h.forward);

        Vector3 planarForwardLocal = headLocalForward;
        if (usePlanarForward) planarForwardLocal.y = 0f;
        if (planarForwardLocal.sqrMagnitude < 1e-6f) planarForwardLocal = Vector3.forward;
        else planarForwardLocal.Normalize();

        Vector3 targetLocal;

        switch (followMode)
        {
            case FollowMode.HeadForward:
                {
                    targetLocal = headLocalPos + planarForwardLocal * forwardDistance;

                    if (followHeadHeightInHeadForward)
                        targetLocal.y = headLocalPos.y + headForwardYOffsetOrFixedY;
                    else
                        targetLocal.y = headForwardYOffsetOrFixedY;

                    targetLocal.x += planarOffsetXZ.x;
                    targetLocal.z += planarOffsetXZ.y;
                    break;
                }

            case FollowMode.FootProjection:
            case FollowMode.Hybrid:
            default:
                {
                    targetLocal = headLocalPos;
                    if (projectToReferencePlane)
                        targetLocal.y = footOrHybridFixedLocalY;

                    targetLocal.x += planarOffsetXZ.x;
                    targetLocal.z += planarOffsetXZ.y;
                    break;
                }
        }

        if (clampPlanarRadius)
        {
            Vector2 xz = new Vector2(targetLocal.x, targetLocal.z);
            float mag = xz.magnitude;
            if (mag > 1e-6f)
            {
                float clamped = Mathf.Clamp(mag, minPlanarRadius, maxPlanarRadius);
                if (!Mathf.Approximately(clamped, mag))
                {
                    xz = xz / mag * clamped;
                    targetLocal.x = xz.x;
                    targetLocal.z = xz.y;
                }
            }
            else
            {
                targetLocal.z = minPlanarRadius;
            }
        }

        bool isDirectChild = (transform.parent == rf);
        if (isDirectChild)
        {
            if (smoothPosition && Application.isPlaying)
            {
                transform.localPosition = Vector3.SmoothDamp(
                    transform.localPosition, targetLocal, ref _positionVelocity, positionSmoothTime, Mathf.Infinity, dt);
            }
            else
            {
                transform.localPosition = targetLocal;
            }
        }
        else
        {
            Vector3 targetWorld = rf.TransformPoint(targetLocal);
            if (smoothPosition && Application.isPlaying)
            {
                transform.position = Vector3.SmoothDamp(
                    transform.position, targetWorld, ref _positionVelocity, positionSmoothTime, Mathf.Infinity, dt);
            }
            else
            {
                transform.position = targetWorld;
            }
        }

        if (orientToForward)
        {
            Vector3 worldForward = rf.TransformDirection(planarForwardLocal);
            if (worldForward.sqrMagnitude > 1e-6f)
            {
                Vector3 up = keepUprightToReference ? rf.up : Vector3.up;
                Quaternion targetRot = Quaternion.LookRotation(worldForward.normalized, up);

                if (smoothRotation && Application.isPlaying)
                {
                    float t = 1f - Mathf.Exp(-rotationLerpSpeed * dt);
                    transform.rotation = Quaternion.Slerp(transform.rotation, targetRot, t);
                }
                else
                {
                    transform.rotation = targetRot;
                }
            }
        }

        if (drawDebugRay && h != null)
            Debug.DrawLine(h.position, transform.position, debugRayColor);
    }

    // ---------------- Marker ----------------

    private void EnsureMarker()
    {
        if (!enableMarker || !spawnMarkerAtRuntime) return;
        if (_markerInstance != null) return;

        if (markerPrefab != null)
        {
            Transform parent = markerParentToWorkCenter ? transform : null;
            _markerInstance = Instantiate(markerPrefab, parent);
            _markerInstance.name = "ScanWorkCenter_Marker";
        }
        else if (autoCreateFallbackMarker)
        {
            _markerInstance = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            _markerInstance.name = "ScanWorkCenter_Marker(Fallback)";
            Collider c = _markerInstance.GetComponent<Collider>();
            if (c != null)
            {
                if (Application.isPlaying) Destroy(c);
                else DestroyImmediate(c);
            }
            if (markerParentToWorkCenter) _markerInstance.transform.SetParent(transform, false);
        }

        if (_markerInstance != null)
            _markerRenderers = _markerInstance.GetComponentsInChildren<Renderer>(true);
    }

    private void UpdateMarkerVisual(bool force)
    {
        if (_markerInstance == null) return;

        Transform mt = _markerInstance.transform;
        if (markerParentToWorkCenter)
        {
            if (mt.parent != transform) mt.SetParent(transform, false);
            mt.localPosition = markerLocalOffset;
            mt.localRotation = Quaternion.identity;
            mt.localScale = markerLocalScale;
        }
        else
        {
            if (mt.parent != null) mt.SetParent(null, true);
            mt.position = transform.position + markerWorldOffset;
            mt.rotation = Quaternion.identity;
            mt.localScale = markerLocalScale;
        }

        if (colorizeMarkerByMode && _markerRenderers != null)
        {
            Color c = GetModeColor(followMode);
            foreach (var r in _markerRenderers)
            {
                if (r == null) continue;
                try
                {
                    var m = r.material;
                    if (m != null)
                    {
                        if (m.HasProperty("_Color")) m.color = c;
                        if (m.HasProperty("_EmissionColor"))
                        {
                            m.EnableKeyword("_EMISSION");
                            m.SetColor("_EmissionColor", c * 0.4f);
                        }
                    }
                }
                catch { }
            }
        }

        if (showMarkerStatusInName)
        {
            _markerInstance.name = $"ScanWorkCenter_Marker[{followMode}|{(enabled ? "Following" : "Paused")}]";
        }
    }

    private void UpdateMarkerActive(bool active)
    {
        if (_markerInstance != null && _markerInstance.activeSelf != active)
            _markerInstance.SetActive(active);
    }

    private Color GetModeColor(FollowMode mode)
    {
        switch (mode)
        {
            case FollowMode.HeadForward: return markerColorHeadForward;
            case FollowMode.FootProjection: return markerColorFootProjection;
            case FollowMode.Hybrid:
            default: return markerColorHybrid;
        }
    }

#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        if (!drawGizmosSelected) return;
        Gizmos.color = debugRayColor;
        Gizmos.DrawWireSphere(transform.position, gizmoRadius);
        if (headTransform != null) Gizmos.DrawLine(headTransform.position, transform.position);
    }
#endif
}