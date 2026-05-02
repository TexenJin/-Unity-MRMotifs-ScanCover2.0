using UnityEngine;
using Meta.XR;

/// <summary>
/// 将一个“输出点”吸附到真实环境表面（床/楼梯/桌面等）。
/// 参考 MRMotifs 的 SurfacePlacement 思路：EnvironmentRaycastManager 向下射线。
///
/// 建议挂在：SessionReferenceFrame/SurfaceSnapPoint
/// 依赖：场景中有 EnvironmentRaycastManager（MRUK）
/// 输入：ScanWorkCenter（由 ScanWorkCenterFollower 驱动）
/// 输出：本对象 transform（SurfaceSnapPoint）
/// </summary>
public class ScanSurfaceSnapper : MonoBehaviour
{
    public enum MissPolicy
    {
        FallbackToWorkCenter, // 未命中时回退到 ScanWorkCenter（最稳）
        HoldLastValidHit,     // 未命中时保持上一次有效命中
        DisableMarkerOnly     // 未命中时不改位置，但隐藏/变色 marker
    }

    public enum RotationMode
    {
        KeepWorkCenterYawUpright,          // 保持 work center 的水平朝向
        AlignUpToSurfaceNormalKeepForward, // up 对齐法线，forward 尽量跟随头部/工作中心方向
        FaceSurfaceNormal                  // forward = normal（调试用）
    }

    [Header("References")]
    [Tooltip("稳定参考系（建议 SessionReferenceFrame）")]
    public Transform sessionReferenceFrame;

    [Tooltip("调度中心（ScanWorkCenter）")]
    public Transform scanWorkCenter;

    [Tooltip("头显（CenterEyeAnchor，可选，用于旋转参考）")]
    public Transform headTransform;

    [Tooltip("MRUK 环境射线管理器；留空自动查找")]
    public EnvironmentRaycastManager raycastManager;

    [Header("Raycast")]
    [Tooltip("从 ScanWorkCenter 上方多少米开始向下打射线")]
    [Min(0f)] public float rayStartHeight = 1.2f;

    [Tooltip("最多允许吸附的垂直高度差（避免误命中过远层）")]
    [Min(0.01f)] public float maxVerticalSnapDistance = 2.5f;

    [Tooltip("命中后沿法线抬起一点，避免视觉穿插")]
    public float surfaceOffset = 0.01f;

    [Tooltip("向下方向使用参考系 -up（推荐）；否则使用世界 Vector3.down")]
    public bool useReferenceDown = true;

    [Tooltip("仅允许法线朝上的表面（床/桌面/台阶）。false 则墙面也可吸附")]
    public bool requireMostlyUpFacingSurface = true;

    [Range(-1f, 1f)]
    [Tooltip("表面法线与“上方向”点积阈值。0.5≈允许最高约60°坡面")]
    public float minUpDot = 0.35f;

    [Header("Miss Handling")]
    public MissPolicy missPolicy = MissPolicy.FallbackToWorkCenter;

    [Tooltip("回退到 WorkCenter 时，附加沿参考上方向偏移")]
    public float fallbackOffset = 0.0f;

    [Header("Smoothing")]
    public bool smoothPosition = true;
    [Min(0.001f)] public float positionSmoothTime = 0.05f;

    public bool smoothRotation = true;
    [Min(0f)] public float rotationLerpSpeed = 14f;

    [Header("Rotation")]
    public RotationMode rotationMode = RotationMode.AlignUpToSurfaceNormalKeepForward;
    public bool useHeadForwardAsRotationHint = true;

    [Header("Visual Marker")]
    public bool enableMarker = true;
    public bool spawnMarkerAtRuntime = true;
    public GameObject markerPrefab;
    public bool autoCreateFallbackMarker = true;

    [Tooltip("true=marker 挂在 SurfaceSnapPoint 下；false=独立放置到世界坐标")]
    public bool markerParentToSnapPoint = true;

    public Vector3 markerLocalOffset = new Vector3(0f, 0.03f, 0f);
    public Vector3 markerWorldOffset = new Vector3(0f, 0.03f, 0f);
    public Vector3 markerLocalScale = new Vector3(0.06f, 0.06f, 0.06f);

    public bool showMarkerStatusInName = true;
    public bool colorizeMarker = true;
    public Color hitColor = new Color(0.2f, 1f, 0.35f);
    public Color missColor = new Color(1f, 0.35f, 0.2f);
    public Color unsupportedColor = new Color(1f, 0.15f, 1f);

    [Header("Marker Upright Lock (NEW)")]
    [Tooltip("让 marker 始终保持竖直（锁定 pitch/roll）")]
    public bool markerKeepUpright = true;

    [Tooltip("竖直方向使用参考系 up（推荐）；否则用世界 Vector3.up")]
    public bool markerUseReferenceUp = true;

    [Tooltip("保留 marker 的水平朝向(yaw)。false 则使用固定朝向。")]
    public bool markerKeepYaw = true;

    [Tooltip("markerKeepYaw=true 时：优先用 ScanWorkCenter 的 forward 作为 yaw 来源；否则用 headTransform")]
    public bool markerYawFromWorkCenter = true;

    [Tooltip("如果 prefab 正面轴不是 +Z，可在这里补一个旋转偏移")]
    public Vector3 markerRotationOffsetEuler = Vector3.zero;

    [Header("Debug Line")]
    public bool drawDebugRay = true;
    public Color debugRayColor = Color.yellow;
    public bool useLineRendererIndicator = true;
    public Material lineMaterial;
    public float lineWidth = 0.008f;

    [Header("Debug")]
    public bool logWarnings = true;
    public bool drawGizmosSelected = true;
    public float gizmoRadius = 0.03f;

    // 状态（只读调试）
    [System.NonSerialized] public bool hasValidHit;
    [System.NonSerialized] public Vector3 lastHitPoint;
    [System.NonSerialized] public Vector3 lastHitNormal = Vector3.up;
    [System.NonSerialized] public float lastVerticalDistance;

    private Vector3 _positionVelocity;
    private bool _warnedRefs;
    private bool _warnedRayMgr;
    private bool _warnedSupport;

    private GameObject _markerInstance;
    private Renderer[] _markerRenderers;
    private LineRenderer _lineRenderer;

    private void Reset()
    {
        if (transform.parent != null) sessionReferenceFrame = transform.parent;
        if (Camera.main != null) headTransform = Camera.main.transform;
    }

    private void Awake()
    {
        TryResolveRefs();
        EnsureMarker();
        EnsureLineRenderer();
    }

    private void OnEnable()
    {
        TryResolveRefs();
        EnsureMarker();
        EnsureLineRenderer();
        UpdateVisuals();
    }

    private void LateUpdate()
    {
        bool ok = TryResolveRefs();
        if (!ok)
        {
            UpdateVisuals();
            return;
        }

        bool supported = EnvironmentRaycastManager.IsSupported;
        if (!supported)
        {
            if (logWarnings && !_warnedSupport)
            {
                Debug.LogWarning($"[{nameof(ScanSurfaceSnapper)}] EnvironmentRaycastManager.IsSupported == false", this);
                _warnedSupport = true;
            }

            hasValidHit = false;
            ApplyMissFallback(Time.deltaTime);
            UpdateVisuals();
            return;
        }
        _warnedSupport = false;

        bool hit = TrySnapOnce(out Vector3 targetPos, out Quaternion targetRot);

        if (hit)
        {
            ApplyPose(targetPos, targetRot, Time.deltaTime);
        }
        else
        {
            ApplyMissFallback(Time.deltaTime);
        }

        UpdateVisuals();
    }

    private bool TryResolveRefs()
    {
        if (sessionReferenceFrame == null && transform.parent != null)
            sessionReferenceFrame = transform.parent;

        if (headTransform == null && Camera.main != null)
            headTransform = Camera.main.transform;

        if (raycastManager == null)
            raycastManager = FindObjectOfType<EnvironmentRaycastManager>();

        bool ok = sessionReferenceFrame != null && scanWorkCenter != null;
        if (!ok && logWarnings && !_warnedRefs)
        {
            Debug.LogWarning(
                $"[{nameof(ScanSurfaceSnapper)}] Missing refs. sessionReferenceFrame={(sessionReferenceFrame ? sessionReferenceFrame.name : "null")}, scanWorkCenter={(scanWorkCenter ? scanWorkCenter.name : "null")}",
                this);
            _warnedRefs = true;
        }
        if (ok) _warnedRefs = false;

        if (raycastManager == null && logWarnings && !_warnedRayMgr)
        {
            Debug.LogWarning($"[{nameof(ScanSurfaceSnapper)}] EnvironmentRaycastManager not found in scene.", this);
            _warnedRayMgr = true;
        }
        if (raycastManager != null) _warnedRayMgr = false;

        return ok && raycastManager != null;
    }

    private bool TrySnapOnce(out Vector3 targetPos, out Quaternion targetRot)
    {
        Vector3 refUp = (sessionReferenceFrame != null) ? sessionReferenceFrame.up : Vector3.up;
        Vector3 rayDown = useReferenceDown ? -refUp : Vector3.down;

        Vector3 rayOriginBase = scanWorkCenter.position;
        Vector3 rayOrigin = rayOriginBase + refUp * rayStartHeight;

        Ray ray = new Ray(rayOrigin, rayDown);

        if (drawDebugRay)
            Debug.DrawRay(rayOrigin, rayDown * maxVerticalSnapDistance, debugRayColor);

        targetPos = transform.position;
        targetRot = transform.rotation;

        float maxRayDistance = Mathf.Max(0.05f, rayStartHeight + maxVerticalSnapDistance + 0.5f);
        if (!raycastManager.Raycast(ray, out var hitInfo, maxRayDistance))
        {
            hasValidHit = false;
            return false;
        }

        Vector3 hitPoint = hitInfo.point;
        Vector3 hitNormal = hitInfo.normal.sqrMagnitude > 1e-6f ? hitInfo.normal.normalized : refUp;

        // 垂直方向距离过滤（避免误命中很远处）
        float verticalDist = Mathf.Abs(Vector3.Dot(hitPoint - rayOriginBase, refUp));
        lastVerticalDistance = verticalDist;
        if (verticalDist > maxVerticalSnapDistance)
        {
            hasValidHit = false;
            return false;
        }

        // 法线过滤（只吸附“较向上”的表面）
        if (requireMostlyUpFacingSurface)
        {
            float upDot = Vector3.Dot(hitNormal, refUp.normalized);
            if (upDot < minUpDot)
            {
                hasValidHit = false;
                return false;
            }
        }

        hasValidHit = true;
        lastHitPoint = hitPoint;
        lastHitNormal = hitNormal;

        targetPos = hitPoint + hitNormal * surfaceOffset;
        targetRot = ComputeTargetRotation(hitNormal, refUp);

        return true;
    }

    private Quaternion ComputeTargetRotation(Vector3 hitNormal, Vector3 refUp)
    {
        switch (rotationMode)
        {
            case RotationMode.KeepWorkCenterYawUpright:
                {
                    Vector3 fwd = scanWorkCenter != null ? scanWorkCenter.forward : Vector3.forward;
                    Vector3 planar = Vector3.ProjectOnPlane(fwd, refUp);
                    if (planar.sqrMagnitude < 1e-6f)
                        planar = Vector3.ProjectOnPlane(Vector3.forward, refUp);
                    planar.Normalize();
                    return Quaternion.LookRotation(planar, refUp);
                }

            case RotationMode.FaceSurfaceNormal:
                {
                    Vector3 up = (Vector3.Dot(hitNormal, refUp) >= 0f) ? refUp : -refUp;
                    return Quaternion.LookRotation(hitNormal, up);
                }

            case RotationMode.AlignUpToSurfaceNormalKeepForward:
            default:
                {
                    Vector3 hintForward = Vector3.forward;
                    if (useHeadForwardAsRotationHint && headTransform != null) hintForward = headTransform.forward;
                    else if (scanWorkCenter != null) hintForward = scanWorkCenter.forward;

                    Vector3 projectedForward = Vector3.ProjectOnPlane(hintForward, hitNormal);
                    if (projectedForward.sqrMagnitude < 1e-6f)
                        projectedForward = Vector3.ProjectOnPlane(scanWorkCenter != null ? scanWorkCenter.right : Vector3.right, hitNormal);
                    if (projectedForward.sqrMagnitude < 1e-6f)
                        projectedForward = Vector3.ProjectOnPlane(Vector3.forward, hitNormal);

                    projectedForward.Normalize();
                    return Quaternion.LookRotation(projectedForward, hitNormal);
                }
        }
    }

    private void ApplyPose(Vector3 targetPos, Quaternion targetRot, float dt)
    {
        if (smoothPosition && Application.isPlaying)
        {
            transform.position = Vector3.SmoothDamp(
                transform.position, targetPos, ref _positionVelocity, positionSmoothTime, Mathf.Infinity, dt);
        }
        else
        {
            transform.position = targetPos;
        }

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

    private void ApplyMissFallback(float dt)
    {
        Vector3 refUp = (sessionReferenceFrame != null) ? sessionReferenceFrame.up : Vector3.up;

        switch (missPolicy)
        {
            case MissPolicy.HoldLastValidHit:
                return;

            case MissPolicy.DisableMarkerOnly:
                return;

            case MissPolicy.FallbackToWorkCenter:
            default:
                {
                    if (scanWorkCenter == null) return;

                    Vector3 fallbackPos = scanWorkCenter.position + refUp * fallbackOffset;
                    Quaternion fallbackRot = scanWorkCenter.rotation;
                    ApplyPose(fallbackPos, fallbackRot, dt);
                    return;
                }
        }
    }

    // ---------------- Visuals ----------------

    private void EnsureMarker()
    {
        if (!enableMarker || !spawnMarkerAtRuntime) return;
        if (_markerInstance != null) return;

        if (markerPrefab != null)
        {
            Transform parent = markerParentToSnapPoint ? transform : null;
            _markerInstance = Instantiate(markerPrefab, parent);
            _markerInstance.name = "SurfaceSnapPoint_Marker";
        }
        else if (autoCreateFallbackMarker)
        {
            _markerInstance = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            _markerInstance.name = "SurfaceSnapPoint_Marker(Fallback)";

            Collider c = _markerInstance.GetComponent<Collider>();
            if (c != null)
            {
                if (Application.isPlaying) Destroy(c);
                else DestroyImmediate(c);
            }

            if (markerParentToSnapPoint)
                _markerInstance.transform.SetParent(transform, false);
        }

        if (_markerInstance != null)
            _markerRenderers = _markerInstance.GetComponentsInChildren<Renderer>(true);
    }

    private void EnsureLineRenderer()
    {
        if (!useLineRendererIndicator) return;

        if (_lineRenderer == null)
        {
            GameObject go = new GameObject("SurfaceSnap_Line");
            go.transform.SetParent(transform, false);

            _lineRenderer = go.AddComponent<LineRenderer>();
            _lineRenderer.positionCount = 2;
            _lineRenderer.useWorldSpace = true;
            _lineRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            _lineRenderer.receiveShadows = false;
            _lineRenderer.motionVectorGenerationMode = MotionVectorGenerationMode.ForceNoMotion;
            _lineRenderer.alignment = LineAlignment.View;
            _lineRenderer.textureMode = LineTextureMode.Stretch;
            _lineRenderer.startWidth = lineWidth;
            _lineRenderer.endWidth = lineWidth;

            if (lineMaterial != null)
            {
                _lineRenderer.material = lineMaterial;
            }
            else
            {
                Shader s = Shader.Find("Sprites/Default");
                if (s != null)
                    _lineRenderer.material = new Material(s);
            }
        }
    }

    private void UpdateVisuals()
    {
        // Marker
        if (_markerInstance != null)
        {
            bool active = enableMarker && (missPolicy != MissPolicy.DisableMarkerOnly || hasValidHit);
            if (_markerInstance.activeSelf != active) _markerInstance.SetActive(active);

            if (_markerInstance.activeSelf)
            {
                Transform mt = _markerInstance.transform;

                if (markerParentToSnapPoint)
                {
                    if (mt.parent != transform) mt.SetParent(transform, false);
                    mt.localPosition = markerLocalOffset;
                    mt.localRotation = Quaternion.identity; // 先重置，后续可被 ApplyMarkerUprightRotation 覆盖
                    mt.localScale = markerLocalScale;
                }
                else
                {
                    if (mt.parent != null) mt.SetParent(null, true);
                    mt.position = transform.position + markerWorldOffset;
                    mt.rotation = Quaternion.identity;
                    mt.localScale = markerLocalScale;
                }

                // NEW: 仅影响 marker 显示，不影响吸附点本体旋转
                ApplyMarkerUprightRotation();

                if (showMarkerStatusInName)
                {
                    string s = hasValidHit ? "Hit" : "Miss";
                    _markerInstance.name = $"SurfaceSnapPoint_Marker[{s}]";
                }

                if (colorizeMarker && _markerRenderers != null)
                {
                    Color c = EnvironmentRaycastManager.IsSupported
                        ? (hasValidHit ? hitColor : missColor)
                        : unsupportedColor;

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
            }
        }

        // 线指示器：从 ScanWorkCenter 指向 SurfaceSnapPoint
        if (_lineRenderer != null && scanWorkCenter != null)
        {
            bool show = useLineRendererIndicator && hasValidHit;
            _lineRenderer.enabled = show;

            if (show)
            {
                _lineRenderer.SetPosition(0, scanWorkCenter.position);
                _lineRenderer.SetPosition(1, transform.position);

                if (_lineRenderer.material != null && _lineRenderer.material.HasProperty("_Color"))
                    _lineRenderer.material.color = hitColor;
            }
        }
    }

    /// <summary>
    /// 让 marker 始终保持竖直（锁定 pitch/roll），可选保留 yaw。
    /// 仅作用于 marker 显示层，不改变 SurfaceSnapPoint 本体旋转。
    /// </summary>
    private void ApplyMarkerUprightRotation()
    {
        if (_markerInstance == null || !markerKeepUpright) return;

        Transform mt = _markerInstance.transform;

        // 1) 定义“竖直方向”
        Vector3 up = Vector3.up;
        if (markerUseReferenceUp && sessionReferenceFrame != null)
            up = sessionReferenceFrame.up;

        if (up.sqrMagnitude < 1e-6f) up = Vector3.up;
        up.Normalize();

        // 2) 选择水平朝向来源（只保留 yaw）
        Vector3 fwd = Vector3.forward;
        bool hasForwardSource = false;

        if (markerKeepYaw)
        {
            if (markerYawFromWorkCenter && scanWorkCenter != null)
            {
                fwd = scanWorkCenter.forward;
                hasForwardSource = true;
            }
            else if (headTransform != null)
            {
                fwd = headTransform.forward;
                hasForwardSource = true;
            }
            else if (scanWorkCenter != null)
            {
                fwd = scanWorkCenter.forward;
                hasForwardSource = true;
            }
        }

        // 3) forward 投影到水平面（相对 up），锁定 pitch/roll
        if (hasForwardSource)
        {
            fwd = Vector3.ProjectOnPlane(fwd, up);
            if (fwd.sqrMagnitude < 1e-6f)
            {
                Vector3 fallback = (sessionReferenceFrame != null) ? sessionReferenceFrame.forward : Vector3.forward;
                fwd = Vector3.ProjectOnPlane(fallback, up);
            }
        }
        else
        {
            Vector3 fallback = (sessionReferenceFrame != null) ? sessionReferenceFrame.forward : Vector3.forward;
            fwd = Vector3.ProjectOnPlane(fallback, up);
        }

        if (fwd.sqrMagnitude < 1e-6f)
            fwd = Vector3.Cross(up, Vector3.right);
        if (fwd.sqrMagnitude < 1e-6f)
            fwd = Vector3.Cross(up, Vector3.forward);

        fwd.Normalize();

        // 4) 构造“始终竖直”的旋转（仅 yaw）
        Quaternion uprightRot = Quaternion.LookRotation(fwd, up);

        // 5) prefab 轴向修正
        Quaternion offsetRot = Quaternion.Euler(markerRotationOffsetEuler);

        // 6) 用世界旋转覆盖（抵消父物体倾斜）
        mt.rotation = uprightRot * offsetRot;
    }

#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        if (!drawGizmosSelected) return;

        Gizmos.color = hasValidHit ? hitColor : missColor;
        Gizmos.DrawWireSphere(transform.position, gizmoRadius);

        if (scanWorkCenter != null)
        {
            Gizmos.DrawLine(scanWorkCenter.position, transform.position);

            Vector3 up = sessionReferenceFrame != null ? sessionReferenceFrame.up : Vector3.up;
            Vector3 rayOrigin = scanWorkCenter.position + up * rayStartHeight;
            Vector3 rayDir = useReferenceDown ? -up : Vector3.down;

            Gizmos.color = debugRayColor;
            Gizmos.DrawLine(rayOrigin, rayOrigin + rayDir * maxVerticalSnapDistance);
        }
    }
#endif
}