using UnityEngine;
using Meta.XR; // EnvironmentRaycastManager / EnvironmentRaycastHit / EnvironmentRaycastHitStatus

/// <summary>
/// ScanShockwaveBridge（完整升级版，含 EnvironmentRaycastManager 混合解算）
///
/// 核心思路：
/// 1) 以 ScanSurfaceSnapper 的结果点（绿球）作为“候选爆点”
/// 2) 用 EnvironmentRaycastManager 对最终爆点做校正（可选）
///    - 相机 -> 候选点 方向射线（优先）
///    - 候选点上方 -> 向下 探测（次选）
/// 3) 在最终爆点实例化 MRMotifs 的 ShockWaveEffect.prefab（AudioSource 根组件）
///
/// 目标：
/// - 保留你现有 ScanWorkCenter / ScanSurfaceSnapper 体系
/// - 不复用扔球逻辑
/// - 但把 MRMotifs 的“环境射线命中保障”补回来，减少无效爆点
/// </summary>
public class ScanShockwaveBridge : MonoBehaviour
{
    public enum UpMode
    {
        KeepPrefabRotation = 0,
        AlignToSurfaceSnapUp = 1,
        AlignToReferenceUp = 2,
        AlignToWorldUp = 3,
    }

    public enum ParentMode
    {
        NoneWorldSpace = 0,
        ParentToSessionReferenceFrame = 1,
        ParentToCustomTransform = 2,
    }

    public enum ResolvePoseMode
    {
        SnapOnly = 0,
        CameraRayThenTopDownThenFallback = 1,
        CameraRayThenFallback = 2,
        TopDownThenCameraRayThenFallback = 3,
        TopDownThenFallback = 4,
    }

    public enum RayOriginMode
    {
        MainCamera = 0,
        OverrideTransform = 1,
        SessionReferenceFrame = 2,
    }

    [Header("References")]
    [Tooltip("固定参考系（推荐 SessionReferenceFrame）")]
    public Transform sessionReferenceFrame;

    [Tooltip("绿球（ScanSurfaceSnapper 的可视/结果点），作为候选爆点来源")]
    public Transform surfaceSnapPoint;

    [Tooltip("可选：若你有单独的法线来源（例如 snapper 输出的法线对齐对象），填这里；为空则用 surfaceSnapPoint.up")]
    public Transform surfaceNormalSource;

    [Tooltip("MRMotifs 的 ShockWaveEffect.prefab（请拖 prefab 根节点上的 AudioSource 组件）")]
    public AudioSource shockWaveEffectPrefab;

    [Tooltip("可选：覆盖 prefab 默认音效；为空则保留 prefab 原音效")]
    public AudioClip scanTriggerClip;

    [Tooltip("可选：用于判定当前 snap 是否有效。若指定且 inactive，则禁止触发")]
    public GameObject validityIndicator;

    [Header("Environment Raycast (Hybrid Resolve)")]
    [Tooltip("MRMotifs/Meta XR 的 EnvironmentRaycastManager（建议从场景中直接拖入 [MR Motif] EnvironmentDepth 上的组件）")]
    public EnvironmentRaycastManager environmentRaycast;

    [Tooltip("是否启用 EnvironmentRaycastManager 参与最终爆点解算")]
    public bool useEnvironmentRaycastForFinalPose = true;

    [Tooltip("最终爆点解算策略")]
    public ResolvePoseMode resolvePoseMode = ResolvePoseMode.CameraRayThenTopDownThenFallback;

    [Tooltip("若启用：ERM 未命中时直接禁止触发（不回退到绿球点）。正式版建议开启；冒烟测试可关闭。")]
    public bool requireEnvironmentRaycastHit = false;

    [Tooltip("当 ERM 状态为 NotReady 时是否允许回退到绿球点")]
    public bool allowFallbackWhenEnvRaycastNotReady = true;

    [Tooltip("当 ERM 状态为 NoHit / OutsideFrustum / RayOccluded 时是否允许回退到绿球点")]
    public bool allowFallbackWhenEnvRaycastNoHit = true;

    [Tooltip("命中法线最小置信度（0=不限制）。若命中但置信度低于阈值，可视作无效。")]
    [Range(0f, 1f)] public float minNormalConfidence = 0f;

    [Tooltip("是否使用 ERM 命中法线作为朝上方向（否则仍使用绿球/参考系 up）")]
    public bool useEnvHitNormalAsUp = true;

    [Tooltip("沿 ERM 命中法线的额外偏移（米），避免爆点埋入表面；与 spawnOffsetAlongUp 叠加）")]
    [Min(0f)] public float envHitExtraOffset = 0.005f;

    [Header("ERM Ray Origin")]
    public RayOriginMode rayOriginMode = RayOriginMode.MainCamera;

    [Tooltip("RayOriginMode=OverrideTransform 时使用")]
    public Transform rayOriginOverride;

    [Tooltip("相机->候选点 射线最大距离（米）")]
    [Min(0.1f)] public float cameraToSnapRayMaxDistance = 8f;

    [Tooltip("给相机->候选点距离增加一点余量（米），避免因为候选点略超出而 miss")]
    [Min(0f)] public float cameraToSnapRayDistancePadding = 0.25f;

    [Header("ERM Top-Down Probe")]
    [Tooltip("是否启用候选点上方 -> 向下 的补充探测")]
    public bool enableTopDownProbe = true;

    [Tooltip("沿参考 up 在候选点上方抬高的探测起点高度（米）")]
    [Min(0.01f)] public float topDownProbeHeight = 0.25f;

    [Tooltip("向下探测最大距离（米）")]
    [Min(0.05f)] public float topDownProbeMaxDistance = 1.0f;

    [Header("Spawn & Orientation")]
    public UpMode upMode = UpMode.AlignToSurfaceSnapUp;

    [Tooltip("让 effect 的 forward 尽量沿参考系前向投影（更稳定）；否则保留 prefab 原 forward")]
    public bool alignForwardToReferenceForward = true;

    [Tooltip("沿最终 up 方向抬起一点，避免爆点埋入表面")]
    [Min(0f)] public float spawnOffsetAlongUp = 0.01f;

    public ParentMode parentMode = ParentMode.NoneWorldSpace;

    [Tooltip("ParentMode=ParentToCustomTransform 时使用")]
    public Transform customParent;

    [Header("Trigger")]
    public bool enableManualKey = true;
    public KeyCode manualTriggerKey = KeyCode.G;

    [Tooltip("是否启用 Meta Touch 手柄按键触发（OVRInput）")]
    public bool enableControllerButton = true;

    [Tooltip("触发按钮：Button.Two 常对应右手B/左手Y；Button.One 常对应右手A/左手X")]
    public global::OVRInput.Button controllerTriggerButton = global::OVRInput.Button.Two;

    [Tooltip("指定手柄（建议固定一只手以避免误触）")]
    public global::OVRInput.Controller controllerTriggerController = global::OVRInput.Controller.RTouch;

    [Tooltip("若指定手柄未命中，是否允许用 Active 手柄作为回退")]
    public bool allowActiveControllerFallback = false;

    [Tooltip("触发冷却（秒），防止连按爆刷")]
    [Min(0f)] public float cooldown = 0.25f;

    [Tooltip("需要稳定持续时间（秒）后才允许触发；0=不要求")]
    [Min(0f)] public float requiredStableTime = 0.12f;

    [Tooltip("稳定判定位置容差（米）：连续帧位移低于此值，累计稳定时间")]
    [Min(0f)] public float stablePositionTolerance = 0.01f;

    [Tooltip("若启用，则 surfaceSnapPoint.activeInHierarchy=false 时视为无效")]
    public bool requireSurfaceSnapPointActive = true;

    [Tooltip("若启用，则必须通过稳定门控才允许触发")]
    public bool requireStableBeforeTrigger = true;

    [Header("Audio Pitch (Optional, mimic OrbSpawnerMotif)")]
    public bool enablePitchRamp = true;
    [Min(0f)] public float pitchDecayPerSecond = 1.0f;
    [Range(0f, 2f)] public float pitchStep = 0.25f;
    [Min(0.1f)] public float minPitch = 1.0f;
    [Min(0.1f)] public float maxPitch = 3.0f;

    [Header("Debug")]
    public bool drawDebug = true;
    public bool logWarnings = true;
    public bool logResolveDetails = true;

    public Color debugSnapPointColor = Color.green;
    public Color debugSpawnPointColor = Color.cyan;
    public Color debugResolvedPointColor = new Color(1f, 0.6f, 0f, 1f); // 橙色
    public Color debugUpColor = Color.yellow;
    public Color debugRayCameraToSnapColor = new Color(1f, 0f, 1f, 1f); // 品红
    public Color debugRayTopDownColor = new Color(0f, 0.8f, 1f, 1f);     // 青蓝
    [Min(0.01f)] public float debugUpLength = 0.15f;
    [Min(0.005f)] public float debugCrossHalf = 0.03f;

    // runtime
    private Vector3 _lastObservedSnapPos;
    private bool _hasObservedSnapPos;
    private float _stableTimer;
    private float _lastTriggerTime = -999f;
    private float _pitchLevel = 1.0f;
    private bool _warnedRefs;

    // runtime debug cache
    private Vector3 _dbgLastResolvedPoint;
    private Vector3 _dbgLastResolvedUp = Vector3.up;
    private bool _dbgHasResolvedPoint;
    private string _dbgLastResolveSource = "";
    private string _dbgLastResolveStatus = "";
    private float _dbgLastResolveTime = -999f;

    private void Reset()
    {
        if (transform.parent != null) sessionReferenceFrame = transform.parent;
        if (surfaceNormalSource == null) surfaceNormalSource = surfaceSnapPoint;
    }

    private void Awake()
    {
        if (surfaceNormalSource == null) surfaceNormalSource = surfaceSnapPoint;
        _pitchLevel = Mathf.Clamp(minPitch, 0.1f, maxPitch);
    }

    private void Update()
    {
        // 模拟 OrbSpawnerMotif 的 pitch 回落
        if (enablePitchRamp)
        {
            _pitchLevel = Mathf.Max(minPitch, _pitchLevel - pitchDecayPerSecond * Time.deltaTime);
        }

        UpdateStability();

        if (drawDebug)
        {
            DrawDebug();
        }

        bool triggerRequested = false;

        // 1) 键盘触发（备用）
        if (enableManualKey && Input.GetKeyDown(manualTriggerKey))
        {
            triggerRequested = true;
        }

        // 2) 手柄触发（OVRInput，Unity6 + Meta XR SDK）
        if (!triggerRequested && enableControllerButton)
        {
            if (global::OVRInput.GetDown(controllerTriggerButton, controllerTriggerController))
            {
                triggerRequested = true;
            }
            else if (allowActiveControllerFallback &&
                     global::OVRInput.GetDown(controllerTriggerButton, global::OVRInput.Controller.Active))
            {
                triggerRequested = true;
            }
        }

        if (triggerRequested)
        {
            TriggerShockwave();
        }
    }

    [ContextMenu("TEST Trigger Shockwave")]
    public void DebugTriggerShockwaveFromContextMenu()
    {
        TriggerShockwave();
    }

    /// <summary>
    /// 外部可调用触发入口（按钮 / 状态机 / 事件）
    /// </summary>
    public bool TriggerShockwave()
    {
        if (!TryValidateRefs())
            return false;

        if (!IsSnapValid())
        {
            if (logWarnings)
            {
                Debug.LogWarning($"[{nameof(ScanShockwaveBridge)}] Trigger blocked: snap invalid.", this);
            }
            return false;
        }

        float dtSinceLast = Time.time - _lastTriggerTime;
        if (dtSinceLast < cooldown)
        {
            if (logWarnings)
            {
                Debug.LogWarning($"[{nameof(ScanShockwaveBridge)}] Trigger blocked: cooldown {dtSinceLast:F3}/{cooldown:F3}s", this);
            }
            return false;
        }

        if (requireStableBeforeTrigger && requiredStableTime > 0f && _stableTimer < requiredStableTime)
        {
            if (logWarnings)
            {
                Debug.LogWarning($"[{nameof(ScanShockwaveBridge)}] Trigger blocked: not stable enough {_stableTimer:F3}/{requiredStableTime:F3}s (stableTol={stablePositionTolerance:F4})", this);
            }
            return false;
        }

        if (!TryResolveFinalSpawnPose(out Vector3 resolvedPoint, out Vector3 finalUp, out string resolveSource, out string resolveStatus))
        {
            if (logWarnings)
            {
                Debug.LogWarning($"[{nameof(ScanShockwaveBridge)}] Trigger blocked: resolve failed. source={resolveSource}, status={resolveStatus}", this);
            }
            return false;
        }

        Vector3 spawnPos = resolvedPoint + finalUp * spawnOffsetAlongUp;
        Quaternion spawnRot = GetSpawnRotation(finalUp);
        Transform parent = GetSpawnParent();

        AudioSource spawnedAudio = null;
        try
        {
            if (parent == null)
                spawnedAudio = Instantiate(shockWaveEffectPrefab, spawnPos, spawnRot);
            else
                spawnedAudio = Instantiate(shockWaveEffectPrefab, spawnPos, spawnRot, parent);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[{nameof(ScanShockwaveBridge)}] Instantiate failed: {e.Message}", this);
            return false;
        }

        if (spawnedAudio == null)
        {
            Debug.LogError($"[{nameof(ScanShockwaveBridge)}] Instantiate returned null.", this);
            return false;
        }

        if (scanTriggerClip != null)
        {
            spawnedAudio.clip = scanTriggerClip;
        }

        if (enablePitchRamp)
        {
            _pitchLevel = Mathf.Clamp(_pitchLevel + pitchStep, minPitch, maxPitch);
            spawnedAudio.pitch = _pitchLevel;
        }

        if (spawnedAudio.clip != null)
        {
            spawnedAudio.Play();
        }

        _dbgLastResolvedPoint = resolvedPoint;
        _dbgLastResolvedUp = finalUp;
        _dbgHasResolvedPoint = true;
        _dbgLastResolveSource = resolveSource;
        _dbgLastResolveStatus = resolveStatus;
        _dbgLastResolveTime = Time.time;

        if (logResolveDetails)
        {
            Debug.Log(
                $"[{nameof(ScanShockwaveBridge)}] Trigger OK -> Spawned '{spawnedAudio.gameObject.name}' " +
                $"source={resolveSource}, status={resolveStatus}, point={resolvedPoint}, up={finalUp}, " +
                $"spawnPos={spawnPos}, parent={(parent ? parent.name : "None(World)")}",
                this);
        }

        _lastTriggerTime = Time.time;
        return true;
    }

    public bool CanTriggerNow()
    {
        if (!TryValidateRefs(false)) return false;
        if (!IsSnapValid()) return false;
        if (Time.time - _lastTriggerTime < cooldown) return false;
        if (requireStableBeforeTrigger && requiredStableTime > 0f && _stableTimer < requiredStableTime) return false;
        return true;
    }

    public float GetStableTimer() => _stableTimer;

    public Vector3 GetPredictedSpawnPosition()
    {
        if (surfaceSnapPoint == null) return Vector3.zero;
        Vector3 up = GetChosenUp();
        return surfaceSnapPoint.position + up * spawnOffsetAlongUp;
    }

    private void UpdateStability()
    {
        if (surfaceSnapPoint == null)
        {
            _hasObservedSnapPos = false;
            _stableTimer = 0f;
            return;
        }

        if (!IsSnapValid())
        {
            _hasObservedSnapPos = false;
            _stableTimer = 0f;
            return;
        }

        Vector3 p = surfaceSnapPoint.position;

        if (!_hasObservedSnapPos)
        {
            _lastObservedSnapPos = p;
            _hasObservedSnapPos = true;
            _stableTimer = 0f;
            return;
        }

        float d = Vector3.Distance(p, _lastObservedSnapPos);
        if (d <= stablePositionTolerance)
            _stableTimer += Time.deltaTime;
        else
            _stableTimer = 0f;

        _lastObservedSnapPos = p;
    }

    private bool TryValidateRefs(bool log = true)
    {
        bool ok = surfaceSnapPoint != null && shockWaveEffectPrefab != null;

        if (!ok)
        {
            if (log && logWarnings && !_warnedRefs)
            {
                Debug.LogWarning(
                    $"[{nameof(ScanShockwaveBridge)}] Missing refs. " +
                    $"surfaceSnapPoint={(surfaceSnapPoint ? surfaceSnapPoint.name : "null")}, " +
                    $"shockWaveEffectPrefab={(shockWaveEffectPrefab ? shockWaveEffectPrefab.name : "null")}",
                    this);
                _warnedRefs = true;
            }
            return false;
        }

        _warnedRefs = false;
        return true;
    }

    private bool IsSnapValid()
    {
        if (surfaceSnapPoint == null)
            return false;

        if (requireSurfaceSnapPointActive && !surfaceSnapPoint.gameObject.activeInHierarchy)
            return false;

        if (validityIndicator != null && !validityIndicator.activeInHierarchy)
            return false;

        return true;
    }

    /// <summary>
    /// 混合解算最终爆点：
    /// - 绿球是候选点
    /// - ERM 用于校正
    /// - 可按策略 fallback
    /// </summary>
    private bool TryResolveFinalSpawnPose(out Vector3 finalPoint, out Vector3 finalUp, out string resolveSource, out string resolveStatus)
    {
        finalPoint = surfaceSnapPoint != null ? surfaceSnapPoint.position : Vector3.zero;
        finalUp = GetChosenUp();
        resolveSource = "SnapFallback";
        resolveStatus = "N/A";

        if (!useEnvironmentRaycastForFinalPose || environmentRaycast == null)
        {
            resolveSource = "SnapOnly";
            resolveStatus = useEnvironmentRaycastForFinalPose ? "ERM_NULL" : "ERM_DISABLED";
            return true;
        }

        bool canFallbackOnFailure = true;

        // 先按策略尝试 ERM 命中
        bool hitSuccess = false;
        Vector3 hitPoint = finalPoint;
        Vector3 hitNormal = finalUp;
        string hitSource = "";
        string hitStatus = "";

        switch (resolvePoseMode)
        {
            case ResolvePoseMode.SnapOnly:
                resolveSource = "SnapOnly";
                resolveStatus = "MODE_SNAP_ONLY";
                return true;

            case ResolvePoseMode.CameraRayThenTopDownThenFallback:
                hitSuccess =
                    TryResolveByCameraRay(out hitPoint, out hitNormal, out hitSource, out hitStatus) ||
                    (enableTopDownProbe && TryResolveByTopDownProbe(out hitPoint, out hitNormal, out hitSource, out hitStatus));
                break;

            case ResolvePoseMode.CameraRayThenFallback:
                hitSuccess = TryResolveByCameraRay(out hitPoint, out hitNormal, out hitSource, out hitStatus);
                break;

            case ResolvePoseMode.TopDownThenCameraRayThenFallback:
                hitSuccess =
                    (enableTopDownProbe && TryResolveByTopDownProbe(out hitPoint, out hitNormal, out hitSource, out hitStatus)) ||
                    TryResolveByCameraRay(out hitPoint, out hitNormal, out hitSource, out hitStatus);
                break;

            case ResolvePoseMode.TopDownThenFallback:
                hitSuccess = enableTopDownProbe && TryResolveByTopDownProbe(out hitPoint, out hitNormal, out hitSource, out hitStatus);
                break;

            default:
                hitSuccess = false;
                break;
        }

        if (hitSuccess)
        {
            finalPoint = hitPoint;
            finalUp = (useEnvHitNormalAsUp && hitNormal.sqrMagnitude > 1e-6f) ? hitNormal.normalized : GetChosenUp();
            finalPoint += finalUp * envHitExtraOffset;
            resolveSource = hitSource;
            resolveStatus = hitStatus;
            return true;
        }

        // 没命中 ERM：看是否允许 fallback
        if (requireEnvironmentRaycastHit)
        {
            canFallbackOnFailure = false;
        }

        if (!canFallbackOnFailure)
        {
            resolveSource = "ERM_REQUIRED";
            resolveStatus = "NO_VALID_HIT";
            return false;
        }

        // 回退绿球点（根据最近一次失败状态决定是否允许）
        if (!IsEnvFallbackAllowedByLastStatus(hitStatus))
        {
            resolveSource = "ERM_FAIL_NO_FALLBACK";
            resolveStatus = hitStatus;
            return false;
        }

        resolveSource = "SnapFallback";
        resolveStatus = string.IsNullOrEmpty(hitStatus) ? "ERM_FAIL_UNKNOWN" : hitStatus;
        finalPoint = surfaceSnapPoint.position;
        finalUp = GetChosenUp();
        return true;
    }

    private bool TryResolveByCameraRay(out Vector3 point, out Vector3 normal, out string source, out string status)
    {
        point = Vector3.zero;
        normal = Vector3.up;
        source = "ERM.CameraRay";
        status = "UNSET";

        if (surfaceSnapPoint == null)
        {
            status = "SNAP_NULL";
            return false;
        }

        if (!TryGetRayOrigin(out Transform originTf))
        {
            status = "RAY_ORIGIN_NULL";
            return false;
        }

        Vector3 origin = originTf.position;
        Vector3 toSnap = surfaceSnapPoint.position - origin;
        float dist = toSnap.magnitude;
        if (dist < 1e-4f)
        {
            status = "TOO_CLOSE";
            return false;
        }

        Vector3 dir = toSnap / dist;
        float maxDist = Mathf.Max(0.1f, Mathf.Min(cameraToSnapRayMaxDistance, dist + cameraToSnapRayDistancePadding));

        if (drawDebug)
        {
            Debug.DrawLine(origin, origin + dir * maxDist, debugRayCameraToSnapColor, 0f, false);
        }

        bool ok = environmentRaycast.Raycast(new Ray(origin, dir), out var hit, maxDist);
        status = $"ok={ok}, status={hit.status}";

        if (!ok)
            return false;

        if (minNormalConfidence > 0f && hit.normalConfidence < minNormalConfidence)
        {
            status = $"{status}, normalConfidence={hit.normalConfidence:F3} < {minNormalConfidence:F3}";
            return false;
        }

        point = hit.point;
        normal = SafeNormal(hit.normal, GetChosenUp());
        status = $"{status}, normalConfidence={hit.normalConfidence:F3}";
        return true;
    }

    private bool TryResolveByTopDownProbe(out Vector3 point, out Vector3 normal, out string source, out string status)
    {
        point = Vector3.zero;
        normal = Vector3.up;
        source = "ERM.TopDownProbe";
        status = "UNSET";

        if (surfaceSnapPoint == null)
        {
            status = "SNAP_NULL";
            return false;
        }

        Vector3 probeUp = GetChosenUp();
        if (probeUp.sqrMagnitude < 1e-6f)
            probeUp = Vector3.up;
        probeUp.Normalize();

        Vector3 origin = surfaceSnapPoint.position + probeUp * topDownProbeHeight;
        Vector3 dir = -probeUp;
        float maxDist = Mathf.Max(0.05f, topDownProbeMaxDistance);

        if (drawDebug)
        {
            Debug.DrawLine(origin, origin + dir * maxDist, debugRayTopDownColor, 0f, false);
        }

        bool ok = environmentRaycast.Raycast(new Ray(origin, dir), out var hit, maxDist);
        status = $"ok={ok}, status={hit.status}";

        if (!ok)
            return false;

        if (minNormalConfidence > 0f && hit.normalConfidence < minNormalConfidence)
        {
            status = $"{status}, normalConfidence={hit.normalConfidence:F3} < {minNormalConfidence:F3}";
            return false;
        }

        point = hit.point;
        normal = SafeNormal(hit.normal, probeUp);
        status = $"{status}, normalConfidence={hit.normalConfidence:F3}";
        return true;
    }

    private bool IsEnvFallbackAllowedByLastStatus(string status)
    {
        if (string.IsNullOrEmpty(status))
            return allowFallbackWhenEnvRaycastNoHit;

        // 粗粒度按状态文本判断即可（避免引入更多状态缓存字段）
        if (status.Contains("NotReady"))
            return allowFallbackWhenEnvRaycastNotReady;

        // NoHit / OutsideFrustum / RayOccluded / HitPointOccluded 等
        return allowFallbackWhenEnvRaycastNoHit;
    }

    private bool TryGetRayOrigin(out Transform originTf)
    {
        originTf = null;
        switch (rayOriginMode)
        {
            case RayOriginMode.OverrideTransform:
                originTf = rayOriginOverride;
                break;
            case RayOriginMode.SessionReferenceFrame:
                originTf = sessionReferenceFrame;
                break;
            case RayOriginMode.MainCamera:
            default:
                originTf = Camera.main != null ? Camera.main.transform : null;
                break;
        }

        // fallback
        if (originTf == null && Camera.main != null) originTf = Camera.main.transform;
        if (originTf == null) originTf = sessionReferenceFrame;

        return originTf != null;
    }

    private Vector3 GetChosenUp()
    {
        switch (upMode)
        {
            case UpMode.AlignToSurfaceSnapUp:
                {
                    Transform nsrc = surfaceNormalSource != null ? surfaceNormalSource : surfaceSnapPoint;
                    if (nsrc != null)
                    {
                        Vector3 n = nsrc.up;
                        if (n.sqrMagnitude > 1e-6f) return n.normalized;
                    }
                    break;
                }
            case UpMode.AlignToReferenceUp:
                if (sessionReferenceFrame != null && sessionReferenceFrame.up.sqrMagnitude > 1e-6f)
                    return sessionReferenceFrame.up.normalized;
                break;
            case UpMode.AlignToWorldUp:
                return Vector3.up;
            case UpMode.KeepPrefabRotation:
            default:
                break;
        }

        if (sessionReferenceFrame != null && sessionReferenceFrame.up.sqrMagnitude > 1e-6f)
            return sessionReferenceFrame.up.normalized;

        return Vector3.up;
    }

    private Quaternion GetSpawnRotation(Vector3 up)
    {
        if (shockWaveEffectPrefab == null)
            return Quaternion.identity;

        Quaternion prefabRot = shockWaveEffectPrefab.transform.rotation;
        if (upMode == UpMode.KeepPrefabRotation)
            return prefabRot;

        Vector3 targetUp = up.sqrMagnitude > 1e-6f ? up.normalized : Vector3.up;

        // 先得到 forward 候选
        Vector3 forward;
        if (alignForwardToReferenceForward && sessionReferenceFrame != null)
        {
            forward = sessionReferenceFrame.forward;
            forward = Vector3.ProjectOnPlane(forward, targetUp);
            if (forward.sqrMagnitude < 1e-6f)
                forward = Vector3.ProjectOnPlane(Camera.main != null ? Camera.main.transform.forward : Vector3.forward, targetUp);
            if (forward.sqrMagnitude < 1e-6f)
                forward = Vector3.ProjectOnPlane(Vector3.forward, targetUp);
            forward.Normalize();
        }
        else
        {
            forward = Vector3.ProjectOnPlane(prefabRot * Vector3.forward, targetUp);
            if (forward.sqrMagnitude < 1e-6f)
                forward = Vector3.ProjectOnPlane(Vector3.forward, targetUp);
            forward.Normalize();
        }

        return Quaternion.LookRotation(forward, targetUp);
    }

    private Transform GetSpawnParent()
    {
        switch (parentMode)
        {
            case ParentMode.ParentToSessionReferenceFrame:
                return sessionReferenceFrame;
            case ParentMode.ParentToCustomTransform:
                return customParent;
            case ParentMode.NoneWorldSpace:
            default:
                return null;
        }
    }

    private static Vector3 SafeNormal(Vector3 n, Vector3 fallback)
    {
        if (n.sqrMagnitude > 1e-6f) return n.normalized;
        if (fallback.sqrMagnitude > 1e-6f) return fallback.normalized;
        return Vector3.up;
    }

    private void DrawDebug()
    {
        if (surfaceSnapPoint == null) return;

        Vector3 snap = surfaceSnapPoint.position;
        Vector3 up = GetChosenUp();
        Vector3 predSpawn = snap + up * spawnOffsetAlongUp;

        // snap 点十字（绿）
        DrawCross(snap, up, debugCrossHalf, debugSnapPointColor);

        // 预测 spawn 点（青）
        DrawCross(predSpawn, up, debugCrossHalf * 0.8f, debugSpawnPointColor);
        Debug.DrawLine(predSpawn, predSpawn + up * debugUpLength, debugUpColor);

        // 最近一次实际 resolved 点（橙）
        if (_dbgHasResolvedPoint && (Time.time - _dbgLastResolveTime) < 2.0f)
        {
            DrawCross(_dbgLastResolvedPoint, _dbgLastResolvedUp, debugCrossHalf * 1.2f, debugResolvedPointColor);
            Debug.DrawLine(_dbgLastResolvedPoint, _dbgLastResolvedPoint + _dbgLastResolvedUp * (debugUpLength * 1.2f), debugResolvedPointColor);
        }
    }

    private void DrawCross(Vector3 center, Vector3 normal, float halfSize, Color color)
    {
        Vector3 n = normal.sqrMagnitude > 1e-6f ? normal.normalized : Vector3.up;
        Vector3 t = Vector3.Cross(n, Vector3.up);
        if (t.sqrMagnitude < 1e-6f) t = Vector3.Cross(n, Vector3.right);
        t.Normalize();
        Vector3 b = Vector3.Cross(n, t).normalized;

        Debug.DrawLine(center - t * halfSize, center + t * halfSize, color);
        Debug.DrawLine(center - b * halfSize, center + b * halfSize, color);
    }
}