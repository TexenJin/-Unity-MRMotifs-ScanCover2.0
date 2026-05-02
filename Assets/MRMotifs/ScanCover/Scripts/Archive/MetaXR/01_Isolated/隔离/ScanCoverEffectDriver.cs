using UnityEngine;

/// <summary>
/// 最小可运行版：把 SurfaceSnapPoint / ScanWorkCenter 的信息写入全局 shader 参数，驱动“扫描环/覆盖”效果。
/// 用法：挂在任意常驻对象（建议 SessionReferenceFrame 下的一个空物体）
/// 依赖：只需要 Transform 引用，不依赖 MRUK 类型。
/// </summary>
public class ScanCoverEffectDriver : MonoBehaviour
{
    [Header("References")]
    [Tooltip("稳定参考系（建议 SessionReferenceFrame）")]
    public Transform sessionReferenceFrame;

    [Tooltip("黄色调度中心（可选，但建议填）")]
    public Transform scanWorkCenter;

    [Tooltip("绿色表面吸附点（推荐作为扫描中心）")]
    public Transform surfaceSnapPoint;

    [Header("Scan Source")]
    [Tooltip("true=扫描中心用 SurfaceSnapPoint；false=用 ScanWorkCenter")]
    public bool useSurfaceSnapPointAsCenter = true;

    [Tooltip("扫描法线优先取 SurfaceSnapPoint.up；否则取 SessionReferenceFrame.up / 世界up")]
    public bool useSurfaceSnapPointUpAsNormal = true;

    [Header("Playback")]
    [Tooltip("自动循环扫描")]
    public bool autoLoop = true;

    [Min(0.01f)]
    [Tooltip("一轮扫描时长（秒）")]
    public float loopDuration = 1.8f;

    [Tooltip("启用后自动开扫")]
    public bool playOnEnable = true;

    [Tooltip("使用非缩放时间（暂停UI时也继续）")]
    public bool useUnscaledTime = false;

    [Header("Radius (meters)")]
    [Min(0f)] public float startRadius = 0.0f;
    [Min(0f)] public float endRadius = 2.0f;

    [Header("Shape")]
    [Min(0.001f)]
    [Tooltip("扫描环带宽（米）")]
    public float bandWidth = 0.18f;

    [Min(0.0001f)]
    [Tooltip("边缘羽化（米）")]
    public float feather = 0.05f;

    [Range(0f, 8f)]
    [Tooltip("扫描高亮强度")]
    public float intensity = 1.0f;

    [Range(0f, 1f)]
    [Tooltip("平面厚度（用于只影响靠近扫描平面的片元）；0=不限制")]
    public float planeThickness = 0.12f;

    [Header("Coverage (optional)")]
    [Tooltip("是否同时输出一个“已覆盖进度”参数（0..1），便于 shader 留痕")]
    public bool outputCoverageProgress = true;

    [Range(0f, 1f)]
    [Tooltip("已覆盖强度（供 shader 用）")]
    public float coverageIntensity = 0.35f;

    [Header("Manual Trigger")]
    [Tooltip("自动循环关闭时，可按键触发一轮扫描")]
    public bool enableManualTriggerKey = true;
    public KeyCode triggerKey = KeyCode.G;

    [Header("Shader Global Property Names")]
    public string pActive = "_ScanActive";
    public string pCenterWS = "_ScanCenterWS";
    public string pNormalWS = "_ScanNormalWS";
    public string pWorkCenterWS = "_ScanWorkCenterWS";
    public string pCenterRF = "_ScanCenterRF";
    public string pRadius = "_ScanRadius";
    public string pBandWidth = "_ScanBandWidth";
    public string pFeather = "_ScanFeather";
    public string pIntensity = "_ScanIntensity";
    public string pPlaneThickness = "_ScanPlaneThickness";
    public string pProgress01 = "_ScanProgress01";
    public string pCoverageIntensity = "_ScanCoverageIntensity";
    public string pScanTime = "_ScanTime";

    [Header("Debug")]
    public bool drawDebug = true;
    public Color debugCenterColor = Color.cyan;
    public Color debugNormalColor = Color.green;
    public Color debugWorkCenterColor = Color.yellow;
    [Min(0.01f)] public float debugNormalLength = 0.35f;
    [Min(0.01f)] public float debugCrossHalfSize = 0.08f;

    // runtime
    private bool _isPlayingScan;
    private float _scanElapsed;
    private bool _warnedMissingRef;

    private void Reset()
    {
        if (transform.parent != null) sessionReferenceFrame = transform.parent;
    }

    private void OnEnable()
    {
        if (playOnEnable)
            TriggerScan();
        else
            PushInactiveGlobals();
    }

    private void OnDisable()
    {
        PushInactiveGlobals();
    }

    private void Update()
    {
        // 手动触发
        if (!autoLoop && enableManualTriggerKey && Input.GetKeyDown(triggerKey))
        {
            TriggerScan();
        }

        // 自动循环
        if (autoLoop && !_isPlayingScan)
        {
            TriggerScan();
        }

        // 缺引用检查
        if (!HasValidCenterSource())
        {
            if (!_warnedMissingRef)
            {
                Debug.LogWarning($"[{nameof(ScanCoverEffectDriver)}] 缺少中心点引用（surfaceSnapPoint/scanWorkCenter）。", this);
                _warnedMissingRef = true;
            }
            PushInactiveGlobals();
            return;
        }
        _warnedMissingRef = false;

        float dt = useUnscaledTime ? Time.unscaledDeltaTime : Time.deltaTime;
        float tNow = useUnscaledTime ? Time.unscaledTime : Time.time;

        if (_isPlayingScan)
        {
            _scanElapsed += Mathf.Max(0f, dt);

            float progress01 = Mathf.Clamp01(_scanElapsed / Mathf.Max(0.01f, loopDuration));
            float radius = Mathf.Lerp(startRadius, endRadius, progress01);

            PushGlobals(radius, progress01, tNow);

            if (_scanElapsed >= loopDuration)
            {
                if (autoLoop)
                {
                    // 下一轮从头开始
                    _scanElapsed = 0f;
                }
                else
                {
                    _isPlayingScan = false;
                    // 保留最后一帧 progress=1 的覆盖参数，active 置 0（扫描环停止，留痕仍可用）
                    PushInactiveButKeepCoverage(tNow);
                }
            }
        }
        else
        {
            // 不在播放时也持续推中心/法线，便于shader调试（active=0）
            PushInactiveButKeepCoverage(tNow);
        }

        if (drawDebug)
            DrawDebug();
    }

    public void TriggerScan()
    {
        _isPlayingScan = true;
        _scanElapsed = 0f;
    }

    public void StopScan(bool clearActive = true)
    {
        _isPlayingScan = false;
        if (clearActive) PushInactiveGlobals();
    }

    public bool IsPlayingScan() => _isPlayingScan;

    private bool HasValidCenterSource()
    {
        if (useSurfaceSnapPointAsCenter)
            return surfaceSnapPoint != null || scanWorkCenter != null; // 允许回退
        return scanWorkCenter != null || surfaceSnapPoint != null;
    }

    private Transform GetCenterTransform()
    {
        if (useSurfaceSnapPointAsCenter)
        {
            if (surfaceSnapPoint != null) return surfaceSnapPoint;
            return scanWorkCenter;
        }
        else
        {
            if (scanWorkCenter != null) return scanWorkCenter;
            return surfaceSnapPoint;
        }
    }

    private Vector3 GetScanCenterWS()
    {
        Transform c = GetCenterTransform();
        return c != null ? c.position : Vector3.zero;
    }

    private Vector3 GetScanNormalWS()
    {
        if (useSurfaceSnapPointUpAsNormal && surfaceSnapPoint != null)
        {
            Vector3 n = surfaceSnapPoint.up;
            if (n.sqrMagnitude > 1e-6f) return n.normalized;
        }

        if (sessionReferenceFrame != null)
        {
            Vector3 n = sessionReferenceFrame.up;
            if (n.sqrMagnitude > 1e-6f) return n.normalized;
        }

        return Vector3.up;
    }

    private void PushGlobals(float radius, float progress01, float tNow)
    {
        Vector3 centerWS = GetScanCenterWS();
        Vector3 normalWS = GetScanNormalWS();
        Vector3 workWS = scanWorkCenter != null ? scanWorkCenter.position : centerWS;

        Vector3 centerRF = centerWS;
        if (sessionReferenceFrame != null)
            centerRF = sessionReferenceFrame.InverseTransformPoint(centerWS);

        // 核心开关
        Shader.SetGlobalFloat(pActive, 1f);

        // 空间参数
        Shader.SetGlobalVector(pCenterWS, new Vector4(centerWS.x, centerWS.y, centerWS.z, 1f));
        Shader.SetGlobalVector(pNormalWS, new Vector4(normalWS.x, normalWS.y, normalWS.z, 0f));
        Shader.SetGlobalVector(pWorkCenterWS, new Vector4(workWS.x, workWS.y, workWS.z, 1f));
        Shader.SetGlobalVector(pCenterRF, new Vector4(centerRF.x, centerRF.y, centerRF.z, 1f));

        // 形状参数
        Shader.SetGlobalFloat(pRadius, radius);
        Shader.SetGlobalFloat(pBandWidth, Mathf.Max(0.001f, bandWidth));
        Shader.SetGlobalFloat(pFeather, Mathf.Max(0.0001f, feather));
        Shader.SetGlobalFloat(pIntensity, intensity);
        Shader.SetGlobalFloat(pPlaneThickness, planeThickness);

        // 进度/时间
        Shader.SetGlobalFloat(pProgress01, progress01);
        Shader.SetGlobalFloat(pCoverageIntensity, coverageIntensity);
        Shader.SetGlobalFloat(pScanTime, tNow);
    }

    private void PushInactiveGlobals()
    {
        float tNow = useUnscaledTime ? Time.unscaledTime : Time.time;
        Vector3 centerWS = GetScanCenterWS();
        Vector3 normalWS = GetScanNormalWS();
        Vector3 workWS = scanWorkCenter != null ? scanWorkCenter.position : centerWS;

        Vector3 centerRF = centerWS;
        if (sessionReferenceFrame != null)
            centerRF = sessionReferenceFrame.InverseTransformPoint(centerWS);

        Shader.SetGlobalFloat(pActive, 0f);
        Shader.SetGlobalVector(pCenterWS, new Vector4(centerWS.x, centerWS.y, centerWS.z, 1f));
        Shader.SetGlobalVector(pNormalWS, new Vector4(normalWS.x, normalWS.y, normalWS.z, 0f));
        Shader.SetGlobalVector(pWorkCenterWS, new Vector4(workWS.x, workWS.y, workWS.z, 1f));
        Shader.SetGlobalVector(pCenterRF, new Vector4(centerRF.x, centerRF.y, centerRF.z, 1f));

        Shader.SetGlobalFloat(pRadius, startRadius);
        Shader.SetGlobalFloat(pBandWidth, Mathf.Max(0.001f, bandWidth));
        Shader.SetGlobalFloat(pFeather, Mathf.Max(0.0001f, feather));
        Shader.SetGlobalFloat(pIntensity, intensity);
        Shader.SetGlobalFloat(pPlaneThickness, planeThickness);

        Shader.SetGlobalFloat(pProgress01, 0f);
        Shader.SetGlobalFloat(pCoverageIntensity, coverageIntensity);
        Shader.SetGlobalFloat(pScanTime, tNow);
    }

    private void PushInactiveButKeepCoverage(float tNow)
    {
        Vector3 centerWS = GetScanCenterWS();
        Vector3 normalWS = GetScanNormalWS();
        Vector3 workWS = scanWorkCenter != null ? scanWorkCenter.position : centerWS;

        Vector3 centerRF = centerWS;
        if (sessionReferenceFrame != null)
            centerRF = sessionReferenceFrame.InverseTransformPoint(centerWS);

        Shader.SetGlobalFloat(pActive, 0f); // 扫描环停
        Shader.SetGlobalVector(pCenterWS, new Vector4(centerWS.x, centerWS.y, centerWS.z, 1f));
        Shader.SetGlobalVector(pNormalWS, new Vector4(normalWS.x, normalWS.y, normalWS.z, 0f));
        Shader.SetGlobalVector(pWorkCenterWS, new Vector4(workWS.x, workWS.y, workWS.z, 1f));
        Shader.SetGlobalVector(pCenterRF, new Vector4(centerRF.x, centerRF.y, centerRF.z, 1f));

        Shader.SetGlobalFloat(pRadius, endRadius);
        Shader.SetGlobalFloat(pBandWidth, Mathf.Max(0.001f, bandWidth));
        Shader.SetGlobalFloat(pFeather, Mathf.Max(0.0001f, feather));
        Shader.SetGlobalFloat(pIntensity, intensity);
        Shader.SetGlobalFloat(pPlaneThickness, planeThickness);

        // 手动模式下停在 1，方便 shader 做“已覆盖”
        float p = autoLoop ? 0f : 1f;
        Shader.SetGlobalFloat(pProgress01, p);
        Shader.SetGlobalFloat(pCoverageIntensity, coverageIntensity);
        Shader.SetGlobalFloat(pScanTime, tNow);
    }

    private void DrawDebug()
    {
        Vector3 center = GetScanCenterWS();
        Vector3 normal = GetScanNormalWS();
        Vector3 work = scanWorkCenter != null ? scanWorkCenter.position : center;

        Debug.DrawLine(center, center + normal * debugNormalLength, debugNormalColor);

        if (scanWorkCenter != null)
            Debug.DrawLine(work, center, debugWorkCenterColor);

        // 在扫描平面画个小十字，便于观察中心位置
        Vector3 tangent = Vector3.Cross(normal, Vector3.up);
        if (tangent.sqrMagnitude < 1e-6f) tangent = Vector3.Cross(normal, Vector3.right);
        tangent.Normalize();
        Vector3 bitangent = Vector3.Cross(normal, tangent).normalized;

        Debug.DrawLine(center - tangent * debugCrossHalfSize, center + tangent * debugCrossHalfSize, debugCenterColor);
        Debug.DrawLine(center - bitangent * debugCrossHalfSize, center + bitangent * debugCrossHalfSize, debugCenterColor);
    }
}