using System;
using System.Reflection;
using System.Text;
using Meta.XR.EnvironmentDepth;
using MyProject.XR;
using UnityEngine;
using UnityEngine.Rendering;

[DisallowMultipleComponent]
[DefaultExecutionOrder(10000)]
public sealed class ScanCoverSamplingWindowDiagnosticTool : MonoBehaviour
{
    private static readonly int EnvironmentDepthTextureId = Shader.PropertyToID("_EnvironmentDepthTexture");
    private static readonly int EnvironmentDepthZBufferParamsId = Shader.PropertyToID("_EnvironmentDepthZBufferParams");

    [Header("Targets")]
    public ScanCoverDepthPointBurstWindow targetWindow;
    public CustomEnvironmentDepthRaycaster depthRaycaster;
    public EnvironmentDepthManager environmentDepthManager;
    public Camera centerCamera;

    [Header("Logging")]
    public bool logOnStart = true;
    public bool continuousLogging = true;
    public float logEverySeconds = 1.25f;
    public bool includeDepthTextureScan = true;
    public bool includeRawEnvironmentDepthProbe = true;
    public bool includeHierarchyScan = true;

    [Header("Raw GPU Depth Probe")]
    public ComputeShader rawDepthProbeShader;

    [Header("Depth Checks")]
    public int sampleRadiusPixels = 12;
    public float expectedMinWindowDistanceMeters = 0.35f;
    public float expectedNominalWindowDistanceMeters = 0.75f;
    public float maxDepthMeters = 8f;
    public float nearStuckDistanceMeters = 0.18f;

    private const BindingFlags InstanceFlags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

    private float _nextLogTime;
    private Vector3 _lastCameraPosition;
    private Vector3 _lastSamplingWindowPosition;
    private bool _hasLastPositions;
    private ComputeBuffer _rawDepthProbeBuffer;
    private AsyncGPUReadbackRequest? _rawDepthReadback;
    private RawDepthProbeStats _lastRawDepthStats;
    private bool _hasRawDepthStats;

    private void Start()
    {
        ResolveRefs();
        if (logOnStart)
            LogDiagnostics("start");
        _nextLogTime = Time.unscaledTime + Mathf.Max(0.1f, logEverySeconds);
    }

    private void Update()
    {
        if (!continuousLogging)
            return;

        if (Time.unscaledTime < _nextLogTime)
            return;

        ResolveRefs();
        LogDiagnostics("tick");
        _nextLogTime = Time.unscaledTime + Mathf.Max(0.1f, logEverySeconds);
    }

    [ContextMenu("Log Sampling Window Diagnostics Now")]
    public void LogNow()
    {
        ResolveRefs();
        LogDiagnostics("manual");
    }

    private void OnDestroy()
    {
        if (_rawDepthProbeBuffer != null)
        {
            _rawDepthProbeBuffer.Dispose();
            _rawDepthProbeBuffer = null;
        }
    }

    private void ResolveRefs()
    {
        if (targetWindow == null)
            targetWindow = FindObjectOfType<ScanCoverDepthPointBurstWindow>(true);

        if (depthRaycaster == null && targetWindow != null)
            depthRaycaster = GetFieldValue<CustomEnvironmentDepthRaycaster>(targetWindow, "depthRaycaster");
        if (depthRaycaster == null)
            depthRaycaster = FindObjectOfType<CustomEnvironmentDepthRaycaster>(true);

        if (environmentDepthManager == null && targetWindow != null)
            environmentDepthManager = GetFieldValue<EnvironmentDepthManager>(targetWindow, "environmentDepthManager");
        if (environmentDepthManager == null && depthRaycaster != null)
            environmentDepthManager = depthRaycaster.depthManager;
        if (environmentDepthManager == null)
            environmentDepthManager = FindObjectOfType<EnvironmentDepthManager>(true);

        if (centerCamera == null && targetWindow != null)
            centerCamera = GetFieldValue<Camera>(targetWindow, "centerCamera");
        if (centerCamera == null)
            centerCamera = Camera.main;
        if (centerCamera == null)
            centerCamera = FindNamedCamera("CenterEyeAnchor");
    }

    private void LogDiagnostics(string reason)
    {
        var sb = new StringBuilder(4096);
        sb.AppendLine($"[ScanCoverSamplingWindowDiagnosticTool] reason={reason} frame={Time.frameCount} time={Time.unscaledTime:F2}");

        AppendObjectState(sb);
        AppendTargetWindowState(sb);
        AppendDepthRaycasterState(sb);
        AppendEnvironmentDepthManagerInternals(sb);
        AppendGlobalDepthState(sb);
        AppendRawEnvironmentDepthProbeState(sb);
        AppendSamplingWindowObjectState(sb);

        DepthScanStats scanStats = default;
        CenterProbeStats centerStats = default;
        bool hasScanStats = false;
        bool hasCenterStats = false;

        if (depthRaycaster != null && centerCamera != null)
        {
            Eye selectedEye = GetFieldValue(targetWindow, "eye", Eye.Right);
            depthRaycaster.SetEye(selectedEye);
            centerStats = ProbeCenter(selectedEye);
            hasCenterStats = true;
            AppendCenterProbeState(sb, centerStats);

            if (includeDepthTextureScan)
            {
                scanStats = ScanDepthTexture();
                hasScanStats = true;
                AppendDepthScanState(sb, scanStats);
                AppendEyeComparisonState(sb);
            }
        }

        AppendHierarchyScan(sb);
        AppendConclusion(sb, hasCenterStats ? centerStats : (CenterProbeStats?)null, hasScanStats ? scanStats : (DepthScanStats?)null);

        Debug.Log(sb.ToString(), this);
    }

    private void AppendObjectState(StringBuilder sb)
    {
        sb.AppendLine("  refs:");
        AppendRefLine(sb, "targetWindow", targetWindow);
        AppendRefLine(sb, "depthRaycaster", depthRaycaster);
        AppendRefLine(sb, "environmentDepthManager", environmentDepthManager);
        AppendRefLine(sb, "centerCamera", centerCamera);
    }

    private void AppendTargetWindowState(StringBuilder sb)
    {
        if (targetWindow == null)
        {
            sb.AppendLine("  targetWindow: missing");
            return;
        }

        bool showSamplingWindow = GetFieldValue(targetWindow, "showSamplingWindow", false);
        bool useWorldSpaceDisplayRoot = GetFieldValue(targetWindow, "useWorldSpaceDisplayRoot", false);
        bool usePhysical = GetFieldValue(targetWindow, "useCameraCenterHitPhysicalWindow", false);
        bool useDepthCenter = GetFieldValue(targetWindow, "useDepthCenterTexCoordForPhysicalWindow", false);
        float sampleDistance = GetFieldValue(targetWindow, "samplingWindowDistanceMeters", -1f);
        float minPhysical = GetFieldValue(targetWindow, "physicalWindowMinCenterDistanceMeters", -1f);
        int searchRadius = GetFieldValue(targetWindow, "physicalWindowCenterSearchRadiusPixels", -1);
        bool hasActive = GetFieldValue(targetWindow, "_hasActiveWorldWindow", false);

        sb.AppendLine("  pointBurstWindow:");
        sb.AppendLine($"    enabled={targetWindow.enabled} active={targetWindow.gameObject.activeInHierarchy} path={GetPath(targetWindow.transform)}");
        sb.AppendLine($"    showSamplingWindow={showSamplingWindow} useWorldSpaceDisplayRoot={useWorldSpaceDisplayRoot} usePhysicalWindow={usePhysical} useDepthCenterTexCoord={useDepthCenter}");
        sb.AppendLine($"    samplingWindowDistance={sampleDistance:F3} physicalMinDistance={minPhysical:F3} physicalSearchRadius={searchRadius} hasActiveWorldWindow={hasActive}");

        object activeWindow = GetBoxedField(targetWindow, "_activeWorldWindow");
        if (activeWindow != null)
        {
            Vector3 center = GetStructField(activeWindow, "center", Vector3.zero);
            float depth = GetStructField(activeWindow, "centerLinearDepth", -1f);
            int x = GetStructField(activeWindow, "centerTextureX", -1);
            int y = GetStructField(activeWindow, "centerTextureY", -1);
            float distance = centerCamera != null ? Vector3.Distance(centerCamera.transform.position, center) : -1f;
            sb.AppendLine($"    activeWindow center={Fmt(center)} camDistance={distance:F3} linearDepth={depth:F3} tex=({x},{y})");
        }
    }

    private void AppendDepthRaycasterState(StringBuilder sb)
    {
        if (depthRaycaster == null)
        {
            sb.AppendLine("  depthRaycaster: missing");
            return;
        }

        sb.AppendLine("  depthRaycaster:");
        sb.AppendLine($"    enabled={depthRaycaster.enabled} active={depthRaycaster.gameObject.activeInHierarchy} depthTextureAvailable={depthRaycaster.IsDepthTextureAvailable} path={GetPath(depthRaycaster.transform)}");
        AppendRefLine(sb, "    raycaster.depthManager", depthRaycaster.depthManager);

        if (environmentDepthManager != null)
        {
            sb.AppendLine("  environmentDepthManager:");
            sb.AppendLine($"    enabled={environmentDepthManager.enabled} active={environmentDepthManager.gameObject.activeInHierarchy} path={GetPath(environmentDepthManager.transform)}");
            sb.AppendLine($"    reflected available={ReadReflectedBool(environmentDepthManager, "IsDepthAvailable")} supported={ReadReflectedBool(environmentDepthManager, "IsSupported")}/{ReadReflectedBool(environmentDepthManager, "IsDepthSupported")}");
        }
    }

    private void AppendEnvironmentDepthManagerInternals(StringBuilder sb)
    {
        if (environmentDepthManager == null)
            return;

        object provider = GetStaticField(environmentDepthManager.GetType(), "_provider");
        object cameraRig = GetBoxedField(environmentDepthManager, "_cameraRig");
        object frameDescriptors = GetBoxedField(environmentDepthManager, "frameDescriptors");

        sb.AppendLine("  environmentDepthInternals:");
        sb.AppendLine($"    occlusionMode={GetBoxedField(environmentDepthManager, "_occlusionShadersMode")} removeHands={GetBoxedField(environmentDepthManager, "_removeHands")} hasPermission={GetBoxedField(environmentDepthManager, "_hasPermission")}");
        sb.AppendLine($"    customTrackingSpace={RefName(environmentDepthManager.CustomTrackingSpace)} cameraRig={RefName(cameraRig as Component)}");
        if (provider == null)
        {
            sb.AppendLine("    provider=missing");
        }
        else
        {
            sb.AppendLine($"    provider={provider.GetType().FullName}");
            sb.AppendLine($"    provider.IsSupported={ReadReflectedBool(provider, "IsSupported")} provider.running={ReadNestedRunning(provider)}");
            sb.AppendLine($"    provider.prevTexture={ReadProviderTextureState(provider)}");
        }

        AppendFrameDescriptors(sb, frameDescriptors as Array);
    }

    private static void AppendFrameDescriptors(StringBuilder sb, Array frameDescriptors)
    {
        if (frameDescriptors == null || frameDescriptors.Length == 0)
        {
            sb.AppendLine("    frameDescriptors=missing");
            return;
        }

        for (int i = 0; i < Mathf.Min(2, frameDescriptors.Length); i++)
        {
            object desc = frameDescriptors.GetValue(i);
            float nearZ = GetObjectField(desc, "nearZ", -1f);
            float farZ = GetObjectField(desc, "farZ", -1f);
            Vector3 pose = GetObjectField(desc, "createPoseLocation", Vector3.zero);
            Quaternion rotation = GetObjectField(desc, "createPoseRotation", Quaternion.identity);
            float left = GetObjectField(desc, "fovLeftAngleTangent", 0f);
            float right = GetObjectField(desc, "fovRightAngleTangent", 0f);
            float top = GetObjectField(desc, "fovTopAngleTangent", 0f);
            float down = GetObjectField(desc, "fovDownAngleTangent", 0f);
            sb.AppendLine($"    frameDesc[{i}] near={nearZ:F3} far={farZ:F3} pose={Fmt(pose)} rot=({rotation.x:F3},{rotation.y:F3},{rotation.z:F3},{rotation.w:F3}) fov=({left:F3},{right:F3},{top:F3},{down:F3})");
        }
    }

    private void AppendGlobalDepthState(StringBuilder sb)
    {
        Texture depthTexture = Shader.GetGlobalTexture(EnvironmentDepthTextureId);
        Vector4 zParams = Shader.GetGlobalVector(EnvironmentDepthZBufferParamsId);

        sb.AppendLine("  globalDepthShaderState:");
        if (depthTexture == null)
        {
            sb.AppendLine("    _EnvironmentDepthTexture=missing");
        }
        else
        {
            sb.AppendLine($"    _EnvironmentDepthTexture={depthTexture.name} type={depthTexture.GetType().Name} size={depthTexture.width}x{depthTexture.height}");
        }

        sb.AppendLine($"    _EnvironmentDepthZBufferParams=({zParams.x:F6},{zParams.y:F6},{zParams.z:F6},{zParams.w:F6})");
        if (Mathf.Abs(zParams.x) < 1e-6f && Mathf.Abs(zParams.y) < 1e-6f)
            sb.AppendLine("    warning=ZBufferParams look zero; depth-to-linear conversion can collapse.");
    }

    private void AppendRawEnvironmentDepthProbeState(StringBuilder sb)
    {
        if (!includeRawEnvironmentDepthProbe)
            return;

        RequestRawEnvironmentDepthProbe();

        sb.AppendLine("  rawEnvironmentDepthProbe:");
        if (!_hasRawDepthStats)
        {
            sb.AppendLine(_rawDepthReadback.HasValue ? "    status=pending" : "    status=not-started");
            return;
        }

        sb.AppendLine($"    frame={_lastRawDepthStats.frame} total={_lastRawDepthStats.total} finite={_lastRawDepthStats.finite} positive={_lastRawDepthStats.positive}");
        sb.AppendLine($"    rawRange={_lastRawDepthStats.minRaw:F6}-{_lastRawDepthStats.maxRaw:F6} zeroOrNegative={_lastRawDepthStats.zeroOrNegative}");
        sb.AppendLine($"    left raw={_lastRawDepthStats.leftMinRaw:F6}-{_lastRawDepthStats.leftMaxRaw:F6} positive={_lastRawDepthStats.leftPositive}");
        sb.AppendLine($"    right raw={_lastRawDepthStats.rightMinRaw:F6}-{_lastRawDepthStats.rightMaxRaw:F6} positive={_lastRawDepthStats.rightPositive}");
    }

    private void RequestRawEnvironmentDepthProbe()
    {
        if (_rawDepthReadback.HasValue && !_rawDepthReadback.Value.done)
            return;

        if (_rawDepthReadback.HasValue && _rawDepthReadback.Value.done)
            _rawDepthReadback = null;

        if (rawDepthProbeShader == null)
            rawDepthProbeShader = Resources.Load<ComputeShader>("ScanCoverRawEnvironmentDepthProbe");
        if (rawDepthProbeShader == null)
            return;

        Texture depthTexture = Shader.GetGlobalTexture(EnvironmentDepthTextureId);
        if (depthTexture == null)
            return;

        const int textureSize = 128;
        const int eyeCount = 2;
        int count = textureSize * textureSize * eyeCount;
        if (_rawDepthProbeBuffer == null)
            _rawDepthProbeBuffer = new ComputeBuffer(count, sizeof(float));

        int kernel = rawDepthProbeShader.FindKernel("CopyRaw");
        rawDepthProbeShader.SetTexture(kernel, EnvironmentDepthTextureId, depthTexture);
        rawDepthProbeShader.SetFloat("_EnvironmentDepthTextureSize", depthTexture.width);
        rawDepthProbeShader.SetBuffer(kernel, "_RawDepthProbe", _rawDepthProbeBuffer);
        rawDepthProbeShader.Dispatch(kernel, 1, 1, 1);

        _rawDepthReadback = AsyncGPUReadback.Request(_rawDepthProbeBuffer, request =>
        {
            if (request.hasError)
                return;

            var data = request.GetData<float>();
            _lastRawDepthStats = BuildRawDepthProbeStats(data, textureSize);
            _lastRawDepthStats.frame = Time.frameCount;
            _hasRawDepthStats = true;
        });
    }

    private static RawDepthProbeStats BuildRawDepthProbeStats(Unity.Collections.NativeArray<float> data, int textureSize)
    {
        RawDepthProbeStats stats = new RawDepthProbeStats { total = data.Length };
        int eyePixels = textureSize * textureSize;
        for (int i = 0; i < data.Length; i++)
        {
            float v = data[i];
            if (float.IsNaN(v) || float.IsInfinity(v))
                continue;

            stats.finite++;
            if (v <= 0f)
            {
                stats.zeroOrNegative++;
                continue;
            }

            stats.positive++;
            stats.minRaw = stats.positive == 1 ? v : Mathf.Min(stats.minRaw, v);
            stats.maxRaw = stats.positive == 1 ? v : Mathf.Max(stats.maxRaw, v);

            if (i < eyePixels)
            {
                stats.leftPositive++;
                stats.leftMinRaw = stats.leftPositive == 1 ? v : Mathf.Min(stats.leftMinRaw, v);
                stats.leftMaxRaw = stats.leftPositive == 1 ? v : Mathf.Max(stats.leftMaxRaw, v);
            }
            else
            {
                stats.rightPositive++;
                stats.rightMinRaw = stats.rightPositive == 1 ? v : Mathf.Min(stats.rightMinRaw, v);
                stats.rightMaxRaw = stats.rightPositive == 1 ? v : Mathf.Max(stats.rightMaxRaw, v);
            }
        }

        return stats;
    }

    private void AppendSamplingWindowObjectState(StringBuilder sb)
    {
        Transform samplingWindow = FindTransformByName("Depth Point Burst Sampling Window");
        Transform samplesRoot = FindTransformByName("Depth Point Burst Window Samples");
        Transform displayRoot = FindTransformByName("[ScanCover] Depth Burst World Display");

        sb.AppendLine("  displayObjects:");
        AppendTransformLine(sb, "displayRoot", displayRoot);
        AppendTransformLine(sb, "samplingWindow", samplingWindow);
        AppendTransformLine(sb, "samplesRoot", samplesRoot);

        if (samplingWindow != null && centerCamera != null)
        {
            float camDistance = Vector3.Distance(centerCamera.transform.position, samplingWindow.position);
            float localMagnitude = samplingWindow.localPosition.magnitude;
            sb.AppendLine($"    samplingWindowDistanceToCamera={camDistance:F3} localMagnitude={localMagnitude:F3}");

            if (_hasLastPositions)
            {
                Vector3 cameraDelta = centerCamera.transform.position - _lastCameraPosition;
                Vector3 windowDelta = samplingWindow.position - _lastSamplingWindowPosition;
                float followDot = cameraDelta.sqrMagnitude > 1e-6f && windowDelta.sqrMagnitude > 1e-6f
                    ? Vector3.Dot(cameraDelta.normalized, windowDelta.normalized)
                    : 0f;
                sb.AppendLine($"    sinceLastLog cameraDelta={cameraDelta.magnitude:F4} windowDelta={windowDelta.magnitude:F4} movementDot={followDot:F3}");
            }

            _lastCameraPosition = centerCamera.transform.position;
            _lastSamplingWindowPosition = samplingWindow.position;
            _hasLastPositions = true;
        }
    }

    private CenterProbeStats ProbeCenter(Eye selectedEye)
    {
        CenterProbeStats stats = new CenterProbeStats();
        stats.eye = selectedEye;
        stats.depthTextureAvailable = depthRaycaster != null && depthRaycaster.IsDepthTextureAvailable;

        if (depthRaycaster == null || centerCamera == null)
            return stats;

        Vector3 camPos = centerCamera.transform.position;
        Vector3 forward = centerCamera.transform.forward;
        float nominalDistance = GetFieldValue(targetWindow, "samplingWindowDistanceMeters", expectedNominalWindowDistanceMeters);
        float minDistance = Mathf.Max(GetFieldValue(targetWindow, "minLinearDepthMeters", 0.05f), expectedMinWindowDistanceMeters);
        float maxDistance = Mathf.Max(minDistance, Mathf.Max(GetFieldValue(targetWindow, "maxLinearDepthMeters", maxDepthMeters), maxDepthMeters));

        Vector3 forwardPoint = camPos + forward * Mathf.Max(0.25f, nominalDistance);
        stats.forwardTexCoord = depthRaycaster.WorldPosToNonNormalizedTextureCoords02(forwardPoint);
        stats.forwardTexDepth = depthRaycaster.SampleDepthTexture02(stats.forwardTexCoord);
        stats.forwardTexWorld = depthRaycaster.WorldPosAtDepthTexCoord02(stats.forwardTexCoord);
        stats.forwardTexDistance = IsFinite(stats.forwardTexWorld) ? Vector3.Distance(camPos, stats.forwardTexWorld) : -1f;
        stats.forwardTexLinearDepth = IsFinite(stats.forwardTexWorld) ? depthRaycaster.WorldPosToLinearDepth02(stats.forwardTexWorld) : -1f;

        Ray centerRay = new Ray(camPos, forward);
        var rayHit = depthRaycaster.Raycast02(centerRay, maxDistance, selectedEye, true);
        stats.raycastStatus = rayHit.status.ToString();
        stats.raycastWorld = rayHit.position;
        stats.raycastDistance = IsFinite(rayHit.position) ? Vector3.Distance(camPos, rayHit.position) : -1f;
        stats.raycastEyeIndex = rayHit.eyeIndex;

        int textureSize = CustomEnvironmentDepthRaycaster.TextureSize;
        int radius = Mathf.Clamp(sampleRadiusPixels, 0, textureSize - 1);
        stats.searchRadius = radius;
        stats.searchBestDistanceScore = float.PositiveInfinity;

        for (int dy = -radius; dy <= radius; dy++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                int x = stats.forwardTexCoord.x + dx;
                int y = stats.forwardTexCoord.y + dy;
                if (x < 0 || y < 0 || x >= textureSize || y >= textureSize)
                    continue;

                Vector2Int tc = new Vector2Int(x, y);
                float raw = depthRaycaster.SampleDepthTexture02(tc);
                if (float.IsNaN(raw) || float.IsInfinity(raw) || raw <= 0f)
                    continue;

                stats.searchFiniteDepthCount++;
                Vector3 world = depthRaycaster.WorldPosAtDepthTexCoord02(tc);
                if (!IsFinite(world))
                    continue;

                float dist = Vector3.Distance(camPos, world);
                stats.searchFiniteWorldCount++;
                stats.searchMinDistance = stats.searchFiniteWorldCount == 1 ? dist : Mathf.Min(stats.searchMinDistance, dist);
                stats.searchMaxDistance = stats.searchFiniteWorldCount == 1 ? dist : Mathf.Max(stats.searchMaxDistance, dist);

                if (dist < minDistance || dist > maxDistance)
                    continue;

                stats.searchInExpectedRangeCount++;
                float texelScore = dx * dx + dy * dy;
                float distanceScore = Mathf.Abs(dist - Mathf.Max(minDistance, nominalDistance)) * 0.1f;
                float score = texelScore + distanceScore;
                if (score < stats.searchBestDistanceScore)
                {
                    stats.searchBestDistanceScore = score;
                    stats.searchBestTexCoord = tc;
                    stats.searchBestWorld = world;
                    stats.searchBestDistance = dist;
                    stats.searchBestRawDepth = raw;
                }
            }
        }

        return stats;
    }

    private void AppendCenterProbeState(StringBuilder sb, CenterProbeStats stats)
    {
        sb.AppendLine("  independentCenterProbe:");
        sb.AppendLine($"    eye={stats.eye} depthTextureAvailable={stats.depthTextureAvailable} forwardTex={stats.forwardTexCoord} rawDepth={stats.forwardTexDepth:F3}");
        sb.AppendLine($"    forwardTexWorld={Fmt(stats.forwardTexWorld)} camDistance={stats.forwardTexDistance:F3} linearDepth={stats.forwardTexLinearDepth:F3}");
        sb.AppendLine($"    raycast status={stats.raycastStatus} eyeIndex={stats.raycastEyeIndex} world={Fmt(stats.raycastWorld)} camDistance={stats.raycastDistance:F3}");
        sb.AppendLine($"    localSearch radius={stats.searchRadius} finiteDepth={stats.searchFiniteDepthCount} finiteWorld={stats.searchFiniteWorldCount} inExpectedRange={stats.searchInExpectedRangeCount} distanceRange={stats.searchMinDistance:F3}-{stats.searchMaxDistance:F3}");
        if (stats.searchInExpectedRangeCount > 0)
            sb.AppendLine($"    localSearchBest tex={stats.searchBestTexCoord} rawDepth={stats.searchBestRawDepth:F3} world={Fmt(stats.searchBestWorld)} camDistance={stats.searchBestDistance:F3}");
    }

    private DepthScanStats ScanDepthTexture()
    {
        DepthScanStats stats = new DepthScanStats();
        if (depthRaycaster == null || centerCamera == null)
            return stats;

        int textureSize = CustomEnvironmentDepthRaycaster.TextureSize;
        Vector3 camPos = centerCamera.transform.position;
        for (int y = 0; y < textureSize; y++)
        {
            for (int x = 0; x < textureSize; x++)
            {
                Vector2Int tc = new Vector2Int(x, y);
                float raw = depthRaycaster.SampleDepthTexture02(tc);
                stats.total++;
                if (float.IsNaN(raw) || float.IsInfinity(raw))
                    continue;

                stats.finiteRaw++;
                if (raw <= 0f)
                    continue;

                stats.positiveRaw++;
                stats.minRaw = stats.positiveRaw == 1 ? raw : Mathf.Min(stats.minRaw, raw);
                stats.maxRaw = stats.positiveRaw == 1 ? raw : Mathf.Max(stats.maxRaw, raw);

                Vector3 world = depthRaycaster.WorldPosAtDepthTexCoord02(tc);
                if (!IsFinite(world))
                    continue;

                float distance = Vector3.Distance(camPos, world);
                stats.finiteWorld++;
                stats.minDistance = stats.finiteWorld == 1 ? distance : Mathf.Min(stats.minDistance, distance);
                stats.maxDistance = stats.finiteWorld == 1 ? distance : Mathf.Max(stats.maxDistance, distance);

                if (distance < nearStuckDistanceMeters)
                    stats.nearStuckCount++;
                else if (distance < expectedMinWindowDistanceMeters)
                    stats.underExpectedMinCount++;
                else
                    stats.atOrBeyondExpectedMinCount++;
            }
        }

        return stats;
    }

    private void AppendDepthScanState(StringBuilder sb, DepthScanStats stats)
    {
        sb.AppendLine("  depthTextureScan:");
        sb.AppendLine($"    total={stats.total} finiteRaw={stats.finiteRaw} positiveRaw={stats.positiveRaw} finiteWorld={stats.finiteWorld}");
        sb.AppendLine($"    rawDepthRange={stats.minRaw:F3}-{stats.maxRaw:F3} worldDistanceRange={stats.minDistance:F3}-{stats.maxDistance:F3}");
        sb.AppendLine($"    distanceBuckets near<{nearStuckDistanceMeters:F2}m={stats.nearStuckCount} underExpected<{expectedMinWindowDistanceMeters:F2}m={stats.underExpectedMinCount} usable>={expectedMinWindowDistanceMeters:F2}m={stats.atOrBeyondExpectedMinCount}");
    }

    private void AppendEyeComparisonState(StringBuilder sb)
    {
        if (depthRaycaster == null || centerCamera == null)
            return;

        Eye originalEye = GetFieldValue(targetWindow, "eye", Eye.Right);
        DepthScanStats left = ScanDepthTextureForEye(Eye.Left);
        DepthScanStats right = ScanDepthTextureForEye(Eye.Right);
        depthRaycaster.SetEye(originalEye);

        sb.AppendLine("  eyeComparison:");
        sb.AppendLine($"    left positive={left.positiveRaw} finiteWorld={left.finiteWorld} raw={left.minRaw:F3}-{left.maxRaw:F3} distance={left.minDistance:F3}-{left.maxDistance:F3} usable>={expectedMinWindowDistanceMeters:F2}m={left.atOrBeyondExpectedMinCount}");
        sb.AppendLine($"    right positive={right.positiveRaw} finiteWorld={right.finiteWorld} raw={right.minRaw:F3}-{right.maxRaw:F3} distance={right.minDistance:F3}-{right.maxDistance:F3} usable>={expectedMinWindowDistanceMeters:F2}m={right.atOrBeyondExpectedMinCount}");
    }

    private DepthScanStats ScanDepthTextureForEye(Eye eye)
    {
        depthRaycaster.SetEye(eye);
        return ScanDepthTexture();
    }

    private void AppendHierarchyScan(StringBuilder sb)
    {
        if (!includeHierarchyScan)
            return;

        sb.AppendLine("  hierarchyWarnings:");
        bool any = false;
        any |= AppendNamedPathWarning(sb, "CenterEyeAnchor");
        any |= AppendNamedPathWarning(sb, "ScanCoverSkeleton");
        any |= AppendNamedPathWarning(sb, "Depth Point Burst Sampling Window");
        any |= AppendNamedPathWarning(sb, "Depth Point Burst Window Samples");
        any |= AppendNamedPathWarning(sb, "[ScanCover] Depth Burst World Display");
        if (!any)
            sb.AppendLine("    none");
    }

    private bool AppendNamedPathWarning(StringBuilder sb, string name)
    {
        Transform t = FindTransformByName(name);
        if (t == null)
            return false;

        sb.AppendLine($"    {name}: path={GetPath(t)} world={Fmt(t.position)} local={Fmt(t.localPosition)}");
        return true;
    }

    private void AppendConclusion(StringBuilder sb, CenterProbeStats? centerStats, DepthScanStats? scanStats)
    {
        sb.AppendLine("  likelyCause:");

        if (environmentDepthManager == null)
        {
            sb.AppendLine("    ENV_MISSING: no EnvironmentDepthManager reference was found.");
            return;
        }

        if (depthRaycaster == null)
        {
            sb.AppendLine("    RAYCASTER_MISSING: no CustomEnvironmentDepthRaycaster is available to consume environment depth.");
            return;
        }

        if (!depthRaycaster.IsDepthTextureAvailable)
        {
            sb.AppendLine("    DEPTH_TEXTURE_NOT_READY: raycaster exists, but its copied depth texture is not available yet.");
            return;
        }

        Transform samplingWindow = FindTransformByName("Depth Point Burst Sampling Window");
        bool samplingUnderHead = samplingWindow != null && IsUnderName(samplingWindow, "CenterEyeAnchor");
        bool samplingUnderSkeleton = samplingWindow != null && IsUnderName(samplingWindow, "ScanCoverSkeleton");
        if (samplingUnderHead || samplingUnderSkeleton)
            sb.AppendLine($"    PARENT_SPACE_RISK: sampling window is under {(samplingUnderHead ? "CenterEyeAnchor" : "ScanCoverSkeleton")}, so local coordinates can make it follow the headset.");

        if (scanStats.HasValue)
        {
            DepthScanStats scan = scanStats.Value;
            if (scan.positiveRaw == 0 || scan.finiteWorld == 0)
            {
                sb.AppendLine("    DEPTH_VALUES_INVALID: depth readback has no usable positive/world depth values.");
                return;
            }

            if (scan.atOrBeyondExpectedMinCount == 0)
            {
                if (_hasRawDepthStats && _lastRawDepthStats.positive == 0)
                {
                    sb.AppendLine("    ENVIRONMENT_DEPTH_TEXTURE_ZERO: _EnvironmentDepthTexture is present, but direct GPU readback is all zero. This is upstream of CustomEnvironmentDepthRaycaster and upstream of the sampling window.");
                    return;
                }

                sb.AppendLine("    DEPTH_COLLAPSED_NEAR_CAMERA: depth texture/world reconstruction contains no points beyond expected minimum distance. Window placement will collapse near CenterEyeAnchor even if parenting is correct.");
                return;
            }
        }

        if (centerStats.HasValue)
        {
            CenterProbeStats center = centerStats.Value;
            if (center.searchInExpectedRangeCount == 0 && center.raycastDistance > 0f && center.raycastDistance < expectedMinWindowDistanceMeters)
            {
                sb.AppendLine("    CENTER_DEPTH_TOO_NEAR: center ray/texcoord resolves to a near-camera point and local search found no acceptable replacement.");
                return;
            }

            if (center.searchInExpectedRangeCount > 0 && samplingWindow != null && centerCamera != null)
            {
                float displayDistance = Vector3.Distance(centerCamera.transform.position, samplingWindow.position);
                if (displayDistance < expectedMinWindowDistanceMeters * 0.5f)
                {
                    sb.AppendLine("    DISPLAY_ROUTE_MISMATCH: independent depth search found a valid farther center, but the visible sampling window is still close to camera. Focus on UpdateSamplingWindowDisplay parent/local transform path.");
                    return;
                }
            }
        }

        sb.AppendLine("    NO_SINGLE_ROOT_CAUSE: core refs and depth look plausible. Compare independentCenterProbe with visible samplingWindowDistanceToCamera above.");
    }

    private static void AppendRefLine(StringBuilder sb, string label, Component component)
    {
        if (component == null)
        {
            sb.AppendLine($"    {label}=missing");
            return;
        }

        sb.AppendLine($"    {label}={component.name} enabled={ReadEnabled(component)} active={component.gameObject.activeInHierarchy} path={GetPath(component.transform)}");
    }

    private static void AppendTransformLine(StringBuilder sb, string label, Transform transform)
    {
        if (transform == null)
        {
            sb.AppendLine($"    {label}=missing");
            return;
        }

        sb.AppendLine($"    {label} path={GetPath(transform)} world={Fmt(transform.position)} local={Fmt(transform.localPosition)}");
    }

    private static bool ReadEnabled(Component component)
    {
        if (component is Behaviour behaviour)
            return behaviour.enabled;
        return component != null;
    }

    private static Camera FindNamedCamera(string name)
    {
        Camera[] cameras = FindObjectsOfType<Camera>(true);
        foreach (Camera cam in cameras)
        {
            if (cam != null && cam.name == name)
                return cam;
        }

        return null;
    }

    private static Transform FindTransformByName(string name)
    {
        Transform[] transforms = FindObjectsOfType<Transform>(true);
        foreach (Transform t in transforms)
        {
            if (t != null && t.name == name)
                return t;
        }

        return null;
    }

    private static bool IsUnderName(Transform transform, string ancestorName)
    {
        for (Transform t = transform; t != null; t = t.parent)
        {
            if (t.name == ancestorName)
                return true;
        }

        return false;
    }

    private static string GetPath(Transform transform)
    {
        if (transform == null)
            return "missing";

        string path = transform.name;
        for (Transform p = transform.parent; p != null; p = p.parent)
            path = p.name + "/" + path;
        return path;
    }

    private static string Fmt(Vector3 v)
    {
        return $"({v.x:F3},{v.y:F3},{v.z:F3})";
    }

    private static bool IsFinite(Vector3 v)
    {
        return !float.IsNaN(v.x) && !float.IsNaN(v.y) && !float.IsNaN(v.z) &&
               !float.IsInfinity(v.x) && !float.IsInfinity(v.y) && !float.IsInfinity(v.z);
    }

    private static T GetFieldValue<T>(object target, string fieldName)
    {
        return GetFieldValue(target, fieldName, default(T));
    }

    private static T GetFieldValue<T>(object target, string fieldName, T fallback)
    {
        if (target == null)
            return fallback;

        FieldInfo field = target.GetType().GetField(fieldName, InstanceFlags);
        if (field == null)
            return fallback;

        object value = field.GetValue(target);
        if (value is T typed)
            return typed;

        return fallback;
    }

    private static object GetBoxedField(object target, string fieldName)
    {
        if (target == null)
            return null;

        FieldInfo field = target.GetType().GetField(fieldName, InstanceFlags);
        return field != null ? field.GetValue(target) : null;
    }

    private static T GetStructField<T>(object boxedStruct, string fieldName, T fallback)
    {
        if (boxedStruct == null)
            return fallback;

        FieldInfo field = boxedStruct.GetType().GetField(fieldName, InstanceFlags);
        if (field == null)
            return fallback;

        object value = field.GetValue(boxedStruct);
        return value is T typed ? typed : fallback;
    }

    private static string ReadReflectedBool(object target, string propertyName)
    {
        if (target == null)
            return "n/a";

        PropertyInfo property = target.GetType().GetProperty(propertyName, InstanceFlags);
        if (property == null || property.PropertyType != typeof(bool))
            return "n/a";

        try
        {
            return ((bool)property.GetValue(target)).ToString();
        }
        catch (Exception)
        {
            return "error";
        }
    }

    private static object GetStaticField(Type type, string fieldName)
    {
        if (type == null)
            return null;

        FieldInfo field = type.GetField(fieldName, BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
        return field != null ? field.GetValue(null) : null;
    }

    private static T GetObjectField<T>(object target, string fieldName, T fallback)
    {
        if (target == null)
            return fallback;

        FieldInfo field = target.GetType().GetField(fieldName, InstanceFlags);
        if (field == null)
            return fallback;

        object value = field.GetValue(target);
        return value is T typed ? typed : fallback;
    }

    private static string RefName(Component component)
    {
        return component != null ? $"{component.name} path={GetPath(component.transform)}" : "missing";
    }

    private static string ReadNestedRunning(object provider)
    {
        if (provider == null)
            return "n/a";

        object occlusionSubsystem = GetBoxedField(provider, "_occlusionSubsystem");
        if (occlusionSubsystem != null)
        {
            string running = ReadReflectedBool(occlusionSubsystem, "running");
            string handSupported = ReadReflectedPropertyString(occlusionSubsystem, "isHandRemovalSupported");
            return $"occlusionSubsystem.running={running} handRemovalSupported={handSupported}";
        }

        object displaySubsystem = GetBoxedField(provider, "_displaySubsystem");
        if (displaySubsystem != null)
            return $"displaySubsystem.running={ReadReflectedBool(displaySubsystem, "running")}";

        return "n/a";
    }

    private static string ReadProviderTextureState(object provider)
    {
        if (provider == null)
            return "n/a";

        object prevNativeTexture = GetBoxedField(provider, "_prevNativeTexture");
        object prevTextureId = GetBoxedField(provider, "_prevTextureId");
        if (prevNativeTexture != null)
            return prevNativeTexture.ToString();
        if (prevTextureId != null)
            return prevTextureId.ToString();
        return "n/a";
    }

    private static string ReadReflectedPropertyString(object target, string propertyName)
    {
        if (target == null)
            return "n/a";

        PropertyInfo property = target.GetType().GetProperty(propertyName, InstanceFlags);
        if (property == null)
            return "n/a";

        try
        {
            object value = property.GetValue(target);
            return value != null ? value.ToString() : "null";
        }
        catch (Exception)
        {
            return "error";
        }
    }

    private struct CenterProbeStats
    {
        public Eye eye;
        public bool depthTextureAvailable;
        public Vector2Int forwardTexCoord;
        public float forwardTexDepth;
        public Vector3 forwardTexWorld;
        public float forwardTexDistance;
        public float forwardTexLinearDepth;
        public string raycastStatus;
        public Vector3 raycastWorld;
        public float raycastDistance;
        public int raycastEyeIndex;
        public int searchRadius;
        public int searchFiniteDepthCount;
        public int searchFiniteWorldCount;
        public int searchInExpectedRangeCount;
        public float searchMinDistance;
        public float searchMaxDistance;
        public float searchBestDistanceScore;
        public Vector2Int searchBestTexCoord;
        public Vector3 searchBestWorld;
        public float searchBestDistance;
        public float searchBestRawDepth;
    }

    private struct DepthScanStats
    {
        public int total;
        public int finiteRaw;
        public int positiveRaw;
        public int finiteWorld;
        public float minRaw;
        public float maxRaw;
        public float minDistance;
        public float maxDistance;
        public int nearStuckCount;
        public int underExpectedMinCount;
        public int atOrBeyondExpectedMinCount;
    }

    private struct RawDepthProbeStats
    {
        public int frame;
        public int total;
        public int finite;
        public int positive;
        public int zeroOrNegative;
        public float minRaw;
        public float maxRaw;
        public int leftPositive;
        public float leftMinRaw;
        public float leftMaxRaw;
        public int rightPositive;
        public float rightMinRaw;
        public float rightMaxRaw;
    }
}
