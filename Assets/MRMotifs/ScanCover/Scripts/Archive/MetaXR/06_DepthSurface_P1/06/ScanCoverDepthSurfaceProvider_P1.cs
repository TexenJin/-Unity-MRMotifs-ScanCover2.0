using System;
using System.Collections.Generic;
using System.Reflection;
using Meta.XR;
using UnityEngine;

[DisallowMultipleComponent]
public class ScanCoverDepthSurfaceProvider_P1 : MonoBehaviour
{
    public enum DepthEyeMode
    {
        Left = 0,
        Right = 1,
        Both = 2,
    }

    public enum BackendMode
    {
        EnvironmentRaycast = 0,
        CustomDepthRaycasterReflection = 1,
    }

    [Serializable]
    public struct SurfaceObservation
    {
        public Vector3 worldPos;
        public Vector3 worldNormal;
        public float confidence;
    }

    [Header("Backend")]
    public BackendMode backend = BackendMode.EnvironmentRaycast;

    [Header("Refs")]
    public ScanCoverSkeletonBuilder_A builder;
    public Transform referenceFrame;
    public EnvironmentRaycastManager environmentRaycast;
    public Camera sampleCamera;
    public ScanCoverCustomDepthRaycaster_P1 scanCoverDepthRaycaster;
    [Tooltip("Legacy fallback. Leave empty when using Scan Cover Depth Raycaster.")]
    public MonoBehaviour customDepthRaycaster;

    [Header("Environment Raycast")]
    [Min(1)] public int raycastSamplesPerStep = 32;
    [Min(0.2f)] public float raycastMaxDistance = 6.0f;
    public bool acceptHitPointOccluded = true;
    [Range(0f, 0.2f)] public float viewportMargin = 0.03f;
    [Min(0f)] public float minHitDistanceMeters = 0.35f;
    public bool enableSelfExclusion = true;
    public Transform[] selfExcludeTransforms;
    [Min(0f)] public float selfExcludeRadiusMeters = 0.25f;
    public bool enableSamplingBoundsFilter = false;
    public Vector2 samplingViewportCenter = new Vector2(0.5f, 0.5f);
    public Vector2 samplingViewportSize = new Vector2(0.4f, 0.5f);
    [Range(0.2f, 1f)] public float samplingViewportMaxSize = 0.9f;
    public bool inheritBuilderSettings = true;

    [Header("Custom Depth Grid")]
    public DepthEyeMode depthEye = DepthEyeMode.Right;
    [Min(1)] public int depthStride = 3;
    [Min(1)] public int depthSamplesPerStep = 96;
    public bool depthPixelVFlip = true;
    [Min(0f)] public float depthMinMeters = 0.05f;
    [Min(0.1f)] public float depthMaxMeters = 8f;
    public bool orientNormalFromDepth = true;
    public bool neighborFill = true;
    [Min(1)] public int neighborRadius = 1;

    [Header("Debug")]
    public bool debugLog = false;

    public bool IsReady
    {
        get
        {
            ResolveRefs();
            if (backend == BackendMode.EnvironmentRaycast)
                return environmentRaycast != null && sampleCamera != null && EnvironmentRaycastManager.IsSupported;
            return GetActiveCustomDepthTarget() != null && BindReflectionIfNeeded();
        }
    }

    private readonly List<Vector2Int> _gridPx = new List<Vector2Int>(16384);
    private int _haltonIndex = 1;
    private int _gridCursor;
    private MethodInfo _miSetEye;
    private MethodInfo _miWorldPosAtDepthTexCoord02;
    private MethodInfo _miWorldPosToLinearDepth02;
    private MethodInfo _miReconstructNormal02;
    private Type _boundReflectionType;
    private object[] _setEyeArgs = new object[1];
    private object[] _vec2Args = new object[1];

    private void Awake()
    {
        ResolveRefs();
        RebuildDepthGrid();
    }

    private void OnEnable()
    {
        ResolveRefs();
        RebuildDepthGrid();
    }

    private void OnValidate()
    {
        depthStride = Mathf.Max(1, depthStride);
        depthSamplesPerStep = Mathf.Max(1, depthSamplesPerStep);
        raycastSamplesPerStep = Mathf.Max(1, raycastSamplesPerStep);
        neighborRadius = Mathf.Max(1, neighborRadius);
        RebuildDepthGrid();
    }

    public int CollectObservations(List<SurfaceObservation> outList, int budgetOverride = -1)
    {
        if (outList == null)
            return 0;

        ResolveRefs();
        outList.Clear();

        switch (backend)
        {
            case BackendMode.CustomDepthRaycasterReflection:
                return CollectDepthGridObservations(outList, budgetOverride > 0 ? budgetOverride : depthSamplesPerStep);
            default:
                return CollectRaycastObservations(outList, budgetOverride > 0 ? budgetOverride : raycastSamplesPerStep);
        }
    }

    private void ResolveRefs()
    {
        if (!builder)
            builder = GetComponent<ScanCoverSkeletonBuilder_A>();
        if (!referenceFrame && builder)
            referenceFrame = builder.referenceFrame;
        if (!referenceFrame)
            referenceFrame = transform;
        if (!environmentRaycast && builder)
            environmentRaycast = builder.environmentRaycast;
        if (!sampleCamera && builder)
            sampleCamera = builder.sampleCamera;
        if (!sampleCamera)
            sampleCamera = Camera.main;
        if (!scanCoverDepthRaycaster)
            scanCoverDepthRaycaster = GetComponent<ScanCoverCustomDepthRaycaster_P1>();
        if (scanCoverDepthRaycaster)
            customDepthRaycaster = scanCoverDepthRaycaster;

        if (!inheritBuilderSettings || builder == null)
            return;

        raycastMaxDistance = builder.maxRayDistance;
        acceptHitPointOccluded = builder.acceptHitPointOccluded;
        minHitDistanceMeters = builder.minHitDistanceMeters;
        enableSelfExclusion = builder.enableSelfExclusion;
        selfExcludeTransforms = builder.selfExcludeTransforms;
        selfExcludeRadiusMeters = builder.selfExcludeRadiusMeters;
        enableSamplingBoundsFilter = builder.enableSamplingBoundsFilter;
        samplingViewportCenter = builder.samplingViewportCenter;
        samplingViewportSize = builder.samplingViewportSize;
        samplingViewportMaxSize = builder.samplingViewportMaxSize;
    }

    private int CollectRaycastObservations(List<SurfaceObservation> outList, int budget)
    {
        if (!environmentRaycast || !sampleCamera || !EnvironmentRaycastManager.IsSupported)
            return 0;

        int added = 0;
        int stepCount = Mathf.Max(1, budget);
        for (int i = 0; i < stepCount; i++)
        {
            Vector2 uv = NextHalton2D();
            Ray ray = sampleCamera.ViewportPointToRay(new Vector3(uv.x, uv.y, 0f));

            bool hitSuccess = environmentRaycast.Raycast(ray, out var hit, maxDistance: raycastMaxDistance);
            bool usable = hitSuccess || (acceptHitPointOccluded && hit.status == EnvironmentRaycastHitStatus.HitPointOccluded);
            if (!usable)
                continue;

            Vector3 hitPos = hit.point;
            Vector3 hitNormal = hit.normal.sqrMagnitude > 1e-6f ? hit.normal.normalized : (-ray.direction).normalized;

            float hitDist = Vector3.Distance(ray.origin, hitPos);
            if (minHitDistanceMeters > 0f && hitDist < minHitDistanceMeters)
                continue;
            if (enableSelfExclusion && selfExcludeRadiusMeters > 0f && IsSelfExcluded(hitPos))
                continue;
            if (enableSamplingBoundsFilter && !IsInsideSamplingBounds(hitPos))
                continue;

            outList.Add(new SurfaceObservation
            {
                worldPos = hitPos,
                worldNormal = hitNormal,
                confidence = hitSuccess ? 1f : 0.6f,
            });
            added++;
        }

        return added;
    }

    private int CollectDepthGridObservations(List<SurfaceObservation> outList, int budget)
    {
        if (GetActiveCustomDepthTarget() == null || !BindReflectionIfNeeded())
            return 0;

        if (_gridPx.Count <= 0)
            RebuildDepthGrid();
        if (_gridPx.Count <= 0)
            return 0;

        if (!TryInvokeSetEye())
            return 0;

        int added = 0;
        int attempts = 0;
        int maxAttempts = Mathf.Max(budget * 3, budget + 8);
        while (added < budget && attempts < maxAttempts)
        {
            Vector2Int px = _gridPx[_gridCursor];
            _gridCursor = (_gridCursor + 1) % _gridPx.Count;
            attempts++;

            if (!TryGetDepthWorldPos(px, out Vector3 worldPos))
                continue;
            if (!IsFinite(worldPos) && neighborFill && !TryGetNeighborWorldPos(px, out worldPos))
                continue;
            if (!IsFinite(worldPos))
                continue;

            float linearDepth = InvokeLinearDepth(worldPos);
            if (linearDepth < depthMinMeters || linearDepth > depthMaxMeters)
                continue;
            if (enableSelfExclusion && selfExcludeRadiusMeters > 0f && IsSelfExcluded(worldPos))
                continue;
            if (enableSamplingBoundsFilter && !IsInsideSamplingBounds(worldPos))
                continue;

            Vector3 normal = orientNormalFromDepth && TryGetDepthNormal(px, out Vector3 n) ? n : Vector3.up;
            if (normal.sqrMagnitude <= 1e-6f)
                normal = Vector3.up;

            outList.Add(new SurfaceObservation
            {
                worldPos = worldPos,
                worldNormal = normal.normalized,
                confidence = 1f,
            });
            added++;
        }

        return added;
    }

    private bool TryInvokeSetEye()
    {
        if (_miSetEye == null)
            return false;

        ParameterInfo[] parameters = _miSetEye.GetParameters();
        if (parameters == null || parameters.Length != 1)
            return false;

        Type eyeType = parameters[0].ParameterType;
        object eyeArg;
        if (eyeType.IsEnum)
        {
            string name = depthEye.ToString();
            if (Enum.IsDefined(eyeType, name))
                eyeArg = Enum.Parse(eyeType, name);
            else
                eyeArg = Enum.GetValues(eyeType).GetValue(0);
        }
        else if (eyeType == typeof(int))
        {
            eyeArg = (int)depthEye;
        }
        else
        {
            return false;
        }

        _setEyeArgs[0] = eyeArg;
        _miSetEye.Invoke(GetActiveCustomDepthTarget(), _setEyeArgs);
        return true;
    }

    private bool BindReflectionIfNeeded()
    {
        MonoBehaviour target = GetActiveCustomDepthTarget();
        if (target == null)
            return false;

        Type t = target.GetType();
        if (_boundReflectionType == t &&
            _miSetEye != null &&
            _miWorldPosAtDepthTexCoord02 != null &&
            _miWorldPosToLinearDepth02 != null)
            return true;

        _boundReflectionType = t;
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        _miSetEye = t.GetMethod("SetEye", flags);
        _miWorldPosAtDepthTexCoord02 = t.GetMethod("WorldPosAtDepthTexCoord02", flags);
        _miWorldPosToLinearDepth02 = t.GetMethod("WorldPosToLinearDepth02", flags);
        _miReconstructNormal02 = t.GetMethod("ReconstructNormal02", flags);

        bool ok = _miSetEye != null && _miWorldPosAtDepthTexCoord02 != null && _miWorldPosToLinearDepth02 != null;
        if (!ok && debugLog)
            Debug.LogWarning("[ScanCoverDepthSurfaceProvider_P1] Custom depth raycaster reflection bind failed.");
        return ok;
    }

    private bool TryGetDepthWorldPos(Vector2Int px, out Vector3 worldPos)
    {
        worldPos = default;
        if (_miWorldPosAtDepthTexCoord02 == null)
            return false;
        _vec2Args[0] = px;
        object result = _miWorldPosAtDepthTexCoord02.Invoke(GetActiveCustomDepthTarget(), _vec2Args);
        if (result is Vector3 v)
        {
            worldPos = v;
            return IsFinite(worldPos);
        }
        return false;
    }

    private bool TryGetDepthNormal(Vector2Int px, out Vector3 normal)
    {
        normal = default;
        if (_miReconstructNormal02 == null)
            return false;
        _vec2Args[0] = px;
        object result = _miReconstructNormal02.Invoke(GetActiveCustomDepthTarget(), _vec2Args);
        if (result is Vector3 v && IsFinite(v) && v.sqrMagnitude > 1e-6f)
        {
            normal = v.normalized;
            return true;
        }
        return false;
    }

    private float InvokeLinearDepth(Vector3 worldPos)
    {
        if (_miWorldPosToLinearDepth02 == null)
            return float.PositiveInfinity;
        object result = _miWorldPosToLinearDepth02.Invoke(GetActiveCustomDepthTarget(), new object[] { worldPos });
        return result is float f ? f : float.PositiveInfinity;
    }

    private bool TryGetNeighborWorldPos(Vector2Int center, out Vector3 worldPos)
    {
        int ts = 128;
        int rMax = Mathf.Max(1, neighborRadius);
        for (int r = 1; r <= rMax; r++)
        {
            for (int dy = -r; dy <= r; dy++)
            {
                for (int dx = -r; dx <= r; dx++)
                {
                    if (dx == 0 && dy == 0)
                        continue;

                    int xx = center.x + dx;
                    int yy = center.y + dy;
                    if ((uint)xx >= (uint)ts || (uint)yy >= (uint)ts)
                        continue;

                    if (TryGetDepthWorldPos(new Vector2Int(xx, yy), out worldPos))
                        return true;
                }
            }
        }

        worldPos = default;
        return false;
    }

    private void RebuildDepthGrid()
    {
        _gridPx.Clear();
        int ts = 128;
        int step = Mathf.Max(1, depthStride);
        int yFlipBase = ts - 1;

        for (int y = 0; y < ts; y += step)
        {
            int py = depthPixelVFlip ? (yFlipBase - y) : y;
            for (int x = 0; x < ts; x += step)
                _gridPx.Add(new Vector2Int(x, py));
        }

        _gridCursor = 0;
    }

    private Vector2 NextHalton2D()
    {
        float u = Halton(_haltonIndex, 2);
        float v = Halton(_haltonIndex, 3);
        _haltonIndex++;
        if (_haltonIndex > 1_000_000)
            _haltonIndex = 1;

        float margin = Mathf.Clamp(viewportMargin, 0f, 0.2f);
        u = Mathf.Lerp(margin, 1f - margin, u);
        v = Mathf.Lerp(margin, 1f - margin, v);
        return new Vector2(u, v);
    }

    private static float Halton(int index, int b)
    {
        float f = 1f;
        float r = 0f;
        int i = index;
        while (i > 0)
        {
            f /= b;
            r += f * (i % b);
            i /= b;
        }
        return r;
    }

    private bool IsSelfExcluded(Vector3 worldPos)
    {
        float radius = selfExcludeRadiusMeters;
        if (radius <= 0f)
            return false;

        float radiusSq = radius * radius;
        if (selfExcludeTransforms != null && selfExcludeTransforms.Length > 0)
        {
            for (int i = 0; i < selfExcludeTransforms.Length; i++)
            {
                Transform tr = selfExcludeTransforms[i];
                if (!tr)
                    continue;
                if ((worldPos - tr.position).sqrMagnitude <= radiusSq)
                    return true;
            }
            return false;
        }

        return sampleCamera && (worldPos - sampleCamera.transform.position).sqrMagnitude <= radiusSq;
    }

    private bool IsInsideSamplingBounds(Vector3 worldPos)
    {
        Camera cam = sampleCamera ? sampleCamera : Camera.main;
        if (!cam)
            return true;

        Vector3 vp = cam.WorldToViewportPoint(worldPos);
        if (vp.z <= 0f)
            return false;

        Vector2 center = new Vector2(
            Mathf.Clamp01(samplingViewportCenter.x),
            Mathf.Clamp01(samplingViewportCenter.y));
        float maxSize = Mathf.Clamp(samplingViewportMaxSize, 0.2f, 1f);
        Vector2 size = new Vector2(
            Mathf.Clamp(samplingViewportSize.x, 0.01f, maxSize),
            Mathf.Clamp(samplingViewportSize.y, 0.01f, maxSize));
        Vector2 half = size * 0.5f;
        return Mathf.Abs(vp.x - center.x) <= half.x && Mathf.Abs(vp.y - center.y) <= half.y;
    }

    private static bool IsFinite(Vector3 p)
    {
        return !float.IsNaN(p.x) && !float.IsNaN(p.y) && !float.IsNaN(p.z) &&
               !float.IsInfinity(p.x) && !float.IsInfinity(p.y) && !float.IsInfinity(p.z);
    }

    private MonoBehaviour GetActiveCustomDepthTarget()
    {
        if (scanCoverDepthRaycaster)
            return scanCoverDepthRaycaster;
        return customDepthRaycaster;
    }
}
