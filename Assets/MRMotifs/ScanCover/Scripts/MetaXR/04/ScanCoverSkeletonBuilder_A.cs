using System;
using System.Collections.Generic;
using Meta.XR;
using UnityEngine;

[DisallowMultipleComponent]
public class ScanCoverSkeletonBuilder_A : MonoBehaviour
{
    [Serializable]
    public struct VoxelKey : IEquatable<VoxelKey>
    {
        public int x, y, z;
        public VoxelKey(int x, int y, int z) { this.x = x; this.y = y; this.z = z; }
        public bool Equals(VoxelKey other) => x == other.x && y == other.y && z == other.z;
        public override bool Equals(object obj) => obj is VoxelKey other && Equals(other);
        public override int GetHashCode()
        {
            unchecked
            {
                int h = 17;
                h = h * 31 + x;
                h = h * 31 + y;
                h = h * 31 + z;
                return h;
            }
        }
        public override string ToString() => $"({x},{y},{z})";
    }

    [Serializable]
    public struct CellInfo
    {
        public VoxelKey key;
        public Vector3 worldPos;
        public Vector3 worldNormal;
        public int count;
        public float firstSeenTime;
        public float lastSeenTime;
    }

    [Serializable]
    public struct SummaryStats
    {
        public int totalCells;
        public int confirmedCells;
        public int recentTouchedCells;
        public int recentNewCells;
        public int recentConfirmedCells;
        public float recentWindow;
        public bool ready;
    }

    private class Cell
    {
        public int count;
        public Vector3 meanPosRef;
        public Vector3 meanNormalRef;
        public float firstSeen;
        public float lastSeen;
    }

    public enum GateMode
    {
        None = 0,
        LatestWave = 1,
        AnyWave = 2,
    }

    [Header("Refs")]
    public Transform referenceFrame;
    public EnvironmentRaycastManager environmentRaycast;
    public Camera sampleCamera;

    [Header("Session")]
    public bool scanEnabled = true;
    [Tooltip("扫描阶段是否允许 prune。冻结式方案建议关闭。")]
    public bool allowPruneWhileScanning = false;

    [Header("Voxel")]
    public bool useInternalRaycastSampling = true;
    [Min(0.02f)] public float cellSizeMeters = 0.08f;
    [Min(1)] public int samplesPerFrame = 64;
    [Min(0.2f)] public float maxRayDistance = 6f;
    public bool acceptHitPointOccluded = true;

    [Header("Hit Filters")]
    [Min(0f)] public float minHitDistanceMeters = 0.35f;
    public bool enableSelfExclusion = true;
    public Transform[] selfExcludeTransforms;
    [Min(0f)] public float selfExcludeRadiusMeters = 0.25f;
    [Range(0f, 0.999f)] public float normalDotMin = 0.5f;
    [Min(0f)] public float maxPosDeviationMeters = 0.10f;
    [Tooltip("有 wave 时提升采样率。<=0 表示不提升。")]
    public int samplesPerFrameWhenWaveActive = 96;

    [Header("Reveal Gate")]
    public bool gateByRevealWaves = true;
    public GateMode gateMode = GateMode.AnyWave;
    [Min(0.02f)] public float revealGlobalsRefreshInterval = 0.10f;
    [Min(0f)] public float gateExtraMeters = 0.05f;

    [Header("Sampling Bounds")]
    [Tooltip("开启后，仅采样框区内命中点")]
    public bool enableSamplingBoundsFilter = false;
    [Tooltip("框区中心（参考系局部坐标）")]
    public Vector2 samplingViewportCenter = new Vector2(0.5f, 0.5f);
    [Tooltip("框区尺寸（米，参考系局部坐标）")]
    public Vector2 samplingViewportSize = new Vector2(0.40f, 0.50f);
    [Range(0.2f, 1f)] public float samplingViewportMaxSize = 0.90f;

    [Header("Readiness")]
    [Min(1)] public int readinessMinHits = 2;
    [Min(0.1f)] public float readinessRecentWindow = 2.0f;
    [Min(1)] public int readinessMinConfirmedCells = 150;
    [Range(0f, 1f)] public float readinessMaxRecentGrowthRatio = 0.04f;

    [Header("Prune")]
    [Min(0f)] public float staleSeconds = 60f;
    [Min(0.1f)] public float pruneIntervalSeconds = 1.0f;
    [Min(1000)] public int softMaxCells = 50000;

    [Header("Debug")]
    public bool debugLog = false;

    private readonly Dictionary<VoxelKey, Cell> _cells = new Dictionary<VoxelKey, Cell>(4096);
    private readonly List<VoxelKey> _toRemove = new List<VoxelKey>(2048);
    private readonly List<CellInfo> _scratchSnapshot = new List<CellInfo>(8192);
    private int _sampleIndex = 1;

    private float _sessionStartTime;
    private float _sessionFreezeTime = -1f;

    private float _nextRevealPullTime;
    private Vector4[] _revealWaves;
    private int _revealCount;
    private float _revealFeather = 0.25f;
    private Vector3 _latestWaveCenterWS;
    private float _latestWaveRadius;
    private int _latestWaveCount;

    private float _nextPruneTime;
    private float _nextStatsLogTime;

    private static readonly int RevealWavesId = Shader.PropertyToID("_RevealWaves");
    private static readonly int RevealWaveCountId = Shader.PropertyToID("_RevealWaveCount");
    private static readonly int RevealWaveParamsId = Shader.PropertyToID("_RevealWaveParams");

    public int CellCount => _cells.Count;
    public bool IsFrozen => !scanEnabled;
    public float SessionStartTime => _sessionStartTime;
    public float SessionFreezeTime => _sessionFreezeTime;

    private void Reset()
    {
        referenceFrame = transform;
    }

    private void Awake()
    {
        if (!referenceFrame) referenceFrame = transform;
        if (!sampleCamera) sampleCamera = Camera.main;
        if (!environmentRaycast) environmentRaycast = FindObjectOfType<EnvironmentRaycastManager>();
        PullRevealGlobals(true);
        if (_sessionStartTime <= 0f) _sessionStartTime = Time.time;
    }

    private void OnEnable()
    {
        _nextRevealPullTime = 0f;
        _nextPruneTime = 0f;
        _nextStatsLogTime = 0f;
        if (_sessionStartTime <= 0f) _sessionStartTime = Time.time;
    }

    private void Update()
    {
        float t = Time.time;

        if (gateByRevealWaves && t >= _nextRevealPullTime)
        {
            PullRevealGlobals(false);
            _nextRevealPullTime = t + Mathf.Max(0.02f, revealGlobalsRefreshInterval);
        }

        if (!scanEnabled)
            return;

        if (useInternalRaycastSampling)
        {
            if (!environmentRaycast || !EnvironmentRaycastManager.IsSupported)
                return;

            if (!sampleCamera)
            {
                sampleCamera = Camera.main;
                if (!sampleCamera) return;
            }

            bool waveActive = gateByRevealWaves && gateMode != GateMode.None && (_revealCount > 0 && _revealWaves != null);
            int spf = Mathf.Max(1, samplesPerFrame);
            if (waveActive && samplesPerFrameWhenWaveActive > 0)
                spf = Mathf.Max(spf, samplesPerFrameWhenWaveActive);

            for (int i = 0; i < spf; i++)
            {
                Vector2 uv = NextHalton2D();
                Ray ray = sampleCamera.ViewportPointToRay(new Vector3(uv.x, uv.y, 0f));

                bool hitSuccess = environmentRaycast.Raycast(ray, out var hit, maxDistance: maxRayDistance);
                bool usable = hitSuccess || (acceptHitPointOccluded && hit.status == EnvironmentRaycastHitStatus.HitPointOccluded);
                if (!usable) continue;

                Vector3 pW = hit.point;
                Vector3 nW = (hit.normal.sqrMagnitude > 1e-6f) ? hit.normal.normalized : (-ray.direction).normalized;

                float hitDist = Vector3.Distance(ray.origin, pW);
                if (minHitDistanceMeters > 0f && hitDist < minHitDistanceMeters)
                    continue;
                if (enableSelfExclusion && selfExcludeRadiusMeters > 0f && IsSelfExcluded(pW))
                    continue;
                if (enableSamplingBoundsFilter && !IsInsideSamplingBounds(pW))
                    continue;
                if (gateByRevealWaves && gateMode != GateMode.None && !PassRevealGate(pW))
                    continue;

                AddSample(pW, nW, t);
            }
        }

        if (allowPruneWhileScanning && staleSeconds > 0f && t >= _nextPruneTime)
        {
            PruneStale(t);
            _nextPruneTime = t + Mathf.Max(0.1f, pruneIntervalSeconds);
        }

        if (softMaxCells > 0 && _cells.Count > softMaxCells)
            PruneAggressive(t);

        if (debugLog && t >= _nextStatsLogTime)
        {
            SummaryStats stats = GetSummaryStats();
            Debug.Log(
                $"[ScanCoverSkeletonBuilder_A] cellCount={stats.totalCells}, confirmedCells={stats.confirmedCells}, " +
                $"recentTouched={stats.recentTouchedCells}, recentNew={stats.recentNewCells}, ready={stats.ready}");
            _nextStatsLogTime = t + 1.0f;
        }
    }

    public void BeginNewSession(bool clearExisting = true)
    {
        if (clearExisting)
            ClearAll();
        scanEnabled = true;
        _sessionStartTime = Time.time;
        _sessionFreezeTime = -1f;
        if (debugLog) Debug.Log("[ScanCoverSkeletonBuilder_A] BeginNewSession");
    }

    public void FreezeSession()
    {
        scanEnabled = false;
        _sessionFreezeTime = Time.time;
        if (debugLog) Debug.Log("[ScanCoverSkeletonBuilder_A] FreezeSession");
    }

    public void ResumeSession()
    {
        scanEnabled = true;
        if (debugLog) Debug.Log("[ScanCoverSkeletonBuilder_A] ResumeSession");
    }

    public void ClearAll()
    {
        _cells.Clear();
        _sampleIndex = 1;
        _sessionStartTime = Time.time;
        _sessionFreezeTime = -1f;
    }

    public void GetCellsSnapshot(List<CellInfo> outList)
    {
        if (outList == null) return;
        outList.Clear();
        outList.Capacity = Mathf.Max(outList.Capacity, _cells.Count);
        foreach (var kv in _cells)
        {
            VoxelKey key = kv.Key;
            Cell c = kv.Value;
            outList.Add(new CellInfo
            {
                key = key,
                worldPos = referenceFrame.TransformPoint(c.meanPosRef),
                worldNormal = referenceFrame.TransformDirection(c.meanNormalRef).normalized,
                count = c.count,
                firstSeenTime = c.firstSeen,
                lastSeenTime = c.lastSeen,
            });
        }
    }

    public SummaryStats GetSummaryStats()
    {
        SummaryStats s = new SummaryStats();
        s.recentWindow = Mathf.Max(0.1f, readinessRecentWindow);
        float t = Time.time;
        float newCut = t - s.recentWindow;

        foreach (var kv in _cells)
        {
            Cell c = kv.Value;
            s.totalCells++;
            if (c.count >= readinessMinHits) s.confirmedCells++;
            if (c.lastSeen >= newCut) s.recentTouchedCells++;
            if (c.firstSeen >= newCut) s.recentNewCells++;
            if (c.count >= readinessMinHits && c.firstSeen >= newCut) s.recentConfirmedCells++;
        }

        float growthRatio = (s.confirmedCells > 0) ? ((float)s.recentConfirmedCells / s.confirmedCells) : 1f;
        s.ready = s.confirmedCells >= readinessMinConfirmedCells && growthRatio <= readinessMaxRecentGrowthRatio;
        return s;
    }

    public bool TryGetLatestWave(out Vector3 centerWS, out float radius, out float feather, out int count)
    {
        centerWS = _latestWaveCenterWS;
        radius = _latestWaveRadius;
        feather = Mathf.Max(0.0001f, _revealFeather);
        count = _latestWaveCount;
        return count > 0;
    }

    public bool TryGetRevealWavesUnsafe(out Vector4[] waves, out int count, out float feather)
    {
        waves = _revealWaves;
        count = _revealCount;
        feather = Mathf.Max(0.0001f, _revealFeather);
        return waves != null && count > 0;
    }

    public bool TryAddExternalObservation(Vector3 worldPos, Vector3 worldNormal, float timeNow = -1f)
    {
        if (!scanEnabled)
            return false;

        if (!sampleCamera)
            sampleCamera = Camera.main;

        if (minHitDistanceMeters > 0f && sampleCamera)
        {
            float hitDist = Vector3.Distance(sampleCamera.transform.position, worldPos);
            if (hitDist < minHitDistanceMeters)
                return false;
        }

        if (enableSelfExclusion && selfExcludeRadiusMeters > 0f && IsSelfExcluded(worldPos))
            return false;
        if (enableSamplingBoundsFilter && !IsInsideSamplingBounds(worldPos))
            return false;
        if (gateByRevealWaves && gateMode != GateMode.None && !PassRevealGate(worldPos))
            return false;

        Vector3 normal = worldNormal.sqrMagnitude > 1e-6f ? worldNormal.normalized : Vector3.up;
        AddSample(worldPos, normal, timeNow >= 0f ? timeNow : Time.time);
        return true;
    }

    private void AddSample(Vector3 worldPos, Vector3 worldNormal, float timeNow)
    {
        Vector3 pR = referenceFrame.InverseTransformPoint(worldPos);
        Vector3 nR = referenceFrame.InverseTransformDirection(worldNormal).normalized;
        VoxelKey key = RefToKey(pR);

        if (!_cells.TryGetValue(key, out Cell cell))
        {
            cell = new Cell
            {
                count = 1,
                meanPosRef = pR,
                meanNormalRef = nR,
                firstSeen = timeNow,
                lastSeen = timeNow,
            };
            _cells.Add(key, cell);
            return;
        }

        if (normalDotMin > 0f)
        {
            float d = Vector3.Dot(cell.meanNormalRef.normalized, nR);
            if (d < normalDotMin)
            {
                cell.lastSeen = timeNow;
                return;
            }
        }

        if (maxPosDeviationMeters > 0f)
        {
            float dp = (pR - cell.meanPosRef).magnitude;
            if (dp > maxPosDeviationMeters)
            {
                cell.lastSeen = timeNow;
                return;
            }
        }

        cell.count++;
        float inv = 1.0f / Mathf.Max(1, cell.count);
        cell.meanPosRef += (pR - cell.meanPosRef) * inv;
        cell.meanNormalRef += (nR - cell.meanNormalRef) * inv;
        if (cell.meanNormalRef.sqrMagnitude > 1e-6f)
            cell.meanNormalRef.Normalize();
        cell.lastSeen = timeNow;
    }

    private VoxelKey RefToKey(Vector3 refLocalPos)
    {
        float s = Mathf.Max(1e-4f, cellSizeMeters);
        return new VoxelKey(
            Mathf.FloorToInt(refLocalPos.x / s),
            Mathf.FloorToInt(refLocalPos.y / s),
            Mathf.FloorToInt(refLocalPos.z / s));
    }

    private Vector2 NextHalton2D()
    {
        float u = Halton(_sampleIndex, 2);
        float v = Halton(_sampleIndex, 3);
        _sampleIndex++;
        if (_sampleIndex > 1_000_000) _sampleIndex = 1;
        const float margin = 0.03f;
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
        float r = selfExcludeRadiusMeters;
        if (r <= 0f) return false;
        float r2 = r * r;
        if (selfExcludeTransforms != null && selfExcludeTransforms.Length > 0)
        {
            for (int i = 0; i < selfExcludeTransforms.Length; i++)
            {
                Transform tr = selfExcludeTransforms[i];
                if (!tr) continue;
                if ((worldPos - tr.position).sqrMagnitude <= r2)
                    return true;
            }
            return false;
        }
        return sampleCamera && (worldPos - sampleCamera.transform.position).sqrMagnitude <= r2;
    }

    private void PullRevealGlobals(bool force)
    {
        if (!gateByRevealWaves && !force) return;
        try
        {
            float cntF = Shader.GetGlobalFloat(RevealWaveCountId);
            int cnt = Mathf.Clamp(Mathf.RoundToInt(cntF), 0, 64);
            Vector4 param = Shader.GetGlobalVector(RevealWaveParamsId);
            float feather = Mathf.Max(0.0001f, param.x);
            Vector4[] arr = Shader.GetGlobalVectorArray(RevealWavesId);
            _revealWaves = arr;
            _revealCount = Mathf.Min(cnt, arr != null ? arr.Length : 0);
            _revealFeather = feather;
            _latestWaveCount = _revealCount;
            if (_revealWaves != null && _revealCount > 0)
            {
                Vector4 w = _revealWaves[_revealCount - 1];
                _latestWaveCenterWS = new Vector3(w.x, w.y, w.z);
                _latestWaveRadius = Mathf.Max(0f, w.w);
            }
            else
            {
                _latestWaveCenterWS = Vector3.zero;
                _latestWaveRadius = 0f;
            }
        }
        catch (Exception e)
        {
            if (debugLog)
                Debug.LogWarning($"[ScanCoverSkeletonBuilder_A] PullRevealGlobals failed: {e.Message}");
            _revealWaves = null;
            _revealCount = 0;
            _revealFeather = 0.25f;
            _latestWaveCount = 0;
            _latestWaveCenterWS = Vector3.zero;
            _latestWaveRadius = 0f;
        }
    }

    private bool PassRevealGate(Vector3 worldPos)
    {
        if (_revealWaves == null || _revealCount <= 0)
            return false;

        float extra = Mathf.Max(0f, gateExtraMeters);
        float feather = Mathf.Max(0.0001f, _revealFeather);

        if (gateMode == GateMode.LatestWave)
        {
            int idx = _revealCount - 1;
            Vector4 w = _revealWaves[idx];
            Vector3 c = new Vector3(w.x, w.y, w.z);
            float r = Mathf.Max(0f, w.w);
            float allow = r + feather + extra;
            return (worldPos - c).sqrMagnitude <= allow * allow;
        }

        if (gateMode == GateMode.AnyWave)
        {
            for (int i = 0; i < _revealCount; i++)
            {
                Vector4 w = _revealWaves[i];
                Vector3 c = new Vector3(w.x, w.y, w.z);
                float r = Mathf.Max(0f, w.w);
                float allow = r + feather + extra;
                if ((worldPos - c).sqrMagnitude <= allow * allow)
                    return true;
            }
            return false;
        }

        return true;
    }

    private void PruneStale(float timeNow)
    {
        float stale = Mathf.Max(0f, staleSeconds);
        if (stale <= 0f) return;
        float threshold = timeNow - stale;
        _toRemove.Clear();
        foreach (var kv in _cells)
        {
            if (kv.Value.lastSeen < threshold)
                _toRemove.Add(kv.Key);
        }
        for (int i = 0; i < _toRemove.Count; i++)
            _cells.Remove(_toRemove[i]);
    }

    private void PruneAggressive(float timeNow)
    {
        if (_cells.Count <= softMaxCells) return;
        GetCellsSnapshot(_scratchSnapshot);
        _scratchSnapshot.Sort((a, b) => a.lastSeenTime.CompareTo(b.lastSeenTime));
        int removeCount = Mathf.Max(1, _cells.Count - softMaxCells);
        for (int i = 0; i < removeCount && i < _scratchSnapshot.Count; i++)
            _cells.Remove(_scratchSnapshot[i].key);
    }

    private bool IsInsideSamplingBounds(Vector3 worldPos)
    {
        Camera cam = sampleCamera ? sampleCamera : Camera.main;
        if (!cam) return true;

        Vector3 vp = cam.WorldToViewportPoint(worldPos);
        if (vp.z <= 0f) return false;

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
}
