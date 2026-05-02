using System.Collections.Generic;
using UnityEngine;

namespace MRMotifs.InstantContentPlacement.DepthEffects
{
    [DisallowMultipleComponent]
    public sealed class ScanCoverInstancedPointCloudRenderer : MonoBehaviour
    {
        public enum PointOrientationMode
        {
            AxisAligned = 0,
            SurfaceAligned = 1,
            CameraFacing = 2
        }

        [Header("Refs")]
        public ScanCoverVoxelPointAccumulator voxelPointAccumulator;
        public Camera sampleCamera;

        [Header("Display")]
        public bool showPointCloud = true;
        public PrimitiveType pointPrimitive = PrimitiveType.Cube;
        public bool billboardPointQuads;
        public bool scalePointByConfidence = true;
        public bool regularizeDisplayPoints = true;
        public float displayRegularizationSpacingMeters = 0.05f;
        public bool clampDisplayPointCount = true;
        public int maxDisplayPoints = 2048;
        public float minDisplayDistanceMeters = 0.0f;
        public float maxDisplayDistanceMeters = 4.5f;
        public PointOrientationMode pointOrientationMode = PointOrientationMode.AxisAligned;
        public float pointScaleMeters = 0.0065f;
        public float minPointScaleMeters = 0.0045f;
        public float maxPointScaleMeters = 0.0085f;
        public float surfaceOffsetMeters = 0.008f;
        public float distanceSurfaceOffsetFactor = 0.0f;
        public float maxSurfaceOffsetMeters = 0.03f;
        public float displayRefreshIntervalSeconds = 0.06f;
        public Color pointColor = new Color(0.92f, 0.96f, 0.99f, 1f);
        public Material pointMaterialOverride;
        public bool debugLog;

        public int LastSourceStablePointCount { get; private set; }
        public int LastDisplayPointCount { get; private set; }
        public int LastDrawBatchCount { get; private set; }

        private readonly List<ScanCoverVoxelPointAccumulator.StablePoint> _stablePoints = new(4096);
        private readonly List<ScanCoverVoxelPointAccumulator.StablePoint> _displayPoints = new(4096);
        private readonly List<GameObject> _pointObjects = new(512);
        private readonly Dictionary<Vector3Int, int> _bucketToIndex = new(4096);

        private Transform _root;
        private Material _runtimeMaterial;
        private float _nextRefreshTime;
        private bool _coalesceRevisionChanges;
        private int _lastRevision = -1;
        private int _baseMaxDisplayPoints;
        private float _baseDisplayRefreshIntervalSeconds;

        private void Awake()
        {
            _baseMaxDisplayPoints = Mathf.Max(1, maxDisplayPoints);
            _baseDisplayRefreshIntervalSeconds = Mathf.Max(0.01f, displayRefreshIntervalSeconds);
            EnsureRefs();
            EnsureRoot();
            EnsureMaterial();
        }

        private void OnEnable()
        {
            _nextRefreshTime = 0f;
            EnsureRefs();
            EnsureRoot();
            EnsureMaterial();
        }

        private void OnDisable()
        {
            SetVisibleCount(0);
        }

        private void OnDestroy()
        {
            for (int i = 0; i < _pointObjects.Count; i++)
            {
                if (_pointObjects[i] != null)
                    Destroy(_pointObjects[i]);
            }

            if (_root != null)
                Destroy(_root.gameObject);

            if (_runtimeMaterial != null)
                Destroy(_runtimeMaterial);
        }

        private void LateUpdate()
        {
            EnsureRefs();
            EnsureRoot();
            EnsureMaterial();

            if (!showPointCloud || voxelPointAccumulator == null)
            {
                LastSourceStablePointCount = 0;
                LastDisplayPointCount = 0;
                LastDrawBatchCount = 0;
                SetVisibleCount(0);
                return;
            }

            int revision = voxelPointAccumulator.Revision;
            bool revisionChanged = revision != _lastRevision;
            if (_coalesceRevisionChanges && !revisionChanged && Time.unscaledTime < _nextRefreshTime)
                return;

            if (!revisionChanged && Time.unscaledTime < _nextRefreshTime)
                return;

            _lastRevision = revision;
            _nextRefreshTime = Time.unscaledTime + Mathf.Max(0.01f, displayRefreshIntervalSeconds);

            voxelPointAccumulator.GetStablePointsNonAlloc(_stablePoints);
            LastSourceStablePointCount = _stablePoints.Count;
            BuildDisplayPoints();
            RenderDisplayPoints();
        }

        public void ApplyPerformanceSlice(float displayPointScale, float displayRefreshScale, bool coalesceRevisionChanges)
        {
            _coalesceRevisionChanges = coalesceRevisionChanges;
            maxDisplayPoints = Mathf.Max(1, Mathf.RoundToInt(_baseMaxDisplayPoints * Mathf.Max(0.05f, displayPointScale)));
            displayRefreshIntervalSeconds = Mathf.Max(0.01f, _baseDisplayRefreshIntervalSeconds * Mathf.Max(0.05f, displayRefreshScale));
        }

        private void BuildDisplayPoints()
        {
            _displayPoints.Clear();
            _bucketToIndex.Clear();

            Camera cam = sampleCamera != null ? sampleCamera : Camera.main;
            Vector3 cameraPos = cam != null ? cam.transform.position : Vector3.zero;
            float minDistanceSq = Mathf.Max(0f, minDisplayDistanceMeters);
            minDistanceSq *= minDistanceSq;
            float maxDistanceSq = Mathf.Max(minDisplayDistanceMeters + 0.01f, maxDisplayDistanceMeters);
            maxDistanceSq *= maxDistanceSq;
            float spacing = Mathf.Max(0.001f, displayRegularizationSpacingMeters);

            for (int i = 0; i < _stablePoints.Count; i++)
            {
                ScanCoverVoxelPointAccumulator.StablePoint point = _stablePoints[i];
                if (cam != null)
                {
                    float distSq = (point.worldPos - cameraPos).sqrMagnitude;
                    if (distSq < minDistanceSq || distSq > maxDistanceSq)
                        continue;
                }

                if (!regularizeDisplayPoints)
                {
                    _displayPoints.Add(point);
                    continue;
                }

                Vector3Int bucket = new Vector3Int(
                    Mathf.RoundToInt(point.worldPos.x / spacing),
                    Mathf.RoundToInt(point.worldPos.y / spacing),
                    Mathf.RoundToInt(point.worldPos.z / spacing));

                if (_bucketToIndex.TryGetValue(bucket, out int existingIndex))
                {
                    if (point.confidence > _displayPoints[existingIndex].confidence)
                        _displayPoints[existingIndex] = point;
                    continue;
                }

                _bucketToIndex.Add(bucket, _displayPoints.Count);
                _displayPoints.Add(point);
            }

            if (clampDisplayPointCount && _displayPoints.Count > Mathf.Max(1, maxDisplayPoints))
            {
                _displayPoints.Sort((a, b) => b.confidence.CompareTo(a.confidence));
                _displayPoints.RemoveRange(maxDisplayPoints, _displayPoints.Count - maxDisplayPoints);
            }

            LastDisplayPointCount = _displayPoints.Count;
            LastDrawBatchCount = _displayPoints.Count > 0 ? 1 : 0;
        }

        private void RenderDisplayPoints()
        {
            EnsurePointObjects(_displayPoints.Count);

            Camera cam = sampleCamera != null ? sampleCamera : Camera.main;
            for (int i = 0; i < _displayPoints.Count; i++)
            {
                ScanCoverVoxelPointAccumulator.StablePoint point = _displayPoints[i];
                GameObject pointObject = _pointObjects[i];

                Vector3 worldPos = point.worldPos + point.normal * ResolveSurfaceOffset(point, cam);
                pointObject.transform.position = worldPos;
                pointObject.transform.rotation = ResolveRotation(point, cam);
                pointObject.transform.localScale = Vector3.one * ResolveScale(point);
                if (!pointObject.activeSelf)
                    pointObject.SetActive(true);
            }

            SetVisibleCount(_displayPoints.Count);

            if (debugLog && Random.value < 0.01f)
            {
                Debug.Log($"[ScanCoverInstancedPointCloudRenderer] stable={LastSourceStablePointCount}, display={LastDisplayPointCount}, batches={LastDrawBatchCount}");
            }
        }

        private float ResolveScale(ScanCoverVoxelPointAccumulator.StablePoint point)
        {
            float scale = Mathf.Max(0.001f, pointScaleMeters);
            if (scalePointByConfidence)
            {
                float t = Mathf.Clamp01(point.confidence);
                float minScale = Mathf.Max(0.001f, minPointScaleMeters);
                float maxScale = Mathf.Max(minScale, maxPointScaleMeters);
                scale = Mathf.Lerp(minScale, maxScale, t);
            }

            return scale;
        }

        private float ResolveSurfaceOffset(ScanCoverVoxelPointAccumulator.StablePoint point, Camera cam)
        {
            float offset = Mathf.Max(0f, surfaceOffsetMeters);
            if (cam != null && distanceSurfaceOffsetFactor > 0f)
            {
                float distance = Vector3.Distance(cam.transform.position, point.worldPos);
                offset += distance * distanceSurfaceOffsetFactor;
            }

            return Mathf.Min(Mathf.Max(0f, maxSurfaceOffsetMeters), offset);
        }

        private Quaternion ResolveRotation(ScanCoverVoxelPointAccumulator.StablePoint point, Camera cam)
        {
            switch (pointOrientationMode)
            {
                case PointOrientationMode.SurfaceAligned:
                {
                    Vector3 up = point.normal.sqrMagnitude > 1e-6f ? point.normal.normalized : Vector3.up;
                    Vector3 forward = cam != null ? cam.transform.forward : Vector3.forward;
                    Vector3 tangent = Vector3.Cross(up, forward);
                    if (tangent.sqrMagnitude <= 1e-6f)
                        tangent = Vector3.Cross(up, Vector3.right);
                    tangent.Normalize();
                    Vector3 bitangent = Vector3.Cross(tangent, up).normalized;
                    return Quaternion.LookRotation(bitangent, up);
                }
                case PointOrientationMode.CameraFacing:
                    return cam != null ? cam.transform.rotation : Quaternion.identity;
                default:
                    return Quaternion.identity;
            }
        }

        private void EnsureRefs()
        {
            if (voxelPointAccumulator == null)
                voxelPointAccumulator = GetComponent<ScanCoverVoxelPointAccumulator>();
            if (sampleCamera == null)
                sampleCamera = Camera.main;
        }

        private void EnsureRoot()
        {
            if (_root != null)
                return;

            GameObject root = new GameObject("ScanCoverInstancedPointCloudRendererRoot");
            root.transform.SetParent(transform, false);
            _root = root.transform;
        }

        private void EnsureMaterial()
        {
            Material source = pointMaterialOverride;
            if (source == null)
                source = Resources.Load<Material>("DepthEffectsPointCloudUnlit");
            if (source == null)
                return;

            if (_runtimeMaterial != null && _runtimeMaterial.shader == source.shader)
            {
                ApplyMaterialColor(_runtimeMaterial);
                return;
            }

            if (_runtimeMaterial != null)
                Destroy(_runtimeMaterial);

            _runtimeMaterial = new Material(source)
            {
                name = $"{source.name}_ScanCoverCompat"
            };
            ApplyMaterialColor(_runtimeMaterial);
        }

        private void ApplyMaterialColor(Material material)
        {
            if (material == null)
                return;
            if (material.HasProperty("_BaseColor"))
                material.SetColor("_BaseColor", pointColor);
            if (material.HasProperty("_Color"))
                material.SetColor("_Color", pointColor);
        }

        private void EnsurePointObjects(int count)
        {
            while (_pointObjects.Count < count)
            {
                GameObject primitive = GameObject.CreatePrimitive(pointPrimitive);
                primitive.name = $"CompatPoint_{_pointObjects.Count:D4}";
                primitive.transform.SetParent(_root, false);
                Collider collider = primitive.GetComponent<Collider>();
                if (collider != null)
                    Destroy(collider);

                MeshRenderer renderer = primitive.GetComponent<MeshRenderer>();
                if (renderer != null && _runtimeMaterial != null)
                {
                    renderer.sharedMaterial = _runtimeMaterial;
                    renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                    renderer.receiveShadows = false;
                }

                primitive.SetActive(false);
                _pointObjects.Add(primitive);
            }
        }

        private void SetVisibleCount(int visibleCount)
        {
            for (int i = visibleCount; i < _pointObjects.Count; i++)
            {
                if (_pointObjects[i] != null && _pointObjects[i].activeSelf)
                    _pointObjects[i].SetActive(false);
            }
        }
    }
}
