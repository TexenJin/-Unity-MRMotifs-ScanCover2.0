using System.Collections.Generic;
using MyProject.XR;
using UnityEngine;

/// <summary>
/// Samples a regular grid on the 128x128 environment depth texture, reconstructs
/// world-space points, and places marker prefabs on the visible surface.
/// Supports a two-layer interpretation for Both-eye mode:
/// StereoConfirmed and MonoSupported.
/// </summary>
public class DepthGridPointCloud : MonoBehaviour
{
    public enum DualEyeMergeMode
    {
        Disabled = 0,
        AverageWhenCloseElseCloser = 1,
        RequireAgreement = 2,
        PreferCloserEye = 3,
    }

    [Header("Dependencies")]
    [SerializeField] private CustomEnvironmentDepthRaycaster depthRaycaster;
    [SerializeField] private GameObject markerPrefab;
    [SerializeField] private Transform markerParent;

    [Header("Eye")]
    [SerializeField] private Eye eye = Eye.Right;
    [SerializeField] private DualEyeMergeMode dualEyeMergeMode = DualEyeMergeMode.AverageWhenCloseElseCloser;
    [SerializeField] private float dualEyeAgreementMeters = 0.05f;
    [SerializeField, Range(0f, 1f)] private float dualEyeNormalAgreementDot = 0.75f;
    [SerializeField] private bool allowMonoFallbackWhenBothDisagree = true;
    [SerializeField] private bool allowMonoFallbackWhenOnlyOneEyeValid = true;

    [Header("Grid")]
    [SerializeField] private int stride = 2;
    [SerializeField] private bool depthPixelVFlip = true;

    [Header("Depth Filter")]
    [SerializeField] private float zMin = 0.05f;
    [SerializeField] private float zMax = 8.0f;

    [Header("Hole Fill")]
    [SerializeField] private bool neighborFill = true;
    [SerializeField] private int neighborRadius = 1;

    [Header("Update")]
    [SerializeField] private bool updateEveryFrame = true;
    [SerializeField] private bool orientToNormal = false;
    [SerializeField] private float surfaceBiasMeters = 0.0015f;

    [Header("Layer Debug")]
    [SerializeField] private bool tintBySupportLayer = true;
    [SerializeField] private bool hideMonoSupported = false;
    [SerializeField] private Color stereoConfirmedColor = new Color(0.15f, 0.95f, 0.95f, 1f);
    [SerializeField] private Color monoSupportedColor = new Color(1.0f, 0.55f, 0.15f, 1f);

    private readonly List<Vector2Int> _gridPx = new List<Vector2Int>();
    private readonly List<GameObject> _pool = new List<GameObject>();
    private readonly List<Renderer[]> _rendererCache = new List<Renderer[]>();
    private readonly List<ScanCoverSurfaceObservation> _currentObservations = new List<ScanCoverSurfaceObservation>(4096);
    private MaterialPropertyBlock _propertyBlock;
    private int _textureSize;
    private int _stereoConfirmedCount;
    private int _monoSupportedCount;
    private int _observationFrameIndex;

    public int StereoConfirmedCount => _stereoConfirmedCount;
    public int MonoSupportedCount => _monoSupportedCount;
    public int ObservationFrameIndex => _observationFrameIndex;
    public IReadOnlyList<ScanCoverSurfaceObservation> CurrentObservations => _currentObservations;

    private void Awake()
    {
        if (!depthRaycaster || !markerPrefab)
        {
            Debug.LogError("[DepthGridPointCloud] Missing depthRaycaster or markerPrefab");
            enabled = false;
            return;
        }

        BuildGrid();
        EnsurePoolSize(_gridPx.Count);
        _propertyBlock = new MaterialPropertyBlock();
        RefreshPointCloud();
    }

    private void Update()
    {
        if (updateEveryFrame)
            RefreshPointCloud();
    }

    public void RebuildWithStride(int newStride)
    {
        stride = Mathf.Max(1, newStride);
        BuildGrid();
        EnsurePoolSize(_gridPx.Count);
        RefreshPointCloud();
    }

    private void BuildGrid()
    {
        _gridPx.Clear();
        _textureSize = CustomEnvironmentDepthRaycaster.TextureSize;
        int step = Mathf.Max(1, stride);
        int yFlipBase = _textureSize - 1;

        for (int y = 0; y < _textureSize; y += step)
        {
            int py = depthPixelVFlip ? (yFlipBase - y) : y;
            for (int x = 0; x < _textureSize; x += step)
                _gridPx.Add(new Vector2Int(x, py));
        }
    }

    private void EnsurePoolSize(int count)
    {
        while (_pool.Count < count)
        {
            GameObject instance = Instantiate(markerPrefab, markerParent ? markerParent : transform);
            instance.SetActive(false);
            _pool.Add(instance);
            _rendererCache.Add(instance.GetComponentsInChildren<Renderer>(true));
        }

        for (int i = count; i < _pool.Count; i++)
        {
            if (_pool[i])
                _pool[i].SetActive(false);
        }
    }

    public void RefreshPointCloud()
    {
        _stereoConfirmedCount = 0;
        _monoSupportedCount = 0;
        _observationFrameIndex++;
        _currentObservations.Clear();

        for (int i = 0; i < _gridPx.Count; i++)
        {
            if (!TrySamplePoint(
                    _gridPx[i],
                    out Vector3 worldPos,
                    out Vector3 worldNormal,
                    out float linearDepth,
                    out Eye sourceEye,
                    out ScanCoverSurfaceSupportLayer supportLayer,
                    out float confidence))
            {
                if (_pool[i])
                    _pool[i].SetActive(false);
                continue;
            }

            GameObject go = _pool[i];
            if (!go)
                continue;

            _currentObservations.Add(new ScanCoverSurfaceObservation(
                true,
                worldPos,
                worldNormal,
                linearDepth,
                sourceEye,
                supportLayer,
                confidence,
                _observationFrameIndex,
                _gridPx[i]));

            if (surfaceBiasMeters > 0f && worldNormal.sqrMagnitude > 1e-6f)
                worldPos += worldNormal.normalized * surfaceBiasMeters;

            go.transform.position = worldPos;

            if (orientToNormal && worldNormal.sqrMagnitude > 1e-6f)
                go.transform.rotation = Quaternion.LookRotation(worldNormal.normalized, Vector3.up);

            ApplySupportLayerVisual(i, supportLayer);

            bool shouldShow = !(hideMonoSupported && supportLayer == ScanCoverSurfaceSupportLayer.MonoSupported);
            if (go.activeSelf != shouldShow)
                go.SetActive(shouldShow);
        }

        depthRaycaster.SetEye(eye == Eye.Both ? Eye.Right : eye);
    }

    private bool TrySamplePoint(
        Vector2Int px,
        out Vector3 worldPos,
        out Vector3 worldNormal,
        out float linearDepth,
        out Eye sourceEye,
        out ScanCoverSurfaceSupportLayer supportLayer,
        out float confidence)
    {
        if (eye == Eye.Both && dualEyeMergeMode != DualEyeMergeMode.Disabled)
            return TrySamplePointBothEyes(px, out worldPos, out worldNormal, out linearDepth, out sourceEye, out supportLayer, out confidence);

        bool ok = TrySamplePointForEye(eye, px, out worldPos, out worldNormal, out linearDepth);
        sourceEye = eye;
        confidence = ok ? 0.5f : 0f;
        supportLayer = ok ? ScanCoverSurfaceSupportLayer.MonoSupported : ScanCoverSurfaceSupportLayer.None;
        if (ok)
            _monoSupportedCount++;
        return ok;
    }

    private bool TrySamplePointBothEyes(
        Vector2Int px,
        out Vector3 worldPos,
        out Vector3 worldNormal,
        out float linearDepth,
        out Eye sourceEye,
        out ScanCoverSurfaceSupportLayer supportLayer,
        out float confidence)
    {
        bool hasLeft = TrySamplePointForEye(Eye.Left, px, out Vector3 leftPos, out Vector3 leftNormal, out float leftDepth);
        bool hasRight = TrySamplePointForEye(Eye.Right, px, out Vector3 rightPos, out Vector3 rightNormal, out float rightDepth);

        if (hasLeft && hasRight)
        {
            float positionDelta = Vector3.Distance(leftPos, rightPos);
            float normalDot = (leftNormal.sqrMagnitude > 1e-6f && rightNormal.sqrMagnitude > 1e-6f)
                ? Vector3.Dot(leftNormal.normalized, rightNormal.normalized)
                : 1f;
            bool agrees = positionDelta <= dualEyeAgreementMeters && normalDot >= dualEyeNormalAgreementDot;

            if (agrees)
            {
                if (dualEyeMergeMode == DualEyeMergeMode.PreferCloserEye)
                {
                    worldPos = leftDepth <= rightDepth ? leftPos : rightPos;
                    worldNormal = leftDepth <= rightDepth ? leftNormal : rightNormal;
                }
                else
                {
                    worldPos = (leftPos + rightPos) * 0.5f;
                    worldNormal = MergeNormals(leftNormal, rightNormal);
                }

                linearDepth = Mathf.Min(leftDepth, rightDepth);
                sourceEye = Eye.Both;
                confidence = 1f;
                supportLayer = ScanCoverSurfaceSupportLayer.StereoConfirmed;
                _stereoConfirmedCount++;
                return true;
            }

            if (!allowMonoFallbackWhenBothDisagree || dualEyeMergeMode == DualEyeMergeMode.RequireAgreement)
            {
                worldPos = default;
                worldNormal = default;
                linearDepth = 0f;
                sourceEye = Eye.Both;
                confidence = 0f;
                supportLayer = ScanCoverSurfaceSupportLayer.None;
                return false;
            }

            SelectFallbackByDistance(leftPos, leftNormal, leftDepth, rightPos, rightNormal, rightDepth, out worldPos, out worldNormal, out linearDepth, out sourceEye);

            confidence = 0.5f;
            supportLayer = ScanCoverSurfaceSupportLayer.MonoSupported;
            _monoSupportedCount++;
            return true;
        }

        if (hasLeft && allowMonoFallbackWhenOnlyOneEyeValid)
        {
            worldPos = leftPos;
            worldNormal = leftNormal;
            linearDepth = leftDepth;
            sourceEye = Eye.Left;
            confidence = 0.4f;
            supportLayer = ScanCoverSurfaceSupportLayer.MonoSupported;
            _monoSupportedCount++;
            return true;
        }

        if (hasRight && allowMonoFallbackWhenOnlyOneEyeValid)
        {
            worldPos = rightPos;
            worldNormal = rightNormal;
            linearDepth = rightDepth;
            sourceEye = Eye.Right;
            confidence = 0.4f;
            supportLayer = ScanCoverSurfaceSupportLayer.MonoSupported;
            _monoSupportedCount++;
            return true;
        }

        worldPos = default;
        worldNormal = default;
        linearDepth = 0f;
        sourceEye = Eye.Both;
        confidence = 0f;
        supportLayer = ScanCoverSurfaceSupportLayer.None;
        return false;
    }

    private bool TrySamplePointForEye(Eye sampleEye, Vector2Int px, out Vector3 worldPos, out Vector3 worldNormal, out float camZ)
    {
        depthRaycaster.SetEye(sampleEye);
        worldPos = depthRaycaster.WorldPosAtDepthTexCoord02(px);

        if (!IsFinite(worldPos) && neighborFill)
            TryGetNeighborWorldPosForCurrentEye(px, out worldPos);

        if (!IsFinite(worldPos))
        {
            worldNormal = Vector3.zero;
            camZ = 0f;
            return false;
        }

        camZ = depthRaycaster.WorldPosToLinearDepth02(worldPos);
        if (camZ < zMin || camZ > zMax)
        {
            worldNormal = Vector3.zero;
            return false;
        }

        worldNormal = depthRaycaster.ReconstructNormal02(px);
        if (worldNormal.sqrMagnitude <= 1e-6f)
            worldNormal = Vector3.forward;
        else
            worldNormal.Normalize();

        return true;
    }

    private bool TryGetNeighborWorldPosForCurrentEye(Vector2Int center, out Vector3 worldPos)
    {
        int radiusMax = Mathf.Max(1, neighborRadius);
        for (int radius = 1; radius <= radiusMax; radius++)
        {
            for (int dy = -radius; dy <= radius; dy++)
            {
                for (int dx = -radius; dx <= radius; dx++)
                {
                    if (dx == 0 && dy == 0)
                        continue;

                    int xx = center.x + dx;
                    int yy = center.y + dy;
                    if ((uint)xx >= (uint)_textureSize || (uint)yy >= (uint)_textureSize)
                        continue;

                    Vector3 p = depthRaycaster.WorldPosAtDepthTexCoord02(new Vector2Int(xx, yy));
                    if (IsFinite(p))
                    {
                        worldPos = p;
                        return true;
                    }
                }
            }
        }

        worldPos = default;
        return false;
    }

    private void ApplySupportLayerVisual(int markerIndex, ScanCoverSurfaceSupportLayer supportLayer)
    {
        if (!tintBySupportLayer || markerIndex < 0 || markerIndex >= _rendererCache.Count)
            return;

        if (_propertyBlock == null)
            _propertyBlock = new MaterialPropertyBlock();

        Renderer[] renderers = _rendererCache[markerIndex];
        if (renderers == null)
            return;

        Color color = supportLayer == ScanCoverSurfaceSupportLayer.StereoConfirmed ? stereoConfirmedColor : monoSupportedColor;
        _propertyBlock.Clear();
        _propertyBlock.SetColor("_BaseColor", color);
        _propertyBlock.SetColor("_Color", color);

        foreach (Renderer renderer in renderers)
        {
            if (renderer)
                renderer.SetPropertyBlock(_propertyBlock);
        }
    }

    private static Vector3 MergeNormals(Vector3 a, Vector3 b)
    {
        if (a.sqrMagnitude > 1e-6f && b.sqrMagnitude > 1e-6f)
            return (a + b).normalized;
        return a.sqrMagnitude > b.sqrMagnitude ? a : b;
    }

    private static void SelectFallbackByDistance(
        Vector3 leftPos,
        Vector3 leftNormal,
        float leftDepth,
        Vector3 rightPos,
        Vector3 rightNormal,
        float rightDepth,
        out Vector3 worldPos,
        out Vector3 worldNormal,
        out float linearDepth,
        out Eye sourceEye)
    {
        if (leftDepth <= rightDepth)
        {
            worldPos = leftPos;
            worldNormal = leftNormal;
            linearDepth = leftDepth;
            sourceEye = Eye.Left;
        }
        else
        {
            worldPos = rightPos;
            worldNormal = rightNormal;
            linearDepth = rightDepth;
            sourceEye = Eye.Right;
        }
    }

    private static bool IsFinite(Vector3 p)
    {
        return !float.IsNaN(p.x) && !float.IsNaN(p.y) && !float.IsNaN(p.z) &&
               !float.IsInfinity(p.x) && !float.IsInfinity(p.y) && !float.IsInfinity(p.z);
    }
}
