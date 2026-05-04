using System.Collections.Generic;
using Meta.XR.EnvironmentDepth;
using MyProject.XR;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// Minimal 64x64 raw depth-wrapped grid snapshot.
/// Samples world positions directly from CustomEnvironmentDepthRaycaster and renders only grid lines.
/// No preprocessor, no remesh, no surface mesh, no color, no normal reconstruction.
/// </summary>
[DefaultExecutionOrder(-40)]
[DisallowMultipleComponent]
public sealed class ScanCoverDepthGridSnapshot64 : MonoBehaviour
{
    private const int GridSize = 64;

    [Header("Dependencies")]
    [SerializeField] private CustomEnvironmentDepthRaycaster depthRaycaster;
    [SerializeField] private EnvironmentDepthManager environmentDepthManager;
    [SerializeField] private ScanCoverSkeletonSessionController sessionController;
    [SerializeField] private Transform displayParent;

    [Header("Snapshot")]
    [SerializeField] private bool captureOnStart = true;
    [SerializeField] private bool retryUntilFirstSnapshot = true;
    [SerializeField] private bool updateEveryFrame;
    [SerializeField] private KeyCode captureKey = KeyCode.Space;
    [SerializeField] private bool captureWhenSessionFrozen = true;
    [SerializeField] private Eye eye = Eye.Right;

    [Header("Roll Lock")]
    [SerializeField] private bool freezeDisplayInWorldSpace = true;
    [SerializeField] private bool compensateHeadsetRollSampling;
    [SerializeField] private Transform rollReference;

    [Header("Depth Filter")]
    [SerializeField] private bool depthPixelVFlip = true;
    [SerializeField] private bool neighborFill = true;
    [SerializeField, Min(1)] private int neighborRadiusPixels = 1;
    [SerializeField, Min(0f)] private float minLinearDepthMeters = 0.05f;
    [SerializeField, Min(0f)] private float maxLinearDepthMeters = 8f;

    [Header("Prefield Renet")]
    [SerializeField] private bool usePrefieldRenet = false;
    [SerializeField, Range(8, 64)] private int renetGridSize = 32;
    [SerializeField, Min(0.01f)] private float maxPrefieldCellEdgeMeters = 0.7f;
    [SerializeField, Min(0.01f)] private float maxPrefieldCellDepthDeltaMeters = 0.45f;
    [SerializeField, Min(0.01f)] private float maxRenetSegmentMeters = 0.8f;
    [SerializeField, Min(0.01f)] private float maxRenetDepthDeltaMeters = 0.5f;

    [Header("World Rule Renet")]
    [SerializeField] private bool useWorldRuleRenet = true;
    [SerializeField, Range(0.1f, 1.5f)] private float worldRuleCoverage = 1f;

    [Header("Debug")]
    [SerializeField] private bool debugLog = true;

    private readonly Vector3[] _positions = new Vector3[GridSize * GridSize];
    private readonly float[] _linearDepths = new float[GridSize * GridSize];
    private readonly bool[] _valid = new bool[GridSize * GridSize];
    private readonly bool[] _cellValid = new bool[(GridSize - 1) * (GridSize - 1)];
    private readonly List<Vector3> _lineVertices = new List<Vector3>(GridSize * GridSize * 4);
    private readonly List<int> _lineIndices = new List<int>(GridSize * GridSize * 4);

    private GameObject _gridObject;
    private Mesh _gridMesh;
    private Material _gridMaterial;
    private bool _pendingInitialCapture;
    private ScanCoverSkeletonSessionController.SessionState _lastSessionState;

    private void Awake()
    {
        ResolveRefs();
        EnsureGridObject();
        _pendingInitialCapture = captureOnStart;
        _lastSessionState = sessionController != null
            ? sessionController.State
            : ScanCoverSkeletonSessionController.SessionState.Idle;
    }

    private void Update()
    {
        if (SessionFreezeTriggered())
        {
            _pendingInitialCapture = false;
            CaptureSnapshot();
        }

        if (CaptureKeyPressed())
        {
            _pendingInitialCapture = false;
            CaptureSnapshot();
        }

        if (updateEveryFrame)
        {
            CaptureSnapshot();
            return;
        }

        if (_pendingInitialCapture && retryUntilFirstSnapshot && CaptureSnapshot())
            _pendingInitialCapture = false;
    }

    private bool CaptureKeyPressed()
    {
        if (captureKey == KeyCode.None)
            return false;

#if ENABLE_INPUT_SYSTEM
        if (captureKey == KeyCode.Space)
            return Keyboard.current != null && Keyboard.current.spaceKey.wasPressedThisFrame;

        return false;
#elif ENABLE_LEGACY_INPUT_MANAGER
        return Input.GetKeyDown(captureKey);
#else
        return false;
#endif
    }

    private bool SessionFreezeTriggered()
    {
        if (!captureWhenSessionFrozen)
            return false;

        if (sessionController == null)
            ResolveRefs();

        if (sessionController == null)
            return false;

        var state = sessionController.State;
        bool triggered = state == ScanCoverSkeletonSessionController.SessionState.Frozen &&
                         _lastSessionState != ScanCoverSkeletonSessionController.SessionState.Frozen;
        _lastSessionState = state;
        return triggered;
    }

    [ContextMenu("Capture 64x64 Depth Grid Snapshot")]
    public bool CaptureSnapshot()
    {
        ResolveRefs();
        EnsureGridObject();
        ConfigureGridObjectTransform();

        if (depthRaycaster == null)
        {
            Log("Depth raycaster is missing.");
            return false;
        }

        depthRaycaster.SetEye(eye);
        if (!depthRaycaster.IsDepthTextureAvailable)
        {
            Log("Depth texture is not ready yet.");
            return false;
        }

        Sample64x64();
        int validCount = CountValid();
        if (validCount <= 0)
        {
            ClearSnapshot();
            Log("Captured no valid depth points.");
            return false;
        }

        if (usePrefieldRenet)
        {
            BuildPrefieldCells();
            BuildRenetFromPrefield();
        }
        else
        {
            BuildLineGrid();
        }

        _gridObject.SetActive(true);

        Log($"Captured 64x64 depth grid. mode={(usePrefieldRenet ? "prefield-renet" : "raw")}, valid={validCount}/{GridSize * GridSize}, lines={_lineIndices.Count / 2}");
        return true;
    }

    [ContextMenu("Clear Snapshot")]
    public void ClearSnapshot()
    {
        if (_gridMesh != null)
            _gridMesh.Clear();
        if (_gridObject != null)
            _gridObject.SetActive(false);
    }

    private void ResolveRefs()
    {
        if (environmentDepthManager == null)
            environmentDepthManager = FindAnyObjectByType<EnvironmentDepthManager>(FindObjectsInactive.Include);

        if (sessionController == null)
            sessionController = FindAnyObjectByType<ScanCoverSkeletonSessionController>(FindObjectsInactive.Include);

        if (depthRaycaster == null)
            depthRaycaster = FindAnyObjectByType<CustomEnvironmentDepthRaycaster>(FindObjectsInactive.Include);

        if (depthRaycaster == null)
            depthRaycaster = CreateRuntimeRaycaster();

        if (depthRaycaster != null)
        {
            if (environmentDepthManager != null && depthRaycaster.depthManager == null)
                depthRaycaster.depthManager = environmentDepthManager;

            if (!depthRaycaster.gameObject.activeSelf)
                depthRaycaster.gameObject.SetActive(true);
            if (!depthRaycaster.enabled)
                depthRaycaster.enabled = true;
        }
    }

    private CustomEnvironmentDepthRaycaster CreateRuntimeRaycaster()
    {
        var raycasterObject = new GameObject("ScanCover Runtime Depth Raycaster");
        raycasterObject.SetActive(false);
        raycasterObject.transform.SetParent(transform, false);
        var raycaster = raycasterObject.AddComponent<CustomEnvironmentDepthRaycaster>();
        raycaster.depthManager = environmentDepthManager;
        raycasterObject.SetActive(true);
        return raycaster;
    }

    private void EnsureGridObject()
    {
        if (_gridObject != null)
            return;

        _gridObject = new GameObject("ScanCover 64x64 Raw Depth Grid Snapshot");
        ConfigureGridObjectTransform();

        _gridMesh = new Mesh { name = "ScanCover_64x64_RawDepthGridSnapshot" };
        _gridObject.AddComponent<MeshFilter>().sharedMesh = _gridMesh;
        _gridObject.AddComponent<MeshRenderer>().sharedMaterial = GetGridMaterial();
        _gridObject.SetActive(false);
    }

    private void ConfigureGridObjectTransform()
    {
        if (_gridObject == null)
            return;

        Transform parent = freezeDisplayInWorldSpace ? null : (displayParent != null ? displayParent : transform);
        if (_gridObject.transform.parent != parent)
            _gridObject.transform.SetParent(parent, false);

        _gridObject.transform.localPosition = Vector3.zero;
        _gridObject.transform.localRotation = Quaternion.identity;
        _gridObject.transform.localScale = Vector3.one;
    }

    private Material GetGridMaterial()
    {
        if (_gridMaterial != null)
            return _gridMaterial;

        Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null)
            shader = Shader.Find("Unlit/Color");

        _gridMaterial = new Material(shader) { name = "ScanCover_64x64_RawGridLine" };
        if (_gridMaterial.HasProperty("_BaseColor"))
            _gridMaterial.SetColor("_BaseColor", Color.white);
        if (_gridMaterial.HasProperty("_Color"))
            _gridMaterial.SetColor("_Color", Color.white);
        return _gridMaterial;
    }

    private void Sample64x64()
    {
        int textureSize = CustomEnvironmentDepthRaycaster.TextureSize;
        int maxCoord = textureSize - 1;

        for (int row = 0; row < GridSize; row++)
        {
            for (int col = 0; col < GridSize; col++)
            {
                int index = ToIndex(col, row);
                if (!TryResolveSamplePixel(col, row, maxCoord, out int x, out int y))
                {
                    _positions[index] = Vector3.zero;
                    _linearDepths[index] = 0f;
                    _valid[index] = false;
                    continue;
                }

                if (TryReadWorldPoint(x, y, out Vector3 worldPoint, out float linearDepthMeters) ||
                    neighborFill && TryReadNeighborWorldPoint(x, y, out worldPoint, out linearDepthMeters))
                {
                    _positions[index] = worldPoint;
                    _linearDepths[index] = linearDepthMeters;
                    _valid[index] = true;
                }
                else
                {
                    _positions[index] = Vector3.zero;
                    _linearDepths[index] = 0f;
                    _valid[index] = false;
                }
            }
        }
    }

    private bool TryResolveSamplePixel(int col, int row, int maxCoord, out int x, out int y)
    {
        Vector2 uv = new Vector2(
            col / (float)(GridSize - 1),
            row / (float)(GridSize - 1));

        if (compensateHeadsetRollSampling)
        {
            float rollDegrees = ResolveRollCompensationDegrees();
            if (Mathf.Abs(rollDegrees) > 0.001f)
            {
                float radians = rollDegrees * Mathf.Deg2Rad;
                float sin = Mathf.Sin(radians);
                float cos = Mathf.Cos(radians);
                Vector2 centered = uv - new Vector2(0.5f, 0.5f);
                uv = new Vector2(
                    centered.x * cos - centered.y * sin,
                    centered.x * sin + centered.y * cos) + new Vector2(0.5f, 0.5f);
            }
        }

        if (uv.x < 0f || uv.x > 1f || uv.y < 0f || uv.y > 1f)
        {
            x = 0;
            y = 0;
            return false;
        }

        x = Mathf.RoundToInt(uv.x * maxCoord);
        float sampledY = depthPixelVFlip ? 1f - uv.y : uv.y;
        y = Mathf.RoundToInt(sampledY * maxCoord);
        return true;
    }

    private float ResolveRollCompensationDegrees()
    {
        Transform reference = ResolveRollReference();
        if (reference == null)
            return 0f;

        Vector3 forward = reference.forward;
        if (forward.sqrMagnitude < 0.000001f)
            return 0f;

        Vector3 cameraUp = Vector3.ProjectOnPlane(reference.up, forward);
        Vector3 worldUp = Vector3.ProjectOnPlane(Vector3.up, forward);
        if (cameraUp.sqrMagnitude < 0.000001f || worldUp.sqrMagnitude < 0.000001f)
            return 0f;

        return Vector3.SignedAngle(cameraUp.normalized, worldUp.normalized, forward.normalized);
    }

    private Transform ResolveRollReference()
    {
        if (rollReference != null)
            return rollReference;

        Camera mainCamera = Camera.main;
        return mainCamera != null ? mainCamera.transform : transform;
    }

    private bool TryReadNeighborWorldPoint(int centerX, int centerY, out Vector3 worldPoint, out float linearDepthMeters)
    {
        int radiusMax = Mathf.Max(1, neighborRadiusPixels);
        for (int radius = 1; radius <= radiusMax; radius++)
        {
            for (int dy = -radius; dy <= radius; dy++)
            {
                for (int dx = -radius; dx <= radius; dx++)
                {
                    if (dx == 0 && dy == 0)
                        continue;

                    int x = centerX + dx;
                    int y = centerY + dy;
                    if (!CustomEnvironmentDepthRaycaster.IsInBounds02(new Vector2Int(x, y)))
                        continue;

                    if (TryReadWorldPoint(x, y, out worldPoint, out linearDepthMeters))
                        return true;
                }
            }
        }

        worldPoint = default;
        linearDepthMeters = 0f;
        return false;
    }

    private bool TryReadWorldPoint(int x, int y, out Vector3 worldPoint, out float linearDepthMeters)
    {
        var texCoord = new Vector2Int(x, y);
        worldPoint = depthRaycaster.WorldPosAtDepthTexCoord02(texCoord);
        linearDepthMeters = 0f;
        if (!IsFinite(worldPoint))
            return false;

        linearDepthMeters = depthRaycaster.WorldPosToLinearDepth02(worldPoint);
        return linearDepthMeters >= minLinearDepthMeters && linearDepthMeters <= maxLinearDepthMeters;
    }

    private void BuildPrefieldCells()
    {
        for (int row = 0; row < GridSize - 1; row++)
        {
            for (int col = 0; col < GridSize - 1; col++)
            {
                int cellIndex = ToCellIndex(col, row);
                int i00 = ToIndex(col, row);
                int i10 = ToIndex(col + 1, row);
                int i01 = ToIndex(col, row + 1);
                int i11 = ToIndex(col + 1, row + 1);

                _cellValid[cellIndex] =
                    IsPrefieldCellUsable(i00, i10, i01, i11);
            }
        }
    }

    private bool IsPrefieldCellUsable(int i00, int i10, int i01, int i11)
    {
        if (!_valid[i00] || !_valid[i10] || !_valid[i01] || !_valid[i11])
            return false;

        float maxEdge = Mathf.Max(0.01f, maxPrefieldCellEdgeMeters);
        if (Vector3.Distance(_positions[i00], _positions[i10]) > maxEdge ||
            Vector3.Distance(_positions[i01], _positions[i11]) > maxEdge ||
            Vector3.Distance(_positions[i00], _positions[i01]) > maxEdge ||
            Vector3.Distance(_positions[i10], _positions[i11]) > maxEdge)
            return false;

        float minDepth = Mathf.Min(Mathf.Min(_linearDepths[i00], _linearDepths[i10]), Mathf.Min(_linearDepths[i01], _linearDepths[i11]));
        float maxDepth = Mathf.Max(Mathf.Max(_linearDepths[i00], _linearDepths[i10]), Mathf.Max(_linearDepths[i01], _linearDepths[i11]));
        return maxDepth - minDepth <= Mathf.Max(0.01f, maxPrefieldCellDepthDeltaMeters);
    }

    private void BuildRenetFromPrefield()
    {
        if (useWorldRuleRenet && TryBuildWorldRuleRenet())
            return;

        Transform root = _gridObject != null ? _gridObject.transform : (displayParent != null ? displayParent : transform);
        _lineVertices.Clear();
        _lineIndices.Clear();

        int size = Mathf.Clamp(renetGridSize, 2, GridSize);
        var renetPositions = new Vector3[size * size];
        var renetDepths = new float[size * size];
        var renetValid = new bool[size * size];

        for (int row = 0; row < size; row++)
        {
            for (int col = 0; col < size; col++)
            {
                Vector2 uv = new Vector2(
                    col / (float)(size - 1),
                    row / (float)(size - 1));

                int index = row * size + col;
                renetValid[index] = TryQueryPrefield(uv, out renetPositions[index], out renetDepths[index]);
            }
        }

        for (int row = 0; row < size; row++)
        {
            for (int col = 0; col < size; col++)
            {
                int index = row * size + col;
                AddRenetEdge(index, col + 1, row, size, renetPositions, renetDepths, renetValid, root);
                AddRenetEdge(index, col, row + 1, size, renetPositions, renetDepths, renetValid, root);
            }
        }

        _gridMesh.Clear();
        _gridMesh.SetVertices(_lineVertices);
        _gridMesh.SetIndices(_lineIndices, MeshTopology.Lines, 0);
        _gridMesh.RecalculateBounds();
    }

    private bool TryBuildWorldRuleRenet()
    {
        Transform root = _gridObject != null ? _gridObject.transform : (displayParent != null ? displayParent : transform);
        int size = Mathf.Clamp(renetGridSize, 2, GridSize);
        Vector2[] fieldPoints = new Vector2[GridSize * GridSize];

        if (!TryBuildFieldCoordinates(fieldPoints, out Vector2 min, out Vector2 max))
            return false;

        Vector2 center = (min + max) * 0.5f;
        Vector2 halfSize = (max - min) * 0.5f * Mathf.Max(0.1f, worldRuleCoverage);
        min = center - halfSize;
        max = center + halfSize;

        var renetPositions = new Vector3[size * size];
        var renetDepths = new float[size * size];
        var renetValid = new bool[size * size];

        for (int row = 0; row < size; row++)
        {
            float y = Mathf.Lerp(min.y, max.y, row / (float)(size - 1));
            for (int col = 0; col < size; col++)
            {
                float x = Mathf.Lerp(min.x, max.x, col / (float)(size - 1));
                int index = row * size + col;
                renetValid[index] = TryQueryPrefieldAtFieldPoint(
                    new Vector2(x, y),
                    fieldPoints,
                    out renetPositions[index],
                    out renetDepths[index]);
            }
        }

        _lineVertices.Clear();
        _lineIndices.Clear();
        for (int row = 0; row < size; row++)
        {
            for (int col = 0; col < size; col++)
            {
                int index = row * size + col;
                AddRenetEdge(index, col + 1, row, size, renetPositions, renetDepths, renetValid, root);
                AddRenetEdge(index, col, row + 1, size, renetPositions, renetDepths, renetValid, root);
            }
        }

        _gridMesh.Clear();
        _gridMesh.SetVertices(_lineVertices);
        _gridMesh.SetIndices(_lineIndices, MeshTopology.Lines, 0);
        _gridMesh.RecalculateBounds();
        return _lineIndices.Count > 0;
    }

    private bool TryBuildFieldCoordinates(Vector2[] fieldPoints, out Vector2 min, out Vector2 max)
    {
        Transform reference = ResolveRollReference();
        Vector3 axisX = reference != null ? reference.right : transform.right;
        Vector3 axisY = reference != null ? reference.up : transform.up;
        if (axisX.sqrMagnitude < 0.000001f || axisY.sqrMagnitude < 0.000001f)
        {
            min = default;
            max = default;
            return false;
        }

        axisX.Normalize();
        axisY.Normalize();

        Vector3 origin = Vector3.zero;
        int originCount = 0;
        for (int i = 0; i < _positions.Length; i++)
        {
            if (!_valid[i])
                continue;

            origin += _positions[i];
            originCount++;
        }

        if (originCount <= 0)
        {
            min = default;
            max = default;
            return false;
        }

        origin /= originCount;
        min = new Vector2(float.PositiveInfinity, float.PositiveInfinity);
        max = new Vector2(float.NegativeInfinity, float.NegativeInfinity);
        bool hasUsableCell = false;

        for (int row = 0; row < GridSize; row++)
        {
            for (int col = 0; col < GridSize; col++)
            {
                int index = ToIndex(col, row);
                Vector3 offset = _positions[index] - origin;
                fieldPoints[index] = new Vector2(Vector3.Dot(offset, axisX), Vector3.Dot(offset, axisY));
            }
        }

        for (int row = 0; row < GridSize - 1; row++)
        {
            for (int col = 0; col < GridSize - 1; col++)
            {
                if (!_cellValid[ToCellIndex(col, row)])
                    continue;

                IncludeFieldPoint(fieldPoints[ToIndex(col, row)], ref min, ref max);
                IncludeFieldPoint(fieldPoints[ToIndex(col + 1, row)], ref min, ref max);
                IncludeFieldPoint(fieldPoints[ToIndex(col, row + 1)], ref min, ref max);
                IncludeFieldPoint(fieldPoints[ToIndex(col + 1, row + 1)], ref min, ref max);
                hasUsableCell = true;
            }
        }

        return hasUsableCell && IsFinite(min) && IsFinite(max) && (max - min).sqrMagnitude > 0.000001f;
    }

    private bool TryQueryPrefieldAtFieldPoint(Vector2 target, Vector2[] fieldPoints, out Vector3 worldPoint, out float linearDepthMeters)
    {
        for (int row = 0; row < GridSize - 1; row++)
        {
            for (int col = 0; col < GridSize - 1; col++)
            {
                if (!_cellValid[ToCellIndex(col, row)])
                    continue;

                int i00 = ToIndex(col, row);
                int i10 = ToIndex(col + 1, row);
                int i01 = ToIndex(col, row + 1);
                int i11 = ToIndex(col + 1, row + 1);

                if (TryInterpolateTriangle(target, fieldPoints[i00], fieldPoints[i10], fieldPoints[i11], i00, i10, i11, out worldPoint, out linearDepthMeters) ||
                    TryInterpolateTriangle(target, fieldPoints[i00], fieldPoints[i11], fieldPoints[i01], i00, i11, i01, out worldPoint, out linearDepthMeters))
                    return true;
            }
        }

        worldPoint = default;
        linearDepthMeters = 0f;
        return false;
    }

    private bool TryInterpolateTriangle(
        Vector2 target,
        Vector2 a,
        Vector2 b,
        Vector2 c,
        int ia,
        int ib,
        int ic,
        out Vector3 worldPoint,
        out float linearDepthMeters)
    {
        if (!TryGetBarycentric(target, a, b, c, out Vector3 weights))
        {
            worldPoint = default;
            linearDepthMeters = 0f;
            return false;
        }

        worldPoint = _positions[ia] * weights.x + _positions[ib] * weights.y + _positions[ic] * weights.z;
        linearDepthMeters = _linearDepths[ia] * weights.x + _linearDepths[ib] * weights.y + _linearDepths[ic] * weights.z;
        return IsFinite(worldPoint);
    }

    private bool TryQueryPrefield(Vector2 uv, out Vector3 worldPoint, out float linearDepthMeters)
    {
        float gridX = Mathf.Clamp01(uv.x) * (GridSize - 1);
        float gridY = Mathf.Clamp01(uv.y) * (GridSize - 1);

        int col = Mathf.Min(Mathf.FloorToInt(gridX), GridSize - 2);
        int row = Mathf.Min(Mathf.FloorToInt(gridY), GridSize - 2);
        float tx = gridX - col;
        float ty = gridY - row;

        if (!_cellValid[ToCellIndex(col, row)])
        {
            worldPoint = default;
            linearDepthMeters = 0f;
            return false;
        }

        int i00 = ToIndex(col, row);
        int i10 = ToIndex(col + 1, row);
        int i01 = ToIndex(col, row + 1);
        int i11 = ToIndex(col + 1, row + 1);

        Vector3 top = Vector3.Lerp(_positions[i00], _positions[i10], tx);
        Vector3 bottom = Vector3.Lerp(_positions[i01], _positions[i11], tx);
        worldPoint = Vector3.Lerp(top, bottom, ty);

        float topDepth = Mathf.Lerp(_linearDepths[i00], _linearDepths[i10], tx);
        float bottomDepth = Mathf.Lerp(_linearDepths[i01], _linearDepths[i11], tx);
        linearDepthMeters = Mathf.Lerp(topDepth, bottomDepth, ty);
        return IsFinite(worldPoint);
    }

    private void AddRenetEdge(
        int fromIndex,
        int toCol,
        int toRow,
        int size,
        Vector3[] renetPositions,
        float[] renetDepths,
        bool[] renetValid,
        Transform root)
    {
        if (!renetValid[fromIndex] || (uint)toCol >= size || (uint)toRow >= size)
            return;

        int toIndex = toRow * size + toCol;
        if (!renetValid[toIndex])
            return;

        if (Vector3.Distance(renetPositions[fromIndex], renetPositions[toIndex]) > Mathf.Max(0.01f, maxRenetSegmentMeters))
            return;

        if (Mathf.Abs(renetDepths[fromIndex] - renetDepths[toIndex]) > Mathf.Max(0.01f, maxRenetDepthDeltaMeters))
            return;

        int lineIndex = _lineVertices.Count;
        _lineVertices.Add(root.InverseTransformPoint(renetPositions[fromIndex]));
        _lineVertices.Add(root.InverseTransformPoint(renetPositions[toIndex]));
        _lineIndices.Add(lineIndex);
        _lineIndices.Add(lineIndex + 1);
    }

    private void BuildLineGrid()
    {
        Transform root = _gridObject != null ? _gridObject.transform : (displayParent != null ? displayParent : transform);
        _lineVertices.Clear();
        _lineIndices.Clear();

        for (int row = 0; row < GridSize; row++)
        {
            for (int col = 0; col < GridSize; col++)
            {
                int index = ToIndex(col, row);
                AddEdge(index, col + 1, row, root);
                AddEdge(index, col, row + 1, root);
            }
        }

        _gridMesh.Clear();
        _gridMesh.SetVertices(_lineVertices);
        _gridMesh.SetIndices(_lineIndices, MeshTopology.Lines, 0);
        _gridMesh.RecalculateBounds();
    }

    private void AddEdge(int fromIndex, int toCol, int toRow, Transform root)
    {
        if (!_valid[fromIndex] || (uint)toCol >= GridSize || (uint)toRow >= GridSize)
            return;

        int toIndex = ToIndex(toCol, toRow);
        if (!_valid[toIndex])
            return;

        int lineIndex = _lineVertices.Count;
        _lineVertices.Add(root.InverseTransformPoint(_positions[fromIndex]));
        _lineVertices.Add(root.InverseTransformPoint(_positions[toIndex]));
        _lineIndices.Add(lineIndex);
        _lineIndices.Add(lineIndex + 1);
    }

    private int CountValid()
    {
        int count = 0;
        for (int i = 0; i < _valid.Length; i++)
        {
            if (_valid[i])
                count++;
        }

        return count;
    }

    private static int ToIndex(int col, int row)
    {
        return row * GridSize + col;
    }

    private static int ToCellIndex(int col, int row)
    {
        return row * (GridSize - 1) + col;
    }

    private static void IncludeFieldPoint(Vector2 value, ref Vector2 min, ref Vector2 max)
    {
        min = Vector2.Min(min, value);
        max = Vector2.Max(max, value);
    }

    private static bool TryGetBarycentric(Vector2 point, Vector2 a, Vector2 b, Vector2 c, out Vector3 weights)
    {
        Vector2 v0 = b - a;
        Vector2 v1 = c - a;
        Vector2 v2 = point - a;
        float denominator = v0.x * v1.y - v1.x * v0.y;
        if (Mathf.Abs(denominator) < 0.000001f)
        {
            weights = default;
            return false;
        }

        float invDenominator = 1f / denominator;
        float v = (v2.x * v1.y - v1.x * v2.y) * invDenominator;
        float w = (v0.x * v2.y - v2.x * v0.y) * invDenominator;
        float u = 1f - v - w;
        const float epsilon = -0.0001f;
        if (u < epsilon || v < epsilon || w < epsilon)
        {
            weights = default;
            return false;
        }

        weights = new Vector3(u, v, w);
        return true;
    }

    private void Log(string message)
    {
        if (debugLog)
            Debug.Log($"[ScanCoverDepthGridSnapshot64] {message}");
    }

    private static bool IsFinite(Vector3 value)
    {
        return float.IsFinite(value.x) && float.IsFinite(value.y) && float.IsFinite(value.z);
    }

    private static bool IsFinite(Vector2 value)
    {
        return float.IsFinite(value.x) && float.IsFinite(value.y);
    }
}
