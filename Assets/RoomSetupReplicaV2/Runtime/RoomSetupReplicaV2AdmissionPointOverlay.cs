using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Read-only view of the exact remaining pre-TSDF layer: callback-owned raw
/// right-eye surface anchors at or below the 3.0 m production range limit,
/// after removing strong-grazing bands no more than five points thick. The
/// left eye is not published. These are exactly
/// the observations made available to projective TSDF integration; the
/// overlay never writes reconstruction state.
/// </summary>
[DisallowMultipleComponent]
public sealed class RoomSetupReplicaV2AdmissionPointOverlay : MonoBehaviour
{
    [Serializable]
    private sealed class FrozenPointReceipt
    {
        public string schema;
        public string createdUtc;
        public string coordinateSpace;
        public string route;
        public string productionAdmissionSource;
        public bool leftEyeFusionEnabled;
        public int platformFrame;
        public int width;
        public int height;
        public int pointCount;
        public int leftEyePointCount;
        public int rightEyePointCount;
        public int stepQualifiedPointCount;
        public long stepSplitOverreachCoreEyePixels;
        public long stepSplitOverreachRolledBackEyePixels;
        public long sharedConvexFlapCandidateEyePixels;
        public long sharedConvexFlapRejectedEyePixels;
        public long structuralStepFamilyRetiredVoxels;
        public long strongGrazingPoints;
        public long thinGrazingRejectedPoints;
        public long wideGrazingRetainedPoints;
        public long thinGrazingBarrierVoxels;
        public long thinGrazingClearedVoxels;
        public long thinGrazingReleasedVoxels;
        public string plyFile;
        public string csvFile;
    }

    [SerializeField, Range(1f, 8f)] private float pointSize = 3.5f;
    [SerializeField, Range(0.05f, 1f)] private float refreshInterval = 0.15f;
    [SerializeField] private Color leftEyeColor = new(1f, 0.72f, 0.08f, 0.82f);
    [SerializeField] private Color rightEyeColor = new(0.08f, 0.92f, 1f, 0.82f);
    [Header("Admission terrain contours (display only)")]
    [SerializeField, Range(0.01f, 0.10f)]
    private float terrainConnectivityMeters = 0.04f;
    [SerializeField, Range(0.0025f, 0.03f)]
    private float terrainContourSpacingMeters = 0.005f;
    [SerializeField, Range(0.01f, 0.15f)]
    private float terrainColorRangeMeters = 0.025f;
    [SerializeField] private Color terrainOutwardColor =
        new(1f, 0.30f, 0.02f, 0.95f);
    [SerializeField] private Color terrainReferenceColor =
        new(0.03f, 0.82f, 1f, 0.95f);
    [SerializeField] private Color terrainInwardColor =
        new(0.92f, 0.02f, 0.82f, 0.95f);

    private readonly List<Vector3> _vertices = new(131072);
    private readonly List<Color> _colors = new(131072);
    private readonly List<int> _indices = new(131072);
    private readonly List<byte> _eyes = new(131072);
    private readonly List<int> _pixelIndices = new(131072);
    private readonly List<byte> _stepQualified = new(131072);
    private readonly List<float> _projectiveDepths = new(131072);
    private readonly List<float> _terrainResiduals = new(131072);
    private readonly List<TerrainComponent> _terrainComponents = new(256);
    private int[] _terrainComponentIds;
    private int[] _terrainQueue;
    private Mesh _mesh;
    private MeshRenderer _renderer;
    private Material _material;
    private bool _visible;
    private bool _frozen;
    private bool _terrainContours;
    private float _nextRefreshAt;
    private int _snapshotWidth;
    private int _snapshotHeight;
    private int _snapshotPlatformFrame;
    private string _snapshotRoute = string.Empty;
    private long _snapshotStepSplitOverreachCoreEyePixels;
    private long _snapshotStepSplitOverreachRolledBackEyePixels;
    private long _snapshotSharedConvexFlapCandidateEyePixels;
    private long _snapshotSharedConvexFlapRejectedEyePixels;
    private long _snapshotStructuralStepFamilyRetiredVoxels;
    private long _snapshotStrongGrazingPoints;
    private long _snapshotThinGrazingRejectedPoints;
    private long _snapshotWideGrazingRetainedPoints;
    private long _snapshotThinGrazingBarrierVoxels;
    private long _snapshotThinGrazingClearedVoxels;
    private long _snapshotThinGrazingReleasedVoxels;

    public bool Visible => _visible;

    private struct TerrainComponent
    {
        public int count;
        public double sumZ;
        public float referenceDepth;
    }

    private void Awake()
    {
        Shader shader = Resources.Load<Shader>("RoomSetupReplicaV2AdmissionPoints");
        if (shader == null)
            shader = Shader.Find("Genesis/RoomSetupReplicaV2AdmissionPoints");
        if (shader == null)
        {
            enabled = false;
            return;
        }

        _mesh = new Mesh
        {
            name = "Room Setup Replica V2 TSDF Input Points",
            indexFormat = IndexFormat.UInt32,
            bounds = new Bounds(Vector3.zero, Vector3.one * 10000f)
        };
        _mesh.MarkDynamic();
        gameObject.AddComponent<MeshFilter>().sharedMesh = _mesh;
        _renderer = gameObject.AddComponent<MeshRenderer>();
        _renderer.shadowCastingMode = ShadowCastingMode.Off;
        _renderer.receiveShadows = false;
        _material = new Material(shader)
        {
            name = "Room Setup Replica V2 TSDF Input Points"
        };
        _material.SetFloat("_PointSize", pointSize);
        _renderer.sharedMaterial = _material;
        _renderer.enabled = false;
    }

    public void SetVisible(bool visible)
    {
        _visible = visible;
        enabled = visible;
        if (_renderer != null)
            _renderer.enabled = visible && _mesh != null && _mesh.vertexCount > 0;
        if (visible)
            _nextRefreshAt = 0f;
    }

    public void SetFrozen(bool frozen)
    {
        _frozen = frozen;
    }

    public void SetTerrainContours(bool enabled)
    {
        if (_terrainContours == enabled)
            return;
        _terrainContours = enabled;
        _nextRefreshAt = 0f;
    }

    public void ClearPoints()
    {
        _vertices.Clear();
        _colors.Clear();
        _indices.Clear();
        _eyes.Clear();
        _pixelIndices.Clear();
        _stepQualified.Clear();
        _projectiveDepths.Clear();
        _terrainResiduals.Clear();
        _snapshotWidth = 0;
        _snapshotHeight = 0;
        _snapshotPlatformFrame = 0;
        _snapshotRoute = string.Empty;
        _snapshotStepSplitOverreachCoreEyePixels = 0;
        _snapshotStepSplitOverreachRolledBackEyePixels = 0;
        _snapshotSharedConvexFlapCandidateEyePixels = 0;
        _snapshotSharedConvexFlapRejectedEyePixels = 0;
        _snapshotStructuralStepFamilyRetiredVoxels = 0;
        _snapshotStrongGrazingPoints = 0;
        _snapshotThinGrazingRejectedPoints = 0;
        _snapshotWideGrazingRetainedPoints = 0;
        _snapshotThinGrazingBarrierVoxels = 0;
        _snapshotThinGrazingClearedVoxels = 0;
        _snapshotThinGrazingReleasedVoxels = 0;
        _mesh?.Clear(false);
        if (_renderer != null)
            _renderer.enabled = false;
    }

    public void Publish(
        NativeArray<Color> leftPixels,
        byte[] leftAccepted,
        float[] leftDepths,
        byte[] leftStepQualified,
        NativeArray<Color> rightPixels,
        byte[] rightAccepted,
        float[] rightDepths,
        byte[] rightStepQualified,
        int width,
        int height,
        int platformFrame,
        string route,
        long stepSplitOverreachCoreEyePixels,
        long stepSplitOverreachRolledBackEyePixels,
        long sharedConvexFlapCandidateEyePixels,
        long sharedConvexFlapRejectedEyePixels,
        long structuralStepFamilyRetiredVoxels,
        long strongGrazingPoints,
        long thinGrazingRejectedPoints,
        long wideGrazingRetainedPoints,
        long thinGrazingBarrierVoxels,
        long thinGrazingClearedVoxels,
        long thinGrazingReleasedVoxels)
    {
        if (!_visible || _frozen || _mesh == null ||
            Time.unscaledTime < _nextRefreshAt)
            return;
        _nextRefreshAt = Time.unscaledTime + Mathf.Max(0.01f, refreshInterval);

        _vertices.Clear();
        _colors.Clear();
        _eyes.Clear();
        _pixelIndices.Clear();
        _stepQualified.Clear();
        _projectiveDepths.Clear();
        Append(leftPixels, leftAccepted, leftDepths, leftStepQualified,
            leftEyeColor, 0, width, height);
        Append(rightPixels, rightAccepted, rightDepths, rightStepQualified,
            rightEyeColor, 1, width, height);
        _snapshotWidth = width;
        _snapshotHeight = height;
        _snapshotPlatformFrame = platformFrame;
        _snapshotRoute = route ?? string.Empty;
        _snapshotStepSplitOverreachCoreEyePixels =
            stepSplitOverreachCoreEyePixels;
        _snapshotStepSplitOverreachRolledBackEyePixels =
            stepSplitOverreachRolledBackEyePixels;
        _snapshotSharedConvexFlapCandidateEyePixels =
            sharedConvexFlapCandidateEyePixels;
        _snapshotSharedConvexFlapRejectedEyePixels =
            sharedConvexFlapRejectedEyePixels;
        _snapshotStructuralStepFamilyRetiredVoxels =
            structuralStepFamilyRetiredVoxels;
        _snapshotStrongGrazingPoints = strongGrazingPoints;
        _snapshotThinGrazingRejectedPoints =
            thinGrazingRejectedPoints;
        _snapshotWideGrazingRetainedPoints =
            wideGrazingRetainedPoints;
        _snapshotThinGrazingBarrierVoxels =
            thinGrazingBarrierVoxels;
        _snapshotThinGrazingClearedVoxels =
            thinGrazingClearedVoxels;
        _snapshotThinGrazingReleasedVoxels =
            thinGrazingReleasedVoxels;

        _indices.Clear();
        for (int i = 0; i < _vertices.Count; i++)
            _indices.Add(i);

        _mesh.Clear(false);
        if (_vertices.Count > 0)
        {
            _mesh.SetVertices(_vertices);
            _mesh.SetColors(_colors);
            _mesh.SetIndices(_indices, MeshTopology.Points, 0, false);
            _mesh.bounds = new Bounds(Vector3.zero, Vector3.one * 10000f);
        }
        _material.SetFloat("_PointSize", pointSize);
        _renderer.enabled = _visible && _vertices.Count > 0;
    }

    private void Append(
        NativeArray<Color> pixels,
        byte[] accepted,
        float[] depths,
        byte[] qualified,
        Color color,
        byte eye,
        int width,
        int height)
    {
        if (!pixels.IsCreated || accepted == null)
            return;
        int count = Mathf.Min(pixels.Length, accepted.Length);
        if (_terrainContours)
            BuildTerrainComponents(accepted, depths, width, height, count);
        for (int i = 0; i < count; i++)
        {
            if (accepted[i] == 0)
                continue;
            Color encoded = pixels[i];
            Vector3 world = new(encoded.r, encoded.g, encoded.b);
            if (!float.IsFinite(world.x) || !float.IsFinite(world.y) ||
                !float.IsFinite(world.z))
                continue;
            float residual = _terrainContours
                ? TerrainResidual(i, depths)
                : 0f;
            _vertices.Add(world);
            _colors.Add(_terrainContours ? TerrainColor(residual) : color);
            _eyes.Add(eye);
            _pixelIndices.Add(i);
            _stepQualified.Add(qualified != null && i < qualified.Length
                ? qualified[i]
                : (byte)0);
            _projectiveDepths.Add(depths != null && i < depths.Length
                ? depths[i]
                : 0f);
            _terrainResiduals.Add(residual);
        }
    }

    private void BuildTerrainComponents(
        byte[] accepted,
        float[] depths,
        int width,
        int height,
        int count)
    {
        if (depths == null || width <= 0 || height <= 0)
            return;
        if (_terrainComponentIds == null ||
            _terrainComponentIds.Length < count)
        {
            _terrainComponentIds = new int[count];
            _terrainQueue = new int[count];
        }
        Array.Fill(_terrainComponentIds, -1, 0, count);
        _terrainComponents.Clear();

        for (int seed = 0; seed < count; seed++)
        {
            if (_terrainComponentIds[seed] >= 0 ||
                accepted[seed] == 0 || depths[seed] <= 0f)
                continue;

            int componentId = _terrainComponents.Count;
            TerrainComponent component = default;
            int read = 0;
            int write = 0;
            _terrainQueue[write++] = seed;
            _terrainComponentIds[seed] = componentId;
            while (read < write)
            {
                int index = _terrainQueue[read++];
                int x = index % width;
                int y = index / width;
                float z = depths[index];
                component.count++;
                component.sumZ += z;

                TryQueueTerrainNeighbour(index - 1, x > 0, componentId,
                    z, accepted, depths, count, ref write);
                TryQueueTerrainNeighbour(index + 1, x + 1 < width,
                    componentId, z, accepted, depths, count, ref write);
                TryQueueTerrainNeighbour(index - width, y > 0, componentId,
                    z, accepted, depths, count, ref write);
                TryQueueTerrainNeighbour(index + width, y + 1 < height,
                    componentId, z, accepted, depths, count, ref write);
            }
            FinishTerrainComponent(ref component);
            _terrainComponents.Add(component);
        }
    }

    private void TryQueueTerrainNeighbour(
        int neighbour,
        bool inBounds,
        int componentId,
        float centerDepth,
        byte[] accepted,
        float[] depths,
        int count,
        ref int write)
    {
        if (!inBounds || neighbour < 0 || neighbour >= count ||
            _terrainComponentIds[neighbour] >= 0 ||
            accepted[neighbour] == 0 || depths[neighbour] <= 0f ||
            Mathf.Abs(depths[neighbour] - centerDepth) >
                terrainConnectivityMeters)
            return;
        _terrainComponentIds[neighbour] = componentId;
        _terrainQueue[write++] = neighbour;
    }

    private static void FinishTerrainComponent(ref TerrainComponent component)
    {
        if (component.count <= 0)
            return;
        // Preserve the very slope this view is meant to expose. A fitted
        // plane erased a linear collapse and made this look like the frontal
        // raw-point view. The connected surface's mean is only a color datum.
        component.referenceDepth = (float)(component.sumZ / component.count);
    }

    private float TerrainResidual(
        int index,
        float[] depths)
    {
        if (depths == null || index < 0 || index >= depths.Length ||
            _terrainComponentIds == null || index >= _terrainComponentIds.Length)
            return 0f;
        int componentId = _terrainComponentIds[index];
        if (componentId < 0 || componentId >= _terrainComponents.Count)
            return 0f;
        TerrainComponent component = _terrainComponents[componentId];
        return depths[index] - component.referenceDepth;
    }

    private Color TerrainColor(float residual)
    {
        float range = Mathf.Max(0.001f, terrainColorRangeMeters);
        float normalized = Mathf.Clamp(residual / range, -1f, 1f);
        Color color = normalized >= 0f
            ? Color.Lerp(terrainReferenceColor, terrainInwardColor, normalized)
            : Color.Lerp(terrainReferenceColor, terrainOutwardColor, -normalized);

        float spacing = Mathf.Max(0.001f, terrainContourSpacingMeters);
        float contourCoordinate = Mathf.Abs(residual) / spacing;
        float repeated = Mathf.Repeat(contourCoordinate, 1f);
        float distanceToContour = Mathf.Min(repeated, 1f - repeated);
        float contour = 1f - Mathf.SmoothStep(0.08f, 0.24f,
            distanceToContour);
        // The raw frontal view is already near-white. Keep the datum cyan and
        // brighten only non-zero 5 mm contours.
        if (Mathf.Abs(residual) < spacing * 0.20f)
            contour = 0f;
        color = Color.Lerp(color, Color.white, contour * 0.38f);
        color.a = 0.95f;
        return color;
    }

    /// <summary>
    /// Writes the exact currently displayed/frozen TSDF-input anchors. The PLY
    /// is convenient for 3D inspection; CSV retains eye, pixel, StepSplit and
    /// projective-depth provenance for offline fitting and residual analysis.
    /// </summary>
    public bool ExportFrozenSnapshot(out string directory)
    {
        directory = string.Empty;
        if (!_frozen || _vertices.Count == 0 ||
            _vertices.Count != _eyes.Count ||
            _vertices.Count != _pixelIndices.Count ||
            _vertices.Count != _stepQualified.Count ||
            _vertices.Count != _projectiveDepths.Count)
            return false;

        string stamp = DateTime.UtcNow.ToString("yyyyMMddTHHmmssfffZ");
        directory = Path.Combine(
            Application.persistentDataPath,
            "RoomSetupReplicaV2",
            "frozen-admission",
            stamp);
        Directory.CreateDirectory(directory);
        const string plyName = "tsdf-input-points.ply";
        const string csvName = "tsdf-input-points.csv";
        WritePly(Path.Combine(directory, plyName));
        WriteCsv(Path.Combine(directory, csvName));

        int left = 0;
        int step = 0;
        for (int i = 0; i < _vertices.Count; i++)
        {
            if (_eyes[i] == 0) left++;
            if (_stepQualified[i] != 0) step++;
        }
        var receipt = new FrozenPointReceipt
        {
            schema = "roomsetup-replica-v2-frozen-tsdf-input-v9",
            createdUtc = DateTime.UtcNow.ToString("O"),
            coordinateSpace = "callback-frame Unity world",
            route = _snapshotRoute,
            productionAdmissionSource =
                "callback-owned right-eye raw at or below 3.0 m minus strong-grazing bands at most 5 points thick; removed bands become TSDF write and mesh barriers until repeated displaced-view direct surface support releases them",
            leftEyeFusionEnabled = false,
            platformFrame = _snapshotPlatformFrame,
            width = _snapshotWidth,
            height = _snapshotHeight,
            pointCount = _vertices.Count,
            leftEyePointCount = left,
            rightEyePointCount = _vertices.Count - left,
            stepQualifiedPointCount = step,
            stepSplitOverreachCoreEyePixels =
                _snapshotStepSplitOverreachCoreEyePixels,
            stepSplitOverreachRolledBackEyePixels =
                _snapshotStepSplitOverreachRolledBackEyePixels,
            sharedConvexFlapCandidateEyePixels =
                _snapshotSharedConvexFlapCandidateEyePixels,
            sharedConvexFlapRejectedEyePixels =
                _snapshotSharedConvexFlapRejectedEyePixels,
            structuralStepFamilyRetiredVoxels =
                _snapshotStructuralStepFamilyRetiredVoxels,
            strongGrazingPoints = _snapshotStrongGrazingPoints,
            thinGrazingRejectedPoints =
                _snapshotThinGrazingRejectedPoints,
            wideGrazingRetainedPoints =
                _snapshotWideGrazingRetainedPoints,
            thinGrazingBarrierVoxels =
                _snapshotThinGrazingBarrierVoxels,
            thinGrazingClearedVoxels =
                _snapshotThinGrazingClearedVoxels,
            thinGrazingReleasedVoxels =
                _snapshotThinGrazingReleasedVoxels,
            plyFile = plyName,
            csvFile = csvName
        };
        File.WriteAllText(
            Path.Combine(directory, "receipt.json"),
            JsonUtility.ToJson(receipt, true) + "\n",
            new UTF8Encoding(false));
        return true;
    }

    private void WritePly(string path)
    {
        using var writer = new StreamWriter(
            path, false, new UTF8Encoding(false));
        writer.WriteLine("ply");
        writer.WriteLine("format ascii 1.0");
        writer.WriteLine("comment RoomSetupReplicaV2 frozen TSDF input anchors");
        writer.WriteLine("element vertex " + _vertices.Count);
        writer.WriteLine("property float x");
        writer.WriteLine("property float y");
        writer.WriteLine("property float z");
        writer.WriteLine("property uchar red");
        writer.WriteLine("property uchar green");
        writer.WriteLine("property uchar blue");
        writer.WriteLine("end_header");
        for (int i = 0; i < _vertices.Count; i++)
        {
            Vector3 point = _vertices[i];
            Color32 color = _colors[i];
            writer.Write(point.x.ToString("R", CultureInfo.InvariantCulture));
            writer.Write(' ');
            writer.Write(point.y.ToString("R", CultureInfo.InvariantCulture));
            writer.Write(' ');
            writer.Write(point.z.ToString("R", CultureInfo.InvariantCulture));
            writer.Write(' ');
            writer.Write(color.r);
            writer.Write(' ');
            writer.Write(color.g);
            writer.Write(' ');
            writer.WriteLine(color.b);
        }
    }

    private void WriteCsv(string path)
    {
        using var writer = new StreamWriter(
            path, false, new UTF8Encoding(false));
        writer.WriteLine(
            "eye,pixel_index,pixel_x,pixel_y,step_qualified,projective_depth_m,terrain_residual_m,world_x,world_y,world_z");
        for (int i = 0; i < _vertices.Count; i++)
        {
            Vector3 point = _vertices[i];
            int pixel = _pixelIndices[i];
            int x = _snapshotWidth > 0 ? pixel % _snapshotWidth : -1;
            int y = _snapshotWidth > 0 ? pixel / _snapshotWidth : -1;
            writer.Write(_eyes[i] == 0 ? "left" : "right");
            writer.Write(',');
            writer.Write(pixel);
            writer.Write(',');
            writer.Write(x);
            writer.Write(',');
            writer.Write(y);
            writer.Write(',');
            writer.Write(_stepQualified[i]);
            writer.Write(',');
            writer.Write(_projectiveDepths[i].ToString(
                "R", CultureInfo.InvariantCulture));
            writer.Write(',');
            writer.Write(_terrainResiduals[i].ToString(
                "R", CultureInfo.InvariantCulture));
            writer.Write(',');
            writer.Write(point.x.ToString("R", CultureInfo.InvariantCulture));
            writer.Write(',');
            writer.Write(point.y.ToString("R", CultureInfo.InvariantCulture));
            writer.Write(',');
            writer.Write(point.z.ToString("R", CultureInfo.InvariantCulture));
            writer.WriteLine();
        }
    }

    private void OnDestroy()
    {
        if (_mesh != null)
            Destroy(_mesh);
        if (_material != null)
            Destroy(_material);
    }
}
