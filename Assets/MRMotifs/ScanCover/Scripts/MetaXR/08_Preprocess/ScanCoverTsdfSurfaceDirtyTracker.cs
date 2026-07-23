using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Builds a geometry-only fingerprint from TSDF zero crossings before mesh extraction.
/// Fusion weight growth is intentionally ignored: a well-supported surface is not dirty
/// merely because another frame observed it. This is a shadow diagnostic and never writes TSDF.
/// </summary>
public sealed class ScanCoverTsdfSurfaceDirtyTracker
{
    public const int BoundaryNegativeX = 1 << 0;
    public const int BoundaryPositiveX = 1 << 1;
    public const int BoundaryNegativeY = 1 << 2;
    public const int BoundaryPositiveY = 1 << 3;
    public const int BoundaryNegativeZ = 1 << 4;
    public const int BoundaryPositiveZ = 1 << 5;

    public struct Settings
    {
        public float PatchSizeMeters;
        public float StableCrossingRatio;
        public float StableCentroidDistanceMeters;
        public float StableBoundsDistanceMeters;
        public float StableNormalDot;
        public int MinCrossingsForNormal;
        public int RetireMissingRebuilds;
    }

    public struct Result
    {
        public int RebuildSequence;
        public int CurrentBlocks;
        public int CurrentCrossings;
        public int CleanBlocks;
        public int DirtyBlocks;
        public int NewBlocks;
        public int ChangedBlocks;
        public int MissingBlocks;
        public int RetiredBlocks;
    }

    private sealed class Signature
    {
        public int CrossingCount;
        public Vector3 Centroid;
        public Vector3 BoundsMin;
        public Vector3 BoundsMax;
        public Vector3 AverageNormal;
        public int BoundaryMask;
    }

    private sealed class Accumulator
    {
        public int Count;
        public Vector3 PositionSum;
        public Vector3 NormalSum;
        public Vector3 BoundsMin;
        public Vector3 BoundsMax;
        public bool HasBounds;
        public int BoundaryMask;
    }

    private sealed class Track
    {
        public Signature Last;
        public int MissingRebuilds;
    }

    private readonly Dictionary<Vector3Int, Track> _tracks = new Dictionary<Vector3Int, Track>(256);
    private readonly Dictionary<Vector3Int, Accumulator> _accumulators =
        new Dictionary<Vector3Int, Accumulator>(256);
    private readonly Dictionary<Vector3Int, Signature> _current =
        new Dictionary<Vector3Int, Signature>(256);
    private readonly Dictionary<Vector3Int, ScanCoverIncrementalPatchShadow.PatchDirtyReason> _dirty =
        new Dictionary<Vector3Int, ScanCoverIncrementalPatchShadow.PatchDirtyReason>(128);
    private readonly Dictionary<Vector3Int, int> _dependencyBoundaryMasks =
        new Dictionary<Vector3Int, int>(128);
    private int _rebuildSequence;

    public Result Update(
        float[] tsdf,
        byte[] weights,
        int dimX,
        int dimY,
        int dimZ,
        Vector3 origin,
        float voxelSizeMeters,
        Settings settings)
    {
        Sanitize(ref settings, voxelSizeMeters);
        _rebuildSequence++;
        _accumulators.Clear();
        _current.Clear();
        _dirty.Clear();
        _dependencyBoundaryMasks.Clear();
        Result result = new Result { RebuildSequence = _rebuildSequence };
        if (tsdf == null || weights == null || dimX < 2 || dimY < 2 || dimZ < 2)
            return result;

        for (int z = 0; z < dimZ; z++)
        {
            for (int y = 0; y < dimY; y++)
            {
                for (int x = 0; x < dimX; x++)
                {
                    int index = Index(x, y, z, dimX, dimY);
                    if (!ValidSample(index, tsdf, weights))
                        continue;
                    if (x + 1 < dimX)
                        AccumulateEdge(x, y, z, x + 1, y, z, index, Index(x + 1, y, z, dimX, dimY), tsdf, weights, dimX, dimY, dimZ, origin, voxelSizeMeters, settings.PatchSizeMeters);
                    if (y + 1 < dimY)
                        AccumulateEdge(x, y, z, x, y + 1, z, index, Index(x, y + 1, z, dimX, dimY), tsdf, weights, dimX, dimY, dimZ, origin, voxelSizeMeters, settings.PatchSizeMeters);
                    if (z + 1 < dimZ)
                        AccumulateEdge(x, y, z, x, y, z + 1, index, Index(x, y, z + 1, dimX, dimY), tsdf, weights, dimX, dimY, dimZ, origin, voxelSizeMeters, settings.PatchSizeMeters);
                }
            }
        }

        foreach (KeyValuePair<Vector3Int, Accumulator> pair in _accumulators)
        {
            Accumulator a = pair.Value;
            if (a.Count <= 0)
                continue;
            Signature signature = new Signature
            {
                CrossingCount = a.Count,
                Centroid = a.PositionSum / a.Count,
                BoundsMin = a.BoundsMin,
                BoundsMax = a.BoundsMax,
                AverageNormal = a.NormalSum.sqrMagnitude > 0.000001f
                    ? a.NormalSum.normalized
                    : Vector3.zero,
                BoundaryMask = a.BoundaryMask
            };
            _current[pair.Key] = signature;
            result.CurrentCrossings += a.Count;
        }
        result.CurrentBlocks = _current.Count;

        foreach (KeyValuePair<Vector3Int, Signature> pair in _current)
        {
            if (!_tracks.TryGetValue(pair.Key, out Track track))
            {
                _tracks[pair.Key] = new Track { Last = Clone(pair.Value) };
                _dirty[pair.Key] = ScanCoverIncrementalPatchShadow.PatchDirtyReason.New;
                _dependencyBoundaryMasks[pair.Key] = pair.Value.BoundaryMask;
                result.NewBlocks++;
                result.DirtyBlocks++;
                continue;
            }
            track.MissingRebuilds = 0;
            if (Compatible(track.Last, pair.Value, settings))
                result.CleanBlocks++;
            else
            {
                _dirty[pair.Key] = ScanCoverIncrementalPatchShadow.PatchDirtyReason.GeometryChanged;
                _dependencyBoundaryMasks[pair.Key] =
                    track.Last.BoundaryMask | pair.Value.BoundaryMask;
                result.ChangedBlocks++;
                result.DirtyBlocks++;
            }
            track.Last = Clone(pair.Value);
        }

        List<Vector3Int> retired = null;
        foreach (KeyValuePair<Vector3Int, Track> pair in _tracks)
        {
            if (_current.ContainsKey(pair.Key))
                continue;
            pair.Value.MissingRebuilds++;
            _dirty[pair.Key] = ScanCoverIncrementalPatchShadow.PatchDirtyReason.Missing;
            _dependencyBoundaryMasks[pair.Key] = pair.Value.Last.BoundaryMask;
            result.MissingBlocks++;
            result.DirtyBlocks++;
            if (pair.Value.MissingRebuilds < settings.RetireMissingRebuilds)
                continue;
            if (retired == null)
                retired = new List<Vector3Int>();
            retired.Add(pair.Key);
        }
        if (retired != null)
        {
            for (int i = 0; i < retired.Count; i++)
            {
                _tracks.Remove(retired[i]);
                result.RetiredBlocks++;
            }
        }
        return result;
    }

    public IReadOnlyDictionary<Vector3Int, ScanCoverIncrementalPatchShadow.PatchDirtyReason> DirtyReasons => _dirty;
    public IReadOnlyDictionary<Vector3Int, int> DependencyBoundaryMasks =>
        _dependencyBoundaryMasks;

    public void Clear()
    {
        _tracks.Clear();
        _accumulators.Clear();
        _current.Clear();
        _dirty.Clear();
        _dependencyBoundaryMasks.Clear();
        _rebuildSequence = 0;
    }

    private void AccumulateEdge(
        int ax, int ay, int az,
        int bx, int by, int bz,
        int aIndex, int bIndex,
        float[] tsdf, byte[] weights,
        int dimX, int dimY, int dimZ,
        Vector3 origin, float voxelSize, float patchSize)
    {
        if (!ValidSample(bIndex, tsdf, weights))
            return;
        float a = tsdf[aIndex];
        float b = tsdf[bIndex];
        if ((a > 0f && b > 0f) || (a < 0f && b < 0f) || Mathf.Abs(a - b) < 0.00001f)
            return;
        float t = Mathf.Clamp01(a / (a - b));
        Vector3 pa = origin + new Vector3((ax + 0.5f) * voxelSize, (ay + 0.5f) * voxelSize, (az + 0.5f) * voxelSize);
        Vector3 pb = origin + new Vector3((bx + 0.5f) * voxelSize, (by + 0.5f) * voxelSize, (bz + 0.5f) * voxelSize);
        Vector3 point = Vector3.Lerp(pa, pb, t);
        Vector3Int key = PatchKey(point, origin, patchSize);
        if (!_accumulators.TryGetValue(key, out Accumulator accumulator))
        {
            accumulator = new Accumulator();
            _accumulators[key] = accumulator;
        }
        Vector3 patchMin = origin + new Vector3(
            key.x * patchSize, key.y * patchSize, key.z * patchSize);
        Vector3 patchMax = patchMin + Vector3.one * patchSize;
        float boundaryBand = Mathf.Max(voxelSize * 1.5f, 0.001f);
        if (point.x - patchMin.x <= boundaryBand)
            accumulator.BoundaryMask |= BoundaryNegativeX;
        if (patchMax.x - point.x <= boundaryBand)
            accumulator.BoundaryMask |= BoundaryPositiveX;
        if (point.y - patchMin.y <= boundaryBand)
            accumulator.BoundaryMask |= BoundaryNegativeY;
        if (patchMax.y - point.y <= boundaryBand)
            accumulator.BoundaryMask |= BoundaryPositiveY;
        if (point.z - patchMin.z <= boundaryBand)
            accumulator.BoundaryMask |= BoundaryNegativeZ;
        if (patchMax.z - point.z <= boundaryBand)
            accumulator.BoundaryMask |= BoundaryPositiveZ;
        Vector3 normal = EstimateGradient(ax, ay, az, tsdf, weights, dimX, dimY, dimZ);
        if (normal.sqrMagnitude > 0.000001f)
        {
            normal.Normalize();
            if (accumulator.NormalSum.sqrMagnitude > 0.000001f && Vector3.Dot(normal, accumulator.NormalSum) < 0f)
                normal = -normal;
            accumulator.NormalSum += normal;
        }
        accumulator.Count++;
        accumulator.PositionSum += point;
        if (!accumulator.HasBounds)
        {
            accumulator.BoundsMin = point;
            accumulator.BoundsMax = point;
            accumulator.HasBounds = true;
        }
        else
        {
            accumulator.BoundsMin = Vector3.Min(accumulator.BoundsMin, point);
            accumulator.BoundsMax = Vector3.Max(accumulator.BoundsMax, point);
        }
    }

    private static Vector3 EstimateGradient(int x, int y, int z, float[] tsdf, byte[] weights, int dimX, int dimY, int dimZ)
    {
        float center = Sample(x, y, z, tsdf, weights, dimX, dimY, dimZ, 0f);
        float xm = Sample(x - 1, y, z, tsdf, weights, dimX, dimY, dimZ, center);
        float xp = Sample(x + 1, y, z, tsdf, weights, dimX, dimY, dimZ, center);
        float ym = Sample(x, y - 1, z, tsdf, weights, dimX, dimY, dimZ, center);
        float yp = Sample(x, y + 1, z, tsdf, weights, dimX, dimY, dimZ, center);
        float zm = Sample(x, y, z - 1, tsdf, weights, dimX, dimY, dimZ, center);
        float zp = Sample(x, y, z + 1, tsdf, weights, dimX, dimY, dimZ, center);
        return new Vector3(xp - xm, yp - ym, zp - zm);
    }

    private static float Sample(int x, int y, int z, float[] tsdf, byte[] weights, int dimX, int dimY, int dimZ, float fallback)
    {
        if (x < 0 || y < 0 || z < 0 || x >= dimX || y >= dimY || z >= dimZ)
            return fallback;
        int index = Index(x, y, z, dimX, dimY);
        return ValidSample(index, tsdf, weights) ? tsdf[index] : fallback;
    }

    private static bool Compatible(Signature a, Signature b, Settings settings)
    {
        if (a == null || b == null || a.CrossingCount <= 0 || b.CrossingCount <= 0)
            return false;
        if (a.BoundaryMask != b.BoundaryMask)
            return false;
        float ratio = b.CrossingCount / (float)a.CrossingCount;
        if (ratio < settings.StableCrossingRatio || ratio > 1f / settings.StableCrossingRatio)
            return false;
        if (Vector3.Distance(a.Centroid, b.Centroid) > settings.StableCentroidDistanceMeters)
            return false;
        Vector3 extentDelta = (b.BoundsMax - b.BoundsMin) - (a.BoundsMax - a.BoundsMin);
        if (Mathf.Max(Mathf.Abs(extentDelta.x), Mathf.Abs(extentDelta.y), Mathf.Abs(extentDelta.z)) > settings.StableBoundsDistanceMeters)
            return false;
        if (a.CrossingCount >= settings.MinCrossingsForNormal && b.CrossingCount >= settings.MinCrossingsForNormal &&
            a.AverageNormal.sqrMagnitude > 0.000001f && b.AverageNormal.sqrMagnitude > 0.000001f &&
            Mathf.Abs(Vector3.Dot(a.AverageNormal, b.AverageNormal)) < settings.StableNormalDot)
            return false;
        return true;
    }

    private static Signature Clone(Signature source)
    {
        return new Signature
        {
            CrossingCount = source.CrossingCount,
            Centroid = source.Centroid,
            BoundsMin = source.BoundsMin,
            BoundsMax = source.BoundsMax,
            AverageNormal = source.AverageNormal,
            BoundaryMask = source.BoundaryMask
        };
    }

    private static bool ValidSample(int index, float[] tsdf, byte[] weights)
    {
        return index >= 0 && index < tsdf.Length && index < weights.Length && weights[index] > 0 &&
               !float.IsNaN(tsdf[index]) && !float.IsInfinity(tsdf[index]);
    }

    private static int Index(int x, int y, int z, int dimX, int dimY)
    {
        return x + dimX * (y + dimY * z);
    }

    private static Vector3Int PatchKey(Vector3 point, Vector3 origin, float patchSize)
    {
        Vector3 local = (point - origin) / Mathf.Max(0.001f, patchSize);
        return new Vector3Int(Mathf.FloorToInt(local.x), Mathf.FloorToInt(local.y), Mathf.FloorToInt(local.z));
    }

    private static void Sanitize(ref Settings settings, float voxelSize)
    {
        settings.PatchSizeMeters = Mathf.Clamp(settings.PatchSizeMeters, Mathf.Max(0.05f, voxelSize * 2f), 2f);
        settings.StableCrossingRatio = Mathf.Clamp(settings.StableCrossingRatio, 0.25f, 1f);
        settings.StableCentroidDistanceMeters = Mathf.Clamp(settings.StableCentroidDistanceMeters, 0.002f, 0.25f);
        settings.StableBoundsDistanceMeters = Mathf.Clamp(settings.StableBoundsDistanceMeters, 0.002f, 0.25f);
        settings.StableNormalDot = Mathf.Clamp(settings.StableNormalDot, 0.5f, 1f);
        settings.MinCrossingsForNormal = Mathf.Clamp(settings.MinCrossingsForNormal, 2, 64);
        settings.RetireMissingRebuilds = Mathf.Clamp(settings.RetireMissingRebuilds, 2, 16);
    }
}
