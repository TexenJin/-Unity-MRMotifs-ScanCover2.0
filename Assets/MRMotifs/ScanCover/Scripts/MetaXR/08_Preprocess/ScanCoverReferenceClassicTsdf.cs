using UnityEngine;

/// <summary>
/// Independent classical weighted TSDF used only as a comparison pipeline.
/// It consumes raw Meta depth samples and never reads or writes the production TSDF.
/// </summary>
public sealed class ScanCoverReferenceClassicTsdf
{
    public float[] Tsdf { get; private set; }
    public float[] Weights { get; private set; }
    public int DimX { get; private set; }
    public int DimY { get; private set; }
    public int DimZ { get; private set; }
    public Vector3 Origin { get; private set; }
    public float VoxelSize { get; private set; }

    public int LastInputSamples { get; private set; }
    public int LastValidSamples { get; private set; }
    public int LastInvalidSamples { get; private set; }
    public int LastDepthRejectedSamples { get; private set; }
    public int LastBandVoxelWrites { get; private set; }
    public int LastGrazingSamples { get; private set; }
    public float LastMeanObservationWeight { get; private set; }
    public int WeightedVoxelCount { get; private set; }
    public int PositiveVoxelCount { get; private set; }
    public int NegativeVoxelCount { get; private set; }

    public bool IsCompatible(int dimX, int dimY, int dimZ, Vector3 origin, float voxelSize)
    {
        return Tsdf != null && Weights != null &&
               DimX == dimX && DimY == dimY && DimZ == dimZ &&
               Vector3.SqrMagnitude(Origin - origin) <= 0.00000001f &&
               Mathf.Abs(VoxelSize - voxelSize) <= 0.000001f;
    }

    public void Reset(int dimX, int dimY, int dimZ, Vector3 origin, float voxelSize)
    {
        DimX = Mathf.Max(0, dimX);
        DimY = Mathf.Max(0, dimY);
        DimZ = Mathf.Max(0, dimZ);
        Origin = origin;
        VoxelSize = Mathf.Max(0.0001f, voxelSize);
        int count = DimX * DimY * DimZ;
        Tsdf = new float[count];
        Weights = new float[count];
        for (int i = 0; i < count; i++)
            Tsdf[i] = 1f;
        ResetFrameDiagnostics();
        WeightedVoxelCount = 0;
        PositiveVoxelCount = 0;
        NegativeVoxelCount = 0;
    }

    public void Clear()
    {
        Tsdf = null;
        Weights = null;
        DimX = DimY = DimZ = 0;
        ResetFrameDiagnostics();
        WeightedVoxelCount = PositiveVoxelCount = NegativeVoxelCount = 0;
    }

    public bool IntegrateFrame(
        Vector3[] positions,
        Vector3[] normals,
        Color[] observationMeta,
        int width,
        int height,
        int stride,
        Vector3 cameraPosition,
        float minDepth,
        float maxDepth,
        float truncation,
        float maximumWeight,
        float minimumAngleWeight,
        float minimumDistanceWeight)
    {
        ResetFrameDiagnostics();
        if (Tsdf == null || Weights == null || positions == null || width <= 1 || height <= 1)
            return false;

        int expected = Mathf.Min(width * height, positions.Length);
        int pixelStride = Mathf.Max(1, stride);
        double observationWeightSum = 0.0;
        for (int y = 0; y < height; y += pixelStride)
        for (int x = 0; x < width; x += pixelStride)
        {
            int index = y * width + x;
            if (index < 0 || index >= expected)
                continue;
            LastInputSamples++;
            Vector3 point = positions[index];
            if (!Finite(point))
            {
                LastInvalidSamples++;
                continue;
            }

            Vector3 toSurface = point - cameraPosition;
            float depth = toSurface.magnitude;
            if (!Finite(toSurface) || depth < minDepth || depth > maxDepth || depth <= 0.0001f)
            {
                LastDepthRejectedSamples++;
                continue;
            }

            Vector3 rayDirection = toSurface / depth;
            float confidence = observationMeta != null && index < observationMeta.Length
                ? Mathf.Clamp01(observationMeta[index].g)
                : 1f;
            if (confidence <= 0.001f)
            {
                LastInvalidSamples++;
                continue;
            }

            float facing = 1f;
            if (normals != null && index < normals.Length && Finite(normals[index]) &&
                normals[index].sqrMagnitude > 0.000001f)
            {
                facing = Mathf.Abs(Vector3.Dot(normals[index].normalized, -rayDirection));
            }
            if (facing < 0.35f)
                LastGrazingSamples++;

            // Angle and range are confidence terms, never hard rejection gates.
            float angleWeight = Mathf.Lerp(Mathf.Clamp01(minimumAngleWeight), 1f, facing * facing);
            float rangeT = Mathf.InverseLerp(minDepth, Mathf.Max(minDepth + 0.001f, maxDepth), depth);
            float distanceWeight = Mathf.Lerp(1f, Mathf.Clamp01(minimumDistanceWeight), rangeT * rangeT);
            float observationWeight = Mathf.Max(0.001f, confidence * angleWeight * distanceWeight);
            LastValidSamples++;
            observationWeightSum += observationWeight;
            IntegrateBand(cameraPosition, rayDirection, depth, observationWeight,
                Mathf.Max(VoxelSize, truncation), Mathf.Max(1f, maximumWeight));
        }

        LastMeanObservationWeight = LastValidSamples > 0
            ? (float)(observationWeightSum / LastValidSamples)
            : 0f;
        return LastValidSamples > 0 && LastBandVoxelWrites > 0;
    }

    public void AuditVolume()
    {
        WeightedVoxelCount = PositiveVoxelCount = NegativeVoxelCount = 0;
        if (Tsdf == null || Weights == null)
            return;
        for (int i = 0; i < Weights.Length; i++)
        {
            if (Weights[i] <= 0.0001f)
                continue;
            WeightedVoxelCount++;
            if (Tsdf[i] < 0f) NegativeVoxelCount++;
            else PositiveVoxelCount++;
        }
    }

    private void IntegrateBand(Vector3 camera, Vector3 ray, float surfaceDepth, float observationWeight,
        float truncation, float maximumWeight)
    {
        float startDepth = Mathf.Max(0.01f, surfaceDepth - truncation);
        float endDepth = surfaceDepth + truncation;
        Vector3 segmentStart = camera + ray * startDepth;
        Vector3 segmentEnd = camera + ray * endDepth;
        float radius = VoxelSize;
        Vector3 minimumGrid = (Vector3.Min(segmentStart, segmentEnd) - Vector3.one * radius - Origin) / VoxelSize;
        Vector3 maximumGrid = (Vector3.Max(segmentStart, segmentEnd) + Vector3.one * radius - Origin) / VoxelSize;
        int minX = Mathf.Clamp(Mathf.FloorToInt(minimumGrid.x), 0, DimX - 1);
        int minY = Mathf.Clamp(Mathf.FloorToInt(minimumGrid.y), 0, DimY - 1);
        int minZ = Mathf.Clamp(Mathf.FloorToInt(minimumGrid.z), 0, DimZ - 1);
        int maxX = Mathf.Clamp(Mathf.CeilToInt(maximumGrid.x), 0, DimX - 1);
        int maxY = Mathf.Clamp(Mathf.CeilToInt(maximumGrid.y), 0, DimY - 1);
        int maxZ = Mathf.Clamp(Mathf.CeilToInt(maximumGrid.z), 0, DimZ - 1);
        float radiusSq = radius * radius;

        for (int z = minZ; z <= maxZ; z++)
        for (int y = minY; y <= maxY; y++)
        for (int x = minX; x <= maxX; x++)
        {
            Vector3 center = Origin + new Vector3(x * VoxelSize, y * VoxelSize, z * VoxelSize);
            float centerDepth = Vector3.Dot(center - camera, ray);
            if (centerDepth < startDepth - VoxelSize * 0.5f || centerDepth > endDepth + VoxelSize * 0.5f)
                continue;
            Vector3 projected = camera + ray * centerDepth;
            if ((center - projected).sqrMagnitude > radiusSq)
                continue;

            int voxelIndex = x + DimX * (y + DimY * z);
            float sampleTsdf = Mathf.Clamp((surfaceDepth - centerDepth) / truncation, -1f, 1f);
            float oldWeight = Weights[voxelIndex];
            float appliedWeight = Mathf.Min(observationWeight, Mathf.Max(0f, maximumWeight - oldWeight));
            if (appliedWeight <= 0.000001f)
                continue;
            float newWeight = oldWeight + appliedWeight;
            Tsdf[voxelIndex] = (Tsdf[voxelIndex] * oldWeight + sampleTsdf * appliedWeight) / newWeight;
            Weights[voxelIndex] = newWeight;
            LastBandVoxelWrites++;
        }
    }

    private void ResetFrameDiagnostics()
    {
        LastInputSamples = LastValidSamples = LastInvalidSamples = LastDepthRejectedSamples = 0;
        LastBandVoxelWrites = LastGrazingSamples = 0;
        LastMeanObservationWeight = 0f;
    }

    private static bool Finite(Vector3 value)
    {
        return !(float.IsNaN(value.x) || float.IsInfinity(value.x) ||
                 float.IsNaN(value.y) || float.IsInfinity(value.y) ||
                 float.IsNaN(value.z) || float.IsInfinity(value.z));
    }
}
