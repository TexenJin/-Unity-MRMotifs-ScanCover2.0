using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Builds a disposable, plane-fitted completion shadow from a measured TSDF.
/// Synthetic voxels never become evidence for other synthetic voxels.
/// </summary>
public static class ScanCoverReferenceTsdfGapCompleter
{
    public sealed class Result
    {
        public sealed class Candidate
        {
            public int Index;
            public int X;
            public int Y;
            public int Z;
            public float Tsdf;
            public float Residual;
            public Vector3 PlaneNormal;
            public Vector3 PlanePoint;
            public float PlaneOffset;
            public bool PlaneValid;
        }

        public sealed class Patch
        {
            public readonly List<Candidate> Voxels = new List<Candidate>(32);
            public float MeanResidual;
            public Vector3 PlaneNormal;
            public Vector3 PlanePoint;
            public float PlaneOffset;
            public bool PlaneValid;
        }

        public float[] Tsdf;
        public float[] Weights;
        public readonly List<Patch> Patches = new List<Patch>(32);
        public int CandidateVoxels;
        public int PredictedVoxels;
        public int FilledVoxels;
        public int BlockedSampleCount;
        public int BlockedSignSupport;
        public int BlockedFit;
        public int BlockedResidual;
        public int BlockedPrediction;
        public int InvalidPlaneCandidates;
        public int PatchAdjacencyChecks;
        public int PatchNormalRejected;
        public int PatchDistanceRejected;
        public int PatchJoined;
        public float MeanResidual;
    }

    public static Result Build(
        float[] measuredTsdf, float[] measuredWeights,
        int dimX, int dimY, int dimZ,
        float minimumMeasuredWeight,
        float completionWeight,
        int radius,
        int minimumSamples,
        float maximumResidual,
        float maximumPredictionMagnitude,
        int maximumFilledVoxels,
        float minimumPatchNormalDot,
        float maximumPatchPlaneDistanceVoxels)
    {
        Result result = new Result();
        if (measuredTsdf == null || measuredWeights == null || measuredTsdf.Length != measuredWeights.Length ||
            measuredTsdf.Length < dimX * dimY * dimZ)
            return result;

        result.Tsdf = (float[])measuredTsdf.Clone();
        result.Weights = (float[])measuredWeights.Clone();
        int r = Mathf.Clamp(radius, 1, 3);
        int minSamples = Mathf.Max(6, minimumSamples);
        float minWeight = Mathf.Max(0.0001f, minimumMeasuredWeight);
        int fillLimit = Mathf.Max(1, maximumFilledVoxels);
        FindObservedBounds(measuredWeights, dimX, dimY, dimZ, minWeight,
            out Vector3Int minimum, out Vector3Int maximum);
        if (maximum.x < minimum.x)
            return result;

        double residualSum = 0.0;
        List<Result.Candidate> predictedCandidates = new List<Result.Candidate>(1024);
        double[] normal = new double[16];
        double[] rhs = new double[4];
        double[] coefficients = new double[4];
        double[] augmented = new double[20];
        for (int z = Mathf.Max(1, minimum.z); z <= Mathf.Min(dimZ - 2, maximum.z); z++)
        for (int y = Mathf.Max(1, minimum.y); y <= Mathf.Min(dimY - 2, maximum.y); y++)
        for (int x = Mathf.Max(1, minimum.x); x <= Mathf.Min(dimX - 2, maximum.x); x++)
        {
            int target = Index(x, y, z, dimX, dimY);
            if (measuredWeights[target] > 0.0001f || !TouchesMeasuredNearSurface(
                    x, y, z, measuredTsdf, measuredWeights, dimX, dimY, dimZ, minWeight))
                continue;
            result.CandidateVoxels++;
            if (result.PredictedVoxels >= fillLimit)
                continue;

            System.Array.Clear(normal, 0, normal.Length);
            System.Array.Clear(rhs, 0, rhs.Length);
            int samples = 0;
            int positive = 0;
            int negative = 0;
            for (int nz = Mathf.Max(0, z - r); nz <= Mathf.Min(dimZ - 1, z + r); nz++)
            for (int ny = Mathf.Max(0, y - r); ny <= Mathf.Min(dimY - 1, y + r); ny++)
            for (int nx = Mathf.Max(0, x - r); nx <= Mathf.Min(dimX - 1, x + r); nx++)
            {
                int source = Index(nx, ny, nz, dimX, dimY);
                float weight = measuredWeights[source];
                float value = measuredTsdf[source];
                if (weight < minWeight || !Finite(value) || Mathf.Abs(value) > 0.9f)
                    continue;
                double w = Mathf.Min(4f, weight);
                double b0 = nx - x;
                double b1 = ny - y;
                double b2 = nz - z;
                for (int row = 0; row < 4; row++)
                {
                    double rowBasis = Basis(row, b0, b1, b2);
                    rhs[row] += w * rowBasis * value;
                    for (int column = 0; column < 4; column++)
                        normal[row * 4 + column] += w * rowBasis * Basis(column, b0, b1, b2);
                }
                samples++;
                if (value > 0.03f) positive++;
                else if (value < -0.03f) negative++;
            }

            if (samples < minSamples)
            {
                result.BlockedSampleCount++;
                continue;
            }
            if (positive < 2 || negative < 2)
            {
                result.BlockedSignSupport++;
                continue;
            }
            if (!Solve4x4(normal, rhs, augmented, coefficients))
            {
                result.BlockedFit++;
                continue;
            }

            double weightedError = 0.0;
            double weightSum = 0.0;
            for (int nz = Mathf.Max(0, z - r); nz <= Mathf.Min(dimZ - 1, z + r); nz++)
            for (int ny = Mathf.Max(0, y - r); ny <= Mathf.Min(dimY - 1, y + r); ny++)
            for (int nx = Mathf.Max(0, x - r); nx <= Mathf.Min(dimX - 1, x + r); nx++)
            {
                int source = Index(nx, ny, nz, dimX, dimY);
                float weight = measuredWeights[source];
                float value = measuredTsdf[source];
                if (weight < minWeight || !Finite(value) || Mathf.Abs(value) > 0.9f)
                    continue;
                double w = Mathf.Min(4f, weight);
                double prediction = coefficients[0] * (nx - x) + coefficients[1] * (ny - y) +
                                    coefficients[2] * (nz - z) + coefficients[3];
                double error = prediction - value;
                weightedError += w * error * error;
                weightSum += w;
            }
            float residual = weightSum > 0.0 ? (float)System.Math.Sqrt(weightedError / weightSum) : float.MaxValue;
            if (!Finite(residual) || residual > maximumResidual)
            {
                result.BlockedResidual++;
                continue;
            }
            float predicted = (float)coefficients[3];
            if (!Finite(predicted) || Mathf.Abs(predicted) > maximumPredictionMagnitude)
            {
                result.BlockedPrediction++;
                continue;
            }

            Vector3 gradient = new Vector3(
                (float)coefficients[0],
                (float)coefficients[1],
                (float)coefficients[2]);
            float gradientMagnitude = gradient.magnitude;
            bool planeValid = Finite(gradientMagnitude) && gradientMagnitude > 0.00001f;
            Vector3 planeNormal = planeValid ? gradient / gradientMagnitude : Vector3.zero;
            Vector3 voxelCenter = new Vector3(x, y, z);
            Vector3 planePoint = planeValid
                ? voxelCenter - planeNormal * (predicted / gradientMagnitude)
                : voxelCenter;
            float planeOffset = planeValid ? -Vector3.Dot(planeNormal, planePoint) : 0f;
            if (!planeValid || !Finite(planePoint) || !Finite(planeOffset))
            {
                planeValid = false;
                result.InvalidPlaneCandidates++;
            }

            predictedCandidates.Add(new Result.Candidate
            {
                Index = target,
                X = x,
                Y = y,
                Z = z,
                Tsdf = Mathf.Clamp(predicted, -1f, 1f),
                Residual = residual,
                PlaneNormal = planeNormal,
                PlanePoint = planePoint,
                PlaneOffset = planeOffset,
                PlaneValid = planeValid
            });
            result.PredictedVoxels++;
            residualSum += residual;
        }
        BuildPatches(
            predictedCandidates,
            result,
            dimX,
            dimY,
            Mathf.Clamp(minimumPatchNormalDot, 0.5f, 1f),
            Mathf.Max(0.01f, maximumPatchPlaneDistanceVoxels));
        result.MeanResidual = result.PredictedVoxels > 0 ? (float)(residualSum / result.PredictedVoxels) : 0f;
        return result;
    }

    public static void ApplyPatch(Result result, Result.Patch patch, float completionWeight, float minimumWeight)
    {
        if (result == null || patch == null || result.Tsdf == null || result.Weights == null)
            return;
        float weight = Mathf.Max(completionWeight, minimumWeight);
        for (int i = 0; i < patch.Voxels.Count; i++)
        {
            Result.Candidate voxel = patch.Voxels[i];
            result.Tsdf[voxel.Index] = voxel.Tsdf;
            result.Weights[voxel.Index] = weight;
        }
    }

    public static void RevertPatch(Result result, Result.Patch patch, float[] measuredTsdf, float[] measuredWeights)
    {
        if (result == null || patch == null || measuredTsdf == null || measuredWeights == null)
            return;
        for (int i = 0; i < patch.Voxels.Count; i++)
        {
            int index = patch.Voxels[i].Index;
            result.Tsdf[index] = measuredTsdf[index];
            result.Weights[index] = measuredWeights[index];
        }
    }

    private static void BuildPatches(
        List<Result.Candidate> candidates,
        Result result,
        int dimX,
        int dimY,
        float minimumNormalDot,
        float maximumPlaneDistanceVoxels)
    {
        Dictionary<int, Result.Candidate> byIndex = new Dictionary<int, Result.Candidate>(candidates.Count);
        for (int i = 0; i < candidates.Count; i++) byIndex[candidates[i].Index] = candidates[i];
        HashSet<int> visited = new HashSet<int>();
        Queue<Result.Candidate> queue = new Queue<Result.Candidate>();
        int[] offsets = { -1, 1, -dimX, dimX, -dimX * dimY, dimX * dimY };
        for (int i = 0; i < candidates.Count; i++)
        {
            Result.Candidate seed = candidates[i];
            if (!visited.Add(seed.Index)) continue;
            Result.Patch patch = new Result.Patch();
            double residualSum = 0.0;
            queue.Enqueue(seed);
            while (queue.Count > 0)
            {
                Result.Candidate current = queue.Dequeue();
                patch.Voxels.Add(current);
                residualSum += current.Residual;
                for (int direction = 0; direction < offsets.Length; direction++)
                {
                    int neighborIndex = current.Index + offsets[direction];
                    if (!byIndex.TryGetValue(neighborIndex, out Result.Candidate neighbor)) continue;
                    int manhattan = Mathf.Abs(neighbor.X - current.X) + Mathf.Abs(neighbor.Y - current.Y) + Mathf.Abs(neighbor.Z - current.Z);
                    if (manhattan != 1 || visited.Contains(neighborIndex)) continue;
                    result.PatchAdjacencyChecks++;
                    if (!PlanesBelongToSamePatch(current, neighbor, minimumNormalDot, maximumPlaneDistanceVoxels, out bool normalRejected))
                    {
                        if (normalRejected) result.PatchNormalRejected++;
                        else result.PatchDistanceRejected++;
                        continue;
                    }
                    if (!ReferenceEquals(current, seed) &&
                        !PlanesBelongToSamePatch(seed, neighbor, minimumNormalDot, maximumPlaneDistanceVoxels, out normalRejected))
                    {
                        if (normalRejected) result.PatchNormalRejected++;
                        else result.PatchDistanceRejected++;
                        continue;
                    }
                    visited.Add(neighborIndex);
                    result.PatchJoined++;
                    queue.Enqueue(neighbor);
                }
            }
            patch.MeanResidual = patch.Voxels.Count > 0 ? (float)(residualSum / patch.Voxels.Count) : 0f;
            patch.PlaneNormal = seed.PlaneNormal;
            patch.PlanePoint = seed.PlanePoint;
            patch.PlaneOffset = seed.PlaneOffset;
            patch.PlaneValid = seed.PlaneValid;
            result.Patches.Add(patch);
        }
        result.Patches.Sort((a, b) => b.Voxels.Count.CompareTo(a.Voxels.Count));
    }

    private static bool PlanesBelongToSamePatch(
        Result.Candidate a,
        Result.Candidate b,
        float minimumNormalDot,
        float maximumPlaneDistanceVoxels,
        out bool normalRejected)
    {
        normalRejected = false;
        if (a == null || b == null || !a.PlaneValid || !b.PlaneValid)
        {
            normalRejected = true;
            return false;
        }

        float signedNormalDot = Vector3.Dot(a.PlaneNormal, b.PlaneNormal);
        if (Mathf.Abs(signedNormalDot) < minimumNormalDot)
        {
            normalRejected = true;
            return false;
        }

        Vector3 bNormal = b.PlaneNormal;
        float bOffset = b.PlaneOffset;
        if (signedNormalDot < 0f)
        {
            bNormal = -bNormal;
            bOffset = -bOffset;
        }

        float aToB = Mathf.Abs(Vector3.Dot(a.PlaneNormal, b.PlanePoint) + a.PlaneOffset);
        float bToA = Mathf.Abs(Vector3.Dot(bNormal, a.PlanePoint) + bOffset);
        return Finite(aToB) && Finite(bToA) &&
               aToB <= maximumPlaneDistanceVoxels &&
               bToA <= maximumPlaneDistanceVoxels;
    }

    private static bool TouchesMeasuredNearSurface(int x, int y, int z,
        float[] tsdf, float[] weights, int dimX, int dimY, int dimZ, float minimumWeight)
    {
        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
        {
            if (dx == 0 && dy == 0 && dz == 0) continue;
            int index = Index(x + dx, y + dy, z + dz, dimX, dimY);
            if (weights[index] >= minimumWeight && Mathf.Abs(tsdf[index]) <= 0.75f)
                return true;
        }
        return false;
    }

    private static double Basis(int component, double x, double y, double z)
    {
        if (component == 0) return x;
        if (component == 1) return y;
        if (component == 2) return z;
        return 1.0;
    }

    private static bool Solve4x4(double[] matrix, double[] rhs, double[] augmented, double[] solution)
    {
        System.Array.Clear(augmented, 0, augmented.Length);
        for (int row = 0; row < 4; row++)
        {
            for (int column = 0; column < 4; column++) augmented[row * 5 + column] = matrix[row * 4 + column];
            augmented[row * 5 + 4] = rhs[row];
        }
        for (int pivot = 0; pivot < 4; pivot++)
        {
            int best = pivot;
            for (int row = pivot + 1; row < 4; row++)
                if (System.Math.Abs(augmented[row * 5 + pivot]) > System.Math.Abs(augmented[best * 5 + pivot])) best = row;
            if (System.Math.Abs(augmented[best * 5 + pivot]) < 0.00000001) return false;
            if (best != pivot)
                for (int column = pivot; column < 5; column++)
                {
                    double swap = augmented[pivot * 5 + column];
                    augmented[pivot * 5 + column] = augmented[best * 5 + column];
                    augmented[best * 5 + column] = swap;
                }
            double divisor = augmented[pivot * 5 + pivot];
            for (int column = pivot; column < 5; column++) augmented[pivot * 5 + column] /= divisor;
            for (int row = 0; row < 4; row++)
            {
                if (row == pivot) continue;
                double factor = augmented[row * 5 + pivot];
                for (int column = pivot; column < 5; column++)
                    augmented[row * 5 + column] -= factor * augmented[pivot * 5 + column];
            }
        }
        for (int row = 0; row < 4; row++) solution[row] = augmented[row * 5 + 4];
        return true;
    }

    private static void FindObservedBounds(float[] weights, int dimX, int dimY, int dimZ, float minimumWeight,
        out Vector3Int minimum, out Vector3Int maximum)
    {
        minimum = new Vector3Int(dimX, dimY, dimZ);
        maximum = new Vector3Int(-1, -1, -1);
        int plane = dimX * dimY;
        for (int index = 0; index < weights.Length; index++)
        {
            if (weights[index] < minimumWeight) continue;
            int z = index / plane;
            int remainder = index - z * plane;
            int y = remainder / dimX;
            int x = remainder - y * dimX;
            minimum = Vector3Int.Min(minimum, new Vector3Int(x, y, z));
            maximum = Vector3Int.Max(maximum, new Vector3Int(x, y, z));
        }
    }

    private static int Index(int x, int y, int z, int dimX, int dimY)
    {
        return x + dimX * (y + dimY * z);
    }

    private static bool Finite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }

    private static bool Finite(Vector3 value)
    {
        return Finite(value.x) && Finite(value.y) && Finite(value.z);
    }
}
