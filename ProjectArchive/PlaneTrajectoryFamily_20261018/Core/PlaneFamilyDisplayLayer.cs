using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Read-only plane-family labelling for already-published mesh fronts.
    /// Mature court planes seed labels, then mesh connectivity grows them over
    /// coplanar triangles. This module owns only display buffers and counters;
    /// disabling or disposing it leaves TSDF, extraction and publication intact.
    /// </summary>
    internal sealed class PlaneFamilyDisplayLayer : IDisposable
    {
        private const int MaximumSeeds = 512;
        private const int PropagationPasses = 32;

        private sealed class SeedFamilyComparer :
            IComparer<RuntimeFinalSurfaceCourt.ProductPlane>
        {
            public int Compare(RuntimeFinalSurfaceCourt.ProductPlane x,
                RuntimeFinalSurfaceCourt.ProductPlane y) =>
                x.FamilyId.CompareTo(y.FamilyId);
        }

        private static readonly SeedFamilyComparer FamilyComparer =
            new SeedFamilyComparer();

        private readonly Transform _space;
        private readonly VolumeIntegrator _volume;
        private readonly ComputeShader _compute;
        private readonly int _prepareKernel;
        private readonly int _seedKernel;
        private readonly int _propagateKernel;
        private readonly int _resolveKernel;
        private readonly int _countKernel;
        private readonly RuntimeFinalSurfaceCourt.ProductPlane[] _seeds =
            new RuntimeFinalSurfaceCourt.ProductPlane[MaximumSeeds];
        private readonly Vector4[] _planes = new Vector4[MaximumSeeds];
        private readonly Vector4[] _centers = new Vector4[MaximumSeeds];
        private readonly uint[] _familyIds = new uint[MaximumSeeds];
        private GraphicsBuffer _planeBuffer;
        private GraphicsBuffer _centerBuffer;
        private GraphicsBuffer _familyIdBuffer;
        private GraphicsBuffer _scratchLabels;

        private static readonly int VerticesId = Shader.PropertyToID("_Vertices");
        private static readonly int IndicesId = Shader.PropertyToID("_Indices");
        private static readonly int FamilyPlanesId = Shader.PropertyToID("_FamilyPlanes");
        private static readonly int FamilyCentersId = Shader.PropertyToID("_FamilyCenters");
        private static readonly int FamilyIdsId = Shader.PropertyToID("_FamilyIds");
        private static readonly int ScratchLabelsId = Shader.PropertyToID("_ScratchLabels");
        private static readonly int OutputFamiliesId = Shader.PropertyToID("_OutputFamilies");
        private static readonly int StatsId = Shader.PropertyToID("_Stats");
        private static readonly int VertexCountId = Shader.PropertyToID("_VertexCount");
        private static readonly int IndexCountId = Shader.PropertyToID("_IndexCount");
        private static readonly int FamilySeedCountId = Shader.PropertyToID("_FamilySeedCount");
        private static readonly int PreservePreviousId = Shader.PropertyToID("_PreservePrevious");
        private static readonly int LocalToWorldId = Shader.PropertyToID("_LocalToWorld");
        private static readonly int WorldToLocalId = Shader.PropertyToID("_WorldToLocal");
        private static readonly int EnterResidualId = Shader.PropertyToID("_EnterResidual");
        private static readonly int ExitResidualId = Shader.PropertyToID("_ExitResidual");
        private static readonly int MinimumNormalDotId = Shader.PropertyToID("_MinimumNormalDot");
        private static readonly int MaximumEdgeLengthId = Shader.PropertyToID("_MaximumEdgeLength");

        internal bool Available => _compute != null;

        internal PlaneFamilyDisplayLayer(Transform space,
            VolumeIntegrator volume)
        {
            _space = space;
            _volume = volume;
            _compute = Resources.Load<ComputeShader>("PlaneFamilyDisplay");
            if (_compute == null ||
                !_compute.HasKernel("PrepareLabels") ||
                !_compute.HasKernel("SeedLabels") ||
                !_compute.HasKernel("PropagateLabels") ||
                !_compute.HasKernel("ResolveFamilies") ||
                !_compute.HasKernel("CountTriangles"))
            {
                _compute = null;
                return;
            }

            _prepareKernel = _compute.FindKernel("PrepareLabels");
            _seedKernel = _compute.FindKernel("SeedLabels");
            _propagateKernel = _compute.FindKernel("PropagateLabels");
            _resolveKernel = _compute.FindKernel("ResolveFamilies");
            _countKernel = _compute.FindKernel("CountTriangles");
            _planeBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                MaximumSeeds, sizeof(float) * 4);
            _centerBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                MaximumSeeds, sizeof(float) * 4);
            _familyIdBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                MaximumSeeds, sizeof(uint));
        }

        internal void Relabel(GPUChunkMeshSnapshot snapshot,
            Bounds localBounds, float voxelSize)
        {
            if (_compute == null || _volume == null || snapshot == null ||
                snapshot.VertexBuffer == null || snapshot.IndexBuffer == null ||
                snapshot.PlaneFamilyBuffer == null ||
                snapshot.PlaneFamilyStatsBuffer == null)
                return;

            int vertexCount = Mathf.Max(0, snapshot.KnownVertexCount);
            int indexCount = Mathf.Max(0, snapshot.KnownDrawVertexCount);
            if (vertexCount == 0 || indexCount < 3)
                return;

            EnsureScratch(vertexCount);
            Matrix4x4 localToWorld = _space != null
                ? _space.localToWorldMatrix : Matrix4x4.identity;
            Matrix4x4 worldToLocal = _space != null
                ? _space.worldToLocalMatrix : Matrix4x4.identity;
            Bounds worldBounds = TransformBounds(localBounds, localToWorld);
            int seedCount = Mathf.Min(MaximumSeeds,
                _volume.CopyProductSurfaceFamilySeeds(worldBounds, _seeds));
            // Label indices inherit this order. The compute flood may then use
            // the smaller index as the deterministic winner when two coplanar
            // families meet through actual mesh connectivity.
            Array.Sort(_seeds, 0, seedCount, FamilyComparer);
            for (int i = 0; i < seedCount; i++)
            {
                _planes[i] = _seeds[i].Equation;
                _centers[i] = _seeds[i].CenterRadius;
                _familyIds[i] = _seeds[i].FamilyId;
            }
            if (seedCount > 0)
            {
                _planeBuffer.SetData(_planes, 0, 0, seedCount);
                _centerBuffer.SetData(_centers, 0, 0, seedCount);
                _familyIdBuffer.SetData(_familyIds, 0, 0, seedCount);
            }

            _compute.SetInt(VertexCountId, vertexCount);
            _compute.SetInt(IndexCountId, indexCount);
            _compute.SetInt(FamilySeedCountId, seedCount);
            _compute.SetInt(PreservePreviousId,
                snapshot.PlaneFamilyInitialized ? 1 : 0);
            _compute.SetMatrix(LocalToWorldId, localToWorld);
            _compute.SetMatrix(WorldToLocalId, worldToLocal);
            _compute.SetFloat(EnterResidualId, 0.05f);
            _compute.SetFloat(ExitResidualId, 0.075f);
            _compute.SetFloat(MinimumNormalDotId, 0.9510565f); // cos(18 degrees)
            _compute.SetFloat(MaximumEdgeLengthId,
                Mathf.Max(0.15f, voxelSize * 2.5f));

            Bind(_prepareKernel, snapshot);
            Bind(_seedKernel, snapshot);
            Bind(_propagateKernel, snapshot);
            Bind(_resolveKernel, snapshot);
            Bind(_countKernel, snapshot);

            int vertexGroups = CeilDiv(vertexCount, 64);
            int triangleGroups = CeilDiv(indexCount / 3, 64);
            _compute.Dispatch(_prepareKernel, vertexGroups, 1, 1);
            _compute.Dispatch(_seedKernel, vertexGroups, 1, 1);
            for (int pass = 0; pass < PropagationPasses; pass++)
                _compute.Dispatch(_propagateKernel, triangleGroups, 1, 1);
            _compute.Dispatch(_resolveKernel, vertexGroups, 1, 1);

            int statsGeneration = snapshot.BeginPlaneFamilyStatsUpdate();
            _compute.Dispatch(_countKernel, triangleGroups, 1, 1);
            snapshot.MarkPlaneFamilyInitialized();
            AsyncGPUReadback.Request(snapshot.PlaneFamilyStatsBuffer, request =>
            {
                if (request.hasError)
                    return;
                var values = request.GetData<uint>();
                if (values.Length < 3)
                    return;
                snapshot.CompletePlaneFamilyStatsUpdate(statsGeneration,
                    values[0], values[1], values[2]);
            });
        }

        private void Bind(int kernel, GPUChunkMeshSnapshot snapshot)
        {
            _compute.SetBuffer(kernel, VerticesId, snapshot.VertexBuffer);
            _compute.SetBuffer(kernel, IndicesId, snapshot.IndexBuffer);
            _compute.SetBuffer(kernel, FamilyPlanesId, _planeBuffer);
            _compute.SetBuffer(kernel, FamilyCentersId, _centerBuffer);
            _compute.SetBuffer(kernel, FamilyIdsId, _familyIdBuffer);
            _compute.SetBuffer(kernel, ScratchLabelsId, _scratchLabels);
            _compute.SetBuffer(kernel, OutputFamiliesId,
                snapshot.PlaneFamilyBuffer);
            _compute.SetBuffer(kernel, StatsId,
                snapshot.PlaneFamilyStatsBuffer);
        }

        private void EnsureScratch(int vertexCount)
        {
            if (_scratchLabels != null && _scratchLabels.count >= vertexCount)
                return;
            _scratchLabels?.Release();
            _scratchLabels = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                Mathf.Max(1, vertexCount), sizeof(uint));
        }

        private static Bounds TransformBounds(Bounds local, Matrix4x4 matrix)
        {
            Vector3 center = matrix.MultiplyPoint3x4(local.center);
            Vector3 ext = local.extents;
            Vector3 x = matrix.MultiplyVector(new Vector3(ext.x, 0f, 0f));
            Vector3 y = matrix.MultiplyVector(new Vector3(0f, ext.y, 0f));
            Vector3 z = matrix.MultiplyVector(new Vector3(0f, 0f, ext.z));
            Vector3 worldExtents = new Vector3(
                Mathf.Abs(x.x) + Mathf.Abs(y.x) + Mathf.Abs(z.x),
                Mathf.Abs(x.y) + Mathf.Abs(y.y) + Mathf.Abs(z.y),
                Mathf.Abs(x.z) + Mathf.Abs(y.z) + Mathf.Abs(z.z));
            return new Bounds(center, worldExtents * 2f);
        }

        private static int CeilDiv(int value, int divisor) =>
            Mathf.Max(1, (value + divisor - 1) / divisor);

        public void Dispose()
        {
            _planeBuffer?.Release();
            _centerBuffer?.Release();
            _familyIdBuffer?.Release();
            _scratchLabels?.Release();
            _planeBuffer = null;
            _centerBuffer = null;
            _familyIdBuffer = null;
            _scratchLabels = null;
        }
    }
}
