using System;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Converts one raw 5 cm Surface-Nets candidate into a delivered chunk.
    /// It owns no history volume and never mutates TSDF.  With no approved
    /// product planes it is deliberately a byte-for-byte pass-through, making
    /// the insertion point independently testable before policy is enabled.
    /// </summary>
    internal sealed class SurfaceProductizer : IDisposable
    {
        private const int VertexStride = 32;
        private const int MaximumLocalPlanes = 32;

        private readonly Transform _space;
        private readonly VolumeIntegrator _volume;
        private readonly ComputeShader _compute;
        private readonly int _copyVerticesKernel;
        private readonly int _copyIndicesKernel;
        private readonly RuntimeFinalSurfaceCourt.ProductPlane[] _productPlanes =
            new RuntimeFinalSurfaceCourt.ProductPlane[MaximumLocalPlanes];
        private readonly RuntimeFinalSurfaceCourt.ProductPlane[] _courtPlanes =
            new RuntimeFinalSurfaceCourt.ProductPlane[MaximumLocalPlanes];
        private readonly Vector4[] _planeEquations = new Vector4[MaximumLocalPlanes];
        private readonly Vector4[] _planeCenters = new Vector4[MaximumLocalPlanes];
        private readonly Vector4[] _planeAxesU = new Vector4[MaximumLocalPlanes];
        private readonly Vector4[] _planeAxesV = new Vector4[MaximumLocalPlanes];
        private GraphicsBuffer _equationsBuffer;
        private GraphicsBuffer _centersBuffer;
        private GraphicsBuffer _axesUBuffer;
        private GraphicsBuffer _axesVBuffer;

        private static readonly int InputVerticesId = Shader.PropertyToID("_InputVertices");
        private static readonly int InputIndicesId = Shader.PropertyToID("_InputIndices");
        private static readonly int InputAdmissionId = Shader.PropertyToID("_InputAdmission");
        private static readonly int OutputVerticesId = Shader.PropertyToID("_OutputVertices");
        private static readonly int OutputIndicesId = Shader.PropertyToID("_OutputIndices");
        private static readonly int OutputAdmissionId = Shader.PropertyToID("_OutputAdmission");
        private static readonly int ProductPlanesId = Shader.PropertyToID("_ProductPlanes");
        private static readonly int ProductPlaneCentersId = Shader.PropertyToID("_ProductPlaneCenters");
        private static readonly int ProductPlaneAxesUId = Shader.PropertyToID("_ProductPlaneAxesU");
        private static readonly int ProductPlaneAxesVId = Shader.PropertyToID("_ProductPlaneAxesV");
        private static readonly int ProductPlaneCountId = Shader.PropertyToID("_ProductPlaneCount");
        private static readonly int VertexCountId = Shader.PropertyToID("_VertexCount");
        private static readonly int IndexCountId = Shader.PropertyToID("_IndexCount");
        private static readonly int LocalToWorldId = Shader.PropertyToID("_LocalToWorld");
        private static readonly int WorldToLocalId = Shader.PropertyToID("_WorldToLocal");
        private static readonly int CoreMinId = Shader.PropertyToID("_CoreMin");
        private static readonly int CoreMaxId = Shader.PropertyToID("_CoreMax");
        private static readonly int EdgeBandId = Shader.PropertyToID("_EdgeBand");
        private static readonly int MaximumProjectionId = Shader.PropertyToID("_MaximumProjection");
        private static readonly int MinimumNormalDotId = Shader.PropertyToID("_MinimumNormalDot");

        internal bool Available => _compute != null;

        internal SurfaceProductizer(Transform space, VolumeIntegrator volume)
        {
            _space = space;
            _volume = volume;
            _compute = Resources.Load<ComputeShader>("SurfaceProductizer");
            if (_compute == null)
                return;
            if (!_compute.HasKernel("ProductizeVertices") ||
                !_compute.HasKernel("CopyIndices"))
            {
                _compute = null;
                return;
            }
            _copyVerticesKernel = _compute.FindKernel("ProductizeVertices");
            _copyIndicesKernel = _compute.FindKernel("CopyIndices");
            _equationsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                MaximumLocalPlanes, sizeof(float) * 4);
            _centersBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                MaximumLocalPlanes, sizeof(float) * 4);
            _axesUBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                MaximumLocalPlanes, sizeof(float) * 4);
            _axesVBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                MaximumLocalPlanes, sizeof(float) * 4);
        }

        internal GPUChunkMeshSnapshot Build(GPUSurfaceNets source,
            int vertexCount, int indexCount, Bounds localCoreBounds,
            float voxelSize)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            var snapshot = new GPUChunkMeshSnapshot();
            if (_compute == null || vertexCount <= 0 || indexCount <= 0)
            {
                source.CopyCurrentMeshTo(snapshot, vertexCount, indexCount);
                return snapshot;
            }

            snapshot.Prepare(vertexCount, indexCount, VertexStride);
            try
            {
                Bounds worldBounds = TransformBounds(localCoreBounds,
                    _space != null ? _space.localToWorldMatrix : Matrix4x4.identity);
                // Captured seed planes are finite post-TSDF rulers. They go
                // first because their exact rectangular support is measured;
                // the runtime court then contributes independent bounded
                // planes in the remaining slots.
                int planeCount = DepthCapture.Instance != null
                    ? DepthCapture.Instance.CopySeedProductConstraints(
                        worldBounds, _planeEquations, _planeCenters,
                        _planeAxesU, _planeAxesV, 0)
                    : 0;
                int courtCount = _volume != null && planeCount < MaximumLocalPlanes
                    ? _volume.CopyProductSurfacePlanes(worldBounds, _courtPlanes)
                    : 0;
                courtCount = Mathf.Min(courtCount, MaximumLocalPlanes - planeCount);
                for (int i = 0; i < courtCount; i++)
                {
                    int destination = planeCount + i;
                    _productPlanes[destination] = _courtPlanes[i];
                    _planeEquations[destination] = _courtPlanes[i].Equation;
                    _planeCenters[destination] = _courtPlanes[i].CenterRadius;
                    // Negative extent selects the court's radial support path.
                    _planeAxesU[destination] = new Vector4(0f, 0f, 0f, -1f);
                    _planeAxesV[destination] = new Vector4(0f, 0f, 0f, -1f);
                }
                planeCount += courtCount;

                if (planeCount > 0)
                {
                    _equationsBuffer.SetData(_planeEquations, 0, 0, planeCount);
                    _centersBuffer.SetData(_planeCenters, 0, 0, planeCount);
                    _axesUBuffer.SetData(_planeAxesU, 0, 0, planeCount);
                    _axesVBuffer.SetData(_planeAxesV, 0, 0, planeCount);
                }

                Matrix4x4 localToWorld = _space != null
                    ? _space.localToWorldMatrix : Matrix4x4.identity;
                Matrix4x4 worldToLocal = _space != null
                    ? _space.worldToLocalMatrix : Matrix4x4.identity;
                _compute.SetInt(ProductPlaneCountId, planeCount);
                _compute.SetInt(VertexCountId, Mathf.Max(0, vertexCount));
                _compute.SetInt(IndexCountId, Mathf.Max(0, indexCount));
                _compute.SetMatrix(LocalToWorldId, localToWorld);
                _compute.SetMatrix(WorldToLocalId, worldToLocal);
                _compute.SetVector(CoreMinId, localCoreBounds.min);
                _compute.SetVector(CoreMaxId, localCoreBounds.max);
                _compute.SetFloat(EdgeBandId, Mathf.Max(0.0025f, voxelSize));
                _compute.SetFloat(MaximumProjectionId,
                    Mathf.Min(0.015f, voxelSize * 0.30f));
                _compute.SetFloat(MinimumNormalDotId, 0.9659258f);

                BindCommon(_copyVerticesKernel, source, snapshot,
                    _equationsBuffer, _centersBuffer, _axesUBuffer,
                    _axesVBuffer);
                BindCommon(_copyIndicesKernel, source, snapshot,
                    _equationsBuffer, _centersBuffer, _axesUBuffer,
                    _axesVBuffer);
                _compute.Dispatch(_copyVerticesKernel,
                    CeilDiv(vertexCount, 64), 1, 1);
                _compute.Dispatch(_copyIndicesKernel,
                    CeilDiv(indexCount, 64), 1, 1);
                return snapshot;
            }
            catch
            {
                snapshot.Dispose();
                snapshot = new GPUChunkMeshSnapshot();
                source.CopyCurrentMeshTo(snapshot, vertexCount, indexCount);
                return snapshot;
            }
        }

        private void BindCommon(int kernel, GPUSurfaceNets source,
            GPUChunkMeshSnapshot output, GraphicsBuffer equations,
            GraphicsBuffer centers, GraphicsBuffer axesU,
            GraphicsBuffer axesV)
        {
            _compute.SetBuffer(kernel, InputVerticesId, source.VertexBuffer);
            _compute.SetBuffer(kernel, InputIndicesId, source.IndexBuffer);
            _compute.SetBuffer(kernel, InputAdmissionId,
                source.VertexAdmissionClassBuffer);
            _compute.SetBuffer(kernel, OutputVerticesId, output.VertexBuffer);
            _compute.SetBuffer(kernel, OutputIndicesId, output.IndexBuffer);
            _compute.SetBuffer(kernel, OutputAdmissionId,
                output.VertexAdmissionClassBuffer);
            _compute.SetBuffer(kernel, ProductPlanesId, equations);
            _compute.SetBuffer(kernel, ProductPlaneCentersId, centers);
            _compute.SetBuffer(kernel, ProductPlaneAxesUId, axesU);
            _compute.SetBuffer(kernel, ProductPlaneAxesVId, axesV);
        }

        private static Bounds TransformBounds(Bounds local, Matrix4x4 matrix)
        {
            Vector3 center = matrix.MultiplyPoint3x4(local.center);
            Vector3 ext = local.extents;
            Vector3 axisX = matrix.MultiplyVector(new Vector3(ext.x, 0f, 0f));
            Vector3 axisY = matrix.MultiplyVector(new Vector3(0f, ext.y, 0f));
            Vector3 axisZ = matrix.MultiplyVector(new Vector3(0f, 0f, ext.z));
            Vector3 worldExtents = new Vector3(
                Mathf.Abs(axisX.x) + Mathf.Abs(axisY.x) + Mathf.Abs(axisZ.x),
                Mathf.Abs(axisX.y) + Mathf.Abs(axisY.y) + Mathf.Abs(axisZ.y),
                Mathf.Abs(axisX.z) + Mathf.Abs(axisY.z) + Mathf.Abs(axisZ.z));
            return new Bounds(center, worldExtents * 2f);
        }

        private static int CeilDiv(int value, int divisor)
        {
            return Mathf.Max(1, (value + divisor - 1) / divisor);
        }

        public void Dispose()
        {
            _equationsBuffer?.Release();
            _centersBuffer?.Release();
            _axesUBuffer?.Release();
            _axesVBuffer?.Release();
            _equationsBuffer = null;
            _centersBuffer = null;
            _axesUBuffer = null;
            _axesVBuffer = null;
            // ComputeShader is a Resources-owned asset. There is no second
            // persistent geometry store or history volume.
        }
    }
}
