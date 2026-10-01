using System;
using Unity.Mathematics;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Immutable front-buffer for a completed chunk extraction.  Extraction is
    /// allowed to overwrite its working buffers; rendering changes only after a
    /// candidate has passed the commit gate and is copied here atomically.
    /// </summary>
    internal sealed class GPUChunkMeshSnapshot : IGPUMeshBufferSource, IDisposable
    {
        public GraphicsBuffer VertexBuffer { get; private set; }
        public GraphicsBuffer IndexBuffer { get; private set; }
        public GraphicsBuffer VertexAdmissionClassBuffer { get; private set; }
        public GraphicsBuffer CoordinateVertexMapBuffer { get; private set; }
        public GraphicsBuffer DrawIndirectArgs { get; private set; }
        public GraphicsBuffer LineDrawIndirectArgs => null;
        public int KnownVertexCount { get; private set; }
        public int KnownDrawVertexCount { get; private set; }
        public bool HasCoordinateVertexMap { get; private set; }
        public int3 MapMin { get; private set; }
        public int3 MapCount { get; private set; }
        public int3 CoreMin { get; private set; }
        public int3 CoreMax { get; private set; }

        public void Prepare(int vertexCount, int indexCount, int vertexStride)
        {
            int actualVertexCount = Mathf.Max(0, vertexCount);
            int actualIndexCount = Mathf.Max(0, indexCount);
            PrepareCapacity(actualVertexCount, actualIndexCount, vertexStride);
            KnownVertexCount = actualVertexCount;
            KnownDrawVertexCount = actualIndexCount;
            HasCoordinateVertexMap = false;
            DrawIndirectArgs.SetData(new uint[] { (uint)actualIndexCount, 1, 0, 0, 0 });
        }

        public void PrepareCoordinateVertexMap(int3 mapMin, int3 mapCount,
            int3 coreMin, int3 coreMax)
        {
            int required = Mathf.Max(1, mapCount.x * mapCount.y * mapCount.z);
            if (CoordinateVertexMapBuffer == null ||
                CoordinateVertexMapBuffer.count < required)
            {
                CoordinateVertexMapBuffer?.Release();
                CoordinateVertexMapBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, required, sizeof(int));
            }

            MapMin = mapMin;
            MapCount = mapCount;
            CoreMin = coreMin;
            CoreMax = coreMax;
            HasCoordinateVertexMap = true;
        }

        /// <summary>
        /// Allocates storage without claiming that the full capacity is visible.
        /// Used by the additive merge kernel, which writes the final indirect
        /// index count only after it has admitted novel mature triangles.
        /// </summary>
        public void PrepareCapacity(int vertexCapacity, int indexCapacity, int vertexStride)
        {
            vertexCapacity = Mathf.Max(1, vertexCapacity);
            indexCapacity = Mathf.Max(1, indexCapacity);
            // Immutable fronts are reused as a two-buffer publication ring.
            // Grow on demand, but do not destroy/reallocate them merely because
            // the next extraction contains fewer elements.
            if (VertexBuffer == null || VertexBuffer.count < vertexCapacity ||
                IndexBuffer == null || IndexBuffer.count < indexCapacity)
            {
                Dispose();
                VertexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                    vertexCapacity, vertexStride);
                IndexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                    indexCapacity, sizeof(uint));
                VertexAdmissionClassBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                    vertexCapacity, sizeof(uint));
                DrawIndirectArgs = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.IndirectArguments,
                    5, sizeof(uint));
            }
            KnownDrawVertexCount = 0;
            KnownVertexCount = 0;
            HasCoordinateVertexMap = false;
            DrawIndirectArgs.SetData(new uint[] { 0, 1, 0, 0, 0 });
        }

        /// <summary>
        /// Publishes the exact number of indices that were admitted by the
        /// completed extraction.  HERA allocates snapshots by capacity first;
        /// the GPU also writes this argument, but publishing it again from the
        /// completed readback keeps the visible front-buffer independent from
        /// command-buffer/renderer enable ordering.
        /// </summary>
        public void SetDrawIndexCount(int indexCount)
        {
            if (DrawIndirectArgs == null)
                return;

            uint visibleIndexCount = (uint)Mathf.Max(0, indexCount);
            KnownDrawVertexCount = (int)visibleIndexCount;
            DrawIndirectArgs.SetData(new uint[] { visibleIndexCount, 1, 0, 0, 0 });
        }

        public void Clear()
        {
            KnownDrawVertexCount = 0;
            KnownVertexCount = 0;
            HasCoordinateVertexMap = false;
            if (DrawIndirectArgs != null)
                DrawIndirectArgs.SetData(new uint[] { 0, 1, 0, 0, 0 });
        }

        public void Dispose()
        {
            VertexBuffer?.Release();
            IndexBuffer?.Release();
            VertexAdmissionClassBuffer?.Release();
            CoordinateVertexMapBuffer?.Release();
            DrawIndirectArgs?.Release();
            VertexBuffer = null;
            IndexBuffer = null;
            VertexAdmissionClassBuffer = null;
            CoordinateVertexMapBuffer = null;
            DrawIndirectArgs = null;
            KnownVertexCount = 0;
            KnownDrawVertexCount = 0;
            HasCoordinateVertexMap = false;
        }
    }
}
