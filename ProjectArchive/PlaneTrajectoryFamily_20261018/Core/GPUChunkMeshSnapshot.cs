using System;
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
        public GraphicsBuffer PlaneFamilyBuffer { get; private set; }
        internal GraphicsBuffer PlaneFamilyStatsBuffer { get; private set; }
        public GraphicsBuffer LayerCrossingEvidenceBuffer { get; private set; }
        public GraphicsBuffer LayerCrossingSpanBuffer { get; private set; }
        public GraphicsBuffer LayerRawSurfaceNetsBuffer { get; private set; }
        public GraphicsBuffer DrawIndirectArgs { get; private set; }
        public int KnownVertexCount { get; private set; }
        public int KnownDrawVertexCount { get; private set; }
        internal uint PlaneFamilyLabeledTriangles { get; private set; }
        internal uint PlaneFamilyConflictTriangles { get; private set; }
        internal uint PlaneFamilyTotalTriangles { get; private set; }
        internal bool PlaneFamilyInitialized { get; private set; }
        public bool PlaneFamilyBufferReady => PlaneFamilyInitialized;
        private int _planeFamilyStatsGeneration;

        public void Prepare(int vertexCount, int indexCount, int vertexStride)
        {
            int actualVertexCount = Mathf.Max(0, vertexCount);
            int actualIndexCount = Mathf.Max(0, indexCount);
            PrepareCapacity(actualVertexCount, actualIndexCount, vertexStride);
            KnownVertexCount = actualVertexCount;
            KnownDrawVertexCount = actualIndexCount;
            DrawIndirectArgs.SetData(new uint[] { (uint)actualIndexCount, 1, 0, 0, 0 });
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
                IndexBuffer == null || IndexBuffer.count < indexCapacity ||
                LayerCrossingEvidenceBuffer == null ||
                LayerCrossingSpanBuffer == null ||
                LayerRawSurfaceNetsBuffer == null ||
                PlaneFamilyBuffer == null ||
                PlaneFamilyStatsBuffer == null)
            {
                Dispose();
                VertexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                    vertexCapacity, vertexStride);
                IndexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                    indexCapacity, sizeof(uint));
                VertexAdmissionClassBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                    vertexCapacity, sizeof(uint));
                PlaneFamilyBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, vertexCapacity,
                    sizeof(uint));
                PlaneFamilyStatsBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, 3, sizeof(uint));
                LayerCrossingEvidenceBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                    vertexCapacity, sizeof(uint) * 4);
                LayerCrossingSpanBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                    vertexCapacity, sizeof(uint) * 3);
                LayerRawSurfaceNetsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                    vertexCapacity, sizeof(float) * 3);
                DrawIndirectArgs = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.IndirectArguments,
                    5, sizeof(uint));
            }
            KnownVertexCount = 0;
            KnownDrawVertexCount = 0;
            PlaneFamilyLabeledTriangles = 0;
            PlaneFamilyConflictTriangles = 0;
            PlaneFamilyTotalTriangles = 0;
            PlaneFamilyInitialized = false;
            _planeFamilyStatsGeneration++;
            DrawIndirectArgs.SetData(new uint[] { 0, 1, 0, 0, 0 });
        }

        internal int BeginPlaneFamilyStatsUpdate()
        {
            PlaneFamilyStatsBuffer?.SetData(new uint[] { 0u, 0u, 0u });
            return ++_planeFamilyStatsGeneration;
        }

        internal void CompletePlaneFamilyStatsUpdate(int generation,
            uint labeled, uint conflict, uint total)
        {
            if (generation != _planeFamilyStatsGeneration)
                return;
            PlaneFamilyLabeledTriangles = labeled;
            PlaneFamilyConflictTriangles = conflict;
            PlaneFamilyTotalTriangles = total;
        }

        internal void MarkPlaneFamilyInitialized()
        {
            PlaneFamilyInitialized = true;
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
            KnownVertexCount = 0;
            KnownDrawVertexCount = 0;
            if (DrawIndirectArgs != null)
                DrawIndirectArgs.SetData(new uint[] { 0, 1, 0, 0, 0 });
        }

        public void Dispose()
        {
            VertexBuffer?.Release();
            IndexBuffer?.Release();
            VertexAdmissionClassBuffer?.Release();
            PlaneFamilyBuffer?.Release();
            PlaneFamilyStatsBuffer?.Release();
            LayerCrossingEvidenceBuffer?.Release();
            LayerCrossingSpanBuffer?.Release();
            LayerRawSurfaceNetsBuffer?.Release();
            DrawIndirectArgs?.Release();
            VertexBuffer = null;
            IndexBuffer = null;
            VertexAdmissionClassBuffer = null;
            PlaneFamilyBuffer = null;
            PlaneFamilyStatsBuffer = null;
            LayerCrossingEvidenceBuffer = null;
            LayerCrossingSpanBuffer = null;
            LayerRawSurfaceNetsBuffer = null;
            DrawIndirectArgs = null;
            KnownVertexCount = 0;
            KnownDrawVertexCount = 0;
            PlaneFamilyLabeledTriangles = 0;
            PlaneFamilyConflictTriangles = 0;
            PlaneFamilyTotalTriangles = 0;
            PlaneFamilyInitialized = false;
            _planeFamilyStatsGeneration++;
        }
    }
}
