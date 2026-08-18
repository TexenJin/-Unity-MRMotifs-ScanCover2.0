using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>Read-only buffers consumed by the GPU mesh renderer.</summary>
    internal interface IGPUMeshBufferSource
    {
        GraphicsBuffer VertexBuffer { get; }
        GraphicsBuffer IndexBuffer { get; }
        GraphicsBuffer VertexAdmissionClassBuffer { get; }
        GraphicsBuffer DrawIndirectArgs { get; }

        /// <summary>
        /// Exact non-indexed draw count when it is already known on the CPU.
        /// A negative value means the GPU indirect argument buffer remains the
        /// authoritative source (the live extraction path).
        /// </summary>
        int KnownDrawVertexCount { get; }
    }
}
