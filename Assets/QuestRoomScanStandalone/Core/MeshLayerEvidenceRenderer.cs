using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Draws immutable, same-candidate extraction provenance captured beside a
    /// published block Front. It never samples the live TSDF and never writes
    /// production buffers, so crossings, means and signed-value arrows cannot
    /// drift to a later integration frame.
    /// </summary>
    public sealed class MeshLayerEvidenceRenderer : MonoBehaviour
    {
        private static readonly int VerticesID = Shader.PropertyToID("_LayerVertices");
        private static readonly int CrossingEvidenceID =
            Shader.PropertyToID("_LayerCrossingEvidence");
        private static readonly int CrossingSpanID =
            Shader.PropertyToID("_LayerCrossingSpan");
        private static readonly int RawSurfaceNetsID =
            Shader.PropertyToID("_LayerRawSurfaceNets");
        private static readonly int VoxCountXID = Shader.PropertyToID("_LayerVoxCountX");
        private static readonly int VoxCountYID = Shader.PropertyToID("_LayerVoxCountY");
        private static readonly int VoxCountZID = Shader.PropertyToID("_LayerVoxCountZ");
        private static readonly int VoxSizeID = Shader.PropertyToID("_LayerVoxSize");
        private static readonly int EvidenceModeID =
            Shader.PropertyToID("_LayerEvidenceMode");

        private Material _pointMaterial;
        private Material _relationMaterial;
        private MaterialPropertyBlock _properties;
        private GPUChunkMeshSnapshot _source;
        private Bounds _bounds;
        private int3 _voxCount;
        private float _voxSize;
        private int _mode;

        internal void Initialize(Material pointMaterial, Material relationMaterial,
            Bounds bounds, int3 voxCount, float voxSize)
        {
            _pointMaterial = pointMaterial;
            _relationMaterial = relationMaterial;
            _bounds = bounds;
            _voxCount = voxCount;
            _voxSize = voxSize;
            _properties = new MaterialPropertyBlock();
        }

        internal void SetSource(GPUChunkMeshSnapshot source) => _source = source;
        internal void SetMode(int mode) => _mode = Mathf.Clamp(mode, 0, 3);

        private void LateUpdate()
        {
            if (_mode <= 0 || _pointMaterial == null || _properties == null ||
                _source == null || _source.KnownVertexCount <= 0 ||
                _source.VertexBuffer == null ||
                _source.LayerCrossingEvidenceBuffer == null ||
                _source.LayerCrossingSpanBuffer == null ||
                _source.LayerRawSurfaceNetsBuffer == null)
                return;

            _properties.Clear();
            _properties.SetBuffer(VerticesID, _source.VertexBuffer);
            _properties.SetBuffer(CrossingEvidenceID,
                _source.LayerCrossingEvidenceBuffer);
            _properties.SetBuffer(CrossingSpanID,
                _source.LayerCrossingSpanBuffer);
            _properties.SetBuffer(RawSurfaceNetsID,
                _source.LayerRawSurfaceNetsBuffer);
            _properties.SetInt(VoxCountXID, _voxCount.x);
            _properties.SetInt(VoxCountYID, _voxCount.y);
            _properties.SetInt(VoxCountZID, _voxCount.z);
            _properties.SetFloat(VoxSizeID, _voxSize);
            _properties.SetInt(EvidenceModeID, _mode);

            var pointParams = new RenderParams(_pointMaterial)
            {
                worldBounds = _bounds,
                matProps = _properties,
                receiveShadows = false,
                shadowCastingMode = ShadowCastingMode.Off,
                layer = gameObject.layer
            };
            int pointsPerVertex = _mode >= 3 ? 3 : (_mode >= 2 ? 20 : 13);
            Graphics.RenderPrimitives(pointParams, MeshTopology.Points,
                _source.KnownVertexCount * pointsPerVertex, 1);

            if (_relationMaterial == null)
                return;

            var relationParams = new RenderParams(_relationMaterial)
            {
                worldBounds = _bounds,
                matProps = _properties,
                receiveShadows = false,
                shadowCastingMode = ShadowCastingMode.Off,
                layer = gameObject.layer
            };
            int relationVerticesPerVertex = _mode >= 3 ? 90 :
                (_mode >= 2 ? 216 : 72);
            Graphics.RenderPrimitives(relationParams, MeshTopology.Triangles,
                _source.KnownVertexCount * relationVerticesPerVertex, 1);
        }
    }
}
