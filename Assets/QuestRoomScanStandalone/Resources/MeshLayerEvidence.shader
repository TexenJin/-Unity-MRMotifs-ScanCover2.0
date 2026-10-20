// Read-only, same-candidate Surface Nets provenance.
// Yellow = the exact production crossing edges with an 8-bit display-only t.
//           Its worst-case position error is edgeLength / 510 (0.196 mm on
//           the active 10 cm baseline), while membership is exact.
// Cyan   = the full-precision arithmetic mean used by production before any
//           optional post-processing.
Shader "QRS/MeshLayerEvidence"
{
    Properties
    {
        _CrossingPointSize ("Crossing point size", Float) = 12.0
        _RawPointSize ("Raw Surface Nets point size", Float) = 15.0
        _SourcePointSize ("TSDF source endpoint size", Float) = 12.0
    }
    SubShader
    {
        Tags { "Queue"="Overlay+80" "RenderType"="Transparent" "IgnoreProjector"="True" }
        Pass
        {
            Cull Off
            ZWrite Off
            ZTest Always
            Blend One One

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5
            #include "UnityCG.cginc"

            struct GPUVertex
            {
                float3 pos;
                float3 norm;
                uint packedColor;
                uint voxelFlatIdx;
            };

            StructuredBuffer<GPUVertex> _LayerVertices;
            StructuredBuffer<uint4> _LayerCrossingEvidence;
            StructuredBuffer<float3> _LayerRawSurfaceNets;
            uint _LayerVoxCountX;
            uint _LayerVoxCountY;
            uint _LayerVoxCountZ;
            float _LayerVoxSize;
            float _CrossingPointSize;
            float _RawPointSize;
            float _SourcePointSize;
            int _LayerEvidenceMode;

            struct v2f
            {
                float4 pos : SV_POSITION;
                float4 color : COLOR0;
                float psize : PSIZE;
            };

            v2f HiddenPoint()
            {
                v2f o;
                o.pos = float4(0, 0, -10, 1);
                o.color = 0;
                o.psize = 0;
                return o;
            }

            int3 CornerOffset(uint corner)
            {
                if (corner == 0u) return int3(0,0,0);
                if (corner == 1u) return int3(1,0,0);
                if (corner == 2u) return int3(1,0,1);
                if (corner == 3u) return int3(0,0,1);
                if (corner == 4u) return int3(0,1,0);
                if (corner == 5u) return int3(1,1,0);
                if (corner == 6u) return int3(1,1,1);
                return int3(0,1,1);
            }

            void EdgeCorners(uint edge, out uint a, out uint b)
            {
                if (edge < 4u)
                {
                    a = edge;
                    b = (edge + 1u) & 3u;
                }
                else if (edge < 8u)
                {
                    a = 4u + (edge - 4u);
                    b = 4u + ((edge - 3u) & 3u);
                }
                else
                {
                    a = edge - 8u;
                    b = a + 4u;
                }
            }

            int3 Unflatten(uint flat)
            {
                uint slice = _LayerVoxCountX * _LayerVoxCountY;
                uint z = flat / slice;
                uint rem = flat - z * slice;
                uint y = rem / _LayerVoxCountX;
                uint x = rem - y * _LayerVoxCountX;
                return int3(x, y, z);
            }

            float3 VoxelToLocal(float3 p)
            {
                float3 count = float3(_LayerVoxCountX, _LayerVoxCountY,
                    _LayerVoxCountZ);
                return (p + 0.5 - count / 2.0) * _LayerVoxSize;
            }

            uint PackedT(uint4 evidence, uint edge)
            {
                uint lane = edge >> 2u;
                uint shift = (edge & 3u) * 8u;
                uint packed = lane == 0u ? evidence.x :
                              (lane == 1u ? evidence.y : evidence.z);
                return (packed >> shift) & 255u;
            }

            v2f vert(uint vertexID : SV_VertexID)
            {
                // Mode 1: yellow crossings + cyan mean.
                // Mode 2: yellow crossings + their TSDF endpoints.
                // Mode 3: yellow crossings only; their two signed source
                // values are rendered as on-axis arrows by the relation pass.
                uint samplesPerVertex = _LayerEvidenceMode >= 3 ? 3u :
                    (_LayerEvidenceMode >= 2 ? 20u : 13u);
                uint sourceVertex = vertexID / samplesPerVertex;
                uint sample = vertexID - sourceVertex * samplesPerVertex;
                uint4 evidence = _LayerCrossingEvidence[sourceVertex];
                float3 world;
                float4 color;
                float size;

                if (_LayerEvidenceMode >= 2)
                {
                    if (sample < 12u)
                    {
                        uint edge = sample;
                        if (_LayerEvidenceMode >= 3)
                            edge = sample == 0u ? 0u :
                                   (sample == 1u ? 3u : 8u);
                        if ((evidence.w & (1u << edge)) == 0u)
                            return HiddenPoint();
                        uint a, b;
                        EdgeCorners(edge, a, b);
                        int3 cell = Unflatten(
                            _LayerVertices[sourceVertex].voxelFlatIdx);
                        float t = (float)PackedT(evidence, edge) / 255.0;
                        float3 voxel = (float3)(cell + CornerOffset(a)) +
                            t * (float3)(CornerOffset(b) - CornerOffset(a));
                        world = VoxelToLocal(voxel);
                        color = float4(1.0, 0.88, 0.0, 1.0);
                        size = _CrossingPointSize;
                    }
                    else
                    {
                        // Draw each contributing TSDF corner only once. A
                        // corner may feed several yellow crossings; repeated
                        // additive points would falsely imply greater weight.
                        uint corner = sample - 12u;
                        bool found = false;
                        bool negative = false;
                        [unroll]
                        for (uint edge = 0u; edge < 12u; edge++)
                        {
                            if (found ||
                                (evidence.w & (1u << edge)) == 0u)
                                continue;
                            uint a, b;
                            EdgeCorners(edge, a, b);
                            bool aNegative =
                                (evidence.w & (1u << (12u + edge))) != 0u;
                            if (corner == a)
                            {
                                found = true;
                                negative = aNegative;
                            }
                            else if (corner == b)
                            {
                                found = true;
                                negative = !aNegative;
                            }
                        }
                        if (!found)
                            return HiddenPoint();

                        int3 cell = Unflatten(
                            _LayerVertices[sourceVertex].voxelFlatIdx);
                        world = VoxelToLocal(
                            (float3)(cell + CornerOffset(corner)));
                        color = negative
                            ? float4(1.0, 0.04, 0.76, 1.0)
                            : float4(0.08, 1.0, 0.24, 1.0);
                        size = _SourcePointSize;
                    }
                }
                else if (sample == 12u)
                {
                    world = VoxelToLocal(_LayerRawSurfaceNets[sourceVertex]);
                    color = float4(0.0, 1.0, 1.0, 1.0);
                    size = _RawPointSize;
                }
                else
                {
                    if ((evidence.w & (1u << sample)) == 0u)
                        return HiddenPoint();
                    uint a, b;
                    EdgeCorners(sample, a, b);
                    int3 cell = Unflatten(_LayerVertices[sourceVertex].voxelFlatIdx);
                    float t = (float)PackedT(evidence, sample) / 255.0;
                    float3 voxel = (float3)(cell + CornerOffset(a)) +
                        t * (float3)(CornerOffset(b) - CornerOffset(a));
                    world = VoxelToLocal(voxel);
                    color = float4(1.0, 0.82, 0.0, 1.0);
                    size = _CrossingPointSize;
                }

                v2f o;
                // Production GPUVertex.pos is already volume/world space and
                // ScanMeshVertexColor sends it through the world-to-clip path.
                // Use the identical convention here so the evidence cannot be
                // displaced by a GameObject transform.
                o.pos = UnityWorldToClipPos(world);
                o.color = color;
                o.psize = size;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                return i.color;
            }
            ENDCG
        }
    }
}
