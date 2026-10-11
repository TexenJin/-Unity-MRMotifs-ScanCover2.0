// Same-candidate causal links for MeshLayerEvidence.
// Mode 1: yellow crossing -> cyan arithmetic mean only.
// Mode 2: emitting 10 cm cell + TSDF endpoints -> yellow crossing.
// Mode 3: the two same-candidate signed TSDF endpoint values form two arrows
//         on the real voxel edge. Magenta A is the canonical interpolation
//         start and cyan B is the other endpoint. A white tail ring marks the
//         endpoint that actually contributed most to crossing motion between
//         two extractions; both tails are ringed when contributions are close.
//         Both arrows terminate at the yellow zero crossing. Arrow length
//         shows the crossing ratio and brightness shows current |D|.
// Each segment is expanded into a screen-space quad so Quest/Vulkan receives
// genuinely thick lines instead of implementation-dependent one-pixel lines.
// Nothing here samples or modifies the live TSDF.
Shader "QRS/MeshLayerRelations"
{
    Properties
    {
        _CausalLineWidth ("Causal line width (pixels)", Float) = 2.8
        _GridLineWidth ("10 cm cell line width (pixels)", Float) = 4.0
        _ArrowLineWidth ("TSDF arrow line width (pixels)", Float) = 3.4
        // This shader's historical NDC convention uses two units per physical
        // display pixel. 28 therefore renders as an unmistakable ~14 px head.
        _ArrowHeadSize ("TSDF arrow head length (NDC pixel units)", Float) = 28.0
        _ArrowTipGap ("TSDF arrow gap around yellow point", Float) = 13.0
        _PusherRingSize ("Temporal pusher tail ring radius", Float) = 24.0
    }
    SubShader
    {
        Tags { "Queue"="Overlay+79" "RenderType"="Transparent" "IgnoreProjector"="True" }
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
            StructuredBuffer<uint3> _LayerCrossingSpan;
            StructuredBuffer<float3> _LayerRawSurfaceNets;
            uint _LayerVoxCountX;
            uint _LayerVoxCountY;
            uint _LayerVoxCountZ;
            float _LayerVoxSize;
            int _LayerEvidenceMode;
            float _CausalLineWidth;
            float _GridLineWidth;
            float _ArrowLineWidth;
            float _ArrowHeadSize;
            float _ArrowTipGap;
            float _PusherRingSize;

            struct v2f
            {
                float4 pos : SV_POSITION;
                float4 color : COLOR0;
                float2 markerUv : TEXCOORD0;
                float markerRing : TEXCOORD1;
            };

            v2f HiddenVertex()
            {
                v2f o;
                o.pos = float4(0, 0, -10, 1);
                o.color = 0;
                o.markerUv = 0;
                o.markerRing = 0;
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

            float3 VoxelToWorld(float3 p)
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

            uint PackedSpan(uint3 spanEvidence, uint edge)
            {
                uint lane = edge >> 2u;
                uint shift = (edge & 3u) * 8u;
                uint packed = lane == 0u ? spanEvidence.x :
                              (lane == 1u ? spanEvidence.y : spanEvidence.z);
                return (packed >> shift) & 63u;
            }

            uint PackedPushClass(uint3 spanEvidence, uint edge)
            {
                uint lane = edge >> 2u;
                uint shift = (edge & 3u) * 8u;
                uint packed = lane == 0u ? spanEvidence.x :
                              (lane == 1u ? spanEvidence.y : spanEvidence.z);
                return ((packed >> shift) >> 6u) & 3u;
            }

            // One physical grid edge is shared by up to four active cells.
            // Give it to exactly one cell in the accountability view: the
            // edge incident to corner zero along +X, +Z or +Y.
            bool IsOwnedAccountabilityEdge(uint edge)
            {
                return edge == 0u || edge == 3u || edge == 8u;
            }

            // A/B are spatial identities here, not local cube labels. A is
            // always the lower-coordinate endpoint; B is the higher one.
            bool CanonicalizeEndpoints(inout uint a, inout uint b)
            {
                int3 oa = CornerOffset(a);
                int3 ob = CornerOffset(b);
                uint keyA = (uint)(oa.x + oa.y * 2 + oa.z * 4);
                uint keyB = (uint)(ob.x + ob.y * 2 + ob.z * 4);
                if (keyA <= keyB)
                    return false;
                uint swap = a;
                a = b;
                b = swap;
                return true;
            }

            uint CanonicalPushClass(uint pushClass, bool endpointsSwapped)
            {
                if (!endpointsSwapped || pushClass == 0u || pushClass == 3u)
                    return pushClass;
                return pushClass == 1u ? 2u : 1u;
            }

            float3 CrossingWorld(int3 cell, uint4 evidence, uint edge)
            {
                uint a, b;
                EdgeCorners(edge, a, b);
                float t = (float)PackedT(evidence, edge) / 255.0;
                float3 voxel = (float3)(cell + CornerOffset(a)) +
                    t * (float3)(CornerOffset(b) - CornerOffset(a));
                return VoxelToWorld(voxel);
            }

            float4 EndpointColor(uint4 evidence, uint edge, uint endpoint)
            {
                bool aNegative =
                    (evidence.w & (1u << (12u + edge))) != 0u;
                bool negative = endpoint == 0u ? aNegative : !aNegative;
                return negative
                    ? float4(1.0, 0.04, 0.76, 1.0)
                    : float4(0.08, 1.0, 0.24, 1.0);
            }

            void GridSegment(int3 cell, uint edge,
                out float3 p0, out float3 p1)
            {
                uint a, b;
                EdgeCorners(edge, a, b);
                p0 = VoxelToWorld((float3)(cell + CornerOffset(a)));
                p1 = VoxelToWorld((float3)(cell + CornerOffset(b)));
            }

            void ResolveSegment(uint mode, uint segment, uint sourceVertex,
                uint4 evidence, uint3 spanEvidence, int3 cell,
                out float3 p0, out float3 p1,
                out float4 c0, out float4 c1, out float pixelWidth,
                out bool valid)
            {
                valid = true;
                p0 = 0;
                p1 = 0;
                c0 = 0;
                c1 = 0;
                pixelWidth = _CausalLineWidth;

                if (mode < 2u)
                {
                    uint edge = segment;
                    if ((evidence.w & (1u << edge)) == 0u)
                    {
                        valid = false;
                        return;
                    }
                    p0 = CrossingWorld(cell, evidence, edge);
                    p1 = VoxelToWorld(_LayerRawSurfaceNets[sourceVertex]);
                    c0 = float4(1.0, 0.88, 0.0, 1.0);
                    c1 = float4(0.0, 1.0, 1.0, 1.0);
                    return;
                }

                if (mode >= 3u)
                {
                    // Two arrows per crossing. Both remain on the real voxel
                    // edge and terminate at the undisplaced yellow crossing.
                    // This is the current same-candidate endpoint state, not a
                    // synthetic screen-facing wave and not a later TSDF frame.
                    uint edge = segment >> 1u;
                    uint half = segment & 1u;
                    if ((evidence.w & (1u << edge)) == 0u)
                    {
                        valid = false;
                        return;
                    }
                    if (!IsOwnedAccountabilityEdge(edge))
                    {
                        valid = false;
                        return;
                    }

                    uint a, b;
                    EdgeCorners(edge, a, b);
                    float t = (float)PackedT(evidence, edge) / 255.0;
                    float span = ((float)PackedSpan(spanEvidence, edge) /
                        63.0) * 2.0;
                    bool aNegative =
                        (evidence.w & (1u << (12u + edge))) != 0u;
                    float valA = (aNegative ? -1.0 : 1.0) * t * span;
                    float valB = (aNegative ? 1.0 : -1.0) *
                        (1.0 - t) * span;
                    bool endpointsSwapped = CanonicalizeEndpoints(a, b);
                    if (endpointsSwapped)
                    {
                        float valueSwap = valA;
                        valA = valB;
                        valB = valueSwap;
                    }
                    float3 crossing = CrossingWorld(cell, evidence, edge);
                    uint pushClass = CanonicalPushClass(
                        PackedPushClass(spanEvidence, edge),
                        endpointsSwapped);
                    bool responsible = pushClass == 3u ||
                        pushClass == half + 1u;
                    float responsibilityGain = responsible ? 1.0 :
                        (pushClass == 0u ? 0.20 : 0.12);

                    pixelWidth = _ArrowLineWidth;
                    if (half == 0u)
                    {
                        p0 = VoxelToWorld(
                            (float3)(cell + CornerOffset(a)));
                        p1 = crossing;
                        float strengthA = 0.55 + 0.45 * saturate(abs(valA));
                        c0 = float4(1.0, 0.02, 0.38, 1.0) *
                            strengthA * responsibilityGain;
                        c1 = c0;
                    }
                    else
                    {
                        p0 = VoxelToWorld(
                            (float3)(cell + CornerOffset(b)));
                        p1 = crossing;
                        float strengthB = 0.55 + 0.45 * saturate(abs(valB));
                        c0 = float4(0.0, 0.92, 1.0, 1.0) *
                            strengthB * responsibilityGain;
                        c1 = c0;
                    }
                    return;
                }

                if (segment < 12u)
                {
                    GridSegment(cell, segment, p0, p1);
                    c0 = float4(0.20, 0.68, 1.0, 1.0);
                    c1 = c0;
                    pixelWidth = _GridLineWidth;
                    return;
                }

                // Each crossing contributes two explicit parent relations:
                // endpoint A -> yellow and endpoint B -> yellow.
                uint relation = segment - 12u;
                uint edge = relation >> 1u;
                uint endpoint = relation & 1u;
                if ((evidence.w & (1u << edge)) == 0u)
                {
                    valid = false;
                    return;
                }
                uint a, b;
                EdgeCorners(edge, a, b);
                uint corner = endpoint == 0u ? a : b;
                p0 = VoxelToWorld((float3)(cell + CornerOffset(corner)));
                p1 = CrossingWorld(cell, evidence, edge);
                c0 = EndpointColor(evidence, edge, endpoint);
                c1 = float4(1.0, 0.88, 0.0, 1.0);
            }

            v2f vert(uint vertexID : SV_VertexID)
            {
                // Six triangle vertices per screen-space line segment.
                // Mode 1: 12 causal segments = 72 vertices.
                // Mode 2: 12 grid + 24 source = 36 segments = 216 vertices.
                // Mode 3: three uniquely owned crossings per cell; each has
                // two 9-vertex arrows plus a six-vertex ring quad per endpoint.
                uint verticesPerCell = _LayerEvidenceMode >= 3 ? 90u :
                    (_LayerEvidenceMode >= 2 ? 216u : 72u);
                uint sourceVertex = vertexID / verticesPerCell;
                uint localVertex = vertexID - sourceVertex * verticesPerCell;
                uint segment;
                uint primitiveVertex;
                bool pusherRing = false;
                uint ringEndpoint = 0u;
                if (_LayerEvidenceMode >= 3)
                {
                    uint ownedEdge = localVertex / 30u;
                    uint edge = ownedEdge == 0u ? 0u :
                                (ownedEdge == 1u ? 3u : 8u);
                    uint edgeVertex = localVertex - ownedEdge * 30u;
                    if (edgeVertex < 9u)
                    {
                        segment = edge * 2u;
                        primitiveVertex = edgeVertex;
                    }
                    else if (edgeVertex < 18u)
                    {
                        segment = edge * 2u + 1u;
                        primitiveVertex = edgeVertex - 9u;
                    }
                    else if (edgeVertex < 24u)
                    {
                        segment = edge * 2u;
                        primitiveVertex = edgeVertex - 18u;
                        pusherRing = true;
                        ringEndpoint = 0u;
                    }
                    else
                    {
                        segment = edge * 2u + 1u;
                        primitiveVertex = edgeVertex - 24u;
                        pusherRing = true;
                        ringEndpoint = 1u;
                    }
                }
                else
                {
                    segment = localVertex / 6u;
                    primitiveVertex = localVertex - segment * 6u;
                }

                uint4 evidence = _LayerCrossingEvidence[sourceVertex];
                uint3 spanEvidence = _LayerCrossingSpan[sourceVertex];
                int3 cell = Unflatten(_LayerVertices[sourceVertex].voxelFlatIdx);
                float3 p0, p1;
                float4 c0, c1;
                float pixelWidth;
                bool valid;
                ResolveSegment((uint)_LayerEvidenceMode, segment, sourceVertex,
                    evidence, spanEvidence, cell, p0, p1, c0, c1,
                    pixelWidth, valid);
                if (!valid)
                    return HiddenVertex();

                float4 clip0 = UnityWorldToClipPos(p0);
                float4 clip1 = UnityWorldToClipPos(p1);
                if (clip0.w <= 0.0 || clip1.w <= 0.0)
                    return HiddenVertex();

                if (pusherRing)
                {
                    uint edge = segment >> 1u;
                    if (!IsOwnedAccountabilityEdge(edge))
                        return HiddenVertex();
                    uint a, b;
                    EdgeCorners(edge, a, b);
                    bool endpointsSwapped = CanonicalizeEndpoints(a, b);
                    uint pushClass = CanonicalPushClass(
                        PackedPushClass(spanEvidence, edge),
                        endpointsSwapped);
                    bool showRing = pushClass == 3u ||
                        pushClass == ringEndpoint + 1u;
                    if (!showRing)
                        return HiddenVertex();

                    float2 corner = primitiveVertex == 0u ? float2(-1.0, -1.0) :
                                    primitiveVertex == 1u ? float2(-1.0,  1.0) :
                                    primitiveVertex == 2u ? float2( 1.0,  1.0) :
                                    primitiveVertex == 3u ? float2(-1.0, -1.0) :
                                    primitiveVertex == 4u ? float2( 1.0,  1.0) :
                                                            float2( 1.0, -1.0);
                    v2f ring;
                    ring.pos = clip0;
                    ring.pos.xy += corner * (_PusherRingSize / _ScreenParams.xy) *
                        ring.pos.w;
                    ring.color = ringEndpoint == 0u
                        ? float4(1.0, 0.02, 0.38, 1.0)
                        : float4(0.0, 0.92, 1.0, 1.0);
                    ring.markerUv = corner;
                    ring.markerRing = 1.0;
                    return ring;
                }

                float2 ndc0 = clip0.xy / clip0.w;
                float2 ndc1 = clip1.xy / clip1.w;
                float2 pixelDelta = (ndc1 - ndc0) * _ScreenParams.xy;
                float pixelLength = length(pixelDelta);
                if (pixelLength < 0.001)
                    return HiddenVertex();

                float2 direction = pixelDelta / pixelLength;
                float2 perpendicular = float2(-direction.y, direction.x);

                if (_LayerEvidenceMode >= 3)
                {
                    // The yellow point pass is 12 physical pixels wide and is
                    // drawn after this relation pass. Stop at its visible rim
                    // instead of burying the entire head under its centre.
                    float tipGap = min(_ArrowTipGap,
                        max(0.0, pixelLength * 0.30));
                    clip1.xy -= direction *
                        (tipGap / _ScreenParams.xy) * clip1.w;
                    ndc1 = clip1.xy / clip1.w;
                    pixelDelta = (ndc1 - ndc0) * _ScreenParams.xy;
                    pixelLength = length(pixelDelta);
                    if (pixelLength < 0.001)
                        return HiddenVertex();
                    direction = pixelDelta / pixelLength;
                    perpendicular = float2(-direction.y, direction.x);
                }

                if (_LayerEvidenceMode >= 3 && primitiveVertex >= 6u)
                {
                    // A compact screen-space arrowhead whose tip stops at the
                    // visible rim of the yellow crossing. Clamp it on short
                    // projected edges so it never turns around or covers the shaft.
                    float headLength = min(_ArrowHeadSize,
                        max(10.0, pixelLength * 0.55));
                    float headHalfWidth = headLength * 0.72;
                    float2 headPixelOffset = 0.0;
                    if (primitiveVertex == 7u)
                        headPixelOffset = -direction * headLength +
                            perpendicular * headHalfWidth;
                    else if (primitiveVertex == 8u)
                        headPixelOffset = -direction * headLength -
                            perpendicular * headHalfWidth;

                    v2f arrow;
                    arrow.pos = clip1;
                    arrow.pos.xy += (headPixelOffset / _ScreenParams.xy) *
                        arrow.pos.w;
                    // Preserve A/B identity right up to the yellow point rim.
                    arrow.color = c0;
                    arrow.markerUv = 0;
                    arrow.markerRing = 0;
                    return arrow;
                }

                uint quadVertex = primitiveVertex;
                float2 offsetNdc = perpendicular *
                    (pixelWidth / _ScreenParams.xy);

                bool useEnd = quadVertex == 1u || quadVertex == 2u ||
                              quadVertex == 4u;
                bool positiveSide = quadVertex == 2u || quadVertex == 4u ||
                                    quadVertex == 5u;
                float side = positiveSide ? 1.0 : -1.0;

                v2f o;
                o.pos = useEnd ? clip1 : clip0;
                o.pos.xy += offsetNdc * o.pos.w * side;
                o.color = useEnd ? c1 : c0;
                o.markerUv = 0;
                o.markerRing = 0;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                if (i.markerRing > 0.5)
                {
                    float radius = length(i.markerUv);
                    if (radius > 1.0 || radius < 0.62)
                        discard;
                }
                return i.color;
            }
            ENDCG
        }
    }
}
