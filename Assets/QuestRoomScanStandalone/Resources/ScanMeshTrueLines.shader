Shader "Genesis/ScanMeshTrueLines"
{
    Properties { }
    SubShader
    {
        Tags
        {
            "RenderType"="Transparent"
            "RenderPipeline"="UniversalPipeline"
            "Queue"="Transparent+40"
        }

        Pass
        {
            Name "TrueMeshLines"
            Tags { "LightMode"="SRPDefaultUnlit" }

            // Replacement blending keeps a shared edge and two coincident
            // submissions equally bright. Separated double surfaces remain
            // visible instead of being hidden by additive-looking alpha.
            Blend One Zero
            // The sealed triangle prepass already owns the nearest depth.
            // Writing it again from one-pixel lines only lets later line draws
            // fight each other at grazing angles.
            ZWrite Off
            ZTest LEqual
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct GPUVertex
            {
                float3 pos;
                float3 norm;
                uint packedColor;
                uint voxelFlatIdx;
            };

            StructuredBuffer<GPUVertex> _SurfaceVerts;
            StructuredBuffer<uint> _SurfaceIndices;
            StructuredBuffer<uint> _VertexAdmissionClass;

            TEXTURE3D(gsVolume);
            SAMPLER(sampler_gsVolume);
            TEXTURE3D(gsConfidence);
            SAMPLER(sampler_gsConfidence);

            float4 gsVoxCount;
            float gsVoxSize;
            float gsConfidenceMidMax;
            float gsConfidenceLowMin;
            float _RSConfidenceViz;
            float _RSMeshStride;
            float _RSTrueLineQuadTopology;
            float _RSGeometryTruthView;
            float4 _RSExtractionColor;
            float _RSJointDiagnostic;
            float _RSSuppressPink;
            float _RSHeraReplayActive;
            float _RSBoundaryUnificationDiagnostic;

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                nointerpolation uint diagnosticClass : TEXCOORD1;
                nointerpolation uint legacyDiagnosticClass : TEXCOORD2;
                nointerpolation uint diagnosticSupportMask : TEXCOORD3;
                nointerpolation uint edgeIndex : TEXCOORD4;
                nointerpolation uint boundaryLineClass : TEXCOORD5;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            uint BoundaryLineClass(uint sourceIndexPosition0,
                                   uint sourceIndexPosition1)
            {
                uint state0 = _VertexAdmissionClass[
                    _SurfaceIndices[sourceIndexPosition0] & 0x3FFFFFFFu];
                uint state1 = _VertexAdmissionClass[
                    _SurfaceIndices[sourceIndexPosition1] & 0x3FFFFFFFu];
                bool boundary0 = (state0 & (1u << 27u)) != 0u;
                bool boundary1 = (state1 & (1u << 27u)) != 0u;
                if (!boundary0 && !boundary1)
                    return 0u;

                bool unresolved0 = boundary0 &&
                    (state0 & (1u << 28u)) == 0u;
                bool unresolved1 = boundary1 &&
                    (state1 & (1u << 28u)) == 0u;
                return unresolved0 || unresolved1 ? 2u : 1u;
            }

            uint LegacyDiagnosticClass(uint triBase)
            {
                uint c0 = _VertexAdmissionClass[_SurfaceIndices[triBase] & 0x3FFFFFFFu];
                uint c1 = _VertexAdmissionClass[_SurfaceIndices[triBase + 1u] & 0x3FFFFFFFu];
                uint c2 = _VertexAdmissionClass[_SurfaceIndices[triBase + 2u] & 0x3FFFFFFFu];
                uint s0 = c0 & 7u, s1 = c1 & 7u, s2 = c2 & 7u;
                uint q0 = (c0 >> 3u) & 3u, q1 = (c1 >> 3u) & 3u, q2 = (c2 >> 3u) & 3u;

                bool sourceMixed = s0 != s1 || s0 != s2 ||
                                   s0 == 4u || s1 == 4u || s2 == 4u;
                bool confirmationMixed = q0 != q1 || q0 != q2 ||
                                         q0 == 3u || q1 == 3u || q2 == 3u;

                if (sourceMixed && confirmationMixed)
                {
                    bool admissionInternal = s0 == 4u || s1 == 4u || s2 == 4u;
                    bool confirmationInternal = q0 == 3u || q1 == 3u || q2 == 3u;
                    bool hasPending = q0 == 1u || q1 == 1u || q2 == 1u;
                    if (admissionInternal && confirmationInternal)
                    {
                        uint doubleMixedCount =
                            ((s0 == 4u && q0 == 3u) ? 1u : 0u) +
                            ((s1 == 4u && q1 == 3u) ? 1u : 0u) +
                            ((s2 == 4u && q2 == 3u) ? 1u : 0u);
                        if (doubleMixedCount == 0u) return 15u;
                        return 15u + doubleMixedCount;
                    }
                    if (admissionInternal) return 13u;
                    if (confirmationInternal) return 14u;
                    if (hasPending) return 2u;
                    return 1u;
                }
                if (sourceMixed) return 4u;
                if (confirmationMixed) return 5u;
                if (q0 == 2u && s0 == 1u) return 6u;
                if (q0 == 2u && s0 == 2u) return 7u;
                if (q0 == 2u && s0 == 3u) return 8u;
                if (q0 == 2u) return 9u;
                if (q0 == 1u && s0 == 3u) return 10u;
                if (q0 == 1u) return 11u;
                return 12u;
            }

            uint TemporalDiagnosticClass(uint triBase, out uint supportMask)
            {
                uint encoded0 = _SurfaceIndices[triBase];
                uint encoded1 = _SurfaceIndices[triBase + 1u];
                uint encoded2 = _SurfaceIndices[triBase + 2u];
                uint c0 = _VertexAdmissionClass[encoded0 & 0x3FFFFFFFu];
                uint c1 = _VertexAdmissionClass[encoded1 & 0x3FFFFFFFu];
                uint c2 = _VertexAdmissionClass[encoded2 & 0x3FFFFFFFu];
                uint e0 = (c0 >> 18u) & 3u;
                uint e1 = (c1 >> 18u) & 3u;
                uint e2 = (c2 >> 18u) & 3u;
                supportMask = (e0 == 0u ? 1u : 0u) |
                              (e1 == 0u ? 2u : 0u) |
                              (e2 == 0u ? 4u : 0u);
                return encoded0 >> 30u;
            }

            float3 WorldToVoxelUVW(float3 worldPos)
            {
                float3 local = worldPos / gsVoxSize + gsVoxCount.xyz / 2.0;
                return saturate(local / gsVoxCount.xyz);
            }

            half3 ApplyConfidenceViz(half3 color, float3 worldPos)
            {
                if (_RSConfidenceViz < 0.5) return color;
                float3 uvw = WorldToVoxelUVW(worldPos);
                float weight = SAMPLE_TEXTURE3D_LOD(gsVolume, sampler_gsVolume, uvw, 0).g;
                if (abs(weight) < 0.01) return half3(0.5, 0.5, 0.5);
                float disagreement = SAMPLE_TEXTURE3D_LOD(
                    gsConfidence, sampler_gsConfidence, uvw, 0).r;
                if (disagreement < gsConfidenceMidMax) return color;
                if (disagreement < gsConfidenceLowMin) return half3(1.0, 0.85, 0.1);
                return half3(0.45, 0.3, 1.0);
            }

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings output = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

                uint triBase;
                uint sourceIndexPosition;
                uint boundarySourceIndexPosition0;
                uint boundarySourceIndexPosition1;
                uint diagnosticEdge;
                if (_RSTrueLineQuadTopology > 0.5)
                {
                    // InfiniTAM stores one quad as two adjacent triangles. The
                    // repeated first corner distinguishes the two winding layouts.
                    // Ten output vertices draw the four outside edges plus the
                    // real shared diagonal once instead of twice.
                    uint quadBase = (vertID / 10u) * 6u;
                    uint topologyVertex = vertID % 10u;
                    uint topologyEdge = topologyVertex / 2u;
                    bool diagonalEdge = topologyEdge == 4u;
                    bool positiveLayout =
                        (_SurfaceIndices[quadBase] & 0x3FFFFFFFu) ==
                        (_SurfaceIndices[quadBase + 3u] & 0x3FFFFFFFu);
                    uint sourceSlot;
                    if (positiveLayout)
                    {
                        sourceSlot = topologyVertex == 0u ? 0u :
                                     topologyVertex == 1u ? 4u :
                                     topologyVertex == 2u ? 4u :
                                     topologyVertex == 3u ? 5u :
                                     topologyVertex == 4u ? 5u :
                                     topologyVertex == 5u ? 2u :
                                     topologyVertex == 6u ? 2u :
                                     topologyVertex == 7u ? 3u :
                                     topologyVertex == 8u ? 0u : 1u;
                        triBase = diagonalEdge || topologyEdge >= 2u
                            ? quadBase : quadBase + 3u;
                        diagnosticEdge = diagonalEdge ? 0u :
                                         topologyEdge == 0u ? 0u :
                                         topologyEdge == 1u ? 1u :
                                         topologyEdge == 2u ? 1u : 2u;
                        uint boundarySlot0 = topologyEdge == 0u ? 0u :
                                             topologyEdge == 1u ? 4u :
                                             topologyEdge == 2u ? 5u :
                                             topologyEdge == 3u ? 2u : 0u;
                        uint boundarySlot1 = topologyEdge == 0u ? 4u :
                                             topologyEdge == 1u ? 5u :
                                             topologyEdge == 2u ? 2u :
                                             topologyEdge == 3u ? 3u : 1u;
                        boundarySourceIndexPosition0 = quadBase + boundarySlot0;
                        boundarySourceIndexPosition1 = quadBase + boundarySlot1;
                    }
                    else
                    {
                        sourceSlot = topologyVertex == 0u ? 0u :
                                     topologyVertex == 1u ? 1u :
                                     topologyVertex == 2u ? 1u :
                                     topologyVertex == 3u ? 2u :
                                     topologyVertex == 4u ? 2u :
                                     topologyVertex == 5u ? 3u :
                                     topologyVertex == 6u ? 3u :
                                     topologyVertex == 7u ? 4u :
                                     topologyVertex == 8u ? 0u : 2u;
                        triBase = diagonalEdge || topologyEdge < 2u
                            ? quadBase : quadBase + 3u;
                        diagnosticEdge = diagonalEdge ? 2u :
                                         topologyEdge == 0u ? 0u :
                                         topologyEdge == 1u ? 1u :
                                         topologyEdge == 2u ? 2u : 0u;
                        uint boundarySlot0 = topologyEdge == 0u ? 0u :
                                             topologyEdge == 1u ? 1u :
                                             topologyEdge == 2u ? 2u :
                                             topologyEdge == 3u ? 3u : 0u;
                        uint boundarySlot1 = topologyEdge == 0u ? 1u :
                                             topologyEdge == 1u ? 2u :
                                             topologyEdge == 2u ? 3u :
                                             topologyEdge == 3u ? 4u : 2u;
                        boundarySourceIndexPosition0 = quadBase + boundarySlot0;
                        boundarySourceIndexPosition1 = quadBase + boundarySlot1;
                    }
                    sourceIndexPosition = quadBase + sourceSlot;
                }
                else
                {
                    triBase = (vertID / 6u) * 3u;
                    uint lineVertex = vertID % 6u;
                    uint sourceCorner = lineVertex == 0u ? 0u :
                                        lineVertex == 1u ? 1u :
                                        lineVertex == 2u ? 1u :
                                        lineVertex == 3u ? 2u :
                                        lineVertex == 4u ? 2u : 0u;
                    sourceIndexPosition = triBase + sourceCorner;
                    diagnosticEdge = lineVertex / 2u;
                    uint edge = lineVertex / 2u;
                    boundarySourceIndexPosition0 = triBase +
                        (edge == 0u ? 0u : edge == 1u ? 1u : 2u);
                    boundarySourceIndexPosition1 = triBase +
                        (edge == 0u ? 1u : edge == 1u ? 2u : 0u);
                }

                output.boundaryLineClass = _RSBoundaryUnificationDiagnostic > 0.5
                    ? BoundaryLineClass(boundarySourceIndexPosition0,
                                        boundarySourceIndexPosition1)
                    : 0u;

                uint meshStride = (uint)max(_RSMeshStride, 1.0);
                if (meshStride > 1u)
                {
                    uint anchorIndex = _SurfaceIndices[triBase] & 0x3FFFFFFFu;
                    uint anchorFlat = _SurfaceVerts[anchorIndex].voxelFlatIdx;
                    uint3 voxelCount = (uint3)gsVoxCount.xyz;
                    uint sliceXY = voxelCount.x * voxelCount.y;
                    uint vz = anchorFlat / sliceXY;
                    uint remainder = anchorFlat - vz * sliceXY;
                    uint vy = remainder / voxelCount.x;
                    uint vx = remainder - vy * voxelCount.x;
                    bool onGrid = vx % meshStride == 0u ||
                                  vy % meshStride == 0u ||
                                  vz % meshStride == 0u;
                    if (!onGrid)
                    {
                        output.positionHCS = float4(2.0, 2.0, 2.0, 1.0);
                        return output;
                    }
                }

                uint encoded = _SurfaceIndices[sourceIndexPosition];
                uint index = encoded & 0x3FFFFFFFu;
                float3 positionWS = _SurfaceVerts[index].pos;
                output.positionWS = positionWS;
                output.positionHCS = TransformWorldToHClip(positionWS);
                float3 viewDelta = positionWS - _WorldSpaceCameraPos.xyz;
                float viewDistanceSq = dot(viewDelta, viewDelta);
                // Keep the visible line just 0.75 mm in front of its own
                // depth-only shell. Moving along the eye ray changes depth but
                // not screen position, so it removes angle-dependent self-
                // occlusion without widening or reshaping the mesh. Genuine
                // rear sheets remain far behind this sub-millimetre allowance.
                if (viewDistanceSq > 1e-8)
                {
                    const float lineDepthBiasMeters = 0.00075;
                    float3 biasedPositionWS = positionWS - viewDelta *
                        (lineDepthBiasMeters * rsqrt(viewDistanceSq));
                    output.positionHCS = TransformWorldToHClip(biasedPositionWS);
                }
                output.legacyDiagnosticClass = 12u;
                output.diagnosticClass = 2u;
                output.diagnosticSupportMask = 7u;
                if (_RSJointDiagnostic > 0.5)
                {
                    output.diagnosticClass = TemporalDiagnosticClass(
                        triBase, output.diagnosticSupportMask);
                }
                else if (_RSHeraReplayActive > 0.5)
                {
                    // HERA colour routing needs only the triangle's two high
                    // index bits; avoid the admission-buffer support audit used
                    // exclusively by the joint diagnostic view.
                    output.diagnosticClass = _SurfaceIndices[triBase] >> 30u;
                }
                else if (_RSSuppressPink > 0.5 &&
                         _RSBoundaryUnificationDiagnostic < 0.5)
                {
                    output.legacyDiagnosticClass = LegacyDiagnosticClass(triBase);
                }
                output.edgeIndex = diagnosticEdge;
                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

                if (_RSHeraReplayActive < 0.5 && _RSJointDiagnostic < 0.5 &&
                    _RSBoundaryUnificationDiagnostic < 0.5 &&
                    _RSSuppressPink > 0.5 && input.legacyDiagnosticClass == 5u)
                    discard;

                if (_RSJointDiagnostic > 0.5 && input.diagnosticClass == 0u)
                    discard;

                if (_RSJointDiagnostic > 0.5 && input.diagnosticClass == 1u)
                {
                    bool supported =
                        (input.edgeIndex == 0u && input.diagnosticSupportMask == 3u) ||
                        (input.edgeIndex == 1u && input.diagnosticSupportMask == 6u) ||
                        (input.edgeIndex == 2u && input.diagnosticSupportMask == 5u);
                    if (!supported)
                        discard;
                }

                bool heraDelegated =
                    _RSHeraReplayActive > 0.5 && input.diagnosticClass == 3u;
                half3 color = _RSHeraReplayActive > 0.5
                    ? (heraDelegated
                        ? half3(1.0, 0.18, 0.32)
                        : half3(0.12, 1.0, 0.28))
                    : _RSJointDiagnostic > 0.5
                        ? half3(0.10, 1.0, 0.25)
                        : _RSExtractionColor.rgb;
                if (_RSBoundaryUnificationDiagnostic > 0.5)
                {
                    color = input.boundaryLineClass == 2u
                        ? half3(1.0, 0.04, 0.08)
                        : input.boundaryLineClass == 1u
                            ? half3(0.05, 1.0, 0.20)
                            : half3(0.20, 0.55, 1.0);
                }
                else
                {
                    color = _RSGeometryTruthView > 0.5
                        ? half3(1.0, 1.0, 1.0)
                        : ApplyConfidenceViz(color, input.positionWS);
                }
                return half4(color, 1.0);
            }
            ENDHLSL
        }
    }
}
