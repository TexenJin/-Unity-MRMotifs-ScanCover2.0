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

            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite On
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
            float _RSGeometryTruthView;
            float4 _RSExtractionColor;
            float _RSJointDiagnostic;
            float _RSSuppressPink;
            float _RSHeraReplayActive;

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                nointerpolation uint diagnosticClass : TEXCOORD1;
                nointerpolation uint legacyDiagnosticClass : TEXCOORD2;
                nointerpolation uint diagnosticSupportMask : TEXCOORD3;
                nointerpolation uint edgeIndex : TEXCOORD4;
                UNITY_VERTEX_OUTPUT_STEREO
            };

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

                uint triBase = (vertID / 6u) * 3u;
                uint lineVertex = vertID % 6u;
                uint sourceCorner = lineVertex == 0u ? 0u :
                                    lineVertex == 1u ? 1u :
                                    lineVertex == 2u ? 1u :
                                    lineVertex == 3u ? 2u :
                                    lineVertex == 4u ? 2u : 0u;

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

                uint encoded = _SurfaceIndices[triBase + sourceCorner];
                uint index = encoded & 0x3FFFFFFFu;
                float3 positionWS = _SurfaceVerts[index].pos;
                output.positionWS = positionWS;
                output.positionHCS = TransformWorldToHClip(positionWS);
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
                else if (_RSSuppressPink > 0.5)
                {
                    output.legacyDiagnosticClass = LegacyDiagnosticClass(triBase);
                }
                output.edgeIndex = lineVertex / 2u;
                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

                if (_RSHeraReplayActive < 0.5 && _RSJointDiagnostic < 0.5 &&
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
                color = _RSGeometryTruthView > 0.5
                    ? half3(0.95, 0.95, 0.95)
                    : ApplyConfidenceViz(color, input.positionWS);
                return half4(color, _RSExtractionColor.a);
            }
            ENDHLSL
        }
    }
}
