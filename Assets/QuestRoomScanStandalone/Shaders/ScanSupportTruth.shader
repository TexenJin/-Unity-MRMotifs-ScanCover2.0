Shader "Genesis/ScanSupportTruth"
{
    Properties { }
    SubShader
    {
        // Paper topology is an isolated foreground route. HERA is hidden while
        // this pass is visible, so no polygon offset is needed to manufacture a
        // second visual layer between the two carriers.
        Tags { "RenderType"="Transparent" "RenderPipeline"="UniversalPipeline" "Queue"="Transparent+35" }

        Pass
        {
            Name "SupportTopologyAudit"
            Tags { "LightMode"="SRPDefaultUnlit" }
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite On
            ZTest LEqual
            Cull Off
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            StructuredBuffer<float4> _SupportPoints;
            StructuredBuffer<float4> _SupportNormals;
            StructuredBuffer<float4> _SupportTopologyPoints;
            StructuredBuffer<float4> _SupportTopologyNormals;
            StructuredBuffer<uint> _SupportTopologyIndices;
            float4 _SupportColor;
            float4 _SupportPaperColor;
            float _SupportSurfelRadius;
            float _SupportAuditMode;
            float _SupportPaperGrid;
            float _SupportTopologyMesh;
            float _RSGridSpacing;
            float _RSWireThickness;

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 discUV : TEXCOORD0;
                nointerpolation float topologyClass : TEXCOORD1;
                float3 positionWS : TEXCOORD2;
                float3 normalWS : TEXCOORD3;
                float3 topologyBarycentric : TEXCOORD4;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            static const float2 kCorners[6] = {
                float2(-1,-1), float2(-1, 1), float2( 1, 1),
                float2(-1,-1), float2( 1, 1), float2( 1,-1)
            };

            Varyings vert(uint vertexID : SV_VertexID)
            {
                Varyings OUT = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);

                bool topologyMesh = _SupportTopologyMesh > 0.5;
                uint pointIndex = topologyMesh
                    ? _SupportTopologyIndices[vertexID]
                    : vertexID / 6u;
                float2 corner = topologyMesh
                    ? float2(0.0, 0.0)
                    : kCorners[vertexID % 6u];
                float3 positionWS = topologyMesh
                    ? _SupportTopologyPoints[pointIndex].xyz
                    : _SupportPoints[pointIndex].xyz;
                float4 normalAndClass = topologyMesh
                    ? _SupportTopologyNormals[pointIndex]
                    : _SupportNormals[pointIndex];
                float3 normalWS = normalize(normalAndClass.xyz);
                float3 surfelPositionWS = positionWS;
                if (!topologyMesh)
                {
                    float3 referenceAxis = abs(normalWS.y) < 0.9
                        ? float3(0, 1, 0)
                        : float3(1, 0, 0);
                    float3 tangentWS = normalize(cross(referenceAxis, normalWS));
                    float3 bitangentWS = cross(normalWS, tangentWS);
                    surfelPositionWS +=
                        (tangentWS * corner.x + bitangentWS * corner.y) * _SupportSurfelRadius;
                }

                OUT.positionHCS = TransformWorldToHClip(surfelPositionWS);
                OUT.discUV = corner;
                OUT.topologyClass = normalAndClass.w;
                OUT.positionWS = surfelPositionWS;
                OUT.normalWS = normalWS;
                uint topologyCorner = vertexID % 3u;
                OUT.topologyBarycentric = topologyCorner == 0u
                    ? float3(1.0, 0.0, 0.0)
                    : topologyCorner == 1u
                        ? float3(0.0, 1.0, 0.0)
                        : float3(0.0, 0.0, 1.0);
                return OUT;
            }

            float2 PaperGridUV(float3 positionWS, float3 normalWS)
            {
                float3 a = abs(normalize(normalWS));
                if (a.x >= a.y && a.x >= a.z) return positionWS.zy;
                if (a.y >= a.z) return positionWS.xz;
                return positionWS.xy;
            }

            float PaperTriangleGrid(float3 positionWS, float3 normalWS)
            {
                float spacing = max(_RSGridSpacing, 0.04);
                float2 p = PaperGridUV(positionWS, normalWS) / spacing;
                float3 family = float3(
                    p.x,
                    0.5 * p.x + 0.8660254 * p.y,
                   -0.5 * p.x + 0.8660254 * p.y);
                float3 distanceToLine = abs(frac(family + 0.5) - 0.5);
                float3 aa = max(fwidth(family) * max(_RSWireThickness, 0.55), 0.0008);
                float3 lineCoverage = 1.0 - smoothstep(aa, aa * 1.8, distanceToLine);
                return max(lineCoverage.x, max(lineCoverage.y, lineCoverage.z));
            }

            half3 TopologyColor(float topologyClass)
            {
                if (topologyClass > 2.5) return half3(1.00, 0.05, 0.85); // stacked layer
                if (topologyClass > 1.5) return half3(0.05, 1.00, 0.22); // connected
                if (topologyClass > 0.5) return half3(1.00, 0.72, 0.05); // fragile
                return half3(1.00, 0.08, 0.04);                          // exposed
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);
                float radius = length(IN.discUV);
                bool topologyMesh = _SupportTopologyMesh > 0.5;
                if (!topologyMesh)
                {
                    if (radius > 1.0)
                        discard;

                    half3 topologyColor = TopologyColor(IN.topologyClass);
                    if (_SupportAuditMode > 0.5)
                    {
                        // Independent-crossing audit remains available as the
                        // raw-material view. It is no longer the paper itself.
                        if (radius < 0.13)
                            return half4(_SupportColor.rgb, 1.0);
                        if (radius > 0.78)
                            return half4(topologyColor, 0.96);
                        return half4(_SupportColor.rgb, 0.12);
                    }
                }

                // Paper mode is now a real locally connected triangle carrier.
                // The raw Surfels stay in GPU buffers as shared vertices, while
                // only the resolved topology reaches this fill/grid pass. Show
                // its real barycentric edges: a synthetic world grid would hide
                // the very connectivity this route is meant to audit.
                half3 paper = _SupportPaperColor.rgb;
                half paperAlpha = saturate(_SupportPaperColor.a);
                if (topologyMesh)
                {
                    float thickness = max(_RSWireThickness, 0.55);
                    float3 bary = IN.topologyBarycentric;
                    float3 dx = ddx(bary);
                    float3 dy = ddy(bary);
                    float3 edgeWidth = max(sqrt(dx * dx + dy * dy), 0.0005);
                    float3 edgeRamp = smoothstep(0.0, edgeWidth * thickness, bary);
                    float edgeCoverage = 1.0 - min(edgeRamp.x,
                        min(edgeRamp.y, edgeRamp.z));
                    paper = lerp(paper, half3(0.96, 0.98, 1.0),
                        saturate(edgeCoverage));
                    // During probe acquisition the fill becomes transparent,
                    // while true topology edges remain legible over passthrough.
                    paperAlpha = lerp(paperAlpha, 0.94h,
                        saturate(edgeCoverage));
                }
                return half4(paper, paperAlpha);
            }
            ENDHLSL
        }
    }
}
