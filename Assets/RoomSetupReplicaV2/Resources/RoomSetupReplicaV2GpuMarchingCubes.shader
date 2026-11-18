Shader "Genesis/RoomSetupReplicaV2GpuMarchingCubes"
{
    Properties
    {
        _BaseColor ("Color", Color) = (0.02, 0.86, 1.0, 0.88)
        _LineWidthPixels ("Line Width", Range(0.75, 3.0)) = 1.5
        _LineReferenceDistance ("Full Width Distance", Float) = 0.5
        _UniqueEdgesOnly ("Unique Edges Only", Float) = 0
    }
    SubShader
    {
        Tags
        {
            "RenderType"="Transparent"
            "Queue"="Transparent+45"
            "RenderPipeline"="UniversalPipeline"
        }
        Pass
        {
            Name "RoomSetupGpuMcWire"
            Tags { "LightMode"="SRPDefaultUnlit" }
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            ZTest LEqual
            // Pull the screen-space strip slightly toward the camera. At sharp
            // corners the adjacent triangle depth prepass can otherwise
            // consume its covered MSAA samples even though the edge is visible.
            Offset -1, -1
            Cull Off

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma target 4.5
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            StructuredBuffer<float3> _RoomSetupMcCandidateVertices;
            StructuredBuffer<float3> _RoomSetupMcFilteredVertices;
            StructuredBuffer<float> _RoomSetupMcCandidateStability;
            StructuredBuffer<float> _RoomSetupMcFilteredStability;
            StructuredBuffer<uint> _RoomSetupMcCounters;
            StructuredBuffer<uint> _RoomSetupMcCompactDisplayEdges;

            struct Attributes
            {
                uint vertexID : SV_VertexID;
                // Explicitly keep SV_InstanceID in the non-instancing compile
                // variant too. UNITY_SETUP_INSTANCE_ID still decodes it for
                // single-pass stereo when XR instancing is enabled.
                uint instanceID : SV_InstanceID;
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float edgeCoordinate : TEXCOORD0;
                nointerpolation float stability : TEXCOORD1;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            CBUFFER_START(UnityPerMaterial)
                half4 _BaseColor;
                float _LineWidthPixels;
                float _LineReferenceDistance;
                float _UniqueEdgesOnly;
            CBUFFER_END

            float3 LoadPosition(uint vertexID)
            {
                return _RoomSetupMcCounters[6] != 0u
                    ? _RoomSetupMcFilteredVertices[vertexID]
                    : _RoomSetupMcCandidateVertices[vertexID];
            }

            float LoadStability(uint vertexID)
            {
                return _RoomSetupMcCounters[6] != 0u
                    ? _RoomSetupMcFilteredStability[vertexID]
                    : _RoomSetupMcCandidateStability[vertexID];
            }

            Varyings Vert(Attributes input)
            {
                Varyings output = (Varyings)0;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                uint displayEdgeIndex = input.instanceID;
                #if UNITY_ANY_INSTANCING_ENABLED
                    displayEdgeIndex = unity_InstanceID;
                #endif
                uint encodedEdge =
                    _RoomSetupMcCompactDisplayEdges[displayEdgeIndex];
                uint triangleIndex = encodedEdge / 3u;
                uint edgeIndex = encodedEdge - triangleIndex * 3u;
                uint cornerA = edgeIndex;
                uint cornerB = edgeIndex == 2u ? 0u : edgeIndex + 1u;
                uint triangleBase = triangleIndex * 3u;
                float3 positionA = LoadPosition(triangleBase + cornerA);
                float3 positionB = LoadPosition(triangleBase + cornerB);
                output.stability = min(
                    LoadStability(triangleBase + cornerA),
                    LoadStability(triangleBase + cornerB));
                float4 clipA = TransformWorldToHClip(positionA);
                float4 clipB = TransformWorldToHClip(positionB);
                float2 ndcA = clipA.xy / max(abs(clipA.w), 0.00001);
                float2 ndcB = clipB.xy / max(abs(clipB.w), 0.00001);
                float2 pixelDelta = (ndcB - ndcA) *
                    max(_ScreenParams.xy, 1.0) * 0.5;
                float pixelLength = max(length(pixelDelta), 0.0001);
                float2 normalPixels =
                    float2(-pixelDelta.y, pixelDelta.x) / pixelLength;

                // Two triangles with corners A-, A+, B-, B-, A+, B+.
                // Boolean expressions avoid dynamic local-array indexing on
                // the Quest mobile shader compiler.
                uint quadVertex = min(input.vertexID, 5u);
                bool useB = quadVertex == 2u || quadVertex == 3u ||
                            quadVertex == 5u;
                float side = (quadVertex == 0u || quadVertex == 2u ||
                              quadVertex == 3u) ? -1.0 : 1.0;
                output.positionHCS = useB ? clipB : clipA;
                float rasterWidth = max(_LineWidthPixels, 1.0);
                float2 offsetNdc = normalPixels * side * rasterWidth /
                                   max(_ScreenParams.xy, 1.0);
                output.positionHCS.xy += offsetNdc * output.positionHCS.w;
                output.edgeCoordinate = side;
                return output;
            }

            half4 Frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                float edge = abs(input.edgeCoordinate);
                float feather = max(fwidth(edge), 0.02);
                float coverage = 1.0 - smoothstep(
                    1.0 - feather, 1.0, edge);
                clip(coverage - 0.01);
                half3 changing = half3(1.0, 0.05, 0.03);
                half3 settling = half3(1.0, 0.75, 0.05);
                half3 stabilityColor = input.stability < 0.5
                    ? lerp(changing, settling,
                           saturate(input.stability * 2.0))
                    : lerp(settling, _BaseColor.rgb,
                           saturate((input.stability - 0.5) * 2.0));
                return half4(stabilityColor, _BaseColor.a * coverage);
            }
            ENDHLSL
        }
    }
}
