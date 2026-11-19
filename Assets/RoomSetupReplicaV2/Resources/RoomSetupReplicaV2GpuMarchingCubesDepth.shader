Shader "Genesis/RoomSetupReplicaV2GpuMarchingCubesDepth"
{
    SubShader
    {
        Tags
        {
            "RenderType"="Opaque"
            "Queue"="Transparent+44"
            "RenderPipeline"="UniversalPipeline"
        }
        Pass
        {
            Name "RoomSetupGpuMcDepth"
            Tags { "LightMode"="SRPDefaultUnlit" }
            Blend Off
            ColorMask 0
            ZWrite On
            ZTest LEqual
            Offset 0, 1
            Cull Off

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma target 4.5
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            StructuredBuffer<float3> _RoomSetupMcCandidateVertices;
            StructuredBuffer<float3> _RoomSetupMcFilteredVertices;
            StructuredBuffer<uint> _RoomSetupMcCounters;
            StructuredBuffer<uint> _RoomSetupMcCandidateRevealed;
            StructuredBuffer<uint> _RoomSetupMcFilteredRevealed;

            struct Attributes
            {
                uint vertexID : SV_VertexID;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                nointerpolation float revealed : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings Vert(Attributes input)
            {
                Varyings output = (Varyings)0;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                float3 positionWS = _RoomSetupMcCounters[6] != 0u
                    ? _RoomSetupMcFilteredVertices[input.vertexID]
                    : _RoomSetupMcCandidateVertices[input.vertexID];
                uint triangleIndex = input.vertexID / 3u;
                uint revealed = _RoomSetupMcCounters[6] != 0u
                    ? _RoomSetupMcFilteredRevealed[triangleIndex]
                    : _RoomSetupMcCandidateRevealed[triangleIndex];
                output.positionHCS = TransformWorldToHClip(positionWS);
                output.revealed = revealed != 0u ? 1.0 : 0.0;
                return output;
            }

            half4 Frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                clip(input.revealed - 0.5);
                return 0;
            }
            ENDHLSL
        }
    }
}
