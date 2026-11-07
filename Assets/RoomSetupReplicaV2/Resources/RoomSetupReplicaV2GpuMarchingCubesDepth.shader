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

            struct Attributes
            {
                uint vertexID : SV_VertexID;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
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
                output.positionHCS = TransformWorldToHClip(positionWS);
                return output;
            }

            half4 Frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                return 0;
            }
            ENDHLSL
        }
    }
}
