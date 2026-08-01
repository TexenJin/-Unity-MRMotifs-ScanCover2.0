Shader "ScanCover/QuestRoomSurfaceNetsWire"
{
    Properties
    {
        _WireColor("Wire Color", Color) = (1.0, 0.62, 0.02, 1.0)
    }
    SubShader
    {
        Tags { "Queue"="Transparent+40" "RenderType"="Transparent" "RenderPipeline"="UniversalPipeline" }
        Pass
        {
            Name "QuestRoomSurfaceNetsWire"
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            ZTest LEqual
            Cull Off

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma target 4.5
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct SurfaceVertex
            {
                float3 position;
                float3 normal;
                uint voxelIndex;
                uint padding;
            };

            StructuredBuffer<SurfaceVertex> _SurfaceVertices;
            StructuredBuffer<uint> _SurfaceLineIndices;
            float4x4 _VolumeLocalToWorld;
            float4 _WireColor;

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float4 color : COLOR0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings Vert(uint vertexID : SV_VertexID)
            {
                Varyings output = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                uint surfaceIndex = _SurfaceLineIndices[vertexID];
                SurfaceVertex vertex = _SurfaceVertices[surfaceIndex];
                float3 worldPosition = mul(_VolumeLocalToWorld, float4(vertex.position, 1.0)).xyz;
                output.positionCS = TransformWorldToHClip(worldPosition);
                output.color = _WireColor;
                return output;
            }

            half4 Frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                return input.color;
            }
            ENDHLSL
        }
    }
}
