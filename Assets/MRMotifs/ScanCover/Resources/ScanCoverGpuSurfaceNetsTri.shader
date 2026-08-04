Shader "ScanCover/GpuSurfaceNetsTri"
{
    Properties
    {
        _SurfaceColor("Surface Color", Color) = (0.18, 0.58, 1.0, 0.85)
    }
    SubShader
    {
        Tags { "Queue"="Transparent+30" "RenderType"="Transparent" "RenderPipeline"="UniversalPipeline" }
        Pass
        {
            Name "GpuSurfaceNetsTri"
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            ZTest LEqual
            Cull Off

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma target 4.5
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            // Must match GPUVertex in ScanCoverGpuSurfaceNets.compute (32B).
            struct SurfaceVertex
            {
                float3 position;
                float3 normal;
                uint voxelIndex;
                uint padding;
            };

            StructuredBuffer<SurfaceVertex> _Vertices;
            StructuredBuffer<uint> _Indices;
            float4x4 _VolumeLocalToWorld;
            float4 _SurfaceColor;

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 normalWS : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings Vert(uint vertexID : SV_VertexID)
            {
                Varyings output = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                uint surfaceIndex = _Indices[vertexID];
                SurfaceVertex vertex = _Vertices[surfaceIndex];
                float3 worldPosition = mul(_VolumeLocalToWorld, float4(vertex.position, 1.0)).xyz;
                float3 worldNormal = normalize(mul((float3x3)_VolumeLocalToWorld, vertex.normal));
                output.positionCS = TransformWorldToHClip(worldPosition);
                output.normalWS = worldNormal;
                return output;
            }

            half4 Frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                float3 n = normalize(input.normalWS);
                // Two-sided lambert against a fixed soft key light + ambient,
                // so curvature/waviness stays readable in-headset.
                float3 lightDir = normalize(float3(0.4, 0.85, 0.35));
                float lambert = abs(dot(n, lightDir));
                float shade = 0.35 + 0.65 * lambert;
                return half4(_SurfaceColor.rgb * shade, _SurfaceColor.a);
            }
            ENDHLSL
        }
    }
}
