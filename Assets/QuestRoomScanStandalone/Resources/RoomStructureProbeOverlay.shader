// Read-only room-structure probe overlay.  Kept in Resources so Android shader
// stripping cannot remove it; it is independent from production mesh materials.
Shader "QRS/RoomStructureProbeOverlay"
{
    Properties
    {
        _LineColor ("Structure Plane Color", Color) = (0.1, 1.0, 0.35, 0.22)
    }
    SubShader
    {
        Tags
        {
            "RenderType"="Transparent"
            "Queue"="Transparent+95"
            "RenderPipeline"="UniversalPipeline"
        }
        Pass
        {
            Name "RoomStructureProbe"
            Tags { "LightMode"="SRPDefaultUnlit" }
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            // Mode 5 is an evidence overlay.  It must remain readable even
            // when the production/occlusion material has already written the
            // depth buffer, so it intentionally renders as an X-ray layer.
            ZTest Always
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            CBUFFER_START(UnityPerMaterial)
                half4 _LineColor;
            CBUFFER_END

            Varyings vert(Attributes input)
            {
                Varyings output = (Varyings)0;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                output.positionHCS = TransformObjectToHClip(input.positionOS.xyz);
                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                return _LineColor;
            }
            ENDHLSL
        }
    }
}
