Shader "Hidden/ScanCover/BLDepthPatchPreview"
{
    Properties
    {
        _DepthTex ("Depth Texture", 2D) = "black" {}
        _DepthScaleMeters ("Depth Scale Meters", Float) = 1.2
        _Alpha ("Alpha", Range(0, 1)) = 0.82
    }

    SubShader
    {
        Tags { "RenderType" = "Transparent" "Queue" = "Overlay" }
        Pass
        {
            ZWrite Off
            ZTest Always
            Cull Off
            Blend SrcAlpha OneMinusSrcAlpha

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag

            #include "UnityCG.cginc"

            sampler2D _DepthTex;
            float _DepthScaleMeters;
            float _Alpha;

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings Vert(Attributes input)
            {
                Varyings output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_OUTPUT(Varyings, output);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                output.positionCS = UnityObjectToClipPos(input.positionOS);
                output.uv = input.uv;
                return output;
            }

            float4 Frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                float depth = tex2D(_DepthTex, input.uv).r;
                if (depth <= 0.0001)
                    return float4(0.0, 0.0, 0.0, 0.18 * _Alpha);

                float normalizedDepth = saturate(depth / max(0.0001, _DepthScaleMeters));
                float value = 1.0 - normalizedDepth;
                float3 nearColor = float3(0.08, 1.0, 0.78);
                float3 farColor = float3(0.12, 0.25, 1.0);
                float3 color = lerp(farColor, nearColor, value);
                return float4(color, _Alpha);
            }
            ENDHLSL
        }
    }

    FallBack Off
}
