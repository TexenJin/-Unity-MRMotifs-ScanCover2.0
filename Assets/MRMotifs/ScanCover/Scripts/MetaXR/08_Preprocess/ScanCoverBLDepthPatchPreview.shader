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
                    return float4(0.08, 0.0, 0.0, _Alpha);

                float value = saturate(depth / max(0.0001, _DepthScaleMeters));
                value = pow(value, 1.25);

                float3 color;
                if (value < 0.25)
                {
                    color = lerp(float3(0.0, 0.08, 0.85), float3(0.0, 0.9, 1.0), value / 0.25);
                }
                else if (value < 0.5)
                {
                    color = lerp(float3(0.0, 0.9, 1.0), float3(0.0, 1.0, 0.22), (value - 0.25) / 0.25);
                }
                else if (value < 0.75)
                {
                    color = lerp(float3(0.0, 1.0, 0.22), float3(1.0, 0.9, 0.0), (value - 0.5) / 0.25);
                }
                else
                {
                    color = lerp(float3(1.0, 0.9, 0.0), float3(1.0, 0.05, 0.0), (value - 0.75) / 0.25);
                }

                float bandPhase = abs(frac(depth / 0.5) - 0.5) * 2.0;
                float bandLine = 1.0 - smoothstep(0.0, 0.08, bandPhase);
                color = lerp(color, color * 0.28, bandLine);

                return float4(color, _Alpha);
            }
            ENDHLSL
        }
    }

    FallBack Off
}
