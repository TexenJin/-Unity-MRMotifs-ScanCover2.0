Shader "Hidden/ScanCover/RawDepthDebugWindow"
{
    Properties
    {
        _EyeIndex ("Eye Index", Float) = 1
        _RawDepthDisplayScale ("Linear Depth Display Scale", Float) = 1
        _Alpha ("Alpha", Range(0, 1)) = 0.86
        _ShowPixelCenterDots ("Show Pixel Center Dots", Float) = 0
        _PixelCenterDotRadius ("Pixel Center Dot Radius", Range(0.02, 0.48)) = 0.18
        _PixelCenterDotAlpha ("Pixel Center Dot Alpha", Range(0, 1)) = 0.85
        _PixelCenterDotColor ("Pixel Center Dot Color", Color) = (1, 1, 1, 1)
    }

    SubShader
    {
        Tags { "RenderType" = "Transparent" "Queue" = "Overlay" }
        Pass
        {
            ZWrite Off
            ZTest Always
            Cull Off
            Blend One Zero

            HLSLPROGRAM
            #pragma target 3.0
            #pragma vertex Vert
            #pragma fragment Frag

            #include "UnityCG.cginc"

            Texture2DArray<float> _EnvironmentDepthTexture;
            SamplerState sampler_EnvironmentDepthTexture;
            float4 _EnvironmentDepthZBufferParams;
            float _EyeIndex;
            float _RawDepthDisplayScale;
            float _Alpha;
            float4 _RawDepthTextureSize;
            float _ShowPixelCenterDots;
            float _PixelCenterDotRadius;
            float _PixelCenterDotAlpha;
            float4 _PixelCenterDotColor;

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

                float eye = clamp(round(_EyeIndex), 0.0, 1.0);
                float raw = _EnvironmentDepthTexture.SampleLevel(
                    sampler_EnvironmentDepthTexture,
                    float3(saturate(input.uv), eye),
                    0).r;

                bool validDepth = raw > 0.00001;
                if (!validDepth)
                    return float4(0.08, 0.0, 0.0, 1.0);

                float linearMeters = (_EnvironmentDepthZBufferParams.x / (raw + _EnvironmentDepthZBufferParams.y));
                float value = saturate(linearMeters * max(0.001, _RawDepthDisplayScale));
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
                if (_ShowPixelCenterDots > 0.5 && _RawDepthTextureSize.x > 1.0 && _RawDepthTextureSize.y > 1.0)
                {
                    float2 pixelCoord = saturate(input.uv) * _RawDepthTextureSize.xy;
                    float2 cell = frac(pixelCoord) - 0.5;
                    float dotDistance = length(cell);
                    float dotMask = 1.0 - smoothstep(
                        _PixelCenterDotRadius,
                        min(0.5, _PixelCenterDotRadius + 0.08),
                        dotDistance);
                    color = lerp(color, _PixelCenterDotColor.rgb, saturate(dotMask * _PixelCenterDotAlpha));
                }

                return float4(saturate(color), 1.0);
            }
            ENDHLSL
        }
    }

    FallBack Off
}
