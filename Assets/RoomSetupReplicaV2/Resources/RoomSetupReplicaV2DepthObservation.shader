Shader "RoomSetupReplicaV2/ReadOnlyDepthObservation"
{
    Properties
    {
        _HasDepth ("Depth Available", Float) = 0
    }

    SubShader
    {
        Tags
        {
            "Queue" = "Overlay+10"
            "RenderType" = "Transparent"
            "RenderPipeline" = "UniversalPipeline"
        }
        Pass
        {
            Name "ReadOnlyDepthObservation"
            Cull Off
            ZWrite Off
            ZTest Always
            Blend SrcAlpha OneMinusSrcAlpha

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma multi_compile_instancing
            #pragma multi_compile _ STEREO_INSTANCING_ON STEREO_MULTIVIEW_ON

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            Texture2DArray<float> _EnvironmentDepthTexture;
            SamplerState sampler_EnvironmentDepthTexture;
            float4 _EnvironmentDepthZBufferParams;
            float _HasDepth;

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
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                output.positionCS = TransformObjectToHClip(input.positionOS.xyz);
                output.uv = input.uv;
                return output;
            }

            bool TryMetricDepth(float2 uv, int slice, out float depth)
            {
                depth = 0.0;
                float encoded = _EnvironmentDepthTexture.SampleLevel(
                    sampler_EnvironmentDepthTexture,
                    float3(saturate(uv), slice), 0).r;
                if (!(encoded > 0.0 && encoded < 1.0))
                    return false;

                float depthNdc = encoded * 2.0 - 1.0;
                float denominator =
                    depthNdc + _EnvironmentDepthZBufferParams.y;
                if (abs(denominator) <= 0.000001)
                    return false;

                depth = _EnvironmentDepthZBufferParams.x / denominator;
                return depth == depth && depth >= 0.30 && depth <= 3.00;
            }

            float3 DepthRamp(float depth)
            {
                float t = saturate((depth - 0.30) / 2.70);
                float3 nearColor = float3(1.0, 0.16, 0.03);
                float3 middleColor = float3(0.03, 0.92, 0.52);
                float3 farColor = float3(0.04, 0.30, 1.0);
                float3 color = t < 0.5
                    ? lerp(nearColor, middleColor, t * 2.0)
                    : lerp(middleColor, farColor, (t - 0.5) * 2.0);

                // Subtle 10 cm contours make small depth motion visible
                // without modifying the sampled value.
                float contourPhase =
                    abs(frac(depth * 10.0) - 0.5) * 2.0;
                float contour = smoothstep(0.90, 1.0, contourPhase);
                return lerp(color, color * 0.58, contour * 0.42);
            }

            half4 Frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                float2 panelUv = input.uv;
                float borderDistance = min(
                    min(panelUv.x, 1.0 - panelUv.x),
                    min(panelUv.y, 1.0 - panelUv.y));
                if (borderDistance < 0.012)
                    return half4(0.08, 0.92, 1.0, 0.96);

                float divider = abs(panelUv.x - 0.5);
                if (divider < 0.004)
                    return half4(0.08, 0.92, 1.0, 0.90);

                if (_HasDepth < 0.5)
                    return half4(0.015, 0.02, 0.025, 0.86);

                int slice = panelUv.x < 0.5 ? 0 : 1;
                float2 depthUv = float2(frac(panelUv.x * 2.0), panelUv.y);
                float depth;
                if (!TryMetricDepth(depthUv, slice, depth))
                    return half4(0.0, 0.0, 0.0, 0.94);

                return half4(DepthRamp(depth), 0.94);
            }
            ENDHLSL
        }
    }
    Fallback Off
}
