Shader "Custom/MRDepthRevealOverlay_Blit_LinesOnly_DisconMask"
{
    Properties
    {
        _RevealColor ("Line Color", Color) = (0.05, 1.00, 0.95, 1)
        _RevealAlpha ("Line Alpha (Revealed)", Range(0,1)) = 0.85
        _BaseAlpha ("Line Alpha (Unrevealed)", Range(0,1)) = 0.0

        _GridScale ("Grid Scale", Float) = 6.0
        _GridThickness ("Grid Thickness", Range(0.001,0.15)) = 0.03
        _GridIntensity ("Grid Intensity", Range(0,5)) = 1.2
        _GridFwidthMax ("Grid Fwidth Max", Range(0.001,2.0)) = 0.25

        _LineThreshold ("Line Threshold", Range(0,1)) = 0.35
        _LineSoftness ("Line Softness", Range(0.0001,0.5)) = 0.08

        _MinDepth01 ("Min Depth01", Range(0,1)) = 0.001

        _DepthEdgeThresh ("Depth Edge Threshold", Range(0,0.2)) = 0.02
        _DepthEdgeSoft ("Depth Edge Softness", Range(0.0001,0.2)) = 0.02
        _DepthEdgeStrength ("Depth Edge Strength", Range(0,1)) = 1.0

        _FadeStartMeters ("Fade Start (m)", Float) = 0.0
        _FadeEndMeters ("Fade End (m)", Float) = 0.0
    }

    SubShader
    {
        Tags { "RenderPipeline"="UniversalPipeline" "RenderType"="Opaque" "Queue"="Overlay" }

        Pass
        {
            Name "BlitComposite"
            ZWrite Off
            ZTest Always
            Cull Off
            Blend Off

            HLSLPROGRAM
            #pragma target 3.0
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"

            #define REVEAL_MAX_WAVES 32

            CBUFFER_START(UnityPerMaterial)
                float4 _RevealColor;
                float _RevealAlpha;
                float _BaseAlpha;

                float _GridScale;
                float _GridThickness;
                float _GridIntensity;
                float _GridFwidthMax;

                float _LineThreshold;
                float _LineSoftness;

                float _MinDepth01;

                float _DepthEdgeThresh;
                float _DepthEdgeSoft;
                float _DepthEdgeStrength;

                float _FadeStartMeters;
                float _FadeEndMeters;
            CBUFFER_END

            // Meta Environment Depth (raw)
            TEXTURE2D_ARRAY(_EnvironmentDepthTexture);
            SAMPLER(sampler_EnvironmentDepthTexture);
            float4x4 _EnvironmentDepthInverseReprojectionMatrices[2];

            // Reveal globals (fill-only)
            float4 _RevealWaves[REVEAL_MAX_WAVES];
            float _RevealWaveCount;
            float4 _RevealWaveParams; // x=edgeFeather

            int GetEyeIndex()
            {
                int eye = 0;
                #if defined(UNITY_STEREO_INSTANCING_ENABLED) || defined(UNITY_STEREO_MULTIVIEW_ENABLED)
                    eye = unity_StereoEyeIndex;
                #else
                    eye = (int)_BlitTexArraySlice;
                #endif
                return eye;
            }

            float SampleEnvDepth01(float2 uv, int eye)
            {
                uv = saturate(uv);
                return SAMPLE_TEXTURE2D_ARRAY(_EnvironmentDepthTexture, sampler_EnvironmentDepthTexture, uv, eye).r;
            }

            float3 ReconstructWorldPos(float2 uv, float depth01, int eye)
            {
                float4 clip = float4(uv * 2.0 - 1.0, depth01 * 2.0 - 1.0, 1.0);
                float4 w = mul(_EnvironmentDepthInverseReprojectionMatrices[eye], clip);
                return w.xyz / max(1e-6, w.w);
            }

            float ComputeFillReveal(float3 worldPos)
            {
                int count = (int)min(_RevealWaveCount, (float)REVEAL_MAX_WAVES);
                float feather = max(0.0001, _RevealWaveParams.x);

                float w = 0.0;
                [loop]
                for (int i = 0; i < REVEAL_MAX_WAVES; i++)
                {
                    if (i >= count) break;
                    float3 c = _RevealWaves[i].xyz;
                    float r = max(0.0001, _RevealWaves[i].w);
                    float d = distance(worldPos, c);
                    float local = 1.0 - smoothstep(r, r + feather, d);
                    w = max(w, local);
                }
                return saturate(w);
            }

            float ComputeGrid(float3 worldPos)
            {
                float3 p = worldPos * _GridScale;

                // Clamp derivative to prevent fwidth blow-ups on depth discontinuities (thick “ink” lines).
                float3 fw = max(fwidth(p), 1e-4);
                float fwMax = max(1e-4, _GridFwidthMax);
                fw = min(fw, fwMax.xxx);

                float3 fracP = frac(p);
                float3 edgeDist = min(fracP, 1.0 - fracP);
                float thickness = _GridThickness;

                float3 lineAxis = 1.0 - smoothstep(thickness, thickness + fw * 1.5, edgeDist);

                float gridStrength = max(lineAxis.x, max(lineAxis.y, lineAxis.z));
                return saturate(gridStrength * _GridIntensity);
            }

            float ComputeDistanceFade(float3 worldPos)
            {
                if (_FadeEndMeters <= _FadeStartMeters || _FadeEndMeters <= 0.0001) return 1.0;
                float d = distance(worldPos, _WorldSpaceCameraPos);
                return saturate((_FadeEndMeters - d) / max(1e-3, (_FadeEndMeters - _FadeStartMeters)));
            }

            float ComputeDepthDisconConfidence(float2 uv, int eye, float depth01)
            {
                float2 du = float2(1.0 / _ScaledScreenParams.x, 0.0);
                float2 dv = float2(0.0, 1.0 / _ScaledScreenParams.y);

                float dL = SampleEnvDepth01(uv - du, eye);
                float dR = SampleEnvDepth01(uv + du, eye);
                float dD = SampleEnvDepth01(uv - dv, eye);
                float dU = SampleEnvDepth01(uv + dv, eye);

                // Any invalid neighbor => treat as hole and suppress
                float hole = (dL <= _MinDepth01 || dR <= _MinDepth01 || dU <= _MinDepth01 || dD <= _MinDepth01) ? 1.0 : 0.0;

                // Neighbor max absolute difference is a strong discontinuity signal
                float maxDiff = 0.0;
                maxDiff = max(maxDiff, abs(depth01 - dL));
                maxDiff = max(maxDiff, abs(depth01 - dR));
                maxDiff = max(maxDiff, abs(depth01 - dU));
                maxDiff = max(maxDiff, abs(depth01 - dD));

                // Also include derivative-based metric (equivalent to fwidth(depth01) = abs(ddx)+abs(ddy))
                float grad = abs(ddx(depth01)) + abs(ddy(depth01));

                float metric = max(maxDiff, grad);

                float edge = smoothstep(_DepthEdgeThresh, _DepthEdgeThresh + max(0.0001, _DepthEdgeSoft), metric);

                float conf = (1.0 - saturate(edge)) * (1.0 - hole);
                return lerp(1.0, conf, saturate(_DepthEdgeStrength));
            }

            half4 Frag(Varyings input) : SV_Target
            {
                half4 baseCol = FragBlit(input, sampler_LinearClamp);

                int eye = GetEyeIndex();
                float2 uv = input.texcoord;

                float depth01 = SampleEnvDepth01(uv, eye);
                if (depth01 <= _MinDepth01)
                    return baseCol;

                float3 worldPos = ReconstructWorldPos(uv, depth01, eye);

                float fill = ComputeFillReveal(worldPos);
                float grid = ComputeGrid(worldPos);

                float t = max(0.0001, _LineSoftness);
                float lineMask = smoothstep(_LineThreshold - t, _LineThreshold + t, grid);

                float alpha = lerp(_BaseAlpha, _RevealAlpha, fill);
                alpha *= lineMask;
                alpha *= ComputeDistanceFade(worldPos);
                alpha *= ComputeDepthDisconConfidence(uv, eye, depth01);

                float3 outRgb = lerp(baseCol.rgb, _RevealColor.rgb, saturate(alpha));
                return half4(outRgb, baseCol.a);
            }
            ENDHLSL
        }
    }

    FallBack Off
}
