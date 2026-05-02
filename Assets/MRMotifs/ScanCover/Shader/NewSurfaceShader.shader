Shader "Custom/URP/ScanCoverUnlitMinimal"
{
    Properties
    {
        _BaseMap ("Base Map", 2D) = "white" {}
        _BaseColor ("Base Color", Color) = (1,1,1,1)

        _ScanColor ("Scan Ring Color", Color) = (0.15, 0.95, 1.0, 1.0)
        _CoveredColor ("Covered Tint Color", Color) = (0.08, 0.22, 0.35, 1.0)

        _LocalScanIntensityMul ("Local Scan Intensity Mul", Range(0,4)) = 1.0
        _LocalCoverageMul ("Local Coverage Mul", Range(0,2)) = 1.0
    }

    SubShader
    {
        Tags
        {
            "RenderPipeline" = "UniversalPipeline"
            "Queue" = "Geometry"
            "RenderType" = "Opaque"
        }

        Pass
        {
            Name "UniversalForward"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv         : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 uv          : TEXCOORD0;
                float3 positionWS  : TEXCOORD1;
            };

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);

            CBUFFER_START(UnityPerMaterial)
                float4 _BaseMap_ST;
                half4  _BaseColor;
                half4  _ScanColor;
                half4  _CoveredColor;
                half   _LocalScanIntensityMul;
                half   _LocalCoverageMul;
            CBUFFER_END

            // ===== 全局参数（由 ScanCoverEffectDriver.cs 写入）=====
            float  _ScanActive;
            float4 _ScanCenterWS;
            float4 _ScanNormalWS;
            float4 _ScanWorkCenterWS;
            float4 _ScanCenterRF;
            float  _ScanRadius;
            float  _ScanBandWidth;
            float  _ScanFeather;
            float  _ScanIntensity;
            float  _ScanPlaneThickness;
            float  _ScanProgress01;
            float  _ScanCoverageIntensity;
            float  _ScanTime;

            Varyings vert (Attributes IN)
            {
                Varyings OUT;
                OUT.positionHCS = TransformObjectToHClip(IN.positionOS.xyz);
                OUT.positionWS  = TransformObjectToWorld(IN.positionOS.xyz);
                OUT.uv          = TRANSFORM_TEX(IN.uv, _BaseMap);
                return OUT;
            }

            half4 frag (Varyings IN) : SV_Target
            {
                half4 baseTex = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, IN.uv);
                half4 col = baseTex * _BaseColor;

                // ---- 扫描参数 ----
                float3 P = IN.positionWS;
                float3 C = _ScanCenterWS.xyz;
                float3 N = _ScanNormalWS.xyz;

                // 防止异常零向量
                float nLen2 = dot(N, N);
                N = (nLen2 > 1e-6) ? normalize(N) : float3(0,1,0);

                float3 toP = P - C;

                // 到扫描平面的距离（沿法线）
                float planeDist = dot(toP, N);

                // 平面内向量与半径
                float3 inPlane = toP - planeDist * N;
                float radial = length(inPlane);

                float bandWidth = max(0.001, _ScanBandWidth);
                float halfBand  = max(0.0005, bandWidth * 0.5);
                float feather   = max(0.0001, _ScanFeather);

                // 环形 mask（radial 接近 _ScanRadius 时亮）
                float ringEdge = abs(radial - _ScanRadius);
                float ringMask = 1.0 - smoothstep(halfBand, halfBand + feather, ringEdge);

                // 平面厚度 mask（减少“上下层一起亮”）
                float planeMask = 1.0;
                if (_ScanPlaneThickness > 0.0001)
                {
                    planeMask = 1.0 - smoothstep(_ScanPlaneThickness, _ScanPlaneThickness + feather, abs(planeDist));
                }

                float active = saturate(_ScanActive);
                float scanMask = ringMask * planeMask * active;

                // ---- 已覆盖留痕（简单版）----
                // 当前半径以内算“已扫过”，配合 progress/intensity 做轻微染色
                float coveredMask = 1.0 - smoothstep(_ScanRadius + halfBand, _ScanRadius + halfBand + feather, radial);

                // 为了不影响远离扫描平面的表面，也乘上 planeMask
                coveredMask *= planeMask;

                // 只有进度>0 才明显可见（手动模式结束后 driver 会把 progress 维持在1）
                float coverageStrength = saturate(_ScanProgress01) * saturate(_ScanCoverageIntensity) * _LocalCoverageMul;
                coveredMask *= coverageStrength;

                // 染色（留痕）
                col.rgb = lerp(col.rgb, _CoveredColor.rgb, saturate(coveredMask));

                // 扫描环高亮（叠加）
                float scanStrength = scanMask * max(0.0, _ScanIntensity) * _LocalScanIntensityMul;
                col.rgb += _ScanColor.rgb * scanStrength;

                return col;
            }
            ENDHLSL
        }
    }

    FallBack Off
}