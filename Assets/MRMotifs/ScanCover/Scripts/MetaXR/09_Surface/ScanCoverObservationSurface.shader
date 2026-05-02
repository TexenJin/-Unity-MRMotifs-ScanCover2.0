Shader "MRMotifs/ScanCover/ObservationSurface"
{
    Properties
    {
        _BaseColor ("Base Color", Color) = (0.18, 0.95, 0.98, 0.28)
        _FresnelColor ("Fresnel Color", Color) = (0.95, 1.0, 1.0, 0.9)
        _GridColor ("Grid Color", Color) = (0.95, 1.0, 1.0, 0.9)
        _BaseAlpha ("Base Alpha", Range(0,1)) = 0.28
        _FresnelPower ("Fresnel Power", Float) = 2.5
        _FresnelStrength ("Fresnel Strength", Range(0,3)) = 1.0
        _GridScale ("Grid Scale", Float) = 4.5
        _GridThickness ("Grid Thickness", Range(0.001,0.2)) = 0.035
        _GridIntensity ("Grid Intensity", Range(0,3)) = 1.1
        _Cull ("Cull", Float) = 0
    }

    SubShader
    {
        Tags
        {
            "Queue"="Transparent"
            "RenderType"="Transparent"
            "RenderPipeline"="UniversalPipeline"
        }

        Pass
        {
            Name "Forward"
            Cull [_Cull]
            ZWrite On
            ZTest LEqual
            Blend SrcAlpha OneMinusSrcAlpha

            HLSLPROGRAM
            #pragma target 3.0
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            CBUFFER_START(UnityPerMaterial)
                float4 _BaseColor;
                float4 _FresnelColor;
                float4 _GridColor;
                float _BaseAlpha;
                float _FresnelPower;
                float _FresnelStrength;
                float _GridScale;
                float _GridThickness;
                float _GridIntensity;
            CBUFFER_END

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float3 normalWS : TEXCOORD1;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(Attributes input)
            {
                Varyings output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                VertexPositionInputs positionInputs = GetVertexPositionInputs(input.positionOS.xyz);
                VertexNormalInputs normalInputs = GetVertexNormalInputs(input.normalOS);
                output.positionHCS = positionInputs.positionCS;
                output.positionWS = positionInputs.positionWS;
                output.normalWS = normalize(normalInputs.normalWS);
                return output;
            }

            float ComputeGridAxis(float value, float thickness)
            {
                float coord = value * _GridScale;
                float cell = abs(frac(coord) - 0.5);
                float fw = max(fwidth(coord), 1e-4);
                return 1.0 - smoothstep(thickness, thickness + fw * 1.5, cell);
            }

            half4 frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

                float3 normalWS = normalize(input.normalWS);
                float3 viewDirWS = SafeNormalize(_WorldSpaceCameraPos.xyz - input.positionWS);
                float viewDot = dot(normalWS, viewDirWS);
                if (viewDot <= 0.0)
                    discard;

                float axisX = ComputeGridAxis(input.positionWS.x, _GridThickness);
                float axisY = ComputeGridAxis(input.positionWS.y, _GridThickness);
                float axisZ = ComputeGridAxis(input.positionWS.z, _GridThickness);
                float3 normalAbs = abs(normalWS);
                float gridMask = axisX * normalAbs.y + axisX * normalAbs.z;
                gridMask = max(gridMask, axisY * normalAbs.x + axisY * normalAbs.z);
                gridMask = max(gridMask, axisZ * normalAbs.x + axisZ * normalAbs.y);
                gridMask = saturate(gridMask * _GridIntensity);

                float fresnel = pow(saturate(1.0 - saturate(viewDot)), max(0.01, _FresnelPower)) * _FresnelStrength;
                float3 litBase = _BaseColor.rgb * lerp(0.65, 1.15, saturate(normalWS.y * 0.5 + 0.5));
                float3 color = litBase;
                color += _FresnelColor.rgb * fresnel;
                color = lerp(color, _GridColor.rgb, gridMask * saturate(0.45 + _GridColor.a * 0.55));

                float alpha = saturate(_BaseAlpha + fresnel * 0.35 + gridMask * lerp(0.08, 0.25, saturate(_GridColor.a)));
                return half4(color, alpha);
            }
            ENDHLSL
        }
    }
}
