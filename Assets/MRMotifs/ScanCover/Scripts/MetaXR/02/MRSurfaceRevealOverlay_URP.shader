Shader "Custom/MRSurfaceRevealOverlay_URP"
{
    Properties
    {
        [Header(Colors)]
        _BaseColor ("Base Color", Color) = (0.10, 0.22, 0.25, 1)
        _RevealColor ("Reveal Color", Color) = (0.05, 1.00, 0.95, 1)

        [Header(Alpha)]
        _BaseAlpha ("Base Alpha", Range(0,1)) = 0.02
        _RevealAlpha ("Reveal Alpha", Range(0,1)) = 0.75

        [Header(Grid)]
        _GridScale ("Grid Scale", Float) = 6.0
        _GridThickness ("Grid Thickness", Range(0.001,0.15)) = 0.03
        _GridIntensity ("Grid Intensity", Range(0,5)) = 1.2

        [Header(Fresnel)]
        _FresnelPower ("Fresnel Power", Range(0.2,8)) = 2.0
        _FresnelIntensity ("Fresnel Intensity", Range(0,5)) = 1.0

        [Header(Reveal Emission)]
        _FillEmission ("Fill Emission", Range(0,10)) = 1.1

        [Header(Render)]
        [Enum(UnityEngine.Rendering.CullMode)] _Cull ("Cull", Float) = 2
    }

    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" "RenderPipeline"="UniversalPipeline" }
        LOD 100

        Pass
        {
            Name "ForwardUnlit"
            Tags { "LightMode"="UniversalForward" }

            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            Cull [_Cull]
            Offset -1, -1

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 3.0
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            #define REVEAL_MAX_WAVES 32

            CBUFFER_START(UnityPerMaterial)
                float4 _BaseColor;
                float4 _RevealColor;
                float _BaseAlpha;
                float _RevealAlpha;
                float _GridScale;
                float _GridThickness;
                float _GridIntensity;
                float _FresnelPower;
                float _FresnelIntensity;
                float _FillEmission;
            CBUFFER_END

            // Globals pushed by RevealManager
            float4 _RevealWaves[REVEAL_MAX_WAVES]; // xyz=center, w=radius
            float _RevealWaveCount;                // float
            float4 _RevealWaveParams;              // x=edgeFeather

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS   : NORMAL;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 worldPos   : TEXCOORD0;
                float3 worldNrm   : TEXCOORD1;
                float3 viewDirWS  : TEXCOORD2;

                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(Attributes v)
            {
                Varyings o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_TRANSFER_INSTANCE_ID(v, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                VertexPositionInputs posInputs = GetVertexPositionInputs(v.positionOS.xyz);
                VertexNormalInputs nrmInputs = GetVertexNormalInputs(v.normalOS);

                o.positionCS = posInputs.positionCS;
                o.worldPos = posInputs.positionWS;
                o.worldNrm = NormalizeNormalPerVertex(nrmInputs.normalWS);
                o.viewDirWS = GetWorldSpaceNormalizeViewDir(posInputs.positionWS);
                return o;
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

                    // inside is revealed, soft edge after radius
                    float local = 1.0 - smoothstep(r, r + feather, d);
                    w = max(w, local);
                }

                return saturate(w);
            }

            float ComputeGridMask(float3 worldPos, float3 worldNrm)
            {
                float3 p = worldPos * _GridScale;
                float3 fw = max(fwidth(p), 1e-4);

                float3 fracP = frac(p);
                float3 edgeDist = min(fracP, 1.0 - fracP);
                float thickness = _GridThickness;

                float3 lineAxis = 1.0 - smoothstep(thickness, thickness + fw * 1.5, edgeDist);

                float3 an = abs(normalize(worldNrm));
                an /= max(an.x + an.y + an.z, 1e-4);

                float lineXPlane = max(lineAxis.y, lineAxis.z);
                float lineYPlane = max(lineAxis.x, lineAxis.z);
                float lineZPlane = max(lineAxis.x, lineAxis.y);

                float grid = lineXPlane * an.x + lineYPlane * an.y + lineZPlane * an.z;
                return saturate(grid);
            }

            half4 frag(Varyings i) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(i);
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

                float3 N = normalize(i.worldNrm);
                float3 V = normalize(i.viewDirWS);

                float fill = ComputeFillReveal(i.worldPos);
                float grid = ComputeGridMask(i.worldPos, N);

                float fresnel = pow(1.0 - saturate(dot(N, V)), max(0.001, _FresnelPower)) * _FresnelIntensity;

                // Outside: barely visible. Inside: show more clearly.
                float alpha = lerp(_BaseAlpha, _RevealAlpha, fill);

                // Lines and subtle shading
                float lineLike = saturate(grid * _GridIntensity + fresnel);
                float show = lerp(0.08, 1.0, fill);

                float3 color = lerp(_BaseColor.rgb, _RevealColor.rgb, fill);
                color *= (0.25 + 0.75 * (lineLike * show));

                // Emission inside revealed area
                color += _RevealColor.rgb * (fill * (0.25 + 0.75 * lineLike) * _FillEmission);

                // Let revealed area be visible even if grid is weak
                alpha *= saturate(0.15 + 0.85 * max(lineLike, fill));

                return half4(color, alpha);
            }
            ENDHLSL
        }
    }

    FallBack Off
}
