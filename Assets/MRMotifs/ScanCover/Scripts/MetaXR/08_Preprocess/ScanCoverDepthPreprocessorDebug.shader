Shader "MRMotifs/ScanCover/DepthPreprocessorDebug"
{
    Properties
    {
        _ViewMode ("View Mode", Float) = 1
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
            ZWrite Off
            ZTest Always
            Cull Off
            Blend SrcAlpha OneMinusSrcAlpha

            HLSLPROGRAM
            #pragma target 3.0
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            TEXTURE2D(_ScanCoverDepthWorldPositionTexture);
            SAMPLER(sampler_ScanCoverDepthWorldPositionTexture);
            TEXTURE2D(_ScanCoverDepthWorldNormalTexture);
            SAMPLER(sampler_ScanCoverDepthWorldNormalTexture);
            TEXTURE2D(_ScanCoverDepthObservationMetaTexture);
            SAMPLER(sampler_ScanCoverDepthObservationMetaTexture);

            CBUFFER_START(UnityPerMaterial)
                float _ViewMode;
            CBUFFER_END

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(Attributes input)
            {
                Varyings output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                output.positionHCS = TransformObjectToHClip(input.positionOS.xyz);
                output.uv = input.uv;
                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

                float4 pos = SAMPLE_TEXTURE2D(_ScanCoverDepthWorldPositionTexture, sampler_ScanCoverDepthWorldPositionTexture, input.uv);
                float4 normal = SAMPLE_TEXTURE2D(_ScanCoverDepthWorldNormalTexture, sampler_ScanCoverDepthWorldNormalTexture, input.uv);
                float4 meta = SAMPLE_TEXTURE2D(_ScanCoverDepthObservationMetaTexture, sampler_ScanCoverDepthObservationMetaTexture, input.uv);

                if (_ViewMode < 0.5)
                    return half4(meta.xxx, 1.0);

                if (_ViewMode < 1.5)
                    return half4(meta.yyy, 1.0);

                if (_ViewMode < 2.5)
                    return half4(saturate(meta.zzz / 6.0), 1.0);

                if (_ViewMode < 3.5)
                    return half4(normal.xyz * 0.5 + 0.5, 1.0);

                if (_ViewMode < 4.5)
                    return half4(abs(pos.xyz) * 0.1, 1.0);

                return half4(normal.w, normal.w, normal.w, 1.0);
            }
            ENDHLSL
        }
    }
}
