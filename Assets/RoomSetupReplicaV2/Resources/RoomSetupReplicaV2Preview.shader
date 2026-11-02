Shader "Genesis/RoomSetupReplicaV2Preview"
{
    Properties
    {
        _BaseColor ("Color", Color) = (0.02, 0.86, 1.0, 0.88)
        _LineWidthPixels ("Line Width Pixels", Range(0.75, 3.0)) = 1.5
        _DepthBias ("Depth Bias", Range(0.0, 0.002)) = 0.00025
    }
    SubShader
    {
        Tags
        {
            "RenderType"="Transparent"
            "Queue"="Transparent+45"
            "RenderPipeline"="UniversalPipeline"
        }
        Pass
        {
            Name "RoomSetupReplicaV2LinesXR"
            Tags { "LightMode"="SRPDefaultUnlit" }
            // Every topology edge is globally unique before reaching this
            // pass. Replace instead of accumulating color so intersections
            // and diagonal MSAA coverage do not look like thicker seams.
            Blend One Zero
            ZWrite Off
            ZTest LEqual
            Cull Off

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma multi_compile_instancing
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 extrusion : TEXCOORD0;
                float3 otherEndpointOS : TEXCOORD1;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                UNITY_VERTEX_OUTPUT_STEREO
            };
            CBUFFER_START(UnityPerMaterial)
                half4 _BaseColor;
                float _LineWidthPixels;
                float _DepthBias;
            CBUFFER_END
            Varyings Vert(Attributes input)
            {
                Varyings output = (Varyings)0;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                float4 positionHCS =
                    TransformObjectToHClip(input.positionOS.xyz);
                float4 otherHCS =
                    TransformObjectToHClip(input.otherEndpointOS);
                float safePositionW = abs(positionHCS.w) > 0.000001
                    ? positionHCS.w
                    : (positionHCS.w < 0.0 ? -0.000001 : 0.000001);
                float safeOtherW = abs(otherHCS.w) > 0.000001
                    ? otherHCS.w
                    : (otherHCS.w < 0.0 ? -0.000001 : 0.000001);
                float2 positionNdc = positionHCS.xy / safePositionW;
                float2 otherNdc = otherHCS.xy / safeOtherW;
                float2 deltaPixels = (otherNdc - positionNdc) *
                    (0.5 * _ScaledScreenParams.xy);
                float lengthSquared = dot(deltaPixels, deltaPixels);
                float2 tangent = lengthSquared > 0.000001
                    ? deltaPixels * rsqrt(lengthSquared)
                    : float2(1.0, 0.0);
                float2 normalPixels = float2(-tangent.y, tangent.x);
                // extrusion.x is -1/+1. _LineWidthPixels is the complete
                // requested width, so this supplies half its width in NDC on
                // each side of the center line.
                float2 offsetNdc = normalPixels * input.extrusion.x *
                    (_LineWidthPixels / _ScaledScreenParams.xy);
                positionHCS.xy += offsetNdc * positionHCS.w;
#if UNITY_REVERSED_Z
                positionHCS.z += _DepthBias * positionHCS.w;
#else
                positionHCS.z -= _DepthBias * positionHCS.w;
#endif
                output.positionHCS = positionHCS;
                return output;
            }
            half4 Frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                return _BaseColor;
            }
            ENDHLSL
        }
    }
}
