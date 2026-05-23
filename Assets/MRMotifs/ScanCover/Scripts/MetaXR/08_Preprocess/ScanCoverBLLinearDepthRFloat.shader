Shader "Hidden/ScanCover/BLLinearDepthRFloat"
{
    SubShader
    {
        Tags { "RenderType" = "Opaque" "Queue" = "Geometry" }
        Pass
        {
            ZWrite On
            ZTest LEqual
            Cull Off

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag

            #include "UnityCG.cginc"

            float _NearClipMeters;
            float _FarClipMeters;

            struct Attributes
            {
                float4 positionOS : POSITION;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float linearDepth : TEXCOORD0;
            };

            Varyings Vert(Attributes input)
            {
                Varyings output;
                float4 viewPos = mul(UNITY_MATRIX_MV, input.positionOS);
                output.positionCS = mul(UNITY_MATRIX_P, viewPos);
                output.linearDepth = -viewPos.z;
                return output;
            }

            float Frag(Varyings input) : SV_Target
            {
                float depth = input.linearDepth;
                return (depth >= _NearClipMeters && depth <= _FarClipMeters) ? depth : 0.0;
            }
            ENDHLSL
        }
    }
    FallBack Off
}
