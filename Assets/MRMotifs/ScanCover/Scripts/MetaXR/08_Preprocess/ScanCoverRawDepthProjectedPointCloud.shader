Shader "Hidden/ScanCover/RawDepthProjectedPointCloud"
{
    Properties
    {
        _PointSizePixels ("Point Size Pixels", Float) = 3
        _Alpha ("Alpha", Range(0, 1)) = 0.95
        _Brightness ("Brightness", Float) = 1
    }

    SubShader
    {
        Tags { "Queue" = "Transparent" "RenderType" = "Transparent" }
        Pass
        {
            ZWrite Off
            Cull Off
            Blend SrcAlpha OneMinusSrcAlpha

            HLSLPROGRAM
            #pragma target 3.0
            #pragma vertex Vert
            #pragma fragment Frag

            #include "UnityCG.cginc"

            float _PointSizePixels;
            float _Alpha;
            float _Brightness;

            struct Attributes
            {
                float4 positionOS : POSITION;
                float4 color : COLOR;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float4 color : COLOR;
                float pointSize : PSIZE;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings Vert(Attributes input)
            {
                Varyings output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_OUTPUT(Varyings, output);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                output.positionCS = UnityObjectToClipPos(input.positionOS);
                output.color = input.color;
                output.color.rgb *= max(0.0, _Brightness);
                output.color.a *= _Alpha;
                output.pointSize = max(1.0, _PointSizePixels);
                return output;
            }

            float4 Frag(Varyings input) : SV_Target
            {
                return input.color;
            }
            ENDHLSL
        }
    }
}
