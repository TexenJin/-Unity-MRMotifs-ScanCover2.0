Shader "Genesis/RoomSetupReplicaV2AdmissionPoints"
{
    Properties { _PointSize ("Point size", Float) = 3.5 }
    SubShader
    {
        Tags { "Queue"="Transparent+41" "RenderType"="Transparent" "IgnoreProjector"="True" }
        Pass
        {
            Cull Off
            ZWrite Off
            ZTest LEqual
            Blend SrcAlpha OneMinusSrcAlpha

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5
            #include "UnityCG.cginc"

            float _PointSize;

            struct appdata
            {
                float4 vertex : POSITION;
                float4 color : COLOR;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float4 color : COLOR;
                float psize : PSIZE;
            };

            v2f vert(appdata input)
            {
                v2f output;
                // Vertices are already callback-frame world positions.
                output.pos = UnityWorldToClipPos(input.vertex.xyz);
                output.color = input.color;
                output.psize = _PointSize;
                return output;
            }

            fixed4 frag(v2f input) : SV_Target
            {
                return input.color;
            }
            ENDCG
        }
    }
}
