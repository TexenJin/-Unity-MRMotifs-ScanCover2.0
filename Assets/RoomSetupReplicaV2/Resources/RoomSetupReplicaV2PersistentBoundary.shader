Shader "Genesis/RoomSetupReplicaV2PersistentBoundary"
{
    SubShader
    {
        Tags { "Queue"="Transparent+42" "RenderType"="Transparent" "IgnoreProjector"="True" }
        Pass
        {
            Cull Off
            ZWrite Off
            ZTest LEqual
            Blend SrcAlpha OneMinusSrcAlpha

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 3.0
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float4 color : COLOR;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float4 color : COLOR;
            };

            v2f vert(appdata input)
            {
                v2f output;
                // Vertices are stored directly in callback-frame world space.
                output.pos = UnityWorldToClipPos(input.vertex.xyz);
                output.color = input.color;
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
