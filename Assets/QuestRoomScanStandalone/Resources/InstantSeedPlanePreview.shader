Shader "QRS/InstantSeedPlanePreview"
{
    Properties { _LineColor ("Line color", Color) = (0.1, 1, 0.35, 0.9) }
    SubShader
    {
        Tags { "Queue"="Transparent+46" "RenderType"="Transparent" "IgnoreProjector"="True" }
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
            fixed4 _LineColor;
            struct appdata { float4 vertex : POSITION; };
            struct v2f { float4 pos : SV_POSITION; };
            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                return o;
            }
            fixed4 frag(v2f i) : SV_Target { return _LineColor; }
            ENDCG
        }
    }
}
