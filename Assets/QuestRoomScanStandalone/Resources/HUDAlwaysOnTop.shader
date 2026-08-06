// Genesis RoomScan - 世界空间 HUD 置顶 shader
// 解决：WorldSpace Canvas 走 UI/Default，ZTest 由 unity_GUIZTestMode 固定为 LEqual，
// 面板会被扫描网格/墙面遮挡。此 shader = 极简 UI 管线 + ZTest Always + Overlay 队列，
// 文字（Alpha8 字体图集，只取 alpha）与背景（白 sprite）共用：col.rgb=顶点色，col.a*=tex.a。
// 放 Resources 下防打包裁剪，C# 侧 Resources.Load<Shader>("HUDAlwaysOnTop")。
Shader "QRS/HUDAlwaysOnTop"
{
    Properties
    {
        [PerRendererData] _MainTex ("Sprite/Font Texture", 2D) = "white" {}
        _Color ("Tint", Color) = (1,1,1,1)
    }
    SubShader
    {
        Tags { "Queue"="Overlay" "IgnoreProjector"="True" "RenderType"="Transparent" "PreviewType"="Plane" }
        Cull Off
        Lighting Off
        ZWrite Off
        ZTest Always
        Blend SrcAlpha OneMinusSrcAlpha

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata_t
            {
                float4 vertex   : POSITION;
                float4 color    : COLOR;
                float2 texcoord : TEXCOORD0;
            };

            struct v2f
            {
                float4 vertex   : SV_POSITION;
                fixed4 color    : COLOR;
                half2  texcoord : TEXCOORD0;
            };

            fixed4 _Color;
            sampler2D _MainTex;

            v2f vert (appdata_t v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.texcoord = v.texcoord;
                o.color = v.color * _Color;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // 动态字体图集是 Alpha8：rgb 为黑、只有 alpha 存字形 → 只取 alpha，
                // rgb 用顶点色（与 UI/Default 同语义）；背景白 sprite 时 tex.a=1 同样正确
                fixed4 col = i.color;
                col.a *= tex2D(_MainTex, i.texcoord).a;
                return col;
            }
            ENDCG
        }
    }
}
