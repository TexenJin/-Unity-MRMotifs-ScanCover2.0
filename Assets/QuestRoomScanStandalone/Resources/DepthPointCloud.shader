// QRS 独立链 - 融合前 BB 反投影留痕。
// _DepthNormalSnapshot: R=采集当刻的 NDC 深度，GBA=世界法线。
// 每层使用各自采集当刻的投影/视图矩阵，因此历史点在世界中锁定，不跟随当前头姿漂移。
Shader "QRS/DepthPointCloud"
{
    Properties
    {
        _DepthNormalSnapshot ("Depth + world normal", 2D) = "black" {}
        _PointSize ("Point size (px)", Float) = 3.0
        _Age01 ("Trail age", Range(0,1)) = 0
    }
    SubShader
    {
        Tags { "Queue"="Transparent+40" "RenderType"="Transparent" "IgnoreProjector"="True" }
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

            sampler2D _DepthNormalSnapshot;
            float4x4 _SnapshotProj;
            float4x4 _SnapshotProjInv;
            float4x4 _SnapshotViewInv;
            float2 _SnapshotTexSize;
            float _PointSize;
            float _Age01;

            struct appdata { float4 vertex : POSITION; };
            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 col : TEXCOORD0;
                float alpha : TEXCOORD1;
                float psize : PSIZE;
            };

            float Linearize(float ndc)
            {
                float z = ndc * 2.0 - 1.0;
                return abs(_SnapshotProj[2][3] / (z + _SnapshotProj[2][2]));
            }

            v2f vert(appdata v)
            {
                v2f o;
                float2 uv = (v.vertex.xy + 0.5) / _SnapshotTexSize;
                float4 packed = tex2Dlod(_DepthNormalSnapshot, float4(uv, 0, 0));
                float ndc = packed.r;
                float linearDepth = Linearize(ndc);
                float3 normal = packed.gba;
                float normalLength = length(normal);

                if (!(linearDepth > 0.12) || linearDepth > 8.0 || normalLength < 0.25)
                {
                    o.pos = float4(0, 0, -10, 1);
                    o.col = 0;
                    o.alpha = 0;
                    o.psize = 0;
                    return o;
                }

                float4 hcs = float4(float3(uv, ndc) * 2.0 - 1.0, 1.0);
                float4 worldH = mul(_SnapshotViewInv, mul(_SnapshotProjInv, hcs));
                float3 world = worldH.xyz / worldH.w;
                float3 cameraWorld = mul(_SnapshotViewInv, float4(0, 0, 0, 1)).xyz;
                float3 toCamera = normalize(cameraWorld - world);
                float facing = saturate(abs(dot(normalize(normal), toCamera)));

                // OpenQuestCapture 式语义：正视近白；中等斜视转黄；掠射转洋红。
                float grazing = 1.0 - smoothstep(0.18, 0.82, facing);
                float3 frontColor = float3(0.92, 0.98, 1.00);
                float3 midColor = float3(1.00, 0.72, 0.08);
                float3 grazingColor = float3(1.00, 0.05, 0.55);
                float3 color = grazing < 0.5
                    ? lerp(frontColor, midColor, grazing * 2.0)
                    : lerp(midColor, grazingColor, (grazing - 0.5) * 2.0);

                o.pos = UnityWorldToClipPos(world);
                o.col = color;
                o.alpha = lerp(0.82, 0.10, saturate(_Age01));
                o.psize = _PointSize;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                return fixed4(i.col, i.alpha);
            }
            ENDCG
        }
    }
}
