// QRS 独立链 - 深度实时预览（HUD 小窗用）
// 直接采样 DepthCapture 全局绑定的 gsDepthTex（slice 0，与融合 kernel 同一只眼），
// 用 gsDepthProj[0] 把 NDC 深度线性化成米。
// v2：turbo 全谱色带（0.3m=红 → 4m=深红，途经橙黄绿青蓝紫）+ 每 0.5m 一条品红对比色等深线，
//     房间内 1~3m 的窄深度区间也能拉开色差，斑块轮廓一眼可辨；无效/超范围=深暗底。
// 放在 Resources 下是为了保证打进包（运行时用 Resources.Load 取，Shader.Find 会被裁剪）。
Shader "QRS/DepthPreview"
{
    Properties
    {
        _MinDist ("最小显示距离(米)", Float) = 0.3
        _MaxDist ("最大显示距离(米)", Float) = 4.0
    }
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" "IgnoreProjector"="True" }
        Cull Off
        ZWrite Off
        ZTest Always
        Blend SrcAlpha OneMinusSrcAlpha

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5
            #include "UnityCG.cginc"

            // 与 DepthCapture.cs 的 Shader.SetGlobalTexture("gsDepthTex", ...) 同名即自动全局绑定
            UNITY_DECLARE_TEX2DARRAY(gsDepthTex);
            uniform float4x4 gsDepthProj[2];

            float _MinDist;
            float _MaxDist;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            // Google turbo 色带多项式拟合：全光谱、单调亮度、相邻深度色差大
            float3 TurboColormap(float x)
            {
                float x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x;
                return saturate(float3(
                    0.13572138 + 4.61539260*x - 42.66032258*x2 + 132.13108234*x3 - 152.94239396*x4 + 59.28637943*x5,
                    0.09140261 + 2.19418839*x + 4.84296658*x2 - 14.18503333*x3 + 4.27729857*x4 + 2.82956604*x5,
                    0.10667330 + 12.64194608*x - 60.58204836*x2 + 110.36276771*x3 - 89.90310912*x4 + 27.34824973*x5));
            }

            fixed4 frag(v2f i) : SV_Target
            {
                // 左右眼对比模式：左半窗=eye0（融合用眼），右半窗=eye1。
                // 裁决"幽灵是否为双眼共享幻觉"：幽灵斑块只在左半存在→右眼可作异议证据源；
                // 两半都有→Meta 深度同源重投影，右眼无独立证据，只能上自有双目。
                int eye = i.uv.x < 0.5 ? 0 : 1;
                float2 duv = float2(frac(i.uv.x * 2.0), i.uv.y);

                float ndc = UNITY_SAMPLE_TEX2DARRAY(gsDepthTex, float3(duv, eye));
                // 与 DepthKit.gsDepthNDCToLinear 同公式
                float z = ndc * 2.0 - 1.0;
                float A = gsDepthProj[eye][2][2];
                float B = gsDepthProj[eye][2][3];
                float lin = abs(B / (z + A));

                // 中缝白线，方便对照同一位置
                if (abs(i.uv.x - 0.5) < 0.004)
                    return fixed4(1, 1, 1, 0.95);

                // 无效（NaN/0/贴脸）或远得离谱（>60m，多半是空洞）→ 深暗底
                if (!(lin > 0.05) || lin > 60.0)
                    return fixed4(0.04, 0.04, 0.07, 0.85);

                float t = saturate((lin - _MinDist) / (_MaxDist - _MinDist));
                float3 col = TurboColormap(t);

                // 每 0.5m 一条等深轮廓线：品红对比色（turbo 色谱里没有品红 → 任意深度都跳，
                // 不用压暗暗纹），深度突变处出现清晰轮廓线，幽灵斑块形状可直接读
                float stripe = frac(lin * 2.0);
                float lineMask = 1.0 - smoothstep(0.0, 0.10, stripe);
                col = lerp(col, float3(1.0, 0.10, 0.55), lineMask * 0.90);

                return fixed4(col, 0.92);
            }
            ENDCG
        }
    }
}
