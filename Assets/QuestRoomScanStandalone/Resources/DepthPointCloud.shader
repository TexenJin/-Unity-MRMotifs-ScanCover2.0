// QRS 独立链 - 实时深度点云（世界空间叠加层）
// 顶点着色器直接采样全局 gsDepthTex（眼别由 _EyeIndex 选，默认右眼），按 gsDepthProjInv/gsDepthViewInv
// 反投影到世界空间——零 CPU 射线、零回读，逐帧实时。
// 网格的"position"语义是深度图像素坐标 (px, py)，由 DepthPointCloudOverlay 建一次。
// 颜色 = turbo 色带（近=蓝紫 0.3m → 远=暗红 4m），无效深度=丢弃该点。
Shader "QRS/DepthPointCloud"
{
    Properties
    {
        _PointSize ("点尺寸(px)", Float) = 4.0
        _MinDist ("最小着色距离(米)", Float) = 0.3
        _MaxDist ("最大着色距离(米)", Float) = 4.0
        _EyeIndex ("深度来源眼(0=左 1=右)", Float) = 1.0
        _BandWidth ("等深色带宽度(米)", Float) = 0.25
    }
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" "IgnoreProjector"="True" }
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

            UNITY_DECLARE_TEX2DARRAY(gsDepthTex);
            uniform float4x4 gsDepthProj[2];
            uniform float4x4 gsDepthProjInv[2];
            uniform float4x4 gsDepthViewInv[2];
            uniform float2 gsDepthTexSize;

            float _PointSize;
            float _MinDist;
            float _MaxDist;
            float _EyeIndex;
            float _BandWidth;

            struct appdata
            {
                float4 vertex : POSITION; // xy = 深度图像素坐标
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 col : TEXCOORD0;
                float psize : PSIZE;
            };

            // Google turbo 色带多项式拟合（与 DepthPreview 一致）
            float3 TurboColormap(float x)
            {
                float x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x;
                return saturate(float3(
                    0.13572138 + 4.61539260*x - 42.66032258*x2 + 132.13108234*x3 - 152.94239396*x4 + 59.28637943*x5,
                    0.09140261 + 2.19418839*x + 4.84296658*x2 - 14.18503333*x3 + 4.27729857*x4 + 2.82956604*x5,
                    0.10667330 + 12.64194608*x - 60.58204836*x2 + 110.36276771*x3 - 89.90310912*x4 + 27.34824973*x5));
            }

            // 与 DepthKit.gsDepthNDCToLinear 同公式
            float Linearize(float ndc, int eye)
            {
                float z = ndc * 2.0 - 1.0;
                float A = gsDepthProj[eye][2][2];
                float B = gsDepthProj[eye][2][3];
                return abs(B / (z + A));
            }

            v2f vert(appdata v)
            {
                v2f o;
                int eye = (int)round(_EyeIndex);
                float2 uv = (v.vertex.xy + 0.5) / gsDepthTexSize;
                float ndc = UNITY_SAMPLE_TEX2DARRAY_LOD(gsDepthTex, float3(uv, eye), 0);
                float lin = Linearize(ndc, eye);

                // 所选眼该像素无效 → 回退另一只眼（整只眼 slice 无数据时整片回退，
                // 点云仍可见——宁可视差错位也不整片消失）
                if (!(lin > 0.05) || lin > 60.0)
                {
                    eye = 1 - eye;
                    ndc = UNITY_SAMPLE_TEX2DARRAY_LOD(gsDepthTex, float3(uv, eye), 0);
                    lin = Linearize(ndc, eye);
                }

                if (!(lin > 0.05) || lin > 60.0)
                {
                    // 双眼都无效：扔到相机后面让裁剪面丢掉（比 NaN 稳，Adreno 兼容性好）
                    o.pos = float4(0, 0, -10, 1);
                    o.col = float3(0, 0, 0);
                    o.psize = 0;
                    return o;
                }

                // NDC → 世界（DepthKit.gsDepthNDCtoWorld 同构，眼别随 fallback 后的 eye）
                float4 hcs = float4(float3(uv, ndc) * 2.0 - 1.0, 1);
                float4 worldH = mul(gsDepthViewInv[eye], mul(gsDepthProjInv[eye], hcs));
                float3 world = worldH.xyz / worldH.w;

                float t = saturate((lin - _MinDist) / (_MaxDist - _MinDist));
                float3 col = TurboColormap(t);

                // 地形图式标注（单眼可读起伏，对比色版）：
                // 等深轮廓线 = 品红亮线（turbo 色谱里没有品红 → 任意深度都是色相级对比，
                // 不用压暗暗纹——透视画面上暗色会沉进背景）；
                // 相邻色带改"提亮"交替（不再压暗），banding 结构保留但整体保持高亮。
                // 幽灵鼓包/凹陷会跨越带边界 → 出现闭合品红轮廓环+颜色跳变，肉眼直接可读
                float band = lin / max(_BandWidth, 0.01);
                float bandIdx = floor(band);
                float stripe = frac(band);
                float lineMask = 1.0 - smoothstep(0.0, 0.12, stripe);
                float alt = fmod(bandIdx, 2.0);
                col = lerp(col, lerp(col, float3(1.0, 1.0, 1.0), 0.30), alt);
                col = lerp(col, float3(1.0, 0.10, 0.55), lineMask * 0.92);

                o.col = col;
                o.pos = UnityWorldToClipPos(world);
                o.psize = _PointSize;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                return fixed4(i.col, 0.85);
            }
            ENDCG
        }
    }
}
