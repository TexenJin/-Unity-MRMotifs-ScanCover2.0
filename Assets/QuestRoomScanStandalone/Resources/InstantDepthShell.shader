// 融合前即时外壳：最近的清洗深度帧各自形成短寿命局部三角膜。
// 所有连通判定只影响本影子显示，不写 TSDF/枪胶/纸皮。
Shader "QRS/InstantDepthShell"
{
    Properties
    {
        _DepthNormalSnapshot ("Depth + world normal", 2D) = "black" {}
        _ShellReviewTex ("Inter-shell review state", 2D) = "black" {}
        _ConnectivityState ("Triangle connectivity state", 2D) = "black" {}
        _PlaneConnectivityState ("Plane-aware shadow state", 2D) = "black" {}
        _Age01 ("Live age", Range(0,1)) = 0
        _FrozenSnapshot ("Frozen diagnostic snapshot", Float) = 0
        _ShowRejectedTriangles ("Show rejected diagnostic edges", Float) = 1
        _CompositeWithProduction ("Depth-aware paper ownership", Float) = 0
        _CompositeDepthBiasMeters ("Paper handoff depth tolerance", Float) = 0.008
    }
    SubShader
    {
        // 稳定纸拓扑(+35)和 TSDF 纸网(+40)先写深度；即时壳(+45)后画。
        // 合流时把壳沿视线后推一个很小的容差：近乎同深时纸接管，壳明显
        // 更靠近相机时仍能通过 ZTest。不能再用“同屏像素有纸”代替深度所有权，
        // 否则墙后纸会把空调等前景壳一并抹掉。
        Tags { "Queue"="Transparent+45" "RenderType"="Transparent" "IgnoreProjector"="True" }
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
            Texture2D<uint> _ShellReviewTex;
            Texture2D<uint> _ConnectivityState;
            Texture2D<uint> _PlaneConnectivityState;
            float4x4 _SnapshotProj;
            float4x4 _SnapshotProjInv;
            float4x4 _SnapshotViewInv;
            float2 _SnapshotTexSize;
            float _Age01;
            float _FrozenSnapshot;
            float _ShowRejectedTriangles;
            float _CompositeWithProduction;
            float _CompositeDepthBiasMeters;

            struct appdata
            {
                float4 vertex : POSITION; // xy=源像素，z=格内三角号
                float2 cell : TEXCOORD0;  // 格左下源像素
                float2 shellData : TEXCOORD1; // x=固定源像素步长
                float4 bary : COLOR;      // 线框重心坐标
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 bary : TEXCOORD0;
                float risk : TEXCOORD1;
                float breakRisk : TEXCOORD2;
                float reviewRisk : TEXCOORD3;
                float reviewConflict : TEXCOORD4;
                float rejected : TEXCOORD5;
                float connectivityReason : TEXCOORD6;
                float rescued : TEXCOORD7;
            };

            float Linearize(float ndc)
            {
                float z = ndc * 2.0 - 1.0;
                return abs(_SnapshotProj[2][3] / (z + _SnapshotProj[2][2]));
            }

            float4 ReadPacked(float2 pixel)
            {
                float2 uv = (pixel + 0.5) / _SnapshotTexSize;
                return tex2Dlod(_DepthNormalSnapshot, float4(uv, 0, 0));
            }

            float ReviewRisk(uint state)
            {
                // 独立视角冲突用强红；同视冲突只用橙色，避免把时序抖动冒充定罪。
                if (state == 7u) return 1.0;
                if (state == 8u) return 0.92;
                if (state == 4u) return 0.68;
                if (state == 5u) return 0.58;
                // 同视重复只能说明“再次看见”，不能冒充独立稳定；保持黄色提醒。
                if (state == 2u) return 0.48;
                if (state == 1u || state == 6u) return 0.36;
                if (state == 3u) return 0.0;
                return 0.55;
            }

            float ReviewConflict(uint state)
            {
                if (state == 7u) return 1.0;
                if (state == 8u) return 0.82;
                if (state == 4u) return 0.42;
                if (state == 5u) return 0.32;
                return 0.0;
            }

            v2f vert(appdata v)
            {
                v2f o;
                float tri = v.vertex.z;
                float2 a = v.cell;
                float2 b;
                float2 c;
                float stepHint = max(v.shellData.x, 1.0);
                if (tri < 0.5)
                {
                    b = v.cell + float2(stepHint, 0);
                    c = v.cell + float2(0, stepHint);
                }
                else
                {
                    b = v.cell + float2(stepHint, stepHint);
                    a = v.cell + float2(stepHint, 0);
                    c = v.cell + float2(0, stepHint);
                }

                float4 pa = ReadPacked(a);
                float4 pb = ReadPacked(b);
                float4 pc = ReadPacked(c);
                uint reviewA = _ShellReviewTex.Load(int3(int2(a), 0));
                uint reviewB = _ShellReviewTex.Load(int3(int2(b), 0));
                uint reviewC = _ShellReviewTex.Load(int3(int2(c), 0));
                uint packedConnectivity = _ConnectivityState.Load(int3(int2(v.cell), 0));
                uint connectivity = tri < 0.5
                    ? (packedConnectivity & 255u)
                    : ((packedConnectivity >> 8) & 255u);
                uint packedPlaneConnectivity = _PlaneConnectivityState.Load(int3(int2(v.cell), 0));
                uint planeConnectivity = tri < 0.5
                    ? (packedPlaneConnectivity & 255u)
                    : ((packedPlaneConnectivity >> 8) & 255u);
                float shellReviewRisk = max(ReviewRisk(reviewA),
                                            max(ReviewRisk(reviewB), ReviewRisk(reviewC)));
                float shellReviewConflict = max(ReviewConflict(reviewA),
                                                max(ReviewConflict(reviewB), ReviewConflict(reviewC)));
                float da = Linearize(pa.r);
                float db = Linearize(pb.r);
                float dc = Linearize(pc.r);
                float3 na = pa.gba;
                float3 nb = pb.gba;
                float3 nc = pc.gba;
                float minDepth = min(da, min(db, dc));
                float maxDepth = max(da, max(db, dc));
                float hardGap = max(0.030, minDepth * 0.025);
                float3 sna = na / max(length(na), 0.0001);
                float3 snb = nb / max(length(nb), 0.0001);
                float3 snc = nc / max(length(nc), 0.0001);
                float normalCoherence = min(dot(sna, snb),
                                            min(dot(snb, snc), dot(snc, sna)));
                bool rescued = planeConnectivity == 1u;
                bool accepted = connectivity == 0u || rescued;
                bool hasWorldEndpoints = connectivity != 1u;
                if (!accepted && (_ShowRejectedTriangles < 0.5 || !hasWorldEndpoints))
                {
                    o.pos = float4(2.0, 2.0, 2.0, 1.0);
                    o.bary = v.bary.rgb;
                    o.risk = 1.0;
                    o.breakRisk = 1.0;
                    o.reviewRisk = shellReviewRisk;
                    o.reviewConflict = shellReviewConflict;
                    o.rejected = 1.0;
                    o.connectivityReason = connectivity;
                    o.rescued = 0.0;
                    return o;
                }

                float2 uv = (v.vertex.xy + 0.5) / _SnapshotTexSize;
                float4 packed = ReadPacked(v.vertex.xy);
                float ndc = packed.r;
                float4 hcs = float4(float3(uv, ndc) * 2.0 - 1.0, 1.0);
                float4 worldH = mul(_SnapshotViewInv, mul(_SnapshotProjInv, hcs));
                float3 world = worldH.xyz / worldH.w;
                float3 cameraWorld = mul(_SnapshotViewInv, float4(0, 0, 0, 1)).xyz;
                float3 toCamera = normalize(cameraWorld - world);
                float3 avgNormal = normalize(na + nb + nc);
                float facing = saturate(abs(dot(avgNormal, toCamera)));

                float3 currentViewRay = normalize(world - _WorldSpaceCameraPos.xyz);
                float3 renderWorld = world + currentViewRay *
                    (_CompositeWithProduction > 0.5 ? max(_CompositeDepthBiasMeters, 0.0) : 0.0);
                o.pos = UnityWorldToClipPos(renderWorld);
                o.bary = v.bary.rgb;
                o.risk = max(1.0 - smoothstep(0.18, 0.78, facing),
                             1.0 - smoothstep(0.35, 0.85, normalCoherence));
                o.breakRisk = saturate((maxDepth - minDepth) / max(hardGap, 0.001));
                o.reviewRisk = shellReviewRisk;
                o.reviewConflict = shellReviewConflict;
                o.rejected = accepted ? 0.0 : 1.0;
                o.connectivityReason = connectivity;
                // 绿色只是独显档的血缘标签。合流档中救回面与普通壳
                // 共用外观，是否可见由纸/壳真实深度及交接容差决定。
                o.rescued = rescued && _ShowRejectedTriangles > 0.5 ? 1.0 : 0.0;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float edgeDistance = min(i.bary.x, min(i.bary.y, i.bary.z));
                float edgeWidth = max(fwidth(edgeDistance) * 1.25, 0.003);
                float edge = 1.0 - smoothstep(edgeWidth, edgeWidth * 2.2, edgeDistance);
                // 独显档仍用年龄表达新旧；壳纸合流档的退场权则交给
                // 纸皮深度。若纸与壳同深，顶点阶段的 8mm 后移 + ZTest 会让
                // 纸接管；若壳是明显更近的空调/柜体，它不得只因时间淡出。
                float liveFade = _CompositeWithProduction > 0.5
                    ? 1.0
                    : (_FrozenSnapshot > 0.5 ? 1.0
                       : lerp(1.0, 0.08, saturate(_Age01)));

                if (i.rejected > 0.5)
                {
                    // 深拒=橙，法拒=紫，双拒=红；只画诊断线，不伪装成已覆盖表面。
                    float3 rejectedColor = i.connectivityReason < 2.5
                        ? float3(1.00, 0.52, 0.04)
                        : (i.connectivityReason < 3.5
                            ? float3(0.85, 0.20, 1.00)
                            : float3(1.00, 0.04, 0.08));
                    return fixed4(rejectedColor, lerp(0.008, 0.76, edge) * liveFade);
                }

                if (i.rescued > 0.5)
                {
                    // 绿色仅代表独显诊断档的“共面救回”血缘。
                    float3 rescuedColor = lerp(float3(0.08, 0.78, 0.22),
                                                float3(0.45, 1.00, 0.58), edge);
                    return fixed4(rescuedColor, lerp(0.10, 0.86, edge) * liveFade);
                }

                float3 safe = _FrozenSnapshot > 0.5
                    ? float3(0.74, 0.32, 1.00)
                    : float3(0.05, 0.90, 1.00);
                float3 yellow = float3(1.00, 0.72, 0.06);
                float3 red = float3(1.00, 0.08, 0.12);
                float risk = max(max(i.risk, i.reviewRisk),
                                 smoothstep(0.55, 0.92, i.breakRisk));
                float3 color = lerp(safe, yellow, saturate(risk * 1.35));
                float redRisk = max(i.reviewConflict,
                                    smoothstep(0.76, 0.96, i.breakRisk));
                color = lerp(color, red, redRisk);
                color = lerp(color, 1.0, edge * 0.32);

                // 合流档中能通过纸皮深度测试的壳，就是尚未被稳定纸
                // 同深接管的前景单元。必须给它实质填充，否则 0.13 alpha 会让
                // 背后不透明墙纸主导视觉，造成“空调已被抹平”的假象。
                float fillAlpha = (_CompositeWithProduction > 0.5
                    ? 0.78
                    : (_FrozenSnapshot > 0.5 ? 0.10 : 0.13)) * liveFade;
                float edgeAlpha = (_CompositeWithProduction > 0.5
                    ? 0.96
                    : (_FrozenSnapshot > 0.5 ? 0.70 : 0.88)) * liveFade;
                return fixed4(color, lerp(fillAlpha, edgeAlpha, edge));
            }
            ENDCG
        }
    }
}
