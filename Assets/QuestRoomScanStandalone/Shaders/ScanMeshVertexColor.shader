Shader "Genesis/ScanMeshVertexColor"
{
    Properties { }
    SubShader
    {
        Tags { "RenderType"="Transparent" "RenderPipeline"="UniversalPipeline" "Queue"="Transparent+40" }

        Pass
        {
            Name "VertexColorUnlit"
            Tags { "LightMode"="SRPDefaultUnlit" }
            // 08-18 晚帧率手术：ZWrite 改开。原"不写深度"在满屏微三角形下=透明队列
            // 无 early-Z，每像素被 N 层三角形各跑一次完整片元（实机定罪：光栅化主猪
            // 独吃 20~30 帧）。线框模式 discard 的内部不写深度，格栅透视感保留；
            // 实体模式重叠层被深度剔除=视觉变"实"，诊断可读性不变。
            // 点云(Transparent)/覆盖片(+20)队列更浅先画且不写深度，互不影响。
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite On
            ZTest LEqual
            // The reconstructed room shell is viewed from its interior.  Do
            // not make triangle winding a visibility/admission rule: ceiling
            // triangles can legitimately be wound away from the headset.
            Cull Off
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct GPUVertex
            {
                float3 pos;
                float3 norm;
                uint   packedColor;
                uint   voxelFlatIdx;
            };
            StructuredBuffer<GPUVertex> _SurfaceVerts;
            StructuredBuffer<uint>      _SurfaceIndices;
            StructuredBuffer<uint>      _VertexAdmissionClass;

            half4 UnpackColor(uint packed)
            {
                return half4(
                    (packed        & 0xFF) / 255.0h,
                    ((packed >> 8) & 0xFF) / 255.0h,
                    ((packed >> 16)& 0xFF) / 255.0h,
                    ((packed >> 24)& 0xFF) / 255.0h);
            }

            // ── Triplanar persistent textures ──
            TEXTURE2D(_RSTriXZ);  SAMPLER(sampler_RSTriXZ);
            TEXTURE2D(_RSTriXY);  SAMPLER(sampler_RSTriXY);
            TEXTURE2D(_RSTriYZ);  SAMPLER(sampler_RSTriYZ);
            TEXTURE2D(_RSTriDepthXZ);  SAMPLER(sampler_RSTriDepthXZ);
            TEXTURE2D(_RSTriDepthXY);  SAMPLER(sampler_RSTriDepthXY);
            TEXTURE2D(_RSTriDepthYZ);  SAMPLER(sampler_RSTriDepthYZ);
            float _RSTriAvailable;

            // ── TSDF volume (for freeze tint) ──
            TEXTURE3D(gsVolume);
            SAMPLER(sampler_gsVolume);
            float4 gsVoxCount;
            float gsVoxSize;

            // ── 置信度通道 v1（只读诊断可视化，全局量由 VolumeIntegrator 下发）──
            TEXTURE3D(gsConfidence);
            SAMPLER(sampler_gsConfidence);
            float _RSConfidenceViz;
            float gsConfidenceMidMax;
            float gsConfidenceLowMin;

            // ── Globals set by RoomScanner ──
            float _RSNoFreezeTint;
            float _RSNormalFallback;
            float _RSWireframe;
            float _RSWireThickness;
            float _RSMeshStride;
            float _RSGridSpacing;
            float _RSPaperGridMode;
            float _RSProductGridMode;
            float _RSGeometryTruthView;
            float4 _RSExtractionColor;
            float _RSJointDiagnostic;
            float _RSSuppressPink;
            float _RSTemporalIllegalActive;
            float _RSHeraReplayActive;

            #define DEPTH_TOLERANCE 0.015

            float2 PaperGridUV(float3 positionWS, float3 normalWS)
            {
                float3 a = abs(normalize(normalWS));
                if (a.x >= a.y && a.x >= a.z) return positionWS.zy;
                if (a.y >= a.z) return positionWS.xz;
                return positionWS.xy;
            }

            float PaperTriangleGrid(float3 positionWS, float3 normalWS)
            {
                float spacing = max(_RSGridSpacing, 0.04);
                float2 p = PaperGridUV(positionWS, normalWS) / spacing;
                float3 family = float3(
                    p.x,
                    0.5 * p.x + 0.8660254 * p.y,
                   -0.5 * p.x + 0.8660254 * p.y);
                float3 distanceToLine = abs(frac(family + 0.5) - 0.5);
                float3 aa = max(fwidth(family) * max(_RSWireThickness, 0.55), 0.0008);
                float3 lineCoverage = 1.0 - smoothstep(aa, aa * 1.8, distanceToLine);
                return max(lineCoverage.x, max(lineCoverage.y, lineCoverage.z));
            }

            float ProductTriangleGrid(float3 positionWS, float3 normalWS)
            {
                float2 p = PaperGridUV(positionWS, normalWS) / 0.10;
                float3 family = float3(
                    p.x,
                    0.5 * p.x + 0.8660254 * p.y,
                   -0.5 * p.x + 0.8660254 * p.y);
                float3 distanceToLine = abs(frac(family + 0.5) - 0.5);
                float3 aa = max(fwidth(family) * max(_RSWireThickness, 0.55),
                                0.0008);
                float3 lineCoverage = 1.0 - smoothstep(aa, aa * 1.8,
                                                       distanceToLine);
                return max(lineCoverage.x, max(lineCoverage.y,
                                               lineCoverage.z));
            }

            float3 WorldToVoxelUVW(float3 worldPos)
            {
                float3 local = worldPos / gsVoxSize + gsVoxCount.xyz / 2.0;
                return saturate(local / gsVoxCount.xyz);
            }

            float2 SignedTriUV(float2 baseUV, float normalComponent)
            {
                return float2(baseUV.x, normalComponent > 0 ? baseUV.y * 0.5 + 0.5 : baseUV.y * 0.5);
            }

            half3 SampleTriplanar(float3 worldPos, float3 normal)
            {
                float3 absN   = abs(normal);
                float3 blend  = absN / (absN.x + absN.y + absN.z + 0.001);
                float3 uvw    = WorldToVoxelUVW(worldPos);

                float2 uvXZ = SignedTriUV(uvw.xz, normal.y);
                float2 uvXY = SignedTriUV(uvw.xy, normal.z);
                float2 uvYZ = SignedTriUV(uvw.yz, normal.x);

                half4 colXZ = SAMPLE_TEXTURE2D(_RSTriXZ, sampler_RSTriXZ, uvXZ);
                half4 colXY = SAMPLE_TEXTURE2D(_RSTriXY, sampler_RSTriXY, uvXY);
                half4 colYZ = SAMPLE_TEXTURE2D(_RSTriYZ, sampler_RSTriYZ, uvYZ);

                float dXZ = SAMPLE_TEXTURE2D(_RSTriDepthXZ, sampler_RSTriDepthXZ, uvXZ).r;
                float dXY = SAMPLE_TEXTURE2D(_RSTriDepthXY, sampler_RSTriDepthXY, uvXY).r;
                float dYZ = SAMPLE_TEXTURE2D(_RSTriDepthYZ, sampler_RSTriDepthYZ, uvYZ).r;

                if (dXZ > 0.001 && abs(uvw.y - dXZ) > DEPTH_TOLERANCE) colXZ = half4(0, 0, 0, 0);
                if (dXY > 0.001 && abs(uvw.z - dXY) > DEPTH_TOLERANCE) colXY = half4(0, 0, 0, 0);
                if (dYZ > 0.001 && abs(uvw.x - dYZ) > DEPTH_TOLERANCE) colYZ = half4(0, 0, 0, 0);

                half3 rgb = colXZ.rgb * blend.y + colXY.rgb * blend.z + colYZ.rgb * blend.x;
                half totalAlpha = colXZ.a * blend.y + colXY.a * blend.z + colYZ.a * blend.x;

                return totalAlpha > 0.01 ? rgb : half3(-1, -1, -1);
            }

            bool IsVoxelFrozen(float3 worldPos)
            {
                float3 uvw = WorldToVoxelUVW(worldPos);
                float2 tsdf = SAMPLE_TEXTURE3D_LOD(gsVolume, sampler_gsVolume, uvw, 0).rg;
                return tsdf.g < 0;
            }

            half3 ApplyFreezeTint(half3 color, float3 worldPos)
            {
                if (_RSNoFreezeTint < 0.5 && IsVoxelFrozen(worldPos))
                    color = lerp(color, half3(0.3, 0.5, 0.9), 0.25);
                return color;
            }

            // 置信度通道 v1 可视化（_RSConfidenceViz 诊断开关，默认关）：
            // 高置信=原色（分歧 EMA 小，逐帧观测一致）；中=黄（1.5~3.8cm 抖动）；
            // 低=蓝紫（几何在打架）；体素已无数据=灰。只读着色，不碰生产逻辑。
            half3 ApplyConfidenceViz(half3 color, float3 worldPos)
            {
                if (_RSConfidenceViz < 0.5) return color;
                float3 uvw = WorldToVoxelUVW(worldPos);
                float w = SAMPLE_TEXTURE3D_LOD(gsVolume, sampler_gsVolume, uvw, 0).g;
                if (abs(w) < 0.01) return half3(0.5, 0.5, 0.5);
                float d = SAMPLE_TEXTURE3D_LOD(gsConfidence, sampler_gsConfidence, uvw, 0).r;
                if (d < gsConfidenceMidMax) return color;
                if (d < gsConfidenceLowMin) return half3(1.0, 0.85, 0.1);
                return half3(0.45, 0.3, 1.0);
            }

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float4 color : COLOR;
                float3 positionWS : TEXCOORD0;
                float3 normalWS : TEXCOORD1;
                float3 barycentric : TEXCOORD2;
                float3 diagnosticColor : TEXCOORD3;
                nointerpolation uint diagnosticClass : TEXCOORD4;
                nointerpolation uint legacyDiagnosticClass : TEXCOORD5;
                nointerpolation uint diagnosticSupportMask : TEXCOORD6;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            uint LegacyDiagnosticClass(uint vertID)
            {
                // Resolve one class for the complete output triangle so a mixed
                // triangle is not disguised by interpolated vertex colours.
                uint triBase = (vertID / 3u) * 3u;
                uint c0 = _VertexAdmissionClass[_SurfaceIndices[triBase] & 0x3FFFFFFFu];
                uint c1 = _VertexAdmissionClass[_SurfaceIndices[triBase + 1u] & 0x3FFFFFFFu];
                uint c2 = _VertexAdmissionClass[_SurfaceIndices[triBase + 2u] & 0x3FFFFFFFu];
                uint s0 = c0 & 7u, s1 = c1 & 7u, s2 = c2 & 7u;
                uint q0 = (c0 >> 3u) & 3u, q1 = (c1 >> 3u) & 3u, q2 = (c2 >> 3u) & 3u;

                bool sourceMixed = s0 != s1 || s0 != s2 ||
                                   s0 == 4u || s1 == 4u || s2 == 4u;
                bool confirmationMixed = q0 != q1 || q0 != q2 ||
                                         q0 == 3u || q1 == 3u || q2 == 3u;

                // White is split into three disjoint reasons.  This remains
                // read-only: the class only affects colour and pink isolation.
                if (sourceMixed && confirmationMixed)
                {
                    bool admissionInternal = s0 == 4u || s1 == 4u || s2 == 4u;
                    bool confirmationInternal = q0 == 3u || q1 == 3u || q2 == 3u;
                    bool hasPending = q0 == 1u || q1 == 1u || q2 == 1u;
                    if (admissionInternal && confirmationInternal)
                    {
                        uint doubleMixedCount =
                            ((s0 == 4u && q0 == 3u) ? 1u : 0u) +
                            ((s1 == 4u && q1 == 3u) ? 1u : 0u) +
                            ((s2 == 4u && q2 == 3u) ? 1u : 0u);
                        if (doubleMixedCount == 0u) return 15u; // mixed states live on different vertices
                        return 15u + doubleMixedCount;          // 16/17/18 = 1/2/3 double-mixed vertices
                    }
                    if (admissionInternal) return 13u;
                    if (confirmationInternal) return 14u;
                    if (hasPending) return 2u;
                    return 1u;
                }
                if (sourceMixed) return 4u;
                if (confirmationMixed) return 5u;

                // A single green class used to hide how confirmed geometry
                // entered the mesh.  Split it by immutable admission source.
                if (q0 == 2u && s0 == 1u) return 6u;
                if (q0 == 2u && s0 == 2u) return 7u;
                if (q0 == 2u && s0 == 3u) return 8u;
                if (q0 == 2u) return 9u;
                if (q0 == 1u && s0 == 3u) return 10u;
                if (q0 == 1u) return 11u;
                return 12u;
            }

            uint TemporalDiagnosticClass(uint vertID, out uint supportMask)
            {
                uint triBase = (vertID / 3u) * 3u;
                uint encoded0 = _SurfaceIndices[triBase];
                uint encoded1 = _SurfaceIndices[triBase + 1u];
                uint encoded2 = _SurfaceIndices[triBase + 2u];
                uint c0 = _VertexAdmissionClass[encoded0 & 0x3FFFFFFFu];
                uint c1 = _VertexAdmissionClass[encoded1 & 0x3FFFFFFFu];
                uint c2 = _VertexAdmissionClass[encoded2 & 0x3FFFFFFFu];

                uint e0 = (c0 >> 18u) & 3u;
                uint e1 = (c1 >> 18u) & 3u;
                uint e2 = (c2 >> 18u) & 3u;
                uint z0 = (c0 >> 20u) & 3u;
                uint z1 = (c1 >> 20u) & 3u;
                uint z2 = (c2 >> 20u) & 3u;

                supportMask = (e0 == 0u ? 1u : 0u) |
                              (e1 == 0u ? 2u : 0u) |
                              (e2 == 0u ? 4u : 0u);

                // The extraction kernel stores one persistent candidate class in
                // the high two bits of every triangle index:
                // 0 hidden/pending, 1 supported edge only, 2 mature, 3 grace-held.
                return encoded0 >> 30u;
            }

            float3 TemporalDiagnosticColor(uint diagnosticClass)
            {
                // Candidate production B is one green product.  Class 1 is not
                // an error colour: it is the supported boundary edge of a
                // never-mature 2/3 triangle, while 2/3 are full/grace surfaces.
                return float3(0.10, 1.00, 0.25);
            }

            float3 HeraRouteColor(bool delegated)
            {
                return delegated ? float3(1.0, 0.18, 0.32) : float3(0.12, 1.0, 0.28);
            }

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings OUT = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);

                // 08-18 大网眼抽稀（用户方案）：只画落在粗格线带上的三角形，
                // 其余三顶点输出同一裁剪外坐标=零面积被光栅化整批丢弃。
                // 08-19 实机判定：条带抽稀观感先天残疾（5cm 微三角形锯齿条带，
                // 做不出 Meta 几何级粗网），默认回 1=关闭，仅留作帧率应急杠杆；
                // 观感修复改由片元级世界格线画法承担（见 frag 线框分支）。
                uint meshStride = (uint)max(_RSMeshStride, 1.0);
                if (meshStride > 1u)
                {
                    uint anchorFlat = _SurfaceVerts[_SurfaceIndices[(vertID / 3u) * 3u] & 0x3FFFFFFFu].voxelFlatIdx;
                    uint3 vc3 = (uint3)gsVoxCount.xyz;
                    uint sliceXY = vc3.x * vc3.y;
                    uint vz = anchorFlat / sliceXY;
                    uint vrem = anchorFlat - vz * sliceXY;
                    uint vy = vrem / vc3.x;
                    uint vx = vrem - vy * vc3.x;
                    bool onGrid = vx % meshStride == 0u || vy % meshStride == 0u || vz % meshStride == 0u;
                    if (!onGrid)
                    {
                        OUT.positionHCS = float4(2.0, 2.0, 2.0, 1.0);
                        return OUT;
                    }
                }

                uint encoded = _SurfaceIndices[vertID];
                uint idx = encoded & 0x3FFFFFFFu;
                GPUVertex gv = _SurfaceVerts[idx];

                OUT.positionWS  = gv.pos;
                OUT.positionHCS = TransformWorldToHClip(gv.pos);
                OUT.normalWS    = gv.norm;
                OUT.color       = UnpackColor(gv.packedColor);
                OUT.legacyDiagnosticClass = LegacyDiagnosticClass(vertID);
                OUT.diagnosticClass = TemporalDiagnosticClass(vertID, OUT.diagnosticSupportMask);
                OUT.diagnosticColor = TemporalDiagnosticColor(OUT.diagnosticClass);

                // Barycentric coords for wireframe: each triangle vertex gets one axis
                uint triVert = vertID % 3;
                OUT.barycentric = triVert == 0 ? float3(1, 0, 0)
                                : triVert == 1 ? float3(0, 1, 0)
                                :                float3(0, 0, 1);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                bool heraDelegated = _RSHeraReplayActive > 0.5 && IN.diagnosticClass == 3u;

                // Pink is quarantined at presentation only.  Its evidence and
                // counters remain available, and it can reappear after its
                // confirmation class changes on a later extraction.
                if (_RSHeraReplayActive < 0.5 && _RSJointDiagnostic < 0.5 &&
                    _RSSuppressPink > 0.5 && IN.legacyDiagnosticClass == 5u)
                    discard;

                // Strict production B keeps pending/retired triangles in the
                // common buffers for audit, but never presents them.
                if (_RSJointDiagnostic > 0.5 && IN.diagnosticClass == 0u)
                    discard;

            // 3. Wireframe: discard interior, white edges blending to vertex color at vertices
            // 08-18 晚帧率手术：整块上移到 baseColor/freeze-tint 之前——线框分支
            // 根本不读 baseColor 和 normal，原顺序让每根线框像素（含被 discard 的
            // 内部片元）白跑三平面 6 次纹理采样+冻结着色 3D 采样再扔掉。
            // Class 1 was the retired 2/3 leniency path.  Production no longer
            // emits it; retaining this guard keeps old GPU buffers harmless.
            bool diagnosticBoundaryOnly = _RSJointDiagnostic > 0.5 &&
                                          IN.diagnosticClass == 1u;
            if (_RSProductGridMode > 0.5)
            {
                // Geometry remains the complete native 5 cm candidate. Only
                // the visible ink is a world-anchored 10 cm triangular grid;
                // unlike index aliasing this cannot tear or stretch topology.
                float grid = ProductTriangleGrid(IN.positionWS, IN.normalWS);
                if (grid < 0.08)
                    discard;
                return half4(0.95, 0.95, 0.95, saturate(grid));
            }
            if (_RSPaperGridMode > 0.5)
            {
                // In the hybrid route HERA contributes only the more precise
                // surface position/normal. Coverage and the visible line
                // pattern come from the same world-anchored paper grid used by
                // SupportTruth, so page boundaries cannot define visual holes.
                float grid = PaperTriangleGrid(IN.positionWS, IN.normalWS);
                half3 paper = half3(0.11, 0.15, 0.18);
                return half4(lerp(paper, half3(0.96, 0.98, 1.0), saturate(grid)), 1.0);
            }

            if (_RSWireframe > 0.5 || diagnosticBoundaryOnly)
            {
                float thickness = max(_RSWireThickness, 0.2);

                if (diagnosticBoundaryOnly)
                {
                    // A boundary triangle with only two supported vertices must
                    // not expose its unsupported sides.  Keep just the edge that
                    // joins the two currently depth-supported endpoints.
                    float3 bary = IN.barycentric;
                    float3 dx = ddx(bary);
                    float3 dy = ddy(bary);
                    float3 edgeWidth = sqrt(dx * dx + dy * dy);
                    float3 edge = smoothstep(0.0, edgeWidth * thickness, bary);
                    float minEdge = min(edge.x, min(edge.y, edge.z));
                    uint mask = IN.diagnosticSupportMask;
                    minEdge = mask == 3u ? edge.z :
                              mask == 5u ? edge.y :
                              mask == 6u ? edge.x : 1.0;

                    // Discard interior — threshold scales inversely with thickness
                    float discardThreshold = saturate(1.0 - thickness * 0.15);
                    if (minEdge > discardThreshold)
                        discard;
                }
                else
                {
                    // 08-19 回退：恢复重心坐标密集真边线框（08-18 晚版式）。
                    // 碎网观感战役两连败记录：①顶点侧条带抽稀=5cm 微三角形锯齿
                    // 条带，先天做不出 Meta 几何级粗网；②片元级世界格线画法=
                    // 实机判定观感不佳，且大网眼不利于观察底层细节。结论：Meta
                    // 的粗网必须几何级生成（路线A粗皮已落地备用，默认关），
                    // 诊断期主显示就用密集真边。注意：纸主网格的新世界格线
                    // 由稳定支撑层兜底覆盖，已不再受这里所述的 HERA 页面缺页约束。
                    float3 bary = IN.barycentric;
                    float3 dx = ddx(bary);
                    float3 dy = ddy(bary);
                    float3 edgeWidth = sqrt(dx * dx + dy * dy);
                    float3 edge = smoothstep(0.0, edgeWidth * thickness, bary);
                    float minEdge = min(edge.x, min(edge.y, edge.z));

                    // Discard interior — threshold scales inversely with thickness
                    float discardThreshold = saturate(1.0 - thickness * 0.15);
                    if (minEdge > discardThreshold)
                        discard;
                }

                // Display-only A/B color supplied by GPUMeshRenderer:
                // production = orange, strict-observed = green.
                float3 lineColor = _RSHeraReplayActive > 0.5
                    ? HeraRouteColor(heraDelegated)
                    : _RSJointDiagnostic > 0.5
                        ? IN.diagnosticColor
                        : _RSExtractionColor.rgb;
                // 第一阶段纯几何观察：统一白线，明确绕开置信色、冻结色和
                // HERA 路由色。它只改变最终片元，不改变任何数据或提取决策。
                lineColor = _RSGeometryTruthView > 0.5
                    ? float3(0.95, 0.95, 0.95)
                    : ApplyConfidenceViz(lineColor, IN.positionWS);
                return half4(lineColor, _RSExtractionColor.a);
            }

                if (_RSGeometryTruthView > 0.5)
                    return half4(0.82, 0.82, 0.82, 1.0);

                float3 normal = normalize(IN.normalWS);

                // 1. Compute base color
                half3 baseColor;
                if (_RSTriAvailable > 0.5)
                {
                    half3 tri = SampleTriplanar(IN.positionWS, normal);
                    baseColor = tri.r >= 0 ? tri : IN.color.rgb;
                }
                else if (_RSNormalFallback > 0.5)
                {
                    baseColor = half3(normal * 0.5 + 0.5);
                }
                else
                {
                    baseColor = IN.color.rgb;
                }

                // 2. Apply freeze tint
                baseColor = ApplyFreezeTint(baseColor, IN.positionWS);
                baseColor = ApplyConfidenceViz(baseColor, IN.positionWS);

                return _RSHeraReplayActive > 0.5
                    ? half4(HeraRouteColor(heraDelegated), 1)
                    : _RSJointDiagnostic > 0.5
                        ? half4(IN.diagnosticColor, 1)
                        : half4(baseColor, 1);
            }
            ENDHLSL
        }

        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode"="DepthOnly" }
            ZWrite On
            ColorMask 0
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct GPUVertex
            {
                float3 pos;
                float3 norm;
                uint   packedColor;
                uint   voxelFlatIdx;
            };
            StructuredBuffer<GPUVertex> _SurfaceVerts;
            StructuredBuffer<uint>      _SurfaceIndices;
            float _RSJointDiagnostic;
            float _RSHeraReplayActive;

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                nointerpolation uint candidateClass : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings OUT = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);
                uint encoded = _SurfaceIndices[vertID];
                uint idx = encoded & 0x3FFFFFFFu;
                OUT.positionHCS = TransformWorldToHClip(_SurfaceVerts[idx].pos);
                OUT.candidateClass = encoded >> 30u;
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);
                if (_RSJointDiagnostic > 0.5 && IN.candidateClass < 2u)
                    discard;
                return 0;
            }
            ENDHLSL
        }

        Pass
        {
            Name "DepthNormals"
            Tags { "LightMode"="DepthNormals" }
            ZWrite On
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct GPUVertex
            {
                float3 pos;
                float3 norm;
                uint   packedColor;
                uint   voxelFlatIdx;
            };
            StructuredBuffer<GPUVertex> _SurfaceVerts;
            StructuredBuffer<uint>      _SurfaceIndices;
            float _RSJointDiagnostic;
            float _RSHeraReplayActive;

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 normalWS : TEXCOORD0;
                nointerpolation uint candidateClass : TEXCOORD1;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings OUT = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);
                uint encoded = _SurfaceIndices[vertID];
                uint idx = encoded & 0x3FFFFFFFu;
                GPUVertex gv = _SurfaceVerts[idx];
                OUT.positionHCS = TransformWorldToHClip(gv.pos);
                OUT.normalWS    = gv.norm;
                OUT.candidateClass = encoded >> 30u;
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);
                if (_RSJointDiagnostic > 0.5 && IN.candidateClass < 2u)
                    discard;
                float3 n = normalize(IN.normalWS);
                return half4(n * 0.5 + 0.5, 1);
            }
            ENDHLSL
        }
    }
}
