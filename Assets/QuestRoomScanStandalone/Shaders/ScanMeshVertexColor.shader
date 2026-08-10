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
            // 对齐原 SC 工程 B 链线框外观：alpha 混合、不写深度（透视 cage 效果）。
            // 实体模式是调试档，在透明队列下自遮挡略弱，可接受。
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            ZTest LEqual

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

            // ── Globals set by RoomScanner ──
            float _RSNoFreezeTint;
            float _RSNormalFallback;
            float _RSWireframe;
            float _RSWireThickness;
            float4 _RSExtractionColor;
            float _RSJointDiagnostic;
            float _RSSuppressPink;
            float _RSTemporalIllegalActive;

            #define DEPTH_TOLERANCE 0.015

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

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings OUT = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);

                uint idx = _SurfaceIndices[vertID] & 0x3FFFFFFFu;
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

                // Pink is quarantined at presentation only.  Its evidence and
                // counters remain available, and it can reappear after its
                // confirmation class changes on a later extraction.
                if (_RSJointDiagnostic < 0.5 && _RSSuppressPink > 0.5 && IN.legacyDiagnosticClass == 5u)
                    discard;

                // Strict production B keeps pending/retired triangles in the
                // common buffers for audit, but never presents them.
                if (_RSJointDiagnostic > 0.5 && IN.diagnosticClass == 0u)
                    discard;

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

            // 3. Wireframe: discard interior, white edges blending to vertex color at vertices
            // Class 1 was the retired 2/3 leniency path.  Production no longer
            // emits it; retaining this guard keeps old GPU buffers harmless.
            bool diagnosticBoundaryOnly = _RSJointDiagnostic > 0.5 &&
                                          IN.diagnosticClass == 1u;
            if (_RSWireframe > 0.5 || diagnosticBoundaryOnly)
            {
                float thickness = max(_RSWireThickness, 0.2);
                float3 bary = IN.barycentric;
                float3 dx = ddx(bary);
                float3 dy = ddy(bary);
                float3 edgeWidth = sqrt(dx * dx + dy * dy);
                float3 edge = smoothstep(0.0, edgeWidth * thickness, bary);
                float minEdge = min(edge.x, min(edge.y, edge.z));

                // A boundary triangle with only two supported vertices must
                // not expose its unsupported sides.  Keep just the edge that
                // joins the two currently depth-supported endpoints.
                if (diagnosticBoundaryOnly)
                {
                    uint mask = IN.diagnosticSupportMask;
                    minEdge = mask == 3u ? edge.z :
                              mask == 5u ? edge.y :
                              mask == 6u ? edge.x : 1.0;
                }

                // Discard interior — threshold scales inversely with thickness
                float discardThreshold = saturate(1.0 - thickness * 0.15);
                if (minEdge > discardThreshold)
                    discard;

                // Display-only A/B color supplied by GPUMeshRenderer:
                // production = orange, strict-observed = green.
                float3 lineColor = _RSJointDiagnostic > 0.5
                    ? IN.diagnosticColor
                    : _RSExtractionColor.rgb;
                return half4(lineColor, _RSExtractionColor.a);
            }

                return _RSJointDiagnostic > 0.5
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
