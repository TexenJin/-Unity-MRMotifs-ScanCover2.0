Shader "Genesis/ScanCoarseSkin"
{
    // 08-19 路线A：粗皮着色器。画 CoarseSkinExtract 产出的几何级大三角形网
    // （默认 20cm 网眼），重心坐标画真三角形边 = Meta 系统网格观感本尊。
    // 与细诊断网（ScanMeshVertexColor）完全独立：无诊断类、无三平面、无冻结
    // 着色——皮只要干净的线。ZWrite On 沿用 08-18 晚帧率手术结论（early-Z
    // 杀透明叠加）。
    Properties { }
    SubShader
    {
        Tags { "RenderType"="Transparent" "RenderPipeline"="UniversalPipeline" "Queue"="Transparent+40" }

        Pass
        {
            Name "CoarseSkinUnlit"
            Tags { "LightMode"="SRPDefaultUnlit" }
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite On
            ZTest LEqual
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            StructuredBuffer<float4> _SkinVerts;    // xyz = worldPos
            StructuredBuffer<uint>   _SkinIndices;

            float4 _SkinColor;        // MPB 下发（默认 Meta 风灰蓝）
            float _RSWireThickness;   // 复用细网线宽全局量（StandaloneRoomScanner 下发）

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 barycentric : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings OUT = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);

                uint idx = _SkinIndices[vertID];
                float3 worldPos = _SkinVerts[idx].xyz;
                OUT.positionHCS = TransformWorldToHClip(worldPos);

                uint triVert = vertID % 3u;
                OUT.barycentric = triVert == 0u ? float3(1, 0, 0)
                                : triVert == 1u ? float3(0, 1, 0)
                                :                float3(0, 0, 1);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                // 重心坐标边缘检测（与细网线框同一公式）：距最近边约 thickness
                // 像素宽内判在线上，fwidth 抗锯齿、屏宽恒定。20cm 大三角形下
                // 每条边都是笔直长线=Meta 观感。
                float thickness = max(_RSWireThickness, 0.2);
                float3 bary = IN.barycentric;
                float3 dx = ddx(bary);
                float3 dy = ddy(bary);
                float3 edgeWidth = sqrt(dx * dx + dy * dy);
                float3 edge = smoothstep(0.0, edgeWidth * thickness, bary);
                float minEdge = min(edge.x, min(edge.y, edge.z));

                float discardThreshold = saturate(1.0 - thickness * 0.15);
                if (minEdge > discardThreshold)
                    discard;

                return half4(_SkinColor.rgb, _SkinColor.a);
            }
            ENDHLSL
        }
    }
}
