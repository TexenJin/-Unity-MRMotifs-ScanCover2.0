Shader "Genesis/ScanCoarseSkin"
{
    // 路线验证：把 CoarseSkinExtract 直接从 TSDF 提取的粗表面画成连续半透明皮。
    // 它不继承 HERA 页、诊断颜色或线框语义，专门暴露底层支架自身的覆盖、
    // 鼓包、双层与抖动。ZWrite On 避免透明表面反复叠加。
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
            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings OUT = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);

                uint idx = _SkinIndices[vertID];
                float3 worldPos = _SkinVerts[idx].xyz;
                OUT.positionHCS = TransformWorldToHClip(worldPos);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                // 路线验证显示的是连续纯色蒙皮，而不是另一套线框。这样可直接观察
                // TSDF 支架本身的覆盖、鼓包和抖动，不再把三角边与页提交误认成数据层。
                return half4(_SkinColor.rgb, _SkinColor.a);
            }
            ENDHLSL
        }
    }
}
