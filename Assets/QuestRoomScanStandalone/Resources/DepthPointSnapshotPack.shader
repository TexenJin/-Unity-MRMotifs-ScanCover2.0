// 把当前选择眼的 NDC 深度与世界法线打包到一张 RGBAHalf 2D 快照。
// 只用于诊断留痕，永不回写生产深度或 TSDF。
Shader "QRS/DepthPointSnapshotPack"
{
    Properties { _EyeIndex ("Depth eye", Float) = 0 }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Pass
        {
            Cull Off ZWrite Off ZTest Always
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            #pragma target 4.5
            #include "UnityCG.cginc"

            UNITY_DECLARE_TEX2DARRAY(gsDepthTex);
            UNITY_DECLARE_TEX2DARRAY(gsDepthNormalTex);
            float _EyeIndex;

            float4 frag(v2f_img i) : SV_Target
            {
                int eye = (int)round(_EyeIndex);
                float ndc = UNITY_SAMPLE_TEX2DARRAY_LOD(gsDepthTex, float3(i.uv, eye), 0);
                float3 normal = UNITY_SAMPLE_TEX2DARRAY_LOD(gsDepthNormalTex, float3(i.uv, eye), 0).xyz;
                return float4(ndc, normal);
            }
            ENDCG
        }
    }
}
