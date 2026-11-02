// 把当前选择眼的 NDC 深度与同帧世界法线打包到一张 RGBAHalf 2D 快照。
// 只用于诊断留痕，永不回写生产深度或 TSDF。
// 法线直接从当前发布的深度重建，因此绕过生产预处理的 RoomSetupReplicaV2
// 也不会误读上一帧/未生成的 gsDepthNormalTex。
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
            float4x4 gsDepthProjInv[2];
            float4x4 gsDepthViewInv[2];
            float2 gsDepthTexSize;
            float _EyeIndex;

            float3 ReconstructWorld(float2 uv, float ndc, int eye)
            {
                float4 hcs = float4(float3(uv, ndc) * 2.0 - 1.0, 1.0);
                float4 worldH = mul(gsDepthViewInv[eye],
                                    mul(gsDepthProjInv[eye], hcs));
                return worldH.xyz / worldH.w;
            }

            float4 frag(v2f_img i) : SV_Target
            {
                int eye = (int)round(_EyeIndex);
                float ndc = UNITY_SAMPLE_TEX2DARRAY_LOD(gsDepthTex, float3(i.uv, eye), 0);
                float2 twoPixels = 2.0 / max(gsDepthTexSize, float2(1.0, 1.0));
                float2 uvH = min(i.uv + float2(twoPixels.x, 0.0),
                                 1.0 - 0.5 / gsDepthTexSize);
                float2 uvV = min(i.uv + float2(0.0, twoPixels.y),
                                 1.0 - 0.5 / gsDepthTexSize);
                float ndcH = UNITY_SAMPLE_TEX2DARRAY_LOD(gsDepthTex, float3(uvH, eye), 0);
                float ndcV = UNITY_SAMPLE_TEX2DARRAY_LOD(gsDepthTex, float3(uvV, eye), 0);
                float3 world = ReconstructWorld(i.uv, ndc, eye);
                float3 derivH = ReconstructWorld(uvH, ndcH, eye) - world;
                float3 derivV = ReconstructWorld(uvV, ndcV, eye) - world;
                float3 crossValue = cross(derivH, derivV);
                float crossLengthSq = dot(crossValue, crossValue);
                float3 normal = crossLengthSq > 1e-12
                    ? -crossValue * rsqrt(crossLengthSq)
                    : float3(0.0, 0.0, 0.0);
                return float4(ndc, normal);
            }
            ENDCG
        }
    }
}
