Shader "QRS/ObservationCoverageTiles"
{
    Properties
    {
        _TsdfVolume ("TSDF volume", 3D) = "" {}
        _WarmupTint ("Warm-up tint", Color) = (0.35,0.55,0.72,0.42)
        _PendingTint ("Observed but pending", Color) = (1,0.76,0.18,0.55)
        _ReadyTint ("Mesh ready", Color) = (0.18,1,0.40,0.58)
        _TileHalfSize ("Tile half size (m)", Float) = 0.014
        _EyeIndex ("Depth eye", Float) = 1
        _ReadinessEnabled ("TSDF readiness enabled", Float) = 0
        _VoxSize ("Voxel size", Float) = 0.05
        _MinWeight ("Mesh weight", Float) = 0.08
    }
    SubShader
    {
        Tags { "Queue"="Transparent+20" "RenderType"="Transparent" "IgnoreProjector"="True" }
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
            Texture3D<float2> _TsdfVolume;
            float4 _WarmupTint;
            float4 _PendingTint;
            float4 _ReadyTint;
            float _TileHalfSize;
            float _EyeIndex;
            float _ReadinessEnabled;
            float3 _VoxCount;
            float _VoxSize;
            float _MinWeight;

            static const int3 kCornerOffs[8] =
            {
                int3(0,0,0), int3(1,0,0), int3(1,0,1), int3(0,0,1),
                int3(0,1,0), int3(1,1,0), int3(1,1,1), int3(0,1,1)
            };
            static const uint kEdgeA[12] = { 0,1,2,3, 4,5,6,7, 0,1,2,3 };
            static const uint kEdgeB[12] = { 1,2,3,0, 5,6,7,4, 4,5,6,7 };
            static const int3 kCellProbeOffs[7] =
            {
                int3(0,0,0), int3(-1,0,0), int3(1,0,0),
                int3(0,-1,0), int3(0,1,0), int3(0,0,-1), int3(0,0,1)
            };

            struct appdata { float4 vertex : POSITION; float2 corner : TEXCOORD0; };
            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 corner : TEXCOORD0;
                float ready : TEXCOORD1;
            };

            bool CellCanEmit(int3 cell)
            {
                int3 maxCell = int3(_VoxCount) - 2;
                if (any(cell < 0) || any(cell > maxCell)) return false;

                uint crossings = 0;
                uint badCrossings = 0;
                [unroll]
                for (uint e = 0; e < 12; e++)
                {
                    int3 cA = cell + kCornerOffs[kEdgeA[e]];
                    int3 cB = cell + kCornerOffs[kEdgeB[e]];
                    float2 sA = _TsdfVolume.Load(int4(cA, 0));
                    float2 sB = _TsdfVolume.Load(int4(cB, 0));
                    bool emptyA = sA.x <= -0.99 || abs(sA.y) < _MinWeight;
                    bool emptyB = sB.x <= -0.99 || abs(sB.y) < _MinWeight;
                    float valA = emptyA ? 0.0 : sA.x;
                    float valB = emptyB ? 0.0 : sB.x;
                    if ((valA < 0.0) != (valB < 0.0))
                    {
                        crossings++;
                        if (emptyA || emptyB) badCrossings++;
                    }
                }
                // Keep this byte-for-byte equivalent to production non-strict
                // Surface Nets classification.
                return crossings >= 3 && crossings != badCrossings;
            }

            float IsMeshReady(float3 world)
            {
                if (_ReadinessEnabled < 0.5 || _VoxSize <= 0.0) return 0.0;
                float3 centeredVoxel = world / _VoxSize + _VoxCount * 0.5 - 0.5;
                int3 baseCell = int3(floor(centeredVoxel));
                [unroll]
                for (uint i = 0; i < 7; i++)
                    if (CellCanEmit(baseCell + kCellProbeOffs[i])) return 1.0;
                return 0.0;
            }

            float Linearize(float ndc, int eye)
            {
                float z = ndc * 2.0 - 1.0;
                return abs(gsDepthProj[eye][2][3] / (z + gsDepthProj[eye][2][2]));
            }

            v2f vert(appdata v)
            {
                v2f o;
                int eye = (int)round(_EyeIndex);
                float2 uv = v.vertex.xy;
                float ndc = UNITY_SAMPLE_TEX2DARRAY_LOD(gsDepthTex, float3(uv, eye), 0);
                float lin = Linearize(ndc, eye);
                if (!(lin > 0.12) || lin > 8.0)
                {
                    eye = 1 - eye;
                    ndc = UNITY_SAMPLE_TEX2DARRAY_LOD(gsDepthTex, float3(uv, eye), 0);
                    lin = Linearize(ndc, eye);
                }
                if (!(lin > 0.12) || lin > 8.0)
                {
                    o.pos = float4(0, 0, -10, 1);
                    o.corner = 0;
                    o.ready = 0;
                    return o;
                }

                float4 hcs = float4(float3(uv, ndc) * 2.0 - 1.0, 1.0);
                float4 worldH = mul(gsDepthViewInv[eye], mul(gsDepthProjInv[eye], hcs));
                float3 world = worldH.xyz / worldH.w;
                float ready = IsMeshReady(world);
                // HLSL matrix indexing is row-major here, while the camera basis
                // lives in the inverse-view columns.  Read the columns explicitly
                // so the tiles cannot shear when the headset rotates.
                float3 right = normalize(float3(gsDepthViewInv[eye]._m00, gsDepthViewInv[eye]._m10, gsDepthViewInv[eye]._m20));
                float3 up = normalize(float3(gsDepthViewInv[eye]._m01, gsDepthViewInv[eye]._m11, gsDepthViewInv[eye]._m21));
                world += (right * v.corner.x + up * v.corner.y) * _TileHalfSize;
                o.pos = UnityWorldToClipPos(world);
                o.corner = v.corner;
                // All four corners of one marker must share the readiness of
                // its depth sample; the cosmetic quad offset must not move the
                // lookup into a neighbouring voxel cell.
                o.ready = ready;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float edge = max(abs(i.corner.x), abs(i.corner.y));
                float4 tint = _ReadinessEnabled < 0.5
                    ? _WarmupTint
                    : lerp(_PendingTint, _ReadyTint, step(0.5, i.ready));
                float alpha = tint.a * (1.0 - smoothstep(0.72, 1.0, edge));
                return fixed4(tint.rgb, alpha);
            }
            ENDCG
        }
    }
}
