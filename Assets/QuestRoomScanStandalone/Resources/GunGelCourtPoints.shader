// 枪胶裁决海面：GPU 候选/浪头缓冲的纯显示视图。
// 绿色只画存活稳定候选，红色只画按 stable ID 锁存的异常观测。
Shader "QRS/GunGelCourtPoints"
{
    Properties
    {
        _PointSize ("Point size (px)", Float) = 4
        _PointColor ("Point color", Color) = (0,1,0,1)
    }
    SubShader
    {
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

            StructuredBuffer<float4> _CandidateCenterSigma;
            StructuredBuffer<uint4> _CandidateMeta;
            StructuredBuffer<uint4> _CandidateEvidence;
            StructuredBuffer<uint> _WavePeakBits;
            StructuredBuffer<float4> _WavePositionResidual;
            StructuredBuffer<uint4> _WaveMeta;

            int _CourtMode;
            float _PointSize;
            float4 _PointColor;

            struct v2f
            {
                float4 pos : SV_POSITION;
                float4 color : COLOR0;
                float psize : PSIZE;
            };

            v2f HiddenPoint()
            {
                v2f o;
                o.pos = float4(0, 0, -10, 1);
                o.color = 0;
                o.psize = 0;
                return o;
            }

            v2f vert(uint vertexID : SV_VertexID)
            {
                float3 world;
                if (_CourtMode == 0)
                {
                    uint4 meta = _CandidateMeta[vertexID];
                    uint stableID = _CandidateEvidence[vertexID].z;
                    if (meta.x == 0u || meta.y == 0u || stableID == 0u)
                        return HiddenPoint();
                    world = _CandidateCenterSigma[vertexID].xyz;
                }
                else
                {
                    uint4 waveMeta = _WaveMeta[vertexID];
                    if (waveMeta.x == 0u || _WavePeakBits[vertexID] == 0u)
                        return HiddenPoint();
                    world = _WavePositionResidual[vertexID].xyz;
                }

                v2f o;
                o.pos = UnityWorldToClipPos(world);
                o.color = _PointColor;
                o.psize = _PointSize;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                return i.color;
            }
            ENDCG
        }
    }
}
