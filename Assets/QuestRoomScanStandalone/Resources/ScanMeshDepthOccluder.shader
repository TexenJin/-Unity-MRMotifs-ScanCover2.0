Shader "Genesis/ScanMeshDepthOccluder"
{
    SubShader
    {
        Tags
        {
            "RenderType"="Transparent"
            "RenderPipeline"="UniversalPipeline"
            "Queue"="Transparent+39"
        }

        Pass
        {
            Name "NearestSurfaceDepth"
            Tags { "LightMode"="SRPDefaultUnlit" }

            Blend Off
            ColorMask 0
            ZWrite On
            ZTest LEqual
            // The colour wire pass draws the exact same front surface one
            // queue later.  A one-unit rear depth bias keeps this invisible
            // shell behind its own wire, while remaining far in front of the
            // genuinely displaced sheets it is meant to hide.
            Offset 0, 1
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
            StructuredBuffer<uint> _SurfaceIndices;
            float4 gsVoxCount;
            float _RSMeshStride;
            float _RSJointDiagnostic;
            float _RSDepthPrepassMaxViewDistance;

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                nointerpolation uint diagnosticClass : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings output = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

                uint triBase = (vertID / 3u) * 3u;
                uint encoded = _SurfaceIndices[vertID];
                uint index = encoded & 0x3FFFFFFFu;

                uint meshStride = (uint)max(_RSMeshStride, 1.0);
                if (meshStride > 1u)
                {
                    uint anchorIndex = _SurfaceIndices[triBase] & 0x3FFFFFFFu;
                    uint anchorFlat = _SurfaceVerts[anchorIndex].voxelFlatIdx;
                    uint3 voxelCount = (uint3)gsVoxCount.xyz;
                    uint sliceXY = voxelCount.x * voxelCount.y;
                    uint vz = anchorFlat / sliceXY;
                    uint remainder = anchorFlat - vz * sliceXY;
                    uint vy = remainder / voxelCount.x;
                    uint vx = remainder - vy * voxelCount.x;
                    bool onGrid = vx % meshStride == 0u ||
                                  vy % meshStride == 0u ||
                                  vz % meshStride == 0u;
                    if (!onGrid)
                    {
                        output.positionHCS = float4(2.0, 2.0, 2.0, 1.0);
                        return output;
                    }
                }

                float3 positionWS = _SurfaceVerts[index].pos;
                output.positionHCS = TransformWorldToHClip(positionWS);

                // Ordinary true-line wireframes are display-limited. Match
                // that same range here so invisible distant surfaces do not
                // continue paying a full-triangle depth cost. Other mesh modes
                // pass zero and retain their established unlimited prepass.
                float maxDistance = _RSDepthPrepassMaxViewDistance;
                if (maxDistance > 0.0)
                {
                    float3 viewDelta = positionWS - _WorldSpaceCameraPos.xyz;
                    if (dot(viewDelta, viewDelta) > maxDistance * maxDistance)
                    {
                        #if UNITY_REVERSED_Z
                            output.positionHCS.z = -output.positionHCS.w;
                        #else
                            output.positionHCS.z = output.positionHCS.w * 2.0;
                        #endif
                        return output;
                    }
                }

                output.diagnosticClass = encoded >> 30u;
                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                if (_RSJointDiagnostic > 0.5 && input.diagnosticClass == 0u)
                    discard;
                return 0;
            }
            ENDHLSL
        }
    }
}
