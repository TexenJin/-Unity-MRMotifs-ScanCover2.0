Shader "Genesis/RoomSetupReplicaV2GpuMarchingCubes"
{
    Properties
    {
        _BaseColor ("Color", Color) = (0.02, 0.86, 1.0, 0.88)
        _LineWidthPixels ("Line Width", Range(0.75, 3.0)) = 1.5
        _LineReferenceDistance ("Full Width Distance", Float) = 0.5
        _UniqueEdgesOnly ("Unique Edges Only", Float) = 0
    }
    SubShader
    {
        Tags
        {
            "RenderType"="Transparent"
            "Queue"="Transparent+45"
            "RenderPipeline"="UniversalPipeline"
        }
        Pass
        {
            Name "RoomSetupGpuMcWire"
            Tags { "LightMode"="SRPDefaultUnlit" }
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            ZTest LEqual
            Cull Off

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma target 4.5
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            StructuredBuffer<float3> _RoomSetupMcCandidateVertices;
            StructuredBuffer<float3> _RoomSetupMcFilteredVertices;
            StructuredBuffer<uint> _RoomSetupMcCounters;
            StructuredBuffer<uint> _RoomSetupMcUniqueEdgeMasks;

            struct Attributes
            {
                uint vertexID : SV_VertexID;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 barycentric : TEXCOORD0;
                nointerpolation uint edgeMask : TEXCOORD1;
                float3 positionWS : TEXCOORD2;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            CBUFFER_START(UnityPerMaterial)
                half4 _BaseColor;
                float _LineWidthPixels;
                float _LineReferenceDistance;
                float _UniqueEdgesOnly;
            CBUFFER_END

            float3 LoadPosition(uint vertexID)
            {
                return _RoomSetupMcCounters[6] != 0u
                    ? _RoomSetupMcFilteredVertices[vertexID]
                    : _RoomSetupMcCandidateVertices[vertexID];
            }

            uint BuildUniqueEdgeMask(uint triangleBase)
            {
                if (_UniqueEdgesOnly < 0.5)
                    return 7u;
                return _RoomSetupMcUniqueEdgeMasks[triangleBase / 3u];
            }

            Varyings Vert(Attributes input)
            {
                Varyings output = (Varyings)0;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                uint triangleBase = input.vertexID - input.vertexID % 3u;
                float3 positionWS = LoadPosition(input.vertexID);
                output.positionHCS = TransformWorldToHClip(positionWS);
                output.positionWS = positionWS;
                output.edgeMask = BuildUniqueEdgeMask(triangleBase);
                uint corner = input.vertexID % 3u;
                output.barycentric = corner == 0u
                    ? float3(1, 0, 0)
                    : corner == 1u
                        ? float3(0, 1, 0)
                        : float3(0, 0, 1);
                return output;
            }

            half4 Frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                if (input.edgeMask == 0u)
                    discard;
                float edgeDistance = 2.0;
                // bit 0 owns v0-v1 (barycentric.z == 0), bit 1 owns
                // v1-v2 (x == 0), and bit 2 owns v2-v0 (y == 0).
                if ((input.edgeMask & 1u) != 0u)
                    edgeDistance = min(
                        edgeDistance, input.barycentric.z);
                if ((input.edgeMask & 2u) != 0u)
                    edgeDistance = min(
                        edgeDistance, input.barycentric.x);
                if ((input.edgeMask & 4u) != 0u)
                    edgeDistance = min(
                        edgeDistance, input.barycentric.y);
                // _LineWidthPixels used directly would keep a constant screen
                // footprint, so it becomes progressively heavier relative to
                // the shrinking projected mesh. Convert it to an inverse-
                // distance width, approximating a stable world-space stroke.
                float distanceWS = distance(
                    GetCameraPositionWS(), input.positionWS);
                float widthPixels = _LineWidthPixels *
                    min(1.0, _LineReferenceDistance /
                        max(distanceWS, 0.0001));

                // Preserve a one-pixel filtering footprint for sub-pixel
                // strokes, while scaling their energy to the requested width.
                float rasterWidthPixels = max(widthPixels, 1.0);
                float subpixelCoverage = min(widthPixels, 1.0);
                float width = max(
                    fwidth(edgeDistance) * rasterWidthPixels,
                                  0.00001);
                float coverage = 1.0 - smoothstep(width, width * 1.8,
                                                  edgeDistance);
                coverage *= subpixelCoverage;
                clip(coverage - 0.01);
                return half4(_BaseColor.rgb,
                             _BaseColor.a * coverage);
            }
            ENDHLSL
        }
    }
}
