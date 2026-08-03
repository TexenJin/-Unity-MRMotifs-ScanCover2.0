Shader "ScanCover/QuestRoomSurfaceNetsWire"
{
    Properties
    {
        _WireColor("Wire Color", Color) = (1.0, 0.62, 0.02, 1.0)
        _EvidenceDiagnosticMode("Evidence Diagnostic Mode", Float) = 1
    }
    SubShader
    {
        Tags { "Queue"="Transparent+40" "RenderType"="Transparent" "RenderPipeline"="UniversalPipeline" }
        Pass
        {
            Name "QuestRoomSurfaceNetsWire"
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            ZTest LEqual
            Cull Off

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma target 4.5
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct SurfaceVertex
            {
                float3 position;
                float3 normal;
                uint voxelIndex;
                uint padding;
            };

            StructuredBuffer<SurfaceVertex> _SurfaceVertices;
            StructuredBuffer<uint> _SurfaceLineIndices;
            float4x4 _VolumeLocalToWorld;
            float4 _WireColor;
            float _EvidenceDiagnosticMode;

            static const uint EvidenceDirect = 1u << 0;
            static const uint EvidenceUnknownCrossing = 1u << 1;
            static const uint EvidenceRear = 1u << 2;
            static const uint EvidenceStaleDirect = 1u << 3;
            static const uint EvidenceBackCap = 1u << 4;
            static const uint EvidenceFreeContradicted = 1u << 5;

            float4 EvidenceColor(uint flags)
            {
                // Priority matches the mutually-exclusive audit counters.
                if ((flags & EvidenceFreeContradicted) != 0)
                    return float4(1.0, 0.03, 0.03, 0.99); // observed free-space contradiction
                if ((flags & EvidenceBackCap) != 0)
                    return float4(1.0, 0.05, 0.72, 0.98); // likely rear-band cap
                if ((flags & EvidenceStaleDirect) != 0)
                    return float4(0.12, 1.0, 0.22, 0.98); // retained old direct support
                if ((flags & EvidenceUnknownCrossing) != 0)
                    return float4(1.0, 0.86, 0.05, 0.98); // unknown-assisted
                if ((flags & EvidenceDirect) != 0)
                    return float4(0.12, 1.0, 0.22, 0.98); // direct surface support
                if ((flags & EvidenceRear) != 0)
                    return float4(0.20, 0.38, 1.0, 0.98); // rear evidence, no hit
                return float4(0.62, 0.62, 0.62, 0.90);   // unresolved
            }

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float4 color : COLOR0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings Vert(uint vertexID : SV_VertexID)
            {
                Varyings output = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
                uint surfaceIndex = _SurfaceLineIndices[vertexID];
                SurfaceVertex vertex = _SurfaceVertices[surfaceIndex];
                float3 worldPosition = mul(_VolumeLocalToWorld, float4(vertex.position, 1.0)).xyz;
                output.positionCS = TransformWorldToHClip(worldPosition);
                output.color = _EvidenceDiagnosticMode > 0.5
                    ? EvidenceColor(vertex.padding)
                    : _WireColor;
                return output;
            }

            half4 Frag(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                return input.color;
            }
            ENDHLSL
        }
    }
}
