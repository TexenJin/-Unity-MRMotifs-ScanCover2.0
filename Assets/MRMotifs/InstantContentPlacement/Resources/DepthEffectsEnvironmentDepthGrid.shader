Shader "MRMotifs/DepthEffectsEnvironmentDepthGrid"
{
    Properties
    {
        _GridColor ("Grid Color", Color) = (0.92, 0.96, 0.99, 0.85)
        _CellSizeMeters ("Cell Size Meters", Float) = 0.22
        _RepresentativeRadiusRatio ("Representative Radius Ratio", Range(0.0, 0.5)) = 0.18
        _LineHalfWidthMeters ("Line Half Width Meters", Float) = 0.012
        _DisplayMode ("Display Mode", Float) = 1
        _PatchFillRatio ("Patch Fill Ratio", Range(0.0, 1.0)) = 0.94
        _PatchConfidenceThreshold ("Patch Confidence Threshold", Range(0.0, 1.0)) = 0.55
        _ViewportRect ("Viewport Rect", Vector) = (0.5, 0.56, 0.72, 0.58)
        _SampleCounts ("Sample Counts", Vector) = (22, 16, 0, 0)
        _PatchCounts ("Patch Counts", Vector) = (4, 4, 3, 0)
        _PatchPlanarityThreshold ("Patch Planarity Threshold", Float) = 0.08
        _SurfaceOffset ("Surface Offset", Float) = 0.01
        _MinLinearDepth ("Min Linear Depth", Float) = 0.2
        _MaxLinearDepth ("Max Linear Depth", Float) = 3.5
        _AxisAlignmentThreshold ("Axis Alignment Threshold", Range(0.0, 1.0)) = 0.7
        _DepthEdgeSuppressStart ("Depth Edge Suppress Start", Float) = 0.06
        _DepthEdgeSuppressEnd ("Depth Edge Suppress End", Float) = 0.18
        _SurfaceStretchSuppressStart ("Surface Stretch Suppress Start", Float) = 0.18
        _SurfaceStretchSuppressEnd ("Surface Stretch Suppress End", Float) = 0.45
        _DebugUseScreenGridPlane ("Debug Use Screen Grid Plane", Float) = 0
    }

    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" "RenderPipeline"="UniversalPipeline" }
        ZWrite Off
        ZTest LEqual
        Cull Off
        Blend SrcAlpha OneMinusSrcAlpha

        Pass
        {
            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            CBUFFER_START(UnityPerMaterial)
                half4 _GridColor;
                float _CellSizeMeters;
                float _RepresentativeRadiusRatio;
                float _LineHalfWidthMeters;
                float _DisplayMode;
                float _PatchFillRatio;
                float _PatchConfidenceThreshold;
                float4 _ViewportRect;
                float4 _SampleCounts;
                float4 _PatchCounts;
                float _PatchPlanarityThreshold;
                float _SurfaceOffset;
                float _MinLinearDepth;
                float _MaxLinearDepth;
                float _AxisAlignmentThreshold;
                float _DepthEdgeSuppressStart;
                float _DepthEdgeSuppressEnd;
                float _SurfaceStretchSuppressStart;
                float _SurfaceStretchSuppressEnd;
                float _DebugUseScreenGridPlane;
            CBUFFER_END

            Texture2DArray<float> _EnvironmentDepthTexture;
            SamplerState bilinearClampSampler;
            float4 _EnvironmentDepthZBufferParams;
            float4x4 _EnvironmentDepthInverseReprojectionMatrices[2];

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
                float2 uv1 : TEXCOORD1;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 localCoord : TEXCOORD0;
                float2 patchLocalCoord : TEXCOORD4;
                half4 color : TEXCOORD1;
                float valid : TEXCOORD2;
                float confidence : TEXCOORD3;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            bool IsValidRawDepth(float value)
            {
                return value > 1e-5 && value < 0.99999;
            }

            float SampleRawDepth(float2 uv, int eye)
            {
                return _EnvironmentDepthTexture.SampleLevel(bilinearClampSampler, float3(saturate(uv), eye), 0).r;
            }

            float LinearizeDepth(float rawDepth)
            {
                float ndcDepth = rawDepth * 2.0 - 1.0;
                return (1.0 / (ndcDepth + _EnvironmentDepthZBufferParams.y)) * _EnvironmentDepthZBufferParams.x;
            }

            float3 ReconstructWorld(float2 uv, float rawDepth, int eye)
            {
                float4 clipPos = float4(uv * 2.0 - 1.0, rawDepth * 2.0 - 1.0, 1.0);
                float4 worldH = mul(_EnvironmentDepthInverseReprojectionMatrices[eye], clipPos);
                return worldH.xyz / max(1e-5, worldH.w);
            }

            float2 CenterUvFromIndex(float2 index, float2 counts)
            {
                float2 denom = max(counts, 1.0.xx);
                float2 uv01 = (index + 0.5.xx) / denom;
                return float2(
                    _ViewportRect.x + (uv01.x - 0.5) * _ViewportRect.z,
                    _ViewportRect.y + (uv01.y - 0.5) * _ViewportRect.w);
            }

            bool TrySampleWorld(float2 uv, int eye, out float linearDepth, out float3 worldPos)
            {
                linearDepth = 0.0;
                worldPos = 0.0.xxx;
                float rawDepth = SampleRawDepth(uv, eye);
                if (!IsValidRawDepth(rawDepth))
                    return false;

                linearDepth = LinearizeDepth(rawDepth);
                if (linearDepth < _MinLinearDepth || linearDepth > _MaxLinearDepth)
                    return false;

                worldPos = ReconstructWorld(uv, rawDepth, eye);
                return true;
            }

            bool TryBuildPatch(
                float2 patchBase,
                float2 patchSize,
                int eye,
                out float3 patchCenter,
                out float3 patchNormal,
                out float3 patchTangent,
                out float3 patchBitangent,
                out float confidence)
            {
                patchCenter = 0.0.xxx;
                patchNormal = 0.0.xxx;
                patchTangent = 0.0.xxx;
                patchBitangent = 0.0.xxx;
                confidence = 0.0;

                float2 counts = max(_SampleCounts.xy, 1.0.xx);
                float2 clampedBase = clamp(patchBase, 0.0.xx, counts - 1.0.xx);
                float2 clampedEnd = clamp(patchBase + patchSize - 1.0.xx, 0.0.xx, counts - 1.0.xx);
                float2 centerIndex = floor((clampedBase + clampedEnd) * 0.5);
                float2 centerUv = CenterUvFromIndex(centerIndex, counts);

                float centerDepth;
                bool hasCenter = TrySampleWorld(centerUv, eye, centerDepth, patchCenter);
                if (!hasCenter)
                    return false;

                float2 rightIndex = clamp(centerIndex + float2(1.0, 0.0), 0.0.xx, counts - 1.0.xx);
                float2 upIndex = clamp(centerIndex + float2(0.0, 1.0), 0.0.xx, counts - 1.0.xx);
                float rightDepth;
                float upDepth;
                float3 worldRight;
                float3 worldUp;
                bool hasRight = TrySampleWorld(CenterUvFromIndex(rightIndex, counts), eye, rightDepth, worldRight);
                if (!hasRight)
                    return false;
                bool hasUp = TrySampleWorld(CenterUvFromIndex(upIndex, counts), eye, upDepth, worldUp);
                if (!hasUp)
                    return false;

                float3 horizontal = worldRight - patchCenter;
                float3 vertical = worldUp - patchCenter;
                if (dot(horizontal, horizontal) <= 1e-6 || dot(vertical, vertical) <= 1e-6)
                    return false;

                patchNormal = -normalize(cross(horizontal, vertical));
                float axisAlignment = max(abs(patchNormal.x), max(abs(patchNormal.y), abs(patchNormal.z)));
                if (axisAlignment < _AxisAlignmentThreshold)
                    return false;

                float3 absNormal = abs(patchNormal);
                if (absNormal.x >= absNormal.y && absNormal.x >= absNormal.z)
                {
                    patchTangent = patchNormal.x >= 0.0 ? float3(0, 0, 1) : float3(0, 0, -1);
                    patchBitangent = float3(0, 1, 0);
                }
                else if (absNormal.y >= absNormal.x && absNormal.y >= absNormal.z)
                {
                    patchTangent = float3(1, 0, 0);
                    patchBitangent = patchNormal.y >= 0.0 ? float3(0, 0, 1) : float3(0, 0, -1);
                }
                else
                {
                    patchTangent = float3(1, 0, 0);
                    patchBitangent = float3(0, 1, 0);
                }

                float2 supportIndices[5];
                supportIndices[0] = centerIndex;
                supportIndices[1] = clampedBase;
                supportIndices[2] = float2(clampedEnd.x, clampedBase.y);
                supportIndices[3] = float2(clampedBase.x, clampedEnd.y);
                supportIndices[4] = clampedEnd;

                int supportCount = 0;
                float accumulatedConfidence = 0.0;
                float linearDepthDelta = max(abs(centerDepth - rightDepth), abs(centerDepth - upDepth));
                float stretch = max(length(horizontal), length(vertical));
                float edgeConfidence = 1.0 - smoothstep(_DepthEdgeSuppressStart, _DepthEdgeSuppressEnd, linearDepthDelta);
                float stretchConfidence = 1.0 - smoothstep(_SurfaceStretchSuppressStart, _SurfaceStretchSuppressEnd, stretch);

                [unroll]
                for (int i = 0; i < 5; i++)
                {
                    float sampleDepth;
                    float3 sampleWorld;
                    bool hasSupport = TrySampleWorld(CenterUvFromIndex(supportIndices[i], counts), eye, sampleDepth, sampleWorld);
                    if (!hasSupport)
                        continue;

                    float planeDistance = abs(dot(sampleWorld - patchCenter, patchNormal));
                    if (planeDistance > _PatchPlanarityThreshold)
                        continue;

                    supportCount++;
                    accumulatedConfidence += 1.0 - smoothstep(0.0, _PatchPlanarityThreshold, planeDistance);
                }

                if (supportCount < (int)round(_PatchCounts.z))
                    return false;

                confidence = saturate((accumulatedConfidence / max(1, supportCount)) * edgeConfidence * stretchConfidence);
                return confidence > 0.01;
            }

            Varyings vert(Attributes input)
            {
                Varyings output;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

                int eye = unity_StereoEyeIndex;
                float rawDepth = SampleRawDepth(input.uv, eye);
                output.valid = 0.0;
                output.confidence = 0.0;
                output.localCoord = input.uv1;
                output.patchLocalCoord = input.uv1;
                output.color = _GridColor;

                if (_DebugUseScreenGridPlane > 0.5)
                {
                    float2 localOffsetDebug = input.uv1 * max(0.02, _CellSizeMeters);
                    float3 worldPosDebug = TransformObjectToWorld(input.positionOS.xyz + float3(localOffsetDebug, 0.0));
                    output.positionHCS = TransformWorldToHClip(worldPosDebug);
                    output.valid = 1.0;
                    output.confidence = 1.0;
                    return output;
                }

                if (!IsValidRawDepth(rawDepth))
                {
                    output.positionHCS = float4(-2.0, -2.0, 1.0, 1.0);
                    return output;
                }

                float linearDepth = LinearizeDepth(rawDepth);
                if (linearDepth < _MinLinearDepth || linearDepth > _MaxLinearDepth)
                {
                    output.positionHCS = float4(-2.0, -2.0, 1.0, 1.0);
                    return output;
                }

                float2 counts = max(_SampleCounts.xy, 1.0.xx);
                float2 uv01 = float2(
                    (input.uv.x - (_ViewportRect.x - _ViewportRect.z * 0.5)) / max(1e-5, _ViewportRect.z),
                    (input.uv.y - (_ViewportRect.y - _ViewportRect.w * 0.5)) / max(1e-5, _ViewportRect.w));
                float2 cellIndex = floor(saturate(uv01) * counts);
                cellIndex = clamp(cellIndex, 0.0.xx, counts - 1.0.xx);

                float2 patchSize = max(_PatchCounts.xy, 1.0.xx);
                float2 patchBase = floor(cellIndex / patchSize) * patchSize;
                float2 localCellIndex = cellIndex - patchBase;

                float3 patchCenter;
                float3 patchNormal;
                float3 patchTangent;
                float3 patchBitangent;
                float patchConfidence;
                if (!TryBuildPatch(patchBase, patchSize, eye, patchCenter, patchNormal, patchTangent, patchBitangent, patchConfidence))
                {
                    output.positionHCS = float4(-2.0, -2.0, 1.0, 1.0);
                    return output;
                }

                float2 patchCellCenterOffset = (localCellIndex + 0.5.xx) - patchSize * 0.5;
                float2 localOffset = (patchCellCenterOffset + input.uv1) * max(0.02, _CellSizeMeters);
                float3 worldPos = patchCenter + patchNormal * max(0.0, _SurfaceOffset);
                worldPos += patchTangent * localOffset.x + patchBitangent * localOffset.y;

                output.positionHCS = TransformWorldToHClip(worldPos);
                output.valid = 1.0;
                output.confidence = patchConfidence;
                output.patchLocalCoord = ((localCellIndex + input.uv1 + 0.5.xx) / max(patchSize, 1.0.xx)) - 0.5.xx;
                return output;
            }

            float GridDistance(float value, float spacing)
            {
                return abs(frac(value / spacing + 0.5) - 0.5) * spacing;
            }

            half4 frag(Varyings input) : SV_Target
            {
                if (input.valid < 0.5)
                    discard;

                float confidence = saturate(input.confidence);
                float alpha = 0.0;

                if (_DisplayMode > 0.5)
                {
                    float2 patchCoord = abs(input.patchLocalCoord) / max(1e-5, _PatchFillRatio * 0.5);
                    float inside = max(patchCoord.x, patchCoord.y) <= 1.0 ? 1.0 : 0.0;
                    float confidenceMask = confidence >= _PatchConfidenceThreshold ? 1.0 : 0.0;
                    alpha = inside * confidenceMask * confidence;
                }
                else
                {
                    float radius = saturate(_RepresentativeRadiusRatio);
                    float dist = length(input.localCoord);
                    alpha = dist <= radius ? 1.0 : 0.0;
                    alpha *= confidence;
                }

                if (alpha <= 0.02)
                    discard;

                half4 color = input.color;
                color.a *= alpha;
                return color;
            }
            ENDHLSL
        }
    }
}
