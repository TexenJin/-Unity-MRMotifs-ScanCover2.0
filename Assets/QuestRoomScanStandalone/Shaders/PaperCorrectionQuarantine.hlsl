#ifndef QRS_PAPER_CORRECTION_QUARANTINE_INCLUDED
#define QRS_PAPER_CORRECTION_QUARANTINE_INCLUDED

// Display authority for the paper-only inspection route.  The CPU publishes
// persistent 5 cm correction cells; no live-depth colour class reaches here.
// A Reject hides immediately.  A recovered cell stays gated until the actual
// extracted paper is within 15 mm of the recovered target plane.
StructuredBuffer<float4> _RSPaperCorrectionKeyHash;
StructuredBuffer<float4> _RSPaperCorrectionTargetHash;
int _RSPaperCorrectionHashMask;
float _RSPaperCorrectionHideActive;

static const uint RS_PAPER_CORRECTION_HASH_PROBES = 24u;
static const float RS_PAPER_CORRECTION_CELL_METRES = 0.05;

uint RSPaperCorrectionHash(int3 cell, uint axisFamily)
{
    uint3 value = asuint(cell);
    uint hash = value.x * 73856093u;
    hash ^= value.y * 19349663u;
    hash ^= value.z * 83492791u;
    hash ^= axisFamily * 2654435761u;
    hash ^= hash >> 16;
    return hash;
}

uint RSPaperCorrectionAxisFamily(float3 normalWS)
{
    float3 n = normalize(normalWS);
    float3 a = abs(n);
    // Ignore only the sign: TSDF extraction and correspondence normals can
    // legitimately choose opposite signs for the same two-sided paper.
    if (a.x >= a.y && a.x >= a.z) return 0u;
    if (a.y >= a.z) return 1u;
    return 2u;
}

bool RSPaperCorrectionPointRejected(float3 positionWS, float3 normalWS)
{
    if (_RSPaperCorrectionHideActive < 0.5 ||
        _RSPaperCorrectionHashMask < 0)
        return false;

    int3 cell = (int3)floor(positionWS / RS_PAPER_CORRECTION_CELL_METRES);
    uint axisFamily = RSPaperCorrectionAxisFamily(normalWS);
    uint mask = (uint)_RSPaperCorrectionHashMask;
    uint start = RSPaperCorrectionHash(cell, axisFamily) & mask;
    [loop]
    for (uint probe = 0u; probe < RS_PAPER_CORRECTION_HASH_PROBES; probe++)
    {
        uint slot = (start + probe) & mask;
        float4 packed = _RSPaperCorrectionKeyHash[slot];
        uint state = (uint)round(packed.w);
        if (state == 0u) return false;
        int3 stored = (int3)round(packed.xyz);
        uint storedFamilyToken = state & 15u;
        if (all(stored == cell) && storedFamilyToken == axisFamily + 1u)
        {
            uint mode = state >> 4u;
            if (mode == 1u) return true;
            if (mode == 2u)
            {
                float3 target = _RSPaperCorrectionTargetHash[slot].xyz;
                float residual = abs(dot(positionWS - target,
                    normalize(normalWS)));
                return residual > 0.015;
            }
            return true;
        }
    }
    return false;
}

#endif
