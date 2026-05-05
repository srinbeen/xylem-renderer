#include "../include/macros.h"
#include "ShaderRegisterMap.hlsli"

// RGBA32_FLOAT
//   .r = raw farthest-depth reduction          -> Hi-Z occlusion, sky included
//   .g = nearest-depth reduction               -> SDSM near
//   .b = sky-excluded farthest-depth reduction -> SDSM far
//   .a = unused
//
// HiZCopy seeds .b with the near-plane sentinel for sky pixels so scene depth
// wins over sky during farthest-depth reduction, while all-sky frames preserve
// the sentinel for SDSM fallback detection.

struct HiZPushConstants {
    uint2 destDimensions;
};
ConstantBuffer<HiZPushConstants> pc : register(XY_REG_B_COMPUTE_HIZ_PUSH_C_DEST_DIMENSIONS);

Texture2D<float4>    t_Source : register(XY_REG_T_COMPUTE_HIZ_SRV_SOURCE);
RWTexture2D<float4>  u_Dest   : register(XY_REG_U_COMPUTE_HIZ_UAV_DEST);

float FarthestDepth(float a, float b)
{
#if XYLEM_USE_REVERSE_Z
    return min(a, b);
#else
    return max(a, b);
#endif
}

float NearestDepth(float a, float b)
{
#if XYLEM_USE_REVERSE_Z
    return max(a, b);
#else
    return min(a, b);
#endif
}

float4 ReduceHiZ(float4 a, float4 b)
{
    return float4(
        FarthestDepth(a.r, b.r),
        NearestDepth(a.g, b.g),
        FarthestDepth(a.b, b.b),
        1.0);
}

float4 LoadSourceClamped(uint2 coord, uint2 maxCoord)
{
    return t_Source[min(coord, maxCoord)];
}

[numthreads(8, 8, 1)]
void HiZDownsample(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= pc.destDimensions.x || dtid.y >= pc.destDimensions.y) return;

    uint srcWidth, srcHeight;
    t_Source.GetDimensions(srcWidth, srcHeight);
    uint2 srcMax = uint2(srcWidth - 1u, srcHeight - 1u);

    uint2 srcCoord = dtid.xy * 2u;
    float4 t00 = LoadSourceClamped(srcCoord + uint2(0u, 0u), srcMax);
    float4 t10 = LoadSourceClamped(srcCoord + uint2(1u, 0u), srcMax);
    float4 t01 = LoadSourceClamped(srcCoord + uint2(0u, 1u), srcMax);
    float4 t11 = LoadSourceClamped(srcCoord + uint2(1u, 1u), srcMax);

    float4 reduced = ReduceHiZ(ReduceHiZ(t00, t10), ReduceHiZ(t01, t11));

    bool hasExtraColumn = (dtid.x == pc.destDimensions.x - 1u) && (srcCoord.x + 2u < srcWidth);
    bool hasExtraRow = (dtid.y == pc.destDimensions.y - 1u) && (srcCoord.y + 2u < srcHeight);

    if (hasExtraColumn)
    {
        uint x = srcCoord.x + 2u;
        reduced = ReduceHiZ(reduced, LoadSourceClamped(uint2(x, srcCoord.y + 0u), srcMax));
        reduced = ReduceHiZ(reduced, LoadSourceClamped(uint2(x, srcCoord.y + 1u), srcMax));
    }

    if (hasExtraRow)
    {
        uint y = srcCoord.y + 2u;
        reduced = ReduceHiZ(reduced, LoadSourceClamped(uint2(srcCoord.x + 0u, y), srcMax));
        reduced = ReduceHiZ(reduced, LoadSourceClamped(uint2(srcCoord.x + 1u, y), srcMax));
    }

    if (hasExtraColumn && hasExtraRow)
        reduced = ReduceHiZ(reduced, LoadSourceClamped(srcCoord + uint2(2u, 2u), srcMax));

    u_Dest[dtid.xy] = reduced;
}
