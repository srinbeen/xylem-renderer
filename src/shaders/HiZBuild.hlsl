#include "../include/macros.h"
#include "ShaderRegisterMap.hlsli"

struct HiZPushConstants {
    uint2 destDimensions;
};
ConstantBuffer<HiZPushConstants> pc : register(XY_REG_B_COMPUTE_HIZ_PUSH_C_DEST_DIMENSIONS);

// Hi-Z texture is two-channel:
//   .r = farthest-depth reduction (drives Hi-Z occlusion culling)
//   .g = nearest-depth reduction  (drives SDSM tight near/far)
//
// Reverse-Z: near=1, far=0
//   farthest = smallest value -> min reduction -> .r
//   nearest  = largest value  -> max reduction -> .g
// Forward-Z swaps both.
Texture2D<float2>    t_Source : register(XY_REG_T_COMPUTE_HIZ_SRV_SOURCE);
RWTexture2D<float2>  u_Dest   : register(XY_REG_U_COMPUTE_HIZ_UAV_DEST);

float2 ReduceDepth(float2 a, float2 b)
{
#if XYLEM_USE_REVERSE_Z
    return float2(min(a.r, b.r), max(a.g, b.g));
#else
    return float2(max(a.r, b.r), min(a.g, b.g));
#endif
}

float2 LoadSourceClamped(uint2 coord, uint2 maxCoord)
{
    return t_Source[min(coord, maxCoord)];
}

// Copy raw depth (D32 read as R32_FLOAT) into both channels of mip 0.
// The source view is a single-channel depth texture; we read .r and seed
// both min and max channels from the same value so downsamples produce
// correct extrema.
[numthreads(8, 8, 1)]
void HiZCopy(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= pc.destDimensions.x || dtid.y >= pc.destDimensions.y) return;
    float d = t_Source[dtid.xy].r;
    u_Dest[dtid.xy] = float2(d, d);
}

// Downsample: .r keeps the farthest depth (conservative Hi-Z occluder),
// .g keeps the nearest depth (SDSM extrema).
[numthreads(8, 8, 1)]
void HiZDownsample(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= pc.destDimensions.x || dtid.y >= pc.destDimensions.y) return;

    uint srcWidth, srcHeight;
    t_Source.GetDimensions(srcWidth, srcHeight);
    uint2 srcMax = uint2(srcWidth - 1u, srcHeight - 1u);

    uint2 srcCoord = dtid.xy * 2;
    float2 d0 = LoadSourceClamped(srcCoord + uint2(0, 0), srcMax);
    float2 d1 = LoadSourceClamped(srcCoord + uint2(1, 0), srcMax);
    float2 d2 = LoadSourceClamped(srcCoord + uint2(0, 1), srcMax);
    float2 d3 = LoadSourceClamped(srcCoord + uint2(1, 1), srcMax);

    float2 reduced = ReduceDepth(ReduceDepth(d0, d1), ReduceDepth(d2, d3));

    bool hasExtraColumn = (dtid.x == pc.destDimensions.x - 1u) && (srcCoord.x + 2u < srcWidth);
    bool hasExtraRow = (dtid.y == pc.destDimensions.y - 1u) && (srcCoord.y + 2u < srcHeight);

    if (hasExtraColumn)
    {
        uint x = srcCoord.x + 2u;
        reduced = ReduceDepth(reduced, LoadSourceClamped(uint2(x, srcCoord.y + 0u), srcMax));
        reduced = ReduceDepth(reduced, LoadSourceClamped(uint2(x, srcCoord.y + 1u), srcMax));
    }

    if (hasExtraRow)
    {
        uint y = srcCoord.y + 2u;
        reduced = ReduceDepth(reduced, LoadSourceClamped(uint2(srcCoord.x + 0u, y), srcMax));
        reduced = ReduceDepth(reduced, LoadSourceClamped(uint2(srcCoord.x + 1u, y), srcMax));
    }

    if (hasExtraColumn && hasExtraRow)
        reduced = ReduceDepth(reduced, LoadSourceClamped(srcCoord + uint2(2u, 2u), srcMax));

    u_Dest[dtid.xy] = reduced;
}
