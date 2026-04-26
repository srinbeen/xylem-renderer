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

    uint2 srcCoord = dtid.xy * 2;
    float2 d0 = t_Source[srcCoord + uint2(0, 0)];
    float2 d1 = t_Source[srcCoord + uint2(1, 0)];
    float2 d2 = t_Source[srcCoord + uint2(0, 1)];
    float2 d3 = t_Source[srcCoord + uint2(1, 1)];

#if XYLEM_USE_REVERSE_Z
    float farthest = min(min(d0.r, d1.r), min(d2.r, d3.r));   // farthest in reverse-Z
    float nearest  = max(max(d0.g, d1.g), max(d2.g, d3.g));   // nearest in reverse-Z
#else
    float farthest = max(max(d0.r, d1.r), max(d2.r, d3.r));
    float nearest  = min(min(d0.g, d1.g), min(d2.g, d3.g));
#endif

    u_Dest[dtid.xy] = float2(farthest, nearest);
}
