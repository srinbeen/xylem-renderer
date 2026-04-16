#include "../include/macros.h"

struct HiZPushConstants {
    uint2 destDimensions;
};
ConstantBuffer<HiZPushConstants> pc : register(b0);

Texture2D<float>    t_Source : register(t0);
RWTexture2D<float>  u_Dest   : register(u0);

// Copy depth texture (D32 read as R32_FLOAT) to Hi-Z mip 0.
[numthreads(8, 8, 1)]
void HiZCopy(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= pc.destDimensions.x || dtid.y >= pc.destDimensions.y) return;
    u_Dest[dtid.xy] = t_Source[dtid.xy];
}

// Downsample source mip to dest mip using conservative reduction.
// For occlusion culling, each Hi-Z texel must store the FARTHEST depth in
// its footprint.  If an occludee is behind even the farthest occluder depth,
// it is guaranteed fully hidden — no false positives.
//   Reverse-Z (near=1, far=0): farthest = smallest value  → min reduction
//   Forward-Z (near=0, far=1): farthest = largest  value  → max reduction
[numthreads(8, 8, 1)]
void HiZDownsample(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= pc.destDimensions.x || dtid.y >= pc.destDimensions.y) return;

    uint2 srcCoord = dtid.xy * 2;
    float d0 = t_Source[srcCoord + uint2(0, 0)];
    float d1 = t_Source[srcCoord + uint2(1, 0)];
    float d2 = t_Source[srcCoord + uint2(0, 1)];
    float d3 = t_Source[srcCoord + uint2(1, 1)];

#if XYLEM_USE_REVERSE_Z
    u_Dest[dtid.xy] = min(min(d0, d1), min(d2, d3));
#else
    u_Dest[dtid.xy] = max(max(d0, d1), max(d2, d3));
#endif
}
