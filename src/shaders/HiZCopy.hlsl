#include "../include/macros.h"
#include "ShaderRegisterMap.hlsli"

// HiZCopy: copies depth texture to RGBA texture for mipmapping
// RGBA32_FLOAT
//   .r = raw farthest-depth reduction          -> Hi-Z mipmap
//   .g = nearest-depth reduction               -> SDSM near
//   .b = sky-excluded farthest-depth reduction -> SDSM far
//   .a = unused

// .b wrote near plane value where sky was in HiZCopy 
// so any scene depth overwrites sky value
// .r has sky value take over mipmap since its furthest depth

struct HiZPushConstants {
    uint2 destDimensions;
};
ConstantBuffer<HiZPushConstants> pc : register(XY_REG_B_COMPUTE_HIZ_PUSH_C_DEST_DIMENSIONS);

Texture2D<float>     t_Source : register(XY_REG_T_COMPUTE_HIZ_SRV_SOURCE);
RWTexture2D<float4>  u_Dest   : register(XY_REG_U_COMPUTE_HIZ_UAV_DEST);

#if XYLEM_USE_REVERSE_Z
    static const float kSkyDepth   = 0.0;
#else
    static const float kSkyDepth   = 1.0;
#endif

[numthreads(8, 8, 1)]
void HiZCopy(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= pc.destDimensions.x || dtid.y >= pc.destDimensions.y) return;
    float d = t_Source[dtid.xy];
    // .b: write near plane value when sky pixel to track scene SDSM far value
    float dSkyExcl = (d == kSkyDepth) ? (1.0 - kSkyDepth) : d;
    u_Dest[dtid.xy] = float4(d, d, dSkyExcl, 1.0);
}
