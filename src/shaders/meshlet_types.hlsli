#ifndef XYLEM_MESHLET_TYPES_HLSLI
#define XYLEM_MESHLET_TYPES_HLSLI

// Mirrors src/include/Meshlet.hpp

#include "../include/MeshletConstants.h"

struct MeshletDesc {
    uint   vertexOffset;
    uint   triangleOffset;
    uint   vertexCount;
    uint   triangleCount;
    float4 bounds;          // xyz center, w radius
    float4 coneApex;
    float4 coneAxisCutoff;  // xyz axis, w cutoff
};

struct AssetLodRange {
    uint meshletOffset;
    uint meshletCount;
    uint vertexAttribBase;
    uint meshletVertBase;
    uint meshletPrimBase;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// AS -> MS payload. Keep small (<= 16KB total).
struct ASPayload {
    uint instanceIdx;
    uint assetLod;
    uint meshletIndices[XYLEM_AS_GROUP_SIZE];  // local meshlet indices that survive cull
};

#endif // XYLEM_MESHLET_TYPES_HLSLI
