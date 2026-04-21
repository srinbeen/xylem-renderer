#ifndef XYLEM_MESHLET_TYPES_HLSLI
#define XYLEM_MESHLET_TYPES_HLSLI

// Mirrors src/include/Meshlet.hpp

#define XYLEM_MAX_MESHLET_VERTS 64
#define XYLEM_MAX_MESHLET_PRIMS 124
#define XYLEM_AS_GROUP_SIZE     32
// MS thread group must cover max(verts, prims) so every declared
// vertex/primitive has a writing thread — otherwise undeclared
// primitives read garbage indices and the GPU hangs.
#define XYLEM_MS_GROUP_SIZE     128

// D3D12 caps DispatchMesh per-axis at 65535. When the work-item count exceeds
// this, CPU spreads across X/Y; AS reconstructs a flat work index using this.
#define XYLEM_DISPATCH_X        65535

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
