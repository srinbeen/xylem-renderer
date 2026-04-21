#ifndef XYLEM_MESHLET_H
#define XYLEM_MESHLET_H

#include <cstdint>
#include <vector>
#include <nvrhi/nvrhi.h>
#include <donut/core/math/math.h>

namespace Xylem::Render {

static constexpr uint32_t k_MaxMeshletVerts = 64;
static constexpr uint32_t k_MaxMeshletPrims = 124;
static constexpr uint32_t k_ASGroupSize     = 32;  // meshlets per AS threadgroup

// One per mesh -- N assets * M LODs = N*M MeshOffsets.
// Layout mirrors HLSL AssetLodRange (meshlet_types.hlsli) — 32 bytes, 8 uints.
struct MeshOffsets {
    uint32_t meshletOffset;
    uint32_t meshletCount;
    uint32_t vertexAttribOffset;
    uint32_t meshletVertOffset;
    uint32_t meshletTriOffset;
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
};

// Layout mirrors HLSL MeshletDesc (meshlet_types.hlsli): (vertOff, triOff, vertCount, triCount, ...)
struct MeshletDesc {
    uint32_t   vertOffset;
    uint32_t   triOffset;
    uint32_t   vertCount;
    uint32_t   triCount;
    dm::float4 bounds;          // xyz center (local), w radius
    dm::float4 coneApex;        // xyz cone apex, w unused
    dm::float4 coneAxisCutoff;  // xyz cone axis, w cone cutoff (from meshopt)
};


struct VertexAttributeMegabuffers {
    std::vector<dm::float3>   positions;
    std::vector<dm::float3>   normals;
    std::vector<dm::float3>   tangents;
    std::vector<dm::float3>   bitangents;
    std::vector<dm::float2>   uvs;
};

struct MeshletMegabuffers {
    std::vector<MeshOffsets>    meshOffsets;
    
    std::vector<MeshletDesc>    meshletDescs;
    VertexAttributeMegabuffers  vertexAttributes;
    std::vector<uint32_t>       localVertIndices;
    std::vector<uint8_t>        localTriIndices;
    
};

} // namespace Xylem::Render

#endif // XYLEM_MESHLET_H
