#ifndef XYLEM_MESHLET_H
#define XYLEM_MESHLET_H

#include <cstdint>
#include <vector>
#include <nvrhi/nvrhi.h>
#include <donut/core/math/math.h>
#include "MeshletConstants.h"

namespace Xylem::Render {

// Values live in MeshletConstants.h (shared with HLSL) — re-exposed here
// as typed C++ constants so existing call sites can keep using Render::k_*.
static constexpr uint32_t k_MaxMeshletVerts    = XYLEM_MAX_MESHLET_VERTS;
static constexpr uint32_t k_MaxMeshletPrims    = XYLEM_MAX_MESHLET_PRIMS;
static constexpr uint32_t k_ASGroupSize        = XYLEM_AS_GROUP_SIZE;   // meshlets per AS threadgroup
static constexpr uint32_t k_MSGroupSize        = XYLEM_MS_GROUP_SIZE;
static constexpr uint32_t k_DispatchAxisMax    = XYLEM_DISPATCH_X;
static constexpr uint32_t k_LeafsPerMeshlet    = XYLEM_LEAFS_PER_MESHLET;
static constexpr uint32_t k_LeafVertsPerLeaf   = XYLEM_LEAF_VERTS_PER_LEAF;

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
