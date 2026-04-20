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

// GPU-visible meshlet descriptor. Mirror of HLSL MeshletDesc in meshlet_types.hlsli.
struct MeshletDesc {
    uint32_t   vertexOffset;   // index into g_MeshletVertIdx
    uint32_t   triangleOffset; // byte offset into g_MeshletPrimIdx (packed u8×3)
    uint32_t   vertexCount;    // <= k_MaxMeshletVerts
    uint32_t   triangleCount;  // <= k_MaxMeshletPrims
    dm::float4 bounds;         // xyz center (local), w radius
    dm::float4 coneApex;       // xyz cone apex, w unused
    dm::float4 coneAxisCutoff; // xyz cone axis, w cone cutoff (from meshopt)
};

// One entry per (asset, lod). Points into the global meshlet table.
struct AssetLodRange {
    uint32_t meshletOffset;    // first meshlet in MeshletDesc mega-buffer
    uint32_t meshletCount;
    uint32_t vertexAttribBase; // first vertex in per-attribute SoA mega-buffers (g_Positions etc)
    uint32_t meshletVertBase;  // first uint in g_MeshletVertIdx for this asset-lod
    uint32_t meshletPrimBase;  // first byte in g_MeshletPrimIdx for this asset-lod
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
};

// Packed work-item written by cull CS, consumed by AS.
// One entry per (instance, chunk-of-k_ASGroupSize-meshlets) pair, so asset-LODs
// with > k_ASGroupSize meshlets span multiple work items sharing instanceIdx.
// Keep 16 bytes for alignment comfort in structured buffers.
struct VisibleInstance {
    uint32_t instanceIdx;       // into persistentInstBuffer
    uint32_t assetLod;          // index into g_AssetLodRanges
    uint32_t meshletChunkBase;  // first meshlet-local index this AS group handles
    uint32_t _pad1;
};

// CPU-side bundle of mega-buffer data produced by rebuildMeshlets().
struct MeshletBuild {
    std::vector<dm::float3>   positions;
    std::vector<dm::float3>   normals;
    std::vector<dm::float3>   tangents;
    std::vector<dm::float3>   bitangents;
    std::vector<dm::float2>   uvs;
    std::vector<uint32_t>     meshletVertIdx;   // per-meshlet local vertex indices
    std::vector<uint8_t>      meshletPrimIdx;   // per-meshlet packed tri indices (u8)
    std::vector<MeshletDesc>  meshlets;
    std::vector<AssetLodRange> assetLodRanges;  // size = numAssets * numLods
};

} // namespace Xylem::Render

#endif // XYLEM_MESHLET_H
