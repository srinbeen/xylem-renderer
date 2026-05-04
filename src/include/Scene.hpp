#ifndef XYLEM_SCENE_H
#define XYLEM_SCENE_H

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <donut/core/math/math.h>

#include <nvrhi/nvrhi.h>

#include "Procgen.hpp"
#include "Render.hpp"
#include "Terrain.hpp"

namespace Xylem::Scene {

struct LSystemInstance {
    std::string     name;
    ProcGen::lgen_t gen;
};

// CPU-only LOD data — no GPU handles. Used by SceneRegistry.
struct TreeLODDef {
    std::vector<dm::float3> positions;
    std::vector<dm::float3> normals;
    std::vector<dm::float3> tangents;
    std::vector<dm::float3> bitangents;
    std::vector<dm::float2> uvs;
    std::vector<uint32_t>   indices;
    dm::box3                bbox;
    uint32_t                radialSegments = 0;
};

// Per-leaf placement in asset-local space. Promoted from a chunk of the trunk vertex
// stream (8 verts / 4 prims per leaf) to a single instance entry that drives a shared
// canonical cross-billboard mesh.
struct LeafInstance {
    dm::float4 centerHalfSize; // xyz = local center, w = half-size
    dm::float4 spine;          // xyz = unit height axis
    dm::float4 right;          // xyz = unit width axis; q2 = cross(spine, right)
    dm::float4 color;          // rgb = leaf albedo, a unused
};
static_assert(sizeof(LeafInstance) == 64, "LeafInstance must match HLSL LeafInstanceData");

// Bounding sphere for one AS-meshlet's worth of leaves (8 leaves). LOD-invariant: the
// contributing terminal set is fixed at bake time; lower LODs only dispatch fewer meshlets.
struct LeafSlot {
    uint32_t leafOffset    = 0;
    uint32_t leafCount     = 0;
    uint32_t meshletOffset = 0;
    uint32_t meshletCount  = 0;
};
static_assert(sizeof(LeafSlot) == 16, "LeafSlot must match HLSL LeafSlotData");

struct LeafMeshlet {
    dm::uint4  meta;   // x = local leaf offset, y = leaf count
    dm::float4 bounds; // xyz = local center, w = radius
};
static_assert(sizeof(LeafMeshlet) == 32, "LeafMeshlet must match HLSL LeafMeshletData");

// Per-asset leaf data. Storage is LOD-prefix ordered: countByLod[lod] is the prefix
// length to dispatch at LOD `lod`. countByLod is monotone non-decreasing as lod
// decreases (LOD0 is the full set; LOD3 is typically 0).
struct LeafAssetDef {
    std::vector<LeafInstance> instances;
    std::vector<uint32_t>     countByLod;
    std::vector<LeafSlot>     lodSlots;
    dm::box3                  localBbox = dm::box3::empty();
    std::vector<LeafMeshlet>  meshlets;
};

// GPU-side LOD data — owned by render passes. SoA: one buffer per attribute stream.
struct TreeLODData {
    Render::VertexBufferSet vbs;
    nvrhi::BufferHandle     indexBuffer;
    uint32_t                indexCount;
    uint32_t                radialSegments;
    dm::box3                bbox;
};

struct TreeAsset {
    std::string                    name;
    LSystemInstance                lsystemInstance;
    ProcGen::TreeGenerator::Params generatorParams;
    ProcGen::lstring_t             lsystemString;
    std::vector<TreeLODData>       lods;
    std::string                    barkTexture;
    uint32_t                       textureSetIdx = 0;
};

struct TreeRegion {
    std::string                              name;
    float                                    density;
    uint32_t                                 instanceCount;
    dm::box2                                 bounds;
    std::vector<uint32_t>                    assetIndices;
    std::vector<Render::InstanceBufferEntry> instanceBuffer;
    std::vector<dm::box3>                    instanceBbox;
    dm::box3                                 cullBox;

    TreeRegion(const std::string& n, float d, const dm::box2& b)
        : name{n}, density{d}, bounds{b}, cullBox{dm::box3::empty()}
    {
        float area = (b.m_maxs.x - b.m_mins.x) * (b.m_maxs.y - b.m_mins.y);
        instanceCount = std::max(1u, static_cast<uint32_t>(std::round(density * area)));
        instanceBuffer.reserve(instanceCount);
        instanceBbox.reserve(instanceCount);
    }
};

} // namespace Xylem::Scene

#endif // XYLEM_SCENE_H

