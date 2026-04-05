#ifndef XYLEM_RENDER_H
#define XYLEM_RENDER_H

#include <nvrhi/nvrhi.h>
#include <donut/core/math/math.h>

namespace Xylem::Render {

struct InstanceBufferEntry {
    dm::float4x4 model;
    dm::float3x3 normal;
    uint32_t     treeId;

    InstanceBufferEntry(const dm::float4x4& m, const dm::float3x3& n, uint32_t t)
        : model{m}, normal{n}, treeId{t} {}
    InstanceBufferEntry(const dm::affine3& m, const dm::float3x3& n, uint32_t t)
        : InstanceBufferEntry(
            dm::affineToHomogeneous(m), 
            n, 
            t
        ) {}
    InstanceBufferEntry()
        : InstanceBufferEntry(
            dm::float4x4::identity(), 
            dm::float3x3::identity(), 
            static_cast<uint32_t>(-1)
        ) {}
    
    ~InstanceBufferEntry() = default;
};

struct ConstantBufferEntry {
    dm::float4x4 view;
    dm::float4x4 projection;
    dm::float4x4 lightViewProj;
    dm::float3   sunLightDir;
    float        _pad0;
    dm::float3x4 _pad1;
};

static constexpr size_t c_ConstantBufferSize = (sizeof(ConstantBufferEntry) + (nvrhi::c_ConstantBufferOffsetSizeAlignment - 1)) & ~(nvrhi::c_ConstantBufferOffsetSizeAlignment - 1);

struct DrawCmd {
    nvrhi::BufferHandle  vertexBuffer;
    nvrhi::BufferHandle  indexBuffer;
    nvrhi::DrawArguments drawArgs;
    uint32_t             textureSetIdx;
};

struct ShadowDrawCmd {
    nvrhi::BufferHandle  vertexBuffer;
    nvrhi::BufferHandle  indexBuffer;
    nvrhi::DrawArguments drawArgs;
};

struct InstanceReference {
    uint32_t regionIdx;
    uint32_t instanceIdx;
    uint32_t treeId;
    uint32_t lodID;
};

// GPU cull instance data — parallel to InstanceBufferEntry in gapped buffer layout.
struct CullInstanceData {
    dm::box3 bbox;
    uint32_t   baseSlot;  // treeId * numLODs
    uint32_t   active;    // 1 = live, 0 = dead capacity slot
};

// Extended constant buffer for compute cull pass.
struct CullConstantBufferEntry {
    // P0 fields (read by VS/PS)
    dm::float4x4 view;
    dm::float4x4 projection;
    dm::float4x4 lightViewProj;
    dm::float3   sunLightDir;
    float        _pad0;
    dm::float3x4 _pad1;

    // Cull fields (read by CullCS)
    dm::frustum  viewFrustum;              
    dm::frustum  lightFrustum;    
    dm::float3   cameraPos;
    uint32_t     totalCapacity;
    dm::float4   lodDistances[3];
    uint32_t     numLods;
    uint32_t     numSlots;
    uint32_t     visBufferSize;
    uint32_t     numAssets;
};

static constexpr size_t c_CullConstantBufferSize =
    (sizeof(CullConstantBufferEntry) + (nvrhi::c_ConstantBufferOffsetSizeAlignment - 1))
    & ~(nvrhi::c_ConstantBufferOffsetSizeAlignment - 1);

} // namespace Xylem::Render

#endif // XYLEM_RENDER_H