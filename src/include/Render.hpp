#ifndef XYLEM_RENDER_H
#define XYLEM_RENDER_H

#include <nvrhi/nvrhi.h>
#include <donut/core/math/math.h>
#include "macros.h"

namespace Xylem::Render {

static constexpr uint32_t c_NumCascades = XYLEM_NUM_CASCADES;

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
    dm::float4x4 viewProj;
    dm::float4x4 viewMatrix;
    dm::float4x4 lightViewProj[c_NumCascades];
    dm::float3   sunLightDir;
    float        _pad0;
    dm::float4   cascadeSplits;
    dm::float4   _pad1[6];
};

static constexpr size_t c_ConstantBufferSize = (sizeof(ConstantBufferEntry) + (nvrhi::c_ConstantBufferOffsetSizeAlignment - 1)) & ~(nvrhi::c_ConstantBufferOffsetSizeAlignment - 1);

struct VertexBufferSet {
    nvrhi::BufferHandle position;
    nvrhi::BufferHandle normal;
    nvrhi::BufferHandle tangent;
    nvrhi::BufferHandle bitangent;
    nvrhi::BufferHandle uv;
};

struct DrawCmd {
    VertexBufferSet      vertexBuffers;
    nvrhi::BufferHandle  indexBuffer;
    nvrhi::DrawArguments drawArgs;
    uint32_t             textureSetIdx;
    uint32_t             leafSlot;
};

struct ShadowDrawCmd {
    nvrhi::BufferHandle  positionBuffer;
    nvrhi::BufferHandle  indexBuffer;
    nvrhi::DrawArguments drawArgs;
    uint32_t             leafSlot;
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
    uint32_t   regionId;  // index into regionVisible buffer
    uint32_t   active;    // 1 = live, 0 = dead capacity slot
};

struct CullRegionData {
    dm::box3 bbox;
};

// Extended constant buffer for compute cull pass.
struct CullConstantBufferEntry {
    // Prefix — layout matches ConstantBufferEntry (read by shared VS/PS)
    dm::float4x4 viewProj;
    dm::float4x4 viewMatrix;
    dm::float4x4 lightViewProj[c_NumCascades];
    dm::float3   sunLightDir;
    float        _pad0;
    dm::float4   cascadeSplits;

    // Cull fields (read by CullCS)
    dm::frustum  viewFrustum;
    dm::float4x4 worldToLight;
    dm::float4   shadowCasterMinLS[c_NumCascades];  // xyz used; w is padding
    dm::float4   shadowCasterMaxLS[c_NumCascades];

    dm::float3   cameraPos;
    uint32_t     numRegions;
    uint32_t     totalCapacity;
    uint32_t     numLods;
    float        _pad1a;
    float        _pad1b;
    dm::float4   lodDistances;

    // Hi-Z fields (read by CullCS for occlusion test)
    dm::float2   hizDimensions;    // mip 0 width, height
    float        maxHiZMip;        // numMips - 1
    uint32_t     hizEnabled;       // 0 = skip Hi-Z test, 1 = enabled
    float        impostorAlphaClip;
    uint32_t     showShadowImpostors  = 1;
    float        shadowImpostorBias   = 0.05f;
    float        _pad2                = 0.0f;
};

static constexpr size_t c_CullConstantBufferSize =
    (sizeof(CullConstantBufferEntry) + (nvrhi::c_ConstantBufferOffsetSizeAlignment - 1))
    & ~(nvrhi::c_ConstantBufferOffsetSizeAlignment - 1);

} // namespace Xylem::Render

#endif // XYLEM_RENDER_H
