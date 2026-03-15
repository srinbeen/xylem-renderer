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
    dm::float4x4 padding[2];
};

static constexpr size_t c_ConstantBufferSize =
    sizeof(ConstantBufferEntry) +
    (sizeof(ConstantBufferEntry) % nvrhi::c_ConstantBufferOffsetSizeAlignment);

struct DrawCmd {
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

} // namespace Xylem::Render

#endif // XYLEM_RENDER_H