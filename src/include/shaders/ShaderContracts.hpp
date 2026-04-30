#ifndef XYLEM_SHADER_CONTRACTS_H
#define XYLEM_SHADER_CONTRACTS_H

#include <cstddef>
#include <cstdint>

#include <nvrhi/nvrhi.h>

#include "../Render.hpp"

namespace Xylem::shader::cb {

constexpr size_t AlignConstantBufferSize(size_t byteSize)
{
    return (byteSize + (nvrhi::c_ConstantBufferOffsetSizeAlignment - 1))
        & ~(nvrhi::c_ConstantBufferOffsetSizeAlignment - 1);
}

using FrameConstants = Render::ConstantBufferEntry;
using CullFrameConstants = Render::CullConstantBufferEntry;

struct MeshASCullConstants {
    dm::float3 cameraPos;
    uint32_t   hizEnabled;
    dm::float2 hizDimensions;
    float      maxHiZMip;
    uint32_t   asConeCullEnabled;
};

// Matches SDSMBuildCascades.hlsl::SDSMInput.
struct SDSMCascadeBuildInput {
    dm::float4x4 worldToLight;
    dm::float4x4 viewToWorldToLight;
    dm::float4   sceneBboxMinLS;
    dm::float4   sceneBboxMaxLS;
    float        tanHalfFovX;
    float        tanHalfFovY;
    float        projA;
    float        projB;
    float        regionEnvelopeNear;
    float        regionEnvelopeFar;
    float        cameraNearPlane;
    uint32_t     shadowRes;
    uint32_t     maxHiZMip;
    float        pssmLambda;
    float        _pad[2];
};

struct SDSMCascadeBuildOutput {
    dm::float4x4 lightViewProj[Render::c_NumCascades];
    dm::float4   cascadeSplits;
    dm::float4   shadowCasterMinLS[Render::c_NumCascades];
    dm::float4   shadowCasterMaxLS[Render::c_NumCascades];

    // Diagnostic fields (debug-only, surfaced to UI via readback ring).
    // .x = nearDepthVal (raw Hi-Z top-mip nearest), .y = farDepthVal (raw farthest)
    // .z = tightNear (post safety-expand + clamp), .w = tightFar
    dm::float4   debugDepthExtents;
};

inline constexpr size_t kFrameSize = Render::c_ConstantBufferSize;
inline constexpr size_t kCullFrameSize = Render::c_CullConstantBufferSize;
inline constexpr size_t kMeshASCullSize = AlignConstantBufferSize(sizeof(MeshASCullConstants));
inline constexpr size_t kSDSMCascadeBuildInputSize = AlignConstantBufferSize(sizeof(SDSMCascadeBuildInput));

} // namespace Xylem::shader::cb

#include "ShaderRegisterMap.hpp"

namespace Xylem::shader::reg {
namespace Compute::HiZ {
inline constexpr uint32_t kPushCDwordCount = 2;
} // namespace Compute::HiZ

namespace Mesh::HiZ {
inline constexpr uint32_t kPushCDwordCount = 2;
} // namespace Mesh::HiZ

namespace Mesh::Draw {
inline constexpr uint32_t kPushCBytes = sizeof(uint32_t);
} // namespace Mesh::Draw

namespace Mesh {
inline constexpr uint32_t kPushC_RootParamIdx = 0;
} // namespace Mesh
} // namespace Xylem::shader::reg

#endif // XYLEM_SHADER_CONTRACTS_H
