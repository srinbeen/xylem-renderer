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
};

inline constexpr size_t kFrameSize = Render::c_ConstantBufferSize;
inline constexpr size_t kCullFrameSize = Render::c_CullConstantBufferSize;
inline constexpr size_t kMeshASCullSize = AlignConstantBufferSize(sizeof(MeshASCullConstants));
inline constexpr size_t kSDSMCascadeBuildInputSize = AlignConstantBufferSize(sizeof(SDSMCascadeBuildInput));

} // namespace Xylem::shader::cb

namespace Xylem::shader::reg {

namespace Traditional {
namespace Tree {
inline constexpr uint32_t kCB_Frame = 0;
inline constexpr uint32_t kTex_Diffuse = 0;
inline constexpr uint32_t kTex_NormalMap = 1;
inline constexpr uint32_t kTex_ShadowMap = 2;
inline constexpr uint32_t kSampler_Main = 0;
inline constexpr uint32_t kSampler_Shadow = 1;
} // namespace Tree

namespace Shadow {
inline constexpr uint32_t kCB_Frame = 0;
inline constexpr uint32_t kPushC_CascadeIndex = 1;
} // namespace Shadow

namespace Terrain {
inline constexpr uint32_t kCB_Frame = 0;
inline constexpr uint32_t kTex_ShadowMap = 0;
inline constexpr uint32_t kSampler_Shadow = 0;
} // namespace Terrain

namespace Sky {
inline constexpr uint32_t kCB_Sky = 0;
} // namespace Sky
} // namespace Traditional

namespace Compute {
namespace Cull {
inline constexpr uint32_t kCB_Frame = 0;
inline constexpr uint32_t kSRV_RegionData = 0;
inline constexpr uint32_t kSRV_InstanceData = 1;
inline constexpr uint32_t kSRV_MainSlotOffsets = 2;
inline constexpr uint32_t kSRV_ShadowSlotOffsets = 3;
inline constexpr uint32_t kSRV_HiZ = 4;
inline constexpr uint32_t kUAV_MainRegionVis = 0;
inline constexpr uint32_t kUAV_MainSlotCount = 1;
inline constexpr uint32_t kUAV_MainVis = 2;
inline constexpr uint32_t kUAV_ShadowSlotCount = 3;
inline constexpr uint32_t kUAV_ShadowVis = 4;
inline constexpr uint32_t kUAV_MainIndirectArgs = 5;
inline constexpr uint32_t kUAV_ShadowIndirectArgs = 6;
inline constexpr uint32_t kUAV_ShadowUniqueCounter = 7;
inline constexpr uint32_t kSampler_HiZ = 0;
} // namespace Cull

namespace Scene {
inline constexpr uint32_t kCB_Frame = 0;
inline constexpr uint32_t kPushC_Slot = 1;
inline constexpr uint32_t kSRV_Vis = 0;
inline constexpr uint32_t kSRV_Instances = 1;
inline constexpr uint32_t kSRV_SlotOffsets = 2;
inline constexpr uint32_t kTex_Diffuse = 3;
inline constexpr uint32_t kTex_NormalMap = 4;
inline constexpr uint32_t kTex_ShadowMap = 5;
inline constexpr uint32_t kSampler_Main = 0;
inline constexpr uint32_t kSampler_Shadow = 1;
} // namespace Scene

namespace Shadow {
inline constexpr uint32_t kCB_Frame = 0;
inline constexpr uint32_t kPushC_AssetCascade = 1;
inline constexpr uint32_t kSRV_Vis = 0;
inline constexpr uint32_t kSRV_Instances = 1;
inline constexpr uint32_t kSRV_SlotOffsets = 2;
} // namespace Shadow

namespace DepthPrepass {
inline constexpr uint32_t kCB_Frame = 0;
inline constexpr uint32_t kPushC_Slot = 1;
inline constexpr uint32_t kSRV_Vis = 0;
inline constexpr uint32_t kSRV_Instances = 1;
inline constexpr uint32_t kSRV_SlotOffsets = 2;
} // namespace DepthPrepass

namespace HiZ {
inline constexpr uint32_t kPushC_DestDimensions = 0;
inline constexpr uint32_t kSRV_Source = 0;
inline constexpr uint32_t kUAV_Dest = 0;
inline constexpr uint32_t kPushCDwordCount = 2;
} // namespace HiZ

namespace SDSM {
inline constexpr uint32_t kCB_Input = 0;
inline constexpr uint32_t kSRV_HiZ = 0;
inline constexpr uint32_t kUAV_CascadeOut = 0;
} // namespace SDSM

namespace Terrain {
inline constexpr uint32_t kCB_Frame = 0;
inline constexpr uint32_t kSampler_Shadow = 0;
inline constexpr uint32_t kTex_ShadowMap = 0;
} // namespace Terrain

namespace Sky {
inline constexpr uint32_t kCB_Sky = 0;
} // namespace Sky
} // namespace Compute

namespace Mesh {
namespace Draw {
inline constexpr uint32_t kCB_Frame = 0;
inline constexpr uint32_t kPushC_Slot = 1;
inline constexpr uint32_t kCB_ASCull = 2;
inline constexpr uint32_t kPushCBytes = sizeof(uint32_t);

inline constexpr uint32_t kSRV_Positions = 0;
inline constexpr uint32_t kSRV_Normals = 1;
inline constexpr uint32_t kSRV_Tangents = 2;
inline constexpr uint32_t kSRV_Bitangents = 3;
inline constexpr uint32_t kSRV_UVs = 4;
inline constexpr uint32_t kSRV_MeshletVertIdx = 5;
inline constexpr uint32_t kSRV_MeshletPrimIdx = 6;
inline constexpr uint32_t kSRV_Meshlets = 7;
inline constexpr uint32_t kSRV_AssetLods = 8;
inline constexpr uint32_t kSRV_Vis = 9;
inline constexpr uint32_t kSRV_SlotOffsets = 10;
inline constexpr uint32_t kSRV_SlotCounts = 11;
inline constexpr uint32_t kSRV_Instances = 12;
inline constexpr uint32_t kSRV_ASInvocations = 13;
inline constexpr uint32_t kTex_Diffuse = 14;
inline constexpr uint32_t kTex_NormalMap = 15;
inline constexpr uint32_t kTex_ShadowMap = 16;
inline constexpr uint32_t kTex_HiZ = 17;
inline constexpr uint32_t kSampler_Main = 0;
inline constexpr uint32_t kSampler_Shadow = 1;
inline constexpr uint32_t kSampler_HiZ = 2;
} // namespace Draw

namespace Cull {
inline constexpr uint32_t kCB_Frame = 0;
inline constexpr uint32_t kSRV_RegionData = 0;
inline constexpr uint32_t kSRV_InstanceData = 1;
inline constexpr uint32_t kSRV_MainSlotOffsets = 2;
inline constexpr uint32_t kSRV_MainInvocations = 3;
inline constexpr uint32_t kSRV_ShadowSlotOffsets = 4;
inline constexpr uint32_t kSRV_ShadowInvocations = 5;
inline constexpr uint32_t kSRV_HiZ = 6;

inline constexpr uint32_t kUAV_MainRegionVis = 0;
inline constexpr uint32_t kUAV_MainCount = 1;
inline constexpr uint32_t kUAV_MainVis = 2;
inline constexpr uint32_t kUAV_MainDispatch = 3;
inline constexpr uint32_t kUAV_ShadowCount = 4;
inline constexpr uint32_t kUAV_ShadowVis = 5;
inline constexpr uint32_t kUAV_ShadowDispatch = 6;
inline constexpr uint32_t kUAV_ShadowUnique = 7;

inline constexpr uint32_t kSampler_HiZ = 0;
} // namespace Cull

namespace DepthPrepass {
inline constexpr uint32_t kCB_Frame = 0;
inline constexpr uint32_t kPushC_Slot = 1;
inline constexpr uint32_t kSRV_Vis = 0;
inline constexpr uint32_t kSRV_Instances = 1;
inline constexpr uint32_t kSRV_SlotOffsets = 2;
} // namespace DepthPrepass

namespace HiZ {
inline constexpr uint32_t kPushC_DestDimensions = 0;
inline constexpr uint32_t kSRV_Source = 0;
inline constexpr uint32_t kUAV_Dest = 0;
inline constexpr uint32_t kPushCDwordCount = 2;
} // namespace HiZ

namespace SDSM {
inline constexpr uint32_t kCB_Input = 0;
inline constexpr uint32_t kSRV_HiZ = 0;
inline constexpr uint32_t kUAV_CascadeOut = 0;
} // namespace SDSM

namespace Terrain {
inline constexpr uint32_t kCB_Frame = 0;
inline constexpr uint32_t kSampler_Shadow = 0;
inline constexpr uint32_t kTex_ShadowMap = 0;
} // namespace Terrain

namespace Sky {
inline constexpr uint32_t kCB_Sky = 0;
} // namespace Sky

inline constexpr uint32_t kPushC_RootParamIdx = 0;
} // namespace Mesh

} // namespace Xylem::shader::reg

#endif // XYLEM_SHADER_CONTRACTS_H
