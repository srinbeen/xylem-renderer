#include "include/MeshShaderRenderPass.hpp"
#include "include/Render.hpp"
#include "include/Globals.hpp"
#include "include/Terrain.hpp"
#include "include/macros.h"
#include "include/shaders/ShaderContracts.hpp"
#include "include/frame/FrameLifecycle.hpp"

#include <nvrhi/utils.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/math/math.h>
#include <donut/app/Timer.h>

#include <meshoptimizer.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>

using namespace donut::math;
#include <donut/shaders/sky_cb.h>

using namespace Xylem;
namespace shader_cb = Xylem::shader::cb;
namespace mesh_reg = Xylem::shader::reg::Mesh;
namespace compute_reg = Xylem::shader::reg::Compute;

// Impostor bake helpers (BuildtoObjectTransform / BuildtoProjTransform / etc.)
// now live in SharedGPUAssets — this pass only consumes the baked atlas.

// ===========================================================================
// Init
// ===========================================================================

bool MeshShaderRenderPass::Init() {
    // Scene-independent one-shot resources: CBs, shaders, samplers, pipelines,
    // binding layouts, command signatures. None of these touch m_Registry.
    if (!_InitShared())         return false;
    if (!_InitDrawResources())  return false;
    if (!_InitLeafResources())  return false;
    if (!_InitCullResources())  return false;
    if (!_InitShadowPass())     return false;
    if (!_InitDepthPrepass())   return false;
    if (!_InitHiZShaders())     return false;
    if (!_InitSDSMPass())       return false;
    if (!_InitSkyPass())        return false;
    if (!_InitImpostorPass())         return false;
    if (!_InitShadowImpostorPass())   return false;

    m_CommandList = GetDevice()->createCommandList();

    return LoadResources();
}

bool MeshShaderRenderPass::LoadResources() {
    auto initCL = GetDevice()->createCommandList();
    initCL->open();

    _RebuildMeshletMegabuffers(initCL);
    _UploadMeshletMegabuffers(initCL);

    if (!_InitTerrainPass(initCL))  { initCL->close(); return false; }

    _BuildRegionWindows();
    _BuildSlotLayout();
    _UploadCullBuffers(initCL);

    _RebuildCullBindingSet();
    _RebuildDrawBindingSet();
    _RebuildShadowBindingSet();
    _RebuildDepthPrepassBindingSet();
    _RebuildLeafBindingSets();
    _RebuildImpostorBindingSets();
    _RebuildShadowImpostorBindingSets();

    initCL->close();
    GetDevice()->executeCommandList(initCL);

    return true;
}

bool MeshShaderRenderPass::_InitShared() {
    m_StageResources.frameShared.constantBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(shader_cb::kCullFrameSize)
            .setIsConstantBuffer(true)
            .setDebugName("MeshShaderPass_CB")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
    );
    if (!m_StageResources.frameShared.constantBuffer) return false;

    m_StageResources.frameShared.asCullCB = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(shader_cb::kMeshASCullSize)
            .setIsConstantBuffer(true)
            .setDebugName("MeshShaderPass_ASCullCB")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
    );
    if (!m_StageResources.frameShared.asCullCB) return false;

    for (uint32_t i = 0; i < k_QueuedFrames; i++) {
        m_ReadbackBuffers[i] = GetDevice()->createBuffer(
            nvrhi::BufferDesc()
                .setByteSize(sizeof(uint32_t))  // sized properly in _UploadCullBuffers
                .setCpuAccess(nvrhi::CpuAccessMode::Read)
                .setInitialState(nvrhi::ResourceStates::CopyDest)
                .setKeepInitialState(true)
                .setDebugName("MeshCullCountReadback_" + std::to_string(i))
        );
    }

    for (uint32_t i = 0; i < k_QueuedFrames; ++i) {
        m_GpuTimers[i] = GetDevice()->createTimerQuery();
    }

    for (size_t s = 0; s < kStageCount; ++s) {
        for (uint32_t i = 0; i < k_QueuedFrames; ++i) {
            m_StageTimers[s][i] = GetDevice()->createTimerQuery();
        }
    }

    return true;
}

bool MeshShaderRenderPass::_InitDrawResources() {
    if (!m_ShaderFactory) { log::error("MeshShaderRenderPass: no ShaderFactory"); return false; }

    m_StageResources.sceneDraw.amplificationShader = m_ShaderFactory->CreateShader("app/MeshShaderPass.hlsl",
        "main_as", nullptr, nvrhi::ShaderType::Amplification);
    m_StageResources.sceneDraw.meshShader = m_ShaderFactory->CreateShader("app/MeshShaderPass.hlsl",
        "main_ms", nullptr, nvrhi::ShaderType::Mesh);
    m_StageResources.sceneDraw.pixelShader = m_ShaderFactory->CreateShader("app/MeshShaderPass.hlsl",
        "main_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_StageResources.sceneDraw.amplificationShader || !m_StageResources.sceneDraw.meshShader || !m_StageResources.sceneDraw.pixelShader) {
        log::error("MeshShaderRenderPass: shader compile failed");
        return false;
    }

    // Bark sampler now lives in SharedGPUAssets; the shadow sampler is pass-local
    // since it is used outside the bark binding.
    m_StageResources.sceneDraw.shadowSampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setMinFilter(true)
            .setMagFilter(true)
            .setMipFilter(false)
            .setReductionType(nvrhi::SamplerReductionType::Comparison)
            .setAllAddressModes(nvrhi::SamplerAddressMode::Border)
            .setBorderColor(nvrhi::Color(1.f))
        );
    if (!m_StageResources.sceneDraw.shadowSampler) return false;

    nvrhi::BindingLayoutDesc bld;
    bld.visibility = nvrhi::ShaderType::All;
    bld.bindings = {
        nvrhi::BindingLayoutItem::PushConstants(mesh_reg::Draw::kPushC_Slot, mesh_reg::Draw::kPushCBytes),
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::Draw::kCB_Frame),
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::Draw::kCB_ASCull),

        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Positions),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Normals),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Tangents),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Bitangents),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_UVs),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_MeshletVertIdx),
        nvrhi::BindingLayoutItem::RawBuffer_SRV(mesh_reg::Draw::kSRV_MeshletPrimIdx),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Meshlets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_AssetLods),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Vis),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_SlotOffsets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_SlotCounts),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Instances),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_ASInvocations),

        nvrhi::BindingLayoutItem::Texture_SRV(mesh_reg::Draw::kTex_Diffuse),
        nvrhi::BindingLayoutItem::Texture_SRV(mesh_reg::Draw::kTex_NormalMap),
        nvrhi::BindingLayoutItem::Texture_SRV(mesh_reg::Draw::kTex_ShadowMap),
        nvrhi::BindingLayoutItem::Texture_SRV(mesh_reg::Draw::kTex_HiZ),

        nvrhi::BindingLayoutItem::RawBuffer_UAV(mesh_reg::Draw::kUAV_MeshletStats),

        nvrhi::BindingLayoutItem::Sampler(mesh_reg::Draw::kSampler_Main),
        nvrhi::BindingLayoutItem::Sampler(mesh_reg::Draw::kSampler_Shadow),
    };
    m_StageResources.sceneDraw.bindingLayout = GetDevice()->createBindingLayout(bld);
    return m_StageResources.sceneDraw.bindingLayout != nullptr;
}

bool MeshShaderRenderPass::_InitLeafResources() {
    if (!m_ShaderFactory) return false;

    auto& L = m_StageResources.sceneLeaves;
    L.amplificationShader = m_ShaderFactory->CreateShader("app/MeshLeaves.hlsl",
        "leaf_as", nullptr, nvrhi::ShaderType::Amplification);
    L.shadowAmplificationShader = m_ShaderFactory->CreateShader("app/MeshLeaves.hlsl",
        "leaf_shadow_as", nullptr, nvrhi::ShaderType::Amplification);
    L.meshShader = m_ShaderFactory->CreateShader("app/MeshLeaves.hlsl",
        "leaf_ms", nullptr, nvrhi::ShaderType::Mesh);
    L.depthMS = m_ShaderFactory->CreateShader("app/MeshLeaves.hlsl",
        "leaf_depth_ms", nullptr, nvrhi::ShaderType::Mesh);
    L.shadowMS = m_ShaderFactory->CreateShader("app/MeshLeaves.hlsl",
        "leaf_shadow_ms", nullptr, nvrhi::ShaderType::Mesh);
    L.pixelShader = m_ShaderFactory->CreateShader("app/MeshLeaves.hlsl",
        "leaf_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!L.amplificationShader || !L.shadowAmplificationShader || !L.meshShader
        || !L.depthMS || !L.shadowMS || !L.pixelShader) {
        log::error("MeshShaderRenderPass: leaf AS/MS/PS compile failed");
        return false;
    }

    // Main color binding layout — must match MeshLeaves.hlsl:
    //   b0 CB (cull frame), b1 PushC (slotIdx), b2 ASCullCB (Hi-Z toggle/dims),
    //   t0..t7 leaf SRVs, t8 shadow Tex2DArray, t9 Hi-Z,
    //   u0 leaf survivor counter (raw),
    //   s0 shadow sampler.
    nvrhi::BindingLayoutDesc mainBLD;
    mainBLD.visibility = nvrhi::ShaderType::All;
    mainBLD.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::Leaf::kCB_Frame),
        nvrhi::BindingLayoutItem::PushConstants(mesh_reg::Leaf::kPushC_Slot, sizeof(uint32_t)),
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::Leaf::kCB_ASCull),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_Vis),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_SlotOffsets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_SlotCounts),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_Instances),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_ASInvocations),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_LeafInstances),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_LeafSlots),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_LeafMeshlets),
        nvrhi::BindingLayoutItem::Texture_SRV(mesh_reg::Leaf::kTex_ShadowMap),
        nvrhi::BindingLayoutItem::Texture_SRV(mesh_reg::Leaf::kTex_HiZ),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(mesh_reg::Leaf::kUAV_LeafSurvivor),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(mesh_reg::Leaf::kUAV_MeshletStats),
        nvrhi::BindingLayoutItem::Sampler(mesh_reg::Leaf::kSampler_Shadow),
    };
    L.mainLayout = GetDevice()->createBindingLayout(mainBLD);
    if (!L.mainLayout) return false;

    // Depth prepass: shares cull-side state (b0/b1/b2 + t0..t7 + t9 Hi-Z).
    // No shadow tex/sampler since there's no PS lighting.
    // Survivor counter is bound to a scratch buffer — values discarded.
    nvrhi::BindingLayoutDesc depthBLD;
    depthBLD.visibility = nvrhi::ShaderType::All;
    depthBLD.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::Leaf::kCB_Frame),
        nvrhi::BindingLayoutItem::PushConstants(mesh_reg::Leaf::kPushC_Slot, sizeof(uint32_t)),
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::Leaf::kCB_ASCull),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_Vis),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_SlotOffsets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_SlotCounts),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_Instances),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_ASInvocations),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_LeafInstances),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_LeafSlots),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_LeafMeshlets),
        nvrhi::BindingLayoutItem::Texture_SRV(mesh_reg::Leaf::kTex_HiZ),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(mesh_reg::Leaf::kUAV_LeafSurvivor),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(mesh_reg::Leaf::kUAV_MeshletStats),
    };
    L.depthLayout = GetDevice()->createBindingLayout(depthBLD);
    if (!L.depthLayout) return false;

    // Shadow leaves use leaf_shadow_as (no Hi-Z / no eye-frustum cull) and bind
    // to shadow vis/slot/count + shadow leaf slots. No b2 / Hi-Z.
    nvrhi::BindingLayoutDesc shadowBLD;
    shadowBLD.visibility = nvrhi::ShaderType::All;
    shadowBLD.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::Leaf::kCB_Frame),
        nvrhi::BindingLayoutItem::PushConstants(mesh_reg::Leaf::kPushC_Slot, sizeof(uint32_t)),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_Vis),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_SlotOffsets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_SlotCounts),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_Instances),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_ASInvocations),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_LeafInstances),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_LeafSlots),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_LeafMeshlets),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(mesh_reg::Leaf::kUAV_LeafSurvivor),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(mesh_reg::Leaf::kUAV_MeshletStats),
    };
    L.shadowLayout = GetDevice()->createBindingLayout(shadowBLD);
    return L.shadowLayout != nullptr;
}

bool MeshShaderRenderPass::_InitCullResources() {
    if (!m_ShaderFactory) return false;

    m_StageResources.cull.mainCS   = m_ShaderFactory->CreateShader("app/MeshCullCS.hlsl",
        "MeshCullMain",   nullptr, nvrhi::ShaderType::Compute);
    m_StageResources.cull.regionCS = m_ShaderFactory->CreateShader("app/MeshCullCS.hlsl",
        "MeshCullRegion", nullptr, nvrhi::ShaderType::Compute);
    m_StageResources.cull.shadowCS = m_ShaderFactory->CreateShader("app/MeshCullCS.hlsl",
        "MeshCullShadow", nullptr, nvrhi::ShaderType::Compute);
    if (!m_StageResources.cull.mainCS || !m_StageResources.cull.regionCS || !m_StageResources.cull.shadowCS) {
        log::error("MeshShaderRenderPass: MeshCullCS compile failed");
        return false;
    }

    nvrhi::BindingLayoutDesc bld;
    bld.visibility = nvrhi::ShaderType::All;
    bld.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::Cull::kCB_Frame),

        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_RegionData),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_InstanceData),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_MainSlotOffsets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_MainInvocations),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_ShadowSlotOffsets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_ShadowInvocations),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_ImpostorSlotOffsets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_MainLeafInvocations),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_ShadowLeafInvocations),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_ShadowImpostorSlotOffsets),

        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_MainRegionVis),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_MainCount),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_MainVis),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_MainDispatch),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_ShadowCount),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_ShadowVis),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_ShadowDispatch),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_ShadowUnique),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_ImpostorCount),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_ImpostorVis),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_ImpostorIndirectArgs),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_MainLeafDispatch),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_ShadowLeafDispatch),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_ShadowImpostorCount),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_ShadowImpostorVis),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_ShadowImpostorIndirectArgs),

        nvrhi::BindingLayoutItem::Texture_SRV(mesh_reg::Cull::kSRV_HiZ),
    };
    m_StageResources.cull.bindingLayout = GetDevice()->createBindingLayout(bld);
    if (!m_StageResources.cull.bindingLayout) return false;

    nvrhi::ComputePipelineDesc psoDescRegion;
    psoDescRegion.CS = m_StageResources.cull.regionCS;
    psoDescRegion.bindingLayouts = { m_StageResources.cull.bindingLayout };
    m_StageResources.cull.regionPipeline = GetDevice()->createComputePipeline(psoDescRegion);

    nvrhi::ComputePipelineDesc psoDescMain;
    psoDescMain.CS = m_StageResources.cull.mainCS;
    psoDescMain.bindingLayouts = { m_StageResources.cull.bindingLayout };
    m_StageResources.cull.mainPipeline = GetDevice()->createComputePipeline(psoDescMain);

    nvrhi::ComputePipelineDesc psoDescShadow;
    psoDescShadow.CS = m_StageResources.cull.shadowCS;
    psoDescShadow.bindingLayouts = { m_StageResources.cull.bindingLayout };
    m_StageResources.cull.shadowPipeline = GetDevice()->createComputePipeline(psoDescShadow);

    return m_StageResources.cull.mainPipeline && m_StageResources.cull.regionPipeline && m_StageResources.cull.shadowPipeline;
}

bool MeshShaderRenderPass::_InitShadowPass() {
    if (!m_ShaderFactory) return false;

    m_StageResources.shadow.amplificationShader = m_ShaderFactory->CreateShader("app/MeshShaderPass.hlsl",
        "shadow_as", nullptr, nvrhi::ShaderType::Amplification);
    m_StageResources.shadow.meshShader = m_ShaderFactory->CreateShader("app/MeshShaderPass.hlsl",
        "shadow_ms", nullptr, nvrhi::ShaderType::Mesh);
    if (!m_StageResources.shadow.amplificationShader || !m_StageResources.shadow.meshShader) {
        log::error("MeshShaderRenderPass: shadow AS/MS compile failed");
        return false;
    }

    // Shadow binding layout mirrors the subset the shadow AS/MS actually touch.
    // Uses the same t0..t13 register layout; the bindings point at shadow cull
    // buffers instead of main ones at bind-set build time.
    nvrhi::BindingLayoutDesc bld;
    bld.visibility = nvrhi::ShaderType::All;
    bld.bindings = {
        nvrhi::BindingLayoutItem::PushConstants(mesh_reg::Draw::kPushC_Slot, mesh_reg::Draw::kPushCBytes),
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::Draw::kCB_Frame),
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::Draw::kCB_ASCull),

        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Positions),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Normals),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Tangents),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Bitangents),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_UVs),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_MeshletVertIdx),
        nvrhi::BindingLayoutItem::RawBuffer_SRV(mesh_reg::Draw::kSRV_MeshletPrimIdx),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Meshlets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_AssetLods),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Vis),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_SlotOffsets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_SlotCounts),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Instances),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_ASInvocations),

        nvrhi::BindingLayoutItem::RawBuffer_UAV(mesh_reg::Draw::kUAV_MeshletStats),
    };
    m_StageResources.shadow.bindingLayout = GetDevice()->createBindingLayout(bld);
    if (!m_StageResources.shadow.bindingLayout) return false;

    // Texture2DArray shadow map - one slice per cascade.
    m_StageResources.shadow.depthTexture = GetDevice()->createTexture(nvrhi::TextureDesc()
        .setDimension(nvrhi::TextureDimension::Texture2DArray)
        .setWidth(k_ShadowRes).setHeight(k_ShadowRes)
        .setArraySize(Render::c_NumCascades)
        .setFormat(nvrhi::Format::D32)
        .setIsRenderTarget(true)
        .setUseClearValue(true)
        .setClearValue(nvrhi::Color(1.f, 1.f, 1.f, 1.f))
        .setInitialState(nvrhi::ResourceStates::DepthWrite)
        .setKeepInitialState(true)
        .setDebugName("MeshShader_ShadowDepth"));
    if (!m_StageResources.shadow.depthTexture) return false;

    for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
        nvrhi::FramebufferDesc fbd;
        fbd.setDepthAttachment(nvrhi::FramebufferAttachment()
            .setTexture(m_StageResources.shadow.depthTexture)
            .setArraySlice(c));
        m_StageResources.shadow.framebuffers[c] = GetDevice()->createFramebuffer(fbd);
        if (!m_StageResources.shadow.framebuffers[c]) return false;
    }

    // Single scratch texture for the currently-selected cascade slice (UI debug view).
    m_StageResources.shadow.debugSelectedCascadeTexture = GetDevice()->createTexture(nvrhi::TextureDesc()
        .setDimension(nvrhi::TextureDimension::Texture2D)
        .setWidth(k_ShadowRes).setHeight(k_ShadowRes)
        .setFormat(nvrhi::Format::R32_FLOAT)
        .setInitialState(nvrhi::ResourceStates::ShaderResource)
        .setKeepInitialState(true)
        .setDebugName("MeshShader_DebugSelectedCascade"));
    if (!m_StageResources.shadow.debugSelectedCascadeTexture) return false;

    // Terrain shadow caster path: AS/MS that culls per terrain meshlet against
    // the cascade's light frustum, then emits depth-only positions through
    // lightViewProj[cascade]. Cascade index travels via push constant at b3.
    // Reuses the eye-view terrain meshlet buffers (vert + meshlet descs/idx);
    // the binding set is built later in _RebuildTerrainBindingSet once those
    // exist.
    m_StageResources.shadow.terrainAS = m_ShaderFactory->CreateShader(
        "app/terrain_meshlet.hlsl", "shadow_terrain_as", nullptr, nvrhi::ShaderType::Amplification);
    m_StageResources.shadow.terrainMS = m_ShaderFactory->CreateShader(
        "app/terrain_meshlet.hlsl", "shadow_terrain_ms", nullptr, nvrhi::ShaderType::Mesh);
    if (!m_StageResources.shadow.terrainAS || !m_StageResources.shadow.terrainMS) {
        log::error("MeshShaderRenderPass: shadow_terrain AS/MS compile failed");
        return false;
    }

    nvrhi::BindingLayoutDesc terrBLD;
    terrBLD.visibility = nvrhi::ShaderType::All;
    terrBLD.bindings = {
        nvrhi::BindingLayoutItem::PushConstants(mesh_reg::Terrain::kPushC_Cascade, sizeof(uint32_t)),
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::Terrain::kCB_Frame),
        nvrhi::BindingLayoutItem::RawBuffer_SRV(mesh_reg::Terrain::kSRV_Positions),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Terrain::kSRV_MeshletDescs),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Terrain::kSRV_MeshletVertIdx),
        nvrhi::BindingLayoutItem::RawBuffer_SRV(mesh_reg::Terrain::kSRV_MeshletPrimIdx),
    };
    m_StageResources.shadow.terrainBindingLayout = GetDevice()->createBindingLayout(terrBLD);
    if (!m_StageResources.shadow.terrainBindingLayout) return false;

    return true;
}

// Bark texture loading lives in SharedGPUAssets.

// ===========================================================================
// Meshlet mega-buffer build - CPU side
// ===========================================================================

void MeshShaderRenderPass::_RebuildMeshletMegabuffers(nvrhi::ICommandList* /*cl*/) {
    const auto& assets      = m_Registry.getAssets();
    const uint32_t numLods  = static_cast<uint32_t>(m_Registry.getLodSegments().size());

    m_MeshletMegabuffers.meshOffsets.assign(assets.size() * static_cast<size_t>(numLods), {});

    m_MeshletMegabuffers.meshletDescs.clear();

    auto& vertexAttributes = m_MeshletMegabuffers.vertexAttributes;
    vertexAttributes.positions.clear();
    vertexAttributes.normals.clear();
    vertexAttributes.tangents.clear();
    vertexAttributes.bitangents.clear();
    vertexAttributes.uvs.clear();

    auto& meshletVerts = m_MeshletMegabuffers.localVertIndices;
    auto& meshletTris  = m_MeshletMegabuffers.localTriIndices;
    auto& meshletDescs = m_MeshletMegabuffers.meshletDescs;
    meshletVerts.clear();
    meshletTris.clear();

    for (size_t ai = 0; ai < assets.size(); ai++) {
        const auto& asset = assets[ai];
        for (uint32_t li = 0; li < asset.lods.size(); li++) {
            const auto& lod = asset.lods[li];

            auto& perMeshOffsets = m_MeshletMegabuffers.meshOffsets[ai*numLods+li];
            perMeshOffsets.meshletOffset       = static_cast<uint32_t>(meshletDescs.size());
            perMeshOffsets.vertexAttribOffset  = static_cast<uint32_t>(vertexAttributes.positions.size());
            perMeshOffsets.meshletVertOffset   = static_cast<uint32_t>(meshletVerts.size());
            perMeshOffsets.meshletTriOffset    = static_cast<uint32_t>(meshletTris.size());

            vertexAttributes.positions.insert(vertexAttributes.positions.end(),
                lod.positions.begin(),  lod.positions.end());
            vertexAttributes.normals.insert(vertexAttributes.normals.end(),
                lod.normals.begin(),    lod.normals.end());
            vertexAttributes.tangents.insert(vertexAttributes.tangents.end(),
                lod.tangents.begin(),   lod.tangents.end());
            vertexAttributes.bitangents.insert(vertexAttributes.bitangents.end(),
                lod.bitangents.begin(), lod.bitangents.end());
            vertexAttributes.uvs.insert(vertexAttributes.uvs.end(),
                lod.uvs.begin(),        lod.uvs.end());

            const size_t maxMeshlets = meshopt_buildMeshletsBound(
                lod.indices.size(), Render::k_MaxMeshletVerts, Render::k_MaxMeshletPrims);
            std::vector<meshopt_Meshlet> ml(maxMeshlets);
            std::vector<unsigned int>    mlVerts(maxMeshlets * Render::k_MaxMeshletVerts);
            std::vector<unsigned char>   mlTris(maxMeshlets * Render::k_MaxMeshletPrims * 3);

            const float* posPtr = lod.positions.empty() ? nullptr
                : reinterpret_cast<const float*>(lod.positions.data());

            size_t meshletCount = 0;
            if (!lod.indices.empty() && posPtr) {
                meshletCount = meshopt_buildMeshlets(
                    ml.data(), mlVerts.data(), mlTris.data(),
                    lod.indices.data(), lod.indices.size(),
                    posPtr, lod.positions.size(), sizeof(dm::float3),
                    Render::k_MaxMeshletVerts, Render::k_MaxMeshletPrims, 0.25f);
            }
            ml.resize(meshletCount);

            meshletDescs.reserve(meshletDescs.size() + meshletCount);
            for (size_t mi = 0; mi < meshletCount; mi++) {
                const meshopt_Meshlet& m = ml[mi];
                meshopt_Bounds b = meshopt_computeMeshletBounds(
                    mlVerts.data() + m.vertex_offset,
                    mlTris.data()  + m.triangle_offset,
                    m.triangle_count,
                    posPtr, lod.positions.size(), sizeof(dm::float3));

                Render::MeshletDesc desc;
                desc.vertOffset = static_cast<uint32_t>(meshletVerts.size()) - perMeshOffsets.meshletVertOffset;
                desc.vertCount  = m.vertex_count;
                desc.triOffset  = static_cast<uint32_t>(meshletTris.size())  - perMeshOffsets.meshletTriOffset;
                desc.triCount   = m.triangle_count;
                desc.bounds         = dm::float4(b.center[0],   b.center[1],   b.center[2],   b.radius);
                desc.coneApex       = dm::float4(b.cone_apex[0],b.cone_apex[1],b.cone_apex[2], 0.f);
                desc.coneAxisCutoff = dm::float4(b.cone_axis[0],b.cone_axis[1],b.cone_axis[2], b.cone_cutoff);
                meshletDescs.push_back(desc);

                meshletVerts.insert(meshletVerts.end(),
                    mlVerts.data() + m.vertex_offset,
                    mlVerts.data() + m.vertex_offset + m.vertex_count);

                meshletTris.insert(meshletTris.end(),
                    mlTris.data() + m.triangle_offset,
                    mlTris.data() + m.triangle_offset + m.triangle_count * 3);
            }

            size_t misalign = (meshletTris.size() - perMeshOffsets.meshletTriOffset) % 4;
            if (misalign) meshletTris.insert(meshletTris.end(), 4 - misalign, 0);

            perMeshOffsets.meshletCount = static_cast<uint32_t>(meshletCount);
        }
    }

    // The shader may load one extra uint when unpacking a triangle that straddles
    // a 4-byte boundary; keep a small sentinel at the end of the packed stream.
    if (!meshletTris.empty())
        meshletTris.insert(meshletTris.end(), 4, 0);
}

void MeshShaderRenderPass::_UploadMeshletMegabuffers(nvrhi::ICommandList* cl) {
    auto& vertexAttributes = m_MeshletMegabuffers.vertexAttributes;
    auto& meshletVerts     = m_MeshletMegabuffers.localVertIndices;
    auto& meshletTris      = m_MeshletMegabuffers.localTriIndices;
    auto& meshletDescs     = m_MeshletMegabuffers.meshletDescs;

    const uint32_t totalVerts     = static_cast<uint32_t>(vertexAttributes.positions.size());
    const uint32_t totalMeshlets  = static_cast<uint32_t>(meshletDescs.size());
    const uint32_t totalVertIdx   = static_cast<uint32_t>(meshletVerts.size());
    const uint32_t totalPrimBytes = static_cast<uint32_t>(meshletTris.size());
    const uint32_t numAssetLods   = static_cast<uint32_t>(m_MeshletMegabuffers.meshOffsets.size());

    auto makeSrv = [&](nvrhi::BufferHandle& h, const void* data, size_t bytes, uint32_t stride,
                       const char* name, bool raw) {
        nvrhi::BufferDesc d;
        d.byteSize      = std::max<size_t>(bytes, stride ? stride : 4);
        d.debugName     = name;
        d.initialState  = nvrhi::ResourceStates::CopyDest;
        if (raw) {
            d.canHaveRawViews = true;
        } else {
            d.structStride = stride;
        }
        h = GetDevice()->createBuffer(d);
        cl->beginTrackingBufferState(h, nvrhi::ResourceStates::CopyDest);
        if (data && bytes > 0) cl->writeBuffer(h, data, bytes);
        cl->setPermanentBufferState(h, nvrhi::ResourceStates::ShaderResource);
    };

    makeSrv(m_StageResources.sceneMeshletData.positions,      vertexAttributes.positions.data(),  totalVerts * sizeof(dm::float3),
            sizeof(dm::float3), "Mesh_Positions",  false);
    makeSrv(m_StageResources.sceneMeshletData.normals,        vertexAttributes.normals.data(),    totalVerts * sizeof(dm::float3),
            sizeof(dm::float3), "Mesh_Normals",    false);
    makeSrv(m_StageResources.sceneMeshletData.tangents,       vertexAttributes.tangents.data(),   totalVerts * sizeof(dm::float3),
            sizeof(dm::float3), "Mesh_Tangents",   false);
    makeSrv(m_StageResources.sceneMeshletData.bitangents,     vertexAttributes.bitangents.data(), totalVerts * sizeof(dm::float3),
            sizeof(dm::float3), "Mesh_Bitangents", false);
    makeSrv(m_StageResources.sceneMeshletData.uvs,            vertexAttributes.uvs.data(),        totalVerts * sizeof(dm::float2),
            sizeof(dm::float2), "Mesh_UVs",        false);
    makeSrv(m_StageResources.sceneMeshletData.meshletVertIdx, meshletVerts.data(), totalVertIdx * sizeof(uint32_t),
            sizeof(uint32_t),   "Mesh_MeshletVertIdx", false);
    makeSrv(m_StageResources.sceneMeshletData.meshletPrimIdx, meshletTris.data(), totalPrimBytes,
            0,                  "Mesh_MeshletTriIdx", true);
    makeSrv(m_StageResources.sceneMeshletData.meshletDescs,   meshletDescs.data(),       totalMeshlets * sizeof(Render::MeshletDesc),
            sizeof(Render::MeshletDesc),   "Mesh_Meshlets", false);
    makeSrv(m_StageResources.sceneMeshletData.assetLodRanges, m_MeshletMegabuffers.meshOffsets.data(),
            numAssetLods * sizeof(Render::MeshOffsets),
            sizeof(Render::MeshOffsets), "Mesh_Offsets", false);

    m_StageResources.sceneMeshletData.totalVertices     = totalVerts;
    m_StageResources.sceneMeshletData.totalMeshlets     = totalMeshlets;
    m_StageResources.sceneMeshletData.totalVertIdx      = totalVertIdx;
    m_StageResources.sceneMeshletData.totalPrimIdxBytes = totalPrimBytes;
    m_StageResources.sceneMeshletData.numAssetLods      = numAssetLods;

    size_t totalBytes =
        totalVerts * (sizeof(dm::float3) * 4 + sizeof(dm::float2)) +
        totalVertIdx * sizeof(uint32_t) +
        totalPrimBytes +
        totalMeshlets * sizeof(Render::MeshletDesc) +
        numAssetLods * sizeof(Render::MeshOffsets);
    m_UI.meshletMegaBufferMB = static_cast<float>(totalBytes) / (1024.f * 1024.f);
    m_UI.totalMeshletCount   = totalMeshlets;
}

// ===========================================================================
// Region windows + slot layout
// ===========================================================================

void MeshShaderRenderPass::_BuildRegionWindows() {
    const auto& regions = m_Registry.getRegions();
    const auto& assets  = m_Registry.getAssets();
    m_RegionWindows.resize(regions.size());
    m_TotalCapacity = 0;

    for (size_t i = 0; i < regions.size(); i++) {
        auto& win = m_RegionWindows[i];
        uint32_t count = static_cast<uint32_t>(regions[i].instances.size());
        win.count    = count;
        win.capacity = std::max(16u, static_cast<uint32_t>(std::ceil(count * k_CapacitySlack)));
        win.offset   = m_TotalCapacity;
        m_TotalCapacity += win.capacity;
    }
    m_TotalCapacity = std::max(1u, m_TotalCapacity);

    const uint32_t numLods = static_cast<uint32_t>(m_Registry.getLodSegments().size());

    m_InstanceStaging.assign(m_TotalCapacity, Render::InstanceBufferEntry{});
    m_CullDataStaging.assign(m_TotalCapacity, Render::CullInstanceData{});
    m_RegionStaging.assign(std::max<size_t>(1, regions.size()), Render::CullRegionData{});

    for (size_t r = 0; r < regions.size(); r++) {
        const auto& win = m_RegionWindows[r];
        const auto& reg = regions[r];
        m_RegionStaging[r] = { reg.cullBox };

        for (uint32_t i = 0; i < reg.instances.size(); i++) {
            const auto& inst = reg.instances[i];
            size_t assetIdx = m_Registry.assetIndexById(inst.assetId);
            if (assetIdx == SIZE_MAX) continue;

            uint32_t idx = win.offset + i;
            m_InstanceStaging[idx] = Render::InstanceBufferEntry(
                inst.model, inst.normal, static_cast<uint32_t>(assetIdx));

            const dm::box3& localBbox = assets[assetIdx].lods[0].bbox;
            dm::box3 worldBbox = localBbox * dm::homogeneousToAffine(inst.model);

            auto& cd = m_CullDataStaging[idx];
            cd.bbox     = worldBbox;
            cd.baseSlot = static_cast<uint32_t>(assetIdx) * numLods;
            cd.regionId = static_cast<uint32_t>(r);
            cd.active   = 1;
        }
    }
}

void MeshShaderRenderPass::_BuildSlotLayout() {
    const auto& assets      = m_Registry.getAssets();
    const uint32_t numLods   = static_cast<uint32_t>(m_Registry.getLodSegments().size());
    const uint32_t numAssets = static_cast<uint32_t>(assets.size());
    const uint32_t nCasc     = Render::c_NumCascades;

    m_NumMainSlots   = std::max(1u, numAssets * numLods);
    m_NumShadowSlots = std::max(1u, numAssets * nCasc);

    std::vector<uint32_t> livePerAsset(std::max(1u, numAssets), 0);
    for (const auto& win : m_RegionWindows) {
        for (uint32_t i = 0; i < win.count; i++) {
            uint32_t idx = win.offset + i;
            if (m_CullDataStaging[idx].active) {
                uint32_t treeId = m_InstanceStaging[idx].treeId;
                if (treeId < numAssets) livePerAsset[treeId]++;
            }
        }
    }

    // Main slots (asset x LOD)
    m_MainSlotOffsets.assign(m_NumMainSlots, 0);
    m_MainASInvocsPerSlot.assign(m_NumMainSlots, 0);
    m_MainLeafASInvocsPerSlot.assign(m_NumMainSlots, 0);
    m_MainSlotTextureSet.assign(m_NumMainSlots, 0);
    m_MainDispatchArgsStaging.assign(m_NumMainSlots, {0, 0, 1, 1});
    m_MainLeafDispatchArgsStaging.assign(m_NumMainSlots, {0, 0, 1, 1});
    m_MainVisBufferSize = 0;

    for (uint32_t ai = 0; ai < numAssets; ai++) {
        for (uint32_t lodi = 0; lodi < numLods; lodi++) {
            uint32_t slot = ai * numLods + lodi;
            m_MainSlotOffsets[slot] = m_MainVisBufferSize;
            m_MainVisBufferSize    += livePerAsset[ai];

            uint32_t meshletCount = (slot < m_MeshletMegabuffers.meshOffsets.size())
                ? m_MeshletMegabuffers.meshOffsets[slot].meshletCount : 0u;
            uint32_t invocations = std::max(1u,
                (meshletCount + Render::k_ASGroupSize - 1) / Render::k_ASGroupSize);
            m_MainASInvocsPerSlot[slot] = invocations;
            m_MainDispatchArgsStaging[slot].slotIdx = slot;
            m_MainLeafDispatchArgsStaging[slot].slotIdx = slot;

            // Leaf AS invocations per visible instance: ceil(leafMeshletCount / AS_GROUP_SIZE).
            // Zero when the asset has no leaves at this LOD; cull shader skips the increment.
            uint32_t leafMeshletCount = 0;
            if (ai < assets.size() && lodi < assets[ai].leafAsset.lodSlots.size())
                leafMeshletCount = assets[ai].leafAsset.lodSlots[lodi].meshletCount;
            m_MainLeafASInvocsPerSlot[slot] = leafMeshletCount > 0
                ? (leafMeshletCount + Render::k_ASGroupSize - 1) / Render::k_ASGroupSize
                : 0u;

            if (ai < assets.size()) {
                uint32_t textureSet = assets[ai].textureSetIdx;
                const auto& barkTextures = m_Shared->barkTextures();
                if (!barkTextures.empty())
                    textureSet = std::min<uint32_t>(textureSet, static_cast<uint32_t>(barkTextures.size() - 1));
                m_MainSlotTextureSet[slot] = textureSet;
            }
        }
    }
    m_MainVisBufferSize = std::max(1u, m_MainVisBufferSize);

    // Shadow slots (asset x cascade, LOD 0 only)
    m_ShadowSlotOffsets.assign(m_NumShadowSlots, 0);
    m_ShadowASInvocsPerSlot.assign(m_NumShadowSlots, 0);
    m_ShadowLeafASInvocsPerSlot.assign(m_NumShadowSlots, 0);
    m_ShadowDispatchArgsStaging.assign(m_NumShadowSlots, {0, 0, 1, 1});
    m_ShadowLeafDispatchArgsStaging.assign(m_NumShadowSlots, {0, 0, 1, 1});
    m_ShadowVisBufferSize = 0;

    const uint32_t lowestLod = numLods > 0 ? numLods - 1 : 0;
    for (uint32_t ai = 0; ai < numAssets; ai++) {
        for (uint32_t c = 0; c < nCasc; c++) {
            uint32_t slot = ai * nCasc + c;
            m_ShadowSlotOffsets[slot] = m_ShadowVisBufferSize;
            m_ShadowVisBufferSize    += livePerAsset[ai];

            // Shadow casts from the lowest (coarsest) LOD to match the compute
            // pipeline. Casting from LOD 0 (round) while the color pass renders
            // LOD N-1 (square inscribed in that circle) self-shadows every face.
            uint32_t shadowLodSlot = ai * numLods + lowestLod;
            uint32_t meshletCount = (shadowLodSlot < m_MeshletMegabuffers.meshOffsets.size())
                ? m_MeshletMegabuffers.meshOffsets[shadowLodSlot].meshletCount : 0u;
            uint32_t invocations = std::max(1u,
                (meshletCount + Render::k_ASGroupSize - 1) / Render::k_ASGroupSize);
            m_ShadowASInvocsPerSlot[slot] = invocations;
            m_ShadowDispatchArgsStaging[slot].slotIdx = slot;
            m_ShadowLeafDispatchArgsStaging[slot].slotIdx = slot;

            uint32_t leafMeshletCount = 0;
            if (ai < assets.size() && lowestLod < assets[ai].leafAsset.lodSlots.size())
                leafMeshletCount = assets[ai].leafAsset.lodSlots[lowestLod].meshletCount;
            m_ShadowLeafASInvocsPerSlot[slot] = leafMeshletCount > 0
                ? (leafMeshletCount + Render::k_ASGroupSize - 1) / Render::k_ASGroupSize
                : 0u;
        }
    }
    m_ShadowVisBufferSize = std::max(1u, m_ShadowVisBufferSize);

    // Impostor terminal-LOD slots: one visibility window per asset.
    m_ImpostorMaxSlotCounts.assign(std::max(1u, numAssets), 0);
    m_ImpostorSlotOffsets.assign(std::max(1u, numAssets), 0);
    m_ImpostorVisBufferSize = 0;
    for (uint32_t ai = 0; ai < numAssets; ai++) {
        m_ImpostorMaxSlotCounts[ai] = livePerAsset[ai];
        m_ImpostorSlotOffsets[ai]   = m_ImpostorVisBufferSize;
        m_ImpostorVisBufferSize    += livePerAsset[ai];
    }
    m_ImpostorVisBufferSize = std::max(1u, m_ImpostorVisBufferSize);

    // Shadow impostor slots: numAssets * numCascades, each sized livePerAsset[ai]
    const uint32_t numShadowImpostorSlots = std::max(1u, numAssets * nCasc);
    m_ShadowImpostorSlotOffsets.assign(numShadowImpostorSlots, 0);
    m_ShadowImpostorMaxSlotCounts.assign(numShadowImpostorSlots, 0);
    m_ShadowImpostorVisBufferSize = 0;
    for (uint32_t ai = 0; ai < numAssets; ai++) {
        for (uint32_t c = 0; c < nCasc; c++) {
            const uint32_t slot = ai * nCasc + c;
            m_ShadowImpostorMaxSlotCounts[slot] = livePerAsset[ai];
            m_ShadowImpostorSlotOffsets[slot]   = m_ShadowImpostorVisBufferSize;
            m_ShadowImpostorVisBufferSize      += livePerAsset[ai];
        }
    }
    m_ShadowImpostorVisBufferSize = std::max(1u, m_ShadowImpostorVisBufferSize);
}

void MeshShaderRenderPass::_UploadCullBuffers(nvrhi::ICommandList* cl) {
    auto device = GetDevice();
    const uint32_t numRegions = std::max(1u,
        static_cast<uint32_t>(m_Registry.getRegions().size()));

    // Persistent + cull data
    m_StageResources.cull.persistentInstBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_TotalCapacity * sizeof(Render::InstanceBufferEntry))
        .setStructStride(sizeof(Render::InstanceBufferEntry))
        .setDebugName("Mesh_PersistentInstBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_StageResources.cull.persistentInstBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_StageResources.cull.persistentInstBuffer, m_InstanceStaging.data(),
        m_TotalCapacity * sizeof(Render::InstanceBufferEntry));
    cl->setPermanentBufferState(m_StageResources.cull.persistentInstBuffer, nvrhi::ResourceStates::ShaderResource);

    m_StageResources.cull.cullDataBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_TotalCapacity * sizeof(Render::CullInstanceData))
        .setStructStride(sizeof(Render::CullInstanceData))
        .setDebugName("Mesh_CullDataBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_StageResources.cull.cullDataBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_StageResources.cull.cullDataBuffer, m_CullDataStaging.data(),
        m_TotalCapacity * sizeof(Render::CullInstanceData));
    cl->setPermanentBufferState(m_StageResources.cull.cullDataBuffer, nvrhi::ResourceStates::ShaderResource);

    m_StageResources.cull.cullRegionDataBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(numRegions * sizeof(Render::CullRegionData))
        .setStructStride(sizeof(Render::CullRegionData))
        .setDebugName("Mesh_CullRegionDataBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_StageResources.cull.cullRegionDataBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_StageResources.cull.cullRegionDataBuffer, m_RegionStaging.data(),
        numRegions * sizeof(Render::CullRegionData));
    cl->setPermanentBufferState(m_StageResources.cull.cullRegionDataBuffer, nvrhi::ResourceStates::ShaderResource);

    // --- Main slot SRVs ---
    m_StageResources.cull.mainSlotOffsetBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumMainSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_MainSlotOffsetBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_StageResources.cull.mainSlotOffsetBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_StageResources.cull.mainSlotOffsetBuffer, m_MainSlotOffsets.data(),
        m_NumMainSlots * sizeof(uint32_t));
    cl->setPermanentBufferState(m_StageResources.cull.mainSlotOffsetBuffer, nvrhi::ResourceStates::ShaderResource);

    m_StageResources.cull.mainASInvocsPerSlotBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumMainSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_MainASInvocsPerSlotBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_StageResources.cull.mainASInvocsPerSlotBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_StageResources.cull.mainASInvocsPerSlotBuffer, m_MainASInvocsPerSlot.data(),
        m_NumMainSlots * sizeof(uint32_t));
    cl->setPermanentBufferState(m_StageResources.cull.mainASInvocsPerSlotBuffer, nvrhi::ResourceStates::ShaderResource);

    // --- Shadow slot SRVs ---
    m_StageResources.cull.shadowSlotOffsetBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumShadowSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_ShadowSlotOffsetBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_StageResources.cull.shadowSlotOffsetBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_StageResources.cull.shadowSlotOffsetBuffer, m_ShadowSlotOffsets.data(),
        m_NumShadowSlots * sizeof(uint32_t));
    cl->setPermanentBufferState(m_StageResources.cull.shadowSlotOffsetBuffer, nvrhi::ResourceStates::ShaderResource);

    m_StageResources.cull.shadowASInvocsPerSlotBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumShadowSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_ShadowASInvocsPerSlotBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_StageResources.cull.shadowASInvocsPerSlotBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_StageResources.cull.shadowASInvocsPerSlotBuffer, m_ShadowASInvocsPerSlot.data(),
        m_NumShadowSlots * sizeof(uint32_t));
    cl->setPermanentBufferState(m_StageResources.cull.shadowASInvocsPerSlotBuffer, nvrhi::ResourceStates::ShaderResource);

    m_StageResources.cull.mainLeafASInvocsPerSlotBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumMainSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_MainLeafASInvocsPerSlotBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_StageResources.cull.mainLeafASInvocsPerSlotBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_StageResources.cull.mainLeafASInvocsPerSlotBuffer, m_MainLeafASInvocsPerSlot.data(),
        m_NumMainSlots * sizeof(uint32_t));
    cl->setPermanentBufferState(m_StageResources.cull.mainLeafASInvocsPerSlotBuffer, nvrhi::ResourceStates::ShaderResource);

    m_StageResources.cull.shadowLeafASInvocsPerSlotBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumShadowSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_ShadowLeafASInvocsPerSlotBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_StageResources.cull.shadowLeafASInvocsPerSlotBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_StageResources.cull.shadowLeafASInvocsPerSlotBuffer, m_ShadowLeafASInvocsPerSlot.data(),
        m_NumShadowSlots * sizeof(uint32_t));
    cl->setPermanentBufferState(m_StageResources.cull.shadowLeafASInvocsPerSlotBuffer, nvrhi::ResourceStates::ShaderResource);

    // --- UAVs ---
    m_StageResources.cull.regionVisibleBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(numRegions * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_RegionVisibleBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.mainCountBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumMainSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_MainCountBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.mainVisBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_MainVisBufferSize * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_MainVisBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.mainDispatchArgsBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumMainSlots * sizeof(DispatchRecord))
        .setDebugName("Mesh_MainDispatchArgsBuffer")
        .setIsDrawIndirectArgs(true)
        .setCanHaveUAVs(true)
        .setCanHaveRawViews(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.shadowCountBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumShadowSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_ShadowCountBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.shadowVisBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_ShadowVisBufferSize * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_ShadowVisBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.shadowDispatchArgsBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumShadowSlots * sizeof(DispatchRecord))
        .setDebugName("Mesh_ShadowDispatchArgsBuffer")
        .setIsDrawIndirectArgs(true)
        .setCanHaveUAVs(true)
        .setCanHaveRawViews(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.mainLeafDispatchArgsBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumMainSlots * sizeof(DispatchRecord))
        .setDebugName("Mesh_MainLeafDispatchArgsBuffer")
        .setIsDrawIndirectArgs(true)
        .setCanHaveUAVs(true)
        .setCanHaveRawViews(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.shadowLeafDispatchArgsBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumShadowSlots * sizeof(DispatchRecord))
        .setDebugName("Mesh_ShadowLeafDispatchArgsBuffer")
        .setIsDrawIndirectArgs(true)
        .setCanHaveUAVs(true)
        .setCanHaveRawViews(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.shadowUniqueCounter = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(sizeof(uint32_t))
        .setDebugName("Mesh_ShadowUniqueCounter")
        .setCanHaveUAVs(true)
        .setCanHaveRawViews(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    auto makeLeafSurvivorBuf = [&](const char* name) {
        return device->createBuffer(nvrhi::BufferDesc()
            .setByteSize(sizeof(uint32_t))
            .setDebugName(name)
            .setCanHaveUAVs(true)
            .setCanHaveRawViews(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));
    };
    m_StageResources.cull.mainLeafSurvivorCounter   = makeLeafSurvivorBuf("Mesh_MainLeafSurvivorCounter");
    m_StageResources.cull.depthLeafSurvivorScratch  = makeLeafSurvivorBuf("Mesh_DepthLeafSurvivorScratch");
    m_StageResources.cull.shadowLeafSurvivorCounter = makeLeafSurvivorBuf("Mesh_ShadowLeafSurvivorCounter");

    m_StageResources.cull.meshletStatsBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(8 * sizeof(uint32_t))
        .setDebugName("Mesh_MeshletStatsBuffer")
        .setCanHaveUAVs(true)
        .setCanHaveRawViews(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));
    m_StageResources.cull.meshletStatsDepthScratch = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(8 * sizeof(uint32_t))
        .setDebugName("Mesh_MeshletStatsDepthScratch")
        .setCanHaveUAVs(true)
        .setCanHaveRawViews(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    // --- Impostor cull buffers (per-asset slots) ---
    const uint32_t numAssets = std::max(1u,
        static_cast<uint32_t>(m_Registry.getAssets().size()));

    m_StageResources.cull.impostorSlotOffsetBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(numAssets * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_ImpostorSlotOffsetBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_StageResources.cull.impostorSlotOffsetBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_StageResources.cull.impostorSlotOffsetBuffer, m_ImpostorSlotOffsets.data(),
        numAssets * sizeof(uint32_t));
    cl->setPermanentBufferState(m_StageResources.cull.impostorSlotOffsetBuffer, nvrhi::ResourceStates::ShaderResource);

    m_StageResources.cull.impostorCountBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(numAssets * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_ImpostorCountBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.impostorVisBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_ImpostorVisBufferSize * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_ImpostorVisBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.impostorIndirectArgsBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(numAssets * sizeof(nvrhi::DrawIndirectArguments))
        .setDebugName("Mesh_ImpostorIndirectArgsBuffer")
        .setIsDrawIndirectArgs(true)
        .setCanHaveUAVs(true)
        .setCanHaveRawViews(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    // The depth prepass consumes "last frame" mesh dispatch/count buffers before
    // the first per-frame cull reset runs. Seed those resources so frame zero is
    // a clean no-op instead of reading undefined indirect args.
    cl->clearBufferUInt(m_StageResources.cull.mainCountBuffer, 0);
    cl->clearBufferUInt(m_StageResources.cull.shadowCountBuffer, 0);
    cl->clearBufferUInt(m_StageResources.cull.shadowUniqueCounter, 0);
    cl->clearBufferUInt(m_StageResources.cull.impostorCountBuffer, 0);
    cl->writeBuffer(m_StageResources.cull.mainDispatchArgsBuffer,
        m_MainDispatchArgsStaging.data(),
        m_MainDispatchArgsStaging.size() * sizeof(DispatchRecord));
    cl->writeBuffer(m_StageResources.cull.shadowDispatchArgsBuffer,
        m_ShadowDispatchArgsStaging.data(),
        m_ShadowDispatchArgsStaging.size() * sizeof(DispatchRecord));
    cl->writeBuffer(m_StageResources.cull.mainLeafDispatchArgsBuffer,
        m_MainLeafDispatchArgsStaging.data(),
        m_MainLeafDispatchArgsStaging.size() * sizeof(DispatchRecord));
    cl->writeBuffer(m_StageResources.cull.shadowLeafDispatchArgsBuffer,
        m_ShadowLeafDispatchArgsStaging.data(),
        m_ShadowLeafDispatchArgsStaging.size() * sizeof(DispatchRecord));

    // Pre-fill impostor indirect args: vertexCount=4 (triangle-strip quad),
    // instanceCount=0 (CS atomically increments each frame).
    {
        std::vector<nvrhi::DrawIndirectArguments> impostorArgs(numAssets);
        for (uint32_t ai = 0; ai < numAssets; ai++) {
            auto& a = impostorArgs[ai];
            a.vertexCount           = 4;
            a.instanceCount         = 0;
            a.startVertexLocation   = 0;
            a.startInstanceLocation = 0;
        }
        cl->writeBuffer(m_StageResources.cull.impostorIndirectArgsBuffer,
            impostorArgs.data(), numAssets * sizeof(nvrhi::DrawIndirectArguments));
    }

    // --- Shadow impostor cull buffers (per (asset, cascade)) ---
    const uint32_t numShadowImpostorSlots =
        std::max(1u, static_cast<uint32_t>(m_ShadowImpostorSlotOffsets.size()));

    m_StageResources.cull.shadowImpostorSlotOffsetBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(numShadowImpostorSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_ShadowImpostorSlotOffsetBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_StageResources.cull.shadowImpostorSlotOffsetBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_StageResources.cull.shadowImpostorSlotOffsetBuffer,
        m_ShadowImpostorSlotOffsets.data(),
        numShadowImpostorSlots * sizeof(uint32_t));
    cl->setPermanentBufferState(m_StageResources.cull.shadowImpostorSlotOffsetBuffer,
        nvrhi::ResourceStates::ShaderResource);

    m_StageResources.cull.shadowImpostorCountBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(numShadowImpostorSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setCanHaveUAVs(true)
        .setDebugName("Mesh_ShadowImpostorCountBuffer")
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.shadowImpostorVisBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_ShadowImpostorVisBufferSize * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setCanHaveUAVs(true)
        .setDebugName("Mesh_ShadowImpostorVisBuffer")
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.shadowImpostorIndirectArgsBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(numShadowImpostorSlots * sizeof(nvrhi::DrawIndirectArguments))
        .setDebugName("Mesh_ShadowImpostorIndirectArgsBuffer")
        .setIsDrawIndirectArgs(true)
        .setCanHaveUAVs(true)
        .setCanHaveRawViews(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    // Pre-fill shadow impostor indirect args: vertexCount=4, instanceCount=0 per (asset, cascade)
    {
        std::vector<nvrhi::DrawIndirectArguments> args(numShadowImpostorSlots);
        for (uint32_t i = 0; i < numShadowImpostorSlots; i++) {
            auto& a = args[i];
            a.vertexCount   = 4;
            a.instanceCount = 0;
            a.startVertexLocation   = 0;
            a.startInstanceLocation = 0;
        }
        cl->writeBuffer(m_StageResources.cull.shadowImpostorIndirectArgsBuffer,
            args.data(), numShadowImpostorSlots * sizeof(nvrhi::DrawIndirectArguments));
    }

    // Readback ring: [mainCounts][shadowCounts][impostorCounts][shadowImpostorCounts][shadowUnique][...]
    m_ReadbackMainEntries           = m_NumMainSlots;
    m_ReadbackShadowEntries         = m_NumShadowSlots;
    m_ReadbackImpostorEntries       = numAssets;
    m_ReadbackShadowImpostorEntries = numAssets * XYLEM_NUM_CASCADES;
    // Layout: [mainCounts][shadowCounts][impostorCounts][shadowImpostorCounts][shadowUnique]
    //         [mainLeafSurvivors][shadowLeafSurvivors][terrainSurvivors][meshletStats × 8]
    const uint64_t readbackSize =
        (m_ReadbackMainEntries + m_ReadbackShadowEntries + m_ReadbackImpostorEntries + m_ReadbackShadowImpostorEntries + 4 + 8) * sizeof(uint32_t);
    for (uint32_t i = 0; i < k_QueuedFrames; i++) {
        m_ReadbackBuffers[i] = device->createBuffer(nvrhi::BufferDesc()
            .setByteSize(readbackSize)
            .setCpuAccess(nvrhi::CpuAccessMode::Read)
            .setInitialState(nvrhi::ResourceStates::CopyDest)
            .setKeepInitialState(true)
            .setDebugName("MeshCullCountReadback_" + std::to_string(i)));
    }
    m_ReadbackFrameIndex = 0;
}

// ===========================================================================
// Binding sets
// ===========================================================================

void MeshShaderRenderPass::_RebuildCullBindingSet() {
    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Cull::kCB_Frame, m_StageResources.frameShared.constantBuffer,
            nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),

        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_RegionData,         m_StageResources.cull.cullRegionDataBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_InstanceData,       m_StageResources.cull.cullDataBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_MainSlotOffsets,   m_StageResources.cull.mainSlotOffsetBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_MainInvocations,     m_StageResources.cull.mainASInvocsPerSlotBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_ShadowSlotOffsets, m_StageResources.cull.shadowSlotOffsetBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_ShadowInvocations,   m_StageResources.cull.shadowASInvocsPerSlotBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_ImpostorSlotOffsets, m_StageResources.cull.impostorSlotOffsetBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_MainLeafInvocations, m_StageResources.cull.mainLeafASInvocsPerSlotBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_ShadowLeafInvocations, m_StageResources.cull.shadowLeafASInvocsPerSlotBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_ShadowImpostorSlotOffsets, m_StageResources.cull.shadowImpostorSlotOffsetBuffer),

        nvrhi::BindingSetItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_MainRegionVis,  m_StageResources.cull.regionVisibleBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_MainCount,      m_StageResources.cull.mainCountBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_MainVis,        m_StageResources.cull.mainVisBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_MainDispatch,          m_StageResources.cull.mainDispatchArgsBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_ShadowCount,    m_StageResources.cull.shadowCountBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_ShadowVis,      m_StageResources.cull.shadowVisBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_ShadowDispatch,        m_StageResources.cull.shadowDispatchArgsBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_ShadowUnique,          m_StageResources.cull.shadowUniqueCounter),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_ImpostorCount,  m_StageResources.cull.impostorCountBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_ImpostorVis,    m_StageResources.cull.impostorVisBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_ImpostorIndirectArgs,  m_StageResources.cull.impostorIndirectArgsBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_MainLeafDispatch,      m_StageResources.cull.mainLeafDispatchArgsBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_ShadowLeafDispatch,    m_StageResources.cull.shadowLeafDispatchArgsBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_ShadowImpostorCount,       m_StageResources.cull.shadowImpostorCountBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_ShadowImpostorVis,         m_StageResources.cull.shadowImpostorVisBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_ShadowImpostorIndirectArgs,       m_StageResources.cull.shadowImpostorIndirectArgsBuffer),

        nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Cull::kSRV_HiZ, m_StageResources.hiz.hizTexture),
    };
    m_StageResources.cull.bindingSet = GetDevice()->createBindingSet(bsd, m_StageResources.cull.bindingLayout);
}

void MeshShaderRenderPass::_RebuildDrawBindingSet() {
    const auto& barkTextures = m_Shared->barkTextures();
    if (barkTextures.empty())                       return;
    if (!m_StageResources.shadow.depthTexture)      return;
    if (!m_StageResources.hiz.hizTexture)           return;

    m_StageResources.sceneDraw.bindingSets.resize(barkTextures.size());

    for (size_t texIdx = 0; texIdx < barkTextures.size(); texIdx++) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::PushConstants(mesh_reg::Draw::kPushC_Slot, mesh_reg::Draw::kPushCBytes),
            nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Draw::kCB_Frame, m_StageResources.frameShared.constantBuffer,
                nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
            nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Draw::kCB_ASCull, m_StageResources.frameShared.asCullCB,
                nvrhi::BufferRange(0, shader_cb::kMeshASCullSize)),

            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Positions,     m_StageResources.sceneMeshletData.positions),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Normals,       m_StageResources.sceneMeshletData.normals),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Tangents,      m_StageResources.sceneMeshletData.tangents),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Bitangents,    m_StageResources.sceneMeshletData.bitangents),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_UVs,           m_StageResources.sceneMeshletData.uvs),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_MeshletVertIdx,      m_StageResources.sceneMeshletData.meshletVertIdx),
            nvrhi::BindingSetItem::RawBuffer_SRV(mesh_reg::Draw::kSRV_MeshletPrimIdx,             m_StageResources.sceneMeshletData.meshletPrimIdx),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Meshlets,      m_StageResources.sceneMeshletData.meshletDescs),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_AssetLods,     m_StageResources.sceneMeshletData.assetLodRanges),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Vis,        m_StageResources.cull.mainVisBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_SlotOffsets,   m_StageResources.cull.mainSlotOffsetBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_SlotCounts,    m_StageResources.cull.mainCountBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Instances,     m_StageResources.cull.persistentInstBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_ASInvocations, m_StageResources.cull.mainASInvocsPerSlotBuffer),

            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Draw::kTex_Diffuse,   barkTextures[texIdx].diffuse),
            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Draw::kTex_NormalMap, barkTextures[texIdx].normalMap),
            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Draw::kTex_ShadowMap, m_StageResources.shadow.depthTexture),
            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Draw::kTex_HiZ,    m_StageResources.hiz.hizTexture),

            nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Draw::kUAV_MeshletStats, m_StageResources.cull.meshletStatsBuffer),

            nvrhi::BindingSetItem::Sampler(mesh_reg::Draw::kSampler_Main,   m_Shared->barkSampler()),
            nvrhi::BindingSetItem::Sampler(mesh_reg::Draw::kSampler_Shadow, m_StageResources.sceneDraw.shadowSampler),
        };
        m_StageResources.sceneDraw.bindingSets[texIdx] = GetDevice()->createBindingSet(bsd, m_StageResources.sceneDraw.bindingLayout);
    }
}

void MeshShaderRenderPass::_RebuildShadowBindingSet() {
    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::PushConstants(mesh_reg::Draw::kPushC_Slot, mesh_reg::Draw::kPushCBytes),
        nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Draw::kCB_Frame, m_StageResources.frameShared.constantBuffer,
            nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
        nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Draw::kCB_ASCull, m_StageResources.frameShared.asCullCB,
            nvrhi::BufferRange(0, shader_cb::kMeshASCullSize)),

        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Positions,     m_StageResources.sceneMeshletData.positions),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Normals,       m_StageResources.sceneMeshletData.normals),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Tangents,      m_StageResources.sceneMeshletData.tangents),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Bitangents,    m_StageResources.sceneMeshletData.bitangents),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_UVs,           m_StageResources.sceneMeshletData.uvs),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_MeshletVertIdx,      m_StageResources.sceneMeshletData.meshletVertIdx),
        nvrhi::BindingSetItem::RawBuffer_SRV(mesh_reg::Draw::kSRV_MeshletPrimIdx,             m_StageResources.sceneMeshletData.meshletPrimIdx),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Meshlets,      m_StageResources.sceneMeshletData.meshletDescs),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_AssetLods,     m_StageResources.sceneMeshletData.assetLodRanges),
        // Shadow-specific redirect:
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Vis,          m_StageResources.cull.shadowVisBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_SlotOffsets,     m_StageResources.cull.shadowSlotOffsetBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_SlotCounts,      m_StageResources.cull.shadowCountBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Instances,       m_StageResources.cull.persistentInstBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_ASInvocations, m_StageResources.cull.shadowASInvocsPerSlotBuffer),

        nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Draw::kUAV_MeshletStats, m_StageResources.cull.meshletStatsBuffer),
    };
    m_StageResources.shadow.bindingSet = GetDevice()->createBindingSet(bsd, m_StageResources.shadow.bindingLayout);
}

void MeshShaderRenderPass::_RebuildLeafBindingSets() {
    auto& L = m_StageResources.sceneLeaves;
    if (!L.mainLayout || !L.depthLayout || !L.shadowLayout) return;
    if (!m_Shared || !m_Shared->leafInstancesBuffer()
        || !m_Shared->leafSlotsBuffer() || !m_Shared->leafShadowSlotsBuffer()
        || !m_Shared->leafMeshletsBuffer()) return;

    // Main color leaves: trunk-cull main vis/count + leaf instance buffers + Hi-Z
    // for AS-side meshlet cull. Layout matches MeshLeaves.hlsl:
    //   b0=CB, b1=PushC, b2=ASCullCB, t0..t7 leaf SRVs,
    //   t8=ShadowMap, t9=Hi-Z, s0=ShadowSampler.
    {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Leaf::kCB_Frame, m_StageResources.frameShared.constantBuffer,
                nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
            nvrhi::BindingSetItem::PushConstants(mesh_reg::Leaf::kPushC_Slot, sizeof(uint32_t)),
            nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Leaf::kCB_ASCull, m_StageResources.frameShared.asCullCB,
                nvrhi::BufferRange(0, shader_cb::kMeshASCullSize)),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_Vis,            m_StageResources.cull.mainVisBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_SlotOffsets,    m_StageResources.cull.mainSlotOffsetBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_SlotCounts,     m_StageResources.cull.mainCountBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_Instances,      m_StageResources.cull.persistentInstBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_ASInvocations,  m_StageResources.cull.mainLeafASInvocsPerSlotBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_LeafInstances,  m_Shared->leafInstancesBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_LeafSlots,      m_Shared->leafSlotsBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_LeafMeshlets,   m_Shared->leafMeshletsBuffer()),
            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Leaf::kTex_ShadowMap,               m_StageResources.shadow.depthTexture),
            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Leaf::kTex_HiZ,                     m_StageResources.hiz.hizTexture),
            nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Leaf::kUAV_LeafSurvivor,          m_StageResources.cull.mainLeafSurvivorCounter),
            nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Leaf::kUAV_MeshletStats,          m_StageResources.cull.meshletStatsBuffer),
            nvrhi::BindingSetItem::Sampler(mesh_reg::Leaf::kSampler_Shadow,                  m_StageResources.sceneDraw.shadowSampler),
        };
        L.mainBindingSet = GetDevice()->createBindingSet(bsd, L.mainLayout);
    }

    // Depth prepass leaves: cull-side state + Hi-Z, no shadow tex/sampler.
    {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Leaf::kCB_Frame, m_StageResources.frameShared.constantBuffer,
                nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
            nvrhi::BindingSetItem::PushConstants(mesh_reg::Leaf::kPushC_Slot, sizeof(uint32_t)),
            nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Leaf::kCB_ASCull, m_StageResources.frameShared.asCullCB,
                nvrhi::BufferRange(0, shader_cb::kMeshASCullSize)),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_Vis,            m_StageResources.cull.mainVisBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_SlotOffsets,    m_StageResources.cull.mainSlotOffsetBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_SlotCounts,     m_StageResources.cull.mainCountBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_Instances,      m_StageResources.cull.persistentInstBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_ASInvocations,  m_StageResources.cull.mainLeafASInvocsPerSlotBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_LeafInstances,  m_Shared->leafInstancesBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_LeafSlots,      m_Shared->leafSlotsBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_LeafMeshlets,   m_Shared->leafMeshletsBuffer()),
            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Leaf::kTex_HiZ,                     m_StageResources.hiz.hizTexture),
            nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Leaf::kUAV_LeafSurvivor,          m_StageResources.cull.depthLeafSurvivorScratch),
            nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Leaf::kUAV_MeshletStats,          m_StageResources.cull.meshletStatsDepthScratch),
        };
        L.depthBindingSet = GetDevice()->createBindingSet(bsd, L.depthLayout);
    }

    // Shadow leaves: redirect to shadow-side cull buffers + shadow leaf slots.
    {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Leaf::kCB_Frame, m_StageResources.frameShared.constantBuffer,
                nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
            nvrhi::BindingSetItem::PushConstants(mesh_reg::Leaf::kPushC_Slot, sizeof(uint32_t)),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_Vis,            m_StageResources.cull.shadowVisBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_SlotOffsets,    m_StageResources.cull.shadowSlotOffsetBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_SlotCounts,     m_StageResources.cull.shadowCountBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_Instances,      m_StageResources.cull.persistentInstBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_ASInvocations,  m_StageResources.cull.shadowLeafASInvocsPerSlotBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_LeafInstances,  m_Shared->leafInstancesBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_LeafSlots,      m_Shared->leafShadowSlotsBuffer()),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Leaf::kSRV_LeafMeshlets,   m_Shared->leafMeshletsBuffer()),
            nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Leaf::kUAV_LeafSurvivor,          m_StageResources.cull.shadowLeafSurvivorCounter),
            nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Leaf::kUAV_MeshletStats,          m_StageResources.cull.meshletStatsBuffer),
        };
        L.shadowBindingSet = GetDevice()->createBindingSet(bsd, L.shadowLayout);
    }
}

// ===========================================================================
// Hot-reload callbacks
// ===========================================================================

void MeshShaderRenderPass::onAssetsDirty(const std::vector<size_t>&) {
    auto cl = GetDevice()->createCommandList();
    cl->open();

    _RebuildMeshletMegabuffers(cl);
    _UploadMeshletMegabuffers(cl);
    _BuildRegionWindows();
    _BuildSlotLayout();
    _UploadCullBuffers(cl);
    // Shared impostor atlas is rebaked by RenderOrchestrator before this fires.

    cl->close();
    GetDevice()->executeCommandList(cl);

    _RebuildCullBindingSet();
    _RebuildDrawBindingSet();
    _RebuildShadowBindingSet();
    _RebuildDepthPrepassBindingSet();
    _RebuildLeafBindingSets();
    _RebuildImpostorBindingSets();
    _RebuildShadowImpostorBindingSets();
}

void MeshShaderRenderPass::onRegionsDirty(const std::vector<size_t>&) {
    auto cl = GetDevice()->createCommandList();
    cl->open();

    _BuildRegionWindows();
    _BuildSlotLayout();
    _UploadCullBuffers(cl);

    cl->close();
    GetDevice()->executeCommandList(cl);

    _RebuildCullBindingSet();
    _RebuildDrawBindingSet();
    _RebuildShadowBindingSet();
    _RebuildDepthPrepassBindingSet();
    _RebuildLeafBindingSets();
    _RebuildImpostorBindingSets();
    _RebuildShadowImpostorBindingSets();
}

// ===========================================================================
// Pipelines + command signatures
// ===========================================================================

void MeshShaderRenderPass::_CreateMainPipelineIfNeeded(nvrhi::IFramebuffer* framebuffer) {
    if (m_StageResources.sceneDraw.pipeline) return;

    nvrhi::MeshletPipelineDesc psoDesc;
    psoDesc.AS = m_StageResources.sceneDraw.amplificationShader;
    psoDesc.MS = m_StageResources.sceneDraw.meshShader;
    psoDesc.PS = m_StageResources.sceneDraw.pixelShader;
    psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
    psoDesc.bindingLayouts = { m_StageResources.sceneDraw.bindingLayout };

    auto& rs = psoDesc.renderState;
    rs.depthStencilState.depthTestEnable  = true;
    rs.depthStencilState.depthWriteEnable = true;
#if XYLEM_USE_REVERSE_Z
    rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::Greater;
#else
    rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::Less;
#endif
    rs.rasterState.cullMode = nvrhi::RasterCullMode::Back;

    m_StageResources.sceneDraw.pipeline = GetDevice()->createMeshletPipeline(psoDesc, framebuffer->getFramebufferInfo());
}

void MeshShaderRenderPass::_CreateShadowPipelineIfNeeded() {
    if (m_StageResources.shadow.pipeline)            return;
    if (!m_StageResources.shadow.framebuffers[0])    return;

    nvrhi::MeshletPipelineDesc psoDesc;
    psoDesc.AS = m_StageResources.shadow.amplificationShader;
    psoDesc.MS = m_StageResources.shadow.meshShader;
    psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
    psoDesc.bindingLayouts = { m_StageResources.shadow.bindingLayout };

    auto& rs = psoDesc.renderState;
    rs.depthStencilState.depthTestEnable  = true;
    rs.depthStencilState.depthWriteEnable = true;
    // Light-space shadow maps use the standard D3D depth range/projection built by
    // ViewHandler::computeCascades, independent of the main camera's reverse-Z mode.
    rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::Less;
    rs.rasterState.cullMode = nvrhi::RasterCullMode::Back;
    rs.rasterState.depthBias = 2;
    rs.rasterState.slopeScaledDepthBias = 2.5f;

    m_StageResources.shadow.pipeline = GetDevice()->createMeshletPipeline(
        psoDesc, m_StageResources.shadow.framebuffers[0]->getFramebufferInfo());
}

void MeshShaderRenderPass::_CreateShadowTerrainPipelineIfNeeded() {
    if (m_StageResources.shadow.terrainPipeline)            return;
    if (!m_StageResources.shadow.terrainAS)                 return;
    if (!m_StageResources.shadow.terrainMS)                 return;
    if (!m_StageResources.shadow.terrainBindingLayout)      return;
    if (!m_StageResources.shadow.framebuffers[0])           return;
    if (m_StageResources.sceneTerrain.meshletCount == 0)    return;

    nvrhi::MeshletPipelineDesc pso;
    pso.AS             = m_StageResources.shadow.terrainAS;
    pso.MS             = m_StageResources.shadow.terrainMS;
    pso.bindingLayouts = { m_StageResources.shadow.terrainBindingLayout };
    pso.primType       = nvrhi::PrimitiveType::TriangleList;
    // Light-space shadow framebuffers are STANDARD Z (clear=1.0, Less) regardless of
    // the main camera's reverse-Z mode — do not branch on XYLEM_USE_REVERSE_Z here.
    pso.renderState.depthStencilState.depthTestEnable  = true;
    pso.renderState.depthStencilState.depthWriteEnable = true;
    pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    pso.renderState.rasterState.setCullBack();
    pso.renderState.rasterState.depthBias            = 2;
    pso.renderState.rasterState.slopeScaledDepthBias = 2.0f;

    m_StageResources.shadow.terrainPipeline = GetDevice()->createMeshletPipeline(
        pso, m_StageResources.shadow.framebuffers[0]->getFramebufferInfo());
}

void MeshShaderRenderPass::_EnsureDispatchMeshSignatures() {
    ID3D12Device* d3dDevice = GetDevice()->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
    if (!d3dDevice) return;

    auto makeSig = [&](nvrhi::MeshletPipelineHandle pipe,
                       nvrhi::RefCountPtr<ID3D12CommandSignature>& out,
                       const char* dbgName) {
        if (out || !pipe) return;
        ID3D12RootSignature* rootSig = pipe->getNativeObject(nvrhi::ObjectTypes::D3D12_RootSignature);
        if (!rootSig) return;

        D3D12_INDIRECT_ARGUMENT_DESC args[2] = {};
        args[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
        args[0].Constant.RootParameterIndex      = mesh_reg::kPushC_RootParamIdx;
        args[0].Constant.DestOffsetIn32BitValues = 0;
        args[0].Constant.Num32BitValuesToSet     = 1;
        args[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;

        D3D12_COMMAND_SIGNATURE_DESC desc = {};
        desc.ByteStride       = sizeof(DispatchRecord);
        desc.NumArgumentDescs = 2;
        desc.pArgumentDescs   = args;

        HRESULT hr = d3dDevice->CreateCommandSignature(&desc, rootSig, IID_PPV_ARGS(&out));
        if (FAILED(hr)) {
            log::error("MeshShaderRenderPass: CreateCommandSignature(%s) failed (0x%08x)", dbgName, hr);
            out = nullptr;
        }
    };

    makeSig(m_StageResources.sceneDraw.pipeline,   m_DispatchMeshSignature,         "main");
    makeSig(m_StageResources.shadow.pipeline, m_StageResources.shadow.dispatchMeshSignature,  "shadow");
}

void MeshShaderRenderPass::_CreateLeafPipelinesIfNeeded(nvrhi::IFramebuffer* framebuffer) {
    auto& L = m_StageResources.sceneLeaves;

    if (!L.mainPipeline && L.mainLayout && L.amplificationShader && L.meshShader && L.pixelShader) {
        nvrhi::MeshletPipelineDesc psoDesc;
        psoDesc.AS = L.amplificationShader;
        psoDesc.MS = L.meshShader;
        psoDesc.PS = L.pixelShader;
        psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
        psoDesc.bindingLayouts = { L.mainLayout };
        auto& rs = psoDesc.renderState;
        rs.depthStencilState.depthTestEnable  = true;
        rs.depthStencilState.depthWriteEnable = true;
    #if XYLEM_USE_REVERSE_Z
        rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::Greater;
    #else
        rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::Less;
    #endif
        rs.rasterState.cullMode = nvrhi::RasterCullMode::None;
        L.mainPipeline = GetDevice()->createMeshletPipeline(psoDesc, framebuffer->getFramebufferInfo());
    }

    if (!L.depthPipeline && L.depthLayout && L.amplificationShader && L.depthMS
        && m_StageResources.depthPrepass.framebuffer) {
        nvrhi::MeshletPipelineDesc psoDesc;
        psoDesc.AS = L.amplificationShader;
        psoDesc.MS = L.depthMS;
        psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
        psoDesc.bindingLayouts = { L.depthLayout };
        auto& rs = psoDesc.renderState;
        rs.depthStencilState.depthTestEnable  = true;
        rs.depthStencilState.depthWriteEnable = true;
    #if XYLEM_USE_REVERSE_Z
        rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::GreaterOrEqual;
    #else
        rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::LessOrEqual;
    #endif
        rs.rasterState.cullMode = nvrhi::RasterCullMode::None;
        L.depthPipeline = GetDevice()->createMeshletPipeline(
            psoDesc, m_StageResources.depthPrepass.framebuffer->getFramebufferInfo());
    }

    if (!L.shadowPipeline && L.shadowLayout && L.shadowAmplificationShader && L.shadowMS
        && m_StageResources.shadow.framebuffers[0]) {
        nvrhi::MeshletPipelineDesc psoDesc;
        psoDesc.AS = L.shadowAmplificationShader;
        psoDesc.MS = L.shadowMS;
        psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
        psoDesc.bindingLayouts = { L.shadowLayout };
        auto& rs = psoDesc.renderState;
        rs.depthStencilState.depthTestEnable  = true;
        rs.depthStencilState.depthWriteEnable = true;
        rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::Less;
        rs.rasterState.cullMode = nvrhi::RasterCullMode::None;
        rs.rasterState.depthBias = 2;
        rs.rasterState.slopeScaledDepthBias = 2.5f;
        L.shadowPipeline = GetDevice()->createMeshletPipeline(
            psoDesc, m_StageResources.shadow.framebuffers[0]->getFramebufferInfo());
    }
}

void MeshShaderRenderPass::_EnsureLeafDispatchMeshSignatures() {
    ID3D12Device* d3dDevice = GetDevice()->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
    if (!d3dDevice) return;

    auto makeSig = [&](nvrhi::MeshletPipelineHandle pipe,
                       nvrhi::RefCountPtr<ID3D12CommandSignature>& out,
                       const char* dbgName) {
        if (out || !pipe) return;
        ID3D12RootSignature* rootSig = pipe->getNativeObject(nvrhi::ObjectTypes::D3D12_RootSignature);
        if (!rootSig) return;

        D3D12_INDIRECT_ARGUMENT_DESC args[2] = {};
        args[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
        args[0].Constant.RootParameterIndex      = mesh_reg::kPushC_RootParamIdx;
        args[0].Constant.DestOffsetIn32BitValues = 0;
        args[0].Constant.Num32BitValuesToSet     = 1;
        args[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;

        D3D12_COMMAND_SIGNATURE_DESC desc = {};
        desc.ByteStride       = sizeof(DispatchRecord);
        desc.NumArgumentDescs = 2;
        desc.pArgumentDescs   = args;

        HRESULT hr = d3dDevice->CreateCommandSignature(&desc, rootSig, IID_PPV_ARGS(&out));
        if (FAILED(hr)) {
            log::error("MeshShaderRenderPass: CreateCommandSignature(leaf %s) failed (0x%08x)", dbgName, hr);
            out = nullptr;
        }
    };

    auto& L = m_StageResources.sceneLeaves;
    makeSig(L.mainPipeline,   L.mainSignature,   "main");
    makeSig(L.depthPipeline,  L.depthSignature,  "depth");
    makeSig(L.shadowPipeline, L.shadowSignature, "shadow");
}

// ===========================================================================
// IRenderPass overrides
// ===========================================================================

void MeshShaderRenderPass::Animate(float /*seconds*/) {
    GetDeviceManager()->SetInformativeWindowTitle("Xylem (MeshShader)");
    // Dirty cycle is driven by RenderOrchestrator (so SharedGPUAssets re-bakes
    // the impostor atlas before this pass rebuilds binding sets that reference
    // the atlas).
}

void MeshShaderRenderPass::BackBufferResizing() {
    m_StageResources.sceneDraw.pipeline          = nullptr;
    m_StageResources.shadow.pipeline        = nullptr;
    m_StageResources.shadow.terrainPipeline = nullptr;
    m_DispatchMeshSignature  = nullptr;
    m_StageResources.shadow.dispatchMeshSignature = nullptr;

    m_StageResources.sceneLeaves.mainPipeline    = nullptr;
    m_StageResources.sceneLeaves.depthPipeline   = nullptr;
    m_StageResources.sceneLeaves.shadowPipeline  = nullptr;
    m_StageResources.sceneLeaves.mainSignature   = nullptr;
    m_StageResources.sceneLeaves.depthSignature  = nullptr;
    m_StageResources.sceneLeaves.shadowSignature = nullptr;

    m_StageResources.depthPrepass.pipeline        = nullptr;
    m_StageResources.depthPrepass.terrainPipeline = nullptr;
    m_StageResources.depthPrepass.depthTexture    = nullptr;
    m_StageResources.depthPrepass.framebuffer     = nullptr;
    m_StageResources.depthPrepass.bindingSet      = nullptr;
    m_StageResources.depthPrepass.dispatchMeshSignature = nullptr;

    m_StageResources.sceneTerrain.pipeline = nullptr;
    m_StageResources.sky.pipeline     = nullptr;

    m_StageResources.hiz.buildBindingSets.clear();
    m_StageResources.hiz.debugMipTextures.clear();
    m_StageResources.hiz.numMips = 0;
    m_UI.hizMipTextures.clear();
}

void MeshShaderRenderPass::Render(nvrhi::IFramebuffer* framebuffer) {
    app::HiResTimer cpuTimer;
    cpuTimer.Start();

    m_UI.shadowMapTexture       = m_StageResources.shadow.depthTexture.Get();
    m_UI.selectedCascadeTexture = m_StageResources.shadow.debugSelectedCascadeTexture.Get();
    m_UI.hizMipTextures.resize(m_StageResources.hiz.debugMipTextures.size());
    for (size_t i = 0; i < m_StageResources.hiz.debugMipTextures.size(); i++)
        m_UI.hizMipTextures[i] = m_StageResources.hiz.debugMipTextures[i].Get();

    const auto& fbInfo = framebuffer->getFramebufferInfo();
    {
        if (!m_StageResources.sceneDraw.pipeline) {
            frame::UpdateProjectionAndViewport(m_ViewHandler, fbInfo);
        }

        m_ViewHandler.view.SetViewMatrix(m_ViewHandler.camera.GetWorldToViewMatrix());
        m_ViewHandler.view.UpdateCache();
    }

    const uint32_t fbW = fbInfo.width;
    const uint32_t fbH = fbInfo.height;
    const float aspectRatio = m_ViewHandler.view.GetAspectRatio();

    _CreateMainPipelineIfNeeded(framebuffer);
    _CreateShadowPipelineIfNeeded();
    _CreateShadowTerrainPipelineIfNeeded();
    _CreateLeafPipelinesIfNeeded(framebuffer);
    _EnsureDispatchMeshSignatures();
    _EnsureLeafDispatchMeshSignatures();

    const dm::float3& camPos = m_ViewHandler.camera.GetPosition();
    const dm::float3& camDir = m_ViewHandler.camera.GetDir();
    frame::ComputeCascades(
        m_ViewHandler,
        m_Registry,
        aspectRatio,
        k_ShadowRes,
        m_UI.pssmLambda);

    bool hizActive = m_UI.hizEnabled;

    m_CommandList->open();
    frame::ResetGpuTimerForFrame(GetDevice(), m_GpuTimers, m_NextTimerIdx);
    m_CommandList->beginTimerQuery(m_GpuTimers[m_NextTimerIdx]);

    // Make sure Hi-Z / depth-prepass resources match the back-buffer resolution.
    _EnsureHiZResources(fbW, fbH);

    // ----- 1. Fill CullConstantBufferEntry (CPU cascades seed; SDSM may overwrite) -----
    Render::CullConstantBufferEntry cb = {};
    frame::FillCommonFrameConstants(cb, m_ViewHandler, m_Registry.getSunSkyState());

    for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
        const dm::box3& cBbox = m_ViewHandler.cascades[c].shadowCasterBboxLS;
        if (cBbox.isempty()) {
            cb.shadowCasterMinLS[c] = dm::float4( 1e30f,  1e30f,  1e30f, 0.f);
            cb.shadowCasterMaxLS[c] = dm::float4(-1e30f, -1e30f, -1e30f, 0.f);
        } else {
            cb.shadowCasterMinLS[c] = dm::float4(cBbox.m_mins, 0.f);
            cb.shadowCasterMaxLS[c] = dm::float4(cBbox.m_maxs, 0.f);
        }
    }

    cb.viewFrustum   = m_ViewHandler.view.GetViewFrustum();
    cb.worldToLight  = dm::affineToHomogeneous(m_ViewHandler.worldToLight);
    cb.cameraPos     = camPos;
    cb.numRegions    = static_cast<uint32_t>(m_Registry.getRegions().size());
    cb.totalCapacity = m_TotalCapacity;

    const auto& lodDistances = m_Registry.getLodDistances();
    cb.numLods = static_cast<uint32_t>(m_Registry.getLodSegments().size());
    for (uint32_t i = 0; i < cb.numLods; i++)
        cb.lodDistances[i] = lodDistances[i];

    cb.hizDimensions = dm::float2(static_cast<float>(fbW), static_cast<float>(fbH));
    cb.maxHiZMip     = static_cast<float>((m_StageResources.hiz.numMips > 0) ? (m_StageResources.hiz.numMips - 1) : 0);
    cb.hizEnabled    = hizActive ? 1u : 0u;
    cb.impostorAlphaClip    = XYLEM_IMPOSTOR_ALPHA_CLIP;
    cb.showShadowImpostors  = m_UI.showShadowImpostors ? 1u : 0u;
    cb.shadowImpostorBias   = XYLEM_SHADOW_IMPOSTOR_BIAS;

    m_CommandList->writeBuffer(m_StageResources.frameShared.constantBuffer, &cb, shader_cb::kCullFrameSize);

    // ----- 1b. Fill AS-cull constants - hizEnabled=0 for the depth prepass (Hi-Z not yet built) -----
    shader_cb::MeshASCullConstants asCB = {};
    asCB.cameraPos         = camPos;
    asCB.hizEnabled        = 0u;
    asCB.hizDimensions     = dm::float2(static_cast<float>(fbW), static_cast<float>(fbH));
    asCB.maxHiZMip         = static_cast<float>((m_StageResources.hiz.numMips > 0) ? (m_StageResources.hiz.numMips - 1) : 0);
    asCB.asConeCullEnabled = 1u;
    m_CommandList->writeBuffer(m_StageResources.frameShared.asCullCB, &asCB, sizeof(shader_cb::MeshASCullConstants));

    // ----- 2. Depth prepass + Hi-Z build + SDSM (skip when bypassed) -----
    // Reads LAST frame's m_StageResources.cull.mainDispatchArgsBuffer - must happen BEFORE we
    // reset/clear the cull output buffers below.
    if (hizActive) {
        m_CommandList->beginMarker("HiZ");

        m_CommandList->beginMarker("DepthPrepass");
        frame::BeginGpuStage(m_CommandList, GetDevice(),
                             m_StageTimers[static_cast<size_t>(frame::FrameStage::DepthPrepass)],
                             m_StageNextIdx[static_cast<size_t>(frame::FrameStage::DepthPrepass)]);
        _RenderDepthPrepass();
        frame::EndGpuStage(m_CommandList,
                           m_StageTimers[static_cast<size_t>(frame::FrameStage::DepthPrepass)],
                           m_StageNextIdx[static_cast<size_t>(frame::FrameStage::DepthPrepass)]);
        m_CommandList->endMarker();

        m_CommandList->beginMarker("BuildMipChain");
        frame::BeginGpuStage(m_CommandList, GetDevice(),
                             m_StageTimers[static_cast<size_t>(frame::FrameStage::HiZ)],
                             m_StageNextIdx[static_cast<size_t>(frame::FrameStage::HiZ)]);
        _BuildHiZMipChain();
        frame::EndGpuStage(m_CommandList,
                           m_StageTimers[static_cast<size_t>(frame::FrameStage::HiZ)],
                           m_StageNextIdx[static_cast<size_t>(frame::FrameStage::HiZ)]);
        m_CommandList->endMarker();

        m_CommandList->beginMarker("SDSM");
        frame::BeginGpuStage(m_CommandList, GetDevice(),
                             m_StageTimers[static_cast<size_t>(frame::FrameStage::SDSM)],
                             m_StageNextIdx[static_cast<size_t>(frame::FrameStage::SDSM)]);
        float regionNear, regionFar;
        _ComputeRegionEnvelope(camPos, camDir, regionNear, regionFar);
        _RunSDSMBuildCascades(m_Registry.getSceneBounds(), aspectRatio, dm::radians(60.f),
                              regionNear, regionFar);
        frame::EndGpuStage(m_CommandList,
                           m_StageTimers[static_cast<size_t>(frame::FrameStage::SDSM)],
                           m_StageNextIdx[static_cast<size_t>(frame::FrameStage::SDSM)]);
        m_CommandList->endMarker();

        m_CommandList->endMarker();
    }

    // Re-write ASCullCB now that Hi-Z is populated - main_as uses it during color pass.
    asCB.hizEnabled = hizActive ? 1u : 0u;
    m_CommandList->writeBuffer(m_StageResources.frameShared.asCullCB, &asCB, sizeof(shader_cb::MeshASCullConstants));

    // ----- 3. Per-frame cull resets -----
    m_CommandList->clearBufferUInt(m_StageResources.cull.mainCountBuffer,     0);
    m_CommandList->clearBufferUInt(m_StageResources.cull.shadowCountBuffer,   0);
    m_CommandList->clearBufferUInt(m_StageResources.cull.shadowUniqueCounter, 0);
    m_CommandList->clearBufferUInt(m_StageResources.cull.impostorCountBuffer, 0);
    m_CommandList->clearBufferUInt(m_StageResources.cull.shadowImpostorCountBuffer, 0);
    // Leaf survivor counters are cleared here too. The depth-prepass scratch
    // counter ran first this frame using last-frame's indirect args; clearing
    // it here keeps it bounded but its contents are never read.
    m_CommandList->clearBufferUInt(m_StageResources.cull.mainLeafSurvivorCounter,   0);
    m_CommandList->clearBufferUInt(m_StageResources.cull.shadowLeafSurvivorCounter, 0);
    m_CommandList->clearBufferUInt(m_StageResources.cull.depthLeafSurvivorScratch,  0);
    m_CommandList->clearBufferUInt(m_StageResources.cull.meshletStatsBuffer,        0);
    m_CommandList->clearBufferUInt(m_StageResources.cull.meshletStatsDepthScratch,  0);

    m_CommandList->writeBuffer(m_StageResources.cull.mainDispatchArgsBuffer,
        m_MainDispatchArgsStaging.data(),
        m_MainDispatchArgsStaging.size() * sizeof(DispatchRecord));
    m_CommandList->writeBuffer(m_StageResources.cull.shadowDispatchArgsBuffer,
        m_ShadowDispatchArgsStaging.data(),
        m_ShadowDispatchArgsStaging.size() * sizeof(DispatchRecord));
    m_CommandList->writeBuffer(m_StageResources.cull.mainLeafDispatchArgsBuffer,
        m_MainLeafDispatchArgsStaging.data(),
        m_MainLeafDispatchArgsStaging.size() * sizeof(DispatchRecord));
    m_CommandList->writeBuffer(m_StageResources.cull.shadowLeafDispatchArgsBuffer,
        m_ShadowLeafDispatchArgsStaging.data(),
        m_ShadowLeafDispatchArgsStaging.size() * sizeof(DispatchRecord));

    // Impostor indirect args reset: vertexCount=4, instanceCount=0 per asset.
    {
        const uint32_t numAssets = std::max(1u,
            static_cast<uint32_t>(m_Registry.getAssets().size()));
        std::vector<nvrhi::DrawIndirectArguments> impostorArgs(numAssets);
        for (uint32_t ai = 0; ai < numAssets; ai++) {
            auto& a = impostorArgs[ai];
            a.vertexCount           = 4;
            a.instanceCount         = 0;
            a.startVertexLocation   = 0;
            a.startInstanceLocation = 0;
        }
        m_CommandList->writeBuffer(m_StageResources.cull.impostorIndirectArgsBuffer,
            impostorArgs.data(), numAssets * sizeof(nvrhi::DrawIndirectArguments));
    }

    // Shadow impostor indirect args reset: vertexCount=4, instanceCount=0 per (asset, cascade)
    {
        const uint32_t numSlots = static_cast<uint32_t>(m_ShadowImpostorSlotOffsets.size());
        if (numSlots > 0) {
            std::vector<nvrhi::DrawIndirectArguments> args(numSlots);
            for (uint32_t i = 0; i < numSlots; i++) {
                auto& a = args[i];
                a.vertexCount   = 4;
                a.instanceCount = 0;
                a.startVertexLocation   = 0;
                a.startInstanceLocation = 0;
            }
            m_CommandList->writeBuffer(m_StageResources.cull.shadowImpostorIndirectArgsBuffer,
                args.data(), numSlots * sizeof(nvrhi::DrawIndirectArguments));
        }
    }

    // ----- 4. GPU cull dispatches -----
    m_CommandList->beginMarker("Cull");
    frame::BeginGpuStage(m_CommandList, GetDevice(),
                         m_StageTimers[static_cast<size_t>(frame::FrameStage::Cull)],
                         m_StageNextIdx[static_cast<size_t>(frame::FrameStage::Cull)]);
    {
        nvrhi::ComputeState cs;
        cs.bindings = { m_StageResources.cull.bindingSet };

        m_CommandList->beginMarker("RegionDispatch");
        cs.pipeline = m_StageResources.cull.regionPipeline;
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch((cb.numRegions + 63) / 64, 1, 1);
        m_CommandList->endMarker();

        m_CommandList->beginMarker("MainDispatch");
        cs.pipeline = m_StageResources.cull.mainPipeline;
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch((m_TotalCapacity + 255) / 256, 1, 1);
        m_CommandList->endMarker();

        m_CommandList->beginMarker("ShadowDispatch");
        cs.pipeline = m_StageResources.cull.shadowPipeline;
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch((m_TotalCapacity + 255) / 256, 1, 1);
        m_CommandList->endMarker();
    }
    frame::EndGpuStage(m_CommandList,
                       m_StageTimers[static_cast<size_t>(frame::FrameStage::Cull)],
                       m_StageNextIdx[static_cast<size_t>(frame::FrameStage::Cull)]);
    m_CommandList->endMarker();

    // Readback ring copy: [main counts][shadow counts][impostor counts][shadowImpostor counts][shadow unique]
    {
        uint32_t ringSlot = m_ReadbackFrameIndex % k_QueuedFrames;
        uint64_t offset = 0;
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], offset,
                                  m_StageResources.cull.mainCountBuffer, 0,
                                  m_ReadbackMainEntries * sizeof(uint32_t));
        offset += m_ReadbackMainEntries * sizeof(uint32_t);
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], offset,
                                  m_StageResources.cull.shadowCountBuffer, 0,
                                  m_ReadbackShadowEntries * sizeof(uint32_t));
        offset += m_ReadbackShadowEntries * sizeof(uint32_t);
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], offset,
                                  m_StageResources.cull.impostorCountBuffer, 0,
                                  m_ReadbackImpostorEntries * sizeof(uint32_t));
        offset += m_ReadbackImpostorEntries * sizeof(uint32_t);
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], offset,
                                  m_StageResources.cull.shadowImpostorCountBuffer, 0,
                                  m_ReadbackShadowImpostorEntries * sizeof(uint32_t));
        offset += m_ReadbackShadowImpostorEntries * sizeof(uint32_t);
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], offset,
                                  m_StageResources.cull.shadowUniqueCounter, 0,
                                  sizeof(uint32_t));
    }

    m_CommandList->beginMarker("Draw");

    // ----- 4. Shadow pass: 4 cascades via ExecuteIndirect(DISPATCH_MESH) -----
    if (m_StageResources.shadow.pipeline && m_StageResources.shadow.dispatchMeshSignature) {
        m_CommandList->beginMarker("Shadow");
        frame::BeginGpuStage(m_CommandList, GetDevice(),
                             m_StageTimers[static_cast<size_t>(frame::FrameStage::Shadow)],
                             m_StageNextIdx[static_cast<size_t>(frame::FrameStage::Shadow)]);

        if (m_Registry.getSunSkyState().shadowsEnabled) {
        // Clear all cascades.
        for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
            nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, m_StageResources.shadow.framebuffers[c], 1.f, 0);
        }

        // Shared shadow meshlet state; re-bind the framebuffer per cascade slice and
        // ExecuteIndirect only for that cascade's interleaved asset slots.
        const uint32_t numAssets = static_cast<uint32_t>(m_Registry.getAssets().size());

        // The shadow slot layout is interleaved as slot = asset*NUM_CASCADES + cascade.

        for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
            std::string cascadeMarker = "C" + std::to_string(c);
            m_CommandList->beginMarker(cascadeMarker.c_str());

            // Trunk
            m_CommandList->beginMarker("Trunk");
            nvrhi::MeshletState ms;
            ms.pipeline       = m_StageResources.shadow.pipeline;
            ms.framebuffer    = m_StageResources.shadow.framebuffers[c];
            ms.bindings       = { m_StageResources.shadow.bindingSet };
            ms.indirectParams = m_StageResources.cull.shadowDispatchArgsBuffer;
            ms.viewport.addViewportAndScissorRect(
                nvrhi::Viewport(float(k_ShadowRes), float(k_ShadowRes)));
            m_CommandList->setMeshletState(ms);

            uint32_t placeholder = 0;
            m_CommandList->setPushConstants(&placeholder, sizeof(uint32_t));

            auto* d3dList = static_cast<ID3D12GraphicsCommandList6*>(
                m_CommandList->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList));
            auto* argBuffer = static_cast<ID3D12Resource*>(
                m_StageResources.cull.shadowDispatchArgsBuffer->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource));

            if (d3dList && argBuffer) {
                for (uint32_t ai = 0; ai < numAssets; ai++) {
                    const uint32_t slot = ai * Render::c_NumCascades + c;
                    d3dList->ExecuteIndirect(
                        m_StageResources.shadow.dispatchMeshSignature.Get(),
                        1,
                        argBuffer,
                        slot * sizeof(DispatchRecord),
                        nullptr,
                        0);
                }
            }
            m_CommandList->endMarker(); // Trunk

            // Leaves into the same cascade slice.
            auto& L = m_StageResources.sceneLeaves;
            if (L.shadowPipeline && L.shadowSignature && L.shadowBindingSet
                && m_StageResources.cull.shadowLeafDispatchArgsBuffer) {
                m_CommandList->beginMarker("Leaves");
                nvrhi::MeshletState leafMS;
                leafMS.pipeline       = L.shadowPipeline;
                leafMS.framebuffer    = m_StageResources.shadow.framebuffers[c];
                leafMS.bindings       = { L.shadowBindingSet };
                leafMS.indirectParams = m_StageResources.cull.shadowLeafDispatchArgsBuffer;
                leafMS.viewport.addViewportAndScissorRect(
                    nvrhi::Viewport(float(k_ShadowRes), float(k_ShadowRes)));
                m_CommandList->setMeshletState(leafMS);

                uint32_t placeholder = 0;
                m_CommandList->setPushConstants(&placeholder, sizeof(uint32_t));

                auto* leafArgBuf = static_cast<ID3D12Resource*>(
                    m_StageResources.cull.shadowLeafDispatchArgsBuffer->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource));
                if (d3dList && leafArgBuf) {
                    for (uint32_t ai = 0; ai < numAssets; ai++) {
                        const uint32_t slot = ai * Render::c_NumCascades + c;
                        d3dList->ExecuteIndirect(
                            L.shadowSignature.Get(),
                            1,
                            leafArgBuf,
                            slot * sizeof(DispatchRecord),
                            nullptr,
                            0);
                    }
                }
                m_CommandList->endMarker(); // Leaves
            }

            // Terrain into this cascade slice via AS/MS — the AS culls each
            // terrain meshlet against the cascade's light frustum (extracted in
            // shader from lightViewProj[c], so SDSM overrides apply). Reuses the
            // eye-view terrain meshlet buffers.
            if (m_StageResources.shadow.terrainPipeline
                && m_StageResources.shadow.terrainBindingSet
                && m_StageResources.sceneTerrain.meshletCount > 0) {
                m_CommandList->beginMarker("Terrain");
                nvrhi::MeshletState terrShadow;
                terrShadow.pipeline    = m_StageResources.shadow.terrainPipeline;
                terrShadow.framebuffer = m_StageResources.shadow.framebuffers[c];
                terrShadow.bindings    = { m_StageResources.shadow.terrainBindingSet };
                terrShadow.viewport.addViewportAndScissorRect(
                    nvrhi::Viewport(float(k_ShadowRes), float(k_ShadowRes)));
                m_CommandList->setMeshletState(terrShadow);

                m_CommandList->setPushConstants(&c, sizeof(c));

                const uint32_t numGroups = (m_StageResources.sceneTerrain.meshletCount
                    + Render::k_ASGroupSize - 1) / Render::k_ASGroupSize;
                m_CommandList->dispatchMesh(numGroups, 1, 1);
                m_CommandList->endMarker(); // Terrain
            }

            if (m_UI.showShadowImpostors) {
                m_CommandList->beginMarker("Impostors");
                _RenderShadowImpostorPass(c);
                m_CommandList->endMarker(); // Impostors
            }

            m_CommandList->endMarker(); // C<c>
        }
        } else {
            _ClearShadowMaps();
        }
        frame::EndGpuStage(m_CommandList,
                           m_StageTimers[static_cast<size_t>(frame::FrameStage::Shadow)],
                           m_StageNextIdx[static_cast<size_t>(frame::FrameStage::Shadow)]);
        m_CommandList->endMarker(); // Shadow
    }

    // Copy the currently-selected cascade slice into the single debug texture for UI display.
    if (m_UI.showShadowMap) {
        uint32_t cascade = std::clamp(m_UI.selectedCascade, 0,
                                      static_cast<int>(Render::c_NumCascades) - 1);
        m_CommandList->copyTexture(
            m_StageResources.shadow.debugSelectedCascadeTexture, nvrhi::TextureSlice(),
            m_StageResources.shadow.depthTexture, nvrhi::TextureSlice().setArraySlice(cascade));
    }

    // ----- 5. Clear main framebuffer -----
    nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));
#if XYLEM_USE_REVERSE_Z
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer, 0.f, 0);
#else
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer, 1.f, 0);
#endif

    m_CommandList->beginMarker("Main");

    m_CommandList->beginMarker("Sky");
    frame::BeginGpuStage(m_CommandList, GetDevice(),
                         m_StageTimers[static_cast<size_t>(frame::FrameStage::Sky)],
                         m_StageNextIdx[static_cast<size_t>(frame::FrameStage::Sky)]);
    _RenderSkyPass(framebuffer);
    frame::EndGpuStage(m_CommandList,
                       m_StageTimers[static_cast<size_t>(frame::FrameStage::Sky)],
                       m_StageNextIdx[static_cast<size_t>(frame::FrameStage::Sky)]);
    m_CommandList->endMarker();

    frame::BeginGpuStage(m_CommandList, GetDevice(),
                         m_StageTimers[static_cast<size_t>(frame::FrameStage::Scene)],
                         m_StageNextIdx[static_cast<size_t>(frame::FrameStage::Scene)]);
    m_CommandList->beginMarker("Trunk");
    _RenderTrunkPass(framebuffer);
    m_CommandList->endMarker();

    m_CommandList->beginMarker("Leaves");
    _RenderLeavesPass(framebuffer);
    m_CommandList->endMarker();

    m_CommandList->beginMarker("Terrain");
    _RenderTerrainPass(framebuffer);
    m_CommandList->endMarker();

    m_CommandList->beginMarker("Impostors");
    _RenderImpostorPass(framebuffer);
    m_CommandList->endMarker();
    frame::EndGpuStage(m_CommandList,
                       m_StageTimers[static_cast<size_t>(frame::FrameStage::Scene)],
                       m_StageNextIdx[static_cast<size_t>(frame::FrameStage::Scene)]);

    m_CommandList->endMarker(); // Main

    m_CommandList->endMarker(); // Draw
    
    if (m_UI.showImpostorAtlas && m_Shared)
        m_Shared->CopySelectedImpostorDebugAtlases(m_CommandList, m_UI.impostorSelectedAsset);

    // ----- 6d. Leaf survivor + meshlet stats readback copy. Has to run AFTER the
    // shadow + main color leaf draws so the AS-side atomics are committed. The
    // trunk readback copy at step 4 wrote into
    // [main][shadow][impostor][shadowImpostor][shadowUnique]; the leaf and
    // terrain counters land at [mainLeafSurvivors][shadowLeafSurvivors][terrainSurvivors],
    // followed by the 8-uint meshlet stats block.
    {
        const uint32_t ringSlot = m_ReadbackFrameIndex % k_QueuedFrames;
        const uint64_t leafBaseOffset =
            (m_ReadbackMainEntries + m_ReadbackShadowEntries + m_ReadbackImpostorEntries + m_ReadbackShadowImpostorEntries + 1)
            * sizeof(uint32_t);
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], leafBaseOffset,
                                  m_StageResources.cull.mainLeafSurvivorCounter, 0,
                                  sizeof(uint32_t));
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], leafBaseOffset + sizeof(uint32_t),
                                  m_StageResources.cull.shadowLeafSurvivorCounter, 0,
                                  sizeof(uint32_t));
        if (m_StageResources.sceneTerrain.visibleCounterBuffer) {
            m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], leafBaseOffset + 2 * sizeof(uint32_t),
                                      m_StageResources.sceneTerrain.visibleCounterBuffer, 0,
                                      sizeof(uint32_t));
        }
        const uint64_t meshletStatsOffset = leafBaseOffset + 3 * sizeof(uint32_t);
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], meshletStatsOffset,
                                  m_StageResources.cull.meshletStatsBuffer, 0,
                                  8 * sizeof(uint32_t));
    }

    m_CommandList->endTimerQuery(m_GpuTimers[m_NextTimerIdx]);
    m_CommandList->close();
    GetDevice()->executeCommandList(m_CommandList);
    frame::RotateAndReadGpuTimer(GetDevice(), m_GpuTimers, m_NextTimerIdx, m_UI.gpuFrameTimeMs);

    for (size_t s = 0; s < kStageCount; ++s) {
        frame::RotateAndReadGpuTimer(GetDevice(),
                                     m_StageTimers[s], m_StageNextIdx[s],
                                     m_UI.gpuStageTimeMs[s]);
    }

    // When Hi-Z is toggled off, DepthPrepass / HiZ / SDSM don't emit timer
    // queries this frame — without this, their UI/CSV slots would show stale
    // values from when Hi-Z was last on. Force them to the -1 sentinel so
    // the UI renders "—" and CSV writes -1.
    if (!hizActive) {
        m_UI.gpuStageTimeMs[static_cast<size_t>(frame::FrameStage::DepthPrepass)] = -1.f;
        m_UI.gpuStageTimeMs[static_cast<size_t>(frame::FrameStage::HiZ)]          = -1.f;
        m_UI.gpuStageTimeMs[static_cast<size_t>(frame::FrameStage::SDSM)]         = -1.f;
    }

    // ----- 7. UI stats -----
    m_UI.totalInstanceCount     = static_cast<uint32_t>(m_Registry.totalInstanceCount());
    m_UI.totalLeafInstanceCount = m_Registry.totalLeafInstanceCount();
    m_UI.totalLeafMeshletCount  = m_Registry.totalLeafMeshletCount();
    // Impostor atlas pointers + view-count metadata are published by SharedGPUAssets.

    if (m_ReadbackFrameIndex >= (k_QueuedFrames - 1)) {
        uint32_t readSlot = (m_ReadbackFrameIndex + 1) % k_QueuedFrames;
        void* pData = GetDevice()->mapBuffer(m_ReadbackBuffers[readSlot], nvrhi::CpuAccessMode::Read);
        if (pData) {
            const uint32_t* counts = static_cast<const uint32_t*>(pData);
            uint32_t visSum = 0;
            for (uint32_t i = 0; i < m_ReadbackMainEntries; i++) visSum += counts[i];

            // Per-LOD geometric counts. Slot layout matches the cull shader:
            // slot = ai * numLods + li.
            const uint32_t numLodsForBreakdown =
                static_cast<uint32_t>(m_Registry.getLodSegments().size());
            const uint32_t lodsForUI = std::min<uint32_t>(numLodsForBreakdown, UIData::kMaxLodsForUI);
            uint32_t perLodCounts[UIData::kMaxLodsForUI] = {};
            if (numLodsForBreakdown > 0) {
                for (uint32_t slot = 0; slot < m_ReadbackMainEntries; ++slot) {
                    const uint32_t li = slot % numLodsForBreakdown;
                    if (li < lodsForUI) perLodCounts[li] += counts[slot];
                }
            }

            // Per-cascade shadow geometry sums. Slot layout is
            // ai * Render::c_NumCascades + c (asset major). Total visible
            // is Σ over all (asset, cascade) slots; we also break out per c.
            uint32_t perCascadeShadow[Render::c_NumCascades] = {};
            uint32_t shadowVisSum = 0;
            const uint32_t numShadowAssets =
                Render::c_NumCascades > 0 ? (m_ReadbackShadowEntries / Render::c_NumCascades) : 0;
            for (uint32_t ai = 0; ai < numShadowAssets; ++ai) {
                for (uint32_t c = 0; c < Render::c_NumCascades; ++c) {
                    const uint32_t v = counts[m_ReadbackMainEntries + ai * Render::c_NumCascades + c];
                    perCascadeShadow[c] += v;
                    shadowVisSum        += v;
                }
            }

            uint32_t impostorVisSum = 0;
            const uint32_t impostorOffset = m_ReadbackMainEntries + m_ReadbackShadowEntries;
            for (uint32_t i = 0; i < m_ReadbackImpostorEntries; i++)
                impostorVisSum += counts[impostorOffset + i];

            // Per-cascade shadow impostor sums. Slot layout same as
            // geometry: ai * Render::c_NumCascades + c.
            const uint32_t shadowImpostorOffset = impostorOffset + m_ReadbackImpostorEntries;
            uint32_t perCascadeShadowImpostor[Render::c_NumCascades] = {};
            uint32_t shadowImpostorVisSum = 0;
            const uint32_t numShadowImpostorAssets =
                Render::c_NumCascades > 0 ? (m_ReadbackShadowImpostorEntries / Render::c_NumCascades) : 0;
            for (uint32_t ai = 0; ai < numShadowImpostorAssets; ++ai) {
                for (uint32_t c = 0; c < Render::c_NumCascades; ++c) {
                    const uint32_t v = counts[shadowImpostorOffset + ai * Render::c_NumCascades + c];
                    perCascadeShadowImpostor[c] += v;
                    shadowImpostorVisSum        += v;
                }
            }

            uint32_t shadowUnique       = counts[shadowImpostorOffset + m_ReadbackShadowImpostorEntries];
            uint32_t leafMainSurvivors  = counts[shadowImpostorOffset + m_ReadbackShadowImpostorEntries + 1];
            uint32_t leafShadowSurvivors= counts[shadowImpostorOffset + m_ReadbackShadowImpostorEntries + 2];
            uint32_t terrainSurvivors   = counts[shadowImpostorOffset + m_ReadbackShadowImpostorEntries + 3];

            const uint32_t meshletStatsOffset = shadowImpostorOffset + m_ReadbackShadowImpostorEntries + 4;
            uint32_t trunkMainConsidered   = counts[meshletStatsOffset + 0];
            uint32_t trunkMainSurvived     = counts[meshletStatsOffset + 1];
            uint32_t trunkShadowConsidered = counts[meshletStatsOffset + 2];
            uint32_t trunkShadowSurvived   = counts[meshletStatsOffset + 3];
            uint32_t leafMainConsidered    = counts[meshletStatsOffset + 4];
            uint32_t leafMainSurvived      = counts[meshletStatsOffset + 5];
            uint32_t leafShadowConsidered  = counts[meshletStatsOffset + 6];
            uint32_t leafShadowSurvived    = counts[meshletStatsOffset + 7];

            GetDevice()->unmapBuffer(m_ReadbackBuffers[readSlot]);

            const uint32_t totalVisible = visSum + impostorVisSum;
            m_UI.visibleInstanceCount = totalVisible;
            m_UI.culledInstanceCount  = (totalVisible <= m_UI.totalInstanceCount)
                ? m_UI.totalInstanceCount - totalVisible : 0;
            m_UI.impostorVisibleCount         = impostorVisSum;
            m_UI.shadowImpostorVisibleCount   = shadowImpostorVisSum;
            m_UI.visibleLeafInstanceCount       = leafMainSurvivors;
            m_UI.shadowVisibleLeafInstanceCount = leafShadowSurvivors;
            m_UI.shadowVisibleCount     = shadowUnique;
            m_UI.shadowCulledCount      = (shadowUnique <= m_UI.totalInstanceCount)
                ? m_UI.totalInstanceCount - shadowUnique : 0;
            m_UI.shadowCascadeDrawCount = shadowVisSum;
            m_UI.shadowOverdrawCount    = (shadowVisSum >= shadowUnique)
                ? shadowVisSum - shadowUnique : 0;
            for (uint32_t c = 0; c < Render::c_NumCascades; ++c) {
                m_UI.shadowGeomDrawsPerCascade[c]      = perCascadeShadow[c];
                m_UI.shadowBillboardDrawsPerCascade[c] = perCascadeShadowImpostor[c];
            }
            m_UI.visibleTerrainMeshletCount = terrainSurvivors;

            m_UI.trunkMainMeshletsDispatched   = trunkMainConsidered;
            m_UI.trunkMainMeshletsRendered     = trunkMainSurvived;
            m_UI.trunkShadowMeshletsDispatched = trunkShadowConsidered;
            m_UI.trunkShadowMeshletsRendered   = trunkShadowSurvived;
            m_UI.leafMainMeshletsDispatched    = leafMainConsidered;
            m_UI.leafMainMeshletsRendered      = leafMainSurvived;
            m_UI.leafShadowMeshletsDispatched  = leafShadowConsidered;
            m_UI.leafShadowMeshletsRendered    = leafShadowSurvived;
            m_UI.lodCountForUI = lodsForUI;
            for (uint32_t li = 0; li < UIData::kMaxLodsForUI; ++li)
                m_UI.lodVisibleCounts[li] = perLodCounts[li];
        }
    }
    m_ReadbackFrameIndex++;

    // SDSM debug readback (mirrors ComputeRenderPass)
    if (hizActive) {
        m_SDSMReadbackFrameIndex++;
    }
    {
        uint32_t readSlot = m_SDSMReadbackFrameIndex % k_QueuedFrames;
        if (m_SDSMReadbackPending[readSlot]) {
            void* pData = GetDevice()->mapBuffer(m_SDSMReadbackBuffers[readSlot], nvrhi::CpuAccessMode::Read);
            if (pData) {
                const auto* out = static_cast<const shader_cb::SDSMCascadeBuildOutput*>(pData);

                m_UI.sdsmDebugValid    = true;
                m_UI.sdsmNearDepthVal  = out->debugDepthExtents.x;
                m_UI.sdsmFarDepthVal   = out->debugDepthExtents.y;
                m_UI.sdsmTightNear     = out->debugDepthExtents.z;
                m_UI.sdsmTightFar      = out->debugDepthExtents.w;

                m_UI.sdsmCascadeSplits[0] = out->cascadeSplits.x;
                m_UI.sdsmCascadeSplits[1] = out->cascadeSplits.y;
                m_UI.sdsmCascadeSplits[2] = out->cascadeSplits.z;
                m_UI.sdsmCascadeSplits[3] = out->cascadeSplits.w;

                for (uint32_t c = 0; c < Render::c_NumCascades; ++c) {
                    m_UI.sdsmShadowCasterMinLS[c][0] = out->shadowCasterMinLS[c].x;
                    m_UI.sdsmShadowCasterMinLS[c][1] = out->shadowCasterMinLS[c].y;
                    m_UI.sdsmShadowCasterMinLS[c][2] = out->shadowCasterMinLS[c].z;
                    m_UI.sdsmShadowCasterMaxLS[c][0] = out->shadowCasterMaxLS[c].x;
                    m_UI.sdsmShadowCasterMaxLS[c][1] = out->shadowCasterMaxLS[c].y;
                    m_UI.sdsmShadowCasterMaxLS[c][2] = out->shadowCasterMaxLS[c].z;

                    float extX = out->shadowCasterMaxLS[c].x - out->shadowCasterMinLS[c].x;
                    m_UI.cascadeTexelSize[c] = (extX > 0.f) ? extX / float(k_ShadowRes) : 0.f;
                }

                GetDevice()->unmapBuffer(m_SDSMReadbackBuffers[readSlot]);
            }
            m_SDSMReadbackPending[readSlot] = false;
        }
    }

    cpuTimer.Stop();
    m_UI.cpuRenderTimeMs = (float)cpuTimer.Milliseconds();
}

// ===========================================================================
// Depth prepass: reuses main_as + depth_ms (no PS) for trees + traditional
// raster for terrain. Reads last-frame mainDispatchArgsBuffer.
// ===========================================================================

bool MeshShaderRenderPass::_InitDepthPrepass() {
    if (!m_ShaderFactory) return false;

    // Tree depth uses shared main_as (cone cull, Hi-Z toggled via ASCullCB) + depth_ms.
    m_StageResources.depthPrepass.amplificationShader = m_StageResources.sceneDraw.amplificationShader;
    m_StageResources.depthPrepass.meshShader = m_ShaderFactory->CreateShader("app/MeshShaderPass.hlsl",
        "depth_ms", nullptr, nvrhi::ShaderType::Mesh);
    if (!m_StageResources.depthPrepass.meshShader) {
        log::error("MeshShaderRenderPass: depth_ms compile failed");
        return false;
    }

    // Tree depth pipeline shares the main draw binding layout - same SRVs/CBs.
    m_StageResources.depthPrepass.bindingLayout = m_StageResources.sceneDraw.bindingLayout;

    // Terrain depth path: per-meshlet frustum + Hi-Z cull via depth_terrain_as,
    // position-only MS output. Reuses the eye-view terrain meshlet buffers and
    // SoA position SRV. No survivor counter UAV — the color pass owns that stat.
    m_StageResources.depthPrepass.terrainAS = m_ShaderFactory->CreateShader(
        "app/terrain_meshlet.hlsl", "depth_terrain_as", nullptr, nvrhi::ShaderType::Amplification);
    m_StageResources.depthPrepass.terrainMS = m_ShaderFactory->CreateShader(
        "app/terrain_meshlet.hlsl", "depth_terrain_ms", nullptr, nvrhi::ShaderType::Mesh);
    if (!m_StageResources.depthPrepass.terrainAS || !m_StageResources.depthPrepass.terrainMS) {
        log::error("MeshShaderRenderPass: depth_terrain AS/MS compile failed");
        return false;
    }

    nvrhi::BindingLayoutDesc terrainBLD;
    terrainBLD.visibility = nvrhi::ShaderType::All;
    terrainBLD.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::Terrain::kCB_Frame),
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::Terrain::kCB_ASCull),
        nvrhi::BindingLayoutItem::RawBuffer_SRV(mesh_reg::Terrain::kSRV_Positions),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Terrain::kSRV_MeshletDescs),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Terrain::kSRV_MeshletVertIdx),
        nvrhi::BindingLayoutItem::RawBuffer_SRV(mesh_reg::Terrain::kSRV_MeshletPrimIdx),
        nvrhi::BindingLayoutItem::Texture_SRV(mesh_reg::Terrain::kTex_HiZ),
    };
    m_StageResources.depthPrepass.terrainBindingLayout = GetDevice()->createBindingLayout(terrainBLD);
    return m_StageResources.depthPrepass.terrainBindingLayout != nullptr;
}

bool MeshShaderRenderPass::_InitHiZShaders() {
    auto device = GetDevice();

    m_StageResources.hiz.copyCS = m_ShaderFactory->CreateShader("app/HiZCopy.hlsl",
        "HiZCopy", nullptr, nvrhi::ShaderType::Compute);
    m_StageResources.hiz.buildCS = m_ShaderFactory->CreateShader("app/HiZBuild.hlsl",
        "HiZDownsample", nullptr, nvrhi::ShaderType::Compute);
    if (!m_StageResources.hiz.copyCS || !m_StageResources.hiz.buildCS) return false;

    nvrhi::BindingLayoutDesc bld;
    bld.visibility = nvrhi::ShaderType::Compute;
    bld.bindings = {
        nvrhi::BindingLayoutItem::PushConstants(mesh_reg::HiZ::kPushC_DestDimensions, sizeof(uint32_t) * mesh_reg::HiZ::kPushCDwordCount),
        nvrhi::BindingLayoutItem::Texture_SRV(mesh_reg::HiZ::kSRV_Source),
        nvrhi::BindingLayoutItem::Texture_UAV(mesh_reg::HiZ::kUAV_Dest),
    };
    m_StageResources.hiz.buildBindingLayout = device->createBindingLayout(bld);
    if (!m_StageResources.hiz.buildBindingLayout) return false;

    nvrhi::ComputePipelineDesc copyPso;
    copyPso.CS = m_StageResources.hiz.copyCS;
    copyPso.bindingLayouts = { m_StageResources.hiz.buildBindingLayout };
    m_StageResources.hiz.copyPipeline = device->createComputePipeline(copyPso);

    nvrhi::ComputePipelineDesc buildPso;
    buildPso.CS = m_StageResources.hiz.buildCS;
    buildPso.bindingLayouts = { m_StageResources.hiz.buildBindingLayout };
    m_StageResources.hiz.buildPipeline = device->createComputePipeline(buildPso);
    if (!m_StageResources.hiz.copyPipeline || !m_StageResources.hiz.buildPipeline) return false;

    // 1x1 placeholder so binding sets resolve before first _EnsureHiZResources.
    m_StageResources.hiz.hizTexture = device->createTexture(nvrhi::TextureDesc()
        .setWidth(1).setHeight(1).setMipLevels(1)
        .setFormat(nvrhi::Format::RGBA32_FLOAT)
        .setIsUAV(true)
        .setInitialState(nvrhi::ResourceStates::ShaderResource)
        .setKeepInitialState(true)
        .setDebugName("MeshHiZ_Placeholder"));
    m_StageResources.hiz.numMips = 1;
    return true;
}

bool MeshShaderRenderPass::_InitSDSMPass() {
    auto device = GetDevice();

    m_StageResources.sdsm.buildCS = m_ShaderFactory->CreateShader("app/SDSMBuildCascades.hlsl",
        "BuildCascades", nullptr, nvrhi::ShaderType::Compute);
    if (!m_StageResources.sdsm.buildCS) return false;

    nvrhi::BindingLayoutDesc bld;
    bld.visibility = nvrhi::ShaderType::Compute;
    bld.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::SDSM::kCB_Input),
        nvrhi::BindingLayoutItem::Texture_SRV(mesh_reg::SDSM::kSRV_HiZ),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(mesh_reg::SDSM::kUAV_CascadeOut),
    };
    m_StageResources.sdsm.buildBindingLayout = device->createBindingLayout(bld);
    if (!m_StageResources.sdsm.buildBindingLayout) return false;

    nvrhi::ComputePipelineDesc pso;
    pso.CS = m_StageResources.sdsm.buildCS;
    pso.bindingLayouts = { m_StageResources.sdsm.buildBindingLayout };
    m_StageResources.sdsm.buildPipeline = device->createComputePipeline(pso);
    if (!m_StageResources.sdsm.buildPipeline) return false;

    m_StageResources.sdsm.inputCB = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(shader_cb::kSDSMCascadeBuildInputSize)
        .setIsConstantBuffer(true)
        .setDebugName("MeshSDSM_InputCB")
        .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer));
    if (!m_StageResources.sdsm.inputCB) return false;

    m_StageResources.sdsm.cascadeDataBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(sizeof(shader_cb::SDSMCascadeBuildOutput))
        .setStructStride(sizeof(shader_cb::SDSMCascadeBuildOutput))
        .setDebugName("MeshSDSM_CascadeData")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));
    if (!m_StageResources.sdsm.cascadeDataBuffer) return false;

    // Initial binding set references placeholder Hi-Z; rebuilt in _EnsureHiZResources.
    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::SDSM::kCB_Input, m_StageResources.sdsm.inputCB),
        nvrhi::BindingSetItem::Texture_SRV(mesh_reg::SDSM::kSRV_HiZ, m_StageResources.hiz.hizTexture),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(mesh_reg::SDSM::kUAV_CascadeOut, m_StageResources.sdsm.cascadeDataBuffer),
    };
    m_StageResources.sdsm.buildBindingSet = device->createBindingSet(bsd, m_StageResources.sdsm.buildBindingLayout);
    if (!m_StageResources.sdsm.buildBindingSet) return false;

    // Debug readback ring (mirrors ComputeRenderPass)
    for (uint32_t i = 0; i < k_QueuedFrames; i++) {
        m_SDSMReadbackBuffers[i] = device->createBuffer(
            nvrhi::BufferDesc()
                .setByteSize(sizeof(shader_cb::SDSMCascadeBuildOutput))
                .setCpuAccess(nvrhi::CpuAccessMode::Read)
                .setDebugName(("MeshSDSM_Readback_" + std::to_string(i)).c_str())
        );
        if (!m_SDSMReadbackBuffers[i]) return false;
    }

    return true;
}

bool MeshShaderRenderPass::_InitTerrainPass(nvrhi::ICommandList* initCL) {
    auto& T = m_StageResources.sceneTerrain;
    const auto* terrain = m_Registry.getTerrain();
    if (!terrain) return true;

    const auto& verts   = terrain->getVertices();
    const auto& indices = terrain->getIndices();
    if (verts.empty() || indices.empty()) return true;

    T.indexCount = static_cast<uint32_t>(indices.size());

    T.amplificationShader = m_ShaderFactory->CreateShader(
        "app/terrain_meshlet.hlsl", "terrain_as", nullptr, nvrhi::ShaderType::Amplification);
    T.meshShader = m_ShaderFactory->CreateShader(
        "app/terrain_meshlet.hlsl", "terrain_ms", nullptr, nvrhi::ShaderType::Mesh);
    T.pixelShader = m_ShaderFactory->CreateShader(
        "app/terrain_meshlet.hlsl", "terrain_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!T.amplificationShader || !T.meshShader || !T.pixelShader) {
        log::error("MeshShaderRenderPass: terrain_meshlet shaders failed to compile");
        return false;
    }

    // Deinterleave into 3 SoA streams. Each is raw-viewable so the meshlet
    // AS/MS can load them as ByteAddressBuffer with explicit offsets. Depth
    // and shadow paths bind only the position stream.
    std::vector<dm::float3> positions(verts.size());
    std::vector<dm::float3> normals  (verts.size());
    std::vector<dm::float2> uvs      (verts.size());
    for (size_t i = 0; i < verts.size(); i++) {
        positions[i] = verts[i].pos;
        normals  [i] = verts[i].normal;
        uvs      [i] = verts[i].uv;
    }

    auto makeStream = [&](const char* name, const void* data, uint64_t bytes,
                          nvrhi::BufferHandle& outHandle)
    {
        nvrhi::BufferDesc d;
        d.byteSize        = bytes;
        d.canHaveRawViews = true;
        d.debugName       = name;
        d.initialState    = nvrhi::ResourceStates::CopyDest;
        outHandle = GetDevice()->createBuffer(d);
        initCL->beginTrackingBufferState(outHandle, nvrhi::ResourceStates::CopyDest);
        initCL->writeBuffer(outHandle, data, bytes);
        initCL->setPermanentBufferState(outHandle, nvrhi::ResourceStates::ShaderResource);
    };
    makeStream("MeshTerrainVB_Pos", positions.data(), positions.size() * sizeof(dm::float3),
               T.positionBuffer);
    makeStream("MeshTerrainVB_Nor", normals.data(),   normals.size()   * sizeof(dm::float3),
               T.normalBuffer);
    makeStream("MeshTerrainVB_UV",  uvs.data(),       uvs.size()       * sizeof(dm::float2),
               T.uvBuffer);

    nvrhi::BufferDesc ibDesc;
    ibDesc.isIndexBuffer = true;
    ibDesc.byteSize      = indices.size() * sizeof(uint32_t);
    ibDesc.debugName     = "MeshTerrainIB";
    ibDesc.initialState  = nvrhi::ResourceStates::CopyDest;
    T.indexBuffer = GetDevice()->createBuffer(ibDesc);
    initCL->beginTrackingBufferState(T.indexBuffer, nvrhi::ResourceStates::CopyDest);
    initCL->writeBuffer(T.indexBuffer, indices.data(), ibDesc.byteSize);
    initCL->setPermanentBufferState(T.indexBuffer, nvrhi::ResourceStates::IndexBuffer);

    // ---- Meshlet GPU buffers ----
    const auto& meshlets = terrain->getMeshlets();
    T.meshletCount = static_cast<uint32_t>(meshlets.descs.size());
    if (T.meshletCount > 0) {
        nvrhi::BufferDesc descsBD;
        descsBD.byteSize     = meshlets.descs.size() * sizeof(Scene::TerrainMeshletDesc);
        descsBD.structStride = sizeof(Scene::TerrainMeshletDesc);
        descsBD.debugName    = "MeshTerrainMeshletDescs";
        descsBD.initialState = nvrhi::ResourceStates::CopyDest;
        T.meshletDescBuffer = GetDevice()->createBuffer(descsBD);
        initCL->beginTrackingBufferState(T.meshletDescBuffer, nvrhi::ResourceStates::CopyDest);
        initCL->writeBuffer(T.meshletDescBuffer, meshlets.descs.data(), descsBD.byteSize);
        initCL->setPermanentBufferState(T.meshletDescBuffer, nvrhi::ResourceStates::ShaderResource);

        nvrhi::BufferDesc vidBD;
        vidBD.byteSize     = meshlets.localVertIndices.size() * sizeof(uint32_t);
        vidBD.structStride = sizeof(uint32_t);
        vidBD.debugName    = "MeshTerrainMeshletVertIdx";
        vidBD.initialState = nvrhi::ResourceStates::CopyDest;
        T.meshletVertIdxBuffer = GetDevice()->createBuffer(vidBD);
        initCL->beginTrackingBufferState(T.meshletVertIdxBuffer, nvrhi::ResourceStates::CopyDest);
        initCL->writeBuffer(T.meshletVertIdxBuffer,
            meshlets.localVertIndices.data(), vidBD.byteSize);
        initCL->setPermanentBufferState(T.meshletVertIdxBuffer, nvrhi::ResourceStates::ShaderResource);

        nvrhi::BufferDesc pidBD;
        pidBD.byteSize       = meshlets.localTriIndices.size();
        pidBD.canHaveRawViews = true;
        pidBD.debugName      = "MeshTerrainMeshletPrimIdx";
        pidBD.initialState   = nvrhi::ResourceStates::CopyDest;
        T.meshletPrimIdxBuffer = GetDevice()->createBuffer(pidBD);
        initCL->beginTrackingBufferState(T.meshletPrimIdxBuffer, nvrhi::ResourceStates::CopyDest);
        initCL->writeBuffer(T.meshletPrimIdxBuffer,
            meshlets.localTriIndices.data(), pidBD.byteSize);
        initCL->setPermanentBufferState(T.meshletPrimIdxBuffer, nvrhi::ResourceStates::ShaderResource);
    }

    // Per-frame survivor counter — single uint32 raw UAV. Cleared at frame start
    // by clearBufferUInt; copied to readback ring after the AS dispatch.
    nvrhi::BufferDesc counterBD;
    counterBD.byteSize       = sizeof(uint32_t);
    counterBD.canHaveUAVs    = true;
    counterBD.canHaveRawViews = true;
    counterBD.debugName      = "MeshTerrainVisibleCounter";
    counterBD.initialState   = nvrhi::ResourceStates::UnorderedAccess;
    counterBD.keepInitialState = true;
    T.visibleCounterBuffer = GetDevice()->createBuffer(counterBD);

    _RebuildTerrainBindingSet();
    m_UI.totalTerrainMeshletCount = T.meshletCount;
    return true;
}

void MeshShaderRenderPass::_RebuildTerrainBindingSet() {
    auto& T = m_StageResources.sceneTerrain;
    if (T.meshletCount == 0)                       return;
    if (!T.meshletDescBuffer)                      return;
    if (!T.positionBuffer)                         return;

    // Shadow terrain binding set only needs positions + meshlet buffers + frame
    // CB — build it as soon as those exist, ahead of the eye-view set's Hi-Z
    // gate. shadow_terrain_ms reads only position (normal/uv unused for depth).
    if (m_StageResources.shadow.terrainBindingLayout) {
        nvrhi::BindingSetDesc shadowBSD;
        shadowBSD.bindings = {
            nvrhi::BindingSetItem::PushConstants(mesh_reg::Terrain::kPushC_Cascade, sizeof(uint32_t)),
            nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Terrain::kCB_Frame, m_StageResources.frameShared.constantBuffer,
                nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
            nvrhi::BindingSetItem::RawBuffer_SRV(mesh_reg::Terrain::kSRV_Positions,        T.positionBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Terrain::kSRV_MeshletDescs,   T.meshletDescBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Terrain::kSRV_MeshletVertIdx, T.meshletVertIdxBuffer),
            nvrhi::BindingSetItem::RawBuffer_SRV(mesh_reg::Terrain::kSRV_MeshletPrimIdx,  T.meshletPrimIdxBuffer),
        };
        m_StageResources.shadow.terrainBindingSet = GetDevice()->createBindingSet(
            shadowBSD, m_StageResources.shadow.terrainBindingLayout);
    }

    if (!m_StageResources.hiz.hizTexture)          return;

    // Depth prepass terrain binding set: same meshlet buffers + position stream
    // + Hi-Z. No textures / no UAV counter / no normal/UV streams.
    if (m_StageResources.depthPrepass.terrainBindingLayout) {
        nvrhi::BindingSetDesc depthBSD;
        depthBSD.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Terrain::kCB_Frame, m_StageResources.frameShared.constantBuffer,
                nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
            nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Terrain::kCB_ASCull,  m_StageResources.frameShared.asCullCB,
                nvrhi::BufferRange(0, shader_cb::kMeshASCullSize)),
            nvrhi::BindingSetItem::RawBuffer_SRV(mesh_reg::Terrain::kSRV_Positions,        T.positionBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Terrain::kSRV_MeshletDescs,   T.meshletDescBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Terrain::kSRV_MeshletVertIdx, T.meshletVertIdxBuffer),
            nvrhi::BindingSetItem::RawBuffer_SRV(mesh_reg::Terrain::kSRV_MeshletPrimIdx,  T.meshletPrimIdxBuffer),
            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Terrain::kTex_HiZ, m_StageResources.hiz.hizTexture),
        };
        m_StageResources.depthPrepass.terrainBindingSet = GetDevice()->createBindingSet(
            depthBSD, m_StageResources.depthPrepass.terrainBindingLayout);
    }

    if (!T.visibleCounterBuffer)                   return;
    if (!m_StageResources.shadow.depthTexture)     return;
    if (!m_Shared)                                 return;

    const auto& terrainTex = m_Shared->terrainTextures();
    if (terrainTex.size() < 4)                     return;

    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Terrain::kCB_Frame, m_StageResources.frameShared.constantBuffer,
            nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
        nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Terrain::kCB_Shading, m_Shared->terrainShadingCB()),
        nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Terrain::kCB_ASCull,  m_StageResources.frameShared.asCullCB,
            nvrhi::BufferRange(0, shader_cb::kMeshASCullSize)),
        nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Terrain::kTex_ShadowMap, m_StageResources.shadow.depthTexture),
        nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Terrain::kTex_ForestDiff, terrainTex[0].diffuse),
        nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Terrain::kTex_ForestNor,  terrainTex[0].normalMap),
        nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Terrain::kTex_DirtDiff,   terrainTex[1].diffuse),
        nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Terrain::kTex_DirtNor,    terrainTex[1].normalMap),
        nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Terrain::kTex_RockDiff,   terrainTex[2].diffuse),
        nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Terrain::kTex_RockNor,    terrainTex[2].normalMap),
        nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Terrain::kTex_SnowDiff,   terrainTex[3].diffuse),
        nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Terrain::kTex_SnowNor,    terrainTex[3].normalMap),
        nvrhi::BindingSetItem::RawBuffer_SRV(mesh_reg::Terrain::kSRV_Positions, T.positionBuffer),
        nvrhi::BindingSetItem::RawBuffer_SRV(mesh_reg::Terrain::kSRV_Normals,   T.normalBuffer),
        nvrhi::BindingSetItem::RawBuffer_SRV(mesh_reg::Terrain::kSRV_UVs,       T.uvBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Terrain::kSRV_MeshletDescs, T.meshletDescBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Terrain::kSRV_MeshletVertIdx, T.meshletVertIdxBuffer),
        nvrhi::BindingSetItem::RawBuffer_SRV(mesh_reg::Terrain::kSRV_MeshletPrimIdx, T.meshletPrimIdxBuffer),
        nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Terrain::kTex_HiZ, m_StageResources.hiz.hizTexture),
        nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Terrain::kUAV_VisibleCounter, T.visibleCounterBuffer),
        nvrhi::BindingSetItem::Sampler(mesh_reg::Terrain::kSampler_Shadow, m_StageResources.sceneDraw.shadowSampler),
        nvrhi::BindingSetItem::Sampler(mesh_reg::Terrain::kSampler_Aniso,  m_Shared->barkSampler()),
    };
    // Null the binding set so CreateBindingSetAndLayout recreates it (it only
    // builds a new set when the handle is null on entry; layout is preserved).
    T.bindingSet = nullptr;
    nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
        bsd, T.bindingLayout, T.bindingSet);
}

bool MeshShaderRenderPass::_InitSkyPass() {
    m_StageResources.sky.vertexShader = m_ShaderFactory->CreateShader(
        "app/sky.hlsl", "sky_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_StageResources.sky.pixelShader = m_ShaderFactory->CreateShader(
        "app/sky.hlsl", "sky_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_StageResources.sky.vertexShader || !m_StageResources.sky.pixelShader) return false;

    nvrhi::BufferDesc cbDesc;
    cbDesc.byteSize         = sizeof(SkyConstants);
    cbDesc.isConstantBuffer = true;
    cbDesc.isVolatile       = true;
    cbDesc.maxVersions      = 16;
    cbDesc.debugName        = "MeshSkyConstants";
    m_StageResources.sky.constantBuffer = GetDevice()->createBuffer(cbDesc);
    if (!m_StageResources.sky.constantBuffer) return false;

    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Sky::kCB_Sky, m_StageResources.sky.constantBuffer),
    };
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
            bsd, m_StageResources.sky.bindingLayout, m_StageResources.sky.bindingSet))
        return false;

    return true;
}

// ===========================================================================
// Binding-set rebuild for depth prepass
// ===========================================================================

void MeshShaderRenderPass::_RebuildDepthPrepassBindingSet() {
    // Tree meshlet depth prepass mirrors sceneDraw[0] but rebinds the meshlet
    // stats UAV to a scratch buffer so the shared main_as doesn't double-count
    // (depth prepass + color both run main_as on the same visible instances).
    if (!m_Shared || !m_StageResources.shadow.depthTexture
        || !m_StageResources.hiz.hizTexture
        || !m_StageResources.sceneDraw.bindingLayout
        || m_Shared->barkTextures().empty()) {
        m_StageResources.depthPrepass.bindingSet = nullptr;
    } else {
        const auto& barkTextures = m_Shared->barkTextures();
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::PushConstants(mesh_reg::Draw::kPushC_Slot, mesh_reg::Draw::kPushCBytes),
            nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Draw::kCB_Frame, m_StageResources.frameShared.constantBuffer,
                nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
            nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Draw::kCB_ASCull, m_StageResources.frameShared.asCullCB,
                nvrhi::BufferRange(0, shader_cb::kMeshASCullSize)),

            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Positions,     m_StageResources.sceneMeshletData.positions),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Normals,       m_StageResources.sceneMeshletData.normals),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Tangents,      m_StageResources.sceneMeshletData.tangents),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Bitangents,    m_StageResources.sceneMeshletData.bitangents),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_UVs,           m_StageResources.sceneMeshletData.uvs),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_MeshletVertIdx, m_StageResources.sceneMeshletData.meshletVertIdx),
            nvrhi::BindingSetItem::RawBuffer_SRV(mesh_reg::Draw::kSRV_MeshletPrimIdx,        m_StageResources.sceneMeshletData.meshletPrimIdx),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Meshlets,      m_StageResources.sceneMeshletData.meshletDescs),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_AssetLods,     m_StageResources.sceneMeshletData.assetLodRanges),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Vis,        m_StageResources.cull.mainVisBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_SlotOffsets,   m_StageResources.cull.mainSlotOffsetBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_SlotCounts,    m_StageResources.cull.mainCountBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Instances,     m_StageResources.cull.persistentInstBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_ASInvocations, m_StageResources.cull.mainASInvocsPerSlotBuffer),

            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Draw::kTex_Diffuse,   barkTextures[0].diffuse),
            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Draw::kTex_NormalMap, barkTextures[0].normalMap),
            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Draw::kTex_ShadowMap, m_StageResources.shadow.depthTexture),
            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Draw::kTex_HiZ,    m_StageResources.hiz.hizTexture),

            nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Draw::kUAV_MeshletStats, m_StageResources.cull.meshletStatsDepthScratch),

            nvrhi::BindingSetItem::Sampler(mesh_reg::Draw::kSampler_Main,   m_Shared->barkSampler()),
            nvrhi::BindingSetItem::Sampler(mesh_reg::Draw::kSampler_Shadow, m_StageResources.sceneDraw.shadowSampler),
        };
        m_StageResources.depthPrepass.bindingSet = GetDevice()->createBindingSet(
            bsd, m_StageResources.sceneDraw.bindingLayout);
    }

    // Terrain depth prepass binding set is now built inside
    // _RebuildTerrainBindingSet (it uses the Mesh::Terrain register namespace
    // shared with the eye-view / shadow terrain meshlet paths).
}

// ===========================================================================
// Hi-Z resource (re)creation - called on back-buffer resize
// ===========================================================================

void MeshShaderRenderPass::_EnsureHiZResources(uint32_t width, uint32_t height) {
    if (m_StageResources.depthPrepass.depthTexture) {
        auto desc = m_StageResources.depthPrepass.depthTexture->getDesc();
        if (desc.width == width && desc.height == height) return;
    }

    auto device = GetDevice();

    m_StageResources.depthPrepass.depthTexture = device->createTexture(nvrhi::TextureDesc()
        .setWidth(width).setHeight(height)
        .setFormat(nvrhi::Format::D32)
        .setIsRenderTarget(true)
        .setUseClearValue(true)
    #if XYLEM_USE_REVERSE_Z
        .setClearValue(nvrhi::Color(0.f))
    #else
        .setClearValue(nvrhi::Color(1.f))
    #endif
        .setInitialState(nvrhi::ResourceStates::DepthWrite)
        .setKeepInitialState(true)
        .setDebugName("MeshDepthPrepass_Depth"));
    m_StageResources.depthPrepass.framebuffer = device->createFramebuffer(
        nvrhi::FramebufferDesc().setDepthAttachment(m_StageResources.depthPrepass.depthTexture));

    m_StageResources.depthPrepass.pipeline         = nullptr;
    m_StageResources.depthPrepass.terrainPipeline  = nullptr;
    m_StageResources.depthPrepass.dispatchMeshSignature = nullptr;

    m_StageResources.hiz.numMips = static_cast<uint32_t>(
        std::floor(std::log2(std::max(width, height)))) + 1;
    m_StageResources.hiz.hizTexture = device->createTexture(nvrhi::TextureDesc()
        .setWidth(width).setHeight(height)
        .setMipLevels(m_StageResources.hiz.numMips)
        .setFormat(nvrhi::Format::RGBA32_FLOAT)
        .setIsUAV(true)
        .setInitialState(nvrhi::ResourceStates::ShaderResource)
        .setKeepInitialState(true)
        .setDebugName("MeshHiZTexture"));

    m_StageResources.hiz.buildBindingSets.resize(m_StageResources.hiz.numMips);

    // Mip 0 - copy depth prepass (D32 -> R32_FLOAT view) into RGBA32 mip 0.
    // .r = raw min (Hi-Z), .g = raw max (SDSM near), .b = sky-excluded min (SDSM far).
    {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::PushConstants(mesh_reg::HiZ::kPushC_DestDimensions, sizeof(uint32_t) * mesh_reg::HiZ::kPushCDwordCount),
            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::HiZ::kSRV_Source, m_StageResources.depthPrepass.depthTexture,
                nvrhi::Format::R32_FLOAT,
                nvrhi::TextureSubresourceSet(0, 1, 0, 1)),
            nvrhi::BindingSetItem::Texture_UAV(mesh_reg::HiZ::kUAV_Dest, m_StageResources.hiz.hizTexture,
                nvrhi::Format::RGBA32_FLOAT,
                nvrhi::TextureSubresourceSet(0, 1, 0, 1)),
        };
        m_StageResources.hiz.buildBindingSets[0] = device->createBindingSet(bsd, m_StageResources.hiz.buildBindingLayout);
    }

    // Mips 1..N-1 - downsample i-1 -> i. HiZDownsample reads the source mip via
    // texel indexing with explicit edge-clamp on odd tails.
    for (uint32_t mip = 1; mip < m_StageResources.hiz.numMips; mip++) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::PushConstants(mesh_reg::HiZ::kPushC_DestDimensions, sizeof(uint32_t) * mesh_reg::HiZ::kPushCDwordCount),
            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::HiZ::kSRV_Source, m_StageResources.hiz.hizTexture,
                nvrhi::Format::RGBA32_FLOAT,
                nvrhi::TextureSubresourceSet(mip - 1, 1, 0, 1)),
            nvrhi::BindingSetItem::Texture_UAV(mesh_reg::HiZ::kUAV_Dest, m_StageResources.hiz.hizTexture,
                nvrhi::Format::RGBA32_FLOAT,
                nvrhi::TextureSubresourceSet(mip, 1, 0, 1)),
        };
        m_StageResources.hiz.buildBindingSets[mip] = device->createBindingSet(bsd, m_StageResources.hiz.buildBindingLayout);
    }

    m_StageResources.hiz.debugMipTextures.resize(m_StageResources.hiz.numMips);
    for (uint32_t mip = 0; mip < m_StageResources.hiz.numMips; mip++) {
        uint32_t mipW = std::max(1u, width  >> mip);
        uint32_t mipH = std::max(1u, height >> mip);
        m_StageResources.hiz.debugMipTextures[mip] = device->createTexture(nvrhi::TextureDesc()
            .setWidth(mipW).setHeight(mipH).setMipLevels(1)
            .setFormat(nvrhi::Format::RGBA32_FLOAT)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true)
            .setDebugName(("MeshHiZ_DebugMip" + std::to_string(mip)).c_str()));
    }

    // Rebuild downstream binding sets that reference the Hi-Z texture.
    _RebuildCullBindingSet();
    _RebuildDrawBindingSet();
    _RebuildDepthPrepassBindingSet();
    _RebuildLeafBindingSets();
    _RebuildTerrainBindingSet();

    if (m_StageResources.sdsm.buildBindingLayout) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::SDSM::kCB_Input, m_StageResources.sdsm.inputCB),
            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::SDSM::kSRV_HiZ, m_StageResources.hiz.hizTexture),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(mesh_reg::SDSM::kUAV_CascadeOut, m_StageResources.sdsm.cascadeDataBuffer),
        };
        m_StageResources.sdsm.buildBindingSet = device->createBindingSet(bsd, m_StageResources.sdsm.buildBindingLayout);
    }
}

// ===========================================================================
// Depth prepass / Hi-Z build / SDSM
// ===========================================================================

void MeshShaderRenderPass::_CreateDepthPrepassPipelineIfNeeded() {
    if (m_StageResources.depthPrepass.pipeline || !m_StageResources.depthPrepass.framebuffer) return;

    nvrhi::MeshletPipelineDesc psoDesc;
    psoDesc.AS = m_StageResources.depthPrepass.amplificationShader;
    psoDesc.MS = m_StageResources.depthPrepass.meshShader;
    psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
    psoDesc.bindingLayouts = { m_StageResources.depthPrepass.bindingLayout };

    auto& rs = psoDesc.renderState;
    rs.depthStencilState.depthTestEnable  = true;
    rs.depthStencilState.depthWriteEnable = true;
#if XYLEM_USE_REVERSE_Z
    rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::GreaterOrEqual;
#else
    rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::LessOrEqual;
#endif
    rs.rasterState.cullMode = nvrhi::RasterCullMode::Back;

    m_StageResources.depthPrepass.pipeline = GetDevice()->createMeshletPipeline(
        psoDesc, m_StageResources.depthPrepass.framebuffer->getFramebufferInfo());

    if (m_StageResources.depthPrepass.pipeline && !m_StageResources.depthPrepass.dispatchMeshSignature) {
        ID3D12Device* d3dDevice = GetDevice()->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
        ID3D12RootSignature* rootSig = m_StageResources.depthPrepass.pipeline->getNativeObject(
            nvrhi::ObjectTypes::D3D12_RootSignature);
        if (d3dDevice && rootSig) {
            D3D12_INDIRECT_ARGUMENT_DESC args[2] = {};
            args[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
            args[0].Constant.RootParameterIndex      = mesh_reg::kPushC_RootParamIdx;
            args[0].Constant.DestOffsetIn32BitValues = 0;
            args[0].Constant.Num32BitValuesToSet     = 1;
            args[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;

            D3D12_COMMAND_SIGNATURE_DESC desc = {};
            desc.ByteStride       = sizeof(DispatchRecord);
            desc.NumArgumentDescs = 2;
            desc.pArgumentDescs   = args;

            HRESULT hr = d3dDevice->CreateCommandSignature(&desc, rootSig,
                IID_PPV_ARGS(&m_StageResources.depthPrepass.dispatchMeshSignature));
            if (FAILED(hr)) {
                log::error("MeshShaderRenderPass: CreateCommandSignature(depth) failed (0x%08x)", hr);
                m_StageResources.depthPrepass.dispatchMeshSignature = nullptr;
            }
        }
    }
}

void MeshShaderRenderPass::_RenderDepthPrepass() {
    _CreateDepthPrepassPipelineIfNeeded();

#if XYLEM_USE_REVERSE_Z
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, m_StageResources.depthPrepass.framebuffer, 0.f, 0);
#else
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, m_StageResources.depthPrepass.framebuffer, 1.f, 0);
#endif

    // Trees: ExecuteIndirect over last-frame mainDispatchArgsBuffer (not yet cleared).
    if (m_StageResources.depthPrepass.pipeline && m_StageResources.depthPrepass.dispatchMeshSignature) {
        nvrhi::MeshletState meshState;
        meshState.pipeline       = m_StageResources.depthPrepass.pipeline;
        meshState.framebuffer    = m_StageResources.depthPrepass.framebuffer;
        meshState.bindings       = { m_StageResources.depthPrepass.bindingSet };
        meshState.indirectParams = m_StageResources.cull.mainDispatchArgsBuffer;
        meshState.viewport.addViewportAndScissorRect(
            m_StageResources.depthPrepass.framebuffer->getFramebufferInfo().getViewport());
        m_CommandList->setMeshletState(meshState);

        uint32_t placeholder = 0;
        m_CommandList->setPushConstants(&placeholder, sizeof(uint32_t));

        auto* d3dList = static_cast<ID3D12GraphicsCommandList6*>(
            m_CommandList->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList));
        auto* argBuffer = static_cast<ID3D12Resource*>(
            m_StageResources.cull.mainDispatchArgsBuffer->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource));
        if (d3dList && argBuffer) {
            d3dList->ExecuteIndirect(
                m_StageResources.depthPrepass.dispatchMeshSignature.Get(),
                m_NumMainSlots, argBuffer, 0, nullptr, 0);
        }
    }

    // Leaves into the same depth target.
    {
        auto& L = m_StageResources.sceneLeaves;
        if (L.depthPipeline && L.depthSignature && L.depthBindingSet
            && m_StageResources.cull.mainLeafDispatchArgsBuffer) {
            nvrhi::MeshletState meshState;
            meshState.pipeline       = L.depthPipeline;
            meshState.framebuffer    = m_StageResources.depthPrepass.framebuffer;
            meshState.bindings       = { L.depthBindingSet };
            meshState.indirectParams = m_StageResources.cull.mainLeafDispatchArgsBuffer;
            meshState.viewport.addViewportAndScissorRect(
                m_StageResources.depthPrepass.framebuffer->getFramebufferInfo().getViewport());
            m_CommandList->setMeshletState(meshState);

            uint32_t placeholder = 0;
            m_CommandList->setPushConstants(&placeholder, sizeof(uint32_t));

            auto* d3dList = static_cast<ID3D12GraphicsCommandList6*>(
                m_CommandList->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList));
            auto* leafArgBuf = static_cast<ID3D12Resource*>(
                m_StageResources.cull.mainLeafDispatchArgsBuffer->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource));
            if (d3dList && leafArgBuf) {
                d3dList->ExecuteIndirect(
                    L.depthSignature.Get(),
                    m_NumMainSlots, leafArgBuf, 0, nullptr, 0);
            }
        }
    }

    // Terrain: meshlet AS/MS with per-meshlet frustum + Hi-Z cull, position-only
    // MS output. Same `terrain_as` logic the color path uses (minus the survivor
    // counter), so distant / off-screen terrain meshlets contribute zero raster
    // work to the prepass and the Hi-Z built from it.
    if (m_StageResources.sceneTerrain.meshletCount > 0
        && m_StageResources.depthPrepass.terrainAS && m_StageResources.depthPrepass.terrainMS
        && m_StageResources.depthPrepass.terrainBindingSet) {
        if (!m_StageResources.depthPrepass.terrainPipeline) {
            nvrhi::MeshletPipelineDesc pso;
            pso.AS = m_StageResources.depthPrepass.terrainAS;
            pso.MS = m_StageResources.depthPrepass.terrainMS;
            pso.primType = nvrhi::PrimitiveType::TriangleList;
            pso.bindingLayouts = { m_StageResources.depthPrepass.terrainBindingLayout };
            auto& rs = pso.renderState;
            rs.depthStencilState.depthTestEnable  = true;
            rs.depthStencilState.depthWriteEnable = true;
        #if XYLEM_USE_REVERSE_Z
            rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::GreaterOrEqual;
        #else
            rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::LessOrEqual;
        #endif
            rs.rasterState.cullMode = nvrhi::RasterCullMode::Back;
            m_StageResources.depthPrepass.terrainPipeline = GetDevice()->createMeshletPipeline(
                pso, m_StageResources.depthPrepass.framebuffer->getFramebufferInfo());
        }

        if (m_StageResources.depthPrepass.terrainPipeline) {
            nvrhi::MeshletState ms;
            ms.pipeline    = m_StageResources.depthPrepass.terrainPipeline;
            ms.framebuffer = m_StageResources.depthPrepass.framebuffer;
            ms.bindings    = { m_StageResources.depthPrepass.terrainBindingSet };
            ms.viewport.addViewportAndScissorRect(
                m_StageResources.depthPrepass.framebuffer->getFramebufferInfo().getViewport());
            m_CommandList->setMeshletState(ms);

            const uint32_t numGroups =
                (m_StageResources.sceneTerrain.meshletCount + Render::k_ASGroupSize - 1)
                / Render::k_ASGroupSize;
            m_CommandList->dispatchMesh(numGroups, 1, 1);
        }
    }
}

void MeshShaderRenderPass::_BuildHiZMipChain() {
    auto desc = m_StageResources.depthPrepass.depthTexture->getDesc();
    uint32_t w = desc.width;
    uint32_t h = desc.height;

    {
        uint32_t dims[2] = { w, h };
        nvrhi::ComputeState cs;
        cs.pipeline = m_StageResources.hiz.copyPipeline;
        cs.bindings = { m_StageResources.hiz.buildBindingSets[0] };
        m_CommandList->setComputeState(cs);
        m_CommandList->setPushConstants(dims, sizeof(dims));
        m_CommandList->dispatch((w + 7) / 8, (h + 7) / 8, 1);
    }

    for (uint32_t mip = 1; mip < m_StageResources.hiz.numMips; mip++) {
        uint32_t mipW = std::max(1u, w >> mip);
        uint32_t mipH = std::max(1u, h >> mip);
        uint32_t dims[2] = { mipW, mipH };

        nvrhi::ComputeState cs;
        cs.pipeline = m_StageResources.hiz.buildPipeline;
        cs.bindings = { m_StageResources.hiz.buildBindingSets[mip] };
        m_CommandList->setComputeState(cs);
        m_CommandList->setPushConstants(dims, sizeof(dims));
        m_CommandList->dispatch((mipW + 7) / 8, (mipH + 7) / 8, 1);
    }

    if (m_UI.showHiZ && !m_StageResources.hiz.debugMipTextures.empty()) {
        for (uint32_t mip = 0; mip < m_StageResources.hiz.numMips; mip++) {
            m_CommandList->copyTexture(
                m_StageResources.hiz.debugMipTextures[mip], nvrhi::TextureSlice(),
                m_StageResources.hiz.hizTexture,           nvrhi::TextureSlice().setMipLevel(mip));
        }
    }
}

void MeshShaderRenderPass::_ComputeRegionEnvelope(const dm::float3& camPos,
                                                  const dm::float3& camDir,
                                                  float& outNearZ, float& outFarZ) const {
    outNearZ = std::numeric_limits<float>::max();
    outFarZ  = std::numeric_limits<float>::lowest();

    // Scene bbox already unions tree regions and terrain; mirrors ViewHandler::computeCascades.
    const dm::box3& sceneBbox = m_Registry.getSceneBounds();
    for (int i = 0; i < dm::box3::numCorners; i++) {
        dm::float3 corner = sceneBbox.getCorner(i);
        float vsZ = dm::dot(corner - camPos, camDir);
        outNearZ = dm::min(outNearZ, vsZ);
        outFarZ  = dm::max(outFarZ,  vsZ);
    }

    outNearZ = dm::max(outNearZ, 0.1f);
    outFarZ  = dm::max(outFarZ,  outNearZ + 1.f);
}

void MeshShaderRenderPass::_RunSDSMBuildCascades(const dm::box3& sceneBbox,
                                                 float aspectRatio, float fovY,
                                                 float regionEnvelopeNear,
                                                 float regionEnvelopeFar) {
    shader_cb::SDSMCascadeBuildInput input{};

    dm::affine3 worldToLightAff = m_ViewHandler.worldToLight;
    dm::affine3 viewToWorldToLightAff =
        m_ViewHandler.view.GetInverseViewMatrix() * worldToLightAff;

    input.worldToLight       = dm::affineToHomogeneous(worldToLightAff);
    input.viewToWorldToLight = dm::affineToHomogeneous(viewToWorldToLightAff);

    dm::box3 sceneBboxLS = sceneBbox * worldToLightAff;
    input.sceneBboxMinLS = dm::float4(sceneBboxLS.m_mins, 0.f);
    input.sceneBboxMaxLS = dm::float4(sceneBboxLS.m_maxs, 0.f);

    float tanHalfFovY = std::tanf(fovY * 0.5f);
    input.tanHalfFovX = tanHalfFovY * aspectRatio;
    input.tanHalfFovY = tanHalfFovY;

    dm::float4x4 proj = m_ViewHandler.view.GetProjectionMatrix();
    input.projA = proj.m_data[2 * 4 + 2];
    input.projB = proj.m_data[3 * 4 + 2];

    input.regionEnvelopeNear = regionEnvelopeNear;
    input.regionEnvelopeFar  = regionEnvelopeFar;
    input.cameraNearPlane    = 0.1f;
    input.shadowRes          = k_ShadowRes;
    input.maxHiZMip          = (m_StageResources.hiz.numMips > 0) ? (m_StageResources.hiz.numMips - 1) : 0;
    input.pssmLambda         = m_UI.pssmLambda;

    m_CommandList->writeBuffer(m_StageResources.sdsm.inputCB, &input, sizeof(shader_cb::SDSMCascadeBuildInput));

    {
        nvrhi::ComputeState cs;
        cs.pipeline = m_StageResources.sdsm.buildPipeline;
        cs.bindings = { m_StageResources.sdsm.buildBindingSet };
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch(1, 1, 1);
    }

    // Copy SDSM results into main CB at CullConstantBufferEntry offsets.
    constexpr size_t kOffLightViewProj   = offsetof(Render::CullConstantBufferEntry, lightViewProj);
    constexpr size_t kOffCascadeSplits   = offsetof(Render::CullConstantBufferEntry, cascadeSplits);
    constexpr size_t kOffShadowCasterMin = offsetof(Render::CullConstantBufferEntry, shadowCasterMinLS);
    constexpr size_t kOffShadowCasterMax = offsetof(Render::CullConstantBufferEntry, shadowCasterMaxLS);

    constexpr size_t kSrcLightViewProj   = offsetof(shader_cb::SDSMCascadeBuildOutput, lightViewProj);
    constexpr size_t kSrcCascadeSplits   = offsetof(shader_cb::SDSMCascadeBuildOutput, cascadeSplits);
    constexpr size_t kSrcShadowCasterMin = offsetof(shader_cb::SDSMCascadeBuildOutput, shadowCasterMinLS);
    constexpr size_t kSrcShadowCasterMax = offsetof(shader_cb::SDSMCascadeBuildOutput, shadowCasterMaxLS);

    m_CommandList->copyBuffer(m_StageResources.frameShared.constantBuffer, kOffLightViewProj,
                              m_StageResources.sdsm.cascadeDataBuffer, kSrcLightViewProj,
                              sizeof(dm::float4x4) * Render::c_NumCascades);
    m_CommandList->copyBuffer(m_StageResources.frameShared.constantBuffer, kOffCascadeSplits,
                              m_StageResources.sdsm.cascadeDataBuffer, kSrcCascadeSplits,
                              sizeof(dm::float4));
    m_CommandList->copyBuffer(m_StageResources.frameShared.constantBuffer, kOffShadowCasterMin,
                              m_StageResources.sdsm.cascadeDataBuffer, kSrcShadowCasterMin,
                              sizeof(dm::float4) * Render::c_NumCascades);
    m_CommandList->copyBuffer(m_StageResources.frameShared.constantBuffer, kOffShadowCasterMax,
                              m_StageResources.sdsm.cascadeDataBuffer, kSrcShadowCasterMax,
                              sizeof(dm::float4) * Render::c_NumCascades);

    // Debug readback: mirror full SDSM output into ring slot.
    {
        uint32_t ringSlot = m_SDSMReadbackFrameIndex % k_QueuedFrames;
        m_CommandList->copyBuffer(m_SDSMReadbackBuffers[ringSlot], 0,
                                  m_StageResources.sdsm.cascadeDataBuffer, 0,
                                  sizeof(shader_cb::SDSMCascadeBuildOutput));
        m_SDSMReadbackPending[ringSlot] = true;
    }
}

// ===========================================================================
// Sky
// ===========================================================================

void MeshShaderRenderPass::_RenderSkyPass(nvrhi::IFramebuffer* framebuffer) {
    if (!m_StageResources.sky.pipeline) {
        nvrhi::GraphicsPipelineDesc pso;
        pso.VS             = m_StageResources.sky.vertexShader;
        pso.PS             = m_StageResources.sky.pixelShader;
        pso.bindingLayouts = { m_StageResources.sky.bindingLayout };
        pso.primType       = nvrhi::PrimitiveType::TriangleStrip;
        pso.renderState.rasterState.setCullNone();
        pso.renderState.depthStencilState
            .enableDepthTest()
            .disableDepthWrite()
            .disableStencil()
    #if XYLEM_USE_REVERSE_Z
            .setDepthFunc(nvrhi::ComparisonFunc::GreaterOrEqual);
    #else
            .setDepthFunc(nvrhi::ComparisonFunc::LessOrEqual);
    #endif
        m_StageResources.sky.pipeline = GetDevice()->createGraphicsPipeline(
            pso, framebuffer->getFramebufferInfo());
    }

    dm::affine3 viewToWorld = dm::affine3(m_ViewHandler.view.GetInverseViewMatrix());
    viewToWorld.m_translation = 0.f;
    dm::float4x4 clipToTranslatedWorld =
        m_ViewHandler.view.GetInverseProjectionMatrix(true) * dm::affineToHomogeneous(viewToWorld);

    const Scene::SunSkyState& state = m_Registry.getSunSkyState();

    SkyConstants skyConstants{};
    skyConstants.matClipToTranslatedWorld = clipToTranslatedWorld;
    skyConstants.params                   = state.skyParams;
    skyConstants.params.directionToLight  = dm::normalize(-state.lightDir);

    m_CommandList->writeBuffer(m_StageResources.sky.constantBuffer, &skyConstants, sizeof(skyConstants));

    nvrhi::GraphicsState skyState;
    skyState.pipeline    = m_StageResources.sky.pipeline;
    skyState.framebuffer = framebuffer;
    skyState.viewport    = m_ViewHandler.view.GetViewportState();
    skyState.bindings    = { m_StageResources.sky.bindingSet };
    m_CommandList->setGraphicsState(skyState);
    m_CommandList->draw(nvrhi::DrawArguments().setVertexCount(4));
}

// ===========================================================================
// Main color passes — trunk meshlets, leaf meshlets, and terrain.
// ===========================================================================

void MeshShaderRenderPass::_RenderShadowPass() {}

void MeshShaderRenderPass::_ClearShadowMaps() {
    for (uint32_t c = 0; c < Render::c_NumCascades; ++c) {
        nvrhi::utils::ClearDepthStencilAttachment(
            m_CommandList,
            m_StageResources.shadow.framebuffers[c],
            1.0f, 0);
    }
}

void MeshShaderRenderPass::_RenderTrunkPass(nvrhi::IFramebuffer* framebuffer) {
    if (!m_DispatchMeshSignature || m_StageResources.sceneDraw.bindingSets.empty())
        return;

    const auto& fbInfo = framebuffer->getFramebufferInfo();

    nvrhi::MeshletState meshState;
    meshState.pipeline       = m_StageResources.sceneDraw.pipeline;
    meshState.framebuffer    = framebuffer;
    meshState.indirectParams = m_StageResources.cull.mainDispatchArgsBuffer;
    meshState.viewport.addViewportAndScissorRect(fbInfo.getViewport());

    auto* d3dList = static_cast<ID3D12GraphicsCommandList6*>(
        m_CommandList->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList));
    auto* argBuffer = static_cast<ID3D12Resource*>(
        m_StageResources.cull.mainDispatchArgsBuffer->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource));
    if (!d3dList || !argBuffer) return;

    for (uint32_t texIdx = 0; texIdx < m_StageResources.sceneDraw.bindingSets.size(); texIdx++) {
        meshState.bindings = { m_StageResources.sceneDraw.bindingSets[texIdx] };
        m_CommandList->setMeshletState(meshState);

        uint32_t placeholder = 0;
        m_CommandList->setPushConstants(&placeholder, sizeof(uint32_t));

        for (uint32_t slot = 0; slot < m_NumMainSlots; slot++) {
            if (slot >= m_MainSlotTextureSet.size()) continue;
            if (m_MainSlotTextureSet[slot] != texIdx) continue;

            d3dList->ExecuteIndirect(
                m_DispatchMeshSignature.Get(),
                1,
                argBuffer,
                slot * sizeof(DispatchRecord),
                nullptr,
                0);
        }
    }
}

void MeshShaderRenderPass::_RenderLeavesPass(nvrhi::IFramebuffer* framebuffer) {
    auto& L = m_StageResources.sceneLeaves;
    if (!L.mainPipeline || !L.mainSignature || !L.mainBindingSet
        || !m_StageResources.cull.mainLeafDispatchArgsBuffer)
        return;

    const auto& fbInfo = framebuffer->getFramebufferInfo();

    nvrhi::MeshletState leafMS;
    leafMS.pipeline       = L.mainPipeline;
    leafMS.framebuffer    = framebuffer;
    leafMS.bindings       = { L.mainBindingSet };
    leafMS.indirectParams = m_StageResources.cull.mainLeafDispatchArgsBuffer;
    leafMS.viewport.addViewportAndScissorRect(fbInfo.getViewport());
    m_CommandList->setMeshletState(leafMS);

    uint32_t placeholder = 0;
    m_CommandList->setPushConstants(&placeholder, sizeof(uint32_t));

    auto* d3dList = static_cast<ID3D12GraphicsCommandList6*>(
        m_CommandList->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList));
    auto* leafArgBuf = static_cast<ID3D12Resource*>(
        m_StageResources.cull.mainLeafDispatchArgsBuffer->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource));
    if (d3dList && leafArgBuf) {
        d3dList->ExecuteIndirect(
            L.mainSignature.Get(),
            m_NumMainSlots, leafArgBuf, 0, nullptr, 0);
    }
}

void MeshShaderRenderPass::_RenderTerrainPass(nvrhi::IFramebuffer* framebuffer) {
    auto& T = m_StageResources.sceneTerrain;
    if (T.meshletCount == 0 || !T.amplificationShader || !T.meshShader || !T.pixelShader)
        return;
    // Binding layout/set may be null on the first frame before _EnsureHiZResources
    // creates the Hi-Z texture and triggers _RebuildTerrainBindingSet.
    if (!T.bindingLayout || !T.bindingSet)
        return;

    if (!T.pipeline) {
        nvrhi::MeshletPipelineDesc psoDesc;
        psoDesc.AS = T.amplificationShader;
        psoDesc.MS = T.meshShader;
        psoDesc.PS = T.pixelShader;
        psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
        psoDesc.bindingLayouts = { T.bindingLayout };

        auto& rs = psoDesc.renderState;
        rs.depthStencilState.depthTestEnable  = true;
        rs.depthStencilState.depthWriteEnable = true;
    #if XYLEM_USE_REVERSE_Z
        rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::Greater;
    #else
        rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::Less;
    #endif
        rs.rasterState.cullMode = nvrhi::RasterCullMode::Back;

        T.pipeline = GetDevice()->createMeshletPipeline(psoDesc, framebuffer->getFramebufferInfo());
        if (!T.pipeline) return;
    }

    // Reset survivor counter for this frame.
    m_CommandList->clearBufferUInt(T.visibleCounterBuffer, 0);

    nvrhi::MeshletState ms;
    ms.pipeline    = T.pipeline;
    ms.framebuffer = framebuffer;
    ms.bindings    = { T.bindingSet };
    ms.viewport.addViewportAndScissorRect(framebuffer->getFramebufferInfo().getViewport());
    m_CommandList->setMeshletState(ms);

    const uint32_t numGroups = (T.meshletCount + Render::k_ASGroupSize - 1) / Render::k_ASGroupSize;
    m_CommandList->dispatchMesh(numGroups, 1, 1);
}

// ===========================================================================
// Hemi-octahedral impostors: render side only. Bake + atlas live in
// SharedGPUAssets — this pass binds those shared resources into per-pass
// binding sets that also reference per-pass cull buffers.
// ===========================================================================

bool MeshShaderRenderPass::_InitImpostorPass() {
    auto& imp = m_StageResources.impostor;

    imp.vertexShader = m_ShaderFactory->CreateShader(
        "app/ImpostorRenderPass.hlsl", "impostor_vs", nullptr, nvrhi::ShaderType::Vertex);
    imp.pixelShader = m_ShaderFactory->CreateShader(
        "app/ImpostorRenderPass.hlsl", "impostor_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!imp.vertexShader || !imp.pixelShader) {
        log::error("MeshShaderRenderPass: impostor shaders failed to compile");
        return false;
    }

    nvrhi::BindingLayoutDesc layoutDesc;
    layoutDesc.visibility = nvrhi::ShaderType::All;
    layoutDesc.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(compute_reg::Impostor::kCB_Frame),
        nvrhi::BindingLayoutItem::PushConstants(compute_reg::Impostor::kPushC_AssetIndex, sizeof(uint32_t)),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_Vis),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_Instances),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_SlotOffsets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_CullData),
        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::Impostor::kTex_Albedo),
        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::Impostor::kTex_Normal),
        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::Impostor::kTex_Depth),
        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::Impostor::kTex_ShadowMap),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_AssetDims),
        nvrhi::BindingLayoutItem::Sampler(compute_reg::Impostor::kSampler_Main),
        nvrhi::BindingLayoutItem::Sampler(compute_reg::Impostor::kSampler_Shadow),
        nvrhi::BindingLayoutItem::Sampler(compute_reg::Impostor::kSampler_Depth),
    };
    imp.bindingLayout = GetDevice()->createBindingLayout(layoutDesc);
    if (!imp.bindingLayout) return false;

    imp.sampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
            .setAllFilters(true));
    imp.depthSampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
            .setAllFilters(false));
    return imp.sampler != nullptr && imp.depthSampler != nullptr;
}

bool MeshShaderRenderPass::_InitShadowImpostorPass() {
    auto& sip = m_StageResources.shadowImpostor;

    sip.vertexShader = m_ShaderFactory->CreateShader(
        "app/ShadowImpostorPass.hlsl", "shadow_impostor_vs", nullptr, nvrhi::ShaderType::Vertex);
    sip.pixelShader = m_ShaderFactory->CreateShader(
        "app/ShadowImpostorPass.hlsl", "shadow_impostor_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!sip.vertexShader || !sip.pixelShader) {
        log::error("MeshShaderRenderPass: shadow impostor shaders failed to compile");
        return false;
    }

    nvrhi::BindingLayoutDesc layoutDesc;
    layoutDesc.visibility = nvrhi::ShaderType::All;
    layoutDesc.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(compute_reg::ShadowImpostor::kCB_Frame),
        nvrhi::BindingLayoutItem::PushConstants(compute_reg::ShadowImpostor::kPushC_AssetCascade, sizeof(uint32_t) * 2),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::ShadowImpostor::kSRV_Vis),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::ShadowImpostor::kSRV_Instances),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::ShadowImpostor::kSRV_SlotOffsets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::ShadowImpostor::kSRV_CullData),
        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::ShadowImpostor::kTex_Albedo),
        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::ShadowImpostor::kTex_Depth),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::ShadowImpostor::kSRV_AssetDims),
        nvrhi::BindingLayoutItem::Sampler(compute_reg::ShadowImpostor::kSampler_Main),
        nvrhi::BindingLayoutItem::Sampler(compute_reg::ShadowImpostor::kSampler_Depth),
    };
    sip.bindingLayout = GetDevice()->createBindingLayout(layoutDesc);
    if (!sip.bindingLayout) return false;

    // Reuse the color impostor pass's samplers — same filter/clamp settings work for shadow.
    sip.sampler      = m_StageResources.impostor.sampler;
    sip.depthSampler = m_StageResources.impostor.depthSampler;

    return sip.sampler != nullptr && sip.depthSampler != nullptr;
}

void MeshShaderRenderPass::_RebuildImpostorBindingSets() {
    auto& imp = m_StageResources.impostor;
    if (!imp.bindingLayout) return;
    if (!m_Shared || !m_Shared->impostorAlbedo() || !m_Shared->impostorNormal() || !m_Shared->impostorDepth()) return;
    if (!m_Shared->assetDimsBuffer()) return;
    if (!m_StageResources.shadow.depthTexture) return;
    if (!m_StageResources.cull.impostorVisBuffer || !m_StageResources.cull.impostorSlotOffsetBuffer) return;
    const auto& barkTextures = m_Shared->barkTextures();
    if (barkTextures.empty()) return;

    auto device = GetDevice();
    imp.bindingSets.assign(barkTextures.size(), nullptr);
    for (size_t i = 0; i < barkTextures.size(); i++) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(compute_reg::Impostor::kCB_Frame,
                m_StageResources.frameShared.constantBuffer,
                nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
            nvrhi::BindingSetItem::PushConstants(compute_reg::Impostor::kPushC_AssetIndex, sizeof(uint32_t)),

            nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_Vis,        m_StageResources.cull.impostorVisBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_Instances,  m_StageResources.cull.persistentInstBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_SlotOffsets,m_StageResources.cull.impostorSlotOffsetBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_CullData,   m_StageResources.cull.cullDataBuffer),

            nvrhi::BindingSetItem::Texture_SRV(compute_reg::Impostor::kTex_Albedo,  m_Shared->impostorAlbedo()),
            nvrhi::BindingSetItem::Texture_SRV(compute_reg::Impostor::kTex_Normal,  m_Shared->impostorNormal()),
            nvrhi::BindingSetItem::Texture_SRV(compute_reg::Impostor::kTex_Depth,   m_Shared->impostorDepth(), nvrhi::Format::R32_FLOAT),
            nvrhi::BindingSetItem::Texture_SRV(compute_reg::Impostor::kTex_ShadowMap, m_StageResources.shadow.depthTexture),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_AssetDims, m_Shared->assetDimsBuffer()),

            nvrhi::BindingSetItem::Sampler(compute_reg::Impostor::kSampler_Main,   imp.sampler),
            nvrhi::BindingSetItem::Sampler(compute_reg::Impostor::kSampler_Shadow, m_StageResources.sceneDraw.shadowSampler),
            nvrhi::BindingSetItem::Sampler(compute_reg::Impostor::kSampler_Depth,  imp.depthSampler),
        };
        imp.bindingSets[i] = device->createBindingSet(bsd, imp.bindingLayout);
    }
    imp.pipeline = nullptr;  // resolution-dependent; recreate next render
}

void MeshShaderRenderPass::_RebuildShadowImpostorBindingSets() {
    auto& sip = m_StageResources.shadowImpostor;
    sip.bindingSets.clear();
    if (!sip.bindingLayout) return;
    if (!m_Shared || !m_Shared->impostorAlbedo() || !m_Shared->impostorDepth()) return;
    if (!m_Shared->assetDimsBuffer()) return;
    if (!m_StageResources.cull.shadowImpostorVisBuffer ||
        !m_StageResources.cull.shadowImpostorSlotOffsetBuffer) return;

    nvrhi::BindingSetDesc setDesc;
    setDesc.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(compute_reg::ShadowImpostor::kCB_Frame,
            m_StageResources.frameShared.constantBuffer,
            nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
        nvrhi::BindingSetItem::PushConstants(compute_reg::ShadowImpostor::kPushC_AssetCascade, sizeof(uint32_t) * 2),

        nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::ShadowImpostor::kSRV_Vis,         m_StageResources.cull.shadowImpostorVisBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::ShadowImpostor::kSRV_Instances,   m_StageResources.cull.persistentInstBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::ShadowImpostor::kSRV_SlotOffsets, m_StageResources.cull.shadowImpostorSlotOffsetBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::ShadowImpostor::kSRV_CullData,    m_StageResources.cull.cullDataBuffer),

        nvrhi::BindingSetItem::Texture_SRV(compute_reg::ShadowImpostor::kTex_Albedo, m_Shared->impostorAlbedo()),
        nvrhi::BindingSetItem::Texture_SRV(compute_reg::ShadowImpostor::kTex_Depth,  m_Shared->impostorDepth(), nvrhi::Format::R32_FLOAT),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::ShadowImpostor::kSRV_AssetDims,  m_Shared->assetDimsBuffer()),

        nvrhi::BindingSetItem::Sampler(compute_reg::ShadowImpostor::kSampler_Main,  sip.sampler),
        nvrhi::BindingSetItem::Sampler(compute_reg::ShadowImpostor::kSampler_Depth, sip.depthSampler),
    };
    sip.bindingSets.push_back(GetDevice()->createBindingSet(setDesc, sip.bindingLayout));
    sip.pipeline = nullptr;  // resolution-dependent; recreate next render
}

void MeshShaderRenderPass::_RenderImpostorPass(nvrhi::IFramebuffer* framebuffer) {
    auto& imp = m_StageResources.impostor;
    const auto& assets = m_Registry.getAssets();
    if (assets.empty() || imp.bindingSets.empty()) return;

    const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();

    if (!imp.pipeline) {
        nvrhi::GraphicsPipelineDesc psoDesc;
        psoDesc.VS = imp.vertexShader;
        psoDesc.PS = imp.pixelShader;
        psoDesc.inputLayout = nullptr;
        psoDesc.bindingLayouts = { imp.bindingLayout };
        psoDesc.primType = nvrhi::PrimitiveType::TriangleStrip;
    #if XYLEM_USE_REVERSE_Z
        psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
    #else
        psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    #endif
        psoDesc.renderState.rasterState.setCullNone();
        // Slope-scaled depth bias pushes impostor fragments toward the camera in
        // reverse-Z so they win the z-test against terrain.
        psoDesc.renderState.rasterState
            .setDepthBias(16)
            .setSlopeScaleDepthBias(1.0f);
        imp.pipeline = GetDevice()->createGraphicsPipeline(psoDesc, fbinfo);
    }

    nvrhi::GraphicsState state;
    state.pipeline = imp.pipeline;
    state.framebuffer = framebuffer;
    state.viewport = m_ViewHandler.view.GetViewportState();
    state.indirectParams = m_StageResources.cull.impostorIndirectArgsBuffer;

    for (uint32_t ai = 0; ai < assets.size(); ai++) {
        if (ai >= m_ImpostorMaxSlotCounts.size() || m_ImpostorMaxSlotCounts[ai] == 0) continue;

        uint32_t texIdx = std::min<uint32_t>(assets[ai].textureSetIdx,
            static_cast<uint32_t>(imp.bindingSets.size() - 1));
        state.bindings = { imp.bindingSets[texIdx] };
        m_CommandList->setGraphicsState(state);
        m_CommandList->setPushConstants(&ai, sizeof(ai));
        m_CommandList->drawIndirect(ai * sizeof(nvrhi::DrawIndirectArguments));
    }
}

void MeshShaderRenderPass::_RenderShadowImpostorPass(uint32_t cascade) {
    auto& sip = m_StageResources.shadowImpostor;
    const auto& assets = m_Registry.getAssets();
    if (assets.empty() || sip.bindingSets.empty()) return;

    auto framebuffer = m_StageResources.shadow.framebuffers[cascade];
    if (!framebuffer) return;

    const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();

    if (!sip.pipeline) {
        nvrhi::GraphicsPipelineDesc psoDesc;
        psoDesc.VS = sip.vertexShader;
        psoDesc.PS = sip.pixelShader;
        psoDesc.inputLayout = nullptr;
        psoDesc.bindingLayouts = { sip.bindingLayout };
        psoDesc.primType = nvrhi::PrimitiveType::TriangleStrip;
        // Shadow framebuffer uses standard Z (cleared to 1.0, smaller depth = closer to light),
        // matching the meshlet shadow pipeline at line 1379. Not reverse-Z.
        psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
        psoDesc.renderState.depthStencilState.setDepthWriteEnable(true);
        psoDesc.renderState.rasterState.setCullNone();
        sip.pipeline = GetDevice()->createGraphicsPipeline(psoDesc, fbinfo);
    }

    nvrhi::GraphicsState state;
    state.pipeline = sip.pipeline;
    state.framebuffer = framebuffer;

    nvrhi::ViewportState vp;
    vp.addViewportAndScissorRect(nvrhi::Viewport(
        (float)fbinfo.width, (float)fbinfo.height));
    state.viewport = vp;

    state.indirectParams = m_StageResources.cull.shadowImpostorIndirectArgsBuffer;
    state.bindings = { sip.bindingSets[0] };
    m_CommandList->setGraphicsState(state);

    const uint32_t numAssets = static_cast<uint32_t>(assets.size());
    const uint32_t nCasc = XYLEM_NUM_CASCADES;
    for (uint32_t ai = 0; ai < numAssets; ai++) {
        const uint32_t slot = ai * nCasc + cascade;
        if (slot >= m_ShadowImpostorMaxSlotCounts.size() ||
            m_ShadowImpostorMaxSlotCounts[slot] == 0) continue;

        uint32_t pc[2] = { ai, cascade };
        m_CommandList->setPushConstants(pc, sizeof(pc));
        m_CommandList->drawIndirect(slot * sizeof(nvrhi::DrawIndirectArguments));
    }
}

