#include "include/TraditionalRenderPass.hpp"
#include "include/Globals.hpp"
#include "include/macros.h"
#include "include/Terrain.hpp"
#include "include/shaders/ShaderContracts.hpp"
#include "include/frame/FrameLifecycle.hpp"

#include <nvrhi/utils.h>
#include <donut/app/imgui_renderer.h>
#include <donut/app/Timer.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/core/math/math.h>
#include <donut/core/json.h>

using namespace donut::math;
#include <donut/shaders/sky_cb.h>

#include <filesystem>

using namespace Xylem;
namespace shader_cb = Xylem::shader::cb;
namespace traditional_reg = Xylem::shader::reg::Traditional;
namespace compute_reg = Xylem::shader::reg::Compute;

// ===========================================================================
// GPU asset upload helpers
// ===========================================================================

void TraditionalRenderPass::_UploadAsset(
    const TreeAssetDef& assetDef,
    GPUTreeAsset& gpuAsset,
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList)
{
    gpuAsset.textureSetIdx = assetDef.textureSetIdx;
    gpuAsset.lods.resize(assetDef.lods.size());

    nvrhi::BufferDesc vDesc;
    vDesc.isVertexBuffer = true;
    vDesc.initialState   = nvrhi::ResourceStates::CopyDest;

    nvrhi::BufferDesc iDesc;
    iDesc.isIndexBuffer = true;
    iDesc.initialState  = nvrhi::ResourceStates::CopyDest;

    for (size_t j = 0; j < assetDef.lods.size(); j++) {
        const auto& lodDef = assetDef.lods[j];
        std::string lodSuffix = "_LOD" + std::to_string(j);

        auto uploadVB = [&](const auto& data, const char* suffix, nvrhi::BufferHandle& out) {
            vDesc.debugName = "VB_" + assetDef.name + lodSuffix + suffix;
            vDesc.byteSize  = data.size() * sizeof(data[0]);
            out = device->createBuffer(vDesc);
            commandList->beginTrackingBufferState(out, nvrhi::ResourceStates::CopyDest);
            commandList->writeBuffer(out, data.data(), vDesc.byteSize);
            commandList->setPermanentBufferState(out, nvrhi::ResourceStates::VertexBuffer);
        };

        uploadVB(lodDef.positions,  "_pos",   gpuAsset.lods[j].vbs.position);
        uploadVB(lodDef.normals,    "_nor",   gpuAsset.lods[j].vbs.normal);
        uploadVB(lodDef.tangents,   "_tan",   gpuAsset.lods[j].vbs.tangent);
        uploadVB(lodDef.bitangents, "_bitan", gpuAsset.lods[j].vbs.bitangent);
        uploadVB(lodDef.uvs,        "_uv",    gpuAsset.lods[j].vbs.uv);

        iDesc.debugName = "IB_" + assetDef.name + lodSuffix;
        iDesc.byteSize  = lodDef.indices.size() * sizeof(uint32_t);
        auto iBuf = device->createBuffer(iDesc);
        commandList->beginTrackingBufferState(iBuf, nvrhi::ResourceStates::CopyDest);
        commandList->writeBuffer(iBuf, lodDef.indices.data(), iDesc.byteSize);
        commandList->setPermanentBufferState(iBuf, nvrhi::ResourceStates::IndexBuffer);

        gpuAsset.lods[j].indexBuffer    = iBuf;
        gpuAsset.lods[j].indexCount     = static_cast<uint32_t>(lodDef.indices.size());
        gpuAsset.lods[j].radialSegments = lodDef.radialSegments;
        gpuAsset.lods[j].bbox           = lodDef.bbox;
    }
}

void TraditionalRenderPass::_UploadAllAssets(nvrhi::IDevice* device, nvrhi::ICommandList* commandList) {
    const auto& assets = m_Registry.getAssets();
    m_GPUAssets.resize(assets.size());
    m_AssetIdToGPUIndex.clear();

    for (size_t i = 0; i < assets.size(); i++) {
        _UploadAsset(assets[i], m_GPUAssets[i], device, commandList);
        m_AssetIdToGPUIndex[assets[i].id] = i;
    }
}

void TraditionalRenderPass::_RebuildInstanceBuffers() {
    uint32_t totalInstances = std::max<uint32_t>(1, m_Registry.totalInstanceCount());

    m_StageResources.sceneTreeStage.instanceBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(totalInstances * sizeof(Render::InstanceBufferEntry))
            .setStructStride(sizeof(Render::InstanceBufferEntry))
            .setDebugName("TreeInstanceBuffer")
            .setIsVertexBuffer(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
    );

    m_StageResources.shadowStage.instanceBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(totalInstances * sizeof(Render::InstanceBufferEntry))
            .setStructStride(sizeof(Render::InstanceBufferEntry))
            .setDebugName("ShadowInstanceBuffer")
            .setIsVertexBuffer(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
    );

    m_VisibleInstanceBuffer.resize(totalInstances);
    for (auto& csd : m_CascadeShadowData)
        csd.instanceBuffer.resize(totalInstances);

    _BuildImpostorSlotLayout();
    _RebuildImpostorBuffers();
}

void TraditionalRenderPass::_BuildImpostorSlotLayout() {
    const uint32_t numAssets = static_cast<uint32_t>(m_GPUAssets.size());
    std::vector<uint32_t> livePerAsset(std::max(1u, numAssets), 0);

    const auto& regions = m_Registry.getRegions();
    for (const auto& region : regions) {
        for (const auto& inst : region.instances) {
            auto it = m_AssetIdToGPUIndex.find(inst.assetId);
            if (it == m_AssetIdToGPUIndex.end()) continue;

            uint32_t gpuIdx = static_cast<uint32_t>(it->second);
            if (gpuIdx < numAssets)
                livePerAsset[gpuIdx]++;
        }
    }

    m_ImpostorSlotOffsets.assign(std::max(1u, numAssets), 0);
    m_ImpostorMaxSlotCounts.assign(std::max(1u, numAssets), 0);
    m_ImpostorVisBufferSize = 0;

    for (uint32_t ai = 0; ai < numAssets; ai++) {
        m_ImpostorSlotOffsets[ai]   = m_ImpostorVisBufferSize;
        m_ImpostorMaxSlotCounts[ai] = livePerAsset[ai];
        m_ImpostorVisBufferSize    += livePerAsset[ai];
    }
    m_ImpostorVisBufferSize = std::max(1u, m_ImpostorVisBufferSize);
}

void TraditionalRenderPass::_RebuildImpostorBuffers() {
    auto* device = GetDevice();
    const uint32_t numAssets = std::max(1u, static_cast<uint32_t>(m_GPUAssets.size()));
    const uint32_t visCapacity = std::max(1u, m_ImpostorVisBufferSize);

    m_StageResources.impostorStage.instanceBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(visCapacity * sizeof(Render::InstanceBufferEntry))
            .setStructStride(sizeof(Render::InstanceBufferEntry))
            .setDebugName("TraditionalImpostorInstanceBuffer")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest));

    m_StageResources.impostorStage.cullDataBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(visCapacity * sizeof(Render::CullInstanceData))
            .setStructStride(sizeof(Render::CullInstanceData))
            .setDebugName("TraditionalImpostorCullDataBuffer")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest));

    m_StageResources.impostorStage.visBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(visCapacity * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("TraditionalImpostorVisBuffer")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest));

    m_StageResources.impostorStage.slotOffsetBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(numAssets * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("TraditionalImpostorSlotOffsetBuffer")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest));

    m_ImpostorInstanceStaging.assign(visCapacity, {});
    m_ImpostorCullDataStaging.assign(visCapacity, {});
    m_ImpostorVisStaging.resize(visCapacity);
    for (uint32_t i = 0; i < visCapacity; i++)
        m_ImpostorVisStaging[i] = i;

    m_ImpostorCounts.assign(numAssets, 0);
    m_ImpostorWriteOffsets.assign(numAssets, 0);
    m_VisibleImpostorReferences.clear();
    m_VisibleImpostorReferences.reserve(m_Registry.totalInstanceCount());
}

void TraditionalRenderPass::_RebuildBindingSets() {
    const auto& barkTextures = m_Shared->barkTextures();
    m_StageResources.sceneTreeStage.bindingSets.resize(barkTextures.size());

    nvrhi::BindingLayoutDesc bld;
    bld.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(traditional_reg::Tree::kCB_Frame),
        nvrhi::BindingLayoutItem::Sampler(traditional_reg::Tree::kSampler_Main),
        nvrhi::BindingLayoutItem::Sampler(traditional_reg::Tree::kSampler_Shadow),
        nvrhi::BindingLayoutItem::Texture_SRV(traditional_reg::Tree::kTex_ShadowMap),
        
        // changes per bark texture
        nvrhi::BindingLayoutItem::Texture_SRV(traditional_reg::Tree::kTex_Diffuse),
        nvrhi::BindingLayoutItem::Texture_SRV(traditional_reg::Tree::kTex_NormalMap),
    };

    m_StageResources.sceneTreeStage.bindingLayout = GetDevice()->createBindingLayout(bld);

    for (size_t i = 0; i < barkTextures.size(); i++) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(traditional_reg::Tree::kCB_Frame, m_StageResources.frameShared.constantBuffer, nvrhi::BufferRange(0, shader_cb::kFrameSize)),
            nvrhi::BindingSetItem::Sampler(traditional_reg::Tree::kSampler_Main, m_Shared->barkSampler()),
            nvrhi::BindingSetItem::Sampler(traditional_reg::Tree::kSampler_Shadow, m_StageResources.shadowStage.comparisonSampler),
            nvrhi::BindingSetItem::Texture_SRV(traditional_reg::Tree::kTex_ShadowMap, m_StageResources.shadowStage.depthTexture),
            
            nvrhi::BindingSetItem::Texture_SRV(traditional_reg::Tree::kTex_Diffuse, barkTextures[i].diffuse),
            nvrhi::BindingSetItem::Texture_SRV(traditional_reg::Tree::kTex_NormalMap, barkTextures[i].normalMap),
        };
        
        m_StageResources.sceneTreeStage.bindingSets[i] = GetDevice()->createBindingSet(bsd, m_StageResources.sceneTreeStage.bindingLayout);
    }

    nvrhi::BindingSetDesc leafBSD;
    leafBSD.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_StageResources.frameShared.constantBuffer, nvrhi::BufferRange(0, shader_cb::kFrameSize)),
        nvrhi::BindingSetItem::PushConstants(1, sizeof(uint32_t) * 2),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_Shared->leafInstancesBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_Shared->leafSlotsBuffer()),
        nvrhi::BindingSetItem::Texture_SRV(2, m_StageResources.shadowStage.depthTexture),
        nvrhi::BindingSetItem::Sampler(0, m_StageResources.shadowStage.comparisonSampler),
    };
    m_StageResources.leafStage.bindingSet =
        GetDevice()->createBindingSet(leafBSD, m_StageResources.leafStage.bindingLayout);

    nvrhi::BindingSetDesc leafShadowBSD;
    leafShadowBSD.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_StageResources.frameShared.constantBuffer, nvrhi::BufferRange(0, shader_cb::kFrameSize)),
        nvrhi::BindingSetItem::PushConstants(1, sizeof(uint32_t) * 2),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_Shared->leafInstancesBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_Shared->leafShadowSlotsBuffer()),
    };
    m_StageResources.leafStage.shadowBindingSet =
        GetDevice()->createBindingSet(leafShadowBSD, m_StageResources.leafStage.shadowBindingLayout);

    m_StageResources.impostorStage.bindingSets.clear();
    if (!m_StageResources.impostorStage.bindingLayout
        || !m_StageResources.impostorStage.instanceBuffer
        || !m_StageResources.impostorStage.cullDataBuffer
        || !m_StageResources.impostorStage.visBuffer
        || !m_StageResources.impostorStage.slotOffsetBuffer
        || !m_StageResources.shadowStage.depthTexture
        || !m_Shared->impostorAlbedo()
        || !m_Shared->impostorNormal()
        || !m_Shared->impostorDepth()
        || !m_Shared->assetDimsBuffer()
        || barkTextures.empty())
    {
        return;
    }

    m_StageResources.impostorStage.bindingSets.resize(barkTextures.size());
    for (size_t i = 0; i < barkTextures.size(); i++) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(
                compute_reg::Impostor::kCB_Frame,
                m_StageResources.frameShared.constantBuffer,
                nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
            nvrhi::BindingSetItem::PushConstants(compute_reg::Impostor::kPushC_AssetIndex, sizeof(uint32_t)),

            nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_Vis, m_StageResources.impostorStage.visBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_Instances, m_StageResources.impostorStage.instanceBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_SlotOffsets, m_StageResources.impostorStage.slotOffsetBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_CullData, m_StageResources.impostorStage.cullDataBuffer),

            nvrhi::BindingSetItem::Texture_SRV(compute_reg::Impostor::kTex_Albedo, m_Shared->impostorAlbedo()),
            nvrhi::BindingSetItem::Texture_SRV(compute_reg::Impostor::kTex_Normal, m_Shared->impostorNormal()),
            nvrhi::BindingSetItem::Texture_SRV(compute_reg::Impostor::kTex_Depth, m_Shared->impostorDepth(), nvrhi::Format::R32_FLOAT),
            nvrhi::BindingSetItem::Texture_SRV(compute_reg::Impostor::kTex_ShadowMap, m_StageResources.shadowStage.depthTexture),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_AssetDims, m_Shared->assetDimsBuffer()),

            nvrhi::BindingSetItem::Sampler(compute_reg::Impostor::kSampler_Main, m_StageResources.impostorStage.sampler),
            nvrhi::BindingSetItem::Sampler(compute_reg::Impostor::kSampler_Shadow, m_StageResources.shadowStage.comparisonSampler),
            nvrhi::BindingSetItem::Sampler(compute_reg::Impostor::kSampler_Depth, m_StageResources.impostorStage.depthSampler),
        };
        m_StageResources.impostorStage.bindingSets[i] = GetDevice()->createBindingSet(
            bsd, m_StageResources.impostorStage.bindingLayout);
    }
}

// ===========================================================================
// Init
// ===========================================================================

bool TraditionalRenderPass::Init() {
    if (!_InitShared())                                 return false;
    if (!_InitShadowPass())                             return false;
    if (!_InitTreePass())                               return false;
    if (!_InitLeafPass())                               return false;
    if (!_InitImpostorPass())                           return false;
    if (!_InitSkyPass())                                return false;
    
    if (!_InitDebug())                                  return false;

    // Persistent render command list for per-frame drawing.
    m_CommandList = GetDevice()->createCommandList();

    // _InitTimerQueries();

    // GPU resources
    if (!LoadResources())                               return false;
    return true;
}

bool TraditionalRenderPass::LoadResources() {
    nvrhi::CommandListHandle initCL = GetDevice()->createCommandList();
    initCL->open();

    // Upload per-asset/LOD vertex and index buffers from registry CPU data.
    _UploadAllAssets(GetDevice(), initCL);

    // _InitTerrainPass uploads terrain VB/IB from m_Registry.getTerrain(); the
    // terrain mesh is part of the scene, so this belongs in LoadResources.
    if (!_InitTerrainPass(initCL)) {
        initCL->close();
        return false;
    }

    initCL->close();
    GetDevice()->executeCommandList(initCL);

    // Rebuild scene-sized GPU instance buffers and binding sets.
    _RebuildInstanceBuffers();
    _RebuildBindingSets();

    uint32_t totalInstances = m_Registry.totalInstanceCount();
    m_UI.totalInstanceCount = totalInstances;

    m_VisibleInstanceReferences.clear();
    m_VisibleInstanceReferences.reserve(totalInstances);
    m_VisibleImpostorReferences.clear();
    m_VisibleImpostorReferences.reserve(totalInstances);

    size_t numAssets = m_GPUAssets.size();
    size_t numLods   = m_Registry.getLodSegments().size();

    m_InstanceCounts.assign(numAssets, std::vector<uint32_t>(numLods, 0));
    m_InstanceOffsets.assign(numAssets, std::vector<uint32_t>(numLods, 0));
    m_ImpostorCounts.assign(std::max<size_t>(1, numAssets), 0);
    m_ImpostorWriteOffsets.assign(std::max<size_t>(1, numAssets), 0);

    m_VisibleInstanceBuffer.assign(std::max<uint32_t>(1, totalInstances), {});
    m_DrawCmds.clear();
    m_DrawCmds.reserve(numAssets * numLods);
    m_ImpostorDrawCallCount = 0;

    for (auto& csd : m_CascadeShadowData) {
        csd.instanceBuffer.assign(std::max<uint32_t>(1, totalInstances), {});
        csd.drawCmds.clear();
        csd.drawCmds.reserve(numAssets);
    }

    return true;
}

// ===========================================================================
// Animate
// ===========================================================================
// The dirty cycle is driven by RenderOrchestrator so SharedGPUAssets re-bakes
// the impostor atlas before per-pass binding-set rebuilds run.

void TraditionalRenderPass::Animate(float /*seconds*/) {
    GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
}

// ===========================================================================
// Hot-reload callbacks
// ===========================================================================

void TraditionalRenderPass::onAssetsDirty(const std::vector<size_t>& dirtyAssetIndices) {
    nvrhi::CommandListHandle cl = GetDevice()->createCommandList();
    cl->open();

    const auto& assets = m_Registry.getAssets();

    // If the GPU asset count doesn't match the registry (asset was removed),
    // re-upload everything from scratch so phantom entries are cleared.
    if (m_GPUAssets.size() != assets.size()) {
        _UploadAllAssets(GetDevice(), cl);
    } else {
        for (size_t idx : dirtyAssetIndices) {
            const auto& assetDef = assets[idx];
            auto it = m_AssetIdToGPUIndex.find(assetDef.id);
            if (it != m_AssetIdToGPUIndex.end()) {
                _UploadAsset(assetDef, m_GPUAssets[it->second], GetDevice(), cl);
            } else {
                GPUTreeAsset gpuAsset;
                _UploadAsset(assetDef, gpuAsset, GetDevice(), cl);
                m_AssetIdToGPUIndex[assetDef.id] = m_GPUAssets.size();
                m_GPUAssets.push_back(std::move(gpuAsset));
            }
        }
    }

    cl->close();
    GetDevice()->executeCommandList(cl);

    // Resize per-asset working arrays
    size_t numAssets = m_GPUAssets.size();
    size_t numLods   = m_Registry.getLodSegments().size();
    m_InstanceCounts.resize(numAssets, std::vector<uint32_t>(numLods, 0));
    m_InstanceOffsets.resize(numAssets, std::vector<uint32_t>(numLods, 0));

    _BuildImpostorSlotLayout();
    _RebuildImpostorBuffers();
    _RebuildBindingSets();
}

void TraditionalRenderPass::onRegionsDirty(const std::vector<size_t>& /*dirtyRegionIndices*/) {
    _RebuildInstanceBuffers();
    _RebuildBindingSets();

    uint32_t newTotal = m_Registry.totalInstanceCount();
    m_UI.totalInstanceCount = newTotal;
    m_VisibleInstanceReferences.reserve(newTotal);
}

// ===========================================================================
// Render
// ===========================================================================

void TraditionalRenderPass::Render(nvrhi::IFramebuffer* framebuffer) {
    m_UI.shadowMapTexture        = m_StageResources.shadowStage.depthTexture.Get();
    m_UI.selectedCascadeTexture  = m_StageResources.shadowStage.debugSelectedCascadeTexture.Get();
    m_UI.hizMipTextures.clear();

    app::HiResTimer cpuTimer;
    cpuTimer.Start();

    const auto& fbInfo = framebuffer->getFramebufferInfo();
    
    { // update viewProj
        // if resized
        if (!m_StageResources.sceneTreeStage.pipeline) {
            frame::UpdateProjectionAndViewport(m_ViewHandler, fbInfo);
        }

        m_ViewHandler.view.SetViewMatrix(m_ViewHandler.camera.GetWorldToViewMatrix());
        m_ViewHandler.view.UpdateCache();
    }

    m_CommandList->open();

    // m_CommandList->beginTimerQuery(m_GpuTimers[m_NextTimerIdx]);

    nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));
    #if XYLEM_USE_REVERSE_Z
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer, 0.f, 0);
    #else
    float farPlaneDistance = m_ViewHandler.view.GetProjectionFrustum().farPlane().distance;
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer,
        farPlaneDistance, 0);
    #endif

    frame::ComputeCascades(
        m_ViewHandler,
        m_Registry,
        m_ViewHandler.view.GetAspectRatio(),
        m_ShadowRes,
        m_UI.pssmLambda);

    Render::CullConstantBufferEntry constants{};
    frame::FillCommonFrameConstants(constants, m_ViewHandler, m_Registry.getSunDirection());

    constants.viewFrustum  = m_ViewHandler.view.GetViewFrustum();
    constants.worldToLight = dm::affineToHomogeneous(m_ViewHandler.worldToLight);
    for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
        const dm::box3& cBbox = m_ViewHandler.cascades[c].shadowCasterBboxLS;
        if (cBbox.isempty()) {
            constants.shadowCasterMinLS[c] = dm::float4(1e30f, 1e30f, 1e30f, 0.f);
            constants.shadowCasterMaxLS[c] = dm::float4(-1e30f, -1e30f, -1e30f, 0.f);
        } else {
            constants.shadowCasterMinLS[c] = dm::float4(cBbox.m_mins, 0.f);
            constants.shadowCasterMaxLS[c] = dm::float4(cBbox.m_maxs, 0.f);
        }
    }

    constants.cameraPos     = m_ViewHandler.camera.GetPosition();
    constants.numRegions    = static_cast<uint32_t>(m_Registry.getRegions().size());
    constants.totalCapacity = m_Registry.totalInstanceCount();
    constants.numLods       = static_cast<uint32_t>(m_Registry.getLodSegments().size());
    const auto& lodDistances = m_Registry.getLodDistances();
    for (uint32_t i = 0; i < constants.numLods && i < lodDistances.size(); i++)
        constants.lodDistances[i] = lodDistances[i];

    constants.hizDimensions = dm::float2(
        static_cast<float>(fbInfo.width),
        static_cast<float>(fbInfo.height));
    constants.maxHiZMip = 0.f;
    constants.hizEnabled = 0u;
    constants.impostorAlphaClip = m_UI.impostorAlphaClip;

    m_CommandList->writeBuffer(m_StageResources.frameShared.constantBuffer, &constants, shader_cb::kCullFrameSize);

    m_CommandList->beginMarker("Draw");

    m_CommandList->beginMarker("Sky");
    _RenderSkyPass(framebuffer);
    m_CommandList->endMarker();

    m_CommandList->beginMarker("Shadow");
    _RenderShadowPass();
    m_CommandList->endMarker();

    if (m_UI.showShadowMap) {
        uint32_t cascade = std::clamp(m_UI.selectedCascade, 0,
                                      static_cast<int>(Render::c_NumCascades) - 1);
        m_CommandList->copyTexture(
            m_StageResources.shadowStage.debugSelectedCascadeTexture, nvrhi::TextureSlice(),
            m_StageResources.shadowStage.depthTexture, nvrhi::TextureSlice().setArraySlice(cascade));
    }

    m_CommandList->beginMarker("Scene");
    _RenderScenePass(framebuffer);
    m_CommandList->endMarker();

    m_CommandList->beginMarker("Impostors");
    _RenderImpostorPass(framebuffer);
    m_CommandList->endMarker();

    m_CommandList->endMarker(); // Draw

    if (m_UI.showImpostorAtlas && m_Shared)
        m_Shared->CopySelectedImpostorDebugAtlases(m_CommandList, m_UI.impostorSelectedAsset);

    // m_CommandList->endTimerQuery(m_GpuTimers[m_NextTimerIdx]);
    m_CommandList->close();
    GetDevice()->executeCommandList(m_CommandList);

    // int prevIdx = (m_NextTimerIdx + m_QueuedFrames - 1) % m_QueuedFrames;
    // if (GetDevice()->pollTimerQuery(m_GpuTimers[prevIdx]))
    //     m_UI.gpuFrameTimeMs = GetDevice()->getTimerQueryTime(m_GpuTimers[prevIdx]) * 1000.0f;

    // m_NextTimerIdx = (m_NextTimerIdx + 1) % m_QueuedFrames;

    const uint32_t meshVisibleCount = static_cast<uint32_t>(m_VisibleInstanceReferences.size());
    const uint32_t impostorVisibleCount = static_cast<uint32_t>(m_VisibleImpostorReferences.size());
    const uint32_t visibleTotal = meshVisibleCount + impostorVisibleCount;

    m_UI.totalInstanceCount   = m_Registry.totalInstanceCount();
    m_UI.visibleInstanceCount = visibleTotal;
    m_UI.impostorVisibleCount = impostorVisibleCount;
    m_UI.culledInstanceCount  = (visibleTotal <= m_UI.totalInstanceCount)
        ? m_UI.totalInstanceCount - visibleTotal : 0;
    m_UI.drawCallCount        = static_cast<uint32_t>(m_DrawCmds.size()) + m_ImpostorDrawCallCount;

    m_UI.shadowVisibleCount   = m_TotalShadowInstancesDrawn;
    m_UI.shadowCulledCount    = m_UI.totalInstanceCount - m_TotalShadowInstancesDrawn;

    m_VisibleInstanceReferences.clear();
    m_VisibleImpostorReferences.clear();

    cpuTimer.Stop();
    m_UI.cpuRenderTimeMs = (float)cpuTimer.Milliseconds();
}

void TraditionalRenderPass::_RenderSkyPass(nvrhi::IFramebuffer* framebuffer) {
    if (!m_StageResources.skyStage.pipeline) {
        nvrhi::GraphicsPipelineDesc pso;
        pso.VS             = m_StageResources.skyStage.vertexShader;
        pso.PS             = m_StageResources.skyStage.pixelShader;
        pso.bindingLayouts = { m_StageResources.skyStage.bindingLayout };
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
        m_StageResources.skyStage.pipeline = GetDevice()->createGraphicsPipeline(
            pso, framebuffer->getFramebufferInfo());
    }

    dm::affine3 viewToWorld = dm::affine3(m_ViewHandler.view.GetInverseViewMatrix());
    viewToWorld.m_translation = 0.f;
    dm::float4x4 clipToTranslatedWorld =
        m_ViewHandler.view.GetInverseProjectionMatrix(true) * dm::affineToHomogeneous(viewToWorld);

    SkyConstants skyConstants{};
    skyConstants.matClipToTranslatedWorld = clipToTranslatedWorld;

    auto& p         = skyConstants.params;
    p.directionToLight  = dm::normalize(-m_Registry.getSunDirection());
    p.angularSizeOfLight = dm::radians(1.0f);
    p.lightColor        = dm::float3(100.f, 98.f, 90.f);
    p.glowSize          = dm::radians(5.f);
    p.skyColor          = dm::float3(0.017f, 0.037f, 0.065f);
    p.glowIntensity     = 0.1f;
    p.horizonColor      = dm::float3(0.050f, 0.070f, 0.092f);
    p.horizonSize       = dm::radians(30.f);
    p.groundColor       = dm::float3(0.062f, 0.059f, 0.055f);
    p.glowSharpness     = 4.f;
    p.directionUp       = dm::float3(0.f, 1.f, 0.f);

    m_CommandList->writeBuffer(m_StageResources.skyStage.constantBuffer, &skyConstants, sizeof(skyConstants));

    nvrhi::GraphicsState skyState;
    skyState.pipeline    = m_StageResources.skyStage.pipeline;
    skyState.framebuffer = framebuffer;
    skyState.viewport    = m_ViewHandler.view.GetViewportState();
    skyState.bindings    = { m_StageResources.skyStage.bindingSet };
    m_CommandList->setGraphicsState(skyState);
    m_CommandList->draw(nvrhi::DrawArguments().setVertexCount(4));
}

void TraditionalRenderPass::_RenderShadowPass() {
    const dm::affine3& worldToLight = m_ViewHandler.worldToLight;
    const auto& lodSegments = m_Registry.getLodSegments();
    uint32_t lodIndex = static_cast<uint32_t>(lodSegments.size() - 1);

    const auto& regions = m_Registry.getRegions();

    m_TotalShadowInstancesDrawn = 0;
    std::vector<uint32_t> shadowCounts[Render::c_NumCascades];
    for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
        m_CascadeShadowData[c].visibleRefs.clear();
        m_CascadeShadowData[c].drawCmds.clear();
        shadowCounts[c].assign(m_GPUAssets.size(), 0);
    }

    if (m_ViewHandler.shadowCasterBboxLS.isempty()) {
        return;
    }

    // Create shadow pipelines on demand (shared across all cascades)
    if (!m_StageResources.shadowStage.treePipeline) {
        nvrhi::GraphicsPipelineDesc pso;
        pso.VS             = m_StageResources.shadowStage.treeVS;
        pso.inputLayout    = m_StageResources.shadowStage.treeInputLayout;
        pso.bindingLayouts = { m_StageResources.shadowStage.bindingLayout };
        pso.primType       = nvrhi::PrimitiveType::TriangleList;
        pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
        // cull=none so flat leaves cast shadows regardless of orientation. Bumped slope-bias
        // a touch since we no longer skip front faces in the shadow pass — keeps trunks
        // from self-shadowing into Peter-Pan.
        pso.renderState.rasterState.setCullNone();
        pso.renderState.rasterState.depthBias            = 2;
        pso.renderState.rasterState.slopeScaledDepthBias = 2.5f;
        m_StageResources.shadowStage.treePipeline = GetDevice()->createGraphicsPipeline(
            pso, m_StageResources.shadowStage.framebuffers[0]->getFramebufferInfo());
    }
    if (!m_StageResources.leafStage.shadowPipeline) {
        nvrhi::GraphicsPipelineDesc pso;
        pso.VS = m_StageResources.leafStage.shadowVS;
        pso.inputLayout = m_StageResources.leafStage.shadowInputLayout;
        pso.bindingLayouts = { m_StageResources.leafStage.shadowBindingLayout };
        pso.primType = nvrhi::PrimitiveType::TriangleList;
        pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
        pso.renderState.rasterState.setCullNone();
        pso.renderState.rasterState.depthBias = 2;
        pso.renderState.rasterState.slopeScaledDepthBias = 2.5f;
        m_StageResources.leafStage.shadowPipeline = GetDevice()->createGraphicsPipeline(
            pso, m_StageResources.shadowStage.framebuffers[0]->getFramebufferInfo());
    }
    if (!m_StageResources.shadowStage.terrainPipeline && m_StageResources.sceneTerrainStage.indexCount > 0) {
        nvrhi::GraphicsPipelineDesc pso;
        pso.VS             = m_StageResources.shadowStage.terrainVS;
        pso.inputLayout    = m_StageResources.shadowStage.terrainInputLayout;
        pso.bindingLayouts = { m_StageResources.shadowStage.bindingLayout };
        pso.primType       = nvrhi::PrimitiveType::TriangleList;
        pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
        pso.renderState.rasterState.setCullBack();
        pso.renderState.rasterState.depthBias            = 2;
        pso.renderState.rasterState.slopeScaledDepthBias = 2.0f;
        m_StageResources.shadowStage.terrainPipeline = GetDevice()->createGraphicsPipeline(
            pso, m_StageResources.shadowStage.framebuffers[0]->getFramebufferInfo());
    }

    nvrhi::Viewport shadowVP(static_cast<float>(m_ShadowRes), static_cast<float>(m_ShadowRes));
    nvrhi::ViewportState shadowVPState;
    shadowVPState.addViewportAndScissorRect(shadowVP);

    const auto* terrainPtr = m_Registry.getTerrain();
    auto& cascades = m_ViewHandler.cascades;

    // regions culls against cascades, then instances
    for (uint32_t ri = 0; ri < regions.size(); ri++) {
        const auto& region = regions[ri];
        dm::box3 regionLS = region.cullBox * worldToLight;

        bool frustumIntersects = false;
        for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
            auto& cBbox = cascades[c].shadowCasterBboxLS; 
            if (cBbox.isempty()) continue;
            if (regionLS.intersects(cBbox)) {
                frustumIntersects = true;
                break;
            }
        }
        // none of the cascades hit region
        if (!frustumIntersects) {
            continue;
        }

        for (uint32_t ii = 0; ii < region.instances.size(); ii++) {
            const auto& inst = region.instances[ii];

            const auto* assetDef = m_Registry.findAsset(inst.assetId);
            if (!assetDef || !assetDef->visible) continue;

            // imgui visibility check -- remove?
            auto visIt = region.assetVisible.find(inst.assetId);
            if (visIt != region.assetVisible.end() && !visIt->second) continue;

            auto it = m_AssetIdToGPUIndex.find(inst.assetId);
            if (it == m_AssetIdToGPUIndex.end()) continue;
            uint32_t gpuIdx = static_cast<uint32_t>(it->second);

            dm::box3 instanceLS = inst.bbox * worldToLight;

            bool drawnYet = false;
            for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
                auto& cBbox = cascades[c].shadowCasterBboxLS;

                if (cBbox.isempty()) continue;
                if (!instanceLS.intersects(cBbox)) continue;
                if (!drawnYet) {
                    m_TotalShadowInstancesDrawn++;
                    drawnYet = true;
                }
                m_CascadeShadowData[c].visibleRefs.push_back({ ri, ii, gpuIdx, lodIndex });
                shadowCounts[c][gpuIdx]++;
            }
        }
    }

    // Build draw commands, fill instance buffers, and render per cascade
    for (uint32_t cascade = 0; cascade < Render::c_NumCascades; cascade++) {
        auto& csd = m_CascadeShadowData[cascade];

        // Build draw commands for this cascade
        std::vector<uint32_t> shadowWriteOff(m_GPUAssets.size(), 0);
        uint32_t shadowOffset = 0;
        for (uint32_t ai = 0; ai < shadowCounts[cascade].size(); ai++) {
            if (shadowCounts[cascade][ai] == 0) continue;
            const auto& lod = m_GPUAssets[ai].lods[lodIndex];
            csd.drawCmds.push_back({
                lod.vbs.position,
                lod.indexBuffer,
                nvrhi::DrawArguments()
                    .setVertexCount(lod.indexCount)
                    .setInstanceCount(shadowCounts[cascade][ai])
                    .setStartInstanceLocation(shadowOffset),
                ai * Render::c_NumCascades + cascade
            });
            shadowWriteOff[ai] = shadowOffset;
            shadowOffset += shadowCounts[cascade][ai];
        }

        // Fill instance buffer for this cascade
        for (const auto& ref : csd.visibleRefs) {
            uint32_t& writeOffset = shadowWriteOff[ref.treeId];
            const auto& inst = regions[ref.regionIdx].instances[ref.instanceIdx];
            csd.instanceBuffer[writeOffset] = Render::InstanceBufferEntry(
                inst.model, inst.normal, ref.treeId);
            writeOffset++;
        }

        if (!csd.visibleRefs.empty())
            m_CommandList->writeBuffer(m_StageResources.shadowStage.instanceBuffer, csd.instanceBuffer.data(),
                csd.visibleRefs.size() * sizeof(Render::InstanceBufferEntry));

        // Clear this cascade's framebuffer
        nvrhi::utils::ClearDepthStencilAttachment(m_CommandList,
            m_StageResources.shadowStage.framebuffers[cascade], 1.0f, 0);

        // Draw trees into this cascade
        nvrhi::GraphicsState shadowState;
        shadowState.pipeline    = m_StageResources.shadowStage.treePipeline;
        shadowState.framebuffer = m_StageResources.shadowStage.framebuffers[cascade];
        shadowState.viewport    = shadowVPState;

        for (const auto& cmd : csd.drawCmds) {
            shadowState.bindings = { m_StageResources.shadowStage.bindingSet };
            shadowState.vertexBuffers = {
                { cmd.positionBuffer, 0, 0 },
                { m_StageResources.shadowStage.instanceBuffer, 1, 0 },
            };
            shadowState.indexBuffer = { cmd.indexBuffer, nvrhi::Format::R32_UINT, 0 };
            m_CommandList->setGraphicsState(shadowState);
            m_CommandList->setPushConstants(&cascade, sizeof(cascade));

            m_CommandList->drawIndexed(cmd.drawArgs);
        }

        if (m_StageResources.leafStage.shadowPipeline && m_StageResources.leafStage.shadowBindingSet) {
            nvrhi::GraphicsState leafShadowState;
            leafShadowState.pipeline = m_StageResources.leafStage.shadowPipeline;
            leafShadowState.framebuffer = m_StageResources.shadowStage.framebuffers[cascade];
            leafShadowState.viewport = shadowVPState;
            leafShadowState.bindings = { m_StageResources.leafStage.shadowBindingSet };
            leafShadowState.vertexBuffers = {
                { m_StageResources.shadowStage.instanceBuffer, 0, 0 },
            };

            const auto& assets = m_Registry.getAssets();
            const uint32_t numShadowSlots = std::max(1u, Render::c_NumCascades);
            for (const auto& cmd : csd.drawCmds) {
                const uint32_t ai = cmd.leafSlot / numShadowSlots;
                if (ai >= assets.size() || assets[ai].leafAsset.lodSlots.empty()) continue;
                const uint32_t lowestLod = static_cast<uint32_t>(assets[ai].leafAsset.lodSlots.size() - 1);
                const uint32_t leafCount = assets[ai].leafAsset.lodSlots[lowestLod].leafCount;
                if (leafCount == 0) continue;

                m_CommandList->setGraphicsState(leafShadowState);
                uint32_t pc[2] = { cmd.leafSlot, cascade };
                m_CommandList->setPushConstants(pc, sizeof(pc));
                m_CommandList->draw(
                    nvrhi::DrawArguments()
                        .setVertexCount(leafCount * 12u)
                        .setInstanceCount(cmd.drawArgs.instanceCount)
                        .setStartInstanceLocation(cmd.drawArgs.startInstanceLocation));
            }
        }

        // Draw terrain into this cascade
        if (m_StageResources.sceneTerrainStage.indexCount > 0 && terrainPtr
            && (terrainPtr->getBbox() * worldToLight).intersects(m_ViewHandler.shadowCasterBboxLS)) {
            nvrhi::GraphicsState terrShadow;
            terrShadow.pipeline    = m_StageResources.shadowStage.terrainPipeline;
            terrShadow.framebuffer = m_StageResources.shadowStage.framebuffers[cascade];
            terrShadow.viewport    = shadowVPState;
            terrShadow.bindings    = { m_StageResources.shadowStage.bindingSet };
            terrShadow.vertexBuffers = { { m_StageResources.sceneTerrainStage.vertexBuffer, 0, 0 } };
            terrShadow.indexBuffer   = { m_StageResources.sceneTerrainStage.indexBuffer, nvrhi::Format::R32_UINT, 0 };
            m_CommandList->setGraphicsState(terrShadow);
            // cascade doesn't matter for terrain render, but push needed for root signature
            m_CommandList->setPushConstants(&cascade, sizeof(cascade));
            m_CommandList->drawIndexed(
                nvrhi::DrawArguments().setVertexCount(m_StageResources.sceneTerrainStage.indexCount));
        }
    }
}

void TraditionalRenderPass::_RenderScenePass(nvrhi::IFramebuffer* framebuffer) {
    const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();
    const auto& lodSegments  = m_Registry.getLodSegments();
    const auto& lodDistances = m_Registry.getLodDistances();
    const auto& regions      = m_Registry.getRegions();

    // Tree pass pipeline
    if (!m_StageResources.sceneTreeStage.pipeline) {
        nvrhi::GraphicsPipelineDesc psoDesc;
        psoDesc.VS           = m_StageResources.sceneTreeStage.vertexShader;
        psoDesc.PS           = m_StageResources.sceneTreeStage.pixelShader;
        psoDesc.inputLayout  = m_StageResources.sceneTreeStage.inputLayout;
        psoDesc.bindingLayouts = { m_StageResources.sceneTreeStage.bindingLayout };
        psoDesc.primType     = nvrhi::PrimitiveType::TriangleList;
        // cull=none so leaf cross-billboards (and trunk back-faces, slight cost) are visible
        // from both sides. Single PSO for trunk + leaf — splitting would double draw count.
        psoDesc.renderState.rasterState.setCullNone();
    #if XYLEM_USE_REVERSE_Z
        psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
    #else
        psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    #endif
        m_StageResources.sceneTreeStage.pipeline = GetDevice()->createGraphicsPipeline(psoDesc, fbinfo);
    }
    if (!m_StageResources.leafStage.pipeline) {
        nvrhi::GraphicsPipelineDesc leafPso;
        leafPso.VS = m_StageResources.leafStage.vertexShader;
        leafPso.PS = m_StageResources.leafStage.pixelShader;
        leafPso.inputLayout = m_StageResources.leafStage.inputLayout;
        leafPso.bindingLayouts = { m_StageResources.leafStage.bindingLayout };
        leafPso.primType = nvrhi::PrimitiveType::TriangleList;
        leafPso.renderState.rasterState.setCullNone();
    #if XYLEM_USE_REVERSE_Z
        leafPso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
    #else
        leafPso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    #endif
        m_StageResources.leafStage.pipeline = GetDevice()->createGraphicsPipeline(leafPso, fbinfo);
    }

    const uint32_t numLods = static_cast<uint32_t>(lodSegments.size());
    const uint32_t impostorThresholdLod = numLods > 0 ? numLods - 1 : 0;

    // Frustum culling + LOD/impostor assignment
    m_VisibleInstanceReferences.clear();
    m_VisibleImpostorReferences.clear();
    m_InstanceCounts.assign(m_GPUAssets.size(), std::vector<uint32_t>(lodSegments.size(), 0));
    m_ImpostorCounts.assign(std::max<size_t>(1, m_GPUAssets.size()), 0);

    for (uint32_t ri = 0; ri < regions.size(); ri++) {
        const auto& region = regions[ri];
        if (!region.cullBox.isempty() && !m_ViewHandler.view.IsBoxVisible(region.cullBox))
            continue;
        for (uint32_t ii = 0; ii < region.instances.size(); ii++) {
            const auto& inst = region.instances[ii];

            // Global asset visibility
            const auto* assetDef = m_Registry.findAsset(inst.assetId);
            if (!assetDef || !assetDef->visible) continue;

            // Per-region asset visibility
            auto visIt = region.assetVisible.find(inst.assetId);
            if (visIt != region.assetVisible.end() && !visIt->second) continue;

            if (!m_ViewHandler.view.IsBoxVisible(inst.bbox)) continue;
            float dist = dm::distance(m_ViewHandler.camera.GetPosition(), inst.bbox);

            auto it = m_AssetIdToGPUIndex.find(inst.assetId);
            if (it == m_AssetIdToGPUIndex.end()) continue;
            uint32_t gpuIdx = static_cast<uint32_t>(it->second);
            if (numLods == 0 || lodDistances.empty()) continue;

            const bool useImpostor = !lodDistances.empty()
                && (dist >= lodDistances[impostorThresholdLod]);
            if (useImpostor) {
                m_VisibleImpostorReferences.push_back({ ri, ii, gpuIdx, impostorThresholdLod });
                m_ImpostorCounts[gpuIdx]++;
            } else {
                uint32_t lod = m_ViewHandler.distToLOD(dist, lodDistances);
                lod = std::min<uint32_t>(lod, numLods > 0 ? numLods - 1 : 0);
                m_VisibleInstanceReferences.push_back({ ri, ii, gpuIdx, lod });
                m_InstanceCounts[gpuIdx][lod]++;
            }
        }
    }

    // Build draw commands
    m_DrawCmds.clear();
    m_InstanceOffsets.assign(m_GPUAssets.size(), std::vector<uint32_t>(lodSegments.size(), 0));
    uint32_t instanceOffset = 0;

    for (uint32_t ai = 0; ai < m_InstanceCounts.size(); ai++) {
        for (uint32_t li = 0; li < lodSegments.size(); li++) {
            uint32_t count = m_InstanceCounts[ai][li];
            if (count == 0) continue;
            const auto& lod = m_GPUAssets[ai].lods[li];
            m_DrawCmds.push_back({
                lod.vbs,
                lod.indexBuffer,
                nvrhi::DrawArguments()
                    .setVertexCount(lod.indexCount)
                    .setInstanceCount(count)
                    .setStartInstanceLocation(instanceOffset),
                m_GPUAssets[ai].textureSetIdx,
                ai * numLods + li
            });
            m_InstanceOffsets[ai][li] = instanceOffset;
            instanceOffset += count;
        }
    }

    for (const auto& ref : m_VisibleInstanceReferences) {
        uint32_t& writeOff = m_InstanceOffsets[ref.treeId][ref.lodID];
        const auto& inst = regions[ref.regionIdx].instances[ref.instanceIdx];
        m_VisibleInstanceBuffer[writeOff] = Render::InstanceBufferEntry(
            inst.model, inst.normal, ref.treeId);
        writeOff++;
    }

    if (!m_VisibleInstanceReferences.empty())
        m_CommandList->writeBuffer(m_StageResources.sceneTreeStage.instanceBuffer, m_VisibleInstanceBuffer.data(),
            m_VisibleInstanceReferences.size() * sizeof(Render::InstanceBufferEntry));

    // Fill impostor staging data in per-asset slot windows.
    std::fill(m_ImpostorWriteOffsets.begin(), m_ImpostorWriteOffsets.end(), 0);
    for (uint32_t ai = 0; ai < m_GPUAssets.size() && ai < m_ImpostorWriteOffsets.size(); ai++)
        m_ImpostorWriteOffsets[ai] = m_ImpostorSlotOffsets[ai];

    for (const auto& ref : m_VisibleImpostorReferences) {
        if (ref.treeId >= m_ImpostorWriteOffsets.size()
            || ref.treeId >= m_ImpostorSlotOffsets.size()
            || ref.treeId >= m_ImpostorMaxSlotCounts.size())
            continue;

        uint32_t& writeOffset = m_ImpostorWriteOffsets[ref.treeId];
        const uint32_t baseOffset = m_ImpostorSlotOffsets[ref.treeId];
        const uint32_t maxCount = m_ImpostorMaxSlotCounts[ref.treeId];
        if ((writeOffset - baseOffset) >= maxCount || writeOffset >= m_ImpostorInstanceStaging.size())
            continue;

        const auto& inst = regions[ref.regionIdx].instances[ref.instanceIdx];
        const uint32_t persistentId = writeOffset;

        m_ImpostorInstanceStaging[persistentId] = Render::InstanceBufferEntry(
            inst.model, inst.normal, ref.treeId);

        Render::CullInstanceData cullData{};
        cullData.bbox     = inst.bbox;
        cullData.baseSlot = ref.treeId * std::max(1u, numLods);
        cullData.regionId = ref.regionIdx;
        cullData.active   = 1;
        m_ImpostorCullDataStaging[persistentId] = cullData;
        m_ImpostorVisStaging[persistentId] = persistentId;

        writeOffset++;
    }

    if (m_StageResources.impostorStage.slotOffsetBuffer && !m_ImpostorSlotOffsets.empty()) {
        m_CommandList->writeBuffer(
            m_StageResources.impostorStage.slotOffsetBuffer,
            m_ImpostorSlotOffsets.data(),
            m_ImpostorSlotOffsets.size() * sizeof(uint32_t));
    }
    if (m_StageResources.impostorStage.instanceBuffer && !m_ImpostorInstanceStaging.empty()) {
        m_CommandList->writeBuffer(
            m_StageResources.impostorStage.instanceBuffer,
            m_ImpostorInstanceStaging.data(),
            m_ImpostorInstanceStaging.size() * sizeof(Render::InstanceBufferEntry));
    }
    if (m_StageResources.impostorStage.cullDataBuffer && !m_ImpostorCullDataStaging.empty()) {
        m_CommandList->writeBuffer(
            m_StageResources.impostorStage.cullDataBuffer,
            m_ImpostorCullDataStaging.data(),
            m_ImpostorCullDataStaging.size() * sizeof(Render::CullInstanceData));
    }
    if (m_StageResources.impostorStage.visBuffer && !m_ImpostorVisStaging.empty()) {
        m_CommandList->writeBuffer(
            m_StageResources.impostorStage.visBuffer,
            m_ImpostorVisStaging.data(),
            m_ImpostorVisStaging.size() * sizeof(uint32_t));
    }

    // Draw trees
    nvrhi::GraphicsState state;
    state.pipeline   = m_StageResources.sceneTreeStage.pipeline;
    state.framebuffer = framebuffer;
    state.viewport   = m_ViewHandler.view.GetViewportState();

    for (const auto& cmd : m_DrawCmds) {
        state.bindings = { m_StageResources.sceneTreeStage.bindingSets[cmd.textureSetIdx] };
        state.vertexBuffers = {
            { cmd.vertexBuffers.position,  0, 0 },
            { cmd.vertexBuffers.normal,    1, 0 },
            { cmd.vertexBuffers.tangent,   2, 0 },
            { cmd.vertexBuffers.bitangent, 3, 0 },
            { cmd.vertexBuffers.uv,        4, 0 },
            { m_StageResources.sceneTreeStage.instanceBuffer, 5, 0 },
        };
        state.indexBuffer = { cmd.indexBuffer, nvrhi::Format::R32_UINT, 0 };
        m_CommandList->setGraphicsState(state);
        m_CommandList->drawIndexed(cmd.drawArgs);
    }

    if (m_StageResources.leafStage.pipeline && m_StageResources.leafStage.bindingSet) {
        nvrhi::GraphicsState leafState;
        leafState.pipeline = m_StageResources.leafStage.pipeline;
        leafState.framebuffer = framebuffer;
        leafState.viewport = m_ViewHandler.view.GetViewportState();
        leafState.bindings = { m_StageResources.leafStage.bindingSet };
        leafState.vertexBuffers = {
            { m_StageResources.sceneTreeStage.instanceBuffer, 0, 0 },
        };

        const auto& assets = m_Registry.getAssets();
        const uint32_t numLodsForLeaves = std::max(1u, numLods);
        for (const auto& cmd : m_DrawCmds) {
            const uint32_t ai = cmd.leafSlot / numLodsForLeaves;
            const uint32_t li = cmd.leafSlot % numLodsForLeaves;
            if (ai >= assets.size() || li >= assets[ai].leafAsset.lodSlots.size()) continue;
            const uint32_t leafCount = assets[ai].leafAsset.lodSlots[li].leafCount;
            if (leafCount == 0) continue;

            m_CommandList->setGraphicsState(leafState);
            uint32_t pc[2] = { cmd.leafSlot, 0 };
            m_CommandList->setPushConstants(pc, sizeof(pc));
            m_CommandList->draw(
                nvrhi::DrawArguments()
                    .setVertexCount(leafCount * 12u)
                    .setInstanceCount(cmd.drawArgs.instanceCount)
                    .setStartInstanceLocation(cmd.drawArgs.startInstanceLocation));
        }
    }

    // Terrain color pass
    if (m_StageResources.sceneTerrainStage.indexCount > 0) {
        if (!m_StageResources.sceneTerrainStage.pipeline) {
            nvrhi::GraphicsPipelineDesc terrainPso;
            terrainPso.VS = m_StageResources.sceneTerrainStage.vertexShader;
            terrainPso.PS = m_StageResources.sceneTerrainStage.pixelShader;
            terrainPso.inputLayout = m_StageResources.sceneTerrainStage.inputLayout;
            terrainPso.bindingLayouts = { m_StageResources.sceneTerrainStage.bindingLayout };
            terrainPso.primType = nvrhi::PrimitiveType::TriangleList;
    #if XYLEM_USE_REVERSE_Z
            terrainPso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
    #else
            terrainPso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    #endif
            terrainPso.renderState.rasterState.setCullNone();
            m_StageResources.sceneTerrainStage.pipeline = GetDevice()->createGraphicsPipeline(terrainPso, fbinfo);
        }

        nvrhi::GraphicsState terrainState;
        terrainState.pipeline   = m_StageResources.sceneTerrainStage.pipeline;
        terrainState.framebuffer = framebuffer;
        terrainState.viewport   = m_ViewHandler.view.GetViewportState();
        terrainState.bindings   = { m_StageResources.sceneTerrainStage.bindingSet };
        terrainState.vertexBuffers = { { m_StageResources.sceneTerrainStage.vertexBuffer, 0, 0 } };
        terrainState.indexBuffer   = { m_StageResources.sceneTerrainStage.indexBuffer, nvrhi::Format::R32_UINT, 0 };
        m_CommandList->setGraphicsState(terrainState);
        m_CommandList->drawIndexed(
            nvrhi::DrawArguments()
                .setVertexCount(m_StageResources.sceneTerrainStage.indexCount));
    }
}

void TraditionalRenderPass::_RenderImpostorPass(nvrhi::IFramebuffer* framebuffer) {
    m_ImpostorDrawCallCount = 0;

    if (m_GPUAssets.empty()
        || m_StageResources.impostorStage.bindingSets.empty()
        || !m_StageResources.impostorStage.instanceBuffer
        || !m_StageResources.impostorStage.cullDataBuffer
        || !m_StageResources.impostorStage.visBuffer
        || !m_StageResources.impostorStage.slotOffsetBuffer)
    {
        return;
    }

    const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();
    if (!m_StageResources.impostorStage.pipeline) {
        nvrhi::GraphicsPipelineDesc psoDesc;
        psoDesc.VS = m_StageResources.impostorStage.vertexShader;
        psoDesc.PS = m_StageResources.impostorStage.pixelShader;
        psoDesc.inputLayout = nullptr;
        psoDesc.bindingLayouts = { m_StageResources.impostorStage.bindingLayout };
        psoDesc.primType = nvrhi::PrimitiveType::TriangleStrip;
    #if XYLEM_USE_REVERSE_Z
        psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
    #else
        psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    #endif
        psoDesc.renderState.rasterState.setCullNone();
        psoDesc.renderState.rasterState
            .setDepthBias(16)
            .setSlopeScaleDepthBias(1.0f);
        m_StageResources.impostorStage.pipeline = GetDevice()->createGraphicsPipeline(psoDesc, fbinfo);
    }

    nvrhi::GraphicsState state;
    state.pipeline = m_StageResources.impostorStage.pipeline;
    state.framebuffer = framebuffer;
    state.viewport = m_ViewHandler.view.GetViewportState();

    for (uint32_t ai = 0; ai < m_GPUAssets.size(); ai++) {
        if (ai >= m_ImpostorCounts.size() || m_ImpostorCounts[ai] == 0)
            continue;

        uint32_t texIdx = std::min<uint32_t>(
            m_GPUAssets[ai].textureSetIdx,
            static_cast<uint32_t>(m_StageResources.impostorStage.bindingSets.size() - 1));
        state.bindings = { m_StageResources.impostorStage.bindingSets[texIdx] };
        m_CommandList->setGraphicsState(state);
        m_CommandList->setPushConstants(&ai, sizeof(ai));
        m_CommandList->draw(
            nvrhi::DrawArguments()
                .setVertexCount(4)
                .setInstanceCount(m_ImpostorCounts[ai]));
        m_ImpostorDrawCallCount++;
    }
}

bool TraditionalRenderPass::_InitShared() {
    m_StageResources.frameShared.constantBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(shader_cb::kCullFrameSize)
            .setIsConstantBuffer(true)
            .setDebugName("CullConstantBuffer")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
    );
    return !!m_StageResources.frameShared.constantBuffer;
}

bool TraditionalRenderPass::_InitTreePass() {
    // Shaders
    m_StageResources.sceneTreeStage.vertexShader = m_ShaderFactory->CreateShader("app/TraditionalRenderPass.hlsl", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_StageResources.sceneTreeStage.pixelShader  = m_ShaderFactory->CreateShader("app/TraditionalRenderPass.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_StageResources.sceneTreeStage.vertexShader || !m_StageResources.sceneTreeStage.pixelShader) return false;

    nvrhi::VertexAttributeDesc attributes[] = {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(0)
            .setElementStride(sizeof(dm::float3)),
        nvrhi::VertexAttributeDesc()
            .setName("NORMAL")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(1)
            .setElementStride(sizeof(dm::float3)),
        nvrhi::VertexAttributeDesc()
            .setName("TANGENT")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(2)
            .setElementStride(sizeof(dm::float3)),
        nvrhi::VertexAttributeDesc()
            .setName("BITANGENT")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(3)
            .setElementStride(sizeof(dm::float3)),
        nvrhi::VertexAttributeDesc()
            .setName("UV")
            .setFormat(nvrhi::Format::RG32_FLOAT)
            .setOffset(0)
            .setBufferIndex(4)
            .setElementStride(sizeof(dm::float2)),
        nvrhi::VertexAttributeDesc()
            .setName("MODEL_MATRIX")
            .setFormat(nvrhi::Format::RGBA32_FLOAT)
            .setArraySize(4)
            .setOffset(offsetof(Render::InstanceBufferEntry, model))
            .setBufferIndex(5)
            .setElementStride(sizeof(Render::InstanceBufferEntry))
            .setIsInstanced(true),
        nvrhi::VertexAttributeDesc()
            .setName("NORMAL_MATRIX")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setArraySize(3)
            .setOffset(offsetof(Render::InstanceBufferEntry, normal))
            .setBufferIndex(5)
            .setElementStride(sizeof(Render::InstanceBufferEntry))
            .setIsInstanced(true),
    };
    m_StageResources.sceneTreeStage.inputLayout = GetDevice()->createInputLayout(attributes, uint32_t(std::size(attributes)), m_StageResources.sceneTreeStage.vertexShader);
    if (!m_StageResources.sceneTreeStage.inputLayout) return false;

    return true;
}

bool TraditionalRenderPass::_InitLeafPass() {
    m_StageResources.leafStage.vertexShader = m_ShaderFactory->CreateShader(
        "app/TraditionalLeaves.hlsl", "leaf_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_StageResources.leafStage.pixelShader = m_ShaderFactory->CreateShader(
        "app/TraditionalLeaves.hlsl", "leaf_ps", nullptr, nvrhi::ShaderType::Pixel);
    m_StageResources.leafStage.shadowVS = m_ShaderFactory->CreateShader(
        "app/TraditionalLeaves.hlsl", "leaf_shadow_vs", nullptr, nvrhi::ShaderType::Vertex);
    if (!m_StageResources.leafStage.vertexShader
        || !m_StageResources.leafStage.pixelShader
        || !m_StageResources.leafStage.shadowVS)
        return false;

    nvrhi::VertexAttributeDesc leafAttrs[] = {
        nvrhi::VertexAttributeDesc()
            .setName("MODEL_MATRIX")
            .setFormat(nvrhi::Format::RGBA32_FLOAT)
            .setArraySize(4)
            .setOffset(offsetof(Render::InstanceBufferEntry, model))
            .setBufferIndex(0)
            .setElementStride(sizeof(Render::InstanceBufferEntry))
            .setIsInstanced(true),
        nvrhi::VertexAttributeDesc()
            .setName("NORMAL_MATRIX")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setArraySize(3)
            .setOffset(offsetof(Render::InstanceBufferEntry, normal))
            .setBufferIndex(0)
            .setElementStride(sizeof(Render::InstanceBufferEntry))
            .setIsInstanced(true),
    };
    m_StageResources.leafStage.inputLayout = GetDevice()->createInputLayout(
        leafAttrs, uint32_t(std::size(leafAttrs)), m_StageResources.leafStage.vertexShader);
    if (!m_StageResources.leafStage.inputLayout) return false;

    nvrhi::VertexAttributeDesc shadowAttrs[] = {
        nvrhi::VertexAttributeDesc()
            .setName("MODEL_MATRIX")
            .setFormat(nvrhi::Format::RGBA32_FLOAT)
            .setArraySize(4)
            .setOffset(offsetof(Render::InstanceBufferEntry, model))
            .setBufferIndex(0)
            .setElementStride(sizeof(Render::InstanceBufferEntry))
            .setIsInstanced(true),
    };
    m_StageResources.leafStage.shadowInputLayout = GetDevice()->createInputLayout(
        shadowAttrs, uint32_t(std::size(shadowAttrs)), m_StageResources.leafStage.shadowVS);
    if (!m_StageResources.leafStage.shadowInputLayout) return false;

    nvrhi::BindingLayoutDesc leafLayout;
    leafLayout.visibility = nvrhi::ShaderType::All;
    leafLayout.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(0),
        nvrhi::BindingLayoutItem::PushConstants(1, sizeof(uint32_t) * 2),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),
        nvrhi::BindingLayoutItem::Texture_SRV(2),
        nvrhi::BindingLayoutItem::Sampler(0),
    };
    m_StageResources.leafStage.bindingLayout = GetDevice()->createBindingLayout(leafLayout);
    if (!m_StageResources.leafStage.bindingLayout) return false;

    nvrhi::BindingLayoutDesc shadowLayout;
    shadowLayout.visibility = nvrhi::ShaderType::All;
    shadowLayout.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(0),
        nvrhi::BindingLayoutItem::PushConstants(1, sizeof(uint32_t) * 2),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),
    };
    m_StageResources.leafStage.shadowBindingLayout = GetDevice()->createBindingLayout(shadowLayout);
    return m_StageResources.leafStage.shadowBindingLayout != nullptr;
}

bool TraditionalRenderPass::_InitImpostorPass() {
    m_StageResources.impostorStage.vertexShader = m_ShaderFactory->CreateShader(
        "app/ImpostorRenderPass.hlsl", "impostor_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_StageResources.impostorStage.pixelShader = m_ShaderFactory->CreateShader(
        "app/ImpostorRenderPass.hlsl", "impostor_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_StageResources.impostorStage.vertexShader || !m_StageResources.impostorStage.pixelShader)
        return false;

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
    m_StageResources.impostorStage.bindingLayout = GetDevice()->createBindingLayout(layoutDesc);
    if (!m_StageResources.impostorStage.bindingLayout) return false;

    m_StageResources.impostorStage.sampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
            .setAllFilters(true));
    if (!m_StageResources.impostorStage.sampler) return false;

    m_StageResources.impostorStage.depthSampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
            .setAllFilters(false));
    return m_StageResources.impostorStage.depthSampler != nullptr;
}

// bool TraditionalRenderPass::_InitTimerQueries() {
//     for (uint32_t i = 0; i < m_QueuedFrames; i++)
//         m_GpuTimers[i] = GetDevice()->createTimerQuery();
//     return true;
// }

bool TraditionalRenderPass::_InitShadowPass() {
    m_StageResources.shadowStage.depthTexture = GetDevice()->createTexture(
        nvrhi::TextureDesc()
            .setWidth(m_ShadowRes).setHeight(m_ShadowRes)
            .setArraySize(Render::c_NumCascades)
            .setDimension(nvrhi::TextureDimension::Texture2DArray)
            .setFormat(nvrhi::Format::D32)
            .setIsRenderTarget(true)
            .setUseClearValue(true)
            .setClearValue(nvrhi::Color(1.f))
            .enableAutomaticStateTracking(nvrhi::ResourceStates::DepthWrite)
            .setDebugName("ShadowMapArray")
    );
    if (!m_StageResources.shadowStage.depthTexture) return false;

    for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
        m_StageResources.shadowStage.framebuffers[c] = GetDevice()->createFramebuffer(
            nvrhi::FramebufferDesc().setDepthAttachment(
                nvrhi::FramebufferAttachment()
                    .setTexture(m_StageResources.shadowStage.depthTexture)
                    .setArraySlice(c)
            ));
        if (!m_StageResources.shadowStage.framebuffers[c]) return false;
    }

    // shaders
    m_StageResources.shadowStage.treeVS = m_ShaderFactory->CreateShader(
        "app/shadow.hlsl", "tree_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_StageResources.shadowStage.terrainVS = m_ShaderFactory->CreateShader(
        "app/shadow.hlsl", "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    if (!m_StageResources.shadowStage.treeVS || !m_StageResources.shadowStage.terrainVS) return false;

    // TODO: deinterleave instance data since shadow pass only needs model matrix
    nvrhi::VertexAttributeDesc treeShadowAttrs[] = {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(0)
            .setElementStride(sizeof(dm::float3)),
        nvrhi::VertexAttributeDesc()
            .setName("MODEL_MATRIX")
            .setFormat(nvrhi::Format::RGBA32_FLOAT)
            .setArraySize(4)
            .setOffset(offsetof(Render::InstanceBufferEntry, model))
            .setBufferIndex(1)
            .setIsInstanced(true)
            .setElementStride(sizeof(Render::InstanceBufferEntry)),
    };
    m_StageResources.shadowStage.treeInputLayout = GetDevice()->createInputLayout(
        treeShadowAttrs, uint32_t(std::size(treeShadowAttrs)), m_StageResources.shadowStage.treeVS);
    if (!m_StageResources.shadowStage.treeInputLayout) return false;

    nvrhi::VertexAttributeDesc terrainShadowAttrs[] = {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(Scene::TerrainVertex, pos))
            .setBufferIndex(0)
            .setElementStride(sizeof(Scene::TerrainVertex)),
    };
    m_StageResources.shadowStage.terrainInputLayout = GetDevice()->createInputLayout(
        terrainShadowAttrs, uint32_t(std::size(terrainShadowAttrs)), m_StageResources.shadowStage.terrainVS);
    if (!m_StageResources.shadowStage.terrainInputLayout) return false;

    m_StageResources.shadowStage.comparisonSampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setMinFilter(true)
            .setMagFilter(true)
            .setMipFilter(false)
            .setReductionType(nvrhi::SamplerReductionType::Comparison)
            .setAllAddressModes(nvrhi::SamplerAddressMode::Border)
            .setBorderColor(nvrhi::Color(1.f))
    );
    if (!m_StageResources.shadowStage.comparisonSampler) return false;

    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(traditional_reg::Shadow::kCB_Frame, m_StageResources.frameShared.constantBuffer,
            nvrhi::BufferRange(0, shader_cb::kFrameSize)),
        nvrhi::BindingSetItem::PushConstants(traditional_reg::Shadow::kPushC_CascadeIndex, sizeof(uint32_t)),
    };
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
            bsd, m_StageResources.shadowStage.bindingLayout, m_StageResources.shadowStage.bindingSet))
        return false;

    return true;
}

bool TraditionalRenderPass::_InitDebug() {
    // Single scratch texture for the currently-selected cascade slice. Copied
    // into each frame from the relevant array slice and sampled by ImGui.
    m_StageResources.shadowStage.debugSelectedCascadeTexture = GetDevice()->createTexture(
        nvrhi::TextureDesc()
            .setWidth(m_ShadowRes).setHeight(m_ShadowRes)
            .setFormat(nvrhi::Format::R32_FLOAT)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true)
            .setDebugName("DebugSelectedCascade")
    );

    return !!m_StageResources.shadowStage.debugSelectedCascadeTexture;
}

bool TraditionalRenderPass::_InitTerrainPass(nvrhi::ICommandList* initCL) {
    const auto* terrain = m_Registry.getTerrain();
    if (!terrain) return true;

    const auto& verts   = terrain->getVertices();
    const auto& indices = terrain->getIndices();
    if (verts.empty() || indices.empty()) return true;

    m_StageResources.sceneTerrainStage.indexCount = static_cast<uint32_t>(indices.size());

    m_StageResources.sceneTerrainStage.vertexShader = m_ShaderFactory->CreateShader("app/terrain.hlsl", "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_StageResources.sceneTerrainStage.pixelShader  = m_ShaderFactory->CreateShader("app/terrain.hlsl", "terrain_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_StageResources.sceneTerrainStage.vertexShader || !m_StageResources.sceneTerrainStage.pixelShader) return false;

    nvrhi::VertexAttributeDesc terrainAttrs[] = {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(Scene::TerrainVertex, pos))
            .setBufferIndex(0)
            .setElementStride(sizeof(Scene::TerrainVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("NORMAL")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(Scene::TerrainVertex, normal))
            .setBufferIndex(0)
            .setElementStride(sizeof(Scene::TerrainVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("UV")
            .setFormat(nvrhi::Format::RG32_FLOAT)
            .setOffset(offsetof(Scene::TerrainVertex, uv))
            .setBufferIndex(0)
            .setElementStride(sizeof(Scene::TerrainVertex)),
    };
    m_StageResources.sceneTerrainStage.inputLayout = GetDevice()->createInputLayout(
        terrainAttrs, uint32_t(std::size(terrainAttrs)), m_StageResources.sceneTerrainStage.vertexShader);
    if (!m_StageResources.sceneTerrainStage.inputLayout) return false;

    nvrhi::BufferDesc vbDesc;
    vbDesc.isVertexBuffer = true;
    vbDesc.byteSize       = verts.size() * sizeof(Scene::TerrainVertex);
    vbDesc.debugName      = "TerrainVB";
    vbDesc.initialState   = nvrhi::ResourceStates::CopyDest;
    m_StageResources.sceneTerrainStage.vertexBuffer = GetDevice()->createBuffer(vbDesc);
    initCL->beginTrackingBufferState(m_StageResources.sceneTerrainStage.vertexBuffer, nvrhi::ResourceStates::CopyDest);
    initCL->writeBuffer(m_StageResources.sceneTerrainStage.vertexBuffer, verts.data(), vbDesc.byteSize);
    initCL->setPermanentBufferState(m_StageResources.sceneTerrainStage.vertexBuffer, nvrhi::ResourceStates::VertexBuffer);

    nvrhi::BufferDesc ibDesc;
    ibDesc.isIndexBuffer = true;
    ibDesc.byteSize      = indices.size() * sizeof(uint32_t);
    ibDesc.debugName     = "TerrainIB";
    ibDesc.initialState  = nvrhi::ResourceStates::CopyDest;
    m_StageResources.sceneTerrainStage.indexBuffer = GetDevice()->createBuffer(ibDesc);
    initCL->beginTrackingBufferState(m_StageResources.sceneTerrainStage.indexBuffer, nvrhi::ResourceStates::CopyDest);
    initCL->writeBuffer(m_StageResources.sceneTerrainStage.indexBuffer, indices.data(), ibDesc.byteSize);
    initCL->setPermanentBufferState(m_StageResources.sceneTerrainStage.indexBuffer, nvrhi::ResourceStates::IndexBuffer);

    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(traditional_reg::Terrain::kCB_Frame, m_StageResources.frameShared.constantBuffer,
            nvrhi::BufferRange(0, shader_cb::kFrameSize)),
        nvrhi::BindingSetItem::Sampler(traditional_reg::Terrain::kSampler_Shadow, m_StageResources.shadowStage.comparisonSampler),
        nvrhi::BindingSetItem::Texture_SRV(traditional_reg::Terrain::kTex_ShadowMap, m_StageResources.shadowStage.depthTexture),
    };
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
            bsd, m_StageResources.sceneTerrainStage.bindingLayout, m_StageResources.sceneTerrainStage.bindingSet))
        return false;

    return true;
}

bool TraditionalRenderPass::_InitSkyPass() {
    m_StageResources.skyStage.vertexShader = m_ShaderFactory->CreateShader("app/sky.hlsl", "sky_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_StageResources.skyStage.pixelShader  = m_ShaderFactory->CreateShader("app/sky.hlsl", "sky_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_StageResources.skyStage.vertexShader || !m_StageResources.skyStage.pixelShader) return false;

    nvrhi::BufferDesc cbDesc;
    cbDesc.byteSize         = sizeof(SkyConstants);
    cbDesc.isConstantBuffer = true;
    cbDesc.isVolatile       = true;
    cbDesc.maxVersions      = 16;
    cbDesc.debugName        = "SkyConstants";
    m_StageResources.skyStage.constantBuffer = GetDevice()->createBuffer(cbDesc);
    if (!m_StageResources.skyStage.constantBuffer) return false;

    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(traditional_reg::Sky::kCB_Sky, m_StageResources.skyStage.constantBuffer),
    };
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
            bsd, m_StageResources.skyStage.bindingLayout, m_StageResources.skyStage.bindingSet))
        return false;

    return true;
}




