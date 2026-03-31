#include "include/TraditionalRenderPass.hpp"
#include "include/Globals.hpp"
#include "include/macros.h"
#include "include/Terrain.hpp"

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

        vDesc.debugName = "VB_" + assetDef.name + "_LOD" + std::to_string(j);
        vDesc.byteSize  = lodDef.vertices.size() * sizeof(ProcGen::TreeVertex);
        auto vBuf = device->createBuffer(vDesc);
        commandList->beginTrackingBufferState(vBuf, nvrhi::ResourceStates::CopyDest);
        commandList->writeBuffer(vBuf, lodDef.vertices.data(), vDesc.byteSize);
        commandList->setPermanentBufferState(vBuf, nvrhi::ResourceStates::VertexBuffer);

        iDesc.debugName = "IB_" + assetDef.name + "_LOD" + std::to_string(j);
        iDesc.byteSize  = lodDef.indices.size() * sizeof(uint32_t);
        auto iBuf = device->createBuffer(iDesc);
        commandList->beginTrackingBufferState(iBuf, nvrhi::ResourceStates::CopyDest);
        commandList->writeBuffer(iBuf, lodDef.indices.data(), iDesc.byteSize);
        commandList->setPermanentBufferState(iBuf, nvrhi::ResourceStates::IndexBuffer);

        gpuAsset.lods[j].vertexBuffer   = vBuf;
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

    m_TreePass.instanceBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(totalInstances * sizeof(Render::InstanceBufferEntry))
            .setStructStride(sizeof(Render::InstanceBufferEntry))
            .setDebugName("TreeInstanceBuffer")
#if XYLEM_USE_STRUCTURED_BUFFER
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource)
#else
            .setIsVertexBuffer(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
#endif
    );

    m_ShadowPass.instanceBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(totalInstances * sizeof(Render::InstanceBufferEntry))
            .setStructStride(sizeof(Render::InstanceBufferEntry))
            .setDebugName("ShadowInstanceBuffer")
            .setIsVertexBuffer(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
    );

    m_VisibleInstanceBuffer.resize(totalInstances);
    m_ShadowInstanceBuffer.resize(totalInstances);
}

void TraditionalRenderPass::_RebuildBindingSets() {
    uint32_t totalInstances = std::max<uint32_t>(1, m_Registry.totalInstanceCount());

    m_TreePass.bindingSets.resize(m_TreePass.textureSets.size());

    for (size_t i = 0; i < m_TreePass.textureSets.size(); i++) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_Shared.constantBuffer, nvrhi::BufferRange(0, Render::c_ConstantBufferSize)),
            nvrhi::BindingSetItem::Sampler(0, m_TreePass.sampler),
            nvrhi::BindingSetItem::Sampler(1, m_ShadowPass.comparisonSampler),
            nvrhi::BindingSetItem::Texture_SRV(0, m_TreePass.textureSets[i].diffuse),
            nvrhi::BindingSetItem::Texture_SRV(1, m_TreePass.textureSets[i].normalMap),
            nvrhi::BindingSetItem::Texture_SRV(2, m_ShadowPass.depthTexture),
#if XYLEM_USE_STRUCTURED_BUFFER
            nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_TreePass.instanceBuffer,
                nvrhi::Format::UNKNOWN,
                nvrhi::BufferRange(0, totalInstances * sizeof(Render::InstanceBufferEntry))),
            nvrhi::BindingSetItem::PushConstants(1, sizeof(uint32_t)),
#endif
        };

        if (i == 0) {
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
                bsd, m_TreePass.bindingLayout, m_TreePass.bindingSets[0]);
        } else {
            m_TreePass.bindingSets[i] = GetDevice()->createBindingSet(bsd, m_TreePass.bindingLayout);
        }
    }
}

// ===========================================================================
// Init
// ===========================================================================

bool TraditionalRenderPass::Init() {
    // CommonRenderPasses must be constructed before opening initCL — its constructor
    // opens its own temporary command list to upload placeholder textures.
    engine::CommonRenderPasses commonPasses(GetDevice(), m_ShaderFactory);

    { // uses a temp CL to free upload buffer memory back to the OS
        nvrhi::CommandListHandle initCL = GetDevice()->createCommandList();
        initCL->open();

        // Upload GPU buffers from registry CPU data
        _UploadAllAssets(GetDevice(), initCL);

        if (!_InitShared())                                return false;
        if (!_InitShadowPass())                            return false;
        if (!_InitTreePass(initCL, commonPasses))          return false;
        if (!_InitTerrainPass(initCL))                     return false;
        if (!_InitSkyPass())                               return false;
        if (!_InitViewHandler())                           return false;

        initCL->close();
        GetDevice()->executeCommandList(initCL);
    }

    // Persistent render command list for per-frame drawing
    m_CommandList = GetDevice()->createCommandList();

    _InitTimerQueries();

    uint32_t totalInstances = m_Registry.totalInstanceCount();
    m_UI.totalInstanceCount = totalInstances;

    m_VisibleInstanceReferences.reserve(totalInstances);

    size_t numAssets = m_GPUAssets.size();
    size_t numLods   = m_Registry.getLodSegments().size();

    m_InstanceCounts.resize(numAssets, std::vector<uint32_t>(numLods, 0));
    m_InstanceOffsets.assign(numAssets, std::vector<uint32_t>(numLods, 0));

    m_VisibleInstanceBuffer.resize(std::max<uint32_t>(1, totalInstances));
    m_DrawCmds.reserve(numAssets * numLods);

    m_ShadowInstanceBuffer.resize(std::max<uint32_t>(1, totalInstances));
    m_ShadowDrawCmds.reserve(numAssets);

    return true;
}

// ===========================================================================
// Animate — drives the hot-reload cycle
// ===========================================================================

void TraditionalRenderPass::Animate(float seconds) {
    m_ViewHandler->camera.Animate(seconds);
    GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);

    // Hot-reload cycle
    if (!m_Registry.anyDirty()) return;

    auto dirtyAssets  = m_Registry.getDirtyAssetIndices();
    auto dirtyRegions = m_Registry.getDirtyRegionIndices();

    m_Registry.rebuildDirtyAssets();
    m_Registry.rebuildDirtyRegions();

    if (!dirtyAssets.empty())  onAssetsDirty(dirtyAssets);
    if (!dirtyRegions.empty()) onRegionsDirty(dirtyRegions);

    m_Registry.clearDirtyFlags();
}

// ===========================================================================
// Hot-reload callbacks
// ===========================================================================

void TraditionalRenderPass::onAssetsDirty(const std::vector<size_t>& dirtyAssetIndices) {
    nvrhi::CommandListHandle cl = GetDevice()->createCommandList();
    cl->open();

    const auto& assets = m_Registry.getAssets();
    for (size_t idx : dirtyAssetIndices) {
        const auto& assetDef = assets[idx];
        auto it = m_AssetIdToGPUIndex.find(assetDef.id);
        if (it != m_AssetIdToGPUIndex.end()) {
            // Re-upload existing asset slot
            _UploadAsset(assetDef, m_GPUAssets[it->second], GetDevice(), cl);
        } else {
            // New asset — append
            GPUTreeAsset gpuAsset;
            _UploadAsset(assetDef, gpuAsset, GetDevice(), cl);
            m_AssetIdToGPUIndex[assetDef.id] = m_GPUAssets.size();
            m_GPUAssets.push_back(std::move(gpuAsset));
        }
    }

    cl->close();
    GetDevice()->executeCommandList(cl);

    // Resize per-asset working arrays
    size_t numAssets = m_GPUAssets.size();
    size_t numLods   = m_Registry.getLodSegments().size();
    m_InstanceCounts.resize(numAssets, std::vector<uint32_t>(numLods, 0));
    m_InstanceOffsets.resize(numAssets, std::vector<uint32_t>(numLods, 0));
}

void TraditionalRenderPass::onRegionsDirty(const std::vector<size_t>& /*dirtyRegionIndices*/) {
    uint32_t newTotal = m_Registry.totalInstanceCount();
    uint32_t oldTotal = static_cast<uint32_t>(m_VisibleInstanceBuffer.size());

    if (newTotal != oldTotal) {
        _RebuildInstanceBuffers();
        _RebuildBindingSets();
    }

    m_UI.totalInstanceCount = newTotal;
    m_VisibleInstanceReferences.reserve(newTotal);
}

// ===========================================================================
// Render
// ===========================================================================

void TraditionalRenderPass::Render(nvrhi::IFramebuffer* framebuffer) {
    app::HiResTimer cpuTimer;
    cpuTimer.Start();

    const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();

    // when window changes size
    if (!m_TreePass.pipeline) {
        m_ViewHandler->view.SetViewport({ float(fbinfo.width), float(fbinfo.height) });
        m_ViewHandler->view.SetProjectionMatrix(
        #if XYLEM_USE_REVERSE_Z
            dm::perspProjD3DStyleReverse(
                dm::radians(60.f),
                float(fbinfo.width)/float(fbinfo.height),
                0.1f
            )
        #else
            dm::perspProjD3DStyle(
                dm::radians(60.f),
                float(fbinfo.width)/float(fbinfo.height),
                0.1f,
                1000.0f
            )
        #endif
        );
    }

    m_CommandList->open();

    m_CommandList->beginTimerQuery(m_GpuTimers[m_NextTimerIdx]);

    m_ViewHandler->view.SetViewMatrix(m_ViewHandler->camera.GetWorldToViewMatrix());
    m_ViewHandler->view.UpdateCache();

    nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));
    #if XYLEM_USE_REVERSE_Z
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer, 0.f, 0);
    #else
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer,
        m_ViewHandler->view.GetViewFrustum().farPlane().distance, 0);
    #endif

    dm::box3 sceneBbox = dm::box3::empty();
    for (const auto& region : m_Registry.getRegions())
        sceneBbox |= region.cullBox;
    const auto* terrain = m_Registry.getTerrain();
    if (terrain)
        sceneBbox |= terrain->getBbox();

    m_ViewHandler->updateShadowVolume(sceneBbox, m_Registry.getSunDirection());

    const dm::box3& shadowCasterBboxLS = m_ViewHandler->shadowCasterBboxLS;
    float maxXY = dm::max(shadowCasterBboxLS.diagonal().x, shadowCasterBboxLS.diagonal().y);
    dm::float3 grow = 0.5f * (dm::float3(maxXY, maxXY, shadowCasterBboxLS.diagonal().z)
                               - shadowCasterBboxLS.diagonal());
    dm::box3 orthoBox = shadowCasterBboxLS.grow(grow);

    dm::float4x4 lightProj = dm::orthoProjD3DStyle(
        orthoBox.m_mins.x, orthoBox.m_maxs.x,
        orthoBox.m_mins.y, orthoBox.m_maxs.y,
        orthoBox.m_mins.z, orthoBox.m_maxs.z);

    dm::float4x4 lightViewProj = dm::affineToHomogeneous(m_ViewHandler->worldToLight) * lightProj;

    Render::ConstantBufferEntry constants{};
    constants.view          = dm::affineToHomogeneous(m_ViewHandler->view.GetViewMatrix());
    constants.projection    = m_ViewHandler->view.GetProjectionMatrix();
    constants.lightViewProj = lightViewProj;
    constants.sunLightDir   = m_Registry.getSunDirection();
    m_CommandList->writeBuffer(m_Shared.constantBuffer, &constants, Render::c_ConstantBufferSize);

    _RenderSkyPass(framebuffer);
    _RenderShadowPass();
    _RenderScenePass(framebuffer);

    m_CommandList->endTimerQuery(m_GpuTimers[m_NextTimerIdx]);
    m_CommandList->close();
    GetDevice()->executeCommandList(m_CommandList);

    int prevIdx = (m_NextTimerIdx + m_QueuedFrames - 1) % m_QueuedFrames;
    if (GetDevice()->pollTimerQuery(m_GpuTimers[prevIdx]))
        m_UI.gpuFrameTimeMs = GetDevice()->getTimerQueryTime(m_GpuTimers[prevIdx]) * 1000.0f;
    m_NextTimerIdx = (m_NextTimerIdx + 1) % m_QueuedFrames;

    m_UI.totalInstanceCount   = m_Registry.totalInstanceCount();
    m_UI.visibleInstanceCount = (uint32_t)m_VisibleInstanceReferences.size();
    m_UI.culledInstanceCount  = m_UI.totalInstanceCount - m_UI.visibleInstanceCount;
    m_UI.drawCallCount        = (uint32_t)m_DrawCmds.size();

    m_UI.shadowVisibleCount   = (uint32_t)m_ShadowVisibleRefs.size();
    m_UI.shadowCulledCount    = m_UI.totalInstanceCount - m_UI.shadowVisibleCount;

    m_VisibleInstanceReferences.clear();

    cpuTimer.Stop();
    m_UI.cpuRenderTimeMs = (float)cpuTimer.Milliseconds();
}

void TraditionalRenderPass::_RenderSkyPass(nvrhi::IFramebuffer* framebuffer) {
    if (!m_SkyPass.pipeline) {
        nvrhi::GraphicsPipelineDesc pso;
        pso.VS             = m_SkyPass.vertexShader;
        pso.PS             = m_SkyPass.pixelShader;
        pso.bindingLayouts = { m_SkyPass.bindingLayout };
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
        m_SkyPass.pipeline = GetDevice()->createGraphicsPipeline(
            pso, framebuffer->getFramebufferInfo());
    }

    dm::affine3 viewToWorld = dm::affine3(m_ViewHandler->view.GetInverseViewMatrix());
    viewToWorld.m_translation = 0.f;
    dm::float4x4 clipToTranslatedWorld =
        m_ViewHandler->view.GetInverseProjectionMatrix(true) * dm::affineToHomogeneous(viewToWorld);

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

    m_CommandList->writeBuffer(m_SkyPass.constantBuffer, &skyConstants, sizeof(skyConstants));

    nvrhi::GraphicsState skyState;
    skyState.pipeline    = m_SkyPass.pipeline;
    skyState.framebuffer = framebuffer;
    skyState.viewport    = m_ViewHandler->view.GetViewportState();
    skyState.bindings    = { m_SkyPass.bindingSet };
    m_CommandList->setGraphicsState(skyState);
    m_CommandList->draw(nvrhi::DrawArguments().setVertexCount(4));
}

void TraditionalRenderPass::_RenderShadowPass() {
    const dm::affine3& worldToLight = m_ViewHandler->worldToLight;
    const auto& lodSegments = m_Registry.getLodSegments();
    uint32_t lodIndex = static_cast<uint32_t>(lodSegments.size() - 1);

    const auto& regions = m_Registry.getRegions();

    m_ShadowVisibleRefs.clear();
    std::vector<uint32_t> shadowCounts(m_GPUAssets.size(), 0);
    m_ShadowDrawCmds.clear();
    
    if (m_ViewHandler->shadowCasterBboxLS.isempty()) {
        return;
    }
    
    // Cull instances against shadow volume
    for (uint32_t ri = 0; ri < regions.size(); ri++) {
        const auto& region = regions[ri];

        dm::box3 regionLS = region.cullBox * worldToLight;
        if (!regionLS.intersects(m_ViewHandler->shadowCasterBboxLS))
            continue;

        for (uint32_t ii = 0; ii < region.instances.size(); ii++) {
            const auto& inst = region.instances[ii];
            dm::box3 instanceLS = inst.bbox * worldToLight;
            if (!instanceLS.intersects(m_ViewHandler->shadowCasterBboxLS))
                continue;

            auto it = m_AssetIdToGPUIndex.find(inst.assetId);
            if (it == m_AssetIdToGPUIndex.end()) continue;
            uint32_t gpuIdx = static_cast<uint32_t>(it->second);

            m_ShadowVisibleRefs.push_back({ ri, ii, gpuIdx, lodIndex });
            shadowCounts[gpuIdx]++;
        }
    }

    // Build draw commands
    std::vector<uint32_t> shadowWriteOff(m_GPUAssets.size(), 0);
    uint32_t shadowOffset = 0;
    for (uint32_t ai = 0; ai < shadowCounts.size(); ai++) {
        if (shadowCounts[ai] == 0) continue;
        const auto& lod = m_GPUAssets[ai].lods[lodIndex];
        m_ShadowDrawCmds.push_back({
            lod.vertexBuffer,
            lod.indexBuffer,
            nvrhi::DrawArguments()
                .setVertexCount(lod.indexCount)
                .setInstanceCount(shadowCounts[ai])
                .setStartInstanceLocation(shadowOffset)
        });
        shadowWriteOff[ai] = shadowOffset;
        shadowOffset += shadowCounts[ai];
    }

    for (const auto& ref : m_ShadowVisibleRefs) {
        uint32_t& writeOffset = shadowWriteOff[ref.treeId];
        const auto& inst = regions[ref.regionIdx].instances[ref.instanceIdx];
        m_ShadowInstanceBuffer[writeOffset] = Render::InstanceBufferEntry(
            inst.model, inst.normal, ref.treeId);
        writeOffset++;
    }

    if (!m_ShadowVisibleRefs.empty())
        m_CommandList->writeBuffer(m_ShadowPass.instanceBuffer, m_ShadowInstanceBuffer.data(),
            m_ShadowVisibleRefs.size() * sizeof(Render::InstanceBufferEntry));

    // Create shadow pipelines on demand
    if (!m_ShadowPass.treePipeline) {
        nvrhi::GraphicsPipelineDesc pso;
        pso.VS             = m_ShadowPass.treeVS;
        pso.inputLayout    = m_TreePass.inputLayout;
        pso.bindingLayouts = { m_ShadowPass.bindingLayout };
        pso.primType       = nvrhi::PrimitiveType::TriangleList;
        pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
        pso.renderState.rasterState.setCullFront();
        pso.renderState.rasterState.depthBias            = 2;
        pso.renderState.rasterState.slopeScaledDepthBias = 2.0f;
        m_ShadowPass.treePipeline = GetDevice()->createGraphicsPipeline(
            pso, m_ShadowPass.framebuffer->getFramebufferInfo());
    }
    if (!m_ShadowPass.terrainPipeline && m_TerrainPass.indexCount > 0) {
        nvrhi::GraphicsPipelineDesc pso;
        pso.VS             = m_ShadowPass.terrainVS;
        pso.inputLayout    = m_ShadowPass.terrainInputLayout;
        pso.bindingLayouts = { m_ShadowPass.bindingLayout };
        pso.primType       = nvrhi::PrimitiveType::TriangleList;
        pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
        pso.renderState.rasterState.setCullBack();
        pso.renderState.rasterState.depthBias            = 2;
        pso.renderState.rasterState.slopeScaledDepthBias = 2.0f;
        m_ShadowPass.terrainPipeline = GetDevice()->createGraphicsPipeline(
            pso, m_ShadowPass.framebuffer->getFramebufferInfo());
    }

    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList,
        m_ShadowPass.framebuffer, 1.0f, 0);

    nvrhi::Viewport shadowVP(static_cast<float>(m_ShadowRes), static_cast<float>(m_ShadowRes));
    nvrhi::ViewportState shadowVPState;
    shadowVPState.addViewportAndScissorRect(shadowVP);

    nvrhi::GraphicsState shadowState;
    shadowState.pipeline    = m_ShadowPass.treePipeline;
    shadowState.framebuffer = m_ShadowPass.framebuffer;
    shadowState.viewport    = shadowVPState;

    for (const auto& cmd : m_ShadowDrawCmds) {
        shadowState.bindings = { m_ShadowPass.bindingSet };
        shadowState.vertexBuffers = {
#if XYLEM_USE_INTERLEAVED_VERTEX_ATTRIBUTES
            { cmd.vertexBuffer, 0, 0 },
    #if !XYLEM_USE_STRUCTURED_BUFFER
            { m_ShadowPass.instanceBuffer, 1, 0 },
    #endif
#else
            { cmd.vertexBuffer, 0, offsetof(ProcGen::TreeVertex, pos) },
            { cmd.vertexBuffer, 1, offsetof(ProcGen::TreeVertex, normal) },
            { cmd.vertexBuffer, 2, offsetof(ProcGen::TreeVertex, tangent) },
            { cmd.vertexBuffer, 3, offsetof(ProcGen::TreeVertex, bitangent) },
            { cmd.vertexBuffer, 4, offsetof(ProcGen::TreeVertex, uv) },
    #if !XYLEM_USE_STRUCTURED_BUFFER
            { m_ShadowPass.instanceBuffer, 5, offsetof(Render::InstanceBufferEntry, model) },
            { m_ShadowPass.instanceBuffer, 6, offsetof(Render::InstanceBufferEntry, normal) },
    #endif
#endif
        };
        shadowState.indexBuffer = { cmd.indexBuffer, nvrhi::Format::R32_UINT, 0 };
        m_CommandList->setGraphicsState(shadowState);
        m_CommandList->drawIndexed(cmd.drawArgs);
    }

    // Draw terrain into shadow map
    const auto* terrainPtr = m_Registry.getTerrain();
    if (m_TerrainPass.indexCount > 0 && terrainPtr
        && (terrainPtr->getBbox() * worldToLight).intersects(m_ViewHandler->shadowCasterBboxLS)) {
        nvrhi::GraphicsState terrShadow;
        terrShadow.pipeline    = m_ShadowPass.terrainPipeline;
        terrShadow.framebuffer = m_ShadowPass.framebuffer;
        terrShadow.viewport    = shadowVPState;
        terrShadow.bindings    = { m_ShadowPass.bindingSet };
        terrShadow.vertexBuffers = { { m_TerrainPass.vertexBuffer, 0, 0 } };
        terrShadow.indexBuffer   = { m_TerrainPass.indexBuffer, nvrhi::Format::R32_UINT, 0 };
        m_CommandList->setGraphicsState(terrShadow);
        m_CommandList->drawIndexed(
            nvrhi::DrawArguments().setVertexCount(m_TerrainPass.indexCount));
    }
}

void TraditionalRenderPass::_RenderScenePass(nvrhi::IFramebuffer* framebuffer) {
    const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();
    const auto& lodSegments  = m_Registry.getLodSegments();
    const auto& lodDistances = m_Registry.getLodDistances();
    const auto& regions      = m_Registry.getRegions();

    // Tree pass pipeline
    if (!m_TreePass.pipeline) {
        nvrhi::GraphicsPipelineDesc psoDesc;
        psoDesc.VS           = m_TreePass.vertexShader;
        psoDesc.PS           = m_TreePass.pixelShader;
        psoDesc.inputLayout  = m_TreePass.inputLayout;
        psoDesc.bindingLayouts = { m_TreePass.bindingLayout };
        psoDesc.primType     = nvrhi::PrimitiveType::TriangleList;
    #if XYLEM_USE_REVERSE_Z
        psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
    #else
        psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    #endif
        m_TreePass.pipeline = GetDevice()->createGraphicsPipeline(psoDesc, fbinfo);
    }

    // Frustum culling & LOD assignment
    m_VisibleInstanceReferences.clear();
    m_InstanceCounts.assign(m_GPUAssets.size(), std::vector<uint32_t>(lodSegments.size(), 0));

    for (uint32_t ri = 0; ri < regions.size(); ri++) {
        const auto& region = regions[ri];
        if (!m_ViewHandler->view.IsBoxVisible(region.cullBox))
            continue;
        for (uint32_t ii = 0; ii < region.instances.size(); ii++) {
            const auto& inst = region.instances[ii];
            if (!m_ViewHandler->view.IsBoxVisible(inst.bbox)) continue;
            float dist = dm::distance(m_ViewHandler->camera.GetPosition(), inst.bbox);
            uint32_t lod = m_ViewHandler->distToLOD(dist, lodDistances);

            auto it = m_AssetIdToGPUIndex.find(inst.assetId);
            if (it == m_AssetIdToGPUIndex.end()) continue;
            uint32_t gpuIdx = static_cast<uint32_t>(it->second);

            m_VisibleInstanceReferences.push_back({ ri, ii, gpuIdx, lod });
            m_InstanceCounts[gpuIdx][lod]++;
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
                lod.vertexBuffer, 
                lod.indexBuffer,
                nvrhi::DrawArguments()
                    .setVertexCount(lod.indexCount)
                    .setInstanceCount(count)
                    .setStartInstanceLocation(instanceOffset),
                m_GPUAssets[ai].textureSetIdx 
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
        m_CommandList->writeBuffer(m_TreePass.instanceBuffer, m_VisibleInstanceBuffer.data(),
            m_VisibleInstanceReferences.size() * sizeof(Render::InstanceBufferEntry));

    // Draw trees
    nvrhi::GraphicsState state;
    state.pipeline   = m_TreePass.pipeline;
    state.framebuffer = framebuffer;
    state.viewport   = m_ViewHandler->view.GetViewportState();

    for (const auto& cmd : m_DrawCmds) {
        state.bindings = { m_TreePass.bindingSets[cmd.textureSetIdx] };
        state.vertexBuffers = {
#if XYLEM_USE_INTERLEAVED_VERTEX_ATTRIBUTES
            { cmd.vertexBuffer, 0, 0 },
#if !XYLEM_USE_STRUCTURED_BUFFER
            { m_TreePass.instanceBuffer, 1, 0 },
#endif
#else
            { cmd.vertexBuffer, 0, offsetof(ProcGen::TreeVertex, pos) },
            { cmd.vertexBuffer, 1, offsetof(ProcGen::TreeVertex, normal) },
            { cmd.vertexBuffer, 2, offsetof(ProcGen::TreeVertex, tangent) },
            { cmd.vertexBuffer, 3, offsetof(ProcGen::TreeVertex, bitangent) },
            { cmd.vertexBuffer, 4, offsetof(ProcGen::TreeVertex, uv) },
#if !XYLEM_USE_STRUCTURED_BUFFER
            { m_TreePass.instanceBuffer, 5, offsetof(Render::InstanceBufferEntry, model) },
            { m_TreePass.instanceBuffer, 6, offsetof(Render::InstanceBufferEntry, normal) },
#endif
#endif
        };
        state.indexBuffer = { cmd.indexBuffer, nvrhi::Format::R32_UINT, 0 };
        m_CommandList->setGraphicsState(state);
#if XYLEM_USE_STRUCTURED_BUFFER
        m_CommandList->setPushConstants(&instanceOffset, sizeof(uint32_t));
#endif
        m_CommandList->drawIndexed(cmd.drawArgs);
    }

    // Terrain color pass
    if (m_TerrainPass.indexCount > 0) {
        if (!m_TerrainPass.pipeline) {
            nvrhi::GraphicsPipelineDesc terrainPso;
            terrainPso.VS = m_TerrainPass.vertexShader;
            terrainPso.PS = m_TerrainPass.pixelShader;
            terrainPso.inputLayout = m_TerrainPass.inputLayout;
            terrainPso.bindingLayouts = { m_TerrainPass.bindingLayout };
            terrainPso.primType = nvrhi::PrimitiveType::TriangleList;
    #if XYLEM_USE_REVERSE_Z
            terrainPso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
    #else
            terrainPso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    #endif
            terrainPso.renderState.rasterState.setCullNone();
            m_TerrainPass.pipeline = GetDevice()->createGraphicsPipeline(terrainPso, fbinfo);
        }

        nvrhi::GraphicsState terrainState;
        terrainState.pipeline   = m_TerrainPass.pipeline;
        terrainState.framebuffer = framebuffer;
        terrainState.viewport   = m_ViewHandler->view.GetViewportState();
        terrainState.bindings   = { m_TerrainPass.bindingSet };
        terrainState.vertexBuffers = { { m_TerrainPass.vertexBuffer, 0, 0 } };
        terrainState.indexBuffer   = { m_TerrainPass.indexBuffer, nvrhi::Format::R32_UINT, 0 };
        m_CommandList->setGraphicsState(terrainState);
        m_CommandList->drawIndexed(
            nvrhi::DrawArguments()
                .setVertexCount(m_TerrainPass.indexCount));
    }
}

// input handling
bool TraditionalRenderPass::KeyboardUpdate(int key, int scancode, int action, int mods) {
    if (ImGui::GetIO().WantCaptureKeyboard) return false;
    m_ViewHandler->camera.KeyboardUpdate(key, scancode, action, mods);
    return true;
}
bool TraditionalRenderPass::MousePosUpdate(double xpos, double ypos) {
    if (ImGui::GetIO().WantCaptureMouse) return false;
    m_ViewHandler->camera.MousePosUpdate(xpos, ypos);
    return true;
}
bool TraditionalRenderPass::MouseScrollUpdate(double xoffset, double yoffset) {
    if (ImGui::GetIO().WantCaptureMouse) return false;
    m_ViewHandler->camera.MouseScrollUpdate(xoffset, yoffset);
    return true;
}
bool TraditionalRenderPass::MouseButtonUpdate(int button, int action, int mods) {
    if (ImGui::GetIO().WantCaptureMouse) return false;
    m_ViewHandler->camera.MouseButtonUpdate(button, action, mods);
    return true;
}
bool TraditionalRenderPass::JoystickButtonUpdate(int button, bool pressed) {
    m_ViewHandler->camera.JoystickButtonUpdate(button, pressed); return true;
}
bool TraditionalRenderPass::JoystickAxisUpdate(int axis, float value) {
    m_ViewHandler->camera.JoystickUpdate(axis, value); return true;
}

bool TraditionalRenderPass::_InitShared() {
    m_Shared.constantBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(Render::c_ConstantBufferSize)
            .setIsConstantBuffer(true)
            .setDebugName("ConstantBuffer")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
    );
    return !!m_Shared.constantBuffer;
}

bool TraditionalRenderPass::_InitTreePass(nvrhi::ICommandList* initCL, engine::CommonRenderPasses& commonPasses) {
    // Shaders
    m_TreePass.vertexShader = m_ShaderFactory->CreateShader("app/TraditionalRenderPass.hlsl", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_TreePass.pixelShader  = m_ShaderFactory->CreateShader("app/TraditionalRenderPass.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_TreePass.vertexShader || !m_TreePass.pixelShader) return false;

    // Vertex attributes
    nvrhi::VertexAttributeDesc attributes[] = {
#if XYLEM_USE_INTERLEAVED_VERTEX_ATTRIBUTES
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(ProcGen::TreeVertex, pos))
            .setBufferIndex(0)
            .setElementStride(sizeof(ProcGen::TreeVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("NORMAL")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(ProcGen::TreeVertex, normal))
            .setBufferIndex(0)
            .setElementStride(sizeof(ProcGen::TreeVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("TANGENT")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(ProcGen::TreeVertex, tangent))
            .setBufferIndex(0)
            .setElementStride(sizeof(ProcGen::TreeVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("BITANGENT")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(ProcGen::TreeVertex, bitangent))
            .setBufferIndex(0)
            .setElementStride(sizeof(ProcGen::TreeVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("UV")
            .setFormat(nvrhi::Format::RG32_FLOAT)
            .setOffset(offsetof(ProcGen::TreeVertex, uv))
            .setBufferIndex(0)
            .setElementStride(sizeof(ProcGen::TreeVertex)),
#if !XYLEM_USE_STRUCTURED_BUFFER
        nvrhi::VertexAttributeDesc()
            .setName("MODEL_MATRIX")
            .setFormat(nvrhi::Format::RGBA32_FLOAT)
            .setArraySize(4)
            .setOffset(offsetof(Render::InstanceBufferEntry, model))
            .setBufferIndex(1)
            .setElementStride(sizeof(Render::InstanceBufferEntry))
            .setIsInstanced(true),
        nvrhi::VertexAttributeDesc()
            .setName("NORMAL_MATRIX")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setArraySize(3)
            .setOffset(offsetof(Render::InstanceBufferEntry, normal))
            .setBufferIndex(1)
            .setElementStride(sizeof(Render::InstanceBufferEntry))
            .setIsInstanced(true),
#endif
#else
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(0)
            .setElementStride(sizeof(ProcGen::TreeVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("NORMAL")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(1)
            .setElementStride(sizeof(ProcGen::TreeVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("TANGENT")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(2)
            .setElementStride(sizeof(ProcGen::TreeVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("BITANGENT")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(3)
            .setElementStride(sizeof(ProcGen::TreeVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("UV")
            .setFormat(nvrhi::Format::RG32_FLOAT)
            .setOffset(0)
            .setBufferIndex(4)
            .setElementStride(sizeof(ProcGen::TreeVertex)),
#if !XYLEM_USE_STRUCTURED_BUFFER
        nvrhi::VertexAttributeDesc()
            .setName("MODEL_MATRIX")
            .setFormat(nvrhi::Format::RGBA32_FLOAT)
            .setArraySize(4)
            .setOffset(0)
            .setBufferIndex(5)
            .setElementStride(sizeof(Render::InstanceBufferEntry))
            .setIsInstanced(true),
        nvrhi::VertexAttributeDesc()
            .setName("NORMAL_MATRIX")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setArraySize(3)
            .setOffset(0)
            .setBufferIndex(6)
            .setElementStride(sizeof(Render::InstanceBufferEntry))
            .setIsInstanced(true),
#endif
#endif
    };
    m_TreePass.inputLayout = GetDevice()->createInputLayout(attributes, uint32_t(std::size(attributes)), m_TreePass.vertexShader);
    if (!m_TreePass.inputLayout) return false;

    // Instance buffer
    uint32_t totalInstances = std::max<uint32_t>(1, m_Registry.totalInstanceCount());
    m_TreePass.instanceBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(totalInstances * sizeof(Render::InstanceBufferEntry))
            .setStructStride(sizeof(Render::InstanceBufferEntry))
            .setDebugName("TreeInstanceBuffer")
#if XYLEM_USE_STRUCTURED_BUFFER
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource)
#else
            .setIsVertexBuffer(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
#endif
    );
    if (!m_TreePass.instanceBuffer) return false;

    // Textures
    const auto& barkTextureSets = m_Registry.getBarkTextureSets();
    engine::TextureCache textureCache(GetDevice(), std::make_shared<vfs::NativeFileSystem>(), nullptr);
    m_TreePass.textureSets.resize(barkTextureSets.size());

    for (size_t i = 0; i < barkTextureSets.size(); i++) {
        std::filesystem::path texDir = g_ProjectDirectory / "media" / barkTextureSets[i] / "textures";
        std::filesystem::path diffPath, normPath;

        for (const auto& entry : std::filesystem::directory_iterator(texDir)) {
            std::string filename = entry.path().filename().string();
            if (filename.find("_diff_") != std::string::npos && entry.path().extension() == ".jpg")
                diffPath = entry.path();
            else if (filename.find("_nor_dx_") != std::string::npos && entry.path().extension() == ".jpg")
                normPath = entry.path();
        }

        auto diffLoaded = textureCache.LoadTextureFromFile(diffPath, true,  &commonPasses, initCL);
        auto normLoaded = textureCache.LoadTextureFromFile(normPath, false, &commonPasses, initCL);

        m_TreePass.textureSets[i].diffuse   = diffLoaded->texture;
        m_TreePass.textureSets[i].normalMap = normLoaded->texture;

        if (!m_TreePass.textureSets[i].diffuse || !m_TreePass.textureSets[i].normalMap)
            return false;
    }

    // Sampler
    m_TreePass.sampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Wrap)
            .setAllFilters(true)
            .setMaxAnisotropy(8.f)
    );
    if (!m_TreePass.sampler) return false;

    // Binding layout and sets
    m_TreePass.bindingSets.resize(m_TreePass.textureSets.size());

    for (size_t i = 0; i < m_TreePass.textureSets.size(); i++) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_Shared.constantBuffer, nvrhi::BufferRange(0, Render::c_ConstantBufferSize)),
            nvrhi::BindingSetItem::Sampler(0, m_TreePass.sampler),
            nvrhi::BindingSetItem::Sampler(1, m_ShadowPass.comparisonSampler),
            nvrhi::BindingSetItem::Texture_SRV(0, m_TreePass.textureSets[i].diffuse),
            nvrhi::BindingSetItem::Texture_SRV(1, m_TreePass.textureSets[i].normalMap),
            nvrhi::BindingSetItem::Texture_SRV(2, m_ShadowPass.depthTexture),
#if XYLEM_USE_STRUCTURED_BUFFER
            nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_TreePass.instanceBuffer,
                nvrhi::Format::UNKNOWN,
                nvrhi::BufferRange(0, totalInstances * sizeof(Render::InstanceBufferEntry))),
            nvrhi::BindingSetItem::PushConstants(1, sizeof(uint32_t)),
#endif
        };

        if (i == 0) {
            if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
                    bsd, m_TreePass.bindingLayout, m_TreePass.bindingSets[0]))
                return false;
        } else {
            m_TreePass.bindingSets[i] = GetDevice()->createBindingSet(bsd, m_TreePass.bindingLayout);
            if (!m_TreePass.bindingSets[i]) return false;
        }
    }

    return !!m_TreePass.bindingLayout;
}

bool TraditionalRenderPass::_InitViewHandler() {
    m_ViewHandler = std::make_unique<ViewHandler>();
    const SceneRegistry::CameraInit& cameraInit = m_Registry.getCameraInit();
    m_ViewHandler->camera.LookTo(cameraInit.pos, cameraInit.cameraDir);
    m_ViewHandler->camera.SetMoveSpeed(cameraInit.moveSpeed);
    return !!m_ViewHandler;
}

void TraditionalRenderPass::ViewHandler::updateShadowVolume(const dm::box3& sceneBbox, dm::float3 sunDirection)
{
    worldToLight = dm::lookatZ(sunDirection) * dm::scaling(dm::float3(1.f, 1.f, -1.f));
    dm::box3 sceneBoundsLS = sceneBbox * worldToLight;

    const dm::float3& camPos = camera.GetPosition();
    float maxShadowDist = 0.f;
    for (int i = 0; i < dm::box3::numCorners; i++) {
        maxShadowDist = dm::max(
            maxShadowDist,
            dm::distance(sceneBbox.getCorner(i), camPos)
        );
    }

    const nvrhi::Viewport& vp = view.GetViewport();
    dm::float4x4 finiteProj = dm::perspProjD3DStyle(
        dm::radians(60.f), vp.width() / vp.height(), 0.1f, maxShadowDist);

    dm::frustum camFrustum(
        dm::affineToHomogeneous(view.GetViewMatrix()) * finiteProj, false);

    dm::box3 frustumLS = dm::box3::empty();
    for (int i = 0; i < dm::frustum::numCorners; i++)
        frustumLS |= worldToLight.transformPoint(camFrustum.getCorner(i));

    frustumLS.m_mins.z = sceneBoundsLS.m_mins.z;

    shadowCasterBboxLS = frustumLS & sceneBoundsLS;
}

bool TraditionalRenderPass::_InitTimerQueries() {
    for (uint32_t i = 0; i < m_QueuedFrames; i++)
        m_GpuTimers[i] = GetDevice()->createTimerQuery();
    return true;
}

bool TraditionalRenderPass::_InitShadowPass() {
    m_ShadowPass.depthTexture = GetDevice()->createTexture(
        nvrhi::TextureDesc()
            .setWidth(m_ShadowRes).setHeight(m_ShadowRes)
            .setFormat(nvrhi::Format::D32)
            .setIsRenderTarget(true)
            .setUseClearValue(true)
            .setClearValue(nvrhi::Color(1.f))
            .setInitialState(nvrhi::ResourceStates::DepthWrite)
            .setKeepInitialState(true)
            .setDebugName("ShadowMap")
    );
    if (!m_ShadowPass.depthTexture) return false;

    m_ShadowPass.framebuffer = GetDevice()->createFramebuffer(
        nvrhi::FramebufferDesc().setDepthAttachment(m_ShadowPass.depthTexture));
    if (!m_ShadowPass.framebuffer) return false;

    m_ShadowPass.treeVS = m_ShaderFactory->CreateShader(
        "app/shadow.hlsl", "tree_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_ShadowPass.terrainVS = m_ShaderFactory->CreateShader(
        "app/shadow.hlsl", "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    if (!m_ShadowPass.treeVS || !m_ShadowPass.terrainVS) return false;

    nvrhi::VertexAttributeDesc terrainShadowAttrs[] = {
        #if XYLEM_USE_INTERLEAVED_VERTEX_ATTRIBUTES
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
        #else
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(0)
            .setElementStride(sizeof(Scene::TerrainVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("NORMAL")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(1)
            .setElementStride(sizeof(Scene::TerrainVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("UV")
            .setFormat(nvrhi::Format::RG32_FLOAT)
            .setOffset(0)
            .setBufferIndex(2)
            .setElementStride(sizeof(Scene::TerrainVertex)),
        #endif
    };
    m_ShadowPass.terrainInputLayout = GetDevice()->createInputLayout(
        terrainShadowAttrs, uint32_t(std::size(terrainShadowAttrs)), m_ShadowPass.terrainVS);
    if (!m_ShadowPass.terrainInputLayout) return false;

    m_ShadowPass.comparisonSampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setMinFilter(true)
            .setMagFilter(true)
            .setMipFilter(false)
            .setReductionType(nvrhi::SamplerReductionType::Comparison)
            .setAllAddressModes(nvrhi::SamplerAddressMode::Border)
            .setBorderColor(nvrhi::Color(1.f))
    );
    if (!m_ShadowPass.comparisonSampler) return false;

    uint32_t totalInstances = std::max<uint32_t>(1, m_Registry.totalInstanceCount());
    m_ShadowPass.instanceBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(totalInstances * sizeof(Render::InstanceBufferEntry))
            .setStructStride(sizeof(Render::InstanceBufferEntry))
            .setDebugName("ShadowInstanceBuffer")
            .setIsVertexBuffer(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
    );
    if (!m_ShadowPass.instanceBuffer) return false;

    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_Shared.constantBuffer,
            nvrhi::BufferRange(0, Render::c_ConstantBufferSize)),
    };
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
            bsd, m_ShadowPass.bindingLayout, m_ShadowPass.bindingSet))
        return false;

    return true;
}

bool TraditionalRenderPass::_InitTerrainPass(nvrhi::ICommandList* initCL) {
    const auto* terrain = m_Registry.getTerrain();
    if (!terrain) return true;

    const auto& verts   = terrain->getVertices();
    const auto& indices = terrain->getIndices();
    if (verts.empty() || indices.empty()) return true;

    m_TerrainPass.indexCount = static_cast<uint32_t>(indices.size());

    m_TerrainPass.vertexShader = m_ShaderFactory->CreateShader("app/terrain.hlsl", "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_TerrainPass.pixelShader  = m_ShaderFactory->CreateShader("app/terrain.hlsl", "terrain_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_TerrainPass.vertexShader || !m_TerrainPass.pixelShader) return false;

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
    m_TerrainPass.inputLayout = GetDevice()->createInputLayout(
        terrainAttrs, uint32_t(std::size(terrainAttrs)), m_TerrainPass.vertexShader);
    if (!m_TerrainPass.inputLayout) return false;

    nvrhi::BufferDesc vbDesc;
    vbDesc.isVertexBuffer = true;
    vbDesc.byteSize       = verts.size() * sizeof(Scene::TerrainVertex);
    vbDesc.debugName      = "TerrainVB";
    vbDesc.initialState   = nvrhi::ResourceStates::CopyDest;
    m_TerrainPass.vertexBuffer = GetDevice()->createBuffer(vbDesc);
    initCL->beginTrackingBufferState(m_TerrainPass.vertexBuffer, nvrhi::ResourceStates::CopyDest);
    initCL->writeBuffer(m_TerrainPass.vertexBuffer, verts.data(), vbDesc.byteSize);
    initCL->setPermanentBufferState(m_TerrainPass.vertexBuffer, nvrhi::ResourceStates::VertexBuffer);

    nvrhi::BufferDesc ibDesc;
    ibDesc.isIndexBuffer = true;
    ibDesc.byteSize      = indices.size() * sizeof(uint32_t);
    ibDesc.debugName     = "TerrainIB";
    ibDesc.initialState  = nvrhi::ResourceStates::CopyDest;
    m_TerrainPass.indexBuffer = GetDevice()->createBuffer(ibDesc);
    initCL->beginTrackingBufferState(m_TerrainPass.indexBuffer, nvrhi::ResourceStates::CopyDest);
    initCL->writeBuffer(m_TerrainPass.indexBuffer, indices.data(), ibDesc.byteSize);
    initCL->setPermanentBufferState(m_TerrainPass.indexBuffer, nvrhi::ResourceStates::IndexBuffer);

    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_Shared.constantBuffer,
            nvrhi::BufferRange(0, Render::c_ConstantBufferSize)),
        nvrhi::BindingSetItem::Sampler(0, m_ShadowPass.comparisonSampler),
        nvrhi::BindingSetItem::Texture_SRV(0, m_ShadowPass.depthTexture),
    };
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
            bsd, m_TerrainPass.bindingLayout, m_TerrainPass.bindingSet))
        return false;

    return true;
}

bool TraditionalRenderPass::_InitSkyPass() {
    m_SkyPass.vertexShader = m_ShaderFactory->CreateShader("app/sky.hlsl", "sky_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_SkyPass.pixelShader  = m_ShaderFactory->CreateShader("app/sky.hlsl", "sky_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_SkyPass.vertexShader || !m_SkyPass.pixelShader) return false;

    nvrhi::BufferDesc cbDesc;
    cbDesc.byteSize         = sizeof(SkyConstants);
    cbDesc.isConstantBuffer = true;
    cbDesc.isVolatile       = true;
    cbDesc.maxVersions      = 16;
    cbDesc.debugName        = "SkyConstants";
    m_SkyPass.constantBuffer = GetDevice()->createBuffer(cbDesc);
    if (!m_SkyPass.constantBuffer) return false;

    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_SkyPass.constantBuffer),
    };
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
            bsd, m_SkyPass.bindingLayout, m_SkyPass.bindingSet))
        return false;

    return true;
}
