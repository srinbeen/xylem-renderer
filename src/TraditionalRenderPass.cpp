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

    m_TreePass.instanceBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(totalInstances * sizeof(Render::InstanceBufferEntry))
            .setStructStride(sizeof(Render::InstanceBufferEntry))
            .setDebugName("TreeInstanceBuffer")
            .setIsVertexBuffer(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
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
    for (auto& csd : m_CascadeShadowData)
        csd.instanceBuffer.resize(totalInstances);
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

        initCL->close();
        GetDevice()->executeCommandList(initCL);
    }

    // Persistent render command list for per-frame drawing
    m_CommandList = GetDevice()->createCommandList();

    // _InitTimerQueries();

    uint32_t totalInstances = m_Registry.totalInstanceCount();
    m_UI.totalInstanceCount = totalInstances;

    m_VisibleInstanceReferences.reserve(totalInstances);

    size_t numAssets = m_GPUAssets.size();
    size_t numLods   = m_Registry.getLodSegments().size();

    m_InstanceCounts.resize(numAssets, std::vector<uint32_t>(numLods, 0));
    m_InstanceOffsets.assign(numAssets, std::vector<uint32_t>(numLods, 0));

    m_VisibleInstanceBuffer.resize(std::max<uint32_t>(1, totalInstances));
    m_DrawCmds.reserve(numAssets * numLods);

    for (auto& csd : m_CascadeShadowData) {
        csd.instanceBuffer.resize(std::max<uint32_t>(1, totalInstances));
        csd.drawCmds.reserve(numAssets);
    }

    return true;
}

// ===========================================================================
// Animate — drives the hot-reload cycle
// ===========================================================================

void TraditionalRenderPass::Animate(float seconds) {
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
    m_UI.shadowMapTexture = m_ShadowPass.depthTexture.Get();
    m_UI.shadowCascadeTextures.resize(Render::c_NumCascades);
    for (uint32_t c = 0; c < Render::c_NumCascades; c++)
        m_UI.shadowCascadeTextures[c] = m_ShadowPass.cascadeDebugTextures[c].Get();

    app::HiResTimer cpuTimer;
    cpuTimer.Start();

    const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();

    // when window changes size
    if (!m_TreePass.pipeline) {
        m_ViewHandler.view.SetViewport({ float(fbinfo.width), float(fbinfo.height) });
        m_ViewHandler.view.SetProjectionMatrix(
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

    // m_CommandList->beginTimerQuery(m_GpuTimers[m_NextTimerIdx]);

    m_ViewHandler.view.SetViewMatrix(m_ViewHandler.camera.GetWorldToViewMatrix());
    m_ViewHandler.view.UpdateCache();

    nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));
    #if XYLEM_USE_REVERSE_Z
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer, 0.f, 0);
    #else
    float farPlaneDistance = m_ViewHandler.view.GetProjectionFrustum().farPlane().distance;
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer,
        farPlaneDistance, 0);
    #endif

    dm::box3 sceneBbox = dm::box3::empty();
    for (const auto& region : m_Registry.getRegions())
        sceneBbox |= region.cullBox;
    const auto* terrain = m_Registry.getTerrain();
    if (terrain)
        sceneBbox |= terrain->getBbox();

    // Compute max shadow distance from scene bbox
    const dm::float3& camPos = m_ViewHandler.camera.GetPosition();
    const dm::float3& camDir = m_ViewHandler.camera.GetDir();
    float maxShadowDist = 0.f;
    for (int i = 0; i < dm::box3::numCorners; i++) {
        dm::float3 corner = sceneBbox.getCorner(i);
        dm::float3 camToCorner = corner - camPos;
        float cornerDir = dm::dot(camToCorner, camDir);
        if (cornerDir > 0.0f) maxShadowDist = dm::max(maxShadowDist, cornerDir);
    }
    // looking away
    maxShadowDist = dm::max(maxShadowDist, 1.f);

    float aspectRatio = float(fbinfo.width) / float(fbinfo.height);
    m_ViewHandler.computeCascades(sceneBbox, m_Registry.getSunDirection(),
        0.1f, maxShadowDist, aspectRatio, dm::radians(60.f), m_ShadowRes);

    Render::ConstantBufferEntry constants{};
    constants.viewProj  = m_ViewHandler.view.GetViewProjectionMatrix();
    constants.viewMatrix = dm::affineToHomogeneous(m_ViewHandler.view.GetViewMatrix());
    for (uint32_t c = 0; c < Render::c_NumCascades; c++)
        constants.lightViewProj[c] = m_ViewHandler.cascades[c].lightViewProj;
    constants.sunLightDir    = m_Registry.getSunDirection();
    constants.cascadeSplits  = m_ViewHandler.cascadeSplitDistances;
    m_CommandList->writeBuffer(m_Shared.constantBuffer, &constants, Render::c_ConstantBufferSize);

    m_CommandList->beginMarker("Draw");

    m_CommandList->beginMarker("Sky");
    _RenderSkyPass(framebuffer);
    m_CommandList->endMarker();

    m_CommandList->beginMarker("Shadow");
    _RenderShadowPass();
    m_CommandList->endMarker();

    // Copy cascade slices to debug textures for UI visualization
    if (m_UI.showShadowMap) {
        for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
            m_CommandList->copyTexture(
                m_ShadowPass.cascadeDebugTextures[c], nvrhi::TextureSlice(),
                m_ShadowPass.depthTexture, nvrhi::TextureSlice().setArraySlice(c));
        }
    }

    m_CommandList->beginMarker("Scene");
    _RenderScenePass(framebuffer);
    m_CommandList->endMarker();

    m_CommandList->endMarker(); // Draw

    // m_CommandList->endTimerQuery(m_GpuTimers[m_NextTimerIdx]);
    m_CommandList->close();
    GetDevice()->executeCommandList(m_CommandList);

    // int prevIdx = (m_NextTimerIdx + m_QueuedFrames - 1) % m_QueuedFrames;
    // if (GetDevice()->pollTimerQuery(m_GpuTimers[prevIdx]))
    //     m_UI.gpuFrameTimeMs = GetDevice()->getTimerQueryTime(m_GpuTimers[prevIdx]) * 1000.0f;

    // m_NextTimerIdx = (m_NextTimerIdx + 1) % m_QueuedFrames;

    m_UI.totalInstanceCount   = m_Registry.totalInstanceCount();
    m_UI.visibleInstanceCount = (uint32_t)m_VisibleInstanceReferences.size();
    m_UI.culledInstanceCount  = m_UI.totalInstanceCount - m_UI.visibleInstanceCount;
    m_UI.drawCallCount        = (uint32_t)m_DrawCmds.size();

    m_UI.shadowVisibleCount   = m_TotalShadowInstancesDrawn;
    m_UI.shadowCulledCount    = m_UI.totalInstanceCount - m_TotalShadowInstancesDrawn;

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

    m_CommandList->writeBuffer(m_SkyPass.constantBuffer, &skyConstants, sizeof(skyConstants));

    nvrhi::GraphicsState skyState;
    skyState.pipeline    = m_SkyPass.pipeline;
    skyState.framebuffer = framebuffer;
    skyState.viewport    = m_ViewHandler.view.GetViewportState();
    skyState.bindings    = { m_SkyPass.bindingSet };
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
    if (!m_ShadowPass.treePipeline) {
        nvrhi::GraphicsPipelineDesc pso;
        pso.VS             = m_ShadowPass.treeVS;
        pso.inputLayout    = m_ShadowPass.treeInputLayout;
        pso.bindingLayouts = { m_ShadowPass.bindingLayout };
        pso.primType       = nvrhi::PrimitiveType::TriangleList;
        pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
        pso.renderState.rasterState.setCullFront();
        pso.renderState.rasterState.depthBias            = 2;
        pso.renderState.rasterState.slopeScaledDepthBias = 2.0f;
        m_ShadowPass.treePipeline = GetDevice()->createGraphicsPipeline(
            pso, m_ShadowPass.framebuffers[0]->getFramebufferInfo());
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
            pso, m_ShadowPass.framebuffers[0]->getFramebufferInfo());
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
                    .setStartInstanceLocation(shadowOffset)
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
            m_CommandList->writeBuffer(m_ShadowPass.instanceBuffer, csd.instanceBuffer.data(),
                csd.visibleRefs.size() * sizeof(Render::InstanceBufferEntry));

        // Clear this cascade's framebuffer
        nvrhi::utils::ClearDepthStencilAttachment(m_CommandList,
            m_ShadowPass.framebuffers[cascade], 1.0f, 0);

        // Draw trees into this cascade
        nvrhi::GraphicsState shadowState;
        shadowState.pipeline    = m_ShadowPass.treePipeline;
        shadowState.framebuffer = m_ShadowPass.framebuffers[cascade];
        shadowState.viewport    = shadowVPState;

        for (const auto& cmd : csd.drawCmds) {
            shadowState.bindings = { m_ShadowPass.bindingSet };
            shadowState.vertexBuffers = {
                { cmd.positionBuffer, 0, 0 },
                { m_ShadowPass.instanceBuffer, 1, 0 },
            };
            shadowState.indexBuffer = { cmd.indexBuffer, nvrhi::Format::R32_UINT, 0 };
            m_CommandList->setGraphicsState(shadowState);
            m_CommandList->setPushConstants(&cascade, sizeof(cascade));

            m_CommandList->drawIndexed(cmd.drawArgs);
        }

        // Draw terrain into this cascade
        if (m_TerrainPass.indexCount > 0 && terrainPtr
            && (terrainPtr->getBbox() * worldToLight).intersects(m_ViewHandler.shadowCasterBboxLS)) {
            nvrhi::GraphicsState terrShadow;
            terrShadow.pipeline    = m_ShadowPass.terrainPipeline;
            terrShadow.framebuffer = m_ShadowPass.framebuffers[cascade];
            terrShadow.viewport    = shadowVPState;
            terrShadow.bindings    = { m_ShadowPass.bindingSet };
            terrShadow.vertexBuffers = { { m_TerrainPass.vertexBuffer, 0, 0 } };
            terrShadow.indexBuffer   = { m_TerrainPass.indexBuffer, nvrhi::Format::R32_UINT, 0 };
            m_CommandList->setGraphicsState(terrShadow);
            // doesn'cascade doesn't matter for terrain render, but needed for root signature
            m_CommandList->setPushConstants(&cascade, sizeof(cascade));
            m_CommandList->drawIndexed(
                nvrhi::DrawArguments().setVertexCount(m_TerrainPass.indexCount));
        }
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
            uint32_t lod = m_ViewHandler.distToLOD(dist, lodDistances);

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
                lod.vbs,
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
    state.viewport   = m_ViewHandler.view.GetViewportState();

    for (const auto& cmd : m_DrawCmds) {
        state.bindings = { m_TreePass.bindingSets[cmd.textureSetIdx] };
        state.vertexBuffers = {
            { cmd.vertexBuffers.position,  0, 0 },
            { cmd.vertexBuffers.normal,    1, 0 },
            { cmd.vertexBuffers.tangent,   2, 0 },
            { cmd.vertexBuffers.bitangent, 3, 0 },
            { cmd.vertexBuffers.uv,        4, 0 },
            { m_TreePass.instanceBuffer, 5, 0 },
        };
        state.indexBuffer = { cmd.indexBuffer, nvrhi::Format::R32_UINT, 0 };
        m_CommandList->setGraphicsState(state);
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
        terrainState.viewport   = m_ViewHandler.view.GetViewportState();
        terrainState.bindings   = { m_TerrainPass.bindingSet };
        terrainState.vertexBuffers = { { m_TerrainPass.vertexBuffer, 0, 0 } };
        terrainState.indexBuffer   = { m_TerrainPass.indexBuffer, nvrhi::Format::R32_UINT, 0 };
        m_CommandList->setGraphicsState(terrainState);
        m_CommandList->drawIndexed(
            nvrhi::DrawArguments()
                .setVertexCount(m_TerrainPass.indexCount));
    }
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

    // Vertex attributes — SoA: one buffer per attribute stream
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
    m_TreePass.inputLayout = GetDevice()->createInputLayout(attributes, uint32_t(std::size(attributes)), m_TreePass.vertexShader);
    if (!m_TreePass.inputLayout) return false;

    // Instance buffer
    uint32_t totalInstances = std::max<uint32_t>(1, m_Registry.totalInstanceCount());
    m_TreePass.instanceBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(totalInstances * sizeof(Render::InstanceBufferEntry))
            .setStructStride(sizeof(Render::InstanceBufferEntry))
            .setDebugName("TreeInstanceBuffer")
            .setIsVertexBuffer(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
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

// bool TraditionalRenderPass::_InitTimerQueries() {
//     for (uint32_t i = 0; i < m_QueuedFrames; i++)
//         m_GpuTimers[i] = GetDevice()->createTimerQuery();
//     return true;
// }

bool TraditionalRenderPass::_InitShadowPass() {
    m_ShadowPass.depthTexture = GetDevice()->createTexture(
        nvrhi::TextureDesc()
            .setWidth(m_ShadowRes).setHeight(m_ShadowRes)
            .setArraySize(Render::c_NumCascades)
            .setDimension(nvrhi::TextureDimension::Texture2DArray)
            .setFormat(nvrhi::Format::D32)
            .setIsRenderTarget(true)
            .setUseClearValue(true)
            .setClearValue(nvrhi::Color(1.f))
            .setInitialState(nvrhi::ResourceStates::DepthWrite)
            .setKeepInitialState(true)
            .setDebugName("ShadowMapArray")
    );
    if (!m_ShadowPass.depthTexture) return false;

    for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
        m_ShadowPass.framebuffers[c] = GetDevice()->createFramebuffer(
            nvrhi::FramebufferDesc().setDepthAttachment(
                nvrhi::FramebufferAttachment()
                    .setTexture(m_ShadowPass.depthTexture)
                    .setArraySlice(c)
            ));
        if (!m_ShadowPass.framebuffers[c]) return false;

        // Per-cascade debug texture for UI visualization (matches Hi-Z debug pattern)
        m_ShadowPass.cascadeDebugTextures[c] = GetDevice()->createTexture(
            nvrhi::TextureDesc()
                .setWidth(m_ShadowRes).setHeight(m_ShadowRes)
                .setFormat(nvrhi::Format::R32_FLOAT)
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true)
                .setDebugName("ShadowCascadeDebug_" + std::to_string(c))
        );
    }

    m_ShadowPass.treeVS = m_ShaderFactory->CreateShader(
        "app/shadow.hlsl", "tree_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_ShadowPass.terrainVS = m_ShaderFactory->CreateShader(
        "app/shadow.hlsl", "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    if (!m_ShadowPass.treeVS || !m_ShadowPass.terrainVS) return false;

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
    m_ShadowPass.treeInputLayout = GetDevice()->createInputLayout(
        treeShadowAttrs, uint32_t(std::size(treeShadowAttrs)), m_ShadowPass.treeVS);
    if (!m_ShadowPass.treeInputLayout) return false;

    nvrhi::VertexAttributeDesc terrainShadowAttrs[] = {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(Scene::TerrainVertex, pos))
            .setBufferIndex(0)
            .setElementStride(sizeof(Scene::TerrainVertex)),
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
        nvrhi::BindingSetItem::PushConstants(1, sizeof(uint32_t)),
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
