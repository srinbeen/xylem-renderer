#include "include/TraditionalRenderPass.hpp"
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

// public
bool TraditionalRenderPass::Init() {
    // CommonRenderPasses must be constructed before opening initCL — its constructor
    // opens its own temporary command list to upload placeholder textures.
    engine::CommonRenderPasses commonPasses(GetDevice(), m_ShaderFactory);

    // Dedicated init command list for uploads + mipmap generation.
    // Freed after init to release DX12 upload heap memory.
    nvrhi::CommandListHandle initCL = GetDevice()->createCommandList();
    initCL->open();

    if (!SceneLoader::Load(g_SceneConfigDirectory, GetDevice(), initCL, m_Scene)) return false;
    if (!_InitShared())                                return false;
    if (!_InitShadowPass())                            return false;
    if (!_InitTreePass(initCL, commonPasses))          return false;
    if (!_InitTerrainPass(initCL))                     return false;
    if (!_InitSkyPass())                               return false;
    if (!_InitViewHandler())                           return false;

    initCL->close();
    GetDevice()->executeCommandList(initCL);
    // initCL goes out of scope — upload buffers released

    // Persistent render command list for per-frame drawing
    m_CommandList = GetDevice()->createCommandList();

    _InitTimerQueries();

    m_UI.totalInstanceCount = m_Scene.regionManager.getTotalInstanceCount();

    m_VisibleInstanceReferences.reserve(m_Scene.regionManager.getTotalInstanceCount());

    // Initialize dynamic LOD counters
    m_InstanceCounts.resize(m_Scene.assets.size(), std::vector<uint32_t>(m_Scene.lodSegments.size(), 0));
    m_InstanceOffsets.assign(m_Scene.assets.size(), std::vector<uint32_t>(m_Scene.lodSegments.size(), 0));

    m_VisibleInstanceBuffer.resize(m_Scene.regionManager.getTotalInstanceCount());
    m_DrawCmds.reserve(m_Scene.assets.size() * m_Scene.lodSegments.size());

    m_ShadowInstanceBuffer.resize(m_Scene.regionManager.getTotalInstanceCount());
    m_ShadowDrawCmds.reserve(m_Scene.assets.size());

    return true;
}

void TraditionalRenderPass::Animate(float seconds) {
    m_ViewHandler->camera.Animate(seconds);
    GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
}

void TraditionalRenderPass::Render(nvrhi::IFramebuffer* framebuffer) {
    app::HiResTimer cpuTimer;
    cpuTimer.Start();

    const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();

    if (!m_TreePass.pipeline) {
        nvrhi::GraphicsPipelineDesc psoDesc;
        psoDesc.VS           = m_TreePass.vertexShader;
        psoDesc.PS           = m_TreePass.pixelShader;
        psoDesc.inputLayout  = m_TreePass.inputLayout;
        psoDesc.bindingLayouts = { m_TreePass.bindingLayout };
        psoDesc.primType     = nvrhi::PrimitiveType::TriangleList;
#if PIPELINER_USE_REVERSE_Z
        psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
#else
        psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
#endif
        m_TreePass.pipeline = GetDevice()->createGraphicsPipeline(psoDesc, fbinfo);

        m_ViewHandler->view.SetViewport({ float(fbinfo.width), float(fbinfo.height) });
#if PIPELINER_USE_REVERSE_Z
        m_ViewHandler->view.SetProjectionMatrix(
            dm::perspProjD3DStyleReverse(dm::radians(60.f), float(fbinfo.width)/float(fbinfo.height), 0.1f));
#else
        m_ViewHandler->view.SetProjectionMatrix(
            dm::perspProjD3DStyle(dm::radians(60.f), float(fbinfo.width)/float(fbinfo.height), 0.1f,
                std::numeric_limits<float>::max()));
#endif
    }

    m_CommandList->open();

    m_CommandList->beginTimerQuery(m_GpuTimers[m_NextTimerIdx]);

    m_ViewHandler->view.SetViewMatrix(m_ViewHandler->camera.GetWorldToViewMatrix());
    m_ViewHandler->view.UpdateCache();

    nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));
#if PIPELINER_USE_REVERSE_Z
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer, 0.f, 0);
#else
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer,
        m_ViewHandler->view.GetViewFrustum().farPlane().distance, 0);
#endif

    // Compute light view-projection for shadow mapping
    dm::float3 sunDir = dm::normalize(dm::float3(-1.f, -1.f, 0.5f));

    dm::box3 sceneBbox = dm::box3::empty();
    for (uint32_t ri = 0; ri < m_Scene.regionManager.size(); ri++)
        sceneBbox |= m_Scene.regionManager[ri].cullBox;
    if (m_Scene.terrain)
        sceneBbox |= m_Scene.terrain->getBbox();

    // Build light view following Donut's PlanarShadowMap pattern
    dm::affine3 lightToWorld = dm::lookatZ(sunDir);
    lightToWorld = dm::scaling(dm::float3(1.f, 1.f, -1.f)) * lightToWorld;
    dm::affine3 worldToLight = dm::inverse(lightToWorld);

    dm::box3 boundsInLight = sceneBbox * worldToLight;
    // Make shadow texels square
    float maxXY = std::max(boundsInLight.diagonal().x, boundsInLight.diagonal().y);
    dm::float3 grow = 0.5f * (dm::float3(maxXY, maxXY, boundsInLight.diagonal().z) - boundsInLight.diagonal());
    boundsInLight = boundsInLight.grow(grow);

    dm::float4x4 lightProj = dm::orthoProjD3DStyle(
        boundsInLight.m_mins.x, boundsInLight.m_maxs.x,
        boundsInLight.m_mins.y, boundsInLight.m_maxs.y,
        -boundsInLight.m_maxs.z, -boundsInLight.m_mins.z);

    dm::float4x4 lightViewProj = dm::affineToHomogeneous(worldToLight) * lightProj;

    Render::ConstantBufferEntry constants{};
    constants.view          = dm::affineToHomogeneous(m_ViewHandler->view.GetViewMatrix());
    constants.projection    = m_ViewHandler->view.GetProjectionMatrix();
    constants.lightViewProj = lightViewProj;
    constants.sunLightDir   = sunDir;
    m_CommandList->writeBuffer(m_Shared.constantBuffer, &constants, Render::c_ConstantBufferSize);

    // --- Sky pass (fullscreen procedural sky) ---
    {
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
#if PIPELINER_USE_REVERSE_Z
                .setDepthFunc(nvrhi::ComparisonFunc::GreaterOrEqual);
#else
                .setDepthFunc(nvrhi::ComparisonFunc::LessOrEqual);
#endif
            m_SkyPass.pipeline = GetDevice()->createGraphicsPipeline(pso, fbinfo);
        }

        dm::affine3 viewToWorld = dm::affine3(m_ViewHandler->view.GetInverseViewMatrix());
        viewToWorld.m_translation = 0.f;
        dm::float4x4 clipToTranslatedWorld =
            m_ViewHandler->view.GetInverseProjectionMatrix(true) * dm::affineToHomogeneous(viewToWorld);

        SkyConstants skyConstants{};
        skyConstants.matClipToTranslatedWorld = clipToTranslatedWorld;

        // Fill procedural sky parameters
        auto& p         = skyConstants.params;
        p.directionToLight  = dm::normalize(-sunDir);
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
        m_CommandList->draw(nvrhi::DrawArguments().setVertexCount(4).setInstanceCount(1));
    }

    // --- Shadow pass: draw ALL instances (lowest LOD, no culling) ---
    {
        uint32_t shadowLod = static_cast<uint32_t>(m_Scene.lodSegments.size() - 1);
        m_ShadowDrawCmds.clear();
        std::vector<uint32_t> shadowCounts(m_Scene.assets.size(), 0);

        // Count instances per asset
        for (uint32_t ri = 0; ri < m_Scene.regionManager.size(); ri++) {
            const auto& region = m_Scene.regionManager[ri];
            for (uint32_t ii = 0; ii < region.instanceCount; ii++) {
                uint32_t tid = region.instanceBuffer[ii].treeId;
                shadowCounts[tid]++;
            }
        }

        // Build draw commands and compute write offsets
        std::vector<uint32_t> shadowWriteOff(m_Scene.assets.size(), 0);
        uint32_t shadowOffset = 0;
        for (uint32_t ai = 0; ai < shadowCounts.size(); ai++) {
            if (shadowCounts[ai] == 0) continue;
            const auto& lod = m_Scene.assets[ai].lods[shadowLod];
            m_ShadowDrawCmds.push_back({ lod.vertexBuffer, lod.indexBuffer,
                nvrhi::DrawArguments()
                    .setVertexCount(lod.indexCount)
                    .setInstanceCount(shadowCounts[ai])
                    .setStartInstanceLocation(shadowOffset),
                m_Scene.assets[ai].textureSetIdx });
            shadowWriteOff[ai] = shadowOffset;
            shadowOffset += shadowCounts[ai];
        }

        // Fill instance buffer (all instances, no culling)
        for (uint32_t ri = 0; ri < m_Scene.regionManager.size(); ri++) {
            const auto& region = m_Scene.regionManager[ri];
            for (uint32_t ii = 0; ii < region.instanceCount; ii++) {
                uint32_t tid = region.instanceBuffer[ii].treeId;
                m_ShadowInstanceBuffer[shadowWriteOff[tid]++] = region.instanceBuffer[ii];
            }
        }

        if (shadowOffset > 0)
            m_CommandList->writeBuffer(m_ShadowPass.instanceBuffer,
                m_ShadowInstanceBuffer.data(),
                shadowOffset * sizeof(Render::InstanceBufferEntry));
    }

    m_VisibleInstanceReferences.clear();
    m_InstanceCounts.assign(m_Scene.assets.size(), std::vector<uint32_t>(m_Scene.lodSegments.size(), 0));
    uint32_t culled = 0;

    for (uint32_t ri = 0; ri < m_Scene.regionManager.size(); ri++) {
        const auto& region = m_Scene.regionManager[ri];
        if (!m_ViewHandler->view.IsBoxVisible(region.cullBox)) {
            culled += region.instanceCount;
            continue;
        }
        for (uint32_t ii = 0; ii < region.instanceCount; ii++) {
            if (!m_ViewHandler->view.IsBoxVisible(region.instanceBbox[ii])) { culled++; continue; }
            float dist = dm::distance(m_ViewHandler->camera.GetPosition(), region.instanceBbox[ii]);
            uint32_t lod = m_ViewHandler->distToLOD(dist, m_Scene.lodDistances);
            m_VisibleInstanceReferences.push_back({ ri, ii, region.instanceBuffer[ii].treeId, lod });
            m_InstanceCounts[region.instanceBuffer[ii].treeId][lod]++;
        }
    }

    m_DrawCmds.clear();
    m_InstanceOffsets.assign(m_Scene.assets.size(), std::vector<uint32_t>(m_Scene.lodSegments.size(), 0));
    uint32_t instanceOffset = 0;

    for (uint32_t ai = 0; ai < m_InstanceCounts.size(); ai++) {
        for (uint32_t li = 0; li < m_Scene.lodSegments.size(); li++) {
            uint32_t count = m_InstanceCounts[ai][li];
            if (count == 0) continue;
            const auto& lod = m_Scene.assets[ai].lods[li];
            m_DrawCmds.push_back({ lod.vertexBuffer, lod.indexBuffer,
                nvrhi::DrawArguments()
                    .setVertexCount(lod.indexCount)
                    .setInstanceCount(count)
                    .setStartInstanceLocation(instanceOffset),
                m_Scene.assets[ai].textureSetIdx });
            m_InstanceOffsets[ai][li] = instanceOffset;
            instanceOffset += count;
        }
    }

    for (const auto& ref : m_VisibleInstanceReferences) {
        uint32_t& writeOff = m_InstanceOffsets[ref.treeId][ref.lodID];
        m_VisibleInstanceBuffer[writeOff] = m_Scene.regionManager[ref.regionIdx].instanceBuffer[ref.instanceIdx];
        writeOff++;
    }

    if (!m_VisibleInstanceReferences.empty())
        m_CommandList->writeBuffer(m_TreePass.instanceBuffer, m_VisibleInstanceBuffer.data(),
            m_VisibleInstanceReferences.size() * sizeof(Render::InstanceBufferEntry));

    // --- Shadow pass ---
    {
        if (!m_ShadowPass.treePipeline) {
            nvrhi::GraphicsPipelineDesc pso;
            pso.VS             = m_ShadowPass.treeVS;
            pso.inputLayout    = m_TreePass.inputLayout;
            pso.bindingLayouts = { m_ShadowPass.bindingLayout };
            pso.primType       = nvrhi::PrimitiveType::TriangleList;
            pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
            pso.renderState.rasterState.setCullNone();
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

        // Draw trees into shadow map
        nvrhi::GraphicsState shadowState;
        shadowState.pipeline    = m_ShadowPass.treePipeline;
        shadowState.framebuffer = m_ShadowPass.framebuffer;
        shadowState.viewport    = shadowVPState;

        for (const auto& cmd : m_ShadowDrawCmds) {
            shadowState.bindings = { m_ShadowPass.bindingSet };
            shadowState.vertexBuffers = {
#if PIPELINER_USE_INTERLEAVED_VERTEX_ATTRIBUTES
                { cmd.vertexBuffer, 0, 0 },
#if !PIPELINER_USE_STRUCTURED_BUFFER
                { m_ShadowPass.instanceBuffer, 1, 0 },
#endif
#else
                { cmd.vertexBuffer, 0, offsetof(ProcGen::TreeVertex, pos) },
                { cmd.vertexBuffer, 1, offsetof(ProcGen::TreeVertex, normal) },
                { cmd.vertexBuffer, 2, offsetof(ProcGen::TreeVertex, tangent) },
                { cmd.vertexBuffer, 3, offsetof(ProcGen::TreeVertex, bitangent) },
                { cmd.vertexBuffer, 4, offsetof(ProcGen::TreeVertex, uv) },
#if !PIPELINER_USE_STRUCTURED_BUFFER
                { m_ShadowPass.instanceBuffer, 5, offsetof(Render::InstanceBufferEntry, model) },
                { m_ShadowPass.instanceBuffer, 6, offsetof(Render::InstanceBufferEntry, normal) },
#endif
#endif
            };
            shadowState.indexBuffer = { cmd.indexBuffer, nvrhi::Format::R32_UINT, 0 };
            m_CommandList->setGraphicsState(shadowState);
#if PIPELINER_USE_STRUCTURED_BUFFER
            m_CommandList->setPushConstants(&instanceOffset, sizeof(uint32_t));
#endif
            m_CommandList->drawIndexed(cmd.drawArgs);
        }

        // Draw terrain into shadow map
        if (m_TerrainPass.indexCount > 0) {
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

    // --- Main color pass (trees) ---
    nvrhi::GraphicsState state;
    state.pipeline   = m_TreePass.pipeline;
    state.framebuffer = framebuffer;
    state.viewport   = m_ViewHandler->view.GetViewportState();

    for (const auto& cmd : m_DrawCmds) {
        state.bindings = { m_TreePass.bindingSets[cmd.textureSetIdx] };
        state.vertexBuffers = {
#if PIPELINER_USE_INTERLEAVED_VERTEX_ATTRIBUTES
            { cmd.vertexBuffer, 0, 0 },
#if !PIPELINER_USE_STRUCTURED_BUFFER
            { m_TreePass.instanceBuffer, 1, 0 },
#endif
#else
            { cmd.vertexBuffer, 0, offsetof(ProcGen::TreeVertex, pos) },
            { cmd.vertexBuffer, 1, offsetof(ProcGen::TreeVertex, normal) },
            { cmd.vertexBuffer, 2, offsetof(ProcGen::TreeVertex, tangent) },
            { cmd.vertexBuffer, 3, offsetof(ProcGen::TreeVertex, bitangent) },
            { cmd.vertexBuffer, 4, offsetof(ProcGen::TreeVertex, uv) },
#if !PIPELINER_USE_STRUCTURED_BUFFER
            { m_TreePass.instanceBuffer, 5, offsetof(Render::InstanceBufferEntry, model) },
            { m_TreePass.instanceBuffer, 6, offsetof(Render::InstanceBufferEntry, normal) },
#endif
#endif
        };
        state.indexBuffer = { cmd.indexBuffer, nvrhi::Format::R32_UINT, 0 };
        m_CommandList->setGraphicsState(state);
#if PIPELINER_USE_STRUCTURED_BUFFER
        m_CommandList->setPushConstants(&instanceOffset, sizeof(uint32_t));
#endif
        m_CommandList->drawIndexed(cmd.drawArgs);
    }

    // --- Terrain color pass ---
    if (m_TerrainPass.indexCount > 0) {
        if (!m_TerrainPass.pipeline) {
            nvrhi::GraphicsPipelineDesc terrainPso;
            terrainPso.VS = m_TerrainPass.vertexShader;
            terrainPso.PS = m_TerrainPass.pixelShader;
            terrainPso.inputLayout = m_TerrainPass.inputLayout;
            terrainPso.bindingLayouts = { m_TerrainPass.bindingLayout };
            terrainPso.primType = nvrhi::PrimitiveType::TriangleList;
#if PIPELINER_USE_REVERSE_Z
            terrainPso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
#else
            terrainPso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
#endif
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

    m_CommandList->endTimerQuery(m_GpuTimers[m_NextTimerIdx]);
    m_CommandList->close();
    GetDevice()->executeCommandList(m_CommandList);

    int prevIdx = (m_NextTimerIdx + m_QueuedFrames - 1) % m_QueuedFrames;
    if (GetDevice()->pollTimerQuery(m_GpuTimers[prevIdx]))
        m_UI.gpuFrameTimeMs = GetDevice()->getTimerQueryTime(m_GpuTimers[prevIdx]) * 1000.0f;
    m_NextTimerIdx = (m_NextTimerIdx + 1) % m_QueuedFrames;

    m_UI.visibleInstanceCount = (uint32_t)m_VisibleInstanceReferences.size();
    m_UI.culledInstanceCount  = culled;
    m_UI.drawCallCount        = (uint32_t)m_DrawCmds.size();
    m_UI.totalInstanceCount   = m_Scene.regionManager.getTotalInstanceCount();

    m_VisibleInstanceReferences.clear();

    cpuTimer.Stop();
    m_UI.cpuRenderTimeMs = (float)cpuTimer.Milliseconds();
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
#if PIPELINER_USE_INTERLEAVED_VERTEX_ATTRIBUTES
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
#if !PIPELINER_USE_STRUCTURED_BUFFER
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
#if !PIPELINER_USE_STRUCTURED_BUFFER
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
    m_TreePass.instanceBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(std::max<size_t>(1, m_Scene.regionManager.getTotalInstanceCount()) * sizeof(Render::InstanceBufferEntry))
            .setStructStride(sizeof(Render::InstanceBufferEntry))
            .setDebugName("TreeInstanceBuffer")
#if PIPELINER_USE_STRUCTURED_BUFFER
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource)
#else
            .setIsVertexBuffer(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
#endif
    );
    if (!m_TreePass.instanceBuffer) return false;

    // Textures
    engine::TextureCache textureCache(GetDevice(), std::make_shared<vfs::NativeFileSystem>(), nullptr);
    m_TreePass.textureSets.resize(m_Scene.barkTextureSets.size());

    for (size_t i = 0; i < m_Scene.barkTextureSets.size(); i++) {
        std::filesystem::path texDir = g_ProjectDirectory / "media" / m_Scene.barkTextureSets[i] / "textures";
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
#if PIPELINER_USE_STRUCTURED_BUFFER
            nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_TreePass.instanceBuffer,
                nvrhi::Format::UNKNOWN,
                nvrhi::BufferRange(0, m_Scene.regionManager.getTotalInstanceCount() * sizeof(Render::InstanceBufferEntry))),
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
    auto config = SceneLoader::ParseFile(g_SceneConfigDirectory);
    const auto& cam = config["camera"];

    // Leveraging donut >> overloads for the camera properties
    dm::float3 pos(0.f);
    dm::float3 cameraDir(0.f, 0.f, 1.f);
    float moveSpeed = 15.f;

    cam["position"] >> pos;
    cam["direction"] >> cameraDir;
    cam["moveSpeed"] >> moveSpeed;

    m_ViewHandler = std::make_unique<ViewHandler>();
    m_ViewHandler->camera.LookTo(pos, dm::normalize(cameraDir));
    m_ViewHandler->camera.SetMoveSpeed(moveSpeed);
    return !!m_ViewHandler;
}

bool TraditionalRenderPass::_InitTimerQueries() {
    for (uint32_t i = 0; i < m_QueuedFrames; i++)
        m_GpuTimers[i] = GetDevice()->createTimerQuery();
    return true;
}

bool TraditionalRenderPass::_InitShadowPass() {
    // Depth texture
    nvrhi::TextureDesc texDesc;
    texDesc.width            = m_ShadowRes;
    texDesc.height           = m_ShadowRes;
    texDesc.format           = nvrhi::Format::D32;
    texDesc.isRenderTarget   = true;
    texDesc.useClearValue    = true;
    texDesc.clearValue       = nvrhi::Color(1.f);
    texDesc.initialState     = nvrhi::ResourceStates::DepthWrite;
    texDesc.keepInitialState = true;
    texDesc.debugName        = "ShadowMap";
    m_ShadowPass.depthTexture = GetDevice()->createTexture(texDesc);
    if (!m_ShadowPass.depthTexture) return false;

    // Framebuffer (depth only)
    m_ShadowPass.framebuffer = GetDevice()->createFramebuffer(
        nvrhi::FramebufferDesc().setDepthAttachment(m_ShadowPass.depthTexture));
    if (!m_ShadowPass.framebuffer) return false;

    // Shaders
    m_ShadowPass.treeVS = m_ShaderFactory->CreateShader(
        "app/shadow.hlsl", "tree_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_ShadowPass.terrainVS = m_ShaderFactory->CreateShader(
        "app/shadow.hlsl", "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    if (!m_ShadowPass.treeVS || !m_ShadowPass.terrainVS) return false;

    // Terrain shadow input layout
    nvrhi::VertexAttributeDesc terrainShadowAttrs[] = {
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
    m_ShadowPass.terrainInputLayout = GetDevice()->createInputLayout(
        terrainShadowAttrs, uint32_t(std::size(terrainShadowAttrs)), m_ShadowPass.terrainVS);
    if (!m_ShadowPass.terrainInputLayout) return false;

    // Comparison sampler (consumed by tree and terrain color passes for shadow sampling)
    nvrhi::SamplerDesc samplerDesc;
    samplerDesc.minFilter     = true;
    samplerDesc.magFilter     = true;
    samplerDesc.mipFilter     = false;
    samplerDesc.reductionType = nvrhi::SamplerReductionType::Comparison;
    samplerDesc.setAllAddressModes(nvrhi::SamplerAddressMode::Border);
    samplerDesc.borderColor   = nvrhi::Color(1.f);
    m_ShadowPass.comparisonSampler = GetDevice()->createSampler(samplerDesc);
    if (!m_ShadowPass.comparisonSampler) return false;

    // Instance buffer (for shadow casters)
    m_ShadowPass.instanceBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(std::max<size_t>(1, m_Scene.regionManager.getTotalInstanceCount()) * sizeof(Render::InstanceBufferEntry))
            .setStructStride(sizeof(Render::InstanceBufferEntry))
            .setDebugName("ShadowInstanceBuffer")
            .setIsVertexBuffer(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
    );
    if (!m_ShadowPass.instanceBuffer) return false;

    // Binding layout + set (CB only)
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
    if (!m_Scene.terrain) return true; // no terrain in scene, not an error

    const auto& verts   = m_Scene.terrain->getVertices();
    const auto& indices = m_Scene.terrain->getIndices();
    if (verts.empty() || indices.empty()) return true;

    m_TerrainPass.indexCount = static_cast<uint32_t>(indices.size());

    // Shaders
    m_TerrainPass.vertexShader = m_ShaderFactory->CreateShader("app/terrain.hlsl", "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_TerrainPass.pixelShader  = m_ShaderFactory->CreateShader("app/terrain.hlsl", "terrain_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_TerrainPass.vertexShader || !m_TerrainPass.pixelShader) return false;

    // Input layout
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

    // Vertex buffer
    nvrhi::BufferDesc vbDesc;
    vbDesc.isVertexBuffer = true;
    vbDesc.byteSize       = verts.size() * sizeof(Scene::TerrainVertex);
    vbDesc.debugName      = "TerrainVB";
    vbDesc.initialState   = nvrhi::ResourceStates::CopyDest;
    m_TerrainPass.vertexBuffer = GetDevice()->createBuffer(vbDesc);
    initCL->beginTrackingBufferState(m_TerrainPass.vertexBuffer, nvrhi::ResourceStates::CopyDest);
    initCL->writeBuffer(m_TerrainPass.vertexBuffer, verts.data(), vbDesc.byteSize);
    initCL->setPermanentBufferState(m_TerrainPass.vertexBuffer, nvrhi::ResourceStates::VertexBuffer);

    // Index buffer
    nvrhi::BufferDesc ibDesc;
    ibDesc.isIndexBuffer = true;
    ibDesc.byteSize      = indices.size() * sizeof(uint32_t);
    ibDesc.debugName     = "TerrainIB";
    ibDesc.initialState  = nvrhi::ResourceStates::CopyDest;
    m_TerrainPass.indexBuffer = GetDevice()->createBuffer(ibDesc);
    initCL->beginTrackingBufferState(m_TerrainPass.indexBuffer, nvrhi::ResourceStates::CopyDest);
    initCL->writeBuffer(m_TerrainPass.indexBuffer, indices.data(), ibDesc.byteSize);
    initCL->setPermanentBufferState(m_TerrainPass.indexBuffer, nvrhi::ResourceStates::IndexBuffer);

    // Binding layout + set (constant buffer + shadow map)
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
