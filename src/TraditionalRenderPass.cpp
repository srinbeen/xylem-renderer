#include "include/TraditionalRenderPass.hpp"
#include "include/macros.h"

#include <nvrhi/utils.h>
#include <donut/app/imgui_renderer.h>
#include <donut/app/Timer.h>
#include <donut/engine/TextureCache.h>
#include <donut/core/math/math.h>
#include <donut/core/json.h>

using namespace Xylem;

// public
bool TraditionalRenderPass::Init() {
    m_CommandList = GetDevice()->createCommandList();
    m_CommandList->open();

    if (!SceneLoader::Load(g_SceneConfigDirectory, GetDevice(), m_CommandList, m_Scene)) return false;
    if (!_InitShaders())                     return false;
    if (!_InitVertexAttributes())            return false;
    if (!_InitBuffers())                     return false;
    if (!_InitTextureAndSampler())           return false;
    if (!_InitBindingLayoutAndSet())         return false;
    if (!_InitViewHandler())                 return false;

    m_CommandList->close();
    GetDevice()->executeCommandList(m_CommandList);

    _InitTimerQueries();

    m_UI.totalInstanceCount = m_Scene.regionManager.getTotalInstanceCount();

    m_VisibleInstanceReferences.reserve(m_Scene.regionManager.getTotalInstanceCount());

    // Initialize dynamic LOD counters
    m_InstanceCounts.resize(m_Scene.assets.size(), std::vector<uint32_t>(m_Scene.lodSegments.size(), 0));
    m_InstanceOffsets.assign(m_Scene.assets.size(), std::vector<uint32_t>(m_Scene.lodSegments.size(), 0));

    m_VisibleInstanceBuffer.resize(m_Scene.regionManager.getTotalInstanceCount());
    m_DrawCmds.reserve(m_Scene.assets.size() * m_Scene.lodSegments.size());

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

    if (!m_Resources.pipeline) {
        nvrhi::GraphicsPipelineDesc psoDesc;
        psoDesc.VS           = m_Resources.vertexShader;
        psoDesc.PS           = m_Resources.pixelShader;
        psoDesc.inputLayout  = m_Resources.inputLayout;
        psoDesc.bindingLayouts = { m_Resources.bindingLayout };
        psoDesc.primType     = nvrhi::PrimitiveType::TriangleList;
#if PIPELINER_USE_REVERSE_Z
        psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
#else
        psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
#endif
        m_Resources.pipeline = GetDevice()->createGraphicsPipeline(psoDesc, fbinfo);

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

    Render::ConstantBufferEntry constants{};
    constants.view       = dm::affineToHomogeneous(m_ViewHandler->view.GetViewMatrix());
    constants.projection = m_ViewHandler->view.GetProjectionMatrix();
    m_CommandList->writeBuffer(m_Resources.constantBuffer, &constants, Render::c_ConstantBufferSize);

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
                    .setStartInstanceLocation(instanceOffset) });
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
        m_CommandList->writeBuffer(m_Resources.instanceBuffer, m_VisibleInstanceBuffer.data(),
            m_VisibleInstanceReferences.size() * sizeof(Render::InstanceBufferEntry));

    nvrhi::GraphicsState state;
    state.pipeline   = m_Resources.pipeline;
    state.framebuffer = framebuffer;
    state.viewport   = m_ViewHandler->view.GetViewportState();
    state.bindings   = { m_Resources.bindingSet };

    for (const auto& cmd : m_DrawCmds) {
        state.vertexBuffers = {
#if PIPELINER_USE_INTERLEAVED_VERTEX_ATTRIBUTES
            { cmd.vertexBuffer, 0, 0 },
#if !PIPELINER_USE_STRUCTURED_BUFFER
            { m_Resources.instanceBuffer, 1, 0 },
#endif
#else
            { cmd.vertexBuffer, 0, offsetof(ProcGen::TreeVertex, pos) },
            { cmd.vertexBuffer, 1, offsetof(ProcGen::TreeVertex, normal) },
            { cmd.vertexBuffer, 2, offsetof(ProcGen::TreeVertex, uv) },
#if !PIPELINER_USE_STRUCTURED_BUFFER
            { m_Resources.instanceBuffer, 3, offsetof(Render::InstanceBufferEntry, model) },
            { m_Resources.instanceBuffer, 4, offsetof(Render::InstanceBufferEntry, normal) },
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

bool TraditionalRenderPass::_InitShaders() {
    m_Resources.vertexShader = m_ShaderFactory->CreateShader("app/shaders.hlsl", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_Resources.pixelShader  = m_ShaderFactory->CreateShader("app/shaders.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
    return !!m_Resources.vertexShader && !!m_Resources.pixelShader;
}

bool TraditionalRenderPass::_InitVertexAttributes() {
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
            .setName("UV")
            .setFormat(nvrhi::Format::RG32_FLOAT)
            .setOffset(0)
            .setBufferIndex(2)
            .setElementStride(sizeof(ProcGen::TreeVertex)),
#if !PIPELINER_USE_STRUCTURED_BUFFER
        nvrhi::VertexAttributeDesc()
            .setName("MODEL_MATRIX")
            .setFormat(nvrhi::Format::RGBA32_FLOAT)
            .setArraySize(4)
            .setOffset(0).
            setBufferIndex(3)
            .setElementStride(sizeof(Render::InstanceBufferEntry))
            .setIsInstanced(true),
        nvrhi::VertexAttributeDesc()
            .setName("NORMAL_MATRIX")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setArraySize(3)
            .setOffset(0).
            setBufferIndex(4)
            .setElementStride(sizeof(Render::InstanceBufferEntry))
            .setIsInstanced(true),
#endif
#endif
    };
    m_Resources.inputLayout = GetDevice()->createInputLayout(attributes, uint32_t(std::size(attributes)), m_Resources.vertexShader);
    return !!m_Resources.inputLayout;
}

bool TraditionalRenderPass::_InitBuffers() {
    m_Resources.instanceBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(std::max<size_t>(1, m_Scene.regionManager.getTotalInstanceCount()) * sizeof(Render::InstanceBufferEntry))
            .setStructStride(sizeof(Render::InstanceBufferEntry))
            .setDebugName("InstanceBuffer")
#if PIPELINER_USE_STRUCTURED_BUFFER
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource)
#else
            .setIsVertexBuffer(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
#endif
    );

    m_Resources.constantBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(Render::c_ConstantBufferSize)
            .setIsConstantBuffer(true)
            .setDebugName("ConstantBuffer")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
    );

    return !!m_Resources.instanceBuffer && !!m_Resources.constantBuffer;
}

bool TraditionalRenderPass::_InitTextureAndSampler() {
    engine::TextureCache textureCache(GetDevice(), std::make_shared<vfs::NativeFileSystem>(), nullptr);
    std::filesystem::path tex =
        app::GetDirectoryWithExecutable().parent_path().parent_path() / "media/bark_willow_02_diff_1k.jpg";
    auto loaded = textureCache.LoadTextureFromFile(tex, true, nullptr, m_CommandList);
    m_Resources.texture = loaded->texture;
    m_Resources.sampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
        .setAllAddressModes(nvrhi::SamplerAddressMode::Wrap)
    );
    return !!m_Resources.texture && !!m_Resources.sampler;
}

bool TraditionalRenderPass::_InitBindingLayoutAndSet() {
    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_Resources.constantBuffer, nvrhi::BufferRange(0, Render::c_ConstantBufferSize)),
        nvrhi::BindingSetItem::Sampler(0, m_Resources.sampler),
        nvrhi::BindingSetItem::Texture_SRV(0, m_Resources.texture),
#if PIPELINER_USE_STRUCTURED_BUFFER
        nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_Resources.instanceBuffer,
            nvrhi::Format::UNKNOWN,
            nvrhi::BufferRange(0, m_Scene.regionManager.getTotalInstanceCount() * sizeof(Render::InstanceBufferEntry))),
        nvrhi::BindingSetItem::PushConstants(1, sizeof(uint32_t)),
#endif
    };
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
            bsd, m_Resources.bindingLayout, m_Resources.bindingSet)) {
        return false;
    }
    return !!m_Resources.bindingLayout && !!m_Resources.bindingSet;
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