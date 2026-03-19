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
    if (!_InitShaders())                     return false;
    if (!_InitVertexAttributes())            return false;
    if (!_InitBuffers())                     return false;
    if (!_InitTextureAndSampler(initCL, commonPasses)) return false;
    if (!_InitBindingLayoutAndSet())         return false;
    if (!_InitTerrain(initCL))              return false;
    if (!_InitViewHandler())                 return false;

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
        m_CommandList->writeBuffer(m_Resources.instanceBuffer, m_VisibleInstanceBuffer.data(),
            m_VisibleInstanceReferences.size() * sizeof(Render::InstanceBufferEntry));

    nvrhi::GraphicsState state;
    state.pipeline   = m_Resources.pipeline;
    state.framebuffer = framebuffer;
    state.viewport   = m_ViewHandler->view.GetViewportState();

    for (const auto& cmd : m_DrawCmds) {
        state.bindings = { m_Resources.bindingSets[cmd.textureSetIdx] };
        state.vertexBuffers = {
#if PIPELINER_USE_INTERLEAVED_VERTEX_ATTRIBUTES
            { cmd.vertexBuffer, 0, 0 },
#if !PIPELINER_USE_STRUCTURED_BUFFER
            { m_Resources.instanceBuffer, 1, 0 },
#endif
#else
            { cmd.vertexBuffer, 0, offsetof(ProcGen::TreeVertex, pos) },
            { cmd.vertexBuffer, 1, offsetof(ProcGen::TreeVertex, normal) },
            { cmd.vertexBuffer, 2, offsetof(ProcGen::TreeVertex, tangent) },
            { cmd.vertexBuffer, 3, offsetof(ProcGen::TreeVertex, bitangent) },
            { cmd.vertexBuffer, 4, offsetof(ProcGen::TreeVertex, uv) },
#if !PIPELINER_USE_STRUCTURED_BUFFER
            { m_Resources.instanceBuffer, 5, offsetof(Render::InstanceBufferEntry, model) },
            { m_Resources.instanceBuffer, 6, offsetof(Render::InstanceBufferEntry, normal) },
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

    // --- Terrain draw ---
    if (m_Resources.terrainIndexCount > 0) {
        if (!m_Resources.terrainPipeline) {
            nvrhi::GraphicsPipelineDesc terrainPso;
            terrainPso.VS = m_Resources.terrainVS;
            terrainPso.PS = m_Resources.terrainPS;
            terrainPso.inputLayout = m_Resources.terrainInputLayout;
            terrainPso.bindingLayouts = { m_Resources.terrainBindingLayout };
            terrainPso.primType = nvrhi::PrimitiveType::TriangleList;
#if PIPELINER_USE_REVERSE_Z
            terrainPso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
#else
            terrainPso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
#endif
            m_Resources.terrainPipeline = GetDevice()->createGraphicsPipeline(terrainPso, fbinfo);
        }

        nvrhi::GraphicsState terrainState;
        terrainState.pipeline   = m_Resources.terrainPipeline;
        terrainState.framebuffer = framebuffer;
        terrainState.viewport   = m_ViewHandler->view.GetViewportState();
        terrainState.bindings   = { m_Resources.terrainBindingSet };
        terrainState.vertexBuffers = { { m_Resources.terrainVertexBuffer, 0, 0 } };
        terrainState.indexBuffer   = { m_Resources.terrainIndexBuffer, nvrhi::Format::R32_UINT, 0 };
        m_CommandList->setGraphicsState(terrainState);
        m_CommandList->drawIndexed(
            nvrhi::DrawArguments()
                .setVertexCount(m_Resources.terrainIndexCount));
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

bool TraditionalRenderPass::_InitTextureAndSampler(nvrhi::ICommandList* initCL, engine::CommonRenderPasses& commonPasses) {
    engine::TextureCache textureCache(GetDevice(), std::make_shared<vfs::NativeFileSystem>(), nullptr);

    m_Resources.textureSets.resize(m_Scene.barkTextureSets.size());

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

        m_Resources.textureSets[i].diffuse   = diffLoaded->texture;
        m_Resources.textureSets[i].normalMap = normLoaded->texture;

        if (!m_Resources.textureSets[i].diffuse || !m_Resources.textureSets[i].normalMap)
            return false;
    }

    m_Resources.sampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Wrap)
            .setAllFilters(true)
            .setMaxAnisotropy(8.f)
    );

    return !!m_Resources.sampler;
}

bool TraditionalRenderPass::_InitBindingLayoutAndSet() {
    m_Resources.bindingSets.resize(m_Resources.textureSets.size());

    for (size_t i = 0; i < m_Resources.textureSets.size(); i++) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_Resources.constantBuffer, nvrhi::BufferRange(0, Render::c_ConstantBufferSize)),
            nvrhi::BindingSetItem::Sampler(0, m_Resources.sampler),
            nvrhi::BindingSetItem::Texture_SRV(0, m_Resources.textureSets[i].diffuse),
            nvrhi::BindingSetItem::Texture_SRV(1, m_Resources.textureSets[i].normalMap),
#if PIPELINER_USE_STRUCTURED_BUFFER
            nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_Resources.instanceBuffer,
                nvrhi::Format::UNKNOWN,
                nvrhi::BufferRange(0, m_Scene.regionManager.getTotalInstanceCount() * sizeof(Render::InstanceBufferEntry))),
            nvrhi::BindingSetItem::PushConstants(1, sizeof(uint32_t)),
#endif
        };

        if (i == 0) {
            if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
                    bsd, m_Resources.bindingLayout, m_Resources.bindingSets[0]))
                return false;
        } else {
            m_Resources.bindingSets[i] = GetDevice()->createBindingSet(bsd, m_Resources.bindingLayout);
            if (!m_Resources.bindingSets[i]) return false;
        }
    }

    return !!m_Resources.bindingLayout;
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

bool TraditionalRenderPass::_InitTerrain(nvrhi::ICommandList* initCL) {
    if (!m_Scene.terrain) return true; // no terrain in scene, not an error

    const auto& verts   = m_Scene.terrain->getVertices();
    const auto& indices = m_Scene.terrain->getIndices();
    if (verts.empty() || indices.empty()) return true;

    m_Resources.terrainIndexCount = static_cast<uint32_t>(indices.size());

    // Shaders
    m_Resources.terrainVS = m_ShaderFactory->CreateShader("app/terrain.hlsl", "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_Resources.terrainPS = m_ShaderFactory->CreateShader("app/terrain.hlsl", "terrain_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_Resources.terrainVS || !m_Resources.terrainPS) return false;

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
    m_Resources.terrainInputLayout = GetDevice()->createInputLayout(
        terrainAttrs, uint32_t(std::size(terrainAttrs)), m_Resources.terrainVS);
    if (!m_Resources.terrainInputLayout) return false;

    // Vertex buffer
    nvrhi::BufferDesc vbDesc;
    vbDesc.isVertexBuffer = true;
    vbDesc.byteSize       = verts.size() * sizeof(Scene::TerrainVertex);
    vbDesc.debugName      = "TerrainVB";
    vbDesc.initialState   = nvrhi::ResourceStates::CopyDest;
    m_Resources.terrainVertexBuffer = GetDevice()->createBuffer(vbDesc);
    initCL->beginTrackingBufferState(m_Resources.terrainVertexBuffer, nvrhi::ResourceStates::CopyDest);
    initCL->writeBuffer(m_Resources.terrainVertexBuffer, verts.data(), vbDesc.byteSize);
    initCL->setPermanentBufferState(m_Resources.terrainVertexBuffer, nvrhi::ResourceStates::VertexBuffer);

    // Index buffer
    nvrhi::BufferDesc ibDesc;
    ibDesc.isIndexBuffer = true;
    ibDesc.byteSize      = indices.size() * sizeof(uint32_t);
    ibDesc.debugName     = "TerrainIB";
    ibDesc.initialState  = nvrhi::ResourceStates::CopyDest;
    m_Resources.terrainIndexBuffer = GetDevice()->createBuffer(ibDesc);
    initCL->beginTrackingBufferState(m_Resources.terrainIndexBuffer, nvrhi::ResourceStates::CopyDest);
    initCL->writeBuffer(m_Resources.terrainIndexBuffer, indices.data(), ibDesc.byteSize);
    initCL->setPermanentBufferState(m_Resources.terrainIndexBuffer, nvrhi::ResourceStates::IndexBuffer);

    // Binding layout + set (only needs constant buffer)
    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_Resources.constantBuffer,
            nvrhi::BufferRange(0, Render::c_ConstantBufferSize)),
    };
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
            bsd, m_Resources.terrainBindingLayout, m_Resources.terrainBindingSet))
        return false;

    return true;
}
