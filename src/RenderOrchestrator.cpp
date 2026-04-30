#include "include/RenderOrchestrator.hpp"
#include "include/frame/FrameLifecycle.hpp"
#include <donut/core/log.h>
#include <imgui.h>

using namespace Xylem;

void RenderOrchestrator::SetShaderFactory(std::shared_ptr<engine::ShaderFactory> sf)
{
    m_ShaderFactory = sf;
    m_Shared.SetShaderFactory(sf);
    m_Traditional.SetShaderFactory(sf);
    m_Compute.SetShaderFactory(sf);
    m_MeshShader.SetShaderFactory(sf);
}

bool RenderOrchestrator::Init()
{
    const SceneRegistry::CameraInit& ci = m_Registry.getCameraInit();
    m_ViewHandler.camera.LookTo(ci.pos, ci.cameraDir);
    m_ViewHandler.camera.SetMoveSpeed(ci.moveSpeed);

    if (!m_Shared.Init()) {
        donut::log::error("RenderOrchestrator: shared GPU assets failed to initialise");
        return false;
    }

    bool traditionalOk = m_Traditional.Init();
    bool computeOk     = m_Compute.Init();
    bool meshShaderOk  = m_MeshShader.Init();
    if (!traditionalOk || !computeOk || !meshShaderOk) {
        donut::log::error("RenderOrchestrator: one or more pipelines failed to initialise");
        return false;
    }

    m_UI.activePipeline = m_UI.requestedPipeline = Pipeline::Traditional;
    m_UIPass.Init(m_ShaderFactory);
    return true;
}

app::IRenderPass* RenderOrchestrator::_activePass()
{
    switch (m_UI.activePipeline) {
        case Pipeline::Traditional:
            return static_cast<app::IRenderPass*>(&m_Traditional);
        case Pipeline::Compute:
            return static_cast<app::IRenderPass*>(&m_Compute);
        case Pipeline::MeshShader:
            return static_cast<app::IRenderPass*>(&m_MeshShader);
        default:
            return nullptr;
    }
}

frame::IFrameStagedPass* RenderOrchestrator::_activeStagedPass()
{
    switch (m_UI.activePipeline) {
        case Pipeline::Traditional:
            return static_cast<frame::IFrameStagedPass*>(&m_Traditional);
        case Pipeline::Compute:
            return static_cast<frame::IFrameStagedPass*>(&m_Compute);
        case Pipeline::MeshShader:
            return static_cast<frame::IFrameStagedPass*>(&m_MeshShader);
        default:
            return nullptr;
    }
}

void RenderOrchestrator::_switchPipelineIfNeeded()
{
    if (m_UI.requestedPipeline == m_UI.activePipeline) return;

    GetDeviceManager()->GetDevice()->waitForIdle();

    m_UI.activePipeline = m_UI.requestedPipeline;

    // Mark everything dirty so the incoming pass re-syncs its GPU buffers.
    m_Registry.markAllDirty();
}

// --- IRenderPass interface ---

void RenderOrchestrator::Animate(float seconds)
{
    _switchPipelineIfNeeded();
    m_ViewHandler.camera.Animate(seconds);

    // Drive the dirty cycle here so SharedGPUAssets re-bakes the impostor atlas
    // *after* CPU mesh data is regenerated and *before* the active pass rebuilds
    // its impostor binding sets that reference the atlas.
    if (m_Registry.anyDirty()) {
        auto dirtyAssets  = m_Registry.getDirtyAssetIndices();
        auto dirtyRegions = m_Registry.getDirtyRegionIndices();

        m_Registry.rebuildDirtyAssets();
        m_Registry.rebuildDirtyRegions();

        if (!dirtyAssets.empty()) m_Shared.OnAssetsDirty();

        switch (m_UI.activePipeline) {
            case Pipeline::Traditional:
                if (!dirtyAssets.empty())  m_Traditional.onAssetsDirty(dirtyAssets);
                if (!dirtyRegions.empty()) m_Traditional.onRegionsDirty(dirtyRegions);
                break;
            case Pipeline::Compute:
                if (!dirtyAssets.empty())  m_Compute.onAssetsDirty(dirtyAssets);
                if (!dirtyRegions.empty()) m_Compute.onRegionsDirty(dirtyRegions);
                break;
            case Pipeline::MeshShader:
                if (!dirtyAssets.empty())  m_MeshShader.onAssetsDirty(dirtyAssets);
                if (!dirtyRegions.empty()) m_MeshShader.onRegionsDirty(dirtyRegions);
                break;
            default: break;
        }

        m_Registry.clearDirtyFlags();
    }

    _activePass()->Animate(seconds);
    m_UIPass.Animate(seconds);
}

void RenderOrchestrator::BackBufferResizing()
{
    m_Traditional.BackBufferResizing();
    m_Compute.BackBufferResizing();
    m_MeshShader.BackBufferResizing();
    m_UIPass.BackBufferResizing();
}

void RenderOrchestrator::Render(nvrhi::IFramebuffer* framebuffer)
{
    // Stage-aware orchestration hook. The active pass owns stage execution;
    // the orchestrator provides a single stage-order contract point.
    if (auto* staged = _activeStagedPass()) {
        const auto& stages = staged->GetFrameStageOrder();
        (void)stages;
    }

    _activePass()->Render(framebuffer);
    m_UIPass.Render(framebuffer);
}

bool RenderOrchestrator::KeyboardUpdate(int key, int scancode, int action, int mods)
{
    if (m_UIPass.KeyboardUpdate(key, scancode, action, mods)) return true;
    if (ImGui::GetIO().WantCaptureKeyboard) return false;
    m_ViewHandler.camera.KeyboardUpdate(key, scancode, action, mods);
    return true;
}

bool RenderOrchestrator::KeyboardCharInput(unsigned int unicode, int mods)
{
    if (m_UIPass.KeyboardCharInput(unicode, mods)) return true;
    return _activePass()->KeyboardCharInput(unicode, mods);
}

bool RenderOrchestrator::MousePosUpdate(double xpos, double ypos)
{
    m_UIPass.MousePosUpdate(xpos, ypos);
    if (!ImGui::GetIO().WantCaptureMouse)
        m_ViewHandler.camera.MousePosUpdate(xpos, ypos);
    return true;
}

bool RenderOrchestrator::MouseScrollUpdate(double xoffset, double yoffset)
{
    if (m_UIPass.MouseScrollUpdate(xoffset, yoffset)) return true;
    if (!ImGui::GetIO().WantCaptureMouse)
        m_ViewHandler.camera.MouseScrollUpdate(xoffset, yoffset);
    return true;
}

bool RenderOrchestrator::MouseButtonUpdate(int button, int action, int mods)
{
    if (m_UIPass.MouseButtonUpdate(button, action, mods)) return true;
    if (!ImGui::GetIO().WantCaptureMouse)
        m_ViewHandler.camera.MouseButtonUpdate(button, action, mods);
    return true;
}

bool RenderOrchestrator::JoystickButtonUpdate(int button, bool pressed)
{
    m_ViewHandler.camera.JoystickButtonUpdate(button, pressed);
    return true;
}

bool RenderOrchestrator::JoystickAxisUpdate(int axis, float value)
{
    m_ViewHandler.camera.JoystickUpdate(axis, value);
    return true;
}
