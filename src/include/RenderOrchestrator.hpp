#ifndef XYLEM_RENDER_ORCHESTRATOR_H
#define XYLEM_RENDER_ORCHESTRATOR_H

#include <memory>

#include <donut/app/ApplicationBase.h>
#include <donut/engine/ShaderFactory.h>

#include "UIData.hpp"
#include "SceneRegistry.hpp"
#include "SharedGPUAssets.hpp"
#include "ViewHandler.hpp"
#include "TraditionalRenderPass.hpp"
#include "ComputeRenderPass.hpp"
#include "MeshShaderRenderPass.hpp"
#include "UIRenderer.hpp"
#include "frame/FrameStages.hpp"

namespace Xylem {

using namespace donut;

// Owns both render passes and the UIRenderer. Handles pipeline switching
// (including camera sync and GPU-idle wait) internally. main.cpp registers
// only this single pass with DeviceManager.
class RenderOrchestrator : public app::IRenderPass {
public:
    RenderOrchestrator(app::DeviceManager* dm, SceneRegistry& registry, UIData& ui)
        : IRenderPass{dm}
        , m_Registry{registry}
        , m_UI{ui}
        , m_Shared{dm->GetDevice(), registry, ui}
        , m_Traditional{dm, registry, ui, m_ViewHandler}
        , m_Compute{dm, registry, ui, m_ViewHandler}
        , m_MeshShader{dm, registry, ui, m_ViewHandler}
        , m_UIPass{dm, &registry, ui, &m_ViewHandler}
    {
        m_Traditional.SetSharedAssets(&m_Shared);
        m_Compute.SetSharedAssets(&m_Shared);
        m_MeshShader.SetSharedAssets(&m_Shared);
    }

    // Must be called before Init().
    void SetShaderFactory(std::shared_ptr<engine::ShaderFactory> sf);

    // Returns false if any pipeline could not be initialized.
    bool Init();

    // --- IRenderPass interface ---
    void Animate(float seconds) override;
    void BackBufferResizing() override;
    void Render(nvrhi::IFramebuffer* framebuffer) override;

    bool KeyboardUpdate(int key, int scancode, int action, int mods) override;
    bool KeyboardCharInput(unsigned int unicode, int mods) override;
    bool MousePosUpdate(double xpos, double ypos) override;
    bool MouseScrollUpdate(double xoffset, double yoffset) override;
    bool MouseButtonUpdate(int button, int action, int mods) override;
    bool JoystickButtonUpdate(int button, bool pressed) override;
    bool JoystickAxisUpdate(int axis, float value) override;

private:
    SceneRegistry&    m_Registry;
    UIData&           m_UI;

    ViewHandler             m_ViewHandler;
    SharedGPUAssets         m_Shared;
    TraditionalRenderPass   m_Traditional;
    ComputeRenderPass       m_Compute;
    MeshShaderRenderPass    m_MeshShader;
    UIRenderer              m_UIPass;

    std::shared_ptr<engine::ShaderFactory> m_ShaderFactory;

    app::IRenderPass* _activePass();
    frame::IFrameStagedPass* _activeStagedPass();

    void _switchPipelineIfNeeded();
    void _loadSceneIfRequested();
};

} // namespace Xylem

#endif // XYLEM_RENDER_ORCHESTRATOR_H
