#ifndef XYLEM_UI_RENDERER_H
#define XYLEM_UI_RENDERER_H

#include <donut/app/imgui_renderer.h>

#include "UIData.hpp"

using namespace donut;

namespace Xylem {

class TraditionalRenderPass;

class UIRenderer : public app::ImGui_Renderer {
private:
    TraditionalRenderPass* m_pass;
    UIData&                m_ui;

public:
    UIRenderer(app::DeviceManager* dm, TraditionalRenderPass* pass, UIData& ui)
        : ImGui_Renderer(dm), m_pass(pass), m_ui(ui)
    {
        ImGui::GetIO().IniFilename = nullptr;
    }

    void Init(std::shared_ptr<engine::ShaderFactory> sf) { ImGui_Renderer::Init(sf); }

    bool KeyboardUpdate(int key, int scancode, int action, int mods);

protected:
    void buildUI();
};

} // namespace Xylem

#endif // XYLEM_UI_RENDER_PASS_H
