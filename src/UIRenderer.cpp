#include "include/UIRenderer.hpp"

using namespace Xylem;

bool UIRenderer::KeyboardUpdate(int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        m_ui.ShowUI = !m_ui.ShowUI;
        return true;
    }
    return ImGui_Renderer::KeyboardUpdate(key, scancode, action, mods);
}

void UIRenderer::buildUI() {
    if (!m_ui.ShowUI) return;

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(400, 250), ImGuiCond_FirstUseEver);
    ImGui::Begin("Xylem (Viewer Mode)", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    // =====================================================================
    // PERFORMANCE METRICS
    // =====================================================================
    if (ImGui::CollapsingHeader("Performance Metrics", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Renderer: %s", GetDeviceManager()->GetRendererString());

        double ft = GetDeviceManager()->GetAverageFrameTimeSeconds();
        if (ft > 0.0)
            ImGui::Text("CPU frame:  %.2f ms  (%.0f FPS)", ft * 1e3, 1.0 / ft);

        ImGui::Text("CPU render: %.2f ms", m_ui.cpuRenderTimeMs);

        if (m_ui.gpuFrameTimeMs >= 0.f)
            ImGui::Text("GPU pass:   %.2f ms", m_ui.gpuFrameTimeMs);
        else
            ImGui::TextDisabled("GPU pass:   (pending)");

        ImGui::Separator();
        ImGui::Text("Instances  visible: %u / %u  (culled: %u)",
            m_ui.visibleInstanceCount, m_ui.totalInstanceCount, m_ui.culledInstanceCount);
        ImGui::Text("Shadow     visible: %u / %u  (culled: %u)",
            m_ui.shadowVisibleCount, m_ui.totalInstanceCount, m_ui.shadowCulledCount);
        ImGui::Text("Draw calls: %u", m_ui.drawCallCount);
    }

    ImGui::Separator();

    // =====================================================================
    // SCENE SUMMARY
    // =====================================================================
    if (ImGui::CollapsingHeader("Scene Summary", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Loaded L-Systems: %zu", m_Registry->getLSystems().size());
        ImGui::Text("Loaded Tree Assets: %zu", m_Registry->getAssets().size());
        ImGui::Text("Loaded Regions: %zu", m_Registry->getRegions().size());
    }

    ImGui::Spacing();
    if (ImGui::Button("Hide UI  [ESC]")) m_ui.ShowUI = false;

    ImGui::End();
}
