#include "include/TraditionalRenderPass.hpp"
#include <donut/app/imgui_renderer.h>

using namespace Xylem;

std::shared_ptr<engine::ShaderFactory> createShaderFactory(app::DeviceManager* deviceManager) {
    std::filesystem::path fwShaderPath =
        g_BinDirectory / "shaders/framework" /
        app::GetShaderTypeName(deviceManager->GetDevice()->getGraphicsAPI());
    std::filesystem::path appShaderPath =
        g_BinDirectory / "shaders/custom" /
        app::GetShaderTypeName(deviceManager->GetDevice()->getGraphicsAPI());
    
    auto rootFS = std::make_shared<vfs::RootFileSystem>();
    rootFS->mount("/shaders/donut", fwShaderPath);
    rootFS->mount("/shaders/app",   appShaderPath);
    
    return std::make_shared<engine::ShaderFactory>(
        deviceManager->GetDevice(), rootFS, "/shaders"
    );
}

// ===========================================================================
// XylemUIRenderer (Static Viewer)
// ===========================================================================
class XylemUIRenderer : public app::ImGui_Renderer {
private:
    TraditionalRenderPass* m_pass;
    UIData&                m_ui;

public:
    XylemUIRenderer(app::DeviceManager* dm, TraditionalRenderPass* pass, UIData& ui)
        : ImGui_Renderer(dm), m_pass(pass), m_ui(ui)
    {
        ImGui::GetIO().IniFilename = nullptr;
    }

    void Init(std::shared_ptr<engine::ShaderFactory> sf) { ImGui_Renderer::Init(sf); }

    bool KeyboardUpdate(int key, int scancode, int action, int mods) override {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            m_ui.ShowUI = !m_ui.ShowUI;
            return true;
        }
        return ImGui_Renderer::KeyboardUpdate(key, scancode, action, mods);
    }

protected:
    void buildUI() override {
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
            ImGui::Text("Draw calls: %u", m_ui.drawCallCount);
        }

        ImGui::Separator();

        // =====================================================================
        // SCENE SUMMARY
        // =====================================================================
        if (ImGui::CollapsingHeader("Scene Summary", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Loaded L-Systems: %zu", m_pass->GetLSystems().size());
            ImGui::Text("Loaded Tree Assets: %zu", m_pass->GetTreeAssets().size());
            ImGui::Text("Loaded Regions: %zu", m_pass->GetRegions().size());
        }

        ImGui::Spacing();
        if (ImGui::Button("Hide UI  [ESC]")) m_ui.ShowUI = false;
        
        ImGui::End();
    }
};

// ===========================================================================
// Entry point
// ===========================================================================
#ifdef WIN32
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
#else
int main(int __argc, const char** __argv)
#endif
{
    nvrhi::GraphicsAPI api = app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    app::DeviceManager* deviceManager = app::DeviceManager::Create(api);

    app::DeviceCreationParameters deviceParams;
#ifdef _DEBUG
    deviceParams.enableDebugRuntime         = true;
    deviceParams.enableNvrhiValidationLayer = true;
#endif
    deviceParams.depthBufferFormat = nvrhi::Format::D16;

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle)) {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }
    log::info("Physical Device: %s", deviceManager->GetRendererString());

    {
        UIData uiData;
        TraditionalRenderPass renderPass(deviceManager, uiData);

        auto shaderFactory = createShaderFactory(deviceManager);
        renderPass.SetShaderFactory(shaderFactory);

        if (renderPass.Init()) {
            XylemUIRenderer uiPass(deviceManager, &renderPass, uiData);
            uiPass.Init(shaderFactory);

            deviceManager->AddRenderPassToBack(&renderPass);
            deviceManager->AddRenderPassToBack(&uiPass);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&uiPass);
            deviceManager->RemoveRenderPass(&renderPass);
        }
    }

    deviceManager->Shutdown();
    delete deviceManager;
    return 0;
}