#include "include/TraditionalRenderPass.hpp"
#include "include/Globals.hpp"
#include "include/SceneLoader.hpp"
#include "include/UIRenderer.hpp"

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
        SceneRegistry registry;
        if (!SceneLoader::Load(g_SceneConfigDirectory, registry)) 
            log::error("Error in scene loading");

        TraditionalRenderPass renderPass(deviceManager, registry, uiData);

        auto shaderFactory = createShaderFactory(deviceManager);
        renderPass.SetShaderFactory(shaderFactory);

        if (renderPass.Init()) {
            UIRenderer uiPass(deviceManager, &registry, uiData);
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