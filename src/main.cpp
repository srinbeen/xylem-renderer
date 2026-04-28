#include "include/RenderOrchestrator.hpp"
#include "include/Globals.hpp"
#include "include/SceneLoader.hpp"

#include <donut/core/log.h>

#include <fstream>

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

    deviceParams.depthBufferFormat = nvrhi::Format::D32;

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle)) {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }
    log::info("Physical Device: %s", deviceManager->GetRendererString());

    {
        auto* d3d12Device = static_cast<ID3D12Device*>(
            deviceManager->GetDevice()->getNativeObject(nvrhi::ObjectTypes::D3D12_Device));
    }

    {
        UIData uiData;
        SceneRegistry registry;
        if (!SceneLoader::Load(g_SceneConfigDirectory, registry))
            log::error("Error in scene loading");

        auto shaderFactory = createShaderFactory(deviceManager);

        RenderOrchestrator orchestrator(deviceManager, registry, uiData);
        orchestrator.SetShaderFactory(shaderFactory);

        if (orchestrator.Init()) {
            deviceManager->AddRenderPassToBack(&orchestrator);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&orchestrator);
        }
    }

    deviceManager->Shutdown();
    delete deviceManager;
    return 0;
}
