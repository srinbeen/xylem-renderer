#include "include/TraditionalRenderPass.hpp"
#include "include/ComputeCullRenderPass.hpp"
#include "include/Globals.hpp"
#include "include/SceneLoader.hpp"
#include "include/UIRenderer.hpp"

#include <GFSDK_Aftermath.h>
#include <GFSDK_Aftermath_GpuCrashDump.h>
#include <donut/core/log.h>

#include <fstream>

static void onGpuCrashDump(const void* pGpuCrashDump, const uint32_t gpuCrashDumpSize, void*)
{
    std::ofstream dumpFile("xylem_crash.nv-gpudmp", std::ios::binary);
    dumpFile.write(static_cast<const char*>(pGpuCrashDump), gpuCrashDumpSize);
    donut::log::error("Aftermath: GPU crash dump written to xylem_crash.nv-gpudmp");
}

static void onShaderDebugInfo(const void* pShaderDebugInfo, const uint32_t shaderDebugInfoSize, void*)
{
    std::ofstream f("xylem_shader_debug.bin", std::ios::binary);
    f.write(static_cast<const char*>(pShaderDebugInfo), shaderDebugInfoSize);
}

static void onCrashDumpDescription(PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription, void*)
{
    addDescription(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName, "XylemRenderer");
}

static void onResolveMarker(const void* pMarker, const uint32_t markerDataSize, void*, void** ppResolvedMarkerData, uint32_t* pResolvedMarkerDataSize)
{
    *ppResolvedMarkerData     = const_cast<void*>(pMarker);
    *pResolvedMarkerDataSize  = markerDataSize;
}

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
    GFSDK_Aftermath_EnableGpuCrashDumps(
        GFSDK_Aftermath_Version_API,
        GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_DX,
        GFSDK_Aftermath_GpuCrashDumpFeatureFlags_DeferDebugInfoCallbacks,
        onGpuCrashDump,
        onShaderDebugInfo,
        onCrashDumpDescription,
        onResolveMarker,
        nullptr);

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
        auto* d3d12Device = static_cast<ID3D12Device*>(
            deviceManager->GetDevice()->getNativeObject(nvrhi::ObjectTypes::D3D12_Device));
        GFSDK_Aftermath_Result aftermathResult = GFSDK_Aftermath_DX12_Initialize(
            GFSDK_Aftermath_Version_API,
            GFSDK_Aftermath_FeatureFlags_EnableMarkers |
            GFSDK_Aftermath_FeatureFlags_EnableResourceTracking |
            GFSDK_Aftermath_FeatureFlags_GenerateShaderDebugInfo |
            GFSDK_Aftermath_FeatureFlags_EnableShaderErrorReporting,
            d3d12Device);
        if (aftermathResult != GFSDK_Aftermath_Result_Success)
            log::warning("Aftermath initialization failed (0x%x) — markers disabled", aftermathResult);
        else
            log::info("Aftermath initialized successfully");
    }

    {
        UIData uiData;
        SceneRegistry registry;
        if (!SceneLoader::Load(g_SceneConfigDirectory, registry)) 
            log::error("Error in scene loading");

        // TraditionalRenderPass renderPass(deviceManager, registry, uiData);
        ComputeCullRenderPass renderPass(deviceManager, registry, uiData);

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