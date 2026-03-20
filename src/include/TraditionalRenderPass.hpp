#ifndef XYLEM_TRADITIONAL_RENDER_PASS_H
#define XYLEM_TRADITIONAL_RENDER_PASS_H

#include <donut/app/ApplicationBase.h>

#include <donut/app/Camera.h>
#include <donut/engine/View.h>

#include <nvrhi/nvrhi.h>

#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>

#include "SceneLoader.hpp"
#include "Render.hpp"
#include "UIData.hpp"

namespace Xylem {

static const char* g_WindowTitle = "Xylem";
static const std::filesystem::path g_BinDirectory = donut::app::GetDirectoryWithExecutable().parent_path();
static const std::filesystem::path g_ProjectDirectory = g_BinDirectory.parent_path();
static const std::filesystem::path g_SceneConfigDirectory = g_ProjectDirectory / "scene/new_scene.json";

using namespace donut;

class TraditionalRenderPass : public app::IRenderPass {
public:
    static constexpr uint32_t m_GlobalSeed   = 0xDEADBEEF;
    static constexpr uint32_t m_QueuedFrames = 4;
    static constexpr uint32_t m_ShadowRes = 2048;

    TraditionalRenderPass(app::DeviceManager* dm, UIData& ui)
        : IRenderPass{dm}, m_UI{ui} {}

    void SetShaderFactory(std::shared_ptr<engine::ShaderFactory> sf) { m_ShaderFactory = std::move(sf); }

    bool Init();
    void Animate(float seconds) override;
    void BackBufferResizing() override { m_TreePass.pipeline = nullptr; m_TerrainPass.pipeline = nullptr; m_ShadowPass.treePipeline = nullptr; m_ShadowPass.terrainPipeline = nullptr; }
    void Render(nvrhi::IFramebuffer* framebuffer) override;

    // Input overrides
    bool KeyboardUpdate(int key, int scancode, int action, int mods) override;
    bool MousePosUpdate(double xpos, double ypos) override;
    bool MouseScrollUpdate(double xoffset, double yoffset) override;
    bool MouseButtonUpdate(int button, int action, int mods) override;
    bool JoystickButtonUpdate(int button, bool pressed) override;
    bool JoystickAxisUpdate(int axis, float value) override;

    const std::map<std::string, std::unique_ptr<ProcGen::LSystem>>& GetLSystems() const { return m_Scene.lsystems; }
    const std::vector<Scene::TreeAsset>& GetTreeAssets() const { return m_Scene.assets; }
    const Scene::RegionManager& GetRegions() const { return m_Scene.regionManager; }

private:
    struct ViewHandler {
        app::FirstPersonCamera camera;
        engine::PlanarView     view;

        uint32_t distToLOD(float value, const std::vector<float>& arr) {
            auto it = std::lower_bound(arr.begin(), arr.end(), value);
            if (it == arr.end()) return static_cast<uint32_t>(arr.size() - 1);
            return static_cast<uint32_t>(std::distance(arr.begin(), it));
        }
    };

    struct TextureSet {
        nvrhi::TextureHandle      diffuse;
        nvrhi::TextureHandle      normalMap;
    };

    // Shared across all passes
    struct SharedResources {
        nvrhi::BufferHandle       constantBuffer;
    };

    // Main color pass — tree geometry
    struct TreePassResources {
        nvrhi::ShaderHandle                    vertexShader;
        nvrhi::ShaderHandle                    pixelShader;
        nvrhi::InputLayoutHandle               inputLayout;
        nvrhi::BufferHandle                    instanceBuffer;
        std::vector<TextureSet>                textureSets;
        nvrhi::SamplerHandle                   sampler;
        nvrhi::BindingLayoutHandle             bindingLayout;
        std::vector<nvrhi::BindingSetHandle>   bindingSets;
        nvrhi::GraphicsPipelineHandle          pipeline;
    };

    // Shadow depth pass
    struct ShadowPassResources {
        nvrhi::TextureHandle                   depthTexture;
        nvrhi::FramebufferHandle               framebuffer;
        nvrhi::ShaderHandle                    treeVS;
        nvrhi::ShaderHandle                    terrainVS;
        nvrhi::InputLayoutHandle               terrainInputLayout;
        nvrhi::SamplerHandle                   comparisonSampler;
        nvrhi::BufferHandle                    instanceBuffer;
        nvrhi::BindingLayoutHandle             bindingLayout;
        nvrhi::BindingSetHandle                bindingSet;
        nvrhi::GraphicsPipelineHandle          treePipeline;
        nvrhi::GraphicsPipelineHandle          terrainPipeline;
    };

    // Terrain color pass
    struct TerrainPassResources {
        nvrhi::ShaderHandle                    vertexShader;
        nvrhi::ShaderHandle                    pixelShader;
        nvrhi::InputLayoutHandle               inputLayout;
        nvrhi::BufferHandle                    vertexBuffer;
        nvrhi::BufferHandle                    indexBuffer;
        uint32_t                               indexCount = 0;
        nvrhi::BindingLayoutHandle             bindingLayout;
        nvrhi::BindingSetHandle                bindingSet;
        nvrhi::GraphicsPipelineHandle          pipeline;
    };

    SharedResources                                    m_Shared;
    TreePassResources                                  m_TreePass;
    ShadowPassResources                                m_ShadowPass;
    TerrainPassResources                               m_TerrainPass;

    nvrhi::CommandListHandle                           m_CommandList;
    std::unique_ptr<ViewHandler>                       m_ViewHandler;
    std::shared_ptr<engine::ShaderFactory>             m_ShaderFactory;

    nvrhi::TimerQueryHandle                            m_GpuTimers[m_QueuedFrames];
    int                                                m_NextTimerIdx = 0;

    UIData&                                            m_UI;

    SceneData                                          m_Scene;

    // populated during render loop
    std::vector<Render::InstanceReference>             m_VisibleInstanceReferences;
    std::vector<std::vector<uint32_t>>                 m_InstanceCounts;
    std::vector<std::vector<uint32_t>>                 m_InstanceOffsets;
    std::vector<Render::DrawCmd>                       m_DrawCmds;
    std::vector<Render::InstanceBufferEntry>           m_VisibleInstanceBuffer;

    // Shadow pass draw data (extended frustum culled, lowest LOD)
    std::vector<Render::DrawCmd>                       m_ShadowDrawCmds;
    std::vector<Render::InstanceBufferEntry>           m_ShadowInstanceBuffer;

    bool _InitShared();
    bool _InitTreePass(nvrhi::ICommandList* initCL, engine::CommonRenderPasses& commonPasses);
    bool _InitShadowPass();
    bool _InitTerrainPass(nvrhi::ICommandList* initCL);
    bool _InitViewHandler();
    bool _InitTimerQueries();
};

} // namespace Xylem

#endif // XYLEM_TRADITIONAL_RENDER_PASS_H
