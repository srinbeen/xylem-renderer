#ifndef XYLEM_TRADITIONAL_RENDER_PASS_H
#define XYLEM_TRADITIONAL_RENDER_PASS_H

#include <donut/app/ApplicationBase.h>

#include <donut/app/Camera.h>
#include <donut/engine/View.h>

#include <nvrhi/nvrhi.h>

#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>

#include <unordered_map>
#include <array>

#include "SceneRegistry.hpp"
#include "SharedGPUAssets.hpp"
#include "Render.hpp"
#include "UIData.hpp"
#include "ViewHandler.hpp"
#include "frame/FrameContracts.hpp"
#include "frame/FrameStages.hpp"

namespace Xylem {

using namespace donut;

class TraditionalRenderPass : public app::IRenderPass, public frame::IFrameStagedPass {
public:
    static constexpr uint32_t m_QueuedFrames = 4;
    static constexpr uint32_t m_ShadowRes = 2048;

    TraditionalRenderPass(app::DeviceManager* dm, SceneRegistry& registry, UIData& ui, ViewHandler& vh)
        : IRenderPass{dm}, m_Registry{registry}, m_UI{ui}, m_ViewHandler{vh} {}

    void SetShaderFactory(std::shared_ptr<engine::ShaderFactory> sf) { m_ShaderFactory = std::move(sf); }
    void SetSharedAssets(SharedGPUAssets* shared) { m_Shared = shared; }

    bool Init();
    bool LoadResources() override;
    void Animate(float seconds) override;
    void BackBufferResizing() override {
        m_StageResources.sceneTreeStage.pipeline = nullptr;
        m_StageResources.sceneTerrainStage.pipeline = nullptr;
        m_StageResources.shadowStage.treePipeline = nullptr;
        m_StageResources.shadowStage.terrainPipeline = nullptr;
        m_StageResources.skyStage.pipeline = nullptr;
    }
    void Render(nvrhi::IFramebuffer* framebuffer) override;
    const frame::FrameStageOrder& GetFrameStageOrder() const override { return frame::kDefaultFrameStageOrder; }

    // Hot-reload callbacks
    void onAssetsDirty(const std::vector<size_t>& dirtyAssetIndices);
    void onRegionsDirty(const std::vector<size_t>& dirtyRegionIndices);

private:
    // GPU-side per-asset data (VB/IB per LOD).
    struct GPUTreeAsset {
        std::vector<Scene::TreeLODData> lods;
        uint32_t                        textureSetIdx;
    };

    // Shared across all passes
    struct SharedResources {
        nvrhi::BufferHandle       constantBuffer;
    };

    // Main color pass - tree geometry. Bark textures + sampler now come from
    // SharedGPUAssets — only the binding sets (one per shared texture set) and
    // the pass-local geometry/instance state live here.
    struct TreePassResources {
        nvrhi::ShaderHandle                    vertexShader;
        nvrhi::ShaderHandle                    pixelShader;
        nvrhi::InputLayoutHandle               inputLayout;
        nvrhi::BufferHandle                    instanceBuffer;
        nvrhi::BindingLayoutHandle             bindingLayout;
        std::vector<nvrhi::BindingSetHandle>   bindingSets;
        nvrhi::GraphicsPipelineHandle          pipeline;
    };

    // Shadow depth pass
    struct ShadowPassResources {
        nvrhi::TextureHandle                   depthTexture;
        nvrhi::FramebufferHandle               framebuffers[Render::c_NumCascades];
        nvrhi::TextureHandle                   cascadeDebugTextures[Render::c_NumCascades];
        nvrhi::ShaderHandle                    treeVS;
        nvrhi::ShaderHandle                    terrainVS;
        nvrhi::InputLayoutHandle               terrainInputLayout;
        nvrhi::InputLayoutHandle               treeInputLayout;
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

    // Sky pass (fullscreen procedural sky)
    struct SkyPassResources {
        nvrhi::ShaderHandle                    vertexShader;
        nvrhi::ShaderHandle                    pixelShader;
        nvrhi::BufferHandle                    constantBuffer;
        nvrhi::BindingLayoutHandle             bindingLayout;
        nvrhi::BindingSetHandle                bindingSet;
        nvrhi::GraphicsPipelineHandle          pipeline;
    };

    struct StageOwnedResources {
        SharedResources      frameShared;
        ShadowPassResources  shadowStage;
        SkyPassResources     skyStage;
        TreePassResources    sceneTreeStage;
        TerrainPassResources sceneTerrainStage;
    };

    StageOwnedResources                                m_StageResources;
    frame::StageOutputs                                m_StageOutputs;

    nvrhi::CommandListHandle                           m_CommandList;
    ViewHandler&                                       m_ViewHandler;
    std::shared_ptr<engine::ShaderFactory>             m_ShaderFactory;

    // nvrhi::TimerQueryHandle                            m_GpuTimers[m_QueuedFrames];
    // int                                                m_NextTimerIdx = 0;

    UIData&                                            m_UI;
    SceneRegistry&                                     m_Registry;
    SharedGPUAssets*                                   m_Shared = nullptr;

    // GPU-side asset tracking
    std::vector<GPUTreeAsset>                          m_GPUAssets;
    std::unordered_map<size_t, size_t>                 m_AssetIdToGPUIndex;

    // populated during render loop
    std::vector<Render::InstanceReference>             m_VisibleInstanceReferences;
    std::vector<std::vector<uint32_t>>                 m_InstanceCounts;
    std::vector<std::vector<uint32_t>>                 m_InstanceOffsets;
    std::vector<Render::DrawCmd>                       m_DrawCmds;
    std::vector<Render::InstanceBufferEntry>           m_VisibleInstanceBuffer;
    uint32_t                                           m_TotalShadowInstancesDrawn;

    // Per-cascade shadow draw data
    struct CascadeShadowData {
        std::vector<Render::InstanceReference>   visibleRefs;
        std::vector<Render::ShadowDrawCmd>       drawCmds;
        std::vector<Render::InstanceBufferEntry> instanceBuffer;
    };
    std::array<CascadeShadowData, Render::c_NumCascades> m_CascadeShadowData;

    bool _InitShared();
    bool _InitTreePass(nvrhi::ICommandList* initCL, engine::CommonRenderPasses& commonPasses);
    bool _InitShadowPass();
    bool _InitTerrainPass(nvrhi::ICommandList* initCL);
    bool _InitSkyPass();
    // bool _InitTimerQueries();

    void _UploadAllAssets(nvrhi::IDevice* device, nvrhi::ICommandList* commandList);
    void _UploadAsset(const TreeAssetDef& assetDef, GPUTreeAsset& gpuAsset,
                      nvrhi::IDevice* device, nvrhi::ICommandList* commandList);
    void _RebuildInstanceBuffers();
    void _RebuildBindingSets();

    void _RenderSkyPass(nvrhi::IFramebuffer* framebuffer);
    void _RenderShadowPass();
    void _RenderScenePass(nvrhi::IFramebuffer* framebuffer);
};

} // namespace Xylem

#endif // XYLEM_TRADITIONAL_RENDER_PASS_H



