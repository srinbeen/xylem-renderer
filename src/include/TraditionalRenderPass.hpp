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
#include "frame/FrameStages.hpp"
#include "frame/FrameLifecycle.hpp"

namespace Xylem {

using namespace donut;

class TraditionalRenderPass : public app::IRenderPass, public frame::IFrameStagedPass {
public:
    static constexpr uint32_t k_QueuedFrames = Xylem::frame::k_QueuedFrames;
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
        m_StageResources.leafStage.pipeline = nullptr;
        m_StageResources.leafStage.shadowPipeline = nullptr;
        m_StageResources.impostorStage.pipeline = nullptr;
        m_StageResources.shadowImpostorStage.pipeline = nullptr;
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

    // Hemi-octahedral impostor render pass. Atlas textures + asset dims come
    // from SharedGPUAssets; this pass owns per-frame visibility/instance SRVs.
    struct ImpostorPassResources {
        nvrhi::ShaderHandle                    vertexShader;
        nvrhi::ShaderHandle                    pixelShader;
        nvrhi::BindingLayoutHandle             bindingLayout;
        nvrhi::SamplerHandle                   sampler;
        nvrhi::SamplerHandle                   depthSampler;
        std::vector<nvrhi::BindingSetHandle>   bindingSets;
        nvrhi::GraphicsPipelineHandle          pipeline;
        nvrhi::BufferHandle                    instanceBuffer;
        nvrhi::BufferHandle                    cullDataBuffer;
        nvrhi::BufferHandle                    visBuffer;
        nvrhi::BufferHandle                    slotOffsetBuffer;
    };

    // Shadow billboard pass — distant trees cast as light-aligned cards
    // sampling the impostor depth atlas instead of full geometry.
    // Slot layout: ai * Render::c_NumCascades + c. Mirrors P1's
    // ShadowImpostorPassResources but CPU-driven (no GPU cull / indirect).
    struct ShadowImpostorPassResources {
        nvrhi::ShaderHandle                    vertexShader;
        nvrhi::ShaderHandle                    pixelShader;
        nvrhi::BindingLayoutHandle             bindingLayout;
        std::vector<nvrhi::BindingSetHandle>   bindingSets;
        nvrhi::GraphicsPipelineHandle          pipeline;
        nvrhi::BufferHandle                    instanceBuffer;
        nvrhi::BufferHandle                    cullDataBuffer;
        nvrhi::BufferHandle                    visBuffer;
        nvrhi::BufferHandle                    slotOffsetBuffer;
    };

    struct LeafPassResources {
        nvrhi::ShaderHandle           vertexShader;
        nvrhi::ShaderHandle           pixelShader;
        nvrhi::ShaderHandle           shadowVS;
        nvrhi::InputLayoutHandle      inputLayout;
        nvrhi::InputLayoutHandle      shadowInputLayout;
        nvrhi::BindingLayoutHandle    bindingLayout;
        nvrhi::BindingLayoutHandle    shadowBindingLayout;
        nvrhi::BindingSetHandle       bindingSet;
        nvrhi::BindingSetHandle       shadowBindingSet;
        nvrhi::GraphicsPipelineHandle pipeline;
        nvrhi::GraphicsPipelineHandle shadowPipeline;
    };

    // Shadow depth pass
    struct ShadowPassResources {
        nvrhi::TextureHandle                   depthTexture;
        nvrhi::FramebufferHandle               framebuffers[Render::c_NumCascades];
        nvrhi::TextureHandle                   debugSelectedCascadeTexture;
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
        SharedResources             frameShared;
        ShadowPassResources         shadowStage;
        SkyPassResources            skyStage;
        TreePassResources           sceneTreeStage;
        LeafPassResources           leafStage;
        ImpostorPassResources       impostorStage;
        ShadowImpostorPassResources shadowImpostorStage;
        TerrainPassResources        sceneTerrainStage;
    };

    StageOwnedResources                                m_StageResources;

    nvrhi::CommandListHandle                           m_CommandList;
    ViewHandler&                                       m_ViewHandler;
    std::shared_ptr<engine::ShaderFactory>             m_ShaderFactory;

    nvrhi::TimerQueryHandle                            m_GpuTimers[k_QueuedFrames];
    uint32_t                                           m_NextTimerIdx = 0;

    // Per-stage GPU timer rings. Indexed by [FrameStage][ringSlot].
    // Wrapped via frame::BeginGpuStage / frame::EndGpuStage in Render().
    static constexpr size_t kStageCount = static_cast<size_t>(frame::FrameStage::COUNT);
    nvrhi::TimerQueryHandle                            m_StageTimers[kStageCount][k_QueuedFrames];
    uint32_t                                           m_StageNextIdx[kStageCount] = {};

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
    std::vector<Render::InstanceReference>             m_VisibleImpostorReferences;
    std::vector<uint32_t>                              m_ImpostorCounts;
    std::vector<uint32_t>                              m_ImpostorWriteOffsets;
    std::vector<uint32_t>                              m_ImpostorSlotOffsets;
    std::vector<uint32_t>                              m_ImpostorMaxSlotCounts;
    uint32_t                                           m_ImpostorVisBufferSize = 0;
    std::vector<Render::InstanceBufferEntry>           m_ImpostorInstanceStaging;
    std::vector<Render::CullInstanceData>              m_ImpostorCullDataStaging;
    std::vector<uint32_t>                              m_ImpostorVisStaging;
    uint32_t                                           m_TotalShadowInstancesDrawn;

    // Per-cascade shadow billboard refs (parallel to m_CascadeShadowData
    // but for the impostor path).
    std::array<std::vector<Render::InstanceReference>, Render::c_NumCascades>
                                                       m_CascadeShadowImpostorRefs;

    // Frame-local staging for the shadow impostor pass. Built by
    // _RenderShadowPass after the cull loop, uploaded with writeBuffer.
    // visStaging is packed asset-major-cascade-minor; counts/offsets
    // are sized numAssets * Render::c_NumCascades.
    std::vector<Render::InstanceBufferEntry>           m_ShadowImpostorInstanceStaging;
    std::vector<Render::CullInstanceData>              m_ShadowImpostorCullStaging;
    std::vector<uint32_t>                              m_ShadowImpostorVisStaging;
    std::vector<uint32_t>                              m_ShadowImpostorCounts;        // numAssets * c_NumCascades
    std::vector<uint32_t>                              m_ShadowImpostorSlotOffsets;   // numAssets * c_NumCascades
    std::vector<uint32_t>                              m_ShadowImpostorMaxSlotCounts; // numAssets * c_NumCascades, allocated capacity
    uint32_t                                           m_ShadowImpostorVisBufferSize = 0;
    uint32_t                                           m_ShadowImpostorInstanceCapacity = 0;

    // Per-cascade shadow draw data
    struct CascadeShadowData {
        std::vector<Render::InstanceReference>   visibleRefs;
        std::vector<Render::ShadowDrawCmd>       drawCmds;
        std::vector<Render::InstanceBufferEntry> instanceBuffer;
    };
    std::array<CascadeShadowData, Render::c_NumCascades> m_CascadeShadowData;

    bool _InitShared();
    bool _InitTreePass();
    bool _InitLeafPass();
    bool _InitImpostorPass();
    bool _InitShadowImpostorPass();
    bool _InitShadowPass();
    bool _InitTerrainPass(nvrhi::ICommandList* initCL);
    bool _InitSkyPass();
    bool _InitDebug();
    // bool _InitTimerQueries();

    void _UploadAllAssets(nvrhi::IDevice* device, nvrhi::ICommandList* commandList);
    void _UploadAsset(const TreeAssetDef& assetDef, GPUTreeAsset& gpuAsset,
                      nvrhi::IDevice* device, nvrhi::ICommandList* commandList);
    void _RebuildInstanceBuffers();
    void _BuildImpostorSlotLayout();
    void _RebuildImpostorBuffers();
    void _BuildShadowImpostorSlotLayout();
    void _RebuildShadowImpostorBuffers();
    void _RebuildBindingSets();

    void _RenderSkyPass(nvrhi::IFramebuffer* framebuffer);
    void _RenderShadowPass();
    void _RenderShadowImpostorPass(uint32_t cascade);
    void _PrepareSceneFrame(nvrhi::IFramebuffer* framebuffer);
    void _RenderTrunkPass(nvrhi::IFramebuffer* framebuffer);
    void _RenderLeavesPass(nvrhi::IFramebuffer* framebuffer);
    void _RenderTerrainPass(nvrhi::IFramebuffer* framebuffer);
    void _RenderImpostorPass(nvrhi::IFramebuffer* framebuffer);
};

} // namespace Xylem

#endif // XYLEM_TRADITIONAL_RENDER_PASS_H



