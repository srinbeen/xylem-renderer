#ifndef XYLEM_COMPUTE_CULL_RENDER_PASS_H
#define XYLEM_COMPUTE_CULL_RENDER_PASS_H

#include <donut/app/ApplicationBase.h>
#include <donut/app/Camera.h>
#include <donut/engine/View.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>

#include <nvrhi/nvrhi.h>
#include <nvrhi/d3d12.h>
#include <GFSDK_Aftermath.h>

#include <unordered_map>

#include "SceneRegistry.hpp"
#include "Render.hpp"
#include "UIData.hpp"
#include "ViewHandler.hpp"

namespace Xylem {

using namespace donut;

class ComputeCullRenderPass : public app::IRenderPass {
public:
    // static constexpr uint32_t k_QueuedFrames  = 4;
    static constexpr uint32_t k_ShadowRes     = 2048;
    static constexpr float    k_CapacitySlack = 1.5f;

    ComputeCullRenderPass(app::DeviceManager* dm, SceneRegistry& registry, UIData& ui, ViewHandler& vh)
        : IRenderPass{dm}, m_Registry{registry}, m_UI{ui}, m_ViewHandler{vh} {}

    void SetShaderFactory(std::shared_ptr<engine::ShaderFactory> sf) { m_ShaderFactory = std::move(sf); }

    ~ComputeCullRenderPass();
    bool Init();
    void Animate(float seconds) override;
    void BackBufferResizing() override;
    void Render(nvrhi::IFramebuffer* framebuffer) override;

    void onAssetsDirty(const std::vector<size_t>& dirtyAssetIndices);
    void onRegionsDirty(const std::vector<size_t>& dirtyRegionIndices);

private:
    struct TextureSet {
        nvrhi::TextureHandle diffuse;
        nvrhi::TextureHandle normalMap;
    };

    struct GPUTreeAsset {
        std::vector<Scene::TreeLODData> lods;
        uint32_t                        textureSetIdx;
    };

    struct RegionBufferWindow {
        uint32_t offset;
        uint32_t capacity;
        uint32_t count;
    };

    // -----------------------------------------------------------------------
    // Resource groups
    // -----------------------------------------------------------------------
    struct SharedResources {
        nvrhi::BufferHandle constantBuffer;  // CullConstantBufferEntry
    };

    // GPU compute cull resources
    struct CullPassResources {
        nvrhi::ShaderHandle              mainCS;
        nvrhi::ShaderHandle              shadowCS;
        nvrhi::ComputePipelineHandle     mainPipeline;
        nvrhi::ComputePipelineHandle     shadowPipeline;
        nvrhi::BindingLayoutHandle       bindingLayout;
        nvrhi::BindingSetHandle          bindingSet;

        nvrhi::BufferHandle              cullDataBuffer;       // SRV CullInstanceData[totalCapacity]
        nvrhi::BufferHandle              persistentInstBuffer; // SRV InstanceBufferEntry[totalCapacity]
        nvrhi::BufferHandle              slotOffsetBuffer;     // SRV uint32[numSlots]
        nvrhi::BufferHandle              countBuffer;          // UAV uint32[numSlots]
        nvrhi::BufferHandle              visibilityBuffer;     // UAV uint32[visBufferSize]
        nvrhi::BufferHandle              shadowCountBuffer;      // UAV uint32[numAssets]
        nvrhi::BufferHandle              shadowVisBuffer;        // UAV uint32[shadowVisBufferSize]
        nvrhi::BufferHandle              shadowSlotOffsetBuffer; // SRV uint32[numAssets]
    };

    // Tree draw pass — uses visibility indirection via structured buffers
    struct TreePassResources {
        nvrhi::ShaderHandle                    vertexShader;
        nvrhi::ShaderHandle                    pixelShader;
        nvrhi::InputLayoutHandle               inputLayout;
        std::vector<TextureSet>                textureSets;
        nvrhi::SamplerHandle                   sampler;
        nvrhi::BindingLayoutHandle             bindingLayout;
        std::vector<nvrhi::BindingSetHandle>   bindingSets;
        nvrhi::GraphicsPipelineHandle          pipeline;
    };

    struct ShadowPassResources {
        nvrhi::TextureHandle                   depthTexture;
        nvrhi::FramebufferHandle               framebuffer;
        nvrhi::ShaderHandle                    treeVS;
        nvrhi::ShaderHandle                    terrainVS;
        nvrhi::InputLayoutHandle               treeInputLayout;
        nvrhi::InputLayoutHandle               terrainInputLayout;
        nvrhi::SamplerHandle                   comparisonSampler;
        nvrhi::BindingLayoutHandle             bindingLayout;
        nvrhi::BindingSetHandle                bindingSet;
        nvrhi::GraphicsPipelineHandle          treePipeline;
        nvrhi::GraphicsPipelineHandle          terrainPipeline;
    };

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

    struct SkyPassResources {
        nvrhi::ShaderHandle                    vertexShader;
        nvrhi::ShaderHandle                    pixelShader;
        nvrhi::BufferHandle                    constantBuffer;
        nvrhi::BindingLayoutHandle             bindingLayout;
        nvrhi::BindingSetHandle                bindingSet;
        nvrhi::GraphicsPipelineHandle          pipeline;
    };

    // -----------------------------------------------------------------------
    // Members
    // -----------------------------------------------------------------------
    SharedResources                                    m_Shared;
    CullPassResources                                  m_CullPass;
    TreePassResources                                  m_TreePass;
    ShadowPassResources                                m_ShadowPass;
    TerrainPassResources                               m_TerrainPass;
    SkyPassResources                                   m_SkyPass;

    nvrhi::CommandListHandle                           m_CommandList;
    GFSDK_Aftermath_ContextHandle                      m_AftermathContext = nullptr;
    ViewHandler&                                       m_ViewHandler;
    std::shared_ptr<engine::ShaderFactory>             m_ShaderFactory;

    // nvrhi::TimerQueryHandle                            m_GpuTimers[k_QueuedFrames];
    // int                                                m_NextTimerIdx = 0;

    UIData&                                            m_UI;
    SceneRegistry&                                     m_Registry;

    // GPU-side asset tracking
    std::vector<GPUTreeAsset>                          m_GPUAssets;
    std::unordered_map<size_t, size_t>                 m_AssetIdToGPUIndex;

    // Persistent instance data (gapped layout, per-region windows)
    std::vector<RegionBufferWindow>                    m_RegionWindows;
    uint32_t                                           m_TotalCapacity = 0;
    std::vector<Render::InstanceBufferEntry>           m_InstanceStaging;
    std::vector<Render::CullInstanceData>              m_CullDataStaging;

    // Slot layout (main pass: numAssets × numLods slots)
    uint32_t                                           m_NumSlots = 0;
    std::vector<uint32_t>                              m_SlotOffsets;       // prefix sums [numSlots]
    std::vector<uint32_t>                              m_MaxSlotCounts;     // max instances per slot [numSlots]
    uint32_t                                           m_VisBufferSize = 0;

    // Shadow slot layout (per-asset, no LOD axis)
    std::vector<uint32_t>                              m_ShadowSlotOffsets; // prefix sums [numAssets]
    uint32_t                                           m_ShadowVisBufferSize = 0;

    // -----------------------------------------------------------------------
    // Init helpers
    // -----------------------------------------------------------------------
    bool _InitShared();
    bool _InitCullPass(nvrhi::ICommandList* initCL);
    bool _InitTreePass(nvrhi::ICommandList* initCL, engine::CommonRenderPasses& commonPasses);
    bool _InitShadowPass();
    bool _InitTerrainPass(nvrhi::ICommandList* initCL);
    bool _InitSkyPass();
    // bool _InitTimerQueries();

    void _UploadAllAssets(nvrhi::IDevice* device, nvrhi::ICommandList* commandList);
    void _UploadAsset(const TreeAssetDef& assetDef, GPUTreeAsset& gpuAsset,
                      nvrhi::IDevice* device, nvrhi::ICommandList* commandList);
    void _BuildRegionWindows();
    void _BuildSlotLayout();
    void _UploadCullBuffers(nvrhi::ICommandList* commandList);
    void _RebuildCullBindings();

    // -----------------------------------------------------------------------
    // Render helpers
    // -----------------------------------------------------------------------
    void _RenderSkyPass(nvrhi::IFramebuffer* framebuffer);
    void _RenderShadowPass();
    void _RenderScenePass(nvrhi::IFramebuffer* framebuffer);
};

} // namespace Xylem

#endif // XYLEM_COMPUTE_CULL_RENDER_PASS_H
