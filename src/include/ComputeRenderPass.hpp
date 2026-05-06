#ifndef XYLEM_COMPUTE_RENDER_PASS_H
#define XYLEM_COMPUTE_RENDER_PASS_H

#include <donut/app/ApplicationBase.h>
#include <donut/app/Camera.h>
#include <donut/engine/View.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>

#include <nvrhi/nvrhi.h>
#include <nvrhi/d3d12.h>

#include <array>
#include <unordered_map>

#include "SceneRegistry.hpp"
#include "SharedGPUAssets.hpp"
#include "Render.hpp"
#include "UIData.hpp"
#include "ViewHandler.hpp"
#include "macros.h"
#include "frame/FrameStages.hpp"

namespace Xylem {

using namespace donut;

class ComputeRenderPass : public app::IRenderPass, public frame::IFrameStagedPass {
public:
    static constexpr uint32_t k_QueuedFrames  = 3;
    static constexpr uint32_t k_ShadowRes     = 1 << 10;
    static constexpr float    k_CapacitySlack = 1.5f;
    // Hemi-octahedral view counts mirror SharedGPUAssets — kept here for the
    // per-pass impostor slot/visibility buffer sizing math.
    static constexpr uint32_t k_ImpostorAzimuthViews   = SharedGPUAssets::k_ImpostorAzimuthViews;
    static constexpr uint32_t k_ImpostorElevationViews = SharedGPUAssets::k_ImpostorElevationViews;
    static constexpr uint32_t k_ImpostorViewCount      = SharedGPUAssets::k_ImpostorViewCount;

    ComputeRenderPass(app::DeviceManager* dm, SceneRegistry& registry, UIData& ui, ViewHandler& vh)
        : IRenderPass{dm}, m_Registry{registry}, m_UI{ui}, m_ViewHandler{vh} {}

    void SetShaderFactory(std::shared_ptr<engine::ShaderFactory> sf) { m_ShaderFactory = std::move(sf); }
    void SetSharedAssets(SharedGPUAssets* shared) { m_Shared = shared; }

    ~ComputeRenderPass();
    bool Init();
    bool LoadResources() override;
    void Animate(float seconds) override;
    void BackBufferResizing() override;
    void Render(nvrhi::IFramebuffer* framebuffer) override;
    const frame::FrameStageOrder& GetFrameStageOrder() const override { return frame::kDefaultFrameStageOrder; }

    void onAssetsDirty(const std::vector<size_t>& dirtyAssetIndices);
    void onRegionsDirty(const std::vector<size_t>& dirtyRegionIndices);

private:
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
        nvrhi::ShaderHandle              regionCS;
        nvrhi::ComputePipelineHandle     mainPipeline;
        nvrhi::ComputePipelineHandle     regionPipeline;
        nvrhi::ComputePipelineHandle     shadowPipeline;
        nvrhi::BindingLayoutHandle       bindingLayout;
        nvrhi::BindingSetHandle          bindingSet;

        nvrhi::BufferHandle              cullDataBuffer;       // SRV CullInstanceData[totalCapacity]
        nvrhi::BufferHandle              persistentInstBuffer; // SRV InstanceBufferEntry[totalCapacity]
        nvrhi::BufferHandle              cullRegionDataBuffer; // SRV InstanceBufferEntry[totalCapacity]
        nvrhi::BufferHandle              slotOffsetBuffer;     // SRV uint32[numSlots]
        nvrhi::BufferHandle              countBuffer;          // UAV uint32[numSlots]
        nvrhi::BufferHandle              visibilityBuffer;     // UAV uint32[visBufferSize]
        nvrhi::BufferHandle              indirectArgsBuffer;   // UAV DrawIndexedIndirectArguments[numSlots] (also indirect args)
        nvrhi::BufferHandle              leafIndirectArgsBuffer; // UAV DrawIndirectArguments[numSlots]
        nvrhi::BufferHandle              impostorCountBuffer;        // UAV uint32[numAssets]
        nvrhi::BufferHandle              impostorVisBuffer;          // UAV uint32[impostorVisBufferSize]
        nvrhi::BufferHandle              impostorSlotOffsetBuffer;   // SRV uint32[numAssets]
        nvrhi::BufferHandle              impostorIndirectArgsBuffer; // UAV DrawIndirectArguments[numAssets]
        nvrhi::BufferHandle              shadowImpostorSlotOffsetBuffer;   // SRV uint32[numAssets * numCascades]
        nvrhi::BufferHandle              shadowImpostorCountBuffer;        // UAV uint32[numAssets * numCascades]
        nvrhi::BufferHandle              shadowImpostorVisBuffer;          // UAV uint32[shadowImpostorVisBufferSize]
        nvrhi::BufferHandle              shadowImpostorIndirectArgsBuffer; // UAV DrawIndirectArguments[numAssets * numCascades]
        nvrhi::BufferHandle              shadowCountBuffer;      // UAV uint32[numAssets]
        nvrhi::BufferHandle              shadowVisBuffer;        // UAV uint32[shadowVisBufferSize]
        nvrhi::BufferHandle              shadowSlotOffsetBuffer; // SRV uint32[numAssets]
        nvrhi::BufferHandle              shadowIndirectArgsBuffer; // UAV DrawIndexedIndirectArguments[numAssets] (also indirect args)
        nvrhi::BufferHandle              leafShadowIndirectArgsBuffer; // UAV DrawIndirectArguments[numAssets x cascades]
        nvrhi::BufferHandle              shadowUniqueCounter;    // UAV uint32 (single counter, raw); incremented once per instance visible in any cascade
        nvrhi::BufferHandle              regionVisibleBuffer;      // SRV uint32[numRegions], CPU-written each frame
    };

    // Tree draw pass — bark textures + sampler come from SharedGPUAssets.
    struct TreePassResources {
        nvrhi::ShaderHandle                    vertexShader;
        nvrhi::ShaderHandle                    pixelShader;
        nvrhi::InputLayoutHandle               inputLayout;
        nvrhi::BindingLayoutHandle             bindingLayout;
        std::vector<nvrhi::BindingSetHandle>   bindingSets;
        nvrhi::GraphicsPipelineHandle          pipeline;
    };

    struct LeafPassResources {
        nvrhi::ShaderHandle                    vertexShader;
        nvrhi::ShaderHandle                    pixelShader;
        nvrhi::ShaderHandle                    depthVS;
        nvrhi::ShaderHandle                    shadowVS;
        nvrhi::BindingLayoutHandle             bindingLayout;
        nvrhi::BindingLayoutHandle             depthBindingLayout;
        nvrhi::BindingLayoutHandle             shadowBindingLayout;
        nvrhi::BindingSetHandle                bindingSet;
        nvrhi::BindingSetHandle                depthBindingSet;
        nvrhi::BindingSetHandle                shadowBindingSet;
        nvrhi::GraphicsPipelineHandle          pipeline;
        nvrhi::GraphicsPipelineHandle          depthPipeline;
        nvrhi::GraphicsPipelineHandle          shadowPipeline;
    };

    // Impostor *render* side. The atlas, asset-dims buffer, debug atlas sheets,
    // and bake pipeline all live in SharedGPUAssets.
    struct ImpostorPassResources {
        nvrhi::ShaderHandle                    vertexShader;
        nvrhi::ShaderHandle                    pixelShader;
        nvrhi::BindingLayoutHandle             bindingLayout;
        nvrhi::SamplerHandle                   sampler;
        nvrhi::SamplerHandle                   depthSampler;
        std::vector<nvrhi::BindingSetHandle>   bindingSets;
        nvrhi::GraphicsPipelineHandle          pipeline;
    };

    // Shadow-pass impostor *render* side. Handles shadow depth writes for
    // terminal-LOD instances using the hemi-octahedral atlas for alpha cutout.
    struct ShadowImpostorPassResources {
        nvrhi::ShaderHandle                    vertexShader;
        nvrhi::ShaderHandle                    pixelShader;
        nvrhi::BindingLayoutHandle             bindingLayout;
        std::vector<nvrhi::BindingSetHandle>   bindingSets;  // size 1 (single binding set)
        nvrhi::GraphicsPipelineHandle          pipeline;
        nvrhi::SamplerHandle                   sampler;
        nvrhi::SamplerHandle                   depthSampler;
    };

    struct ShadowPassResources {
        nvrhi::TextureHandle                                              depthTexture;      // Tex2DArray, 4 cascades
        std::array<nvrhi::FramebufferHandle, Render::c_NumCascades>       framebuffers;      // one per cascade slice
        nvrhi::TextureHandle                                              debugSelectedCascadeTexture; // single slice, for UI display
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

    struct DepthPrepassResources {
        nvrhi::TextureHandle                   depthTexture;       // D32, main FB resolution
        nvrhi::FramebufferHandle               framebuffer;        // depth-only
        nvrhi::ShaderHandle                    treeVS;
        nvrhi::ShaderHandle                    terrainVS;
        nvrhi::InputLayoutHandle               treeInputLayout;    // position-only
        nvrhi::InputLayoutHandle               terrainInputLayout; // pos+normal+uv (match VB stride)
        nvrhi::GraphicsPipelineHandle          treePipeline;
        nvrhi::GraphicsPipelineHandle          terrainPipeline;
        nvrhi::BindingLayoutHandle             bindingLayout;      // CB(0) + PushConstants(1) + SRV(0,1,2)
        nvrhi::BindingSetHandle                bindingSet;
    };

    struct HiZPassResources {
        nvrhi::TextureHandle                   hizTexture;         // RGBA32_FLOAT, mipchain, UAV.
                                                                    // .r = raw farthest (Hi-Z occlusion, sky included)
                                                                    // .g = nearest      (SDSM near)
                                                                    // .b = sky-excluded farthest (SDSM far)
                                                                    // .a = unused
        uint32_t                               numMips = 0;
        nvrhi::ShaderHandle                    copyCS;             // HiZCopy
        nvrhi::ShaderHandle                    buildCS;            // HiZDownsample
        nvrhi::ComputePipelineHandle           copyPipeline;
        nvrhi::ComputePipelineHandle           buildPipeline;
        nvrhi::BindingLayoutHandle             buildBindingLayout; // PushConstants(0) + SRV(0) + UAV(0)
        std::vector<nvrhi::BindingSetHandle>   buildBindingSets;   // one per mip transition
        // Debug view: one single-mip RGBA32_FLOAT texture per mip level, GPU-copied for ImGui display
        std::vector<nvrhi::TextureHandle>      debugMipTextures;
    };

    // SDSM: GPU-side cascade construction from reduced depth bounds.
    struct SDSMPassResources {
        nvrhi::ShaderHandle                    buildCS;
        nvrhi::ComputePipelineHandle           buildPipeline;
        nvrhi::BindingLayoutHandle             buildBindingLayout; // CB(0) + SRV(0 hiz) + UAV(0 out)
        nvrhi::BindingSetHandle                buildBindingSet;
        nvrhi::BufferHandle                    inputCB;            // SDSMInput (CPU-written each frame)
        nvrhi::BufferHandle                    cascadeDataBuffer;  // UAV, copied into main CB
    };

    // -----------------------------------------------------------------------
    // Members
    // -----------------------------------------------------------------------
    struct StageOwnedResources {
        SharedResources             frameShared;
        DepthPrepassResources       depthPrepass;
        HiZPassResources            hiz;
        SDSMPassResources           sdsm;
        CullPassResources           cull;
        ImpostorPassResources       impostor;
        ShadowImpostorPassResources shadowImpostor;
        ShadowPassResources         shadow;
        SkyPassResources            sky;
        TreePassResources           sceneTree;
        LeafPassResources           sceneLeaves;
        TerrainPassResources        sceneTerrain;
    };

    StageOwnedResources                                m_StageResources;

    nvrhi::CommandListHandle                           m_CommandList;
    ViewHandler&                                       m_ViewHandler;
    std::shared_ptr<engine::ShaderFactory>             m_ShaderFactory;

    // Readback ring buffer for GPU cull count display
    nvrhi::BufferHandle                                m_ReadbackBuffers[k_QueuedFrames];
    uint32_t                                           m_ReadbackFrameIndex    = 0;
    uint32_t                                           m_ReadbackCountEntries  = 0;
    uint32_t                                           m_ReadbackImpostorEntries = 0;
    uint32_t                                           m_ReadbackShadowEntries = 0;

    // Readback ring for SDSM debug (mirrors cascadeDataBuffer; one slot per queued frame)
    nvrhi::BufferHandle                                m_SDSMReadbackBuffers[k_QueuedFrames];
    uint32_t                                           m_SDSMReadbackFrameIndex = 0;
    bool                                               m_SDSMReadbackPending[k_QueuedFrames] = { false, false, false };

    UIData&                                            m_UI;
    SceneRegistry&                                     m_Registry;
    SharedGPUAssets*                                   m_Shared = nullptr;

    // GPU-side asset tracking
    std::vector<GPUTreeAsset>                          m_GPUAssets;
    std::unordered_map<size_t, size_t>                 m_AssetIdToGPUIndex;

    // Persistent instance data (gapped layout, per-region windows)
    std::vector<RegionBufferWindow>                    m_RegionWindows;
    uint32_t                                           m_TotalCapacity = 0;
    std::vector<Render::InstanceBufferEntry>           m_InstanceStaging;
    std::vector<Render::CullInstanceData>              m_CullDataStaging;
    std::vector<Render::CullRegionData>                m_RegionStaging;

    // Slot layout (main pass: numAssets x numLods slots)
    uint32_t                                           m_NumSlots = 0;
    std::vector<uint32_t>                              m_SlotOffsets;       // prefix sums [numSlots]
    std::vector<uint32_t>                              m_MaxSlotCounts;     // max instances per slot [numSlots]
    uint32_t                                           m_VisBufferSize = 0;

    // Impostor terminal LOD layout (main pass: numAssets slots)
    std::vector<uint32_t>                              m_ImpostorSlotOffsets;
    std::vector<uint32_t>                              m_ImpostorMaxSlotCounts;
    uint32_t                                           m_ImpostorVisBufferSize = 0;

    // Shadow impostor slot layout: numAssets * numCascades
    std::vector<uint32_t>                              m_ShadowImpostorSlotOffsets;
    std::vector<uint32_t>                              m_ShadowImpostorMaxSlotCounts;
    uint32_t                                           m_ShadowImpostorVisBufferSize = 0;
    uint32_t                                           m_ReadbackShadowImpostorEntries = 0;

    // Region CPU-frustum-cull visibility (written each frame, uploaded to regionVisibleBuffer)
    std::vector<uint32_t>                              m_RegionVisibleStaging;

    // Indirect args staging (pre-filled with indexCount, instanceCount=0)
    std::vector<nvrhi::DrawIndexedIndirectArguments>   m_IndirectArgsStaging;
    std::vector<nvrhi::DrawIndirectArguments>          m_LeafIndirectArgsStaging;
    std::vector<nvrhi::DrawIndirectArguments>          m_ImpostorIndirectArgsStaging;
    std::vector<nvrhi::DrawIndirectArguments>          m_ShadowImpostorIndirectArgsStaging;
    std::vector<nvrhi::DrawIndexedIndirectArguments>   m_ShadowIndirectArgsStaging;
    std::vector<nvrhi::DrawIndirectArguments>          m_LeafShadowIndirectArgsStaging;

    // Shadow slot layout (per-asset, no LOD axis)
    std::vector<uint32_t>                              m_ShadowSlotOffsets; // prefix sums [numAssets]
    uint32_t                                           m_ShadowVisBufferSize = 0;

    // -----------------------------------------------------------------------
    // Init helpers
    // -----------------------------------------------------------------------
    bool _InitShared();
    bool _InitCullPass(nvrhi::ICommandList* initCL);
    bool _InitTreePass(nvrhi::ICommandList* initCL, engine::CommonRenderPasses& commonPasses);
    bool _InitLeafPass();
    bool _InitImpostorPass();
    bool _InitShadowImpostorPass();
    bool _InitShadowPass();
    bool _InitTerrainPass(nvrhi::ICommandList* initCL);
    bool _InitSkyPass();
    bool _InitHiZShaders();
    bool _InitSDSMPass();
    // bool _InitTimerQueries();

    void _UploadAllAssets(nvrhi::ICommandList* commandList);
    void _UploadAsset(const TreeAssetDef& assetDef, GPUTreeAsset& gpuAsset, nvrhi::ICommandList* commandList);
    void _BuildRegionWindows();
    void _BuildSlotLayout();
    void _UploadCullBuffers(nvrhi::ICommandList* commandList);
    void _RebuildCullBindings();
    void _RebuildShadowImpostorBindingSets();

    // -----------------------------------------------------------------------
    // Render helpers
    // -----------------------------------------------------------------------
    void _EnsureHiZResources(uint32_t width, uint32_t height);
    void _RenderDepthPrepass();
    void _BuildHiZMipChain();
    void _RunSDSMBuildCascades(const dm::box3& sceneBbox, float aspectRatio, float fovY,
                               float regionEnvelopeNear, float regionEnvelopeFar);
    void _ComputeRegionEnvelope(const dm::float3& camPos, const dm::float3& camDir,
                                float& outNearZ, float& outFarZ) const;
    void _RenderSkyPass(nvrhi::IFramebuffer* framebuffer);
    void _RenderShadowPass();
    void _RenderScenePass(nvrhi::IFramebuffer* framebuffer);
    void _RenderImpostorPass(nvrhi::IFramebuffer* framebuffer);
    void _RenderShadowImpostorPass(uint32_t cascade);
};

} // namespace Xylem

#endif // XYLEM_COMPUTE_CULL_RENDER_PASS_H




