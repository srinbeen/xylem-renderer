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
#include "Render.hpp"
#include "UIData.hpp"
#include "ViewHandler.hpp"

namespace Xylem {

using namespace donut;

class ComputeRenderPass : public app::IRenderPass {
public:
    static constexpr uint32_t k_QueuedFrames  = 3;
    static constexpr uint32_t k_ShadowRes     = 2048;
    static constexpr float    k_CapacitySlack = 1.5f;

    ComputeRenderPass(app::DeviceManager* dm, SceneRegistry& registry, UIData& ui, ViewHandler& vh)
        : IRenderPass{dm}, m_Registry{registry}, m_UI{ui}, m_ViewHandler{vh} {}

    void SetShaderFactory(std::shared_ptr<engine::ShaderFactory> sf) { m_ShaderFactory = std::move(sf); }

    ~ComputeRenderPass();
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
        nvrhi::BufferHandle              shadowCountBuffer;      // UAV uint32[numAssets]
        nvrhi::BufferHandle              shadowVisBuffer;        // UAV uint32[shadowVisBufferSize]
        nvrhi::BufferHandle              shadowSlotOffsetBuffer; // SRV uint32[numAssets]
        nvrhi::BufferHandle              shadowIndirectArgsBuffer; // UAV DrawIndexedIndirectArguments[numAssets] (also indirect args)
        nvrhi::BufferHandle              shadowUniqueCounter;    // UAV uint32 (single counter, raw); incremented once per instance visible in any cascade
        nvrhi::BufferHandle              regionVisibleBuffer;      // SRV uint32[numRegions], CPU-written each frame
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
        nvrhi::TextureHandle                                              depthTexture;      // Tex2DArray, 4 cascades
        std::array<nvrhi::FramebufferHandle, Render::c_NumCascades>       framebuffers;      // one per cascade slice
        std::array<nvrhi::TextureHandle, Render::c_NumCascades>           debugTextures;     // single-slice, for UI display
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
        nvrhi::TextureHandle                   hizTexture;         // RG32_FLOAT, mipchain, UAV.
                                                                    // .r = farthest (Hi-Z occlusion)
                                                                    // .g = nearest  (SDSM extrema)
        uint32_t                               numMips = 0;
        nvrhi::ShaderHandle                    copyCS;             // HiZCopy
        nvrhi::ShaderHandle                    buildCS;            // HiZDownsample
        nvrhi::ComputePipelineHandle           copyPipeline;
        nvrhi::ComputePipelineHandle           buildPipeline;
        nvrhi::BindingLayoutHandle             buildBindingLayout; // PushConstants(0) + SRV(0) + UAV(0)
        std::vector<nvrhi::BindingSetHandle>   buildBindingSets;   // one per mip transition
        nvrhi::SamplerHandle                   pointSampler;       // point/clamp for cull shader
        // Debug view: one single-mip RG32_FLOAT texture per mip level, GPU-copied for ImGui display
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
    SharedResources                                    m_Shared;
    CullPassResources                                  m_CullPass;
    TreePassResources                                  m_TreePass;
    ShadowPassResources                                m_ShadowPass;
    TerrainPassResources                               m_TerrainPass;
    SkyPassResources                                   m_SkyPass;
    DepthPrepassResources                              m_DepthPrepass;
    HiZPassResources                                   m_HiZ;
    SDSMPassResources                                  m_SDSM;

    nvrhi::CommandListHandle                           m_CommandList;
    ViewHandler&                                       m_ViewHandler;
    std::shared_ptr<engine::ShaderFactory>             m_ShaderFactory;

    // Readback ring buffer for GPU cull count display
    nvrhi::BufferHandle                                m_ReadbackBuffers[k_QueuedFrames];
    uint32_t                                           m_ReadbackFrameIndex    = 0;
    uint32_t                                           m_ReadbackCountEntries  = 0;
    uint32_t                                           m_ReadbackShadowEntries = 0;

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
    std::vector<Render::CullRegionData>                m_RegionStaging;

    // Slot layout (main pass: numAssets × numLods slots)
    uint32_t                                           m_NumSlots = 0;
    std::vector<uint32_t>                              m_SlotOffsets;       // prefix sums [numSlots]
    std::vector<uint32_t>                              m_MaxSlotCounts;     // max instances per slot [numSlots]
    uint32_t                                           m_VisBufferSize = 0;

    // Region CPU-frustum-cull visibility (written each frame, uploaded to regionVisibleBuffer)
    std::vector<uint32_t>                              m_RegionVisibleStaging;

    // Indirect args staging (pre-filled with indexCount, instanceCount=0)
    std::vector<nvrhi::DrawIndexedIndirectArguments>   m_IndirectArgsStaging;
    std::vector<nvrhi::DrawIndexedIndirectArguments>   m_ShadowIndirectArgsStaging;

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
    bool _InitHiZShaders();
    bool _InitSDSMPass();
    // bool _InitTimerQueries();

    void _UploadAllAssets(nvrhi::ICommandList* commandList);
    void _UploadAsset(const TreeAssetDef& assetDef, GPUTreeAsset& gpuAsset, nvrhi::ICommandList* commandList);
    void _BuildRegionWindows();
    void _BuildSlotLayout();
    void _UploadCullBuffers(nvrhi::ICommandList* commandList);
    void _RebuildCullBindings();

    // -----------------------------------------------------------------------
    // Render helpers
    // -----------------------------------------------------------------------
    void _EnsureHiZResources(uint32_t width, uint32_t height);
    void _RenderDepthPrepass();
    void _BuildHiZMipChain();
    void _RunSDSMBuildCascades(const dm::box3& sceneBbox, float aspectRatio, float fovY,
                               float regionEnvelopeNear, float regionEnvelopeFar);
    void _ComputeRegionEnvelope(const dm::frustum& viewFrustum,
                                const dm::float3& camPos, const dm::float3& camDir,
                                float& outNearZ, float& outFarZ) const;
    void _RenderSkyPass(nvrhi::IFramebuffer* framebuffer);
    void _RenderShadowPass();
    void _RenderScenePass(nvrhi::IFramebuffer* framebuffer);
};

} // namespace Xylem

#endif // XYLEM_COMPUTE_CULL_RENDER_PASS_H
