#ifndef XYLEM_MESH_SHADER_RENDER_PASS_H
#define XYLEM_MESH_SHADER_RENDER_PASS_H

#include <donut/app/ApplicationBase.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>

#include <nvrhi/nvrhi.h>
#include <nvrhi/d3d12.h>

#include <array>
#include <unordered_map>
#include <vector>

#include "SceneRegistry.hpp"
#include "SharedGPUAssets.hpp"
#include "Render.hpp"
#include "UIData.hpp"
#include "ViewHandler.hpp"
#include "Meshlet.hpp"
#include "macros.h"
#include "shaders/ShaderContracts.hpp"
#include "frame/FrameStages.hpp"
#include "frame/FrameLifecycle.hpp"

namespace Xylem {

using namespace donut;

// GPU-driven mesh-shader pipeline. Mirrors ComputeRenderPass feature-for-feature:
//   1. Depth prepass via mesh pipeline -> D32 depth target
//   2. Hi-Z mip chain (RGBA32_FLOAT: .r=raw farthest for Hi-Z, .g=nearest for SDSM, .b=sky-excluded farthest for SDSM)
//   3. SDSM cascade build (consumes reduced Hi-Z depth)
//   4. GPU region + instance cull (MeshCullCS) -> per-slot DISPATCH_MESH args
//   5. Shadow pass: 4 cascades into Texture2DArray via mesh pipeline
//   6. Terrain + sky via traditional raster pipelines (reusing the compute path's shaders)
//   7. Main color pass via mesh pipeline with AS-side meshlet cone + Hi-Z cull + PCF shadows
class MeshShaderRenderPass : public app::IRenderPass, public frame::IFrameStagedPass {
public:
    static constexpr uint32_t k_QueuedFrames = Xylem::frame::k_QueuedFrames;
    static constexpr uint32_t k_ShadowRes     = 2048;
    static constexpr float    k_CapacitySlack = 1.5f;
    // Hemi-octahedral view counts mirror SharedGPUAssets — kept here for
    // per-pass impostor slot/visibility buffer sizing math.
    static constexpr uint32_t k_ImpostorAzimuthViews   = SharedGPUAssets::k_ImpostorAzimuthViews;
    static constexpr uint32_t k_ImpostorElevationViews = SharedGPUAssets::k_ImpostorElevationViews;
    static constexpr uint32_t k_ImpostorViewCount      = SharedGPUAssets::k_ImpostorViewCount;

    MeshShaderRenderPass(app::DeviceManager* dm, SceneRegistry& registry, UIData& ui, ViewHandler& vh)
        : IRenderPass{dm}, m_Registry{registry}, m_UI{ui}, m_ViewHandler{vh} {}

    void SetShaderFactory(std::shared_ptr<engine::ShaderFactory> sf) { m_ShaderFactory = std::move(sf); }
    void SetSharedAssets(SharedGPUAssets* shared) { m_Shared = shared; }

    bool Init();
    bool LoadResources() override;
    void Animate(float seconds) override;
    void BackBufferResizing() override;
    void Render(nvrhi::IFramebuffer* framebuffer) override;
    const frame::FrameStageOrder& GetFrameStageOrder() const override { return frame::kDefaultFrameStageOrder; }

    void onAssetsDirty(const std::vector<size_t>& dirtyAssetIndices);
    void onRegionsDirty(const std::vector<size_t>& dirtyRegionIndices);

private:
    struct RegionBufferWindow {
        uint32_t offset;
        uint32_t capacity;
        uint32_t count;
    };

    struct SharedResources {
        nvrhi::BufferHandle constantBuffer;   // CullConstantBufferEntry
        nvrhi::BufferHandle asCullCB;         // shader::cb::MeshASCullConstants
    };

    struct MeshletResources {
        nvrhi::BufferHandle positions;
        nvrhi::BufferHandle normals;
        nvrhi::BufferHandle tangents;
        nvrhi::BufferHandle bitangents;
        nvrhi::BufferHandle uvs;
        nvrhi::BufferHandle meshletVertIdx;
        nvrhi::BufferHandle meshletPrimIdx;   // raw byte buffer
        nvrhi::BufferHandle meshletDescs;
        nvrhi::BufferHandle assetLodRanges;

        uint32_t totalVertices     = 0;
        uint32_t totalMeshlets     = 0;
        uint32_t totalVertIdx      = 0;
        uint32_t totalPrimIdxBytes = 0;
        uint32_t numAssetLods      = 0;
    };

    // GPU cull pass resources. 16-byte DISPATCH_MESH records per slot
    // (slotIdx + groupsX/Y/Z).
    struct CullPassResources {
        nvrhi::ShaderHandle              mainCS;
        nvrhi::ShaderHandle              regionCS;
        nvrhi::ShaderHandle              shadowCS;
        nvrhi::ComputePipelineHandle     mainPipeline;
        nvrhi::ComputePipelineHandle     regionPipeline;
        nvrhi::ComputePipelineHandle     shadowPipeline;
        nvrhi::BindingLayoutHandle       bindingLayout;
        nvrhi::BindingSetHandle          bindingSet;

        nvrhi::BufferHandle              persistentInstBuffer;
        nvrhi::BufferHandle              cullDataBuffer;
        nvrhi::BufferHandle              cullRegionDataBuffer;

        // Main slot buffers (numMainSlots = numAssets * numLods)
        nvrhi::BufferHandle              mainSlotOffsetBuffer;
        nvrhi::BufferHandle              mainASInvocsPerSlotBuffer;
        nvrhi::BufferHandle              mainCountBuffer;
        nvrhi::BufferHandle              mainVisBuffer;
        nvrhi::BufferHandle              mainDispatchArgsBuffer;
        nvrhi::BufferHandle              mainLeafASInvocsPerSlotBuffer;  // SRV, AS groups per leaf-slot per instance
        nvrhi::BufferHandle              mainLeafDispatchArgsBuffer;     // UAV + indirect, parallel to mainDispatchArgsBuffer

        // Shadow slot buffers (numShadowSlots = numAssets * NUM_CASCADES, LOD 0 only)
        nvrhi::BufferHandle              shadowSlotOffsetBuffer;
        nvrhi::BufferHandle              shadowASInvocsPerSlotBuffer;
        nvrhi::BufferHandle              shadowCountBuffer;
        nvrhi::BufferHandle              shadowVisBuffer;
        nvrhi::BufferHandle              shadowDispatchArgsBuffer;
        nvrhi::BufferHandle              shadowLeafASInvocsPerSlotBuffer;
        nvrhi::BufferHandle              shadowLeafDispatchArgsBuffer;
        nvrhi::BufferHandle              shadowUniqueCounter;   // UAV raw, single uint32

        // Per-frame leaf survivor counters (single uint, raw UAV). leaf_as /
        // leaf_shadow_as atomically accumulate Σ meshlet.meta.y over surviving
        // meshlets. Cleared at frame start, copied to readback ring.
        nvrhi::BufferHandle              mainLeafSurvivorCounter;
        nvrhi::BufferHandle              depthLeafSurvivorScratch;  // bound to depth-prepass leaf set; never read
        nvrhi::BufferHandle              shadowLeafSurvivorCounter;

        // Per-frame meshlet AS cull stats (8 uints, raw UAV). Layout:
        //   [0] trunk main   considered  [1] trunk main   survived
        //   [2] trunk shadow considered  [3] trunk shadow survived
        //   [4] leaf  main   considered  [5] leaf  main   survived
        //   [6] leaf  shadow considered  [7] leaf  shadow survived
        // Cleared at frame start, copied to readback ring tail.
        nvrhi::BufferHandle              meshletStatsBuffer;
        // Depth prepass shares main_as / leaf_as with the color pass; binding the
        // real buffer to both paths would double-count main considered/survived.
        // Depth prepass writes here instead; values are discarded.
        nvrhi::BufferHandle              meshletStatsDepthScratch;

        // Impostor terminal-LOD slot buffers (numAssets slots, one per asset)
        nvrhi::BufferHandle              impostorSlotOffsetBuffer;   // SRV uint32[numAssets]
        nvrhi::BufferHandle              impostorCountBuffer;        // UAV uint32[numAssets]
        nvrhi::BufferHandle              impostorVisBuffer;          // UAV uint32[impostorVisBufferSize]
        nvrhi::BufferHandle              impostorIndirectArgsBuffer; // UAV DrawIndirectArguments[numAssets]

        // Shadow impostor cull buffers (per-(asset, cascade) slots)
        nvrhi::BufferHandle              shadowImpostorSlotOffsetBuffer;   // SRV uint32[numAssets * numCascades]
        nvrhi::BufferHandle              shadowImpostorCountBuffer;        // UAV uint32[numAssets * numCascades]
        nvrhi::BufferHandle              shadowImpostorVisBuffer;          // UAV uint32[shadowImpostorVisBufferSize]
        nvrhi::BufferHandle              shadowImpostorIndirectArgsBuffer; // UAV DrawIndirectArguments[numAssets * numCascades]

        nvrhi::BufferHandle              regionVisibleBuffer;
    };

    // Bark textures + sampler now come from SharedGPUAssets.
    struct DrawResources {
        nvrhi::ShaderHandle              amplificationShader;
        nvrhi::ShaderHandle              meshShader;
        nvrhi::ShaderHandle              pixelShader;
        nvrhi::SamplerHandle             shadowSampler;
        nvrhi::BindingLayoutHandle       bindingLayout;
        std::vector<nvrhi::BindingSetHandle> bindingSets;
        nvrhi::MeshletPipelineHandle     pipeline;
    };

    // Leaf AS/MS pipelines mirror DrawResources but consume LeafCommon.hlsli's
    // SRV layout (b0..b1, t0..t8) and the shared leaf instance/slot/meshlet
    // buffers from SharedGPUAssets. Three pipelines: main color, depth prepass,
    // and shadow. v1 has no per-leaf-meshlet cull — AS dispatch count is fixed
    // by leaf meshlet count for visible trees.
    struct LeafDrawResources {
        nvrhi::ShaderHandle              amplificationShader;
        nvrhi::ShaderHandle              shadowAmplificationShader;  // leaf_shadow_as: no Hi-Z cull
        nvrhi::ShaderHandle              meshShader;
        nvrhi::ShaderHandle              depthMS;
        nvrhi::ShaderHandle              shadowMS;
        nvrhi::ShaderHandle              pixelShader;
        nvrhi::BindingLayoutHandle       mainLayout;
        nvrhi::BindingLayoutHandle       depthLayout;
        nvrhi::BindingLayoutHandle       shadowLayout;
        nvrhi::BindingSetHandle          mainBindingSet;
        nvrhi::BindingSetHandle          depthBindingSet;
        nvrhi::BindingSetHandle          shadowBindingSet;
        nvrhi::MeshletPipelineHandle     mainPipeline;
        nvrhi::MeshletPipelineHandle     depthPipeline;
        nvrhi::MeshletPipelineHandle     shadowPipeline;
        nvrhi::RefCountPtr<ID3D12CommandSignature> mainSignature;
        nvrhi::RefCountPtr<ID3D12CommandSignature> depthSignature;
        nvrhi::RefCountPtr<ID3D12CommandSignature> shadowSignature;
    };

    // Hemi-octahedral impostor *render* side. Atlas, asset-dims, debug atlases,
    // and the bake pipeline live in SharedGPUAssets.
    struct ImpostorPassResources {
        nvrhi::ShaderHandle                    vertexShader;
        nvrhi::ShaderHandle                    pixelShader;
        nvrhi::BindingLayoutHandle             bindingLayout;
        nvrhi::SamplerHandle                   sampler;
        nvrhi::SamplerHandle                   depthSampler;
        std::vector<nvrhi::BindingSetHandle>   bindingSets;     // one per shared texture set
        nvrhi::GraphicsPipelineHandle          pipeline;
    };

    // Shadow-pass impostor *render* side. Handles shadow depth writes for
    // terminal-LOD instances using the hemi-octahedral atlas for alpha cutout.
    struct ShadowImpostorPassResources {
        nvrhi::ShaderHandle                    vertexShader;
        nvrhi::ShaderHandle                    pixelShader;
        nvrhi::BindingLayoutHandle             bindingLayout;
        std::vector<nvrhi::BindingSetHandle>   bindingSets;  // size 1 — single binding set
        nvrhi::GraphicsPipelineHandle          pipeline;
        nvrhi::SamplerHandle                   sampler;
        nvrhi::SamplerHandle                   depthSampler;
    };

    struct ShadowPassResources {
        nvrhi::TextureHandle                                              depthTexture;   // Texture2DArray, 4 cascades
        std::array<nvrhi::FramebufferHandle, Render::c_NumCascades>       framebuffers;
        nvrhi::TextureHandle                                              debugSelectedCascadeTexture; // single slice, for UI display

        nvrhi::ShaderHandle              amplificationShader;  // shadow_as
        nvrhi::ShaderHandle              meshShader;           // shadow_ms
        nvrhi::BindingLayoutHandle       bindingLayout;
        nvrhi::BindingSetHandle          bindingSet;
        nvrhi::MeshletPipelineHandle     pipeline;

        nvrhi::RefCountPtr<ID3D12CommandSignature> dispatchMeshSignature;

        // Terrain shadow caster (AS/MS path with per-meshlet light-frustum cull).
        // Reuses the eye-view terrain meshlet buffers; minimal binding layout
        // (push-constant cascade index + FrameCB + the four meshlet SRVs).
        nvrhi::ShaderHandle              terrainAS;
        nvrhi::ShaderHandle              terrainMS;
        nvrhi::BindingLayoutHandle       terrainBindingLayout;
        nvrhi::BindingSetHandle          terrainBindingSet;
        nvrhi::MeshletPipelineHandle     terrainPipeline;
    };

    struct DepthPrepassResources {
        nvrhi::TextureHandle             depthTexture;   // D32
        nvrhi::FramebufferHandle         framebuffer;    // depth-only

        nvrhi::ShaderHandle              amplificationShader;  // main_as
        nvrhi::ShaderHandle              meshShader;           // depth_ms
        nvrhi::BindingLayoutHandle       bindingLayout;
        nvrhi::BindingSetHandle          bindingSet;
        nvrhi::MeshletPipelineHandle     pipeline;

        nvrhi::RefCountPtr<ID3D12CommandSignature> dispatchMeshSignature;

        // Terrain into the same depth target via meshlet AS/MS — frustum +
        // Hi-Z culled per terrain meshlet, position-only MS output.
        nvrhi::ShaderHandle              terrainAS;        // depth_terrain_as
        nvrhi::ShaderHandle              terrainMS;        // depth_terrain_ms
        nvrhi::MeshletPipelineHandle     terrainPipeline;
        nvrhi::BindingLayoutHandle       terrainBindingLayout;
        nvrhi::BindingSetHandle          terrainBindingSet;
    };

    struct HiZPassResources {
        nvrhi::TextureHandle                   hizTexture;
        uint32_t                               numMips = 0;
        nvrhi::ShaderHandle                    copyCS;
        nvrhi::ShaderHandle                    buildCS;
        nvrhi::ComputePipelineHandle           copyPipeline;
        nvrhi::ComputePipelineHandle           buildPipeline;
        nvrhi::BindingLayoutHandle             buildBindingLayout;
        std::vector<nvrhi::BindingSetHandle>   buildBindingSets;
        std::vector<nvrhi::TextureHandle>      debugMipTextures;
    };

    struct SDSMPassResources {
        nvrhi::ShaderHandle                    buildCS;
        nvrhi::ComputePipelineHandle           buildPipeline;
        nvrhi::BindingLayoutHandle             buildBindingLayout;
        nvrhi::BindingSetHandle                buildBindingSet;
        nvrhi::BufferHandle                    inputCB;
        nvrhi::BufferHandle                    cascadeDataBuffer;
    };

    struct TerrainPassResources {
        // Vertex streams are SoA — separate raw buffers for positions, normals,
        // UVs. The meshlet AS/MS reads them as ByteAddressBuffer SRVs; the depth
        // prepass meshlet path binds only positions.
        nvrhi::BufferHandle                    positionBuffer;
        nvrhi::BufferHandle                    normalBuffer;
        nvrhi::BufferHandle                    uvBuffer;
        nvrhi::BufferHandle                    indexBuffer;  // unused by mesh path, kept for parity
        uint32_t                               indexCount = 0;

        // Mesh-shader scene path
        nvrhi::ShaderHandle                    amplificationShader;
        nvrhi::ShaderHandle                    meshShader;
        nvrhi::ShaderHandle                    pixelShader;
        nvrhi::BufferHandle                    meshletDescBuffer;
        nvrhi::BufferHandle                    meshletVertIdxBuffer;
        nvrhi::BufferHandle                    meshletPrimIdxBuffer;  // raw byte buffer
        nvrhi::BufferHandle                    visibleCounterBuffer;        // RW raw UAV, single uint (main pass)
        nvrhi::BufferHandle                    shadowVisibleCounterBuffer;  // RW raw UAV, single uint (sum across cascades)
        uint32_t                               meshletCount = 0;
        nvrhi::BindingLayoutHandle             bindingLayout;
        nvrhi::BindingSetHandle                bindingSet;
        nvrhi::MeshletPipelineHandle           pipeline;
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
    struct StageOwnedResources {
        SharedResources       frameShared;
        MeshletResources      sceneMeshletData;
        CullPassResources     cull;
        DrawResources         sceneDraw;
        LeafDrawResources     sceneLeaves;
        ImpostorPassResources       impostor;
        ShadowImpostorPassResources shadowImpostor;
        ShadowPassResources         shadow;
        DepthPrepassResources depthPrepass;
        HiZPassResources      hiz;
        SDSMPassResources     sdsm;
        TerrainPassResources  sceneTerrain;
        SkyPassResources      sky;
    };

    StageOwnedResources                               m_StageResources;

    nvrhi::CommandListHandle                          m_CommandList;
    ViewHandler&                                      m_ViewHandler;
    std::shared_ptr<engine::ShaderFactory>            m_ShaderFactory;

    UIData&                                           m_UI;
    SceneRegistry&                                    m_Registry;
    SharedGPUAssets*                                  m_Shared = nullptr;

    // Native D3D12 command signature for ExecuteIndirect(DISPATCH_MESH) on the
    // main color pipeline.
    nvrhi::RefCountPtr<ID3D12CommandSignature>        m_DispatchMeshSignature;

    // CPU mirror of mega-buffer layout
    Render::MeshletMegabuffers                        m_MeshletMegabuffers;

    // Persistent instance buffer bookkeeping
    std::vector<RegionBufferWindow>                   m_RegionWindows;
    uint32_t                                          m_TotalCapacity = 0;
    std::vector<Render::InstanceBufferEntry>          m_InstanceStaging;
    std::vector<Render::CullInstanceData>             m_CullDataStaging;
    std::vector<Render::CullRegionData>               m_RegionStaging;

    // Main slot layout (asset x LOD)
    uint32_t                                          m_NumMainSlots = 0;
    std::vector<uint32_t>                             m_MainSlotOffsets;
    std::vector<uint32_t>                             m_MainASInvocsPerSlot;
    std::vector<uint32_t>                             m_MainLeafASInvocsPerSlot;
    std::vector<uint32_t>                             m_MainSlotTextureSet;
    uint32_t                                          m_MainVisBufferSize = 0;

    // Shadow slot layout (asset x cascade, always LOD 0)
    uint32_t                                          m_NumShadowSlots = 0;
    std::vector<uint32_t>                             m_ShadowSlotOffsets;
    std::vector<uint32_t>                             m_ShadowASInvocsPerSlot;
    std::vector<uint32_t>                             m_ShadowLeafASInvocsPerSlot;
    uint32_t                                          m_ShadowVisBufferSize = 0;

    // Impostor terminal-LOD slot layout (per-asset: numAssets slots)
    std::vector<uint32_t>                             m_ImpostorSlotOffsets;
    std::vector<uint32_t>                             m_ImpostorMaxSlotCounts;
    uint32_t                                          m_ImpostorVisBufferSize = 0;

    // Shadow impostor slot layout: numAssets * numCascades
    std::vector<uint32_t>                             m_ShadowImpostorSlotOffsets;
    std::vector<uint32_t>                             m_ShadowImpostorMaxSlotCounts;
    uint32_t                                          m_ShadowImpostorVisBufferSize = 0;

    // Dispatch arg templates (re-uploaded each frame to reset groupsX to 0).
    struct DispatchRecord { uint32_t slotIdx; uint32_t gx; uint32_t gy; uint32_t gz; };
    std::vector<DispatchRecord>                       m_MainDispatchArgsStaging;
    std::vector<DispatchRecord>                       m_ShadowDispatchArgsStaging;
    std::vector<DispatchRecord>                       m_MainLeafDispatchArgsStaging;
    std::vector<DispatchRecord>                       m_ShadowLeafDispatchArgsStaging;

    // Readback ring: [main counts][shadow counts][impostor counts][shadow unique]
    nvrhi::BufferHandle                               m_ReadbackBuffers[k_QueuedFrames];
    uint32_t                                          m_ReadbackFrameIndex            = 0;
    uint32_t                                          m_ReadbackMainEntries           = 0;
    uint32_t                                          m_ReadbackShadowEntries         = 0;
    uint32_t                                          m_ReadbackImpostorEntries       = 0;
    uint32_t                                          m_ReadbackShadowImpostorEntries = 0;

    // Readback ring for SDSM debug (mirrors ComputeRenderPass)
    nvrhi::BufferHandle                               m_SDSMReadbackBuffers[k_QueuedFrames];
    uint32_t                                          m_SDSMReadbackFrameIndex = 0;
    bool                                              m_SDSMReadbackPending[k_QueuedFrames] = {};

    // GPU frame timer ring buffer
    nvrhi::TimerQueryHandle                            m_GpuTimers[k_QueuedFrames];
    uint32_t                                           m_NextTimerIdx = 0;

    // Per-stage GPU timer rings. Indexed by [FrameStage][ringSlot].
    // Wrapped via frame::BeginGpuStage / frame::EndGpuStage in Render().
    static constexpr size_t kStageCount = static_cast<size_t>(frame::FrameStage::COUNT);
    nvrhi::TimerQueryHandle                            m_StageTimers[kStageCount][k_QueuedFrames];
    uint32_t                                           m_StageNextIdx[kStageCount] = {};

    // -----------------------------------------------------------------------
    // Init helpers
    // -----------------------------------------------------------------------
    bool _InitShared();
    bool _InitDrawResources();
    bool _InitLeafResources();
    bool _InitCullResources();
    bool _InitShadowPass();
    bool _InitDepthPrepass();
    bool _InitHiZShaders();
    bool _InitSDSMPass();
    bool _InitTerrainPass(nvrhi::ICommandList* initCL);
    bool _InitSkyPass();
    bool _InitImpostorPass();
    bool _InitShadowImpostorPass();

    void _RebuildMeshletMegabuffers(nvrhi::ICommandList* cl);
    void _UploadMeshletMegabuffers(nvrhi::ICommandList* cl);
    void _BuildRegionWindows();
    void _BuildSlotLayout();
    void _UploadCullBuffers(nvrhi::ICommandList* cl);
    void _RebuildCullBindingSet();
    void _RebuildDrawBindingSet();
    void _RebuildShadowBindingSet();
    void _RebuildDepthPrepassBindingSet();
    void _RebuildLeafBindingSets();
    void _RebuildTerrainBindingSet();

    void _CreateMainPipelineIfNeeded(nvrhi::IFramebuffer* framebuffer);
    void _CreateShadowPipelineIfNeeded();
    void _CreateShadowTerrainPipelineIfNeeded();
    void _CreateDepthPrepassPipelineIfNeeded();
    void _CreateLeafPipelinesIfNeeded(nvrhi::IFramebuffer* framebuffer);
    void _EnsureDispatchMeshSignatures();
    void _EnsureLeafDispatchMeshSignatures();

    // Per-frame helpers
    void _EnsureHiZResources(uint32_t width, uint32_t height);
    void _RenderDepthPrepass();
    void _BuildHiZMipChain();
    void _RunSDSMBuildCascades(const dm::box3& sceneBbox, float aspectRatio, float fovY,
                               float regionEnvelopeNear, float regionEnvelopeFar);
    void _ComputeRegionEnvelope(const dm::float3& camPos, const dm::float3& camDir,
                                float& outNearZ, float& outFarZ) const;
    void _RenderSkyPass(nvrhi::IFramebuffer* framebuffer);
    void _RenderShadowPass();
    void _RenderTrunkPass(nvrhi::IFramebuffer* framebuffer);
    void _RenderLeavesPass(nvrhi::IFramebuffer* framebuffer);
    void _RenderTerrainPass(nvrhi::IFramebuffer* framebuffer);
    void _RenderImpostorPass(nvrhi::IFramebuffer* framebuffer);
    void _RebuildImpostorBindingSets();
    void _ClearShadowMaps();
    void _RenderShadowImpostorPass(uint32_t cascade);
    void _RebuildShadowImpostorBindingSets();
};

} // namespace Xylem

#endif // XYLEM_MESH_SHADER_RENDER_PASS_H




