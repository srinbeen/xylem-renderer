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
#include "Render.hpp"
#include "UIData.hpp"
#include "ViewHandler.hpp"
#include "Meshlet.hpp"

namespace Xylem {

using namespace donut;

// GPU-driven mesh-shader pipeline. Per-instance cull runs on the GPU
// (MeshCullCS fork of CullCS), writes D3D12_DISPATCH_MESH_ARGUMENTS into a
// per-slot indirect buffer, and ExecuteIndirect launches one DispatchMesh per
// (asset × LOD) slot. Hi-Z + shadow paths deferred.
class MeshShaderRenderPass : public app::IRenderPass {
public:
    static constexpr uint32_t k_QueuedFrames  = 3;
    static constexpr float    k_CapacitySlack = 1.5f;

    MeshShaderRenderPass(app::DeviceManager* dm, SceneRegistry& registry, UIData& ui, ViewHandler& vh)
        : IRenderPass{dm}, m_Registry{registry}, m_UI{ui}, m_ViewHandler{vh} {}

    void SetShaderFactory(std::shared_ptr<engine::ShaderFactory> sf) { m_ShaderFactory = std::move(sf); }

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

    struct RegionBufferWindow {
        uint32_t offset;
        uint32_t capacity;
        uint32_t count;
    };

    struct SharedResources {
        nvrhi::BufferHandle constantBuffer;  // CullConstantBufferEntry — prefix is ConstantBufferEntry
    };

    // CPU-side meshlet mega-buffers (concatenation of per-(asset × LOD) data)
    struct MeshletResources {
        nvrhi::BufferHandle positions;
        nvrhi::BufferHandle normals;
        nvrhi::BufferHandle tangents;
        nvrhi::BufferHandle bitangents;
        nvrhi::BufferHandle uvs;
        nvrhi::BufferHandle meshletVertIdx;   // uint32 SRV
        nvrhi::BufferHandle meshletPrimIdx;   // raw byte buffer (ByteAddressBuffer in HLSL)
        nvrhi::BufferHandle meshletDescs;     // MeshletDesc SRV
        nvrhi::BufferHandle assetLodRanges;   // MeshOffsets SRV

        uint32_t totalVertices     = 0;
        uint32_t totalMeshlets     = 0;
        uint32_t totalVertIdx      = 0;
        uint32_t totalPrimIdxBytes = 0;
        uint32_t numAssetLods      = 0;
    };

    // GPU cull pass resources — mirrors ComputeRenderPass::CullPassResources,
    // but without shadows and with a custom 16-byte-per-slot dispatch-args buffer
    // (slotIdx + 3× uint for D3D12_DISPATCH_MESH_ARGUMENTS).
    struct CullPassResources {
        nvrhi::ShaderHandle              mainCS;
        nvrhi::ShaderHandle              regionCS;
        nvrhi::ComputePipelineHandle     mainPipeline;
        nvrhi::ComputePipelineHandle     regionPipeline;
        nvrhi::BindingLayoutHandle       bindingLayout;
        nvrhi::BindingSetHandle          bindingSet;

        nvrhi::BufferHandle              persistentInstBuffer;  // SRV InstanceBufferEntry[totalCapacity]
        nvrhi::BufferHandle              cullDataBuffer;        // SRV CullInstanceData[totalCapacity]
        nvrhi::BufferHandle              cullRegionDataBuffer;  // SRV CullRegionData[numRegions]
        nvrhi::BufferHandle              slotOffsetBuffer;      // SRV uint32[numSlots]
        nvrhi::BufferHandle              ASInvocsPerSlotBuffer; // SRV uint32[numSlots]
        nvrhi::BufferHandle              regionVisibleBuffer;   // UAV uint32[numRegions]
        nvrhi::BufferHandle              countBuffer;           // UAV uint32[numSlots]
        nvrhi::BufferHandle              visibilityBuffer;      // UAV uint32[visBufferSize]
        nvrhi::BufferHandle              dispatchArgsBuffer;    // UAV raw + indirect, 16 bytes per slot
    };

    struct DrawResources {
        nvrhi::ShaderHandle              amplificationShader;
        nvrhi::ShaderHandle              meshShader;
        nvrhi::ShaderHandle              pixelShader;
        std::vector<TextureSet>          textureSets;
        nvrhi::SamplerHandle             sampler;
        nvrhi::BindingLayoutHandle       bindingLayout;
        nvrhi::BindingSetHandle          bindingSet;
        nvrhi::MeshletPipelineHandle     pipeline;
    };

    // -----------------------------------------------------------------------
    // Members
    // -----------------------------------------------------------------------
    SharedResources                                   m_Shared;
    MeshletResources                                  m_Meshlet;
    CullPassResources                                 m_Cull;
    DrawResources                                     m_Draw;

    nvrhi::CommandListHandle                          m_CommandList;
    ViewHandler&                                      m_ViewHandler;
    std::shared_ptr<engine::ShaderFactory>            m_ShaderFactory;

    UIData&                                           m_UI;
    SceneRegistry&                                    m_Registry;

    // Native D3D12 command signature for ExecuteIndirect(DISPATCH_MESH).
    // NVRHI does not expose dispatchMeshIndirect; this is invoked directly on the
    // native graphics command list after setMeshletState().
    nvrhi::RefCountPtr<ID3D12CommandSignature>        m_DispatchMeshSignature;

    // CPU mirror of mega-buffer layout — rebuilt on asset dirty.
    Render::MeshletMegabuffers                        m_MeshletMegabuffers;

    // Persistent instance buffer bookkeeping
    std::vector<RegionBufferWindow>                   m_RegionWindows;
    uint32_t                                          m_TotalCapacity;
    std::vector<Render::InstanceBufferEntry>          m_InstanceStaging;
    std::vector<Render::CullInstanceData>             m_CullDataStaging;
    std::vector<Render::CullRegionData>               m_RegionStaging;

    // Slot (asset × lod) layout
    uint32_t                                          m_NumSlots = 0;
    std::vector<uint32_t>                             m_SlotOffsets;
    std::vector<uint32_t>                             m_ASInvocsPerSlot;
    uint32_t                                          m_VisBufferSize = 0;

    // Each frame-initial record for dispatchArgsBuffer: { slotIdx, 0, 1, 1 }.
    // Layout matches the command signature: u32 slotIdx (consumed by root
    // constant), then u32 groupsX/Y/Z consumed by DISPATCH_MESH.
    struct DispatchRecord { uint32_t slotIdx; uint32_t gx; uint32_t gy; uint32_t gz; };
    std::vector<DispatchRecord>                       m_DispatchArgsStaging;

    // Readback ring for visible-instance count (UI stats)
    nvrhi::BufferHandle                               m_ReadbackBuffers[k_QueuedFrames];
    uint32_t                                          m_ReadbackFrameIndex = 0;

    // -----------------------------------------------------------------------
    // Init helpers
    // -----------------------------------------------------------------------
    bool _InitShared();
    bool _InitDrawResources();
    bool _InitCullResources();
    bool _LoadBarkTextures(nvrhi::ICommandList* initCL, engine::CommonRenderPasses& commonPasses);

    void _RebuildMeshletMegabuffers(nvrhi::ICommandList* cl);
    void _UploadMeshletMegabuffers(nvrhi::ICommandList* cl);
    void _BuildRegionWindows();
    void _BuildSlotLayout();
    void _UploadCullBuffers(nvrhi::ICommandList* cl);
    void _RebuildCullBindingSet();
    void _RebuildDrawBindingSet();

    void _CreatePipelineIfNeeded(nvrhi::IFramebuffer* framebuffer);
    void _EnsureDispatchMeshSignature();
};

} // namespace Xylem

#endif // XYLEM_MESH_SHADER_RENDER_PASS_H
