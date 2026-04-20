#ifndef XYLEM_MESH_SHADER_RENDER_PASS_H
#define XYLEM_MESH_SHADER_RENDER_PASS_H

#include <donut/app/ApplicationBase.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>

#include <nvrhi/nvrhi.h>

#include <unordered_map>
#include <vector>

#include "SceneRegistry.hpp"
#include "Render.hpp"
#include "UIData.hpp"
#include "ViewHandler.hpp"
#include "Meshlet.hpp"

namespace Xylem {

using namespace donut;

// MVP mesh-shader pipeline. Trees only, CPU-side frustum + LOD cull,
// AS refines meshlets, MS emits verts+prims, PS does bark+normal+Lambert.
// Shadows / Hi-Z / terrain / sky are deferred follow-ups.
class MeshShaderRenderPass : public app::IRenderPass {
public:
    static constexpr uint32_t k_QueuedFrames = 3;

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

    struct SharedResources {
        nvrhi::BufferHandle constantBuffer;  // Render::ConstantBufferEntry
    };

    // One big bundle of GPU mega-buffers — owned by this pass.
    struct MeshletResources {
        // Per-vertex SoA (flattened across every asset × LOD)
        nvrhi::BufferHandle positions;
        nvrhi::BufferHandle normals;
        nvrhi::BufferHandle tangents;
        nvrhi::BufferHandle bitangents;
        nvrhi::BufferHandle uvs;
        // Meshlet index tables (packed across all meshlets)
        nvrhi::BufferHandle meshletVertIdx;   // uint32 SRV
        nvrhi::BufferHandle meshletPrimIdx;   // raw byte buffer (ByteAddressBuffer in HLSL)
        // Per-meshlet descriptors + per-assetLod ranges
        nvrhi::BufferHandle meshletDescs;     // MeshletDesc SRV
        nvrhi::BufferHandle assetLodRanges;   // AssetLodRange SRV
        // Per-frame visible-instance list (CPU culled for MVP)
        nvrhi::BufferHandle visibleInstances; // VisibleInstance SRV
        nvrhi::BufferHandle instanceBuffer;   // InstanceBufferEntry SRV (model+normal+treeId)

        uint32_t totalVertices     = 0;
        uint32_t totalMeshlets     = 0;
        uint32_t totalVertIdx      = 0;
        uint32_t totalPrimIdxBytes = 0;
        uint32_t numAssetLods      = 0;
        uint32_t visCapacity       = 0;
        uint32_t instanceCapacity  = 0;
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
    DrawResources                                     m_Draw;

    nvrhi::CommandListHandle                          m_CommandList;
    ViewHandler&                                      m_ViewHandler;
    std::shared_ptr<engine::ShaderFactory>            m_ShaderFactory;

    UIData&                                           m_UI;
    SceneRegistry&                                    m_Registry;

    // CPU mirror of mega-buffer layout — rebuilt on asset dirty.
    Render::MeshletBuild                              m_MeshletCPU;

    // Per-frame staging
    std::vector<Render::InstanceBufferEntry>          m_InstanceStaging;
    std::vector<Render::VisibleInstance>              m_VisibleStaging;

    bool _InitShared();
    bool _InitDrawResources();
    bool _LoadBarkTextures(nvrhi::ICommandList* initCL, engine::CommonRenderPasses& commonPasses);

    void _RebuildMeshletMegaBuffers(nvrhi::ICommandList* cl);

    void _CreatePipelineIfNeeded(nvrhi::IFramebuffer* framebuffer);
    void _RebuildBindingSet();
};

} // namespace Xylem

#endif // XYLEM_MESH_SHADER_RENDER_PASS_H
