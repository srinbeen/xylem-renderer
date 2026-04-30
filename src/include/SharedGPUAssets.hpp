#ifndef XYLEM_SHARED_GPU_ASSETS_H
#define XYLEM_SHARED_GPU_ASSETS_H

#include <memory>
#include <vector>

#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/ShaderFactory.h>
#include <nvrhi/nvrhi.h>

#include "SceneRegistry.hpp"
#include "UIData.hpp"
#include "macros.h"

namespace Xylem {

// Pipeline-agnostic GPU resources used by every render pass:
//   - Bark texture sets (diffuse + normal map per material)
//   - Hemi-octahedral impostor atlas (Texture2DArrays + per-asset bbox dims)
// Owned by RenderOrchestrator. Each pass binds these resources directly into
// its own per-pass binding sets — they're nvrhi-refcounted so sharing is safe.
//
// Render-side impostor pipeline (PSO + binding sets that reference per-pass
// cull buffers) stays inside each pass; only the bake side and atlas live here.
class SharedGPUAssets {
public:
    static constexpr uint32_t k_ImpostorAzimuthViews   = XYLEM_IMPOSTOR_AZIMUTH_VIEWS;
    static constexpr uint32_t k_ImpostorElevationViews = XYLEM_IMPOSTOR_ELEVATION_VIEWS;
    static constexpr uint32_t k_ImpostorViewCount      = XYLEM_IMPOSTOR_VIEW_COUNT;
    static constexpr uint32_t k_ImpostorBakeResolution = 256;

    struct TextureSet {
        nvrhi::TextureHandle diffuse;
        nvrhi::TextureHandle normalMap;
    };

    SharedGPUAssets(nvrhi::IDevice* device, SceneRegistry& registry, UIData& ui)
        : m_Device(device), m_Registry(registry), m_UI(ui) {}

    void SetShaderFactory(std::shared_ptr<donut::engine::ShaderFactory> sf) { m_ShaderFactory = std::move(sf); }

    // Loads bark textures + bakes the impostor atlas. Opens its own command
    // list. Call before any pass's Init() so passes can reference these.
    bool Init();

    // Re-runs bark-texture load + impostor bake in response to asset-list changes.
    // Does NOT clear registry dirty flags — leaves that to the active pass's
    // RunDirtyCycle. Call from RenderOrchestrator before delegating dirty fanout.
    bool OnAssetsDirty();

    // --- Accessors used by render passes when building binding sets ---
    const std::vector<TextureSet>& barkTextures() const { return m_BarkTextures; }
    nvrhi::SamplerHandle           barkSampler()  const { return m_BarkSampler; }

    nvrhi::ITexture* impostorAlbedo() const { return m_AlbedoAlphaTexture; }
    nvrhi::ITexture* impostorNormal() const { return m_NormalTexture; }
    nvrhi::ITexture* impostorDepth()  const { return m_DepthTexture; }
    nvrhi::IBuffer*  assetDimsBuffer() const { return m_AssetDimsBuffer; }

    nvrhi::ITexture* impostorDebugAlbedoAtlas() const { return m_DebugAlbedoAtlasTexture; }
    nvrhi::ITexture* impostorDebugNormalAtlas() const { return m_DebugNormalAtlasTexture; }
    nvrhi::ITexture* impostorDebugDepthAtlas()  const { return m_DebugDepthAtlasTexture; }

private:
    bool _InitBakePipeline();
    bool _LoadBarkTextures(nvrhi::ICommandList* cl, donut::engine::CommonRenderPasses& commonPasses);
    bool _BakeImpostors(nvrhi::ICommandList* cl);
    bool _RebuildAssetDimsBuffer(nvrhi::ICommandList* cl);

    nvrhi::IDevice*                                  m_Device;
    SceneRegistry&                                   m_Registry;
    UIData&                                          m_UI;
    std::shared_ptr<donut::engine::ShaderFactory>    m_ShaderFactory;

    // Bark textures (one TextureSet per registered bark material)
    std::vector<TextureSet>                          m_BarkTextures;
    nvrhi::SamplerHandle                             m_BarkSampler;

    // Impostor bake pipeline
    nvrhi::ShaderHandle                              m_BakeVS;
    nvrhi::ShaderHandle                              m_BakePS;
    nvrhi::InputLayoutHandle                         m_BakeInputLayout;
    nvrhi::BindingLayoutHandle                       m_BakeBindingLayout;
    std::vector<nvrhi::BindingSetHandle>             m_BakeBindingSets;
    nvrhi::BufferHandle                              m_BakeConstantBuffer;
    nvrhi::GraphicsPipelineHandle                    m_BakePipeline;

    // Impostor atlas + asset dims
    nvrhi::TextureHandle                             m_AlbedoAlphaTexture;
    nvrhi::TextureHandle                             m_NormalTexture;
    nvrhi::TextureHandle                             m_DepthTexture;
    nvrhi::TextureHandle                             m_DebugAlbedoAtlasTexture;
    nvrhi::TextureHandle                             m_DebugNormalAtlasTexture;
    nvrhi::TextureHandle                             m_DebugDepthAtlasTexture;
    nvrhi::BufferHandle                              m_AssetDimsBuffer;
};

} // namespace Xylem

#endif // XYLEM_SHARED_GPU_ASSETS_H
