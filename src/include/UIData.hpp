#ifndef XYLEM_UI_DATA_H
#define XYLEM_UI_DATA_H

#include <cstdint>
#include <string>
#include <vector>

namespace Xylem {

enum class Pipeline { Traditional = 0, Compute = 1, MeshShader = 2 };

struct UIData {
    bool ShowUI = true;

    Pipeline activePipeline    = Pipeline::Compute;
    Pipeline requestedPipeline = Pipeline::Compute;

    float    gpuFrameTimeMs       = -1.0f; // -1 = not yet available
    float    cpuRenderTimeMs      = 0.0f;
    uint32_t visibleInstanceCount = 0;
    uint32_t impostorVisibleCount = 0;
    uint32_t drawCallCount        = 0;
    uint32_t totalInstanceCount   = 0;
    uint32_t culledInstanceCount  = 0;
    uint32_t shadowVisibleCount     = 0; // unique instances contributing to any cascade
    uint32_t shadowCulledCount      = 0; // totalInstanceCount - shadowVisibleCount
    uint32_t shadowCascadeDrawCount = 0; // sum over all cascades of per-cascade visible (counts overdraw)
    uint32_t shadowOverdrawCount    = 0; // shadowCascadeDrawCount - shadowVisibleCount

    bool showDebugTopDown       = false;
    bool showDebugShadowTopDown = false;
    bool showShadowMap          = false;
    bool showHiZ                = false;
    bool showImpostorAtlas      = false;

    float hizBypassAngle   = 1.0f;  // downwardness threshold to skip Hi-Z (0=never bypass, 1=always)
    bool  hizActiveThisFrame = true; // read-only, set by render pass

    float pssmLambda       = 0.85f; // PSSM/SDSM blend: 0=linear splits, 1=logarithmic splits
    float impostorAlphaClip = 0.25f;

    // Mesh-shader pipeline stats
    uint32_t asMeshletsDispatched = 0;
    uint32_t asMeshletsCulled     = 0;
    uint32_t msInvocations        = 0;
    float    meshletMegaBufferMB  = 0.0f;
    uint32_t totalMeshletCount    = 0;

    // SDSM debug readback (populated when hizActiveThisFrame and SDSM ran).
    bool     sdsmDebugValid       = false;
    float    sdsmNearDepthVal     = 0.f;
    float    sdsmFarDepthVal      = 0.f;
    float    sdsmTightNear        = 0.f;
    float    sdsmTightFar         = 0.f;
    float    sdsmCascadeSplits[4] = { 0.f, 0.f, 0.f, 0.f };
    float    sdsmCascade0MinLS[3] = { 0.f, 0.f, 0.f };
    float    sdsmCascade0MaxLS[3] = { 0.f, 0.f, 0.f };

    void*    shadowMapTexture = nullptr;  // nvrhi::ITexture*, set by active render pass
    // Single per-pass scratch texture into which the active pass copies the
    // currently-selected cascade slice. UIRenderer drives selectedCascade via the slider.
    int      selectedCascade        = 0;
    void*    selectedCascadeTexture = nullptr;
    // Hi-Z: one nvrhi::ITexture* per mip level (single-mip scratch textures), set by ComputeCullRenderPass
    // Each entry is a separate R32_FLOAT texture containing exactly one mip, copied each frame.
    std::vector<void*> hizMipTextures;

    // Impostor atlas debug: one entry per (asset, view) slice. Set by ComputeRenderPass after the
    // bake completes. Layout: index = assetIdx * impostorViewsPerAsset + viewIdx.
    uint32_t impostorViewsPerAsset = 0;
    uint32_t impostorAzimuthViews  = 0;
    uint32_t impostorElevationViews = 0;
    uint32_t impostorAssetCount    = 0;
    uint32_t impostorSelectedAsset = 0;
    void*    impostorAlbedoAtlasTexture = nullptr;
    void*    impostorNormalAtlasTexture = nullptr;
    void*    impostorDepthAtlasTexture = nullptr;

    // --- Scene Save / Load ---
    // UIRenderer writes; RenderOrchestrator reads + clears.
    bool        requestedSceneLoad     = false;
    std::string requestedScenePath;
    // Status surface for the Scene File UI section. Empty = no message displayed.
    // Save writes this directly from UIRenderer; Load writes it from the orchestrator
    // after _loadSceneIfRequested completes.
    std::string sceneLoadStatus;
    bool        sceneLoadStatusIsError = false;
};

} // namespace Xylem

#endif // XYLEM_UI_DATA_H
