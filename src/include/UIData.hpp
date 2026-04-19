#ifndef XYLEM_UI_DATA_H
#define XYLEM_UI_DATA_H

#include <cstdint>
#include <vector>

namespace Xylem {

enum class Pipeline { Traditional = 0, ComputeCull = 1 };

struct UIData {
    bool ShowUI = true;

    Pipeline activePipeline    = Pipeline::ComputeCull;
    Pipeline requestedPipeline = Pipeline::ComputeCull;

    float    gpuFrameTimeMs       = -1.0f; // -1 = not yet available
    float    cpuRenderTimeMs      = 0.0f;
    uint32_t visibleInstanceCount = 0;
    uint32_t drawCallCount        = 0;
    uint32_t totalInstanceCount   = 0;
    uint32_t culledInstanceCount  = 0;
    uint32_t shadowVisibleCount   = 0;
    uint32_t shadowCulledCount    = 0;

    bool showDebugTopDown       = false;
    bool showDebugShadowTopDown = false;
    bool showShadowMap          = false;
    bool showHiZ                = false;

    float hizBypassAngle   = 1.0f;  // downwardness threshold to skip Hi-Z (0=never bypass, 1=always)
    bool  hizActiveThisFrame = true; // read-only, set by render pass

    float pssmLambda       = 0.85f; // PSSM/SDSM blend: 0=linear splits, 1=logarithmic splits

    void*    shadowMapTexture = nullptr;  // nvrhi::ITexture*, set by active render pass
    std::vector<void*> shadowCascadeTextures;  // per-cascade nvrhi::ITexture* for debug display
    // Hi-Z: one nvrhi::ITexture* per mip level (single-mip scratch textures), set by ComputeCullRenderPass
    // Each entry is a separate R32_FLOAT texture containing exactly one mip, copied each frame.
    std::vector<void*> hizMipTextures;
};

} // namespace Xylem

#endif // XYLEM_UI_DATA_H
