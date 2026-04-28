#ifndef XYLEM_FRAME_CONTRACTS_H
#define XYLEM_FRAME_CONTRACTS_H

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

#include <donut/core/math/math.h>
#include <nvrhi/nvrhi.h>

#include "../Render.hpp"
#include "../UIData.hpp"
#include "FrameStages.hpp"

namespace Xylem::frame {

struct FrameContext {
    nvrhi::IFramebuffer* framebuffer = nullptr;
    uint32_t             frameWidth = 0;
    uint32_t             frameHeight = 0;
    float                aspectRatio = 1.f;
    float                verticalFovRadians = 0.f;
    float                nearPlane = 0.1f;
    float                farPlane = 1000.f;
    dm::box3             sceneBounds = dm::box3::empty();
    float                maxShadowDistance = 1.f;
    bool                 hizActive = false;
};

struct StageResourceDescriptor {
    FrameStage       stage;
    std::string_view owner;
    std::string_view name;
};

struct StageResourceCatalog {
    std::vector<StageResourceDescriptor> gpuResources;
    std::vector<StageResourceDescriptor> cpuResources;

    void AddGpu(FrameStage stage, std::string_view owner, std::string_view name)
    {
        gpuResources.push_back({ stage, owner, name });
    }

    void AddCpu(FrameStage stage, std::string_view owner, std::string_view name)
    {
        cpuResources.push_back({ stage, owner, name });
    }
};

struct FrameDebugOutputs {
    void* shadowMapTexture = nullptr;
    std::array<void*, Render::c_NumCascades> shadowCascadeTextures{};
    std::vector<void*> hizMipTextures;
};

struct StageOutputs {
    FrameDebugOutputs debug;
};

inline void PublishStageOutputsToUI(const StageOutputs& outputs, UIData& ui)
{
    ui.shadowMapTexture = outputs.debug.shadowMapTexture;

    ui.shadowCascadeTextures.resize(Render::c_NumCascades);
    for (uint32_t i = 0; i < Render::c_NumCascades; i++)
        ui.shadowCascadeTextures[i] = outputs.debug.shadowCascadeTextures[i];

    ui.hizMipTextures = outputs.debug.hizMipTextures;
}

} // namespace Xylem::frame

#endif // XYLEM_FRAME_CONTRACTS_H
