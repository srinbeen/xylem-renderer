#ifndef XYLEM_FRAME_LIFECYCLE_H
#define XYLEM_FRAME_LIFECYCLE_H

#include <donut/core/math/math.h>
#include <cmath>

#include "../Render.hpp"
#include "../SceneRegistry.hpp"
#include "../ViewHandler.hpp"
#include "../macros.h"

namespace Xylem::frame {

inline constexpr float kDefaultNearPlane = 0.1f;
inline constexpr float kDefaultFarPlane = 1000.f;
inline constexpr float kDefaultVerticalFovDegrees = 60.f;

inline constexpr float k_ShadowDistanceBucketRatio = 1.1f;

// Shared across all render passes — depth of the per-pass timer query
// and GPU readback rings. Set to the minimum verified-safe value so
// stalls are avoided without paying for unnecessary in-flight memory.
inline constexpr uint32_t k_QueuedFrames = 3;

inline void UpdateProjectionAndViewport(ViewHandler& viewHandler,
                                        const nvrhi::FramebufferInfoEx& fbInfo,
                                        float verticalFovRadians = dm::radians(kDefaultVerticalFovDegrees),
                                        float nearPlane = kDefaultNearPlane,
                                        float farPlane = kDefaultFarPlane)
{
    viewHandler.view.SetViewport({ 
            static_cast<float>(fbInfo.width), 
            static_cast<float>(fbInfo.height) 
    });
    const float aspectRatio = viewHandler.view.GetAspectRatio();

    viewHandler.view.SetProjectionMatrix(
#if XYLEM_USE_REVERSE_Z
        dm::perspProjD3DStyleReverse(verticalFovRadians, aspectRatio, nearPlane)
#else
        dm::perspProjD3DStyle(verticalFovRadians, aspectRatio, nearPlane, farPlane)
#endif
    );
}

inline float ComputeShadowDistance(const dm::box3& sceneBbox,
                                   const dm::float3& cameraPos)
{
    float maxShadowDist = 0.f;
    for (int i = 0; i < dm::box3::numCorners; i++) {
        dm::float3 corner = sceneBbox.getCorner(i);
        maxShadowDist = dm::max(maxShadowDist, dm::length(corner - cameraPos));
    }
    maxShadowDist = dm::max(maxShadowDist, 1.f);

    // Quantize up to next geometric bucket: buckets at {ratio^k} for k in N.
    const float lnRatio   = std::logf(k_ShadowDistanceBucketRatio);
    const float bucketIdx = std::ceilf(std::logf(maxShadowDist) / lnRatio);
    return std::expf(bucketIdx * lnRatio);
}

template<typename TConstantBuffer>
inline void FillCommonFrameConstants(TConstantBuffer& constants,
                                     const ViewHandler& viewHandler,
                                     const dm::float3& sunDirection)
{
    constants.viewProj = viewHandler.view.GetViewProjectionMatrix();
    constants.viewMatrix = dm::affineToHomogeneous(viewHandler.view.GetViewMatrix());
    for (uint32_t c = 0; c < Render::c_NumCascades; c++)
        constants.lightViewProj[c] = viewHandler.cascades[c].lightViewProj;
    constants.sunLightDir = sunDirection;
    constants.cascadeSplits = viewHandler.cascadeSplitDistances;
}

inline void ComputeCascades(ViewHandler& viewHandler,
                             const SceneRegistry& registry,
                             float aspectRatio,
                             uint32_t shadowResolution,
                             float pssmLambda,
                             float verticalFovRadians = dm::radians(kDefaultVerticalFovDegrees),
                             float nearPlane = kDefaultNearPlane)
{
    const dm::box3& sceneBounds = registry.getSceneBounds();

    float maxShadowDist = ComputeShadowDistance(
        sceneBounds,
        viewHandler.camera.GetPosition());

    viewHandler.computeCascades(
        sceneBounds,
        registry.getSunDirection(),
        nearPlane,
        maxShadowDist,
        aspectRatio,
        verticalFovRadians,
        shadowResolution,
        pssmLambda);
}

// Per-pass GPU timer ring rotate-and-read.
//
// Call once per frame after executeCommandList. Reads the previous-frame's
// timer result if available (non-blocking poll), writes outMs in milliseconds,
// and advances nextIdx. Each pass owns its own ring; this just encapsulates
// the rotate/poll boilerplate so it isn't copy-pasted across three passes.
//
// outMs is left unchanged when no result is available yet (typical for the
// first k_QueuedFrames-1 frames after init or after a pipeline switch).
inline void RotateAndReadGpuTimer(
    nvrhi::IDevice* device,
    nvrhi::TimerQueryHandle (&timers)[k_QueuedFrames],
    uint32_t& nextIdx,
    float& outMs)
{
    const uint32_t prevIdx = (nextIdx + k_QueuedFrames - 1) % k_QueuedFrames;
    if (timers[prevIdx] && device->pollTimerQuery(timers[prevIdx])) {
        outMs = device->getTimerQueryTime(timers[prevIdx]) * 1000.0f;
    }
    nextIdx = (nextIdx + 1) % k_QueuedFrames;
}

} // namespace Xylem::frame

#endif // XYLEM_FRAME_LIFECYCLE_H
