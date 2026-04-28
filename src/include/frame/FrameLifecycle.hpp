#ifndef XYLEM_FRAME_LIFECYCLE_H
#define XYLEM_FRAME_LIFECYCLE_H

#include <algorithm>
#include <cstddef>

#include <donut/core/math/math.h>

#include "../SceneRegistry.hpp"
#include "../ViewHandler.hpp"
#include "FrameContracts.hpp"

namespace Xylem::frame {

inline constexpr float kDefaultNearPlane = 0.1f;
inline constexpr float kDefaultFarPlane = 1000.f;
inline constexpr float kDefaultVerticalFovDegrees = 60.f;

template<typename PassT>
inline void RunDirtyCycle(SceneRegistry& registry, PassT& pass)
{
    if (!registry.anyDirty()) return;

    auto dirtyAssets = registry.getDirtyAssetIndices();
    auto dirtyRegions = registry.getDirtyRegionIndices();

    registry.rebuildDirtyAssets();
    registry.rebuildDirtyRegions();

    if (!dirtyAssets.empty()) pass.onAssetsDirty(dirtyAssets);
    if (!dirtyRegions.empty()) pass.onRegionsDirty(dirtyRegions);

    registry.clearDirtyFlags();
}

inline dm::box3 ComputeSceneBounds(const SceneRegistry& registry)
{
    dm::box3 sceneBbox = dm::box3::empty();
    for (const auto& region : registry.getRegions())
        sceneBbox |= region.cullBox;

    const auto* terrain = registry.getTerrain();
    if (terrain)
        sceneBbox |= terrain->getBbox();

    return sceneBbox;
}

inline float ComputeForwardShadowDistance(const dm::box3& sceneBbox,
                                          const dm::float3& cameraPos,
                                          const dm::float3& cameraDir)
{
    float maxShadowDist = 0.f;
    for (int i = 0; i < dm::box3::numCorners; i++) {
        dm::float3 corner = sceneBbox.getCorner(i);
        float cornerDir = dm::dot(corner - cameraPos, cameraDir);
        if (cornerDir > 0.f)
            maxShadowDist = dm::max(maxShadowDist, cornerDir);
    }
    return dm::max(maxShadowDist, 1.f);
}

inline void EnsurePerspectiveProjection(ViewHandler& viewHandler,
                                        uint32_t width,
                                        uint32_t height,
                                        bool projectionDirty,
                                        float verticalFovRadians = dm::radians(kDefaultVerticalFovDegrees),
                                        float nearPlane = kDefaultNearPlane,
                                        float farPlane = kDefaultFarPlane)
{
    if (!projectionDirty) return;

    float aspect = float(width) / float(height);
    viewHandler.view.SetViewport({ float(width), float(height) });
    viewHandler.view.SetProjectionMatrix(
#if XYLEM_USE_REVERSE_Z
        dm::perspProjD3DStyleReverse(verticalFovRadians, aspect, nearPlane)
#else
        dm::perspProjD3DStyle(verticalFovRadians, aspect, nearPlane, farPlane)
#endif
    );
}

inline void UpdateView(ViewHandler& viewHandler)
{
    viewHandler.view.SetViewMatrix(viewHandler.camera.GetWorldToViewMatrix());
    viewHandler.view.UpdateCache();
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

inline FrameContext BuildFrameContext(const SceneRegistry& registry,
                                      ViewHandler& viewHandler,
                                      nvrhi::IFramebuffer* framebuffer,
                                      bool projectionDirty,
                                      float verticalFovRadians = dm::radians(kDefaultVerticalFovDegrees),
                                      float nearPlane = kDefaultNearPlane,
                                      float farPlane = kDefaultFarPlane)
{
    const auto& fbInfo = framebuffer->getFramebufferInfo();
    EnsurePerspectiveProjection(
        viewHandler, fbInfo.width, fbInfo.height, projectionDirty,
        verticalFovRadians, nearPlane, farPlane);

    UpdateView(viewHandler);

    FrameContext context;
    context.framebuffer = framebuffer;
    context.frameWidth = fbInfo.width;
    context.frameHeight = fbInfo.height;
    context.aspectRatio = float(fbInfo.width) / float(fbInfo.height);
    context.verticalFovRadians = verticalFovRadians;
    context.nearPlane = nearPlane;
    context.farPlane = farPlane;
    context.sceneBounds = ComputeSceneBounds(registry);
    context.maxShadowDistance = ComputeForwardShadowDistance(
        context.sceneBounds,
        viewHandler.camera.GetPosition(),
        viewHandler.camera.GetDir());
    return context;
}

inline void ComputeCascades(FrameContext& context,
                            ViewHandler& viewHandler,
                            const SceneRegistry& registry,
                            uint32_t shadowResolution,
                            float pssmLambda)
{
    viewHandler.computeCascades(
        context.sceneBounds,
        registry.getSunDirection(),
        context.nearPlane,
        context.maxShadowDistance,
        context.aspectRatio,
        context.verticalFovRadians,
        shadowResolution,
        pssmLambda);
}

inline void SetShadowDebugOutputs(StageOutputs& outputs,
                                  nvrhi::ITexture* shadowMap,
                                  const std::array<nvrhi::TextureHandle, Render::c_NumCascades>& cascades)
{
    outputs.debug.shadowMapTexture = shadowMap;
    for (uint32_t c = 0; c < Render::c_NumCascades; c++)
        outputs.debug.shadowCascadeTextures[c] = cascades[c].Get();
}

inline void SetShadowDebugOutputs(StageOutputs& outputs,
                                  nvrhi::ITexture* shadowMap,
                                  nvrhi::TextureHandle cascades[Render::c_NumCascades])
{
    outputs.debug.shadowMapTexture = shadowMap;
    for (uint32_t c = 0; c < Render::c_NumCascades; c++)
        outputs.debug.shadowCascadeTextures[c] = cascades[c].Get();
}

inline void SetHiZDebugOutputs(StageOutputs& outputs,
                               const std::vector<nvrhi::TextureHandle>& hizMips)
{
    outputs.debug.hizMipTextures.resize(hizMips.size());
    for (size_t i = 0; i < hizMips.size(); i++)
        outputs.debug.hizMipTextures[i] = hizMips[i].Get();
}

inline void ClearHiZDebugOutputs(StageOutputs& outputs)
{
    outputs.debug.hizMipTextures.clear();
}

} // namespace Xylem::frame

#endif // XYLEM_FRAME_LIFECYCLE_H
