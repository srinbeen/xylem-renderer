#include "include/ViewHandler.hpp"
#include "include/macros.h"
#include <cmath>

using namespace Xylem;

void ViewHandler::updateShadowVolume(const dm::box3& sceneBbox, dm::float3 sunDirection)
{
    worldToLight = dm::lookatZ(sunDirection, dm::float3(0,1,0)) * dm::scaling(dm::float3(1.f, 1.f, -1.f));
    dm::box3 sceneBoundsLS = sceneBbox * worldToLight;

    const dm::float3& camPos = camera.GetPosition();
    const dm::float3& camDir = camera.GetDir();
    const dm::frustum& viewFrustum = view.GetViewFrustum();
    shadowCasterBboxLS = dm::box3::empty();
    if (!viewFrustum.intersectsWith(sceneBbox)) return;

    #if !XYLEM_USE_REVERSE_Z
    const dm::frustum& camFrustum = viewFrustum;
    #else
    float maxShadowDist = 0.f;
    for (int i = 0; i < dm::box3::numCorners; i++) {
        dm::float3 corner = sceneBbox.getCorner(i);
        dm::float3 camToCorner = corner - camPos;
        float cornerDir = dm::dot(camToCorner, camDir);
        if (cornerDir > 0.0f) maxShadowDist = dm::max(maxShadowDist, dm::length(camToCorner));
    }

    const nvrhi::Viewport& vp = view.GetViewport();
    dm::float4x4 finiteProj = dm::perspProjD3DStyle(
        dm::radians(60.f), 
        vp.width() / vp.height(), 
        0.1f, maxShadowDist
    );

    const dm::frustum camFrustum(
        dm::affineToHomogeneous(view.GetViewMatrix()) * finiteProj, false
    );
    #endif

    for (int i = 0; i < dm::frustum::numCorners; i++) {
        dm::float3 clampedCornerLS = worldToLight.transformPoint(camFrustum.getCorner(i));
        clampedCornerLS = sceneBoundsLS.clamp(clampedCornerLS);
        shadowCasterBboxLS |= clampedCornerLS;
    }
}

void ViewHandler::computeCascades(const dm::box3& sceneBbox, dm::float3 sunDirection,
                                   float nearPlane, float farPlane, float aspectRatio, float fovY,
                                   uint32_t shadowRes)
{
    worldToLight = dm::lookatZ(sunDirection, dm::float3(0,1,0)) * dm::scaling(dm::float3(1.f, 1.f, -1.f));
    dm::box3 sceneBboxLS = sceneBbox * worldToLight;

    // TODO: read Parallel-Split Shadow Maps
    constexpr float lambda = 0.7f;
    constexpr float pssmNearPlane = 10.f;
    constexpr uint32_t N = Render::c_NumCascades;
    float splits[N + 1];
    // float splits[N + 1] = {
    //     nearPlane,
    //     dm::lerp(nearPlane, farPlane, 0.05f),
    //     dm::lerp(nearPlane, farPlane, 0.15f),
    //     dm::lerp(nearPlane, farPlane, 0.5f),
    //     dm::lerp(nearPlane, farPlane, 1.0f),
    // };

    splits[0] = nearPlane;
    for (uint32_t i = 1; i <= N; i++) {
        float t = float(i) / float(N);
        float logSplit = pssmNearPlane * std::powf(farPlane / pssmNearPlane, t);
        float linSplit = pssmNearPlane + (farPlane - pssmNearPlane) * t;
        splits[i] = lambda * logSplit + (1.0f - lambda) * linSplit;
    }

    cascadeSplitDistances = dm::float4(splits[1], splits[2], splits[3], splits[4]);

    float tanHalfFovY = std::tanf(fovY * 0.5f);
    float tanHalfFovX = tanHalfFovY * aspectRatio;

    dm::affine3 viewToWorldToLight = view.GetInverseViewMatrix() * worldToLight;

    shadowCasterBboxLS = dm::box3::empty();

    for (uint32_t c = 0; c < N; c++) {
        float cNear = splits[c];
        float cFar  = splits[c + 1];

        // 8 frustum corners in view space
        dm::float3 cornersVS[dm::frustum::numCorners] = {
            { -tanHalfFovX * cNear, -tanHalfFovY * cNear, cNear },
            {  tanHalfFovX * cNear, -tanHalfFovY * cNear, cNear },
            { -tanHalfFovX * cNear,  tanHalfFovY * cNear, cNear },
            {  tanHalfFovX * cNear,  tanHalfFovY * cNear, cNear },
            { -tanHalfFovX * cFar,  -tanHalfFovY * cFar,  cFar  },
            {  tanHalfFovX * cFar,  -tanHalfFovY * cFar,  cFar  },
            { -tanHalfFovX * cFar,   tanHalfFovY * cFar,  cFar  },
            {  tanHalfFovX * cFar,   tanHalfFovY * cFar,  cFar  },
        };

        // Transform to light space and compute AABB
        dm::box3 cascadeBboxLS = dm::box3::empty();
        for (int i = 0; i < dm::box3::numCorners; i++) {
            dm::float3 cornerLS = viewToWorldToLight.transformPoint(cornersVS[i]);
            cascadeBboxLS |= cornerLS;
        }

        // squares bbox for uniform texels
        dm::float3 cascadeExtent = cascadeBboxLS.diagonal();
        float maxXY = dm::max(cascadeExtent.x, cascadeExtent.y);
        dm::float3 growVec = 0.5f * (dm::float3(maxXY, maxXY, cascadeExtent.z) - cascadeExtent);
        cascadeBboxLS = cascadeBboxLS.grow(growVec);
        cascadeExtent = cascadeBboxLS.diagonal();

        // clamps bbox to texel grid
        float texelSize = cascadeExtent.x / float(shadowRes);
        if (texelSize > 0.f) {
            cascadeBboxLS.m_mins.x = std::floorf(cascadeBboxLS.m_mins.x / texelSize) * texelSize;
            cascadeBboxLS.m_mins.y = std::floorf(cascadeBboxLS.m_mins.y / texelSize) * texelSize;
            cascadeBboxLS.m_maxs.x = cascadeBboxLS.m_mins.x + cascadeExtent.x;
            cascadeBboxLS.m_maxs.y = cascadeBboxLS.m_mins.y + cascadeExtent.y;
        }

        cascadeBboxLS.m_mins.z = dm::min(cascadeBboxLS.m_mins.z, sceneBboxLS.m_mins.z);
        cascadeBboxLS.m_maxs.z = dm::min(cascadeBboxLS.m_maxs.z, sceneBboxLS.m_maxs.z);
        dm::float4x4 lightProj = dm::orthoProjD3DStyle(
            cascadeBboxLS.m_mins.x, cascadeBboxLS.m_maxs.x,
            cascadeBboxLS.m_mins.y, cascadeBboxLS.m_maxs.y,
            cascadeBboxLS.m_mins.z, cascadeBboxLS.m_maxs.z);

        cascades[c].lightViewProj = dm::affineToHomogeneous(worldToLight) * lightProj;
        cascades[c].shadowCasterBboxLS = cascadeBboxLS;
        shadowCasterBboxLS |= cascadeBboxLS;
    }

    shadowCasterBboxLS &= sceneBboxLS;
}
