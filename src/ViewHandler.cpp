#include "include/ViewHandler.hpp"
#include <cmath>

using namespace Xylem;

void ViewHandler::computeCascades(const dm::box3& sceneBbox, dm::float3 sunDirection,
                                   float nearPlane, float farPlane, float aspectRatio, float fovY,
                                   uint32_t shadowRes, float pssmLambda)
{
    static_assert(Render::c_NumCascades >= 1 && Render::c_NumCascades <= 5,
        "cascadeSplitDistances stores up to four split thresholds");

    worldToLight = dm::lookatZ(sunDirection, dm::float3(0,1,0)) * dm::scaling(dm::float3(1.f, 1.f, -1.f));
    dm::box3 sceneBboxLS = sceneBbox * worldToLight;

    const float lambda = pssmLambda;
    float pssmNearPlane = nearPlane;
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

    cascadeSplitDistances = dm::float4(splits[N]);
    constexpr uint32_t kPackedSplitCount = (N < 4u) ? N : 4u;
    for (uint32_t i = 0; i < kPackedSplitCount; ++i)
        cascadeSplitDistances[i] = splits[i + 1];

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

        // cascade is not looking at anything
        if (!cascadeBboxLS.intersects(sceneBboxLS)) {
            cascades[c].lightViewProj = dm::float4x4::identity();
            cascades[c].shadowCasterBboxLS = dm::box3::empty();
            continue;
        }

        const dm::float3 centroidVS(0.f, 0.f, (cNear + cFar) * 0.5f);
        // 4 -- to a far corner
        const float radius = dm::length(cornersVS[4] - centroidVS);
        dm::float3 centroidLS = viewToWorldToLight.transformPoint(centroidVS);

        float zMinLS = dm::min(cascadeBboxLS.m_mins.z, sceneBboxLS.m_mins.z);
        float zMaxLS = dm::min(cascadeBboxLS.m_maxs.z, sceneBboxLS.m_maxs.z);

        const float T = (2.f * radius) / float(shadowRes);
        if (T > 0.f) {
            centroidLS.x = std::roundf(centroidLS.x / T) * T;
            centroidLS.y = std::roundf(centroidLS.y / T) * T;
        }

        dm::float4x4 lightProj = dm::orthoProjD3DStyle(
            centroidLS.x - radius, centroidLS.x + radius,
            centroidLS.y - radius, centroidLS.y + radius,
            zMinLS, zMaxLS);

        cascades[c].lightViewProj = dm::affineToHomogeneous(worldToLight) * lightProj;
        cascades[c].shadowCasterBboxLS = dm::box3(
            dm::float3(centroidLS.x - radius, centroidLS.y - radius, zMinLS),
            dm::float3(centroidLS.x + radius, centroidLS.y + radius, zMaxLS));
        shadowCasterBboxLS |= cascades[c].shadowCasterBboxLS;
    }

    shadowCasterBboxLS &= sceneBboxLS;
}
