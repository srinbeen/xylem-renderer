#include "include/ViewHandler.hpp"

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

    dm::frustum camFrustum(
        dm::affineToHomogeneous(view.GetViewMatrix()) * finiteProj, false
    );

    for (int i = 0; i < dm::frustum::numCorners; i++) {
        dm::float3 clampedCornerLS = worldToLight.transformPoint(camFrustum.getCorner(i));
        clampedCornerLS = dm::max(clampedCornerLS, sceneBoundsLS.m_mins);
        clampedCornerLS = dm::min(clampedCornerLS, sceneBoundsLS.m_maxs);
        shadowCasterBboxLS |= clampedCornerLS;
    }
}
