#include "include/ViewHandler.hpp"

using namespace Xylem;

void ViewHandler::updateShadowVolume(const dm::box3& sceneBbox, dm::float3 sunDirection)
{
    worldToLight = dm::lookatZ(sunDirection) * dm::scaling(dm::float3(1.f, 1.f, -1.f));
    dm::box3 sceneBoundsLS = sceneBbox * worldToLight;

    const dm::float3& camPos = camera.GetPosition();
    float maxShadowDist = 0.f;
    for (int i = 0; i < dm::box3::numCorners; i++) {
        maxShadowDist = dm::max(
            maxShadowDist,
            dm::length(sceneBbox.getCorner(i) - camPos)
        );
    }

    const nvrhi::Viewport& vp = view.GetViewport();
    dm::float4x4 finiteProj = dm::perspProjD3DStyle(
        dm::radians(60.f), vp.width() / vp.height(), 0.1f, maxShadowDist);

    dm::frustum camFrustum(
        dm::affineToHomogeneous(view.GetViewMatrix()) * finiteProj, false);

    dm::box3 frustumLS = dm::box3::empty();
    for (int i = 0; i < dm::frustum::numCorners; i++)
        frustumLS |= worldToLight.transformPoint(camFrustum.getCorner(i));

    frustumLS.m_mins.z = sceneBoundsLS.m_mins.z;

    shadowCasterBboxLS = frustumLS & sceneBoundsLS;
}
