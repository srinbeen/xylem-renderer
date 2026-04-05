#ifndef XYLEM_VIEW_HANDLER_H
#define XYLEM_VIEW_HANDLER_H

#include <donut/app/Camera.h>
#include <donut/engine/View.h>
#include <nvrhi/nvrhi.h>
#include <vector>

namespace Xylem {

using namespace donut;

struct ViewHandler {
    app::FirstPersonCamera camera;
    engine::PlanarView     view;
    dm::affine3            worldToLight;
    dm::box3               shadowCasterBboxLS;

    void updateShadowVolume(const dm::box3& sceneBbox, dm::float3 sunDirection);

    uint32_t distToLOD(float value, const std::vector<float>& arr) {
        auto it = std::lower_bound(arr.begin(), arr.end(), value);
        if (it == arr.end()) return static_cast<uint32_t>(arr.size() - 1);
        return static_cast<uint32_t>(std::distance(arr.begin(), it));
    }
};

} // namespace Xylem

#endif // XYLEM_VIEW_HANDLER_H
