#ifndef XYLEM_UI_DATA_H
#define XYLEM_UI_DATA_H

#include <cstdint>

namespace Xylem {

struct UIData {
    bool ShowUI = true;

    float    gpuFrameTimeMs       = -1.0f; // -1 = not yet available
    float    cpuRenderTimeMs      = 0.0f;
    uint32_t visibleInstanceCount = 0;
    uint32_t drawCallCount        = 0;
    uint32_t totalInstanceCount   = 0;
    uint32_t culledInstanceCount  = 0;
    uint32_t shadowVisibleCount   = 0;
    uint32_t shadowCulledCount    = 0;
};

} // namespace Xylem

#endif // XYLEM_UI_DATA_H
