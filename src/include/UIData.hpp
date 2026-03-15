#ifndef XYLEM_UIDATA_H
#define XYLEM_UIDATA_H

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
    };

}
#endif // XYLEM_UIDATA_H