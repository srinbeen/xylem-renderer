#ifndef XYLEM_SHADOW_CASCADE_COMMON_HLSLI
#define XYLEM_SHADOW_CASCADE_COMMON_HLSLI

#include "../include/macros.h"

#if XYLEM_NUM_CASCADES < 1 || XYLEM_NUM_CASCADES > 5
#error XYLEM_NUM_CASCADES must be in [1, 5]; cascadeSplits stores up to four split thresholds.
#endif

uint SelectShadowCascade(float viewZ, float4 cascadeSplits)
{
    uint cascadeIdx = XYLEM_NUM_CASCADES - 1;

    [unroll]
    for (uint c = 0; c + 1 < XYLEM_NUM_CASCADES; ++c)
    {
        if (viewZ < cascadeSplits[c])
        {
            cascadeIdx = c;
            break;
        }
    }

    return cascadeIdx;
}

#endif // XYLEM_SHADOW_CASCADE_COMMON_HLSLI
