#ifndef XYLEM_HASH_H
#define XYLEM_HASH_H

#include <cstdint>

namespace Xylem {

    inline uint32_t hash(uint32_t x, uint32_t seed) {
        x ^= seed;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        return x;
    }

    inline float hashToFloat(uint32_t x, uint32_t seed) {
        return (float)hash(x, seed) / (float)UINT32_MAX;
    }

}

#endif // XYLEM_HASH_H