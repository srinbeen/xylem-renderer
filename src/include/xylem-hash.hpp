#ifndef XYLEM_HASH_H
#define XYLEM_HASH_H

#include <cstdint>

namespace Xylem {

    uint32_t    hash(uint32_t x, uint32_t seed);
    float       hashToFloat(uint32_t x, uint32_t seed);

}

#endif // XYLEM_HASH_H