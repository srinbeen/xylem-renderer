#include "include/xylem-hash.hpp"

uint32_t Xylem::hash(uint32_t x, uint32_t seed) {
    x ^= seed;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

float Xylem::hashToFloat(uint32_t x, uint32_t seed) {
    return (float)hash(x, seed) / (float)UINT32_MAX;
}