#ifndef XYLEM_NOISE_H
#define XYLEM_NOISE_H

#include <cmath>
#include <cstdint>
#include "hash.hpp"

namespace Xylem::Noise {

// Attempt to produce a unit gradient vector from a hash of integer grid coords + seed.
// Returns a 2D gradient by selecting from 8 evenly spaced unit vectors.
inline void grad2D(int ix, int iy, uint32_t seed, float& gx, float& gy) {
    uint32_t h = Xylem::hash(static_cast<uint32_t>(ix) * 73856093u ^ static_cast<uint32_t>(iy) * 19349663u, seed);
    // Use low 3 bits to pick one of 8 directions
    switch (h & 7) {
        case 0: gx =  1.f; gy =  0.f; break;
        case 1: gx = -1.f; gy =  0.f; break;
        case 2: gx =  0.f; gy =  1.f; break;
        case 3: gx =  0.f; gy = -1.f; break;
        case 4: gx =  0.707107f; gy =  0.707107f; break;
        case 5: gx = -0.707107f; gy =  0.707107f; break;
        case 6: gx =  0.707107f; gy = -0.707107f; break;
        case 7: gx = -0.707107f; gy = -0.707107f; break;
    }
}

inline float smoothstep(float t) {
    return t * t * t * (t * (t * 6.f - 15.f) + 10.f); // quintic (improved Perlin)
}

inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// Classic 2D Perlin noise, returns roughly [-1, 1]
inline float perlin2D(float x, float y, uint32_t seed = 0) {
    int x0 = (int)std::floor(x);
    int y0 = (int)std::floor(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float sx = x - (float)x0;
    float sy = y - (float)y0;

    float u = smoothstep(sx);
    float v = smoothstep(sy);

    float gx, gy;

    // Dot products at four corners
    grad2D(x0, y0, seed, gx, gy);
    float n00 = gx * sx + gy * sy;

    grad2D(x1, y0, seed, gx, gy);
    float n10 = gx * (sx - 1.f) + gy * sy;

    grad2D(x0, y1, seed, gx, gy);
    float n01 = gx * sx + gy * (sy - 1.f);

    grad2D(x1, y1, seed, gx, gy);
    float n11 = gx * (sx - 1.f) + gy * (sy - 1.f);

    float ix0 = lerp(n00, n10, u);
    float ix1 = lerp(n01, n11, u);
    return lerp(ix0, ix1, v);
}

// Fractional Brownian Motion — stacks multiple octaves of Perlin noise
inline float fbm2D(float x, float y, uint32_t seed = 0,
                   int octaves = 4, float lacunarity = 2.f, float persistence = 0.5f) {
    float value = 0.f;
    float amplitude = 1.f;
    float frequency = 1.f;
    float maxAmplitude = 0.f;

    for (int i = 0; i < octaves; i++) {
        value += amplitude * perlin2D(x * frequency, y * frequency, seed + static_cast<uint32_t>(i) * 31u);
        maxAmplitude += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    return value / maxAmplitude; // normalize to roughly [-1, 1]
}

// 2D Worley/Voronoi F1: distance from (x, y) to the nearest jittered feature
// point on the integer lattice. One feature point per unit cell, position
// jittered into the cell by a hash of its integer coords. Returns roughly
// [0, ~1.06] — ~0 at a feature point, larger near cell corners.
inline float worleyF1_2D(float x, float y, uint32_t seed = 0) {
    int ix = (int)std::floor(x);
    int iy = (int)std::floor(y);
    float fx = x - (float)ix;
    float fy = y - (float)iy;

    float minDistSq = 1e30f;
    for (int oy = -1; oy <= 1; oy++) {
        for (int ox = -1; ox <= 1; ox++) {
            int      cx = ix + ox;
            int      cy = iy + oy;
            uint32_t cellHash =
                static_cast<uint32_t>(cx) * 73856093u ^ static_cast<uint32_t>(cy) * 19349663u;
            float jx = Xylem::hashToFloat(cellHash, seed);
            float jy = Xylem::hashToFloat(cellHash, seed ^ 0xDEADBEEFu);
            float dx = (float)ox + jx - fx;
            float dy = (float)oy + jy - fy;
            float dSq = dx * dx + dy * dy;
            if (dSq < minDistSq) minDistSq = dSq;
        }
    }
    return std::sqrt(minDistSq);
}

} // namespace Xylem::Noise

#endif // XYLEM_NOISE_H
