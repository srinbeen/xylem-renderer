#include "include/Terrain.hpp"
#include "include/Noise.hpp"

#include <algorithm>
#include <cmath>

using namespace Xylem::Scene;

void Terrain::generate(const TerrainConfig& config) {
    m_Config = config;

    m_GridWidth = static_cast<uint32_t>(std::floor((config.worldMaxX - config.worldMinX) / config.gridSpacing)) + 1;
    m_GridDepth = static_cast<uint32_t>(std::floor((config.worldMaxZ - config.worldMinZ) / config.gridSpacing)) + 1;

    const uint32_t vertexCount = m_GridWidth * m_GridDepth;
    m_Heights.resize(vertexCount);
    m_Vertices.resize(vertexCount);

    // Generate heightmap
    float minY =  FLT_MAX;
    float maxY = -FLT_MAX;

    for (uint32_t iz = 0; iz < m_GridDepth; iz++) {
        for (uint32_t ix = 0; ix < m_GridWidth; ix++) {
            float worldX = config.worldMinX + ix * config.gridSpacing;
            float worldZ = config.worldMinZ + iz * config.gridSpacing;

            float h = config.amplitude * Noise::fbm2D(
                worldX * config.frequency,
                worldZ * config.frequency,
                config.seed,
                config.octaves,
                config.lacunarity,
                config.persistence);

            uint32_t idx = iz * m_GridWidth + ix;
            m_Heights[idx] = h;

            float u = static_cast<float>(ix) / static_cast<float>(m_GridWidth - 1);
            float v = static_cast<float>(iz) / static_cast<float>(m_GridDepth - 1);

            m_Vertices[idx].pos    = dm::float3(worldX, h, worldZ);
            m_Vertices[idx].normal = dm::float3(0.f, 1.f, 0.f); // placeholder
            m_Vertices[idx].uv     = dm::float2(u, v);

            minY = std::min(minY, h);
            maxY = std::max(maxY, h);
        }
    }

    _computeNormals();

    // Build index buffer: two triangles per quad
    uint32_t quadsX = m_GridWidth - 1;
    uint32_t quadsZ = m_GridDepth - 1;
    m_Indices.resize(quadsX * quadsZ * 6);
    uint32_t idx = 0;

    for (uint32_t iz = 0; iz < quadsZ; iz++) {
        for (uint32_t ix = 0; ix < quadsX; ix++) {
            uint32_t topLeft     = iz * m_GridWidth + ix;
            uint32_t topRight    = topLeft + 1;
            uint32_t bottomLeft  = topLeft + m_GridWidth;
            uint32_t bottomRight = bottomLeft + 1;

            // Triangle 1
            m_Indices[idx++] = topLeft;
            m_Indices[idx++] = bottomLeft;
            m_Indices[idx++] = topRight;

            // Triangle 2
            m_Indices[idx++] = topRight;
            m_Indices[idx++] = bottomLeft;
            m_Indices[idx++] = bottomRight;
        }
    }

    m_Bbox = dm::box3(
        dm::float3(config.worldMinX, minY, config.worldMinZ),
        dm::float3(config.worldMaxX, maxY, config.worldMaxZ));

    _buildMeshlets();
}

void Terrain::_buildMeshlets() {
    m_Meshlets = {};
    if (m_GridWidth < 2 || m_GridDepth < 2) return;

    const uint32_t Q = k_TerrainMeshletQuadsPerSide;
    const uint32_t quadsX = m_GridWidth - 1;
    const uint32_t quadsZ = m_GridDepth - 1;
    const uint32_t tilesX = (quadsX + Q - 1) / Q;
    const uint32_t tilesZ = (quadsZ + Q - 1) / Q;

    m_Meshlets.descs.reserve(static_cast<size_t>(tilesX) * tilesZ);
    m_Meshlets.localVertIndices.reserve(static_cast<size_t>(tilesX) * tilesZ * (Q + 1) * (Q + 1));
    m_Meshlets.localTriIndices.reserve(static_cast<size_t>(tilesX) * tilesZ * Q * Q * 6);

    for (uint32_t tz = 0; tz < tilesZ; tz++) {
        const uint32_t z0     = tz * Q;
        const uint32_t z1     = std::min(z0 + Q, quadsZ);
        const uint32_t tileQz = z1 - z0;

        for (uint32_t tx = 0; tx < tilesX; tx++) {
            const uint32_t x0     = tx * Q;
            const uint32_t x1     = std::min(x0 + Q, quadsX);
            const uint32_t tileQx = x1 - x0;

            const uint32_t vsX = tileQx + 1;
            const uint32_t vsZ = tileQz + 1;

            TerrainMeshletDesc d{};
            d.vertOffset = static_cast<uint32_t>(m_Meshlets.localVertIndices.size());
            d.triOffset  = static_cast<uint32_t>(m_Meshlets.localTriIndices.size());
            d.vertCount  = vsX * vsZ;
            d.triCount   = tileQx * tileQz * 2;

            float minY =  FLT_MAX;
            float maxY = -FLT_MAX;
            for (uint32_t lz = 0; lz < vsZ; lz++) {
                for (uint32_t lx = 0; lx < vsX; lx++) {
                    const uint32_t gx = x0 + lx;
                    const uint32_t gz = z0 + lz;
                    const uint32_t globalIdx = gz * m_GridWidth + gx;
                    m_Meshlets.localVertIndices.push_back(globalIdx);
                    const float h = m_Heights[globalIdx];
                    minY = std::min(minY, h);
                    maxY = std::max(maxY, h);
                }
            }

            for (uint32_t qz = 0; qz < tileQz; qz++) {
                for (uint32_t qx = 0; qx < tileQx; qx++) {
                    const uint8_t tl = static_cast<uint8_t>(qz       * vsX + qx);
                    const uint8_t tr = static_cast<uint8_t>(tl + 1);
                    const uint8_t bl = static_cast<uint8_t>((qz + 1) * vsX + qx);
                    const uint8_t br = static_cast<uint8_t>(bl + 1);
                    m_Meshlets.localTriIndices.push_back(tl);
                    m_Meshlets.localTriIndices.push_back(bl);
                    m_Meshlets.localTriIndices.push_back(tr);
                    m_Meshlets.localTriIndices.push_back(tr);
                    m_Meshlets.localTriIndices.push_back(bl);
                    m_Meshlets.localTriIndices.push_back(br);
                }
            }

            const float wMinX = m_Config.worldMinX + x0 * m_Config.gridSpacing;
            const float wMinZ = m_Config.worldMinZ + z0 * m_Config.gridSpacing;
            const float wMaxX = m_Config.worldMinX + (x0 + tileQx) * m_Config.gridSpacing;
            const float wMaxZ = m_Config.worldMinZ + (z0 + tileQz) * m_Config.gridSpacing;
            d.aabbMin = dm::float3(wMinX, minY, wMinZ);
            d.aabbMax = dm::float3(wMaxX, maxY, wMaxZ);

            m_Meshlets.descs.push_back(d);
        }
    }

    // Pad packed tri stream so the shader's 4-byte-aligned ByteAddressBuffer
    // load doesn't run off the end when reading the last triangle.
    m_Meshlets.localTriIndices.insert(m_Meshlets.localTriIndices.end(), 4, 0);
}

void Terrain::_computeNormals() {
    float s = m_Config.gridSpacing;

    for (uint32_t iz = 0; iz < m_GridDepth; iz++) {
        for (uint32_t ix = 0; ix < m_GridWidth; ix++) {
            uint32_t ixL = (ix > 0) ? ix - 1 : ix;
            uint32_t ixR = (ix < m_GridWidth - 1) ? ix + 1 : ix;
            uint32_t izD = (iz > 0) ? iz - 1 : iz;
            uint32_t izU = (iz < m_GridDepth - 1) ? iz + 1 : iz;

            float hL = m_Heights[iz  * m_GridWidth + ixL];
            float hR = m_Heights[iz  * m_GridWidth + ixR];
            float hD = m_Heights[izD * m_GridWidth + ix];
            float hU = m_Heights[izU * m_GridWidth + ix];

            float dx = (ixR - ixL) * s;
            float dz = (izU - izD) * s;
            if (dx < 1e-6f) dx = s;
            if (dz < 1e-6f) dz = s;

            dm::float3 n = dm::normalize(dm::float3(
                (hL - hR) / dx * s,
                2.0f * s,
                (hD - hU) / dz * s));

            m_Vertices[iz * m_GridWidth + ix].normal = n;
        }
    }
}

void Terrain::_worldToGrid(float worldX, float worldZ, float& gx, float& gz) const {
    gx = (worldX - m_Config.worldMinX) / m_Config.gridSpacing;
    gz = (worldZ - m_Config.worldMinZ) / m_Config.gridSpacing;
    gx = std::clamp(gx, 0.f, static_cast<float>(m_GridWidth - 1));
    gz = std::clamp(gz, 0.f, static_cast<float>(m_GridDepth - 1));
}

float Terrain::getHeightAt(float worldX, float worldZ) const {
    if (m_Heights.empty()) return 0.f;

    float gx, gz;
    _worldToGrid(worldX, worldZ, gx, gz);

    uint32_t ix0 = static_cast<uint32_t>(gx);
    uint32_t iz0 = static_cast<uint32_t>(gz);
    uint32_t ix1 = std::min(ix0 + 1, m_GridWidth - 1);
    uint32_t iz1 = std::min(iz0 + 1, m_GridDepth - 1);

    float fx = gx - static_cast<float>(ix0);
    float fz = gz - static_cast<float>(iz0);

    float h00 = m_Heights[iz0 * m_GridWidth + ix0];
    float h10 = m_Heights[iz0 * m_GridWidth + ix1];
    float h01 = m_Heights[iz1 * m_GridWidth + ix0];
    float h11 = m_Heights[iz1 * m_GridWidth + ix1];

    float h0 = h00 + fx * (h10 - h00);
    float h1 = h01 + fx * (h11 - h01);
    return h0 + fz * (h1 - h0);
}

dm::float3 Terrain::getNormalAt(float worldX, float worldZ) const {
    if (m_Vertices.empty()) return dm::float3(0.f, 1.f, 0.f);

    float gx, gz;
    _worldToGrid(worldX, worldZ, gx, gz);

    uint32_t ix0 = static_cast<uint32_t>(gx);
    uint32_t iz0 = static_cast<uint32_t>(gz);
    uint32_t ix1 = std::min(ix0 + 1, m_GridWidth - 1);
    uint32_t iz1 = std::min(iz0 + 1, m_GridDepth - 1);

    float fx = gx - static_cast<float>(ix0);
    float fz = gz - static_cast<float>(iz0);

    dm::float3 n00 = m_Vertices[iz0 * m_GridWidth + ix0].normal;
    dm::float3 n10 = m_Vertices[iz0 * m_GridWidth + ix1].normal;
    dm::float3 n01 = m_Vertices[iz1 * m_GridWidth + ix0].normal;
    dm::float3 n11 = m_Vertices[iz1 * m_GridWidth + ix1].normal;

    dm::float3 n0 = n00 + fx * (n10 - n00);
    dm::float3 n1 = n01 + fx * (n11 - n01);
    return dm::normalize(n0 + fz * (n1 - n0));
}
