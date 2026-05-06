#ifndef XYLEM_TERRAIN_H
#define XYLEM_TERRAIN_H

#include <array>
#include <string>
#include <vector>
#include <cstdint>
#include <donut/core/math/math.h>

namespace Xylem::Scene {

struct TerrainVertex {
    dm::float3 pos;
    dm::float3 normal;
    dm::float2 uv;
};

struct TerrainShadingParams {
    float tileSize       = 4.0f;
    float forestToDirtY  = 5.0f;
    float dirtToSnowY    = 25.0f;
    float bandWidth      = 8.0f;
    float slopeLo        = 0.35f;
    float slopeHi        = 0.65f;
    float macroNoiseAmp  = 3.0f;
    std::array<std::string, 4> textureSetNames; // [forestFloor, dirt, rock, snow]
};

struct TerrainConfig {
    float worldMinX  = -50.f;
    float worldMinZ  = -50.f;
    float worldMaxX  =  50.f;
    float worldMaxZ  =  50.f;
    float gridSpacing = 0.5f;

    uint32_t seed       = 42;
    int      octaves    = 4;
    float    frequency  = 0.05f;
    float    amplitude  = 8.0f;
    float    lacunarity = 2.0f;
    float    persistence = 0.5f;
    TerrainShadingParams shading;
};

class Terrain {
public:
    void generate(const TerrainConfig& config);

    float      getHeightAt(float worldX, float worldZ) const;
    dm::float3 getNormalAt(float worldX, float worldZ) const;

    const std::vector<TerrainVertex>& getVertices() const { return m_Vertices; }
    const std::vector<uint32_t>&      getIndices()  const { return m_Indices; }
    const dm::box3&                   getBbox()     const { return m_Bbox; }
    const TerrainConfig&              getConfig()   const { return m_Config; }

    uint32_t getGridWidth() const { return m_GridWidth; }
    uint32_t getGridDepth() const { return m_GridDepth; }

private:
    TerrainConfig m_Config;

    std::vector<float> m_Heights; // row-major: m_Heights[z * m_GridWidth + x]
    uint32_t m_GridWidth = 0;
    uint32_t m_GridDepth = 0;

    std::vector<TerrainVertex> m_Vertices;
    std::vector<uint32_t>      m_Indices;
    dm::box3                   m_Bbox;

    void _computeNormals();
    void _worldToGrid(float worldX, float worldZ, float& gx, float& gz) const;
};

} // namespace Xylem::Scene

#endif // XYLEM_TERRAIN_H
