#ifndef XYLEM_SCENE_H
#define XYLEM_SCENE_H

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <donut/core/math/math.h>

#include <nvrhi/nvrhi.h>

#include "Procgen.hpp"
#include "Render.hpp"
#include "Terrain.hpp"

namespace Xylem::Scene {

struct LSystemInstance {
    std::string     name;
    ProcGen::lgen_t gen;
};

// CPU-only LOD data — no GPU handles. Used by SceneRegistry.
struct TreeLODDef {
    std::vector<ProcGen::TreeVertex> vertices;
    std::vector<uint32_t>            indices;
    dm::box3                         bbox;
    uint32_t                         radialSegments = 0;
};

// GPU-side LOD data — owned by render passes.
struct TreeLODData {
    nvrhi::BufferHandle vertexBuffer;
    nvrhi::BufferHandle indexBuffer;
    uint32_t            indexCount;
    uint32_t            radialSegments;
    dm::box3            bbox;
};

struct TreeAsset {
    std::string                    name;
    LSystemInstance                lsystemInstance;
    ProcGen::TreeGenerator::Params generatorParams;
    ProcGen::lstring_t             lsystemString;
    std::vector<TreeLODData>       lods;
    std::string                    barkTexture;
    uint32_t                       textureSetIdx = 0;
};

struct TreeRegion {
    std::string                              name;
    float                                    density;
    uint32_t                                 instanceCount;
    dm::box2                                 bounds;
    std::vector<uint32_t>                    assetIndices;
    std::vector<Render::InstanceBufferEntry> instanceBuffer;
    std::vector<dm::box3>                    instanceBbox;
    dm::box3                                 cullBox;

    TreeRegion(const std::string& n, float d, const dm::box2& b)
        : name{n}, density{d}, bounds{b}, cullBox{dm::box3::empty()}
    {
        float area = (b.m_maxs.x - b.m_mins.x) * (b.m_maxs.y - b.m_mins.y);
        instanceCount = std::max(1u, static_cast<uint32_t>(std::round(density * area)));
        instanceBuffer.reserve(instanceCount);
        instanceBbox.reserve(instanceCount);
    }
};

} // namespace Xylem::Scene

#endif // XYLEM_SCENE_H