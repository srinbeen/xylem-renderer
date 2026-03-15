#ifndef XYLEM_SCENE_H
#define XYLEM_SCENE_H

#include <string>
#include <vector>

#include <donut/core/math/math.h>

#include <nvrhi/nvrhi.h>

#include "Procgen.hpp"
#include "Render.hpp"

namespace Xylem {
namespace Scene {

struct LSystemInstance {
    std::string     name;
    ProcGen::lgen_t gen;
};

struct TreeLODData {
    nvrhi::BufferHandle vertexBuffer;
    nvrhi::BufferHandle indexBuffer;
    uint32_t            indexCount;
    uint32_t            radialSegments;
    donut::math::box3   bbox;
};

struct TreeAsset {
    std::string                    name;
    LSystemInstance                lsystemInstance;
    ProcGen::TreeGenerator::Params generatorParams;
    ProcGen::lstring_t             lsystemString;
    std::vector<TreeLODData>       lods;
};

struct TreeRegion {
    std::string                              name;
    uint32_t                                 instanceCount;
    dm::box2                                 bounds;
    std::vector<uint32_t>                    assetIndices;
    std::vector<Render::InstanceBufferEntry> instanceBuffer;
    std::vector<dm::box3>                    instanceBbox;
    dm::box3                                 cullBox;

    TreeRegion(const std::string& n, uint32_t c, const dm::box2& b)
        : name{n}, instanceCount{c}, bounds{b}, cullBox{dm::box3::empty()}
    {
        instanceBuffer.reserve(instanceCount);
        instanceBbox.reserve(instanceCount);
    }
};

} // namespace Scene
} // namespace Xylem

#endif // XYLEM_SCENE_H