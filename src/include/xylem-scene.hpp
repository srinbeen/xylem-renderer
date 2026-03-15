#ifndef XYLEM_SCENE_H
#define XYLEM_SCENE_H

#include <string>
#include <vector>
#include <map>

#include <donut/core/math/math.h>

#include <nvrhi/nvrhi.h>

#include "xylem-procgen.hpp"
#include "xylem-render.hpp"

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

class RegionManager {
private:
    std::map<std::string, uint32_t> m_RegionIndexMap;
    std::vector<TreeRegion>         m_Regions;
    uint32_t                        m_TotalInstanceCount = 0;

public:
    inline TreeRegion& operator[](const std::string& s)       { return m_Regions[m_RegionIndexMap.at(s)]; }
    inline TreeRegion& operator[](uint32_t idx)               { return m_Regions[idx]; }
    inline const TreeRegion& operator[](uint32_t idx)   const { return m_Regions[idx]; }

    inline auto begin()       { return m_Regions.begin(); }
    inline auto end()         { return m_Regions.end(); }
    inline auto begin() const { return m_Regions.begin(); }
    inline auto end()   const { return m_Regions.end(); }

    inline size_t   size()                     const { return m_Regions.size(); }
    inline uint32_t getTotalInstanceCount()    const { return m_TotalInstanceCount; }

    void clear();
    void addRegion(const TreeRegion& r);
    void addRegion(const std::string& name, uint32_t instanceCount, const dm::box2& bounds);
    void updateRegion(size_t idx, const std::vector<TreeAsset>& assets);
};

} // namespace Scene
} // namespace Xylem

#endif // XYLEM_SCENE_H