#ifndef XYLEM_REGION_MANAGER_H
#define XYLEM_REGION_MANAGER_H

#include <map>
#include <string>
#include <vector>

#include "Scene.hpp"

namespace Xylem {
namespace Scene {

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

    inline size_t   size()                  const { return m_Regions.size(); }
    inline uint32_t getTotalInstanceCount() const { return m_TotalInstanceCount; }

    void clear();
    void addRegion(const TreeRegion& r);
    void addRegion(const std::string& name, uint32_t instanceCount, const dm::box2& bounds);
    void updateRegion(size_t idx, const std::vector<TreeAsset>& assets);
};

} // namespace Scene
} // namespace Xylem

#endif // XYLEM_REGION_MANAGER_H
