#include "include/RegionManager.hpp"
#include "include/hash.hpp"
#include "include/Terrain.hpp"

using namespace Xylem;
using namespace Xylem::Scene;

void RegionManager::clear() {
    m_Regions.clear();
    m_RegionIndexMap.clear();
    m_TotalInstanceCount = 0;
}

void RegionManager::addRegion(const TreeRegion& r) {
    if (m_RegionIndexMap.count(r.name)) return;
    m_TotalInstanceCount += r.instanceCount;
    m_RegionIndexMap[r.name] = static_cast<uint32_t>(m_Regions.size());
    m_Regions.push_back(r);
}

void RegionManager::addRegion(const std::string& name, float density, const dm::box2& bounds) {
    addRegion(TreeRegion(name, density, bounds));
}

void RegionManager::updateRegion(size_t idx, const std::vector<TreeAsset>& assets, const Terrain* terrain) {
    if (idx >= m_Regions.size()) return;
    auto& region = m_Regions[idx];

    region.instanceBuffer.clear();
    region.instanceBbox.clear();
    region.cullBox = dm::box3::empty();

    if (region.assetIndices.empty() || assets.empty()) return;

    region.instanceBuffer.reserve(region.instanceCount);
    region.instanceBbox.reserve(region.instanceCount);

    const uint32_t seedPos = 0xDEADBEEF ^ static_cast<uint32_t>(idx);
    const uint32_t seedRot = 0xFEEDBEEF ^ static_cast<uint32_t>(idx);
    const float rangeX = region.bounds.m_maxs.x - region.bounds.m_mins.x;
    const float rangeZ = region.bounds.m_maxs.y - region.bounds.m_mins.y;

    for (uint32_t i = 0; i < region.instanceCount; i++) {
        float posX = region.bounds.m_mins.x + hashToFloat(i * 2u,     seedPos) * rangeX;
        float posZ = region.bounds.m_mins.y + hashToFloat(i * 2u + 1, seedPos) * rangeZ;
        float rotY = hashToFloat(i, seedRot) * dm::PI_f;

        uint32_t assetIdx = region.assetIndices[i % region.assetIndices.size()];
        if (assetIdx >= assets.size()) continue;

        float posY = terrain ? terrain->getHeightAt(posX, posZ) : 0.f;
        dm::affine3 worldMatrix = dm::rotation(dm::float3(1.f, 0.f, 0.f), -dm::PI_f/2.0f)
            * dm::rotation(dm::float3(0.f, 1.f, 0.f), rotY)
            * dm::translation(dm::float3(posX, posY, posZ));
        dm::float3x3 normalMatrix;
        if (dm::isnear(worldMatrix.m_linear[0][0], worldMatrix.m_linear[1][1]) && 
            dm::isnear(worldMatrix.m_linear[1][1], worldMatrix.m_linear[2][2])) {
                normalMatrix = worldMatrix.m_linear;
        }
        else {
            normalMatrix = dm::transpose(dm::inverse(worldMatrix.m_linear));
        }
        

        region.instanceBuffer.emplace_back(worldMatrix, normalMatrix, assetIdx);

        const dm::box3& assetBox = assets[assetIdx].lods[0].bbox;
        dm::box3 wTreeBbox = assetBox * worldMatrix;
        region.instanceBbox.push_back(wTreeBbox);

        region.cullBox |= wTreeBbox;
    }
}