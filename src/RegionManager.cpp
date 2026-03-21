#include "include/RegionManager.hpp"
#include "include/hash.hpp"
#include "include/Terrain.hpp"

#include <cmath>
#include <algorithm>

using namespace Xylem;
using namespace Xylem::Scene;

// ---------------------------------------------------------------------------
// Grid-accelerated relaxation helpers
// ---------------------------------------------------------------------------
namespace {

constexpr uint32_t kRelaxIterations = 15;

struct TreePlacement {
    dm::float2 pos;
    float      radius;
    uint32_t   assetIdx;
    float      rotY;
};

struct SpatialGrid {
    std::vector<std::vector<uint32_t>> m_Cells;
    dm::uint2                          m_GridDim;
    float                              m_CellSize;
    dm::float2                         m_Origin;

    SpatialGrid(dm::uint2 gridDim, float cs, dm::float2 origin)
        :   m_GridDim{gridDim},
            m_CellSize{cs},
            m_Origin{origin} 
    {
        m_Cells.resize(gridDim.x*gridDim.y);
    }

    void clear() { for (auto& c : m_Cells) c.clear(); }

    inline dm::uint2 getCell(const dm::float2 pos) const {
        dm::uint2 cell = static_cast<dm::uint2>((pos - m_Origin) / m_CellSize);
        return dm::clamp(cell, {0,0}, m_GridDim - dm::uint2{1,1});
    }

    inline uint32_t flatten(dm::uint2 cell) const { return cell.y * m_GridDim.x + cell.x; }

    void insert(uint32_t treeIdx, dm::float2 pos) {
        m_Cells[flatten(getCell(pos))].push_back(treeIdx);
    }
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// RegionManager
// ---------------------------------------------------------------------------
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

    const uint32_t seedPos = 0xDEADBEEF ^ static_cast<uint32_t>(idx);
    const uint32_t seedRot = 0xFEEDBEEF ^ static_cast<uint32_t>(idx);

    const dm::float2 range = region.bounds.diagonal();
    const dm::affine3 xRotOnly = dm::rotation(dm::float3(1.f, 0.f, 0.f), -dm::PI_f / 2.0f);
    
    // used to create cell size
    float maxRadius = 0.f;
    for (uint32_t ai : region.assetIndices) {
        dm::box3   rotatedBox = assets[ai].lods[0].bbox * xRotOnly;
        dm::float3 diag       = rotatedBox.diagonal();
        float r = dm::max(diag.x, diag.z) * 0.5f;
        maxRadius = dm::max(maxRadius, r);
    }


    std::vector<TreePlacement> placements(region.instanceCount);

    for (uint32_t i = 0; i < region.instanceCount; i++) {
        float posX = region.bounds.m_mins.x + hashToFloat(i * 2u,     seedPos) * range.x;
        float posZ = region.bounds.m_mins.y + hashToFloat(i * 2u + 1, seedPos) * range.y;
        float rotY = hashToFloat(i, seedRot) * dm::PI_f;

        uint32_t assetIdx = region.assetIndices[i % region.assetIndices.size()];
        float radius = dm::length(assets[assetIdx].lods[0].bbox.diagonal()) * 0.5f;

        placements[i] = { {posX, posZ}, radius, assetIdx, rotY };
    }

    // makes sure that each cell has only one tree
    float cellSize = maxRadius * 2.f;
    if (cellSize < 0.001f) cellSize = 1.f;

    uint32_t gridW = static_cast<uint32_t>(std::ceil(range.x / cellSize));
    uint32_t gridH = static_cast<uint32_t>(std::ceil(range.y / cellSize));

    SpatialGrid grid{{gridW, gridH}, cellSize, region.bounds.m_mins};

    // -- Phase C: relaxation loop -----------------------------------------------
    std::vector<dm::float2> displacements(region.instanceCount);

    for (uint32_t iter = 0; iter < kRelaxIterations; iter++) {
        grid.clear();
        for (uint32_t i = 0; i < region.instanceCount; i++)
            grid.insert(i, placements[i].pos);

        for (auto& d : displacements) d = { 0.f, 0.f };

        for (uint32_t cz = 0; cz < gridH; cz++) {
            for (uint32_t cx = 0; cx < gridW; cx++) {
                const auto& cell = grid.m_Cells[grid.flatten({cx, cz})];

                for (int dz = -1; dz <= 1; dz++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = static_cast<int>(cx) + dx;
                        int nz = static_cast<int>(cz) + dz;
                        if (nx < 0 || 
                            nz < 0 ||
                            nx >= static_cast<int>(gridW) ||
                            nz >= static_cast<int>(gridH))
                            continue;

                        const auto& neighbor = grid.m_Cells[grid.flatten(dm::uint2(nx, nz))];

                        for (uint32_t a = 0; a < cell.size(); a++) {
                            uint32_t tI = cell[a];
                            for (uint32_t b = 0; b < neighbor.size(); b++) {
                                uint32_t tJ = neighbor[b];
                                if (tJ <= tI) continue; // skip self & double-count

                                float deltaX = placements[tJ].pos.x - placements[tI].pos.x;
                                float deltaZ = placements[tJ].pos.y - placements[tI].pos.y;
                                float distSq = deltaX * deltaX + deltaZ * deltaZ;
                                float minDist = placements[tI].radius + placements[tJ].radius;

                                if (distSq >= minDist * minDist) continue;

                                float dist    = std::sqrt(distSq);
                                float overlap = minDist - dist;

                                float dirX, dirZ;
                                if (dist < 1e-5f) {
                                    // Deterministic fallback for coincident trees
                                    float angle = hashToFloat(tI ^ tJ, 0xBAADF00Du) * 2.f * dm::PI_f;
                                    dirX = std::cos(angle);
                                    dirZ = std::sin(angle);
                                } else {
                                    dirX = deltaX / dist;
                                    dirZ = deltaZ / dist;
                                }

                                float push = overlap * 0.5f;
                                
                                displacements[tI].x  -= dirX * push;
                                displacements[tI].y -= dirZ * push;
                                
                                displacements[tJ].x  += dirX * push;
                                displacements[tJ].y += dirZ * push;
                            }
                        }
                    }
                }
            }
        }

        // Apply displacements, clamped to region bounds
        for (uint32_t i = 0; i < region.instanceCount; i++) {
            placements[i].pos = dm::float2(
                std::clamp(
                    placements[i].pos.x + displacements[i].x,
                    region.bounds.m_mins.x, 
                    region.bounds.m_maxs.x
                ),
                std::clamp(
                    placements[i].pos.y + displacements[i].y,
                    region.bounds.m_mins.y, 
                    region.bounds.m_maxs.y
                )
            );
        }
    }

    // -- Phase D: rebuild instance buffers from relaxed positions ----------------
    region.instanceBuffer.reserve(region.instanceCount);
    region.instanceBbox.reserve(region.instanceCount);

    for (uint32_t i = 0; i < region.instanceCount; i++) {
        const auto& p = placements[i];
        if (p.assetIdx >= assets.size()) continue;

        float posY = terrain ? terrain->getHeightAt(p.pos.x, p.pos.y) : 0.f;

        dm::affine3 worldMatrix =
            dm::rotation(dm::float3(1.f, 0.f, 0.f), -dm::PI_f / 2.0f)
            * dm::rotation(dm::float3(0.f, 1.f, 0.f), p.rotY)
            * dm::translation(dm::float3(p.pos.x, posY - 0.5f, p.pos.y));

        dm::float3x3 normalMatrix;
        if (dm::isnear(worldMatrix.m_linear[0][0], worldMatrix.m_linear[1][1]) &&
            dm::isnear(worldMatrix.m_linear[1][1], worldMatrix.m_linear[2][2])) {
            normalMatrix = worldMatrix.m_linear;
        } else {
            normalMatrix = dm::transpose(dm::inverse(worldMatrix.m_linear));
        }

        region.instanceBuffer.emplace_back(worldMatrix, normalMatrix, p.assetIdx);

        const dm::box3& assetBox = assets[p.assetIdx].lods[0].bbox;
        dm::box3 wTreeBbox = assetBox * worldMatrix;
        region.instanceBbox.push_back(wTreeBbox);

        region.cullBox |= wTreeBbox;
    }
}
