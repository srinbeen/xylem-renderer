#include "include/SceneRegistry.hpp"
#include "include/hash.hpp"
#include "include/Noise.hpp"

#include <algorithm>
#include <cmath>

using namespace Xylem;

// ===========================================================================
// Grid-accelerated relaxation helpers (moved from RegionManager.cpp)
// ===========================================================================
namespace {

constexpr uint32_t kRelaxIterations = 50;

struct TreePlacement {
    dm::float2 pos;
    float      radius;
    size_t     assetId;
    float      rotY;
};

struct SpatialGrid {
    std::vector<std::vector<uint32_t>> m_Cells;
    dm::uint2                          m_GridDim;
    float                              m_CellSize;
    dm::float2                         m_Origin;

    SpatialGrid(dm::uint2 gridDim, float cs, dm::float2 origin)
        : m_GridDim{gridDim}, m_CellSize{cs}, m_Origin{origin}
    {
        m_Cells.resize(gridDim.x * gridDim.y);
    }

    void clear() { for (auto& c : m_Cells) c.clear(); }

    dm::uint2 getCell(const dm::float2 pos) const {
        dm::uint2 cell = static_cast<dm::uint2>((pos - m_Origin) / m_CellSize);
        return dm::clamp(cell, {0, 0}, m_GridDim - dm::uint2{1, 1});
    }

    uint32_t flatten(dm::uint2 cell) const { return cell.y * m_GridDim.x + cell.x; }

    void insert(uint32_t treeIdx, dm::float2 pos) {
        m_Cells[flatten(getCell(pos))].push_back(treeIdx);
    }
};

} // anonymous namespace

// ===========================================================================
// Read access
// ===========================================================================

uint32_t SceneRegistry::totalInstanceCount() const {
    uint32_t total = 0;
    for (const auto& r : m_Regions)
        total += r.instanceCount;
    return total;
}

TreeAssetDef* SceneRegistry::findAsset(size_t id) {
    auto it = m_AssetIdToIndex.find(id);
    return it != m_AssetIdToIndex.end() ? &m_Assets[it->second] : nullptr;
}

const TreeAssetDef* SceneRegistry::findAsset(size_t id) const {
    auto it = m_AssetIdToIndex.find(id);
    return it != m_AssetIdToIndex.end() ? &m_Assets[it->second] : nullptr;
}

size_t SceneRegistry::assetIndexById(size_t id) const {
    auto it = m_AssetIdToIndex.find(id);
    return it != m_AssetIdToIndex.end() ? it->second : SIZE_MAX;
}

// ===========================================================================
// Mutation
// ===========================================================================

void SceneRegistry::addLSystem(const std::string& name, std::unique_ptr<ProcGen::LSystem> ls) {
    m_LSystems[name] = std::move(ls);
}

size_t SceneRegistry::addAsset(
    const std::string& name,
    const Scene::LSystemInstance& lsInstance,
    const ProcGen::TreeGenerator::Params& params,
    const std::string& barkTexture)
{
    TreeAssetDef def;
    def.id              = m_NextAssetId++;
    def.name            = name;
    def.lsystemInstance = lsInstance;
    def.genParams       = params;
    def.barkTexture     = barkTexture;
    def.textureSetIdx   = registerBarkTexture(barkTexture);
    def.dirty           = true;

    m_AssetIdToIndex[def.id] = m_Assets.size();
    m_Assets.push_back(std::move(def));
    return m_Assets.back().id;
}

void SceneRegistry::setAssetVisible(size_t id, bool visible) {
    auto* asset = findAsset(id);
    if (asset) asset->visible = visible;
}

void SceneRegistry::modifyAsset(size_t id, const ProcGen::TreeGenerator::Params& params) {
    auto* asset = findAsset(id);
    if (!asset) return;
    asset->genParams = params;
    asset->dirty = true;

    // Bounding boxes are recomputed during asset rebuild, so any region
    // containing this asset must also be rebuilt to update inst.bbox / cullBox.
    for (auto& region : m_Regions) {
        if (std::find(region.assetIds.begin(), region.assetIds.end(), id) != region.assetIds.end())
            region.dirty = true;
    }
}

void SceneRegistry::modifyAssetExtended(size_t                                id,
                                        const ProcGen::TreeGenerator::Params& genParams,
                                        const ProcGen::SCParams&              colonization,
                                        const LeafParams&                     leaf,
                                        bool                                  hasLeaves) {
    auto* asset = findAsset(id);
    if (!asset) return;
    asset->genParams    = genParams;
    asset->colonization = colonization;
    asset->leaf         = leaf;
    asset->hasLeaves    = hasLeaves;
    asset->dirty        = true;

    for (auto& region : m_Regions) {
        if (std::find(region.assetIds.begin(), region.assetIds.end(), id) != region.assetIds.end())
            region.dirty = true;
    }
}

void SceneRegistry::removeAsset(size_t id) {
    auto it = m_AssetIdToIndex.find(id);
    if (it == m_AssetIdToIndex.end()) return;

    size_t idx = it->second;
    m_Assets.erase(m_Assets.begin() + idx);
    _refreshAssetIdMap();

    // Remove this asset ID from all regions and mark them dirty.
    for (auto& region : m_Regions) {
        auto& ids = region.assetIds;
        auto removeIt = std::remove(ids.begin(), ids.end(), id);
        if (removeIt != ids.end()) {
            ids.erase(removeIt, ids.end());
            region.dirty = true;
        }
    }

    // Mark all remaining assets dirty so onAssetsDirty fires and detects the count change.
    for (auto& a : m_Assets) a.dirty = true;
    // Affected regions are already marked dirty above; mark the rest too so the
    // gapped buffer layout (which spans all regions) gets fully rebuilt.
    for (auto& r : m_Regions) r.dirty = true;
}

void SceneRegistry::addRegion(const std::string& name, float density, const dm::box2& bounds,
                              const std::vector<size_t>& assetIds) {
    RegionDef def(name, density, bounds);
    def.assetIds = assetIds;
    def.dirty    = true;
    m_Regions.push_back(std::move(def));
}

void SceneRegistry::modifyRegion(size_t idx, float density, const dm::box2& bounds) {
    if (idx >= m_Regions.size()) return;
    auto& r = m_Regions[idx];
    r.density = density;
    r.bounds  = bounds;
    float area = (bounds.m_maxs.x - bounds.m_mins.x) * (bounds.m_maxs.y - bounds.m_mins.y);
    r.instanceCount = std::max(1u, static_cast<uint32_t>(std::round(density * area)));
    r.dirty = true;
}

void SceneRegistry::modifyRegionAssets(size_t idx, const std::vector<size_t>& assetIds) {
    if (idx >= m_Regions.size()) return;
    m_Regions[idx].assetIds = assetIds;
    m_Regions[idx].dirty    = true;
}

void SceneRegistry::setRegionAssetVisible(size_t regionIdx, size_t assetId, bool visible) {
    if (regionIdx >= m_Regions.size()) return;
    m_Regions[regionIdx].assetVisible[assetId] = visible;
}

void SceneRegistry::removeRegion(size_t idx) {
    if (idx >= m_Regions.size()) return;
    m_Regions.erase(m_Regions.begin() + idx);
    // Mark all remaining regions dirty so onRegionsDirty fires and rebuilds
    // the gapped buffer layout (offsets shift after removal).
    for (auto& r : m_Regions) r.dirty = true;
}

// ===========================================================================
// Dirty flags
// ===========================================================================

bool SceneRegistry::anyAssetsDirty() const {
    return std::any_of(m_Assets.begin(), m_Assets.end(),
        [](const TreeAssetDef& a) { return a.dirty; });
}

bool SceneRegistry::anyRegionsDirty() const {
    return std::any_of(m_Regions.begin(), m_Regions.end(),
        [](const RegionDef& r) { return r.dirty; });
}

std::vector<size_t> SceneRegistry::getDirtyAssetIndices() const {
    std::vector<size_t> result;
    for (size_t i = 0; i < m_Assets.size(); ++i)
        if (m_Assets[i].dirty) result.push_back(i);
    return result;
}

std::vector<size_t> SceneRegistry::getDirtyRegionIndices() const {
    std::vector<size_t> result;
    for (size_t i = 0; i < m_Regions.size(); ++i)
        if (m_Regions[i].dirty) result.push_back(i);
    return result;
}

void SceneRegistry::clearDirtyFlags() {
    for (auto& a : m_Assets) a.dirty = false;
    for (auto& r : m_Regions) r.dirty = false;
}

void SceneRegistry::markAllDirty() {
    for (auto& a : m_Assets) a.dirty = true;
    for (auto& r : m_Regions) r.dirty = true;
}

// ===========================================================================
// Rebuild
// ===========================================================================

void SceneRegistry::rebuildDirtyAssets() {
    for (auto& asset : m_Assets)
        if (asset.dirty) _rebuildAsset(asset);
}

void SceneRegistry::rebuildDirtyRegions() {
    bool anyRebuilt = false;
    for (auto& region : m_Regions) {
        if (region.dirty) {
            _rebuildRegion(region);
            anyRebuilt = true;
        }
    }
    if (anyRebuilt) _recomputeSceneBounds();
}

void SceneRegistry::_recomputeSceneBounds() {
    m_SceneBounds = dm::box3::empty();
    for (const auto& region : m_Regions)
        m_SceneBounds |= region.cullBox;
    if (m_Terrain)
        m_SceneBounds |= m_Terrain->getBbox();
}

void SceneRegistry::_rebuildAsset(TreeAssetDef& asset) {
    // Re-expand L-System string.
    auto lsIt = m_LSystems.find(asset.lsystemInstance.name);
    if (lsIt == m_LSystems.end()) return;

    lsIt->second->reset();
    // Per-asset seed drives both ring/branch jitter (in TreeGenerator) and stochastic rule
    // selection (in LSystem). Sharing the field keeps the JSON simple: bumping `seed` on an
    // asset re-rolls everything coherently.
    lsIt->second->setSeed(asset.genParams.seed);
    lsIt->second->generate(asset.lsystemInstance.gen);
    asset.lsystemString = lsIt->second->getCurrentString();

    // Generate CPU vertex/index data for each LOD. SC graph is built once on LOD-0's tips
    // (positions are LOD-independent — only ring tessellation varies per LOD) then reused.
    asset.lods.resize(m_LodSegments.size());
    auto savedSegments = asset.genParams.radialSegments;

    ProcGen::SpaceColonizer sc;

    for (size_t j = 0; j < m_LodSegments.size(); ++j) {
        asset.genParams.radialSegments = m_LodSegments[j];
        m_TreeGenerator->setParams(asset.genParams);

        ProcGen::Buffers lod;
        m_TreeGenerator->generateVertexAndIndexBuffers(asset.lsystemString, lod);

        if (j == 0) {
            // Build SC graph from LOD-0 tips. SC seed defaults to a salted form of the asset
            // seed so each asset gets a unique colonization without requiring the user to set
            // a separate seed; an explicit colonization.seed > 0 overrides.
            ProcGen::SCParams scParams = asset.colonization;
            if (scParams.seed == 0) scParams.seed = asset.genParams.seed ^ 0xC010D11Bu;
            sc.setParams(scParams);
            sc.grow(lod.branchTipPositions, 
                    lod.branchTipDirs, 
                    lod.branchTipRights, 
                    lod.branchTipRadii, 
                    lod.branchTipBranchLengths);
        }

        if (!sc.nodes().empty()) {
            m_TreeGenerator->emitColonizationCylinders(sc.nodes(), lod, m_LodSegments[j]);

            // Leaves: one cross-billboard cluster per terminal SC node, scaled by the
            // per-asset LOD multiplier. LOD3's default multiplier is 0 → distant trees lose
            // leaves entirely. countPerTip is rounded; sub-1 values still yield 0 verts.
            if (asset.hasLeaves && j < asset.leaf.lodMultipliers.size()) {
                const float    mult        = asset.leaf.lodMultipliers[j];
                const uint32_t countPerTip = static_cast<uint32_t>(std::round(asset.leaf.perTip * mult));
                if (countPerTip > 0) {
                    const uint32_t leafSeed = (asset.genParams.seed == 0 ? 1u : asset.genParams.seed) ^ 0x1EAF7E5Du;
                    m_TreeGenerator->emitLeafCrosses(sc.nodes(), sc.terminals(),
                                                     countPerTip, asset.leaf.size, asset.leaf.color,
                                                     leafSeed, lod);
                }
            }
        }

        asset.lods[j].positions      = std::move(lod.positions);
        asset.lods[j].normals        = std::move(lod.normals);
        asset.lods[j].tangents       = std::move(lod.tangents);
        asset.lods[j].bitangents     = std::move(lod.bitangents);
        asset.lods[j].uvs            = std::move(lod.uvs);
        asset.lods[j].indices        = std::move(lod.indices);
        asset.lods[j].bbox           = lod.bbox;
        asset.lods[j].radialSegments = m_LodSegments[j];
    }

    asset.genParams.radialSegments = savedSegments;
}

void SceneRegistry::_rebuildRegion(RegionDef& region) {
    region.instances.clear();
    region.cullBox = dm::box3::empty();

    if (region.assetIds.empty() || m_Assets.empty()) return;

    const uint32_t seedPos = g_MasterSeed ^ 0xBEEFDEAD ^ static_cast<uint32_t>(&region - m_Regions.data());
    const uint32_t seedRot = g_MasterSeed ^ 0xFEEDBEEF ^ static_cast<uint32_t>(&region - m_Regions.data());

    const dm::float2 range = region.bounds.diagonal();

    // Compute max radius for spatial grid cell size.
    float maxRadius = 0.f;
    for (size_t aid : region.assetIds) {
        const auto* asset = findAsset(aid);
        if (!asset || asset->lods.empty()) continue;
        dm::float3 diag = asset->lods[0].bbox.diagonal();
        float r = dm::max(diag.x, diag.z) * 0.5f;
        maxRadius = dm::max(maxRadius, r);
    }

    // Phase A: rejection-sampled placement against a Voronoi (Worley F1)
    // density field. One feature point per cell of size kCellSize world units;
    // density peaks at the feature point and falls off to 0 at kClusterRadius.
    // Produces discrete tree islands with real clearings between them.
    // Phases B/C still run on top, so trunks are guaranteed not to overlap.
    constexpr float kCellSize       = 50.f;  // world units between cluster centers
    constexpr float kClusterRadius  = 0.55f; // fraction of cell occupied (0..1, ~sqrt(2)/2 max)
    constexpr float kDensityPower   = 1.5f;  // higher = harder cluster edges

    std::vector<TreePlacement> placements;
    placements.reserve(region.instanceCount);

    const uint32_t maxAttempts = region.instanceCount * 32u;
    uint32_t       attempts    = 0;

    while (placements.size() < region.instanceCount && attempts < maxAttempts) {
        float u = hashToFloat(attempts * 3u + 0u, seedPos);
        float v = hashToFloat(attempts * 3u + 1u, seedPos);
        float r = hashToFloat(attempts * 3u + 2u, seedPos);

        float posX = region.bounds.m_mins.x + u * range.x;
        float posZ = region.bounds.m_mins.y + v * range.y;

        float w = Noise::worleyF1_2D(posX / kCellSize, posZ / kCellSize, seedPos);
        // Map distance-to-nearest-feature -> density (close = dense, far = empty).
        float t       = std::clamp(1.f - (w / kClusterRadius), 0.f, 1.f);
        float density = std::pow(t, kDensityPower);

        if (r < density) {
            uint32_t    i       = static_cast<uint32_t>(placements.size());
            float       rotY    = hashToFloat(i, seedRot) * dm::PI_f;
            size_t      assetId = region.assetIds[i % region.assetIds.size()];
            const auto* asset   = findAsset(assetId);
            float       radius  = asset && !asset->lods.empty() ? maxRadius : 1.f;

            placements.push_back({ {posX, posZ}, radius, assetId, rotY });
        }
        attempts++;
    }

    // Fallback: if rejection thinned us out, top up uniformly. Downstream phases
    // assume placements.size() == region.instanceCount.
    while (placements.size() < region.instanceCount) {
        uint32_t    i       = static_cast<uint32_t>(placements.size());
        float       posX    = region.bounds.m_mins.x + hashToFloat(i * 2u,     seedPos) * range.x;
        float       posZ    = region.bounds.m_mins.y + hashToFloat(i * 2u + 1, seedPos) * range.y;
        float       rotY    = hashToFloat(i, seedRot) * dm::PI_f;
        size_t      assetId = region.assetIds[i % region.assetIds.size()];
        const auto* asset   = findAsset(assetId);
        float       radius  = asset && !asset->lods.empty() ? maxRadius : 1.f;

        placements.push_back({ {posX, posZ}, radius, assetId, rotY });
    }

    // Phase B: spatial grid.
    float cellSize = maxRadius * 2.f;
    if (cellSize < 0.001f) cellSize = 1.f;

    uint32_t gridW = static_cast<uint32_t>(std::ceil(range.x / cellSize));
    uint32_t gridH = static_cast<uint32_t>(std::ceil(range.y / cellSize));
    if (gridW == 0) gridW = 1;
    if (gridH == 0) gridH = 1;

    SpatialGrid grid{{gridW, gridH}, cellSize, region.bounds.m_mins};

    // Phase C: Poisson relaxation.
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
                        if (nx < 0 || nz < 0 ||
                            nx >= static_cast<int>(gridW) ||
                            nz >= static_cast<int>(gridH))
                            continue;

                        const auto& neighbor = grid.m_Cells[grid.flatten(dm::uint2(nx, nz))];

                        for (uint32_t a = 0; a < cell.size(); a++) {
                            uint32_t tI = cell[a];
                            for (uint32_t b = 0; b < neighbor.size(); b++) {
                                uint32_t tJ = neighbor[b];
                                if (tJ <= tI) continue;

                                float deltaX = placements[tJ].pos.x - placements[tI].pos.x;
                                float deltaZ = placements[tJ].pos.y - placements[tI].pos.y;
                                float distSq = deltaX * deltaX + deltaZ * deltaZ;
                                float minDist = placements[tI].radius + placements[tJ].radius;

                                if (distSq >= minDist * minDist) continue;

                                float dist    = std::sqrt(distSq);
                                float overlap = minDist - dist;

                                float dirX, dirZ;
                                if (dist < 1e-5f) {
                                    float angle = hashToFloat(tI ^ tJ, 0xBAADF00Du) * 2.f * dm::PI_f;
                                    dirX = std::cos(angle);
                                    dirZ = std::sin(angle);
                                } else {
                                    dirX = deltaX / dist;
                                    dirZ = deltaZ / dist;
                                }

                                float push = overlap * 0.4f;

                                displacements[tI].x -= dirX * push;
                                displacements[tI].y -= dirZ * push;

                                displacements[tJ].x += dirX * push;
                                displacements[tJ].y += dirZ * push;
                            }
                        }
                    }
                }
            }
        }

        for (uint32_t i = 0; i < region.instanceCount; i++) {
            placements[i].pos = dm::float2(
                std::clamp(placements[i].pos.x + displacements[i].x,
                           region.bounds.m_mins.x, region.bounds.m_maxs.x),
                std::clamp(placements[i].pos.y + displacements[i].y,
                           region.bounds.m_mins.y, region.bounds.m_maxs.y)
            );
        }
    }

    // Phase D: build instance data from relaxed positions.
    region.instances.reserve(region.instanceCount);

    for (uint32_t i = 0; i < region.instanceCount; i++) {
        const auto& p = placements[i];
        const auto* asset = findAsset(p.assetId);
        if (!asset || asset->lods.empty()) continue;

        float posY = m_Terrain ? m_Terrain->getHeightAt(p.pos.x, p.pos.y) : 0.f;

        dm::affine3 worldMatrix =
            dm::rotation(dm::float3(0.f, 1.f, 0.f), p.rotY)
            * dm::translation(dm::float3(p.pos.x, posY - 0.5f, p.pos.y));

        dm::float3x3 normalMatrix;
        if (dm::isnear(worldMatrix.m_linear[0][0], worldMatrix.m_linear[1][1]) &&
            dm::isnear(worldMatrix.m_linear[1][1], worldMatrix.m_linear[2][2])) {
            normalMatrix = worldMatrix.m_linear;
        } else {
            normalMatrix = dm::transpose(dm::inverse(worldMatrix.m_linear));
        }

        const dm::box3& assetBox = asset->lods[0].bbox;
        dm::box3 wTreeBbox = assetBox * worldMatrix;

        region.instances.push_back({
            dm::affineToHomogeneous(worldMatrix),
            normalMatrix,
            p.assetId,
            wTreeBbox
        });

        region.cullBox |= wTreeBbox;
    }
}

// ===========================================================================
// Direct setters
// ===========================================================================

void SceneRegistry::setLodConfig(std::vector<uint32_t> segments, std::vector<float> distances) {
    m_LodSegments  = std::move(segments);
    m_LodDistances = std::move(distances);
}

void SceneRegistry::setSunDirection(dm::float3 dir) {
    m_SunDirection = dm::normalize(dir);
}

void SceneRegistry::setCameraInit(const CameraInit& cameraInit) {
    m_CameraInit = cameraInit;
}

void SceneRegistry::setTerrain(std::unique_ptr<Scene::Terrain> terrain) {
    m_Terrain = std::move(terrain);
    _recomputeSceneBounds();
}

void SceneRegistry::setTreeGenerator(std::unique_ptr<ProcGen::TreeGenerator> gen) {
    m_TreeGenerator = std::move(gen);
}

uint32_t SceneRegistry::registerBarkTexture(const std::string& textureName) {
    auto it = std::find(m_BarkTextureSets.begin(), m_BarkTextureSets.end(), textureName);
    if (it != m_BarkTextureSets.end())
        return static_cast<uint32_t>(std::distance(m_BarkTextureSets.begin(), it));
    m_BarkTextureSets.push_back(textureName);
    return static_cast<uint32_t>(m_BarkTextureSets.size() - 1);
}

void SceneRegistry::_refreshAssetIdMap() {
    m_AssetIdToIndex.clear();
    for (size_t i = 0; i < m_Assets.size(); ++i)
        m_AssetIdToIndex[m_Assets[i].id] = i;
}

// ===========================================================================
// Clear
// ===========================================================================

void SceneRegistry::clear() {
    m_LSystems.clear();
    m_Assets.clear();
    m_Regions.clear();
    m_Terrain.reset();
    m_TreeGenerator.reset();
    m_BarkTextureSets.clear();
    m_LodSegments.clear();
    m_LodDistances.clear();
    m_SunDirection = dm::float3(0.f, -1.f, 0.f);
    m_CameraInit  = CameraInit{};
    m_NextAssetId = 0;
    m_AssetIdToIndex.clear();
    m_SceneBounds = dm::box3::empty();
}
