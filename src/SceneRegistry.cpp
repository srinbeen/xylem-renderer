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

constexpr uint32_t kLeavesPerMeshlet = 8;

dm::float3 xyz(const dm::float4& v) {
    return dm::float3(v.x, v.y, v.z);
}

void includePoint(dm::box3& box, const dm::float3& p) {
    box.m_mins = dm::min(box.m_mins, p);
    box.m_maxs = dm::max(box.m_maxs, p);
}

void includeLeafCorners(dm::box3& box, const Scene::LeafInstance& leaf) {
    const dm::float3 center = xyz(leaf.centerHalfSize);
    const dm::float3 spine  = xyz(leaf.spine);
    const dm::float3 right  = xyz(leaf.right);
    const dm::float3 q2     = dm::normalize(dm::cross(spine, right));
    const float half        = leaf.centerHalfSize.w;

    const dm::float3 axes[2] = { right, q2 };
    for (const dm::float3& axis : axes) {
        includePoint(box, center + axis * -half + spine * -half);
        includePoint(box, center + axis *  half + spine * -half);
        includePoint(box, center + axis *  half + spine *  half);
        includePoint(box, center + axis * -half + spine *  half);
    }
}

Scene::LeafMeshlet buildLeafMeshlet(const std::vector<Scene::LeafInstance>& leaves,
                                    uint32_t firstLeaf,
                                    uint32_t leafCount)
{
    dm::box3 bounds = dm::box3::empty();
    for (uint32_t i = 0; i < leafCount; i++)
        includeLeafCorners(bounds, leaves[firstLeaf + i]);

    const dm::float3 center = (bounds.m_mins + bounds.m_maxs) * 0.5f;
    const float radius = dm::length(bounds.diagonal()) * 0.5f;

    Scene::LeafMeshlet meshlet{};
    meshlet.meta   = dm::uint4(firstLeaf, leafCount, 0, 0);
    meshlet.bounds = dm::float4(center, radius);
    return meshlet;
}

// Tuning knob for cluster-radius cap. R_MAX = kClusterRadiusFactor * cbrt(K) *
// cbrt(canopy_volume / leaf_count) — a multiple of the mean leaf spacing scaled
// up to fit ~K leaves in a sphere. Larger values produce fewer, looser clusters
// (more leaves per meshlet on average); smaller values produce more, tighter
// clusters (better AS-side culling, more meshlet overhead).
constexpr float kClusterRadiusFactor = 1.5f;

// Greedy seeded nearest-neighbor cluster: starts from the lowest unassigned leaf
// and pulls in nearest unassigned leaves one at a time, stopping when either the
// cluster reaches kLeavesPerMeshlet members or adding the next nearest would
// exceed `rMax`. A 7-leaf cluster with one far outlier becomes a tight 7-leaf
// meshlet plus a 1-leaf meshlet, which is the right tradeoff: AS-side culling
// benefits from tight bounds far more than from packing meshlets full.
struct LeafCluster {
    std::vector<uint32_t> indices;
    dm::box3              bounds = dm::box3::empty();
};

bool tryAddToCluster(LeafCluster&                              c,
                     const std::vector<Scene::LeafInstance>&   leaves,
                     uint32_t                                  leafIdx,
                     float                                     rMax)
{
    dm::box3 newBounds = c.bounds;
    includeLeafCorners(newBounds, leaves[leafIdx]);
    const float newRadius = dm::length(newBounds.diagonal()) * 0.5f;
    if (newRadius > rMax) return false;
    c.bounds = newBounds;
    c.indices.push_back(leafIdx);
    return true;
}

std::vector<LeafCluster> clusterLeaves(const std::vector<Scene::LeafInstance>& leaves,
                                       float                                    rMax)
{
    std::vector<uint8_t> assigned(leaves.size(), 0);
    std::vector<LeafCluster> clusters;

    for (uint32_t seed = 0; seed < leaves.size(); ++seed) {
        if (assigned[seed]) continue;

        LeafCluster c;
        includeLeafCorners(c.bounds, leaves[seed]);
        c.indices.push_back(seed);
        assigned[seed] = 1;

        while (c.indices.size() < kLeavesPerMeshlet) {
            const dm::float3 center = (c.bounds.m_mins + c.bounds.m_maxs) * 0.5f;

            // O(N) nearest-unassigned-to-center scan. Total clustering cost is
            // O(N^2 / kLeavesPerMeshlet); for typical asset sizes (~hundreds of
            // leaves) this runs in well under a millisecond on a single thread.
            float    bestDistSq = std::numeric_limits<float>::max();
            uint32_t bestIdx    = UINT32_MAX;
            for (uint32_t i = 0; i < leaves.size(); ++i) {
                if (assigned[i]) continue;
                const dm::float3 leafPos = xyz(leaves[i].centerHalfSize);
                const dm::float3 d       = leafPos - center;
                const float      distSq  = dm::dot(d, d);
                if (distSq < bestDistSq) { bestDistSq = distSq; bestIdx = i; }
            }
            if (bestIdx == UINT32_MAX) break;
            if (!tryAddToCluster(c, leaves, bestIdx, rMax)) break;
            assigned[bestIdx] = 1;
        }

        clusters.push_back(std::move(c));
    }
    return clusters;
}

float computeClusterRadiusCap(const dm::box3& bbox, uint32_t leafCount)
{
    if (leafCount <= kLeavesPerMeshlet) {
        // Whole canopy fits in one meshlet — cap at the canopy's own bounding sphere.
        return dm::length(bbox.diagonal()) * 0.5f + 1e-3f;
    }
    const dm::float3 d      = bbox.diagonal();
    const float      volume = std::max(d.x * d.y * d.z, 1e-6f);
    // Mean spacing for a uniform distribution of `leafCount` points in `volume`.
    const float meanSpacing  = std::cbrt(volume / static_cast<float>(leafCount));
    // Radius that nominally encloses kLeavesPerMeshlet such points.
    const float clusterScale = std::cbrt(static_cast<float>(kLeavesPerMeshlet));
    return kClusterRadiusFactor * clusterScale * meanSpacing;
}

Scene::LeafAssetDef buildLeafAsset(const std::vector<ProcGen::SCNode>& nodes,
                                   const std::vector<uint32_t>&         terminals,
                                   const LeafParams&                    params,
                                   bool                                 hasLeaves,
                                   uint32_t                             numLods,
                                   uint32_t                             seed)
{
    Scene::LeafAssetDef out;
    out.countByLod.assign(numLods, 0);
    out.lodSlots.assign(numLods, {});

    if (!hasLeaves || params.perTip == 0 || params.size <= 0.f || terminals.empty())
        return out;

    const float halfSize = params.size * 0.5f;

    // Generation pass: emit one leaf per (terminal, perTip rank). Order is irrelevant
    // here since the cluster pass below reorders by spatial proximity anyway. v1 has
    // no per-LOD count thinning — every pipeline renders the full set.
    for (uint32_t k = 0; k < params.perTip; ++k) {
        for (size_t t = 0; t < terminals.size(); ++t) {
            const uint32_t nodeIdx = terminals[t];
            if (nodeIdx >= nodes.size()) continue;
            const auto& n = nodes[nodeIdx];

            dm::float3 forward = n.dir;
            const float fLen = dm::length(forward);
            forward = (fLen > 1e-6f) ? forward / fLen : ProcGen::unit_j;

            dm::float3 right = n.right - forward * dm::dot(n.right, forward);
            const float rLen = dm::length(right);
            right = (rLen > 1e-6f) ? right / rLen
                : ((std::abs(forward.y) < 0.9f) ? dm::normalize(dm::cross(forward, ProcGen::unit_j))
                                                : dm::normalize(dm::cross(forward, ProcGen::unit_i)));
            const dm::float3 up = dm::cross(forward, right);

            const uint32_t streamIdx = static_cast<uint32_t>(t) * 1024u + k;
            const float phi    = Xylem::hashToFloat(streamIdx, seed ^ 0xD0D0D0Du) * 2.f * dm::PI_f;
            const float cosPhi = std::cos(phi);
            const float sinPhi = std::sin(phi);
            const dm::float3 rolledRight = right * cosPhi + up * sinPhi;
            const dm::float3 rolledUp    = up    * cosPhi - right * sinPhi;

            const float theta = Xylem::hashToFloat(streamIdx, seed ^ 0xB1A5FEEDu) * dm::PI_f * 0.5f;
            const dm::float3 spine = dm::normalize(forward * std::cos(theta) + rolledUp * std::sin(theta));
            const dm::float3 q2 = dm::normalize(dm::cross(spine, rolledRight));

            dm::float3 center = n.pos;
            if (k > 0) {
                const float jx = (Xylem::hashToFloat(streamIdx * 3u + 0u, seed) - 0.5f) * params.size;
                const float jy = (Xylem::hashToFloat(streamIdx * 3u + 1u, seed) - 0.5f) * params.size;
                const float jz = (Xylem::hashToFloat(streamIdx * 3u + 2u, seed) - 0.5f) * params.size;
                center = n.pos + rolledRight * jx + spine * jy + q2 * jz;
            }

            const float scaleHash = Xylem::hashToFloat(streamIdx, seed ^ 0xF01FA11u);
            const float leafHalf  = halfSize * (0.5f + scaleHash);

            Scene::LeafInstance leaf{};
            leaf.centerHalfSize = dm::float4(center, leafHalf);
            leaf.spine          = dm::float4(spine, 0.f);
            leaf.right          = dm::float4(dm::normalize(rolledRight), 0.f);
            leaf.color          = dm::float4(params.color, 1.f);
            out.instances.push_back(leaf);
            includeLeafCorners(out.localBbox, leaf);
        }
    }

    // Spatial clustering: greedy seeded nearest-neighbor with a density-adaptive
    // radius cap. Each cluster becomes one variable-size leaf meshlet (1..K leaves).
    const float rMax = computeClusterRadiusCap(out.localBbox,
        static_cast<uint32_t>(out.instances.size()));
    const std::vector<LeafCluster> clusters = clusterLeaves(out.instances, rMax);

    // Reorder leaf instances so cluster K's leaves sit immediately after cluster K-1's.
    // This is the invariant that lets `LeafSlot.leafCount = Σ meta.y` and the per-leaf
    // prefix [leafOffset, leafOffset + leafCount) match the meshlet prefix exactly.
    std::vector<Scene::LeafInstance> reordered;
    reordered.reserve(out.instances.size());
    out.meshlets.clear();
    out.meshlets.reserve(clusters.size());

    for (const auto& cluster : clusters) {
        const uint32_t leafBase = static_cast<uint32_t>(reordered.size());
        for (uint32_t idx : cluster.indices)
            reordered.push_back(out.instances[idx]);

        Scene::LeafMeshlet meshlet{};
        meshlet.meta = dm::uint4(leafBase,
                                 static_cast<uint32_t>(cluster.indices.size()),
                                 0, 0);
        const dm::float3 center = (cluster.bounds.m_mins + cluster.bounds.m_maxs) * 0.5f;
        const float      radius = dm::length(cluster.bounds.diagonal()) * 0.5f;
        meshlet.bounds = dm::float4(center, radius);
        out.meshlets.push_back(meshlet);
    }
    out.instances = std::move(reordered);

    // No per-LOD thinning in v1 — every LOD slot is the full leaf set. Keeping the
    // per-LOD slot array (rather than collapsing to a single slot) lets the existing
    // SharedGPUAssets / render-pass code index by `slot = ai * numLods + li` without
    // change. When AS-side leaf-meshlet culling lands as the perf lever, it operates
    // on the meshlet array directly and still doesn't need per-LOD slots.
    Scene::LeafSlot fullSlot{};
    fullSlot.leafOffset    = 0;
    fullSlot.leafCount     = static_cast<uint32_t>(out.instances.size());
    fullSlot.meshletOffset = 0;
    fullSlot.meshletCount  = static_cast<uint32_t>(out.meshlets.size());

    out.countByLod.assign(numLods, fullSlot.leafCount);
    out.lodSlots.assign(numLods, fullSlot);

    return out;
}

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

uint32_t SceneRegistry::totalLeafInstanceCount() const {
    uint32_t total = 0;
    for (const auto& r : m_Regions) {
        for (const auto& inst : r.instances) {
            const auto* asset = findAsset(inst.assetId);
            if (asset && asset->hasLeaves)
                total += static_cast<uint32_t>(asset->leafAsset.instances.size());
        }
    }
    return total;
}

uint32_t SceneRegistry::totalLeafMeshletCount() const {
    uint32_t total = 0;
    for (const auto& r : m_Regions) {
        for (const auto& inst : r.instances) {
            const auto* asset = findAsset(inst.assetId);
            if (asset && asset->hasLeaves)
                total += static_cast<uint32_t>(asset->leafAsset.meshlets.size());
        }
    }
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

    const uint32_t leafSeed = (asset.genParams.seed == 0 ? 1u : asset.genParams.seed) ^ 0x1EAF7E5Du;
    asset.leafAsset = buildLeafAsset(sc.nodes(), sc.terminals(), asset.leaf, asset.hasLeaves,
                                     static_cast<uint32_t>(m_LodSegments.size()), leafSeed);
    if (!asset.leafAsset.localBbox.isempty()) {
        for (auto& lod : asset.lods)
            lod.bbox |= asset.leafAsset.localBbox;
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

    constexpr float    k_TiltFactor    = 0.3f;
    constexpr float    k_BaseSink      = 0.5f;
    constexpr float    k_SinkScale     = 1.0f;
    constexpr float    k_TiltCutoff    = 0.99f;
    const dm::float3   worldUp(0.f, 1.f, 0.f);

    for (uint32_t i = 0; i < region.instanceCount; i++) {
        const auto& p = placements[i];
        const auto* asset = findAsset(p.assetId);
        if (!asset || asset->lods.empty()) continue;

        float      terrainY = m_Terrain ? m_Terrain->getHeightAt(p.pos.x, p.pos.y) : 0.f;
        dm::float3 normal   = m_Terrain ? m_Terrain->getNormalAt(p.pos.x, p.pos.y) : worldUp;

        dm::float3 tiltedUp = dm::normalize(dm::lerp(worldUp, normal, k_TiltFactor));
        float      sink     = k_BaseSink + k_SinkScale * (1.f - normal.y);

        dm::affine3 selfRot         = dm::rotation(worldUp, p.rotY);
        dm::affine3 terrainPlace    = dm::translation(dm::float3(p.pos.x, terrainY - sink, p.pos.y));

        dm::affine3 worldMatrix;
        // if they're not super parallel
        if (dm::dot(worldUp, tiltedUp) < k_TiltCutoff) {
            dm::float3  tiltAxisUnNorm  = dm::cross(worldUp, tiltedUp);
            float       tiltAngle       = std::asinf(dm::length(tiltAxisUnNorm));
            dm::affine3 terrainTilt     = dm::rotation(tiltAxisUnNorm/tiltAngle, tiltAngle);
            worldMatrix = selfRot * terrainTilt * terrainPlace;
        }
        else {
            worldMatrix = selfRot * terrainPlace;
        }

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

