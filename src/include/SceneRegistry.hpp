#ifndef XYLEM_SCENE_REGISTRY_H
#define XYLEM_SCENE_REGISTRY_H

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <donut/core/math/math.h>

#include "Procgen.hpp"
#include "Scene.hpp"
#include "SpaceColonizer.hpp"
#include "Terrain.hpp"

namespace Xylem {

// Per-asset leaf rendering parameters. Leaf geometry (cross-billboards) is emitted
// at every space-colonization terminal, scaled per-LOD by `lodMultipliers`.
struct LeafParams {
    dm::float3            color          = dm::float3(0.20f, 0.55f, 0.18f);
    float                 size           = 0.25f;
    uint32_t              perTip         = 1;          // 1 leaf per terminal by default — bumping
                                                       // produces clumping at SC-dense regions.
    std::array<float, 4>  lodMultipliers = {1.0f, 0.5f, 0.25f, 0.0f};
};

// CPU-only asset definition — no GPU handles.
struct TreeAssetDef {
    size_t                            id;
    std::string                       name;
    Scene::LSystemInstance            lsystemInstance;
    ProcGen::TreeGenerator::Params    genParams;
    ProcGen::lstring_t                lsystemString;
    std::vector<Scene::TreeLODDef>    lods;
    std::string                       barkTexture;
    uint32_t                          textureSetIdx = 0;
    ProcGen::SCParams                 colonization;          // disabled when attractorCount == 0
    LeafParams                        leaf;
    bool                              hasLeaves = true;      // master switch for leaf emission
    bool                              dirty   = true;
    bool                              visible = true;  // render-time visibility toggle
};

// CPU-only instance data — stable asset ID, no GPU index.
struct InstanceData {
    dm::float4x4 model;
    dm::float3x3 normal;
    size_t       assetId;
    dm::box3     bbox;
};

// CPU-only region definition.
struct RegionDef {
    std::string               name;
    float                     density;
    uint32_t                  instanceCount;
    dm::box2                  bounds;
    std::vector<size_t>       assetIds;     // stable asset IDs
    std::vector<InstanceData> instances;
    dm::box3                  cullBox;
    bool                      dirty = true;
    // Per-asset render-time visibility within this region (absent = visible).
    std::unordered_map<size_t, bool> assetVisible;

    RegionDef() = default;
    RegionDef(const std::string& n, float d, const dm::box2& b)
        : name{n}, density{d}, bounds{b}, cullBox{dm::box3::empty()}
    {
        float area = (b.m_maxs.x - b.m_mins.x) * (b.m_maxs.y - b.m_mins.y);
        instanceCount = std::max(1u, static_cast<uint32_t>(std::round(d * area)));
    }
};

// CPU-only scene data with dirty-flag driven hot-reload support.
class SceneRegistry {
public:

    struct CameraInit {
        dm::float3 pos = 0.f;
        dm::float3 cameraDir = {0.f, 0.f, 1.f};
        float moveSpeed = 15.f;
    };

    // -------------------------------------------------------------------
    // Read access
    // -------------------------------------------------------------------
    const std::map<std::string, std::unique_ptr<ProcGen::LSystem>>& getLSystems() const { return m_LSystems; }
    const std::vector<TreeAssetDef>&                                getAssets()   const { return m_Assets; }
    const std::vector<RegionDef>&                                   getRegions()  const { return m_Regions; }
    const Scene::Terrain*                                           getTerrain()  const { return m_Terrain.get(); }
    const std::vector<uint32_t>&                                    getLodSegments()  const { return m_LodSegments; }
    const std::vector<float>&                                       getLodDistances() const { return m_LodDistances; }
    const std::vector<std::string>&                                 getBarkTextureSets() const { return m_BarkTextureSets; }
    dm::float3                                                      getSunDirection() const { return m_SunDirection; }
    const CameraInit&                                               getCameraInit() const { return m_CameraInit; }

    // Union of all region cullBoxes plus the terrain bbox. Refreshed lazily
    // inside rebuildDirtyRegions() and setTerrain(); avoids per-frame recompute.
    const dm::box3& getSceneBounds() const { return m_SceneBounds; }

    uint32_t totalInstanceCount() const;

    // Lookup asset by stable ID (returns nullptr if not found).
    TreeAssetDef*       findAsset(size_t id);
    const TreeAssetDef* findAsset(size_t id) const;

    // Lookup asset vector index by stable ID (returns SIZE_MAX if not found).
    size_t assetIndexById(size_t id) const;

    // -------------------------------------------------------------------
    // Mutation — auto-sets dirty flags
    // -------------------------------------------------------------------
    // Wipes the registry to a default-constructed state. Used by the in-session
    // scene reload flow so a fresh SceneLoader::Load can populate from scratch.
    void clear();

    void addLSystem(const std::string& name, std::unique_ptr<ProcGen::LSystem> ls);

    // Returns the stable ID assigned to the new asset.
    size_t addAsset(const std::string& name,
                      const Scene::LSystemInstance& lsInstance,
                      const ProcGen::TreeGenerator::Params& params,
                      const std::string& barkTexture = "bark_willow_02_1k");

    void modifyAsset(size_t id, const ProcGen::TreeGenerator::Params& params);
    // Full extended modification — sets gen + colonization + leaf params at once and marks
    // the asset dirty. Used by the UI's Foliage / Space Colonization editors.
    void modifyAssetExtended(size_t                                id,
                             const ProcGen::TreeGenerator::Params& genParams,
                             const ProcGen::SCParams&              colonization,
                             const LeafParams&                     leaf,
                             bool                                  hasLeaves);
    void setAssetVisible(size_t id, bool visible);
    void removeAsset(size_t id);

    void addRegion(const std::string& name, float density, const dm::box2& bounds,
                   const std::vector<size_t>& assetIds);
    void modifyRegion(size_t idx, float density, const dm::box2& bounds);
    void modifyRegionAssets(size_t idx, const std::vector<size_t>& assetIds);
    void setRegionAssetVisible(size_t regionIdx, size_t assetId, bool visible);
    void removeRegion(size_t idx);

    // -------------------------------------------------------------------
    // Dirty flag queries
    // -------------------------------------------------------------------
    bool anyAssetsDirty() const;
    bool anyRegionsDirty() const;
    bool anyDirty() const { return anyAssetsDirty() || anyRegionsDirty(); }

    std::vector<size_t> getDirtyAssetIndices() const;
    std::vector<size_t> getDirtyRegionIndices() const;

    void clearDirtyFlags();
    void markAllDirty();

    // -------------------------------------------------------------------
    // Rebuild — regenerates CPU data for dirty items
    // -------------------------------------------------------------------
    void rebuildDirtyAssets();
    void rebuildDirtyRegions();

    // -------------------------------------------------------------------
    // Direct setters for scene loading
    // -------------------------------------------------------------------
    void setLodConfig(std::vector<uint32_t> segments, std::vector<float> distances);
    void setSunDirection(dm::float3 dir);
    void setCameraInit(const CameraInit& cameraInit);
    void setTerrain(std::unique_ptr<Scene::Terrain> terrain);
    void setTreeGenerator(std::unique_ptr<ProcGen::TreeGenerator> gen);

    // Register a bark texture set, returning its index. Deduplicates by name.
    uint32_t registerBarkTexture(const std::string& textureName);

private:
    std::map<std::string, std::unique_ptr<ProcGen::LSystem>> m_LSystems;
    std::vector<TreeAssetDef>                                m_Assets;
    std::vector<RegionDef>                                   m_Regions;

    std::unique_ptr<ProcGen::TreeGenerator>                  m_TreeGenerator;
    std::unique_ptr<Scene::Terrain>                          m_Terrain;

    std::vector<uint32_t>                                    m_LodSegments;
    std::vector<float>                                       m_LodDistances;
    std::vector<std::string>                                 m_BarkTextureSets;
    dm::float3                                               m_SunDirection = dm::float3(0.f, -1.f, 0.f);
    CameraInit                                               m_CameraInit;

    dm::box3                                                 m_SceneBounds = dm::box3::empty();

    size_t m_NextAssetId = 0;

    // Internal ID-to-index map for fast lookup.
    std::unordered_map<size_t, size_t> m_AssetIdToIndex;

    void _rebuildAsset(TreeAssetDef& asset);
    void _rebuildRegion(RegionDef& region);
    void _refreshAssetIdMap();
    void _recomputeSceneBounds();
};

} // namespace Xylem

#endif // XYLEM_SCENE_REGISTRY_H
