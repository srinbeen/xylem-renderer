#ifndef XYLEM_SCENE_REGISTRY_H
#define XYLEM_SCENE_REGISTRY_H

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <donut/core/math/math.h>

#include "Procgen.hpp"
#include "Scene.hpp"
#include "Terrain.hpp"

namespace Xylem {

// CPU-only asset definition — no GPU handles.
struct TreeAssetDef {
    uint32_t                          id;
    std::string                       name;
    Scene::LSystemInstance            lsystemInstance;
    ProcGen::TreeGenerator::Params    genParams;
    ProcGen::lstring_t                lsystemString;
    std::vector<Scene::TreeLODDef>    lods;
    std::string                       barkTexture;
    uint32_t                          textureSetIdx = 0;
    bool                              dirty = true;
};

// CPU-only instance data — stable asset ID, no GPU index.
struct InstanceData {
    dm::float4x4 model;
    dm::float3x3 normal;
    uint32_t     assetId;
    dm::box3     bbox;
};

// CPU-only region definition.
struct RegionDef {
    std::string               name;
    float                     density;
    uint32_t                  instanceCount;
    dm::box2                  bounds;
    std::vector<uint32_t>     assetIds;     // stable asset IDs
    std::vector<InstanceData> instances;
    dm::box3                  cullBox;
    bool                      dirty = true;

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

    uint32_t totalInstanceCount() const;

    // Lookup asset by stable ID (returns nullptr if not found).
    TreeAssetDef*       findAsset(uint32_t id);
    const TreeAssetDef* findAsset(uint32_t id) const;

    // Lookup asset vector index by stable ID (returns SIZE_MAX if not found).
    size_t assetIndexById(uint32_t id) const;

    // -------------------------------------------------------------------
    // Mutation — auto-sets dirty flags
    // -------------------------------------------------------------------
    void addLSystem(const std::string& name, std::unique_ptr<ProcGen::LSystem> ls);

    // Returns the stable ID assigned to the new asset.
    uint32_t addAsset(const std::string& name,
                      const Scene::LSystemInstance& lsInstance,
                      const ProcGen::TreeGenerator::Params& params,
                      const std::string& barkTexture = "bark_willow_02_1k");

    void modifyAsset(uint32_t id, const ProcGen::TreeGenerator::Params& params);
    void removeAsset(uint32_t id);

    void addRegion(const std::string& name, float density, const dm::box2& bounds,
                   const std::vector<uint32_t>& assetIds);
    void modifyRegion(size_t idx, float density, const dm::box2& bounds);
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

    uint32_t m_NextAssetId = 0;

    // Internal ID-to-index map for fast lookup.
    std::unordered_map<uint32_t, size_t> m_AssetIdToIndex;

    void _rebuildAsset(TreeAssetDef& asset);
    void _rebuildRegion(RegionDef& region);
    void _refreshAssetIdMap();
};

} // namespace Xylem

#endif // XYLEM_SCENE_REGISTRY_H
