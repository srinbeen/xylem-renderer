#include "include/SceneLoader.hpp"
#include "include/Terrain.hpp"

#include <algorithm>
#include <fstream>

#include <donut/core/json.h>

using namespace Xylem;
using namespace Xylem::Scene;
using namespace Xylem::ProcGen;

// static
bool SceneLoader::Load(const std::filesystem::path& path, SceneRegistry& registry) {
    const Json::Value root = ParseFile(path);

    // -------------------------------------------------------------------------
    // LODs
    // -------------------------------------------------------------------------
    {
        std::vector<uint32_t> lodSegments;
        std::vector<float>    lodDistances;
        if (root.isMember("lods")) {
            const auto& lodNode = root["lods"];
            for (const auto& segment : lodNode["radialSegments"]) {
                uint32_t segments = 8;
                segment >> segments;
                lodSegments.push_back(segments);
            }
            for (const auto& distance : lodNode["distances"]) {
                float    distances = 100.f;
                distance >> distances;
                lodDistances.push_back(distances);
            }
        } else {
            lodSegments  = { 16, 8, 4 };
            lodDistances = { 16.0f, 64.0f, 256.0f };
        }
        registry.setLodConfig(std::move(lodSegments), std::move(lodDistances));
    }

    // -------------------------------------------------------------------------
    // Sun
    // -------------------------------------------------------------------------
    {
        dm::float3 sunDir = dm::float3(0.f, -1.f, 0.f);
        if (root.isMember("sun")) {
            const auto& sunNode = root["sun"];
            sunNode["direction"] >> sunDir;
        }
        registry.setSunDirection(sunDir);
    }


    // -------------------------------------------------------------------------
    // Camera
    // -------------------------------------------------------------------------
    {
        const auto& cam = root["camera"];

        SceneRegistry::CameraInit cameraInit;

        cam["position"] >> cameraInit.pos;
        cam["direction"] >> cameraInit.cameraDir;
        cam["moveSpeed"] >> cameraInit.moveSpeed;

        registry.setCameraInit(cameraInit);
    }

    // -------------------------------------------------------------------------
    // L-Systems
    // -------------------------------------------------------------------------
    for (const auto& lsNode : root["lsystems"]) {
        std::string name = "", axiom = "";
        lsNode["name"]  >> name;
        lsNode["axiom"] >> axiom;

        std::unordered_map<char, std::string> rules;
        for (const auto& key : lsNode["rules"].getMemberNames())
            if (!key.empty()) rules[key[0]] = lsNode["rules"][key].asString();

        registry.addLSystem(name, std::make_unique<LSystem>(axiom, rules));
    }
    if (registry.getLSystems().empty()) return false;

    // -------------------------------------------------------------------------
    // Tree Generator
    // -------------------------------------------------------------------------
    registry.setTreeGenerator(std::make_unique<TreeGenerator>());

    // -------------------------------------------------------------------------
    // Assets
    // -------------------------------------------------------------------------
    // Keep a name→stableId map for region wiring below.
    std::unordered_map<std::string, size_t> assetNameToId;

    for (const auto& aNode : root["assets"]) {
        std::string assetName;
        std::string lsName = registry.getLSystems().begin()->first;
        uint32_t    gen    = 3;

        aNode["name"]       >> assetName;
        aNode["lsystem"]    >> lsName;
        aNode["generation"] >> gen;

        TreeGenerator::Params params;
        params.radialSegments = 16;
        params.stepLength     = 1.0f;
        params.branchAngle    = 25.0f;
        params.taperRatio     = 0.9f;
        params.stepRatio      = 0.95f;
        params.seed           = 0;

        aNode["radialSegments"] >> params.radialSegments;
        aNode["stepLength"]     >> params.stepLength;
        aNode["branchAngle"]    >> params.branchAngle;
        aNode["taperRatio"]     >> params.taperRatio;
        aNode["stepRatio"]      >> params.stepRatio;
        aNode["seed"]           >> params.seed;

        params.branchAngle = dm::radians(params.branchAngle);

        std::string barkTexture = "bark_willow_02_1k";
        aNode["barkTexture"] >> barkTexture;

        LSystemInstance lsInstance { lsName, gen };
        size_t stableId = registry.addAsset(assetName, lsInstance, params, barkTexture);
        assetNameToId[assetName] = stableId;
    }

    // -------------------------------------------------------------------------
    // Regions (collect first, rebuild after terrain generation)
    // -------------------------------------------------------------------------
    struct PendingRegion {
        size_t                  index;
        std::vector<size_t>     assetIds;
    };
    std::vector<PendingRegion> pendingRegions;

    for (const auto& rNode : root["regions"]) {
        float    density = 0.02f;
        float    minX = 0.f, minY = 0.f, maxX = 10.f, maxY = 10.f;
        std::string key = "Region";

        rNode["density"]    >> density;
        rNode["boundsMinX"] >> minX;
        rNode["boundsMinY"] >> minY;
        rNode["boundsMaxX"] >> maxX;
        rNode["boundsMaxY"] >> maxY;
        rNode["name"]       >> key;

        dm::box2 bounds(dm::float2(minX, minY), dm::float2(maxX, maxY));

        std::vector<size_t> regionAssetIds;
        if (rNode.isMember("assets") && rNode["assets"].size() > 0) {
            for (const auto& aName : rNode["assets"]) {
                std::string n = aName.asString();
                auto it = assetNameToId.find(n);
                if (it != assetNameToId.end())
                    regionAssetIds.push_back(it->second);
            }
        }

        size_t regionIdx = registry.getRegions().size();
        registry.addRegion(key, density, bounds, regionAssetIds);
        if (!regionAssetIds.empty())
            pendingRegions.push_back({ regionIdx, regionAssetIds });
    }

    // -------------------------------------------------------------------------
    // Terrain — generate before rebuilding regions so trees snap to height
    // -------------------------------------------------------------------------
    {
        TerrainConfig terrainConfig;

        float padding = 10.f;
        float extMinX =  FLT_MAX, extMinZ =  FLT_MAX;
        float extMaxX = -FLT_MAX, extMaxZ = -FLT_MAX;
        for (const auto& r : registry.getRegions()) {
            extMinX = std::min(extMinX, r.bounds.m_mins.x);
            extMinZ = std::min(extMinZ, r.bounds.m_mins.y);
            extMaxX = std::max(extMaxX, r.bounds.m_maxs.x);
            extMaxZ = std::max(extMaxZ, r.bounds.m_maxs.y);
        }
        terrainConfig.worldMinX = extMinX - padding;
        terrainConfig.worldMinZ = extMinZ - padding;
        terrainConfig.worldMaxX = extMaxX + padding;
        terrainConfig.worldMaxZ = extMaxZ + padding;

        if (root.isMember("terrain")) {
            const auto& tNode = root["terrain"];
            tNode["seed"]        >> terrainConfig.seed;
            tNode["octaves"]     >> terrainConfig.octaves;
            tNode["frequency"]   >> terrainConfig.frequency;
            tNode["amplitude"]   >> terrainConfig.amplitude;
            tNode["lacunarity"]  >> terrainConfig.lacunarity;
            tNode["persistence"] >> terrainConfig.persistence;
            tNode["gridSpacing"] >> terrainConfig.gridSpacing;
        }

        auto terrain = std::make_unique<Terrain>();
        terrain->generate(terrainConfig);
        registry.setTerrain(std::move(terrain));
    }

    // -------------------------------------------------------------------------
    // Rebuild all CPU data (mesh generation + instance placement)
    // -------------------------------------------------------------------------
    registry.rebuildDirtyAssets();
    registry.rebuildDirtyRegions();
    registry.clearDirtyFlags();

    return true;
}

// static
Json::Value SceneLoader::ParseFile(const std::filesystem::path& path) {
    Json::CharReaderBuilder reader;
    Json::Value root;
    std::string errors;
    std::ifstream jsonStream(path);
    Json::parseFromStream(reader, jsonStream, &root, &errors);
    return root;
}
