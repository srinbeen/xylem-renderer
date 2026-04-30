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

        // Rule values are either a plain string (deterministic) or an array of
        //   { "weight": <float>, "rhs": "<string>" }
        // objects (stochastic). We classify per-rule so a single L-system can mix both.
        std::unordered_map<char, std::string>                                   detRules;
        std::unordered_map<char, std::vector<std::pair<float, std::string>>>    weightedRules;
        bool anyWeighted = false;

        const auto& rulesNode = lsNode["rules"];
        for (const auto& key : rulesNode.getMemberNames()) {
            if (key.empty()) continue;
            const auto& v = rulesNode[key];
            if (v.isArray()) {
                std::vector<std::pair<float, std::string>> productions;
                productions.reserve(v.size());
                for (const auto& prod : v) {
                    float       w   = 1.f;
                    std::string rhs;
                    prod["weight"] >> w;
                    prod["rhs"]    >> rhs;
                    productions.emplace_back(w, std::move(rhs));
                }
                weightedRules[key[0]] = std::move(productions);
                anyWeighted = true;
            } else {
                detRules[key[0]] = v.asString();
            }
        }

        std::unique_ptr<LSystem> ls;
        if (anyWeighted) {
            // Lift any deterministic rules into weight=1 single-production entries so the
            // weighted constructor sees a complete rule set.
            for (const auto& [c, s] : detRules)
                weightedRules[c].push_back({ 1.f, s });
            ls = std::make_unique<LSystem>(axiom, weightedRules);
        } else {
            ls = std::make_unique<LSystem>(axiom, detRules);
        }

        // Optional per-lsystem seed for stochastic rule selection. 0 = legacy deterministic.
        uint32_t seed = 0;
        if (lsNode.isMember("seed")) lsNode["seed"] >> seed;
        ls->setSeed(seed);

        registry.addLSystem(name, std::move(ls));
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

        // Optional colonization block — absent means SC stays off (attractorCount=0 default).
        if (aNode.isMember("colonization")) {
            const auto&        cNode = aNode["colonization"];
            ProcGen::SCParams  scp;
            cNode["attractorCount"]     >> scp.attractorCount;
            cNode["influenceDistance"]  >> scp.influenceDistance;
            cNode["killDistance"]       >> scp.killDistance;
            cNode["segmentLength"]      >> scp.segmentLength;
            cNode["maxIterations"]      >> scp.maxIterations;
            cNode["crownRadiusFactor"]  >> scp.crownRadiusFactor;
            cNode["crownYOffsetFactor"] >> scp.crownYOffsetFactor;
            cNode["branchletRadius"]    >> scp.branchletRadius;
            cNode["branchletTaper"]     >> scp.branchletTaper;
            cNode["seed"]               >> scp.seed;

            if (auto* a = registry.findAsset(stableId)) a->colonization = scp;
        }

        // Optional leaf block — fields default from LeafParams when absent.
        if (aNode.isMember("leaf")) {
            const auto& lNode = aNode["leaf"];
            LeafParams  lp;
            // Default values come from LeafParams's struct defaults; only overwrite what's present.
            if (auto* a = registry.findAsset(stableId)) lp = a->leaf;

            if (lNode.isMember("color") && lNode["color"].isArray() && lNode["color"].size() == 3) {
                lp.color.x = lNode["color"][0].asFloat();
                lp.color.y = lNode["color"][1].asFloat();
                lp.color.z = lNode["color"][2].asFloat();
            }
            lNode["size"]    >> lp.size;
            lNode["perTip"]  >> lp.perTip;
            if (lNode.isMember("lodMultipliers") && lNode["lodMultipliers"].isArray()) {
                const auto& arr = lNode["lodMultipliers"];
                for (Json::ArrayIndex i = 0; i < arr.size() && i < lp.lodMultipliers.size(); ++i)
                    lp.lodMultipliers[i] = arr[i].asFloat();
            }

            if (auto* a = registry.findAsset(stableId)) a->leaf = lp;
        }

        // Optional master switch — defaults to true (TreeAssetDef::hasLeaves = true).
        if (aNode.isMember("hasLeaves")) {
            bool hasLeaves = true;
            aNode["hasLeaves"] >> hasLeaves;
            if (auto* a = registry.findAsset(stableId)) a->hasLeaves = hasLeaves;
        }
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
