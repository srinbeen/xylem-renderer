#include "include/SceneLoader.hpp"
#include "include/Terrain.hpp"

#include <algorithm>
#include <fstream>

#include <donut/core/json.h>
#include <donut/core/log.h>

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
        params.baseLength     = 1.0f;
        params.baseRadius     = 1.0f;
        params.branchAngle    = 25.0f;
        params.taperRatio     = 0.9f;
        params.stepRatio      = 0.95f;
        params.seed           = 0;

        aNode["radialSegments"] >> params.radialSegments;
        aNode["baseLength"]     >> params.baseLength;
        aNode["baseRadius"]     >> params.baseRadius;
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
            cNode["branchletTaper"]     >> scp.branchletTaper;
            cNode["seed"]               >> scp.seed;

            if (cNode.isMember("crown")) {
                const auto& crownNode = cNode["crown"];
                crownNode["translation"] >> scp.crownTranslation;
                crownNode["rotation"]    >> scp.crownRotationDegrees;
                crownNode["scale"]       >> scp.crownScale;
                crownNode["shear"]       >> scp.crownShear;
            }

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
            // `lodMultipliers` was a per-LOD count thinning knob in earlier versions;
            // dropped from LeafParams. Tolerate (ignore) the field if it appears in
            // older scene files so they still load.

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

            if (tNode.isMember("largeScale")) {
                const auto& lsNode = tNode["largeScale"];
                if (lsNode.isMember("amplitude")) lsNode["amplitude"] >> terrainConfig.largeScaleAmp;
                if (lsNode.isMember("periodX"))   lsNode["periodX"]   >> terrainConfig.largeScalePeriodX;
                if (lsNode.isMember("periodZ"))   lsNode["periodZ"]   >> terrainConfig.largeScalePeriodZ;
                if (lsNode.isMember("phaseX"))    lsNode["phaseX"]    >> terrainConfig.largeScalePhaseX;
                if (lsNode.isMember("phaseZ"))    lsNode["phaseZ"]    >> terrainConfig.largeScalePhaseZ;
            }

            // Required: textures block with all four named slots.
            if (!tNode.isMember("textures")) {
                donut::log::error("SceneLoader: terrain block missing required 'textures' sub-block");
                return false;
            }
            const auto& texNode = tNode["textures"];
            const char* slotKeys[4] = { "forestFloor", "dirt", "rock", "snow" };
            for (int i = 0; i < 4; i++) {
                if (!texNode.isMember(slotKeys[i]) || !texNode[slotKeys[i]].isString()) {
                    donut::log::error("SceneLoader: terrain.textures missing or non-string slot '%s'",
                                      slotKeys[i]);
                    return false;
                }
                terrainConfig.shading.textureSetNames[i] = texNode[slotKeys[i]].asString();
            }

            // Optional: shading block — per-field defaults.
            if (tNode.isMember("shading")) {
                const auto& sNode = tNode["shading"];
                if (sNode.isMember("tileSize"))      sNode["tileSize"]      >> terrainConfig.shading.tileSize;
                if (sNode.isMember("forestToDirtY")) sNode["forestToDirtY"] >> terrainConfig.shading.forestToDirtY;
                if (sNode.isMember("dirtToSnowY"))   sNode["dirtToSnowY"]   >> terrainConfig.shading.dirtToSnowY;
                if (sNode.isMember("bandWidth"))     sNode["bandWidth"]     >> terrainConfig.shading.bandWidth;
                if (sNode.isMember("slopeLo"))       sNode["slopeLo"]       >> terrainConfig.shading.slopeLo;
                if (sNode.isMember("slopeHi"))       sNode["slopeHi"]       >> terrainConfig.shading.slopeHi;
                if (sNode.isMember("macroNoiseAmp")) sNode["macroNoiseAmp"] >> terrainConfig.shading.macroNoiseAmp;
            }
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
bool SceneLoader::Save(const std::filesystem::path& path, const SceneRegistry& registry) {
    Json::Value root(Json::objectValue);

    // -------------------------------------------------------------------------
    // Sun
    // -------------------------------------------------------------------------
    {
        Json::Value sunNode(Json::objectValue);
        Json::Value dirArr(Json::arrayValue);
        const dm::float3 sun = registry.getSunDirection();
        dirArr.append(sun.x); dirArr.append(sun.y); dirArr.append(sun.z);
        sunNode["direction"] = dirArr;
        root["sun"] = sunNode;
    }

    // -------------------------------------------------------------------------
    // Camera (init values, not the live ViewHandler camera)
    // -------------------------------------------------------------------------
    {
        const auto& ci = registry.getCameraInit();
        Json::Value cam(Json::objectValue);
        Json::Value pos(Json::arrayValue);
        pos.append(ci.pos.x); pos.append(ci.pos.y); pos.append(ci.pos.z);
        Json::Value dir(Json::arrayValue);
        dir.append(ci.cameraDir.x); dir.append(ci.cameraDir.y); dir.append(ci.cameraDir.z);
        cam["position"]  = pos;
        cam["direction"] = dir;
        cam["moveSpeed"] = ci.moveSpeed;
        root["camera"] = cam;
    }

    // -------------------------------------------------------------------------
    // LODs
    // -------------------------------------------------------------------------
    {
        Json::Value lods(Json::objectValue);
        Json::Value segs(Json::arrayValue);
        for (uint32_t s : registry.getLodSegments()) segs.append(s);
        Json::Value dists(Json::arrayValue);
        for (float d : registry.getLodDistances()) dists.append(d);
        lods["radialSegments"] = segs;
        lods["distances"]      = dists;
        root["lods"] = lods;
    }

    // -------------------------------------------------------------------------
    // L-Systems
    // -------------------------------------------------------------------------
    {
        Json::Value lsArr(Json::arrayValue);
        for (const auto& [name, lsPtr] : registry.getLSystems()) {
            Json::Value ls(Json::objectValue);
            ls["name"]  = name;
            ls["axiom"] = lsPtr->getAxiomStr();

            Json::Value rules(Json::objectValue);
            if (lsPtr->isStochastic()) {
                // Weighted form: { "X": [{"weight": w, "rhs": "..."}], ... }
                for (const auto& [ch, productions] : lsPtr->getWeightedRulesStr()) {
                    Json::Value arr(Json::arrayValue);
                    for (const auto& [w, rhs] : productions) {
                        Json::Value prod(Json::objectValue);
                        prod["weight"] = w;
                        prod["rhs"]    = rhs;
                        arr.append(prod);
                    }
                    rules[std::string(1, ch)] = arr;
                }
            } else {
                // Deterministic form: { "X": "F[+X]F", ... }
                for (const auto& [ch, rhs] : lsPtr->getRulesStr())
                    rules[std::string(1, ch)] = rhs;
            }
            ls["rules"] = rules;

            if (lsPtr->getSeed() != 0)
                ls["seed"] = lsPtr->getSeed();

            lsArr.append(ls);
        }
        root["lsystems"] = lsArr;
    }

    // -------------------------------------------------------------------------
    // Assets
    // -------------------------------------------------------------------------
    {
        Json::Value assetsArr(Json::arrayValue);
        for (const auto& asset : registry.getAssets()) {
            Json::Value a(Json::objectValue);
            a["name"]           = asset.name;
            a["lsystem"]        = asset.lsystemInstance.name;
            a["generation"]     = asset.lsystemInstance.gen;
            a["radialSegments"] = asset.genParams.radialSegments;
            a["baseLength"]     = asset.genParams.baseLength;
            a["baseRadius"]     = asset.genParams.baseRadius;
            a["branchAngle"]    = dm::degrees(asset.genParams.branchAngle); // mirror loader
            a["taperRatio"]     = asset.genParams.taperRatio;
            a["stepRatio"]      = asset.genParams.stepRatio;
            a["seed"]           = asset.genParams.seed;
            a["barkTexture"]    = asset.barkTexture;

            // Optional colonization block — only when on.
            if (asset.colonization.attractorCount > 0) {
                const auto& sc = asset.colonization;
                Json::Value c(Json::objectValue);
                c["attractorCount"]     = sc.attractorCount;
                c["influenceDistance"]  = sc.influenceDistance;
                c["killDistance"]       = sc.killDistance;
                c["segmentLength"]      = sc.segmentLength;
                c["maxIterations"]      = sc.maxIterations;
                c["branchletTaper"]     = sc.branchletTaper;
                c["seed"]               = sc.seed;

                {
                    auto vec3ToArr = [](const dm::float3& v) {
                        Json::Value arr(Json::arrayValue);
                        arr.append(v.x); arr.append(v.y); arr.append(v.z);
                        return arr;
                    };
                    Json::Value crown(Json::objectValue);
                    crown["translation"] = vec3ToArr(sc.crownTranslation);
                    crown["rotation"]    = vec3ToArr(sc.crownRotationDegrees);
                    crown["scale"]       = vec3ToArr(sc.crownScale);
                    crown["shear"]       = vec3ToArr(sc.crownShear);
                    c["crown"] = crown;
                }
                a["colonization"] = c;
            }

            // Leaf block — always emitted (cheap, round-trip stable).
            {
                const auto& lp = asset.leaf;
                Json::Value leaf(Json::objectValue);
                Json::Value color(Json::arrayValue);
                color.append(lp.color.x); color.append(lp.color.y); color.append(lp.color.z);
                leaf["color"]  = color;
                leaf["size"]   = lp.size;
                leaf["perTip"] = lp.perTip;
                a["leaf"] = leaf;
            }

            // hasLeaves — only when false (default true).
            if (!asset.hasLeaves)
                a["hasLeaves"] = false;

            assetsArr.append(a);
        }
        root["assets"] = assetsArr;
    }

    // -------------------------------------------------------------------------
    // Terrain
    // -------------------------------------------------------------------------
    {
        if (const auto* terrain = registry.getTerrain()) {
            const auto& cfg = terrain->getConfig();
            Json::Value t(Json::objectValue);
            t["seed"]        = cfg.seed;
            t["octaves"]     = cfg.octaves;
            t["frequency"]   = cfg.frequency;
            t["amplitude"]   = cfg.amplitude;
            t["lacunarity"]  = cfg.lacunarity;
            t["persistence"] = cfg.persistence;
            t["gridSpacing"] = cfg.gridSpacing;
            // worldMin/Max are derived from region bounds at load time, so don't save them.

            if (cfg.largeScaleAmp != 0.f ||
                cfg.largeScalePeriodX != 0.f ||
                cfg.largeScalePeriodZ != 0.f) {
                Json::Value ls(Json::objectValue);
                ls["amplitude"] = cfg.largeScaleAmp;
                ls["periodX"]   = cfg.largeScalePeriodX;
                ls["periodZ"]   = cfg.largeScalePeriodZ;
                ls["phaseX"]    = cfg.largeScalePhaseX;
                ls["phaseZ"]    = cfg.largeScalePhaseZ;
                t["largeScale"] = ls;
            }

            // textures
            Json::Value tex(Json::objectValue);
            tex["forestFloor"] = cfg.shading.textureSetNames[0];
            tex["dirt"]        = cfg.shading.textureSetNames[1];
            tex["rock"]        = cfg.shading.textureSetNames[2];
            tex["snow"]        = cfg.shading.textureSetNames[3];
            t["textures"] = tex;

            // shading
            Json::Value shd(Json::objectValue);
            shd["tileSize"]       = cfg.shading.tileSize;
            shd["forestToDirtY"]  = cfg.shading.forestToDirtY;
            shd["dirtToSnowY"]    = cfg.shading.dirtToSnowY;
            shd["bandWidth"]      = cfg.shading.bandWidth;
            shd["slopeLo"]        = cfg.shading.slopeLo;
            shd["slopeHi"]        = cfg.shading.slopeHi;
            shd["macroNoiseAmp"]  = cfg.shading.macroNoiseAmp;
            t["shading"] = shd;

            root["terrain"] = t;
        }
    }

    // -------------------------------------------------------------------------
    // Regions
    // -------------------------------------------------------------------------
    {
        Json::Value regionsArr(Json::arrayValue);
        for (const auto& region : registry.getRegions()) {
            Json::Value r(Json::objectValue);
            r["name"]       = region.name;
            r["density"]    = region.density;
            r["boundsMinX"] = region.bounds.m_mins.x;
            r["boundsMinY"] = region.bounds.m_mins.y;
            r["boundsMaxX"] = region.bounds.m_maxs.x;
            r["boundsMaxY"] = region.bounds.m_maxs.y;

            Json::Value assetsRef(Json::arrayValue);
            for (size_t aid : region.assetIds) {
                if (const auto* a = registry.findAsset(aid))
                    assetsRef.append(a->name);
            }
            r["assets"] = assetsRef;

            regionsArr.append(r);
        }
        root["regions"] = regionsArr;
    }

    // -------------------------------------------------------------------------
    // Atomic write: <path>.tmp -> rename to <path>
    // -------------------------------------------------------------------------
    std::filesystem::path tmpPath = path;
    tmpPath += ".tmp";

    {
        std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
        if (!out) {
            donut::log::error("SceneLoader::Save: failed to open %s for writing",
                              tmpPath.string().c_str());
            return false;
        }

        Json::StreamWriterBuilder writerBuilder;
        writerBuilder["indentation"] = "  ";
        writerBuilder["precision"]   = 6;
        std::unique_ptr<Json::StreamWriter> writer(writerBuilder.newStreamWriter());
        writer->write(root, &out);
        out << '\n';
    }

    std::error_code ec;
    std::filesystem::rename(tmpPath, path, ec);
    if (ec) {
        donut::log::error("SceneLoader::Save: rename failed: %s", ec.message().c_str());
        std::filesystem::remove(tmpPath, ec); // best-effort cleanup; ignore second error
        return false;
    }

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
