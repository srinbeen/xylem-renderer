#include "include/SceneLoader.hpp"

#include <algorithm>
#include <fstream>

#include <donut/core/json.h>

using namespace Xylem;
using namespace Xylem::Scene;
using namespace Xylem::ProcGen;

// static
bool SceneLoader::Load(
    const std::filesystem::path& path,
    nvrhi::IDevice*              device,
    nvrhi::ICommandList*         commandList,
    SceneData&                   out)
{
    const Json::Value root = ParseFile(path);

    // -------------------------------------------------------------------------
    // LODs
    // -------------------------------------------------------------------------
    if (root.isMember("lods") && root["lods"].isArray()) {
        for (const auto& lodNode : root["lods"]) {
            uint32_t segments = 8;
            float    distance = 100.f;
            lodNode["segments"] >> segments;
            lodNode["distance"] >> distance;
            out.lodSegments.push_back(segments);
            out.lodDistances.push_back(distance);
        }
    } else {
        out.lodSegments  = { 16, 8, 4 };
        out.lodDistances = { 16.0f, 64.0f, 256.0f };
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

        out.lsystems.try_emplace(name, std::make_unique<LSystem>(axiom, rules));
    }
    if (out.lsystems.empty()) return false;

    // -------------------------------------------------------------------------
    // Assets
    // -------------------------------------------------------------------------
    out.treeGenerator = std::make_unique<TreeGenerator>();

    for (const auto& aNode : root["assets"]) {
        TreeAsset asset;
        std::string lsName = out.lsystems.begin()->first;
        uint32_t    gen    = 3;

        aNode["name"]    >> asset.name;
        aNode["lsystem"] >> lsName;
        aNode["generation"] >> gen;
        asset.lsystemInstance = { lsName, gen };

        asset.generatorParams.radialSegments = 16;
        asset.generatorParams.stepLength     = 1.0f;
        asset.generatorParams.branchAngle    = 25.0f;
        asset.generatorParams.taperRatio     = 0.9f;
        asset.generatorParams.stepRatio      = 0.95f;
        asset.generatorParams.seed           = 0;

        aNode["radialSegments"] >> asset.generatorParams.radialSegments;
        aNode["stepLength"]     >> asset.generatorParams.stepLength;
        aNode["branchAngle"]    >> asset.generatorParams.branchAngle;
        aNode["taperRatio"]     >> asset.generatorParams.taperRatio;
        aNode["stepRatio"]      >> asset.generatorParams.stepRatio;
        aNode["seed"]           >> asset.generatorParams.seed;

        asset.barkTexture = "bark_willow_02_1k";
        aNode["barkTexture"] >> asset.barkTexture;

        // Assign texture set index — deduplicate by folder name
        auto tsIt = std::find(out.barkTextureSets.begin(), out.barkTextureSets.end(), asset.barkTexture);
        if (tsIt == out.barkTextureSets.end()) {
            asset.textureSetIdx = static_cast<uint32_t>(out.barkTextureSets.size());
            out.barkTextureSets.push_back(asset.barkTexture);
        } else {
            asset.textureSetIdx = static_cast<uint32_t>(std::distance(out.barkTextureSets.begin(), tsIt));
        }

        asset.generatorParams.branchAngle = dm::radians(asset.generatorParams.branchAngle);

        auto lsIt = out.lsystems.find(lsName);
        if (lsIt == out.lsystems.end()) continue;
        lsIt->second->reset();
        lsIt->second->generate(gen);
        asset.lsystemString = lsIt->second->getCurrentString();

        asset.lods.resize(out.lodSegments.size());
        _BuildTreeAssetBuffers(asset, out.lodSegments, *out.treeGenerator, device, commandList);
        out.assets.push_back(std::move(asset));
    }

    // -------------------------------------------------------------------------
    // Regions
    // -------------------------------------------------------------------------
    std::unordered_map<std::string, uint32_t> assetNameToIdx;
    for (uint32_t i = 0; i < out.assets.size(); i++)
        assetNameToIdx[out.assets[i].name] = i;

    for (const auto& rNode : root["regions"]) {
        uint32_t count = 10;
        float    minX = 0.f, minY = 0.f, maxX = 10.f, maxY = 10.f;
        std::string key = "Region";

        rNode["instanceCount"] >> count;
        rNode["boundsMinX"]    >> minX;
        rNode["boundsMinY"]    >> minY;
        rNode["boundsMaxX"]    >> maxX;
        rNode["boundsMaxY"]    >> maxY;
        rNode["name"]          >> key;

        dm::box2 bounds(dm::float2(minX, minY), dm::float2(maxX, maxY));
        out.regionManager.addRegion(key, count, bounds);
        auto& r = out.regionManager[key];

        if (rNode.isMember("assets") && rNode["assets"].size() > 0) {
            r.assetIndices.clear();
            for (const auto& aName : rNode["assets"]) {
                std::string n = aName.asString();
                if (assetNameToIdx.count(n)) r.assetIndices.push_back(assetNameToIdx[n]);
            }
            out.regionManager.updateRegion(out.regionManager.size() - 1, out.assets);
        }
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

// static
void SceneLoader::_BuildTreeAssetBuffers(
    TreeAsset&                   asset,
    const std::vector<uint32_t>& lodSegments,
    TreeGenerator&               generator,
    nvrhi::IDevice*              device,
    nvrhi::ICommandList*         commandList)
{
    nvrhi::BufferDesc vDesc;
    vDesc.isVertexBuffer = true;
    vDesc.initialState   = nvrhi::ResourceStates::CopyDest;

    nvrhi::BufferDesc iDesc;
    iDesc.isIndexBuffer = true;
    iDesc.initialState  = nvrhi::ResourceStates::CopyDest;

    auto& genParams = asset.generatorParams;
    for (size_t j = 0; j < lodSegments.size(); j++) {
        genParams.radialSegments = lodSegments[j];
        generator.setParams(genParams);

        Buffers lod;
        generator.generateVertexAndIndexBuffers(asset.lsystemString, lod);

        auto& vBuf = asset.lods[j].vertexBuffer;
        auto& iBuf = asset.lods[j].indexBuffer;

        vDesc.debugName = "VB_" + asset.name + "_LOD" + std::to_string(j);
        vDesc.byteSize  = lod.vertices.size() * sizeof(TreeVertex);
        vBuf = device->createBuffer(vDesc);
        commandList->beginTrackingBufferState(vBuf, nvrhi::ResourceStates::CopyDest);
        commandList->writeBuffer(vBuf, lod.vertices.data(), vDesc.byteSize);
        commandList->setPermanentBufferState(vBuf, nvrhi::ResourceStates::VertexBuffer);

        iDesc.debugName = "IB_" + asset.name + "_LOD" + std::to_string(j);
        iDesc.byteSize  = lod.indices.size() * sizeof(uint32_t);
        iBuf = device->createBuffer(iDesc);
        commandList->beginTrackingBufferState(iBuf, nvrhi::ResourceStates::CopyDest);
        commandList->writeBuffer(iBuf, lod.indices.data(), iDesc.byteSize);
        commandList->setPermanentBufferState(iBuf, nvrhi::ResourceStates::IndexBuffer);

        asset.lods[j].indexCount     = static_cast<uint32_t>(lod.indices.size());
        asset.lods[j].radialSegments = genParams.radialSegments;
        asset.lods[j].bbox           = lod.bbox;
    }
}
