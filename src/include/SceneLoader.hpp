#ifndef XYLEM_SCENE_LOADER_H
#define XYLEM_SCENE_LOADER_H

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nvrhi/nvrhi.h>
#include <json/json.h>

#include "Scene.hpp"
#include "RegionManager.hpp"
#include "Procgen.hpp"

namespace Xylem {

// Output of a successful scene load — everything the render pass needs.
struct SceneData {
    std::map<std::string, std::unique_ptr<ProcGen::LSystem>> lsystems;
    std::unique_ptr<ProcGen::TreeGenerator>                  treeGenerator;
    std::vector<Scene::TreeAsset>                            assets;
    Scene::RegionManager                                     regionManager;
    std::vector<uint32_t>                                    lodSegments;
    std::vector<float>                                       lodDistances;
};

class SceneLoader {
public:
    // Parses the JSON file and uploads GPU buffers via commandList.
    // Returns false if the scene is invalid (no L-systems, missing assets, etc.).
    static bool Load(
        const std::filesystem::path& path,
        nvrhi::IDevice*              device,
        nvrhi::ICommandList*         commandList,
        SceneData&                   out);

    static Json::Value ParseFile(const std::filesystem::path& path);

private:
    static void _BuildTreeAssetBuffers(
        Scene::TreeAsset&            asset,
        const std::vector<uint32_t>& lodSegments,
        ProcGen::TreeGenerator&      generator,
        nvrhi::IDevice*              device,
        nvrhi::ICommandList*         commandList);
};

} // namespace Xylem

#endif // XYLEM_SCENE_LOADER_H
