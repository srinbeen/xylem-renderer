#ifndef XYLEM_SCENE_LOADER_H
#define XYLEM_SCENE_LOADER_H

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <json/json.h>

#include "SceneRegistry.hpp"
#include "Procgen.hpp"
#include "Terrain.hpp"

namespace Xylem {

class SceneLoader {
public:
    // CPU-only scene load — populates the SceneRegistry, no GPU work.
    // Returns false if the scene is invalid (no L-systems, missing assets, etc.).
    static bool Load(const std::filesystem::path& path, SceneRegistry& registry);

    // CPU-only scene save — writes the loader-shaped JSON subset to `path`.
    // Returns false on any I/O error. Atomic: writes to <path>.tmp then renames.
    static bool Save(const std::filesystem::path& path, const SceneRegistry& registry);

    static Json::Value ParseFile(const std::filesystem::path& path);
};

} // namespace Xylem

#endif // XYLEM_SCENE_LOADER_H
