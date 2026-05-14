#ifndef XYLEM_GLOBALS
#define XYLEM_GLOBALS

#include <filesystem>

namespace Xylem {
    inline const char* g_WindowTitle = "Xylem";
    inline const std::filesystem::path g_BinDirectory = donut::app::GetDirectoryWithExecutable().parent_path();
    inline const std::filesystem::path g_ProjectDirectory = g_BinDirectory.parent_path();
    inline const std::filesystem::path g_SceneConfigDirectory = g_ProjectDirectory / "scene/orchard.json";
}

#endif // XYLEM_GLOBALS