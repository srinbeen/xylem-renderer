#ifndef XYLEM_UI_RENDERER_H
#define XYLEM_UI_RENDERER_H

#include <string>
#include <unordered_map>
#include <vector>

#include <donut/app/imgui_renderer.h>
#include <misc/cpp/imgui_stdlib.h>

#include "UIData.hpp"
#include "SceneRegistry.hpp"
#include "ViewHandler.hpp"

using namespace donut;

namespace Xylem {

class UIRenderer : public app::ImGui_Renderer {
private:
    SceneRegistry* m_Registry;
    UIData&        m_ui;
    ViewHandler*   m_ViewHandler;

    // -----------------------------------------------------------------------
    // Transient edit state — not persisted, only alive while UI is open
    // -----------------------------------------------------------------------

    // L-Systems
    struct LSystemEditState {
        bool        open  = false;
        std::string axiom;
        std::string rules; // one "K=VALUE" per line
    };

    // Assets
    struct AssetEditState {
        bool                           open       = false;
        int                            pendingGen = 3;
        ProcGen::TreeGenerator::Params params;
    };

    struct AddAssetState {
        bool                           open       = false;
        std::string                    name;
        std::string                    bark       = "bark_willow_02_1k";
        int                            gen        = 3;
        int                            lsystemIdx = 0;
        ProcGen::TreeGenerator::Params params;
    };

    // Regions
    struct RegionEditState {
        bool     open             = false;
        float    pendingDensity   = 0.02f;
        float    bounds[4]        = {};  // minX, minY, maxX, maxY
        std::vector<uint8_t> assetChecked;  // indexed by m_Registry->getAssets()
    };

    struct AddRegionState {
        bool        open      = false;
        std::string name;
        float       density   = 0.02f;
        float       bounds[4] = { -20.f, -20.f, 20.f, 20.f };
        std::vector<uint8_t> assetChecked;
    };


    
    std::unordered_map<std::string, LSystemEditState> m_LsEditState;
    std::unordered_map<size_t,      AssetEditState>   m_AssetEditState;
    std::unordered_map<size_t,      RegionEditState>  m_RegionEditState;
    AddAssetState  m_AddAsset;
    AddRegionState m_AddRegion;

public:
    UIRenderer(app::DeviceManager* dm, SceneRegistry* registry, UIData& ui, ViewHandler* vh)
        : ImGui_Renderer(dm), m_Registry(registry), m_ui(ui), m_ViewHandler(vh)
    {
        ImGui::GetIO().IniFilename = nullptr;
    }

    void Init(std::shared_ptr<engine::ShaderFactory> sf) { ImGui_Renderer::Init(sf); }

    bool KeyboardUpdate(int key, int scancode, int action, int mods);

protected:
    void buildUI();

private:
    void _buildLSystemsSection();
    void _buildAssetsSection();
    void _buildRegionsSection();
    void _buildDebugTopDownSection();
    void _buildShadowMapSection();
};

} // namespace Xylem

#endif // XYLEM_UI_RENDER_PASS_H
