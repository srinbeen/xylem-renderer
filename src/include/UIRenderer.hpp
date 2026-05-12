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
#include "SceneLoader.hpp"
#include <donut/app/ApplicationBase.h>
#include "Globals.hpp"

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
        ProcGen::SCParams              colonization;
        LeafParams                     leaf;
        bool                           hasLeaves  = true;
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

    // Scene file path text field
    std::string m_ScenePath;
    std::string m_BenchmarkNameField;       // backing for Name InputText
    int         m_BenchmarkActiveIdx = 0;   // backing for Combo
    std::string m_LastSyncedActiveName;     // change-detect for Name auto-sync

public:
    UIRenderer(app::DeviceManager* dm, SceneRegistry* registry, UIData& ui, ViewHandler* vh)
        : ImGui_Renderer(dm), m_Registry(registry), m_ui(ui), m_ViewHandler(vh)
    {
        ImGui::GetIO().IniFilename = nullptr;
        m_ScenePath = g_SceneConfigDirectory.string();
    }

    void Init(std::shared_ptr<engine::ShaderFactory> sf) { ImGui_Renderer::Init(sf); }

    bool KeyboardUpdate(int key, int scancode, int action, int mods);

protected:
    void buildUI();

private:
    void _buildSceneFileSection();
    void _buildRuntimeSettingsSection();
    void _buildBenchmarkSection();
    void _buildLSystemsSection();
    void _buildAssetsSection();
    void _buildRegionsSection();
    void _buildTreesFunnel();
    void _buildLeavesFunnel();
    void _buildTerrainFunnel();
    void _buildDebugTopDownSection();
    void _buildDebugShadowTopDownSection();
    void _buildShadowMapSection();
    void _buildHiZSection();
    void _buildImpostorAtlasSection();

    // Format a uint32 with thousands grouping ("12,345"). Returns a
    // pointer into a thread-local buffer; valid only until the next call
    // on the same thread. Used inline in funnel rows.
    static const char* _fmtCount(uint64_t v);

    // Shared top-down canvas context — world XZ <-> screen pixel transform.
    struct TopDownCanvas {
        ImDrawList* dl = nullptr;
        ImVec2      canvasPos{};
        ImVec2      canvasSize{};
        float       canvasCenterX = 0.f, canvasCenterY = 0.f;
        float       worldCenterX  = 0.f, worldCenterZ  = 0.f;
        float       scale         = 1.f;

        ImVec2 toScreen(float wx, float wz) const {
            return ImVec2(canvasCenterX + (wx - worldCenterX) * scale,
                          canvasCenterY + (wz - worldCenterZ) * scale);
        }
    };

    // Opens an ImGui window, computes a top-down XZ transform from scene bounds,
    // reserves canvas, fills background. Returns true if the caller should draw;
    // in that case the caller owns the matching ImGui::End().
    bool _beginTopDown(const char* title, bool* show, const char* canvasId,
                       TopDownCanvas& out);
};

} // namespace Xylem

#endif // XYLEM_UI_RENDER_PASS_H
