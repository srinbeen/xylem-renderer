#include "include/UIRenderer.hpp"

#include <donut/core/math/math.h>
#include <sstream>
#include <string>

using namespace Xylem;

bool UIRenderer::KeyboardUpdate(int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        m_ui.ShowUI = !m_ui.ShowUI;
        return true;
    }
    return ImGui_Renderer::KeyboardUpdate(key, scancode, action, mods);
}

// ---------------------------------------------------------------------------
// Top-level buildUI
// ---------------------------------------------------------------------------

void UIRenderer::buildUI() {
    if (!m_ui.ShowUI) return;

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
    ImGui::Begin("Xylem", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    // =====================================================================
    // PERFORMANCE METRICS
    // =====================================================================
    if (ImGui::CollapsingHeader("Performance Metrics", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Renderer: %s", GetDeviceManager()->GetRendererString());

        double ft = GetDeviceManager()->GetAverageFrameTimeSeconds();
        if (ft > 0.0)
            ImGui::Text("CPU frame:  %.2f ms  (%.0f FPS)", ft * 1e3, 1.0 / ft);

        ImGui::Text("CPU render: %.2f ms", m_ui.cpuRenderTimeMs);

        if (m_ui.gpuFrameTimeMs >= 0.f)
            ImGui::Text("GPU pass:   %.2f ms", m_ui.gpuFrameTimeMs);
        else
            ImGui::TextDisabled("GPU pass:   (pending)");

        ImGui::Separator();
        ImGui::Text("Instances  visible: %u / %u  (culled: %u)",
            m_ui.visibleInstanceCount, m_ui.totalInstanceCount, m_ui.culledInstanceCount);
        ImGui::Text("Shadow     visible: %u / %u  (culled: %u)",
            m_ui.shadowVisibleCount, m_ui.totalInstanceCount, m_ui.shadowCulledCount);
        ImGui::Text("Draw calls: %u", m_ui.drawCallCount);
    }

    _buildLSystemsSection();
    _buildAssetsSection();
    _buildRegionsSection();

    ImGui::Spacing();
    if (ImGui::Button("Hide UI  [ESC]")) m_ui.ShowUI = false;

    ImGui::End();
}

// ---------------------------------------------------------------------------
// L-Systems section
// ---------------------------------------------------------------------------

void UIRenderer::_buildLSystemsSection() {
    if (!ImGui::CollapsingHeader("L-Systems")) return;

    for (auto& [lsName, lsPtr] : m_Registry->getLSystems()) {
        ImGui::PushID(lsName.c_str());

        auto& state = m_LsEditState[lsName];

        if (ImGui::TreeNode(lsName.c_str())) {
            // Read-only summary
            ImGui::TextDisabled("Axiom: %s", lsPtr->getAxiomStr().c_str());
            for (const auto& [ch, rule] : lsPtr->getRulesStr())
                ImGui::TextDisabled("  %c = %s", ch, rule.c_str());

            if (!state.open) {
                if (ImGui::Button("Edit")) {
                    state.open  = true;
                    state.axiom = lsPtr->getAxiomStr();

                    state.rules.clear();
                    for (const auto& [ch, rule] : lsPtr->getRulesStr())
                        state.rules += ch + std::string("=") + rule + "\n";
                }
            } else {
                ImGui::InputText("Axiom", &state.axiom);
                ImGui::InputTextMultiline("Rules", &state.rules, ImVec2(-1, 80));
                ImGui::TextDisabled("Format: one rule per line  X=F[+X]-X");

                if (ImGui::Button("Apply")) {
                    state.open = false;

                    // Parse rules from text
                    std::unordered_map<char, std::string> newRules;
                    std::istringstream ss(state.rules);
                    std::string line;
                    while (std::getline(ss, line)) {
                        if (line.size() >= 3 && line[1] == '=')
                            newRules[line[0]] = line.substr(2);
                    }

                    lsPtr->setAxiom(state.axiom);
                    lsPtr->setRules(newRules);

                    // Mark all m_Registry->getAssets() referencing this L-System dirty
                    for (const auto& asset : m_Registry->getAssets()) {
                        if (asset.lsystemInstance.name == lsName)
                            m_Registry->modifyAsset(asset.id, asset.genParams);
                    }
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel"))
                    state.open = false;
            }

            ImGui::TreePop();
        }

        ImGui::PopID();
    }
}

// ---------------------------------------------------------------------------
// Assets section
// ---------------------------------------------------------------------------

void UIRenderer::_buildAssetsSection() {
    if (!ImGui::CollapsingHeader("Tree Assets")) return;

    // Collect asset IDs first (removeAsset mutates the vector mid-iteration)
    std::vector<size_t> assetIds;
    for (const auto& a : m_Registry->getAssets())
        assetIds.push_back(a.id);

    for (size_t id : assetIds) {
        const auto* asset = m_Registry->findAsset(id);
        if (!asset) continue;

        ImGui::PushID(("asset_" + std::to_string(id)).c_str());

        auto& state = m_AssetEditState[id];

        // Visibility checkbox before the tree node
        bool assetVis = asset->visible;
        if (ImGui::Checkbox("##vis", &assetVis))
            m_Registry->setAssetVisible(id, assetVis);
        ImGui::SameLine();

        bool headerOpen = ImGui::TreeNode(asset->name.c_str());

        // Remove button on the same line as the header
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.1f, 0.1f, 1.f));
        if (ImGui::SmallButton("Remove")) {
            m_AssetEditState.erase(id);
            m_Registry->removeAsset(id);
            ImGui::PopStyleColor();
            if (headerOpen) ImGui::TreePop();
            ImGui::PopID();
            continue;
        }
        ImGui::PopStyleColor();

        if (headerOpen) {
            ImGui::TextDisabled("L-System: %s  gen=%u",
                asset->lsystemInstance.name.c_str(), asset->lsystemInstance.gen);
            ImGui::TextDisabled("step=%.2f  angle=%.1f deg  taper=%.2f  stepR=%.2f  seed=%u",
                asset->genParams.stepLength,
                dm::degrees(asset->genParams.branchAngle),
                asset->genParams.taperRatio,
                asset->genParams.stepRatio,
                asset->genParams.seed);

            if (!state.open) {
                if (ImGui::Button("Edit")) {
                    state.open       = true;
                    state.pendingGen = static_cast<int>(asset->lsystemInstance.gen);
                    state.params     = asset->genParams;
                }
            } else {
                float branchAngleDeg = dm::degrees(state.params.branchAngle);
                int   seed           = static_cast<int>(state.params.seed);

                ImGui::SliderInt("Generation",   &state.pendingGen,         1,    8);
                ImGui::SliderFloat("Step Length",  &state.params.stepLength, 0.1f, 5.0f);
                ImGui::SliderFloat("Branch Angle", &branchAngleDeg,          5.0f, 90.f);
                ImGui::SliderFloat("Taper Ratio",  &state.params.taperRatio, 0.5f, 1.0f);
                ImGui::SliderFloat("Step Ratio",   &state.params.stepRatio,  0.5f, 1.0f);
                ImGui::InputInt("Seed (0=none)",   &seed);

                state.params.branchAngle = dm::radians(branchAngleDeg);
                state.params.seed        = static_cast<uint32_t>(seed);

                if (ImGui::Button("Apply")) {
                    state.open = false;

                    if (static_cast<uint32_t>(state.pendingGen) != asset->lsystemInstance.gen) {
                        auto* mutableAsset = m_Registry->findAsset(id);
                        mutableAsset->lsystemInstance.gen = static_cast<uint32_t>(state.pendingGen);
                    }

                    m_Registry->modifyAsset(id, state.params);
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel"))
                    state.open = false;
            }

            ImGui::TreePop();
        }

        ImGui::PopID();
    }

    ImGui::Separator();

    // -- Add Asset ----------------------------------------------------------
    if (!m_AddAsset.open) {
        if (ImGui::Button("+ Add Asset")) {
            m_AddAsset = AddAssetState{};
            m_AddAsset.open = true;
        }
    }

    if (m_AddAsset.open) {
        ImGui::Indent();
        ImGui::InputText("Name##add",         &m_AddAsset.name);
        ImGui::InputText("Bark Texture##add", &m_AddAsset.bark);

        // L-System combo
        const auto& lsystems = m_Registry->getLSystems();
        std::vector<const char*> lsNames;
        for (const auto& [n, _] : lsystems)
            lsNames.push_back(n.c_str());

        if (!lsNames.empty()) {
            if (m_AddAsset.lsystemIdx >= static_cast<int>(lsNames.size()))
                m_AddAsset.lsystemIdx = 0;
            ImGui::Combo("L-System##add", &m_AddAsset.lsystemIdx, lsNames.data(),
                static_cast<int>(lsNames.size()));
        }

        float addBranchAngleDeg = dm::degrees(m_AddAsset.params.branchAngle);
        int   addSeed           = static_cast<int>(m_AddAsset.params.seed);

        ImGui::SliderInt("Generation##add",     &m_AddAsset.gen,                  1,    8);
        ImGui::SliderFloat("Step Length##add",  &m_AddAsset.params.stepLength,    0.1f, 5.0f);
        ImGui::SliderFloat("Branch Angle##add", &addBranchAngleDeg,               5.0f, 90.f);
        ImGui::SliderFloat("Taper Ratio##add",  &m_AddAsset.params.taperRatio,    0.5f, 1.0f);
        ImGui::SliderFloat("Step Ratio##add",   &m_AddAsset.params.stepRatio,     0.5f, 1.0f);
        ImGui::InputInt("Seed (0=none)##add",   &addSeed);

        m_AddAsset.params.branchAngle = dm::radians(addBranchAngleDeg);
        m_AddAsset.params.seed        = static_cast<uint32_t>(addSeed);

        if (ImGui::Button("Add") && !m_AddAsset.name.empty() && !lsNames.empty()) {
            Scene::LSystemInstance lsInst { lsNames[m_AddAsset.lsystemIdx],
                                             static_cast<uint32_t>(m_AddAsset.gen) };
            m_Registry->addAsset(m_AddAsset.name, lsInst, m_AddAsset.params, m_AddAsset.bark);
            m_AddAsset.open = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel##add"))
            m_AddAsset.open = false;

        ImGui::Unindent();
    }
}

// ---------------------------------------------------------------------------
// Regions section
// ---------------------------------------------------------------------------

void UIRenderer::_buildRegionsSection() {
    if (!ImGui::CollapsingHeader("Regions")) return;

    size_t regionCount = m_Registry->getRegions().size();
    for (size_t ri = 0; ri < regionCount; ri++) {
        // Re-fetch each iteration — removeRegion may have shifted indices
        if (ri >= m_Registry->getRegions().size()) break;
        const auto& region = m_Registry->getRegions()[ri];

        ImGui::PushID(("region_" + std::to_string(ri)).c_str());

        auto& state = m_RegionEditState[ri];

        bool headerOpen = ImGui::TreeNode(region.name.c_str());

        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.1f, 0.1f, 1.f));
        if (ImGui::SmallButton("Remove")) {
            m_RegionEditState.erase(ri);
            m_Registry->removeRegion(ri);
            ImGui::PopStyleColor();
            if (headerOpen) ImGui::TreePop();
            ImGui::PopID();
            regionCount--;
            ri--;
            continue;
        }
        ImGui::PopStyleColor();

        if (headerOpen) {
            ImGui::TextDisabled("Instances: %u  Density: %.4f", region.instanceCount, region.density);
            ImGui::TextDisabled("Bounds: [%.1f, %.1f] -> [%.1f, %.1f]",
                region.bounds.m_mins.x, region.bounds.m_mins.y,
                region.bounds.m_maxs.x, region.bounds.m_maxs.y);

            // Per-asset visibility within this region
            if (!region.assetIds.empty()) {
                ImGui::Spacing();
                ImGui::Text("Asset visibility:");
                ImGui::SameLine();
                ImGui::PushID("region_vis_buttons");
                if (ImGui::SmallButton("All")) {
                    for (size_t aid : region.assetIds)
                        m_Registry->setRegionAssetVisible(ri, aid, true);
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("None")) {
                    for (size_t aid : region.assetIds)
                        m_Registry->setRegionAssetVisible(ri, aid, false);
                }
                ImGui::PopID();

                ImGui::PushID("region_vis_checks");
                for (size_t aid : region.assetIds) {
                    const auto* a = m_Registry->findAsset(aid);
                    if (!a) continue;
                    auto visIt = region.assetVisible.find(aid);
                    bool vis = (visIt == region.assetVisible.end()) ? true : visIt->second;
                    ImGui::PushID(static_cast<int>(aid));
                    if (ImGui::Checkbox(a->name.c_str(), &vis))
                        m_Registry->setRegionAssetVisible(ri, aid, vis);
                    ImGui::PopID();
                }
                ImGui::PopID();
            }

            ImGui::Separator();

            if (!state.open) {
                if (ImGui::Button("Edit")) {
                    state.open              = true;
                    state.pendingDensity    = region.density;
                    state.bounds[0]         = region.bounds.m_mins.x;
                    state.bounds[1]         = region.bounds.m_mins.y;
                    state.bounds[2]         = region.bounds.m_maxs.x;
                    state.bounds[3]         = region.bounds.m_maxs.y;

                    // Build checkbox state from current assetIds
                    state.assetChecked.assign(m_Registry->getAssets().size(), false);
                    for (size_t ai = 0; ai < m_Registry->getAssets().size(); ai++) {
                        for (size_t aid : region.assetIds)
                            if (m_Registry->getAssets()[ai].id == aid) { state.assetChecked[ai] = true; break; }
                    }
                }
            } else {
                ImGui::InputFloat("Density##r",  &state.pendingDensity, 0.001f, 0.01f, "%.4f");
                ImGui::DragFloat2("Bounds Min", state.bounds,   1.f);
                ImGui::DragFloat2("Bounds Max", state.bounds+2, 1.f);

                // Ensure checkbox array stays in sync if m_Registry->getAssets() were added/removed
                state.assetChecked.resize(m_Registry->getAssets().size(), false);
                ImGui::Text("Assets:");
                ImGui::PushID("edit_m_Registry->getAssets()");
                for (size_t ai = 0; ai < m_Registry->getAssets().size(); ai++) {
                    ImGui::PushID(static_cast<int>(ai));
                    ImGui::Checkbox(m_Registry->getAssets()[ai].name.c_str(), reinterpret_cast<bool*>(&state.assetChecked[ai]));
                    ImGui::PopID();
                }
                ImGui::PopID();

                if (ImGui::Button("Apply")) {
                    state.open = false;

                    dm::box2 newBounds(
                        dm::float2(state.bounds[0], state.bounds[1]),
                        dm::float2(state.bounds[2], state.bounds[3]));
                    m_Registry->modifyRegion(ri, state.pendingDensity, newBounds);

                    std::vector<size_t> newIds;
                    for (size_t ai = 0; ai < m_Registry->getAssets().size() && ai < state.assetChecked.size(); ai++)
                        if (state.assetChecked[ai]) newIds.push_back(m_Registry->getAssets()[ai].id);
                    m_Registry->modifyRegionAssets(ri, newIds);
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel"))
                    state.open = false;
            }

            ImGui::TreePop();
        }

        ImGui::PopID();
    }

    ImGui::Separator();

    // -- Add Region --------------------------------------------------------
    if (!m_AddRegion.open) {
        if (ImGui::Button("+ Add Region")) {
            m_AddRegion = AddRegionState{};
            m_AddRegion.open = true;
            m_AddRegion.assetChecked.assign(m_Registry->getAssets().size(), false);
        }
    }

    if (m_AddRegion.open) {
        ImGui::Indent();
        ImGui::InputText("Name##radd", &m_AddRegion.name);
        ImGui::InputFloat("Density##radd",  &m_AddRegion.density, 0.001f, 0.01f, "%.4f");
        ImGui::DragFloat2("Bounds Min##radd", m_AddRegion.bounds,   1.f);
        ImGui::DragFloat2("Bounds Max##radd", m_AddRegion.bounds+2, 1.f);

        m_AddRegion.assetChecked.resize(m_Registry->getAssets().size(), false);
        ImGui::Text("Assets:");
        ImGui::PushID("add_m_Registry->getAssets()");
        for (size_t ai = 0; ai < m_Registry->getAssets().size(); ai++) {
            ImGui::PushID(static_cast<int>(ai));
            ImGui::Checkbox(m_Registry->getAssets()[ai].name.c_str(), reinterpret_cast<bool*>(&m_AddRegion.assetChecked[ai]));
            ImGui::PopID();
        }
        ImGui::PopID();

        if (ImGui::Button("Add##radd") && !m_AddRegion.name.empty()) {
            dm::box2 newBounds(
                dm::float2(m_AddRegion.bounds[0], m_AddRegion.bounds[1]),
                dm::float2(m_AddRegion.bounds[2], m_AddRegion.bounds[3]));

            std::vector<size_t> selectedIds;
            for (size_t ai = 0; ai < m_Registry->getAssets().size() && ai < m_AddRegion.assetChecked.size(); ai++)
                if (m_AddRegion.assetChecked[ai]) selectedIds.push_back(m_Registry->getAssets()[ai].id);

            m_Registry->addRegion(m_AddRegion.name, m_AddRegion.density, newBounds, selectedIds);
            m_AddRegion.open = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel##radd"))
            m_AddRegion.open = false;

        ImGui::Unindent();
    }
}
