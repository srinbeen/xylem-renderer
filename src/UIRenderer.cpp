#include "include/UIRenderer.hpp"

#include <donut/core/math/math.h>
#include <filesystem>
#include <sstream>
#include <string>
#include <algorithm>

using namespace Xylem;

bool UIRenderer::KeyboardUpdate(int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        m_ui.ShowUI = !m_ui.ShowUI;
        return true;
    }
    return ImGui_Renderer::KeyboardUpdate(key, scancode, action, mods);
}


void UIRenderer::buildUI() {
    if (!m_ui.ShowUI) return;

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
    ImGui::Begin("Xylem", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    _buildSceneFileSection();

    // =====================================================================
    // PERFORMANCE METRICS
    // =====================================================================
    if (ImGui::CollapsingHeader("Performance Metrics", ImGuiTreeNodeFlags_DefaultOpen)) {
        static const char* pipelineNames[] = { "Traditional (CPU cull)", "Compute Cull (GPU cull)", "Mesh Shader (AS/MS)" };
        int pipelineIdx = static_cast<int>(m_ui.requestedPipeline);
        if (ImGui::Combo("Pipeline", &pipelineIdx, pipelineNames, IM_ARRAYSIZE(pipelineNames)))
            m_ui.requestedPipeline = static_cast<Pipeline>(pipelineIdx);
        if (m_ui.activePipeline != m_ui.requestedPipeline)
            ImGui::TextDisabled("(switching...)");
        ImGui::Separator();

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
        if (m_ui.impostorVisibleCount > 0) {
            ImGui::Text("Instances  visible: %u / %u  (impostors: %u, culled: %u)",
                m_ui.visibleInstanceCount, m_ui.totalInstanceCount,
                m_ui.impostorVisibleCount, m_ui.culledInstanceCount);
        } else {
            ImGui::Text("Instances  visible: %u / %u  (culled: %u)",
                m_ui.visibleInstanceCount, m_ui.totalInstanceCount, m_ui.culledInstanceCount);
        }
        ImGui::Text("Shadow casters:    %u / %u  (culled: %u)",
            m_ui.shadowVisibleCount, m_ui.totalInstanceCount, m_ui.shadowCulledCount);
        ImGui::Text("Cascade draws:     %u  (overdraw: %u)",
            m_ui.shadowCascadeDrawCount, m_ui.shadowOverdrawCount);
        ImGui::Text("Draw calls: %u", m_ui.drawCallCount);
    }

    _buildLSystemsSection();
    _buildAssetsSection();
    _buildRegionsSection();

    if (ImGui::CollapsingHeader("Debug")) {
        ImGui::Checkbox("Top-Down View", &m_ui.showDebugTopDown);
        ImGui::Checkbox("Shadow Top-Down View", &m_ui.showDebugShadowTopDown);
        ImGui::Checkbox("Shadow Map", &m_ui.showShadowMap);
        ImGui::Checkbox("Hi-Z Mip Chain", &m_ui.showHiZ);
        ImGui::Checkbox("Impostor Atlas", &m_ui.showImpostorAtlas);
        ImGui::Separator();
        ImGui::SliderFloat("Hi-Z Bypass Angle", &m_ui.hizBypassAngle, 0.f, 1.f, "%.2f");
        ImGui::SameLine();
        ImGui::TextDisabled(m_ui.hizActiveThisFrame ? "(active)" : "(bypassed)");
        ImGui::SliderFloat("PSSM Lambda", &m_ui.pssmLambda, 0.f, 1.f, "%.2f");
        ImGui::SliderFloat("Impostor Alpha Clip", &m_ui.impostorAlphaClip, 0.01f, 0.95f, "%.2f");

        if (ImGui::TreeNode("SDSM Debug")) {
            if (!m_ui.sdsmDebugValid) {
                ImGui::TextDisabled("(no SDSM readback yet)");
            } else {
                ImGui::Text("nearDepthVal: %.6f", m_ui.sdsmNearDepthVal);
                ImGui::Text("farDepthVal:  %.6f", m_ui.sdsmFarDepthVal);
                ImGui::Text("tightNear:    %.4f", m_ui.sdsmTightNear);
                ImGui::Text("tightFar:     %.4f", m_ui.sdsmTightFar);
                ImGui::Separator();
                ImGui::Text("splits: %.3f  %.3f  %.3f  %.3f",
                    m_ui.sdsmCascadeSplits[0], m_ui.sdsmCascadeSplits[1],
                    m_ui.sdsmCascadeSplits[2], m_ui.sdsmCascadeSplits[3]);
            }
            ImGui::TreePop();
        }
    }

    ImGui::Spacing();
    if (ImGui::Button("Hide UI  [ESC]")) m_ui.ShowUI = false;

    ImGui::End();

    _buildDebugTopDownSection();
    _buildDebugShadowTopDownSection();
    _buildShadowMapSection();
    _buildHiZSection();
    _buildImpostorAtlasSection();
}


void UIRenderer::_buildSceneFileSection() {
    if (!ImGui::CollapsingHeader("Scene File", ImGuiTreeNodeFlags_DefaultOpen))
        return;

    ImGui::InputText("Path", &m_ScenePath);

    if (ImGui::Button("Save")) {
        if (m_ScenePath.empty()) {
            m_ui.sceneLoadStatus        = "Save failed: path is empty";
            m_ui.sceneLoadStatusIsError = true;
        } else {
            std::filesystem::path p(m_ScenePath);
            if (SceneLoader::Save(p, *m_Registry)) {
                m_ui.sceneLoadStatus        = "Saved " + p.string();
                m_ui.sceneLoadStatusIsError = false;
            } else {
                m_ui.sceneLoadStatus        = "Save failed: " + p.string();
                m_ui.sceneLoadStatusIsError = true;
            }
        }
    }

    ImGui::SameLine();

    if (ImGui::Button("Load")) {
        if (m_ScenePath.empty()) {
            m_ui.sceneLoadStatus        = "Load failed: path is empty";
            m_ui.sceneLoadStatusIsError = true;
        } else {
            // Defer the actual load to RenderOrchestrator — it owns the GPU sync,
            // registry replace, and camera reset.
            m_ui.requestedSceneLoad = true;
            m_ui.requestedScenePath = m_ScenePath;
            // Don't pre-write a status; the orchestrator writes it after the load.
        }
    }

    if (!m_ui.sceneLoadStatus.empty()) {
        ImVec4 col = m_ui.sceneLoadStatusIsError
            ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f)   // red
            : ImVec4(0.7f, 0.7f, 0.7f, 1.0f);  // gray
        ImGui::TextColored(col, "Status: %s", m_ui.sceneLoadStatus.c_str());
    }
}


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
            ImGui::TextDisabled("step=%.2f radius=%.2f  angle=%.1f deg  taper=%.2f  stepR=%.2f  seed=%u",
                asset->genParams.baseLength,
                asset->genParams.baseRadius,
                dm::degrees(asset->genParams.branchAngle),
                asset->genParams.taperRatio,
                asset->genParams.stepRatio,
                asset->genParams.seed);

            if (!state.open) {
                if (ImGui::Button("Edit")) {
                    state.open         = true;
                    state.pendingGen   = static_cast<int>(asset->lsystemInstance.gen);
                    state.params       = asset->genParams;
                    state.colonization = asset->colonization;
                    state.leaf         = asset->leaf;
                    state.hasLeaves    = asset->hasLeaves;
                }
            } else {
                float branchAngleDeg = dm::degrees(state.params.branchAngle);
                int   seed           = static_cast<int>(state.params.seed);

                ImGui::SliderInt("Generation",   &state.pendingGen,         1,    8);
                ImGui::SliderFloat("Base Length",  &state.params.baseLength, 0.1f, 5.0f);
                ImGui::SliderFloat("Base Radius",  &state.params.baseRadius, 0.1f, 5.0f);
                ImGui::SliderFloat("Branch Angle", &branchAngleDeg,          5.0f, 90.f);
                ImGui::SliderFloat("Taper Ratio",  &state.params.taperRatio, 0.5f, 1.0f);
                ImGui::SliderFloat("Step Ratio",   &state.params.stepRatio,  0.5f, 1.0f);
                ImGui::InputInt("Seed (0=none)",   &seed);

                state.params.branchAngle = dm::radians(branchAngleDeg);
                state.params.seed        = static_cast<uint32_t>(seed);

                // ---- Space Colonization (canopy branchlets) ----------------------
                if (ImGui::TreeNode("Space Colonization")) {
                    int   attrCount = static_cast<int>(state.colonization.attractorCount);
                    int   maxIters  = static_cast<int>(state.colonization.maxIterations);
                    int   scSeed    = static_cast<int>(state.colonization.seed);

                    ImGui::SliderInt  ("Attractors (0 = off)", &attrCount,                              0,    2048);
                    ImGui::SliderFloat("Influence Distance",   &state.colonization.influenceDistance,   0.5f, 16.f);
                    ImGui::SliderFloat("Kill Distance",        &state.colonization.killDistance,        0.1f, 4.f);
                    ImGui::SliderFloat("Segment Length",       &state.colonization.segmentLength,       0.05f, 2.f);
                    ImGui::SliderInt  ("Max Iterations",       &maxIters,                               1,    256);

                    if (ImGui::TreeNode("Crown Transform")) {
                        ImGui::DragFloat3("Translation",        &state.colonization.crownTranslation.x,     0.1f);
                        ImGui::DragFloat3("Rotation (deg)",     &state.colonization.crownRotationDegrees.x, 0.5f);
                        ImGui::DragFloat3("Scale",              &state.colonization.crownScale.x,           0.05f, 0.01f, 100.f);
                        ImGui::DragFloat3("Shear (XY,XZ,YZ)",   &state.colonization.crownShear.x,           0.01f);
                        ImGui::TreePop();
                    }

                    ImGui::SliderFloat("Branchlet Taper",      &state.colonization.branchletTaper,      0.5f, 1.f);
                    ImGui::InputInt   ("SC Seed (0=auto)",     &scSeed);

                    state.colonization.attractorCount = static_cast<uint32_t>(std::max(0, attrCount));
                    state.colonization.maxIterations  = static_cast<uint32_t>(std::max(1, maxIters));
                    state.colonization.seed           = static_cast<uint32_t>(std::max(0, scSeed));

                    ImGui::TreePop();
                }

                // ---- Foliage (leaves) --------------------------------------------
                if (ImGui::TreeNode("Foliage")) {
                    ImGui::Checkbox("Enable Leaves", &state.hasLeaves);

                    int leavesPerTip = static_cast<int>(state.leaf.perTip);

                    ImGui::ColorEdit3 ("Leaf Color",    &state.leaf.color.x);
                    ImGui::SliderFloat("Leaf Size",     &state.leaf.size,        0.05f, 2.0f);
                    ImGui::SliderInt  ("Leaves / Tip",  &leavesPerTip,           0,     16);

                    ImGui::SliderFloat("LOD0 Mult",     &state.leaf.lodMultipliers[0], 0.f, 1.f);
                    ImGui::SliderFloat("LOD1 Mult",     &state.leaf.lodMultipliers[1], 0.f, 1.f);
                    ImGui::SliderFloat("LOD2 Mult",     &state.leaf.lodMultipliers[2], 0.f, 1.f);
                    ImGui::SliderFloat("LOD3 Mult",     &state.leaf.lodMultipliers[3], 0.f, 1.f);

                    state.leaf.perTip = static_cast<uint32_t>(std::max(0, leavesPerTip));

                    ImGui::TreePop();
                }

                if (ImGui::Button("Apply")) {
                    state.open = false;

                    if (static_cast<uint32_t>(state.pendingGen) != asset->lsystemInstance.gen) {
                        auto* mutableAsset = m_Registry->findAsset(id);
                        mutableAsset->lsystemInstance.gen = static_cast<uint32_t>(state.pendingGen);
                    }

                    m_Registry->modifyAssetExtended(id, state.params, state.colonization,
                                                    state.leaf, state.hasLeaves);
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
        ImGui::SliderFloat("Step Length##add",  &m_AddAsset.params.baseLength,    0.1f, 5.0f);
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


bool UIRenderer::_beginTopDown(const char* title, bool* show, const char* canvasId,
                               TopDownCanvas& out) {
    if (!*show) return false;

    ImGui::SetNextWindowSize(ImVec2(420, 460), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(title, show)) {
        ImGui::End();
        return false;
    }

    const auto& regions = m_Registry->getRegions();
    if (regions.empty()) {
        ImGui::TextDisabled("No regions in scene.");
        ImGui::End();
        return false;
    }

    // Scene bbox in XZ with 10% padding.
    dm::box3 sceneBbox = dm::box3::empty();
    for (const auto& region : regions)
        sceneBbox |= region.cullBox;
    if (const auto* terrain = m_Registry->getTerrain())
        sceneBbox |= terrain->getBbox();

    float worldMinX = sceneBbox.m_mins.x, worldMaxX = sceneBbox.m_maxs.x;
    float worldMinZ = sceneBbox.m_mins.z, worldMaxZ = sceneBbox.m_maxs.z;
    float extentX = std::max(1.f, worldMaxX - worldMinX);
    float extentZ = std::max(1.f, worldMaxZ - worldMinZ);
    float padX = extentX * 0.1f, padZ = extentZ * 0.1f;
    worldMinX -= padX; worldMaxX += padX;
    worldMinZ -= padZ; worldMaxZ += padZ;
    extentX = worldMaxX - worldMinX;
    extentZ = worldMaxZ - worldMinZ;

    out.worldCenterX = (worldMinX + worldMaxX) * 0.5f;
    out.worldCenterZ = (worldMinZ + worldMaxZ) * 0.5f;
    out.canvasPos    = ImGui::GetCursorScreenPos();
    out.canvasSize   = ImGui::GetContentRegionAvail();
    if (out.canvasSize.x < 50.f) out.canvasSize.x = 50.f;
    if (out.canvasSize.y < 50.f) out.canvasSize.y = 50.f;
    ImGui::InvisibleButton(canvasId, out.canvasSize);

    out.canvasCenterX = out.canvasPos.x + out.canvasSize.x * 0.5f;
    out.canvasCenterY = out.canvasPos.y + out.canvasSize.y * 0.5f;
    out.scale         = std::min(out.canvasSize.x / extentX, out.canvasSize.y / extentZ);
    out.dl            = ImGui::GetWindowDrawList();

    out.dl->AddRectFilled(out.toScreen(worldMinX, worldMinZ),
                          out.toScreen(worldMaxX, worldMaxZ),
                          IM_COL32(20, 20, 20, 255));
    return true;
}


void UIRenderer::_buildDebugTopDownSection() {
    TopDownCanvas ctx;
    if (!_beginTopDown("Top-Down Debug View", &m_ui.showDebugTopDown, "##canvas", ctx))
        return;

    const auto& regions = m_Registry->getRegions();
    dm::frustum frust = m_ViewHandler->view.GetViewFrustum();

    const ImU32 colRegion  = IM_COL32(100,100,100, 150);
    const ImU32 colCulled  = IM_COL32(200, 60, 60, 180);
    const ImU32 colVisible = IM_COL32(60, 200, 60, 220);
    const float radCulled  = 0.5f;
    const float radVisible = 0.75f;

    for (const auto& region : regions) {
        ImVec2 rMin = ctx.toScreen(region.bounds.m_mins.x, region.bounds.m_mins.y);
        ImVec2 rMax = ctx.toScreen(region.bounds.m_maxs.x, region.bounds.m_maxs.y);

        bool regionVis = frust.intersectsWith(region.cullBox);
        ctx.dl->AddRect(rMin, rMax, regionVis ? colRegion : colCulled, 0.f, 0, 1.f);
        if (!regionVis) continue;

        for (const auto& inst : region.instances) {
            bool vis = frust.intersectsWith(inst.bbox);
            float wx = inst.model[3][0];
            float wz = inst.model[3][2];
            ImVec2 sp = ctx.toScreen(wx, wz);
            float  r  = vis ? radVisible : radCulled;
            ImU32  c  = vis ? colVisible : colCulled;
            ctx.dl->AddRectFilled(ImVec2(sp.x-r,sp.y-r), ImVec2(sp.x+r,sp.y+r), c);
        }
    }

    // Camera frustum wireframe (corner bits: 0=right, 1=top, 2=far).
    ImVec2 corners[8];
    for (int i = 0; i < 8; i++) {
        dm::float3 c = frust.getCorner(i);
        corners[i] = ctx.toScreen(c.x, c.z);
    }
    const ImU32 frustumColor = IM_COL32(255, 255, 0, 255);
    const float thickness = 2.0f;
    static const int edges[12][2] = {
        {0,1},{1,3},{3,2},{2,0},   // near
        {4,5},{5,7},{7,6},{6,4},   // far
        {0,4},{1,5},{2,6},{3,7},   // near->far
    };
    for (const auto& e : edges)
        ctx.dl->AddLine(corners[e[0]], corners[e[1]], frustumColor, thickness);

    dm::float3 camPos = m_ViewHandler->camera.GetPosition();
    ctx.dl->AddCircleFilled(ctx.toScreen(camPos.x, camPos.z), 5.f, IM_COL32(0, 255, 255, 255));

    // Legend.
    const float legendScale = 4.f;
    ImVec2 legendPos = ImVec2(ctx.canvasPos.x + 5.f, ctx.canvasPos.y + 5.f);
    ctx.dl->AddRectFilled(
        ImVec2(legendPos.x + 5.f-radVisible*legendScale, legendPos.y + 6.f-radVisible*legendScale),
        ImVec2(legendPos.x + 5.f+radVisible*legendScale, legendPos.y + 6.f+radVisible*legendScale),
        colVisible);
    ctx.dl->AddText(ImVec2(legendPos.x + 14.f, legendPos.y), IM_COL32(200, 200, 200, 255), "Visible");
    ctx.dl->AddRectFilled(
        ImVec2(legendPos.x + 5.f-radCulled*legendScale, legendPos.y + 20.f-radCulled*legendScale),
        ImVec2(legendPos.x + 5.f+radCulled*legendScale, legendPos.y + 20.f+radCulled*legendScale),
        colCulled);
    ctx.dl->AddText(ImVec2(legendPos.x + 14.f, legendPos.y + 14.f), IM_COL32(200, 200, 200, 255), "Culled");
    ctx.dl->AddCircleFilled(ImVec2(legendPos.x + 5.f, legendPos.y + 34.f), 4.f, IM_COL32(0, 255, 255, 255));
    ctx.dl->AddText(ImVec2(legendPos.x + 14.f, legendPos.y + 28.f), IM_COL32(200, 200, 200, 255), "Camera");

    ImGui::End();
}


void UIRenderer::_buildDebugShadowTopDownSection() {
    TopDownCanvas ctx;
    if (!_beginTopDown("Shadow Top-Down View", &m_ui.showDebugShadowTopDown, "##shadowCanvas", ctx))
        return;

    const auto& regions = m_Registry->getRegions();

    const dm::float3 cascadeRGB[Render::c_NumCascades] = {
        { 1.00f, 0.90f, 0.20f },  // yellow
        { 0.30f, 1.00f, 0.40f },  // green
        { 0.30f, 0.70f, 1.00f },  // cyan
        // { 0.80f, 0.40f, 1.00f },  // purple
    };
    auto packColor = [](dm::float3 rgb, float a) -> ImU32 {
        auto clamp01 = [](float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); };
        return IM_COL32(
            int(clamp01(rgb.x) * 255.f), int(clamp01(rgb.y) * 255.f),
            int(clamp01(rgb.z) * 255.f), int(clamp01(a)    * 255.f));
    };

    const dm::affine3& worldToLight = m_ViewHandler->worldToLight;

    // Lowest-index cascade whose light-space shadow-caster bbox overlaps this instance's
    // world-space bbox transformed into light space. Mirrors CullShadow in CullCS.hlsl.
    auto firstCascadeFor = [&](const dm::box3& wsBbox) -> int {
        dm::box3 lsBbox = wsBbox * worldToLight;
        for (uint32_t c = 0; c < Render::c_NumCascades; ++c) {
            const dm::box3& lsCasc = m_ViewHandler->cascades[c].shadowCasterBboxLS;
            if (!lsCasc.isempty() && lsBbox.intersects(lsCasc)) return int(c);
        }
        return -1;
    };

    const ImU32 colRegion   = IM_COL32(100, 100, 100, 150);
    const ImU32 colExcluded = IM_COL32(70, 70, 70, 180);
    const float radIncluded = 0.9f;
    const float radExcluded = 0.5f;

    for (const auto& region : regions) {
        ImVec2 rMin = ctx.toScreen(region.bounds.m_mins.x, region.bounds.m_mins.y);
        ImVec2 rMax = ctx.toScreen(region.bounds.m_maxs.x, region.bounds.m_maxs.y);
        ctx.dl->AddRect(rMin, rMax, colRegion, 0.f, 0, 1.f);

        for (const auto& inst : region.instances) {
            int ci = firstCascadeFor(inst.bbox);
            float wx = inst.model[3][0];
            float wz = inst.model[3][2];
            ImVec2 sp = ctx.toScreen(wx, wz);

            ImU32 c = (ci >= 0) ? packColor(cascadeRGB[ci], 0.9f) : colExcluded;
            float r = (ci >= 0) ? radIncluded : radExcluded;
            ctx.dl->AddRectFilled(ImVec2(sp.x - r, sp.y - r), ImVec2(sp.x + r, sp.y + r), c);
        }
    }

    dm::float3 camPos = m_ViewHandler->camera.GetPosition();
    ctx.dl->AddCircleFilled(ctx.toScreen(camPos.x, camPos.z), 5.f, IM_COL32(0, 255, 255, 255));

    // Legend.
    const float legendScale = 4.f;
    const float r = 0.75f;
    ImVec2 legendPos = ImVec2(ctx.canvasPos.x + 5.f, ctx.canvasPos.y + 5.f);
    char label[32];
    for (uint32_t ci = 0; ci < Render::c_NumCascades; ++ci) {
        float y = float(ci) * 14.f;
        ImU32 col = packColor(cascadeRGB[ci], 1.f);
        ctx.dl->AddRectFilled(
            ImVec2(legendPos.x + 5.f - r * legendScale, legendPos.y + y + 6.f - r * legendScale),
            ImVec2(legendPos.x + 5.f + r * legendScale, legendPos.y + y + 6.f + r * legendScale),
            col);
        snprintf(label, sizeof(label), "Cascade %u", ci);
        ctx.dl->AddText(ImVec2(legendPos.x + 14.f, legendPos.y + y), IM_COL32(220, 220, 220, 255), label);
    }
    float camY = float(Render::c_NumCascades) * 14.f + 6.0f;
    ctx.dl->AddCircleFilled(ImVec2(legendPos.x + 5.f, legendPos.y + camY + 6.f), 4.f, IM_COL32(0, 255, 255, 255));
    ctx.dl->AddText(ImVec2(legendPos.x + 14.f, legendPos.y + camY), IM_COL32(220, 220, 220, 255), "Camera");

    ImGui::End();
}


void UIRenderer::_buildShadowMapSection() {
    if (!m_ui.showShadowMap) return;

    constexpr int numCascades = static_cast<int>(Render::c_NumCascades);

    ImGui::SetNextWindowSize(ImVec2(520, 560), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Shadow Map Cascades", &m_ui.showShadowMap)) {
        ImGui::End();
        return;
    }

    if (m_ui.selectedCascade >= numCascades) m_ui.selectedCascade = 0;
    if (m_ui.selectedCascade < 0)             m_ui.selectedCascade = 0;

    ImGui::Text("Cascade:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(180.f);
    ImGui::SliderInt("##cascade", &m_ui.selectedCascade, 0, numCascades - 1);
    ImGui::SameLine();
    ImGui::TextDisabled("(%d cascades)", numCascades);

    ImGui::Separator();

    if (m_ui.selectedCascadeTexture) {
        ImVec2 avail = ImGui::GetContentRegionAvail();
        float size  = std::max(std::min(avail.x, avail.y), 64.f);
        ImGui::Image(ImTextureRef(m_ui.selectedCascadeTexture), ImVec2(size, size));
    }

    ImGui::End();
}


void UIRenderer::_buildHiZSection() {
    if (!m_ui.showHiZ || m_ui.hizMipTextures.empty()) return;

    ImGui::SetNextWindowSize(ImVec2(520, 560), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Hi-Z Mip Chain", &m_ui.showHiZ)) {
        ImGui::End();
        return;
    }

    int numMips = static_cast<int>(m_ui.hizMipTextures.size());

    static int selectedMip = 0;
    if (selectedMip >= numMips) selectedMip = 0;

    ImGui::Text("Mip level:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(180.f);
    ImGui::SliderInt("##hizmip", &selectedMip, 0, numMips - 1);
    ImGui::SameLine();
    ImGui::TextDisabled("(%d mips total)", numMips);
    ImGui::TextDisabled("Resolution halves each mip. Mip 0 = full framebuffer.");

    ImGui::Separator();

    void* tex = m_ui.hizMipTextures[selectedMip];
    if (tex) {
        ImVec2 avail = ImGui::GetContentRegionAvail();
        float size  = std::max(std::min(avail.x, avail.y), 64.f);
        ImGui::Image(ImTextureRef(tex), ImVec2(size, size));
    } else {
        ImGui::TextDisabled("(not yet available — enable Hi-Z panel before first frame)");
    }

    ImGui::End();
}


void UIRenderer::_buildImpostorAtlasSection() {
    if (!m_ui.showImpostorAtlas) return;

    const uint32_t numAssets = m_ui.impostorAssetCount;
    const uint32_t numViews  = m_ui.impostorViewsPerAsset;
    const uint32_t numAzimuthViews = m_ui.impostorAzimuthViews ? m_ui.impostorAzimuthViews : numViews;
    const uint32_t numElevationViews = m_ui.impostorElevationViews ? m_ui.impostorElevationViews : 1;
    if (numAssets == 0 || numViews == 0) return;

    ImGui::SetNextWindowSize(ImVec2(620, 720), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Impostor Atlas", &m_ui.showImpostorAtlas)) {
        ImGui::End();
        return;
    }

    int selectedAsset = static_cast<int>(m_ui.impostorSelectedAsset);
    if (selectedAsset >= (int)numAssets) selectedAsset = 0;

    ImGui::Text("Asset:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(160.f);
    ImGui::SliderInt("##impAsset", &selectedAsset, 0, (int)numAssets - 1);
    ImGui::SameLine();
    ImGui::TextDisabled("(%u assets)", numAssets);

    ImGui::Separator();

    m_ui.impostorSelectedAsset = static_cast<uint32_t>(selectedAsset);

    void* albedoAtlas = m_ui.impostorAlbedoAtlasTexture;
    void* normalAtlas = m_ui.impostorNormalAtlasTexture;
    void* depthAtlas  = m_ui.impostorDepthAtlasTexture;

    ImVec2 avail = ImGui::GetContentRegionAvail();

    static int atlasChannel = 0;
    const char* channels[] = { "Albedo+Alpha", "Normal", "Depth" };
    ImGui::Text("Selected asset atlas:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(160.f);
    ImGui::Combo("##impAtlasChannel", &atlasChannel, channels, 3);

    void* atlasTex = atlasChannel == 0 ? albedoAtlas : (atlasChannel == 1 ? normalAtlas : depthAtlas);
    if (atlasTex) {
        float atlasWidth = std::max(avail.x, 64.f);
        float atlasHeight = atlasWidth * (float)numElevationViews / (float)numAzimuthViews;
        ImGui::Image(ImTextureRef(atlasTex), ImVec2(atlasWidth, atlasHeight));
    } else {
        ImGui::TextDisabled("(atlas preview not available yet)");
    }

    ImGui::End();
}
