#include "include/UIRenderer.hpp"

#include <donut/core/math/math.h>
#include <filesystem>
#include <sstream>
#include <string>
#include <algorithm>

using namespace Xylem;

const char* UIRenderer::_fmtCount(uint64_t v) {
    // Rotating ring of thread-local scratch buffers so that multiple
    // _fmtCount calls in the same ImGui::Text varargs list don't trample
    // each other (a single shared buffer would make all %s arguments
    // resolve to the same — last-written — value).
    thread_local char bufs[8][32];
    thread_local int  next = 0;
    char* buf = bufs[next];
    next = (next + 1) & 7;

    char tmp[24];
    int  n = snprintf(tmp, sizeof(tmp), "%llu", static_cast<unsigned long long>(v));
    if (n <= 0) { buf[0] = '0'; buf[1] = 0; return buf; }

    // Walk tmp right-to-left into buf right-to-left, inserting a comma
    // every 3 digits.
    int outPos = 0;
    int groupCount = 0;
    for (int i = n - 1; i >= 0; --i) {
        if (groupCount == 3) {
            buf[outPos++] = ',';
            groupCount = 0;
        }
        buf[outPos++] = tmp[i];
        ++groupCount;
    }
    buf[outPos] = 0;

    // Reverse buf in place.
    for (int a = 0, b = outPos - 1; a < b; ++a, --b) {
        char t = buf[a];
        buf[a] = buf[b];
        buf[b] = t;
    }
    return buf;
}

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
    _buildRuntimeSettingsSection();
    _buildBenchmarkSection();
    _buildSunSkySection();

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

        // ----- Per-stage GPU breakdown -----
        {
            const size_t kCount = static_cast<size_t>(frame::FrameStage::COUNT);

            // Hide the section entirely if no stage has reported a value yet
            // (warmup right after a pipeline switch).
            bool anyStageReady = false;
            for (size_t s = 0; s < kCount; ++s) {
                if (m_ui.gpuStageTimeMs[s] >= 0.f) { anyStageReady = true; break; }
            }

            if (anyStageReady) {
                ImGui::Spacing();
                ImGui::TextDisabled("GPU stages:");

                // Fixed palette per FrameStage so colors are stable across
                // pipeline switches. Defined once and reused for both the
                // colorized text rows (which act as the legend) and the bar.
                static constexpr ImU32 kStageColors[7] = {
                    IM_COL32( 64, 192, 192, 255), // DepthPrepass — teal
                    IM_COL32(160, 224,  64, 255), // HiZ          — lime
                    IM_COL32(240, 200,  64, 255), // SDSM         — gold
                    IM_COL32(240, 144,  64, 255), // Cull         — orange
                    IM_COL32(208,  96, 192, 255), // Shadow       — magenta
                    IM_COL32( 96, 176, 240, 255), // Sky          — sky
                    IM_COL32(144, 144, 168, 255), // Scene        — slate
                };
                const ImU32 kMiscColor = IM_COL32(96, 96, 96, 255);

                // Colorized stage rows. Each row is rendered in its bar color
                // — that's the legend; the bar is then unambiguous without a
                // separate swatch list.
                float stageSum = 0.f;
                for (size_t s = 0; s < kCount; ++s) {
                    const char* name = frame::ToString(static_cast<frame::FrameStage>(s));
                    const float v = m_ui.gpuStageTimeMs[s];
                    const ImVec4 col = ImGui::ColorConvertU32ToFloat4(kStageColors[s]);
                    if (v >= 0.f) {
                        ImGui::TextColored(col, "  %-13s %.2f ms", name, v);
                        stageSum += v;
                    } else {
                        // Stage not run by this pipeline — render in dim gray
                        // so it doesn't visually claim a color slot in the bar.
                        ImGui::TextDisabled("  %-13s   \xE2\x80\x94", name);
                    }
                }

                // Misc row: gpuFrameTimeMs - sum(stages where >= 0).
                // Hidden if gpuFrameTimeMs hasn't reported yet (warmup).
                float miscMs = -1.f;
                if (m_ui.gpuFrameTimeMs >= 0.f) {
                    miscMs = m_ui.gpuFrameTimeMs - stageSum;
                    if (miscMs < 0.f) miscMs = 0.f;  // clamp negative noise
                    const ImVec4 miscCol = ImGui::ColorConvertU32ToFloat4(kMiscColor);
                    ImGui::TextColored(miscCol, "  %-13s %.2f ms", "Misc", miscMs);
                }

                const float barTotal = (miscMs >= 0.f) ? (stageSum + miscMs) : stageSum;
                if (barTotal > 0.f) {
                    const float barHeight = 20.f;
                    const float barWidth  = ImGui::GetContentRegionAvail().x;
                    const ImVec2 cursor   = ImGui::GetCursorScreenPos();
                    ImDrawList* dl = ImGui::GetWindowDrawList();

                    float xOffset = 0.f;
                    for (size_t s = 0; s < kCount; ++s) {
                        const float v = m_ui.gpuStageTimeMs[s];
                        if (v < 0.f) continue;
                        const float w = (v / barTotal) * barWidth;
                        dl->AddRectFilled(
                            ImVec2(cursor.x + xOffset,     cursor.y),
                            ImVec2(cursor.x + xOffset + w, cursor.y + barHeight),
                            kStageColors[s]);
                        xOffset += w;
                    }
                    if (miscMs > 0.f) {
                        const float w = (miscMs / barTotal) * barWidth;
                        dl->AddRectFilled(
                            ImVec2(cursor.x + xOffset,     cursor.y),
                            ImVec2(cursor.x + xOffset + w, cursor.y + barHeight),
                            kMiscColor);
                    }

                    // Reserve the bar's screen space.
                    ImGui::Dummy(ImVec2(barWidth, barHeight));
                }
            }
        }

        ImGui::Separator();
        _buildTreesFunnel();
        ImGui::Separator();
        _buildLeavesFunnel();
        ImGui::Spacing();
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
        if (m_ui.activePipeline == Pipeline::MeshShader && m_ui.totalTerrainMeshletCount > 0) {
            ImGui::Separator();
            _buildTerrainFunnel();
        }
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


void UIRenderer::_buildRuntimeSettingsSection() {
    if (!ImGui::CollapsingHeader("Runtime Settings", ImGuiTreeNodeFlags_DefaultOpen))
        return;

    static const char* pipelineNames[] = { "Traditional (CPU cull)", "Compute Cull (GPU cull)", "Mesh Shader (AS/MS)" };
    int pipelineIdx = static_cast<int>(m_ui.requestedPipeline);
    if (ImGui::Combo("Pipeline", &pipelineIdx, pipelineNames, IM_ARRAYSIZE(pipelineNames)))
        m_ui.requestedPipeline = static_cast<Pipeline>(pipelineIdx);
    if (m_ui.activePipeline != m_ui.requestedPipeline)
        ImGui::TextDisabled("(switching...)");

    ImGui::Checkbox("Hi-Z", &m_ui.hizEnabled);
    ImGui::SliderFloat("PSSM Lambda", &m_ui.pssmLambda, 0.f, 1.f, "%.2f");
    ImGui::Checkbox("Shadow Impostors", &m_ui.showShadowImpostors);
}


void UIRenderer::_buildBenchmarkSection() {
    if (!ImGui::CollapsingHeader("Benchmark", ImGuiTreeNodeFlags_DefaultOpen))
        return;

    // Path indicator (runner publishes the filename to UIData).
    const char* pathStr = m_ui.benchmarkPathFileName.empty()
        ? "(no path loaded)"
        : m_ui.benchmarkPathFileName.c_str();
    ImGui::TextDisabled("Path: %s", pathStr);
    ImGui::SameLine();
    ImGui::BeginDisabled(m_ui.benchmarkInProgress);
    if (ImGui::SmallButton("Reload")) {
        m_ui.benchmarkPathReloadRequested = true;
    }
    ImGui::EndDisabled();

    // Active path dropdown. Picking auto-loads the selected entry's waypoints
    // into the editor — destroys unsaved captures in the prior entry by design.
    ImGui::BeginDisabled(m_ui.benchmarkInProgress);
    {
        const auto& names = m_ui.benchmarkPathNames;
        // Keep the selected index in sync with the runner-published active name.
        int activeFromRunner = -1;
        for (size_t i = 0; i < names.size(); ++i) {
            if (names[i] == m_ui.benchmarkActivePathName) {
                activeFromRunner = static_cast<int>(i);
                break;
            }
        }
        if (activeFromRunner >= 0) {
            m_BenchmarkActiveIdx = activeFromRunner;
        }

        // Auto-sync the Name field to the active name, but only on actual change
        // — otherwise we'd clobber the user's typing every frame.
        if (m_ui.benchmarkActivePathName != m_LastSyncedActiveName) {
            m_BenchmarkNameField = m_ui.benchmarkActivePathName;
            m_LastSyncedActiveName = m_ui.benchmarkActivePathName;
        }

        // ImGui::Combo's simple overload wants a vector<const char*>.
        std::vector<const char*> nameCstrs;
        nameCstrs.reserve(names.size());
        for (const auto& n : names) nameCstrs.push_back(n.c_str());

        if (names.empty()) {
            ImGui::TextDisabled("Active: (no paths in file)");
        } else {
            if (ImGui::Combo("Active", &m_BenchmarkActiveIdx,
                             nameCstrs.data(),
                             static_cast<int>(nameCstrs.size())))
            {
                // Selection changed — request the runner to swap entries.
                if (m_BenchmarkActiveIdx >= 0 &&
                    m_BenchmarkActiveIdx < static_cast<int>(names.size()))
                {
                    m_ui.benchmarkSelectName      = names[m_BenchmarkActiveIdx];
                    m_ui.benchmarkSelectRequested = true;
                }
            }
        }
    }
    ImGui::EndDisabled();

    ImGui::TextWrapped("Vsync will be disabled during the run and restored after.");

    // Pipelines: three checkboxes, disabled mid-run.
    ImGui::BeginDisabled(m_ui.benchmarkInProgress);
    ImGui::TextUnformatted("Pipelines:");
    ImGui::SameLine();
    ImGui::Checkbox("Traditional", &m_ui.benchmarkEnabledPipelines[0]);
    ImGui::SameLine();
    ImGui::Checkbox("Compute",     &m_ui.benchmarkEnabledPipelines[1]);
    ImGui::SameLine();
    ImGui::Checkbox("MeshShader",  &m_ui.benchmarkEnabledPipelines[2]);
    ImGui::EndDisabled();

    const bool anyPipelineEnabled =
        m_ui.benchmarkEnabledPipelines[0] ||
        m_ui.benchmarkEnabledPipelines[1] ||
        m_ui.benchmarkEnabledPipelines[2];
    const bool pathRunnable = m_ui.benchmarkWaypoints.size() >= 2;

    if (m_ui.benchmarkInProgress) {
        if (ImGui::Button("Cancel##bench")) {
            m_ui.benchmarkCancelRequested = true;
        }
        if (!m_ui.benchmarkProgressLabel.empty()) {
            ImGui::TextUnformatted(m_ui.benchmarkProgressLabel.c_str());
        }
    } else {
        const bool canRun = anyPipelineEnabled && pathRunnable;
        ImGui::BeginDisabled(!canRun);
        if (ImGui::Button("Run Benchmark")) {
            m_ui.benchmarkRunRequested = true;
        }
        ImGui::EndDisabled();
        if (!canRun && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            if (!anyPipelineEnabled) {
                ImGui::SetTooltip("Select at least one pipeline.");
            } else {
                ImGui::SetTooltip("Path needs at least 2 waypoints.");
            }
        }
    }

    if (!m_ui.benchmarkLastStatus.empty()) {
        const ImVec4 col = m_ui.benchmarkLastStatusIsError
            ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f)   // red
            : ImVec4(0.4f, 1.0f, 0.4f, 1.0f);  // green
        ImGui::TextColored(col, "Status: %s", m_ui.benchmarkLastStatus.c_str());
    }

    // Collapsing child for waypoint editing. Default-closed to keep the
    // benchmark section compact when the user isn't editing.
    if (ImGui::TreeNode("Waypoints")) {
        ImGui::BeginDisabled(m_ui.benchmarkInProgress);

        if (ImGui::Button("Capture Current Camera")) {
            m_ui.benchmarkCaptureWaypointRequested = true;
        }

        // Iterate the snapshot the runner publishes. The time-edit floats live
        // in a parallel vector so ImGui has stable storage; runner diffs them
        // against authoritative state in PreAnimate.
        const size_t n = m_ui.benchmarkWaypoints.size();
        if (n != m_ui.benchmarkWaypointTimesEdited.size()) {
            // Runner hasn't published yet (or just resized). Skip rows this frame.
        } else {
            for (size_t i = 0; i < n; ++i) {
                ImGui::PushID(static_cast<int>(i));
                ImGui::Text("#%zu", i);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(80.f);
                ImGui::InputFloat("t (s)", &m_ui.benchmarkWaypointTimesEdited[i],
                                  0.f, 0.f, "%.2f");
                ImGui::SameLine();
                if (ImGui::SmallButton("Preview")) {
                    m_ui.benchmarkPreviewWaypointIndex = static_cast<int>(i);
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("Delete")) {
                    m_ui.benchmarkDeleteWaypointIndex = static_cast<int>(i);
                }
                ImGui::PopID();
            }
        }

        ImGui::Separator();
        ImGui::InputText("Name", &m_BenchmarkNameField);
        ImGui::SameLine();
        if (ImGui::Button("Save")) {
            m_ui.benchmarkNameField     = m_BenchmarkNameField;
            m_ui.benchmarkSaveRequested = true;
        }

        ImGui::EndDisabled();
        ImGui::TreePop();
    }
}


void UIRenderer::_buildSunSkySection() {
    if (!ImGui::CollapsingHeader("Sky & Day/Night"))
        return;

    auto& s = m_Registry->getSunSkyMutable();

    // Top-level controls
    ImGui::Checkbox("Paused", &s.paused);
    ImGui::SliderFloat("Phase", &s.phase, 0.0f, 1.0f, "%.4f");
    ImGui::SliderFloat("Azimuth (deg)", &s.azimuthDeg, 0.0f, 360.0f, "%.1f");
    ImGui::DragFloat("Above-horizon angular velocity (deg/s)", &s.angularVelocityDegPerSec,             0.1f, 0.01f, 360.0f, "%.2f");
    ImGui::DragFloat("Below-horizon angular velocity (deg/s)", &s.belowHorizonAngularVelocityDegPerSec, 0.5f, 0.01f, 360.0f, "%.2f");
    ImGui::DragFloat("Phase fade (s)",                         &s.phaseFadeSeconds,                     0.1f, 0.0f,  60.0f,  "%.2f");
    ImGui::DragFloat("Dawn/Dusk below horizon (deg)",          &s.dawnDuskBelowHorizonDeg,              0.5f, 0.0f,  45.0f,  "%.1f");
    ImGui::DragFloat("Horizon fade band (deg)",                &s.horizonFadeAngleDeg,                  0.5f, 0.1f,  30.0f,  "%.1f");

    // Derived read-out
    const float cycle = Xylem::Scene::TotalCycleSeconds(s);
    ImGui::Text("Cycle: %.1f s", cycle);

    // Keyframes
    const char* kfNames[5] = { "Dawn", "Noon", "Dusk", "Dark", "Moonlight" };
    if (ImGui::TreeNode("Keyframes")) {
        for (uint32_t i = 0; i < Xylem::Scene::SK_Count; ++i) {
            ImGui::PushID(static_cast<int>(i));
            if (ImGui::TreeNode(kfNames[i])) {
                auto& kf = s.keyframes[i];
                ImGui::ColorEdit3("sunColor (RGB)", &kf.sunColor.x,
                                  ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);

                ImGui::Separator();
                ImGui::Text("Sky shader parameters");
                ImGui::DragFloat("angularSizeOfLight", &kf.sky.angularSizeOfLight, 0.001f, 0.0f, 1.0f, "%.4f");
                ImGui::ColorEdit3("lightColor",  &kf.sky.lightColor.x,  ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
                ImGui::DragFloat("glowSize",     &kf.sky.glowSize, 0.001f, 0.0f, 3.14f, "%.4f");
                ImGui::ColorEdit3("skyColor",    &kf.sky.skyColor.x,    ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
                ImGui::DragFloat("glowIntensity",&kf.sky.glowIntensity, 0.01f, 0.0f, 10.0f, "%.3f");
                ImGui::ColorEdit3("horizonColor",&kf.sky.horizonColor.x,ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
                ImGui::DragFloat("horizonSize",  &kf.sky.horizonSize, 0.001f, 0.0f, 3.14f, "%.4f");
                ImGui::ColorEdit3("groundColor", &kf.sky.groundColor.x, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
                ImGui::DragFloat("glowSharpness",&kf.sky.glowSharpness, 0.1f, 0.0f, 64.0f, "%.2f");
                ImGui::ColorEdit3("directionUp", &kf.sky.directionUp.x, ImGuiColorEditFlags_Float);

                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        ImGui::TreePop();
    }

    // Read-only state diagnostic
    const auto& st = m_Registry->getSunSkyState();
    ImGui::Separator();
    ImGui::Text("lightDir: (%.2f, %.2f, %.2f)",  st.lightDir.x, st.lightDir.y, st.lightDir.z);
    ImGui::Text("sunColor: (%.2f, %.2f, %.2f)",  st.sunColor.x, st.sunColor.y, st.sunColor.z);
    ImGui::Text("shadowsEnabled: %s",            st.shadowsEnabled ? "yes" : "no");
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


void UIRenderer::_buildTreesFunnel() {
    ImGui::Text("Trees");
    ImGui::Indent();

    const uint32_t total     = m_ui.totalInstanceCount;
    const uint32_t culled    = m_ui.culledInstanceCount;
    const uint32_t impostors = m_ui.impostorVisibleCount;
    const uint32_t numLods   = std::min<uint32_t>(m_ui.lodCountForUI, UIData::kMaxLodsForUI);

    uint32_t meshTotal = 0;
    for (uint32_t li = 0; li < numLods; ++li) meshTotal += m_ui.lodVisibleCounts[li];

    ImGui::Text("Total instances: %s", _fmtCount(total));

    // Two stacked bars side by side:
    //   Bar 1 (overview) — top->bottom: Culled (red) | Billboarded (blue) | Mesh (one green segment).
    //                       Total height encodes m_ui.totalInstanceCount.
    //   Bar 2 (zoom)     — top->bottom: per-LOD breakdown of the mesh segment, lightest green at
    //                       top (coarsest LOD) -> darkest green at bottom (LOD 0). Total height
    //                       encodes meshTotal so the LOD slices fill the bar.
    // Connector lines fan out from the mesh segment of bar 1 to the full extent of bar 2.
    const float kBarWidth     = 56.f;
    const float kBarHeight    = 240.f;
    const float kLeftLabelW   = 200.f;   // labels to the LEFT of bar 1, right-aligned
    const float kRightLabelW  = 200.f;   // labels to the RIGHT of bar 2
    const float kConnectorGap = 70.f;    // horizontal space between the two bars
    const float kPadding      = 6.f;
    const float kSwatchGap    = 6.f;

    const ImU32 colBg      = IM_COL32(28, 28, 28, 255);
    const ImU32 colBorder  = IM_COL32(90, 90, 90, 255);
    const ImU32 colCulled  = IM_COL32(220, 70, 70, 255);
    const ImU32 colImpost  = IM_COL32(70, 140, 235, 255);
    const ImU32 colMesh    = IM_COL32(105, 180, 105, 255);
    const ImU32 colText    = IM_COL32(225, 225, 225, 255);
    const ImU32 colConnect = IM_COL32(140, 200, 140, 220);

    auto greenForLod = [&](uint32_t li) -> ImU32 {
        // Top of stack (li = numLods-1) = lightest; LOD 0 = darkest.
        float t = (numLods > 1) ? float(numLods - 1 - li) / float(numLods - 1) : 1.f;
        auto lerp8 = [](int lo, int hi, float k) {
            int v = lo + int((hi - lo) * k + 0.5f);
            return (uint8_t)std::clamp(v, 0, 255);
        };
        // light (170,230,170) -> dark (40,130,50)
        return IM_COL32(lerp8(170, 40, t), lerp8(230, 130, t), lerp8(170, 50, t), 255);
    };

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2      origin = ImGui::GetCursorScreenPos();

    const float bar1X = origin.x + kLeftLabelW + kPadding;
    const float bar2X = bar1X + kBarWidth + kConnectorGap;
    const float barT  = origin.y;
    const float barB  = origin.y + kBarHeight;

    dl->AddRectFilled(ImVec2(bar1X, barT), ImVec2(bar1X + kBarWidth, barB), colBg);
    dl->AddRectFilled(ImVec2(bar2X, barT), ImVec2(bar2X + kBarWidth, barB), colBg);

    const float invTotal  = total     > 0 ? 1.f / float(total)     : 0.f;
    const float invMesh   = meshTotal > 0 ? 1.f / float(meshTotal) : 0.f;
    const float textLineH = ImGui::GetTextLineHeight();
    const float swatchSz  = textLineH * 0.7f;
    const float labelStep = textLineH + 2.f; // min vertical advance between adjacent labels

    // Labels are queued (not drawn) during segment iteration so that we can run a
    // two-pass layout afterwards: a backward pass pulls earlier labels UP to make
    // room when a later label's natural Y would collide, then a forward pass
    // pushes any remaining overlap down. This avoids the "tiny last segment gets
    // shoved off the chart" failure mode of a single forward-only pass.
    struct PendingLabel { float segTop; float segH; ImU32 color; char text[96]; };
    std::vector<PendingLabel> labelsLeft;
    std::vector<PendingLabel> labelsRight;
    labelsLeft.reserve(8);
    labelsRight.reserve(UIData::kMaxLodsForUI);

    auto queueLabel = [&](std::vector<PendingLabel>& bucket,
                          float segTop, float segH, ImU32 color,
                          const char* text) {
        bucket.emplace_back();
        PendingLabel& L = bucket.back();
        L.segTop = segTop;
        L.segH   = segH;
        L.color  = color;
        snprintf(L.text, sizeof(L.text), "%s", text);
    };

    auto layoutAndDrawLabels = [&](const std::vector<PendingLabel>& labels, bool leftSide) {
        if (labels.empty()) return;
        std::vector<float> y(labels.size());

        // Pass 0: natural Y per label (centered when segment fits a line, else
        // top-aligned at the segment top).
        for (size_t i = 0; i < labels.size(); ++i) {
            const auto& L = labels[i];
            y[i] = (L.segH >= textLineH + 2.f)
                ? L.segTop + L.segH * 0.5f - textLineH * 0.5f
                : L.segTop;
        }

        // Backward pass: pull each label UP to clear the one below it.
        for (int i = int(labels.size()) - 2; i >= 0; --i)
            y[i] = std::min(y[i], y[i + 1] - labelStep);

        // Forward pass: push down anything still colliding with the one above
        // (handles the case where backward over-corrected past the natural top).
        for (size_t i = 1; i < labels.size(); ++i)
            y[i] = std::max(y[i], y[i - 1] + labelStep);

        for (size_t i = 0; i < labels.size(); ++i) {
            const auto& L  = labels[i];
            float textY    = y[i];
            float swY      = textY + (textLineH - swatchSz) * 0.5f;
            if (leftSide) {
                ImVec2 ts = ImGui::CalcTextSize(L.text);
                float swX = bar1X - kPadding - swatchSz;
                float txX = swX - kSwatchGap - ts.x;
                dl->AddText(ImVec2(txX, textY), colText, L.text);
                dl->AddRectFilled(ImVec2(swX, swY),
                                  ImVec2(swX + swatchSz, swY + swatchSz), L.color);
            } else {
                float swX = bar2X + kBarWidth + kPadding;
                float txX = swX + swatchSz + kSwatchGap;
                dl->AddRectFilled(ImVec2(swX, swY),
                                  ImVec2(swX + swatchSz, swY + swatchSz), L.color);
                dl->AddText(ImVec2(txX, textY), colText, L.text);
            }
        }
    };

    // ---- Chart 1: overview ----
    float    y1     = barT;
    float    accum1 = 0.f;
    uint32_t cum1   = 0;
    float    meshTopY = barT;

    auto drawSeg1 = [&](uint32_t count, ImU32 color, const char* labelHead) -> float {
        if (total == 0) return y1;
        cum1 += count;
        float target = float(cum1) * invTotal * kBarHeight;
        float h      = target - accum1;
        accum1       = target;
        float top    = y1;
        float bot    = y1 + h;
        if (count > 0) {
            dl->AddRectFilled(ImVec2(bar1X, top), ImVec2(bar1X + kBarWidth, bot), color);
            char buf[96];
            float pct = 100.f * float(count) * invTotal;
            snprintf(buf, sizeof(buf), "%s: %s  (%.1f%%)", labelHead, _fmtCount(count), pct);
            queueLabel(labelsLeft, top, h, color, buf);
        }
        y1 = bot;
        return top;
    };

    drawSeg1(culled,    colCulled, "Culled");
    drawSeg1(impostors, colImpost, "Billboarded");
    meshTopY = drawSeg1(meshTotal, colMesh, "Mesh");

    dl->AddRect(ImVec2(bar1X, barT), ImVec2(bar1X + kBarWidth, barB), colBorder);

    // ---- Chart 2: per-LOD zoom of the mesh segment ----
    if (meshTotal > 0) {
        float    y2     = barT;
        float    accum2 = 0.f;
        uint32_t cum2   = 0;
        for (int li = int(numLods) - 1; li >= 0; --li) {
            uint32_t count = m_ui.lodVisibleCounts[li];
            cum2 += count;
            float target = float(cum2) * invMesh * kBarHeight;
            float h      = target - accum2;
            accum2       = target;
            float top    = y2;
            float bot    = y2 + h;
            ImU32 col    = greenForLod((uint32_t)li);
            if (count > 0) {
                dl->AddRectFilled(ImVec2(bar2X, top), ImVec2(bar2X + kBarWidth, bot), col);
                char buf[96];
                float pct = 100.f * float(count) * invMesh;
                snprintf(buf, sizeof(buf), "LOD %d: %s  (%.1f%%)", li, _fmtCount(count), pct);
                queueLabel(labelsRight, top, h, col, buf);
            }
            y2 = bot;
        }
    }
    dl->AddRect(ImVec2(bar2X, barT), ImVec2(bar2X + kBarWidth, barB), colBorder);

    // Lay out + draw labels after all segments are placed (two-pass spread).
    layoutAndDrawLabels(labelsLeft,  /*leftSide=*/true);
    layoutAndDrawLabels(labelsRight, /*leftSide=*/false);

    // Connector lines: top of mesh segment in chart 1 -> top of chart 2,
    // bottom of chart 1 -> bottom of chart 2 (a fan-out / magnifier shape).
    if (meshTotal > 0) {
        const float thickness = 1.5f;
        dl->AddLine(ImVec2(bar1X + kBarWidth, meshTopY), ImVec2(bar2X, barT), colConnect, thickness);
        dl->AddLine(ImVec2(bar1X + kBarWidth, barB),     ImVec2(bar2X, barB), colConnect, thickness);
    }

    // Reserve total layout space so subsequent widgets clear both bars + labels.
    const float totalW = (bar2X + kBarWidth + kPadding + kRightLabelW) - origin.x;
    ImGui::Dummy(ImVec2(totalW, kBarHeight));

    ImGui::Spacing();
    ImGui::Text("Shadow draws (actual, per cascade):");
    ImGui::Indent();
    uint32_t shadowGeomSum = 0;
    uint32_t shadowBillSum = 0;
    for (uint32_t c = 0; c < Render::c_NumCascades; ++c) {
        const uint32_t geom = m_ui.shadowGeomDrawsPerCascade[c];
        const uint32_t bill = m_ui.shadowBillboardDrawsPerCascade[c];
        ImGui::Text("Cascade %u:          %s   (%s billboarded)", c, _fmtCount(geom), _fmtCount(bill));
        shadowGeomSum += geom;
        shadowBillSum += bill;
    }
    ImGui::Text("Total:              %s   (%s billboarded)", _fmtCount(shadowGeomSum), _fmtCount(shadowBillSum));
    ImGui::Unindent();

    if (m_ui.activePipeline == Pipeline::MeshShader) {
        ImGui::Spacing();
        ImGui::TextDisabled("-- mesh-shader AS --");
        const uint32_t asDisp = m_ui.trunkMainMeshletsDispatched;
        const uint32_t asRend = m_ui.trunkMainMeshletsRendered;
        const uint32_t asCull = (asDisp >= asRend) ? (asDisp - asRend) : 0;
        ImGui::Text("Total AS post-instance: %s",                  _fmtCount(asDisp));
        ImGui::Text("Post-AS culling:        %s   (%s culled)",    _fmtCount(asRend), _fmtCount(asCull));
        ImGui::Text("Shadow AS draws:        %s   (no AS cull on shadow path)",
                    _fmtCount(m_ui.trunkShadowMeshletsRendered));
    }

    ImGui::Unindent();
}

void UIRenderer::_buildLeavesFunnel() {
    ImGui::Text("Leaves");
    ImGui::Indent();

    const uint32_t total  = m_ui.totalLeafInstanceCount;
    const uint32_t drawn  = m_ui.visibleLeafInstanceCount;
    const uint32_t culled = (total >= drawn) ? (total - drawn) : 0;

    ImGui::Text("Total leaf instances: %s",                                       _fmtCount(total));
    ImGui::Text("Post-cull:            %s   (%s culled, incl. leaves on billboarded trees)",
                _fmtCount(drawn), _fmtCount(culled));
    ImGui::Spacing();
    ImGui::Text("Shadow draws:         %s   (cascade-summed)", _fmtCount(m_ui.shadowVisibleLeafInstanceCount));

    if (m_ui.activePipeline == Pipeline::MeshShader) {
        ImGui::Spacing();
        ImGui::TextDisabled("-- mesh-shader AS --");
        const uint32_t asDisp = m_ui.leafMainMeshletsDispatched;
        const uint32_t asRend = m_ui.leafMainMeshletsRendered;
        const uint32_t asCull = (asDisp >= asRend) ? (asDisp - asRend) : 0;
        ImGui::Text("Total AS post-instance: %s",                  _fmtCount(asDisp));
        ImGui::Text("Post-AS culling:        %s   (%s culled)",    _fmtCount(asRend), _fmtCount(asCull));
        ImGui::Text("Shadow AS draws:        %s",                  _fmtCount(m_ui.leafShadowMeshletsRendered));
    }

    ImGui::Unindent();
}

void UIRenderer::_buildTerrainFunnel() {
    ImGui::Text("Terrain (P2)");
    ImGui::Indent();

    const uint32_t total        = m_ui.totalTerrainMeshletCount;
    const uint32_t drawn        = m_ui.visibleTerrainMeshletCount;
    const uint32_t culled       = (total >= drawn) ? (total - drawn) : 0;
    const uint32_t shadowDrawn  = m_ui.shadowVisibleTerrainMeshletCount;

    ImGui::Text("Total meshlets:       %s",                  _fmtCount(total));
    ImGui::Text("Post-cull:            %s   (%s culled)",    _fmtCount(drawn), _fmtCount(culled));
    ImGui::Text("Shadow (\xCE\xA3 cascades): %s",            _fmtCount(shadowDrawn));

    ImGui::Unindent();
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

    // Cascade source: SDSM (GPU) when available, else CPU PSSM seed (Traditional pipeline,
    // pre-first-readback frames). Both share m_ViewHandler->worldToLight so the instance
    // transform is identical.
    dm::box3 cascadeBboxLS[Render::c_NumCascades];
    if (m_ui.sdsmDebugValid) {
        for (uint32_t c = 0; c < Render::c_NumCascades; ++c) {
            cascadeBboxLS[c] = dm::box3{
                dm::float3(m_ui.sdsmShadowCasterMinLS[c][0], m_ui.sdsmShadowCasterMinLS[c][1], m_ui.sdsmShadowCasterMinLS[c][2]),
                dm::float3(m_ui.sdsmShadowCasterMaxLS[c][0], m_ui.sdsmShadowCasterMaxLS[c][1], m_ui.sdsmShadowCasterMaxLS[c][2])
            };
        }
    } else {
        for (uint32_t c = 0; c < Render::c_NumCascades; ++c)
            cascadeBboxLS[c] = m_ViewHandler->cascades[c].shadowCasterBboxLS;
    }

    // Lowest-index cascade whose light-space shadow-caster bbox overlaps this instance's
    // world-space bbox transformed into light space. Mirrors CullShadow in CullCS.hlsl.
    auto firstCascadeFor = [&](const dm::box3& wsBbox) -> int {
        dm::box3 lsBbox = wsBbox * worldToLight;
        for (uint32_t c = 0; c < Render::c_NumCascades; ++c) {
            if (!cascadeBboxLS[c].isempty() && lsBbox.intersects(cascadeBboxLS[c])) return int(c);
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
        // Reserve room below the image for the texel-size readout.
        const float kFooterHeight = ImGui::GetTextLineHeightWithSpacing() * (numCascades + 2);
        float size  = std::max(std::min(avail.x, avail.y - kFooterHeight), 64.f);
        ImGui::Image(ImTextureRef(m_ui.selectedCascadeTexture), ImVec2(size, size));
    }

    ImGui::Separator();
    ImGui::TextDisabled("Texel size (world units / shadow texel):");
    for (int c = 0; c < numCascades; ++c) {
        float t = m_ui.cascadeTexelSize[c];
        bool  isSel = (c == m_ui.selectedCascade);
        if (t > 0.f) {
            if (isSel) ImGui::Text("> C%d: %.4f", c, t);
            else       ImGui::Text("  C%d: %.4f", c, t);
        } else {
            if (isSel) ImGui::TextDisabled("> C%d: (empty)", c);
            else       ImGui::TextDisabled("  C%d: (empty)", c);
        }
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

    // Channel layout (RGBA32_FLOAT):
    //   .r = raw farthest          -> Hi-Z occlusion (sky included)
    //   .g = nearest                -> SDSM near
    //   .b = sky-excluded farthest -> SDSM far
    int chan = static_cast<int>(m_ui.hizDebugChannel);
    const char* channels[] = { "All (RGB)", "R: Hi-Z (sky inc.)", "G: Nearest", "B: SDSM far (sky excl.)" };
    ImGui::Text("Channel:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(220.f);
    if (ImGui::Combo("##hizchan", &chan, channels, IM_ARRAYSIZE(channels)))
        m_ui.hizDebugChannel = static_cast<UIData::HiZDebugChannel>(chan);

    ImGui::Separator();

    void* tex = m_ui.hizMipTextures[selectedMip];
    if (tex) {
        ImVec4 tint;
        ImVec2 avail = ImGui::GetContentRegionAvail();
        float size  = std::max(std::min(avail.x, avail.y), 64.f);
        switch (m_ui.hizDebugChannel) {
            case UIData::HiZDebugChannel::R_Farthest: tint = ImVec4(1, 0, 0, 1); break;
            case UIData::HiZDebugChannel::G_Nearest:  tint = ImVec4(0, 1, 0, 1); break;
            case UIData::HiZDebugChannel::B_SDSMFar:  tint = ImVec4(0, 0, 1, 1); break;
            case UIData::HiZDebugChannel::RGB:        tint = ImVec4(1, 1, 1, 1); break;
        }
        ImGui::ImageWithBg(ImTextureRef(tex), ImVec2(size, size), ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0), tint);
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
