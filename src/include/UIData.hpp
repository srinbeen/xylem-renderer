#ifndef XYLEM_UI_DATA_H
#define XYLEM_UI_DATA_H

#include <cstdint>
#include <string>
#include <vector>

#include "Render.hpp"
#include "frame/FrameStages.hpp"

namespace Xylem {

enum class Pipeline { Traditional = 0, Compute = 1, MeshShader = 2 };

struct UIData {
    // Cap on per-LOD breakdown surfaced to the UI. The pipelines today configure
    // 3 LODs (16/8/4 segments); 8 leaves headroom without dragging an allocation
    // into the per-frame UI write path.
    static constexpr uint32_t kMaxLodsForUI = 8;

    bool ShowUI = true;

    Pipeline activePipeline    = Pipeline::Traditional;
    Pipeline requestedPipeline = Pipeline::Traditional;

    float    gpuFrameTimeMs       = -1.0f; // -1 = not yet available
    float    cpuRenderTimeMs      = 0.0f;

    // Sun elevation above the horizon, degrees. Populated by RenderOrchestrator
    // right after advanceSunSky() so every per-frame consumer (BenchmarkRunner,
    // future UI overlays) reads a fresh value derived from the current
    // SunSkyState::lightDir. Range: roughly [-90, 90]; negative = below horizon.
    float    sunElevationDeg      = 0.0f;

    // Per-stage GPU timing in milliseconds. Indexed by frame::FrameStage.
    // -1 sentinel means "no value yet" — either the active pipeline doesn't
    // run this stage, or the timer ring is still warming up after a switch.
    // Reset to all -1 in RenderOrchestrator::_switchPipelineIfNeeded.
    float    gpuStageTimeMs[static_cast<size_t>(frame::FrameStage::COUNT)] =
        { -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f };
    uint32_t visibleInstanceCount = 0;
    uint32_t impostorVisibleCount = 0;
    uint32_t totalInstanceCount   = 0;
    uint32_t culledInstanceCount  = 0;

    // Per-LOD geometric instance counts (excludes impostors). Indexed by LOD,
    // 0 = highest detail. Σ over [0..lodCountForUI) + impostorVisibleCount ==
    // visibleInstanceCount. Populated by every render pass each frame.
    uint32_t lodCountForUI                       = 0;
    uint32_t lodVisibleCounts[kMaxLodsForUI]     = {};
    uint32_t shadowVisibleCount     = 0; // unique instances contributing to any cascade
    uint32_t shadowCulledCount      = 0; // totalInstanceCount - shadowVisibleCount
    uint32_t shadowCascadeDrawCount = 0; // sum over all cascades of per-cascade visible (counts overdraw)
    uint32_t shadowOverdrawCount    = 0; // shadowCascadeDrawCount - shadowVisibleCount

    // Shadow draws by cascade. Geometry and billboard tracks are kept
    // separate so the UI can show "<cascade total> (<billboarded>)" per row.
    // Indexed by cascade [0..Render::c_NumCascades). All-zero in a fresh
    // frame; populated by each render pass's per-frame UI write.
    //   shadowCascadeDrawCount     = Σ_c shadowGeomDrawsPerCascade[c]      (sanity identity)
    //   shadowImpostorVisibleCount = Σ_c shadowBillboardDrawsPerCascade[c] (sanity identity)
    uint32_t shadowGeomDrawsPerCascade[Render::c_NumCascades]      = {};
    uint32_t shadowBillboardDrawsPerCascade[Render::c_NumCascades] = {};

    // Scene-wide leaf totals (CPU-derived capacity) and per-frame visible counts.
    // visibleLeafInstanceCount / shadowVisibleLeafInstanceCount are written by
    // every render pass each frame, but their semantics differ by pipeline:
    //   - P0 (Traditional): CPU tally — sum of leafCount over draw-issued trunks.
    //   - P1 (Compute):     CPU tally — sum of leafCount over cull-survived trunks.
    //   - P2 (MeshShader):  GPU readback — Σ meta.y from leaf_as / leaf_shadow_as
    //                       AS-cull atomics (i.e. post-AS, so smaller than the
    //                       CPU tally would imply for the same trunk set).
    // The shadow variant is cascade-summed under all three pipelines.
    uint32_t totalLeafInstanceCount        = 0;  // Σ leafCount per instance
    uint32_t totalLeafMeshletCount         = 0;  // Σ leafMeshletCount per instance
    uint32_t visibleLeafInstanceCount      = 0;  // see semantics note above
    uint32_t shadowVisibleLeafInstanceCount = 0; // see semantics note above

    bool showDebugTopDown       = false;
    bool showDebugShadowTopDown = false;
    bool showShadowMap          = false;
    bool showHiZ                = false;
    bool showImpostorAtlas      = false;

    bool  hizEnabled       = true;

    float pssmLambda       = 0.85f; // PSSM/SDSM blend: 0=linear splits, 1=logarithmic splits

    // Shadow impostor tier (P1 + P2). When false, cull falls back to
    // geometry-only shadows for A/B comparison.
    bool     showShadowImpostors      = true;
    uint32_t shadowImpostorVisibleCount = 0;     // total across (asset, cascade)

    // Mesh-shader pipeline stats
    float    meshletMegaBufferMB  = 0.0f;
    uint32_t totalMeshletCount    = 0;
    // Per-instance-fanned trunk meshlet workload at LOD 0: Σ over scene instances
    // of their asset's LOD 0 meshlet count. This is the max possible per-frame
    // trunk-meshlet "dispatched" count (every instance visible, at full detail),
    // parallel in semantics to totalLeafMeshletCount. P2-only.
    uint32_t totalTrunkInstanceMeshletCount = 0;
    uint32_t totalTerrainVertexCount    = 0;
    uint32_t totalTerrainMeshletCount         = 0;
    uint32_t visibleTerrainMeshletCount       = 0;
    // Sum of terrain meshlets that survived AS-side light-frustum cull across
    // ALL cascades in the shadow pass. P2-only.
    uint32_t shadowVisibleTerrainMeshletCount = 0;

    // Mesh-shader pipeline meshlet AS cull stats (P2 only, GPU readback,
    // k_QueuedFrames-1 latency). "Dispatched" reflects only post-instance-cull
    // visible instances × per-slot meshlet count; AS-level cone+Hi-Z reduces
    // this further to "rendered". Shadow rows have no per-meshlet cull, so
    // dispatched == rendered (culled = 0) on the shadow path.
    uint32_t trunkMainMeshletsDispatched   = 0;
    uint32_t trunkMainMeshletsRendered     = 0;
    uint32_t trunkShadowMeshletsDispatched = 0;
    uint32_t trunkShadowMeshletsRendered   = 0;
    uint32_t leafMainMeshletsDispatched    = 0;
    uint32_t leafMainMeshletsRendered      = 0;
    uint32_t leafShadowMeshletsDispatched  = 0;
    uint32_t leafShadowMeshletsRendered    = 0;

    // SDSM debug readback (populated when Hi-Z is enabled and SDSM ran).
    bool     sdsmDebugValid       = false;
    float    sdsmNearDepthVal     = 0.f;
    float    sdsmFarDepthVal      = 0.f;
    float    sdsmTightNear        = 0.f;
    float    sdsmTightFar         = 0.f;
    float    sdsmCascadeSplits[4] = { 0.f, 0.f, 0.f, 0.f };
    // Per-cascade light-space shadow caster AABB written by SDSMBuildCascades.hlsl
    // (xyz min/max). Used by the shadow top-down debug visualizer.
    float    sdsmShadowCasterMinLS[4][3] = {};
    float    sdsmShadowCasterMaxLS[4][3] = {};

    // World-space size of one shadow texel per cascade (cascade XY extent / shadowRes).
    // Populated by every render pass after its cascade fit; 0 if cascade is empty.
    float    cascadeTexelSize[Render::c_NumCascades] = {};

    void*    shadowMapTexture = nullptr;  // nvrhi::ITexture*, set by active render pass
    // Single per-pass scratch texture into which the active pass copies the
    // currently-selected cascade slice. UIRenderer drives selectedCascade via the slider.
    int      selectedCascade        = 0;
    void*    selectedCascadeTexture = nullptr;
    // Hi-Z: one nvrhi::ITexture* per mip level (single-mip scratch textures), set by the active render pass.
    // Each entry is a separate RGBA32_FLOAT texture containing exactly one mip, copied each frame.
    //   .r = raw farthest (Hi-Z occlusion, sky included)
    //   .g = nearest      (SDSM near)
    //   .b = sky-excluded farthest (SDSM far)
    std::vector<void*> hizMipTextures;
    enum class HiZDebugChannel : int { RGB = 0, R_Farthest = 1, G_Nearest = 2, B_SDSMFar = 3 };
    HiZDebugChannel hizDebugChannel = HiZDebugChannel::RGB;

    // Impostor atlas debug: one entry per (asset, view) slice. Set by ComputeRenderPass after the
    // bake completes. Layout: index = assetIdx * impostorViewsPerAsset + viewIdx.
    uint32_t impostorViewsPerAsset = 0;
    uint32_t impostorAzimuthViews  = 0;
    uint32_t impostorElevationViews = 0;
    uint32_t impostorAssetCount    = 0;
    uint32_t impostorSelectedAsset = 0;
    void*    impostorAlbedoAtlasTexture = nullptr;
    void*    impostorNormalAtlasTexture = nullptr;
    void*    impostorDepthAtlasTexture = nullptr;

    // --- Scene Save / Load ---
    // UIRenderer writes; RenderOrchestrator reads + clears.
    bool        requestedSceneLoad     = false;
    std::string requestedScenePath;
    // Status surface for the Scene File UI section. Empty = no message displayed.
    // Save writes this directly from UIRenderer; Load writes it from the orchestrator
    // after _loadSceneIfRequested completes.
    std::string sceneLoadStatus;
    bool        sceneLoadStatusIsError = false;

    // --- Benchmark ---
    // UIRenderer writes the *Requested flags; BenchmarkRunner clears them and
    // writes the read-only progress/status fields. Mirrors the scene-load
    // pattern above so UIRenderer never directly references BenchmarkRunner.
    bool        benchmarkRunRequested        = false;  // UIRenderer writes; runner clears
    bool        benchmarkCancelRequested     = false;
    bool        benchmarkPathReloadRequested = false;
    bool        benchmarkInProgress          = false;  // runner writes
    std::string benchmarkProgressLabel;                // runner writes (e.g. "Pipeline 1/3 ...")
    std::string benchmarkLastStatus;                   // runner writes on completion
    bool        benchmarkLastStatusIsError   = false;

    // --- Benchmark waypoint editor ---
    // UIRenderer writes the *Requested fields; BenchmarkRunner clears them in
    // PreAnimate when idle, after applying the edit. Index sentinels are -1.
    bool        benchmarkCaptureWaypointRequested = false;
    int         benchmarkDeleteWaypointIndex      = -1;
    int         benchmarkPreviewWaypointIndex     = -1;

    // Inline per-row time edits. UIRenderer mutates entries in-place via
    // ImGui::InputFloat; runner diffs against CameraPath::GetWaypoints()[i].time
    // each idle frame and pushes any change through CameraPath::SetTime.
    // Runner resizes/refills this vector on snapshot-size changes (capture /
    // delete / load / save-as).
    std::vector<float> benchmarkWaypointTimesEdited;

    // Read-only snapshot of the current waypoint list. Runner rewrites every
    // PreAnimate when idle. UI iterates this for per-row display.
    struct WaypointSnapshot { float t; float px, py, pz; float dx, dy, dz; };
    std::vector<WaypointSnapshot> benchmarkWaypoints;

    // Runner-published filename of the currently loaded path (drives the
    // "Path: ..." indicator in the UI). Updated on Init, on Reload, and on
    // a successful Save-As (because Save-As updates m_PathFile).
    std::string benchmarkPathFileName;

    // --- Pipeline selection for the next benchmark run ---
    // UIRenderer mutates via checkboxes; runner reads at _BeginSession to
    // filter the pipelines vector. Index order matches the Pipeline enum
    // (Traditional=0, Compute=1, MeshShader=2).
    bool benchmarkEnabledPipelines[3] = { true, true, true };

    // --- Benchmark path collection (replaces Save-As flow) ---
    // UIRenderer writes the *Requested fields + select-name; runner clears them
    // in PreAnimate when idle, after applying. Name field is a pure backing
    // string for the ImGui InputText — runner reads it on Save.
    std::string benchmarkNameField;        // UI input; runner consumes on Save
    bool        benchmarkSaveRequested        = false;
    bool        benchmarkSelectRequested      = false;
    std::string benchmarkSelectName;       // dropdown pick; runner reads on Service

    // Runner publishes these every idle frame for the dropdown + name auto-sync.
    std::vector<std::string> benchmarkPathNames;
    std::string              benchmarkActivePathName;
};

} // namespace Xylem

#endif // XYLEM_UI_DATA_H
