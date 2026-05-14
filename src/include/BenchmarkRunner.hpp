#ifndef XYLEM_BENCHMARK_RUNNER_H
#define XYLEM_BENCHMARK_RUNNER_H

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <donut/app/ApplicationBase.h>
#include <donut/core/math/math.h>
#include <donut/engine/KeyframeAnimation.h>

#include "UIData.hpp"
#include "ViewHandler.hpp"

namespace Xylem {

// ---------------------------------------------------------------------------
// CameraPath
//
// Loads a benchmark camera path from a JSON file and exposes time-sampled
// evaluation of position and look direction.  Internally the path is stored
// as two Donut animation tracks inside a Sequence:
//   "position"  — CatmullRomSpline on dm::float3 (w = 0)
//   "rotation"  — Slerp on a quaternion derived from each waypoint's lookDir
//
// The Evaluate method is non-const because donut::engine::animation::Sequence
// ::Evaluate is non-const (it does lazy cache work internally).
// ---------------------------------------------------------------------------
class CameraPath
{
public:
    struct Waypoint
    {
        float       time     = 0.f;       // seconds; monotonic-increasing across the list
        dm::float3  position = dm::float3(0.f, 0.f, 0.f);
        dm::float3  lookDir  = dm::float3(0.f, 0.f, 1.f);  // unit vector; default = canonical +Z
    };

    struct Sample
    {
        dm::float3 position;
        dm::float3 lookDir;  // unit vector
    };

    bool               IsLoaded()            const { return m_Loaded; }
    float              GetDurationSeconds()  const { return m_DurationSeconds; }

    // Evaluate the path at the given simulation time (seconds).
    // Returns nullopt if not loaded or time is out of range.
    // Non-const because Sequence::Evaluate is non-const.
    std::optional<Sample> Evaluate(float simTimeSeconds);

    // ---- Editing ----
    // Each mutator rebuilds m_Sequence from m_Waypoints, refreshes m_DurationSeconds,
    // and updates m_Loaded (true iff size() >= 2 and Sequence build succeeded).
    // SetTime intentionally does NOT enforce monotonicity — the user typing into the
    // first row could otherwise lock themselves out. Monotonicity is validated by
    // SaveAs and by BenchmarkRunner before a run begins.
    void Append(const Waypoint& wp);
    void Erase(size_t idx);
    void SetTime(size_t idx, float t);

    // Wholesale replace m_Waypoints (used when the user picks a different
    // named entry from the collection dropdown).
    void SetWaypoints(const std::vector<Waypoint>& wps);

    const std::vector<Waypoint>& GetWaypoints() const { return m_Waypoints; }

private:
    bool                                       m_Loaded          = false;
    float                                      m_DurationSeconds = 0.f;
    donut::engine::animation::Sequence         m_Sequence;
    std::vector<Waypoint>                      m_Waypoints;

    // Rebuilds m_Sequence from m_Waypoints + refreshes m_DurationSeconds and m_Loaded.
    void _RebuildSequence();
};

// ---------------------------------------------------------------------------
// CameraPathFile
//
// On-disk camera-path collection. The JSON file holds:
//   {
//     "simulationDtMs": <float>,
//     "warmupFrames":   <uint>,
//     "paths": {
//       "<name>": { "waypoints": [ <Waypoint>, ... ] },
//       ...
//     }
//   }
// The bundled jsoncpp (1.9.6) serializes object keys alphabetically, and
// getMemberNames() returns sorted order. We keep m_Entries sorted by name on
// every SetPath so the in-memory dropdown order matches what Save+Reload
// produces (avoids one-shot "new entry appears at the end then jumps after
// reload" UX surprise).
// ---------------------------------------------------------------------------
class CameraPathFile {
public:
    bool Load(const std::filesystem::path& jsonPath);
    bool Save(const std::filesystem::path& jsonPath);

    std::vector<std::string> GetNames() const;
    bool                     Has(const std::string& name) const;

    // Returns nullptr if name is not present.
    const std::vector<CameraPath::Waypoint>* GetPath(const std::string& name) const;

    // Replaces in-place if name exists; otherwise inserts in alphabetical
    // position to match what Save+Reload would produce.
    void SetPath(const std::string& name,
                 const std::vector<CameraPath::Waypoint>& waypoints);

    bool Empty() const { return m_Entries.empty(); }

    float              GetSimulationDtMs() const { return m_SimulationDtMs; }
    uint32_t           GetWarmupFrames()   const { return m_WarmupFrames; }
    const std::string& GetLastError()      const { return m_LastError; }

private:
    struct Entry {
        std::string name;
        std::vector<CameraPath::Waypoint> waypoints;
    };
    std::vector<Entry>                      m_Entries;
    std::unordered_map<std::string, size_t> m_NameToIdx;

    float       m_SimulationDtMs = 16.6667f;
    uint32_t    m_WarmupFrames   = 30;
    std::string m_LastError;
};

// ---------------------------------------------------------------------------
// BenchmarkRunner
//
// Drives an automated benchmark flythrough: cycles through all three
// pipelines, records GPU/CPU frame times along the camera path, and writes
// per-pipeline CSVs + a run summary.  The class is declared here in full
// (all State enum values) so that Task 10's state-machine implementation
// does not need to touch the header again.
// ---------------------------------------------------------------------------
class BenchmarkRunner {
public:
    enum class State {
        Idle,
        StartPipeline,
        WarmupFrames,
        Recording,
        DrainTimers,
        FlushPipelineCsv,
        WriteRunSummary,
        RestoreState
    };

    BenchmarkRunner(donut::app::DeviceManager* dm,
                    UIData& ui,
                    ViewHandler& view);

    // Loads the path file once at startup. Reports success/failure via
    // m_UI.benchmarkLastStatus.
    void Init(std::filesystem::path pathFile = "scene/benchmark_path.json");

    // Called from RenderOrchestrator::Animate before camera animation.
    // Returns true if the runner is currently overriding the camera
    // (orchestrator should suppress its own camera.Animate call).
    bool PreAnimate(float seconds);

    // Called from RenderOrchestrator::Render after the active pass renders
    // and before the UI pass renders.
    void PostRender();

    bool IsRunning() const { return m_State != State::Idle; }

private:
    donut::app::DeviceManager* m_DeviceManager = nullptr;
    UIData&                    m_UI;
    ViewHandler&               m_View;
    State                      m_State = State::Idle;
    CameraPath                 m_Path;
    CameraPathFile             m_File;
    std::string                m_ActiveName;
    std::filesystem::path      m_PathFile;

    struct MetricsRow {
        uint32_t   frameIdx   = 0;
        float      simTimeMs  = 0.f;
        float      cpuMs      = 0.f;
        float      gpuMs      = -1.f;
        // Per-stage GPU times in ms, indexed by frame::FrameStage.
        // -1 = stage not run by this row's pipeline (or not yet bound).
        // Populated atomically with gpuMs in _LateBindGpuTimes.
        float      gpuStageMs[static_cast<size_t>(frame::FrameStage::COUNT)] =
            { -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f };
        bool       gpuMsBound = false;  // false until late-binding fills gpuMs
        dm::float3 camPos{};
        dm::float3 camDir{};
    };

    // Scene-content totals. Snapshotted from UIData when each pipeline's
    // recording window closes (in _WritePipelineCsv). P0/P1 zero out
    // trunkMeshlets / terrainMeshlets every frame (no meshlet concept), so the
    // run summary takes the max across all PipelineRun snapshots — that picks
    // up P2's values if P2 ran in the session, and surfaces 0 otherwise.
    struct SceneStatsSnapshot {
        uint32_t trunkInstances   = 0;
        uint32_t trunkMeshlets    = 0; // P2-only
        uint32_t leafInstances    = 0;
        uint32_t leafMeshlets     = 0;
        uint32_t terrainVerts     = 0;
        uint32_t terrainMeshlets  = 0; // P2-only
    };

    // Per-pipeline run state, reset on each StartPipeline transition.
    struct PipelineRun {
        Pipeline                pipeline            = Pipeline::Traditional;
        uint32_t                warmupFrameCount    = 0;
        uint32_t                recordedFrameCount  = 0;
        uint32_t                drainFrameCount     = 0;
        float                   simTimeSeconds      = 0.f;
        std::vector<MetricsRow> rows;
        bool                    completed           = false;
        SceneStatsSnapshot      sceneStats{};
    };

    // Session-scoped state, lives for the whole back-to-back run.
    struct Session {
        std::filesystem::path        outputFolder;
        std::string                  timestampStr;
        std::vector<PipelineRun>     pipelines;       // 3 entries, one per Pipeline
        size_t                       currentIdx                = 0;
        Pipeline                     savedRequestedPipeline    = Pipeline::Traditional;
        bool                         savedVsync                = false;
        bool                         vsyncWasOverridden        = false;
        bool                         aborted                   = false;
    };

    std::optional<Session> m_Session;

    void _BeginSession();
    void _EndSession();
    void _StartNextPipeline();
    void _OverrideCameraThisFrame();
    void _AppendRowFromUI();
    void _LateBindGpuTimes();
    void _UpdateProgressLabel();
    void _WritePipelineCsv(const PipelineRun& run);
    void _WriteRunSummary();
    void _ServiceWaypointEdits();
    void _PublishWaypointSnapshot();

    // After m_File has been (re)loaded, pick the active entry — preferring
    // `preferred` if non-empty and present, otherwise the first entry — and
    // apply it to m_Path. Updates m_ActiveName. If m_File is empty, clears
    // m_ActiveName and gives m_Path an empty waypoint list.
    void _AdoptActiveEntry(const std::string& preferred);
};

} // namespace Xylem

#endif // XYLEM_BENCHMARK_RUNNER_H
