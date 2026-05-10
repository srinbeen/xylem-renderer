#ifndef XYLEM_BENCHMARK_RUNNER_H
#define XYLEM_BENCHMARK_RUNNER_H

#include <filesystem>
#include <optional>
#include <string>
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
    struct Sample
    {
        dm::float3 position;
        dm::float3 lookDir;  // unit vector
    };

    // Load (or reload) the path from a JSON file.  Returns true on success.
    // On failure the reason is available via GetLastError().
    bool Load(const std::filesystem::path& jsonPath);

    bool               IsLoaded()            const { return m_Loaded; }
    float              GetDurationSeconds()  const { return m_DurationSeconds; }
    float              GetSimulationDtMs()   const { return m_SimulationDtMs; }
    uint32_t           GetWarmupFrames()     const { return m_WarmupFrames; }
    const std::string& GetName()             const { return m_Name; }
    const std::string& GetLastError()        const { return m_LastError; }

    // Evaluate the path at the given simulation time (seconds).
    // Returns nullopt if not loaded or time is out of range.
    // Non-const because Sequence::Evaluate is non-const.
    std::optional<Sample> Evaluate(float simTimeSeconds);

private:
    bool                                       m_Loaded          = false;
    std::string                                m_Name;
    std::string                                m_LastError;
    float                                      m_DurationSeconds = 0.f;
    float                                      m_SimulationDtMs  = 16.6667f;
    uint32_t                                   m_WarmupFrames    = 30;
    donut::engine::animation::Sequence         m_Sequence;
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
    std::filesystem::path      m_PathFile;

    struct MetricsRow {
        uint32_t   frameIdx   = 0;
        float      simTimeMs  = 0.f;
        float      cpuMs      = 0.f;
        float      gpuMs      = -1.f;
        bool       gpuMsBound = false;  // false until late-binding fills gpuMs
        dm::float3 camPos{};
        dm::float3 camDir{};
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
};

} // namespace Xylem

#endif // XYLEM_BENCHMARK_RUNNER_H
