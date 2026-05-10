#include "include/BenchmarkRunner.hpp"
#include "include/Globals.hpp"
#include "include/frame/FrameLifecycle.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <memory>
#include <utility>

#include <donut/core/json.h>
#include <donut/core/log.h>
#include <donut/core/math/quat.h>

using namespace Xylem;
using namespace donut::engine::animation;

// ---------------------------------------------------------------------------
// File-local helpers
// ---------------------------------------------------------------------------
namespace {

// Build the quaternion that rotates canonical +Z onto `forward`.
// `forward` must already be normalised.
dm::quat QuatFromForward(dm::float3 forward)
{
    const dm::float3 z(0.f, 0.f, 1.f);
    const float d = dm::dot(z, forward);
    if (d > 0.9999f)
        return dm::quat::fromWXYZ(1.f, dm::float3(0.f, 0.f, 0.f));
    if (d < -0.9999f)
        return dm::rotationQuat(dm::float3(0.f, 1.f, 0.f), dm::PI_f);
    const dm::float3 axis  = dm::normalize(dm::cross(z, forward));
    const float      angle = std::acos(dm::clamp(d, -1.f, 1.f));
    return dm::rotationQuat(axis, angle);
}

// Recover the +Z-relative forward direction from a quaternion.
dm::float3 ForwardFromQuat(const dm::quat& q)
{
    return dm::normalize(dm::applyQuat(q, dm::float3(0.f, 0.f, 1.f)));
}

// Read a [x, y, z] JSON array into a dm::float3.
// Returns dm::float3(0) and logs a warning if the node is malformed.
dm::float3 ReadFloat3(const Json::Value& parent, const char* key)
{
    const Json::Value& node = parent[key];
    if (!node.isArray() || node.size() < 3)
    {
        donut::log::warning("CameraPath: '%s' is not a 3-element array; defaulting to zero.", key);
        return dm::float3(0.f);
    }
    return dm::float3(
        node[0].asFloat(),
        node[1].asFloat(),
        node[2].asFloat());
}

std::string MakeTimestampString()
{
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
    localtime_s(&tm, &now);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", &tm);
    return buf;
}

const char* PipelineFileName(Xylem::Pipeline p)
{
    switch (p) {
        case Xylem::Pipeline::Traditional: return "traditional.csv";
        case Xylem::Pipeline::Compute:     return "compute.csv";
        case Xylem::Pipeline::MeshShader:  return "meshshader.csv";
    }
    return "unknown.csv";
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// CameraPath::Load
// ---------------------------------------------------------------------------
bool CameraPath::Load(const std::filesystem::path& jsonPath)
{
    // Reset all state.
    m_Loaded          = false;
    m_LastError.clear();
    m_Name.clear();
    m_DurationSeconds = 0.f;
    m_SimulationDtMs  = 16.6667f;
    m_WarmupFrames    = 30;
    m_Sequence        = Sequence{};

    // Open the file.
    std::ifstream jsonStream(jsonPath);
    if (!jsonStream.is_open())
    {
        m_LastError = "could not open " + jsonPath.string();
        return false;
    }

    // Parse JSON.
    Json::CharReaderBuilder reader;
    Json::Value root;
    std::string parseErrors;
    if (!Json::parseFromStream(reader, jsonStream, &root, &parseErrors))
    {
        m_LastError = "JSON parse failed: " + parseErrors;
        return false;
    }

    // Top-level scalar fields.
    m_Name           = root.get("name", "unnamed").asString();
    m_SimulationDtMs = root.get("simulationDtMs", 16.6667f).asFloat();
    m_WarmupFrames   = root.get("warmupFrames", 30u).asUInt();

    if (m_SimulationDtMs <= 0.f)
    {
        m_LastError = "simulationDtMs must be > 0";
        return false;
    }

    // Waypoints array.
    const Json::Value& waypoints = root["waypoints"];
    if (!waypoints.isArray())
    {
        m_LastError = "waypoints must be a JSON array";
        return false;
    }
    if (waypoints.size() < 2)
    {
        m_LastError = "waypoints must contain at least 2 entries";
        return false;
    }

    // Build Sampler tracks.
    auto positionTrack = std::make_shared<Sampler>();
    auto rotationTrack = std::make_shared<Sampler>();
    positionTrack->SetInterpolationMode(InterpolationMode::CatmullRomSpline);
    rotationTrack->SetInterpolationMode(InterpolationMode::Slerp);

    float prevTime = -std::numeric_limits<float>::infinity();

    for (Json::ArrayIndex i = 0; i < waypoints.size(); ++i)
    {
        const Json::Value& wp = waypoints[i];

        const float t = wp.get("time", 0.f).asFloat();
        if (t <= prevTime)
        {
            m_LastError = "waypoint times must be strictly increasing (failed at index "
                          + std::to_string(i) + ")";
            return false;
        }
        prevTime = t;

        const dm::float3 pos = ReadFloat3(wp, "position");

        dm::float3 dir = ReadFloat3(wp, "lookDir");
        const float dirLen = dm::length(dir);
        if (dirLen < 1e-6f)
        {
            m_LastError = "waypoint lookDir is degenerate (zero length) at index "
                          + std::to_string(i);
            return false;
        }
        dir = dir / dirLen;

        const dm::quat q = QuatFromForward(dir);

        Keyframe pk;
        pk.time  = t;
        pk.value = dm::float4(pos.x, pos.y, pos.z, 0.f);
        positionTrack->AddKeyframe(pk);

        Keyframe rk;
        rk.time  = t;
        rk.value = dm::float4(q.x, q.y, q.z, q.w);
        rotationTrack->AddKeyframe(rk);
    }

    m_Sequence.AddTrack("position", positionTrack);
    m_Sequence.AddTrack("rotation", rotationTrack);

    m_DurationSeconds = m_Sequence.GetDuration();
    m_Loaded          = true;
    return true;
}

// ---------------------------------------------------------------------------
// CameraPath::Evaluate
// ---------------------------------------------------------------------------
std::optional<CameraPath::Sample> CameraPath::Evaluate(float simTimeSeconds)
{
    if (!m_Loaded)
        return std::nullopt;

    if (simTimeSeconds >= m_DurationSeconds)
        return std::nullopt;

    std::optional<dm::float4> pv = m_Sequence.Evaluate("position", simTimeSeconds);
    std::optional<dm::float4> rv = m_Sequence.Evaluate("rotation", simTimeSeconds);

    if (!pv || !rv)
        return std::nullopt;

    Sample s;
    s.position = dm::float3(pv->x, pv->y, pv->z);
    s.lookDir  = ForwardFromQuat(dm::quat::fromXYZW(*rv));
    return s;
}

// ===========================================================================
// BenchmarkRunner
// ===========================================================================

BenchmarkRunner::BenchmarkRunner(donut::app::DeviceManager* dm,
                                 UIData& ui,
                                 ViewHandler& view)
    : m_DeviceManager(dm)
    , m_UI(ui)
    , m_View(view)
{
}

void BenchmarkRunner::Init(std::filesystem::path pathFile)
{
    // Anchor relative paths on the workspace root (matches g_SceneConfigDirectory
    // convention in Globals.hpp). The .exe's cwd is launcher-dependent and not
    // reliably the workspace root.
    if (pathFile.is_relative()) {
        pathFile = Xylem::g_ProjectDirectory / pathFile;
    }
    m_PathFile = std::move(pathFile);
    if (m_Path.Load(m_PathFile)) {
        m_UI.benchmarkLastStatus        = "Path loaded: " + m_PathFile.string();
        m_UI.benchmarkLastStatusIsError = false;
    } else {
        m_UI.benchmarkLastStatus        = "Path load failed: " + m_Path.GetLastError();
        m_UI.benchmarkLastStatusIsError = true;
    }
    donut::log::info("BenchmarkRunner: %s", m_UI.benchmarkLastStatus.c_str());
}

bool BenchmarkRunner::PreAnimate(float /*seconds*/)
{
    // 1) Honour UI inputs.
    if (m_UI.benchmarkPathReloadRequested) {
        m_UI.benchmarkPathReloadRequested = false;
        if (m_Path.Load(m_PathFile)) {
            m_UI.benchmarkLastStatus        = "Path reloaded: " + m_PathFile.string();
            m_UI.benchmarkLastStatusIsError = false;
        } else {
            m_UI.benchmarkLastStatus        = "Path reload failed: " + m_Path.GetLastError();
            m_UI.benchmarkLastStatusIsError = true;
        }
    }

    if (m_UI.benchmarkRunRequested) {
        m_UI.benchmarkRunRequested = false;
        if (m_State == State::Idle && m_Path.IsLoaded()) {
            _BeginSession();
        }
    }

    if (m_UI.benchmarkCancelRequested) {
        m_UI.benchmarkCancelRequested = false;
        if (m_State != State::Idle && m_Session) {
            m_Session->aborted = true;
        }
    }

    if (m_State == State::Idle) {
        m_UI.benchmarkInProgress = false;
        return false;
    }

    m_UI.benchmarkInProgress = true;

    if (m_Session && m_Session->aborted) {
        // Push current pipeline straight to FlushPipelineCsv (preserves what we have).
        if (m_State != State::WriteRunSummary && m_State != State::RestoreState) {
            m_State = State::FlushPipelineCsv;
        }
    }

    Session& sess = *m_Session;

    switch (m_State) {
        case State::StartPipeline: {
            // Request the switch; orchestrator picks it up in this same Animate
            // (because we run before _switchPipelineIfNeeded). On the next
            // frame, activePipeline will equal run.pipeline.
            PipelineRun& run = sess.pipelines[sess.currentIdx];
            m_UI.requestedPipeline = run.pipeline;
            m_State = State::WarmupFrames;
            run.warmupFrameCount = 0;
            run.simTimeSeconds = 0.f;
            return true;  // override camera even on this hand-off frame
        }
        case State::WarmupFrames:
        case State::Recording:
        case State::DrainTimers: {
            _OverrideCameraThisFrame();
            return true;
        }
        case State::FlushPipelineCsv:
        case State::WriteRunSummary:
        case State::RestoreState:
            // currentIdx may be past the end here (post-WriteRunSummary), so
            // do NOT index sess.pipelines. State transitions handled in PostRender.
            return false;
        default:
            return false;
    }
}

void BenchmarkRunner::PostRender()
{
    if (m_State == State::Idle || !m_Session) return;
    Session& sess = *m_Session;

    // _LateBindGpuTimes is safe only when currentIdx is in range — i.e., we
    // still have an active per-pipeline run. After the last pipeline's
    // FlushPipelineCsv increments currentIdx past the end, we're in
    // WriteRunSummary / RestoreState and there's nothing to bind.
    const bool hasActiveRun = sess.currentIdx < sess.pipelines.size();
    if (hasActiveRun) {
        _LateBindGpuTimes();
    }

    switch (m_State) {
        case State::WarmupFrames: {
            PipelineRun& run = sess.pipelines[sess.currentIdx];
            ++run.warmupFrameCount;
            if (run.warmupFrameCount >= m_Path.GetWarmupFrames()) {
                m_State = State::Recording;
                run.simTimeSeconds   = 0.f;
                run.recordedFrameCount = 0;
            }
            _UpdateProgressLabel();
            break;
        }
        case State::Recording: {
            PipelineRun& run = sess.pipelines[sess.currentIdx];
            // Append a row for the camera state we just rendered.
            _AppendRowFromUI();
            ++run.recordedFrameCount;

            // Advance simulated clock for the *next* frame.
            run.simTimeSeconds += m_Path.GetSimulationDtMs() / 1000.f;
            if (run.simTimeSeconds >= m_Path.GetDurationSeconds()) {
                m_State = State::DrainTimers;
                run.drainFrameCount = 0;
            }
            _UpdateProgressLabel();
            break;
        }
        case State::DrainTimers: {
            PipelineRun& run = sess.pipelines[sess.currentIdx];
            ++run.drainFrameCount;
            // Run drain frames until every recorded row has its gpuMs bound
            // OR we've drained more than k_QueuedFrames + 2 frames (cap so we
            // don't loop forever if a slot's GPU result never lands — e.g. on
            // very slow GPUs / tiny scenes).
            const bool allBound = std::all_of(run.rows.begin(), run.rows.end(),
                [](const MetricsRow& r) { return r.gpuMsBound; });
            if (allBound || run.drainFrameCount >= Xylem::frame::k_QueuedFrames + 2u) {
                run.completed = !sess.aborted;
                m_State = State::FlushPipelineCsv;
            }
            _UpdateProgressLabel();
            break;
        }
        case State::FlushPipelineCsv: {
            // currentIdx still points at the just-finished pipeline at this
            // point; _StartNextPipeline advances it.
            _WritePipelineCsv(sess.pipelines[sess.currentIdx]);
            _StartNextPipeline();
            break;
        }
        case State::WriteRunSummary: {
            _WriteRunSummary();
            m_State = State::RestoreState;
            break;
        }
        case State::RestoreState: {
            _EndSession();
            m_State = State::Idle;
            break;
        }
        default: break;
    }
}

// ---------------------------------------------------------------------------
// BenchmarkRunner helpers
// ---------------------------------------------------------------------------

void BenchmarkRunner::_BeginSession()
{
    Session sess;
    sess.timestampStr = MakeTimestampString();
    // Anchor on g_BinDirectory (workspace/bin/) so output lands in the same
    // place regardless of the launcher's cwd. Mirrors Init's path-resolution
    // convention.
    sess.outputFolder = Xylem::g_BinDirectory / "benchmarks" / sess.timestampStr;

    std::error_code ec;
    std::filesystem::create_directories(sess.outputFolder, ec);
    if (ec) {
        m_UI.benchmarkLastStatus        = "Could not create output folder: " + sess.outputFolder.string();
        m_UI.benchmarkLastStatusIsError = true;
        return;
    }

    sess.pipelines.push_back({Pipeline::Traditional, 0, 0, 0, 0.f, {}, false});
    sess.pipelines.push_back({Pipeline::Compute,     0, 0, 0, 0.f, {}, false});
    sess.pipelines.push_back({Pipeline::MeshShader,  0, 0, 0, 0.f, {}, false});

    sess.savedRequestedPipeline = m_UI.requestedPipeline;
    sess.savedVsync = m_DeviceManager->IsVsyncEnabled();
    if (sess.savedVsync) {
        m_DeviceManager->SetVsyncEnabled(false);
        sess.vsyncWasOverridden = true;
    }

    m_Session = std::move(sess);
    m_State = State::StartPipeline;
}

void BenchmarkRunner::_EndSession()
{
    if (!m_Session) return;
    if (m_Session->vsyncWasOverridden) {
        m_DeviceManager->SetVsyncEnabled(m_Session->savedVsync);
    }
    m_UI.requestedPipeline = m_Session->savedRequestedPipeline;
    if (m_Session->aborted) {
        m_UI.benchmarkLastStatus        = "Cancelled. Partial CSVs in: " + m_Session->outputFolder.string();
        m_UI.benchmarkLastStatusIsError = true;
    } else {
        m_UI.benchmarkLastStatus        = "Completed: " + m_Session->outputFolder.string();
        m_UI.benchmarkLastStatusIsError = false;
    }
    m_UI.benchmarkProgressLabel.clear();
    m_Session.reset();
}

void BenchmarkRunner::_StartNextPipeline()
{
    if (!m_Session) return;
    auto& sess = *m_Session;
    sess.currentIdx++;
    if (sess.currentIdx >= sess.pipelines.size() || sess.aborted) {
        m_State = State::WriteRunSummary;
    } else {
        m_State = State::StartPipeline;
    }
}

void BenchmarkRunner::_OverrideCameraThisFrame()
{
    if (!m_Session) return;
    const auto& run = m_Session->pipelines[m_Session->currentIdx];
    auto sample = m_Path.Evaluate(run.simTimeSeconds);
    if (!sample.has_value()) return;
    m_View.camera.LookTo(sample->position, sample->lookDir);
}

void BenchmarkRunner::_AppendRowFromUI()
{
    if (!m_Session) return;
    auto& run = m_Session->pipelines[m_Session->currentIdx];
    auto sample = m_Path.Evaluate(run.simTimeSeconds);

    MetricsRow r;
    r.frameIdx   = run.recordedFrameCount;
    r.simTimeMs  = run.simTimeSeconds * 1000.f;
    r.cpuMs      = m_UI.cpuRenderTimeMs;
    r.gpuMs      = -1.f;
    r.gpuMsBound = false;
    if (sample.has_value()) {
        r.camPos = sample->position;
        r.camDir = sample->lookDir;
    }
    run.rows.push_back(r);
}

void BenchmarkRunner::_LateBindGpuTimes()
{
    if (!m_Session) return;
    auto& run = m_Session->pipelines[m_Session->currentIdx];
    if (run.rows.empty()) return;

    // Bind the most recent successful GPU read to the earliest un-bound row.
    // This is best-effort alignment: with k_QueuedFrames in flight, the GPU
    // result for row N may arrive a few frames late. The DrainTimers state
    // keeps frames ticking until every row is bound.
    for (auto& r : run.rows) {
        if (!r.gpuMsBound) {
            // Sentinel: m_UI.gpuFrameTimeMs starts at -1.0f and is reset to
            // -1.0f on pipeline switch by the orchestrator. Only bind when
            // the active pass has reported a real value.
            if (m_UI.gpuFrameTimeMs >= 0.f) {
                r.gpuMs = m_UI.gpuFrameTimeMs;
                // Snapshot per-stage values atomically with gpuMs. They came
                // from the same frame's GPU submission and share alignment.
                // Stage slots that are -1 (pipeline doesn't run that stage,
                // or Hi-Z toggled off) are copied as-is and surface as -1
                // in the CSV.
                for (size_t s = 0; s < static_cast<size_t>(frame::FrameStage::COUNT); ++s) {
                    r.gpuStageMs[s] = m_UI.gpuStageTimeMs[s];
                }
                r.gpuMsBound = true;
            }
            return;  // bind only one row per frame to maintain alignment
        }
    }
}

void BenchmarkRunner::_UpdateProgressLabel()
{
    if (!m_Session) return;
    const auto& sess = *m_Session;
    const auto& run  = sess.pipelines[sess.currentIdx];
    const char* pname =
        run.pipeline == Pipeline::Traditional ? "Traditional" :
        run.pipeline == Pipeline::Compute     ? "Compute"     :
                                                "MeshShader";
    const char* phase =
        m_State == State::WarmupFrames ? "Warmup"    :
        m_State == State::Recording    ? "Recording" :
        m_State == State::DrainTimers  ? "Draining"  : "?";
    const uint32_t expectedFrames = static_cast<uint32_t>(
        m_Path.GetDurationSeconds() * 1000.f / m_Path.GetSimulationDtMs());

    char buf[160];
    std::snprintf(buf, sizeof(buf),
        "Pipeline %zu/%zu: %s | Frame %u / ~%u | %s",
        sess.currentIdx + 1, sess.pipelines.size(),
        pname,
        m_State == State::WarmupFrames ? run.warmupFrameCount : run.recordedFrameCount,
        m_State == State::WarmupFrames ? m_Path.GetWarmupFrames() : expectedFrames,
        phase);
    m_UI.benchmarkProgressLabel = buf;
}

void BenchmarkRunner::_WritePipelineCsv(const PipelineRun& run)
{
    if (!m_Session) return;

    // Partial CSVs (cancelled or incomplete) get a `.partial` suffix so an
    // automated consumer doesn't mistake them for a clean run.
    const std::string fname = run.completed
        ? PipelineFileName(run.pipeline)
        : (std::string(PipelineFileName(run.pipeline)) + ".partial");
    const std::filesystem::path full = m_Session->outputFolder / fname;

    std::ofstream out(full, std::ios::binary);  // binary => raw LF on Windows
    if (!out) {
        m_UI.benchmarkLastStatus        = "Failed to open " + full.string();
        m_UI.benchmarkLastStatusIsError = true;
        return;
    }

    // Stage columns inserted between gpuMs and camPosX. Order matches
    // frame::FrameStage enum (DepthPrepass=0 ... Scene=6). Stages a pipeline
    // doesn't run write -1; analyst code can mask them out as `df > 0`.
    out << "frameIdx,simTimeMs,cpuMs,gpuMs,"
           "depthPrepassMs,hizMs,sdsmMs,cullMs,shadowMs,skyMs,sceneMs,"
           "camPosX,camPosY,camPosZ,camDirX,camDirY,camDirZ\n";
    for (const auto& r : run.rows) {
        out << r.frameIdx << ','
            << r.simTimeMs << ','
            << r.cpuMs << ','
            << (r.gpuMsBound ? r.gpuMs : -1.f) << ',';
        for (size_t s = 0; s < static_cast<size_t>(frame::FrameStage::COUNT); ++s) {
            out << (r.gpuMsBound ? r.gpuStageMs[s] : -1.f) << ',';
        }
        out << r.camPos.x << ',' << r.camPos.y << ',' << r.camPos.z << ','
            << r.camDir.x << ',' << r.camDir.y << ',' << r.camDir.z << '\n';
    }
}

void BenchmarkRunner::_WriteRunSummary()
{
    if (!m_Session) return;
    const auto& sess = *m_Session;

    Json::Value root;
    root["timestamp"]      = sess.timestampStr;
    root["pathFile"]       = m_PathFile.string();
    root["pathName"]       = m_Path.GetName();
    root["sceneFile"]      = m_UI.requestedScenePath.empty()
                                ? std::string("(default)")
                                : m_UI.requestedScenePath;
    root["warmupFrames"]   = m_Path.GetWarmupFrames();
    root["simulationDtMs"] = m_Path.GetSimulationDtMs();

    uint32_t totalRecorded = 0;
    for (const auto& r : sess.pipelines) totalRecorded += static_cast<uint32_t>(r.rows.size());
    root["totalRecordedFrames"] = totalRecorded;

    root["vsyncOverride"] = sess.vsyncWasOverridden ? "forced_off" : "none";

#ifdef NDEBUG
    root["buildConfig"] = "Release";
#else
    root["buildConfig"] = "Debug";
#endif

    if (auto* fb = m_DeviceManager->GetCurrentFramebuffer()) {
        const auto& fbInfo = fb->getFramebufferInfo();
        Json::Value res(Json::arrayValue);
        res.append(static_cast<int>(fbInfo.width));
        res.append(static_cast<int>(fbInfo.height));
        root["resolution"] = res;
    }

    if (const char* renderer = m_DeviceManager->GetRendererString()) {
        root["gpuName"] = renderer;
    }

    Json::Value pipes(Json::arrayValue);
    for (const auto& r : sess.pipelines) {
        Json::Value entry;
        entry["name"] =
            r.pipeline == Pipeline::Traditional ? "Traditional" :
            r.pipeline == Pipeline::Compute     ? "Compute"     :
                                                  "MeshShader";
        entry["frames"]    = static_cast<uint32_t>(r.rows.size());
        entry["completed"] = r.completed;
        if (r.completed) {
            entry["csv"] = PipelineFileName(r.pipeline);
        } else if (!r.rows.empty()) {
            entry["csv"] = std::string(PipelineFileName(r.pipeline)) + ".partial";
        } else {
            entry["csv"] = Json::Value::nullSingleton();
        }
        pipes.append(entry);
    }
    root["pipelines"] = pipes;

    const std::filesystem::path summaryPath = sess.outputFolder / "run_summary.json";
    std::ofstream out(summaryPath, std::ios::binary);
    if (!out) {
        m_UI.benchmarkLastStatus        = "Failed to write " + summaryPath.string();
        m_UI.benchmarkLastStatusIsError = true;
        return;
    }

    Json::StreamWriterBuilder wb;
    wb["indentation"] = "  ";
    std::unique_ptr<Json::StreamWriter> writer(wb.newStreamWriter());
    writer->write(root, &out);
    out << '\n';
}
