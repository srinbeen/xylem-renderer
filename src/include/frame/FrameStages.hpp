#ifndef XYLEM_FRAME_STAGES_H
#define XYLEM_FRAME_STAGES_H

#include <array>
#include <cstdint>

namespace Xylem::frame {

enum class FrameStage : uint8_t {
    DepthPrepass = 0,
    HiZ,
    SDSM,
    Cull,
    Shadow,
    Sky,
    Scene,
    COUNT
};

using FrameStageOrder = std::array<FrameStage, static_cast<size_t>(FrameStage::COUNT)>;

inline constexpr FrameStageOrder kDefaultFrameStageOrder = {
    FrameStage::DepthPrepass,
    FrameStage::HiZ,
    FrameStage::SDSM,
    FrameStage::Cull,
    FrameStage::Shadow,
    FrameStage::Sky,
    FrameStage::Scene
};

inline const char* ToString(FrameStage stage)
{
    switch (stage) {
        case FrameStage::DepthPrepass: return "DepthPrepass";
        case FrameStage::HiZ:          return "HiZ";
        case FrameStage::SDSM:         return "SDSM";
        case FrameStage::Cull:         return "Cull";
        case FrameStage::Shadow:       return "Shadow";
        case FrameStage::Sky:          return "Sky";
        case FrameStage::Scene:        return "Scene";
        default:                       return "Unknown";
    }
}

class IFrameStagedPass {
public:
    virtual ~IFrameStagedPass() = default;
    virtual const FrameStageOrder& GetFrameStageOrder() const = 0;

    // Rebuild every GPU resource that depends on the current SceneRegistry
    // contents (per-asset/LOD vertex buffers, instance buffer, indirect args,
    // visibility / slot buffers, binding sets that reference these).
    //
    // Caller MUST have called device->waitForIdle() first.
    //
    // Called from Init() (after the scene-independent _Init* helpers), from
    // RenderOrchestrator::_loadSceneIfRequested (after registry swap), and
    // from RenderOrchestrator::_switchPipelineIfNeeded (the incoming pass's
    // GPU state may be stale from edits while it was inactive).
    virtual bool LoadResources() = 0;
};

} // namespace Xylem::frame

#endif // XYLEM_FRAME_STAGES_H
