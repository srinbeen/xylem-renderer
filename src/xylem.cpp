#include <donut/app/ApplicationBase.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/json.h>
#include <donut/app/Timer.h>
#include <nvrhi/utils.h>

#include <donut/app/Camera.h>
#include <donut/engine/View.h>

#include <donut/app/imgui_renderer.h>

#include "include/procgen.hpp"
#include "include/pipeliner_macros.h"

#include <json/json.h>
#include <set>
#include <numeric>
#include <algorithm>
#include <fstream>

using namespace donut;
using namespace ProcGen;

static const char* g_WindowTitle = "Xylem";

// Deterministic random number generator using hash (replicable on CPU and GPU)
uint32_t hash(uint32_t x, uint32_t seed) {
    x ^= seed;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

float hashToFloat(uint32_t x, uint32_t seed) {
    return (float)hash(x, seed) / (float)UINT32_MAX;
}

// ---------------------------------------------------------------------------
// UIData: shared state between the render pass and the UI renderer.
// Extend this struct when adding new toggles, metric readouts, or UI flags.
// ---------------------------------------------------------------------------
struct UIData {
    bool ShowUI = true;

    // Performance metrics (updated each frame by TraditionalRenderPass)
    float gpuFrameTimeMs    = -1.0f;   // -1 = not yet available
    float cpuRenderTimeMs   =  0.0f;
    uint32_t visibleInstanceCount = 0;
    uint32_t drawCallCount        = 0;
    uint32_t totalInstanceCount   = 0;
    uint32_t culledInstanceCount  = 0;

    // Scene I/O feedback
    bool     sceneSaveSuccess = false;
    bool     sceneLoadSuccess = false;
    float    sceneFeedbackTimer = 0.0f;
};


// ===========================================================================
// TraditionalRenderPass
//   Implements the classic vertex/pixel shader pipeline with instanced LOD
//   rendering of procedurally generated trees.
// ===========================================================================
class TraditionalRenderPass : public app::IRenderPass {
public:
    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------
    static constexpr uint32_t   c_LODs[]            = { 16,    8,     4    };
    static constexpr float      c_LODDistances[]    = { 16.0f, 64.0f, 256.0f };
    static constexpr uint32_t   c_NumLODs           = sizeof(c_LODs) / sizeof(uint32_t);
    static constexpr uint32_t   m_GlobalSeed        = 0xDEADBEEF;
    static constexpr uint32_t   k_QueuedFrames      = 4;   // GPU timer ring-buffer depth

    // -----------------------------------------------------------------------
    // Nested types
    // -----------------------------------------------------------------------

    // LSystemInfo owns an LSystem instance plus the plain-text axiom/rules
    // needed for GUI display and scene serialization.
    struct LSystemInfo {
        std::string name;
        std::string axiomStr;
        std::unordered_map<char, std::string> rulesStr;
        std::unique_ptr<LSystem> system;

        LSystemInfo(const std::string& n,
                    const std::string& axiom,
                    const std::unordered_map<char, std::string>& rules)
            : name(n), axiomStr(axiom), rulesStr(rules),
              system(std::make_unique<LSystem>(axiom, rules))
        {}

        // Move-only (unique_ptr)
        LSystemInfo(LSystemInfo&&) = default;
        LSystemInfo& operator=(LSystemInfo&&) = default;
        LSystemInfo(const LSystemInfo&) = delete;
        LSystemInfo& operator=(const LSystemInfo&) = delete;
    };

    struct InstanceBufferEntry {
        dm::float4x4 model;
        dm::float3x3 normal;
        uint32_t     treeId;

        InstanceBufferEntry(const dm::float4x4& m, const dm::float3x3& n, uint32_t t)
            : model{m}, normal{n}, treeId{t} {}
        InstanceBufferEntry(const dm::affine3& m, const dm::float3x3& n, uint32_t t)
            : InstanceBufferEntry(dm::affineToHomogeneous(m), n, t) {}
    };

    struct ConstantBufferEntry {
        dm::float4x4 view;
        dm::float4x4 projection;
        dm::float4x4 padding[2];
    };
    static constexpr size_t c_ConstantBufferSize =
        sizeof(ConstantBufferEntry) +
        (sizeof(ConstantBufferEntry) % nvrhi::c_ConstantBufferOffsetSizeAlignment);

    struct ViewHandler {
        app::FirstPersonCamera camera;
        engine::PlanarView     view;

        uint32_t distToLOD(float value, const float* arr, int size) {
            const float* it = std::lower_bound(arr, arr + size, value);
            if (it == arr + size) return size - 1;
            return static_cast<uint32_t>(it - arr);
        }
    };

    struct TreeGenerationConfig {
        size_t              lsystemIndex;
        lgen_t              generation;
        TreeGenerator::Params params;
    };

    struct TreeLODData {
        nvrhi::BufferHandle vertexBuffer;
        nvrhi::BufferHandle indexBuffer;
        uint32_t            indexCount;
        uint32_t            radialSegments;
        dm::box3            bbox;
    };

    struct TreeAsset {
        std::string                         name;
        TreeGenerationConfig                config;
        lstring_t                           lsystemString;
        std::array<TreeLODData, c_NumLODs>  lods;
    };

    struct TreeRegion {
        static inline uint32_t s_TotalInstanceCount = 0;

        std::string                      name;
        uint32_t                         instanceCount;
        dm::box2                         bounds;
        std::vector<size_t>              assetIndices;
        std::vector<InstanceBufferEntry> instanceBuffer;
        std::vector<dm::box3>            instanceBbox;
        dm::box3                         cullBox;

        TreeRegion(const std::string& n, uint32_t c, const dm::box2& b)
            : name(n), instanceCount{c}, bounds{b}, cullBox{dm::box3::empty()}
        {
            instanceBuffer.reserve(instanceCount);
            instanceBbox.reserve(instanceCount);
            s_TotalInstanceCount += instanceCount;
        }
    };

    struct DrawCmd {
        nvrhi::BufferHandle  vertexBuffer;
        nvrhi::BufferHandle  indexBuffer;
        nvrhi::DrawArguments drawArgs;
    };

    struct InstanceReference {
        uint32_t regionIdx;
        uint32_t instanceIdx;
        uint32_t treeId;
        uint32_t lodID;
    };

    struct GPUResources {
        nvrhi::ShaderHandle           vertexShader;
        nvrhi::ShaderHandle           pixelShader;
        nvrhi::TextureHandle          texture;
        nvrhi::SamplerHandle          sampler;
        nvrhi::InputLayoutHandle      inputLayout;
        nvrhi::BufferHandle           constantBuffer;
        nvrhi::BufferHandle           instanceBuffer;
        nvrhi::BindingLayoutHandle    bindingLayout;
        nvrhi::BindingSetHandle       bindingSet;
        nvrhi::GraphicsPipelineHandle pipeline;
    };

public:
    TraditionalRenderPass(app::DeviceManager* dm, UIData& ui)
        : IRenderPass(dm), m_UI(ui)
    {}

private:
    // -----------------------------------------------------------------------
    // Members
    // -----------------------------------------------------------------------
    GPUResources                                   m_Resources;
    nvrhi::CommandListHandle                       m_CommandList;
    std::unique_ptr<ViewHandler>                   m_ViewHandler;

    // GPU timer ring-buffer (same pattern as the work_graphs example)
    nvrhi::TimerQueryHandle                        m_GpuTimers[k_QueuedFrames];
    int                                            m_NextTimerIdx = 0;

    UIData&                                        m_UI;

    // Scene data — owned on the heap via std::vector
    std::vector<LSystemInfo>                       m_LSystems;
    std::unique_ptr<TreeGenerator>                 m_TreeGenerator;
    std::vector<TreeAsset>                         m_TreeAssets;
    std::vector<TreeRegion>                        m_Regions;

    // Per-frame CPU-side visibility lists (reserved once, reused every frame)
    std::vector<InstanceReference>                 m_VisibleInstanceReferences;
    std::vector<std::array<uint32_t, c_NumLODs>>  m_InstanceCounts;
    std::vector<std::array<uint32_t, c_NumLODs>>  m_InstanceOffsets;
    std::vector<DrawCmd>                           m_DrawCmds;
    std::vector<InstanceBufferEntry>               m_VisibleInstanceBuffer;

    // -----------------------------------------------------------------------
    // Private helpers: GPU buffer creation
    // -----------------------------------------------------------------------

    // Builds vertex/index buffers for all LODs of a single asset.
    // Caller must have opened m_CommandList before calling this.
    void _BuildTreeAssetBuffers(TreeAsset& asset, std::array<ProcGenBuffers, c_NumLODs>& lods) {
        nvrhi::BufferDesc vDesc;
        vDesc.isVertexBuffer = true;
        vDesc.initialState   = nvrhi::ResourceStates::CopyDest;

        nvrhi::BufferDesc iDesc;
        iDesc.isIndexBuffer = true;
        iDesc.initialState  = nvrhi::ResourceStates::CopyDest;

        auto& genParams = asset.config.params;
        for (size_t j = 0; j < c_NumLODs; j++) {
            genParams.radialSegments = c_LODs[j];
            m_TreeGenerator->setParams(genParams);
            m_TreeGenerator->resetRandomGenerator();
            m_TreeGenerator->generateVertexAndIndexBuffers(asset.lsystemString, lods[j]);

            auto& vBuf = asset.lods[j].vertexBuffer;
            auto& iBuf = asset.lods[j].indexBuffer;

            vDesc.debugName = "VB_" + asset.name + "_LOD" + std::to_string(j);
            vDesc.byteSize  = lods[j].vertices.size() * sizeof(TreeVertex);
            vBuf = GetDevice()->createBuffer(vDesc);
            m_CommandList->beginTrackingBufferState(vBuf, nvrhi::ResourceStates::CopyDest);
            m_CommandList->writeBuffer(vBuf, lods[j].vertices.data(), vDesc.byteSize);
            m_CommandList->setPermanentBufferState(vBuf, nvrhi::ResourceStates::VertexBuffer);

            iDesc.debugName = "IB_" + asset.name + "_LOD" + std::to_string(j);
            iDesc.byteSize  = lods[j].indices.size() * sizeof(uint32_t);
            iBuf = GetDevice()->createBuffer(iDesc);
            m_CommandList->beginTrackingBufferState(iBuf, nvrhi::ResourceStates::CopyDest);
            m_CommandList->writeBuffer(iBuf, lods[j].indices.data(), iDesc.byteSize);
            m_CommandList->setPermanentBufferState(iBuf, nvrhi::ResourceStates::IndexBuffer);

            asset.lods[j].indexCount     = static_cast<uint32_t>(lods[j].indices.size());
            asset.lods[j].radialSegments = genParams.radialSegments;
            asset.lods[j].bbox           = lods[j].bbox;
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers: scene initialisation
    // -----------------------------------------------------------------------
    bool _InitTreeData() {
        m_LSystems.emplace_back(
            "Standard", "X",
            std::unordered_map<char, std::string>{
                {'X', "F+[[>X]-<X]-F[-FX]+X"},
                {'F', "FF"}
            }
        );

        m_TreeGenerator = std::make_unique<TreeGenerator>();

        std::vector<TreeGenerationConfig> genConfigs = {
            { 0, 3, { c_LODs[0], 1.f, dm::radians(25.f), 0.9f, 0.95f, 0 } },
            { 0, 4, { c_LODs[0], 1.f, dm::radians(30.f), 0.9f, 0.95f, 0 } },
        };

        m_TreeAssets.resize(genConfigs.size());
        std::array<ProcGenBuffers, c_NumLODs> lods;

        for (size_t i = 0; i < genConfigs.size(); i++) {
            auto& asset     = m_TreeAssets[i];
            auto& cfg       = genConfigs[i];

            asset.name   = "Asset_" + std::to_string(i);
            asset.config = cfg;

            if (cfg.lsystemIndex >= m_LSystems.size()) {
                log::error("Invalid L-System index: %zu", cfg.lsystemIndex);
                return false;
            }

            auto& lsInfo = m_LSystems[cfg.lsystemIndex];
            lsInfo.system->reset();
            lsInfo.system->generate(cfg.generation);
            asset.lsystemString = lsInfo.system->getCurrentString();

            _BuildTreeAssetBuffers(asset, lods);
        }
        return true;
    }

    bool _InitRegions() {
        m_Regions = {
            TreeRegion("Region_0", 10, { dm::float2(-10.f,-20.f), dm::float2(0.f,0.f) }),
            TreeRegion("Region_1", 25, { dm::float2(10.f,0.f),   dm::float2(20.f,20.f) }),
        };

        for (auto& r : m_Regions)
            for (size_t i = 0; i < m_TreeAssets.size(); ++i)
                r.assetIndices.push_back(i);

        for (size_t regionIdx = 0; regionIdx < m_Regions.size(); regionIdx++)
            _PopulateRegionInstances(regionIdx);

        return true;
    }

    bool _InitShaders() {
        std::filesystem::path frameworkShaderPath =
            app::GetDirectoryWithExecutable().parent_path() / "shaders/framework" /
            app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path appShaderPath =
            app::GetDirectoryWithExecutable().parent_path() / "shaders/custom" /
            app::GetShaderTypeName(GetDevice()->getGraphicsAPI());

        auto rootFS = std::make_shared<vfs::RootFileSystem>();
        rootFS->mount("/shaders/donut", frameworkShaderPath);
        rootFS->mount("/shaders/app",   appShaderPath);

        auto sf = std::make_shared<engine::ShaderFactory>(GetDevice(), rootFS, "/shaders");
        m_Resources.vertexShader = sf->CreateShader("app/shaders.hlsl", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
        m_Resources.pixelShader  = sf->CreateShader("app/shaders.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
        return !!m_Resources.vertexShader && !!m_Resources.pixelShader;
    }

    bool _InitVertexAttributes() {
        nvrhi::VertexAttributeDesc attributes[] = {
#if PIPELINER_USE_INTERLEAVED_VERTEX_ATTRIBUTES
            nvrhi::VertexAttributeDesc()
                .setName("POSITION").setFormat(nvrhi::Format::RGB32_FLOAT)
                .setOffset(offsetof(TreeVertex, pos)).setBufferIndex(0).setElementStride(sizeof(TreeVertex)),
            nvrhi::VertexAttributeDesc()
                .setName("NORMAL").setFormat(nvrhi::Format::RGB32_FLOAT)
                .setOffset(offsetof(TreeVertex, normal)).setBufferIndex(0).setElementStride(sizeof(TreeVertex)),
            nvrhi::VertexAttributeDesc()
                .setName("UV").setFormat(nvrhi::Format::RG32_FLOAT)
                .setOffset(offsetof(TreeVertex, uv)).setBufferIndex(0).setElementStride(sizeof(TreeVertex)),
#if !PIPELINER_USE_STRUCTURED_BUFFER
            nvrhi::VertexAttributeDesc()
                .setName("MODEL_MATRIX").setFormat(nvrhi::Format::RGBA32_FLOAT).setArraySize(4)
                .setOffset(offsetof(InstanceBufferEntry, model)).setBufferIndex(1)
                .setElementStride(sizeof(InstanceBufferEntry)).setIsInstanced(true),
            nvrhi::VertexAttributeDesc()
                .setName("NORMAL_MATRIX").setFormat(nvrhi::Format::RGB32_FLOAT).setArraySize(3)
                .setOffset(offsetof(InstanceBufferEntry, normal)).setBufferIndex(1)
                .setElementStride(sizeof(InstanceBufferEntry)).setIsInstanced(true),
#endif
#else
            nvrhi::VertexAttributeDesc()
                .setName("POSITION").setFormat(nvrhi::Format::RGB32_FLOAT)
                .setOffset(0).setBufferIndex(0).setElementStride(sizeof(TreeVertex)),
            nvrhi::VertexAttributeDesc()
                .setName("NORMAL").setFormat(nvrhi::Format::RGB32_FLOAT)
                .setOffset(0).setBufferIndex(1).setElementStride(sizeof(TreeVertex)),
            nvrhi::VertexAttributeDesc()
                .setName("UV").setFormat(nvrhi::Format::RG32_FLOAT)
                .setOffset(0).setBufferIndex(2).setElementStride(sizeof(TreeVertex)),
#if !PIPELINER_USE_STRUCTURED_BUFFER
            nvrhi::VertexAttributeDesc()
                .setName("MODEL_MATRIX").setFormat(nvrhi::Format::RGBA32_FLOAT).setArraySize(4)
                .setOffset(0).setBufferIndex(3).setElementStride(sizeof(InstanceBufferEntry)).setIsInstanced(true),
            nvrhi::VertexAttributeDesc()
                .setName("NORMAL_MATRIX").setFormat(nvrhi::Format::RGB32_FLOAT).setArraySize(3)
                .setOffset(0).setBufferIndex(4).setElementStride(sizeof(InstanceBufferEntry)).setIsInstanced(true),
#endif
#endif
        };
        m_Resources.inputLayout = GetDevice()->createInputLayout(
            attributes, uint32_t(std::size(attributes)), m_Resources.vertexShader);
        return !!m_Resources.inputLayout;
    }

    bool _InitBuffers() {
        m_Resources.instanceBuffer = GetDevice()->createBuffer(
            nvrhi::BufferDesc()
                .setByteSize(std::max<size_t>(1, TreeRegion::s_TotalInstanceCount) * sizeof(InstanceBufferEntry))
                .setStructStride(sizeof(InstanceBufferEntry))
                .setDebugName("InstanceBuffer")
#if PIPELINER_USE_STRUCTURED_BUFFER
                .enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource)
#else
                .setIsVertexBuffer(true)
                .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
#endif
        );

        m_Resources.constantBuffer = GetDevice()->createBuffer(
            nvrhi::BufferDesc()
                .setByteSize(c_ConstantBufferSize)
                .setIsConstantBuffer(true)
                .setDebugName("ConstantBuffer")
                .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
        );

        return !!m_Resources.instanceBuffer && !!m_Resources.constantBuffer;
    }

    bool _InitTextureAndSampler() {
        engine::TextureCache textureCache(GetDevice(), std::make_shared<vfs::NativeFileSystem>(), nullptr);
        std::filesystem::path tex =
            app::GetDirectoryWithExecutable().parent_path().parent_path() / "media/bark_willow_02_diff_1k.jpg";
        auto loaded = textureCache.LoadTextureFromFile(tex, true, nullptr, m_CommandList);
        m_Resources.texture = loaded->texture;
        m_Resources.sampler = GetDevice()->createSampler(
            nvrhi::SamplerDesc().setAllAddressModes(nvrhi::SamplerAddressMode::Wrap));
        return !!m_Resources.texture && !!m_Resources.sampler;
    }

    bool _InitBindingLayoutAndSet() {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_Resources.constantBuffer, nvrhi::BufferRange(0, c_ConstantBufferSize)),
            nvrhi::BindingSetItem::Sampler(0, m_Resources.sampler),
            nvrhi::BindingSetItem::Texture_SRV(0, m_Resources.texture),
#if PIPELINER_USE_STRUCTURED_BUFFER
            nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_Resources.instanceBuffer,
                nvrhi::Format::UNKNOWN,
                nvrhi::BufferRange(0, TreeRegion::s_TotalInstanceCount * sizeof(InstanceBufferEntry))),
            nvrhi::BindingSetItem::PushConstants(1, sizeof(uint32_t)),
#endif
        };
        if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
                bsd, m_Resources.bindingLayout, m_Resources.bindingSet)) {
            log::error("Couldn't create the binding set or layout");
            return false;
        }
        return !!m_Resources.bindingLayout && !!m_Resources.bindingSet;
    }

    bool _InitViewHandler() {
        m_ViewHandler = std::make_unique<ViewHandler>();
        m_ViewHandler->camera.LookTo(dm::float3(5, 5, -5), dm::float3(0, 0, 1));
        m_ViewHandler->camera.SetMoveSpeed(15.f);
        return !!m_ViewHandler;
    }

    bool _InitTimerQueries() {
        for (uint32_t i = 0; i < k_QueuedFrames; i++)
            m_GpuTimers[i] = GetDevice()->createTimerQuery();
        return true;
    }

    // -----------------------------------------------------------------------
    // Private helpers: instance placement
    // -----------------------------------------------------------------------
    void _PopulateRegionInstances(size_t regionIdx) {
        if (regionIdx >= m_Regions.size()) return;
        TreeRegion& r = m_Regions[regionIdx];
        r.instanceBuffer.clear();
        r.instanceBbox.clear();
        r.cullBox = dm::box3::empty();

        dm::float2 min   = r.bounds.m_mins;
        dm::float2 range = r.bounds.diagonal();

        for (uint32_t i = 0; i < r.instanceCount; i++) {
            float posX = min.x + hashToFloat(i, m_GlobalSeed ^ 0x12345678) * range.x;
            float posZ = min.y + hashToFloat(i, m_GlobalSeed ^ 0x87654321) * range.y;

            dm::affine3 world =
                dm::rotation(normalize(dm::float3(1, 0, 0)), -dm::PI_f / 2.f) *
                dm::translation(dm::float3(posX, 0.f, posZ));

            const auto& m = world.m_linear;
            float sX = dm::lengthSquared(m.row0), sY = dm::lengthSquared(m.row1), sZ = dm::lengthSquared(m.row2);
            dm::float3x3 normalMatrix = (dm::isnear(sX, sY) && dm::isnear(sY, sZ))
                ? m : dm::transpose(dm::inverse(m));

            size_t assetIdx = 0;
            if (!r.assetIndices.empty()) {
                uint32_t choice = hash(i, m_GlobalSeed ^ 0xFEEDBEEF) % (uint32_t)r.assetIndices.size();
                assetIdx = r.assetIndices[choice];
            }

            r.instanceBuffer.emplace_back(world, normalMatrix, (uint32_t)assetIdx);
            dm::box3 bbox = m_TreeAssets[assetIdx].lods[0].bbox * world;
            r.instanceBbox.push_back(bbox);
            r.cullBox |= bbox;
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers: regeneration
    // -----------------------------------------------------------------------
    void _RegenerateTreeAssets() {
        for (auto& asset : m_TreeAssets)
            for (auto& lod : asset.lods) { lod.vertexBuffer = nullptr; lod.indexBuffer = nullptr; }
        m_TreeAssets.clear();

        m_CommandList->open();
        _InitTreeData();
        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
        m_CommandList = GetDevice()->createCommandList();
    }

    void _RebuildInstanceBuffer() {
        m_Resources.instanceBuffer = GetDevice()->createBuffer(
            nvrhi::BufferDesc()
                .setByteSize(std::max<size_t>(1, TreeRegion::s_TotalInstanceCount) * sizeof(InstanceBufferEntry))
                .setStructStride(sizeof(InstanceBufferEntry))
                .setDebugName("InstanceBuffer")
#if PIPELINER_USE_STRUCTURED_BUFFER
                .enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource)
#else
                .setIsVertexBuffer(true)
                .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
#endif
        );

        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_Resources.constantBuffer, nvrhi::BufferRange(0, c_ConstantBufferSize)),
            nvrhi::BindingSetItem::Sampler(0, m_Resources.sampler),
            nvrhi::BindingSetItem::Texture_SRV(0, m_Resources.texture),
#if PIPELINER_USE_STRUCTURED_BUFFER
            nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_Resources.instanceBuffer,
                nvrhi::Format::UNKNOWN,
                nvrhi::BufferRange(0, TreeRegion::s_TotalInstanceCount * sizeof(InstanceBufferEntry))),
            nvrhi::BindingSetItem::PushConstants(1, sizeof(uint32_t)),
#endif
        };
        nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
            bsd, m_Resources.bindingLayout, m_Resources.bindingSet);

        m_InstanceCounts.assign(m_TreeAssets.size(), {0});
        m_InstanceOffsets.assign(m_TreeAssets.size(), {0});
        m_VisibleInstanceBuffer.resize(TreeRegion::s_TotalInstanceCount);
    }

    void _RegenerateRegions() {
        TreeRegion::s_TotalInstanceCount = 0;
        m_Regions.clear();
        _InitRegions();
        _RebuildInstanceBuffer();
    }

// ---------------------------------------------------------------------------
public:
// ---------------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Read-only accessors (for UI renderer)
    // -----------------------------------------------------------------------
    const std::vector<LSystemInfo>&  GetLSystems()   const { return m_LSystems; }
    const std::vector<TreeAsset>&    GetTreeAssets() const { return m_TreeAssets; }
    const std::vector<TreeRegion>&   GetRegions()    const { return m_Regions; }
    std::vector<TreeAsset>&          GetTreeAssets()       { return m_TreeAssets; }
    std::vector<TreeRegion>&         GetRegions()          { return m_Regions; }

    // -----------------------------------------------------------------------
    // Scene mutators
    // -----------------------------------------------------------------------
    void AddLSystem(const std::string& name,
                    const std::string& axiom,
                    const std::unordered_map<char, std::string>& rules)
    {
        m_LSystems.emplace_back(name, axiom, rules);
    }

    void AddTreeAsset(const std::string& name,
                      size_t lsystemIdx,
                      uint32_t generation,
                      const TreeGenerator::Params& params)
    {
        if (lsystemIdx >= m_LSystems.size()) return;

        TreeAsset asset;
        asset.name   = name;
        asset.config = { lsystemIdx, generation, params };

        auto& lsInfo = m_LSystems[lsystemIdx];
        lsInfo.system->reset();
        lsInfo.system->generate(generation);
        asset.lsystemString = lsInfo.system->getCurrentString();

        m_CommandList->open();
        std::array<ProcGenBuffers, c_NumLODs> lods;
        _BuildTreeAssetBuffers(asset, lods);
        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
        m_CommandList = GetDevice()->createCommandList();

        m_TreeAssets.push_back(std::move(asset));
        m_InstanceCounts.resize(m_TreeAssets.size(), {0});
        m_InstanceOffsets.resize(m_TreeAssets.size(), {0});
        m_VisibleInstanceBuffer.resize(TreeRegion::s_TotalInstanceCount);
    }

    void UpdateTreeAsset(size_t idx,
                         const std::string& name,
                         size_t lsystemIdx,
                         uint32_t generation,
                         const TreeGenerator::Params& params)
    {
        if (idx >= m_TreeAssets.size() || lsystemIdx >= m_LSystems.size()) return;

        auto& asset          = m_TreeAssets[idx];
        asset.name           = name;
        asset.config.lsystemIndex = lsystemIdx;
        asset.config.generation   = generation;
        asset.config.params       = params;

        auto& lsInfo = m_LSystems[lsystemIdx];
        lsInfo.system->reset();
        lsInfo.system->generate(generation);
        asset.lsystemString = lsInfo.system->getCurrentString();

        for (auto& lod : asset.lods) { lod.vertexBuffer = nullptr; lod.indexBuffer = nullptr; }

        m_CommandList->open();
        std::array<ProcGenBuffers, c_NumLODs> lods;
        _BuildTreeAssetBuffers(asset, lods);
        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
        m_CommandList = GetDevice()->createCommandList();
    }

    void RemoveTreeAsset(size_t idx) {
        if (idx >= m_TreeAssets.size()) return;
        for (auto& region : m_Regions) {
            region.assetIndices.erase(
                std::remove(region.assetIndices.begin(), region.assetIndices.end(), idx),
                region.assetIndices.end());
            for (auto& i : region.assetIndices)
                if (i > idx) i--;
        }
        m_TreeAssets.erase(m_TreeAssets.begin() + idx);
        m_InstanceCounts.resize(m_TreeAssets.size(), {0});
        m_InstanceOffsets.resize(m_TreeAssets.size(), {0});
    }

    void AddRegion(const std::string& name, uint32_t instanceCount, const dm::box2& bounds) {
        TreeRegion::s_TotalInstanceCount += instanceCount;
        m_Regions.emplace_back(name, instanceCount, bounds);
        auto& r = m_Regions.back();
        for (size_t i = 0; i < m_TreeAssets.size(); ++i) r.assetIndices.push_back(i);
        _PopulateRegionInstances(m_Regions.size() - 1);
        _RebuildInstanceBuffer();
    }

    void RemoveRegion(size_t idx) {
        if (idx >= m_Regions.size()) return;
        TreeRegion::s_TotalInstanceCount -= m_Regions[idx].instanceCount;
        m_Regions.erase(m_Regions.begin() + idx);
        _RebuildInstanceBuffer();
    }

    void RegenerateAssets()  { _RegenerateTreeAssets(); }
    void RegenerateRegions() { _RegenerateRegions(); }

    // -----------------------------------------------------------------------
    // Scene serialization (uses donut's bundled jsoncpp)
    // -----------------------------------------------------------------------
    bool SaveScene(const std::filesystem::path& path) const {
        Json::Value root;

        // L-Systems
        for (const auto& ls : m_LSystems) {
            Json::Value lsNode;
            lsNode["name"]  = ls.name;
            lsNode["axiom"] = ls.axiomStr;
            for (const auto& [c, s] : ls.rulesStr)
                lsNode["rules"][std::string(1, c)] = s;
            root["lsystems"].append(lsNode);
        }

        // Tree assets
        for (const auto& asset : m_TreeAssets) {
            Json::Value aNode;
            aNode["name"] = asset.name;
            aNode["lsystem"] = (asset.config.lsystemIndex < m_LSystems.size())
                ? m_LSystems[asset.config.lsystemIndex].name : "";
            aNode["generation"]     = (Json::UInt)asset.config.generation;
            aNode["radialSegments"] = (Json::UInt)asset.config.params.radialSegments;
            aNode["stepLength"]     = asset.config.params.stepLength;
            aNode["branchAngle"]    = asset.config.params.branchAngle;
            aNode["taperRatio"]     = asset.config.params.taperRatio;
            aNode["stepRatio"]      = asset.config.params.stepRatio;
            aNode["seed"]           = (Json::UInt)asset.config.params.seed;
            root["assets"].append(aNode);
        }

        // Regions
        for (const auto& region : m_Regions) {
            Json::Value rNode;
            rNode["name"]          = region.name;
            rNode["instanceCount"] = (Json::UInt)region.instanceCount;
            rNode["boundsMinX"]    = region.bounds.m_mins.x;
            rNode["boundsMinY"]    = region.bounds.m_mins.y;
            rNode["boundsMaxX"]    = region.bounds.m_maxs.x;
            rNode["boundsMaxY"]    = region.bounds.m_maxs.y;
            for (size_t idx : region.assetIndices)
                if (idx < m_TreeAssets.size())
                    rNode["assets"].append(m_TreeAssets[idx].name);
            root["regions"].append(rNode);
        }

        std::ofstream file(path);
        if (!file.is_open()) { log::error("SaveScene: cannot open %s", path.string().c_str()); return false; }
        Json::StreamWriterBuilder builder;
        builder["indentation"] = "  ";
        file << Json::writeString(builder, root);
        log::info("Scene saved to %s", path.string().c_str());
        return true;
    }

    bool LoadScene(const std::filesystem::path& path) {
        std::ifstream file(path);
        if (!file.is_open()) { log::error("LoadScene: cannot open %s", path.string().c_str()); return false; }

        Json::CharReaderBuilder reader;
        Json::Value root;
        std::string errors;
        if (!Json::parseFromStream(reader, file, &root, &errors)) {
            log::error("LoadScene parse error: %s", errors.c_str());
            return false;
        }

        // --- Clear existing scene ---
        for (auto& asset : m_TreeAssets)
            for (auto& lod : asset.lods) { lod.vertexBuffer = nullptr; lod.indexBuffer = nullptr; }
        m_TreeAssets.clear();
        m_LSystems.clear();
        TreeRegion::s_TotalInstanceCount = 0;
        m_Regions.clear();

        // --- L-Systems ---
        for (const auto& lsNode : root["lsystems"]) {
            std::string name  = lsNode["name"].asString();
            std::string axiom = lsNode["axiom"].asString();
            std::unordered_map<char, std::string> rules;
            for (const auto& key : lsNode["rules"].getMemberNames())
                if (!key.empty()) rules[key[0]] = lsNode["rules"][key].asString();
            m_LSystems.emplace_back(name, axiom, rules);
        }
        if (m_LSystems.empty()) { log::error("LoadScene: no L-systems in file"); return false; }

        // Build name->index map for resolving asset references
        std::unordered_map<std::string, size_t> lsysNameToIdx;
        for (size_t i = 0; i < m_LSystems.size(); i++) lsysNameToIdx[m_LSystems[i].name] = i;

        // --- Tree assets ---
        if (!m_TreeGenerator) m_TreeGenerator = std::make_unique<TreeGenerator>();

        m_CommandList->open();
        for (const auto& aNode : root["assets"]) {
            TreeAsset asset;
            asset.name = aNode["name"].asString();
            std::string lsName = aNode.get("lsystem", "").asString();
            size_t lsIdx = lsysNameToIdx.count(lsName) ? lsysNameToIdx[lsName] : 0;

            TreeGenerator::Params p;
            p.generation      = aNode.get("generation", 3).asUInt();  // stored for convenience
            p.radialSegments  = aNode.get("radialSegments", 16).asUInt();
            p.stepLength      = aNode.get("stepLength", 1.f).asFloat();
            p.branchAngle     = aNode.get("branchAngle", dm::radians(25.f)).asFloat();
            p.taperRatio      = aNode.get("taperRatio", 0.9f).asFloat();
            p.stepRatio       = aNode.get("stepRatio", 0.95f).asFloat();
            p.seed            = aNode.get("seed", 0).asUInt();

            uint32_t generation = aNode.get("generation", 3).asUInt();
            asset.config = { lsIdx, generation, p };

            auto& lsInfo = m_LSystems[lsIdx];
            lsInfo.system->reset();
            lsInfo.system->generate(generation);
            asset.lsystemString = lsInfo.system->getCurrentString();

            std::array<ProcGenBuffers, c_NumLODs> lods;
            _BuildTreeAssetBuffers(asset, lods);
            m_TreeAssets.push_back(std::move(asset));
        }
        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
        m_CommandList = GetDevice()->createCommandList();

        // Build name->index map for assets
        std::unordered_map<std::string, size_t> assetNameToIdx;
        for (size_t i = 0; i < m_TreeAssets.size(); i++) assetNameToIdx[m_TreeAssets[i].name] = i;

        // --- Regions ---
        for (const auto& rNode : root["regions"]) {
            uint32_t count = rNode.get("instanceCount", 10).asUInt();
            dm::box2 bounds(
                dm::float2(rNode.get("boundsMinX", 0.f).asFloat(), rNode.get("boundsMinY", 0.f).asFloat()),
                dm::float2(rNode.get("boundsMaxX", 10.f).asFloat(), rNode.get("boundsMaxY", 10.f).asFloat())
            );
            m_Regions.emplace_back(rNode.get("name", "Region").asString(), count, bounds);
            auto& r = m_Regions.back();
            for (const auto& aName : rNode["assets"]) {
                std::string n = aName.asString();
                if (assetNameToIdx.count(n)) r.assetIndices.push_back(assetNameToIdx[n]);
            }
            if (r.assetIndices.empty())
                for (size_t i = 0; i < m_TreeAssets.size(); i++) r.assetIndices.push_back(i);
            _PopulateRegionInstances(m_Regions.size() - 1);
        }

        _RebuildInstanceBuffer();

        m_InstanceCounts.assign(m_TreeAssets.size(), {0});
        m_InstanceOffsets.assign(m_TreeAssets.size(), {0});
        m_VisibleInstanceReferences.reserve(TreeRegion::s_TotalInstanceCount);
        m_DrawCmds.reserve(m_TreeAssets.size() * c_NumLODs);

        log::info("Scene loaded from %s", path.string().c_str());
        return true;
    }

    // -----------------------------------------------------------------------
    // IRenderPass interface
    // -----------------------------------------------------------------------
    bool Init() {
        m_CommandList = GetDevice()->createCommandList();
        m_CommandList->open();

        if (!_InitTreeData())            return false;
        if (!_InitRegions())             return false;
        if (!_InitShaders())             return false;
        if (!_InitVertexAttributes())    return false;
        if (!_InitBuffers())             return false;
        if (!_InitTextureAndSampler())   return false;
        if (!_InitBindingLayoutAndSet()) return false;
        if (!_InitViewHandler())         return false;

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);

        _InitTimerQueries();

        m_UI.totalInstanceCount = TreeRegion::s_TotalInstanceCount;
        m_VisibleInstanceReferences.reserve(TreeRegion::s_TotalInstanceCount);
        m_InstanceCounts.resize(m_TreeAssets.size(), {0});
        m_InstanceOffsets.assign(m_TreeAssets.size(), {0});
        m_DrawCmds.reserve(m_TreeAssets.size() * c_NumLODs);
        m_VisibleInstanceBuffer.resize(TreeRegion::s_TotalInstanceCount);

        return true;
    }

    void Animate(float seconds) override {
        m_ViewHandler->camera.Animate(seconds);
        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }

    void BackBufferResizing() override { m_Resources.pipeline = nullptr; }

    void Render(nvrhi::IFramebuffer* framebuffer) override {
        // CPU timer
        app::HiResTimer cpuTimer;
        cpuTimer.Start();

        const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();

        if (!m_Resources.pipeline) {
            nvrhi::GraphicsPipelineDesc psoDesc;
            psoDesc.VS           = m_Resources.vertexShader;
            psoDesc.PS           = m_Resources.pixelShader;
            psoDesc.inputLayout  = m_Resources.inputLayout;
            psoDesc.bindingLayouts = { m_Resources.bindingLayout };
            psoDesc.primType     = nvrhi::PrimitiveType::TriangleList;
#if PIPELINER_USE_REVERSE_Z
            psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
#else
            psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
#endif
            m_Resources.pipeline = GetDevice()->createGraphicsPipeline(psoDesc, fbinfo);

            m_ViewHandler->view.SetViewport({ float(fbinfo.width), float(fbinfo.height) });
#if PIPELINER_USE_REVERSE_Z
            m_ViewHandler->view.SetProjectionMatrix(
                dm::perspProjD3DStyleReverse(dm::radians(60.f), float(fbinfo.width)/float(fbinfo.height), 0.1f));
#else
            m_ViewHandler->view.SetProjectionMatrix(
                dm::perspProjD3DStyle(dm::radians(60.f), float(fbinfo.width)/float(fbinfo.height), 0.1f,
                    std::numeric_limits<float>::max()));
#endif
        }

        m_CommandList->open();

        // --- GPU timer: begin ---
        m_CommandList->beginTimerQuery(m_GpuTimers[m_NextTimerIdx]);

        m_ViewHandler->view.SetViewMatrix(m_ViewHandler->camera.GetWorldToViewMatrix());
        m_ViewHandler->view.UpdateCache();

        nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));
#if PIPELINER_USE_REVERSE_Z
        nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer, 0.f, 0);
#else
        nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer,
            m_ViewHandler->view.GetViewFrustum().farPlane().distance, 0);
#endif

        ConstantBufferEntry constants{};
        constants.view       = dm::affineToHomogeneous(m_ViewHandler->view.GetViewMatrix());
        constants.projection = m_ViewHandler->view.GetProjectionMatrix();
        m_CommandList->writeBuffer(m_Resources.constantBuffer, &constants, c_ConstantBufferSize);

        // --- CPU-side frustum culling and LOD selection ---
        m_VisibleInstanceReferences.clear();
        m_InstanceCounts.assign(m_TreeAssets.size(), {0});
        uint32_t culled = 0;

        for (uint32_t ri = 0; ri < m_Regions.size(); ri++) {
            const auto& region = m_Regions[ri];
            if (!m_ViewHandler->view.IsBoxVisible(region.cullBox)) {
                culled += region.instanceCount;
                continue;
            }
            for (uint32_t ii = 0; ii < region.instanceCount; ii++) {
                if (!m_ViewHandler->view.IsBoxVisible(region.instanceBbox[ii])) { culled++; continue; }
                float dist = dm::distance(m_ViewHandler->camera.GetPosition(), region.instanceBbox[ii]);
                uint32_t lod = m_ViewHandler->distToLOD(dist, c_LODDistances, c_NumLODs);
                m_VisibleInstanceReferences.push_back({ ri, ii, region.instanceBuffer[ii].treeId, lod });
                m_InstanceCounts[region.instanceBuffer[ii].treeId][lod]++;
            }
        }

        // --- Build draw commands ---
        m_DrawCmds.clear();
        m_InstanceOffsets.assign(m_TreeAssets.size(), {0});
        uint32_t instanceOffset = 0;
        for (uint32_t ai = 0; ai < m_InstanceCounts.size(); ai++) {
            for (uint32_t li = 0; li < c_NumLODs; li++) {
                uint32_t count = m_InstanceCounts[ai][li];
                if (count == 0) continue;
                const auto& lod = m_TreeAssets[ai].lods[li];
                m_DrawCmds.push_back({ lod.vertexBuffer, lod.indexBuffer,
                    nvrhi::DrawArguments()
                        .setVertexCount(lod.indexCount)
                        .setInstanceCount(count)
                        .setStartInstanceLocation(instanceOffset) });
                m_InstanceOffsets[ai][li] = instanceOffset;
                instanceOffset += count;
            }
        }

        // --- Fill visible instance buffer ---
        for (const auto& ref : m_VisibleInstanceReferences) {
            uint32_t& writeOff = m_InstanceOffsets[ref.treeId][ref.lodID];
            m_VisibleInstanceBuffer[writeOff] = m_Regions[ref.regionIdx].instanceBuffer[ref.instanceIdx];
            writeOff++;
        }

        if (!m_VisibleInstanceReferences.empty())
            m_CommandList->writeBuffer(m_Resources.instanceBuffer, m_VisibleInstanceBuffer.data(),
                m_VisibleInstanceReferences.size() * sizeof(InstanceBufferEntry));

        // --- Draw ---
        nvrhi::GraphicsState state;
        state.pipeline   = m_Resources.pipeline;
        state.framebuffer = framebuffer;
        state.viewport   = m_ViewHandler->view.GetViewportState();
        state.bindings   = { m_Resources.bindingSet };

        for (const auto& cmd : m_DrawCmds) {
            state.vertexBuffers = {
#if PIPELINER_USE_INTERLEAVED_VERTEX_ATTRIBUTES
                { cmd.vertexBuffer, 0, 0 },
#if !PIPELINER_USE_STRUCTURED_BUFFER
                { m_Resources.instanceBuffer, 1, 0 },
#endif
#else
                { cmd.vertexBuffer, 0, offsetof(TreeVertex, pos) },
                { cmd.vertexBuffer, 1, offsetof(TreeVertex, normal) },
                { cmd.vertexBuffer, 2, offsetof(TreeVertex, uv) },
#if !PIPELINER_USE_STRUCTURED_BUFFER
                { m_Resources.instanceBuffer, 3, offsetof(InstanceBufferEntry, model) },
                { m_Resources.instanceBuffer, 4, offsetof(InstanceBufferEntry, normal) },
#endif
#endif
            };
            state.indexBuffer = { cmd.indexBuffer, nvrhi::Format::R32_UINT, 0 };
            m_CommandList->setGraphicsState(state);
#if PIPELINER_USE_STRUCTURED_BUFFER
            m_CommandList->setPushConstants(&instanceOffset, sizeof(uint32_t));
#endif
            m_CommandList->drawIndexed(cmd.drawArgs);
        }

        // --- GPU timer: end ---
        m_CommandList->endTimerQuery(m_GpuTimers[m_NextTimerIdx]);

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);

        // Poll the previous frame's GPU timer (the current one hasn't been processed yet)
        int prevIdx = (m_NextTimerIdx + k_QueuedFrames - 1) % k_QueuedFrames;
        if (GetDevice()->pollTimerQuery(m_GpuTimers[prevIdx]))
            m_UI.gpuFrameTimeMs = GetDevice()->getTimerQueryTime(m_GpuTimers[prevIdx]) * 1000.0f;
        m_NextTimerIdx = (m_NextTimerIdx + 1) % k_QueuedFrames;

        // Update UI stats
        m_UI.visibleInstanceCount = (uint32_t)m_VisibleInstanceReferences.size();
        m_UI.culledInstanceCount  = culled;
        m_UI.drawCallCount        = (uint32_t)m_DrawCmds.size();
        m_UI.totalInstanceCount   = TreeRegion::s_TotalInstanceCount;

        m_VisibleInstanceReferences.clear();
        m_InstanceCounts.assign(m_TreeAssets.size(), {0});

        cpuTimer.Stop();
        m_UI.cpuRenderTimeMs = (float)cpuTimer.Milliseconds();
    }

    // -----------------------------------------------------------------------
    // Input forwarding
    // -----------------------------------------------------------------------
    bool KeyboardUpdate(int key, int scancode, int action, int mods) override {
        m_ViewHandler->camera.KeyboardUpdate(key, scancode, action, mods); return true;
    }
    bool MousePosUpdate(double xpos, double ypos) override {
        m_ViewHandler->camera.MousePosUpdate(xpos, ypos); return true;
    }
    bool MouseScrollUpdate(double xoffset, double yoffset) override {
        m_ViewHandler->camera.MouseScrollUpdate(xoffset, yoffset); return true;
    }
    bool MouseButtonUpdate(int button, int action, int mods) override {
        m_ViewHandler->camera.MouseButtonUpdate(button, action, mods); return true;
    }
    bool JoystickButtonUpdate(int button, bool pressed) override {
        m_ViewHandler->camera.JoystickButtonUpdate(button, pressed); return true;
    }
    bool JoystickAxisUpdate(int axis, float value) override {
        m_ViewHandler->camera.JoystickUpdate(axis, value); return true;
    }
};


// ===========================================================================
// XylemUIRenderer
//   Drives all Dear ImGui panels. It reads/writes UIData and calls
//   mutator methods on the TraditionalRenderPass.
// ===========================================================================
class XylemUIRenderer : public app::ImGui_Renderer {
private:
    TraditionalRenderPass* m_pass;
    UIData&                m_ui;

    // -- Asset edit state --------------------------------------------------
    struct AssetEditState {
        char     name[128]    = "NewAsset";
        uint32_t generation   = 3;
        uint32_t radialSegs   = 16;
        float    branchAngle  = 25.f;
        float    taperRatio   = 0.9f;
        float    stepRatio    = 0.95f;
        int      lsysIdx      = 0;
        bool     editing      = false;
        size_t   editingIndex = 0;
    } m_asset;

    // -- Region edit state -------------------------------------------------
    struct RegionEditState {
        char     name[128]     = "NewRegion";
        uint32_t instanceCount = 20;
        float    minX = 0.f, minY = 0.f;
        float    maxX = 10.f, maxY = 10.f;
        std::set<size_t> selectedAssets;
        bool     editing      = false;
        size_t   editingIndex = 0;
    } m_region;

    // -- L-System add state ------------------------------------------------
    struct LSystemEditState {
        char name[64]  = "NewLSystem";
        char axiom[64] = "X";
        // Up to 8 character rules for the simple editor
        struct Rule { char from[4] = ""; char to[256] = ""; };
        Rule rules[8];
        int  ruleCount = 1;
    } m_lsystem;

    // -- Scene I/O ---------------------------------------------------------
    char m_scenePath[512] = "scene.json";

public:
    XylemUIRenderer(app::DeviceManager* dm, TraditionalRenderPass* pass, UIData& ui)
        : ImGui_Renderer(dm), m_pass(pass), m_ui(ui)
    {
        ImGui::GetIO().IniFilename = nullptr;
    }

    void Init(std::shared_ptr<engine::ShaderFactory> sf) { ImGui_Renderer::Init(sf); }

    bool KeyboardUpdate(int key, int scancode, int action, int mods) override {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            m_ui.ShowUI = !m_ui.ShowUI;
            return true;
        }
        return false;
    }

protected:
    void buildUI() override {
        if (!m_ui.ShowUI) return;

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(620, 900), ImGuiCond_FirstUseEver);
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
            ImGui::Text("Draw calls: %u", m_ui.drawCallCount);
        }

        ImGui::Separator();

        // =====================================================================
        // SCENE SAVE / LOAD
        // =====================================================================
        if (ImGui::CollapsingHeader("Scene Save / Load")) {
            ImGui::InputText("Path##scenepath", m_scenePath, sizeof(m_scenePath));
            if (ImGui::Button("Save Scene")) {
                m_ui.sceneSaveSuccess = m_pass->SaveScene(m_scenePath);
                m_ui.sceneFeedbackTimer = 3.f;
            }
            ImGui::SameLine();
            if (ImGui::Button("Load Scene")) {
                m_ui.sceneLoadSuccess = m_pass->LoadScene(m_scenePath);
                m_ui.sceneFeedbackTimer = 3.f;
            }
            if (m_ui.sceneFeedbackTimer > 0.f) {
                m_ui.sceneFeedbackTimer -= ImGui::GetIO().DeltaTime;
                if (m_ui.sceneSaveSuccess) ImGui::TextColored({0,1,0,1}, "Scene saved successfully.");
                if (m_ui.sceneLoadSuccess) ImGui::TextColored({0,1,0,1}, "Scene loaded successfully.");
                if (!m_ui.sceneSaveSuccess && !m_ui.sceneLoadSuccess)
                    ImGui::TextColored({1,0,0,1}, "Operation failed - check log.");
            }
        }

        ImGui::Separator();

        // =====================================================================
        // L-SYSTEMS
        // =====================================================================
        if (ImGui::CollapsingHeader("L-Systems")) {
            const auto& lsystems = m_pass->GetLSystems();
            ImGui::Text("Count: %zu", lsystems.size());

            for (size_t i = 0; i < lsystems.size(); i++) {
                ImGui::PushID((int)(i + 2000));
                const auto& ls = lsystems[i];
                if (ImGui::TreeNode(ls.name.c_str())) {
                    ImGui::Text("Axiom: %s", ls.axiomStr.c_str());
                    ImGui::Text("Rules:");
                    for (const auto& [c, r] : ls.rulesStr)
                        ImGui::BulletText("'%c' -> %s", c, r.c_str());
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }

            ImGui::Separator();
            ImGui::Text("Add L-System:");
            ImGui::InputText("Name##lsname",  m_lsystem.name,  sizeof(m_lsystem.name));
            ImGui::InputText("Axiom##lsaxiom", m_lsystem.axiom, sizeof(m_lsystem.axiom));

            for (int r = 0; r < m_lsystem.ruleCount; r++) {
                ImGui::PushID(r);
                ImGui::SetNextItemWidth(40);
                ImGui::InputText("##from", m_lsystem.rules[r].from, sizeof(m_lsystem.rules[r].from));
                ImGui::SameLine(); ImGui::Text("->"); ImGui::SameLine();
                ImGui::SetNextItemWidth(200);
                ImGui::InputText("##to", m_lsystem.rules[r].to, sizeof(m_lsystem.rules[r].to));
                ImGui::PopID();
            }
            if (m_lsystem.ruleCount < 8 && ImGui::SmallButton("+Rule"))
                m_lsystem.ruleCount++;
            ImGui::SameLine();
            if (m_lsystem.ruleCount > 1 && ImGui::SmallButton("-Rule"))
                m_lsystem.ruleCount--;

            if (ImGui::Button("Add L-System##addls")) {
                std::unordered_map<char, std::string> rules;
                for (int r = 0; r < m_lsystem.ruleCount; r++)
                    if (m_lsystem.rules[r].from[0] != '\0')
                        rules[m_lsystem.rules[r].from[0]] = m_lsystem.rules[r].to;
                m_pass->AddLSystem(m_lsystem.name, m_lsystem.axiom, rules);
                // reset
                strcpy_s(m_lsystem.name,  "NewLSystem");
                strcpy_s(m_lsystem.axiom, "X");
                for (auto& rule : m_lsystem.rules) { rule.from[0] = '\0'; rule.to[0] = '\0'; }
                m_lsystem.ruleCount = 1;
            }
        }

        ImGui::Separator();

        // =====================================================================
        // TREE ASSETS
        // =====================================================================
        if (ImGui::CollapsingHeader("Tree Assets", ImGuiTreeNodeFlags_DefaultOpen)) {
            const auto& assets   = m_pass->GetTreeAssets();
            const auto& lsystems = m_pass->GetLSystems();

            static constexpr uint32_t genMin = 1,  genMax = 12;
            static constexpr uint32_t segMin = 3,  segMax = 32;

            ImGui::Text("Count: %zu", assets.size());
            ImGui::Separator();

            // ---- Editor form ----
            ImGui::Text(m_asset.editing ? "Editing Asset #%zu" : "New Asset", m_asset.editingIndex);
            ImGui::InputText("Name##aname", m_asset.name, sizeof(m_asset.name));

            // L-System combo
            if (!lsystems.empty()) {
                const char* preview = lsystems[(size_t)m_asset.lsysIdx].name.c_str();
                if (ImGui::BeginCombo("L-System##acombo", preview)) {
                    for (size_t i = 0; i < lsystems.size(); i++) {
                        bool sel = (i == (size_t)m_asset.lsysIdx);
                        if (ImGui::Selectable(lsystems[i].name.c_str(), sel))
                            m_asset.lsysIdx = (int)i;
                        if (sel) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
            }

            ImGui::SliderScalar("Generation",    ImGuiDataType_U32, &m_asset.generation,   &genMin, &genMax);
            ImGui::SliderScalar("Radial Segs",   ImGuiDataType_U32, &m_asset.radialSegs,   &segMin, &segMax);
            ImGui::SliderFloat("Branch Angle°",  &m_asset.branchAngle, 5.f, 90.f);
            ImGui::SliderFloat("Taper Ratio",    &m_asset.taperRatio,  0.5f, 1.0f);
            ImGui::SliderFloat("Step Ratio",     &m_asset.stepRatio,   0.5f, 1.0f);

            TreeGenerator::Params params;
            params.radialSegments = m_asset.radialSegs;
            params.branchAngle    = dm::radians(m_asset.branchAngle);
            params.taperRatio     = m_asset.taperRatio;
            params.stepRatio      = m_asset.stepRatio;

            if (m_asset.editing) {
                if (ImGui::Button("Update##uasset")) {
                    m_pass->UpdateTreeAsset(m_asset.editingIndex, m_asset.name,
                        (size_t)m_asset.lsysIdx, m_asset.generation, params);
                    m_asset.editing = false;
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel##casset")) m_asset.editing = false;
            } else {
                if (ImGui::Button("Add Asset##addasset")) {
                    m_pass->AddTreeAsset(m_asset.name, (size_t)m_asset.lsysIdx, m_asset.generation, params);
                    strcpy_s(m_asset.name, "NewAsset");
                    m_asset.generation = 3; m_asset.radialSegs = 16;
                    m_asset.branchAngle = 25.f; m_asset.taperRatio = 0.9f; m_asset.stepRatio = 0.95f;
                }
            }

            ImGui::Separator();
            ImGui::Text("Existing:");
            for (size_t i = 0; i < assets.size(); i++) {
                const auto& a = assets[i];
                ImGui::PushID((int)i);
                ImGui::Text("#%zu  %s  (gen %u, segs %u)",
                    i, a.name.c_str(), a.config.generation, a.config.params.radialSegments);
                if (ImGui::Button("Edit##ea")) {
                    m_asset.editing      = true;
                    m_asset.editingIndex = i;
                    m_asset.lsysIdx      = (int)a.config.lsystemIndex;
                    m_asset.generation   = a.config.generation;
                    m_asset.radialSegs   = a.config.params.radialSegments;
                    m_asset.branchAngle  = dm::degrees(a.config.params.branchAngle);
                    m_asset.taperRatio   = a.config.params.taperRatio;
                    m_asset.stepRatio    = a.config.params.stepRatio;
                    strcpy_s(m_asset.name, a.name.c_str());
                }
                ImGui::SameLine();
                if (ImGui::Button("Remove##ra")) m_pass->RemoveTreeAsset(i);
                ImGui::PopID();
            }
        }

        ImGui::Separator();

        // =====================================================================
        // REGIONS
        // =====================================================================
        if (ImGui::CollapsingHeader("Regions", ImGuiTreeNodeFlags_DefaultOpen)) {
            const auto& regions = m_pass->GetRegions();
            const auto& assets  = m_pass->GetTreeAssets();

            static constexpr uint32_t instMin = 1, instMax = 2000;

            ImGui::Text("Count: %zu", regions.size());
            ImGui::Separator();

            ImGui::Text(m_region.editing ? "Editing Region #%zu" : "New Region", m_region.editingIndex);
            ImGui::InputText("Name##rname",       m_region.name, sizeof(m_region.name));
            ImGui::SliderScalar("Instance Count", ImGuiDataType_U32, &m_region.instanceCount, &instMin, &instMax);
            ImGui::DragFloat2("Bounds Min##rmin", &m_region.minX, 0.5f);
            ImGui::DragFloat2("Bounds Max##rmax", &m_region.maxX, 0.5f);

            ImGui::Text("Assets used:");
            for (size_t i = 0; i < assets.size(); i++) {
                bool sel = m_region.selectedAssets.count(i) > 0;
                if (ImGui::Checkbox(assets[i].name.c_str(), &sel)) {
                    if (sel) m_region.selectedAssets.insert(i);
                    else     m_region.selectedAssets.erase(i);
                }
            }

            dm::box2 bounds(dm::float2(m_region.minX, m_region.minY),
                            dm::float2(m_region.maxX, m_region.maxY));

            if (m_region.editing) {
                if (ImGui::Button("Update##ureg")) {
                    // Rebuild region in-place
                    auto& r = m_pass->GetRegions()[m_region.editingIndex];
                    r.name          = m_region.name;
                    r.instanceCount = m_region.instanceCount;
                    r.bounds        = bounds;
                    r.assetIndices.clear();
                    for (size_t i : m_region.selectedAssets) r.assetIndices.push_back(i);
                    m_pass->RegenerateRegions();
                    m_region.editing = false;
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel##creg")) m_region.editing = false;
            } else {
                if (ImGui::Button("Add Region##addreg"))
                    m_pass->AddRegion(m_region.name, m_region.instanceCount, bounds);
            }

            ImGui::Separator();
            ImGui::Text("Existing:");
            for (size_t i = 0; i < regions.size(); i++) {
                const auto& r = regions[i];
                ImGui::PushID((int)(i + 1000));
                ImGui::Text("#%zu  %s  (instances: %u)", i, r.name.c_str(), r.instanceCount);
                if (ImGui::Button("Edit##er")) {
                    m_region.editing      = true;
                    m_region.editingIndex = i;
                    m_region.instanceCount = r.instanceCount;
                    m_region.minX = r.bounds.m_mins.x; m_region.minY = r.bounds.m_mins.y;
                    m_region.maxX = r.bounds.m_maxs.x; m_region.maxY = r.bounds.m_maxs.y;
                    m_region.selectedAssets.clear();
                    for (size_t idx : r.assetIndices) m_region.selectedAssets.insert(idx);
                    strcpy_s(m_region.name, r.name.c_str());
                }
                ImGui::SameLine();
                if (ImGui::Button("Remove##rr")) m_pass->RemoveRegion(i);
                ImGui::PopID();
            }
        }

        if (ImGui::Button("Hide UI  [ESC]")) m_ui.ShowUI = false;
        ImGui::End();
    }
};


// ===========================================================================
// Entry point
// ===========================================================================
#ifdef WIN32
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
#else
int main(int __argc, const char** __argv)
#endif
{
    nvrhi::GraphicsAPI api = app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    app::DeviceManager* deviceManager = app::DeviceManager::Create(api);

    app::DeviceCreationParameters deviceParams;
#ifdef _DEBUG
    deviceParams.enableDebugRuntime         = true;
    deviceParams.enableNvrhiValidationLayer = true;
#endif
    deviceParams.depthBufferFormat = nvrhi::Format::D16;

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle)) {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }
    log::info("Physical Device: %s", deviceManager->GetRendererString());

    {
        UIData uiData;
        TraditionalRenderPass renderPass(deviceManager, uiData);

        if (renderPass.Init()) {
            std::filesystem::path fwShaderPath =
                app::GetDirectoryWithExecutable().parent_path() / "shaders/framework" /
                app::GetShaderTypeName(deviceManager->GetDevice()->getGraphicsAPI());
            auto rootFS = std::make_shared<vfs::RootFileSystem>();
            rootFS->mount("/shaders/donut", fwShaderPath);
            auto sf = std::make_shared<engine::ShaderFactory>(
                deviceManager->GetDevice(), rootFS, "/shaders");

            XylemUIRenderer uiPass(deviceManager, &renderPass, uiData);
            uiPass.Init(sf);

            deviceManager->AddRenderPassToBack(&renderPass);
            deviceManager->AddRenderPassToBack(&uiPass);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&uiPass);
            deviceManager->RemoveRenderPass(&renderPass);
        }
    }

    deviceManager->Shutdown();
    delete deviceManager;
    return 0;
}
