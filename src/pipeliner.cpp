#include <donut/app/ApplicationBase.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <nvrhi/utils.h>

#include <donut/app/Camera.h>
#include <donut/engine/View.h>

// #include <donut/engine/SceneGraph.h>
#include <donut/app/imgui_renderer.h>

#include "include/procgen.hpp"
#include "include/pipeliner_macros.h"

#include <set>
#include <numeric>
#include <algorithm>

using namespace donut;
using namespace ProcGen;

static const char* g_WindowTitle = "Pipeliner";

#pragma region CUBE_DATA
// static const Vertex g_Vertices[] = {
//     { {-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f} }, // front face
//     { { 0.5f, -0.5f, -0.5f}, {1.0f, 1.0f} },
//     { {-0.5f, -0.5f, -0.5f}, {0.0f, 1.0f} },
//     { { 0.5f,  0.5f, -0.5f}, {1.0f, 0.0f} },

//     { { 0.5f, -0.5f, -0.5f}, {0.0f, 1.0f} }, // right side face
//     { { 0.5f,  0.5f,  0.5f}, {1.0f, 0.0f} },
//     { { 0.5f, -0.5f,  0.5f}, {1.0f, 1.0f} },
//     { { 0.5f,  0.5f, -0.5f}, {0.0f, 0.0f} },

//     { {-0.5f,  0.5f,  0.5f}, {0.0f, 0.0f} }, // left side face
//     { {-0.5f, -0.5f, -0.5f}, {1.0f, 1.0f} },
//     { {-0.5f, -0.5f,  0.5f}, {0.0f, 1.0f} },
//     { {-0.5f,  0.5f, -0.5f}, {1.0f, 0.0f} },

//     { { 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f} }, // back face
//     { {-0.5f, -0.5f,  0.5f}, {1.0f, 1.0f} },
//     { { 0.5f, -0.5f,  0.5f}, {0.0f, 1.0f} },
//     { {-0.5f,  0.5f,  0.5f}, {1.0f, 0.0f} },

//     { {-0.5f,  0.5f, -0.5f}, {0.0f, 1.0f} }, // top face
//     { { 0.5f,  0.5f,  0.5f}, {1.0f, 0.0f} },
//     { { 0.5f,  0.5f, -0.5f}, {1.0f, 1.0f} },
//     { {-0.5f,  0.5f,  0.5f}, {0.0f, 0.0f} },

//     { { 0.5f, -0.5f,  0.5f}, {1.0f, 1.0f} }, // bottom face
//     { {-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f} },
//     { { 0.5f, -0.5f, -0.5f}, {1.0f, 0.0f} },
//     { {-0.5f, -0.5f,  0.5f}, {0.0f, 1.0f} },
// };

// static const uint32_t g_Indices[] = {
//      0,  1,  2,   0,  3,  1, // front face
//      4,  5,  6,   4,  7,  5, // left face
//      8,  9, 10,   8, 11,  9, // right face
//     12, 13, 14,  12, 15, 13, // back face
//     16, 17, 18,  16, 19, 17, // top face
//     20, 21, 22,  20, 23, 21, // bottom face
// };
#pragma endregion


// Deterministic random number generator using hash
// This can be replicated on CPU and GPU
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

struct UIData {
    bool ShowUI = true;
};


class Pipeliner : public app::IRenderPass {
public:
    static constexpr uint32_t   c_LODs[]            = { 16,     8,      4 };
    static constexpr float      c_LODDistances[]    = { 16.0f,  64.0f,  256.0f };
    static constexpr uint32_t   c_NumLODs           = sizeof(c_LODs) / sizeof(uint32_t);
    
    static constexpr uint32_t   m_GlobalSeed        = 0xDEADBEEF;

    struct LSystemInfo {
        std::string name;
        std::unique_ptr<LSystem> system;
        
        LSystemInfo(const std::string& n, std::unique_ptr<LSystem> s)
            : name(n), system(std::move(s)) {}
    };

    struct InstanceBufferEntry
    {
        dm::float4x4    model;
        dm::float3x3    normal;
        uint32_t        treeId;

        InstanceBufferEntry(const dm::float4x4& m, const dm::float3x3& n, uint32_t t) : model{m}, normal{n}, treeId{t} {}
        InstanceBufferEntry(const dm::affine3& m, const dm::float3x3& n, uint32_t t) 
            : InstanceBufferEntry(dm::affineToHomogeneous(m), n, t) { }
    };
    // static constexpr size_t c_InstanceBufferSize = sizeof(InstanceBufferEntry) * c_InstanceCount;

    struct ConstantBufferEntry {
        dm::float4x4 view;
        dm::float4x4 projection;

        dm::float4x4 padding[2];
    };
    static constexpr size_t c_ConstantBufferSize = sizeof(ConstantBufferEntry) + (sizeof(ConstantBufferEntry) % nvrhi::c_ConstantBufferOffsetSizeAlignment);

    struct ViewHandler {
        app::FirstPersonCamera      camera;
        engine::PlanarView          view;

        uint32_t distToLOD(float value, const float* arr, int size) {
            // We use pointers as iterators: arr is the start, arr + size is the end
            const float* it = std::lower_bound(arr, arr + size, value);

            if (it == arr + size) {
                return size - 1; // Value is greater than all elements
            }

            // Pointer subtraction gives the index
            return static_cast<uint32_t>(it - arr);
        }
    };

    struct LSystemSet {
        std::unique_ptr<LSystem>    lSystem;
        std::set<lgen_t>            generations;             
    };
    
    struct TreeGenerationConfig {
        size_t                      lsystemIndex;  // Index into the LSystems map
        lgen_t                      generation;
        TreeGenerator::Params       params;
    };

    struct TreeLODData {
        nvrhi::BufferHandle vertexBuffer;
        nvrhi::BufferHandle indexBuffer;
        uint32_t            indexCount;
        uint32_t            radialSegments;
        dm::box3            bbox;
    };

    struct TreeAsset {
        std::string                         name;  // Friendly name
        TreeGenerationConfig                config;
        lstring_t                           lsystemString;
        std::array<TreeLODData, c_NumLODs>  lods;
    };

    struct TreeRegion {
        static inline uint32_t s_TotalInstanceCount = 0;

        std::string                         name;  // Friendly name
        uint32_t                            instanceCount;
        dm::box2                            bounds;
        std::vector<size_t>                 assetIndices;  // Which assets this region uses
        std::vector<InstanceBufferEntry>    instanceBuffer;
        std::vector<dm::box3>               instanceBbox;
        dm::box3                            cullBox;

        TreeRegion(const std::string& n, uint32_t c, const dm::box2& b)
            : name(n), instanceCount{c}, bounds{b}, cullBox{dm::box3::empty()}
        { 
            instanceBuffer.reserve(instanceCount); 
            instanceBbox.reserve(instanceCount); 
            s_TotalInstanceCount += instanceCount; 
        }
    };

    struct DrawCmd {
        nvrhi::BufferHandle     vertexBuffer;
        nvrhi::BufferHandle     indexBuffer;
        nvrhi::DrawArguments    drawArgs;
    };
    
    struct InstanceReference {
        uint32_t regionIdx;
        uint32_t instanceIdx;
        uint32_t treeId;
        uint32_t lodID;
    };
    
    struct GPUResources {
        nvrhi::ShaderHandle             vertexShader;
        nvrhi::ShaderHandle             pixelShader;
        nvrhi::TextureHandle            texture;
        nvrhi::SamplerHandle            sampler;
        
        nvrhi::InputLayoutHandle        inputLayout;

        nvrhi::BufferHandle             constantBuffer;
        nvrhi::BufferHandle             instanceBuffer;

        nvrhi::BindingLayoutHandle      bindingLayout;
        nvrhi::BindingSetHandle         bindingSet;
        
        nvrhi::GraphicsPipelineHandle   pipeline;
    };

public:
    using IRenderPass::IRenderPass;

private:
    
    GPUResources                                                            m_Resources;
    nvrhi::CommandListHandle                                                m_CommandList;

    std::unique_ptr<ViewHandler>                                            m_ViewHandler;

    // generators - map avoids reindexing when removing
    std::map<std::string, std::unique_ptr<LSystem>>                         m_LSystems;
    std::unique_ptr<TreeGenerator>                                          m_TreeGenerator;

    std::vector<TreeAsset>                                                  m_TreeAssets;
    
    // Instance regions
    std::vector<TreeRegion>                                                 m_Regions;
    
    
    // std::vector<std::array<nvrhi::BufferHandle, c_NumLODs>>                 m_VertexBuffers;
    // std::vector<std::array<nvrhi::BufferHandle, c_NumLODs>>                 m_IndexBuffers;
    
    std::vector<InstanceReference>                                          m_VisibleInstanceReferences;
    std::vector<std::array<uint32_t, c_NumLODs>>                            m_InstanceCounts;
    std::vector<std::array<uint32_t, c_NumLODs>>                            m_InstanceOffsets;
    std::vector<DrawCmd>                                                    m_DrawCmds;
    std::vector<InstanceBufferEntry>                                        m_VisibleInstanceBuffer;

private:
#pragma region INIT
    bool _InitTreeData() {
        // Initialize L-Systems map with friendly names
        m_LSystems["Standard"] = std::make_unique<LSystem>(
            "X", 
            std::unordered_map<char, std::string>{
                {'X', "F+[[>X]-<X]-F[-FX]+X"},
                {'F', "FF"}
            }
        );

        m_TreeGenerator = std::make_unique<TreeGenerator>();

        // Define tree generation configs with lsystem indices
        std::vector<TreeGenerationConfig> genConfigs = {
            {
                0,  // Standard L-System index (will be determined from map)
                3,
                { c_LODs[0], 1.f, dm::radians(25.f), 0.9f, 0.95f, 0 },
            },
            {
                0,  // Standard L-System index
                4,
                { c_LODs[0], 1.f, dm::radians(30.f), 0.9f, 0.95f, 0 },
            },
        };

        // Build a name->index map for LSystems
        std::map<std::string, size_t> lsystemIndexMap;
        size_t lsysIdx = 0;
        for (const auto& [name, system] : m_LSystems) {
            lsystemIndexMap[name] = lsysIdx++;
        }

        // Update config indices to use proper map-based indices
        genConfigs[0].lsystemIndex = lsystemIndexMap["Standard"];
        genConfigs[1].lsystemIndex = lsystemIndexMap["Standard"];

        nvrhi::BufferDesc vertexBufferDesc;
        vertexBufferDesc.isVertexBuffer = true;
        vertexBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;

        nvrhi::BufferDesc indexBufferDesc;
        indexBufferDesc.isIndexBuffer = true;
        indexBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;

        m_TreeAssets.resize(genConfigs.size());
        std::array<ProcGenBuffers, c_NumLODs>  lods;
        
        size_t configIdx = 0;
        for (auto& treeAsset : m_TreeAssets) {
            auto& genConfig = genConfigs[configIdx];
            
            treeAsset.name = "Asset_" + std::to_string(configIdx);
            treeAsset.config = genConfig;

            // Get the L-System by finding it in the map
            LSystem* lSystem = nullptr;
            size_t idx = 0;
            for (auto& [name, system] : m_LSystems) {
                if (idx == genConfig.lsystemIndex) {
                    lSystem = system.get();
                    break;
                }
                idx++;
            }

            if (!lSystem) {
                log::error("Invalid L-System index: %zu", genConfig.lsystemIndex);
                return false;
            }

            lSystem->reset();
            lSystem->generate(genConfig.generation);
            treeAsset.lsystemString = lSystem->getCurrentString();

            auto& genParams = treeAsset.config.params;
            for (size_t j = 0; j < c_NumLODs; j++) {
                genParams.radialSegments = c_LODs[j];
                m_TreeGenerator->setParams(genParams);
                m_TreeGenerator->resetRandomGenerator();
                m_TreeGenerator->generateVertexAndIndexBuffers(treeAsset.lsystemString, lods[j]);

                auto& vBuf = treeAsset.lods[j].vertexBuffer;
                auto& iBuf = treeAsset.lods[j].indexBuffer;
                vertexBufferDesc.debugName = "VertexBuffer_" + treeAsset.name + "_LOD" + std::to_string(j);
                vertexBufferDesc.byteSize = lods[j].vertices.size() * sizeof(ProcGen::TreeVertex);
                vBuf = GetDevice()->createBuffer(vertexBufferDesc);

                m_CommandList->beginTrackingBufferState(vBuf, nvrhi::ResourceStates::CopyDest);
                m_CommandList->writeBuffer(vBuf, lods[j].vertices.data(), vertexBufferDesc.byteSize);
                m_CommandList->setPermanentBufferState(vBuf, nvrhi::ResourceStates::VertexBuffer);

                indexBufferDesc.debugName = "IndexBuffer_" + treeAsset.name + "_LOD" + std::to_string(j);
                indexBufferDesc.byteSize = lods[j].indices.size() * sizeof(uint32_t);
                treeAsset.lods[j].indexBuffer = GetDevice()->createBuffer(indexBufferDesc);
                
                m_CommandList->beginTrackingBufferState(iBuf, nvrhi::ResourceStates::CopyDest);
                m_CommandList->writeBuffer(iBuf, lods[j].indices.data(), indexBufferDesc.byteSize);
                m_CommandList->setPermanentBufferState(iBuf, nvrhi::ResourceStates::IndexBuffer);
                
                treeAsset.lods[j].indexCount = static_cast<uint32_t>(lods[j].indices.size());
                treeAsset.lods[j].radialSegments = genParams.radialSegments;
                treeAsset.lods[j].bbox = lods[j].bbox;
            }

            configIdx++;
        }

        return true;
    }

    bool _InitRegions() {
        m_Regions = {
            TreeRegion("Region_0", 10, { dm::float2(-10.f,-20.f), dm::float2(0.f,0.f) }),
            TreeRegion("Region_1", 25, { dm::float2(10.f,0.f), dm::float2(20.f,20.f) }),
        };

        // Initialize asset indices for each region (use all assets by default)
        for (auto& r : m_Regions) {
            for (size_t i = 0; i < m_TreeAssets.size(); ++i) {
                r.assetIndices.push_back(i);
            }
        }

        for (auto& r : m_Regions) {
            dm::float2& min = r.bounds.m_mins;
            dm::float2& range = r.bounds.diagonal();

            for (uint32_t i = 0; i < r.instanceCount; ++i) {
                uint32_t seedX = m_GlobalSeed ^ 0x12345678;
                uint32_t seedZ = m_GlobalSeed ^ 0x87654321;

                float randX = hashToFloat(i, seedX);
                float randZ = hashToFloat(i, seedZ);

                float posX = min.x + (randX * range.x);
                float posZ = min.y + (randZ * range.y);

                dm::affine3 worldMatrix = 
                    dm::rotation(normalize(dm::float3(1,0,0)), -dm::PI_f/2.0f) * 
                    dm::translation(dm::float3(posX, 0.f, posZ))
                ;
                dm::float3x3 normalMatrix;

                const auto& m = worldMatrix.m_linear;

                float scaleSqX = dm::lengthSquared(m.row0);
                float scaleSqY = dm::lengthSquared(m.row1);
                float scaleSqZ = dm::lengthSquared(m.row2);

                if (dm::isnear(scaleSqX, scaleSqY) && dm::isnear(scaleSqY, scaleSqZ)) {
                    normalMatrix = m;
                }
                else {
                    normalMatrix = dm::transpose(dm::inverse(m));
                }

                // Pick a random asset from this region's asset set
                size_t assetIdx = 0;
                if (!r.assetIndices.empty()) {
                    uint32_t randomAssetChoice = hash(i, m_GlobalSeed ^ 0xFEEDBEEF) % r.assetIndices.size();
                    assetIdx = r.assetIndices[randomAssetChoice];
                }

                r.instanceBuffer.emplace_back(worldMatrix, normalMatrix, assetIdx);

                dm::box3 treeBBox = m_TreeAssets[assetIdx].lods[0].bbox * worldMatrix;
                r.instanceBbox.push_back(treeBBox);

                r.cullBox |= treeBBox;
            }
        }

        return true;
    }

    bool _InitShaders() {
        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable().parent_path() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable().parent_path() / "shaders/custom" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());

        std::shared_ptr<vfs::RootFileSystem> rootFS = std::make_shared<vfs::RootFileSystem>();
        rootFS->mount("/shaders/donut", frameworkShaderPath);
        rootFS->mount("/shaders/app", appShaderPath);

        std::shared_ptr<engine::ShaderFactory> shaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), rootFS, "/shaders");
        m_Resources.vertexShader = shaderFactory->CreateShader("app/shaders.hlsl", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
        m_Resources.pixelShader = shaderFactory->CreateShader("app/shaders.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);

        return (!!m_Resources.vertexShader && !!m_Resources.pixelShader);
    }

    bool _InitVertexAttributes() {
        nvrhi::VertexAttributeDesc attributes[] = {
        #if PIPELINER_USE_INTERLEAVED_VERTEX_ATTRIBUTES
            nvrhi::VertexAttributeDesc()
                .setName("POSITION")
                .setFormat(nvrhi::Format::RGB32_FLOAT)
                .setOffset(offsetof(TreeVertex, pos))
                .setBufferIndex(0)
                .setElementStride(sizeof(TreeVertex)),
            nvrhi::VertexAttributeDesc()
                .setName("NORMAL")
                .setFormat(nvrhi::Format::RGB32_FLOAT)
                .setOffset(offsetof(TreeVertex, normal))
                .setBufferIndex(0)
                .setElementStride(sizeof(TreeVertex)),
            nvrhi::VertexAttributeDesc()
                .setName("UV")
                .setFormat(nvrhi::Format::RG32_FLOAT)
                .setOffset(offsetof(TreeVertex, uv))
                .setBufferIndex(0)
                .setElementStride(sizeof(TreeVertex)),
            #if !PIPELINER_USE_STRUCTURED_BUFFER
            nvrhi::VertexAttributeDesc()
                .setName("MODEL_MATRIX")
                .setFormat(nvrhi::Format::RGBA32_FLOAT)
                .setArraySize(4)
                .setOffset(offsetof(InstanceBufferEntry, model))
                .setBufferIndex(1)
                .setElementStride(sizeof(InstanceBufferEntry))
                .setIsInstanced(true),
            nvrhi::VertexAttributeDesc()
                .setName("NORMAL_MATRIX")
                .setFormat(nvrhi::Format::RGB32_FLOAT)
                .setArraySize(3)
                .setOffset(offsetof(InstanceBufferEntry, normal))
                .setBufferIndex(1)
                .setElementStride(sizeof(InstanceBufferEntry))
                .setIsInstanced(true),
            #endif
        #else
            nvrhi::VertexAttributeDesc()
                .setName("POSITION")
                .setFormat(nvrhi::Format::RGB32_FLOAT)
                .setOffset(0)
                .setBufferIndex(0)
                .setElementStride(sizeof(TreeVertex)),
            nvrhi::VertexAttributeDesc()
                .setName("NORMAL")
                .setFormat(nvrhi::Format::RGB32_FLOAT)
                .setOffset(0)
                .setBufferIndex(1)
                .setElementStride(sizeof(TreeVertex)),
            nvrhi::VertexAttributeDesc()
                .setName("UV")
                .setFormat(nvrhi::Format::RG32_FLOAT)
                .setOffset(0)
                .setBufferIndex(2)
                .setElementStride(sizeof(TreeVertex)),
            #if !PIPELINER_USE_STRUCTURED_BUFFER
            nvrhi::VertexAttributeDesc()
                .setName("MODEL_MATRIX")
                .setFormat(nvrhi::Format::RGBA32_FLOAT)
                .setArraySize(4)
                .setOffset(0)
                .setBufferIndex(3)
                .setElementStride(sizeof(InstanceBufferEntry))
                .setIsInstanced(true),
            nvrhi::VertexAttributeDesc()
                .setName("NORMAL_MATRIX")
                .setFormat(nvrhi::Format::RGB32_FLOAT)
                .setArraySize(3)
                .setOffset(0)
                .setBufferIndex(4)
                .setElementStride(sizeof(InstanceBufferEntry))
                .setIsInstanced(true),
            #endif
        #endif
        };

        m_Resources.inputLayout = GetDevice()->createInputLayout(attributes, uint32_t(std::size(attributes)), m_Resources.vertexShader);
        return !!m_Resources.inputLayout;
    }

    bool _InitBuffers() {
        m_Resources.instanceBuffer = GetDevice()->createBuffer(
            nvrhi::BufferDesc()
                .setByteSize(TreeRegion::s_TotalInstanceCount * sizeof(InstanceBufferEntry))
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
                .setStructStride(0)
                // .setIsVolatile(true) // what it do
                // .setMaxVersions(8)
                .setIsConstantBuffer(true)
                .setDebugName("ConstantBuffer")
                .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
        );

        return !!m_Resources.instanceBuffer && !!m_Resources.constantBuffer;
    }

    bool _InitTextureAndSampler() {
        engine::TextureCache textureCache(GetDevice(), std::make_shared<vfs::NativeFileSystem>(), nullptr);
        std::filesystem::path textureFileName = app::GetDirectoryWithExecutable().parent_path().parent_path() / "media/bark_willow_02_diff_1k.jpg";
        std::shared_ptr<engine::LoadedTexture> texture = textureCache.LoadTextureFromFile(textureFileName, true, nullptr, m_CommandList);
        
        m_Resources.texture = texture->texture;
        m_Resources.sampler = GetDevice()->createSampler(
            nvrhi::SamplerDesc()
                .setAllAddressModes(nvrhi::SamplerAddressMode::Wrap)
        );

        return !!m_Resources.texture && !!m_Resources.sampler;
    }

    bool _InitBindingLayoutAndSet() {
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_Resources.constantBuffer, nvrhi::BufferRange(0, c_ConstantBufferSize)),
            nvrhi::BindingSetItem::Sampler(0, m_Resources.sampler),
            nvrhi::BindingSetItem::Texture_SRV(0, m_Resources.texture),
            #if PIPELINER_USE_STRUCTURED_BUFFER
            nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_Resources.instanceBuffer, nvrhi::Format::UNKNOWN, nvrhi::BufferRange(0, TreeRegion::s_TotalInstanceCount * sizeof(InstanceBufferEntry))),
            nvrhi::BindingSetItem::PushConstants(1, sizeof(uint32_t)),
            #else
            //
            #endif
        };

        // Create the binding layout (if it's empty -- so, on the first iteration) and the binding set.
        if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_Resources.bindingLayout, m_Resources.bindingSet))
        {
            log::error("Couldn't create the binding set or layout");
            return false;
        }

        return !!m_Resources.bindingLayout && !!m_Resources.bindingSet;
    }

    bool _InitViewHandler() {
        m_ViewHandler = std::make_unique<ViewHandler>();
        m_ViewHandler->camera.LookTo(dm::float3(5,5,-5), dm::float3(0,0,1));
        m_ViewHandler->camera.SetMoveSpeed(15.f);

        return !!m_ViewHandler;
    }

    void _RegenerateTreeAssets() {
        // Clear old data
        for (auto& asset : m_TreeAssets) {
            for (auto& lod : asset.lods) {
                lod.vertexBuffer = nullptr;
                lod.indexBuffer = nullptr;
            }
        }
        m_TreeAssets.clear();

        // Regenerate
        m_CommandList->open();
        _InitTreeData();
        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
        m_CommandList = GetDevice()->createCommandList();
    }

    void _RegenerateRegions() {
        // Clear old data
        TreeRegion::s_TotalInstanceCount = 0;
        m_Regions.clear();

        // Regenerate
        _InitRegions();

        // Recreate the instance buffer with the new size
        m_Resources.instanceBuffer = GetDevice()->createBuffer(
            nvrhi::BufferDesc()
                .setByteSize(TreeRegion::s_TotalInstanceCount * sizeof(InstanceBufferEntry))
                .setStructStride(sizeof(InstanceBufferEntry))
                .setDebugName("InstanceBuffer")
        #if PIPELINER_USE_STRUCTURED_BUFFER
                .enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource)
        #else
                .setIsVertexBuffer(true)
                .enableAutomaticStateTracking(nvrhi::ResourceStates::CopyDest)
        #endif
        );

        // Update instance counts and offsets arrays
        m_InstanceCounts.assign(m_TreeAssets.size(), {0});
        m_InstanceOffsets.assign(m_TreeAssets.size(), {0});

        // Recreate binding set with new instance buffer
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_Resources.constantBuffer, nvrhi::BufferRange(0, c_ConstantBufferSize)),
            nvrhi::BindingSetItem::Sampler(0, m_Resources.sampler),
            nvrhi::BindingSetItem::Texture_SRV(0, m_Resources.texture),
            #if PIPELINER_USE_STRUCTURED_BUFFER
            nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_Resources.instanceBuffer, nvrhi::Format::UNKNOWN, nvrhi::BufferRange(0, TreeRegion::s_TotalInstanceCount * sizeof(InstanceBufferEntry))),
            nvrhi::BindingSetItem::PushConstants(1, sizeof(uint32_t)),
            #else
            //
            #endif
        };
        nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_Resources.bindingLayout, m_Resources.bindingSet);
    }

#pragma endregion
public:
    // Accessors for UI
    std::vector<TreeAsset>& GetTreeAssets() { return m_TreeAssets; }
    std::vector<TreeRegion>& GetRegions() { return m_Regions; }
    const std::vector<TreeAsset>& GetTreeAssets() const { return m_TreeAssets; }
    const std::vector<TreeRegion>& GetRegions() const { return m_Regions; }
    const std::map<std::string, std::unique_ptr<LSystem>>& GetLSystems() const { return m_LSystems; }
    
    // Helper to get LSystem by index
    LSystem* GetLSystemByIndex(size_t idx) {
        size_t currentIdx = 0;
        for (auto& [name, system] : m_LSystems) {
            if (currentIdx == idx) return system.get();
            currentIdx++;
        }
        return nullptr;
    }
    
    // Add a new asset
    void AddTreeAsset(const std::string& name, size_t lsystemIndex, uint32_t generation, const TreeGenerator::Params& params) {
        TreeAsset newAsset;
        newAsset.name = name;
        newAsset.config = { lsystemIndex, generation, params };
        
        LSystem* lsys = GetLSystemByIndex(lsystemIndex);
        if (!lsys) return;
        
        lsys->reset();
        lsys->generate(generation);
        newAsset.lsystemString = lsys->getCurrentString();
        
        // Generate LOD data
        m_CommandList->open();
        std::array<ProcGenBuffers, c_NumLODs> lods;
        for (size_t j = 0; j < c_NumLODs; j++) {
            auto genParams = params;
            genParams.radialSegments = c_LODs[j];
            m_TreeGenerator->setParams(genParams);
            m_TreeGenerator->resetRandomGenerator();
            m_TreeGenerator->generateVertexAndIndexBuffers(newAsset.lsystemString, lods[j]);
            
            auto& vBuf = newAsset.lods[j].vertexBuffer;
            auto& iBuf = newAsset.lods[j].indexBuffer;
            
            nvrhi::BufferDesc vertexBufferDesc;
            vertexBufferDesc.isVertexBuffer = true;
            vertexBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
            vertexBufferDesc.debugName = "VertexBuffer_" + name + "_LOD" + std::to_string(j);
            vertexBufferDesc.byteSize = lods[j].vertices.size() * sizeof(ProcGen::TreeVertex);
            vBuf = GetDevice()->createBuffer(vertexBufferDesc);
            
            m_CommandList->beginTrackingBufferState(vBuf, nvrhi::ResourceStates::CopyDest);
            m_CommandList->writeBuffer(vBuf, lods[j].vertices.data(), vertexBufferDesc.byteSize);
            m_CommandList->setPermanentBufferState(vBuf, nvrhi::ResourceStates::VertexBuffer);
            
            nvrhi::BufferDesc indexBufferDesc;
            indexBufferDesc.isIndexBuffer = true;
            indexBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
            indexBufferDesc.debugName = "IndexBuffer_" + name + "_LOD" + std::to_string(j);
            indexBufferDesc.byteSize = lods[j].indices.size() * sizeof(uint32_t);
            iBuf = GetDevice()->createBuffer(indexBufferDesc);
            
            m_CommandList->beginTrackingBufferState(iBuf, nvrhi::ResourceStates::CopyDest);
            m_CommandList->writeBuffer(iBuf, lods[j].indices.data(), indexBufferDesc.byteSize);
            m_CommandList->setPermanentBufferState(iBuf, nvrhi::ResourceStates::IndexBuffer);
            
            newAsset.lods[j].indexCount = static_cast<uint32_t>(lods[j].indices.size());
            newAsset.lods[j].radialSegments = genParams.radialSegments;
            newAsset.lods[j].bbox = lods[j].bbox;
        }
        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
        m_CommandList = GetDevice()->createCommandList();
        
        m_TreeAssets.push_back(newAsset);
    }
    
    // Remove an asset by index
    void RemoveTreeAsset(size_t assetIdx) {
        if (assetIdx >= m_TreeAssets.size()) return;
        
        // Remove references from all regions
        for (auto& region : m_Regions) {
            auto it = std::find(region.assetIndices.begin(), region.assetIndices.end(), assetIdx);
            if (it != region.assetIndices.end()) {
                region.assetIndices.erase(it);
            }
            // Adjust indices greater than the removed one
            for (auto& idx : region.assetIndices) {
                if (idx > assetIdx) idx--;
            }
        }
        
        m_TreeAssets.erase(m_TreeAssets.begin() + assetIdx);
    }
    
    // Add a region
    void AddRegion(const std::string& name, uint32_t instanceCount, const dm::box2& bounds) {
        m_Regions.emplace_back(name, instanceCount, bounds);
        auto& newRegion = m_Regions.back();
        
        // Initialize with all assets
        for (size_t i = 0; i < m_TreeAssets.size(); ++i) {
            newRegion.assetIndices.push_back(i);
        }
        
        // Generate instances
        _RegenerateRegion(m_Regions.size() - 1);
        _RegenerateRegions(); // Recreate buffers
    }
    
    // Regenerate a specific region's instances
    void _RegenerateRegion(size_t regionIdx) {
        if (regionIdx >= m_Regions.size()) return;
        
        TreeRegion& r = m_Regions[regionIdx];
        r.instanceBuffer.clear();
        r.instanceBbox.clear();
        r.cullBox = dm::box3::empty();
        
        dm::float2& min = r.bounds.m_mins;
        dm::float2& range = r.bounds.diagonal();
        
        for (uint32_t i = 0; i < r.instanceCount; ++i) {
            uint32_t seedX = m_GlobalSeed ^ 0x12345678;
            uint32_t seedZ = m_GlobalSeed ^ 0x87654321;
            
            float randX = hashToFloat(i, seedX);
            float randZ = hashToFloat(i, seedZ);
            
            float posX = min.x + (randX * range.x);
            float posZ = min.y + (randZ * range.y);
            
            dm::affine3 worldMatrix = 
                dm::rotation(normalize(dm::float3(1,0,0)), -dm::PI_f/2.0f) * 
                dm::translation(dm::float3(posX, 0.f, posZ));
            dm::float3x3 normalMatrix;
            
            const auto& m = worldMatrix.m_linear;
            float scaleSqX = dm::lengthSquared(m.row0);
            float scaleSqY = dm::lengthSquared(m.row1);
            float scaleSqZ = dm::lengthSquared(m.row2);
            
            if (dm::isnear(scaleSqX, scaleSqY) && dm::isnear(scaleSqY, scaleSqZ)) {
                normalMatrix = m;
            } else {
                normalMatrix = dm::transpose(dm::inverse(m));
            }
            
            size_t assetIdx = 0;
            if (!r.assetIndices.empty()) {
                uint32_t randomAssetChoice = hash(i, m_GlobalSeed ^ 0xFEEDBEEF) % r.assetIndices.size();
                assetIdx = r.assetIndices[randomAssetChoice];
            }
            
            r.instanceBuffer.emplace_back(worldMatrix, normalMatrix, assetIdx);
            dm::box3 treeBBox = m_TreeAssets[assetIdx].lods[0].bbox * worldMatrix;
            r.instanceBbox.push_back(treeBBox);
            r.cullBox |= treeBBox;
        }
    }
    
    void RegenerateAssets() { _RegenerateTreeAssets(); }
    void RegenerateRegions() { _RegenerateRegions(); }
    bool Init()
    {
        m_CommandList = GetDevice()->createCommandList();
        m_CommandList->open();
        assert(!!m_CommandList);
        
        assert(_InitTreeData());
        assert(_InitRegions());
        
        assert(_InitShaders());
        

        assert(_InitVertexAttributes());
        assert(_InitBuffers());

        assert(_InitTextureAndSampler());

        assert(_InitBindingLayoutAndSet());

        assert(_InitViewHandler());

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);


        m_VisibleInstanceReferences.reserve(TreeRegion::s_TotalInstanceCount);

        // initialize to 0s, so assign
        m_InstanceCounts.resize(m_TreeAssets.size(), {0});
        // initialize to 0s, so assign
        m_InstanceOffsets.assign(m_TreeAssets.size(), {0});

        m_DrawCmds.reserve(m_TreeAssets.size() * c_NumLODs);
        m_VisibleInstanceBuffer.reserve(TreeRegion::s_TotalInstanceCount);

        return true;
    }

    void Animate(float seconds) override
    {
        m_ViewHandler->camera.Animate(seconds);
        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }

    void BackBufferResizing() override
    { 
        m_Resources.pipeline = nullptr;
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();

        if (!m_Resources.pipeline)
        {
            nvrhi::GraphicsPipelineDesc psoDesc;
            psoDesc.VS = m_Resources.vertexShader;
            psoDesc.PS = m_Resources.pixelShader;
            psoDesc.inputLayout = m_Resources.inputLayout;
            psoDesc.bindingLayouts = { m_Resources.bindingLayout };
            psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
        #if PIPELINER_USE_REVERSE_Z
            psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
        #else
            psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
        #endif

            m_Resources.pipeline = GetDevice()->createGraphicsPipeline(psoDesc, fbinfo);

            m_ViewHandler->view.SetViewport({float(fbinfo.width), float(fbinfo.height)});
        #if PIPELINER_USE_REVERSE_Z
            m_ViewHandler->view.SetProjectionMatrix(
                dm::perspProjD3DStyleReverse(
                    dm::radians(60.f), 
                    float(fbinfo.width) / float(fbinfo.height), 
                    0.1f
                )
            );
        #else
            m_ViewHandler->view.SetProjectionMatrix(
                dm::perspProjD3DStyle(
                    dm::radians(60.f), 
                    float(fbinfo.width) / float(fbinfo.height), 
                    0.1f,
                    std::numeric_limits<float>::max()
                )
            );
        #endif
        }

        m_CommandList->open();
        
        m_ViewHandler->view.SetViewMatrix(m_ViewHandler->camera.GetWorldToViewMatrix());
        m_ViewHandler->view.UpdateCache();


        nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));
    #if PIPELINER_USE_REVERSE_Z
        nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer, 0.f, 0);
    #else
        nvrhi::utils::ClearDepthStencilAttachment(
            m_CommandList, 
            framebuffer, 
            m_ViewHandler->view.GetViewFrustum().farPlane().distance, 
            0);
    #endif

        ConstantBufferEntry constants{};
        constants.view = dm::affineToHomogeneous(m_ViewHandler->view.GetViewMatrix());
        constants.projection = m_ViewHandler->view.GetProjectionMatrix();

        m_CommandList->writeBuffer(m_Resources.constantBuffer, &constants, c_ConstantBufferSize);

        m_VisibleInstanceReferences.clear();
        m_InstanceCounts.assign(m_TreeAssets.size(), {0});

        for (uint32_t regionIdx = 0; regionIdx < m_Regions.size(); regionIdx++) {
            const auto& region = m_Regions[regionIdx];

            // cull region
            if (!m_ViewHandler->view.IsBoxVisible(region.cullBox)) continue;

            for (uint32_t instanceIdx = 0; instanceIdx < region.instanceCount; instanceIdx++) {
                const auto& instance = region.instanceBuffer[instanceIdx];
                const auto& treeBbox = region.instanceBbox[instanceIdx];
                // cull tree
                if (!m_ViewHandler->view.IsBoxVisible(treeBbox)) continue;

                float distToCamera = dm::distance(m_ViewHandler->camera.GetPosition(), treeBbox);
                uint32_t lodIdx = m_ViewHandler->distToLOD(distToCamera, c_LODDistances, c_NumLODs);
                
                m_VisibleInstanceReferences.push_back({regionIdx, instanceIdx, instance.treeId, lodIdx});
                m_InstanceCounts[instance.treeId][lodIdx]++;
            }
        }

        m_DrawCmds.clear();
        m_InstanceOffsets.assign(m_TreeAssets.size(), {0});
        uint32_t instanceOffset = 0;

        for (uint32_t assetIdx = 0; assetIdx < m_InstanceCounts.size(); assetIdx++) {
            for (uint32_t lodIdx = 0; lodIdx < c_NumLODs; lodIdx++) {
                uint32_t instanceCount = m_InstanceCounts[assetIdx][lodIdx];
                if (instanceCount == 0) continue;

                const auto& lodData = m_TreeAssets[assetIdx].lods[lodIdx];
                m_DrawCmds.push_back(
                    {
                        lodData.vertexBuffer,
                        lodData.indexBuffer,
                        nvrhi::DrawArguments()
                            .setVertexCount(lodData.indexCount)
                            .setInstanceCount(instanceCount)
                            .setStartInstanceLocation(instanceOffset)
                    }
                );

                m_InstanceOffsets[assetIdx][lodIdx] = instanceOffset;
                instanceOffset+=instanceCount;
            }
        }

        InstanceBufferEntry* UNSAFE_VIEW_m_VisibleInstanceBuffer = m_VisibleInstanceBuffer.data();
        for (const auto& instanceRef : m_VisibleInstanceReferences) {
            const auto& instanceBufferEntry = m_Regions[instanceRef.regionIdx].instanceBuffer[instanceRef.instanceIdx];
            uint32_t& writeOffset = m_InstanceOffsets[instanceRef.treeId][instanceRef.lodID];

            UNSAFE_VIEW_m_VisibleInstanceBuffer[writeOffset] = instanceBufferEntry;
            writeOffset++;
        }

        m_CommandList->writeBuffer(
            m_Resources.instanceBuffer, m_VisibleInstanceBuffer.data(), 
            m_VisibleInstanceReferences.size() * sizeof(InstanceBufferEntry)
        );

        // set state
        {
            nvrhi::GraphicsState state;
            
            state.pipeline = m_Resources.pipeline;
            state.framebuffer = framebuffer;
            state.viewport = m_ViewHandler->view.GetViewportState();

            state.bindings = { m_Resources.bindingSet };

            for (const auto& drawCmd : m_DrawCmds) {
                state.vertexBuffers = {
                #if PIPELINER_USE_INTERLEAVED_VERTEX_ATTRIBUTES
                        { drawCmd.vertexBuffer,         /*slot*/ 0, /*offset*/ 0 },
                    #if !PIPELINER_USE_STRUCTURED_BUFFER
                        { m_Resources.instanceBuffer,   /*slot*/ 1, /*offset*/ 0 },
                    #endif
                #else
                        { drawCmd.vertexBuffer, /*slot*/ 0, /*offset*/ offsetof(TreeVertex, pos) },
                        { drawCmd.vertexBuffer, /*slot*/ 1, /*offset*/ offsetof(TreeVertex, normal) },
                        { drawCmd.vertexBuffer, /*slot*/ 2, /*offset*/ offsetof(TreeVertex, uv) },
                    #if !PIPELINER_USE_STRUCTURED_BUFFER
                        { m_Resources.instanceBuffer, /*slot*/ 3, /*offset*/ offsetof(InstanceBufferEntry, model) },
                        { m_Resources.instanceBuffer, /*slot*/ 4, /*offset*/ offsetof(InstanceBufferEntry, normal) },
                    #endif
                #endif
                };
                state.indexBuffer = { drawCmd.indexBuffer, nvrhi::Format::R32_UINT, 0 };

                m_CommandList->setGraphicsState(state);
                    
                #if PIPELINER_USE_STRUCTURED_BUFFER
                m_CommandList->setPushConstants(&instanceOffset, sizeof(uint32_t));
                #endif

                m_CommandList->drawIndexed(drawCmd.drawArgs);
            }
        }

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);

        m_VisibleInstanceReferences.clear();
        m_InstanceCounts.assign(m_TreeAssets.size(), {0});
    }

    #pragma region INPUT_HANDLING
    bool KeyboardUpdate(int key, int scancode, int action, int mods) override { 
        m_ViewHandler->camera.KeyboardUpdate(key, scancode, action, mods); 
        return true;
    }

    bool MousePosUpdate(double xpos, double ypos) override { 
        m_ViewHandler->camera.MousePosUpdate(xpos, ypos); 
        return true; 
    }

    bool MouseScrollUpdate(double xoffset, double yoffset) override { 
        m_ViewHandler->camera.MouseScrollUpdate(xoffset, yoffset);
        return true;
    }
    bool MouseButtonUpdate(int button, int action, int mods) override {
        m_ViewHandler->camera.MouseButtonUpdate(button, action, mods);
        return true; 
    }
    bool JoystickButtonUpdate(int button, bool pressed) override {
        m_ViewHandler->camera.JoystickButtonUpdate(button, pressed);
        return true; 
    }
    bool JoystickAxisUpdate(int axis, float value) override {
        m_ViewHandler->camera.JoystickUpdate(axis, value);
        return true; 
    }
    #pragma endregion
};

class PipelinerUIRenderer : public app::ImGui_Renderer
{
private:
    Pipeliner* m_pipeliner;
    UIData& m_ui;
    nvrhi::CommandListHandle m_CommandList;

    // UI state for creating/editing assets
    struct AssetEditState {
        char name[128] = "NewAsset";
        uint32_t generation = 3;
        uint32_t radialSegments = 16;
        float branchAngle = 25.f;
        float taperRatio = 0.9f;
        float stepRatio = 0.95f;
        size_t selectedLSystem = 0;
        bool editing = false;
        size_t editingIndex = 0;
    } m_assetEditState;

    struct RegionEditState {
        char name[128] = "NewRegion";
        uint32_t instanceCount = 20;
        float boundsMinX = 0.f, boundsMinY = 0.f;
        float boundsMaxX = 10.f, boundsMaxY = 10.f;
        std::set<size_t> selectedAssets;
        bool editing = false;
        size_t editingIndex = 0;
    } m_regionEditState;

public:
    PipelinerUIRenderer(app::DeviceManager* deviceManager, Pipeliner* pipeliner, UIData& ui)
        : ImGui_Renderer(deviceManager)
        , m_pipeliner(pipeliner)
        , m_ui(ui)
    {
        m_CommandList = GetDevice()->createCommandList();
        ImGui::GetIO().IniFilename = nullptr;
    }

    void Init(std::shared_ptr<engine::ShaderFactory> shaderFactory)
    {
        ImGui_Renderer::Init(shaderFactory);
    }

    bool KeyboardUpdate(int key, int scancode, int action, int mods) override
    {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        {
            m_ui.ShowUI = !m_ui.ShowUI;
            return true;
        }
        return false;
    }

protected:
    virtual void buildUI() override
    {
        if (!m_ui.ShowUI)
            return;

        int width, height;
        GetDeviceManager()->GetWindowDimensions(width, height);

        float fontSize = ImGui::GetFontSize();
        ImGui::SetNextWindowPos(ImVec2(fontSize * 0.6f, fontSize * 0.6f), 0);
        ImGui::SetNextWindowSize(ImVec2(600, 800), ImGuiCond_FirstUseEver);
        ImGui::Begin("Pipeliner Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        
        ImGui::Text("Renderer: %s", GetDeviceManager()->GetRendererString());
        double frameTime = GetDeviceManager()->GetAverageFrameTimeSeconds();
        if (frameTime > 0.0)
            ImGui::Text("%.3f ms/frame (%.1f FPS)", frameTime * 1e3, 1.0 / frameTime);

        ImGui::Separator();
        
        // ============ TREE ASSETS SECTION ============
        if (ImGui::CollapsingHeader("Tree Assets", ImGuiTreeNodeFlags_DefaultOpen))
        {
            const auto& assets = m_pipeliner->GetTreeAssets();
            ImGui::Text("Total Assets: %zu", assets.size());
            
            ImGui::Separator();
            ImGui::Text("Asset Editor");
            ImGui::Separator();
            
            ImGui::InputText("Asset Name##assetname", m_assetEditState.name, sizeof(m_assetEditState.name));
            
            // L-System selector
            const auto& lsystems = m_pipeliner->GetLSystems();
            std::vector<std::string> lsysNames;
            for (const auto& [name, sys] : lsystems) {
                lsysNames.push_back(name);
            }
            
            if (!lsysNames.empty()) {
                static int selectedLSys = 0;
                if (ImGui::BeginCombo("L-System", lsysNames[selectedLSys].c_str())) {
                    for (size_t i = 0; i < lsysNames.size(); i++) {
                        bool isSelected = (i == (size_t)selectedLSys);
                        if (ImGui::Selectable(lsysNames[i].c_str(), isSelected)) {
                            selectedLSys = (int)i;
                            m_assetEditState.selectedLSystem = i;
                        }
                    }
                    ImGui::EndCombo();
                }
            }
            
            ImGui::SliderScalar("Generation", ImGuiDataType_U32, &m_assetEditState.generation, 
                new uint32_t(1), new uint32_t(10));
            ImGui::SliderScalar("Radial Segments", ImGuiDataType_U32, &m_assetEditState.radialSegments,
                new uint32_t(3), new uint32_t(32));
            ImGui::SliderFloat("Branch Angle (deg)", &m_assetEditState.branchAngle, 5.f, 90.f);
            ImGui::SliderFloat("Taper Ratio", &m_assetEditState.taperRatio, 0.5f, 1.0f);
            ImGui::SliderFloat("Step Ratio", &m_assetEditState.stepRatio, 0.5f, 1.0f);
            
            ImGui::Spacing();
            if (m_assetEditState.editing) {
                if (ImGui::Button("Update Asset##confirm_edit")) {
                    // TODO: Implement asset editing with parameter updates
                    m_assetEditState.editing = false;
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel##cancel_edit")) {
                    m_assetEditState.editing = false;
                }
            } else {
                if (ImGui::Button("Add New Asset##add_asset")) {
                    TreeGenerator::Params params;
                    params.radialSegments = m_assetEditState.radialSegments;
                    params.branchAngle = dm::radians(m_assetEditState.branchAngle);
                    params.taperRatio = m_assetEditState.taperRatio;
                    params.stepRatio = m_assetEditState.stepRatio;
                    
                    m_pipeliner->AddTreeAsset(std::string(m_assetEditState.name), 
                        m_assetEditState.selectedLSystem,
                        m_assetEditState.generation, params);
                    
                    memset(m_assetEditState.name, 0, sizeof(m_assetEditState.name));
                    strcpy_s(m_assetEditState.name, "NewAsset");
                    m_assetEditState.generation = 3;
                    m_assetEditState.radialSegments = 16;
                    m_assetEditState.branchAngle = 25.f;
                    m_assetEditState.taperRatio = 0.9f;
                    m_assetEditState.stepRatio = 0.95f;
                }
            }
            
            ImGui::Separator();
            ImGui::Text("Existing Assets:");
            
            for (size_t i = 0; i < assets.size(); ++i)
            {
                const auto& asset = assets[i];
                ImGui::PushID((int)i);
                
                ImGui::Text("#%zu: %s", i, asset.name.c_str());
                ImGui::SameLine();
                ImGui::Text("Gen: %u, Segs: %u", asset.config.generation, asset.config.params.radialSegments);
                
                if (ImGui::CollapsingHeader("Details##asset", ImGuiTreeNodeFlags_SpanAvailWidth))
                {
                    ImGui::Text("Branch Angle: %.2f°", dm::degrees(asset.config.params.branchAngle));
                    ImGui::Text("Taper Ratio: %.2f", asset.config.params.taperRatio);
                    ImGui::Text("Step Ratio: %.2f", asset.config.params.stepRatio);
                    ImGui::Text("L-System String Length: %zu", asset.lsystemString.size());
                }
                
                if (ImGui::Button("Edit##edit_asset")) {
                    m_assetEditState.editing = true;
                    m_assetEditState.editingIndex = i;
                    memset(m_assetEditState.name, 0, sizeof(m_assetEditState.name));
                    strcpy_s(m_assetEditState.name, asset.name.c_str());
                    m_assetEditState.generation = asset.config.generation;
                    m_assetEditState.radialSegments = asset.config.params.radialSegments;
                    m_assetEditState.branchAngle = dm::degrees(asset.config.params.branchAngle);
                    m_assetEditState.taperRatio = asset.config.params.taperRatio;
                    m_assetEditState.stepRatio = asset.config.params.stepRatio;
                }
                ImGui::SameLine();
                if (ImGui::Button("Remove##remove_asset")) {
                    m_pipeliner->RemoveTreeAsset(i);
                }
                
                ImGui::PopID();
                ImGui::Separator();
            }
        }
        
        // ============ REGIONS SECTION ============
        if (ImGui::CollapsingHeader("Regions", ImGuiTreeNodeFlags_DefaultOpen))
        {
            const auto& regions = m_pipeliner->GetRegions();
            const auto& assets = m_pipeliner->GetTreeAssets();
            ImGui::Text("Total Regions: %zu", regions.size());
            
            ImGui::Separator();
            ImGui::Text("Region Editor");
            ImGui::Separator();
            
            ImGui::InputText("Region Name##regionname", m_regionEditState.name, sizeof(m_regionEditState.name));
            ImGui::SliderScalar("Instance Count", ImGuiDataType_U32, &m_regionEditState.instanceCount,
                new uint32_t(1), new uint32_t(1000));
            
            ImGui::InputFloat("Bounds Min X##minx", &m_regionEditState.boundsMinX);
            ImGui::InputFloat("Bounds Min Y##miny", &m_regionEditState.boundsMinY);
            ImGui::InputFloat("Bounds Max X##maxx", &m_regionEditState.boundsMaxX);
            ImGui::InputFloat("Bounds Max Y##maxy", &m_regionEditState.boundsMaxY);
            
            ImGui::Text("Select Assets for Region:");
            for (size_t i = 0; i < assets.size(); ++i) {
                bool selected = m_regionEditState.selectedAssets.count(i) > 0;
                if (ImGui::Checkbox(("Asset " + std::to_string(i) + ": " + assets[i].name).c_str(), &selected)) {
                    if (selected) {
                        m_regionEditState.selectedAssets.insert(i);
                    } else {
                        m_regionEditState.selectedAssets.erase(i);
                    }
                }
            }
            
            ImGui::Spacing();
            if (m_regionEditState.editing) {
                if (ImGui::Button("Update Region##confirm_region")) {
                    m_regionEditState.editing = false;
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel##cancel_region")) {
                    m_regionEditState.editing = false;
                }
            } else {
                if (ImGui::Button("Add New Region##add_region")) {
                    dm::box2 bounds(
                        dm::float2(m_regionEditState.boundsMinX, m_regionEditState.boundsMinY),
                        dm::float2(m_regionEditState.boundsMaxX, m_regionEditState.boundsMaxY)
                    );
                    m_pipeliner->AddRegion(std::string(m_regionEditState.name), m_regionEditState.instanceCount, bounds);
                    
                    memset(m_regionEditState.name, 0, sizeof(m_regionEditState.name));
                    strcpy_s(m_regionEditState.name, "NewRegion");
                    m_regionEditState.instanceCount = 20;
                    m_regionEditState.boundsMinX = 0.f;
                    m_regionEditState.boundsMinY = 0.f;
                    m_regionEditState.boundsMaxX = 10.f;
                    m_regionEditState.boundsMaxY = 10.f;
                    m_regionEditState.selectedAssets.clear();
                }
            }
            
            ImGui::Separator();
            ImGui::Text("Existing Regions:");
            
            for (size_t i = 0; i < regions.size(); ++i)
            {
                const auto& region = regions[i];
                ImGui::PushID((int)(i + 1000));
                
                ImGui::Text("#%zu: %s", i, region.name.c_str());
                ImGui::SameLine();
                ImGui::Text("Instances: %u, Assets: %zu", region.instanceCount, region.assetIndices.size());
                
                if (ImGui::CollapsingHeader("Bounds##region", ImGuiTreeNodeFlags_SpanAvailWidth))
                {
                    const auto& bounds = region.bounds;
                    ImGui::Text("Min: (%.2f, %.2f)", bounds.m_mins.x, bounds.m_mins.y);
                    ImGui::Text("Max: (%.2f, %.2f)", bounds.m_maxs.x, bounds.m_maxs.y);
                    ImGui::Text("Diagonal: (%.2f, %.2f)", bounds.diagonal().x, bounds.diagonal().y);
                    ImGui::Text("Used Assets: ");
                    ImGui::SameLine();
                    for (size_t idx : region.assetIndices) {
                        if (idx < assets.size()) {
                            ImGui::Text("%s ", assets[idx].name.c_str());
                            ImGui::SameLine();
                        }
                    }
                }
                
                if (ImGui::Button("Edit##edit_region")) {
                    m_regionEditState.editing = true;
                    m_regionEditState.editingIndex = i;
                    memset(m_regionEditState.name, 0, sizeof(m_regionEditState.name));
                    strcpy_s(m_regionEditState.name, region.name.c_str());
                    m_regionEditState.instanceCount = region.instanceCount;
                    m_regionEditState.boundsMinX = region.bounds.m_mins.x;
                    m_regionEditState.boundsMinY = region.bounds.m_mins.y;
                    m_regionEditState.boundsMaxX = region.bounds.m_maxs.x;
                    m_regionEditState.boundsMaxY = region.bounds.m_maxs.y;
                    m_regionEditState.selectedAssets.clear();
                    for (size_t idx : region.assetIndices) {
                        m_regionEditState.selectedAssets.insert(idx);
                    }
                }
                ImGui::SameLine();
                if (ImGui::Button("Remove##remove_region")) {
                    m_pipeliner->GetRegions().erase(m_pipeliner->GetRegions().begin() + i);
                    m_pipeliner->RegenerateRegions();
                }
                
                ImGui::PopID();
                ImGui::Separator();
            }
        }
        
        // ============ L-SYSTEMS SECTION ============
        if (ImGui::CollapsingHeader("L-Systems", ImGuiTreeNodeFlags_DefaultOpen))
        {
            const auto& lsystems = m_pipeliner->GetLSystems();
            ImGui::Text("Total L-Systems: %zu", lsystems.size());
            for (const auto& [name, sys] : lsystems) {
                ImGui::BulletText("%s", name.c_str());
            }
            ImGui::Text("To add/modify L-systems, edit _InitTreeData() method");
        }
        
        ImGui::Separator();
        if (ImGui::Button("Close UI (ESC)"))
            m_ui.ShowUI = false;

        ImGui::End();
    }
};

#ifdef WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    nvrhi::GraphicsAPI api = app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    app::DeviceManager* deviceManager = app::DeviceManager::Create(api);

    app::DeviceCreationParameters deviceParams;
#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true; 
    deviceParams.enableNvrhiValidationLayer = true;
#endif

    deviceParams.depthBufferFormat = nvrhi::Format::D16;

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }

    log::info("Physical Device: %s", deviceManager->GetRendererString());

    {
        UIData uiData;
        Pipeliner pass(deviceManager);
        
        if (pass.Init())
        {
            // Create shader factory for UI renderer
            std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable().parent_path() / "shaders/framework" / app::GetShaderTypeName(deviceManager->GetDevice()->getGraphicsAPI());
            std::shared_ptr<vfs::RootFileSystem> rootFS = std::make_shared<vfs::RootFileSystem>();
            rootFS->mount("/shaders/donut", frameworkShaderPath);
            std::shared_ptr<engine::ShaderFactory> shaderFactory = std::make_shared<engine::ShaderFactory>(deviceManager->GetDevice(), rootFS, "/shaders");

            PipelinerUIRenderer uiPass(deviceManager, &pass, uiData);
            uiPass.Init(shaderFactory);

            deviceManager->AddRenderPassToBack(&pass);
            deviceManager->AddRenderPassToBack(&uiPass);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&uiPass);
            deviceManager->RemoveRenderPass(&pass);
        }
    }

    deviceManager->Shutdown();

    delete deviceManager;

    return 0;
}
