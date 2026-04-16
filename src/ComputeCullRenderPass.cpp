#include "include/ComputeCullRenderPass.hpp"
#include "include/Globals.hpp"
#include "include/macros.h"
#include "include/Terrain.hpp"

#include <GFSDK_Aftermath_GpuCrashDump.h>

#include <nvrhi/utils.h>
#include <donut/app/imgui_renderer.h>
#include <donut/app/Timer.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/core/math/math.h>
#include <donut/core/json.h>

using namespace donut::math;
#include <donut/shaders/sky_cb.h>

#include <filesystem>

using namespace Xylem;

// ===========================================================================
// GPU asset upload helpers (identical to P0)
// ===========================================================================

void ComputeCullRenderPass::_UploadAsset(
    const TreeAssetDef& assetDef,
    GPUTreeAsset& gpuAsset,
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList)
{
    gpuAsset.textureSetIdx = assetDef.textureSetIdx;
    gpuAsset.lods.resize(assetDef.lods.size());

    nvrhi::BufferDesc vDesc;
    vDesc.isVertexBuffer = true;
    vDesc.initialState   = nvrhi::ResourceStates::CopyDest;

    nvrhi::BufferDesc iDesc;
    iDesc.isIndexBuffer = true;
    iDesc.initialState  = nvrhi::ResourceStates::CopyDest;

    for (size_t j = 0; j < assetDef.lods.size(); j++) {
        const auto& lodDef = assetDef.lods[j];

        vDesc.debugName = "VB_" + assetDef.name + "_LOD" + std::to_string(j);
        vDesc.byteSize  = lodDef.vertices.size() * sizeof(ProcGen::TreeVertex);
        auto vBuf = device->createBuffer(vDesc);
        commandList->beginTrackingBufferState(vBuf, nvrhi::ResourceStates::CopyDest);
        commandList->writeBuffer(vBuf, lodDef.vertices.data(), vDesc.byteSize);
        commandList->setPermanentBufferState(vBuf, nvrhi::ResourceStates::VertexBuffer);

        iDesc.debugName = "IB_" + assetDef.name + "_LOD" + std::to_string(j);
        iDesc.byteSize  = lodDef.indices.size() * sizeof(uint32_t);
        auto iBuf = device->createBuffer(iDesc);
        commandList->beginTrackingBufferState(iBuf, nvrhi::ResourceStates::CopyDest);
        commandList->writeBuffer(iBuf, lodDef.indices.data(), iDesc.byteSize);
        commandList->setPermanentBufferState(iBuf, nvrhi::ResourceStates::IndexBuffer);

        gpuAsset.lods[j].vertexBuffer   = vBuf;
        gpuAsset.lods[j].indexBuffer    = iBuf;
        gpuAsset.lods[j].indexCount     = static_cast<uint32_t>(lodDef.indices.size());
        gpuAsset.lods[j].radialSegments = lodDef.radialSegments;
        gpuAsset.lods[j].bbox           = lodDef.bbox;
    }
}

void ComputeCullRenderPass::_UploadAllAssets(nvrhi::IDevice* device, nvrhi::ICommandList* commandList) {
    const auto& assets = m_Registry.getAssets();
    m_GPUAssets.resize(assets.size());
    m_AssetIdToGPUIndex.clear();

    for (size_t i = 0; i < assets.size(); i++) {
        _UploadAsset(assets[i], m_GPUAssets[i], device, commandList);
        m_AssetIdToGPUIndex[assets[i].id] = i;
    }
}

// ===========================================================================
// Region windows — gapped persistent buffer layout
// ===========================================================================

void ComputeCullRenderPass::_BuildRegionWindows() {
    const auto& regions = m_Registry.getRegions();
    m_RegionWindows.resize(regions.size());
    m_TotalCapacity = 0;

    for (size_t i = 0; i < regions.size(); i++) {
        auto& win = m_RegionWindows[i];
        uint32_t count = static_cast<uint32_t>(regions[i].instances.size());
        win.count    = count;
        win.capacity = std::max(16u, static_cast<uint32_t>(std::ceil(count * k_CapacitySlack)));
        win.offset   = m_TotalCapacity;
        m_TotalCapacity += win.capacity;
    }

    m_TotalCapacity = std::max(1u, m_TotalCapacity);

    // Build CPU-side staging arrays (gapped layout)
    const uint32_t numLods = static_cast<uint32_t>(m_Registry.getLodSegments().size());

    m_InstanceStaging.assign(m_TotalCapacity, Render::InstanceBufferEntry{});
    m_CullDataStaging.assign(m_TotalCapacity, Render::CullInstanceData{});
    m_RegionStaging.assign(regions.size(), Render::CullRegionData{});

    for (size_t r = 0; r < regions.size(); r++) {
        const auto& win = m_RegionWindows[r];
        const auto& reg = regions[r];
        m_RegionStaging[r] = {reg.cullBox};

        for (uint32_t i = 0; i < reg.instances.size(); i++) {
            const auto& inst = reg.instances[i];
            auto it = m_AssetIdToGPUIndex.find(inst.assetId);
            if (it == m_AssetIdToGPUIndex.end()) continue;
            uint32_t gpuIdx = static_cast<uint32_t>(it->second);
            uint32_t idx = win.offset + i;

            m_InstanceStaging[idx] = Render::InstanceBufferEntry(
                inst.model, inst.normal, gpuIdx);

            // World-space AABB from LOD[0] bbox (largest, conservative)
            const dm::box3& localBbox = m_GPUAssets[gpuIdx].lods[0].bbox;
            dm::box3 worldBbox = localBbox * dm::homogeneousToAffine(inst.model);

            auto& cd = m_CullDataStaging[idx];
            cd.bbox     = worldBbox;
            cd.baseSlot = gpuIdx * numLods;
            cd.regionId = static_cast<uint32_t>(r);
            cd.active   = 1;
        }
        // Dead slots remain zero-initialized (active=0)
    }
}

// ===========================================================================
// Slot layout — compute slot offsets and visibility buffer sizing
// ===========================================================================

void ComputeCullRenderPass::_BuildSlotLayout() {
    const uint32_t numLods   = static_cast<uint32_t>(m_Registry.getLodSegments().size());
    const uint32_t numAssets = static_cast<uint32_t>(m_GPUAssets.size());

    m_NumSlots = numAssets * numLods;

    // Count live instances per asset
    std::vector<uint32_t> livePerAsset(numAssets, 0);
    for (const auto& win : m_RegionWindows) {
        // Count from the staging data which instances are active per asset
        for (uint32_t i = 0; i < win.count; i++) {
            uint32_t idx = win.offset + i;
            uint32_t treeId = m_InstanceStaging[idx].treeId;
            livePerAsset[treeId]++;
        }
    }

    // Per-slot max count = all live instances of that asset (worst case: all in one LOD)
    m_MaxSlotCounts.resize(m_NumSlots);
    m_SlotOffsets.resize(m_NumSlots);
    m_VisBufferSize = 0;

    for (uint32_t ai = 0; ai < numAssets; ai++) {
        for (uint32_t lodi = 0; lodi < numLods; lodi++) {
            uint32_t slotIdx = ai * numLods + lodi;
            
            m_MaxSlotCounts[slotIdx] = livePerAsset[ai];
            m_SlotOffsets[slotIdx]   = m_VisBufferSize;

            m_VisBufferSize += livePerAsset[ai];
        }
    }

    m_VisBufferSize = std::max(1u, m_VisBufferSize);

    // Shadow slot layout — one slot per asset (no LOD axis)
    m_ShadowSlotOffsets.resize(numAssets);
    m_ShadowVisBufferSize = 0;
    for (uint32_t ai = 0; ai < numAssets; ai++) {
        m_ShadowSlotOffsets[ai] = m_ShadowVisBufferSize;
        m_ShadowVisBufferSize  += livePerAsset[ai];
    }
    m_ShadowVisBufferSize = std::max(1u, m_ShadowVisBufferSize);

    // Build indirect args staging — pre-fill with indexCount, instanceCount=0
    m_IndirectArgsStaging.resize(std::max(1u, m_NumSlots));
    for (uint32_t ai = 0; ai < numAssets; ai++) {
        for (uint32_t lodi = 0; lodi < numLods; lodi++) {
            uint32_t slotIdx = ai * numLods + lodi;
            auto& args = m_IndirectArgsStaging[slotIdx];
            args.indexCount            = m_GPUAssets[ai].lods[lodi].indexCount;
            args.instanceCount         = 0;  // CS will atomically increment
            args.startIndexLocation    = 0;
            args.baseVertexLocation    = 0;
            args.startInstanceLocation = 0;
        }
    }

    // Shadow indirect args — one per asset, using lowest LOD
    const uint32_t lowestLOD = numLods > 0 ? numLods - 1 : 0;
    m_ShadowIndirectArgsStaging.resize(std::max(1u, numAssets));
    for (uint32_t ai = 0; ai < numAssets; ai++) {
        auto& args = m_ShadowIndirectArgsStaging[ai];
        args.indexCount            = m_GPUAssets[ai].lods[lowestLOD].indexCount;
        args.instanceCount         = 0;
        args.startIndexLocation    = 0;
        args.baseVertexLocation    = 0;
        args.startInstanceLocation = 0;
    }
}

// ===========================================================================
// Upload cull buffers to GPU
// ===========================================================================

void ComputeCullRenderPass::_UploadCullBuffers(nvrhi::ICommandList* commandList) {
    auto device = GetDevice();
    const uint32_t numRegions = static_cast<uint32_t>(m_Registry.getRegions().size());

    // PersistentInstanceBuffer — SRV structured buffer (permanent ShaderResource after upload)
    m_CullPass.persistentInstBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(m_TotalCapacity * sizeof(Render::InstanceBufferEntry))
            .setStructStride(sizeof(Render::InstanceBufferEntry))
            .setDebugName("PersistentInstanceBuffer")
            .setCanHaveUAVs(false)
            .setInitialState(nvrhi::ResourceStates::CopyDest)
    );
    commandList->beginTrackingBufferState(m_CullPass.persistentInstBuffer, nvrhi::ResourceStates::CopyDest);
    commandList->writeBuffer(m_CullPass.persistentInstBuffer,
        m_InstanceStaging.data(),
        m_TotalCapacity * sizeof(Render::InstanceBufferEntry));
    commandList->setPermanentBufferState(m_CullPass.persistentInstBuffer, nvrhi::ResourceStates::ShaderResource);

    // CullDataBuffer — SRV structured buffer (permanent ShaderResource after upload)
    m_CullPass.cullDataBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(m_TotalCapacity * sizeof(Render::CullInstanceData))
            .setStructStride(sizeof(Render::CullInstanceData))
            .setDebugName("CullDataBuffer")
            .setCanHaveUAVs(false)
            .setInitialState(nvrhi::ResourceStates::CopyDest)
    );
    commandList->beginTrackingBufferState(m_CullPass.cullDataBuffer, nvrhi::ResourceStates::CopyDest);
    commandList->writeBuffer(m_CullPass.cullDataBuffer,
        m_CullDataStaging.data(),
        m_TotalCapacity * sizeof(Render::CullInstanceData));
    commandList->setPermanentBufferState(m_CullPass.cullDataBuffer, nvrhi::ResourceStates::ShaderResource);

    // CullRegionDataBuffer — SRV structured buffer (permanent ShaderResource after upload)
    m_CullPass.cullRegionDataBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(numRegions * sizeof(Render::CullRegionData))
            .setStructStride(sizeof(Render::CullRegionData))
            .setDebugName("CullRegionDataBuffer")
            .setCanHaveUAVs(false)
            .setInitialState(nvrhi::ResourceStates::CopyDest)
    );
    commandList->beginTrackingBufferState(m_CullPass.cullRegionDataBuffer, nvrhi::ResourceStates::CopyDest);
    commandList->writeBuffer(m_CullPass.cullRegionDataBuffer,
        m_RegionStaging.data(),
        numRegions * sizeof(Render::CullRegionData));
    commandList->setPermanentBufferState(m_CullPass.cullRegionDataBuffer, nvrhi::ResourceStates::ShaderResource);

    // SlotOffsetBuffer — SRV structured buffer (permanent ShaderResource after upload)
    m_CullPass.slotOffsetBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(std::max(1u, m_NumSlots) * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("SlotOffsetBuffer")
            .setCanHaveUAVs(false)
            .setInitialState(nvrhi::ResourceStates::CopyDest)
    );
    commandList->beginTrackingBufferState(m_CullPass.slotOffsetBuffer, nvrhi::ResourceStates::CopyDest);
    if (m_NumSlots > 0)
        commandList->writeBuffer(m_CullPass.slotOffsetBuffer,
            m_SlotOffsets.data(), m_NumSlots * sizeof(uint32_t));
    commandList->setPermanentBufferState(m_CullPass.slotOffsetBuffer, nvrhi::ResourceStates::ShaderResource);

    // CountBuffer — UAV structured buffer (cleared each frame, read as SRV by draw passes)
    m_CullPass.countBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(std::max(1u, m_NumSlots) * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("CountBuffer")
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    // VisibilityBuffer — UAV structured buffer (written by CS, read as SRV by draw passes)
    m_CullPass.visibilityBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(m_VisBufferSize * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("VisibilityBuffer")
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    const uint32_t numAssets = static_cast<uint32_t>(m_GPUAssets.size());

    // ShadowCountBuffer — UAV uint32[numAssets], one counter per asset
    m_CullPass.shadowCountBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(std::max(1u, numAssets) * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("ShadowCountBuffer")
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    // ShadowVisBuffer — UAV uint32[shadowVisBufferSize], per-asset windows
    m_CullPass.shadowVisBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(m_ShadowVisBufferSize * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("ShadowVisBuffer")
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    // ShadowSlotOffsetBuffer — SRV uint32[numAssets], prefix sums into ShadowVisBuffer
    m_CullPass.shadowSlotOffsetBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(std::max(1u, numAssets) * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("ShadowSlotOffsetBuffer")
            .setCanHaveUAVs(false)
            .setInitialState(nvrhi::ResourceStates::CopyDest)
    );
    commandList->beginTrackingBufferState(m_CullPass.shadowSlotOffsetBuffer, nvrhi::ResourceStates::CopyDest);
    if (numAssets > 0)
        commandList->writeBuffer(m_CullPass.shadowSlotOffsetBuffer,
            m_ShadowSlotOffsets.data(), numAssets * sizeof(uint32_t));
    commandList->setPermanentBufferState(m_CullPass.shadowSlotOffsetBuffer, nvrhi::ResourceStates::ShaderResource);

    // IndirectArgsBuffer — UAV + indirect draw args, DrawIndexedIndirectArguments[numSlots]
    // Pre-filled with indexCount per slot, instanceCount=0 (CS atomically increments each frame).
    uint32_t indirectArgsBufSize = std::max(1u, m_NumSlots) * sizeof(nvrhi::DrawIndexedIndirectArguments);
    m_CullPass.indirectArgsBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(indirectArgsBufSize)
            .setDebugName("IndirectArgsBuffer")
            .setIsDrawIndirectArgs(true)
            .setCanHaveUAVs(true)
            .setCanHaveRawViews(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    // ShadowIndirectArgsBuffer — UAV + indirect draw args, DrawIndexedIndirectArguments[numAssets]
    uint32_t shadowIndirectArgsBufSize = std::max(1u, numAssets) * sizeof(nvrhi::DrawIndexedIndirectArguments);
    m_CullPass.shadowIndirectArgsBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(shadowIndirectArgsBufSize)
            .setDebugName("ShadowIndirectArgsBuffer")
            .setIsDrawIndirectArgs(true)
            .setCanHaveUAVs(true)
            .setCanHaveRawViews(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    // RegionVisibleBuffer — SRV uint32[numRegions], CPU-written each frame via writeBuffer
    m_RegionVisibleStaging.assign(std::max(1u, numRegions), 1u);
    m_CullPass.regionVisibleBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(std::max(1u, numRegions) * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("RegionVisibleBuffer")
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource)
    );

    // Readback ring buffers — combined [countBuffer | shadowCountBuffer] per slot
    m_ReadbackCountEntries  = std::max(1u, m_NumSlots);
    m_ReadbackShadowEntries = std::max(1u, numAssets);
    uint64_t readbackSize = (m_ReadbackCountEntries + m_ReadbackShadowEntries) * sizeof(uint32_t);

    for (uint32_t i = 0; i < k_QueuedFrames; i++) {
        m_ReadbackBuffers[i] = device->createBuffer(
            nvrhi::BufferDesc()
                .setByteSize(readbackSize)
                .setCpuAccess(nvrhi::CpuAccessMode::Read)
                .setInitialState(nvrhi::ResourceStates::CopyDest)
                .setKeepInitialState(true)
                .setDebugName("CullCountReadback_" + std::to_string(i))
        );
    }
    m_ReadbackFrameIndex = 0;
}

// ===========================================================================
// Rebuild cull binding sets
// ===========================================================================

void ComputeCullRenderPass::_RebuildCullBindings() {
    auto device = GetDevice();

    // Cull compute binding set
    nvrhi::BindingSetDesc cullBSD;
    cullBSD.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_Shared.constantBuffer,
            nvrhi::BufferRange(0, Render::c_CullConstantBufferSize)),

        nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_CullPass.cullRegionDataBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_CullPass.cullDataBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_CullPass.slotOffsetBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_CullPass.shadowSlotOffsetBuffer),

        nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_CullPass.regionVisibleBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(1, m_CullPass.countBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(2, m_CullPass.visibilityBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(3, m_CullPass.shadowCountBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(4, m_CullPass.shadowVisBuffer),

        nvrhi::BindingSetItem::RawBuffer_UAV(5, m_CullPass.indirectArgsBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(6, m_CullPass.shadowIndirectArgsBuffer),
    };

    m_CullPass.bindingSet = device->createBindingSet(cullBSD, m_CullPass.bindingLayout);

    // Tree pass binding sets — visibility/instance/slotOffset SRVs + textures
    m_TreePass.bindingSets.resize(m_TreePass.textureSets.size());
    for (size_t i = 0; i < m_TreePass.textureSets.size(); i++) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_Shared.constantBuffer,
                nvrhi::BufferRange(0, Render::c_CullConstantBufferSize)),
            nvrhi::BindingSetItem::PushConstants(1, sizeof(uint32_t)),

            nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_CullPass.visibilityBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_CullPass.persistentInstBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_CullPass.slotOffsetBuffer),

            nvrhi::BindingSetItem::Texture_SRV(3, m_TreePass.textureSets[i].diffuse),
            nvrhi::BindingSetItem::Texture_SRV(4, m_TreePass.textureSets[i].normalMap),
            nvrhi::BindingSetItem::Texture_SRV(5, m_ShadowPass.depthTexture),

            nvrhi::BindingSetItem::Sampler(0, m_TreePass.sampler),
            nvrhi::BindingSetItem::Sampler(1, m_ShadowPass.comparisonSampler),
        };
        m_TreePass.bindingSets[i] = device->createBindingSet(bsd, m_TreePass.bindingLayout);
    }

    // Shadow pass binding set
    nvrhi::BindingSetDesc shadowBSD;
    shadowBSD.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_Shared.constantBuffer,
            nvrhi::BufferRange(0, Render::c_CullConstantBufferSize)),
        nvrhi::BindingSetItem::PushConstants(1, sizeof(uint32_t)),

        nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_CullPass.shadowVisBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_CullPass.persistentInstBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_CullPass.shadowSlotOffsetBuffer),
    };
    m_ShadowPass.bindingSet = device->createBindingSet(shadowBSD, m_ShadowPass.bindingLayout);
}

// ===========================================================================
// Destructor
// ===========================================================================

ComputeCullRenderPass::~ComputeCullRenderPass() {
    if (m_AftermathContext) {
        GFSDK_Aftermath_ReleaseContextHandle(m_AftermathContext);
        m_AftermathContext = nullptr;
    }
}

// ===========================================================================
// Init
// ===========================================================================

bool ComputeCullRenderPass::Init() {
    engine::CommonRenderPasses commonPasses(GetDevice(), m_ShaderFactory);

    {
        nvrhi::CommandListHandle initCL = GetDevice()->createCommandList();
        initCL->open();

        _UploadAllAssets(GetDevice(), initCL);

        if (!_InitShared())                                return false;
        if (!_InitShadowPass())                            return false;
        if (!_InitTreePass(initCL, commonPasses))          return false;
        if (!_InitTerrainPass(initCL))                     return false;
        if (!_InitSkyPass())                               return false;

        _BuildRegionWindows();
        _BuildSlotLayout();

        if (!_InitCullPass(initCL))                        return false;

        _UploadCullBuffers(initCL);
        _RebuildCullBindings();

        initCL->close();
        GetDevice()->executeCommandList(initCL);
    }

    m_CommandList = GetDevice()->createCommandList();
    // _InitTimerQueries();

    // Aftermath: create context handle from the D3D12 device.
    {
        auto* d3d12Device = static_cast<ID3D12Device*>(
            GetDevice()->getNativeObject(nvrhi::ObjectTypes::D3D12_Device));
        GFSDK_Aftermath_DX12_CreateContextHandle(d3d12Device, &m_AftermathContext);
    }

    m_UI.totalInstanceCount = m_Registry.totalInstanceCount();

    return true;
}

// ===========================================================================
// Animate
// ===========================================================================

void ComputeCullRenderPass::Animate(float seconds) {
    GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);

    if (!m_Registry.anyDirty()) return;

    auto dirtyAssets  = m_Registry.getDirtyAssetIndices();
    auto dirtyRegions = m_Registry.getDirtyRegionIndices();

    m_Registry.rebuildDirtyAssets();
    m_Registry.rebuildDirtyRegions();

    if (!dirtyAssets.empty())  onAssetsDirty(dirtyAssets);
    if (!dirtyRegions.empty()) onRegionsDirty(dirtyRegions);

    m_Registry.clearDirtyFlags();
}

void ComputeCullRenderPass::BackBufferResizing() {
    m_TreePass.pipeline          = nullptr;
    m_TerrainPass.pipeline       = nullptr;
    m_ShadowPass.treePipeline    = nullptr;
    m_ShadowPass.terrainPipeline = nullptr;
    m_SkyPass.pipeline           = nullptr;
}

// ===========================================================================
// Hot-reload callbacks
// ===========================================================================

void ComputeCullRenderPass::onAssetsDirty(const std::vector<size_t>& dirtyAssetIndices) {
    nvrhi::CommandListHandle cl = GetDevice()->createCommandList();
    cl->open();

    const auto& assets = m_Registry.getAssets();

    // If the GPU asset count doesn't match the registry (asset was removed),
    // re-upload everything from scratch so phantom entries are cleared.
    if (m_GPUAssets.size() != assets.size()) {
        _UploadAllAssets(GetDevice(), cl);
    } else {
        for (size_t idx : dirtyAssetIndices) {
            const auto& assetDef = assets[idx];
            auto it = m_AssetIdToGPUIndex.find(assetDef.id);
            if (it != m_AssetIdToGPUIndex.end()) {
                _UploadAsset(assetDef, m_GPUAssets[it->second], GetDevice(), cl);
            } else {
                GPUTreeAsset gpuAsset;
                _UploadAsset(assetDef, gpuAsset, GetDevice(), cl);
                m_AssetIdToGPUIndex[assetDef.id] = m_GPUAssets.size();
                m_GPUAssets.push_back(std::move(gpuAsset));
            }
        }
    }

    // Re-upload CullDataBuffer bboxes (asset geometry changed → bboxes changed)
    _BuildRegionWindows();
    _BuildSlotLayout();
    _UploadCullBuffers(cl);
    _RebuildCullBindings();

    cl->close();
    GetDevice()->executeCommandList(cl);
}

void ComputeCullRenderPass::onRegionsDirty(const std::vector<size_t>& /*dirtyRegionIndices*/) {
    nvrhi::CommandListHandle cl = GetDevice()->createCommandList();
    cl->open();

    _BuildRegionWindows();
    _BuildSlotLayout();
    _UploadCullBuffers(cl);
    _RebuildCullBindings();

    cl->close();
    GetDevice()->executeCommandList(cl);

    m_UI.totalInstanceCount = m_Registry.totalInstanceCount();
}

// ===========================================================================
// Render
// ===========================================================================

void ComputeCullRenderPass::Render(nvrhi::IFramebuffer* framebuffer) {
    m_UI.shadowMapTexture = m_ShadowPass.depthTexture.Get();

    app::HiResTimer cpuTimer;
    cpuTimer.Start();

    const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();

    if (!m_TreePass.pipeline) {
        m_ViewHandler.view.SetViewport({ float(fbinfo.width), float(fbinfo.height) });
        m_ViewHandler.view.SetProjectionMatrix(
        #if XYLEM_USE_REVERSE_Z
            dm::perspProjD3DStyleReverse(
                dm::radians(60.f),
                float(fbinfo.width)/float(fbinfo.height),
                0.1f
            )
        #else
            dm::perspProjD3DStyle(
                dm::radians(60.f),
                float(fbinfo.width)/float(fbinfo.height),
                0.1f,
                1000.0f
            )
        #endif
        );
    }

    m_CommandList->open();
    // m_CommandList->beginTimerQuery(m_GpuTimers[m_NextTimerIdx]);

    GFSDK_Aftermath_SetEventMarker(m_AftermathContext, "Frame::Begin", 0);

    m_ViewHandler.view.SetViewMatrix(m_ViewHandler.camera.GetWorldToViewMatrix());
    m_ViewHandler.view.UpdateCache();

    nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));
    #if XYLEM_USE_REVERSE_Z
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer, 0.f, 0);
    #else
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer,
        m_ViewHandler.view.GetViewFrustum().farPlane().distance, 0);
    #endif

    dm::box3 sceneBbox = dm::box3::empty();
    for (const auto& region : m_Registry.getRegions())
        sceneBbox |= region.cullBox;
    const auto* terrain = m_Registry.getTerrain();
    if (terrain)
        sceneBbox |= terrain->getBbox();

    m_ViewHandler.updateShadowVolume(sceneBbox, m_Registry.getSunDirection());

    const dm::box3& shadowCasterBboxLS = m_ViewHandler.shadowCasterBboxLS;
    float maxXY = dm::max(shadowCasterBboxLS.diagonal().x, shadowCasterBboxLS.diagonal().y);
    dm::float3 grow = 0.5f * (dm::float3(maxXY, maxXY, shadowCasterBboxLS.diagonal().z)
                               - shadowCasterBboxLS.diagonal());
    dm::box3 orthoBox = shadowCasterBboxLS.grow(grow);

    dm::float4x4 lightProj = dm::orthoProjD3DStyle(
        orthoBox.m_mins.x, orthoBox.m_maxs.x,
        orthoBox.m_mins.y, orthoBox.m_maxs.y,
        orthoBox.m_mins.z, orthoBox.m_maxs.z);

    dm::float4x4 lightViewProj = dm::affineToHomogeneous(m_ViewHandler.worldToLight) * lightProj;

    // --- Fill CullConstantBufferEntry ---
    Render::CullConstantBufferEntry constants{};
    constants.view          = dm::affineToHomogeneous(m_ViewHandler.view.GetViewMatrix());
    constants.projection    = m_ViewHandler.view.GetProjectionMatrix();
    constants.lightViewProj = lightViewProj;
    constants.sunLightDir   = m_Registry.getSunDirection();

    constants.viewFrustum = m_ViewHandler.view.GetViewFrustum();
    constants.worldToLight     = dm::affineToHomogeneous(m_ViewHandler.worldToLight);
    constants.shadowCasterMinLS = m_ViewHandler.shadowCasterBboxLS.m_mins;
    constants.shadowCasterMaxLS = m_ViewHandler.shadowCasterBboxLS.m_maxs;

    constants.numRegions = static_cast<uint32_t>(m_Registry.getRegions().size());

    constants.cameraPos     = m_ViewHandler.camera.GetPosition();
    constants.totalCapacity = m_TotalCapacity;

    const auto& lodDistances = m_Registry.getLodDistances();
    constants.numLods = static_cast<uint32_t>(m_Registry.getLodSegments().size());
    for (uint32_t i = 0; i < constants.numLods; i++)
        constants.lodDistances[i].x = lodDistances[i];

    m_CommandList->writeBuffer(m_Shared.constantBuffer, &constants, Render::c_CullConstantBufferSize);

    // --- GPU Cull Dispatch ---
    // Clear UAV counters
    m_CommandList->clearBufferUInt(m_CullPass.countBuffer, 0);
    m_CommandList->clearBufferUInt(m_CullPass.shadowCountBuffer, 0);

    // Reset indirect args buffers (re-upload staging with instanceCount=0)
    m_CommandList->writeBuffer(m_CullPass.indirectArgsBuffer,
        m_IndirectArgsStaging.data(),
        m_IndirectArgsStaging.size() * sizeof(nvrhi::DrawIndexedIndirectArguments));
    m_CommandList->writeBuffer(m_CullPass.shadowIndirectArgsBuffer,
        m_ShadowIndirectArgsStaging.data(),
        m_ShadowIndirectArgsStaging.size() * sizeof(nvrhi::DrawIndexedIndirectArguments));

    // region camera cull
    GFSDK_Aftermath_SetEventMarker(m_AftermathContext, "Cull::RegionDispatch", 0);
    {
        nvrhi::ComputeState cs;
        cs.pipeline = m_CullPass.regionPipeline;
        cs.bindings = { m_CullPass.bindingSet };
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch(
            (constants.numRegions + 63) / 64, 1, 1);
    }
    
    // Main camera cull
    GFSDK_Aftermath_SetEventMarker(m_AftermathContext, "Cull::MainDispatch", 0);
    {
        nvrhi::ComputeState cs;
        cs.pipeline = m_CullPass.mainPipeline;
        cs.bindings = { m_CullPass.bindingSet };
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch(
            (m_TotalCapacity + 255) / 256, 1, 1);
    }

    // Shadow cull
    GFSDK_Aftermath_SetEventMarker(m_AftermathContext, "Cull::ShadowDispatch", 0);
    {
        nvrhi::ComputeState cs;
        cs.pipeline = m_CullPass.shadowPipeline;
        cs.bindings = { m_CullPass.bindingSet };
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch(
            (m_TotalCapacity + 63) / 64, 1, 1);
    }

    // --- Copy cull counts to readback ring ---
    {
        uint32_t ringSlot   = m_ReadbackFrameIndex % k_QueuedFrames;
        uint64_t countSize  = m_ReadbackCountEntries  * sizeof(uint32_t);
        uint64_t shadowSize = m_ReadbackShadowEntries * sizeof(uint32_t);

        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], 0,
                                  m_CullPass.countBuffer, 0, countSize);
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], countSize,
                                  m_CullPass.shadowCountBuffer, 0, shadowSize);
    }

    // --- Draw passes (auto barriers: UAV→SRV transitions handled by setGraphicsState) ---
    GFSDK_Aftermath_SetEventMarker(m_AftermathContext, "Draw::Sky", 0);
    _RenderSkyPass(framebuffer);
    GFSDK_Aftermath_SetEventMarker(m_AftermathContext, "Draw::Shadow", 0);
    _RenderShadowPass();
    GFSDK_Aftermath_SetEventMarker(m_AftermathContext, "Draw::Scene", 0);
    _RenderScenePass(framebuffer);
    GFSDK_Aftermath_SetEventMarker(m_AftermathContext, "Frame::End", 0);

    // m_CommandList->endTimerQuery(m_GpuTimers[m_NextTimerIdx]);
    m_CommandList->close();
    GetDevice()->executeCommandList(m_CommandList);

    // int prevIdx = (m_NextTimerIdx + k_QueuedFrames - 1) % k_QueuedFrames;
    // if (GetDevice()->pollTimerQuery(m_GpuTimers[prevIdx]))
    //     m_UI.gpuFrameTimeMs = GetDevice()->getTimerQueryTime(m_GpuTimers[prevIdx]) * 1000.0f;
    // m_NextTimerIdx = (m_NextTimerIdx + 1) % k_QueuedFrames;

    m_UI.totalInstanceCount = m_Registry.totalInstanceCount();
    m_UI.drawCallCount      = m_NumSlots;

    // Read back cull counts from oldest ring slot (2 frames ago, GPU-complete)
    if (m_ReadbackFrameIndex >= (k_QueuedFrames - 1)) {
        uint32_t readSlot = (m_ReadbackFrameIndex + 1) % k_QueuedFrames;

        void* pData = GetDevice()->mapBuffer(m_ReadbackBuffers[readSlot], nvrhi::CpuAccessMode::Read);
        if (pData) {
            const uint32_t* counts = static_cast<const uint32_t*>(pData);

            uint32_t visibleSum = 0;
            for (uint32_t i = 0; i < m_ReadbackCountEntries; i++)
                visibleSum += counts[i];

            uint32_t shadowVisSum = 0;
            for (uint32_t i = 0; i < m_ReadbackShadowEntries; i++)
                shadowVisSum += counts[m_ReadbackCountEntries + i];

            GetDevice()->unmapBuffer(m_ReadbackBuffers[readSlot]);

            m_UI.visibleInstanceCount = visibleSum;
            m_UI.culledInstanceCount  = m_UI.totalInstanceCount - visibleSum;
            m_UI.shadowVisibleCount   = shadowVisSum;
            m_UI.shadowCulledCount    = m_UI.totalInstanceCount - shadowVisSum;
        }
    }
    m_ReadbackFrameIndex++;

    cpuTimer.Stop();
    m_UI.cpuRenderTimeMs = (float)cpuTimer.Milliseconds();
}

// ===========================================================================
// Sky pass (identical to P0)
// ===========================================================================

void ComputeCullRenderPass::_RenderSkyPass(nvrhi::IFramebuffer* framebuffer) {
    if (!m_SkyPass.pipeline) {
        nvrhi::GraphicsPipelineDesc pso;
        pso.VS             = m_SkyPass.vertexShader;
        pso.PS             = m_SkyPass.pixelShader;
        pso.bindingLayouts = { m_SkyPass.bindingLayout };
        pso.primType       = nvrhi::PrimitiveType::TriangleStrip;
        pso.renderState.rasterState.setCullNone();
        pso.renderState.depthStencilState
            .enableDepthTest()
            .disableDepthWrite()
            .disableStencil()
    #if XYLEM_USE_REVERSE_Z
            .setDepthFunc(nvrhi::ComparisonFunc::GreaterOrEqual);
    #else
            .setDepthFunc(nvrhi::ComparisonFunc::LessOrEqual);
    #endif
        m_SkyPass.pipeline = GetDevice()->createGraphicsPipeline(
            pso, framebuffer->getFramebufferInfo());
    }

    dm::affine3 viewToWorld = dm::affine3(m_ViewHandler.view.GetInverseViewMatrix());
    viewToWorld.m_translation = 0.f;
    dm::float4x4 clipToTranslatedWorld =
        m_ViewHandler.view.GetInverseProjectionMatrix(true) * dm::affineToHomogeneous(viewToWorld);

    SkyConstants skyConstants{};
    skyConstants.matClipToTranslatedWorld = clipToTranslatedWorld;

    auto& p         = skyConstants.params;
    p.directionToLight  = dm::normalize(-m_Registry.getSunDirection());
    p.angularSizeOfLight = dm::radians(1.0f);
    p.lightColor        = dm::float3(100.f, 98.f, 90.f);
    p.glowSize          = dm::radians(5.f);
    p.skyColor          = dm::float3(0.017f, 0.037f, 0.065f);
    p.glowIntensity     = 0.1f;
    p.horizonColor      = dm::float3(0.050f, 0.070f, 0.092f);
    p.horizonSize       = dm::radians(30.f);
    p.groundColor       = dm::float3(0.062f, 0.059f, 0.055f);
    p.glowSharpness     = 4.f;
    p.directionUp       = dm::float3(0.f, 1.f, 0.f);

    m_CommandList->writeBuffer(m_SkyPass.constantBuffer, &skyConstants, sizeof(skyConstants));

    nvrhi::GraphicsState skyState;
    skyState.pipeline    = m_SkyPass.pipeline;
    skyState.framebuffer = framebuffer;
    skyState.viewport    = m_ViewHandler.view.GetViewportState();
    skyState.bindings    = { m_SkyPass.bindingSet };
    m_CommandList->setGraphicsState(skyState);
    m_CommandList->draw(nvrhi::DrawArguments().setVertexCount(4));
}

// ===========================================================================
// Shadow pass — GPU cull results, visibility buffer indirection in VS
// ===========================================================================

void ComputeCullRenderPass::_RenderShadowPass() {
    if (m_ViewHandler.shadowCasterBboxLS.isempty())
        return;

    if (!m_ShadowPass.treePipeline) {
        nvrhi::GraphicsPipelineDesc pso;
        pso.VS             = m_ShadowPass.treeVS;
        pso.inputLayout    = m_ShadowPass.treeInputLayout;
        pso.bindingLayouts = { m_ShadowPass.bindingLayout };
        pso.primType       = nvrhi::PrimitiveType::TriangleList;
        pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
        pso.renderState.rasterState.setCullFront();
        pso.renderState.rasterState.depthBias            = 2;
        pso.renderState.rasterState.slopeScaledDepthBias = 2.0f;
        m_ShadowPass.treePipeline = GetDevice()->createGraphicsPipeline(
            pso, m_ShadowPass.framebuffer->getFramebufferInfo());
    }
    if (!m_ShadowPass.terrainPipeline && m_TerrainPass.indexCount > 0) {
        nvrhi::GraphicsPipelineDesc pso;
        pso.VS             = m_ShadowPass.terrainVS;
        pso.inputLayout    = m_ShadowPass.terrainInputLayout;
        pso.bindingLayouts = { m_ShadowPass.bindingLayout };
        pso.primType       = nvrhi::PrimitiveType::TriangleList;
        pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
        pso.renderState.rasterState.setCullBack();
        pso.renderState.rasterState.depthBias            = 2;
        pso.renderState.rasterState.slopeScaledDepthBias = 2.0f;
        m_ShadowPass.terrainPipeline = GetDevice()->createGraphicsPipeline(
            pso, m_ShadowPass.framebuffer->getFramebufferInfo());
    }

    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList,
        m_ShadowPass.framebuffer, 1.0f, 0);

    nvrhi::Viewport shadowVP(static_cast<float>(k_ShadowRes), static_cast<float>(k_ShadowRes));
    nvrhi::ViewportState shadowVPState;
    shadowVPState.addViewportAndScissorRect(shadowVP);

    // One draw call per asset using lowest LOD. The VS reads from the asset's
    // per-slot window in shadowVisBuf via shadowSlotOffsets[ai], drawing only
    // shadowCount[ai] instances — no cross-asset bleed.
    const uint32_t numLods   = static_cast<uint32_t>(m_Registry.getLodSegments().size());
    const uint32_t lowestLOD = numLods > 0 ? numLods - 1 : 0;

    nvrhi::GraphicsState shadowState;
    shadowState.pipeline    = m_ShadowPass.treePipeline;
    shadowState.framebuffer = m_ShadowPass.framebuffer;
    shadowState.viewport    = shadowVPState;
    shadowState.bindings    = { m_ShadowPass.bindingSet };
    shadowState.indirectParams = m_CullPass.shadowIndirectArgsBuffer;

    for (uint32_t ai = 0; ai < m_GPUAssets.size(); ai++) {
        uint32_t maxCount = m_MaxSlotCounts[ai * numLods];  // livePerAsset[ai]
        if (maxCount == 0) continue;

        const auto& lod = m_GPUAssets[ai].lods[lowestLOD];

        shadowState.vertexBuffers = { { lod.vertexBuffer, 0, 0 } };
        shadowState.indexBuffer   = { lod.indexBuffer, nvrhi::Format::R32_UINT, 0 };
        m_CommandList->setGraphicsState(shadowState);

        m_CommandList->setPushConstants(&ai, sizeof(ai));

        m_CommandList->drawIndexedIndirect(
            ai * sizeof(nvrhi::DrawIndexedIndirectArguments));
    }

    // Terrain shadow (unchanged — no instancing)
    const auto* terrainPtr = m_Registry.getTerrain();
    if (m_TerrainPass.indexCount > 0 && terrainPtr && 
        (terrainPtr->getBbox() * m_ViewHandler.worldToLight)
            .intersects(m_ViewHandler.shadowCasterBboxLS)) {
        nvrhi::GraphicsState terrShadow;
        terrShadow.pipeline    = m_ShadowPass.terrainPipeline;
        terrShadow.framebuffer = m_ShadowPass.framebuffer;
        terrShadow.viewport    = shadowVPState;
        terrShadow.bindings    = { m_ShadowPass.bindingSet };
        terrShadow.vertexBuffers = { { m_TerrainPass.vertexBuffer, 0, 0 } };
        terrShadow.indexBuffer   = { m_TerrainPass.indexBuffer, nvrhi::Format::R32_UINT, 0 };
        m_CommandList->setGraphicsState(terrShadow);
        
        // terrain does not se push constant but needs it for root signature
        uint32_t c = 0;
        m_CommandList->setPushConstants(&c, sizeof(c));

        m_CommandList->drawIndexed(
            nvrhi::DrawArguments()
                .setVertexCount(m_TerrainPass.indexCount)
        );
    }
}

// ===========================================================================
// Scene pass — GPU cull results, visibility buffer indirection in VS
// ===========================================================================

void ComputeCullRenderPass::_RenderScenePass(nvrhi::IFramebuffer* framebuffer) {
    const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();
    const uint32_t numLods = static_cast<uint32_t>(m_Registry.getLodSegments().size());

    if (!m_TreePass.pipeline) {
        nvrhi::GraphicsPipelineDesc psoDesc;
        psoDesc.VS           = m_TreePass.vertexShader;
        psoDesc.PS           = m_TreePass.pixelShader;
        psoDesc.inputLayout  = m_TreePass.inputLayout;
        psoDesc.bindingLayouts = { m_TreePass.bindingLayout };
        psoDesc.primType     = nvrhi::PrimitiveType::TriangleList;
    #if XYLEM_USE_REVERSE_Z
        psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
    #else
        psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    #endif
        m_TreePass.pipeline = GetDevice()->createGraphicsPipeline(psoDesc, fbinfo);
    }

    nvrhi::GraphicsState state;
    state.pipeline    = m_TreePass.pipeline;
    state.framebuffer = framebuffer;
    state.viewport    = m_ViewHandler.view.GetViewportState();
    state.indirectParams = m_CullPass.indirectArgsBuffer;

    // Indirect draw: one call per slot (asset × LOD).
    // Instance count comes from the indirect args buffer (written by the cull CS).
    for (uint32_t ai = 0; ai < m_GPUAssets.size(); ai++) {
        for (uint32_t li = 0; li < numLods; li++) {
            uint32_t slot = ai * numLods + li;
            if (m_MaxSlotCounts[slot] == 0) continue;

            const auto& lod = m_GPUAssets[ai].lods[li];

            state.bindings = { m_TreePass.bindingSets[m_GPUAssets[ai].textureSetIdx] };
            state.vertexBuffers = { { lod.vertexBuffer, 0, 0 } };
            state.indexBuffer   = { lod.indexBuffer, nvrhi::Format::R32_UINT, 0 };
            m_CommandList->setGraphicsState(state);

            m_CommandList->setPushConstants(&slot, sizeof(slot));

            m_CommandList->drawIndexedIndirect(
                slot * sizeof(nvrhi::DrawIndexedIndirectArguments));
        }
    }

    // Terrain color pass
    if (m_TerrainPass.indexCount > 0) {
        if (!m_TerrainPass.pipeline) {
            nvrhi::GraphicsPipelineDesc terrainPso;
            terrainPso.VS = m_TerrainPass.vertexShader;
            terrainPso.PS = m_TerrainPass.pixelShader;
            terrainPso.inputLayout = m_TerrainPass.inputLayout;
            terrainPso.bindingLayouts = { m_TerrainPass.bindingLayout };
            terrainPso.primType = nvrhi::PrimitiveType::TriangleList;
    #if XYLEM_USE_REVERSE_Z
            terrainPso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
    #else
            terrainPso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    #endif
            terrainPso.renderState.rasterState.setCullNone();
            m_TerrainPass.pipeline = GetDevice()->createGraphicsPipeline(terrainPso, fbinfo);
        }

        nvrhi::GraphicsState terrainState;
        terrainState.pipeline   = m_TerrainPass.pipeline;
        terrainState.framebuffer = framebuffer;
        terrainState.viewport   = m_ViewHandler.view.GetViewportState();
        terrainState.bindings   = { m_TerrainPass.bindingSet };
        terrainState.vertexBuffers = { { m_TerrainPass.vertexBuffer, 0, 0 } };
        terrainState.indexBuffer   = { m_TerrainPass.indexBuffer, nvrhi::Format::R32_UINT, 0 };
        m_CommandList->setGraphicsState(terrainState);
        
        m_CommandList->drawIndexed(
            nvrhi::DrawArguments()
                .setVertexCount(m_TerrainPass.indexCount)
        );
    }
}

// ===========================================================================
// Init helpers
// ===========================================================================

bool ComputeCullRenderPass::_InitShared() {
    m_Shared.constantBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(Render::c_CullConstantBufferSize)
            .setIsConstantBuffer(true)
            .setDebugName("CullConstantBuffer")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
    );
    return !!m_Shared.constantBuffer;
}

bool ComputeCullRenderPass::_InitCullPass(nvrhi::ICommandList* /*initCL*/) {
    // Create compute shaders
    m_CullPass.mainCS = m_ShaderFactory->CreateShader(
        "app/CullCS.hlsl", "CullMain", nullptr, nvrhi::ShaderType::Compute);
    m_CullPass.shadowCS = m_ShaderFactory->CreateShader(
        "app/CullCS.hlsl", "CullShadow", nullptr, nvrhi::ShaderType::Compute);
    m_CullPass.regionCS = m_ShaderFactory->CreateShader(
        "app/CullCS.hlsl", "CullRegion", nullptr, nvrhi::ShaderType::Compute);
    if (!m_CullPass.mainCS || !m_CullPass.shadowCS || !m_CullPass.regionCS) return false;

    // Create cull binding layout
    nvrhi::BindingLayoutDesc cullLayoutDesc;
    cullLayoutDesc.visibility = nvrhi::ShaderType::All;
    cullLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(0),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0), // regionData
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1), // instanceData  
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2), // mainSlotOffsets  
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(3), // shadowSlotOffsets

        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0), // mainRegionVisBuf
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(1), // mainSlotCountBuf
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2), // mainVisBuf
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(3), // shadowSlotCountBuf
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(4), // shadowVisBuf

        nvrhi::BindingLayoutItem::RawBuffer_UAV(5),        // mainIndirectArgs
        nvrhi::BindingLayoutItem::RawBuffer_UAV(6),        // shadowIndirectArgs
    };
    m_CullPass.bindingLayout = GetDevice()->createBindingLayout(cullLayoutDesc);
    if (!m_CullPass.bindingLayout) return false;

    // Create compute pipelines
    nvrhi::ComputePipelineDesc regionPsoDesc;
    regionPsoDesc.CS = m_CullPass.regionCS;
    regionPsoDesc.bindingLayouts = { m_CullPass.bindingLayout };
    m_CullPass.regionPipeline = GetDevice()->createComputePipeline(regionPsoDesc);
    
    nvrhi::ComputePipelineDesc mainPsoDesc;
    mainPsoDesc.CS = m_CullPass.mainCS;
    mainPsoDesc.bindingLayouts = { m_CullPass.bindingLayout };
    m_CullPass.mainPipeline = GetDevice()->createComputePipeline(mainPsoDesc);

    nvrhi::ComputePipelineDesc shadowPsoDesc;
    shadowPsoDesc.CS = m_CullPass.shadowCS;
    shadowPsoDesc.bindingLayouts = { m_CullPass.bindingLayout };
    m_CullPass.shadowPipeline = GetDevice()->createComputePipeline(shadowPsoDesc);

    if (!m_CullPass.mainPipeline || !m_CullPass.shadowPipeline) return false;

    return true;
}

bool ComputeCullRenderPass::_InitTreePass(nvrhi::ICommandList* initCL, engine::CommonRenderPasses& commonPasses) {
    m_TreePass.vertexShader = m_ShaderFactory->CreateShader(
        "app/ComputeCullRenderPass.hlsl", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_TreePass.pixelShader  = m_ShaderFactory->CreateShader(
        "app/ComputeCullRenderPass.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_TreePass.vertexShader || !m_TreePass.pixelShader) return false;

    nvrhi::VertexAttributeDesc attributes[] = {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(ProcGen::TreeVertex, pos))
            .setBufferIndex(0)
            .setElementStride(sizeof(ProcGen::TreeVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("NORMAL")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(ProcGen::TreeVertex, normal))
            .setBufferIndex(0)
            .setElementStride(sizeof(ProcGen::TreeVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("TANGENT")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(ProcGen::TreeVertex, tangent))
            .setBufferIndex(0)
            .setElementStride(sizeof(ProcGen::TreeVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("BITANGENT")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(ProcGen::TreeVertex, bitangent))
            .setBufferIndex(0)
            .setElementStride(sizeof(ProcGen::TreeVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("UV")
            .setFormat(nvrhi::Format::RG32_FLOAT)
            .setOffset(offsetof(ProcGen::TreeVertex, uv))
            .setBufferIndex(0)
            .setElementStride(sizeof(ProcGen::TreeVertex)),
    };
    m_TreePass.inputLayout = GetDevice()->createInputLayout(
        attributes, uint32_t(std::size(attributes)), m_TreePass.vertexShader);
    if (!m_TreePass.inputLayout) return false;

    // Load textures
    const auto& barkTextureSets = m_Registry.getBarkTextureSets();
    engine::TextureCache textureCache(GetDevice(), std::make_shared<vfs::NativeFileSystem>(), nullptr);
    m_TreePass.textureSets.resize(barkTextureSets.size());

    for (size_t i = 0; i < barkTextureSets.size(); i++) {
        std::filesystem::path texDir = g_ProjectDirectory / "media" / barkTextureSets[i] / "textures";
        std::filesystem::path diffPath, normPath;

        for (const auto& entry : std::filesystem::directory_iterator(texDir)) {
            std::string filename = entry.path().filename().string();
            if (filename.find("_diff_") != std::string::npos && entry.path().extension() == ".jpg")
                diffPath = entry.path();
            else if (filename.find("_nor_dx_") != std::string::npos && entry.path().extension() == ".jpg")
                normPath = entry.path();
        }

        auto diffLoaded = textureCache.LoadTextureFromFile(diffPath, true,  &commonPasses, initCL);
        auto normLoaded = textureCache.LoadTextureFromFile(normPath, false, &commonPasses, initCL);

        m_TreePass.textureSets[i].diffuse   = diffLoaded->texture;
        m_TreePass.textureSets[i].normalMap = normLoaded->texture;

        if (!m_TreePass.textureSets[i].diffuse || !m_TreePass.textureSets[i].normalMap)
            return false;
    }

    m_TreePass.sampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Wrap)
            .setAllFilters(true)
            .setMaxAnisotropy(8.f)
    );
    if (!m_TreePass.sampler) return false;

    // Create tree pass binding layout (with push constants for slot index)
    nvrhi::BindingLayoutDesc treeLayoutDesc;
    treeLayoutDesc.visibility = nvrhi::ShaderType::All;
    treeLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(0),                    
        nvrhi::BindingLayoutItem::PushConstants(1, sizeof(uint32_t)),   // slot
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0),              // visBuf
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),              // instBuf
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2),              // slotOffsets
        nvrhi::BindingLayoutItem::Texture_SRV(3),                       // t_Diffuse
        nvrhi::BindingLayoutItem::Texture_SRV(4),                       // t_NormalMap
        nvrhi::BindingLayoutItem::Texture_SRV(5),                       // t_ShadowMap
        nvrhi::BindingLayoutItem::Sampler(0),                           // s_Sampler
        nvrhi::BindingLayoutItem::Sampler(1),                           // s_ShadowSampler
    };
    m_TreePass.bindingLayout = GetDevice()->createBindingLayout(treeLayoutDesc);
    if (!m_TreePass.bindingLayout) return false;

    // Binding sets are created in _RebuildCullBindings after buffers exist
    return true;
}

// bool ComputeCullRenderPass::_InitTimerQueries() {
//     for (uint32_t i = 0; i < k_QueuedFrames; i++)
//         m_GpuTimers[i] = GetDevice()->createTimerQuery();
//     return true;
// }

bool ComputeCullRenderPass::_InitShadowPass() {
    m_ShadowPass.depthTexture = GetDevice()->createTexture(
        nvrhi::TextureDesc()
            .setWidth(k_ShadowRes).setHeight(k_ShadowRes)
            .setFormat(nvrhi::Format::D32)
            .setIsRenderTarget(true)
            .setUseClearValue(true)
            .setClearValue(nvrhi::Color(1.f))
            .setInitialState(nvrhi::ResourceStates::DepthWrite)
            .setKeepInitialState(true)
            .setDebugName("ShadowMap")
    );
    if (!m_ShadowPass.depthTexture) return false;

    m_ShadowPass.framebuffer = GetDevice()->createFramebuffer(
        nvrhi::FramebufferDesc().setDepthAttachment(m_ShadowPass.depthTexture));
    if (!m_ShadowPass.framebuffer) return false;

    m_ShadowPass.treeVS = m_ShaderFactory->CreateShader(
        "app/shadow_compute.hlsl", "tree_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_ShadowPass.terrainVS = m_ShaderFactory->CreateShader(
        "app/shadow_compute.hlsl", "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    if (!m_ShadowPass.treeVS || !m_ShadowPass.terrainVS) return false;

    // Tree shadow input layout — vertex only (instance data via SRV indirection)
    nvrhi::VertexAttributeDesc treeShadowAttrs[] = {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(ProcGen::TreeVertex, pos))
            .setBufferIndex(0)
            .setElementStride(sizeof(ProcGen::TreeVertex)),
    };
    m_ShadowPass.treeInputLayout = GetDevice()->createInputLayout(
        treeShadowAttrs, uint32_t(std::size(treeShadowAttrs)), m_ShadowPass.treeVS);
    if (!m_ShadowPass.treeInputLayout) return false;

    // Terrain shadow input layout
    nvrhi::VertexAttributeDesc terrainShadowAttrs[] = {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(Scene::TerrainVertex, pos))
            .setBufferIndex(0)
            .setElementStride(sizeof(Scene::TerrainVertex)),
    };
    m_ShadowPass.terrainInputLayout = GetDevice()->createInputLayout(
        terrainShadowAttrs, uint32_t(std::size(terrainShadowAttrs)), m_ShadowPass.terrainVS);
    if (!m_ShadowPass.terrainInputLayout) return false;

    m_ShadowPass.comparisonSampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setMinFilter(true)
            .setMagFilter(true)
            .setMipFilter(false)
            .setReductionType(nvrhi::SamplerReductionType::Comparison)
            .setAllAddressModes(nvrhi::SamplerAddressMode::Border)
            .setBorderColor(nvrhi::Color(1.f))
    );
    if (!m_ShadowPass.comparisonSampler) return false;

    // Shadow binding layout
    nvrhi::BindingLayoutDesc shadowLayoutDesc;
    shadowLayoutDesc.visibility = nvrhi::ShaderType::All;
    shadowLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(0),
        nvrhi::BindingLayoutItem::PushConstants(1, sizeof(uint32_t)),  // assetIndex
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0),             // shadowVisBuf
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),             // instanceBuf
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2),             // shadowSlotOffsets
    };
    m_ShadowPass.bindingLayout = GetDevice()->createBindingLayout(shadowLayoutDesc);
    if (!m_ShadowPass.bindingLayout) return false;

    // Binding set created in _RebuildCullBindings after buffers exist
    return true;
}

bool ComputeCullRenderPass::_InitTerrainPass(nvrhi::ICommandList* initCL) {
    const auto* terrain = m_Registry.getTerrain();
    if (!terrain) return true;

    const auto& verts   = terrain->getVertices();
    const auto& indices = terrain->getIndices();
    if (verts.empty() || indices.empty()) return true;

    m_TerrainPass.indexCount = static_cast<uint32_t>(indices.size());

    m_TerrainPass.vertexShader = m_ShaderFactory->CreateShader("app/terrain.hlsl", "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_TerrainPass.pixelShader  = m_ShaderFactory->CreateShader("app/terrain.hlsl", "terrain_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_TerrainPass.vertexShader || !m_TerrainPass.pixelShader) return false;

    nvrhi::VertexAttributeDesc terrainAttrs[] = {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(Scene::TerrainVertex, pos))
            .setBufferIndex(0)
            .setElementStride(sizeof(Scene::TerrainVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("NORMAL")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(Scene::TerrainVertex, normal))
            .setBufferIndex(0)
            .setElementStride(sizeof(Scene::TerrainVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("UV")
            .setFormat(nvrhi::Format::RG32_FLOAT)
            .setOffset(offsetof(Scene::TerrainVertex, uv))
            .setBufferIndex(0)
            .setElementStride(sizeof(Scene::TerrainVertex)),
    };
    m_TerrainPass.inputLayout = GetDevice()->createInputLayout(
        terrainAttrs, uint32_t(std::size(terrainAttrs)), m_TerrainPass.vertexShader);
    if (!m_TerrainPass.inputLayout) return false;

    nvrhi::BufferDesc vbDesc;
    vbDesc.isVertexBuffer = true;
    vbDesc.byteSize       = verts.size() * sizeof(Scene::TerrainVertex);
    vbDesc.debugName      = "TerrainVB";
    vbDesc.initialState   = nvrhi::ResourceStates::CopyDest;
    m_TerrainPass.vertexBuffer = GetDevice()->createBuffer(vbDesc);
    initCL->beginTrackingBufferState(m_TerrainPass.vertexBuffer, nvrhi::ResourceStates::CopyDest);
    initCL->writeBuffer(m_TerrainPass.vertexBuffer, verts.data(), vbDesc.byteSize);
    initCL->setPermanentBufferState(m_TerrainPass.vertexBuffer, nvrhi::ResourceStates::VertexBuffer);

    nvrhi::BufferDesc ibDesc;
    ibDesc.isIndexBuffer = true;
    ibDesc.byteSize      = indices.size() * sizeof(uint32_t);
    ibDesc.debugName     = "TerrainIB";
    ibDesc.initialState  = nvrhi::ResourceStates::CopyDest;
    m_TerrainPass.indexBuffer = GetDevice()->createBuffer(ibDesc);
    initCL->beginTrackingBufferState(m_TerrainPass.indexBuffer, nvrhi::ResourceStates::CopyDest);
    initCL->writeBuffer(m_TerrainPass.indexBuffer, indices.data(), ibDesc.byteSize);
    initCL->setPermanentBufferState(m_TerrainPass.indexBuffer, nvrhi::ResourceStates::IndexBuffer);

    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_Shared.constantBuffer,
            nvrhi::BufferRange(0, Render::c_CullConstantBufferSize)),
        nvrhi::BindingSetItem::Sampler(0, m_ShadowPass.comparisonSampler),
        nvrhi::BindingSetItem::Texture_SRV(0, m_ShadowPass.depthTexture),
    };
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
            bsd, m_TerrainPass.bindingLayout, m_TerrainPass.bindingSet))
        return false;

    return true;
}

bool ComputeCullRenderPass::_InitSkyPass() {
    m_SkyPass.vertexShader = m_ShaderFactory->CreateShader("app/sky.hlsl", "sky_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_SkyPass.pixelShader  = m_ShaderFactory->CreateShader("app/sky.hlsl", "sky_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_SkyPass.vertexShader || !m_SkyPass.pixelShader) return false;

    nvrhi::BufferDesc cbDesc;
    cbDesc.byteSize         = sizeof(SkyConstants);
    cbDesc.isConstantBuffer = true;
    cbDesc.isVolatile       = true;
    cbDesc.maxVersions      = 16;
    cbDesc.debugName        = "SkyConstants";
    m_SkyPass.constantBuffer = GetDevice()->createBuffer(cbDesc);
    if (!m_SkyPass.constantBuffer) return false;

    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_SkyPass.constantBuffer),
    };
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
            bsd, m_SkyPass.bindingLayout, m_SkyPass.bindingSet))
        return false;

    return true;
}
