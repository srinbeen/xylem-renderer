#include "include/ComputeRenderPass.hpp"
#include "include/Globals.hpp"
#include "include/macros.h"
#include "include/Terrain.hpp"


#include <nvrhi/utils.h>
#include <donut/app/imgui_renderer.h>
#include <donut/app/Timer.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/core/math/math.h>
#include <donut/core/json.h>

using namespace donut::math;
#include <donut/shaders/sky_cb.h>

#include <algorithm>
#include <cmath>
#include <filesystem>

using namespace Xylem;

namespace {

struct ImpostorBakeCB {
    dm::float4x4    mvp;
    dm::float4      _pad[12];
};

static constexpr size_t c_ImpostorBakeCBSize =
    (sizeof(ImpostorBakeCB) + (nvrhi::c_ConstantBufferOffsetSizeAlignment - 1))
    & ~(nvrhi::c_ConstantBufferOffsetSizeAlignment - 1);

dm::affine3 BuildtoObjectTransform()
{
    return dm::affine3::identity();
}

dm::float3 HemiOctahedronToUnitVector(const dm::float2& uv)
{
    const float planeX = uv.x * 2.f - 1.f;
    const float planeY = uv.y * 2.f - 1.f;
    const float octX = (planeX + planeY) * 0.5f;
    const float octZ = (planeY - planeX) * 0.5f;
    const float y = std::max(1.f - std::abs(octX) - std::abs(octZ), 0.f);
    return dm::normalize(dm::float3(octX - octZ, y, octX + octZ));
}

dm::float2 ImpostorViewGridUV(uint32_t viewIndex)
{
    const uint32_t x = viewIndex % ComputeRenderPass::k_ImpostorAzimuthViews;
    const uint32_t y = viewIndex / ComputeRenderPass::k_ImpostorAzimuthViews;
    return dm::float2(
        float(x) / float(std::max(1u, ComputeRenderPass::k_ImpostorAzimuthViews - 1u)),
        float(y) / float(std::max(1u, ComputeRenderPass::k_ImpostorElevationViews - 1u)));
}

dm::float3 ImpostorViewDirection(uint32_t viewIndex)
{
    return HemiOctahedronToUnitVector(ImpostorViewGridUV(viewIndex));
}

void ImpostorViewBasis(uint32_t viewIndex, dm::float3& dirToCamera, dm::float3& right, dm::float3& viewUp)
{
    dirToCamera = ImpostorViewDirection(viewIndex);

    const dm::float3 worldUp(0.f, 1.f, 0.f);
    const dm::float3 horizontal(dirToCamera.x, 0.f, dirToCamera.z);
    const float horizontalLen = std::sqrt(horizontal.x * horizontal.x + horizontal.z * horizontal.z);
    right = horizontalLen >= 1e-6f
        ? dm::normalize(dm::cross(worldUp, -horizontal / horizontalLen))
        : dm::float3(1.f, 0.f, 0.f);
    viewUp = dm::normalize(dm::cross(-dirToCamera, right));
}

dm::float4x4 BuildtoProjTransform(const dm::box3& localBbox, uint32_t viewIndex)
{
    const dm::affine3 modelxfm = BuildtoObjectTransform();

    dm::float3 treeToCam;
    dm::float3 right;
    dm::float3 trueUp;
    ImpostorViewBasis(viewIndex, treeToCam, right, trueUp);

    const dm::float3 look = -treeToCam;
    const dm::affine3 viewxfm = dm::affine3::from_cols(right, trueUp, look, dm::float3::zero());
    const dm::affine modelviewxfm = modelxfm * viewxfm;

    dm::box3 bbox = localBbox * modelxfm;
    bbox *= viewxfm;

    const dm::float4x4 projxfm = dm::orthoProjD3DStyle(
        bbox.m_mins.x, bbox.m_maxs.x,
        bbox.m_mins.y, bbox.m_maxs.y,
        bbox.m_mins.z, bbox.m_maxs.z
    );

    return dm::affineToHomogeneous(modelviewxfm) * projxfm;
}

} // namespace

// ===========================================================================
// GPU asset upload helpers (identical to P0)
// ===========================================================================

void ComputeRenderPass::_UploadAsset(
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
        std::string lodSuffix = "_LOD" + std::to_string(j);

        auto uploadVB = [&](const auto& data, const char* suffix, nvrhi::BufferHandle& out) {
            vDesc.debugName = "VB_" + assetDef.name + lodSuffix + suffix;
            vDesc.byteSize  = data.size() * sizeof(data[0]);
            out = device->createBuffer(vDesc);
            commandList->beginTrackingBufferState(out, nvrhi::ResourceStates::CopyDest);
            commandList->writeBuffer(out, data.data(), vDesc.byteSize);
            commandList->setPermanentBufferState(out, nvrhi::ResourceStates::VertexBuffer);
        };

        uploadVB(lodDef.positions,  "_pos",   gpuAsset.lods[j].vbs.position);
        uploadVB(lodDef.normals,    "_nor",   gpuAsset.lods[j].vbs.normal);
        uploadVB(lodDef.tangents,   "_tan",   gpuAsset.lods[j].vbs.tangent);
        uploadVB(lodDef.bitangents, "_bitan", gpuAsset.lods[j].vbs.bitangent);
        uploadVB(lodDef.uvs,        "_uv",    gpuAsset.lods[j].vbs.uv);

        iDesc.debugName = "IB_" + assetDef.name + lodSuffix;
        iDesc.byteSize  = lodDef.indices.size() * sizeof(uint32_t);
        auto iBuf = device->createBuffer(iDesc);
        commandList->beginTrackingBufferState(iBuf, nvrhi::ResourceStates::CopyDest);
        commandList->writeBuffer(iBuf, lodDef.indices.data(), iDesc.byteSize);
        commandList->setPermanentBufferState(iBuf, nvrhi::ResourceStates::IndexBuffer);

        gpuAsset.lods[j].indexBuffer    = iBuf;
        gpuAsset.lods[j].indexCount     = static_cast<uint32_t>(lodDef.indices.size());
        gpuAsset.lods[j].radialSegments = lodDef.radialSegments;
        gpuAsset.lods[j].bbox           = lodDef.bbox;
    }
}

void ComputeRenderPass::_UploadAllAssets(nvrhi::IDevice* device, nvrhi::ICommandList* commandList) {
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

void ComputeRenderPass::_BuildRegionWindows() {
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

void ComputeRenderPass::_BuildSlotLayout() {
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

    // Impostor terminal LOD slots: one visibility window per asset.
    m_ImpostorMaxSlotCounts.resize(numAssets);
    m_ImpostorSlotOffsets.resize(numAssets);
    m_ImpostorVisBufferSize = 0;
    for (uint32_t ai = 0; ai < numAssets; ai++) {
        m_ImpostorMaxSlotCounts[ai] = livePerAsset[ai];
        m_ImpostorSlotOffsets[ai]   = m_ImpostorVisBufferSize;
        m_ImpostorVisBufferSize    += livePerAsset[ai];
    }
    m_ImpostorVisBufferSize = std::max(1u, m_ImpostorVisBufferSize);

    // Shadow slot layout — numAssets × NUM_CASCADES slots, interleaved as
    // slot = ai * NUM_CASCADES + c. Each slot reserves livePerAsset[ai] vis entries
    // (worst case: every instance lives in that cascade).
    const uint32_t numShadowSlots = numAssets * Render::c_NumCascades;
    m_ShadowSlotOffsets.resize(numShadowSlots);
    m_ShadowVisBufferSize = 0;
    for (uint32_t ai = 0; ai < numAssets; ai++) {
        for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
            uint32_t slot = ai * Render::c_NumCascades + c;
            m_ShadowSlotOffsets[slot] = m_ShadowVisBufferSize;
            m_ShadowVisBufferSize    += livePerAsset[ai];
        }
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

    // Shadow indirect args — one per (asset × cascade), using lowest LOD.
    // Slot indexing matches m_ShadowSlotOffsets: slot = ai * NUM_CASCADES + c.
    // Impostor indirect args: one non-indexed quad draw per asset.
    m_ImpostorIndirectArgsStaging.resize(std::max(1u, numAssets));
    for (uint32_t ai = 0; ai < numAssets; ai++) {
        auto& args = m_ImpostorIndirectArgsStaging[ai];
        args.vertexCount           = 4;
        args.instanceCount         = 0;
        args.startVertexLocation   = 0;
        args.startInstanceLocation = 0;
    }

    const uint32_t lowestLOD = numLods > 0 ? numLods - 1 : 0;
    m_ShadowIndirectArgsStaging.resize(std::max(1u, numShadowSlots));
    for (uint32_t ai = 0; ai < numAssets; ai++) {
        for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
            uint32_t slot = ai * Render::c_NumCascades + c;
            auto& args = m_ShadowIndirectArgsStaging[slot];
            args.indexCount            = m_GPUAssets[ai].lods[lowestLOD].indexCount;
            args.instanceCount         = 0;
            args.startIndexLocation    = 0;
            args.baseVertexLocation    = 0;
            args.startInstanceLocation = 0;
        }
    }
}

// ===========================================================================
// Upload cull buffers to GPU
// ===========================================================================

void ComputeRenderPass::_UploadCullBuffers(nvrhi::ICommandList* commandList) {
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

    const uint32_t numAssets      = static_cast<uint32_t>(m_GPUAssets.size());
    const uint32_t numShadowSlots = numAssets * Render::c_NumCascades;

    m_CullPass.impostorCountBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(std::max(1u, numAssets) * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("ImpostorCountBuffer")
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    m_CullPass.impostorVisBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(m_ImpostorVisBufferSize * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("ImpostorVisibilityBuffer")
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    m_CullPass.impostorSlotOffsetBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(std::max(1u, numAssets) * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("ImpostorSlotOffsetBuffer")
            .setCanHaveUAVs(false)
            .setInitialState(nvrhi::ResourceStates::CopyDest)
    );
    commandList->beginTrackingBufferState(m_CullPass.impostorSlotOffsetBuffer, nvrhi::ResourceStates::CopyDest);
    if (numAssets > 0)
        commandList->writeBuffer(m_CullPass.impostorSlotOffsetBuffer,
            m_ImpostorSlotOffsets.data(), numAssets * sizeof(uint32_t));
    commandList->setPermanentBufferState(m_CullPass.impostorSlotOffsetBuffer, nvrhi::ResourceStates::ShaderResource);

    // ShadowCountBuffer — UAV uint32[numShadowSlots], one counter per (asset × cascade)
    m_CullPass.shadowCountBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(std::max(1u, numShadowSlots) * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("ShadowCountBuffer")
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    // ShadowVisBuffer — UAV uint32[shadowVisBufferSize], per-(asset × cascade) windows
    m_CullPass.shadowVisBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(m_ShadowVisBufferSize * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("ShadowVisBuffer")
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    // ShadowSlotOffsetBuffer — SRV uint32[numShadowSlots], prefix sums into ShadowVisBuffer
    m_CullPass.shadowSlotOffsetBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(std::max(1u, numShadowSlots) * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("ShadowSlotOffsetBuffer")
            .setCanHaveUAVs(false)
            .setInitialState(nvrhi::ResourceStates::CopyDest)
    );
    commandList->beginTrackingBufferState(m_CullPass.shadowSlotOffsetBuffer, nvrhi::ResourceStates::CopyDest);
    if (numShadowSlots > 0)
        commandList->writeBuffer(m_CullPass.shadowSlotOffsetBuffer,
            m_ShadowSlotOffsets.data(), numShadowSlots * sizeof(uint32_t));
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

    // ShadowIndirectArgsBuffer — UAV + indirect draw args, DrawIndexedIndirectArguments[numShadowSlots]
    uint32_t impostorIndirectArgsBufSize = std::max(1u, numAssets) * sizeof(nvrhi::DrawIndirectArguments);
    m_CullPass.impostorIndirectArgsBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(impostorIndirectArgsBufSize)
            .setDebugName("ImpostorIndirectArgsBuffer")
            .setIsDrawIndirectArgs(true)
            .setCanHaveUAVs(true)
            .setCanHaveRawViews(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    uint32_t shadowIndirectArgsBufSize = std::max(1u, numShadowSlots) * sizeof(nvrhi::DrawIndexedIndirectArguments);
    m_CullPass.shadowIndirectArgsBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(shadowIndirectArgsBufSize)
            .setDebugName("ShadowIndirectArgsBuffer")
            .setIsDrawIndirectArgs(true)
            .setCanHaveUAVs(true)
            .setCanHaveRawViews(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    // ShadowUniqueCounter — single uint32, incremented once per instance that
    // passes at least one cascade. Used by UI to report unique-caster count
    // distinct from the per-cascade summed count (which double-counts).
    m_CullPass.shadowUniqueCounter = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(sizeof(uint32_t))
            .setDebugName("ShadowUniqueCounter")
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

    // Readback ring buffers — combined [mesh LOD counts | impostor counts | shadow counts | shadow unique].
    m_ReadbackCountEntries    = std::max(1u, m_NumSlots);
    m_ReadbackImpostorEntries = std::max(1u, numAssets);
    m_ReadbackShadowEntries   = std::max(1u, numShadowSlots);
    uint64_t readbackSize =
        (m_ReadbackCountEntries + m_ReadbackImpostorEntries + m_ReadbackShadowEntries + 1)
        * sizeof(uint32_t);

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

void ComputeRenderPass::_RebuildCullBindings() {
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
        nvrhi::BindingSetItem::StructuredBuffer_SRV(5, m_CullPass.impostorSlotOffsetBuffer),

        nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_CullPass.regionVisibleBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(1, m_CullPass.countBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(2, m_CullPass.visibilityBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(3, m_CullPass.shadowCountBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(4, m_CullPass.shadowVisBuffer),

        nvrhi::BindingSetItem::RawBuffer_UAV(5, m_CullPass.indirectArgsBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(6, m_CullPass.shadowIndirectArgsBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(7, m_CullPass.shadowUniqueCounter),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(8, m_CullPass.impostorCountBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(9, m_CullPass.impostorVisBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(10, m_CullPass.impostorIndirectArgsBuffer),

        nvrhi::BindingSetItem::Texture_SRV(4, m_HiZ.hizTexture),
        nvrhi::BindingSetItem::Sampler(0, m_HiZ.pointSampler),
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

    m_ImpostorPass.bindingSets.resize(m_TreePass.textureSets.size());
    for (size_t i = 0; i < m_TreePass.textureSets.size(); i++) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_Shared.constantBuffer,
                nvrhi::BufferRange(0, Render::c_CullConstantBufferSize)),
            nvrhi::BindingSetItem::PushConstants(1, sizeof(uint32_t)),

            nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_CullPass.impostorVisBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_CullPass.persistentInstBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_CullPass.impostorSlotOffsetBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_CullPass.cullDataBuffer),

            nvrhi::BindingSetItem::Texture_SRV(4, m_ImpostorPass.albedoAlphaTexture),
            nvrhi::BindingSetItem::Texture_SRV(5, m_ImpostorPass.normalTexture),
            nvrhi::BindingSetItem::Texture_SRV(6, m_ImpostorPass.depthTexture, nvrhi::Format::R32_FLOAT),
            nvrhi::BindingSetItem::Texture_SRV(7, m_ShadowPass.depthTexture),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(8, m_ImpostorPass.assetDimsBuffer),

            nvrhi::BindingSetItem::Sampler(0, m_ImpostorPass.sampler),
            nvrhi::BindingSetItem::Sampler(1, m_ShadowPass.comparisonSampler),
            nvrhi::BindingSetItem::Sampler(2, m_ImpostorPass.depthSampler),
        };
        m_ImpostorPass.bindingSets[i] = device->createBindingSet(bsd, m_ImpostorPass.bindingLayout);
    }

    // Shadow pass binding set
    nvrhi::BindingSetDesc shadowBSD;
    shadowBSD.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_Shared.constantBuffer,
            nvrhi::BufferRange(0, Render::c_CullConstantBufferSize)),
        nvrhi::BindingSetItem::PushConstants(1, sizeof(uint32_t) * 2),

        nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_CullPass.shadowVisBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_CullPass.persistentInstBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_CullPass.shadowSlotOffsetBuffer),
    };
    m_ShadowPass.bindingSet = device->createBindingSet(shadowBSD, m_ShadowPass.bindingLayout);
}

// ===========================================================================
// Ensure Hi-Z resources exist at the given resolution (lazy init / resize)
// ===========================================================================

void ComputeRenderPass::_EnsureHiZResources(uint32_t width, uint32_t height) {
    // Skip if already at the right size
    if (m_DepthPrepass.depthTexture) {
        auto desc = m_DepthPrepass.depthTexture->getDesc();
        if (desc.width == width && desc.height == height)
            return;
    }

    auto device = GetDevice();

    // --- Depth prepass texture + framebuffer ---
    m_DepthPrepass.depthTexture = device->createTexture(
        nvrhi::TextureDesc()
            .setWidth(width).setHeight(height)
            .setFormat(nvrhi::Format::D32)
            .setIsRenderTarget(true)
            .setUseClearValue(true)
        #if XYLEM_USE_REVERSE_Z
            .setClearValue(nvrhi::Color(0.f))
        #else
            .setClearValue(nvrhi::Color(1.f))
        #endif
            .setInitialState(nvrhi::ResourceStates::DepthWrite)
            .setKeepInitialState(true)
            .setDebugName("DepthPrepass_Depth")
    );
    m_DepthPrepass.framebuffer = device->createFramebuffer(
        nvrhi::FramebufferDesc().setDepthAttachment(m_DepthPrepass.depthTexture));

    // Invalidate pipelines (resolution-dependent)
    m_DepthPrepass.treePipeline    = nullptr;
    m_DepthPrepass.terrainPipeline = nullptr;

    // --- Hi-Z texture with full mip chain (RG32: .r=farthest for Hi-Z, .g=nearest for SDSM) ---
    m_HiZ.numMips = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
    m_HiZ.hizTexture = device->createTexture(
        nvrhi::TextureDesc()
            .setWidth(width).setHeight(height)
            .setMipLevels(m_HiZ.numMips)
            .setFormat(nvrhi::Format::RG32_FLOAT)
            .setIsUAV(true)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true)
            .setDebugName("HiZTexture")
    );

    // --- Per-mip binding sets for Hi-Z build ---
    m_HiZ.buildBindingSets.resize(m_HiZ.numMips);

    // Set 0: copy from depth prepass (D32 read as R32_FLOAT) -> Hi-Z mip 0 (RG32)
    // Shader seeds both channels from the single-channel source.
    {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::PushConstants(0, sizeof(uint32_t) * 2),
            nvrhi::BindingSetItem::Texture_SRV(0, m_DepthPrepass.depthTexture,
                nvrhi::Format::R32_FLOAT,
                nvrhi::TextureSubresourceSet(0, 1, 0, 1)),
            nvrhi::BindingSetItem::Texture_UAV(0, m_HiZ.hizTexture,
                nvrhi::Format::RG32_FLOAT,
                nvrhi::TextureSubresourceSet(0, 1, 0, 1)),
        };
        m_HiZ.buildBindingSets[0] = device->createBindingSet(bsd, m_HiZ.buildBindingLayout);
    }

    // Sets 1..N-1: downsample mip i-1 -> mip i (both RG32)
    for (uint32_t mip = 1; mip < m_HiZ.numMips; mip++) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::PushConstants(0, sizeof(uint32_t) * 2),
            nvrhi::BindingSetItem::Texture_SRV(0, m_HiZ.hizTexture,
                nvrhi::Format::RG32_FLOAT,
                nvrhi::TextureSubresourceSet(mip - 1, 1, 0, 1)),
            nvrhi::BindingSetItem::Texture_UAV(0, m_HiZ.hizTexture,
                nvrhi::Format::RG32_FLOAT,
                nvrhi::TextureSubresourceSet(mip, 1, 0, 1)),
        };
        m_HiZ.buildBindingSets[mip] = device->createBindingSet(bsd, m_HiZ.buildBindingLayout);
    }

    // --- Per-mip debug view textures (single-mip, RG32_FLOAT, for ImGui display) ---
    m_HiZ.debugMipTextures.resize(m_HiZ.numMips);
    for (uint32_t mip = 0; mip < m_HiZ.numMips; mip++) {
        uint32_t mipW = std::max(1u, width  >> mip);
        uint32_t mipH = std::max(1u, height >> mip);
        m_HiZ.debugMipTextures[mip] = device->createTexture(
            nvrhi::TextureDesc()
                .setWidth(mipW).setHeight(mipH)
                .setMipLevels(1)
                .setFormat(nvrhi::Format::RG32_FLOAT)
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true)
                .setDebugName(("HiZ_Debug_Mip" + std::to_string(mip)).c_str())
        );
    }

    // --- Depth prepass binding set (references main cull vis/inst/slot buffers) ---
    nvrhi::BindingSetDesc prepassBSD;
    prepassBSD.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_Shared.constantBuffer,
            nvrhi::BufferRange(0, Render::c_CullConstantBufferSize)),
        nvrhi::BindingSetItem::PushConstants(1, sizeof(uint32_t)),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_CullPass.visibilityBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_CullPass.persistentInstBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_CullPass.slotOffsetBuffer),
    };
    m_DepthPrepass.bindingSet = device->createBindingSet(prepassBSD, m_DepthPrepass.bindingLayout);

    // Rebuild cull binding set to reference the new Hi-Z texture
    _RebuildCullBindings();

    // Rebuild SDSM binding set (reads top mip of Hi-Z texture)
    if (m_SDSM.buildBindingLayout) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_SDSM.inputCB),
            nvrhi::BindingSetItem::Texture_SRV(0, m_HiZ.hizTexture),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_SDSM.cascadeDataBuffer),
        };
        m_SDSM.buildBindingSet = device->createBindingSet(bsd, m_SDSM.buildBindingLayout);
    }
}

// ===========================================================================
// Depth prepass — draw last frame's visible set depth-only
// ===========================================================================

void ComputeRenderPass::_RenderDepthPrepass() {
    const uint32_t numLods = static_cast<uint32_t>(m_Registry.getLodSegments().size());

    // Clear depth to far plane
#if XYLEM_USE_REVERSE_Z
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList,
        m_DepthPrepass.framebuffer, 0.f, 0);
#else
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList,
        m_DepthPrepass.framebuffer, 1.f, 0);
#endif

    // Lazy pipeline creation
    if (!m_DepthPrepass.treePipeline) {
        nvrhi::GraphicsPipelineDesc pso;
        pso.VS             = m_DepthPrepass.treeVS;
        pso.inputLayout    = m_DepthPrepass.treeInputLayout;
        pso.bindingLayouts = { m_DepthPrepass.bindingLayout };
        pso.primType       = nvrhi::PrimitiveType::TriangleList;
    #if XYLEM_USE_REVERSE_Z
        pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
    #else
        pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    #endif
        pso.renderState.rasterState.setCullBack();
        m_DepthPrepass.treePipeline = GetDevice()->createGraphicsPipeline(
            pso, m_DepthPrepass.framebuffer->getFramebufferInfo());
    }

    // Draw trees using last frame's indirect args (not yet cleared)
    nvrhi::GraphicsState state;
    state.pipeline    = m_DepthPrepass.treePipeline;
    state.framebuffer = m_DepthPrepass.framebuffer;
    state.viewport    = m_ViewHandler.view.GetViewportState();
    state.bindings    = { m_DepthPrepass.bindingSet };
    state.indirectParams = m_CullPass.indirectArgsBuffer;

    for (uint32_t ai = 0; ai < m_GPUAssets.size(); ai++) {
        for (uint32_t li = 0; li < numLods; li++) {
            uint32_t slot = ai * numLods + li;
            if (m_MaxSlotCounts[slot] == 0) continue;

            const auto& lod = m_GPUAssets[ai].lods[li];
            state.vertexBuffers = { { lod.vbs.position, 0, 0 } };
            state.indexBuffer   = { lod.indexBuffer, nvrhi::Format::R32_UINT, 0 };
            m_CommandList->setGraphicsState(state);
            m_CommandList->setPushConstants(&slot, sizeof(slot));
            m_CommandList->drawIndexedIndirect(
                slot * sizeof(nvrhi::DrawIndexedIndirectArguments));
        }
    }

    // Draw terrain (non-instanced, always visible)
    if (m_TerrainPass.indexCount > 0) {
        if (!m_DepthPrepass.terrainPipeline) {
            nvrhi::GraphicsPipelineDesc pso;
            pso.VS             = m_DepthPrepass.terrainVS;
            pso.inputLayout    = m_DepthPrepass.terrainInputLayout;
            pso.bindingLayouts = { m_DepthPrepass.bindingLayout };
            pso.primType       = nvrhi::PrimitiveType::TriangleList;
        #if XYLEM_USE_REVERSE_Z
            pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
        #else
            pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
        #endif
            pso.renderState.rasterState.setCullNone();
            m_DepthPrepass.terrainPipeline = GetDevice()->createGraphicsPipeline(
                pso, m_DepthPrepass.framebuffer->getFramebufferInfo());
        }

        nvrhi::GraphicsState terrState;
        terrState.pipeline    = m_DepthPrepass.terrainPipeline;
        terrState.framebuffer = m_DepthPrepass.framebuffer;
        terrState.viewport    = m_ViewHandler.view.GetViewportState();
        terrState.bindings    = { m_DepthPrepass.bindingSet };
        terrState.vertexBuffers = { { m_TerrainPass.vertexBuffer, 0, 0 } };
        terrState.indexBuffer   = { m_TerrainPass.indexBuffer, nvrhi::Format::R32_UINT, 0 };
        m_CommandList->setGraphicsState(terrState);

        uint32_t c = 0;
        m_CommandList->setPushConstants(&c, sizeof(c));
        m_CommandList->drawIndexed(
            nvrhi::DrawArguments().setVertexCount(m_TerrainPass.indexCount));
    }
}

// ===========================================================================
// Build Hi-Z mip chain from depth prepass
// ===========================================================================

void ComputeRenderPass::_BuildHiZMipChain() {
    auto desc = m_DepthPrepass.depthTexture->getDesc();
    uint32_t w = desc.width;
    uint32_t h = desc.height;

    // Pass 0: Copy depth texture -> Hi-Z mip 0
    {
        uint32_t dims[2] = { w, h };
        nvrhi::ComputeState cs;
        cs.pipeline = m_HiZ.copyPipeline;
        cs.bindings = { m_HiZ.buildBindingSets[0] };
        m_CommandList->setComputeState(cs);
        m_CommandList->setPushConstants(dims, sizeof(dims));
        m_CommandList->dispatch((w + 7) / 8, (h + 7) / 8, 1);
    }

    // Passes 1..N-1: Downsample mip i-1 -> mip i
    for (uint32_t mip = 1; mip < m_HiZ.numMips; mip++) {
        uint32_t mipW = std::max(1u, w >> mip);
        uint32_t mipH = std::max(1u, h >> mip);
        uint32_t dims[2] = { mipW, mipH };

        nvrhi::ComputeState cs;
        cs.pipeline = m_HiZ.buildPipeline;
        cs.bindings = { m_HiZ.buildBindingSets[mip] };
        m_CommandList->setComputeState(cs);
        m_CommandList->setPushConstants(dims, sizeof(dims));
        m_CommandList->dispatch((mipW + 7) / 8, (mipH + 7) / 8, 1);
    }

    // Copy each mip into its debug view texture (for ImGui display)
    if (m_UI.showHiZ && !m_HiZ.debugMipTextures.empty()) {
        for (uint32_t mip = 0; mip < m_HiZ.numMips; mip++) {
            m_CommandList->copyTexture(
                m_HiZ.debugMipTextures[mip], nvrhi::TextureSlice(),
                m_HiZ.hizTexture,           nvrhi::TextureSlice().setMipLevel(mip));
        }
    }
}

// ===========================================================================
// SDSM — GPU cascade construction
// ===========================================================================

namespace {
    struct SDSMInput {
        dm::float4x4 worldToLight;
        dm::float4x4 viewToWorldToLight;
        dm::float4   sceneBboxMinLS;
        dm::float4   sceneBboxMaxLS;
        float        tanHalfFovX;
        float        tanHalfFovY;
        float        projA;
        float        projB;
        float        regionEnvelopeNear;
        float        regionEnvelopeFar;
        float        cameraNearPlane;
        uint32_t     shadowRes;
        uint32_t     maxHiZMip;
        float        pssmLambda;
        float        _pad[2];
    };

    struct SDSMCascadeOut {
        dm::float4x4 lightViewProj[Render::c_NumCascades];
        dm::float4   cascadeSplits;
        dm::float4   shadowCasterMinLS[Render::c_NumCascades];
        dm::float4   shadowCasterMaxLS[Render::c_NumCascades];
    };
}

void ComputeRenderPass::_ComputeRegionEnvelope(const dm::frustum& viewFrustum,
                                                    const dm::float3& camPos,
                                                    const dm::float3& camDir,
                                                    float& outNearZ, float& outFarZ) const {
    outNearZ = std::numeric_limits<float>::max();
    outFarZ  = std::numeric_limits<float>::lowest();
    bool any = false;

    for (const auto& region : m_Registry.getRegions()) {
        if (!viewFrustum.intersectsWith(region.cullBox)) continue;
        for (int i = 0; i < dm::box3::numCorners; i++) {
            dm::float3 corner = region.cullBox.getCorner(i);
            float vsZ = dm::dot(corner - camPos, camDir);
            outNearZ = dm::min(outNearZ, vsZ);
            outFarZ  = dm::max(outFarZ,  vsZ);
            any = true;
        }
    }

    if (!any) {
        outNearZ = 0.1f;
        outFarZ  = 1.f;
    }
    outNearZ = dm::max(outNearZ, 0.1f);
    outFarZ  = dm::max(outFarZ,  outNearZ + 1.f);
}

void ComputeRenderPass::_RunSDSMBuildCascades(const dm::box3& sceneBbox,
                                                   float aspectRatio, float fovY,
                                                   float regionEnvelopeNear,
                                                   float regionEnvelopeFar) {
    // Pack invariants into SDSMInput CB
    SDSMInput input{};

    dm::affine3 worldToLightAff = m_ViewHandler.worldToLight;
    dm::affine3 viewToWorldToLightAff = m_ViewHandler.view.GetInverseViewMatrix() * worldToLightAff;

    input.worldToLight       = dm::affineToHomogeneous(worldToLightAff);
    input.viewToWorldToLight = dm::affineToHomogeneous(viewToWorldToLightAff);

    dm::box3 sceneBboxLS = sceneBbox * worldToLightAff;
    input.sceneBboxMinLS = dm::float4(sceneBboxLS.m_mins, 0.f);
    input.sceneBboxMaxLS = dm::float4(sceneBboxLS.m_maxs, 0.f);

    float tanHalfFovY = std::tanf(fovY * 0.5f);
    input.tanHalfFovX = tanHalfFovY * aspectRatio;
    input.tanHalfFovY = tanHalfFovY;

    // Depth -> viewZ conversion: viewZ = projB / (depth - projA)
    dm::float4x4 proj = m_ViewHandler.view.GetProjectionMatrix();
    input.projA = proj.m_data[2 * 4 + 2];  // proj[2][2]
    input.projB = proj.m_data[3 * 4 + 2];  // proj[3][2]

    input.regionEnvelopeNear = regionEnvelopeNear;
    input.regionEnvelopeFar  = regionEnvelopeFar;
    input.cameraNearPlane    = 0.1f;
    input.shadowRes          = k_ShadowRes;
    input.maxHiZMip          = (m_HiZ.numMips > 0) ? (m_HiZ.numMips - 1) : 0;
    input.pssmLambda         = m_UI.pssmLambda;

    m_CommandList->writeBuffer(m_SDSM.inputCB, &input, sizeof(SDSMInput));

    // Dispatch — one threadgroup, NUM_CASCADES threads
    {
        nvrhi::ComputeState cs;
        cs.pipeline = m_SDSM.buildPipeline;
        cs.bindings = { m_SDSM.buildBindingSet };
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch(1, 1, 1);
    }

    // Copy cascade fields into the main CB at their CullConstantBufferEntry offsets.
    constexpr size_t kOffLightViewProj     = offsetof(Render::CullConstantBufferEntry, lightViewProj);
    constexpr size_t kOffCascadeSplits     = offsetof(Render::CullConstantBufferEntry, cascadeSplits);
    constexpr size_t kOffShadowCasterMin   = offsetof(Render::CullConstantBufferEntry, shadowCasterMinLS);
    constexpr size_t kOffShadowCasterMax   = offsetof(Render::CullConstantBufferEntry, shadowCasterMaxLS);

    constexpr size_t kSrcLightViewProj     = offsetof(SDSMCascadeOut, lightViewProj);
    constexpr size_t kSrcCascadeSplits     = offsetof(SDSMCascadeOut, cascadeSplits);
    constexpr size_t kSrcShadowCasterMin   = offsetof(SDSMCascadeOut, shadowCasterMinLS);
    constexpr size_t kSrcShadowCasterMax   = offsetof(SDSMCascadeOut, shadowCasterMaxLS);

    m_CommandList->copyBuffer(m_Shared.constantBuffer, kOffLightViewProj,
                              m_SDSM.cascadeDataBuffer, kSrcLightViewProj,
                              sizeof(dm::float4x4) * Render::c_NumCascades);
    m_CommandList->copyBuffer(m_Shared.constantBuffer, kOffCascadeSplits,
                              m_SDSM.cascadeDataBuffer, kSrcCascadeSplits,
                              sizeof(dm::float4));
    m_CommandList->copyBuffer(m_Shared.constantBuffer, kOffShadowCasterMin,
                              m_SDSM.cascadeDataBuffer, kSrcShadowCasterMin,
                              sizeof(dm::float4) * Render::c_NumCascades);
    m_CommandList->copyBuffer(m_Shared.constantBuffer, kOffShadowCasterMax,
                              m_SDSM.cascadeDataBuffer, kSrcShadowCasterMax,
                              sizeof(dm::float4) * Render::c_NumCascades);
}

// ===========================================================================
// Destructor
// ===========================================================================

ComputeRenderPass::~ComputeRenderPass() {
    
}

// ===========================================================================
// Init
// ===========================================================================

bool ComputeRenderPass::Init() {
    engine::CommonRenderPasses commonPasses(GetDevice(), m_ShaderFactory);

    {
        nvrhi::CommandListHandle initCL = GetDevice()->createCommandList();
        initCL->open();

        _UploadAllAssets(GetDevice(), initCL);

        if (!_InitShared())                                return false;
        if (!_InitShadowPass())                            return false;
        if (!_InitTreePass(initCL, commonPasses))          return false;
        if (!_InitImpostorPass())                          return false;
        if (!_BakeImpostors(initCL))                       return false;
        if (!_InitTerrainPass(initCL))                     return false;
        if (!_InitSkyPass())                               return false;
        if (!_InitHiZShaders())                            return false;
        if (!_InitSDSMPass())                              return false;

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

    m_UI.totalInstanceCount = m_Registry.totalInstanceCount();

    return true;
}

// ===========================================================================
// Animate
// ===========================================================================

void ComputeRenderPass::Animate(float seconds) {
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

void ComputeRenderPass::BackBufferResizing() {
    m_TreePass.pipeline          = nullptr;
    m_ImpostorPass.pipeline      = nullptr;
    m_TerrainPass.pipeline       = nullptr;
    m_ShadowPass.treePipeline    = nullptr;
    m_ShadowPass.terrainPipeline = nullptr;
    m_SkyPass.pipeline           = nullptr;

    // Hi-Z resources are resolution-dependent — force recreation
    m_DepthPrepass.treePipeline    = nullptr;
    m_DepthPrepass.terrainPipeline = nullptr;
    m_DepthPrepass.depthTexture    = nullptr;
    m_DepthPrepass.framebuffer     = nullptr;
    m_DepthPrepass.bindingSet      = nullptr;

    m_HiZ.buildBindingSets.clear();
    m_HiZ.debugMipTextures.clear();
    m_HiZ.numMips = 0;
    m_UI.hizMipTextures.clear();
}

// ===========================================================================
// Hot-reload callbacks
// ===========================================================================

void ComputeRenderPass::onAssetsDirty(const std::vector<size_t>& dirtyAssetIndices) {
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
    _BakeImpostors(cl);
    _RebuildCullBindings();

    cl->close();
    GetDevice()->executeCommandList(cl);
}

void ComputeRenderPass::onRegionsDirty(const std::vector<size_t>& /*dirtyRegionIndices*/) {
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

void ComputeRenderPass::Render(nvrhi::IFramebuffer* framebuffer) {
    m_UI.shadowMapTexture = m_ShadowPass.depthTexture.Get();
    m_UI.shadowCascadeTextures.resize(Render::c_NumCascades);
    for (uint32_t c = 0; c < Render::c_NumCascades; c++)
        m_UI.shadowCascadeTextures[c] = m_ShadowPass.debugTextures[c].Get();
    // Populate Hi-Z debug pointers for the UI (pointers are stable between resizes)
    m_UI.hizMipTextures.resize(m_HiZ.debugMipTextures.size());
    for (size_t i = 0; i < m_HiZ.debugMipTextures.size(); i++)
        m_UI.hizMipTextures[i] = m_HiZ.debugMipTextures[i].Get();
    // Impostor atlas debug pointers. These are stable preview textures; the
    // selected array slice is copied into them at the end of Render.
    m_UI.impostorAssetCount    = static_cast<uint32_t>(m_GPUAssets.size());
    m_UI.impostorViewsPerAsset = k_ImpostorViewCount;
    m_UI.impostorAzimuthViews  = k_ImpostorAzimuthViews;
    m_UI.impostorElevationViews = k_ImpostorElevationViews;
    m_UI.impostorAlbedoTexture = m_ImpostorPass.debugAlbedoTexture;
    m_UI.impostorNormalTexture = m_ImpostorPass.debugNormalTexture;
    m_UI.impostorDepthTexture = m_ImpostorPass.debugDepthTexture;
    m_UI.impostorAlbedoAtlasTexture = m_ImpostorPass.debugAlbedoAtlasTexture;
    m_UI.impostorNormalAtlasTexture = m_ImpostorPass.debugNormalAtlasTexture;
    m_UI.impostorDepthAtlasTexture = m_ImpostorPass.debugDepthAtlasTexture;

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

    // m_CommandList->beginMarker("Frame");

    m_ViewHandler.view.SetViewMatrix(m_ViewHandler.camera.GetWorldToViewMatrix());
    m_ViewHandler.view.UpdateCache();

    nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));
    #if XYLEM_USE_REVERSE_Z
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer, 0.f, 0);
    #else
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer,
        m_ViewHandler.view.GetProjectionFrustum().farPlane().distance, 0);
    #endif

    dm::box3 sceneBbox = dm::box3::empty();
    for (const auto& region : m_Registry.getRegions())
        sceneBbox |= region.cullBox;
    const auto* terrain = m_Registry.getTerrain();
    if (terrain)
        sceneBbox |= terrain->getBbox();

    // Compute max shadow distance from scene bbox (same logic as P0)
    const dm::float3& camPos = m_ViewHandler.camera.GetPosition();
    const dm::float3& camDir = m_ViewHandler.camera.GetDir();
    float maxShadowDist = 0.f;
    for (int i = 0; i < dm::box3::numCorners; i++) {
        dm::float3 corner = sceneBbox.getCorner(i);
        dm::float3 camToCorner = corner - camPos;
        float cornerDir = dm::dot(camToCorner, camDir);
        if (cornerDir > 0.0f) maxShadowDist = dm::max(maxShadowDist, cornerDir);
    }
    maxShadowDist = dm::max(maxShadowDist, 1.f);

    float aspectRatio = float(fbinfo.width) / float(fbinfo.height);
    m_ViewHandler.computeCascades(sceneBbox, m_Registry.getSunDirection(),
        0.1f, maxShadowDist, aspectRatio, dm::radians(60.f), k_ShadowRes, m_UI.pssmLambda);

    // --- Fill CullConstantBufferEntry ---
    Render::CullConstantBufferEntry constants{};
    constants.viewProj    = m_ViewHandler.view.GetViewProjectionMatrix();
    constants.viewMatrix  = dm::affineToHomogeneous(m_ViewHandler.view.GetViewMatrix());
    for (uint32_t c = 0; c < Render::c_NumCascades; c++)
        constants.lightViewProj[c] = m_ViewHandler.cascades[c].lightViewProj;
    constants.sunLightDir   = m_Registry.getSunDirection();
    constants.cascadeSplits = m_ViewHandler.cascadeSplitDistances;

    constants.viewFrustum  = m_ViewHandler.view.GetViewFrustum();
    constants.worldToLight = dm::affineToHomogeneous(m_ViewHandler.worldToLight);
    for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
        const dm::box3& cBbox = m_ViewHandler.cascades[c].shadowCasterBboxLS;
        if (cBbox.isempty()) {
            // Empty cascade — CullCS detects this via (min > max)
            constants.shadowCasterMinLS[c] = dm::float4( 1e30f,  1e30f,  1e30f, 0.f);
            constants.shadowCasterMaxLS[c] = dm::float4(-1e30f, -1e30f, -1e30f, 0.f);
        } else {
            constants.shadowCasterMinLS[c] = dm::float4(cBbox.m_mins, 0.f);
            constants.shadowCasterMaxLS[c] = dm::float4(cBbox.m_maxs, 0.f);
        }
    }

    constants.numRegions = static_cast<uint32_t>(m_Registry.getRegions().size());

    constants.cameraPos     = m_ViewHandler.camera.GetPosition();
    constants.totalCapacity = m_TotalCapacity;

    const auto& lodDistances = m_Registry.getLodDistances();
    constants.numLods = static_cast<uint32_t>(m_Registry.getLodSegments().size());
    for (uint32_t i = 0; i < constants.numLods; i++)
        constants.lodDistances[i].x = lodDistances[i];

    // Hi-Z fields — bypass when camera looks steeply downward (bird's-eye view)
    float downwardness = -m_ViewHandler.camera.GetDir().y;  // 0 = horizontal, 1 = straight down
    bool hizActive = downwardness < m_UI.hizBypassAngle;
    m_UI.hizActiveThisFrame = hizActive;

    constants.hizDimensions = dm::float2(static_cast<float>(fbinfo.width),
                                         static_cast<float>(fbinfo.height));
    constants.maxHiZMip     = static_cast<float>(m_HiZ.numMips - 1);
    constants.hizEnabled    = hizActive ? 1u : 0u;
    constants.impostorAlphaClip = m_UI.impostorAlphaClip;

    m_CommandList->writeBuffer(m_Shared.constantBuffer, &constants, Render::c_CullConstantBufferSize);

    // --- Ensure Hi-Z resources at current resolution ---
    _EnsureHiZResources(fbinfo.width, fbinfo.height);

    // --- Depth prepass + Hi-Z build (skip entirely when bypassed) ---
    if (hizActive) {
        m_CommandList->beginMarker("HiZ");

        m_CommandList->beginMarker("DepthPrepass");
        _RenderDepthPrepass();
        m_CommandList->endMarker();

        m_CommandList->beginMarker("BuildMipChain");
        _BuildHiZMipChain();
        m_CommandList->endMarker();

        m_CommandList->endMarker(); // HiZ

        // SDSM — GPU replaces cascade fields in the shared CB from the reduced depth.
        m_CommandList->beginMarker("SDSM");
        float regionNear, regionFar;
        _ComputeRegionEnvelope(m_ViewHandler.view.GetViewFrustum(), camPos, camDir,
                               regionNear, regionFar);
        _RunSDSMBuildCascades(sceneBbox, aspectRatio, dm::radians(60.f),
                              regionNear, regionFar);
        m_CommandList->endMarker();
    }

    // --- GPU Cull Dispatch ---
    // Clear UAV counters (AFTER depth prepass which reads last frame's data)
    m_CommandList->clearBufferUInt(m_CullPass.countBuffer, 0);
    m_CommandList->clearBufferUInt(m_CullPass.impostorCountBuffer, 0);
    m_CommandList->clearBufferUInt(m_CullPass.shadowCountBuffer, 0);
    m_CommandList->clearBufferUInt(m_CullPass.shadowUniqueCounter, 0);

    // Reset indirect args buffers (re-upload staging with instanceCount=0)
    m_CommandList->writeBuffer(m_CullPass.indirectArgsBuffer,
        m_IndirectArgsStaging.data(),
        m_IndirectArgsStaging.size() * sizeof(nvrhi::DrawIndexedIndirectArguments));
    m_CommandList->writeBuffer(m_CullPass.impostorIndirectArgsBuffer,
        m_ImpostorIndirectArgsStaging.data(),
        m_ImpostorIndirectArgsStaging.size() * sizeof(nvrhi::DrawIndirectArguments));
    m_CommandList->writeBuffer(m_CullPass.shadowIndirectArgsBuffer,
        m_ShadowIndirectArgsStaging.data(),
        m_ShadowIndirectArgsStaging.size() * sizeof(nvrhi::DrawIndexedIndirectArguments));

    m_CommandList->beginMarker("Cull");

    // region camera cull
    m_CommandList->beginMarker("RegionDispatch");
    {
        nvrhi::ComputeState cs;
        cs.pipeline = m_CullPass.regionPipeline;
        cs.bindings = { m_CullPass.bindingSet };
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch(
            (constants.numRegions + 63) / 64, 1, 1);
    }
    m_CommandList->endMarker();

    // Main camera cull
    m_CommandList->beginMarker("MainDispatch");
    {
        nvrhi::ComputeState cs;
        cs.pipeline = m_CullPass.mainPipeline;
        cs.bindings = { m_CullPass.bindingSet };
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch(
            (m_TotalCapacity + 255) / 256, 1, 1);
    }
    m_CommandList->endMarker();

    // Shadow cull
    m_CommandList->beginMarker("ShadowDispatch");
    {
        nvrhi::ComputeState cs;
        cs.pipeline = m_CullPass.shadowPipeline;
        cs.bindings = { m_CullPass.bindingSet };
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch(
            (m_TotalCapacity + 255) / 256, 1, 1);
    }
    m_CommandList->endMarker();

    m_CommandList->endMarker(); // Cull

    // --- Copy cull counts to readback ring ---
    {
        uint32_t ringSlot   = m_ReadbackFrameIndex % k_QueuedFrames;
        uint64_t countSize  = m_ReadbackCountEntries  * sizeof(uint32_t);
        uint64_t impostorSize = m_ReadbackImpostorEntries * sizeof(uint32_t);
        uint64_t shadowSize   = m_ReadbackShadowEntries * sizeof(uint32_t);

        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], 0,
                                  m_CullPass.countBuffer, 0, countSize);
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], countSize,
                                  m_CullPass.impostorCountBuffer, 0, impostorSize);
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], countSize + impostorSize,
                                  m_CullPass.shadowCountBuffer, 0, shadowSize);
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], countSize + impostorSize + shadowSize,
                                  m_CullPass.shadowUniqueCounter, 0, sizeof(uint32_t));
    }

    // --- Draw passes (auto barriers: UAV→SRV transitions handled by setGraphicsState) ---
    m_CommandList->beginMarker("Draw");

    m_CommandList->beginMarker("Sky");
    _RenderSkyPass(framebuffer);
    m_CommandList->endMarker();

    m_CommandList->beginMarker("Shadow");
    _RenderShadowPass();
    m_CommandList->endMarker();

    m_CommandList->beginMarker("Scene");
    _RenderScenePass(framebuffer);
    m_CommandList->endMarker();

    m_CommandList->beginMarker("Impostors");
    _RenderImpostorPass(framebuffer);
    m_CommandList->endMarker();

    m_CommandList->endMarker(); // Draw

    if (m_UI.showImpostorAtlas
        && m_ImpostorPass.albedoAlphaTexture
        && m_ImpostorPass.normalTexture
        && m_ImpostorPass.depthTexture
        && m_ImpostorPass.debugAlbedoTexture
        && m_ImpostorPass.debugNormalTexture
        && m_ImpostorPass.debugDepthTexture
        && m_ImpostorPass.debugAlbedoAtlasTexture
        && m_ImpostorPass.debugNormalAtlasTexture
        && m_ImpostorPass.debugDepthAtlasTexture)
    {
        const uint32_t assetIdx = std::min(
            m_UI.impostorSelectedAsset,
            m_GPUAssets.empty() ? 0u : static_cast<uint32_t>(m_GPUAssets.size() - 1));
        const uint32_t azIdx = std::min(m_UI.impostorSelectedAzimuth, k_ImpostorAzimuthViews - 1);
        const uint32_t elIdx = std::min(m_UI.impostorSelectedElevation, k_ImpostorElevationViews - 1);
        const uint32_t viewIdx = elIdx * k_ImpostorAzimuthViews + azIdx;
        const uint32_t slice = assetIdx * k_ImpostorViewCount + viewIdx;

        m_CommandList->copyTexture(
            m_ImpostorPass.debugAlbedoTexture, nvrhi::TextureSlice(),
            m_ImpostorPass.albedoAlphaTexture, nvrhi::TextureSlice().setArraySlice(slice));
        m_CommandList->copyTexture(
            m_ImpostorPass.debugNormalTexture, nvrhi::TextureSlice(),
            m_ImpostorPass.normalTexture, nvrhi::TextureSlice().setArraySlice(slice));
        m_CommandList->copyTexture(
            m_ImpostorPass.debugDepthTexture, nvrhi::TextureSlice(),
            m_ImpostorPass.depthTexture, nvrhi::TextureSlice().setArraySlice(slice));

        for (uint32_t view = 0; view < k_ImpostorViewCount; view++) {
            const uint32_t tileX = view % k_ImpostorAzimuthViews;
            const uint32_t tileY = view / k_ImpostorAzimuthViews;
            const uint32_t tileSlice = assetIdx * k_ImpostorViewCount + view;
            nvrhi::TextureSlice dst = nvrhi::TextureSlice()
                .setOrigin(tileX * k_ImpostorBakeResolution, tileY * k_ImpostorBakeResolution, 0)
                .setWidth(k_ImpostorBakeResolution)
                .setHeight(k_ImpostorBakeResolution);

            m_CommandList->copyTexture(
                m_ImpostorPass.debugAlbedoAtlasTexture, dst,
                m_ImpostorPass.albedoAlphaTexture, nvrhi::TextureSlice().setArraySlice(tileSlice));
            m_CommandList->copyTexture(
                m_ImpostorPass.debugNormalAtlasTexture, dst,
                m_ImpostorPass.normalTexture, nvrhi::TextureSlice().setArraySlice(tileSlice));
            m_CommandList->copyTexture(
                m_ImpostorPass.debugDepthAtlasTexture, dst,
                m_ImpostorPass.depthTexture, nvrhi::TextureSlice().setArraySlice(tileSlice));
        }

        m_CommandList->setTextureState(
            m_ImpostorPass.debugAlbedoTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
        m_CommandList->setTextureState(
            m_ImpostorPass.debugNormalTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
        m_CommandList->setTextureState(
            m_ImpostorPass.debugDepthTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
        m_CommandList->setTextureState(
            m_ImpostorPass.debugAlbedoAtlasTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
        m_CommandList->setTextureState(
            m_ImpostorPass.debugNormalAtlasTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
        m_CommandList->setTextureState(
            m_ImpostorPass.debugDepthAtlasTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
        m_CommandList->setTextureState(
            m_ImpostorPass.albedoAlphaTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
        m_CommandList->setTextureState(
            m_ImpostorPass.normalTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
        m_CommandList->setTextureState(
            m_ImpostorPass.depthTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
        m_CommandList->commitBarriers();
    }

    // m_CommandList->endMarker(); // Frame

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

            uint32_t impostorVisibleSum = 0;
            const uint32_t impostorOffset = m_ReadbackCountEntries;
            for (uint32_t i = 0; i < m_ReadbackImpostorEntries; i++)
                impostorVisibleSum += counts[impostorOffset + i];

            uint32_t shadowVisSum = 0;
            const uint32_t shadowOffset = impostorOffset + m_ReadbackImpostorEntries;
            for (uint32_t i = 0; i < m_ReadbackShadowEntries; i++)
                shadowVisSum += counts[shadowOffset + i];

            uint32_t shadowUnique = counts[shadowOffset + m_ReadbackShadowEntries];

            GetDevice()->unmapBuffer(m_ReadbackBuffers[readSlot]);

            const uint32_t visibleTotal = visibleSum + impostorVisibleSum;
            m_UI.visibleInstanceCount = visibleTotal;
            m_UI.impostorVisibleCount = impostorVisibleSum;
            m_UI.culledInstanceCount  = (visibleTotal <= m_UI.totalInstanceCount)
                ? m_UI.totalInstanceCount - visibleTotal : 0;
            m_UI.shadowVisibleCount     = shadowUnique;
            m_UI.shadowCulledCount      = (shadowUnique <= m_UI.totalInstanceCount)
                ? m_UI.totalInstanceCount - shadowUnique : 0;
            m_UI.shadowCascadeDrawCount = shadowVisSum;
            m_UI.shadowOverdrawCount    = (shadowVisSum >= shadowUnique)
                ? shadowVisSum - shadowUnique : 0;
        }
    }
    m_ReadbackFrameIndex++;

    cpuTimer.Stop();
    m_UI.cpuRenderTimeMs = (float)cpuTimer.Milliseconds();
}

// ===========================================================================
// Sky pass (identical to P0)
// ===========================================================================

void ComputeRenderPass::_RenderSkyPass(nvrhi::IFramebuffer* framebuffer) {
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

void ComputeRenderPass::_RenderShadowPass() {
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
            pso, m_ShadowPass.framebuffers[0]->getFramebufferInfo());
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
            pso, m_ShadowPass.framebuffers[0]->getFramebufferInfo());
    }

    nvrhi::Viewport shadowVP(static_cast<float>(k_ShadowRes), static_cast<float>(k_ShadowRes));
    nvrhi::ViewportState shadowVPState;
    shadowVPState.addViewportAndScissorRect(shadowVP);

    const uint32_t numLods   = static_cast<uint32_t>(m_Registry.getLodSegments().size());
    const uint32_t lowestLOD = numLods > 0 ? numLods - 1 : 0;
    const auto* terrainPtr = m_Registry.getTerrain();
    const auto& cascades   = m_ViewHandler.cascades;

    for (uint32_t cascade = 0; cascade < Render::c_NumCascades; cascade++) {
        const auto& cBbox = cascades[cascade].shadowCasterBboxLS;
        if (cBbox.isempty()) continue;

        nvrhi::utils::ClearDepthStencilAttachment(m_CommandList,
            m_ShadowPass.framebuffers[cascade], 1.0f, 0);

        // Trees: one indirect draw per asset, reading its per-(asset × cascade)
        // window in shadowVisBuf via shadowSlotOffsets[ai * NUM_CASCADES + cascade].
        nvrhi::GraphicsState shadowState;
        shadowState.pipeline    = m_ShadowPass.treePipeline;
        shadowState.framebuffer = m_ShadowPass.framebuffers[cascade];
        shadowState.viewport    = shadowVPState;
        shadowState.bindings    = { m_ShadowPass.bindingSet };
        shadowState.indirectParams = m_CullPass.shadowIndirectArgsBuffer;

        std::string marker = "C" + std::to_string(cascade);
        m_CommandList->beginMarker(marker.c_str());
        for (uint32_t ai = 0; ai < m_GPUAssets.size(); ai++) {
            uint32_t maxCount = m_MaxSlotCounts[ai * numLods];  // livePerAsset[ai]
            if (maxCount == 0) continue;

            const auto& lod = m_GPUAssets[ai].lods[lowestLOD];

            shadowState.vertexBuffers = { { lod.vbs.position, 0, 0 } };
            shadowState.indexBuffer   = { lod.indexBuffer, nvrhi::Format::R32_UINT, 0 };
            m_CommandList->setGraphicsState(shadowState);

            uint32_t pushConstants[2] = { ai, cascade };
            m_CommandList->setPushConstants(pushConstants, sizeof(pushConstants));

            uint32_t slot = ai * Render::c_NumCascades + cascade;

            std::string marker = "A" + std::to_string(ai);
            m_CommandList->beginMarker(marker.c_str());
            m_CommandList->drawIndexedIndirect(
                slot * sizeof(nvrhi::DrawIndexedIndirectArguments));
            m_CommandList->endMarker();
        }
        m_CommandList->endMarker();

        // Terrain shadow — only if terrain bbox intersects this cascade's LS bbox.
        if (m_TerrainPass.indexCount > 0 && terrainPtr
            && (terrainPtr->getBbox() * m_ViewHandler.worldToLight).intersects(cBbox)) {
            nvrhi::GraphicsState terrShadow;
            terrShadow.pipeline    = m_ShadowPass.terrainPipeline;
            terrShadow.framebuffer = m_ShadowPass.framebuffers[cascade];
            terrShadow.viewport    = shadowVPState;
            terrShadow.bindings    = { m_ShadowPass.bindingSet };
            terrShadow.vertexBuffers = { { m_TerrainPass.vertexBuffer, 0, 0 } };
            terrShadow.indexBuffer   = { m_TerrainPass.indexBuffer, nvrhi::Format::R32_UINT, 0 };
            m_CommandList->setGraphicsState(terrShadow);

            // Terrain VS only reads cascadeIdx; assetIndex unused but required for root sig.
            uint32_t pushConstants[2] = { 0u, cascade };
            m_CommandList->setPushConstants(pushConstants, sizeof(pushConstants));

            m_CommandList->drawIndexed(
                nvrhi::DrawArguments().setVertexCount(m_TerrainPass.indexCount));
        }
    }

    // Copy per-cascade slices into debug textures for UI display
    if (m_UI.showShadowMap) {
        for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
            m_CommandList->copyTexture(
                m_ShadowPass.debugTextures[c], nvrhi::TextureSlice(),
                m_ShadowPass.depthTexture, nvrhi::TextureSlice().setArraySlice(c));
        }
    }
}

// ===========================================================================
// Scene pass — GPU cull results, visibility buffer indirection in VS
// ===========================================================================

void ComputeRenderPass::_RenderScenePass(nvrhi::IFramebuffer* framebuffer) {
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
            state.vertexBuffers = {
                { lod.vbs.position,  0, 0 },
                { lod.vbs.normal,    1, 0 },
                { lod.vbs.tangent,   2, 0 },
                { lod.vbs.bitangent, 3, 0 },
                { lod.vbs.uv,        4, 0 },
            };
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

void ComputeRenderPass::_RenderImpostorPass(nvrhi::IFramebuffer* framebuffer) {
    if (m_GPUAssets.empty() || m_ImpostorPass.bindingSets.empty())
        return;

    const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();

    if (!m_ImpostorPass.pipeline) {
        nvrhi::GraphicsPipelineDesc psoDesc;
        psoDesc.VS = m_ImpostorPass.vertexShader;
        psoDesc.PS = m_ImpostorPass.pixelShader;
        psoDesc.inputLayout = nullptr;
        psoDesc.bindingLayouts = { m_ImpostorPass.bindingLayout };
        psoDesc.primType = nvrhi::PrimitiveType::TriangleStrip;
    #if XYLEM_USE_REVERSE_Z
        psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
    #else
        psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    #endif
        psoDesc.renderState.rasterState.setCullNone();
        // Push impostor fragments slightly toward camera in reverse-Z to win
        // z-fight against the terrain depth written earlier in the scene pass.
        // Slope-scaled bias auto-scales with view angle so a single setting
        // works at both close and far range.
        // Reverse-Z: positive bias pushes toward camera (higher NDC depth).
        // Enough to win against terrain without overriding closer geometry.
        psoDesc.renderState.rasterState
            .setDepthBias(16)
            .setSlopeScaleDepthBias(1.0f);
        m_ImpostorPass.pipeline = GetDevice()->createGraphicsPipeline(psoDesc, fbinfo);
    }

    nvrhi::GraphicsState state;
    state.pipeline = m_ImpostorPass.pipeline;
    state.framebuffer = framebuffer;
    state.viewport = m_ViewHandler.view.GetViewportState();
    state.indirectParams = m_CullPass.impostorIndirectArgsBuffer;

    for (uint32_t ai = 0; ai < m_GPUAssets.size(); ai++) {
        if (ai >= m_ImpostorMaxSlotCounts.size() || m_ImpostorMaxSlotCounts[ai] == 0)
            continue;

        state.bindings = { m_ImpostorPass.bindingSets[m_GPUAssets[ai].textureSetIdx] };
        m_CommandList->setGraphicsState(state);
        m_CommandList->setPushConstants(&ai, sizeof(ai));
        m_CommandList->drawIndirect(ai * sizeof(nvrhi::DrawIndirectArguments));
    }
}

// ===========================================================================
// Init helpers
// ===========================================================================

bool ComputeRenderPass::_InitShared() {
    m_Shared.constantBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(Render::c_CullConstantBufferSize)
            .setIsConstantBuffer(true)
            .setDebugName("CullConstantBuffer")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
    );
    return !!m_Shared.constantBuffer;
}

bool ComputeRenderPass::_InitCullPass(nvrhi::ICommandList* /*initCL*/) {
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
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(5), // impostorSlotOffsets

        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0), // mainRegionVisBuf
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(1), // mainSlotCountBuf
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2), // mainVisBuf
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(3), // shadowSlotCountBuf
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(4), // shadowVisBuf

        nvrhi::BindingLayoutItem::RawBuffer_UAV(5),        // mainIndirectArgs
        nvrhi::BindingLayoutItem::RawBuffer_UAV(6),        // shadowIndirectArgs
        nvrhi::BindingLayoutItem::RawBuffer_UAV(7),        // shadowUniqueCounter
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(8), // impostorSlotCountBuf
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(9), // impostorVisBuf
        nvrhi::BindingLayoutItem::RawBuffer_UAV(10),       // impostorIndirectArgs

        nvrhi::BindingLayoutItem::Texture_SRV(4),          // hizTexture
        nvrhi::BindingLayoutItem::Sampler(0),              // hizSampler (point/clamp)
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

bool ComputeRenderPass::_InitTreePass(nvrhi::ICommandList* initCL, engine::CommonRenderPasses& commonPasses) {
    m_TreePass.vertexShader = m_ShaderFactory->CreateShader(
        "app/ComputeRenderPass.hlsl", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_TreePass.pixelShader  = m_ShaderFactory->CreateShader(
        "app/ComputeRenderPass.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_TreePass.vertexShader || !m_TreePass.pixelShader) return false;

    nvrhi::VertexAttributeDesc attributes[] = {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(0)
            .setElementStride(sizeof(dm::float3)),
        nvrhi::VertexAttributeDesc()
            .setName("NORMAL")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(1)
            .setElementStride(sizeof(dm::float3)),
        nvrhi::VertexAttributeDesc()
            .setName("TANGENT")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(2)
            .setElementStride(sizeof(dm::float3)),
        nvrhi::VertexAttributeDesc()
            .setName("BITANGENT")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(3)
            .setElementStride(sizeof(dm::float3)),
        nvrhi::VertexAttributeDesc()
            .setName("UV")
            .setFormat(nvrhi::Format::RG32_FLOAT)
            .setOffset(0)
            .setBufferIndex(4)
            .setElementStride(sizeof(dm::float2)),
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

bool ComputeRenderPass::_InitImpostorPass() {
    m_ImpostorPass.vertexShader = m_ShaderFactory->CreateShader(
        "app/ImpostorRenderPass.hlsl", "impostor_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_ImpostorPass.pixelShader = m_ShaderFactory->CreateShader(
        "app/ImpostorRenderPass.hlsl", "impostor_ps", nullptr, nvrhi::ShaderType::Pixel);
    m_ImpostorPass.bakeVertexShader = m_ShaderFactory->CreateShader(
        "app/ImpostorBake.hlsl", "bake_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_ImpostorPass.bakePixelShader = m_ShaderFactory->CreateShader(
        "app/ImpostorBake.hlsl", "bake_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_ImpostorPass.vertexShader || !m_ImpostorPass.pixelShader
        || !m_ImpostorPass.bakeVertexShader || !m_ImpostorPass.bakePixelShader)
        return false;

    nvrhi::VertexAttributeDesc attributes[] = {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(0)
            .setElementStride(sizeof(dm::float3)),
        nvrhi::VertexAttributeDesc()
            .setName("NORMAL")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(1)
            .setElementStride(sizeof(dm::float3)),
        nvrhi::VertexAttributeDesc()
            .setName("TANGENT")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(2)
            .setElementStride(sizeof(dm::float3)),
        nvrhi::VertexAttributeDesc()
            .setName("BITANGENT")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(3)
            .setElementStride(sizeof(dm::float3)),
        nvrhi::VertexAttributeDesc()
            .setName("UV")
            .setFormat(nvrhi::Format::RG32_FLOAT)
            .setOffset(0)
            .setBufferIndex(4)
            .setElementStride(sizeof(dm::float2)),
    };
    m_ImpostorPass.bakeInputLayout = GetDevice()->createInputLayout(
        attributes, uint32_t(std::size(attributes)), m_ImpostorPass.bakeVertexShader);
    if (!m_ImpostorPass.bakeInputLayout) return false;

    nvrhi::BindingLayoutDesc layoutDesc;
    layoutDesc.visibility = nvrhi::ShaderType::All;
    layoutDesc.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(0),
        nvrhi::BindingLayoutItem::PushConstants(1, sizeof(uint32_t)),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0), // impostorVisBuf
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1), // instBuf
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2), // impostorSlotOffsets
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(3), // cullData
        nvrhi::BindingLayoutItem::Texture_SRV(4),          // impostor albedo+alpha
        nvrhi::BindingLayoutItem::Texture_SRV(5),          // impostor normal
        nvrhi::BindingLayoutItem::Texture_SRV(6),          // impostor depth
        nvrhi::BindingLayoutItem::Texture_SRV(7),          // shadow map
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(8), // assetDims (bake-space bbox halfExtents)
        nvrhi::BindingLayoutItem::Sampler(0),
        nvrhi::BindingLayoutItem::Sampler(1),
        nvrhi::BindingLayoutItem::Sampler(2),
    };
    m_ImpostorPass.bindingLayout = GetDevice()->createBindingLayout(layoutDesc);
    if (!m_ImpostorPass.bindingLayout) return false;

    m_ImpostorPass.sampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
            .setAllFilters(true));
    if (!m_ImpostorPass.sampler) return false;

    // Point-clamp sampler for the depth atlas. Bilinear filtering across
    // silhouette edges averages cleared background depth into surface depth,
    // producing fat banded SV_Depth artifacts on the parallax-displaced cards.
    m_ImpostorPass.depthSampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
            .setAllFilters(false));
    if (!m_ImpostorPass.depthSampler) return false;

    nvrhi::BindingLayoutDesc bakeLayoutDesc;
    bakeLayoutDesc.visibility = nvrhi::ShaderType::All;
    bakeLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
        nvrhi::BindingLayoutItem::Texture_SRV(0),
        nvrhi::BindingLayoutItem::Texture_SRV(1),
        nvrhi::BindingLayoutItem::Sampler(0),
    };
    m_ImpostorPass.bakeBindingLayout = GetDevice()->createBindingLayout(bakeLayoutDesc);
    if (!m_ImpostorPass.bakeBindingLayout) return false;

    m_ImpostorPass.bakeConstantBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(c_ImpostorBakeCBSize)
            .setIsConstantBuffer(true)
            .setIsVolatile(true)
            .setMaxVersions(std::max(1u, static_cast<uint32_t>(m_GPUAssets.size()) * k_ImpostorViewCount))
            .setDebugName("ImpostorBakeCB")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer));
    if (!m_ImpostorPass.bakeConstantBuffer) return false;

    return true;
}

bool ComputeRenderPass::_BakeImpostors(nvrhi::ICommandList* commandList) {
    if (!commandList || !m_ImpostorPass.bakeVertexShader || !m_ImpostorPass.bakePixelShader)
        return false;

    auto device = GetDevice();
    const uint32_t numAssets = static_cast<uint32_t>(m_GPUAssets.size());
    const uint32_t numSlices = std::max(1u, numAssets * k_ImpostorViewCount);

    if (!m_ImpostorPass.bakeConstantBuffer
        || m_ImpostorPass.bakeConstantBuffer->getDesc().maxVersions < numSlices)
    {
        m_ImpostorPass.bakeConstantBuffer = device->createBuffer(
            nvrhi::BufferDesc()
                .setByteSize(c_ImpostorBakeCBSize)
                .setIsConstantBuffer(true)
                .setIsVolatile(true)
                .setMaxVersions(numSlices)
                .setDebugName("ImpostorBakeCB")
                .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer));
        if (!m_ImpostorPass.bakeConstantBuffer)
            return false;
    }

    // Per-asset bake bounds data. Runtime virtual projection is centered on the
    // instance world bbox, so only half extents are needed on the GPU.
    {
        const dm::affine3 toImpostor = BuildtoObjectTransform();
        std::vector<dm::float4> assetDims(std::max(1u, numAssets), dm::float4(0.f));
        for (uint32_t ai = 0; ai < numAssets; ai++) {
            const auto& asset = m_GPUAssets[ai];
            if (asset.lods.empty()) continue;

            dm::box3 rotated = asset.lods[0].bbox * toImpostor;
            const dm::float3 half = rotated.diagonal() * 0.5f;
            assetDims[ai] = dm::float4(half, 0.f);
        }

        m_ImpostorPass.assetDimsBuffer = device->createBuffer(
            nvrhi::BufferDesc()
                .setByteSize(assetDims.size() * sizeof(dm::float4))
                .setStructStride(sizeof(dm::float4))
                .setDebugName("ImpostorAssetDims")
                .setCanHaveUAVs(false)
                .setInitialState(nvrhi::ResourceStates::CopyDest));
        if (!m_ImpostorPass.assetDimsBuffer)
            return false;

        commandList->beginTrackingBufferState(m_ImpostorPass.assetDimsBuffer, nvrhi::ResourceStates::CopyDest);
        commandList->writeBuffer(m_ImpostorPass.assetDimsBuffer,
            assetDims.data(), assetDims.size() * sizeof(dm::float4));
        commandList->setPermanentBufferState(m_ImpostorPass.assetDimsBuffer, nvrhi::ResourceStates::ShaderResource);
    }

    m_ImpostorPass.albedoAlphaTexture = device->createTexture(
        nvrhi::TextureDesc()
            .setWidth(k_ImpostorBakeResolution)
            .setHeight(k_ImpostorBakeResolution)
            .setArraySize(numSlices)
            .setDimension(nvrhi::TextureDimension::Texture2DArray)
            .setFormat(nvrhi::Format::RGBA8_UNORM)
            .setIsRenderTarget(true)
            .setClearValue(nvrhi::Color(0.f, 0.f, 0.f, 0.f))
            .enableAutomaticStateTracking(nvrhi::ResourceStates::RenderTarget)
            .setDebugName("ImpostorAlbedoAlphaArray"));

    m_ImpostorPass.normalTexture = device->createTexture(
        nvrhi::TextureDesc()
            .setWidth(k_ImpostorBakeResolution)
            .setHeight(k_ImpostorBakeResolution)
            .setArraySize(numSlices)
            .setDimension(nvrhi::TextureDimension::Texture2DArray)
            .setFormat(nvrhi::Format::RGBA8_UNORM)
            .setIsRenderTarget(true)
            .setClearValue(nvrhi::Color(0.5f, 0.5f, 1.f, 0.f))
            .enableAutomaticStateTracking(nvrhi::ResourceStates::RenderTarget)
            .setDebugName("ImpostorNormalArray"));

    m_ImpostorPass.depthTexture = device->createTexture(
        nvrhi::TextureDesc()
            .setWidth(k_ImpostorBakeResolution)
            .setHeight(k_ImpostorBakeResolution)
            .setArraySize(numSlices)
            .setDimension(nvrhi::TextureDimension::Texture2DArray)
            .setFormat(nvrhi::Format::D32)
            .setIsRenderTarget(true)
            .setClearValue(nvrhi::Color(1.f))
            .enableAutomaticStateTracking(nvrhi::ResourceStates::DepthWrite)
            .setDebugName("ImpostorDepthArray"));

    if (!m_ImpostorPass.albedoAlphaTexture || !m_ImpostorPass.normalTexture || !m_ImpostorPass.depthTexture)
        return false;

    m_ImpostorPass.bakeFramebuffers.resize(numSlices);
    for (uint32_t slice = 0; slice < numSlices; slice++) {
        nvrhi::FramebufferDesc fbDesc;
        fbDesc
            .addColorAttachment(nvrhi::FramebufferAttachment()
                .setTexture(m_ImpostorPass.albedoAlphaTexture)
                .setArraySlice(slice))
            .addColorAttachment(nvrhi::FramebufferAttachment()
                .setTexture(m_ImpostorPass.normalTexture)
                .setArraySlice(slice))
            .setDepthAttachment(nvrhi::FramebufferAttachment()
                .setTexture(m_ImpostorPass.depthTexture)
                .setArraySlice(slice));

        m_ImpostorPass.bakeFramebuffers[slice] = device->createFramebuffer(fbDesc);
        if (!m_ImpostorPass.bakeFramebuffers[slice])
            return false;
    }

    m_ImpostorPass.bakeBindingSets.resize(m_TreePass.textureSets.size());
    for (size_t i = 0; i < m_TreePass.textureSets.size(); i++) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_ImpostorPass.bakeConstantBuffer),
            nvrhi::BindingSetItem::Texture_SRV(0, m_TreePass.textureSets[i].diffuse),
            nvrhi::BindingSetItem::Texture_SRV(1, m_TreePass.textureSets[i].normalMap),
            nvrhi::BindingSetItem::Sampler(0, m_TreePass.sampler),
        };
        m_ImpostorPass.bakeBindingSets[i] = device->createBindingSet(bsd, m_ImpostorPass.bakeBindingLayout);
        if (!m_ImpostorPass.bakeBindingSets[i])
            return false;
    }

    nvrhi::GraphicsPipelineDesc psoDesc;
    psoDesc.VS = m_ImpostorPass.bakeVertexShader;
    psoDesc.PS = m_ImpostorPass.bakePixelShader;
    psoDesc.inputLayout = m_ImpostorPass.bakeInputLayout;
    psoDesc.bindingLayouts = { m_ImpostorPass.bakeBindingLayout };
    psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
    psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    psoDesc.renderState.rasterState.setCullNone();
    m_ImpostorPass.bakePipeline = device->createGraphicsPipeline(
        psoDesc, m_ImpostorPass.bakeFramebuffers[0]->getFramebufferInfo());
    if (!m_ImpostorPass.bakePipeline)
        return false;

    nvrhi::GraphicsState state;
    state.pipeline = m_ImpostorPass.bakePipeline;
    state.viewport.addViewportAndScissorRect(nvrhi::Viewport(
        float(k_ImpostorBakeResolution), float(k_ImpostorBakeResolution)));

    for (uint32_t ai = 0; ai < numAssets; ai++) {
        const auto& asset = m_GPUAssets[ai];
        if (asset.lods.empty() || asset.textureSetIdx >= m_ImpostorPass.bakeBindingSets.size())
            continue;

        const auto& lod = asset.lods[0];
        if (lod.indexCount == 0)
            continue;

        state.bindings = { m_ImpostorPass.bakeBindingSets[asset.textureSetIdx] };
        state.vertexBuffers = {
            { lod.vbs.position,  0, 0 },
            { lod.vbs.normal,    1, 0 },
            { lod.vbs.tangent,   2, 0 },
            { lod.vbs.bitangent, 3, 0 },
            { lod.vbs.uv,        4, 0 },
        };
        state.indexBuffer = { lod.indexBuffer, nvrhi::Format::R32_UINT, 0 };

        for (uint32_t vi = 0; vi < k_ImpostorViewCount; vi++) {
            const uint32_t slice = ai * k_ImpostorViewCount + vi;
            nvrhi::IFramebuffer* framebuffer = m_ImpostorPass.bakeFramebuffers[slice];

            nvrhi::utils::ClearColorAttachment(commandList, framebuffer, 0, nvrhi::Color(0.f, 0.f, 0.f, 0.f));
            nvrhi::utils::ClearColorAttachment(commandList, framebuffer, 1, nvrhi::Color(0.5f, 0.5f, 1.f, 0.f));
            nvrhi::utils::ClearDepthStencilAttachment(commandList, framebuffer, 1.f, 0);

            ImpostorBakeCB cb;
            cb.mvp = BuildtoProjTransform(lod.bbox, vi);
            commandList->writeBuffer(m_ImpostorPass.bakeConstantBuffer, &cb, sizeof(cb));

            state.framebuffer = framebuffer;
            commandList->setGraphicsState(state);
            commandList->drawIndexed(nvrhi::DrawArguments().setVertexCount(lod.indexCount));
        }
    }

    // Per-slice debug copies for ImGui inspection of the baked atlas. Mirrors the
    // shadow-cascade debug-texture pattern. Created here (after the bake) so the
    // copies happen while the array textures are still in RenderTarget/DepthWrite
    // state — copyTexture handles the transition.
    {
        m_ImpostorPass.debugAlbedoTexture = device->createTexture(
            nvrhi::TextureDesc()
                .setWidth(k_ImpostorBakeResolution).setHeight(k_ImpostorBakeResolution)
                .setFormat(nvrhi::Format::RGBA8_UNORM)
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true)
                .setDebugName("ImpostorAlbedoDebug"));
        m_ImpostorPass.debugNormalTexture = device->createTexture(
            nvrhi::TextureDesc()
                .setWidth(k_ImpostorBakeResolution).setHeight(k_ImpostorBakeResolution)
                .setFormat(nvrhi::Format::RGBA8_UNORM)
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true)
                .setDebugName("ImpostorNormalDebug"));
        m_ImpostorPass.debugDepthTexture = device->createTexture(
            nvrhi::TextureDesc()
                .setWidth(k_ImpostorBakeResolution).setHeight(k_ImpostorBakeResolution)
                .setFormat(nvrhi::Format::R32_FLOAT)
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true)
                .setDebugName("ImpostorDepthDebug"));
        m_ImpostorPass.debugAlbedoAtlasTexture = device->createTexture(
            nvrhi::TextureDesc()
                .setWidth(k_ImpostorBakeResolution * k_ImpostorAzimuthViews)
                .setHeight(k_ImpostorBakeResolution * k_ImpostorElevationViews)
                .setFormat(nvrhi::Format::RGBA8_UNORM)
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true)
                .setDebugName("ImpostorAlbedoAtlasDebug"));
        m_ImpostorPass.debugNormalAtlasTexture = device->createTexture(
            nvrhi::TextureDesc()
                .setWidth(k_ImpostorBakeResolution * k_ImpostorAzimuthViews)
                .setHeight(k_ImpostorBakeResolution * k_ImpostorElevationViews)
                .setFormat(nvrhi::Format::RGBA8_UNORM)
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true)
                .setDebugName("ImpostorNormalAtlasDebug"));
        m_ImpostorPass.debugDepthAtlasTexture = device->createTexture(
            nvrhi::TextureDesc()
                .setWidth(k_ImpostorBakeResolution * k_ImpostorAzimuthViews)
                .setHeight(k_ImpostorBakeResolution * k_ImpostorElevationViews)
                .setFormat(nvrhi::Format::R32_FLOAT)
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true)
                .setDebugName("ImpostorDepthAtlasDebug"));
        if (!m_ImpostorPass.debugAlbedoTexture
            || !m_ImpostorPass.debugNormalTexture
            || !m_ImpostorPass.debugDepthTexture
            || !m_ImpostorPass.debugAlbedoAtlasTexture
            || !m_ImpostorPass.debugNormalAtlasTexture
            || !m_ImpostorPass.debugDepthAtlasTexture)
            return false;

        commandList->copyTexture(
            m_ImpostorPass.debugAlbedoTexture, nvrhi::TextureSlice(),
            m_ImpostorPass.albedoAlphaTexture, nvrhi::TextureSlice());
        commandList->copyTexture(
            m_ImpostorPass.debugNormalTexture, nvrhi::TextureSlice(),
            m_ImpostorPass.normalTexture, nvrhi::TextureSlice());
        commandList->copyTexture(
            m_ImpostorPass.debugDepthTexture, nvrhi::TextureSlice(),
            m_ImpostorPass.depthTexture, nvrhi::TextureSlice());
    }

    commandList->setTextureState(m_ImpostorPass.debugAlbedoTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
    commandList->setTextureState(m_ImpostorPass.debugNormalTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
    commandList->setTextureState(m_ImpostorPass.debugDepthTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
    commandList->setTextureState(m_ImpostorPass.debugAlbedoAtlasTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
    commandList->setTextureState(m_ImpostorPass.debugNormalAtlasTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
    commandList->setTextureState(m_ImpostorPass.debugDepthAtlasTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
    commandList->setTextureState(m_ImpostorPass.albedoAlphaTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
    commandList->setTextureState(m_ImpostorPass.normalTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
    commandList->setTextureState(m_ImpostorPass.depthTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
    commandList->commitBarriers();

    m_ImpostorPass.pipeline = nullptr;
    return true;
}

// bool ComputeRenderPass::_InitTimerQueries() {
//     for (uint32_t i = 0; i < k_QueuedFrames; i++)
//         m_GpuTimers[i] = GetDevice()->createTimerQuery();
//     return true;
// }

bool ComputeRenderPass::_InitShadowPass() {
    m_ShadowPass.depthTexture = GetDevice()->createTexture(
        nvrhi::TextureDesc()
            .setWidth(k_ShadowRes).setHeight(k_ShadowRes)
            .setArraySize(Render::c_NumCascades)
            .setDimension(nvrhi::TextureDimension::Texture2DArray)
            .setFormat(nvrhi::Format::D32)
            .setIsRenderTarget(true)
            .setUseClearValue(true)
            .setClearValue(nvrhi::Color(1.f))
            .setInitialState(nvrhi::ResourceStates::DepthWrite)
            .setKeepInitialState(true)
            .setDebugName("ShadowMapArray")
    );
    if (!m_ShadowPass.depthTexture) return false;

    for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
        m_ShadowPass.framebuffers[c] = GetDevice()->createFramebuffer(
            nvrhi::FramebufferDesc().setDepthAttachment(
                nvrhi::FramebufferAttachment()
                    .setTexture(m_ShadowPass.depthTexture)
                    .setArraySlice(c)));
        if (!m_ShadowPass.framebuffers[c]) return false;

        m_ShadowPass.debugTextures[c] = GetDevice()->createTexture(
            nvrhi::TextureDesc()
                .setWidth(k_ShadowRes).setHeight(k_ShadowRes)
                .setFormat(nvrhi::Format::R32_FLOAT)
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true)
                .setDebugName("ShadowCascadeDebug_" + std::to_string(c))
        );
    }

    m_ShadowPass.treeVS = m_ShaderFactory->CreateShader(
        "app/shadow_compute.hlsl", "tree_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_ShadowPass.terrainVS = m_ShaderFactory->CreateShader(
        "app/shadow_compute.hlsl", "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    if (!m_ShadowPass.treeVS || !m_ShadowPass.terrainVS) return false;

    // Tree shadow input layout — position-only (instance data via SRV indirection)
    nvrhi::VertexAttributeDesc treeShadowAttrs[] = {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(0)
            .setElementStride(sizeof(dm::float3)),
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
        nvrhi::BindingLayoutItem::PushConstants(1, sizeof(uint32_t) * 2),  // {assetIndex, cascadeIdx}
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0),                 // shadowVisBuf
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),                 // instanceBuf
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2),                 // shadowSlotOffsets
    };
    m_ShadowPass.bindingLayout = GetDevice()->createBindingLayout(shadowLayoutDesc);
    if (!m_ShadowPass.bindingLayout) return false;

    // Binding set created in _RebuildCullBindings after buffers exist
    return true;
}

bool ComputeRenderPass::_InitTerrainPass(nvrhi::ICommandList* initCL) {
    const auto* terrain = m_Registry.getTerrain();
    if (!terrain) return true;

    const auto& verts   = terrain->getVertices();
    const auto& indices = terrain->getIndices();
    if (verts.empty() || indices.empty()) return true;

    m_TerrainPass.indexCount = static_cast<uint32_t>(indices.size());

    m_TerrainPass.vertexShader = m_ShaderFactory->CreateShader("app/terrain_compute.hlsl", "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_TerrainPass.pixelShader  = m_ShaderFactory->CreateShader("app/terrain_compute.hlsl", "terrain_ps", nullptr, nvrhi::ShaderType::Pixel);
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

bool ComputeRenderPass::_InitHiZShaders() {
    auto device = GetDevice();

    // --- Depth prepass shaders ---
    m_DepthPrepass.treeVS = m_ShaderFactory->CreateShader(
        "app/DepthPrepass.hlsl", "tree_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_DepthPrepass.terrainVS = m_ShaderFactory->CreateShader(
        "app/DepthPrepass.hlsl", "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    if (!m_DepthPrepass.treeVS || !m_DepthPrepass.terrainVS) return false;

    // Tree input layout — position-only
    nvrhi::VertexAttributeDesc treePrepassAttrs[] = {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(0)
            .setElementStride(sizeof(dm::float3)),
    };
    m_DepthPrepass.treeInputLayout = device->createInputLayout(
        treePrepassAttrs, uint32_t(std::size(treePrepassAttrs)), m_DepthPrepass.treeVS);
    if (!m_DepthPrepass.treeInputLayout) return false;

    // Terrain input layout — pos+normal+uv (must match VB stride)
    nvrhi::VertexAttributeDesc terrainPrepassAttrs[] = {
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
    m_DepthPrepass.terrainInputLayout = device->createInputLayout(
        terrainPrepassAttrs, uint32_t(std::size(terrainPrepassAttrs)), m_DepthPrepass.terrainVS);
    if (!m_DepthPrepass.terrainInputLayout) return false;

    // Depth prepass binding layout — same shape as shadow pass
    nvrhi::BindingLayoutDesc prepassLayoutDesc;
    prepassLayoutDesc.visibility = nvrhi::ShaderType::All;
    prepassLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(0),
        nvrhi::BindingLayoutItem::PushConstants(1, sizeof(uint32_t)),  // slot
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0),             // visBuf
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),             // instBuf
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2),             // slotOffsets
    };
    m_DepthPrepass.bindingLayout = device->createBindingLayout(prepassLayoutDesc);
    if (!m_DepthPrepass.bindingLayout) return false;

    // --- Hi-Z build shaders ---
    m_HiZ.copyCS = m_ShaderFactory->CreateShader(
        "app/HiZBuild.hlsl", "HiZCopy", nullptr, nvrhi::ShaderType::Compute);
    m_HiZ.buildCS = m_ShaderFactory->CreateShader(
        "app/HiZBuild.hlsl", "HiZDownsample", nullptr, nvrhi::ShaderType::Compute);
    if (!m_HiZ.copyCS || !m_HiZ.buildCS) return false;

    // Hi-Z build binding layout: push constants + SRV(source) + UAV(dest)
    nvrhi::BindingLayoutDesc hizBuildLayoutDesc;
    hizBuildLayoutDesc.visibility = nvrhi::ShaderType::Compute;
    hizBuildLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::PushConstants(0, sizeof(uint32_t) * 2),  // destDimensions
        nvrhi::BindingLayoutItem::Texture_SRV(0),                          // source
        nvrhi::BindingLayoutItem::Texture_UAV(0),                          // dest
    };
    m_HiZ.buildBindingLayout = device->createBindingLayout(hizBuildLayoutDesc);
    if (!m_HiZ.buildBindingLayout) return false;

    // Compute pipelines
    nvrhi::ComputePipelineDesc copyPso;
    copyPso.CS = m_HiZ.copyCS;
    copyPso.bindingLayouts = { m_HiZ.buildBindingLayout };
    m_HiZ.copyPipeline = device->createComputePipeline(copyPso);

    nvrhi::ComputePipelineDesc buildPso;
    buildPso.CS = m_HiZ.buildCS;
    buildPso.bindingLayouts = { m_HiZ.buildBindingLayout };
    m_HiZ.buildPipeline = device->createComputePipeline(buildPso);

    if (!m_HiZ.copyPipeline || !m_HiZ.buildPipeline) return false;

    // Point/clamp sampler for Hi-Z reads in the cull shader
    m_HiZ.pointSampler = device->createSampler(
        nvrhi::SamplerDesc()
            .setAllFilters(false)
            .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
    );
    if (!m_HiZ.pointSampler) return false;

    // Create 1x1 placeholder Hi-Z texture so binding sets can reference it before first frame
    m_HiZ.hizTexture = device->createTexture(
        nvrhi::TextureDesc()
            .setWidth(1).setHeight(1)
            .setMipLevels(1)
            .setFormat(nvrhi::Format::RG32_FLOAT)
            .setIsUAV(true)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true)
            .setDebugName("HiZTexture_Placeholder")
    );
    m_HiZ.numMips = 1;

    return true;
}

bool ComputeRenderPass::_InitSDSMPass() {
    auto device = GetDevice();

    m_SDSM.buildCS = m_ShaderFactory->CreateShader(
        "app/SDSMBuildCascades.hlsl", "BuildCascades", nullptr, nvrhi::ShaderType::Compute);
    if (!m_SDSM.buildCS) return false;

    nvrhi::BindingLayoutDesc layoutDesc;
    layoutDesc.visibility = nvrhi::ShaderType::Compute;
    layoutDesc.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(0),   // SDSMInput
        nvrhi::BindingLayoutItem::Texture_SRV(0),      // hizTexture
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0),  // cascade data out
    };
    m_SDSM.buildBindingLayout = device->createBindingLayout(layoutDesc);
    if (!m_SDSM.buildBindingLayout) return false;

    nvrhi::ComputePipelineDesc pso;
    pso.CS = m_SDSM.buildCS;
    pso.bindingLayouts = { m_SDSM.buildBindingLayout };
    m_SDSM.buildPipeline = device->createComputePipeline(pso);
    if (!m_SDSM.buildPipeline) return false;

    // Input CB (non-volatile, state-tracked so writeBuffer works inside Render)
    constexpr size_t kInputCBSize =
        (sizeof(SDSMInput) + (nvrhi::c_ConstantBufferOffsetSizeAlignment - 1))
        & ~(nvrhi::c_ConstantBufferOffsetSizeAlignment - 1);
    m_SDSM.inputCB = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(kInputCBSize)
            .setIsConstantBuffer(true)
            .setDebugName("SDSMInputCB")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
    );
    if (!m_SDSM.inputCB) return false;

    // Cascade output — structured UAV, single element holding all per-cascade data
    m_SDSM.cascadeDataBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(sizeof(SDSMCascadeOut))
            .setStructStride(sizeof(SDSMCascadeOut))
            .setDebugName("SDSMCascadeData")
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );
    if (!m_SDSM.cascadeDataBuffer) return false;

    // Binding set references the placeholder Hi-Z texture; rebuilt in _EnsureHiZResources
    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_SDSM.inputCB),
        nvrhi::BindingSetItem::Texture_SRV(0, m_HiZ.hizTexture),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_SDSM.cascadeDataBuffer),
    };
    m_SDSM.buildBindingSet = device->createBindingSet(bsd, m_SDSM.buildBindingLayout);
    if (!m_SDSM.buildBindingSet) return false;

    return true;
}

bool ComputeRenderPass::_InitSkyPass() {
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
