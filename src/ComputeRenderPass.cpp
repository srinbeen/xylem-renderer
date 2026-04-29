#include "include/ComputeRenderPass.hpp"
#include "include/Globals.hpp"
#include "include/macros.h"
#include "include/Terrain.hpp"
#include "include/shaders/ShaderContracts.hpp"
#include "include/frame/FrameLifecycle.hpp"


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
namespace shader_cb = Xylem::shader::cb;
namespace compute_reg = Xylem::shader::reg::Compute;

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
            out = GetDevice()->createBuffer(vDesc);
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
        auto iBuf = GetDevice()->createBuffer(iDesc);
        commandList->beginTrackingBufferState(iBuf, nvrhi::ResourceStates::CopyDest);
        commandList->writeBuffer(iBuf, lodDef.indices.data(), iDesc.byteSize);
        commandList->setPermanentBufferState(iBuf, nvrhi::ResourceStates::IndexBuffer);

        gpuAsset.lods[j].indexBuffer    = iBuf;
        gpuAsset.lods[j].indexCount     = static_cast<uint32_t>(lodDef.indices.size());
        gpuAsset.lods[j].radialSegments = lodDef.radialSegments;
        gpuAsset.lods[j].bbox           = lodDef.bbox;
    }
}

void ComputeRenderPass::_UploadAllAssets(nvrhi::ICommandList* commandList) {
    const auto& assets = m_Registry.getAssets();
    m_GPUAssets.resize(assets.size());
    m_AssetIdToGPUIndex.clear();

    for (size_t i = 0; i < assets.size(); i++) {
        _UploadAsset(assets[i], m_GPUAssets[i], commandList);
        m_AssetIdToGPUIndex[assets[i].id] = i;
    }
}

// ===========================================================================
// Region windows - gapped persistent buffer layout
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
// Slot layout - compute slot offsets and visibility buffer sizing
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

    // Shadow slot layout - numAssets x NUM_CASCADES slots, interleaved as
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

    // Build indirect args staging - pre-fill with indexCount, instanceCount=0
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

    // Shadow indirect args - one per (asset x cascade), using lowest LOD.
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

    // PersistentInstanceBuffer - SRV structured buffer (permanent ShaderResource after upload)
    m_StageResources.cull.persistentInstBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(m_TotalCapacity * sizeof(Render::InstanceBufferEntry))
            .setStructStride(sizeof(Render::InstanceBufferEntry))
            .setDebugName("PersistentInstanceBuffer")
            .setCanHaveUAVs(false)
            .setInitialState(nvrhi::ResourceStates::CopyDest)
    );
    commandList->beginTrackingBufferState(m_StageResources.cull.persistentInstBuffer, nvrhi::ResourceStates::CopyDest);
    commandList->writeBuffer(m_StageResources.cull.persistentInstBuffer,
        m_InstanceStaging.data(),
        m_TotalCapacity * sizeof(Render::InstanceBufferEntry));
    commandList->setPermanentBufferState(m_StageResources.cull.persistentInstBuffer, nvrhi::ResourceStates::ShaderResource);

    // CullDataBuffer - SRV structured buffer (permanent ShaderResource after upload)
    m_StageResources.cull.cullDataBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(m_TotalCapacity * sizeof(Render::CullInstanceData))
            .setStructStride(sizeof(Render::CullInstanceData))
            .setDebugName("CullDataBuffer")
            .setCanHaveUAVs(false)
            .setInitialState(nvrhi::ResourceStates::CopyDest)
    );
    commandList->beginTrackingBufferState(m_StageResources.cull.cullDataBuffer, nvrhi::ResourceStates::CopyDest);
    commandList->writeBuffer(m_StageResources.cull.cullDataBuffer,
        m_CullDataStaging.data(),
        m_TotalCapacity * sizeof(Render::CullInstanceData));
    commandList->setPermanentBufferState(m_StageResources.cull.cullDataBuffer, nvrhi::ResourceStates::ShaderResource);

    // CullRegionDataBuffer - SRV structured buffer (permanent ShaderResource after upload)
    m_StageResources.cull.cullRegionDataBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(numRegions * sizeof(Render::CullRegionData))
            .setStructStride(sizeof(Render::CullRegionData))
            .setDebugName("CullRegionDataBuffer")
            .setCanHaveUAVs(false)
            .setInitialState(nvrhi::ResourceStates::CopyDest)
    );
    commandList->beginTrackingBufferState(m_StageResources.cull.cullRegionDataBuffer, nvrhi::ResourceStates::CopyDest);
    commandList->writeBuffer(m_StageResources.cull.cullRegionDataBuffer,
        m_RegionStaging.data(),
        numRegions * sizeof(Render::CullRegionData));
    commandList->setPermanentBufferState(m_StageResources.cull.cullRegionDataBuffer, nvrhi::ResourceStates::ShaderResource);

    // SlotOffsetBuffer - SRV structured buffer (permanent ShaderResource after upload)
    m_StageResources.cull.slotOffsetBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(std::max(1u, m_NumSlots) * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("SlotOffsetBuffer")
            .setCanHaveUAVs(false)
            .setInitialState(nvrhi::ResourceStates::CopyDest)
    );
    commandList->beginTrackingBufferState(m_StageResources.cull.slotOffsetBuffer, nvrhi::ResourceStates::CopyDest);
    if (m_NumSlots > 0)
        commandList->writeBuffer(m_StageResources.cull.slotOffsetBuffer,
            m_SlotOffsets.data(), m_NumSlots * sizeof(uint32_t));
    commandList->setPermanentBufferState(m_StageResources.cull.slotOffsetBuffer, nvrhi::ResourceStates::ShaderResource);

    // CountBuffer - UAV structured buffer (cleared each frame, read as SRV by draw passes)
    m_StageResources.cull.countBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(std::max(1u, m_NumSlots) * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("CountBuffer")
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    // VisibilityBuffer - UAV structured buffer (written by CS, read as SRV by draw passes)
    m_StageResources.cull.visibilityBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(m_VisBufferSize * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("VisibilityBuffer")
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    const uint32_t numAssets      = static_cast<uint32_t>(m_GPUAssets.size());
    const uint32_t numShadowSlots = numAssets * Render::c_NumCascades;

    m_StageResources.cull.impostorCountBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(std::max(1u, numAssets) * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("ImpostorCountBuffer")
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    m_StageResources.cull.impostorVisBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(m_ImpostorVisBufferSize * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("ImpostorVisibilityBuffer")
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    m_StageResources.cull.impostorSlotOffsetBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(std::max(1u, numAssets) * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("ImpostorSlotOffsetBuffer")
            .setCanHaveUAVs(false)
            .setInitialState(nvrhi::ResourceStates::CopyDest)
    );
    commandList->beginTrackingBufferState(m_StageResources.cull.impostorSlotOffsetBuffer, nvrhi::ResourceStates::CopyDest);
    if (numAssets > 0)
        commandList->writeBuffer(m_StageResources.cull.impostorSlotOffsetBuffer,
            m_ImpostorSlotOffsets.data(), numAssets * sizeof(uint32_t));
    commandList->setPermanentBufferState(m_StageResources.cull.impostorSlotOffsetBuffer, nvrhi::ResourceStates::ShaderResource);

    // ShadowCountBuffer - UAV uint32[numShadowSlots], one counter per (asset x cascade)
    m_StageResources.cull.shadowCountBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(std::max(1u, numShadowSlots) * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("ShadowCountBuffer")
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    // ShadowVisBuffer - UAV uint32[shadowVisBufferSize], per-(asset x cascade) windows
    m_StageResources.cull.shadowVisBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(m_ShadowVisBufferSize * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("ShadowVisBuffer")
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    // ShadowSlotOffsetBuffer - SRV uint32[numShadowSlots], prefix sums into ShadowVisBuffer
    m_StageResources.cull.shadowSlotOffsetBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(std::max(1u, numShadowSlots) * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("ShadowSlotOffsetBuffer")
            .setCanHaveUAVs(false)
            .setInitialState(nvrhi::ResourceStates::CopyDest)
    );
    commandList->beginTrackingBufferState(m_StageResources.cull.shadowSlotOffsetBuffer, nvrhi::ResourceStates::CopyDest);
    if (numShadowSlots > 0)
        commandList->writeBuffer(m_StageResources.cull.shadowSlotOffsetBuffer,
            m_ShadowSlotOffsets.data(), numShadowSlots * sizeof(uint32_t));
    commandList->setPermanentBufferState(m_StageResources.cull.shadowSlotOffsetBuffer, nvrhi::ResourceStates::ShaderResource);

    // IndirectArgsBuffer - UAV + indirect draw args, DrawIndexedIndirectArguments[numSlots]
    // Pre-filled with indexCount per slot, instanceCount=0 (CS atomically increments each frame).
    uint32_t indirectArgsBufSize = std::max(1u, m_NumSlots) * sizeof(nvrhi::DrawIndexedIndirectArguments);
    m_StageResources.cull.indirectArgsBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(indirectArgsBufSize)
            .setDebugName("IndirectArgsBuffer")
            .setIsDrawIndirectArgs(true)
            .setCanHaveUAVs(true)
            .setCanHaveRawViews(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    // ImpostorIndirectArgsBuffer - UAV + indirect draw args, DrawIndirectArguments[numAssets]
    uint32_t impostorIndirectArgsBufSize = std::max(1u, numAssets) * sizeof(nvrhi::DrawIndirectArguments);
    m_StageResources.cull.impostorIndirectArgsBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(impostorIndirectArgsBufSize)
            .setDebugName("ImpostorIndirectArgsBuffer")
            .setIsDrawIndirectArgs(true)
            .setCanHaveUAVs(true)
            .setCanHaveRawViews(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    // ShadowIndirectArgsBuffer - UAV + indirect draw args, DrawIndexedIndirectArguments[numShadowSlots]
    uint32_t shadowIndirectArgsBufSize = std::max(1u, numShadowSlots) * sizeof(nvrhi::DrawIndexedIndirectArguments);
    m_StageResources.cull.shadowIndirectArgsBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(shadowIndirectArgsBufSize)
            .setDebugName("ShadowIndirectArgsBuffer")
            .setIsDrawIndirectArgs(true)
            .setCanHaveUAVs(true)
            .setCanHaveRawViews(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    // ShadowUniqueCounter - single uint32, incremented once per instance that
    // passes at least one cascade. Used by UI to report unique-caster count
    // distinct from the per-cascade summed count (which double-counts).
    m_StageResources.cull.shadowUniqueCounter = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(sizeof(uint32_t))
            .setDebugName("ShadowUniqueCounter")
            .setCanHaveUAVs(true)
            .setCanHaveRawViews(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );

    // RegionVisibleBuffer - SRV uint32[numRegions], CPU-written each frame via writeBuffer
    m_RegionVisibleStaging.assign(std::max(1u, numRegions), 1u);
    m_StageResources.cull.regionVisibleBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(std::max(1u, numRegions) * sizeof(uint32_t))
            .setStructStride(sizeof(uint32_t))
            .setDebugName("RegionVisibleBuffer")
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource)
    );

    // Readback ring buffers - combined [mesh LOD counts | impostor counts | shadow counts | shadow unique].
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
        nvrhi::BindingSetItem::ConstantBuffer(compute_reg::Cull::kCB_Frame, m_StageResources.frameShared.constantBuffer,
            nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),

        nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Cull::kSRV_RegionData, m_StageResources.cull.cullRegionDataBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Cull::kSRV_InstanceData, m_StageResources.cull.cullDataBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Cull::kSRV_MainSlotOffsets, m_StageResources.cull.slotOffsetBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Cull::kSRV_ShadowSlotOffsets, m_StageResources.cull.shadowSlotOffsetBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Cull::kSRV_ImpostorSlotOffsets, m_StageResources.cull.impostorSlotOffsetBuffer),

        nvrhi::BindingSetItem::StructuredBuffer_UAV(compute_reg::Cull::kUAV_MainRegionVis, m_StageResources.cull.regionVisibleBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(compute_reg::Cull::kUAV_MainSlotCount, m_StageResources.cull.countBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(compute_reg::Cull::kUAV_MainVis, m_StageResources.cull.visibilityBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(compute_reg::Cull::kUAV_ShadowSlotCount, m_StageResources.cull.shadowCountBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(compute_reg::Cull::kUAV_ShadowVis, m_StageResources.cull.shadowVisBuffer),

        nvrhi::BindingSetItem::RawBuffer_UAV(compute_reg::Cull::kUAV_MainIndirectArgs, m_StageResources.cull.indirectArgsBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(compute_reg::Cull::kUAV_ShadowIndirectArgs, m_StageResources.cull.shadowIndirectArgsBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(compute_reg::Cull::kUAV_ShadowUniqueCounter, m_StageResources.cull.shadowUniqueCounter),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(compute_reg::Cull::kUAV_ImpostorCount, m_StageResources.cull.impostorCountBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(compute_reg::Cull::kUAV_ImpostorVis, m_StageResources.cull.impostorVisBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(compute_reg::Cull::kUAV_ImpostorIndirectArgs, m_StageResources.cull.impostorIndirectArgsBuffer),

        nvrhi::BindingSetItem::Texture_SRV(compute_reg::Cull::kSRV_HiZ, m_StageResources.hiz.hizTexture),
        nvrhi::BindingSetItem::Sampler(compute_reg::Cull::kSampler_HiZ, m_StageResources.hiz.pointSampler),
    };

    m_StageResources.cull.bindingSet = device->createBindingSet(cullBSD, m_StageResources.cull.bindingLayout);

    // Tree pass binding sets - visibility/instance/slotOffset SRVs + textures
    m_StageResources.sceneTree.bindingSets.resize(m_StageResources.sceneTree.textureSets.size());
    for (size_t i = 0; i < m_StageResources.sceneTree.textureSets.size(); i++) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(compute_reg::Scene::kCB_Frame, m_StageResources.frameShared.constantBuffer,
                nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
            nvrhi::BindingSetItem::PushConstants(compute_reg::Scene::kPushC_Slot, sizeof(uint32_t)),

            nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Scene::kSRV_Vis, m_StageResources.cull.visibilityBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Scene::kSRV_Instances, m_StageResources.cull.persistentInstBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Scene::kSRV_SlotOffsets, m_StageResources.cull.slotOffsetBuffer),

            nvrhi::BindingSetItem::Texture_SRV(compute_reg::Scene::kTex_Diffuse, m_StageResources.sceneTree.textureSets[i].diffuse),
            nvrhi::BindingSetItem::Texture_SRV(compute_reg::Scene::kTex_NormalMap, m_StageResources.sceneTree.textureSets[i].normalMap),
            nvrhi::BindingSetItem::Texture_SRV(compute_reg::Scene::kTex_ShadowMap, m_StageResources.shadow.depthTexture),

            nvrhi::BindingSetItem::Sampler(compute_reg::Scene::kSampler_Main, m_StageResources.sceneTree.sampler),
            nvrhi::BindingSetItem::Sampler(compute_reg::Scene::kSampler_Shadow, m_StageResources.shadow.comparisonSampler),
        };
        m_StageResources.sceneTree.bindingSets[i] = device->createBindingSet(bsd, m_StageResources.sceneTree.bindingLayout);
    }

    m_StageResources.impostor.bindingSets.resize(m_StageResources.sceneTree.textureSets.size());
    for (size_t i = 0; i < m_StageResources.sceneTree.textureSets.size(); i++) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(compute_reg::Impostor::kCB_Frame, m_StageResources.frameShared.constantBuffer,
                nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
            nvrhi::BindingSetItem::PushConstants(compute_reg::Impostor::kPushC_AssetIndex, sizeof(uint32_t)),

            nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_Vis, m_StageResources.cull.impostorVisBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_Instances, m_StageResources.cull.persistentInstBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_SlotOffsets, m_StageResources.cull.impostorSlotOffsetBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_CullData, m_StageResources.cull.cullDataBuffer),

            nvrhi::BindingSetItem::Texture_SRV(compute_reg::Impostor::kTex_Albedo, m_StageResources.impostor.albedoAlphaTexture),
            nvrhi::BindingSetItem::Texture_SRV(compute_reg::Impostor::kTex_Normal, m_StageResources.impostor.normalTexture),
            nvrhi::BindingSetItem::Texture_SRV(compute_reg::Impostor::kTex_Depth, m_StageResources.impostor.depthTexture, nvrhi::Format::R32_FLOAT),
            nvrhi::BindingSetItem::Texture_SRV(compute_reg::Impostor::kTex_ShadowMap, m_StageResources.shadow.depthTexture),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_AssetDims, m_StageResources.impostor.assetDimsBuffer),

            nvrhi::BindingSetItem::Sampler(compute_reg::Impostor::kSampler_Main, m_StageResources.impostor.sampler),
            nvrhi::BindingSetItem::Sampler(compute_reg::Impostor::kSampler_Shadow, m_StageResources.shadow.comparisonSampler),
            nvrhi::BindingSetItem::Sampler(compute_reg::Impostor::kSampler_Depth, m_StageResources.impostor.depthSampler),
        };
        m_StageResources.impostor.bindingSets[i] = device->createBindingSet(bsd, m_StageResources.impostor.bindingLayout);
    }

    // Shadow pass binding set
    nvrhi::BindingSetDesc shadowBSD;
    shadowBSD.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(compute_reg::Shadow::kCB_Frame, m_StageResources.frameShared.constantBuffer,
            nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
        nvrhi::BindingSetItem::PushConstants(compute_reg::Shadow::kPushC_AssetCascade, sizeof(uint32_t) * 2),

        nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Shadow::kSRV_Vis, m_StageResources.cull.shadowVisBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Shadow::kSRV_Instances, m_StageResources.cull.persistentInstBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::Shadow::kSRV_SlotOffsets, m_StageResources.cull.shadowSlotOffsetBuffer),
    };
    m_StageResources.shadow.bindingSet = device->createBindingSet(shadowBSD, m_StageResources.shadow.bindingLayout);
}

// ===========================================================================
// Ensure Hi-Z resources exist at the given resolution (lazy init / resize)
// ===========================================================================

void ComputeRenderPass::_EnsureHiZResources(uint32_t width, uint32_t height) {
    // Skip if already at the right size
    if (m_StageResources.depthPrepass.depthTexture) {
        auto desc = m_StageResources.depthPrepass.depthTexture->getDesc();
        if (desc.width == width && desc.height == height)
            return;
    }

    auto device = GetDevice();

    // --- Depth prepass texture + framebuffer ---
    m_StageResources.depthPrepass.depthTexture = device->createTexture(
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
    m_StageResources.depthPrepass.framebuffer = device->createFramebuffer(
        nvrhi::FramebufferDesc().setDepthAttachment(m_StageResources.depthPrepass.depthTexture));

    // Invalidate pipelines (resolution-dependent)
    m_StageResources.depthPrepass.treePipeline    = nullptr;
    m_StageResources.depthPrepass.terrainPipeline = nullptr;

    // --- Hi-Z texture with full mip chain (RG32: .r=farthest for Hi-Z, .g=nearest for SDSM) ---
    m_StageResources.hiz.numMips = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
    m_StageResources.hiz.hizTexture = device->createTexture(
        nvrhi::TextureDesc()
            .setWidth(width).setHeight(height)
            .setMipLevels(m_StageResources.hiz.numMips)
            .setFormat(nvrhi::Format::RG32_FLOAT)
            .setIsUAV(true)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true)
            .setDebugName("HiZTexture")
    );

    // --- Per-mip binding sets for Hi-Z build ---
    m_StageResources.hiz.buildBindingSets.resize(m_StageResources.hiz.numMips);

    // Set 0: copy from depth prepass (D32 read as R32_FLOAT) -> Hi-Z mip 0 (RG32)
    // Shader seeds both channels from the single-channel source.
    {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::PushConstants(compute_reg::HiZ::kPushC_DestDimensions, sizeof(uint32_t) * compute_reg::HiZ::kPushCDwordCount),
            nvrhi::BindingSetItem::Texture_SRV(compute_reg::HiZ::kSRV_Source, m_StageResources.depthPrepass.depthTexture,
                nvrhi::Format::R32_FLOAT,
                nvrhi::TextureSubresourceSet(0, 1, 0, 1)),
            nvrhi::BindingSetItem::Texture_UAV(compute_reg::HiZ::kUAV_Dest, m_StageResources.hiz.hizTexture,
                nvrhi::Format::RG32_FLOAT,
                nvrhi::TextureSubresourceSet(0, 1, 0, 1)),
        };
        m_StageResources.hiz.buildBindingSets[0] = device->createBindingSet(bsd, m_StageResources.hiz.buildBindingLayout);
    }

    // Sets 1..N-1: downsample mip i-1 -> mip i (both RG32)
    for (uint32_t mip = 1; mip < m_StageResources.hiz.numMips; mip++) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::PushConstants(compute_reg::HiZ::kPushC_DestDimensions, sizeof(uint32_t) * compute_reg::HiZ::kPushCDwordCount),
            nvrhi::BindingSetItem::Texture_SRV(compute_reg::HiZ::kSRV_Source, m_StageResources.hiz.hizTexture,
                nvrhi::Format::RG32_FLOAT,
                nvrhi::TextureSubresourceSet(mip - 1, 1, 0, 1)),
            nvrhi::BindingSetItem::Texture_UAV(compute_reg::HiZ::kUAV_Dest, m_StageResources.hiz.hizTexture,
                nvrhi::Format::RG32_FLOAT,
                nvrhi::TextureSubresourceSet(mip, 1, 0, 1)),
        };
        m_StageResources.hiz.buildBindingSets[mip] = device->createBindingSet(bsd, m_StageResources.hiz.buildBindingLayout);
    }

    // --- Per-mip debug view textures (single-mip, RG32_FLOAT, for ImGui display) ---
    m_StageResources.hiz.debugMipTextures.resize(m_StageResources.hiz.numMips);
    for (uint32_t mip = 0; mip < m_StageResources.hiz.numMips; mip++) {
        uint32_t mipW = std::max(1u, width  >> mip);
        uint32_t mipH = std::max(1u, height >> mip);
        m_StageResources.hiz.debugMipTextures[mip] = device->createTexture(
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
        nvrhi::BindingSetItem::ConstantBuffer(compute_reg::DepthPrepass::kCB_Frame, m_StageResources.frameShared.constantBuffer,
            nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
        nvrhi::BindingSetItem::PushConstants(compute_reg::DepthPrepass::kPushC_Slot, sizeof(uint32_t)),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::DepthPrepass::kSRV_Vis, m_StageResources.cull.visibilityBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::DepthPrepass::kSRV_Instances, m_StageResources.cull.persistentInstBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(compute_reg::DepthPrepass::kSRV_SlotOffsets, m_StageResources.cull.slotOffsetBuffer),
    };
    m_StageResources.depthPrepass.bindingSet = device->createBindingSet(prepassBSD, m_StageResources.depthPrepass.bindingLayout);

    // Rebuild cull binding set to reference the new Hi-Z texture
    _RebuildCullBindings();

    // Rebuild SDSM binding set (reads top mip of Hi-Z texture)
    if (m_StageResources.sdsm.buildBindingLayout) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(compute_reg::SDSM::kCB_Input, m_StageResources.sdsm.inputCB),
            nvrhi::BindingSetItem::Texture_SRV(compute_reg::SDSM::kSRV_HiZ, m_StageResources.hiz.hizTexture),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(compute_reg::SDSM::kUAV_CascadeOut, m_StageResources.sdsm.cascadeDataBuffer),
        };
        m_StageResources.sdsm.buildBindingSet = device->createBindingSet(bsd, m_StageResources.sdsm.buildBindingLayout);
    }
}

// ===========================================================================
// Depth prepass - draw last frame's visible set depth-only
// ===========================================================================

void ComputeRenderPass::_RenderDepthPrepass() {
    const uint32_t numLods = static_cast<uint32_t>(m_Registry.getLodSegments().size());

    // Clear depth to far plane
#if XYLEM_USE_REVERSE_Z
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList,
        m_StageResources.depthPrepass.framebuffer, 0.f, 0);
#else
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList,
        m_StageResources.depthPrepass.framebuffer, 1.f, 0);
#endif

    // Lazy pipeline creation
    if (!m_StageResources.depthPrepass.treePipeline) {
        nvrhi::GraphicsPipelineDesc pso;
        pso.VS             = m_StageResources.depthPrepass.treeVS;
        pso.inputLayout    = m_StageResources.depthPrepass.treeInputLayout;
        pso.bindingLayouts = { m_StageResources.depthPrepass.bindingLayout };
        pso.primType       = nvrhi::PrimitiveType::TriangleList;
    #if XYLEM_USE_REVERSE_Z
        pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
    #else
        pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    #endif
        pso.renderState.rasterState.setCullBack();
        m_StageResources.depthPrepass.treePipeline = GetDevice()->createGraphicsPipeline(
            pso, m_StageResources.depthPrepass.framebuffer->getFramebufferInfo());
    }

    // Draw trees using last frame's indirect args (not yet cleared)
    nvrhi::GraphicsState state;
    state.pipeline    = m_StageResources.depthPrepass.treePipeline;
    state.framebuffer = m_StageResources.depthPrepass.framebuffer;
    state.viewport    = m_ViewHandler.view.GetViewportState();
    state.bindings    = { m_StageResources.depthPrepass.bindingSet };
    state.indirectParams = m_StageResources.cull.indirectArgsBuffer;

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
    if (m_StageResources.sceneTerrain.indexCount > 0) {
        if (!m_StageResources.depthPrepass.terrainPipeline) {
            nvrhi::GraphicsPipelineDesc pso;
            pso.VS             = m_StageResources.depthPrepass.terrainVS;
            pso.inputLayout    = m_StageResources.depthPrepass.terrainInputLayout;
            pso.bindingLayouts = { m_StageResources.depthPrepass.bindingLayout };
            pso.primType       = nvrhi::PrimitiveType::TriangleList;
        #if XYLEM_USE_REVERSE_Z
            pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
        #else
            pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
        #endif
            pso.renderState.rasterState.setCullBack();
            m_StageResources.depthPrepass.terrainPipeline = GetDevice()->createGraphicsPipeline(
                pso, m_StageResources.depthPrepass.framebuffer->getFramebufferInfo());
        }

        nvrhi::GraphicsState terrState;
        terrState.pipeline    = m_StageResources.depthPrepass.terrainPipeline;
        terrState.framebuffer = m_StageResources.depthPrepass.framebuffer;
        terrState.viewport    = m_ViewHandler.view.GetViewportState();
        terrState.bindings    = { m_StageResources.depthPrepass.bindingSet };
        terrState.vertexBuffers = { { m_StageResources.sceneTerrain.vertexBuffer, 0, 0 } };
        terrState.indexBuffer   = { m_StageResources.sceneTerrain.indexBuffer, nvrhi::Format::R32_UINT, 0 };
        m_CommandList->setGraphicsState(terrState);

        uint32_t c = 0;
        m_CommandList->setPushConstants(&c, sizeof(c));
        m_CommandList->drawIndexed(
            nvrhi::DrawArguments().setVertexCount(m_StageResources.sceneTerrain.indexCount));
    }
}

// ===========================================================================
// Build Hi-Z mip chain from depth prepass
// ===========================================================================

void ComputeRenderPass::_BuildHiZMipChain() {
    auto desc = m_StageResources.depthPrepass.depthTexture->getDesc();
    uint32_t w = desc.width;
    uint32_t h = desc.height;

    // Pass 0: Copy depth texture -> Hi-Z mip 0
    {
        uint32_t dims[2] = { w, h };
        nvrhi::ComputeState cs;
        cs.pipeline = m_StageResources.hiz.copyPipeline;
        cs.bindings = { m_StageResources.hiz.buildBindingSets[0] };
        m_CommandList->setComputeState(cs);
        m_CommandList->setPushConstants(dims, sizeof(dims));
        m_CommandList->dispatch((w + 7) / 8, (h + 7) / 8, 1);
    }

    // Passes 1..N-1: Downsample mip i-1 -> mip i
    for (uint32_t mip = 1; mip < m_StageResources.hiz.numMips; mip++) {
        uint32_t mipW = std::max(1u, w >> mip);
        uint32_t mipH = std::max(1u, h >> mip);
        uint32_t dims[2] = { mipW, mipH };

        nvrhi::ComputeState cs;
        cs.pipeline = m_StageResources.hiz.buildPipeline;
        cs.bindings = { m_StageResources.hiz.buildBindingSets[mip] };
        m_CommandList->setComputeState(cs);
        m_CommandList->setPushConstants(dims, sizeof(dims));
        m_CommandList->dispatch((mipW + 7) / 8, (mipH + 7) / 8, 1);
    }

    // Copy each mip into its debug view texture (for ImGui display)
    if (m_UI.showHiZ && !m_StageResources.hiz.debugMipTextures.empty()) {
        for (uint32_t mip = 0; mip < m_StageResources.hiz.numMips; mip++) {
            m_CommandList->copyTexture(
                m_StageResources.hiz.debugMipTextures[mip], nvrhi::TextureSlice(),
                m_StageResources.hiz.hizTexture,           nvrhi::TextureSlice().setMipLevel(mip));
        }
    }
}

// ===========================================================================
// SDSM - GPU cascade construction
// ===========================================================================

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
    // Pack invariants into the SDSM cascade-build input CB.
    shader_cb::SDSMCascadeBuildInput input{};

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
    input.maxHiZMip          = (m_StageResources.hiz.numMips > 0) ? (m_StageResources.hiz.numMips - 1) : 0;
    input.pssmLambda         = m_UI.pssmLambda;

    m_CommandList->writeBuffer(m_StageResources.sdsm.inputCB, &input, sizeof(shader_cb::SDSMCascadeBuildInput));

    // Dispatch - one threadgroup, NUM_CASCADES threads
    {
        nvrhi::ComputeState cs;
        cs.pipeline = m_StageResources.sdsm.buildPipeline;
        cs.bindings = { m_StageResources.sdsm.buildBindingSet };
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch(1, 1, 1);
    }

    // Copy cascade fields into the main CB at their CullConstantBufferEntry offsets.
    constexpr size_t kOffLightViewProj     = offsetof(Render::CullConstantBufferEntry, lightViewProj);
    constexpr size_t kOffCascadeSplits     = offsetof(Render::CullConstantBufferEntry, cascadeSplits);
    constexpr size_t kOffShadowCasterMin   = offsetof(Render::CullConstantBufferEntry, shadowCasterMinLS);
    constexpr size_t kOffShadowCasterMax   = offsetof(Render::CullConstantBufferEntry, shadowCasterMaxLS);

    constexpr size_t kSrcLightViewProj     = offsetof(shader_cb::SDSMCascadeBuildOutput, lightViewProj);
    constexpr size_t kSrcCascadeSplits     = offsetof(shader_cb::SDSMCascadeBuildOutput, cascadeSplits);
    constexpr size_t kSrcShadowCasterMin   = offsetof(shader_cb::SDSMCascadeBuildOutput, shadowCasterMinLS);
    constexpr size_t kSrcShadowCasterMax   = offsetof(shader_cb::SDSMCascadeBuildOutput, shadowCasterMaxLS);

    m_CommandList->copyBuffer(m_StageResources.frameShared.constantBuffer, kOffLightViewProj,
                              m_StageResources.sdsm.cascadeDataBuffer, kSrcLightViewProj,
                              sizeof(dm::float4x4) * Render::c_NumCascades);
    m_CommandList->copyBuffer(m_StageResources.frameShared.constantBuffer, kOffCascadeSplits,
                              m_StageResources.sdsm.cascadeDataBuffer, kSrcCascadeSplits,
                              sizeof(dm::float4));
    m_CommandList->copyBuffer(m_StageResources.frameShared.constantBuffer, kOffShadowCasterMin,
                              m_StageResources.sdsm.cascadeDataBuffer, kSrcShadowCasterMin,
                              sizeof(dm::float4) * Render::c_NumCascades);
    m_CommandList->copyBuffer(m_StageResources.frameShared.constantBuffer, kOffShadowCasterMax,
                              m_StageResources.sdsm.cascadeDataBuffer, kSrcShadowCasterMax,
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

        _UploadAllAssets(initCL);

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
    frame::RunDirtyCycle(m_Registry, *this);
}

void ComputeRenderPass::BackBufferResizing() {
    m_StageResources.sceneTree.pipeline          = nullptr;
    m_StageResources.impostor.pipeline           = nullptr;
    m_StageResources.sceneTerrain.pipeline       = nullptr;
    m_StageResources.shadow.treePipeline    = nullptr;
    m_StageResources.shadow.terrainPipeline = nullptr;
    m_StageResources.sky.pipeline           = nullptr;

    // Hi-Z resources are resolution-dependent - force recreation
    m_StageResources.depthPrepass.treePipeline    = nullptr;
    m_StageResources.depthPrepass.terrainPipeline = nullptr;
    m_StageResources.depthPrepass.depthTexture    = nullptr;
    m_StageResources.depthPrepass.framebuffer     = nullptr;
    m_StageResources.depthPrepass.bindingSet      = nullptr;

    m_StageResources.hiz.buildBindingSets.clear();
    m_StageResources.hiz.debugMipTextures.clear();
    m_StageResources.hiz.numMips = 0;
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
        _UploadAllAssets(cl);
    } else {
        for (size_t idx : dirtyAssetIndices) {
            const auto& assetDef = assets[idx];
            auto it = m_AssetIdToGPUIndex.find(assetDef.id);
            if (it != m_AssetIdToGPUIndex.end()) {
                _UploadAsset(assetDef, m_GPUAssets[it->second], cl);
            } else {
                GPUTreeAsset gpuAsset;
                _UploadAsset(assetDef, gpuAsset, cl);
                m_AssetIdToGPUIndex[assetDef.id] = m_GPUAssets.size();
                m_GPUAssets.push_back(std::move(gpuAsset));
            }
        }
    }

    // Re-upload CullDataBuffer bboxes (asset geometry changed -> bboxes changed)
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
    frame::SetShadowDebugOutputs(
        m_StageOutputs,
        m_StageResources.shadow.depthTexture.Get(),
        m_StageResources.shadow.debugTextures);
    frame::SetHiZDebugOutputs(m_StageOutputs, m_StageResources.hiz.debugMipTextures);
    frame::PublishStageOutputsToUI(m_StageOutputs, m_UI);

    // Impostor atlas debug pointers. Stable preview textures; the
    // selected asset's atlas sheet is copied into them at the end of Render.
    m_UI.impostorAssetCount     = static_cast<uint32_t>(m_GPUAssets.size());
    m_UI.impostorViewsPerAsset  = k_ImpostorViewCount;
    m_UI.impostorAzimuthViews   = k_ImpostorAzimuthViews;
    m_UI.impostorElevationViews = k_ImpostorElevationViews;
    m_UI.impostorAlbedoAtlasTexture  = m_StageResources.impostor.debugAlbedoAtlasTexture;
    m_UI.impostorNormalAtlasTexture  = m_StageResources.impostor.debugNormalAtlasTexture;
    m_UI.impostorDepthAtlasTexture   = m_StageResources.impostor.debugDepthAtlasTexture;

    app::HiResTimer cpuTimer;
    cpuTimer.Start();

    frame::FrameContext frameContext = frame::BuildFrameContext(
        m_Registry,
        m_ViewHandler,
        framebuffer,
        !m_StageResources.sceneTree.pipeline);

    m_CommandList->open();
    // m_CommandList->beginTimerQuery(m_GpuTimers[m_NextTimerIdx]);

    // m_CommandList->beginMarker("Frame");

    nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));
    #if XYLEM_USE_REVERSE_Z
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer, 0.f, 0);
    #else
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer,
        m_ViewHandler.view.GetProjectionFrustum().farPlane().distance, 0);
    #endif

    const dm::float3& camPos = m_ViewHandler.camera.GetPosition();
    const dm::float3& camDir = m_ViewHandler.camera.GetDir();
    frame::ComputeCascades(
        frameContext,
        m_ViewHandler,
        m_Registry,
        k_ShadowRes,
        m_UI.pssmLambda);

    // --- Fill CullConstantBufferEntry ---
    Render::CullConstantBufferEntry constants{};
    frame::FillCommonFrameConstants(constants, m_ViewHandler, m_Registry.getSunDirection());

    constants.viewFrustum  = m_ViewHandler.view.GetViewFrustum();
    constants.worldToLight = dm::affineToHomogeneous(m_ViewHandler.worldToLight);
    for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
        const dm::box3& cBbox = m_ViewHandler.cascades[c].shadowCasterBboxLS;
        if (cBbox.isempty()) {
            // Empty cascade - CullCS detects this via (min > max)
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
        constants.lodDistances[i] = lodDistances[i];

    // Hi-Z fields - bypass when camera looks steeply downward (bird's-eye view)
    float downwardness = -m_ViewHandler.camera.GetDir().y;  // 0 = horizontal, 1 = straight down
    bool hizActive = downwardness < m_UI.hizBypassAngle;
    m_UI.hizActiveThisFrame = hizActive;

    constants.hizDimensions = dm::float2(static_cast<float>(frameContext.frameWidth),
                                         static_cast<float>(frameContext.frameHeight));
    constants.maxHiZMip     = static_cast<float>(m_StageResources.hiz.numMips - 1);
    constants.hizEnabled    = hizActive ? 1u : 0u;
    constants.impostorAlphaClip = m_UI.impostorAlphaClip;

    m_CommandList->writeBuffer(m_StageResources.frameShared.constantBuffer, &constants, shader_cb::kCullFrameSize);

    // --- Ensure Hi-Z resources at current resolution ---
    _EnsureHiZResources(frameContext.frameWidth, frameContext.frameHeight);

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

        // SDSM - GPU replaces cascade fields in the shared CB from the reduced depth.
        m_CommandList->beginMarker("SDSM");
        float regionNear, regionFar;
        _ComputeRegionEnvelope(m_ViewHandler.view.GetViewFrustum(), camPos, camDir,
                               regionNear, regionFar);
        _RunSDSMBuildCascades(frameContext.sceneBounds, frameContext.aspectRatio, dm::radians(60.f),
                              regionNear, regionFar);
        m_CommandList->endMarker();
    }

    // --- GPU Cull Dispatch ---
    // Clear UAV counters (AFTER depth prepass which reads last frame's data)
    m_CommandList->clearBufferUInt(m_StageResources.cull.countBuffer, 0);
    m_CommandList->clearBufferUInt(m_StageResources.cull.impostorCountBuffer, 0);
    m_CommandList->clearBufferUInt(m_StageResources.cull.shadowCountBuffer, 0);
    m_CommandList->clearBufferUInt(m_StageResources.cull.shadowUniqueCounter, 0);

    // Reset indirect args buffers (re-upload staging with instanceCount=0)
    m_CommandList->writeBuffer(m_StageResources.cull.indirectArgsBuffer,
        m_IndirectArgsStaging.data(),
        m_IndirectArgsStaging.size() * sizeof(nvrhi::DrawIndexedIndirectArguments));
    m_CommandList->writeBuffer(m_StageResources.cull.impostorIndirectArgsBuffer,
        m_ImpostorIndirectArgsStaging.data(),
        m_ImpostorIndirectArgsStaging.size() * sizeof(nvrhi::DrawIndirectArguments));
    m_CommandList->writeBuffer(m_StageResources.cull.shadowIndirectArgsBuffer,
        m_ShadowIndirectArgsStaging.data(),
        m_ShadowIndirectArgsStaging.size() * sizeof(nvrhi::DrawIndexedIndirectArguments));

    m_CommandList->beginMarker("Cull");

    // region camera cull
    m_CommandList->beginMarker("RegionDispatch");
    {
        nvrhi::ComputeState cs;
        cs.pipeline = m_StageResources.cull.regionPipeline;
        cs.bindings = { m_StageResources.cull.bindingSet };
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch(
            (constants.numRegions + 63) / 64, 1, 1);
    }
    m_CommandList->endMarker();

    // Main camera cull
    m_CommandList->beginMarker("MainDispatch");
    {
        nvrhi::ComputeState cs;
        cs.pipeline = m_StageResources.cull.mainPipeline;
        cs.bindings = { m_StageResources.cull.bindingSet };
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch(
            (m_TotalCapacity + 255) / 256, 1, 1);
    }
    m_CommandList->endMarker();

    // Shadow cull
    m_CommandList->beginMarker("ShadowDispatch");
    {
        nvrhi::ComputeState cs;
        cs.pipeline = m_StageResources.cull.shadowPipeline;
        cs.bindings = { m_StageResources.cull.bindingSet };
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
                                  m_StageResources.cull.countBuffer, 0, countSize);
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], countSize,
                                  m_StageResources.cull.impostorCountBuffer, 0, impostorSize);
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], countSize + impostorSize,
                                  m_StageResources.cull.shadowCountBuffer, 0, shadowSize);
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], countSize + impostorSize + shadowSize,
                                  m_StageResources.cull.shadowUniqueCounter, 0, sizeof(uint32_t));
    }

    // --- Draw passes (auto barriers: UAV->SRV transitions handled by setGraphicsState) ---
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
        && m_StageResources.impostor.albedoAlphaTexture
        && m_StageResources.impostor.normalTexture
        && m_StageResources.impostor.depthTexture
        && m_StageResources.impostor.debugAlbedoAtlasTexture
        && m_StageResources.impostor.debugNormalAtlasTexture
        && m_StageResources.impostor.debugDepthAtlasTexture)
    {
        const uint32_t assetIdx = std::min(
            m_UI.impostorSelectedAsset,
            m_GPUAssets.empty() ? 0u : static_cast<uint32_t>(m_GPUAssets.size() - 1));

        for (uint32_t view = 0; view < k_ImpostorViewCount; view++) {
            const uint32_t tileX = view % k_ImpostorAzimuthViews;
            const uint32_t tileY = view / k_ImpostorAzimuthViews;
            const uint32_t tileSlice = assetIdx * k_ImpostorViewCount + view;
            nvrhi::TextureSlice dst = nvrhi::TextureSlice()
                .setOrigin(tileX * k_ImpostorBakeResolution, tileY * k_ImpostorBakeResolution, 0)
                .setWidth(k_ImpostorBakeResolution)
                .setHeight(k_ImpostorBakeResolution);

            m_CommandList->copyTexture(
                m_StageResources.impostor.debugAlbedoAtlasTexture, dst,
                m_StageResources.impostor.albedoAlphaTexture, nvrhi::TextureSlice().setArraySlice(tileSlice));
            m_CommandList->copyTexture(
                m_StageResources.impostor.debugNormalAtlasTexture, dst,
                m_StageResources.impostor.normalTexture, nvrhi::TextureSlice().setArraySlice(tileSlice));
            m_CommandList->copyTexture(
                m_StageResources.impostor.debugDepthAtlasTexture, dst,
                m_StageResources.impostor.depthTexture, nvrhi::TextureSlice().setArraySlice(tileSlice));
        }
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
    if (!m_StageResources.sky.pipeline) {
        nvrhi::GraphicsPipelineDesc pso;
        pso.VS             = m_StageResources.sky.vertexShader;
        pso.PS             = m_StageResources.sky.pixelShader;
        pso.bindingLayouts = { m_StageResources.sky.bindingLayout };
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
        m_StageResources.sky.pipeline = GetDevice()->createGraphicsPipeline(
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

    m_CommandList->writeBuffer(m_StageResources.sky.constantBuffer, &skyConstants, sizeof(skyConstants));

    nvrhi::GraphicsState skyState;
    skyState.pipeline    = m_StageResources.sky.pipeline;
    skyState.framebuffer = framebuffer;
    skyState.viewport    = m_ViewHandler.view.GetViewportState();
    skyState.bindings    = { m_StageResources.sky.bindingSet };
    m_CommandList->setGraphicsState(skyState);
    m_CommandList->draw(nvrhi::DrawArguments().setVertexCount(4));
}

// ===========================================================================
// Shadow pass - GPU cull results, visibility buffer indirection in VS
// ===========================================================================

void ComputeRenderPass::_RenderShadowPass() {
    if (m_ViewHandler.shadowCasterBboxLS.isempty())
        return;

    if (!m_StageResources.shadow.treePipeline) {
        nvrhi::GraphicsPipelineDesc pso;
        pso.VS             = m_StageResources.shadow.treeVS;
        pso.inputLayout    = m_StageResources.shadow.treeInputLayout;
        pso.bindingLayouts = { m_StageResources.shadow.bindingLayout };
        pso.primType       = nvrhi::PrimitiveType::TriangleList;
        pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
        pso.renderState.rasterState.setCullFront();
        pso.renderState.rasterState.depthBias            = 2;
        pso.renderState.rasterState.slopeScaledDepthBias = 2.0f;
        m_StageResources.shadow.treePipeline = GetDevice()->createGraphicsPipeline(
            pso, m_StageResources.shadow.framebuffers[0]->getFramebufferInfo());
    }
    if (!m_StageResources.shadow.terrainPipeline && m_StageResources.sceneTerrain.indexCount > 0) {
        nvrhi::GraphicsPipelineDesc pso;
        pso.VS             = m_StageResources.shadow.terrainVS;
        pso.inputLayout    = m_StageResources.shadow.terrainInputLayout;
        pso.bindingLayouts = { m_StageResources.shadow.bindingLayout };
        pso.primType       = nvrhi::PrimitiveType::TriangleList;
        pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
        pso.renderState.rasterState.setCullBack();
        pso.renderState.rasterState.depthBias            = 2;
        pso.renderState.rasterState.slopeScaledDepthBias = 2.0f;
        m_StageResources.shadow.terrainPipeline = GetDevice()->createGraphicsPipeline(
            pso, m_StageResources.shadow.framebuffers[0]->getFramebufferInfo());
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
            m_StageResources.shadow.framebuffers[cascade], 1.0f, 0);

        // Trees: one indirect draw per asset, reading its per-(asset x cascade)
        // window in shadowVisBuf via shadowSlotOffsets[ai * NUM_CASCADES + cascade].
        nvrhi::GraphicsState shadowState;
        shadowState.pipeline    = m_StageResources.shadow.treePipeline;
        shadowState.framebuffer = m_StageResources.shadow.framebuffers[cascade];
        shadowState.viewport    = shadowVPState;
        shadowState.bindings    = { m_StageResources.shadow.bindingSet };
        shadowState.indirectParams = m_StageResources.cull.shadowIndirectArgsBuffer;

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

        // Terrain shadow - only if terrain bbox intersects this cascade's LS bbox.
        if (m_StageResources.sceneTerrain.indexCount > 0 && terrainPtr
            && (terrainPtr->getBbox() * m_ViewHandler.worldToLight).intersects(cBbox)) {
            nvrhi::GraphicsState terrShadow;
            terrShadow.pipeline    = m_StageResources.shadow.terrainPipeline;
            terrShadow.framebuffer = m_StageResources.shadow.framebuffers[cascade];
            terrShadow.viewport    = shadowVPState;
            terrShadow.bindings    = { m_StageResources.shadow.bindingSet };
            terrShadow.vertexBuffers = { { m_StageResources.sceneTerrain.vertexBuffer, 0, 0 } };
            terrShadow.indexBuffer   = { m_StageResources.sceneTerrain.indexBuffer, nvrhi::Format::R32_UINT, 0 };
            m_CommandList->setGraphicsState(terrShadow);

            // Terrain VS only reads cascadeIdx; assetIndex unused but required for root sig.
            uint32_t pushConstants[2] = { 0u, cascade };
            m_CommandList->setPushConstants(pushConstants, sizeof(pushConstants));

            m_CommandList->drawIndexed(
                nvrhi::DrawArguments().setVertexCount(m_StageResources.sceneTerrain.indexCount));
        }
    }

    // Copy per-cascade slices into debug textures for UI display
    if (m_UI.showShadowMap) {
        for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
            m_CommandList->copyTexture(
                m_StageResources.shadow.debugTextures[c], nvrhi::TextureSlice(),
                m_StageResources.shadow.depthTexture, nvrhi::TextureSlice().setArraySlice(c));
        }
    }
}

// ===========================================================================
// Scene pass - GPU cull results, visibility buffer indirection in VS
// ===========================================================================

void ComputeRenderPass::_RenderScenePass(nvrhi::IFramebuffer* framebuffer) {
    const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();
    const uint32_t numLods = static_cast<uint32_t>(m_Registry.getLodSegments().size());

    if (!m_StageResources.sceneTree.pipeline) {
        nvrhi::GraphicsPipelineDesc psoDesc;
        psoDesc.VS           = m_StageResources.sceneTree.vertexShader;
        psoDesc.PS           = m_StageResources.sceneTree.pixelShader;
        psoDesc.inputLayout  = m_StageResources.sceneTree.inputLayout;
        psoDesc.bindingLayouts = { m_StageResources.sceneTree.bindingLayout };
        psoDesc.primType     = nvrhi::PrimitiveType::TriangleList;
    #if XYLEM_USE_REVERSE_Z
        psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
    #else
        psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    #endif
        m_StageResources.sceneTree.pipeline = GetDevice()->createGraphicsPipeline(psoDesc, fbinfo);
    }

    nvrhi::GraphicsState state;
    state.pipeline    = m_StageResources.sceneTree.pipeline;
    state.framebuffer = framebuffer;
    state.viewport    = m_ViewHandler.view.GetViewportState();
    state.indirectParams = m_StageResources.cull.indirectArgsBuffer;

    // Indirect draw: one call per slot (asset x LOD).
    // Instance count comes from the indirect args buffer (written by the cull CS).
    for (uint32_t ai = 0; ai < m_GPUAssets.size(); ai++) {
        for (uint32_t li = 0; li < numLods; li++) {
            uint32_t slot = ai * numLods + li;
            if (m_MaxSlotCounts[slot] == 0) continue;

            const auto& lod = m_GPUAssets[ai].lods[li];

            state.bindings = { m_StageResources.sceneTree.bindingSets[m_GPUAssets[ai].textureSetIdx] };
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
    if (m_StageResources.sceneTerrain.indexCount > 0) {
        if (!m_StageResources.sceneTerrain.pipeline) {
            nvrhi::GraphicsPipelineDesc terrainPso;
            terrainPso.VS = m_StageResources.sceneTerrain.vertexShader;
            terrainPso.PS = m_StageResources.sceneTerrain.pixelShader;
            terrainPso.inputLayout = m_StageResources.sceneTerrain.inputLayout;
            terrainPso.bindingLayouts = { m_StageResources.sceneTerrain.bindingLayout };
            terrainPso.primType = nvrhi::PrimitiveType::TriangleList;
    #if XYLEM_USE_REVERSE_Z
            terrainPso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
    #else
            terrainPso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    #endif
            terrainPso.renderState.rasterState.setCullNone();
            m_StageResources.sceneTerrain.pipeline = GetDevice()->createGraphicsPipeline(terrainPso, fbinfo);
        }

        nvrhi::GraphicsState terrainState;
        terrainState.pipeline   = m_StageResources.sceneTerrain.pipeline;
        terrainState.framebuffer = framebuffer;
        terrainState.viewport   = m_ViewHandler.view.GetViewportState();
        terrainState.bindings   = { m_StageResources.sceneTerrain.bindingSet };
        terrainState.vertexBuffers = { { m_StageResources.sceneTerrain.vertexBuffer, 0, 0 } };
        terrainState.indexBuffer   = { m_StageResources.sceneTerrain.indexBuffer, nvrhi::Format::R32_UINT, 0 };
        m_CommandList->setGraphicsState(terrainState);
        
        m_CommandList->drawIndexed(
            nvrhi::DrawArguments()
                .setVertexCount(m_StageResources.sceneTerrain.indexCount)
        );
    }
}

void ComputeRenderPass::_RenderImpostorPass(nvrhi::IFramebuffer* framebuffer) {
    if (m_GPUAssets.empty() || m_StageResources.impostor.bindingSets.empty())
        return;

    const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();

    if (!m_StageResources.impostor.pipeline) {
        nvrhi::GraphicsPipelineDesc psoDesc;
        psoDesc.VS = m_StageResources.impostor.vertexShader;
        psoDesc.PS = m_StageResources.impostor.pixelShader;
        psoDesc.inputLayout = nullptr;
        psoDesc.bindingLayouts = { m_StageResources.impostor.bindingLayout };
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
        m_StageResources.impostor.pipeline = GetDevice()->createGraphicsPipeline(psoDesc, fbinfo);
    }

    nvrhi::GraphicsState state;
    state.pipeline = m_StageResources.impostor.pipeline;
    state.framebuffer = framebuffer;
    state.viewport = m_ViewHandler.view.GetViewportState();
    state.indirectParams = m_StageResources.cull.impostorIndirectArgsBuffer;

    for (uint32_t ai = 0; ai < m_GPUAssets.size(); ai++) {
        if (ai >= m_ImpostorMaxSlotCounts.size() || m_ImpostorMaxSlotCounts[ai] == 0)
            continue;

        state.bindings = { m_StageResources.impostor.bindingSets[m_GPUAssets[ai].textureSetIdx] };
        m_CommandList->setGraphicsState(state);
        m_CommandList->setPushConstants(&ai, sizeof(ai));
        m_CommandList->drawIndirect(ai * sizeof(nvrhi::DrawIndirectArguments));
    }
}

// ===========================================================================
// Init helpers
// ===========================================================================

bool ComputeRenderPass::_InitShared() {
    m_StageResources.frameShared.constantBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(shader_cb::kCullFrameSize)
            .setIsConstantBuffer(true)
            .setDebugName("CullConstantBuffer")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
    );
    return !!m_StageResources.frameShared.constantBuffer;
}

bool ComputeRenderPass::_InitCullPass(nvrhi::ICommandList* /*initCL*/) {
    // Create compute shaders
    m_StageResources.cull.mainCS = m_ShaderFactory->CreateShader(
        "app/CullCS.hlsl", "CullMain", nullptr, nvrhi::ShaderType::Compute);
    m_StageResources.cull.shadowCS = m_ShaderFactory->CreateShader(
        "app/CullCS.hlsl", "CullShadow", nullptr, nvrhi::ShaderType::Compute);
    m_StageResources.cull.regionCS = m_ShaderFactory->CreateShader(
        "app/CullCS.hlsl", "CullRegion", nullptr, nvrhi::ShaderType::Compute);
    if (!m_StageResources.cull.mainCS || !m_StageResources.cull.shadowCS || !m_StageResources.cull.regionCS) return false;

    // Create cull binding layout
    nvrhi::BindingLayoutDesc cullLayoutDesc;
    cullLayoutDesc.visibility = nvrhi::ShaderType::All;
    cullLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(compute_reg::Cull::kCB_Frame),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Cull::kSRV_RegionData),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Cull::kSRV_InstanceData),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Cull::kSRV_MainSlotOffsets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Cull::kSRV_ShadowSlotOffsets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Cull::kSRV_ImpostorSlotOffsets),

        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(compute_reg::Cull::kUAV_MainRegionVis),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(compute_reg::Cull::kUAV_MainSlotCount),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(compute_reg::Cull::kUAV_MainVis),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(compute_reg::Cull::kUAV_ShadowSlotCount),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(compute_reg::Cull::kUAV_ShadowVis),

        nvrhi::BindingLayoutItem::RawBuffer_UAV(compute_reg::Cull::kUAV_MainIndirectArgs),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(compute_reg::Cull::kUAV_ShadowIndirectArgs),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(compute_reg::Cull::kUAV_ShadowUniqueCounter),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(compute_reg::Cull::kUAV_ImpostorCount),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(compute_reg::Cull::kUAV_ImpostorVis),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(compute_reg::Cull::kUAV_ImpostorIndirectArgs),

        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::Cull::kSRV_HiZ),
        nvrhi::BindingLayoutItem::Sampler(compute_reg::Cull::kSampler_HiZ),
    };
    m_StageResources.cull.bindingLayout = GetDevice()->createBindingLayout(cullLayoutDesc);
    if (!m_StageResources.cull.bindingLayout) return false;

    // Create compute pipelines
    nvrhi::ComputePipelineDesc regionPsoDesc;
    regionPsoDesc.CS = m_StageResources.cull.regionCS;
    regionPsoDesc.bindingLayouts = { m_StageResources.cull.bindingLayout };
    m_StageResources.cull.regionPipeline = GetDevice()->createComputePipeline(regionPsoDesc);
    
    nvrhi::ComputePipelineDesc mainPsoDesc;
    mainPsoDesc.CS = m_StageResources.cull.mainCS;
    mainPsoDesc.bindingLayouts = { m_StageResources.cull.bindingLayout };
    m_StageResources.cull.mainPipeline = GetDevice()->createComputePipeline(mainPsoDesc);

    nvrhi::ComputePipelineDesc shadowPsoDesc;
    shadowPsoDesc.CS = m_StageResources.cull.shadowCS;
    shadowPsoDesc.bindingLayouts = { m_StageResources.cull.bindingLayout };
    m_StageResources.cull.shadowPipeline = GetDevice()->createComputePipeline(shadowPsoDesc);

    if (!m_StageResources.cull.mainPipeline || !m_StageResources.cull.shadowPipeline) return false;

    return true;
}

bool ComputeRenderPass::_InitTreePass(nvrhi::ICommandList* initCL, engine::CommonRenderPasses& commonPasses) {
    m_StageResources.sceneTree.vertexShader = m_ShaderFactory->CreateShader(
        "app/ComputeRenderPass.hlsl", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_StageResources.sceneTree.pixelShader  = m_ShaderFactory->CreateShader(
        "app/ComputeRenderPass.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_StageResources.sceneTree.vertexShader || !m_StageResources.sceneTree.pixelShader) return false;

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
    m_StageResources.sceneTree.inputLayout = GetDevice()->createInputLayout(
        attributes, uint32_t(std::size(attributes)), m_StageResources.sceneTree.vertexShader);
    if (!m_StageResources.sceneTree.inputLayout) return false;

    // Load textures
    const auto& barkTextureSets = m_Registry.getBarkTextureSets();
    engine::TextureCache textureCache(GetDevice(), std::make_shared<vfs::NativeFileSystem>(), nullptr);
    m_StageResources.sceneTree.textureSets.resize(barkTextureSets.size());

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

        m_StageResources.sceneTree.textureSets[i].diffuse   = diffLoaded->texture;
        m_StageResources.sceneTree.textureSets[i].normalMap = normLoaded->texture;

        if (!m_StageResources.sceneTree.textureSets[i].diffuse || !m_StageResources.sceneTree.textureSets[i].normalMap)
            return false;
    }

    m_StageResources.sceneTree.sampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Wrap)
            .setAllFilters(true)
            .setMaxAnisotropy(8.f)
    );
    if (!m_StageResources.sceneTree.sampler) return false;

    // Create tree pass binding layout (with push constants for slot index)
    nvrhi::BindingLayoutDesc treeLayoutDesc;
    treeLayoutDesc.visibility = nvrhi::ShaderType::All;
    treeLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(compute_reg::Scene::kCB_Frame),
        nvrhi::BindingLayoutItem::PushConstants(compute_reg::Scene::kPushC_Slot, sizeof(uint32_t)),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Scene::kSRV_Vis),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Scene::kSRV_Instances),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Scene::kSRV_SlotOffsets),
        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::Scene::kTex_Diffuse),
        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::Scene::kTex_NormalMap),
        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::Scene::kTex_ShadowMap),
        nvrhi::BindingLayoutItem::Sampler(compute_reg::Scene::kSampler_Main),
        nvrhi::BindingLayoutItem::Sampler(compute_reg::Scene::kSampler_Shadow),
    };
    m_StageResources.sceneTree.bindingLayout = GetDevice()->createBindingLayout(treeLayoutDesc);
    if (!m_StageResources.sceneTree.bindingLayout) return false;

    // Binding sets are created in _RebuildCullBindings after buffers exist
    return true;
}

bool ComputeRenderPass::_InitImpostorPass() {
    m_StageResources.impostor.vertexShader = m_ShaderFactory->CreateShader(
        "app/ImpostorRenderPass.hlsl", "impostor_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_StageResources.impostor.pixelShader = m_ShaderFactory->CreateShader(
        "app/ImpostorRenderPass.hlsl", "impostor_ps", nullptr, nvrhi::ShaderType::Pixel);
    m_StageResources.impostor.bakeVertexShader = m_ShaderFactory->CreateShader(
        "app/ImpostorBake.hlsl", "bake_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_StageResources.impostor.bakePixelShader = m_ShaderFactory->CreateShader(
        "app/ImpostorBake.hlsl", "bake_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_StageResources.impostor.vertexShader || !m_StageResources.impostor.pixelShader
        || !m_StageResources.impostor.bakeVertexShader || !m_StageResources.impostor.bakePixelShader)
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
    m_StageResources.impostor.bakeInputLayout = GetDevice()->createInputLayout(
        attributes, uint32_t(std::size(attributes)), m_StageResources.impostor.bakeVertexShader);
    if (!m_StageResources.impostor.bakeInputLayout) return false;

    nvrhi::BindingLayoutDesc layoutDesc;
    layoutDesc.visibility = nvrhi::ShaderType::All;
    layoutDesc.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(compute_reg::Impostor::kCB_Frame),
        nvrhi::BindingLayoutItem::PushConstants(compute_reg::Impostor::kPushC_AssetIndex, sizeof(uint32_t)),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_Vis),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_Instances),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_SlotOffsets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_CullData),
        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::Impostor::kTex_Albedo),
        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::Impostor::kTex_Normal),
        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::Impostor::kTex_Depth),
        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::Impostor::kTex_ShadowMap),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Impostor::kSRV_AssetDims),
        nvrhi::BindingLayoutItem::Sampler(compute_reg::Impostor::kSampler_Main),
        nvrhi::BindingLayoutItem::Sampler(compute_reg::Impostor::kSampler_Shadow),
        nvrhi::BindingLayoutItem::Sampler(compute_reg::Impostor::kSampler_Depth),
    };
    m_StageResources.impostor.bindingLayout = GetDevice()->createBindingLayout(layoutDesc);
    if (!m_StageResources.impostor.bindingLayout) return false;

    m_StageResources.impostor.sampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
            .setAllFilters(true));
    if (!m_StageResources.impostor.sampler) return false;

    // Point-clamp sampler for the depth atlas. Bilinear filtering across
    // silhouette edges averages cleared background depth into surface depth,
    // producing fat banded SV_Depth artifacts on the parallax-displaced cards.
    m_StageResources.impostor.depthSampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
            .setAllFilters(false));
    if (!m_StageResources.impostor.depthSampler) return false;

    nvrhi::BindingLayoutDesc bakeLayoutDesc;
    bakeLayoutDesc.visibility = nvrhi::ShaderType::All;
    bakeLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(compute_reg::ImpostorBake::kCB_Bake),
        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::ImpostorBake::kTex_Diffuse),
        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::ImpostorBake::kTex_NormalMap),
        nvrhi::BindingLayoutItem::Sampler(compute_reg::ImpostorBake::kSampler_Main),
    };
    m_StageResources.impostor.bakeBindingLayout = GetDevice()->createBindingLayout(bakeLayoutDesc);
    if (!m_StageResources.impostor.bakeBindingLayout) return false;

    m_StageResources.impostor.bakeConstantBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(c_ImpostorBakeCBSize)
            .setIsConstantBuffer(true)
            .setIsVolatile(true)
            .setMaxVersions(std::max(1u, static_cast<uint32_t>(m_GPUAssets.size()) * k_ImpostorViewCount))
            .setDebugName("ImpostorBakeCB")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer));
    if (!m_StageResources.impostor.bakeConstantBuffer) return false;

    return true;
}

bool ComputeRenderPass::_BakeImpostors(nvrhi::ICommandList* commandList) {
    if (!commandList || !m_StageResources.impostor.bakeVertexShader || !m_StageResources.impostor.bakePixelShader)
        return false;

    auto device = GetDevice();
    const uint32_t numAssets = static_cast<uint32_t>(m_GPUAssets.size());
    const uint32_t numSlices = std::max(1u, numAssets * k_ImpostorViewCount);

    if (!m_StageResources.impostor.bakeConstantBuffer
        || m_StageResources.impostor.bakeConstantBuffer->getDesc().maxVersions < numSlices)
    {
        m_StageResources.impostor.bakeConstantBuffer = device->createBuffer(
            nvrhi::BufferDesc()
                .setByteSize(c_ImpostorBakeCBSize)
                .setIsConstantBuffer(true)
                .setIsVolatile(true)
                .setMaxVersions(numSlices)
                .setDebugName("ImpostorBakeCB")
                .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer));
        if (!m_StageResources.impostor.bakeConstantBuffer)
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

        m_StageResources.impostor.assetDimsBuffer = device->createBuffer(
            nvrhi::BufferDesc()
                .setByteSize(assetDims.size() * sizeof(dm::float4))
                .setStructStride(sizeof(dm::float4))
                .setDebugName("ImpostorAssetDims")
                .setCanHaveUAVs(false)
                .setInitialState(nvrhi::ResourceStates::CopyDest));
        if (!m_StageResources.impostor.assetDimsBuffer)
            return false;

        commandList->beginTrackingBufferState(m_StageResources.impostor.assetDimsBuffer, nvrhi::ResourceStates::CopyDest);
        commandList->writeBuffer(m_StageResources.impostor.assetDimsBuffer,
            assetDims.data(), assetDims.size() * sizeof(dm::float4));
        commandList->setPermanentBufferState(m_StageResources.impostor.assetDimsBuffer, nvrhi::ResourceStates::ShaderResource);
    }

    m_StageResources.impostor.albedoAlphaTexture = device->createTexture(
        nvrhi::TextureDesc()
            .setWidth(k_ImpostorBakeResolution)
            .setHeight(k_ImpostorBakeResolution)
            .setArraySize(numSlices)
            .setDimension(nvrhi::TextureDimension::Texture2DArray)
            .setFormat(nvrhi::Format::RGBA8_UNORM)
            .setIsRenderTarget(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::RenderTarget)
            .setDebugName("ImpostorAlbedoAlphaArray"));

    m_StageResources.impostor.normalTexture = device->createTexture(
        nvrhi::TextureDesc()
            .setWidth(k_ImpostorBakeResolution)
            .setHeight(k_ImpostorBakeResolution)
            .setArraySize(numSlices)
            .setDimension(nvrhi::TextureDimension::Texture2DArray)
            .setFormat(nvrhi::Format::RGBA8_UNORM)
            .setIsRenderTarget(true)
            // .setClearValue(nvrhi::Color(0.5f, 0.5f, 1.f, 0.f))
            .enableAutomaticStateTracking(nvrhi::ResourceStates::RenderTarget)
            .setDebugName("ImpostorNormalArray"));

    m_StageResources.impostor.depthTexture = device->createTexture(
        nvrhi::TextureDesc()
            .setWidth(k_ImpostorBakeResolution)
            .setHeight(k_ImpostorBakeResolution)
            .setArraySize(numSlices)
            .setDimension(nvrhi::TextureDimension::Texture2DArray)
            .setFormat(nvrhi::Format::D32)
            .setIsRenderTarget(true)
            // .setClearValue(nvrhi::Color(1.f))
            .enableAutomaticStateTracking(nvrhi::ResourceStates::DepthWrite)
            .setDebugName("ImpostorDepthArray"));

    if (!m_StageResources.impostor.albedoAlphaTexture || !m_StageResources.impostor.normalTexture || !m_StageResources.impostor.depthTexture)
        return false;

    // Framebuffers for each array slice are created just-in-time inside the bake
    // loop below. They consume RTV/DSV descriptor heap slots, so persisting one
    // per (asset × view) blew the heap once asset counts grew. They're only
    // needed during this single bake, so let them go out of scope per-slice.
    auto makeBakeFramebuffer = [&](uint32_t slice) {
        nvrhi::FramebufferDesc fbDesc;
        fbDesc
            .addColorAttachment(nvrhi::FramebufferAttachment()
                .setTexture(m_StageResources.impostor.albedoAlphaTexture)
                .setArraySlice(slice))
            .addColorAttachment(nvrhi::FramebufferAttachment()
                .setTexture(m_StageResources.impostor.normalTexture)
                .setArraySlice(slice))
            .setDepthAttachment(nvrhi::FramebufferAttachment()
                .setTexture(m_StageResources.impostor.depthTexture)
                .setArraySlice(slice));
        return device->createFramebuffer(fbDesc);
    };

    nvrhi::FramebufferHandle pipelineProtoFB = makeBakeFramebuffer(0);
    if (!pipelineProtoFB)
        return false;

    m_StageResources.impostor.bakeBindingSets.resize(m_StageResources.sceneTree.textureSets.size());
    for (size_t i = 0; i < m_StageResources.sceneTree.textureSets.size(); i++) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(compute_reg::ImpostorBake::kCB_Bake, m_StageResources.impostor.bakeConstantBuffer),
            nvrhi::BindingSetItem::Texture_SRV(compute_reg::ImpostorBake::kTex_Diffuse, m_StageResources.sceneTree.textureSets[i].diffuse),
            nvrhi::BindingSetItem::Texture_SRV(compute_reg::ImpostorBake::kTex_NormalMap, m_StageResources.sceneTree.textureSets[i].normalMap),
            nvrhi::BindingSetItem::Sampler(compute_reg::ImpostorBake::kSampler_Main, m_StageResources.sceneTree.sampler),
        };
        m_StageResources.impostor.bakeBindingSets[i] = device->createBindingSet(bsd, m_StageResources.impostor.bakeBindingLayout);
        if (!m_StageResources.impostor.bakeBindingSets[i])
            return false;
    }

    nvrhi::GraphicsPipelineDesc psoDesc;
    psoDesc.VS = m_StageResources.impostor.bakeVertexShader;
    psoDesc.PS = m_StageResources.impostor.bakePixelShader;
    psoDesc.inputLayout = m_StageResources.impostor.bakeInputLayout;
    psoDesc.bindingLayouts = { m_StageResources.impostor.bakeBindingLayout };
    psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
    psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    psoDesc.renderState.rasterState.setCullNone();
    m_StageResources.impostor.bakePipeline = device->createGraphicsPipeline(
        psoDesc, pipelineProtoFB->getFramebufferInfo());
    if (!m_StageResources.impostor.bakePipeline)
        return false;

    nvrhi::GraphicsState state;
    state.pipeline = m_StageResources.impostor.bakePipeline;
    state.viewport.addViewportAndScissorRect(nvrhi::Viewport(
        float(k_ImpostorBakeResolution), float(k_ImpostorBakeResolution)));

    for (uint32_t ai = 0; ai < numAssets; ai++) {
        const auto& asset = m_GPUAssets[ai];
        if (asset.lods.empty() || asset.textureSetIdx >= m_StageResources.impostor.bakeBindingSets.size())
            continue;

        const auto& lod = asset.lods[0];
        if (lod.indexCount == 0)
            continue;

        state.bindings = { m_StageResources.impostor.bakeBindingSets[asset.textureSetIdx] };
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
            nvrhi::FramebufferHandle framebuffer = (slice == 0)
                ? pipelineProtoFB
                : makeBakeFramebuffer(slice);
            if (!framebuffer)
                return false;

            nvrhi::utils::ClearColorAttachment(commandList, framebuffer, 0, nvrhi::Color(0.f, 0.f, 0.f, 0.f));
            nvrhi::utils::ClearColorAttachment(commandList, framebuffer, 1, nvrhi::Color(0.5f, 0.5f, 1.f, 0.f));
            nvrhi::utils::ClearDepthStencilAttachment(commandList, framebuffer, 1.f, 0);

            ImpostorBakeCB cb;
            cb.mvp = BuildtoProjTransform(lod.bbox, vi);
            commandList->writeBuffer(m_StageResources.impostor.bakeConstantBuffer, &cb, sizeof(cb));

            state.framebuffer = framebuffer;
            commandList->setGraphicsState(state);
            commandList->drawIndexed(nvrhi::DrawArguments().setVertexCount(lod.indexCount));
        }
    }

    // Per-asset atlas-sheet debug textures for ImGui inspection of the baked atlas.
    // Created here (after the bake) so the copies happen while the array textures
    // are still in RenderTarget/DepthWrite state — copyTexture handles the transition.
    {
        m_StageResources.impostor.debugAlbedoAtlasTexture = device->createTexture(
            nvrhi::TextureDesc()
                .setWidth(k_ImpostorBakeResolution * k_ImpostorAzimuthViews)
                .setHeight(k_ImpostorBakeResolution * k_ImpostorElevationViews)
                .setFormat(nvrhi::Format::RGBA8_UNORM)
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true)
                .setDebugName("ImpostorAlbedoAtlasDebug"));
        m_StageResources.impostor.debugNormalAtlasTexture = device->createTexture(
            nvrhi::TextureDesc()
                .setWidth(k_ImpostorBakeResolution * k_ImpostorAzimuthViews)
                .setHeight(k_ImpostorBakeResolution * k_ImpostorElevationViews)
                .setFormat(nvrhi::Format::RGBA8_UNORM)
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true)
                .setDebugName("ImpostorNormalAtlasDebug"));
        m_StageResources.impostor.debugDepthAtlasTexture = device->createTexture(
            nvrhi::TextureDesc()
                .setWidth(k_ImpostorBakeResolution * k_ImpostorAzimuthViews)
                .setHeight(k_ImpostorBakeResolution * k_ImpostorElevationViews)
                .setFormat(nvrhi::Format::R32_FLOAT)
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true)
                .setDebugName("ImpostorDepthAtlasDebug"));
        if (!m_StageResources.impostor.debugAlbedoAtlasTexture
            || !m_StageResources.impostor.debugNormalAtlasTexture
            || !m_StageResources.impostor.debugDepthAtlasTexture)
            return false;
    }

    m_StageResources.impostor.pipeline = nullptr;
    return true;
}

// bool ComputeRenderPass::_InitTimerQueries() {
//     for (uint32_t i = 0; i < k_QueuedFrames; i++)
//         m_GpuTimers[i] = GetDevice()->createTimerQuery();
//     return true;
// }

bool ComputeRenderPass::_InitShadowPass() {
    m_StageResources.shadow.depthTexture = GetDevice()->createTexture(
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
    if (!m_StageResources.shadow.depthTexture) return false;

    for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
        m_StageResources.shadow.framebuffers[c] = GetDevice()->createFramebuffer(
            nvrhi::FramebufferDesc().setDepthAttachment(
                nvrhi::FramebufferAttachment()
                    .setTexture(m_StageResources.shadow.depthTexture)
                    .setArraySlice(c)));
        if (!m_StageResources.shadow.framebuffers[c]) return false;

        m_StageResources.shadow.debugTextures[c] = GetDevice()->createTexture(
            nvrhi::TextureDesc()
                .setWidth(k_ShadowRes).setHeight(k_ShadowRes)
                .setFormat(nvrhi::Format::R32_FLOAT)
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true)
                .setDebugName("ShadowCascadeDebug_" + std::to_string(c))
        );
    }

    m_StageResources.shadow.treeVS = m_ShaderFactory->CreateShader(
        "app/shadow_compute.hlsl", "tree_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_StageResources.shadow.terrainVS = m_ShaderFactory->CreateShader(
        "app/shadow_compute.hlsl", "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    if (!m_StageResources.shadow.treeVS || !m_StageResources.shadow.terrainVS) return false;

    // Tree shadow input layout - position-only (instance data via SRV indirection)
    nvrhi::VertexAttributeDesc treeShadowAttrs[] = {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(0)
            .setElementStride(sizeof(dm::float3)),
    };
    m_StageResources.shadow.treeInputLayout = GetDevice()->createInputLayout(
        treeShadowAttrs, uint32_t(std::size(treeShadowAttrs)), m_StageResources.shadow.treeVS);
    if (!m_StageResources.shadow.treeInputLayout) return false;

    // Terrain shadow input layout
    nvrhi::VertexAttributeDesc terrainShadowAttrs[] = {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(Scene::TerrainVertex, pos))
            .setBufferIndex(0)
            .setElementStride(sizeof(Scene::TerrainVertex)),
    };
    m_StageResources.shadow.terrainInputLayout = GetDevice()->createInputLayout(
        terrainShadowAttrs, uint32_t(std::size(terrainShadowAttrs)), m_StageResources.shadow.terrainVS);
    if (!m_StageResources.shadow.terrainInputLayout) return false;

    m_StageResources.shadow.comparisonSampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setMinFilter(true)
            .setMagFilter(true)
            .setMipFilter(false)
            .setReductionType(nvrhi::SamplerReductionType::Comparison)
            .setAllAddressModes(nvrhi::SamplerAddressMode::Border)
            .setBorderColor(nvrhi::Color(1.f))
    );
    if (!m_StageResources.shadow.comparisonSampler) return false;

    // Shadow binding layout
    nvrhi::BindingLayoutDesc shadowLayoutDesc;
    shadowLayoutDesc.visibility = nvrhi::ShaderType::All;
    shadowLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(compute_reg::Shadow::kCB_Frame),
        nvrhi::BindingLayoutItem::PushConstants(compute_reg::Shadow::kPushC_AssetCascade, sizeof(uint32_t) * 2),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Shadow::kSRV_Vis),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Shadow::kSRV_Instances),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::Shadow::kSRV_SlotOffsets),
    };
    m_StageResources.shadow.bindingLayout = GetDevice()->createBindingLayout(shadowLayoutDesc);
    if (!m_StageResources.shadow.bindingLayout) return false;

    // Binding set created in _RebuildCullBindings after buffers exist
    return true;
}

bool ComputeRenderPass::_InitTerrainPass(nvrhi::ICommandList* initCL) {
    const auto* terrain = m_Registry.getTerrain();
    if (!terrain) return true;

    const auto& verts   = terrain->getVertices();
    const auto& indices = terrain->getIndices();
    if (verts.empty() || indices.empty()) return true;

    m_StageResources.sceneTerrain.indexCount = static_cast<uint32_t>(indices.size());

    m_StageResources.sceneTerrain.vertexShader = m_ShaderFactory->CreateShader("app/terrain_compute.hlsl", "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_StageResources.sceneTerrain.pixelShader  = m_ShaderFactory->CreateShader("app/terrain_compute.hlsl", "terrain_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_StageResources.sceneTerrain.vertexShader || !m_StageResources.sceneTerrain.pixelShader) return false;

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
    m_StageResources.sceneTerrain.inputLayout = GetDevice()->createInputLayout(
        terrainAttrs, uint32_t(std::size(terrainAttrs)), m_StageResources.sceneTerrain.vertexShader);
    if (!m_StageResources.sceneTerrain.inputLayout) return false;

    nvrhi::BufferDesc vbDesc;
    vbDesc.isVertexBuffer = true;
    vbDesc.byteSize       = verts.size() * sizeof(Scene::TerrainVertex);
    vbDesc.debugName      = "TerrainVB";
    vbDesc.initialState   = nvrhi::ResourceStates::CopyDest;
    m_StageResources.sceneTerrain.vertexBuffer = GetDevice()->createBuffer(vbDesc);
    initCL->beginTrackingBufferState(m_StageResources.sceneTerrain.vertexBuffer, nvrhi::ResourceStates::CopyDest);
    initCL->writeBuffer(m_StageResources.sceneTerrain.vertexBuffer, verts.data(), vbDesc.byteSize);
    initCL->setPermanentBufferState(m_StageResources.sceneTerrain.vertexBuffer, nvrhi::ResourceStates::VertexBuffer);

    nvrhi::BufferDesc ibDesc;
    ibDesc.isIndexBuffer = true;
    ibDesc.byteSize      = indices.size() * sizeof(uint32_t);
    ibDesc.debugName     = "TerrainIB";
    ibDesc.initialState  = nvrhi::ResourceStates::CopyDest;
    m_StageResources.sceneTerrain.indexBuffer = GetDevice()->createBuffer(ibDesc);
    initCL->beginTrackingBufferState(m_StageResources.sceneTerrain.indexBuffer, nvrhi::ResourceStates::CopyDest);
    initCL->writeBuffer(m_StageResources.sceneTerrain.indexBuffer, indices.data(), ibDesc.byteSize);
    initCL->setPermanentBufferState(m_StageResources.sceneTerrain.indexBuffer, nvrhi::ResourceStates::IndexBuffer);

    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(compute_reg::Terrain::kCB_Frame, m_StageResources.frameShared.constantBuffer,
            nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
        nvrhi::BindingSetItem::Sampler(compute_reg::Terrain::kSampler_Shadow, m_StageResources.shadow.comparisonSampler),
        nvrhi::BindingSetItem::Texture_SRV(compute_reg::Terrain::kTex_ShadowMap, m_StageResources.shadow.depthTexture),
    };
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
            bsd, m_StageResources.sceneTerrain.bindingLayout, m_StageResources.sceneTerrain.bindingSet))
        return false;

    return true;
}

bool ComputeRenderPass::_InitHiZShaders() {
    auto device = GetDevice();

    // --- Depth prepass shaders ---
    m_StageResources.depthPrepass.treeVS = m_ShaderFactory->CreateShader(
        "app/DepthPrepass.hlsl", "tree_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_StageResources.depthPrepass.terrainVS = m_ShaderFactory->CreateShader(
        "app/DepthPrepass.hlsl", "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    if (!m_StageResources.depthPrepass.treeVS || !m_StageResources.depthPrepass.terrainVS) return false;

    // Tree input layout - position-only
    nvrhi::VertexAttributeDesc treePrepassAttrs[] = {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION")
            .setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0)
            .setBufferIndex(0)
            .setElementStride(sizeof(dm::float3)),
    };
    m_StageResources.depthPrepass.treeInputLayout = device->createInputLayout(
        treePrepassAttrs, uint32_t(std::size(treePrepassAttrs)), m_StageResources.depthPrepass.treeVS);
    if (!m_StageResources.depthPrepass.treeInputLayout) return false;

    // Terrain input layout - pos+normal+uv (must match VB stride)
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
    m_StageResources.depthPrepass.terrainInputLayout = device->createInputLayout(
        terrainPrepassAttrs, uint32_t(std::size(terrainPrepassAttrs)), m_StageResources.depthPrepass.terrainVS);
    if (!m_StageResources.depthPrepass.terrainInputLayout) return false;

    // Depth prepass binding layout - same shape as shadow pass
    nvrhi::BindingLayoutDesc prepassLayoutDesc;
    prepassLayoutDesc.visibility = nvrhi::ShaderType::All;
    prepassLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(compute_reg::DepthPrepass::kCB_Frame),
        nvrhi::BindingLayoutItem::PushConstants(compute_reg::DepthPrepass::kPushC_Slot, sizeof(uint32_t)),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::DepthPrepass::kSRV_Vis),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::DepthPrepass::kSRV_Instances),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(compute_reg::DepthPrepass::kSRV_SlotOffsets),
    };
    m_StageResources.depthPrepass.bindingLayout = device->createBindingLayout(prepassLayoutDesc);
    if (!m_StageResources.depthPrepass.bindingLayout) return false;

    // --- Hi-Z build shaders ---
    m_StageResources.hiz.copyCS = m_ShaderFactory->CreateShader(
        "app/HiZBuild.hlsl", "HiZCopy", nullptr, nvrhi::ShaderType::Compute);
    m_StageResources.hiz.buildCS = m_ShaderFactory->CreateShader(
        "app/HiZBuild.hlsl", "HiZDownsample", nullptr, nvrhi::ShaderType::Compute);
    if (!m_StageResources.hiz.copyCS || !m_StageResources.hiz.buildCS) return false;

    // Hi-Z build binding layout: push constants + SRV(source) + UAV(dest)
    nvrhi::BindingLayoutDesc hizBuildLayoutDesc;
    hizBuildLayoutDesc.visibility = nvrhi::ShaderType::Compute;
    hizBuildLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::PushConstants(compute_reg::HiZ::kPushC_DestDimensions, sizeof(uint32_t) * compute_reg::HiZ::kPushCDwordCount),
        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::HiZ::kSRV_Source),
        nvrhi::BindingLayoutItem::Texture_UAV(compute_reg::HiZ::kUAV_Dest),
    };
    m_StageResources.hiz.buildBindingLayout = device->createBindingLayout(hizBuildLayoutDesc);
    if (!m_StageResources.hiz.buildBindingLayout) return false;

    // Compute pipelines
    nvrhi::ComputePipelineDesc copyPso;
    copyPso.CS = m_StageResources.hiz.copyCS;
    copyPso.bindingLayouts = { m_StageResources.hiz.buildBindingLayout };
    m_StageResources.hiz.copyPipeline = device->createComputePipeline(copyPso);

    nvrhi::ComputePipelineDesc buildPso;
    buildPso.CS = m_StageResources.hiz.buildCS;
    buildPso.bindingLayouts = { m_StageResources.hiz.buildBindingLayout };
    m_StageResources.hiz.buildPipeline = device->createComputePipeline(buildPso);

    if (!m_StageResources.hiz.copyPipeline || !m_StageResources.hiz.buildPipeline) return false;

    // Point/clamp sampler for Hi-Z reads in the cull shader
    m_StageResources.hiz.pointSampler = device->createSampler(
        nvrhi::SamplerDesc()
            .setAllFilters(false)
            .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
    );
    if (!m_StageResources.hiz.pointSampler) return false;

    // Create 1x1 placeholder Hi-Z texture so binding sets can reference it before first frame
    m_StageResources.hiz.hizTexture = device->createTexture(
        nvrhi::TextureDesc()
            .setWidth(1).setHeight(1)
            .setMipLevels(1)
            .setFormat(nvrhi::Format::RG32_FLOAT)
            .setIsUAV(true)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true)
            .setDebugName("HiZTexture_Placeholder")
    );
    m_StageResources.hiz.numMips = 1;

    return true;
}

bool ComputeRenderPass::_InitSDSMPass() {
    auto device = GetDevice();

    m_StageResources.sdsm.buildCS = m_ShaderFactory->CreateShader(
        "app/SDSMBuildCascades.hlsl", "BuildCascades", nullptr, nvrhi::ShaderType::Compute);
    if (!m_StageResources.sdsm.buildCS) return false;

    nvrhi::BindingLayoutDesc layoutDesc;
    layoutDesc.visibility = nvrhi::ShaderType::Compute;
    layoutDesc.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(compute_reg::SDSM::kCB_Input),
        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::SDSM::kSRV_HiZ),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(compute_reg::SDSM::kUAV_CascadeOut),
    };
    m_StageResources.sdsm.buildBindingLayout = device->createBindingLayout(layoutDesc);
    if (!m_StageResources.sdsm.buildBindingLayout) return false;

    nvrhi::ComputePipelineDesc pso;
    pso.CS = m_StageResources.sdsm.buildCS;
    pso.bindingLayouts = { m_StageResources.sdsm.buildBindingLayout };
    m_StageResources.sdsm.buildPipeline = device->createComputePipeline(pso);
    if (!m_StageResources.sdsm.buildPipeline) return false;

    // Input CB (non-volatile, state-tracked so writeBuffer works inside Render)
    m_StageResources.sdsm.inputCB = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(shader_cb::kSDSMCascadeBuildInputSize)
            .setIsConstantBuffer(true)
            .setDebugName("SDSMInputCB")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
    );
    if (!m_StageResources.sdsm.inputCB) return false;

    // Cascade output - structured UAV, single element holding all per-cascade data
    m_StageResources.sdsm.cascadeDataBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(sizeof(shader_cb::SDSMCascadeBuildOutput))
            .setStructStride(sizeof(shader_cb::SDSMCascadeBuildOutput))
            .setDebugName("SDSMCascadeData")
            .setCanHaveUAVs(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess)
    );
    if (!m_StageResources.sdsm.cascadeDataBuffer) return false;

    // Binding set references the placeholder Hi-Z texture; rebuilt in _EnsureHiZResources
    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(compute_reg::SDSM::kCB_Input, m_StageResources.sdsm.inputCB),
        nvrhi::BindingSetItem::Texture_SRV(compute_reg::SDSM::kSRV_HiZ, m_StageResources.hiz.hizTexture),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(compute_reg::SDSM::kUAV_CascadeOut, m_StageResources.sdsm.cascadeDataBuffer),
    };
    m_StageResources.sdsm.buildBindingSet = device->createBindingSet(bsd, m_StageResources.sdsm.buildBindingLayout);
    if (!m_StageResources.sdsm.buildBindingSet) return false;

    return true;
}

bool ComputeRenderPass::_InitSkyPass() {
    m_StageResources.sky.vertexShader = m_ShaderFactory->CreateShader("app/sky.hlsl", "sky_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_StageResources.sky.pixelShader  = m_ShaderFactory->CreateShader("app/sky.hlsl", "sky_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_StageResources.sky.vertexShader || !m_StageResources.sky.pixelShader) return false;

    nvrhi::BufferDesc cbDesc;
    cbDesc.byteSize         = sizeof(SkyConstants);
    cbDesc.isConstantBuffer = true;
    cbDesc.isVolatile       = true;
    cbDesc.maxVersions      = 16;
    cbDesc.debugName        = "SkyConstants";
    m_StageResources.sky.constantBuffer = GetDevice()->createBuffer(cbDesc);
    if (!m_StageResources.sky.constantBuffer) return false;

    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(compute_reg::Sky::kCB_Sky, m_StageResources.sky.constantBuffer),
    };
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
            bsd, m_StageResources.sky.bindingLayout, m_StageResources.sky.bindingSet))
        return false;

    return true;
}





