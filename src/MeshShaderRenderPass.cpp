#include "include/MeshShaderRenderPass.hpp"
#include "include/Render.hpp"
#include "include/Globals.hpp"
#include "include/macros.h"

#include <nvrhi/utils.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/math/math.h>

#include <meshoptimizer.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>

using namespace donut::math;
using namespace Xylem;

namespace {

// Draw-pass register assignments — kept in lockstep with MeshShaderPass.hlsl.
constexpr uint32_t k_CB_Draw            = 0;
constexpr uint32_t k_PushC_Draw         = 1;   // root constant b1 (slotIdx)
constexpr uint32_t k_PushCBytes         = sizeof(uint32_t);
constexpr uint32_t k_SRV_Positions      = 0;
constexpr uint32_t k_SRV_Normals        = 1;
constexpr uint32_t k_SRV_Tangents       = 2;
constexpr uint32_t k_SRV_Bitangents     = 3;
constexpr uint32_t k_SRV_UVs            = 4;
constexpr uint32_t k_SRV_MVertIdx       = 5;
constexpr uint32_t k_SRV_MPrimIdx       = 6;  // RawBuffer
constexpr uint32_t k_SRV_Meshlets       = 7;
constexpr uint32_t k_SRV_AssetLods      = 8;
constexpr uint32_t k_SRV_VisBuf         = 9;
constexpr uint32_t k_SRV_SlotOffsets    = 10;
constexpr uint32_t k_SRV_SlotCounts     = 11;
constexpr uint32_t k_SRV_Instances      = 12;
constexpr uint32_t k_SRV_ASInvocsPerSlot= 13;
constexpr uint32_t k_SRV_Diffuse        = 14;
constexpr uint32_t k_SRV_NormalMap      = 15;
constexpr uint32_t k_Sampler_Draw       = 0;

// Cull-pass register assignments — match MeshCullCS.hlsl.
constexpr uint32_t k_CB_Cull            = 0;
constexpr uint32_t k_SRV_CullRegion     = 0;
constexpr uint32_t k_SRV_CullInstance   = 1;
constexpr uint32_t k_SRV_CullSlotOffs   = 2;
constexpr uint32_t k_SRV_CullChunks     = 3;
constexpr uint32_t k_SRV_CullHiZ        = 4;   // dummy (hizEnabled=0)
constexpr uint32_t k_UAV_RegionVisible  = 0;
constexpr uint32_t k_UAV_CountBuf       = 1;
constexpr uint32_t k_UAV_VisBuf         = 2;
constexpr uint32_t k_UAV_DispatchArgs   = 3;   // raw, 16 bytes per slot
constexpr uint32_t k_Sampler_Cull       = 0;

// Root parameter index of the push-constant block in the mesh pipeline's root
// signature. NVRHI emits root-constants first (see d3d12-resource-bindings.cpp
// BindingLayout construction: rootConstants before volatileCBs / tables), and
// the mesh pipeline has a single binding layout so the offset is zero.
constexpr uint32_t k_PushC_RootParamIdx = 0;

} // namespace

// ===========================================================================
// Init
// ===========================================================================

bool MeshShaderRenderPass::Init() {
    engine::CommonRenderPasses commonPasses(GetDevice(), m_ShaderFactory);

    if (!_InitShared())         return false;
    if (!_InitDrawResources())  return false;
    if (!_InitCullResources())  return false;

    {
        auto initCL = GetDevice()->createCommandList();
        initCL->open();

        _RebuildMeshletMegabuffers(initCL);
        _UploadMeshletMegabuffers(initCL);

        if (!_LoadBarkTextures(initCL, commonPasses)) { initCL->close(); return false; }

        _BuildRegionWindows();
        _BuildSlotLayout();
        _UploadCullBuffers(initCL);

        _RebuildCullBindingSet();
        _RebuildDrawBindingSet();

        initCL->close();
        GetDevice()->executeCommandList(initCL);
    }

    m_CommandList = GetDevice()->createCommandList();
    return true;
}

bool MeshShaderRenderPass::_InitShared() {
    m_Shared.constantBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(Render::c_CullConstantBufferSize)
            .setIsConstantBuffer(true)
            .setDebugName("MeshShaderPass_CB")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
    );
    if (!m_Shared.constantBuffer) return false;

    // Readback ring for countBuffer → UI stats.
    for (uint32_t i = 0; i < k_QueuedFrames; i++) {
        m_ReadbackBuffers[i] = GetDevice()->createBuffer(
            nvrhi::BufferDesc()
                .setByteSize(sizeof(uint32_t))  // sized in _UploadCullBuffers once m_NumSlots is known
                .setCpuAccess(nvrhi::CpuAccessMode::Read)
                .setInitialState(nvrhi::ResourceStates::CopyDest)
                .setKeepInitialState(true)
                .setDebugName("MeshCullCountReadback_" + std::to_string(i))
        );
    }
    return true;
}

bool MeshShaderRenderPass::_InitDrawResources() {
    if (!m_ShaderFactory) { log::error("MeshShaderRenderPass: no ShaderFactory"); return false; }

    m_Draw.amplificationShader = m_ShaderFactory->CreateShader("app/MeshShaderPass.hlsl",
        "main_as", nullptr, nvrhi::ShaderType::Amplification);
    m_Draw.meshShader = m_ShaderFactory->CreateShader("app/MeshShaderPass.hlsl",
        "main_ms", nullptr, nvrhi::ShaderType::Mesh);
    m_Draw.pixelShader = m_ShaderFactory->CreateShader("app/MeshShaderPass.hlsl",
        "main_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_Draw.amplificationShader || !m_Draw.meshShader || !m_Draw.pixelShader) {
        log::error("MeshShaderRenderPass: shader compile failed");
        return false;
    }

    m_Draw.sampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Wrap)
            .setAllFilters(true)
            .setMaxAnisotropy(8.f));
    if (!m_Draw.sampler) return false;

    nvrhi::BindingLayoutDesc bld;
    bld.visibility = nvrhi::ShaderType::All;
    bld.bindings = {
        nvrhi::BindingLayoutItem::PushConstants(k_PushC_Draw, k_PushCBytes),
        nvrhi::BindingLayoutItem::ConstantBuffer(k_CB_Draw),

        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_Positions),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_Normals),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_Tangents),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_Bitangents),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_UVs),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_MVertIdx),
        nvrhi::BindingLayoutItem::RawBuffer_SRV(k_SRV_MPrimIdx),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_Meshlets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_AssetLods),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_VisBuf),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_SlotOffsets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_SlotCounts),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_Instances),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_ASInvocsPerSlot),
        nvrhi::BindingLayoutItem::Texture_SRV(k_SRV_Diffuse),
        nvrhi::BindingLayoutItem::Texture_SRV(k_SRV_NormalMap),
        nvrhi::BindingLayoutItem::Sampler(k_Sampler_Draw),
    };
    m_Draw.bindingLayout = GetDevice()->createBindingLayout(bld);
    return m_Draw.bindingLayout != nullptr;
}

bool MeshShaderRenderPass::_InitCullResources() {
    if (!m_ShaderFactory) return false;

    m_Cull.mainCS   = m_ShaderFactory->CreateShader("app/MeshCullCS.hlsl",
        "MeshCullMain",   nullptr, nvrhi::ShaderType::Compute);
    m_Cull.regionCS = m_ShaderFactory->CreateShader("app/MeshCullCS.hlsl",
        "MeshCullRegion", nullptr, nvrhi::ShaderType::Compute);
    if (!m_Cull.mainCS || !m_Cull.regionCS) {
        log::error("MeshShaderRenderPass: MeshCullCS compile failed");
        return false;
    }

    nvrhi::BindingLayoutDesc bld;
    bld.visibility = nvrhi::ShaderType::All;
    bld.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(k_CB_Cull),

        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_CullRegion),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_CullInstance),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_CullSlotOffs),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_CullChunks),

        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(k_UAV_RegionVisible),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(k_UAV_CountBuf),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(k_UAV_VisBuf),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(k_UAV_DispatchArgs),

        nvrhi::BindingLayoutItem::Texture_SRV(k_SRV_CullHiZ),  // dummy
        nvrhi::BindingLayoutItem::Sampler(k_Sampler_Cull),
    };
    m_Cull.bindingLayout = GetDevice()->createBindingLayout(bld);
    if (!m_Cull.bindingLayout) return false;

    nvrhi::ComputePipelineDesc psoDescRegion;
    psoDescRegion.CS = m_Cull.regionCS;
    psoDescRegion.bindingLayouts = { m_Cull.bindingLayout };
    m_Cull.regionPipeline = GetDevice()->createComputePipeline(psoDescRegion);

    nvrhi::ComputePipelineDesc psoDescMain;
    psoDescMain.CS = m_Cull.mainCS;
    psoDescMain.bindingLayouts = { m_Cull.bindingLayout };
    m_Cull.mainPipeline = GetDevice()->createComputePipeline(psoDescMain);

    return m_Cull.mainPipeline != nullptr && m_Cull.regionPipeline != nullptr;
}

bool MeshShaderRenderPass::_LoadBarkTextures(nvrhi::ICommandList* initCL,
                                             engine::CommonRenderPasses& commonPasses) {
    const auto& barkTextureSets = m_Registry.getBarkTextureSets();
    if (barkTextureSets.empty()) {
        log::error("MeshShaderRenderPass: no bark texture sets registered");
        return false;
    }

    engine::TextureCache textureCache(GetDevice(), std::make_shared<vfs::NativeFileSystem>(), nullptr);
    m_Draw.textureSets.resize(barkTextureSets.size());

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
        m_Draw.textureSets[i].diffuse   = diffLoaded ? diffLoaded->texture : nullptr;
        m_Draw.textureSets[i].normalMap = normLoaded ? normLoaded->texture : nullptr;
        if (!m_Draw.textureSets[i].diffuse || !m_Draw.textureSets[i].normalMap) {
            log::error("MeshShaderRenderPass: bark texture load failed");
            return false;
        }
    }
    return true;
}

// ===========================================================================
// Meshlet mega-buffer build — CPU side
// ===========================================================================

void MeshShaderRenderPass::_RebuildMeshletMegabuffers(nvrhi::ICommandList* /*cl*/) {
    const auto& assets      = m_Registry.getAssets();
    const uint32_t numLods  = static_cast<uint32_t>(m_Registry.getLodDistances().size());

    m_MeshletMegabuffers.meshOffsets.assign(assets.size() * static_cast<size_t>(numLods), {});

    m_MeshletMegabuffers.meshletDescs.clear();

    auto& vertexAttributes = m_MeshletMegabuffers.vertexAttributes;
    vertexAttributes.positions.clear();
    vertexAttributes.normals.clear();
    vertexAttributes.tangents.clear();
    vertexAttributes.bitangents.clear();
    vertexAttributes.uvs.clear();

    auto& meshletVerts = m_MeshletMegabuffers.localVertIndices;
    auto& meshletTris  = m_MeshletMegabuffers.localTriIndices;
    auto& meshletDescs = m_MeshletMegabuffers.meshletDescs;
    meshletVerts.clear();
    meshletTris.clear();

    for (size_t ai = 0; ai < assets.size(); ai++) {
        const auto& asset = assets[ai];
        for (uint32_t li = 0; li < asset.lods.size(); li++) {
            const auto& lod = asset.lods[li];

            auto& perMeshOffsets = m_MeshletMegabuffers.meshOffsets[ai*numLods+li];
            perMeshOffsets.meshletOffset       = static_cast<uint32_t>(meshletDescs.size());
            perMeshOffsets.vertexAttribOffset  = static_cast<uint32_t>(vertexAttributes.positions.size());
            perMeshOffsets.meshletVertOffset   = static_cast<uint32_t>(meshletVerts.size());
            perMeshOffsets.meshletTriOffset    = static_cast<uint32_t>(meshletTris.size());

            vertexAttributes.positions.insert(vertexAttributes.positions.end(),
                lod.positions.begin(),  lod.positions.end());
            vertexAttributes.normals.insert(vertexAttributes.normals.end(),
                lod.normals.begin(),    lod.normals.end());
            vertexAttributes.tangents.insert(vertexAttributes.tangents.end(),
                lod.tangents.begin(),   lod.tangents.end());
            vertexAttributes.bitangents.insert(vertexAttributes.bitangents.end(),
                lod.bitangents.begin(), lod.bitangents.end());
            vertexAttributes.uvs.insert(vertexAttributes.uvs.end(),
                lod.uvs.begin(),        lod.uvs.end());

            const size_t maxMeshlets = meshopt_buildMeshletsBound(
                lod.indices.size(), Render::k_MaxMeshletVerts, Render::k_MaxMeshletPrims);
            std::vector<meshopt_Meshlet> ml(maxMeshlets);
            std::vector<unsigned int>    mlVerts(maxMeshlets * Render::k_MaxMeshletVerts);
            std::vector<unsigned char>   mlTris(maxMeshlets * Render::k_MaxMeshletPrims * 3);

            const float* posPtr = lod.positions.empty() ? nullptr
                : reinterpret_cast<const float*>(lod.positions.data());

            size_t meshletCount = 0;
            if (!lod.indices.empty() && posPtr) {
                meshletCount = meshopt_buildMeshlets(
                    ml.data(), mlVerts.data(), mlTris.data(),
                    lod.indices.data(), lod.indices.size(),
                    posPtr, lod.positions.size(), sizeof(dm::float3),
                    Render::k_MaxMeshletVerts, Render::k_MaxMeshletPrims, 0.25f);
            }
            ml.resize(meshletCount);

            meshletDescs.reserve(meshletDescs.size() + meshletCount);
            for (size_t mi = 0; mi < meshletCount; mi++) {
                const meshopt_Meshlet& m = ml[mi];
                meshopt_Bounds b = meshopt_computeMeshletBounds(
                    mlVerts.data() + m.vertex_offset,
                    mlTris.data()  + m.triangle_offset,
                    m.triangle_count,
                    posPtr, lod.positions.size(), sizeof(dm::float3));

                Render::MeshletDesc desc;
                desc.vertOffset = static_cast<uint32_t>(meshletVerts.size()) - perMeshOffsets.meshletVertOffset;
                desc.vertCount  = m.vertex_count;
                desc.triOffset  = static_cast<uint32_t>(meshletTris.size())  - perMeshOffsets.meshletTriOffset;
                desc.triCount   = m.triangle_count;
                desc.bounds         = dm::float4(b.center[0],   b.center[1],   b.center[2],   b.radius);
                desc.coneApex       = dm::float4(b.cone_apex[0],b.cone_apex[1],b.cone_apex[2], 0.f);
                desc.coneAxisCutoff = dm::float4(b.cone_axis[0],b.cone_axis[1],b.cone_axis[2], b.cone_cutoff);
                meshletDescs.push_back(desc);

                meshletVerts.insert(meshletVerts.end(),
                    mlVerts.data() + m.vertex_offset,
                    mlVerts.data() + m.vertex_offset + m.vertex_count);

                meshletTris.insert(meshletTris.end(),
                    mlTris.data() + m.triangle_offset,
                    mlTris.data() + m.triangle_offset + m.triangle_count * 3);
            }

            // Pad meshletPrimIdx slice to 4-byte alignment for ByteAddressBuffer.
            size_t misalign = (meshletTris.size() - perMeshOffsets.meshletTriOffset) % 4;
            if (misalign) meshletTris.insert(meshletTris.end(), 4 - misalign, 0);

            perMeshOffsets.meshletCount = static_cast<uint32_t>(meshletCount);
        }
    }
}

void MeshShaderRenderPass::_UploadMeshletMegabuffers(nvrhi::ICommandList* cl) {
    auto& vertexAttributes = m_MeshletMegabuffers.vertexAttributes;
    auto& meshletVerts     = m_MeshletMegabuffers.localVertIndices;
    auto& meshletTris      = m_MeshletMegabuffers.localTriIndices;
    auto& meshletDescs     = m_MeshletMegabuffers.meshletDescs;

    const uint32_t totalVerts     = static_cast<uint32_t>(vertexAttributes.positions.size());
    const uint32_t totalMeshlets  = static_cast<uint32_t>(meshletDescs.size());
    const uint32_t totalVertIdx   = static_cast<uint32_t>(meshletVerts.size());
    const uint32_t totalPrimBytes = static_cast<uint32_t>(meshletTris.size());
    const uint32_t numAssetLods   = static_cast<uint32_t>(m_MeshletMegabuffers.meshOffsets.size());

    auto makeSrv = [&](nvrhi::BufferHandle& h, const void* data, size_t bytes, uint32_t stride,
                       const char* name, bool raw) {
        nvrhi::BufferDesc d;
        d.byteSize      = std::max<size_t>(bytes, stride ? stride : 4);
        d.debugName     = name;
        d.initialState  = nvrhi::ResourceStates::CopyDest;
        if (raw) {
            d.canHaveRawViews = true;
        } else {
            d.structStride = stride;
        }
        h = GetDevice()->createBuffer(d);
        cl->beginTrackingBufferState(h, nvrhi::ResourceStates::CopyDest);
        if (data && bytes > 0) cl->writeBuffer(h, data, bytes);
        cl->setPermanentBufferState(h, nvrhi::ResourceStates::ShaderResource);
    };

    makeSrv(m_Meshlet.positions,      vertexAttributes.positions.data(),  totalVerts * sizeof(dm::float3),
            sizeof(dm::float3), "Mesh_Positions",  false);
    makeSrv(m_Meshlet.normals,        vertexAttributes.normals.data(),    totalVerts * sizeof(dm::float3),
            sizeof(dm::float3), "Mesh_Normals",    false);
    makeSrv(m_Meshlet.tangents,       vertexAttributes.tangents.data(),   totalVerts * sizeof(dm::float3),
            sizeof(dm::float3), "Mesh_Tangents",   false);
    makeSrv(m_Meshlet.bitangents,     vertexAttributes.bitangents.data(), totalVerts * sizeof(dm::float3),
            sizeof(dm::float3), "Mesh_Bitangents", false);
    makeSrv(m_Meshlet.uvs,            vertexAttributes.uvs.data(),        totalVerts * sizeof(dm::float2),
            sizeof(dm::float2), "Mesh_UVs",        false);
    makeSrv(m_Meshlet.meshletVertIdx, meshletVerts.data(), totalVertIdx * sizeof(uint32_t),
            sizeof(uint32_t),   "Mesh_MeshletVertIdx", false);
    makeSrv(m_Meshlet.meshletPrimIdx, meshletTris.data(), totalPrimBytes,
            0,                  "Mesh_MeshletTriIdx", true);
    makeSrv(m_Meshlet.meshletDescs,   meshletDescs.data(),       totalMeshlets * sizeof(Render::MeshletDesc),
            sizeof(Render::MeshletDesc),   "Mesh_Meshlets", false);
    makeSrv(m_Meshlet.assetLodRanges, m_MeshletMegabuffers.meshOffsets.data(),
            numAssetLods * sizeof(Render::MeshOffsets),
            sizeof(Render::MeshOffsets), "Mesh_Offsets", false);

    m_Meshlet.totalVertices     = totalVerts;
    m_Meshlet.totalMeshlets     = totalMeshlets;
    m_Meshlet.totalVertIdx      = totalVertIdx;
    m_Meshlet.totalPrimIdxBytes = totalPrimBytes;
    m_Meshlet.numAssetLods      = numAssetLods;

    // UI reporting
    size_t totalBytes =
        totalVerts * (sizeof(dm::float3) * 4 + sizeof(dm::float2)) +
        totalVertIdx * sizeof(uint32_t) +
        totalPrimBytes +
        totalMeshlets * sizeof(Render::MeshletDesc) +
        numAssetLods * sizeof(Render::MeshOffsets);
    m_UI.meshletMegaBufferMB = static_cast<float>(totalBytes) / (1024.f * 1024.f);
    m_UI.totalMeshletCount   = totalMeshlets;
}

// ===========================================================================
// Region windows + slot layout — mirrors ComputeRenderPass
// ===========================================================================

void MeshShaderRenderPass::_BuildRegionWindows() {
    const auto& regions = m_Registry.getRegions();
    const auto& assets  = m_Registry.getAssets();
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

    const uint32_t numLods = static_cast<uint32_t>(m_Registry.getLodDistances().size());

    m_InstanceStaging.assign(m_TotalCapacity, Render::InstanceBufferEntry{});
    m_CullDataStaging.assign(m_TotalCapacity, Render::CullInstanceData{});
    m_RegionStaging.assign(std::max<size_t>(1, regions.size()), Render::CullRegionData{});

    for (size_t r = 0; r < regions.size(); r++) {
        const auto& win = m_RegionWindows[r];
        const auto& reg = regions[r];
        m_RegionStaging[r] = { reg.cullBox };

        for (uint32_t i = 0; i < reg.instances.size(); i++) {
            const auto& inst = reg.instances[i];
            size_t assetIdx = m_Registry.assetIndexById(inst.assetId);
            if (assetIdx == SIZE_MAX) continue;

            uint32_t idx = win.offset + i;
            m_InstanceStaging[idx] = Render::InstanceBufferEntry(
                inst.model, inst.normal, static_cast<uint32_t>(assetIdx));

            const dm::box3& localBbox = assets[assetIdx].lods[0].bbox;
            dm::box3 worldBbox = localBbox * dm::homogeneousToAffine(inst.model);

            auto& cd = m_CullDataStaging[idx];
            cd.bbox     = worldBbox;
            cd.baseSlot = static_cast<uint32_t>(assetIdx) * numLods;
            cd.regionId = static_cast<uint32_t>(r);
            cd.active   = 1;
        }
    }
}

void MeshShaderRenderPass::_BuildSlotLayout() {
    const uint32_t numLods   = static_cast<uint32_t>(m_Registry.getLodDistances().size());
    const uint32_t numAssets = static_cast<uint32_t>(m_Registry.getAssets().size());

    m_NumSlots = std::max(1u, numAssets * numLods);

    // Per-asset live instance count — upper bound on visible instances per slot.
    std::vector<uint32_t> livePerAsset(std::max(1u, numAssets), 0);
    for (const auto& win : m_RegionWindows) {
        for (uint32_t i = 0; i < win.count; i++) {
            uint32_t idx = win.offset + i;
            if (m_CullDataStaging[idx].active) {
                uint32_t treeId = m_InstanceStaging[idx].treeId;
                if (treeId < numAssets) livePerAsset[treeId]++;
            }
        }
    }

    m_SlotOffsets.assign(m_NumSlots, 0);
    m_ASInvocsPerSlot.assign(m_NumSlots, 0);
    m_DispatchArgsStaging.assign(m_NumSlots, {0, 0, 1, 1});
    m_VisBufferSize = 0;

    for (uint32_t ai = 0; ai < numAssets; ai++) {
        for (uint32_t lodi = 0; lodi < numLods; lodi++) {
            uint32_t slot = ai * numLods + lodi;

            m_SlotOffsets[slot] = m_VisBufferSize;
            m_VisBufferSize    += livePerAsset[ai];

            uint32_t meshletCount = (slot < m_MeshletMegabuffers.meshOffsets.size())
                ? m_MeshletMegabuffers.meshOffsets[slot].meshletCount : 0u;
            uint32_t invocations = std::max(1u,
                (meshletCount + Render::k_ASGroupSize - 1) / Render::k_ASGroupSize);
            m_ASInvocsPerSlot[slot] = invocations;

            // Template DISPATCH_MESH record — frame-start clear + MeshCullMain
            // atomically accumulates groupsX as survivors * chunks.
            m_DispatchArgsStaging[slot].slotIdx = slot;
        }
    }
    m_VisBufferSize = std::max(1u, m_VisBufferSize);
}

void MeshShaderRenderPass::_UploadCullBuffers(nvrhi::ICommandList* cl) {
    auto device = GetDevice();
    const uint32_t numRegions = std::max(1u,
        static_cast<uint32_t>(m_Registry.getRegions().size()));

    // SRVs — permanent ShaderResource after upload.
    m_Cull.persistentInstBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_TotalCapacity * sizeof(Render::InstanceBufferEntry))
        .setStructStride(sizeof(Render::InstanceBufferEntry))
        .setDebugName("Mesh_PersistentInstBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_Cull.persistentInstBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_Cull.persistentInstBuffer, m_InstanceStaging.data(),
        m_TotalCapacity * sizeof(Render::InstanceBufferEntry));
    cl->setPermanentBufferState(m_Cull.persistentInstBuffer, nvrhi::ResourceStates::ShaderResource);

    m_Cull.cullDataBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_TotalCapacity * sizeof(Render::CullInstanceData))
        .setStructStride(sizeof(Render::CullInstanceData))
        .setDebugName("Mesh_CullDataBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_Cull.cullDataBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_Cull.cullDataBuffer, m_CullDataStaging.data(),
        m_TotalCapacity * sizeof(Render::CullInstanceData));
    cl->setPermanentBufferState(m_Cull.cullDataBuffer, nvrhi::ResourceStates::ShaderResource);

    m_Cull.cullRegionDataBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(numRegions * sizeof(Render::CullRegionData))
        .setStructStride(sizeof(Render::CullRegionData))
        .setDebugName("Mesh_CullRegionDataBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_Cull.cullRegionDataBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_Cull.cullRegionDataBuffer, m_RegionStaging.data(),
        numRegions * sizeof(Render::CullRegionData));
    cl->setPermanentBufferState(m_Cull.cullRegionDataBuffer, nvrhi::ResourceStates::ShaderResource);

    m_Cull.slotOffsetBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_SlotOffsetBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_Cull.slotOffsetBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_Cull.slotOffsetBuffer, m_SlotOffsets.data(), m_NumSlots * sizeof(uint32_t));
    cl->setPermanentBufferState(m_Cull.slotOffsetBuffer, nvrhi::ResourceStates::ShaderResource);

    m_Cull.ASInvocsPerSlotBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_ASInvocationsPerSlotBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_Cull.ASInvocsPerSlotBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_Cull.ASInvocsPerSlotBuffer, m_ASInvocsPerSlot.data(),
        m_NumSlots * sizeof(uint32_t));
    cl->setPermanentBufferState(m_Cull.ASInvocsPerSlotBuffer, nvrhi::ResourceStates::ShaderResource);

    // UAVs — auto-tracked.
    // populated by CS
    m_Cull.regionVisibleBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(numRegions * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_RegionVisibleBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_Cull.countBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_CountBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_Cull.visibilityBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_VisBufferSize * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_VisibilityBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_Cull.dispatchArgsBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumSlots * sizeof(DispatchRecord))
        .setDebugName("Mesh_DispatchArgsBuffer")
        .setIsDrawIndirectArgs(true)
        .setCanHaveUAVs(true)
        .setCanHaveRawViews(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    // Resize readback buffers to match current slot count.
    const uint64_t readbackSize = m_NumSlots * sizeof(uint32_t);
    for (uint32_t i = 0; i < k_QueuedFrames; i++) {
        m_ReadbackBuffers[i] = device->createBuffer(nvrhi::BufferDesc()
            .setByteSize(readbackSize)
            .setCpuAccess(nvrhi::CpuAccessMode::Read)
            .setInitialState(nvrhi::ResourceStates::CopyDest)
            .setKeepInitialState(true)
            .setDebugName("MeshCullCountReadback_" + std::to_string(i)));
    }
    m_ReadbackFrameIndex = 0;
}

// ===========================================================================
// Binding sets
// ===========================================================================

void MeshShaderRenderPass::_RebuildCullBindingSet() {
    // Dummy 1x1 texture + sampler to satisfy the Hi-Z binding slot (hizEnabled=0).
    // Lazily created and cached on the pass.
    static nvrhi::TextureHandle s_hizDummyTexture;
    static nvrhi::SamplerHandle s_hizDummySampler;
    if (!s_hizDummyTexture) {
        s_hizDummyTexture = GetDevice()->createTexture(nvrhi::TextureDesc()
            .setWidth(1).setHeight(1).setMipLevels(1)
            .setFormat(nvrhi::Format::RG32_FLOAT)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true)
            .setDebugName("MeshCull_DummyHiZ"));
    }
    if (!s_hizDummySampler) {
        s_hizDummySampler = GetDevice()->createSampler(nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
            .setAllFilters(false));
    }

    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(k_CB_Cull, m_Shared.constantBuffer,
            nvrhi::BufferRange(0, Render::c_CullConstantBufferSize)),

        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_CullRegion,   m_Cull.cullRegionDataBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_CullInstance, m_Cull.cullDataBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_CullSlotOffs, m_Cull.slotOffsetBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_CullChunks,   m_Cull.ASInvocsPerSlotBuffer),

        nvrhi::BindingSetItem::StructuredBuffer_UAV(k_UAV_RegionVisible, m_Cull.regionVisibleBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(k_UAV_CountBuf,      m_Cull.countBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(k_UAV_VisBuf,        m_Cull.visibilityBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(k_UAV_DispatchArgs,         m_Cull.dispatchArgsBuffer),

        nvrhi::BindingSetItem::Texture_SRV(k_SRV_CullHiZ, s_hizDummyTexture),
        nvrhi::BindingSetItem::Sampler(k_Sampler_Cull, s_hizDummySampler),
    };
    m_Cull.bindingSet = GetDevice()->createBindingSet(bsd, m_Cull.bindingLayout);
}

void MeshShaderRenderPass::_RebuildDrawBindingSet() {
    if (m_Draw.textureSets.empty()) return;

    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::PushConstants(k_PushC_Draw, k_PushCBytes),
        nvrhi::BindingSetItem::ConstantBuffer(k_CB_Draw, m_Shared.constantBuffer,
            nvrhi::BufferRange(0, Render::c_CullConstantBufferSize)),

        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_Positions,     m_Meshlet.positions),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_Normals,       m_Meshlet.normals),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_Tangents,      m_Meshlet.tangents),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_Bitangents,    m_Meshlet.bitangents),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_UVs,           m_Meshlet.uvs),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_MVertIdx,      m_Meshlet.meshletVertIdx),
        nvrhi::BindingSetItem::RawBuffer_SRV(k_SRV_MPrimIdx,             m_Meshlet.meshletPrimIdx),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_Meshlets,      m_Meshlet.meshletDescs),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_AssetLods,     m_Meshlet.assetLodRanges),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_VisBuf,        m_Cull.visibilityBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_SlotOffsets,   m_Cull.slotOffsetBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_SlotCounts,    m_Cull.countBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_Instances,     m_Cull.persistentInstBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_ASInvocsPerSlot, m_Cull.ASInvocsPerSlotBuffer),

        nvrhi::BindingSetItem::Texture_SRV(k_SRV_Diffuse,   m_Draw.textureSets[0].diffuse),
        nvrhi::BindingSetItem::Texture_SRV(k_SRV_NormalMap, m_Draw.textureSets[0].normalMap),
        nvrhi::BindingSetItem::Sampler(k_Sampler_Draw, m_Draw.sampler),
    };
    m_Draw.bindingSet = GetDevice()->createBindingSet(bsd, m_Draw.bindingLayout);
}

// ===========================================================================
// Hot-reload callbacks
// ===========================================================================

void MeshShaderRenderPass::onAssetsDirty(const std::vector<size_t>&) {
    auto cl = GetDevice()->createCommandList();
    cl->open();

    _RebuildMeshletMegabuffers(cl);
    _UploadMeshletMegabuffers(cl);
    _BuildRegionWindows();
    _BuildSlotLayout();
    _UploadCullBuffers(cl);

    cl->close();
    GetDevice()->executeCommandList(cl);

    _RebuildCullBindingSet();
    _RebuildDrawBindingSet();
}

void MeshShaderRenderPass::onRegionsDirty(const std::vector<size_t>&) {
    auto cl = GetDevice()->createCommandList();
    cl->open();

    _BuildRegionWindows();
    _BuildSlotLayout();
    _UploadCullBuffers(cl);

    cl->close();
    GetDevice()->executeCommandList(cl);

    _RebuildCullBindingSet();
    _RebuildDrawBindingSet();
}

// ===========================================================================
// Pipeline + command signature
// ===========================================================================

void MeshShaderRenderPass::_CreatePipelineIfNeeded(nvrhi::IFramebuffer* framebuffer) {
    if (m_Draw.pipeline) return;

    nvrhi::MeshletPipelineDesc psoDesc;
    psoDesc.AS = m_Draw.amplificationShader;
    psoDesc.MS = m_Draw.meshShader;
    psoDesc.PS = m_Draw.pixelShader;
    psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
    psoDesc.bindingLayouts = { m_Draw.bindingLayout };

    auto& rs = psoDesc.renderState;
    rs.depthStencilState.depthTestEnable  = true;
    rs.depthStencilState.depthWriteEnable = true;
#if XYLEM_USE_REVERSE_Z
    rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::GreaterOrEqual;
#else
    rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::LessOrEqual;
#endif
    rs.rasterState.cullMode = nvrhi::RasterCullMode::Back;

    m_Draw.pipeline = GetDevice()->createMeshletPipeline(psoDesc, framebuffer->getFramebufferInfo());
}

void MeshShaderRenderPass::_EnsureDispatchMeshSignature() {
    if (m_DispatchMeshSignature) return;
    if (!m_Draw.pipeline) return;

    ID3D12Device* d3dDevice = GetDevice()->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
    ID3D12RootSignature* rootSig = m_Draw.pipeline->getNativeObject(
        nvrhi::ObjectTypes::D3D12_RootSignature);
    if (!d3dDevice || !rootSig) return;

    D3D12_INDIRECT_ARGUMENT_DESC args[2] = {};
    args[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
    args[0].Constant.RootParameterIndex      = k_PushC_RootParamIdx;
    args[0].Constant.DestOffsetIn32BitValues = 0;
    args[0].Constant.Num32BitValuesToSet     = 1;

    args[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;

    D3D12_COMMAND_SIGNATURE_DESC desc = {};
    desc.ByteStride       = sizeof(DispatchRecord);  // 16 bytes
    desc.NumArgumentDescs = 2;
    desc.pArgumentDescs   = args;

    HRESULT hr = d3dDevice->CreateCommandSignature(
        &desc, rootSig, IID_PPV_ARGS(&m_DispatchMeshSignature));
    if (FAILED(hr)) {
        log::error("MeshShaderRenderPass: CreateCommandSignature failed (0x%08x)", hr);
        m_DispatchMeshSignature = nullptr;
    }
}

// ===========================================================================
// IRenderPass overrides
// ===========================================================================

void MeshShaderRenderPass::Animate(float /*seconds*/) {
    GetDeviceManager()->SetInformativeWindowTitle("Xylem (MeshShader)");

    if (!m_Registry.anyDirty()) return;
    auto dirtyAssets  = m_Registry.getDirtyAssetIndices();
    auto dirtyRegions = m_Registry.getDirtyRegionIndices();
    m_Registry.rebuildDirtyAssets();
    m_Registry.rebuildDirtyRegions();
    if (!dirtyAssets.empty())  onAssetsDirty(dirtyAssets);
    if (!dirtyRegions.empty()) onRegionsDirty(dirtyRegions);
    m_Registry.clearDirtyFlags();
}

void MeshShaderRenderPass::BackBufferResizing() {
    m_Draw.pipeline         = nullptr;
    m_DispatchMeshSignature = nullptr;  // rebuilt alongside the pipeline
}

void MeshShaderRenderPass::Render(nvrhi::IFramebuffer* framebuffer) {
    const auto& fbInfo = framebuffer->getFramebufferInfo();
    const uint32_t fbW = fbInfo.width;
    const uint32_t fbH = fbInfo.height;

    if (!m_Draw.pipeline) {
        m_ViewHandler.view.SetViewport({ float(fbW), float(fbH) });
        m_ViewHandler.view.SetProjectionMatrix(
        #if XYLEM_USE_REVERSE_Z
            dm::perspProjD3DStyleReverse(dm::radians(60.f), float(fbW)/float(fbH), 0.1f)
        #else
            dm::perspProjD3DStyle(dm::radians(60.f), float(fbW)/float(fbH), 0.1f, 1000.f)
        #endif
        );
    }
    m_ViewHandler.view.SetViewMatrix(m_ViewHandler.camera.GetWorldToViewMatrix());
    m_ViewHandler.view.UpdateCache();

    _CreatePipelineIfNeeded(framebuffer);
    _EnsureDispatchMeshSignature();

    m_CommandList->open();

    // ----- 1. Fill CullConstantBufferEntry (minus Hi-Z / shadow fields) -----
    Render::CullConstantBufferEntry cb = {};
    cb.viewProj    = m_ViewHandler.view.GetViewProjectionMatrix();
    cb.viewMatrix  = dm::affineToHomogeneous(m_ViewHandler.view.GetViewMatrix());
    cb.sunLightDir = m_Registry.getSunDirection();
    cb.cascadeSplits = dm::float4(0.f);
    for (uint32_t c = 0; c < Render::c_NumCascades; c++)
        cb.lightViewProj[c] = dm::float4x4::identity();

    cb.viewFrustum   = m_ViewHandler.view.GetViewFrustum();
    cb.worldToLight  = dm::float4x4::identity();
    for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
        cb.shadowCasterMinLS[c] = dm::float4( 1e30f,  1e30f,  1e30f, 0.f);
        cb.shadowCasterMaxLS[c] = dm::float4(-1e30f, -1e30f, -1e30f, 0.f);
    }

    cb.cameraPos     = m_ViewHandler.camera.GetPosition();
    cb.numRegions    = static_cast<uint32_t>(m_Registry.getRegions().size());
    cb.totalCapacity = m_TotalCapacity;

    const auto& lodDistances = m_Registry.getLodDistances();
    cb.numLods = static_cast<uint32_t>(lodDistances.size());
    for (uint32_t i = 0; i < cb.numLods; i++)
        cb.lodDistances[i] = lodDistances[i];

    cb.hizDimensions = dm::float2(static_cast<float>(fbW), static_cast<float>(fbH));
    cb.maxHiZMip     = 0.f;
    cb.hizEnabled    = 0u;  // Hi-Z deferred on the mesh-shader path

    m_CommandList->writeBuffer(m_Shared.constantBuffer, &cb, Render::c_CullConstantBufferSize);

    // ----- 2. Per-frame resets -----
    m_CommandList->clearBufferUInt(m_Cull.countBuffer,         0);
    m_CommandList->clearBufferUInt(m_Cull.regionVisibleBuffer, 0);
    m_CommandList->writeBuffer(m_Cull.dispatchArgsBuffer,
        m_DispatchArgsStaging.data(),
        m_DispatchArgsStaging.size() * sizeof(DispatchRecord));

    // ----- 3. GPU cull dispatches -----
    m_CommandList->beginMarker("MeshCull");
    {
        m_CommandList->beginMarker("RegionDispatch");
        nvrhi::ComputeState cs;
        cs.bindings = { m_Cull.bindingSet };

        cs.pipeline = m_Cull.regionPipeline;
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch((cb.numRegions + 63) / 64, 1, 1);
        m_CommandList->endMarker();

        m_CommandList->beginMarker("MainDispatch");
        cs.pipeline = m_Cull.mainPipeline;
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch((m_TotalCapacity + 255) / 256, 1, 1);
        m_CommandList->endMarker();
    }
    m_CommandList->endMarker();

    // Copy countBuffer → readback ring (for UI visible-count stat).
    {
        uint32_t ringSlot = m_ReadbackFrameIndex % k_QueuedFrames;
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], 0,
                                  m_Cull.countBuffer, 0,
                                  m_NumSlots * sizeof(uint32_t));
    }

    // ----- 4. Clear framebuffer + setMeshletState -----
    nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0,
        nvrhi::Color(0.4f, 0.6f, 0.85f, 1.f));
#if XYLEM_USE_REVERSE_Z
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer, 0.f, 0);
#else
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer, 1.f, 0);
#endif

    m_CommandList->beginMarker("MeshDraw");
    nvrhi::MeshletState meshState;
    meshState.pipeline       = m_Draw.pipeline;
    meshState.framebuffer    = framebuffer;
    meshState.bindings       = { m_Draw.bindingSet };
    meshState.indirectParams = m_Cull.dispatchArgsBuffer;  // auto-barrier → INDIRECT_ARGUMENT
    meshState.viewport.addViewportAndScissorRect(fbInfo.getViewport());
    m_CommandList->setMeshletState(meshState);

    // Placeholder push-constant value — the command signature's CONSTANT arg
    // overrides this per-record with the slotIdx stored in dispatchArgsBuffer.
    uint32_t placeholder = 0;
    m_CommandList->setPushConstants(&placeholder, sizeof(uint32_t));

    // ----- 5. Native ExecuteIndirect(DISPATCH_MESH) -----
    if (m_DispatchMeshSignature) {
        auto* d3dList = static_cast<ID3D12GraphicsCommandList6*>(
            m_CommandList->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList));
        auto* argBuffer = static_cast<ID3D12Resource*>(
            m_Cull.dispatchArgsBuffer->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource));

        if (d3dList && argBuffer) {
            d3dList->ExecuteIndirect(
                m_DispatchMeshSignature.Get(),
                m_NumSlots,
                argBuffer,
                0,
                nullptr,
                0);
        }
    }
    m_CommandList->endMarker();

    m_CommandList->close();
    GetDevice()->executeCommandList(m_CommandList);

    // ----- 6. UI stats -----
    m_UI.totalInstanceCount = static_cast<uint32_t>(m_Registry.totalInstanceCount());
    m_UI.drawCallCount      = m_NumSlots;

    if (m_ReadbackFrameIndex >= (k_QueuedFrames - 1)) {
        uint32_t readSlot = (m_ReadbackFrameIndex + 1) % k_QueuedFrames;
        void* pData = GetDevice()->mapBuffer(m_ReadbackBuffers[readSlot], nvrhi::CpuAccessMode::Read);
        if (pData) {
            const uint32_t* counts = static_cast<const uint32_t*>(pData);
            uint32_t visSum = 0;
            for (uint32_t i = 0; i < m_NumSlots; i++) visSum += counts[i];
            GetDevice()->unmapBuffer(m_ReadbackBuffers[readSlot]);

            m_UI.visibleInstanceCount = visSum;
            m_UI.culledInstanceCount  = (visSum <= m_UI.totalInstanceCount)
                ? m_UI.totalInstanceCount - visSum : 0;
        }
    }
    m_ReadbackFrameIndex++;
}
