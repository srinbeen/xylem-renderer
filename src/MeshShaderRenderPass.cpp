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
#include <cstring>
#include <filesystem>

using namespace donut::math;
using namespace Xylem;

namespace {

// Register assignments — kept in lockstep with MeshShaderPass.hlsl.
constexpr uint32_t k_CB            = 0;
constexpr uint32_t k_SRVPositions  = 0;
constexpr uint32_t k_SRVNormals    = 1;
constexpr uint32_t k_SRVTangents   = 2;
constexpr uint32_t k_SRVBitangents = 3;
constexpr uint32_t k_SRVUVs        = 4;
constexpr uint32_t k_SRVMVertIdx   = 5;
constexpr uint32_t k_SRVMPrimIdx   = 6;
constexpr uint32_t k_SRVMeshlets   = 7;
constexpr uint32_t k_SRVAssetLods  = 8;
constexpr uint32_t k_SRVVisible    = 9;
constexpr uint32_t k_SRVInstances  = 10;
constexpr uint32_t k_SRVDiffuse    = 11;
constexpr uint32_t k_SRVNormalMap  = 12;
constexpr uint32_t k_Sampler       = 0;

// D3D12 DispatchMesh caps each axis at 65535. Must match XYLEM_DISPATCH_X in
// meshlet_types.hlsli — AS reconstructs workIdx = gid.y * k_DispatchXCap + gid.x.
constexpr uint32_t k_DispatchXCap = 65535;

} // namespace

// ===========================================================================
// Init
// ===========================================================================

bool MeshShaderRenderPass::Init() {
    if (!_InitShared()) return false;
    if (!_InitDrawResources()) return false;

    engine::CommonRenderPasses commonPasses(GetDevice(), m_ShaderFactory);

    auto initCL = GetDevice()->createCommandList();
    initCL->open();
    if (!_LoadBarkTextures(initCL, commonPasses)) { initCL->close(); return false; }

    _RebuildMeshletMegaBuffers(initCL);
    _RebuildBindingSet();

    initCL->close();
    GetDevice()->executeCommandList(initCL);

    m_CommandList = GetDevice()->createCommandList();
    return true;
}

bool MeshShaderRenderPass::_InitShared() {
    m_Shared.constantBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(Render::c_ConstantBufferSize)
            .setIsConstantBuffer(true)
            .setIsVolatile(true)
            .setMaxVersions(k_QueuedFrames * 4)
            .setDebugName("MeshShaderPass_CB")
    );
    return m_Shared.constantBuffer != nullptr;
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
        nvrhi::BindingLayoutItem::ConstantBuffer(k_CB),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRVPositions),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRVNormals),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRVTangents),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRVBitangents),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRVUVs),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRVMVertIdx),
        nvrhi::BindingLayoutItem::RawBuffer_SRV(k_SRVMPrimIdx),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRVMeshlets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRVAssetLods),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRVVisible),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRVInstances),
        nvrhi::BindingLayoutItem::Texture_SRV(k_SRVDiffuse),
        nvrhi::BindingLayoutItem::Texture_SRV(k_SRVNormalMap),
        nvrhi::BindingLayoutItem::Sampler(k_Sampler),
    };
    m_Draw.bindingLayout = GetDevice()->createBindingLayout(bld);
    return m_Draw.bindingLayout != nullptr;
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
// Meshlet mega-buffer build — runs on asset dirty
// ===========================================================================

void MeshShaderRenderPass::_RebuildMeshletMegaBuffers(nvrhi::ICommandList* cl) {
    const auto& assets = m_Registry.getAssets();
    const uint32_t numLods = assets.empty() ? 0 : static_cast<uint32_t>(assets[0].lods.size());

    auto& build = m_MeshletCPU;
    build.positions.clear();
    build.normals.clear();
    build.tangents.clear();
    build.bitangents.clear();
    build.uvs.clear();
    build.meshletVertIdx.clear();
    build.meshletPrimIdx.clear();
    build.meshlets.clear();
    build.assetLodRanges.assign(assets.size() * static_cast<size_t>(numLods), {});

    for (size_t ai = 0; ai < assets.size(); ai++) {
        const auto& asset = assets[ai];
        for (uint32_t li = 0; li < asset.lods.size(); li++) {
            const auto& lod = asset.lods[li];
            Render::AssetLodRange range;
            range.vertexAttribBase = static_cast<uint32_t>(build.positions.size());
            range.meshletVertBase  = static_cast<uint32_t>(build.meshletVertIdx.size());
            range.meshletPrimBase  = static_cast<uint32_t>(build.meshletPrimIdx.size());
            range.meshletOffset    = static_cast<uint32_t>(build.meshlets.size());

            // Concatenate vertex attributes
            build.positions.insert(build.positions.end(),  lod.positions.begin(),  lod.positions.end());
            build.normals.insert(build.normals.end(),      lod.normals.begin(),    lod.normals.end());
            build.tangents.insert(build.tangents.end(),    lod.tangents.begin(),   lod.tangents.end());
            build.bitangents.insert(build.bitangents.end(),lod.bitangents.begin(), lod.bitangents.end());
            build.uvs.insert(build.uvs.end(),              lod.uvs.begin(),        lod.uvs.end());

            // Build meshlets
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

            // Per-meshlet: compute bounds, append descriptor + vert/prim data
            for (size_t mi = 0; mi < meshletCount; mi++) {
                const meshopt_Meshlet& m = ml[mi];
                meshopt_Bounds b = meshopt_computeMeshletBounds(
                    mlVerts.data() + m.vertex_offset,
                    mlTris.data()  + m.triangle_offset,
                    m.triangle_count,
                    posPtr, lod.positions.size(), sizeof(dm::float3));

                Render::MeshletDesc desc;
                desc.vertexOffset   = static_cast<uint32_t>(build.meshletVertIdx.size() - range.meshletVertBase);
                desc.triangleOffset = static_cast<uint32_t>(build.meshletPrimIdx.size() - range.meshletPrimBase);
                desc.vertexCount    = m.vertex_count;
                desc.triangleCount  = m.triangle_count;
                desc.bounds         = dm::float4(b.center[0], b.center[1], b.center[2], b.radius);
                desc.coneApex       = dm::float4(b.cone_apex[0], b.cone_apex[1], b.cone_apex[2], 0.f);
                desc.coneAxisCutoff = dm::float4(b.cone_axis[0], b.cone_axis[1], b.cone_axis[2], b.cone_cutoff);

                build.meshlets.push_back(desc);

                // Append meshlet-local vertex indices
                build.meshletVertIdx.insert(build.meshletVertIdx.end(),
                    mlVerts.data() + m.vertex_offset,
                    mlVerts.data() + m.vertex_offset + m.vertex_count);

                // Append packed triangle indices (3 bytes per tri)
                build.meshletPrimIdx.insert(build.meshletPrimIdx.end(),
                    mlTris.data() + m.triangle_offset,
                    mlTris.data() + m.triangle_offset + m.triangle_count * 3);
            }

            // Pad meshletPrimIdx to 4-byte alignment (ByteAddressBuffer load32 needs it)
            size_t misalign = (build.meshletPrimIdx.size() - range.meshletPrimBase) % 4;
            if (misalign) {
                build.meshletPrimIdx.insert(build.meshletPrimIdx.end(), 4 - misalign, 0);
            }

            range.meshletCount = static_cast<uint32_t>(meshletCount);
            build.assetLodRanges[ai * numLods + li] = range;
        }
    }

    // Upload to GPU
    auto device = GetDevice();
    const uint32_t totalVerts     = static_cast<uint32_t>(build.positions.size());
    const uint32_t totalMeshlets  = static_cast<uint32_t>(build.meshlets.size());
    const uint32_t totalVertIdx   = static_cast<uint32_t>(build.meshletVertIdx.size());
    const uint32_t totalPrimBytes = static_cast<uint32_t>(build.meshletPrimIdx.size());
    const uint32_t numAssetLods   = static_cast<uint32_t>(build.assetLodRanges.size());

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
        h = device->createBuffer(d);
        cl->beginTrackingBufferState(h, nvrhi::ResourceStates::CopyDest);
        if (data && bytes > 0) cl->writeBuffer(h, data, bytes);
        cl->setPermanentBufferState(h, nvrhi::ResourceStates::ShaderResource);
    };

    makeSrv(m_Meshlet.positions,      build.positions.data(),      totalVerts * sizeof(dm::float3),
            sizeof(dm::float3), "Mesh_Positions",  false);
    makeSrv(m_Meshlet.normals,        build.normals.data(),        totalVerts * sizeof(dm::float3),
            sizeof(dm::float3), "Mesh_Normals",    false);
    makeSrv(m_Meshlet.tangents,       build.tangents.data(),       totalVerts * sizeof(dm::float3),
            sizeof(dm::float3), "Mesh_Tangents",   false);
    makeSrv(m_Meshlet.bitangents,     build.bitangents.data(),     totalVerts * sizeof(dm::float3),
            sizeof(dm::float3), "Mesh_Bitangents", false);
    makeSrv(m_Meshlet.uvs,            build.uvs.data(),            totalVerts * sizeof(dm::float2),
            sizeof(dm::float2), "Mesh_UVs",        false);
    makeSrv(m_Meshlet.meshletVertIdx, build.meshletVertIdx.data(), totalVertIdx * sizeof(uint32_t),
            sizeof(uint32_t),   "Mesh_MeshletVertIdx", false);
    makeSrv(m_Meshlet.meshletPrimIdx, build.meshletPrimIdx.data(), totalPrimBytes,
            0,                  "Mesh_MeshletPrimIdx", true);
    makeSrv(m_Meshlet.meshletDescs,   build.meshlets.data(),       totalMeshlets * sizeof(Render::MeshletDesc),
            sizeof(Render::MeshletDesc),   "Mesh_Meshlets", false);
    makeSrv(m_Meshlet.assetLodRanges, build.assetLodRanges.data(), numAssetLods * sizeof(Render::AssetLodRange),
            sizeof(Render::AssetLodRange), "Mesh_AssetLodRanges", false);

    m_Meshlet.totalVertices     = totalVerts;
    m_Meshlet.totalMeshlets     = totalMeshlets;
    m_Meshlet.totalVertIdx      = totalVertIdx;
    m_Meshlet.totalPrimIdxBytes = totalPrimBytes;
    m_Meshlet.numAssetLods      = numAssetLods;

    // Visible-work-item and instance-data buffers — UAV-less SRV updated per frame.
    // Each instance produces ceil(meshletCount / k_ASGroupSize) work items, so
    // size the visible buffer to the absolute worst case (all instances at their
    // highest-meshlet LOD). That caps in-flight work items with zero runtime overflow.
    uint32_t maxChunksPerLod = 1;
    for (const auto& r : build.assetLodRanges) {
        uint32_t chunks = (r.meshletCount + Render::k_ASGroupSize - 1) / Render::k_ASGroupSize;
        maxChunksPerLod = std::max(maxChunksPerLod, chunks);
    }
    const uint32_t instCap = std::max<uint32_t>(1, m_Registry.totalInstanceCount());
    const uint32_t visCap  = std::max<uint32_t>(1, instCap * maxChunksPerLod);

    m_Meshlet.visibleInstances = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(visCap * sizeof(Render::VisibleInstance))
            .setStructStride(sizeof(Render::VisibleInstance))
            .setDebugName("Mesh_VisibleInstances")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource));
    m_Meshlet.visCapacity = visCap;

    m_Meshlet.instanceBuffer = device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(instCap * sizeof(Render::InstanceBufferEntry))
            .setStructStride(sizeof(Render::InstanceBufferEntry))
            .setDebugName("Mesh_Instances")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource));
    m_Meshlet.instanceCapacity = instCap;

    // UI reporting
    size_t totalBytes =
        totalVerts * (sizeof(dm::float3) * 4 + sizeof(dm::float2)) +
        totalVertIdx * sizeof(uint32_t) +
        totalPrimBytes +
        totalMeshlets * sizeof(Render::MeshletDesc) +
        numAssetLods * sizeof(Render::AssetLodRange);
    m_UI.meshletMegaBufferMB = static_cast<float>(totalBytes) / (1024.f * 1024.f);
    m_UI.totalMeshletCount   = totalMeshlets;
}

// ===========================================================================
// Hot-reload callbacks — for MVP, any dirty asset triggers a full rebuild.
// ===========================================================================

void MeshShaderRenderPass::onAssetsDirty(const std::vector<size_t>&) {
    auto cl = GetDevice()->createCommandList();
    cl->open();
    _RebuildMeshletMegaBuffers(cl);
    _RebuildBindingSet();
    cl->close();
    GetDevice()->executeCommandList(cl);
}

void MeshShaderRenderPass::onRegionsDirty(const std::vector<size_t>&) {
    // Instance capacity might grow — resize if needed.
    uint32_t instCap = std::max<uint32_t>(1, static_cast<uint32_t>(m_Registry.totalInstanceCount()));
    if (instCap > m_Meshlet.instanceCapacity) {
        m_Meshlet.instanceBuffer = GetDevice()->createBuffer(
            nvrhi::BufferDesc()
                .setByteSize(instCap * sizeof(Render::InstanceBufferEntry))
                .setStructStride(sizeof(Render::InstanceBufferEntry))
                .setDebugName("Mesh_Instances")
                .enableAutomaticStateTracking(nvrhi::ResourceStates::ShaderResource));
        m_Meshlet.instanceCapacity = instCap;
        _RebuildBindingSet();
    }
}

// ===========================================================================
// Binding set — rebuilt whenever any underlying buffer is recreated.
// ===========================================================================

void MeshShaderRenderPass::_RebuildBindingSet() {
    if (m_Draw.textureSets.empty()) return;

    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(k_CB, m_Shared.constantBuffer,
            nvrhi::BufferRange(0, Render::c_ConstantBufferSize)),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRVPositions,  m_Meshlet.positions),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRVNormals,    m_Meshlet.normals),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRVTangents,   m_Meshlet.tangents),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRVBitangents, m_Meshlet.bitangents),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRVUVs,        m_Meshlet.uvs),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRVMVertIdx,   m_Meshlet.meshletVertIdx),
        nvrhi::BindingSetItem::RawBuffer_SRV(k_SRVMPrimIdx,          m_Meshlet.meshletPrimIdx),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRVMeshlets,   m_Meshlet.meshletDescs),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRVAssetLods,  m_Meshlet.assetLodRanges),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRVVisible,    m_Meshlet.visibleInstances),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRVInstances,  m_Meshlet.instanceBuffer),
        nvrhi::BindingSetItem::Texture_SRV(k_SRVDiffuse,   m_Draw.textureSets[0].diffuse),
        nvrhi::BindingSetItem::Texture_SRV(k_SRVNormalMap, m_Draw.textureSets[0].normalMap),
        nvrhi::BindingSetItem::Sampler(k_Sampler, m_Draw.sampler),
    };
    m_Draw.bindingSet = GetDevice()->createBindingSet(bsd, m_Draw.bindingLayout);
}

// ===========================================================================
// Pipeline creation (deferred until framebuffer info is known)
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
    rs.rasterState.cullMode     = nvrhi::RasterCullMode::Back;
    rs.rasterState.frontCounterClockwise = true;

    m_Draw.pipeline = GetDevice()->createMeshletPipeline(psoDesc, framebuffer->getFramebufferInfo());
}

// ===========================================================================
// IRenderPass overrides
// ===========================================================================

void MeshShaderRenderPass::Animate(float seconds) {
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
    m_Draw.pipeline = nullptr;
}

void MeshShaderRenderPass::Render(nvrhi::IFramebuffer* framebuffer) {
    auto* device = GetDevice();
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

    // Build per-frame visible list.
    const auto& regions      = m_Registry.getRegions();
    const auto& assets       = m_Registry.getAssets();
    const auto& lodDistances = m_Registry.getLodDistances();
    const uint32_t numLods   = lodDistances.size();

    dm::frustum viewFrustum = m_ViewHandler.view.GetViewFrustum();
    dm::float3 camPos = m_ViewHandler.camera.GetPosition();

    m_InstanceStaging.clear();
    m_VisibleStaging.clear();

    uint32_t totalInst = 0, visInst = 0;
    bool     capacityReached = false;
    for (const auto& reg : regions) {
        if (capacityReached) break;
        for (const auto& inst : reg.instances) {
            totalInst++;
            size_t assetVecIdx = m_Registry.assetIndexById(inst.assetId);
            if (assetVecIdx == SIZE_MAX) continue;

            if (!m_ViewHandler.view.IsBoxVisible(inst.bbox)) continue;
            
            float dist = dm::distance(m_ViewHandler.camera.GetPosition(), inst.bbox);
            uint32_t lod = m_ViewHandler.distToLOD(dist, lodDistances);

            uint32_t assetLodIdx = static_cast<uint32_t>(assetVecIdx * numLods + lod);
            if (assetLodIdx >= m_Meshlet.numAssetLods) continue;

            const auto& range = m_MeshletCPU.assetLodRanges[assetLodIdx];
            if (range.meshletCount == 0) continue;

            // One AS group handles up to k_ASGroupSize meshlets, so emit
            // ceil(meshletCount / k_ASGroupSize) work items per instance. All
            // chunks for one instance share instanceIdx and differ in chunkBase.
            const uint32_t chunks =
                (range.meshletCount + Render::k_ASGroupSize - 1) / Render::k_ASGroupSize;

            if (visInst + 1 > m_Meshlet.instanceCapacity ||
                static_cast<uint32_t>(m_VisibleStaging.size()) + chunks > m_Meshlet.visCapacity) {
                capacityReached = true;
                break;
            }

            const uint32_t slot = visInst;
            for (uint32_t c = 0; c < chunks; c++) {
                Render::VisibleInstance ve;
                ve.instanceIdx      = slot;
                ve.assetLod         = assetLodIdx;
                ve.meshletChunkBase = c * Render::k_ASGroupSize;
                ve._pad1            = 0;
                m_VisibleStaging.push_back(ve);
            }

            m_InstanceStaging.emplace_back(inst.model, inst.normal,
                static_cast<uint32_t>(assetVecIdx));
            visInst++;
        }
    }

    const uint32_t workItems = static_cast<uint32_t>(m_VisibleStaging.size());

    m_UI.totalInstanceCount   = totalInst;
    m_UI.visibleInstanceCount = visInst;
    m_UI.culledInstanceCount  = totalInst > visInst ? (totalInst - visInst) : 0;
    m_UI.drawCallCount        = 1; // single dispatchMesh
    m_UI.asMeshletsDispatched = workItems * Render::k_ASGroupSize; // AS threads launched
    m_UI.asMeshletsCulled     = 0;
    m_UI.msInvocations        = 0;

    // Build CB
    Render::ConstantBufferEntry cb = {};
    cb.viewProj    = m_ViewHandler.view.GetViewProjectionMatrix();
    cb.viewMatrix  = dm::affineToHomogeneous(m_ViewHandler.camera.GetWorldToViewMatrix());
    cb.sunLightDir = m_Registry.getSunDirection();
    cb.cascadeSplits = dm::float4(0.f);
    for (uint32_t c = 0; c < Render::c_NumCascades; c++) cb.lightViewProj[c] = dm::float4x4::identity();

    // The shared CB struct carries a float `_pad0` slot adjacent to sunLightDir.
    // MeshShaderPass co-opts its 4 bytes as a uint work-item count — the AS
    // reads it via asuint(_pad0) to early-out tail threadgroups from 2D dispatch.
    std::memcpy(&cb._pad0, &workItems, sizeof(workItems));

    m_CommandList->open();

    // Upload
    m_CommandList->writeBuffer(m_Shared.constantBuffer, &cb, sizeof(cb));
    if (!m_InstanceStaging.empty())
        m_CommandList->writeBuffer(m_Meshlet.instanceBuffer,
            m_InstanceStaging.data(),
            m_InstanceStaging.size() * sizeof(Render::InstanceBufferEntry));
    if (!m_VisibleStaging.empty())
        m_CommandList->writeBuffer(m_Meshlet.visibleInstances,
            m_VisibleStaging.data(),
            m_VisibleStaging.size() * sizeof(Render::VisibleInstance));

    // Clear color + depth
    nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.4f, 0.6f, 0.85f, 1.f));
#if XYLEM_USE_REVERSE_Z
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer, 0.f, 0);
#else
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer, 1.f, 0);
#endif

    if (workItems > 0) {
        nvrhi::MeshletState state;
        state.pipeline    = m_Draw.pipeline;
        state.framebuffer = framebuffer;
        state.bindings    = { m_Draw.bindingSet };
        state.viewport.addViewportAndScissorRect(fbInfo.getViewport());

        m_CommandList->setMeshletState(state);

        // 2D grid so we never exceed D3D12's 65535 per-axis dispatch cap.
        // Must match XYLEM_DISPATCH_X in meshlet_types.hlsli.
        const uint32_t gX = std::min(workItems, k_DispatchXCap);
        const uint32_t gY = (workItems + k_DispatchXCap - 1) / k_DispatchXCap;
        m_CommandList->dispatchMesh(gX, gY, 1);
    }

    m_CommandList->close();
    device->executeCommandList(m_CommandList);
}
