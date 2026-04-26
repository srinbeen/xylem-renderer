#include "include/MeshShaderRenderPass.hpp"
#include "include/Render.hpp"
#include "include/Globals.hpp"
#include "include/Terrain.hpp"
#include "include/macros.h"
#include "include/shaders/ShaderContracts.hpp"
#include "include/frame/FrameLifecycle.hpp"

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
#include <limits>

using namespace donut::math;
#include <donut/shaders/sky_cb.h>

using namespace Xylem;
namespace shader_cb = Xylem::shader::cb;
namespace mesh_reg = Xylem::shader::reg::Mesh;

// ===========================================================================
// Init
// ===========================================================================

bool MeshShaderRenderPass::Init() {
    engine::CommonRenderPasses commonPasses(GetDevice(), m_ShaderFactory);

    if (!_InitShared())         return false;
    if (!_InitDrawResources())  return false;
    if (!_InitCullResources())  return false;
    if (!_InitShadowPass())     return false;
    if (!_InitDepthPrepass())   return false;
    if (!_InitHiZShaders())     return false;
    if (!_InitSDSMPass())       return false;
    if (!_InitSkyPass())        return false;

    {
        auto initCL = GetDevice()->createCommandList();
        initCL->open();

        _RebuildMeshletMegabuffers(initCL);
        _UploadMeshletMegabuffers(initCL);

        if (!_LoadBarkTextures(initCL, commonPasses)) { initCL->close(); return false; }
        if (!_InitTerrainPass(initCL))               { initCL->close(); return false; }

        _BuildRegionWindows();
        _BuildSlotLayout();
        _UploadCullBuffers(initCL);

        _RebuildCullBindingSet();
        _RebuildDrawBindingSet();
        _RebuildShadowBindingSet();
        _RebuildDepthPrepassBindingSet();

        initCL->close();
        GetDevice()->executeCommandList(initCL);
    }

    m_CommandList = GetDevice()->createCommandList();
    return true;
}

bool MeshShaderRenderPass::_InitShared() {
    m_StageResources.frameShared.constantBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(shader_cb::kCullFrameSize)
            .setIsConstantBuffer(true)
            .setDebugName("MeshShaderPass_CB")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
    );
    if (!m_StageResources.frameShared.constantBuffer) return false;

    m_StageResources.frameShared.asCullCB = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(shader_cb::kMeshASCullSize)
            .setIsConstantBuffer(true)
            .setDebugName("MeshShaderPass_ASCullCB")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
    );
    if (!m_StageResources.frameShared.asCullCB) return false;

    for (uint32_t i = 0; i < k_QueuedFrames; i++) {
        m_ReadbackBuffers[i] = GetDevice()->createBuffer(
            nvrhi::BufferDesc()
                .setByteSize(sizeof(uint32_t))  // sized properly in _UploadCullBuffers
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

    m_StageResources.sceneDraw.amplificationShader = m_ShaderFactory->CreateShader("app/MeshShaderPass.hlsl",
        "main_as", nullptr, nvrhi::ShaderType::Amplification);
    m_StageResources.sceneDraw.meshShader = m_ShaderFactory->CreateShader("app/MeshShaderPass.hlsl",
        "main_ms", nullptr, nvrhi::ShaderType::Mesh);
    m_StageResources.sceneDraw.pixelShader = m_ShaderFactory->CreateShader("app/MeshShaderPass.hlsl",
        "main_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_StageResources.sceneDraw.amplificationShader || !m_StageResources.sceneDraw.meshShader || !m_StageResources.sceneDraw.pixelShader) {
        log::error("MeshShaderRenderPass: shader compile failed");
        return false;
    }

    m_StageResources.sceneDraw.sampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Wrap)
            .setAllFilters(true)
            .setMaxAnisotropy(8.f));
    m_StageResources.sceneDraw.shadowSampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setMinFilter(true)
            .setMagFilter(true)
            .setMipFilter(false)
            .setReductionType(nvrhi::SamplerReductionType::Comparison)
            .setAllAddressModes(nvrhi::SamplerAddressMode::Border)
            .setBorderColor(nvrhi::Color(1.f))
        );
    m_StageResources.sceneDraw.hizSampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
            .setAllFilters(false));
    if (!m_StageResources.sceneDraw.sampler || !m_StageResources.sceneDraw.shadowSampler || !m_StageResources.sceneDraw.hizSampler) return false;

    nvrhi::BindingLayoutDesc bld;
    bld.visibility = nvrhi::ShaderType::All;
    bld.bindings = {
        nvrhi::BindingLayoutItem::PushConstants(mesh_reg::Draw::kPushC_Slot, mesh_reg::Draw::kPushCBytes),
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::Draw::kCB_Frame),
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::Draw::kCB_ASCull),

        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Positions),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Normals),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Tangents),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Bitangents),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_UVs),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_MeshletVertIdx),
        nvrhi::BindingLayoutItem::RawBuffer_SRV(mesh_reg::Draw::kSRV_MeshletPrimIdx),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Meshlets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_AssetLods),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Vis),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_SlotOffsets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_SlotCounts),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Instances),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_ASInvocations),

        nvrhi::BindingLayoutItem::Texture_SRV(mesh_reg::Draw::kTex_Diffuse),
        nvrhi::BindingLayoutItem::Texture_SRV(mesh_reg::Draw::kTex_NormalMap),
        nvrhi::BindingLayoutItem::Texture_SRV(mesh_reg::Draw::kTex_ShadowMap),
        nvrhi::BindingLayoutItem::Texture_SRV(mesh_reg::Draw::kTex_HiZ),

        nvrhi::BindingLayoutItem::Sampler(mesh_reg::Draw::kSampler_Main),
        nvrhi::BindingLayoutItem::Sampler(mesh_reg::Draw::kSampler_Shadow),
        nvrhi::BindingLayoutItem::Sampler(mesh_reg::Draw::kSampler_HiZ),
    };
    m_StageResources.sceneDraw.bindingLayout = GetDevice()->createBindingLayout(bld);
    return m_StageResources.sceneDraw.bindingLayout != nullptr;
}

bool MeshShaderRenderPass::_InitCullResources() {
    if (!m_ShaderFactory) return false;

    m_StageResources.cull.mainCS   = m_ShaderFactory->CreateShader("app/MeshCullCS.hlsl",
        "MeshCullMain",   nullptr, nvrhi::ShaderType::Compute);
    m_StageResources.cull.regionCS = m_ShaderFactory->CreateShader("app/MeshCullCS.hlsl",
        "MeshCullRegion", nullptr, nvrhi::ShaderType::Compute);
    m_StageResources.cull.shadowCS = m_ShaderFactory->CreateShader("app/MeshCullCS.hlsl",
        "MeshCullShadow", nullptr, nvrhi::ShaderType::Compute);
    if (!m_StageResources.cull.mainCS || !m_StageResources.cull.regionCS || !m_StageResources.cull.shadowCS) {
        log::error("MeshShaderRenderPass: MeshCullCS compile failed");
        return false;
    }

    nvrhi::BindingLayoutDesc bld;
    bld.visibility = nvrhi::ShaderType::All;
    bld.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::Cull::kCB_Frame),

        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_RegionData),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_InstanceData),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_MainSlotOffsets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_MainInvocations),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_ShadowSlotOffsets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_ShadowInvocations),

        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_MainRegionVis),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_MainCount),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_MainVis),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_MainDispatch),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_ShadowCount),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_ShadowVis),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_ShadowDispatch),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_ShadowUnique),

        nvrhi::BindingLayoutItem::Texture_SRV(mesh_reg::Cull::kSRV_HiZ),
        nvrhi::BindingLayoutItem::Sampler(mesh_reg::Cull::kSampler_HiZ),
    };
    m_StageResources.cull.bindingLayout = GetDevice()->createBindingLayout(bld);
    if (!m_StageResources.cull.bindingLayout) return false;

    nvrhi::ComputePipelineDesc psoDescRegion;
    psoDescRegion.CS = m_StageResources.cull.regionCS;
    psoDescRegion.bindingLayouts = { m_StageResources.cull.bindingLayout };
    m_StageResources.cull.regionPipeline = GetDevice()->createComputePipeline(psoDescRegion);

    nvrhi::ComputePipelineDesc psoDescMain;
    psoDescMain.CS = m_StageResources.cull.mainCS;
    psoDescMain.bindingLayouts = { m_StageResources.cull.bindingLayout };
    m_StageResources.cull.mainPipeline = GetDevice()->createComputePipeline(psoDescMain);

    nvrhi::ComputePipelineDesc psoDescShadow;
    psoDescShadow.CS = m_StageResources.cull.shadowCS;
    psoDescShadow.bindingLayouts = { m_StageResources.cull.bindingLayout };
    m_StageResources.cull.shadowPipeline = GetDevice()->createComputePipeline(psoDescShadow);

    return m_StageResources.cull.mainPipeline && m_StageResources.cull.regionPipeline && m_StageResources.cull.shadowPipeline;
}

bool MeshShaderRenderPass::_InitShadowPass() {
    if (!m_ShaderFactory) return false;

    m_StageResources.shadow.amplificationShader = m_ShaderFactory->CreateShader("app/MeshShaderPass.hlsl",
        "shadow_as", nullptr, nvrhi::ShaderType::Amplification);
    m_StageResources.shadow.meshShader = m_ShaderFactory->CreateShader("app/MeshShaderPass.hlsl",
        "shadow_ms", nullptr, nvrhi::ShaderType::Mesh);
    if (!m_StageResources.shadow.amplificationShader || !m_StageResources.shadow.meshShader) {
        log::error("MeshShaderRenderPass: shadow AS/MS compile failed");
        return false;
    }

    // Shadow binding layout mirrors the subset the shadow AS/MS actually touch.
    // Uses the same t0..t13 register layout; the bindings point at shadow cull
    // buffers instead of main ones at bind-set build time.
    nvrhi::BindingLayoutDesc bld;
    bld.visibility = nvrhi::ShaderType::All;
    bld.bindings = {
        nvrhi::BindingLayoutItem::PushConstants(mesh_reg::Draw::kPushC_Slot, mesh_reg::Draw::kPushCBytes),
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::Draw::kCB_Frame),

        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Positions),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Normals),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Tangents),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Bitangents),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_UVs),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_MeshletVertIdx),
        nvrhi::BindingLayoutItem::RawBuffer_SRV(mesh_reg::Draw::kSRV_MeshletPrimIdx),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Meshlets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_AssetLods),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Vis),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_SlotOffsets),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_SlotCounts),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Instances),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_ASInvocations),
    };
    m_StageResources.shadow.bindingLayout = GetDevice()->createBindingLayout(bld);
    if (!m_StageResources.shadow.bindingLayout) return false;

    // Texture2DArray shadow map - one slice per cascade.
    m_StageResources.shadow.depthTexture = GetDevice()->createTexture(nvrhi::TextureDesc()
        .setDimension(nvrhi::TextureDimension::Texture2DArray)
        .setWidth(k_ShadowRes).setHeight(k_ShadowRes)
        .setArraySize(Render::c_NumCascades)
        .setFormat(nvrhi::Format::D32)
        .setIsRenderTarget(true)
        .setUseClearValue(true)
        .setClearValue(nvrhi::Color(1.f, 1.f, 1.f, 1.f))
        .setInitialState(nvrhi::ResourceStates::DepthWrite)
        .setKeepInitialState(true)
        .setDebugName("MeshShader_ShadowDepth"));
    if (!m_StageResources.shadow.depthTexture) return false;

    for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
        nvrhi::FramebufferDesc fbd;
        fbd.setDepthAttachment(nvrhi::FramebufferAttachment()
            .setTexture(m_StageResources.shadow.depthTexture)
            .setArraySlice(c));
        m_StageResources.shadow.framebuffers[c] = GetDevice()->createFramebuffer(fbd);
        if (!m_StageResources.shadow.framebuffers[c]) return false;

        m_StageResources.shadow.debugTextures[c] = GetDevice()->createTexture(nvrhi::TextureDesc()
            .setDimension(nvrhi::TextureDimension::Texture2D)
            .setWidth(k_ShadowRes).setHeight(k_ShadowRes)
            .setFormat(nvrhi::Format::R32_FLOAT)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true)
            .setDebugName("MeshShader_ShadowDebug_" + std::to_string(c)));
    }

    return true;
}

bool MeshShaderRenderPass::_LoadBarkTextures(nvrhi::ICommandList* initCL,
                                             engine::CommonRenderPasses& commonPasses) {
    const auto& barkTextureSets = m_Registry.getBarkTextureSets();
    if (barkTextureSets.empty()) {
        log::error("MeshShaderRenderPass: no bark texture sets registered");
        return false;
    }

    engine::TextureCache textureCache(GetDevice(), std::make_shared<vfs::NativeFileSystem>(), nullptr);
    m_StageResources.sceneDraw.textureSets.resize(barkTextureSets.size());

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
        m_StageResources.sceneDraw.textureSets[i].diffuse   = diffLoaded ? diffLoaded->texture : nullptr;
        m_StageResources.sceneDraw.textureSets[i].normalMap = normLoaded ? normLoaded->texture : nullptr;
        if (!m_StageResources.sceneDraw.textureSets[i].diffuse || !m_StageResources.sceneDraw.textureSets[i].normalMap) {
            log::error("MeshShaderRenderPass: bark texture load failed");
            return false;
        }
    }
    return true;
}

// ===========================================================================
// Meshlet mega-buffer build - CPU side
// ===========================================================================

void MeshShaderRenderPass::_RebuildMeshletMegabuffers(nvrhi::ICommandList* /*cl*/) {
    const auto& assets      = m_Registry.getAssets();
    const uint32_t numLods  = static_cast<uint32_t>(m_Registry.getLodSegments().size());

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

            size_t misalign = (meshletTris.size() - perMeshOffsets.meshletTriOffset) % 4;
            if (misalign) meshletTris.insert(meshletTris.end(), 4 - misalign, 0);

            perMeshOffsets.meshletCount = static_cast<uint32_t>(meshletCount);
        }
    }

    // The shader may load one extra uint when unpacking a triangle that straddles
    // a 4-byte boundary; keep a small sentinel at the end of the packed stream.
    if (!meshletTris.empty())
        meshletTris.insert(meshletTris.end(), 4, 0);
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

    makeSrv(m_StageResources.sceneMeshletData.positions,      vertexAttributes.positions.data(),  totalVerts * sizeof(dm::float3),
            sizeof(dm::float3), "Mesh_Positions",  false);
    makeSrv(m_StageResources.sceneMeshletData.normals,        vertexAttributes.normals.data(),    totalVerts * sizeof(dm::float3),
            sizeof(dm::float3), "Mesh_Normals",    false);
    makeSrv(m_StageResources.sceneMeshletData.tangents,       vertexAttributes.tangents.data(),   totalVerts * sizeof(dm::float3),
            sizeof(dm::float3), "Mesh_Tangents",   false);
    makeSrv(m_StageResources.sceneMeshletData.bitangents,     vertexAttributes.bitangents.data(), totalVerts * sizeof(dm::float3),
            sizeof(dm::float3), "Mesh_Bitangents", false);
    makeSrv(m_StageResources.sceneMeshletData.uvs,            vertexAttributes.uvs.data(),        totalVerts * sizeof(dm::float2),
            sizeof(dm::float2), "Mesh_UVs",        false);
    makeSrv(m_StageResources.sceneMeshletData.meshletVertIdx, meshletVerts.data(), totalVertIdx * sizeof(uint32_t),
            sizeof(uint32_t),   "Mesh_MeshletVertIdx", false);
    makeSrv(m_StageResources.sceneMeshletData.meshletPrimIdx, meshletTris.data(), totalPrimBytes,
            0,                  "Mesh_MeshletTriIdx", true);
    makeSrv(m_StageResources.sceneMeshletData.meshletDescs,   meshletDescs.data(),       totalMeshlets * sizeof(Render::MeshletDesc),
            sizeof(Render::MeshletDesc),   "Mesh_Meshlets", false);
    makeSrv(m_StageResources.sceneMeshletData.assetLodRanges, m_MeshletMegabuffers.meshOffsets.data(),
            numAssetLods * sizeof(Render::MeshOffsets),
            sizeof(Render::MeshOffsets), "Mesh_Offsets", false);

    m_StageResources.sceneMeshletData.totalVertices     = totalVerts;
    m_StageResources.sceneMeshletData.totalMeshlets     = totalMeshlets;
    m_StageResources.sceneMeshletData.totalVertIdx      = totalVertIdx;
    m_StageResources.sceneMeshletData.totalPrimIdxBytes = totalPrimBytes;
    m_StageResources.sceneMeshletData.numAssetLods      = numAssetLods;

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
// Region windows + slot layout
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

    const uint32_t numLods = static_cast<uint32_t>(m_Registry.getLodSegments().size());

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
    const auto& assets      = m_Registry.getAssets();
    const uint32_t numLods   = static_cast<uint32_t>(m_Registry.getLodSegments().size());
    const uint32_t numAssets = static_cast<uint32_t>(assets.size());
    const uint32_t nCasc     = Render::c_NumCascades;

    m_NumMainSlots   = std::max(1u, numAssets * numLods);
    m_NumShadowSlots = std::max(1u, numAssets * nCasc);

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

    // Main slots (asset x LOD)
    m_MainSlotOffsets.assign(m_NumMainSlots, 0);
    m_MainASInvocsPerSlot.assign(m_NumMainSlots, 0);
    m_MainSlotTextureSet.assign(m_NumMainSlots, 0);
    m_MainDispatchArgsStaging.assign(m_NumMainSlots, {0, 0, 1, 1});
    m_MainVisBufferSize = 0;

    for (uint32_t ai = 0; ai < numAssets; ai++) {
        for (uint32_t lodi = 0; lodi < numLods; lodi++) {
            uint32_t slot = ai * numLods + lodi;
            m_MainSlotOffsets[slot] = m_MainVisBufferSize;
            m_MainVisBufferSize    += livePerAsset[ai];

            uint32_t meshletCount = (slot < m_MeshletMegabuffers.meshOffsets.size())
                ? m_MeshletMegabuffers.meshOffsets[slot].meshletCount : 0u;
            uint32_t invocations = std::max(1u,
                (meshletCount + Render::k_ASGroupSize - 1) / Render::k_ASGroupSize);
            m_MainASInvocsPerSlot[slot] = invocations;
            m_MainDispatchArgsStaging[slot].slotIdx = slot;

            if (ai < assets.size()) {
                uint32_t textureSet = assets[ai].textureSetIdx;
                if (!m_StageResources.sceneDraw.textureSets.empty())
                    textureSet = std::min<uint32_t>(textureSet, static_cast<uint32_t>(m_StageResources.sceneDraw.textureSets.size() - 1));
                m_MainSlotTextureSet[slot] = textureSet;
            }
        }
    }
    m_MainVisBufferSize = std::max(1u, m_MainVisBufferSize);

    // Shadow slots (asset x cascade, LOD 0 only)
    m_ShadowSlotOffsets.assign(m_NumShadowSlots, 0);
    m_ShadowASInvocsPerSlot.assign(m_NumShadowSlots, 0);
    m_ShadowDispatchArgsStaging.assign(m_NumShadowSlots, {0, 0, 1, 1});
    m_ShadowVisBufferSize = 0;

    for (uint32_t ai = 0; ai < numAssets; ai++) {
        for (uint32_t c = 0; c < nCasc; c++) {
            uint32_t slot = ai * nCasc + c;
            m_ShadowSlotOffsets[slot] = m_ShadowVisBufferSize;
            m_ShadowVisBufferSize    += livePerAsset[ai];

            // Shadow uses LOD 0 meshlet count for the corresponding asset.
            uint32_t lod0Slot = ai * numLods + 0;
            uint32_t meshletCount = (lod0Slot < m_MeshletMegabuffers.meshOffsets.size())
                ? m_MeshletMegabuffers.meshOffsets[lod0Slot].meshletCount : 0u;
            uint32_t invocations = std::max(1u,
                (meshletCount + Render::k_ASGroupSize - 1) / Render::k_ASGroupSize);
            m_ShadowASInvocsPerSlot[slot] = invocations;
            m_ShadowDispatchArgsStaging[slot].slotIdx = slot;
        }
    }
    m_ShadowVisBufferSize = std::max(1u, m_ShadowVisBufferSize);
}

void MeshShaderRenderPass::_UploadCullBuffers(nvrhi::ICommandList* cl) {
    auto device = GetDevice();
    const uint32_t numRegions = std::max(1u,
        static_cast<uint32_t>(m_Registry.getRegions().size()));

    // Persistent + cull data
    m_StageResources.cull.persistentInstBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_TotalCapacity * sizeof(Render::InstanceBufferEntry))
        .setStructStride(sizeof(Render::InstanceBufferEntry))
        .setDebugName("Mesh_PersistentInstBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_StageResources.cull.persistentInstBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_StageResources.cull.persistentInstBuffer, m_InstanceStaging.data(),
        m_TotalCapacity * sizeof(Render::InstanceBufferEntry));
    cl->setPermanentBufferState(m_StageResources.cull.persistentInstBuffer, nvrhi::ResourceStates::ShaderResource);

    m_StageResources.cull.cullDataBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_TotalCapacity * sizeof(Render::CullInstanceData))
        .setStructStride(sizeof(Render::CullInstanceData))
        .setDebugName("Mesh_CullDataBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_StageResources.cull.cullDataBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_StageResources.cull.cullDataBuffer, m_CullDataStaging.data(),
        m_TotalCapacity * sizeof(Render::CullInstanceData));
    cl->setPermanentBufferState(m_StageResources.cull.cullDataBuffer, nvrhi::ResourceStates::ShaderResource);

    m_StageResources.cull.cullRegionDataBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(numRegions * sizeof(Render::CullRegionData))
        .setStructStride(sizeof(Render::CullRegionData))
        .setDebugName("Mesh_CullRegionDataBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_StageResources.cull.cullRegionDataBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_StageResources.cull.cullRegionDataBuffer, m_RegionStaging.data(),
        numRegions * sizeof(Render::CullRegionData));
    cl->setPermanentBufferState(m_StageResources.cull.cullRegionDataBuffer, nvrhi::ResourceStates::ShaderResource);

    // --- Main slot SRVs ---
    m_StageResources.cull.mainSlotOffsetBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumMainSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_MainSlotOffsetBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_StageResources.cull.mainSlotOffsetBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_StageResources.cull.mainSlotOffsetBuffer, m_MainSlotOffsets.data(),
        m_NumMainSlots * sizeof(uint32_t));
    cl->setPermanentBufferState(m_StageResources.cull.mainSlotOffsetBuffer, nvrhi::ResourceStates::ShaderResource);

    m_StageResources.cull.mainASInvocsPerSlotBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumMainSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_MainASInvocsPerSlotBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_StageResources.cull.mainASInvocsPerSlotBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_StageResources.cull.mainASInvocsPerSlotBuffer, m_MainASInvocsPerSlot.data(),
        m_NumMainSlots * sizeof(uint32_t));
    cl->setPermanentBufferState(m_StageResources.cull.mainASInvocsPerSlotBuffer, nvrhi::ResourceStates::ShaderResource);

    // --- Shadow slot SRVs ---
    m_StageResources.cull.shadowSlotOffsetBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumShadowSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_ShadowSlotOffsetBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_StageResources.cull.shadowSlotOffsetBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_StageResources.cull.shadowSlotOffsetBuffer, m_ShadowSlotOffsets.data(),
        m_NumShadowSlots * sizeof(uint32_t));
    cl->setPermanentBufferState(m_StageResources.cull.shadowSlotOffsetBuffer, nvrhi::ResourceStates::ShaderResource);

    m_StageResources.cull.shadowASInvocsPerSlotBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumShadowSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_ShadowASInvocsPerSlotBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_StageResources.cull.shadowASInvocsPerSlotBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_StageResources.cull.shadowASInvocsPerSlotBuffer, m_ShadowASInvocsPerSlot.data(),
        m_NumShadowSlots * sizeof(uint32_t));
    cl->setPermanentBufferState(m_StageResources.cull.shadowASInvocsPerSlotBuffer, nvrhi::ResourceStates::ShaderResource);

    // --- UAVs ---
    m_StageResources.cull.regionVisibleBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(numRegions * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_RegionVisibleBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.mainCountBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumMainSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_MainCountBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.mainVisBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_MainVisBufferSize * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_MainVisBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.mainDispatchArgsBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumMainSlots * sizeof(DispatchRecord))
        .setDebugName("Mesh_MainDispatchArgsBuffer")
        .setIsDrawIndirectArgs(true)
        .setCanHaveUAVs(true)
        .setCanHaveRawViews(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.shadowCountBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumShadowSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_ShadowCountBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.shadowVisBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_ShadowVisBufferSize * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_ShadowVisBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.shadowDispatchArgsBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumShadowSlots * sizeof(DispatchRecord))
        .setDebugName("Mesh_ShadowDispatchArgsBuffer")
        .setIsDrawIndirectArgs(true)
        .setCanHaveUAVs(true)
        .setCanHaveRawViews(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_StageResources.cull.shadowUniqueCounter = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(sizeof(uint32_t))
        .setDebugName("Mesh_ShadowUniqueCounter")
        .setCanHaveUAVs(true)
        .setCanHaveRawViews(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    // The depth prepass consumes "last frame" mesh dispatch/count buffers before
    // the first per-frame cull reset runs. Seed those resources so frame zero is
    // a clean no-op instead of reading undefined indirect args.
    cl->clearBufferUInt(m_StageResources.cull.mainCountBuffer, 0);
    cl->clearBufferUInt(m_StageResources.cull.shadowCountBuffer, 0);
    cl->clearBufferUInt(m_StageResources.cull.shadowUniqueCounter, 0);
    cl->writeBuffer(m_StageResources.cull.mainDispatchArgsBuffer,
        m_MainDispatchArgsStaging.data(),
        m_MainDispatchArgsStaging.size() * sizeof(DispatchRecord));
    cl->writeBuffer(m_StageResources.cull.shadowDispatchArgsBuffer,
        m_ShadowDispatchArgsStaging.data(),
        m_ShadowDispatchArgsStaging.size() * sizeof(DispatchRecord));

    // Readback ring: [mainCounts][shadowCounts][shadowUnique]
    m_ReadbackMainEntries   = m_NumMainSlots;
    m_ReadbackShadowEntries = m_NumShadowSlots;
    const uint64_t readbackSize =
        (m_ReadbackMainEntries + m_ReadbackShadowEntries + 1) * sizeof(uint32_t);
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
    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Cull::kCB_Frame, m_StageResources.frameShared.constantBuffer,
            nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),

        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_RegionData,         m_StageResources.cull.cullRegionDataBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_InstanceData,       m_StageResources.cull.cullDataBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_MainSlotOffsets,   m_StageResources.cull.mainSlotOffsetBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_MainInvocations,     m_StageResources.cull.mainASInvocsPerSlotBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_ShadowSlotOffsets, m_StageResources.cull.shadowSlotOffsetBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Cull::kSRV_ShadowInvocations,   m_StageResources.cull.shadowASInvocsPerSlotBuffer),

        nvrhi::BindingSetItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_MainRegionVis,  m_StageResources.cull.regionVisibleBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_MainCount,      m_StageResources.cull.mainCountBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_MainVis,        m_StageResources.cull.mainVisBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_MainDispatch,          m_StageResources.cull.mainDispatchArgsBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_ShadowCount,    m_StageResources.cull.shadowCountBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(mesh_reg::Cull::kUAV_ShadowVis,      m_StageResources.cull.shadowVisBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_ShadowDispatch,        m_StageResources.cull.shadowDispatchArgsBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(mesh_reg::Cull::kUAV_ShadowUnique,          m_StageResources.cull.shadowUniqueCounter),

        nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Cull::kSRV_HiZ, m_StageResources.hiz.hizTexture),
        nvrhi::BindingSetItem::Sampler(mesh_reg::Cull::kSampler_HiZ, m_StageResources.hiz.pointSampler),
    };
    m_StageResources.cull.bindingSet = GetDevice()->createBindingSet(bsd, m_StageResources.cull.bindingLayout);
}

void MeshShaderRenderPass::_RebuildDrawBindingSet() {
    if (m_StageResources.sceneDraw.textureSets.empty()) return;
    if (!m_StageResources.shadow.depthTexture)     return;
    if (!m_StageResources.hiz.hizTexture)          return;

    m_StageResources.sceneDraw.bindingSets.resize(m_StageResources.sceneDraw.textureSets.size());

    for (size_t texIdx = 0; texIdx < m_StageResources.sceneDraw.textureSets.size(); texIdx++) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::PushConstants(mesh_reg::Draw::kPushC_Slot, mesh_reg::Draw::kPushCBytes),
            nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Draw::kCB_Frame, m_StageResources.frameShared.constantBuffer,
                nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
            nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Draw::kCB_ASCull, m_StageResources.frameShared.asCullCB,
                nvrhi::BufferRange(0, shader_cb::kMeshASCullSize)),

            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Positions,     m_StageResources.sceneMeshletData.positions),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Normals,       m_StageResources.sceneMeshletData.normals),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Tangents,      m_StageResources.sceneMeshletData.tangents),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Bitangents,    m_StageResources.sceneMeshletData.bitangents),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_UVs,           m_StageResources.sceneMeshletData.uvs),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_MeshletVertIdx,      m_StageResources.sceneMeshletData.meshletVertIdx),
            nvrhi::BindingSetItem::RawBuffer_SRV(mesh_reg::Draw::kSRV_MeshletPrimIdx,             m_StageResources.sceneMeshletData.meshletPrimIdx),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Meshlets,      m_StageResources.sceneMeshletData.meshletDescs),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_AssetLods,     m_StageResources.sceneMeshletData.assetLodRanges),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Vis,        m_StageResources.cull.mainVisBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_SlotOffsets,   m_StageResources.cull.mainSlotOffsetBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_SlotCounts,    m_StageResources.cull.mainCountBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Instances,     m_StageResources.cull.persistentInstBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_ASInvocations, m_StageResources.cull.mainASInvocsPerSlotBuffer),

            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Draw::kTex_Diffuse,   m_StageResources.sceneDraw.textureSets[texIdx].diffuse),
            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Draw::kTex_NormalMap, m_StageResources.sceneDraw.textureSets[texIdx].normalMap),
            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Draw::kTex_ShadowMap, m_StageResources.shadow.depthTexture),
            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Draw::kTex_HiZ,    m_StageResources.hiz.hizTexture),

            nvrhi::BindingSetItem::Sampler(mesh_reg::Draw::kSampler_Main,   m_StageResources.sceneDraw.sampler),
            nvrhi::BindingSetItem::Sampler(mesh_reg::Draw::kSampler_Shadow, m_StageResources.sceneDraw.shadowSampler),
            nvrhi::BindingSetItem::Sampler(mesh_reg::Draw::kSampler_HiZ,    m_StageResources.sceneDraw.hizSampler),
        };
        m_StageResources.sceneDraw.bindingSets[texIdx] = GetDevice()->createBindingSet(bsd, m_StageResources.sceneDraw.bindingLayout);
    }
}

void MeshShaderRenderPass::_RebuildShadowBindingSet() {
    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::PushConstants(mesh_reg::Draw::kPushC_Slot, mesh_reg::Draw::kPushCBytes),
        nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Draw::kCB_Frame, m_StageResources.frameShared.constantBuffer,
            nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),

        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Positions,     m_StageResources.sceneMeshletData.positions),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Normals,       m_StageResources.sceneMeshletData.normals),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Tangents,      m_StageResources.sceneMeshletData.tangents),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Bitangents,    m_StageResources.sceneMeshletData.bitangents),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_UVs,           m_StageResources.sceneMeshletData.uvs),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_MeshletVertIdx,      m_StageResources.sceneMeshletData.meshletVertIdx),
        nvrhi::BindingSetItem::RawBuffer_SRV(mesh_reg::Draw::kSRV_MeshletPrimIdx,             m_StageResources.sceneMeshletData.meshletPrimIdx),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Meshlets,      m_StageResources.sceneMeshletData.meshletDescs),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_AssetLods,     m_StageResources.sceneMeshletData.assetLodRanges),
        // Shadow-specific redirect:
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Vis,          m_StageResources.cull.shadowVisBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_SlotOffsets,     m_StageResources.cull.shadowSlotOffsetBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_SlotCounts,      m_StageResources.cull.shadowCountBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_Instances,       m_StageResources.cull.persistentInstBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::Draw::kSRV_ASInvocations, m_StageResources.cull.shadowASInvocsPerSlotBuffer),
    };
    m_StageResources.shadow.bindingSet = GetDevice()->createBindingSet(bsd, m_StageResources.shadow.bindingLayout);
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
    _RebuildShadowBindingSet();
    _RebuildDepthPrepassBindingSet();
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
    _RebuildShadowBindingSet();
    _RebuildDepthPrepassBindingSet();
}

// ===========================================================================
// Pipelines + command signatures
// ===========================================================================

void MeshShaderRenderPass::_CreateMainPipelineIfNeeded(nvrhi::IFramebuffer* framebuffer) {
    if (m_StageResources.sceneDraw.pipeline) return;

    nvrhi::MeshletPipelineDesc psoDesc;
    psoDesc.AS = m_StageResources.sceneDraw.amplificationShader;
    psoDesc.MS = m_StageResources.sceneDraw.meshShader;
    psoDesc.PS = m_StageResources.sceneDraw.pixelShader;
    psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
    psoDesc.bindingLayouts = { m_StageResources.sceneDraw.bindingLayout };

    auto& rs = psoDesc.renderState;
    rs.depthStencilState.depthTestEnable  = true;
    rs.depthStencilState.depthWriteEnable = true;
#if XYLEM_USE_REVERSE_Z
    rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::Greater;
#else
    rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::Less;
#endif
    rs.rasterState.cullMode = nvrhi::RasterCullMode::Back;

    m_StageResources.sceneDraw.pipeline = GetDevice()->createMeshletPipeline(psoDesc, framebuffer->getFramebufferInfo());
}

void MeshShaderRenderPass::_CreateShadowPipelineIfNeeded() {
    if (m_StageResources.shadow.pipeline)            return;
    if (!m_StageResources.shadow.framebuffers[0])    return;

    nvrhi::MeshletPipelineDesc psoDesc;
    psoDesc.AS = m_StageResources.shadow.amplificationShader;
    psoDesc.MS = m_StageResources.shadow.meshShader;
    psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
    psoDesc.bindingLayouts = { m_StageResources.shadow.bindingLayout };

    auto& rs = psoDesc.renderState;
    rs.depthStencilState.depthTestEnable  = true;
    rs.depthStencilState.depthWriteEnable = true;
    // Light-space shadow maps use the standard D3D depth range/projection built by
    // ViewHandler::computeCascades, independent of the main camera's reverse-Z mode.
    rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::Less;
    rs.rasterState.cullMode = nvrhi::RasterCullMode::Front;
    rs.rasterState.depthBias = 2;
    rs.rasterState.slopeScaledDepthBias = 2.f;

    m_StageResources.shadow.pipeline = GetDevice()->createMeshletPipeline(
        psoDesc, m_StageResources.shadow.framebuffers[0]->getFramebufferInfo());
}

void MeshShaderRenderPass::_EnsureDispatchMeshSignatures() {
    ID3D12Device* d3dDevice = GetDevice()->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
    if (!d3dDevice) return;

    auto makeSig = [&](nvrhi::MeshletPipelineHandle pipe,
                       nvrhi::RefCountPtr<ID3D12CommandSignature>& out,
                       const char* dbgName) {
        if (out || !pipe) return;
        ID3D12RootSignature* rootSig = pipe->getNativeObject(nvrhi::ObjectTypes::D3D12_RootSignature);
        if (!rootSig) return;

        D3D12_INDIRECT_ARGUMENT_DESC args[2] = {};
        args[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
        args[0].Constant.RootParameterIndex      = mesh_reg::kPushC_RootParamIdx;
        args[0].Constant.DestOffsetIn32BitValues = 0;
        args[0].Constant.Num32BitValuesToSet     = 1;
        args[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;

        D3D12_COMMAND_SIGNATURE_DESC desc = {};
        desc.ByteStride       = sizeof(DispatchRecord);
        desc.NumArgumentDescs = 2;
        desc.pArgumentDescs   = args;

        HRESULT hr = d3dDevice->CreateCommandSignature(&desc, rootSig, IID_PPV_ARGS(&out));
        if (FAILED(hr)) {
            log::error("MeshShaderRenderPass: CreateCommandSignature(%s) failed (0x%08x)", dbgName, hr);
            out = nullptr;
        }
    };

    makeSig(m_StageResources.sceneDraw.pipeline,   m_DispatchMeshSignature,         "main");
    makeSig(m_StageResources.shadow.pipeline, m_StageResources.shadow.dispatchMeshSignature,  "shadow");
}

// ===========================================================================
// IRenderPass overrides
// ===========================================================================

void MeshShaderRenderPass::Animate(float /*seconds*/) {
    GetDeviceManager()->SetInformativeWindowTitle("Xylem (MeshShader)");
    frame::RunDirtyCycle(m_Registry, *this);
}

void MeshShaderRenderPass::BackBufferResizing() {
    m_StageResources.sceneDraw.pipeline          = nullptr;
    m_StageResources.shadow.pipeline        = nullptr;
    m_DispatchMeshSignature  = nullptr;
    m_StageResources.shadow.dispatchMeshSignature = nullptr;

    m_StageResources.depthPrepass.pipeline        = nullptr;
    m_StageResources.depthPrepass.terrainPipeline = nullptr;
    m_StageResources.depthPrepass.depthTexture    = nullptr;
    m_StageResources.depthPrepass.framebuffer     = nullptr;
    m_StageResources.depthPrepass.bindingSet      = nullptr;
    m_StageResources.depthPrepass.dispatchMeshSignature = nullptr;

    m_StageResources.sceneTerrain.pipeline = nullptr;
    m_StageResources.sky.pipeline     = nullptr;

    m_StageResources.hiz.buildBindingSets.clear();
    m_StageResources.hiz.debugMipTextures.clear();
    m_StageResources.hiz.numMips = 0;
    m_UI.hizMipTextures.clear();
}

void MeshShaderRenderPass::Render(nvrhi::IFramebuffer* framebuffer) {
    frame::SetShadowDebugOutputs(
        m_StageOutputs,
        m_StageResources.shadow.depthTexture.Get(),
        m_StageResources.shadow.debugTextures);
    frame::SetHiZDebugOutputs(m_StageOutputs, m_StageResources.hiz.debugMipTextures);
    frame::PublishStageOutputsToUI(m_StageOutputs, m_UI);

    frame::FrameContext frameContext = frame::BuildFrameContext(
        m_Registry,
        m_ViewHandler,
        framebuffer,
        !m_StageResources.sceneDraw.pipeline);

    const auto& fbInfo = framebuffer->getFramebufferInfo();
    const uint32_t fbW = frameContext.frameWidth;
    const uint32_t fbH = frameContext.frameHeight;

    _CreateMainPipelineIfNeeded(framebuffer);
    _CreateShadowPipelineIfNeeded();
    _EnsureDispatchMeshSignatures();

    const dm::float3& camPos = m_ViewHandler.camera.GetPosition();
    const dm::float3& camDir = m_ViewHandler.camera.GetDir();
    frame::ComputeCascades(
        frameContext,
        m_ViewHandler,
        m_Registry,
        k_ShadowRes,
        m_UI.pssmLambda);

    // Hi-Z bypass when camera looks steeply downward (mirrors ComputeRenderPass).
    float downwardness = -m_ViewHandler.camera.GetDir().y;
    bool hizActive = downwardness < m_UI.hizBypassAngle;
    m_UI.hizActiveThisFrame = hizActive;

    m_CommandList->open();

    // Make sure Hi-Z / depth-prepass resources match the back-buffer resolution.
    _EnsureHiZResources(fbW, fbH);

    // ----- 1. Fill CullConstantBufferEntry (CPU cascades seed; SDSM may overwrite) -----
    Render::CullConstantBufferEntry cb = {};
    frame::FillCommonFrameConstants(cb, m_ViewHandler, m_Registry.getSunDirection());

    for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
        const dm::box3& cBbox = m_ViewHandler.cascades[c].shadowCasterBboxLS;
        if (cBbox.isempty()) {
            cb.shadowCasterMinLS[c] = dm::float4( 1e30f,  1e30f,  1e30f, 0.f);
            cb.shadowCasterMaxLS[c] = dm::float4(-1e30f, -1e30f, -1e30f, 0.f);
        } else {
            cb.shadowCasterMinLS[c] = dm::float4(cBbox.m_mins, 0.f);
            cb.shadowCasterMaxLS[c] = dm::float4(cBbox.m_maxs, 0.f);
        }
    }

    cb.viewFrustum   = m_ViewHandler.view.GetViewFrustum();
    cb.worldToLight  = dm::affineToHomogeneous(m_ViewHandler.worldToLight);
    cb.cameraPos     = camPos;
    cb.numRegions    = static_cast<uint32_t>(m_Registry.getRegions().size());
    cb.totalCapacity = m_TotalCapacity;

    const auto& lodDistances = m_Registry.getLodDistances();
    cb.numLods = static_cast<uint32_t>(m_Registry.getLodSegments().size());
    for (uint32_t i = 0; i < cb.numLods; i++)
        cb.lodDistances[i] = lodDistances[i];

    cb.hizDimensions = dm::float2(static_cast<float>(frameContext.frameWidth), static_cast<float>(frameContext.frameHeight));
    cb.maxHiZMip     = static_cast<float>((m_StageResources.hiz.numMips > 0) ? (m_StageResources.hiz.numMips - 1) : 0);
    cb.hizEnabled    = hizActive ? 1u : 0u;

    m_CommandList->writeBuffer(m_StageResources.frameShared.constantBuffer, &cb, shader_cb::kCullFrameSize);

    // ----- 1b. Fill AS-cull constants - hizEnabled=0 for the depth prepass (Hi-Z not yet built) -----
    shader_cb::MeshASCullConstants asCB = {};
    asCB.cameraPos         = camPos;
    asCB.hizEnabled        = 0u;
    asCB.hizDimensions     = dm::float2(static_cast<float>(fbW), static_cast<float>(fbH));
    asCB.maxHiZMip         = static_cast<float>((m_StageResources.hiz.numMips > 0) ? (m_StageResources.hiz.numMips - 1) : 0);
    asCB.asConeCullEnabled = 1u;
    m_CommandList->writeBuffer(m_StageResources.frameShared.asCullCB, &asCB, sizeof(shader_cb::MeshASCullConstants));

    // ----- 2. Depth prepass + Hi-Z build + SDSM (skip when bypassed) -----
    // Reads LAST frame's m_StageResources.cull.mainDispatchArgsBuffer - must happen BEFORE we
    // reset/clear the cull output buffers below.
    if (hizActive) {
        m_CommandList->beginMarker("HiZ");

        m_CommandList->beginMarker("DepthPrepass");
        _RenderDepthPrepass();
        m_CommandList->endMarker();

        m_CommandList->beginMarker("BuildMipChain");
        _BuildHiZMipChain();
        m_CommandList->endMarker();

        m_CommandList->endMarker();

        m_CommandList->beginMarker("SDSM");
        float regionNear, regionFar;
        _ComputeRegionEnvelope(m_ViewHandler.view.GetViewFrustum(), camPos, camDir,
                               regionNear, regionFar);
        _RunSDSMBuildCascades(frameContext.sceneBounds, frameContext.aspectRatio, dm::radians(60.f),
                              regionNear, regionFar);
        m_CommandList->endMarker();
    }

    // Re-write ASCullCB now that Hi-Z is populated - main_as uses it during color pass.
    asCB.hizEnabled = hizActive ? 1u : 0u;
    m_CommandList->writeBuffer(m_StageResources.frameShared.asCullCB, &asCB, sizeof(shader_cb::MeshASCullConstants));

    // ----- 3. Per-frame cull resets -----
    m_CommandList->clearBufferUInt(m_StageResources.cull.mainCountBuffer,     0);
    m_CommandList->clearBufferUInt(m_StageResources.cull.shadowCountBuffer,   0);
    m_CommandList->clearBufferUInt(m_StageResources.cull.shadowUniqueCounter, 0);

    m_CommandList->writeBuffer(m_StageResources.cull.mainDispatchArgsBuffer,
        m_MainDispatchArgsStaging.data(),
        m_MainDispatchArgsStaging.size() * sizeof(DispatchRecord));
    m_CommandList->writeBuffer(m_StageResources.cull.shadowDispatchArgsBuffer,
        m_ShadowDispatchArgsStaging.data(),
        m_ShadowDispatchArgsStaging.size() * sizeof(DispatchRecord));

    // ----- 4. GPU cull dispatches -----
    m_CommandList->beginMarker("MeshCull");
    {
        nvrhi::ComputeState cs;
        cs.bindings = { m_StageResources.cull.bindingSet };

        m_CommandList->beginMarker("RegionDispatch");
        cs.pipeline = m_StageResources.cull.regionPipeline;
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch((cb.numRegions + 63) / 64, 1, 1);
        m_CommandList->endMarker();

        m_CommandList->beginMarker("MainDispatch");
        cs.pipeline = m_StageResources.cull.mainPipeline;
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch((m_TotalCapacity + 255) / 256, 1, 1);
        m_CommandList->endMarker();

        m_CommandList->beginMarker("ShadowDispatch");
        cs.pipeline = m_StageResources.cull.shadowPipeline;
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch((m_TotalCapacity + 255) / 256, 1, 1);
        m_CommandList->endMarker();
    }
    m_CommandList->endMarker();

    // Readback ring copy: [main counts][shadow counts][shadow unique]
    {
        uint32_t ringSlot = m_ReadbackFrameIndex % k_QueuedFrames;
        uint64_t offset = 0;
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], offset,
                                  m_StageResources.cull.mainCountBuffer, 0,
                                  m_ReadbackMainEntries * sizeof(uint32_t));
        offset += m_ReadbackMainEntries * sizeof(uint32_t);
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], offset,
                                  m_StageResources.cull.shadowCountBuffer, 0,
                                  m_ReadbackShadowEntries * sizeof(uint32_t));
        offset += m_ReadbackShadowEntries * sizeof(uint32_t);
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], offset,
                                  m_StageResources.cull.shadowUniqueCounter, 0,
                                  sizeof(uint32_t));
    }

    // ----- 4. Shadow pass: 4 cascades via ExecuteIndirect(DISPATCH_MESH) -----
    if (m_StageResources.shadow.pipeline && m_StageResources.shadow.dispatchMeshSignature) {
        m_CommandList->beginMarker("ShadowMeshDraw");

        // Clear all cascades.
        for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
            nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, m_StageResources.shadow.framebuffers[c], 1.f, 0);
        }

        // Shared shadow meshlet state; re-bind the framebuffer per cascade slice and
        // ExecuteIndirect only for that cascade's interleaved asset slots.
        const uint32_t numAssets = static_cast<uint32_t>(m_Registry.getAssets().size());

        // The shadow slot layout is interleaved as slot = asset*NUM_CASCADES + cascade.

        for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
            nvrhi::MeshletState ms;
            ms.pipeline       = m_StageResources.shadow.pipeline;
            ms.framebuffer    = m_StageResources.shadow.framebuffers[c];
            ms.bindings       = { m_StageResources.shadow.bindingSet };
            ms.indirectParams = m_StageResources.cull.shadowDispatchArgsBuffer;
            ms.viewport.addViewportAndScissorRect(
                nvrhi::Viewport(float(k_ShadowRes), float(k_ShadowRes)));
            m_CommandList->setMeshletState(ms);

            uint32_t placeholder = 0;
            m_CommandList->setPushConstants(&placeholder, sizeof(uint32_t));

            auto* d3dList = static_cast<ID3D12GraphicsCommandList6*>(
                m_CommandList->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList));
            auto* argBuffer = static_cast<ID3D12Resource*>(
                m_StageResources.cull.shadowDispatchArgsBuffer->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource));

            if (d3dList && argBuffer) {
                for (uint32_t ai = 0; ai < numAssets; ai++) {
                    const uint32_t slot = ai * Render::c_NumCascades + c;
                    d3dList->ExecuteIndirect(
                        m_StageResources.shadow.dispatchMeshSignature.Get(),
                        1,
                        argBuffer,
                        slot * sizeof(DispatchRecord),
                        nullptr,
                        0);
                }
            }
        }
        m_CommandList->endMarker();
    }

    // Copy shadow cascade slices into debug textures for UI display.
    if (m_UI.showShadowMap) {
        for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
            m_CommandList->copyTexture(
                m_StageResources.shadow.debugTextures[c], nvrhi::TextureSlice(),
                m_StageResources.shadow.depthTexture, nvrhi::TextureSlice().setArraySlice(c));
        }
    }

    // ----- 5. Clear main framebuffer -----
    nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));
#if XYLEM_USE_REVERSE_Z
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer, 0.f, 0);
#else
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, framebuffer, 1.f, 0);
#endif

    // ----- 5b. Sky -----
    m_CommandList->beginMarker("Sky");
    _RenderSkyPass(framebuffer);
    m_CommandList->endMarker();

    // ----- 6. Main color pass via ExecuteIndirect(DISPATCH_MESH) -----
    m_CommandList->beginMarker("MeshDraw");
    {
        nvrhi::MeshletState meshState;
        meshState.pipeline       = m_StageResources.sceneDraw.pipeline;
        meshState.framebuffer    = framebuffer;
        meshState.indirectParams = m_StageResources.cull.mainDispatchArgsBuffer;
        meshState.viewport.addViewportAndScissorRect(fbInfo.getViewport());
        if (m_DispatchMeshSignature && !m_StageResources.sceneDraw.bindingSets.empty()) {
            auto* d3dList = static_cast<ID3D12GraphicsCommandList6*>(
                m_CommandList->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList));
            auto* argBuffer = static_cast<ID3D12Resource*>(
                m_StageResources.cull.mainDispatchArgsBuffer->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource));

            if (d3dList && argBuffer) {
                for (uint32_t texIdx = 0; texIdx < m_StageResources.sceneDraw.bindingSets.size(); texIdx++) {
                    meshState.bindings = { m_StageResources.sceneDraw.bindingSets[texIdx] };
                    m_CommandList->setMeshletState(meshState);

                    uint32_t placeholder = 0;
                    m_CommandList->setPushConstants(&placeholder, sizeof(uint32_t));

                    for (uint32_t slot = 0; slot < m_NumMainSlots; slot++) {
                        if (slot >= m_MainSlotTextureSet.size()) continue;
                        if (m_MainSlotTextureSet[slot] != texIdx) continue;

                        d3dList->ExecuteIndirect(
                            m_DispatchMeshSignature.Get(),
                            1,
                            argBuffer,
                            slot * sizeof(DispatchRecord),
                            nullptr,
                            0);
                    }
                }
            }
        }
    }
    m_CommandList->endMarker();

    // ----- 6b. Terrain color -----
    m_CommandList->beginMarker("Terrain");
    _RenderScenePass(framebuffer);
    m_CommandList->endMarker();

    m_CommandList->close();
    GetDevice()->executeCommandList(m_CommandList);

    // ----- 7. UI stats -----
    m_UI.totalInstanceCount = static_cast<uint32_t>(m_Registry.totalInstanceCount());
    m_UI.drawCallCount      = m_NumMainSlots + m_NumShadowSlots;

    if (m_ReadbackFrameIndex >= (k_QueuedFrames - 1)) {
        uint32_t readSlot = (m_ReadbackFrameIndex + 1) % k_QueuedFrames;
        void* pData = GetDevice()->mapBuffer(m_ReadbackBuffers[readSlot], nvrhi::CpuAccessMode::Read);
        if (pData) {
            const uint32_t* counts = static_cast<const uint32_t*>(pData);
            uint32_t visSum = 0;
            for (uint32_t i = 0; i < m_ReadbackMainEntries; i++) visSum += counts[i];

            uint32_t shadowVisSum = 0;
            for (uint32_t i = 0; i < m_ReadbackShadowEntries; i++)
                shadowVisSum += counts[m_ReadbackMainEntries + i];

            uint32_t shadowUnique = counts[m_ReadbackMainEntries + m_ReadbackShadowEntries];

            GetDevice()->unmapBuffer(m_ReadbackBuffers[readSlot]);

            m_UI.visibleInstanceCount = visSum;
            m_UI.culledInstanceCount  = (visSum <= m_UI.totalInstanceCount)
                ? m_UI.totalInstanceCount - visSum : 0;
            m_UI.shadowVisibleCount     = shadowUnique;
            m_UI.shadowCulledCount      = (shadowUnique <= m_UI.totalInstanceCount)
                ? m_UI.totalInstanceCount - shadowUnique : 0;
            m_UI.shadowCascadeDrawCount = shadowVisSum;
            m_UI.shadowOverdrawCount    = (shadowVisSum >= shadowUnique)
                ? shadowVisSum - shadowUnique : 0;
        }
    }
    m_ReadbackFrameIndex++;
}

// ===========================================================================
// Depth prepass: reuses main_as + depth_ms (no PS) for trees + traditional
// raster for terrain. Reads last-frame mainDispatchArgsBuffer.
// ===========================================================================

bool MeshShaderRenderPass::_InitDepthPrepass() {
    if (!m_ShaderFactory) return false;

    // Tree depth uses shared main_as (cone cull, Hi-Z toggled via ASCullCB) + depth_ms.
    m_StageResources.depthPrepass.amplificationShader = m_StageResources.sceneDraw.amplificationShader;
    m_StageResources.depthPrepass.meshShader = m_ShaderFactory->CreateShader("app/MeshShaderPass.hlsl",
        "depth_ms", nullptr, nvrhi::ShaderType::Mesh);
    if (!m_StageResources.depthPrepass.meshShader) {
        log::error("MeshShaderRenderPass: depth_ms compile failed");
        return false;
    }

    // Tree depth pipeline shares the main draw binding layout - same SRVs/CBs.
    m_StageResources.depthPrepass.bindingLayout = m_StageResources.sceneDraw.bindingLayout;

    // Terrain depth path uses the existing DepthPrepass.hlsl terrain_vs (CB only).
    m_StageResources.depthPrepass.terrainVS = m_ShaderFactory->CreateShader("app/DepthPrepass.hlsl",
        "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    if (!m_StageResources.depthPrepass.terrainVS) {
        log::error("MeshShaderRenderPass: DepthPrepass.terrain_vs compile failed");
        return false;
    }

    nvrhi::VertexAttributeDesc terrainAttrs[] = {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION").setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(Scene::TerrainVertex, pos))
            .setBufferIndex(0).setElementStride(sizeof(Scene::TerrainVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("NORMAL").setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(Scene::TerrainVertex, normal))
            .setBufferIndex(0).setElementStride(sizeof(Scene::TerrainVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("UV").setFormat(nvrhi::Format::RG32_FLOAT)
            .setOffset(offsetof(Scene::TerrainVertex, uv))
            .setBufferIndex(0).setElementStride(sizeof(Scene::TerrainVertex)),
    };
    m_StageResources.depthPrepass.terrainInputLayout = GetDevice()->createInputLayout(
        terrainAttrs, uint32_t(std::size(terrainAttrs)), m_StageResources.depthPrepass.terrainVS);
    if (!m_StageResources.depthPrepass.terrainInputLayout) return false;

    // Terrain depth binding layout: CB(0) only - DepthPrepass.terrain_vs ignores the
    // vis/inst/slot SRVs declared in that file.
    nvrhi::BindingLayoutDesc terrainBLD;
    terrainBLD.visibility = nvrhi::ShaderType::All;
    terrainBLD.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::DepthPrepass::kCB_Frame),
        nvrhi::BindingLayoutItem::PushConstants(mesh_reg::DepthPrepass::kPushC_Slot, sizeof(uint32_t)),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::DepthPrepass::kSRV_Vis),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::DepthPrepass::kSRV_Instances),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(mesh_reg::DepthPrepass::kSRV_SlotOffsets),
    };
    m_StageResources.depthPrepass.terrainBindingLayout = GetDevice()->createBindingLayout(terrainBLD);
    return m_StageResources.depthPrepass.terrainBindingLayout != nullptr;
}

bool MeshShaderRenderPass::_InitHiZShaders() {
    auto device = GetDevice();

    m_StageResources.hiz.copyCS = m_ShaderFactory->CreateShader("app/HiZBuild.hlsl",
        "HiZCopy", nullptr, nvrhi::ShaderType::Compute);
    m_StageResources.hiz.buildCS = m_ShaderFactory->CreateShader("app/HiZBuild.hlsl",
        "HiZDownsample", nullptr, nvrhi::ShaderType::Compute);
    if (!m_StageResources.hiz.copyCS || !m_StageResources.hiz.buildCS) return false;

    nvrhi::BindingLayoutDesc bld;
    bld.visibility = nvrhi::ShaderType::Compute;
    bld.bindings = {
        nvrhi::BindingLayoutItem::PushConstants(mesh_reg::HiZ::kPushC_DestDimensions, sizeof(uint32_t) * mesh_reg::HiZ::kPushCDwordCount),
        nvrhi::BindingLayoutItem::Texture_SRV(mesh_reg::HiZ::kSRV_Source),
        nvrhi::BindingLayoutItem::Texture_UAV(mesh_reg::HiZ::kUAV_Dest),
    };
    m_StageResources.hiz.buildBindingLayout = device->createBindingLayout(bld);
    if (!m_StageResources.hiz.buildBindingLayout) return false;

    nvrhi::ComputePipelineDesc copyPso;
    copyPso.CS = m_StageResources.hiz.copyCS;
    copyPso.bindingLayouts = { m_StageResources.hiz.buildBindingLayout };
    m_StageResources.hiz.copyPipeline = device->createComputePipeline(copyPso);

    nvrhi::ComputePipelineDesc buildPso;
    buildPso.CS = m_StageResources.hiz.buildCS;
    buildPso.bindingLayouts = { m_StageResources.hiz.buildBindingLayout };
    m_StageResources.hiz.buildPipeline = device->createComputePipeline(buildPso);
    if (!m_StageResources.hiz.copyPipeline || !m_StageResources.hiz.buildPipeline) return false;

    m_StageResources.hiz.pointSampler = device->createSampler(nvrhi::SamplerDesc()
        .setAllFilters(false)
        .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp));
    if (!m_StageResources.hiz.pointSampler) return false;

    // 1x1 placeholder so binding sets resolve before first _EnsureHiZResources.
    m_StageResources.hiz.hizTexture = device->createTexture(nvrhi::TextureDesc()
        .setWidth(1).setHeight(1).setMipLevels(1)
        .setFormat(nvrhi::Format::RG32_FLOAT)
        .setIsUAV(true)
        .setInitialState(nvrhi::ResourceStates::ShaderResource)
        .setKeepInitialState(true)
        .setDebugName("MeshHiZ_Placeholder"));
    m_StageResources.hiz.numMips = 1;
    return true;
}

bool MeshShaderRenderPass::_InitSDSMPass() {
    auto device = GetDevice();

    m_StageResources.sdsm.buildCS = m_ShaderFactory->CreateShader("app/SDSMBuildCascades.hlsl",
        "BuildCascades", nullptr, nvrhi::ShaderType::Compute);
    if (!m_StageResources.sdsm.buildCS) return false;

    nvrhi::BindingLayoutDesc bld;
    bld.visibility = nvrhi::ShaderType::Compute;
    bld.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(mesh_reg::SDSM::kCB_Input),
        nvrhi::BindingLayoutItem::Texture_SRV(mesh_reg::SDSM::kSRV_HiZ),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(mesh_reg::SDSM::kUAV_CascadeOut),
    };
    m_StageResources.sdsm.buildBindingLayout = device->createBindingLayout(bld);
    if (!m_StageResources.sdsm.buildBindingLayout) return false;

    nvrhi::ComputePipelineDesc pso;
    pso.CS = m_StageResources.sdsm.buildCS;
    pso.bindingLayouts = { m_StageResources.sdsm.buildBindingLayout };
    m_StageResources.sdsm.buildPipeline = device->createComputePipeline(pso);
    if (!m_StageResources.sdsm.buildPipeline) return false;

    m_StageResources.sdsm.inputCB = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(shader_cb::kSDSMCascadeBuildInputSize)
        .setIsConstantBuffer(true)
        .setDebugName("MeshSDSM_InputCB")
        .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer));
    if (!m_StageResources.sdsm.inputCB) return false;

    m_StageResources.sdsm.cascadeDataBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(sizeof(shader_cb::SDSMCascadeBuildOutput))
        .setStructStride(sizeof(shader_cb::SDSMCascadeBuildOutput))
        .setDebugName("MeshSDSM_CascadeData")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));
    if (!m_StageResources.sdsm.cascadeDataBuffer) return false;

    // Initial binding set references placeholder Hi-Z; rebuilt in _EnsureHiZResources.
    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::SDSM::kCB_Input, m_StageResources.sdsm.inputCB),
        nvrhi::BindingSetItem::Texture_SRV(mesh_reg::SDSM::kSRV_HiZ, m_StageResources.hiz.hizTexture),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(mesh_reg::SDSM::kUAV_CascadeOut, m_StageResources.sdsm.cascadeDataBuffer),
    };
    m_StageResources.sdsm.buildBindingSet = device->createBindingSet(bsd, m_StageResources.sdsm.buildBindingLayout);
    return m_StageResources.sdsm.buildBindingSet != nullptr;
}

bool MeshShaderRenderPass::_InitTerrainPass(nvrhi::ICommandList* initCL) {
    const auto* terrain = m_Registry.getTerrain();
    if (!terrain) return true;

    const auto& verts   = terrain->getVertices();
    const auto& indices = terrain->getIndices();
    if (verts.empty() || indices.empty()) return true;

    m_StageResources.sceneTerrain.indexCount = static_cast<uint32_t>(indices.size());

    m_StageResources.sceneTerrain.vertexShader = m_ShaderFactory->CreateShader(
        "app/terrain_compute.hlsl", "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_StageResources.sceneTerrain.pixelShader = m_ShaderFactory->CreateShader(
        "app/terrain_compute.hlsl", "terrain_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_StageResources.sceneTerrain.vertexShader || !m_StageResources.sceneTerrain.pixelShader) return false;

    nvrhi::VertexAttributeDesc attrs[] = {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION").setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(Scene::TerrainVertex, pos))
            .setBufferIndex(0).setElementStride(sizeof(Scene::TerrainVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("NORMAL").setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(offsetof(Scene::TerrainVertex, normal))
            .setBufferIndex(0).setElementStride(sizeof(Scene::TerrainVertex)),
        nvrhi::VertexAttributeDesc()
            .setName("UV").setFormat(nvrhi::Format::RG32_FLOAT)
            .setOffset(offsetof(Scene::TerrainVertex, uv))
            .setBufferIndex(0).setElementStride(sizeof(Scene::TerrainVertex)),
    };
    m_StageResources.sceneTerrain.inputLayout = GetDevice()->createInputLayout(
        attrs, uint32_t(std::size(attrs)), m_StageResources.sceneTerrain.vertexShader);
    if (!m_StageResources.sceneTerrain.inputLayout) return false;

    nvrhi::BufferDesc vbDesc;
    vbDesc.isVertexBuffer = true;
    vbDesc.byteSize       = verts.size() * sizeof(Scene::TerrainVertex);
    vbDesc.debugName      = "MeshTerrainVB";
    vbDesc.initialState   = nvrhi::ResourceStates::CopyDest;
    m_StageResources.sceneTerrain.vertexBuffer = GetDevice()->createBuffer(vbDesc);
    initCL->beginTrackingBufferState(m_StageResources.sceneTerrain.vertexBuffer, nvrhi::ResourceStates::CopyDest);
    initCL->writeBuffer(m_StageResources.sceneTerrain.vertexBuffer, verts.data(), vbDesc.byteSize);
    initCL->setPermanentBufferState(m_StageResources.sceneTerrain.vertexBuffer, nvrhi::ResourceStates::VertexBuffer);

    nvrhi::BufferDesc ibDesc;
    ibDesc.isIndexBuffer = true;
    ibDesc.byteSize      = indices.size() * sizeof(uint32_t);
    ibDesc.debugName     = "MeshTerrainIB";
    ibDesc.initialState  = nvrhi::ResourceStates::CopyDest;
    m_StageResources.sceneTerrain.indexBuffer = GetDevice()->createBuffer(ibDesc);
    initCL->beginTrackingBufferState(m_StageResources.sceneTerrain.indexBuffer, nvrhi::ResourceStates::CopyDest);
    initCL->writeBuffer(m_StageResources.sceneTerrain.indexBuffer, indices.data(), ibDesc.byteSize);
    initCL->setPermanentBufferState(m_StageResources.sceneTerrain.indexBuffer, nvrhi::ResourceStates::IndexBuffer);

    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Terrain::kCB_Frame, m_StageResources.frameShared.constantBuffer,
            nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
        nvrhi::BindingSetItem::Sampler(mesh_reg::Terrain::kSampler_Shadow, m_StageResources.sceneDraw.shadowSampler),
        nvrhi::BindingSetItem::Texture_SRV(mesh_reg::Terrain::kTex_ShadowMap, m_StageResources.shadow.depthTexture),
    };
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
            bsd, m_StageResources.sceneTerrain.bindingLayout, m_StageResources.sceneTerrain.bindingSet))
        return false;

    return true;
}

bool MeshShaderRenderPass::_InitSkyPass() {
    m_StageResources.sky.vertexShader = m_ShaderFactory->CreateShader(
        "app/sky.hlsl", "sky_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_StageResources.sky.pixelShader = m_ShaderFactory->CreateShader(
        "app/sky.hlsl", "sky_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_StageResources.sky.vertexShader || !m_StageResources.sky.pixelShader) return false;

    nvrhi::BufferDesc cbDesc;
    cbDesc.byteSize         = sizeof(SkyConstants);
    cbDesc.isConstantBuffer = true;
    cbDesc.isVolatile       = true;
    cbDesc.maxVersions      = 16;
    cbDesc.debugName        = "MeshSkyConstants";
    m_StageResources.sky.constantBuffer = GetDevice()->createBuffer(cbDesc);
    if (!m_StageResources.sky.constantBuffer) return false;

    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::Sky::kCB_Sky, m_StageResources.sky.constantBuffer),
    };
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
            bsd, m_StageResources.sky.bindingLayout, m_StageResources.sky.bindingSet))
        return false;

    return true;
}

// ===========================================================================
// Binding-set rebuild for depth prepass
// ===========================================================================

void MeshShaderRenderPass::_RebuildDepthPrepassBindingSet() {
    // Tree meshlet depth prepass shares the main Draw binding set.
    m_StageResources.depthPrepass.bindingSet = m_StageResources.sceneDraw.bindingSets.empty()
        ? nullptr
        : m_StageResources.sceneDraw.bindingSets[0];

    // Terrain depth prepass: CB + dummy vis/inst/slotOffsets (not read by terrain_vs).
    if (!m_StageResources.depthPrepass.terrainBindingLayout) return;
    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::DepthPrepass::kCB_Frame, m_StageResources.frameShared.constantBuffer,
            nvrhi::BufferRange(0, shader_cb::kCullFrameSize)),
        nvrhi::BindingSetItem::PushConstants(mesh_reg::DepthPrepass::kPushC_Slot, sizeof(uint32_t)),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::DepthPrepass::kSRV_Vis, m_StageResources.cull.mainVisBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::DepthPrepass::kSRV_Instances, m_StageResources.cull.persistentInstBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(mesh_reg::DepthPrepass::kSRV_SlotOffsets, m_StageResources.cull.mainSlotOffsetBuffer),
    };
    m_StageResources.depthPrepass.terrainBindingSet = GetDevice()->createBindingSet(
        bsd, m_StageResources.depthPrepass.terrainBindingLayout);
}

// ===========================================================================
// Hi-Z resource (re)creation - called on back-buffer resize
// ===========================================================================

void MeshShaderRenderPass::_EnsureHiZResources(uint32_t width, uint32_t height) {
    if (m_StageResources.depthPrepass.depthTexture) {
        auto desc = m_StageResources.depthPrepass.depthTexture->getDesc();
        if (desc.width == width && desc.height == height) return;
    }

    auto device = GetDevice();

    m_StageResources.depthPrepass.depthTexture = device->createTexture(nvrhi::TextureDesc()
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
        .setDebugName("MeshDepthPrepass_Depth"));
    m_StageResources.depthPrepass.framebuffer = device->createFramebuffer(
        nvrhi::FramebufferDesc().setDepthAttachment(m_StageResources.depthPrepass.depthTexture));

    m_StageResources.depthPrepass.pipeline         = nullptr;
    m_StageResources.depthPrepass.terrainPipeline  = nullptr;
    m_StageResources.depthPrepass.dispatchMeshSignature = nullptr;

    m_StageResources.hiz.numMips = static_cast<uint32_t>(
        std::floor(std::log2(std::max(width, height)))) + 1;
    m_StageResources.hiz.hizTexture = device->createTexture(nvrhi::TextureDesc()
        .setWidth(width).setHeight(height)
        .setMipLevels(m_StageResources.hiz.numMips)
        .setFormat(nvrhi::Format::RG32_FLOAT)
        .setIsUAV(true)
        .setInitialState(nvrhi::ResourceStates::ShaderResource)
        .setKeepInitialState(true)
        .setDebugName("MeshHiZTexture"));

    m_StageResources.hiz.buildBindingSets.resize(m_StageResources.hiz.numMips);

    // Mip 0 - copy depth prepass (D32 -> R32_FLOAT view) into RG32 mip 0.
    {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::PushConstants(mesh_reg::HiZ::kPushC_DestDimensions, sizeof(uint32_t) * mesh_reg::HiZ::kPushCDwordCount),
            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::HiZ::kSRV_Source, m_StageResources.depthPrepass.depthTexture,
                nvrhi::Format::R32_FLOAT,
                nvrhi::TextureSubresourceSet(0, 1, 0, 1)),
            nvrhi::BindingSetItem::Texture_UAV(mesh_reg::HiZ::kUAV_Dest, m_StageResources.hiz.hizTexture,
                nvrhi::Format::RG32_FLOAT,
                nvrhi::TextureSubresourceSet(0, 1, 0, 1)),
        };
        m_StageResources.hiz.buildBindingSets[0] = device->createBindingSet(bsd, m_StageResources.hiz.buildBindingLayout);
    }

    // Mips 1..N-1 - downsample i-1 -> i.
    for (uint32_t mip = 1; mip < m_StageResources.hiz.numMips; mip++) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::PushConstants(mesh_reg::HiZ::kPushC_DestDimensions, sizeof(uint32_t) * mesh_reg::HiZ::kPushCDwordCount),
            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::HiZ::kSRV_Source, m_StageResources.hiz.hizTexture,
                nvrhi::Format::RG32_FLOAT,
                nvrhi::TextureSubresourceSet(mip - 1, 1, 0, 1)),
            nvrhi::BindingSetItem::Texture_UAV(mesh_reg::HiZ::kUAV_Dest, m_StageResources.hiz.hizTexture,
                nvrhi::Format::RG32_FLOAT,
                nvrhi::TextureSubresourceSet(mip, 1, 0, 1)),
        };
        m_StageResources.hiz.buildBindingSets[mip] = device->createBindingSet(bsd, m_StageResources.hiz.buildBindingLayout);
    }

    m_StageResources.hiz.debugMipTextures.resize(m_StageResources.hiz.numMips);
    for (uint32_t mip = 0; mip < m_StageResources.hiz.numMips; mip++) {
        uint32_t mipW = std::max(1u, width  >> mip);
        uint32_t mipH = std::max(1u, height >> mip);
        m_StageResources.hiz.debugMipTextures[mip] = device->createTexture(nvrhi::TextureDesc()
            .setWidth(mipW).setHeight(mipH).setMipLevels(1)
            .setFormat(nvrhi::Format::RG32_FLOAT)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true)
            .setDebugName(("MeshHiZ_DebugMip" + std::to_string(mip)).c_str()));
    }

    // Rebuild downstream binding sets that reference the Hi-Z texture.
    _RebuildCullBindingSet();
    _RebuildDrawBindingSet();
    _RebuildDepthPrepassBindingSet();

    if (m_StageResources.sdsm.buildBindingLayout) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(mesh_reg::SDSM::kCB_Input, m_StageResources.sdsm.inputCB),
            nvrhi::BindingSetItem::Texture_SRV(mesh_reg::SDSM::kSRV_HiZ, m_StageResources.hiz.hizTexture),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(mesh_reg::SDSM::kUAV_CascadeOut, m_StageResources.sdsm.cascadeDataBuffer),
        };
        m_StageResources.sdsm.buildBindingSet = device->createBindingSet(bsd, m_StageResources.sdsm.buildBindingLayout);
    }
}

// ===========================================================================
// Depth prepass / Hi-Z build / SDSM
// ===========================================================================

void MeshShaderRenderPass::_CreateDepthPrepassPipelineIfNeeded() {
    if (m_StageResources.depthPrepass.pipeline || !m_StageResources.depthPrepass.framebuffer) return;

    nvrhi::MeshletPipelineDesc psoDesc;
    psoDesc.AS = m_StageResources.depthPrepass.amplificationShader;
    psoDesc.MS = m_StageResources.depthPrepass.meshShader;
    psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
    psoDesc.bindingLayouts = { m_StageResources.depthPrepass.bindingLayout };

    auto& rs = psoDesc.renderState;
    rs.depthStencilState.depthTestEnable  = true;
    rs.depthStencilState.depthWriteEnable = true;
#if XYLEM_USE_REVERSE_Z
    rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::GreaterOrEqual;
#else
    rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::LessOrEqual;
#endif
    rs.rasterState.cullMode = nvrhi::RasterCullMode::Back;

    m_StageResources.depthPrepass.pipeline = GetDevice()->createMeshletPipeline(
        psoDesc, m_StageResources.depthPrepass.framebuffer->getFramebufferInfo());

    if (m_StageResources.depthPrepass.pipeline && !m_StageResources.depthPrepass.dispatchMeshSignature) {
        ID3D12Device* d3dDevice = GetDevice()->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
        ID3D12RootSignature* rootSig = m_StageResources.depthPrepass.pipeline->getNativeObject(
            nvrhi::ObjectTypes::D3D12_RootSignature);
        if (d3dDevice && rootSig) {
            D3D12_INDIRECT_ARGUMENT_DESC args[2] = {};
            args[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
            args[0].Constant.RootParameterIndex      = mesh_reg::kPushC_RootParamIdx;
            args[0].Constant.DestOffsetIn32BitValues = 0;
            args[0].Constant.Num32BitValuesToSet     = 1;
            args[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;

            D3D12_COMMAND_SIGNATURE_DESC desc = {};
            desc.ByteStride       = sizeof(DispatchRecord);
            desc.NumArgumentDescs = 2;
            desc.pArgumentDescs   = args;

            HRESULT hr = d3dDevice->CreateCommandSignature(&desc, rootSig,
                IID_PPV_ARGS(&m_StageResources.depthPrepass.dispatchMeshSignature));
            if (FAILED(hr)) {
                log::error("MeshShaderRenderPass: CreateCommandSignature(depth) failed (0x%08x)", hr);
                m_StageResources.depthPrepass.dispatchMeshSignature = nullptr;
            }
        }
    }
}

void MeshShaderRenderPass::_RenderDepthPrepass() {
    _CreateDepthPrepassPipelineIfNeeded();

#if XYLEM_USE_REVERSE_Z
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, m_StageResources.depthPrepass.framebuffer, 0.f, 0);
#else
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, m_StageResources.depthPrepass.framebuffer, 1.f, 0);
#endif

    // Trees: ExecuteIndirect over last-frame mainDispatchArgsBuffer (not yet cleared).
    if (m_StageResources.depthPrepass.pipeline && m_StageResources.depthPrepass.dispatchMeshSignature) {
        nvrhi::MeshletState meshState;
        meshState.pipeline       = m_StageResources.depthPrepass.pipeline;
        meshState.framebuffer    = m_StageResources.depthPrepass.framebuffer;
        meshState.bindings       = { m_StageResources.depthPrepass.bindingSet };
        meshState.indirectParams = m_StageResources.cull.mainDispatchArgsBuffer;
        meshState.viewport.addViewportAndScissorRect(
            m_StageResources.depthPrepass.framebuffer->getFramebufferInfo().getViewport());
        m_CommandList->setMeshletState(meshState);

        uint32_t placeholder = 0;
        m_CommandList->setPushConstants(&placeholder, sizeof(uint32_t));

        auto* d3dList = static_cast<ID3D12GraphicsCommandList6*>(
            m_CommandList->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList));
        auto* argBuffer = static_cast<ID3D12Resource*>(
            m_StageResources.cull.mainDispatchArgsBuffer->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource));
        if (d3dList && argBuffer) {
            d3dList->ExecuteIndirect(
                m_StageResources.depthPrepass.dispatchMeshSignature.Get(),
                m_NumMainSlots, argBuffer, 0, nullptr, 0);
        }
    }

    // Terrain: direct draw (always visible).
    if (m_StageResources.sceneTerrain.indexCount > 0 && m_StageResources.depthPrepass.terrainVS
        && m_StageResources.depthPrepass.terrainBindingSet) {
        if (!m_StageResources.depthPrepass.terrainPipeline) {
            nvrhi::GraphicsPipelineDesc pso;
            pso.VS             = m_StageResources.depthPrepass.terrainVS;
            pso.inputLayout    = m_StageResources.depthPrepass.terrainInputLayout;
            pso.bindingLayouts = { m_StageResources.depthPrepass.terrainBindingLayout };
            pso.primType       = nvrhi::PrimitiveType::TriangleList;
            auto& rs = pso.renderState;
            rs.depthStencilState.depthTestEnable  = true;
            rs.depthStencilState.depthWriteEnable = true;
        #if XYLEM_USE_REVERSE_Z
            rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::GreaterOrEqual;
        #else
            rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::LessOrEqual;
        #endif
            rs.rasterState.setCullNone();
            m_StageResources.depthPrepass.terrainPipeline = GetDevice()->createGraphicsPipeline(
                pso, m_StageResources.depthPrepass.framebuffer->getFramebufferInfo());
        }

        nvrhi::GraphicsState state;
        state.pipeline    = m_StageResources.depthPrepass.terrainPipeline;
        state.framebuffer = m_StageResources.depthPrepass.framebuffer;
        state.bindings    = { m_StageResources.depthPrepass.terrainBindingSet };
        state.vertexBuffers = { { m_StageResources.sceneTerrain.vertexBuffer, 0, 0 } };
        state.indexBuffer   = { m_StageResources.sceneTerrain.indexBuffer, nvrhi::Format::R32_UINT, 0 };
        state.viewport.addViewportAndScissorRect(
            m_StageResources.depthPrepass.framebuffer->getFramebufferInfo().getViewport());
        m_CommandList->setGraphicsState(state);

        uint32_t zero = 0;
        m_CommandList->setPushConstants(&zero, sizeof(zero));
        m_CommandList->drawIndexed(nvrhi::DrawArguments().setVertexCount(m_StageResources.sceneTerrain.indexCount));
    }
}

void MeshShaderRenderPass::_BuildHiZMipChain() {
    auto desc = m_StageResources.depthPrepass.depthTexture->getDesc();
    uint32_t w = desc.width;
    uint32_t h = desc.height;

    {
        uint32_t dims[2] = { w, h };
        nvrhi::ComputeState cs;
        cs.pipeline = m_StageResources.hiz.copyPipeline;
        cs.bindings = { m_StageResources.hiz.buildBindingSets[0] };
        m_CommandList->setComputeState(cs);
        m_CommandList->setPushConstants(dims, sizeof(dims));
        m_CommandList->dispatch((w + 7) / 8, (h + 7) / 8, 1);
    }

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

    if (m_UI.showHiZ && !m_StageResources.hiz.debugMipTextures.empty()) {
        for (uint32_t mip = 0; mip < m_StageResources.hiz.numMips; mip++) {
            m_CommandList->copyTexture(
                m_StageResources.hiz.debugMipTextures[mip], nvrhi::TextureSlice(),
                m_StageResources.hiz.hizTexture,           nvrhi::TextureSlice().setMipLevel(mip));
        }
    }
}

void MeshShaderRenderPass::_ComputeRegionEnvelope(const dm::frustum& viewFrustum,
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

    if (!any) { outNearZ = 0.1f; outFarZ = 1.f; }
    outNearZ = dm::max(outNearZ, 0.1f);
    outFarZ  = dm::max(outFarZ,  outNearZ + 1.f);
}

void MeshShaderRenderPass::_RunSDSMBuildCascades(const dm::box3& sceneBbox,
                                                 float aspectRatio, float fovY,
                                                 float regionEnvelopeNear,
                                                 float regionEnvelopeFar) {
    shader_cb::SDSMCascadeBuildInput input{};

    dm::affine3 worldToLightAff = m_ViewHandler.worldToLight;
    dm::affine3 viewToWorldToLightAff =
        m_ViewHandler.view.GetInverseViewMatrix() * worldToLightAff;

    input.worldToLight       = dm::affineToHomogeneous(worldToLightAff);
    input.viewToWorldToLight = dm::affineToHomogeneous(viewToWorldToLightAff);

    dm::box3 sceneBboxLS = sceneBbox * worldToLightAff;
    input.sceneBboxMinLS = dm::float4(sceneBboxLS.m_mins, 0.f);
    input.sceneBboxMaxLS = dm::float4(sceneBboxLS.m_maxs, 0.f);

    float tanHalfFovY = std::tanf(fovY * 0.5f);
    input.tanHalfFovX = tanHalfFovY * aspectRatio;
    input.tanHalfFovY = tanHalfFovY;

    dm::float4x4 proj = m_ViewHandler.view.GetProjectionMatrix();
    input.projA = proj.m_data[2 * 4 + 2];
    input.projB = proj.m_data[3 * 4 + 2];

    input.regionEnvelopeNear = regionEnvelopeNear;
    input.regionEnvelopeFar  = regionEnvelopeFar;
    input.cameraNearPlane    = 0.1f;
    input.shadowRes          = k_ShadowRes;
    input.maxHiZMip          = (m_StageResources.hiz.numMips > 0) ? (m_StageResources.hiz.numMips - 1) : 0;
    input.pssmLambda         = m_UI.pssmLambda;

    m_CommandList->writeBuffer(m_StageResources.sdsm.inputCB, &input, sizeof(shader_cb::SDSMCascadeBuildInput));

    {
        nvrhi::ComputeState cs;
        cs.pipeline = m_StageResources.sdsm.buildPipeline;
        cs.bindings = { m_StageResources.sdsm.buildBindingSet };
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch(1, 1, 1);
    }

    // Copy SDSM results into main CB at CullConstantBufferEntry offsets.
    constexpr size_t kOffLightViewProj   = offsetof(Render::CullConstantBufferEntry, lightViewProj);
    constexpr size_t kOffCascadeSplits   = offsetof(Render::CullConstantBufferEntry, cascadeSplits);
    constexpr size_t kOffShadowCasterMin = offsetof(Render::CullConstantBufferEntry, shadowCasterMinLS);
    constexpr size_t kOffShadowCasterMax = offsetof(Render::CullConstantBufferEntry, shadowCasterMaxLS);

    constexpr size_t kSrcLightViewProj   = offsetof(shader_cb::SDSMCascadeBuildOutput, lightViewProj);
    constexpr size_t kSrcCascadeSplits   = offsetof(shader_cb::SDSMCascadeBuildOutput, cascadeSplits);
    constexpr size_t kSrcShadowCasterMin = offsetof(shader_cb::SDSMCascadeBuildOutput, shadowCasterMinLS);
    constexpr size_t kSrcShadowCasterMax = offsetof(shader_cb::SDSMCascadeBuildOutput, shadowCasterMaxLS);

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
// Sky
// ===========================================================================

void MeshShaderRenderPass::_RenderSkyPass(nvrhi::IFramebuffer* framebuffer) {
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

    auto& p = skyConstants.params;
    p.directionToLight   = dm::normalize(-m_Registry.getSunDirection());
    p.angularSizeOfLight = dm::radians(1.0f);
    p.lightColor         = dm::float3(100.f, 98.f, 90.f);
    p.glowSize           = dm::radians(5.f);
    p.skyColor           = dm::float3(0.017f, 0.037f, 0.065f);
    p.glowIntensity      = 0.1f;
    p.horizonColor       = dm::float3(0.050f, 0.070f, 0.092f);
    p.horizonSize        = dm::radians(30.f);
    p.groundColor        = dm::float3(0.062f, 0.059f, 0.055f);
    p.glowSharpness      = 4.f;
    p.directionUp        = dm::float3(0.f, 1.f, 0.f);

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
// Terrain color pass - used inside _RenderScenePass below
// ===========================================================================

void MeshShaderRenderPass::_RenderShadowPass() {}

void MeshShaderRenderPass::_RenderScenePass(nvrhi::IFramebuffer* framebuffer) {
    if (!m_StageResources.sceneTerrain.pipeline && m_StageResources.sceneTerrain.indexCount > 0) {
        nvrhi::GraphicsPipelineDesc pso;
        pso.VS             = m_StageResources.sceneTerrain.vertexShader;
        pso.PS             = m_StageResources.sceneTerrain.pixelShader;
        pso.inputLayout    = m_StageResources.sceneTerrain.inputLayout;
        pso.bindingLayouts = { m_StageResources.sceneTerrain.bindingLayout };
        pso.primType       = nvrhi::PrimitiveType::TriangleList;
        pso.renderState.rasterState.setCullNone();
        pso.renderState.depthStencilState.depthTestEnable  = true;
        pso.renderState.depthStencilState.depthWriteEnable = true;
    #if XYLEM_USE_REVERSE_Z
        pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
    #else
        pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    #endif
        m_StageResources.sceneTerrain.pipeline = GetDevice()->createGraphicsPipeline(
            pso, framebuffer->getFramebufferInfo());
    }

    if (m_StageResources.sceneTerrain.indexCount > 0 && m_StageResources.sceneTerrain.pipeline) {
        nvrhi::GraphicsState state;
        state.pipeline    = m_StageResources.sceneTerrain.pipeline;
        state.framebuffer = framebuffer;
        state.bindings    = { m_StageResources.sceneTerrain.bindingSet };
        state.vertexBuffers = { { m_StageResources.sceneTerrain.vertexBuffer, 0, 0 } };
        state.indexBuffer   = { m_StageResources.sceneTerrain.indexBuffer, nvrhi::Format::R32_UINT, 0 };
        state.viewport.addViewportAndScissorRect(framebuffer->getFramebufferInfo().getViewport());
        m_CommandList->setGraphicsState(state);
        m_CommandList->drawIndexed(
            nvrhi::DrawArguments().setVertexCount(m_StageResources.sceneTerrain.indexCount));
    }
}





