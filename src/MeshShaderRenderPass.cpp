#include "include/MeshShaderRenderPass.hpp"
#include "include/Render.hpp"
#include "include/Globals.hpp"
#include "include/Terrain.hpp"
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
#include <limits>

using namespace donut::math;
#include <donut/shaders/sky_cb.h>

using namespace Xylem;

namespace {
    // Matches SDSMBuildCascades.hlsl::SDSMInput
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

    constexpr size_t c_SDSMInputCBSize =
        (sizeof(SDSMInput) + (nvrhi::c_ConstantBufferOffsetSizeAlignment - 1))
        & ~(nvrhi::c_ConstantBufferOffsetSizeAlignment - 1);
} // namespace

namespace {

// Draw-pass register assignments — kept in lockstep with MeshShaderPass.hlsl.
constexpr uint32_t k_CB_Draw              = 0;
constexpr uint32_t k_PushC_Draw           = 1;   // root constant b1 (slotIdx)
constexpr uint32_t k_CB_ASCull            = 2;   // b2
constexpr uint32_t k_PushCBytes           = sizeof(uint32_t);
constexpr uint32_t k_SRV_Positions        = 0;
constexpr uint32_t k_SRV_Normals          = 1;
constexpr uint32_t k_SRV_Tangents         = 2;
constexpr uint32_t k_SRV_Bitangents       = 3;
constexpr uint32_t k_SRV_UVs              = 4;
constexpr uint32_t k_SRV_MVertIdx         = 5;
constexpr uint32_t k_SRV_MPrimIdx         = 6;
constexpr uint32_t k_SRV_Meshlets         = 7;
constexpr uint32_t k_SRV_AssetLods        = 8;
constexpr uint32_t k_SRV_VisBuf           = 9;
constexpr uint32_t k_SRV_SlotOffsets      = 10;
constexpr uint32_t k_SRV_SlotCounts       = 11;
constexpr uint32_t k_SRV_Instances        = 12;
constexpr uint32_t k_SRV_ASInvocsPerSlot  = 13;
constexpr uint32_t k_SRV_Diffuse          = 14;
constexpr uint32_t k_SRV_NormalMap        = 15;
constexpr uint32_t k_SRV_ShadowMap        = 16;
constexpr uint32_t k_SRV_HiZTex           = 17;
constexpr uint32_t k_Sampler_Draw         = 0;
constexpr uint32_t k_Sampler_Shadow       = 1;
constexpr uint32_t k_Sampler_HiZ          = 2;

// Cull-pass register assignments — match MeshCullCS.hlsl.
constexpr uint32_t k_CB_Cull                 = 0;
constexpr uint32_t k_SRV_CullRegion          = 0;
constexpr uint32_t k_SRV_CullInstance        = 1;
constexpr uint32_t k_SRV_CullMainSlotOffs    = 2;
constexpr uint32_t k_SRV_CullMainInvocs      = 3;
constexpr uint32_t k_SRV_CullShadowSlotOffs  = 4;
constexpr uint32_t k_SRV_CullShadowInvocs    = 5;
constexpr uint32_t k_SRV_CullHiZ             = 6;
constexpr uint32_t k_UAV_MainRegionVis       = 0;
constexpr uint32_t k_UAV_MainCount           = 1;
constexpr uint32_t k_UAV_MainVis             = 2;
constexpr uint32_t k_UAV_MainDispatch        = 3;
constexpr uint32_t k_UAV_ShadowCount         = 4;
constexpr uint32_t k_UAV_ShadowVis           = 5;
constexpr uint32_t k_UAV_ShadowDispatch      = 6;
constexpr uint32_t k_UAV_ShadowUnique        = 7;
constexpr uint32_t k_Sampler_Cull            = 0;

// Root parameter index of the push-constant block — NVRHI emits root-constants
// first (see d3d12 BindingLayout construction). Single binding layout -> index 0.
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
    m_Shared.constantBuffer = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(Render::c_CullConstantBufferSize)
            .setIsConstantBuffer(true)
            .setDebugName("MeshShaderPass_CB")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
    );
    if (!m_Shared.constantBuffer) return false;

    m_Shared.asCullCB = GetDevice()->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(c_ASCullCBSize)
            .setIsConstantBuffer(true)
            .setDebugName("MeshShaderPass_ASCullCB")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer)
    );
    if (!m_Shared.asCullCB) return false;

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
    m_Draw.shadowSampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setMinFilter(true)
            .setMagFilter(true)
            .setMipFilter(false)
            .setReductionType(nvrhi::SamplerReductionType::Comparison)
            .setAllAddressModes(nvrhi::SamplerAddressMode::Border)
            .setBorderColor(nvrhi::Color(1.f))
        );
    m_Draw.hizSampler = GetDevice()->createSampler(
        nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp)
            .setAllFilters(false));
    if (!m_Draw.sampler || !m_Draw.shadowSampler || !m_Draw.hizSampler) return false;

    nvrhi::BindingLayoutDesc bld;
    bld.visibility = nvrhi::ShaderType::All;
    bld.bindings = {
        nvrhi::BindingLayoutItem::PushConstants(k_PushC_Draw, k_PushCBytes),
        nvrhi::BindingLayoutItem::ConstantBuffer(k_CB_Draw),
        nvrhi::BindingLayoutItem::ConstantBuffer(k_CB_ASCull),

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
        nvrhi::BindingLayoutItem::Texture_SRV(k_SRV_ShadowMap),
        nvrhi::BindingLayoutItem::Texture_SRV(k_SRV_HiZTex),

        nvrhi::BindingLayoutItem::Sampler(k_Sampler_Draw),
        nvrhi::BindingLayoutItem::Sampler(k_Sampler_Shadow),
        nvrhi::BindingLayoutItem::Sampler(k_Sampler_HiZ),
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
    m_Cull.shadowCS = m_ShaderFactory->CreateShader("app/MeshCullCS.hlsl",
        "MeshCullShadow", nullptr, nvrhi::ShaderType::Compute);
    if (!m_Cull.mainCS || !m_Cull.regionCS || !m_Cull.shadowCS) {
        log::error("MeshShaderRenderPass: MeshCullCS compile failed");
        return false;
    }

    nvrhi::BindingLayoutDesc bld;
    bld.visibility = nvrhi::ShaderType::All;
    bld.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(k_CB_Cull),

        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_CullRegion),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_CullInstance),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_CullMainSlotOffs),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_CullMainInvocs),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_CullShadowSlotOffs),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(k_SRV_CullShadowInvocs),

        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(k_UAV_MainRegionVis),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(k_UAV_MainCount),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(k_UAV_MainVis),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(k_UAV_MainDispatch),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(k_UAV_ShadowCount),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(k_UAV_ShadowVis),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(k_UAV_ShadowDispatch),
        nvrhi::BindingLayoutItem::RawBuffer_UAV(k_UAV_ShadowUnique),

        nvrhi::BindingLayoutItem::Texture_SRV(k_SRV_CullHiZ),
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

    nvrhi::ComputePipelineDesc psoDescShadow;
    psoDescShadow.CS = m_Cull.shadowCS;
    psoDescShadow.bindingLayouts = { m_Cull.bindingLayout };
    m_Cull.shadowPipeline = GetDevice()->createComputePipeline(psoDescShadow);

    return m_Cull.mainPipeline && m_Cull.regionPipeline && m_Cull.shadowPipeline;
}

bool MeshShaderRenderPass::_InitShadowPass() {
    if (!m_ShaderFactory) return false;

    m_Shadow.amplificationShader = m_ShaderFactory->CreateShader("app/MeshShaderPass.hlsl",
        "shadow_as", nullptr, nvrhi::ShaderType::Amplification);
    m_Shadow.meshShader = m_ShaderFactory->CreateShader("app/MeshShaderPass.hlsl",
        "shadow_ms", nullptr, nvrhi::ShaderType::Mesh);
    if (!m_Shadow.amplificationShader || !m_Shadow.meshShader) {
        log::error("MeshShaderRenderPass: shadow AS/MS compile failed");
        return false;
    }

    // Shadow binding layout mirrors the subset the shadow AS/MS actually touch.
    // Uses the same t0..t13 register layout; the bindings point at shadow cull
    // buffers instead of main ones at bind-set build time.
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
    };
    m_Shadow.bindingLayout = GetDevice()->createBindingLayout(bld);
    if (!m_Shadow.bindingLayout) return false;

    // Texture2DArray shadow map — one slice per cascade.
    m_Shadow.depthTexture = GetDevice()->createTexture(nvrhi::TextureDesc()
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
    if (!m_Shadow.depthTexture) return false;

    for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
        nvrhi::FramebufferDesc fbd;
        fbd.setDepthAttachment(nvrhi::FramebufferAttachment()
            .setTexture(m_Shadow.depthTexture)
            .setArraySlice(c));
        m_Shadow.framebuffers[c] = GetDevice()->createFramebuffer(fbd);
        if (!m_Shadow.framebuffers[c]) return false;

        m_Shadow.debugTextures[c] = GetDevice()->createTexture(nvrhi::TextureDesc()
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
    const uint32_t numLods   = static_cast<uint32_t>(m_Registry.getLodSegments().size());
    const uint32_t numAssets = static_cast<uint32_t>(m_Registry.getAssets().size());
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

    // Main slots (asset × LOD)
    m_MainSlotOffsets.assign(m_NumMainSlots, 0);
    m_MainASInvocsPerSlot.assign(m_NumMainSlots, 0);
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
        }
    }
    m_MainVisBufferSize = std::max(1u, m_MainVisBufferSize);

    // Shadow slots (asset × cascade, LOD 0 only)
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

    // --- Main slot SRVs ---
    m_Cull.mainSlotOffsetBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumMainSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_MainSlotOffsetBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_Cull.mainSlotOffsetBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_Cull.mainSlotOffsetBuffer, m_MainSlotOffsets.data(),
        m_NumMainSlots * sizeof(uint32_t));
    cl->setPermanentBufferState(m_Cull.mainSlotOffsetBuffer, nvrhi::ResourceStates::ShaderResource);

    m_Cull.mainASInvocsPerSlotBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumMainSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_MainASInvocsPerSlotBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_Cull.mainASInvocsPerSlotBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_Cull.mainASInvocsPerSlotBuffer, m_MainASInvocsPerSlot.data(),
        m_NumMainSlots * sizeof(uint32_t));
    cl->setPermanentBufferState(m_Cull.mainASInvocsPerSlotBuffer, nvrhi::ResourceStates::ShaderResource);

    // --- Shadow slot SRVs ---
    m_Cull.shadowSlotOffsetBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumShadowSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_ShadowSlotOffsetBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_Cull.shadowSlotOffsetBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_Cull.shadowSlotOffsetBuffer, m_ShadowSlotOffsets.data(),
        m_NumShadowSlots * sizeof(uint32_t));
    cl->setPermanentBufferState(m_Cull.shadowSlotOffsetBuffer, nvrhi::ResourceStates::ShaderResource);

    m_Cull.shadowASInvocsPerSlotBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumShadowSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_ShadowASInvocsPerSlotBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest));
    cl->beginTrackingBufferState(m_Cull.shadowASInvocsPerSlotBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_Cull.shadowASInvocsPerSlotBuffer, m_ShadowASInvocsPerSlot.data(),
        m_NumShadowSlots * sizeof(uint32_t));
    cl->setPermanentBufferState(m_Cull.shadowASInvocsPerSlotBuffer, nvrhi::ResourceStates::ShaderResource);

    // --- UAVs ---
    m_Cull.regionVisibleBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(numRegions * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_RegionVisibleBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_Cull.mainCountBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumMainSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_MainCountBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_Cull.mainVisBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_MainVisBufferSize * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_MainVisBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_Cull.mainDispatchArgsBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumMainSlots * sizeof(DispatchRecord))
        .setDebugName("Mesh_MainDispatchArgsBuffer")
        .setIsDrawIndirectArgs(true)
        .setCanHaveUAVs(true)
        .setCanHaveRawViews(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_Cull.shadowCountBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumShadowSlots * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_ShadowCountBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_Cull.shadowVisBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_ShadowVisBufferSize * sizeof(uint32_t))
        .setStructStride(sizeof(uint32_t))
        .setDebugName("Mesh_ShadowVisBuffer")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_Cull.shadowDispatchArgsBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(m_NumShadowSlots * sizeof(DispatchRecord))
        .setDebugName("Mesh_ShadowDispatchArgsBuffer")
        .setIsDrawIndirectArgs(true)
        .setCanHaveUAVs(true)
        .setCanHaveRawViews(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    m_Cull.shadowUniqueCounter = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(sizeof(uint32_t))
        .setDebugName("Mesh_ShadowUniqueCounter")
        .setCanHaveUAVs(true)
        .setCanHaveRawViews(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));

    // The depth prepass consumes "last frame" mesh dispatch/count buffers before
    // the first per-frame cull reset runs. Seed those resources so frame zero is
    // a clean no-op instead of reading undefined indirect args.
    cl->clearBufferUInt(m_Cull.mainCountBuffer, 0);
    cl->clearBufferUInt(m_Cull.shadowCountBuffer, 0);
    cl->clearBufferUInt(m_Cull.shadowUniqueCounter, 0);
    cl->writeBuffer(m_Cull.mainDispatchArgsBuffer,
        m_MainDispatchArgsStaging.data(),
        m_MainDispatchArgsStaging.size() * sizeof(DispatchRecord));
    cl->writeBuffer(m_Cull.shadowDispatchArgsBuffer,
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
        nvrhi::BindingSetItem::ConstantBuffer(k_CB_Cull, m_Shared.constantBuffer,
            nvrhi::BufferRange(0, Render::c_CullConstantBufferSize)),

        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_CullRegion,         m_Cull.cullRegionDataBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_CullInstance,       m_Cull.cullDataBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_CullMainSlotOffs,   m_Cull.mainSlotOffsetBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_CullMainInvocs,     m_Cull.mainASInvocsPerSlotBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_CullShadowSlotOffs, m_Cull.shadowSlotOffsetBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_CullShadowInvocs,   m_Cull.shadowASInvocsPerSlotBuffer),

        nvrhi::BindingSetItem::StructuredBuffer_UAV(k_UAV_MainRegionVis,  m_Cull.regionVisibleBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(k_UAV_MainCount,      m_Cull.mainCountBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(k_UAV_MainVis,        m_Cull.mainVisBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(k_UAV_MainDispatch,          m_Cull.mainDispatchArgsBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(k_UAV_ShadowCount,    m_Cull.shadowCountBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(k_UAV_ShadowVis,      m_Cull.shadowVisBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(k_UAV_ShadowDispatch,        m_Cull.shadowDispatchArgsBuffer),
        nvrhi::BindingSetItem::RawBuffer_UAV(k_UAV_ShadowUnique,          m_Cull.shadowUniqueCounter),

        nvrhi::BindingSetItem::Texture_SRV(k_SRV_CullHiZ, m_HiZ.hizTexture),
        nvrhi::BindingSetItem::Sampler(k_Sampler_Cull, m_HiZ.pointSampler),
    };
    m_Cull.bindingSet = GetDevice()->createBindingSet(bsd, m_Cull.bindingLayout);
}

void MeshShaderRenderPass::_RebuildDrawBindingSet() {
    if (m_Draw.textureSets.empty()) return;
    if (!m_Shadow.depthTexture)     return;
    if (!m_HiZ.hizTexture)          return;

    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::PushConstants(k_PushC_Draw, k_PushCBytes),
        nvrhi::BindingSetItem::ConstantBuffer(k_CB_Draw, m_Shared.constantBuffer,
            nvrhi::BufferRange(0, Render::c_CullConstantBufferSize)),
        nvrhi::BindingSetItem::ConstantBuffer(k_CB_ASCull, m_Shared.asCullCB,
            nvrhi::BufferRange(0, c_ASCullCBSize)),

        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_Positions,     m_Meshlet.positions),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_Normals,       m_Meshlet.normals),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_Tangents,      m_Meshlet.tangents),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_Bitangents,    m_Meshlet.bitangents),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_UVs,           m_Meshlet.uvs),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_MVertIdx,      m_Meshlet.meshletVertIdx),
        nvrhi::BindingSetItem::RawBuffer_SRV(k_SRV_MPrimIdx,             m_Meshlet.meshletPrimIdx),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_Meshlets,      m_Meshlet.meshletDescs),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_AssetLods,     m_Meshlet.assetLodRanges),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_VisBuf,        m_Cull.mainVisBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_SlotOffsets,   m_Cull.mainSlotOffsetBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_SlotCounts,    m_Cull.mainCountBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_Instances,     m_Cull.persistentInstBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_ASInvocsPerSlot, m_Cull.mainASInvocsPerSlotBuffer),

        nvrhi::BindingSetItem::Texture_SRV(k_SRV_Diffuse,   m_Draw.textureSets[0].diffuse),
        nvrhi::BindingSetItem::Texture_SRV(k_SRV_NormalMap, m_Draw.textureSets[0].normalMap),
        nvrhi::BindingSetItem::Texture_SRV(k_SRV_ShadowMap, m_Shadow.depthTexture),
        nvrhi::BindingSetItem::Texture_SRV(k_SRV_HiZTex,    m_HiZ.hizTexture),

        nvrhi::BindingSetItem::Sampler(k_Sampler_Draw,   m_Draw.sampler),
        nvrhi::BindingSetItem::Sampler(k_Sampler_Shadow, m_Draw.shadowSampler),
        nvrhi::BindingSetItem::Sampler(k_Sampler_HiZ,    m_Draw.hizSampler),
    };
    m_Draw.bindingSet = GetDevice()->createBindingSet(bsd, m_Draw.bindingLayout);
}

void MeshShaderRenderPass::_RebuildShadowBindingSet() {
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
        // Shadow-specific redirect:
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_VisBuf,          m_Cull.shadowVisBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_SlotOffsets,     m_Cull.shadowSlotOffsetBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_SlotCounts,      m_Cull.shadowCountBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_Instances,       m_Cull.persistentInstBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(k_SRV_ASInvocsPerSlot, m_Cull.shadowASInvocsPerSlotBuffer),
    };
    m_Shadow.bindingSet = GetDevice()->createBindingSet(bsd, m_Shadow.bindingLayout);
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
    rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::Greater;
#else
    rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::Less;
#endif
    rs.rasterState.cullMode = nvrhi::RasterCullMode::Back;

    m_Draw.pipeline = GetDevice()->createMeshletPipeline(psoDesc, framebuffer->getFramebufferInfo());
}

void MeshShaderRenderPass::_CreateShadowPipelineIfNeeded() {
    if (m_Shadow.pipeline)            return;
    if (!m_Shadow.framebuffers[0])    return;

    nvrhi::MeshletPipelineDesc psoDesc;
    psoDesc.AS = m_Shadow.amplificationShader;
    psoDesc.MS = m_Shadow.meshShader;
    psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
    psoDesc.bindingLayouts = { m_Shadow.bindingLayout };

    auto& rs = psoDesc.renderState;
    rs.depthStencilState.depthTestEnable  = true;
    rs.depthStencilState.depthWriteEnable = true;
    // Light-space shadow maps use the standard D3D depth range/projection built by
    // ViewHandler::computeCascades, independent of the main camera's reverse-Z mode.
    rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::Less;
    rs.rasterState.cullMode = nvrhi::RasterCullMode::Front;
    rs.rasterState.depthBias = 2;
    rs.rasterState.slopeScaledDepthBias = 2.f;

    m_Shadow.pipeline = GetDevice()->createMeshletPipeline(
        psoDesc, m_Shadow.framebuffers[0]->getFramebufferInfo());
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
        args[0].Constant.RootParameterIndex      = k_PushC_RootParamIdx;
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

    makeSig(m_Draw.pipeline,   m_DispatchMeshSignature,         "main");
    makeSig(m_Shadow.pipeline, m_Shadow.dispatchMeshSignature,  "shadow");
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
    m_Draw.pipeline          = nullptr;
    m_Shadow.pipeline        = nullptr;
    m_DispatchMeshSignature  = nullptr;
    m_Shadow.dispatchMeshSignature = nullptr;

    m_DepthPrepass.pipeline        = nullptr;
    m_DepthPrepass.terrainPipeline = nullptr;
    m_DepthPrepass.depthTexture    = nullptr;
    m_DepthPrepass.framebuffer     = nullptr;
    m_DepthPrepass.bindingSet      = nullptr;
    m_DepthPrepass.dispatchMeshSignature = nullptr;

    m_TerrainPass.pipeline = nullptr;
    m_SkyPass.pipeline     = nullptr;

    m_HiZ.buildBindingSets.clear();
    m_HiZ.debugMipTextures.clear();
    m_HiZ.numMips = 0;
    m_UI.hizMipTextures.clear();
}

void MeshShaderRenderPass::Render(nvrhi::IFramebuffer* framebuffer) {
    // Expose shadow + Hi-Z debug textures to the UI.
    m_UI.shadowMapTexture = m_Shadow.depthTexture.Get();
    m_UI.shadowCascadeTextures.resize(Render::c_NumCascades);
    for (uint32_t c = 0; c < Render::c_NumCascades; c++)
        m_UI.shadowCascadeTextures[c] = m_Shadow.debugTextures[c].Get();
    m_UI.hizMipTextures.resize(m_HiZ.debugMipTextures.size());
    for (size_t i = 0; i < m_HiZ.debugMipTextures.size(); i++)
        m_UI.hizMipTextures[i] = m_HiZ.debugMipTextures[i].Get();

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

    _CreateMainPipelineIfNeeded(framebuffer);
    _CreateShadowPipelineIfNeeded();
    _EnsureDispatchMeshSignatures();

    // Compute scene bbox + max shadow distance (mirrors ComputeRenderPass).
    dm::box3 sceneBbox = dm::box3::empty();
    for (const auto& region : m_Registry.getRegions())
        sceneBbox |= region.cullBox;
    const auto* terrain = m_Registry.getTerrain();
    if (terrain) sceneBbox |= terrain->getBbox();

    const dm::float3& camPos = m_ViewHandler.camera.GetPosition();
    const dm::float3& camDir = m_ViewHandler.camera.GetDir();
    float maxShadowDist = 0.f;
    for (int i = 0; i < dm::box3::numCorners; i++) {
        dm::float3 corner = sceneBbox.getCorner(i);
        float cornerDir = dm::dot(corner - camPos, camDir);
        if (cornerDir > 0.0f) maxShadowDist = dm::max(maxShadowDist, cornerDir);
    }
    maxShadowDist = dm::max(maxShadowDist, 1.f);

    float aspectRatio = float(fbW) / float(fbH);
    m_ViewHandler.computeCascades(sceneBbox, m_Registry.getSunDirection(),
        0.1f, maxShadowDist, aspectRatio, dm::radians(60.f), k_ShadowRes, m_UI.pssmLambda);

    // Hi-Z bypass when camera looks steeply downward (mirrors ComputeRenderPass).
    float downwardness = -m_ViewHandler.camera.GetDir().y;
    bool hizActive = downwardness < m_UI.hizBypassAngle;
    m_UI.hizActiveThisFrame = hizActive;

    m_CommandList->open();

    // Make sure Hi-Z / depth-prepass resources match the back-buffer resolution.
    _EnsureHiZResources(fbW, fbH);

    // ----- 1. Fill CullConstantBufferEntry (CPU cascades seed; SDSM may overwrite) -----
    Render::CullConstantBufferEntry cb = {};
    cb.viewProj    = m_ViewHandler.view.GetViewProjectionMatrix();
    cb.viewMatrix  = dm::affineToHomogeneous(m_ViewHandler.view.GetViewMatrix());
    cb.sunLightDir = m_Registry.getSunDirection();
    cb.cascadeSplits = m_ViewHandler.cascadeSplitDistances;

    for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
        cb.lightViewProj[c] = m_ViewHandler.cascades[c].lightViewProj;
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

    cb.hizDimensions = dm::float2(static_cast<float>(fbW), static_cast<float>(fbH));
    cb.maxHiZMip     = static_cast<float>((m_HiZ.numMips > 0) ? (m_HiZ.numMips - 1) : 0);
    cb.hizEnabled    = hizActive ? 1u : 0u;

    m_CommandList->writeBuffer(m_Shared.constantBuffer, &cb, Render::c_CullConstantBufferSize);

    // ----- 1b. Fill ASCullCBEntry — hizEnabled=0 for the depth prepass (Hi-Z not yet built) -----
    ASCullCBEntry asCB = {};
    asCB.cameraPos         = camPos;
    asCB.hizEnabled        = 0u;
    asCB.hizDimensions     = dm::float2(static_cast<float>(fbW), static_cast<float>(fbH));
    asCB.maxHiZMip         = static_cast<float>((m_HiZ.numMips > 0) ? (m_HiZ.numMips - 1) : 0);
    asCB.asConeCullEnabled = 1u;
    m_CommandList->writeBuffer(m_Shared.asCullCB, &asCB, sizeof(ASCullCBEntry));

    // ----- 2. Depth prepass + Hi-Z build + SDSM (skip when bypassed) -----
    // Reads LAST frame's m_Cull.mainDispatchArgsBuffer — must happen BEFORE we
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
        _RunSDSMBuildCascades(sceneBbox, aspectRatio, dm::radians(60.f),
                              regionNear, regionFar);
        m_CommandList->endMarker();
    }

    // Re-write ASCullCB now that Hi-Z is populated — main_as uses it during color pass.
    asCB.hizEnabled = hizActive ? 1u : 0u;
    m_CommandList->writeBuffer(m_Shared.asCullCB, &asCB, sizeof(ASCullCBEntry));

    // ----- 3. Per-frame cull resets -----
    m_CommandList->clearBufferUInt(m_Cull.mainCountBuffer,     0);
    m_CommandList->clearBufferUInt(m_Cull.shadowCountBuffer,   0);
    m_CommandList->clearBufferUInt(m_Cull.shadowUniqueCounter, 0);

    m_CommandList->writeBuffer(m_Cull.mainDispatchArgsBuffer,
        m_MainDispatchArgsStaging.data(),
        m_MainDispatchArgsStaging.size() * sizeof(DispatchRecord));
    m_CommandList->writeBuffer(m_Cull.shadowDispatchArgsBuffer,
        m_ShadowDispatchArgsStaging.data(),
        m_ShadowDispatchArgsStaging.size() * sizeof(DispatchRecord));

    // ----- 4. GPU cull dispatches -----
    m_CommandList->beginMarker("MeshCull");
    {
        nvrhi::ComputeState cs;
        cs.bindings = { m_Cull.bindingSet };

        m_CommandList->beginMarker("RegionDispatch");
        cs.pipeline = m_Cull.regionPipeline;
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch((cb.numRegions + 63) / 64, 1, 1);
        m_CommandList->endMarker();

        m_CommandList->beginMarker("MainDispatch");
        cs.pipeline = m_Cull.mainPipeline;
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch((m_TotalCapacity + 255) / 256, 1, 1);
        m_CommandList->endMarker();

        m_CommandList->beginMarker("ShadowDispatch");
        cs.pipeline = m_Cull.shadowPipeline;
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
                                  m_Cull.mainCountBuffer, 0,
                                  m_ReadbackMainEntries * sizeof(uint32_t));
        offset += m_ReadbackMainEntries * sizeof(uint32_t);
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], offset,
                                  m_Cull.shadowCountBuffer, 0,
                                  m_ReadbackShadowEntries * sizeof(uint32_t));
        offset += m_ReadbackShadowEntries * sizeof(uint32_t);
        m_CommandList->copyBuffer(m_ReadbackBuffers[ringSlot], offset,
                                  m_Cull.shadowUniqueCounter, 0,
                                  sizeof(uint32_t));
    }

    // ----- 4. Shadow pass: 4 cascades via ExecuteIndirect(DISPATCH_MESH) -----
    if (m_Shadow.pipeline && m_Shadow.dispatchMeshSignature) {
        m_CommandList->beginMarker("ShadowMeshDraw");

        // Clear all cascades.
        for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
            nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, m_Shadow.framebuffers[c], 1.f, 0);
        }

        // Shared shadow meshlet state; re-bind the framebuffer per cascade slice and
        // ExecuteIndirect only for that cascade's interleaved asset slots.
        const uint32_t numAssets = static_cast<uint32_t>(m_Registry.getAssets().size());

        // The shadow slot layout is interleaved as slot = asset*NUM_CASCADES + cascade.

        for (uint32_t c = 0; c < Render::c_NumCascades; c++) {
            nvrhi::MeshletState ms;
            ms.pipeline       = m_Shadow.pipeline;
            ms.framebuffer    = m_Shadow.framebuffers[c];
            ms.bindings       = { m_Shadow.bindingSet };
            ms.indirectParams = m_Cull.shadowDispatchArgsBuffer;
            ms.viewport.addViewportAndScissorRect(
                nvrhi::Viewport(float(k_ShadowRes), float(k_ShadowRes)));
            m_CommandList->setMeshletState(ms);

            uint32_t placeholder = 0;
            m_CommandList->setPushConstants(&placeholder, sizeof(uint32_t));

            auto* d3dList = static_cast<ID3D12GraphicsCommandList6*>(
                m_CommandList->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList));
            auto* argBuffer = static_cast<ID3D12Resource*>(
                m_Cull.shadowDispatchArgsBuffer->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource));

            if (d3dList && argBuffer) {
                for (uint32_t ai = 0; ai < numAssets; ai++) {
                    const uint32_t slot = ai * Render::c_NumCascades + c;
                    d3dList->ExecuteIndirect(
                        m_Shadow.dispatchMeshSignature.Get(),
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
                m_Shadow.debugTextures[c], nvrhi::TextureSlice(),
                m_Shadow.depthTexture, nvrhi::TextureSlice().setArraySlice(c));
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
        meshState.pipeline       = m_Draw.pipeline;
        meshState.framebuffer    = framebuffer;
        meshState.bindings       = { m_Draw.bindingSet };
        meshState.indirectParams = m_Cull.mainDispatchArgsBuffer;
        meshState.viewport.addViewportAndScissorRect(fbInfo.getViewport());
        m_CommandList->setMeshletState(meshState);

        uint32_t placeholder = 0;
        m_CommandList->setPushConstants(&placeholder, sizeof(uint32_t));

        if (m_DispatchMeshSignature) {
            auto* d3dList = static_cast<ID3D12GraphicsCommandList6*>(
                m_CommandList->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList));
            auto* argBuffer = static_cast<ID3D12Resource*>(
                m_Cull.mainDispatchArgsBuffer->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource));

            if (d3dList && argBuffer) {
                d3dList->ExecuteIndirect(
                    m_DispatchMeshSignature.Get(),
                    m_NumMainSlots,
                    argBuffer,
                    0,
                    nullptr,
                    0);
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
    m_DepthPrepass.amplificationShader = m_Draw.amplificationShader;
    m_DepthPrepass.meshShader = m_ShaderFactory->CreateShader("app/MeshShaderPass.hlsl",
        "depth_ms", nullptr, nvrhi::ShaderType::Mesh);
    if (!m_DepthPrepass.meshShader) {
        log::error("MeshShaderRenderPass: depth_ms compile failed");
        return false;
    }

    // Tree depth pipeline shares the main draw binding layout — same SRVs/CBs.
    m_DepthPrepass.bindingLayout = m_Draw.bindingLayout;

    // Terrain depth path uses the existing DepthPrepass.hlsl terrain_vs (CB only).
    m_DepthPrepass.terrainVS = m_ShaderFactory->CreateShader("app/DepthPrepass.hlsl",
        "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    if (!m_DepthPrepass.terrainVS) {
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
    m_DepthPrepass.terrainInputLayout = GetDevice()->createInputLayout(
        terrainAttrs, uint32_t(std::size(terrainAttrs)), m_DepthPrepass.terrainVS);
    if (!m_DepthPrepass.terrainInputLayout) return false;

    // Terrain depth binding layout: CB(0) only — DepthPrepass.terrain_vs ignores the
    // vis/inst/slot SRVs declared in that file.
    nvrhi::BindingLayoutDesc terrainBLD;
    terrainBLD.visibility = nvrhi::ShaderType::All;
    terrainBLD.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(0),
        nvrhi::BindingLayoutItem::PushConstants(1, sizeof(uint32_t)),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2),
    };
    m_DepthPrepass.terrainBindingLayout = GetDevice()->createBindingLayout(terrainBLD);
    return m_DepthPrepass.terrainBindingLayout != nullptr;
}

bool MeshShaderRenderPass::_InitHiZShaders() {
    auto device = GetDevice();

    m_HiZ.copyCS = m_ShaderFactory->CreateShader("app/HiZBuild.hlsl",
        "HiZCopy", nullptr, nvrhi::ShaderType::Compute);
    m_HiZ.buildCS = m_ShaderFactory->CreateShader("app/HiZBuild.hlsl",
        "HiZDownsample", nullptr, nvrhi::ShaderType::Compute);
    if (!m_HiZ.copyCS || !m_HiZ.buildCS) return false;

    nvrhi::BindingLayoutDesc bld;
    bld.visibility = nvrhi::ShaderType::Compute;
    bld.bindings = {
        nvrhi::BindingLayoutItem::PushConstants(0, sizeof(uint32_t) * 2),
        nvrhi::BindingLayoutItem::Texture_SRV(0),
        nvrhi::BindingLayoutItem::Texture_UAV(0),
    };
    m_HiZ.buildBindingLayout = device->createBindingLayout(bld);
    if (!m_HiZ.buildBindingLayout) return false;

    nvrhi::ComputePipelineDesc copyPso;
    copyPso.CS = m_HiZ.copyCS;
    copyPso.bindingLayouts = { m_HiZ.buildBindingLayout };
    m_HiZ.copyPipeline = device->createComputePipeline(copyPso);

    nvrhi::ComputePipelineDesc buildPso;
    buildPso.CS = m_HiZ.buildCS;
    buildPso.bindingLayouts = { m_HiZ.buildBindingLayout };
    m_HiZ.buildPipeline = device->createComputePipeline(buildPso);
    if (!m_HiZ.copyPipeline || !m_HiZ.buildPipeline) return false;

    m_HiZ.pointSampler = device->createSampler(nvrhi::SamplerDesc()
        .setAllFilters(false)
        .setAllAddressModes(nvrhi::SamplerAddressMode::Clamp));
    if (!m_HiZ.pointSampler) return false;

    // 1x1 placeholder so binding sets resolve before first _EnsureHiZResources.
    m_HiZ.hizTexture = device->createTexture(nvrhi::TextureDesc()
        .setWidth(1).setHeight(1).setMipLevels(1)
        .setFormat(nvrhi::Format::RG32_FLOAT)
        .setIsUAV(true)
        .setInitialState(nvrhi::ResourceStates::ShaderResource)
        .setKeepInitialState(true)
        .setDebugName("MeshHiZ_Placeholder"));
    m_HiZ.numMips = 1;
    return true;
}

bool MeshShaderRenderPass::_InitSDSMPass() {
    auto device = GetDevice();

    m_SDSM.buildCS = m_ShaderFactory->CreateShader("app/SDSMBuildCascades.hlsl",
        "BuildCascades", nullptr, nvrhi::ShaderType::Compute);
    if (!m_SDSM.buildCS) return false;

    nvrhi::BindingLayoutDesc bld;
    bld.visibility = nvrhi::ShaderType::Compute;
    bld.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(0),
        nvrhi::BindingLayoutItem::Texture_SRV(0),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0),
    };
    m_SDSM.buildBindingLayout = device->createBindingLayout(bld);
    if (!m_SDSM.buildBindingLayout) return false;

    nvrhi::ComputePipelineDesc pso;
    pso.CS = m_SDSM.buildCS;
    pso.bindingLayouts = { m_SDSM.buildBindingLayout };
    m_SDSM.buildPipeline = device->createComputePipeline(pso);
    if (!m_SDSM.buildPipeline) return false;

    m_SDSM.inputCB = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(c_SDSMInputCBSize)
        .setIsConstantBuffer(true)
        .setDebugName("MeshSDSM_InputCB")
        .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer));
    if (!m_SDSM.inputCB) return false;

    m_SDSM.cascadeDataBuffer = device->createBuffer(nvrhi::BufferDesc()
        .setByteSize(sizeof(SDSMCascadeOut))
        .setStructStride(sizeof(SDSMCascadeOut))
        .setDebugName("MeshSDSM_CascadeData")
        .setCanHaveUAVs(true)
        .enableAutomaticStateTracking(nvrhi::ResourceStates::UnorderedAccess));
    if (!m_SDSM.cascadeDataBuffer) return false;

    // Initial binding set references placeholder Hi-Z; rebuilt in _EnsureHiZResources.
    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_SDSM.inputCB),
        nvrhi::BindingSetItem::Texture_SRV(0, m_HiZ.hizTexture),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_SDSM.cascadeDataBuffer),
    };
    m_SDSM.buildBindingSet = device->createBindingSet(bsd, m_SDSM.buildBindingLayout);
    return m_SDSM.buildBindingSet != nullptr;
}

bool MeshShaderRenderPass::_InitTerrainPass(nvrhi::ICommandList* initCL) {
    const auto* terrain = m_Registry.getTerrain();
    if (!terrain) return true;

    const auto& verts   = terrain->getVertices();
    const auto& indices = terrain->getIndices();
    if (verts.empty() || indices.empty()) return true;

    m_TerrainPass.indexCount = static_cast<uint32_t>(indices.size());

    m_TerrainPass.vertexShader = m_ShaderFactory->CreateShader(
        "app/terrain_compute.hlsl", "terrain_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_TerrainPass.pixelShader = m_ShaderFactory->CreateShader(
        "app/terrain_compute.hlsl", "terrain_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_TerrainPass.vertexShader || !m_TerrainPass.pixelShader) return false;

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
    m_TerrainPass.inputLayout = GetDevice()->createInputLayout(
        attrs, uint32_t(std::size(attrs)), m_TerrainPass.vertexShader);
    if (!m_TerrainPass.inputLayout) return false;

    nvrhi::BufferDesc vbDesc;
    vbDesc.isVertexBuffer = true;
    vbDesc.byteSize       = verts.size() * sizeof(Scene::TerrainVertex);
    vbDesc.debugName      = "MeshTerrainVB";
    vbDesc.initialState   = nvrhi::ResourceStates::CopyDest;
    m_TerrainPass.vertexBuffer = GetDevice()->createBuffer(vbDesc);
    initCL->beginTrackingBufferState(m_TerrainPass.vertexBuffer, nvrhi::ResourceStates::CopyDest);
    initCL->writeBuffer(m_TerrainPass.vertexBuffer, verts.data(), vbDesc.byteSize);
    initCL->setPermanentBufferState(m_TerrainPass.vertexBuffer, nvrhi::ResourceStates::VertexBuffer);

    nvrhi::BufferDesc ibDesc;
    ibDesc.isIndexBuffer = true;
    ibDesc.byteSize      = indices.size() * sizeof(uint32_t);
    ibDesc.debugName     = "MeshTerrainIB";
    ibDesc.initialState  = nvrhi::ResourceStates::CopyDest;
    m_TerrainPass.indexBuffer = GetDevice()->createBuffer(ibDesc);
    initCL->beginTrackingBufferState(m_TerrainPass.indexBuffer, nvrhi::ResourceStates::CopyDest);
    initCL->writeBuffer(m_TerrainPass.indexBuffer, indices.data(), ibDesc.byteSize);
    initCL->setPermanentBufferState(m_TerrainPass.indexBuffer, nvrhi::ResourceStates::IndexBuffer);

    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_Shared.constantBuffer,
            nvrhi::BufferRange(0, Render::c_CullConstantBufferSize)),
        nvrhi::BindingSetItem::Sampler(0, m_Draw.shadowSampler),
        nvrhi::BindingSetItem::Texture_SRV(0, m_Shadow.depthTexture),
    };
    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0,
            bsd, m_TerrainPass.bindingLayout, m_TerrainPass.bindingSet))
        return false;

    return true;
}

bool MeshShaderRenderPass::_InitSkyPass() {
    m_SkyPass.vertexShader = m_ShaderFactory->CreateShader(
        "app/sky.hlsl", "sky_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_SkyPass.pixelShader = m_ShaderFactory->CreateShader(
        "app/sky.hlsl", "sky_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_SkyPass.vertexShader || !m_SkyPass.pixelShader) return false;

    nvrhi::BufferDesc cbDesc;
    cbDesc.byteSize         = sizeof(SkyConstants);
    cbDesc.isConstantBuffer = true;
    cbDesc.isVolatile       = true;
    cbDesc.maxVersions      = 16;
    cbDesc.debugName        = "MeshSkyConstants";
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

// ===========================================================================
// Binding-set rebuild for depth prepass
// ===========================================================================

void MeshShaderRenderPass::_RebuildDepthPrepassBindingSet() {
    // Tree meshlet depth prepass shares the main Draw binding set.
    m_DepthPrepass.bindingSet = m_Draw.bindingSet;

    // Terrain depth prepass: CB + dummy vis/inst/slotOffsets (not read by terrain_vs).
    if (!m_DepthPrepass.terrainBindingLayout) return;
    nvrhi::BindingSetDesc bsd;
    bsd.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_Shared.constantBuffer,
            nvrhi::BufferRange(0, Render::c_CullConstantBufferSize)),
        nvrhi::BindingSetItem::PushConstants(1, sizeof(uint32_t)),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_Cull.mainVisBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_Cull.persistentInstBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_Cull.mainSlotOffsetBuffer),
    };
    m_DepthPrepass.terrainBindingSet = GetDevice()->createBindingSet(
        bsd, m_DepthPrepass.terrainBindingLayout);
}

// ===========================================================================
// Hi-Z resource (re)creation — called on back-buffer resize
// ===========================================================================

void MeshShaderRenderPass::_EnsureHiZResources(uint32_t width, uint32_t height) {
    if (m_DepthPrepass.depthTexture) {
        auto desc = m_DepthPrepass.depthTexture->getDesc();
        if (desc.width == width && desc.height == height) return;
    }

    auto device = GetDevice();

    m_DepthPrepass.depthTexture = device->createTexture(nvrhi::TextureDesc()
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
    m_DepthPrepass.framebuffer = device->createFramebuffer(
        nvrhi::FramebufferDesc().setDepthAttachment(m_DepthPrepass.depthTexture));

    m_DepthPrepass.pipeline         = nullptr;
    m_DepthPrepass.terrainPipeline  = nullptr;
    m_DepthPrepass.dispatchMeshSignature = nullptr;

    m_HiZ.numMips = static_cast<uint32_t>(
        std::floor(std::log2(std::max(width, height)))) + 1;
    m_HiZ.hizTexture = device->createTexture(nvrhi::TextureDesc()
        .setWidth(width).setHeight(height)
        .setMipLevels(m_HiZ.numMips)
        .setFormat(nvrhi::Format::RG32_FLOAT)
        .setIsUAV(true)
        .setInitialState(nvrhi::ResourceStates::ShaderResource)
        .setKeepInitialState(true)
        .setDebugName("MeshHiZTexture"));

    m_HiZ.buildBindingSets.resize(m_HiZ.numMips);

    // Mip 0 — copy depth prepass (D32 -> R32_FLOAT view) into RG32 mip 0.
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

    // Mips 1..N-1 — downsample i-1 -> i.
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

    m_HiZ.debugMipTextures.resize(m_HiZ.numMips);
    for (uint32_t mip = 0; mip < m_HiZ.numMips; mip++) {
        uint32_t mipW = std::max(1u, width  >> mip);
        uint32_t mipH = std::max(1u, height >> mip);
        m_HiZ.debugMipTextures[mip] = device->createTexture(nvrhi::TextureDesc()
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
// Depth prepass / Hi-Z build / SDSM
// ===========================================================================

void MeshShaderRenderPass::_CreateDepthPrepassPipelineIfNeeded() {
    if (m_DepthPrepass.pipeline || !m_DepthPrepass.framebuffer) return;

    nvrhi::MeshletPipelineDesc psoDesc;
    psoDesc.AS = m_DepthPrepass.amplificationShader;
    psoDesc.MS = m_DepthPrepass.meshShader;
    psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
    psoDesc.bindingLayouts = { m_DepthPrepass.bindingLayout };

    auto& rs = psoDesc.renderState;
    rs.depthStencilState.depthTestEnable  = true;
    rs.depthStencilState.depthWriteEnable = true;
#if XYLEM_USE_REVERSE_Z
    rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::GreaterOrEqual;
#else
    rs.depthStencilState.depthFunc = nvrhi::ComparisonFunc::LessOrEqual;
#endif
    rs.rasterState.cullMode = nvrhi::RasterCullMode::Back;

    m_DepthPrepass.pipeline = GetDevice()->createMeshletPipeline(
        psoDesc, m_DepthPrepass.framebuffer->getFramebufferInfo());

    if (m_DepthPrepass.pipeline && !m_DepthPrepass.dispatchMeshSignature) {
        ID3D12Device* d3dDevice = GetDevice()->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
        ID3D12RootSignature* rootSig = m_DepthPrepass.pipeline->getNativeObject(
            nvrhi::ObjectTypes::D3D12_RootSignature);
        if (d3dDevice && rootSig) {
            D3D12_INDIRECT_ARGUMENT_DESC args[2] = {};
            args[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
            args[0].Constant.RootParameterIndex      = k_PushC_RootParamIdx;
            args[0].Constant.DestOffsetIn32BitValues = 0;
            args[0].Constant.Num32BitValuesToSet     = 1;
            args[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;

            D3D12_COMMAND_SIGNATURE_DESC desc = {};
            desc.ByteStride       = sizeof(DispatchRecord);
            desc.NumArgumentDescs = 2;
            desc.pArgumentDescs   = args;

            HRESULT hr = d3dDevice->CreateCommandSignature(&desc, rootSig,
                IID_PPV_ARGS(&m_DepthPrepass.dispatchMeshSignature));
            if (FAILED(hr)) {
                log::error("MeshShaderRenderPass: CreateCommandSignature(depth) failed (0x%08x)", hr);
                m_DepthPrepass.dispatchMeshSignature = nullptr;
            }
        }
    }
}

void MeshShaderRenderPass::_RenderDepthPrepass() {
    _CreateDepthPrepassPipelineIfNeeded();

#if XYLEM_USE_REVERSE_Z
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, m_DepthPrepass.framebuffer, 0.f, 0);
#else
    nvrhi::utils::ClearDepthStencilAttachment(m_CommandList, m_DepthPrepass.framebuffer, 1.f, 0);
#endif

    // Trees: ExecuteIndirect over last-frame mainDispatchArgsBuffer (not yet cleared).
    if (m_DepthPrepass.pipeline && m_DepthPrepass.dispatchMeshSignature) {
        nvrhi::MeshletState meshState;
        meshState.pipeline       = m_DepthPrepass.pipeline;
        meshState.framebuffer    = m_DepthPrepass.framebuffer;
        meshState.bindings       = { m_DepthPrepass.bindingSet };
        meshState.indirectParams = m_Cull.mainDispatchArgsBuffer;
        meshState.viewport.addViewportAndScissorRect(
            m_DepthPrepass.framebuffer->getFramebufferInfo().getViewport());
        m_CommandList->setMeshletState(meshState);

        uint32_t placeholder = 0;
        m_CommandList->setPushConstants(&placeholder, sizeof(uint32_t));

        auto* d3dList = static_cast<ID3D12GraphicsCommandList6*>(
            m_CommandList->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList));
        auto* argBuffer = static_cast<ID3D12Resource*>(
            m_Cull.mainDispatchArgsBuffer->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource));
        if (d3dList && argBuffer) {
            d3dList->ExecuteIndirect(
                m_DepthPrepass.dispatchMeshSignature.Get(),
                m_NumMainSlots, argBuffer, 0, nullptr, 0);
        }
    }

    // Terrain: direct draw (always visible).
    if (m_TerrainPass.indexCount > 0 && m_DepthPrepass.terrainVS
        && m_DepthPrepass.terrainBindingSet) {
        if (!m_DepthPrepass.terrainPipeline) {
            nvrhi::GraphicsPipelineDesc pso;
            pso.VS             = m_DepthPrepass.terrainVS;
            pso.inputLayout    = m_DepthPrepass.terrainInputLayout;
            pso.bindingLayouts = { m_DepthPrepass.terrainBindingLayout };
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
            m_DepthPrepass.terrainPipeline = GetDevice()->createGraphicsPipeline(
                pso, m_DepthPrepass.framebuffer->getFramebufferInfo());
        }

        nvrhi::GraphicsState state;
        state.pipeline    = m_DepthPrepass.terrainPipeline;
        state.framebuffer = m_DepthPrepass.framebuffer;
        state.bindings    = { m_DepthPrepass.terrainBindingSet };
        state.vertexBuffers = { { m_TerrainPass.vertexBuffer, 0, 0 } };
        state.indexBuffer   = { m_TerrainPass.indexBuffer, nvrhi::Format::R32_UINT, 0 };
        state.viewport.addViewportAndScissorRect(
            m_DepthPrepass.framebuffer->getFramebufferInfo().getViewport());
        m_CommandList->setGraphicsState(state);

        uint32_t zero = 0;
        m_CommandList->setPushConstants(&zero, sizeof(zero));
        m_CommandList->drawIndexed(nvrhi::DrawArguments().setVertexCount(m_TerrainPass.indexCount));
    }
}

void MeshShaderRenderPass::_BuildHiZMipChain() {
    auto desc = m_DepthPrepass.depthTexture->getDesc();
    uint32_t w = desc.width;
    uint32_t h = desc.height;

    {
        uint32_t dims[2] = { w, h };
        nvrhi::ComputeState cs;
        cs.pipeline = m_HiZ.copyPipeline;
        cs.bindings = { m_HiZ.buildBindingSets[0] };
        m_CommandList->setComputeState(cs);
        m_CommandList->setPushConstants(dims, sizeof(dims));
        m_CommandList->dispatch((w + 7) / 8, (h + 7) / 8, 1);
    }

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

    if (m_UI.showHiZ && !m_HiZ.debugMipTextures.empty()) {
        for (uint32_t mip = 0; mip < m_HiZ.numMips; mip++) {
            m_CommandList->copyTexture(
                m_HiZ.debugMipTextures[mip], nvrhi::TextureSlice(),
                m_HiZ.hizTexture,           nvrhi::TextureSlice().setMipLevel(mip));
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
    SDSMInput input{};

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
    input.maxHiZMip          = (m_HiZ.numMips > 0) ? (m_HiZ.numMips - 1) : 0;
    input.pssmLambda         = m_UI.pssmLambda;

    m_CommandList->writeBuffer(m_SDSM.inputCB, &input, sizeof(SDSMInput));

    {
        nvrhi::ComputeState cs;
        cs.pipeline = m_SDSM.buildPipeline;
        cs.bindings = { m_SDSM.buildBindingSet };
        m_CommandList->setComputeState(cs);
        m_CommandList->dispatch(1, 1, 1);
    }

    // Copy SDSM results into main CB at CullConstantBufferEntry offsets.
    constexpr size_t kOffLightViewProj   = offsetof(Render::CullConstantBufferEntry, lightViewProj);
    constexpr size_t kOffCascadeSplits   = offsetof(Render::CullConstantBufferEntry, cascadeSplits);
    constexpr size_t kOffShadowCasterMin = offsetof(Render::CullConstantBufferEntry, shadowCasterMinLS);
    constexpr size_t kOffShadowCasterMax = offsetof(Render::CullConstantBufferEntry, shadowCasterMaxLS);

    constexpr size_t kSrcLightViewProj   = offsetof(SDSMCascadeOut, lightViewProj);
    constexpr size_t kSrcCascadeSplits   = offsetof(SDSMCascadeOut, cascadeSplits);
    constexpr size_t kSrcShadowCasterMin = offsetof(SDSMCascadeOut, shadowCasterMinLS);
    constexpr size_t kSrcShadowCasterMax = offsetof(SDSMCascadeOut, shadowCasterMaxLS);

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
// Sky
// ===========================================================================

void MeshShaderRenderPass::_RenderSkyPass(nvrhi::IFramebuffer* framebuffer) {
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
// Terrain color pass — used inside _RenderScenePass below
// ===========================================================================

void MeshShaderRenderPass::_RenderShadowPass() {}

void MeshShaderRenderPass::_RenderScenePass(nvrhi::IFramebuffer* framebuffer) {
    if (!m_TerrainPass.pipeline && m_TerrainPass.indexCount > 0) {
        nvrhi::GraphicsPipelineDesc pso;
        pso.VS             = m_TerrainPass.vertexShader;
        pso.PS             = m_TerrainPass.pixelShader;
        pso.inputLayout    = m_TerrainPass.inputLayout;
        pso.bindingLayouts = { m_TerrainPass.bindingLayout };
        pso.primType       = nvrhi::PrimitiveType::TriangleList;
        pso.renderState.rasterState.setCullNone();
        pso.renderState.depthStencilState.depthTestEnable  = true;
        pso.renderState.depthStencilState.depthWriteEnable = true;
    #if XYLEM_USE_REVERSE_Z
        pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Greater);
    #else
        pso.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    #endif
        m_TerrainPass.pipeline = GetDevice()->createGraphicsPipeline(
            pso, framebuffer->getFramebufferInfo());
    }

    if (m_TerrainPass.indexCount > 0 && m_TerrainPass.pipeline) {
        nvrhi::GraphicsState state;
        state.pipeline    = m_TerrainPass.pipeline;
        state.framebuffer = framebuffer;
        state.bindings    = { m_TerrainPass.bindingSet };
        state.vertexBuffers = { { m_TerrainPass.vertexBuffer, 0, 0 } };
        state.indexBuffer   = { m_TerrainPass.indexBuffer, nvrhi::Format::R32_UINT, 0 };
        state.viewport.addViewportAndScissorRect(framebuffer->getFramebufferInfo().getViewport());
        m_CommandList->setGraphicsState(state);
        m_CommandList->drawIndexed(
            nvrhi::DrawArguments().setVertexCount(m_TerrainPass.indexCount));
    }
}
