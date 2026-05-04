#include "include/SharedGPUAssets.hpp"

#include <donut/app/ApplicationBase.h>

#include "include/Globals.hpp"
#include "include/Render.hpp"
#include "include/Scene.hpp"
#include "include/shaders/ShaderContracts.hpp"

#include <nvrhi/utils.h>

#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/TextureCache.h>
#include <donut/core/log.h>
#include <donut/core/math/math.h>
#include <donut/core/vfs/VFS.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

namespace dm = donut::math;
namespace vfs = donut::vfs;
namespace engine = donut::engine;

namespace Xylem {

namespace {

namespace compute_reg = shader::reg::Compute;

struct ImpostorBakeCB {
    dm::float4x4 mvp;
    dm::float4   _pad[12];
};

constexpr size_t c_ImpostorBakeCBSize =
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
    const uint32_t x = viewIndex % SharedGPUAssets::k_ImpostorAzimuthViews;
    const uint32_t y = viewIndex / SharedGPUAssets::k_ImpostorAzimuthViews;
    return dm::float2(
        float(x) / float(std::max(1u, SharedGPUAssets::k_ImpostorAzimuthViews - 1u)),
        float(y) / float(std::max(1u, SharedGPUAssets::k_ImpostorElevationViews - 1u)));
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
        bbox.m_mins.z, bbox.m_maxs.z);

    return dm::affineToHomogeneous(modelviewxfm) * projxfm;
}

} // namespace

bool SharedGPUAssets::Init()
{
    if (!m_Device || !m_ShaderFactory) {
        donut::log::error("SharedGPUAssets::Init missing device or shader factory");
        return false;
    }

    if (!_InitBakePipeline()) return false;

    engine::CommonRenderPasses commonPasses(m_Device, m_ShaderFactory);

    nvrhi::CommandListHandle initCL = m_Device->createCommandList();
    initCL->open();

    if (!_LoadBarkTextures(initCL, commonPasses)) { initCL->close(); return false; }
    if (!_BakeImpostors(initCL))                  { initCL->close(); return false; }

    initCL->close();
    m_Device->executeCommandList(initCL);

    // Publish stable atlas info to the UI.
    m_UI.impostorAssetCount     = static_cast<uint32_t>(m_Registry.getAssets().size());
    m_UI.impostorViewsPerAsset  = k_ImpostorViewCount;
    m_UI.impostorAzimuthViews   = k_ImpostorAzimuthViews;
    m_UI.impostorElevationViews = k_ImpostorElevationViews;
    m_UI.impostorAlbedoAtlasTexture = m_DebugAlbedoAtlasTexture;
    m_UI.impostorNormalAtlasTexture = m_DebugNormalAtlasTexture;
    m_UI.impostorDepthAtlasTexture  = m_DebugDepthAtlasTexture;

    return true;
}

bool SharedGPUAssets::OnAssetsDirty()
{
    nvrhi::CommandListHandle cl = m_Device->createCommandList();
    cl->open();

    // Bark textures track registered texture-set names; reload if the set list grew.
    // (Removed sets are tolerated — handles drop refcount on rebuild.)
    if (m_BarkTextures.size() != m_Registry.getBarkTextureSets().size()) {
        engine::CommonRenderPasses commonPasses(m_Device, m_ShaderFactory);
        if (!_LoadBarkTextures(cl, commonPasses)) { cl->close(); return false; }
    }

    if (!_BakeImpostors(cl)) { cl->close(); return false; }

    cl->close();
    m_Device->executeCommandList(cl);

    m_UI.impostorAssetCount         = static_cast<uint32_t>(m_Registry.getAssets().size());
    m_UI.impostorAlbedoAtlasTexture = m_DebugAlbedoAtlasTexture;
    m_UI.impostorNormalAtlasTexture = m_DebugNormalAtlasTexture;
    m_UI.impostorDepthAtlasTexture  = m_DebugDepthAtlasTexture;

    return true;
}

bool SharedGPUAssets::_InitBakePipeline()
{
    m_BakeVS = m_ShaderFactory->CreateShader(
        "app/ImpostorBake.hlsl", "bake_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_BakePS = m_ShaderFactory->CreateShader(
        "app/ImpostorBake.hlsl", "bake_ps", nullptr, nvrhi::ShaderType::Pixel);
    if (!m_BakeVS || !m_BakePS) {
        donut::log::error("SharedGPUAssets: impostor bake shaders failed to compile");
        return false;
    }

    nvrhi::VertexAttributeDesc attributes[] = {
        nvrhi::VertexAttributeDesc()
            .setName("POSITION").setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0).setBufferIndex(0).setElementStride(sizeof(dm::float3)),
        nvrhi::VertexAttributeDesc()
            .setName("NORMAL").setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0).setBufferIndex(1).setElementStride(sizeof(dm::float3)),
        nvrhi::VertexAttributeDesc()
            .setName("TANGENT").setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0).setBufferIndex(2).setElementStride(sizeof(dm::float3)),
        nvrhi::VertexAttributeDesc()
            .setName("BITANGENT").setFormat(nvrhi::Format::RGB32_FLOAT)
            .setOffset(0).setBufferIndex(3).setElementStride(sizeof(dm::float3)),
        nvrhi::VertexAttributeDesc()
            .setName("UV").setFormat(nvrhi::Format::RG32_FLOAT)
            .setOffset(0).setBufferIndex(4).setElementStride(sizeof(dm::float2)),
    };
    m_BakeInputLayout = m_Device->createInputLayout(
        attributes, uint32_t(std::size(attributes)), m_BakeVS);
    if (!m_BakeInputLayout) return false;

    nvrhi::BindingLayoutDesc bakeLayoutDesc;
    bakeLayoutDesc.visibility = nvrhi::ShaderType::All;
    bakeLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(compute_reg::ImpostorBake::kCB_Bake),
        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::ImpostorBake::kTex_Diffuse),
        nvrhi::BindingLayoutItem::Texture_SRV(compute_reg::ImpostorBake::kTex_NormalMap),
        nvrhi::BindingLayoutItem::Sampler(compute_reg::ImpostorBake::kSampler_Main),
    };
    m_BakeBindingLayout = m_Device->createBindingLayout(bakeLayoutDesc);
    if (!m_BakeBindingLayout) return false;

    const uint32_t numAssets = std::max(1u,
        static_cast<uint32_t>(m_Registry.getAssets().size()));
    m_BakeConstantBuffer = m_Device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(c_ImpostorBakeCBSize)
            .setIsConstantBuffer(true)
            .setIsVolatile(true)
            .setMaxVersions(numAssets * k_ImpostorViewCount)
            .setDebugName("Shared_ImpostorBakeCB")
            .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer));
    if (!m_BakeConstantBuffer) return false;

    m_BarkSampler = m_Device->createSampler(
        nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Wrap)
            .setAllFilters(true)
            .setMaxAnisotropy(8.f));
    return m_BarkSampler != nullptr;
}

bool SharedGPUAssets::_LoadBarkTextures(nvrhi::ICommandList* cl, engine::CommonRenderPasses& commonPasses)
{
    const auto& barkTextureSets = m_Registry.getBarkTextureSets();
    if (barkTextureSets.empty()) {
        donut::log::error("SharedGPUAssets: no bark texture sets registered");
        return false;
    }

    engine::TextureCache textureCache(m_Device, std::make_shared<vfs::NativeFileSystem>(), nullptr);
    m_BarkTextures.assign(barkTextureSets.size(), {});

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

        auto diffLoaded = textureCache.LoadTextureFromFile(diffPath, true,  &commonPasses, cl);
        auto normLoaded = textureCache.LoadTextureFromFile(normPath, false, &commonPasses, cl);
        m_BarkTextures[i].diffuse   = diffLoaded ? diffLoaded->texture : nullptr;
        m_BarkTextures[i].normalMap = normLoaded ? normLoaded->texture : nullptr;
        if (!m_BarkTextures[i].diffuse || !m_BarkTextures[i].normalMap) {
            donut::log::error("SharedGPUAssets: bark texture load failed");
            return false;
        }
    }
    return true;
}

bool SharedGPUAssets::_RebuildAssetDimsBuffer(nvrhi::ICommandList* cl)
{
    const auto& assets = m_Registry.getAssets();
    const uint32_t numAssets = static_cast<uint32_t>(assets.size());

    const dm::affine3 toImpostor = BuildtoObjectTransform();
    std::vector<dm::float4> assetDims(std::max(1u, numAssets), dm::float4(0.f));
    for (uint32_t ai = 0; ai < numAssets; ai++) {
        if (assets[ai].lods.empty()) continue;
        dm::box3 rotated = assets[ai].lods[0].bbox * toImpostor;
        const dm::float3 half = rotated.diagonal() * 0.5f;
        assetDims[ai] = dm::float4(half, 0.f);
    }

    m_AssetDimsBuffer = m_Device->createBuffer(
        nvrhi::BufferDesc()
            .setByteSize(assetDims.size() * sizeof(dm::float4))
            .setStructStride(sizeof(dm::float4))
            .setDebugName("Shared_ImpostorAssetDims")
            .setInitialState(nvrhi::ResourceStates::CopyDest));
    if (!m_AssetDimsBuffer) return false;

    cl->beginTrackingBufferState(m_AssetDimsBuffer, nvrhi::ResourceStates::CopyDest);
    cl->writeBuffer(m_AssetDimsBuffer, assetDims.data(), assetDims.size() * sizeof(dm::float4));
    cl->setPermanentBufferState(m_AssetDimsBuffer, nvrhi::ResourceStates::ShaderResource);
    return true;
}

bool SharedGPUAssets::_BakeImpostors(nvrhi::ICommandList* cl)
{
    if (!cl || !m_BakeVS || !m_BakePS) return false;

    const auto& assets = m_Registry.getAssets();
    const uint32_t numAssets = static_cast<uint32_t>(assets.size());
    const uint32_t numSlices = std::max(1u, numAssets * k_ImpostorViewCount);

    if (!m_BakeConstantBuffer
        || m_BakeConstantBuffer->getDesc().maxVersions < numSlices)
    {
        m_BakeConstantBuffer = m_Device->createBuffer(
            nvrhi::BufferDesc()
                .setByteSize(c_ImpostorBakeCBSize)
                .setIsConstantBuffer(true)
                .setIsVolatile(true)
                .setMaxVersions(numSlices)
                .setDebugName("Shared_ImpostorBakeCB")
                .enableAutomaticStateTracking(nvrhi::ResourceStates::ConstantBuffer));
        if (!m_BakeConstantBuffer) return false;
    }

    if (!_RebuildAssetDimsBuffer(cl)) return false;

    m_AlbedoAlphaTexture = m_Device->createTexture(
        nvrhi::TextureDesc()
            .setWidth(k_ImpostorBakeResolution).setHeight(k_ImpostorBakeResolution)
            .setArraySize(numSlices)
            .setDimension(nvrhi::TextureDimension::Texture2DArray)
            .setFormat(nvrhi::Format::RGBA8_UNORM)
            .setIsRenderTarget(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::RenderTarget)
            .setDebugName("Shared_ImpostorAlbedoAlphaArray"));
    m_NormalTexture = m_Device->createTexture(
        nvrhi::TextureDesc()
            .setWidth(k_ImpostorBakeResolution).setHeight(k_ImpostorBakeResolution)
            .setArraySize(numSlices)
            .setDimension(nvrhi::TextureDimension::Texture2DArray)
            .setFormat(nvrhi::Format::RGBA8_UNORM)
            .setIsRenderTarget(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::RenderTarget)
            .setDebugName("Shared_ImpostorNormalArray"));
    m_DepthTexture = m_Device->createTexture(
        nvrhi::TextureDesc()
            .setWidth(k_ImpostorBakeResolution).setHeight(k_ImpostorBakeResolution)
            .setArraySize(numSlices)
            .setDimension(nvrhi::TextureDimension::Texture2DArray)
            .setFormat(nvrhi::Format::D32)
            .setIsRenderTarget(true)
            .enableAutomaticStateTracking(nvrhi::ResourceStates::DepthWrite)
            .setDebugName("Shared_ImpostorDepthArray"));
    if (!m_AlbedoAlphaTexture || !m_NormalTexture || !m_DepthTexture) return false;

    // Per-slice framebuffers consume RTV/DSV descriptors. Build them
    // just-in-time inside the bake loop so the heap stays flat.
    auto makeBakeFramebuffer = [&](uint32_t slice) {
        nvrhi::FramebufferDesc fbDesc;
        fbDesc
            .addColorAttachment(nvrhi::FramebufferAttachment()
                .setTexture(m_AlbedoAlphaTexture).setArraySlice(slice))
            .addColorAttachment(nvrhi::FramebufferAttachment()
                .setTexture(m_NormalTexture).setArraySlice(slice))
            .setDepthAttachment(nvrhi::FramebufferAttachment()
                .setTexture(m_DepthTexture).setArraySlice(slice));
        return m_Device->createFramebuffer(fbDesc);
    };

    nvrhi::FramebufferHandle pipelineProtoFB = makeBakeFramebuffer(0);
    if (!pipelineProtoFB) return false;

    m_BakeBindingSets.assign(m_BarkTextures.size(), nullptr);
    for (size_t i = 0; i < m_BarkTextures.size(); i++) {
        nvrhi::BindingSetDesc bsd;
        bsd.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(compute_reg::ImpostorBake::kCB_Bake, m_BakeConstantBuffer),
            nvrhi::BindingSetItem::Texture_SRV(compute_reg::ImpostorBake::kTex_Diffuse, m_BarkTextures[i].diffuse),
            nvrhi::BindingSetItem::Texture_SRV(compute_reg::ImpostorBake::kTex_NormalMap, m_BarkTextures[i].normalMap),
            nvrhi::BindingSetItem::Sampler(compute_reg::ImpostorBake::kSampler_Main, m_BarkSampler),
        };
        m_BakeBindingSets[i] = m_Device->createBindingSet(bsd, m_BakeBindingLayout);
        if (!m_BakeBindingSets[i]) return false;
    }

    nvrhi::GraphicsPipelineDesc psoDesc;
    psoDesc.VS = m_BakeVS;
    psoDesc.PS = m_BakePS;
    psoDesc.inputLayout = m_BakeInputLayout;
    psoDesc.bindingLayouts = { m_BakeBindingLayout };
    psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
    psoDesc.renderState.depthStencilState.setDepthFunc(nvrhi::ComparisonFunc::Less);
    psoDesc.renderState.rasterState.setCullNone();
    m_BakePipeline = m_Device->createGraphicsPipeline(psoDesc, pipelineProtoFB->getFramebufferInfo());
    if (!m_BakePipeline) return false;

    nvrhi::GraphicsState state;
    state.pipeline = m_BakePipeline;
    state.viewport.addViewportAndScissorRect(nvrhi::Viewport(
        float(k_ImpostorBakeResolution), float(k_ImpostorBakeResolution)));

    auto makeVB = [&](const std::vector<dm::float3>& src, const char* dbgName) {
        if (src.empty()) return nvrhi::BufferHandle();
        auto buf = m_Device->createBuffer(nvrhi::BufferDesc()
            .setByteSize(src.size() * sizeof(dm::float3))
            .setIsVertexBuffer(true)
            .setDebugName(dbgName)
            .setInitialState(nvrhi::ResourceStates::CopyDest));
        cl->beginTrackingBufferState(buf, nvrhi::ResourceStates::CopyDest);
        cl->writeBuffer(buf, src.data(), src.size() * sizeof(dm::float3));
        cl->setPermanentBufferState(buf, nvrhi::ResourceStates::VertexBuffer);
        return buf;
    };
    auto makeUVB = [&](const std::vector<dm::float2>& src, const char* dbgName) {
        if (src.empty()) return nvrhi::BufferHandle();
        auto buf = m_Device->createBuffer(nvrhi::BufferDesc()
            .setByteSize(src.size() * sizeof(dm::float2))
            .setIsVertexBuffer(true)
            .setDebugName(dbgName)
            .setInitialState(nvrhi::ResourceStates::CopyDest));
        cl->beginTrackingBufferState(buf, nvrhi::ResourceStates::CopyDest);
        cl->writeBuffer(buf, src.data(), src.size() * sizeof(dm::float2));
        cl->setPermanentBufferState(buf, nvrhi::ResourceStates::VertexBuffer);
        return buf;
    };
    auto makeIB = [&](const std::vector<uint32_t>& src, const char* dbgName) {
        if (src.empty()) return nvrhi::BufferHandle();
        auto buf = m_Device->createBuffer(nvrhi::BufferDesc()
            .setByteSize(src.size() * sizeof(uint32_t))
            .setIsIndexBuffer(true)
            .setDebugName(dbgName)
            .setInitialState(nvrhi::ResourceStates::CopyDest));
        cl->beginTrackingBufferState(buf, nvrhi::ResourceStates::CopyDest);
        cl->writeBuffer(buf, src.data(), src.size() * sizeof(uint32_t));
        cl->setPermanentBufferState(buf, nvrhi::ResourceStates::IndexBuffer);
        return buf;
    };

    for (uint32_t ai = 0; ai < numAssets; ai++) {
        const auto& asset = assets[ai];
        if (asset.lods.empty()) continue;
        if (asset.textureSetIdx >= m_BakeBindingSets.size()) continue;

        const Scene::TreeLODDef& lod = asset.lods[0];
        if (lod.indices.empty()) continue;

        nvrhi::BufferHandle posBuf  = makeVB(lod.positions,  "ImpostorBake_Pos");
        nvrhi::BufferHandle normBuf = makeVB(lod.normals,    "ImpostorBake_Norm");
        nvrhi::BufferHandle tanBuf  = makeVB(lod.tangents,   "ImpostorBake_Tan");
        nvrhi::BufferHandle bitBuf  = makeVB(lod.bitangents, "ImpostorBake_Bit");
        nvrhi::BufferHandle uvBuf   = makeUVB(lod.uvs,       "ImpostorBake_UV");
        nvrhi::BufferHandle idxBuf  = makeIB(lod.indices,    "ImpostorBake_Idx");
        if (!posBuf || !normBuf || !tanBuf || !bitBuf || !uvBuf || !idxBuf) return false;

        state.bindings = { m_BakeBindingSets[asset.textureSetIdx] };
        state.vertexBuffers = {
            { posBuf,  0, 0 },
            { normBuf, 1, 0 },
            { tanBuf,  2, 0 },
            { bitBuf,  3, 0 },
            { uvBuf,   4, 0 },
        };
        state.indexBuffer = { idxBuf, nvrhi::Format::R32_UINT, 0 };

        for (uint32_t vi = 0; vi < k_ImpostorViewCount; vi++) {
            const uint32_t slice = ai * k_ImpostorViewCount + vi;
            nvrhi::FramebufferHandle framebuffer = (slice == 0)
                ? pipelineProtoFB
                : makeBakeFramebuffer(slice);
            if (!framebuffer) return false;

            nvrhi::utils::ClearColorAttachment(cl, framebuffer, 0, nvrhi::Color(0.f, 0.f, 0.f, 0.f));
            nvrhi::utils::ClearColorAttachment(cl, framebuffer, 1, nvrhi::Color(0.5f, 0.5f, 1.f, 0.f));
            nvrhi::utils::ClearDepthStencilAttachment(cl, framebuffer, 1.f, 0);

            ImpostorBakeCB cb;
            cb.mvp = BuildtoProjTransform(lod.bbox, vi);
            cl->writeBuffer(m_BakeConstantBuffer, &cb, sizeof(cb));

            state.framebuffer = framebuffer;
            cl->setGraphicsState(state);
            cl->drawIndexed(nvrhi::DrawArguments().setVertexCount(static_cast<uint32_t>(lod.indices.size())));
        }
    }

    // Per-asset atlas-sheet debug textures for ImGui inspection.
    m_DebugAlbedoAtlasTexture = m_Device->createTexture(nvrhi::TextureDesc()
        .setWidth(k_ImpostorBakeResolution * k_ImpostorAzimuthViews)
        .setHeight(k_ImpostorBakeResolution * k_ImpostorElevationViews)
        .setFormat(nvrhi::Format::RGBA8_UNORM)
        .setInitialState(nvrhi::ResourceStates::ShaderResource)
        .setKeepInitialState(true)
        .setDebugName("Shared_ImpostorAlbedoAtlasDebug"));
    m_DebugNormalAtlasTexture = m_Device->createTexture(nvrhi::TextureDesc()
        .setWidth(k_ImpostorBakeResolution * k_ImpostorAzimuthViews)
        .setHeight(k_ImpostorBakeResolution * k_ImpostorElevationViews)
        .setFormat(nvrhi::Format::RGBA8_UNORM)
        .setInitialState(nvrhi::ResourceStates::ShaderResource)
        .setKeepInitialState(true)
        .setDebugName("Shared_ImpostorNormalAtlasDebug"));
    m_DebugDepthAtlasTexture = m_Device->createTexture(nvrhi::TextureDesc()
        .setWidth(k_ImpostorBakeResolution * k_ImpostorAzimuthViews)
        .setHeight(k_ImpostorBakeResolution * k_ImpostorElevationViews)
        .setFormat(nvrhi::Format::R32_FLOAT)
        .setInitialState(nvrhi::ResourceStates::ShaderResource)
        .setKeepInitialState(true)
        .setDebugName("Shared_ImpostorDepthAtlasDebug"));
    if (!m_DebugAlbedoAtlasTexture || !m_DebugNormalAtlasTexture || !m_DebugDepthAtlasTexture)
        return false;

    return true;
}

bool SharedGPUAssets::CopySelectedImpostorDebugAtlases(nvrhi::ICommandList* cl, uint32_t selectedAsset)
{
    if (!cl
        || !m_AlbedoAlphaTexture
        || !m_NormalTexture
        || !m_DepthTexture
        || !m_DebugAlbedoAtlasTexture
        || !m_DebugNormalAtlasTexture
        || !m_DebugDepthAtlasTexture)
    {
        return false;
    }

    const uint32_t numAssets = static_cast<uint32_t>(m_Registry.getAssets().size());
    const uint32_t assetIdx = std::min(
        selectedAsset,
        numAssets == 0 ? 0u : numAssets - 1);

    for (uint32_t view = 0; view < k_ImpostorViewCount; view++) {
        const uint32_t tileX = view % k_ImpostorAzimuthViews;
        const uint32_t tileY = view / k_ImpostorAzimuthViews;
        const uint32_t tileSlice = assetIdx * k_ImpostorViewCount + view;
        nvrhi::TextureSlice dst = nvrhi::TextureSlice()
            .setOrigin(tileX * k_ImpostorBakeResolution, tileY * k_ImpostorBakeResolution, 0)
            .setWidth(k_ImpostorBakeResolution)
            .setHeight(k_ImpostorBakeResolution);

        cl->copyTexture(
            m_DebugAlbedoAtlasTexture, dst,
            m_AlbedoAlphaTexture, nvrhi::TextureSlice().setArraySlice(tileSlice));
        cl->copyTexture(
            m_DebugNormalAtlasTexture, dst,
            m_NormalTexture, nvrhi::TextureSlice().setArraySlice(tileSlice));
        cl->copyTexture(
            m_DebugDepthAtlasTexture, dst,
            m_DepthTexture, nvrhi::TextureSlice().setArraySlice(tileSlice));
    }

    return true;
}

} // namespace Xylem
