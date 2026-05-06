#pragma pack_matrix(row_major)

#include "../include/macros.h"
#include "types.hlsli"
#include "ShaderRegisterMap.hlsli"

cbuffer CB : register(XY_REG_B_COMPUTE_SHADOWIMPOSTOR_CB_FRAME)
{
    float4x4 viewProj;
    float4x4 viewMatrix;
    float4x4 lightViewProj[XYLEM_NUM_CASCADES];
    float3   sunLightDir;
    float    _pad0;
    float4   cascadeSplits;

    frustum  viewFrustum;
    float4x4 worldToLight;
    float4   shadowCasterMinLS[XYLEM_NUM_CASCADES];
    float4   shadowCasterMaxLS[XYLEM_NUM_CASCADES];

    float3   cameraPos;
    uint     numRegions;
    uint     totalCapacity;
    uint     numLods;
    float    _pad1a;
    float    _pad1b;
    float4   lodDistances;

    float2   hizDimensions;
    float    maxHiZMip;
    uint     hizEnabled;
    float    impostorAlphaClip;
    uint     showShadowImpostors;  // present in shared layout, unused by this shader
    float    shadowImpostorBias;
    float    _pad2;
};

struct RootConstant { uint assetIndex; uint cascadeIndex; };
ConstantBuffer<RootConstant> rc : register(XY_REG_B_COMPUTE_SHADOWIMPOSTOR_PUSH_C_ASSET_CASCADE);

struct InstanceRenderData
{
    float4x4 model;
    float3x3 normal;
    uint     treeId;
};

struct CullInstanceData
{
    box3 bbox;
    uint baseSlot;
    uint regionId;
    uint active;
};

StructuredBuffer<uint>               shadowImpostorVisBuf : register(XY_REG_T_COMPUTE_SHADOWIMPOSTOR_SRV_VIS);
StructuredBuffer<InstanceRenderData> instBuf              : register(XY_REG_T_COMPUTE_SHADOWIMPOSTOR_SRV_INSTANCES);
StructuredBuffer<uint>               shadowImpostorSlotOffsets : register(XY_REG_T_COMPUTE_SHADOWIMPOSTOR_SRV_SLOT_OFFSETS);
StructuredBuffer<CullInstanceData>   cullData             : register(XY_REG_T_COMPUTE_SHADOWIMPOSTOR_SRV_CULL_DATA);

Texture2DArray                       t_ImpostorAlbedoAlpha : register(XY_REG_T_COMPUTE_SHADOWIMPOSTOR_TEX_ALBEDO);
Texture2DArray<float>                t_ImpostorDepth       : register(XY_REG_T_COMPUTE_SHADOWIMPOSTOR_TEX_DEPTH);
StructuredBuffer<float4>             assetDims             : register(XY_REG_T_COMPUTE_SHADOWIMPOSTOR_SRV_ASSET_DIMS);

SamplerState                         s_Sampler             : register(XY_REG_S_COMPUTE_SHADOWIMPOSTOR_SAMPLER_MAIN);
SamplerState                         s_DepthSampler        : register(XY_REG_S_COMPUTE_SHADOWIMPOSTOR_SAMPLER_DEPTH);

void ViewBasis(float3 dirToLight, out float3 right, out float3 viewUp)
{
    float3 worldUp = float3(0.0, 1.0, 0.0);
    float3 horizontal = float3(dirToLight.x, 0.0, dirToLight.z);
    float horizontalLen = length(horizontal);
    right = (horizontalLen >= 1e-6)
        ? normalize(cross(worldUp, -horizontal / horizontalLen))
        : float3(1.0, 0.0, 0.0);
    viewUp = normalize(cross(-dirToLight, right));
}

float2 UnitVectorToHemiOctahedron(float3 dirToLight)
{
    float3 n = normalize(dirToLight);
    n.y = max(n.y, 0.0);

    float2 oct = float2(n.x + n.z, n.z - n.x) * 0.5;
    oct *= rcp(max(abs(oct.x) + n.y + abs(oct.y), 1e-6));

    float2 plane = float2(oct.x - oct.y, oct.x + oct.y);
    return saturate(plane * 0.5 + 0.5);
}

uint NearestImpostorViewIndex(float3 dirToLightBake)
{
    float2 grid = UnitVectorToHemiOctahedron(dirToLightBake);
    grid.x *= (float)(XYLEM_IMPOSTOR_AZIMUTH_VIEWS - 1u);
    grid.y *= (float)(XYLEM_IMPOSTOR_ELEVATION_VIEWS - 1u);
    uint2 g = uint2(round(grid));
    g.x = min(g.x, XYLEM_IMPOSTOR_AZIMUTH_VIEWS - 1u);
    g.y = min(g.y, XYLEM_IMPOSTOR_ELEVATION_VIEWS - 1u);
    return g.y * XYLEM_IMPOSTOR_AZIMUTH_VIEWS + g.x;
}

void ImpostorViewBasisLight(uint viewIndex, out float3 dirToLight, out float3 right, out float3 viewUp)
{
    // Decode the same hemi-octahedron mapping used at bake time.
    uint x = viewIndex % XYLEM_IMPOSTOR_AZIMUTH_VIEWS;
    uint y = viewIndex / XYLEM_IMPOSTOR_AZIMUTH_VIEWS;
    float2 uv = float2(
        (float)x / (float)max(1u, XYLEM_IMPOSTOR_AZIMUTH_VIEWS - 1u),
        (float)y / (float)max(1u, XYLEM_IMPOSTOR_ELEVATION_VIEWS - 1u));
    float2 plane = uv * 2.0 - 1.0;
    float2 oct = float2(plane.x + plane.y, plane.y - plane.x) * 0.5;
    float yc = max(1.0 - abs(oct.x) - abs(oct.y), 0.0);
    dirToLight = normalize(float3(oct.x - oct.y, yc, oct.x + oct.y));
    ViewBasis(dirToLight, right, viewUp);
}

void ProjectImpostorFrame(
    uint viewIndex,
    float3 dims,
    float3 center,
    float3 relBake,
    float3x3 instanceNormal,
    out float2 uv,
    out float3 frameWorldPos,
    out float3 depthAxis,
    out float halfDepth)
{
    float3 sampledDirBake, sampledRightBake, sampledUpBake;
    ImpostorViewBasisLight(viewIndex, sampledDirBake, sampledRightBake, sampledUpBake);

    float halfWidth  = max(dot(dims, abs(sampledRightBake)), 0.01);
    float halfHeight = max(dot(dims, abs(sampledUpBake)),    0.01);
    halfDepth        = max(dot(dims, abs(sampledDirBake)),   0.01);

    float frameX = dot(relBake, sampledRightBake);
    float frameY = dot(relBake, sampledUpBake);
    float3 frameBakePos = sampledRightBake * frameX + sampledUpBake * frameY;

    uv = float2(frameX / (2.0 * halfWidth) + 0.5, 0.5 - frameY / (2.0 * halfHeight));
    frameWorldPos = center + mul(frameBakePos, instanceNormal);
    depthAxis = normalize(mul(sampledDirBake, instanceNormal));
}

struct V2P
{
    float4 pos           : SV_Position;
    float3 frameWorldPos : FRAME_WORLD_POS;
    float3 depthAxis     : DEPTH_AXIS;
    float2 uv            : UV0;
    nointerpolation float halfDepth  : HALF_DEPTH;
    nointerpolation uint  viewIndex  : VIEW_INDEX;
};

void shadow_impostor_vs(
    in uint vertexId   : SV_VertexID,
    in uint instanceId : SV_InstanceID,
    out V2P o)
{
    uint slot = rc.assetIndex * XYLEM_NUM_CASCADES + rc.cascadeIndex;
    uint persistentId = shadowImpostorVisBuf[shadowImpostorSlotOffsets[slot] + instanceId];

    CullInstanceData cull = cullData[persistentId];
    InstanceRenderData inst = instBuf[persistentId];

    float3 dims = max(assetDims[rc.assetIndex].xyz, float3(0.01, 0.01, 0.01));
    float3 worldCenter = (cull.bbox.min + cull.bbox.max) * 0.5;

    float3 dirToLightW    = normalize(-sunLightDir);
    float3 dirToLightBake = normalize(mul(dirToLightW, transpose(inst.normal)));
    uint viewIndex = NearestImpostorViewIndex(dirToLightBake);

    float3 rightW, upW;
    ViewBasis(dirToLightW, rightW, upW);
    float3 worldHalfExt = (cull.bbox.max - cull.bbox.min) * 0.5;
    float cardHalfWidth  = max(dot(worldHalfExt, abs(rightW)), 0.01);
    float cardHalfHeight = max(dot(worldHalfExt, abs(upW)),    0.01);

    float2 corner;
    corner.x = (vertexId == 1 || vertexId == 3) ? 1.0 : -1.0;
    corner.y = (vertexId >= 2)                  ? 1.0 : -1.0;

    float3 worldPos = worldCenter
        + rightW * (corner.x * cardHalfWidth)
        + upW    * (corner.y * cardHalfHeight);

    float3 vertWorld = worldPos - worldCenter;
    float3 vertBake  = mul(vertWorld, transpose(inst.normal));

    float2 uv;
    float3 frameWorldPos;
    float3 depthAxis;
    float halfDepth;
    ProjectImpostorFrame(viewIndex, dims, worldCenter, vertBake, inst.normal,
                         uv, frameWorldPos, depthAxis, halfDepth);

    o.pos = mul(float4(worldPos, 1.0), lightViewProj[rc.cascadeIndex]);
    o.frameWorldPos = frameWorldPos;
    o.depthAxis = depthAxis;
    o.uv = uv;
    o.halfDepth = halfDepth;
    o.viewIndex = viewIndex;
}

void shadow_impostor_ps(
    in V2P i,
    out float o_depth : SV_Depth)
{
    uint viewCount = XYLEM_IMPOSTOR_AZIMUTH_VIEWS * XYLEM_IMPOSTOR_ELEVATION_VIEWS;
    uint slice = rc.assetIndex * viewCount + i.viewIndex;

    float alpha = t_ImpostorAlbedoAlpha.Sample(s_Sampler, float3(i.uv, (float)slice)).a;
    clip(alpha - impostorAlphaClip);

    float depthAtlas = t_ImpostorDepth.SampleLevel(s_DepthSampler, float3(i.uv, (float)slice), 0.0);
    float3 depthWorldPos = i.frameWorldPos + normalize(i.depthAxis) *
                           ((0.5 - depthAtlas) * (2.0 * i.halfDepth));

    // Bias: push caster away from the light so the receiver wins its self-shadow
    // comparison (Peter-Pan rather than acne). Shadow framebuffer is STANDARD Z
    // (Less test, far=1.0), so adding sunLightDir produces a larger projected depth.
    depthWorldPos += sunLightDir * shadowImpostorBias;

    float4 clipPos = mul(float4(depthWorldPos, 1.0), lightViewProj[rc.cascadeIndex]);
    o_depth = clipPos.z / clipPos.w;
}
