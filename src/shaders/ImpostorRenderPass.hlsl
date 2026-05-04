#pragma pack_matrix(row_major)

#include "../include/macros.h"
#include "types.hlsli"
#include "ShaderRegisterMap.hlsli"

cbuffer CB : register(XY_REG_B_COMPUTE_IMPOSTOR_CB_FRAME)
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
    float3   _pad2;
};

struct RootConstant { uint assetIndex; };
ConstantBuffer<RootConstant> rc : register(XY_REG_B_COMPUTE_IMPOSTOR_PUSH_C_ASSET_INDEX);

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

StructuredBuffer<uint>               impostorVisBuf     : register(XY_REG_T_COMPUTE_IMPOSTOR_SRV_VIS);
StructuredBuffer<InstanceRenderData> instBuf            : register(XY_REG_T_COMPUTE_IMPOSTOR_SRV_INSTANCES);
StructuredBuffer<uint>               impostorSlotOffsets : register(XY_REG_T_COMPUTE_IMPOSTOR_SRV_SLOT_OFFSETS);
StructuredBuffer<CullInstanceData>   cullData           : register(XY_REG_T_COMPUTE_IMPOSTOR_SRV_CULL_DATA);

Texture2DArray                       t_ImpostorAlbedoAlpha : register(XY_REG_T_COMPUTE_IMPOSTOR_TEX_ALBEDO);
Texture2DArray                       t_ImpostorNormal      : register(XY_REG_T_COMPUTE_IMPOSTOR_TEX_NORMAL);
Texture2DArray<float>                t_ImpostorDepth       : register(XY_REG_T_COMPUTE_IMPOSTOR_TEX_DEPTH);
Texture2DArray                       t_ShadowMap           : register(XY_REG_T_COMPUTE_IMPOSTOR_TEX_SHADOW_MAP);

// Per-asset bake-space bbox half extents. The render card and virtual
// projections both use these bounds around the instance bbox center.
StructuredBuffer<float4>             assetDims           : register(XY_REG_T_COMPUTE_IMPOSTOR_SRV_ASSET_DIMS);

SamplerState                         s_Sampler          : register(XY_REG_S_COMPUTE_IMPOSTOR_SAMPLER_MAIN);
SamplerComparisonState               s_ShadowSampler    : register(XY_REG_S_COMPUTE_IMPOSTOR_SAMPLER_SHADOW);
SamplerState                         s_DepthSampler     : register(XY_REG_S_COMPUTE_IMPOSTOR_SAMPLER_DEPTH);

struct V2P
{
    float4 pos       : SV_Position;
    float3 frameWorldPos0 : FRAME_WORLD_POS0;
    float3 frameWorldPos1 : FRAME_WORLD_POS1;
    float3 frameWorldPos2 : FRAME_WORLD_POS2;
    float3 depthAxis0 : DEPTH_AXIS0;
    float3 depthAxis1 : DEPTH_AXIS1;
    float3 depthAxis2 : DEPTH_AXIS2;
    float2 uv0        : UV0;
    float2 uv1        : UV1;
    float2 uv2        : UV2;
    nointerpolation float3 weights : BLEND_WEIGHTS;
    nointerpolation float3 halfDepths : HALF_DEPTHS;
    nointerpolation uint viewIndex0   : VIEW_INDEX0;
    nointerpolation uint viewIndex1   : VIEW_INDEX1;
    nointerpolation uint viewIndex2   : VIEW_INDEX2;
    nointerpolation uint persistentId : PERSISTENT_ID;
};

float3 HemiOctahedronToUnitVector(float2 uv)
{
    float2 plane = uv * 2.0 - 1.0;
    float2 oct = float2(plane.x + plane.y, plane.y - plane.x) * 0.5;
    float y = max(1.0 - abs(oct.x) - abs(oct.y), 0.0);
    return normalize(float3(oct.x - oct.y, y, oct.x + oct.y));
}

float2 UnitVectorToHemiOctahedron(float3 dirToCamera)
{
    float3 n = normalize(dirToCamera);
    n.y = max(n.y, 0.0);

    float2 oct = float2(n.x + n.z, n.z - n.x) * 0.5;
    oct *= rcp(max(abs(oct.x) + n.y + abs(oct.y), 1e-6));

    float2 plane = float2(oct.x - oct.y, oct.x + oct.y);
    return saturate(plane * 0.5 + 0.5);
}

float2 ImpostorViewGridUV(uint viewIndex)
{
    uint x = viewIndex % XYLEM_IMPOSTOR_AZIMUTH_VIEWS;
    uint y = viewIndex / XYLEM_IMPOSTOR_AZIMUTH_VIEWS;
    return float2(
        (float)x / (float)max(1u, XYLEM_IMPOSTOR_AZIMUTH_VIEWS - 1u),
        (float)y / (float)max(1u, XYLEM_IMPOSTOR_ELEVATION_VIEWS - 1u));
}

float3 ImpostorViewDirection(uint viewIndex)
{
    return HemiOctahedronToUnitVector(ImpostorViewGridUV(viewIndex));
}

void ViewBasis(float3 dirToCamera, out float3 right, out float3 viewUp)
{
    float3 worldUp = float3(0.0, 1.0, 0.0);
    float3 horizontal = float3(dirToCamera.x, 0.0, dirToCamera.z);
    float horizontalLen = length(horizontal);
    right = (horizontalLen >= 1e-6)
        ? normalize(cross(worldUp, -horizontal / horizontalLen))
        : float3(1.0, 0.0, 0.0);
    viewUp = normalize(cross(-dirToCamera, right));
}

void ImpostorViewBasis(uint viewIndex, out float3 dirToCamera, out float3 right, out float3 viewUp)
{
    dirToCamera = ImpostorViewDirection(viewIndex);
    ViewBasis(dirToCamera, right, viewUp);
}


uint ImpostorViewIndex(uint2 gridCoord)
{
    uint x = min(gridCoord.x, XYLEM_IMPOSTOR_AZIMUTH_VIEWS - 1u);
    uint y = min(gridCoord.y, XYLEM_IMPOSTOR_ELEVATION_VIEWS - 1u);
    return y * XYLEM_IMPOSTOR_AZIMUTH_VIEWS + x;
}

void SelectImpostorViews(float3 dirToCamera, out uint3 viewIndices, out float3 weights)
{
    float2 gridMax = float2(
        (float)(XYLEM_IMPOSTOR_AZIMUTH_VIEWS - 1u),
        (float)(XYLEM_IMPOSTOR_ELEVATION_VIEWS - 1u));
    float2 grid = UnitVectorToHemiOctahedron(dirToCamera) * gridMax;

    uint maxBaseX = (XYLEM_IMPOSTOR_AZIMUTH_VIEWS > 1u) ? (XYLEM_IMPOSTOR_AZIMUTH_VIEWS - 2u) : 0u;
    uint maxBaseY = (XYLEM_IMPOSTOR_ELEVATION_VIEWS > 1u) ? (XYLEM_IMPOSTOR_ELEVATION_VIEWS - 2u) : 0u;
    // bottom-left atlas index within a quad
    // clamped to one ring of outermost atlas to allow for blending
    float2 baseF = min(floor(grid), float2((float)maxBaseX, (float)maxBaseY));
    float2 frac = grid - baseF;

    // bottom/top-left/right
    uint2 bl = uint2((uint)baseF.x, (uint)baseF.y);
    uint2 br = uint2(min(bl.x + 1u, XYLEM_IMPOSTOR_AZIMUTH_VIEWS - 1u), bl.y);
    uint2 tl = uint2(bl.x, min(bl.y + 1u, XYLEM_IMPOSTOR_ELEVATION_VIEWS - 1u));
    uint2 tr = uint2(br.x, tl.y);

    // within the right triangle (defined cw)
    // frac.x + frac.y == 1.0 : on the diagonal
    // therefore <= 1.0 is inside tri, > 1.0 is in other tri
    /*
     *  |\
     *  | \  <-- outside tri barycentric coords > 1
     *  |__\
     */
    
    float bary = frac.x + frac.y;
    if (bary <= 1.0)
    {
        viewIndices = uint3(
            ImpostorViewIndex(bl),
            ImpostorViewIndex(tl),
            ImpostorViewIndex(br));
        weights = float3(1.0 - bary, frac.y, frac.x);
    }
    else
    {
        viewIndices = uint3(
            ImpostorViewIndex(br),
            ImpostorViewIndex(tl),
            ImpostorViewIndex(tr));
        // treat tr as origin and 1-x as +x and 1-y as +y
        weights = float3(1.0 - frac.y, 1.0 - frac.x, bary - 1.0);
    }

    weights = max(weights, 0.0);
    weights *= rcp(max(weights.x + weights.y + weights.z, 1e-6));
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
    float3 sampledDirToCameraBake;
    float3 sampledRightBake;
    float3 sampledUpBake;
    ImpostorViewBasis(viewIndex, sampledDirToCameraBake, sampledRightBake, sampledUpBake);

    float halfWidth = max(dot(dims, abs(sampledRightBake)), 0.01);
    float halfHeight = max(dot(dims, abs(sampledUpBake)), 0.01);
    halfDepth = max(dot(dims, abs(sampledDirToCameraBake)), 0.01);

    float frameX = dot(relBake, sampledRightBake);
    float frameY = dot(relBake, sampledUpBake);
    float3 frameBakePos = sampledRightBake * frameX + sampledUpBake * frameY;

    uv = float2(frameX / (2.0 * halfWidth) + 0.5, 0.5 - frameY / (2.0 * halfHeight));
    frameWorldPos = center + mul(frameBakePos, instanceNormal);
    depthAxis = normalize(mul(sampledDirToCameraBake, instanceNormal));
}

void impostor_vs(
    in uint vertexId   : SV_VertexID,
    in uint instanceId : SV_InstanceID,

    out V2P o)
{
    uint persistentId = impostorVisBuf[impostorSlotOffsets[rc.assetIndex] + instanceId];
    CullInstanceData cull = cullData[persistentId];
    InstanceRenderData inst = instBuf[persistentId];

    float3 dims = max(assetDims[rc.assetIndex].xyz, float3(0.01, 0.01, 0.01));

    float3 worldCenter = (cull.bbox.min + cull.bbox.max) * 0.5;
    float3 dirToCameraWorld = normalize(cameraPos - worldCenter);
    // inverses the model matrix transformations
    float3 dirToCameraBake = normalize(mul(dirToCameraWorld, transpose(inst.normal)));

    uint3 viewIndices;
    float3 weights;
    SelectImpostorViews(dirToCameraBake, viewIndices, weights);

    // Construct a viewpoint-oriented basis that prevents roll and faces the camera directly
    float3 worldUp = float3(0.0, 1.0, 0.0);
    float3 horizontal = float3(dirToCameraWorld.x, 0.0, dirToCameraWorld.z);
    float horizontalLen = length(horizontal);

    float3 camRightInWorld, camUpInWorld;
    ViewBasis(dirToCameraWorld, camRightInWorld, camUpInWorld);
    float3 worldHalfExt = (cull.bbox.max - cull.bbox.min) * 0.5;


    float cardHalfWidth = max(dot(worldHalfExt, abs(camRightInWorld)), 0.01);
    float cardHalfHeight = max(dot(worldHalfExt, abs(camUpInWorld)), 0.01);

    float2 corner;
    corner.x = (vertexId == 1 || vertexId == 3) ? 1.0 : -1.0;
    corner.y = (vertexId >= 2) ? 1.0 : -1.0;

    float3 worldPos = worldCenter
        + camRightInWorld  * (corner.x * cardHalfWidth)
        + camUpInWorld     * (corner.y * cardHalfHeight);

    float3 vertWorld = worldPos - worldCenter;
    float3 vertBake = mul(vertWorld, transpose(inst.normal));

    float2 uv0;
    float2 uv1;
    float2 uv2;
    float3 frameWorldPos0;
    float3 frameWorldPos1;
    float3 frameWorldPos2;
    float3 depthAxis0;
    float3 depthAxis1;
    float3 depthAxis2;
    float halfDepth0;
    float halfDepth1;
    float halfDepth2;
    ProjectImpostorFrame(viewIndices.x, dims, worldCenter, vertBake, inst.normal, uv0, frameWorldPos0, depthAxis0, halfDepth0);
    ProjectImpostorFrame(viewIndices.y, dims, worldCenter, vertBake, inst.normal, uv1, frameWorldPos1, depthAxis1, halfDepth1);
    ProjectImpostorFrame(viewIndices.z, dims, worldCenter, vertBake, inst.normal, uv2, frameWorldPos2, depthAxis2, halfDepth2);

    o.pos          = mul(float4(worldPos, 1.0), viewProj);
    o.frameWorldPos0 = frameWorldPos0;
    o.frameWorldPos1 = frameWorldPos1;
    o.frameWorldPos2 = frameWorldPos2;
    o.depthAxis0    = depthAxis0;
    o.depthAxis1    = depthAxis1;
    o.depthAxis2    = depthAxis2;
    o.uv0           = uv0;
    o.uv1           = uv1;
    o.uv2           = uv2;
    o.weights       = weights;
    o.halfDepths    = float3(halfDepth0, halfDepth1, halfDepth2);
    o.viewIndex0    = viewIndices.x;
    o.viewIndex1    = viewIndices.y;
    o.viewIndex2    = viewIndices.z;
    o.persistentId = persistentId;
}

float SampleShadowCascade(float3 worldPos, uint cascadeIdx)
{
    float4 posLS = mul(float4(worldPos, 1.0), lightViewProj[cascadeIdx]);
    float2 shadowUV = posLS.xy * float2(0.5, -0.5) + 0.5;

    uint width, height, elements;
    t_ShadowMap.GetDimensions(width, height, elements);
    float2 texelSize = 1.0 / float2(width, height);

    float shadow = 0.0;

    static const float weights[3][3] = {
        { 1.0, 2.0, 1.0 },
        { 2.0, 4.0, 2.0 },
        { 1.0, 2.0, 1.0 }
    };

    [unroll]
    for (int x = -1; x <= 1; ++x)
    {
        [unroll]
        for (int y = -1; y <= 1; ++y)
        {
            float2 offset = float2(x, y) * texelSize;
            float weight = weights[x + 1][y + 1];
            shadow += weight * t_ShadowMap.SampleCmpLevelZero(
                s_ShadowSampler,
                float3(shadowUV + offset, float(cascadeIdx)),
                posLS.z);
        }
    }
    return shadow / 16.0;
}

void impostor_ps(
    in V2P i,
    out float4 o_color : SV_Target0,
    out float o_depth : SV_Depth)
{
    uint viewCount = XYLEM_IMPOSTOR_AZIMUTH_VIEWS * XYLEM_IMPOSTOR_ELEVATION_VIEWS;
    uint slice0 = rc.assetIndex * viewCount + i.viewIndex0;
    uint slice1 = rc.assetIndex * viewCount + i.viewIndex1;
    uint slice2 = rc.assetIndex * viewCount + i.viewIndex2;

    float4 albedoAlpha0 = t_ImpostorAlbedoAlpha.Sample(s_Sampler, float3(i.uv0, (float)slice0));
    float4 albedoAlpha1 = t_ImpostorAlbedoAlpha.Sample(s_Sampler, float3(i.uv1, (float)slice1));
    float4 albedoAlpha2 = t_ImpostorAlbedoAlpha.Sample(s_Sampler, float3(i.uv2, (float)slice2));

    float3 alphaWeights = i.weights * float3(albedoAlpha0.a, albedoAlpha1.a, albedoAlpha2.a);
    float alphaWeightSum = max(alphaWeights.x + alphaWeights.y + alphaWeights.z, 1e-6);
    float blendedAlpha = dot(i.weights, float3(albedoAlpha0.a, albedoAlpha1.a, albedoAlpha2.a));
    clip(blendedAlpha - impostorAlphaClip);

    // The bake target clears transparent texels to black. Linear filtering near
    // silhouettes therefore returns premultiplied-looking edge samples; divide
    // by alpha before the cross-frame blend so black background cannot fringe.
    float3 albedo0 = saturate(albedoAlpha0.rgb / max(albedoAlpha0.a, 1e-4));
    float3 albedo1 = saturate(albedoAlpha1.rgb / max(albedoAlpha1.a, 1e-4));
    float3 albedo2 = saturate(albedoAlpha2.rgb / max(albedoAlpha2.a, 1e-4));
    float3 blendedAlbedo =
        (albedo0 * alphaWeights.x +
         albedo1 * alphaWeights.y +
         albedo2 * alphaWeights.z) / alphaWeightSum;

    float impostorDepth0 = t_ImpostorDepth.SampleLevel(s_DepthSampler, float3(i.uv0, (float)slice0), 0.0);
    float impostorDepth1 = t_ImpostorDepth.SampleLevel(s_DepthSampler, float3(i.uv1, (float)slice1), 0.0);
    float impostorDepth2 = t_ImpostorDepth.SampleLevel(s_DepthSampler, float3(i.uv2, (float)slice2), 0.0);
    float3 depthWorldPos0 = i.frameWorldPos0 + normalize(i.depthAxis0) * ((0.5 - impostorDepth0) * (2.0 * i.halfDepths.x));
    float3 depthWorldPos1 = i.frameWorldPos1 + normalize(i.depthAxis1) * ((0.5 - impostorDepth1) * (2.0 * i.halfDepths.y));
    float3 depthWorldPos2 = i.frameWorldPos2 + normalize(i.depthAxis2) * ((0.5 - impostorDepth2) * (2.0 * i.halfDepths.z));
    float3 depthWorldPos =
        (depthWorldPos0 * alphaWeights.x +
         depthWorldPos1 * alphaWeights.y +
         depthWorldPos2 * alphaWeights.z) / alphaWeightSum;

    float3 objectNormal0 = t_ImpostorNormal.Sample(s_Sampler, float3(i.uv0, (float)slice0)).rgb * 2.0 - 1.0;
    float3 objectNormal1 = t_ImpostorNormal.Sample(s_Sampler, float3(i.uv1, (float)slice1)).rgb * 2.0 - 1.0;
    float3 objectNormal2 = t_ImpostorNormal.Sample(s_Sampler, float3(i.uv2, (float)slice2)).rgb * 2.0 - 1.0;
    float3 objectNormal = normalize(
        (objectNormal0 * alphaWeights.x +
         objectNormal1 * alphaWeights.y +
         objectNormal2 * alphaWeights.z) / alphaWeightSum);
    float3 worldNormal = normalize(mul(objectNormal, instBuf[i.persistentId].normal));
    float3 lightDir = -normalize(sunLightDir);
    // Wrap diffuse: softens the back-side falloff so cross-billboard leaves don't band
    // sharply when tilted away from the sun. Must stay in lockstep with the leaf
    // pixel shaders so impostor and runtime canopies look identical at the LOD swap.
    float diffuse = saturate(dot(worldNormal, lightDir) * 0.5 + 0.5);

    float viewZ = mul(float4(depthWorldPos, 1.0), viewMatrix).z;

    uint cascadeIdx = 3;
    if      (viewZ < cascadeSplits.x) cascadeIdx = 0;
    else if (viewZ < cascadeSplits.y) cascadeIdx = 1;
    else if (viewZ < cascadeSplits.z) cascadeIdx = 2;

    float notInShadow = SampleShadowCascade(depthWorldPos, cascadeIdx);
    float lighting = XYLEM_TREE_AMBIENT + (1.0 - XYLEM_TREE_AMBIENT) * diffuse * notInShadow;

    // Keep depth writes stable by using the card depth. Writing the baked
    // per-pixel depth exposes atlas-resolution quantization as visible bands.
    // Apply a tiny shader-side bias because SV_Depth bypasses raster depth bias.

float3 toCamera = normalize(cameraPos - depthWorldPos);
float worldSpaceBias = 0.1; 
float3 biasedWorldPos = depthWorldPos; // + toCamera * worldSpaceBias;

float4 clipPos = mul(float4(biasedWorldPos, 1.0), viewProj);
float projectedDepth = clipPos.z / clipPos.w;

#if XYLEM_USE_REVERSE_Z
    o_depth = projectedDepth;
#else
    o_depth = projectedDepth;
#endif
    o_color = float4(blendedAlbedo * lighting, blendedAlpha);
}
