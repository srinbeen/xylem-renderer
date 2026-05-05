#pragma pack_matrix(row_major)

#include "../include/macros.h"
#include "types.hlsli"
#include "ShaderRegisterMap.hlsli"

cbuffer CB : register(XY_REG_B_COMPUTE_CULL_CB_FRAME)
{
    // Prefix — matches Render::ConstantBufferEntry / CullConstantBufferEntry
    float4x4 viewProj;
    float4x4 viewMatrix;
    float4x4 lightViewProj[XYLEM_NUM_CASCADES];
    float3   sunLightDir;
    float    _pad0;
    float4   cascadeSplits;

    // Cull fields
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

    // Hi-Z fields
    float2   hizDimensions;
    float    maxHiZMip;
    uint     hizEnabled;
    float    impostorAlphaClip;
    float3   _pad2;
};

struct CullInstanceData
{
    box3   bbox;
    uint   baseSlot;   // treeId * numLODs
    uint   regionId;   // index into regionVisible

    uint   active;     // 1 = live, 0 = dead
};

struct CullRegionData
{
    box3   bbox;
};

StructuredBuffer<CullRegionData>   regionData           : register(XY_REG_T_COMPUTE_CULL_SRV_REGION_DATA);
StructuredBuffer<CullInstanceData> instanceData         : register(XY_REG_T_COMPUTE_CULL_SRV_INSTANCE_DATA);
StructuredBuffer<uint>             mainSlotOffsets      : register(XY_REG_T_COMPUTE_CULL_SRV_MAIN_SLOT_OFFSETS);
StructuredBuffer<uint>             shadowSlotOffsets    : register(XY_REG_T_COMPUTE_CULL_SRV_SHADOW_SLOT_OFFSETS);
StructuredBuffer<uint>             impostorSlotOffsets  : register(XY_REG_T_COMPUTE_CULL_SRV_IMPOSTOR_SLOT_OFFSETS);

RWStructuredBuffer<uint>           mainRegionVisBuf     : register(XY_REG_U_COMPUTE_CULL_UAV_MAIN_REGION_VIS);
RWStructuredBuffer<uint>           mainSlotCountBuf     : register(XY_REG_U_COMPUTE_CULL_UAV_MAIN_SLOT_COUNT);
RWStructuredBuffer<uint>           mainVisBuf           : register(XY_REG_U_COMPUTE_CULL_UAV_MAIN_VIS);
RWStructuredBuffer<uint>           shadowSlotCountBuf   : register(XY_REG_U_COMPUTE_CULL_UAV_SHADOW_SLOT_COUNT);
RWStructuredBuffer<uint>           shadowVisBuf         : register(XY_REG_U_COMPUTE_CULL_UAV_SHADOW_VIS);

RWByteAddressBuffer                mainIndirectArgs     : register(XY_REG_U_COMPUTE_CULL_UAV_MAIN_INDIRECT_ARGS);
RWByteAddressBuffer                shadowIndirectArgs   : register(XY_REG_U_COMPUTE_CULL_UAV_SHADOW_INDIRECT_ARGS);
RWByteAddressBuffer                shadowUniqueCounter  : register(XY_REG_U_COMPUTE_CULL_UAV_SHADOW_UNIQUE_COUNTER);
RWStructuredBuffer<uint>           impostorSlotCountBuf : register(XY_REG_U_COMPUTE_CULL_UAV_IMPOSTOR_COUNT);
RWStructuredBuffer<uint>           impostorVisBuf       : register(XY_REG_U_COMPUTE_CULL_UAV_IMPOSTOR_VIS);
RWByteAddressBuffer                impostorIndirectArgs : register(XY_REG_U_COMPUTE_CULL_UAV_IMPOSTOR_INDIRECT_ARGS);
RWByteAddressBuffer                leafMainIndirectArgs : register(u11);
RWByteAddressBuffer                leafShadowIndirectArgs : register(u12);

Texture2D<float4>                  hizTexture           : register(XY_REG_T_COMPUTE_CULL_SRV_HI_Z);

bool DoesAABBIntersectFrustum(box3 bbox, frustum f);
uint SelectLOD(box3 bbox);
bool SelectImpostor(box3 bbox);
bool IsOccludedByHiZ(box3 bbox);
float FarthestHiZDepth(float a, float b);
float LoadHiZFarthestForRect(float2 minUV, float2 maxUV, float mipLevel);

[numthreads(64, 1, 1)]
void CullRegion(uint3 dtid : SV_DispatchThreadID)
{
    uint idx = dtid.x;
    if (idx >= numRegions) return;

    mainRegionVisBuf[idx] = (uint)DoesAABBIntersectFrustum(regionData[idx].bbox, viewFrustum);
}

[numthreads(256, 1, 1)]
void CullMain(uint3 dtid : SV_DispatchThreadID)
{
    uint idx = dtid.x;
    // outside region windows
    if (idx >= totalCapacity) return;

    CullInstanceData inst = instanceData[idx];
    // not a real instance, but in allocated memory
    if (!inst.active) return;
    // region culled
    if (!mainRegionVisBuf[inst.regionId]) return;
    // instance frustum culled
    if (!DoesAABBIntersectFrustum(inst.bbox, viewFrustum)) return;
    // hi-z occlusion
    if (IsOccludedByHiZ(inst.bbox)) return;

    if (SelectImpostor(inst.bbox))
    {
        uint ai = inst.baseSlot / numLods;

        uint writeIdx;
        InterlockedAdd(impostorSlotCountBuf[ai], 1, writeIdx);
        impostorVisBuf[impostorSlotOffsets[ai] + writeIdx] = idx;

        uint dummy;
        impostorIndirectArgs.InterlockedAdd(ai * 16 + 4, 1, dummy);
        return;
    }

    uint lod  = SelectLOD(inst.bbox);
    uint slot = inst.baseSlot + lod;

    uint writeIdx;
    InterlockedAdd(mainSlotCountBuf[slot], 1, writeIdx);
    uint writePos = mainSlotOffsets[slot] + writeIdx;
    mainVisBuf[writePos] = idx;

    // Atomically increment instanceCount in indirect draw args for this slot.
    // DrawIndexedIndirectArguments layout (20 bytes per slot):
    //   offset 0: indexCount, offset 4: instanceCount, offset 8: startIndexLocation, ...
    uint dummy;
    mainIndirectArgs.InterlockedAdd(slot * 20 + 4, 1, dummy);
    leafMainIndirectArgs.InterlockedAdd(slot * 16 + 4, 1, dummy);
}

[numthreads(256, 1, 1)]
void CullShadow(uint3 dtid : SV_DispatchThreadID)
{
    uint idx = dtid.x;
    if (idx >= totalCapacity) return;

    CullInstanceData inst = instanceData[idx];
    if (!inst.active) return;

    // Compute instance world-space AABB in light space
    float3 cornersWS[8] = {
        float3(inst.bbox.min.x, inst.bbox.min.y, inst.bbox.min.z),
        float3(inst.bbox.max.x, inst.bbox.min.y, inst.bbox.min.z),
        float3(inst.bbox.min.x, inst.bbox.max.y, inst.bbox.min.z),
        float3(inst.bbox.max.x, inst.bbox.max.y, inst.bbox.min.z),
        float3(inst.bbox.min.x, inst.bbox.min.y, inst.bbox.max.z),
        float3(inst.bbox.max.x, inst.bbox.min.y, inst.bbox.max.z),
        float3(inst.bbox.min.x, inst.bbox.max.y, inst.bbox.max.z),
        float3(inst.bbox.max.x, inst.bbox.max.y, inst.bbox.max.z),
    };

    float3 bboxMinLS = float3( 1e30,  1e30,  1e30);
    float3 bboxMaxLS = float3(-1e30, -1e30, -1e30);
    [unroll]
    for (int i = 0; i < 8; i++)
    {
        float3 ls = mul(float4(cornersWS[i], 1), worldToLight).xyz;
        bboxMinLS = min(bboxMinLS, ls);
        bboxMaxLS = max(bboxMaxLS, ls);
    }

    uint ai = inst.baseSlot / numLods;

    bool anyCascadeVisible = false;

    [unroll]
    for (uint c = 0; c < XYLEM_NUM_CASCADES; c++)
    {
        float3 cMin = shadowCasterMinLS[c].xyz;
        float3 cMax = shadowCasterMaxLS[c].xyz;

        // Empty cascade AABB (mins > maxs) → skip
        if (any(cMin > cMax)) continue;

        // AABB-vs-AABB intersection test in light space
        if (any(bboxMinLS > cMax) || any(bboxMaxLS < cMin)) continue;

        uint slot = ai * XYLEM_NUM_CASCADES + c;

        uint writeIdx;
        InterlockedAdd(shadowSlotCountBuf[slot], 1, writeIdx);
        shadowVisBuf[shadowSlotOffsets[slot] + writeIdx] = idx;

        // Atomically increment instanceCount in shadow indirect draw args for this slot.
        uint dummy;
        shadowIndirectArgs.InterlockedAdd(slot * 20 + 4, 1, dummy);
        leafShadowIndirectArgs.InterlockedAdd(slot * 16 + 4, 1, dummy);

        anyCascadeVisible = true;
    }

    // Count each instance once if it contributes to any cascade (for UI stats
    // that need unique shadow-caster count, not summed per-cascade overdraw).
    if (anyCascadeVisible)
    {
        uint dummy;
        shadowUniqueCounter.InterlockedAdd(0, 1, dummy);
    }
}

bool DoesAABBIntersectFrustum(box3 bbox, frustum f)
{
    [unroll]
    for (int i = 0; i < 6; i++)
    {
        float3 n = f[i].n;
        float3 p;
        p.x = (n.x > 0) ? bbox.min.x : bbox.max.x;
        p.y = (n.y > 0) ? bbox.min.y : bbox.max.y;
        p.z = (n.z > 0) ? bbox.min.z : bbox.max.z;
        if (dot(n, p) > f[i].d) return false;
    }
    return true;
}

uint SelectLOD(box3 bbox)
{
    float3 nearest = clamp(cameraPos, bbox.min, bbox.max);
    float dist = distance(cameraPos, nearest);

    for (uint i = 0; i < numLods; i++)
    {
        if (dist < lodDistances[i])
            return i;
    }
    return numLods - 1;
}

bool SelectImpostor(box3 bbox)
{
    float3 nearest = clamp(cameraPos, bbox.min, bbox.max);
    float dist = distance(cameraPos, nearest);
    return dist >= lodDistances[numLods - 1].x;
}

bool IsOccludedByHiZ(box3 bbox)
{
    // Disabled by CPU (bird's-eye view or frame 0)
    if (!hizEnabled) return false;

    float3 corners[8] = {
        float3(bbox.min.x, bbox.min.y, bbox.min.z),
        float3(bbox.max.x, bbox.min.y, bbox.min.z),
        float3(bbox.min.x, bbox.max.y, bbox.min.z),
        float3(bbox.max.x, bbox.max.y, bbox.min.z),
        float3(bbox.min.x, bbox.min.y, bbox.max.z),
        float3(bbox.max.x, bbox.min.y, bbox.max.z),
        float3(bbox.min.x, bbox.max.y, bbox.max.z),
        float3(bbox.max.x, bbox.max.y, bbox.max.z),
    };

    float2 minUV = float2(1, 1);
    float2 maxUV = float2(0, 0);
#if XYLEM_USE_REVERSE_Z
    float closestDepth = 0.0;   // far plane in reverse-Z
#else
    float closestDepth = 1.0;   // far plane in forward-Z
#endif

    // closestDepth: AABB point closest to near plane
    [unroll]
    for (int i = 0; i < 8; i++)
    {
        float4 clip = mul(float4(corners[i], 1), viewProj);
        // AABB straddles or is behind near plane — treat as visible
        if (clip.w <= 0.0) return false;

        float3 ndc = clip.xyz / clip.w;
        float2 uv = saturate(ndc.xy * float2(0.5, -0.5) + 0.5);

        minUV = min(minUV, uv);
        maxUV = max(maxUV, uv);
#if XYLEM_USE_REVERSE_Z
        closestDepth = max(closestDepth, ndc.z);
#else
        closestDepth = min(closestDepth, ndc.z);
#endif
    }

    // Pick mip level based on projected footprint in pixels
    float2 footprint = (maxUV - minUV) * hizDimensions;
    float mipLevel = ceil(log2(max(footprint.x, footprint.y)));
    mipLevel = clamp(mipLevel, 0, maxHiZMip);

    float hiz = LoadHiZFarthestForRect(minUV, maxUV, mipLevel);

    // Occlusion test
#if XYLEM_USE_REVERSE_Z
    return (closestDepth < hiz);
#else
    return (closestDepth > hiz);
#endif
}

float FarthestHiZDepth(float a, float b)
{
#if XYLEM_USE_REVERSE_Z
    return min(a, b);
#else
    return max(a, b);
#endif
}

float LoadHiZFarthestForRect(float2 minUV, float2 maxUV, float mipLevel)
{
    uint mip = (uint)mipLevel;
    uint mipWidth, mipHeight, mipCount;
    hizTexture.GetDimensions(mip, mipWidth, mipHeight, mipCount);

    uint2 lastTexel = uint2(mipWidth - 1u, mipHeight - 1u);
    float2 mipDims = float2((float)mipWidth, (float)mipHeight);

    uint2 lo = min((uint2)floor(minUV * mipDims), lastTexel);
    uint2 hi = min((uint2)floor(maxUV * mipDims), lastTexel);

    float hiz = hizTexture.Load(int3(lo.x, lo.y, mip)).r;

    if (hi.x != lo.x)
        hiz = FarthestHiZDepth(hiz, hizTexture.Load(int3(hi.x, lo.y, mip)).r);
    if (hi.y != lo.y)
        hiz = FarthestHiZDepth(hiz, hizTexture.Load(int3(lo.x, hi.y, mip)).r);
    if ((hi.x != lo.x) && (hi.y != lo.y))
        hiz = FarthestHiZDepth(hiz, hizTexture.Load(int3(hi.x, hi.y, mip)).r);

    return hiz;
}
