#pragma pack_matrix(row_major)

#include "../include/macros.h"
#include "types.hlsli"

static const uint NUM_CASCADES = 4;

cbuffer CB : register(b0)
{
    // Prefix — matches Render::ConstantBufferEntry / CullConstantBufferEntry
    float4x4 viewProj;
    float4x4 viewMatrix;
    float4x4 lightViewProj[NUM_CASCADES];
    float3   sunLightDir;
    float    _pad0;
    float4   cascadeSplits;

    // Cull fields
    frustum  viewFrustum;
    float4x4 worldToLight;
    float4   shadowCasterMinLS[NUM_CASCADES];
    float4   shadowCasterMaxLS[NUM_CASCADES];

    float3   cameraPos;
    uint     numRegions;
    uint     totalCapacity;
    uint     numLods;
    float    _pad1a;
    float    _pad1b;
    float4   lodDistances[3];

    // Hi-Z fields
    float2   hizDimensions;
    float    maxHiZMip;
    uint     hizEnabled;
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

StructuredBuffer<CullRegionData>   regionData           : register(t0);
StructuredBuffer<CullInstanceData> instanceData         : register(t1);
StructuredBuffer<uint>             mainSlotOffsets      : register(t2);
StructuredBuffer<uint>             shadowSlotOffsets    : register(t3);

RWStructuredBuffer<uint>           mainRegionVisBuf     : register(u0);
RWStructuredBuffer<uint>           mainSlotCountBuf     : register(u1);
RWStructuredBuffer<uint>           mainVisBuf           : register(u2);
RWStructuredBuffer<uint>           shadowSlotCountBuf   : register(u3);
RWStructuredBuffer<uint>           shadowVisBuf         : register(u4);

RWByteAddressBuffer                mainIndirectArgs     : register(u5);
RWByteAddressBuffer                shadowIndirectArgs   : register(u6);
RWByteAddressBuffer                shadowUniqueCounter  : register(u7);

Texture2D<float2>                  hizTexture           : register(t4);
SamplerState                       hizSampler           : register(s0);

bool DoesAABBIntersectFrustum(box3 bbox, frustum f);
uint SelectLOD(box3 bbox);
bool IsOccludedByHiZ(box3 bbox);


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
    // Hi-Z occlusion culled
    if (IsOccludedByHiZ(inst.bbox)) return;

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
    for (uint c = 0; c < NUM_CASCADES; c++)
    {
        float3 cMin = shadowCasterMinLS[c].xyz;
        float3 cMax = shadowCasterMaxLS[c].xyz;

        // Empty cascade AABB (mins > maxs) → skip
        if (any(cMin > cMax)) continue;

        // AABB-vs-AABB intersection test in light space
        if (any(bboxMinLS > cMax) || any(bboxMaxLS < cMin)) continue;

        uint slot = ai * NUM_CASCADES + c;

        uint writeIdx;
        InterlockedAdd(shadowSlotCountBuf[slot], 1, writeIdx);
        shadowVisBuf[shadowSlotOffsets[slot] + writeIdx] = idx;

        // Atomically increment instanceCount in shadow indirect draw args for this slot.
        uint dummy;
        shadowIndirectArgs.InterlockedAdd(slot * 20 + 4, 1, dummy);

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
        if (dist < lodDistances[i].x)
            return i;
    }
    return numLods - 1;
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

    [unroll]
    for (int i = 0; i < 8; i++)
    {
        float4 clip = mul(float4(corners[i], 1), viewProj);
        // AABB straddles or is behind near plane — treat as visible
        if (clip.w <= 0.0) return false;

        float3 ndc = clip.xyz / clip.w;
        float2 uv = ndc.xy * 0.5 + 0.5;
        uv.y = 1.0 - uv.y;

        minUV = min(minUV, uv);
        maxUV = max(maxUV, uv);
#if XYLEM_USE_REVERSE_Z
        closestDepth = max(closestDepth, ndc.z);  // max = nearest in reverse-Z
#else
        closestDepth = min(closestDepth, ndc.z);  // min = nearest in forward-Z
#endif
    }

    // Clamp to screen bounds
    minUV = saturate(minUV);
    maxUV = saturate(maxUV);

    // Pick mip level based on projected footprint in pixels
    float2 footprint = (maxUV - minUV) * hizDimensions;
    float mipLevel = ceil(log2(max(footprint.x, footprint.y)));
    mipLevel = clamp(mipLevel, 0, maxHiZMip);

    // Sample Hi-Z at center of projected rect (.r = farthest-depth reduction)
    float2 centerUV = (minUV + maxUV) * 0.5;
    float hizDepth = hizTexture.SampleLevel(hizSampler, centerUV, mipLevel).r;

    // Occlusion test
#if XYLEM_USE_REVERSE_Z
    // Reverse-Z: object's nearest depth (large value) < Hi-Z (nearest occluder, large value)
    // means object is behind the occluder
    return (closestDepth < hizDepth);
#else
    // Forward-Z: object's nearest depth (small value) > Hi-Z (nearest occluder, small value)
    return (closestDepth > hizDepth);
#endif
}
