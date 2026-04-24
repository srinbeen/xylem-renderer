#pragma pack_matrix(row_major)

#include "../include/macros.h"
#include "types.hlsli"
#include "meshlet_types.hlsli"

// MeshShader pipeline cull — writes D3D12_DISPATCH_MESH_ARGUMENTS records
// (with a slotIdx prefix) for main + shadow passes.

static const uint NUM_CASCADES = 4;

void UpdateDispatchMeshArgs(RWByteAddressBuffer argsBuffer, uint slot, uint totalGroups);

cbuffer CB : register(b0)
{
    float4x4 viewProj;
    float4x4 viewMatrix;
    float4x4 lightViewProj[NUM_CASCADES];
    float3   sunLightDir;
    float    _pad0;
    float4   cascadeSplits;

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
    float4   lodDistances;

    float2   hizDimensions;
    float    maxHiZMip;
    uint     hizEnabled;
};

struct CullInstanceData
{
    box3   bbox;
    uint   baseSlot;
    uint   regionId;
    uint   active;
};

struct CullRegionData
{
    box3   bbox;
};

StructuredBuffer<CullRegionData>   regionData             : register(t0);
StructuredBuffer<CullInstanceData> instanceData           : register(t1);
StructuredBuffer<uint>             mainSlotOffsets        : register(t2);
StructuredBuffer<uint>             mainInvocsPerSlot      : register(t3);
StructuredBuffer<uint>             shadowSlotOffsets      : register(t4);
StructuredBuffer<uint>             shadowInvocsPerSlot    : register(t5);

RWStructuredBuffer<uint>           mainRegionVisBuf       : register(u0);
RWStructuredBuffer<uint>           mainSlotCountBuf       : register(u1);
RWStructuredBuffer<uint>           mainVisBuf             : register(u2);
RWByteAddressBuffer                mainDispatchArgs       : register(u3);
RWStructuredBuffer<uint>           shadowSlotCountBuf     : register(u4);
RWStructuredBuffer<uint>           shadowVisBuf           : register(u5);
RWByteAddressBuffer                shadowDispatchArgs     : register(u6);
RWByteAddressBuffer                shadowUniqueCounter    : register(u7);

Texture2D<float2>                  hizTexture             : register(t6);
SamplerState                       hizSampler             : register(s0);

bool DoesAABBIntersectFrustum(box3 bbox, frustum f);
uint SelectLOD(box3 bbox);
bool IsOccludedByHiZ(box3 bbox);

[numthreads(64, 1, 1)]
void MeshCullRegion(uint3 dtid : SV_DispatchThreadID)
{
    uint idx = dtid.x;
    if (idx >= numRegions) return;
    mainRegionVisBuf[idx] = (uint)DoesAABBIntersectFrustum(regionData[idx].bbox, viewFrustum);
}

[numthreads(256, 1, 1)]
void MeshCullMain(uint3 dtid : SV_DispatchThreadID)
{
    uint idx = dtid.x;
    if (idx >= totalCapacity) return;

    CullInstanceData inst = instanceData[idx];
    if (!inst.active) return;
    if (!mainRegionVisBuf[inst.regionId]) return;
    if (!DoesAABBIntersectFrustum(inst.bbox, viewFrustum)) return;
    if (IsOccludedByHiZ(inst.bbox)) return;

    uint lod  = SelectLOD(inst.bbox);
    uint slot = inst.baseSlot + lod;

    uint writeIdx;
    InterlockedAdd(mainSlotCountBuf[slot], 1, writeIdx);
    mainVisBuf[mainSlotOffsets[slot] + writeIdx] = idx;

    // DISPATCH_MESH record: [slotIdx | groupsX | groupsY | groupsZ] = 16 bytes
    UpdateDispatchMeshArgs(mainDispatchArgs, slot, (writeIdx + 1u) * mainInvocsPerSlot[slot]);
}

[numthreads(256, 1, 1)]
void MeshCullShadow(uint3 dtid : SV_DispatchThreadID)
{
    uint idx = dtid.x;
    if (idx >= totalCapacity) return;

    CullInstanceData inst = instanceData[idx];
    if (!inst.active) return;

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
    bool anyCascade = false;

    [unroll]
    for (uint c = 0; c < NUM_CASCADES; c++)
    {
        float3 cMin = shadowCasterMinLS[c].xyz;
        float3 cMax = shadowCasterMaxLS[c].xyz;
        if (any(cMin > cMax)) continue;
        if (any(bboxMinLS > cMax) || any(bboxMaxLS < cMin)) continue;

        uint slot = ai * NUM_CASCADES + c;
        uint writeIdx;
        InterlockedAdd(shadowSlotCountBuf[slot], 1, writeIdx);
        shadowVisBuf[shadowSlotOffsets[slot] + writeIdx] = idx;

        UpdateDispatchMeshArgs(shadowDispatchArgs, slot, (writeIdx + 1u) * shadowInvocsPerSlot[slot]);
        anyCascade = true;
    }

    if (anyCascade)
    {
        uint dummy;
        shadowUniqueCounter.InterlockedAdd(0, 1, dummy);
    }
}

void UpdateDispatchMeshArgs(RWByteAddressBuffer argsBuffer, uint slot, uint totalGroups)
{
    uint groupsX = min(totalGroups, XYLEM_DISPATCH_X);
    uint groupsY = min((totalGroups + XYLEM_DISPATCH_X - 1u) / XYLEM_DISPATCH_X, XYLEM_DISPATCH_X);

    uint dummy;
    argsBuffer.InterlockedMax(slot * 16 + 4, groupsX, dummy);
    argsBuffer.InterlockedMax(slot * 16 + 8, max(groupsY, 1u), dummy);
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

bool IsOccludedByHiZ(box3 bbox)
{
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
    float closestDepth = 0.0;
#else
    float closestDepth = 1.0;
#endif

    [unroll]
    for (int i = 0; i < 8; i++)
    {
        float4 clip = mul(float4(corners[i], 1), viewProj);
        if (clip.w <= 0.0) return false;
        float3 ndc = clip.xyz / clip.w;
        float2 uv = ndc.xy * 0.5 + 0.5;
        uv.y = 1.0 - uv.y;
        minUV = min(minUV, uv);
        maxUV = max(maxUV, uv);
#if XYLEM_USE_REVERSE_Z
        closestDepth = max(closestDepth, ndc.z);
#else
        closestDepth = min(closestDepth, ndc.z);
#endif
    }

    minUV = saturate(minUV);
    maxUV = saturate(maxUV);

    float2 footprint = (maxUV - minUV) * hizDimensions;
    float mipLevel = ceil(log2(max(footprint.x, footprint.y)));
    mipLevel = clamp(mipLevel, 0, maxHiZMip);

    float2 centerUV = (minUV + maxUV) * 0.5;
    float hizDepth = hizTexture.SampleLevel(hizSampler, centerUV, mipLevel).r;

#if XYLEM_USE_REVERSE_Z
    return (closestDepth < hizDepth);
#else
    return (closestDepth > hizDepth);
#endif
}
