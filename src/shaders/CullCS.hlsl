#pragma pack_matrix(row_major)

#include "types.hlsli"

cbuffer CB : register(b0)
{
    // P0 fields (also read by VS/PS)
    float4x4 view;
    float4x4 projection;
    float4x4 lightViewProj;
    float3   sunLightDir;
    float    _pad0;
    float3x4 _pad1;

    // Cull fields
    frustum   viewFrustum;
    float4x4  worldToLight;
    float3    shadowCasterMinLS;
    float     _pad2;
    float3    shadowCasterMaxLS;
    uint     numRegions;
    float3   cameraPos;
    uint     totalCapacity;
    float4   lodDistances[3];
    uint     numLods;
    float    _pad3[3];
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

bool DoesAABBIntersectFrustum(box3 bbox, frustum f);
uint SelectLOD(box3 bbox);


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

    if (any(shadowCasterMinLS > shadowCasterMaxLS)) return;

    // Transform world-space AABB corners into light space, compute light-space AABB
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

    // AABB-vs-AABB intersection test in light space
    if (any(bboxMinLS > shadowCasterMaxLS) || any(bboxMaxLS < shadowCasterMinLS))
        return;

    uint ai = inst.baseSlot / numLods;

    uint writeIdx;
    InterlockedAdd(shadowSlotCountBuf[ai], 1, writeIdx);
    shadowVisBuf[shadowSlotOffsets[ai] + writeIdx] = idx;

    // Atomically increment instanceCount in shadow indirect draw args for this asset.
    uint dummy;
    shadowIndirectArgs.InterlockedAdd(ai * 20 + 4, 1, dummy);
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