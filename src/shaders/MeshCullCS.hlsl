#pragma pack_matrix(row_major)

#include "../include/macros.h"
#include "types.hlsli"

// MeshShader pipeline cull — fork of CullCS.hlsl that writes
// D3D12_DISPATCH_MESH_ARGUMENTS (with a slotIdx prefix) instead of
// DrawIndexedIndirectArguments. Shadow cull is not included here (not bound
// on the mesh-shader path in this PR).

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
    float4   lodDistances;

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
StructuredBuffer<uint>             invocationsPerSlot   : register(t3);

RWStructuredBuffer<uint>           mainRegionVisBuf     : register(u0);
RWStructuredBuffer<uint>           mainSlotCountBuf     : register(u1);
RWStructuredBuffer<uint>           mainVisBuf           : register(u2);

// Records of 16 bytes each: { uint slotIdx; uint groupsX; uint groupsY; uint groupsZ; }
// The slotIdx is consumed by the command signature's CONSTANT argument;
// the remaining three uints are the DISPATCH_MESH_ARGUMENTS payload.
RWByteAddressBuffer                meshDispatchArgs     : register(u3);

Texture2D<float2>                  hizTexture           : register(t4);
SamplerState                       hizSampler           : register(s0);

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
    // if (IsOccludedByHiZ(inst.bbox)) return;

    uint lod  = SelectLOD(inst.bbox);
    uint slot = inst.baseSlot + lod;

    uint writeIdx;
    InterlockedAdd(mainSlotCountBuf[slot], 1, writeIdx);
    uint writePos = mainSlotOffsets[slot] + writeIdx;
    mainVisBuf[writePos] = idx;

    // Each surviving instance contributes invocationsPerSlot[slot] AS threadgroups
    // (one AS invocation covers up to XYLEM_AS_GROUP_SIZE meshlets of one instance).
    // Atomically add that into groupsX at offset slot*16 + 4 (skipping slotIdx at +0).
    uint dummy;
    meshDispatchArgs.InterlockedAdd(slot * 16 + 4, invocationsPerSlot[slot], dummy);
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
