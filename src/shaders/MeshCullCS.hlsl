#pragma pack_matrix(row_major)

#include "../include/macros.h"
#include "types.hlsli"
#include "meshlet_types.hlsli"
#include "ShaderRegisterMap.hlsli"

void UpdateDispatchMeshArgs(RWByteAddressBuffer argsBuffer, uint slot, uint totalGroups);

// ---------------------------------------------------------------------------
// Wave helpers for cycling-ballot aggregation. WaveActiveBallot returns a
// uint4 mask covering up to 128 lanes. Current SM 6.5 hardware tops out at
// wave64, so .z/.w are always zero in practice — kept for portability.
// ---------------------------------------------------------------------------
uint WaveLeaderLaneFromBallot(uint4 ballot)
{
    if (ballot.x != 0u) return firstbitlow(ballot.x);
    if (ballot.y != 0u) return 32u + firstbitlow(ballot.y);
    if (ballot.z != 0u) return 64u + firstbitlow(ballot.z);
    return 96u + firstbitlow(ballot.w);
}

uint WaveMaskCountBits(uint4 mask)
{
    return countbits(mask.x) + countbits(mask.y) + countbits(mask.z) + countbits(mask.w);
}

// Count of set bits in `mask` strictly below the current lane index.
uint WaveMaskPrefixCountBits(uint4 mask)
{
    uint lane        = WaveGetLaneIndex();
    uint laneInWord  = lane & 31u;
    uint belowInWord = (laneInWord == 0u) ? 0u : ((1u << laneInWord) - 1u);

    if (lane < 32u)
        return countbits(mask.x & belowInWord);
    if (lane < 64u)
        return countbits(mask.x) + countbits(mask.y & belowInWord);
    if (lane < 96u)
        return countbits(mask.x) + countbits(mask.y) + countbits(mask.z & belowInWord);
    return countbits(mask.x) + countbits(mask.y) + countbits(mask.z) + countbits(mask.w & belowInWord);
}

cbuffer CB : register(XY_REG_B_MESH_CULL_CB_FRAME)
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
    uint     showShadowImpostors;
    float    shadowImpostorBias;
    float    _pad2;
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

StructuredBuffer<CullRegionData>   regionData               : register(XY_REG_T_MESH_CULL_SRV_REGION_DATA);
StructuredBuffer<CullInstanceData> instanceData             : register(XY_REG_T_MESH_CULL_SRV_INSTANCE_DATA);
StructuredBuffer<uint>             mainSlotOffsets          : register(XY_REG_T_MESH_CULL_SRV_MAIN_SLOT_OFFSETS);
StructuredBuffer<uint>             mainInvocsPerSlot        : register(XY_REG_T_MESH_CULL_SRV_MAIN_INVOCATIONS);
StructuredBuffer<uint>             shadowSlotOffsets        : register(XY_REG_T_MESH_CULL_SRV_SHADOW_SLOT_OFFSETS);
StructuredBuffer<uint>             shadowInvocsPerSlot      : register(XY_REG_T_MESH_CULL_SRV_SHADOW_INVOCATIONS);
StructuredBuffer<uint>             impostorSlotOffsets      : register(XY_REG_T_MESH_CULL_SRV_IMPOSTOR_SLOT_OFFSETS);
StructuredBuffer<uint>             mainLeafInvocsPerSlot    : register(XY_REG_T_MESH_CULL_SRV_MAIN_LEAF_INVOCATIONS);
StructuredBuffer<uint>             shadowLeafInvocsPerSlot  : register(XY_REG_T_MESH_CULL_SRV_SHADOW_LEAF_INVOCATIONS);
StructuredBuffer<uint>             shadowImpostorSlotOffsets : register(XY_REG_T_MESH_CULL_SRV_SHADOW_IMPOSTOR_SLOT_OFFSETS);

RWStructuredBuffer<uint>           mainRegionVisBuf       : register(XY_REG_U_MESH_CULL_UAV_MAIN_REGION_VIS);
RWStructuredBuffer<uint>           mainSlotCountBuf       : register(XY_REG_U_MESH_CULL_UAV_MAIN_COUNT);
RWStructuredBuffer<uint>           mainVisBuf             : register(XY_REG_U_MESH_CULL_UAV_MAIN_VIS);
RWByteAddressBuffer                mainDispatchArgs       : register(XY_REG_U_MESH_CULL_UAV_MAIN_DISPATCH);
RWStructuredBuffer<uint>           shadowSlotCountBuf     : register(XY_REG_U_MESH_CULL_UAV_SHADOW_COUNT);
RWStructuredBuffer<uint>           shadowVisBuf           : register(XY_REG_U_MESH_CULL_UAV_SHADOW_VIS);
RWByteAddressBuffer                shadowDispatchArgs     : register(XY_REG_U_MESH_CULL_UAV_SHADOW_DISPATCH);
RWByteAddressBuffer                shadowUniqueCounter    : register(XY_REG_U_MESH_CULL_UAV_SHADOW_UNIQUE);
RWStructuredBuffer<uint>           impostorSlotCountBuf   : register(XY_REG_U_MESH_CULL_UAV_IMPOSTOR_COUNT);
RWStructuredBuffer<uint>           impostorVisBuf         : register(XY_REG_U_MESH_CULL_UAV_IMPOSTOR_VIS);
RWByteAddressBuffer                impostorIndirectArgs   : register(XY_REG_U_MESH_CULL_UAV_IMPOSTOR_INDIRECT_ARGS);
RWByteAddressBuffer                mainLeafDispatchArgs   : register(XY_REG_U_MESH_CULL_UAV_MAIN_LEAF_DISPATCH);
RWByteAddressBuffer                shadowLeafDispatchArgs : register(XY_REG_U_MESH_CULL_UAV_SHADOW_LEAF_DISPATCH);

RWStructuredBuffer<uint>           shadowImpostorCountBuf    : register(XY_REG_U_MESH_CULL_UAV_SHADOW_IMPOSTOR_COUNT);
RWStructuredBuffer<uint>           shadowImpostorVisBuf      : register(XY_REG_U_MESH_CULL_UAV_SHADOW_IMPOSTOR_VIS);
RWByteAddressBuffer                shadowImpostorIndirectArgs : register(XY_REG_U_MESH_CULL_UAV_SHADOW_IMPOSTOR_INDIRECT_ARGS);

Texture2D<float4>                  hizTexture             : register(XY_REG_T_MESH_CULL_SRV_HI_Z);

bool DoesAABBIntersectFrustum(box3 bbox, frustum f);
uint SelectLOD(box3 bbox);
bool SelectImpostor(box3 bbox);
bool IsOccludedByHiZ(box3 bbox);
float FarthestHiZDepth(float a, float b);
float LoadHiZFarthestForRect(float2 minUV, float2 maxUV, float mipLevel);

[numthreads(64, 1, 1)]
void MeshCullRegion(uint3 dtid : SV_DispatchThreadID)
{
    uint idx = dtid.x;
    if (idx >= numRegions) return;
    mainRegionVisBuf[idx] = (uint)DoesAABBIntersectFrustum(regionData[idx].bbox, viewFrustum);
}

// Aggregation-key encoding: high bit flags an impostor-track lane so impostor
// slot N and trunk slot N never share a match-group.
static const uint kImpostorKeyBit = 0x80000000u;

[numthreads(256, 1, 1)]
void MeshCullMain(uint3 dtid : SV_DispatchThreadID)
{
    uint idx = dtid.x;
    bool inRange = (idx < totalCapacity);

    // Per-lane survivor classification. Predicate-only — no early return,
    // so every lane participates in subsequent wave intrinsics.
    bool alive      = false;
    bool isImpostor = false;
    uint slot       = 0;
    uint ai         = 0;

    if (inRange)
    {
        CullInstanceData inst = instanceData[idx];
        if (inst.active != 0u &&
            mainRegionVisBuf[inst.regionId] != 0u &&
            DoesAABBIntersectFrustum(inst.bbox, viewFrustum) &&
            !IsOccludedByHiZ(inst.bbox))
        {
            alive = true;
            ai    = inst.baseSlot / numLods;
            if (SelectImpostor(inst.bbox))
            {
                isImpostor = true;
                slot       = ai;  // impostor slot is per-asset
            }
            else
            {
                uint lod = SelectLOD(inst.bbox);
                slot     = inst.baseSlot + lod;
            }
        }
    }

    uint key = alive ? (isImpostor ? (kImpostorKeyBit | ai) : slot) : 0u;

    // Cycling ballot: each iteration aggregates one (wave, key) into a single
    // atomic. Inactive lanes start claimed and never enter the ballot.
    bool claimed    = !alive;
    uint myWriteIdx = 0;

    [allow_uav_condition]
    while (true)
    {
        uint4 anyBallot = WaveActiveBallot(!claimed);
        if ((anyBallot.x | anyBallot.y | anyBallot.z | anyBallot.w) == 0u) break;

        uint electedLane = WaveLeaderLaneFromBallot(anyBallot);
        uint electedKey  = WaveReadLaneAt(key, electedLane);

        uint4 matchMask = WaveActiveBallot(!claimed && key == electedKey);
        uint  total     = WaveMaskCountBits(matchMask);
        uint  myRank    = WaveMaskPrefixCountBits(matchMask);

        uint slotBase = 0;
        if (WaveIsFirstLane())
        {
            uint dummy;
            if (electedKey & kImpostorKeyBit)
            {
                uint ai_ = electedKey & ~kImpostorKeyBit;
                InterlockedAdd(impostorSlotCountBuf[ai_], total, slotBase);
                // DrawIndirectArguments instanceCount at offset 4 of 16-byte record
                impostorIndirectArgs.InterlockedAdd(ai_ * 16u + 4u, total, dummy);
            }
            else
            {
                InterlockedAdd(mainSlotCountBuf[electedKey], total, slotBase);
            }
        }
        slotBase = WaveReadLaneFirst(slotBase);

        bool matched = !claimed && (key == electedKey);
        if (matched)
        {
            myWriteIdx = slotBase + myRank;
            claimed    = true;

            // Tail lane of the match-group issues the dispatch-args InterlockedMax.
            // Original code did this for every surviving instance; InterlockedMax
            // is monotone, so only the highest writeIdx in this wave matters.
            if (!isImpostor && (myRank == total - 1u))
            {
                uint highest = (slotBase + total) * mainInvocsPerSlot[slot];
                UpdateDispatchMeshArgs(mainDispatchArgs, slot, highest);

                uint leafInvocs = mainLeafInvocsPerSlot[slot];
                if (leafInvocs > 0u)
                    UpdateDispatchMeshArgs(mainLeafDispatchArgs, slot, (slotBase + total) * leafInvocs);
            }
        }
    }

    if (alive)
    {
        if (isImpostor)
            impostorVisBuf[impostorSlotOffsets[ai] + myWriteIdx] = idx;
        else
            mainVisBuf[mainSlotOffsets[slot] + myWriteIdx] = idx;
    }
}

[numthreads(256, 1, 1)]
void MeshCullShadow(uint3 dtid : SV_DispatchThreadID)
{
    uint idx = dtid.x;
    bool inRange = (idx < totalCapacity);

    CullInstanceData inst = (CullInstanceData)0;
    bool active = false;
    if (inRange)
    {
        inst   = instanceData[idx];
        active = (inst.active != 0u);
    }

    // Light-space bbox + per-instance flags computed once for the cascade loop.
    float3 bboxMinLS  = float3( 1e30,  1e30,  1e30);
    float3 bboxMaxLS  = float3(-1e30, -1e30, -1e30);
    uint   ai         = 0;
    bool   isImpostor = false;

    if (active)
    {
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
        [unroll]
        for (int i = 0; i < 8; i++)
        {
            float3 ls = mul(float4(cornersWS[i], 1), worldToLight).xyz;
            bboxMinLS = min(bboxMinLS, ls);
            bboxMaxLS = max(bboxMaxLS, ls);
        }

        ai         = inst.baseSlot / numLods;
        isImpostor = (showShadowImpostors != 0u) && SelectImpostor(inst.bbox);
    }

    bool anyCascade = false;

    [unroll]
    for (uint c = 0; c < XYLEM_NUM_CASCADES; c++)
    {
        bool aliveForCascade = false;
        uint slot            = 0;

        if (active)
        {
            float3 cMin = shadowCasterMinLS[c].xyz;
            float3 cMax = shadowCasterMaxLS[c].xyz;
            bool casterValid = !any(cMin > cMax);
            bool overlap     = !(any(bboxMinLS > cMax) || any(bboxMaxLS < cMin));
            if (casterValid && overlap)
            {
                aliveForCascade = true;
                slot            = ai * XYLEM_NUM_CASCADES + c;
                anyCascade      = true;
            }
        }

        uint key = aliveForCascade ? (isImpostor ? (kImpostorKeyBit | slot) : slot) : 0u;

        bool claimed    = !aliveForCascade;
        uint myWriteIdx = 0;

        [allow_uav_condition]
        while (true)
        {
            uint4 anyBallot = WaveActiveBallot(!claimed);
            if ((anyBallot.x | anyBallot.y | anyBallot.z | anyBallot.w) == 0u) break;

            uint electedLane = WaveLeaderLaneFromBallot(anyBallot);
            uint electedKey  = WaveReadLaneAt(key, electedLane);

            uint4 matchMask = WaveActiveBallot(!claimed && key == electedKey);
            uint  total     = WaveMaskCountBits(matchMask);
            uint  myRank    = WaveMaskPrefixCountBits(matchMask);

            uint slotBase = 0;
            if (WaveIsFirstLane())
            {
                uint dummy;
                if (electedKey & kImpostorKeyBit)
                {
                    uint slot_ = electedKey & ~kImpostorKeyBit;
                    InterlockedAdd(shadowImpostorCountBuf[slot_], total, slotBase);
                    shadowImpostorIndirectArgs.InterlockedAdd(slot_ * 16u + 4u, total, dummy);
                }
                else
                {
                    InterlockedAdd(shadowSlotCountBuf[electedKey], total, slotBase);
                }
            }
            slotBase = WaveReadLaneFirst(slotBase);

            bool matched = !claimed && (key == electedKey);
            if (matched)
            {
                myWriteIdx = slotBase + myRank;
                claimed    = true;

                if (!isImpostor && (myRank == total - 1u))
                {
                    uint highest = (slotBase + total) * shadowInvocsPerSlot[slot];
                    UpdateDispatchMeshArgs(shadowDispatchArgs, slot, highest);

                    uint leafInvocs = shadowLeafInvocsPerSlot[slot];
                    if (leafInvocs > 0u)
                        UpdateDispatchMeshArgs(shadowLeafDispatchArgs, slot, (slotBase + total) * leafInvocs);
                }
            }
        }

        if (aliveForCascade)
        {
            if (isImpostor)
                shadowImpostorVisBuf[shadowImpostorSlotOffsets[slot] + myWriteIdx] = idx;
            else
                shadowVisBuf[shadowSlotOffsets[slot] + myWriteIdx] = idx;
        }
    }

    // shadowUniqueCounter: one atomic per wave instead of one per instance.
    uint uniqueCount = WaveActiveCountBits(anyCascade);
    if (WaveIsFirstLane() && uniqueCount > 0u)
    {
        uint dummy;
        shadowUniqueCounter.InterlockedAdd(0, uniqueCount, dummy);
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

bool SelectImpostor(box3 bbox)
{
    float3 nearest = clamp(cameraPos, bbox.min, bbox.max);
    float dist = distance(cameraPos, nearest);
    return dist >= lodDistances[numLods - 1].x;
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

    float hizDepth = LoadHiZFarthestForRect(minUV, maxUV, mipLevel);

#if XYLEM_USE_REVERSE_Z
    return (closestDepth < hizDepth);
#else
    return (closestDepth > hizDepth);
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
