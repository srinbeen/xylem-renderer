#include "../include/macros.h"
#include "meshlet_types.hlsli"
#include "LeafCommon.hlsli"

#pragma pack_matrix(row_major)

cbuffer CB : register(b0)
{
    float4x4 viewProj;
    float4x4 viewMatrix;
    float4x4 lightViewProj[XYLEM_NUM_CASCADES];
    float3   sunLightDir;
    float    _pad0;
    float4   cascadeSplits;
};

cbuffer PushC : register(b1)
{
    uint g_SlotIdx;
};

StructuredBuffer<uint>               g_VisBuf          : register(t0);
StructuredBuffer<uint>               g_SlotOffsets     : register(t1);
StructuredBuffer<uint>               g_SlotCounts      : register(t2);
StructuredBuffer<InstanceRenderData> g_Instances       : register(t3);
StructuredBuffer<uint>               g_ASInvocsPerSlot : register(t4);
StructuredBuffer<LeafInstanceData>   g_Leaves          : register(t5);
StructuredBuffer<LeafSlotData>       g_LeafSlots       : register(t6);
StructuredBuffer<LeafMeshletData>    g_LeafMeshlets    : register(t7);
Texture2DArray                       t_ShadowMap       : register(t8);
SamplerComparisonState               s_ShadowSampler   : register(s0);

struct LeafPayload
{
    uint instanceIdx;
    uint slotIdx;
    uint meshletIndices[XYLEM_AS_GROUP_SIZE];
};

groupshared LeafPayload s_payload;
groupshared uint s_survivors;

[numthreads(XYLEM_AS_GROUP_SIZE, 1, 1)]
void leaf_as(uint3 gid : SV_GroupID, uint gtid : SV_GroupThreadID)
{
    if (gtid == 0) s_survivors = 0;
    GroupMemoryBarrierWithGroupSync();

    uint slotIdx = g_SlotIdx;
    uint invocsPerInst = max(1u, g_ASInvocsPerSlot[slotIdx]);
    uint visibleCount = g_SlotCounts[slotIdx];
    LeafSlotData leafSlot = g_LeafSlots[slotIdx];

    uint flatGroup = gid.x + gid.y * XYLEM_DISPATCH_X;
    uint instanceInSlot = flatGroup / invocsPerInst;
    uint invocIdx = flatGroup % invocsPerInst;

    if (instanceInSlot < visibleCount)
    {
        uint persistentInstIdx = g_VisBuf[g_SlotOffsets[slotIdx] + instanceInSlot];
        if (gtid == 0)
        {
            s_payload.instanceIdx = persistentInstIdx;
            s_payload.slotIdx = slotIdx;
        }

        uint meshletLocalIdx = invocIdx * XYLEM_AS_GROUP_SIZE + gtid;
        if (meshletLocalIdx < leafSlot.meshletCount)
        {
            uint survivor;
            InterlockedAdd(s_survivors, 1, survivor);
            s_payload.meshletIndices[survivor] = meshletLocalIdx;
        }
    }

    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(s_survivors, 1, 1, s_payload);
}

struct LeafV2P
{
    float4 pos      : SV_Position;
    float3 worldPos : WORLD_POS;
    float  viewZ    : VIEW_Z;
    float3 normal   : NORMAL;
    float3 color    : COLOR0;
};

LeafV2P BuildMeshLeafVertex(uint gtid, uint survivorIdx, LeafPayload payload)
{
    LeafSlotData slot = g_LeafSlots[payload.slotIdx];
    LeafMeshletData meshlet = g_LeafMeshlets[slot.meshletOffset + payload.meshletIndices[survivorIdx]];

    uint leafInMeshlet = gtid / 8u;
    uint corner = gtid % 8u;
    LeafInstanceData leaf = g_Leaves[meshlet.meta.x + leafInMeshlet];
    InstanceRenderData inst = g_Instances[payload.instanceIdx];

    float3 localPos, localNormal;
    BuildLeafCorner(leaf, corner, localPos, localNormal);

    float4 worldPos = mul(float4(localPos, 1.0), inst.model);

    LeafV2P v;
    v.pos = mul(worldPos, viewProj);
    v.worldPos = worldPos.xyz;
    v.viewZ = mul(worldPos, viewMatrix).z;
    v.normal = normalize(mul(localNormal, inst.normal));
    v.color = leaf.color.rgb;
    return v;
}

uint3 LeafTri(uint tri)
{
    uint leaf = tri / 4u;
    uint triInLeaf = tri % 4u;
    uint base = leaf * 8u;

    if (triInLeaf == 0) return uint3(base + 0, base + 1, base + 2);
    if (triInLeaf == 1) return uint3(base + 0, base + 2, base + 3);
    if (triInLeaf == 2) return uint3(base + 4, base + 5, base + 6);
    return uint3(base + 4, base + 6, base + 7);
}

[numthreads(XYLEM_MS_GROUP_SIZE, 1, 1)]
[outputtopology("triangle")]
void leaf_ms(
    uint gtid : SV_GroupThreadID,
    uint3 gid : SV_GroupID,
    in payload LeafPayload payload,
    out indices uint3 o_tris[XYLEM_MAX_MESHLET_PRIMS],
    out vertices LeafV2P o_verts[XYLEM_MAX_MESHLET_VERTS])
{
    LeafSlotData slot = g_LeafSlots[payload.slotIdx];
    LeafMeshletData meshlet = g_LeafMeshlets[slot.meshletOffset + payload.meshletIndices[gid.x]];
    uint leafCount = meshlet.meta.y;
    SetMeshOutputCounts(leafCount * 8u, leafCount * 4u);

    if (gtid < leafCount * 8u)
        o_verts[gtid] = BuildMeshLeafVertex(gtid, gid.x, payload);
    if (gtid < leafCount * 4u)
        o_tris[gtid] = LeafTri(gtid);
}

struct DepthV2P
{
    float4 pos : SV_Position;
};

[numthreads(XYLEM_MS_GROUP_SIZE, 1, 1)]
[outputtopology("triangle")]
void leaf_depth_ms(
    uint gtid : SV_GroupThreadID,
    uint3 gid : SV_GroupID,
    in payload LeafPayload payload,
    out indices uint3 o_tris[XYLEM_MAX_MESHLET_PRIMS],
    out vertices DepthV2P o_verts[XYLEM_MAX_MESHLET_VERTS])
{
    LeafSlotData slot = g_LeafSlots[payload.slotIdx];
    LeafMeshletData meshlet = g_LeafMeshlets[slot.meshletOffset + payload.meshletIndices[gid.x]];
    uint leafCount = meshlet.meta.y;
    SetMeshOutputCounts(leafCount * 8u, leafCount * 4u);

    if (gtid < leafCount * 8u)
    {
        LeafV2P v = BuildMeshLeafVertex(gtid, gid.x, payload);
        o_verts[gtid].pos = v.pos;
    }
    if (gtid < leafCount * 4u)
        o_tris[gtid] = LeafTri(gtid);
}

[numthreads(XYLEM_MS_GROUP_SIZE, 1, 1)]
[outputtopology("triangle")]
void leaf_shadow_ms(
    uint gtid : SV_GroupThreadID,
    uint3 gid : SV_GroupID,
    in payload LeafPayload payload,
    out indices uint3 o_tris[XYLEM_MAX_MESHLET_PRIMS],
    out vertices DepthV2P o_verts[XYLEM_MAX_MESHLET_VERTS])
{
    LeafSlotData slot = g_LeafSlots[payload.slotIdx];
    LeafMeshletData meshlet = g_LeafMeshlets[slot.meshletOffset + payload.meshletIndices[gid.x]];
    uint leafCount = meshlet.meta.y;
    uint cascade = payload.slotIdx % XYLEM_NUM_CASCADES;
    SetMeshOutputCounts(leafCount * 8u, leafCount * 4u);

    if (gtid < leafCount * 8u)
    {
        uint leafInMeshlet = gtid / 8u;
        uint corner = gtid % 8u;
        LeafInstanceData leaf = g_Leaves[meshlet.meta.x + leafInMeshlet];
        InstanceRenderData inst = g_Instances[payload.instanceIdx];

        float3 localPos, localNormal;
        BuildLeafCorner(leaf, corner, localPos, localNormal);
        float4 worldPos = mul(float4(localPos, 1.0), inst.model);
        o_verts[gtid].pos = mul(worldPos, lightViewProj[cascade]);
    }
    if (gtid < leafCount * 4u)
        o_tris[gtid] = LeafTri(gtid);
}

float SampleShadowCascade(float3 worldPos, uint selectedCascade)
{
    float4 posLS = mul(float4(worldPos, 1.0), lightViewProj[selectedCascade]);
    float2 shadowUV = posLS.xy * float2(0.5, -0.5) + 0.5;

    uint width, height, elements;
    t_ShadowMap.GetDimensions(width, height, elements);
    float2 texelSize = 1.0 / float2(width, height);

    float shadow = 0.0;
    [unroll]
    for (int x = -1; x <= 1; ++x) {
        [unroll]
        for (int y = -1; y <= 1; ++y) {
            shadow += t_ShadowMap.SampleCmpLevelZero(
                s_ShadowSampler,
                float3(shadowUV + float2(x, y) * texelSize, float(selectedCascade)),
                posLS.z);
        }
    }
    return shadow / 9.0;
}

void leaf_ps(in LeafV2P i, out float4 o_color : SV_Target0)
{
    uint selectedCascade = XYLEM_NUM_CASCADES - 1;
    if      (i.viewZ < cascadeSplits.x) selectedCascade = 0;
    else if (i.viewZ < cascadeSplits.y) selectedCascade = 1;
    else if (i.viewZ < cascadeSplits.z) selectedCascade = 2;

    float3 lightDir = -normalize(sunLightDir);
    float notInShadow = SampleShadowCascade(i.worldPos, selectedCascade);
    // Wrap diffuse: must match ImpostorRenderPass.hlsl so leaves and impostors
    // shade identically across the LOD swap.
    float diffuse = saturate(dot(normalize(i.normal), lightDir) * 0.5 + 0.5);
    float ambient = 0.20;
    float lighting = ambient + (1.0 - ambient) * diffuse * notInShadow;
    o_color = float4(lighting * i.color, 1.0);
}
