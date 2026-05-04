#include "../include/macros.h"
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

cbuffer LeafPush : register(b1)
{
    uint leafSlot;
    uint cascadeIdx;
};

StructuredBuffer<uint>               g_VisBuf      : register(t0);
StructuredBuffer<InstanceRenderData> g_Instances   : register(t1);
StructuredBuffer<uint>               g_SlotOffsets : register(t2);
StructuredBuffer<LeafInstanceData>   g_Leaves      : register(t3);
StructuredBuffer<LeafSlotData>       g_LeafSlots   : register(t4);
Texture2DArray                       t_ShadowMap   : register(t5);
SamplerComparisonState               s_ShadowSampler : register(s0);

struct V2P
{
    float4 pos      : SV_Position;
    float3 worldPos : WORLD_POS;
    float  viewZ    : VIEW_Z;
    float3 normal   : NORMAL;
    float3 color    : COLOR0;
};

InstanceRenderData LoadVisibleInstance(uint instanceId)
{
    uint trueId = g_VisBuf[g_SlotOffsets[leafSlot] + instanceId];
    return g_Instances[trueId];
}

void BuildWorldLeaf(uint vertexId, InstanceRenderData inst,
                    out float4 worldPos, out float3 worldNormal, out float3 color)
{
    LeafSlotData slot = g_LeafSlots[leafSlot];
    LeafInstanceData leaf = g_Leaves[slot.leafOffset + LeafIndexFromVertex(vertexId)];

    float3 localPos, localNormal;
    BuildLeafVertex(leaf, LeafVertexInLeaf(vertexId), localPos, localNormal);

    worldPos = mul(float4(localPos, 1.0), inst.model);
    worldNormal = normalize(mul(localNormal, inst.normal));
    color = leaf.color.rgb;
}

void leaf_vs(in uint vertexId : SV_VertexID,
             in uint instanceId : SV_InstanceID,
             out V2P o)
{
    InstanceRenderData inst = LoadVisibleInstance(instanceId);
    float4 worldPos;
    BuildWorldLeaf(vertexId, inst, worldPos, o.normal, o.color);

    o.pos = mul(worldPos, viewProj);
    o.worldPos = worldPos.xyz;
    o.viewZ = mul(worldPos, viewMatrix).z;
}

void leaf_depth_vs(in uint vertexId : SV_VertexID,
                   in uint instanceId : SV_InstanceID,
                   out float4 o_pos : SV_Position)
{
    InstanceRenderData inst = LoadVisibleInstance(instanceId);
    float4 worldPos;
    float3 worldNormal, color;
    BuildWorldLeaf(vertexId, inst, worldPos, worldNormal, color);
    o_pos = mul(worldPos, viewProj);
}

void leaf_shadow_vs(in uint vertexId : SV_VertexID,
                    in uint instanceId : SV_InstanceID,
                    out float4 o_pos : SV_Position)
{
    uint shadowSlot = leafSlot * XYLEM_NUM_CASCADES + cascadeIdx;
    uint trueId = g_VisBuf[g_SlotOffsets[shadowSlot] + instanceId];
    InstanceRenderData inst = g_Instances[trueId];
    LeafSlotData slot = g_LeafSlots[shadowSlot];
    LeafInstanceData leaf = g_Leaves[slot.leafOffset + LeafIndexFromVertex(vertexId)];

    float3 localPos, localNormal;
    BuildLeafVertex(leaf, LeafVertexInLeaf(vertexId), localPos, localNormal);
    float4 worldPos = mul(float4(localPos, 1.0), inst.model);
    o_pos = mul(worldPos, lightViewProj[cascadeIdx]);
}

float SampleShadowCascade(float3 worldPos, uint selectedCascade)
{
    float4 posLS = mul(float4(worldPos, 1), lightViewProj[selectedCascade]);
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

void leaf_ps(in V2P i, out float4 o_color : SV_Target0)
{
    uint selectedCascade = XYLEM_NUM_CASCADES - 1;
    if      (i.viewZ < cascadeSplits.x) selectedCascade = 0;
    else if (i.viewZ < cascadeSplits.y) selectedCascade = 1;
    else if (i.viewZ < cascadeSplits.z) selectedCascade = 2;

    float3 lightDir = -normalize(sunLightDir);
    float notInShadow = SampleShadowCascade(i.worldPos, selectedCascade);
    // Single-sided diffuse to match the trunk/impostor shading model.
    float diffuse = max(dot(normalize(i.normal), lightDir), 0.0);
    float ambient = 0.20;
    float lighting = ambient + (1.0 - ambient) * diffuse * notInShadow;
    o_color = float4(lighting * i.color, 1.0);
}
