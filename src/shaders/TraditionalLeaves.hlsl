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
    float4   _pad1[6];
};

cbuffer LeafPush : register(b1)
{
    uint leafSlot;
    uint cascadeIdx;
};

StructuredBuffer<LeafInstanceData> g_Leaves : register(t0);
StructuredBuffer<LeafSlotData>     g_LeafSlots : register(t1);
Texture2DArray                     t_ShadowMap : register(t2);
SamplerComparisonState             s_ShadowSampler : register(s0);

struct V2P
{
    float4 pos      : SV_Position;
    float3 worldPos : WORLD_POS;
    float  viewZ    : VIEW_Z;
    float3 normal   : NORMAL;
    float3 color    : COLOR0;
};

void leaf_vs(
    in float4 im_model[4] : MODEL_MATRIX,
    in float3 im_norm[3]  : NORMAL_MATRIX,
    in uint vertexId      : SV_VertexID,
    out V2P o)
{
    float4x4 model = float4x4(im_model[0], im_model[1], im_model[2], im_model[3]);
    float3x3 normalMat = float3x3(im_norm[0], im_norm[1], im_norm[2]);

    LeafSlotData slot = g_LeafSlots[leafSlot];
    uint leafIdx = slot.leafOffset + LeafIndexFromVertex(vertexId);
    LeafInstanceData leaf = g_Leaves[leafIdx];

    float3 localPos, localNormal;
    BuildLeafVertex(leaf, LeafVertexInLeaf(vertexId), localPos, localNormal);

    float4 worldPos = mul(float4(localPos, 1.0), model);
    o.pos = mul(worldPos, viewProj);
    o.worldPos = worldPos.xyz;
    o.viewZ = mul(worldPos, viewMatrix).z;
    o.normal = normalize(mul(localNormal, normalMat));
    o.color = leaf.color.rgb;
}

void leaf_shadow_vs(
    in float4 im_model[4] : MODEL_MATRIX,
    in uint vertexId      : SV_VertexID,
    out float4 o_pos      : SV_Position)
{
    float4x4 model = float4x4(im_model[0], im_model[1], im_model[2], im_model[3]);

    LeafSlotData slot = g_LeafSlots[leafSlot];
    uint leafIdx = slot.leafOffset + LeafIndexFromVertex(vertexId);
    LeafInstanceData leaf = g_Leaves[leafIdx];

    float3 localPos, localNormal;
    BuildLeafVertex(leaf, LeafVertexInLeaf(vertexId), localPos, localNormal);

    float4 worldPos = mul(float4(localPos, 1.0), model);
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
