#pragma pack_matrix(row_major)
#include "ShaderRegisterMap.hlsli"
#include "../include/macros.h"

cbuffer CB : register(XY_REG_B_COMPUTE_SHADOW_CB_FRAME)
{
    float4x4 viewProj;
    float4x4 viewMatrix;
    float4x4 lightViewProj[XYLEM_NUM_CASCADES];
};

struct RootConstant
{
    uint assetIndex;
    uint cascadeIdx;
};
ConstantBuffer<RootConstant> rc : register(XY_REG_B_COMPUTE_SHADOW_PUSH_C_ASSET_CASCADE);

// Layout matches C++ Render::InstanceBufferEntry (104 bytes, packed).
struct InstanceRenderData
{
    float4x4 model;     // offset 0,  size 64
    float3x3 normal;    // offset 64, size 36
    uint     treeId;    // offset 100, size 4
};

StructuredBuffer<uint>                  shadowVisBuf      : register(XY_REG_T_COMPUTE_SHADOW_SRV_VIS);
StructuredBuffer<InstanceRenderData>    instanceBuf       : register(XY_REG_T_COMPUTE_SHADOW_SRV_INSTANCES);
StructuredBuffer<uint>                  shadowSlotOffsets : register(XY_REG_T_COMPUTE_SHADOW_SRV_SLOT_OFFSETS);

void tree_vs(
    in float3  i_pos       : POSITION,
    in uint    i_id        : SV_InstanceID,

    out float4 o_pos       : SV_Position
)
{
    uint slot = rc.assetIndex * XYLEM_NUM_CASCADES + rc.cascadeIdx;

    uint trueID    = shadowVisBuf[shadowSlotOffsets[slot] + i_id];
    float4x4 model = instanceBuf[trueID].model;
    o_pos = mul(mul(float4(i_pos, 1), model), lightViewProj[rc.cascadeIdx]);
}

void terrain_vs(
    in float3  i_pos    : POSITION,

    out float4 o_pos    : SV_Position
)
{
    o_pos = mul(float4(i_pos, 1), lightViewProj[rc.cascadeIdx]);
}
