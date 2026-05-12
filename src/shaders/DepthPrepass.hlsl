#pragma pack_matrix(row_major)
#include "ShaderRegisterMap.hlsli"

cbuffer CB : register(XY_REG_B_COMPUTE_DEPTHPREPASS_CB_FRAME)
{
    float4x4 viewProj;
};

struct RootConstant { uint slot; };
ConstantBuffer<RootConstant> rc : register(XY_REG_B_COMPUTE_DEPTHPREPASS_PUSH_C_SLOT);

// Layout matches C++ Render::InstanceBufferEntry (104 bytes, packed).
struct InstanceRenderData
{
    float4x4 model;     // offset 0,  size 64
    float3x3 normal;    // offset 64, size 36
    uint     treeId;    // offset 100, size 4
};

StructuredBuffer<uint>                  visBuf      : register(XY_REG_T_COMPUTE_DEPTHPREPASS_SRV_VIS);
StructuredBuffer<InstanceRenderData>    instBuf     : register(XY_REG_T_COMPUTE_DEPTHPREPASS_SRV_INSTANCES);
StructuredBuffer<uint>                  slotOffsets : register(XY_REG_T_COMPUTE_DEPTHPREPASS_SRV_SLOT_OFFSETS);

void tree_vs(
    in float3  i_pos : POSITION,
    in uint    i_id  : SV_InstanceID,
    out float4 o_pos : SV_Position
)
{
    uint trueID    = visBuf[slotOffsets[rc.slot] + i_id];
    float4x4 model = instBuf[trueID].model;
    o_pos = mul(mul(float4(i_pos, 1), model), viewProj);
}

void terrain_vs(
    in float3  i_pos : POSITION,
    out float4 o_pos : SV_Position
)
{
    o_pos = mul(float4(i_pos, 1), viewProj);
}
