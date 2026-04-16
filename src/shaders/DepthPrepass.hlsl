#pragma pack_matrix(row_major)

cbuffer CB : register(b0)
{
    float4x4 viewProj;
    float4x4 lightViewProj;
    float3   sunLightDir;
    float    _pad0;
    float3x4 _pad1;
};

struct RootConstant { uint slot; };
ConstantBuffer<RootConstant> rc : register(b1);

// Layout matches C++ Render::InstanceBufferEntry (104 bytes, packed).
// See ComputeCullRenderPass.hlsl for the full explanation.
struct InstanceRenderData
{
    float4x4 model;     // offset 0,  size 64
    float3x3 normal;    // offset 64, size 36
    uint     treeId;    // offset 100, size 4
};

StructuredBuffer<uint>                  visBuf      : register(t0);
StructuredBuffer<InstanceRenderData>    instBuf     : register(t1);
StructuredBuffer<uint>                  slotOffsets : register(t2);

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
    in float3  i_pos    : POSITION,
    in float3  i_normal : NORMAL,
    in float2  i_uv     : UV,
    out float4 o_pos    : SV_Position
)
{
    o_pos = mul(float4(i_pos, 1), viewProj);
}
