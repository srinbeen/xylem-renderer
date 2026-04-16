#pragma pack_matrix(row_major)

cbuffer CB : register(b0)
{
    float4x4 viewProj;
    float4x4 lightViewProj;
    float3   sunLightDir;
    float    _pad0;
    float3x4 _pad1;
};

struct RootConstant { uint assetIndex; };
ConstantBuffer<RootConstant> rootConstant : register(b1);

// Layout matches C++ Render::InstanceBufferEntry (104 bytes, packed).
// See ComputeCullRenderPass.hlsl for the full explanation.
struct InstanceRenderData
{
    float4x4 model;     // offset 0,  size 64
    float3x3 normal;    // offset 64, size 36
    uint     treeId;    // offset 100, size 4
};

StructuredBuffer<uint>                  shadowVisBuf      : register(t0);
StructuredBuffer<InstanceRenderData>    instanceBuf       : register(t1);
StructuredBuffer<uint>                  shadowSlotOffsets : register(t2);

void tree_vs(
    in float3  i_pos       : POSITION,
    in uint    i_id        : SV_InstanceID,

    out float4 o_pos       : SV_Position
)
{
    uint ai = rootConstant.assetIndex;

    uint trueID    = shadowVisBuf[shadowSlotOffsets[ai] + i_id];
    float4x4 model = instanceBuf[trueID].model;
    o_pos = mul(mul(float4(i_pos, 1), model), lightViewProj);
}

void terrain_vs(
    in float3  i_pos    : POSITION,

    out float4 o_pos    : SV_Position
)
{
    o_pos = mul(float4(i_pos, 1), lightViewProj);
}
