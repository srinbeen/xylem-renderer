#pragma pack_matrix(row_major)

// ---------------------------------------------------------------------------
// Constant buffer — same as ComputeCullRenderPass (P0 fields only)
// ---------------------------------------------------------------------------
cbuffer CB : register(b0)
{
    float4x4 view;
    float4x4 projection;
    float4x4 lightViewProj;
    float3   sunLightDir;
    float    _pad0;
    float3x4 _pad1;
};

// ---------------------------------------------------------------------------
// Structured buffers for shadow visibility indirection
// ---------------------------------------------------------------------------
struct Instance
{
    float4x4 model;
    float3x3 normal;
    uint     treeId;
};

StructuredBuffer<uint>     shadowVisBuf      : register(t0);
StructuredBuffer<Instance> instanceBuffer    : register(t1);
StructuredBuffer<uint>     shadowCount       : register(t2);  // [numAssets]
StructuredBuffer<uint>     shadowSlotOffsets : register(t3);  // [numAssets]

struct RootConstant { uint assetIndex; };
ConstantBuffer<RootConstant> rootConstant : register(b1);

// ---------------------------------------------------------------------------
// Shadow tree VS — per-asset slot indirection
// ---------------------------------------------------------------------------
void tree_vs(
    in float3  i_pos       : POSITION,
    in uint    i_id        : SV_InstanceID,

    out float4 o_pos       : SV_Position
)
{
    uint ai = rootConstant.assetIndex;

    if (i_id >= shadowCount[ai])
    {
        o_pos = asfloat(0x7fc00000);
        return;
    }

    uint trueID    = shadowVisBuf[shadowSlotOffsets[ai] + i_id];
    float4x4 model = instanceBuffer[trueID].model;
    o_pos = mul(mul(float4(i_pos, 1), model), lightViewProj);
}

// ---------------------------------------------------------------------------
// Shadow terrain VS — unchanged (no instancing)
// ---------------------------------------------------------------------------
void terrain_vs(
    in float3  i_pos    : POSITION,

    out float4 o_pos    : SV_Position
)
{
    o_pos = mul(float4(i_pos, 1), lightViewProj);
}
