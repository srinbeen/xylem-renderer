#pragma pack_matrix(row_major)

#include "types.hlsli"

cbuffer CB : register(b0)
{
    // P0 fields (also read by VS/PS)
    float4x4 view;
    float4x4 projection;
    float4x4 lightViewProj;
    float3   sunLightDir;
    float    _pad0;
    float3x4 _pad1;

    // did not specify the cull fields
};

struct RootConstant { uint slot; };
ConstantBuffer<RootConstant> rc : register(b1);

struct InstanceRenderData
{
    float4x4 model;
    float3x3 normal;
    uint     treeId;
};

StructuredBuffer<uint>                  visBuf       : register(t0);
StructuredBuffer<InstanceRenderData>    instBuf      : register(t1);
StructuredBuffer<uint>                  slotOffsets  : register(t2);


void main_vs(
    in float3  i_pos       : POSITION,
    in float3  i_normal    : NORMAL,
    in float3  i_tangent   : TANGENT,
    in float3  i_bitangent : BITANGENT,
    in float2  i_uv        : UV,
    in uint    i_id        : SV_InstanceID,

    out float4 o_pos       : SV_Position,
    out float4 o_pos_LS    : POSITION_LS,
    out float3 o_normal    : NORMAL,
    out float3 o_tangent   : TANGENT,
    out float3 o_bitangent : BITANGENT,
    out float2 o_uv        : UV
)
{
    uint trueID    = visBuf[slotOffsets[rc.slot] + i_id];
    float4x4 model = instBuf[trueID].model;
    float3x3 normalMat = instBuf[trueID].normal;

    float4x4 viewProj = mul(view, projection);
    float4 worldPos = mul(float4(i_pos, 1), model);

    o_pos       = mul(worldPos, viewProj);
    o_pos_LS    = mul(worldPos, lightViewProj);
    o_normal    = normalize(mul(i_normal,    normalMat));
    o_tangent   = normalize(mul(i_tangent,   normalMat));
    o_bitangent = normalize(mul(i_bitangent, normalMat));
    o_uv        = i_uv;
}


Texture2D              t_Diffuse        : register(t3);
Texture2D              t_NormalMap      : register(t4);
Texture2D              t_ShadowMap      : register(t5);

SamplerState           s_Sampler        : register(s0);
SamplerComparisonState s_ShadowSampler  : register(s1);


void main_ps(
    in float4  i_pos       : SV_Position,
    in float4  i_pos_LS    : POSITION_LS,
    in float3  i_normal    : NORMAL,
    in float3  i_tangent   : TANGENT,
    in float3  i_bitangent : BITANGENT,
    in float2  i_uv        : UV,

    out float4 o_color     : SV_Target0
)
{
    float3 T = normalize(i_tangent);
    float3 B = normalize(i_bitangent);
    float3 N = normalize(i_normal);
    float3x3 TBN = float3x3(T, B, N);

    float3 tangentNormal = normalize(t_NormalMap.Sample(s_Sampler, i_uv).rgb * 2.0 - 1.0);
    float3 worldNormal = normalize(mul(tangentNormal, TBN));

    float3 lightDir = -normalize(sunLightDir);
    float diffuse = max(dot(worldNormal, lightDir), 0);

    float2 shadowUV    = i_pos_LS.xy * float2(0.5, -0.5) + 0.5;
    float  notInShadow = t_ShadowMap.SampleCmpLevelZero(s_ShadowSampler, shadowUV, i_pos_LS.z);

    float ambient  = 0.15;
    float lighting = ambient + (1.0 - ambient) * diffuse * notInShadow;
    o_color = float4(lighting * t_Diffuse.Sample(s_Sampler, i_uv).rgb, 1);
}
