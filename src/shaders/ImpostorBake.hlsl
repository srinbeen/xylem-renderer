#pragma pack_matrix(row_major)

#include "ShaderRegisterMap.hlsli"

cbuffer BakeCB : register(XY_REG_B_COMPUTE_IMPOSTORBAKE_CB_BAKE)
{
    float4x4    mvp;
    float4      _pad[12];
};

Texture2D    t_Diffuse   : register(XY_REG_T_COMPUTE_IMPOSTORBAKE_TEX_DIFFUSE);
Texture2D    t_NormalMap : register(XY_REG_T_COMPUTE_IMPOSTORBAKE_TEX_NORMAL_MAP);
SamplerState s_Sampler   : register(XY_REG_S_COMPUTE_IMPOSTORBAKE_SAMPLER_MAIN);

struct V2P
{
    float4 pos       : SV_Position;
    float3 normal    : NORMAL;
    float3 tangent   : TANGENT;
    float3 bitangent : BITANGENT;
    float2 uv        : UV;
};

void bake_vs(
    in float3 i_pos       : POSITION,
    in float3 i_normal    : NORMAL,
    in float3 i_tangent   : TANGENT,
    in float3 i_bitangent : BITANGENT,
    in float2 i_uv        : UV,

    out V2P o)
{
    o.pos       = mul(float4(i_pos, 1.0), mvp);
    o.normal    = normalize(i_normal);
    // Leaf vertices (uv.x < 0) pack the per-asset leaf color in TANGENT — pass through
    // unnormalized so the RGB survives interpolation. Matches the runtime tree shaders.
    if (i_uv.x < 0.0)
    {
        o.tangent   = i_tangent;
        o.bitangent = i_bitangent;
    }
    else
    {
        o.tangent   = normalize(i_tangent);
        o.bitangent = normalize(i_bitangent);
    }
    o.uv        = i_uv;
}

struct BakeOutput
{
    float4 albedoAlpha : SV_Target0;
    float4 normal      : SV_Target1;
};

BakeOutput bake_ps(in V2P i)
{
    BakeOutput o;

    // Leaf path: cross-billboards have no diffuse/normal texture — color is the
    // per-asset leaf color packed into TANGENT, and the surface normal is the
    // billboard's own face normal. Alpha = 1 (opaque), no clip.
    if (i.uv.x < 0.0)
    {
        o.albedoAlpha = float4(i.tangent, 1.0);
        float3 N = normalize(i.normal);
        o.normal = float4(saturate(N * 0.5 + 0.5), 1.0);
        return o;
    }

    // Trunk / branchlet path: tangent-space normal map sample.
    float3 T = normalize(i.tangent);
    float3 B = normalize(i.bitangent);
    float3 N = normalize(i.normal);
    float3x3 TBN = float3x3(T, B, N);

    float3 tangentNormal = normalize(t_NormalMap.Sample(s_Sampler, i.uv).rgb * 2.0 - 1.0);
    float3 objectNormal = normalize(mul(tangentNormal, TBN));

    float4 diffuse = t_Diffuse.Sample(s_Sampler, i.uv);
    clip(diffuse.a - 0.25);

    o.albedoAlpha = diffuse;
    o.normal = float4(saturate(objectNormal * 0.5 + 0.5), 1.0);
    return o;
}
