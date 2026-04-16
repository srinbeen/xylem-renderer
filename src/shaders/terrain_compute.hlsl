#include "../include/macros.h"

#pragma pack_matrix(row_major)

cbuffer CB : register(b0)
{
    float4x4 viewProj;
    float4x4 lightViewProj;
    float3   sunLightDir;
    float    _pad0;
    float3x4 _pad1;
};

void terrain_vs(
    in float3  i_pos    : POSITION,
    in float3  i_normal : NORMAL,
    in float2  i_uv     : UV,

    out float4 o_pos      : SV_Position,
    out float4 o_pos_LS   : POSITION_LS,
    out float3 o_normal   : NORMAL,
    out float2 o_uv       : UV,
    out float  o_height   : HEIGHT
)
{
    o_pos      = mul(float4(i_pos, 1), viewProj);
    o_pos_LS   = mul(float4(i_pos, 1), lightViewProj);
    o_normal   = i_normal;
    o_uv       = i_uv;
    o_height   = i_pos.y;
}

Texture2D              t_ShadowMap     : register(t0);
SamplerComparisonState s_ShadowSampler : register(s0);

void terrain_ps(
    in float4 i_pos     : SV_Position,
    in float4 i_pos_LS  : POSITION_LS,
    in float3 i_normal  : NORMAL,
    in float2 i_uv      : UV,
    in float  i_height  : HEIGHT,

    out float4 o_color : SV_Target0
)
{
    float3 N = normalize(i_normal);

    float3 lightDir = -normalize(sunLightDir);
    float  diffuse  = max(dot(N, lightDir), 0);

    float2 shadowUV    = i_pos_LS.xy * float2(0.5, -0.5) + 0.5;
    float  notInShadow = t_ShadowMap.SampleCmpLevelZero(s_ShadowSampler, shadowUV, i_pos_LS.z);

    float ambient  = 0.15;
    float lighting = ambient + (1.0 - ambient) * diffuse * notInShadow;

    float slope = 1.0 - N.y;

    float3 grassColor = float3(0.15, 0.35, 0.08);
    float3 dirtColor  = float3(0.45, 0.35, 0.2);
    float3 rockColor  = float3(0.5, 0.48, 0.45);

    float heightFactor = saturate(i_height * 0.1 + 0.3);
    float3 baseColor   = lerp(grassColor, dirtColor, heightFactor);

    float slopeFactor = saturate(slope * 4.0);
    float3 color      = lerp(baseColor, rockColor, slopeFactor);

    o_color = float4(lighting * color, 1);
}
