#include "../include/macros.h"

#pragma pack_matrix(row_major)

static const uint NUM_CASCADES = 4;

cbuffer CB : register(b0)
{
    float4x4 viewProj;
    float4x4 viewMatrix;
    float4x4 lightViewProj[NUM_CASCADES];
    float3   sunLightDir;
    float    _pad0;
    float4   cascadeSplits;
    float    _pad1[24];
};

void terrain_vs(
    in float3  i_pos    : POSITION,
    in float3  i_normal : NORMAL,
    in float2  i_uv     : UV,

    out float4 o_pos     : SV_Position,
    out float3 o_worldPos : WORLD_POS,
    out float  o_viewZ   : VIEW_Z,
    out float3 o_normal  : NORMAL,
    out float2 o_uv      : UV,
    out float  o_height  : HEIGHT
)
{
    o_pos      = mul(float4(i_pos, 1), viewProj);
    o_worldPos = i_pos;
    o_viewZ    = mul(float4(i_pos, 1), viewMatrix).z;
    o_normal   = i_normal;
    o_uv       = i_uv;
    o_height   = i_pos.y;
}

Texture2DArray t_ShadowMap : register(t0);
SamplerComparisonState s_ShadowSampler : register(s0);

float SampleShadowCascade(float3 worldPos, uint cascadeIdx)
{
    float4 posLS = mul(float4(worldPos, 1), lightViewProj[cascadeIdx]);
    float2 shadowUV = posLS.xy * float2(0.5, -0.5) + 0.5;
    return t_ShadowMap.SampleCmpLevelZero(s_ShadowSampler,
        float3(shadowUV, float(cascadeIdx)), posLS.z);
}

void terrain_ps(
    in float4  i_pos     : SV_Position,
    in float3  i_worldPos : WORLD_POS,
    in float   i_viewZ   : VIEW_Z,
    in float3  i_normal  : NORMAL,
    in float2  i_uv      : UV,
    in float   i_height  : HEIGHT,

    out float4 o_color   : SV_Target0
)
{
    float3 N = normalize(i_normal);

    float3 lightDir = -normalize(sunLightDir);
    float diffuse   = max(dot(N, lightDir), 0);

    // Cascade selection by view-space depth
    uint cascadeIdx = 3;
    if      (i_viewZ < cascadeSplits.x) cascadeIdx = 0;
    else if (i_viewZ < cascadeSplits.y) cascadeIdx = 1;
    else if (i_viewZ < cascadeSplits.z) cascadeIdx = 2;

    float notInShadow = SampleShadowCascade(i_worldPos, cascadeIdx);

    float ambient  = 0.15;
    float lighting = ambient + (1.0 - ambient) * diffuse * notInShadow;

    // Procedural terrain coloring based on height and slope
    float slope = 1.0 - N.y; // 0 = flat, 1 = vertical

    float3 grassColor = float3(0.15, 0.35, 0.08);
    float3 dirtColor  = float3(0.45, 0.35, 0.2);
    float3 rockColor  = float3(0.5, 0.48, 0.45);

    // Height blend: low = grass, high = dirt
    float heightFactor = saturate(i_height * 0.1 + 0.3);
    float3 baseColor = lerp(grassColor, dirtColor, heightFactor);

    // Slope blend: steep = rock
    float slopeFactor = saturate(slope * 4.0);
    float3 color = lerp(baseColor, rockColor, slopeFactor);

    o_color = float4(lighting * color, 1);
}
