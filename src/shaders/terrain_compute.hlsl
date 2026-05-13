#include "../include/macros.h"
#include "ShaderRegisterMap.hlsli"
#include "ShadowCascadeCommon.hlsli"
#include "terrain_shading.hlsli"

#pragma pack_matrix(row_major)

cbuffer FrameCB : register(XY_REG_B_COMPUTE_TERRAIN_CB_FRAME)
{
    float4x4 viewProj;
    float4x4 viewMatrix;
    float4x4 lightViewProj[XYLEM_NUM_CASCADES];
    float3   sunLightDir;
    float    _pad0;
    float3   sunColor;
    float    _pad0b;
    float4   cascadeSplits;
};

cbuffer ShadingCB : register(XY_REG_B_COMPUTE_TERRAIN_CB_SHADING)
{
    float g_TileSize;
    float g_ForestToDirtY;
    float g_DirtToSnowY;
    float g_BandWidth;
    float g_SlopeLo;
    float g_SlopeHi;
    float g_MacroNoiseAmp;
    float g_Pad0;
};

void terrain_vs(
    in float3  i_pos    : POSITION,
    in float3  i_normal : NORMAL,
    in float2  i_uv     : UV,

    out float4 o_pos      : SV_Position,
    out float3 o_worldPos : WORLD_POS,
    out float  o_viewZ    : VIEW_Z,
    out float3 o_normal   : NORMAL,
    out float2 o_uv       : UV,
    out float  o_height   : HEIGHT
)
{
    o_pos      = mul(float4(i_pos, 1), viewProj);
    o_worldPos = i_pos;
    o_viewZ    = mul(float4(i_pos, 1), viewMatrix).z;
    o_normal   = i_normal;
    o_uv       = i_uv;
    o_height   = i_pos.y;
}

Texture2DArray         t_ShadowMap     : register(XY_REG_T_COMPUTE_TERRAIN_TEX_SHADOW_MAP);
SamplerComparisonState s_ShadowSampler : register(XY_REG_S_COMPUTE_TERRAIN_SAMPLER_SHADOW);

Texture2D    t_ForestDiff : register(XY_REG_T_COMPUTE_TERRAIN_TEX_FOREST_DIFF);
Texture2D    t_ForestNor  : register(XY_REG_T_COMPUTE_TERRAIN_TEX_FOREST_NOR);
Texture2D    t_DirtDiff   : register(XY_REG_T_COMPUTE_TERRAIN_TEX_DIRT_DIFF);
Texture2D    t_DirtNor    : register(XY_REG_T_COMPUTE_TERRAIN_TEX_DIRT_NOR);
Texture2D    t_RockDiff   : register(XY_REG_T_COMPUTE_TERRAIN_TEX_ROCK_DIFF);
Texture2D    t_RockNor    : register(XY_REG_T_COMPUTE_TERRAIN_TEX_ROCK_NOR);
Texture2D    t_SnowDiff   : register(XY_REG_T_COMPUTE_TERRAIN_TEX_SNOW_DIFF);
Texture2D    t_SnowNor    : register(XY_REG_T_COMPUTE_TERRAIN_TEX_SNOW_NOR);
SamplerState s_Aniso      : register(XY_REG_S_COMPUTE_TERRAIN_SAMPLER_ANISO);

float SampleShadowCascade(float3 worldPos, uint cascadeIdx)
{
    float4 posLS    = mul(float4(worldPos, 1), lightViewProj[cascadeIdx]);
    float2 shadowUV = posLS.xy * float2(0.5, -0.5) + 0.5;
    return t_ShadowMap.SampleCmpLevelZero(s_ShadowSampler,
        float3(shadowUV, float(cascadeIdx)), posLS.z);
}

void terrain_ps(
    in float4  i_pos      : SV_Position,
    in float3  i_worldPos : WORLD_POS,
    in float   i_viewZ    : VIEW_Z,
    in float3  i_normal   : NORMAL,
    in float2  i_uv       : UV,
    in float   i_height   : HEIGHT,

    out float4 o_color    : SV_Target0
)
{
    float3 vertN = normalize(i_normal);

    TerrainShadingParams p;
    p.tileSize       = g_TileSize;
    p.forestToDirtY  = g_ForestToDirtY;
    p.dirtToSnowY    = g_DirtToSnowY;
    p.bandWidth      = g_BandWidth;
    p.slopeLo        = g_SlopeLo;
    p.slopeHi        = g_SlopeHi;
    p.macroNoiseAmp  = g_MacroNoiseAmp;

    float4 weights = ComputeLayerWeights(i_worldPos, vertN, i_height, p);
    float3 triW    = ComputeTriplanarWeights(vertN);
    float3 worldPosScaled = i_worldPos / max(g_TileSize, 1e-3);

    float3 albedoSum = 0.0;
    float3 normalSum = 0.0;
    AccumulateLayer(t_ForestDiff, t_ForestNor, s_Aniso, worldPosScaled, triW, vertN, weights.x, albedoSum, normalSum);
    albedoSum = saturate(albedoSum * GREEN_BOOST);
    AccumulateLayer(t_DirtDiff,   t_DirtNor,   s_Aniso, worldPosScaled, triW, vertN, weights.y, albedoSum, normalSum);
    AccumulateLayer(t_RockDiff,   t_RockNor,   s_Aniso, worldPosScaled, triW, vertN, weights.z, albedoSum, normalSum);
    AccumulateLayer(t_SnowDiff,   t_SnowNor,   s_Aniso, worldPosScaled, triW, vertN, weights.w, albedoSum, normalSum);

    float3 worldN = normalize(normalSum + 1e-5 * vertN);

    float3 lightDir    = -normalize(sunLightDir);
    float  diffuse     = max(dot(worldN, lightDir), 0);
    uint   cascadeIdx  = SelectShadowCascade(i_viewZ, cascadeSplits);
    float  notInShadow = SampleShadowCascade(i_worldPos, cascadeIdx);
    float3 lighting    = float3(0.15, 0.15, 0.15)
                       + 0.85 * sunColor * diffuse * notInShadow;

    o_color = float4(lighting * albedoSum, 1);
}
