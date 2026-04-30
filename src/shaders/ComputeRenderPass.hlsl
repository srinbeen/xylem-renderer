#pragma pack_matrix(row_major)

#include "types.hlsli"
#include "ShaderRegisterMap.hlsli"

static const uint NUM_CASCADES = 4;

cbuffer CB : register(XY_REG_B_COMPUTE_SCENE_CB_FRAME)
{
    float4x4 viewProj;
    float4x4 viewMatrix;
    float4x4 lightViewProj[NUM_CASCADES];
    float3   sunLightDir;
    float    _pad0;
    float4   cascadeSplits;
};

struct RootConstant { uint slot; };
ConstantBuffer<RootConstant> rc : register(XY_REG_B_COMPUTE_SCENE_PUSH_C_SLOT);

// Layout matches C++ Render::InstanceBufferEntry (104 bytes, packed — HLSL
// StructuredBuffers use natural layout, not cbuffer 16-byte-row padding).
struct InstanceRenderData
{
    float4x4 model;     // offset 0,  size 64
    float3x3 normal;    // offset 64, size 36
    uint     treeId;    // offset 100, size 4
};

StructuredBuffer<uint>                  visBuf       : register(XY_REG_T_COMPUTE_SCENE_SRV_VIS);
StructuredBuffer<InstanceRenderData>    instBuf      : register(XY_REG_T_COMPUTE_SCENE_SRV_INSTANCES);
StructuredBuffer<uint>                  slotOffsets  : register(XY_REG_T_COMPUTE_SCENE_SRV_SLOT_OFFSETS);


void main_vs(
    in float3  i_pos       : POSITION,
    in float3  i_normal    : NORMAL,
    in float3  i_tangent   : TANGENT,
    in float3  i_bitangent : BITANGENT,
    in float2  i_uv        : UV,
    in uint    i_id        : SV_InstanceID,

    out float4 o_pos       : SV_Position,
    out float3 o_worldPos  : WORLD_POS,
    out float  o_viewZ     : VIEW_Z,
    out float3 o_normal    : NORMAL,
    out float3 o_tangent   : TANGENT,
    out float3 o_bitangent : BITANGENT,
    out float2 o_uv        : UV
)
{
    uint trueID    = visBuf[slotOffsets[rc.slot] + i_id];
    float4x4 model = instBuf[trueID].model;
    float3x3 normalMat = instBuf[trueID].normal;

    float4 worldPos = mul(float4(i_pos, 1), model);

    o_pos       = mul(worldPos, viewProj);
    o_worldPos  = worldPos.xyz;
    o_viewZ     = mul(worldPos, viewMatrix).z;
    o_normal    = normalize(mul(i_normal,    normalMat));

    // Leaf vertices (uv.x < 0) carry color in TANGENT — pass through without rotating.
    if (i_uv.x < 0.0)
    {
        o_tangent   = i_tangent;
        o_bitangent = i_bitangent;
    }
    else
    {
        o_tangent   = normalize(mul(i_tangent,   normalMat));
        o_bitangent = normalize(mul(i_bitangent, normalMat));
    }
    o_uv        = i_uv;
}


Texture2D              t_Diffuse        : register(XY_REG_T_COMPUTE_SCENE_TEX_DIFFUSE);
Texture2D              t_NormalMap      : register(XY_REG_T_COMPUTE_SCENE_TEX_NORMAL_MAP);
Texture2DArray         t_ShadowMap      : register(XY_REG_T_COMPUTE_SCENE_TEX_SHADOW_MAP);

SamplerState           s_Sampler        : register(XY_REG_S_COMPUTE_SCENE_SAMPLER_MAIN);
SamplerComparisonState s_ShadowSampler  : register(XY_REG_S_COMPUTE_SCENE_SAMPLER_SHADOW);


float SampleShadowCascade(float3 worldPos, uint cascadeIdx)
{
    float4 posLS = mul(float4(worldPos, 1), lightViewProj[cascadeIdx]);
    float2 shadowUV = posLS.xy * float2(0.5, -0.5) + 0.5;

    uint width, height, elements;
    t_ShadowMap.GetDimensions(width, height, elements);
    float2 texelSize = 1.0 / float2(width, height);

    float shadow = 0.0;

    static const float weights[3][3] = {
        { 1.0, 2.0, 1.0 },
        { 2.0, 4.0, 2.0 },
        { 1.0, 2.0, 1.0 }
    };

    [unroll]
    for (int x = -1; x <= 1; ++x)
    {
        [unroll]
        for (int y = -1; y <= 1; ++y)
        {
            float2 offset = float2(x, y) * texelSize;
            float weight = weights[x + 1][y + 1];

            shadow += weight * t_ShadowMap.SampleCmpLevelZero(
                s_ShadowSampler,
                float3(shadowUV + offset, float(cascadeIdx)),
                posLS.z
            );
        }
    }

    return shadow / 16.0;
}


void main_ps(
    in float4  i_pos       : SV_Position,
    in float3  i_worldPos  : WORLD_POS,
    in float   i_viewZ     : VIEW_Z,
    in float3  i_normal    : NORMAL,
    in float3  i_tangent   : TANGENT,
    in float3  i_bitangent : BITANGENT,
    in float2  i_uv        : UV,

    out float4 o_color     : SV_Target0
)
{
    // Cascade selection (shared between trunk + leaf paths)
    uint cascadeIdx = 3;
    if      (i_viewZ < cascadeSplits.x) cascadeIdx = 0;
    else if (i_viewZ < cascadeSplits.y) cascadeIdx = 1;
    else if (i_viewZ < cascadeSplits.z) cascadeIdx = 2;

    float3 lightDir = -normalize(sunLightDir);
    float notInShadow = SampleShadowCascade(i_worldPos, cascadeIdx);

    // Leaves are tagged with uv = (-1, -1) by the CPU emitter. Double-sided lambert,
    // color packed into the TANGENT slot per-vertex.
    if (i_uv.x < 0.0)
    {
        float3 leafColor = i_tangent;
        float3 N = normalize(i_normal);
        float diffuse = abs(dot(N, lightDir));

        float ambient  = 0.20;
        float lighting = ambient + (1.0 - ambient) * diffuse * notInShadow;
        o_color = float4(lighting * leafColor, 1);
        return;
    }

    // Trunk / branchlet path: tangent-space normal mapping
    float3 T = normalize(i_tangent);
    float3 B = normalize(i_bitangent);
    float3 N = normalize(i_normal);
    float3x3 TBN = float3x3(T, B, N);

    float3 tangentNormal = normalize(t_NormalMap.Sample(s_Sampler, i_uv).rgb * 2.0 - 1.0);
    float3 worldNormal = normalize(mul(tangentNormal, TBN));

    float diffuse = max(dot(worldNormal, lightDir), 0);

    float ambient  = 0.15;
    float lighting = ambient + (1.0 - ambient) * diffuse * notInShadow;
    o_color = float4(lighting * t_Diffuse.Sample(s_Sampler, i_uv).rgb, 1);
}
