#pragma pack_matrix(row_major)

#include "ShaderRegisterMap.hlsli"
#include "LeafCommon.hlsli"

cbuffer BakeCB : register(XY_REG_B_COMPUTE_IMPOSTORBAKE_CB_BAKE)
{
    float4x4    mvp;
    float4      _pad[12];
};

Texture2D    t_Diffuse   : register(XY_REG_T_COMPUTE_IMPOSTORBAKE_TEX_DIFFUSE);
Texture2D    t_NormalMap : register(XY_REG_T_COMPUTE_IMPOSTORBAKE_TEX_NORMAL_MAP);
SamplerState s_Sampler   : register(XY_REG_S_COMPUTE_IMPOSTORBAKE_SAMPLER_MAIN);

// Leaf bake bindings live on a separate root layout — one CB at b0 (mvp shared with trunk path),
// a leaf-instance push constant at b1, and the per-asset leaf instance buffer at t0.
cbuffer LeafBakePush : register(b1)
{
    uint g_LeafOffset;
    uint g_LeafCount;
    uint _leafPad0;
    uint _leafPad1;
};

StructuredBuffer<LeafInstanceData> g_BakeLeaves : register(t0);

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
    o.tangent   = normalize(i_tangent);
    o.bitangent = normalize(i_bitangent);
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

    // Trunk / branchlet path: tangent-space normal map sample. Leaves bake through
    // a dedicated leaf bake pass (asset-local leaf instances, no normal-map sampling).
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

struct LeafBakeV2P
{
    float4 pos    : SV_Position;
    float3 color  : COLOR0;
    float3 normal : NORMAL;
};

void leaf_bake_vs(in uint vertexId : SV_VertexID, out LeafBakeV2P o)
{
    uint leafIdx = LeafIndexFromVertex(vertexId);
    uint vertInLeaf = LeafVertexInLeaf(vertexId);
    LeafInstanceData leaf = g_BakeLeaves[g_LeafOffset + leafIdx];

    float3 localPos, localNormal;
    BuildLeafVertex(leaf, vertInLeaf, localPos, localNormal);

    o.pos    = mul(float4(localPos, 1.0), mvp);
    o.color  = leaf.color.rgb;
    o.normal = localNormal;
}

BakeOutput leaf_bake_ps(in LeafBakeV2P i)
{
    BakeOutput o;
    o.albedoAlpha = float4(i.color, 1.0);
    o.normal = float4(saturate(normalize(i.normal) * 0.5 + 0.5), 1.0);
    return o;
}
