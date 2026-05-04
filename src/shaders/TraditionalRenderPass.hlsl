#include "../include/macros.h"
#include "ShaderRegisterMap.hlsli"

#pragma pack_matrix(row_major)

cbuffer CB : register(XY_REG_B_TRADITIONAL_TREE_CB_FRAME)
{
    float4x4 viewProj;
    float4x4 viewMatrix;
    float4x4 lightViewProj[XYLEM_NUM_CASCADES];
    float3   sunLightDir;
    float    _pad0;
    float4   cascadeSplits;
    float4   _pad1[6];
};

void main_vs(
	// inputs
	in float3 	i_pos 		: POSITION,
    in float3 	i_normal 	: NORMAL,
	in float3 	i_tangent 	: TANGENT,
	in float3 	i_bitangent : BITANGENT,
    in float2 	i_uv 		: UV,

	in float4	im_model[4]	: MODEL_MATRIX,
	in float3	im_norm[3]	: NORMAL_MATRIX,
	in uint 	i_id 		: SV_InstanceID,

	// outputs
	out float4 	o_pos 		: SV_Position,
	out float3	o_worldPos 	: WORLD_POS,
	out float	o_viewZ 	: VIEW_Z,
	out float3 	o_normal 	: NORMAL,
	out float3 	o_tangent 	: TANGENT,
	out float3 	o_bitangent	: BITANGENT,
	out float2 	o_uv 		: UV
)
{
	float4x4 model = float4x4(
		im_model[0],
		im_model[1],
		im_model[2],
		im_model[3]
	);

	float3x3 normalMat = float3x3(
		im_norm[0],
		im_norm[1],
		im_norm[2]
	);

	float4 worldPos	= mul(float4(i_pos, 1), model);
	o_pos 			= mul(worldPos, viewProj);
	o_worldPos		= worldPos.xyz;
	o_viewZ			= mul(worldPos, viewMatrix).z;
	o_normal    	= normalize(mul(i_normal, normalMat));

	// Leaf vertices (uv.x < 0) carry leaf color in the TANGENT slot — skip the rotation +
	// normalize so the RGB values reach the PS intact. Trunk verts take the standard path
	// (tangent transformed for normal mapping).
	if (i_uv.x < 0.0)
	{
		o_tangent   = i_tangent;
		o_bitangent = i_bitangent;
	}
	else
	{
		o_tangent   = normalize(mul(i_tangent, normalMat));
		o_bitangent = normalize(mul(i_bitangent, normalMat));
	}
	o_uv 			= i_uv;
}


Texture2D      t_Diffuse       : register(XY_REG_T_TRADITIONAL_TREE_TEX_DIFFUSE);
Texture2D      t_NormalMap     : register(XY_REG_T_TRADITIONAL_TREE_TEX_NORMAL_MAP);
Texture2DArray t_ShadowMap     : register(XY_REG_T_TRADITIONAL_TREE_TEX_SHADOW_MAP);
SamplerState           s_Sampler        : register(XY_REG_S_TRADITIONAL_TREE_SAMPLER_MAIN);
SamplerComparisonState s_ShadowSampler  : register(XY_REG_S_TRADITIONAL_TREE_SAMPLER_SHADOW);

float SampleShadowCascade(float3 worldPos, uint cascadeIdx)
{
    float4 posLS = mul(float4(worldPos, 1), lightViewProj[cascadeIdx]);
    float2 shadowUV = posLS.xy * float2(0.5, -0.5) + 0.5;

    uint width, height, elements;
    t_ShadowMap.GetDimensions(width, height, elements);
    float2 texelSize = 1.0 / float2(width, height);

    float shadow = 0.0;

	// gaussian filter
    static const float weights[3][3] = {
        { 1.0, 2.0, 1.0 },
        { 2.0, 4.0, 2.0 },
        { 1.0, 2.0, 1.0 }
    };

    [unroll]
    for (int x = -1; x <= 1; ++x)
    {
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
	in float4 	i_pos 		: SV_Position,
	in float3	i_worldPos 	: WORLD_POS,
	in float	i_viewZ 	: VIEW_Z,
	in float3 	i_normal 	: NORMAL,
	in float3 	i_tangent 	: TANGENT,
	in float3 	i_bitangent	: BITANGENT,
	in float2 	i_uv 		: UV,

	out float4 o_color : SV_Target0
)
{
	// Cascade selection (shared between trunk + leaf paths)
	uint cascadeIdx = 3;
	if      (i_viewZ < cascadeSplits.x) cascadeIdx = 0;
	else if (i_viewZ < cascadeSplits.y) cascadeIdx = 1;
	else if (i_viewZ < cascadeSplits.z) cascadeIdx = 2;

	float3 lightDir = -normalize(sunLightDir);
	float notInShadow = SampleShadowCascade(i_worldPos, cascadeIdx);

	// Leaves are tagged with uv = (-1, -1) by the CPU emitter. They take a simple
	// double-sided lambert path (abs(N·L) — both sides lit since cull=none) and read
	// their color from the TANGENT slot, where the per-asset leaf color was packed.
	if (i_uv.x < 0.0)
	{
		float3 leafColor = i_tangent;   // tangent stream carries leaf color for leaf verts
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
