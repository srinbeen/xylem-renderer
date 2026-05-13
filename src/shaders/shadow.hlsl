#include "../include/macros.h"
#include "ShaderRegisterMap.hlsli"

#pragma pack_matrix(row_major)

cbuffer CB : register(XY_REG_B_TRADITIONAL_SHADOW_CB_FRAME)
{
    float4x4 viewProj;
    float4x4 viewMatrix;
    float4x4 lightViewProj[XYLEM_NUM_CASCADES];
    float3   sunLightDir;
    float    _pad0;
    float3   sunColor;
    float    _pad0b;
    float4   cascadeSplits;
    float    _pad1[20];
};

struct CascadeIdx { uint idx; };
ConstantBuffer<CascadeIdx> cascadeRC : register(XY_REG_B_TRADITIONAL_SHADOW_PUSH_C_CASCADE_INDEX);

void tree_vs(
	in float3 	i_pos 		: POSITION,
	in float4	im_model[4]	: MODEL_MATRIX,
	in uint 	i_id 		: SV_InstanceID,

	out float4 	o_pos 		: SV_Position
)
{
	float4x4 model = float4x4(
		im_model[0],
		im_model[1],
		im_model[2],
		im_model[3]
	);

	o_pos = mul(mul(float4(i_pos, 1), model), lightViewProj[cascadeRC.idx]);
}

void terrain_vs(
    in float3  i_pos    : POSITION,

    out float4 o_pos    : SV_Position
)
{
    o_pos = mul(float4(i_pos, 1), lightViewProj[cascadeRC.idx]);
}
