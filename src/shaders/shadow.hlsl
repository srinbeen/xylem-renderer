#include "../include/macros.h"

#pragma pack_matrix(row_major)

cbuffer CB : register(b0)
{
    float4x4 view;
    float4x4 projection;
    float4x4 lightViewProj;
    float3   sunLightDir;
    float    _pad0;
    float3x4 _pad1;
};

#if XYLEM_USE_STRUCTURED_BUFFER
	struct RootConstant
	{
		uint instanceOffset;
	};
	ConstantBuffer<RootConstant> rootConstant : register(b1);

	struct Instance {
		float4x4 model;
		float3x3 normal;
	};
	StructuredBuffer<Instance> instanceBuffer : register(t0);
#endif

void tree_vs(
	in float3 	i_pos 		: POSITION,

	#if !XYLEM_USE_STRUCTURED_BUFFER
	in float4	im_model[4]	: MODEL_MATRIX,
	in uint 	i_id 		: SV_InstanceID,
	#endif


	out float4 	o_pos 		: SV_Position
)
{
	float4x4 model;

#if XYLEM_USE_STRUCTURED_BUFFER
	uint trueID = rootConstant.instanceOffset + i_id;
	model = instanceBuffer[trueID].model;
#else
	model = float4x4(
		im_model[0],
		im_model[1],
		im_model[2],
		im_model[3]
	);
#endif

	o_pos = mul(mul(float4(i_pos, 1), model), lightViewProj);
}

void terrain_vs(
    in float3  i_pos    : POSITION,

    out float4 o_pos    : SV_Position
)
{
    o_pos = mul(float4(i_pos, 1), lightViewProj);
}
