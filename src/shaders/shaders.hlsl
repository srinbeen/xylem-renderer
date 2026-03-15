#include "../include/macros.h"

#pragma pack_matrix(row_major)

cbuffer CB : register(b0)
{
    float4x4 view;
    float4x4 projection;

	float4x4 padding[2];
};

#if PIPELINER_USE_STRUCTURED_BUFFER
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

void main_vs(
	// inputs
	in float3 	i_pos 		: POSITION,
    in float3 	i_normal 	: NORMAL,
    in float2 	i_uv 		: UV,

	in uint 	i_id 		: SV_InstanceID,
	// in uint 	i_id_offset : SV_StartInstanceLocation,

	#if !PIPELINER_USE_STRUCTURED_BUFFER
	in float4	im_model[4]	: MODEL_MATRIX,
	in float3	im_norm[3]	: NORMAL_MATRIX,
	#endif

	// outputs
	out float4 	o_pos 		: SV_Position,
	out float3 	o_normal 	: NORMAL,
	out float2 	o_uv 		: UV
)
{

	float4x4 viewProj = mul(view, projection);
	float4x4 model;
	float3x3 normal;

#if PIPELINER_USE_STRUCTURED_BUFFER
	uint trueID = rootConstant.instanceOffset + i_id;
	model = instanceBuffer[trueID].model;
	normal = instanceBuffer[trueID].normal;
#else
	model = float4x4(
		im_model[0],
		im_model[1],
		im_model[2],
		im_model[3]
	);

	normal = float3x3(
		im_norm[0],
		im_norm[1],
		im_norm[2]
	);
#endif

	o_pos = mul(
		mul(float4(i_pos, 1), model), 
		viewProj
	);
	o_normal = normalize(mul(i_normal, normal));
	o_uv = i_uv;
}


Texture2D t_Texture : register(t0);
SamplerState s_Sampler : register(s0);

void main_ps(
	in float4 	i_pos 		: SV_Position,
	in float3 	i_normal 	: NORMAL,
	in float2 	i_uv 		: UV,
	
	out float4 o_color : SV_Target0
)
{
	
	float3 N = normalize(i_normal);
	float dif = max(dot(N, -normalize(float3(0,0,1))), 0);
    o_color = float4((dif * t_Texture.Sample(s_Sampler, i_uv).rgb), 1);
}
