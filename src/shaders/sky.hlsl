#include <donut/shaders/sky.hlsli>

#pragma pack_matrix(row_major)

cbuffer c_Sky : register(b0)
{
    SkyConstants g_Sky;
};

void sky_vs(
    in uint iVertex : SV_VertexID,
    out float4 o_posClip : SV_Position,
    out float2 o_uv : UV)
{
    uint u = iVertex & 1;
    uint v = (iVertex >> 1) & 1;

    o_posClip = float4(float(u) * 2 - 1, 1 - float(v) * 2, 0, 1);
    o_uv = float2(u, v);
}

void sky_ps(
    in float4 i_position : SV_Position,
    in float2 i_uv : UV,
    out float4 o_color : SV_Target0)
{
    float4 clipPos;
    clipPos.x = i_uv.x * 2 - 1;
    clipPos.y = 1 - i_uv.y * 2;
    clipPos.z = 0.5;
    clipPos.w = 1;
    float4 translatedWorldPos = mul(clipPos, g_Sky.matClipToTranslatedWorld);
    float3 direction = normalize(translatedWorldPos.xyz / translatedWorldPos.w);

    float angularSizeOfPixel = max(length(ddx(direction)), length(ddy(direction)));

    o_color.rgb = ProceduralSky(g_Sky.params, direction, angularSizeOfPixel);
    o_color.a = 0;
}
