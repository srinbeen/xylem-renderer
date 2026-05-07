#ifndef XYLEM_TERRAIN_SHADING_HLSLI
#define XYLEM_TERRAIN_SHADING_HLSLI

#pragma pack_matrix(row_major)

#define GREEN_BOOST (float3(1.0, 1.25, 1.0))

struct TerrainShadingParams {
    float tileSize;
    float forestToDirtY;
    float dirtToSnowY;
    float bandWidth;
    float slopeLo;
    float slopeHi;
    float macroNoiseAmp;
};

float Hash2D(float2 p) {
    p = frac(p * float2(443.897, 441.423));
    p += dot(p, p.yx + 19.19);
    return frac((p.x + p.y) * p.x);
}

// Bilinear-interpolated value noise — smooth, no aliasing, ~5 ALU ops per call
// over Hash2D. Use this for any spatially-continuous use of noise.
float ValueNoise2D(float2 p) {
    float2 i = floor(p);
    float2 f = frac(p);
    float  a = Hash2D(i);
    float  b = Hash2D(i + float2(1, 0));
    float  c = Hash2D(i + float2(0, 1));
    float  d = Hash2D(i + float2(1, 1));
    float2 u = f * f * (3.0 - 2.0 * f);  // smoothstep
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

float3 TriplanarBlendNormals(
    float3 nXts, float3 nYts, float3 nZts,
    float3 vertN, float3 triW)
{
    float3 wX = float3(nXts.z * sign(vertN.x), nXts.y, nXts.x);
    float3 wY = float3(nYts.x, nYts.z * sign(vertN.y), nYts.y);
    float3 wZ = float3(nZts.x, nZts.y, nZts.z * sign(vertN.z));
    return wX * triW.x + wY * triW.y + wZ * triW.z;
}

void AccumulateLayer(
    Texture2D    diff,
    Texture2D    nor,
    SamplerState samp,
    float3       worldPosScaled,
    float3       triW,
    float3       vertN,
    float        layerWeight,
    inout float3 albedoSum,
    inout float3 normalSum)
{
    if (layerWeight < 0.01) return;

    float2 uvX = worldPosScaled.zy;
    float2 uvY = worldPosScaled.xz;
    float2 uvZ = worldPosScaled.xy;

    float3 diffX = diff.Sample(samp, uvX).rgb;
    float3 diffY = diff.Sample(samp, uvY).rgb;
    float3 diffZ = diff.Sample(samp, uvZ).rgb;
    float3 layerAlbedo = diffX * triW.x + diffY * triW.y + diffZ * triW.z;

    float3 nXts = normalize(nor.Sample(samp, uvX).xyz * 2.0 - 1);
    float3 nYts = normalize(nor.Sample(samp, uvY).xyz * 2.0 - 1);
    float3 nZts = normalize(nor.Sample(samp, uvZ).xyz * 2.0 - 1);
    float3 layerNormal = TriplanarBlendNormals(nXts, nYts, nZts, vertN, triW);

    albedoSum += layerWeight * layerAlbedo;
    normalSum += layerWeight * layerNormal;
}

float4 ComputeLayerWeights(
    float3 worldPos, float3 vertN, float height,
    TerrainShadingParams p)
{
    float slope        = 1.0 - saturate(vertN.y);
    float slopeWeight  = smoothstep(p.slopeLo, p.slopeHi, slope);

    float jitter       = (ValueNoise2D(worldPos.xz * 0.05) - 0.5) * 2.0 * p.macroNoiseAmp;
    float yJ           = height + jitter;

    float wForest      = 1.0 - smoothstep(p.forestToDirtY - p.bandWidth,
                                          p.forestToDirtY + p.bandWidth, yJ);
    float wSnow        =       smoothstep(p.dirtToSnowY  - p.bandWidth,
                                          p.dirtToSnowY  + p.bandWidth, yJ);
    float wDirt        = max(0.0, 1.0 - wForest - wSnow);

    float4 flatWeights = float4(wForest, wDirt, 0.0, wSnow);
    float4 rockOnly    = float4(0.0, 0.0, 1.0, 0.0);
    return lerp(flatWeights, rockOnly, slopeWeight);
}

float3 ComputeTriplanarWeights(float3 vertN) {
    float3 absN = abs(vertN);
    float3 triW = pow(absN, 4.0);
    triW       /= max(dot(triW, float3(1, 1, 1)), 1e-5);
    return triW;
}

#endif // XYLEM_TERRAIN_SHADING_HLSLI
