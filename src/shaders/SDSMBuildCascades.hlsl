#pragma pack_matrix(row_major)

static const uint NUM_CASCADES = 4;

cbuffer SDSMInput : register(b0)
{
    float4x4 worldToLight;          // view space is camera-space; worldToLight maps world->light
    float4x4 viewToWorldToLight;    // = inverse(viewMatrix) * worldToLight
    float4   sceneBboxMinLS;        // .xyz used; scene AABB transformed into light space
    float4   sceneBboxMaxLS;
    float    tanHalfFovX;
    float    tanHalfFovY;
    float    projA;                 // projection matrix [2][2] — depth = (A*viewZ + B) / viewZ
    float    projB;                 // projection matrix [3][2]
    float    regionEnvelopeNear;    // CPU-side conservative near from visible region corners
    float    regionEnvelopeFar;     // CPU-side conservative far  from visible region corners
    float    cameraNearPlane;
    uint     shadowRes;
    uint     maxHiZMip;             // mip index of the 1x1 top of the Hi-Z chain
    float    pssmLambda;            // PSSM/SDSM blend: 0=linear splits, 1=log splits
    float2   _pad;
};

Texture2D<float2> hizTexture : register(t0);

struct SDSMCascadeOut {
    float4x4 lightViewProj[NUM_CASCADES];
    float4   cascadeSplits;
    float4   shadowCasterMinLS[NUM_CASCADES];
    float4   shadowCasterMaxLS[NUM_CASCADES];
};
RWStructuredBuffer<SDSMCascadeOut> outBuffer : register(u0);

groupshared float g_splits[NUM_CASCADES + 1];

[numthreads(NUM_CASCADES, 1, 1)]
void BuildCascades(uint3 gtid : SV_GroupThreadID)
{
    uint c = gtid.x;

    if (c == 0) {
        // Pull 1x1 min/max depth from top of Hi-Z chain.
        float2 globalMinMax = hizTexture.Load(int3(0, 0, maxHiZMip));
        float farDepthVal  = globalMinMax.r;   // reverse-Z: min value -> farthest
        float nearDepthVal = globalMinMax.g;   // reverse-Z: max value -> nearest

        // D3D-style perspective: depth_ndc = A + B/viewZ  ->  viewZ = B / (depth - A)
        // Guard against degenerate/empty frame (depth==cleared == 0 in reverse-Z).
        float sdsmFar;
        float sdsmNear;
        if (farDepthVal <= 0.0 || nearDepthVal <= 0.0) {
            // Empty depth buffer: fall back to region envelope
            sdsmFar  = regionEnvelopeFar;
            sdsmNear = regionEnvelopeNear;
        } else {
            sdsmFar  = projB / (farDepthVal  - projA);
            sdsmNear = projB / (nearDepthVal - projA);
        }

        // Safety expand SDSM bounds (within-region instance pop-in), then clamp
        // outward against CPU region envelope (new-region pop-in).
        float tightNear = min(sdsmNear * 0.95, regionEnvelopeNear);
        float tightFar  = max(sdsmFar  * 1.05, regionEnvelopeFar);
        tightNear = max(tightNear, cameraNearPlane);
        tightFar  = max(tightFar,  tightNear + 1.0);    // incase they are very close or get swapped

        // Log-split (constant far/near ratio) — mirrors ViewHandler::computeCascades.
        const float lambda = pssmLambda;
        float pssmNear = tightNear;

        g_splits[0] = tightNear;
        [unroll]
        for (uint i = 1; i <= NUM_CASCADES; i++) {
            float t = float(i) / float(NUM_CASCADES);
            float logSplit = pssmNear * pow(tightFar / pssmNear, t);
            float linSplit = pssmNear + (tightFar - pssmNear) * t;
            g_splits[i] = lambda * logSplit + (1.0 - lambda) * linSplit;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    float cNear = g_splits[c];
    float cFar  = g_splits[c + 1];

    // 8 frustum corners in view space at this cascade's near/far planes.
    float3 cornersVS[8] = {
        float3(-tanHalfFovX * cNear, -tanHalfFovY * cNear, cNear),
        float3( tanHalfFovX * cNear, -tanHalfFovY * cNear, cNear),
        float3(-tanHalfFovX * cNear,  tanHalfFovY * cNear, cNear),
        float3( tanHalfFovX * cNear,  tanHalfFovY * cNear, cNear),
        float3(-tanHalfFovX * cFar,  -tanHalfFovY * cFar,  cFar ),
        float3( tanHalfFovX * cFar,  -tanHalfFovY * cFar,  cFar ),
        float3(-tanHalfFovX * cFar,   tanHalfFovY * cFar,  cFar ),
        float3( tanHalfFovX * cFar,   tanHalfFovY * cFar,  cFar )
    };

    float3 bboxMinLS = float3( 1e30,  1e30,  1e30);
    float3 bboxMaxLS = float3(-1e30, -1e30, -1e30);
    [unroll]
    for (int i = 0; i < 8; i++) {
        float3 ls = mul(float4(cornersVS[i], 1), viewToWorldToLight).xyz;
        bboxMinLS = min(bboxMinLS, ls);
        bboxMaxLS = max(bboxMaxLS, ls);
    }

    // Square XY extent for uniform texels across both axes.
    float extX  = bboxMaxLS.x - bboxMinLS.x;
    float extY  = bboxMaxLS.y - bboxMinLS.y;
    float maxXY = max(extX, extY);
    float growX = 0.5 * (maxXY - extX);
    float growY = 0.5 * (maxXY - extY);
    bboxMinLS.x -= growX;
    bboxMaxLS.x += growX;
    bboxMinLS.y -= growY;
    bboxMaxLS.y += growY;

    // Texel snap so shadow samples fall on a stable world-space grid.
    float texelSize = maxXY / float(shadowRes);
    if (texelSize > 0.0) {
        bboxMinLS.x = floor(bboxMinLS.x / texelSize) * texelSize;
        bboxMinLS.y = floor(bboxMinLS.y / texelSize) * texelSize;
        bboxMaxLS.x = bboxMinLS.x + maxXY;
        bboxMaxLS.y = bboxMinLS.y + maxXY;
    }

    // Z clamp to scene bbox in light space (matches ViewHandler: both mins use min()).
    bboxMinLS.z = min(bboxMinLS.z, sceneBboxMinLS.z);
    bboxMaxLS.z = min(bboxMaxLS.z, sceneBboxMaxLS.z);

    // Build orthographic projection — mirrors donut::math::orthoProjD3DStyle(l, r, b, t, zn, zf):
    //   [2/(r-l),  0,        0,           0]
    //   [0,        2/(t-b),  0,           0]
    //   [0,        0,        1/(zf-zn),   0]
    //   [-(l+r)*xs, -(b+t)*ys, -zn*zs,    1]
    float l = bboxMinLS.x, r = bboxMaxLS.x;
    float b = bboxMinLS.y, t = bboxMaxLS.y;
    float zn = bboxMinLS.z, zf = bboxMaxLS.z;
    float xScale = 1.0 / (r - l);
    float yScale = 1.0 / (t - b);
    float zScale = 1.0 / (zf - zn);

    float4x4 lightProj = float4x4(
        2.0 * xScale,    0,              0,             0,
        0,               2.0 * yScale,   0,             0,
        0,               0,              zScale,        0,
        -(l + r) * xScale, -(b + t) * yScale, -zn * zScale, 1.0
    );

    // Compose worldToLight then lightProj (row-vector convention: v*W*P).
    float4x4 lightViewProj = mul(worldToLight, lightProj);

    outBuffer[0].lightViewProj[c]     = lightViewProj;
    outBuffer[0].shadowCasterMinLS[c] = float4(bboxMinLS, 0.0);
    outBuffer[0].shadowCasterMaxLS[c] = float4(bboxMaxLS, 0.0);

    if (c == 0) {
        outBuffer[0].cascadeSplits = float4(g_splits[1], g_splits[2], g_splits[3], g_splits[4]);
    }
}
