#pragma pack_matrix(row_major)
#include "ShaderRegisterMap.hlsli"
#include "../include/macros.h"


struct SDSMInput {
    float4x4 worldToLight;
    float4x4 viewToWorldToLight;    // = inverse(viewMatrix) * worldToLight
    float4   sceneBboxMinLS;        // bbox min light space
    float4   sceneBboxMaxLS;
    float    tanHalfFovX;
    float    tanHalfFovY;
    float    projA;                 // projection matrix [2][2] -- depth = (A*viewZ + B) / viewZ
    float    projB;                 // projection matrix [3][2]
    float    regionEnvelopeNear;    // CPU-side conservative near from visible region corners
    float    regionEnvelopeFar;     // CPU-side conservative far  from visible region corners
    float    cameraNearPlane;
    uint     shadowRes;
    uint     maxHiZMip;             // mip index of the 1x1 top of the Hi-Z chain
    float    pssmLambda;            // PSSM/SDSM blend: 0=linear splits, 1=log splits
    float2   _pad;
};
ConstantBuffer<SDSMInput> cb : register(XY_REG_B_COMPUTE_SDSM_CB_INPUT);
// RGBA32_FLOAT: .r = Hi-Z occluder min (sky included, unused here),
//               .g = nearest max (SDSM near),
//               .b = sky-excluded min (SDSM far).
Texture2D<float4> hizTexture : register(XY_REG_T_COMPUTE_SDSM_SRV_HI_Z);

struct SDSMCascadeOut {
    float4x4 lightViewProj[XYLEM_NUM_CASCADES];
    float4   cascadeSplits;
    float4   shadowCasterMinLS[XYLEM_NUM_CASCADES];
    float4   shadowCasterMaxLS[XYLEM_NUM_CASCADES];
    float4   debugDepthExtents;  // (nearDepthVal, farDepthVal, tightNear, tightFar)
};
RWStructuredBuffer<SDSMCascadeOut> outBuffer : register(XY_REG_U_COMPUTE_SDSM_UAV_CASCADE_OUT);

groupshared float g_splits[XYLEM_NUM_CASCADES + 1];

[numthreads(XYLEM_NUM_CASCADES, 1, 1)]
void BuildCascades(uint3 gtid : SV_GroupThreadID)
{
    uint c = gtid.x;

    if (c == 0) {
        // Pull 1x1 reduced depth from top of Hi-Z chain.
        // .g = nearest (max),  .b = sky-excluded farthest (min, sky encoded as 1).
        float4 globalMinMax = hizTexture.Load(int3(0, 0, cb.maxHiZMip));
        float nearDepthVal = globalMinMax.g;
        float farDepthVal  = globalMinMax.b;

        // D3D-style perspective: depth_ndc = A + B/viewZ  ->  viewZ = B / (depth - A)
        // Fallback when no geometry is visible: full-sky frame collapses .b to the
        // sentinel 1.0, no-geometry frame collapses .g to 0.
        float sdsmFar;
        float sdsmNear;
        
        // if not looking at scene, use regionEnvelope as bounds
        #if XYLEM_USE_REVERSE_Z
        const bool notInSceneBounds = farDepthVal == 1.0 || nearDepthVal == 0.0;
        #else
        const bool notInSceneBounds = nearDepthVal == 1.0 || farDepthVal == 0.0;
        #endif
        
        if (notInSceneBounds) {
            sdsmFar  = cb.regionEnvelopeFar;
            sdsmNear = cb.regionEnvelopeNear;
        } else {
            sdsmFar  = cb.projB / (farDepthVal  - cb.projA);
            sdsmNear = cb.projB / (nearDepthVal - cb.projA);
        }

        // SDSM values based on pre-pass so frame N-1 objects
        // fast camera movement might make bounds inaccurate for one frame
        // add padding as a heuristic
        float tightNear = max(sdsmNear * (1-XYLEM_SDSM_PADDING), cb.regionEnvelopeNear);
        float tightFar  = min(sdsmFar * (1+XYLEM_SDSM_PADDING), cb.regionEnvelopeFar);
        tightNear = max(tightNear, cb.cameraNearPlane);
        tightFar  = max(tightFar,  tightNear + 1.0);    // incase they are very close or get swapped

        // Log-split (constant far/near ratio)
        const float lambda = cb.pssmLambda;
        float pssmNear = tightNear;

        g_splits[0] = tightNear;
        [unroll]
        for (uint i = 1; i <= XYLEM_NUM_CASCADES; i++) {
            float t = float(i) / float(XYLEM_NUM_CASCADES);
            float logSplit = pssmNear * pow(tightFar / pssmNear, t);
            float linSplit = pssmNear + (tightFar - pssmNear) * t;
            g_splits[i] = lambda * logSplit + (1.0 - lambda) * linSplit;
        }

        // readback values
        outBuffer[0].debugDepthExtents = float4(nearDepthVal, farDepthVal, tightNear, tightFar);
    }

    GroupMemoryBarrierWithGroupSync();

    float cNear = g_splits[c];
    float cFar  = g_splits[c + 1];

    // 8 frustum corners in view space at this cascade's near/far planes.
    float3 cornersVS[8] = {
        float3(-cb.tanHalfFovX * cNear, -cb.tanHalfFovY * cNear, cNear),
        float3( cb.tanHalfFovX * cNear, -cb.tanHalfFovY * cNear, cNear),
        float3(-cb.tanHalfFovX * cNear,  cb.tanHalfFovY * cNear, cNear),
        float3( cb.tanHalfFovX * cNear,  cb.tanHalfFovY * cNear, cNear),
        float3(-cb.tanHalfFovX * cFar,  -cb.tanHalfFovY * cFar,  cFar ),
        float3( cb.tanHalfFovX * cFar,  -cb.tanHalfFovY * cFar,  cFar ),
        float3(-cb.tanHalfFovX * cFar,   cb.tanHalfFovY * cFar,  cFar ),
        float3( cb.tanHalfFovX * cFar,   cb.tanHalfFovY * cFar,  cFar )
    };

    float3 bboxMinLS = float3( 1e30,  1e30,  1e30);
    float3 bboxMaxLS = float3(-1e30, -1e30, -1e30);
    [unroll]
    for (int i = 0; i < 8; i++) {
        float3 ls = mul(float4(cornersVS[i], 1), cb.viewToWorldToLight).xyz;
        bboxMinLS = min(bboxMinLS, ls);
        bboxMaxLS = max(bboxMaxLS, ls);
    }

    
    float3 centroidVS = float3(0.0, 0.0, (cNear + cFar) * 0.5);
    float radius = length(cornersVS[4] - centroidVS);
    float3 centroidLS = mul(float4(centroidVS, 1.0), cb.viewToWorldToLight).xyz;

    // Z range comes from the 8-corner pass above; XY of bboxMin/MaxLS no longer used.
    float zMinLS = min(bboxMinLS.z, cb.sceneBboxMinLS.z);
    float zMaxLS = min(bboxMaxLS.z, cb.sceneBboxMaxLS.z);

    float T = (2.0 * radius) / float(cb.shadowRes);
    if (T > 0.0) {
        centroidLS.x = round(centroidLS.x / T) * T;
        centroidLS.y = round(centroidLS.y / T) * T;
    }

    // Pack into the bbox vars consumed by the ortho-construction block below.
    bboxMinLS = float3(centroidLS.x - radius, centroidLS.y - radius, zMinLS);
    bboxMaxLS = float3(centroidLS.x + radius, centroidLS.y + radius, zMaxLS);

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
    float4x4 lightViewProj = mul(cb.worldToLight, lightProj);

    outBuffer[0].lightViewProj[c]     = lightViewProj;
    outBuffer[0].shadowCasterMinLS[c] = float4(bboxMinLS, 0.0);
    outBuffer[0].shadowCasterMaxLS[c] = float4(bboxMaxLS, 0.0);

    if (c == 0) {
        float4 packedSplits = float4(g_splits[XYLEM_NUM_CASCADES],
                                     g_splits[XYLEM_NUM_CASCADES],
                                     g_splits[XYLEM_NUM_CASCADES],
                                     g_splits[XYLEM_NUM_CASCADES]);
#if XYLEM_NUM_CASCADES > 0
        packedSplits.x = g_splits[1];
#endif
#if XYLEM_NUM_CASCADES > 1
        packedSplits.y = g_splits[2];
#endif
#if XYLEM_NUM_CASCADES > 2
        packedSplits.z = g_splits[3];
#endif
#if XYLEM_NUM_CASCADES > 3
        packedSplits.w = g_splits[4];
#endif
        outBuffer[0].cascadeSplits = packedSplits;
    }
}
