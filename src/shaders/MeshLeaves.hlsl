#include "../include/macros.h"
#include "types.hlsli"
#include "meshlet_types.hlsli"
#include "LeafCommon.hlsli"
#include "ShadowCascadeCommon.hlsli"

#pragma pack_matrix(row_major)

// Mirrors Render::CullConstantBufferEntry through the `viewFrustum` field. The
// leaf binding set binds the full kCullFrameSize range, so any prefix of the C++
// struct is readable here. Anything past `viewFrustum` is unused in this shader.
cbuffer CB : register(b0)
{
    float4x4 viewProj;
    float4x4 viewMatrix;
    float4x4 lightViewProj[XYLEM_NUM_CASCADES];
    float3   sunLightDir;
    float    _pad0;
    float4   cascadeSplits;
    frustum  viewFrustum;
};

cbuffer PushC : register(b1)
{
    uint g_SlotIdx;
};

// Per-frame Hi-Z / AS-cull constants. Same buffer as MeshShaderPass.hlsl's
// trunk path — toggling `g_HizEnabled` via the Hi-Z bypass affects both.
// Bound only by the main + depth leaf binding sets; the shadow leaf path
// uses `leaf_shadow_as` which doesn't read this CB.
cbuffer ASCullCB : register(b2)
{
    float3   g_CameraPos;
    uint     g_HizEnabled;
    float2   g_HizDimensions;
    float    g_MaxHiZMip;
    uint     _ascullPad;
};

StructuredBuffer<uint>               g_VisBuf          : register(t0);
StructuredBuffer<uint>               g_SlotOffsets     : register(t1);
StructuredBuffer<uint>               g_SlotCounts      : register(t2);
StructuredBuffer<InstanceRenderData> g_Instances       : register(t3);
StructuredBuffer<uint>               g_ASInvocsPerSlot : register(t4);
StructuredBuffer<LeafInstanceData>   g_Leaves          : register(t5);
StructuredBuffer<LeafSlotData>       g_LeafSlots       : register(t6);
StructuredBuffer<LeafMeshletData>    g_LeafMeshlets    : register(t7);
Texture2DArray                       t_ShadowMap       : register(t8);
Texture2D<float4>                    t_HiZ             : register(t9);
SamplerComparisonState               s_ShadowSampler   : register(s0);

float FarthestHiZDepth(float a, float b)
{
#if XYLEM_USE_REVERSE_Z
    return min(a, b);
#else
    return max(a, b);
#endif
}

float LoadHiZFarthestForRect(float2 minUV, float2 maxUV, float mipLevel)
{
    uint mip = (uint)mipLevel;
    uint mipWidth, mipHeight, mipCount;
    t_HiZ.GetDimensions(mip, mipWidth, mipHeight, mipCount);

    uint2 lastTexel = uint2(mipWidth - 1u, mipHeight - 1u);
    float2 mipDims = float2((float)mipWidth, (float)mipHeight);

    uint2 lo = min((uint2)floor(minUV * mipDims), lastTexel);
    uint2 hi = min((uint2)floor(maxUV * mipDims), lastTexel);

    float hiz = t_HiZ.Load(int3(lo.x, lo.y, mip)).r;

    if (hi.x != lo.x)
        hiz = FarthestHiZDepth(hiz, t_HiZ.Load(int3(hi.x, lo.y, mip)).r);
    if (hi.y != lo.y)
        hiz = FarthestHiZDepth(hiz, t_HiZ.Load(int3(lo.x, hi.y, mip)).r);
    if ((hi.x != lo.x) && (hi.y != lo.y))
        hiz = FarthestHiZDepth(hiz, t_HiZ.Load(int3(hi.x, hi.y, mip)).r);

    return hiz;
}

// Single-uint global counter — atomically accumulates the per-frame total of
// leaf cross-billboards that survive AS cull. Each binding set (main / depth /
// shadow) points this slot at its own counter so a single AS shader can
// service all three pipelines without double-counting.
RWByteAddressBuffer                  g_LeafSurvivors   : register(u0);

// =============================================================================
// AS helpers — leaf-meshlet frustum + Hi-Z cull
// =============================================================================

// Bounds (center+radius) live in asset-local space inside LeafMeshletData. Both
// helpers transform that sphere by the per-instance model matrix, assuming
// rotation + uniform-ish scale (project convention for tree placements).
float LeafMeshletWorldRadius(float rLS, float4x4 model)
{
    float3 sx = float3(model._m00, model._m01, model._m02);
    float3 sy = float3(model._m10, model._m11, model._m12);
    float3 sz = float3(model._m20, model._m21, model._m22);
    float  scale = max(length(sx), max(length(sy), length(sz)));
    return rLS * scale;
}

bool LeafMeshletFrustumCulled(LeafMeshletData m, float4x4 model)
{
    float3 cWS = mul(float4(m.bounds.xyz, 1.0), model).xyz;
    float  rWS = LeafMeshletWorldRadius(m.bounds.w, model);

    // plane convention: dot(n, p) > d means in front of the plane (outside)
    [unroll]
    for (int i = 0; i < 6; ++i)
    {
        float3 n = viewFrustum[i].n;
        float  d = viewFrustum[i].d;
        if (dot(n, cWS) - d > rWS) return true;   // sphere fully outside this plane
    }
    return false;
}

bool LeafMeshletHiZOccluded(LeafMeshletData m, float4x4 model)
{
    if (!g_HizEnabled) return false;

    float3 cWS = mul(float4(m.bounds.xyz, 1.0), model).xyz;
    float  rWS = LeafMeshletWorldRadius(m.bounds.w, model);
    float3 bmin = cWS - rWS;
    float3 bmax = cWS + rWS;

    float3 corners[8] = {
        float3(bmin.x, bmin.y, bmin.z), float3(bmax.x, bmin.y, bmin.z),
        float3(bmin.x, bmax.y, bmin.z), float3(bmax.x, bmax.y, bmin.z),
        float3(bmin.x, bmin.y, bmax.z), float3(bmax.x, bmin.y, bmax.z),
        float3(bmin.x, bmax.y, bmax.z), float3(bmax.x, bmax.y, bmax.z),
    };

    float2 minUV = float2(1, 1);
    float2 maxUV = float2(0, 0);
#if XYLEM_USE_REVERSE_Z
    float closestDepth = 0.0;
#else
    float closestDepth = 1.0;
#endif

    [unroll]
    for (int i = 0; i < 8; i++)
    {
        float4 clip = mul(float4(corners[i], 1.0), viewProj);
        if (clip.w <= 0.0) return false;       // straddles near plane: keep
        float3 ndc = clip.xyz / clip.w;
        float2 uv = ndc.xy * 0.5 + 0.5;
        uv.y = 1.0 - uv.y;
        minUV = min(minUV, uv);
        maxUV = max(maxUV, uv);
#if XYLEM_USE_REVERSE_Z
        closestDepth = max(closestDepth, ndc.z);
#else
        closestDepth = min(closestDepth, ndc.z);
#endif
    }

    minUV = saturate(minUV);
    maxUV = saturate(maxUV);

    float2 footprint = (maxUV - minUV) * g_HizDimensions;
    float mipLevel = ceil(log2(max(footprint.x, footprint.y)));
    mipLevel = clamp(mipLevel, 0, g_MaxHiZMip);

    float hizDepth = LoadHiZFarthestForRect(minUV, maxUV, mipLevel);

#if XYLEM_USE_REVERSE_Z
    return (closestDepth < hizDepth);
#else
    return (closestDepth > hizDepth);
#endif
}

struct LeafPayload
{
    uint instanceIdx;
    uint slotIdx;
    uint meshletIndices[XYLEM_AS_GROUP_SIZE];
};

groupshared LeafPayload s_payload;
groupshared uint s_survivors;
groupshared uint s_survivingLeafCount;  // Σ meshlet.meta.y over surviving meshlets in this AS group.

// Main + depth-prepass leaf AS. Eye-camera frustum + Hi-Z reject per leaf
// meshlet using the asset-local bounding sphere transformed by the visible
// instance's model matrix. Off-screen and occluded leaves stop the
// MS dispatch entirely for that meshlet.
[numthreads(XYLEM_AS_GROUP_SIZE, 1, 1)]
void leaf_as(uint3 gid : SV_GroupID, uint gtid : SV_GroupThreadID)
{
    if (gtid == 0)
    {
        s_survivors          = 0;
        s_survivingLeafCount = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint slotIdx = g_SlotIdx;
    uint invocsPerInst = max(1u, g_ASInvocsPerSlot[slotIdx]);
    uint visibleCount = g_SlotCounts[slotIdx];
    LeafSlotData leafSlot = g_LeafSlots[slotIdx];

    uint flatGroup = gid.x + gid.y * XYLEM_DISPATCH_X;
    uint instanceInSlot = flatGroup / invocsPerInst;
    uint invocIdx = flatGroup % invocsPerInst;

    if (instanceInSlot < visibleCount)
    {
        uint persistentInstIdx = g_VisBuf[g_SlotOffsets[slotIdx] + instanceInSlot];
        InstanceRenderData inst = g_Instances[persistentInstIdx];
        if (gtid == 0)
        {
            s_payload.instanceIdx = persistentInstIdx;
            s_payload.slotIdx = slotIdx;
        }

        uint meshletLocalIdx = invocIdx * XYLEM_AS_GROUP_SIZE + gtid;
        if (meshletLocalIdx < leafSlot.meshletCount)
        {
            LeafMeshletData m = g_LeafMeshlets[leafSlot.meshletOffset + meshletLocalIdx];
            bool cull =
                LeafMeshletFrustumCulled(m, inst.model) ||
                LeafMeshletHiZOccluded(m, inst.model);
            if (!cull)
            {
                uint survivor;
                InterlockedAdd(s_survivors, 1, survivor);
                s_payload.meshletIndices[survivor] = meshletLocalIdx;
                InterlockedAdd(s_survivingLeafCount, m.meta.y);
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // One global atomic per AS group instead of one per surviving meshlet.
    if (gtid == 0 && s_survivingLeafCount > 0)
    {
        uint dummy;
        g_LeafSurvivors.InterlockedAdd(0, s_survivingLeafCount, dummy);
    }

    DispatchMesh(s_survivors, 1, 1, s_payload);
}

// Shadow leaf AS. The eye-built Hi-Z and eye view frustum don't apply to
// light-camera rendering, so this variant skips meshlet-level cull and
// dispatches every cluster of every visible-shadow-caster instance. The
// CullCS already rejected instances outside the cascade caster bbox at
// the per-instance level.
[numthreads(XYLEM_AS_GROUP_SIZE, 1, 1)]
void leaf_shadow_as(uint3 gid : SV_GroupID, uint gtid : SV_GroupThreadID)
{
    if (gtid == 0)
    {
        s_survivors          = 0;
        s_survivingLeafCount = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint slotIdx = g_SlotIdx;
    uint invocsPerInst = max(1u, g_ASInvocsPerSlot[slotIdx]);
    uint visibleCount = g_SlotCounts[slotIdx];
    LeafSlotData leafSlot = g_LeafSlots[slotIdx];

    uint flatGroup = gid.x + gid.y * XYLEM_DISPATCH_X;
    uint instanceInSlot = flatGroup / invocsPerInst;
    uint invocIdx = flatGroup % invocsPerInst;

    if (instanceInSlot < visibleCount)
    {
        uint persistentInstIdx = g_VisBuf[g_SlotOffsets[slotIdx] + instanceInSlot];
        if (gtid == 0)
        {
            s_payload.instanceIdx = persistentInstIdx;
            s_payload.slotIdx = slotIdx;
        }

        uint meshletLocalIdx = invocIdx * XYLEM_AS_GROUP_SIZE + gtid;
        if (meshletLocalIdx < leafSlot.meshletCount)
        {
            LeafMeshletData m = g_LeafMeshlets[leafSlot.meshletOffset + meshletLocalIdx];
            uint survivor;
            InterlockedAdd(s_survivors, 1, survivor);
            s_payload.meshletIndices[survivor] = meshletLocalIdx;
            InterlockedAdd(s_survivingLeafCount, m.meta.y);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (gtid == 0 && s_survivingLeafCount > 0)
    {
        uint dummy;
        g_LeafSurvivors.InterlockedAdd(0, s_survivingLeafCount, dummy);
    }

    DispatchMesh(s_survivors, 1, 1, s_payload);
}

struct LeafV2P
{
    float4 pos      : SV_Position;
    float3 worldPos : WORLD_POS;
    float  viewZ    : VIEW_Z;
    float3 normal   : NORMAL;
    float3 color    : COLOR0;
};

LeafV2P BuildMeshLeafVertex(uint gtid, uint survivorIdx, LeafPayload payload)
{
    LeafSlotData slot = g_LeafSlots[payload.slotIdx];
    LeafMeshletData meshlet = g_LeafMeshlets[slot.meshletOffset + payload.meshletIndices[survivorIdx]];

    uint leafInMeshlet = gtid / 8u;
    uint corner = gtid % 8u;
    LeafInstanceData leaf = g_Leaves[meshlet.meta.x + leafInMeshlet];
    InstanceRenderData inst = g_Instances[payload.instanceIdx];

    float3 localPos, localNormal;
    BuildLeafCorner(leaf, corner, localPos, localNormal);

    float4 worldPos = mul(float4(localPos, 1.0), inst.model);

    LeafV2P v;
    v.pos = mul(worldPos, viewProj);
    v.worldPos = worldPos.xyz;
    v.viewZ = mul(worldPos, viewMatrix).z;
    v.normal = normalize(mul(localNormal, inst.normal));
    v.color = leaf.color.rgb;
    return v;
}

uint3 LeafTri(uint tri)
{
    uint leaf = tri / 4u;
    uint triInLeaf = tri % 4u;
    uint base = leaf * 8u;

    if (triInLeaf == 0) return uint3(base + 0, base + 1, base + 2);
    if (triInLeaf == 1) return uint3(base + 0, base + 2, base + 3);
    if (triInLeaf == 2) return uint3(base + 4, base + 5, base + 6);
    return uint3(base + 4, base + 6, base + 7);
}

[numthreads(XYLEM_MS_GROUP_SIZE, 1, 1)]
[outputtopology("triangle")]
void leaf_ms(
    uint gtid : SV_GroupThreadID,
    uint3 gid : SV_GroupID,
    in payload LeafPayload payload,
    out indices uint3 o_tris[XYLEM_MAX_MESHLET_PRIMS],
    out vertices LeafV2P o_verts[XYLEM_MAX_MESHLET_VERTS])
{
    LeafSlotData slot = g_LeafSlots[payload.slotIdx];
    LeafMeshletData meshlet = g_LeafMeshlets[slot.meshletOffset + payload.meshletIndices[gid.x]];
    uint leafCount = meshlet.meta.y;
    SetMeshOutputCounts(leafCount * 8u, leafCount * 4u);

    if (gtid < leafCount * 8u)
        o_verts[gtid] = BuildMeshLeafVertex(gtid, gid.x, payload);
    if (gtid < leafCount * 4u)
        o_tris[gtid] = LeafTri(gtid);
}

struct DepthV2P
{
    float4 pos : SV_Position;
};

[numthreads(XYLEM_MS_GROUP_SIZE, 1, 1)]
[outputtopology("triangle")]
void leaf_depth_ms(
    uint gtid : SV_GroupThreadID,
    uint3 gid : SV_GroupID,
    in payload LeafPayload payload,
    out indices uint3 o_tris[XYLEM_MAX_MESHLET_PRIMS],
    out vertices DepthV2P o_verts[XYLEM_MAX_MESHLET_VERTS])
{
    LeafSlotData slot = g_LeafSlots[payload.slotIdx];
    LeafMeshletData meshlet = g_LeafMeshlets[slot.meshletOffset + payload.meshletIndices[gid.x]];
    uint leafCount = meshlet.meta.y;
    SetMeshOutputCounts(leafCount * 8u, leafCount * 4u);

    if (gtid < leafCount * 8u)
    {
        LeafV2P v = BuildMeshLeafVertex(gtid, gid.x, payload);
        o_verts[gtid].pos = v.pos;
    }
    if (gtid < leafCount * 4u)
        o_tris[gtid] = LeafTri(gtid);
}

[numthreads(XYLEM_MS_GROUP_SIZE, 1, 1)]
[outputtopology("triangle")]
void leaf_shadow_ms(
    uint gtid : SV_GroupThreadID,
    uint3 gid : SV_GroupID,
    in payload LeafPayload payload,
    out indices uint3 o_tris[XYLEM_MAX_MESHLET_PRIMS],
    out vertices DepthV2P o_verts[XYLEM_MAX_MESHLET_VERTS])
{
    LeafSlotData slot = g_LeafSlots[payload.slotIdx];
    LeafMeshletData meshlet = g_LeafMeshlets[slot.meshletOffset + payload.meshletIndices[gid.x]];
    uint leafCount = meshlet.meta.y;
    uint cascade = payload.slotIdx % XYLEM_NUM_CASCADES;
    SetMeshOutputCounts(leafCount * 8u, leafCount * 4u);

    if (gtid < leafCount * 8u)
    {
        uint leafInMeshlet = gtid / 8u;
        uint corner = gtid % 8u;
        LeafInstanceData leaf = g_Leaves[meshlet.meta.x + leafInMeshlet];
        InstanceRenderData inst = g_Instances[payload.instanceIdx];

        float3 localPos, localNormal;
        BuildLeafCorner(leaf, corner, localPos, localNormal);
        float4 worldPos = mul(float4(localPos, 1.0), inst.model);
        o_verts[gtid].pos = mul(worldPos, lightViewProj[cascade]);
    }
    if (gtid < leafCount * 4u)
        o_tris[gtid] = LeafTri(gtid);
}

float SampleShadowCascade(float3 worldPos, uint selectedCascade)
{
    float4 posLS = mul(float4(worldPos, 1.0), lightViewProj[selectedCascade]);
    float2 shadowUV = posLS.xy * float2(0.5, -0.5) + 0.5;

    uint width, height, elements;
    t_ShadowMap.GetDimensions(width, height, elements);
    float2 texelSize = 1.0 / float2(width, height);

    float shadow = 0.0;
    [unroll]
    for (int x = -1; x <= 1; ++x) {
        [unroll]
        for (int y = -1; y <= 1; ++y) {
            shadow += t_ShadowMap.SampleCmpLevelZero(
                s_ShadowSampler,
                float3(shadowUV + float2(x, y) * texelSize, float(selectedCascade)),
                posLS.z);
        }
    }
    return shadow / 9.0;
}

void leaf_ps(in LeafV2P i, out float4 o_color : SV_Target0)
{
    uint selectedCascade = SelectShadowCascade(i.viewZ, cascadeSplits);

    float3 lightDir = -normalize(sunLightDir);
    float notInShadow = SampleShadowCascade(i.worldPos, selectedCascade);
    // Single-sided diffuse to match the trunk/impostor shading model.
    float diffuse = max(dot(normalize(i.normal), lightDir), 0.0);
    float ambient = 0.20;
    float lighting = ambient + (1.0 - ambient) * diffuse * notInShadow;
    o_color = float4(lighting * i.color, 1.0);
}
