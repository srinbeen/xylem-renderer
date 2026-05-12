#include "../include/macros.h"
#include "ShaderRegisterMap.hlsli"
#include "ShadowCascadeCommon.hlsli"
#include "terrain_shading.hlsli"
#include "meshlet_types.hlsli"

#pragma pack_matrix(row_major)

// FrameCB layout mirrors Render::CullConstantBufferEntry up through viewFrustum.
// The full backing buffer is sized to kCullFrameSize so reading viewFrustum is
// safe even though the existing terrain VS path doesn't declare it.
cbuffer FrameCB : register(XY_REG_B_MESH_TERRAIN_CB_FRAME)
{
    float4x4 viewProj;
    float4x4 viewMatrix;
    float4x4 lightViewProj[XYLEM_NUM_CASCADES];
    float3   sunLightDir;
    float    _pad0;
    float4   cascadeSplits;
    float4   viewFrustum[6];
};

cbuffer ShadingCB : register(XY_REG_B_MESH_TERRAIN_CB_SHADING)
{
    float g_TileSize;
    float g_ForestToDirtY;
    float g_DirtToSnowY;
    float g_BandWidth;
    float g_SlopeLo;
    float g_SlopeHi;
    float g_MacroNoiseAmp;
    float g_Pad0;
};

cbuffer ASCullCB : register(XY_REG_B_MESH_TERRAIN_CB_ASCULL)
{
    float3 g_CameraPos;
    uint   g_HizEnabled;
    float2 g_HizDimensions;
    float  g_MaxHiZMip;
    uint   g_ASConeCullEnabled;  // unused for terrain (no per-meshlet cones)
};

struct TerrainVertex
{
    float3 pos;
    float3 normal;
    float2 uv;
};

// Three SoA raw buffers — one per attribute stream. Depth-only paths bind only
// g_Positions; color/shadow paths bind all three.
ByteAddressBuffer g_Positions : register(XY_REG_T_MESH_TERRAIN_SRV_POSITIONS);
ByteAddressBuffer g_Normals   : register(XY_REG_T_MESH_TERRAIN_SRV_NORMALS);
ByteAddressBuffer g_UVs       : register(XY_REG_T_MESH_TERRAIN_SRV_UVS);

float3 LoadTerrainPos(uint globalIdx)    { return asfloat(g_Positions.Load3(globalIdx * 12u)); }
float3 LoadTerrainNormal(uint globalIdx) { return asfloat(g_Normals.Load3(globalIdx * 12u)); }
float2 LoadTerrainUV(uint globalIdx)     { return asfloat(g_UVs.Load2(globalIdx * 8u)); }

TerrainVertex LoadTerrainVertex(uint globalIdx)
{
    TerrainVertex v;
    v.pos    = LoadTerrainPos(globalIdx);
    v.normal = LoadTerrainNormal(globalIdx);
    v.uv     = LoadTerrainUV(globalIdx);
    return v;
}

struct TerrainMeshletDesc
{
    float3 aabbMin;
    uint   vertOffset;
    float3 aabbMax;
    uint   triOffset;
    uint   vertCount;
    uint   triCount;
    uint   _pad0;
    uint   _pad1;
};

StructuredBuffer<TerrainMeshletDesc> g_Meshlets       : register(XY_REG_T_MESH_TERRAIN_SRV_MESHLET_DESCS);
StructuredBuffer<uint>               g_MeshletVertIdx : register(XY_REG_T_MESH_TERRAIN_SRV_MESHLET_VERT_IDX);
ByteAddressBuffer                    g_MeshletPrimIdx : register(XY_REG_T_MESH_TERRAIN_SRV_MESHLET_PRIM_IDX);
Texture2D<float4>                    t_HiZ            : register(XY_REG_T_MESH_TERRAIN_TEX_HI_Z);
Texture2DArray                       t_ShadowMap      : register(XY_REG_T_MESH_TERRAIN_TEX_SHADOW_MAP);

Texture2D    t_ForestDiff : register(XY_REG_T_MESH_TERRAIN_TEX_FOREST_DIFF);
Texture2D    t_ForestNor  : register(XY_REG_T_MESH_TERRAIN_TEX_FOREST_NOR);
Texture2D    t_DirtDiff   : register(XY_REG_T_MESH_TERRAIN_TEX_DIRT_DIFF);
Texture2D    t_DirtNor    : register(XY_REG_T_MESH_TERRAIN_TEX_DIRT_NOR);
Texture2D    t_RockDiff   : register(XY_REG_T_MESH_TERRAIN_TEX_ROCK_DIFF);
Texture2D    t_RockNor    : register(XY_REG_T_MESH_TERRAIN_TEX_ROCK_NOR);
Texture2D    t_SnowDiff   : register(XY_REG_T_MESH_TERRAIN_TEX_SNOW_DIFF);
Texture2D    t_SnowNor    : register(XY_REG_T_MESH_TERRAIN_TEX_SNOW_NOR);

RWByteAddressBuffer                  u_VisibleCounter : register(XY_REG_U_MESH_TERRAIN_UAV_VISIBLE_COUNTER);

SamplerComparisonState s_ShadowSampler : register(XY_REG_S_MESH_TERRAIN_SAMPLER_SHADOW);
SamplerState           s_Aniso         : register(XY_REG_S_MESH_TERRAIN_SAMPLER_ANISO);

// =============================================================================
// AABB-vs-frustum. Mirrors CullCS.hlsl::DoesAABBIntersectFrustum.
// dm::frustum convention: 6 planes, normals pointing OUT of the volume, plane
// stored as float4(n.xyz, d) with "inside" iff dot(n, x) <= d. To reject, pick
// the min-dot corner (= per-axis bmin if n>0 else bmax) and test whether even
// that corner is on the outside.
// =============================================================================
bool AABBOutsideFrustum(float3 bmin, float3 bmax)
{
    [unroll]
    for (int i = 0; i < 6; i++)
    {
        float4 p = viewFrustum[i];
        float3 v = float3(
            p.x > 0 ? bmin.x : bmax.x,
            p.y > 0 ? bmin.y : bmax.y,
            p.z > 0 ? bmin.z : bmax.z);
        if (dot(p.xyz, v) > p.w)
            return true;
    }
    return false;
}

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
    float2 mipDims  = float2((float)mipWidth, (float)mipHeight);

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

bool AABBHiZOccluded(float3 bmin, float3 bmax)
{
    if (!g_HizEnabled) return false;

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
        float4 clip = mul(float4(corners[i], 1), viewProj);
        if (clip.w <= 0.0) return false;
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

// =============================================================================
// AS — frustum + Hi-Z cull per terrain meshlet
// =============================================================================
struct TerrainASPayload
{
    uint meshletIndices[XYLEM_AS_GROUP_SIZE];
};

groupshared TerrainASPayload s_payload;
groupshared uint             s_survivors;

[numthreads(XYLEM_AS_GROUP_SIZE, 1, 1)]
void terrain_as(uint3 gid : SV_GroupID, uint gtid : SV_GroupThreadID)
{
    if (gtid == 0) s_survivors = 0;
    GroupMemoryBarrierWithGroupSync();

    uint totalMeshlets, meshletStride;
    g_Meshlets.GetDimensions(totalMeshlets, meshletStride);

    uint meshletIdx = gid.x * XYLEM_AS_GROUP_SIZE + gtid;
    if (meshletIdx < totalMeshlets)
    {
        TerrainMeshletDesc m = g_Meshlets[meshletIdx];
        bool cull = AABBOutsideFrustum(m.aabbMin, m.aabbMax)
                 || AABBHiZOccluded(m.aabbMin, m.aabbMax);
        if (!cull)
        {
            uint slot;
            InterlockedAdd(s_survivors, 1, slot);
            s_payload.meshletIndices[slot] = meshletIdx;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (gtid == 0 && s_survivors > 0)
        u_VisibleCounter.InterlockedAdd(0, s_survivors);

    DispatchMesh(s_survivors, 1, 1, s_payload);
}

// =============================================================================
// MS — emit one terrain tile (vertices + triangles) per group
// =============================================================================
struct V2P
{
    float4 pos      : SV_Position;
    float3 worldPos : WORLD_POS;
    float  viewZ    : VIEW_Z;
    float3 normal   : NORMAL;
    float2 uv       : UV;
    float  height   : HEIGHT;
};

uint3 LoadMeshletTriangle(uint triByteOffset)
{
    uint alignedOffset = triByteOffset & ~3u;
    uint byteShift     = (triByteOffset - alignedOffset) * 8u;

    uint w0 = g_MeshletPrimIdx.Load(alignedOffset);
    uint tri24 = (w0 >> byteShift) & 0x00FFFFFFu;

    if (byteShift != 0)
    {
        uint w1 = g_MeshletPrimIdx.Load(alignedOffset + 4u);
        tri24 = ((w0 >> byteShift) | (w1 << (32u - byteShift))) & 0x00FFFFFFu;
    }

    return uint3(tri24 & 0xFFu,
                 (tri24 >> 8) & 0xFFu,
                 (tri24 >> 16) & 0xFFu);
}

[numthreads(XYLEM_MS_GROUP_SIZE, 1, 1)]
[outputtopology("triangle")]
void terrain_ms(
    uint   gtid : SV_GroupThreadID,
    uint3  gid  : SV_GroupID,
    in payload TerrainASPayload i_payload,
    out indices  uint3 o_tris[XYLEM_MAX_MESHLET_PRIMS],
    out vertices V2P   o_verts[XYLEM_MAX_MESHLET_VERTS])
{
    uint meshletIdx = i_payload.meshletIndices[gid.x];
    TerrainMeshletDesc m = g_Meshlets[meshletIdx];

    SetMeshOutputCounts(m.vertCount, m.triCount);

    if (gtid < m.vertCount)
    {
        uint globalVertIdx = g_MeshletVertIdx[m.vertOffset + gtid];
        TerrainVertex v = LoadTerrainVertex(globalVertIdx);

        V2P o;
        o.pos      = mul(float4(v.pos, 1), viewProj);
        o.worldPos = v.pos;
        o.viewZ    = mul(float4(v.pos, 1), viewMatrix).z;
        o.normal   = v.normal;
        o.uv       = v.uv;
        o.height   = v.pos.y;
        o_verts[gtid] = o;
    }

    if (gtid < m.triCount)
    {
        uint triByte = m.triOffset + gtid * 3u;
        o_tris[gtid] = LoadMeshletTriangle(triByte);
    }
}

// =============================================================================
// PS — identical shading to terrain_compute.hlsl::terrain_ps
// =============================================================================
float SampleShadowCascade(float3 worldPos, uint cascadeIdx)
{
    float4 posLS    = mul(float4(worldPos, 1), lightViewProj[cascadeIdx]);
    float2 shadowUV = posLS.xy * float2(0.5, -0.5) + 0.5;
    return t_ShadowMap.SampleCmpLevelZero(s_ShadowSampler,
        float3(shadowUV, float(cascadeIdx)), posLS.z);
}

void terrain_ps(
    in float4  i_pos      : SV_Position,
    in float3  i_worldPos : WORLD_POS,
    in float   i_viewZ    : VIEW_Z,
    in float3  i_normal   : NORMAL,
    in float2  i_uv       : UV,
    in float   i_height   : HEIGHT,

    out float4 o_color    : SV_Target0)
{
    float3 vertN = normalize(i_normal);

    TerrainShadingParams p;
    p.tileSize       = g_TileSize;
    p.forestToDirtY  = g_ForestToDirtY;
    p.dirtToSnowY    = g_DirtToSnowY;
    p.bandWidth      = g_BandWidth;
    p.slopeLo        = g_SlopeLo;
    p.slopeHi        = g_SlopeHi;
    p.macroNoiseAmp  = g_MacroNoiseAmp;

    float4 weights = ComputeLayerWeights(i_worldPos, vertN, i_height, p);
    float3 triW    = ComputeTriplanarWeights(vertN);
    float3 worldPosScaled = i_worldPos / max(g_TileSize, 1e-3);

    float3 albedoSum = 0.0;
    float3 normalSum = 0.0;
    AccumulateLayer(t_ForestDiff, t_ForestNor, s_Aniso, worldPosScaled, triW, vertN, weights.x, albedoSum, normalSum);
    albedoSum = saturate(albedoSum * GREEN_BOOST);
    AccumulateLayer(t_DirtDiff,   t_DirtNor,   s_Aniso, worldPosScaled, triW, vertN, weights.y, albedoSum, normalSum);
    AccumulateLayer(t_RockDiff,   t_RockNor,   s_Aniso, worldPosScaled, triW, vertN, weights.z, albedoSum, normalSum);
    AccumulateLayer(t_SnowDiff,   t_SnowNor,   s_Aniso, worldPosScaled, triW, vertN, weights.w, albedoSum, normalSum);

    float3 worldN = normalize(normalSum + 1e-5 * vertN);

    float3 lightDir    = -normalize(sunLightDir);
    float  diffuse     = max(dot(worldN, lightDir), 0);
    uint   cascadeIdx  = SelectShadowCascade(i_viewZ, cascadeSplits);
    float  notInShadow = SampleShadowCascade(i_worldPos, cascadeIdx);
    float  lighting    = 0.15 + 0.85 * diffuse * notInShadow;

    o_color = float4(lighting * albedoSum, 1);
}

// =============================================================================
// Shadow path: AS culls per-meshlet against the cascade's light frustum, MS
// emits depth-only positions via lightViewProj[cascade]. Cascade index travels
// through a root-constant CB at b3.
// =============================================================================

cbuffer ShadowPushCB : register(XY_REG_B_MESH_TERRAIN_PUSH_C_CASCADE)
{
    uint g_CascadeIdx;
};

// Gribb-Hartmann frustum extraction from a row-major lightViewProj. Planes use
// the same dm::frustum convention as AABBOutsideFrustum: float4(n.xyz, d) with
// "inside" iff dot(n, x) <= d.
void ExtractLightFrustumPlanes(float4x4 m, out float4 planes[6])
{
    // m is row-major and we mul(point, m) -> clip = point * m. Rows of m are
    // therefore (cols of column-major M^T). We want the 6 planes of the clip
    // volume {-w <= clip.x,y,z <= +w} pulled back to world space.
    // For row-vector convention with row-major storage:
    //   clip.x = dot(point, m[0..3].x col)  i.e. uses column 0 of m -> (m[0][0], m[1][0], m[2][0], m[3][0])
    // Easier: derive from rows of M^T, equivalent to columns of m. Build by hand.
    float4 c0 = float4(m[0][0], m[1][0], m[2][0], m[3][0]);
    float4 c1 = float4(m[0][1], m[1][1], m[2][1], m[3][1]);
    float4 c2 = float4(m[0][2], m[1][2], m[2][2], m[3][2]);
    float4 c3 = float4(m[0][3], m[1][3], m[2][3], m[3][3]);

    // dm::frustum stores plane as (n, d) with inside = dot(n,x) <= d. The
    // standard extraction gives planes with inside = dot(n,x) + d >= 0, i.e.
    // float4(-n, d) in dm's convention. Convert by negating xyz and keeping w.
    float4 left   = c3 + c0;  // inside: dot(left.xyz,x) + left.w >= 0
    float4 right  = c3 - c0;
    float4 bottom = c3 + c1;
    float4 top    = c3 - c1;
    float4 znear  = c2;       // standard DX clip range [0,1]: inside iff clip.z >= 0
    float4 zfar   = c3 - c2;

    planes[0] = float4(-left.xyz,   left.w);
    planes[1] = float4(-right.xyz,  right.w);
    planes[2] = float4(-bottom.xyz, bottom.w);
    planes[3] = float4(-top.xyz,    top.w);
    planes[4] = float4(-znear.xyz,  znear.w);
    planes[5] = float4(-zfar.xyz,   zfar.w);
}

bool AABBOutsideLightFrustum(float3 bmin, float3 bmax, float4 planes[6])
{
    [unroll]
    for (int i = 0; i < 6; i++)
    {
        float4 p = planes[i];
        float3 v = float3(
            p.x > 0 ? bmin.x : bmax.x,
            p.y > 0 ? bmin.y : bmax.y,
            p.z > 0 ? bmin.z : bmax.z);
        if (dot(p.xyz, v) > p.w)
            return true;
    }
    return false;
}

groupshared TerrainASPayload s_shadowPayload;
groupshared uint             s_shadowSurvivors;

[numthreads(XYLEM_AS_GROUP_SIZE, 1, 1)]
void shadow_terrain_as(uint3 gid : SV_GroupID, uint gtid : SV_GroupThreadID)
{
    if (gtid == 0) s_shadowSurvivors = 0;
    GroupMemoryBarrierWithGroupSync();

    float4 planes[6];
    ExtractLightFrustumPlanes(lightViewProj[g_CascadeIdx], planes);

    uint totalMeshlets, meshletStride;
    g_Meshlets.GetDimensions(totalMeshlets, meshletStride);

    uint meshletIdx = gid.x * XYLEM_AS_GROUP_SIZE + gtid;
    if (meshletIdx < totalMeshlets)
    {
        TerrainMeshletDesc m = g_Meshlets[meshletIdx];
        if (!AABBOutsideLightFrustum(m.aabbMin, m.aabbMax, planes))
        {
            uint slot;
            InterlockedAdd(s_shadowSurvivors, 1, slot);
            s_shadowPayload.meshletIndices[slot] = meshletIdx;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    DispatchMesh(s_shadowSurvivors, 1, 1, s_shadowPayload);
}

struct ShadowV2P
{
    float4 pos : SV_Position;
};

[numthreads(XYLEM_MS_GROUP_SIZE, 1, 1)]
[outputtopology("triangle")]
void shadow_terrain_ms(
    uint   gtid : SV_GroupThreadID,
    uint3  gid  : SV_GroupID,
    in payload TerrainASPayload i_payload,
    out indices  uint3     o_tris[XYLEM_MAX_MESHLET_PRIMS],
    out vertices ShadowV2P o_verts[XYLEM_MAX_MESHLET_VERTS])
{
    uint meshletIdx = i_payload.meshletIndices[gid.x];
    TerrainMeshletDesc m = g_Meshlets[meshletIdx];

    SetMeshOutputCounts(m.vertCount, m.triCount);

    if (gtid < m.vertCount)
    {
        uint globalVertIdx = g_MeshletVertIdx[m.vertOffset + gtid];
        float3 pos = LoadTerrainPos(globalVertIdx);

        ShadowV2P o;
        o.pos = mul(float4(pos, 1), lightViewProj[g_CascadeIdx]);
        o_verts[gtid] = o;
    }

    if (gtid < m.triCount)
    {
        uint triByte = m.triOffset + gtid * 3u;
        o_tris[gtid] = LoadMeshletTriangle(triByte);
    }
}

// =============================================================================
// Depth prepass path: eye-view, position-only output, no survivor counter.
// AS mirrors terrain_as (frustum + Hi-Z) but drops the InterlockedAdd into
// u_VisibleCounter so we don't have to bind that UAV in the prepass layout and
// don't double-count the per-frame survivor stat (color pass owns that count).
// =============================================================================

groupshared TerrainASPayload s_depthPayload;
groupshared uint             s_depthSurvivors;

[numthreads(XYLEM_AS_GROUP_SIZE, 1, 1)]
void depth_terrain_as(uint3 gid : SV_GroupID, uint gtid : SV_GroupThreadID)
{
    if (gtid == 0) s_depthSurvivors = 0;
    GroupMemoryBarrierWithGroupSync();

    uint totalMeshlets, meshletStride;
    g_Meshlets.GetDimensions(totalMeshlets, meshletStride);

    uint meshletIdx = gid.x * XYLEM_AS_GROUP_SIZE + gtid;
    if (meshletIdx < totalMeshlets)
    {
        TerrainMeshletDesc m = g_Meshlets[meshletIdx];
        bool cull = AABBOutsideFrustum(m.aabbMin, m.aabbMax)
                 || AABBHiZOccluded(m.aabbMin, m.aabbMax);
        if (!cull)
        {
            uint slot;
            InterlockedAdd(s_depthSurvivors, 1, slot);
            s_depthPayload.meshletIndices[slot] = meshletIdx;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    DispatchMesh(s_depthSurvivors, 1, 1, s_depthPayload);
}

[numthreads(XYLEM_MS_GROUP_SIZE, 1, 1)]
[outputtopology("triangle")]
void depth_terrain_ms(
    uint   gtid : SV_GroupThreadID,
    uint3  gid  : SV_GroupID,
    in payload TerrainASPayload i_payload,
    out indices  uint3     o_tris[XYLEM_MAX_MESHLET_PRIMS],
    out vertices ShadowV2P o_verts[XYLEM_MAX_MESHLET_VERTS])
{
    uint meshletIdx = i_payload.meshletIndices[gid.x];
    TerrainMeshletDesc m = g_Meshlets[meshletIdx];

    SetMeshOutputCounts(m.vertCount, m.triCount);

    if (gtid < m.vertCount)
    {
        uint globalVertIdx = g_MeshletVertIdx[m.vertOffset + gtid];
        float3 pos = LoadTerrainPos(globalVertIdx);

        ShadowV2P o;
        o.pos = mul(float4(pos, 1), viewProj);
        o_verts[gtid] = o;
    }

    if (gtid < m.triCount)
    {
        uint triByte = m.triOffset + gtid * 3u;
        o_tris[gtid] = LoadMeshletTriangle(triByte);
    }
}
