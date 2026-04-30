#pragma pack_matrix(row_major)

#include "../include/macros.h"
#include "meshlet_types.hlsli"
#include "ShaderRegisterMap.hlsli"

static const uint NUM_CASCADES = 4;

cbuffer CB : register(XY_REG_B_MESH_DRAW_CB_FRAME)
{
    float4x4 viewProj;
    float4x4 viewMatrix;
    float4x4 lightViewProj[NUM_CASCADES];
    float3   sunLightDir;
    float    _pad0;
    float4   cascadeSplits;

    // Padding fields mirror CullConstantBufferEntry so the shadow shader can
    // address LOD-0 ranges from the shared asset*LOD meshlet table.
    float4   _unusedViewFrustum[6];
    float4x4 _unusedWorldToLight;
    float4   _unusedShadowCasterMinLS[NUM_CASCADES];
    float4   _unusedShadowCasterMaxLS[NUM_CASCADES];
    float3   _unusedCameraPos;
    uint     _unusedNumRegions;
    uint     _unusedTotalCapacity;
    uint     g_NumLods;
    float    _unusedPad1a;
    float    _unusedPad1b;
};

// Root constant (PushConstants): slot identity forwarded by the command signature's
// CONSTANT argument. Each ExecuteIndirect record rewrites this before DISPATCH_MESH.
cbuffer PushC : register(XY_REG_B_MESH_DRAW_PUSH_C_SLOT)
{
    uint g_SlotIdx;
};

// Per-frame Hi-Z / AS-cull knobs. Written alongside the main CB.
cbuffer ASCullCB : register(XY_REG_B_MESH_DRAW_CB_ASCULL)
{
    float3   g_CameraPos;
    uint     g_HizEnabled;
    float2   g_HizDimensions;
    float    g_MaxHiZMip;
    uint     g_ASConeCullEnabled;
};

struct InstanceRenderData
{
    float4x4 model;     // offset   0, size 64
    float3x3 normal;    // offset  64, size 36
    uint     treeId;    // offset 100, size 4
};

// Mega-buffers
StructuredBuffer<float3>             g_Positions      : register(XY_REG_T_MESH_DRAW_SRV_POSITIONS);
StructuredBuffer<float3>             g_Normals        : register(XY_REG_T_MESH_DRAW_SRV_NORMALS);
StructuredBuffer<float3>             g_Tangents       : register(XY_REG_T_MESH_DRAW_SRV_TANGENTS);
StructuredBuffer<float3>             g_Bitangents     : register(XY_REG_T_MESH_DRAW_SRV_BITANGENTS);
StructuredBuffer<float2>             g_UVs            : register(XY_REG_T_MESH_DRAW_SRV_UVS);
StructuredBuffer<uint>               g_MeshletVertIdx : register(XY_REG_T_MESH_DRAW_SRV_MESHLET_VERT_IDX);
ByteAddressBuffer                    g_MeshletPrimIdx : register(XY_REG_T_MESH_DRAW_SRV_MESHLET_PRIM_IDX);
StructuredBuffer<MeshletDesc>        g_Meshlets       : register(XY_REG_T_MESH_DRAW_SRV_MESHLETS);
StructuredBuffer<AssetLodRange>      g_AssetLodRanges : register(XY_REG_T_MESH_DRAW_SRV_ASSET_LODS);

// GPU-cull outputs (main-pass indirection; shadow pass uses a parallel binding set
// with the shadow SRVs bound into these same slots).
StructuredBuffer<uint>               g_VisBuf           : register(XY_REG_T_MESH_DRAW_SRV_VIS);
StructuredBuffer<uint>               g_SlotOffsets      : register(XY_REG_T_MESH_DRAW_SRV_SLOT_OFFSETS);
StructuredBuffer<uint>               g_SlotCounts       : register(XY_REG_T_MESH_DRAW_SRV_SLOT_COUNTS);
StructuredBuffer<InstanceRenderData> g_Instances        : register(XY_REG_T_MESH_DRAW_SRV_INSTANCES);
StructuredBuffer<uint>               g_ASInvocsPerSlot  : register(XY_REG_T_MESH_DRAW_SRV_ASINVOCATIONS);

Texture2D                            t_Diffuse        : register(XY_REG_T_MESH_DRAW_TEX_DIFFUSE);
Texture2D                            t_NormalMap      : register(XY_REG_T_MESH_DRAW_TEX_NORMAL_MAP);
Texture2DArray                       t_ShadowMap      : register(XY_REG_T_MESH_DRAW_TEX_SHADOW_MAP);
Texture2D<float2>                    t_HiZ            : register(XY_REG_T_MESH_DRAW_TEX_HI_Z);

SamplerState                         s_Sampler        : register(XY_REG_S_MESH_DRAW_SAMPLER_MAIN);
SamplerComparisonState               s_ShadowSampler  : register(XY_REG_S_MESH_DRAW_SAMPLER_SHADOW);
SamplerState                         s_HizSampler     : register(XY_REG_S_MESH_DRAW_SAMPLER_HI_Z);

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

// =============================================================================
// AS helpers — meshlet cone cull + Hi-Z occlusion
// =============================================================================

bool MeshletConeCull(MeshletDesc m, float4x4 model, float3 cameraWS)
{
    if (!g_ASConeCullEnabled)            return false;
    if (m.coneAxisCutoff.w >= 1.0)       return false; // meshopt disables via cutoff=1

    float3 apexLS = m.coneApex.xyz;
    float3 axisLS = m.coneAxisCutoff.xyz;

    float3 apexWS = mul(float4(apexLS, 1), model).xyz;
    // Assumes rotation + uniform scale (per project convention).
    float3 axisWS = normalize(mul(float4(axisLS, 0), model).xyz);

    float3 toMeshlet = apexWS - cameraWS;
    float  len = length(toMeshlet);
    if (len < 1e-6) return false;
    float cosAngle = dot(axisWS, toMeshlet) / len;
    return cosAngle >= m.coneAxisCutoff.w;
}

bool MeshletHiZOccluded(MeshletDesc m, float4x4 model)
{
    if (!g_HizEnabled) return false;

    // Meshlet bounding sphere -> WS AABB.
    float3 cLS = m.bounds.xyz;
    float  rLS = m.bounds.w;
    float3 cWS = mul(float4(cLS, 1), model).xyz;

    float3 sx = float3(model._m00, model._m01, model._m02);
    float3 sy = float3(model._m10, model._m11, model._m12);
    float3 sz = float3(model._m20, model._m21, model._m22);
    float  scale = max(length(sx), max(length(sy), length(sz)));
    float  rWS   = rLS * scale;

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

    float2 centerUV = (minUV + maxUV) * 0.5;
    float hizDepth = t_HiZ.SampleLevel(s_HizSampler, centerUV, mipLevel).r;

#if XYLEM_USE_REVERSE_Z
    return (closestDepth < hizDepth);
#else
    return (closestDepth > hizDepth);
#endif
}

// =============================================================================
// Amplification Shader — cone + Hi-Z meshlet cull
// =============================================================================

groupshared ASPayload s_payload;
groupshared uint      s_survivors;

[numthreads(XYLEM_AS_GROUP_SIZE, 1, 1)]
void main_as(uint3 gid  : SV_GroupID,
             uint  gtid : SV_GroupThreadID)
{
    if (gtid == 0) s_survivors = 0;
    GroupMemoryBarrierWithGroupSync();

    uint slotIdx            = g_SlotIdx;
    uint ASInvocsPerInst    = max(1u, g_ASInvocsPerSlot[slotIdx]);
    uint visibleCount       = g_SlotCounts[slotIdx];

    uint flatGroup      = gid.x + gid.y * XYLEM_DISPATCH_X;
    uint instanceInSlot = flatGroup / ASInvocsPerInst;
    uint invocIdx       = flatGroup % ASInvocsPerInst;

    if (instanceInSlot < visibleCount)
    {
        uint persistentInstIdx = g_VisBuf[g_SlotOffsets[slotIdx] + instanceInSlot];
        AssetLodRange al = g_AssetLodRanges[slotIdx];
        InstanceRenderData inst = g_Instances[persistentInstIdx];

        if (gtid == 0)
        {
            s_payload.instanceIdx = persistentInstIdx;
            s_payload.assetLod    = slotIdx;
        }

        uint meshletLocalIdx = invocIdx * XYLEM_AS_GROUP_SIZE + gtid;
        if (meshletLocalIdx < al.meshletCount)
        {
            MeshletDesc m = g_Meshlets[al.meshletOffset + meshletLocalIdx];
            bool cull =
                MeshletConeCull(m, inst.model, g_CameraPos) || 
                MeshletHiZOccluded(m, inst.model);
            if (!cull)
            {
                uint slot;
                InterlockedAdd(s_survivors, 1, slot);
                s_payload.meshletIndices[slot] = meshletLocalIdx;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    DispatchMesh(s_survivors, 1, 1, s_payload);
}

// =============================================================================
// Mesh Shader — main color pass
// =============================================================================

struct V2P
{
    float4 pos       : SV_Position;
    float3 worldPos  : WORLD_POS;
    float  viewZ     : VIEW_Z;
    float3 normal    : NORMAL;
    float3 tangent   : TANGENT;
    float3 bitangent : BITANGENT;
    float2 uv        : UV;
};

[numthreads(XYLEM_MS_GROUP_SIZE, 1, 1)]
[outputtopology("triangle")]
void main_ms(
    uint   gtid : SV_GroupThreadID,
    uint3  gid  : SV_GroupID,
    in payload ASPayload i_payload,
    out indices  uint3 o_tris[XYLEM_MAX_MESHLET_PRIMS],
    out vertices V2P   o_verts[XYLEM_MAX_MESHLET_VERTS])
{
    uint meshletLocalIdx = i_payload.meshletIndices[gid.x];

    AssetLodRange      al   = g_AssetLodRanges[i_payload.assetLod];
    MeshletDesc        m    = g_Meshlets[al.meshletOffset + meshletLocalIdx];
    InstanceRenderData inst = g_Instances[i_payload.instanceIdx];

    SetMeshOutputCounts(m.vertexCount, m.triangleCount);

    if (gtid < m.vertexCount)
    {
        uint localVertIdx  = g_MeshletVertIdx[al.meshletVertBase + m.vertexOffset + gtid];
        uint globalVertIdx = al.vertexAttribBase + localVertIdx;

        float3 p  = g_Positions[globalVertIdx];
        float3 n  = g_Normals[globalVertIdx];
        float3 t  = g_Tangents[globalVertIdx];
        float3 b  = g_Bitangents[globalVertIdx];
        float2 uv = g_UVs[globalVertIdx];

        float4 worldPos = mul(float4(p, 1.0), inst.model);

        V2P v;
        v.pos       = mul(worldPos, viewProj);
        v.worldPos  = worldPos.xyz;
        v.viewZ     = mul(worldPos, viewMatrix).z;
        v.normal    = normalize(mul(n, inst.normal));
        v.tangent   = normalize(mul(t, inst.normal));
        v.bitangent = normalize(mul(b, inst.normal));
        v.uv        = uv;
        o_verts[gtid] = v;
    }

    if (gtid < m.triangleCount)
    {
        uint triByteOffset = al.meshletPrimBase + m.triangleOffset + gtid * 3u;
        o_tris[gtid] = LoadMeshletTriangle(triByteOffset);
    }
}

// =============================================================================
// Shadow pass — AS/MS, no PS (depth-only).
// Slot encoding: shadowSlot = assetIdx * NUM_CASCADES + cascade
// Bindings reuse t9..t13 — C++ binds shadow buffers at those slots.
// =============================================================================

[numthreads(XYLEM_AS_GROUP_SIZE, 1, 1)]
void shadow_as(uint3 gid  : SV_GroupID,
               uint  gtid : SV_GroupThreadID)
{
    if (gtid == 0) s_survivors = 0;
    GroupMemoryBarrierWithGroupSync();

    uint slotIdx         = g_SlotIdx;              // shadow slot = ai*NUM_CASCADES + c
    // Shadow casts from the lowest LOD (matches compute pipeline). Casting from
    // LOD 0 self-shadows the inscribed lower-LOD surface in the color pass.
    uint assetLodSlot    = (slotIdx / NUM_CASCADES) * g_NumLods + (g_NumLods - 1);
    uint ASInvocsPerInst = max(1u, g_ASInvocsPerSlot[slotIdx]);
    uint visibleCount    = g_SlotCounts[slotIdx];

    uint flatGroup      = gid.x + gid.y * XYLEM_DISPATCH_X;
    uint instanceInSlot = flatGroup / ASInvocsPerInst;
    uint invocIdx       = flatGroup % ASInvocsPerInst;

    if (instanceInSlot < visibleCount)
    {
        uint persistentInstIdx = g_VisBuf[g_SlotOffsets[slotIdx] + instanceInSlot];
        AssetLodRange al = g_AssetLodRanges[assetLodSlot];

        if (gtid == 0)
        {
            s_payload.instanceIdx = persistentInstIdx;
            s_payload.assetLod    = slotIdx;  // shadow MS decodes cascade from this
        }

        uint meshletLocalIdx = invocIdx * XYLEM_AS_GROUP_SIZE + gtid;
        if (meshletLocalIdx < al.meshletCount)
        {
            uint slot;
            InterlockedAdd(s_survivors, 1, slot);
            s_payload.meshletIndices[slot] = meshletLocalIdx;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    DispatchMesh(s_survivors, 1, 1, s_payload);
}

struct ShadowV2P
{
    float4 pos : SV_Position;
};

[numthreads(XYLEM_MS_GROUP_SIZE, 1, 1)]
[outputtopology("triangle")]
void shadow_ms(
    uint   gtid : SV_GroupThreadID,
    uint3  gid  : SV_GroupID,
    in payload ASPayload i_payload,
    out indices   uint3     o_tris[XYLEM_MAX_MESHLET_PRIMS],
    out vertices  ShadowV2P o_verts[XYLEM_MAX_MESHLET_VERTS])
{
    uint meshletLocalIdx = i_payload.meshletIndices[gid.x];
    uint shadowSlot      = i_payload.assetLod;
    uint assetLodSlot    = (shadowSlot / NUM_CASCADES) * g_NumLods + (g_NumLods - 1);
    uint cascade         = shadowSlot % NUM_CASCADES;

    AssetLodRange      al   = g_AssetLodRanges[assetLodSlot];
    MeshletDesc        m    = g_Meshlets[al.meshletOffset + meshletLocalIdx];
    InstanceRenderData inst = g_Instances[i_payload.instanceIdx];

    SetMeshOutputCounts(m.vertexCount, m.triangleCount);

    if (gtid < m.vertexCount)
    {
        uint localVertIdx  = g_MeshletVertIdx[al.meshletVertBase + m.vertexOffset + gtid];
        uint globalVertIdx = al.vertexAttribBase + localVertIdx;

        float3 p = g_Positions[globalVertIdx];
        float4 worldPos = mul(float4(p, 1.0), inst.model);

        ShadowV2P v;
        v.pos = mul(worldPos, lightViewProj[cascade]);
        o_verts[gtid] = v;
    }

    if (gtid < m.triangleCount)
    {
        uint triByteOffset = al.meshletPrimBase + m.triangleOffset + gtid * 3u;
        o_tris[gtid] = LoadMeshletTriangle(triByteOffset);
    }
}

// =============================================================================
// Depth-prepass mesh shader — reuses main_as for culling, no PS.
// =============================================================================

struct DepthV2P
{
    float4 pos : SV_Position;
};

[numthreads(XYLEM_MS_GROUP_SIZE, 1, 1)]
[outputtopology("triangle")]
void depth_ms(
    uint   gtid : SV_GroupThreadID,
    uint3  gid  : SV_GroupID,
    in payload ASPayload i_payload,
    out indices   uint3    o_tris[XYLEM_MAX_MESHLET_PRIMS],
    out vertices  DepthV2P o_verts[XYLEM_MAX_MESHLET_VERTS])
{
    uint meshletLocalIdx = i_payload.meshletIndices[gid.x];

    AssetLodRange      al   = g_AssetLodRanges[i_payload.assetLod];
    MeshletDesc        m    = g_Meshlets[al.meshletOffset + meshletLocalIdx];
    InstanceRenderData inst = g_Instances[i_payload.instanceIdx];

    SetMeshOutputCounts(m.vertexCount, m.triangleCount);

    if (gtid < m.vertexCount)
    {
        uint localVertIdx  = g_MeshletVertIdx[al.meshletVertBase + m.vertexOffset + gtid];
        uint globalVertIdx = al.vertexAttribBase + localVertIdx;

        float3 p = g_Positions[globalVertIdx];
        float4 worldPos = mul(float4(p, 1.0), inst.model);

        DepthV2P v;
        v.pos = mul(worldPos, viewProj);
        o_verts[gtid] = v;
    }

    if (gtid < m.triangleCount)
    {
        uint triByteOffset = al.meshletPrimBase + m.triangleOffset + gtid * 3u;
        o_tris[gtid] = LoadMeshletTriangle(triByteOffset);
    }
}

// =============================================================================
// Pixel Shader — ambient + Lambert + bark + normal map + 4-cascade PCF shadow
// =============================================================================

float SampleShadowCascade(float3 worldPos, uint cascadeIdx)
{
    float4 posLS = mul(float4(worldPos, 1), lightViewProj[cascadeIdx]);
    float2 shadowUV = posLS.xy * float2(0.5, -0.5) + 0.5;

    uint width, height, elements;
    t_ShadowMap.GetDimensions(width, height, elements);
    float2 texelSize = 1.0 / float2(width, height);

    float shadow = 0.0;

    static const float weights[3][3] = {
        { 1.0, 2.0, 1.0 },
        { 2.0, 4.0, 2.0 },
        { 1.0, 2.0, 1.0 }
    };

    [unroll]
    for (int x = -1; x <= 1; ++x)
    {
        [unroll]
        for (int y = -1; y <= 1; ++y)
        {
            float2 offset = float2(x, y) * texelSize;
            float  weight = weights[x + 1][y + 1];

            shadow += weight * t_ShadowMap.SampleCmpLevelZero(
                s_ShadowSampler,
                float3(shadowUV + offset, float(cascadeIdx)),
                posLS.z
            );
        }
    }
    return shadow / 16.0;
}

void main_ps(in V2P i_v, out float4 o_color : SV_Target0)
{
    float3 T = normalize(i_v.tangent);
    float3 B = normalize(i_v.bitangent);
    float3 N = normalize(i_v.normal);
    float3x3 TBN = float3x3(T, B, N);

    float3 tangentNormal = normalize(t_NormalMap.Sample(s_Sampler, i_v.uv).rgb * 2.0 - 1.0);
    float3 worldNormal   = normalize(mul(tangentNormal, TBN));

    float3 lightDir = -normalize(sunLightDir);
    float  diffuse  = max(dot(worldNormal, lightDir), 0.0);

    uint cascadeIdx = 3;
    if      (i_v.viewZ < cascadeSplits.x) cascadeIdx = 0;
    else if (i_v.viewZ < cascadeSplits.y) cascadeIdx = 1;
    else if (i_v.viewZ < cascadeSplits.z) cascadeIdx = 2;

    float notInShadow = SampleShadowCascade(i_v.worldPos, cascadeIdx);

    float ambient  = 0.15;
    float lighting = ambient + (1.0 - ambient) * diffuse * notInShadow;
    float3 albedo  = t_Diffuse.Sample(s_Sampler, i_v.uv).rgb;
    o_color = float4(lighting * albedo, 1.0);
}
