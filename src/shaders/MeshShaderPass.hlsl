#pragma pack_matrix(row_major)

#include "meshlet_types.hlsli"

static const uint NUM_CASCADES = 4;

cbuffer CB : register(b0)
{
    float4x4 viewProj;
    float4x4 viewMatrix;
    float4x4 lightViewProj[NUM_CASCADES];
    float3   sunLightDir;
    float    _pad0;
    float4   cascadeSplits;
};

// Root constant (PushConstants): slot identity forwarded by the command signature's
// CONSTANT argument. Each ExecuteIndirect record rewrites this before DISPATCH_MESH.
cbuffer PushC : register(b1)
{
    uint g_SlotIdx;
};

struct InstanceRenderData
{
    float4x4 model;     // offset   0, size 64
    float3x3 normal;    // offset  64, size 36
    uint     treeId;    // offset 100, size 4
};

// Mega-buffers
StructuredBuffer<float3>             g_Positions      : register(t0);
StructuredBuffer<float3>             g_Normals        : register(t1);
StructuredBuffer<float3>             g_Tangents       : register(t2);
StructuredBuffer<float3>             g_Bitangents     : register(t3);
StructuredBuffer<float2>             g_UVs            : register(t4);
StructuredBuffer<uint>               g_MeshletVertIdx : register(t5);
ByteAddressBuffer                    g_MeshletPrimIdx : register(t6);
StructuredBuffer<MeshletDesc>        g_Meshlets       : register(t7);
StructuredBuffer<AssetLodRange>      g_AssetLodRanges : register(t8);

// GPU-cull outputs
StructuredBuffer<uint>               g_VisBuf           : register(t9);   // compacted persistent-inst indices
StructuredBuffer<uint>               g_SlotOffsets      : register(t10);  // per-slot base into g_VisBuf
StructuredBuffer<uint>               g_SlotCounts       : register(t11);  // per-slot visible instance count
StructuredBuffer<InstanceRenderData> g_Instances        : register(t12);  // persistent instance buffer
StructuredBuffer<uint>               g_ASInvocsPerSlot  : register(t13);  // ceil(meshletCount / AS_GROUP_SIZE)

Texture2D                            t_Diffuse        : register(t14);
Texture2D                            t_NormalMap      : register(t15);
SamplerState                         s_Sampler        : register(s0);

// =============================================================================
// Amplification Shader — pass-through (no cone / no Hi-Z)
// =============================================================================
//
// One AS threadgroup = one "chunk" = up to XYLEM_AS_GROUP_SIZE meshlets of one
// instance. gid.x is the flat chunk index within the slot (slot = asset*LOD).
// Total groups.x per slot = visibleInstances * chunksPerSlot — produced by the
// GPU cull pass into a DISPATCH_MESH_ARGUMENTS record.

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

    uint instanceInSlot = gid.x / ASInvocsPerInst;
    uint invocIdx       = gid.x % ASInvocsPerInst;

    if (instanceInSlot < visibleCount)
    {
        uint persistentInstIdx = g_VisBuf[g_SlotOffsets[slotIdx] + instanceInSlot];
        AssetLodRange al = g_AssetLodRanges[slotIdx];

        if (gtid == 0)
        {
            s_payload.instanceIdx = persistentInstIdx;
            s_payload.assetLod    = slotIdx;
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

// =============================================================================
// Mesh Shader
// =============================================================================

struct V2P
{
    float4 pos       : SV_Position;
    float3 worldPos  : WORLD_POS;
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
        v.normal    = normalize(mul(n, inst.normal));
        v.tangent   = normalize(mul(t, inst.normal));
        v.bitangent = normalize(mul(b, inst.normal));
        v.uv        = uv;
        o_verts[gtid] = v;
    }

    if (gtid < m.triangleCount)
    {
        uint triByteOffset = al.meshletPrimBase + m.triangleOffset + gtid * 3u;
        uint alignedOffset = triByteOffset & ~3u;
        uint byteShift     = (triByteOffset - alignedOffset) * 8u;

        uint w0 = g_MeshletPrimIdx.Load(alignedOffset);
        uint w1 = g_MeshletPrimIdx.Load(alignedOffset + 4u);
        uint lo = (w0 >> byteShift);
        uint hi = (byteShift == 0) ? 0 : (w1 << (32u - byteShift));
        uint tri24 = (lo | hi) & 0x00FFFFFFu;

        o_tris[gtid] = uint3(tri24 & 0xFFu,
                             (tri24 >> 8) & 0xFFu,
                             (tri24 >> 16) & 0xFFu);
    }
}

// =============================================================================
// Pixel Shader (ambient + Lambert + bark + normal map; no shadows)
// =============================================================================

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

    float ambient  = 0.15;
    float lighting = ambient + (1.0 - ambient) * diffuse;
    float3 albedo  = t_Diffuse.Sample(s_Sampler, i_v.uv).rgb;
    o_color = float4(lighting * albedo, 1.0);
}
