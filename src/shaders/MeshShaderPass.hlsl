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

struct InstanceRenderData
{
    float4x4 model;     // offset   0, size 64
    float3x3 normal;    // offset  64, size 36
    uint     treeId;    // offset 100, size 4
};

// Mega-buffers
StructuredBuffer<float3>            g_Positions     : register(t0);
StructuredBuffer<float3>            g_Normals       : register(t1);
StructuredBuffer<float3>            g_Tangents      : register(t2);
StructuredBuffer<float3>            g_Bitangents    : register(t3);
StructuredBuffer<float2>            g_UVs           : register(t4);
StructuredBuffer<uint>              g_MeshletVertIdx: register(t5);
ByteAddressBuffer                   g_MeshletPrimIdx: register(t6);
StructuredBuffer<MeshletDesc>       g_Meshlets      : register(t7);
StructuredBuffer<AssetLodRange>     g_AssetLodRanges: register(t8);
StructuredBuffer<VisibleInstance>   g_VisibleInsts  : register(t9);
StructuredBuffer<InstanceRenderData> g_Instances    : register(t10);

Texture2D              t_Diffuse                   : register(t11);
Texture2D              t_NormalMap                 : register(t12);
SamplerState           s_Sampler                   : register(s0);

// Frustum sphere test using 6 planes derived from viewProj.
// Plane form: dot(plane.xyz, p) + plane.w >= 0 means "inside".
void extractFrustumPlanes(out float4 planes[6])
{
    // Row-major viewProj: rows are basis vectors.
    float4 r0 = float4(viewProj._11, viewProj._21, viewProj._31, viewProj._41);
    float4 r1 = float4(viewProj._12, viewProj._22, viewProj._32, viewProj._42);
    float4 r2 = float4(viewProj._13, viewProj._23, viewProj._33, viewProj._43);
    float4 r3 = float4(viewProj._14, viewProj._24, viewProj._34, viewProj._44);

    planes[0] = r3 + r0; // left
    planes[1] = r3 - r0; // right
    planes[2] = r3 + r1; // bottom
    planes[3] = r3 - r1; // top
    planes[4] = r2;      // near (reverse-Z: z>=0 in clip)
    planes[5] = r3 - r2; // far
    [unroll] for (uint i = 0; i < 6; i++)
    {
        float len = length(planes[i].xyz);
        if (len > 0.0) planes[i] /= len;
    }
}

bool sphereInFrustum(float3 center, float radius, float4 planes[6])
{
    [unroll] for (uint i = 0; i < 6; i++)
    {
        if (dot(planes[i].xyz, center) + planes[i].w < -radius) return false;
    }
    return true;
}

// =============================================================================
// Amplification Shader
// =============================================================================

groupshared ASPayload s_payload;
groupshared uint      s_survivors;

[numthreads(XYLEM_AS_GROUP_SIZE, 1, 1)]
void main_as(uint3 gid  : SV_GroupID,
             uint  gtid : SV_GroupThreadID)
{
    if (gtid == 0) s_survivors = 0;
    GroupMemoryBarrierWithGroupSync();

    // 2D dispatch: reconstruct a flat work-item index. CPU launches
    // dispatchMesh(min(W, XYLEM_DISPATCH_X), ceil(W / XYLEM_DISPATCH_X), 1).
    // Tail groups past the last real work item leave s_survivors at 0 — the
    // final DispatchMesh emits zero mesh groups. Guarded rather than early-
    // returned to keep DispatchMesh the single dominating call (DXC requires it).
    uint workIdx        = gid.y * XYLEM_DISPATCH_X + gid.x;
    uint totalWorkItems = asuint(_pad0);
    bool validWork      = (workIdx < totalWorkItems);

    if (validWork)
    {
        // Each work item = one chunk of up to XYLEM_AS_GROUP_SIZE meshlets of
        // one instance. Multiple chunks per instance share instanceIdx + assetLod.
        VisibleInstance    vi   = g_VisibleInsts[workIdx];
        AssetLodRange      al   = g_AssetLodRanges[vi.assetLod];
        InstanceRenderData inst = g_Instances[vi.instanceIdx];

        if (gtid == 0)
        {
            s_payload.instanceIdx = vi.instanceIdx;
            s_payload.assetLod    = vi.assetLod;
        }

        uint meshletLocalIdx = vi.meshletChunkBase + gtid;
        if (meshletLocalIdx < al.meshletCount)
        {
            MeshletDesc m = g_Meshlets[al.meshletOffset + meshletLocalIdx];

            // Transform sphere to world space using the instance model matrix.
            // Radius scaled by max axis length (approx; OK for uniform scales).
            float3 centerWS = mul(float4(m.bounds.xyz, 1.0), inst.model).xyz;
            float  scale = max(max(length(inst.model._11_12_13),
                                    length(inst.model._21_22_23)),
                               length(inst.model._31_32_33));
            float  radiusWS = m.bounds.w * scale;

            float4 planes[6];
            extractFrustumPlanes(planes);
            if (sphereInFrustum(centerWS, radiusWS, planes))
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

    // Emit vertices
    if (gtid < m.vertexCount)
    {
        uint localVertIdx = g_MeshletVertIdx[al.meshletVertBase + m.vertexOffset + gtid];
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

    // Emit triangles. Packed 3 bytes per triangle in g_MeshletPrimIdx at byte
    // (al.meshletPrimBase + m.triangleOffset + gtid*3). Load as 4 bytes and mask.
    if (gtid < m.triangleCount)
    {
        uint triByteOffset = al.meshletPrimBase + m.triangleOffset + gtid * 3u;
        // Align-down to 4 bytes and extract the 3 bytes spanning the boundary.
        uint alignedOffset = triByteOffset & ~3u;
        uint byteShift     = (triByteOffset - alignedOffset) * 8u;

        // Load two u32s to cover any boundary crossing (triByte occupies 3 bytes).
        uint w0 = g_MeshletPrimIdx.Load(alignedOffset);
        uint w1 = g_MeshletPrimIdx.Load(alignedOffset + 4u);
        // Combine to a 64-bit-ish chunk and extract bytes [byteShift, byteShift+24) bits.
        uint lo = (w0 >> byteShift);
        uint hi = (byteShift == 0) ? 0 : (w1 << (32u - byteShift));
        uint tri24 = (lo | hi) & 0x00FFFFFFu;

        o_tris[gtid] = uint3(tri24 & 0xFFu,
                             (tri24 >> 8) & 0xFFu,
                             (tri24 >> 16) & 0xFFu);
    }
}

// =============================================================================
// Pixel Shader (MVP: ambient + Lambert + bark + normal map; no shadows)
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
