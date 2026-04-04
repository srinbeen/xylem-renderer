#pragma pack_matrix(row_major)

// ---------------------------------------------------------------------------
// Constant buffer — matches Render::CullConstantBufferEntry (C++ side)
// ---------------------------------------------------------------------------

struct plane {
    float3 n;
    float d;
};
typedef plane frustum[6];

struct box3 {
    float3 min;
    float3 max;
};

cbuffer CullCB : register(b0)
{
    // P0 fields (also read by VS/PS)
    float4x4 view;
    float4x4 projection;
    float4x4 lightViewProj;
    float3   sunLightDir;
    float    _pad0;
    float3x4 _pad1;

    // Cull fields
    frustum   viewFrustum;
    frustum   lightFrustum;
    float3   cameraPos;
    uint     totalCapacity;
    float4   lodDistances[3];   // .x = distance threshold; .yzw unused
    uint     numLods;
    uint     numSlots;
    uint     visBufferSize;
    uint     _pad2;
};

// ---------------------------------------------------------------------------
// Structured buffers
// ---------------------------------------------------------------------------
struct CullInstanceData
{
    box3   bbox;
    uint   baseSlot;   // treeId * numLODs
    uint   active;     // 1 = live, 0 = dead
};

StructuredBuffer<CullInstanceData> cullData       : register(t0);
StructuredBuffer<uint>             slotOffsets    : register(t1);

RWStructuredBuffer<uint>           countBuffer    : register(u0);  // [numSlots]
RWStructuredBuffer<uint>           visibilityBuf  : register(u1);  // [visBufferSize]
RWStructuredBuffer<uint>           shadowCount    : register(u2);  // [1]
RWStructuredBuffer<uint>           shadowVisBuf   : register(u3);  // [totalCapacity]

bool FrustumCullAABB(box3 bbox, frustum f);
uint SelectLOD(box3 bbox);

// ---------------------------------------------------------------------------
// CSMain — main camera frustum cull + LOD
// ---------------------------------------------------------------------------

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID)
{
    uint idx = dtid.x;
    if (idx >= totalCapacity) return;

    CullInstanceData inst = cullData[idx];
    if (!inst.active) return;

    if (!FrustumCullAABB(inst.bbox, viewFrustum)) return;

    uint lod  = SelectLOD(inst.bbox);
    uint slot = inst.baseSlot + lod;
    if (slot >= numSlots) return;

    uint writeIdx;
    InterlockedAdd(countBuffer[slot], 1, writeIdx);
    uint writePos = slotOffsets[slot] + writeIdx;
    if (writePos >= visBufferSize) return;
    visibilityBuf[writePos] = idx;
}

// ---------------------------------------------------------------------------
// CSShadow — shadow frustum cull (lowest LOD only, no LOD selection)
// ---------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void CSShadow(uint3 dtid : SV_DispatchThreadID)
{
    uint idx = dtid.x;
    if (idx >= totalCapacity) return;

    CullInstanceData inst = cullData[idx];
    if (!inst.active) return;

    if (!FrustumCullAABB(inst.bbox, lightFrustum)) return;

    uint writeIdx;
    InterlockedAdd(shadowCount[0], 1, writeIdx);
    shadowVisBuf[writeIdx] = idx;
}

// ---------------------------------------------------------------------------
// AABB frustum test — p-vertex method
// ---------------------------------------------------------------------------
bool FrustumCullAABB(box3 bbox, frustum f)
{
    [unroll]
    for (int i = 0; i < 6; i++)
    {
        float3 n = f[i].n;
        float3 p;
        p.x = (n.x > 0) ? bbox.min.x : bbox.max.x;
        p.y = (n.y > 0) ? bbox.min.y : bbox.max.y;
        p.z = (n.z > 0) ? bbox.min.z : bbox.max.z;
        if (dot(n, p) > f[i].d) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// LOD selection — distance from camera to nearest point on AABB
// ---------------------------------------------------------------------------
uint SelectLOD(box3 bbox)
{
    float3 nearest = clamp(cameraPos, bbox.min, bbox.max);
    float dist = distance(cameraPos, nearest);

    for (uint i = 0; i < numLods; i++)
    {
        if (dist < lodDistances[i].x)
            return i;
    }
    return numLods - 1;
}