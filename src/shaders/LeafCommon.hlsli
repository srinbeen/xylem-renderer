#ifndef XYLEM_LEAF_COMMON_HLSLI
#define XYLEM_LEAF_COMMON_HLSLI

// Force row-major matrix packing inside this header so InstanceRenderData reads
// the C++ Render::InstanceBufferEntry layout correctly even when the including
// shader hasn't issued #pragma pack_matrix(row_major) yet. Without this, the
// model float4x4 is interpreted column-major: rotation is transposed and the
// translation row collapses into the w component, dumping every leaf near the
// world origin after the perspective divide.
#pragma pack_matrix(row_major)

struct InstanceRenderData
{
    float4x4 model;
    float3x3 normal;
    uint     treeId;
};

struct LeafInstanceData
{
    float4 centerHalfSize;
    float4 spine;
    float4 right;
    float4 color;
};

struct LeafSlotData
{
    uint leafOffset;
    uint leafCount;
    uint meshletOffset;
    uint meshletCount;
};

struct LeafMeshletData
{
    uint4  meta;   // x = global leaf offset, y = leaf count
    float4 bounds; // xyz = local center, w = radius
};

static const uint XYLEM_LEAF_VERTS_PER_LEAF = 12;
static const uint XYLEM_LEAFS_PER_MESHLET = 8;

uint LeafCornerIndex(uint vertexInLeaf)
{
    static const uint indices[12] = {
        0, 1, 2, 0, 2, 3,
        4, 5, 6, 4, 6, 7
    };
    return indices[vertexInLeaf];
}

void BuildLeafCorner(
    LeafInstanceData leaf,
    uint corner,
    out float3 localPos,
    out float3 localNormal)
{
    const float3 center = leaf.centerHalfSize.xyz;
    const float halfSize = leaf.centerHalfSize.w;
    const float3 spine = normalize(leaf.spine.xyz);
    const float3 right = normalize(leaf.right.xyz);
    const float3 q2 = normalize(cross(spine, right));

    const bool secondQuad = corner >= 4;
    const uint quadCorner = secondQuad ? corner - 4 : corner;
    const float3 widthAxis = secondQuad ? q2 : right;
    localNormal = secondQuad ? right : q2;

    const float sx = (quadCorner == 1 || quadCorner == 2) ? 1.0 : -1.0;
    const float sy = (quadCorner == 2 || quadCorner == 3) ? 1.0 : -1.0;
    localPos = center + widthAxis * (sx * halfSize) + spine * (sy * halfSize);
}

void BuildLeafVertex(
    LeafInstanceData leaf,
    uint vertexInLeaf,
    out float3 localPos,
    out float3 localNormal)
{
    BuildLeafCorner(leaf, LeafCornerIndex(vertexInLeaf), localPos, localNormal);
}

uint LeafIndexFromVertex(uint vertexId)
{
    return vertexId / XYLEM_LEAF_VERTS_PER_LEAF;
}

uint LeafVertexInLeaf(uint vertexId)
{
    return vertexId % XYLEM_LEAF_VERTS_PER_LEAF;
}

#endif // XYLEM_LEAF_COMMON_HLSLI
