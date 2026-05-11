#ifndef XYLEM_MESHLET_CONSTANTS_H
#define XYLEM_MESHLET_CONSTANTS_H

// Single source of truth for meshlet pipeline sizing constants. Included
// from both C++ (Meshlet.hpp / MeshShaderRenderPass.cpp via header chain)
// and HLSL (meshlet_types.hlsli, LeafCommon.hlsli) — keep it preprocessor-
// only so it stays valid in both languages.

// meshopt_buildMeshlets ceilings — must match the values passed to meshopt
// when building MeshletMegabuffers.
#define XYLEM_MAX_MESHLET_VERTS                 64
#define XYLEM_MAX_MESHLET_PRIMS                 124

// AS threadgroup size = meshlets considered per amplification group.
// Sized to a single wave on every supported arch; payload arrays in
// meshlet_types.hlsli scale with this.
#define XYLEM_AS_GROUP_SIZE                     32

// MS threadgroup size must cover max(verts, prims) so every declared
// vertex/primitive has a writing thread — otherwise undeclared
// primitives read garbage indices and the GPU hangs.
#define XYLEM_MS_GROUP_SIZE                     128

// D3D12 caps DispatchMesh per-axis at 65535. When the work-item count
// exceeds this, the CPU spreads across X/Y; AS reconstructs a flat work
// index using this.
#define XYLEM_DISPATCH_X                        65535

// Leaf meshlet packing.
//   MS path (MeshLeaves.hlsl, indexed): 8 unique corners per leaf, 4 prims per
//     leaf → 8 leaves * 8 = 64 verts (== XYLEM_MAX_MESHLET_VERTS), 8 * 4 = 32
//     prims (< XYLEM_MAX_MESHLET_PRIMS).
//   VS path (TraditionalLeaves / ComputeLeaves, unindexed): emits 12 verts per
//     leaf via LeafCornerIndex (corners 0 and 2 duplicated to form two quads).
//     That's where XYLEM_LEAF_VERTS_PER_LEAF = 12 is used.
#define XYLEM_LEAFS_PER_MESHLET                 8
#define XYLEM_LEAF_VERTS_PER_LEAF               12

#endif // XYLEM_MESHLET_CONSTANTS_H
