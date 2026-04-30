#ifndef XYLEM_SPACE_COLONIZER_H
#define XYLEM_SPACE_COLONIZER_H

#include <cstdint>
#include <vector>

#include <donut/core/math/math.h>

namespace Xylem::ProcGen {

    struct SCParams {
        // attractorCount = 0 disables space colonization entirely. Default is OFF so existing
        // scenes that don't define a `colonization` JSON block render identically to before;
        // new assets opt in by setting attractorCount > 0 in scene JSON or via the UI.
        uint32_t attractorCount     = 0;
        float    influenceDistance  = 4.0f;
        float    killDistance       = 0.6f;
        float    segmentLength      = 0.4f;
        uint32_t maxIterations      = 64;
        float    crownRadiusFactor  = 0.5f;   // multiple of L-system tree height
        float    crownYOffsetFactor = 1.0f;   // sphere center placed at bbox.min.y + factor * height
        float    branchletRadius    = 0.05f;  // radius of root SC nodes (seeded from L-system tips)
        float    branchletTaper     = 0.95f;  // child radius = parent radius * taper
        uint32_t seed               = 0;      // 0 = derive from asset seed
    };

    struct SCNode {
        dm::float3 pos;
        dm::float3 dir;            // segment direction INTO this node (forward of its ring's plane)
        dm::float3 right;          // local X axis for ring vertex layout — parallel-transported from parent
        int32_t    parent;         // -1 for L-system tip seeds
        bool       terminal;       // true after grow() if no child node is spawned from this one
        float      radius;
        float      branchLength;   // arc-length distance from the originating L-system tip (uv.y continuity)
    };

    // Runions/Lane-style space colonization. Consumes a list of L-system branch tips,
    // grows a graph of nodes toward random attractor points placed in a crown volume
    // around the L-system bbox, and exposes the resulting graph + terminal indices.
    class SpaceColonizer {
    public:
        void setParams(const SCParams& p) { m_p = p; }
        const SCParams& getParams() const { return m_p; }

        // Reset internal state and grow a fresh graph from the given tips. Tip dirs are
        // unit vectors representing the L-system turtle's forward at the tip.
        // tipRights / tipRadii / tipBranchLengths inherit from the L-system tip so the
        // branchlet's first ring matches the L-system tip's basis (no twist), tapers
        // smoothly, and continues UV.y across the join. When empty, fallbacks are used.
        void grow(const std::vector<dm::float3>& tipPositions,
                  const std::vector<dm::float3>& tipDirs,
                  const dm::box3&                lsystemBbox,
                  const std::vector<dm::float3>& tipRights        = {},
                  const std::vector<float>&      tipRadii         = {},
                  const std::vector<float>&      tipBranchLengths = {});

        const std::vector<SCNode>&   nodes()     const { return m_nodes; }
        const std::vector<uint32_t>& terminals() const { return m_terminals; }

    private:
        SCParams              m_p;
        std::vector<SCNode>   m_nodes;
        std::vector<uint32_t> m_terminals;
    };

} // namespace Xylem::ProcGen

#endif // XYLEM_SPACE_COLONIZER_H
