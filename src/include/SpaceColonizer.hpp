#ifndef XYLEM_SPACE_COLONIZER_H
#define XYLEM_SPACE_COLONIZER_H

#include <cstdint>
#include <vector>

#include <donut/core/math/math.h>

namespace Xylem::ProcGen {

    struct SCParams {
        // attractorCount = 0 disables space colonization entirely
        uint32_t attractorCount     = 0;
        float    influenceDistance  = 4.0f;
        float    killDistance       = 0.6f;
        float    segmentLength      = 0.4f;
        uint32_t maxIterations      = 64;

        // Affine transform applied to a unit ball to produce the attractor cloud.
        // Stored decomposed (composition order S → Shear → R → T) so the JSON and UI stay
        // collapse to a dm::affine3 via crownTransform().
        dm::float3 crownTranslation     = dm::float3(0.f, 8.f, 0.f);
        dm::float3 crownRotationDegrees = dm::float3(0.f, 0.f, 0.f);   // XYZ Euler
        dm::float3 crownScale           = dm::float3(5.f, 5.f, 5.f);
        dm::float3 crownShear           = dm::float3(0.f, 0.f, 0.f);   // (XY, XZ, YZ)

        float    branchletTaper     = 0.95f;  // child radius = parent radius * taper
        uint32_t seed               = 0;      // 0 = derive from asset seed

        dm::affine3 crownTransform() const;
    };

    struct SCNode {
        dm::float3 pos;
        dm::float3 dir;
        dm::float3 right;
        int32_t    parent;         // -1 for L-system tip seeds
        bool       terminal;
        float      radius;
        float      branchLength;
    };

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
                  const std::vector<dm::float3>& tipRights,
                  const std::vector<float>&      tipRadii,
                  const std::vector<float>&      tipBranchLengths);

        const std::vector<SCNode>&   nodes()     const { return m_nodes; }
        const std::vector<uint32_t>& terminals() const { return m_terminals; }

    private:
        SCParams              m_p;
        std::vector<SCNode>   m_nodes;
        std::vector<uint32_t> m_terminals;
    };

} // namespace Xylem::ProcGen

#endif // XYLEM_SPACE_COLONIZER_H
