#ifndef PROC_GEN_H
#define PROC_GEN_H

#include <string>
#include <unordered_map>
#include <numeric>
#include <utility>
#include <vector>

#include <donut/core/math/math.h>

#include "hash.hpp"

namespace Xylem::ProcGen {

    struct Buffers {
        std::vector<dm::float3> positions;
        std::vector<dm::float3> normals;
        std::vector<dm::float3> tangents;
        std::vector<dm::float3> bitangents;
        std::vector<dm::float2> uvs;
        std::vector<uint32_t>   indices;
        dm::box3                bbox;

        // Branch-tip records emitted by the turtle: every closed [..F..] subtree contributes
        // one (pos, forward, right, radius, branchLength) tuple, plus the trailing tip of
        // the trunk if it ended on F. Consumed by SpaceColonizer::grow() to seed branchlets
        // that taper from the L-system's tip radius, continue UV.y across the join, and use
        // the L-system's ring basis so the connecting quads don't twist.
        std::vector<dm::float3> branchTipPositions;
        std::vector<dm::float3> branchTipDirs;
        std::vector<dm::float3> branchTipRights;
        std::vector<float>      branchTipRadii;
        std::vector<float>      branchTipBranchLengths;
    };

    enum LSystemAlphabet : uint16_t {
        F,
        X,
        Y_POS,  // '+' yaw left
        Y_NEG,  // '-' yaw right
        P_POS,  // '>' pitch up
        P_NEG,  // '<' pitch down
        R_POS,  // '&' roll left
        R_NEG,  // '^' roll right
        S_PUSH, // '[' push state
        S_POP   // ']' pop state
    };

    inline constexpr dm::float3 unit_i = dm::float3(1.f, 0.f, 0.f);
    inline constexpr dm::float3 unit_j = dm::float3(0.f, 1.f, 0.f);
    inline constexpr dm::float3 unit_k = dm::float3(0.f, 0.f, 1.f);

    using lstring_t = std::vector<LSystemAlphabet>;

    // A single production for a stochastic L-system rule.
    // - weight: positive; relative probability among productions of the same predecessor symbol.
    // - rhs:    string form (kept for round-trip with UI/JSON).
    // - rhsBin: pre-binarized form, the actual thing appended during generate().
    struct WeightedProduction {
        float       weight;
        std::string rhs;
        lstring_t   rhsBin;
    };

    using lrules_t     = std::unordered_map<LSystemAlphabet, std::vector<WeightedProduction>>;
    using lrules_str_t = std::unordered_map<char, std::vector<std::pair<float, std::string>>>;
    using lgen_t       = uint32_t;

    class LSystem {
    public:
        // Bidirectional symbol maps — public so callers can validate input.
        static inline const std::unordered_map<char, LSystemAlphabet> charToBin = {
            {'F', F}, {'X', X},
            {'+', Y_POS}, {'-', Y_NEG},
            {'>', P_POS}, {'<', P_NEG},
            {'&', R_POS}, {'^', R_NEG},
            {'[', S_PUSH}, {']', S_POP}
        };
        static inline const std::unordered_map<LSystemAlphabet, char> binToChar = {
            {F, 'F'}, {X, 'X'},
            {Y_POS, '+'}, {Y_NEG, '-'},
            {P_POS, '>'}, {P_NEG, '<'},
            {R_POS, '&'}, {R_NEG, '^'},
            {S_PUSH, '['}, {S_POP, ']'}
        };

    private:
        std::string                            m_axiomStr;
        // Legacy single-rule round-trip: holds the FIRST production per char. Used by the
        // existing UI text editor. When the system is stochastic, m_weightedRulesStr below
        // is also populated; otherwise it stays empty.
        std::unordered_map<char, std::string>  m_rulesStr;
        lrules_str_t                           m_weightedRulesStr;

        lstring_t  m_axiom;
        lrules_t   m_rules;
        lstring_t  m_current;
        uint32_t   m_growthFactor = 1;
        uint32_t   m_seed         = 0;   // 0 = legacy deterministic (always pick first production)

    public:
        // Legacy deterministic constructor.
        LSystem(const std::string& axiom, const std::unordered_map<char, std::string>& rules)
            : m_axiomStr{axiom}, m_rulesStr{rules}
        {
            m_axiom   = _toBinary(axiom);
            m_rules   = _toBinaryRules(rules);
            m_current = m_axiom;
            _updateGrowthFactor();
        }

        // Weighted/stochastic constructor.
        LSystem(const std::string& axiom, const lrules_str_t& weightedRules)
            : m_axiomStr{axiom}, m_weightedRulesStr{weightedRules}
        {
            m_axiom   = _toBinary(axiom);
            m_rules   = _toBinaryWeightedRules(weightedRules);
            m_current = m_axiom;
            _refreshLegacyRulesStr();
            _updateGrowthFactor();
        }

        void setAxiom(const std::string& axiom) {
            m_axiomStr = axiom;
            m_axiom    = _toBinary(axiom);
            m_current  = m_axiom;
        }

        // Legacy deterministic setter — clears any weighted form.
        void setRules(const std::unordered_map<char, std::string>& rules) {
            m_rulesStr = rules;
            m_weightedRulesStr.clear();
            m_rules    = _toBinaryRules(rules);
            m_current  = m_axiom;
            _updateGrowthFactor();
        }

        // Weighted/stochastic setter.
        void setWeightedRules(const lrules_str_t& weightedRules) {
            m_weightedRulesStr = weightedRules;
            m_rules            = _toBinaryWeightedRules(weightedRules);
            m_current          = m_axiom;
            _refreshLegacyRulesStr();
            _updateGrowthFactor();
        }

        // Per-instance seed for deterministic stochastic generation. 0 = pick first production.
        void setSeed(uint32_t seed) { m_seed = seed; }
        uint32_t getSeed() const { return m_seed; }

        void reset() { m_current = m_axiom; }

        void generate(uint32_t iterations);

        const std::string&                           getAxiomStr()   const { return m_axiomStr; }
        const std::unordered_map<char, std::string>& getRulesStr()   const { return m_rulesStr; }
        const lrules_str_t&                          getWeightedRulesStr() const { return m_weightedRulesStr; }
        bool                                         isStochastic() const { return !m_weightedRulesStr.empty(); }
        const lstring_t&                             getCurrentString() const { return m_current; }

        std::string getCurrentStringAsText() const {
            std::string result;
            result.reserve(m_current.size());
            for (auto sym : m_current) result += binToChar.at(sym);
            return result;
        }

        static lstring_t toBinary(const std::string& s)                                     { return LSystem::_toBinary(s); }
        static lrules_t  toBinaryRules(const std::unordered_map<char, std::string>& rules)  { return LSystem::_toBinaryRules(rules); }

    private:
        static lstring_t _toBinary(const std::string& s) {
            lstring_t res;
            res.reserve(s.size());
            for (char c : s) res.push_back(charToBin.at(c));
            return res;
        }

        static lrules_t _toBinaryRules(const std::unordered_map<char, std::string>& rules) {
            lrules_t res;
            res.reserve(rules.size());
            for (const auto& [c, s] : rules)
                res.insert({ charToBin.at(c), { WeightedProduction{ 1.f, s, _toBinary(s) } } });
            return res;
        }

        static lrules_t _toBinaryWeightedRules(const lrules_str_t& wRules) {
            lrules_t res;
            res.reserve(wRules.size());
            for (const auto& [c, prods] : wRules) {
                std::vector<WeightedProduction> vec;
                vec.reserve(prods.size());
                for (const auto& [w, s] : prods)
                    vec.push_back({ std::max(0.f, w), s, _toBinary(s) });
                res.insert({ charToBin.at(c), std::move(vec) });
            }
            return res;
        }

        // After (re)building m_rules from a weighted setter, populate m_rulesStr with the
        // first production per char so legacy consumers (UI text editor) keep working.
        void _refreshLegacyRulesStr() {
            m_rulesStr.clear();
            for (const auto& [sym, prods] : m_rules) {
                if (prods.empty()) continue;
                m_rulesStr[binToChar.at(sym)] = prods.front().rhs;
            }
        }

        void _updateGrowthFactor() {
            m_growthFactor = 1;
            for (const auto& [sym, prods] : m_rules)
                for (const auto& p : prods)
                    m_growthFactor = std::max(m_growthFactor, static_cast<uint32_t>(p.rhsBin.size()));
            if (m_growthFactor == 0) m_growthFactor = 1;
        }
    };


    // Forward-declared from SpaceColonizer.hpp so TreeGenerator can consume the SC graph
    // without Procgen.hpp pulling in the full colonizer header. The .cpp includes the full
    // definition.
    struct SCNode;

    class TreeGenerator {
    public:
        struct Params {
            uint32_t radialSegments = 8;
            float    baseLength     = 1.f;
            float    baseRadius     = 1.0f;
            float    branchAngle    = 25.f;
            float    taperRatio     = 0.9f;
            float    stepRatio      = 0.95f;
            uint32_t seed           = 0;  // 0 = no randomization
        };

        struct TurtleState {
        dm::float3  pos;
        dm::quat    orientation;
        float       radius;
        float       stepLength;
        uint32_t    baseRingIndex;
        float       branchLength;
        // True if this state has emitted at least one F since its last [ push or root.
        // Used to detect branch tips for space-colonization seeding.
        bool        lastDrewSegment;

        TurtleState() : pos{0.f}, orientation{}, radius{1.0f}, stepLength{1.0f}, baseRingIndex{0}, branchLength{0.f}, lastDrewSegment{false} {}
        TurtleState(const Params& p) : pos{0.f}, orientation{}, radius{p.baseRadius}, stepLength{p.baseLength}, baseRingIndex{0}, branchLength{0.f}, lastDrewSegment{false} {}
    };

    private:
        Params   params;
        uint32_t m_branchCounter = 0;
        uint32_t m_ringCounter   = 0;

        // Returns [-1, 1], or 0 when seed == 0 (no randomization).
        // Branch-level randomization — independent of radial segment count.
        float branchRand() {
            if (params.seed == 0) return 0.f;
            return Xylem::hashToFloat(m_branchCounter++, params.seed) * 2.f - 1.f;
        }
        // Ring vertex randomization — varies with LOD, which is expected.
        float ringRand() {
            if (params.seed == 0) return 0.f;
            return Xylem::hashToFloat(m_ringCounter++, params.seed ^ 0x9e3779b9u) * 2.f - 1.f;
        }

    public:
        TreeGenerator(const Params& p) : params{p} {}
        TreeGenerator() = default;

        void setParams(const Params& p) { params = p; }
        const Params& getParams() const { return params; }

        void     generateVertexAndIndexBuffers(const lstring_t& lSystemString, Buffers& buffers);
        uint32_t createRing(const TurtleState& state, Buffers& buffers);

        // Append cylinder rings + connecting quads for every parent→child edge in the SC
        // graph. Each node becomes a ring at node.pos oriented along node.dir; non-root
        // nodes also generate quad indices joining their ring to the parent's ring.
        // Branchlet uvs respect the global tree invariant (uv.x ∈ [0,1], uv.y ≥ 0) so the
        // future leaf sentinel `uv.x < 0` is unambiguous.
        void emitColonizationCylinders(const std::vector<SCNode>& nodes,
                                       Buffers&                   buffers,
                                       uint32_t                   radialSegments);

        // Append solid cross-billboard leaves (two perpendicular quads per leaf, 8 verts /
        // 12 indices) at each terminal SC node. Leaf vertices are tagged in-band via
        // `uv = (-1, -1)` so pixel shaders can route them through a leaf-shading branch
        // without an extra vertex stream. `countPerTip` is the LOD-scaled count (caller
        // does the multiplier math); when 0, the call is a no-op.
        //
        // The per-asset leaf color is packed into each leaf vertex's TANGENT slot. This
        // avoids plumbing a new per-instance attribute (treeId) and a per-asset SRV through
        // all three pipelines. Trunk vertices store a real tangent for normal mapping;
        // leaf vertices store an RGB color. The PS branches on `uv.x < 0` and reinterprets
        // the tangent stream accordingly.
        void emitLeafCrosses(const std::vector<SCNode>&   nodes,
                             const std::vector<uint32_t>& terminals,
                             uint32_t                     countPerTip,
                             float                        size,
                             const dm::float3&            color,
                             uint32_t                     seed,
                             Buffers&                     buffers);
    };

} // namespace Xylem::ProcGen

#endif // PROC_GEN_H
