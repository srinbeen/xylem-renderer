#ifndef PROC_GEN_H
#define PROC_GEN_H

#include <string>
#include <unordered_map>
#include <numeric>

#include <donut/core/math/math.h>

#include "hash.hpp"

namespace Xylem::ProcGen {

    struct TreeVertex {
        dm::float3  pos;
        dm::float3  normal;
        dm::float3  tangent;
        dm::float3  bitangent;
        dm::float2  uv;

        TreeVertex(dm::float3 p, dm::float3 n, dm::float3 t, dm::float3 b, dm::float2 tc)
            : pos{p}, normal{n}, tangent{t}, bitangent{b}, uv{tc} {}
    };

    struct TurtleState {
        dm::float3  pos;
        dm::quat    orientation;
        float       radius;
        float       stepLength;
        uint32_t    baseRingIndex;
        float       branchLength;

        TurtleState() : pos{0.f}, orientation{}, radius{0.75f}, stepLength{1.0f}, baseRingIndex{0}, branchLength{0.f} {}
    };

    struct Buffers {
        std::vector<TreeVertex> vertices;
        std::vector<uint32_t>   indices;
        dm::box3                bbox;
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
    using lrules_t  = std::unordered_map<LSystemAlphabet, lstring_t>;
    using lgen_t    = uint32_t;

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
        std::unordered_map<char, std::string>  m_rulesStr;

        lstring_t  m_axiom;
        lrules_t   m_rules;
        lstring_t  m_current;
        uint32_t   m_growthFactor;

    public:
        LSystem(const std::string& axiom, const std::unordered_map<char, std::string>& rules)
            : m_axiomStr{axiom}, m_rulesStr{rules}
        {
            m_axiom   = _toBinary(axiom);
            m_rules   = _toBinaryRules(rules);
            m_current = m_axiom;
            _updateGrowthFactor();
        }


        void setAxiom(const std::string& axiom) {
            m_axiomStr = axiom;
            m_axiom    = _toBinary(axiom);
            m_current  = m_axiom;
        }

        void setRules(const std::unordered_map<char, std::string>& rules) {
            m_rulesStr = rules;
            m_rules    = _toBinaryRules(rules);
            m_current  = m_axiom;
            _updateGrowthFactor();
        }

        void reset() { m_current = m_axiom; }

        void generate(uint32_t iterations);

        const std::string&                           getAxiomStr()   const { return m_axiomStr; }
        const std::unordered_map<char, std::string>& getRulesStr()   const { return m_rulesStr; }
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
                res.insert({ charToBin.at(c), _toBinary(s) });
            return res;
        }

        void _updateGrowthFactor() {
            m_growthFactor = std::accumulate(m_rules.cbegin(), m_rules.cend(), 0,
                [](uint32_t cur, const auto& pair) -> uint32_t { return std::max(cur, static_cast<uint32_t>(pair.second.size())); });
        }
    };


    class TreeGenerator {
    public:
        struct Params {
            uint32_t radialSegments = 8;
            float    stepLength     = 1.f;
            float    branchAngle    = 25.f;
            float    taperRatio     = 0.9f;
            float    stepRatio      = 0.95f;
            uint32_t seed           = 0;  // 0 = no randomization
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

        void     generateVertexAndIndexBuffers(const lstring_t& lSystemString, Buffers& buffers);
        uint32_t createRing(const TurtleState& state, Buffers& buffers);
    };

} // namespace Xylem::ProcGen

#endif // PROC_GEN_H
