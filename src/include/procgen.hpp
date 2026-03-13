#ifndef PROC_GEN_H
#define PROC_GEN_H

#include <string>
#include <unordered_map>
#include <map>
#include <numeric>

#include <random>
// #include <cstdlib>

#include <donut/core/math/math.h>

namespace ProcGen {
    struct TreeVertex {
        dm::float3  pos;
        dm::float3  normal;
        dm::float2  uv;

        TreeVertex(dm::float3 p, dm::float3 n, dm::float2 t) : pos{p}, normal{n}, uv{t} {}
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

    struct ProcGenBuffers {
        std::vector<TreeVertex> vertices;
        std::vector<uint32_t>   indices;
        dm::box3                bbox;
    };

    enum LSystemAlphabet : uint16_t {
        F,
        X,
        Y_POS,
        Y_NEG,
        P_POS,
        P_NEG,
        R_POS,
        R_NEG,
        S_PUSH,
        S_POP
    };

    inline constexpr dm::float3 unit_i = dm::float3(1.f, 0.f, 0.f);
    inline constexpr dm::float3 unit_j = dm::float3(0.f, 1.f, 0.f);
    inline constexpr dm::float3 unit_k = dm::float3(0.f, 0.f, 1.f);

    using lstring_t = std::vector<LSystemAlphabet>;
    using lrules_t = std::unordered_map<LSystemAlphabet, lstring_t>;
    using lgen_t = uint32_t;
    
    // using lstring_t = std::string;
    // using lrules_t = std::unordered_map<char, lstring_t>;

    class LSystem {
    public:
        static inline const std::unordered_map<char, LSystemAlphabet> charToBin = {
            {'F', F},
            {'X', X},
            {'+', Y_POS},
            {'-', Y_NEG},
            {'>', P_POS},
            {'<', P_NEG},
            {'&', R_POS},
            {'^', R_NEG},
            {'[', S_PUSH},
            {']', S_POP}
        };

    private:
        const lstring_t axiom;
        const lrules_t rules;
        
        // uint32_t currentIter;
        lstring_t current;
        uint32_t growthFactor;

        // std::vector<lstring_t> lStringCache;

    public:
        LSystem(const lstring_t& start, const lrules_t& r)
            :   axiom{start}, 
                rules{r}, 
                // currentIter{0},
                current{start}
                // lStringCache{start}
            { setGrowthFactor(); }
        
        LSystem(const std::string& start, const std::unordered_map<char, std::string>& r)
            :   LSystem(
                    std::move(getBinaryFromString(start)),
                    std::move(getBinaryMapFromStringMap(r))
                ) 
            { }

    public:
        void generate(uint32_t iterations);
        void reset() { current = axiom; /* currentIter = 0; */ }

        lstring_t getBinaryFromString(const std::string& s) const;
        lrules_t getBinaryMapFromStringMap(const std::unordered_map<char, std::string>& m) const;

        const lstring_t& getCurrentString() const { return current; }



    private:
        void setGrowthFactor();
    };


    class TreeGenerator {
    public:
        // to be able to hold multiple versions of param in the generator
        struct Params {
            uint32_t radialSegments = 8;
            float stepLength = 1.f;
            float branchAngle = dm::radians(25.f);
            float taperRatio = 0.9f;
            float stepRatio = 0.95f;

            // seed = 0 -> no randomization
            uint32_t seed = 0;
        };
    private:
        Params params;

        std::default_random_engine rand_gen;
        std::uniform_real_distribution<float> rand_dist;
        

    public:
        // seed of 0 is no randomization
        TreeGenerator(const Params& params) : params{params}, rand_gen{params.seed}, rand_dist{} {}
        TreeGenerator() = default;

    public:
        void setParams(const Params& p) { params = p; };
        
        void generateVertexAndIndexBuffers(const lstring_t& lSystemString, ProcGenBuffers& buffers);
        uint32_t createRing(const TurtleState& state, ProcGenBuffers& buffers);
        void resetRandomGenerator() { rand_gen.seed(params.seed); rand_dist.reset(); }
    };
};

#endif // PROC_GEN_H