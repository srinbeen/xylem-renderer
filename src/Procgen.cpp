#include "include/Procgen.hpp"
#include "include/SpaceColonizer.hpp"

#include <cmath>
#include <stack>
#include <stdexcept>

using namespace Xylem::ProcGen;

namespace {
constexpr dm::float3 kGrowthForwardAxis = unit_j;
constexpr dm::float3 kYawAxis           = -unit_k;
constexpr dm::float3 kPitchAxis         = unit_i;
constexpr dm::float3 kRollAxis          = kGrowthForwardAxis;
}

void LSystem::generate(uint32_t iterations) {
    // Per-symbol counter feeds the production-selection hash. Reset each generate() call
    // so a given (seed, iteration count) always yields the same string.
    uint32_t streamCounter = 0;

    lstring_t next;
    for (uint32_t iter = 0; iter < iterations; iter++) {
        next.reserve(m_current.size() * m_growthFactor);
        next.clear();
        for (const auto& op : m_current) {
            auto it = m_rules.find(op);
            if (it == m_rules.cend()) {
                next.push_back(op);
                continue;
            }

            const auto& prods = it->second;
            if (prods.empty()) {
                next.push_back(op);
                continue;
            }

            // Single production: append directly, skip hashing.
            if (prods.size() == 1) {
                const auto& rhs = prods.front().rhsBin;
                next.insert(next.end(), rhs.begin(), rhs.end());
                continue;
            }

            // Multiple productions: pick by weight. Seed=0 forces deterministic first-production
            // so legacy scenes load unchanged.
            const WeightedProduction* picked = &prods.front();
            if (m_seed != 0) {
                float totalWeight = 0.f;
                for (const auto& p : prods) totalWeight += p.weight;
                if (totalWeight > 0.f) {
                    const uint32_t saltedSeed = m_seed ^ (iter * 0x9e3779b9u);
                    const float    r          = Xylem::hashToFloat(streamCounter++, saltedSeed) * totalWeight;
                    float          accum      = 0.f;
                    for (const auto& p : prods) {
                        accum += p.weight;
                        if (r < accum) { picked = &p; break; }
                    }
                }
            }
            next.insert(next.end(), picked->rhsBin.begin(), picked->rhsBin.end());
        }
        m_current = std::move(next);
    }
}

void TreeGenerator::generateVertexAndIndexBuffers(const lstring_t& lSystemString, Buffers& buffers) {
    buffers.positions.clear();
    buffers.normals.clear();
    buffers.tangents.clear();
    buffers.bitangents.clear();
    buffers.uvs.clear();
    buffers.indices.clear();
    buffers.branchTipPositions.clear();
    buffers.branchTipDirs.clear();
    buffers.branchTipRights.clear();
    buffers.branchTipRadii.clear();
    buffers.branchTipBranchLengths.clear();
    buffers.bbox = dm::box3::empty();
    m_branchCounter = 0;
    m_ringCounter   = 0;

    auto recordTipIfNeeded = [&](const TurtleState& s) {
        if (!s.lastDrewSegment) return;
        buffers.branchTipPositions.push_back(s.pos);
        buffers.branchTipDirs.push_back(dm::applyQuat(s.orientation, kGrowthForwardAxis));
        // Pass the L-system ring's actual right basis through to SC. createRing() at the
        // tip uses applyQuat(orientation, unit_i) for `right`, so vertex angles around the
        // ring are 0..2π measured from this axis. SC root nodes inherit it directly and
        // children parallel-transport it, eliminating roll discontinuities at the join.
        buffers.branchTipRights.push_back(dm::applyQuat(s.orientation, unit_i));
        buffers.branchTipRadii.push_back(s.radius);
        buffers.branchTipBranchLengths.push_back(s.branchLength);
    };

    std::stack<TurtleState> stateStack;
    TurtleState state(params);

    state.baseRingIndex = createRing(state, buffers);

    for (const auto& op : lSystemString) {
        switch (op) {
        case F:
            state.branchLength += state.stepLength;
            state.pos          += dm::applyQuat(state.orientation, kGrowthForwardAxis) * state.stepLength;
            state.radius       *= std::sqrtf(params.taperRatio)  * (1.f + 0.03f * branchRand());
            state.stepLength   *= std::sqrtf(params.stepRatio)   * (1.f + 0.03f * branchRand());
            state.baseRingIndex = createRing(state, buffers);
            state.lastDrewSegment = true;
            break;
        case X:
            break;
        case Y_POS:
            state.orientation *= dm::rotationQuat(kYawAxis,   params.branchAngle);  break;
        case Y_NEG:
            state.orientation *= dm::rotationQuat(kYawAxis,  -params.branchAngle);  break;
        case P_POS:
            state.orientation *= dm::rotationQuat(kPitchAxis, params.branchAngle);  break;
        case P_NEG:
            state.orientation *= dm::rotationQuat(kPitchAxis,-params.branchAngle);  break;
        case R_POS:
            state.orientation *= dm::rotationQuat(kRollAxis,  params.branchAngle);  break;
        case R_NEG:
            state.orientation *= dm::rotationQuat(kRollAxis, -params.branchAngle);  break;
        case S_PUSH:
            stateStack.push(state);
            state.radius     *= params.taperRatio  * (1.f + 0.03f * branchRand());
            state.stepLength *= params.stepRatio   * (1.f + 0.03f * branchRand());
            state.lastDrewSegment = false;
            break;
        case S_POP:
            if (stateStack.empty())
                throw std::runtime_error("L-System stack underflow on ']'");
            // The state about to be popped represents a closed [..F..] subtree; record its
            // endpoint as a branch tip if it actually emitted at least one segment.
            recordTipIfNeeded(state);
            state = stateStack.top();
            stateStack.pop();
            break;
        }
    }

    // Trailing trunk tip (the turtle's final state, if it ended on F).
    recordTipIfNeeded(state);
}

uint32_t TreeGenerator::createRing(const TurtleState& state, Buffers& buffers) {
    const dm::float3 forward = dm::applyQuat(state.orientation, kGrowthForwardAxis);
    const dm::float3 right   = dm::applyQuat(state.orientation, kPitchAxis);
    const dm::float3 up      = dm::cross(forward, right);

    uint32_t nextRingIndex = static_cast<uint32_t>(buffers.positions.size());
    uint32_t segs          = params.radialSegments;

    float firstRJitter = 0.f, firstNJitter = 0.f;
    for (uint32_t i = 0; i <= segs; i++) {
        float percentage = dm::clamp((float)i / (float)segs, 0.f, 1.f);
        float angle      = 2.f * dm::PI_f * percentage;

        float rJitter, nJitter;
        if (i < segs) {
            rJitter = state.radius * 0.02f * ringRand();
            nJitter = 0.015f * ringRand();
            if (i == 0) { firstRJitter = rJitter; firstNJitter = nJitter; }
        } else {
            rJitter = firstRJitter;
            nJitter = firstNJitter;
        }

        dm::float3 normal        = dm::normalize(std::cos(angle) * right + std::sin(angle) * up);
        dm::float3 shadingNormal = dm::normalize(normal + nJitter * forward);
        dm::float3 pos           = state.pos + normal * (state.radius + rJitter);

        dm::float3 bitangent = forward;
        dm::float3 tangent   = dm::cross(bitangent, normal);

        dm::float3 usedNormal = segs != 2 ? shadingNormal : up;
        buffers.positions.push_back(pos);
        buffers.normals.push_back(usedNormal);
        buffers.tangents.push_back(tangent);
        buffers.bitangents.push_back(bitangent);
        buffers.uvs.push_back(dm::float2(percentage, state.branchLength * uvYtilingFactor));
    }

    if (nextRingIndex != 0) {
        uint32_t baseRingIndex = state.baseRingIndex;
        for (uint32_t i = 0; i < segs; i++) {
            buffers.indices.insert(buffers.indices.end(), {
                baseRingIndex + ((i+1) % (segs+1)), nextRingIndex + ((i+1) % (segs+1)), baseRingIndex + i,
                baseRingIndex + i,                  nextRingIndex + ((i+1) % (segs+1)), nextRingIndex + i,
            });
        }
    }

    // conservatively updates bounding box the enclose potential that ring could lie on
    // buffers.bbox |= box3(state.pos - state.radius, state.pos + state.radius)
    buffers.bbox.m_mins = dm::min(buffers.bbox.m_mins, state.pos - state.radius);
    buffers.bbox.m_maxs = dm::max(buffers.bbox.m_maxs, state.pos + state.radius);

    return nextRingIndex;
}

void TreeGenerator::emitColonizationCylinders(const std::vector<SCNode>& nodes,
                                              Buffers&                   buffers,
                                              uint32_t                   radialSegments)
{
    if (nodes.empty()) return;

    const uint32_t segs = radialSegments;

    // One ring per node; child rings are quad-connected to their parent's ring.
    std::vector<uint32_t> nodeBaseRingIdx(nodes.size(), 0);

    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& n = nodes[i];

        const dm::float3 forward = n.dir;
        const dm::float3 right = n.right;
        const dm::float3 up = dm::cross(forward, right);

        const uint32_t newRingFirstVert = static_cast<uint32_t>(buffers.positions.size());

        for (uint32_t s = 0; s <= segs; ++s) {
            const float percentage = dm::clamp(static_cast<float>(s) / static_cast<float>(segs), 0.f, 1.f);
            const float angle      = 2.f * dm::PI_f * percentage;

            const dm::float3 normal = dm::normalize(std::cos(angle) * right + std::sin(angle) * up);
            const dm::float3 pos    = n.pos + normal * n.radius;

            const dm::float3 bitangent = forward;
            const dm::float3 tangent   = dm::cross(bitangent, normal);

            buffers.positions.push_back(pos);
            buffers.normals.push_back(normal);
            buffers.tangents.push_back(tangent);
            buffers.bitangents.push_back(bitangent);
            buffers.uvs.push_back(dm::float2(percentage, n.branchLength * uvYtilingFactor));
        }
        nodeBaseRingIdx[i] = newRingFirstVert;

        // Connect this ring to the parent ring with quads (matches createRing's winding).
        if (n.parent >= 0) {
            const uint32_t parentRing = nodeBaseRingIdx[n.parent];
            for (uint32_t s = 0; s < segs; ++s) {
                buffers.indices.insert(buffers.indices.end(), {
                    parentRing       + ((s + 1) % (segs + 1)),
                    newRingFirstVert + ((s + 1) % (segs + 1)),
                    parentRing       + s,
                    parentRing       + s,
                    newRingFirstVert + ((s + 1) % (segs + 1)),
                    newRingFirstVert + s,
                });
            }
        }

        buffers.bbox.m_mins = dm::min(buffers.bbox.m_mins, n.pos - n.radius);
        buffers.bbox.m_maxs = dm::max(buffers.bbox.m_maxs, n.pos + n.radius);
    }
}

