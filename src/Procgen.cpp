#include "include/Procgen.hpp"

#include <stack>
#include <stdexcept>

using namespace Xylem::ProcGen;

void LSystem::generate(uint32_t iterations) {
    lstring_t next;
    for (uint32_t i = 0; i < iterations; i++) {
        next.reserve(m_current.size() * m_growthFactor);
        next.clear();
        for (const auto& op : m_current) {
            auto it = m_rules.find(op);
            if (it != m_rules.cend())
                next.insert(next.end(), it->second.begin(), it->second.end());
            else
                next.push_back(op);
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
    buffers.bbox = dm::box3::empty();
    m_branchCounter = 0;
    m_ringCounter   = 0;

    std::stack<TurtleState> stateStack;
    TurtleState state;

    state.baseRingIndex = createRing(state, buffers);

    for (const auto& op : lSystemString) {
        switch (op) {
        case F:
            state.branchLength += state.stepLength;
            state.pos          += dm::applyQuat(state.orientation, unit_k) * state.stepLength;
            state.radius       *= params.taperRatio  * (1.f + 0.03f * branchRand());
            state.stepLength   *= params.stepRatio   * (1.f + 0.03f * branchRand());
            state.baseRingIndex = createRing(state, buffers);
            break;
        case X:
            break;
        case Y_POS:
            state.orientation *= dm::rotationQuat(unit_j,  params.branchAngle);  break;
        case Y_NEG:
            state.orientation *= dm::rotationQuat(unit_j, -params.branchAngle);  break;
        case P_POS:
            state.orientation *= dm::rotationQuat(unit_i,  params.branchAngle);  break;
        case P_NEG:
            state.orientation *= dm::rotationQuat(unit_i, -params.branchAngle);  break;
        case R_POS:
            state.orientation *= dm::rotationQuat(unit_k,  params.branchAngle);  break;
        case R_NEG:
            state.orientation *= dm::rotationQuat(unit_k, -params.branchAngle);  break;
        case S_PUSH:
            stateStack.push(state);
            state.radius     *= params.taperRatio  * (1.f + 0.03f * branchRand());
            state.stepLength *= params.stepRatio   * (1.f + 0.03f * branchRand());
            break;
        case S_POP:
            if (stateStack.empty())
                throw std::runtime_error("L-System stack underflow on ']'");
            state = stateStack.top();
            stateStack.pop();
            break;
        }
    }
}

uint32_t TreeGenerator::createRing(const TurtleState& state, Buffers& buffers) {
    const dm::float3 forward = dm::applyQuat(state.orientation, unit_k);
    const dm::float3 right   = dm::applyQuat(state.orientation, unit_i);
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
        buffers.uvs.push_back(dm::float2(percentage, state.branchLength));
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

    buffers.bbox.m_mins = dm::min(buffers.bbox.m_mins, state.pos + state.radius * dm::normalize(dm::float3(-1.f)));
    buffers.bbox.m_maxs = dm::max(buffers.bbox.m_maxs, state.pos + state.radius * dm::normalize(dm::float3( 1.f)));

    return nextRingIndex;
}
