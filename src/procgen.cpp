#include "include/procgen.hpp"

#include <stack>
#include <stdexcept>

using namespace ProcGen;

void LSystem::generate(uint32_t iterations) {    
    lstring_t next;
    // uint32_t finalNumIterations = currentIter+iterations;
    
    // check cache
    // cache has exact iteration or larger
    // if (finalNumIterations < lStringCache.size()) {
    //     current = lStringCache[finalNumIterations];
    //     goto SET_ITER;
    // }
    // cache has larger iteration than current but not the requested iteration
    // else if (lStringCache.size()-1 > currentIter) {
    //     current = lStringCache.back();
    // }

    for (uint32_t i = 0; i < iterations; i++) {
        next.reserve(current.size() * growthFactor);
        next.clear();

        for (const auto& op : current) {
            if (rules.find(op) != rules.cend()) {
                next.insert(next.end(), rules.at(op).begin(), rules.at(op).end());
            } else {
                next.push_back(op);
            }
        }

        // lStringCache.push_back(next);
        current = std::move(next);
    }

    //SET_ITER:
    //currentIter = finalNumIterations;
    return;
}

lstring_t LSystem::getBinaryFromString(const std::string& s) const {
    lstring_t res;
    res.reserve(s.size());

    for (const auto& c : s) {
        res.push_back(charToBin.at(c));
    }

    return res;
}

lrules_t LSystem::getBinaryMapFromStringMap(const std::unordered_map<char, std::string>& m) const {
    lrules_t res;
    res.reserve(m.size());

    for (const auto& [c,s] : m) {
        res.insert({charToBin.at(c), getBinaryFromString(s)});
    }

    return res;
}

void LSystem::setGrowthFactor() {
    growthFactor = std::accumulate(rules.cbegin(), rules.cend(), 0,
        [](size_t currentMax, const auto& pair) {
            return std::max(currentMax, pair.second.size());
        }
    );
}

void TreeGenerator::generateVertexAndIndexBuffers(const lstring_t& lSystemString, ProcGenBuffers& buffers) {
    buffers.vertices.clear();
    buffers.indices.clear();
    buffers.bbox = dm::box3::empty();

    std::stack<TurtleState> stateStack{};
    TurtleState state;

    const dm::float2 randomStepRange{0.8f,1.f};

    // set base ring
    state.baseRingIndex = createRing(state, buffers);
    for (const auto& op : lSystemString) {
        switch (op) {
        case F:
            state.branchLength += state.stepLength;
            state.pos += dm::applyQuat(state.orientation, unit_k) * state.stepLength;
                // * (params.seed ? dm::lerp(randomStepRange.x,randomStepRange.y,rand_dist(rand_gen)) : 1);
            state.radius *= params.taperRatio;
                // * (params.seed ? dm::lerp(randomStepRange.x,randomStepRange.y,rand_dist(rand_gen)) : 1);
            state.stepLength *= params.stepRatio;
                // * (params.seed ? dm::lerp(randomStepRange.x,randomStepRange.y,rand_dist(rand_gen)) : 1);
            state.baseRingIndex = createRing(state, buffers);
            break;
        case X:
            break;
            
        // rotate on an axis and apply locally
        case Y_POS:
            state.orientation *= dm::rotationQuat(unit_j, 
                params.branchAngle // * (params.seed ? dm::lerp(randomStepRange.x,randomStepRange.y,rand_dist(rand_gen)) : 1)
            );
            break;
        case Y_NEG:
            state.orientation *= dm::rotationQuat(unit_j, 
                -params.branchAngle // * (params.seed ? dm::lerp(randomStepRange.x,randomStepRange.y,rand_dist(rand_gen)) : 1)
            );
            break;
        case P_POS:
            state.orientation *= dm::rotationQuat(unit_i, 
                params.branchAngle // * (params.seed ? dm::lerp(randomStepRange.x,randomStepRange.y,rand_dist(rand_gen)) : 1)
            );
            break;
        case P_NEG:
            state.orientation *= dm::rotationQuat(unit_i, 
                -params.branchAngle // * (params.seed ? dm::lerp(randomStepRange.x,randomStepRange.y,rand_dist(rand_gen)) : 1)
            );
            break;
        case R_POS:
            state.orientation *= dm::rotationQuat(unit_k, 
                params.branchAngle // * (params.seed ? dm::lerp(randomStepRange.x,randomStepRange.y,rand_dist(rand_gen)) : 1)
            );
            break;
        case R_NEG:
            state.orientation *= dm::rotationQuat(unit_k, 
                -params.branchAngle // * (params.seed ? dm::lerp(randomStepRange.x,randomStepRange.y,rand_dist(rand_gen)) : 1)
            );
            break;
        case S_PUSH:
            stateStack.push(state);
            state.radius *= params.taperRatio;
                // * (params.seed ? dm::lerp(randomStepRange.x,randomStepRange.y,rand_dist(rand_gen)) : 1);
            state.stepLength *= params.stepRatio;
                // * (params.seed ? dm::lerp(randomStepRange.x,randomStepRange.y,rand_dist(rand_gen)) : 1);
            break;
        case S_POP:   
            if (!stateStack.empty()) {
                state = stateStack.top();
                stateStack.pop();
            }
            else {
                throw std::runtime_error("stack shouldn't be empty when popping!");
            }
            break;     
        default:
            throw std::runtime_error("unknown operation!");
            break;
        }
    }
}

uint32_t TreeGenerator::createRing(const TurtleState& state, ProcGenBuffers& buffers) {
    // assumes forward is local +z using left handed system
    const dm::float3 forward = dm::applyQuat(state.orientation, unit_k);
    const dm::float3 right = dm::applyQuat(state.orientation, unit_i);
    const dm::float3 up = dm::cross(forward, right); // dm::applyQuat(state.orientation, unit_j);

    const dm::float2 normalRandomRange{0.9f,1.f};

    float a = 2 * dm::PI_f * state.radius / params.radialSegments;
    dm::float2 tangentRandomRange{0,a/8.0f};
    dm::float2 radiusRandomRange{0,2 * dm::PI_f};
    
    // starting index of newly generated ring
    uint32_t nextRingIndex = static_cast<uint32_t>(buffers.vertices.size());

    uint32_t radialSegments =  params.radialSegments;
    for (uint32_t i = 0; i < radialSegments+1; i++) {
        float percentage = dm::clamp((float)i / (float)radialSegments, 0.f, 1.f);
        float angle = 2.f * dm::PI_f * percentage;

        dm::float3 normal = dm::normalize(std::cos(angle) * right + std::sin(angle) * up);
        dm::float3 tangent = dm::cross(normal, forward);
        
        // float randAngle = (params.seed ? dm::lerp(radiusRandomRange.x,radiusRandomRange.y,rand_dist(rand_gen)) : 1);
        dm::float3 pos = state.pos + normal 
            * state.radius; // * (params.seed ? dm::lerp(normalRandomRange.x,normalRandomRange.y,rand_dist(rand_gen)) : 1)
            // + ((params.seed && nextRingIndex != 0 && radialSegments != 2) ? (std::cos(randAngle) * tangent + std::sin(randAngle) * forward) 
            //     * dm::lerp(tangentRandomRange.x,tangentRandomRange.y,rand_dist(rand_gen)) : dm::float3(0.f));

        buffers.vertices.emplace_back(
            pos,
            radialSegments != 2 ? normal : up,
            dm::float2(percentage, state.branchLength)
        );
    }
    
    // don't bridge rings if first one
    if (nextRingIndex != 0) {
        uint32_t baseRingIndex = state.baseRingIndex;
        
        for (uint32_t i = 0; i < radialSegments; i++) {
            // rings are added ccw, but i want cw faces
            /*
            
            1 4-5
            |\ \|
            0-2 3

            where 2=3 and 1=4, and 2/3 is the base ring start
            */
            buffers.indices.insert(buffers.indices.end(), {
                baseRingIndex+((i+1)%(radialSegments+1)), nextRingIndex+((i+1)%(radialSegments+1)), baseRingIndex+i,
                baseRingIndex+i, nextRingIndex+((i+1)%(radialSegments+1)), nextRingIndex+i,
            });
        }
    }

    // conservatively update bbox by increasing bounds based on the current position 
    // and the max position of the joint's ring
    buffers.bbox.m_mins = dm::min(
        buffers.bbox.m_mins, 
        state.pos + state.radius * dm::normalize(dm::float3(-1.f))
    );
    buffers.bbox.m_maxs = dm::max(
        buffers.bbox.m_maxs, 
        state.pos + state.radius * dm::normalize(dm::float3(1.f))
    );

    return nextRingIndex;
}