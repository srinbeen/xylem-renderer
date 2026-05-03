#include "include/SpaceColonizer.hpp"
#include "include/hash.hpp"

#include <algorithm>
#include <cmath>

using namespace Xylem::ProcGen;

namespace {

// Uniform sample inside a unit sphere using three hashed scalars.
// Rejection-free: pick (theta, phi) on a sphere shell, then a cube-root radius.
inline dm::float3 sampleUnitBall(uint32_t streamIdx, uint32_t seed) {
    const float u1 = Xylem::hashToFloat(streamIdx * 3u + 0u, seed);
    const float u2 = Xylem::hashToFloat(streamIdx * 3u + 1u, seed);
    const float u3 = Xylem::hashToFloat(streamIdx * 3u + 2u, seed);

    const float cosTheta = 1.f - 2.f * u1;
    const float sinTheta = std::sqrt(std::max(0.f, 1.f - cosTheta * cosTheta));
    const float phi      = 2.f * dm::PI_f * u2;
    const float r        = std::cbrt(u3);

    return dm::float3(r * sinTheta * std::cos(phi),
                      r * cosTheta,
                      r * sinTheta * std::sin(phi));
}

} // namespace

dm::affine3 SCParams::crownTransform() const {
    const dm::float3 eulerRad(dm::radians(crownRotationDegrees.x),
                              dm::radians(crownRotationDegrees.y),
                              dm::radians(crownRotationDegrees.z));

    // Row-vector shear: v'.x += shXY*v.y + shXZ*v.z, v'.y += shYZ*v.z. Off-diagonals live in
    // column 0 (rows 1,2) and column 1 (row 2) so the multiplication v*S yields the formulas
    // above; transposing this would shear in the wrong direction.
    const dm::affine3 shearAffine(
        dm::float3x3(1.f,          0.f,          0.f,
                     crownShear.x, 1.f,          0.f,
                     crownShear.y, crownShear.z, 1.f),
        dm::float3(0.f));

    // affine composition (a*b).transformPoint(v) = b.transformPoint(a.transformPoint(v)),
    // so this list reads in apply-order: scale, then shear, then rotate, then translate.
    return dm::scaling(crownScale)
         * shearAffine
         * dm::rotation(eulerRad)
         * dm::translation(crownTranslation);
}

namespace {

inline dm::float3 nearestRight(const dm::float3& parentDir,
                               const dm::float3& parentRight,
                               const dm::float3& childDir)
{
    // |axis| = sin(theta)
    const dm::float3    axis = dm::cross(parentDir, childDir);
    const float         sina = dm::length(axis);
    const float         cosa = dm::clamp(dm::dot(parentDir, childDir), -1.f, 1.f);

    // either parallel or anti-parallel, return same right vector
    if (dm::isnear(sina, 0.0f)) return parentRight;

    const dm::float3 nAxis = axis / sina;
    const float      angle = std::atan2(sina, cosa);
    const dm::quat   q     = dm::rotationQuat(nAxis, angle);
    dm::float3       r     = dm::applyQuat(q, parentRight);

    // take out any component of childDir in r (should be none since orthogonal)
    // but error can accumulate, then normalize
    r = dm::normalize(r - childDir * dm::dot(r, childDir));
    return r;
}

} // namespace

void SpaceColonizer::grow(const std::vector<dm::float3>& tipPositions,
                          const std::vector<dm::float3>& tipDirs,
                          const std::vector<dm::float3>& tipRights,
                          const std::vector<float>&      tipRadii,
                          const std::vector<float>&      tipBranchLengths)
{
    m_nodes.clear();
    m_terminals.clear();

    if (tipPositions.empty() || m_p.attractorCount == 0 || m_p.maxIterations == 0)
        return;

    // user-transformed unit ball
    const dm::affine3 crownXf = m_p.crownTransform();

    std::vector<dm::float3> attractors;
    std::vector<uint8_t>    attractorAlive;
    attractors.reserve(m_p.attractorCount);
    attractorAlive.reserve(m_p.attractorCount);
    for (uint32_t i = 0; i < m_p.attractorCount; ++i) {
        const dm::float3 unit = sampleUnitBall(i, m_p.seed ^ 0xA1F00DEu);
        attractors.push_back(crownXf.transformPoint(unit));
        attractorAlive.push_back(1);
    }

    // L-system terminals are start of SC
    // if any are too close to each other they are deleted as a seed
    const size_t numTips    = tipPositions.size();
    const float  dedupR2    = m_p.segmentLength * m_p.segmentLength;
    std::vector<size_t> keptTipIdx;
    keptTipIdx.reserve(numTips);
    for (size_t i = 0; i < numTips; ++i) {
        bool tooClose = false;
        for (size_t k : keptTipIdx) {
            if (dm::lengthSquared(tipPositions[i] - tipPositions[k]) < dedupR2) { tooClose = true; break; }
        }
        if (!tooClose) keptTipIdx.push_back(i);
    }

    // reserves for a heuristic of each seed growing maxIterations
    // not truly upper-bound, but good approximate to avoid reallocs
    m_nodes.reserve(keptTipIdx.size() + m_p.maxIterations * keptTipIdx.size());
    for (size_t i : keptTipIdx) {
        SCNode n;
        n.pos          = tipPositions[i];
        n.dir          = dm::normalize(tipDirs[i]);
        n.right        = dm::normalize(tipRights[i]);
        n.parent       = -1;
        n.terminal     = false;
        n.radius       = tipRadii[i];
        n.branchLength = tipBranchLengths[i];
        m_nodes.push_back(n);
    }

    // ---- Iterative growth.
    const float influence2 = m_p.influenceDistance * m_p.influenceDistance;
    const float kill2      = m_p.killDistance      * m_p.killDistance;

    
    for (uint32_t iter = 0; iter < m_p.maxIterations; ++iter) {
        std::vector<dm::float3> pendingDir(m_nodes.size(), dm::float3(0.f));

        // For each live attractor, find nearest node within influence distance.
        bool anyAttractorLive = false;
        for (size_t a = 0; a < attractors.size(); ++a) {
            if (!attractorAlive[a]) continue;
            anyAttractorLive = true;

            int32_t bestIdx = -1;
            float   bestD2  = influence2;
            dm::float3 closestNodetoAttractor = dm::float3::zero();
            
            for (size_t n = 0; n < m_nodes.size(); ++n) {
                const dm::float3 testNodeToAttractor = attractors[a] - m_nodes[n].pos;
                const float d2 = dm::lengthSquared(testNodeToAttractor);
                if (d2 < bestD2) {
                    bestD2 = d2; 
                    closestNodetoAttractor = testNodeToAttractor;
                    bestIdx = static_cast<int32_t>(n); 
                }
            }
            // no node was attracted to it
            if (bestIdx < 0) continue;
            // node is at kill distance
            if (bestD2 < kill2) {
                attractorAlive[a] = 0;
                continue;
            }

            const float distToAttrractor = std::sqrtf(bestD2);
            if (!dm::isnear(distToAttrractor, 0.0f)) {
                pendingDir[bestIdx]    += dm::normalize(closestNodetoAttractor);
            }
        }
        if (!anyAttractorLive) break;

        // Spawn children for nodes that received attractor pull this iteration.
        const size_t prevNodeCount = m_nodes.size();
        bool anyGrowth = false;
        for (size_t n = 0; n < prevNodeCount; ++n) {
            // if never pulled, or pulled in opposite directions and cancelled
            if (dm::isnear(dm::lengthSquared(pendingDir[n]), 0.0f, 1e-12f)) continue;
            const dm::float3 growDir = dm::normalize(pendingDir[n]);

            SCNode child;
            child.pos          = m_nodes[n].pos + growDir * m_p.segmentLength;
            child.dir          = growDir;
            child.right        = nearestRight(m_nodes[n].dir, m_nodes[n].right, growDir);
            child.parent       = static_cast<int32_t>(n);
            child.terminal     = false;
            child.radius       = m_nodes[n].radius * std::sqrt(m_p.branchletTaper);
            child.branchLength = m_nodes[n].branchLength + m_p.segmentLength;

            m_nodes.push_back(child);
            anyGrowth = true;
        }

        if (!anyGrowth) break;
    }

    // reverse lookup
    std::vector<uint8_t> hasChild(m_nodes.size(), 0);
    for (const auto& n : m_nodes) {
        if (n.parent >= 0) hasChild[n.parent] = 1;
    }

    // Collect terminal candidates (no children, parent != -1), ordered by
    // branchLength descending so spatial dedup keeps the most-developed tip.
    std::vector<uint32_t> candidates;
    candidates.reserve(m_nodes.size());
    for (uint32_t i = 0; i < m_nodes.size(); ++i) {
        if (hasChild[i]) continue;
        if (m_nodes[i].parent < 0) continue;
        candidates.push_back(i);
    }

    // kill nodes within kill distance of each other (if they converged to an attractor)
    for (uint32_t i : candidates) {
        bool tooClose = false;
        for (uint32_t k : m_terminals) {
            if (dm::lengthSquared(m_nodes[i].pos - m_nodes[k].pos) < kill2) {
                tooClose = true;
                break;
            }
        }
        if (tooClose) continue;
        m_nodes[i].terminal = true;
        m_terminals.push_back(i);
    }
}
