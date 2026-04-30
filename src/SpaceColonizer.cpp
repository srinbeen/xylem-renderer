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

inline float dist2(const dm::float3& a, const dm::float3& b) {
    const dm::float3 d = a - b;
    return dm::dot(d, d);
}

} // namespace

namespace {

// Parallel-transport `parentRight` (a unit vector ⊥ parentDir) onto the plane ⊥ childDir.
// This is the rotation that takes parentDir → childDir, applied to parentRight. Falls back
// gracefully when the directions are parallel (no rotation needed) or anti-parallel (pick
// any orthogonal axis). Result is unit-length and ⊥ childDir.
inline dm::float3 parallelTransportRight(const dm::float3& parentDir,
                                         const dm::float3& parentRight,
                                         const dm::float3& childDir)
{
    const dm::float3 axis = dm::cross(parentDir, childDir);
    const float      sina = dm::length(axis);
    const float      cosa = dm::clamp(dm::dot(parentDir, childDir), -1.f, 1.f);

    if (sina < 1e-6f) {
        // Parallel (cosa ≈ +1) — basis unchanged. Anti-parallel (cosa ≈ -1) is degenerate;
        // fabricate any orthogonal axis to avoid NaNs. The case shouldn't occur in practice
        // because growth never reverses 180° between adjacent SC segments.
        if (cosa > 0.f) return parentRight;
        const dm::float3 helper = (std::abs(parentDir.y) < 0.9f) ? dm::float3(0.f, 1.f, 0.f)
                                                                 : dm::float3(1.f, 0.f, 0.f);
        const dm::float3 fallback = dm::cross(parentDir, helper);
        const float      fl       = dm::length(fallback);
        return (fl > 1e-6f) ? fallback / fl : dm::float3(1.f, 0.f, 0.f);
    }

    const dm::float3 nAxis = axis / sina;
    const float      angle = std::atan2(sina, cosa);
    const dm::quat   q     = dm::rotationQuat(nAxis, angle);
    dm::float3       r     = dm::applyQuat(q, parentRight);

    // Re-orthogonalize against childDir to scrub any drift.
    r = r - childDir * dm::dot(r, childDir);
    const float rl = dm::length(r);
    return (rl > 1e-6f) ? r / rl : r;
}

} // namespace

void SpaceColonizer::grow(const std::vector<dm::float3>& tipPositions,
                          const std::vector<dm::float3>& tipDirs,
                          const dm::box3&                lsystemBbox,
                          const std::vector<dm::float3>& tipRights,
                          const std::vector<float>&      tipRadii,
                          const std::vector<float>&      tipBranchLengths)
{
    m_nodes.clear();
    m_terminals.clear();

    if (tipPositions.empty() || m_p.attractorCount == 0 || m_p.maxIterations == 0)
        return;

    // ---- Crown volume: sphere centered above the L-system bbox.
    const dm::float3 bboxMin    = lsystemBbox.m_mins;
    const dm::float3 bboxMax    = lsystemBbox.m_maxs;
    const dm::float3 bboxCenter = 0.5f * (bboxMin + bboxMax);
    const float      treeHeight = std::max(0.01f, bboxMax.y - bboxMin.y);
    const dm::float3 crownCenter(bboxCenter.x,
                                 bboxMin.y + m_p.crownYOffsetFactor * treeHeight,
                                 bboxCenter.z);
    const float      crownRadius = m_p.crownRadiusFactor * treeHeight;

    // ---- Place attractors uniformly inside the crown sphere.
    std::vector<dm::float3> attractors;
    std::vector<uint8_t>    attractorAlive;
    attractors.reserve(m_p.attractorCount);
    attractorAlive.reserve(m_p.attractorCount);
    for (uint32_t i = 0; i < m_p.attractorCount; ++i) {
        const dm::float3 unit = sampleUnitBall(i, m_p.seed ^ 0xA1F00DEu);
        attractors.push_back(crownCenter + unit * crownRadius);
        attractorAlive.push_back(1);
    }

    // ---- Seed nodes from L-system tips. Inherit pos, forward, right, radius, branchLength
    // so the join with the L-system is seamless (no thickness step, no UV restart, no twist
    // in the connecting quads).
    //
    // Spatially dedup the tips first: L-system grammars (especially stochastic ones with
    // 3-4 generations) produce many tips packed close together, which would all become
    // independent SC roots. Roots that fail to grow remain terminal AT their seed position,
    // so without dedup you get stacks of leaf clusters at every L-system branch end.
    // Dedup radius = segmentLength keeps roots at least one SC-segment apart — visually
    // sufficient and very cheap (O(N²) for N ≈ 100).
    const size_t numTips    = std::min(tipPositions.size(), tipDirs.size());
    const float  dedupR2    = m_p.segmentLength * m_p.segmentLength;
    std::vector<size_t> keptTipIdx;
    keptTipIdx.reserve(numTips);
    for (size_t i = 0; i < numTips; ++i) {
        bool tooClose = false;
        for (size_t k : keptTipIdx) {
            if (dist2(tipPositions[i], tipPositions[k]) < dedupR2) { tooClose = true; break; }
        }
        if (!tooClose) keptTipIdx.push_back(i);
    }

    m_nodes.reserve(keptTipIdx.size() + m_p.maxIterations * keptTipIdx.size());
    for (size_t i : keptTipIdx) {
        SCNode n;
        n.pos      = tipPositions[i];
        const float dl = dm::length(tipDirs[i]);
        n.dir      = (dl > 1e-6f) ? tipDirs[i] / dl : dm::float3(0.f, 1.f, 0.f);

        // Inherit right axis from L-system; re-orthogonalize against forward and unit-norm
        // it to scrub any encoding drift.
        if (i < tipRights.size()) {
            dm::float3 r = tipRights[i] - n.dir * dm::dot(tipRights[i], n.dir);
            const float rl = dm::length(r);
            n.right = (rl > 1e-6f) ? r / rl : dm::float3(1.f, 0.f, 0.f);
        } else {
            const dm::float3 helper = (std::abs(n.dir.y) < 0.9f) ? dm::float3(0.f, 1.f, 0.f)
                                                                 : dm::float3(1.f, 0.f, 0.f);
            const dm::float3 r = dm::cross(n.dir, helper);
            const float      rl = dm::length(r);
            n.right = (rl > 1e-6f) ? r / rl : dm::float3(1.f, 0.f, 0.f);
        }

        n.parent       = -1;
        n.terminal     = false;
        n.radius       = (i < tipRadii.size())         ? tipRadii[i]         : m_p.branchletRadius;
        n.branchLength = (i < tipBranchLengths.size()) ? tipBranchLengths[i] : 0.f;
        m_nodes.push_back(n);
    }

    // ---- Iterative growth.
    const float influence2 = m_p.influenceDistance * m_p.influenceDistance;
    const float kill2      = m_p.killDistance      * m_p.killDistance;

    std::vector<dm::float3> pendingDir(m_nodes.size(), dm::float3(0.f));
    std::vector<int32_t>    pendingCount(m_nodes.size(), 0);

    for (uint32_t iter = 0; iter < m_p.maxIterations; ++iter) {
        std::fill(pendingDir.begin(), pendingDir.end(), dm::float3(0.f));
        std::fill(pendingCount.begin(), pendingCount.end(), 0);

        // For each live attractor, find nearest node within influence distance.
        bool anyAttractorLive = false;
        for (size_t a = 0; a < attractors.size(); ++a) {
            if (!attractorAlive[a]) continue;
            anyAttractorLive = true;

            int32_t bestIdx = -1;
            float   bestD2  = influence2;
            for (size_t n = 0; n < m_nodes.size(); ++n) {
                const float d2 = dist2(m_nodes[n].pos, attractors[a]);
                if (d2 < bestD2) { bestD2 = d2; bestIdx = static_cast<int32_t>(n); }
            }
            if (bestIdx < 0) continue;

            const dm::float3 toA = attractors[a] - m_nodes[bestIdx].pos;
            const float      la  = dm::length(toA);
            if (la > 1e-6f) {
                pendingDir[bestIdx]    += toA / la;
                pendingCount[bestIdx]  += 1;
            }
        }
        if (!anyAttractorLive) break;

        // Spawn children for nodes that received attractor pull this iteration.
        const size_t prevNodeCount = m_nodes.size();
        bool anyGrowth = false;
        for (size_t n = 0; n < prevNodeCount; ++n) {
            if (pendingCount[n] == 0) continue;
            const float dirLen = dm::length(pendingDir[n]);
            if (dirLen < 1e-6f) continue;

            const dm::float3 growDir = pendingDir[n] / dirLen;

            SCNode child;
            child.pos          = m_nodes[n].pos + growDir * m_p.segmentLength;
            child.dir          = growDir;
            // Parallel-transport parent's right onto the child's plane so the ring vertex
            // angles flow continuously down the chain (no roll discontinuity).
            child.right        = parallelTransportRight(m_nodes[n].dir, m_nodes[n].right, growDir);
            child.parent       = static_cast<int32_t>(n);
            child.terminal     = false;
            child.radius       = m_nodes[n].radius * m_p.branchletTaper;
            child.branchLength = m_nodes[n].branchLength + m_p.segmentLength;

            m_nodes.push_back(child);
            anyGrowth = true;
        }

        // Resize pending vectors for any newly added nodes (so next iter has slots).
        pendingDir.resize(m_nodes.size(), dm::float3(0.f));
        pendingCount.resize(m_nodes.size(), 0);

        // Kill attractors near any node.
        for (size_t a = 0; a < attractors.size(); ++a) {
            if (!attractorAlive[a]) continue;
            for (size_t n = 0; n < m_nodes.size(); ++n) {
                if (dist2(m_nodes[n].pos, attractors[a]) < kill2) {
                    attractorAlive[a] = 0;
                    break;
                }
            }
        }

        if (!anyGrowth) break;
    }

    // ---- Mark terminals: nodes with no children.
    std::vector<uint8_t> hasChild(m_nodes.size(), 0);
    for (const auto& n : m_nodes)
        if (n.parent >= 0) hasChild[n.parent] = 1;

    for (size_t i = 0; i < m_nodes.size(); ++i) {
        if (!hasChild[i]) {
            m_nodes[i].terminal = true;
            m_terminals.push_back(static_cast<uint32_t>(i));
        }
    }
}
