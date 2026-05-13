#ifndef XYLEM_SCENE_SUN_SKY_HPP
#define XYLEM_SCENE_SUN_SKY_HPP

#include <array>
#include <cmath>
#include <donut/core/math/math.h>

// sky_cb.h uses HLSL types (float3, float4x4) directly. The Donut convention
// (matched in TraditionalRenderPass.cpp:16) is to bring donut::math symbols
// into unqualified scope before including it. This pollutes the global
// namespace inside any TU that includes SunSky.hpp, which matches the
// codebase's existing pattern for sky_cb.h consumers.
using namespace donut::math;
#include <donut/shaders/sky_cb.h>   // ProceduralSkyShaderParameters

namespace Xylem::Scene {

using donut::math::float3;

struct SkyKeyframe {
    float3                            sunColor;    // RGB; (0,0,0) = pure ambient
    ProceduralSkyShaderParameters     sky;
};

enum SkyKeyframeIdx : uint32_t {
    SK_Dawn      = 0,
    SK_Noon      = 1,
    SK_Dusk      = 2,
    SK_Dark      = 3,
    SK_Moonlight = 4,
    SK_Count     = 5
};

struct SunSky {
    float                                       azimuthDeg                       = 0.f;

    // Sun angular speed (deg/sec) when above the horizon (θ ∈ [0°, 180°]).
    float                                       angularVelocityDegPerSec         = 6.f;

    // Sun angular speed (deg/sec) when below the horizon (θ ∈ [180°, 360°]).
    // Typically faster than the day speed — the underground sweep contributes
    // no directional light, so we can race through it without losing visible
    // detail. With night > day, the cycle "lingers" during daylight and
    // sprints through night.
    float                                       nightAngularVelocityDegPerSec    = 24.f;

    float                                       stateHoldSeconds                 = 2.f;
    bool                                        paused                           = true;
    float                                       phase                            = 0.f;

    // Where the Twilight anchors live, expressed as degrees below horizon.
    // The sun pauses BRIEFLY at these positions to register the post-sunset /
    // pre-dawn dim-warm-purple sky color. With horizon fade also kicking in,
    // sunColor is 0 throughout this stretch — only sky color changes.
    float                                       dawnDuskBelowHorizonDeg          = 15.0f;

    // Smooth-fade band width above the horizon. sunColor is multiplied by a
    // smoothstep that goes from 0 (sun on or below horizon) to 1 (sun
    // `horizonFadeAngleDeg` or more above horizon). Together with the
    // continuous-arc design, this means sunColor is naturally 0 throughout
    // the entire underground sweep — no teleport needed.
    float                                       horizonFadeAngleDeg              = 1.0f;

    std::array<SkyKeyframe, SK_Count>           keyframes{};
};

struct SunSkyState {
    float3                            lightDir       = {0.f, -1.f, 0.f};
    float3                            sunColor       = {1.f, 1.f, 1.f};
    ProceduralSkyShaderParameters     skyParams      = {};
    bool                              shadowsEnabled = true;
};

// ---------------------------------------------------------------------------
// Default keyframe values (used when JSON omits a keyframe or the whole block)
// ---------------------------------------------------------------------------
inline std::array<SkyKeyframe, SK_Count> DefaultKeyframes() {
    std::array<SkyKeyframe, SK_Count> kf{};

    auto fillSky = [](ProceduralSkyShaderParameters& p,
                      float3 light, float3 sky, float3 horizon, float3 ground)
    {
        p.directionToLight   = float3(0, 1, 0);        // overridden per-frame
        p.angularSizeOfLight = donut::math::radians(1.0f);
        p.lightColor         = light;
        p.glowSize           = donut::math::radians(5.0f);
        p.skyColor           = sky;
        p.glowIntensity      = 0.1f;
        p.horizonColor       = horizon;
        p.horizonSize        = donut::math::radians(30.0f);
        p.groundColor        = ground;
        p.glowSharpness      = 4.0f;
        p.directionUp        = float3(0, 1, 0);
        p.pad1               = 0.f;
    };

    kf[SK_Dawn].sunColor = float3(1.0f, 0.6f, 0.3f);
    fillSky(kf[SK_Dawn].sky,
            /*light*/   float3(50.f, 40.f, 25.f),
            /*sky*/     float3(0.15f, 0.20f, 0.30f),
            /*horizon*/ float3(0.90f, 0.50f, 0.30f),
            /*ground*/  float3(0.10f, 0.08f, 0.07f));

    kf[SK_Noon].sunColor = float3(1.0f, 0.97f, 0.9f);
    fillSky(kf[SK_Noon].sky,
            /*light*/   float3(100.f, 98.f, 90.f),
            /*sky*/     float3(0.15f, 0.35f, 0.65f),     // bright cerulean at zenith
            /*horizon*/ float3(0.55f, 0.70f, 0.90f),     // pale blue near horizon (atmospheric scatter)
            /*ground*/  float3(0.10f, 0.10f, 0.10f));    // slight bump for daylight ground response

    kf[SK_Dusk].sunColor = float3(1.0f, 0.4f, 0.2f);
    fillSky(kf[SK_Dusk].sky,
            float3(60.f, 30.f, 15.f),
            float3(0.30f, 0.15f, 0.25f),
            float3(1.00f, 0.40f, 0.20f),
            float3(0.10f, 0.07f, 0.05f));

    kf[SK_Dark].sunColor = float3(0.f, 0.f, 0.f);
    fillSky(kf[SK_Dark].sky,
            float3(0.f, 0.f, 0.f),
            float3(0.005f, 0.010f, 0.020f),
            float3(0.010f, 0.015f, 0.025f),
            float3(0.005f, 0.005f, 0.010f));

    kf[SK_Moonlight].sunColor = float3(0.14f, 0.21f, 0.35f);
    fillSky(kf[SK_Moonlight].sky,
            /*light*/   float3(1.4f, 2.1f, 3.5f),
            /*sky*/     float3(0.014f, 0.028f, 0.056f),
            /*horizon*/ float3(0.035f, 0.056f, 0.105f),
            /*ground*/  float3(0.010f, 0.014f, 0.021f));

    // Make the moon bigger than the sun. Apply to night-phase keyframes
    // (Dark + Moonlight) so the disc size stays consistent across the
    // entire night arc.
    const float kMoonAngularRadius = donut::math::radians(3.0f);
    kf[SK_Dark].sky.angularSizeOfLight      = kMoonAngularRadius;
    kf[SK_Moonlight].sky.angularSizeOfLight = kMoonAngularRadius;

    return kf;
}

// ---------------------------------------------------------------------------
// Cycle math
// ---------------------------------------------------------------------------
//
// Two symmetric phases. Sun is the active light source during the day phase;
// moon during the night phase. Both walk the same θ-parameter range
// [-bH, 180°+bH] through PointOnArc, with three "position" anchors plus a
// fourth "sky-only fade" anchor at the same θ as the third (the active
// source is paused at that position while the sky lerps to the next phase's
// starting color).
//
//   Day phase   (4 anchors)
//     Dawn (-bH) → Noon (90°) → Dusk (180°+bH) → DuskFade (180°+bH, sky→Moonlight)
//
//   Night phase (4 anchors)
//     MoonStart (-bH) → Zenith (90°) → MoonEnd (180°+bH) → DawnFade (180°+bH, sky→Dawn)
//
// During the visible portion of each phase the active source is the only
// visible body; the sun (or moon) walks an above-horizon arc and `sunColor`
// is faded only at horizon by the existing horizon-fade smoothstep.
//
// Moonlight is HELD across the entire moon arc (all three position anchors
// of the night phase use SK_Moonlight, so the lerps between them are
// no-ops). Only the DawnFade segment introduces a color change.
//
// Each phase ends with a sky-only "fade during pause" segment (Δθ = 0):
// position stays at 180°+bH, sky lerps to the NEXT phase's starting
// keyframe over `stateHoldSeconds`. Boundary anchors match across the
// teleport (DuskFade→MoonStart both SK_Moonlight; DawnFade→Dawn both
// SK_Dawn), so the world-position teleport is sky-color-continuous in
// both directions.

inline constexpr float kShadowEpsilonSq = 1e-4f;

inline constexpr int kMaxAnchorsPerPhase = 5;

struct PhaseDescriptor {
    int             numAnchors;
    float           anchorThetaDeg[kMaxAnchorsPerPhase];
    SkyKeyframeIdx  anchorKfIdx[kMaxAnchorsPerPhase];
};

inline PhaseDescriptor GetDayPhase(const SunSky& /*s*/, float bH) {
    PhaseDescriptor d{};
    d.numAnchors = 4;
    d.anchorThetaDeg[0] = -bH;             // Dawn:     sun below east horizon
    d.anchorThetaDeg[1] = 90.f;            // Noon:     zenith
    d.anchorThetaDeg[2] = 180.f + bH;      // Dusk:     sun below west horizon
    d.anchorThetaDeg[3] = 180.f + bH;      // DuskFade: same θ; sky-only lerp to Moonlight
    d.anchorKfIdx[0]    = SK_Dawn;
    d.anchorKfIdx[1]    = SK_Noon;
    d.anchorKfIdx[2]    = SK_Dusk;
    d.anchorKfIdx[3]    = SK_Moonlight;    // Day phase ends already colored as Moonlight,
                                           // so the teleport into night is invisible.
    return d;
}

inline PhaseDescriptor GetNightPhase(const SunSky& /*s*/, float bH) {
    PhaseDescriptor d{};
    d.numAnchors = 4;
    d.anchorThetaDeg[0] = -bH;             // MoonStart: moon below east horizon
    d.anchorThetaDeg[1] = 90.f;            // Zenith:    moon at zenith
    d.anchorThetaDeg[2] = 180.f + bH;      // MoonEnd:   moon below west horizon
    d.anchorThetaDeg[3] = 180.f + bH;      // DawnFade:  same θ; sky-only lerp to Dawn
    d.anchorKfIdx[0]    = SK_Moonlight;    // Moonlight held through the entire moon arc;
    d.anchorKfIdx[1]    = SK_Moonlight;    // sky doesn't change while moon moves.
    d.anchorKfIdx[2]    = SK_Moonlight;
    d.anchorKfIdx[3]    = SK_Dawn;         // Night phase ends already colored as Dawn,
                                           // so the teleport back to day is invisible.
    return d;
}

// Per-segment angular velocity: day speed when the source is on the
// above-horizon arc θ ∈ [0°, 180°]; night speed when underground. Segments
// that straddle a horizon crossing (e.g. `-bH → 90°` crosses θ=0°) are
// split at the crossing so each sub-segment runs at the right speed.
//
// Special case: if Δθ = 0 (same position), this is a sky-only "fade
// during pause" segment — return `stateHoldSeconds` as its duration.
inline float TransitionSecondsForSegment(float thetaA, float thetaB, const SunSky& s) {
    if (thetaA == thetaB) {
        return donut::math::max(s.stateHoldSeconds, 0.f);
    }

    const float dayOmega   = donut::math::max(s.angularVelocityDegPerSec,      0.01f);
    const float nightOmega = donut::math::max(s.nightAngularVelocityDegPerSec, 0.01f);

    auto omegaFor = [&](float a, float b) {
        // Above horizon iff both endpoints (treated inclusively) lie in [0°, 180°].
        const bool above = (a >= 0.f && a <= 180.f) && (b >= 0.f && b <= 180.f);
        return above ? dayOmega : nightOmega;
    };

    // Assumes thetaA < thetaB. Possible horizon crossings: θ=0° and θ=180°.
    float total = 0.f;
    float curr  = thetaA;
    if (curr < 0.f && thetaB > 0.f) {
        total += (0.f - curr) / omegaFor(curr, 0.f);
        curr = 0.f;
    }
    if (curr < 180.f && thetaB > 180.f) {
        total += (180.f - curr) / omegaFor(curr, 180.f);
        curr = 180.f;
    }
    total += (thetaB - curr) / omegaFor(curr, thetaB);
    return total;
}

inline float PhaseSecondsOf(const PhaseDescriptor& p, float H, const SunSky& s) {
    float total = float(p.numAnchors) * H;
    for (int i = 0; i < p.numAnchors - 1; ++i) {
        total += TransitionSecondsForSegment(p.anchorThetaDeg[i], p.anchorThetaDeg[i + 1], s);
    }
    return total;
}

inline float TotalCycleSeconds(const SunSky& s) {
    const float H  = donut::math::max(s.stateHoldSeconds, 0.f);
    const float bH = donut::math::max(s.dawnDuskBelowHorizonDeg, 0.f);
    return PhaseSecondsOf(GetDayPhase(s, bH), H, s)
         + PhaseSecondsOf(GetNightPhase(s, bH), H, s);
}

// Phase at the start of the Noon hold (sun at zenith) for legacy-scene seeding.
inline float NoonArrivalPhase(const SunSky& s) {
    const float cycle = TotalCycleSeconds(s);
    if (cycle <= 0.f) return 0.f;
    const float H  = donut::math::max(s.stateHoldSeconds, 0.f);
    const float bH = donut::math::max(s.dawnDuskBelowHorizonDeg, 0.f);
    const PhaseDescriptor day = GetDayPhase(s, bH);
    // Noon is anchor 1 in day phase. Time to reach: 1 hold (Dawn) + 1 transition (Dawn→Noon).
    const float t = H + TransitionSecondsForSegment(day.anchorThetaDeg[0], day.anchorThetaDeg[1], s);
    return t / cycle;
}

// Compute unit-direction vector at arc angle θ (degrees) on the great circle.
// The great circle plane contains world +Y (zenith) and the horizontal axis
// u_horizon = (cos(az), 0, sin(az)). At θ=0 the result is u_horizon
// (east-horizon); at θ=90° it is +Y (zenith); at θ=180° it is -u_horizon
// (west-horizon); at θ=270° it is -Y (nadir).
inline float3 PointOnArc(float thetaDeg, float azimuthDeg) {
    const float t  = donut::math::radians(thetaDeg);
    const float a  = donut::math::radians(azimuthDeg);
    const float ct = std::cos(t);
    const float st = std::sin(t);
    return float3(ct * std::cos(a), st, ct * std::sin(a));
}

// Component-wise linear blend of every ProceduralSkyShaderParameters field
// (except directionToLight, which is set by the per-frame state from lightDir).
inline ProceduralSkyShaderParameters LerpSky(
    const ProceduralSkyShaderParameters& a,
    const ProceduralSkyShaderParameters& b,
    float u)
{
    ProceduralSkyShaderParameters r{};
    const float inv = 1.f - u;
    r.directionToLight   = float3(0,1,0);  // overridden by caller
    r.angularSizeOfLight = a.angularSizeOfLight * inv + b.angularSizeOfLight * u;
    r.lightColor         = a.lightColor         * inv + b.lightColor         * u;
    r.glowSize           = a.glowSize           * inv + b.glowSize           * u;
    r.skyColor           = a.skyColor           * inv + b.skyColor           * u;
    r.glowIntensity      = a.glowIntensity      * inv + b.glowIntensity      * u;
    r.horizonColor       = a.horizonColor       * inv + b.horizonColor       * u;
    r.horizonSize        = a.horizonSize        * inv + b.horizonSize        * u;
    r.groundColor        = a.groundColor        * inv + b.groundColor        * u;
    r.glowSharpness      = a.glowSharpness      * inv + b.glowSharpness      * u;
    r.directionUp        = a.directionUp        * inv + b.directionUp        * u;
    r.pad1               = 0.f;
    return r;
}

// Resolve current cycle position to a per-frame state.
//
// Two-phase cycle. Day phase walks the sun through 5 anchors; night phase
// walks the moon through 4 anchors. Moonlight is HELD across the moon's
// entire above-horizon arc (anchors 1 and 2 of the night phase both use
// SK_Moonlight). At each phase boundary, the active source's world
// position teleports — both endpoints are below horizon by `bH`, so the
// disc is invisible on both sides and the teleport is not visually
// observable. Sky params are SK_Twilight on both sides of the boundary,
// so they lerp continuously through the teleport.
inline SunSkyState ResolveSunSky(const SunSky& s)
{
    SunSkyState out{};

    const float H  = donut::math::max(s.stateHoldSeconds, 0.f);
    const float bH = donut::math::max(s.dawnDuskBelowHorizonDeg, 0.f);
    const PhaseDescriptor day   = GetDayPhase(s, bH);
    const PhaseDescriptor night = GetNightPhase(s, bH);
    const float daySec   = PhaseSecondsOf(day,   H, s);
    const float nightSec = PhaseSecondsOf(night, H, s);
    const float cycle    = daySec + nightSec;

    auto fillSkyParamsAndShadow = [&](float3 sourcePos) {
        out.lightDir = -sourcePos;
        out.skyParams.directionToLight = -out.lightDir;

        const float fadeBound = std::sin(donut::math::radians(
            donut::math::max(s.horizonFadeAngleDeg, 1e-3f)));
        const float u = donut::math::clamp(sourcePos.y / fadeBound, 0.f, 1.f);
        const float fade = u * u * (3.f - 2.f * u);
        out.sunColor *= fade;

        out.shadowsEnabled = donut::math::dot(out.sunColor, out.sunColor) > kShadowEpsilonSq;
    };

    if (cycle <= 0.f) {
        out.sunColor  = s.keyframes[SK_Noon].sunColor;
        out.skyParams = s.keyframes[SK_Noon].sky;
        fillSkyParamsAndShadow(PointOnArc(90.f, s.azimuthDeg));
        return out;
    }

    const float phaseWrapped = s.phase - std::floor(s.phase);
    const float t            = phaseWrapped * cycle;

    const bool             isNightPhase = (t >= daySec);
    const float            tInPhase     = t - (isNightPhase ? daySec : 0.f);
    const PhaseDescriptor& p            = isNightPhase ? night : day;

    SkyKeyframe kfFrom  = s.keyframes[p.anchorKfIdx[0]];
    SkyKeyframe kfTo    = kfFrom;
    float       thetaDeg = p.anchorThetaDeg[0];
    float       blendU  = 0.f;

    float cursor = 0.f;
    for (int i = 0; i < p.numAnchors; ++i) {
        const float holdEnd = cursor + H;
        if (tInPhase < holdEnd) {
            kfFrom = s.keyframes[p.anchorKfIdx[i]];
            kfTo   = kfFrom;
            thetaDeg = p.anchorThetaDeg[i];
            blendU = 0.f;
            break;
        }
        cursor = holdEnd;

        if (i < p.numAnchors - 1) {
            const float T = TransitionSecondsForSegment(p.anchorThetaDeg[i], p.anchorThetaDeg[i + 1], s);
            const float transEnd = cursor + T;
            if (tInPhase < transEnd) {
                kfFrom = s.keyframes[p.anchorKfIdx[i]];
                kfTo   = s.keyframes[p.anchorKfIdx[i + 1]];
                blendU = (T > 0.f) ? (tInPhase - cursor) / T : 0.f;
                thetaDeg = p.anchorThetaDeg[i] + (p.anchorThetaDeg[i + 1] - p.anchorThetaDeg[i]) * blendU;
                break;
            }
            cursor = transEnd;
        }
    }

    out.sunColor  = kfFrom.sunColor * (1.f - blendU) + kfTo.sunColor * blendU;
    out.skyParams = LerpSky(kfFrom.sky, kfTo.sky, blendU);

    fillSkyParamsAndShadow(PointOnArc(thetaDeg, s.azimuthDeg));
    return out;
}

} // namespace Xylem::Scene

#endif // XYLEM_SCENE_SUN_SKY_HPP
