"""Shared colour palettes for benchmark plots.

PIPELINE_COLORS and STAGE_COLORS are the canonical maps every plot should
use so the same pipeline keeps the same colour across all figures.
"""
from __future__ import annotations

import colorsys


PIPELINE_COLORS: dict[str, str] = {
    "Traditional": "#4C72B0",  # blue
    "Compute":     "#DD8452",  # orange
    "MeshShader":  "#55A467",  # green
}


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"expected 6-digit hex colour, got {hex_color!r}")
    return (int(h[0:2], 16) / 255.0,
            int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0)


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = (max(0.0, min(1.0, c)) for c in rgb)
    return f"#{int(round(r * 255)):02X}{int(round(g * 255)):02X}{int(round(b * 255)):02X}"


def variant_color(base_hex: str, variant_idx: int) -> str:
    """Return a shade of `base_hex` for the N-th run of the same pipeline.

    Variant 0 returns the base unchanged; subsequent variants are progressively
    darkened in HLS space so the eye still groups them as 'a kind of orange'
    while reading them as distinct series.
    """
    if variant_idx <= 0:
        return base_hex
    r, g, b = _hex_to_rgb(base_hex)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    new_l = max(0.12, l - 0.20 * variant_idx)
    # Bump saturation slightly so the darker shades don't read as muddy.
    new_s = min(1.0, s + 0.05 * variant_idx)
    return _rgb_to_hex(colorsys.hls_to_rgb(h, new_l, new_s))


# Order matches the per-frame CSV column order: depthPrepass, hizMs, sdsmMs,
# cullMs, shadowMs, skyMs, sceneMs. Picked from a colour-blind-safe sequence.
STAGE_COLORS: dict[str, str] = {
    "depthPrepassMs": "#4C72B0",
    "hizMs":          "#DD8452",
    "sdsmMs":         "#55A467",
    "cullMs":         "#C44E52",
    "shadowMs":       "#8172B2",
    "skyMs":          "#937860",
    "sceneMs":        "#DA8BC3",
}
