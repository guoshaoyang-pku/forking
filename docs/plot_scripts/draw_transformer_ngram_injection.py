#!/usr/bin/env python3
"""Draw the approved Candidate-D overview of n-gram injection routes."""

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figs" / "main" / "fig_transformer_ngram_injection_routes.svg"
W, H = 976, 700


def text(x, y, value, cls="body", anchor="middle"):
    return (
        f'<text class="tni {cls}" x="{x}" y="{y}" '
        f'text-anchor="{anchor}">{escape(value)}</text>'
    )


def rect(x, y, w, h, cls="box", rx=10):
    return f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/>'


def line(x1, y1, x2, y2, cls="flow", marker="url(#tni-a)"):
    return (
        f'<path class="{cls}" d="M{x1} {y1}L{x2} {y2}" '
        f'marker-end="{marker}"/>'
    )


def plus(x, y):
    return (
        f'<circle class="plus" cx="{x}" cy="{y}" r="14"/>'
        f'<path class="plus-mark" d="M{x-7} {y}h14M{x} {y-7}v14"/>'
    )


def route(y, title, mode):
    return (
        rect(34, y, 304, 72, "route-box")
        + text(186, y + 31, title, "route-title")
        + text(186, y + 55, mode, "route-sub")
    )


def render():
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="176mm" height="126.23mm" '
        f'viewBox="0 0 {W} {H}" role="img">',
        "<title>Three alternative n-gram injection routes in a Transformer block</title>",
        "<desc>Token and position input remains outside the block. "
        "The input route enters before QKV and may be gated or ungated; "
        "V and y routes are alternative additive sites.</desc>",
        """<defs>
<marker id="tni-a" viewBox="0 0 10 10" refX="10" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0 0L10 5L0 10z" fill="#333"/></marker>
<marker id="tni-g" viewBox="0 0 10 10" refX="10" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto"><path d="M0 0L10 5L0 10z" fill="#1B6B5F"/></marker>
<style>
.tni{font-family:Arial,Helvetica,sans-serif;fill:#1F1F1F}
.body{font-size:18px;font-weight:700}
.sub{font-size:15px;font-weight:400;fill:#6B6B6B}
.route-title{font-size:18px;font-weight:700;fill:#1B6B5F}
.route-sub{font-size:15px;font-weight:400;fill:#1B6B5F}
.label{font-size:16px;font-weight:400;fill:#1B6B5F}
.legend-title{font-size:15px;font-weight:700;fill:#6B6B6B;letter-spacing:1px}
.legend-text{font-size:15px;font-weight:400;fill:#333}
.box{fill:#FFFFFF;stroke:#333;stroke-width:2}
.block{fill:none;stroke:#C5C0B6;stroke-width:2}
.route-box{fill:#DDECE7;stroke:#1B6B5F;stroke-width:2.5}
.flow{fill:none;stroke:#333;stroke-width:2}
.inject{fill:none;stroke:#1B6B5F;stroke-width:2;stroke-dasharray:9 7}
.plus{fill:#FFF;stroke:#1B6B5F;stroke-width:2}
.plus-mark{stroke:#1B6B5F;stroke-width:2;stroke-linecap:round}
.legend{fill:#FCFBF8;stroke:#DDD8CF;stroke-width:1.5}
</style></defs>""",
        '<rect width="100%" height="100%" fill="#FFF"/>',
        rect(620, 24, 280, 78),
        text(760, 57, "Token + position", "body"),
        text(760, 84, "representation xℓ", "sub"),
        line(760, 102, 760, 213),
        rect(506, 170, 470, 400, "block"),
        text(530, 205, "Transformer block ℓ", "body", "start"),
        rect(620, 245, 280, 58),
        text(760, 281, "Q, K, V projection"),
        line(760, 303, 760, 346),
        rect(594, 346, 332, 58),
        text(760, 382, "Causal self-attention"),
        line(760, 404, 760, 447),
        rect(620, 447, 280, 58),
        text(760, 483, "MLP"),
        line(760, 505, 760, 625),
        rect(620, 625, 280, 58),
        text(760, 661, "Next Transformer block"),
        route(195, "(a) input route", "gated / ungated"),
        route(299, "(b) V route", "gated"),
        route(403, "(c) y route", "gated"),
        text(450, 220, "before QKV", "label"),
        text(450, 324, "before attention", "label"),
        text(450, 428, "after attention", "label"),
        '<path class="inject" d="M338 231H746" marker-end="url(#tni-g)"/>',
        '<path class="inject" d="M338 335H746" marker-end="url(#tni-g)"/>',
        '<path class="inject" d="M338 439H746" marker-end="url(#tni-g)"/>',
        plus(760, 231),
        plus(760, 335),
        plus(760, 439),
        rect(34, 520, 304, 130, "legend", 9),
        text(56, 549, "LEGEND", "legend-title", "start"),
        '<path class="inject" d="M56 579H112" marker-end="url(#tni-g)"/>',
        text(132, 584, "n-gram injection", "legend-text", "start"),
        plus(78, 614),
        text(132, 619, "additive injection point", "legend-text", "start"),
        text(56, 643, "one route enabled per run", "sub", "start"),
        "</svg>",
    ]
    return "".join(parts)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render() + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()