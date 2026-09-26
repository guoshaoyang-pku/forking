# Discovery path prototype

This prototype visualizes the paper's discovery sequence:

**Forking** -> **n-gram memory** -> **novel continuations** -> **local sharpening**

The arrows carry the operations that led from one observation to the next:
OPHIS screening plus table ablation, frequency analysis, and replay dynamics.
The final node is marked **Proposed mechanism** because the full arrow chain is a
research narrative; the local-sharpening interpretation is supported by the
interventions and diagnostics described later in the paper.

The mini-plots are schematic and carry no new measurements. The displayed
bigram exponents (0.25 for the online gap and 0.23 for directly measured
novelty) are the same matched-bin values reported in Section 4 and Appendix E.

Source: docs/plot_scripts/discovery_story_prototype.tex.

Render from the repository root with pdflatex, then convert the PDF with
pdftocairo to SVG and PNG.
