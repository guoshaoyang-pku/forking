#!/usr/bin/env python3
"""Build the static project page and guides from pinned assets and Markdown."""
import html
import json
import posixpath
import re
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPO = "https://github.com/guoshaoyang-pku/forking"
REPO_TREE = REPO + "/tree/release"
FIGURES = {f["id"]: f for f in json.loads((DOCS / "figs/manifest.json").read_text())["figures"]}


def head(title, description):
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · Forking</title><meta name="description" content="{html.escape(description)}">
<meta name="theme-color" content="#152c42"><meta property="og:title" content="Forking: Sudden Overfitting Under Replay">
<meta property="og:description" content="When the same text returns, fitting and generalization can take different paths.">
<meta property="og:type" content="website"><meta property="og:image" content="https://guoshaoyang-pku.github.io/forking/site-assets/share-card.png">
<meta property="og:image:width" content="1200"><meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image"><meta name="twitter:image" content="https://guoshaoyang-pku.github.io/forking/site-assets/share-card.png">
<link rel="icon" href="site-assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="site-assets/site.css"></head><body>
<a class="skip" href="#main">Skip to content</a><header class="topbar"><nav class="wrap nav" aria-label="Main navigation">
<a class="brand" href="index.html"><svg class="mark" viewBox="0 0 30 30" aria-hidden="true"><path d="M3 25L12 16L26 4" fill="none" stroke="#196693" stroke-width="3"/><path d="M12 16L26 22" fill="none" stroke="#c65b2e" stroke-width="3"/></svg>Forking</a>
<div class="nav-links"><a href="index.html#findings">Findings</a><a href="REVIEW_GUIDE.html" class="optional">Review</a><a href="index.html#reproduce">Reproduce</a><a href="{REPO_TREE}">GitHub ↗</a></div></nav></header>'''


def footer():
    return f'''<footer class="wrap footer"><div>Forking · Sudden Overfitting Under Replay</div><div><a href="EVIDENCE.html">Evidence notes</a><a href="AVAILABILITY.html">Materials</a><a href="{REPO_TREE}">Code ↗</a></div></footer></body></html>'''


def figure(slug, title, caption, alt, hero=False):
    f = FIGURES[slug]
    image_href = f"figs/{slug}.pdf" if f.get("pdf") else f"figs/{slug}.png"
    image_label = "Vector PDF ↗" if f.get("pdf") else "PNG ↗"
    return f'''<figure class="{'hero-figure' if hero else 'panel'}" id="figure-{slug}"><a class="image-link" href="{image_href}" aria-label="Open {html.escape(title)}"><img src="figs/{slug}.png" width="{f['width']}" height="{f['height']}" {'fetchpriority="high"' if hero else 'loading="lazy"'} alt="{html.escape(alt)}"></a>
<figcaption><strong>{title}</strong> {caption}</figcaption><div class="source-line"><a href="{image_href}">{image_label}</a><a href="FIGURES.html#{slug}">Source &amp; scope ↗</a></div></figure>'''


def build_home():
    page = head("Sudden Overfitting Under Replay", "Controlled experiments on memory, repeated data, and continuation coverage in language models.")
    page += """<main id="main" class="wrap"><section class="hero" aria-labelledby="title">
<h1 id="title">Forking</h1><p class="subtitle">Sudden Overfitting Under Replay</p>
<p class="lede">In language models with local n-gram memory, replay can drive training loss down while held-out loss rises. We study when this split appears, which predictions are hurt, and how it relates to continuation coverage.</p>
<div class="actions"><a class="button primary" href="#reproduce">Reproduce</a><a class="button" href="REVIEW_GUIDE.html">Evidence &amp; code</a><a class="button" href="FIGURES.html">Paper figures</a></div>"""
    page += figure("overview", "Forking under repeated exposure.", "The first-pass panel is the initial segment of a replay run. The representation diagram is schematic.", "Local n-gram memory and the divergence between training and held-out loss under replay.", True)
    page += """</section><section id="findings" class="section"><h2>What we find</h2>
<ol class="findings"><li><strong>The divergence follows replay.</strong> Separation grows as training revisits the finite corpus.</li><li><strong>A seen context can have an unseen continuation.</strong> Improved fitting of observed continuations can coexist with worse predictions for plausible alternatives.</li><li><strong>Continuation coverage organizes the gap.</strong> Three-seed frequency-bin measurements support an approximate relationship between novelty and the online gap.</li></ol>
<p>These findings motivate evaluating memory-augmented models across repeated passes, beyond their initial improvement in fitting.</p></section>
<section class="section"><h2>1. The timing follows repeated exposure</h2><p>Short-run traces show structure aligned with replay boundaries; longer runs follow the gap across 20 passes.</p>"""
    page += figure("replay", "Replay trajectories.", "Finite-budget measurements with a no-gram control. Online gap is fixed-validation CE after the update minus current-batch training CE before the same logged update; losses are in nats.", "Gap trajectories over 2000 updates and 20 passes.")
    page += """</section><section id="continuations" class="section"><h2>2. Inspect the next-token predictions</h2><p>These context probes show how probability changes between the first and fifth pass, comparing the no-gram and local-memory models.</p>"""
    page += figure("context-examples", "Three contexts, two models.", "Open points: epoch 1. Filled points: epoch 5. Higher means more probable. Colors describe corpus membership. Seed 42; steps 337 and 1685. Lines connect endpoints and do not imply monotone intermediate change.", "Measured log probabilities for three context probes at epochs one and five.")
    page += """<p class="reading-note">The input is BOS plus exactly three context tokens. “Others” is a separately reranked top-20 mean probability. See <a href="EVIDENCE.html">measurement details</a> and the <a href="FIGURES.html#continuations">aggregate continuation diagnostic</a>.</p>
<h3>Sharpening and recovery</h3><p>The mechanism schematic labels the replay effect <strong>Sharpening</strong> and the contrasting recovery path <strong>Only no-gram can do</strong>. Its bar heights are illustrative. The measured companion below separates output shape from true-token damage.</p>
<div class="two-up"><figure class="panel"><a class="image-link" href="figs/mechanism.pdf"><img src="figs/mechanism.png" alt="Schematic of sharpening and recovery, including Only no-gram can do."></a><figcaption><strong>Mechanism schematic.</strong> Explanatory, not measured bar heights.</figcaption></figure><figure class="panel"><a class="image-link" href="figs/sharpening-vs-ce.png"><img src="figs/sharpening-vs-ce.png" alt="Measured replay sharpening and validation cross-entropy diagnostic."></a><figcaption><strong>Measured companion.</strong> Twenty replay epochs compare entropy, probability margin, and validation cross-entropy for local memory and the matched no-gram control. <a href="SHARPENING.html">Scope and packet</a>.</figcaption></figure></div></section>
<section class="section"><h2>3. Relate the gap to continuation coverage</h2><p>The fraction of continuations not observed in training provides a useful description of the gap across context-frequency bins.</p>"""
    page += figure("coverage", "An independently reproducible calculation.", "Equal-bin affine fits at steps 670, 1000, 1340, and 2000. Means and sample SD use seeds 42, 43, and 44. This is a descriptive relationship within the measured setting.", "Training-count novelty estimates and measured bigram and trigram coverage-gap relationships.")
    page += """<p class="reading-note">Additional paper figures cover <a href="FIGURES.html#frequency">frequency</a>, <a href="FIGURES.html#capacity">table capacity</a>, <a href="FIGURES.html#corpus">corpus size</a>, and <a href="FIGURES.html#interventions">retrieval masking and freezing</a>. Their sources and scope are recorded individually.</p></section>
<section id="reproduce" class="section"><h2>Reproduce the analysis</h2><p>Clone the release branch, then run two standard-library checks:</p><pre><code>git clone --branch release --single-branch https://github.com/guoshaoyang-pku/forking.git
cd forking
python3 reproduce/verify_coverage.py
python3 reproduce/verify_contexts.py</code></pre><p>Python 3.10 or newer. No packages, GPU, or model download are needed for these checks. They recompute eight coverage fits and check 72 context endpoints from the bundled observations.</p>
<p><a href="REPRODUCE.html">Analysis and replotting</a> · <a href="TRAINING.html">Baseline and no-gram training</a> · <a href="REVIEW_GUIDE.html">Code map</a></p>
<p class="reading-note">The same fixed context-to-row rule is used in training and validation. The results concern learned behavior under replay; they do not establish independent context subspaces or equivalence to every sparse architecture.</p>
<p class="reading-note">Full training requires external tokenized data and the frequency index. Final manuscript/citation, tokenizer acquisition details, and the license remain to be attached. See <a href="AVAILABILITY.html">material availability</a>.</p></section></main>"""
    (DOCS / "index.html").write_text(page + footer())


def build_guides():
    sources = {"REPRODUCE": ROOT / "reproduce/README.md"}
    sources.update({name: DOCS / f"{name}.md" for name in ("REVIEW_GUIDE", "EVIDENCE", "AVAILABILITY", "TRAINING", "ANNOUNCEMENT", "SHARPENING")})
    for name, source in sources.items():
        if not source.exists():
            raise SystemExit(f"Missing guide: {source.name}")
        body = markdown.markdown(source.read_text(), extensions=["tables", "fenced_code", "toc"])
        def link(match):
            target = match.group(1)
            if target.startswith(("http:", "https:", "mailto:", "#")):
                return match.group(0)
            path, sep, fragment = target.partition("#")
            resolved = (source.parent / path).resolve()
            if resolved == ROOT / "reproduce/README.md":
                new = "REPRODUCE.html"
            elif resolved.is_relative_to(DOCS):
                new = resolved.relative_to(DOCS).as_posix()
                if new.endswith(".md"):
                    new = new[:-3] + ".html"
            else:
                new = REPO + "/blob/release/" + resolved.relative_to(ROOT).as_posix()
            return 'href="' + new + (sep + fragment if sep else "") + '"'
        body = re.sub(r'href="([^"]+)"', link, body)
        title = source.read_text().splitlines()[0].removeprefix("# ")
        (DOCS / f"{name}.html").write_text(head(title, title) + '<main id="main" class="wrap document">' + body + '</main>' + footer())
    body = '<main id="main" class="wrap document"><p class="eyebrow">Paper figures</p><h1>Figures, sources, and scope</h1><p>Exact vector assets from the manuscript, with individually pinned hashes. The context examples are measured endpoints; the mechanism illustration is schematic.</p>'
    for slug, f in FIGURES.items():
        asset_links = (f'<a href="figs/{slug}.pdf">Vector PDF</a> · ' if f.get("pdf") else '') + f'<a href="figs/{slug}.png">PNG image</a>'
        source_hash = f.get("source_sha256", "not applicable; packaged PNG is pinned below")
        body += f'<section id="{slug}"><h2>{html.escape(slug.replace("-", " ").title())}</h2><p>{html.escape(f["scope"])}</p><p>{asset_links}</p><p>Source: <code>{html.escape(f["source"])}</code><br>Source SHA-256: <code>{source_hash}</code><br>Packaged PNG SHA-256: <code>{f["png_sha256"]}</code></p></section>'
    body += '<p>Machine-readable provenance: <a href="figs/manifest.json">figure manifest</a>.</p></main>'
    (DOCS / "FIGURES.html").write_text(head("Figures and provenance", "Original manuscript figures and source identities") + body + footer())


if __name__ == "__main__":
    build_home()
    build_guides()
    print("Built project page and eight concise reader guides")
