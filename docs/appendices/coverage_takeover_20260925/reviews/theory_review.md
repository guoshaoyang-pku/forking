The derivation is complete: under Poissonized, matched-distribution sampling, the shifted ratio of expectations equals the validation-token-weighted novelty probability. The empirical ratio has no general exact-unbiasedness guarantee. I also found that, for the same stored per-position data, the affine fit’s intercept is negative while every measured seen-continuation gap \(S(F)\) is positive. A close affine fit therefore does not establish constant \(D\) and \(S\).The shifted Good–Turing identity is correct **under explicit Poissonized, matched-distribution assumptions**, but the attachment overstates what follows from it. The empirical ratio is not generally exactly unbiased; the reported strata do not hold frequency fixed; and the affine fits support a descriptive relationship, not an independent causal validation of coverage.

I read `agents.md` and the specified sources, and recomputed key statistics from the saved summaries. No files were modified, no training or remote jobs were launched, and no subagents were used.

The correct identity is as follows.

Let context \(c\) have population probability \(q_c\), with conditional continuation probabilities \(p_{cy}\). Model the training sample by
\[
K_c\sim\operatorname{Poisson}(\lambda_c),
\qquad \lambda_c=\tau q_c,
\]
with continuations conditionally i.i.d. given the context. An independent validation token must follow the **same** \(q_c,p_{cy}\) distribution.

For a particular context, the expected missing continuation mass after \(F\) observations is
\[
m_c(F)=\sum_y p_{cy}(1-p_{cy})^F.
\]
If \(S_c(k)\) counts singleton continuation types in \(k\) draws, then
\[
\mathbb E[S_c(F+1)\mid K_c=F+1]
=(F+1)m_c(F).
\]
This is the elementary leave-one-out identity: a token is a singleton precisely when its continuation is absent from the other \(F\) draws.

Define
\[
n(k)=\sum_c\mathbf1\{K_c=k\},
\qquad
N_1(k)=\sum_c\mathbf1\{K_c=k\}S_c(k),
\]
and write \(\pi_F(\lambda)=e^{-\lambda}\lambda^F/F!\). The relevant population target is
\[
f_{\mathrm{tok}}^\star(F)
=
\Pr(\text{validation continuation unseen}\mid K_C=F)
=
\frac{\sum_c q_c\pi_F(\lambda_c)m_c(F)}
     {\sum_c q_c\pi_F(\lambda_c)}.
\]
Using \((F+1)\pi_{F+1}(\lambda_c)=\lambda_c\pi_F(\lambda_c)\),
\[
\boxed{
\frac{\mathbb E[N_1(F+1)]}
     {(F+1)\mathbb E[n(F+1)]}
=
\frac{\sum_c\lambda_c\pi_F(\lambda_c)m_c(F)}
     {\sum_c\lambda_c\pi_F(\lambda_c)}
=
f_{\mathrm{tok}}^\star(F).
}
\]

The attachment’s proposed Appendix E derivation therefore has the right central step. Its main-text statement needs the assumptions and target above; conditional independence of continuations alone is insufficient. See the [attachment, proposed identity and derivation](</Users/guoshaoyang/.codex-phybench/attachments/df62704d-0cda-495d-9013-da676205f4cc/Pasted text.txt:69>).

Three distinctions are essential:

- **Token weighting versus context weighting.** Uniformly averaging contexts selected by \(K_c=F\) gives weights proportional to \(\pi_F(\lambda_c)\). A validation token selects them with weights proportional to \(q_c\pi_F(\lambda_c)\). The shift supplies precisely this additional size weighting under Poissonization. These targets need not agree.
- **Ratio of expectations versus expectation of a ratio.** The identity does not generally establish
  \[
  \mathbb E\!\left[\frac{N_1(F+1)}{(F+1)n(F+1)}\right]
  =f_{\mathrm{tok}}^\star(F).
  \]
  The denominator is random and can vanish. An empirical ratio requires sufficient concentration or an additional argument; it is not automatically exactly unbiased.
- **Population target versus the realized corpus.** For a fixed training corpus \(T\), its conditional held-out novelty probability is
  \[
  f_T(F)=
  \frac{\sum_c q_c\mathbf1\{K_c(T)=F\}
        \sum_y p_{cy}\mathbf1\{N_{cy}(T)=0\}}
       {\sum_c q_c\mathbf1\{K_c(T)=F\}}.
  \]
  The boxed identity is not an equality to every realized \(f_T(F)\), nor generally to \(\mathbb E_T[f_T(F)]\). It averages numerator and denominator before division.

Poissonization also matters for the exact finite-sample claim. With exactly \(N\) i.i.d. context–continuation pairs,
\[
(F+1)\Pr\{\operatorname{Bin}(N,q_c)=F+1\}
=Nq_c\Pr\{\operatorname{Bin}(N-1,q_c)=F\}.
\]
Consequently, the corresponding ratio of expectations computed from an \(N\)-pair sample targets the validation-token novelty probability for an **\(N-1\)-pair training sample**. Treating it as the \(N\)-sample target is an approximation. Overlapping text windows and document dependence introduce further departures from the idealized sampling model.

The substantive audit findings are:

1. **“Computed from training counts alone” needs qualification.**

   The raw exact-layer statistic is train-only. The displayed agreement is not wholly train-only:

   - Frequency-band estimates are averaged using **observed validation-token counts** in [analyze_prop.py](/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/docs/appendices/gt_audit_20260925/scripts/analyze_prop.py:11).
   - The figure estimates \(k=\operatorname{median}(\widehat f/f_{\mathrm{val}})\) on the held-out data and plots \(\widehat f/k\), in [draft_figs2.py](/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/docs/appendices/gt_audit_20260925/scripts/draft_figs2.py:15).

   I reproduced the uncalibrated bandwise ratios over the plotted \(F\lesssim10^3\) window: approximately **0.870–0.885 for bigrams** and **0.857–0.928 for trigrams**. This supports a close shape correspondence with systematic underestimation, followed by empirical calibration.

   A train-only band statistic consistent with the population identity would instead be
   \[
   \widehat f_{\mathrm{GT}}(B)=
   \frac{\sum_{F\in B}N_1(F+1)}
        {\sum_{F\in B}(F+1)n(F+1)}.
   \]
   Again, its ratio of expectations has the exact population interpretation; its realized ratio needs an estimation argument.

   **Defensible wording:** “A shifted singleton-count estimator computed from the training corpus closely tracks the frequency dependence of held-out novelty. On this corpus, its uncalibrated values systematically underestimate the observed fractions.”

2. **The within-shard check does not identify document repetition as the cause of the calibration factor.**

   The implementation splits 2049-token chunks by parity or contiguous halves; it does not manipulate or explicitly analyze document boundaries. I reproduced median ratios of **0.869** and **0.870**. See [split_check.py](/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/docs/appendices/gt_audit_20260925/scripts/split_check.py:11).

   This shows that the discrepancy also occurs within the training shard. It weakens an explanation requiring a difference specifically between the original training and validation shards. It does not distinguish document repetition from other dependencies, topic heterogeneity, nonstationarity, or estimator effects.

   **Defensible wording:** “A similar discrepancy appears in within-shard splits, suggesting that differences between the original training and validation shards are not its sole explanation.”

3. **The highlighted strata retain substantial frequency confounding.**

   The analysis groups \(F\) into bands including 9–38, 39–220, and 221–1244. Calling these “fixed \(F\)” is inaccurate. See [analyze_strata.py](/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/docs/appendices/gt_audit_20260925/scripts/analyze_strata.py:24).

   For the both-table model, seed 42, step 1000, I reproduced the highlighted bigram contrast from [perpos_decomp.json](/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/docs/appendices/gt_audit_20260925/perpos_decomp.json):

   | \(F=9\)–38 stratum | Novel fraction \(f\) | Gap | \(D\) | \(S\) |
   |---|---:|---:|---:|---:|
   | \(N_{1,c}/F_c<0.25\) | 0.182 | 2.199 | 8.302 | 0.839 |
   | \(N_{1,c}/F_c\ge0.75\) | 0.764 | 4.783 | 5.873 | 1.262 |

   But **38.2%** of the high-singleton stratum’s validation tokens have \(F=9\)–12, compared with **18.6%** in the low-singleton stratum. At \(F=27\)–38, the shares are **13.6%** and **32.7%**, respectively. The frequency distributions are visibly unmatched in the direction relevant to the proposed alternative.

   The stored cells also contain useful exact-count evidence: at **\(F=3\)**, the same extreme strata have \(f=0.286\) versus \(0.804\), and gaps of \(3.572\) versus \(6.189\) nats. Thus the association is not restricted to broad bands. This can challenge a strict model in which the expected gap depends only on exact \(F\), but it does not establish coverage as the causal variable.

   **Defensible wording:** “Within broad frequency bands—and in some available exact-count comparisons—higher training singleton fractions are associated with higher held-out novelty and larger gaps.”

4. **Neither coarse bands nor exact-\(F\) comparisons exclude optimizer explanations.**

   Even at exact \(F\), singleton fraction changes with continuation diversity, gradient directions and cancellation, conditional entropy, and the ease of fitting the training targets. Equal occurrence counts therefore do not imply equal optimizer states or effective updates. Hash collisions and other active context paths introduce additional differences.

   Coverage and optimization are also compatible mechanisms: one can determine which continuations lack evidence while the other determines how strongly their probabilities are suppressed. The repository’s own [claims ledger, C4](/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/docs/claims-ledger.md:10) already recognizes optimizer effects on gap shape and amplitude.

   **Do not claim:** “This excludes frequency or optimizer explanations,” or “Rare contexts fork because more of their continuation mass is unseen.”

   **Defensible wording:** “These associations support a role for continuation coverage, while leaving optimizer-mediated and other context-dependent explanations unresolved.”

5. **The affine fits are not algebraically tautological, but their current use is partly circular.**

   The decomposition
   \[
   G(F)=f(F)D(F)+(1-f(F))S(F)
   \]
   is an accounting identity. A common two-parameter approximation \(G(F)\approx af(F)+b\) across many bins is an additional, testable restriction. It is therefore legitimate descriptive evidence.

   However, the scripts fit \(a,b\) to the same observed \(f\) and gap points used to report the residuals. They do not independently predict those gaps from training counts. See [analyze_prop.py](/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/docs/appendices/gt_audit_20260925/scripts/analyze_prop.py:32) and [draft_figs2.py](/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/docs/appendices/gt_audit_20260925/scripts/draft_figs2.py:24). A multiplicative calibration of \(f\) is also absorbed exactly by the free slope, so affine goodness of fit cannot validate the absolute Good–Turing calibration.

   More seriously, a close affine fit does **not** establish approximately constant \(D,S\). For the same both-table, seed-42, step-1000 per-position data:

   - The bigram fit is approximately \(G=9.03f-0.487\).
   - Every measured \(S(F)\) in the fitted bins is **positive**, ranging from **0.348 to 2.231**.
   - \(D(F)\) ranges from **6.110 to 8.430**.

   The fitted intercept cannot be identified with the measured constant seen-continuation term in this example. Frequency-dependent \(D,S\) can compensate and produce a nearly straight aggregate relationship.

   **Defensible wording:** “Across the analyzed frequency bins, a two-parameter affine function of observed novelty summarizes the diagnostic gaps well. This empirical approximation does not imply frequency-independent component losses.”

6. **The numerical fit claims need explicit definitions and scope.**

   I reproduced the reported slope endpoints for `fig1v5_rp_both_s{42,43,44}_fixed`: approximately **4.70→10.20** for bigrams and **2.39→9.51** for trigrams, from step 670 to 2000.

   But:

   - There are **22 bigram bins and 20 trigram bins** under the implemented cutoffs, not 22 for both.
   - The code reports absolute RMS residuals without defining “relative RMS.”
   - At step 1000, seed 42, RMS residuals are approximately **0.255** and **0.151 nats**. Dividing by the observed gap range gives about **4.4%** and **4.5%**; dividing by RMS gap gives about **6.4%** and **8.1%**. “1–5%” is therefore not interpretable without its denominator.
   - Agreement across model seeds on one shared corpus and probe does not measure uncertainty across corpora or documents.
   - The supplied fitting script does not itself substantiate the full optimizer, injection-route, and learning-rate universality claim.

   The recomputation used the saved [per-frequency extract](/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/data/gt_audit_20260925/perF_extract.json), whose extraction logic is in [extract_perF.py](/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/docs/appendices/gt_audit_20260925/scripts/extract_perF.py:19). The corresponding raw three-seed run logs were not available at the checked local canonical paths, so this verifies the saved summaries rather than their entire provenance chain.

7. **The decomposition and exponent claims still need measurement boundaries.**

   The implemented gap is the difference between separately token-averaged validation and fixed-training-probe losses within a band. It does not match individual contexts and apply common context weights. The identity remains exact with that scalar training baseline, but “training loss at the same contexts” suggests stronger matching than the code performs. See [analyze_prop.py](/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/docs/appendices/gt_audit_20260925/scripts/analyze_prop.py:17) and [perpos_decomp.py](/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/docs/appendices/gt_audit_20260925/scripts/perpos_decomp.py:43). This diagnostic gap also differs from the repository’s primary online-training gap.

   In the \(F=9\)–38 contrast above, the weighted novel component increases by **2.972 nats**, while the weighted seen component decreases by **0.388**, producing a net increase of **2.584**. The precise statement is that the novel component **more than accounts for the net increase, partly offset by the seen component**. This is an accounting attribution, not a causal intervention.

   The existing paper’s 73%/77%/100% table concerns **excess validation loss relative to no-gram**, and includes unseen contexts. It cannot establish that \(S(F)\) is small in a within-model train–validation decomposition. Compare [04_mechanism.tex](/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/paper/sections/04_mechanism.tex:14) with [appendix_e_theory.tex](/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/paper/sections/appendix_e_theory.tex:51).

   Finally, even if \(f(F)=CF^{-\alpha}\) exactly and \(G=af+b\),
   \[
   -\frac{d\log G}{d\log F}
   =\alpha\,\frac{af}{af+b}.
   \]
   Thus the gap generally does **not** inherit the same exponent when \(b\ne0\). The current claim that coverage “fixes the exponent” is unsupported: [04_mechanism.tex](/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/paper/sections/04_mechanism.tex:31). The existing conditional Zipf derivation is reasonable within its stated intermediate asymptotic regime; it does not establish a common exponent after mixing heterogeneous contexts selected by observed frequency: [appendix_e_theory.tex](/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/paper/sections/appendix_e_theory.tex:13).

A defensible replacement for the central claim would be:

> Under a Poissonized model with independent conditional continuations and matched training and validation distributions, the ratio of expected singleton counts in the \(F+1\) layer to its expected token count equals the validation-token-weighted probability of an unseen continuation for contexts observed \(F\) times. We use the corresponding empirical ratio as a training-count estimator, without claiming exact finite-sample unbiasedness. On the analyzed corpus, this estimator closely tracks held-out novelty across frequency, with a systematic calibration discrepancy that also appears in within-shard splits.
>
> Across the analyzed frequency bins, observed novelty is strongly associated with the fixed-probe train–validation gap, and a fitted affine relationship provides a compact empirical description. Singleton-stratified comparisons provide further evidence that continuation statistics matter beyond coarse frequency categories. These results are consistent with a coverage-based account, but they do not establish constant component losses, a corpus-determined gap exponent, or the exclusion of optimizer-mediated effects.