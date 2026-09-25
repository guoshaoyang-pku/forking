Write clear English research prose for human readers. Guidance from Orchestra ML-paper-writing: organize around a single narrative; every sentence should advance it; put familiar information first and new information last; keep subjects close to verbs; prefer precise verbs and concrete examples. Preserve the author's existing phrasing when it works. No grand claims, rhetorical contrasts, audit language, or dense lists of qualifications. Do not invent facts, citations or numbers. Output LaTeX-ready prose. This is editorial work, not new research.

Scientific constraints: a novel continuation is a token absent AFTER THAT CONTEXT in training, with context count F>=1. It need not be a new token or a new context. Replay reinforces observed continuations without new evidence for absent alternatives. Idealized gradient descent in logit coordinates grows seen-to-unseen odds, but actual networks/optimizers have coupled updates. The independent context-mask experiment supports weak interaction. Coverage contributes to the frequency dependence, but does not uniquely determine the exponent. The exact frequency-bin decomposition is G_t(F)=f(F)D_t(F)+(1-f(F))S_t(F), where D=novel val loss minus online train loss and S=seen val loss minus same online train loss; S need not be small. Empirically an affine relation G≈a_t f+b_t describes the measured bins. Only if S is negligible and D is almost F-independent does G≈D_t f. No universal proportionality or corpus-fixed exact exponent. Good--Turing estimates missing mass, under sampling assumptions. Conditional heavy tail p_r~A r^-s gives expected missing mass ~F^-(1-1/s) for a fixed context in an intermediate asymptotic regime. Heterogeneous context cross-sections are not that experiment. Old exact exponent agreement was invalid and must not be restored. Main text should give intuition, appendix gives conditions.
Task: Independently draft a short first subsection for Appendix E (title "From novel continuations to the frequency law", label app:coverage). Main Chapter 4 will begin with crowding-out and context subspaces. All quantitative coverage evidence lives in E. Keep existing rigorous derivations later in E intact.
We need an inviting equation-driven explanation, ~200--300 words excluding equations, not a rewritten audit. Define f as held-out novel fraction within a frequency bin. Display the exact two-component decomposition with short definitions of D and S. Then show the approximation G≈a_t f+b_t (label eq:gap_j) and explain that proportionality is the special case with negligible seen term and F-independent novel penalty; distinguish an empirical affine fit from constant component losses. Briefly connect GT and the conditional-tail power law, with appendix cross-references app:gt-identity and eq:occupancy. Explain the shape clearly without overstating a cross-context exponent theorem. No detailed Poisson formula here; it already appears in next subsection. No fitting numbers or statistical protocol; those appear in app:coverage-tests.
End with 1 sentence directing to the existing linear figure fig:coverage and detailed fits. Deliver ONLY the LaTeX subsection. Do not duplicate the full GT derivation or tail integral. Keep formulas compact, elegant and correctly conditioned. The original user said the few equations made the picture clear; preserve that quality.
Current detailed appendix (to remain after your new subsection):
\section{Mathematical Derivations and Model Tests}
\label{app:math}

\subsection{Good--Turing estimation and continuation coverage}
\label{app:gt-identity}
For context $i$, let $p_{iy}$ be its continuation distribution and $m_i(F)=\sum_y p_{iy}(1-p_{iy})^F$ its expected missing mass after $F$ independent draws. If $K_{i,1}(k)$ counts continuation types occurring exactly once in $k$ draws, then
\begin{equation}
\frac{\mathbb E K_{i,1}(F+1)}{F+1}
=\sum_y p_{iy}(1-p_{iy})^F=m_i(F).
\label{eq:gt-single-context}
\end{equation}
This is an expectation identity for a fixed context. By contrast, $K_{i,1}(F)/F$ has expectation $m_i(F-1)$ and equals one at $F=1$ by construction. Neither statement by itself identifies the validation-token-weighted missing mass across different contexts observed $F$ times.

\paragraph{Heterogeneous contexts and token weighting.}
Assume training context counts $N_i$ are Poisson with means $\lambda_i=Tq_i$, continuations are conditionally independent, and an independent validation token has context distribution $q_i$ and the same conditional probabilities $p_{iy}$. Write $\pi_F(\lambda)=e^{-\lambda}\lambda^F/F!$. Conditional on the validation context having training count $F$, its population novelty probability is
\begin{equation}
f_\star(F)=\frac{\sum_i q_i\pi_F(\lambda_i)m_i(F)}{\sum_i q_i\pi_F(\lambda_i)}.
\end{equation}
Define $n(F)=\sum_i\mathbf1[N_i=F]$ and $N_1(F)=\sum_i\mathbf1[N_i=F]K_{i,1}(F)$. Using $(F+1)\pi_{F+1}(\lambda)=\lambda\pi_F(\lambda)$ gives
\begin{equation}
\begin{aligned}
\mathbb E N_1(F+1)&=T\sum_i q_i\pi_F(\lambda_i)m_i(F),\\
(F+1)\mathbb E n(F+1)&=T\sum_i q_i\pi_F(\lambda_i),\\
f_\star(F)&=\frac{\mathbb E N_1(F+1)}{(F+1)\mathbb E n(F+1)}.
\end{aligned}
\label{eq:gt-poisson}
\end{equation}
This is a ratio of expectations under the sampling model, not an equality for every realized corpus or an unbiasedness guarantee for the empirical ratio. For a bin $B$ of exact counts, the corresponding train-only estimator is
\begin{equation}
\widehat f_{\mathrm{GT}}(B)=\frac{\sum_{F\in B}N_1(F+1)}{\sum_{F\in B}(F+1)n(F+1)}.
\label{eq:gt-bin}
\end{equation}
Its pooling weights use training counts. Weighting exact-layer estimates by held-out token counts would use validation information.

\paragraph{Uncalibrated corpus check.}
Counts use training shard 1 (49,716,936 tokens); the full held-out check uses 49,102,848 prediction positions from shard 2. Context counts follow the recorded raw-concatenation frequency index, while validation keys follow the model's 2049-token chunk layout. Natural text need not satisfy the Poisson or conditional-independence assumptions. In the 18 native bins over $F=1$--$1000$, Equation~\eqref{eq:gt-bin} gives 0.870--0.885 times the directly measured novel fraction for bigram contexts and 0.857--0.937 for trigram contexts (Figure~\ref{fig:coverage}a). At $F=1$, measured novelty is 0.790 and 0.780, compared with shifted estimates 0.699 and 0.669. No validation-fitted scaling factor is applied. A similar underestimate appears in within-training-shard splits, suggesting that train--validation domain shift alone is insufficient to explain it; these observational checks do not identify document repetition as its cause.

\paragraph{Conditional tails.}
For a context $c_i$ with $F_i=F$, let $y_{i,(r)}$ denote the $r$-th most likely continuation and $p_{i,(r)}$ its true conditional probability. Model the observed continuations $Y_{i,1},\ldots,Y_{i,F}$ as independent draws, with counts $n_{i,(r)}(F)=\#\{k\in\{1,\ldots,F\}:Y_{i,k}=y_{i,(r)}\}$. The novel-continuation rate $f_i(F)$ and its expectation under this sampling model are
\begin{equation}
f_i(F)=\sum_r p_{i,(r)}
\mathbf{1}\left[n_{i,(r)}(F)=0\right],
\quad
\mathbb{E}[f_i(F)]=\sum_r p_{i,(r)}(1-p_{i,(r)})^F.
\end{equation}

Assume a conditional tail $p_{i,(r)}\sim A_i r^{-s}$ with $s>1$. For large $F$ within the tail's scaling range, before finite-vocabulary effects become significant, the dominant probabilities are of order $1/F$. Approximating $(1-p_{i,(r)})^F$ by $e^{-Fp_{i,(r)}}$ and the rank sum by an integral then gives
\citep{gnedin2007occupancy}
\begin{equation}
\begin{aligned}
\mathbb{E}[f_i(F)]
&\sim
\int_1^\infty A_i r^{-s}e^{-F A_i r^{-s}}\,\mathrm{d}r\\
&\sim
\frac{A_i^{1/s}}{s}
\Gamma\left(1-\frac{1}{s}\right)
F^{-(1-1/s)}\\
&\propto F^{-\alpha},
\quad
\alpha=1-\frac{1}{s}.
\end{aligned}
\label{eq:occupancy}
\end{equation}

This gives an asymptotic account of slow missing-mass decay for a fixed context; it does not determine the cross-context gap exponent in Section~\ref{sec:coverage}. Finite vocabulary limits the power-law regime to an intermediate range. Heterogeneity across contexts may alter the observed effective exponent.

\subsection{Crowding-out within a context subspace}
\label{app:crowding-derivation}

Fix a context $c_i$ and call the next-token logits computed from it, $a_v$ for $v$ in the vocabulary, its \emph{context subspace}; the $n$-gram table gives $c_i$ a learned coordinate, which can make its logits respond differently from those of other contexts (Section~\ref{sec:context-locality}). Partition the vocabulary into the continuations of $c_i$ seen in training ($\mathcal{S}$), those that occur only in validation ($\mathcal{V}$), and the rest. With $p_v=\softmax{a_v}$, one idealized gradient step directly in logit coordinates on a training target $C\in\mathcal{S}$ moves every logit by
\begin{equation}
a_v^{(t+1)}-a_v^{(t)}=-\eta\frac{\partial\ell_C}{\partial a_v}=-\eta\left(p_v^{(t)}-\mathbf{1}[v=C]\right),\qquad \eta>0,
\label{eq:av}
\end{equation}
raising $a_C$ by $\eta(1-p_C)$ and lowering every other logit by $\eta p_v$. For any $u\notin\mathcal{S}$ the odds $z(t)=p_C^{(t)}/p_u^{(t)}$ therefore satisfy
\begin{equation}
z(t+1)=z(t)\exp\!\left[\eta\left(1-p_C^{(t)}+p_u^{(t)}\right)\right]>z(t),
\label{eq:zt}
\end{equation}
For this one distribution, $\ell_u-\ell_C=\ln z$, so the loss difference between an unseen alternative and the observed target grows. This does not imply equal and opposite changes in their absolute losses, or identify $\ln z$ with the bin-averaged $D_t$ in Equation~\eqref{eq:gap-decomp}. In a parameterized network a small gradient step instead gives $\Delta a\simeq-\eta JJ^\top(p-e_C)$, where $J$ is the logit Jacobian; optimizer preconditioning changes this coupling further. Equations~\eqref{eq:av}--\eqref{eq:zt} therefore describe an idealized sharpening mechanism, rather than a derivation of the measured coefficients.

\subsection{Position-class decomposition of the excess validation loss}
\label{app:novel-share}

Table~\ref{tab:novel-share} decomposes the both-table model's excess validation loss over the no-gram control (runs \code{s1v5_128_marm_both_ckpt_fixed} and \code{s1v5_128_marm_nogram_ckpt_fixed}, seed 42, step 1000) by whether the true continuation followed its context in the training shard. Contexts are counted at bigram or trigram granularity; \emph{novel} includes positions whose context itself is absent from training.

\begin{table}[!htbp]
\centering\small
\caption{Share of the step-1000 excess validation loss ($1.68$ nats per token, 589{,}248 positions) carried by each position class.}
\label{tab:novel-share}
\begin{tabular}{llrr}
\toprule
context & class & share of positions & share of excess loss \\
\midrule
bigram & novel continuation & 31\% & 73\% \\
bigram & novel or seen with $F\le 31$ & 39\% & 77\% \\
trigram & novel continuation & 66\% & 100\% \\
\bottomrule
\end{tabular}
\end{table}

These are shares of the excess \emph{validation} loss relative to no-gram, not shares of a model's train--validation gap. They do not imply $S_t(F)\approx0$. In particular, the table includes $F=0$ positions; the frequency-gap tests below exclude them because no training counterpart exists.

\subsection{Tests of the coverage approximation}
\label{app:coverage-tests}
\label{app:gt-fits}
\paragraph{Protocol.}
The online-gap tests use \code{fig1v5_rp_both_s42_fixed}, \code{fig1v5_rp_both_s43_fixed} and \code{fig1v5_rp_both_s44_fixed}, with matched \code{fig1v5_rp_nogram_s42_fixed}, \code{fig1v5_rp_nogram_s43_fixed} and \code{fig1v5_rp_nogram_s44_fixed} controls. They use the standard input-injection setting: RMSProp tables without momentum, table LR $128\times$, $2^{20}$ rows per table, backbone LR $0.0006$, and a 100-step warmup followed by constant LR. Training replays shard 1; the fixed validation pool contains 589,824 positions from the first four batches of shard 2. Direct novelty counts match every plotted validation bin's token count exactly. Training losses are recorded before the update and validation losses after it, at the same logged step.

All 18 native bins contained in $1\le F\le1000$ have at least 300 train and validation tokens at the reported steps; no positive-gap filtering is applied. The bins are $1,2,3,4,5,6$--$8,9$--$12,13$--$20,21$--$30,31$--$50,51$--$75,76$--$100,101$--$150,151$--$200,201$--$300,301$--$500,501$--$750,751$--$1000$. Steps 670, 1000, 1340 and 2000 lie near the ends of passes 2, 3, 4 and 6, but are not exact pass boundaries. We fit the seed-mean gap by equal-bin least squares. The same held-out corpus supplies $f$ for all seeds.

\begin{table}[!htbp]
\centering\small
\caption{Online gap fit $af+b$ on 18 bins. Coefficients and rms errors are in nats; relative rms is divided by mean absolute bin gap. The holdout fit uses alternating frequency bins and evaluates the others, testing interpolation on this corpus rather than corpus transfer.}
\label{tab:affine}
\begin{tabular}{llrrrrr}
\toprule
Context & Step & $a$ & $b$ & rms & Relative & Holdout\\
\midrule
Bigram & 670 & 4.87 & $-0.48$ & 0.137 & 8.2\% & 0.169\\
 & 1000 & 9.58 & $-0.53$ & 0.233 & 6.3\% & 0.269\\
 & 1340 & 10.86 & 0.16 & 0.205 & 4.1\% & 0.228\\
 & 2000 & 10.90 & 1.89 & 0.202 & 3.0\% & 0.233\\
Trigram & 670 & 2.35 & $-0.50$ & 0.063 & 10.7\% & 0.060\\
 & 1000 & 5.81 & $-1.11$ & 0.155 & 9.9\% & 0.132\\
 & 1340 & 8.00 & $-1.31$ & 0.172 & 7.2\% & 0.180\\
 & 2000 & 10.42 & $-0.79$ & 0.129 & 3.2\% & 0.132\\
\bottomrule
\end{tabular}
\end{table}

Validation-token weighting changes relative rms to 3.3--10.5\% (bigram) and 3.2--9.1\% (trigram). Residuals are structured and exceed seed variation; three training seeds do not quantify uncertainty across corpora. The fit is a compact description, not a prediction of its own coefficients. Moreover, a close affine relation need not imply constant $D_t$ or $S_t$: in the separate seed-42 step-1000 fixed-probe diagnostic, the bigram fit has a negative intercept while all measured seen terms are positive. Frequency-dependent terms can compensate.

\paragraph{Exact-count comparisons.}
The stored \code{s1v5_128_marm_both_ckpt_fixed} step-1000 probes (seed 42) allow a supplementary comparison at each exact $F=2,\ldots,8$. Contexts are stratified by their training singleton fraction into $[0,.25)$, $[.25,.5)$, $[.5,.75)$ and $[.75,1]$, retaining cells with at least 200 train and validation positions. At bigram $F=4$, the extreme strata have $f=0.247$ versus $0.795$ and diagnostic gaps 3.234 versus 6.013 nats. At trigram $F=4$, gaps rise from 1.448 to 2.619 nats and then decline to 2.505 in the highest stratum. Thus the association goes beyond count alone but is not universally monotone. Singleton fraction also reflects conditional entropy and concentration; it is not randomized. These losses use fixed probes and probabilities floored at $5.96\times10^{-8}$ (up to 1,094 zero probabilities in an array), so they are censored diagnostics distinct from the primary online gap. Novelty counts are unaffected by the loss floor.

\paragraph{What an exponent can mean.}
Even if $f(F)=cF^{-\alpha}$ exactly and $\gap=af+b$, the local gap exponent would be
\begin{equation}
\beta_{\mathrm{eff}}(F)=-\frac{\mathrm d\log\gap}{\mathrm d\log F}
=\alpha\,\frac{af(F)}{af(F)+b}.
\label{eq:coverage-effective-exponent}
\end{equation}
Here $a$ and $b$ are held fixed as $F$ varies, and the logarithm requires a positive gap. The exponent equals $\alpha$ only when the intercept is negligible. Changes in coefficients with training or table capacity can change an observed exponent without changing the corpus. The registered table-size sweep and the time-dependent fits therefore preclude a corpus-fixed gap exponent. The evidence supports coverage as a contributor to the frequency dependence under the measured protocol; it does not establish a universal affine law across corpora, optimizers or injection paths, or coverage as the sole causal mediator.
