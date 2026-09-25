Write clear English research prose for human readers. Guidance from Orchestra ML-paper-writing: organize around a single narrative; every sentence should advance it; put familiar information first and new information last; keep subjects close to verbs; prefer precise verbs and concrete examples. Preserve the author's existing phrasing when it works. No grand claims, rhetorical contrasts, audit language, or dense lists of qualifications. Do not invent facts, citations or numbers. Output LaTeX-ready prose. This is editorial work, not new research.

Scientific constraints: a novel continuation is a token absent AFTER THAT CONTEXT in training, with context count F>=1. It need not be a new token or a new context. Replay reinforces observed continuations without new evidence for absent alternatives. Idealized gradient descent in logit coordinates grows seen-to-unseen odds, but actual networks/optimizers have coupled updates. The independent context-mask experiment supports weak interaction. Coverage contributes to the frequency dependence, but does not uniquely determine the exponent. The exact frequency-bin decomposition is G_t(F)=f(F)D_t(F)+(1-f(F))S_t(F), where D=novel val loss minus online train loss and S=seen val loss minus same online train loss; S need not be small. Empirically an affine relation G≈a_t f+b_t describes the measured bins. Only if S is negligible and D is almost F-independent does G≈D_t f. No universal proportionality or corpus-fixed exact exponent. Good--Turing estimates missing mass, under sampling assumptions. Conditional heavy tail p_r~A r^-s gives expected missing mass ~F^-(1-1/s) for a fixed context in an intermediate asymptotic regime. Heterogeneous context cross-sections are not that experiment. Old exact exponent agreement was invalid and must not be restored. Main text should give intuition, appendix gives conditions.
Task: The author dislikes the long coverage subsection in current Chapter 4. We will move ALL of that subsection, its equations and its figure to Appendix E. Chapter 4 will start with current "Crowding-out under replay" (renumbered 4.1), followed by current "Context subspaces are independent" (renumbered 4.2). The latter must remain verbatim.
Provide ONLY these three replacement snippets, with simple headings:
1. Chapter opening: 3 sentences, 55--85 words. Define seen context, novel continuation immediately and orient reader to replay crowding-out and context-specific memory. No equations, fitting numbers or protocol.
2. First 1--2 sentences of new 4.1, replacing the old "Coverage says what is missing..." opening, ending just before "For a context c_i...". Preserve "forking is ordinary overfitting confined to very small problems" if it fits. Avoid repeating the intro.
3. Short bridge at the end of 4.1: 2--3 sentences, <=85 words, linking rare contexts, novel fraction, and why conditional heavy tails make an approximate power-law trend plausible. Point to Appendix~\ref{app:math} for the coverage decomposition and Good--Turing calculation. No universal exponent claim and no fitted coefficients. This should be readable and concrete, not a forest of caveats.
Do not rewrite existing examples, experiments or context-mask prose.
Current Chapter 4:
\section{Mechanism of Forking}
\label{sec:mechanism}

Section~\ref{sec:empirical} left two questions: why the gap depends on context frequency, and why replay keeps widening it. We answer the first with continuation \emph{coverage} and the second with \emph{crowding-out} inside context subspaces, then show that the $n$-gram module is what makes those subspaces independent. Additional experimental details are in Appendix~\ref{app:mechanistic}.

\subsection{Coverage and the frequency law}
\label{sec:coverage}
Section~\ref{sec:forking-factors} found that low-frequency contexts carry most of the gap. A finite training corpus leaves some plausible continuations of each context unobserved. Let $f(F)$ be the fraction of validation positions whose context occurs $F\ge1$ times in training but whose true continuation never followed it there---\emph{seen context, novel continuation}. Averaged over a frequency bin, the gap splits exactly into a novel and a seen term,
\begin{equation}
\begin{aligned}
\gap_t(F)&=f(F)D_t(F)+\big(1-f(F)\big)S_t(F),\\
D_t(F)&=\ell^{\mathrm{val}}_{\mathrm{novel},t}(F)-\ell^{\mathrm{tr}}_t(F),\qquad
S_t(F)=\ell^{\mathrm{val}}_{\mathrm{seen},t}(F)-\ell^{\mathrm{tr}}_t(F).
\end{aligned}
\label{eq:gap-decomp}
\end{equation}
Here $\ell^{\mathrm{tr}}_t$ is the current-batch training loss and the validation means use the fixed validation pool at the same logged step. These are means within a frequency bin, not losses matched context by context. Coverage sets the weights; $D_t$ and $S_t$ describe how the model responds. The seen term need not be small.

\paragraph{From coverage to the frequency law.}
If $D_t$ and $S_t$ were constant across $F$, the decomposition would be affine in $f$. We test the corresponding two-parameter approximation directly,
\begin{equation}
\gap_t(F)\approx a_t f(F)+b_t.
\label{eq:gap_j}
\end{equation}
For the both-table model, 18 frequency bins spanning $F=1$--$1000$ lie close to a line in the directly measured novel fraction (Figure~\ref{fig:coverage}b,c). For fits to the three-seed mean at steps 670, 1000, 1340 and 2000, relative rms residuals are 3--8\% for bigram and 3--11\% for trigram contexts. The intercept matters: at step 2000 the bigram fit has $b_t=1.89$ nats; forcing the line through zero increases rms error from 0.20 to 0.76 nats. This approximate affine relation summarizes the frequency dependence, but does not establish constant component losses: a fitted $b_t$ is not in general the measured $S_t(F)$ (Appendix~\ref{app:coverage-tests}).

\paragraph{Coverage from corpus counts.}
For a context with independent continuation draws, the expected singleton fraction after $F+1$ observations equals the expected missing mass after $F$. Pooling heterogeneous contexts additionally requires a model for their occurrence counts. Under Poisson counts and matched train/validation distributions, the token-weighted population novel fraction satisfies
\begin{equation}
f_\star(F)=\frac{\mathbb E N_1(F+1)}{(F+1)\mathbb E n(F+1)},\qquad
\widehat f_{\mathrm{GT}}(F)=\frac{N_1(F+1)}{(F+1)n(F+1)},
\label{eq:gt}
\end{equation}
where $n(k)$ counts contexts observed $k$ times and $N_1(k)$ their singleton context--continuation types. The first equality is a Good--Turing expectation identity; the second defines its empirical estimator \citep{good1953population,gale1995good}. The derivation and assumptions are in Appendix~\ref{app:gt-identity}. On our corpus the uncalibrated estimate tracks held-out novelty but underestimates it: its ratio to the direct count is 0.870--0.885 for bigram and 0.857--0.937 for trigram contexts (Figure~\ref{fig:coverage}a). No validation-fitted correction is applied.

\begin{figure}[!t]
\centering
\includegraphics[width=\linewidth]{fig_coverage_linear.pdf}
\caption{Coverage and the gap. (a) Novel fraction predicted from training counts against directly measured held-out novelty; the dashed diagonal is equality. (b,c) Online gap against the directly measured novel fraction, with affine fits at four training steps. Each point is a frequency bin; bars in (b,c) show the standard deviation across three seeds. The no-gram control is shown at step 1000. All panels use linear axes; protocol and residuals are in Appendix~\ref{app:coverage-tests}.}
\label{fig:coverage}
\end{figure}

Coverage also varies among contexts with the same count. At exact $F=4$, splitting bigram contexts by their training singleton fraction gives novel fractions 0.25 versus 0.80 and diagnostic gaps 3.23 versus 6.01 nats in the extreme strata (seed 42, step 1000; fixed-probe diagnostic in Appendix~\ref{app:coverage-tests}). This supports a role for continuation statistics beyond count alone, without isolating coverage from conditional entropy or optimization. Conditional heavy tails explain why missing mass can decay slowly, but do not fix a cross-context gap exponent. Coverage describes which continuations lack evidence; the replay dynamics below determine the response to that missing evidence.

\begin{figure}[!t]
\centering
\begin{subfigure}[b]{0.58\linewidth}
\centering
\includegraphics[width=\linewidth]{figures/walk_their_dog_loss.pdf}
\subcaption{}
\label{fig:walk-their-dog}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.38\linewidth}
\centering
\includegraphics[width=\linewidth]{figures/fig09_crowd_ce_seen_novel.pdf}
\subcaption{}
\label{fig:crowd-ce}
\end{subfigure}
\caption{Crowding-out: (a) context-only continuation losses for ``walk their dog''; (b) validation cross-entropy of train-seen and novel continuations, pooled over contexts.}
\label{fig:crowding}
\end{figure}

\subsection{Crowding-out under replay}
\label{sec:sharpening}
Coverage says what is missing. The more important questions are why replay turns the missing coverage into loss, why the $n$-gram module is needed for it, and how the damage evolves with the number of passes. Our answer is that forking is ordinary overfitting confined to very small problems. For a context $c_i$, call the next-token logits computed from it its \emph{context subspace}, and split the vocabulary into the continuations of $c_i$ seen in training, those that appear only in validation, and the rest (Appendix~\ref{app:crowding-derivation}). If training shows a single continuation $C$, the loss in that subspace is $-\ln p_C$; an idealized step directly in logit coordinates raises the logit of $C$ by $\eta(1-p_C)$ and, through the softmax, lowers every other logit by $\eta p_v$. Mass flows from the unseen continuations onto the seen one, so the seen-to-unseen odds grow by a factor above one at every pass with nothing to stop them while $p_C<1$ (Appendix Equation~\eqref{eq:zt}). We call this \emph{crowding-out}; it provides an idealized account of the growth in the novel-continuation penalty with replay.

Figure~\ref{fig:walk-their-dog} shows the picture on one context. ``walk their dog'' is followed by \emph{down} once in training and, in the fixed validation window, once each by \emph{daily} and \emph{for}. Probed with context-only input, the probability of \emph{down} rises by more than an order of magnitude over five passes while \emph{daily} and \emph{for} are compressed---\emph{daily} by orders of magnitude---and so is the total mass outside the continuations observed in either sample. Figure~\ref{fig:crowd-ce} shows that the context is typical. Pooling validation positions whose bigram context occurs in training, split by whether the continuation was seen there (405{,}058 positions) or not (158{,}790), both models do well on seen continuations: the control's cross-entropy falls from 4.3 to 2.8 nats and the both-table model's stays near 3.4. On novel continuations they diverge: the control still improves, 6.9 to 5.6 nats, while the both-table model deteriorates from 7.5 to 9.0 with a step at every epoch boundary---the staircase of Figure~\ref{fig:staircase}. Consistently, at epoch~3 the both-table model's predictive entropy on validation is 2.93 nats against a cross-entropy of 5.26, whereas the control's are 3.59 and 3.58 (same runs as Figure~\ref{fig:crowd-ce}): far more confident than accurate, the signature of a model that has committed to few continuations per context \citep{carlsson2025hyperfitting}. Crowding-out also explains why freezing the tables does not stop forking (Section~\ref{sec:interventions}): the subspace coordinate is already in place, and the backbone can continue the descent on its own. Coverage sets the fraction of novel continuations, while replay changes the novel and seen loss differences in Equation~\eqref{eq:gap-decomp}.

\subsection{Context subspaces are independent}
\label{sec:context-locality}
\begin{figure}[!t]
\centering
\begin{subfigure}[b]{0.45\linewidth}
\includegraphics[width=\linewidth]{figures/fig07a_context_mask.pdf}
\subcaption{Gap contribution.}
\label{fig:ctxmask-gap}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.45\linewidth}
\includegraphics[width=\linewidth]{figures/fig07b_seen_mass.pdf}
\subcaption{Mass on train-seen continuations.}
\label{fig:ctxmask-mass}
\end{subfigure}
\caption{Random context loss masking.}
\label{fig:context-subspace-mask}
\end{figure}
Section~\ref{sec:sharpening} treats each context subspace as evolving on its own, with the sharpening within it as the engine of forking. Both statements are claims about what the $n$-gram module buys: a private coordinate per context, so that updates driven by one context barely touch the conditional distribution of another. We test this by ablation. We hold out a random 2\% of training trigram contexts and exclude the next-token loss at their positions from the start of epoch~2 or epoch~3 onward, keeping inputs, trainable parameters and the loss denominator unchanged; a disjoint 2\% serves as unmasked peers, and each masked run is paired with normal training from the same checkpoint. Figure~\ref{fig:ctxmask-gap} plots their gap contribution $C_S=\sum_{i\in S}q_i[\bar\ell_{\mathrm{val},i}-\bar\ell_{\mathrm{tr},i}]$ on fixed batches. Until the mask fires the masked curves track normal training; afterwards they freeze---for two and three further passes---while the unmasked and normal curves take a fresh jump at every replay boundary. The gap of a masked context stops growing the moment its own replay stops, although the remaining 98\% of contexts keep training throughout: gap growth is generated inside each subspace, not imported from the others.

Figure~\ref{fig:ctxmask-mass} follows the sharpening coordinate itself, the predictive mass on the continuations of each context that training has seen. In the n-gram model the unmasked contexts pour mass into their seen continuations pass after pass ($0.43\to0.69$ by epoch~4) while the masked ones freeze ($0.43\to0.40$): once a context's own loss is removed, nothing else sharpens its subspace. Without the $n$-gram module the mask changes nothing ($0.22\to0.39$ masked vs.\ $0.41$ unmasked), and replay raises the mass on seen and validation-only continuations alike ($0.013\to0.06$): every context's conditional distribution lives in parameters shared with all others, and repetition improves it holistically. The table therefore partitions optimization into weakly interacting context subspaces; inside a subspace, replay-driven sharpening proceeds unchecked by the continuations of other contexts, which is exactly the over-memorization assumed in Section~\ref{sec:sharpening}. Masked contexts are not immutable---the shared backbone still drifts---but that drift moves train and validation losses together and never enters the gap of Figure~\ref{fig:ctxmask-gap}.

