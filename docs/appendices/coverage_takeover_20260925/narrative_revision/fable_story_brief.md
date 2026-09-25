You are revising an English ML paper for a human author. Use a restrained scientific discovery narrative, not theatrical suspense. Guidance from Orchestra: one clear story; one purpose per paragraph; each sentence advances the argument; familiar information before new; concrete verbs; preserve good handwritten prose. No invented history, references, measurements or proof claims. Output LaTeX-ready prose.

Fixed corpus, input injection, clean both-tables R=2^20, 128x RMSProp without momentum, backbone AdamW .0006, warmup_constant100. Definition: novel continuation = a validation target that never followed this context in training, among training-seen contexts F>=1. GT estimates uncovered continuation mass from training singletons under sampling assumptions. Conditional heavy tails give a missing-mass power-law asymptotic for a fixed context; NOT a proof of a universal cross-context gap exponent. Empirical coverage fits: gap≈a_t f(F)+b_t over 18 native bins F1..1000, 3 seeds. At step1000 relative rms 6.3% bi,9.9%tri. Fig frequency already plots gap vs F and dashed curves obtained by scaling AND offsetting direct measured f. Do not pretend dashed curves are fitted power laws or training-only predictions. Main-text fit may be one simple equation, all qualifications/GT indexing/mathematical derivations belong in E. Do NOT restore false exponent equality within .01 or universal gap∝GT.
Narrative temporal clue: once a first complete pass has observed the fixed corpus, f does not change on replay but gap keeps increasing. In verified both-table seed42 long replay: end-pass gaps e1=-.062,e2=1.199,e3=2.730,e4=3.917,e5=4.803,e10=8.988,e20=12.674. Early increments1.261,1.531,1.187 then .885; long-term increments shrink to .241 at pass20. Say similar increments over the first few passes if useful, not exact constant across training; do NOT call these novel CE increments (they are total online gap). The separate seen/novel CE graph shows novel deterioration at epoch ends, but no dense trace of within-epoch novel CE. A static distribution mismatch suggests which samples are vulnerable, does not by itself explain progressive damage on replay; it is not ruled out as a statistical cause.
Crowding-out idealization: a gradient step directly in logits raises observed target C and lowers all others; odds pC/pu grow. Actual optimizer parameterization has cross-context coupling, not theorem of independent dynamics.
Important provenance issue: Original context-mask prose claims frozen gap after masking own contexts. Registry §63 reports type and row masking reruns failed to reproduce; shared backbone drift continues. Original collaborator figure protocol not reconciled. Do not strengthen independent-subspace claim. That section will be minimally qualified independently by coordinator; your task ends before it.
Author's latest preferred story: "First we found a power law, then its link to novel continuations—a clue from natural-language statistics. Why should novel continuations be harmed? Initially a probability-distribution mismatch looked sufficient, but replay adds further damage each epoch. This suggests training itself suppresses the other tokens, particularly within a context-specific subspace. Then show the example and tests. Put the frequency/coverage match up front, complete mathematics and correction terms in appendix."
Task: Draft (1) a ONE-sentence chapter opener that poses the frequency-law mystery and does not reveal crowding-out; (2) a SHORT new subsection 4.1 titled "From the frequency law to novel continuations", label sec:coverage, at most 180 words plus ONE short displayed equation gap_t(F)≈a_t f(F)+b_t (label eq:coverage-main, separate from E's equation). Start with approximate power-law decline already visible in Fig frequency. Define novel simply. Explain that direct novelty follows the gap after a scale/offset and GT/heavy tails connect it to text statistics. End on unresolved issue: coverage locates vulnerability, but not why replay worsens it. Don't clutter main text with assumptions/fit protocol/numbers; put ref app:math. (3) a 60--100-word beginning of 4.2 Crowding-out under replay, label sec:sharpening, that progresses from initially plausible finite-sample distribution mismatch to the fixed-f but growing-gap temporal clue, comparable early-pass increments, then a hypothesis that repeatedly rewarding seen continuations progressively suppresses alternatives. End just before original "For a context c_i, call..." so the original theory/example paragraphs can follow with minimal edits. Avoid claiming that static mismatch is disproved, or calling total gap increases novel-loss increases.
Return only the three snippets. Do not rewrite the preserved original paragraphs below.
\section{Mechanism of Forking}
\label{sec:mechanism}

A finite corpus shows each context only some of its plausible continuations, so a validation position can pair a \emph{seen context} with a \emph{novel continuation} that never followed it in training. Replay reinforces the observed continuations without supplying evidence for the others, and can crowd those alternatives out. We show how this happens, then test why the $n$-gram module allows it to proceed separately within each context; additional experimental details are in Appendix~\ref{app:mechanistic}.

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
Forking is ordinary overfitting confined to very small problems. For a context $c_i$, call the next-token logits computed from it its \emph{context subspace}, and split the vocabulary into the continuations of $c_i$ seen in training, those that appear only in validation, and the rest (Appendix~\ref{app:crowding-derivation}). If training shows a single continuation $C$, the loss in that subspace is $-\ln p_C$; an idealized step directly in logit coordinates raises the logit of $C$ by $\eta(1-p_C)$ and, through the softmax, lowers every other logit by $\eta p_v$. Mass flows from the unseen continuations onto the seen one, so the seen-to-unseen odds grow by a factor above one at every pass with nothing to stop them while $p_C<1$ (Appendix Equation~\eqref{eq:zt}). We call this \emph{crowding-out}; it provides an idealized account of the growth in the novel-continuation penalty with replay.

Figure~\ref{fig:walk-their-dog} shows the picture on one context. ``walk their dog'' is followed by \emph{down} once in training and, in the fixed validation window, once each by \emph{daily} and \emph{for}. Probed with context-only input, the probability of \emph{down} rises by more than an order of magnitude over five passes while \emph{daily} and \emph{for} are compressed---\emph{daily} by orders of magnitude---and so is the total mass outside the continuations observed in either sample. Figure~\ref{fig:crowd-ce} shows that the context is typical. Pooling validation positions whose bigram context occurs in training, split by whether the continuation was seen there (405{,}058 positions) or not (158{,}790), both models do well on seen continuations over the first three passes: the control's cross-entropy falls from 4.3 to 2.8 nats and the both-table model's stays near 3.4. On novel continuations they diverge: the control still improves, 6.9 to 5.6 nats, while the both-table model deteriorates from 7.5 to 9.0 with a step at every epoch boundary---the staircase of Figure~\ref{fig:staircase}. Consistently, at epoch~3 the both-table model's predictive entropy on validation is 2.93 nats against a cross-entropy of 5.26, whereas the control's are 3.59 and 3.58 (same runs as Figure~\ref{fig:crowd-ce}): far more confident than accurate, the signature of a model that has committed to few continuations per context \citep{carlsson2025hyperfitting}. Crowding-out also explains why freezing the tables does not stop forking (Section~\ref{sec:interventions}): the subspace coordinate is already in place, and the backbone can continue the descent on its own.

Crowding-out hurts more where novel continuations are common. Rare contexts have had fewer chances to reveal their continuations; under heavy-tailed conditional distributions, the novel fraction falls slowly with frequency, offering an explanation for the approximate power-law trend in the gap. Appendix~\ref{app:math} develops this connection through the seen/novel gap decomposition and Good--Turing estimation, including the terms that prevent exact proportionality.

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

