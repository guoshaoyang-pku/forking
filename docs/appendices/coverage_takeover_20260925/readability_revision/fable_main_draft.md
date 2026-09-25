**1. Chapter opening**

```latex
\section{Mechanism of Forking}
\label{sec:mechanism}

A finite corpus shows each context only some of its plausible continuations, so a validation position can pair a \emph{seen context}---one that occurs in training---with a \emph{novel continuation} that never followed it there. This chapter argues that replay turns these positions into loss: each pass shifts probability from unobserved continuations onto observed ones, crowding the novel ones out. We then show that this crowding-out happens inside context-specific memory: the $n$-gram module gives each context its own coordinate, so the damage stays local.
```

*(3 sentences, 84 words.)*

**2. New opening of 4.1 (Crowding-out under replay)** — replaces everything from "Coverage says what is missing" up to "Our answer is that forking is ordinary overfitting confined to very small problems."; the text then continues verbatim with "For a context $c_i$, call..."

```latex
Why should repeating true data hurt a model on data it has never seen? Our answer is that forking is ordinary overfitting confined to very small problems: each context defines its own next-token prediction problem, and replay overfits each one separately.
```

**3. Bridge at the end of 4.1** — replaces the final sentence of the subsection ("Coverage sets the fraction of novel continuations, while replay changes the novel and seen loss differences in Equation~\eqref{eq:gap-decomp}."), whose equation reference now lives in the appendix:

```latex
Crowding-out costs most where novel continuations are common: a rare context has had few chances to reveal its continuations, so a large share of what it must predict at validation is novel. If conditional distributions are heavy-tailed, this novel fraction decays only slowly with context frequency, making an approximate power-law trend in the gap plausible. Appendix~\ref{app:math} makes the accounting precise, decomposing the gap by novel fraction and estimating coverage from corpus counts with a Good--Turing identity.
```

*(3 sentences, 77 words.)*

A few notes on choices:

- The opening defines both terms in its first sentence and previews only the two mechanisms Chapter 4 now contains, in their new order; the coverage machinery is left entirely to the appendix, so the chapter no longer promises a "frequency law" section it doesn't deliver.
- Snippet 2 drops the old "Coverage says what is missing" hook (its antecedent subsection is gone) and replaces the three-question list with a single question the intro hasn't already answered, keeping the "ordinary overfitting" sentence as the pivot into the $c_i$ construction.
- The bridge names the causal chain once — rare context → high novel fraction → slow decay under heavy tails → plausible power-law trend — without claiming an exponent or citing fits, and hands both the exact decomposition and the Good–Turing identity to Appendix~\ref{app:math}.
- Section 4.2 ("Context subspaces are independent") needs no edits: its opening reference to Section~\ref{sec:sharpening} still resolves correctly after renumbering.