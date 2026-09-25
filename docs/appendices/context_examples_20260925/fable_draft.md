**Block 1 — figure caption (172 words)**

```latex
\caption{Next-token distributions for three contexts, before and after
memorization. Rows show the contexts \emph{petting your}, \emph{introducing
your}, and \emph{there is a}; the left column is the no-gram model, the right
column the model with bigram and trigram tables. Bars rise from zero and show
negative natural log probability in nats, so taller bars mean lower
probability. Light bars are epoch~1 and solid bars epoch~5 (steps 337 and
1685, seed 42). Each input is BOS plus exactly the three context tokens;
probabilities come from the full 8192-token softmax and are not renormalized
over the displayed animals. Labels record whether a token follows this exact
context in the training shard, the held-out shards, both, or neither, counted
at valid positions within 2049-token chunks. \emph{Others} is the negative
log of the arithmetic mean probability of the top twenty tokens outside the
five animals, chosen anew for each distribution; it is not total remaining
mass. These selected contexts illustrate the shift but do not establish
causal cross-context transfer. All panels share one y scale.}
```

**Block 2 — main-text replacement (two sentences)**

```latex
After the context \emph{petting your}, the n-gram model shifts sharply toward
\emph{dog} and away from \emph{cat} between the first and fifth epochs, while
the no-gram model instead drifts toward \emph{cat}
(\cref{fig:actual-logits-examples}). The bottom row of the figure shows the
actual \emph{there is a} distributions behind the schematic above, but both
\emph{dog} and \emph{cat} follow that context in training, so we do not claim
it matches the schematic.
```

**Block 3 — validation-CE caption (73 words)**

```latex
\caption{Validation cross-entropy across five epochs for the both-table and
no-gram models (seed 42; checkpoints at epochs 1--5, steps 337, 674, 1011,
1348, 1685). Positions are split by final bigram context: context and actual
continuation both seen in training (405{,}058 positions) versus seen context
with a novel continuation (158{,}790); positions with unseen contexts
(25{,}400) are excluded. Models receive full natural causal prefixes; each
point is the position-averaged negative log probability of the actual next
token.}
```