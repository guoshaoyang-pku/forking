Six bullets on the narrative arc, with the exact hedged phrasings that keep the mystery honest:

- **Open with the regularity, stated as measurement.** Lead with the empirical fit — "the train–validation gap tracks a simple function of context frequency, gap ≈ a·f(F)+b, across 18 frequency bins and 3 seeds (relative RMS 6.3% bigram, 9.9% trigram at step 1000)." Keep one equation in the main text; push GT indexing, derivations, and all qualifications to Appendix E. For the figure, say the dashed curves are "the directly measured f, rescaled and offset to each checkpoint" — never "fitted power laws" or "predicted from training alone."

- **Explain the shape via novel continuations.** Define them plainly (a validation target that never followed this context in training, among contexts with F ≥ 1), then: "Good–Turing estimates from training singletons suggest the *uncovered continuation mass* declines with F in the same way." Say "conditional heavy tails yield a power-law asymptotic for the missing mass of a fixed context" — an asymptotic and a suggestion, not a universal cross-context exponent or gap ∝ GT.

- **Turn the clue with time, not drama.** The pivot sentence: "Once a full pass has observed the corpus, f is fixed on replay — yet the gap keeps growing." Report "similar increments over the first few passes, shrinking at long horizons" (e.g., ~0.9–1.5 nats early, 0.24 by pass 20); call them **total online gap increments**, never novel-CE increments.

- **Frame the dichotomy carefully.** "A static coverage mismatch identifies *which* samples are vulnerable but does not by itself explain *progressive* damage" — while noting a statistical account is not ruled out. The seen/novel CE plot supports epoch-end deterioration; don't imply a dense within-epoch trace exists.

- **Present crowding-out as an idealization.** "In a logit-space caricature, each step on an observed continuation raises its odds against all unobserved alternatives" — explicitly note real optimizers couple contexts; this is a mechanism sketch, not a theorem.

- **Subspace section: keep the question, drop the verdict.** Rephrase as "masking experiments *probe* whether damage travels through shared backbone directions," present results as preliminary/unreconciled, and delete "frozen gap" as established fact. Its conceptual role — localizing the interference — survives without claiming confirmed independence. (Final qualification is the coordinator's; your edit just avoids strengthening it.)

(~340 words)