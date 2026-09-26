# Evidence and figure provenance

The release bundles two compact numerical packets and ten paper figure PDFs. The [figure manifest](figs/manifest.json) records exact source filenames, revision coordinates, content hashes, rendering dimensions, and scope. A file hash establishes identity with the packaged asset; it does not establish the validity of the experiment behind it.

| Material | What is included | Independently executable check |
| --- | --- | --- |
| Coverage | 1,512 derived observations; corpus-bin summaries; recorded fits; seeds 42/43/44 | verify_coverage.py recomputes eight selected fits from 432 observations |
| Context examples | 72 endpoint values; 15 token-class rows; four checkpoint identities | verify_contexts.py checks the export's internal consistency |
| Other paper figures | Exact vector PDF copies and web PNG renders | tools/check_release.py verifies packaged file and figure hashes |

## Reading the context examples

The input is BOS plus exactly three context tokens. The visible phrases are “petting your”, “introducing your”, and “there is a”; a tokenizer's token count is not the displayed word count. Inference was exported in float32 on CPU from ep5ckpt_v5_{nogram,both}_s42_fixed.

Open points show epoch 1 / step 337; filled points show epoch 5 / step 1685. The vertical axis is natural log probability. Higher values mean greater probability. Lines connect two measured endpoints and imply no monotonicity between them.

Blue tokens occur in training (either both corpora or training only); orange tokens occur only in the held-out corpus; gray tokens occur in neither. These are corpus-membership categories, not separate predictions for training and validation inputs. Counts refer to exact context–token occurrences at valid within-chunk positions, using training shard 1 and held-out shards 2–4.

“Others” is the log of the mean probability of the top 20 remaining tokens, reranked separately for each distribution. It is neither the sum of all remaining probability nor a fixed token set. The package includes the derived statistic, not the complete logits needed to independently reconstruct that ranking.

## Reading the coverage relationship

The packaged fit uses directly measured novel-continuation fractions and the online gap. Novel continuations of a training-seen context differ from contexts unseen in training. The latter do not have a defined training-side gap. Error bars are sample standard deviations over three training seeds, not confidence intervals over independent corpora.

The Good–Turing panels are training-count estimates. Agreement in this setting does not establish a universal exponent, universal calibration, or a unique causal explanation of the gap. The affine fit includes an intercept; the verifier also reports the through-origin residual for comparison.

## Other figures and interpretation

The overview mixes a schematic representation diagram with empirical curves. Its first-pass panel is the initial segment of a replay experiment. The local sharpening / cross-context recovery illustration is entirely schematic; its heights are not measured probabilities.

The replay, table-size, corpus-size, aggregate continuation, and intervention panels are included as paper assets. The minimal numerical packet does not reproduce all of their underlying training runs. The aggregate continuation panel is fixed-validation cross-entropy, whereas the coverage and replay panels use a train–validation gap. Consult each manifest entry before comparing their axes or seed scope.

The same fixed addressing rule applies to training and validation contexts. Learned table values and shared model parameters evolve during replay. Intervention results support the importance of the local-memory pathway within the tested settings; they do not by themselves establish independent context subspaces or equivalence to learned MoE routing.
