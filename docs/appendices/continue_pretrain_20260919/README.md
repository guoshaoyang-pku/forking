All six runs initialize from sftbase_ep20_tri_ckpt_fixed/final_model.pt, then restart optimizer state and a 100-step warmup. The source run IDs are retained for provenance; semantically these are continued language-model pretraining runs, not response-only instruction SFT. The two arms differ only in whether the n-gram table is trainable or frozen.

gap = fixed_val_loss - online_train_loss. The dotted line in each loss panel is train_loss + epoch-1 gap: it is the validation trajectory expected if the initial gap had stayed fixed. Therefore, the increase in gap is exactly the validation improvement erased relative to that counterfactual.

The source launchers logged validation only at epoch boundaries (step = 333, 666, 999, 1332, 1665). The figures are explicitly labelled epoch-boundary observations. They use the Chapter 3–4 visual language (train solid, translucent validation, dashed gap, restrained blue/coral palette), but they are not dense freq=10 curves. A genuine freq=10 version requires rerunning from the same checkpoint with --val_interval 10.

Generated figures:

- fig_continue_pretrain_id.svg
- fig_continue_pretrain_magicoder.svg
- fig_continue_pretrain_openr1.svg

At epoch 5, the gap increase and corresponding erased validation improvement are:

- ID: trainable +2.389, frozen +0.330 nats.
- Magicoder OOD: trainable +1.008, frozen +0.247 nats.
- OpenR1 Math OOD: trainable +0.733, frozen +0.129 nats.

The plot's dashed line is `train_loss + epoch-1 gap`. Because `val = train + gap`, the amount by which gap increases is exactly the validation improvement erased relative to that fixed-gap counterfactual. At epoch 5: ID/trainable has gap increase +2.389 and observed val change +1.019 (a validation regression despite train improving by 1.370); ID/frozen has +0.330 gap increase and val change +0.070 versus -0.260 under fixed gap. Magicoder/trainable has +1.008 erased and only -0.206 val improvement versus -1.214 counterfactual; OpenR1/trainable has +0.733 erased and only -0.109 versus -0.842. Frozen OOD arms still improve in validation, but their +0.247 (Magicoder) and +0.129 (OpenR1) gap increases remove the same amount from the expected improvement.
