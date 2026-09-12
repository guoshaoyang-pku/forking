"""Model entry point for the 5-gram frequency-gap experiment.

This module provides a single ``GPT`` / ``GPTConfig`` / ``MODEL_PROVENANCE``
interface to the trainer.  It loads the repository's minimal ``NanoGPT`` first,
then the launcher-synced copy of the same source.

When the user supplies a new architecture, replace the body of this file with
that implementation; the trainer only depends on the three names above plus a
``num_scaling_params()`` method and an optional ``architecture_manifest()``.
"""

from __future__ import annotations

import hashlib
import os
import sys
import types
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Try to load the repository's minimal model first.  The launcher-synced copy
# is the same file because cluster workspaces do not contain the full repository
# tree.
_MAIN_TRAIN_PY = _PROJECT_ROOT / "code" / "train.py"
_SYNCED_TRAIN_PY = _PROJECT_ROOT / "train.py"

NanoGPTOriginal = None
NanoGPTMLPTrunk = None
_cluster_model_loaded = False
_cluster_model_source = None

for _candidate in (_MAIN_TRAIN_PY, _SYNCED_TRAIN_PY):
    if _candidate.exists():
        try:
            source = _candidate.read_text(encoding="utf-8")
            marker = (
                "# ---------------------------------------------------------------------------\n"
                "# Hyperparameters (edit these directly, no CLI flags needed)\n"
                "# ---------------------------------------------------------------------------"
            )
            definition_source = (
                source if _candidate in (_MAIN_TRAIN_PY, _SYNCED_TRAIN_PY)
                else source.split(marker, 1)[0]
            )

            # The prefix imports sibling modules (observable, lib, ...), so
            # make its directory importable.  Register the temporary module as
            # dataclasses resolve annotations through ``sys.modules``.
            _train_dir = str(_candidate.parent)
            if _train_dir not in sys.path:
                sys.path.insert(0, _train_dir)
            digest = hashlib.sha256(str(_candidate).encode()).hexdigest()[:12]
            module_name = f"_ngram5_cluster_defs_{digest}"
            _mod = types.ModuleType(module_name)
            _mod.__file__ = str(_candidate)
            _mod.__package__ = ""
            sys.modules[module_name] = _mod

            import builtins
            _orig_print = builtins.print
            builtins.print = lambda *a, **k: None
            try:
                exec(compile(definition_source, str(_candidate), "exec"), _mod.__dict__)
            finally:
                builtins.print = _orig_print
            if hasattr(_mod, "NanoGPT") and hasattr(_mod, "Config"):
                NanoGPTOriginal = _mod.NanoGPT
                NanoGPTMLPTrunk = None
                GPTConfig = _mod.Config
                MixedOptimizer = getattr(_mod, "MixedOptimizer", None)
                _cluster_model_loaded = True
                _cluster_model_source = str(_candidate)
                break
            if hasattr(_mod, "NanoGPTOriginal"):
                NanoGPTOriginal = _mod.NanoGPTOriginal
                NanoGPTMLPTrunk = getattr(_mod, "NanoGPTMLPTrunk", None)
                GPTConfig = getattr(_mod, "NanoGPTConfig", None) or getattr(_mod, "GPTConfig", None)
                MixedOptimizer = None
                _cluster_model_loaded = True
                _cluster_model_source = str(_candidate)
                break
        except Exception:
            if "module_name" in locals():
                sys.modules.pop(module_name, None)
            continue

# Trunk selection: "transformer" (default, the baseline) or "mlp" (position-wise
# MLP blocks; n-gram tables, embeddings, head, optimizer grouping unchanged).
TRUNK_VARIANT = os.environ.get("NGRAM5_TRUNK", "transformer").strip().lower()

if NanoGPTOriginal is not None:
    if TRUNK_VARIANT == "mlp":
        if NanoGPTMLPTrunk is None:
            raise RuntimeError(
                "NGRAM5_TRUNK=mlp requested but the cluster model definitions "
                f"at {_cluster_model_source} do not provide NanoGPTMLPTrunk"
            )
        GPT = NanoGPTMLPTrunk
        MODEL_PROVENANCE = {
            "source": _cluster_model_source,
            "source_description": "cluster NanoGPTMLPTrunk (position-wise MLP trunk; n-gram injection tables live)",
            "note": "loaded from the definition-only prefix of cluster train.py",
            "trunk": "mlp",
        }
    elif TRUNK_VARIANT in ("transformer", "baseline"):
        if _cluster_model_source == str(_MAIN_TRAIN_PY):
            _BaseGPT = NanoGPTOriginal

            class GPT(_BaseGPT):
                def setup_optimizer(self, **kwargs):
                    if MixedOptimizer is None:
                        raise RuntimeError("repository MixedOptimizer is unavailable")
                    return MixedOptimizer(
                        self,
                        lr=kwargs.get("nanogpt_adam_lr", 0.0006),
                        ngram_betas=kwargs.get("ngram_ve_betas", (0.0, 0.99)),
                        adam_betas=kwargs.get("adam_betas", (0.8, 0.95)),
                        weight_decay=kwargs.get("weight_decay", 0.1),
                        table_optimizer="rmsprop",
                        table_lr_scale=kwargs.get("ngram_ve_lr_scale", 128.0),
                        table_betas=kwargs.get("ngram_ve_betas", (0.0, 0.99)),
                    )

                def num_scaling_params(self):
                    total = sum(parameter.numel() for parameter in self.parameters())
                    return {"total": total}
        else:
            GPT = NanoGPTOriginal
        MODEL_PROVENANCE = {
            "source": _cluster_model_source,
            "source_description": "repository NanoGPT with n-gram injection tables"
            if _cluster_model_source == str(_MAIN_TRAIN_PY)
            else "launcher-synced repository NanoGPT with n-gram injection tables",
            "note": "loaded from the repository model definition"
            if _cluster_model_source == str(_MAIN_TRAIN_PY)
            else "loaded from the launcher-synced repository model",
            "trunk": "transformer",
        }
    else:
        raise ValueError(f"unknown NGRAM5_TRUNK: {TRUNK_VARIANT!r}")
else:
    raise RuntimeError(
        "cannot load the repository model from code/train.py or the "
        "launcher-synced train.py"
    )

__all__ = ["GPT", "GPTConfig", "MODEL_PROVENANCE", "NanoGPTOriginal", "NanoGPTMLPTrunk", "TRUNK_VARIANT"]
