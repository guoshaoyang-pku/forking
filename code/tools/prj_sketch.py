"""prj_sketch.py -- fast deterministic random projection for 2048-dim residuals.

Deliverable for the n-gram distillation pipeline: compress one 2048-dim teacher
residual vector to `out_dim` (default 128) dims fast enough to ride along inside
a CPU-bound production loop, with no fitting/training and bit-identical output
for a given seed across workers/machines (same numpy version).

Public API (frozen -- another worker integrates against it):

    class Sketch:
        def __init__(self, in_dim=2048, out_dim=128, nonzeros=16, seed=20261021,
                     kind="dense", backend="numpy")
        def info(self) -> dict
        @staticmethod
        def from_info(info) -> "Sketch"
        def project(self, X) -> np.ndarray            # (n, in_dim) f32 -> (n, out_dim) f32
        def encode_i8(self, S) -> (int8, float16)     # symmetric per-row quant
        @staticmethod
        def decode_i8(codes, scales) -> np.ndarray    # float32

Kinds
-----
  "dense"  (default): X @ P with a dense float32 matrix.  For in_dim=2048,
            out_dim=128 the matrix is only 1 MB (L2-resident) and BLAS uses
            AVX512, so this is the FASTEST numpy variant measured on this box
            (median 82 ms at n=28000) and also the most faithful (distance-vector
            Pearson 0.966 at od=128).  With backend="torch" the matmul runs in
            bf16 on avx512_bf16 (median 5.1 ms at 8 threads, 2.1 ms at 32).
  "sparse": sparse Rademacher JL (Li/Achlioptas).  Each of the `out_dim` output
            dims is the signed sum of `nonzeros` distinct randomly chosen input
            coords.  Gather via a precomputed int16 index array (out_dim,
            nonzeros) + float32 signs, using np.take (8x faster than X[:, idx]).
            Cost ~ n * out_dim * nonzeros MACs, no dense matrix.  Measured
            slower than dense here (median 109 ms at nz=16) and less faithful
            (Pearson 0.844); kept for comparison / memory-constrained use.
  "fft":    fast JL: random sign flip, FFT along the input axis, take the first
            out_dim/2 real + out_dim/2 imag components (scaled).  O(n * in_dim
            log in_dim); measured 226 ms, faithful (Pearson 0.965).

Determinism
-----------
All randomness comes from numpy's PCG64 seeded with `seed`; index/sign arrays
are built once in __init__ and cached.  project() allocates only the output
(plus the gather intermediate), never the whole projection matrix.

Thread-friendliness
-------------------
The numpy paths are pure numpy and hold no locks; safe to call from one thread
while other threads run the aggregation.  The torch backend (kind="dense",
backend="torch") uses whatever torch thread count the caller has set; it is
opt-in and never changes torch's thread pool itself.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Sketch"]

_DEFAULT_SEED = 20261021


class Sketch:
    """Deterministic random projection 2048 -> out_dim."""

    def __init__(self, in_dim: int = 2048, out_dim: int = 128,
                 nonzeros: int = 16, seed: int = _DEFAULT_SEED,
                 kind: str = "dense", backend: str = "numpy"):
        in_dim = int(in_dim)
        out_dim = int(out_dim)
        nonzeros = int(nonzeros)
        seed = int(seed)
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim and out_dim must be positive")
        if kind not in ("sparse", "dense", "fft"):
            raise ValueError(f"unknown kind {kind!r}")
        if backend not in ("numpy", "torch"):
            raise ValueError(f"unknown backend {backend!r}")
        if kind == "sparse":
            if nonzeros <= 0:
                raise ValueError("nonzeros must be positive")
            if nonzeros > in_dim:
                raise ValueError("nonzeros must be <= in_dim")
        if kind == "fft" and out_dim % 2 != 0:
            raise ValueError("fft kind requires an even out_dim")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonzeros = nonzeros
        self.seed = seed
        self.kind = kind
        self.backend = backend

        # scaling: JL keeps E||Sx||^2 == ||x||^2.
        #  sparse: each output coord is a sum of `nonzeros` signed entries, so
        #    E||Sx||^2 = out_dim * nonzeros * (1/nonzeros) * E[x^2]
        #              = (out_dim/in_dim) ||x||^2 -> scale by sqrt(in_dim/(out_dim*nonzeros)).
        #  dense:  P has iid N(0,1) entries -> E||Px||^2 = out_dim*||x||^2
        #    -> scale by 1/sqrt(out_dim).
        #  fft:    rfft of a sign-flipped (white) vector.  For E[x^2]=1 the bins
        #    k=1..N/2-1 have E|F_k|^2 = in_dim, and bin 0 is real-only.  Taking
        #    `half`=out_dim/2 real + `half` imag components gives
        #    E||out||^2 = half * in_dim * E[x^2] = half * ||x||^2
        #    -> scale by 1/sqrt(half).
        if kind == "sparse":
            self.scaling = float(np.sqrt(in_dim / (out_dim * nonzeros)))
        elif kind == "dense":
            self.scaling = float(1.0 / np.sqrt(out_dim))
        else:  # fft
            self.scaling = float(1.0 / np.sqrt(out_dim // 2))

        self._idx = None      # (out_dim, nonzeros) int16  (sparse)
        self._idx_flat = None  # (out_dim*nonzeros,) int16  (sparse, for np.take)
        self._signs = None    # (out_dim, nonzeros) float32 (sparse)
        self._P = None        # (in_dim, out_dim) float32  (dense)
        self._sign_vec = None  # (in_dim,) float32         (fft)
        self._tP = None       # torch bf16 (in_dim, out_dim) (dense, torch backend)
        self._build()

    # -- construction ------------------------------------------------------- #
    def _build(self):
        rng = np.random.default_rng(self.seed)
        if self.kind == "sparse":
            # For each output dim, choose `nonzeros` distinct input coords.
            # argsort of random keys gives distinct coords without rejection.
            keys = rng.random((self.out_dim, self.in_dim), dtype=np.float64)
            idx = np.argsort(keys, axis=1)[:, : self.nonzeros]
            idx = np.ascontiguousarray(idx, dtype=np.int16)
            signs = rng.integers(0, 2, size=(self.out_dim, self.nonzeros))
            signs = np.where(signs == 0, -1.0, 1.0).astype(np.float32)
            self._idx = idx
            self._idx_flat = np.ascontiguousarray(idx.ravel())
            self._signs = signs
        elif self.kind == "dense":
            P = rng.standard_normal((self.in_dim, self.out_dim)).astype(np.float32)
            self._P = P
            if self.backend == "torch":
                import torch
                self._tP = torch.from_numpy(P).to(torch.bfloat16)
        else:  # fft
            s = rng.integers(0, 2, size=(self.in_dim,))
            self._sign_vec = np.where(s == 0, -1.0, 1.0).astype(np.float32)

    # -- description -------------------------------------------------------- #
    def info(self) -> dict:
        """Reproducible description; feed to from_info() to rebuild exactly."""
        return {
            "kind": self.kind,
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "nonzeros": self.nonzeros,
            "seed": self.seed,
            "scaling": self.scaling,
            "backend": self.backend,
        }

    @staticmethod
    def from_info(info) -> "Sketch":
        return Sketch(in_dim=int(info["in_dim"]), out_dim=int(info["out_dim"]),
                      nonzeros=int(info.get("nonzeros", 16)),
                      seed=int(info["seed"]), kind=str(info.get("kind", "dense")),
                      backend=str(info.get("backend", "numpy")))

    # -- projection --------------------------------------------------------- #
    def project(self, X):
        """(n, in_dim) float32 -> (n, out_dim) float32.

        Accepts a numpy array (always) or, when backend="torch", a torch tensor
        (returned as a torch tensor, avoiding a host round-trip).
        """
        if self.kind == "dense" and self.backend == "torch":
            import torch
            if isinstance(X, torch.Tensor):
                Xt = X
                if Xt.ndim == 1:
                    Xt = Xt[None, :]
                if Xt.shape[1] != self.in_dim:
                    raise ValueError(f"expected in_dim={self.in_dim}, got {Xt.shape[1]}")
                out = (Xt.to(torch.bfloat16) @ self._tP).to(torch.float32)
                return out * self.scaling
            Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
            if Xt.ndim == 1:
                Xt = Xt[None, :]
            if Xt.shape[1] != self.in_dim:
                raise ValueError(f"expected in_dim={self.in_dim}, got {Xt.shape[1]}")
            out = (Xt.to(torch.bfloat16) @ self._tP).to(torch.float32)
            return (out * self.scaling).numpy()

        X = np.asarray(X)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.in_dim:
            raise ValueError(f"expected in_dim={self.in_dim}, got {X.shape[1]}")

        Xf = np.ascontiguousarray(X, dtype=np.float32)

        if self.kind == "sparse":
            # np.take along axis=1 is ~8x faster than X[:, idx] fancy indexing
            # (contiguous gather vs advanced-index copy).  Then contract the
            # (n, out_dim, nonzeros) block with the signs in fp32.
            n = Xf.shape[0]
            g = np.take(Xf, self._idx_flat, axis=1).reshape(n, self.out_dim, self.nonzeros)
            out = np.einsum("nok,ok->no", g, self._signs, optimize=True)
            out *= self.scaling
            return out.astype(np.float32, copy=False)

        if self.kind == "dense":
            out = Xf @ self._P
            out *= self.scaling
            return out.astype(np.float32, copy=False)

        # fft: sign flip, rfft, take out_dim/2 real + out_dim/2 imag bins.
        half = self.out_dim // 2
        Y = Xf * self._sign_vec
        F = np.fft.rfft(Y, axis=1)                     # (n, in_dim//2+1) complex
        out = np.empty((Xf.shape[0], self.out_dim), dtype=np.float32)
        out[:, :half] = F[:, :half].real
        out[:, half:] = F[:, :half].imag
        out *= self.scaling
        return out

    # -- int8 quantization -------------------------------------------------- #
    def encode_i8(self, S):
        """Symmetric per-row int8 quantization.

        scale = max|row| / 127 ; codes = round(row / scale) clipped to [-127,127].
        Returns (int8 codes, float16 scales).  A zero row gets scale 0 and codes 0.
        """
        S = np.asarray(S, dtype=np.float32)
        if S.ndim == 1:
            S = S[None, :]
        amax = np.abs(S).max(axis=1)
        scale = (amax / 127.0).astype(np.float16)
        denom = np.where(amax > 0, amax, 1.0).astype(np.float32)
        codes = np.rint(S * (127.0 / denom)[:, None])
        codes = np.clip(codes, -127.0, 127.0).astype(np.int8)
        return codes, scale

    @staticmethod
    def decode_i8(codes, scales) -> np.ndarray:
        codes = np.asarray(codes, dtype=np.float32)
        scales = np.asarray(scales, dtype=np.float32)
        if codes.ndim == 1:
            codes = codes[None, :]
        return codes * scales[:, None]
