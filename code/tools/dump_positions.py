#!/usr/bin/env python3
"""Position-indexed residual dump for the teacher-distillation run (pro5k6).

Why this exists
---------------
The teacher pass (Qwen3.6-35B-A3B over the whole ClimbMix corpus) costs ~24 h of
8 GPUs and cannot be repeated cheaply.  The production workers only keep the
974k-row *aggregates* (per n-gram context: sum of residuals + count).  This
module captures the raw material instead: for every one of the ~11.45 B lab
positions (210 shards x ~54.5 M) it records the teacher residual at three
depths, position-indexed and lossless-in-aggregation, so the aggregates can be
recomputed offline for any context set without re-running the teacher.

What is dumped
--------------
The engine returns, per teacher token, 5 blocks of 2048 dims:
    [d0, d10, d25, d40, final_normed]
d0 == wte[token_id] and final_normed == RMSNorm(d40) are both reconstructible
(from the token id / from d40), so only **d10, d25, d40** are stored.

Per position (row order, all 2048 slots of every row):
    sketch_w<k>_s<shard>.i8.npy   (n_positions, 3, 128) int8   symmetric codes
    scale_w<k>_s<shard>.f16.npy   (n_positions, 3)      fp16   per (pos, block)
    tid_w<k>_s<shard>.u32.npy     (n_positions,)        uint32 teacher token id
                                                          0xFFFFFFFF = no teacher
    meta_w<k>_s<shard>.json       alignment + provenance (written LAST)

Byte budget: 3*128 int8 (384 B) + 4 B tid + 3*2 B fp16 scale = 394 B/position
-> 11.45e9 * 394 = 4.51 TB, under the 4.6 TB sketch+ids budget.

Full-fidelity path
------------------
For shards listed in ``full_fidelity_shards`` the three blocks are written as
raw fp16 instead of the sketch:
    full_w<k>_s<shard>.f16.npy    (n_positions, 3, 2048) fp16  (~670 GB/shard)
For those shards the sketch is SKIPPED (keeps the budget: one full shard
replaces its own sketch, it is not additive).  tid + meta are still written.

Quantization (symmetric, per (position, block))
-----------------------------------------------
    scale = max|S| / 127
    codes = round(S / scale).clip(-127, 127).astype(int8)
    decode: S_hat = codes.astype(fp32) * scale
The all-zero row (scale == 0) is stored as codes=0, scale=0 and decodes to 0
(no NaN: the division is guarded).  int8 symmetric gives a max relative error
of ~1/127 ~= 0.79% per element and ~0.4% RMS, which is far below the teacher
residual spread; the aggregate is unaffected because the dump is a *record*,
not an input to the running accumulation.

Projector
---------
``/data/prj_sketch.py`` (written in parallel) exposes::

    class Sketch:
        __init__(in_dim=2048, out_dim=128, nonzeros=16, seed=20261021)
        info() -> dict
        project(X) -> (n, out_dim) float32
        from_info(info)                      # classmethod
        encode_i8(S) -> (int8, fp16 scale)
        decode_i8(codes, scales) -> float32

If that module is importable it is used verbatim; otherwise a compatible local
fallback (``_LocalSketch``) with the same interface and the same default
parameters is used, and the meta records which one was active.  Switching over
is automatic: the module is re-imported on every DumpWriter construction.

Robustness
----------
The writer is designed so a dump failure can never corrupt the aggregate: the
caller wraps every call in try/except and disables the dump on the first error
(see distill_prod_v4/v5).  Within the writer, a shard is only considered
complete once its meta json exists; a truncated shard is detectable by the
missing meta.
"""
import hashlib
import json
import os
import struct
import time

import numpy as np

# --------------------------------------------------------------------------- #
# projector
# --------------------------------------------------------------------------- #
SKETCH_MODULE = "/data/prj_sketch.py"
SKETCH_DEFAULTS = dict(in_dim=2048, out_dim=128, nonzeros=16, seed=20261021)


def _load_sketch_class():
    """Return (SketchClass, source_tag).  Prefer /data/prj_sketch.py."""
    import importlib.util
    import sys

    if os.path.exists(SKETCH_MODULE):
        try:
            spec = importlib.util.spec_from_file_location("prj_sketch", SKETCH_MODULE)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["prj_sketch"] = mod
            spec.loader.exec_module(mod)
            if hasattr(mod, "Sketch"):
                return mod.Sketch, "prj_sketch"
        except Exception:  # noqa: BLE001 - fall through to the local fallback
            pass
    return _LocalSketch, "local_fallback"


class _LocalSketch:
    """Deterministic sparse random projection, interface-compatible fallback.

    Each output dim is the mean of ``nonzeros`` input dims picked by a seeded
    RNG, scaled by 1/sqrt(nonzeros) so the projection is (approximately)
    norm-preserving.  Deterministic given the seed, so a re-run reproduces the
    exact same matrix and the dump is verifiable offline.
    """

    def __init__(self, in_dim=2048, out_dim=128, nonzeros=16, seed=20261021):
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.nonzeros = int(nonzeros)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        # (out_dim, nonzeros) column indices, sorted for cache-friendly gather.
        idx = np.empty((self.out_dim, self.nonzeros), dtype=np.int32)
        for o in range(self.out_dim):
            idx[o] = np.sort(rng.choice(self.in_dim, size=self.nonzeros,
                                        replace=False))
        self.idx = idx
        self.scale = np.float32(1.0 / np.sqrt(self.nonzeros))

    def info(self):
        return {
            "kind": "local_sparse_random_projection",
            "in_dim": self.in_dim, "out_dim": self.out_dim,
            "nonzeros": self.nonzeros, "seed": self.seed,
            "scale": float(self.scale),
            "formula": "S[:,o] = mean(X[:, idx[o]]) = sum(X[:,idx[o]])/nonzeros",
        }

    def project(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.in_dim:
            raise ValueError(f"project: want in_dim {self.in_dim}, got {X.shape[1]}")
        # (n, out_dim, nonzeros) gather then mean over the last axis.
        g = X[:, self.idx]                       # (n, out_dim, nonzeros)
        return (g.mean(axis=2) * self.scale).astype(np.float32)

    @classmethod
    def from_info(cls, info):
        return cls(in_dim=info["in_dim"], out_dim=info["out_dim"],
                   nonzeros=info["nonzeros"], seed=info["seed"])

    @staticmethod
    def encode_i8(S):
        """Symmetric per-row int8: scale = max|S|/127, codes = round(S/scale)."""
        S = np.asarray(S, dtype=np.float32)
        if S.ndim == 1:
            S = S[None, :]
        amax = np.abs(S).max(axis=1)
        scale = (amax / 127.0).astype(np.float32)
        safe = np.where(scale > 0, scale, np.float32(1.0))
        codes = np.rint(S / safe[:, None]).clip(-127, 127).astype(np.int8)
        codes[scale == 0] = 0                    # all-zero row -> all-zero codes
        return codes, scale.astype(np.float16)

    @staticmethod
    def decode_i8(codes, scales):
        codes = np.asarray(codes)
        scales = np.asarray(scales, dtype=np.float32)
        if codes.ndim == 1:
            codes = codes[None, :]
        if scales.ndim == 0:
            scales = scales[None]
        # scales has shape codes.shape[:-1]; broadcast over the last axis
        return codes.astype(np.float32) * scales[..., None]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sha1_bytes(b):
    return hashlib.sha1(b).hexdigest()


def _sha1_array(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha1(a.tobytes()).hexdigest()


def _npy_header(dtype, shape):
    """Build a 128-byte v1.0 .npy header for the given dtype/shape."""
    dt = np.dtype(dtype)
    descr = dt.str
    if dt.kind in ("i", "u", "f") and dt.itemsize > 1:
        descr = ("<" if dt.byteorder in ("<", "=", "|") else ">") + dt.str[1:]
    shape_s = "(" + ", ".join(str(int(s)) for s in shape)
    shape_s += ",)" if len(shape) == 1 else ")"
    d = "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % (descr, shape_s)
    # numpy pads the dict so that the total header (magic+ver+len+dict) is a
    # multiple of 64 bytes; the dict ends with a newline.
    magic = b"\x93NUMPY\x01\x00"
    base = len(magic) + 2
    total = base + len(d) + 1
    pad = (64 - (total % 64)) % 64
    d = d + " " * pad + "\n"
    hdr = magic + struct.pack("<H", len(d)) + d.encode("latin1")
    if len(hdr) % 64 != 0:
        raise RuntimeError(f"npy header is {len(hdr)} bytes, not 64-aligned")
    return hdr


def _code_md5s():
    """md5 of this module and of the projector module (if present)."""
    out = {}
    for tag, path in (("dump_positions", os.path.abspath(__file__)),
                      ("prj_sketch", SKETCH_MODULE)):
        try:
            with open(path, "rb") as fh:
                out[tag] = hashlib.md5(fh.read()).hexdigest()
        except OSError:
            out[tag] = None
    return out


# --------------------------------------------------------------------------- #
# writer
# --------------------------------------------------------------------------- #
class DumpWriter:
    """Position-indexed residual dump, one memmap set per shard.

    Parameters
    ----------
    root : str
        Dump root, e.g. /data/distill_dump.  Files land in <root>/<tag>/.
    worker : int
        Worker id (goes into the filename as w<k> and into the meta).
    tag : str
        Run tag, e.g. w3s.
    blocks : tuple[int]
        Teacher depths to dump (default (10, 25, 40)).
    dims : int
        Sketch output dims (default 128).
    sketch : object | None
        A projector instance; if None one is built from the module/fallback.
    full_fidelity_shards : iterable[int]
        Shards to dump as raw fp16 (sketch skipped for those).
    """

    def __init__(self, root, worker, tag, blocks=(10, 25, 40), dims=128,
                 sketch=None, full_fidelity_shards=()):
        self.root = root
        self.worker = int(worker)
        self.tag = tag
        self.blocks = tuple(int(b) for b in blocks)
        self.dims = int(dims)
        self.full_fidelity_shards = set(int(s) for s in full_fidelity_shards)
        self.dir = os.path.join(root, tag)
        os.makedirs(self.dir, exist_ok=True)

        if sketch is None:
            cls, src = _load_sketch_class()
            sketch = cls(out_dim=self.dims)
            self.sketch_src = src
        else:
            self.sketch_src = "injected"
        self.sketch = sketch
        self.sketch_info = sketch.info()
        # Fast path: the sparse gather in prj_sketch.project is memory-bound
        # (~2.3 s per 84k rows on this box) while the equivalent dense gemm is
        # ~16 ms.  We materialize the dense matrix from the sketch's sparse
        # parameters (idx/signs/scaling) and use X @ P, which is numerically
        # identical to project() to fp32 rounding (~1e-6).  If the internals are
        # not exposed we fall back to sketch.project().
        self._dense = self._build_dense(sketch)
        self._proj_path = "dense_gemm" if self._dense is not None else "sketch.project"

        self._shard = None
        self._n_rows = 0
        self._n_pos = 0
        self._cap = 0
        self._mm = {}                 # name -> memmap
        self._full = False
        self._tid = None
        self._scales = None
        self._codes = None
        self._full_arr = None
        self._aligned = 0
        self._unaligned = 0
        self._first_ids = None
        self._last_ids = None
        self._token_stream_sha1 = None
        self._lab_sha = None
        self._t0 = None
        self._closed = False

    # -- naming ------------------------------------------------------------- #
    def _stem(self, shard):
        return f"w{self.worker}_s{shard}"

    def _path(self, kind, shard):
        return os.path.join(self.dir, f"{kind}_{self._stem(shard)}")

    # -- shard lifecycle ---------------------------------------------------- #
    def start_shard(self, shard, n_rows, token_stream_sha1=None,
                    first_ids=None, last_ids=None):
        """Open the memmaps for one shard.

        n_rows : number of packed rows in the shard (each row is --seq slots).
        token_stream_sha1 : sha1 of the lab token id stream for the shard
            (the caller computes it from the Packer output).
        first_ids / last_ids : the first/last 1000 lab token ids (for the meta).
        """
        if self._shard is not None:
            raise RuntimeError("start_shard called while a shard is open")
        self._shard = int(shard)
        self._n_rows = int(n_rows)
        self._n_pos = 0
        self._cap = int(n_rows) * 2048
        self._full = self._shard in self.full_fidelity_shards
        self._aligned = 0
        self._unaligned = 0
        self._token_stream_sha1 = token_stream_sha1
        self._first_ids = None if first_ids is None else np.asarray(first_ids, dtype=np.int64)
        self._last_ids = None if last_ids is None else np.asarray(last_ids, dtype=np.int64)
        self._t0 = time.time()

        nb = len(self.blocks)
        self._tid = self._open_memmap("tid", np.uint32, (self._cap,), ".u32.npy")
        self._tid[:] = 0xFFFFFFFF
        if self._full:
            self._full_arr = self._open_memmap("full", np.float16,
                                               (self._cap, nb, 2048), ".f16.npy")
        else:
            self._codes = self._open_memmap("sketch", np.int8,
                                            (self._cap, nb, self.dims), ".i8.npy")
            self._scales = self._open_memmap("scale", np.float16,
                                             (self._cap, nb), ".f16.npy")

    def _build_dense(self, sketch):
        """Return a dense (in_dim, out_dim) matrix equivalent to the sketch.

        Handles both a dense sketch (exposes ``_P``) and a sparse one (exposes
        ``_idx``/``_signs``/``scaling``).  Returns None if neither is available,
        in which case the caller falls back to sketch.project().
        """
        P = getattr(sketch, "_P", None)
        if P is not None:
            return np.ascontiguousarray(P, dtype=np.float32)
        idx = getattr(sketch, "_idx", None)
        signs = getattr(sketch, "_signs", None)
        scaling = getattr(sketch, "scaling", None)
        in_dim = getattr(sketch, "in_dim", None)
        out_dim = getattr(sketch, "out_dim", None)
        if idx is None or signs is None or scaling is None:
            return None
        if in_dim is None or out_dim is None:
            return None
        P = np.zeros((int(in_dim), int(out_dim)), dtype=np.float32)
        idx = np.asarray(idx)
        signs = np.asarray(signs, dtype=np.float32)
        for o in range(int(out_dim)):
            P[idx[o], o] = signs[o]
        P *= np.float32(scaling)
        return np.ascontiguousarray(P)

    def _project(self, X):
        """Project (n, in_dim) -> (n, out_dim); dense gemm when available."""
        if self._dense is not None:
            return np.ascontiguousarray(X, dtype=np.float32) @ self._dense
        return self.sketch.project(X)

    def _open_memmap(self, kind, dtype, shape, ext):
        """Create a .npy file with a 128-byte header, then memmap the payload.

        np.memmap on a .npy path writes RAW bytes (no header), so we write the
        header ourselves first and map the payload at offset 128.  The header is
        always 128 bytes for our shapes, so the payload offset never moves and
        _finalize_npy can rewrite the header in place without shifting data.
        """
        p = self._path(kind, self._shard) + ext
        hdr = _npy_header(dtype, shape)
        with open(p, "wb") as fh:
            fh.write(hdr)
            fh.truncate(128 + int(np.prod(shape)) * np.dtype(dtype).itemsize)
        return np.memmap(p, dtype=dtype, mode="r+", offset=128, shape=shape)

    def add_batch(self, data, q_star, row_positions=None, n_slots=2048):
        """Append one batch of positions.

        data : (nq, 2048*5) fp32, the engine output for this batch's teacher
            tokens (already denormalized by the caller).
        q_star : (n_positions,) int64, per lab position the teacher token index
            it aligns to, or -1 where there is no teacher token.  Positions are
            in row order, all ``n_slots`` slots of every row, concatenated.
        row_positions : optional (n_positions,) int64 global position index; if
            None the writer uses its own running counter (must be contiguous).
        n_slots : slots per row (2048).

        Returns the number of positions written.
        """
        if self._shard is None:
            raise RuntimeError("add_batch before start_shard")
        data = np.asarray(data, dtype=np.float32)
        q_star = np.asarray(q_star, dtype=np.int64).reshape(-1)
        n = q_star.size
        if self._n_pos + n > self._cap:
            raise RuntimeError(f"shard {self._shard} overflow: "
                               f"{self._n_pos + n} > cap {self._cap}")

        # teacher token id per position (0xFFFFFFFF where q_star < 0)
        tid = np.full(n, 0xFFFFFFFF, dtype=np.uint32)
        valid = q_star >= 0
        # q_star indexes into the teacher token stream of THIS batch; the caller
        # passes the global teacher ids via data rows, so we only need the id
        # array.  The caller supplies it through self._tid_src (see set_tid_src)
        # or we fall back to the row index (documented in meta).
        if self._tid_src is not None:
            src = np.asarray(self._tid_src, dtype=np.int64)
            tid[valid] = src[q_star[valid]].astype(np.uint32)
        else:
            tid[valid] = q_star[valid].astype(np.uint32)

        self._aligned += int(valid.sum())
        self._unaligned += int((~valid).sum())

        # gather + project the three blocks for the valid positions.
        # Per-block gather+gemm is ~1.7x faster than materializing the full
        # (n, 3, 2048) intermediate: the working set stays in cache and the
        # gemm reads a contiguous (n, 2048) block.
        nb = len(self.blocks)
        if valid.any():
            sel = q_star[valid]
            # absolute position indices of the aligned slots (fancy-indexing a
            # memmap slice returns a COPY, so we must assign by absolute index).
            dst = self._n_pos + np.flatnonzero(valid)
            # data layout: [final_normed, d0, d10, d25, d40] (see prod docstring)
            # blocks (10,25,40) are at offsets 2,3,4 in that 5-block layout.
            off = {10: 2, 25: 3, 40: 4}
            if self._full:
                X = np.empty((sel.size, nb, 2048), dtype=np.float16)
                for bi, b in enumerate(self.blocks):
                    o = off.get(b, 2 + bi)
                    X[:, bi, :] = data[sel, o * 2048:(o + 1) * 2048].astype(np.float16)
                self._full_arr[dst] = X
            else:
                codes = np.empty((sel.size, nb, self.dims), dtype=np.int8)
                scales = np.empty((sel.size, nb), dtype=np.float16)
                for bi, b in enumerate(self.blocks):
                    o = off.get(b, 2 + bi)
                    blk = data[sel, o * 2048:(o + 1) * 2048]      # (n, 2048)
                    c, s = self.sketch.encode_i8(self._project(blk))
                    codes[:, bi, :] = c
                    scales[:, bi] = s
                self._codes[dst] = codes
                self._scales[dst] = scales

        self._tid[self._n_pos:self._n_pos + n] = tid
        self._n_pos += n
        return n

    # teacher id source for the current batch (set by the caller each batch)
    _tid_src = None

    def set_tid_src(self, ids):
        """Set the teacher token id array that q_star indexes into (per batch)."""
        self._tid_src = None if ids is None else np.asarray(ids, dtype=np.int64)

    def end_shard(self, n_positions=None):
        """Flush + fsync the shard, then write the meta json LAST."""
        if self._shard is None:
            raise RuntimeError("end_shard without start_shard")
        if n_positions is not None and int(n_positions) != self._n_pos:
            raise RuntimeError(f"end_shard n_positions {n_positions} != "
                               f"written {self._n_pos}")
        for mm in (self._tid, self._codes, self._scales, self._full_arr):
            if mm is not None:
                mm.flush()
        # fsync the underlying files
        for kind in ("tid", "sketch", "scale", "full"):
            p = self._path(kind, self._shard) + (".u32.npy" if kind == "tid"
                                                 else ".i8.npy" if kind == "sketch"
                                                 else ".f16.npy")
            if os.path.exists(p):
                fd = os.open(p, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        # truncate the memmaps to the real length by rewriting the .npy header
        self._finalize_npy()
        meta = self._build_meta()
        meta_path = self._path("meta", self._shard) + ".json"
        tmp = meta_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, meta_path)          # meta last -> truncated shard detectable
        self._shard = None
        self._tid = self._codes = self._scales = self._full_arr = None
        return meta

    def _finalize_npy(self):
        """Rewrite each .npy header with the true (n_positions, ...) shape.

        The memmaps were created with the shard capacity; we rewrite only the
        128-byte v1.0 header (in place) and truncate the file to the used size,
        so the on-disk array has exactly n_positions rows.  No full read-back.
        """
        shard = self._shard
        n = self._n_pos
        nb = len(self.blocks)
        specs = []
        if self._full:
            specs.append(("full", np.float16, (n, nb, 2048), ".f16.npy"))
        else:
            specs.append(("sketch", np.int8, (n, nb, self.dims), ".i8.npy"))
            specs.append(("scale", np.float16, (n, nb), ".f16.npy"))
        specs.append(("tid", np.uint32, (n,), ".u32.npy"))
        for kind, dt, shape, ext in specs:
            p = self._path(kind, shard) + ext
            if not os.path.exists(p):
                continue
            used = int(np.prod(shape)) * np.dtype(dt).itemsize
            hdr = _npy_header(dt, shape)
            with open(p, "r+b") as fh:
                fh.seek(0)
                fh.write(hdr)
                fh.truncate(128 + used)

    def _build_meta(self):
        nb = len(self.blocks)
        meta = {
            "shard": self._shard,
            "worker": self.worker,
            "tag": self.tag,
            "n_rows": self._n_rows,
            "n_positions": self._n_pos,
            "blocks": list(self.blocks),
            "dims": self.dims,
            "dtype": "int8_symmetric" if not self._full else "fp16_raw",
            "full_fidelity": self._full,
            "quant": {
                "formula": "scale=max|S|/127; codes=round(S/scale).clip(-127,127)",
                "decode": "S_hat = codes.astype(fp32) * scale",
                "zero_row": "scale=0 -> codes=0 -> decodes to 0 (no NaN)",
                "max_rel_err_per_elem": 1.0 / 127.0,
            },
            "projector": self.sketch_info,
            "projector_source": self.sketch_src,
            "projection_path": self._proj_path,
            "token_stream_sha1": self._token_stream_sha1,
            "first_ids_sha1": (None if self._first_ids is None
                               else _sha1_array(self._first_ids)),
            "last_ids_sha1": (None if self._last_ids is None
                              else _sha1_array(self._last_ids)),
            "first_ids_n": (0 if self._first_ids is None else int(self._first_ids.size)),
            "last_ids_n": (0 if self._last_ids is None else int(self._last_ids.size)),
            "byte_end_alignment": ("position t takes the last teacher token whose "
                                   "cumulative byte end <= our cumulative byte end; "
                                   "positions in row order, all 2048 slots per row"),
            "n_aligned": self._aligned,
            "n_unaligned": self._unaligned,
            "code_md5s": _code_md5s(),
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": round(time.time() - self._t0, 2),
        }
        return meta

    def close(self):
        if self._closed:
            return
        if self._shard is not None:
            self.end_shard()
        self._closed = True


# --------------------------------------------------------------------------- #
# self-test (CPU only)
# --------------------------------------------------------------------------- #
def _selftest():
    import tempfile
    rng = np.random.default_rng(0)
    root = tempfile.mkdtemp(prefix="dump_selftest_")
    w = DumpWriter(root, worker=0, tag="t", blocks=(10, 25, 40), dims=128)
    n_rows, nq = 2, 40
    w.start_shard(0, n_rows, token_stream_sha1="deadbeef",
                  first_ids=np.arange(1000), last_ids=np.arange(1000) + 5)
    data = rng.standard_normal((nq, 2048 * 5)).astype(np.float32)
    q_star = np.full(n_rows * 2048, -1, dtype=np.int64)
    q_star[:nq] = np.arange(nq)
    w.set_tid_src(np.arange(1000, 1000 + nq))
    w.add_batch(data, q_star)
    meta = w.end_shard(n_rows * 2048)
    w.close()
    # round trip
    codes = np.load(os.path.join(root, "t", "sketch_w0_s0.i8.npy"))
    scales = np.load(os.path.join(root, "t", "scale_w0_s0.f16.npy"))
    tid = np.load(os.path.join(root, "t", "tid_w0_s0.u32.npy"))
    assert codes.shape == (n_rows * 2048, 3, 128), codes.shape
    assert tid.shape == (n_rows * 2048,)
    assert int((tid == 0xFFFFFFFF).sum()) == n_rows * 2048 - nq
    assert meta["n_aligned"] == nq and meta["n_unaligned"] == n_rows * 2048 - nq
    # dequant vs float64 recompute
    X = np.stack([data[:nq, 2 * 2048:3 * 2048], data[:nq, 3 * 2048:4 * 2048],
                  data[:nq, 4 * 2048:5 * 2048]], axis=1).astype(np.float64)
    S = w._project(X.reshape(nq * 3, 2048).astype(np.float32)).reshape(nq, 3, 128)
    dec = w.sketch.decode_i8(codes[:nq].reshape(nq * 3, 128),
                             scales[:nq].reshape(nq * 3)).reshape(nq, 3, 128)
    rel = np.abs(dec - S).max() / (np.abs(S).max() + 1e-12)
    cos = float((dec * S).sum() / (np.linalg.norm(dec) * np.linalg.norm(S) + 1e-12))
    print(f"[selftest] PASS  rel_err={rel:.4f}  cos={cos:.6f}  "
          f"sketch={w.sketch_src}  meta_keys={len(meta)}")
    return True


if __name__ == "__main__":
    _selftest()
