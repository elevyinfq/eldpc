#!/usr/bin/env python3
"""The Relay-BP unit-exactness test (a tool; the library does no I/O).

One memory-BP leg (fixed gamma per column, the leg's input marginals = the priors) is run on the
library's own kernel through the test-only FASTBP_TEST_RELAY_DUMP build, and every message nu(t) and
every marginal M(t), t = 1..T, is compared with an independent CPU reference in double precision
that uses this kernel's addition order:

  * check update: per row the two smallest |nu| over the row's edges (magnitude bits, strict <,
    second minimum with multiplicity) and the count of nonpositive messages; the edge's message is
    (|nu_e| == m1 ? m2 : m1) with sign ((count + nonpositive(nu_e) + syndrome) & 1), times alpha;
  * column update: t starts at the memory bias Lambda_j(t) = (1 - gamma_j) lambda_j + gamma_j
    M_j(t-1) (paper Eq. 4), then every edge in increasing row order: pre_e = t; t += mu_e; the
    marginal M = t; the backward pass b2c_e = pre_e + t2, t2 += mu_e from the last edge down;
  * iteration 1's check summary comes from the priors (nu_j->i(0) = lambda_j).

The reference's sequential sums are numpy float64 cumulative sums (strictly sequential per column)
and per-row segmented sorts for the two minima; the tool self-checks the cumsum against a pure
Python loop first and fails if they ever differ.

Messages are read in CSC edge order through the test export's gather; the fused kernel runs with
its b2c window off (the same values in global memory, the same arithmetic).

Fresh shots: stim.Circuit.from_file(circuit.stim).compile_detector_sampler(seed).sample(n,
separate_observables=True, bit_packed=True), a seed given on the command line (generate a new one
with `python3 -c 'import secrets; print(secrets.randbits(64))'`).

usage: python3 -I t_relay_unit.py --lib <relaydump .so> --problem <corpus problem dir>
                                  --seed <64-bit> [--shots N] [--iters T]
exit 0 iff every message and every marginal of every tested shot is bit-identical.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))
from dbench.abi import AbiDecoder, AbiLibrary                 # noqa: E402
from dbench.formats import load_problem_dir                   # noqa: E402

MASK = np.uint64(0x7FFFFFFFFFFFFFFF)
INF_BITS = np.uint64(0x7FF0000000000000)
DBL_MAX_BITS = 0x7FEFFFFFFFFFFFFF


def f64_bits(a):
    return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)


def bits_f64(a):
    return np.ascontiguousarray(a, dtype=np.uint64).view(np.float64)


def nonpositive_bits(b):
    """v <= 0.0 for every bit pattern, as the kernel's integer form: NaN never, +-0 yes."""
    m = b & MASK
    sign = (b >> np.uint64(63)) != np.uint64(0)
    zero = m == np.uint64(0)
    return ((m <= INF_BITS) & (sign | zero)).astype(np.int64)


def splitmix(z: int) -> int:
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def cumsum_selfcheck(rng) -> tuple[bool, str]:
    """numpy's float64 cumsum along an axis must be the strict sequential sum the kernel does."""
    worst = 0
    for _ in range(200):
        d = int(rng.integers(1, 12))
        row = rng.standard_normal(d) * 10.0
        got = np.cumsum(row)
        acc = 0.0
        want = []
        for v in row:
            acc = acc + float(v)
            want.append(acc)
        want = np.array(want)
        if not np.array_equal(got.view(np.uint64), want.view(np.uint64)):
            return False, f"cumsum differs on {row!r}"
        worst = max(worst, d)
    return True, f"{worst} columns wide"


class Ref:
    """The paper's DMem-BP leg in this kernel's double-precision addition order."""

    def __init__(self, H, lam, syn):
        Hc = H.tocsc()
        self.m, self.n = Hc.shape
        self.cptr = np.ascontiguousarray(Hc.indptr, dtype=np.int64)
        self.rowind = np.ascontiguousarray(Hc.indices, dtype=np.int64)
        self.nnz = int(self.cptr[-1])
        self.lam = np.array(lam, dtype=np.float64)
        self.syn = np.asarray(syn, dtype=np.int64)
        # row -> edges (CSC edge ids), ascending columns
        rowptr = np.zeros(self.m + 1, dtype=np.int64)
        np.add.at(rowptr, self.rowind + 1, 1)
        self.rowptr = np.cumsum(rowptr)
        self.row_edge = np.zeros(self.nnz, dtype=np.int64)
        fill = self.rowptr[:-1].copy()
        for j in range(self.n):
            for k in range(self.cptr[j], self.cptr[j + 1]):
                r = int(self.rowind[k])
                self.row_edge[fill[r]] = k
                fill[r] += 1
        self.row_of_rowedge = np.repeat(np.arange(self.m, dtype=np.int64), np.diff(self.rowptr))
        self.deg = np.diff(self.cptr)
        self.cols_by_deg = [np.nonzero(self.deg == d)[0] for d in range(int(self.deg.max()) + 1)]

    def check_update(self, nu):
        """nu in CSC edge order -> (mu in CSC edge order, m1, m2, cnt per row)."""
        nb = f64_bits(nu)
        mags = nb & MASK
        nonpos = nonpositive_bits(nb)
        re = self.row_edge
        starts = self.rowptr[:-1]
        cnt = np.add.reduceat(nonpos[re], starts) if self.nnz else np.zeros(0, np.int64)
        # two smallest magnitudes per row: sort each row's run by (row, magnitude)
        order = np.lexsort((mags[re], self.row_of_rowedge))
        sorted_mag = mags[re][order]
        deg = np.diff(self.rowptr)
        m1 = np.full(self.m, np.uint64(DBL_MAX_BITS), dtype=np.uint64)
        m2 = np.full(self.m, np.uint64(DBL_MAX_BITS), dtype=np.uint64)
        has = deg > 0
        m1[has] = sorted_mag[starts[has]]
        has2 = deg > 1
        m2[has2] = sorted_mag[starts[has2] + 1]
        base = np.where(mags == m1[self.rowind], m2[self.rowind], m1[self.rowind])
        sgn = (cnt[self.rowind] + nonpos + self.syn[self.rowind]) & 1
        mu = bits_f64(base.astype(np.uint64))
        mu = np.where(sgn != 0, -mu, mu)
        return mu, m1, m2, cnt

    def column_update(self, mu, bias):
        """The kernel's part 1 / part 2 per column, sequential per column, from bias."""
        pre = np.empty(self.nnz, dtype=np.float64)
        out = np.empty(self.nnz, dtype=np.float64)   # b2c for the next iteration
        M = np.empty(self.n, dtype=np.float64)
        for d, cols in enumerate(self.cols_by_deg):
            if cols.size == 0:
                continue
            if d == 0:
                M[cols] = bias[cols]
                continue
            idx = self.cptr[cols][:, None] + np.arange(d)[None, :]
            mu2 = mu[idx]
            seq = np.concatenate([bias[cols][:, None], mu2], axis=1)
            cs = np.cumsum(seq, axis=1)          # strictly sequential per row
            M[cols] = cs[:, d]
            pre[idx] = cs[:, :d]
            rev = np.cumsum(mu2[:, ::-1], axis=1)[:, ::-1]   # rev[i] = mu_d-1 + ... + mu_i
            t2 = np.zeros((cols.size, d), dtype=np.float64)
            t2[:, : d - 1] = rev[:, 1:]
            out[idx] = pre[idx] + t2
        return out, M

    def run(self, gamma, omg, T, alpha=1.0):
        """One leg from M(0) = lambda.  Yields (nu(t), M(t)) for t = 1..T (no early stop)."""
        M = self.lam.copy()
        nu = np.repeat(self.lam, self.deg) if self.nnz else np.zeros(0)
        for t in range(1, T + 1):
            bias = omg * self.lam + gamma * M
            mu, _, _, _ = self.check_update(nu)
            if alpha != 1.0:
                mu = mu * alpha
            nu, M = self.column_update(mu, bias)
            yield t, nu, M


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--gamma-seed", type=int, default=20261018)
    a = ap.parse_args(argv)
    import stim

    rng = np.random.default_rng(0)
    ok, msg = cumsum_selfcheck(rng)
    print(f"reference self-check: numpy cumsum sequential ({msg}): {'ok' if ok else 'FAILED'}")
    if not ok:
        return 1

    prob = load_problem_dir(a.problem)
    H = prob.H
    m, n = H.shape
    p = np.asarray(prob.priors, dtype=np.float64)
    circuit = os.path.join(a.problem, "circuit.stim")
    sampler = stim.Circuit.from_file(circuit).compile_detector_sampler(seed=int(a.seed))
    det, obs = sampler.sample(a.shots, separate_observables=True, bit_packed=True)
    det = np.asarray(det, dtype=np.uint8)
    if det.shape[1] != (m + 7) // 8:
        raise SystemExit(f"sampled detectors {det.shape} do not match m={m}")
    det = np.unpackbits(det, axis=1, count=m, bitorder="little").astype(np.uint8)

    cfg = ("decoder=bp;bp_method=ms;max_iter=1;relay=1;relay_legs=1;relay_pre_iter=1;"
           "relay_set_iter=1;relay_seed=1")
    lib = AbiLibrary(a.lib)
    dec = AbiDecoder(lib, cfg, prob)
    dec.decode(np.zeros(m, dtype=np.uint8))
    lam = dec.soft_output()
    print(f"problem {os.path.basename(a.problem)}: m {m} n {n} nnz {int(H.nnz)}; seed {a.seed}; "
          f"iters {a.iters}; shots {a.shots}")

    # a fixed, disordered gamma per column (deterministic; the leg's exactness does not depend on
    # the draw, only on the vector being the same on both sides)
    gamma = np.empty(n, dtype=np.float64)
    for j in range(n):
        u = float(splitmix((a.gamma_seed + 0x9E3779B97F4A7C15 * (j + 1)) & 0xFFFFFFFFFFFFFFFF) >> 11) * 2.0 ** -53
        gamma[j] = -0.24 + (0.66 - -0.24) * u
    omg = 1.0 - gamma

    L = lib.lib
    L.fastbp_test_relay_leg.restype = __import__("ctypes").c_int
    import ctypes
    from dbench.abi import _u8p, _f64p
    L.fastbp_test_relay_leg.argtypes = [ctypes.c_void_p, _u8p, ctypes.c_uint32, _f64p, _f64p,
                                        ctypes.c_uint32, _f64p, _f64p, ctypes.POINTER(ctypes.c_uint32)]
    msgbuf = np.zeros(int(H.nnz), dtype=np.float64)
    postbuf = np.zeros(n, dtype=np.float64)
    itbuf = ctypes.c_uint32(0)

    fails = 0
    checks = 0
    for s in range(a.shots):
        syn = det[s].astype(np.uint8)
        ref = Ref(H, lam, syn)
        gen = ref.run(gamma, omg, a.iters)
        for t, nu_ref, M_ref in gen:
            # the run of t+1 iterations leaves nu(t) in b2c and M(t) in the parity buffer of t
            rc = L.fastbp_test_relay_leg(dec._h, _ptr_u8(syn), t + 1, _ptr_f64(gamma), _ptr_f64(omg),
                                         t & 1, _ptr_f64(msgbuf), _ptr_f64(postbuf),
                                         ctypes.byref(itbuf))
            if rc != 0:
                print(f"shot {s} t {t}: test export rc {rc}")
                fails += 1
                continue
            if int(itbuf.value) != t + 1:
                print(f"shot {s} t {t}: kernel ran {itbuf.value} iterations (converged early?)")
                fails += 1
                break
            got_nu = msgbuf.copy()
            got_M = postbuf.copy()
            bad_nu = np.nonzero(got_nu.view(np.uint64) != nu_ref.view(np.uint64))[0]
            bad_M = np.nonzero(got_M.view(np.uint64) != M_ref.view(np.uint64))[0]
            checks += 1
            if bad_nu.size or bad_M.size:
                fails += 1
                print(f"shot {s} t {t}: MISMATCH messages {bad_nu.size}/{nu_ref.size} "
                      f"marginals {bad_M.size}/{n}")
                for e in bad_nu[:4]:
                    print(f"   edge {e}: kernel {got_nu[e]!r} ({got_nu[e].view(np.uint64)!r}) "
                          f"ref {nu_ref[e]!r} ({nu_ref[e].view(np.uint64)!r})")
                for j in bad_M[:4]:
                    print(f"   col {j}: kernel {got_M[j]!r} ref {M_ref[j]!r}")
            else:
                print(f"shot {s} t {t}: {nu_ref.size} messages and {n} marginals identical")
    dec.close()
    print(f"relay unit exactness: {checks} iteration checks, {fails} failures "
          f"({'PASS' if fails == 0 else 'FAIL'})")
    return 0 if fails == 0 else 1


import ctypes  # noqa: E402


def _ptr_u8(a):
    from dbench.abi import _u8p
    return a.ctypes.data_as(_u8p)


def _ptr_f64(a):
    from dbench.abi import _f64p
    return a.ctypes.data_as(_f64p)


if __name__ == "__main__":
    sys.exit(main())
