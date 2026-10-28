#!/usr/bin/env python3
"""The Relay-BP model check (a tool; the shipped library does no I/O).

Compares one relay handle's decode with an independent CPU implementation of the paper's
Relay-BP (Mueller et al., arXiv 2506.01779) on the same problem, priors, syndrome and gamma
draw, bit for bit:

  * DMem-BP per leg: bias Lambda_j(t) = (1 - gamma_j) lambda_j + gamma_j M_j(t-1); check update
    = min over the other edges with the final sign; column update in increasing row order from
    Lambda(t) (the kernel's pre / suffix association); marginal M_j(t) = Lambda(t) + sum_i mu.
  * legs chained through their final marginals; a leg stops at the first state that reproduces the
    syndrome; the decoder returns the lowest-weight solution, w(e) = sum e_j lambda_j.

The library's prior LLRs are read back from an all-zero-syndrome soft output, so the model uses
exactly the values the kernel holds (not a re-evaluation of log((1-p)/p)).

usage: python3 -I t_relay_model.py <lib.so> [--shots N] [--seed N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))
from dbench.abi import AbiDecoder, AbiLibrary          # noqa: E402
from dbench.formats import ProblemArrays               # noqa: E402
from scipy.sparse import csc_matrix as sparse_csc      # noqa: E402

MASK = 0x7FFFFFFFFFFFFFFF
INF = 0x7FF0000000000000
DBLMAX = 0x7FEFFFFFFFFFFFFF


def bits(v: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", v))[0]


def mag(v: float) -> int:
    return bits(v) & MASK


def nonpositive(v: float) -> int:
    b = bits(v)
    m = b & MASK
    return 1 if (m <= INF and ((b >> 63) or m == 0)) else 0


def splitmix(z: int) -> int:
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def gamma_vec(leg: int, n: int, g0: float, lo: float, hi: float, seed: int) -> np.ndarray:
    """The kernel's relay_gamma_kernel, replicated: leg 0 uniform, later legs SplitMix64/uniform."""
    out = np.empty(n, dtype=np.float64)
    if leg == 0:
        out.fill(g0)
        return out
    for j in range(n):
        z = (seed ^ ((0x9E3779B97F4A7C15 * (leg + 1)) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
        z = splitmix((z + 0xD1B54A32D192ED03 * (j + 1)) & 0xFFFFFFFFFFFFFFFF)
        u = float(z >> 11) * (2.0 ** -53)
        out[j] = lo + (hi - lo) * u
    return out


class Model:
    """The paper's Relay-BP-S on one problem, in the kernel's arithmetic."""

    def __init__(self, m, n, colptr, rowind, lam, syn, alpha=1.0):
        self.m, self.n = m, n
        self.colptr = colptr
        self.rowind = rowind
        self.lam = lam
        self.syn = syn
        self.alpha = alpha
        # rows -> columns and columns -> rows (the latter is what a leg walks per column; the
        # original per-column scan of every row is O(n*m) per iteration and unusable on the shapes
        # the schedule tests need)
        self.rows = [[] for _ in range(m)]
        self.crows = [[] for _ in range(n)]
        for j in range(n):
            for k in range(colptr[j], colptr[j + 1]):
                self.rows[rowind[k]].append(j)
                self.crows[j].append(rowind[k])

    def check_update(self, nu, syn):
        """mu[(j,i)] from the incoming nu (one entry per (col,row) edge)."""
        mu = {}
        for i in range(self.m):
            js = self.rows[i]
            vals = [nu[(j, i)] for j in js]
            m1 = DBLMAX
            m2 = DBLMAX
            cnt = 0
            for v in vals:
                a = mag(v)
                cnt += nonpositive(v)
                if a < m1:
                    m2 = m1
                    m1 = a
                elif a < m2:
                    m2 = a
            for jj, j in enumerate(js):
                base = m2 if mag(vals[jj]) == m1 else m1
                sgn = (cnt + nonpositive(vals[jj]) + syn[i]) & 1
                mu[(j, i)] = -float_from_bits(base) if sgn else float_from_bits(base)
        return mu

    def leg(self, gamma, w, max_iter):
        """One DMem-BP leg.  w is the leg's input marginals M(0).  Returns (final M, converged,
        iteration, correction)."""
        nu = {}
        for j in range(self.n):
            for k in range(self.colptr[j], self.colptr[j + 1]):
                nu[(j, self.rowind[k])] = self.lam[j]
        M = list(w)
        for t in range(1, max_iter + 1):
            im1 = [1.0 - g for g in gamma]
            lam_t = [im1[j] * self.lam[j] + gamma[j] * M[j] for j in range(self.n)]
            mu = self.check_update(nu, self.syn)
            newM = [0.0] * self.n
            newnu = {}
            for j in range(self.n):
                js = self.crows[j]
                t0 = lam_t[j]
                pre = []
                cb = []
                for i in js:
                    m = mu[(j, i)]
                    if self.alpha != 1.0:
                        # dmul(base, +-alpha) is the kernel's single multiply; here model it as
                        # the same multiply of the (signed) message by alpha
                        m = m * self.alpha
                    cb.append(m)
                    pre.append(t0)
                    t0 = t0 + m
                newM[j] = t0
                t2 = 0.0
                for idx in range(len(js) - 1, -1, -1):
                    newnu[(j, js[idx])] = pre[idx] + t2
                    t2 = t2 + cb[idx]
            M = newM
            nu = newnu
            dec = [1 if nonpositive(v) else 0 for v in M]
            if self.syndrome_of(dec) == self.syn:
                return M, True, t, dec
        dec = [1 if nonpositive(v) else 0 for v in M]
        return M, False, max_iter, dec

    def _has_edge(self, j, i):
        for k in range(self.colptr[j], self.colptr[j + 1]):
            if self.rowind[k] == i:
                return True
        return False

    def syndrome_of(self, dec):
        out = [0] * self.m
        for j, b in enumerate(dec):
            if b:
                for k in range(self.colptr[j], self.colptr[j + 1]):
                    out[self.rowind[k]] ^= 1
        return out

    def weight(self, dec):
        return sum(self.lam[j] for j in range(self.n) if dec[j])


def float_from_bits(b: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", b))[0]


def problem_arrays(H, priors):
    Hc = H.tocsc()
    O = np.zeros((1, H.shape[1]), dtype=np.uint8)
    O[0, 0] = 1
    return ProblemArrays(H=Hc, O=sparse_csc(O), priors=priors, dem_text=None, meta={})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("lib")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--errp", type=float, default=0.12)
    ap.add_argument("--shape", choices=("ilv", "highdeg", "mixed", "uniform4", "wide", "fusilv"),
                    default="ilv",
                    help="ilv: short rows/columns (the interleaved in-place kernel); highdeg: column "
                         "degree 12 (the non-interleaved in-place kernel); mixed: degree-12 columns "
                         "and a very long row (the two-barrier summary kernel for a relay handle); "
                         "fusilv: column degree 4 over 2200 rows, whose in-place grid (275 blocks) "
                         "exceeds the SM count (the fused single-barrier interleaved kernel)")
    ap.add_argument("--probe-lib", default="",
                    help="a layout-probe build of the library; a second handle on the same synthetic "
                         "problem prints the schedule (LayoutStats) the shape routes to")
    ap.add_argument("--legs", type=int, default=6)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--problem", default="",
                    help="a corpus problem directory: decode fresh stim shots of it (a real code's "
                         "problem) instead of a synthetic shape; the schedule printed is the one the "
                         "handle takes for that problem")
    a = ap.parse_args(argv)
    from scipy.sparse import random as sprandom, csr_matrix
    rng = np.random.default_rng(a.seed)
    if a.problem:
        pass  # the synthetic shape is not built: the problem's own H is used below
    elif a.shape == "ilv":
        m, n = 16, 40
        Hd = np.zeros((m, n), dtype=np.uint8)
        for j in range(n):
            deg = int(rng.integers(2, 5))
            rows = rng.choice(m, size=deg, replace=False)
            Hd[rows, j] = 1
    elif a.shape == "highdeg":
        m, n = 40, 24
        Hd = np.zeros((m, n), dtype=np.uint8)
        for j in range(n):
            rows = rng.choice(m, size=12, replace=False)
            Hd[rows, j] = 1
    elif a.shape == "uniform4":
        # uniform column degree 4: the interleaved layout's waste gate accepts it and the grid is
        # small, so the in-place interleaved kernel (col_pass_ilv) runs.
        m, n = 40, 200
        Hd = np.zeros((m, n), dtype=np.uint8)
        for j in range(n):
            Hd[rng.choice(m, size=4, replace=False), j] = 1
    elif a.shape == "fusilv":
        # uniform column degree 4 over 1600 rows: need_rows = 200 in-place blocks, more than the SM
        # count, so the wide-grid rule asks for the summary formulation; every column has degree 4
        # (row degree >= 2 everywhere, so the extra-edge repair adds nothing), so the interleaved
        # waste ratio is 100.4 % and the interleaved layout is accepted: the fused single-barrier
        # kernel (sum_mode 1) with the b2c window runs -- the kernel the bb288 z problem runs.
        m, n = 1600, 4400
        Hd = np.zeros((m, n), dtype=np.uint8)
        for j in range(n):
            Hd[rng.choice(m, size=4, replace=False), j] = 1
    elif a.shape == "wide":
        # A row of degree 100 (wpr = 2) over 2200 rows needs 550 in-place blocks against a cap of
        # 252 (3 blocks/SM x 84), so the two-barrier summary is taken; the columns have degree 11,
        # so the interleaved layout is refused as well, and for a relay handle the fused schedule is
        # refused too: the two-barrier, non-interleaved kernel (col_sum_pass) runs.
        m, n = 2200, 2000
        Hd = np.zeros((m, n), dtype=np.uint8)
        for j in range(n):
            rows = rng.choice(np.arange(1, m), size=10, replace=False)
            Hd[rows, j] = 1
            if j < 100:
                Hd[0, j] = 1
            else:
                Hd[rng.integers(1, m), j] = 1
    else:  # mixed: every column has degree 12 (no interleaved layout) and row 0 touches all, so
        # the row degree 120 forces the split-row (wpr > 1) formulation: for a relay handle the
        # fused schedule is refused (non-interleaved) and the two-barrier summary kernel runs.
        m, n = 24, 120
        Hd = np.zeros((m, n), dtype=np.uint8)
        for j in range(n):
            rows = rng.choice(np.arange(1, m), size=11, replace=False)
            Hd[rows, j] = 1
            Hd[0, j] = 1
    if a.problem:
        from dbench.formats import load_problem_dir
        prob = load_problem_dir(a.problem)
        H = prob.H
        m, n = H.shape
        Hd = None
        print(f"problem {os.path.basename(a.problem)}: m {m} n {n} nnz {int(H.nnz)}; "
              f"shots {a.shots}; stim seed {a.seed}")
    else:
        # make sure every row has at least two edges
        for i in range(m):
            if Hd[i].sum() < 2:
                Hd[i, int(rng.integers(0, n))] = 1
        H = csr_matrix(Hd)
        priors = np.full(n, 0.08)
        prob = problem_arrays(H, priors)
    cfg = (f"decoder=bp;bp_method=ms;max_iter=1;relay=5;relay_legs={a.legs};"
           f"relay_pre_iter={a.iters};relay_set_iter={a.iters};relay_gamma0=0.5;"
           "relay_gamma_lo=-0.24;relay_gamma_hi=0.66;relay_seed=7")
    lib = AbiLibrary(a.lib)
    dec = AbiDecoder(lib, cfg, prob)
    # The all-zero syndrome short-circuits to 0 iterations and its soft output is the prior LLR
    # vector: read the exact values the kernel holds (not a re-evaluation of log((1-p)/p)).
    dec.decode(np.zeros(m, dtype=np.uint8))
    lam = dec.soft_output()
    if Hd is not None:
        print(f"shape {a.shape}: m {m} n {n} max col degree {int(Hd.sum(axis=0).max())} "
              f"max row degree {int(Hd.sum(axis=1).max())}")
    if a.probe_lib:
        import ctypes
        FIELDS = ["words", "sms", "nblocks", "nthreads", "sum_mode", "ilv", "balanced", "rslot",
                  "rec", "ns_max", "ns_uni", "ns_bal", "gate", "ns_uni_6e", "ns_bal_6e", "gate_6e",
                  "bal_cost_ratio_ppm", "rslot_bytes", "meta_max", "fused_shbytes", "max_optin",
                  "win_rounds", "win_bytes", "max_block_cols", "win_request", "fused_static",
                  "fused_avail_dyn", "fused_max_dyn_attr"]
        pdec = AbiDecoder(AbiLibrary(a.probe_lib), cfg, prob)
        fn = pdec.lib.lib.fastbp_test_layout
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
        fn.restype = ctypes.c_uint32
        buf = (ctypes.c_uint32 * 64)()
        nw = int(fn(pdec._h, buf, 64))
        lay = {f: int(buf[i]) for i, f in enumerate(FIELDS) if i < nw}
        print(f"schedule ({os.path.basename(a.probe_lib)}): " + json.dumps(lay, sort_keys=True))
        pdec.close()
    model = Model(m, n, H.tocsc().indptr.astype(np.int64), H.tocsc().indices.astype(np.int64),
                  list(lam), [0] * m)
    fails = 0
    shots_det = None
    if a.problem:
        import stim
        sampler = stim.Circuit.from_file(os.path.join(a.problem, "circuit.stim")) \
            .compile_detector_sampler(seed=int(a.seed))
        sd, _ = sampler.sample(a.shots, separate_observables=True, bit_packed=True)
        shots_det = np.unpackbits(np.asarray(sd, dtype=np.uint8), axis=1, count=m,
                                  bitorder="little").astype(np.uint8)
    for s in range(a.shots):
        if shots_det is not None:
            syn = shots_det[s]
        else:
            e = (rng.random(n) < a.errp).astype(np.uint8)
            syn = (Hd @ e) % 2
        obs, corr, status = dec.decode(syn)
        it = int(status >> 16)
        flags = int(status & 0xFFFF)
        if not syn.any():
            # the ABI's all-zero short circuit: zeros, BP_CONVERGED, 0 iterations, soft = priors;
            # the model below always runs the leg loop, so compare the short circuit on its own.
            ok = (not corr.any() and not obs.any() and flags == 0x0001 and it == 0
                  and np.array_equal(lam, dec.soft_output()))
            print(f"shot {s}: all-zero syndrome short circuit -> {'ok' if ok else 'MISMATCH'}")
            if not ok:
                fails += 1
            continue
        model.syn = [int(x) for x in syn]
        # the model's full relay: same leg parameter set
        M = list(lam)
        best = None
        best_w = None
        found = 0
        total = 0
        for leg in range(a.legs):
            g = gamma_vec(leg, n, 0.5, -0.24, 0.66, 7)
            T = a.iters
            Ml, conv, tt, d = model.leg(list(g), M, T)
            total += tt
            M = Ml
            if conv:
                w = model.weight(d)
                found += 1
                if best is None or w < best_w:
                    best, best_w = d, w
                if found >= 5:
                    break
        want_corr = np.array(best if best is not None else d, dtype=np.uint8)
        mflags = 0x0001 if found else 0x0010
        ok = (np.array_equal(corr, want_corr) and flags == mflags and it == min(total, 0xFFFF)
              and np.array_equal(M, dec.soft_output()))
        print(f"shot {s}: it {it}/{total} flags {flags:#x}/{mflags:#x} found {found} "
              f"corr_equal {np.array_equal(corr, want_corr)} post_equal "
              f"{np.array_equal(M, dec.soft_output())} -> {'ok' if ok else 'MISMATCH'}")
        if not ok:
            fails += 1
            if not np.array_equal(corr, want_corr):
                diff = np.nonzero(corr != want_corr)[0][:10]
                print("   corr differs at", diff, "got", corr[diff], "want", want_corr[diff])
            if not np.array_equal(M, dec.soft_output()):
                d0 = np.nonzero(np.array(M) != dec.soft_output())[0][:10]
                print("   post differs at", d0)
                print("   got ", dec.soft_output()[d0])
                print("   want", np.array(M)[d0])
    dec.close()
    print(f"relay model check: {a.shots - fails}/{a.shots} shots identical")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
