#!/usr/bin/env python3
"""The GPU Relay-BP against relay-bp[stim]==0.2.2 on the same fresh shots.

Runs the same shots through
  * relay-bp 0.2.2 (RelayDecoderF64, the CPU reference), qLDPC's defaults (gamma0 0.1, pre_iter 80,
    num_sets 300, set_max_iter 60, stop_nconv = S) and the gross code's disordered interval
    [-0.24, 0.66], and
  * the GPU library (config key relay=S with the same parameters and a config seed),
reports each decoder's logical error rate with its 99 % Clopper-Pearson interval, a two-proportion
z-test between them, each one's convergence count, and the single-shot latency distribution of the
GPU decoder (mean / p50 / p99 / p99.9 / max, measured around dbench_decode).

Fresh shots only: stim.Circuit.from_file(circuit.stim).compile_detector_sampler(seed).sample(n,
separate_observables=True, bit_packed=True) with the seed given on the command line.  Every output
line carries the seed, the problem and the decoder settings.

Run with a Python environment that has relay-bp[stim]==0.2.2 installed.

usage: python -I t_relay_crate.py --lib <so> --problem <dir> --seed <64-bit> --shots N --s 1,5
                                  [--tag NAME] [--relay-seed N] [--no-crate]
Prints one JSON object per S.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))
from dbench.abi import AbiDecoder, AbiLibrary                 # noqa: E402
from dbench.formats import load_problem_dir                   # noqa: E402


def clopper_pearson(k: int, n: int, alpha: float = 0.01):
    """The 99 % exact (Clopper-Pearson) interval for k failures in n trials."""
    from scipy.stats import beta
    if n <= 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)


def two_proportion(k1: int, n1: int, k2: int, n2: int):
    """Two-sided two-proportion z-test (pooled); returns (z, p)."""
    from scipy.stats import norm
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = (p * (1.0 - p) * (1.0 / n1 + 1.0 / n2)) ** 0.5
    if se == 0.0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    return (float(z), float(2.0 * norm.sf(abs(z))))


def dist(v):
    if not v:
        return {}
    a = np.asarray(v, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean_us": float(a.mean()),
        "p50_us": float(np.percentile(a, 50)),
        "p99_us": float(np.percentile(a, 99)),
        "p999_us": float(np.percentile(a, 99.9)),
        "max_us": float(a.max()),
        "min_us": float(a.min()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--shots", type=int, required=True)
    ap.add_argument("--s", default="1,5")
    ap.add_argument("--tag", default="")
    ap.add_argument("--relay-seed", type=int, default=20261018)
    ap.add_argument("--gamma0", type=float, default=0.1)
    ap.add_argument("--pre-iter", type=int, default=80)
    ap.add_argument("--legs", type=int, default=300)
    ap.add_argument("--set-iter", type=int, default=60)
    ap.add_argument("--gamma-lo", type=float, default=-0.24)
    ap.add_argument("--gamma-hi", type=float, default=0.66)
    ap.add_argument("--no-crate", action="store_true")
    ap.add_argument("--no-gpu-timing", action="store_true")
    a = ap.parse_args(argv)

    import stim
    prob = load_problem_dir(a.problem)
    H = prob.H.tocsc()
    O = prob.O.tocsc()
    m, n = H.shape
    k = O.shape[0]
    priors = np.asarray(prob.priors, dtype=np.float64)
    circuit = os.path.join(a.problem, "circuit.stim")
    sampler = stim.Circuit.from_file(circuit).compile_detector_sampler(seed=int(a.seed))
    det, obs = sampler.sample(a.shots, separate_observables=True, bit_packed=True)
    det = np.asarray(det, dtype=np.uint8)
    det = np.unpackbits(det, axis=1, count=m, bitorder="little").astype(np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)
    obs = np.unpackbits(obs, axis=1, count=k, bitorder="little").astype(np.uint8)
    print(f"# problem {os.path.basename(a.problem)} m {m} n {n} k {k} nnz {H.nnz}; shots "
          f"{a.shots}; stim seed {a.seed}; relay seed {a.relay_seed}; tag {a.tag}")

    Os = O.tocsr()
    for s in [int(x) for x in a.s.split(",") if x]:
        cfg = (f"decoder=bp;bp_method=ms;max_iter=1;relay={s};relay_legs={a.legs};"
               f"relay_pre_iter={a.pre_iter};relay_set_iter={a.set_iter};relay_gamma0={a.gamma0};"
               f"relay_gamma_lo={a.gamma_lo};relay_gamma_hi={a.gamma_hi};relay_seed={a.relay_seed}")
        lib = AbiLibrary(a.lib)
        dec = AbiDecoder(lib, cfg, prob)
        t0 = time.perf_counter()
        ours_obs, ours_corr, ours_st = dec.decode_batch(det)
        ours_wall = time.perf_counter() - t0
        ours_fail = int(np.count_nonzero(np.any(ours_obs != obs, axis=1)))
        ours_conv = int(np.count_nonzero((ours_st & 0xFFFF) == 0x0001))
        ours_iters = (ours_st >> 16).astype(np.int64)
        rec = {
            "tag": a.tag, "problem": os.path.basename(a.problem), "s": s, "shots": a.shots,
            "stim_seed": int(a.seed), "relay_seed": int(a.relay_seed),
            "config": cfg,
            "ours": {"fail": ours_fail, "converged": ours_conv,
                     "ler": ours_fail / a.shots,
                     "ci99": clopper_pearson(ours_fail, a.shots),
                     "iters_mean": float(ours_iters.mean()),
                     "iters_p50": float(np.percentile(ours_iters, 50)),
                     "iters_max": int(ours_iters.max()),
                     "wall_s": float(ours_wall),
                     "wall_us_per_shot": float(ours_wall / a.shots * 1e6)},
        }
        if not a.no_crate:
            from scipy.sparse import csr_matrix
            from relay_bp.bp import RelayDecoderF64
            cdec = RelayDecoderF64(check_matrix=csr_matrix(H), error_priors=priors,
                                   gamma0=a.gamma0, pre_iter=a.pre_iter, num_sets=a.legs,
                                   set_max_iter=a.set_iter,
                                   gamma_dist_interval=(a.gamma_lo, a.gamma_hi), stop_nconv=s)
            tc0 = time.perf_counter()
            res = cdec.decode_detailed_batch(det)
            crate_wall = time.perf_counter() - tc0
            crate_corr = np.array([np.asarray(r.decoding, dtype=np.uint8) for r in res])
            crate_obs = np.asarray((Os @ crate_corr.T).T % 2, dtype=np.uint8)
            crate_fail = int(np.count_nonzero(np.any(crate_obs != obs, axis=1)))
            crate_conv = int(np.count_nonzero([bool(r.success) for r in res]))
            crate_iters = np.array([int(r.iterations) for r in res], dtype=np.int64)
            z, p = two_proportion(ours_fail, a.shots, crate_fail, a.shots)
            rec["crate"] = {"fail": crate_fail, "converged": crate_conv,
                            "ler": crate_fail / a.shots,
                            "ci99": clopper_pearson(crate_fail, a.shots),
                            "iters_mean": float(crate_iters.mean()),
                            "iters_p50": float(np.percentile(crate_iters, 50)),
                            "iters_max": int(crate_iters.max()),
                            "wall_s": float(crate_wall),
                            "wall_us_per_shot": float(crate_wall / a.shots * 1e6)}
            rec["two_proportion"] = {"z": z, "p": p}
        if not a.no_gpu_timing:
            lat = []
            # single-shot latency: one handle, back-to-back calls on this shot set
            dec2 = AbiDecoder(AbiLibrary(a.lib), cfg, prob)
            for i in range(a.shots):
                st = time.perf_counter_ns()
                dec2.decode(det[i])
                lat.append((time.perf_counter_ns() - st) / 1000.0)
            rec["latency_us"] = dist(lat)
            dec2.close()
        dec.close()
        print(json.dumps(rec, sort_keys=True))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
