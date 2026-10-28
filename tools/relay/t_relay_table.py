#!/usr/bin/env python3
"""The Relay-BP LER and latency table on one fresh shot set.

Runs several decoders of the library on the SAME fresh shots and prints one JSON row per decoder:
logical failures / shots, the 99 % Clopper-Pearson interval, the convergence count and the
single-shot latency distribution (mean / p50 / p99 / p99.9 / max / min, host clock around
dbench_decode, one handle, back to back).

Decoders: relay1, relay5 (relay=S with qLDPC's defaults: gamma0 0.1, pre_iter 80, num_sets 300,
set_max_iter 60, the gross code's [-0.24, 0.66] disordered interval), bposd-a03125c3 (alpha
0.3125, max_iter 3, OSD-0) and bposd-a05ms31 (alpha 0.5, max_iter 31, OSD-0).

Fresh shots: stim.Circuit.from_file(circuit.stim).compile_detector_sampler(seed).sample(n,
separate_observables=True, bit_packed=True); the seed is required and printed in every row.

usage: python -I t_relay_table.py --lib <so> --problem <dir> --seed <64-bit> --shots N
                                  [--decoders relay1,relay5,bposd-a03125c3,bposd-a05ms31]
                                  [--relay-seed N] [--tag NAME] [--no-timing]
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

RELAY = ("decoder=bp;bp_method=ms;max_iter=1;relay={s};relay_legs={legs};relay_pre_iter={pre};"
         "relay_set_iter={setit};relay_gamma0={gamma0};relay_gamma_lo={lo};relay_gamma_hi={hi};"
         "relay_seed={seed}")
BPOSD = ("decoder=bposd;bp_method=ms;max_iter={mi};ms_scaling_factor={alpha};schedule=parallel;"
         "osd_method=osd0;osd_order=0")


def clopper_pearson(k: int, n: int, alpha: float = 0.01):
    from scipy.stats import beta
    if n <= 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return [lo, hi]


def dist(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {}
    return {"n": int(a.size), "mean_us": float(a.mean()), "p50_us": float(np.percentile(a, 50)),
            "p99_us": float(np.percentile(a, 99)), "p999_us": float(np.percentile(a, 99.9)),
            "max_us": float(a.max()), "min_us": float(a.min())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--shots", type=int, required=True)
    ap.add_argument("--decoders", default="relay1,relay5,bposd-a03125c3,bposd-a05ms31")
    ap.add_argument("--relay-seed", type=int, default=20261018)
    ap.add_argument("--gamma0", type=float, default=0.1)
    ap.add_argument("--pre-iter", type=int, default=80)
    ap.add_argument("--legs", type=int, default=300)
    ap.add_argument("--set-iter", type=int, default=60)
    ap.add_argument("--gamma-lo", type=float, default=-0.24)
    ap.add_argument("--gamma-hi", type=float, default=0.66)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--timing-shots", type=int, default=0)
    a = ap.parse_args(argv)

    import stim
    prob = load_problem_dir(a.problem)
    H = prob.H.tocsc()
    m, n = H.shape
    k = prob.O.shape[0]
    sampler = stim.Circuit.from_file(os.path.join(a.problem, "circuit.stim")).compile_detector_sampler(
        seed=int(a.seed))
    det, obs = sampler.sample(a.shots, separate_observables=True, bit_packed=True)
    det = np.unpackbits(np.asarray(det, dtype=np.uint8), axis=1, count=m, bitorder="little").astype(np.uint8)
    obs = np.unpackbits(np.asarray(obs, dtype=np.uint8), axis=1, count=k, bitorder="little").astype(np.uint8)
    print(f"# problem {os.path.basename(a.problem)} m {m} n {n} k {k} nnz {H.nnz}; shots {a.shots}; "
          f"stim seed {a.seed}; relay seed {a.relay_seed}; tag {a.tag}")
    rows = []
    for name in [x for x in a.decoders.split(",") if x]:
        if name.startswith("relay"):
            s = int(name[5:])
            cfg = RELAY.format(s=s, legs=a.legs, pre=a.pre_iter, setit=a.set_iter, gamma0=a.gamma0,
                               lo=a.gamma_lo, hi=a.gamma_hi, seed=a.relay_seed)
        elif name == "bposd-a03125c3":
            cfg = BPOSD.format(mi=3, alpha=0.3125)
        elif name == "bposd-a05ms31":
            cfg = BPOSD.format(mi=31, alpha=0.5)
        else:
            raise SystemExit(f"unknown decoder {name}")
        dec = AbiDecoder(AbiLibrary(a.lib), cfg, prob)
        t0 = time.perf_counter()
        outs, corr, st = dec.decode_batch(det)
        wall = time.perf_counter() - t0
        fail = int(np.count_nonzero(np.any(outs != obs, axis=1)))
        conv = int(np.count_nonzero((st & 0xFFFF) == 0x0001))
        iters = (st >> 16).astype(np.int64)
        lat = None
        if not a.no_timing:
            nth = a.timing_shots if a.timing_shots > 0 else a.shots
            nth = min(nth, a.shots)
            dec2 = AbiDecoder(AbiLibrary(a.lib), cfg, prob)
            for i in range(min(100, a.shots)):   # warm-up, not recorded
                dec2.decode(det[i])
            v = []
            for i in range(nth):
                q = time.perf_counter_ns()
                dec2.decode(det[i])
                v.append((time.perf_counter_ns() - q) / 1000.0)
            lat = dist(v)
            dec2.close()
        row = {"tag": a.tag, "problem": os.path.basename(a.problem), "decoder": name,
               "shots": a.shots, "stim_seed": int(a.seed), "relay_seed": int(a.relay_seed),
               "config": cfg, "fail": fail, "ler": fail / a.shots,
               "ci99": clopper_pearson(fail, a.shots), "converged": conv,
               "iters_mean": float(iters.mean()), "iters_p50": float(np.percentile(iters, 50)),
               "batch_wall_s": float(wall), "latency_us": lat}
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
        sys.stdout.flush()
        dec.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
